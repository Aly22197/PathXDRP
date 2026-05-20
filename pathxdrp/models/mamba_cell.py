"""
GeneMamba cell-line encoder — foundation model alternative to PathwaySetEncoder.

Wraps the pretrained GeneMamba backbone (24 Bi-Mamba layers, 512 dim, 65.7M params,
pretrained on 30M cells, weights on HuggingFace mineself2016/GeneMamba) and adapts
its output to PathXDRP's pathway-token interface for the cross-attention.

Reference
---------
GeneMamba: Efficient and Effective Foundation Model on Single Cell Data (2025)
https://arxiv.org/abs/2504.16956
https://huggingface.co/mineself2016/GeneMamba

Pipeline
--------
  expr (B, n_genes)  Z-scored log2(TPM+1)
    |                                                 # rank top-K genes per cell
    v
  rank-encode (B, top_k)  gene token IDs
    |                                                 # frozen GeneMamba backbone
    v
  GeneMamba (B, top_k, 512)
    |                                                 # group-by-pathway, mean-pool
    v
  Pathway adapter (B, n_pathways, hidden_dim)
    |                                                 # plug into cross-attention
    v
  PathwayMaskedCrossAttention(K, V)

Import guards
-------------
`mamba_ssm` requires NVIDIA CUDA 11.6+. If unavailable, importing this module
will succeed but `GeneMambaCellEncoder.__init__` raises ImportError. Smoke tests
detect this and skip Mamba-specific tests.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

# Soft import — module loads even if mamba_ssm isn't installed (CUDA-only).
# Real check happens in __init__ so the module can be imported for code introspection
# (e.g. by smoke tests that gate on MAMBA_AVAILABLE).
try:
    from mamba_ssm import Mamba2 as _Mamba                  # noqa: F401
    MAMBA_AVAILABLE = True
except Exception:
    try:
        from mamba_ssm import Mamba as _Mamba               # noqa: F401
        MAMBA_AVAILABLE = True
    except Exception:
        _Mamba = None
        MAMBA_AVAILABLE = False

try:
    from transformers import AutoModel, AutoTokenizer
    HF_AVAILABLE = True
except Exception:
    AutoModel = AutoTokenizer = None
    HF_AVAILABLE = False


# ---
# Lightweight stand-alone Bi-Mamba block (used when pretrained weights absent)
# ---

class BiMambaBlock(nn.Module):
    """
    Bidirectional Mamba block: forward + reverse Mamba over the sequence,
    gated combination, residual + LayerNorm.
    Used as a small from-scratch alternative if the HF GeneMamba checkpoint
    is unavailable, so the architecture is testable without external downloads.
    """

    def __init__(self, hidden_dim: int, d_state: int = 16, d_conv: int = 4) -> None:
        super().__init__()
        if not MAMBA_AVAILABLE:
            raise ImportError(
                "mamba_ssm not installed. Install with:\n"
                "  pip install mamba-ssm[causal-conv1d] --no-build-isolation\n"
                "Requires NVIDIA GPU + CUDA 11.6+."
            )
        self.fwd  = _Mamba(d_model=hidden_dim, d_state=d_state, d_conv=d_conv)
        self.bwd  = _Mamba(d_model=hidden_dim, d_state=d_state, d_conv=d_conv)
        self.gate = nn.Linear(hidden_dim * 2, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, D) -> (B, L, D)"""
        h_f = self.fwd(x)
        h_b = self.bwd(torch.flip(x, dims=[1]))
        h_b = torch.flip(h_b, dims=[1])
        h   = self.gate(torch.cat([h_f, h_b], dim=-1))
        return self.norm(x + h)


# ---
# GeneMamba foundation-model wrapper
# ---

class GeneMambaCellEncoder(nn.Module):
    """
    Pretrained GeneMamba backbone + small trainable pathway adapter.

    Drop-in replacement for PathwaySetEncoder:
      Input:  expr (B, n_genes) Z-scored log2(TPM+1)
      Output: pathway tokens (B, n_pathways, hidden_dim) for cross-attention

    Args:
        n_genes:           Number of input genes (e.g. 19,193).
        gene_symbols:      List of gene symbols matching the columns of expr,
                           in the same order. Used to build the HGNC -> GeneMamba
                           token-ID mapping.
        pathway_gene_map:  {pathway_name: [gene_indices_into_expr_vector]}.
                           Same format as PathwaySetEncoder.
        hidden_dim:        Output dim for the pathway tokens (default 128).
        top_k:             Number of top-expressed genes fed to GeneMamba (2048 or 4096).
        backbone_id:       HuggingFace repo id for pretrained weights.
        freeze_backbone:   If True (default) only the adapter is trained.
        n_layers_fallback: Number of Bi-Mamba layers if the HF checkpoint is unavailable.
    """

    HF_DEFAULT = "mineself2016/GeneMamba"

    def __init__(
        self,
        n_genes: int,
        gene_symbols: list[str],
        pathway_gene_map: dict[str, list[int]],
        hidden_dim: int = 128,
        top_k: int = 2048,
        backbone_id: str = HF_DEFAULT,
        freeze_backbone: bool = True,
        n_layers_fallback: int = 4,
        backbone_dim_fallback: int = 256,
    ) -> None:
        super().__init__()
        if not MAMBA_AVAILABLE:
            raise ImportError(
                "mamba_ssm not installed. Install with:\n"
                "  pip install mamba-ssm[causal-conv1d] --no-build-isolation\n"
                "Requires NVIDIA GPU + CUDA 11.6+."
            )

        self.n_genes          = n_genes
        self.top_k            = min(top_k, n_genes)
        self.hidden_dim       = hidden_dim
        self.gene_symbols     = list(gene_symbols)
        self.pathway_gene_map = pathway_gene_map
        self.pathway_names    = sorted(pathway_gene_map.keys())
        self.n_pathways       = len(self.pathway_names)

        # ---- Backbone ----
        backbone, backbone_dim = self._load_backbone(
            backbone_id, n_layers_fallback, backbone_dim_fallback
        )
        self.backbone     = backbone
        self.backbone_dim = backbone_dim

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

        # ---- Gene-symbol -> backbone token-ID mapping (set by setup_token_map) ----
        # Stored as a buffer so it moves with the model and is preserved in checkpoints.
        # Filled lazily after the tokenizer/vocab is loaded.
        self.register_buffer(
            "gene_token_ids",
            torch.full((n_genes,), -1, dtype=torch.long),
            persistent=True,
        )
        self._token_map_initialised = False

        # ---- Adapter: backbone_dim -> hidden_dim per gene token ----
        self.gene_adapter = nn.Sequential(
            nn.Linear(backbone_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.pathway_norm = nn.LayerNorm(hidden_dim)
        self.dropout      = nn.Dropout(0.1)

    # ------------------------------------------------------------------
    # Backbone loading
    # ------------------------------------------------------------------

    def _load_backbone(
        self,
        backbone_id: str,
        n_layers_fallback: int,
        backbone_dim_fallback: int,
    ) -> tuple[nn.Module, int]:
        """
        Try to load pretrained GeneMamba via HuggingFace.
        If unavailable (no internet, missing weights), instantiate a small
        from-scratch Bi-Mamba stack so the architecture is still trainable.
        """
        if HF_AVAILABLE:
            try:
                model = AutoModel.from_pretrained(backbone_id, trust_remote_code=True)
                dim   = getattr(model.config, "hidden_size", None) or \
                        getattr(model.config, "d_model", backbone_dim_fallback)
                return model, int(dim)
            except Exception as e:
                print(f"[GeneMambaCellEncoder] HF load failed ({e}); using from-scratch fallback.")

        # Fallback: small Bi-Mamba stack (untrained, but architecture-equivalent)
        backbone = nn.ModuleList([
            BiMambaBlock(backbone_dim_fallback) for _ in range(n_layers_fallback)
        ])
        return backbone, backbone_dim_fallback

    # ------------------------------------------------------------------
    # Tokenization map (gene symbol -> backbone vocab id)
    # ------------------------------------------------------------------

    def setup_token_map(self, tokenizer_id: Optional[str] = None) -> None:
        """
        Build the gene_symbols -> backbone-vocab-id mapping. Call once after
        construction, before training. Genes not in the backbone vocab map to -1
        and are masked out at forward time.
        """
        tokenizer_id = tokenizer_id or self.HF_DEFAULT
        if not HF_AVAILABLE:
            # Fallback: deterministic hash-based pseudo-tokens for testability
            ids = torch.tensor(
                [abs(hash(g)) % 25_000 for g in self.gene_symbols],
                dtype=torch.long,
            )
            self.gene_token_ids.copy_(ids)
            self._token_map_initialised = True
            return

        try:
            from tqdm.auto import tqdm
            tok = AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=True)
            vocab = tok.get_vocab() if hasattr(tok, "get_vocab") else {}
            ids = []
            for g in tqdm(self.gene_symbols, desc="  Mapping gene symbols -> tokens",
                          unit="gene", leave=False):
                tid = vocab.get(g, vocab.get(g.upper(), -1))
                ids.append(tid)
            self.gene_token_ids.copy_(torch.tensor(ids, dtype=torch.long))
            n_known = int((self.gene_token_ids >= 0).sum().item())
            print(f"[GeneMambaCellEncoder] Mapped {n_known}/{self.n_genes} genes "
                  f"({100 * n_known / self.n_genes:.1f}%) to backbone vocab.", flush=True)
        except Exception as e:
            print(f"[GeneMambaCellEncoder] Tokenizer load failed ({e}); using hash fallback.")
            ids = torch.tensor(
                [abs(hash(g)) % 25_000 for g in self.gene_symbols],
                dtype=torch.long,
            )
            self.gene_token_ids.copy_(ids)

        self._token_map_initialised = True

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, expr: torch.Tensor) -> torch.Tensor:
        """
        Args:
            expr: (B, n_genes) Z-scored log2(TPM+1).

        Returns:
            (B, n_pathways, hidden_dim) — drop-in replacement for PathwaySetEncoder.
        """
        if not self._token_map_initialised:
            # Lazy init at first forward — uses hash fallback if tokenizer unreachable.
            self.setup_token_map()

        B = expr.size(0)
        device = expr.device

        # ---- 1. Rank-based top-K gene selection per cell (GeneMamba convention) ----
        topk_vals, topk_idx = expr.topk(self.top_k, dim=1)        # (B, K)

        # ---- 2. Encode genes through backbone ----
        gene_embeds = self._encode_topk(topk_idx, topk_vals)      # (B, K, backbone_dim)
        gene_embeds = self.gene_adapter(gene_embeds)              # (B, K, hidden_dim)

        # ---- 3. Build (B, K) pathway-membership mask + per-pathway pooling ----
        pathway_tokens = self._pool_into_pathways(
            gene_embeds, topk_idx, device
        )                                                         # (B, P, hidden_dim)
        return self.dropout(pathway_tokens)

    def _encode_topk(
        self,
        topk_idx:  torch.Tensor,   # (B, K) gene indices into expr columns
        topk_vals: torch.Tensor,   # (B, K) expression values (rank order)
    ) -> torch.Tensor:
        """Run the backbone over the top-K gene tokens for each cell."""
        # Map gene-column indices -> backbone token IDs (-1 for unknown)
        token_ids = self.gene_token_ids[topk_idx]                 # (B, K)
        token_ids = token_ids.clamp(min=0)                        # treat unknown as token 0

        if isinstance(self.backbone, nn.ModuleList):
            # Fallback path: from-scratch Bi-Mamba stack
            # Embed tokens via a tiny embedding lookup then run the stack.
            if not hasattr(self, "_fallback_embed"):
                vocab = max(int(self.gene_token_ids.max().item()) + 1, 25_000)
                self._fallback_embed = nn.Embedding(vocab, self.backbone_dim).to(token_ids.device)
            x = self._fallback_embed(token_ids)                   # (B, K, backbone_dim)
            for blk in self.backbone:
                x = blk(x)
            return x

        # HuggingFace pretrained path
        with torch.no_grad():
            out = self.backbone(input_ids=token_ids)
        # Try common output shapes: last_hidden_state or [0]
        h = getattr(out, "last_hidden_state", None)
        if h is None and isinstance(out, (tuple, list)):
            h = out[0]
        return h                                                  # (B, K, backbone_dim)

    def _pool_into_pathways(
        self,
        gene_embeds: torch.Tensor,    # (B, K, hidden_dim)
        topk_idx:    torch.Tensor,    # (B, K) — column-index in expr each token came from
        device:      torch.device,
    ) -> torch.Tensor:
        """
        For each pathway, mean-pool the embeddings of any of its genes that
        appear in the top-K for each cell. Pathways with no genes in the
        top-K get a zero token (masked at attention time via padding mask).
        """
        B, K, D = gene_embeds.shape
        out = gene_embeds.new_zeros(B, self.n_pathways, D)

        # Build per-pathway gene-set masks once and cache.
        if not hasattr(self, "_pathway_gene_mask"):
            mask = torch.zeros(self.n_pathways, self.n_genes, dtype=torch.bool)
            for p_i, pname in enumerate(self.pathway_names):
                for g_idx in self.pathway_gene_map[pname]:
                    mask[p_i, g_idx] = True
            self.register_buffer("_pathway_gene_mask", mask, persistent=False)
        pw_gene_mask = self._pathway_gene_mask.to(device)        # (P, n_genes) bool

        # For each pathway, mark which top-K positions belong to it.
        # topk_idx: (B, K)  -> gather pw_gene_mask[:, topk_idx] -> (P, B, K)
        belong = pw_gene_mask[:, topk_idx]                       # (P, B, K) bool
        belong = belong.permute(1, 0, 2).float()                 # (B, P, K)
        counts = belong.sum(dim=-1, keepdim=True).clamp(min=1.0) # (B, P, 1)

        # Weighted mean: (B, P, K) @ (B, K, D) — use einsum
        out = torch.einsum("bpk,bkd->bpd", belong, gene_embeds) / counts
        return self.pathway_norm(out)


# ---
# Public API
# ---

__all__ = ["GeneMambaCellEncoder", "BiMambaBlock", "MAMBA_AVAILABLE"]
