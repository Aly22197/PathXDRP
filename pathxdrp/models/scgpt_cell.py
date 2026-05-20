"""
scGPT cell-line encoder — foundation-model alternative to PathwaySetEncoder.

Wraps the pretrained scGPT model (Cui et al., Nature Methods 2024;
https://github.com/bowang-lab/scGPT) as the cell-line backbone, then projects
its gene embeddings into pathway tokens for the cross-attention.

Pipeline
--------
  expr (B, n_genes)  Z-scored log2(TPM+1)
    |                                        # gene token IDs from scGPT vocab
    v
  scGPT (frozen)  -> (B, n_genes, embsize)
    |                                        # group-by-pathway, mean-pool
    v
  pathway tokens (B, n_pathways, hidden_dim) -> cross-attention (K, V)

Import guards
-------------
scGPT requires `flash_attn` and CUDA. If unavailable, this module loads but
`ScGPTCellEncoder.__init__` raises ImportError. Smoke tests detect this and
skip. A from-scratch Transformer fallback is provided for testability.

Note: this is a scaffold — wire `scgpt` weights via the official scGPT
loader once installed. The fallback Transformer ensures the architecture
runs on CPU for tests.

Reference
---------
Cui, Wang et al. "scGPT: toward building a foundation model for single-cell
multi-omics using generative AI." Nature Methods, 2024.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

# Soft import — scGPT bundle (CUDA-only; flash-attn dependency)
try:
    import scgpt as _scgpt          # noqa: F401
    SCGPT_AVAILABLE = True
except Exception:
    _scgpt = None
    SCGPT_AVAILABLE = False


class _FallbackTransformerBlock(nn.Module):
    """Tiny Transformer block used when scgpt is not installed (testability)."""

    def __init__(self, dim: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm1(x + h)
        x = self.norm2(x + self.ffn(x))
        return x


class ScGPTCellEncoder(nn.Module):
    """
    Pretrained scGPT backbone + small trainable pathway adapter.

    Drop-in replacement for PathwaySetEncoder with the same I/O contract:
      Input:  expr (B, n_genes) Z-scored log2(TPM+1)
      Output: pathway tokens (B, n_pathways, hidden_dim)

    Args:
        n_genes:           Number of input genes (e.g. 19,193).
        gene_symbols:      Gene symbols matching expr columns, in order.
        pathway_gene_map:  {pathway_name: [gene_indices_into_expr_vector]}.
        hidden_dim:        Output dim for the pathway tokens.
        ckpt_dir:          Local path to a downloaded scGPT checkpoint dir
                           (whole_human / pancancer / etc.). If absent, the
                           from-scratch fallback is used.
        embsize:           Backbone embedding size (scGPT's `embsize`, default 512).
        n_layers_fallback: Layers of the from-scratch Transformer (testability).
        freeze_backbone:   If True (default) only the adapter is trained.
        top_k:             Top-K most-expressed genes per cell to forward
                           through scGPT (linear cost in K). 2048 is the
                           scGPT recipe default.
    """

    def __init__(
        self,
        n_genes: int,
        gene_symbols: list[str],
        pathway_gene_map: dict[str, list[int]],
        hidden_dim: int = 128,
        ckpt_dir: Optional[str] = None,
        embsize: int = 512,
        n_layers_fallback: int = 4,
        n_heads_fallback: int = 8,
        freeze_backbone: bool = True,
        top_k: int = 2048,
    ) -> None:
        super().__init__()
        self.n_genes          = n_genes
        self.top_k            = min(top_k, n_genes)
        self.hidden_dim       = hidden_dim
        self.gene_symbols     = list(gene_symbols)
        self.pathway_gene_map = pathway_gene_map
        self.pathway_names    = sorted(pathway_gene_map.keys())
        self.n_pathways       = len(self.pathway_names)
        self.embsize          = embsize

        # Backbone
        backbone, self.embsize = self._load_backbone(
            ckpt_dir, n_layers_fallback, n_heads_fallback,
        )
        self.backbone = backbone
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

        # Gene-symbol -> backbone vocab id buffer (filled by setup_token_map)
        self.register_buffer(
            "gene_token_ids",
            torch.full((n_genes,), -1, dtype=torch.long),
            persistent=True,
        )
        self._token_map_initialised = False

        # Adapter: embsize -> hidden_dim (per gene token)
        self.gene_adapter = nn.Sequential(
            nn.Linear(self.embsize, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.pathway_norm = nn.LayerNorm(hidden_dim)
        self.dropout      = nn.Dropout(0.1)

        # Pathway gene-set mask (P, n_genes), filled lazily on first forward
        self._pathway_mask_cached = False

    # ------------------------------------------------------------------
    # Backbone loading
    # ------------------------------------------------------------------

    def _load_backbone(
        self,
        ckpt_dir: Optional[str],
        n_layers_fallback: int,
        n_heads_fallback: int,
    ) -> tuple[nn.Module, int]:
        """
        Try the official scGPT loader. If it fails (no GPU, no flash-attn,
        bad checkpoint path), fall back to a small Transformer so the
        architecture is testable. The fallback's vocab embedding is built
        on demand inside forward().
        """
        if SCGPT_AVAILABLE and ckpt_dir is not None and Path(ckpt_dir).exists():
            try:
                from scgpt.model import TransformerModel  # type: ignore
                # Loading checkpoint config + weights as per scgpt docs.
                # Detailed config keys depend on the specific checkpoint;
                # consumers should follow the upstream tutorial.
                model = TransformerModel.from_pretrained(ckpt_dir)
                emb = getattr(model.config, "embsize", self.embsize)
                return model, int(emb)
            except Exception as e:
                print(f"[ScGPTCellEncoder] scGPT load failed ({e}); using fallback.")

        # From-scratch fallback Transformer
        backbone = nn.ModuleList([
            _FallbackTransformerBlock(self.embsize, n_heads_fallback)
            for _ in range(n_layers_fallback)
        ])
        return backbone, self.embsize

    # ------------------------------------------------------------------
    # Token map
    # ------------------------------------------------------------------

    def setup_token_map(self, vocab: Optional[dict] = None) -> None:
        """
        Build {gene_symbol -> token_id} from a scGPT vocab dict.
        If `vocab` is None, falls back to a deterministic hash so the model
        is still trainable for smoke tests.
        """
        if vocab is None:
            ids = torch.tensor(
                [abs(hash(g)) % 25_000 for g in self.gene_symbols],
                dtype=torch.long,
            )
        else:
            from tqdm.auto import tqdm
            ids_list = []
            for g in tqdm(self.gene_symbols, desc="  Mapping gene symbols (scGPT)",
                          unit="gene", leave=False):
                ids_list.append(int(vocab.get(g, vocab.get(g.upper(), -1))))
            ids = torch.tensor(ids_list, dtype=torch.long)
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
            (B, n_pathways, hidden_dim) pathway tokens.
        """
        if not self._token_map_initialised:
            self.setup_token_map()

        device = expr.device

        # ---- Top-K gene selection per cell ----
        topk_vals, topk_idx = expr.topk(self.top_k, dim=1)        # (B, K)

        # ---- Token embedding via backbone ----
        gene_embeds = self._encode_topk(topk_idx, topk_vals)       # (B, K, embsize)
        gene_embeds = self.gene_adapter(gene_embeds)               # (B, K, hidden_dim)

        # ---- Pathway pool ----
        if not self._pathway_mask_cached:
            mask = torch.zeros(self.n_pathways, self.n_genes, dtype=torch.bool)
            for p_i, pname in enumerate(self.pathway_names):
                for g_idx in self.pathway_gene_map[pname]:
                    mask[p_i, g_idx] = True
            self.register_buffer("_pathway_gene_mask", mask, persistent=False)
            self._pathway_mask_cached = True
        pw_mask = self._pathway_gene_mask.to(device)               # (P, n_genes)

        belong = pw_mask[:, topk_idx]                              # (P, B, K)
        belong = belong.permute(1, 0, 2).float()                   # (B, P, K)
        counts = belong.sum(dim=-1, keepdim=True).clamp(min=1.0)
        out    = torch.einsum("bpk,bkd->bpd", belong, gene_embeds) / counts
        return self.dropout(self.pathway_norm(out))

    def _encode_topk(
        self,
        topk_idx:  torch.Tensor,   # (B, K)
        topk_vals: torch.Tensor,   # (B, K)
    ) -> torch.Tensor:
        token_ids = self.gene_token_ids[topk_idx].clamp(min=0)     # (B, K)

        if isinstance(self.backbone, nn.ModuleList):
            # Fallback: lookup -> stack of Transformer blocks
            if not hasattr(self, "_fallback_embed"):
                vocab_size = max(int(self.gene_token_ids.max().item()) + 1, 25_000)
                self._fallback_embed = nn.Embedding(vocab_size, self.embsize).to(token_ids.device)
            x = self._fallback_embed(token_ids)                    # (B, K, embsize)
            # Modulate by expression magnitude (cheap hack matching scGPT's value embedding)
            x = x * topk_vals.unsqueeze(-1)
            for blk in self.backbone:
                x = blk(x)
            return x

        # Official scGPT path — exact API depends on the loaded checkpoint.
        # Document the expected interface; integrators should adapt as needed.
        with torch.no_grad():
            return self.backbone.encode(token_ids, topk_vals)      # (B, K, embsize)


__all__ = ["ScGPTCellEncoder", "SCGPT_AVAILABLE"]
