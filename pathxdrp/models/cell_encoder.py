"""
Cell-line encoder — expression mode only.

Applies a shared MLP within each KEGG pathway gene set, then mean-pools
to produce one embedding token per pathway.

Input:  (B, N_genes) Z-scored log2(TPM+1) gene expression
Output: (B, N_pathways, hidden_dim)  — used as Key/Value in cross-attention
"""
from __future__ import annotations

import torch
import torch.nn as nn


class PathwaySetEncoder(nn.Module):
    """
    Groups genes by KEGG pathway, computes per-pathway summary statistics,
    projects to hidden_dim tokens, then optionally applies a cross-pathway
    Transformer to model inter-pathway co-activation.

    Per-pathway features (3 statistics):
      - mean expression   : average signal strength across pathway genes
      - std expression    : variability / heterogeneity within the pathway
      - frac_active       : fraction of pathway genes above Z=0 (population mean)
                            distinguishes "uniformly low" from "many genes on"

    Cross-pathway Transformer:
      One TransformerEncoder layer (pre-LayerNorm, 4 heads) lets each pathway
      token attend to all others, capturing biological crosstalk such as
      PI3K→AKT→mTOR co-activation or DNA-repair/cell-cycle mutual exclusivity.
      Controlled by n_pw_transformer_layers (0 = disabled).

    Args:
        n_genes:                  Total number of genes in the expression vector.
        pathway_gene_map:         {pathway_name: [gene_indices_into_expr_vector]}
        hidden_dim:               Output embedding dimension.
        dropout:                  Dropout probability applied to final tokens.
        n_pw_transformer_layers:  Number of cross-pathway TransformerEncoder layers
                                  (0 to disable, 1 is the recommended default).
    """

    def __init__(
        self,
        n_genes: int,
        pathway_gene_map: dict[str, list[int]],
        hidden_dim: int = 128,
        dropout: float = 0.1,
        n_pw_transformer_layers: int = 1,
        n_pw_stats: int = 4,
        frac_active_sharpness: float = 0.0,
    ) -> None:
        """frac_active_sharpness controls the threshold used for the
        ``frac_active`` pathway statistic. Default 0.0 keeps the historical
        hard threshold ``mean(x > 0)`` (preserves checkpoint compatibility).
        Setting a positive value (e.g. 10.0) replaces it with a sigmoid soft
        threshold ``mean(sigmoid(k * x))`` so that small Gaussian noise added
        to the expression input by ``--expr_noise_std`` no longer flips
        ~10% of the underlying boolean per batch."""
        super().__init__()
        self.pathway_gene_map = pathway_gene_map
        self.pathway_names = sorted(pathway_gene_map.keys())
        self.n_pathways = len(self.pathway_names)
        self.hidden_dim = hidden_dim
        self.frac_active_sharpness = float(frac_active_sharpness)

        # Per-pathway projection: [mean, std, frac_active, (optionally max)] -> hidden_dim
        # n_pw_stats=3 for v1 checkpoints; n_pw_stats=4 for v2+ (adds max Z-score).
        self.n_pw_stats = n_pw_stats
        self.gene_proj = nn.Sequential(
            nn.Linear(n_pw_stats, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        # Cross-pathway Transformer (optional)
        # Pre-LayerNorm (norm_first=True) for training stability.
        # n_heads: hidden_dim // 32 gives 32D per head (4 heads at 128D, 8 at 256D).
        # No dropout inside the transformer — sparse pathway attention is already
        # regularising; extra dropout fights against the sparse attention signal.
        _n_heads = max(1, hidden_dim // 32)
        if n_pw_transformer_layers > 0:
            self.pathway_transformer: nn.Module | None = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=_n_heads,
                    dim_feedforward=hidden_dim * 2,
                    dropout=0.0,
                    batch_first=True,
                    norm_first=True,
                ),
                num_layers=n_pw_transformer_layers,
                enable_nested_tensor=False,
            )
        else:
            self.pathway_transformer = None

        self.pathway_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # ---- Pre-compute a dense averaging matrix for the vectorized forward ----
        # avg_matrix[p, i] = 1/|pathway_p| if gene-pair i belongs to pathway p, else 0.
        # Shape: (n_pathways, n_pairs).  Stored as a float32 buffer; a single
        # cuBLAS GEMM replaces the Python for-loop AND the slow index_add_ scatter.
        # Memory: (370 x 37950) x 4 bytes ~= 56 MB -- affordable and constant.
        flat_gene: list[int] = []
        flat_pw:   list[int] = []
        for p_idx, pname in enumerate(self.pathway_names):
            for g in pathway_gene_map[pname]:
                flat_gene.append(g)
                flat_pw.append(p_idx)

        n_pairs = len(flat_gene)
        pw_sizes = torch.tensor(
            [float(max(len(pathway_gene_map[pname]), 1)) for pname in self.pathway_names],
            dtype=torch.float32,
        )                                                           # (n_pathways,)

        avg_mat = torch.zeros(self.n_pathways, max(n_pairs, 1))
        for i, (pw, _g) in enumerate(zip(flat_pw, flat_gene)):
            avg_mat[pw, i] = 1.0 / pw_sizes[pw].item()

        self.register_buffer("avg_matrix",   avg_mat)              # (n_pathways, n_pairs)
        self.register_buffer("flat_gene_idx",
            torch.tensor(flat_gene, dtype=torch.long) if flat_gene
            else torch.zeros(0, dtype=torch.long))
        # flat_pw_idx: only needed for n_pw_stats=4 (scatter-reduce amax).
        # Not registered for n_pw_stats=3 so v1 checkpoints load cleanly.
        if n_pw_stats >= 4:
            self.register_buffer("flat_pw_idx",
                torch.tensor(flat_pw, dtype=torch.long) if flat_pw
                else torch.zeros(0, dtype=torch.long))
        self._n_pairs = n_pairs

    def forward(self, expr: torch.Tensor) -> torch.Tensor:
        """
        Args:
            expr: (B, N_genes) gene expression Z-scores.
        Returns:
            h_cell: (B, N_pathways, hidden_dim)
        """
        B = expr.size(0)

        if self._n_pairs == 0:
            return self.dropout(
                torch.zeros(B, self.n_pathways, self.hidden_dim,
                            device=expr.device, dtype=expr.dtype)
            )

        # 1. Select all pathway-gene scalars (reuse avg_matrix GEMM for all stats).
        gene_expr = expr[:, self.flat_gene_idx]                     # (B, n_pairs)
        gene_expr_f = gene_expr.to(self.avg_matrix.dtype)

        # mean: weighted average of expression values per pathway
        mean_pw = torch.matmul(gene_expr_f, self.avg_matrix.T)     # (B, n_pathways)

        # std: Var(X) = E[X²] - E[X]²  (one extra GEMM, no loops)
        mean_sq_pw = torch.matmul(gene_expr_f ** 2, self.avg_matrix.T)  # (B, n_pathways)
        std_pw = (mean_sq_pw - mean_pw ** 2).clamp(min=0).sqrt()   # (B, n_pathways)

        # frac_active: fraction of pathway genes with Z-score > 0.
        # Hard threshold (sharpness=0) preserves the historical (x > 0) boolean
        # used by v3 checkpoints. A positive sharpness uses a sigmoid soft
        # threshold sigmoid(k * x), which is differentiable and stable under
        # the small Gaussian noise added by --expr_noise_std (otherwise the
        # boolean flips on ~10% of genes per batch, adding label-uncorrelated
        # noise the model has to compensate for).
        if self.frac_active_sharpness > 0.0:
            frac_active_input = torch.sigmoid(self.frac_active_sharpness * gene_expr_f)
        else:
            frac_active_input = (gene_expr_f > 0).to(self.avg_matrix.dtype)
        frac_pw = torch.matmul(frac_active_input, self.avg_matrix.T)  # (B, n_pathways)

        # max expression (v2+ only, n_pw_stats=4)
        if self.n_pw_stats >= 4:
            max_pw = gene_expr_f.new_full((B, self.n_pathways), fill_value=-10.0)
            max_pw.scatter_reduce_(
                1, self.flat_pw_idx.unsqueeze(0).expand(B, -1), gene_expr_f,
                reduce='amax', include_self=True,
            )
            features = torch.stack([mean_pw, std_pw, frac_pw, max_pw], dim=-1)  # (B, P, 4)
        else:
            features = torch.stack([mean_pw, std_pw, frac_pw], dim=-1)           # (B, P, 3)

        # 2. Project per-pathway features → hidden_dim.
        h_cell = self.gene_proj(features)                           # (B, n_pathways, D)

        # 3. Cross-pathway Transformer (models inter-pathway crosstalk).
        #    Skipped if n_pw_transformer_layers=0 at construction.
        if self.pathway_transformer is not None:
            h_cell = self.pathway_transformer(h_cell)               # (B, n_pathways, D)

        # 4. Final LayerNorm + Dropout
        return self.dropout(self.pathway_norm(h_cell))
