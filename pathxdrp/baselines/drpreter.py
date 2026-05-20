"""
DRPreter baseline (Shin et al. 2023, Briefings in Bioinformatics).

Drug:  GATv2 graph encoder (identical featurisation to PathXDRP for a fair comparison).
Cell:  Pathway-grouped Transformer encoder:
         - Same KEGG pathway grouping as PathXDRP's PathwaySetEncoder.
         - Per-gene projection -> mean-pool per pathway -> Transformer encoder.
         - The key difference from PathXDRP: NO pathway-masked cross-attention;
           instead, simple cross-attention (no KEGG mask, no entropy regularisation).
Head:  Mean-pool of cross-attended atom features || global cell embedding -> MLP -> IC50.
Loss:  MSE (no evidential uncertainty).

Reference
---------
Shin Y et al. "DRPreter: Interpretable Anticancer Drug Response Prediction Using
Knowledge-guided Graph Neural Networks and Transformer." IJMS, 2023.
DOI: 10.3390/ijms232112173

(We adapt the architecture to our expression-based cell encoder for a fair comparison;
 the published code uses mutation + CNV as additional omics tracks.)

Differences from PathXDRP
--------------------------
- Cross-attention has NO KEGG pathway mask (hard or soft).
- No entropy regularisation on attention weights.
- No evidential regression head; uses plain MSE loss.
- Lighter Transformer: 2 layers, 4 heads over pathway tokens.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import GATv2Conv, global_add_pool

from pathxdrp.models.cell_encoder import PathwaySetEncoder  # reuse same pathway grouping


# ---
# Drug encoder (GATv2 — same as PathXDRP for fairness)
# ---

class GATDrugEncoder(nn.Module):
    def __init__(
        self,
        node_in_dim: int,
        edge_in_dim: int,
        hidden_dim: int = 128,
        n_layers: int = 3,
        n_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(node_in_dim, hidden_dim)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(n_layers):
            self.convs.append(
                GATv2Conv(hidden_dim, hidden_dim // n_heads, heads=n_heads,
                          edge_dim=edge_in_dim, dropout=dropout, concat=True)
            )
            self.norms.append(nn.LayerNorm(hidden_dim))
        self.drop = nn.Dropout(dropout)

    def forward(self, drug_batch: Batch, batch_idx: torch.Tensor):
        x = F.relu(self.input_proj(drug_batch.x))
        for conv, norm in zip(self.convs, self.norms):
            x = norm(F.relu(conv(x, drug_batch.edge_index, drug_batch.edge_attr)))
            x = self.drop(x)
        h_mol = global_add_pool(x, batch_idx)    # (B, hidden_dim)
        return x, h_mol                           # (N_atoms, D), (B, D)


# ---
# Unmasked cross-attention
# ---

class UnmaskedCrossAttention(nn.Module):
    """Standard MHA: drug atoms attend over pathway tokens — no KEGG mask."""

    def __init__(self, hidden_dim: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.mha  = nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout, batch_first=False)
        self.norm = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        h_drug: torch.Tensor,        # (N_atoms, D)
        h_cell: torch.Tensor,        # (B, N_pathways, D)
        atom_batch: torch.Tensor,    # (N_atoms,)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B = h_cell.size(0)
        D = h_drug.size(-1)

        # Local index of each atom within its molecule
        atoms_per_mol = atom_batch.bincount(minlength=B)          # (B,)
        max_atoms = int(atoms_per_mol.max().item())
        mol_offsets = torch.zeros(B, dtype=torch.long, device=h_drug.device)
        mol_offsets[1:] = atoms_per_mol[:-1].cumsum(0)
        local_idx = (
            torch.arange(len(atom_batch), device=h_drug.device)
            - mol_offsets[atom_batch]
        )                                                           # (N_atoms,)

        # Padded query: (max_atoms, B, D)  — pad positions stay zero
        h_padded = h_drug.new_zeros(max_atoms, B, D)
        h_padded[local_idx, atom_batch] = h_drug

        # Keys/values: (P, B, D)
        kv = h_cell.permute(1, 0, 2).contiguous()                 # (P, B, D)

        # Single batched MHA — replaces B individual kernel launches
        ctx_padded, w = self.mha(h_padded, kv, kv)
        # ctx_padded: (max_atoms, B, D) | w: (B, max_atoms, P)

        # Scatter back to flat atoms + residual + norm
        ctx_flat = ctx_padded[local_idx, atom_batch]               # (N_atoms, D)
        out_context = self.norm(ctx_flat + h_drug)

        # Flatten attention weights (drop padding positions)
        out_weights = w.permute(1, 0, 2)[local_idx, atom_batch]   # (N_atoms, P)

        return out_context, out_weights


# ---
# Full model
# ---

class DRPreter(nn.Module):
    """
    DRPreter: GATv2 drug + pathway Transformer cell + unmasked cross-attention.

    Identical interface to PathXDRP:
      out = model(drug_batch=batch, expr=expr, y=y)
    """

    def __init__(
        self,
        node_in_dim: int,
        edge_in_dim: int,
        n_genes: int,
        pathway_gene_map: dict[str, list[int]],
        hidden_dim: int = 128,
        n_gat_layers: int = 3,
        n_attn_heads: int = 8,
        dropout: float = 0.1,
        n_pw_stats: int = 4,
    ) -> None:
        super().__init__()
        self.drug_enc = GATDrugEncoder(
            node_in_dim, edge_in_dim, hidden_dim, n_gat_layers, n_attn_heads, dropout
        )
        self.cell_enc = PathwaySetEncoder(n_genes, pathway_gene_map, hidden_dim, dropout,
                                          n_pw_stats=n_pw_stats)
        self.cross_attn = UnmaskedCrossAttention(hidden_dim, n_attn_heads, dropout)
        self.pool_proj = nn.Linear(hidden_dim, hidden_dim)
        self.pool_norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def forward(
        self,
        drug_batch: Batch,
        expr: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        **_,
    ) -> dict:
        h_atom, h_mol = self.drug_enc(drug_batch, drug_batch.batch)
        h_cell = self.cell_enc(expr)                           # (B, P, D)

        ctx, attn = self.cross_attn(h_atom, h_cell, drug_batch.batch)

        # Attention-weighted pool
        a_w = attn.mean(dim=-1, keepdim=True)
        h_drug_ctx = global_add_pool(ctx * a_w, drug_batch.batch)  # (B, D)
        h_drug_ctx = self.pool_norm(self.pool_proj(h_drug_ctx))

        h_cell_global = h_cell.mean(dim=1)                    # (B, D)
        z = torch.cat([h_drug_ctx, h_cell_global], dim=-1)    # (B, 2D)
        pred = self.head(z).squeeze(-1)                        # (B,)

        out: dict = {"pred": pred, "attn_weights": attn}
        if y is not None:
            out["loss"] = F.mse_loss(pred, y)
        return out
