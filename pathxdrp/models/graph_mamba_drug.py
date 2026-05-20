"""
Graph-Mamba drug encoder — alternative to DrugGATEncoder.

Hybrid design: a small GAT stack for local message-passing followed by a
bidirectional Mamba block over the atom sequence (degree-prioritised) for
long-range context. Linear-time complexity in the number of atoms; on
drug-like molecules (~30-50 atoms) the wall-clock difference vs GATv2 is
modest, but the architecture demonstrates the SSM-on-graphs paradigm and
gives the paper an "edge-device inference" story.

References
----------
Graph-Mamba: Towards Long-Range Graph Sequence Modeling with Selective
State Spaces (Wang et al., 2024). https://arxiv.org/abs/2402.00789
GitHub: bowang-lab/Graph-Mamba

MOL-Mamba (AAAI 2025) for the GAT + Mamba hybrid pattern.

Pipeline
--------
  drug_batch (PyG)
    |                                              # local message-passing
    v
  GATv2 ×N_gat (N_atoms, hidden_dim)
    |                                              # to dense, prioritise by degree
    v
  per-batch (B, max_n, D), pad mask
    |                                              # bidirectional Mamba over sequence
    v
  Bi-Mamba ×N_mamba (B, max_n, D)
    |                                              # unpad
    v
  (N_atoms, hidden_dim) — same interface as DrugGATEncoder

Import guards
-------------
`mamba_ssm` is CUDA-only. If unavailable, importing this module succeeds
but `GraphMambaDrugEncoder.__init__` raises ImportError. Smoke tests gate
on `MAMBA_AVAILABLE`.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch_geometric.utils import degree, to_dense_batch

# Soft import — see mamba_cell.py for the same pattern.
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


# ---
# Bi-Mamba block over a padded atom sequence
# ---

class _BiMambaBlock(nn.Module):
    """Forward + reverse Mamba with a gated combination + residual + LayerNorm."""

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

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        """
        x:        (B, L, D) padded atom-sequence embeddings
        pad_mask: (B, L)    True = real atom, False = padding
        """
        # Zero out padding before feeding to Mamba so they don't pollute the SSM state
        x_in = x * pad_mask.unsqueeze(-1).float()
        h_f  = self.fwd(x_in)
        h_b  = torch.flip(self.bwd(torch.flip(x_in, dims=[1])), dims=[1])
        h    = self.gate(torch.cat([h_f, h_b], dim=-1))
        return self.norm(x + h)


# ---
# Hybrid GAT + Bi-Mamba drug encoder
# ---

class GraphMambaDrugEncoder(nn.Module):
    """
    Drug-graph encoder: local GAT message passing + global Bi-Mamba sequence model.

    Drop-in replacement for `DrugGATEncoder`:
      forward(data, smiles_list=None, batch=None) -> (h_atom, h_mol)
      where h_atom is (N_atoms_total, hidden_dim) and h_mol is (B, hidden_dim).

    Args:
        node_in_dim:   Atom feature dimension.
        edge_in_dim:   Bond feature dimension.
        hidden_dim:    Internal embedding dimension.
        n_gat_layers:  Local message-passing layers (default 2).
        n_mamba_layers: Bi-Mamba blocks over the atom sequence (default 2).
        n_heads:       GAT heads.
        dropout:       Dropout probability.
        ordering:      "degree" (default), "canonical" (preserve PyG ordering),
                       or "random" (shuffle each forward — for ablation).
    """

    def __init__(
        self,
        node_in_dim: int,
        edge_in_dim: int,
        hidden_dim: int = 128,
        n_gat_layers: int = 2,
        n_mamba_layers: int = 2,
        n_heads: int = 8,
        dropout: float = 0.1,
        ordering: str = "degree",
    ) -> None:
        super().__init__()
        if not MAMBA_AVAILABLE:
            raise ImportError(
                "mamba_ssm not installed. Install with:\n"
                "  pip install mamba-ssm[causal-conv1d] --no-build-isolation\n"
                "Requires NVIDIA GPU + CUDA 11.6+."
            )
        if ordering not in ("degree", "canonical", "random"):
            raise ValueError(f"Unknown ordering: {ordering}")

        self.hidden_dim    = hidden_dim
        self.ordering      = ordering
        self.n_mamba_layers = n_mamba_layers

        # ---- Input projection ----
        self.node_proj = nn.Linear(node_in_dim, hidden_dim)

        # ---- Local message passing (GAT) ----
        self.gat_convs = nn.ModuleList()
        self.gat_norms = nn.ModuleList()
        for _ in range(n_gat_layers):
            self.gat_convs.append(
                GATv2Conv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim // n_heads,
                    heads=n_heads,
                    edge_dim=edge_in_dim,
                    concat=True,
                    dropout=dropout,
                )
            )
            self.gat_norms.append(nn.LayerNorm(hidden_dim))

        # ---- Global Mamba (over atom sequence) ----
        self.mamba_blocks = nn.ModuleList([
            _BiMambaBlock(hidden_dim) for _ in range(n_mamba_layers)
        ])

        self.dropout = nn.Dropout(dropout)
        self.act     = nn.GELU()

    # ------------------------------------------------------------------
    # Sequence ordering
    # ------------------------------------------------------------------

    def _atom_order(self, edge_index: torch.Tensor, batch_idx: torch.Tensor) -> torch.Tensor:
        """
        Return a permutation (N_atoms,) giving the order in which atoms enter
        the Mamba sequence within each batch element. Defaults to descending
        degree (high-connectivity atoms first — analogous to Graph-Mamba's
        prioritisation strategy). Stable per-batch.
        """
        n = batch_idx.numel()
        if self.ordering == "canonical":
            return torch.arange(n, device=batch_idx.device)

        if self.ordering == "random":
            # Independent permutation per batch element
            order = torch.empty(n, dtype=torch.long, device=batch_idx.device)
            for b in batch_idx.unique():
                mask = batch_idx == b
                idx  = torch.nonzero(mask, as_tuple=False).squeeze(-1)
                order[mask] = idx[torch.randperm(idx.numel(), device=batch_idx.device)]
            return order

        # "degree" — sort each batch element by descending node degree
        deg = degree(edge_index[0], num_nodes=n).to(batch_idx.device)
        order = torch.empty(n, dtype=torch.long, device=batch_idx.device)
        for b in batch_idx.unique():
            mask = batch_idx == b
            idx  = torch.nonzero(mask, as_tuple=False).squeeze(-1)
            sub_deg = deg[idx]
            sorted_local = torch.argsort(sub_deg, descending=True)
            order[mask] = idx[sorted_local]
        return order

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        data: Batch,
        smiles_list: Optional[list[str]] = None,   # unused — kept for interface parity
        batch: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            data:  PyG Batch with x, edge_index, edge_attr, batch.
            batch: (N_atoms,) batch index. Inferred from data if None.

        Returns:
            h_atom: (N_atoms_total, hidden_dim)  — per-atom embeddings
            h_mol:  (B, hidden_dim)              — graph-level pooled embedding
        """
        if batch is None:
            batch = data.batch if hasattr(data, "batch") and data.batch is not None else \
                    torch.zeros(data.x.size(0), dtype=torch.long, device=data.x.device)

        # ---- 1. Input projection ----
        x = self.act(self.node_proj(data.x))

        # ---- 2. Local message passing ----
        for conv, norm in zip(self.gat_convs, self.gat_norms):
            x_new = conv(x, data.edge_index, data.edge_attr)
            x = norm(x + self.dropout(x_new))
            x = self.act(x)

        # ---- 3. Permute atoms by chosen ordering, then pad to dense batch ----
        perm     = self._atom_order(data.edge_index, batch)
        inv_perm = torch.empty_like(perm)
        inv_perm[perm] = torch.arange(perm.numel(), device=perm.device)

        x_perm        = x[perm]
        batch_perm    = batch[perm]
        x_pad, pad_m  = to_dense_batch(x_perm, batch_perm)        # (B, max_n, D), (B, max_n)

        # ---- 4. Bi-Mamba over atom sequence ----
        for blk in self.mamba_blocks:
            x_pad = blk(x_pad, pad_m)

        # ---- 5. Unpad and invert the permutation ----
        x_perm_out = x_pad[pad_m]                                  # (N_atoms, D), order = perm
        x_out      = torch.empty_like(x_perm_out)
        x_out[perm] = x_perm_out                                   # back to original ordering

        # ---- 6. Graph-level pool ----
        h_mol = global_mean_pool(x_out, batch)                     # (B, D)

        return x_out, h_mol


# ---
# Public API
# ---

__all__ = ["GraphMambaDrugEncoder", "MAMBA_AVAILABLE"]
