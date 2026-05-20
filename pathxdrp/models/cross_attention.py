"""
Pathway-masked multi-head cross-attention.

Drug atom embeddings act as Query (Q).
Cell pathway tokens act as Key (K) and Value (V).

Memory-efficient design
-----------------------
Atoms arrive as a flat (N_atoms_total, D) tensor with a batch-index vector.
Instead of expanding K/V to (N_atoms_total, N_pathways, D) — which OOMs at
batch_size=256 with 370 pathways — we:

  1. Pad atoms to a dense batch:  (B, max_n_atoms, D)  via to_dense_batch.
  2. Compute attention in (B, H, max_n, P) space.                ← 20× less memory
  3. Unpad the output back to (N_atoms_total, D).

Mask types
----------
  none : standard scaled dot-product attention.
  soft : add a single learnable scalar prior to logits (trained jointly).
  hard : add -inf to (atom, pathway) pairs not connected via KEGG/STRING.
         Requires a (N_atoms_total, N_pathways) boolean mask passed as hard_mask.
         (hard_mask construction: scripts/build_pathway_mask.py; not yet built)

Entropy regularisation
----------------------
L_entropy = mean over atoms of H(attn_distribution_over_pathways).
Penalises flat distributions; encourages sparse, pathway-specific attention.
Multiplied by entropy_reg_weight in pathxdrp.py before adding to the main loss.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch


class PathwayMaskedCrossAttention(nn.Module):
    """
    Memory-efficient cross-attention between drug atoms and cell pathway tokens.

    Args:
        hidden_dim:          Input/output embedding dimension.
        n_heads:             Number of attention heads.
        dropout:             Dropout applied to attention weights.
        mask_type:           "none" | "soft" | "hard".
        entropy_reg_weight:  Coefficient for the entropy regularisation loss
                             (stored as _entropy_loss, added by PathXDRP.forward).
        use_residual:        If True, the attention output is added back to the
                             query (drug atom) embedding and LayerNorm-ed —
                             standard transformer pattern. This makes the
                             cross-attention path load-bearing so gradients
                             actually flow into the attention weights. Without
                             it, the head can ignore the attention output
                             entirely (the model relies on h_mol + h_cell_global
                             instead, leaving attention as dead code with
                             arbitrary weights). The XAI diagnostic flagged
                             this as the dominant cause of PathXDRP's
                             unfaithful attention. Default False keeps backward
                             compat: setting True adds an ``attn_norm``
                             LayerNorm parameter that old checkpoints lack, so
                             toggling on means retraining from scratch.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        n_heads: int = 8,
        dropout: float = 0.1,
        mask_type: str = "soft",
        entropy_reg_weight: float = 0.01,
        n_pathways: int = 0,
        use_residual: bool = False,
    ) -> None:
        super().__init__()
        assert hidden_dim % n_heads == 0, "hidden_dim must be divisible by n_heads"
        self.hidden_dim        = hidden_dim
        self.n_heads           = n_heads
        self.head_dim          = hidden_dim // n_heads
        self.scale             = math.sqrt(self.head_dim)
        self.mask_type         = mask_type
        self.entropy_reg_weight = entropy_reg_weight
        self.use_residual      = use_residual

        self.q_proj  = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj  = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj  = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm_q  = nn.LayerNorm(hidden_dim)
        self.norm_kv = nn.LayerNorm(hidden_dim)
        self.attn_drop = nn.Dropout(dropout)
        # Post-attention LayerNorm is only registered when residual is enabled,
        # so old checkpoints load cleanly with use_residual=False.
        self.attn_norm: nn.Module | None = (
            nn.LayerNorm(hidden_dim) if use_residual else None
        )

        # Soft mask: one learnable prior per pathway (broadcast over heads/atoms).
        # Shape (1, 1, 1, n_pathways) gives different bias per pathway, which
        # actually influences softmax (unlike the old (1,H,1,1) shape that added
        # a uniform constant and had zero effect on softmax outputs).
        if mask_type == "soft" and n_pathways > 0:
            self.soft_mask_logit = nn.Parameter(
                torch.zeros(1, 1, 1, n_pathways)
            )

        # Cache for XAI
        self._last_attn:    Optional[torch.Tensor] = None
        self._entropy_loss: torch.Tensor = torch.tensor(0.0)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        h_drug: torch.Tensor,             # (N_atoms_total, D)
        h_cell: torch.Tensor,             # (B, N_pathways, D)
        atom_batch: torch.Tensor,         # (N_atoms_total,) int — batch index per atom
        hard_mask: Optional[torch.Tensor] = None,
            # (N_atoms_total, N_pathways) bool — True = allowed
            # (used only when mask_type == "hard")
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        context      : (N_atoms_total, D)        — attended joint features
        attn_weights : (N_atoms_total, N_pathways) — per-atom attention map (XAI)
        """
        B          = h_cell.size(0)
        N_pathways = h_cell.size(1)
        N_atoms    = h_drug.size(0)

        # ---- Project ----
        Q = self.q_proj(self.norm_q(h_drug))      # (N_atoms, D)
        K = self.k_proj(self.norm_kv(h_cell))     # (B, P, D)
        V = self.v_proj(h_cell)                   # (B, P, D)  (no norm on V)

        # ---- Pad atoms to dense batch  (avoids per-atom expansion of K, V) ----
        Q_pad, pad_mask = to_dense_batch(Q, atom_batch)   # (B, max_n, D), (B, max_n) bool
        max_n = Q_pad.size(1)

        # ---- Reshape for multi-head  → (B, H, seq, Dh) ----
        def to_mh(x: torch.Tensor, seq: int) -> torch.Tensor:
            return x.view(x.size(0), seq, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        Q_mh = to_mh(Q_pad, max_n)         # (B, H, max_n, Dh)
        K_mh = to_mh(K,     N_pathways)    # (B, H, P,     Dh)
        V_mh = to_mh(V,     N_pathways)    # (B, H, P,     Dh)

        # ---- Attention scores  (B, H, max_n, P) ----
        scores = torch.einsum("bhnd,bhpd->bhnp", Q_mh, K_mh) / self.scale

        # Mask padding atoms: use a large finite value instead of -inf.
        # Using -inf causes softmax to produce NaN for all-masked rows, and
        # SoftmaxBackward0 propagates those NaN gradients even through nan_to_num
        # (since 0 * NaN = NaN in IEEE 754), corrupting model weights from batch 2.
        # Use -1e4 instead of -1e9: fits in float16 (max ~65504) for AMP safety.
        pad_atom_mask = ~pad_mask                              # True = padding position
        scores = scores.masked_fill(
            pad_atom_mask.unsqueeze(1).unsqueeze(-1),          # (B,1,max_n,1)
            -1e4,
        )

        # Apply pathway mask
        if self.mask_type == "hard" and hard_mask is not None:
            # hard_mask: (N_atoms, P) — True = allowed
            # Expand to (B, 1, max_n, P) via to_dense_batch
            hm_pad, _ = to_dense_batch(hard_mask.float(), atom_batch)  # (B, max_n, P)
            # False (not allowed) → large negative (not -inf, to avoid NaN in backward)
            forbidden = hm_pad.unsqueeze(1) == 0                       # (B,1,max_n,P)
            scores = scores.masked_fill(forbidden, -1e4)

        elif self.mask_type == "soft":
            # Broadcast (1, H, 1, 1) learned bias
            scores = scores + self.soft_mask_logit

        # ---- Softmax + dropout ----
        # With -1e4 masking (not -inf), softmax never produces NaN, and -1e4
        # nan_to_num is no longer needed.
        attn = F.softmax(scores, dim=-1)            # (B, H, max_n, P)
        attn = self.attn_drop(attn)

        # ---- Aggregate ----
        context_pad = torch.einsum("bhnp,bhpd->bhnd", attn, V_mh)  # (B,H,max_n,Dh)
        context_pad = (
            context_pad.permute(0, 2, 1, 3)
            .contiguous()
            .view(B, max_n, self.hidden_dim)
        )                                                            # (B, max_n, D)
        context_pad = self.out_proj(context_pad)

        # ---- Unpad  → (N_atoms, D) ----
        context = context_pad[pad_mask]              # (N_atoms, D)

        # ---- Residual + LayerNorm (standard transformer pattern) ----
        # Optional and opt-in via use_residual. Without this, the attention
        # output is one block in a 5-block concat that the head can ignore;
        # with it, every gradient step on the prediction loss touches every
        # attention weight, forcing them to be useful.
        if self.use_residual and self.attn_norm is not None:
            context = self.attn_norm(context + h_drug)

        # ---- Per-atom attention weights for XAI (mean over heads) ----
        attn_mean = attn.mean(dim=1)                 # (B, max_n, P)
        attn_flat  = attn_mean[pad_mask]             # (N_atoms, P)

        # ---- Entropy regularisation ----
        p = attn_flat + 1e-9
        entropy_loss = -(p * p.log()).sum(dim=-1).mean()

        # Cache for XAI + loss
        self._last_attn    = attn_flat.detach()
        self._entropy_loss = entropy_loss

        return context, attn_flat
