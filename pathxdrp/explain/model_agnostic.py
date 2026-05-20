"""Model-agnostic attribution methods.

These work for ANY model whose forward pass takes (drug_input, expr) -> scalar
prediction. Required for cross-model XAI comparison: only one of the existing
methods (Integrated Gradients on expression) currently runs across all four
trained architectures. This module adds two more (permutation importance and
occlusion) and computes them at both gene and pathway granularity.

Both methods perturb the *input expression vector* and measure the resulting
change in prediction. They do not need access to attention maps, gradients,
or any architecture-specific internals — so CDRScan and GraphDRP get the same
treatment as PathXDRP and DRPreter.

Naming conventions
------------------
- ``forward_fn``: callable ``forward_fn(expr) -> Tensor[B]`` that wraps the
  drug branch + the model and returns one prediction per row of ``expr``.
  Built once per (drug, model, batch) by the runner so this module stays
  agnostic to the four models' different APIs.
- ``baseline``: prediction on the unperturbed expression matrix.
- Returned scores are always ``(n_features,)`` numpy arrays where higher means
  "perturbing this feature changed the prediction more" — i.e. higher = more
  attributed.
"""
from __future__ import annotations

from typing import Callable, Iterable, Sequence

import numpy as np
import torch


# ---
# Permutation importance
# ---

@torch.no_grad()
def permutation_importance_genes(
    forward_fn: Callable[[torch.Tensor], torch.Tensor],
    expr: torch.Tensor,
    gene_indices: Sequence[int] | None = None,
    n_shuffles: int = 1,
    aggregate: str = "mean_abs_diff",
) -> np.ndarray:
    """Per-gene permutation importance.

    For each gene g, shuffle that column across the batch dimension and
    measure the change in prediction. ``gene_indices=None`` scores every
    gene (slow on 19 193 genes — pass a subset).

    aggregate:
      - ``mean_abs_diff``  : mean |y_perm - y_base| across batch (default)
      - ``rmse_diff``      : sqrt(mean((y_perm - y_base)**2))
    """
    device = expr.device
    n_genes = expr.size(1)
    if gene_indices is None:
        gene_indices = range(n_genes)
    gene_indices = list(gene_indices)

    base = forward_fn(expr).detach().cpu().numpy()                # (B,)
    scores = np.zeros(n_genes, dtype=np.float64)
    for g in gene_indices:
        gscore = 0.0
        for _ in range(n_shuffles):
            perm = torch.randperm(expr.size(0), device=device)
            perturbed = expr.clone()
            perturbed[:, g] = expr[perm, g]
            y = forward_fn(perturbed).detach().cpu().numpy()
            if aggregate == "mean_abs_diff":
                gscore += float(np.mean(np.abs(y - base)))
            elif aggregate == "rmse_diff":
                gscore += float(np.sqrt(np.mean((y - base) ** 2)))
            else:
                raise ValueError(f"unknown aggregate: {aggregate}")
        scores[g] = gscore / n_shuffles
    return scores


@torch.no_grad()
def permutation_importance_pathways(
    forward_fn: Callable[[torch.Tensor], torch.Tensor],
    expr: torch.Tensor,
    pathway_gene_indices: Sequence[Sequence[int]],
    n_shuffles: int = 1,
    aggregate: str = "mean_abs_diff",
) -> np.ndarray:
    """Per-pathway permutation importance.

    For each pathway p, jointly shuffle all genes in p across the batch
    dimension and measure prediction change. Cheaper and biologically more
    meaningful than gene-level permutation: a single gene rarely matters in
    isolation but a coordinated shuffle of a pathway does.
    """
    device = expr.device
    base = forward_fn(expr).detach().cpu().numpy()
    scores = np.zeros(len(pathway_gene_indices), dtype=np.float64)
    for p, members in enumerate(pathway_gene_indices):
        if not members:
            continue
        members_t = torch.tensor(list(members), device=device, dtype=torch.long)
        pscore = 0.0
        for _ in range(n_shuffles):
            perm = torch.randperm(expr.size(0), device=device)
            perturbed = expr.clone()
            perturbed[:, members_t] = expr[perm][:, members_t]
            y = forward_fn(perturbed).detach().cpu().numpy()
            if aggregate == "mean_abs_diff":
                pscore += float(np.mean(np.abs(y - base)))
            elif aggregate == "rmse_diff":
                pscore += float(np.sqrt(np.mean((y - base) ** 2)))
            else:
                raise ValueError(f"unknown aggregate: {aggregate}")
        scores[p] = pscore / n_shuffles
    return scores


# ---
# Occlusion
# ---

@torch.no_grad()
def occlusion_genes(
    forward_fn: Callable[[torch.Tensor], torch.Tensor],
    expr: torch.Tensor,
    gene_indices: Sequence[int] | None = None,
    baseline_value: float = 0.0,
) -> np.ndarray:
    """Per-gene occlusion: zero out one gene at a time, measure |Δ prediction|.

    Cleaner than permutation when the input is already z-scored (baseline=0
    means "average expression" rather than a permuted value from another cell
    line). No randomness — single forward pass per gene.
    """
    n_genes = expr.size(1)
    if gene_indices is None:
        gene_indices = range(n_genes)
    gene_indices = list(gene_indices)

    base = forward_fn(expr).detach().cpu().numpy()
    scores = np.zeros(n_genes, dtype=np.float64)
    for g in gene_indices:
        perturbed = expr.clone()
        perturbed[:, g] = baseline_value
        y = forward_fn(perturbed).detach().cpu().numpy()
        scores[g] = float(np.mean(np.abs(y - base)))
    return scores


@torch.no_grad()
def occlusion_pathways(
    forward_fn: Callable[[torch.Tensor], torch.Tensor],
    expr: torch.Tensor,
    pathway_gene_indices: Sequence[Sequence[int]],
    baseline_value: float = 0.0,
) -> np.ndarray:
    """Per-pathway occlusion: zero out all genes in a pathway, measure |Δ pred|.

    The pathway-level analogue of ``occlusion_genes``. Outputs ``(n_pathways,)``.
    """
    device = expr.device
    base = forward_fn(expr).detach().cpu().numpy()
    scores = np.zeros(len(pathway_gene_indices), dtype=np.float64)
    for p, members in enumerate(pathway_gene_indices):
        if not members:
            continue
        members_t = torch.tensor(list(members), device=device, dtype=torch.long)
        perturbed = expr.clone()
        perturbed[:, members_t] = baseline_value
        y = forward_fn(perturbed).detach().cpu().numpy()
        scores[p] = float(np.mean(np.abs(y - base)))
    return scores


# ---
# Faithfulness curve (ROAR-light: no retraining)
# ---

@torch.no_grad()
def faithfulness_curve(
    forward_fn: Callable[[torch.Tensor], torch.Tensor],
    expr: torch.Tensor,
    feature_scores: np.ndarray,
    fractions: Iterable[float] = (0.05, 0.10, 0.20, 0.30, 0.50),
    direction: str = "comp",
    baseline_value: float = 0.0,
) -> dict:
    """Faithfulness over multiple top-K fractions, returns per-fraction AUC.

    The previous benchmark reported a single number (faithfulness at K=20%).
    This version sweeps K and reports the area under the |Δ prediction| curve,
    which is the ROAR/MORF metric without retraining (full ROAR would retrain
    after each removal — too expensive for 4 models × 5 splits × 5 seeds).

    direction:
      - ``comp`` (comprehensiveness, MORF): zero the TOP-K most-attributed
        features. A faithful attribution causes a large drop. Higher = better.
      - ``suff`` (sufficiency, LERF): zero the BOTTOM-K (keep only top
        features). A faithful attribution causes only a small change. Lower
        = better.

    Returns a dict with per-fraction deltas and a scalar AUC summary.
    """
    base = forward_fn(expr).detach().cpu().numpy()
    n_features = feature_scores.shape[0]
    order_desc = np.argsort(feature_scores)[::-1]   # most attributed first

    deltas = []
    fracs_used = []
    for f in fractions:
        k = max(1, int(round(n_features * float(f))))
        idx = order_desc[:k] if direction == "comp" else order_desc[k:]
        # idx for "suff" keeps only the top-k features intact and zeros the rest;
        # idx for "comp" zeros the top-k features.
        if direction == "comp":
            mask_idx = idx
        elif direction == "suff":
            mask_idx = idx          # zero everything BUT the top-k
        else:
            raise ValueError(f"direction must be 'comp' or 'suff', got {direction!r}")

        perturbed = expr.clone()
        mask_t = torch.tensor(list(mask_idx), device=expr.device, dtype=torch.long)
        perturbed[:, mask_t] = baseline_value
        y = forward_fn(perturbed).detach().cpu().numpy()
        deltas.append(float(np.mean(np.abs(y - base))))
        fracs_used.append(float(f))

    # Simple AUC with the trapezoid rule over fractions
    auc = float(np.trapz(deltas, fracs_used)) if len(deltas) >= 2 else float("nan")
    return {
        "fractions": fracs_used,
        "deltas":    deltas,
        "auc":       auc,
        "direction": direction,
    }


# ---
# Method-agreement / stability helpers (operate on already-collected scores)
# ---

def topk_jaccard(a: np.ndarray, b: np.ndarray, k: int) -> float:
    """Jaccard overlap of the top-k indices of two attribution arrays."""
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    k = min(k, a.shape[0])
    ai = set(np.argsort(a)[::-1][:k].tolist())
    bi = set(np.argsort(b)[::-1][:k].tolist())
    union = ai | bi
    return float(len(ai & bi) / len(union)) if union else float("nan")


def spearman_agreement(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation between two attribution vectors."""
    from scipy.stats import spearmanr
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    rho, _ = spearmanr(a, b)
    return float(rho) if rho == rho else float("nan")
