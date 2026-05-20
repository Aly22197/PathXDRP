"""
All evaluation metrics used in the paper.

Regression:  RMSE, MAE, Pearson r, Spearman ρ, R²
Calibration: ECE, risk-coverage curve, OOD AUROC
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy import stats
from scipy.stats import ConstantInputWarning, NearConstantInputWarning
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score

# Per-drug / per-cell PCC handles the constant-input case explicitly via NaN
# filtering. The warning is therefore noise, not a bug indicator.
warnings.filterwarnings("ignore", category=ConstantInputWarning)
warnings.filterwarnings("ignore", category=NearConstantInputWarning)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    r, _ = stats.pearsonr(y_true, y_pred)
    return float(r)


def spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    rho, _ = stats.spearmanr(y_true, y_pred)
    return float(rho)


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(r2_score(y_true, y_pred))


def per_drug_pearson(y_true: np.ndarray, y_pred: np.ndarray, drug_ids: np.ndarray) -> float:
    """Average Pearson r across drugs (must have ≥ 2 samples per drug)."""
    unique = np.unique(drug_ids)
    rs = []
    for d in unique:
        mask = drug_ids == d
        if mask.sum() < 2:
            continue
        r, _ = stats.pearsonr(y_true[mask], y_pred[mask])
        if np.isnan(r):
            continue
        rs.append(r)
    return float(np.mean(rs)) if rs else float("nan")


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def per_cell_pearson(y_true: np.ndarray, y_pred: np.ndarray, cell_ids: np.ndarray) -> float:
    """Average Pearson r across cell lines (≥ 2 samples per cell required)."""
    unique = np.unique(cell_ids)
    rs = []
    for c in unique:
        mask = cell_ids == c
        if mask.sum() < 2:
            continue
        r, _ = stats.pearsonr(y_true[mask], y_pred[mask])
        if np.isnan(r):
            continue
        rs.append(r)
    return float(np.mean(rs)) if rs else float("nan")


def regression_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    drug_ids: np.ndarray | None = None,
    cell_ids: np.ndarray | None = None,
    uncertainties: np.ndarray | None = None,
) -> dict:
    """
    Full regression + calibration report. When `uncertainties` is provided
    (e.g. epistemic variance from the evidential head), this also returns
    ECE, risk-coverage curve, and selective RMSE at common coverage levels.
    """
    out: dict = {
        "RMSE":     rmse(y_true, y_pred),
        "MAE":      mae(y_true, y_pred),
        "PCC":      pearson(y_true, y_pred),
        "Spearman": spearman(y_true, y_pred),
        "R2":       r2(y_true, y_pred),
    }
    if drug_ids is not None:
        out["Per-drug PCC"] = per_drug_pearson(y_true, y_pred, drug_ids)
    if cell_ids is not None:
        out["Per-cell PCC"] = per_cell_pearson(y_true, y_pred, cell_ids)

    if uncertainties is not None and len(uncertainties) == len(y_true):
        # Calibration
        try:
            out["ECE"] = expected_calibration_error(y_true, y_pred, uncertainties)
        except Exception:
            out["ECE"] = float("nan")
        # Risk-coverage curve, stored as parallel arrays for downstream plotting
        try:
            cov, risk = risk_coverage_curve(y_true, y_pred, uncertainties)
            out["risk_coverage"] = {
                "coverages": cov.tolist(),
                "rmses":     risk.tolist(),
            }
            # Pinned points
            for cv in (0.5, 0.7, 0.9, 1.0):
                out[f"sel_RMSE@{int(cv*100)}"] = selective_rmse_at_coverage(
                    y_true, y_pred, uncertainties, coverage=cv,
                )
        except Exception:
            pass

    return out


# ---
# Calibration / uncertainty metrics
# ---

def expected_calibration_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    uncertainties: np.ndarray,
    n_bins: int = 15,
) -> float:
    """
    ECE adapted for regression:
    Sort by predicted uncertainty. In each bin, compare empirical error vs. predicted σ.
    ECE = mean |empirical_rmse(bin) - mean_sigma(bin)|.
    """
    order = np.argsort(uncertainties)
    bins = np.array_split(order, n_bins)
    ece = 0.0
    for b in bins:
        if len(b) == 0:
            continue
        empirical = np.sqrt(np.mean((y_true[b] - y_pred[b]) ** 2))
        predicted = np.sqrt(np.mean(uncertainties[b]))
        ece += abs(empirical - predicted) * (len(b) / len(y_true))
    return float(ece)


def risk_coverage_curve(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    uncertainties: np.ndarray,
    n_thresholds: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """
    For increasing confidence thresholds (lower uncertainty), compute:
    - Coverage: fraction of predictions retained.
    - Risk: RMSE on retained predictions.
    Returns (coverage_array, risk_array).
    """
    thresholds = np.percentile(uncertainties, np.linspace(0, 95, n_thresholds))
    coverages, risks = [], []
    for thr in thresholds:
        mask = uncertainties <= thr
        if mask.sum() < 2:
            continue
        coverages.append(mask.mean())
        risks.append(rmse(y_true[mask], y_pred[mask]))
    return np.array(coverages), np.array(risks)


def ood_auroc(
    in_dist_uncertainties: np.ndarray,
    ood_uncertainties: np.ndarray,
) -> float:
    """AUROC for detecting OOD samples using uncertainty as score."""
    labels = np.concatenate([
        np.zeros(len(in_dist_uncertainties)),
        np.ones(len(ood_uncertainties)),
    ])
    scores = np.concatenate([in_dist_uncertainties, ood_uncertainties])
    return float(roc_auc_score(labels, scores))


def selective_rmse_at_coverage(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    uncertainties: np.ndarray,
    coverage: float = 0.8,
) -> float:
    """RMSE on the top-`coverage` fraction by confidence (lowest uncertainty)."""
    n = max(1, int(len(y_true) * coverage))
    order = np.argsort(uncertainties)[:n]
    return rmse(y_true[order], y_pred[order])
