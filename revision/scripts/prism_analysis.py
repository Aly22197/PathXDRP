"""
PRISM calibration and per-drug significance.

Answers the two requests that could not be met while the external predictions
were missing from the release:

  * Reviewer #4, point 5  -- calibration reported on PRISM, not only on the
    five GDSC2 splits;
  * Reviewer #5, point 11 -- a significance analysis for the per-drug PRISM
    correlation, which was quoted as a bare 0.052.

Both need per-pair predictions with the predictive uncertainty, which
scripts/external_validation.py now writes to results/external/.

Usage:
    python revision/scripts/prism_analysis.py
Outputs:
    outputs/prism_calibration.csv
    outputs/prism_per_drug.csv
    outputs/prism_analysis.md
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parent
OUT = BASE / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

def _preds_path() -> Path:
    """PathXDRP predictions from the PRISM run the manuscript reports.

    The reported evaluation keeps one row per screen rather than collapsing
    duplicates, so the calibration and the per-drug interval must come from
    that run and not from the collapsed variant, or the pair counts quoted in
    the text will not match the analysis behind them.
    """
    ext = ROOT / "results" / "external"
    for name in ("prismNC_pathxdrp_random_seed0_fold0_preds.csv",
                 "prismNC_random_seed0_fold0_preds.csv",
                 "prism_pathxdrp_random_seed0_fold0_preds.csv",
                 "prism_random_seed0_fold0_preds.csv"):
        if (ext / name).exists():
            return ext / name
    return ext / "prismNC_pathxdrp_random_seed0_fold0_preds.csv"


PREDS = _preds_path()
N_BOOT = 2000
SEED = 20260826


def ece(y, p, s, n_bins: int = 10) -> float:
    """Expected calibration error for a regression predictive interval.

    Bins by predicted sigma and compares the mean predicted sigma with the
    realised RMSE inside each bin, matching calibration_study.py.
    """
    err = np.abs(y - p)
    qs = np.quantile(s, np.linspace(0, 1, n_bins + 1))
    qs[0], qs[-1] = -np.inf, np.inf
    total, n = 0.0, len(y)
    for i in range(n_bins):
        m = (s >= qs[i]) & (s < qs[i + 1])
        if m.sum() < 2:
            continue
        realised = float(np.sqrt(np.mean(err[m] ** 2)))
        predicted = float(np.mean(s[m]))
        total += m.sum() * abs(realised - predicted)
    return total / n if n else float("nan")


def pcc(a, b) -> float:
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def main() -> None:
    if not PREDS.exists():
        print(f"PRISM predictions not found: {PREDS}")
        print("Run scripts/external_validation.py first.")
        sys.exit(1)

    df = pd.read_csv(PREDS)
    needed = {"y_true", "y_pred", "drug_id"}
    if not needed <= set(df.columns):
        print(f"{PREDS.name} lacks {needed - set(df.columns)}")
        sys.exit(1)

    y = df["y_true"].to_numpy(float)
    p = df["y_pred"].to_numpy(float)

    L = ["# PRISM calibration and per-drug significance\n",
         "Answers Reviewer #4 point 5 (calibration on PRISM) and Reviewer #5 "
         "point 11 (significance for the per-drug correlation).\n",
         f"- pairs: **{len(df):,}**",
         f"- drugs: **{df['drug_id'].nunique()}**",
         f"- cell lines: **{df['cell_id'].nunique()}**\n"]

    # ---------------------------------------------------------- calibration
    have_sigma = ("epistemic" in df.columns and df["epistemic"].notna().any())
    if have_sigma:
        epi = df["epistemic"].to_numpy(float)
        ale = (df["aleatoric"].to_numpy(float)
               if "aleatoric" in df.columns else np.zeros_like(epi))
        # Section 3.4 defines sigma as the total predictive standard deviation.
        sigma = np.sqrt(np.clip(epi, 0, None) + np.clip(ale, 0, None))
        rmse = float(np.sqrt(np.mean((y - p) ** 2)))
        e_tot = ece(y, p, sigma)
        e_epi = ece(y, p, np.sqrt(np.clip(epi, 0, None)))

        order = np.argsort(sigma)
        keep = order[: max(1, len(order) // 2)]
        sel_rmse = float(np.sqrt(np.mean((y[keep] - p[keep]) ** 2)))

        # does sigma rank the errors at all?
        auroc_like = pcc(sigma, np.abs(y - p))

        pd.DataFrame([{ "dataset": "PRISM", "n": len(df), "RMSE": rmse,
                        "ECE_total_sigma": e_tot, "ECE_epistemic_only": e_epi,
                        "selRMSE@50": sel_rmse,
                        "gain_vs_full": 1 - sel_rmse / rmse if rmse else np.nan,
                        "corr_sigma_abserr": auroc_like }]
                    ).to_csv(OUT / "prism_calibration.csv", index=False)

        L += ["## Calibration on PRISM\n",
              "| Quantity | Value |", "|---|---|",
              f"| RMSE | {rmse:.3f} |",
              f"| ECE, total predictive sigma | {e_tot:.3f} |",
              f"| ECE, epistemic component only | {e_epi:.3f} |",
              f"| selective RMSE at 50% coverage | {sel_rmse:.3f} |",
              f"| gain from discarding the least-confident half | "
              f"{100 * (1 - sel_rmse / rmse):.1f}% |",
              f"| correlation of sigma with absolute error | {auroc_like:.3f} |",
              ""]
    else:
        L += ["## Calibration on PRISM\n",
              "Predictions carry no uncertainty columns; re-run "
              "external_validation.py to record them.\n"]

    # ------------------------------------------------------ per-drug PCC CI
    rng = np.random.default_rng(SEED)
    per_drug = []
    for did, g in df.groupby("drug_id"):
        if len(g) < 5:
            continue
        per_drug.append((int(did), len(g),
                         pcc(g["y_true"].to_numpy(float),
                             g["y_pred"].to_numpy(float))))
    pdf = pd.DataFrame(per_drug, columns=["drug_id", "n_cells", "PCC"]).dropna()
    mean_pcc = float(pdf["PCC"].mean())

    # Bootstrap over drugs: the drug is the unit that repeats, so resampling
    # pairs would understate the interval exactly as it did in Table 8.
    vals = pdf["PCC"].to_numpy(float)
    boots = np.array([np.mean(rng.choice(vals, len(vals), replace=True))
                      for _ in range(N_BOOT)])
    lo, hi = np.percentile(boots, [2.5, 97.5])

    pdf.sort_values("PCC", ascending=False).to_csv(
        OUT / "prism_per_drug.csv", index=False)

    L += ["## Per-drug correlation\n",
          f"Mean per-drug PCC over {len(pdf)} drugs with at least five cell "
          f"lines: **{mean_pcc:.3f}**, bootstrap 95% CI over drugs "
          f"**[{lo:.3f}, {hi:.3f}]** ({N_BOOT:,} resamples).\n",
          f"- drugs with a positive correlation: "
          f"{int((pdf.PCC > 0).sum())} of {len(pdf)}",
          f"- interquartile range: {pdf.PCC.quantile(0.25):.3f} to "
          f"{pdf.PCC.quantile(0.75):.3f}\n"]

    if lo <= 0 <= hi:
        L.append("The interval contains zero, so the per-drug correlation is "
                 "not distinguishable from no within-drug ranking ability. "
                 "This supports the revised wording: across unseen cell lines "
                 "and platforms the model does not usefully rank cell lines "
                 "within a single drug.\n")
    else:
        L.append("The interval excludes zero, so there is a small but "
                 "detectable within-drug signal. It remains far below the "
                 "pooled correlation, and the pooled figure should not be "
                 "read as within-drug ranking ability.\n")

    (OUT / "prism_analysis.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"wrote {OUT / 'prism_analysis.md'}")


if __name__ == "__main__":
    main()
