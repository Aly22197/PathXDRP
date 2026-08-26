"""
Run-to-run variance of attention faithfulness.

Found while verifying the ablation: two runs of the SAME configuration and the
SAME seed reproduce prediction accuracy to four decimal places but differ in
attention faithfulness by 26%.

    published full model   PCC 0.93089   comprehensiveness 0.6033
    variant F (fresh)      PCC 0.93078   comprehensiveness 0.4479

The only differences between the two are `--norm foldwise` (which the leakage
diagnostic shows is a no-op on the random split, and which the PCC agreement
confirms empirically) and `--pool_mode mean` (equivalent to `auto` when the
residual is on). They stopped at different epochs (48 vs 46) under cuDNN
non-determinism, so they are different checkpoints of the same recipe.

This matters because attention faithfulness is the paper's central metric, and
the headline comparison -- PathXDRP 0.603 against DRPreter 0.407 -- is a
single run of each. If run-to-run variance is of the same order as that gap, the
comparison is not established.

This script separates the two variance sources:
  * WITHIN-run: bootstrap over the 237 benchmark drugs, holding the model fixed.
  * BETWEEN-run: the observed spread across checkpoints of the same recipe.

Usage:
    python revision/scripts/faithfulness_variance.py
Outputs:
    outputs/faithfulness_variance.md
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parents[1] / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

RUNS = {
    "PathXDRP (published)": "xai_multimodel_pathxdrp.json",
    "PathXDRP (variant F, fresh)": "xai_multimodel_pathxdrp_abF.json",
    "DRPreter (published)": "xai_multimodel_drpreter.json",
    "variant A (baseline)": "xai_multimodel_pathxdrp_abA.json",
    "variant A' (pooling only)": "xai_multimodel_pathxdrp_abAp.json",
}
KEY = "attn_faithfulness_comp"


def per_drug(fname: str) -> np.ndarray:
    f = ROOT / "results" / "xai" / fname
    if not f.exists():
        return np.array([])
    pd_ = json.loads(f.read_text()).get("per_drug", [])
    return np.array([r[KEY] for r in pd_ if r.get(KEY) is not None], dtype=float)


def boot(v: np.ndarray, n=4000, seed=0):
    v = v[~np.isnan(v)]
    if len(v) < 5:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    m = np.array([v[rng.integers(0, len(v), len(v))].mean() for _ in range(n)])
    return float(v.mean()), *np.percentile(m, [2.5, 97.5])


def main() -> None:
    rows = []
    for label, fname in RUNS.items():
        v = per_drug(fname)
        if len(v) == 0:
            continue
        mu, lo, hi = boot(v)
        rows.append((label, len(v), mu, lo, hi, float(np.std(v, ddof=1))))

    L = ["# Run-to-run variance of attention faithfulness\n",
         "Discovered while verifying the ablation, and material to the paper's "
         "central claim.\n",
         "## The observation\n",
         "Two runs of the same configuration and the same seed:\n",
         "| | PCC | faithfulness (comp.) |",
         "|---|---|---|",
         "| published full model | 0.93089 | 0.6033 |",
         "| variant F, fresh run | 0.93078 | 0.4479 |",
         "| **difference** | **0.0001** | **0.155 (26%)** |\n",
         "Prediction accuracy reproduces to four decimal places. Faithfulness "
         "does not. The two runs stopped at different epochs (48 vs 46) under "
         "cuDNN non-determinism, so they are different checkpoints of one "
         "recipe.\n",
         "## Within-run uncertainty, for comparison\n",
         "Bootstrap over the benchmark drugs, model held fixed:\n",
         "| Run | drugs | mean | 95% CI (over drugs) | SD across drugs |",
         "|---|---|---|---|---|"]
    for label, n, mu, lo, hi, sd in rows:
        L.append(f"| {label} | {n} | {mu:.4f} | [{lo:.4f}, {hi:.4f}] | {sd:.4f} |")
    L.append("")

    d = {r[0]: r for r in rows}
    pub = d.get("PathXDRP (published)")
    fresh = d.get("PathXDRP (variant F, fresh)")
    drp = d.get("DRPreter (published)")

    L.append("## What this means for the headline comparison\n")
    if pub and fresh and drp:
        overlap = not (pub[3] > fresh[4] or fresh[3] > pub[4])
        L.append(
            f"The paper's headline is PathXDRP {pub[2]:.3f} against DRPreter "
            f"{drp[2]:.3f}, a gap of {pub[2]-drp[2]:.3f}. The two PathXDRP runs "
            f"of the same recipe differ by {abs(pub[2]-fresh[2]):.3f} --- "
            f"{'comparable to' if abs(pub[2]-fresh[2]) > 0.6*abs(pub[2]-drp[2]) else 'smaller than'}"
            f" that gap. The fresh run ({fresh[2]:.3f}) still exceeds DRPreter "
            f"({drp[2]:.3f}), but by {fresh[2]-drp[2]:.3f} rather than "
            f"{pub[2]-drp[2]:.3f}.\n")
        L.append(
            "The within-run bootstrap intervals "
            + ("overlap" if overlap else "do not overlap")
            + ", so drug sampling does not explain the discrepancy. It is "
              "genuine between-checkpoint variance.\n")

    L.append("## Consequence, and what we do about it\n")
    L.append(
        "**The single-run comparison of faithfulness between architectures is "
        "not sound, and we should not have reported it as a point estimate.** "
        "Faithfulness depends on which checkpoint early stopping happens to "
        "select, and that varies between runs that are indistinguishable on "
        "accuracy.\n\n"
        "Three consequences:\n\n"
        "1. The manuscript reports faithfulness with a run-to-run spread, not "
        "as a single number, and states the number of runs behind it.\n"
        "2. The PathXDRP-versus-DRPreter faithfulness comparison is softened. "
        "Both observed PathXDRP values exceed DRPreter's, so the direction "
        "holds, but the magnitude is not established from one run each.\n"
        "3. The ablation conclusion is unaffected in direction. Variants A and "
        "A' sit near 0.02--0.03 while corrected models sit at 0.45--0.60. A "
        "gap of 15-20x is not closed by a between-run spread of 0.15, and the "
        "A-versus-A' finding rests on that gap rather than on a precise "
        "value.\n\n"
        "The honest framing is that the head redesign moves faithfulness by an "
        "order of magnitude, which is robust, while the residual differences "
        "between already-corrected architectures are within run-to-run noise "
        "at one run each.\n")
    (OUT / "faithfulness_variance.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
