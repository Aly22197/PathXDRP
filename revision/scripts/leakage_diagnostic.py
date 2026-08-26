"""
W3 -- Quantify the normalisation leakage that Reviewer #3 (point 3) identified.

The submitted pipeline Z-scored every gene over all 697 cell lines BEFORE the
split was applied. For a split that holds out cell lines, the moments used to
transform the test data were partly estimated from those very test cell lines.

This script measures, per split and seed:
  (a) how many test cell lines are absent from the training partition -- the
      only channel through which the moments can leak;
  (b) the distance between the cohort-wide moments and the correct fold-wise
      moments, per gene;
  (c) the resulting per-sample distortion of the model input.

It needs no GPU and no retraining, and it tells us which splits the re-run can
actually change.

Usage:
    python revision/scripts/leakage_diagnostic.py
Outputs:
    outputs/leakage_diagnostic.csv
    outputs/leakage_diagnostic.md
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parents[1] / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

SPLITS = ["random", "cell_blind", "drug_blind", "scaffold_blind", "tissue_blind"]
SEEDS = [0, 1, 2, 3, 4]
PRETTY = {"random": "random", "cell_blind": "cell-blind", "drug_blind": "drug-blind",
          "scaffold_blind": "scaffold-blind", "tissue_blind": "tissue-blind"}


def main() -> None:
    import sys
    sys.path.insert(0, str(ROOT))
    from pathxdrp.data.loader import build_master_df, fit_gene_stats
    from pathxdrp.data.splits import load_split

    print("Loading raw expression + master df ...")
    df, expr = build_master_df(require_smiles=True, standardize=False)

    # cohort-wide moments = what the submitted pipeline used
    coh_mean = expr.mean(axis=0)
    coh_std = expr.std(axis=0)
    coh_std[coh_std == 0] = 1.0

    rows = []
    for split in SPLITS:
        for seed in SEEDS:
            try:
                tr, va, te = load_split(split, seed, 0)
            except Exception as exc:
                print(f"  [skip] {split}/seed{seed}: {exc}")
                continue
            tr_cells = set(df.iloc[tr]["COSMIC_ID"].astype(int))
            te_cells = set(df.iloc[te]["COSMIC_ID"].astype(int))
            unseen = te_cells - tr_cells

            fm, fs = fit_gene_stats(expr, tr_cells)

            # Distortion of the test inputs: difference between the value the
            # model saw under cohort normalisation and under correct
            # fold-wise normalisation, in units of the fold-wise sigma.
            te_expr = expr.loc[sorted(te_cells)]
            z_cohort = (te_expr - coh_mean) / coh_std
            z_fold = (te_expr - fm) / fs
            delta = (z_cohort - z_fold).to_numpy(dtype="float32")

            rows.append({
                "split": split, "seed": seed,
                "n_test_cells": len(te_cells),
                "n_unseen_test_cells": len(unseen),
                "pct_unseen": 100.0 * len(unseen) / max(len(te_cells), 1),
                "n_train_cells": len(tr_cells),
                "mean_abs_delta_z": float(np.abs(delta).mean()),
                "p99_abs_delta_z": float(np.percentile(np.abs(delta), 99)),
                "max_abs_delta_z": float(np.abs(delta).max()),
                "mean_shift_per_gene": float(np.abs(coh_mean - fm).mean()),
                "std_ratio_med": float(np.median(coh_std / fs)),
            })
            print(f"  {split:16s} seed{seed}  unseen={len(unseen):4d}/{len(te_cells):4d} "
                  f"({rows[-1]['pct_unseen']:5.1f}%)  mean|dz|={rows[-1]['mean_abs_delta_z']:.4f}")

    out = pd.DataFrame(rows)
    out.to_csv(OUT / "leakage_diagnostic.csv", index=False)

    # ---------------- report ----------------
    g = out.groupby("split").agg(
        n_test_cells=("n_test_cells", "mean"),
        n_unseen=("n_unseen_test_cells", "mean"),
        pct_unseen=("pct_unseen", "mean"),
        mean_abs_dz=("mean_abs_delta_z", "mean"),
        p99_abs_dz=("p99_abs_delta_z", "mean"),
    ).reindex(SPLITS)

    L = ["# W3 -- Normalisation leakage diagnostic\n",
         "Answers Reviewer #3, point 3.\n",
         "The submitted pipeline fitted per-gene Z-score moments over all 697 cell",
         "lines before splitting. Leakage is possible only for test cell lines that",
         "do not also appear in the training partition -- for every other test row",
         "the moments were legitimately estimable from training data.\n",
         "## Where leakage can occur\n",
         "| Split | Test cell lines | Unseen in train | % unseen | mean abs delta-z on test inputs | p99 abs delta-z |",
         "|---|---|---|---|---|---|"]
    for s in SPLITS:
        if s not in g.index:
            continue
        r = g.loc[s]
        L.append(f"| {PRETTY[s]} | {r.n_test_cells:.0f} | {r.n_unseen:.0f} | "
                 f"{r.pct_unseen:.1f}% | {r.mean_abs_dz:.4f} | {r.p99_abs_dz:.4f} |")
    L.append("")

    affected = [s for s in SPLITS if s in g.index and g.loc[s].pct_unseen > 1]
    unaffected = [s for s in SPLITS if s in g.index and g.loc[s].pct_unseen <= 1]

    L.append("## Interpretation\n")
    L.append(
        f"**Splits with a genuine leakage channel:** {', '.join(PRETTY[s] for s in affected) or 'none'}.\n"
    )
    L.append(
        f"**Splits with no leakage channel:** {', '.join(PRETTY[s] for s in unaffected) or 'none'}. "
        "These hold out drugs or scaffolds, not cell lines, so every test cell "
        "line also appears in training and the fold-wise moments are fitted on "
        "(essentially) the same cohort as the cohort-wide moments.\n"
    )
    L.append(
        "This does not make the original pipeline correct -- the fix is applied to "
        "all five splits and all four models, so the comparison stays uniform. It "
        "does tell us where the numbers can move, and it is the honest quantitative "
        "answer to the reviewer's question.\n"
    )
    (OUT / "leakage_diagnostic.md").write_text("\n".join(L), encoding="utf-8")
    print(f"\nwrote {OUT/'leakage_diagnostic.md'}")
    print(g.to_string())


if __name__ == "__main__":
    main()
