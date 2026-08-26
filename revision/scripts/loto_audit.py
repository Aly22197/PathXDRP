"""
W3b -- Leave-one-tissue-out is already built; only fold 0 was ever run.

Reviewer #5 (point 6) asked for a leave-one-tissue-out evaluation, noting that
the reported tissue-blind numbers are effectively a LUAD-only probe.

Inspecting `pathxdrp/data/splits.py:tissue_blind_split` shows it already emits
one fold PER TISSUE over the five most-represented tissues. Every run in the
submitted sweep used `--fold 0`, i.e. the single largest tissue. The
leave-one-tissue-out experiment the reviewer asks for is therefore folds 1-4 of
a split that already exists, not a new protocol.

This script documents which tissue each fold holds out and how large it is, so
the manuscript can describe the regime correctly and the remaining folds can be
queued.

Usage:
    python revision/scripts/loto_audit.py
Outputs:
    outputs/loto_folds.csv
    outputs/loto_audit.md
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parents[1] / "outputs"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))

N_FOLDS = 5


def main() -> None:
    from pathxdrp.data.loader import build_master_df
    from pathxdrp.data.splits import load_split

    df, _ = build_master_df(require_smiles=True, standardize=False)
    counts = df["tissue_2"].value_counts()
    top = counts.index[:N_FOLDS].tolist()

    rows = []
    for fold in range(N_FOLDS):
        try:
            tr, va, te = load_split("tissue_blind", 0, fold)
        except Exception as exc:
            print(f"  [skip] fold {fold}: {exc}")
            continue
        sub = df.iloc[te]
        held = sub["tissue_2"].fillna("unknown").value_counts()
        rows.append({
            "fold": fold,
            "held_out_tissue": top[fold] if fold < len(top) else "?",
            "test_rows": len(te),
            "val_rows": len(va),
            "train_rows": len(tr),
            "test_cell_lines": sub["COSMIC_ID"].nunique(),
            "test_drugs": sub["DRUG_ID"].nunique(),
            "purity": float(held.iloc[0] / len(sub)) if len(held) else np.nan,
        })
        print(f"  fold {fold}: {top[fold]:<12s} test={len(te):>6,} rows "
              f"({sub['COSMIC_ID'].nunique()} cell lines)")

    out = pd.DataFrame(rows)
    out.to_csv(OUT / "loto_folds.csv", index=False)

    L = ["# W3b -- Leave-one-tissue-out: the split already exists\n",
         "Answers Reviewer #5, point 6.\n",
         "## Finding\n",
         "`pathxdrp/data/splits.py:tissue_blind_split` builds **one fold per "
         "tissue** over the five most-represented tissues. Every run in the "
         "submitted sweep passed `--fold 0`. The reported `tissue_blind` column "
         "of Table 4 is therefore a single-tissue probe on the largest tissue, "
         "exactly as the reviewer suspected -- but the remaining folds are "
         "already generated and only need to be run.\n",
         "## The five folds\n",
         "| Fold | Held-out tissue | Test rows | Test cell lines | Train rows | Run in submission? |",
         "|---|---|---|---|---|---|"]
    for _, r in out.iterrows():
        used = "**yes**" if r.fold == 0 else "no"
        L.append(f"| {int(r.fold)} | {r.held_out_tissue} | {int(r.test_rows):,} | "
                 f"{int(r.test_cell_lines)} | {int(r.train_rows):,} | {used} |")
    L.append("")
    L.append("## Consequence\n")
    if len(out):
        L.append(
            f"The submitted tissue-blind number holds out "
            f"**{out.iloc[0].held_out_tissue}** only. A genuine "
            "leave-one-tissue-out result is the mean over folds 0--4, with the "
            "per-tissue spread reported as the uncertainty. That spread is the "
            "quantity the submitted error bars should have been, and it will be "
            "much larger than the initialisation noise they actually measured.\n\n"
            "Cost: 4 additional folds x 4 models. Because each fold holds out "
            "one tissue rather than five, individual runs are comparable in "
            "cost to the existing tissue-blind runs.\n"
        )
    L.append("## Wording fix for the manuscript\n")
    L.append(
        "The submitted text is self-contradictory on this point: "
        "Section 3.3 says the split \"holds out one of the top-five tissues by "
        "row count\" while the caption of the split-size table says it \"holds "
        "out the top-five tissues\". The code holds out one tissue per fold. "
        "The revised text says so, and reports which tissue.\n"
    )
    (OUT / "loto_audit.md").write_text("\n".join(L), encoding="utf-8")
    print(f"\nwrote {OUT/'loto_audit.md'}")


if __name__ == "__main__":
    main()
