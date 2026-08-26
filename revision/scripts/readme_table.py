"""Rewrite the README's headline table from the canonical ledger.

The README and the manuscript disagreed for two releases because the table was
maintained by hand against a file nobody regenerated. This closes that loop:
the table lives between two HTML comment markers in README.md and is written
by this script from `revision/outputs/ledger_summary.csv`, which
`build_ledger.py` produces from `results/<model>/<split>_seed<S>_fold0.json`.

Run it after any change to results/:

    python revision/scripts/build_ledger.py
    python revision/scripts/readme_table.py

`--check` rewrites nothing and exits 1 if the README is out of date, which is
the form to put in CI or to run before tagging a release.

Usage:
    python revision/scripts/readme_table.py [--check]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
LEDGER = Path(__file__).resolve().parents[1] / "outputs" / "ledger_summary.csv"
README = ROOT / "README.md"

BEGIN = "<!-- BEGIN HEADLINE TABLE -->"
END = "<!-- END HEADLINE TABLE -->"

MODELS = ["pathxdrp", "drpreter", "graphdrp", "cdrscan"]
LABELS = {"pathxdrp": "PathXDRP", "drpreter": "DRPreter",
          "graphdrp": "GraphDRP", "cdrscan": "CDRScan"}
SPLITS = ["random", "cell_blind", "drug_blind", "scaffold_blind", "tissue_blind"]
SPLIT_LABELS = {"random": "random", "cell_blind": "cell-blind",
                "drug_blind": "drug-blind", "scaffold_blind": "scaffold-blind",
                "tissue_blind": "tissue-blind"}


def build_table(summ: pd.DataFrame) -> str:
    rows = ["| Split | " + " | ".join(LABELS[m] for m in MODELS) + " |",
            "| --- | " + " | ".join("---" for _ in MODELS) + " |"]
    for s in SPLITS:
        cells = []
        means = {}
        for m in MODELS:
            r = summ[(summ.model == m) & (summ.split == s)]
            means[m] = float(r.iloc[0]["PCC_mean"]) if len(r) else float("nan")
        best = max(means, key=lambda m: means[m])
        for m in MODELS:
            r = summ[(summ.model == m) & (summ.split == s)]
            if not len(r):
                cells.append("--")
                continue
            r = r.iloc[0]
            cell = f"{r['PCC_mean']:.3f} ± {r['PCC_std']:.3f}"
            cells.append(f"**{cell}**" if m == best else cell)
        rows.append(f"| {SPLIT_LABELS[s]} | " + " | ".join(cells) + " |")

    n = int(summ[summ.split == "random"].iloc[0]["n_seeds"])
    rows.append("")
    rows.append(
        f"<sub>Test Pearson correlation on GDSC2 LN-IC50, mean ± population "
        f"standard deviation over {n} seeds (fold 0). Bold marks the best model "
        f"per split. Generated from `revision/outputs/ledger_summary.csv` by "
        f"`revision/scripts/readme_table.py`; these are the values in Table 4 of "
        f"the manuscript.</sub>"
    )
    return "\n".join(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if README.md is out of date; write nothing")
    args = ap.parse_args()

    if not LEDGER.exists():
        print(f"missing {LEDGER}; run build_ledger.py first", file=sys.stderr)
        return 2

    table = build_table(pd.read_csv(LEDGER))
    text = README.read_text(encoding="utf-8")
    if BEGIN not in text or END not in text:
        print(f"markers {BEGIN} / {END} not found in README.md", file=sys.stderr)
        return 2

    head, rest = text.split(BEGIN, 1)
    _, tail = rest.split(END, 1)
    new = f"{head}{BEGIN}\n{table}\n{END}{tail}"

    if new == text:
        print("README headline table is up to date")
        return 0
    if args.check:
        print("README headline table is STALE; run without --check to rewrite",
              file=sys.stderr)
        return 1
    README.write_text(new, encoding="utf-8")
    print(f"rewrote the headline table in {README}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
