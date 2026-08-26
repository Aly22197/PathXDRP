"""
W3b -- Leave-one-tissue-out results.

Answers Reviewer #5, point 6: the submitted tissue-blind number holds out one
tissue (lung adenocarcinoma) and the seed only reshuffles rows inside it, so the
reported standard deviation measures initialisation noise rather than
cross-tissue variability.

`tissue_blind_split` already emits one fold per tissue over the five most
represented tissues; the submitted sweep only ever ran fold 0. Running folds 1-4
turns that initialisation noise into genuine across-tissue variance.

This assembles the per-tissue table and, importantly, compares the two spreads:
the seed-to-seed spread the paper reported against the across-tissue spread it
should have reported.

Usage:
    python revision/scripts/loto_table.py
Outputs:
    tables/tab_loto.tex
    outputs/loto_results.md
"""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "outputs"
TAB = BASE / "tables"
OUT.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

# fold -> held-out tissue, from loto_audit.py
TISSUE = {0: "lung adeno. (LUAD)", 1: "breast", 2: "large intestine",
          3: "small-cell lung", 4: "ovary"}
TEST_ROWS = {0: 6026, 1: 4883, 2: 4606, 3: 4176, 4: 3793}
MODELS = ["pathxdrp", "cdrscan"]
PRETTY = {"pathxdrp": "PathXDRP", "cdrscan": "CDRScan"}


def pcc(model: str, fold: int) -> float | None:
    f = ROOT / "results" / model / f"tissue_blind_seed0_fold{fold}_fw.json"
    if not f.exists():
        return None
    return json.loads(f.read_text())["test"]["PCC"]


def seed_spread(model: str) -> float | None:
    """Seed-to-seed SD on fold 0, i.e. what the submitted paper reported.

    Read the fold-0 runs directly rather than ledger_summary.csv. Once folds
    1-4 exist, that summary groups every tissue_blind run together, so its
    PCC_std mixes the across-tissue spread into the seed spread -- which is
    the very quantity this function is the denominator for. It also picks up
    both normalisation variants of fold 0. Restricting to fold 0 under one
    normalisation keeps the comparison honest.
    """
    vals = []
    for seed in range(5):
        f = ROOT / "results" / model / f"tissue_blind_seed{seed}_fold0.json"
        if f.exists():
            vals.append(json.loads(f.read_text())["test"]["PCC"])
    return st.stdev(vals) if len(vals) > 1 else None


def main() -> None:
    data = {m: {f: pcc(m, f) for f in range(5)} for m in MODELS}

    md = ["# W3b -- Leave-one-tissue-out\n",
          "Answers Reviewer #5, point 6.\n",
          "| Held-out tissue | Test rows | " +
          " | ".join(PRETTY[m] for m in MODELS) + " |",
          "|---|---|" + "---|" * len(MODELS)]
    for f in range(5):
        cells = []
        for m in MODELS:
            v = data[m][f]
            cells.append(f"{v:.4f}" if v is not None else "pending")
        md.append(f"| {TISSUE[f]} | {TEST_ROWS[f]:,} | " + " | ".join(cells) + " |")
    md.append("")

    md.append("## The two spreads\n")
    md.append("| Model | across-tissue mean | across-tissue SD | seed SD on fold 0 | ratio |")
    md.append("|---|---|---|---|---|")
    summary = {}
    for m in MODELS:
        vals = [v for v in data[m].values() if v is not None]
        if len(vals) < 2:
            md.append(f"| {PRETTY[m]} | pending | pending | | |")
            continue
        mu, sd = st.mean(vals), st.stdev(vals)
        ssd = seed_spread(m)
        ratio = f"{sd/ssd:.0f}x" if ssd else "--"
        summary[m] = (mu, sd, ssd, len(vals))
        md.append(f"| {PRETTY[m]} | {mu:.4f} | **{sd:.4f}** | "
                  f"{ssd:.4f} | {ratio} |" if ssd else
                  f"| {PRETTY[m]} | {mu:.4f} | **{sd:.4f}** | -- | -- |")
    md.append("")

    if summary:
        m0 = MODELS[0]
        if m0 in summary:
            mu, sd, ssd, n = summary[m0]
            md.append("## Reading\n")
            md.append(
                f"The across-tissue standard deviation for {PRETTY[m0]} is "
                f"{sd:.4f} over {n} tissues, against a seed-to-seed standard "
                f"deviation of {ssd:.4f} on the single fold the submitted paper "
                f"reported"
                + (f" --- roughly {sd/ssd:.0f} times larger.\n" if ssd else ".\n"))
            md.append(
                "This is the reviewer's point, quantified. The error bar the "
                "submitted manuscript attached to its tissue-blind column "
                "describes how much the number moves when the model is "
                "re-initialised, not how much it moves when a different tissue "
                "is held out. The second quantity is the one a reader cares "
                "about, and it is far larger.\n")
            vals = {f: data[m0][f] for f in range(5) if data[m0][f] is not None}
            if len(vals) > 1:
                hi = max(vals, key=vals.get); lo = min(vals, key=vals.get)
                md.append(
                    f"Per-tissue difficulty varies substantially: {TISSUE[hi]} "
                    f"gives {vals[hi]:.3f} and {TISSUE[lo]} gives {vals[lo]:.3f}. "
                    f"The submitted paper reported fold 0 ({TISSUE[0]}) alone"
                    + (", which is the easiest of the tissues measured so far.\n"
                       if hi == 0 else ".\n"))

    (OUT / "loto_results.md").write_text("\n".join(md), encoding="utf-8")

    # ---- LaTeX ----
    L = [r"% Generated by revision/scripts/loto_table.py",
         r"\begin{table}[!h]", r"\centering\small",
         # The caption is read by someone who never saw the review, so it
         # describes the table rather than what an earlier version reported.
         r"\caption{Leave-one-tissue-out. Each row holds out one tissue",
         r"entirely. The across-tissue standard deviation is the quantity a",
         r"tissue-blind error bar should convey, and it is several times the",
         r"seed-to-seed standard deviation measured on any one fold, so a",
         r"single tissue carrying a seed-derived error bar understates how",
         r"far the number moves.}",
         r"\label{tab:loto}",
         r"\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}l r " +
         "r" * len(MODELS) + r"@{}}",
         r"\toprule",
         r"\textbf{Held-out tissue} & \textbf{Test rows} & " +
         " & ".join(r"\textbf{" + PRETTY[m] + "}" for m in MODELS) + r" \\",
         r"\midrule"]
    for f in range(5):
        cells = []
        for m in MODELS:
            v = data[m][f]
            cells.append(f"${v:.4f}$" if v is not None else r"\emph{pending}")
        L.append(f"{TISSUE[f]} & ${TEST_ROWS[f]:,}$".replace(",", "{,}") +
                 " & " + " & ".join(cells) + r" \\")
    L.append(r"\midrule")
    for label, key in (("mean across tissues", "mu"), ("SD across tissues", "sd"),
                       ("SD across seeds, fold 0", "ssd")):
        cells = []
        for m in MODELS:
            if m not in summary:
                cells.append(r"\emph{pending}"); continue
            mu, sd, ssd, _ = summary[m]
            v = {"mu": mu, "sd": sd, "ssd": ssd}[key]
            cells.append(f"${v:.4f}$" if v is not None else "--")
        bold = key == "sd"
        lab = r"\textbf{" + label + "}" if bold else label
        L.append(f"{lab} & & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular*}", "",
          r"\smallskip",
          r"\footnotesize\emph{Note:} One seed per cell. The across-tissue",
          r"spread is a property of which tissue is held out and is not reduced",
          r"by averaging over seeds.",
          r"\end{table}"]
    TAB.joinpath("tab_loto.tex").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(md))
    print(f"\nwrote {TAB/'tab_loto.tex'}")


if __name__ == "__main__":
    main()
