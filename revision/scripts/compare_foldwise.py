"""
W3 -- Compare cohort-wide against fold-wise expression normalisation.

Answers Reviewer #3, point 3. Run after `run_foldwise_sweep.py`.

Pairs each re-run (`_fw` tag) against the corresponding submitted run and
reports the change in every metric. The leakage diagnostic predicts that
cell-blind and tissue-blind can move and that random, drug-blind and
scaffold-blind cannot; this checks that prediction against the actual numbers
rather than asserting it.

Usage:
    python revision/scripts/compare_foldwise.py
Outputs:
    outputs/foldwise_comparison.csv
    outputs/foldwise_comparison.md
    tables/tab_foldwise_delta.tex
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

MODELS = ["pathxdrp", "drpreter", "graphdrp", "cdrscan", "deepcdr"]
SPLITS = ["random", "cell_blind", "drug_blind", "scaffold_blind", "tissue_blind"]
SEEDS = [0, 1, 2, 3, 4]
METRICS = ["PCC", "RMSE", "Spearman", "R2", "Per-drug PCC", "Per-cell PCC"]
PRETTY = {"random": "random", "cell_blind": "cell-blind", "drug_blind": "drug-blind",
          "scaffold_blind": "scaffold-blind", "tissue_blind": "tissue-blind"}
MODEL_PRETTY = {"pathxdrp": "PathXDRP", "drpreter": "DRPreter",
                "graphdrp": "GraphDRP", "cdrscan": "CDRScan",
                "deepcdr": "DeepCDR"}
# Splits where a test cell line can be absent from training, i.e. where the
# cohort-wide moments could leak. See outputs/leakage_diagnostic.md.
AFFECTED = {"cell_blind", "tissue_blind"}


def load(model: str, split: str, seed: int, tag: str = "") -> dict | None:
    suf = f"_{tag}" if tag else ""
    f = ROOT / "results" / model / f"{split}_seed{seed}_fold0{suf}.json"
    if not f.exists():
        return None
    return json.loads(f.read_text()).get("test", {})


def main() -> None:
    rows = []
    for m in MODELS:
        for sp in SPLITS:
            for sd in SEEDS:
                a = load(m, sp, sd)            # submitted: cohort-wide
                b = load(m, sp, sd, "fw")      # re-run: fold-wise
                if a is None or b is None:
                    continue
                r = {"model": m, "split": sp, "seed": sd}
                for k in METRICS:
                    if k in a and k in b:
                        r[f"{k}_cohort"] = a[k]
                        r[f"{k}_foldwise"] = b[k]
                        r[f"{k}_delta"] = b[k] - a[k]
                rows.append(r)

    if not rows:
        print("No paired runs yet. Run run_foldwise_sweep.py first.")
        (OUT / "foldwise_comparison.md").write_text(
            "# W3 -- fold-wise vs cohort-wide\n\n"
            "_No paired runs available yet; the re-run is still in progress._\n",
            encoding="utf-8")
        return

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "foldwise_comparison.csv", index=False)

    L = ["# W3 -- fold-wise vs cohort-wide normalisation\n",
         "Answers Reviewer #3, point 3.\n",
         f"Paired runs available: **{len(df)}** of "
         f"{len(MODELS)*len(SPLITS)*len(SEEDS)}.\n",
         "## Change in PCC (fold-wise minus cohort-wide)\n",
         "| Split | Leakage channel? | Model | cohort | fold-wise | delta |",
         "|---|---|---|---|---|---|"]
    for sp in SPLITS:
        for m in MODELS:
            g = df[(df.split == sp) & (df.model == m)]
            if g.empty or "PCC_delta" not in g:
                continue
            L.append(
                f"| {PRETTY[sp]} | {'**yes**' if sp in AFFECTED else 'no'} | "
                f"{MODEL_PRETTY[m]} | {g.PCC_cohort.mean():.4f} | "
                f"{g.PCC_foldwise.mean():.4f} | {g.PCC_delta.mean():+.4f} |")
    L.append("")

    # ---- disentangle the two corrections that both moved ECE ----
    # Two independent fixes landed in the same revision: fold-wise
    # normalisation, and using the total predictive variance rather than the
    # epistemic component alone for calibration. Reporting their combined
    # effect as if it were one would be misleading, so recompute ECE from the
    # saved predictions holding one factor fixed at a time.
    ecerows = []
    for m in ["pathxdrp"]:
        for sp in SPLITS:
            for sd in SEEDS:
                fa = ROOT / "results" / m / f"{sp}_seed{sd}_fold0_preds.csv"
                fb = ROOT / "results" / m / f"{sp}_seed{sd}_fold0_fw_preds.csv"
                if not (fa.exists() and fb.exists()):
                    continue
                import numpy as np

                def _ece(path):
                    d = pd.read_csv(path)
                    if "epistemic" not in d.columns:
                        return None, None
                    y = d.y_true.to_numpy(float); p = d.y_pred.to_numpy(float)
                    epi = d.epistemic.to_numpy(float)
                    tot = epi + d.aleatoric.to_numpy(float)

                    def e(v):
                        o = np.argsort(v); acc = 0.0
                        for b in np.array_split(o, 15):
                            if len(b) == 0:
                                continue
                            acc += abs(np.sqrt(np.mean((y[b] - p[b]) ** 2))
                                       - np.sqrt(np.mean(v[b]))) * len(b) / len(y)
                        return float(acc)
                    return e(epi), e(tot)

                ea_epi, ea_tot = _ece(fa)
                eb_epi, eb_tot = _ece(fb)
                if ea_epi is None or eb_epi is None:
                    continue
                ecerows.append({
                    "split": sp, "seed": sd,
                    "cohort_epistemic": ea_epi, "cohort_total": ea_tot,
                    "foldwise_epistemic": eb_epi, "foldwise_total": eb_tot,
                    "d_definition": ea_tot - ea_epi,     # holding norm fixed
                    "d_normalisation": eb_tot - ea_tot,  # holding definition fixed
                })
    if ecerows:
        ed = pd.DataFrame(ecerows)
        ed.to_csv(OUT / "ece_attribution.csv", index=False)
        L.append("## Attributing the ECE change to the right fix\n")
        L.append(
            "Two independent corrections landed together: fold-wise "
            "normalisation, and using the total predictive variance instead of "
            "the epistemic component alone (the latter is what our own Methods "
            "section always specified). Recomputing from the saved predictions "
            "separates them.\n")
        L.append("| Split | ECE published | + definition fix | + fold-wise norm | "
                 "due to definition | due to normalisation |")
        L.append("|---|---|---|---|---|---|")
        for sp in SPLITS:
            g = ed[ed.split == sp]
            if g.empty:
                continue
            L.append(f"| {PRETTY[sp]} | {g.cohort_epistemic.mean():.4f} | "
                     f"{g.cohort_total.mean():.4f} | {g.foldwise_total.mean():.4f} | "
                     f"{g.d_definition.mean():+.4f} | {g.d_normalisation.mean():+.4f} |")
        L.append("")

    if "PCC_delta" in df:
        aff = df[df.split.isin(AFFECTED)].PCC_delta
        un = df[~df.split.isin(AFFECTED)].PCC_delta
        L.append("## Was the diagnostic right?\n")
        if len(un):
            L.append(f"- Splits with **no** leakage channel: mean |delta PCC| = "
                     f"{un.abs().mean():.5f}, max {un.abs().max():.5f}.")
        if len(aff):
            L.append(f"- Splits **with** a leakage channel: mean |delta PCC| = "
                     f"{aff.abs().mean():.5f}, max {aff.abs().max():.5f}.")
        L.append("")
        if len(aff) and len(un):
            L.append(
                "The prediction from `leakage_diagnostic.md` was that only "
                "cell-blind and tissue-blind can move, because only they hold "
                "out cell lines. "
                + ("That is what happened."
                   if aff.abs().mean() > un.abs().mean()
                   else "The measured pattern does not match the prediction, "
                        "which needs explaining before the claim is made.")
                + " Note that runs also differ by initialisation noise, so a "
                  "small non-zero delta on an unaffected split is expected.\n")

    (OUT / "foldwise_comparison.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L[:30]))
    print(f"\nwrote {OUT/'foldwise_comparison.md'}")


if __name__ == "__main__":
    main()
