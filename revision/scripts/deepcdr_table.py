"""
DeepCDR baseline results (Reviewer #3 point 4, Reviewer #5 point 7).

DeepCDR was added to answer the request for a more recent baseline. It runs
through the same loaders, split files, fold-wise normalisation and metric code
as the other models, so differences are attributable to architecture.

Two caveats belong with the table and are printed into its caption rather than
left to the reader:

  * it is the expression-only configuration, because this benchmark carries no
    mutation or methylation channel, so it will score below the figure in the
    original paper and is not a reproduction of it;
  * it is one seed per split, against five seeds for the models in Table 4, so
    it carries no standard deviation and is not directly comparable at the
    resolution of a seed error bar.

Usage:
    python revision/scripts/deepcdr_table.py
Outputs:
    tables/tab_deepcdr.tex
    outputs/deepcdr_results.md
"""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "outputs"
TAB = BASE / "tables"
OUT.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

SPLITS = [("random", "random"), ("cell_blind", "cell-blind"),
          ("drug_blind", "drug-blind"), ("scaffold_blind", "scaffold-blind"),
          ("tissue_blind", "tissue-blind")]
METRICS = [("PCC", "PCC"), ("RMSE", "RMSE"), ("Spearman", "Spearman")]
PEERS = ["pathxdrp", "graphdrp", "drpreter", "cdrscan"]


def deepcdr(split: str) -> dict | None:
    f = ROOT / "results" / "deepcdr" / f"{split}_seed0_fold0_fw.json"
    if not f.exists():
        return None
    return json.loads(f.read_text())["test"]


def peer_range(split: str, metric: str) -> tuple[float, float] | None:
    """Min and max of the five-seed means for the four established models."""
    means = []
    for m in PEERS:
        vals = []
        for seed in range(5):
            f = ROOT / "results" / m / f"{split}_seed{seed}_fold0.json"
            if f.exists():
                vals.append(json.loads(f.read_text())["test"][metric])
        if vals:
            means.append(st.mean(vals))
    return (min(means), max(means)) if means else None


def main() -> None:
    rows = []
    for key, pretty in SPLITS:
        t = deepcdr(key)
        if t is None:
            continue
        rng = peer_range(key, "PCC")
        rows.append((pretty, t["PCC"], t["RMSE"], t["Spearman"], rng))

    if not rows:
        print("no DeepCDR results found")
        return

    # ---------------------------------------------------------------- LaTeX
    L = [r"\begin{table}[!ht]", r"\centering\small",
         r"\caption{DeepCDR on the five split regimes. Expression-only "
         r"configuration, "
         r"since this benchmark carries no mutation or methylation channel, so "
         r"these numbers are below those in the original paper and are not a "
         r"reproduction of them. Single seed under fold-wise normalisation, "
         r"against five seeds for the models in Table~\ref{tab:pred}; no "
         r"standard deviation is therefore quoted. The final column gives the "
         r"range of five-seed mean PCC across GraphDRP, DRPreter, CDRScan and "
         r"PathXDRP on the same split, for placement rather than for a "
         r"significance claim.}",
         r"\label{tab:deepcdr}",
         r"\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}lrrrc@{}}",
         r"\toprule",
         r"\textbf{Split} & \textbf{PCC} & \textbf{RMSE} & "
         r"\textbf{Spearman} & \textbf{Others (PCC)} \\",
         r"\midrule"]
    for pretty, pcc, rmse, sp, rng in rows:
        band = f"{rng[0]:.3f}--{rng[1]:.3f}" if rng else "--"
        L.append(f"{pretty} & {pcc:.4f} & {rmse:.4f} & {sp:.4f} & {band} \\\\")
    L += [r"\bottomrule", r"\end{tabular*}", r"\end{table}", ""]
    (TAB / "tab_deepcdr.tex").write_text("\n".join(L), encoding="utf-8")

    # ------------------------------------------------------------- markdown
    M = ["# DeepCDR baseline\n",
         "Answers Reviewer #3 point 4 and Reviewer #5 point 7.\n",
         "Expression-only configuration, single seed, fold-wise normalisation.",
         "The comparison band is the range of five-seed mean PCC over the four "
         "established models on the same split.\n",
         "| Split | PCC | RMSE | Spearman | others (PCC range) |",
         "|---|---|---|---|---|"]
    for pretty, pcc, rmse, sp, rng in rows:
        band = f"{rng[0]:.3f}-{rng[1]:.3f}" if rng else "--"
        M.append(f"| {pretty} | {pcc:.4f} | {rmse:.4f} | {sp:.4f} | {band} |")

    inside = [p for p, pcc, _, _, r in rows if r and r[0] <= pcc <= r[1]]
    below = [p for p, pcc, _, _, r in rows if r and pcc < r[0]]
    above = [p for p, pcc, _, _, r in rows if r and pcc > r[1]]
    M += ["", "## Reading", "",
          f"DeepCDR lands inside the band of the four established models on "
          f"{len(inside)} of {len(rows)} splits"
          + (f" ({', '.join(inside)})" if inside else "")
          + (f", below all of them on {', '.join(below)}" if below else "")
          + (f", and above all of them on {', '.join(above)}" if above else "")
          + ".",
          "",
          "This is the expected place for a single-omics configuration of a "
          "multi-omics model, and it does not change the finding that these "
          "architectures cannot be separated on prediction accuracy."]
    (OUT / "deepcdr_results.md").write_text("\n".join(M) + "\n", encoding="utf-8")

    print("\n".join(M))
    print(f"\nwrote {TAB / 'tab_deepcdr.tex'}")


if __name__ == "__main__":
    main()
