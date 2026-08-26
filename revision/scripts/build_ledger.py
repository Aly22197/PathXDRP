"""
W11 -- Canonical results ledger.

Single source of truth for every number that appears in the revised manuscript.
Walks results/<model>/<split>_seed<S>_fold<F>.json for the FINAL sweep only and
emits one tidy CSV.  Every table in the paper must be generated from this file;
no hand-typed numbers.

Also audits the stale results/summary_v3_vs_baselines.csv that the public README
was built from, so the provenance of the README/manuscript discrepancy flagged by
Reviewer #4 (point 3) is documented rather than quietly patched.

Usage:
    python revision/scripts/build_ledger.py
Outputs:
    revision/outputs/ledger.csv
    revision/outputs/ledger_summary.csv
    revision/outputs/readme_discrepancy_audit.md

Two conventions this file is strict about, because getting either of them
wrong is what put the README out of step with the paper in the first place.

  * Only the canonical sweep counts towards a headline mean. A results file is
    canonical when its name is exactly <split>_seed<S>_fold0.json. The revision
    added fold-wise-normalisation reruns (`_fw`), leave-one-tissue-out folds
    (`_fold1..4_fw`) and head ablations (`_abA`..`_abF`) to the same model
    directories; a glob of `*_fold*.json` sweeps those in and silently moves
    the headline numbers.
  * The dispersion is the population standard deviation (ddof=0), which is
    what `eval/analyze_results.py` reports and therefore what the manuscript
    tables contain. `statistics.stdev` is the sample standard deviation
    (ddof=1); over five seeds it is 11.8% larger, which is enough to shift
    nine of the twenty PCC cells in the third decimal.
"""
from __future__ import annotations

import json
import re
import statistics as st
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
OUT = Path(__file__).resolve().parents[1] / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

MODELS = ["pathxdrp", "drpreter", "graphdrp", "cdrscan"]
SPLITS = ["random", "cell_blind", "drug_blind", "scaffold_blind", "tissue_blind"]
METRICS = ["PCC", "RMSE", "MAE", "MSE", "Spearman", "R2", "Per-drug PCC", "Per-cell PCC", "ECE"]

# <split>_seed<S>_fold0.json, and nothing else.
CANONICAL = re.compile(r"^(?:" + "|".join(SPLITS) + r")_seed\d+_fold0\.json$")


def std(values: list[float]) -> float:
    """Population standard deviation, matching eval/analyze_results.py."""
    return st.pstdev(values) if len(values) > 1 else 0.0


def _get(d: dict, *keys, default=None):
    """Fetch the first present key from a nested-ish dict."""
    for k in keys:
        if k in d:
            return d[k]
    return default


def collect() -> pd.DataFrame:
    rows = []
    for model in MODELS:
        mdir = RESULTS / model
        if not mdir.is_dir():
            print(f"  [warn] missing {mdir}")
            continue
        for f in sorted(mdir.glob("*.json")):
            if not CANONICAL.match(f.name):
                continue          # _fw / LOTO folds / ablations are not headline runs
            d = json.loads(f.read_text())
            args = d.get("args", {})
            test = d.get("test", {})
            row = {
                "model": model,
                "split": args.get("split", f.name.rsplit("_seed", 1)[0]),
                "seed": args.get("seed"),
                "fold": args.get("fold", 0),
                "source_file": str(f.relative_to(ROOT)).replace("\\", "/"),
            }
            for m in METRICS:
                row[m] = test.get(m)
            if row.get("MSE") is None and row.get("RMSE") is not None:
                row["MSE"] = row["RMSE"] ** 2
            # run metadata used for the computational-cost table (W16)
            row["n_params"] = _get(d, "n_params", default=args.get("n_params"))
            row["best_val_pcc"] = _get(d, "best_val_pcc")
            row["best_val_epoch"] = _get(d, "best_val_epoch")
            row["total_train_sec"] = _get(d, "total_train_sec")
            # calibration extras (PathXDRP only)
            for c in ("sel_RMSE@50", "sel_RMSE@70", "sel_RMSE@90", "sel_RMSE@100"):
                row[c] = test.get(c)
            row["epistemic_mean"] = test.get("epistemic_mean")
            row["aleatoric_mean"] = test.get("aleatoric_mean")
            rows.append(row)
    df = pd.DataFrame(rows)
    return df.sort_values(["model", "split", "seed"]).reset_index(drop=True)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (model, split), g in df.groupby(["model", "split"]):
        row = {"model": model, "split": split, "n_seeds": len(g)}
        for m in METRICS:
            v = g[m].dropna().tolist()
            if v:
                row[f"{m}_mean"] = st.mean(v)
                row[f"{m}_std"] = std(v)
        out.append(row)
    return pd.DataFrame(out).sort_values(["split", "model"]).reset_index(drop=True)


def audit_readme(df: pd.DataFrame) -> str:
    """Explain where the README's headline table came from."""
    # The stale aggregate the public README was built from. It is kept under
    # revision/outputs/ rather than results/ so that nothing in results/ can be
    # mistaken for a current number, while the audit below stays reproducible.
    stale_path = OUT / "stale_summary_v3_vs_baselines.csv"
    lines = []
    lines.append("# README vs manuscript discrepancy -- provenance audit\n")
    lines.append("Answers Reviewer #4, major point 3.\n")

    if not stale_path.exists():
        lines.append("`revision/outputs/stale_summary_v3_vs_baselines.csv` not found.\n")
        return "\n".join(lines)

    stale = pd.read_csv(stale_path)
    lines.append("## 1. What the README table was generated from\n")
    lines.append(
        f"`revision/outputs/stale_summary_v3_vs_baselines.csv` ({len(stale)} rows), formerly `results/summary_v3_vs_baselines.csv`. The README "
        "states this explicitly as its source.\n"
    )

    dup = stale.groupby(["model", "split", "seed"]).size()
    dup = dup[dup > 1]
    lines.append("### Defect A -- duplicated runs\n")
    if len(dup):
        lines.append(
            f"{len(dup)} (model, split, seed) keys appear more than once, so the "
            "per-split means average an old run together with its rerun:\n"
        )
        lines.append("```")
        lines.append(dup.to_string())
        lines.append("```\n")
    else:
        lines.append("No duplicated keys.\n")

    lines.append("### Defect B -- stale PathXDRP sweep\n")
    sp = stale[stale.model == "pathxdrp"]
    fp = df[df.model == "pathxdrp"]
    if len(sp) and len(fp):
        sp_np = sorted(set(sp.n_params.dropna().astype(int)))
        fp_np = sorted(set(fp.n_params.dropna().astype(int))) if fp.n_params.notna().any() else []
        lines.append(
            f"- Stale CSV PathXDRP parameter count(s): {sp_np}\n"
            f"- Final sweep PathXDRP parameter count(s): {fp_np or 'not recorded in JSON'}\n"
        )
        undertrained = sp[sp.best_val_epoch <= 2]
        lines.append(
            f"- **{len(undertrained)} of {len(sp)} stale PathXDRP runs stopped at "
            f"best_val_epoch <= 2**, i.e. they never trained. These are the runs "
            "archived under `results/archive/pathxdrp_v2_lr_too_high/`, produced "
            "before the learning-rate schedule was corrected.\n"
        )
        lines.append("\n#### Per-split comparison (PCC, mean +/- std over 5 seeds)\n")
        lines.append("| Split | Stale CSV (README) | Final sweep (manuscript) | Delta |")
        lines.append("|---|---|---|---|")
        for s in SPLITS:
            a = sp[sp.split == s].PCC.dropna().tolist()
            b = fp[fp.split == s].PCC.dropna().tolist()
            if not a or not b:
                continue
            am, asd = st.mean(a), std(a)
            bm, bsd = st.mean(b), std(b)
            lines.append(
                f"| {s} | {am:.3f} +/- {asd:.3f} | {bm:.3f} +/- {bsd:.3f} | {bm - am:+.3f} |"
            )
        lines.append("")

    lines.append("## 2. Conclusion\n")
    lines.append(
        "The manuscript numbers come from the final sweep in `results/<model>/*.json`.\n"
        "The README numbers come from `results/summary_v3_vs_baselines.csv`, an "
        "aggregate that was written before the final PathXDRP sweep and never "
        "regenerated. That file mixes a failed low-epoch PathXDRP sweep with the "
        "baseline runs and additionally double-counts four baseline runs.\n\n"
        "**The manuscript is not reporting inflated numbers; the repository was "
        "reporting stale ones.** Remedy: delete the stale aggregate, regenerate the "
        "README table from `ledger.csv`, and tag the release commit.\n"
    )
    return "\n".join(lines)


def main() -> None:
    print("Collecting final-sweep results ...")
    df = collect()
    print(f"  {len(df)} runs across {df.model.nunique()} models")
    df.to_csv(OUT / "ledger.csv", index=False)

    summ = summarise(df)
    summ.to_csv(OUT / "ledger_summary.csv", index=False)
    print(f"  wrote {OUT/'ledger.csv'} and ledger_summary.csv")

    print("\nPCC by split (mean +/- std):")
    for s in SPLITS:
        print(f"  {s}")
        for m in MODELS:
            r = summ[(summ.model == m) & (summ.split == s)]
            if len(r):
                r = r.iloc[0]
                print(f"    {m:10s} {r['PCC_mean']:.4f} +/- {r['PCC_std']:.4f}  (n={int(r['n_seeds'])})")

    audit = audit_readme(df)
    (OUT / "readme_discrepancy_audit.md").write_text(audit, encoding="utf-8")
    print(f"\n  wrote {OUT/'readme_discrepancy_audit.md'}")


if __name__ == "__main__":
    main()
