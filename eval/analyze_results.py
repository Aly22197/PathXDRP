"""
Aggregate sweep JSON results into paper-ready tables.

Output formats
--------------
- Console: one pivot per metric, mean +/- std across seeds.
- Per-metric LaTeX tables (--latex), one per file or all into one --input-able file.
- A summary CSV of (model, split, seed) -> all metrics for downstream analysis.

Examples
--------
  # Console summary across all metrics:
  python eval/analyze_results.py

  # Write all four headline tables to manuscript/auto_tables.tex:
  python eval/analyze_results.py --latex --out manuscript/auto_tables.tex

  # Just PCC, only random + drug_blind splits:
  python eval/analyze_results.py --metrics PCC --splits random drug_blind

  # Save per-(model, split, seed) raw data as CSV for further analysis:
  python eval/analyze_results.py --csv results/all_runs.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

ROOT        = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results"

# Display order
MODEL_ORDER = ["pathxdrp", "drpreter", "graphdrp", "cdrscan"]
MODEL_LABELS = {
    "pathxdrp": "PathXDRP (ours)",
    "drpreter": "DRPreter",
    "graphdrp": "GraphDRP",
    "cdrscan":  "CDRScan",
}
SPLIT_ORDER  = ["random", "cell_blind", "drug_blind", "scaffold_blind", "tissue_blind"]
SPLIT_LABELS = {
    "random":         "random",
    "cell_blind":     "cell-blind",
    "drug_blind":     "drug-blind",
    "scaffold_blind": "scaffold-blind",
    "tissue_blind":   "tissue-blind",
}

# Metrics where lower is better (so bolding picks the minimum)
LOWER_IS_BETTER = {"RMSE", "MAE", "ECE",
                   "sel_RMSE@50", "sel_RMSE@70", "sel_RMSE@90", "sel_RMSE@100"}

DEFAULT_METRICS = ["PCC", "RMSE", "Spearman", "R2", "Per-drug PCC", "Per-cell PCC"]
PRECISION = 3   # decimal places in tables; bumped from 4 because std typically dwarfs the 4th place


# ---
# Loading
# ---

def load_all_results() -> pd.DataFrame:
    """Scan results/ tree and return one row per (model, split, seed, fold, run_tag)."""
    rows = []
    for json_path in RESULTS_DIR.rglob("*.json"):
        # Skip the archive directory and non-run files
        if "archive" in json_path.parts:
            continue
        # Skip per-drug XAI outputs, preds CSVs etc.
        if any(s in json_path.stem for s in ["xai", "preds"]):
            continue

        try:
            model  = json_path.parent.name
            stem   = json_path.stem          # e.g. random_seed0_fold0  or random_seed0_fold0_v3
            tokens = stem.split("_")
            # split name may itself contain underscores (cell_blind, drug_blind...)
            seed_idx = next(i for i, t in enumerate(tokens) if t.startswith("seed"))
            fold_idx = seed_idx + 1
            split  = "_".join(tokens[:seed_idx])
            seed   = int(tokens[seed_idx].replace("seed", ""))
            fold   = int(tokens[fold_idx].replace("fold", ""))
            # Anything after fold token is the run_tag (e.g. v2, v3, fp_nomask)
            run_tag = "_".join(tokens[fold_idx + 1:]) if len(tokens) > fold_idx + 1 else ""
        except (IndexError, ValueError, StopIteration):
            continue

        try:
            with open(json_path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            print(f"  warn: could not read {json_path}", file=sys.stderr)
            continue

        test = data.get("test", {})
        if not test:
            continue

        row: dict = {"model": model, "split": split, "seed": seed, "fold": fold,
                     "run_tag": run_tag}
        # Copy primitive metrics; skip nested structures like risk_coverage
        for k, v in test.items():
            if isinstance(v, (int, float)):
                row[k] = v
        # Useful metadata for diagnostics
        row["best_val_pcc"]    = data.get("best_val_pcc")
        row["best_val_epoch"]  = data.get("best_val_epoch")
        row["total_train_sec"] = data.get("total_train_sec")
        row["n_params"]        = data.get("n_params")
        rows.append(row)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


# ---
# Aggregation
# ---

def aggregate(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """Mean +/- std across seeds (and folds) for each (model, split, run_tag)."""
    records = []
    for (model, split, run_tag), grp in df.groupby(["model", "split", "run_tag"]):
        row = {"model": model, "split": split, "run_tag": run_tag, "n": len(grp)}
        for m in metrics:
            if m not in grp.columns:
                continue
            vals = grp[m].dropna().values
            if len(vals) == 0:
                continue
            row[f"{m}_mean"] = float(np.mean(vals))
            row[f"{m}_std"]  = float(np.std(vals, ddof=0))
            row[m]            = f"{row[f'{m}_mean']:.{PRECISION}f} +/- {row[f'{m}_std']:.{PRECISION}f}"
        records.append(row)
    return pd.DataFrame(records)


# ---
# Console pivot
# ---

def pivot_table(agg: pd.DataFrame, metric: str, splits: Iterable[str],
                run_tag: str = "") -> pd.DataFrame:
    rows = []
    agg_filt = agg[agg.run_tag == run_tag] if run_tag else agg
    for model in MODEL_ORDER:
        m_data = agg_filt[agg_filt.model == model]
        row = {"Model": MODEL_LABELS.get(model, model)}
        for split in splits:
            cell = m_data[m_data.split == split]
            row[SPLIT_LABELS.get(split, split)] = cell[metric].values[0] if not cell.empty and metric in cell else "-"
        rows.append(row)
    return pd.DataFrame(rows).set_index("Model")


# ---
# LaTeX
# ---

def _best_per_column(agg: pd.DataFrame, metric: str, splits: list[str]) -> dict[str, str]:
    """For each split, return which model has the best mean."""
    best = {}
    for split in splits:
        rows = agg[(agg.split == split) & (f"{metric}_mean" in agg.columns)]
        rows = agg[agg.split == split]
        if rows.empty:
            continue
        col = f"{metric}_mean"
        if col not in rows.columns:
            continue
        if metric in LOWER_IS_BETTER:
            best[split] = rows.loc[rows[col].idxmin(), "model"]
        else:
            best[split] = rows.loc[rows[col].idxmax(), "model"]
    return best


def metric_table_latex(
    agg: pd.DataFrame,
    metric: str,
    splits: list[str],
    caption: str,
    label: str,
) -> str:
    """Single metric x split table. Bold-best per column."""
    best = _best_per_column(agg, metric, splits)
    n_cols = len(splits)

    # Wide 6-column tables (Model + 5 splits) overflow cas-dc's ~84mm
    # single column, so we emit them as two-column-wide table* with
    # tabular* sized to \tblwidth. The numeric columns are stretched to
    # fill the column width using @{\extracolsep{\fill}}, which prevents
    # cells from looking sparse on the page.
    lines = [
        r"\begin{table*}[!htbp]",
        r"\centering\small",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\begin{tabular*}{\tblwidth}{@{\extracolsep{\fill}}l" + "c" * n_cols + "@{}}",
        r"\toprule",
        "Model & " + " & ".join(SPLIT_LABELS.get(s, s) for s in splits) + r" \\",
        r"\midrule",
    ]

    for model in MODEL_ORDER:
        m_data = agg[agg.model == model]
        if m_data.empty:
            continue
        label_str = MODEL_LABELS.get(model, model)
        if model == "pathxdrp":
            label_str = r"\textbf{" + label_str + "}"
        cells = [label_str]
        for split in splits:
            cell = m_data[m_data.split == split]
            mean_col = f"{metric}_mean"
            std_col  = f"{metric}_std"
            if cell.empty or mean_col not in cell.columns or pd.isna(cell[mean_col].values[0]):
                cells.append("-")
                continue
            mean = cell[mean_col].values[0]
            std  = cell[std_col].values[0]
            entry = rf"{mean:.{PRECISION}f} $\pm$ {std:.{PRECISION}f}"
            if best.get(split) == model:
                entry = r"\textbf{" + entry + "}"
            cells.append(entry)
        lines.append(" & ".join(cells) + r" \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular*}",
        r"\end{table*}",
    ]
    return "\n".join(lines)


METRIC_CAPTIONS = {
    "PCC":          ("Pearson correlation between predicted and observed LN\\_IC50 across all five split regimes. Mean $\\pm$ standard deviation across five seeds; bold marks the best per split.",
                     "tab:main_results"),
    "RMSE":         ("Root mean squared error (LN\\_IC50 units) across the five split regimes. Mean $\\pm$ standard deviation across five seeds; bold marks the best per split.",
                     "tab:rmse"),
    "Spearman":     ("Spearman rank correlation between predicted and observed LN\\_IC50 across the five split regimes. Mean $\\pm$ standard deviation across five seeds; bold marks the best per split.",
                     "tab:spearman"),
    "R2":           ("Coefficient of determination ($R^2$) across the five split regimes. Mean $\\pm$ standard deviation across five seeds; bold marks the best per split.",
                     "tab:r2"),
    "Per-drug PCC": ("Per-drug Pearson correlation: average within-drug PCC, requiring $\\geq 2$ samples per drug. This metric exposes whether the model ranks cell lines correctly within a drug.",
                     "tab:per_drug_pcc"),
    "Per-cell PCC": ("Per-cell Pearson correlation: average within-cell-line PCC. Captures whether the model ranks compounds correctly for a given cell line, which is the screening-application target.",
                     "tab:per_cell_pcc"),
}


def all_tables_latex(agg: pd.DataFrame, metrics: list[str], splits: list[str]) -> str:
    """Concatenate one metric-table per metric into a single \\input-able block."""
    blocks = [
        f"% Auto-generated by eval/analyze_results.py.",
        f"% Do not edit by hand; re-run the script after the sweep finishes.",
        "",
    ]
    for metric in metrics:
        if metric not in METRIC_CAPTIONS:
            caption = f"Test {metric} across the five split regimes. Mean $\\pm$ standard deviation across five seeds."
            label   = f"tab:{metric.lower().replace(' ', '_')}"
        else:
            caption, label = METRIC_CAPTIONS[metric]
        blocks.append(metric_table_latex(agg, metric, splits, caption=caption, label=label))
        blocks.append("")
    return "\n".join(blocks)


# ---
# CLI
# ---

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--splits",   nargs="+", default=SPLIT_ORDER)
    p.add_argument("--metrics",  nargs="+", default=DEFAULT_METRICS)
    p.add_argument("--run_tag",  type=str,  default="",
                   help="Filter ALL models to this run_tag (e.g. 'v3'). "
                        "Default (empty) shows all tags together, which inflates std.")
    p.add_argument("--focal_tag", type=str, default="",
                   help="Per-model tag: for models WITH this tag use it; "
                        "for models WITHOUT fall back to the empty-tag runs. "
                        "Use '--focal_tag v3' to compare PathXDRP v3 vs baselines.")
    p.add_argument("--latex",    action="store_true",
                   help="Emit LaTeX. With --out, all metric tables go into one file.")
    p.add_argument("--out",      type=str, default=None,
                   help="Write LaTeX to this file (default: stdout).")
    p.add_argument("--csv",      type=str, default=None,
                   help="Also dump the raw per-run dataframe as CSV.")
    args = p.parse_args()

    raw = load_all_results()
    if raw.empty:
        print("No results found in results/  -- run some training first.")
        return

    # Show available run_tags for diagnostics
    if "run_tag" in raw.columns:
        tags_present = sorted(raw["run_tag"].unique())
        print(f"Run tags present: {tags_present}")
        if args.run_tag:
            raw = raw[raw["run_tag"] == args.run_tag]
            if raw.empty:
                print(f"No results for --run_tag={args.run_tag!r}")
                return
            print(f"Filtered to run_tag={args.run_tag!r}")
        elif args.focal_tag:
            # For each model: use focal_tag if it has that tag, else use "" runs
            models_with_tag = set(raw[raw["run_tag"] == args.focal_tag]["model"].unique())
            keep = []
            for model in raw["model"].unique():
                if model in models_with_tag:
                    keep.append(raw[(raw["model"] == model) & (raw["run_tag"] == args.focal_tag)])
                else:
                    keep.append(raw[(raw["model"] == model) & (raw["run_tag"] == "")])
            raw = pd.concat(keep, ignore_index=True) if keep else raw
            if raw.empty:
                print(f"No results for --focal_tag={args.focal_tag!r}")
                return
            print(f"Focal tag={args.focal_tag!r}: "
                  f"{', '.join(f'{m}({args.focal_tag})' if m in models_with_tag else f'{m}(canonical)' for m in MODEL_ORDER if m in raw['model'].unique())}")

    n_models  = raw.model.nunique()
    n_splits  = raw.split.nunique()
    n_seeds   = raw.seed.nunique()
    n_runs    = len(raw)
    print(f"Loaded {n_runs} run(s): {n_models} models, {n_splits} splits, {n_seeds} seeds")

    # Coverage table
    coverage = raw.groupby(["model", "split"]).size().unstack(fill_value=0)
    print("\nRun count per (model, split):")
    print(coverage.to_string())
    print()

    if args.csv:
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        raw.to_csv(args.csv, index=False)
        print(f"\nRaw runs CSV -> {args.csv}")

    agg = aggregate(raw, args.metrics)

    # Console: one pivot per metric
    for metric in args.metrics:
        if f"{metric}_mean" not in agg.columns:
            continue
        print(f"\n{'='*60}")
        print(f"  {metric}  (mean +/- std)")
        print(f"{'='*60}")
        print(pivot_table(agg, metric, args.splits, run_tag=args.run_tag).to_string())

    if args.latex:
        tex = all_tables_latex(agg, args.metrics, args.splits)
        if args.out:
            out_path = Path(args.out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(tex, encoding="utf-8")
            print(f"\nLaTeX tables -> {out_path}")
        else:
            print("\n" + tex)


if __name__ == "__main__":
    main()
