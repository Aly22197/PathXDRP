"""
Deep-ensemble evaluation — Phase 5.

Loads N already-trained PathXDRP checkpoints (one per seed in seeds 0..4) for
the same (split, fold), averages their predictions on the test set, and
recomputes the full metric report. Epistemic uncertainty for the ensemble is
the variance of per-seed predictions plus the mean of per-seed evidential
epistemic — the standard deep-ensemble decomposition.

Inputs (must already exist):
  results/pathxdrp/<split>_seed{s}_fold{f}_preds.csv   for s in seeds

Outputs:
  results/pathxdrp/<split>_ensemble_fold{f}.json
  results/pathxdrp/<split>_ensemble_fold{f}_preds.csv

Usage:
  python scripts/ensemble_eval.py --split random --fold 0
  python scripts/ensemble_eval.py --splits random cell_blind drug_blind \\
      --seeds 0 1 2 3 4 --fold 0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

ROOT = Path(__file__).parent.parent


def load_seed_preds(model: str, split: str, seed: int, fold: int) -> pd.DataFrame | None:
    p = ROOT / "results" / model / f"{split}_seed{seed}_fold{fold}_preds.csv"
    if not p.exists():
        return None
    return pd.read_csv(p)


def ensemble(
    split: str,
    seeds: list[int],
    fold: int,
    model: str = "pathxdrp",
) -> dict | None:
    from pathxdrp.eval.metrics import regression_report

    dfs = []
    missing = []
    for s in seeds:
        d = load_seed_preds(model, split, s, fold)
        if d is None:
            missing.append(s)
            continue
        d = d.rename(columns={"y_pred": f"y_pred_{s}", "epistemic": f"epi_{s}"})
        dfs.append(d.set_index(["drug_id", "cosmic_id", "y_true"]))

    if not dfs:
        print(f"[{split}/fold{fold}] No seed prediction CSVs found — skipping.", flush=True)
        return None
    if missing:
        print(f"[{split}/fold{fold}] Missing seeds: {missing}", flush=True)

    merged = pd.concat(dfs, axis=1).reset_index()

    # Average across seeds
    pred_cols = [c for c in merged.columns if c.startswith("y_pred_")]
    epi_cols  = [c for c in merged.columns if c.startswith("epi_")]
    y_pred_arr = merged[pred_cols].to_numpy()                # (N, K)
    y_pred_mean = y_pred_arr.mean(axis=1)                    # (N,)
    y_pred_var  = y_pred_arr.var(axis=1, ddof=0)             # disagreement
    if epi_cols:
        epi_mean = merged[epi_cols].to_numpy().mean(axis=1)  # avg evidential
    else:
        epi_mean = np.zeros_like(y_pred_mean)
    epistemic_total = y_pred_var + epi_mean                  # ensemble decomposition

    y_true     = merged["y_true"].to_numpy()
    drug_ids   = merged["drug_id"].to_numpy()
    cosmic_ids = merged["cosmic_id"].to_numpy()

    report = regression_report(
        y_true, y_pred_mean,
        drug_ids=drug_ids,
        cell_ids=cosmic_ids,
        uncertainties=epistemic_total,
    )
    report["n_seeds_used"]   = len(dfs)
    report["epistemic_mean"] = float(epistemic_total.mean())

    # Persist
    out_dir   = ROOT / "results" / model
    out_path  = out_dir / f"{split}_ensemble_fold{fold}.json"
    preds_out = out_dir / f"{split}_ensemble_fold{fold}_preds.csv"

    with open(out_path, "w") as f:
        json.dump({
            "args":   {"split": split, "seeds": seeds, "fold": fold, "model": model},
            "test":   report,
            "n_seeds": len(dfs),
        }, f, indent=2)

    pd.DataFrame({
        "drug_id":   drug_ids,
        "cosmic_id": cosmic_ids,
        "y_true":    y_true,
        "y_pred":    y_pred_mean,
        "epistemic": epistemic_total,
        "disagreement": y_pred_var,
    }).to_csv(preds_out, index=False)

    print(f"[{split}/fold{fold}] ensemble PCC={report['PCC']:.4f} "
          f"RMSE={report['RMSE']:.4f} ECE={report.get('ECE', float('nan')):.4f} "
          f"-> {out_path}", flush=True)
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--splits", nargs="+",
                   default=["random", "cell_blind", "drug_blind",
                            "scaffold_blind", "tissue_blind"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--fold",  type=int, default=0)
    p.add_argument("--model", default="pathxdrp")
    args = p.parse_args()

    pbar = tqdm(args.splits, desc="Ensembling splits", unit="split")
    for split in pbar:
        pbar.set_postfix(split=split)
        ensemble(split, args.seeds, args.fold, model=args.model)
    pbar.close()


if __name__ == "__main__":
    main()
