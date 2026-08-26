"""
W7 -- Compare the evidential uncertainty against simple baselines.

Answers the second half of Reviewer #4, point 5: "compare with simple
uncertainty baselines". The submitted paper reported the evidential head's
selective-RMSE curve with nothing to compare it against, so a reader could not
tell whether the machinery bought anything over a cheap heuristic.

The question is whether sigma ranks errors better than proxies that need no
uncertainty head at all. Four baselines, all computable from data the model
already has:

  extremeness      |predicted - training mean|. Extreme predictions are
                   usually wrong more often; this needs only the prediction.
  drug_rarity      1 / (training rows for that drug). Rare drugs are harder.
  cell_rarity      1 / (training rows for that cell line).
  expr_distance    distance from the test cell line's expression profile to
                   the nearest training cell line, in PCA space. This is the
                   standard "how far outside the training distribution is
                   this?" heuristic.

Each is scored the same way as sigma: AUROC for identifying the larger-error
half of the test set, Spearman against absolute error, and the RMSE reduction
from discarding the least-confident half.

No GPU required.

Usage:
    python revision/scripts/uncertainty_baselines.py
Outputs:
    outputs/uncertainty_baselines.csv
    outputs/uncertainty_baselines.md
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

SPLITS = ["random", "cell_blind", "drug_blind", "scaffold_blind", "tissue_blind"]
SEEDS = [0, 1, 2, 3, 4]
PRETTY = {"random": "random", "cell_blind": "cell-blind", "drug_blind": "drug-blind",
          "scaffold_blind": "scaffold-blind", "tissue_blind": "tissue-blind"}


def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def score(y, p, u):
    """AUROC / Spearman / selective-RMSE gain for an uncertainty proxy u."""
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score
    err = np.abs(y - p)
    lab = (err > np.median(err)).astype(int)
    auroc = np.nan if lab.min() == lab.max() else roc_auc_score(lab, u)
    rho = spearmanr(u, err).statistic
    keep = np.argsort(u)[: max(1, len(y) // 2)]
    gain = 1 - rmse(y[keep], p[keep]) / rmse(y, p)
    return float(auroc), float(rho), float(gain)


def main() -> None:
    from pathxdrp.data.loader import build_master_df
    from pathxdrp.data.splits import load_split

    print("Loading data for the distribution-distance baseline ...")
    df, expr = build_master_df(require_smiles=True, standardize=False)

    # PCA of expression once; used for the nearest-training-cell distance
    from sklearn.decomposition import PCA
    X = expr.to_numpy(dtype="float32")
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-8)
    pcs = PCA(n_components=20, random_state=0).fit_transform(Xs)
    cell_pc = {int(c): pcs[i] for i, c in enumerate(expr.index)}

    rows = []
    for split in SPLITS:
        for seed in SEEDS:
            f = ROOT / "results" / "pathxdrp" / f"{split}_seed{seed}_fold0_preds.csv"
            if not f.exists():
                continue
            d = pd.read_csv(f)
            if "epistemic" not in d.columns:
                continue
            try:
                tr, _, _ = load_split(split, seed, 0)
            except Exception:
                continue
            train = df.iloc[tr]
            y = d.y_true.to_numpy(float)
            p = d.y_pred.to_numpy(float)

            sigma = (d.epistemic + d.aleatoric).to_numpy(float)
            train_mean = float(train.LN_IC50.mean())
            extreme = np.abs(p - train_mean)

            drug_n = train.DRUG_ID.value_counts()
            cell_n = train.COSMIC_ID.value_counts()
            drug_rare = 1.0 / d.drug_id.map(lambda x: drug_n.get(x, 0) + 1).to_numpy(float)
            cell_rare = 1.0 / d.cosmic_id.map(lambda x: cell_n.get(x, 0) + 1).to_numpy(float)

            tr_cells = sorted({int(c) for c in train.COSMIC_ID.unique() if int(c) in cell_pc})
            TR = np.stack([cell_pc[c] for c in tr_cells])
            uniq = {int(c) for c in d.cosmic_id.unique() if int(c) in cell_pc}
            dist_map = {}
            for c in uniq:
                dist_map[c] = float(np.min(np.linalg.norm(TR - cell_pc[c], axis=1)))
            expr_dist = d.cosmic_id.map(lambda x: dist_map.get(int(x), np.nan)).to_numpy(float)
            expr_dist = np.nan_to_num(expr_dist, nan=float(np.nanmedian(expr_dist)))

            for name, u in (("evidential sigma", sigma),
                            ("extremeness", extreme),
                            ("drug rarity", drug_rare),
                            ("cell rarity", cell_rare),
                            ("expression distance", expr_dist)):
                a, r, g = score(y, p, u)
                rows.append({"split": split, "seed": seed, "method": name,
                             "auroc": a, "spearman": r, "sel_gain": g})
            print(f"  {split:16s} seed{seed} done")

    out = pd.DataFrame(rows)
    out.to_csv(OUT / "uncertainty_baselines.csv", index=False)

    g = out.groupby(["split", "method"]).agg(
        auroc=("auroc", "mean"), rho=("spearman", "mean"),
        gain=("sel_gain", "mean")).reset_index()

    METHODS = ["evidential sigma", "extremeness", "drug rarity", "cell rarity",
               "expression distance"]
    L = ["# W7 -- Evidential uncertainty against simple baselines\n",
         "Answers the second half of Reviewer #4, point 5.\n",
         "Each column is an uncertainty proxy scored the same way: AUROC for "
         "identifying the larger-error half of the test set. Higher is better; "
         "0.5 is chance. None of the baselines needs an uncertainty head.\n",
         "| Split | " + " | ".join(METHODS) + " |",
         "|---" * (len(METHODS) + 1) + "|"]
    for sp in SPLITS:
        cells = []
        sub = g[g.split == sp]
        if sub.empty:
            continue
        best = sub.auroc.max()
        for m in METHODS:
            r = sub[sub.method == m]
            if r.empty:
                cells.append("--"); continue
            v = r.auroc.iloc[0]
            cells.append(f"**{v:.3f}**" if abs(v - best) < 1e-9 else f"{v:.3f}")
        L.append(f"| {PRETTY[sp]} | " + " | ".join(cells) + " |")
    L.append("")

    L.append("## Selective-RMSE gain from discarding the least-confident half\n")
    L.append("| Split | " + " | ".join(METHODS) + " |")
    L.append("|---" * (len(METHODS) + 1) + "|")
    for sp in SPLITS:
        sub = g[g.split == sp]
        if sub.empty:
            continue
        cells = []
        best = sub.gain.max()
        for m in METHODS:
            r = sub[sub.method == m]
            if r.empty:
                cells.append("--"); continue
            v = r.gain.iloc[0]
            cells.append(f"**{100*v:.1f}%**" if abs(v - best) < 1e-9 else f"{100*v:.1f}%")
        L.append(f"| {PRETTY[sp]} | " + " | ".join(cells) + " |")
    L.append("")

    ev = g[g.method == "evidential sigma"]
    others = g[g.method != "evidential sigma"]
    wins = 0
    for sp in SPLITS:
        e = ev[ev.split == sp]
        o = others[others.split == sp]
        if len(e) and len(o) and e.auroc.iloc[0] >= o.auroc.max():
            wins += 1
    L.append("## Reading\n")
    L.append(
        f"The evidential sigma is the best error-ranking signal on **{wins} of "
        f"{len(SPLITS)}** splits. Where it is not, the winner is worth noting "
        "rather than hiding: a cheap heuristic that beats a trained uncertainty "
        "head is a real result about the head.\n\n"
        "The comparison also gives the selective-prediction claim a reference "
        "point it lacked in the submitted version. A 24% RMSE reduction sounds "
        "impressive on its own; it is more informative alongside what "
        "`|prediction - training mean|` achieves for free.\n")
    (OUT / "uncertainty_baselines.md").write_text("\n".join(L), encoding="utf-8")
    print(f"\nwrote {OUT/'uncertainty_baselines.md'}")


if __name__ == "__main__":
    main()
