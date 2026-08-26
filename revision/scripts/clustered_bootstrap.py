"""
W4 -- Clustered bootstrap replacing the row-level bootstrap of Table 8.

Answers Reviewer #3 (point 2) and Reviewer #4 (point 6).

The submitted manuscript bootstrapped individual (drug, cell) ROWS on a single
held-out fold.  Rows that share a drug or a cell line are statistically
dependent, so row resampling underestimates the variance of any metric computed
over the drug-cell response matrix and produces over-optimistic intervals.

This script recomputes the PathXDRP-minus-baseline PCC difference under five
resampling schemes:

  rows      naive row bootstrap (the submitted, invalid scheme -- kept for
            comparison so the correction is visible)
  drug      resample DRUG_ID clusters with replacement
  cell      resample COSMIC_ID clusters with replacement
  scaffold  resample Bemis-Murcko scaffold clusters with replacement
  twoway    Cameron-Gelbach-Miller multiway estimator:
            Var_2way = Var_drug + Var_cell - Var_rows

The paired structure is preserved: both models are evaluated on exactly the same
resampled index set in each replicate, so the difference is a paired statistic.

Usage:
    python revision/scripts/clustered_bootstrap.py [--n-boot 2000]
Outputs:
    outputs/clustered_bootstrap_seed0.csv     direct replacement for Table 8
    outputs/clustered_bootstrap_allseeds.csv  per-seed detail
    outputs/clustered_bootstrap_summary.md    human-readable summary
    tables/tab_clustered_bootstrap.tex        LaTeX table for the manuscript
"""
from __future__ import annotations

import argparse
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "outputs"
TAB = BASE / "tables"
OUT.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

SPLITS = ["random", "cell_blind", "drug_blind", "scaffold_blind", "tissue_blind"]
BASELINES = ["drpreter", "graphdrp", "cdrscan"]
SEEDS = [0, 1, 2, 3, 4]
SCHEMES = ["rows", "drug", "cell", "scaffold", "twoway"]

PRETTY = {
    "random": "random", "cell_blind": "cell-blind", "drug_blind": "drug-blind",
    "scaffold_blind": "scaffold-blind", "tissue_blind": "tissue-blind",
}
MODEL_PRETTY = {"drpreter": "DRPreter", "graphdrp": "GraphDRP", "cdrscan": "CDRScan"}


# ---------------------------------------------------------------- scaffolds

def scaffold_map() -> dict[int, str]:
    """DRUG_ID -> Bemis-Murcko scaffold SMILES (same recipe as splits.py)."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem.Scaffolds import MurckoScaffold

    RDLogger.DisableLog("rdApp.*")
    df = pd.read_parquet(ROOT / "data" / "processed" / "drugs_with_smiles.parquet")
    out: dict[int, str] = {}
    for did, smi in zip(df.DRUG_ID.values, df.SMILES.values):
        mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
        if mol is None:
            out[int(did)] = "__invalid__"
            continue
        try:
            out[int(did)] = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        except Exception:
            out[int(did)] = "__error__"
    return out


# ---------------------------------------------------------------- core stats

def pcc(y: np.ndarray, p: np.ndarray) -> float:
    if len(y) < 3:
        return np.nan
    ys, ps = y.std(), p.std()
    if ys == 0 or ps == 0:
        return np.nan
    return float(((y - y.mean()) * (p - p.mean())).mean() / (ys * ps))


def load_pair(split: str, seed: int, baseline: str,
              scaf: dict[int, str]) -> pd.DataFrame | None:
    """Merge PathXDRP and baseline predictions on the shared test rows."""
    fa = RESULTS / "pathxdrp" / f"{split}_seed{seed}_fold0_preds.csv"
    fb = RESULTS / baseline / f"{split}_seed{seed}_fold0_preds.csv"
    if not (fa.exists() and fb.exists()):
        return None
    a = pd.read_csv(fa)[["drug_id", "cosmic_id", "y_true", "y_pred"]]
    b = pd.read_csv(fb)[["drug_id", "cosmic_id", "y_true", "y_pred"]]
    a = a.rename(columns={"y_pred": "p_a"})
    b = b.rename(columns={"y_pred": "p_b"}).drop(columns=["y_true"])
    m = a.merge(b, on=["drug_id", "cosmic_id"], how="inner")
    if len(m) == 0:
        return None
    m["scaffold"] = m.drug_id.map(lambda d: scaf.get(int(d), "__unknown__"))
    return m


def _cluster_index(codes: np.ndarray, n_clusters: int) -> list[np.ndarray]:
    """Row indices grouped by cluster code."""
    order = np.argsort(codes, kind="stable")
    sorted_codes = codes[order]
    bounds = np.searchsorted(sorted_codes, np.arange(n_clusters + 1))
    return [order[bounds[i]:bounds[i + 1]] for i in range(n_clusters)]


def bootstrap_delta(m: pd.DataFrame, scheme: str, n_boot: int,
                    rng: np.random.Generator) -> np.ndarray:
    """n_boot replicates of PCC(PathXDRP) - PCC(baseline)."""
    y = m.y_true.to_numpy(np.float64)
    pa = m.p_a.to_numpy(np.float64)
    pb = m.p_b.to_numpy(np.float64)
    n = len(m)

    if scheme == "rows":
        deltas = np.empty(n_boot)
        for i in range(n_boot):
            idx = rng.integers(0, n, n)
            deltas[i] = pcc(y[idx], pa[idx]) - pcc(y[idx], pb[idx])
        return deltas

    key = {"drug": "drug_id", "cell": "cosmic_id", "scaffold": "scaffold"}[scheme]
    codes, uniq = pd.factorize(m[key])
    groups = _cluster_index(codes, len(uniq))
    k = len(groups)
    if k < 3:
        return np.full(n_boot, np.nan)

    deltas = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, k, k)
        idx = np.concatenate([groups[j] for j in pick])
        deltas[i] = pcc(y[idx], pa[idx]) - pcc(y[idx], pb[idx])
    return deltas


def summarise(deltas: np.ndarray, point: float) -> dict:
    d = deltas[~np.isnan(deltas)]
    if len(d) < 10:
        return {"delta": point, "se": np.nan, "lo": np.nan, "hi": np.nan, "p": np.nan}
    lo, hi = np.percentile(d, [2.5, 97.5])
    # two-sided bootstrap p: fraction of replicates on the wrong side of zero, x2
    frac = (d <= 0).mean() if point > 0 else (d >= 0).mean()
    p = min(1.0, 2 * max(frac, 1.0 / len(d)))
    return {"delta": point, "se": float(d.std(ddof=1)), "lo": float(lo),
            "hi": float(hi), "p": float(p)}


# ---------------------------------------------------------------- driver

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    print("Building scaffold map ...")
    scaf = scaffold_map()
    print(f"  {len(scaf)} drugs, {len(set(scaf.values()))} distinct scaffolds")

    rows = []
    for split in SPLITS:
        for baseline in BASELINES:
            for seed in SEEDS:
                m = load_pair(split, seed, baseline, scaf)
                if m is None:
                    print(f"  [skip] {split}/{baseline}/seed{seed}")
                    continue
                point = pcc(m.y_true.to_numpy(), m.p_a.to_numpy()) - \
                        pcc(m.y_true.to_numpy(), m.p_b.to_numpy())
                # Python's hash() is salted per process (PYTHONHASHSEED), so
                # seeding from it made the bootstrap irreproducible: the same
                # command gave a different significant share on every run, and
                # the manuscript could not stay in step with it. crc32 over the
                # cell's identity is stable across processes and machines.
                key = f"{split}|{baseline}|{seed}".encode("utf-8")
                rng = np.random.default_rng(zlib.crc32(key))

                res = {}
                for sch in ["rows", "drug", "cell", "scaffold"]:
                    d = bootstrap_delta(m, sch, args.n_boot, rng)
                    res[sch] = summarise(d, point)

                # Cameron-Gelbach-Miller two-way variance
                v = (res["drug"]["se"] ** 2 + res["cell"]["se"] ** 2
                     - res["rows"]["se"] ** 2)
                se2 = float(np.sqrt(v)) if v > 0 else float(
                    max(res["drug"]["se"], res["cell"]["se"]))
                from math import erfc, sqrt
                z = abs(point) / se2 if se2 > 0 else np.inf
                res["twoway"] = {
                    "delta": point, "se": se2,
                    "lo": point - 1.96 * se2, "hi": point + 1.96 * se2,
                    "p": float(erfc(z / sqrt(2))),
                }

                for sch in SCHEMES:
                    r = res[sch]
                    rows.append({
                        "split": split, "baseline": baseline, "seed": seed,
                        "scheme": sch, "n_rows": len(m),
                        "n_drugs": m.drug_id.nunique(),
                        "n_cells": m.cosmic_id.nunique(),
                        "n_scaffolds": m.scaffold.nunique(),
                        **r,
                        "significant": bool(r["lo"] > 0 or r["hi"] < 0)
                        if not np.isnan(r["lo"]) else False,
                    })
                print(f"  {split:15s} {baseline:9s} seed{seed}  "
                      f"dPCC={point:+.4f}  rows CI=[{res['rows']['lo']:+.4f},"
                      f"{res['rows']['hi']:+.4f}]  2way CI=[{res['twoway']['lo']:+.4f},"
                      f"{res['twoway']['hi']:+.4f}]")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "clustered_bootstrap_allseeds.csv", index=False)
    s0 = df[df.seed == 0].copy()
    s0.to_csv(OUT / "clustered_bootstrap_seed0.csv", index=False)
    print(f"\nwrote {OUT/'clustered_bootstrap_allseeds.csv'}")

    write_summary(df)
    write_latex(df)


def write_summary(df: pd.DataFrame) -> None:
    lines = ["# W4 -- Clustered bootstrap results\n",
             "Answers Reviewer #3.2 and Reviewer #4.6.\n",
             "`rows` is the submitted (invalid) scheme; the others cluster the "
             "resampling unit so that dependent observations move together.\n"]

    lines.append("## How many (split, baseline, seed) comparisons stay significant\n")
    lines.append("| Scheme | Significant at 95% | of total | share |")
    lines.append("|---|---|---|---|")
    for sch in SCHEMES:
        g = df[df.scheme == sch]
        n = len(g)
        s = int(g.significant.sum())
        lines.append(f"| {sch} | {s} | {n} | {100*s/n:.0f}% |")
    lines.append("")

    lines.append("## Seed-0 fold, direct replacement for submitted Table 8\n")
    lines.append("| Split | Baseline | dPCC | rows 95% CI | drug 95% CI | cell 95% CI | two-way 95% CI |")
    lines.append("|---|---|---|---|---|---|---|")
    s0 = df[df.seed == 0]
    for split in SPLITS:
        for b in BASELINES:
            g = s0[(s0.split == split) & (s0.baseline == b)]
            if g.empty:
                continue
            def ci(sch):
                r = g[g.scheme == sch]
                if r.empty or np.isnan(r.iloc[0].lo):
                    return "n/a"
                r = r.iloc[0]
                star = "" if (r.lo <= 0 <= r.hi) else "*"
                return f"[{r.lo:+.3f}, {r.hi:+.3f}]{star}"
            d = g.iloc[0].delta
            lines.append(f"| {PRETTY[split]} | {MODEL_PRETTY[b]} | {d:+.4f} | "
                         f"{ci('rows')} | {ci('drug')} | {ci('cell')} | {ci('twoway')} |")
    lines.append("\n`*` marks an interval that excludes zero.\n")

    lines.append("## Interpretation\n")
    r_share = df[df.scheme == 'rows'].significant.mean()
    t_share = df[df.scheme == 'twoway'].significant.mean()
    lines.append(
        f"Under naive row resampling {100*r_share:.0f}% of the model-vs-baseline "
        f"PCC differences look significant. Under the two-way cluster estimator "
        f"only {100*t_share:.0f}% do. The row-level scheme was inflating "
        "significance exactly as Reviewer #3 predicted: the effective sample size "
        "is the number of drugs and cell lines, not the number of matrix cells.\n"
    )
    lines.append(
        "This is consistent with the Friedman test already reported in the "
        "submitted manuscript (chi2 = 4.44, p = 0.218): the four architectures "
        "cannot be separated on prediction accuracy. The revised manuscript "
        "should state this as the headline finding rather than as a caveat.\n"
    )
    (OUT / "clustered_bootstrap_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT/'clustered_bootstrap_summary.md'}")


def write_latex(df: pd.DataFrame) -> None:
    s0 = df[df.seed == 0]
    L = [
        r"% Generated by revision/scripts/clustered_bootstrap.py",
        r"\begin{table*}[!h]",
        r"\centering\small",
        r"\caption{Cluster bootstrap of the difference in test PCC (PathXDRP minus",
        r"baseline) on the seed-0 fold of each split ($2{,}000$ resamples). Rows are",
        r"statistically dependent because many share a drug or a cell line, so the",
        r"naive row bootstrap (first interval) understates the variance. The",
        r"drug-, cell- and two-way clustered intervals resample whole drugs, whole",
        r"cell lines and both margins respectively; the two-way interval uses the",
        r"Cameron--Gelbach--Miller multiway variance estimator. An interval that",
        r"excludes zero is marked $^{*}$.}",
        r"\label{tab:clustered_bootstrap}",
        r"\begin{tabular*}{\tblwidth}{@{\extracolsep{\fill}}ll r cccc@{}}",
        r"\toprule",
        r"\textbf{Split} & \textbf{Baseline} & $\boldsymbol{\Delta}$\textbf{PCC} & "
        r"\textbf{Rows (naive)} & \textbf{By drug} & \textbf{By cell} & \textbf{Two-way} \\",
        r"\midrule",
    ]
    for si, split in enumerate(SPLITS):
        for bi, b in enumerate(BASELINES):
            g = s0[(s0.split == split) & (s0.baseline == b)]
            if g.empty:
                continue
            def ci(sch):
                r = g[g.scheme == sch]
                if r.empty or np.isnan(r.iloc[0].lo):
                    return "n/a"
                r = r.iloc[0]
                star = r"$^{*}$" if not (r.lo <= 0 <= r.hi) else ""
                return f"$[{r.lo:+.3f},{r.hi:+.3f}]${star}"
            d = g.iloc[0].delta
            first = rf"\multirow{{3}}{{*}}{{{PRETTY[split]}}}" if bi == 0 else ""
            L.append(f"{first} & {MODEL_PRETTY[b]} & ${d:+.4f}$ & {ci('rows')} & "
                     f"{ci('drug')} & {ci('cell')} & {ci('twoway')} \\\\")
        if si < len(SPLITS) - 1:
            L.append(r"\addlinespace")
    L += [
        r"\bottomrule",
        r"\end{tabular*}",
        r"",
        r"\smallskip",
        r"\footnotesize\emph{Note:} $\Delta$PCC is the observed difference",
        r"(PathXDRP $-$ baseline) on the seed-0 test fold; positive favours",
        r"PathXDRP. Intervals are percentile bootstrap intervals except the",
        r"two-way column, which is a Wald interval built on the",
        r"Cameron--Gelbach--Miller variance",
        r"$\widehat{V}_{\mathrm{2way}}=\widehat{V}_{\mathrm{drug}}+"
        r"\widehat{V}_{\mathrm{cell}}-\widehat{V}_{\mathrm{rows}}$.",
        r"\end{table*}",
    ]
    (TAB / "tab_clustered_bootstrap.tex").write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {TAB/'tab_clustered_bootstrap.tex'}")


if __name__ == "__main__":
    main()
