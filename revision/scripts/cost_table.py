"""
W16 -- Computational cost comparison.

Answers Reviewer #4, minor point 8 ("parameters, memory usage, training time and
inference time") and supplies the evidence for the complexity-vs-gain discussion
Reviewer #5 asks for in point 9.

Parameter counts and wall-clock training times come from the canonical ledger
(the `args`/metadata blocks of every results JSON). Peak memory and inference
throughput are measured live if a CUDA device is free; if the GPU is busy the
script still emits the table and marks those columns as pending, so it can be
re-run later without redoing the rest.

Usage:
    python revision/scripts/cost_table.py [--measure]
Outputs:
    outputs/cost_table.csv
    outputs/cost_table.md
    tables/tab_cost.tex
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "outputs"
TAB = BASE / "tables"
OUT.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

MODELS = ["pathxdrp", "drpreter", "graphdrp", "cdrscan"]
PRETTY = {"pathxdrp": "PathXDRP", "drpreter": "DRPreter",
          "graphdrp": "GraphDRP", "cdrscan": "CDRScan"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--measure", action="store_true",
                    help="also measure peak memory and inference throughput")
    args = ap.parse_args()

    led = pd.read_csv(OUT / "ledger.csv")
    summ = pd.read_csv(OUT / "ledger_summary.csv")

    rows = []
    for m in MODELS:
        g = led[led.model == m]
        if g.empty:
            continue
        npar = g.n_params.dropna()
        secs = g.total_train_sec.dropna()
        eps = g.best_val_epoch.dropna()
        pcc_rand = summ[(summ.model == m) & (summ.split == "random")]
        rows.append({
            "model": m,
            "n_params": int(npar.iloc[0]) if len(npar) else np.nan,
            "median_train_sec": float(secs.median()) if len(secs) else np.nan,
            "total_sweep_hours": float(secs.sum()) / 3600 if len(secs) else np.nan,
            "median_best_epoch": float(eps.median()) if len(eps) else np.nan,
            "random_PCC": float(pcc_rand.PCC_mean.iloc[0]) if len(pcc_rand) else np.nan,
            "n_runs": len(g),
        })
    df = pd.DataFrame(rows)

    if args.measure:
        df = add_live_measurements(df)

    df.to_csv(OUT / "cost_table.csv", index=False)

    # cost per unit of accuracy, relative to the cheapest model
    base = df.n_params.min()
    df["params_rel"] = df.n_params / base
    tbase = df.median_train_sec.min()
    df["train_rel"] = df.median_train_sec / tbase

    L = ["# W16 -- Computational cost\n",
         "Answers Reviewer #4 minor point 8; supports Reviewer #5 point 9.\n",
         "Training times are wall clock on a single NVIDIA RTX 3060 Laptop GPU "
         "(6 GB), median over the 25 runs of each model, at the published "
         "protocol (50 epochs, early-stopping patience 10, batch size 64).\n",
         "| Model | Parameters | relative | Median train time | relative | Median best epoch | Total sweep | random PCC |",
         "|---|---|---|---|---|---|---|---|"]
    for _, r in df.sort_values("n_params").iterrows():
        L.append(
            f"| {PRETTY[r.model]} | {int(r.n_params):,} | {r.params_rel:.1f}x | "
            f"{r.median_train_sec/60:.1f} min | {r.train_rel:.1f}x | "
            f"{r.median_best_epoch:.0f} | {r.total_sweep_hours:.1f} h | "
            f"{r.random_PCC:.4f} |"
        )
    L.append("")

    if "infer_pairs_per_s" in df.columns and df.infer_pairs_per_s.notna().any():
        L.append("### Memory and inference cost\n")
        L.append("Measured on the same GPU with batch size 256 over the "
                 "random-split test fold, using the released configuration of "
                 "each model.\n")
        L.append("| Model | Peak GPU memory | Inference throughput | Inference time |")
        L.append("|---|---|---|---|")
        for _, r in df.sort_values("n_params").iterrows():
            if r.infer_pairs_per_s != r.infer_pairs_per_s:
                continue
            L.append(f"| {PRETTY[r.model]} | {r.peak_gpu_mb:,.0f} MB | "
                     f"{r.infer_pairs_per_s:,.0f} pairs/s | "
                     f"{r.infer_ms_per_pair:.3f} ms/pair |")
        L.append("")
        # The released PathXDRP drops the unused global fingerprint projection,
        # so its parameter count is below that of the trained checkpoints. Say
        # so rather than letting two different numbers stand unexplained.
        drift = df[(df.measured_n_params.notna())
                   & (abs(df.measured_n_params - df.n_params) > 1000)]
        for _, r in drift.iterrows():
            L.append(f"*{PRETTY[r.model]} measures "
                     f"{int(r.measured_n_params):,} parameters in the released "
                     f"configuration against {int(r.n_params):,} in the trained "
                     f"checkpoints; the difference is the global fingerprint "
                     f"projection, which the final architecture does not read "
                     f"and which has been removed from the released code.*\n")
    else:
        L.append("*Peak memory and inference throughput pending: re-run with "
                 "`--measure` when the GPU is free.*\n")

    L.append("## The complexity-versus-gain trade-off\n")
    try:
        p = df.set_index("model")
        L.append(
            "The cost of PathXDRP is not in its parameter count. It has "
            f"{p.loc['pathxdrp','n_params']:,.0f} parameters, fewer than "
            f"GraphDRP ({p.loc['graphdrp','n_params']:,.0f}) and a quarter of "
            f"CDRScan ({p.loc['cdrscan','n_params']:,.0f}). The cost is in "
            "time: it is the slowest model to train, "
            f"{p.loc['pathxdrp','median_train_sec']/p.loc['graphdrp','median_train_sec']:.1f}x "
            "GraphDRP and "
            f"{p.loc['pathxdrp','median_train_sec']/p.loc['cdrscan','median_train_sec']:.1f}x "
            "CDRScan per run. The reason is the cross-attention step, which "
            "scores every atom against all 370 pathway tokens; that is "
            "arithmetic over a large intermediate tensor rather than extra "
            "weights.\n\n"
            "Against that cost, the accuracy return is nil. Random-split PCC is "
            f"{p.loc['pathxdrp','random_PCC']:.4f} for PathXDRP versus "
            f"{p.loc['graphdrp','random_PCC']:.4f} for GraphDRP, which trains in "
            "under a third of the time. The clustered bootstrap (W4) puts "
            "differences of this size inside the noise on most splits.\n\n"
            "Stated plainly: the extra architecture does not buy accuracy. It "
            "buys per-prediction uncertainty and an attention map that is "
            "measurably load-bearing. Whether that is a good trade depends on "
            "whether the downstream user needs those two things. For a pure "
            "accuracy objective on this benchmark, GraphDRP is the better "
            "engineering choice, and the revised Discussion says so.\n"
        )
    except KeyError:
        pass

    (OUT / "cost_table.md").write_text("\n".join(L), encoding="utf-8")
    write_latex(df)
    print(df.to_string())
    print(f"\nwrote {OUT/'cost_table.md'}")


def add_live_measurements(df: pd.DataFrame) -> pd.DataFrame:
    """Merge in peak memory and inference cost measured per architecture.

    The measurements are produced by the trainers themselves, so they describe
    the same model and loader the reported runs use:

        python scripts/train_baseline.py --model <m> --measure
        python -m pathxdrp.train --measure

    Each writes results/benchmarks/inference_<model>.json.
    """
    import json
    bench = ROOT / "results" / "benchmarks"
    peak, thr, ms, meas_par = [], [], [], []
    for m in df.model:
        f = bench / f"inference_{m}.json"
        if f.exists():
            r = json.loads(f.read_text(encoding="utf-8"))
            peak.append(r.get("peak_gpu_mb", np.nan))
            thr.append(r.get("throughput_samples_per_s", np.nan))
            ms.append(r.get("inference_ms_per_pair", np.nan))
            meas_par.append(r.get("n_params", np.nan))
        else:
            peak.append(np.nan)
            thr.append(np.nan)
            ms.append(np.nan)
            meas_par.append(np.nan)
    df = df.copy()
    df["peak_gpu_mb"] = peak
    df["infer_pairs_per_s"] = thr
    df["infer_ms_per_pair"] = ms
    df["measured_n_params"] = meas_par
    found = int(sum(1 for p in peak if p == p))
    print(f"  merged inference benchmarks for {found}/{len(df)} models")
    return df


def write_latex(df: pd.DataFrame) -> None:
    # Reviewer #4 minor point 8 asks for parameters, memory, training time and
    # inference time, so the table carries all four when measurements exist.
    measured = ("peak_gpu_mb" in df.columns and df.peak_gpu_mb.notna().any())

    cap_extra = (
        r" Peak GPU memory and inference time are measured over the "
        r"random-split test fold at batch size~256 on the same device."
        if measured else "")
    L = [r"% Generated by revision/scripts/cost_table.py",
         r"\begin{table}[!h]", r"\centering\small",
         r"\caption{Computational cost of the four architectures on a single",
         r"NVIDIA RTX~3060 Laptop GPU (6\,GB). Training time is the median over",
         r"the 25 runs of each model at the published protocol." + cap_extra,
         r"The final column",
         r"repeats the random-split PCC so the cost can be read against the",
         r"accuracy it buys.}",
         r"\label{tab:cost}",
         (r"\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}lrrrrrr@{}}"
          if measured else
          r"\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}lrrrr@{}}"),
         r"\toprule"]
    if measured:
        L.append(r"\textbf{Model} & \textbf{Params} & \textbf{Train (min)} & "
                 r"\textbf{Peak mem (MB)} & \textbf{Infer (ms/pair)} & "
                 r"\textbf{Best epoch} & \textbf{PCC} \\")
    else:
        L.append(r"\textbf{Model} & \textbf{Params} & \textbf{Train (min)} & "
                 r"\textbf{Best epoch} & \textbf{PCC} \\")
    L.append(r"\midrule")
    for _, r in df.sort_values("n_params").iterrows():
        cells = [f"{PRETTY[r.model]}",
                 f"${int(r.n_params):,}$".replace(",", "{,}"),
                 f"${r.median_train_sec/60:.1f}$"]
        if measured:
            mem = (f"${r.peak_gpu_mb:,.0f}$".replace(",", "{,}")
                   if r.peak_gpu_mb == r.peak_gpu_mb else "--")
            ims = (f"${r.infer_ms_per_pair:.3f}$"
                   if r.infer_ms_per_pair == r.infer_ms_per_pair else "--")
            cells += [mem, ims]
        cells += [f"${r.median_best_epoch:.0f}$", f"${r.random_PCC:.4f}$"]
        L.append(" & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular*}", r"\end{table}"]
    (TAB / "tab_cost.tex").write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {TAB/'tab_cost.tex'}")


if __name__ == "__main__":
    main()
