"""
Publication-ready figures for the PathXDRP paper.

Figures
-------
fig1_split_comparison.pdf    Bar chart of test PCC by split for all models.
fig2_risk_coverage.pdf       Risk-coverage curves (selective prediction).
fig3_uncertainty_scatter.pdf Aleatoric vs. epistemic uncertainty per prediction.
fig4_xai_benchmark.pdf       Attribution-method comparison vs MoA ground truth.
fig5_calibration.pdf         Predicted-uncertainty bins vs empirical RMSE per bin.
fig6_per_drug_pcc.pdf        Violin of per-drug PCC across splits.
fig7_headline.pdf            Two-panel summary (split comparison + calibration).

Usage
-----
  python eval/plot_results.py                        # all figures
  python eval/plot_results.py --figures 1 5 6        # specific subset
  python eval/plot_results.py --out_dir figures/
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# Optional matplotlib import
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import matplotlib.patches as mpatches
    from matplotlib.colors import LinearSegmentedColormap
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    LinearSegmentedColormap = None  # type: ignore

ROOT        = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"

# ---
# Editorial Slate palette
# ---
# Earth-toned, mature, biological. Designed so that no one colour visually
# shouts on the page (similar saturation across the set), the focal method
# carries gravity (deep slate), and each colour stays distinct in luminance
# so the figures hold up in greyscale print.
# Q1-journal standard palette — Wong (2011) colorblind-safe, the
# de-facto recommendation of Nature Methods and widely adopted across
# Nature, Science, Cell, and PNAS. Precisely the "shade of orange and
# blue" visible in top-tier publications.
PALETTE = {
    "pathxdrp": "#0072B2",  # CB blue            — PathXDRP (focal)
    "drpreter": "#E69F00",  # CB orange          — DRPreter
    "graphdrp": "#009E73",  # CB bluish-green    — GraphDRP
    "cdrscan":  "#CC79A7",  # CB reddish-purple  — CDRScan
}

INK         = "#1A1A1A"   # body text, bar edges
INK_MUTED   = "#5C5C5C"   # secondary text, axis labels
INK_FAINT   = "#999999"   # reference lines (y=x, baselines)
GRID        = "#E0E0E0"   # gridlines
PANEL_FILL  = "#EBF4FB"   # light blue tint — ideal-quadrant highlight

if HAS_MPL:
    # Sequential: pale blue → mid → deep navy (perceptually uniform)
    WARM_CMAP = LinearSegmentedColormap.from_list(
        "q1_seq",
        ["#F7FBFF", "#9ECAE1", "#4292C6", "#08306B"],
        N=256,
    )
    # Diverging: deep blue ← neutral ← white → neutral → deep orange-red
    DIVERGING_CMAP = LinearSegmentedColormap.from_list(
        "q1_div",
        ["#053061", "#2166AC", "#DEEBF7", "#FEE0D2", "#D6604D", "#67001F"],
        N=256,
    )
else:
    WARM_CMAP = None
    DIVERGING_CMAP = None
MODEL_LABELS = {
    "pathxdrp": "PathXDRP",
    "drpreter": "DRPreter",
    "graphdrp": "GraphDRP",
    "cdrscan":  "CDRScan",
}
MODEL_ORDER  = ["pathxdrp", "drpreter", "graphdrp", "cdrscan"]
SPLIT_ORDER  = ["random", "cell_blind", "drug_blind", "scaffold_blind", "tissue_blind"]
SPLIT_LABELS = {
    "random":         "Random",
    "cell_blind":     "Cell-blind",
    "drug_blind":     "Drug-blind",
    "scaffold_blind": "Scaffold-blind",
    "tissue_blind":   "Tissue-blind",
}

if HAS_MPL:
    plt.rcParams.update({
        # --- Typography  (Q1 journal standard: 8–9 pt; NOT poster/slide sizes) ---
        "font.family":           "sans-serif",
        "font.sans-serif":       ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size":             8.5,
        "axes.labelsize":        9.0,
        "axes.labelweight":      "regular",
        "axes.titlesize":        9.0,
        "axes.titleweight":      "bold",
        # --- Legend (framed with a hairline border looks clean in print) ---
        "legend.fontsize":       7.5,
        "legend.frameon":        True,
        "legend.framealpha":     0.92,
        "legend.edgecolor":      "#CCCCCC",
        "legend.borderpad":      0.5,
        "legend.labelspacing":   0.32,
        "legend.handlelength":   1.5,
        "legend.handletextpad":  0.4,
        "legend.columnspacing":  1.2,
        # --- Tick marks ---
        "xtick.labelsize":       8.0,
        "ytick.labelsize":       8.0,
        "xtick.major.size":      3.5,
        "ytick.major.size":      3.5,
        "xtick.minor.size":      2.0,
        "ytick.minor.size":      2.0,
        "xtick.major.width":     0.7,
        "ytick.major.width":     0.7,
        "xtick.minor.width":     0.5,
        "ytick.minor.width":     0.5,
        "xtick.direction":       "out",
        "ytick.direction":       "out",
        "xtick.color":           INK,
        "ytick.color":           INK,
        # --- Axes frame ---
        "axes.edgecolor":        "#888888",
        "axes.labelcolor":       INK,
        "axes.linewidth":        0.75,
        "axes.spines.top":       False,
        "axes.spines.right":     False,
        # --- Grid (subtle guide, not dominant) ---
        "grid.color":            "#EBEBEB",
        "grid.linestyle":        "-",
        "grid.linewidth":        0.55,
        # --- Output ---
        "figure.dpi":            150,
        "savefig.dpi":           300,
        "savefig.bbox":          "tight",
        "savefig.pad_inches":    0.06,
        # --- Lines & patches ---
        "patch.linewidth":       0.6,
        "lines.linewidth":       1.5,
        "lines.markersize":      5,
    })

# Journal column widths (inches): single 3.46", double 7.09"
_FW = 7.09   # full  (double-column)
_HW = 3.46   # half  (single-column)
_FH = 3.2    # standard axis height for full-width figures
_HH = 3.3    # height for half-width square-ish figures

# Output format(s). Set by main() from --format. Default png because PDF needs
# a viewer; PNG renders inline in GitHub, slides, and chat. PDF is still ideal
# for the manuscript LaTeX include — pass --format both to write both.
SAVE_FORMATS: tuple[str, ...] = ("png",)


def _slug(s: str) -> str:
    """File-safe slug: lowercase, spaces/dashes -> underscore."""
    return s.lower().replace(" ", "_").replace("-", "_").replace("/", "_")


def _save(fig, out_path_no_ext: Path) -> list[Path]:
    """Write the current figure in every requested format. Returns paths."""
    written: list[Path] = []
    for ext in SAVE_FORMATS:
        p = out_path_no_ext.with_suffix(f".{ext}")
        fig.savefig(p)
        written.append(p)
    return written


def _log_saved(tag: str, written: list[Path], note: str = "") -> None:
    """Uniform per-figure log line: '<tag> -> stem [png+pdf] (note)'."""
    if not written:
        print(f"  {tag}: no output written {note}".rstrip())
        return
    stem = written[0].with_suffix("")
    exts = "+".join(p.suffix.lstrip(".") for p in written)
    suffix = f"  ({note})" if note else ""
    print(f"  {tag} -> {stem}.[{exts}]{suffix}")


def _panel_label(ax, label: str, x: float = -0.12, y: float = 1.04) -> None:
    """Place a bold panel label (a, b, …) in the upper-left corner of an axes.
    Uses transform=ax.transAxes so it is safe with constrained_layout."""
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="top", ha="right", color=INK)


def _note(ax, text: str, loc: str = "upper right", **kwargs) -> None:
    """Add a small italic inset annotation (e.g. n=237, PCC=0.74) at a corner."""
    xy_map = {
        "upper right":  (0.97, 0.97, "right", "top"),
        "upper left":   (0.03, 0.97, "left",  "top"),
        "lower right":  (0.97, 0.03, "right", "bottom"),
        "lower left":   (0.03, 0.03, "left",  "bottom"),
    }
    x, y, ha, va = xy_map.get(loc, (0.97, 0.97, "right", "top"))
    kw = dict(fontsize=7.5, color=INK_MUTED, fontstyle="italic",
              ha=ha, va=va, transform=ax.transAxes)
    kw.update(kwargs)
    ax.text(x, y, text, **kw)


# ---
# Data helpers
# ---

def load_metric(model: str, split: str, metric: str = "PCC",
                run_tag: str = "") -> list[float]:
    """Per-seed metric values for (model, split). Skips missing files.

    Parameters
    ----------
    run_tag : str
        If non-empty, only load files whose stem ends with ``_<run_tag>``
        (e.g. "v3" matches ``random_seed0_fold0_v3.json``).
        If empty (default), load the canonical files without any suffix
        (``random_seed*_fold0.json``).
    """
    vals = []
    model_dir = RESULTS_DIR / model
    if not model_dir.exists():
        return vals
    if run_tag:
        pattern = f"{split}_seed*_fold0_{run_tag}.json"
    else:
        pattern = f"{split}_seed*_fold0.json"
    for json_path in sorted(model_dir.glob(pattern)):
        # If no run_tag requested, skip versioned files (anything with an extra suffix)
        if not run_tag:
            tokens = json_path.stem.split("_")
            try:
                fold_idx = next(i for i, t in enumerate(tokens) if t.startswith("fold"))
            except StopIteration:
                continue
            if len(tokens) > fold_idx + 1:
                continue   # has extra suffix like _v3, _fp_nomask
        try:
            with open(json_path) as f:
                d = json.load(f)
            v = d.get("test", {}).get(metric)
            if v is not None and np.isfinite(v):
                vals.append(float(v))
        except (json.JSONDecodeError, OSError):
            continue
    return vals


def load_risk_coverage(model: str, split: str = "random", seed: int = 0):
    # Try requested seed first, then fall back to seeds 1-4
    candidates = [seed] + [s for s in range(1, 6) if s != seed]
    for s in candidates:
        p = RESULTS_DIR / model / f"{split}_seed{s}_fold0.json"
        # skip ablation variants (files with extra suffixes)
        if not p.exists():
            continue
        with open(p) as f:
            d = json.load(f)
        rc = d.get("test", {}).get("risk_coverage")
        if rc is None:
            continue
        covs  = rc.get("coverages")
        rmses = rc.get("rmses")
        if covs and rmses:
            return covs, rmses
    return None, None


def load_predictions(model: str, split: str = "random", seed: int = 0) -> pd.DataFrame | None:
    # Try requested seed first, then fall back to seeds 1-4
    candidates = [seed] + [s for s in range(1, 6) if s != seed]
    for s in candidates:
        p = RESULTS_DIR / model / f"{split}_seed{s}_fold0_preds.csv"
        if p.exists():
            return pd.read_csv(p)
    return None


# ---
# Figure 1: Split comparison
# ---

def fig_split_comparison(out_dir: Path, metric: str = "PCC",
                         splits: list[str] | None = None,
                         load_metric_fn=None) -> None:
    if not HAS_MPL:
        return
    splits = splits or SPLIT_ORDER
    _lm = load_metric_fn if load_metric_fn is not None else load_metric

    fig, ax = plt.subplots(figsize=(_FW, _FH + 0.3), layout="constrained")
    n_models = len(MODEL_ORDER)
    bar_w    = 0.72 / n_models
    x        = np.arange(len(splits))

    plotted = False
    for i, model in enumerate(MODEL_ORDER):
        means, stds = [], []
        for split in splits:
            vals = _lm(model, split, metric)
            means.append(np.mean(vals) if vals else np.nan)
            stds.append (np.std(vals)  if vals else 0.0)
        if all(np.isnan(m) for m in means):
            continue
        plotted = True
        offset = (i - (n_models - 1) / 2) * bar_w
        ax.bar(
            x + offset, means, bar_w,
            yerr=stds, capsize=3,
            color=PALETTE[model], label=MODEL_LABELS[model],
            alpha=0.85, error_kw={"elinewidth": 1.4, "ecolor": INK},
        )

    if not plotted:
        print(f"  fig1: no data for metric={metric}; skipping")
        plt.close(fig)
        return

    ax.set_xticks(x)
    ax.set_xticklabels([SPLIT_LABELS.get(s, s) for s in splits], rotation=0, ha="center")
    ax.set_ylabel(metric)
    bot, top = ax.get_ylim()
    ax.set_ylim(bottom=max(0.0, bot - 0.02), top=min(1.0, top) * 1.16)
    ax.legend(ncol=2, loc="upper left")
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.grid(axis="y", which="major")
    ax.set_axisbelow(True)

    stem = out_dir / f"fig1_split_comparison_{_slug(metric)}"
    written = _save(fig, stem)
    plt.close(fig)
    _log_saved("fig1", written)


# ---
# Figure 2: Risk-coverage curve
# ---

def fig_risk_coverage(out_dir: Path, split: str = "random") -> None:
    if not HAS_MPL:
        return

    fig, ax = plt.subplots(figsize=(_HW, _HH), layout="constrained")
    found_any = False
    # Only PathXDRP provides uncertainty estimates; plot baseline flat lines for reference
    for model in MODEL_ORDER:
        covs, rmses = load_risk_coverage(model, split=split)
        if covs is None:
            # For baselines, draw a flat horizontal RMSE line (no selective prediction)
            # using the 5-seed mean RMSE as the constant risk
            vals = load_metric(model, split, "RMSE")
            if vals:
                mean_rmse = float(np.mean(vals))
                ax.axhline(mean_rmse, color=PALETTE[model], linestyle="--",
                           linewidth=1.5, alpha=0.6, label=MODEL_LABELS[model])
                found_any = True
            continue
        ax.plot(covs, rmses, color=PALETTE[model], label=MODEL_LABELS[model],
                linewidth=2.5 if model == "pathxdrp" else 1.4,
                linestyle="-" if model == "pathxdrp" else "--", alpha=1.0)
        found_any = True

    if not found_any:
        ax.text(0.5, 0.5, "No risk-coverage data available yet.",
                ha="center", va="center", transform=ax.transAxes, color=INK_MUTED)
        ax.set_xticks([]); ax.set_yticks([])
    else:
        ax.set_xlabel("Coverage (fraction of predictions retained)")
        ax.set_ylabel("RMSE on retained predictions")
        ax.legend(loc="lower right")
        ax.grid(axis="y")
        ax.set_axisbelow(True)
        # Annotate the 50% coverage point for PathXDRP
        covs50, rmses50 = load_risk_coverage("pathxdrp", split=split)
        if covs50:
            idx = min(range(len(covs50)), key=lambda i: abs(covs50[i] - 0.50))
            ax.annotate(f"50% cov.\nRMSE={rmses50[idx]:.3f}",
                        xy=(covs50[idx], rmses50[idx]),
                        xytext=(covs50[idx] + 0.08, rmses50[idx] + 0.08),
                        arrowprops=dict(arrowstyle="->", color=INK_FAINT,
                                        lw=0.8),
                        fontsize=7, color=INK_MUTED)

    stem = out_dir / f"fig2_risk_coverage_{split}"
    written = _save(fig, stem)
    plt.close(fig)
    _log_saved("fig2", written, note="" if found_any else "placeholder; no data")


# ---
# Figure 3: Uncertainty scatter
# ---

def fig_uncertainty_scatter(out_dir: Path, split: str = "random", seed: int = 0) -> None:
    if not HAS_MPL:
        return
    df = load_predictions("pathxdrp", split=split, seed=seed)
    if df is None:
        print(f"  fig3: no predictions CSV for pathxdrp/{split}/seed{seed}; skipping")
        return
    if not {"aleatoric", "epistemic", "y_pred", "y_true"}.issubset(df.columns):
        print(f"  fig3: predictions CSV missing required columns; skipping")
        return

    err = np.abs(df["y_pred"] - df["y_true"])
    al  = df["aleatoric"].clip(lower=1e-8)
    ep  = df["epistemic"].clip(lower=1e-8)

    fig, ax = plt.subplots(figsize=(_HW, _HH), layout="constrained")
    sc = ax.scatter(al, ep, c=err, cmap=WARM_CMAP, s=3, alpha=0.5, rasterized=True)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
    cbar.outline.set_visible(False)
    cbar.set_label(r"|prediction error|  (LN IC$_{50}$)")
    ax.set_xlabel("Aleatoric uncertainty"); ax.set_ylabel("Epistemic uncertainty")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.grid(True, which="major")
    ax.set_axisbelow(True)

    stem = out_dir / f"fig3_uncertainty_scatter_{split}"
    written = _save(fig, stem)
    plt.close(fig)
    _log_saved("fig3", written)


# ---
# Figure 4: XAI benchmark
# ---

def fig_xai_benchmark(out_dir: Path, xai_json: str | Path | None = None) -> None:
    """Cross-model XAI comparison bar chart using xai_multimodel_summary.json."""
    if not HAS_MPL:
        return
    # Prefer the multimodel summary; fall back to legacy single-model file
    if xai_json:
        xai_path = Path(xai_json)
    else:
        xai_path = ROOT / "results" / "xai" / "xai_multimodel_summary.json"
        if not xai_path.exists():
            xai_path = ROOT / "results" / "xai" / "xai_benchmark_results.json"
    if not xai_path.exists():
        print(f"  fig4: no XAI file at {xai_path}; skipping")
        return

    with open(xai_path) as f:
        data = json.load(f)

    # multimodel_summary: {model: {metric: value, ...}, ...}
    # legacy single-model: {method: {metric: value, ...}, ...}
    is_multimodel = any(k in data for k in MODEL_ORDER)

    if is_multimodel:
        metrics = [
            ("ig_target_auroc_mean",            "IG AUROC\n(target genes)"),
            ("attn_target_auroc_mean",           "Attn AUROC\n(target genes)"),
            ("attn_sensitivity_alignment_mean",  "Sens. alignment\n(attn)"),
            ("attn_faithfulness_suff_mean",       "Faithfulness\n(sufficiency, ↓)"),
        ]
        models_present = [m for m in MODEL_ORDER if m in data]
        n_drugs = max((data[m].get("n_drugs_evaluated", 0) for m in models_present), default=0)
        x = np.arange(len(metrics))
        bar_w = 0.78 / max(len(models_present), 1)

        fig, ax = plt.subplots(figsize=(_FW, _FH + 0.2), layout="constrained")
        for i, model in enumerate(models_present):
            vals = [data[model].get(mk, np.nan) for mk, _ in metrics]
            offset = (i - (len(models_present) - 1) / 2) * bar_w
            ax.bar(x + offset, vals, bar_w,
                   label=MODEL_LABELS.get(model, model),
                   color=PALETTE.get(model, f"C{i}"),
                   edgecolor="none", linewidth=0)

        ax.set_xticks(x)
        ax.set_xticklabels([ml for _, ml in metrics])
        ax.set_ylabel("Score")
        ax.legend(ncol=2, loc="upper right")
        ax.set_ylim(0, 1.08)
        ax.axhline(0.5, color=INK_FAINT, linewidth=0.8, linestyle="--", zorder=0)
        ax.text(len(metrics) - 0.5, 0.515, "random baseline",
                color=INK_FAINT, fontsize=7, va="bottom", ha="right",
                style="italic")
        ax.grid(axis="y", which="major")
        ax.set_axisbelow(True)
        if n_drugs:
            ax.text(0.0, -0.16, f"n = {n_drugs} drugs",
                    transform=ax.transAxes, fontsize=7.5, color=INK_MUTED)

    else:
        # Legacy single-model format
        methods = list(data.keys())
        metrics_legacy = [
            ("target_auroc",        "Target AUROC"),
            ("sensitivity_alignment", "Sens. alignment"),
            ("faithfulness_suff",    "Faithfulness (Suff.)"),
            ("sparsity",             "Sparsity"),
        ]
        x = np.arange(len(metrics_legacy))
        bar_w = 0.7 / max(len(methods), 1)
        cmap  = plt.get_cmap("tab10")
        fig, ax = plt.subplots(figsize=(8.5, 4.8), layout="constrained")
        for i, method in enumerate(methods):
            vals = [data[method].get(mk, np.nan) for mk, _ in metrics_legacy]
            offset = (i - (len(methods) - 1) / 2) * bar_w
            ax.bar(x + offset, vals, bar_w, label=method, color=cmap(i), alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([ml for _, ml in metrics_legacy])
        ax.set_ylabel("Score")
        ax.legend(ncol=len(methods))
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", which="major")
        ax.set_axisbelow(True)

    stem = out_dir / "fig4_xai_benchmark"
    written = _save(fig, stem)
    plt.close(fig)
    _log_saved("fig4", written)


# ---
# Figure 5: Calibration plot
# ---

def fig_calibration(out_dir: Path, split: str = "random", seed: int = 0,
                    n_bins: int = 15) -> None:
    """Predicted-uncertainty deciles vs empirical RMSE in each decile.

    A perfectly calibrated regression model has y = x; the line should hug
    the diagonal. Deviations above the diagonal are overconfidence;
    deviations below are underconfidence.
    """
    if not HAS_MPL:
        return
    df = load_predictions("pathxdrp", split=split, seed=seed)
    if df is None:
        print(f"  fig5: no predictions CSV for pathxdrp/{split}/seed{seed}; skipping")
        return
    if not {"epistemic", "aleatoric", "y_pred", "y_true"}.issubset(df.columns):
        print(f"  fig5: predictions CSV missing required columns; skipping")
        return

    # Predicted std = sqrt(epistemic + aleatoric)
    pred_var = (df["epistemic"].clip(lower=0) + df["aleatoric"].clip(lower=0)).values
    pred_std = np.sqrt(pred_var)
    err      = np.abs(df["y_pred"] - df["y_true"]).values

    order = np.argsort(pred_std)
    bins  = np.array_split(order, n_bins)
    bin_pred = np.array([pred_std[b].mean() for b in bins if len(b) > 0])
    bin_emp  = np.array([np.sqrt((err[b]**2).mean()) for b in bins if len(b) > 0])

    fig, ax = plt.subplots(figsize=(_HW, _HH), layout="constrained")
    lim = max(bin_pred.max(), bin_emp.max()) * 1.05
    ax.plot([0, lim], [0, lim], "--", color=INK_FAINT, linewidth=1.2,
            label="Perfect calibration")
    ax.plot(bin_pred, bin_emp, "o-", color=PALETTE["pathxdrp"],
            linewidth=2.6, markersize=7, label="PathXDRP",
            markeredgecolor="white", markeredgewidth=1.0)
    ax.set_xlabel("Predicted standard deviation (binned)")
    ax.set_ylabel("Empirical RMSE in bin")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_aspect("equal")
    ax.legend(loc="upper left")
    ax.grid(True)
    ax.set_axisbelow(True)

    stem = out_dir / f"fig5_calibration_{split}"
    written = _save(fig, stem)
    plt.close(fig)
    _log_saved("fig5", written)


# ---
# Figure 6: Per-drug PCC violin across splits
# ---

def fig_per_drug_pcc(out_dir: Path, load_metric_fn=None) -> None:
    """Violin of per-drug PCC across splits for each model.

    This is the metric where the drug-blind story is most visible: even when
    overall PCC is mediocre on drug-blind, per-drug PCC tells you whether the
    model ranks cell lines correctly within a held-out drug.
    """
    if not HAS_MPL:
        return

    metric = "Per-drug PCC"
    _lm = load_metric_fn if load_metric_fn is not None else load_metric
    fig, ax = plt.subplots(figsize=(_FW, _FH + 0.3), layout="constrained")
    n_splits = len(SPLIT_ORDER)
    n_models = len(MODEL_ORDER)
    width    = 0.72 / n_models

    plotted_any = False
    legend_handles = []
    for i, model in enumerate(MODEL_ORDER):
        positions = []
        data      = []
        for j, split in enumerate(SPLIT_ORDER):
            vals = _lm(model, split, metric)
            if not vals:
                continue
            positions.append(j + (i - (n_models - 1) / 2) * width)
            data.append(vals)
        if not data:
            continue
        plotted_any = True
        # Use boxplot if too few seeds for a meaningful violin
        if all(len(d) < 4 for d in data):
            bp = ax.boxplot(data, positions=positions, widths=width * 0.85,
                            patch_artist=True, showfliers=False, manage_ticks=False)
            for box in bp["boxes"]:
                box.set(facecolor=PALETTE[model], alpha=0.85, edgecolor=INK)
            for med in bp["medians"]:
                med.set(color="white", linewidth=1.6)
        else:
            parts = ax.violinplot(data, positions=positions, widths=width * 0.9,
                                  showmeans=True, showextrema=False)
            for pc in parts["bodies"]:
                pc.set_facecolor(PALETTE[model])
                pc.set_alpha(0.85)
                pc.set_edgecolor(INK)
        legend_handles.append(plt.Rectangle((0, 0), 1, 1,
                                            color=PALETTE[model], alpha=0.85,
                                            label=MODEL_LABELS[model]))

    if not plotted_any:
        print(f"  fig6: no per-drug PCC data; skipping")
        plt.close(fig)
        return

    ax.set_xticks(np.arange(n_splits))
    ax.set_xticklabels([SPLIT_LABELS.get(s, s) for s in SPLIT_ORDER],
                       rotation=0, ha="center")
    ax.set_ylabel("Per-drug PCC")
    ax.axhline(0, color=INK_FAINT, linewidth=0.7, linestyle="-")
    ax.grid(axis="y", which="major")
    ax.set_axisbelow(True)
    ax.legend(handles=legend_handles, ncol=2, loc="upper left")

    stem = out_dir / "fig6_per_drug_pcc"
    written = _save(fig, stem)
    plt.close(fig)
    _log_saved("fig6", written)


# ---
# Figure 7: Headline two-panel summary
# ---

def fig_headline(out_dir: Path, load_metric_fn=None) -> None:
    """Two-panel summary: (left) PCC by split, (right) calibration plot.

    Designed as a candidate Figure 1 / abstract figure for the manuscript.
    """
    if not HAS_MPL:
        return

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(_FW, _FH),
                                             layout="constrained")

    # --- Left: split comparison (compact) ---
    n_models = len(MODEL_ORDER)
    bar_w    = 0.8 / n_models
    x        = np.arange(len(SPLIT_ORDER))
    plotted_left = False
    _lm = load_metric_fn if load_metric_fn is not None else load_metric
    for i, model in enumerate(MODEL_ORDER):
        means, stds = [], []
        for split in SPLIT_ORDER:
            vals = _lm(model, split, "PCC")
            means.append(np.mean(vals) if vals else np.nan)
            stds.append (np.std(vals)  if vals else 0.0)
        if all(np.isnan(m) for m in means):
            continue
        plotted_left = True
        offset = (i - (n_models - 1) / 2) * bar_w
        ax_left.bar(x + offset, means, bar_w, yerr=stds, capsize=2.5,
                    color=PALETTE[model], label=MODEL_LABELS[model],
                    edgecolor="none",
                    error_kw={"elinewidth": 1.0, "ecolor": INK_MUTED})

    ax_left.set_xticks(x)
    ax_left.set_xticklabels([SPLIT_LABELS[s] for s in SPLIT_ORDER],
                            rotation=0, ha="center")
    ax_left.set_ylabel("Test PCC")
    ax_left.grid(axis="y", which="major")
    ax_left.set_axisbelow(True)
    _panel_label(ax_left, "a")
    if plotted_left:
        ax_left.legend(ncol=2, loc="upper left")

    # --- Right: calibration ---
    df = load_predictions("pathxdrp", split="random", seed=0)
    plotted_right = False
    if df is not None and {"epistemic", "aleatoric", "y_pred", "y_true"}.issubset(df.columns):
        pred_var = (df["epistemic"].clip(lower=0) + df["aleatoric"].clip(lower=0)).values
        pred_std = np.sqrt(pred_var)
        err      = np.abs(df["y_pred"] - df["y_true"]).values
        order = np.argsort(pred_std)
        bins  = np.array_split(order, 15)
        bin_pred = np.array([pred_std[b].mean()           for b in bins if len(b) > 0])
        bin_emp  = np.array([np.sqrt((err[b]**2).mean())  for b in bins if len(b) > 0])
        lim = max(bin_pred.max(), bin_emp.max()) * 1.05
        ax_right.plot([0, lim], [0, lim], "--", color=INK_FAINT, linewidth=1.0,
                      label="Perfect calibration")
        ax_right.plot(bin_pred, bin_emp, "o-", color=PALETTE["pathxdrp"],
                      linewidth=2.0, markersize=5, label="PathXDRP",
                      markeredgecolor="white", markeredgewidth=0.8)
        ax_right.set_xlim(0, lim); ax_right.set_ylim(0, lim)
        ax_right.set_aspect("equal")
        ax_right.legend(loc="upper left")
        plotted_right = True

    ax_right.set_xlabel("Predicted standard deviation (binned)")
    ax_right.set_ylabel("Empirical RMSE in bin")
    ax_right.grid(True)
    ax_right.set_axisbelow(True)
    _panel_label(ax_right, "b")

    if not plotted_right:
        ax_right.text(0.5, 0.5, "Calibration data unavailable yet",
                      ha="center", va="center", transform=ax_right.transAxes,
                      color=INK_MUTED)

    stem = out_dir / "fig7_headline"
    written = _save(fig, stem)
    plt.close(fig)
    _log_saved("fig7", written)


# ---
# CLI
# ---

def fig_predicted_vs_actual(out_dir: Path, split: str = "random", seed: int = 0) -> None:
    """Scatter of predicted vs actual LN_IC50 for PathXDRP, coloured by density."""
    if not HAS_MPL:
        return
    df = load_predictions("pathxdrp", split=split, seed=seed)
    if df is None:
        print(f"  fig8: no predictions CSV for pathxdrp/{split}; skipping")
        return
    if not {"y_pred", "y_true"}.issubset(df.columns):
        print(f"  fig8: predictions CSV missing y_pred/y_true; skipping")
        return

    from scipy.stats import pearsonr
    pcc, _ = pearsonr(df["y_true"], df["y_pred"])

    try:
        from scipy.stats import gaussian_kde
        xy  = np.vstack([df["y_true"], df["y_pred"]])
        kde = gaussian_kde(xy, bw_method=0.15)
        c   = kde(xy)
        order = c.argsort()
        x_plot = df["y_true"].values[order]
        y_plot = df["y_pred"].values[order]
        c_plot = c[order]
    except Exception:
        x_plot = df["y_true"].values
        y_plot = df["y_pred"].values
        c_plot = None

    fig, ax = plt.subplots(figsize=(_HW, _HH + 0.1), layout="constrained")
    if c_plot is not None:
        sc = ax.scatter(x_plot, y_plot, c=c_plot, cmap=WARM_CMAP,
                        s=2, alpha=0.6, rasterized=True)
        cb = fig.colorbar(sc, ax=ax, label="Density", fraction=0.04, pad=0.02)
        cb.outline.set_visible(False)
    else:
        ax.scatter(x_plot, y_plot, s=2, alpha=0.4,
                   color=PALETTE["pathxdrp"], rasterized=True)
    lo = min(df["y_true"].min(), df["y_pred"].min()) - 0.3
    hi = max(df["y_true"].max(), df["y_pred"].max()) + 0.3
    ax.plot([lo, hi], [lo, hi], "--", color=INK_FAINT, linewidth=1.2,
            label="y = x")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel(r"Measured LN IC$_{50}$")
    ax.set_ylabel(r"Predicted LN IC$_{50}$")
    _note(ax, f"PCC = {pcc:.3f}", loc="upper left")
    ax.grid(True)
    ax.set_axisbelow(True)

    stem = out_dir / f"fig8_predicted_vs_actual_{split}"
    written = _save(fig, stem)
    plt.close(fig)
    _log_saved("fig8", written)


# ---
# Figure 9: Attention AUROC vs Faithfulness (XAI quadrant scatter)
# ---

def fig_xai_quadrant(out_dir: Path) -> None:
    """Single-pane XAI summary: each model is one dot in (target AUROC, faith).

    Tells the reader at a glance which model has attention that is BOTH well
    aligned with known targets AND load-bearing in the prediction. The
    top-right quadrant is what an interpretable DRP model should occupy.
    Models without attention output (CDRScan, GraphDRP) are omitted.
    """
    if not HAS_MPL:
        return
    xai_path = ROOT / "results" / "xai" / "xai_multimodel_summary.json"
    if not xai_path.exists():
        print(f"  fig9: no XAI summary at {xai_path}; skipping")
        return
    with open(xai_path) as f:
        data = json.load(f)

    pts = []
    for m in MODEL_ORDER:
        if m not in data:
            continue
        x = data[m].get("attn_target_auroc_mean")
        y = data[m].get("attn_faithfulness_comp_mean")  # comp: bigger = more faithful
        if x is None or y is None:
            continue
        if isinstance(x, float) and (x != x):  # NaN
            continue
        if isinstance(y, float) and (y != y):
            continue
        pts.append((m, x, y))
    if not pts:
        print("  fig9: no model has attention metrics; skipping")
        return

    fig, ax = plt.subplots(figsize=(_HW + 0.2, _HH + 0.2), layout="constrained")

    # Soft cream tint marks the "ideal interpretability" quadrant — the
    # eye lands on it first without competing with the data markers.
    ax.axhspan(ymin=0.5, ymax=1.05, xmin=0.5, xmax=1.0,
               color=PANEL_FILL, alpha=0.55, zorder=0)

    # Reference lines
    ax.axvline(0.5, color=INK_FAINT, linestyle="--", linewidth=0.9, zorder=1)
    ax.axhline(0.5, color=INK_FAINT, linestyle="--", linewidth=0.9, zorder=1)

    for m, x, y in pts:
        ax.scatter(x, y, s=220, color=PALETTE[m], edgecolor="white",
                   linewidth=1.8, label=MODEL_LABELS[m], zorder=3)

    label_offset = {
        "pathxdrp": ( 10,  10),
        "drpreter": ( 10,  10),
        "graphdrp": ( 10,  10),
        "cdrscan":  ( 10,  10),
    }
    for m, x, y in pts:
        dx, dy = label_offset.get(m, (8, 8))
        ha = "right" if dx < 0 else "left"
        va = "top"   if dy < 0 else "bottom"
        ax.annotate(MODEL_LABELS[m], xy=(x, y), xytext=(dx, dy),
                    textcoords="offset points", fontsize=8, color=INK,
                    ha=ha, va=va)

    # Quadrant guide labels
    ax.text(0.99, 0.97, "ideal", color=INK_MUTED, fontsize=7,
            fontstyle="italic", va="top", ha="right",
            transform=ax.transAxes)
    ax.text(0.505, 0.02, "AUROC = 0.5", color=INK_FAINT,
            fontsize=7, va="bottom", ha="left", rotation=90)
    ax.text(0.01,  0.505, "faithfulness = 0.5", color=INK_FAINT,
            fontsize=7, va="bottom", ha="left")

    ax.set_xlabel("Attention target-gene AUROC")
    ax.set_ylabel("Attention faithfulness (comprehensiveness)")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True)
    ax.set_axisbelow(True)
    stem = out_dir / "fig9_xai_quadrant"
    written = _save(fig, stem)
    plt.close(fig)
    _log_saved("fig9", written)


# ---
# Figure 10: per-drug XAI heatmap (curated MoA subset x models)
# ---

def fig_xai_per_drug_heatmap(out_dir: Path,
                              metric: str = "ig_target_auroc",
                              n_drugs: int = 30) -> None:
    """Drug x model heatmap of an XAI metric.

    Pulls per-drug records from results/xai/xai_multimodel_<model>.json. Sorts
    drugs by mean score across models so the most-explained drugs sit at the
    top. ``metric`` defaults to IG target AUROC (covers all 4 models);
    ``attn_target_auroc`` is also valid (only PathXDRP + DRPreter have it).
    """
    if not HAS_MPL:
        return
    xai_dir = ROOT / "results" / "xai"
    per_drug: dict[str, dict[str, float]] = {}  # {drug: {model: value}}
    models_with_data: list[str] = []
    for m in MODEL_ORDER:
        p = xai_dir / f"xai_multimodel_{m}.json"
        if not p.exists():
            continue
        with open(p) as f:
            d = json.load(f)
        kept = False
        for rec in d.get("per_drug", []):
            v = rec.get(metric)
            if v is None:
                continue
            try:
                v = float(v)
            except (TypeError, ValueError):
                continue
            if v != v:  # NaN
                continue
            per_drug.setdefault(rec["drug"], {})[m] = v
            kept = True
        if kept:
            models_with_data.append(m)

    if not per_drug or not models_with_data:
        print(f"  fig10: no per-drug XAI data for metric {metric}; skipping")
        return

    # Rank drugs by mean across the models that scored them
    def _drug_mean(drug: str) -> float:
        vs = [per_drug[drug][m] for m in models_with_data if m in per_drug[drug]]
        return float(np.mean(vs)) if vs else 0.0

    ranked = sorted(per_drug.keys(), key=_drug_mean, reverse=True)[:n_drugs]
    matrix = np.full((len(ranked), len(models_with_data)), np.nan)
    for i, drug in enumerate(ranked):
        for j, m in enumerate(models_with_data):
            if m in per_drug[drug]:
                matrix[i, j] = per_drug[drug][m]

    fig_h = max(4.0, 0.26 * len(ranked) + 1.2)
    fig, ax = plt.subplots(figsize=(1.6 + 1.8 * len(models_with_data), fig_h),
                           layout="constrained")

    # Clip the colourmap to the range where the data actually lives. The
    # full [0,1] scale wastes nearly half its dynamic range on values we
    # never see; [0.6,1.0] turns the heatmap into a real contrast surface.
    finite = matrix[np.isfinite(matrix)]
    vmin = max(0.55, float(np.nanpercentile(finite, 2)))  if finite.size else 0.55
    vmax = 1.0

    im = ax.imshow(matrix, aspect="auto", cmap=WARM_CMAP,
                   vmin=vmin, vmax=vmax)
    # Thin white grid between cells
    ax.set_xticks(np.arange(-.5, len(models_with_data), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(ranked), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.4)
    ax.tick_params(which="minor", length=0)

    ax.set_yticks(range(len(ranked)))
    ax.set_yticklabels(ranked, fontsize=7)
    ax.set_xticks(range(len(models_with_data)))
    ax.set_xticklabels([MODEL_LABELS[m] for m in models_with_data],
                       rotation=0, ha="center")
    ax.tick_params(axis="both", which="major", length=0)

    # Annotate cells. WARM_CMAP goes pale cream → ochre → terracotta → deep
    # slate as the value rises. White-on-slate reads well at the high end;
    # dark-on-cream reads well at the low end. Threshold at the midpoint.
    mid = (vmin + vmax) / 2 + 0.05
    for i in range(len(ranked)):
        for j in range(len(models_with_data)):
            v = matrix[i, j]
            if np.isnan(v):
                ax.text(j, i, "–", ha="center", va="center",
                        fontsize=9, color=INK_FAINT)
            else:
                txt_col = "white" if v > mid else INK
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=7, color=txt_col)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.outline.set_visible(False)
    metric_labels = {
        "ig_target_auroc":   "IG target-gene AUROC",
        "attn_target_auroc": "Attention target-gene AUROC",
    }
    cbar.set_label(metric_labels.get(metric, metric.replace("_", " ")))

    # Strip the heatmap spines
    for spine in ax.spines.values():
        spine.set_visible(False)
    stem = out_dir / f"fig10_per_drug_xai_{_slug(metric)}"
    written = _save(fig, stem)
    plt.close(fig)
    _log_saved("fig10", written)


# ---
# Figure 0: Data representation — three separate panels (SMILES, 2D, graph)
# ---

# Atom-element palette — Jmol/CPK convention, the universal colour code
# in chemistry. Reviewers (and any chemist reader) expect O to be red,
# N blue, S yellow, halogens green; deviating from this is a small but
# real reading cost. Carbon stays neutral grey as the structural backbone.
ELEM_COLOR = {
    "C":  "#4D4D4D",  # neutral charcoal — backbone
    "N":  "#3050F8",  # CPK blue
    "O":  "#FF0D0D",  # CPK red
    "F":  "#90E050",  # CPK green (halogen)
    "Cl": "#1FF01F",  # CPK green (halogen)
    "S":  "#E8C432",  # CPK yellow (slightly muted for print)
    "P":  "#FF8000",  # CPK orange
}
DEFAULT_ELEM_COLOR = "#9E9E9E"


def _load_fig0_drugs(drug_names: list[str] | None = None):
    """Helper: returns RDKit `Chem` module + a DataFrame of drug rows with
    SMILES and TARGET columns. Returns (None, None) if dependencies or
    data files are missing.
    """
    try:
        from rdkit import Chem  # noqa: F401
    except Exception:
        print("  fig0: RDKit not available; skipping")
        return None, None
    drug_names = drug_names or ["Erlotinib", "Olaparib", "Trametinib", "Dabrafenib"]
    parquet = ROOT / "data" / "processed" / "drugs_with_smiles.parquet"
    if not parquet.exists():
        print(f"  fig0: drugs_with_smiles.parquet missing; skipping")
        return None, None
    import pandas as pd
    drugs = pd.read_parquet(parquet).drop_duplicates("DRUG_NAME")
    rows  = drugs[drugs.DRUG_NAME.isin(drug_names)].drop_duplicates("DRUG_NAME")
    rows  = rows.set_index("DRUG_NAME").reindex(drug_names).dropna(subset=["SMILES"])
    if rows.empty:
        print("  fig0: no matching drugs found; skipping")
        return None, None
    return Chem, rows


def fig_data_smiles(out_dir: Path,
                    drug_names: list[str] | None = None) -> None:
    """Figure 0a — canonical SMILES strings for the four reference drugs."""
    if not HAS_MPL:
        return
    import textwrap
    loaded = _load_fig0_drugs(drug_names)
    if loaded[0] is None:
        return
    _Chem, rows = loaded

    n = len(rows)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.4), layout="constrained")
    if n == 1:
        axes = [axes]
    for ax, (name, row) in zip(axes, rows.iterrows()):
        smiles = str(row["SMILES"])
        target = str(row.get("TARGET", "")) or "—"
        ax.axis("off")
        ax.text(0.5, 0.93, name, transform=ax.transAxes,
                ha="center", va="top", fontsize=9, fontweight="bold",
                color=INK)
        wrapped = "\n".join(textwrap.wrap(smiles, width=28)) or smiles
        ax.text(0.5, 0.55, wrapped,
                ha="center", va="center", fontsize=8,
                family="monospace", color=INK,
                transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.5", fc="#F4EEE2",
                          ec="#D6CFC0", lw=0.8))
        ax.text(0.5, 0.10, f"target: {target}",
                ha="center", va="top", fontsize=8,
                color=INK_MUTED, style="italic",
                transform=ax.transAxes)

    stem = out_dir / "fig0a_data_smiles"
    written = _save(fig, stem)
    plt.close(fig)
    _log_saved("fig0a", written)


def fig_data_structure(out_dir: Path,
                       drug_names: list[str] | None = None) -> None:
    """Figure 0b — RDKit 2D structural drawings of the reference drugs."""
    if not HAS_MPL:
        return
    loaded = _load_fig0_drugs(drug_names)
    if loaded[0] is None:
        return
    Chem, rows = loaded
    from rdkit.Chem import AllChem, Draw

    n = len(rows)
    fig, axes = plt.subplots(1, n, figsize=(3.4 * n, 4.5), layout="constrained")
    if n == 1:
        axes = [axes]
    for ax, (name, row) in zip(axes, rows.iterrows()):
        smiles = str(row["SMILES"])
        ax.axis("off")
        ax.text(0.5, 0.98, name, transform=ax.transAxes,
                ha="center", va="top", fontsize=9, fontweight="bold",
                color=INK)
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            ax.text(0.5, 0.5, "structure render failed",
                    ha="center", va="center",
                    transform=ax.transAxes, color=INK_FAINT)
            continue
        try:
            AllChem.Compute2DCoords(mol)
            img = Draw.MolToImage(mol, size=(720, 720),
                                  kekulize=True, wedgeBonds=True)
            ax.imshow(img)
        except Exception:
            ax.text(0.5, 0.5, "structure render failed",
                    ha="center", va="center",
                    transform=ax.transAxes, color=INK_FAINT)

    stem = out_dir / "fig0b_data_structure"
    written = _save(fig, stem)
    plt.close(fig)
    _log_saved("fig0b", written)


def fig_data_graph(out_dir: Path,
                   drug_names: list[str] | None = None) -> None:
    """Figure 0c — node–edge graph view of the reference drugs."""
    if not HAS_MPL:
        return
    loaded = _load_fig0_drugs(drug_names)
    if loaded[0] is None:
        return
    Chem, rows = loaded
    from rdkit.Chem import AllChem

    n = len(rows)
    fig, axes = plt.subplots(1, n, figsize=(3.4 * n, 4.8), layout="constrained")
    if n == 1:
        axes = [axes]
    for ax, (name, row) in zip(axes, rows.iterrows()):
        smiles = str(row["SMILES"])
        ax.text(0.5, 1.02, name, transform=ax.transAxes,
                ha="center", va="bottom", fontsize=9, fontweight="bold",
                color=INK)
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            ax.text(0.5, 0.5, "graph render failed",
                    ha="center", va="center",
                    transform=ax.transAxes, color=INK_FAINT)
            ax.axis("off")
            continue
        try:
            AllChem.Compute2DCoords(mol)
            conf = mol.GetConformer()
            xs = [conf.GetAtomPosition(i).x for i in range(mol.GetNumAtoms())]
            ys = [conf.GetAtomPosition(i).y for i in range(mol.GetNumAtoms())]
            for b in mol.GetBonds():
                a, c = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
                bt = b.GetBondTypeAsDouble()
                lw = 1.6 if bt == 1.0 else 2.6 if bt == 2.0 else 3.2
                ax.plot([xs[a], xs[c]], [ys[a], ys[c]],
                        color=INK_FAINT, linewidth=lw, alpha=0.85, zorder=1)
            for i, atom in enumerate(mol.GetAtoms()):
                sym = atom.GetSymbol()
                col = ELEM_COLOR.get(sym, DEFAULT_ELEM_COLOR)
                ax.scatter(xs[i], ys[i], s=340, color=col,
                           edgecolor="white", linewidth=1.4, zorder=3)
                ax.text(xs[i], ys[i], sym, ha="center", va="center",
                        fontsize=9, fontweight="bold", color="white",
                        zorder=4)
            ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            n_atoms = mol.GetNumAtoms()
            n_bonds = mol.GetNumBonds()
            ax.text(0.5, -0.04, f"{n_atoms} atoms · {n_bonds} bonds",
                    transform=ax.transAxes, ha="center", va="top",
                    fontsize=7.5, color=INK_MUTED)
        except Exception:
            ax.text(0.5, 0.5, "graph render failed",
                    ha="center", va="center",
                    transform=ax.transAxes, color=INK_FAINT)
            ax.axis("off")

    legend_handles = [
        plt.Line2D([], [], marker="o", color=col, linestyle="",
                   markersize=8, markeredgecolor="white",
                   markeredgewidth=0.8, label=elem)
        for elem, col in ELEM_COLOR.items()
    ]
    fig.legend(handles=legend_handles, loc="lower center",
               ncol=len(ELEM_COLOR), handletextpad=0.3, columnspacing=1.2)

    stem = out_dir / "fig0c_data_graph"
    written = _save(fig, stem)
    plt.close(fig)
    _log_saved("fig0c", written)


# ---
# Figure 11: ROAR-style faithfulness curve (comprehensiveness vs K%)
# ---

def fig_faithfulness_curve(out_dir: Path) -> None:
    """Mean attention-faithfulness comprehensiveness as K% of top features
    are removed, swept across K in {5, 10, 20, 30, 50}%.

    Pulls per-drug ``faith_curve_comp_deltas`` from each model's
    ``xai_modelagnostic_<model>.json`` and averages over the curated MoA
    drugs. This is the cleanest single-figure summary of the
    ROAR-style XAI contribution the paper introduces but currently has no
    figure for — a faithful explanation gives a monotone increasing curve;
    a hollow attention path produces a flat curve.
    """
    if not HAS_MPL:
        return
    xai_dir = ROOT / "results" / "xai"
    K_PCT = np.array([5, 10, 20, 30, 50])

    fig, ax = plt.subplots(figsize=(4.5, _HH - 0.2), layout="constrained")
    plotted_any = False
    for m in MODEL_ORDER:
        p = xai_dir / f"xai_modelagnostic_{m}.json"
        if not p.exists():
            continue
        with open(p) as f:
            d = json.load(f)
        records = d.get("per_drug", [])
        # Collect per-drug 5-element curves; skip drugs whose curve is missing
        # or has the wrong length.
        curves = []
        for rec in records:
            c = rec.get("faith_curve_comp_deltas")
            if isinstance(c, list) and len(c) == len(K_PCT):
                arr = np.array(c, dtype=float)
                if np.all(np.isfinite(arr)):
                    curves.append(arr)
        if not curves:
            continue
        M = np.stack(curves)                       # (n_drugs, 5)
        mean_curve = M.mean(axis=0)
        sem_curve  = M.std(axis=0) / np.sqrt(M.shape[0])

        ax.plot(K_PCT, mean_curve, "o-",
                color=PALETTE[m], linewidth=2.0, markersize=5,
                label=f"{MODEL_LABELS[m]}  (n={M.shape[0]})",
                markeredgecolor="white", markeredgewidth=0.8)
        plotted_any = True

    if not plotted_any:
        print("  fig11: no faithfulness-curve data; skipping")
        plt.close(fig)
        return

    ax.set_xlabel("Top-K% of attributed features removed")
    ax.set_ylabel(r"|$\Delta f$| (LN IC$_{50}$ shift after masking)")
    ax.set_xticks(K_PCT)
    ax.set_xticklabels([f"{k}%" for k in K_PCT])
    ax.grid(True)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right")
    _note(ax, "more faithful ↑", loc="upper right")

    stem = out_dir / "fig11_faithfulness_curve"
    written = _save(fig, stem)
    plt.close(fig)
    _log_saved("fig11", written)


# ---
# Figure 1-alt A: Lollipop (horizontal dot-and-stem)
# ---

def _collect_means_stds(load_fn, metric: str, splits: list) -> tuple[dict, dict]:
    """Collect per-model {split: mean} and {split: std} dicts."""
    means: dict[str, list[float]] = {}
    stds:  dict[str, list[float]] = {}
    for model in MODEL_ORDER:
        mu_list, sd_list = [], []
        for split in splits:
            vals = load_fn(model, split, metric)
            if vals:
                mu_list.append(float(np.mean(vals)))
                sd_list.append(float(np.std(vals)))
            else:
                mu_list.append(float("nan"))
                sd_list.append(0.0)
        if any(not np.isnan(v) for v in mu_list):
            means[model] = mu_list
            stds[model]  = sd_list
    return means, stds


def fig_split_comparison_dot(
    out_dir: Path,
    metric: str = "PCC",
    load_metric_fn=None,
) -> None:
    """Lollipop alternative to grouped bar chart (fig1).
    Each model's mean is shown as a dot with error bar on a horizontal stem."""
    if load_metric_fn is None:
        load_metric_fn = load_metric
    means, stds = _collect_means_stds(load_metric_fn, metric, SPLIT_ORDER)
    if not means:
        print(f"  fig1b_lollipop ({metric}): no data; skipping")
        return
    n_splits = len(SPLIT_ORDER)
    n_models = len(MODEL_ORDER)
    offsets  = np.linspace(-0.30, 0.30, n_models)
    fig, ax  = plt.subplots(figsize=(_FW, max(3.0, 0.9 * n_splits + 0.8)),
                            layout="constrained")
    for mi, model in enumerate(MODEL_ORDER):
        if model not in means:
            continue
        col = PALETTE[model]
        for si, split in enumerate(SPLIT_ORDER):
            mu  = means[model][si]
            err = stds[model][si]
            if np.isnan(mu):
                continue
            y = si + offsets[mi]
            ax.hlines(y, 0, mu, color=col, linewidth=1.4, alpha=0.55)
            ax.errorbar(mu, y, xerr=err, fmt="o", color=col, markersize=7,
                        capsize=3, elinewidth=1.2,
                        markeredgewidth=0.5, markeredgecolor="white",
                        label=MODEL_LABELS[model] if si == 0 else "_nolegend_")
    ax.set_yticks(range(n_splits))
    ax.set_yticklabels([SPLIT_LABELS.get(s, s) for s in SPLIT_ORDER])
    ax.set_xlabel(metric)
    ax.axvline(0, color=INK_FAINT, linewidth=0.7)
    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.grid(axis="x", which="major")
    ax.set_axisbelow(True)
    ax.legend(ncol=2, loc="lower right")
    safe = metric.replace(" ", "_").lower()
    stem = out_dir / f"fig1b_lollipop_{safe}"
    written = _save(fig, stem)
    plt.close(fig)
    _log_saved(f"fig1b_lollipop_{metric}", written)


# ---
# Figure 1-alt B: Radar / spider chart
# ---

def fig_split_radar(
    out_dir: Path,
    metric: str = "PCC",
    load_metric_fn=None,
) -> None:
    """Radar chart showing each model across all 5 splits."""
    if load_metric_fn is None:
        load_metric_fn = load_metric
    means, _ = _collect_means_stds(load_metric_fn, metric, SPLIT_ORDER)
    if not means:
        print(f"  fig1c_radar ({metric}): no data; skipping")
        return
    n = len(SPLIT_ORDER)
    labels = [SPLIT_LABELS.get(s, s) for s in SPLIT_ORDER]
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(3.8, 3.8), subplot_kw={"polar": True},
                           layout="constrained")
    for model in MODEL_ORDER:
        if model not in means or len(means[model]) < n:
            continue
        vals = [v if not np.isnan(v) else 0.0 for v in means[model]]
        vals_closed = vals + [vals[0]]
        ax.plot(angles, vals_closed, "o-", linewidth=2, markersize=5,
                color=PALETTE[model], label=MODEL_LABELS[model])
        ax.fill(angles, vals_closed, alpha=0.08, color=PALETTE[model])
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.legend(ncol=2, loc="upper right")
    safe = metric.replace(" ", "_").lower()
    stem = out_dir / f"fig1c_radar_{safe}"
    written = _save(fig, stem)
    plt.close(fig)
    _log_saved(f"fig1c_radar_{metric}", written)


# ---
# Figure 4-alt: Horizontal Cleveland dot chart
# ---

def fig_xai_benchmark_dot(out_dir: Path) -> None:
    """Cleveland dot chart alternative to fig4 bar chart."""
    xai_path = RESULTS_DIR / "xai" / "xai_multimodel_summary.json"
    if not xai_path.exists():
        print("  fig4b_dot: xai_multimodel_summary.json not found; skipping")
        return
    with open(xai_path) as fh:
        data = json.load(fh)
    metrics = [
        ("ig_target_auroc",   "IG AUROC"),
        ("attn_target_auroc", "Attn AUROC"),
        ("faithfulness_comp", "Faithfulness"),
    ]
    models_present = [m for m in MODEL_ORDER if m in data]
    if not models_present:
        print("  fig4b_dot: no models found; skipping")
        return
    n_met = len(metrics)
    n_mod = len(models_present)
    fig, axes = plt.subplots(1, n_met,
                             figsize=(_FW, max(2.4, 0.55 * n_mod + 1.2)),
                             sharey=True, layout="constrained")
    if n_met == 1:
        axes = [axes]
    for ci, (met_key, met_label) in enumerate(metrics):
        ax = axes[ci]
        for mi, model in enumerate(models_present):
            mdata = data[model]
            mu = mdata.get(met_key, None)
            sd = mdata.get(met_key + "_std", 0.0)
            if mu is None:
                continue
            ax.barh(mi, float(mu), height=0.55, color=PALETTE[model], alpha=0.82,
                    xerr=float(sd), capsize=3,
                    error_kw={"elinewidth": 1.2, "ecolor": INK})
        ax.set_yticks(range(n_mod))
        ax.set_yticklabels([MODEL_LABELS[m] for m in models_present] if ci == 0 else [])
        ax.set_xlabel(met_label)
        ax.set_xlim(0, 1.0)
        ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())
        ax.grid(axis="x", which="major")
        ax.set_axisbelow(True)
    stem = out_dir / "fig4b_xai_dotchart"
    written = _save(fig, stem)
    plt.close(fig)
    _log_saved("fig4b_xai_dotchart", written)


# ---
# Figure 6-alt: Strip + box plot
# ---

def fig_per_drug_pcc_strip(out_dir: Path, load_metric_fn=None) -> None:
    """Strip + box plot alternative to per-drug PCC violin (fig6)."""
    if load_metric_fn is None:
        load_metric_fn = load_metric
    fig, ax = plt.subplots(figsize=(_FW, _FH), layout="constrained")
    plotted = False
    legend_handles = []
    rng = np.random.default_rng(42)
    for mi, model in enumerate(MODEL_ORDER):
        all_vals: list[float] = []
        for split in SPLIT_ORDER:
            all_vals.extend(load_metric_fn(model, split, "Per-drug PCC"))
        if not all_vals:
            continue
        plotted = True
        col = PALETTE[model]
        arr = np.array(all_vals, dtype=float)
        jitter = rng.uniform(-0.20, 0.20, size=len(arr))
        ax.scatter(arr, mi + jitter, s=6, color=col, alpha=0.35,
                   linewidths=0, zorder=2)
        q1, q2, q3 = np.percentile(arr, [25, 50, 75])
        iqr = q3 - q1
        lo = max(arr.min(), q1 - 1.5 * iqr)
        hi = min(arr.max(), q3 + 1.5 * iqr)
        ax.broken_barh([(q1, iqr)], (mi - 0.28, 0.56),
                       facecolors="none", edgecolors=col, linewidth=1.8, zorder=3)
        ax.hlines(mi, lo, hi, color=col, linewidth=1.2, zorder=3)
        ax.vlines(q2, mi - 0.28, mi + 0.28, color=col, linewidth=2.2, zorder=4)
        legend_handles.append(mpatches.Patch(facecolor=col, label=MODEL_LABELS[model]))
    if not plotted:
        plt.close(fig)
        return
    ax.set_yticks(range(len(MODEL_ORDER)))
    ax.set_yticklabels([MODEL_LABELS[m] for m in MODEL_ORDER])
    ax.set_xlabel("Per-drug PCC")
    ax.axvline(0, color=INK_FAINT, linewidth=0.7)
    ax.grid(axis="x", which="major")
    ax.set_axisbelow(True)
    ax.legend(handles=legend_handles, ncol=2, loc="upper left")
    stem = out_dir / "fig6b_per_drug_strip"
    written = _save(fig, stem)
    plt.close(fig)
    _log_saved("fig6b_per_drug_strip", written)


# ---
# Figure 11-alt: Faithfulness filled-area chart
# ---

def fig_faithfulness_area(out_dir: Path) -> None:
    """Filled-area faithfulness CDF with +-std ribbon (alternative to fig11)."""
    xai_path = RESULTS_DIR / "xai" / "xai_multimodel_pathxdrp.json"
    if not xai_path.exists():
        print("  fig11b_area: no per-drug XAI data; skipping")
        return
    with open(xai_path) as fh:
        drug_data = json.load(fh)
    vals = [float(d.get("attn_faithfulness_comp"))
            for d in drug_data.get("per_drug", drug_data) if isinstance(d, dict)
            and d.get("attn_faithfulness_comp") is not None]
    if len(vals) < 5:
        print(f"  fig11b_area: only {len(vals)} faithfulness values; skipping")
        return
    arr = np.array(sorted(vals), dtype=float)
    n   = len(arr)
    cum = np.arange(1, n + 1) / n
    w   = max(1, n // 10)
    smooth = np.convolve(arr, np.ones(w) / w, mode="same")
    ribbon = np.array([arr[max(0, i - w):min(n, i + w)].std() for i in range(n)])
    fig, ax = plt.subplots(figsize=(_HW, _HH - 0.3), layout="constrained")
    ax.fill_betweenx(cum, smooth - ribbon, smooth + ribbon,
                     color=PALETTE["pathxdrp"], alpha=0.20, label="+/-1 SD ribbon")
    ax.plot(smooth, cum, color=PALETTE["pathxdrp"], linewidth=2.2,
            label="PathXDRP (smoothed)")
    ax.axvline(0,   color=INK_FAINT, linewidth=0.8, linestyle="--")
    ax.axvline(0.5, color=INK_FAINT, linewidth=0.7, linestyle=":")
    ax.set_xlabel("Faithfulness score (per drug)")
    ax.set_ylabel("Cumulative fraction of drugs")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(0, 1.05)
    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.grid(axis="both", which="major")
    ax.set_axisbelow(True)
    ax.legend(loc="upper left")
    stem = out_dir / "fig11b_faithfulness_area"
    written = _save(fig, stem)
    plt.close(fig)
    _log_saved("fig11b_faithfulness_area", written)


# ---
# Figure 12: Attention matrix  (12a heatmap / 12b frequency / 12c scatter)
# ---

def fig_attention_matrix(out_dir: Path) -> None:
    """Three-panel attention figure from XAI pipeline.

    12a  Drug x pathway heatmap (rank-weighted score).
    12b  Pathway frequency bar (how many drugs rank each pathway in top-5).
    12c  Entropy vs cosine-similarity scatter per drug.
    """
    diag_path = RESULTS_DIR / "xai" / "attention_diagnostic.json"
    if not diag_path.exists():
        print("  fig12: attention_diagnostic.json not found; skipping")
        return
    with open(diag_path) as fh:
        diag = json.load(fh)
    per_drug_list = diag.get("per_drug", [])
    if not per_drug_list:
        print("  fig12: per_drug section empty; skipping")
        return

    # Build score matrix (rows = drugs, cols = pathways)
    all_paths: set[str] = set()
    for dd in per_drug_list:
        for pname in dd.get("top5_pathways", []):
            if isinstance(pname, str) and pname:
                all_paths.add(pname)
    pathway_list = sorted(all_paths)
    drug_list    = [dd.get("drug", dd.get("drug_id", f"drug_{i}"))
                    for i, dd in enumerate(per_drug_list)]
    path_idx     = {p: i for i, p in enumerate(pathway_list)}
    n_drugs      = len(drug_list)
    n_paths      = len(pathway_list)
    score_mat    = np.zeros((n_drugs, n_paths), dtype=float)
    path_freq: dict[str, int] = {p: 0 for p in pathway_list}

    for di, dd in enumerate(per_drug_list):
        top = dd.get("top5_pathways", [])
        n_top = len(top)
        for rank, pname in enumerate(top, start=1):
            if isinstance(pname, str) and pname in path_idx:
                score_mat[di, path_idx[pname]] = 1.0 - (rank - 1) / max(n_top, 1)
                path_freq[pname] += 1

    # Sort pathways by frequency
    sorted_paths = sorted(pathway_list, key=lambda p: -path_freq[p])
    sorted_idx   = [path_idx[p] for p in sorted_paths]
    mat_sorted   = score_mat[:, sorted_idx]

    # Sort drugs by total score
    drug_order  = np.argsort(-mat_sorted.sum(axis=1))
    mat_sorted  = mat_sorted[drug_order, :]
    drug_sorted = [drug_list[i] for i in drug_order]

    # Trim pathways for readability
    n_show  = min(40, n_paths)
    mat_show = mat_sorted[:, :n_show]
    plabels  = sorted_paths[:n_show]
    freqs    = [path_freq[p] for p in plabels]

    # ---- 12a heatmap ----
    h_heat = max(6, 0.22 * n_drugs + 2)
    w_heat = max(10, 0.22 * n_show + 2)
    fig_a, ax_h = plt.subplots(figsize=(w_heat, h_heat), layout="constrained")
    im = ax_h.imshow(mat_show, aspect="auto", cmap=WARM_CMAP,
                     interpolation="nearest", vmin=0, vmax=1)
    ax_h.set_yticks(range(n_drugs))
    ax_h.set_yticklabels(drug_sorted, fontsize=6.5)
    ax_h.set_xticks(range(n_show))
    ax_h.set_xticklabels(plabels, rotation=50, ha="right", fontsize=6.5)
    ax_h.set_xlabel("Pathway (sorted by cross-drug frequency)")
    ax_h.set_ylabel("Drug")
    cb = fig_a.colorbar(im, ax=ax_h, fraction=0.025, pad=0.02)
    cb.outline.set_visible(False)
    cb.set_label("Rank score  (1.0 = top, 0 = absent)")
    stem_a = out_dir / "fig12a_attention_heatmap"
    _log_saved("fig12a", _save(fig_a, stem_a))
    plt.close(fig_a)

    # ---- 12b pathway frequency bar ----
    top_n = min(30, n_show)
    w_bar = max(8, 0.30 * top_n + 1.5)
    fig_b, ax_b = plt.subplots(figsize=(w_bar, 3.5), layout="constrained")
    colors_bar = [PALETTE["pathxdrp"] if freqs[i] >= 3 else "#9ECAE1"
                  for i in range(top_n)]
    ax_b.bar(np.arange(top_n), freqs[:top_n], color=colors_bar,
             width=0.68, edgecolor="none")
    ax_b.set_xticks(np.arange(top_n))
    ax_b.set_xticklabels(plabels[:top_n], rotation=50, ha="right", fontsize=7)
    ax_b.set_ylabel("Number of drugs with pathway in top-5")
    ax_b.axhline(3, color=INK_FAINT, linewidth=0.8, linestyle="--",
                 label="≥3 drugs threshold")
    ax_b.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax_b.grid(axis="y", which="major")
    ax_b.set_axisbelow(True)
    ax_b.legend(loc="upper right")
    stem_b = out_dir / "fig12b_pathway_frequency"
    _log_saved("fig12b", _save(fig_b, stem_b))
    plt.close(fig_b)

    # ---- 12c entropy vs cosine scatter ----
    entropies, cosims, names_s = [], [], []
    for dd in per_drug_list:
        ent = dd.get("drug_attn_entropy", None)
        cos = dd.get("within_atom_cosine", None)
        if ent is not None and cos is not None:
            entropies.append(float(ent))
            cosims.append(float(cos))
            names_s.append(dd.get("drug", ""))
    if len(entropies) >= 4:
        ent_a = np.array(entropies)
        cos_a = np.array(cosims)
        norm_ = (ent_a - ent_a.min()) / (ent_a.ptp() + 1e-9)
        fig_c, ax_c = plt.subplots(figsize=(_HW, _HH), layout="constrained")
        sc = ax_c.scatter(cos_a, ent_a, c=norm_, cmap=WARM_CMAP,
                          s=12, alpha=0.80, linewidths=0.0,
                          edgecolors="none")
        cb = fig_c.colorbar(sc, ax=ax_c, fraction=0.03, pad=0.02)
        cb.outline.set_visible(False)
        cb.set_label("Entropy (normalised)")
        ax_c.set_xlabel("Within-drug attention cosine similarity")
        ax_c.set_ylabel("Attention entropy (nats)")
        med_c, med_e = np.median(cos_a), np.median(ent_a)
        ax_c.axvline(med_c, color=INK_FAINT, linewidth=0.8, linestyle="--")
        ax_c.axhline(med_e, color=INK_FAINT, linewidth=0.8, linestyle="--")
        ax_c.text(med_c + 0.005, ax_c.get_ylim()[1] * 0.97,
                  "focused", fontsize=8, color=INK_MUTED, ha="left", va="top")
        ax_c.text(med_c - 0.005, ax_c.get_ylim()[1] * 0.97,
                  "diffuse", fontsize=8, color=INK_MUTED, ha="right", va="top")
        ax_c.grid(axis="both", which="major")
        ax_c.set_axisbelow(True)
        stem_c = out_dir / "fig12c_attn_stats"
        _log_saved("fig12c", _save(fig_c, stem_c))
        plt.close(fig_c)


# ---
def main() -> None:
    p = argparse.ArgumentParser(
        description="Generate ALL publication figures for PathXDRP.",
    )
    p.add_argument("--figures", nargs="+", type=int,
                   default=list(range(14)),
                   choices=list(range(14)),
                   metavar="{0..13}",
                   help="Which figure IDs to generate (default: all 0–13).")
    p.add_argument("--metric",   default="all",
                   help="Metric for figure 1 (PCC, RMSE, Spearman, R2, ...). "
                        "Use 'all' (default) to generate fig1 for every metric.")
    p.add_argument("--split",    default="all",
                   help="Split for risk-coverage / scatter / calibration. "
                        "Use 'all' (default) to loop all splits.")
    p.add_argument("--seed",     type=int, default=0,
                   help="Seed used for predictions-based figures.")
    p.add_argument("--run_tag",   type=str,  default="")
    p.add_argument("--focal_tag", type=str,  default="")
    p.add_argument("--out_dir",   default=str(FIGURES_DIR))
    p.add_argument("--format", default="both",
                   choices=["png", "pdf", "both"],
                   help="Output format ('both' writes PNG + PDF).")
    p.add_argument("--xai_per_drug_metric", default="all",
                   help="Metric column to plot in fig10. Use 'all' (default) "
                        "to generate for both ig_target_auroc and attn_target_auroc.")
    args = p.parse_args()

    global SAVE_FORMATS
    SAVE_FORMATS = ("png", "pdf") if args.format == "both" else (args.format,)

    if not HAS_MPL:
        print("Install matplotlib: pip install matplotlib")
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing figures to {out_dir}")

    import functools
    if args.focal_tag:
        def _focal_load(model: str, split: str, metric: str = "PCC",
                        run_tag: str = "") -> list[float]:
            vals = load_metric(model, split, metric, run_tag=args.focal_tag)
            if not vals:
                vals = load_metric(model, split, metric, run_tag="")
            return vals
        _load = _focal_load
    else:
        _load = functools.partial(load_metric, run_tag=args.run_tag)

    # ---- Which metrics to iterate for fig1 / alt figs ----
    ALL_METRICS = ["PCC", "RMSE", "Spearman", "R2", "Per-drug PCC", "Per-cell PCC"]
    metrics_to_run = ALL_METRICS if args.metric == "all" else [args.metric]

    # ---- Which splits to iterate for fig2/3/5/8 ----
    all_splits = SPLIT_ORDER
    splits_to_run = all_splits if args.split == "all" else [args.split]

    # ---- Which XAI metrics to iterate for fig10 ----
    XAI_METRICS = ["ig_target_auroc", "attn_target_auroc"]
    xai_metrics_to_run = XAI_METRICS if args.xai_per_drug_metric == "all" else [args.xai_per_drug_metric]

    for fig_id in sorted(set(args.figures)):
        print(f"\n--- Figure {fig_id} ---")

        if fig_id == 0:
            fig_data_smiles(out_dir)
            fig_data_structure(out_dir)
            fig_data_graph(out_dir)

        elif fig_id == 1:
            for metric in metrics_to_run:
                print(f"  fig1 [{metric}] bar chart")
                fig_split_comparison(out_dir, metric, load_metric_fn=_load)
                print(f"  fig1 [{metric}] lollipop")
                fig_split_comparison_dot(out_dir, metric, load_metric_fn=_load)
                print(f"  fig1 [{metric}] radar")
                fig_split_radar(out_dir, metric, load_metric_fn=_load)

        elif fig_id == 2:
            for split in splits_to_run:
                print(f"  fig2 [{split}]")
                fig_risk_coverage(out_dir, split)

        elif fig_id == 3:
            for split in splits_to_run:
                print(f"  fig3 [{split}]")
                fig_uncertainty_scatter(out_dir, split, args.seed)

        elif fig_id == 4:
            print("  fig4 bar chart")
            fig_xai_benchmark(out_dir)
            print("  fig4 horizontal dot chart")
            fig_xai_benchmark_dot(out_dir)

        elif fig_id == 5:
            for split in splits_to_run:
                print(f"  fig5 [{split}]")
                fig_calibration(out_dir, split, args.seed)

        elif fig_id == 6:
            print("  fig6 violin")
            fig_per_drug_pcc(out_dir, load_metric_fn=_load)
            print("  fig6 strip+box")
            fig_per_drug_pcc_strip(out_dir, load_metric_fn=_load)

        elif fig_id == 7:
            fig_headline(out_dir, load_metric_fn=_load)

        elif fig_id == 8:
            for split in splits_to_run:
                print(f"  fig8 [{split}]")
                fig_predicted_vs_actual(out_dir, split, args.seed)

        elif fig_id == 9:
            fig_xai_quadrant(out_dir)

        elif fig_id == 10:
            for xai_met in xai_metrics_to_run:
                print(f"  fig10 [{xai_met}]")
                fig_xai_per_drug_heatmap(out_dir, metric=xai_met)

        elif fig_id == 11:
            print("  fig11 line")
            fig_faithfulness_curve(out_dir)
            print("  fig11 filled-area")
            fig_faithfulness_area(out_dir)

        elif fig_id == 12:
            print("  fig12 attention matrix (12a/12b/12c)")
            fig_attention_matrix(out_dir)


if __name__ == "__main__":
    main()
