"""Attention-collapse diagnostic for PathXDRP.

The XAI benchmark shows PathXDRP's attention has near-random target AUROC
(~0.49) and almost zero faithfulness (0.02–0.03), with sparsity 0.9999.
Three explanations are plausible:

  1. **Saturation collapse** — softmax always concentrates on the same one or
     two pathways regardless of the drug. Diagnose by computing the
     drug-to-drug variance of each pathway's mean attention. If variance is
     near zero, the encoder ignores the drug.
  2. **Per-atom uniformity** — within a single drug, all atoms attend to the
     same pathway distribution. Diagnose by computing the mean pairwise
     cosine similarity of atom-level attention vectors per drug.
  3. **Drug-independent collapse** — attention is non-zero but drug-invariant
     (a constant prior). Diagnose by computing the cosine similarity of
     drug-level attention vectors across pairs of drugs.

This script runs N drugs through a trained PathXDRP checkpoint, dumps:

  results/xai/attention_diagnostic.json      — numerical summaries
  figures/diag_attention_perdrug.png          — top-attended pathways per drug
  figures/diag_attention_pathway_variance.png — pathway variance histogram
  figures/diag_attention_drug_similarity.png  — drug-pair similarity histogram

Run after the per-atom IG benchmark has produced clean numbers.

Usage:
  python scripts/diagnose_attention.py \\
      --ckpt checkpoints/pathxdrp/random_seed0_fold0.pt \\
      --n_drugs 30
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Batch

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_pathxdrp(ckpt_path: Path, device: torch.device):
    """Reuse the loader pattern from run_xai_benchmark.py for argument
    discovery, then build the model with the right kwargs."""
    from pathxdrp.data.loader import build_master_df
    from pathxdrp.train import build_graph_cache
    from pathxdrp.models.pathxdrp import PathXDRP

    print("Loading data and model", flush=True)
    df, expr_matrix = build_master_df(version="GDSC2", require_smiles=True)
    drugs_df = df[["DRUG_ID", "DRUG_NAME", "SMILES"]].drop_duplicates(subset=["DRUG_ID"])
    graph_cache, _ = build_graph_cache(drugs_df)

    pgm_path = ROOT / "data" / "processed" / "pathway_gene_map.json"
    with open(pgm_path) as f:
        pathway_gene_symbols = json.load(f)
    gene_list = list(expr_matrix.columns)
    gene_to_idx = {g: i for i, g in enumerate(gene_list)}
    pathway_gene_map = {
        pw: [gene_to_idx[g] for g in genes if g in gene_to_idx]
        for pw, genes in pathway_gene_symbols.items()
        if any(g in gene_to_idx for g in genes)
    }
    pathway_names = list(pathway_gene_map.keys())

    # Discover training args from the matching results JSON.
    stem = ckpt_path.stem
    results_json = ROOT / "results" / "pathxdrp" / f"{stem}.json"
    train_args = {}
    if results_json.exists():
        with open(results_json) as f:
            train_args = json.load(f).get("args", {})

    fp_cache = {}
    if train_args.get("use_morgan_fp"):
        from pathxdrp.baselines.cdrscan import build_fp_cache
        fp_cache = build_fp_cache(drugs_df)

    sample_g = next(iter(graph_cache.values()))
    ckpt_raw = torch.load(ckpt_path, map_location=device, weights_only=True)
    n_pw_stats = int(ckpt_raw["cell_enc.gene_proj.0.weight"].shape[1])

    model = PathXDRP(
        node_in_dim=sample_g.x.size(1),
        edge_in_dim=sample_g.edge_attr.size(1),
        n_genes=expr_matrix.shape[1],
        pathway_gene_map=pathway_gene_map,
        hidden_dim=train_args.get("hidden_dim", 256),
        n_gat_layers=train_args.get("n_gat_layers", 4),
        n_attn_heads=train_args.get("n_attn_heads", 8),
        dropout=train_args.get("dropout", 0.1),
        mask_type=train_args.get("mask_type", "none"),
        n_pw_transformer_layers=train_args.get("n_pw_transformer_layers", 2),
        n_pw_stats=n_pw_stats,
        use_morgan_fp=train_args.get("use_morgan_fp", False),
        aux_auc_weight=train_args.get("aux_auc_weight", 0.0),
        cross_attn_residual=train_args.get("cross_attn_residual", False),
        drop_h_mol=train_args.get("drop_h_mol", False),
        attn_aux_weight=train_args.get("attn_aux_weight", 0.0),
    ).to(device)
    model.load_state_dict(ckpt_raw)
    model.eval()
    return (model, df, expr_matrix, graph_cache, fp_cache,
            pathway_gene_map, pathway_names, gene_list, drugs_df)


def _drug_id_to_name(drugs_df: pd.DataFrame) -> dict[int, str]:
    return dict(zip(drugs_df["DRUG_ID"].astype(int), drugs_df["DRUG_NAME"].astype(str)))


def _build_batch(drug_id, cosmic_ids, graph_cache, expr_matrix, fp_cache, device):
    g = graph_cache[drug_id]
    graphs = [g for _ in cosmic_ids]
    drug_batch = Batch.from_data_list(graphs).to(device)
    expr_rows = []
    for cid in cosmic_ids:
        if cid in expr_matrix.index:
            expr_rows.append(expr_matrix.loc[cid].values)
    if not expr_rows:
        return None
    expr = torch.tensor(np.stack(expr_rows), dtype=torch.float, device=device)
    morgan_fp = None
    if fp_cache and drug_id in fp_cache:
        fp = fp_cache[drug_id]
        morgan_fp = torch.tensor(np.tile(fp, (len(expr_rows), 1)),
                                 dtype=torch.float, device=device)
    return drug_batch, expr, morgan_fp


def _select_drugs(df, drugs_df, n_drugs, n_cells):
    """Pick the top-N drugs by IC50 sample count, then their N most-sensitive cells."""
    counts = df.groupby("DRUG_ID").size().sort_values(ascending=False)
    drug_ids = []
    for did in counts.index:
        if did in set(drugs_df["DRUG_ID"]):
            drug_ids.append(int(did))
        if len(drug_ids) >= n_drugs:
            break
    out = []
    for did in drug_ids:
        sub = df[df["DRUG_ID"] == did].nsmallest(n_cells, "LN_IC50")
        out.append((did, sub["COSMIC_ID"].astype(int).tolist()))
    return out


@torch.no_grad()
def collect_attention(model, drug_batch, expr, morgan_fp):
    """Forward once. Returns (atom-level attn (N_atoms, P), drug-level mean (P,))."""
    out = model(drug_batch=drug_batch, expr=expr, morgan_fp=morgan_fp)
    atom_attn = out["attn_weights"].detach().cpu().numpy()      # (N_atoms, P)
    drug_attn = atom_attn.mean(axis=0)                          # (P,) — pool across atoms
    return atom_attn, drug_attn


def _entropy(p, eps=1e-12):
    p = np.asarray(p, dtype=np.float64)
    p = p / (p.sum() + eps)
    return float(-(p * np.log(p + eps)).sum())


def _cosine(a, b, eps=1e-12):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + eps))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True,
                   help="PathXDRP checkpoint (e.g. checkpoints/pathxdrp/random_seed0_fold0.pt)")
    p.add_argument("--n_drugs", type=int, default=30,
                   help="How many drugs to sample.")
    p.add_argument("--n_cells", type=int, default=5,
                   help="How many sensitive cells per drug.")
    p.add_argument("--out_json",
                   default=str(ROOT / "results" / "xai" / "attention_diagnostic.json"))
    p.add_argument("--fig_dir", default=str(ROOT / "figures"))
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    (model, df, expr_matrix, graph_cache, fp_cache,
     pathway_gene_map, pathway_names, _gene_list, drugs_df) = _load_pathxdrp(
        Path(args.ckpt), device)
    drug_name_map = _drug_id_to_name(drugs_df)

    # ------------------------------------------------------------------
    # Collect per-drug attention
    # ------------------------------------------------------------------
    selection = _select_drugs(df, drugs_df, args.n_drugs, args.n_cells)
    print(f"Selected {len(selection)} drugs.", flush=True)

    drug_records = []      # one entry per drug
    drug_attn_matrix = []  # (n_drugs, P) for cross-drug comparisons
    drug_labels      = []
    for drug_id, cosmic_ids in selection:
        if drug_id not in graph_cache or not cosmic_ids:
            continue
        built = _build_batch(drug_id, cosmic_ids, graph_cache,
                             expr_matrix, fp_cache, device)
        if built is None:
            continue
        drug_batch, expr, morgan_fp = built
        atom_attn, drug_attn = collect_attention(model, drug_batch, expr, morgan_fp)

        # Per-atom uniformity (within-drug spread of attention vectors)
        # cosine similarity matrix over atoms; we report mean pairwise sim.
        if atom_attn.shape[0] >= 2:
            norms = np.linalg.norm(atom_attn, axis=1, keepdims=True) + 1e-12
            an = atom_attn / norms
            sim_mat = an @ an.T
            iu = np.triu_indices(sim_mat.shape[0], k=1)
            within_atom_sim = float(sim_mat[iu].mean())
        else:
            within_atom_sim = float("nan")

        # Per-atom entropy over pathways (averaged across atoms)
        atom_ent = float(np.mean([_entropy(row) for row in atom_attn]))
        drug_ent = _entropy(drug_attn)
        top1 = int(drug_attn.argmax())
        top5 = drug_attn.argsort()[::-1][:5].tolist()

        drug_records.append({
            "drug_id":              int(drug_id),
            "drug":                 drug_name_map.get(int(drug_id), f"drug_{drug_id}"),
            "n_atoms":              int(atom_attn.shape[0]),
            "drug_attn_entropy":    drug_ent,
            "max_entropy":          float(np.log(atom_attn.shape[1])),
            "within_atom_cosine":   within_atom_sim,
            "mean_per_atom_entropy": atom_ent,
            "top1_pathway":         pathway_names[top1],
            "top5_pathways":        [pathway_names[i] for i in top5],
        })
        drug_attn_matrix.append(drug_attn)
        drug_labels.append(drug_name_map.get(int(drug_id), f"drug_{drug_id}"))

    drug_attn_matrix = np.stack(drug_attn_matrix, axis=0)  # (n_drugs, P)

    # ------------------------------------------------------------------
    # Cross-drug variance per pathway: how much does each pathway's attention
    # change with the drug? If variance is near zero everywhere, attention is
    # drug-invariant.
    # ------------------------------------------------------------------
    pw_mean = drug_attn_matrix.mean(axis=0)
    pw_var  = drug_attn_matrix.var(axis=0)
    pw_var_normalised = pw_var / (pw_mean + 1e-12)

    # Cross-drug similarity (drug-to-drug)
    norms = np.linalg.norm(drug_attn_matrix, axis=1, keepdims=True) + 1e-12
    dn = drug_attn_matrix / norms
    drug_sim = dn @ dn.T
    iu = np.triu_indices(drug_sim.shape[0], k=1)
    cross_drug_pair_sim = drug_sim[iu]

    summary = {
        "n_drugs":                       len(drug_records),
        "n_pathways":                    int(drug_attn_matrix.shape[1]),
        "mean_drug_attn_entropy":        float(np.mean([r["drug_attn_entropy"] for r in drug_records])),
        "max_possible_entropy":          float(np.log(drug_attn_matrix.shape[1])),
        "mean_per_atom_entropy":         float(np.mean([r["mean_per_atom_entropy"] for r in drug_records])),
        "mean_within_atom_cosine_sim":   float(np.nanmean([r["within_atom_cosine"] for r in drug_records])),
        "mean_cross_drug_cosine_sim":    float(cross_drug_pair_sim.mean()),
        "median_cross_drug_cosine_sim":  float(np.median(cross_drug_pair_sim)),
        "max_pathway_variance":          float(pw_var.max()),
        "frac_pathways_with_zero_var":   float((pw_var < 1e-10).mean()),
        # Verdict thresholds (heuristics for the 3 collapse modes):
        "verdict_drug_invariant_collapse":
            float(cross_drug_pair_sim.mean() > 0.95),
        "verdict_within_atom_uniformity":
            float(np.nanmean([r["within_atom_cosine"] for r in drug_records]) > 0.95),
        "verdict_softmax_saturation":
            float(np.mean([r["drug_attn_entropy"] for r in drug_records])
                  < 0.1 * float(np.log(drug_attn_matrix.shape[1]))),
    }
    out = {"summary": summary, "per_drug": drug_records}
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Diagnostic JSON -> {out_path}", flush=True)

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping figures.")
        return

    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Figure A: cross-drug pathway variance distribution
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.hist(pw_var, bins=40, color="#0072B2", alpha=0.85)
    ax.set_xlabel("Per-pathway variance of mean attention across drugs")
    ax.set_ylabel("Number of pathways")
    ax.set_title("If most pathways have ~0 variance, attention is drug-invariant")
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(fig_dir / "diag_attention_pathway_variance.png", dpi=300)
    plt.close(fig)

    # Figure B: drug-pair attention similarity distribution
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.hist(cross_drug_pair_sim, bins=40, color="#E69F00", alpha=0.85)
    ax.axvline(0.95, color="#c00", linestyle="--", linewidth=1, label="0.95 collapse threshold")
    ax.set_xlabel("Cosine similarity between drug-level attention vectors")
    ax.set_ylabel("Number of drug pairs")
    ax.set_title(
        f"Mean drug-pair similarity = {cross_drug_pair_sim.mean():.3f}  "
        f"(closer to 1 = more drug-invariant)"
    )
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(fig_dir / "diag_attention_drug_similarity.png", dpi=300)
    plt.close(fig)

    # Figure C: top-1 attended pathway per drug (categorical bar)
    top1s = [r["top1_pathway"] for r in drug_records]
    counts = pd.Series(top1s).value_counts()
    fig, ax = plt.subplots(figsize=(8, max(3.0, 0.32 * len(counts) + 1.0)))
    ax.barh(range(len(counts)), counts.values, color="#009E73", alpha=0.9)
    ax.set_yticks(range(len(counts)))
    ax.set_yticklabels(counts.index, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Number of drugs with this as top-1 attended pathway")
    ax.set_title("If one or two pathways dominate, attention is collapsed")
    fig.tight_layout()
    fig.savefig(fig_dir / "diag_attention_perdrug.png", dpi=300)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Console verdict
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("ATTENTION-COLLAPSE DIAGNOSTIC")
    print("=" * 70)
    print(f"  Drugs analysed:              {summary['n_drugs']}")
    print(f"  Pathways:                    {summary['n_pathways']}")
    print(f"  Max possible entropy:        {summary['max_possible_entropy']:.3f}")
    print(f"  Mean drug-level entropy:     {summary['mean_drug_attn_entropy']:.3f}  "
          f"({100 * summary['mean_drug_attn_entropy'] / summary['max_possible_entropy']:.0f}% of max)")
    print(f"  Mean per-atom entropy:       {summary['mean_per_atom_entropy']:.3f}")
    print(f"  Within-atom cosine sim:      {summary['mean_within_atom_cosine_sim']:.3f}  "
          "(>0.95 = atoms attend identically)")
    print(f"  Cross-drug cosine sim:       {summary['mean_cross_drug_cosine_sim']:.3f}  "
          "(>0.95 = drug-invariant)")
    print(f"  Frac pathways zero-var:      {summary['frac_pathways_with_zero_var']:.3f}")
    print()
    print("  Verdict flags (1.0 = symptom present):")
    print(f"    drug_invariant_collapse:   {summary['verdict_drug_invariant_collapse']}")
    print(f"    within_atom_uniformity:    {summary['verdict_within_atom_uniformity']}")
    print(f"    softmax_saturation:        {summary['verdict_softmax_saturation']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
