"""
End-to-end XAI benchmark runner — Phase 6.

For each curated MoA drug:
  1. Find IC50-sensitive cell lines (bottom 25% of LN_IC50 for that drug).
  2. Build a representative drug+cell batch.
  3. Run all available attribution methods (attention, IG, GNNExplainer).
  4. Score each method against ground-truth targets (target_auroc, hit@5,
     faithfulness suff/comp, sparsity).
  5. Aggregate to results/xai/xai_benchmark_results.json
     (the format expected by eval/plot_results.py fig_xai_benchmark).

Usage:
  python scripts/run_xai_benchmark.py \\
      --ckpt checkpoints/random_seed0_fold0.pt \\
      --split random --seed 0 --fold 0
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Batch
from tqdm.auto import tqdm

ROOT = Path(__file__).parent.parent
# Ensure the project root is on sys.path so `pathxdrp` resolves when this script
# is invoked directly (matches train_baseline.py's behaviour).
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_args_for_ckpt(ckpt_path: Path) -> dict:
    """Find the results JSON next to this checkpoint and return its training args.

    Checkpoint paths look like  checkpoints/<split>_seed<s>_fold<f>.pt
    Results paths look like     results/pathxdrp/<split>_seed<s>_fold<f>.json
    """
    stem = ckpt_path.stem  # e.g. random_seed0_fold0
    candidates = [
        ROOT / "results" / "pathxdrp" / f"{stem}.json",
        ROOT / "results" / "archive" / "pathxdrp_v2" / f"{stem}.json",
    ]
    for c in candidates:
        if c.exists():
            with open(c) as f:
                data = json.load(f)
            args = data.get("args", {})
            if args:
                print(f"Loaded training args from {c}", flush=True)
                return args
    print(f"WARNING: no results JSON found for {stem}; "
          f"falling back to defaults. This will likely fail with size-mismatch.", flush=True)
    return {}


def _load_pathxdrp_for_inference(ckpt_path: Path, device: torch.device):
    """Reuse the data-loading + model-construction logic from train.py."""
    from pathxdrp.data.loader import build_master_df
    from pathxdrp.train import build_graph_cache
    from pathxdrp.models.pathxdrp import PathXDRP

    print("Loading data and model", flush=True)
    df, expr_matrix = build_master_df(version="GDSC2", require_smiles=True)
    drugs_df = df[["DRUG_ID", "SMILES"]].drop_duplicates()
    graph_cache, _ = build_graph_cache(drugs_df)

    pgm_path = ROOT / "data" / "processed" / "pathway_gene_map.json"
    with open(pgm_path) as f:
        pathway_gene_symbols = json.load(f)
    gene_list   = list(expr_matrix.columns)
    gene_to_idx = {g: i for i, g in enumerate(gene_list)}
    pathway_gene_map = {
        pw: [gene_to_idx[g] for g in genes if g in gene_to_idx]
        for pw, genes in pathway_gene_symbols.items()
        if any(g in gene_to_idx for g in genes)
    }

    # Look up training args from the matching results JSON so the model is
    # reconstructed with the same hidden_dim / n_gat_layers / etc as the checkpoint.
    train_args = _load_args_for_ckpt(ckpt_path)

    # Build Morgan FP cache if the checkpoint used --use_morgan_fp
    fp_cache: dict = {}
    if train_args.get("use_morgan_fp", False):
        from pathxdrp.baselines.cdrscan import build_fp_cache
        drugs_df = df[["DRUG_ID", "SMILES"]].drop_duplicates()
        fp_cache = build_fp_cache(drugs_df)
        print(f"Morgan FP cache: {len(fp_cache)} drugs", flush=True)

    sample_g = next(iter(graph_cache.values()))

    # Determine n_pw_stats by peeking at the checkpoint weight shape.
    # v1 checkpoints have gene_proj.0.weight shape (hidden, 3);
    # v2+ checkpoints have shape (hidden, 4).
    ckpt_raw = torch.load(ckpt_path, map_location=device, weights_only=True)
    _gp_w = ckpt_raw.get("cell_enc.gene_proj.0.weight")
    n_pw_stats = int(_gp_w.shape[1]) if _gp_w is not None else 4
    print(f"Checkpoint n_pw_stats: {n_pw_stats}", flush=True)

    model = PathXDRP(
        node_in_dim=sample_g.x.size(1),
        edge_in_dim=sample_g.edge_attr.size(1),
        n_genes=expr_matrix.shape[1],
        pathway_gene_map=pathway_gene_map,
        hidden_dim=train_args.get("hidden_dim", 256),
        n_gat_layers=train_args.get("n_gat_layers", 4),
        n_attn_heads=train_args.get("n_attn_heads", 8),
        dropout=train_args.get("dropout", 0.1),
        mask_type=train_args.get("mask_type", "soft"),
        n_pw_transformer_layers=train_args.get("n_pw_transformer_layers", 1),
        n_pw_stats=n_pw_stats,
        use_morgan_fp=train_args.get("use_morgan_fp", False),
        aux_auc_weight=train_args.get("aux_auc_weight", 0.0),
        cross_attn_residual=train_args.get("cross_attn_residual", False),
        drop_h_mol=train_args.get("drop_h_mol", False),
        attn_aux_weight=train_args.get("attn_aux_weight", 0.0),
    ).to(device)
    model.load_state_dict(ckpt_raw)
    del ckpt_raw
    model.eval()

    # CRITICAL: pathway_names must use the model's internal pathway order
    # (insertion order of pathway_gene_map, which matches the attention tensor's
    # column order). Sorting alphabetically here breaks the mapping between
    # attention weights and pathway labels.
    pathway_names = list(pathway_gene_map.keys())
    return model, df, expr_matrix, graph_cache, pathway_gene_map, pathway_names, gene_list, fp_cache


def _select_sensitive_cells(df: pd.DataFrame, drug_id: int, k: int = 5) -> list[int]:
    """Return the k most-sensitive (lowest LN_IC50) cell COSMIC_IDs for this drug."""
    sub = df[df["DRUG_ID"] == drug_id].nsmallest(k, "LN_IC50")
    return sub["COSMIC_ID"].astype(int).tolist()


def _build_inference_batch(
    drug_id: int,
    cosmic_ids: list[int],
    graph_cache: dict,
    expr_matrix,
    device: torch.device,
    fp_cache: dict | None = None,
):
    if drug_id not in graph_cache:
        return None
    graphs = [graph_cache[drug_id] for _ in cosmic_ids]
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
        fp_arr = fp_cache[drug_id]                                  # (2048,) float32
        morgan_fp = torch.tensor(
            np.tile(fp_arr, (len(expr_rows), 1)), dtype=torch.float, device=device
        )  # (B, 2048)
    return drug_batch, expr, morgan_fp


@torch.no_grad()
def attention_attribution(
    model, drug_batch, expr, pathway_gene_map, pathway_names, gene_list,
    morgan_fp=None,
):
    """
    Forward once, read attn_weights from PathwayMaskedCrossAttention,
    project to (1) per-pathway scores and (2) per-gene scores.
    """
    out = model(drug_batch=drug_batch, expr=expr, morgan_fp=morgan_fp)
    attn = out["attn_weights"].detach().cpu().numpy()           # (N_atoms, P)

    # Weight each atom's attention distribution by that atom's importance.
    # Importance = max attention to any single pathway (matches the model's
    # own pooling step: a_weights = attn_weights.max(dim=-1)).
    # A uniform mean gives equal weight to structural atoms (which have ~1/P
    # attention everywhere) and pharmacophoric atoms, washing out the signal.
    atom_importance = attn.max(axis=1)                          # (N_atoms,)
    pw_weighted = (attn * atom_importance[:, None]).sum(axis=0) # (P,)
    total = pw_weighted.sum()
    pathway_scores = pw_weighted / total if total > 0 else pw_weighted  # (P,) normalised

    # Project pathway scores back to genes via membership
    n_genes = len(gene_list)
    gene_scores = np.zeros(n_genes, dtype=np.float64)
    for p_i, pname in enumerate(pathway_names):
        members = pathway_gene_map.get(pname, [])
        if not members:
            continue
        for g_idx in members:
            gene_scores[g_idx] += pathway_scores[p_i]
    return pathway_scores, gene_scores, attn


def integrated_gradients_expr(
    model, drug_batch, expr: torch.Tensor,
    n_steps: int = 20,
    morgan_fp=None,
) -> np.ndarray:
    """Per-gene IG attribution over the expression input.

    Baseline: zero expression vector.  Method: midpoint Riemann sum.
    Avoids NaN at x=0 by using (i+0.5)/n_steps alphas (never exactly zero).

    Returns: (n_genes,) float64 array of mean absolute attributions.
    """
    model.eval()
    baseline = torch.zeros_like(expr)
    alphas = (torch.arange(n_steps, device=expr.device).float() + 0.5) / n_steps
    total_grads = torch.zeros_like(expr)
    extra_kw = {} if morgan_fp is None else {"morgan_fp": morgan_fp}
    for a in alphas:
        e = baseline + a * (expr - baseline)
        e = e.detach().requires_grad_(True)
        out = model(drug_batch=drug_batch, expr=e, **extra_kw)
        pred = out["pred"]["pred"].sum()
        grads = torch.autograd.grad(pred, e, retain_graph=False, create_graph=False)[0]
        total_grads = total_grads + grads.detach()
    avg_grads = total_grads / n_steps
    ig = (expr - baseline) * avg_grads           # (B, n_genes)
    return ig.abs().mean(dim=0).cpu().numpy()    # (n_genes,)


def faithfulness_pair(model, drug_batch, expr, atom_scores, frac=0.2, morgan_fp=None):
    """Returns (suff_delta, comp_delta) — small suff + large comp = faithful.

    Single-K version kept for backwards compat. Use ``faithfulness_curve_atoms``
    below for the multi-K AUC summary that's harder to game.
    """
    model.eval()
    with torch.no_grad():
        base = model(drug_batch=drug_batch, expr=expr, morgan_fp=morgan_fp)["pred"]["pred"].mean().item()

    n = atom_scores.shape[0]
    k = max(1, int(n * frac))
    order = np.argsort(atom_scores)

    def _zero(idx):
        b = drug_batch.clone()
        b.x = b.x.clone()
        b.x[idx] = 0.0
        with torch.no_grad():
            return model(drug_batch=b, expr=expr, morgan_fp=morgan_fp)["pred"]["pred"].mean().item()

    low_idx  = order[:k]
    high_idx = order[-k:]
    suff = abs(base - _zero(low_idx))
    comp = abs(base - _zero(high_idx))
    return suff, comp


def faithfulness_curve_atoms(model, drug_batch, expr, atom_scores,
                             fractions=(0.05, 0.10, 0.20, 0.30, 0.50),
                             morgan_fp=None):
    """Multi-K faithfulness curve over drug atoms.

    Returns dict with per-fraction comp deltas (zeroing the most-attributed
    atoms) and a scalar AUC summary. Reports the area under the |Δ pred|
    curve as a function of K — a faithful attribution rises sharply at small
    K, an unfaithful one stays near zero. AUC is harder to game than single-K
    because gaming one K slice doesn't fix the whole curve.
    """
    import numpy as np
    model.eval()
    with torch.no_grad():
        base = model(drug_batch=drug_batch, expr=expr, morgan_fp=morgan_fp)["pred"]["pred"].mean().item()

    order = np.argsort(atom_scores)[::-1]   # most-attributed first
    n = atom_scores.shape[0]

    deltas = []
    fracs_used = []
    for f in fractions:
        k = max(1, int(round(n * float(f))))
        idx = order[:k]
        b = drug_batch.clone()
        b.x = b.x.clone()
        b.x[idx] = 0.0
        with torch.no_grad():
            y = model(drug_batch=b, expr=expr, morgan_fp=morgan_fp)["pred"]["pred"].mean().item()
        deltas.append(abs(base - y))
        fracs_used.append(float(f))

    auc = float(np.trapz(deltas, fracs_used)) if len(deltas) >= 2 else float("nan")
    return {"fractions": fracs_used, "deltas": deltas, "auc": auc}


from pathxdrp.explain.target_resolver import resolve_targets as _resolve_target_genes


_PATHWAY_STOPWORDS = {"signaling", "pathway", "and", "or", "in", "the", "of", "to"}
# GDSC labels that mean "no meaningful pathway annotation" — must NOT be matched.
# Returning a fuzzy match for these produces nonsense pairings like
# "Other" -> "Ubiquinone and other terpenoid-quinone biosynthesis".
_PATHWAY_SENTINELS = {"", "other", "unclassified", "n/a", "unknown", "none"}


def _fuzzy_match_pathway(target_pw: str, pathway_names: list[str]) -> Optional[str]:
    """Rank KEGG pathway names by content-token overlap with ``target_pw``.

    Drops generic tokens like 'signaling' and 'pathway' that dominate KEGG names.
    Sentinel labels ('', 'Other', 'Unclassified', ...) short-circuit to None so
    we don't pair drugs that have no annotated pathway with arbitrary KEGG names
    (the old matcher returned "Ubiquinone biosynthesis" for everything).
    """
    import re
    if not target_pw or target_pw.strip().lower() in _PATHWAY_SENTINELS:
        return None
    tp_tokens = {
        t.lower() for t in re.split(r"[\s/,\-]+", target_pw)
        if t.lower() not in _PATHWAY_STOPWORDS and len(t) > 1
    }
    if not tp_tokens:
        return None
    best_match, best_score = None, 0
    for pn in pathway_names:
        pn_tokens = {
            t.lower() for t in re.split(r"[\s/,\-]+", pn)
            if t.lower() not in _PATHWAY_STOPWORDS and len(t) > 1
        }
        score = len(tp_tokens & pn_tokens)
        if score > best_score:
            best_score = score
            best_match = pn
    return best_match if best_score >= 1 else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="Path to a trained PathXDRP .pt")
    p.add_argument("--moa_json", default=str(ROOT / "data" / "processed" / "moa_benchmark_all.json"))
    p.add_argument("--out_json",  default=str(ROOT / "results" / "xai" / "xai_benchmark_results.json"))
    p.add_argument("--n_cells_per_drug", type=int, default=5)
    p.add_argument("--device", default=None)
    p.add_argument("--debug", action="store_true",
                   help="Print intermediate values for the first drug; do not swallow exceptions")
    args = p.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    moa_path = Path(args.moa_json)
    if not moa_path.exists():
        raise FileNotFoundError(f"{moa_path} not found. Run scripts/build_moa_benchmark.py first.")
    with open(moa_path) as f:
        moa = json.load(f)

    (model, df, expr_matrix, graph_cache, pathway_gene_map,
     pathway_names, gene_list, fp_cache) = _load_pathxdrp_for_inference(Path(args.ckpt), device)

    from pathxdrp.explain.benchmark import ExplainBenchmark
    bench = ExplainBenchmark(model, moa_json=moa_path, device=str(device))

    aggregate: dict = {
        "attention":           {"target_auroc": [], "sensitivity_alignment": [],
                                "faithfulness_suff": [], "faithfulness_comp": [],
                                "sparsity": []},
        "integrated_gradients": {"target_auroc": [], "faithfulness_suff": [],
                                 "faithfulness_comp": [], "sparsity": []},
    }
    per_drug: list[dict] = []

    # Print pathway-name diagnostics once so we can verify ordering
    print(f"\nFirst 5 pathway_names (model order): {pathway_names[:5]}", flush=True)
    print(f"Last 5 pathway_names (model order):  {pathway_names[-5:]}", flush=True)
    print(f"Total pathways: {len(pathway_names)}", flush=True)

    debug_first = args.debug
    pbar = tqdm(moa.items(), desc="XAI benchmark", unit="drug")
    for drug_name, info in pbar:
        drug_id  = int(info["drug_id"])
        raw_target_genes = info.get("target_genes", []) or []
        target_pw        = info.get("target_pathway", "") or ""
        if drug_id not in graph_cache:
            continue
        sens = _select_sensitive_cells(df, drug_id, k=args.n_cells_per_drug)
        if not sens:
            continue
        built = _build_inference_batch(drug_id, sens, graph_cache, expr_matrix, device, fp_cache)
        if built is None:
            continue
        drug_batch, expr, morgan_fp = built

        # ---- Attention attribution ----
        t0 = time.time()
        pw_scores, gene_scores, attn = attention_attribution(
            model, drug_batch, expr, pathway_gene_map, pathway_names, gene_list,
            morgan_fp=morgan_fp,
        )
        attn_time = time.time() - t0

        # Resolve target genes through alias-aware mapping
        target_genes = _resolve_target_genes(raw_target_genes, gene_list)

        # Match GDSC target_pathway against KEGG pathway names by token overlap
        best_match = _fuzzy_match_pathway(target_pw, pathway_names)

        if debug_first:
            print(f"\n--- DEBUG: first drug {drug_name} ---", flush=True)
            print(f"  raw targets:  {raw_target_genes}", flush=True)
            print(f"  resolved:     {target_genes}", flush=True)
            print(f"  target_pw:    {target_pw!r}", flush=True)
            print(f"  best KEGG:    {best_match!r}", flush=True)
            print(f"  attn shape:   {attn.shape}", flush=True)
            print(f"  pw_scores [first 5]: {pw_scores[:5]}", flush=True)
            top5_idx = np.argsort(pw_scores)[::-1][:5].tolist()
            print(f"  pw_scores [argtop5]: {top5_idx}", flush=True)
            print(f"  top5 pathways: {[pathway_names[i] for i in top5_idx]}", flush=True)
            print(f"  matched_pathway index: {pathway_names.index(best_match) if best_match in pathway_names else 'NOT FOUND'}", flush=True)
            print(f"  matched_pathway rank:  {sorted(np.argsort(pw_scores)[::-1].tolist()).index(pathway_names.index(best_match)) if best_match in pathway_names else 'N/A'}", flush=True)
            print(f"  gene_scores stats: min={gene_scores.min():.6f}, max={gene_scores.max():.6f}, "
                  f"mean={gene_scores.mean():.6f}, nan={np.isnan(gene_scores).sum()}", flush=True)
            debug_first = False

        rec = {
            "drug":         drug_name,
            "n_cells":      len(sens),
            "attn_time_s":  attn_time,
            "raw_targets":  raw_target_genes,
            "resolved_targets": target_genes,
            "target_pathway":    target_pw,
            "matched_pathway":   best_match,
        }

        # ---- Target-gene AUROC ----
        if target_genes:
            try:
                auroc = bench.target_auroc(gene_scores, target_genes, gene_list)
                rec["attention_target_auroc"] = auroc
                if not np.isnan(auroc):
                    aggregate["attention"]["target_auroc"].append(auroc)
            except Exception as e:
                rec["target_auroc_error"] = str(e)
        else:
            rec["target_auroc_skip"] = "no resolved target genes"

        # ---- Sensitivity-alignment AUROC ----
        # Tests whether the top-attended pathways are more active in sensitive cells.
        # Split all cells with IC50 data for this drug into sensitive (bottom 25%)
        # vs resistant (top 25%) and score each cell by the sum of its expression
        # weighted by pw_scores.  AUROC > 0.5 means attended pathways discriminate
        # sensitivity — which is what the model was trained to do.
        try:
            drug_rows = df[df["DRUG_ID"] == drug_id]
            if len(drug_rows) >= 10:
                q25 = drug_rows["LN_IC50"].quantile(0.25)
                q75 = drug_rows["LN_IC50"].quantile(0.75)
                sens_ids  = drug_rows[drug_rows["LN_IC50"] <= q25]["COSMIC_ID"].tolist()
                res_ids   = drug_rows[drug_rows["LN_IC50"] >= q75]["COSMIC_ID"].tolist()
                def _cell_pw_score(cosmic_ids_group):
                    scores = []
                    for cid in cosmic_ids_group:
                        if cid in expr_matrix.index:
                            expr_vec = expr_matrix.loc[cid].values  # (n_genes,)
                            # Map gene expression to pathway-level mean, then dot with pw_scores
                            pw_expr = np.zeros(len(pathway_names), dtype=np.float32)
                            for p_i, pname in enumerate(pathway_names):
                                g_idxs = pathway_gene_map.get(pname, [])
                                if g_idxs:
                                    pw_expr[p_i] = expr_vec[g_idxs].mean()
                            scores.append(float(pw_expr @ pw_scores))
                    return scores
                sens_scores = _cell_pw_score(sens_ids)
                res_scores  = _cell_pw_score(res_ids)
                if sens_scores and res_scores:
                    labels = [1] * len(sens_scores) + [0] * len(res_scores)
                    preds  = sens_scores + res_scores
                    from sklearn.metrics import roc_auc_score
                    raw_sa = roc_auc_score(labels, preds)
                    sa = max(raw_sa, 1.0 - raw_sa)
                    rec["attention_sensitivity_alignment"] = sa
                    aggregate["attention"]["sensitivity_alignment"].append(sa)
                else:
                    rec["sensitivity_alignment_skip"] = "no overlapping cells in expr_matrix"
            else:
                rec["sensitivity_alignment_skip"] = f"too few drug rows ({len(drug_rows)})"
        except Exception as e:
            rec["sensitivity_alignment_error"] = str(e)

        # ---- Faithfulness on atom max-attention (NOT mean — mean is uniform by softmax identity) ----
        atom_scores = attn.max(axis=1)  # (N_atoms,) most-concentrated attention per atom
        try:
            suff, comp = faithfulness_pair(model, drug_batch, expr, atom_scores, morgan_fp=morgan_fp)
            rec["attention_faithfulness_suff"] = suff
            rec["attention_faithfulness_comp"] = comp
            aggregate["attention"]["faithfulness_suff"].append(suff)
            aggregate["attention"]["faithfulness_comp"].append(comp)
        except Exception as e:
            rec["faithfulness_error"] = str(e)

        # ---- Multi-K faithfulness curve (ROAR-light: no retraining) ----
        try:
            fc = faithfulness_curve_atoms(model, drug_batch, expr, atom_scores,
                                          morgan_fp=morgan_fp)
            rec["attention_faith_curve_auc"] = fc["auc"]
            rec["attention_faith_curve_deltas"] = fc["deltas"]
            aggregate["attention"].setdefault("faith_curve_auc", []).append(fc["auc"])
        except Exception as e:
            rec["attention_faith_curve_error"] = str(e)

        rec["attention_sparsity"] = bench.sparsity(atom_scores)
        aggregate["attention"]["sparsity"].append(rec["attention_sparsity"])

        # ---- Expression IG: gene-level attributions → ig_target_auroc ----
        try:
            ig_gene_scores = integrated_gradients_expr(
                model, drug_batch, expr, n_steps=20, morgan_fp=morgan_fp
            )
            if target_genes:
                ig_auroc = bench.target_auroc(ig_gene_scores, target_genes, gene_list)
                rec["ig_target_auroc"] = ig_auroc
                if not np.isnan(ig_auroc):
                    aggregate["integrated_gradients"]["target_auroc"].append(ig_auroc)
        except Exception as e:
            rec["ig_expr_error"] = str(e)

        # ---- Atom-level IG: faithfulness + sparsity ----
        try:
            ig_attr = bench.integrated_gradients(
                drug_batch, n_steps=20, expr=expr, morgan_fp=morgan_fp
            )
            ig_atom_scores = ig_attr.abs().mean(dim=-1).numpy()
            rec["ig_sparsity"] = bench.sparsity(ig_atom_scores)
            aggregate["integrated_gradients"]["sparsity"].append(rec["ig_sparsity"])
            try:
                suff, comp = faithfulness_pair(
                    model, drug_batch, expr, ig_atom_scores, morgan_fp=morgan_fp
                )
                rec["ig_faithfulness_suff"] = suff
                rec["ig_faithfulness_comp"] = comp
                aggregate["integrated_gradients"]["faithfulness_suff"].append(suff)
                aggregate["integrated_gradients"]["faithfulness_comp"].append(comp)
                # Multi-K faithfulness curve on per-atom IG scores
                fc_ig = faithfulness_curve_atoms(model, drug_batch, expr, ig_atom_scores,
                                                  morgan_fp=morgan_fp)
                rec["ig_faith_curve_auc"]    = fc_ig["auc"]
                rec["ig_faith_curve_deltas"] = fc_ig["deltas"]
                aggregate["integrated_gradients"].setdefault("faith_curve_auc", []).append(fc_ig["auc"])
            except Exception as e_inner:
                rec["ig_faithfulness_error"] = str(e_inner)
        except Exception as e:
            rec["ig_atom_error"] = str(e)

        per_drug.append(rec)
        pbar.set_postfix(drug=drug_name[:14], done=len(per_drug))
    pbar.close()

    # Aggregate means
    summary = {}
    for method, metrics in aggregate.items():
        summary[method] = {
            m: float(np.mean(vs)) if vs else float("nan")
            for m, vs in metrics.items()
        }

    out = {"summary": summary, "per_drug": per_drug}
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\nXAI benchmark done. {len(per_drug)} drugs scored. -> {args.out_json}", flush=True)
    print("Method summary:", flush=True)
    for method, scores in summary.items():
        print(f"  {method}: " +
              ", ".join(f"{k}={v:.3f}" for k, v in scores.items()), flush=True)


if __name__ == "__main__":
    main()
