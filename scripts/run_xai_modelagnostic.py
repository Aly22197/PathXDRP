"""Cross-model XAI runner for permutation importance + occlusion + faithfulness curves.

Complements ``run_xai_multimodel.py``, which only runs IG (gradient-based,
needs a differentiable forward pass) and attention (architecture-specific).
This script adds two model-agnostic methods that work for ALL four
architectures, including CDRScan, by perturbing the input expression matrix
directly:

  - Permutation importance (gene + pathway granularity)
  - Occlusion (gene + pathway granularity)
  - Faithfulness curve (sweeps top-K removal at K = 5, 10, 20, 30, 50%)

For each (model, drug) pair we score every pathway. Optionally we can score
genes too, but at 19 193 genes per drug this is slow — defaults to off.

Outputs:
  results/xai/xai_modelagnostic_<model>.json
  results/xai/xai_modelagnostic_summary.json    (cross-model aggregate)

Usage:
  python scripts/run_xai_modelagnostic.py --models pathxdrp drpreter graphdrp cdrscan \\
      --n_cells_per_drug 5 --score_genes
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Batch
from tqdm.auto import tqdm

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Reuse the model loaders + helpers from the multimodel script.
from scripts.run_xai_multimodel import (
    DEFAULT_CKPTS, RESULTS_JSONS,
    _load_train_args, load_data, load_model,
    get_prediction, get_morgan_fp,
    select_sensitive_cells, build_drug_input, expr_tensor,
    _resolve_target_genes,
)
from pathxdrp.explain.model_agnostic import (
    permutation_importance_pathways,
    permutation_importance_genes,
    occlusion_pathways,
    occlusion_genes,
    faithfulness_curve,
)


def _make_forward_fn(model, model_name: str, drug_input, drug_kw: str,
                     morgan_fp=None):
    """Build a callable ``forward_fn(expr) -> Tensor[B]`` closing over the
    drug branch. Used by every perturbation method below."""
    extra_kw = {} if morgan_fp is None else {"morgan_fp": morgan_fp}

    def fn(expr_in: torch.Tensor) -> torch.Tensor:
        out = model(**{drug_kw: drug_input}, expr=expr_in, **extra_kw)
        return get_prediction(model_name, out)

    return fn


def _aggregate_gene_to_pathway(gene_scores: np.ndarray,
                               pathway_gene_indices: list[list[int]]) -> np.ndarray:
    """Sum gene-level scores within each pathway."""
    out = np.zeros(len(pathway_gene_indices), dtype=np.float64)
    for i, members in enumerate(pathway_gene_indices):
        if members:
            out[i] = float(np.sum(gene_scores[list(members)]))
    return out


def run_model(model_name: str, args, df, expr_matrix, graph_cache, fp_cache,
              pathway_gene_map, pathway_names, gene_list, moa, device):
    print(f"\n=== {model_name.upper()} ===", flush=True)
    sample_g = next(iter(graph_cache.values()))
    model = load_model(model_name, expr_matrix, pathway_gene_map, sample_g, device)

    pathway_gene_indices = [pathway_gene_map[pn] for pn in pathway_names]
    gene_set = set(gene_list)

    per_drug = []
    pbar = tqdm(moa.items(), desc=f"{model_name} model-agnostic XAI", unit="drug")
    for drug_name, info in pbar:
        drug_id = int(info["drug_id"])
        cosmic_ids = select_sensitive_cells(df, drug_id, k=args.n_cells_per_drug)
        if not cosmic_ids:
            continue
        drug_input, drug_kw = build_drug_input(model_name, drug_id, len(cosmic_ids),
                                               graph_cache, fp_cache, device)
        if drug_input is None:
            continue
        expr = expr_tensor(cosmic_ids, expr_matrix, device)
        if expr is None:
            continue
        # CDRScan / graphs may need separate handling; rebuild correct n_cells
        n_cells = expr.size(0)
        if model_name != "cdrscan" and isinstance(drug_input, Batch):
            # n_cells may differ from cosmic_ids if some had no expression — rebuild.
            graphs = [graph_cache[drug_id] for _ in range(n_cells)]
            drug_input = Batch.from_data_list(graphs).to(device)
        elif model_name == "cdrscan":
            drug_input = drug_input[:n_cells].contiguous()

        morgan_fp = get_morgan_fp(model_name, drug_id, n_cells, fp_cache, device)
        fn = _make_forward_fn(model, model_name, drug_input, drug_kw, morgan_fp)

        rec = {
            "drug":            drug_name,
            "drug_id":         drug_id,
            "n_cells":         n_cells,
            "raw_targets":     info.get("target_genes", []) or [],
            "target_pathway":  info.get("target_pathway", "") or "",
        }
        rec["resolved_targets"] = _resolve_target_genes(rec["raw_targets"], gene_set)

        # ---------- Pathway-level perturbation methods ----------
        try:
            t0 = time.time()
            perm_pw = permutation_importance_pathways(
                fn, expr, pathway_gene_indices,
                n_shuffles=args.n_shuffles,
            )
            rec["perm_pathway_time_s"] = time.time() - t0
            rec["perm_top5_pathways"] = [
                pathway_names[i] for i in np.argsort(perm_pw)[::-1][:5]
            ]
        except Exception as e:
            rec["perm_pathway_error"] = str(e)
            perm_pw = None

        try:
            t0 = time.time()
            occ_pw = occlusion_pathways(fn, expr, pathway_gene_indices,
                                        baseline_value=args.occlusion_baseline)
            rec["occ_pathway_time_s"] = time.time() - t0
            rec["occ_top5_pathways"] = [
                pathway_names[i] for i in np.argsort(occ_pw)[::-1][:5]
            ]
        except Exception as e:
            rec["occ_pathway_error"] = str(e)
            occ_pw = None

        # ---------- Optional gene-level (slow) ----------
        if args.score_genes:
            try:
                # Subset to genes that belong to ANY pathway in our map (8.4k genes,
                # vs the full 19.2k). Reduces time by ~57% with no downstream loss.
                covered = sorted({g for members in pathway_gene_indices for g in members})
                t0 = time.time()
                occ_g = occlusion_genes(fn, expr, gene_indices=covered,
                                        baseline_value=args.occlusion_baseline)
                rec["occ_gene_time_s"] = time.time() - t0
                # Aggregate gene -> pathway as a sanity check vs occ_pw
                occ_g_to_pw = _aggregate_gene_to_pathway(occ_g, pathway_gene_indices)
                rec["occ_gene_top5_pathways"] = [
                    pathway_names[i] for i in np.argsort(occ_g_to_pw)[::-1][:5]
                ]
            except Exception as e:
                rec["occ_gene_error"] = str(e)
                occ_g = None
        else:
            occ_g = None

        # ---------- Faithfulness curves on the OCCLUSION pathway scores ----------
        # We sweep K = {5, 10, 20, 30, 50}% and remove the top-K pathways by
        # occlusion score. A faithful attribution causes a large prediction
        # change at small K. AUC is a single summary number.
        if occ_pw is not None:
            try:
                # Convert pathway scores to gene-level mask: zero all genes in the top-K pathways
                def _pw_to_gene_scores(pw_scores):
                    gscores = np.zeros(expr.shape[1], dtype=np.float64)
                    for p_i, members in enumerate(pathway_gene_indices):
                        for g in members:
                            gscores[g] = max(gscores[g], pw_scores[p_i])
                    return gscores

                gene_scores_for_curve = _pw_to_gene_scores(occ_pw)
                fc = faithfulness_curve(
                    fn, expr, gene_scores_for_curve,
                    fractions=(0.05, 0.10, 0.20, 0.30, 0.50),
                    direction="comp",
                    baseline_value=args.occlusion_baseline,
                )
                rec["faith_curve_comp_auc"] = fc["auc"]
                rec["faith_curve_comp_deltas"] = fc["deltas"]
            except Exception as e:
                rec["faith_curve_error"] = str(e)

        # ---------- Pathway-level target hit (if MoA pathway resolved) ----------
        target_pw = info.get("target_pathway", "") or ""
        from scripts.run_xai_benchmark import _fuzzy_match_pathway
        matched = _fuzzy_match_pathway(target_pw, pathway_names)
        rec["matched_pathway"] = matched
        if matched is not None and matched in pathway_names:
            tgt_idx = pathway_names.index(matched)
            for tag, scores in (("perm_pw", perm_pw), ("occ_pw", occ_pw)):
                if scores is None:
                    continue
                order = np.argsort(scores)[::-1]
                rank = int(np.where(order == tgt_idx)[0][0])
                rec[f"{tag}_target_rank"]    = rank
                rec[f"{tag}_target_in_top5"] = int(rank < 5)
                rec[f"{tag}_target_in_top10"] = int(rank < 10)
                # ROC AUC against a one-hot positive
                from sklearn.metrics import roc_auc_score
                labels = np.zeros(len(pathway_names))
                labels[tgt_idx] = 1
                if labels.sum() > 0 and labels.sum() < len(labels):
                    rec[f"{tag}_target_pathway_auroc"] = float(
                        roc_auc_score(labels, scores)
                    )

        per_drug.append(rec)
        pbar.set_postfix(done=len(per_drug))
    pbar.close()
    return per_drug


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+",
                   default=["pathxdrp", "drpreter", "graphdrp", "cdrscan"],
                   choices=["pathxdrp", "drpreter", "graphdrp", "cdrscan"])
    p.add_argument("--moa_json",
                   default=str(ROOT / "data" / "processed" / "moa_benchmark_all.json"))
    p.add_argument("--out_dir", default=str(ROOT / "results" / "xai"))
    p.add_argument("--n_cells_per_drug", type=int, default=5)
    p.add_argument("--n_shuffles", type=int, default=1,
                   help="Permutation-importance shuffles per pathway (more = less noisy, slower).")
    p.add_argument("--occlusion_baseline", type=float, default=0.0,
                   help="Value substituted into occluded positions. 0.0 corresponds to "
                        "average expression because the input is z-scored per gene.")
    p.add_argument("--score_genes", action="store_true",
                   help="Also run gene-level occlusion (slow: ~8 400 genes per drug).")
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    moa = json.load(open(args.moa_json))
    print(f"Benchmark contains {len(moa)} drugs (from {args.moa_json})", flush=True)

    df, expr_matrix, graph_cache, fp_cache, pathway_gene_map, pathway_names, gene_list = load_data()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries = {}
    for model_name in args.models:
        per_drug = run_model(model_name, args, df, expr_matrix, graph_cache, fp_cache,
                             pathway_gene_map, pathway_names, gene_list, moa, device)
        out_json = out_dir / f"xai_modelagnostic_{model_name}.json"

        # Aggregate: mean across drugs for each scalar metric
        def _safe_mean(key):
            vals = [r.get(key) for r in per_drug
                    if isinstance(r.get(key), (int, float))
                    and not (isinstance(r.get(key), float) and r[key] != r[key])]
            return float(np.mean(vals)) if vals else float("nan")

        summary = {
            "n_drugs_evaluated":            len(per_drug),
            "perm_pw_target_pathway_auroc_mean":
                _safe_mean("perm_pw_target_pathway_auroc"),
            "occ_pw_target_pathway_auroc_mean":
                _safe_mean("occ_pw_target_pathway_auroc"),
            "perm_pw_target_in_top5_rate":
                _safe_mean("perm_pw_target_in_top5"),
            "occ_pw_target_in_top5_rate":
                _safe_mean("occ_pw_target_in_top5"),
            "perm_pw_target_in_top10_rate":
                _safe_mean("perm_pw_target_in_top10"),
            "occ_pw_target_in_top10_rate":
                _safe_mean("occ_pw_target_in_top10"),
            "faith_curve_comp_auc_mean":
                _safe_mean("faith_curve_comp_auc"),
        }
        summaries[model_name] = summary
        with open(out_json, "w") as f:
            json.dump({"summary": summary, "per_drug": per_drug}, f, indent=2,
                      default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x)
        print(f"-> {out_json}", flush=True)
        for k, v in summary.items():
            print(f"     {k:42s} {v}", flush=True)

    cross_summary_path = out_dir / "xai_modelagnostic_summary.json"
    with open(cross_summary_path, "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"\nCross-model summary -> {cross_summary_path}", flush=True)


if __name__ == "__main__":
    main()
