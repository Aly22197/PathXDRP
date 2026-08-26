"""
Unified XAI benchmark runner — runs the same MoA evaluation on any of the four
trained DRP models (PathXDRP, DRPreter, GraphDRP, CDRScan).

For every drug in the MoA benchmark, on its top-K sensitive cell lines:

  - All models: integrated gradients on the expression input → per-gene attribution
                → target-gene AUROC against curated MoA targets.
  - PathXDRP & DRPreter: cross-attention weights → per-pathway attribution + gene
                         projection → target-gene AUROC and pathway-hit@k.
  - Faithfulness (sufficiency / comprehensiveness) on the per-atom max-attention
    score (where applicable).

Output: one JSON per (model, run_tag).

Usage
-----
  # Run all four models on the expanded benchmark:
  python scripts/run_xai_multimodel.py --moa_json data/processed/moa_benchmark_all.json \\
      --models pathxdrp drpreter graphdrp cdrscan
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Batch
from tqdm.auto import tqdm

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---
# Constants / helpers shared across models
# ---

DEFAULT_CKPTS = {
    "pathxdrp": ROOT / "checkpoints" / "pathxdrp" / "random_seed0_fold0.pt",
    "drpreter": ROOT / "checkpoints" / "drpreter" / "random_seed0_fold0.pt",
    "graphdrp": ROOT / "checkpoints" / "graphdrp" / "random_seed0_fold0.pt",
    "cdrscan":  ROOT / "checkpoints" / "cdrscan"  / "random_seed0_fold0.pt",
}

RESULTS_JSONS = {
    "pathxdrp": ROOT / "results" / "pathxdrp" / "random_seed0_fold0.json",
    "drpreter": ROOT / "results" / "drpreter" / "random_seed0_fold0.json",
    "graphdrp": ROOT / "results" / "graphdrp" / "random_seed0_fold0.json",
    "cdrscan":  ROOT / "results" / "cdrscan"  / "random_seed0_fold0.json",
}


# Optional run tag: when set (via --run_tag) the checkpoint and the train-args
# JSON are read from `<split>_seed<S>_fold<F>_<tag>` instead of the default
# random/seed0 run. This is what lets the W6 ablation score each variant's
# faithfulness with the same code path as the headline models.
RUN_TAG: str = ""


def _tagged(path: Path) -> Path:
    if not RUN_TAG:
        return path
    return path.with_name(f"{path.stem}_{RUN_TAG}{path.suffix}")


def _load_train_args(model_name: str) -> dict:
    p = _tagged(RESULTS_JSONS[model_name])
    if p.exists():
        with open(p) as f:
            return json.load(f).get("args", {})
    p = RESULTS_JSONS[model_name]
    if p.exists():
        with open(p) as f:
            return json.load(f).get("args", {})
    return {}


_PATHWAY_STOPWORDS = {"signaling", "pathway", "and", "or", "in", "the", "of", "to"}
# Sentinel labels that mean "no annotation" — must NOT be fuzzy-matched
# (otherwise every drug with target_pathway="Other" gets paired with the
# alphabetically first KEGG pathway that happens to contain "other").
_PATHWAY_SENTINELS = {"", "other", "unclassified", "n/a", "unknown", "none"}


from pathxdrp.explain.target_resolver import resolve_targets as _resolve_target_genes


def _fuzzy_match_pathway(target_pw: str, pathway_names: list[str]) -> Optional[str]:
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


# ---
# Shared data loading
# ---

def load_data():
    from pathxdrp.data.loader import build_master_df
    from pathxdrp.train import build_graph_cache

    print("Loading data", flush=True)
    df, expr_matrix = build_master_df(version="GDSC2", require_smiles=True)
    drugs_df = df[["DRUG_ID", "SMILES"]].drop_duplicates()
    graph_cache, _ = build_graph_cache(drugs_df)

    # Fingerprint cache (only needed for CDRScan, but cheap)
    from pathxdrp.baselines.cdrscan import build_fp_cache
    fp_cache = build_fp_cache(drugs_df)

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
    # IMPORTANT: pathway_names must use insertion order of pathway_gene_map,
    # which matches the attention tensor's column order. Sorting alphabetically
    # breaks the attention-to-pathway mapping.
    pathway_names = list(pathway_gene_map.keys())
    return df, expr_matrix, graph_cache, fp_cache, pathway_gene_map, pathway_names, gene_list


# ---
# Model loading per architecture
# ---

def load_model(model_name: str, expr_matrix, pathway_gene_map, sample_g, device):
    args = _load_train_args(model_name)
    ckpt = _tagged(DEFAULT_CKPTS[model_name])
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    if model_name == "pathxdrp":
        from pathxdrp.models.pathxdrp import PathXDRP
        ckpt_path = _tagged(DEFAULT_CKPTS["pathxdrp"])
        ckpt_raw = torch.load(ckpt_path, map_location=device, weights_only=True)
        _gp_w = ckpt_raw.get("cell_enc.gene_proj.0.weight")
        n_pw_stats = int(_gp_w.shape[1]) if _gp_w is not None else 4
        model = PathXDRP(
            node_in_dim=sample_g.x.size(1),
            edge_in_dim=sample_g.edge_attr.size(1),
            n_genes=expr_matrix.shape[1],
            pathway_gene_map=pathway_gene_map,
            hidden_dim=args.get("hidden_dim", 256),
            n_gat_layers=args.get("n_gat_layers", 4),
            n_attn_heads=args.get("n_attn_heads", 8),
            dropout=args.get("dropout", 0.1),
            mask_type=args.get("mask_type", "none"),
            n_pw_transformer_layers=args.get("n_pw_transformer_layers", 2),
            n_pw_stats=n_pw_stats,
            use_morgan_fp=args.get("use_morgan_fp", False),
            aux_auc_weight=args.get("aux_auc_weight", 0.0),
            cross_attn_residual=args.get("cross_attn_residual", False),
            drop_h_mol=args.get("drop_h_mol", False),
            attn_aux_weight=args.get("attn_aux_weight", 0.0),
            pool_mode=args.get("pool_mode", "auto"),
        )
        model = model.to(device)
        model.load_state_dict(ckpt_raw)
        del ckpt_raw
        model.eval()
        return model
    elif model_name == "drpreter":
        from pathxdrp.baselines.drpreter import DRPreter
        ckpt_raw = torch.load(DEFAULT_CKPTS["drpreter"], map_location=device, weights_only=True)
        _gp_w = ckpt_raw.get("cell_enc.gene_proj.0.weight")
        n_pw_stats = int(_gp_w.shape[1]) if _gp_w is not None else 4
        model = DRPreter(
            node_in_dim=sample_g.x.size(1),
            edge_in_dim=sample_g.edge_attr.size(1),
            n_genes=expr_matrix.shape[1],
            pathway_gene_map=pathway_gene_map,
            hidden_dim=args.get("hidden_dim", 128),
            n_gat_layers=args.get("n_gat_layers", 3),
            n_attn_heads=args.get("n_attn_heads", 8),
            dropout=args.get("dropout", 0.1),
            n_pw_stats=n_pw_stats,
        )
        model = model.to(device)
        model.load_state_dict(ckpt_raw)
        del ckpt_raw
        model.eval()
        return model
    elif model_name == "graphdrp":
        from pathxdrp.baselines.graphdrp import GraphDRP
        model = GraphDRP(
            node_in_dim=sample_g.x.size(1),
            n_genes=expr_matrix.shape[1],
            hidden_dim=args.get("hidden_dim", 128),
            n_gin_layers=args.get("n_gin_layers", 5),
            dropout=args.get("dropout", 0.1),
        )
    elif model_name == "cdrscan":
        from pathxdrp.baselines.cdrscan import CDRScan
        model = CDRScan(
            n_genes=expr_matrix.shape[1],
            hidden_dim=args.get("hidden_dim", 256),
            dropout=args.get("dropout", 0.1),
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model = model.to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()
    return model


def get_prediction(model_name: str, out: dict):
    if model_name == "pathxdrp":
        return out["pred"]["pred"]
    return out["pred"]


def get_morgan_fp(model_name: str, drug_id: int, n_cells: int,
                  fp_cache: dict, device) -> Optional[torch.Tensor]:
    """Return (n_cells, 2048) morgan_fp tensor for PathXDRP when use_morgan_fp=True."""
    if model_name != "pathxdrp" or drug_id not in fp_cache:
        return None
    fp_np = fp_cache[drug_id]
    return torch.tensor(
        np.tile(fp_np, (n_cells, 1)), dtype=torch.float, device=device
    )


def model_supports_attention(model_name: str) -> bool:
    return model_name in {"pathxdrp", "drpreter"}


# ---
# Per-drug helpers
# ---

def select_sensitive_cells(df: pd.DataFrame, drug_id: int, k: int = 5) -> list[int]:
    sub = df[df["DRUG_ID"] == drug_id].nsmallest(k, "LN_IC50")
    return sub["COSMIC_ID"].astype(int).tolist()


def build_drug_input(model_name: str, drug_id: int, n_cells: int,
                     graph_cache: dict, fp_cache: dict, device):
    """Return (drug_input, kwargs_key) appropriate for the model.

    For graph-based models: PyG Batch with n_cells copies of the drug graph.
    For CDRScan: fingerprint tensor with n_cells copies.
    """
    if model_name == "cdrscan":
        if drug_id not in fp_cache:
            return None, None
        fp = torch.tensor(fp_cache[drug_id], dtype=torch.float, device=device)
        fp_batch = fp.unsqueeze(0).expand(n_cells, -1).contiguous()
        return fp_batch, "fp"
    else:
        if drug_id not in graph_cache:
            return None, None
        graphs = [graph_cache[drug_id] for _ in range(n_cells)]
        return Batch.from_data_list(graphs).to(device), "drug_batch"


def expr_tensor(cosmic_ids, expr_matrix, device):
    rows = [expr_matrix.loc[cid].values for cid in cosmic_ids if cid in expr_matrix.index]
    if not rows:
        return None
    return torch.tensor(np.stack(rows), dtype=torch.float, device=device)


# ---
# Integrated gradients on expression input (model-agnostic)
# ---

def integrated_gradients_expr(model, model_name: str, drug_input, drug_kw: str,
                              expr: torch.Tensor, n_steps: int = 20,
                              morgan_fp=None) -> np.ndarray:
    """Per-cell IG attribution over the expression input.

    Returns: (n_genes,) array — mean of absolute attributions across the batch.
    Baseline: zero vector. Method: Riemann sum (cheaper than gausslegendre).
    """
    model.eval()
    baseline = torch.zeros_like(expr)
    # Start alpha strictly > 0 to avoid NaN gradients in PathXDRP/DRPreter, which
    # use sqrt(var).clamp(min=0) in the PathwaySetEncoder — at all-zero input the
    # per-pathway variance is exactly 0, and d/dx sqrt(x) at x=0 is undefined.
    # Midpoint Riemann sum: alpha ∈ {(i+0.5)/n_steps : i=0..n-1}.
    alphas = (torch.arange(n_steps, device=expr.device).float() + 0.5) / n_steps

    total_grads = torch.zeros_like(expr)
    extra_kw = {}
    if morgan_fp is not None:
        extra_kw["morgan_fp"] = morgan_fp
    for a in alphas:
        e = baseline + a * (expr - baseline)
        e = e.detach().requires_grad_(True)
        out = model(**{drug_kw: drug_input}, expr=e, **extra_kw)
        pred = get_prediction(model_name, out).sum()
        grads = torch.autograd.grad(pred, e, retain_graph=False, create_graph=False)[0]
        total_grads = total_grads + grads.detach()

    avg_grads = total_grads / n_steps
    ig = (expr - baseline) * avg_grads          # (B, n_genes)
    return ig.abs().mean(dim=0).cpu().numpy()    # (n_genes,)


# ---
# Attention attribution (PathXDRP / DRPreter only)
# ---

@torch.no_grad()
def attention_attribution(model, drug_input, drug_kw, expr,
                          pathway_gene_map: dict, pathway_names: list[str],
                          n_genes: int, morgan_fp=None):
    kw = {drug_kw: drug_input}
    if morgan_fp is not None:
        kw["morgan_fp"] = morgan_fp
    out = model(**kw, expr=expr)
    attn = out["attn_weights"].detach().cpu().numpy()     # (N_atoms, P)
    # Importance-weighted pathway scores: weight each atom by its max attention
    # to any single pathway (matches the model's own a_weights pooling step).
    atom_importance = attn.max(axis=1)                    # (N_atoms,)
    pw_weighted = (attn * atom_importance[:, None]).sum(axis=0)  # (P,)
    total = pw_weighted.sum()
    pathway_scores = pw_weighted / total if total > 0 else pw_weighted
    gene_scores = np.zeros(n_genes, dtype=np.float64)
    for p_i, pname in enumerate(pathway_names):
        members = pathway_gene_map.get(pname, [])
        if not members:
            continue
        for g_idx in members:
            gene_scores[g_idx] += pathway_scores[p_i]
    return pathway_scores, gene_scores, attn


# ---
# Scoring metrics
# ---

def target_auroc(gene_scores: np.ndarray, target_genes: list[str],
                 gene_list: list[str]) -> float:
    from sklearn.metrics import roc_auc_score
    target_set = set(target_genes)
    labels = np.array([1 if g in target_set else 0 for g in gene_list])
    if labels.sum() == 0 or labels.sum() == len(labels):
        return float("nan")
    return float(roc_auc_score(labels, gene_scores))


def pathway_hitatk(pathway_scores: np.ndarray, known_pathway: str,
                   pathway_names: list[str], k: int = 5) -> float:
    top_k_idx = np.argsort(pathway_scores)[::-1][:k]
    top_k_names = [pathway_names[i] for i in top_k_idx]
    return float(known_pathway in top_k_names)


def faithfulness_pair(model, drug_input, drug_kw, expr, atom_scores, model_name,
                      frac=0.2, morgan_fp=None):
    """Suff: drop low-attention atoms. Comp: drop high-attention atoms."""
    if model_name == "cdrscan":
        return None, None
    base_kw = {drug_kw: drug_input, "expr": expr}
    if morgan_fp is not None:
        base_kw["morgan_fp"] = morgan_fp
    with torch.no_grad():
        base = get_prediction(model_name, model(**base_kw)).mean().item()

    n = atom_scores.shape[0]
    k = max(1, int(n * frac))
    order = np.argsort(atom_scores)

    def _zero(idx):
        b = drug_input.clone()
        b.x = b.x.clone()
        b.x[idx] = 0.0
        kw = {drug_kw: b, "expr": expr}
        if morgan_fp is not None:
            kw["morgan_fp"] = morgan_fp
        with torch.no_grad():
            return get_prediction(model_name, model(**kw)).mean().item()

    return abs(base - _zero(order[:k])), abs(base - _zero(order[-k:]))


# ---
# Per-model run loop
# ---

def run_model(model_name: str, args, df, expr_matrix, graph_cache, fp_cache,
              pathway_gene_map, pathway_names, gene_list, moa: dict, device):
    sample_g = next(iter(graph_cache.values()))
    model = load_model(model_name, expr_matrix, pathway_gene_map, sample_g, device)

    n_genes = len(gene_list)
    gene_set = set(gene_list)
    per_drug = []

    pbar = tqdm(moa.items(), desc=f"{model_name:10s}", unit="drug", dynamic_ncols=True)
    for drug_name, info in pbar:
        drug_id = int(info["drug_id"])
        raw_targets = info.get("target_genes", []) or []
        target_pw   = info.get("target_pathway", "") or ""
        sens = select_sensitive_cells(df, drug_id, k=args.n_cells_per_drug)
        if not sens:
            continue
        drug_input, drug_kw = build_drug_input(model_name, drug_id,
                                               len(sens), graph_cache, fp_cache, device)
        if drug_input is None:
            continue
        expr = expr_tensor(sens, expr_matrix, device)
        if expr is None:
            continue

        resolved = _resolve_target_genes(raw_targets, gene_set)
        best_pw  = _fuzzy_match_pathway(target_pw, pathway_names)

        rec = {
            "drug":             drug_name,
            "n_cells":          len(sens),
            "raw_targets":      raw_targets,
            "resolved_targets": resolved,
            "target_pathway":   target_pw,
            "matched_pathway":  best_pw,
        }

        # --- IG attribution ---
        t0 = time.time()
        morgan_fp_for_ig = get_morgan_fp(model_name, drug_id, len(sens), fp_cache, device)
        try:
            ig_gene_scores = integrated_gradients_expr(
                model, model_name, drug_input, drug_kw, expr, n_steps=args.ig_steps,
                morgan_fp=morgan_fp_for_ig,
            )
            rec["ig_time_s"] = time.time() - t0
            if resolved:
                rec["ig_target_auroc"] = target_auroc(ig_gene_scores, resolved, gene_list)

            # Pathway-level metric: aggregate gene-level IG to pathway sums.
            # The previous one-hot AUROC against the fuzzy-matched target pathway
            # was random for every model because the fuzzy matcher is noisy.
            # We replace it with **gene-set recall@K**: do any of the resolved
            # target genes appear in the gene sets of the top-K attended
            # pathways? This is the biologically meaningful question and
            # bypasses the matcher entirely.
            if resolved:
                ig_pw_scores = np.array([
                    float(np.sum(ig_gene_scores[list(pathway_gene_map[pn])]))
                    if pathway_gene_map.get(pn) else 0.0
                    for pn in pathway_names
                ], dtype=np.float64)
                topk_order = np.argsort(ig_pw_scores)[::-1]
                resolved_set = set(resolved)
                for k in (5, 10, 20):
                    top_pws = [pathway_names[i] for i in topk_order[:k]]
                    union_genes = {g for pn in top_pws for g in
                                   [gene_list[i] for i in pathway_gene_map.get(pn, [])]}
                    rec[f"ig_geneset_recall_at_{k}"] = float(
                        len(union_genes & resolved_set) / max(1, len(resolved_set))
                    )
                    rec[f"ig_geneset_hit_at_{k}"] = int(
                        len(union_genes & resolved_set) > 0
                    )

            # Keep the matched-pathway one-hot AUROC as a legacy diagnostic for
            # transparency, but downstream tables should use the recall@K above.
            if best_pw is not None and best_pw in pathway_names:
                ig_pw_scores = np.array([
                    float(np.sum(ig_gene_scores[list(pathway_gene_map[pn])]))
                    if pathway_gene_map.get(pn) else 0.0
                    for pn in pathway_names
                ], dtype=np.float64)
                tgt_idx = pathway_names.index(best_pw)
                from sklearn.metrics import roc_auc_score
                labels = np.zeros(len(pathway_names))
                labels[tgt_idx] = 1
                rec["ig_target_pathway_auroc"] = float(roc_auc_score(labels, ig_pw_scores))
                order = np.argsort(ig_pw_scores)[::-1]
                rec["ig_target_pathway_rank"] = int(np.where(order == tgt_idx)[0][0])
                rec["ig_target_pathway_in_top5"] = int(rec["ig_target_pathway_rank"] < 5)
        except Exception as e:
            rec["ig_error"] = str(e)

        # --- Attention attribution (PathXDRP, DRPreter only) ---
        if model_supports_attention(model_name):
            try:
                morgan_fp = get_morgan_fp(model_name, drug_id, len(sens), fp_cache, device)
                pw_scores, gene_scores, attn = attention_attribution(
                    model, drug_input, drug_kw, expr,
                    pathway_gene_map, pathway_names, n_genes,
                    morgan_fp=morgan_fp,
                )
                if resolved:
                    rec["attn_target_auroc"] = target_auroc(gene_scores, resolved, gene_list)

                # Pathway-level metric for attention: gene-set recall@K
                # (see IG branch above for why we replaced the AUROC).
                if resolved:
                    topk_order = np.argsort(pw_scores)[::-1]
                    resolved_set = set(resolved)
                    for k in (5, 10, 20):
                        top_pws = [pathway_names[i] for i in topk_order[:k]]
                        union_genes = {g for pn in top_pws for g in
                                       [gene_list[i] for i in pathway_gene_map.get(pn, [])]}
                        rec[f"attn_geneset_recall_at_{k}"] = float(
                            len(union_genes & resolved_set) / max(1, len(resolved_set))
                        )
                        rec[f"attn_geneset_hit_at_{k}"] = int(
                            len(union_genes & resolved_set) > 0
                        )

                # Legacy one-hot AUROC (kept for transparency)
                if best_pw is not None and best_pw in pathway_names:
                    tgt_idx = pathway_names.index(best_pw)
                    from sklearn.metrics import roc_auc_score
                    labels = np.zeros(len(pathway_names))
                    labels[tgt_idx] = 1
                    rec["attn_target_pathway_auroc"] = float(
                        roc_auc_score(labels, pw_scores)
                    )
                    order = np.argsort(pw_scores)[::-1]
                    rec["attn_target_pathway_rank"] = int(np.where(order == tgt_idx)[0][0])
                    rec["attn_target_pathway_in_top5"] = int(rec["attn_target_pathway_rank"] < 5)

                # Sensitivity alignment: do attended pathways discriminate
                # sensitive vs resistant cells? Report max(AUROC, 1-AUROC) so
                # the metric is directionally neutral (model may learn resistance
                # markers rather than sensitivity markers — both are correct).
                drug_rows = df[df["DRUG_ID"] == drug_id]
                if len(drug_rows) >= 10:
                    q25 = drug_rows["LN_IC50"].quantile(0.25)
                    q75 = drug_rows["LN_IC50"].quantile(0.75)
                    sens_ids = drug_rows[drug_rows["LN_IC50"] <= q25]["COSMIC_ID"].tolist()
                    res_ids  = drug_rows[drug_rows["LN_IC50"] >= q75]["COSMIC_ID"].tolist()
                    def _pw_cell_score(cids):
                        scores = []
                        for cid in cids:
                            if cid in expr_matrix.index:
                                ev = expr_matrix.loc[cid].values
                                pw_expr = np.array([
                                    ev[pathway_gene_map[pn]].mean() if pathway_gene_map.get(pn) else 0.0
                                    for pn in pathway_names
                                ], dtype=np.float32)
                                scores.append(float(pw_expr @ pw_scores))
                        return scores
                    s_sc = _pw_cell_score(sens_ids)
                    r_sc = _pw_cell_score(res_ids)
                    if s_sc and r_sc:
                        from sklearn.metrics import roc_auc_score
                        labels = [1]*len(s_sc) + [0]*len(r_sc)
                        preds  = s_sc + r_sc
                        raw_sa = roc_auc_score(labels, preds)
                        rec["attn_sensitivity_alignment"] = max(raw_sa, 1.0 - raw_sa)

                # Faithfulness on per-atom max-attention
                atom_scores = attn.max(axis=1)
                suff, comp = faithfulness_pair(model, drug_input, drug_kw, expr,
                                               atom_scores, model_name,
                                               morgan_fp=morgan_fp)
                rec["attn_faithfulness_suff"] = suff
                rec["attn_faithfulness_comp"] = comp
                rec["attn_top5_pathways"] = [
                    pathway_names[i] for i in np.argsort(pw_scores)[::-1][:5]
                ]
            except Exception as e:
                rec["attn_error"] = str(e)

        per_drug.append(rec)

    pbar.close()
    return per_drug


# ---
# CLI
# ---

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+",
                   default=["pathxdrp", "drpreter", "graphdrp", "cdrscan"],
                   choices=["pathxdrp", "drpreter", "graphdrp", "cdrscan"])
    p.add_argument("--moa_json", default=str(ROOT / "data" / "processed" / "moa_benchmark_all.json"))
    p.add_argument("--out_dir",  default=str(ROOT / "results" / "xai"))
    p.add_argument("--n_cells_per_drug", type=int, default=5)
    p.add_argument("--ig_steps", type=int, default=20)
    p.add_argument("--device", default=None)
    p.add_argument("--run_tag", default="",
                   help="Score the checkpoint/results tagged with this "
                        "suffix (e.g. abA) instead of the default run.")
    args = p.parse_args()

    global RUN_TAG
    RUN_TAG = args.run_tag

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    moa = json.load(open(args.moa_json))
    print(f"Benchmark contains {len(moa)} drugs (from {args.moa_json})", flush=True)

    df, expr_matrix, graph_cache, fp_cache, pathway_gene_map, pathway_names, gene_list = load_data()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries = {}
    for model_name in args.models:
        print(f"\n=== {model_name.upper()} ===", flush=True)
        per_drug = run_model(model_name, args, df, expr_matrix,
                             graph_cache, fp_cache,
                             pathway_gene_map, pathway_names, gene_list,
                             moa, device)
        _sfx = f"_{RUN_TAG}" if RUN_TAG else ""
        out_json = out_dir / f"xai_multimodel_{model_name}{_sfx}.json"

        # Aggregate
        def _mean(key):
            vals = [r.get(key) for r in per_drug if r.get(key) is not None and not (isinstance(r.get(key), float) and np.isnan(r.get(key)))]
            return float(np.mean(vals)) if vals else float("nan")

        summary = {
            "n_drugs_evaluated":               len(per_drug),
            # Gene-level
            "ig_target_auroc_mean":            _mean("ig_target_auroc"),
            "ig_target_auroc_median":          float(np.nanmedian([
                r.get("ig_target_auroc", np.nan) for r in per_drug
            ])),
            "attn_target_auroc_mean":          _mean("attn_target_auroc"),
            "attn_target_auroc_median":        float(np.nanmedian([
                r.get("attn_target_auroc", np.nan) for r in per_drug
            ])),
            # Pathway-level: gene-set recall@K (replaces old fuzzy-matched AUROC).
            # "Of the drug's known target genes, what fraction sit in the union
            # of gene sets of the top-K attended pathways?"
            "ig_geneset_recall_at_5_mean":    _mean("ig_geneset_recall_at_5"),
            "ig_geneset_recall_at_10_mean":   _mean("ig_geneset_recall_at_10"),
            "ig_geneset_recall_at_20_mean":   _mean("ig_geneset_recall_at_20"),
            "ig_geneset_hit_at_5_rate":       _mean("ig_geneset_hit_at_5"),
            "ig_geneset_hit_at_10_rate":      _mean("ig_geneset_hit_at_10"),
            "attn_geneset_recall_at_5_mean":  _mean("attn_geneset_recall_at_5"),
            "attn_geneset_recall_at_10_mean": _mean("attn_geneset_recall_at_10"),
            "attn_geneset_recall_at_20_mean": _mean("attn_geneset_recall_at_20"),
            "attn_geneset_hit_at_5_rate":     _mean("attn_geneset_hit_at_5"),
            "attn_geneset_hit_at_10_rate":    _mean("attn_geneset_hit_at_10"),
            # Legacy diagnostic (matcher-based; kept for transparency, do not headline)
            "ig_target_pathway_auroc_mean":    _mean("ig_target_pathway_auroc"),
            "attn_target_pathway_auroc_mean":  _mean("attn_target_pathway_auroc"),
            # Sensitivity + faithfulness
            "attn_sensitivity_alignment_mean": _mean("attn_sensitivity_alignment"),
            "attn_faithfulness_suff_mean":     _mean("attn_faithfulness_suff"),
            "attn_faithfulness_comp_mean":     _mean("attn_faithfulness_comp"),
        }
        summaries[model_name] = summary
        with open(out_json, "w") as f:
            json.dump({"summary": summary, "per_drug": per_drug}, f, indent=2,
                      default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x)
        print(f"-> {out_json}", flush=True)
        print(f"   {model_name} summary:", flush=True)
        for k, v in summary.items():
            print(f"     {k:32s} {v}", flush=True)

    # Cross-model summary
    _sfx = f"_{RUN_TAG}" if RUN_TAG else ""
    cross_summary_path = out_dir / f"xai_multimodel_summary{_sfx}.json"
    with open(cross_summary_path, "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"\nCross-model summary -> {cross_summary_path}", flush=True)


if __name__ == "__main__":
    main()
