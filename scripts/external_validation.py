"""
External validation on CCLE — Phase 5.

Loads a trained PathXDRP checkpoint and evaluates it on a held-out CCLE
expression file with overlapping drugs (CTRPv2 / PRISM Repurposing).

Expected files (paths overridable via CLI):
  data/external/CCLE_expression.csv          # (cells × genes), DepMap format
  data/external/CTRPv2_response.csv          # columns: CCLE_NAME, COMPOUND, AUC or LN_IC50

If those files are not yet provided, this script writes a stub manifest
explaining what to download — so it does not silently no-op.

Outputs:
  results/external/ccle_<ckpt_name>.json      — full metric report
  results/external/ccle_<ckpt_name>_preds.csv — raw per-pair predictions

Usage:
  python scripts/external_validation.py --ckpt checkpoints/random_seed0_fold0.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Batch
from tqdm.auto import tqdm

ROOT = Path(__file__).parent.parent
EXT_DIR = ROOT / "data" / "external"
RESULTS_DIR = ROOT / "results" / "external"

DOWNLOAD_NOTES = """\
External validation requires two files in data/external/:

  1. CCLE_expression.csv
     - Source: DepMap CCLE RNA-seq, e.g.
       https://depmap.org/portal/download (latest "CCLE_expression.csv")
     - Format: rows = ModelID (ACH-XXXXXX), columns = "GENE_SYMBOL (entrez_id)"
     - The same format as our DepMap 24Q4 file — different cells.

  2. CTRPv2_response.csv  (or PRISM_response.csv / GDSC_separate.csv)
     - Source: NCI CTD2 / DepMap PRISM Repurposing
     - Required columns: ModelID, DRUG_NAME (or CID/SMILES), and a
       continuous response value: LN_IC50 or AUC
     - This script auto-detects the response column.

Once both files are in place, re-run this script. We auto-match drugs
to our master_df by DRUG_NAME (case-insensitive) and SMILES if available.
"""


# ---
# Loaders
# ---

def load_external_expression(path: Path) -> pd.DataFrame:
    expr = pd.read_csv(path, index_col=0)
    # Strip "(entrez_id)" suffix to match our gene-symbol convention
    expr.columns = pd.Index([c.split(" (")[0].strip() for c in expr.columns])
    return expr


def align_genes_to_train(expr_ext: pd.DataFrame, train_genes: list[str]) -> pd.DataFrame:
    """Reindex external expression to the training gene order; missing -> 0."""
    return expr_ext.reindex(columns=train_genes, fill_value=0.0).astype("float32")


def load_external_response(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Detect response column
    for col in ("LN_IC50", "AUC", "ic50_ln", "ic50"):
        if col in df.columns:
            df = df.rename(columns={col: "y_true"})
            break
    if "y_true" not in df.columns:
        raise RuntimeError(f"No response column found in {path} (looked for LN_IC50, AUC, ic50_ln, ic50).")
    if "DRUG_NAME" not in df.columns:
        for alt in ("compound_name", "compound", "drug", "Drug", "COMPOUND"):
            if alt in df.columns:
                df = df.rename(columns={alt: "DRUG_NAME"})
                break
    if "DRUG_NAME" not in df.columns:
        raise RuntimeError(f"No drug-name column in {path}.")
    return df


# ---
# Main
# ---

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",     required=True)
    p.add_argument("--expr",     default=str(EXT_DIR / "CCLE_expression.csv"))
    p.add_argument("--response", default=str(EXT_DIR / "CTRPv2_response.csv"))
    p.add_argument("--model", default="pathxdrp",
                   choices=["pathxdrp", "graphdrp", "drpreter", "cdrscan",
                            "deepcdr"],
                   help="architecture of the checkpoint being evaluated")
    p.add_argument("--source_name", default="ccle_ctrpv2",
                   help="Used in output filenames")
    p.add_argument("--exclude-train-cells", dest="exclude_train_cells",
                   action="store_true", default=True,
                   help="drop external cell lines that are in the GDSC2 "
                        "training cohort (default: on)")
    p.add_argument("--keep-train-cells", dest="exclude_train_cells",
                   action="store_false",
                   help="keep them, for a diagnostic run")
    p.add_argument("--clip-sigma", dest="clip_sigma", type=float, default=10.0,
                   help="clip z-scored external expression to +/- this many "
                        "sigma; 0 to disable (default: 10)")
    p.add_argument("--iqr-fence", dest="iqr_fence", type=float, default=3.0,
                   help="drop responses outside Q1/Q3 -/+ fence*IQR; 0 to "
                        "disable (default: 3)")
    args = p.parse_args()

    expr_path = Path(args.expr)
    resp_path = Path(args.response)
    if not expr_path.exists() or not resp_path.exists():
        print("External-validation files not found.\n", flush=True)
        print(DOWNLOAD_NOTES, flush=True)
        EXT_DIR.mkdir(parents=True, exist_ok=True)
        manifest = EXT_DIR / "DOWNLOAD_INSTRUCTIONS.txt"
        manifest.write_text(DOWNLOAD_NOTES)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Reuse training pipeline for model + drug graphs ---
    from pathxdrp.data.loader import build_master_df
    from pathxdrp.train import build_graph_cache
    from pathxdrp.models.pathxdrp import PathXDRP
    from pathxdrp.eval.metrics import regression_report

    print("Loading training data and model", flush=True)
    df_train, expr_train = build_master_df(version="GDSC2", require_smiles=True)
    drugs_df = df_train[["DRUG_ID", "DRUG_NAME", "SMILES"]].drop_duplicates()
    graph_cache, _ = build_graph_cache(drugs_df)

    pgm_path = ROOT / "data" / "processed" / "pathway_gene_map.json"
    with open(pgm_path) as f:
        pathway_gene_symbols = json.load(f)
    train_genes  = list(expr_train.columns)
    gene_to_idx  = {g: i for i, g in enumerate(train_genes)}
    pathway_gene_map = {
        pw: [gene_to_idx[g] for g in genes if g in gene_to_idx]
        for pw, genes in pathway_gene_symbols.items()
        if any(g in gene_to_idx for g in genes)
    }
    # Build the model from the configuration the checkpoint was trained with,
    # not from constructor defaults. The published run uses hidden_dim 256 and
    # four GAT layers, so defaults give a shape mismatch on load_state_dict.
    # Every run records its arguments beside its results, so read them there.
    ckpt_path = Path(args.ckpt)
    cfg = {}
    cfg_src = "constructor defaults"
    res_json = (ROOT / "results" / ckpt_path.parent.name /
                f"{ckpt_path.stem}.json")
    if res_json.exists():
        rec = json.loads(res_json.read_text(encoding="utf-8"))
        rec_args = rec.get("args", {})
        keep = ("hidden_dim", "n_gat_layers", "n_attn_heads", "dropout",
                "mask_type", "evidential_lam", "drug_encoder_type",
                "cell_encoder_type", "n_pw_transformer_layers",
                "frac_active_sharpness", "use_morgan_fp", "aux_auc_weight",
                "cross_attn_residual", "drop_h_mol", "attn_aux_weight")
        cfg = {k: rec_args[k] for k in keep if k in rec_args}
        cfg_src = res_json.name
    print(f"  model configuration from: {cfg_src}", flush=True)
    if cfg:
        print("   ", {k: cfg[k] for k in sorted(cfg)}, flush=True)

    # The trained PathXDRP concatenates a global Morgan fingerprint into h_mol.
    # With drop_h_mol the head never reads h_mol, so the fingerprint cannot
    # change a prediction, but the encoder still requires the tensor. CDRScan
    # is fingerprint-based throughout and needs it as its drug input.
    fp_cache = None
    if cfg.get("use_morgan_fp") or args.model == "cdrscan":
        from pathxdrp.baselines.cdrscan import build_fp_cache
        print("  building Morgan fingerprints for the drug set", flush=True)
        fp_cache = build_fp_cache(drugs_df)

    sample_g = next(iter(graph_cache.values()))
    node_dim = sample_g.x.size(1)
    edge_dim = (sample_g.edge_attr.size(1)
                if sample_g.edge_attr is not None
                and sample_g.edge_attr.numel() > 0 else 9)
    n_genes = expr_train.shape[1]

    def _pick(*names, **defaults):
        """Constructor kwargs recorded for this run, with fallbacks."""
        out = dict(defaults)
        for n in names:
            if n in cfg:
                out[n] = cfg[n]
        return out

    if args.model == "pathxdrp":
        model = PathXDRP(node_in_dim=node_dim, edge_in_dim=edge_dim,
                         n_genes=n_genes, pathway_gene_map=pathway_gene_map,
                         **cfg).to(device)
    elif args.model == "graphdrp":
        from pathxdrp.baselines.graphdrp import GraphDRP
        model = GraphDRP(node_in_dim=node_dim, n_genes=n_genes,
                         **_pick("hidden_dim", "n_gin_layers", "dropout")
                         ).to(device)
    elif args.model == "deepcdr":
        from pathxdrp.baselines.deepcdr import DeepCDR
        model = DeepCDR(node_in_dim=node_dim, n_genes=n_genes,
                        **_pick("hidden_dim", "n_gcn_layers", "dropout")
                        ).to(device)
    elif args.model == "drpreter":
        from pathxdrp.baselines.drpreter import DRPreter
        kw = _pick("hidden_dim", "n_gat_layers", "n_attn_heads", "dropout")
        # n_pw_stats is not recorded in the run arguments and its default has
        # changed since these checkpoints were trained. The first gene
        # projection has one input per statistic, so the checkpoint states its
        # own value; read it rather than assume the current default.
        _sd = torch.load(args.ckpt, map_location="cpu")
        _w = _sd.get("cell_enc.gene_proj.0.weight")
        if _w is not None:
            kw["n_pw_stats"] = int(_w.shape[1])
            print(f"  n_pw_stats from checkpoint: {kw['n_pw_stats']}",
                  flush=True)
        model = DRPreter(node_in_dim=node_dim, edge_in_dim=edge_dim,
                         n_genes=n_genes, pathway_gene_map=pathway_gene_map,
                         **kw).to(device)
    elif args.model == "cdrscan":
        from pathxdrp.baselines.cdrscan import CDRScan
        model = CDRScan(n_genes=n_genes,
                        **_pick("hidden_dim", "dropout")).to(device)
    else:
        raise ValueError(f"unknown model: {args.model}")

    # Some checkpoints predate buffers that are now derived at construction
    # (an index table, for instance). Those are recomputed correctly from the
    # pathway map, so a missing buffer is harmless, but anything else is not:
    # report exactly what did not match instead of silently tolerating it.
    _state = torch.load(args.ckpt, map_location=device)
    missing, unexpected = model.load_state_dict(_state, strict=False)
    if missing or unexpected:
        print(f"  state_dict: {len(missing)} missing, "
              f"{len(unexpected)} unexpected", flush=True)
        for k in list(missing)[:6]:
            print(f"    missing   : {k}", flush=True)
        for k in list(unexpected)[:6]:
            print(f"    unexpected: {k}", flush=True)
        weighty = [k for k in list(missing) + list(unexpected)
                   if k.endswith((".weight", ".bias"))]
        if weighty:
            raise RuntimeError(
                "checkpoint does not match the model on learned parameters: "
                + ", ".join(weighty[:5]))
    model.eval()

    # --- Load + align external data ---
    print("Loading external CCLE expression", flush=True)
    expr_ext = load_external_expression(expr_path)
    expr_ext = align_genes_to_train(expr_ext, train_genes)

    # Apply our training-time mean/std (z-score using OUR statistics, not theirs)
    train_mean = expr_train.mean(axis=0)
    train_std  = expr_train.std(axis=0).replace(0, 1.0)
    expr_ext = (expr_ext - train_mean) / train_std

    # A small number of externally-normalised cell lines carry extreme values
    # after this transform, which are artefacts of the source normalisation
    # rather than biology. Clip them so a handful of genes cannot dominate.
    if args.clip_sigma and args.clip_sigma > 0:
        n_clipped = int((expr_ext.abs() > args.clip_sigma).sum().sum())
        expr_ext = expr_ext.clip(-args.clip_sigma, args.clip_sigma)
        print(f"  clipped {n_clipped:,} expression values to "
              f"+/-{args.clip_sigma:g} sigma", flush=True)

    print("Loading external response file", flush=True)
    resp = load_external_response(resp_path)

    # --- Match drug names ---
    name_to_id = {n.lower(): int(d) for n, d in zip(drugs_df["DRUG_NAME"], drugs_df["DRUG_ID"])}
    resp["matched_drug_id"] = resp["DRUG_NAME"].astype(str).str.lower().map(name_to_id)
    resp = resp.dropna(subset=["matched_drug_id"]).copy()
    resp["matched_drug_id"] = resp["matched_drug_id"].astype(int)
    # Cell lines that appear in the GDSC2 training cohort are not a held-out
    # test: the model has already learned representations for them. Drop them
    # so the transfer estimate is genuinely out-of-cohort.
    if args.exclude_train_cells:
        map_path = ROOT / "data" / "processed" / "cosmic_to_depmap.csv"
        if map_path.exists() and "ModelID" in resp.columns:
            cmap = pd.read_csv(map_path)
            train_cosmic = set(int(c) for c in expr_train.index
                               if pd.notna(c))
            train_models = set(
                cmap.loc[cmap["COSMICID"].isin(train_cosmic), "ModelID"]
                    .dropna().astype(str))
            before_cells = resp["ModelID"].nunique()
            before_rows = len(resp)
            resp = resp[~resp["ModelID"].astype(str).isin(train_models)].copy()
            print(f"  Excluded training cell lines: "
                  f"{before_cells - resp['ModelID'].nunique()} of "
                  f"{before_cells} cell lines "
                  f"({before_rows - len(resp):,} rows)", flush=True)

    # Dose-response fits far outside the bulk are extrapolations beyond the
    # tested concentration range rather than measurements.
    if args.iqr_fence and args.iqr_fence > 0 and len(resp):
        q1, q3 = resp["y_true"].quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - args.iqr_fence * iqr, q3 + args.iqr_fence * iqr
        n_before = len(resp)
        resp = resp[(resp["y_true"] >= lo) & (resp["y_true"] <= hi)].copy()
        print(f"  IQR x{args.iqr_fence:g} fence [{lo:.2f}, {hi:.2f}] "
              f"removed {n_before - len(resp):,} implausible fits", flush=True)

    print(f"  Matched {len(resp):,} response rows to {resp['matched_drug_id'].nunique()} train-time drugs",
          flush=True)

    # --- Predict ---
    y_true_all, y_pred_all, drug_ids_all, cell_ids_all = [], [], [], []
    epi_all, ale_all = [], []
    pbar = tqdm(resp.itertuples(index=False), total=len(resp),
                desc="External eval", unit="row")
    with torch.no_grad():
        for r in pbar:
            drug_id = int(r.matched_drug_id)
            if drug_id not in graph_cache:
                continue
            cell = getattr(r, "ModelID", None) or getattr(r, "CCLE_NAME", None)
            if cell is None or cell not in expr_ext.index:
                continue
            expr_row = torch.tensor(expr_ext.loc[cell].values, dtype=torch.float, device=device).unsqueeze(0)
            fp = None
            if fp_cache is not None:
                arr = fp_cache.get(drug_id)
                if arr is None:
                    continue
                fp = torch.tensor(arr, dtype=torch.float, device=device).unsqueeze(0)

            if args.model == "cdrscan":
                # fingerprint model: no molecular graph in the forward pass
                out = model(fp=fp, expr=expr_row)
            elif args.model == "pathxdrp":
                batch = Batch.from_data_list([graph_cache[drug_id]]).to(device)
                out = model(drug_batch=batch, expr=expr_row, morgan_fp=fp)
            else:
                batch = Batch.from_data_list([graph_cache[drug_id]]).to(device)
                out = model(drug_batch=batch, expr=expr_row)

            # PathXDRP returns the evidential parameters under "pred"; the
            # baselines return a plain tensor there.
            _p = out["pred"]
            if isinstance(_p, dict):
                y_pred_all.append(float(_p["pred"].item()))
                epi_all.append(float(_p["epistemic"].item())
                               if "epistemic" in _p else float("nan"))
                ale_all.append(float(_p["aleatoric"].item())
                               if "aleatoric" in _p else float("nan"))
            else:
                y_pred_all.append(float(_p.item()))
                epi_all.append(float("nan"))
                ale_all.append(float("nan"))
            y_true_all.append(float(r.y_true))
            drug_ids_all.append(drug_id)
            cell_ids_all.append(cell)
    pbar.close()

    if not y_pred_all:
        print("No matched (drug, cell) pairs found — nothing to evaluate.", flush=True)
        return

    y_true = np.array(y_true_all)
    y_pred = np.array(y_pred_all)
    rep = regression_report(y_true, y_pred, drug_ids=np.array(drug_ids_all))
    rep["n_pairs"] = int(len(y_true))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_name = Path(args.ckpt).stem
    # Checkpoint stems are identical across architectures, so the model name
    # has to be part of the filename or each run overwrites the previous one.
    tag = f"{args.source_name}_{args.model}_{ckpt_name}"
    out_path  = RESULTS_DIR / f"{tag}.json"
    preds_path = RESULTS_DIR / f"{tag}_preds.csv"

    with open(out_path, "w") as f:
        json.dump({"args": vars(args), "test": rep}, f, indent=2)
    pd.DataFrame({
        "drug_id": drug_ids_all,
        "cell_id": cell_ids_all,
        "y_true":  y_true,
        "y_pred":  y_pred,
        "epistemic": epi_all,
        "aleatoric": ale_all,
    }).to_csv(preds_path, index=False)

    print(f"\nExternal results -> {out_path}", flush=True)
    print(f"  PCC={rep['PCC']:.4f} | RMSE={rep['RMSE']:.4f} | n={rep['n_pairs']:,}",
          flush=True)


if __name__ == "__main__":
    main()
