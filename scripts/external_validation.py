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
    p.add_argument("--source_name", default="ccle_ctrpv2",
                   help="Used in output filenames")
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
    sample_g = next(iter(graph_cache.values()))
    model = PathXDRP(
        node_in_dim=sample_g.x.size(1),
        edge_in_dim=sample_g.edge_attr.size(1),
        n_genes=expr_train.shape[1],
        pathway_gene_map=pathway_gene_map,
    ).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    # --- Load + align external data ---
    print("Loading external CCLE expression", flush=True)
    expr_ext = load_external_expression(expr_path)
    expr_ext = align_genes_to_train(expr_ext, train_genes)

    # Apply our training-time mean/std (z-score using OUR statistics, not theirs)
    train_mean = expr_train.mean(axis=0)
    train_std  = expr_train.std(axis=0).replace(0, 1.0)
    expr_ext = (expr_ext - train_mean) / train_std

    print("Loading external response file", flush=True)
    resp = load_external_response(resp_path)

    # --- Match drug names ---
    name_to_id = {n.lower(): int(d) for n, d in zip(drugs_df["DRUG_NAME"], drugs_df["DRUG_ID"])}
    resp["matched_drug_id"] = resp["DRUG_NAME"].astype(str).str.lower().map(name_to_id)
    resp = resp.dropna(subset=["matched_drug_id"]).copy()
    resp["matched_drug_id"] = resp["matched_drug_id"].astype(int)
    print(f"  Matched {len(resp):,} response rows to {resp['matched_drug_id'].nunique()} train-time drugs",
          flush=True)

    # --- Predict ---
    y_true_all, y_pred_all, drug_ids_all, cell_ids_all = [], [], [], []
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
            batch    = Batch.from_data_list([graph_cache[drug_id]]).to(device)
            out = model(drug_batch=batch, expr=expr_row)
            y_true_all.append(float(r.y_true))
            y_pred_all.append(float(out["pred"]["pred"].item()))
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
    out_path  = RESULTS_DIR / f"{args.source_name}_{ckpt_name}.json"
    preds_path = RESULTS_DIR / f"{args.source_name}_{ckpt_name}_preds.csv"

    with open(out_path, "w") as f:
        json.dump({"args": vars(args), "test": rep}, f, indent=2)
    pd.DataFrame({
        "drug_id": drug_ids_all,
        "cell_id": cell_ids_all,
        "y_true":  y_true,
        "y_pred":  y_pred,
    }).to_csv(preds_path, index=False)

    print(f"\nExternal results -> {out_path}", flush=True)
    print(f"  PCC={rep['PCC']:.4f} | RMSE={rep['RMSE']:.4f} | n={rep['n_pairs']:,}",
          flush=True)


if __name__ == "__main__":
    main()
