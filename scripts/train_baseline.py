"""
Unified baseline training script.

Supports: graphdrp | drpreter | cdrscan

Usage
-----
GraphDRP:
  python scripts/train_baseline.py --model graphdrp \\
      --split random --seed 0 --fold 0 --epochs 150

DRPreter:
  python scripts/train_baseline.py --model drpreter \\
      --split cell_blind --seed 0 --fold 0

CDRScan:
  python scripts/train_baseline.py --model cdrscan \\
      --split random --seed 0 --fold 0

Results are saved to results/<model>/<split>_seed<s>_fold<f>.json
and use the same evaluation metrics as PathXDRP (regression_report).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

ROOT = Path(__file__).parent.parent
# Ensure the project root is on sys.path so `pathxdrp` is importable when this
# script is invoked as a subprocess (the CWD trick doesn't add it automatically).
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_IS_TTY = sys.stdout.isatty()


def _fmt_eta(seconds: float) -> str:
    if seconds < 0 or not np.isfinite(seconds):
        return "?"
    td = timedelta(seconds=int(seconds))
    s = str(td)
    return s if seconds >= 3600 else s.split(":", 1)[1]


def _now() -> str:
    return datetime.now().strftime("%H:%M:%S")


# ---
# Helpers shared with train.py
# ---

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---
# Train / eval loops (model-agnostic for graph-based baselines)
# ---

def train_epoch_graph(model, loader, optimizer, device, grad_clip=1.0,
                      epoch=0, n_epochs=0) -> float:
    model.train()
    total, n = 0.0, 0
    pbar = tqdm(
        loader, desc=f"  Epoch {epoch:3d}/{n_epochs} train",
        unit="batch", leave=False, dynamic_ncols=True,
        disable=not _IS_TTY,
    )
    for batch in pbar:
        optimizer.zero_grad()
        drug_batch = batch["drug_batch"].to(device)
        expr = batch["expr"].to(device)
        y    = batch["y"].to(device)
        out  = model(drug_batch=drug_batch, expr=expr, y=y)
        loss = out["loss"]
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total += loss.item() * len(y)
        n     += len(y)
        pbar.set_postfix(loss=f"{total / max(n, 1):.4f}")
    pbar.close()
    return total / n


@torch.no_grad()
def eval_graph(model, loader, device, desc="eval", return_predictions: bool = False) -> dict:
    model.eval()
    preds, trues, drug_ids, cosmic_ids = [], [], [], []
    pbar = tqdm(loader, desc=f"  {desc}", unit="batch", leave=False, dynamic_ncols=True,
                disable=not _IS_TTY)
    for batch in pbar:
        out = model(
            drug_batch=batch["drug_batch"].to(device),
            expr=batch["expr"].to(device),
        )
        preds.append(out["pred"].cpu().numpy())
        trues.append(batch["y"].numpy())
        drug_ids.append(batch["drug_ids"].numpy())
        cosmic_ids.append(batch["cosmic_ids"].numpy())
    pbar.close()
    from pathxdrp.eval.metrics import regression_report
    y_pred = np.concatenate(preds)
    y_true = np.concatenate(trues)
    drug_ids_np   = np.concatenate(drug_ids)
    cosmic_ids_np = np.concatenate(cosmic_ids)
    rep = regression_report(y_true, y_pred, drug_ids=drug_ids_np, cell_ids=cosmic_ids_np)
    if return_predictions:
        rep["_preds"] = {
            "y_true": y_true, "y_pred": y_pred,
            "drug_ids": drug_ids_np, "cosmic_ids": cosmic_ids_np,
        }
    return rep


def train_epoch_fp(model, loader, optimizer, device, grad_clip=1.0,
                   epoch=0, n_epochs=0) -> float:
    model.train()
    total, n = 0.0, 0
    pbar = tqdm(
        loader, desc=f"  Epoch {epoch:3d}/{n_epochs} train",
        unit="batch", leave=False, dynamic_ncols=True,
        disable=not _IS_TTY,
    )
    for batch in pbar:
        optimizer.zero_grad()
        fp   = batch["fp"].to(device)
        expr = batch["expr"].to(device)
        y    = batch["y"].to(device)
        out  = model(fp=fp, expr=expr, y=y)
        loss = out["loss"]
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total += loss.item() * len(y)
        n     += len(y)
        pbar.set_postfix(loss=f"{total / max(n, 1):.4f}")
    pbar.close()
    return total / n


@torch.no_grad()
def eval_fp(model, loader, device, desc="eval", return_predictions: bool = False) -> dict:
    model.eval()
    preds, trues, drug_ids, cosmic_ids = [], [], [], []
    pbar = tqdm(loader, desc=f"  {desc}", unit="batch", leave=False, dynamic_ncols=True,
                disable=not _IS_TTY)
    for batch in pbar:
        out = model(fp=batch["fp"].to(device), expr=batch["expr"].to(device))
        preds.append(out["pred"].cpu().numpy())
        trues.append(batch["y"].numpy())
        drug_ids.append(batch["drug_ids"].numpy())
        cosmic_ids.append(batch["cosmic_ids"].numpy())
    pbar.close()
    from pathxdrp.eval.metrics import regression_report
    y_pred = np.concatenate(preds)
    y_true = np.concatenate(trues)
    drug_ids_np   = np.concatenate(drug_ids)
    cosmic_ids_np = np.concatenate(cosmic_ids)
    rep = regression_report(y_true, y_pred, drug_ids=drug_ids_np, cell_ids=cosmic_ids_np)
    if return_predictions:
        rep["_preds"] = {
            "y_true": y_true, "y_pred": y_pred,
            "drug_ids": drug_ids_np, "cosmic_ids": cosmic_ids_np,
        }
    return rep


# ---
# Main
# ---

def main(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Model: {args.model}  |  Device: {device}", flush=True)

    # ---- Data ----
    from pathxdrp.data.loader import build_master_df
    from pathxdrp.data.splits import load_split

    print("Loading data", flush=True)
    df, expr_matrix = build_master_df(version=args.dataset, require_smiles=True)
    df = df[["DRUG_ID", "COSMIC_ID", "LN_IC50", "SMILES"]]
    train_idx, val_idx, test_idx = load_split(args.split, args.seed, args.fold)
    print(f"Split sizes ({args.split}/seed{args.seed}/fold{args.fold}): "
          f"train={len(train_idx):,} | val={len(val_idx):,} | test={len(test_idx):,}",
          flush=True)

    drugs_df = df[["DRUG_ID", "SMILES"]].drop_duplicates()

    # num_workers > 0 can deadlock on Windows
    import platform
    n_workers = 0 if platform.system() == "Windows" else 4

    # ---- CDRScan branch (fingerprint-based) ----
    if args.model == "cdrscan":
        from pathxdrp.baselines.cdrscan import (
            CDRScan, CDRScanDataset, build_fp_cache, cdrscan_collate,
        )
        print("Building fingerprint cache", flush=True)
        fp_cache = build_fp_cache(drugs_df)

        # df has a 0-based positional index (loader resets it), so iloc == loc.
        # Split indices are also positional (np.arange-based). Filter them to
        # rows whose DRUG_ID has a valid fingerprint using a boolean positional mask.
        fp_drug_set = set(fp_cache.keys())
        fp_mask = df["DRUG_ID"].isin(fp_drug_set).values   # (len(df),) bool

        train_idx_fp = train_idx[fp_mask[train_idx]]
        val_idx_fp   = val_idx[fp_mask[val_idx]]
        test_idx_fp  = test_idx[fp_mask[test_idx]]

        ds_kw = dict(fp_cache=fp_cache, expr_matrix=expr_matrix)
        train_ds = CDRScanDataset(df.iloc[train_idx_fp], **ds_kw)
        val_ds   = CDRScanDataset(df.iloc[val_idx_fp],   **ds_kw)
        test_ds  = CDRScanDataset(df.iloc[test_idx_fp],  **ds_kw)

        ldr_kw = dict(batch_size=args.batch_size, collate_fn=cdrscan_collate,
                      num_workers=n_workers, pin_memory=(device.type == "cuda"),
                      persistent_workers=(n_workers > 0))
        train_loader = DataLoader(train_ds, shuffle=True,  **ldr_kw)
        val_loader   = DataLoader(val_ds,   shuffle=False, **ldr_kw)
        test_loader  = DataLoader(test_ds,  shuffle=False, **ldr_kw)

        model = CDRScan(n_genes=expr_matrix.shape[1], hidden_dim=args.hidden_dim,
                        dropout=args.dropout).to(device)
        train_fn = train_epoch_fp
        eval_fn  = eval_fp

    # ---- Graph-based baselines (GraphDRP, DRPreter) ----
    else:
        from pathxdrp.train import GDSCDataset, collate_fn, build_graph_cache

        print("Building graph cache", flush=True)
        graph_cache, _ = build_graph_cache(drugs_df)

        ldr_kw = dict(batch_size=args.batch_size, collate_fn=collate_fn,
                      num_workers=n_workers, pin_memory=(device.type == "cuda"),
                      persistent_workers=(n_workers > 0))
        train_ds = GDSCDataset(df.iloc[train_idx], graph_cache, expr_matrix)
        val_ds   = GDSCDataset(df.iloc[val_idx],   graph_cache, expr_matrix)
        test_ds  = GDSCDataset(df.iloc[test_idx],  graph_cache, expr_matrix)
        train_loader = DataLoader(train_ds, shuffle=True,  **ldr_kw)
        val_loader   = DataLoader(val_ds,   shuffle=False, **ldr_kw)
        test_loader  = DataLoader(test_ds,  shuffle=False, **ldr_kw)

        sample_g = next(iter(graph_cache.values()))
        node_dim = sample_g.x.size(1)
        edge_dim = (
            sample_g.edge_attr.size(1)
            if sample_g.edge_attr is not None and sample_g.edge_attr.numel() > 0 else 9
        )
        n_genes = expr_matrix.shape[1]

        if args.model == "graphdrp":
            from pathxdrp.baselines.graphdrp import GraphDRP
            model = GraphDRP(
                node_in_dim=node_dim,
                n_genes=n_genes,
                hidden_dim=args.hidden_dim,
                n_gin_layers=args.n_gin_layers,
                dropout=args.dropout,
            ).to(device)

        elif args.model == "drpreter":
            pgm_path = ROOT / "data" / "processed" / "pathway_gene_map.json"
            if not pgm_path.exists():
                raise FileNotFoundError(
                    f"Pathway gene map not found: {pgm_path}\n"
                    "Run: python scripts/build_pathway_mask.py"
                )
            import json as _json
            with open(pgm_path) as f:
                pw_syms = _json.load(f)
            gene_list   = list(expr_matrix.columns)
            gene_to_idx = {g: i for i, g in enumerate(gene_list)}
            pathway_gene_map = {
                pw: [gene_to_idx[g] for g in gs if g in gene_to_idx]
                for pw, gs in pw_syms.items()
                if any(g in gene_to_idx for g in gs)
            }
            from pathxdrp.baselines.drpreter import DRPreter
            model = DRPreter(
                node_in_dim=node_dim,
                edge_in_dim=edge_dim,
                n_genes=n_genes,
                pathway_gene_map=pathway_gene_map,
                hidden_dim=args.hidden_dim,
                n_gat_layers=args.n_gat_layers,
                n_attn_heads=args.n_attn_heads,
                dropout=args.dropout,
            ).to(device)
        else:
            raise ValueError(f"Unknown model: {args.model}")

        train_fn = train_epoch_graph
        eval_fn  = eval_graph

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # Linear warmup then cosine decay. Warmup is important when batch_size is small
    # because per-step gradient noise is high; jumping straight to peak LR can cause
    # divergence (observed in v2 of the sweep recipe).
    _warmup_epochs   = min(10, max(2, args.epochs // 5))
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=_warmup_epochs
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs - _warmup_epochs, 1)
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[_warmup_epochs],
    )

    ckpt_dir = ROOT / "checkpoints" / args.model
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{args.split}_seed{args.seed}_fold{args.fold}.pt"

    best_val_pcc             = -float("inf")
    best_val_epoch           = 0
    epochs_since_improvement = 0
    train_start              = time.time()
    epoch_times              = []

    print(f"\nTraining {args.model.upper()} for {args.epochs} epochs "
          f"(split={args.split}, seed={args.seed}, fold={args.fold})", flush=True)

    epoch_pbar = tqdm(
        range(1, args.epochs + 1),
        desc=f"  {args.model.upper()} epochs",
        unit="ep", dynamic_ncols=True,
        # Always show this bar — it gives the only live signal when batch bars are off.
        # In non-TTY mode tqdm writes one line per update (no \r spam).
        disable=False,
    )

    for epoch in epoch_pbar:
        epoch_start = time.time()
        loss = train_fn(model, train_loader, optimizer, device,
                        epoch=epoch, n_epochs=args.epochs)
        val  = eval_fn(model, val_loader, device,
                       desc=f"Epoch {epoch:3d}/{args.epochs} val  ")
        scheduler.step()

        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        avg_epoch  = np.mean(epoch_times[-5:])
        eta        = avg_epoch * (args.epochs - epoch)

        improved = val["PCC"] > best_val_pcc
        if improved:
            best_val_pcc             = val["PCC"]
            best_val_epoch           = epoch
            epochs_since_improvement = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            epochs_since_improvement += 1

        marker = "*" if improved else " "
        lr_now = optimizer.param_groups[0]["lr"]
        epoch_pbar.set_postfix(
            loss=f"{loss:.4f}",
            PCC=f"{val['PCC']:.4f}",
            best=f"{best_val_pcc:.4f}@{best_val_epoch}",
            ETA=_fmt_eta(eta),
        )
        tqdm.write(
            f"[{_now()}] {marker} epoch {epoch:3d}/{args.epochs} | "
            f"loss={loss:.4f} | "
            f"RMSE={val['RMSE']:.4f} | "
            f"PCC={val['PCC']:.4f} | "
            f"Spr={val['Spearman']:.4f} | "
            f"R2={val['R2']:.4f} | "
            f"best_PCC={best_val_pcc:.4f}@{best_val_epoch} | "
            f"lr={lr_now:.1e} | "
            f"t={epoch_time:.1f}s | ETA={_fmt_eta(eta)}"
        )

        if args.early_stop_patience > 0 and epochs_since_improvement >= args.early_stop_patience:
            tqdm.write(f"\n[{_now()}] Early stop: no val PCC improvement for "
                       f"{args.early_stop_patience} epochs (best={best_val_pcc:.4f}@{best_val_epoch})")
            break

    epoch_pbar.close()

    total_time = time.time() - train_start
    print(f"\nTotal training time: {_fmt_eta(total_time)} "
          f"(avg {np.mean(epoch_times):.1f}s/epoch)", flush=True)
    print(f"Best val PCC: {best_val_pcc:.4f} at epoch {best_val_epoch}", flush=True)

    print(f"\nLoading best checkpoint (epoch {best_val_epoch})",
          flush=True)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    test = eval_fn(model, test_loader, device, desc="test eval", return_predictions=True)
    preds_blob = test.pop("_preds")

    print(f"\n--- test results: {args.model} / {args.split} / seed{args.seed} / fold{args.fold} ---",
          flush=True)
    for k, v in test.items():
        if isinstance(v, float):
            print(f"  {k:<18s} {v:.4f}", flush=True)
    print()

    results_dir = ROOT / "results" / args.model
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path   = results_dir / f"{args.split}_seed{args.seed}_fold{args.fold}.json"
    preds_path = results_dir / f"{args.split}_seed{args.seed}_fold{args.fold}_preds.csv"

    with open(out_path, "w") as f:
        json.dump({
            "args":            vars(args),
            "test":            test,
            "best_val_pcc":    best_val_pcc,
            "best_val_epoch":  best_val_epoch,
            "total_train_sec": float(total_time),
            "n_params":        int(n_params),
        }, f, indent=2)

    pd.DataFrame({
        "drug_id":   preds_blob["drug_ids"],
        "cosmic_id": preds_blob["cosmic_ids"],
        "y_true":    preds_blob["y_true"],
        "y_pred":    preds_blob["y_pred"],
    }).to_csv(preds_path, index=False)
    print(f"\nSaved -> {out_path}", flush=True)
    print(f"Predictions saved -> {preds_path}", flush=True)


# ---
# CLI
# ---

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train DRP baseline")
    p.add_argument("--model",    required=True, choices=["graphdrp", "drpreter", "cdrscan"])
    p.add_argument("--dataset",  default="GDSC2")
    p.add_argument("--split",    default="random",
                   choices=["random", "cell_blind", "drug_blind", "scaffold_blind", "tissue_blind"])
    p.add_argument("--seed",     type=int,   default=0)
    p.add_argument("--fold",     type=int,   default=0)
    p.add_argument("--hidden_dim",   type=int,   default=128)
    p.add_argument("--n_gin_layers", type=int,   default=5)
    p.add_argument("--n_gat_layers", type=int,   default=3)
    p.add_argument("--n_attn_heads", type=int,   default=8)
    p.add_argument("--dropout",  type=float, default=0.1)
    p.add_argument("--batch_size",   type=int,   default=256)
    p.add_argument("--epochs",   type=int,   default=150)
    p.add_argument("--lr",       type=float, default=1e-3)
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--early_stop_patience", type=int, default=0,
                   help="Stop if val PCC has not improved for this many epochs. 0 = disabled.")
    main(p.parse_args())
