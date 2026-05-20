"""
PathXDRP training entry point.

Loads the DepMap expression matrix and GDSC2 IC50 data, builds the full
PathXDRP model (expression mode only), trains with AdamW + cosine LR,
evaluates on a held-out test split, and saves results as JSON.

Prerequisites:
  1. python scripts/download_expression.py    (507 MB download)
  2. python scripts/build_pathway_mask.py     (KEGG gene-pathway map)
  3. python scripts/build_splits.py           (already done)

Usage:
  python -m pathxdrp.train \\
      --split random --seed 0 --fold 0 \\
      --hidden_dim 128 --n_gat_layers 3 --epochs 150
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
import warnings

# torch-scatter is a legacy C++ extension; PyTorch 2.0+ has native scatter_reduce.
# PyG still warns about it, but there is no actual performance loss on modern PyTorch.
warnings.filterwarnings("ignore", message=".*torch-scatter.*", category=UserWarning)
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch
from tqdm.auto import tqdm

ROOT = Path(__file__).parent.parent
_IS_TTY = sys.stderr.isatty()  # False when piped; suppresses per-batch spam


# --- Pretty-printing utilities ---

def _fmt_eta(seconds: float) -> str:
    """Format seconds as H:MM:SS, dropping the hours when < 1h."""
    if seconds < 0 or not np.isfinite(seconds):
        return "?"
    td = timedelta(seconds=int(seconds))
    s = str(td)
    return s if seconds >= 3600 else s.split(":", 1)[1]  # MM:SS for under an hour


def _now() -> str:
    return datetime.now().strftime("%H:%M:%S")


# --- Dataset ---

class GDSCDataset(Dataset):
    """
    Yields one (drug_graph, expr_vector, y) triple per IC50 row.

    Args:
        df:           Master DataFrame subset (train / val / test).
        graph_cache:  {DRUG_ID: PyG Data} — pre-built molecular graphs.
        expr_matrix:  pd.DataFrame indexed by COSMIC_ID, columns = gene symbols,
                      values = Z-scored log2(TPM+1), dtype float32.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        graph_cache: dict,
        expr_matrix: pd.DataFrame,
        fp_cache: dict | None = None,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.graph_cache = graph_cache
        self.fp_cache = fp_cache or {}
        # Convert to numpy for fast row lookups
        self.expr_np = expr_matrix.values.astype("float32")          # (n_cells, n_genes)
        self.cosmic_to_row: dict[int, int] = {
            int(cid): i for i, cid in enumerate(expr_matrix.index)
        }

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        drug_id  = int(row["DRUG_ID"])
        cosmic_id = int(row["COSMIC_ID"])

        drug_graph = self.graph_cache[drug_id]
        expr_row   = self.expr_np[self.cosmic_to_row[cosmic_id]]  # (n_genes,)
        # Optional Morgan fingerprint per drug (when --use_morgan_fp is set).
        # Returns a fixed-shape vector (2048,) of float32; otherwise None.
        morgan_fp = self.fp_cache.get(drug_id) if self.fp_cache else None

        return {
            "drug_graph": drug_graph,
            "expr":       expr_row,
            "morgan_fp":  morgan_fp,
            "y":          float(row["LN_IC50"]),
            "auc":        float(row["AUC"]) if "AUC" in self.df.columns and pd.notna(row["AUC"]) else None,
            "drug_id":    drug_id,
            "cosmic_id":  cosmic_id,
        }


def collate_fn(batch: list[dict]) -> dict:
    drug_batch = Batch.from_data_list([b["drug_graph"] for b in batch])
    out = {
        "drug_batch": drug_batch,
        "expr":       torch.tensor(
                          np.stack([b["expr"] for b in batch]), dtype=torch.float
                      ),
        "y":          torch.tensor([b["y"] for b in batch], dtype=torch.float),
        "drug_ids":   torch.tensor([b["drug_id"] for b in batch], dtype=torch.long),
        "cosmic_ids": torch.tensor([b["cosmic_id"] for b in batch], dtype=torch.long),
    }
    # Morgan fingerprints (only when the dataset was built with a fp_cache)
    fps = [b.get("morgan_fp") for b in batch]
    if all(fp is not None for fp in fps):
        out["morgan_fp"] = torch.tensor(np.stack(fps), dtype=torch.float)
    # AUC auxiliary target
    aucs = [b.get("auc") for b in batch]
    if all(a is not None for a in aucs):
        out["auc"] = torch.tensor(aucs, dtype=torch.float)
    return out


# --- Graph cache ---

def build_graph_cache(drugs_df: pd.DataFrame) -> tuple[dict, dict]:
    """
    Build {DRUG_ID: PyG Data} for all drugs with valid SMILES.
    Returns (graph_cache, fg_vocab).
    """
    from pathxdrp.data.graph_utils import build_fg_vocab, smiles_to_graph

    smiles_list = drugs_df["SMILES"].dropna().tolist()
    print(f"  Building Morgan FG vocab from {len(smiles_list)} SMILES")
    fg_vocab = build_fg_vocab(smiles_list)
    print(f"  FG vocab size: {len(fg_vocab)}")

    cache: dict = {}
    n_failed = 0
    pbar = tqdm(
        drugs_df.iterrows(), total=len(drugs_df),
        desc="  Building drug graphs", unit="drug",
        disable=not _IS_TTY,
    )
    for _, row in pbar:
        smi = row["SMILES"]
        if pd.isna(smi):
            n_failed += 1
            continue
        g = smiles_to_graph(smi, fg_vocab=fg_vocab, label=None)
        if g is not None:
            cache[int(row["DRUG_ID"])] = g
        else:
            n_failed += 1
        pbar.set_postfix(ok=len(cache), failed=n_failed)
    pbar.close()

    print(f"  Graph cache: {len(cache)} drugs ({n_failed} failed/skipped)", flush=True)
    return cache, fg_vocab


# --- Training / evaluation ---

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    amp_dtype: torch.dtype = torch.float32,
    grad_clip: float = 1.0,
    epoch: int = 0,
    n_epochs: int = 0,
    max_skip_frac: float = 0.10,
    mixup_alpha: float = 0.0,
    mixup_prob: float = 0.5,
    expr_noise_std: float = 0.0,
    aux_auc_weight: float = 0.0,
) -> float:
    model.train()
    total_loss = 0.0
    n_samples = 0
    use_amp = (amp_dtype != torch.float32) and (device.type == "cuda")

    pbar = tqdm(
        loader,
        desc=f"  Epoch {epoch:3d}/{n_epochs} train",
        unit="batch", leave=False, dynamic_ncols=True,
        disable=not _IS_TTY,
    )
    n_skipped = 0
    n_total = 0
    for batch in pbar:
        n_total += 1
        optimizer.zero_grad()
        drug_batch = batch["drug_batch"].to(device)
        expr = batch["expr"].to(device)
        y    = batch["y"].to(device)
        morgan_fp = batch.get("morgan_fp")
        if morgan_fp is not None:
            morgan_fp = morgan_fp.to(device)

        # ---- Expression input augmentation (Gaussian noise) ----
        # The expression matrix is Z-scored per gene (std=1), so noise with
        # std=0.05 corresponds to ~5% of the per-gene scale. Mild, training-only.
        if expr_noise_std > 0.0 and model.training:
            expr = expr + expr_noise_std * torch.randn_like(expr)

        # ---- Mixup on (expression, label) ----
        # Apply mixup PROBABILISTICALLY: every batch flips a coin (p=mixup_prob),
        # and mixup is used when the coin says yes; otherwise the standard
        # evidential loss is used. Without this gate, every batch trained MSE
        # on the prediction mean and the evidential head (ν, α, β) NEVER got
        # gradient signal — calibration metrics became noise.
        # Beta(α, α) interpolates expression and label; the drug graph stays
        # fixed because mixing molecular graphs is not well-defined.
        mixup_active = (
            mixup_alpha > 0.0
            and model.training
            and float(np.random.rand()) < mixup_prob
        )
        if mixup_active:
            lam = float(np.random.beta(mixup_alpha, mixup_alpha))
            perm = torch.randperm(expr.size(0), device=expr.device)
            expr_mixed = lam * expr + (1.0 - lam) * expr[perm]
            y_mixed    = lam * y    + (1.0 - lam) * y[perm]
            with torch.amp.autocast(device_type="cuda", enabled=use_amp, dtype=amp_dtype):
                out = model(drug_batch=drug_batch, expr=expr_mixed, morgan_fp=morgan_fp)
                pred = out["pred"]["pred"]
                loss = F.mse_loss(pred, y_mixed)
        else:
            auc_batch = batch.get("auc")
            if auc_batch is not None and aux_auc_weight > 0.0:
                auc_batch = auc_batch.to(device)
            else:
                auc_batch = None
            with torch.amp.autocast(device_type="cuda", enabled=use_amp, dtype=amp_dtype):
                out  = model(drug_batch=drug_batch, expr=expr, y=y, auc=auc_batch, morgan_fp=morgan_fp)
                loss = out["loss"]

        if not torch.isfinite(loss):
            n_skipped += 1
            optimizer.zero_grad()
            continue

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item() * len(y)
        n_samples  += len(y)
        pbar.set_postfix(loss=f"{total_loss / max(n_samples, 1):.4f}",
                         skip=n_skipped if n_skipped else None)

    skip_frac = n_skipped / max(n_total, 1)
    if n_skipped:
        tqdm.write(f"  [epoch {epoch}] skipped {n_skipped}/{n_total} batches "
                   f"({skip_frac*100:.1f}%) with non-finite loss")
    if skip_frac > max_skip_frac:
        raise RuntimeError(
            f"Training divergence detected at epoch {epoch}: "
            f"{n_skipped}/{n_total} ({skip_frac*100:.1f}%) batches produced non-finite loss, "
            f"exceeding the {max_skip_frac*100:.0f}% threshold. "
            f"Recommended actions: (1) ensure --precision is bf16 or fp32 (NOT fp16); "
            f"(2) lower --lr; (3) lower --evidential_lam or extend --lam_warmup_epochs."
        )
    pbar.close()

    return total_loss / n_samples


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    desc: str = "eval",
    return_predictions: bool = False,
    amp_dtype: torch.dtype = torch.float32,
) -> dict:
    """
    Run the model on `loader` and return a metric report.

    When `return_predictions=True`, the dict also contains
        "_preds": dict with raw arrays (y_true, y_pred, epistemic, aleatoric,
                  drug_ids, cosmic_ids)
    so callers can dump them to CSV for risk-coverage / uncertainty plots.
    """
    model.eval()
    preds, trues, drug_ids_list, cosmic_ids_list = [], [], [], []
    epistemic_list, aleatoric_list = [], []
    use_amp = (amp_dtype != torch.float32) and (device.type == "cuda")

    pbar = tqdm(
        loader, desc=f"  {desc}",
        unit="batch", leave=False, dynamic_ncols=True,
        disable=not _IS_TTY,
    )
    for batch in pbar:
        drug_batch = batch["drug_batch"].to(device)
        expr       = batch["expr"].to(device)
        morgan_fp = batch.get("morgan_fp")
        if morgan_fp is not None:
            morgan_fp = morgan_fp.to(device)

        with torch.amp.autocast(device_type="cuda", enabled=use_amp, dtype=amp_dtype):
            out = model(drug_batch=drug_batch, expr=expr, morgan_fp=morgan_fp)
        preds.append(out["pred"]["pred"].cpu().numpy())
        trues.append(batch["y"].numpy())
        epistemic_list.append(out["pred"]["epistemic"].cpu().numpy())
        aleatoric_list.append(out["pred"]["aleatoric"].cpu().numpy())
        drug_ids_list.append(batch["drug_ids"].numpy())
        cosmic_ids_list.append(batch["cosmic_ids"].numpy())
    pbar.close()

    from pathxdrp.eval.metrics import regression_report
    y_pred     = np.concatenate(preds)
    y_true     = np.concatenate(trues)
    drug_ids   = np.concatenate(drug_ids_list)
    cosmic_ids = np.concatenate(cosmic_ids_list)
    epistemic  = np.concatenate(epistemic_list)
    aleatoric  = np.concatenate(aleatoric_list)

    n_bad = int((~np.isfinite(y_pred)).sum())
    if n_bad > 0:
        y_mean = float(np.nanmean(y_true))
        y_lo   = float(np.nanmin(y_true))
        y_hi   = float(np.nanmax(y_true))
        print(f"  WARNING [{desc}]: {n_bad}/{len(y_pred)} non-finite predictions; "
              f"clipping to [y_min, y_max] for metrics", flush=True)
        y_pred    = np.nan_to_num(y_pred,    nan=y_mean, posinf=y_hi, neginf=y_lo)
        epistemic = np.nan_to_num(epistemic, nan=0.0,    posinf=0.0,  neginf=0.0)
        aleatoric = np.nan_to_num(aleatoric, nan=0.0,    posinf=0.0,  neginf=0.0)

    report = regression_report(
        y_true, y_pred,
        drug_ids=drug_ids,
        cell_ids=cosmic_ids,
        uncertainties=epistemic,
    )
    report["epistemic_mean"] = float(np.nan_to_num(epistemic, nan=0.0, posinf=0.0).mean())
    report["aleatoric_mean"] = float(np.nan_to_num(aleatoric, nan=0.0, posinf=0.0).mean())

    if return_predictions:
        report["_preds"] = {
            "y_true":     y_true,
            "y_pred":     y_pred,
            "epistemic":  epistemic,
            "aleatoric":  aleatoric,
            "drug_ids":   drug_ids,
            "cosmic_ids": cosmic_ids,
        }
    return report


# --- Main ---

def main(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    # ---- Data ----
    from pathxdrp.data.loader import build_master_df
    from pathxdrp.data.splits import load_split

    print(f"\nLoading data", flush=True)
    df, expr_matrix = build_master_df(version=args.dataset, require_smiles=True)
    df = df[["DRUG_ID", "COSMIC_ID", "LN_IC50", "AUC", "SMILES"]]
    n_genes = expr_matrix.shape[1]
    print(f"  df: {len(df):,} rows | expr_matrix: {expr_matrix.shape}", flush=True)

    # ---- Pathway gene map ----
    pgm_path = ROOT / "data" / "processed" / "pathway_gene_map.json"
    if not pgm_path.exists():
        raise FileNotFoundError(
            f"Pathway gene map not found: {pgm_path}\n"
            "Run: python scripts/build_pathway_mask.py"
        )
    with open(pgm_path) as f:
        pathway_gene_symbols: dict[str, list[str]] = json.load(f)

    # Convert gene symbols -> indices relative to expression matrix columns
    gene_list   = list(expr_matrix.columns)
    gene_to_idx = {g: i for i, g in enumerate(gene_list)}
    pathway_gene_map = {
        pw: [gene_to_idx[g] for g in genes if g in gene_to_idx]
        for pw, genes in pathway_gene_symbols.items()
        if any(g in gene_to_idx for g in genes)
    }
    n_pairs       = sum(len(v) for v in pathway_gene_map.values())
    avg_per_pw    = n_pairs / max(len(pathway_gene_map), 1)
    unique_genes  = len({g for genes in pathway_gene_map.values() for g in genes})
    print(f"Pathway gene map: {len(pathway_gene_map)} pathways | "
          f"{n_pairs:,} (pathway, gene) pairs | "
          f"{unique_genes:,} unique genes | avg {avg_per_pw:.1f} genes/pathway")

    # ---- Molecular graph cache ----
    print("Building graph cache", flush=True)
    drugs_df = df[["DRUG_ID", "SMILES"]].drop_duplicates()
    graph_cache, _ = build_graph_cache(drugs_df)
    # Optional Morgan-fingerprint cache for the global drug FP feature.
    fp_cache: dict = {}
    if args.use_morgan_fp:
        from pathxdrp.baselines.cdrscan import build_fp_cache
        print("Building fingerprint cache (2048-bit, radius=2)", flush=True)
        fp_cache = build_fp_cache(drugs_df)
        print(f"  FP cache: {len(fp_cache)} drugs")
    print(f"Graph cache: {len(graph_cache)} drugs")

    # ---- Splits ----
    train_idx, val_idx, test_idx = load_split(args.split, args.seed, args.fold)
    print(f"Split sizes ({args.split}/seed{args.seed}/fold{args.fold}): "
          f"train={len(train_idx):,} | val={len(val_idx):,} | test={len(test_idx):,}")

    train_ds = GDSCDataset(df.iloc[train_idx], graph_cache, expr_matrix, fp_cache=fp_cache)
    val_ds   = GDSCDataset(df.iloc[val_idx],   graph_cache, expr_matrix, fp_cache=fp_cache)
    test_ds  = GDSCDataset(df.iloc[test_idx],  graph_cache, expr_matrix, fp_cache=fp_cache)

    # num_workers > 0 can deadlock on Windows; use 0 on win32, 4 elsewhere
    import platform
    n_workers = 0 if platform.system() == "Windows" else 4
    loader_kwargs = dict(
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        num_workers=n_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(n_workers > 0),
    )
    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(test_ds,  shuffle=False, **loader_kwargs)

    # ---- Model ----
    sample_g = next(iter(graph_cache.values()))
    node_dim = sample_g.x.size(1)
    edge_dim = (
        sample_g.edge_attr.size(1)
        if sample_g.edge_attr is not None and sample_g.edge_attr.numel() > 0
        else 9
    )

    from pathxdrp.models.pathxdrp import PathXDRP
    model = PathXDRP(
        node_in_dim=node_dim,
        edge_in_dim=edge_dim,
        n_genes=n_genes,
        pathway_gene_map=pathway_gene_map,
        hidden_dim=args.hidden_dim,
        n_gat_layers=args.n_gat_layers,
        n_attn_heads=args.n_attn_heads,
        dropout=args.dropout,
        mask_type=args.mask_type,
        evidential_lam=args.evidential_lam,
        n_pw_transformer_layers=args.n_pw_transformer_layers,
        frac_active_sharpness=args.frac_active_sharpness,
        use_morgan_fp=args.use_morgan_fp,
        aux_auc_weight=args.aux_auc_weight,
        cross_attn_residual=args.cross_attn_residual,
        drop_h_mol=args.drop_h_mol,
        attn_aux_weight=args.attn_aux_weight,
        # Phase 4 encoder switches
        drug_encoder_type=args.drug_encoder_type,
        cell_encoder_type=args.cell_encoder_type,
        gene_symbols=gene_list if args.cell_encoder_type == "gene_mamba" else None,
        graph_mamba_kwargs={
            "n_gat_layers":   args.gm_n_gat_layers,
            "n_mamba_layers": args.gm_n_mamba_layers,
            "ordering":       args.gm_ordering,
        } if args.drug_encoder_type == "graph_mamba" else None,
        gene_mamba_kwargs={
            "backbone_id":     args.gnm_backbone_id,
            "top_k":           args.gnm_top_k,
            "freeze_backbone": args.gnm_freeze_backbone,
        } if args.cell_encoder_type == "gene_mamba" else None,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}", flush=True)

    # ---- Precision setup ----
    # bf16 is the default on CUDA: same exponent range as fp32 (no overflow in
    # the evidential head's log/Gamma operations) without the fp16 instability
    # that produced runaway NaN cascades. fp32 is the safe fallback.
    if device.type == "cuda" and args.precision == "bf16":
        amp_dtype = torch.bfloat16
    elif device.type == "cuda" and args.precision == "fp16":
        amp_dtype = torch.float16
        print("WARNING: fp16 is unstable for the evidential head; "
              "consider --precision bf16 instead.", flush=True)
    else:
        amp_dtype = torch.float32
    print(f"Precision: {args.precision} (autocast dtype: {amp_dtype})", flush=True)

    # ---- Optimiser + scheduler ----
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-4
    )
    # 10-epoch linear warmup then cosine decay over remaining epochs.
    # Warmup prevents large gradient steps early when the new h_mol/cell_pool_q
    # parameters are randomly initialised and the loss landscape is steep.
    # Bumped from 5 to 10 in v3 because batch=64 produces noisier gradients;
    # the model needs more low-LR epochs to settle before reaching peak LR.
    _warmup_epochs = min(10, max(2, args.epochs // 5))
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

    # ---- Stochastic Weight Averaging (SWA) ----
    # When enabled (--swa_start_epoch >= 1), maintain a running average of model
    # weights starting from swa_start_epoch. After the final epoch, the averaged
    # model is used for test evaluation (LayerNorm stats updated with a single
    # epoch sweep over the train loader). Reliably gives +0.005-0.015 PCC on
    # regression tasks by converging to flatter minima.
    from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
    swa_active = args.swa_start_epoch > 0
    swa_model: AveragedModel | None = None
    swa_scheduler: SWALR | None = None
    if swa_active:
        swa_model = AveragedModel(model)
        swa_scheduler = SWALR(optimizer, swa_lr=args.lr * 0.1, anneal_epochs=3,
                              anneal_strategy="linear")
        print(f"SWA: enabled, start_epoch={args.swa_start_epoch}, "
              f"swa_lr={args.lr * 0.1:.2e}", flush=True)

    # ---- Evidential lambda schedule ----
    # The NIG regularisation term |y−γ|·(2ν+α) grows as the model gains
    # confidence, creating an increasingly large counter-gradient that suppresses
    # prediction accuracy.  We ramp lambda from 0 → target over the first
    # lam_warmup_epochs, letting the model first learn a solid MSE-like
    # representation before adding calibration pressure.
    _lam_warmup = args.lam_warmup_epochs  # 0 disables warmup (instant full lambda)
    def _get_lam(epoch: int) -> float:
        if _lam_warmup <= 0:
            return args.evidential_lam
        frac = min(epoch / _lam_warmup, 1.0)
        return args.evidential_lam * frac

    # ---- Output naming ----
    # When --run_tag is set, suffix every output (checkpoint, results JSON,
    # predictions CSV) so different model variants don't overwrite each other.
    tag_suffix = f"_{args.run_tag}" if args.run_tag else ""
    base_stem  = f"{args.split}_seed{args.seed}_fold{args.fold}{tag_suffix}"

    # ---- Checkpoint path ----
    ckpt_dir = ROOT / "checkpoints" / "pathxdrp"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{base_stem}.pt"

    best_val_pcc            = -float("inf")
    best_val_epoch          = 0
    epochs_since_improvement = 0
    train_start             = time.time()
    epoch_times             = []

    # ---- Training loop ----
    print(f"\nTraining PathXDRP for {args.epochs} epochs "
          f"(split={args.split}, seed={args.seed}, fold={args.fold})", flush=True)

    epoch_pbar = tqdm(
        range(1, args.epochs + 1),
        desc="  PathXDRP epochs",
        unit="ep", dynamic_ncols=True,
        disable=False,
    )

    for epoch in epoch_pbar:
        epoch_start = time.time()

        # Update evidential lambda on the model for this epoch
        current_lam = _get_lam(epoch)
        model.evidential_lam = current_lam

        train_loss  = train_one_epoch(
            model, train_loader, optimizer, device,
            amp_dtype=amp_dtype,
            epoch=epoch, n_epochs=args.epochs,
            mixup_alpha=args.mixup_alpha,
            mixup_prob=args.mixup_prob,
            expr_noise_std=args.expr_noise_std,
            aux_auc_weight=args.aux_auc_weight,
        )
        val_metrics = evaluate(model, val_loader, device,
                               desc=f"Epoch {epoch:3d}/{args.epochs} val  ",
                               amp_dtype=amp_dtype)
        # Once SWA phase starts, the SWA scheduler takes over (constant lr).
        # Both schedulers stepping after the same optimizer step is fine since
        # we only call .step() on one of them per epoch.
        if swa_active and epoch >= args.swa_start_epoch:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            scheduler.step()

        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        avg_epoch  = np.mean(epoch_times[-5:])  # smooth ETA over last 5 epochs
        eta        = avg_epoch * (args.epochs - epoch)

        improved = val_metrics["PCC"] > best_val_pcc
        if improved:
            best_val_pcc             = val_metrics["PCC"]
            best_val_epoch           = epoch
            epochs_since_improvement = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            epochs_since_improvement += 1

        marker = "*" if improved else " "
        lr_now = optimizer.param_groups[0]["lr"]
        epoch_pbar.set_postfix(
            loss=f"{train_loss:.4f}",
            PCC=f"{val_metrics['PCC']:.4f}",
            best=f"{best_val_pcc:.4f}@{best_val_epoch}",
            ETA=_fmt_eta(eta),
        )
        tqdm.write(
            f"[{_now()}] {marker} epoch {epoch:3d}/{args.epochs} | "
            f"loss={train_loss:.4f} | "
            f"RMSE={val_metrics['RMSE']:.4f} | "
            f"PCC={val_metrics['PCC']:.4f} | "
            f"Spr={val_metrics['Spearman']:.4f} | "
            f"R2={val_metrics['R2']:.4f} | "
            f"epi={val_metrics['epistemic_mean']:.4f} | "
            f"best_PCC={best_val_pcc:.4f}@{best_val_epoch} | "
            f"lam={current_lam:.4f} | "
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
          f"(avg {np.mean(epoch_times):.1f}s/epoch over {len(epoch_times)} epochs)", flush=True)
    print(f"Best val PCC: {best_val_pcc:.4f} at epoch {best_val_epoch}", flush=True)

    # ---- Test evaluation (full predictions captured) ----
    # If SWA was active, use the SWA-averaged weights for test evaluation
    # (and save them as an additional checkpoint). Otherwise fall back to the
    # best-val checkpoint.
    if swa_active and swa_model is not None:
        # Skip update_bn for LayerNorm-only architectures: it sweeps the train
        # loader once (~10 min on this dataset) to refresh BN running stats
        # that don't exist. Detect explicitly so we don't silently waste time
        # on a model that adds BatchNorm later.
        has_bn = any(
            isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))
            for m in swa_model.modules()
        )
        if has_bn:
            print(f"\nUpdating BatchNorm statistics for SWA model", flush=True)
            try:
                update_bn(train_loader, swa_model, device=device)
            except Exception as e:
                print(f"  (update_bn skipped: {e})", flush=True)
        else:
            print(f"\nSkipping update_bn (model has no BatchNorm)", flush=True)
        # Sanity warning: SWA assumes near-flat-loss convergence. If we hit our
        # best val PCC very early (typical of overfitting on hard splits), the
        # SWA average is dominated by post-overfit weights and can hurt test
        # performance. The fix is per-run (raise --swa_start_epoch), not code.
        if best_val_epoch < args.swa_start_epoch:
            print(
                f"  WARNING: best val PCC was at epoch {best_val_epoch}, before SWA started "
                f"at {args.swa_start_epoch}. SWA averaged weights from worse-than-best "
                f"epochs and may underperform the best-val checkpoint. Consider "
                f"--swa_start_epoch <= {best_val_epoch} or skipping SWA.",
                flush=True,
            )
        eval_model = swa_model
        swa_ckpt = ckpt_dir / f"{base_stem}_swa.pt"
        torch.save(swa_model.module.state_dict(), swa_ckpt)
        print(f"  SWA weights saved -> {swa_ckpt}")
        print(f"  Using SWA model for test eval")
    else:
        print(f"\nLoading best checkpoint (epoch {best_val_epoch})", flush=True)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        eval_model = model

    test_metrics = evaluate(
        eval_model, test_loader, device, desc="test eval", return_predictions=True, amp_dtype=amp_dtype,
    )
    preds_blob = test_metrics.pop("_preds")  # raw arrays, not JSON-serialisable

    print(f"\n--- test results: {args.split} / seed{args.seed} / fold{args.fold} ---", flush=True)
    for k, v in test_metrics.items():
        if isinstance(v, float):
            print(f"  {k:<18s} {v:.4f}", flush=True)
    print()

    # ---- Persist results JSON (metrics) + predictions CSV (raw) ----
    results_dir = ROOT / "results" / "pathxdrp"
    results_dir.mkdir(parents=True, exist_ok=True)

    out_path   = results_dir / f"{base_stem}.json"
    preds_path = results_dir / f"{base_stem}_preds.csv"

    with open(out_path, "w") as f:
        json.dump(
            {
                "args":            vars(args),
                "test":            test_metrics,
                "best_val_pcc":    best_val_pcc,
                "best_val_epoch":  best_val_epoch,
                "total_train_sec": float(total_time),
                "n_params":        int(n_params),
            },
            f,
            indent=2,
        )

    pd.DataFrame({
        "drug_id":   preds_blob["drug_ids"],
        "cosmic_id": preds_blob["cosmic_ids"],
        "y_true":    preds_blob["y_true"],
        "y_pred":    preds_blob["y_pred"],
        "epistemic": preds_blob["epistemic"],
        "aleatoric": preds_blob["aleatoric"],
    }).to_csv(preds_path, index=False)
    print(f"Predictions saved -> {preds_path}", flush=True)
    print(f"\nResults saved -> {out_path}", flush=True)


# --- CLI ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PathXDRP")
    parser.add_argument("--dataset",       default="GDSC2")
    parser.add_argument(
        "--split", default="random",
        choices=["random", "cell_blind", "drug_blind", "scaffold_blind", "tissue_blind"],
    )
    parser.add_argument("--seed",          type=int,   default=0)
    parser.add_argument("--fold",          type=int,   default=0)
    parser.add_argument("--hidden_dim",    type=int,   default=256)
    parser.add_argument("--n_gat_layers",  type=int,   default=4)
    parser.add_argument("--n_attn_heads",  type=int,   default=8)
    parser.add_argument("--dropout",       type=float, default=0.1)
    parser.add_argument("--mask_type",     default="none", choices=["hard", "soft", "none"],
                        help="Cross-attention pathway-mask type. Default 'none' (no mask): "
                             "we found that the 'soft' learned bias is drug-independent and "
                             "collapses attention to a constant distribution across drugs.")
    parser.add_argument("--cross_attn_residual", action="store_true",
                        help="Add a residual + LayerNorm around the cross-attention output "
                             "(standard transformer pattern) and switch the pooling from the "
                             "max-attention-weighted form to plain mean pool. Required for the "
                             "attention weights to be faithful (load-bearing); without it, the "
                             "head can ignore the attention output entirely. New checkpoints "
                             "trained with this flag are not loadable by the old code path.")
    parser.add_argument("--drop_h_mol", action="store_true",
                        help="Drop the parallel GAT global readout (h_mol) from the head input. "
                             "Forces all drug information to flow through the cross-attention "
                             "path, raising attention faithfulness 2-3x at a small expected PCC "
                             "cost (~0.005). Requires --cross_attn_residual to make sense.")
    parser.add_argument("--attn_aux_weight", type=float, default=0.0,
                        help="Weight of the attention-only auxiliary loss. When >0, an MLP "
                             "predicts LN_IC50 from h_drug_context alone; this loss is added "
                             "with the given weight to force the post-attention representation "
                             "to be predictive by itself. 0.3 is a sane starting value. "
                             "Improves attention faithfulness directly without retraining the "
                             "main head architecture.")
    parser.add_argument("--evidential_lam",      type=float, default=0.01,
                        help="Evidential NIG regularisation weight (final value)")
    parser.add_argument("--lam_warmup_epochs",   type=int,   default=50,
                        help="Ramp lambda linearly from 0 → evidential_lam over this many epochs; "
                             "0 = use full lambda from epoch 1")
    parser.add_argument("--batch_size",    type=int,   default=256)
    parser.add_argument("--epochs",        type=int,   default=150)
    parser.add_argument("--lr",            type=float, default=1e-3)
    parser.add_argument("--early_stop_patience", type=int, default=0,
                        help="Stop if val PCC has not improved for this many epochs. "
                             "0 = disabled. Recommended: 10 for hard splits, 0 for random.")
    parser.add_argument("--precision", default="bf16", choices=["bf16", "fp32", "fp16"],
                        help="Numerical precision for autocast. bf16 (default) has fp32 exponent "
                             "range and is stable for evidential heads. fp16 is fast but unstable. "
                             "fp32 is the slowest but safest fallback.")
    parser.add_argument("--log_interval",  type=int,   default=10)
    parser.add_argument("--n_pw_transformer_layers", type=int, default=2,
                        help="Cross-pathway TransformerEncoder layers in PathwaySetEncoder "
                             "(0 = disabled, 1 = original, 2 = default: deeper cross-pathway "
                             "attention improves cell-blind and drug-blind generalisation)")
    # Phase 4 encoder switches
    parser.add_argument("--drug_encoder_type", default="gat",
                        choices=["gat", "molformer", "graph_mamba"])
    parser.add_argument("--cell_encoder_type", default="pathway_set",
                        choices=["pathway_set", "gene_mamba", "scgpt"])
    # Graph-Mamba drug-encoder hyperparams
    parser.add_argument("--gm_n_gat_layers",   type=int, default=2)
    parser.add_argument("--gm_n_mamba_layers", type=int, default=2)
    parser.add_argument("--gm_ordering",       default="degree",
                        choices=["degree", "canonical", "random"])
    # GeneMamba cell-encoder hyperparams
    parser.add_argument("--gnm_backbone_id",   default="mineself2016/GeneMamba")
    parser.add_argument("--gnm_top_k",         type=int, default=2048)
    parser.add_argument("--gnm_freeze_backbone", action="store_true", default=True)
    # Global Morgan fingerprint feature (CDRScan-style 2048-bit drug descriptor)
    parser.add_argument("--use_morgan_fp", action="store_true",
                        help="Concatenate a projected 2048-bit Morgan radius-2 fingerprint "
                             "to the drug-level embedding. Adds CDRScan's substructure prior "
                             "alongside the GAT's learned features. Helps drug-blind splits.")
    # Optimization / regularisation techniques
    parser.add_argument("--mixup_alpha", type=float, default=0.0,
                        help="Mixup interpolation strength (Beta(α,α)) on (expression, label). "
                             "Drug graph stays fixed. 0 disables mixup; 0.2 is a mild default.")
    parser.add_argument("--mixup_prob",  type=float, default=0.5,
                        help="Probability that any given batch uses mixup. The remaining batches "
                             "train the evidential head normally. Without this gate, mixup "
                             "starves the evidential head (ν, α, β) of gradient signal and "
                             "uncertainty calibration becomes noise. 0.5 is a balanced default.")
    parser.add_argument("--expr_noise_std", type=float, default=0.0,
                        help="Std of Gaussian noise added to Z-scored expression during training. "
                             "0 disables; 0.05 ~= 5%% of per-gene scale and is a typical value. "
                             "Pair with --frac_active_sharpness > 0 to avoid noise flipping the "
                             "frac_active boolean for the ~10%% of genes near the Z=0 threshold.")
    parser.add_argument("--frac_active_sharpness", type=float, default=0.0,
                        help="Sigmoid sharpness for the frac_active pathway statistic. 0 (default) "
                             "uses the historical hard threshold (x>0). >0 uses sigmoid(k*x); "
                             "10 is a sane value that smooths the boundary without losing signal. "
                             "Only matters when --expr_noise_std > 0.")
    parser.add_argument("--swa_start_epoch", type=int, default=0,
                        help="Epoch at which to start Stochastic Weight Averaging. "
                             "0 disables SWA. Typical: epochs//1.5 (e.g. 30 if epochs=50). "
                             "After training, the SWA-averaged weights are used for test eval.")
    parser.add_argument("--aux_auc_weight", type=float, default=0.0,
                        help="Weight for the AUC multi-task auxiliary MSE loss. "
                             "0 disables (default). 0.2 adds a correlated second response metric "
                             "that regularises the shared drug-cell representation and improves "
                             "drug-blind generalisation.")
    # Output-naming tag — lets different model variants coexist in results/
    parser.add_argument("--run_tag", default="",
                        help="Optional suffix for checkpoint / results / preds filenames. "
                             "Example: --run_tag fp_nomask  writes  random_seed0_fold0_fp_nomask.*")
    args = parser.parse_args()
    main(args)
