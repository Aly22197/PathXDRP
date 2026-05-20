"""
Inference benchmark — Phase 4.4.

For each (drug_encoder_type × cell_encoder_type) combination, measures:
  - parameter count
  - throughput on the test split (samples/sec, batch_size=256)
  - peak GPU memory (MB)
  - single-sample latency (ms, batch_size=1) — deployment scenario
  - CPU-only single-sample latency (ms) — edge-device scenario

Outputs a JSON to results/benchmarks/inference_benchmark.json and a markdown
table to the same directory.

Usage:
  python scripts/benchmark_inference.py
  python scripts/benchmark_inference.py --combos gat,pathway_set gat,gene_mamba
  python scripts/benchmark_inference.py --skip-cpu        # skip CPU latency
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results" / "benchmarks"


# ---
# Combos to benchmark
# ---

DEFAULT_COMBOS = [
    ("gat",          "pathway_set"),  # baseline PathXDRP
    ("molformer",    "pathway_set"),  # +MolFormer drug encoder
    ("graph_mamba",  "pathway_set"),  # +Graph-Mamba drug encoder
    ("gat",          "gene_mamba"),   # +GeneMamba cell encoder
    ("gat",          "scgpt"),        # +scGPT cell encoder
    ("graph_mamba",  "gene_mamba"),   # full Mamba PathXDRP
]


# ---
# Build the model + a small test loader (uses one fold of the random split)
# ---

def _build_eval_loader(args):
    from pathxdrp.data.loader import build_master_df
    from pathxdrp.data.splits import load_split
    from pathxdrp.train import GDSCDataset, build_graph_cache, collate_fn

    print("Loading data", flush=True)
    df, expr_matrix = build_master_df(version="GDSC2", require_smiles=True)
    _, _, test_idx = load_split("random", seed=0, fold=0)
    test_idx = test_idx[: args.n_samples]   # subsample for speed

    drugs_df = df[["DRUG_ID", "SMILES"]].drop_duplicates()
    graph_cache, _ = build_graph_cache(drugs_df)

    test_ds = GDSCDataset(df.iloc[test_idx], graph_cache, expr_matrix)
    loader  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                         collate_fn=collate_fn, num_workers=0)
    return df, expr_matrix, graph_cache, loader


def _build_model(drug_type: str, cell_type: str, df, expr_matrix, graph_cache,
                 hidden_dim: int = 128) -> torch.nn.Module:
    import json as _json
    from pathxdrp.models.pathxdrp import PathXDRP

    # Pathway gene map (always needed)
    pgm_path = ROOT / "data" / "processed" / "pathway_gene_map.json"
    with open(pgm_path) as f:
        pathway_gene_symbols = _json.load(f)

    gene_list   = list(expr_matrix.columns)
    gene_to_idx = {g: i for i, g in enumerate(gene_list)}
    pathway_gene_map = {
        pw: [gene_to_idx[g] for g in genes if g in gene_to_idx]
        for pw, genes in pathway_gene_symbols.items()
        if any(g in gene_to_idx for g in genes)
    }

    sample_g = next(iter(graph_cache.values()))
    node_dim = sample_g.x.size(1)
    edge_dim = sample_g.edge_attr.size(1)

    return PathXDRP(
        node_in_dim=node_dim,
        edge_in_dim=edge_dim,
        n_genes=expr_matrix.shape[1],
        pathway_gene_map=pathway_gene_map,
        hidden_dim=hidden_dim,
        n_gat_layers=3,
        n_attn_heads=8,
        mask_type="soft",
        drug_encoder_type=drug_type,
        cell_encoder_type=cell_type,
        gene_symbols=gene_list if cell_type in ("gene_mamba", "scgpt") else None,
        graph_mamba_kwargs={"n_gat_layers": 2, "n_mamba_layers": 2, "ordering": "degree"}
            if drug_type == "graph_mamba" else None,
        gene_mamba_kwargs={"top_k": 2048, "freeze_backbone": True}
            if cell_type == "gene_mamba" else None,
    )


# ---
# Benchmark routines
# ---

@torch.no_grad()
def measure_throughput(model, loader, device, n_warmup: int = 3, label: str = "") -> dict:
    """Returns samples/sec and peak GPU mem on the device."""
    model.eval().to(device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    # Warm-up
    it = iter(loader)
    for _ in range(min(n_warmup, len(loader))):
        batch = next(it)
        model(drug_batch=batch["drug_batch"].to(device), expr=batch["expr"].to(device))
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Time the full pass
    n_total = 0
    t0 = time.time()
    pbar = tqdm(loader, desc=f"  throughput {label}", unit="batch", leave=False)
    for batch in pbar:
        out = model(
            drug_batch=batch["drug_batch"].to(device),
            expr=batch["expr"].to(device),
        )
        n_total += int(out["pred"]["pred"].numel())
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - t0
    pbar.close()

    return {
        "throughput_sps": n_total / max(elapsed, 1e-6),
        "n_samples":      n_total,
        "elapsed_sec":    elapsed,
        "peak_mem_mb":    (torch.cuda.max_memory_allocated(device) / 1024 ** 2)
                          if device.type == "cuda" else 0.0,
    }


@torch.no_grad()
def measure_latency(model, sample_batch, device, n_iters: int = 50,
                    n_warmup: int = 5) -> dict:
    """Returns median, p95, mean latency in ms for batch_size=1."""
    from torch_geometric.data import Batch
    model.eval().to(device)

    # Build a single-sample batch
    g0 = sample_batch["drug_batch"].get_example(0)
    bs1 = Batch.from_data_list([g0]).to(device)
    expr1 = sample_batch["expr"][:1].to(device)

    # Warm-up
    for _ in range(n_warmup):
        model(drug_batch=bs1, expr=expr1)
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Time loop
    times = []
    for _ in range(n_iters):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        model(drug_batch=bs1, expr=expr1)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)

    arr = np.array(times)
    return {
        "latency_median_ms": float(np.median(arr)),
        "latency_p95_ms":    float(np.percentile(arr, 95)),
        "latency_mean_ms":   float(arr.mean()),
        "latency_std_ms":    float(arr.std()),
        "n_iters":           n_iters,
    }


def count_params(model: torch.nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"params_total": int(total), "params_trainable": int(trainable)}


# ---
# Main
# ---

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--combos", nargs="+", default=None,
                   help="(drug,cell) pairs e.g. gat,pathway_set graph_mamba,gene_mamba")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--n_samples",  type=int, default=4000,
                   help="Subset of test split for throughput timing (full = ~15k).")
    p.add_argument("--latency_iters", type=int, default=50)
    p.add_argument("--skip_cpu",   action="store_true")
    p.add_argument("--out_dir",    default=str(RESULTS_DIR))
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.combos:
        combos = [tuple(c.split(",")) for c in args.combos]
    else:
        combos = DEFAULT_COMBOS

    df, expr_matrix, graph_cache, loader = _build_eval_loader(args)
    sample_batch = next(iter(loader))

    has_cuda = torch.cuda.is_available()
    cuda_dev = torch.device("cuda" if has_cuda else "cpu")
    cpu_dev  = torch.device("cpu")
    print(f"\nGPU available: {has_cuda} | combos to benchmark: {len(combos)}", flush=True)

    all_results: list[dict] = []
    pbar = tqdm(combos, desc="Benchmarking combos", unit="combo")
    for drug_type, cell_type in pbar:
        pbar.set_postfix(combo=f"{drug_type}+{cell_type}")
        try:
            model = _build_model(drug_type, cell_type, df, expr_matrix, graph_cache)
        except (ImportError, ValueError, NotImplementedError) as e:
            tqdm.write(f"  [skip] {drug_type}+{cell_type}: {e}")
            all_results.append({
                "drug_encoder": drug_type, "cell_encoder": cell_type,
                "skipped": True, "reason": str(e),
            })
            continue

        row = {"drug_encoder": drug_type, "cell_encoder": cell_type}
        row.update(count_params(model))

        if has_cuda:
            row.update(measure_throughput(model, loader, cuda_dev,
                                          label=f"{drug_type}+{cell_type} (GPU)"))
            row.update({f"gpu_{k}": v for k, v in
                        measure_latency(model, sample_batch, cuda_dev,
                                        n_iters=args.latency_iters).items()})

        if not args.skip_cpu:
            try:
                row.update({f"cpu_{k}": v for k, v in
                            measure_latency(model, sample_batch, cpu_dev,
                                            n_iters=min(args.latency_iters, 20)).items()})
            except Exception as e:
                row["cpu_latency_error"] = str(e)

        all_results.append(row)
        # Free memory between combos
        del model
        gc.collect()
        if has_cuda:
            torch.cuda.empty_cache()

        # Streaming print
        tqdm.write(
            f"  {drug_type:<12s} + {cell_type:<11s} | "
            f"params={row.get('params_trainable', 0):>9,d} | "
            f"GPU sps={row.get('throughput_sps', 0):>7.0f} | "
            f"GPU mem={row.get('peak_mem_mb', 0):>5.0f}MB | "
            f"GPU p95={row.get('gpu_latency_p95_ms', 0):>5.1f}ms | "
            f"CPU med={row.get('cpu_latency_median_ms', 0):>6.1f}ms"
        )
    pbar.close()

    # Save JSON + markdown
    out_json = out_dir / "inference_benchmark.json"
    out_md   = out_dir / "inference_benchmark.md"
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    with open(out_md, "w") as f:
        f.write("| Drug encoder | Cell encoder | Params | GPU sps | GPU MB | GPU p95 ms | CPU med ms |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|\n")
        for r in all_results:
            if r.get("skipped"):
                f.write(f"| {r['drug_encoder']} | {r['cell_encoder']} | "
                        f"_skipped: {r.get('reason','')}_ |  |  |  |  |\n")
                continue
            f.write(f"| {r['drug_encoder']} | {r['cell_encoder']} | "
                    f"{r.get('params_trainable',0):,} | "
                    f"{r.get('throughput_sps',0):.0f} | "
                    f"{r.get('peak_mem_mb',0):.0f} | "
                    f"{r.get('gpu_latency_p95_ms',0):.1f} | "
                    f"{r.get('cpu_latency_median_ms',0):.1f} |\n")

    print(f"\nBenchmark JSON: {out_json}", flush=True)
    print(f"Benchmark MD:   {out_md}",   flush=True)


if __name__ == "__main__":
    main()
