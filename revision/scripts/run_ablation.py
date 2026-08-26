"""
W6 -- Clean, one-factor-at-a-time ablation with faithfulness metrics.

Answers Reviewer #3 point 5, Reviewer #4 point 4 and Reviewer #5 point 3.

The submitted Table 11 varied three flags but the atom-pooling change rode along
with the residual, because `pathxdrp/models/pathxdrp.py` chose the pooling mode
from `cross_attn_residual`. Reviewer #5 correctly pointed out that mean pooling
alone might relieve attention saturation, which would mean the faithfulness gain
was not attributable to the architectural correction at all.

The model now takes an explicit `pool_mode` (auto|mean|attention) that is
independent of the residual, so the confound can be broken. Variant A' below is
the decisive control: the old architecture with ONLY the pooling changed.

  Variant                         Res+LN  drop h_mol  Aux   Pool
  A   baseline (old head)           -         -        -    attention
  A'  pooling-only control          -         -        -    mean
  B   residual+LN only              x         -        -    mean
  C   drop h_mol only               -         x        -    mean
  D   attention-aux only            -         -        x    mean
  E   B + C                         x         x        -    mean
  F   full PathXDRP                 x         x        x    mean

Each variant reports prediction metrics AND faithfulness, which is what
Reviewer #3 point 5 asked for and what the submitted table omitted.

Usage:
    python revision/scripts/run_ablation.py --dry-run
    python revision/scripts/run_ablation.py --seeds 0 1 2
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parents[1] / "outputs"
LOGDIR = OUT / "ablation_logs"
OUT.mkdir(parents=True, exist_ok=True)
LOGDIR.mkdir(parents=True, exist_ok=True)

SPLIT = "random"

# (tag, residual, drop_h_mol, aux_weight, pool_mode)
VARIANTS = [
    ("abA",  False, False, 0.0, "attention"),
    ("abAp", False, False, 0.0, "mean"),        # A' -- the decisive control
    ("abB",  True,  False, 0.0, "mean"),
    ("abC",  False, True,  0.0, "mean"),
    ("abD",  False, False, 0.3, "mean"),
    ("abE",  True,  True,  0.0, "mean"),
    ("abF",  True,  True,  0.3, "mean"),        # full PathXDRP
]

BASE = ["--epochs", "50", "--early_stop_patience", "10",
        "--lr", "5e-4", "--batch_size", "64",
        "--hidden_dim", "256", "--n_gat_layers", "4", "--n_attn_heads", "8",
        "--dropout", "0.1", "--mask_type", "none",
        "--n_pw_transformer_layers", "2",
        "--use_morgan_fp", "--aux_auc_weight", "0.2",
        "--evidential_lam", "0.01", "--lam_warmup_epochs", "20",
        "--precision", "bf16", "--norm", "foldwise"]


def build_cmd(tag, res, drop, aux, pool, seed):
    cmd = [sys.executable, "-m", "pathxdrp.train",
           "--split", SPLIT, "--seed", str(seed), "--fold", "0",
           "--run_tag", tag, *BASE,
           "--pool_mode", pool,
           "--attn_aux_weight", str(aux)]
    if res:
        cmd.append("--cross_attn_residual")
    if drop:
        cmd.append("--drop_h_mol")
    return cmd


def result_path(tag, seed):
    return ROOT / "results" / "pathxdrp" / f"{SPLIT}_seed{seed}_fold0_{tag}.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    jobs = [(v, s) for v in VARIANTS for s in args.seeds]
    todo = [(v, s) for v, s in jobs
            if args.force or not result_path(v[0], s).exists()]
    print(f"{len(jobs)} jobs, {len(todo)} to run")

    if args.dry_run:
        for v, s in todo:
            print("  ", " ".join(build_cmd(*v, s)))
        return

    manifest = OUT / "ablation_manifest.jsonl"
    t0 = time.time()
    for i, (v, seed) in enumerate(todo, 1):
        tag = v[0]
        log = LOGDIR / f"{tag}_seed{seed}.log"
        print(f"\n[{i}/{len(todo)}] variant {tag} seed{seed}", flush=True)
        t = time.time()
        with open(log, "w", encoding="utf-8") as fh:
            rc = subprocess.call(build_cmd(*v, seed), cwd=ROOT,
                                 stdout=fh, stderr=subprocess.STDOUT)
        dt = time.time() - t
        ok = rc == 0 and result_path(tag, seed).exists()
        print(f"    rc={rc} in {dt/60:.1f} min -> {'OK' if ok else 'FAILED'}",
              flush=True)
        with open(manifest, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"tag": tag, "seed": seed, "rc": rc, "ok": ok,
                                 "seconds": dt}) + "\n")
        el = time.time() - t0
        print(f"    elapsed {el/3600:.2f} h | ETA "
              f"{(len(todo)-i)*(el/i)/3600:.2f} h", flush=True)

    print(f"\nAblation finished in {(time.time()-t0)/3600:.2f} h")


if __name__ == "__main__":
    main()
