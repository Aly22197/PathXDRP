"""
Experiment sweep runner for PathXDRP and all baselines.

Runs every (model, split, seed, fold) combination in `SWEEP_CONFIGS`,
skipping any that already have a results JSON file.

Usage
-----
# Full sweep (all models × all splits × seeds 0-4 × fold 0):
  python scripts/run_sweep.py

# PathXDRP only, random split:
  python scripts/run_sweep.py --models pathxdrp --splits random --seeds 0 1 2

# Dry-run (print commands without executing):
  python scripts/run_sweep.py --dry_run

# Use fold 1 for all runs:
  python scripts/run_sweep.py --fold 1

Output
------
Results land in:
  results/pathxdrp/<split>_seed<s>_fold<f>.json
  results/graphdrp/<split>_seed<s>_fold<f>.json
  results/drpreter/<split>_seed<s>_fold<f>.json
  results/cdrscan/<split>_seed<s>_fold<f>.json

Run eval/analyze_results.py afterwards to aggregate into a table.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent

PYTHON = sys.executable

# Evaluation protocol: fold 0 of each seed for the main table (5 seeds, 5 splits).
# For the paper's ablation table we use all 5 folds of seed 0.
DEFAULT_MODELS = ["pathxdrp", "graphdrp", "drpreter", "cdrscan"]
DEFAULT_SPLITS = ["random", "cell_blind", "drug_blind", "scaffold_blind", "tissue_blind"]
DEFAULT_SEEDS  = [0, 1, 2, 3, 4]
DEFAULT_FOLD   = 0

# Unified training recipe (v3): batch=64 with sqrt-scaled LR.
#
# v2 used batch=64 with the original lr=1e-3, which produced training instability
# on the random split: loss diverged after epoch 9 (overshoot) and test PCC fell
# from 0.9307 to 0.9036. Cause: per-step gradient noise scales as 1/batch_size,
# so dropping batch by 4x at fixed LR is equivalent to quadrupling the effective
# step size. Fix: square-root scaling rule, lr = 1e-3 * sqrt(64/256) = 5e-4. Also
# extend the LR warmup from 5 to 10 epochs to give the model more time to settle
# in a stable region before reaching peak LR.
EPOCHS = {
    "pathxdrp": 50,
    "graphdrp": 50,
    "drpreter": 50,
    "cdrscan":  60,
}

EARLY_STOP_PATIENCE = 10
BATCH_SIZE          = 64
LR                  = 5e-4   # sqrt-scaled from 1e-3 at batch=256

# Common flags shared by all training scripts (train.py and train_baseline.py)
_COMMON = [
    "--batch_size",          str(BATCH_SIZE),
    "--lr",                  str(LR),
    "--early_stop_patience", str(EARLY_STOP_PATIENCE),
]

# Per-model extra CLI flags
EXTRA_FLAGS: dict[str, list[str]] = {
    "pathxdrp": ["--mask_type", "soft"] + _COMMON,
    "graphdrp": ["--n_gin_layers", "5"] + _COMMON,
    "drpreter": []                       + _COMMON,
    "cdrscan":  ["--hidden_dim", "256"] + _COMMON,
}


def results_path(model: str, split: str, seed: int, fold: int) -> Path:
    return ROOT / "results" / model / f"{split}_seed{seed}_fold{fold}.json"


def build_command(model: str, split: str, seed: int, fold: int, epochs: int) -> list[str]:
    # -u: unbuffered stdout so progress is visible in subprocess output
    if model == "pathxdrp":
        cmd = [
            PYTHON, "-u", "-m", "pathxdrp.train",
            "--split",  split,
            "--seed",   str(seed),
            "--fold",   str(fold),
            "--epochs", str(epochs),
        ]
    else:
        cmd = [
            PYTHON, "-u", str(ROOT / "scripts" / "train_baseline.py"),
            "--model", model,
            "--split",  split,
            "--seed",   str(seed),
            "--fold",   str(fold),
            "--epochs", str(epochs),
        ]

    cmd += EXTRA_FLAGS.get(model, [])
    return cmd


def run_sweep(
    models:   list[str],
    splits:   list[str],
    seeds:    list[int],
    fold:     int,
    dry_run:  bool,
) -> None:
    import time
    from datetime import datetime, timedelta

    jobs = [
        (model, split, seed, fold)
        for model in models
        for split in splits
        for seed  in seeds
    ]
    total  = len(jobs)
    done   = 0
    skipped = 0
    failed  = 0
    job_times: list[float] = []
    sweep_start = time.time()

    print(f"\nSweep: {total} jobs  ({len(models)} models × {len(splits)} splits × {len(seeds)} seeds)",
          flush=True)

    def _fmt(s):
        if s < 0: return "?"
        return str(timedelta(seconds=int(s)))

    for i, (model, split, seed, fold_) in enumerate(jobs, 1):
        rpath = results_path(model, split, seed, fold_)
        eta_str = _fmt((sum(job_times) / max(len(job_times), 1)) * (total - i + 1)) \
                  if job_times else "?"
        prefix = f"[{i:3d}/{total}] [{datetime.now().strftime('%H:%M:%S')}] " \
                 f"sweep_ETA={eta_str}"

        if rpath.exists():
            skipped += 1
            print(f"{prefix} SKIP {model}/{split}/seed{seed}/fold{fold_} (already done)",
                  flush=True)
            continue

        cmd = build_command(model, split, seed, fold_, EPOCHS[model])
        print(f"\n{prefix} START {model}/{split}/seed{seed}/fold{fold_}", flush=True)
        print("  cmd: " + " ".join(cmd), flush=True)

        if dry_run:
            continue

        rpath.parent.mkdir(parents=True, exist_ok=True)
        log_dir = ROOT / "logs"
        log_dir.mkdir(exist_ok=True)
        log_path = log_dir / f"{model}_{split}_seed{seed}_fold{fold_}.log"

        job_start = time.time()
        with open(log_path, "w", buffering=1, encoding="utf-8", errors="replace") as log_f:
            proc = subprocess.Popen(
                cmd, cwd=str(ROOT),
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                bufsize=1, encoding="utf-8", errors="replace",
            )
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log_f.write(line)
            proc.wait()
        result_rc = proc.returncode
        job_time = time.time() - job_start
        job_times.append(job_time)

        if result_rc != 0:
            failed += 1
            print(f"  [FAILED] exit={result_rc} after {_fmt(job_time)} | log: {log_path}",
                  flush=True)
        else:
            done += 1
            print(f"  [OK in {_fmt(job_time)}] -> {rpath} | log: {log_path}", flush=True)

    total_time = time.time() - sweep_start
    print(f"\nSweep done: {done} ran, {skipped} skipped, {failed} failed "
          f"in {_fmt(total_time)}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--models",   nargs="+", default=DEFAULT_MODELS,
                   choices=DEFAULT_MODELS + ["all"])
    p.add_argument("--splits",   nargs="+", default=DEFAULT_SPLITS)
    p.add_argument("--seeds",    nargs="+", type=int, default=DEFAULT_SEEDS)
    p.add_argument("--fold",     type=int, default=DEFAULT_FOLD)
    p.add_argument("--dry_run",  action="store_true",
                   help="Print commands without executing")
    args = p.parse_args()

    models = DEFAULT_MODELS if "all" in args.models else args.models
    run_sweep(models, args.splits, args.seeds, args.fold, args.dry_run)


if __name__ == "__main__":
    main()
