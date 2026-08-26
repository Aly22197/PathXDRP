"""
W3 -- Re-run the benchmark with fold-wise expression standardisation.

Answers Reviewer #3, point 3.

Every run is tagged `_fw` so it lands beside, and never overwrites, the results
that back the submitted manuscript. Compare with
revision/scripts/compare_foldwise.py.

Ordering rationale (see outputs/leakage_diagnostic.md): only the cell-blind and
tissue-blind splits hold out cell lines, so only they have a channel through
which cohort-wide moments can leak. Those two splits therefore run first for all
four models. The remaining three splits are queued afterwards as confirmation
runs -- the diagnostic predicts they are unchanged and the runs verify it.

Usage:
    python revision/scripts/run_foldwise_sweep.py            # everything
    python revision/scripts/run_foldwise_sweep.py --phase 1  # leakage-affected only
    python revision/scripts/run_foldwise_sweep.py --dry-run
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
OUT.mkdir(parents=True, exist_ok=True)
LOGDIR = OUT / "sweep_logs"
LOGDIR.mkdir(parents=True, exist_ok=True)

TAG = "fw"
SEEDS = [0, 1, 2, 3, 4]
MODELS = ["pathxdrp", "graphdrp", "drpreter", "cdrscan"]

# Phase 1: the two splits that hold out cell lines -> the only leakage channel.
PHASE1_SPLITS = ["cell_blind", "tissue_blind"]
# Phase 2: confirmation that the unaffected splits really are unchanged.
PHASE2_SPLITS = ["random", "drug_blind", "scaffold_blind"]

# Phase 3: leave-one-tissue-out. tissue_blind_split already emits one fold per
# tissue over the five most-represented tissues; the submitted sweep only ever
# ran fold 0 (lung_NSCLC_adenocarcinoma). Folds 1-4 hold out breast,
# large_intestine, lung_small_cell_carcinoma and ovary respectively. Running
# them turns the reported initialisation variance into genuine across-tissue
# variance, which is what Reviewer #5 point 6 asked for.
LOTO_FOLDS = [1, 2, 3, 4]
LOTO_SEEDS = [0]


def result_path(model: str, split: str, seed: int, fold: int = 0) -> Path:
    return ROOT / "results" / model / f"{split}_seed{seed}_fold{fold}_{TAG}.json"


# Both entry points DEFAULT to epochs=150 and early_stop_patience=0 (no early
# stopping). The published sweep used 50 epochs and patience 10, recorded in the
# `args` block of every results JSON, so these must be passed explicitly or the
# re-run silently uses a different protocol.
PROTOCOL = ["--epochs", "50", "--early_stop_patience", "10",
            "--lr", "5e-4", "--batch_size", "64"]

# Everything else is copied verbatim from scripts/run_final_sweep.ps1, the
# script that produced the submitted results. `--norm` is the ONLY intended
# difference between this sweep and the published one; anything else that
# differed would confound the leakage measurement. In particular
# `--use_morgan_fp` is retained even though the fingerprint only feeds h_mol,
# which `--drop_h_mol` removes from the head: it is dead compute, but dropping
# it here would also perturb the weight-initialisation RNG stream and muddy the
# comparison. It is disabled by default in the released code instead.
PATHXDRP_CFG = [
    "--hidden_dim", "256", "--n_gat_layers", "4", "--n_attn_heads", "8",
    "--dropout", "0.1", "--mask_type", "none", "--n_pw_transformer_layers", "2",
    "--use_morgan_fp", "--aux_auc_weight", "0.2",
    "--cross_attn_residual", "--drop_h_mol", "--attn_aux_weight", "0.3",
    "--evidential_lam", "0.01", "--lam_warmup_epochs", "20",
    "--precision", "bf16",
]


def build_cmd(model: str, split: str, seed: int, fold: int = 0) -> list[str]:
    if model == "pathxdrp":
        return [
            sys.executable, "-m", "pathxdrp.train",
            "--split", split, "--seed", str(seed), "--fold", str(fold),
            "--norm", "foldwise", "--run_tag", TAG, *PROTOCOL, *PATHXDRP_CFG,
        ]
    return [
        sys.executable, "scripts/train_baseline.py",
        "--model", model,
        "--split", split, "--seed", str(seed), "--fold", str(fold),
        "--norm", "foldwise", "--run_tag", TAG, *PROTOCOL,
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", type=int, default=0, choices=[0, 1, 2, 3],
                    help="0 = phases 1+2, 1 = leakage-affected splits, "
                         "2 = confirmation splits, 3 = leave-one-tissue-out "
                         "(tissue_blind folds 1-4)")
    ap.add_argument("--models", nargs="*", default=MODELS)
    ap.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="re-run even if the result JSON already exists")
    args = ap.parse_args()

    if args.phase == 3:
        jobs = [(m, "tissue_blind", sd, f)
                for f in LOTO_FOLDS for m in args.models for sd in LOTO_SEEDS]
    else:
        splits = []
        if args.phase in (0, 1):
            splits += PHASE1_SPLITS
        if args.phase in (0, 2):
            splits += PHASE2_SPLITS
        jobs = [(m, sp, sd, 0)
                for sp in splits for m in args.models for sd in args.seeds]
    todo = [j for j in jobs if args.force or not result_path(*j).exists()]

    print(f"{len(jobs)} jobs, {len(jobs)-len(todo)} already done, {len(todo)} to run")
    if args.dry_run:
        for m, sp, sd, fd in todo:
            print("  ", " ".join(build_cmd(m, sp, sd, fd)))
        return

    manifest = OUT / "foldwise_sweep_manifest.jsonl"
    t_start = time.time()
    for i, (model, split, seed, fold) in enumerate(todo, 1):
        log = LOGDIR / f"{model}_{split}_seed{seed}_fold{fold}_{TAG}.log"
        cmd = build_cmd(model, split, seed, fold)
        print(f"\n[{i}/{len(todo)}] {model} {split} seed{seed} fold{fold}",
              flush=True)
        t0 = time.time()
        with open(log, "w", encoding="utf-8") as fh:
            rc = subprocess.call(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT)
        dt = time.time() - t0
        ok = rc == 0 and result_path(model, split, seed, fold).exists()
        print(f"    rc={rc} in {dt/60:.1f} min  -> {'OK' if ok else 'FAILED'}", flush=True)
        with open(manifest, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({
                "model": model, "split": split, "seed": seed, "fold": fold,
                "tag": TAG,
                "rc": rc, "ok": ok, "seconds": dt, "log": str(log.name),
                "cmd": " ".join(cmd),
            }) + "\n")
        elapsed = time.time() - t_start
        rate = elapsed / i
        print(f"    elapsed {elapsed/3600:.2f} h | ETA "
              f"{(len(todo)-i)*rate/3600:.2f} h", flush=True)

    print(f"\nSweep finished in {(time.time()-t_start)/3600:.2f} h")


if __name__ == "__main__":
    main()
