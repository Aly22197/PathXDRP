"""
Master GPU queue, ordered by scientific value rather than by convenience.

The earlier sweep was killed part-way through, so this replaces it with a single
resumable queue that survives interruption: every stage checks for its own output
before running, and the queue can be restarted at any time with no loss.

Ordering rationale
------------------
Stage 1 (ablation, seed 0) is first because it is the only stage answering a
claim the revised manuscript already makes but cannot support. Section 5.5 now
describes a seven-variant, one-factor-at-a-time ablation, and variant A' is the
control that decides whether Reviewer #5 is right that mean pooling alone
explains the faithfulness gain. Without it that section is a promise.

Stage 2 (XAI scoring of those variants) follows immediately, because it is what
converts stage 1 into the faithfulness columns the reviewers asked for. In the
first version of this queue it sat behind the fold-wise sweep; that was wrong,
and it mattered once random-split runs turned out to cost ~2 h each rather than
the ~40 min seen on cell-blind.

Stage 3 (fold-wise, seed 0) answers Reviewer #3 point 3 empirically. A
defensible answer already exists without it -- the leakage diagnostic proves
only cell-blind and tissue-blind have a channel and quantifies the distortion --
so this adds a paired check rather than creating the answer.

Stage 4 (leave-one-tissue-out) answers Reviewer #5 point 6 and is cheap: the
folds already exist and only fold 0 was ever run.

Stage 5 places DeepCDR in the comparison table.

Scope was cut from 85 jobs to 35 after measuring the real per-run cost. The cuts
are seeds and baselines on questions whose direction is already settled, never a
whole reviewer point. See outputs/queue_scope.md for what was dropped and why.

Usage:
    python revision/scripts/run_queue.py --list
    python revision/scripts/run_queue.py            # run all
    python revision/scripts/run_queue.py --stages 1 3
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = Path(__file__).resolve().parents[1] / "outputs"
LOGDIR = OUT / "queue_logs"
OUT.mkdir(parents=True, exist_ok=True)
LOGDIR.mkdir(parents=True, exist_ok=True)

PY = sys.executable
TAG = "fw"

PROTOCOL = ["--epochs", "50", "--early_stop_patience", "10",
            "--lr", "5e-4", "--batch_size", "64"]
PATHXDRP_CFG = [
    "--hidden_dim", "256", "--n_gat_layers", "4", "--n_attn_heads", "8",
    "--dropout", "0.1", "--mask_type", "none", "--n_pw_transformer_layers", "2",
    "--use_morgan_fp", "--aux_auc_weight", "0.2",
    "--cross_attn_residual", "--drop_h_mol", "--attn_aux_weight", "0.3",
    "--evidential_lam", "0.01", "--lam_warmup_epochs", "20",
    "--precision", "bf16",
]

# (tag, residual, drop_h_mol, aux_weight, pool_mode)
ABLATION = [
    ("abA",  False, False, 0.0, "attention"),
    ("abAp", False, False, 0.0, "mean"),
    ("abF",  True,  True,  0.3, "mean"),
    ("abB",  True,  False, 0.0, "mean"),
    ("abC",  False, True,  0.0, "mean"),
    ("abD",  False, False, 0.3, "mean"),
    ("abE",  True,  True,  0.0, "mean"),
]
MODELS = ["pathxdrp", "graphdrp", "drpreter", "cdrscan"]


def res_path(model: str, split: str, seed: int, fold: int, tag: str) -> Path:
    suf = f"_{tag}" if tag else ""
    return ROOT / "results" / model / f"{split}_seed{seed}_fold{fold}{suf}.json"


def train_cmd(model, split, seed, fold, tag, extra=()):
    if model == "pathxdrp":
        return [PY, "-m", "pathxdrp.train", "--split", split, "--seed", str(seed),
                "--fold", str(fold), "--norm", "foldwise", "--run_tag", tag,
                *PROTOCOL, *PATHXDRP_CFG, *extra]
    return [PY, "scripts/train_baseline.py", "--model", model, "--split", split,
            "--seed", str(seed), "--fold", str(fold), "--norm", "foldwise",
            "--run_tag", tag, *PROTOCOL, *extra]


def build_jobs() -> list[dict]:
    jobs: list[dict] = []

    # ---- stage 1: ablation, seed 0 (A, A' and F first: they carry the answer)
    for tag, res, drop, aux, pool in ABLATION:
        extra = ["--pool_mode", pool, "--attn_aux_weight", str(aux)]
        if res:
            extra.append("--cross_attn_residual")
        if drop:
            extra.append("--drop_h_mol")
        # attn_aux_weight appears in PATHXDRP_CFG too; the later flag wins
        cmd = [PY, "-m", "pathxdrp.train", "--split", "random", "--seed", "0",
               "--fold", "0", "--norm", "foldwise", "--run_tag", tag,
               *PROTOCOL,
               "--hidden_dim", "256", "--n_gat_layers", "4", "--n_attn_heads", "8",
               "--dropout", "0.1", "--mask_type", "none",
               "--n_pw_transformer_layers", "2", "--use_morgan_fp",
               "--aux_auc_weight", "0.2", "--evidential_lam", "0.01",
               "--lam_warmup_epochs", "20", "--precision", "bf16", *extra]
        jobs.append({"stage": 1, "name": f"ablation/{tag}/seed0", "cmd": cmd,
                     "out": res_path("pathxdrp", "random", 0, 0, tag)})

    # ---- stage 2: XAI scoring of the ablation variants.
    # This must follow stage 1 IMMEDIATELY, not after the fold-wise sweep: it is
    # what converts those seven training runs into the faithfulness columns that
    # answer Reviewers #3.5, #4.4 and #5.3. Training runs on the random split
    # turned out to cost ~2 h each rather than the ~40 min seen on cell-blind,
    # so ordering matters far more than originally assumed.
    for tag, *_ in ABLATION:
        jobs.append({
            "stage": 2, "name": f"xai/{tag}",
            "cmd": [PY, "scripts/run_xai_multimodel.py", "--models", "pathxdrp",
                    "--run_tag", tag],
            "out": ROOT / "results" / "xai" / f"xai_multimodel_pathxdrp_{tag}.json"})

    # ---- stage 3: fold-wise, leakage-affected splits, SEED 0 ONLY.
    # Scope cut from 3 seeds to 1. A defensible answer to Reviewer #3.3 already
    # exists without any of these runs (the leakage diagnostic proves only these
    # two splits have a channel, and quantifies the input distortion at 0.02
    # sigma); these add the paired empirical check across all four models. The
    # extra seeds were costing ~16 h to narrow an error bar on a result that is
    # already directionally clear.
    for split in ["cell_blind", "tissue_blind"]:
        for model in MODELS:
            jobs.append({
                "stage": 3, "name": f"fw/{split}/{model}/seed0",
                "cmd": train_cmd(model, split, 0, 0, TAG),
                "out": res_path(model, split, 0, 0, TAG)})

    # ---- stage 4: leave-one-tissue-out, folds 1-4.
    # PathXDRP plus CDRScan only. CDRScan is the strongest baseline on
    # tissue-blind, so this still gives a two-model comparison across tissues,
    # which is what Reviewer #5.6 asked for; adding the other two would double
    # the cost to sharpen a comparison that is not in dispute.
    for fold in [1, 2, 3, 4]:
        for model in ["pathxdrp", "cdrscan"]:
            jobs.append({
                "stage": 4, "name": f"loto/fold{fold}/{model}",
                "cmd": train_cmd(model, "tissue_blind", 0, fold, TAG),
                "out": res_path(model, "tissue_blind", 0, fold, TAG)})

    # ---- stage 5: DeepCDR, seed 0 across the five splits.
    # Enough to place it in the comparison table; more seeds only if time allows.
    for split in ["random", "cell_blind", "drug_blind", "scaffold_blind",
                  "tissue_blind"]:
        jobs.append({
            "stage": 5, "name": f"deepcdr/{split}/seed0",
            "cmd": train_cmd("deepcdr", split, 0, 0, TAG),
            "out": res_path("deepcdr", split, 0, 0, TAG)})

    return jobs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", nargs="*", type=int,
                    default=[1, 2, 3, 4, 5])
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    jobs = [j for j in build_jobs() if j["stage"] in args.stages]
    todo = [j for j in jobs if args.force or not j["out"].exists()]

    print(f"{len(jobs)} jobs in stages {args.stages}; "
          f"{len(jobs)-len(todo)} already done; {len(todo)} to run")
    if args.list:
        for j in todo:
            print(f"  [{j['stage']}] {j['name']}")
        return

    manifest = OUT / "queue_manifest.jsonl"
    t0 = time.time()
    for i, j in enumerate(todo, 1):
        log = LOGDIR / (j["name"].replace("/", "_") + ".log")
        print(f"\n[{i}/{len(todo)}] stage {j['stage']} :: {j['name']}", flush=True)
        t = time.time()
        with open(log, "w", encoding="utf-8") as fh:
            rc = subprocess.call(j["cmd"], cwd=ROOT, stdout=fh,
                                 stderr=subprocess.STDOUT)
        dt = time.time() - t
        ok = rc == 0 and j["out"].exists()
        print(f"    rc={rc} in {dt/60:.1f} min -> {'OK' if ok else 'FAILED'}",
              flush=True)
        with open(manifest, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"stage": j["stage"], "name": j["name"],
                                 "rc": rc, "ok": ok, "seconds": dt}) + "\n")
        el = time.time() - t0
        print(f"    elapsed {el/3600:.2f} h | ETA {(len(todo)-i)*(el/i)/3600:.2f} h",
              flush=True)

    print(f"\nQueue finished in {(time.time()-t0)/3600:.2f} h")


if __name__ == "__main__":
    main()
