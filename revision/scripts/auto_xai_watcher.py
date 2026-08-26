"""
Score each ablation variant's faithfulness as soon as its checkpoint appears.

The training queue and the XAI scoring are separate steps, and XAI is cheap
(~12 min) next to training (~2 h). Waiting for all seven variants to finish
training before scoring any of them wastes the gap. This watcher polls for new
checkpoints and scores them one at a time, so the faithfulness column fills in
as training proceeds.

Runs one scoring job at a time so it never contends with itself; it does share
the GPU with the training queue, which has headroom (~2.5 GB of 6 GB in use).

Usage:
    python revision/scripts/auto_xai_watcher.py
Stops when every variant in ABLATION has been scored, or after --max-hours.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parents[1] / "outputs"
LOGDIR = OUT / "queue_logs"
LOGDIR.mkdir(parents=True, exist_ok=True)

TAGS = ["abA", "abAp", "abB", "abC", "abD", "abE", "abF"]


def ckpt(tag: str) -> Path:
    return ROOT / "checkpoints" / "pathxdrp" / f"random_seed0_fold0_{tag}.pt"


def result(tag: str) -> Path:
    return ROOT / "results" / "pathxdrp" / f"random_seed0_fold0_{tag}.json"


def scored(tag: str) -> Path:
    return ROOT / "results" / "xai" / f"xai_multimodel_pathxdrp_{tag}.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-hours", type=float, default=24.0)
    ap.add_argument("--poll", type=int, default=120)
    args = ap.parse_args()

    t0 = time.time()
    while time.time() - t0 < args.max_hours * 3600:
        pending = [t for t in TAGS if not scored(t).exists()]
        if not pending:
            print("all variants scored", flush=True)
            return

        # A checkpoint is only safe to score once the run's results JSON exists;
        # that file is written last, so its presence means training finished.
        ready = [t for t in pending if ckpt(t).exists() and result(t).exists()]
        if not ready:
            time.sleep(args.poll)
            continue

        tag = ready[0]
        print(f"[{time.strftime('%H:%M:%S')}] scoring {tag}", flush=True)
        log = LOGDIR / f"xai_{tag}_auto.log"
        with open(log, "w", encoding="utf-8") as fh:
            rc = subprocess.call(
                [sys.executable, "scripts/run_xai_multimodel.py",
                 "--models", "pathxdrp", "--run_tag", tag],
                cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT)
        ok = rc == 0 and scored(tag).exists()
        print(f"    {tag}: rc={rc} -> {'OK' if ok else 'FAILED'}", flush=True)
        if not ok:
            # Do not spin on a variant that cannot be scored; leave a marker so
            # the loop moves on and a human can see what happened.
            scored(tag).parent.mkdir(parents=True, exist_ok=True)
            (LOGDIR / f"xai_{tag}_FAILED").write_text(f"rc={rc}\n")
            TAGS.remove(tag)

    print("max-hours reached", flush=True)


if __name__ == "__main__":
    main()
