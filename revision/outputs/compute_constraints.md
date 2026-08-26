# Compute constraints on this machine, and one job lost to them

Recorded because it changes how the remaining queue must be run, and because a
lost run should be documented rather than quietly re-run.

## What happened

Ablation variant C (drop_h_mol only) died at epoch 11 with:

```
numpy.core._exceptions._ArrayMemoryError:
Unable to allocate 4.69 MiB for an array with shape (64, 19193) and dtype float32
```

A 4.69 MiB allocation failing is host RAM exhaustion, not GPU memory. The queue
moved on to variant D, so C produced no results JSON and must be re-run.

## Cause

| Consumer | Working set | Private |
|---|---|---|
| `vmmemWSL` | 3.8 GB | **8.2 GB** |
| Memory Compression | 2.6 GB | — |
| everything else | ~2 GB | — |
| **free physical** | **0.19 GB of 15.85 GB** | |

WSL is holding roughly half the machine's RAM. Training PathXDRP needs ~2-3 GB
of host RAM for the expression matrix, the master dataframe and the graph and
fingerprint caches, and it was fitting into what was left with very little
headroom.

The trigger was mine: I had started an auto-XAI watcher that scores each
ablation variant as soon as its checkpoint appears, so a second process was
loading the same data at the same time. Variants A, A', B and F all completed
while XAI was run manually, one at a time, with training paused between. C is
the first run that overlapped with a fully concurrent second loader, and it is
the one that died.

**The watcher has been stopped.** Nothing else should run concurrently with
training on this machine.

## Consequences for the remaining plan

1. **No concurrent GPU/data jobs.** XAI scoring is cheap (~12 min against ~2 h
   of training) but it loads the full dataset, and there is no RAM headroom for
   a second loader. Score variants only while no training is running.
2. **Variant C must be re-run.** Nothing else is lost; the queue's resumability
   means it will be picked up automatically.
3. **The queue's own ordering already handles this correctly** -- training
   stages then a scoring stage -- so the fix is simply not to run scoring
   out-of-band, which is what the watcher was doing.

## What would help, if you want to intervene

`wsl --shutdown` would return roughly 8 GB and remove the pressure entirely.
I have not run it, because WSL may be hosting work of yours and reclaiming it is
not a decision I should make unattended. If WSL is idle, shutting it down is the
single highest-value action available for the remaining queue -- it would also
let scoring run concurrently again and recover several hours.

Without it the queue still completes; it just has no margin, and another
concurrent process would likely kill another run.


---

## Second interruption: the Windows session shut down (2026-08-25)

The queue stopped again after 9.9 hours with 9 of 35 jobs unrun. The return
codes identify the cause precisely:

| Job | Return code | Meaning |
|---|---|---|
| loto/fold3/pathxdrp | `0x40010004` | `DBG_TERMINATE_PROCESS` -- killed 43.7 min into the run |
| the following 8 jobs | `0xC000026B` | `STATUS_DLL_INIT_FAILED_LOGOFF` -- process could not start because the session was shutting down |

So the machine logged off, slept, or restarted. Every subsequent Python launch
failed instantly rather than running and erroring, which is the signature of a
session teardown rather than a fault in the code or the data.

Sleep-on-AC is disabled (`STANDBYIDLE` index `0x0`), so this was not an idle
timeout on mains power. Battery sleep, a user logoff, or a restart are the
remaining explanations, and none is visible from inside the process.

**Nothing was lost beyond the one interrupted run.** Every stage checks for its
own output before running, so resuming re-ran only the 9 outstanding jobs.

### Practical note

Two interruptions in two days, from unrelated causes (host RAM exhaustion, then
session teardown), on a 35-job queue. For a workload of this length the
resumability is doing real work, and it is worth keeping any future sweep
structured the same way -- one job per process, output checked before launch --
rather than as a single long-running script.
