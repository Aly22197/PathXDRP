# Reproducibility

This file maps every table and figure in the manuscript to the script that
generates it and the data file it reads. It exists because Reviewer #4 of the
Knowledge-Based Systems submission found that the README's headline numbers
disagreed with the manuscript's, and asked for "the exact commit hash,
configuration files, seeds, result CSV files and scripts corresponding to the
submitted version". That request is answered here in full.

---

## 1. The discrepancy Reviewer #4 found

There were three defects, in two rounds. The first two are what the reviewer
saw on the public repository; the third was introduced while fixing them and is
recorded here because it has the same cause — a number maintained by hand
instead of generated.

### Round 1: the stale aggregate

**What happened.** The README's headline table was generated from
`results/summary_v3_vs_baselines.csv`. That file is a stale aggregate: it was
written before the final PathXDRP sweep and never regenerated. It has two
defects.

1. **A failed PathXDRP sweep.** Sixteen of its twenty-five PathXDRP rows have
   `best_val_epoch <= 2` — those runs never trained. They were produced under a
   learning-rate schedule that was subsequently corrected; the same runs are
   archived under `results/archive/pathxdrp_v2_lr_too_high/`. Averaging them
   yields random PCC 0.919 and scaffold-blind PCC 0.441, which is exactly what
   the README reported.
2. **Double-counted baselines.** Four `(model, split, seed)` keys appear twice,
   so those per-split means average an old run together with its rerun.

**What was correct.** The manuscript reported the final sweep in
`results/<model>/<split>_seed<S>_fold0.json`. Those files reproduce the
manuscript's Table 4 exactly (random 0.930, cell-blind 0.864, drug-blind 0.513,
scaffold-blind 0.553, tissue-blind 0.848).

**Conclusion.** The manuscript was not reporting inflated numbers; the
repository was reporting stale ones. Full audit with the reproduction of both
number sets:
[`revision/outputs/readme_discrepancy_audit.md`](revision/outputs/readme_discrepancy_audit.md).

### Round 2: two ways to average the same runs

Regenerating the README exposed two further ways for the same result files to
produce different tables. Both are now closed in `revision/scripts/build_ledger.py`.

3. **Sample vs population standard deviation.** The ledger used
   `statistics.stdev` (ddof=1); `eval/analyze_results.py`, and therefore the
   manuscript, uses `numpy.std(..., ddof=0)`. Over five seeds the sample
   estimate is `sqrt(5/4)` = 11.8% larger, which moved nine of the twenty PCC
   cells in the third decimal — same means, different error bars. The ledger
   now uses the population standard deviation.
4. **Non-canonical runs swept into the headline means.** The revision wrote
   fold-wise-normalisation reruns (`_fw`), leave-one-tissue-out folds
   (`_fold1..4_fw`) and head ablations (`_abA`..`_abF`) into the same
   `results/<model>/` directories. The ledger globbed `*_fold*.json`, so it
   averaged twelve PathXDRP random runs instead of five and ten tissue-blind
   runs instead of five. It now matches `<split>_seed<S>_fold0.json` exactly
   and reports `n_seeds = 5` for every cell.

**Remedy.** `results/summary_v3_vs_baselines.csv` has been removed from
`results/` so that nothing there can be mistaken for a current number; it is
retained as `revision/outputs/stale_summary_v3_vs_baselines.csv` so the audit
above stays reproducible. The README table is generated from the canonical
ledger by `revision/scripts/readme_table.py`, and
`python revision/scripts/readme_table.py --check` exits non-zero if the README
has drifted from `results/`. Run it before tagging a release.

**Verification.** These two commands must print the same twenty values, and
both must match Table 4:

```bash
python eval/analyze_results.py --metrics PCC     # the repo's own aggregator
python revision/scripts/build_ledger.py          # the ledger
```

---

## 2. Canonical results ledger

One command rebuilds every number in the paper from the raw run outputs:

```bash
python revision/scripts/build_ledger.py
```

It writes:

| File | Contents |
|---|---|
| `revision/outputs/ledger.csv` | one row per (model, split, seed, fold) with every test metric |
| `revision/outputs/ledger_summary.csv` | mean ± std over seeds, per (model, split) |
| `revision/outputs/readme_discrepancy_audit.md` | the audit described above |

No number in the revised manuscript is typed by hand. Every table is emitted by
a script listed in section 4.

### Release commit

Reviewer #4 asked for "the exact commit hash, configuration files, seeds,
result CSV files and scripts corresponding to the submitted version". The
release checklist is:

1. `python revision/scripts/build_ledger.py`
2. `python revision/scripts/readme_table.py --check` — must exit 0
3. `python eval/analyze_results.py --metrics PCC` — must print the README table
4. commit, tag (suggested: `kbs-revision-1`), push
5. write the hash into the placeholder below and into the paper's
   data-availability statement

**Release commit:** `<fill in after tagging>`

Steps 2 and 3 are the guard the first submission did not have: they fail loudly
if the README, the ledger and the manuscript have drifted apart, which is the
failure this document exists to describe.

---

## 3. Environment and protocol

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 3060 Laptop, 6 GB VRAM |
| Python | 3.10.11 |
| PyTorch | 2.6.0+cu124 |
| PyTorch Geometric | 2.5 |
| OS | Windows 11 Pro 10.0.26200 |
| Seeds | 0, 1, 2, 3, 4 |
| Folds | 0 (one fold per seed) |

**Training protocol**, identical for all four models and recorded in the `args`
block of every results JSON:

```
--epochs 50  --early_stop_patience 10  --lr 5e-4  --batch_size 64
```

> **Caution for anyone re-running.** Both entry points default to
> `--epochs 150` and `--early_stop_patience 0` (no early stopping). The
> published protocol is *not* the default and must be passed explicitly.
> `revision/scripts/run_foldwise_sweep.py` does this.

**Data sources** (not redistributed; fetch from the providers):

| Source | Version | Where |
|---|---|---|
| GDSC2 dose-response | release 8.4 | <https://www.cancerrxgene.org/downloads/bulk_download> |
| DepMap RNA-seq | 24Q4 | <https://depmap.org/portal/download/all/> |
| KEGG pathway-gene map | via REST | <https://rest.kegg.jp/> |
| PubChem SMILES | PUG-REST | <https://pubchem.ncbi.nlm.nih.gov/> |
| PRISM Repurposing | secondary screen | DepMap portal |

Deterministic split index files live in `data/processed/splits/`. They are
~73 MB and are **not** committed; `scripts/build_splits.py` regenerates them
from the GDSC2 raw file, and the generator is seeded, so a rebuild reproduces
the exact row indices behind the published runs.

---

## 4. Table and figure provenance

| Manuscript item | Generated by | Reads |
|---|---|---|
| Table: prediction accuracy across splits | `revision/scripts/build_ledger.py` | `results/<model>/<split>_seed<S>_fold0.json` |
| README headline table | `revision/scripts/readme_table.py` | `revision/outputs/ledger_summary.csv` |
| Table: fold-wise normalisation correction | `revision/scripts/foldwise_table.py` | `results/<model>/*_fw.json` |
| Table: DeepCDR on five splits | `revision/scripts/deepcdr_table.py` | `results/deepcdr/*.json` |
| Table: clustered bootstrap | `revision/scripts/clustered_bootstrap.py` | `results/<model>/*_preds.csv` |
| Table: leave-one-tissue-out | `revision/scripts/loto_table.py`, `loto_audit.py` | `results/<model>/tissue_blind_*_fw.json` |
| Table: calibration on all splits | `revision/scripts/calibration_study.py` | `results/pathxdrp/*_preds.csv` |
| Table: XAI controls / size-matched nulls | `revision/scripts/xai_controls.py` | `results/xai/*.json`, `data/processed/pathway_gene_map.json` |
| Table: head ablation | `revision/scripts/ablation_table.py` | `results/pathxdrp/*_ab*.json`, `results/xai/*_ab*.json` |
| Table: PRISM cross-platform transfer | `revision/scripts/prism_analysis.py`, `prism_table.py` | `results/external/*.csv` |
| Table: computational cost | `revision/scripts/cost_table.py` | ledger, `results/benchmarks/*.json` |
| Leakage diagnostic | `revision/scripts/leakage_diagnostic.py` | splits + raw expression |
| Split/seed audit | `revision/scripts/split_seed_audit.py` | splits |
| Uncertainty utility analysis | `revision/scripts/uncertainty_utility.py` | `results/pathxdrp/*_preds.csv` |

All of these write a `.md` report and a `.csv` to `revision/outputs/`; the
table-emitting ones also write LaTeX to `revision/tables/`.

The LaTeX-side tooling for the manuscript itself (bibliography audit, `.tex`
linting, mechanical source fixes, figure generation for the paper) is not part
of this code release — it operates on manuscript sources that are distributed
through the journal, not here.

---

## 5. Corrections applied during revision

These are changes to the code, not to the writing. Each affects published
numbers and is listed so a reader can tell which version produced what.

### 5.1 Fold-wise expression standardisation

*Was:* `pathxdrp/data/loader.py` Z-scored every gene over all 697 cell lines
before any split was applied, so the moments used to transform a held-out cell
line were partly estimated from that cell line.

*Now:* `load_expression(standardize=False)` returns the raw log2(TPM+1) matrix;
`fit_gene_stats(expr, train_cosmic_ids)` and `apply_gene_stats` fit the moments
on the fold's training cell lines only. Both training entry points take
`--norm {foldwise,cohort}` and default to `foldwise`. Per-fold moments are saved
next to the checkpoint as `<stem>_genestats.npz`.

*Scope:* only cell-blind and tissue-blind hold out cell lines, so only they have
a leakage channel; for random, drug-blind and scaffold-blind every test cell
line also appears in training and the two schemes agree to ~1e-7. Quantified in
`revision/outputs/leakage_diagnostic.md`. The fix is applied to
all five splits and all four models regardless, so the comparison stays uniform.

### 5.2 Calibration used the wrong uncertainty

*Was:* Section 3.4 of the submitted manuscript defines the calibration input as
the total predictive standard deviation
`sigma = sqrt(sigma^2_epistemic + sigma^2_aleatoric)`, but `pathxdrp/train.py`
passed `epistemic` alone to `regression_report(uncertainties=...)`. Every
published ECE and selective-RMSE number therefore used only the epistemic
component.

*Now:* the total variance is used, matching the documented definition. On the
random split this moves ECE from 0.244 to 0.213 — the documented definition is
also the better-calibrated one.

### 5.3 Baseline runs can be tagged

`scripts/train_baseline.py` gained `--run_tag`, so a re-run lands beside rather
than on top of the results backing the submitted manuscript.

### 5.4 Expression matrix is cached

`load_expression(cache=True)` writes
`data/processed/expression_raw_by_cosmic.parquet` so the 507 MB CSV is parsed
once per machine rather than once per run.

---

## 6. Known gaps

Stated rather than hidden.

- **PRISM predictions were regenerated, not recovered.** The predictions behind
  the PRISM table in the *submitted* manuscript were not retained. The revision
  re-ran `scripts/external_validation.py`; `results/external/` now holds those
  predictions and is committed, so the revised PRISM table is reproducible. The
  submitted version's is not.
- **Trained checkpoints (~4.3 GB) are not distributed.** Reproduce with the
  sweep script; the per-fold `_genestats.npz` files make evaluation
  reproducible without them.
- **Split index files (~73 MB) are not distributed.** `scripts/build_splits.py`
  is deterministic and regenerates them.
- **PRISM bulk downloads (~1.9 GB) are not redistributed.** Fetch from the
  DepMap portal; `scripts/prepare_prism.py` reduces them to `data/external/`.
- **The `use_morgan_fp` global fingerprint is computed but unused** once
  `drop_h_mol` is set, which is the default. It is dead compute during training
  and contributes nothing at inference. Removed in the revision.
