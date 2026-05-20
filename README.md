# PathXDRP

**A reproducible evaluation toolkit and quantitative explainability
benchmark for graph-based drug response prediction.**

This repository releases the code, the curated mechanism-of-action (MoA)
benchmark, and the deterministic cross-validation splits behind the paper:

> Gouda, A. H., Zaky, A. B., Anter, A. M. *PathXDRP: a pathway-masked
> cross-attention model and quantitative explainability benchmark for drug
> response prediction.* Submitted to *Information Sciences*.

The manuscript PDF and LaTeX sources are distributed separately (Zenodo /
journal); they are not part of this code release.

---

## Pipeline at a glance

![Pipeline](figures/Pipeline.pdf)

End-to-end flow: raw GDSC2 + DepMap + PubChem + KEGG inputs → graph and
expression tensors → four DRP models trained under five split protocols →
unified metrics, calibration, and the XAI benchmark.

---

## What you get

1. **PathXDRP** — a graph + pathway cross-attention model with a
   residual + LayerNorm wrapped attention block and an attention-only
   auxiliary loss that keeps the attention path load-bearing. The
   regression head returns evidential uncertainty in a single forward pass.
2. **Three reimplemented DRP baselines** sharing the same data loaders,
   splits, training loop and metric pipeline:
   - GraphDRP (Nguyen et al. 2021) — GIN drug encoder + 1D-CNN over expression.
   - DRPreter (Shin et al. 2023) — GATv2 drug + pathway Transformer cell +
     unmasked cross-attention.
   - CDRScan (Chang et al. 2018) — Morgan fingerprints + expression MLP.
3. **Five-split protocol** — `random`, `cell_blind`, `drug_blind`,
   `scaffold_blind`, `tissue_blind` — five seeds each, deterministic
   row-index files in [`data/processed/splits/`](data/processed/splits/)
   (regeneratable from the GDSC2 raw file with
   [`scripts/build_splits.py`](scripts/build_splits.py)).
4. **Quantitative XAI benchmark** — cross-attention, integrated gradients
   on expression, integrated gradients on atom features, permutation
   importance, and occlusion, scored against 237 curated MoA drugs with
   target-gene AUROC, gene-set recall *K*, sensitivity alignment, and
   ROAR-style faithfulness curves.
5. **Calibration metrics** — Expected Calibration Error, risk–coverage
   curves and selective RMSE at coverage 50 / 70 / 90 / 100 %.

---

## Headline results

Test Pearson correlation on GDSC2 IC50, mean ± std over five seeds (fold 0):

| Split          | PathXDRP        | DRPreter      | GraphDRP      | CDRScan         |
| -------------- | --------------- | ------------- | ------------- | --------------- |
| random         | 0.919 ± .010    | 0.925 ± .002  | **0.937 ± .001** | 0.934 ± .001 |
| cell-blind     | 0.867 ± .002    | 0.864 ± .005  | 0.884 ± .002  | **0.890 ± .002** |
| drug-blind     | 0.601 ± .020    | 0.577 ± .043  | 0.538 ± .033  | **0.634 ± .002** |
| scaffold-blind | 0.441 ± .085    | 0.530 ± .115  | **0.548 ± .120** | 0.385 ± .128 |
| tissue-blind   | 0.847 ± .006    | 0.849 ± .004  | 0.862 ± .004  | **0.871 ± .002** |

The XAI benchmark — not raw accuracy — is the central contribution of this
repo. PathXDRP is competitive with the baselines on `cell_blind`,
`drug_blind` and `tissue_blind`, but does not lead on raw IC50 correlation
on any split. Numbers above come from
[`results/summary_v3_vs_baselines.csv`](results/summary_v3_vs_baselines.csv);
the corresponding XAI tables and ROAR-style faithfulness curves are
written by [`scripts/run_xai_multimodel.py`](scripts/run_xai_multimodel.py)
and [`scripts/run_xai_modelagnostic.py`](scripts/run_xai_modelagnostic.py)
to [`results/xai/`](results/xai/).

---

## Repository layout

```
.
├── pathxdrp/                # source: model, baselines, data, eval, explain
│   ├── models/              # PathXDRP model
│   ├── baselines/           # GraphDRP, DRPreter, CDRScan
│   ├── data/                # graph builder, loader, split utilities
│   ├── eval/                # metrics + risk/coverage
│   ├── explain/             # XAI benchmark + model-agnostic attribution
│   └── train.py             # PathXDRP training entry point
├── scripts/                 # CLI runners (data prep, sweeps, XAI, validation)
├── eval/                    # results aggregation + plotting
├── tests/                   # smoke tests
├── configs/default.yaml     # PathXDRP hyperparameter config
├── data/processed/          # curated MoA benchmark, pathway map, ID maps, SMILES
├── results/                 # current per-run metrics, predictions, XAI outputs
│   ├── pathxdrp/ graphdrp/ drpreter/ cdrscan/   # one JSON + preds CSV per run
│   ├── xai/                                      # XAI benchmark outputs
│   └── summary_v3_vs_baselines.csv               # aggregated headline table
├── figures/Pipeline.pdf     # the diagram shown above
└── requirements.txt
```

What is **not** in the repo:

- **Raw data** (GDSC2 CSVs, DepMap RNA-seq matrix, KEGG dumps, Open Targets
  files). Fetch from the upstream providers — see
  [Data sources](#data-sources) below.
- **Trained checkpoints** (~4.3 GB total). Reproduce with
  `scripts/run_sweep.py` on a CUDA GPU.
- **Per-run training logs**, **archived sweeps**, and **generated figures
  other than `figures/Pipeline.pdf`**. All regeneratable from the code.
- **Manuscript LaTeX sources and compiled PDFs.** Distributed separately
  through Zenodo / the journal.

---

## Reproducing the paper

### 0. Environment

```bash
pip install -r requirements.txt
```

`torch-scatter`, `torch-sparse` and `torch-cluster` must be installed
against your CUDA version — follow the PyG install instructions.

### 1. Fetch and prepare data

```bash
# DepMap 24Q4 RNA-seq matrix into data/raw/
python scripts/download_expression.py

# Canonical SMILES from PubChem PUG-REST
python scripts/fetch_smiles.py

# KEGG pathway -> gene map
python scripts/build_pathway_mask.py

# Five-split row-index files (deterministic)
python scripts/build_splits.py

# Curated mechanism-of-action benchmark for the XAI evaluation
python scripts/build_moa_benchmark.py --all
```

You will also need the GDSC2 dose-response CSV (`GDSC2-dataset.csv`) at the
repo root. Download it from CancerRxGene bulk downloads (release 8.4); see
[Data sources](#data-sources).

After this step the curated artefacts live under `data/processed/`:

| File                              | Contents                                    |
| --------------------------------- | ------------------------------------------- |
| `moa_benchmark.json`              | Curated 25-drug MoA panel (manual review)   |
| `moa_benchmark_all.json`          | 237-drug auto-curated MoA list              |
| `pathway_gene_map.json`           | 370 KEGG pathways → 8,403 unique genes      |
| `cosmic_to_depmap.csv`            | 946 GDSC2 cells → DepMap ModelID            |
| `drugs_with_smiles.parquet`       | 247 / 295 GDSC2 drugs with PubChem SMILES   |
| `splits/`                         | Five splits × five seeds × row-index files  |

### 2. Run the full sweep

```bash
python scripts/run_sweep.py
```

This trains every (model × split × seed) combination and writes one JSON
plus a predictions CSV per run under `results/<model>/`. Wall-clock time
is ≈ 33 hours on a single RTX 3060 Laptop GPU; a desktop card cuts it
substantially.

Single-run commands:

```bash
python -u -m pathxdrp.train --split random --seed 0 --fold 0
python -u scripts/train_baseline.py --model graphdrp --split random --seed 0 --fold 0
```

### 3. Aggregate, plot, and run the XAI benchmark

```bash
python eval/analyze_results.py --latex          # writes the LaTeX tables
python eval/plot_results.py                     # writes PDFs/PNGs to figures/

python scripts/run_xai_multimodel.py            # cross-model attention + IG → results/xai/
python scripts/run_xai_modelagnostic.py         # ROAR-style faithfulness curves
python scripts/xai_aggregate.py                 # unified summary across methods
```

The unified XAI summary lands at
[`results/xai/xai_unified_summary.json`](results/xai/xai_unified_summary.json),
with per-model breakdowns in `xai_multimodel_<model>.json` and
`xai_modelagnostic_<model>.json`.

### 4. Optional: external validation

```bash
python scripts/external_validation.py           # CCLE; requires data/external/
python scripts/ensemble_eval.py                 # multi-seed ensembling
python scripts/diagnose_attention.py            # PathXDRP attention diagnostics
```

---

## Data sources

This repository does **not** redistribute primary data. All sources are
public and should be obtained from their providers:

- **GDSC2 dose-response** — Genomics of Drug Sensitivity in Cancer, release
  8.4 — <https://www.cancerrxgene.org/downloads/bulk_download>.
- **DepMap 24Q4 RNA-seq** —
  `OmicsExpressionProteinCodingGenesTPMLogp1.csv` from
  <https://depmap.org/portal/download/all/>.
- **KEGG human pathway–gene mappings** — fetched live through the KEGG REST
  API by `scripts/build_pathway_mask.py`.
- **PubChem canonical SMILES** — fetched live through PUG-REST by
  `scripts/fetch_smiles.py`.

After all filtering steps (cells with RNA-seq, drugs with SMILES, valid
IC50 rows), the final dataset comprises **150,459 rows × 247 drugs × 697
cells** drawn from the original 242,036-row GDSC2 release.

---

## Citation

Pending acceptance. Once published, citation metadata will be added here
and in the Zenodo record.
