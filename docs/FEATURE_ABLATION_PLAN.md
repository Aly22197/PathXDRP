# Feature Ablation Plan — PathXDRP

**Purpose:** Document why we keep / discard each feature from each source dataset, and lay out the ablation experiments needed to defend those choices in the manuscript. Ties directly into XAI novelty axis 4 (Quantitative XAI Benchmark).

**Created:** 2026-05-07
**Status:** Pending Phase 3 gate

---

## 1. Feature inventory — what comes in, what goes to the model

### 1.1 GDSC2-dataset.csv (242,036 rows × 19 cols)

| Column | Used? | Reason |
|--------|-------|--------|
| `DRUG_ID` | Merge key | Links to molecular graph |
| `COSMIC_ID` | Merge key | Links to expression vector |
| `LN_IC50` | **TARGET** | Standard DRP regression label |
| `AUC` | Discarded as input | Alternative response metric — candidate for multi-task auxiliary loss |
| `Z_SCORE` | Discarded | **Label leakage** — derived from LN_IC50 itself |
| `RMSE` | Discarded as input | Curve-fit quality — use for noisy-label filtering |
| `MIN_CONC`, `MAX_CONC` | Discarded as input | Experimental artifact — use for in-range label filtering |
| `PUTATIVE_TARGET` | Discarded as input | Curated label leakage; **kept as XAI ground truth** |
| `PATHWAY_NAME` | Discarded as input | Curated label leakage; **kept as XAI ground truth** |
| `CELL_LINE_NAME`, `DRUG_NAME` | Discarded | Human-readable; IDs are used |
| `TCGA_DESC`, `SANGER_MODEL_ID` | Discarded | Metadata |
| `DATASET`, `NLME_*`, `COMPANY_ID`, `WEBRELEASE` | Discarded | Administrative |

### 1.2 drugs_with_smiles.parquet (621 rows × 5 cols)

| Column | Used? | Reason |
|--------|-------|--------|
| `DRUG_ID` | Merge key | — |
| `SMILES` | **DRUG INPUT** | Converted to PyG graph (334-dim nodes, 10-dim edges), then dropped from df |
| `TARGET` | Discarded as input | Identical to GDSC2 `PUTATIVE_TARGET`; XAI validation only |
| `TARGET_PATHWAY` | Discarded as input | Identical to GDSC2 `PATHWAY_NAME`; XAI validation only |
| `DRUG_NAME` | Discarded | Metadata |

### 1.3 Cell_Lines_Details.xlsx (1,002 rows × 13 cols)

| Column | Used? | Reason |
|--------|-------|--------|
| `COSMIC_ID` | Merge key | — |
| `tissue_2` | **Split construction only** | Defines `tissue_blind` test folds; never a model input |
| `tissue_1`, `cancer_type`, `msi_status`, `growth_props` | Discarded | Derivable from expression; many NaN |

### 1.4 OmicsExpressionProteinCodingGenesTPMLogp1.csv

| Column | Used? | Reason |
|--------|-------|--------|
| `ModelID` (index) | Mapped to COSMIC_ID | — |
| 19,193 gene columns | **CELL INPUT** | Z-scored log2(TPM+1) → PathwaySetEncoder |

### 1.5 pathway_gene_map.json

Not row-level data. 370 KEGG pathways → gene index lists. Defines the **structure** of the cell encoder (pathway grouping + masked attention). Critical for XAI axis 1 (pathway-masked cross-attention).

---

## 2. The four reasons a feature gets discarded

Every discard falls into one of these four categories. Knowing which applies tells us how to defend it in the paper.

| Reason | Examples | Defense in paper |
|--------|----------|------------------|
| **Label leakage** | `Z_SCORE`, `TARGET`, `PATHWAY_NAME` | Show that excluding them prevents trivial shortcuts; *use as XAI ground truth instead* |
| **Experimental artifact** | `MIN_CONC`, `MAX_CONC`, `RMSE` | Show that they reflect lab choices, not biology; *use them for data cleaning* |
| **Derivable from another input** | `cancer_type`, `tissue_1`, `msi_status` | Show that the expression encoder learns these implicitly (mini-ablation) |
| **Pure metadata** | `DRUG_NAME`, `SANGER_MODEL_ID`, `DATASET` | No defense needed — IDs already used |

---

## 3. The XAI angle (axis 4)

The discarded `TARGET` and `TARGET_PATHWAY` annotations are the strongest XAI validation signal we have. The MoA benchmark (`scripts/build_moa_benchmark.py`) already operationalises this:

> **Claim:** PathXDRP's pathway cross-attention assigns higher weight to the drug's known target pathway than to off-target pathways, without ever seeing target annotations during training.

This is a stronger publishable claim than using TARGET as an input feature. Use this framing in §4 of the manuscript.

---

## 4. Ablations — WORTH doing

These directly defend feature choices and add publishable Methods sections.

### 4.1 Gene-panel ablation (cell-side feature selection)

**Question:** Do we need all 19,193 genes, or is a curated subset sufficient?

| Panel | Genes | Hypothesis |
|-------|-------|------------|
| Full transcriptome (current) | 19,193 | Baseline |
| KEGG-pathway-covered only | ~8,403 | Match coverage of pathway map |
| COSMIC cancer gene panel | ~700 | Cancer-relevant only |
| LINCS L1000 landmark | 978 | Industry-standard subset |
| Random-1000 (control) | 1,000 | Sanity baseline |

**Expected result:** COSMIC panel is within 0.01-0.02 PCC of full → deployment story (smaller, cheaper) + biological story (cancer genes suffice).

**Implementation needed:**
- Add `--gene_panel {full,kegg,cosmic,lincs,random1000}` flag to `pathxdrp/train.py`
- Apply mask to `expr_matrix.columns` after loading, before passing to model
- Store gene lists in `data/processed/gene_panels/` (need to download COSMIC list; LINCS L1000 is public)

### 4.2 Drug-feature ablation (drug-side feature selection)

**Question:** Which atom-feature blocks contribute most? Are 256 Morgan FG bits doing the heavy lifting, or the topology?

| Variant | Node dim | What's removed |
|---------|----------|----------------|
| Full (current) | 334 | Baseline |
| No Morgan FG | 78 | Drop the 256 fingerprint bits |
| No stereo (chirality + bond stereo) | 328 | Test stereochemistry contribution |
| Atom + degree only (GraphDRP-style) | 55 | Minimal featurization |

**Expected result:** Removing FG bits costs 0.005-0.015 PCC; the GAT learns much of that signal from topology. Removing stereo costs little. Atom-only loses 0.02-0.03 PCC.

**Implementation needed:**
- Add `--node_feature_set {full,no_fg,no_stereo,atom_only}` flag to `pathxdrp/train.py`
- Modify `pathxdrp/data/graph_utils.py::atom_features()` to accept a feature-set spec
- Re-build graph cache per ablation (cheap — done once at startup)

### 4.3 Noisy-label filtering

**Question:** Do experimental quality filters improve generalisation?

| Variant | Rows | Filter |
|---------|------|--------|
| All rows (current) | 150,459 | None |
| In-range LN_IC50 only | ~135k | Drop rows where exp(LN_IC50) is outside [MIN_CONC, MAX_CONC] |
| Low-RMSE only | ~140k | Drop rows where curve-fit RMSE > 0.3 |
| Both filters | ~125k | Strictest |

**Expected result:** Mild improvement (0.005-0.010 PCC) at fixed compute, possibly bigger gain on hard splits.

**Implementation needed:**
- Add `--filter_in_range`, `--max_curve_rmse 0.3` flags to `pathxdrp/data/loader.py::build_master_df()`
- Apply filter after expression-coverage filter, before reset_index

### 4.4 Multi-task with AUC

**Question:** Does AUC as auxiliary target regularise the encoder?

```
loss = MSE(LN_IC50_pred, LN_IC50_true) + lambda_auc * MSE(AUC_pred, AUC_true)
```

**Expected result:** Small but real gain (0.005-0.015 PCC) from regularisation pressure on shared representations.

**Implementation needed:**
- Add second output head to `pathxdrp/models/evidential_head.py` or a parallel MSE head
- Add `--multitask_auc --lambda_auc 0.5` flags to `pathxdrp/train.py`
- Modify `GDSCDataset.__getitem__` to also return AUC (need to add `AUC` back to slim df)

### 4.5 Categorical metadata add-on (negative result expected)

**Question:** Do explicit `tissue_2` / `cancer_type` one-hot features help, or has the expression encoder already learned this?

**Variant:** Concatenate one-hot `tissue_2` (53 dims) + `cancer_type` (31 dims, with imputed "unknown") to the cell embedding `h_cell` before the cross-attention.

**Expected result:** Negligible gain (< 0.005 PCC), defending the expression-only choice. **Publishable as a negative result** that strengthens the XAI story.

**Implementation needed:**
- Add `tissue_2`, `cancer_type` back to slim df
- Add `--use_categorical_metadata` flag
- Modify `pathxdrp/models/cell_encoder.py` to optionally concatenate one-hots

---

## 5. Ablations NOT worth doing

| Ablation | Why skip |
|----------|----------|
| PCA / UMAP on expression | Destroys per-pathway interpretability — kills XAI axis 4 |
| Autoencoder bottleneck on expression | Same problem; black-box latent space |
| LASSO / elastic-net feature selection | Couples selection to a specific train split → CV contamination |
| Top-N variance filter | Arbitrary cutoff, no biological grounding |
| Adding `TARGET` / `PATHWAY_NAME` as input | Label leakage — destroys MoA validation claim |
| Adding `CELL_LINE_NAME` or `DRUG_NAME` text embeddings | Adds NLP confound; no signal beyond IDs |
| Using GDSC1 + GDSC2 combined | Cross-protocol noise; standard practice is GDSC2 only |

---

## 6. Compute budget summary

| Block | Runs | Approx GPU hrs (7.5h/run) |
|-------|------|---------------------------|
| 4.1 Gene panel (5 variants × random/seed0) | 5 | 37.5 |
| 4.2 Drug feature (4 variants × random/seed0, 1 is baseline) | 3 | 22.5 |
| 4.3 Noisy-label (3 variants) | 3 | 22.5 |
| 4.4 Multi-task AUC | 1 | 7.5 |
| 4.5 Categorical metadata | 1 | 7.5 |
| **Total** | **13** | **~97 GPU hrs** |

This is on top of the main sweep (Phase 3 + 5-seed full sweep). Run only after Phase 3 gate clears.

---

## 7. Commands (after implementation work)

All commands assume the current default architecture and run on `random/seed0/fold0` for the ablation table. After establishing the best variant, re-run on all 5 splits × 5 seeds for the main paper table.

### 7.1 Gene-panel ablations

```bash
# Baseline (already part of Phase 3 gate)
python -u -m pathxdrp.train --split random --seed 0 --fold 0 --epochs 150

# KEGG-covered genes only
python -u -m pathxdrp.train --split random --seed 0 --fold 0 --epochs 150 --gene_panel kegg

# COSMIC cancer panel
python -u -m pathxdrp.train --split random --seed 0 --fold 0 --epochs 150 --gene_panel cosmic

# LINCS L1000
python -u -m pathxdrp.train --split random --seed 0 --fold 0 --epochs 150 --gene_panel lincs

# Random-1000 control
python -u -m pathxdrp.train --split random --seed 0 --fold 0 --epochs 150 --gene_panel random1000
```

### 7.2 Drug-feature ablations

```bash
# Drop Morgan FG bits (node_dim 334 -> 78)
python -u -m pathxdrp.train --split random --seed 0 --fold 0 --epochs 150 --node_feature_set no_fg

# Drop chirality + bond stereo
python -u -m pathxdrp.train --split random --seed 0 --fold 0 --epochs 150 --node_feature_set no_stereo

# Atom + degree only (minimal, GraphDRP-style)
python -u -m pathxdrp.train --split random --seed 0 --fold 0 --epochs 150 --node_feature_set atom_only
```

### 7.3 Noisy-label filtering

```bash
# In-range LN_IC50 only
python -u -m pathxdrp.train --split random --seed 0 --fold 0 --epochs 150 --filter_in_range

# Low curve-fit RMSE only
python -u -m pathxdrp.train --split random --seed 0 --fold 0 --epochs 150 --max_curve_rmse 0.3

# Both filters combined
python -u -m pathxdrp.train --split random --seed 0 --fold 0 --epochs 150 --filter_in_range --max_curve_rmse 0.3
```

### 7.4 Multi-task with AUC

```bash
python -u -m pathxdrp.train --split random --seed 0 --fold 0 --epochs 150 --multitask_auc --lambda_auc 0.5
```

### 7.5 Categorical metadata add-on

```bash
python -u -m pathxdrp.train --split random --seed 0 --fold 0 --epochs 150 --use_categorical_metadata
```

---

## 8. Manuscript-ready language

Drop these into the relevant Methods sections:

### Justifying the discard of `TARGET` / `PATHWAY_NAME` (§3.1 Inputs)

> We deliberately exclude curated drug-target and pathway annotations (`TARGET`, `PATHWAY_NAME`) from model inputs. These annotations are added post-hoc by domain experts who have already observed the drug's IC50 patterns, so including them risks shortcut learning that bypasses molecular structure. Equally important, drug-blind and scaffold-blind generalisation requires predicting for novel compounds where such annotations do not exist. Instead, we retain these annotations as held-out **ground truth for the XAI validation in §4**: the cross-attention pathway weights of a trained model are evaluated against these annotations to test whether the model has implicitly learned mechanism-of-action.

### Justifying the discard of experimental artifacts (§3.2 Data Cleaning)

> Experimental fields `MIN_CONC`, `MAX_CONC`, and `RMSE` (curve-fit quality) reflect assay protocol choices rather than tumour biology and are therefore excluded as model inputs. We do, however, use them for **label-quality filtering** (§3.2.1): rows whose fitted LN_IC50 falls outside the tested concentration range are extrapolated and noisy.

### Justifying expression-only cell representation (§3.3 Cell Encoder)

> Categorical cell metadata (`tissue_2`, `cancer_type`, `msi_status`) is excluded on the principle that whole-transcriptome expression already encodes tissue identity, MSI status, and cancer subtype. We verify this empirically with an ablation (§5.X.5): adding one-hot tissue and cancer-type features yields no significant gain, confirming the expression encoder has learned these signals end-to-end.

### Justifying full transcriptome (§3.3.1 Gene Selection)

> We feed all 19,193 protein-coding genes to the model rather than a curated panel, because PathXDRP's interpretability claim (§4) is defined at the pathway level — restricting genes a priori would weaken the biological prior. We compare with three curated panels (COSMIC cancer genes, LINCS L1000, KEGG-covered genes) in §5.X.1 and show that the full transcriptome is competitive while preserving pathway-level XAI granularity.

---

## 9. Notes

- **All ablations require code changes first** — none of the flags above currently exist. Implementation is straightforward (1-2 hours per ablation block) but should land before queueing GPU runs.
- **Run after Phase 3 gate clears.** No point ablating an architecture that hasn't reached PCC ≥ 0.93 baseline.
- **Use random/seed0/fold0 for the ablation table.** Once the best variant is identified, re-run on 5 splits × 5 seeds for the main paper.
- **Aggregate via `eval/analyze_results.py`** — the script auto-discovers result subdirectories.
