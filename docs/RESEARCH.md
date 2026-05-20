# PathXDRP — Research Documentation

**Pathway-masked Cross-attention Drug Response Predictor with Foundation-model Encoders and Calibrated Evidential Uncertainty**

Target venue: *Briefings in Bioinformatics* (IF ~9) or *Bioinformatics* (IF ~5.8)

---

## Table of Contents

1. [Research Problem](#1-research-problem)
2. [Why This Approach Is Novel](#2-why-this-approach-is-novel)
3. [Four Claimed Contributions](#3-four-claimed-contributions)
4. [Dataset](#4-dataset)
5. [Feature Engineering](#5-feature-engineering)
6. [Model Architecture](#6-model-architecture)
7. [Foundation Model Encoders (Phase 4)](#7-foundation-model-encoders-phase-4)
8. [Baseline Models](#8-baseline-models)
9. [Loss Function and Training](#9-loss-function-and-training)
10. [Evaluation Protocol](#10-evaluation-protocol)
11. [Uncertainty Quantification](#11-uncertainty-quantification)
12. [Quantitative Explainability Benchmark](#12-quantitative-explainability-benchmark)
13. [External Validation](#13-external-validation)
14. [Implementation Details](#14-implementation-details)

---

## 1. Research Problem

### 1.1 What Is Drug Response Prediction?

Cancer cell lines respond differently to the same drug. A cell line derived from a lung adenocarcinoma may be exquisitely sensitive to erlotinib (an EGFR inhibitor) while a colorectal cancer cell line is completely resistant. **Drug response prediction (DRP)** is the task of predicting the quantitative sensitivity of a given cell line to a given drug — typically expressed as **LN(IC50)**, the natural logarithm of the half-maximal inhibitory concentration in micromolar.

Getting this prediction right has direct clinical consequences:
- **Precision oncology**: guide which drug to give a patient based on their tumour's molecular profile.
- **Drug discovery**: prioritise compounds in pre-clinical screens across hundreds of cancer subtypes.
- **Combination therapy**: identify synergistic drug pairs for a specific cancer type.

### 1.2 What Data Do We Have?

The **Genomics of Drug Sensitivity in Cancer (GDSC2)** database is the primary resource. It provides systematic pharmacological profiling of ~295 anti-cancer compounds across ~969 cancer cell lines, yielding ~242,000 dose-response measurements. Each measurement gives a fitted **LN(IC50)** (log µM) as the regression target.

Cell lines are characterised by gene expression data from the **DepMap 24Q4** release: RNA-seq log2(TPM+1) profiles for 19,193 protein-coding genes across 1,673 cell models, standardised to the COSMIC cell line registry. After filtering to cell lines with both IC50 and expression data, and drugs with valid SMILES, the working dataset is **150,459 IC50 measurements across 247 drugs and 697 cell lines**.

### 1.3 Why Is This Hard?

**Biological complexity.** Drug sensitivity depends on thousands of interacting molecular factors: somatic mutations, copy number alterations, methylation patterns, and most critically, the transcriptional state of the cell. The relationship between a drug's molecular structure and a cell's transcriptional state is highly non-linear and involves multiple biological scales.

**Data scarcity.** 150,459 samples sounds large, but when split drug-blind (held-out drugs never seen in training), each held-out drug has a completely novel chemical scaffold the model has never seen. Similarly in cell-blind evaluation, each held-out cell line's expression profile is entirely new to the model.

**Evaluation leakage.** Much published DRP literature reports random-split Pearson r, where train and test share both drugs and cell lines. In this regime, a model that has memorised "cell line X is generally sensitive/resistant" will score well, regardless of whether it has learned true drug-cell biology. **Drug-blind** and **cell-blind** evaluations are the biologically meaningful tests, and DRP models typically perform significantly worse on them.

**Interpretability gap.** Oncologists and pharmacologists need to understand *why* a model predicts sensitivity — which molecular features of the drug drive the prediction, and which biological pathways of the cell are implicated. Most DRP models are black boxes whose explanations are anecdotal and unvalidated.

---

## 2. Why This Approach Is Novel

### 2.1 Prior Art Landscape

Cross-attention between drug and cell representations has been explored, most directly by **XGDP** (Cai et al., *Scientific Reports* 2024), which uses a GNN drug encoder, CNN cell encoder, and standard cross-attention, then applies GNNExplainer and Integrated Gradients. The architectural overlap with a naive "GAT + expression + cross-attention" design is near-complete.

Other near-precedents:

| Model | Key Overlap |
|-------|------------|
| DRPreter (Shin 2023, *IJMS*) | GATv2 drug + pathway-grouped Transformer cell + cross-attention + MSE |
| DGDRP (2024, *Front. Genet.*) | Drug-specific gene selection via biological-network re-ranking |
| TransCDR (2024, *BMC Biology*) | Drug SMILES Transformer + expression Transformer + cross-attention |
| GSDRP (2024) | Cross-attention fusing omics and drug graph features |
| drGAT (2024) | Heterogeneous attention over drug-gene-cell graph |
| DeepCCDS (2025, *Adv. Science*) | Cancer-driver-signal + self-supervised pretraining; current SOTA |

### 2.2 What Remains Genuinely Under-explored

The literature review reveals four axes where PathXDRP occupies novel ground, each independently defensible:

1. **Cross-attention constrained by KEGG pathway membership** (not just free-form) — turns attention weights into mechanistically interpretable outputs by construction, not post-hoc.

2. **Foundation model encoders integrated with pathway structure** — GeneMamba (pretrained on 30M cells), Graph-Mamba drug encoder — no GDSC DRP paper has coupled these with pathway-structured cross-attention.

3. **Calibrated evidential uncertainty with selective prediction analysis** — prior DRP work reports point estimates only. A 2026 bioRxiv study showed ~64% MSE reduction by filtering predictions with low uncertainty; PathXDRP makes this a first-class output, not an afterthought.

4. **Quantitative XAI benchmark against known drug–target ground truth** — all prior DRP explainability is anecdotal saliency maps. PathXDRP benchmarks six XAI methods against verified drug targets (AUROC against COSMIC Cancer Gene Census), producing the first systematic XAI comparison in this domain.

---

## 3. Four Claimed Contributions

### C1 — Pathway-masked Cross-attention

Drug atoms attend over cell-line pathways, not over individual genes. The attention mask is constructed from **KEGG human pathway** gene memberships (370 pathways, 37,950 gene-pathway pairs), so each attention weight has a direct biological interpretation: *"atom a of the drug activates/inhibits biological pathway P of the cell."*

This is the architectural thesis of the paper. It converts interpretability from a post-hoc analysis into a structural inductive bias. The attention map does not need to be reverse-engineered — it is the mechanism. Two mask variants are implemented:
- **Hard mask**: binary, built once from KEGG priors; certain atom-pathway combinations are structurally forbidden.
- **Soft mask**: a learned scalar prior added to logits for each pathway, allowing the model to up- or down-weight biological priors during training.

### C2 — Foundation-model Dual Encoders

Standard DRP drug encoders (GIN, GATv2) are trained from scratch on 247 GDSC drugs — too few examples to learn rich chemical representations. Standard cell encoders compress 19,193 genes through a shallow CNN, discarding most of the transcriptional signal.

PathXDRP replaces both with pretrained foundation models:
- **Drug branch**: MolFormer-XL (IBM, *Nat. Mach. Intell.* 2022) — SMILES-pretrained on 1.1 billion molecules — or Graph-Mamba (GAT + bidirectional Mamba over atom sequences).
- **Cell branch**: GeneMamba (65.7M params, pretrained on 30M single cells, *arXiv* 2025) — or scGPT (*Nat. Methods* 2024) in bulk-mode.

Both foundation backbones are frozen by default; only a small adapter network is trained on GDSC. This drastically reduces the parameter count of the trainable portion (~1.8M) while leveraging pre-learned representations far richer than what 150K examples could produce.

### C3 — Calibrated Evidential Uncertainty

PathXDRP produces four outputs per prediction, not one: **µ** (point estimate), **aleatoric variance** σ²_a (irreducible data noise), and **epistemic variance** σ²_e (model uncertainty due to limited data), all from a single forward pass via an evidential regression head (Normal-Inverse-Gamma parameterisation, Amini et al., NeurIPS 2020).

For deployment, epistemic uncertainty is the clinically useful quantity: high epistemic uncertainty means the model has not seen drugs or cell lines similar to this test case, and the prediction should not be trusted. PathXDRP provides **selective prediction** — on a held-out test set, predictions are ranked by confidence (ascending epistemic uncertainty) and performance at each coverage level is reported via **risk-coverage curves**. The target is a ≥25% RMSE reduction at 80% coverage versus full-coverage prediction.

A deep ensemble of 5 seeds provides an independent uncertainty estimate to validate the single-model evidential head.

### C4 — Quantitative Explainability Benchmark

The paper presents the first systematic benchmarking of six XAI methods on a curated panel of 25 GDSC drugs with **known mechanism of action**, scored against verified biological ground truth (COSMIC Cancer Gene Census, KEGG pathway annotations). This benchmark is itself a contribution independent of PathXDRP's predictive performance. The headline claim is: **PathXDRP's architectural attention rollout outperforms post-hoc GNNExplainer and Integrated Gradients on target-AUROC for ≥3 of 6 metrics**, while being orders of magnitude cheaper to compute.

---

## 4. Dataset

### 4.1 Drug Response Data — GDSC2

**Source:** Genomics of Drug Sensitivity in Cancer, release 8.5 (Sanger/EMBL-EBI).

**Content:** Fitted dose-response IC50 values for 295 anti-cancer compounds screened against up to 969 cancer cell lines using a standard cell viability assay (CellTiter-Glo).

**Target variable:** `LN_IC50` — the natural logarithm of the IC50 in micromolar. Using the log-transformed value linearises the heavily right-skewed raw IC50 distribution and gives better-behaved regression residuals.

**After filtering:**
- Require valid SMILES (via PubChem REST API): 247/295 drugs retained.
- Require DepMap RNA-seq expression data (see §4.2): 697/969 cell lines retained.
- After these two filters: **150,459 (drug, cell) IC50 pairs**, de-duplicated.

**Drug annotation:** `Compounds-annotation.csv` provides `DRUG_ID`, `DRUG_NAME`, `TARGET` (gene name of primary target), and `TARGET_PATHWAY` (e.g. "EGFR signalling", "DNA replication").

**Cell line metadata:** `Cell_Lines_Details.xlsx` provides `COSMIC_ID`, `CELL_LINE_NAME`, `tissue_1` (broad tissue type), `tissue_2` (detailed subtype), `cancer_type` (TCGA classification), and `MSI_STATUS` (microsatellite instability).

### 4.2 Gene Expression Data — DepMap 24Q4

**Source:** Broad Institute DepMap release 24Q4 (October 2024), downloaded from Figshare article 27993248.

**File:** `OmicsExpressionProteinCodingGenesTPMLogp1.csv` (~507 MB compressed).

**Content:** log2(TPM + 1) gene expression values for **19,193 protein-coding genes** across **1,673 cancer models** (indexed by DepMap ModelID, e.g. `ACH-000001`). Column names have the format `GENE_SYMBOL (entrez_id)`, e.g. `TSPAN6 (7105)`.

**COSMIC-to-DepMap mapping:** GDSC2 identifies cell lines by COSMIC ID (integer), while DepMap uses ModelIDs. A mapping file `cosmic_to_depmap.csv` (built from DepMap's `Model.csv` metadata file) provides the correspondence. 946 of 969 GDSC2 cell lines have a DepMap ModelID; 697 of those have RNA-seq expression data in the 24Q4 release.

**Preprocessing:**
1. Strip entrez ID suffixes from column names: `TSPAN6 (7105)` → `TSPAN6`.
2. Remap rows from DepMap ModelID to COSMIC ID (integer).
3. **Z-score per gene across cell lines**: subtract mean, divide by standard deviation (computed on the full 1,673-model set before filtering). This is the standard transformation for gene expression in DRP — it removes mean expression level (which reflects library prep and gene length) and leaves relative over/under-expression across cell lines.
4. Cast to float32.

**Output:** `expr_matrix` — shape (697, 19193), dtype float32, indexed by COSMIC_ID.

### 4.3 KEGG Pathway Gene Map

**Source:** KEGG REST API, endpoint `https://rest.kegg.jp/link/pathway/hsa` (bulk gene-pathway links for *Homo sapiens*).

**Content:** All human KEGG pathway–gene associations, expressed as Entrez gene IDs linked to KEGG pathway IDs (e.g. `hsa00010` = "Glycolysis / Gluconeogenesis"). Pathway names are fetched separately via `https://rest.kegg.jp/list/pathway/hsa`.

**Entrez-to-symbol mapping:** The file `gene_identifiers_20241212.csv` (COSMIC gene census) maps Entrez IDs to HGNC gene symbols. Only genes present in the DepMap expression matrix are retained.

**Output:** `pathway_gene_map.json` — a dict `{pathway_name: [gene_symbols]}` with:
- **370 KEGG human pathways** with ≥1 gene in the expression matrix.
- **37,950 total (pathway, gene) pairs**.
- **8,403 unique genes** covered (43.8% of the 19,193-gene expression matrix).
- Average 102.6 genes per pathway; range 1–514.

In training, gene symbols are converted to integer column indices in the expression matrix:
```python
gene_to_idx = {g: i for i, g in enumerate(expr_matrix.columns)}
pathway_gene_map = {
    pw: [gene_to_idx[g] for g in genes if g in gene_to_idx]
    for pw, genes in pathway_gene_symbols.items()
}
```

### 4.4 MoA Benchmark Panel

For the explainability benchmark (C4), a curated panel of 25 GDSC2 drugs with well-characterised single or paired targets is assembled from DrugBank, KEGG, and primary literature:

| Drug | Primary Target | Target Pathway |
|------|---------------|----------------|
| Erlotinib | EGFR | EGFR signalling |
| Gefitinib | EGFR | EGFR signalling |
| Lapatinib | EGFR / HER2 | EGFR signalling |
| Trametinib | MEK1/2 (MAP2K1/2) | RAS–MAPK |
| Dabrafenib | BRAF | RAS–MAPK |
| PLX4720 | BRAF V600E | RAS–MAPK |
| Vemurafenib | BRAF V600E | RAS–MAPK |
| Nilotinib | BCR-ABL / KIT | BCR-ABL signalling |
| Imatinib | ABL / KIT / PDGFR | BCR-ABL signalling |
| Olaparib | PARP1/2 | DNA repair |
| Navitoclax | BCL2 / BCL-XL | Apoptosis |
| Venetoclax | BCL2 | Apoptosis |
| AZD7762 | CHK1 | Cell cycle / DNA damage |
| MK-2206 | AKT | PI3K–AKT |
| AZD6244 (Selumetinib) | MEK1/2 | RAS–MAPK |
| Tamoxifen | ESR1 | Estrogen signalling |
| 5-Fluorouracil | TYMS | Pyrimidine metabolism |
| Cisplatin | DNA (double-strand break) | DNA replication |
| Bortezomib | Proteasome (PSMB5) | Protein degradation |
| Crizotinib | ALK / MET | EGFR / MET signalling |
| JNJ-7706621 | CDK1/2/4 | Cell cycle |
| Nutlin-3 | MDM2 | p53 signalling |
| AZD8055 | mTOR | mTOR signalling |
| KU-55933 | ATM | DNA damage response |
| Cetuximab | EGFR | EGFR signalling |

For each drug, ground-truth target genes (HGNC symbols), parent KEGG pathway ID, and key chemical substructure annotations are stored in `data/processed/moa_benchmark.json`.

---

## 5. Feature Engineering

### 5.1 Drug Molecular Graph

SMILES strings are converted to PyTorch Geometric `Data` objects with **RDKit**. Each drug is represented as an undirected molecular graph where atoms are nodes and bonds are edges; all bond directions are made bidirectional (each bond generates two directed edges in the graph).

**Atom features (335 dimensions total):**

| Feature group | Dimensions | Details |
|--------------|-----------|---------|
| Atom type | 44 + 1 | One-hot over 43 common elements (C, N, O, S, F, Si, P, Cl, Br, Mg, Na, ...) + "other" |
| Degree | 11 | One-hot over 0–10 bonds |
| Formal charge | 7 | Bucketed to range [−3, +3] |
| Hybridisation | 5 + 1 | SP, SP2, SP3, SP3D, SP3D2, other |
| Aromaticity | 1 | Boolean |
| In-ring | 1 | Boolean |
| Chirality | 3 | CW, CCW, unspecified |
| Num Hs | 5 | 0–4 H count |
| Morgan FG bits | 256 | One-hot over top-256 Morgan-radius-2 fragment types (functional group identifier per atom) |

The Morgan FG bits are the most distinctive feature. For each atom, we compute the Morgan radius-2 fingerprint of its local neighbourhood (the substructure centered on that atom within 2 bond hops). The bit position in the 256-bit vocabulary encodes the specific functional group or chemical environment. This gives the model a local chemical context for each atom that a simple atom type cannot capture — a carbon in an amide group vs. a carbon in a vinyl ether are indistinguishable from atomic type alone but differ at bit position level.

The FG vocabulary is built from the training drugs:
```python
fg_vocab = build_fg_vocab(smiles_list, top_k=256)
# {morgan_bit_id: index_0..255}
```

**Bond features (10 dimensions):**

| Feature | Dimensions | Details |
|---------|-----------|---------|
| Bond type | 4 + 1 | Single, double, triple, aromatic, other |
| Conjugated | 1 | Boolean |
| In-ring | 1 | Boolean |
| Stereo | 3 | E, Z, unspecified |

**Graph cache:** all 247 drug graphs are precomputed once before training and stored in a dict `{DRUG_ID: PyG Data}`. The FG vocabulary is fixed after the initial computation and reused for all graphs.

### 5.2 Gene Expression Processing

**Z-scoring:** each gene is normalised across the 697 cell lines to mean 0, std 1. A Z-score above 0 means the gene is expressed above the population mean for that cell line, below 0 means below. This captures relative over/under-expression, which is the biologically meaningful signal for drug sensitivity (a gene being absolutely highly expressed is less informative than it being unusually highly expressed relative to other cell lines).

**Pathway statistics** (computed in `PathwaySetEncoder`): for each of the 370 KEGG pathways, three statistics are computed over the genes in that pathway in a given cell line:
- **Mean expression**: average Z-score across pathway genes. Indicates overall activity level of the pathway.
- **Std expression**: standard deviation of Z-scores within the pathway. High std indicates heterogeneous gene activation within the pathway; low std indicates uniform up- or down-regulation.
- **Fraction active**: fraction of pathway genes with Z-score > 0 (i.e., expressed above the cell-line population mean). Distinguishes pathways where many genes are coordinately activated (high frac_active, high mean) from pathways where one outlier gene dominates (high mean, low frac_active).

These three statistics are computed vectorially using a precomputed **averaging matrix** (n_pathways × n_pairs, float32), avoiding Python loops:
```python
mean_pw = gene_expr @ avg_matrix.T         # (B, P)
std_pw  = sqrt(mean_sq_pw - mean_pw²)      # (B, P)
frac_pw = (gene_expr > 0) @ avg_matrix.T   # (B, P)
```
This is equivalent to three dense matrix multiplications (cuBLAS GEMMs on GPU), making the pathway encoding negligible compared to the attention computation.

---

## 6. Model Architecture

### 6.1 Overview

```
Drug (SMILES)                Cell Line (RNA-seq)
     │                              │
     ▼                              ▼
[DrugGATEncoder]           [PathwaySetEncoder]
     │                              │
h_atom (N_atoms, D)        h_cell (B, P, D)
h_mol  (B, 2D)             │
     │                     │
     └──────────┬───────────┘
                ▼
  [PathwayMaskedCrossAttention]
        Q = h_atom,  K = V = h_cell
                │
        context (N_atoms, D)
        attn_weights (N_atoms, P)
                │
                ▼
  [Attention-weighted pooling]
        h_drug_context (B, D)
                │
  [Learned attention pooling over pathways]
        h_cell_global (B, D)
                │
  Concatenate: [h_drug_context || h_mol || h_cell_global || h_drug_context ⊙ h_cell_global]
        z (B, 5D)
                │
  [EvidentialRegressionHead]
        µ, ν, α, β  →  LN_IC50 prediction + uncertainty
```

### 6.2 Drug Encoder — DrugGATEncoder

**Architecture:** input projection → GATv2 message-passing stack → dual global pooling.

**Input projection:** `Linear(335, D)` + GELU. Projects atom features to the working dimension D.

**GATv2 layers** (default: 4 layers, 8 heads, D=256):
- Each layer: `GATv2Conv(D, D//n_heads, heads=n_heads, edge_dim=10)` with `concat=True`.
- After each layer: residual connection, `LayerNorm(D)`, GELU.
- Edge features (10-dim bond features) are used as additional key context in GATv2.
- GATv2 (Brody et al., 2022) fixes the "static attention" problem of original GAT by computing attention scores as `a^T · LeakyReLU(W_l · h_i + W_r · h_j + W_e · e_ij)` — the scoring function sees both source and target features jointly, allowing genuinely dynamic attention.

**Output:**
- `h_atom`: (N_atoms_total, D) — per-atom embeddings, passed as Query to cross-attention.
- `h_mol`: (B, 2D) — concatenation of global mean-pool and global max-pool over all atoms. Mean pool captures average chemical environment; max pool captures the most activated chemical feature.

**Optional MolFormer injection** (Phase 4, `drug_encoder_type='molformer'`):
- Frozen IBM MolFormer-XL (768-dim CLS) projected to D and added to all atom embeddings.
- Broadcasts a global sequence-level context to every atom.

**Parameter count (default, D=256, 4 layers, 8 heads):** ~1.1M.

### 6.3 Cell Encoder — PathwaySetEncoder

**Architecture:** three-statistic pathway features → per-pathway projection → cross-pathway Transformer.

**Step 1: Pathway statistics** (described in §5.2). For each of the 370 KEGG pathways, compute [mean, std, frac_active] across the pathway's genes. Output shape: (B, 370, 3).

**Step 2: Per-pathway projection.**
```python
self.gene_proj = nn.Sequential(
    nn.Linear(3, D),   # maps [mean, std, frac] → D-dimensional token
    nn.GELU(),
    nn.LayerNorm(D),
)
```
Output: (B, 370, D) — 370 pathway tokens per cell line.

**Step 3: Cross-pathway Transformer** (default: 1 layer, n_heads = D//32).
```python
nn.TransformerEncoder(
    nn.TransformerEncoderLayer(
        d_model=D, nhead=D//32, dim_feedforward=D*2,
        dropout=0.0, batch_first=True, norm_first=True,  # pre-LN
    ),
    num_layers=1,
    enable_nested_tensor=False,
)
```
This is the critical addition over DRPreter: the transformer allows each pathway token to attend to all other pathway tokens, modelling **inter-pathway crosstalk**. Biologically, this captures co-activation patterns: the PI3K pathway token can "see" the state of the MAPK pathway and adjust its representation accordingly. Pre-LayerNorm (norm_first=True) is used for training stability. No dropout inside the transformer — sparse pathway attention is already a regularising signal.

**Step 4: LayerNorm + Dropout.** Output: (B, 370, D).

**Inductive bias:** by grouping genes into pathway sets *before* projection, the encoder is forced to use biologically meaningful units as its vocabulary, rather than individual genes. This prevents the model from overfitting to idiosyncratic expression levels of a few genes.

**Parameter count (D=256, 1 Transformer layer):** ~440K.

### 6.4 Cross-Attention — PathwayMaskedCrossAttention

This is the architectural centrepiece of PathXDRP. Drug atoms (Query) attend over cell-line pathway tokens (Key, Value).

**Memory-efficient design:** the naive approach of expanding K and V to match each atom would create tensors of shape (N_atoms_total, P, D), which at batch_size=256 with 370 pathways and D=256 requires ~1 GB just for K. Instead:

1. **Pad atoms to dense batch**: `to_dense_batch(Q, atom_batch)` produces (B, max_n_atoms, D) with a boolean padding mask. Memory: O(B × max_n × D).
2. **Compute attention in (B, H, max_n, P) space**: 20× less memory than the naive approach.
3. **Unpad back to (N_atoms_total, D)**: apply the inverse of the padding mask.

**Padding mask handling:** padding positions receive score `-1e4` (not `-inf`). Using `-inf` produces `NaN` gradients via `0 × NaN = NaN` in IEEE 754, corrupting training from batch 2 onwards. The `-1e4` value is small enough to make softmax attention negligible but avoids the NaN pathology. It also fits in float16 (max ~65,504) for automatic mixed precision safety.

**Mask types:**

*None:* standard scaled dot-product attention. Used as ablation baseline.

*Soft:* one learnable scalar logit per pathway, shape (1, 1, 1, P), added to attention scores before softmax:
```python
scores = scores + self.soft_mask_logit   # (1, 1, 1, P) broadcast
```
This is a per-pathway learned prior — the model can up-weight biologically relevant pathways globally. The critical design choice is the (1, 1, 1, P) shape (not (1, H, 1, 1) which would broadcast uniformly and cancel in softmax).

*Hard:* binary (N_atoms, P) mask where `True` = attention allowed. Built from KEGG priors. A `False` entry receives score `-1e4`. This is the most biologically constrained variant.

**Multi-head attention:** Q, K, V projections with `LayerNorm` on Q and KV. `n_heads` heads of dimension D/n_heads each. Attention scores: `einsum("bhnd,bhpd->bhnp", Q_mh, K_mh) / sqrt(D/n_heads)`.

**Entropy regularisation:**
```
L_entropy = -mean over atoms of sum_p [ a_p * log(a_p) ]
```
This penalises flat attention distributions (high entropy = the atom doesn't know which pathway to attend to) and rewards focused, pathway-specific attention patterns. Added to the main loss weighted by `entropy_reg_weight` (default 0.01). It serves two purposes: improved XAI (sparser attention maps are more interpretable) and a gentle regulariser against attention collapse.

**Output:**
- `context`: (N_atoms_total, D) — each atom's attended joint drug-cell feature.
- `attn_weights`: (N_atoms_total, P) — per-atom attention over 370 pathways, used for XAI.
- `_entropy_loss`: scalar, cached for use in the total loss.

**Parameter count:** ~525K (4 projection matrices + out_proj + LayerNorms).

### 6.5 Pooling and Joint Representation

**Drug-side pooling (attention-weighted global pool):**

After cross-attention, each atom has a context vector and an attention weight distribution over 370 pathways. We pool atoms to a molecule-level vector using per-atom importance derived from the attention weights:
```python
a_weights = attn_weights.max(dim=-1, keepdim=True)[0]  # (N_atoms, 1)
h_drug_context = global_add_pool(context * a_weights, drug_batch.batch)  # (B, D)
h_drug_context = LayerNorm(Linear(D, D)(h_drug_context))
```
The importance weight is the **maximum attention to any single pathway** (not mean). Using `mean(dim=-1)` gives `1/P` for every atom regardless of distribution (uniform attention = mean = 1/P), making the weighting a no-op. Using `max(dim=-1)` rewards atoms that focus sharply on a single pathway — i.e., atoms that have a specific and confident pathway interaction.

**Cell-side pooling (learned attention over pathway tokens):**

A single learnable query vector (shape (1, 1, D), initialised near zero) attends over all 370 pathway tokens to produce a cell-global representation:
```python
pool_scores  = (cell_pool_q @ h_cell.T) / sqrt(D)    # (B, 1, P)
pool_weights = softmax(pool_scores, dim=-1)            # (B, 1, P)
h_cell_global = (pool_weights @ h_cell).squeeze(1)    # (B, D)
```
This learns which pathways are globally most informative across all drugs.

**Joint representation:**

The final input to the prediction head concatenates four complementary representations:
```python
h_int = h_drug_context * h_cell_global               # (B, D) — bilinear interaction
z = cat([h_drug_context, h_mol, h_cell_global, h_int], dim=-1)  # (B, 5D)
```

| Component | Shape | What it captures |
|-----------|-------|-----------------|
| `h_drug_context` | (B, D) | Drug-atom context after attending over cell pathways |
| `h_mol` | (B, 2D) | Raw GAT molecular embedding (mean + max pool) |
| `h_cell_global` | (B, D) | Cell pathway state, globally pooled |
| `h_int = h_drug_context ⊙ h_cell_global` | (B, D) | Element-wise product — explicit drug-cell feature co-occurrence ("does this drug feature match this cell vulnerability?") |

The interaction term is a bilinear product: for each dimension d, it computes how much the drug's context in dimension d aligns with the cell's state in dimension d. This is analogous to asking "which chemical motifs of the drug match which biological vulnerabilities of the cell." Without this term, the head would need to learn this alignment internally from concatenated features.

### 6.6 Evidential Regression Head — EvidentialRegressionHead

**Architecture:** `Linear(5D, D)` → GELU → `LayerNorm(D)` → `Linear(D, 4)`.

The four outputs are mapped to the Normal-Inverse-Gamma (NIG) parameters:
```
γ     = out[:, 0]                          # mean prediction (LN_IC50)
ν     = softplus(out[:, 1]) + 0.1          # degrees of freedom (evidence)
α     = softplus(out[:, 2]) + 1.0 + 0.01  # shape parameter (α > 1)
β     = softplus(out[:, 3]) + 0.01         # rate parameter
```

The `ν > 0` floor of 0.1 (vs. ε=1e-2) is critical: low `ν` can game the regularisation loss by making the evidence term small, evading the penalty for wrong predictions while reporting high uncertainty. The floor prevents this.

**Derived quantities:**
```
µ          = γ              (point estimate)
σ²_aleatoric = β / (α - 1)   (irreducible noise variance)
σ²_epistemic = β / (ν(α - 1)) (model uncertainty)
```

**Numerical stability under AMP:** The evidential head explicitly disables float16 computation:
```python
with torch.amp.autocast(device_type=z.device.type, enabled=False):
    out = out.float()
    # ... NIG parameterisation in fp32
```
The product `ν * (α - 1)` in the epistemic formula can underflow to 0 in fp16 when both factors are small (early training), causing epistemic uncertainty to diverge to `+inf`. Forcing fp32 prevents this.

---

## 7. Foundation Model Encoders (Phase 4)

The standard GATv2 + PathwaySetEncoder is PathXDRP v0. Phase 4 adds richer encoders as alternative branches, ablated systematically.

### 7.1 GeneMamba Cell Encoder

**Reference:** "GeneMamba: Efficient and Effective Foundation Model on Single Cell Data" (2025, arXiv 2504.16956). Pretrained on 30 million cells with a bidirectional Mamba SSM backbone (24 layers, 512-dim, 65.7M parameters). Available at `mineself2016/GeneMamba` on HuggingFace.

**Why:** The PathwaySetEncoder sees each gene as a single scalar (its Z-score); it cannot model gene-gene coregulation within or across pathways. GeneMamba was trained to model single-cell transcriptomics autoregressively — it has learned which genes covary, which are tissue-specific master regulators, and which represent cell state transitions.

**Integration with PathXDRP:**

1. **Rank-based top-K selection:** for each cell line, select the `top_k=2048` most highly expressed genes (by Z-score). GeneMamba expects a ranked token sequence, not a fixed-length vector.

2. **Gene → backbone token ID mapping:** GeneMamba has a gene symbol vocabulary. A one-time mapping matches HGNC symbols from the DepMap expression matrix to GeneMamba's vocabulary indices.

3. **GeneMamba backbone (frozen):** processes the top-K gene tokens → (B, K, 512) gene embeddings.

4. **Adapter:** `Linear(512, D)` → GELU → `LayerNorm(D)`. Trainable.

5. **Pathway pooling:** group top-K gene embeddings by their pathway membership (same KEGG map as PathwaySetEncoder) → mean pool per pathway → (B, 370, D). This maintains the same pathway-token interface as PathwaySetEncoder, making it a drop-in replacement.

**Fallback:** if the HuggingFace checkpoint is unavailable (offline environment, e.g. Kaggle without internet), a lightweight from-scratch Bi-Mamba stack (4 layers, 256-dim) is instantiated. It has the same architecture family and can be trained from scratch without the pretrained benefit, useful for ablations.

**Requirement:** `mamba-ssm` (CUDA-only, ~25 min to compile on Kaggle T4).

### 7.2 Graph-Mamba Drug Encoder

**Reference:** "Graph-Mamba: Towards Long-Range Graph Sequence Modeling with Selective State Spaces" (Wang et al., 2024, arXiv 2402.00789); MOL-Mamba (AAAI 2025) for the GAT + Mamba hybrid pattern.

**Architecture:** local GATv2 message-passing → atom sequence ordering → bidirectional Mamba over atom sequence → unpad.

**Why Mamba over atoms:** standard GATv2 has receptive field limited by message-passing depth. A 4-layer GAT can see at most 4 hops away from any atom. Mamba's SSM operates over the full sequence, enabling long-range interactions between atoms at opposite ends of a large molecule (e.g. a macrocycle or a bifunctional drug).

**Atom ordering:** Mamba is a sequential model, so atom order matters. Three ordering strategies are supported:
- `degree` (default): sort atoms descending by graph degree within each molecule. High-degree atoms (hubs) are processed first — analogous to Graph-Mamba's prioritisation heuristic. Stable and biologically motivated (hub atoms like ring junctions are often pharmacophore centers).
- `canonical`: preserve PyG's default ordering (canonical SMILES atom order).
- `random`: shuffle each forward pass (ablation only — not recommended for deployment).

**Steps:**
1. Input projection + GATv2 stack (same as DrugGATEncoder, default 2 layers).
2. Sort atoms by chosen ordering within each batch element.
3. `to_dense_batch()` → (B, max_n, D) + padding mask.
4. Bidirectional Mamba blocks: forward Mamba + reverse Mamba + gated combination + residual + LayerNorm.
5. Unpad and invert the permutation.
6. `global_mean_pool` → h_mol (B, D).

**Key difference from DrugGATEncoder:** h_mol is (B, D) not (B, 2D) (single pool, not mean+max). The 5D head formula adjusts accordingly.

### 7.3 MolFormer-XL Drug Encoder

MolFormer-XL (IBM Research, *Nat. Mach. Intell.* 2022) is a large language model pretrained on 1.1 billion SMILES strings. It produces a 768-dimensional CLS embedding representing the global molecular chemistry.

**Integration:** frozen MolFormer CLS embedding is projected `Linear(768, D)` + GELU and broadcast-added to all atom embeddings after the GATv2 stack. This injects global molecular context (pharmacophore class, drug-likeness, scaffold type) into each atom's local representation.

**Trade-off:** adds ~15 ms per batch (tokenisation + frozen forward pass), justified by richer molecular representation.

### 7.4 scGPT Cell Encoder

scGPT (Cui et al., *Nat. Methods* 2024) is a Transformer-based foundation model pretrained on 33 million single cells for generative single-cell biology. Its gene embeddings encode cell-state context learned from a vastly larger and more diverse dataset than GDSC.

**Integration:** same top-K → backbone → adapter → pathway-pool pipeline as GeneMamba, but using scGPT's Transformer backbone. Requires `flash_attn` and CUDA; a lightweight from-scratch Transformer fallback is provided for environments without scGPT.

---

## 8. Baseline Models

All three baselines are re-implemented from scratch with identical data loading, loss functions, and evaluation code to ensure fair comparison. They are trained on the exact same splits.

### 8.1 GraphDRP

**Reference:** Nguyen et al., "GraphDRP: predicting drug response in cancer cell lines using graph convolutional networks," *Bioinformatics* 2021.

**Drug encoder:** 5-layer **GIN** (Graph Isomorphism Network, Xu et al. 2019) with BatchNorm and ReLU, global mean-pooling to a (B, D) drug vector. GIN is theoretically as expressive as the Weisfeiler-Lehman graph isomorphism test.

**Cell encoder:** 3-layer **1D-CNN** over the expression vector treated as a 1D signal:
- `Conv1d(1, 32, k=8, s=4)` → `Conv1d(32, 64, k=8, s=4)` → `Conv1d(64, 128, k=8, s=4)` → `AdaptiveMaxPool1d(64)`.
- Flattens to 8,192 → `Linear(8192, 2D)` → `Linear(2D, D)`.

No pathway structure, no cross-attention, no uncertainty.

**Head:** `Linear(2D, 512)` → ReLU → Dropout → `Linear(512, 256)` → ReLU → Dropout → `Linear(256, 1)`.

**Loss:** MSE.

**Published baseline PCC (random split):** ~0.9363.

### 8.2 DRPreter

**Reference:** Shin et al., "DRPreter: Interpretable Anticancer Drug Response Prediction Using Knowledge-guided Graph Neural Networks and Transformer," *IJMS* 2023.

**Key architectural difference from PathXDRP:** DRPreter uses the same KEGG pathway grouping for the cell encoder (via PathwaySetEncoder) and uses cross-attention between drug atoms and pathway tokens, but the cross-attention has **no KEGG pathway mask** — it is unmasked standard multi-head attention. Attention weights are not entropy-regularised.

**Drug encoder:** GATv2 (same architecture as PathXDRP drug encoder), but returns `global_add_pool` (not mean+max).

**Cell encoder:** PathwaySetEncoder (same code as PathXDRP's), shared for a fair comparison.

**Cross-attention:** standard `nn.MultiheadAttention` without any biological mask. Drug atoms as Query, pathway tokens as Key/Value.

**Head:** same 3-layer MLP as GraphDRP, but on [h_drug_context || h_cell_global] (2D input, not 5D).

**Loss:** MSE.

**Published baseline PCC (random split):** ~0.922.

### 8.3 CDRScan

**Reference:** Chang et al., "Cancer Drug Response Profile scan (CDRscan)," *Scientific Reports* 2018.

The canonical "fingerprint + expression MLP" baseline. No graph, no pathway structure, no attention. Serves as a lower-bound reference.

**Drug representation:** Morgan circular fingerprint (radius=2, 2048 bits, RDKit `GetMorganFingerprintAsBitVect`). Binary vector encoding the presence of specific chemical substructures within a 2-bond radius of each atom.

**Drug encoder:** `Linear(2048, D)` → ReLU → Dropout → `Linear(D, D)` → ReLU → Dropout → `Linear(D, D//2)`.

**Cell encoder:** `Linear(n_genes, 2D)` → ReLU → Dropout → `Linear(2D, D)` → ReLU → Dropout → `Linear(D, D//2)`.

**Head:** `Linear(D, 256)` → ReLU → Dropout → `Linear(256, 128)` → ReLU → Dropout → `Linear(128, 1)`.

**Loss:** MSE.

---

## 9. Loss Function and Training

### 9.1 Evidential NIG-NLL Loss

The Normal-Inverse-Gamma (NIG) negative log-likelihood for a single sample:

```
NIG-NLL(γ, ν, α, β | y) =
  ½ log(π/ν)
  − α log(2β(1+ν))
  + (α + ½) log(ν(y−γ)² + 2β(1+ν))
  + log Γ(α)
  − log Γ(α + ½)
```

where `Γ` is the gamma function. This is the marginal likelihood of observing `y` under the NIG prior, integrating out the unknown mean and variance. Minimising it simultaneously optimises the point estimate `γ` (closer to `y`) and the uncertainty parameters `(ν, α, β)` (consistent with the observed residuals).

**Evidence regularisation term:**
```
L_reg = |y − γ| · (2ν + α)
```
Without this term, the model can achieve a lower NIG-NLL by setting `ν` and `α` to extreme values (making the distribution very peaked or very flat). The regularisation penalises high evidence (large `2ν + α`) on wrong predictions — if `|y − γ|` is large, the model should not be confident. The weight `λ` controls the strength of this penalty.

**Total loss:**
```
L = NIG-NLL + λ · L_reg + ε_reg · L_entropy
```
where `L_entropy` is the attention entropy regularisation from cross-attention (§6.4).

### 9.2 Lambda Warmup Schedule

The critical training insight is that `λ` should be ramped slowly, not set fixed from epoch 1.

**Why this matters:** early in training, the model makes large errors (`|y − γ|` is large). Combined with a large `λ`, the evidence regularisation `λ · |y − γ| · (2ν + α)` creates a gradient that pushes `ν` and `α` down (discouraging confidence when wrong). This conflicts with the NIG-NLL gradient which is trying to increase confidence on correct predictions. The net effect is a large counter-gradient that suppresses learning of accurate predictions.

**Solution:** ramp `λ` from 0 → target over the first 50 epochs:
```python
current_lam = target_lam * min(epoch / 50, 1.0)
model.evidential_lam = current_lam  # updated before each epoch
```
During the first 50 epochs, the model learns purely from the NIG-NLL (getting the point estimate right). After epoch 50, the regularisation gradually introduces calibration pressure. This is why the default `target_lam` is 0.01 (not the commonly used 0.1 from the original Amini paper) — with the warmup, the effective pressure is already reduced.

### 9.3 Optimizer and Learning Rate Schedule

**Optimizer:** AdamW with `lr=1e-3`, `weight_decay=1e-4`. The weight decay acts as L2 regularisation on the model parameters, preventing overfitting given the ~1.8M parameter model trained on 120K training samples.

**LR schedule:** 5-epoch linear warmup followed by cosine annealing:
- Warmup: `lr` scales linearly from `0.1 × 1e-3` to `1e-3` over 5 epochs. Prevents large gradient steps early when the newly initialised `cell_pool_q`, `h_mol` interaction weights, and head weights have random large norms.
- Cosine: `lr` follows a half-cosine from `1e-3` to 0 over the remaining epochs.

**Gradient clipping:** `clip_grad_norm_(model.parameters(), max_norm=1.0)`. Prevents gradient explosion, especially important with the evidential head which can produce large gradients when the prediction is far off target.

**Automatic Mixed Precision (AMP):** `torch.amp.GradScaler` + `torch.amp.autocast` enabled on CUDA. The evidential head and loss computation are explicitly forced to fp32 via `autocast(enabled=False)` to avoid fp16 instability in the NIG parameterisation.

**Checkpointing:** best model by validation PCC is saved. At test time, the best checkpoint is restored.

### 9.4 Hyperparameter Defaults

| Hyperparameter | Default | Notes |
|---------------|---------|-------|
| `hidden_dim` | 256 | Model width |
| `n_gat_layers` | 4 | Drug encoder depth |
| `n_attn_heads` | 8 | Cross-attention heads (32-dim per head) |
| `dropout` | 0.1 | Applied after projections |
| `mask_type` | soft | Learned per-pathway bias |
| `n_pw_transformer_layers` | 1 | Cross-pathway Transformer depth |
| `evidential_lam` | 0.01 | Final NIG regularisation weight |
| `lam_warmup_epochs` | 50 | Linear ramp of λ |
| `entropy_reg_weight` | 0.01 | Attention entropy penalty |
| `batch_size` | 256 | |
| `epochs` | 150 | |
| `lr` | 1e-3 | Peak LR |

---

## 10. Evaluation Protocol

### 10.1 Five Split Regimes

Each regime is run with 5 random seeds × 5 folds = 25 runs per model. Results are reported as mean ± std across seeds.

**Random split:** standard 5-fold cross-validation over all 150,459 (drug, cell) pairs. Train, val, and test sets share both drugs and cell lines. This is the weakest test but required for literature comparability. Most published DRP numbers are on random splits.

**Cell-blind split:** 5-fold `GroupKFold` over `COSMIC_ID`. All IC50 measurements for a given cell line are either entirely in train or entirely in test — never both. Tests whether the model generalises to **unseen cell lines**. Biologically equivalent to the question: "given a new patient's tumour expression profile, can the model predict drug sensitivity?" This is the most clinically relevant evaluation regime.

**Drug-blind split:** 5-fold `GroupKFold` over `DRUG_ID`. All measurements for a given drug are either entirely in train or entirely in test. Tests whether the model generalises to **unseen drugs**. Answers: "given a new candidate compound, can the model predict its cancer activity profile?"

**Scaffold-blind split:** Bemis-Murcko scaffold clustering of the drug molecules (RDKit `MurckoScaffold.MurckoScaffoldSmiles`). Drugs sharing a scaffold (common core ring system) are kept together in train or test. Harder than drug-blind because drugs with related scaffolds are also excluded. The gold standard for molecular generalisation (used in MoleculeNet benchmark).

**Tissue-blind split:** leave-one-cancer-type-out. The top-5 most represented tissues by cell count serve as test sets. Tests whether the model trained on, say, lung cancer cell lines can generalise to breast cancer cell lines without retraining.

**Baseline performance pattern:** random >> tissue-blind > cell-blind >> drug-blind > scaffold-blind. The gap between random and cell-blind is the key metric of clinical relevance — a model that collapses dramatically from PCC=0.93 (random) to PCC=0.55 (cell-blind) has not learned drug-cell biology, only cell-line identity.

### 10.2 Regression Metrics

All metrics are computed on LN(IC50) values (continuous regression):

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **RMSE** | √(mean((ŷ−y)²)) | Absolute error in log-µM units |
| **MAE** | mean(|ŷ−y|) | Median-robust absolute error |
| **PCC** (Pearson r) | Corr(ŷ, y) | Linear correlation; most commonly reported in literature |
| **Spearman ρ** | Corr(rank(ŷ), rank(y)) | Rank correlation; robust to outliers |
| **R²** | 1 − SS_res/SS_tot | Variance explained; interpretable as fraction of variance captured |
| **Per-drug PCC** | mean_d(Corr(ŷ_d, y_d)) | Pearson r computed per drug, then averaged; tests within-drug cell-rank ordering |
| **Per-cell PCC** | mean_c(Corr(ŷ_c, y_c)) | Pearson r per cell, then averaged; tests within-cell drug-rank ordering |

**Primary metric:** Pearson r (PCC) for comparability with literature. Per-drug PCC is the most clinically relevant — it tests whether the model correctly ranks drugs by efficacy for a specific cell line.

### 10.3 Uncertainty Metrics (C3)

**Expected Calibration Error (ECE):** sort predictions by predicted uncertainty into 15 equal-frequency bins. In each bin, compare empirical RMSE against mean predicted σ. Well-calibrated model: empirical error ≈ predicted σ in each bin. ECE = weighted mean absolute difference.

**Risk-coverage curve:** rank predictions by ascending epistemic uncertainty (most confident first). At each coverage level θ (fraction of predictions retained), compute RMSE on the retained θ fraction. A well-calibrated uncertainty model shows a monotonically increasing risk as coverage decreases (removing uncertain predictions reduces RMSE).

**Selective RMSE at fixed coverages:** RMSE at 50%, 70%, 90%, 100% coverage. Published as a table row for easy comparison.

**OOD AUROC:** treat random-split test predictions as in-distribution and cell-blind / drug-blind test predictions as OOD. Use epistemic uncertainty as the OOD score. AUROC measures how well epistemic uncertainty distinguishes in-distribution from out-of-distribution predictions.

### 10.4 Anti-Leakage Protocol

1. **Split before feature transformation.** All normalisation parameters (gene expression mean/std, FG vocabulary) are computed on the training fold only and applied to val/test.

2. **No pathway mask from labels.** The KEGG/STRING pathway mask is built from biological databases only, never from GDSC IC50 values. This ensures the mask doesn't encode the drug-response signal.

3. **MoA benchmark isolation.** The 25 drugs in the XAI benchmark are not used to inform any training decisions. Their ground-truth target genes are revealed only at evaluation time, never during training or model selection.

4. **Pre-registered evaluation.** The evaluation script and metric definitions are committed to git before any external validation runs, preventing post-hoc choice of favourable metrics.

5. **Seed reproducibility.** All randomness is controlled via explicit seeding: Python `random.seed`, NumPy `random.seed`, `torch.manual_seed`, `torch.cuda.manual_seed_all`.

---

## 11. Uncertainty Quantification

### 11.1 Normal-Inverse-Gamma Parameterisation

PathXDRP uses **Deep Evidential Regression** (Amini et al., NeurIPS 2020) to produce a full posterior distribution over the mean and variance of LN(IC50) in a single forward pass.

The prediction assumes:
```
y ~ Normal(µ, σ²)
µ ~ Normal(γ, σ²/ν)
σ² ~ InverseGamma(α, β)
```

This Normal-Inverse-Gamma prior is conjugate to the Gaussian likelihood, making the posterior analytically tractable. The four head outputs (γ, ν, α, β) parameterise this distribution.

**Aleatoric uncertainty** σ²_a = β/(α−1): the irreducible variance due to biological noise, measurement error, and fundamental unpredictability of drug response given only expression data. This is the variance of the Gaussian likelihood — it tells you how spread the IC50 distribution is for drugs structurally similar to this one in cells transcriptionally similar to this one.

**Epistemic uncertainty** σ²_e = β/(ν(α−1)): the model's uncertainty due to limited training data. High epistemic uncertainty means the model hasn't seen enough examples like this (drug, cell) pair to make a confident prediction. This is the clinically useful quantity for selective prediction — if σ²_e is high, we should flag the prediction for experimental validation rather than acting on it directly.

### 11.2 Deep Ensemble for Uncertainty Estimation

In Phase 5, five independently trained PathXDRP models (seeds 0–4) form a **deep ensemble**. The ensemble mean and variance provide a second, independent uncertainty estimate:
```
µ_ensemble = mean({µ_i})
σ²_ensemble = mean({σ²_i}) + var({µ_i})
```
The first term is the average aleatoric uncertainty; the second term is the disagreement between models (epistemic). This provides a calibration cross-check against the single-model evidential head.

### 11.3 Selective Prediction

In a drug screening context, a clinical decision support system can choose to abstain on low-confidence predictions rather than flag every case. PathXDRP supports this via selective prediction: given a coverage budget θ (e.g., "show predictions for 80% of drug-cell pairs"), select the θ fraction with lowest epistemic uncertainty and only report those.

Target performance: ≥25% RMSE reduction at 80% coverage vs. full-coverage prediction, matching the threshold demonstrated in a 2026 bioRxiv benchmark of uncertainty methods in DRP.

---

## 12. Quantitative Explainability Benchmark

### 12.1 Motivation

Prior DRP explainability work produces qualitative saliency maps (atom coloured by importance, gene ranked by gradient magnitude) and shows case studies for well-known drugs like erlotinib. There is no systematic evaluation of whether these explanations are correct — whether the atoms highlighted for erlotinib correspond to the aniline-quinazoline EGFR warhead, and whether the genes ranked highly correspond to EGFR pathway members.

PathXDRP's XAI benchmark fills this gap by defining a quantitative evaluation framework and running six methods against ground-truth biological annotations.

### 12.2 Six Attribution Methods

| Method | Type | Cost |
|--------|------|------|
| **Attention rollout** | Architectural (PathXDRP only) | Free — cached from forward pass |
| **Integrated Gradients** | Gradient-based | 50 forward-backward passes |
| **GNNExplainer** | Mask-optimisation | ~100 forward-backward passes per sample |
| **PGExplainer** | Parametric (amortised) | Train once, fast inference |
| **SubgraphX** | MCTS | ~1000 evaluations per sample (slow) |
| **SHAP / DeepLIFT** | Gradient-based (dense head) | ~10 forward passes |

**Attention rollout:** PathXDRP's cross-attention weights `attn_weights` (N_atoms, P) directly encode which drug atoms attend to which cell pathways. Rolling these up over attention heads (mean over H) gives a free, architecturally grounded explanation.

**Integrated Gradients (Captum):** computes the path integral of gradients from a baseline (zero atom features) to the actual input. Gives (N_atoms, 335) atom-feature attributions and (B, n_genes) expression attributions.

**GNNExplainer (PyG):** jointly optimises masks on edges and node features to find a minimal subgraph that explains the prediction. Computationally expensive but produces a compact molecular subgraph explanation.

**PGExplainer (PyG):** trains a parametric network that predicts edge masks from graph structure. Amortised — trained once, inference is fast. Useful for producing explanations at scale.

### 12.3 Six Evaluation Metrics

**Target-AUROC:** for a given drug, the model should assign high attribution to genes that are the known targets (or direct interactors) of that drug. AUROC measures how well the attribution scores (summed over atoms attending to pathways containing each gene) rank known target genes above random. For erlotinib, EGFR (and ERBB2, ERBB3 as secondary targets) should receive the highest gene attributions.

**Pathway hit-rate@k:** what fraction of the top-k most attended pathways contain the drug's known primary KEGG pathway? For erlotinib (EGFR signalling), the EGFR signalling pathway should appear in the top-k attended pathways.

**Faithfulness (Sufficiency):** remove the low-attribution atoms/genes (the model says these don't matter). Measure the delta in predicted LN(IC50). A faithful explanation means removing unimportant features barely changes the prediction — small delta.

**Faithfulness (Comprehensiveness):** remove the high-attribution atoms/genes (the model says these are critical). A faithful explanation means the prediction changes substantially — large delta.

**Stability:** run the same explanation method on the same (drug, cell) pair across 5 different random seeds. Compute cosine similarity between attribution maps. A robust explanation should be consistent across random seeds; if maps differ wildly, the explanation is unreliable.

**Sparsity:** entropy of the attribution distribution over atoms (or genes). Sparse explanations (few high-attribution features) are more interpretable. Penalises diffuse "everything is equally important" explanations.

### 12.4 Case Studies for the Paper

Three drugs serve as illustrative case studies:

1. **Erlotinib (EGFR inhibitor):** the aniline-quinazoline warhead is the known EGFR-binding pharmacophore. A correct explanation highlights these atoms. The relevant pathways are EGFR signalling (hsa04012), Ras signalling (hsa04014), PI3K-AKT (hsa04151). Sensitive cell lines typically have EGFR amplification or ERBB2 overexpression.

2. **Dabrafenib (BRAF V600E inhibitor):** the diaryl-urea + pyrimidine scaffold is the BRAF-binding pharmacophore. Relevant pathways: MAPK (hsa04010), Melanoma (hsa05218). Sensitive cell lines are melanoma-derived with BRAF V600E mutation.

3. **Olaparib (PARP1/2 inhibitor):** the phthalazinone warhead is the nicotinamide-mimetic PARP trap. Relevant pathways: DNA repair (hsa03440), p53 signalling (hsa04115). Sensitive cell lines have BRCA1/2 loss-of-function or other HR deficiencies.

For each, a 2D molecular saliency overlay (RDKit `SimilarityMaps`) annotates atoms with attention weights, and a KEGG pathway diagram highlights the top-attended pathways.

---

## 13. External Validation

### 13.1 CCLE / DepMap PRISM

**Purpose:** zero-shot transfer — test PathXDRP trained on GDSC2 on an entirely independent pharmacological screen.

**Data:** PRISM secondary screen (Broad Institute, DepMap) screens ~4,518 compounds across ~900 cancer cell lines. Expression data from the same DepMap 24Q4 release.

**Challenge:** different drug vocabulary (only ~80 drugs overlap GDSC2 and PRISM directly; most overlap by matching PubChem CID). Need to harmonise by canonical SMILES, then restrict evaluation to the overlapping set.

**Metric:** zero-shot Pearson r on the overlapping drug-cell pairs. A model that has genuinely learned drug-cell biology (not memorised GDSC2 patterns) should transfer.

### 13.2 TCGA (Patient Data)

**Purpose:** test whether cell-line training translates to patient response.

**Challenge:** TCGA RNA-seq profiles are fundamentally different from cell-line RNA-seq — different preprocessing, different normalisation, different biological context (tumour microenvironment, stromal contamination). Also, clinical response is typically binary (responder/non-responder) or ordinal (RECIST criteria), not a continuous IC50.

**Approach:** for treatment-annotated TCGA cohorts (e.g., breast cancer / tamoxifen responders; lung cancer / EGFR inhibitor responders), predict drug sensitivity using PathXDRP with TCGA RNA-seq as cell input. Evaluate as **AUROC for responder classification** rather than regression. This is a weaker signal than GDSC regression but clinically more meaningful.

### 13.3 NIBR PDXE

**Purpose:** patient-derived xenograft (PDX) models as a pre-clinical validation bridge.

**Content:** ~1,000 PDX models across several tumour types, screened against a small panel of targeted agents. RNA-seq available.

**Use:** same zero-shot regression evaluation as CCLE. PDX models are considered closer to patient biology than cell lines, making this a strong test of clinical transferability.

---

## 14. Implementation Details

### 14.1 Software Stack

| Component | Library | Version |
|-----------|---------|---------|
| Deep learning | PyTorch | 2.6.0+cu124 |
| Graph neural networks | PyTorch Geometric | 2.6.1 |
| Molecular processing | RDKit | 2024.09 |
| Tabular data | pandas | ≥2.0 |
| Statistics | scipy, scikit-learn | — |
| SSM (optional) | mamba-ssm + causal-conv1d | ≥2.2, ≥1.4 |
| Foundation models | transformers (HuggingFace) | ≥4.40 |
| XAI | captum | ≥0.7 |
| Progress bars | tqdm | — |

### 14.2 Hardware Requirements

**Minimum:** any NVIDIA GPU with ≥8 GB VRAM (e.g. RTX 3060). Training 150 epochs takes ~7.5 hours on this hardware.

**Recommended:** RTX 3090 / 4090 (24 GB VRAM). Allows batch_size=512 and higher hidden_dim.

**Mamba encoders:** require NVIDIA CUDA ≥11.6. mamba-ssm compiles C++/CUDA extensions during installation (~25 minutes on a modern GPU). CPU-only is not supported for Mamba.

**Kaggle:** T4 GPU (16 GB VRAM) runs the standard (non-Mamba) model comfortably. P100 also works. A Kaggle notebook (`notebooks/pathxdrp_kaggle.ipynb`) is provided that auto-downloads the expression matrix and installs all dependencies.

### 14.3 Key Engineering Decisions

**Windows DataLoader:** `num_workers=0` on Windows (the multiprocessing pickle mechanism deadlocks with PyG Data objects on `win32`). Set to 4 on Linux (including Kaggle).

**Memory-efficient cross-attention:** the naive `K[atom_batch]` expansion to per-atom K/V tensors creates O(N_atoms × P × D) memory — ~980 MB for a batch of 256 at 370 pathways and D=128. The padded-batch approach reduces this to O(B × max_n × D × n_heads) ≈ 60 MB. See §6.4.

**AMP with evidential head:** GradScaler + autocast is used for training speed, but the evidential parameterisation and loss computation are forced to fp32. The key failure mode is `ν * (α-1)` underflowing to 0 in fp16 when both factors are small (early training), causing epistemic uncertainty to diverge. Wrapping these operations in `autocast(enabled=False)` prevents this without sacrificing the speed benefit of mixed precision elsewhere.

**Softmax NaN prevention:** padding positions in cross-attention use `-1e4` (not `-inf`) as the mask value. `-inf` causes softmax to output 0 for masked positions but NaN for rows where all positions are masked (e.g., an empty batch slot), and these NaNs propagate through `0 * NaN = NaN` in the backward pass, corrupting subsequent batches. `-1e4` gives softmax values of `exp(-10000)` ≈ 0 without the NaN pathology.

**FG vocabulary is training-set-specific:** Morgan functional group bits are indexed by their frequency in the training SMILES. The vocabulary is built once per fold and applied to all splits. This is a subtle source of leakage if not handled correctly — the vocabulary build is always done on training SMILES only.

### 14.4 Module Organisation

```
D:\Masters\GNN Drug Discovery\
├── GDSC2-dataset.csv                       # IC50 response data
├── Compounds-annotation.csv                # Drug metadata + targets
├── Cell_Lines_Details.xlsx                 # Cell line tissue/cancer type
├── gene_identifiers_20241212.csv           # COSMIC gene census (entrez → HGNC)
│
├── data/
│   ├── raw/
│   │   └── OmicsExpressionProteinCodingGenesTPMLogp1.csv  # DepMap 24Q4 RNA-seq
│   └── processed/
│       ├── drugs_with_smiles.parquet       # SMILES for 247 drugs
│       ├── cosmic_to_depmap.csv            # COSMIC_ID → DepMap ModelID mapping
│       ├── pathway_gene_map.json           # {pathway_name: [gene_symbols]}
│       └── splits/                         # 5 regimes × 5 seeds × 5 folds
│
├── pathxdrp/
│   ├── data/
│   │   ├── fetch_smiles.py                 # PubChem REST API SMILES fetcher
│   │   ├── graph_utils.py                  # SMILES → PyG Data (atom/bond features)
│   │   ├── loader.py                       # load_expression(), build_master_df()
│   │   └── splits.py                       # 5 split regime builders
│   ├── models/
│   │   ├── drug_encoder.py                 # DrugGATEncoder (GATv2)
│   │   ├── cell_encoder.py                 # PathwaySetEncoder (stats + Transformer)
│   │   ├── cross_attention.py              # PathwayMaskedCrossAttention (padded-batch)
│   │   ├── evidential_head.py              # NIG head + evidential_loss()
│   │   ├── pathxdrp.py                     # End-to-end PathXDRP model
│   │   ├── graph_mamba_drug.py             # Graph-Mamba drug encoder (CUDA)
│   │   ├── mamba_cell.py                   # GeneMamba cell encoder (CUDA)
│   │   └── scgpt_cell.py                   # scGPT cell encoder (CUDA)
│   ├── baselines/
│   │   ├── graphdrp.py                     # GIN drug + 1D-CNN cell + MSE
│   │   ├── drpreter.py                     # GATv2 drug + unmasked pathway cross-attn + MSE
│   │   └── cdrscan.py                      # Morgan FP + expression MLP + MSE
│   ├── eval/
│   │   └── metrics.py                      # RMSE, PCC, Spearman, R², ECE, risk-coverage
│   ├── explain/
│   │   └── benchmark.py                    # XAI benchmark: 4 methods × 6 metrics
│   └── train.py                            # PathXDRP training entry point (CLI)
│
├── scripts/
│   ├── download_expression.py              # Figshare download (507 MB)
│   ├── build_pathway_mask.py               # KEGG REST → pathway_gene_map.json
│   ├── build_splits.py                     # Build all 5 × 5 × 5 splits
│   ├── train_baseline.py                   # Baseline training (graphdrp|drpreter|cdrscan)
│   ├── run_sweep.py                        # Full sweep (all models × splits × seeds)
│   ├── ensemble_eval.py                    # Aggregate 5-seed ensemble predictions
│   ├── external_validation.py              # CCLE / TCGA transfer evaluation
│   ├── run_xai_benchmark.py                # End-to-end XAI runner
│   ├── build_moa_benchmark.py              # Build moa_benchmark.json from drug panel
│   └── benchmark_inference.py             # Latency benchmark (ONNX / INT8)
│
├── tests/
│   └── test_smoke.py                       # 8 smoke tests (all passing)
│
└── notebooks/
    └── pathxdrp_kaggle.ipynb               # Self-contained Kaggle training notebook
```

### 14.5 Running Order

```bash
# 1. Download expression matrix (~507 MB, one-time)
python scripts/download_expression.py

# 2. Fetch SMILES from PubChem (one-time, ~5 min)
python -m pathxdrp.data.fetch_smiles

# 3. Build KEGG pathway map (one-time, ~2 min, requires internet)
python scripts/build_pathway_mask.py

# 4. Build all splits (one-time, ~3 min)
python scripts/build_splits.py

# 5. Train PathXDRP (main result)
python -m pathxdrp.train \
    --split random --seed 0 --fold 0 \
    --hidden_dim 256 --n_gat_layers 4 --epochs 150

# 6. Train with GeneMamba cell encoder (requires mamba-ssm + CUDA)
python -m pathxdrp.train \
    --split random --seed 0 --fold 0 \
    --cell_encoder_type gene_mamba \
    --gnm_backbone_id mineself2016/GeneMamba

# 7. Train baselines
python scripts/train_baseline.py --model graphdrp  --split random --seed 0 --fold 0
python scripts/train_baseline.py --model drpreter   --split random --seed 0 --fold 0
python scripts/train_baseline.py --model cdrscan    --split random --seed 0 --fold 0

# 8. Full sweep (25 runs per model × 4 models = 100 runs)
python scripts/run_sweep.py

# 9. Aggregate results
python eval/analyze_results.py --latex --out results/table1.tex
```

---

## Summary of Novelty

PathXDRP is not another drug-response GNN. It is an architectural thesis about the integration of biological structure into the attention mechanism:

1. **Cross-attention is constrained by biology, not free.** Atoms attend over pathways, not genes. The 37,950 KEGG gene-pathway memberships define what is mechanistically meaningful, and the model is rewarded for respecting them via entropy regularisation.

2. **Interpretability is structural, not post-hoc.** The attention map is not extracted after training and reverse-engineered; it is the mechanism of the model. This makes the XAI benchmark meaningful — we are evaluating whether the model's internal computation aligns with known biology.

3. **Uncertainty is a first-class output.** A single forward pass returns four NIG parameters, giving both point estimate and aleatoric/epistemic decomposition. This enables selective prediction and clinical risk stratification in a way that MSE-trained baselines cannot.

4. **Foundation models are integrated with biological structure.** GeneMamba and Graph-Mamba bring 65.7M and ~1.2M pretrained parameters respectively, but their outputs are fed into the same KEGG-structured pathway pooling, not used as black-box embeddings. The pretraining knowledge is channelled through biological priors.

The result is a model where every prediction comes with an answer to three questions: *What is the predicted IC50? How uncertain is this prediction? Which drug atoms interact with which biological pathways to produce it?*
