# PathXDRP — Detailed Methodology Reference

> Use this document as a reference when building the architecture flowchart and
> when describing the model in derivative documents (slides, talks, supplements).
> Numbers and dimensions correspond to the **final** configuration used in all
> reported experiments (see `scripts/run_final_sweep.ps1`). This document is
> kept in sync with `manuscript/main.tex`; if the two disagree, the manuscript
> wins and this file should be updated.

---

## 0. What changed in the final version

PathXDRP went through three architectural revisions. The version reported in
the paper is **v4 (final)**, characterised by three transformer-engineering
choices that together make the cross-attention path load-bearing:

1. **Residual + LayerNorm** around the cross-attention output
   (`--cross_attn_residual`).
2. **No parallel GAT global-readout highway** into the head
   (`--drop_h_mol`).
3. **Attention-only auxiliary loss** that supervises a small MLP reading the
   post-attention drug representation alone (`--attn_aux_weight 0.3`).

The earlier "v3" architecture — five-block head input including `h_mol`,
max-attention-weighted atom pool, no residual — is preserved in code as the
default for backward-compatibility with old checkpoints but is **not** what
the paper reports. Sections 5, 6, and 8 below describe the final design; the
v3 details are kept as a side-note in §12 for ablation reproducibility.

---

## 1. High-Level Pipeline Overview

```
Input pair: (Drug SMILES, Cell-line expression vector)
        │                            │
        ▼                            ▼
[Drug Graph Construction]    [Expression Z-scoring]
        │                            │
        ▼                            ▼
[DrugGATEncoder]             [PathwaySetEncoder]
  → atom embeddings H_drug     → pathway tokens P
  → global mol embedding h_mol   (370 × 128)
  → (+ Morgan FP projected to 128)
        │                            │
        └──────────┬─────────────────┘
                   ▼
     [PathwayMaskedCrossAttention]
          Drug atoms (Q) attend over
          Cell pathways (K, V)
          → out_proj(H'_drug) + H_drug → LN  (residual + LayerNorm)
                   │
                   ▼
        [Mean-pool over atoms]
          h_drug_context = mean_i(LN_i)             (B, 128)
          h_cell_global  = softmax(q · P) · P       (B, 128)
                   │
                   ▼
        [Concatenation → z vector — no h_mol highway]
          z = [h_drug_context ‖ h_cell_global ‖ h_drug_context ⊙ h_cell_global]
          Dimension: 3 × 128 = 384
                   │
                   ▼
        [EvidentialRegressionHead]
          → γ (predicted LN_IC50)
          → ν, α, β (NIG uncertainty params)
          → aleatoric variance = β / (α − 1)
          → epistemic variance = β / (ν(α − 1))
                   │
                   ├──> [Attention-only aux head]
                   │      g_aux(h_drug_context) → ŷ_aux
                   │      L_aux = MSE(ŷ_aux, y) × 0.3
                   │
                   └──> [AUC aux head]
                          σ(MLP(z)) → AUC_pred
                          L_auc = MSE(AUC_pred, AUC) × 0.2
```

Three loss terms are summed in training:
```
L_total = L_NIG-NLL(γ, ν, α, β; y)
         + entropy_reg_weight × L_entropy           (default 0.01)
         + attn_aux_weight    × L_attn-aux          (default 0.3)
         + aux_auc_weight     × L_auc               (default 0.2)
```

---

## 2. Input Representation

### 2.1 Drug Input

- **Source:** Canonical SMILES fetched from PubChem via PUG-REST API.
- **Parsing:** RDKit → atom-bond graph.
- **Coverage:** 247 of 295 GDSC2 drugs (48 biologics/peptides excluded — no
  resolved SMILES).

#### Atom Features — 334 dimensions total

| Feature block | Encoding | Dim |
|---|---|---|
| Atom type | One-hot over 43 elements + "other" | 44 |
| Degree | Buckets 0–10 | 11 |
| Formal charge | Buckets −3 to +3 | 7 |
| Hybridisation | sp, sp², sp³, sp³d, sp³d², other | 6 |
| Aromaticity | Binary flag | 1 |
| In-ring | Binary flag | 1 |
| Chirality | CW / CCW / unspecified | 3 |
| Implicit H count | Buckets 0–4 | 5 |
| Morgan FG fragments | Top-256 radius-2 functional groups (vocab built over all 247 SMILES) | 256 |
| **Total** | | **334** |

#### Bond (Edge) Features — 10 dimensions total

| Feature | Encoding | Dim |
|---|---|---|
| Bond type | single / double / triple / aromatic / none | 5 |
| Conjugation | Binary | 1 |
| In-ring | Binary | 1 |
| Stereo | 3-class | 3 |
| **Total** | | **10** |

#### Global Morgan Fingerprint (enabled in final config)

- 2048-bit radius-2 Morgan fingerprint (whole molecule).
- Projected: `Linear(2048 → 128) → GELU → LayerNorm → Dropout`.
- Concatenated to the final global drug embedding `h_mol` *inside the encoder*.
- With `--use_morgan_fp` (final default): `h_mol_dim = 256 + 128 = 384`.

> Note: `h_mol` is computed and exposed by the drug encoder for backward
> compatibility, but in the final architecture (`--drop_h_mol`) it is **not**
> concatenated into the head input. It is still used by the AUC auxiliary head
> when that is enabled, since the AUC head reads the same `z` as the main
> evidential head.

### 2.2 Cell-Line Input

- **Source:** DepMap 24Q4 RNA-seq
  (`OmicsExpressionProteinCodingGenesTPMLogp1.csv`).
- **Values:** log₂(TPM + 1) for 19,193 protein-coding genes across 1,673
  DepMap models.
- **Coverage:** 697 cell lines after mapping DepMap ModelIDs → GDSC COSMIC_IDs.
- **Normalisation:** Z-score per gene across the 697 cell lines (mean 0, std 1
  per column). Zero-variance genes receive a Z-score of 0.

---

## 3. Drug Encoder: DrugGATEncoder

**Purpose:** Molecular graph → per-atom embeddings (Query for cross-attention)
+ global drug summary.

```
Node features (334-dim)
        │
[Linear(334 → 128)]          ← node_proj
        │
[GATv2Conv × 4 layers]       ← each: 128 → 128, 8 heads, concat=True
  + LayerNorm + GELU + Dropout(0.1)  per layer
  + Residual connection
        │
  H_drug: (N_atoms, 128)     ← per-atom embeddings (Query for cross-attention)
        │
[global_mean_pool(H_drug)]   → h_mean: (B, 128)
[global_max_pool(H_drug)]    → h_max:  (B, 128)
        │
  h_mol = concat[h_mean ‖ h_max]  → (B, 256)

  + Morgan FP projection:
      fp_proj(morgan_fp_2048) → (B, 128)
      h_mol = concat[h_mol ‖ fp_proj] → (B, 384)
```

### GATv2Conv Details

- **Layers:** 4 (final config; was 3 in v3)
- **Heads:** 8
- **Head dim:** 128 / 8 = 16
- **concat=True** → output stays at 128 after each layer
- **edge_dim:** 10 (bond features fed into attention)
- Residual added after each layer (input/output dims match after the first projection)

### Output Dimensions

| Tensor | Shape | Description |
|---|---|---|
| `H_drug` | `(N_atoms_in_batch, 128)` | Per-atom embeddings — used as Query |
| `h_mol`  | `(B, 384)` (with Morgan FP) | Global drug summary — used only for AUC aux head in the final config |

---

## 4. Cell Encoder: PathwaySetEncoder

**Purpose:** Z-scored expression vector (19,193 genes) → pathway token matrix
(370 × 128).

### Pathway Map

- **Source:** KEGG human pathway-to-gene mappings via KEGG REST API.
- **Coverage:** 370 pathways, 37,950 (pathway, gene) pairs, 8,403 unique genes
  (43.8% of 19,193).
- **Storage:** `data/processed/pathway_gene_map.json`.

### Processing Steps (final config: `n_pw_stats = 4`)

```
Expression vector x: (B, 19193)   ← Z-scored log₂(TPM+1)
        │
For each pathway p (370 total) with gene set G_p:
  μ_p  = mean(x[G_p])           ← pathway mean
  σ_p  = std(x[G_p])            ← intra-pw std dev
  f_p  = mean(x[G_p] > 0)       ← fraction active
  m_p  = max(x[G_p])            ← max expression (driver-gene signal)

stats matrix: (B, 370, 4)
        │
[Linear(4 → 128)]                  ← shared per-pathway projection
  + LayerNorm + GELU
        │
  tokens: (B, 370, 128)
        │
[Transformer Encoder × 2 layers]   ← cross-pathway self-attention
  • n_heads = max(1, 128//32) = 4
  • Pre-LN (LayerNorm before attention)
  • FFN dim = 2 × 128 = 256
  • Dropout 0.0 inside the transformer
        │
  P: (B, 370, 128)                 ← final pathway embeddings
```

### Why Max Statistic?

- Mean-pooling dilutes the signal of a single highly-expressed oncogene (e.g.
  BRAF, EGFR) across all genes in the pathway.
- For targeted therapies, a single driver gene's expression is often the key
  sensitivity predictor.
- Implemented via `scatter_reduce_(reduce='amax')` — requires PyTorch ≥ 2.0.

### Why Two Transformer Layers?

- One layer captures first-order pathway co-activation (e.g. MAPK and PI3K
  both active).
- Two layers allow second-order interactions: PI3K → AKT → mTOR cascade needs
  at least two message-passing steps between pathway tokens.

---

## 5. Pathway-Masked Cross-Attention: PathwayMaskedCrossAttention

**Purpose:** Drug atom embeddings attend over cell pathway embeddings, fusing
molecular and transcriptomic information **and producing the saliency map that
the XAI benchmark scores**.

### Memory-Efficient Padded-Batch Formulation

Naive batched cross-attention with a PyG graph batch (variable N_atoms per
molecule) would expand K, V to shape `(N_total_atoms, 370, 128)` → ~1960 MB
at batch 256. PathXDRP uses `to_dense_batch` to produce `(B, N_max, 128)`
with a padding mask, then computes attention in `(B, h, N_max, 370)`.

**Memory comparison at batch=256, hidden=128, 370 pathways:**

| Tensor | Naive expansion | Padded-batch |
|---|---|---|
| Keys K | ~980 MB | ~30 MB |
| Values V | ~980 MB | ~30 MB |
| Attention scores | ~760 MB | ~24 MB |
| **Total** | **~2,720 MB** | **~84 MB** |

> Padding positions are masked from softmax with a finite large-negative
> value (−10⁴) rather than −∞, which avoids the `0 × NaN = NaN` cascade that
> the latter produces under mixed-precision backprop.

### Attention Computation

```
Q = H_drug · W_Q        (N_atoms, d_h)    ← drug atoms as queries
K = P     · W_K         (370, d_h)         ← pathways as keys
V = P     · W_V         (370, d_h)         ← pathways as values

A_ij = (Q_i · K_j^T) / sqrt(d_h) + M_ij

H'_drug = softmax(A) @ V                  (N_atoms, 128)
```

Where `d_h = 128 / 8 = 16` per head.

### Mask Options (`--mask_type`)

| Mask type | M_ij value | Description |
|---|---|---|
| `none` **(final default)** | 0 everywhere | Standard cross-attention; specialisation comes from the head (next subsection), not from a hand-set prior |
| `soft` | learnable per-pathway scalar (1, 1, 1, 370) added to logits | Up-/down-weights pathways globally; on our data collapses to a drug-independent prior |
| `hard` | −10⁴ on (atom, pathway) pairs where the pathway has no annotated drug-target gene | Leaks target labels into the model; reported only as an upper-bound diagnostic |

The paper's title contains "pathway-masked cross-attention" because the
mechanism is architecturally available and is what makes the saliency map
biologically interpretable; the *default trained model* runs `mask_type=none`
and the specialisation to pathways is enforced by the head redesign (§6).

### Entropy Regularisation

An auxiliary loss penalises pathways with near-uniform attention
(no focus):

```
L_entropy = −mean_i [ sum_j softmax(A_ij) · log softmax(A_ij) ]
L_total += entropy_reg_weight × L_entropy
```

`entropy_reg_weight = 0.01` by default.

### Residual + LayerNorm (final default: `--cross_attn_residual`)

The cross-attention block ends with the standard transformer pattern:

```
H'_drug ← LayerNorm( out_proj(softmax(A) V) + H_drug )
```

Without this residual, gradient flow to the attention weights is diffuse and
the head can ignore the attention path entirely. With it, every gradient step
on the prediction loss touches every attention weight — which is what makes
the attention map load-bearing for XAI (faithfulness comprehensiveness rises
roughly 6× against the no-residual v3 baseline at a cost of a few thousandths
of random-split PCC).

### Output

- `H'_drug`: `(N_atoms, 128)` — pathway-context-updated atom embeddings (after
  residual + LayerNorm).
- `attn_weights`: `(N_atoms, 370)` — exported for XAI scoring.
- `_entropy_loss`: scalar — available for total loss computation.

---

## 6. Interaction Vector: z (final architecture)

After cross-attention, the model pools and concatenates features into a
**3D interaction vector** (no h_mol highway):

```
h_drug_context = mean_pool(H'_drug)     (B, 128)   ← attended drug summary
h_cell_global  = softmax(q · P) · P     (B, 128)   ← learnable-query cell pool
h_int          = h_drug_context ⊙ h_cell_global   (B, 128)   ← bilinear

z = concat[ h_drug_context ‖ h_cell_global ‖ h_int ]      → (B, 384)
```

The bilinear (element-wise product) term encodes drug–cell interaction
explicitly. The `|h_drug_context − h_cell|` difference term used in v3 is
dropped in the final config; the difference signal is absorbed by the
attention path via the residual.

### Why no h_mol in z

In the v3 architecture the head input was a five-block concatenation that
included `h_mol` (the GAT global readout). The head learned to route the
prediction signal around the attention path through `h_mol`, leaving attention
as a parallel dead branch (`Δf` on top-K atom removal < 0.03 LN_IC50).
Dropping `h_mol` from the head input forces all drug information through the
cross-attention path. The cost is at most a few thousandths of Pearson
correlation; the benefit is a roughly 3× improvement in attention
faithfulness comprehensiveness.

### Cell-side pooling (learnable query)

A single trainable query vector `q ∈ R^{1×1×128}` (initialised N(0, 0.02))
attends over the 370 pathway tokens:

```
scores  = (q · P^T) / sqrt(128)              (B, 1, 370)
weights = softmax(scores, dim=-1)             (B, 1, 370)
h_cell_global = weights @ P                  (B, 128)
```

Learns which pathways are globally most informative across all drugs.

---

## 7. Evidential Regression Head: EvidentialRegressionHead

**Purpose:** Predict LN_IC50 with calibrated aleatoric and epistemic
uncertainty.

```
z: (B, 384)
        │
[Linear(384 → 128)]
[GELU]
[LayerNorm(128)]
[Linear(128 → 4)]       ← outputs: [raw_γ, raw_ν, raw_α, raw_β]
        │
  (computed in fp32 under autocast)
  γ = raw_γ                             ← predicted mean (LN_IC50 estimate)
  ν = softplus(raw_ν) + 0.1             ← evidence of mean (floor 0.1)
  α = softplus(raw_α) + 1.0 + 0.01      ← shape param (floor > 1)
  β = softplus(raw_β) + 0.01            ← rate param (floor > 0)
```

### Output Quantities

| Quantity | Formula | Interpretation |
|---|---|---|
| Predicted mean | `μ = γ` | Point estimate of LN_IC50 |
| Aleatoric variance | `σ²_al = β / (α − 1)` | Irreducible noise in measurements |
| Epistemic variance | `σ²_ep = β / (ν(α − 1))` | Model uncertainty (reducible with more data) |

`γ, ν, α, β` are forced to fp32 inside the head to prevent underflow in
`ν × (α − 1)` under bfloat16 autocast.

### Training Loss

```
L_NIG = 0.5 · log(π/ν)
       − α · log(2β(1+ν))
       + (α + 0.5) · log(ν(y−γ)² + 2β(1+ν))
       + lgamma(α) − lgamma(α + 0.5)

L_reg  = |y − γ| · (2ν + α)

L      = L_NIG + λ(t) · L_reg

λ(t)   = lam_target · min(1.0, t / lam_warmup_epochs)
         lam_target        = 0.01
         lam_warmup_epochs = 20  (final config)
         → epoch 0:  λ ≈ 0    (pure NLL, MSE-like)
         → epoch 20: λ = 0.01 (full evidential regularisation)
```

### Auxiliary heads

```
1. Attention-only aux head  (attn_aux_weight = 0.3, final default)

   g_aux(h_drug_context) → ŷ_aux                # 2-layer MLP, 128 → 64 → 1
   L_attn-aux = MSE(ŷ_aux, y)
   L_total += attn_aux_weight × L_attn-aux

   Rationale: forces h_drug_context (the post-attention drug representation)
   to be predictive of LN_IC50 on its own. Together with the residual and the
   dropped h_mol, this is what closes the "attention is a dead branch"
   failure mode.

2. AUC aux head  (aux_auc_weight = 0.2, final default)

   AUC_pred = σ(MLP(z))    # 2-layer MLP, 384 → 32 → 1, sigmoid
   L_auc    = MSE(AUC_pred, AUC_target)
   L_total += aux_auc_weight × L_auc

   Rationale: LN_IC50 and AUC are correlated but distinct drug-response
   metrics. Joint training regularises the shared drug–cell representation
   and reduces overfitting to IC50-specific noise, particularly on the
   drug-blind split.
```

---

## 8. Training Procedure

### Optimizer & Schedule (final config)

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 5 × 10⁻⁴ (sqrt-scaled from 1e-3 at batch 256) |
| Weight decay | 1 × 10⁻⁴ |
| Batch size | 64 |
| Gradient clip | ‖∇‖₂ = 1.0 |
| Max epochs | 50 |
| Early stopping patience | 10 epochs (on validation PCC) |
| LR warmup | 10 epochs linear ramp from 10% to full LR |
| LR decay | Cosine decay after warmup |
| Mixed precision | bfloat16 (fp32 exponent range; evidential head forced fp32) |
| Lambda warmup epochs | 20 |
| Entropy reg weight | 0.01 |
| Attention-aux weight | 0.3 |
| AUC-aux weight | 0.2 |

### Training Loop (per batch)

```
1. Load batch: drug_graph, expr (Z-scored), y (LN_IC50), auc (optional), morgan_fp
2. Move to GPU
3. Forward pass under bfloat16 autocast:
     out = model(drug_graph, expr, y, auc, morgan_fp)
4. loss = out["loss"]
5. Backprop + gradient clip + AdamW step
6. Accumulate loss for epoch average
```

### Validation & Checkpointing

- Evaluate on val split every epoch: compute PCC, RMSE, Spearman, R².
- Save checkpoint if `val_PCC > best_val_PCC`.
- Early-stop if no improvement for `patience` epochs.
- **Final test evaluation** uses the best checkpoint, not the last epoch.

---

## 9. Data Splits

Five generalisation regimes, each with 5 seeds × fold 0:

| Split | What is held out | Train rows | Val rows | Test rows |
|---|---|---|---|---|
| `random`        | Random (drug, cell) pairs       | ~120,367 | ~15,046 | ~15,046 |
| `cell_blind`    | Entire cell lines (COSMIC_ID)   | ~120,356 | ~15,051 | ~15,052 |
| `drug_blind`    | Entire drugs (DRUG_ID)          | ~120,395 | ~15,032 | ~15,032 |
| `scaffold_blind`| Bemis–Murcko scaffolds          | ~119,967 | ~15,246 | ~15,246 |
| `tissue_blind`  | Top-5 tissues by row count      | ~138,407 | ~6,026  | ~6,026  |

**Key design notes:**
- `cell_blind`: `GroupKFold` on COSMIC_ID — zero cell overlap train/test.
- `drug_blind`: `GroupKFold` on DRUG_ID — zero drug overlap train/test.
- `scaffold_blind`: harder than drug-blind — chemically similar compounds
  cluster together; a test drug's scaffold has never been seen in training.
- `tissue_blind`: held-out tissue is fixed per fold (not random) — seed only
  shuffles val/test assignment within held-out tissue, so variance across
  seeds reflects model-init variance only.
- **Zero leakage verified** across all 125 split files.

---

## 10. Evaluation Metrics

### Point prediction

| Metric | Formula | Notes |
|---|---|---|
| PCC | Pearson correlation(ŷ, y) | Primary metric |
| RMSE | √mean((ŷ − y)²) | Primary metric |
| Spearman ρ | Rank correlation | Robust to outliers |
| R² | 1 − SS_res / SS_tot | Coefficient of determination |
| Per-drug PCC | mean over drugs of PCC(ŷ_d, y_d) | Within-drug ranking quality |
| Per-cell PCC | mean over cells of PCC(ŷ_c, y_c) | Within-cell ranking quality |

### Uncertainty

| Metric | Description |
|---|---|
| ECE (regression) | Bin predictions by predicted σ; compare empirical RMSE per bin vs predicted σ |
| Sel-RMSE@k% | RMSE on the k% most-confident predictions (sorted by ascending epistemic σ); k ∈ {50, 70, 90, 100} |
| Risk-coverage curve | Sel-RMSE as a function of coverage — should be monotonically increasing |

**Headline:** Sel-RMSE@50% = 0.772 vs Sel-RMSE@100% = 1.018 on the random
split (5-seed mean) — a 24% RMSE reduction at 50% coverage.

### XAI

| Attribution method | Scope |
|---|---|
| Cross-attention | PathXDRP, DRPreter (architecturally exposed) |
| Integrated gradients on expression | All four models |
| Integrated gradients on atom features | PathXDRP (manual loop, not Captum — see §11) |
| Permutation importance | All four models |
| Occlusion | All four models |

| XAI metric | What it measures |
|---|---|
| Target-AUROC | Per-drug AUROC of gene attribution against resolved drug-target gene set |
| Recall@K (gene-set) | Fraction of resolved target genes contained in the gene-set union of the top-K attended pathways; K ∈ {5, 10, 20} |
| Sensitivity alignment | AUROC of `(pathway-mean expression · attribution)` against IC50-sensitive vs -resistant labels |
| Faithfulness sufficiency (↓ is better) | `|f(x) − f(mask_bottom-K%(x))|` — small means low-score features are truly unimportant |
| Faithfulness comprehensiveness (↑) | `|f(x) − f(mask_top-K%(x))|` — large means top-score features genuinely drive the prediction |
| Faithfulness curve AUC | Trapezoidal area of comprehensiveness over K ∈ {5, 10, 20, 30, 50}% — ROAR-style summary without retraining |
| Sparsity | Normalised entropy of the attribution distribution; lower is sparser (more interpretable) |

### Attention-collapse diagnostic (random / seed 0 only)

| Quantity | Threshold for "collapsed" |
|---|---|
| Mean drug-level attention entropy | `< 10% × log₂(370) ≈ 0.59 nats` |
| Within-atom cosine similarity | `> 0.95` |
| Cross-drug cosine similarity | `> 0.95` |

Reported PathXDRP values: entropy 31% of max (1.846 nats), within-atom cos
0.366, cross-drug cos 0.543 — **no collapse**.

---

## 11. Model Parameter Count (default final config)

| Component | Parameters |
|---|---|
| DrugGATEncoder (4 GATv2 layers, d=128) | ~0.9 M |
| PathwaySetEncoder (4-stat + 2-layer Transformer) | ~0.4 M |
| PathwayMaskedCrossAttention (8-head, d=128) | ~0.2 M |
| Cell-pool learnable query | trivial |
| Morgan FP projection | ~0.27 M |
| EvidentialRegressionHead | ~0.05 M |
| Attention-only aux head | ~0.01 M |
| AUC aux head (optional) | ~0.01 M |
| **Total trainable** | **~1.85 M** |

---

## 12. Notes for Reviewers / Reproducibility

### XAI implementation note

Captum's `IntegratedGradients` reshapes `drug_batch.x` to
`(n_steps × N_atoms, ...)` and desynchronises the
`drug_batch.batch` index, causing per-atom IG to silently return zeros for
every drug in a graph batch. PathXDRP replaces this with a **manual
midpoint Riemann-sum** over the (zero-baseline → input) path, which is
mathematically equivalent and produces non-zero gradients. The fix is in
`pathxdrp/explain/benchmark.py`.

### Fuzzy pathway resolver

GDSC's free-text `TARGET` field is resolved against KEGG pathway names using
a fuzzy string matcher with sentinel filters that drop labels such as
"Other"/"Unclassified" before matching. Drug-class shorthand is expanded
("DNA methyltransferases" → DNMT1/3A/3B). The 25-drug curated MoA panel
referenced in the supplements is the authoritative target set; the fuzzy
matcher is the fallback for drugs outside the panel and is reported only on
the 143 drugs with at least one resolved gene present in the DepMap matrix.

### v3 vs final architecture (for ablation reproducibility)

The v3 architecture is the model the v3 checkpoints in `checkpoints/pathxdrp`
were trained under. v3 differs from the final config in three places:

| Component | v3 | Final |
|---|---|---|
| Cross-attention residual + LayerNorm | off | on |
| Head input | `[h_drug_context ‖ h_mol ‖ h_cell_global ‖ h_drug_context ⊙ h_cell_global]` (5D = 640) | `[h_drug_context ‖ h_cell_global ‖ h_drug_context ⊙ h_cell_global]` (3D = 384) |
| Atom-to-molecule pool | max-attention-weighted (`global_add_pool(context × max_j A_ij)`) | plain mean pool |
| Attention-only auxiliary loss | not present | weight 0.3 |
| Lambda warmup epochs | 50 | 20 |

The v3 model achieves 0.919 ± 0.009 random-split PCC vs 0.930 ± 0.002 for the
final configuration and has attention faithfulness comprehensiveness ≈ 0.10
vs the final ≈ 0.60 (6× improvement).

---

## 13. Run Command — Final Configuration

```bash
python -u -m pathxdrp.train \
    --split                random \
    --seed                 0 \
    --fold                 0 \
    --epochs               50 \
    --early_stop_patience  10 \
    --batch_size           64 \
    --lr                   5e-4 \
    --hidden_dim           256 \
    --n_gat_layers         4 \
    --n_attn_heads         8 \
    --dropout              0.1 \
    --mask_type            none \
    --n_pw_transformer_layers 2 \
    --use_morgan_fp \
    --aux_auc_weight       0.2 \
    --cross_attn_residual \
    --drop_h_mol \
    --attn_aux_weight      0.3 \
    --evidential_lam       0.01 \
    --lam_warmup_epochs    20 \
    --precision            bf16 \
    2>&1 | tee logs/pathxdrp_final_random_seed0.log
```

The full sweep across 5 splits × 5 seeds is driven by:

```powershell
pwsh ./scripts/run_final_sweep.ps1
```

which writes results to `results/pathxdrp/<split>_seed<s>_fold0.json`,
re-generates the auto LaTeX tables at `manuscript/auto_tables.tex`, and
re-renders every figure in PNG and PDF.
