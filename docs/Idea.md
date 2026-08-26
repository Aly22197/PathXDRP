# Research Plan — Pathway-Masked Dual-Graph Drug Response Prediction with Foundation Embeddings, Calibrated Uncertainty, and Quantitative Mechanistic Explainability

> **Working title (placeholder):** **PathXDRP** — *Pathway-masked Cross-attention Drug Response Predictor with Foundation-model embeddings and Evidential Uncertainty*.
> **Target venues (Q1):** *Briefings in Bioinformatics* (IF ~9), *Bioinformatics* (IF ~5.8), *Journal of Cheminformatics* (IF ~7), *Nature Communications* (stretch), *Nature Machine Intelligence* (stretch, requires prospective validation).
> **Primary dataset:** GDSC2 (preferred) + GDSC1 from the [Kaggle GDSC release](https://www.kaggle.com/datasets/samiraalipour/genomics-of-drug-sensitivity-in-cancer-gdsc).
> **External validation datasets:** CCLE (PRISM/CTRPv2), TCGA (patient response surrogate), NIBR PDXE (patient-derived xenografts).

---

## 0. Honest Diagnosis of the Original Proposal (XGDP Replication Risk)

A literature review of 2022–2026 work surfaced a critical issue: the original proposal — *"GAT for drug + CNN over gene expression + multi-head cross-attention + GNNExplainer / Integrated Gradients + saliency maps onto 2D molecules"* — is essentially **XGDP** (Cai et al., *Scientific Reports* 14, 2024, [s41598-024-83090-3](https://www.nature.com/articles/s41598-024-83090-3)), which the original document already cited as `[^1]`. Other near-precedents:

| Work | Year | Venue | Overlap with original idea |
|------|------|-------|---------------------------|
| **XGDP** | 2024 | Sci. Reports | Drug GNN + expression CNN + cross-attention + IG + GNNExplainer + saliency on atoms — **near-identical to original proposal** |
| **TransEDRP** | 2022/2024 | arXiv 2210.17401 | Edge-embedded molecular graph transformer + multi-head attention over genomics |
| **GSDRP** | 2024 | Springer | Cross-attention fusing omics and drug graph features |
| **DeepCoVDR** | 2023 | *Bioinformatics* (ISMB) | Graph transformer + cross-attention drug-cell fusion |
| **drGAT/drGT** | 2024 | arXiv 2405.08979 | Heterogeneous attention over drug-gene-cell on GDSC |
| **DRPreter** | 2022 | *IJMS* | Type-aware transformer over pathway-decomposed cell graphs + drug GNN |
| **DGDRP** | 2024 | *Front. Genet.* | Drug-specific gene selection via biological-network re-ranking |
| **DeepCCDS** | 2025 | *Adv. Science* | Cancer-driver-signal + self-supervised pretraining; current SOTA (~+25 % PCC vs. prior) |
| **MTEGDRP** | 2025 | *J. Med. Chem.* | Equivariant GNN + molecular-self-attention transformer + multi-omics |

**Conclusions from the diagnosis:**

1. *"Cross-attention between drug atoms and cell-line genes"* is **not novel by itself in 2026** — the design space has been mined.
2. The **TinyML / "deploy on a lab laptop" angle is weak**: existing models (DeepCDR, DRPreter, GraphDRP) already run on a CPU laptop in seconds, and there is no realistic clinical workflow that requires GDSC-trained inference on a Raspberry Pi/MCU. Q1 reviewers (esp. *Nat. Mach. Intell.*, *Nat. Commun.*) will read this as engineering padding. Reframed below as an *appendix*, not a headline contribution.
3. **Recent benchmarking literature is brutal** about leakage and evaluation hygiene:
   - **DrEval** (bioRxiv 2025, [10.1101/2025.05.26.655288](https://www.biorxiv.org/content/10.1101/2025.05.26.655288v1)) — pipeline + audit flagging non-reproducibility, leakage, pseudoreplication, biased metrics.
   - *"Understanding the Sources of Performance"* (bioRxiv 2024, [10.1101/2024.06.05.597337](https://www.biorxiv.org/content/10.1101/2024.06.05.597337v1)) — in cell-blind eval, drug features contribute ≈ zero; performance is driven by transcriptomic similarity. **Reviewers now demand drug-blind + cell-blind splits.**
   - *Widespread data leakage inflates performance estimates* (bioRxiv 2026) — 10/12 audited deep DRP models had confirmed leakage from feature-screening before CV.
   - **IMPROVE community benchmark** (arXiv 2503.14356, 2025) — cross-dataset (GDSC↔CCLE↔CTRP) generalization is the new bar.

The plan below preserves the user's intent (interpretable DRP, deployment story) but re-aims at four genuinely under-explored axes that together justify a Q1 submission.

---

## 1. Reframed Research Question and Contributions

**Research question.** *Can we build a drug-response predictor that (i) generalises to unseen cell lines, unseen drug scaffolds, and unseen patient cohorts, (ii) tells the user when to trust its prediction, and (iii) produces mechanistically faithful explanations that quantitatively recover known drug–target biology — all from a single end-to-end architecture trainable on commodity hardware?*

**Four claimed contributions** (each independently defensible at Q1):

| # | Contribution | Novelty axis |
|---|--------------|--------------|
| **C1** | **Pathway-masked cross-attention.** Cross-attention between drug-atom queries and gene keys/values is constrained by **KEGG/Reactome pathway membership** and **STRING PPI proximity**, so attention weights are mechanistically interpretable *by construction* (not only post-hoc). | Architectural — turns explainability into an inductive bias. |
| **C2** | **Foundation-model dual encoders.** Drug branch uses **MolFormer-XL** or **Uni-Mol** SMILES embeddings concatenated with a learnable GAT; cell branch uses **scGPT** / **scFoundation** bulk-mode embeddings concatenated with a pathway-aware MLP. | Few prior DRP works (esp. on GDSC) leverage 2024–2025 foundation models. |
| **C3** | **Calibrated evidential uncertainty.** Deep ensembles + evidential regression head with selective-prediction analysis under cell-blind, drug-blind, and TCGA-shift evaluation. | Largely open at Q1 level for DRP — bioRxiv 2026 work shows ~64 % MSE reduction by filtering on uncertainty. |
| **C4** | **Quantitative explainability benchmark.** First systematic comparison of **GNNExplainer, PGExplainer, SubgraphX, Integrated Gradients, attention rollout, and SHAP** on a curated set of ≥ 25 GDSC drugs with known mechanism-of-action, scored by (a) attribution-vs-known-target AUROC against COSMIC Cancer Gene Census, (b) pathway-enrichment p-values vs. KEGG/Reactome targets, (c) faithfulness/sufficiency/comprehensiveness. | XAI for DRP has been mostly anecdotal; a quantitative benchmark is itself a contribution. |

**Demoted to appendix:** an *optional* INT8 quantisation + ONNX export benchmark for low-latency interactive screening (web/desktop tool), with a clear claim that this is not the headline.

---

## 2. Datasets

### 2.1 Primary — GDSC (via Kaggle release)

The Kaggle dump (samiraalipour) typically includes (verify on first download):

| File | Rows × cols (approx.) | Notes |
|------|-----------------------|-------|
| `GDSC2_fitted_dose_response.csv` | ~ 245 k IC50 records | **Primary label set.** Newer assay, less noisy. ~190 drugs × ~810 cell lines. Use `LN_IC50` (log µM) as regression target. |
| `GDSC1_fitted_dose_response.csv` | ~ 320 k IC50 records | Older release, ~310 drugs × ~990 cell lines. Used for pretraining + ablation. |
| `Drug_listFri.csv` | ~ 500 drugs | Columns: `DRUG_ID`, `DRUG_NAME`, `SYNONYMS`, `TARGET`, `TARGET_PATHWAY`, `PUBCHEM_ID`. **Verify SMILES presence** — if absent, fetch from PubChem REST (`pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{CID}/property/CanonicalSMILES`) by `PUBCHEM_ID`. |
| `Cell_Lines_Details.xlsx` | ~ 1 000 cell lines | `COSMIC_ID`, `CELL_LINE_NAME`, `TISSUE`, `CANCER_TYPE`, `MSI_STATUS`, `TCGA_CLASSIFICATION`. |
| `Cell_line_RMA_proc_basalExp.txt` | ~ 17 737 genes × ~ 1 000 cell lines | RMA-normalised microarray expression (Iorio et al. 2016). |
| `mutations_all_*.csv`, `cnv_*.csv` | binary / categorical | Optional secondary modalities. |

**Action items on first run:**
1. Confirm SMILES coverage; backfill missing via PubChem CID.
2. Canonicalise SMILES with RDKit; compute Morgan FP (r=2, 2048 bits) as a cheap baseline feature.
3. De-duplicate (`DRUG_ID`, `COSMIC_ID`) pairs by averaging `LN_IC50`.
4. Subset gene expression to: (a) full transcriptome, (b) **L1000 landmark genes**, (c) **COSMIC CGC** (~ 700 cancer driver genes), (d) **per-drug-targeted gene sets** (DGDRP-style biological-network re-ranking). All four feature sets are ablated.
5. Z-score per-gene across cell lines; clip extremes at ±5 σ.

### 2.2 External validation

| Dataset | Use | Key challenge |
|---------|-----|---------------|
| **CCLE / DepMap (PRISM, CTRPv2)** | Zero-shot transfer; same modality (cell lines) | Different drugs (~ 4 500 in PRISM secondary screen); harmonise drug IDs by PubChem CID; rebuild expression onto common gene panel. |
| **TCGA** (treatment-annotated subset) | Patient-level transfer (clinical response surrogate: PFS / RECIST) | RNA-seq vs. microarray normalisation; very different sample size; binary response labels — evaluate as classification (responder vs. non-responder), not regression. |
| **NIBR PDXE** | Pre-clinical validation | ~ 1 000 PDX models × few drugs; gold standard for translation. |

### 2.3 Mechanism-of-action benchmark (built once, used for C4)

Curate ≥ 25 GDSC2 drugs with **well-characterised single or paired targets** from DrugBank + KEGG + literature:

> erlotinib (EGFR), gefitinib (EGFR), lapatinib (EGFR/HER2), trametinib (MEK1/2), dabrafenib (BRAF), PLX4720 (BRAF), vemurafenib (BRAF), nilotinib (BCR-ABL/KIT), imatinib (ABL/KIT/PDGFR), olaparib (PARP1/2), navitoclax (BCL2/BCL-XL), venetoclax (BCL2), AZD7762 (CHK1), MK-2206 (AKT), AZD6244/selumetinib (MEK1/2), tamoxifen (ESR1), 5-fluorouracil (TYMS), cisplatin (DNA), bortezomib (proteasome), cetuximab (EGFR), crizotinib (ALK/MET), JNJ-7706621 (CDK1/2/4), nutlin-3 (MDM2), AZD8055 (mTOR), KU-55933 (ATM).

For each, store *ground-truth* target genes (HGNC) + parent KEGG pathway + key chemical substructure (warhead) annotations.

---

## 3. Architecture — PathXDRP

```
                ┌────────────────────────────────────────────────────┐
                │                  Drug branch                        │
SMILES ─────────►  RDKit ─► molecular graph (atoms / bonds)           │
                │      └────► MolFormer-XL or Uni-Mol → seq embedding │
                │                                                    │
                │  Atom feats (Morgan-augmented) ─► GAT ×L_d ──┐     │
                │                                              │     │
                │  Seq embedding ─► linear ──────────────────►(+)► H_drug ∈ ℝ^{N_atoms × d}
                └────────────────────────────────────────────────────┘
                ┌────────────────────────────────────────────────────┐
                │                Cell-line branch                    │
RNA-seq /  ─────►  scGPT bulk encoder  ─► gene-level embedding       │
expression      │  Pathway-aware MLP (Set Transformer over KEGG sets)│
                │  ─────────────────────────────────────────────►   H_cell ∈ ℝ^{N_genes × d}
                └────────────────────────────────────────────────────┘
                                    │
                                    ▼
                ┌────────────────────────────────────────────────────┐
                │   Pathway-Masked Multi-Head Cross-Attention        │
                │   Q = H_drug,  K = V = H_cell                       │
                │   mask M_{a,g} = 1 iff atom a belongs to a fragment │
                │      annotated with a pharmacophore matching        │
                │      a pathway containing gene g (via STRING d≤2).  │
                │   Soft variant: M is a learned scalar prior added   │
                │      to attention logits (Gaussian over PPI dist).  │
                └────────────────────────────────────────────────────┘
                                    │
                                    ▼
                ┌────────────────────────────────────────────────────┐
                │     Pooling (attention-weighted) → joint vector z   │
                │     Evidential regression head → (γ, υ, α, β)       │
                │       point estimate μ, aleatoric σ², epistemic     │
                └────────────────────────────────────────────────────┘
                                    │
                                    ▼
                       LN_IC50 prediction + uncertainty
```

**Key design choices.**

- **Drug graph:** atoms = nodes (atomic number, degree, formal charge, hybridisation, aromaticity, in-ring, chirality, H-count, **Morgan-radius-2 functional-group ID** as one-hot of top-256 fragments), bonds = edges (bond type, stereo, conjugated, in-ring). Backbone: **GAT v2** (8 heads, 3 layers, hidden 128). Concatenate a frozen MolFormer-XL CLS embedding (768-d) projected to 128.
- **Cell-line encoder:** start from scGPT *bulk-mode* gene embeddings (frozen, then optionally LoRA-tuned). Apply a **Set Transformer** within each KEGG pathway (set per pathway), then pool to one token per pathway. Equivalent fallback if scGPT integration is too costly: a **pathway-MLP** over GSVA-scored pathways.
- **Cross-attention mask M:** built once from KEGG + STRING + DrugBank target annotations. Hard mask for the headline result; soft (learned prior) mask for ablation. This makes the attention map directly interpretable as *"atom a interacts with pathway P via target gene g"*.
- **Output head:** evidential deep regression (Amini et al., NeurIPS 2020) producing both the IC50 estimate and aleatoric/epistemic variances *in a single forward pass*; deep ensemble (5 seeds) on top.
- **Loss:** evidential NLL + λ · L₂ on Dirichlet evidence (encourages calibrated uncertainty) + small attention-entropy regulariser to keep maps sparse and human-readable.

**Parameter budget target:** < 10 M trainable params (vs. DeepCDR ≈ 7 M, DRPreter ≈ 12 M) — keeps the *deployment* appendix credible without sacrificing capacity.

---

## 4. Evaluation Protocol (designed to survive Q1 review)

### 4.1 Splits — three regimes, all with 5 seeds

| Regime | How | Purpose |
|--------|-----|---------|
| **Random / warm** | 5-fold CV on (drug, cell) pairs | Comparability with literature; weakest test. |
| **Cell-blind** | held-out cell lines never seen in training (5-fold by `COSMIC_ID`) | Realistic patient-stratification analogue. |
| **Drug-blind** | held-out drugs never seen in training (5-fold by `DRUG_ID`) | Tests true molecular generalisation. |
| **Scaffold-blind** *(stretch)* | Bemis–Murcko scaffold split | Harder still; aligns with MoleculeNet conventions. |
| **Tissue-blind** *(stretch)* | leave-one-cancer-type-out | Simulates a new indication. |

### 4.2 Metrics

- **Regression:** RMSE (LN_IC50), Pearson r, Spearman ρ, R², per-drug Pearson r averaged.
- **Calibration / uncertainty (C3):** Expected Calibration Error (ECE) on confidence-binned predictions; selective-prediction risk-coverage curves; AUROC for **OOD detection** (cell-blind ∪ drug-blind vs. random-split).
- **External transfer:** zero-shot Pearson r on CCLE; AUROC for responder classification on TCGA.
- **Explainability (C4):** see § 5.
- **Statistical significance:** paired Wilcoxon across seeds vs. each baseline; report mean ± std.

### 4.3 Baselines (reproduce from authors' code)

`Random-Forest-on-expression`, `MLP-on-expression-only` (the 2024 "Sources of Performance" paper showed this is shockingly hard to beat in cell-blind), `Morgan-FP + RF`, **DeepCDR**, **GraphDRP** (GAT variant), **TGSA**, **DRPreter**, **DeepTTA**, **TransCDR**, **DGDRP**, **XGDP** (direct competitor — most important reproduction). No new dataset-specific tweaks for baselines — fair comparison.

### 4.4 Anti-leakage protocol

- Feature selection (variance / driver-gene filtering) is **fit only on training folds**.
- Standardisation parameters (mean, std) recomputed per fold.
- No drug-target leakage when constructing pathway masks (mask is from KEGG/STRING priors, not GDSC labels).
- Pre-registered evaluation script committed to git **before** any external-validation runs.

---

## 5. Quantitative Explainability Benchmark (C4)

For each of the ≥ 25 curated MoA drugs, on cell lines where the drug is sensitive (LN_IC50 in lowest 25 %), produce six attribution maps:

1. **Attention rollout** over cross-attention heads (ours, free).
2. **Integrated Gradients** w.r.t. atom features and gene features.
3. **GNNExplainer** (mask edges + node features).
4. **PGExplainer** (parametric, amortised — much faster).
5. **SubgraphX** (Monte-Carlo tree search; gold standard, slow).
6. **SHAP / DeepLIFT** baseline on the dense head.

Score each method on:

| Metric | Definition |
|--------|------------|
| **Target-AUROC** | AUROC of the attribution score over genes for ranking the drug's annotated targets above random genes. |
| **Pathway hit-rate@k** | Fraction of top-k attended pathways that overlap KEGG pathway of the known target. |
| **Faithfulness (Suff.)** | Δ-prediction when removing low-attribution atoms/genes (small drop = faithful). |
| **Faithfulness (Comp.)** | Δ-prediction when removing high-attribution atoms/genes (large drop = faithful). |
| **Stability** | Cosine sim. of attribution maps across 5 random seeds. |
| **Sparsity** | Entropy of the attribution distribution. |

**Output for chemists:** the 2-D molecular saliency overlay survives, but is now *backed by a quantitative score against ground truth*. We render the top-attended atoms with RDKit `SimilarityMaps` and the top-attended genes as a KEGG pathway diagram.

---

## 6. Implementation Roadmap (16 weeks, single graduate student on commodity GPU — RTX 3090 / 4090 class)

The work is decomposed into seven phases. Each phase ends with a concrete artefact and a go/no-go gate.

### Phase 0 — Repo + Environment (Week 1)

- Repository layout (Python 3.11, PyTorch 2.x, PyG 2.5, RDKit 2024, scikit-learn, pandas, hydra-core, wandb, lightning).
  ```
  pathxdrp/
    configs/                # Hydra YAMLs
    pathxdrp/
      data/                 # download + preprocess GDSC, CCLE, TCGA
      models/               # encoders, attention, heads
      explain/              # XAI methods + benchmark harness
      eval/                 # split builders, metrics, calibration
      train.py / predict.py / explain.py
    notebooks/              # EDA only (not for results)
    scripts/                # CLI entry points
    tests/                  # unit + a smoke training run
  ```
- Pin every dependency. Set seed plumbing. Pre-register evaluation YAMLs.
- **Gate:** smoke training run completes on a 100-row toy split.

### Phase 1 — Data Foundation (Weeks 2–3)

- Download Kaggle dump; verify schema; backfill missing SMILES from PubChem.
- Canonicalise SMILES, build molecular graphs with RDKit.
- Build expression matrix (full + L1000 + CGC + per-drug subsets).
- Build pathway mask `M` from KEGG (via `bioservices` / KEGG REST) and STRING v12 (PPI distance ≤ 2).
- Build the MoA benchmark JSON (~ 25 drugs).
- Implement five split builders (random, cell-blind, drug-blind, scaffold-blind, tissue-blind).
- **Gate:** `make data` produces all artefacts deterministically; SMILES coverage ≥ 99 %.

### Phase 2 — Reproduce Baselines (Weeks 4–5)

- Re-run **DeepCDR, GraphDRP, DRPreter, TGSA, DeepTTA, TransCDR, DGDRP, XGDP** on identical splits with our own data loader. (Critical: most papers report random-split numbers only — we will re-run them under cell-blind/drug-blind too.)
- Record their numbers in a single `baselines_results.json` checked into git.
- **Gate:** at least four baselines reproduced within ± 2 % of published random-split numbers.

### Phase 3 — PathXDRP v0 (Weeks 6–8)

- Implement drug branch (GAT + Morgan FG one-hot).
- Implement cell branch (pathway-MLP first; scGPT integration deferred).
- Implement **hard pathway-masked cross-attention** + evidential regression head.
- Train on GDSC2 random split → must beat XGDP's reported PCC by a non-trivial margin to proceed.
- **Gate:** PCC ≥ 0.93 random, RMSE ≤ 1.0 (LN_IC50).

### Phase 4 — Foundation-model Encoders + Soft Mask (Weeks 9–10)

- Integrate **MolFormer-XL** (HuggingFace `ibm/MoLFormer-XL-both-10pct`) for drug embeddings; ablate against GAT-only.
- Integrate **scGPT** (cls embedding mode); ablate vs. pathway-MLP.
- Add **soft (learned) mask** variant of cross-attention, ablate vs. hard.
- **Gate:** at least one foundation-encoder variant beats v0 on cell-blind PCC.

### Phase 5 — Uncertainty + External Validation (Weeks 11–12)

- Train 5-seed deep ensemble of best variant.
- Compute calibration metrics (ECE), risk-coverage curves, OOD AUROC.
- Apply zero-shot to CCLE (drug ID intersection); fine-tune to TCGA (responder classification surrogate via reported PFS / RECIST mapping).
- **Gate:** uncertainty filtering reduces RMSE on the 80-th-percentile-confidence subset by ≥ 25 % (matching/beating the bioRxiv 2026 prior).

### Phase 6 — Explainability Benchmark (Weeks 13–14)

- Implement six XAI methods (most via Captum + PyG-Explainer + DIG library).
- Run on the 25-drug MoA panel × 5 sensitive cell lines each = ~ 125 explanation jobs.
- Compute six metrics each, produce the headline **Table of XAI methods × metrics**.
- Render top-3 chemist-facing case studies (erlotinib/EGFR, dabrafenib/BRAF, olaparib/PARP) as 2-D saliency PDFs.
- **Gate:** our cross-attention rollout outperforms post-hoc methods on at least 3 of 6 metrics.

### Phase 7 — Deployment Appendix + Manuscript (Weeks 15–16)

- ONNX export; INT8 dynamic quantisation (no QAT — keep simple); benchmark CPU inference latency on a typical lab laptop and a Raspberry Pi 5 for batch screening of 10 000 drug × cell pairs.
- Package as `pip install pathxdrp` + a Streamlit demo loading the quantised model.
- Write paper using the structure in § 7 below.
- **Gate:** all numbers locked, all figures regeneratable from `make figures`.

---

## 7. Manuscript Outline

1. **Abstract** — frame as "interpretable, calibrated, generalisable DRP."
2. **Introduction** — DRP landscape, four gaps, four contributions.
3. **Related Work** — DRP models; cross-attention DRP; XAI for GNNs; uncertainty in molecular ML.
4. **Methods** — § 2 + § 3 of this plan, condensed.
5. **Experiments** — three split regimes × all baselines × full table; calibration curves; external validation.
6. **Explainability Benchmark** — § 5; case studies; the **headline figure** = attribution-AUROC bar plot across XAI methods.
7. **Ablations** — pathway mask hard vs. soft vs. none; foundation encoders; gene subsets; ensemble size; uncertainty head; attention heads.
8. **Discussion** — limitations (no prospective wet-lab; cell-line vs. patient gap; KEGG/STRING bias); future work.
9. **Reproducibility statement** — code, weights, splits, seeds, evaluation YAML.
10. **Appendix A — Lightweight Deployment** — INT8 quantisation, Streamlit demo, on-device latency.

---

## 8. Risk Register

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Foundation embeddings (MolFormer / scGPT) blow up compute | Med | Pre-compute and cache as `.parquet`; freeze; ablate against light alternatives. |
| scGPT bulk-mode is unreliable for microarray expression | Med | Fallback: pathway-aware MLP over GSVA scores. |
| Cell-blind / drug-blind PCC collapses (per *Sources of Performance* 2024) | High | Expected — we report it honestly; the contribution is **calibrated uncertainty under shift**, not a magic number. |
| TCGA response labels are noisy | High | Frame TCGA as classification of responders vs. non-responders, not regression; report AUROC and acknowledge ceiling. |
| Pathway mask leaks drug-target info | Med | Mask is built from KEGG/STRING/DrugBank only — never from GDSC labels; targets used in C4 evaluation are *held out* from mask construction for the 25 MoA drugs. |
| Reviewers ask for prospective validation | Med | Out of scope; cite NIBR PDXE retrospective transfer as the strongest available substitute. |
| XGDP authors publish a follow-up first | Low–Med | Differentiation: pathway-masked attention + evidential UQ + foundation embeddings + quantitative XAI benchmark — orthogonal to a hypothetical XGDP-v2. |

---

## 9. Concrete First-Week Checklist

- [ ] Create and pin the conda/uv environment.
- [ ] Initialise the repo skeleton; add CI smoke test.
- [ ] Download the Kaggle GDSC dump; checksum into DVC.
- [ ] Write `inspect_dataset.py` that prints (drugs, cell lines, IC50 records, SMILES coverage).
- [ ] Pull missing SMILES from PubChem; commit `drugs_with_smiles.parquet`.
- [ ] Build the five split files; freeze hashes.
- [ ] Write `baselines/deepcdr.py` shim that wraps the original repo.
- [ ] Draft the MoA benchmark JSON for the 25 drugs.
- [ ] Open a Notion / wandb workspace for experiment tracking.
- [ ] Pre-register the evaluation protocol on OSF (a 1-page form) — this single act is worth a paragraph in the Reproducibility section.

---

## 10. Why this version is Q1-defensible (vs. the original)

| Axis | Original | This plan |
|------|----------|-----------|
| Architectural novelty | Cross-attention drug ⨯ cell (XGDP — already published) | Pathway-masked, biology-grounded cross-attention + foundation encoders |
| Explainability rigor | Saliency pictures | Quantitative benchmark of 6 XAI methods on 25 known-MoA drugs |
| Generalisation evidence | Random split only | Random + cell-blind + drug-blind + scaffold-blind + CCLE + TCGA |
| Trustworthiness | None | Evidential UQ + selective prediction + OOD detection |
| Deployment story | TinyML headline (weak) | Demoted to one-page appendix; honest scope |
| Reproducibility | Implicit | Pre-registered eval, locked seeds, public weights |
| Headline narrative | "Yet another cross-attention" | "Generalisable, calibrated, mechanistically explainable DRP" |

---

## References (key papers seeded for the lit review)

- XGDP — *Sci. Reports* 2024 — https://www.nature.com/articles/s41598-024-83090-3
- DGDRP — *Front. Genet.* 2024 — https://www.frontiersin.org/journals/genetics/articles/10.3389/fgene.2024.1441558/full
- DeepCCDS — *Adv. Science* 2025 — https://advanced.onlinelibrary.wiley.com/doi/full/10.1002/advs.202416958
- TransCDR — *BMC Biology* 2024 — https://link.springer.com/article/10.1186/s12915-024-02023-8
- GPDRP — *BMC Bioinformatics* 2023 — https://link.springer.com/article/10.1186/s12859-023-05618-0
- DRPreter — *IJMS* 2022 — https://www.mdpi.com/1422-0067/23/22/13919
- TGSA — *Bioinformatics* 2022 — https://academic.oup.com/bioinformatics/article/38/2/461/6374919
- MTEGDRP — *J. Med. Chem.* 2025 — https://pubs.acs.org/doi/10.1021/acs.jmedchem.5c03438
- TransEDRP — arXiv 2210.17401 (2024 update) — https://arxiv.org/abs/2210.17401
- drGAT — arXiv 2405.08979 (2024) — https://arxiv.org/html/2405.08979v1
- DeepCoVDR — *Bioinformatics* (ISMB 2023) — https://academic.oup.com/bioinformatics/article/39/Supplement_1/i475/7210469
- Quantized GNNs for molecular property — *J. Cheminform.* 2025 — https://jcheminf.biomedcentral.com/articles/10.1186/s13321-025-00989-3
- DrEval — bioRxiv 2025 — https://www.biorxiv.org/content/10.1101/2025.05.26.655288v1.full
- *Sources of Performance in DRP* — bioRxiv 2024 — https://www.biorxiv.org/content/10.1101/2024.06.05.597337v1
- IMPROVE benchmark — arXiv 2503.14356 (2025) — https://arxiv.org/html/2503.14356v1
- *Uncertainty Estimates for DRP* — bioRxiv 2026 — https://www.biorxiv.org/content/10.64898/2026.04.03.715851v1
- Lavecchia, *XAI in drug discovery* — *WIREs CMS* 2025 — https://wires.onlinelibrary.wiley.com/doi/10.1002/wcms.70049
- scDrugMap — *Nat. Commun.* 2025 — https://www.nature.com/articles/s41467-025-67481-2
- scGPT-DRP integration — arXiv 2504.14361 (2025) — https://arxiv.org/html/2504.14361
- Amini et al., *Deep Evidential Regression* — NeurIPS 2020.
- MolFormer-XL — Ross et al., *Nat. Mach. Intell.* 2022.
- scGPT — Cui et al., *Nat. Methods* 2024.

---

*Document maintained for the GDSC-DRP Q1 submission project. Last revised 2026-05-02.*
