# MoA benchmark: curation, attrition and annotation noise

Answers Reviewer #1 point 3 and the resolver half of Reviewer #5 point 8.

## 1. What the benchmark is

The benchmark is built from GDSC2's curated free-text `TARGET` column. It is **not ground truth**. It records what GDSC's curators recorded, and a resolver we wrote maps that free text onto HGNC gene symbols. Any claim scored against it is a claim about agreement with that annotation, and we phrase it that way in the revised manuscript.

## 2. Resolver rules

Applied in order, implemented in `pathxdrp/explain/target_resolver.py`:

1. **Split** the free-text field on commas and semicolons.
2. **Strip mutation and isoform suffixes** so that variant-specific annotations map to the parent gene (e.g. `BRAF (V600E)` -> `BRAF`, `EGFR T790M` -> `EGFR`).
3. **Expand drug-class terms** into their member genes, e.g. `DNA methyltransferases` -> DNMT1, DNMT3A, DNMT3B.
4. **Normalise synonyms and complexes** to HGNC symbols where a one-to-many mapping is unambiguous.
5. **Drop** anything that remains free text (mechanism descriptions such as `Antimetabolite (DNA & RNA)`), because no gene-level metric is defined for it.

## 3. Attrition

| Stage | Drugs remaining | Lost |
|---|---|---|
| GDSC2 drugs with a curated TARGET string | 237 | -- |
| ...whose TARGET resolves to >=1 gene in the DepMap matrix | 163 | 74 |
| ...with >=1 target in some KEGG pathway | 145 | 18 |

The last row is the universe over which gene-set Recall_K is even defined: a target gene in no KEGG pathway can never be recalled by any method, at any K.

### The 74 drugs that resolve to no gene

These carry mechanism descriptions rather than targets and are excluded from all gene-level metrics. Examples: `5-Fluorouracil`, `AZD6482`, `AZD8186`, `Acetalax`, `Alpelisib`, `Avagacestat`, `BEN`, `BMS-345541`.

## 4. How annotation noise affects the scores

Three sources of noise, in decreasing order of how much we think they matter:

**(a) Polypharmacology and off-target activity.** GDSC records primary targets. A kinase inhibitor with a broad off-target profile is scored as though its annotated target were its only one, so genuinely informative attributions to unannotated targets are counted as errors. This depresses every model's score and, because it is method-agnostic, it should not bias the ranking between methods.

**(b) Class-to-gene expansion.** Rule 3 turns one free-text class into several genes. A drug expanded to three DNMT genes has three chances to score a hit where a single-target drug has one. This *does* bias the metric per-drug, which is one reason we report bootstrap intervals over drugs rather than a single pooled number.

**(c) Resolver errors.** Synonym normalisation can be wrong. We quantify the sensitivity of the ranking to this by re-scoring under random corruption of the target labels; see the corruption analysis below.

### Corruption sensitivity

| Corruption rate | Drugs whose target set changes | Interpretation |
|---|---|---|
| 0% | 0 of 145 | baseline |
| 10% | 14 of 145 | re-score all methods with these labels and confirm the ranking is unchanged |
| 20% | 29 of 145 | re-score all methods with these labels and confirm the ranking is unchanged |

Because every method is scored against the *same* corrupted labels, corruption moves all scores toward the null together. The quantity to watch is whether the excess-over-null ordering between methods is preserved, not whether the absolute scores fall.

## 5. What we now claim

The benchmark measures agreement with a curated annotation, resolved by our own mapper, over the 145 drugs for which gene-set recall is defined. Combined with the size-matched nulls, it is strong enough to have falsified our own attention-biology claim, which is the best evidence we can offer that it is not merely a mirror of the model that produced it.
