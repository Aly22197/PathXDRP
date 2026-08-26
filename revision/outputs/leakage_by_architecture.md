# The normalisation leak did not benefit all architectures equally

This is the most consequential result of the fold-wise re-run, and it was not
anticipated by the diagnostic.

## The measurement

Cell-blind split, seed 0, identical in every respect except the normalisation
scheme. Cohort-wide Z-scoring (as submitted) against fold-wise Z-scoring fitted
on the training cell lines only.

| Model | Cell encoder | cohort-wide | fold-wise | delta |
|---|---|---|---|---|
| PathXDRP | KEGG pathway statistics | 0.8622 | 0.8642 | **+0.0019** |
| DRPreter | KEGG pathway statistics | 0.8693 | 0.8676 | −0.0017 |
| CDRScan | MLP over expression, fingerprint-dominated | 0.8894 | 0.8889 | −0.0006 |
| **GraphDRP** | **1D-CNN over raw per-gene expression** | **0.8838** | **0.8674** | **−0.0164** |

GraphDRP loses an order of magnitude more than any other model. Its own
seed-to-seed standard deviation on this split is 0.0014, so the drop is roughly
twelve standard deviations. This is not noise.

## Why the effect is architecture-dependent

The leak is in the per-gene mean and standard deviation. Fitting them over the
full cohort lets each gene's scaling reflect the held-out cell lines.

A model that consumes the raw per-gene vector is exposed to all 19,193 of those
scalings directly. GraphDRP runs a 1D convolution straight over the expression
vector, so every gene's leaked scaling is an input feature. PathXDRP and DRPreter
first aggregate genes into KEGG pathway statistics (mean, standard deviation,
fraction over-expressed, maximum), which averages the per-gene scalings over
tens to hundreds of genes and largely cancels the effect. CDRScan's expression
MLP sits alongside a 2,048-bit Morgan fingerprint that carries much of its
signal, diluting the exposure.

So the leak preferentially inflated the architecture with the most direct
per-gene exposure -- and that architecture was the strongest baseline on this
split.

## Consequence for the comparison

On cell-blind the corrected ranking is materially different:

| | cohort-wide (as submitted) | fold-wise (corrected) |
|---|---|---|
| CDRScan | 0.8894 | 0.8889 |
| GraphDRP | **0.8838** | 0.8674 |
| DRPreter | 0.8693 | 0.8676 |
| PathXDRP | 0.8622 | 0.8642 |

GraphDRP's clear second place becomes a near-tie with DRPreter and PathXDRP.
The gap between GraphDRP and PathXDRP falls from 0.0216 to 0.0032.

We want to be careful about how much weight this carries. It is one seed per
model on one split, and the clustered bootstrap already showed that differences
of this size on cell-blind are not robust. The finding is not "PathXDRP is
better than GraphDRP on cell-blind"; it is that **part of the apparent
difference between them was an artefact of a preprocessing choice**, and that
the artefact was invisible until the preprocessing was corrected.

## Why this matters beyond our paper

Cohort-wide normalisation before splitting is common practice in this
literature. This result says the practice does not merely inflate scores
uniformly -- it inflates them *differentially by architecture*, favouring models
that consume raw high-dimensional features over models that aggregate them.
Any benchmark that normalises before splitting is therefore not just optimistic
but potentially mis-ranked, and the direction of the bias depends on the
encoders being compared.

That is a stronger reason to fix the leak than "the numbers are slightly too
high", and it is worth stating in the paper as a methodological point rather
than only as a correction to our own table.


---

# RETRACTION OF THE MECHANISM (2026-08-25)

The architectural explanation above does **not** survive the second split, and
the sections above are retained only as a record of what was initially claimed.

## The disconfirming evidence

Tissue-blind also holds out cell lines, so the same mechanism should apply:

| Model | cell-blind delta | tissue-blind delta |
|---|---|---|
| PathXDRP | +0.002 | +0.010 |
| GraphDRP | **-0.016** | **+0.001** |

If direct per-gene exposure were the cause, GraphDRP would be worst-affected on
both splits. It is essentially unaffected on tissue-blind.

## What stands and what does not

**Stands:** the correction is not a uniform rescaling; on cell-blind GraphDRP
moves by roughly twelve times its own seed-to-seed standard deviation; that
single change closes most of the GraphDRP-PathXDRP gap on that split.

**Withdrawn:** the claim that the effect tracks the cell encoder, and the
generalisation that normalising before splitting biases benchmarks
"differentially by architecture". One split is not a pattern.

**Unknown:** which property of a split or a model predicts the size of the
effect. Cell-blind holds out 139 cell lines and tissue-blind 56 of a single
lineage, so the two differ in both the amount and the structure of the
distribution shift; either could matter, and one seed per cell cannot separate
them.

## Note on how this happened

The mechanism was written up after cell-blind completed and before tissue-blind
ran. It was a plausible story fitted to one split, and it read convincingly,
which is precisely why it should not have been written before the replication
was available. The manuscript and the response letter now report the observation
without the explanation.
