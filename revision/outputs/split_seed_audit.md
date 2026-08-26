# What the seed actually varies in each split regime

Extends Reviewer #5, point 6, from tissue-blind to all five regimes.

Mean pairwise Jaccard overlap of the TEST partition across the five seeds.
A value near 1.0 means the seeds share the same held-out set, so the
reported standard deviation is initialisation noise, not data variability.

| Split | Test rows | Row overlap | Cell-line overlap | Drug overlap |
|---|---|---|---|---|
| random | 15,046 | 0.110 | 0.998 | 0.989 |
| cell-blind | 15,052 | 0.333 | 1.000 | 1.000 |
| drug-blind | 15,032 | 0.333 | 0.999 | 1.000 |
| scaffold-blind | 14,909 | 0.058 | 0.999 | 0.122 |
| tissue-blind | 6,026 | 0.336 | 1.000 | 1.000 |

## What the seed varies, by the group the split holds out

| Split | Held-out unit | Overlap of held-out unit across seeds | Seed varies the held-out set? |
|---|---|---|---|
| random | rows | 0.110 | yes |
| cell-blind | cell lines | 1.000 | **no** |
| drug-blind | drugs | 1.000 | **no** |
| scaffold-blind | scaffolds (via drugs) | 0.122 | yes |
| tissue-blind | tissues (via cell lines) | 1.000 | **no** |

## Consequence for the manuscript

In **cell-blind, drug-blind, tissue-blind** the seed does not change which groups are held out. It only reshuffles the validation/test assignment inside a fixed held-out pool (mean row overlap 0.333, the value expected for two random halves of one pool) and re-initialises the model.

Therefore the `+/- std` columns of Table 4 quantify **initialisation and val/test-partition noise, not data variability**, for these regimes. The submitted manuscript states this for tissue-blind only (Section 3.3 and the Limitations). It is equally true of cell-blind and drug-blind and must be stated for all three.

This strengthens rather than weakens the reframing in W1: the error bars in Table 4 are narrower than true cross-dataset variability, so small between-model margins are even less meaningful than they look. It is also an independent argument for the leave-one-tissue-out protocol (W3b) and for the clustered bootstrap (W4).
