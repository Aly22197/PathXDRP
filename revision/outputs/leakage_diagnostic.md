# W3 -- Normalisation leakage diagnostic

Answers Reviewer #3, point 3.

The submitted pipeline fitted per-gene Z-score moments over all 697 cell
lines before splitting. Leakage is possible only for test cell lines that
do not also appear in the training partition -- for every other test row
the moments were legitimately estimable from training data.

## Where leakage can occur

| Split | Test cell lines | Unseen in train | % unseen | mean abs delta-z on test inputs | p99 abs delta-z |
|---|---|---|---|---|---|
| random | 696 | 0 | 0.0% | 0.0000 | 0.0000 |
| cell-blind | 139 | 139 | 100.0% | 0.0217 | 0.1234 |
| drug-blind | 697 | 0 | 0.0% | 0.0000 | 0.0000 |
| scaffold-blind | 696 | 0 | 0.0% | 0.0000 | 0.0000 |
| tissue-blind | 56 | 56 | 100.0% | 0.0204 | 0.1048 |

## Interpretation

**Splits with a genuine leakage channel:** cell-blind, tissue-blind.

**Splits with no leakage channel:** random, drug-blind, scaffold-blind. These hold out drugs or scaffolds, not cell lines, so every test cell line also appears in training and the fold-wise moments are fitted on (essentially) the same cohort as the cohort-wide moments.

This does not make the original pipeline correct -- the fix is applied to all five splits and all four models, so the comparison stays uniform. It does tell us where the numbers can move, and it is the honest quantitative answer to the reviewer's question.
