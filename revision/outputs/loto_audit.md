# W3b -- Leave-one-tissue-out: the split already exists

Answers Reviewer #5, point 6.

## Finding

`pathxdrp/data/splits.py:tissue_blind_split` builds **one fold per tissue** over the five most-represented tissues. Every run in the submitted sweep passed `--fold 0`. The reported `tissue_blind` column of Table 4 is therefore a single-tissue probe on the largest tissue, exactly as the reviewer suspected -- but the remaining folds are already generated and only need to be run.

## The five folds

| Fold | Held-out tissue | Test rows | Test cell lines | Train rows | Run in submission? |
|---|---|---|---|---|---|
| 0 | lung_NSCLC_adenocarcinoma | 6,026 | 56 | 138,407 | **yes** |
| 1 | breast | 4,883 | 45 | 140,694 | no |
| 2 | large_intestine | 4,606 | 41 | 141,247 | no |
| 3 | lung_small_cell_carcinoma | 4,176 | 41 | 142,108 | no |
| 4 | ovary | 3,793 | 36 | 142,874 | no |

## Consequence

The submitted tissue-blind number holds out **lung_NSCLC_adenocarcinoma** only. A genuine leave-one-tissue-out result is the mean over folds 0--4, with the per-tissue spread reported as the uncertainty. That spread is the quantity the submitted error bars should have been, and it will be much larger than the initialisation noise they actually measured.

Cost: 4 additional folds x 4 models. Because each fold holds out one tissue rather than five, individual runs are comparable in cost to the existing tissue-blind runs.

## Wording fix for the manuscript

The submitted text is self-contradictory on this point: Section 3.3 says the split "holds out one of the top-five tissues by row count" while the caption of the split-size table says it "holds out the top-five tissues". The code holds out one tissue per fold. The revised text says so, and reports which tissue.
