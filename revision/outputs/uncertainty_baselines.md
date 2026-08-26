# W7 -- Evidential uncertainty against simple baselines

Answers the second half of Reviewer #4, point 5.

Each column is an uncertainty proxy scored the same way: AUROC for identifying the larger-error half of the test set. Higher is better; 0.5 is chance. None of the baselines needs an uncertainty head.

| Split | evidential sigma | extremeness | drug rarity | cell rarity | expression distance |
|---|---|---|---|---|---|
| random | **0.648** | 0.487 | 0.521 | 0.503 | 0.500 |
| cell-blind | **0.637** | 0.481 | 0.502 | 0.500 | 0.485 |
| drug-blind | 0.542 | **0.547** | 0.500 | 0.484 | 0.500 |
| scaffold-blind | **0.526** | 0.518 | 0.500 | 0.495 | 0.500 |
| tissue-blind | **0.617** | 0.485 | 0.506 | 0.500 | 0.497 |

## Selective-RMSE gain from discarding the least-confident half

| Split | evidential sigma | extremeness | drug rarity | cell rarity | expression distance |
|---|---|---|---|---|---|
| random | **24.2%** | -0.5% | 6.2% | -0.4% | 5.8% |
| cell-blind | **23.8%** | -1.3% | 0.3% | -0.3% | -3.8% |
| drug-blind | **17.8%** | 11.0% | -0.4% | -2.3% | -0.4% |
| scaffold-blind | 2.7% | **7.7%** | -0.4% | -1.0% | -0.4% |
| tissue-blind | **21.0%** | 0.5% | 1.6% | -1.0% | -0.6% |

## Reading

The evidential sigma is the best error-ranking signal on **4 of 5** splits. Where it is not, the winner is worth noting rather than hiding: a cheap heuristic that beats a trained uncertainty head is a real result about the head.

The comparison also gives the selective-prediction claim a reference point it lacked in the submitted version. A 24% RMSE reduction sounds impressive on its own; it is more informative alongside what `|prediction - training mean|` achieves for free.
