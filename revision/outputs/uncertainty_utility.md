# W8 -- Practical value of the predicted uncertainty

Answers Reviewer #1, point 2.

A hit is a (drug, cell) pair whose measured LN-IC50 lies in the most sensitive 10% of the test fold. Both strategies spend the same budget of k assays; the uncertainty-aware one first discards the least-confident predictions, then ranks what is left.

| Split | coverage | budget k | precision@k, naive | precision@k, uncertainty-aware | change |
|---|---|---|---|---|---|
| cell-blind | 30% | 100 | 0.990 | 0.940 | -0.050 |
| cell-blind | 30% | 500 | 0.928 | 0.311 | -0.616 |
| cell-blind | 30% | 1000 | 0.876 | 0.156 | -0.719 |
| cell-blind | 50% | 100 | 0.990 | 0.988 | -0.002 |
| cell-blind | 50% | 500 | 0.928 | 0.535 | -0.393 |
| cell-blind | 50% | 1000 | 0.876 | 0.279 | -0.597 |
| cell-blind | 70% | 100 | 0.990 | 0.986 | -0.004 |
| cell-blind | 70% | 500 | 0.928 | 0.770 | -0.157 |
| cell-blind | 70% | 1000 | 0.876 | 0.481 | -0.395 |
| drug-blind | 30% | 100 | 0.918 | 0.730 | -0.188 |
| drug-blind | 30% | 500 | 0.737 | 0.478 | -0.259 |
| drug-blind | 30% | 1000 | 0.640 | 0.299 | -0.341 |
| drug-blind | 50% | 100 | 0.918 | 0.762 | -0.156 |
| drug-blind | 50% | 500 | 0.737 | 0.550 | -0.188 |
| drug-blind | 50% | 1000 | 0.640 | 0.397 | -0.243 |
| drug-blind | 70% | 100 | 0.918 | 0.772 | -0.146 |
| drug-blind | 70% | 500 | 0.737 | 0.593 | -0.144 |
| drug-blind | 70% | 1000 | 0.640 | 0.460 | -0.180 |
| random | 30% | 100 | 0.996 | 1.000 | +0.004 |
| random | 30% | 500 | 0.983 | 0.906 | -0.077 |
| random | 30% | 1000 | 0.952 | 0.514 | -0.438 |
| random | 50% | 100 | 0.996 | 1.000 | +0.004 |
| random | 50% | 500 | 0.983 | 0.970 | -0.013 |
| random | 50% | 1000 | 0.952 | 0.676 | -0.277 |
| random | 70% | 100 | 0.996 | 1.000 | +0.004 |
| random | 70% | 500 | 0.983 | 0.979 | -0.004 |
| random | 70% | 1000 | 0.952 | 0.781 | -0.171 |
| scaffold-blind | 30% | 100 | 0.830 | 0.748 | -0.082 |
| scaffold-blind | 30% | 500 | 0.727 | 0.468 | -0.258 |
| scaffold-blind | 30% | 1000 | 0.579 | 0.336 | -0.243 |
| scaffold-blind | 50% | 100 | 0.830 | 0.734 | -0.096 |
| scaffold-blind | 50% | 500 | 0.727 | 0.520 | -0.207 |
| scaffold-blind | 50% | 1000 | 0.579 | 0.391 | -0.188 |
| scaffold-blind | 70% | 100 | 0.830 | 0.720 | -0.110 |
| scaffold-blind | 70% | 500 | 0.727 | 0.579 | -0.148 |
| scaffold-blind | 70% | 1000 | 0.579 | 0.431 | -0.148 |
| tissue-blind | 30% | 100 | 0.948 | 0.454 | -0.494 |
| tissue-blind | 30% | 500 | 0.800 | 0.094 | -0.706 |
| tissue-blind | 30% | 1000 | 0.536 | 0.047 | -0.489 |
| tissue-blind | 50% | 100 | 0.948 | 0.732 | -0.216 |
| tissue-blind | 50% | 500 | 0.800 | 0.205 | -0.595 |
| tissue-blind | 50% | 1000 | 0.536 | 0.106 | -0.430 |
| tissue-blind | 70% | 100 | 0.948 | 0.930 | -0.018 |
| tissue-blind | 70% | 500 | 0.800 | 0.402 | -0.398 |
| tissue-blind | 70% | 1000 | 0.536 | 0.215 | -0.322 |

## Result: the filter HURTS hit-rate triage, and we can say why

Confidence filtering lowers precision@k in 42 of 45 configurations, on every split and at every budget tested. This is a negative result and the revised manuscript reports it as one. It also corrects a claim the submitted version came close to making, that the selective-prediction gain translates into better screening decisions.

### Mechanism

| Split | Spearman(predicted LN-IC50, sigma) | median sigma in the most-sensitive predicted decile / overall |
|---|---|---|
| random | -0.166 | 1.20x |
| cell-blind | -0.439 | 1.42x |
| drug-blind | -0.346 | 1.09x |
| scaffold-blind | +0.035 | 1.69x |
| tissue-blind | -0.749 | 2.22x |

On four of the five splits predicted sigma is *anti*-correlated with predicted LN-IC50 -- the more sensitive the model thinks a pair is, the less confident it is about it -- and on scaffold-blind the correlation is flat. In every case, including scaffold-blind, the most-sensitive predicted decile carries 1.1x to 2.2x the median uncertainty. Sensitive responses sit in the sparse tail of the training distribution, so the model is genuinely least certain exactly where the interesting candidates are. A confidence filter therefore removes candidate hits preferentially, and precision@k falls.

### What this means for the paper

1. The selective-RMSE result is real but narrower than it looks. RMSE improves under filtering partly because filtering removes the extreme responses, which are both the highest-error and the highest-value predictions. Reporting selective RMSE alone overstates the operational value of the uncertainty.
2. The honest use case is not hit triage. It is *flagging*: sigma tells a screening scientist which predictions to distrust, which is useful for deciding where a confirmatory assay is needed, not for choosing which compounds look promising.
3. Section 5.2 of the revised manuscript states both the selective-RMSE gain and this counter-result, so that a reader cannot take the former as a screening recommendation.
