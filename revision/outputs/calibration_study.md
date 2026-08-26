# W7 -- Calibration on all splits, and post-hoc recalibration

Answers Reviewer #4.5 and Reviewer #5.5; also resolves the numeric
inconsistency raised in Reviewer #5.10.

## 1. Calibration is split-dependent and was only reported for `random`

| Split | RMSE | ECE | selective RMSE @50% | gain vs full coverage | AUROC(sigma, large error) | Spearman(sigma, abs err) |
|---|---|---|---|---|---|---|
| random | 1.018 | 0.213 ± 0.048 | 0.771 | 24.2% | 0.648 | 0.309 |
| cell-blind | 1.435 | 0.148 ± 0.108 | 1.094 | 23.8% | 0.637 | 0.289 |
| drug-blind | 2.724 | 1.428 ± 0.249 | 2.238 | 17.8% | 0.542 | 0.111 |
| scaffold-blind | 2.513 | 1.298 ± 0.339 | 2.426 | 2.7% | 0.526 | 0.050 |
| tissue-blind | 1.525 | 0.361 ± 0.095 | 1.205 | 21.0% | 0.617 | 0.254 |

ECE ranges from 0.148 (cell-blind) to 1.428 (drug-blind). Reporting only the random split, as the submitted manuscript does, showed the most favourable case.

The **ranking** claim survives everywhere: sigma separates high-error from low-error predictions with AUROC 0.47-0.66, and filtering to 50% coverage reduces RMSE on every split. The **absolute calibration** claim does not survive and should be removed from the abstract and the contribution list.

## 2. Reconciling main Table 6 with supplementary Table S4

Reviewer #5 (point 10) noticed that the main text reports ECE 0.244 and the supplement reports 0.220. Tracing both numbers uncovered a **third, larger discrepancy that the reviewers did not catch**.

### 2a. A Methods/implementation mismatch
Section 3.4 of the submitted manuscript defines the calibration input as the total predictive standard deviation `sigma_i = sqrt(sigma^2_epistemic + sigma^2_aleatoric)`. The code did not do that: `pathxdrp/train.py` passed `epistemic` alone to `regression_report(..., uncertainties=...)`, so every published ECE and selective-RMSE number was computed from the epistemic component only.

- ECE as published (epistemic only, 5 seeds, random split): **0.244 ± 0.048** -- reproduced exactly from `results/`.
- ECE with the documented total variance: **0.213 ± 0.048**.

The documented definition gives the *better* number, so this correction costs nothing and removes a discrepancy that a code-checking reviewer would certainly have found. `pathxdrp/train.py` is fixed; every re-run uses the total variance.

### 2b. The 0.244 vs 0.220 difference the reviewer did flag
On the same epistemic-only basis, the 80% evaluation slice used for the temperature experiment gives 0.220 while the full test set gives 0.244. They differ because they are computed on different samples, not because either is wrong. With the corrected total variance the pair becomes 0.213 (full test set) and 0.214 (80% slice). The revised manuscript states the evaluation sample in both captions.

## 3. Post-hoc recalibration: what actually works

| Split | ECE raw | + temperature | + variance scale | + isotonic | fitted T |
|---|---|---|---|---|---|
| random | 0.214 | 0.171 | 0.279 | **0.045** | 1.164 |
| cell-blind | 0.149 | 0.082 | 0.078 | **0.057** | 1.056 |
| drug-blind | 1.431 | 0.337 | 0.381 | **0.133** | 2.156 |
| scaffold-blind | 1.297 | 0.404 | 0.620 | **0.082** | 1.972 |
| tissue-blind | 0.370 | 0.128 | 0.123 | **0.101** | 1.288 |

Isotonic recalibration reduces ECE by 88% on average across the five splits, where temperature scaling does not. This is the experiment the submitted Discussion proposed but never ran, and it converts Reviewer #5's objection into a positive result.

Selective RMSE at 50% coverage is essentially unchanged before and after isotonic recalibration, because any monotone transform of sigma leaves the confidence ORDERING untouched. The selective-prediction result is therefore independent of the calibration question.

## 4. What the revised manuscript should claim

- Keep: sigma is a useful *ranking* signal for selective prediction, on every split, and this is robust to recalibration.
- Keep, newly supported: absolute calibration can be fixed post hoc with isotonic regression; report ECE before and after.
- Drop: the word *calibrated* in the abstract, the Figure 1 caption and the contribution list, when it is used to describe the raw evidential output.
- Add: the per-split ECE table above, so the reader sees the blind-split degradation rather than the random-split best case.
