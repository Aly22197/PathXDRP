# W4 -- Clustered bootstrap results

Answers Reviewer #3.2 and Reviewer #4.6.

`rows` is the submitted (invalid) scheme; the others cluster the resampling unit so that dependent observations move together.

## How many (split, baseline, seed) comparisons stay significant

| Scheme | Significant at 95% | of total | share |
|---|---|---|---|
| rows | 69 | 75 | 92% |
| drug | 43 | 75 | 57% |
| cell | 62 | 75 | 83% |
| scaffold | 45 | 75 | 60% |
| twoway | 36 | 75 | 48% |

## Seed-0 fold, direct replacement for submitted Table 8

| Split | Baseline | dPCC | rows 95% CI | drug 95% CI | cell 95% CI | two-way 95% CI |
|---|---|---|---|---|---|---|
| random | DRPreter | +0.0034 | [+0.002, +0.005]* | [+0.002, +0.006]* | [+0.002, +0.005]* | [+0.001, +0.006]* |
| random | GraphDRP | -0.0064 | [-0.008, -0.005]* | [-0.010, -0.004]* | [-0.008, -0.005]* | [-0.010, -0.003]* |
| random | CDRScan | -0.0031 | [-0.004, -0.002]* | [-0.005, -0.002]* | [-0.004, -0.002]* | [-0.005, -0.001]* |
| cell-blind | DRPreter | -0.0071 | [-0.010, -0.004]* | [-0.012, -0.003]* | [-0.017, +0.002] | [-0.017, +0.003] |
| cell-blind | GraphDRP | -0.0216 | [-0.025, -0.018]* | [-0.030, -0.016]* | [-0.032, -0.012]* | [-0.033, -0.010]* |
| cell-blind | CDRScan | -0.0272 | [-0.031, -0.024]* | [-0.037, -0.020]* | [-0.039, -0.017]* | [-0.041, -0.014]* |
| drug-blind | DRPreter | -0.0821 | [-0.099, -0.065]* | [-0.338, +0.149] | [-0.096, -0.069]* | [-0.335, +0.170] |
| drug-blind | GraphDRP | -0.1087 | [-0.120, -0.098]* | [-0.284, +0.040] | [-0.118, -0.099]* | [-0.270, +0.053] |
| drug-blind | CDRScan | -0.1789 | [-0.190, -0.168]* | [-0.321, -0.038]* | [-0.188, -0.169]* | [-0.324, -0.034]* |
| scaffold-blind | DRPreter | +0.0313 | [+0.017, +0.046]* | [-0.164, +0.273] | [+0.019, +0.045]* | [-0.190, +0.253] |
| scaffold-blind | GraphDRP | -0.0724 | [-0.087, -0.057]* | [-0.304, +0.183] | [-0.084, -0.060]* | [-0.316, +0.172] |
| scaffold-blind | CDRScan | +0.2412 | [+0.226, +0.257]* | [-0.007, +0.443] | [+0.227, +0.255]* | [+0.015, +0.467]* |
| tissue-blind | DRPreter | -0.0077 | [-0.011, -0.004]* | [-0.014, -0.003]* | [-0.020, +0.005] | [-0.021, +0.005] |
| tissue-blind | GraphDRP | -0.0174 | [-0.021, -0.014]* | [-0.023, -0.013]* | [-0.028, -0.007]* | [-0.029, -0.006]* |
| tissue-blind | CDRScan | -0.0237 | [-0.028, -0.020]* | [-0.031, -0.018]* | [-0.039, -0.008]* | [-0.041, -0.007]* |

`*` marks an interval that excludes zero.

## Interpretation

Under naive row resampling 92% of the model-vs-baseline PCC differences look significant. Under the two-way cluster estimator only 48% do. The row-level scheme was inflating significance exactly as Reviewer #3 predicted: the effective sample size is the number of drugs and cell lines, not the number of matrix cells.

This is consistent with the Friedman test already reported in the submitted manuscript (chi2 = 4.44, p = 0.218): the four architectures cannot be separated on prediction accuracy. The revised manuscript should state this as the headline finding rather than as a caveat.
