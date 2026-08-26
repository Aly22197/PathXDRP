# PRISM calibration and per-drug significance

Answers Reviewer #4 point 5 (calibration on PRISM) and Reviewer #5 point 11 (significance for the per-drug correlation).

- pairs: **14,423**
- drugs: **99**
- cell lines: **244**

## Calibration on PRISM

| Quantity | Value |
|---|---|
| RMSE | 2.670 |
| ECE, total predictive sigma | 1.578 |
| ECE, epistemic component only | 1.636 |
| selective RMSE at 50% coverage | 2.692 |
| gain from discarding the least-confident half | -0.8% |
| correlation of sigma with absolute error | 0.027 |

## Per-drug correlation

Mean per-drug PCC over 95 drugs with at least five cell lines: **0.016**, bootstrap 95% CI over drugs **[-0.005, 0.037]** (2,000 resamples).

- drugs with a positive correlation: 45 of 95
- interquartile range: -0.053 to 0.081

The interval contains zero, so the per-drug correlation is not distinguishable from no within-drug ranking ability. This supports the revised wording: across unseen cell lines and platforms the model does not usefully rank cell lines within a single drug.

