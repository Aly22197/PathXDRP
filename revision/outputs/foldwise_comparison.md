# W3 -- fold-wise vs cohort-wide normalisation

Answers Reviewer #3, point 3.

Paired runs available: **9** of 125.

## Change in PCC (fold-wise minus cohort-wide)

| Split | Leakage channel? | Model | cohort | fold-wise | delta |
|---|---|---|---|---|---|
| cell-blind | **yes** | PathXDRP | 0.8621 | 0.8623 | +0.0002 |
| cell-blind | **yes** | DRPreter | 0.8693 | 0.8676 | -0.0017 |
| cell-blind | **yes** | GraphDRP | 0.8838 | 0.8674 | -0.0164 |
| cell-blind | **yes** | CDRScan | 0.8894 | 0.8889 | -0.0006 |
| tissue-blind | **yes** | PathXDRP | 0.8462 | 0.8562 | +0.0100 |
| tissue-blind | **yes** | DRPreter | 0.8539 | 0.8535 | -0.0004 |
| tissue-blind | **yes** | GraphDRP | 0.8636 | 0.8644 | +0.0008 |
| tissue-blind | **yes** | CDRScan | 0.8699 | 0.8733 | +0.0035 |

## Attributing the ECE change to the right fix

Two independent corrections landed together: fold-wise normalisation, and using the total predictive variance instead of the epistemic component alone (the latter is what our own Methods section always specified). Recomputing from the saved predictions separates them.

| Split | ECE published | + definition fix | + fold-wise norm | due to definition | due to normalisation |
|---|---|---|---|---|---|
| cell-blind | 0.2448 | 0.2217 | 0.1475 | -0.0230 | -0.0742 |
| tissue-blind | 0.5887 | 0.3811 | 0.4469 | -0.2076 | +0.0658 |

## Was the diagnostic right?

- Splits **with** a leakage channel: mean |delta PCC| = 0.00409, max 0.01639.
