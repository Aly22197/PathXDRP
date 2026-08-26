# Cross-platform transfer to PRISM

Regenerated from the public PRISM Repurposing 20Q2 secondary screen by `scripts/prepare_prism.py` and `scripts/external_validation.py`. Every model sees the same pairs: drugs shared by name with GDSC2, PRISM cell lines absent from the GDSC2 training cohort, implausible curve fits removed by an IQR x3 fence.

| Model | PCC | Spearman | RMSE | R2_res | per-drug PCC | n |
|---|---|---|---|---|---|---|
| CDRScan | 0.7036 | 0.6923 | 4.381 | -2.290 | 0.1388 | 14,423 |
| PathXDRP | 0.6841 | 0.6764 | 2.670 | -0.222 | 0.0283 | 14,423 |
| DeepCDR | 0.6786 | 0.6648 | 3.202 | -0.757 | 0.0683 | 14,423 |
| GraphDRP | 0.6678 | 0.6553 | 3.333 | -0.904 | 0.0548 | 14,423 |
| DRPreter | 0.6412 | 0.6202 | 3.562 | -1.175 | -0.0144 | 14,423 |

## Reading

All 5 architectures transfer into a PCC band of 0.641 to 0.704, a spread of 0.062. CDRScan is nominally best. No architecture distinguishes itself, and the ordering here should not be read as a ranking: the band is narrower than the seed-to-seed spread within several of the GDSC2 blind splits.

The residual coefficient of determination is negative for every model, which is the platform offset rather than a failure to discriminate; see the decomposition in the manuscript.

