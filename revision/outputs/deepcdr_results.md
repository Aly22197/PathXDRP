# DeepCDR baseline

Answers Reviewer #3 point 4 and Reviewer #5 point 7.

Expression-only configuration, single seed, fold-wise normalisation.
The comparison band is the range of five-seed mean PCC over the four established models on the same split.

| Split | PCC | RMSE | Spearman | others (PCC range) |
|---|---|---|---|---|
| random | 0.9220 | 1.1024 | 0.8801 | 0.925-0.937 |
| cell-blind | 0.8855 | 1.3380 | 0.8388 | 0.864-0.890 |
| drug-blind | 0.5964 | 2.5191 | 0.5027 | 0.513-0.634 |
| scaffold-blind | 0.4237 | 2.8775 | 0.4020 | 0.385-0.553 |
| tissue-blind | 0.8695 | 1.4201 | 0.8194 | 0.848-0.871 |

## Reading

DeepCDR lands inside the band of the four established models on 4 of 5 splits (cell-blind, drug-blind, scaffold-blind, tissue-blind), below all of them on random.

This is the expected place for a single-omics configuration of a multi-omics model, and it does not change the finding that these architectures cannot be separated on prediction accuracy.
