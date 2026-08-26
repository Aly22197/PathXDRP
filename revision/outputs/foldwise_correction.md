# Fold-wise normalisation: paired correction (seed 0)

| Split | Model | Cell encoder | cohort | fold-wise | delta |
|---|---|---|---|---|---|
| cell-blind | PathXDRP | KEGG pathway statistics | 0.8622 | 0.8642 | +0.0019 |
| cell-blind | DRPreter | KEGG pathway statistics | 0.8693 | 0.8676 | -0.0017 |
| cell-blind | GraphDRP | 1D-CNN over raw genes | 0.8838 | 0.8674 | **-0.0164**
| cell-blind | CDRScan | expression MLP $+$ fingerprint | 0.8894 | 0.8889 | -0.0006 |
| tissue-blind | PathXDRP | KEGG pathway statistics | 0.8462 | 0.8562 | +0.0100 |
| tissue-blind | DRPreter | KEGG pathway statistics | 0.8539 | 0.8535 | -0.0004 |
| tissue-blind | GraphDRP | 1D-CNN over raw genes | 0.8636 | 0.8644 | +0.0008 |
| tissue-blind | CDRScan | expression MLP $+$ fingerprint | 0.8699 | 0.8733 | +0.0035 |
