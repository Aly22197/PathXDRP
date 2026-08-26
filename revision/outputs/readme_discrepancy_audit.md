# README vs manuscript discrepancy -- provenance audit

Answers Reviewer #4, major point 3.

## 1. What the README table was generated from

`revision/outputs/stale_summary_v3_vs_baselines.csv` (106 rows), formerly `results/summary_v3_vs_baselines.csv`. The README states this explicitly as its source.

### Defect A -- duplicated runs

4 (model, split, seed) keys appear more than once, so the per-split means average an old run together with its rerun:

```
model     split       seed
cdrscan   random      0       2
drpreter  random      0       2
graphdrp  drug_blind  0       2
          random      0       2
```

### Defect B -- stale PathXDRP sweep

- Stale CSV PathXDRP parameter count(s): [3031429]
- Final sweep PathXDRP parameter count(s): [2819206]

- **16 of 25 stale PathXDRP runs stopped at best_val_epoch <= 2**, i.e. they never trained. These are the runs archived under `results/archive/pathxdrp_v2_lr_too_high/`, produced before the learning-rate schedule was corrected.


#### Per-split comparison (PCC, mean +/- std over 5 seeds)

| Split | Stale CSV (README) | Final sweep (manuscript) | Delta |
|---|---|---|---|
| random | 0.919 +/- 0.009 | 0.930 +/- 0.002 | +0.011 |
| cell_blind | 0.867 +/- 0.002 | 0.864 +/- 0.002 | -0.003 |
| drug_blind | 0.601 +/- 0.019 | 0.513 +/- 0.063 | -0.088 |
| scaffold_blind | 0.441 +/- 0.074 | 0.553 +/- 0.137 | +0.112 |
| tissue_blind | 0.847 +/- 0.005 | 0.848 +/- 0.004 | +0.002 |

## 2. Conclusion

The manuscript numbers come from the final sweep in `results/<model>/*.json`.
The README numbers come from `results/summary_v3_vs_baselines.csv`, an aggregate that was written before the final PathXDRP sweep and never regenerated. That file mixes a failed low-epoch PathXDRP sweep with the baseline runs and additionally double-counts four baseline runs.

**The manuscript is not reporting inflated numbers; the repository was reporting stale ones.** Remedy: delete the stale aggregate, regenerate the README table from `ledger.csv`, and tag the release commit.
