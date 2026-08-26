# W16 -- Computational cost

Answers Reviewer #4 minor point 8; supports Reviewer #5 point 9.

Training times are wall clock on a single NVIDIA RTX 3060 Laptop GPU (6 GB), median over the 25 runs of each model, at the published protocol (50 epochs, early-stopping patience 10, batch size 64).

| Model | Parameters | relative | Median train time | relative | Median best epoch | Total sweep | random PCC |
|---|---|---|---|---|---|---|---|
| DRPreter | 627,073 | 1.0x | 36.2 min | 1.7x | 6 | 23.5 h | 0.9254 |
| PathXDRP | 2,819,206 | 4.5x | 71.2 min | 3.3x | 14 | 35.8 h | 0.9302 |
| GraphDRP | 2,859,494 | 4.6x | 21.5 min | 1.0x | 8 | 12.2 h | 0.9366 |
| CDRScan | 10,713,601 | 17.1x | 29.2 min | 1.4x | 29 | 12.2 h | 0.9331 |

*Peak memory and inference throughput pending: re-run with `--measure` when the GPU is free.*

## The complexity-versus-gain trade-off

The cost of PathXDRP is not in its parameter count. It has 2,819,206 parameters, fewer than GraphDRP (2,859,494) and a quarter of CDRScan (10,713,601). The cost is in time: it is the slowest model to train, 3.3x GraphDRP and 2.4x CDRScan per run. The reason is the cross-attention step, which scores every atom against all 370 pathway tokens; that is arithmetic over a large intermediate tensor rather than extra weights.

Against that cost, the accuracy return is nil. Random-split PCC is 0.9302 for PathXDRP versus 0.9366 for GraphDRP, which trains in under a third of the time. The clustered bootstrap (W4) puts differences of this size inside the noise on most splits.

Stated plainly: the extra architecture does not buy accuracy. It buys per-prediction uncertainty and an attention map that is measurably load-bearing. Whether that is a good trade depends on whether the downstream user needs those two things. For a pure accuracy objective on this benchmark, GraphDRP is the better engineering choice, and the revised Discussion says so.
