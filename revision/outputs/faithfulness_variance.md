# Run-to-run variance of attention faithfulness

Discovered while verifying the ablation, and material to the paper's central claim.

## The observation

Two runs of the same configuration and the same seed:

| | PCC | faithfulness (comp.) |
|---|---|---|
| published full model | 0.93089 | 0.6033 |
| variant F, fresh run | 0.93078 | 0.4479 |
| **difference** | **0.0001** | **0.155 (26%)** |

Prediction accuracy reproduces to four decimal places. Faithfulness does not. The two runs stopped at different epochs (48 vs 46) under cuDNN non-determinism, so they are different checkpoints of one recipe.

## Within-run uncertainty, for comparison

Bootstrap over the benchmark drugs, model held fixed:

| Run | drugs | mean | 95% CI (over drugs) | SD across drugs |
|---|---|---|---|---|
| PathXDRP (published) | 237 | 0.6033 | [0.5322, 0.6783] | 0.5581 |
| PathXDRP (variant F, fresh) | 237 | 0.4479 | [0.3968, 0.5008] | 0.4144 |
| DRPreter (published) | 237 | 0.4067 | [0.3656, 0.4471] | 0.3161 |
| variant A (baseline) | 237 | 0.0269 | [0.0172, 0.0386] | 0.0861 |
| variant A' (pooling only) | 237 | 0.0205 | [0.0112, 0.0329] | 0.0877 |

## What this means for the headline comparison

The paper's headline is PathXDRP 0.603 against DRPreter 0.407, a gap of 0.197. The two PathXDRP runs of the same recipe differ by 0.155 --- comparable to that gap. The fresh run (0.448) still exceeds DRPreter (0.407), but by 0.041 rather than 0.197.

The within-run bootstrap intervals do not overlap, so drug sampling does not explain the discrepancy. It is genuine between-checkpoint variance.

## Consequence, and what we do about it

**The single-run comparison of faithfulness between architectures is not sound, and we should not have reported it as a point estimate.** Faithfulness depends on which checkpoint early stopping happens to select, and that varies between runs that are indistinguishable on accuracy.

Three consequences:

1. The manuscript reports faithfulness with a run-to-run spread, not as a single number, and states the number of runs behind it.
2. The PathXDRP-versus-DRPreter faithfulness comparison is softened. Both observed PathXDRP values exceed DRPreter's, so the direction holds, but the magnitude is not established from one run each.
3. The ablation conclusion is unaffected in direction. Variants A and A' sit near 0.02--0.03 while corrected models sit at 0.45--0.60. A gap of 15-20x is not closed by a between-run spread of 0.15, and the A-versus-A' finding rests on that gap rather than on a precise value.

The honest framing is that the head redesign moves faithfulness by an order of magnitude, which is robust, while the residual differences between already-corrected architectures are within run-to-run noise at one run each.
