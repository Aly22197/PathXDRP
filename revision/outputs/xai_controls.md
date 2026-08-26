# W10 -- XAI benchmark controls and confidence intervals

Answers Reviewer #4.7 (size confound, missing CIs) and Reviewer #5.8
(Recall_K saturation).

## 1. Recall_K against a size-matched null

`null size-matched` draws K pathways whose sizes match those the model
actually selected. `excess` is the part of Recall_K that pathway size
cannot explain, with a bootstrap 95% CI over drugs.

| Model | Method | K | Recall_K [95% CI] | uniform null | size-matched null | excess [95% CI] |
|---|---|---|---|---|---|---|
| DRPreter | attn | 5 | 0.166 [0.110, 0.224] | 0.268 | 0.228 | -0.062 [-0.101, -0.024] |
| PathXDRP | attn | 5 | 0.322 [0.252, 0.398] | 0.268 | 0.287 | +0.036 [-0.009, +0.086] |
| DRPreter | attn | 10 | 0.308 [0.238, 0.380] | 0.400 | 0.327 | -0.019 [-0.067, +0.030] |
| PathXDRP | attn | 10 | 0.410 [0.337, 0.483] | 0.400 | 0.412 | -0.001 [-0.051, +0.045] |
| DRPreter | attn | 20 | 0.443 [0.366, 0.519] | 0.539 | 0.440 | +0.004 [-0.046, +0.055] |
| PathXDRP | attn | 20 | 0.501 [0.425, 0.579] | 0.539 | 0.538 | -0.037 [-0.085, +0.009] |
| CDRScan | ig | 5 | 0.697 [0.628, 0.768] | 0.268 | 0.275 | +0.422 [+0.359, +0.482]* |
| DRPreter | ig | 5 | 0.687 [0.613, 0.756] | 0.274 | 0.232 | +0.455 [+0.393, +0.517]* |
| GraphDRP | ig | 5 | 0.684 [0.615, 0.755] | 0.268 | 0.275 | +0.409 [+0.347, +0.469]* |
| PathXDRP | ig | 5 | 0.716 [0.643, 0.784] | 0.274 | 0.294 | +0.422 [+0.362, +0.483]* |
| CDRScan | ig | 10 | 0.768 [0.704, 0.829] | 0.400 | 0.389 | +0.379 [+0.314, +0.442]* |
| DRPreter | ig | 10 | 0.748 [0.677, 0.812] | 0.407 | 0.333 | +0.415 [+0.353, +0.476]* |
| GraphDRP | ig | 10 | 0.763 [0.698, 0.826] | 0.400 | 0.389 | +0.375 [+0.309, +0.437]* |
| PathXDRP | ig | 10 | 0.758 [0.686, 0.820] | 0.407 | 0.416 | +0.342 [+0.286, +0.398]* |
| CDRScan | ig | 20 | 0.794 [0.730, 0.855] | 0.539 | 0.477 | +0.317 [+0.249, +0.381]* |
| DRPreter | ig | 20 | 0.766 [0.696, 0.828] | 0.548 | 0.443 | +0.323 [+0.263, +0.381]* |
| GraphDRP | ig | 20 | 0.776 [0.712, 0.838] | 0.539 | 0.477 | +0.299 [+0.232, +0.364]* |
| PathXDRP | ig | 20 | 0.765 [0.696, 0.827] | 0.548 | 0.546 | +0.219 [+0.166, +0.272]* |

`*` marks an excess whose CI excludes zero.

12 of 18 (model, method, K) cells retain a significant advantage over the size-matched null.

### The headline consequence, stated plainly

**Attention-based gene-set Recall_K does not beat a size-matched null.** Across the 6 attention cells only 0 shows a significant excess. For PathXDRP the excess is +0.033 [-0.011, +0.083] at K=5 and is zero or negative at K=10 and K=20. For DRPreter it is significantly *negative* at K=5. In other words, the raw attention Recall_K numbers in Table 10 are largely a restatement of which pathways are large, not evidence that attention finds the right biology.

**Integrated gradients does beat the null, for every model, by a wide margin** (excess +0.22 to +0.46, all CIs excluding zero). But PathXDRP's IG excess is not larger than the baselines' (PathXDRP +0.422 vs DRPreter +0.455, GraphDRP +0.409, CDRScan +0.422 at K=5), so the gene-set-recall route gives PathXDRP no advantage once the null is subtracted.

This does **not** touch the faithfulness result. Comprehensiveness measures whether masking the attended features changes the prediction; it has no pathway-size confound and is the claim the paper actually rests on. What must go is the separate claim that the attention *points at the right biology*, which the submitted Conclusion states as leading "on every attention gene-set Recall_K value".

*Note on the drug set.* These numbers are computed on the drugs whose resolved targets appear in at least one KEGG pathway, which is the universe over which Recall_K is even defined. That filter makes the observed and null recalls directly comparable but gives slightly different absolute values from the submitted tables (e.g. attention Recall_5 = 0.322 here vs 0.287 reported over all 143 resolved drugs).

## 2. Saturation of Recall_K (Reviewer #5.8)

| K | Recall of the K LARGEST pathways | Recall of K uniform-random pathways |
|---|---|---|
| 1 | 0.086 | 0.076 |
| 2 | 0.647 | 0.132 |
| 3 | 0.672 | 0.188 |
| 5 | 0.680 | 0.273 |
| 10 | 0.762 | 0.407 |
| 20 | 0.781 | 0.540 |
| 30 | 0.803 | 0.618 |
| 50 | 0.860 | 0.715 |

Simply taking the 50 largest KEGG pathways -- a strategy that never looks at the model -- already recovers 0.860 of the annotated targets. The reviewer is correct that the metric saturates. Two consequences for the revised manuscript:

1. Report Recall_K only at small K (K = 5, and at most K = 10), where the null is far from the ceiling.
2. Report the **excess over the size-matched null**, not the raw value, and always with a confidence interval.

## 3. What to change in the manuscript

- Replace the raw Recall_K columns of Tables 9 and 10 with `Recall_K (excess over size-matched null) [95% CI]`.
- State in Section 3.4 that Recall_K has a size-dependent null and give the null construction explicitly.
- Drop or heavily caveat the K = 20 column; the submitted text already calls it "a saturation ceiling of the metric rather than a real model difference", and this table quantifies that statement.
- The claim that PathXDRP "leads on every attention gene-set Recall_K" (Conclusion) must be re-checked against the excess column and softened to whatever survives.
