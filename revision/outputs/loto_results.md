# W3b -- Leave-one-tissue-out

Answers Reviewer #5, point 6.

| Held-out tissue | Test rows | PathXDRP | CDRScan |
|---|---|---|---|
| lung adeno. (LUAD) | 6,026 | 0.8562 | 0.8733 |
| breast | 4,883 | 0.8082 | 0.8421 |
| large intestine | 4,606 | 0.8293 | 0.8848 |
| small-cell lung | 4,176 | 0.8605 | 0.8744 |
| ovary | 3,793 | 0.8601 | 0.8876 |

## The two spreads

| Model | across-tissue mean | across-tissue SD | seed SD on fold 0 | ratio |
|---|---|---|---|---|
| PathXDRP | 0.8428 | **0.0233** | 0.0045 | 5x |
| CDRScan | 0.8725 | **0.0181** | 0.0019 | 10x |

## Reading

The across-tissue standard deviation for PathXDRP is 0.0233 over 5 tissues, against a seed-to-seed standard deviation of 0.0045 on the single fold the submitted paper reported --- roughly 5 times larger.

This is the reviewer's point, quantified. The error bar the submitted manuscript attached to its tissue-blind column describes how much the number moves when the model is re-initialised, not how much it moves when a different tissue is held out. The second quantity is the one a reader cares about, and it is far larger.

Per-tissue difficulty varies substantially: small-cell lung gives 0.861 and breast gives 0.808. The submitted paper reported fold 0 (lung adeno. (LUAD)) alone.
