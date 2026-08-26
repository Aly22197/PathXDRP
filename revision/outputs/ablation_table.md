# W6 -- Ablation with faithfulness metrics

Answers Reviewer #3.5, #4.4 and #5.3.

| Variant | Res+LN | drop h_mol | Aux | Pool | PCC | RMSE | Comp $\uparrow$ | Suff $\downarrow$ | attn AUROC |
|---|---|---|---|---|---|---|---|---|---|
| A  baseline (old head) | -- | -- | -- | attention | 0.9323 | 1.0152 | 0.027 | 0.027 | 0.645 |
| A' pooling-only control | -- | -- | -- | mean | 0.9331 | 1.0107 | 0.020 | 0.022 | 0.634 |
| B  + residual/LN | yes | -- | -- | mean | 0.9321 | 1.0196 | 0.055 | 0.053 | 0.628 |
| C  + drop $h_{mol}$ | -- | yes | -- | mean | 0.9169 | 1.1161 | 0.372 | 0.438 | 0.817 |
| D  + attention-aux | -- | -- | yes | mean | 0.9298 | 1.0337 | 0.100 | 0.232 | 0.824 |
| E  B + C | yes | yes | -- | mean | 0.9338 | 1.0019 | 0.501 | 0.501 | 0.591 |
| F  full PathXDRP | yes | yes | yes | mean | 0.9308 | 1.0238 | 0.448 | 0.453 | 0.817 |

## The control that decides Reviewer #5's objection

- A  (nothing changed):        comprehensiveness 0.027
- A' (pooling changed only):   comprehensiveness 0.020
- F  (full PathXDRP):          comprehensiveness 0.448   [variant F, this ablation]

**Mean pooling alone accounts for -1.5% of the total gain.**

Reviewer #5's alternative explanation is ruled out. Switching the atom pooling from attention-weighted to mean, with nothing else changed, leaves faithfulness essentially where it was (0.027 to 0.020); it does not move it toward the corrected model's 0.448. The residual, the dropped highway and the auxiliary loss are what make the attention load-bearing.

**Caveat.** These are single-seed runs. The A-versus-A' difference itself (0.027 vs 0.020) is small enough to be seed noise and we do not interpret its sign. The conclusion does not rest on it: it rests on both variants sitting near 0.03 while the corrected model sits at 0.45, a gap of roughly 17x that no plausible seed variation closes.
