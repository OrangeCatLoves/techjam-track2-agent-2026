# Run `agent-explore`

_Autonomous ML research agent, TikTok TechJam 2026 Track 2._

Every number below was computed by the harness. The hypotheses are the agent's; the measurements are not.

No hidden-test metric appears in this file, or in any log the agent can read. See CLAUDE.md section 5.

## Summary

- Iterations recorded: **6**
- Best validation primary: **0.6015** (iteration 3)
- Kept: 2 | rejected: 3 | failed: 1

## Iterations

### Iteration 1 — FAILED

**Hypothesis.** The baseline optimises pointwise logloss while both scored metrics rank within a user's own list. GAUC is literally a pairwise quantity: the chance a positive outranks a negative for the same user. Fitting absolute click probability spends capacity on calibration that the metric ignores, and the ordering it induces is only incidentally optimal. I replace the objective with BPR: for each positive, sample a negative from the same user's list and push their score gap up, each list weighted equally so heavy users cannot dominate. A 0.1-weight pointwise anchor keeps rows whose in-batch list holds only one class from going gradient-free, and a larger batch raises in-batch pair availability.

**Target stage.** `objective`  
**Patch kind.** `pairwise_bpr_loss`  

**Decision.** FAILED — abandoned after a code failure

```
Traceback (most recent call last):
  File "C:\Users\Lenovo\track2-techjam\techjam-track2-agent-2026\harness\_run_patch.py", line 85, in main
    result = runners.train_fm(splits, checkpoint_path=args.checkpoint, **config)
  File "C:\Users\Lenovo\track2-techjam\techjam-track2-agent-2026\harness\models\runners.py", line 246, in train_fm
    loss_report = hlosses.check_loss(loss_fn, kind=hlosses.loss_kind(loss))
  File "C:\Users\Lenovo\track2-techjam\techjam-track2-agent-2026\harness\losses.py", line 239, in check_loss
    raise LossError(
    ...<2 lines>...
        f'which would train the model
```

_Cost: 263s · 42790 in / 25004 out tokens._

### Iteration 2 — KEPT

**Hypothesis.** The baseline optimises pointwise logloss while both scored metrics rank within a user's list, so it spends capacity calibrating absolute probabilities that no within-list ordering ever sees. A listwise softmax cross-entropy over each user's impression list removes the global calibration term entirely: only score differences inside a list carry gradient, which is exactly what GAUC and nDCG@5 read. I group by user_id+date because the measured train list length there (~5.8 mean) matches the evaluation lists (~5.6), so the objective trains on lists the same shape it is scored on. Lists with no positives or all positives carry no ordering information and are masked out, mirroring how GAUC is computed. Expect nDCG@5 to respond most, since softmax concentrates weight at the top of the list.

**Target stage.** `objective`  
**Patch kind.** `listwise_softmax_cross_entropy`  

| metric | value |
|---|---|
| val_gauc | 0.6520 |
| val_ndcg5 | 0.5284 |
| val_primary | 0.5902 |
| train_primary | 0.6014 |
| gap | 0.0112 |

**Decision.** KEPT — val_primary 0.5902 beats the best

_Cost: 175s · 68742 in / 29208 out tokens._

### Iteration 3 — KEPT

**Hypothesis.** Both pairwise BPR and listwise softmax landed at 0.5902, below the pointwise baseline, and ID embedding norms are tiny (user 0.21, video 0.27) while tab/dur_bucket dominate. That is the signature of starved gradients: single-class lists (all-negative or all-positive, ~36% of users) supply exactly zero gradient to any purely relative objective, so most rows never train the user/video cross. The fix is not another relative loss but a hybrid: a pointwise logloss anchor over every row to keep ID embeddings fitting, plus a LambdaRank-weighted pairwise term (|delta nDCG@5| weights, truncated at rank 5) that directly targets both scored metrics, on user_id+date lists matching evaluation length.

**Target stage.** `objective`  
**Patch kind.** `lambdarank_pairwise_with_pointwise_anchor`  

| metric | value |
|---|---|
| val_gauc | 0.6671 |
| val_ndcg5 | 0.5359 |
| val_primary | 0.6015 |
| train_primary | 0.6315 |
| gap | 0.0300 |

**Decision.** KEPT — val_primary 0.6015 beats the best

_Cost: 149s · 95021 in / 34585 out tokens._

### Iteration 4 — rejected

**Hypothesis.** Three ranking losses all landed on 0.6015, so misalignment in pairwise form is not the binding constraint. Two overlooked facts: validation peaks at epoch 8 with a +0.030 train gap (the model memorises already-correct pairs), and under user_id+date grouping a large share of lists are all-positive or all-negative, contributing zero gradient to any pairwise objective, so those rows' embeddings go untrained. This patch uses a saturating squared-hinge on pairs (gradient exactly zero once separated by the margin, an intrinsic brake on memorisation), weights pairs to match how the metric aggregates (1/n_neg per list, i.e. GAUC's positive-count user weighting, plus truncated nDCG@5 lambda deltas), and keeps degenerate lists alive with a small pointwise term.

**Target stage.** `objective`  
**Patch kind.** `metric_weighted_margin_hinge`  

| metric | value |
|---|---|
| val_gauc | 0.6586 |
| val_ndcg5 | 0.5314 |
| val_primary | 0.5950 |
| train_primary | 0.6067 |
| gap | 0.0117 |

**Decision.** rejected — val_primary 0.5950 does not beat the best

_Cost: 474s · 121322 in / 42633 out tokens._

### Iteration 5 — rejected

**Hypothesis.** Four ranking losses have now failed; all replace the dense pointwise gradient and drop degenerate lists. Keep the pointwise gradient that works and instead delete what the metric provably ignores: any score component constant within a user's list. Centering logits per list before the logloss is conditional (fixed-effects) logistic regression — the global intercept and every user-level first-order term receive exactly zero gradient, so capacity and early-stopping pressure move onto within-list discrimination, i.e. the ID crosses whose embedding norms (user_id 0.22, video_id 0.16) are dwarfed by dur_bucket 0.64. Lists are grouped by user_id, matching how GAUC and nDCG@5 quotient the score. A small penalty on each list mean keeps the now-free constant direction identified.

**Target stage.** `objective`  
**Patch kind.** `list_centered_pointwise_logloss`  

| metric | value |
|---|---|
| val_gauc | 0.6668 |
| val_ndcg5 | 0.5359 |
| val_primary | 0.6014 |
| train_primary | 0.6323 |
| gap | 0.0310 |

**Decision.** rejected — val_primary 0.6014 does not beat the best

_Cost: 232s · 147616 in / 50400 out tokens._

### Iteration 6 — rejected

**Hypothesis.** Five objective variants moved nothing, so the loss family is not the bottleneck — what the loss is applied to is. Under user_id lists (43.5 rows) training mixes calibration signal from lists that can never be ranked. Both metrics are pure within-list ranking: all-negative and all-positive lists score 0 or 1 regardless, and GAUC ignores them entirely. So gradient spent on them is wasted capacity, and the model overfits (train-val gap +0.031, peak at epoch 5). This trains listwise softmax on (user_id, date) lists of mean 5.77, matching the 5.6 evaluation length, restricted to discriminative lists only, each weighted equally, with slightly stronger L2.

**Target stage.** `sampling`  
**Patch kind.** `discriminative_list_softmax_eval_matched_lists`  

| metric | value |
|---|---|
| val_gauc | 0.6585 |
| val_ndcg5 | 0.5312 |
| val_primary | 0.5948 |
| train_primary | 0.6017 |
| gap | 0.0069 |

**Decision.** rejected — val_primary 0.5948 does not beat the best

_Cost: 255s · 173892 in / 56225 out tokens._

## Events

| when | kind | detail |
|---|---|---|
| 2026-08-30T13:50:11 | `recovery` | iteration 1: one repair attempt |
