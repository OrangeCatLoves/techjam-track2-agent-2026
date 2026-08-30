# Run `agent-explore3`

_Autonomous ML research agent, TikTok TechJam 2026 Track 2._

Every number below was computed by the harness. The hypotheses are the agent's; the measurements are not.

No hidden-test metric appears in this file, or in any log the agent can read. See CLAUDE.md section 5.

## Summary

- Iterations recorded: **8**
- Best validation primary: **0.6013** (iteration 5)
- Kept: 2 | rejected: 5 | failed: 1

## Iterations

### Iteration 1 — KEPT

**Hypothesis.** The baseline optimises pointwise logloss while both scored metrics are within-user ranking metrics; a row's absolute probability is irrelevant, only its order inside its own list matters. GAUC is literally a per-user pairwise statistic, so a BPR objective optimises it directly. The list-size measurement supports grouping by user_id+date: train mean 5.77 (median 3) versus valid 5.58, an almost exact match, whereas user_id alone gives 43.5, 7.8x too long, which would train the model on a ranking problem of the wrong shape. I weight each list equally so short and long lists contribute alike. Expect GAUC to move more than nDCG@5.

**Target stage.** `objective`  
**Patch kind.** `pairwise_bpr_loss`  

| metric | value |
|---|---|
| val_gauc | 0.6576 |
| val_ndcg5 | 0.5321 |
| val_primary | 0.5948 |
| train_primary | 0.6012 |
| gap | 0.0064 |

**Decision.** KEPT — val_primary 0.5948 beats the best

_Cost: 204s · 26582 in / 3424 out tokens._

### Iteration 2 — rejected

**Hypothesis.** BPR moved nothing (0.5948) and the diagnostics say why: user_id/video_id embedding norms (0.109/0.142) are five times smaller than tab/dur_bucket, so the ID cross that carries all within-user signal is barely trained, and every pair got equal weight regardless of where in the list it sat — no pressure on the top-5 that nDCG@5 actually scores. I replace uniform pairwise weighting with LambdaRank weights: each positive-negative pair is weighted by half a constant (GAUC term, pairwise) plus half |ΔnDCG@5| from swapping them, normalised per list, lists grouped by user_id+date to match the 5.6-row evaluation lists. Learning rate is raised to 0.003 to actually move the under-trained ID embeddings.

**Target stage.** `objective`  
**Patch kind.** `lambdarank_ndcg5_weighted_pairwise`  

| metric | value |
|---|---|
| val_gauc | 0.6525 |
| val_ndcg5 | 0.5287 |
| val_primary | 0.5906 |
| train_primary | 0.6023 |
| gap | 0.0117 |

**Decision.** rejected — val_primary 0.5906 does not beat the best

_Cost: 199s · 53499 in / 8721 out tokens._

### Iteration 3 — rejected

**Hypothesis.** Both tried objectives were pairwise, and both stalled: GAUC moved 0.0000 while train-val gap grew, so the model is fitting pair margins that don't reorder the top of short lists. nDCG@5 on ~5.6-item lists is dominated by which single item is ranked first, and 57.8% of valid users are discriminative with typically one or two positives. A listwise softmax cross-entropy over each (user_id, date) list normalises scores within exactly the unit the metric scores, puts gradient mass on the top item rather than every pair equally, and drops all-negative/all-positive lists that carry no ordering signal. Training lists then average 5.8 against 5.6 at evaluation. This targets nDCG@5, the metric neither pairwise attempt moved.

**Target stage.** `objective`  
**Patch kind.** `listwise_softmax_ce`  

| metric | value |
|---|---|
| val_gauc | 0.6517 |
| val_ndcg5 | 0.5283 |
| val_primary | 0.5900 |
| train_primary | 0.6017 |
| gap | 0.0118 |

**Decision.** rejected — val_primary 0.5900 does not beat the best

_Cost: 149s · 80422 in / 10843 out tokens._

### Iteration 4 — FAILED

**Hypothesis.** Three consecutive ranking-loss rewrites (BPR, LambdaRank, listwise softmax) all landed below the reference pointwise FM, and the last one overfit (train 0.6017 vs val 0.5900, peak at epoch 8/13). The evidence says the objective is not the bottleneck: with user_id embedding norm 0.223 vs tab 1.164, each fit is finding a different, partial ID-cross solution and the loss change only perturbs which one. Variance, not misalignment, is what I can attack. So: abandon the objective stage, return to the proven pointwise loss, and train 5 seeds blended by within-user rank. Rank-averaging is non-monotone across members, so unlike single-model normalisation it is not a no-op, and it cancels seed-specific noise in the weakly-identified ID embeddings.

**Target stage.** `ensemble`  
**Patch kind.** `seed_ensemble_pointwise_rank_blend`  

**Decision.** FAILED — abandoned after a code failure

```
Traceback (most recent call last):
  File "C:\Users\Lenovo\track2-techjam\techjam-track2-agent-2026\harness\_run_patch.py", line 92, in main
    result = runners.train_ensemble(
        splits, seeds=tuple(seeds),
        normalise=config.pop('normalise', 'within_user_rank'),
        weights=config.pop('weights', None),
        checkpoint_path=args.checkpoint, **config)
  File "C:\Users\Lenovo\track2-techjam\techjam-track2-agent-2026\harness\models\runners.py", line 465, in train_ensemble
    member = train_fm(splits, seed=seed, with_diagnostics=False,
                      checkpoint_path=mem
```

_Cost: 123s · 123593 in / 17646 out tokens._

### Iteration 5 — KEPT

**Hypothesis.** Pure ranking objectives (BPR, lambdarank, listwise softmax) all scored below the pointwise baseline, so the problem is not that ranking losses are wrong but that they alone give too sparse and too noisy a gradient for 27k x 7.5k ID embeddings: only within-list contrasts update parameters, and short lists (mean 5.77 per user+date) with 31% positives yield few informative contrasts. The pointwise term supplies dense per-row signal that fits the embedding table; the listwise term supplies the metric-aligned top-of-list pressure. A 50/50 hybrid, gradient-normalised so both terms have comparable magnitude at the default lr, should retain the baseline's fit while adding ordering pressure on exactly the (user, date) lists whose size matches evaluation lists (5.77 vs 5.58).

**Target stage.** `objective`  
**Patch kind.** `hybrid_pointwise_listwise_softmax`  

| metric | value |
|---|---|
| val_gauc | 0.6668 |
| val_ndcg5 | 0.5358 |
| val_primary | 0.6013 |
| train_primary | 0.6328 |
| gap | 0.0315 |

**Decision.** KEPT — val_primary 0.6013 beats the best

_Cost: 115s · 150205 in / 21426 out tokens._

### Iteration 6 — rejected

**Hypothesis.** Five objective swaps (BPR, lambdarank, softmax-CE, hybrids) all landed within noise of pointwise, so the loss *family* is not the bottleneck — the list construction is. Every one was trained on user_id+date lists averaging 5.8 rows with median 3, so most lists are pure and carry no ordering signal at all; that starves the ID crosses, which is visible in the field norms (dur_bucket 1.66, tab 1.35 vs user_id 0.35, video_id 0.27). The model is essentially a duration/tab prior. Train on full user_id lists (mean 43.5) with a top-k Plackett-Luce objective: each positive must outrank every remaining item, giving ~8x more informative contrasts per user and truncating at k to match nDCG@5.

**Target stage.** `objective`  
**Patch kind.** `topk_plackett_luce_on_full_user_lists`  

| metric | value |
|---|---|
| val_gauc | 0.6662 |
| val_ndcg5 | 0.5356 |
| val_primary | 0.6009 |
| train_primary | 0.6302 |
| gap | 0.0293 |

**Decision.** rejected — val_primary 0.6009 does not beat the best

_Cost: 204s · 177200 in / 29994 out tokens._

### Iteration 7 — rejected

**Hypothesis.** Six objectives all land at ~0.601 because none optimises the metric that is actually scored. LambdaRank and Plackett-Luce ran on full user lists (mean 43.5), where a top-5 cut touches 11% of the list; at evaluation the top-5 cut covers ~90% of a 5.6-item list. So the truncation that defines nDCG@5 was never in the training signal. I optimise a smooth ApproxNDCG@5 surrogate (soft ranks from pairwise sigmoids, soft top-5 mask) on (user_id, date) lists, mean 5.77, matching evaluation length. Degenerate lists are dropped: 42.2% of validation users are all-positive or all-negative and contribute a constant to both metrics, so their gradient is noise.

**Target stage.** `objective`  
**Patch kind.** `approx_ndcg5_surrogate_on_session_lists`  

| metric | value |
|---|---|
| val_gauc | 0.6550 |
| val_ndcg5 | 0.5310 |
| val_primary | 0.5930 |
| train_primary | 0.5965 |
| gap | 0.0034 |

**Decision.** rejected — val_primary 0.5930 does not beat the best

_Cost: 295s · 204209 in / 38527 out tokens._

### Iteration 8 — rejected

**Hypothesis.** Every ranking surrogate tried so far replaced binary supervision with a normalised competition over the list, and all lost to pointwise. Softmax-family losses are ill-posed here: with user_id+date lists averaging 5.8 and 30% of validation users all-negative and 12% all-positive, a large share of training lists have no winner or many, so the target is undefined or diluted. But plain pointwise wastes capacity fitting each user's absolute level, which is provably irrelevant within-user. This objective keeps per-row binary logloss and only removes the list's mean logit, so the gradient becomes the list-centred residual: it learns within-list contrast, is well defined for degenerate lists (down-weighted, not undefined), and never normalises probabilities across items.

**Target stage.** `objective`  
**Patch kind.** `list_centered_logistic_loss`  

| metric | value |
|---|---|
| val_gauc | 0.6591 |
| val_ndcg5 | 0.5327 |
| val_primary | 0.5959 |
| train_primary | 0.6018 |
| gap | 0.0059 |

**Decision.** rejected — val_primary 0.5959 does not beat the best

_Cost: 190s · 231171 in / 44357 out tokens._

## Events

| when | kind | detail |
|---|---|---|
| 2026-08-30T14:47:43 | `recovery` | iteration 4: one repair attempt |
