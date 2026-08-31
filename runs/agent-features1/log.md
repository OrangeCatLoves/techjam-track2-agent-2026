# Run `agent-features1`

_Autonomous ML research agent, TikTok TechJam 2026 Track 2._

Every number below was computed by the harness. The hypotheses are the agent's; the measurements are not.

No hidden-test metric appears in this file, or in any log the agent can read. See CLAUDE.md section 5.

## Summary

- Iterations recorded: **9**
- Best validation primary: **0.6026** (iteration 8)
- Kept: 5 | rejected: 3 | failed: 1

## Iterations

### Iteration 1 — KEPT

**Hypothesis.** The baseline optimises pointwise logloss while both scored metrics rank within a user's list. GAUC is literally a pairwise quantity (P(positive outranks negative) per user), so the objective is misaligned with the metric, not the features or the capacity (both measured dead ends). I train the same FM with an all-pairs BPR loss over each user's list. Grouping: list_size_profile shows user_id+date averages 5.77 items on train against 5.58 per user on valid, so it matches evaluation list length almost exactly, while user_id alone is 43.5 and would optimise a much longer ranking than is ever scored. Each list is weighted equally by pair count so long lists do not dominate.

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

_Cost: 242s · 27075 in / 4259 out tokens._

### Iteration 2 — rejected

**Hypothesis.** Iteration 2 changed the objective to pairwise BPR and moved nothing, yet the diagnostics show why: ID fields are barely learned (user_id 0.109, video_id 0.142) while tab and dur_bucket dominate. All-pairs BPR spends most gradient on easy cross-list-average contrasts and treats every pair equally, so it never sharpens the top of a six-item list, which is exactly what nDCG@5 scores. A listwise softmax over (user_id, date) lists normalises within the list, weights each list equally like nDCG does, and concentrates gradient on the highest-scoring wrong item. Lists with no positives are dropped since they carry no ordering signal. Same grouping as last run, so the loss is the only variable.

**Target stage.** `objective`  
**Patch kind.** `listwise_softmax_cross_entropy`  

| metric | value |
|---|---|
| val_gauc | 0.6569 |
| val_ndcg5 | 0.5319 |
| val_primary | 0.5944 |
| train_primary | 0.6011 |
| gap | 0.0067 |

**Decision.** rejected — val_primary 0.5944 does not beat the best

_Cost: 175s · 54487 in / 7439 out tokens._

### Iteration 3 — FAILED

**Hypothesis.** Both ranking losses landed below the reference FM, so the objective is not the bottleneck; revert to pointwise logloss and attack the item side instead. Field norms say video_id 0.140 and author_id 0.141 are barely learned, while tab 0.795 and dur_bucket 0.735 dominate — with ~7.5k video IDs the model leans on coarse fields and underfits item quality. A causally-windowed, smoothed historical long-view rate is a dense surrogate for that sparse ID: it varies within a user's list, so unlike user-side fields it can reorder it. Adding video rate, video log-exposure and author rate should raise both metrics. Field norms will show directly whether the new fields were used or ignored.

**Target stage.** `features`  
**Patch kind.** `causal_item_history_encodings`  

**Decision.** FAILED — abandoned after a code failure

```
Traceback (most recent call last):
  File "C:\Users\Lenovo\track2-techjam\techjam-track2-agent-2026\harness\_run_patch.py", line 98, in main
    result = runners.train_fm(splits, checkpoint_path=args.checkpoint,
                              **config)
  File "C:\Users\Lenovo\track2-techjam\techjam-track2-agent-2026\harness\models\runners.py", line 252, in train_fm
    enc, dim, feature_names = fregistry.encode_with_features(
                              ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        splits, features, prior=feature_prior, n_buckets=feature_buckets)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^
```

_Cost: 110s · 98647 in / 13234 out tokens._

### Iteration 4 — KEPT

**Hypothesis.** Both prior objectives treated every misordered pair as equally costly, so neither targeted nDCG@5's position discount, and neither beat pointwise logloss. LambdaRank fixes this: each positive-negative pair's gradient is scaled by the |ΔnDCG| that swapping them would cause, so pairs near the top of a list dominate while deep pairs still contribute (which GAUC needs). Lists are built as (user_id, date), whose measured mean size of 5.77 matches validation's 5.58, so training lists and scoring lists optimise the same shape. This is a genuinely different weighting of the pair space, not a retune of BPR.

**Target stage.** `objective`  
**Patch kind.** `lambdarank_delta_ndcg_pairwise_loss`  

| metric | value |
|---|---|
| val_gauc | 0.6589 |
| val_ndcg5 | 0.5314 |
| val_primary | 0.5952 |
| train_primary | 0.6037 |
| gap | 0.0085 |

**Decision.** KEPT — val_primary 0.5952 beats the best

_Cost: 238s · 125778 in / 18829 out tokens._

### Iteration 5 — KEPT

**Hypothesis.** Every ranking loss tried so far scores below the pointwise baseline, and the field norms say why: user_id 0.131, video_id 0.163, author_id 0.162 against tab 0.883. The ID embeddings that carry the user x video cross have collapsed, because a pure listwise/pairwise gradient is zero on any list with one row or with no mixed labels, and with median list size 3 that is a large share of training rows. So the ranking objective is learning from a fraction of the data and leaning on the two dense low-cardinality fields. Mix the two: a per-row pointwise logloss term keeps dense gradient on every ID, a softmax term supplies the rank alignment.

**Target stage.** `objective`  
**Patch kind.** `hybrid_pointwise_listwise_loss`  

| metric | value |
|---|---|
| val_gauc | 0.6652 |
| val_ndcg5 | 0.5349 |
| val_primary | 0.6001 |
| train_primary | 0.6151 |
| gap | 0.0150 |

**Decision.** KEPT — val_primary 0.6001 beats the best

_Cost: 109s · 153190 in / 23258 out tokens._

### Iteration 6 — KEPT

**Hypothesis.** Every loss variant tried has landed at or below the pointwise baseline, and the field norms explain why: under a within-list objective, user_id cannot vary inside a list, so its embedding starved (0.320) while the dense within-list fields dur_bucket (1.332) and tab (1.227) absorbed everything. The model is close to a global 'short videos get completed' prior with almost no personalisation. So revert to pointwise and attack the weak user representation directly: back user_id off onto a dense, causal user-propensity field that crosses with duration, giving 'this user tolerates long videos'. Pair it with duration measured against the user's own slate, which breaks dur_bucket ties inside a list.

**Target stage.** `features`  
**Patch kind.** `user_propensity_backoff_plus_slate_relative_duration`  

| metric | value |
|---|---|
| val_gauc | 0.6677 |
| val_ndcg5 | 0.5357 |
| val_primary | 0.6017 |
| train_primary | 0.6272 |
| gap | 0.0255 |

**Decision.** KEPT — val_primary 0.6017 beats the best

_Cost: 284s · 180673 in / 31918 out tokens._

### Iteration 7 — rejected

**Hypothesis.** Field norms say the model ranks mostly by dur_bucket (1.46) and tab (1.17); video_id (0.24) is barely used, so item quality is under-represented. Raw historical video rates duplicate that duration prior — long_view is mechanically a completion event, so a video's rate is largely its duration's rate — which likely explains why the earlier history encodings did nothing. This feature subtracts the duration-bucket prior and keeps only the orthogonal part: quality net of duration, shrunk toward the author when the video is thin. It varies within a list, and the FM crosses it with user_id, giving per-user quality sensitivity that the noisy 42-impression user embeddings cannot express. Loss left pointwise deliberately: with 5.6-item lists, ranking surrogates have already shown nothing.

**Target stage.** `features`  
**Patch kind.** `duration_residual_item_quality_features`  

| metric | value |
|---|---|
| val_gauc | 0.6643 |
| val_ndcg5 | 0.5348 |
| val_primary | 0.5995 |
| train_primary | 0.6348 |
| gap | 0.0352 |

**Decision.** rejected — val_primary 0.5995 does not beat the best

_Cost: 283s · 208172 in / 40043 out tokens._

### Iteration 8 — KEPT

**Hypothesis.** Six iterations of objective and feature changes all landed within noise, and the field norms show why: the ID embeddings are weak (user_id 0.34, video_id 0.26) while dur_bucket and tab dominate. With ~43 training rows per user, each seed's user and item embeddings are dominated by fit noise, so a single run's within-user ordering is partly random rather than systematically wrong. No loss rewrite can remove estimation variance. Rank-averaging several independently seeded models cancels that noise directly, and because the averaging is done on within-user ranks it is non-monotone across members and so genuinely changes the ordering, unlike per-user rescaling of one model. Diagnostics will show blend versus best member, settling whether variance or bias is the binding constraint.

**Target stage.** `ensemble`  
**Patch kind.** `seed_ensemble_within_user_rank_blend`  

| metric | value |
|---|---|
| val_gauc | 0.6688 |
| val_ndcg5 | 0.5364 |
| val_primary | 0.6026 |

**Decision.** KEPT — val_primary 0.6026 beats the best

_Cost: 712s · 235610 in / 43969 out tokens._

### Iteration 9 — rejected

**Hypothesis.** Diagnostics say the model barely uses the ID crosses: dur_bucket norm 1.578 and tab 1.315 versus user_id 0.353, video_id 0.298. So it ranks mostly by duration and surface, and the user x video cross that should carry preference is under-trained, not saturated. Two causes: easy rows already explained by duration dominate the pointwise gradient, and rare IDs get too few Adam updates before validation peaks (epoch 7 of 11). Fix both at once: focal down-weighting of well-fit rows so gradient flows to residual ID structure, plus 4x smaller batches and a longer schedule so sparse embeddings actually grow. Three seeds blended by within-user rank.

**Target stage.** `objective`  
**Patch kind.** `focal_hard_example_logloss_with_sparse_id_schedule`  

| metric | value |
|---|---|
| val_gauc | 0.6678 |
| val_ndcg5 | 0.5352 |
| val_primary | 0.6015 |

**Decision.** rejected — val_primary 0.6015 does not beat the best

_Cost: 801s · 263031 in / 53078 out tokens._
