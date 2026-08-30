# Run `agent-explore4`

_Autonomous ML research agent, TikTok TechJam 2026 Track 2._

Every number below was computed by the harness. The hypotheses are the agent's; the measurements are not.

No hidden-test metric appears in this file, or in any log the agent can read. See CLAUDE.md section 5.

## Summary

- Iterations recorded: **7**
- Best validation primary: **0.6034** (iteration 4)
- Kept: 3 | rejected: 4 | failed: 0

## Iterations

### Iteration 1 — KEPT

**Hypothesis.** The baseline optimises pointwise logloss while both metrics score within-user order; GAUC is literally a pairwise statistic, so the objective is misaligned. I replace it with all-pairs BPR over each list, weighted so every list contributes equally, plus a 0.1 pointwise anchor so rows in degenerate lists still receive gradient. Grouping is (user_id, date): measured train mean 5.77 items vs valid 5.58, an almost exact match, whereas user_id alone is 43.5 and would optimise lists 8x longer than evaluation. Only discriminative lists produce pairs (57.8% of valid users), which is exactly the set GAUC is computed over. Expect GAUC to move more than nDCG@5.

**Target stage.** `objective`  
**Patch kind.** `pairwise_bpr_loss`  

| metric | value |
|---|---|
| val_gauc | 0.6497 |
| val_ndcg5 | 0.5279 |
| val_primary | 0.5888 |
| train_primary | 0.6084 |
| gap | 0.0196 |

**Decision.** KEPT — val_primary 0.5888 beats the best

_Cost: 1266s · 26582 in / 8588 out tokens._

### Iteration 2 — KEPT

**Hypothesis.** BPR scored 0.5888, below the 0.6016 pointwise baseline, and the field norms explain why: video_id 0.169 and user_id 0.220 versus dur_bucket 0.864 mean the ID cross that drives the FM stopped training. Under user_id+date lists averaging 4.3 rows with a 0.31 positive rate, a large share of lists are all-positive or all-negative and emit zero pairwise gradient, so most rows never update their embeddings. Fix both: score lists by user_id (mean 43.5, almost always mixed) with a listwise softmax cross-entropy against the normalised label distribution, and keep a pointwise anchor term so homogeneous lists still train IDs. Larger batches keep each user's rows in one list.

**Target stage.** `objective`  
**Patch kind.** `listwise_softmax_with_pointwise_anchor`  

| metric | value |
|---|---|
| val_gauc | 0.6659 |
| val_ndcg5 | 0.5349 |
| val_primary | 0.6004 |
| train_primary | 0.6801 |
| gap | 0.0797 |

**Decision.** KEPT — val_primary 0.6004 beats the best

_Cost: 139s · 53505 in / 15206 out tokens._

### Iteration 3 — rejected

**Hypothesis.** Both tried objectives grouped a user's whole 43.5-item history into one list, 7.8x longer than the 5.6-item evaluation lists, and both landed at 0.600. Two things are wrong. First, list geometry: gradients are spread over items seen across 14 different days, so the model never optimises the ordering problem it is scored on. Second, neither objective knows the metric's position weighting — BPR treats every pair equally and softmax only chases the top. LambdaRank weights each within-list pair by the |dNDCG| that swapping it would cause, which is directly the nDCG@5 gradient while remaining pairwise, the form GAUC measures. Applied over (user_id, date) lists, whose mean 5.77 matches validation's 5.58.

**Target stage.** `objective`  
**Patch kind.** `lambdarank_ndcg_pairwise_daily_lists`  

| metric | value |
|---|---|
| val_gauc | 0.6568 |
| val_ndcg5 | 0.5320 |
| val_primary | 0.5944 |
| train_primary | 0.6009 |
| gap | 0.0065 |

**Decision.** rejected — val_primary 0.5944 does not beat the best

_Cost: 166s · 80443 in / 20339 out tokens._

### Iteration 4 — KEPT

**Hypothesis.** Three objective rewrites moved nothing, and the diagnostics say why: train-valid gap is only +0.0065, so the model underfits rather than overfits, and ID embedding norms (user 0.10, video 0.14) are 5-7x smaller than tab/dur_bucket (0.79/0.74) — the user x video cross that carries the ranking signal is barely trained, while single-seed noise (~0.001) swamps the small real differences. So I change stage: reduce variance by rank-averaging five seeds, and give rare IDs 4x more Adam updates per epoch via batch 2048. Rank-normalising within user before blending is non-monotone across members, so unlike single-model normalisation it can genuinely reorder. Member diagnostics separate the batch effect from the blend effect.

**Target stage.** `ensemble`  
**Patch kind.** `seed_ensemble_rank_blend_small_batch`  

| metric | value |
|---|---|
| val_gauc | 0.6700 |
| val_ndcg5 | 0.5367 |
| val_primary | 0.6034 |

**Decision.** KEPT — val_primary 0.6034 beats the best

_Cost: 552s · 107383 in / 26965 out tokens._

### Iteration 5 — rejected

**Hypothesis.** Every ranking objective so far was fed short lists: lambdarank used (user,date) lists whose median is 3, where a large share of lists are all-positive or all-negative and therefore contribute no pair signal at all. Grouping by user_id gives mean 43.5 items per list at a 0.31 positive rate, so nearly every list is discriminative and the pair count per user rises ~50x. GAUC is literally per-user AUC, so summing every pos-neg pair inside a user's full history and normalising each user to equal weight optimises the metric directly rather than a sampled proxy. The weak ID embedding norms (user 0.355, video 0.241 vs dur_bucket 1.782) say the user-item cross is underfit; dense within-user contrast is the signal that trains it.

**Target stage.** `objective`  
**Patch kind.** `all_pairs_ranknet_full_user_history`  

| metric | value |
|---|---|
| val_gauc | 0.6664 |
| val_ndcg5 | 0.5356 |
| val_primary | 0.6010 |
| train_primary | 0.6308 |
| gap | 0.0298 |

**Decision.** rejected — val_primary 0.6010 does not beat the best

_Cost: 225s · 134301 in / 37193 out tokens._

### Iteration 6 — rejected

**Hypothesis.** Every objective tried so far spreads gradient evenly over all pairs in a 43-row training list, but both metrics only care about the head: evaluation lists average 5.6 items, so nDCG@5 is essentially the whole list, and GAUC is decided by the few negatives a model actually ranks high. All-pairs RankNet moved nothing (+0.0000) because most of its gradient goes to easy, already-correct pairs — which also explains the train/val gap of +0.0298 and the epoch-5 peak. I propose a WARP-style top-K hinge: within each user's full history, only negatives currently inside the top-5 (or the single hardest negative) form pairs, and only violating pairs within a margin get gradient. Easy pairs get exactly zero, focusing capacity on the head instead of over-separating.

**Target stage.** `objective`  
**Patch kind.** `topk_hard_negative_margin_loss`  

| metric | value |
|---|---|
| val_gauc | 0.6648 |
| val_ndcg5 | 0.5348 |
| val_primary | 0.5998 |
| train_primary | 0.6381 |
| gap | 0.0384 |

**Decision.** rejected — val_primary 0.5998 does not beat the best

_Cost: 194s · 161280 in / 45277 out tokens._

### Iteration 7 — rejected

**Hypothesis.** Three loss rewrites failed, so I attack sampling instead. Under pointwise logloss every row carries equal gradient, so training is dominated by long user histories (mean 43.5 rows vs 5.6 at evaluation) and by degenerate lists: 30.3% of valid users are all-negative and 11.9% all-positive, and a list with one label class carries zero within-list ordering information yet supplies most of the gradient. I reweight rows to match the metric's own accounting: lists defined as (user_id, date) to mirror evaluation list length, each contributing equal total mass, positives and negatives balanced within a list, and degenerate lists downweighted to 0.3 rather than dropped (they still anchor embeddings). Should lift GAUC, which is computed over discriminative users only.

**Target stage.** `sampling`  
**Patch kind.** `metric_matched_list_balanced_sample_weights`  

| metric | value |
|---|---|
| val_gauc | 0.6668 |
| val_ndcg5 | 0.5358 |
| val_primary | 0.6013 |
| train_primary | 0.6308 |
| gap | 0.0295 |

**Decision.** rejected — val_primary 0.6013 does not beat the best

_Cost: 115s · 188312 in / 49053 out tokens._

## Events

| when | kind | detail |
|---|---|---|
| 2026-08-30T15:17:45 | `recovery` | timeout; retrying once at a 30% subsample |
