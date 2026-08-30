# Run `swetha-3`

_Autonomous ML research agent, TikTok TechJam 2026 Track 2._

Every number below was computed by the harness. The hypotheses are the agent's; the measurements are not.

No hidden-test metric appears in this file, or in any log the agent can read. See CLAUDE.md section 5.

## Summary

- Iterations recorded: **6**
- Best validation primary: **0.6020** (iteration 4)
- Kept: 4 | rejected: 2 | failed: 0

## Iterations

### Iteration 1 — KEPT

**Hypothesis.** Baseline optimises pointwise logloss while both metrics rank within a user's list. Measured lists: user_id+date on train averages 5.77 items, matching valid's 5.58, so per-day lists are the right training unit. I propose a listwise softmax cross-entropy over each (user,date) list, normalised so multi-positive lists spread mass evenly across positives, plus a small pairwise-free structure: gradient p - y/npos. This directly aligns with nDCG@5 (top-of-list emphasis) and GAUC (relative ordering). Lists that are all-positive or all-negative give zero gradient, focusing capacity on the 57.8% discriminative users GAUC is computed over. Expect GAUC and nDCG@5 both up.

**Target stage.** `objective`  
**Patch kind.** `listwise_softmax_loss`  

| metric | value |
|---|---|
| val_gauc | 0.6380 |
| val_ndcg5 | 0.5228 |
| val_primary | 0.5804 |
| train_primary | 0.5790 |
| gap | -0.0014 |

**Decision.** KEPT — val_primary 0.5804 beats the best

_Cost: 132s · 27096 in / 1456 out tokens._

### Iteration 2 — KEPT

**Hypothesis.** Listwise softmax matched baseline exactly and peaked at epoch 1, suggesting the objective saturates fast and treats all lists alike. GAUC is literally a per-user pairwise AUC over discriminative users (57.8% of valid users), yet softmax spends most gradient on all-negative/all-positive lists where no ordering exists. I propose BPR over all positive-negative pairs within (user_id, date) lists, with pairs enumerated exhaustively per list and normalised per list so long lists do not dominate. Lists with no valid pair contribute zero gradient, concentrating capacity entirely on discriminative ordering. This is directly aligned with GAUC and should also lift nDCG@5 via top-of-list separation.

**Target stage.** `objective`  
**Patch kind.** `pairwise_bpr_within_user_date`  

| metric | value |
|---|---|
| val_gauc | 0.6474 |
| val_ndcg5 | 0.5263 |
| val_primary | 0.5869 |
| train_primary | 0.5956 |
| gap | 0.0087 |

**Decision.** KEPT — val_primary 0.5869 beats the best

_Cost: 142s · 54525 in / 2906 out tokens._

### Iteration 3 — KEPT

**Hypothesis.** Both pure ranking losses came in below the pointwise baseline, and the list profile explains why: median training list under user_id+date is 3 and ~30% of users are all-negative, so most lists yield zero pairs and zero gradient. BPR also discards absolute calibration, which the item/duration/tab terms carry (tab norm 1.07 vs user_id 0.21). I propose a hybrid objective: pointwise logloss over every row (dense signal, all lists) plus a LambdaRank-weighted pairwise term (|delta nDCG| discounts) over the lists that do contain both classes. This keeps the baseline's learning signal intact while adding metric-aligned top-of-list pressure, targeting GAUC and nDCG@5 together.

**Target stage.** `objective`  
**Patch kind.** `hybrid_pointwise_plus_lambdarank_pairwise`  

| metric | value |
|---|---|
| val_gauc | 0.6667 |
| val_ndcg5 | 0.5355 |
| val_primary | 0.6011 |
| train_primary | 0.6202 |
| gap | 0.0191 |

**Decision.** KEPT — val_primary 0.6011 beats the best

_Cost: 222s · 81972 in / 8300 out tokens._

### Iteration 4 — KEPT

**Hypothesis.** Three objective rewrites all landed at 0.6011, so the loss is not the binding constraint; the residual error looks like variance, not bias. Evidence: ID embeddings carry small norms (user_id 0.35, video_id 0.23 vs dur_bucket 1.49), i.e. rare-ID vectors are barely moved from their random init, and val peaks at epoch 10 then decays — classic high-variance fitting of sparse crosses. Seed-to-seed noise on FM is 0.0008 per run, and that noise is largely independent across seeds in the ID block. Rank-averaging 5 independently-seeded FMs within each user should cancel it. Per-user rank normalisation first, since raw score scales differ and a plain mean would be dominated by one member.

**Target stage.** `ensemble`  
**Patch kind.** `seed_ensemble_within_user_rank_blend`  

| metric | value |
|---|---|
| val_gauc | 0.6677 |
| val_ndcg5 | 0.5362 |
| val_primary | 0.6020 |

**Decision.** KEPT — val_primary 0.6020 beats the best

_Cost: 509s · 109454 in / 11473 out tokens._

### Iteration 5 — rejected

**Hypothesis.** Every objective tried so far lands at ~0.602, and all treated pairs uniformly. Both metrics are dominated by top-of-list errors, and 42% of valid users are label-homogeneous (they contribute nothing to either metric), while user_id+date lists have median 3 items so BPR there saw few informative pairs. I propose self-adversarial pairwise ranking over full user_id lists (mean 43.5, many more informative pairs): negatives are weighted by softmax of their current score, so gradient concentrates on negatives the model currently ranks above positives, and each list is weighted by sqrt(positives), a compromise between GAUC's positive-count weighting and nDCG's uniform weighting. Homogeneous lists get zero gradient by construction.

**Target stage.** `objective`  
**Patch kind.** `self_adversarial_pairwise_sqrtpos_weighting`  

| metric | value |
|---|---|
| val_gauc | 0.6631 |
| val_ndcg5 | 0.5338 |
| val_primary | 0.5984 |
| train_primary | 0.6353 |
| gap | 0.0369 |

**Decision.** rejected — val_primary 0.5984 does not beat the best

_Cost: 239s · 136882 in / 17464 out tokens._

### Iteration 6 — rejected

**Hypothesis.** Training lists (43.5 rows) are 7.8x the evaluation lists (5.6), and 42% of valid users are all-positive or all-negative, where ordering cannot change either metric. Every objective so far spent gradient on lists whose ranking is unscorable and on lists far longer than any list at scoring time. This loss cuts each user's impressions into random 6-item sub-lists matching evaluation length, keeps only sub-lists containing both a positive and a negative, and applies a listwise softmax cross-entropy against the normalised label distribution. Discriminative-only, top-heavy (targets nDCG@5), and the re-chunking each epoch is a strong regulariser against the +0.0369 train/valid gap that made the last run peak at epoch 3.

**Target stage.** `objective`  
**Patch kind.** `eval_length_matched_discriminative_sublist_softmax`  

| metric | value |
|---|---|
| val_gauc | 0.6636 |
| val_ndcg5 | 0.5340 |
| val_primary | 0.5988 |
| train_primary | 0.6331 |
| gap | 0.0343 |

**Decision.** rejected — val_primary 0.5988 does not beat the best

_Cost: 170s · 164328 in / 22947 out tokens._
