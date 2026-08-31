# Run `swetha-1`

_Autonomous ML research agent, TikTok TechJam 2026 Track 2._

Every number below was computed by the harness. The hypotheses are the agent's; the measurements are not.

No hidden-test metric appears in this file, or in any log the agent can read. See CLAUDE.md section 5.

## Summary

- Iterations recorded: **6**
- Best validation primary: **0.6020** (iteration 6)
- Kept: 3 | rejected: 3 | failed: 0

## Iterations

### Iteration 1 — KEPT

**Hypothesis.** The baseline optimises pointwise logloss while both scored metrics rank within a user's list, so its gradient spends capacity on calibrating absolute probabilities that within-list ordering discards. A listwise softmax (ListNet top-1) over each list makes the objective purely ordinal and concentrates gradient on the top of the list, which is what nDCG@5 measures, while still separating positives from negatives per list for GAUC. Grouping choice is the real design decision: measured train lists are 43.5 mean by user_id versus 5.77 by user_id+date, and valid lists average 5.58, so user_id+date matches evaluation list length almost exactly. Lists with no positives give zero gradient and are dropped.

**Target stage.** `objective`  
**Patch kind.** `listwise_softmax_ce`  

| metric | value |
|---|---|
| val_gauc | 0.6523 |
| val_ndcg5 | 0.5289 |
| val_primary | 0.5906 |
| train_primary | 0.6006 |
| gap | 0.0100 |

**Decision.** KEPT — val_primary 0.5906 beats the best

_Cost: 149s · 27096 in / 2380 out tokens._

### Iteration 2 — rejected

**Hypothesis.** Listwise softmax over ~5.8-item lists gave nothing, plausibly because with a 31% positive rate most lists have several positives, so softmax cross-entropy just splits mass among them and supplies no explicit ordering pressure between a positive and a negative. Both scored metrics are pairwise/top-heavy. I therefore try LambdaRank: every positive-negative pair inside a (user_id,date) list gets a BPR logistic term weighted by |delta nDCG@5| from swapping them, plus a constant floor so pairs below rank 5 still push GAUC. This directly optimises the primary metric's two components, weighting the top of each list where nDCG@5 lives, and unlike softmax it cannot be satisfied by mass-splitting among positives.

**Target stage.** `objective`  
**Patch kind.** `lambdarank_pairwise_loss`  

| metric | value |
|---|---|
| val_gauc | 0.6517 |
| val_ndcg5 | 0.5287 |
| val_primary | 0.5902 |
| train_primary | 0.6006 |
| gap | 0.0104 |

**Decision.** rejected — val_primary 0.5902 does not beat the best

_Cost: 244s · 54528 in / 6904 out tokens._

### Iteration 3 — KEPT

**Hypothesis.** Pairwise and listwise objectives both landed flat. Likely cause: under user_id+date the median list is 3 and many lists are single-item or homogeneous, so a pure ranking loss gives zero gradient on a large share of rows — effective training data shrinks and the calibrated duration/tab signal is lost. Fix both at once: (a) group by user_id, where train lists average 43.5 rows so nearly every list has both classes and yields many pairs, and (b) keep a pointwise logloss term on every row so degenerate lists still train the embeddings, adding a sampled BPR term on top for the ranking alignment GAUC rewards. Hybrid, not replacement.

**Target stage.** `objective`  
**Patch kind.** `hybrid_pointwise_plus_sampled_bpr_user_lists`  

| metric | value |
|---|---|
| val_gauc | 0.6677 |
| val_ndcg5 | 0.5356 |
| val_primary | 0.6017 |
| train_primary | 0.6491 |
| gap | 0.0475 |

**Decision.** KEPT — val_primary 0.6017 beats the best

_Cost: 219s · 81977 in / 11770 out tokens._

### Iteration 4 — rejected

**Hypothesis.** Three objective variants all landed at 0.6017, so loss shape alone is not the lever; the model wastes capacity on signal that cannot move a within-user metric (user_id embedding norm 0.265, train-valid gap 0.0475, peak at epoch 5). Any component constant across a list is provably worthless, yet every loss tried still fits per-list intercepts. I center logits within each list before the logistic, making the objective exactly invariant to per-list shifts: constant-within-user structure gets zero gradient. Lists are (user_id, date), mean 5.77, matching evaluation's 5.58 rather than 43.5. Non-discriminative lists are masked (they cannot affect GAUC or nDCG). Three seeds blended by within-user rank; diagnostics still report each member solo.

**Target stage.** `objective`  
**Patch kind.** `list_centered_logistic_loss_plus_seed_blend`  

| metric | value |
|---|---|
| val_gauc | 0.6626 |
| val_ndcg5 | 0.5338 |
| val_primary | 0.5982 |

**Decision.** rejected — val_primary 0.5982 does not beat the best

_Cost: 634s · 109452 in / 17112 out tokens._

### Iteration 5 — rejected

**Hypothesis.** Diagnostics show the model is effectively a duration/tab ranker: dur_bucket embedding norm 1.027 vs user_id 0.106 and video_id 0.153, so the ID crosses that should carry personalisation are near-dead. Every objective tried so far (logistic, softmax CE, LambdaRank, BPR) saturates once a pair is ordered, supplying no pressure to grow rare ID embeddings, which is why GAUC has not moved. WARP's fixed-margin hinge never saturates: it keeps demanding absolute separation and weights each positive by log(1+#violating negatives), concentrating gradient on positives buried deep in the list (nDCG@5) while still being pairwise (GAUC). Long user_id lists (mean 43.5) make the rank estimate meaningful.

**Target stage.** `objective`  
**Patch kind.** `warp_rank_weighted_hinge`  

| metric | value |
|---|---|
| val_gauc | 0.6670 |
| val_ndcg5 | 0.5353 |
| val_primary | 0.6012 |
| train_primary | 0.6305 |
| gap | 0.0294 |

**Decision.** rejected — val_primary 0.6012 does not beat the best

_Cost: 269s · 136861 in / 25550 out tokens._

### Iteration 6 — KEPT

**Hypothesis.** Five loss experiments in a row moved nothing beyond noise, so the objective is not the binding constraint: the fitted embeddings are. With 27k user and 7.5k video embeddings learned from 1.14M rows, each ID's vector is estimated from ~40 rows, so per-seed init/SGD variance is large relative to signal. Averaging independent fits cancels that variance without adding capacity, which the organisers showed is not the bottleneck. Because ranking is within-user, I rank-normalise each member inside a user's list before averaging so no member's score scale dominates. Prior seed blending was confounded with a novel loss; this isolates variance reduction on the known-best pointwise configuration. Diagnostics report member-vs-blend, so a null result is still informative.

**Target stage.** `ensemble`  
**Patch kind.** `seed_ensemble_rank_averaged_pointwise`  

| metric | value |
|---|---|
| val_gauc | 0.6677 |
| val_ndcg5 | 0.5362 |
| val_primary | 0.6020 |

**Decision.** KEPT — val_primary 0.6020 beats the best

_Cost: 492s · 164413 in / 27846 out tokens._
