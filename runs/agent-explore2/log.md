# Run `agent-explore2`

_Autonomous ML research agent, TikTok TechJam 2026 Track 2._

Every number below was computed by the harness. The hypotheses are the agent's; the measurements are not.

No hidden-test metric appears in this file, or in any log the agent can read. See CLAUDE.md section 5.

## Summary

- Iterations recorded: **6**
- Best validation primary: **0.6010** (iteration 6)
- Kept: 3 | rejected: 3 | failed: 0

## Iterations

### Iteration 1 — KEPT

**Hypothesis.** The baseline optimises pointwise logloss while both scored metrics rank within a user's list; GAUC is literally the per-user probability that a positive outranks a negative. Pointwise training spends capacity on calibrating absolute rates, which is invariant-free for within-list order. I replace the objective with all-pairs BPR over each list, which directly optimises the pairwise ordering GAUC measures and pushes positives above negatives near the top for nDCG@5. Lists are built as user_id+date: measured train mean 5.77 items versus valid 5.58, so training lists match evaluation lists almost exactly, while user_id alone (43.5) would optimise an 8x longer list than is ever scored.

**Target stage.** `objective`  
**Patch kind.** `pairwise_bpr_loss`  

| metric | value |
|---|---|
| val_gauc | 0.6581 |
| val_ndcg5 | 0.5311 |
| val_primary | 0.5946 |
| train_primary | 0.6026 |
| gap | 0.0080 |

**Decision.** KEPT — val_primary 0.5946 beats the best

_Cost: 180s · 26368 in / 5024 out tokens._

### Iteration 2 — rejected

**Hypothesis.** BPR moved nothing because every in-list pair is weighted equally, so gradient is spent on pairs deep in the list that neither metric rewards, and the FM's ID embeddings (norms 0.14-0.17 vs 0.90 for tab) barely move. Both scored metrics are rank-position sensitive: nDCG@5 only sees the top of a 5.8-item list. LambdaRank weights each positive-negative pair by the |delta nDCG| that swapping them causes, concentrating gradient on top-of-list inversions while remaining a pairwise objective aligned with GAUC. Lists are built as user_id+date (mean 5.77) to match evaluation lists (5.58); all-positive and all-negative lists contribute zero pairs, matching GAUC's own denominator.

**Target stage.** `objective`  
**Patch kind.** `lambdarank_ndcg_weighted_pairwise`  

| metric | value |
|---|---|
| val_gauc | 0.6546 |
| val_ndcg5 | 0.5307 |
| val_primary | 0.5927 |
| train_primary | 0.5936 |
| gap | 0.0010 |

**Decision.** rejected — val_primary 0.5927 does not beat the best

_Cost: 152s · 53072 in / 10382 out tokens._

### Iteration 3 — KEPT

**Hypothesis.** Both tried objectives were pairwise over (user_id, date) lists and both landed flat. The diagnostic that matters is field participation: user_id embedding norm 0.029, video_id 0.077, while dur_bucket and tab dominate. The model is ranking by duration/tab priors, not by the user x item cross the FM exists to learn. Date-sliced lists (median 3, many degenerate) give each user's embedding almost no contrastive signal. Fix both at once: group by user_id so every user contributes tens of contrasts, but chunk each user's rows into random ~6-item sublists per epoch so the optimised list length matches evaluation (5.6), and use listwise softmax cross-entropy, which is top-heavy and aligned with nDCG@5 rather than only GAUC.

**Target stage.** `objective`  
**Patch kind.** `listwise_softmax_over_resampled_sublists`  

| metric | value |
|---|---|
| val_gauc | 0.6660 |
| val_ndcg5 | 0.5354 |
| val_primary | 0.6007 |
| train_primary | 0.6304 |
| gap | 0.0297 |

**Decision.** KEPT — val_primary 0.6007 beats the best

_Cost: 127s · 79776 in / 14472 out tokens._

### Iteration 4 — rejected

**Hypothesis.** Four straight pure-ranking objectives all landed at 0.6007, at or just under the pointwise baseline, so the loss family is not the problem: discarding absolute propensity is. GAUC/nDCG ignore ordering inside all-positive and all-negative lists (42% of validation users), and under user_id+date grouping (mean 5.77, median 3, p10 1) most lists are degenerate, so softmax and BPR give them zero gradient and throw away that data entirely. I blend a per-list softmax over mixed lists with pointwise logloss over every row: mixed lists get the ranking signal, degenerate and singleton lists still teach calibration that transfers to mixed lists.

**Target stage.** `objective`  
**Patch kind.** `hybrid_listwise_pointwise_loss`  

| metric | value |
|---|---|
| val_gauc | 0.6631 |
| val_ndcg5 | 0.5344 |
| val_primary | 0.5988 |
| train_primary | 0.6162 |
| gap | 0.0175 |

**Decision.** rejected — val_primary 0.5988 does not beat the best

_Cost: 144s · 106520 in / 19961 out tokens._

### Iteration 5 — rejected

**Hypothesis.** Both metrics are invariant to any list-level offset, so capacity spent on absolute calibration is wasted — and the field norms (dur_bucket 1.27, tab 1.25 vs user_id 0.36, video_id 0.22) say the model is mostly learning a list-independent duration/tab prior rather than within-list contrast. Softmax also fixes list-level offsets, but it forces the items to compete for one unit of probability mass, which is wrong when lists average 5.6 items at a 0.31 positive rate (multi-positive lists are the norm). A logistic loss on scores centred within each list keeps the offset-invariance without the single-winner assumption, gives every item gradient, and weights each list equally to match per-user metric averaging.

**Target stage.** `objective`  
**Patch kind.** `within_list_centered_logistic_loss`  

| metric | value |
|---|---|
| val_gauc | 0.6562 |
| val_ndcg5 | 0.5320 |
| val_primary | 0.5941 |
| train_primary | 0.5963 |
| gap | 0.0022 |

**Decision.** rejected — val_primary 0.5941 does not beat the best

_Cost: 144s · 133263 in / 25802 out tokens._

### Iteration 6 — KEPT

**Hypothesis.** We have never changed which rows carry gradient weight. In validation, 42% of users' lists are all-positive or all-negative; those lists can only teach a global bias, which is monotone within a list and therefore worth exactly zero to GAUC and nDCG@5, yet they currently consume most of the pointwise gradient. Meanwhile embedding norms show user_id (0.033) and video_id (0.068) barely participate, i.e. the ID cross is starved. This is a sampling change, not a new objective: keep plain logloss, but give each label-mixed (user_id, date) list equal total weight split evenly between its positives and negatives, and down-weight homogeneous lists to 0.15. Gradient budget moves to rows that can actually reorder a list.

**Target stage.** `sampling`  
**Patch kind.** `list_mixedness_and_class_balanced_sample_weighting`  

| metric | value |
|---|---|
| val_gauc | 0.6663 |
| val_ndcg5 | 0.5358 |
| val_primary | 0.6010 |
| train_primary | 0.6254 |
| gap | 0.0244 |

**Decision.** KEPT — val_primary 0.6010 beats the best

_Cost: 100s · 160057 in / 29359 out tokens._
