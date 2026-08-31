# Run `swetha-4`

_Autonomous ML research agent, TikTok TechJam 2026 Track 2._

Every number below was computed by the harness. The hypotheses are the agent's; the measurements are not.

No hidden-test metric appears in this file, or in any log the agent can read. See CLAUDE.md section 5.

## Summary

- Iterations recorded: **4**
- Best validation primary: **0.6008** (iteration 4)
- Kept: 3 | rejected: 1 | failed: 0

## Iterations

### Iteration 1 — KEPT

**Hypothesis.** The baseline optimises pointwise logloss while both metrics rank within a user's short list, so gradient effort is spent calibrating absolute probabilities rather than ordering. Listwise softmax cross-entropy over a user's list directly maximises the normalised score mass on the positives, which is what nDCG@5 rewards, and still enforces every within-list ordering through the shared partition function. List construction is the real choice: measured train lists are 43.5 items under user_id but 5.77 under user_id+date, against 5.58 in valid, so user_id+date matches evaluation list length almost exactly. I expect nDCG@5 to move more than GAUC.

**Target stage.** `objective`  
**Patch kind.** `listwise_softmax_loss`  

| metric | value |
|---|---|
| val_gauc | 0.6572 |
| val_ndcg5 | 0.5309 |
| val_primary | 0.5941 |
| train_primary | 0.5932 |
| gap | -0.0009 |

**Decision.** KEPT — val_primary 0.5941 beats the best

_Cost: 224s · 27096 in / 3250 out tokens._

### Iteration 2 — KEPT

**Hypothesis.** Listwise softmax over (user,date) lists moved nothing, and diagnostics show user_id embeddings barely participate (norm 0.072 vs tab 0.771) — the model is ranking mostly by tab/duration priors, which are near-constant within a short list. Two changes together: (a) a pairwise BPR objective, which is the exact quantity GAUC measures, and (b) group by user_id, not user_id+date. Evaluation groups by user_id across the whole 7-day valid window; its 5.6 mean is a density artefact, not a grouping rule. Training on 43-row user lists gives ~100x more within-user pairs and forces cross-day user preference into the user embedding rather than within-session position effects.

**Target stage.** `objective`  
**Patch kind.** `pairwise_bpr_all_pairs_user_grouping`  

| metric | value |
|---|---|
| val_gauc | 0.6629 |
| val_ndcg5 | 0.5328 |
| val_primary | 0.5979 |
| train_primary | 0.6333 |
| gap | 0.0354 |

**Decision.** KEPT — val_primary 0.5979 beats the best

_Cost: 148s · 54528 in / 6603 out tokens._

### Iteration 3 — rejected

**Hypothesis.** All-pairs BPR on user_id lists (mean 43.5 vs 5.6 at eval) weights heavy users quadratically and treats every inversion equally, so it never targets the top of the short lists nDCG@5 actually scores; it landed below baseline and overfits (train-valid gap 0.035, peak at epoch 3). I change two things at once: group by user_id+date, whose mean list length 5.77 matches the evaluation 5.58, and weight each pair by its |delta nDCG| so swaps near the top dominate, with per-list normalisation so every list counts equally, matching nDCG's equal-user weighting while staying pairwise for GAUC. Slightly higher L2 to counter the observed overfit.

**Target stage.** `objective`  
**Patch kind.** `lambdarank_pairwise_per_date_lists`  

| metric | value |
|---|---|
| val_gauc | 0.6577 |
| val_ndcg5 | 0.5317 |
| val_primary | 0.5947 |
| train_primary | 0.5932 |
| gap | -0.0015 |

**Decision.** rejected — val_primary 0.5947 does not beat the best

_Cost: 266s · 81980 in / 10374 out tokens._

### Iteration 4 — KEPT

**Hypothesis.** Every pure ranking loss so far has landed below the pointwise baseline, and the embedding-norm diagnostic explains why: user_id 0.054 and video_id 0.105 are barely learning, while dur_bucket/tab dominate. Listwise and pairwise objectives give exactly zero gradient on all-positive and all-negative lists — 42% of valid users, likely a similar share of train day-lists — so the sparse ID embeddings lose most of their update signal and the model falls back on dense fields. A hybrid keeps full pointwise logloss on every row (trains IDs on all data) and adds a per-list softmax cross-entropy term only on discriminative lists (aligns ordering where the metric is actually decided). Expect gains in both GAUC and nDCG@5.

**Target stage.** `objective`  
**Patch kind.** `hybrid_pointwise_plus_discriminative_listwise`  

| metric | value |
|---|---|
| val_gauc | 0.6666 |
| val_ndcg5 | 0.5351 |
| val_primary | 0.6008 |
| train_primary | 0.6079 |
| gap | 0.0071 |

**Decision.** KEPT — val_primary 0.6008 beats the best

_Cost: 175s · 109435 in / 14520 out tokens._
