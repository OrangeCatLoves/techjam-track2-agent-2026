# Method corpus

Reference material the agent retrieves from when proposing an experiment.

**This is not a queue.** There is no prescribed order and no pre-written configuration to execute. Each note describes what a method is, when it applies, and what evidence supports it. The agent decides what to try, based on its own analysis of the data and previous results, and writes the code itself.

The problem statement sanctions this explicitly: "The agent is expected to draw on whatever published methods it can find — that is what makes it a research agent."

**Version 2**, rewritten after the organisers published their own ablation results.

---

## Read this first: what has already been measured to fail

The starter kit README publishes ablations the organisers ran themselves. Do not spend iterations reproducing them.

| Tried | Result |
|---|---|
| All 13 CWM feature fields added (`music_id`, `video_type`, `upload_type`, plus 6 user-side coarse buckets) | primary 0.5940 vs 0.5950 for the 5-field baseline. No gain, marginally worse |
| Embedding dimension k = 8 / 16 / 32 | 0.5895 / 0.5902 / 0.5887. Essentially flat |

Their explanation: the `user_id x video_id` cross already absorbs most of the learnable signal, and 1.14M training rows cannot support more capacity.

**The bottleneck is neither features nor model size.** Any proposal that amounts to "add more static feature fields" or "make the model bigger" should be rejected unless there is a specific new argument for why it differs from what was already tested.

`ablation_features.py` in the starter kit reproduces these numbers.

---

## Task shape

Both scored metrics rank **within a single user's logged impression list**, averaging about six items in the evaluation splits.

Consequence, and the organisers measured this directly: **a feature constant across one user's list contributes exactly zero.** Pure user-side first-order terms cannot change within-user ordering. User-side information can only act through crosses with item-side terms.

There is no retrieval, no candidate generation, and no diversity objective. Methods from those stages do not apply.

---

## The single most promising direction: change the loss function

The baseline optimises **pointwise logloss** while both scored metrics are **ranking metrics**. That mismatch is the clearest structural weakness in the baseline, and the organisers rank it first among untried directions.

### Pairwise: BPR

Within a user's impressions, sample (positive, negative) pairs and optimise:

```
L = -log(sigmoid(z_pos - z_neg))
```

GAUC is itself a pairwise measure (probability a random positive outranks a random negative, per user), so a pairwise objective is directly aligned with it. Expect GAUC to respond more than nDCG@5.

### Listwise: softmax over the user's list

Softmax the scores within a user's impression list and take cross-entropy against the labels. Puts more weight on getting the top of the list right, which is what nDCG@5 measures.

### The real design question: how to build the lists

This is not a detail; it is the experiment.

A mismatch between training list length and evaluation list length changes what the objective optimises, so the grouping rule is a real choice with a measurable consequence.

**Candidate groupings:** `user_id` across all training days; `(user_id, date)`; `(user_id, session)` from `time_ms` gaps.

**Measure them. Do not take a remembered figure for any of them.**

```
analyse(kind="list_size_profile", split="train")
analyse(kind="list_size_profile", split="valid")
```

This section previously carried a table of estimated list sizes. One of those estimates was wrong in a way that pointed away from a promising option: it quoted a median as if it were a mean and concluded from that the grouping did not match. The table has been removed rather than corrected, because the tool answers the question directly in under a second and a number in a document cannot be re-derived when it turns out to be stale.

### Implementation note

`FM.step()` in `baseline.py` computes gradients in a clean, isolated way. The gradient of the loss with respect to the logits `z` is the only thing a new objective needs to supply; everything downstream (the `np.add.at` scatter into `gV` and `gW`, the Adam update) is unchanged. A pairwise variant is roughly 40 lines.

This is why the harness exposes `loss_and_grad(z, y, groups)` as a frozen interface.

---

## User history sequences

Completely unused by the baseline. Each user has hundreds to thousands of interactions in the training period, and none of that sequence information enters the model.

DIN and SIM-style interest modelling attend over a user's past item embeddings, weighted by relevance to the candidate item. This is genuinely untouched territory on this benchmark and has the highest ceiling.

It also has the highest build cost. Cheaper approximations worth considering first:

- Mean-pooled embedding of the user's last N watched items, crossed with the candidate item
- Count of prior impressions from the same `author_id`
- Time since the user's last impression
- Whether the user has previously long-viewed anything from this author

All of these are causal target encodings and must respect the expanding-window rule (see below).

---

## Multi-task learning

The log carries `is_click`, `is_like`, `is_follow`, `is_comment`, `is_forward` and `play_time_ms`. These are **never inputs** (they are outcomes of the same impression, so using them as features is leakage) but they are legitimate **auxiliary training targets**.

Predicting several signals jointly over a shared representation, with `long_view` as the main task, can regularise and improve the main task. The tension is between shared parameters, which transfer knowledge, and task-specific parameters, which stop conflicting tasks degrading each other (the "seesaw" problem). MMoE and PLE are the standard architectures.

Cheapest useful version on top of the existing FM: a second output head sharing the same embedding table, trained on `is_click`, with a weighted sum of the two losses.

---

## Watch-time modelling and duration

`long_view = 1` when `play_time_ms >= duration_ms` for videos up to 18 seconds, or `play_time_ms >= 18,000` for longer ones. The label is mechanically tied to duration, and short videos are far more likely to be completed.

Zhao et al., "Counteracting Duration Bias in Video Recommendation via Counterfactual Watch Time," KDD 2024, treats this as a **censored regression** problem: a completed play means the true watch time was truncated by video length, so a one-sided loss is correct and squared error is not.

Caveats from the organisers: CWM pins `torch==1.6.0`, ships no Recall implementation, and evaluates on a rebuilt `long_view2` label. Treat the idea as transferable, not the code.

Note the baseline already uses `dur_bucket` (10 quantile buckets of `duration_ms`) as one of its five fields, so naive duration bucketing is not new. What is new is modelling the censoring explicitly, or using raw duration rather than a bucket.

---

## Causal target encoding — correctness, not a differentiator

Any historical statistic (a video's past long-view rate, a user's affinity for an author) is a target encoding. Computing it over all of training and applying it to training rows leaks each row's own label into its own feature. Validation inflates; test does not move.

The fix is an expanding window: for a training row on date `d`, the statistic uses only dates strictly before `d`. For validation rows, train dates only. The feature API enforces this; a feature reading outside its window raises.

**Why this is hygiene rather than a headline:** the organisers measured that static feature additions produce no gain on this benchmark. Historical statistics are a form of feature addition. They may still help where static ones did not, because they carry temporal information that static fields do not, but the prior should be modest.

Always pair with Bayesian smoothing toward a parent prior and an explicit backoff chain: video -> author -> duration bucket -> global.

---

## Ensembling

Averaging several models usually gains a little and rarely loses.

**Normalise within each user before averaging.** Models produce scores on different scales; a raw average is dominated by whichever has the wider range.

**Per-user normalisation of a single model is a no-op.** It is a monotone transform within the list, so it cannot change GAUC or nDCG@5. The organisers state they measured this: `item_pop x user_bias` and plain `item_pop` scored identically to the last digit.

**Duration-conditioned recalibration is different and is a real lever**, because duration varies across the items in one user's list, so the adjustment is non-monotone within the list.

Tune ensemble weights once. Weights tuned repeatedly against validation will overfit it.

---

## Sample weighting to match the metric

GAUC weights each user by their positive count and counts only users with `0 < positives < impressions`, so high-engagement discriminative users dominate it. nDCG@5 weights all users equally and includes zero-positive users as 0.

On the test set, 27.1% of users are all-negative, 9.2% all-positive, and only 63.7% are discriminative. **GAUC is computed over that 63.7% alone.** Training weight spent on the other 36.3% moves nDCG only.

Weighting training samples by the user's positive count is a cheap experiment that should raise GAUC. Watch whether nDCG falls by more than it gains, and instrument both metrics separately every iteration.

---

## Model architectures

Deprioritised by the organisers below the loss, sequence, multi-task and watch-time directions, because capacity was measured not to be the bottleneck.

**Note on gradient-boosted trees.** LightGBM and similar cannot represent the `user_id x video_id` cross that drives the FM. 27,285 users by 7,551 videos is not splittable by a tree. Using a GBDT means replacing IDs with target encodings, which is exactly the static-feature approach that already showed no gain. Worth one experiment for the ablation table; not a primary path.

**DeepFM, DCN, xDeepFM** add explicit or implicit higher-order crosses on top of the FM structure. More plausible than a GBDT here, but still below the loss change in expected value.

---

## Distribution shifts to be aware of

**Density.** Training averages ~81,500 rows/day; validation and test ~17,000/day. Per-user impression counts differ by roughly an order of magnitude. Raw count features will not transfer. Prefer rates.

**Time.** The test window runs 8 to 17 days after training ends. Item popularity decays and new content appears.

**User mix.** The validation oracle ceiling is 0.8484; the test oracle ceiling is 0.8645. The splits are not identical in composition, so validation is an imperfect proxy for test even before any overfitting.

---

## Unbiased validation with the randomised-exposure log

`log_random_4_22_to_5_08_pure.csv` holds 1.18M randomly-exposed impressions. The organisers suggest it as an **extra unbiased validation set** to check whether a model has only overfitted to biased traffic.

Two constraints: it spans 20220422 to 20220508, so everything from 20220429 is inside the test window and is off limits. And whether it may be used at all is still an open question with the organisers.

Using it for validation is lower risk than using it for training. It provides no data before the validation window, so it cannot be added to the training period at all.

---

## Methods that do not apply

- Retrieval, recall, candidate generation. The impression list is given.
- Re-ranking for diversity. Not rewarded by either metric.
- Negative sampling from an unseen catalogue. Negatives are the logged non-long-view impressions. Negative *weighting* and pair sampling *within* a user's list are different and are valid.
- Distributed training, GPU-dependent methods. The whole baseline runs in 40 seconds on one CPU core.
