"""Variance-reduced seed ensemble on the pointwise base model.

Why this and not another loss:
  * measured train primary 0.6009 vs valid 0.5944 -> a +0.0065 gap. The model
    is not overfitting, so a sharper objective has little to bite on.
  * mean embedding norms: tab 0.789, dur_bucket 0.744, author_id 0.138,
    video_id 0.137, user_id 0.102. The two low-cardinality fields dominate the
    crosses; the high-cardinality ID embeddings that must carry within-user
    ordering are an order of magnitude weaker. Rare rows simply do not receive
    enough updates: 1.14M rows at batch 8192 is ~139 Adam steps per epoch, and
    validation peaked at epoch 12. Cutting the batch to 2048 quadruples the
    step count per epoch at the same data cost, which is the cheapest way to
    let sparse ID embeddings grow without adding capacity (k is measured flat).
  * three of the last iterations moved the metrics by <= 0.0001, i.e. inside
    the seed noise floor. Averaging five seeds attacks that noise directly.

Why the blend is not a no-op: per-user normalisation of a SINGLE model is a
monotone transform inside the list and cannot change GAUC or nDCG@5. Rank
normalising each member BEFORE averaging is different -- the average of several
within-user rank vectors is not a monotone function of any one member, so the
blended ranking can differ from every member's. Equal weights, chosen once, so
nothing is tuned against validation.

No loss is registered: the default pointwise objective is the strongest base
measured so far, and this experiment deliberately isolates the sampling and
variance axes rather than stacking another untested objective on top.

diagnostics["ensemble"] reports each member's own validation primary, which
separates the two effects: member mean vs the reference FM tells me whether the
smaller batch helped the ID embeddings, and blend vs best member tells me
whether averaging added anything beyond picking a lucky seed.
"""

CONFIG = {
    "ensemble": [11, 23, 37, 53, 71],
    "normalise": "within_user_rank",
    "batch": 2048,
    "patience": 5,
}
