"""Five-seed ensemble of the reference pointwise FM, blended by within-user rank.

Why this and not another loss:
  Iterations 1-3 replaced the objective (pairwise BPR, LambdaRank, listwise
  softmax). All three scored below the pointwise reference, and the listwise run
  showed a +0.0118 train/valid gap with validation peaking at epoch 8 of 13.
  That is an estimation-variance signature, not an objective-misalignment one.

Why rank-blending is not the measured no-op:
  Per-user normalisation of a SINGLE model is monotone within the list and so
  cannot move GAUC or nDCG@5. Averaging the within-user ranks of SEVERAL models
  is not monotone in any one member's score, so it can and does reorder lists.

What the field norms imply:
  tab 1.164 and dur_bucket 0.824 dominate, while user_id 0.223, video_id 0.283
  and author_id 0.286 are weakly identified. Weakly identified embeddings are
  exactly the parameters whose fitted values swing most across seeds, so the
  personalisation signal is the part of the model that averaging should recover.

No custom loss is registered: the trainer's default pointwise logloss is the
strongest objective measured so far on this benchmark, and this experiment
changes one thing only, the ensemble stage.
"""

CONFIG = {
    "ensemble": 5,
    "normalise": "within_user_rank",
}
