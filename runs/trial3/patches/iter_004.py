"""Variance-reduction experiment: 5-seed FM ensemble, rank-blended within user.

Rationale (see hypothesis): the objective has been changed three times with no
measurable move, while the diagnostics point at variance rather than bias --
sparse user_id/video_id/author_id embeddings retain near-init norms and the
validation curve peaks mid-training then decays. Independent seeds perturb
exactly that sparse block, so averaging across seeds should cancel init noise
while leaving the shared, well-determined dur_bucket/tab structure intact.

Blending is done on within-user ranks, not raw scores: GAUC and nDCG@5 only see
the within-list ordering, and raw-score averaging would let whichever member has
the widest score range dominate the blend. Equal weights, tuned once and not
revisited, to avoid fitting the validation split with blend weights.

The base objective is deliberately left at the reference pointwise logloss so
this measures the ensemble effect alone and is not confounded with a fourth
loss rewrite. diagnostics["ensemble"] reports each member and the blend-vs-best
delta, which tells us whether averaging added signal or merely smoothed noise.
"""

CONFIG = {
    "ensemble": [11, 23, 37, 53, 71],
    "normalise": "within_user_rank",
}
