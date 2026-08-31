"""Variance-reduction experiment: rank-averaged 5-seed ensemble.

No new objective and no new capacity. The base learner is the reference
pointwise configuration that currently holds the best validation primary.
The only change is that five independent fits are blended.

Rationale: each user_id embedding is estimated from roughly forty training
rows and each video_id embedding from a similarly thin slice, so the fitted
embedding table carries substantial estimation variance from random
initialisation and minibatch order. That variance is independent across
seeds, so averaging shrinks it while leaving the shared signal intact.
This is orthogonal to the loss changes tried in iterations 1-5, all of which
reshaped the gradient without changing how much of the fit is noise.

Blending is done on within-user ranks rather than raw scores. Both metrics
depend only on the within-list ordering, so a member's absolute score scale
is meaningless; rank-normalising first stops the member with the widest
logit range from silently dominating the average. Weights are left equal:
tuning them against the same validation split used for selection would
spend the split's remaining independence for a fraction of the gain.

Diagnostics report each member's own validation primary alongside the
blend, so if the blend fails to exceed the best single member that is a
direct measurement that seed variance is not the limiting factor here.
"""

CONFIG = {
    "ensemble": [11, 23, 37, 53, 71],
    "normalise": "within_user_rank",
    "group_by": "user_id",
}
