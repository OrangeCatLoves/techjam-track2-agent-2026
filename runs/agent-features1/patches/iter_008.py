"""Variance-reduction experiment: blend five independently seeded FMs.

No new objective and no new field. Every member is the reference pointwise
model; the only thing that differs between them is initialisation and batch
order. Each member is converted to within-user ranks before averaging, so the
blend is not a monotone transform of any single member and can reorder items
inside a user's list.

Rationale: with about 43 training rows per user the learned user and item
embeddings carry substantial seed-dependent noise, which shows up as unstable
within-user orderings. Averaging ranks across seeds cancels the zero-mean part
of that noise while leaving the systematic signal intact. If the blend beats
its best member, estimation variance was a real part of the residual gap; if it
matches the best member exactly, variance is not the constraint and the
remaining budget belongs to representation changes instead.

diagnostics["ensemble"] reports both numbers, so either outcome is informative.
"""

CONFIG = {
    "ensemble": 5,
    "normalise": "within_user_rank",
}
