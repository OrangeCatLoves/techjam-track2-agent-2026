import numpy as np
from harness.losses import register_loss


@register_loss("metric_matched_list_balanced_sampling_v1", kind="pointwise")
def metric_matched_list_balanced_sampling(z, y, groups):
    """Pointwise logloss with metric-matched sample weights.

    Three corrections, all on the sampling side, none on the objective:

    1. Equal mass per list. Training lists average 43.5 rows under user_id and
       5.77 under (user_id, date); evaluation lists average 5.58. Unweighted
       logloss lets heavy users dominate. Each list here contributes the same
       total weight regardless of length.
    2. Balanced within a list. Positives share half the list's mass, negatives
       the other half, so a list whose long_view rate is far from 0.5 still
       produces a gradient that separates its two classes rather than one that
       chases the list's base rate. Base rate is constant within a list and so
       contributes nothing to either scored metric.
    3. Degenerate lists downweighted, not dropped. A list with zero positives or
       zero negatives has no within-list ordering to learn; GAUC ignores those
       users entirely. They are kept at BETA=0.3 of a normal list's mass because
       they still supply cross-list signal for the user and video embeddings
       (1.6% of valid rows have a cold user, so embedding coverage matters).
    """
    z = np.asarray(z, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    _, inv = np.unique(groups, return_inverse=True)
    n_list = np.bincount(inv).astype(np.float64)
    pos_list = np.bincount(inv, weights=y).astype(np.float64)
    neg_list = n_list - pos_list

    BETA = 0.3
    degenerate = (pos_list <= 0.0) | (neg_list <= 0.0)

    flat = BETA / np.maximum(n_list, 1.0)
    w_pos_list = np.where(degenerate, flat, 0.5 / np.maximum(pos_list, 1.0))
    w_neg_list = np.where(degenerate, flat, 0.5 / np.maximum(neg_list, 1.0))

    is_pos = y > 0.5
    weight = np.where(is_pos, w_pos_list[inv], w_neg_list[inv])
    total = weight.sum()
    if total <= 0.0 or not np.isfinite(total):
        weight = np.full_like(z, 1.0 / max(1, z.shape[0]))
    else:
        weight = weight / total

    p = 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))
    loss = float(-(weight * (y * np.log(p + 1e-9)
                             + (1.0 - y) * np.log(1.0 - p + 1e-9))).sum())
    grad = (weight * (p - y)).astype(np.float32)
    return loss, grad


CONFIG = {
    "loss": "metric_matched_list_balanced_sampling_v1",
    "group_by": "user_id+date",
}
