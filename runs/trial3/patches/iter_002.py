import numpy as np
from harness.losses import register_loss


def _list_slices(groups):
    """Contiguous-safe grouping: return list of index arrays, one per group id."""
    order = np.argsort(groups, kind="stable")
    g_sorted = groups[order]
    # boundaries where the group id changes
    if len(g_sorted) == 0:
        return []
    cuts = np.flatnonzero(g_sorted[1:] != g_sorted[:-1]) + 1
    return np.split(order, cuts)


@register_loss("bpr_pairwise_v1", kind="pairwise")
def bpr_pairwise(z, y, groups):
    """BPR over all positive-negative pairs formed WITHIN each group.

    For each group g with positives P and negatives N, every (p, n) pair
    contributes -log(sigmoid(z_p - z_n)). Each group's total contribution is
    normalised by |P|*|N| so a 200-impression list does not outweigh a
    5-impression list, then all groups are averaged. Groups that are entirely
    positive or entirely negative produce no pairs and no gradient, which is
    exactly the set of users GAUC ignores.
    """
    z = np.asarray(z, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    grad = np.zeros_like(z)
    total = 0.0
    n_groups = 0

    for idx in _list_slices(np.asarray(groups)):
        yy = y[idx]
        pos = idx[yy > 0.5]
        neg = idx[yy <= 0.5]
        npos = pos.size
        nneg = neg.size
        if npos == 0 or nneg == 0:
            continue
        n_groups += 1
        # difference matrix: rows = positives, cols = negatives
        d = z[pos][:, None] - z[neg][None, :]
        d = np.clip(d, -30.0, 30.0)
        # -log sigmoid(d) computed stably
        total += float(np.logaddexp(0.0, -d).sum()) / (npos * nneg)
        # d/dd of -log sigmoid(d) = -sigmoid(-d)
        s = -1.0 / (1.0 + np.exp(d))
        s /= (npos * nneg)
        np.add.at(grad, pos, s.sum(axis=1))
        np.add.at(grad, neg, -s.sum(axis=0))

    if n_groups == 0:
        return 0.0, grad.astype(np.float32)

    total /= n_groups
    grad /= n_groups
    return float(total), grad.astype(np.float32)


CONFIG = {
    "loss": "bpr_pairwise_v1",
    "group_by": "user_id+date",
    "k": 16,
    "lr": 0.005,
    "l2": 1e-6,
    "batch": 8192,
    "max_epochs": 25,
    "patience": 5,
}
