import numpy as np
from harness.losses import register_loss


@register_loss("listwise_softmax_ce_v1", kind="listwise")
def listwise_softmax_ce(z, y, groups):
    """Softmax cross-entropy over each impression list.

    Scores are normalised inside the list, so only within-list ordering is
    optimised -- the same thing GAUC and nDCG@5 measure. The target is the
    positives spread uniformly (y / n_pos), which makes the gradient
    p - t: it pushes down whichever negative currently holds probability
    mass at the top of the list, rather than every negative equally.

    Lists with no positive, or with a single row, carry no ordering signal
    and are given zero weight. Remaining lists are weighted equally, which
    matches nDCG@5's equal weight per user.
    """
    z = np.asarray(z, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()

    uniq, idx = np.unique(groups, return_inverse=True)
    n_lists = uniq.size

    # per-list max for numerical stability
    zmax = np.full(n_lists, -np.inf, dtype=np.float64)
    np.maximum.at(zmax, idx, z)
    e = np.exp(np.clip(z - zmax[idx], -60.0, 0.0))
    denom = np.bincount(idx, weights=e, minlength=n_lists)
    p = e / np.maximum(denom[idx], 1e-30)

    size = np.bincount(idx, minlength=n_lists).astype(np.float64)
    pos = np.bincount(idx, weights=y, minlength=n_lists)
    valid = (pos > 0.0) & (pos < size) & (size > 1.0)

    vrow = valid[idx]
    t = np.where(vrow, y / np.maximum(pos[idx], 1e-9), 0.0)

    n_valid = float(max(1, int(valid.sum())))
    w = np.where(vrow, 1.0 / n_valid, 0.0)

    loss = float(-(w * t * np.log(p + 1e-12)).sum())
    grad = w * (np.where(vrow, p, 0.0) - t)
    return loss, grad.astype(np.float32)


CONFIG = {"loss": "listwise_softmax_ce_v1", "group_by": "user_id+date"}
