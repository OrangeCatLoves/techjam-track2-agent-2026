import numpy as np
from harness.losses import register_loss


@register_loss("listwise_softmax_v1", kind="listwise")
def listwise_softmax(z, y, groups):
    """Softmax cross-entropy over each user's impression list.

    Both scored metrics are within-list ranking metrics, so only score
    *differences* inside a list matter. Softmax CE optimises exactly that:
    it is shift-invariant per list, so no gradient is spent on absolute
    calibration. Targets are the labels normalised to sum to 1 over the
    list, which spreads mass evenly across a list's positives.

    Lists that are all-positive or all-negative carry no ordering
    information (their nDCG is 1.0 or 0.0 whatever the model does, and they
    are excluded from GAUC), so they are dropped: they would otherwise only
    inject a constant-target gradient that pushes on calibration again.

    Returns mean-over-lists loss and dL/dz, so each list counts once,
    matching the per-user averaging in GAUC and nDCG@5.
    """
    z = np.asarray(z, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    g = np.asarray(groups)

    # Contiguous list blocks via a sort on the group id (stable: preserves
    # within-list row order, so the result does not depend on input order).
    order = np.argsort(g, kind="stable")
    gs = g[order]
    # Boundaries of each run of equal group id.
    starts = np.flatnonzero(np.r_[True, gs[1:] != gs[:-1]])
    ends = np.r_[starts[1:], gs.size]

    grad = np.zeros_like(z)
    total = 0.0
    n_lists = 0

    for s, e in zip(starts, ends):
        idx = order[s:e]
        if idx.size < 2:
            continue
        yy = y[idx]
        npos = yy.sum()
        if npos <= 0.0 or npos >= idx.size:
            continue  # no learnable ordering in this list

        zz = z[idx]
        m = zz.max()
        ex = np.exp(zz - m)
        p = ex / ex.sum()

        t = yy / npos  # target distribution over the list
        total += float(-(t * np.log(p + 1e-12)).sum())
        grad[idx] = p - t
        n_lists += 1

    if n_lists == 0:
        return 0.0, np.zeros_like(z, dtype=np.float32)

    inv = 1.0 / n_lists
    return float(total * inv), (grad * inv).astype(np.float32)


CONFIG = {
    "loss": "listwise_softmax_v1",
    "group_by": "user_id+date",
    "lr": 0.01,
    "batch": 65536,
}
