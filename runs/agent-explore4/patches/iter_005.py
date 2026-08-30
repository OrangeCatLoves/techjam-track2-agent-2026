import numpy as np
from harness.losses import register_loss, sigmoid


@register_loss("allpairs_ranknet_full_history_v1", kind="pairwise")
def allpairs_ranknet_full_history(z, y, groups):
    """Exhaustive pos-neg RankNet over each user's whole impression history.

    Every (positive, negative) pair inside a list contributes
    log(1 + exp(-(z_pos - z_neg))). Each list is normalised by its own pair
    count so a 809-impression user does not drown a 5-impression user, which
    matches the per-user averaging both scored metrics use.

    Lists carrying only one class (or arriving as a single-class fragment if
    the trainer splits a group across batches) have no pair signal; they keep a
    small pointwise calibration weight instead of being dropped, so no rows are
    wasted.
    """
    z64 = np.asarray(z, dtype=np.float64).ravel()
    y64 = np.asarray(y, dtype=np.float64).ravel()
    n = z64.shape[0]
    grad = np.zeros(n, dtype=np.float64)
    if n == 0:
        return 0.0, grad.astype(np.float32)

    g = np.asarray(groups).ravel()
    order = np.argsort(g, kind="stable")
    gs = g[order]
    starts = np.flatnonzero(np.concatenate(([True], gs[1:] != gs[:-1])))
    ends = np.concatenate((starts[1:], [n]))

    ALPHA = 0.05  # total weight left on lists that carry no pair signal
    loss = 0.0
    wsum = 0.0
    degenerate = []

    for s, e in zip(starts, ends):
        idx = order[s:e]
        yy = y64[idx]
        ispos = yy > 0.5
        npos = int(ispos.sum())
        if npos == 0 or npos == idx.size:
            degenerate.append(idx)
            continue
        pos = idx[ispos]
        neg = idx[~ispos]
        d = z64[pos][:, None] - z64[neg][None, :]
        w = 1.0 / float(pos.size * neg.size)
        loss += w * float(np.logaddexp(0.0, -d).sum())
        sg = sigmoid(-d)  # dL/dd, large where the pair is violated
        grad[pos] += -w * np.asarray(sg, dtype=np.float64).sum(axis=1)
        grad[neg] += w * np.asarray(sg, dtype=np.float64).sum(axis=0)
        wsum += 1.0

    if degenerate:
        di = np.concatenate(degenerate)
        p = np.asarray(sigmoid(z64[di]), dtype=np.float64)
        w = ALPHA / float(di.size)
        yd = y64[di]
        loss += float(-(w * (yd * np.log(p + 1e-9)
                             + (1.0 - yd) * np.log(1.0 - p + 1e-9))).sum())
        grad[di] += w * (p - yd)
        wsum += ALPHA

    if wsum > 0.0:
        loss /= wsum
        grad /= wsum
    return float(loss), grad.astype(np.float32)


CONFIG = {"loss": "allpairs_ranknet_full_history_v1",
          "group_by": "user_id",
          "max_epochs": 30,
          "patience": 5}
