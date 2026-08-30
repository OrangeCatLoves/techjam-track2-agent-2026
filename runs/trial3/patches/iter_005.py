import numpy as np
from harness.losses import register_loss


def _segments(groups):
    """Contiguous index segments, one per group id."""
    order = np.argsort(groups, kind="stable")
    g = np.asarray(groups)[order]
    if g.size == 0:
        return order, np.array([0], dtype=np.int64)
    cuts = np.flatnonzero(np.concatenate(([True], g[1:] != g[:-1], [True])))
    return order, cuts


@register_loss("adversarial_pairwise_sqrtpos_v1", kind="pairwise")
def adversarial_pairwise_sqrtpos(z, y, groups):
    """Pairwise ranking loss with self-adversarial negative weighting.

    Within each user's list, every (positive, negative) pair contributes
    softplus(-(z_pos - z_neg)), but negatives are re-weighted by a softmax over
    their own current scores, so gradient concentrates on the negatives the
    model currently places near the top of the list -- exactly the errors that
    nDCG@5 and GAUC punish hardest. Each list is weighted by sqrt(#positives),
    between GAUC's positive-count weighting and nDCG's uniform weighting.
    Label-homogeneous lists produce no pairs and therefore no gradient.
    """
    z = np.asarray(z, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    grad = np.zeros_like(z)

    order, cuts = _segments(groups)

    segs = []
    wsum = 0.0
    for a, b in zip(cuts[:-1], cuts[1:]):
        idx = order[a:b]
        yy = y[idx]
        pmask = yy > 0.5
        pos = idx[pmask]
        neg = idx[~pmask]
        if pos.size == 0 or neg.size == 0:
            continue
        w = float(np.sqrt(pos.size))
        wsum += w
        segs.append((pos, neg, w))

    if wsum <= 0.0 or not segs:
        return 0.0, grad.astype(np.float32)

    total = 0.0
    for pos, neg, w in segs:
        zn = z[neg]
        zp = z[pos]
        # self-adversarial weights over the negatives of this list
        aw = np.exp(zn - zn.max())
        aw /= aw.sum()
        d = zp[:, None] - zn[None, :]
        cw = w / (float(pos.size) * wsum)
        total += cw * float((np.logaddexp(0.0, -np.clip(d, -30.0, 30.0)) * aw[None, :]).sum())
        s = 1.0 / (1.0 + np.exp(np.clip(d, -30.0, 30.0)))  # sigmoid(-(zp - zn))
        sw = s * aw[None, :]
        grad[pos] += -cw * sw.sum(axis=1)
        grad[neg] += cw * sw.sum(axis=0)

    return float(total), grad.astype(np.float32)


CONFIG = {
    "loss": "adversarial_pairwise_sqrtpos_v1",
    "group_by": "user_id",
    "lr": 0.002,
    "max_epochs": 40,
    "patience": 5,
}
