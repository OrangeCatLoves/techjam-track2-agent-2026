import numpy as np
from harness.losses import register_loss, sigmoid

_RNG = np.random.default_rng(20260830)
_ANCHOR = 0.1  # small pointwise term, keeps unpaired rows from being gradient-free


@register_loss("bpr_pairwise_anchored_v1", kind="pairwise")
def bpr_pairwise_anchored(z, y, groups):
    """BPR over within-list (positive, negative) pairs, plus a small pointwise anchor.

    For every positive row one negative is drawn uniformly from the same group and
    -log(sigmoid(z_pos - z_neg)) is minimised. Pairs are normalised so each list
    contributes equal total weight. Rows in single-class or singleton groups get no
    pair, so a 0.1-weight logloss anchor is added over all rows.
    """
    z64 = np.asarray(z, dtype=np.float64).ravel()
    y64 = np.asarray(y, dtype=np.float64).ravel()
    n = int(z64.size)
    grad = np.zeros(n, dtype=np.float64)
    loss = 0.0
    if n == 0:
        return 0.0, grad.astype(np.float32)

    p = sigmoid(z64)
    loss += _ANCHOR * float(-(y64 * np.log(p + 1e-9)
                              + (1.0 - y64) * np.log(1.0 - p + 1e-9)).mean())
    grad += _ANCHOR * (p - y64) / n

    inv = np.unique(np.asarray(groups).ravel(), return_inverse=True)[1]
    G = int(inv.max()) + 1
    ispos = y64 > 0.5
    pos_idx = np.flatnonzero(ispos)
    neg_idx = np.flatnonzero(~ispos)

    if pos_idx.size and neg_idx.size:
        neg_g = inv[neg_idx]
        order = np.argsort(neg_g, kind="stable")
        neg_sorted = neg_idx[order]
        counts = np.bincount(neg_g, minlength=G)
        starts = np.concatenate((np.zeros(1, dtype=np.int64),
                                 np.cumsum(counts)[:-1].astype(np.int64)))
        pg = inv[pos_idx]
        c = counts[pg]
        keep = c > 0
        pos_idx = pos_idx[keep]
        pg = pg[keep]
        c = c[keep]
        if pos_idx.size:
            draw = (_RNG.random(pos_idx.size) * c).astype(np.int64)
            np.minimum(draw, c - 1, out=draw)
            sel = neg_sorted[starts[pg] + draw]
            d = z64[pos_idx] - z64[sel]
            s = sigmoid(d)  # sigmoid(d), for gradient formula
            npos = np.bincount(pg, minlength=G)
            w = 1.0 / npos[pg]
            w = w / w.sum()
            loss += float((w * -np.log(sigmoid(d) + 1e-9)).sum())
            # Gradient: dL/d(z_pos) = -(1 - sigmoid(d)) = sigmoid(d) - 1
            #           dL/d(z_neg) = 1 - sigmoid(d)
            np.add.at(grad, pos_idx, w * (s - 1.0))
            np.add.at(grad, sel, w * (1.0 - s))

    return float(loss), grad.astype(np.float32)


CONFIG = {"loss": "bpr_pairwise_anchored_v1",
          "group_by": "user_id",
          "lr": 0.005,
          "batch": 32768,
          "max_epochs": 40,
          "patience": 5}