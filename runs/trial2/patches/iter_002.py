import numpy as np
from harness.losses import register_loss

ALPHA = 0.5  # weight on the listwise term; 1-ALPHA on pointwise


@register_loss("hybrid_point_listwise_v1", kind="listwise")
def hybrid_point_listwise(z, y, groups):
    """Pointwise logloss over every row + within-list softmax cross-entropy.

    The listwise term is a softmax over the scores of one user-day impression
    list, with the target being the label distribution normalised over that
    list's positives. It is only defined for lists holding at least one
    positive and at least one negative; every other row still receives the
    pointwise gradient, so no row is discarded the way a pure pairwise loss
    discards them. Both terms are normalised per row so ALPHA is a true mix.
    """
    z = np.asarray(z, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    n = z.shape[0]
    if n == 0:
        return 0.0, np.zeros(0, dtype=np.float32)
    inv_n = 1.0 / float(n)
    zc = np.clip(z, -30.0, 30.0)

    # ---- pointwise ----
    p = 1.0 / (1.0 + np.exp(-zc))
    loss_pt = float(-(y * np.log(p + 1e-9) + (1.0 - y) * np.log(1.0 - p + 1e-9)).sum() * inv_n)
    g_pt = (p - y) * inv_n

    # ---- listwise softmax within group ----
    _, gidx = np.unique(np.asarray(groups).ravel(), return_inverse=True)
    ng = int(gidx.max()) + 1

    gmax = np.full(ng, -np.inf, dtype=np.float64)
    np.maximum.at(gmax, gidx, zc)
    e = np.exp(zc - gmax[gidx])
    gsum = np.zeros(ng, dtype=np.float64)
    np.add.at(gsum, gidx, e)
    soft = e / np.maximum(gsum[gidx], 1e-12)

    pos = np.zeros(ng, dtype=np.float64)
    np.add.at(pos, gidx, y)
    cnt = np.zeros(ng, dtype=np.float64)
    np.add.at(cnt, gidx, 1.0)

    valid = (pos > 0.0) & (pos < cnt)
    vmask = valid[gidx]

    t = np.zeros(n, dtype=np.float64)
    g_lw = np.zeros(n, dtype=np.float64)
    loss_lw = 0.0
    if vmask.any():
        denom = pos[gidx][vmask]
        t[vmask] = y[vmask] / denom
        loss_lw = float(-(t[vmask] * np.log(soft[vmask] + 1e-12)).sum() * inv_n)
        g_lw[vmask] = (soft[vmask] - t[vmask]) * inv_n

    loss = (1.0 - ALPHA) * loss_pt + ALPHA * loss_lw
    grad = (1.0 - ALPHA) * g_pt + ALPHA * g_lw
    return float(loss), grad.astype(np.float32)


CONFIG = {
    "loss": "hybrid_point_listwise_v1",
    "group_by": "user_id+date",
    "max_epochs": 20,
    "patience": 4,
}
