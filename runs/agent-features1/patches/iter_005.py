import numpy as np
from harness.losses import register_loss

# Mixing weight on the pointwise term. 0.5 gives both terms equal say once
# each is normalised per participating row.
ALPHA = 0.5


@register_loss("hybrid_softmax_logloss_v1", kind="listwise")
def hybrid_softmax_logloss(z, y, groups):
    """Convex mix of pointwise logloss and within-list softmax cross-entropy.

    The listwise term is exactly zero on singleton lists and on lists that are
    all-positive or all-negative, which is most of the training data under
    user_id+date grouping. Those rows still carry information about which items
    get watched at all, and the pointwise term is what lets the ID embeddings
    absorb it. Both terms are averaged over the rows they actually touch, so
    ALPHA is a genuine mix rather than a scale accident.
    """
    z = np.asarray(z, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    n = z.shape[0]
    if n == 0:
        return 0.0, np.zeros(0, dtype=np.float32)

    grad = np.zeros(n, dtype=np.float64)

    # ---- pointwise term: dense gradient on every row, every field ----
    p = 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))
    loss_pt = float(-(y * np.log(p + 1e-9) + (1.0 - y) * np.log(1.0 - p + 1e-9)).mean())
    grad += ALPHA * (p - y) / n

    # ---- listwise term: softmax cross-entropy inside each list ----
    _, gid = np.unique(np.asarray(groups), return_inverse=True)
    ng = int(gid.max()) + 1

    zmax = np.full(ng, -np.inf, dtype=np.float64)
    np.maximum.at(zmax, gid, z)
    e = np.exp(np.clip(z - zmax[gid], -60.0, 0.0))
    denom = np.zeros(ng, dtype=np.float64)
    np.add.at(denom, gid, e)
    soft = e / np.maximum(denom[gid], 1e-12)

    pos = np.zeros(ng, dtype=np.float64)
    np.add.at(pos, gid, y)
    size = np.zeros(ng, dtype=np.float64)
    np.add.at(size, gid, 1.0)

    # only lists that can actually be ordered contribute
    orderable = (pos > 0.0) & (pos < size)
    m = orderable[gid]
    n_lw = float(max(1, int(m.sum())))

    loss_lw = 0.0
    if m.any():
        target = np.zeros(n, dtype=np.float64)
        target[m] = y[m] / pos[gid][m]
        loss_lw = float(-(target[m] * np.log(soft[m] + 1e-9)).sum() / n_lw)
        grad[m] += (1.0 - ALPHA) * (soft[m] - target[m]) / n_lw

    loss = ALPHA * loss_pt + (1.0 - ALPHA) * loss_lw
    return float(loss), grad.astype(np.float32)


CONFIG = {
    "loss": "hybrid_softmax_logloss_v1",
    "group_by": "user_id+date",
    "max_epochs": 40,
    "patience": 5,
}
