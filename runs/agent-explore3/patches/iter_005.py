import numpy as np
from harness.losses import register_loss

ALPHA = 0.5  # weight on the listwise term; 1-ALPHA on the pointwise term


@register_loss("hybrid_pointwise_listwise_v1", kind="listwise")
def hybrid_pointwise_listwise(z, y, groups):
    """Convex blend of pointwise logloss and within-list softmax cross-entropy.

    Both terms are normalised by the number of rows in the batch so their
    gradient magnitudes are comparable and the baseline learning rate stays
    appropriate. The listwise term uses the label distribution y / sum(y) as
    the target and is only active on lists that hold at least one positive and
    at least two items; all other lists contribute zero to it.
    """
    z = np.asarray(z, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    n = max(1, z.size)

    # ---- pointwise logloss ----
    p = 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))
    point_loss = float(-(y * np.log(p + 1e-9) + (1.0 - y) * np.log(1.0 - p + 1e-9)).sum() / n)
    g_point = (p - y) / n

    # ---- listwise softmax cross-entropy ----
    uniq, gi = np.unique(np.asarray(groups).ravel(), return_inverse=True)
    ng = uniq.size

    gmax = np.full(ng, -np.inf, dtype=np.float64)
    np.maximum.at(gmax, gi, z)
    e = np.exp(np.clip(z - gmax[gi], -60.0, 0.0))
    denom = np.bincount(gi, weights=e, minlength=ng)
    soft = e / np.maximum(denom[gi], 1e-12)

    pos = np.bincount(gi, weights=y, minlength=ng)
    size = np.bincount(gi, minlength=ng).astype(np.float64)
    valid = ((pos > 0.0) & (size > 1.0)).astype(np.float64)

    target = y / np.maximum(pos[gi], 1e-9)
    active = valid[gi]
    list_loss = float(-(active * target * np.log(soft + 1e-12)).sum() / n)
    g_list = active * (soft - target) / n

    loss = (1.0 - ALPHA) * point_loss + ALPHA * list_loss
    grad = (1.0 - ALPHA) * g_point + ALPHA * g_list
    return float(loss), grad.astype(np.float32)


CONFIG = {"loss": "hybrid_pointwise_listwise_v1", "group_by": "user_id+date", "patience": 5}
