import numpy as np
from harness.losses import register_loss


@register_loss("listwise_softmax_ce_v1", kind="listwise")
def listwise_softmax_ce(z, y, groups):
    """ListNet-style listwise cross-entropy within each impression list.

    For every list g: p = softmax(z_g), target t = y_g / sum(y_g).
    Loss = -sum_g sum_i t_i log p_i, averaged over lists that contain at
    least one positive. Lists with no positives carry no ordering signal and
    are masked out entirely (zero loss, zero gradient).

    The objective is invariant to any per-list additive shift of z, so it
    optimises pure within-list ordering rather than absolute calibration.
    It depends on `groups` through both the softmax normaliser and the target
    distribution, so permuting the grouping changes the loss.
    """
    z = np.asarray(z, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()

    _, idx = np.unique(groups, return_inverse=True)
    idx = idx.astype(np.int64, copy=False)
    m = int(idx.max()) + 1 if idx.size else 0
    if m == 0:
        return 0.0, np.zeros_like(z, dtype=np.float32)

    # per-list max for a numerically stable softmax
    mx = np.full(m, -np.inf, dtype=np.float64)
    np.maximum.at(mx, idx, z)
    e = np.exp(np.clip(z - mx[idx], -60.0, 0.0))
    denom = np.bincount(idx, weights=e, minlength=m)
    denom = np.maximum(denom, 1e-12)
    p = e / denom[idx]

    pos = np.bincount(idx, weights=y, minlength=m)
    has_pos = pos > 0.0
    n_valid = float(has_pos.sum())
    if n_valid == 0.0:
        return 0.0, np.zeros_like(z, dtype=np.float32)

    safe_pos = np.where(has_pos, pos, 1.0)
    t = y / safe_pos[idx]
    mask = has_pos[idx].astype(np.float64)

    loss = float(-(mask * t * np.log(p + 1e-12)).sum() / n_valid)
    grad = (mask * (p - t) / n_valid)
    return loss, grad.astype(np.float32)


CONFIG = {
    "loss": "listwise_softmax_ce_v1",
    "group_by": "user_id+date",
    "k": 16,
    "lr": 0.003,
    "l2": 1e-6,
    "batch": 8192,
    "max_epochs": 40,
    "patience": 5,
}
