import numpy as np
from harness.losses import register_loss


@register_loss("discriminative_list_softmax_v1", kind="listwise")
def discriminative_list_softmax(z, y, groups):
    """Listwise softmax cross-entropy, computed only over discriminative lists.

    A list with zero positives or with every item positive has a fixed metric
    contribution (nDCG 0 or 1) and is excluded from GAUC altogether. Its
    gradient therefore only shapes calibration, which neither scored metric
    reads. Those lists are dropped here so all capacity goes to lists whose
    ordering can actually change the score.

    Multi-positive lists use a uniform target over their positives. Every
    surviving list is weighted equally, so a long list cannot dominate.
    """
    z = np.asarray(z, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    _, inv = np.unique(groups, return_inverse=True)
    n_groups = int(inv.max()) + 1 if inv.size else 0
    grad = np.zeros_like(z)
    if n_groups == 0:
        return 0.0, grad.astype(np.float32)

    size = np.bincount(inv, minlength=n_groups).astype(np.float64)
    npos = np.bincount(inv, weights=y, minlength=n_groups)
    discriminative = (npos > 0.0) & (npos < size)
    keep = discriminative[inv]
    if not keep.any():
        return 0.0, grad.astype(np.float32)

    zk = z[keep]
    yk = y[keep]
    _, gk = np.unique(inv[keep], return_inverse=True)
    m = int(gk.max()) + 1

    mx = np.full(m, -np.inf, dtype=np.float64)
    np.maximum.at(mx, gk, zk)
    e = np.exp(np.clip(zk - mx[gk], -60.0, 0.0))
    denom = np.bincount(gk, weights=e, minlength=m)
    p = e / np.maximum(denom[gk], 1e-12)

    pos_per_list = np.bincount(gk, weights=yk, minlength=m)
    target = yk / np.maximum(pos_per_list[gk], 1e-12)

    w = 1.0 / float(m)
    loss = float(-w * (target * np.log(p + 1e-12)).sum())
    grad[keep] = w * (p - target)
    return loss, grad.astype(np.float32)


CONFIG = {
    "loss": "discriminative_list_softmax_v1",
    "group_by": "user_id+date",
    "l2": 1e-5,
    "max_epochs": 30,
    "patience": 5,
}
