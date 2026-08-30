import numpy as np
from harness.losses import register_loss


@register_loss("listwise_softmax_ce_v1", kind="listwise")
def listwise_softmax_ce(z, y, groups):
    """Softmax cross-entropy over each impression list.

    Within a list the target is the normalised label vector (uniform over the
    positives). The gradient w.r.t. the logits is (p - q), so only score
    differences inside a list matter -- any constant added to a whole list has
    zero gradient, which matches how the metrics read the scores.

    Lists with zero positives or with every item positive admit no ordering and
    are masked to zero gradient, mirroring the users GAUC is computed over.

    Fully vectorised (bincount / maximum.at); no Python loop over groups.
    """
    z = np.asarray(z, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    g = np.asarray(groups).ravel()
    n = z.shape[0]
    grad = np.zeros(n, dtype=np.float64)
    if n == 0:
        return 0.0, grad.astype(np.float32)

    _, idx = np.unique(g, return_inverse=True)
    idx = np.asarray(idx, dtype=np.int64).ravel()
    n_groups = int(idx.max()) + 1

    zc = np.clip(z, -30.0, 30.0)

    # per-list max for numerical stability
    gmax = np.full(n_groups, -np.inf, dtype=np.float64)
    np.maximum.at(gmax, idx, zc)
    e = np.exp(zc - gmax[idx])
    denom = np.bincount(idx, weights=e, minlength=n_groups)
    p = e / np.maximum(denom[idx], 1e-12)

    pos = np.bincount(idx, weights=y, minlength=n_groups)
    size = np.bincount(idx, minlength=n_groups).astype(np.float64)
    usable = (pos > 0.0) & (pos < size)
    mask = usable[idx]

    q = np.zeros(n, dtype=np.float64)
    q[mask] = y[mask] / np.maximum(pos[idx][mask], 1e-12)

    scale = float(mask.sum())
    if scale <= 0.0:
        return 0.0, grad.astype(np.float32)

    loss = float(-(q[mask] * np.log(p[mask] + 1e-12)).sum() / scale)
    grad[mask] = (p[mask] - q[mask]) / scale
    grad = np.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
    return loss, grad.astype(np.float32)


CONFIG = {
    "loss": "listwise_softmax_ce_v1",
    "group_by": "user_id+date",
    "k": 16,
    "lr": 0.003,
    "max_epochs": 40,
    "patience": 5,
}
