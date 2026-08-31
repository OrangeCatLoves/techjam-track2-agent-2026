import numpy as np
from harness.losses import register_loss


@register_loss("listwise_softmax_ce_v1", kind="listwise")
def listwise_softmax_ce(z, y, groups):
    """Softmax cross-entropy over each user's impression list.

    Within every group the scores are softmaxed and matched against the
    label distribution (uniform over that list's positives). Groups with no
    positive carry no signal and are dropped; each contributing list gets
    equal weight, mirroring the per-user averaging in GAUC and nDCG@5.
    """
    z = np.asarray(z, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    g = np.asarray(groups).ravel()

    _uniq, gi = np.unique(g, return_inverse=True)
    n_groups = int(gi.max()) + 1 if gi.size else 0
    if n_groups == 0:
        return 0.0, np.zeros_like(z, dtype=np.float32)

    # numerically stable within-group softmax
    zmax = np.full(n_groups, -np.inf, dtype=np.float64)
    np.maximum.at(zmax, gi, z)
    shifted = np.clip(z - zmax[gi], -60.0, 0.0)
    e = np.exp(shifted)
    denom = np.bincount(gi, weights=e, minlength=n_groups)
    p = e / np.maximum(denom[gi], 1e-12)

    npos = np.bincount(gi, weights=y, minlength=n_groups)
    valid = npos > 0.0
    active = valid[gi]
    t = np.where(active, y / np.maximum(npos[gi], 1e-12), 0.0)

    n_valid = float(max(1, int(valid.sum())))
    w = active.astype(np.float64) / n_valid

    loss = float(-(w * t * np.log(p + 1e-12)).sum())
    grad = w * (p - t)
    return loss, grad.astype(np.float32)


CONFIG = {"loss": "listwise_softmax_ce_v1", "group_by": "user_id+date"}
