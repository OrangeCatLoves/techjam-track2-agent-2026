import numpy as np
from harness.losses import register_loss


@register_loss("listwise_softmax_ce_v1", kind="listwise")
def listwise_softmax_ce(z, y, groups):
    """Softmax cross-entropy over each impression list.

    For every list we softmax the logits and match the distribution against the
    normalised label vector (uniform over that list's positives). Lists with no
    positives, or with every item positive, carry no within-list ordering
    information; they get zero gradient rather than pushing all scores in one
    direction, which is a monotone no-op for GAUC/nDCG anyway.

    Gradient wrt z is (p - target), the standard softmax-CE form, so it drops
    straight into the existing scatter/Adam path.
    """
    z = np.asarray(z, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    uniq, inv = np.unique(groups, return_inverse=True)
    n_lists = uniq.size

    # per-list max for a stable softmax
    zmax = np.full(n_lists, -np.inf, dtype=np.float64)
    np.maximum.at(zmax, inv, z)
    e = np.exp(np.clip(z - zmax[inv], -60.0, 0.0))
    denom = np.bincount(inv, weights=e, minlength=n_lists)
    p = e / np.maximum(denom[inv], 1e-12)

    size = np.bincount(inv, minlength=n_lists).astype(np.float64)
    pos = np.bincount(inv, weights=y, minlength=n_lists)
    # a list only teaches ordering if it has at least one positive and one negative
    usable = (pos > 0.0) & (pos < size)
    n_used = float(max(1, int(usable.sum())))

    target = np.where(usable[inv], y / np.maximum(pos[inv], 1e-12), 0.0)
    active = usable[inv].astype(np.float64)

    loss = float(-(target * np.log(p + 1e-12)).sum() / n_used)
    grad = ((p - target) * active) / n_used
    return loss, grad.astype(np.float32)


CONFIG = {
    "loss": "listwise_softmax_ce_v1",
    "group_by": "user_id+date",
    "lr": 0.003,
    "max_epochs": 40,
    "patience": 5,
}
