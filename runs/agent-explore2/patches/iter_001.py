import numpy as np
from harness.losses import register_loss, sigmoid

_MAX_PAIRS = 2_000_000


@register_loss("bpr_allpairs_v1", kind="pairwise")
def bpr_allpairs(z, y, groups):
    """All-pairs BPR within each list.

    L = mean over (positive, negative) pairs of -log sigmoid(z_pos - z_neg),
    where pairs are formed only inside the same group. GAUC is the per-user
    probability that a positive outranks a negative, so this objective is a
    smooth surrogate for the metric itself, unlike pointwise logloss.

    Fully vectorised: groups are never iterated in Python. Pair enumeration is
    exact; if a batch would produce more than _MAX_PAIRS pairs, a deterministic
    random subsample is taken so memory stays bounded.
    """
    z64 = np.asarray(z, dtype=np.float64).ravel()
    n = z64.shape[0]
    grad = np.zeros(n, dtype=np.float64)

    pos_bool = np.asarray(y).ravel() > 0.5
    _, ginv = np.unique(np.asarray(groups).ravel(), return_inverse=True)
    n_groups = int(ginv.max()) + 1 if ginv.size else 0
    if n_groups == 0:
        return 0.0, grad.astype(np.float32)

    pos_idx = np.nonzero(pos_bool)[0]
    neg_idx = np.nonzero(~pos_bool)[0]
    if pos_idx.size == 0 or neg_idx.size == 0:
        return 0.0, grad.astype(np.float32)

    # negatives contiguous by group, so each group's negatives form one block
    neg_g = ginv[neg_idx]
    order = np.argsort(neg_g, kind="stable")
    neg_idx = neg_idx[order]
    neg_g = neg_g[order]
    cn = np.bincount(neg_g, minlength=n_groups)
    neg_start = np.concatenate(([0], np.cumsum(cn)[:-1]))

    pos_g = ginv[pos_idx]
    counts = cn[pos_g]
    keep = counts > 0
    if not np.any(keep):
        return 0.0, grad.astype(np.float32)
    pos_idx = pos_idx[keep]
    pos_g = pos_g[keep]
    counts = counts[keep]

    total = int(counts.sum())
    if total == 0:
        return 0.0, grad.astype(np.float32)

    # for each positive, expand the whole negative block of its own group
    cum = np.concatenate(([0], np.cumsum(counts)[:-1]))
    offsets = np.repeat(neg_start[pos_g] - cum, counts)
    flat = offsets + np.arange(total)
    pi = np.repeat(pos_idx, counts)
    ni = neg_idx[flat]

    if total > _MAX_PAIRS:
        rng = np.random.default_rng(12345)
        sel = rng.choice(total, size=_MAX_PAIRS, replace=False)
        pi = pi[sel]
        ni = ni[sel]
        total = _MAX_PAIRS

    d = np.clip(z64[pi] - z64[ni], -30.0, 30.0)
    loss = float(np.logaddexp(0.0, -d).sum() / total)

    # d/dd of -log sigmoid(d) = -sigmoid(-d)
    coef = -sigmoid(-d) / total
    np.add.at(grad, pi, coef)
    np.add.at(grad, ni, -coef)
    return loss, grad.astype(np.float32)


CONFIG = {
    "loss": "bpr_allpairs_v1",
    "group_by": "user_id+date",
    "max_epochs": 60,
    "patience": 6,
}
