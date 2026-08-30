import numpy as np
from harness.losses import register_loss, sigmoid


@register_loss("bpr_within_list_v1", kind="pairwise")
def bpr_within_list(z, y, groups):
    """Bayesian Personalised Ranking over all pos/neg pairs inside each list.

    L = mean over lists of  mean over (p, n) pairs of  -log sigmoid(z_p - z_n).

    Only relative order inside a list is penalised, which is exactly what GAUC
    and nDCG@5 measure. Lists that are all-positive or all-negative generate no
    pairs and therefore no gradient -- they cannot be ordered wrongly.
    Each list carries equal weight so heavy users do not dominate.
    """
    z = np.asarray(z, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    g = np.asarray(groups).ravel()
    n = z.shape[0]
    grad_sorted = np.zeros(n, dtype=np.float64)

    order = np.argsort(g, kind="stable")
    gs = g[order]
    zs = z[order]
    ys = y[order]

    if n == 0:
        return 0.0, np.zeros(0, dtype=np.float32)

    cuts = np.flatnonzero(gs[1:] != gs[:-1]) + 1
    starts = np.concatenate(([0], cuts))
    ends = np.concatenate((cuts, [n]))

    total_loss = 0.0
    n_lists = 0

    for s, e in zip(starts, ends):
        yy = ys[s:e]
        pos = np.flatnonzero(yy > 0.5)
        if pos.size == 0 or pos.size == yy.size:
            continue
        neg = np.flatnonzero(yy <= 0.5)
        n_lists += 1

        zp = zs[s + pos]
        zn = zs[s + neg]
        d = np.clip(zp[:, None] - zn[None, :], -30.0, 30.0)
        sig = 1.0 / (1.0 + np.exp(-d))          # sigmoid(z_p - z_n)
        w = 1.0 / float(pos.size * neg.size)     # equal weight per list

        total_loss += -float(np.log(sig + 1e-9).sum()) * w

        # dL/dz_p = -(1 - sig),  dL/dz_n = +(1 - sig)
        one_minus = 1.0 - sig
        grad_sorted[s + pos] -= one_minus.sum(axis=1) * w
        grad_sorted[s + neg] += one_minus.sum(axis=0) * w

    denom = float(max(n_lists, 1))
    grad_sorted /= denom
    loss = total_loss / denom

    grad = np.zeros(n, dtype=np.float64)
    grad[order] = grad_sorted
    return float(loss), grad.astype(np.float32)


CONFIG = {"loss": "bpr_within_list_v1", "group_by": "user_id+date"}
