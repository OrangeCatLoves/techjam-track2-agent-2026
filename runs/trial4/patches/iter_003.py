import numpy as np
from harness.losses import register_loss


@register_loss("lambdarank_ndcg_daylist_v1", kind="pairwise")
def lambdarank_ndcg_daylist(z, y, groups):
    """LambdaRank: pairwise logistic loss weighted by |delta nDCG| within a list.

    Pairs are formed strictly inside a group (a user's impressions on one day
    when group_by is user_id+date). Each pair (pos, neg) carries weight
    |disc_pos - disc_neg| / idealDCG, where disc = 1/log2(rank+2) under the
    model's current within-list ordering. Inversions near the top of a list
    therefore dominate the gradient, which is what nDCG@5 rewards, while the
    objective stays pairwise and so remains aligned with GAUC.

    Each list is normalised by its own pair count and the total by the number
    of contributing lists, so a 200-impression list cannot outweigh a
    5-impression one. Lists that are all-positive or all-negative contribute
    nothing, exactly as they contribute nothing to GAUC.
    """
    z = np.asarray(z, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    grad = np.zeros_like(z)
    total = 0.0

    order = np.argsort(groups, kind="stable")
    gs = np.asarray(groups)[order]
    if gs.size == 0:
        return 0.0, grad.astype(np.float32)
    starts = np.flatnonzero(np.concatenate(([True], gs[1:] != gs[:-1])))
    bounds = np.concatenate((starts, [gs.size]))

    n_lists = 0
    for a, b in zip(bounds[:-1], bounds[1:]):
        idx = order[a:b]
        yy = y[idx]
        n = idx.size
        n_pos = int((yy > 0.5).sum())
        if n_pos == 0 or n_pos == n:
            continue

        zz = z[idx]
        srt = np.argsort(-zz, kind="stable")
        rank = np.empty(n, dtype=np.float64)
        rank[srt] = np.arange(n, dtype=np.float64)
        disc = 1.0 / np.log2(rank + 2.0)
        idcg = float((1.0 / np.log2(np.arange(n_pos, dtype=np.float64) + 2.0)).sum())
        if idcg <= 0.0:
            continue

        pi = np.flatnonzero(yy > 0.5)
        ni = np.flatnonzero(yy <= 0.5)
        d = zz[pi][:, None] - zz[ni][None, :]
        dc = np.clip(d, -30.0, 30.0)
        s = 1.0 / (1.0 + np.exp(dc))          # sigmoid(-(z_pos - z_neg))
        dw = np.abs(disc[pi][:, None] - disc[ni][None, :]) / idcg
        w = 1.0 / float(pi.size * ni.size)

        total += float((dw * np.logaddexp(0.0, -dc)).sum()) * w
        sw = s * dw
        grad[idx[pi]] += -sw.sum(axis=1) * w
        grad[idx[ni]] += sw.sum(axis=0) * w
        n_lists += 1

    if n_lists > 0:
        total /= n_lists
        grad /= n_lists
    return float(total), grad.astype(np.float32)


CONFIG = {
    "loss": "lambdarank_ndcg_daylist_v1",
    "group_by": "user_id+date",
    "l2": 1e-5,
    "patience": 4,
}
