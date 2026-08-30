import numpy as np
from harness.losses import register_loss


@register_loss("lambdarank_ndcg_v1", kind="pairwise")
def lambdarank_ndcg(z, y, groups):
    """LambdaRank: within-list positive/negative pairs weighted by |dNDCG|.

    For each list, rank items by their current score. For every (positive,
    negative) pair, the logistic pairwise loss is weighted by the change in
    NDCG that swapping the two items would produce. That makes the gradient
    an approximation of the nDCG gradient while keeping the pairwise form
    that GAUC measures. Lists with no positive or no negative contribute
    nothing, which is correct: they are unrankable and GAUC excludes them.
    """
    z = np.asarray(z, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    grad = np.zeros_like(z)
    total = 0.0
    sigma = 1.0

    order = np.argsort(groups, kind="stable")
    gs = np.asarray(groups)[order]
    if gs.shape[0] == 0:
        return 0.0, grad.astype(np.float32)
    starts = np.flatnonzero(np.concatenate(([True], gs[1:] != gs[:-1])))
    ends = np.concatenate((starts[1:], [gs.shape[0]]))

    n_lists = 0
    for s, e in zip(starts, ends):
        idx = order[s:e]
        n = idx.shape[0]
        if n < 2:
            continue
        yy = y[idx]
        npos = int(round(float(yy.sum())))
        if npos <= 0 or npos >= n:
            continue
        zz = z[idx]

        # position of each item under the current ranking (0 = top)
        rank = np.empty(n, dtype=np.int64)
        rank[np.argsort(-zz, kind="stable")] = np.arange(n)
        disc = 1.0 / np.log2(rank.astype(np.float64) + 2.0)
        idcg = float(np.sum(1.0 / np.log2(np.arange(npos, dtype=np.float64) + 2.0)))
        if idcg <= 0.0:
            continue

        pos = np.flatnonzero(yy > 0.5)
        neg = np.flatnonzero(yy <= 0.5)

        d = zz[pos][:, None] - zz[neg][None, :]
        d = np.clip(d, -30.0, 30.0)
        dndcg = np.abs(disc[pos][:, None] - disc[neg][None, :]) / idcg

        # sigmoid(-sigma * d): how wrong the pair currently is
        wrong = 1.0 / (1.0 + np.exp(sigma * d))
        lam = sigma * wrong * dndcg

        total += float(np.sum(dndcg * np.logaddexp(0.0, -sigma * d)))
        grad[idx[pos]] += -lam.sum(axis=1)
        grad[idx[neg]] += lam.sum(axis=0)
        n_lists += 1

    if n_lists > 0:
        total /= n_lists
        grad /= n_lists
    if not np.all(np.isfinite(grad)):
        grad = np.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
    return float(total), grad.astype(np.float32)


CONFIG = {"loss": "lambdarank_ndcg_v1", "group_by": "user_id+date"}
