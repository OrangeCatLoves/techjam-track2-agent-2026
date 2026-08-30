import numpy as np
from harness.losses import register_loss


@register_loss("lambdarank_ndcg_v1", kind="pairwise")
def lambdarank_ndcg(z, y, groups):
    """LambdaRank: pairwise logistic loss weighted by |delta nDCG| of a swap.

    Within each list the rows are ranked by current score. Every
    (positive, negative) pair is weighted by the change in DCG that swapping
    the two would produce, normalised by the list's ideal DCG. Pairs near the
    top of the list get much larger weight than deep pairs, which is what
    nDCG@5 measures, while the pairwise form stays aligned with GAUC.
    Lists with no positives or no negatives contribute nothing, exactly as
    they contribute nothing to GAUC.
    """
    z = np.asarray(z, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    n = z.shape[0]
    grad = np.zeros(n, dtype=np.float64)
    total = 0.0
    if n == 0:
        return 0.0, grad.astype(np.float32)

    # contiguous blocks per list, rows ordered by descending score inside a list
    order = np.lexsort((-z, groups))
    gs = groups[order]
    starts = np.flatnonzero(np.concatenate(([True], gs[1:] != gs[:-1])))
    bounds = np.concatenate((starts, [gs.shape[0]]))

    for b in range(bounds.shape[0] - 1):
        idx = order[bounds[b]:bounds[b + 1]]
        m = idx.shape[0]
        if m < 2:
            continue
        yy = y[idx]
        npos = int((yy > 0).sum())
        if npos == 0 or npos == m:
            continue

        disc = 1.0 / np.log2(np.arange(m, dtype=np.float64) + 2.0)
        ideal = np.sort(yy)[::-1]
        idcg = float((ideal * disc).sum())
        if idcg <= 1e-12:
            continue

        pos = np.flatnonzero(yy > 0)
        neg = np.flatnonzero(yy <= 0)
        zz = z[idx]
        d = np.clip(zz[pos][:, None] - zz[neg][None, :], -30.0, 30.0)
        w = np.abs(disc[pos][:, None] - disc[neg][None, :]) / idcg

        total += float((w * np.logaddexp(0.0, -d)).sum())
        lam = w / (1.0 + np.exp(d))  # w * sigmoid(-d)
        grad[idx[pos]] -= lam.sum(axis=1)
        grad[idx[neg]] += lam.sum(axis=0)

    scale = 1.0 / float(max(1, n))
    return total * scale, (grad * scale).astype(np.float32)


CONFIG = {
    "loss": "lambdarank_ndcg_v1",
    "group_by": "user_id+date",
    "lr": 0.002,
    "max_epochs": 40,
    "patience": 5,
}
