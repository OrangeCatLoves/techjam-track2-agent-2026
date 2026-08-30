import numpy as np
from harness.losses import register_loss


def _group_bounds(groups):
    """Contiguous index blocks, one per list, without materialising a mask."""
    order = np.argsort(groups, kind="stable")
    g = np.asarray(groups)[order]
    if g.size == 0:
        return order, np.zeros(1, dtype=np.int64)
    cuts = np.flatnonzero(np.concatenate(([True], g[1:] != g[:-1], [True])))
    return order, cuts


@register_loss("lambda_pairwise_ndcg5_v1", kind="pairwise")
def lambda_pairwise_ndcg5(z, y, groups):
    """LambdaRank: pairwise hinge on score differences, each pair weighted by
    the nDCG@5 change that swapping it would cause.

    The primary metric is the mean of GAUC and nDCG@5. GAUC is a plain pairwise
    quantity, so half the weight is constant across pairs. nDCG@5 only cares
    about the head of the list, so the other half is |delta nDCG@5| under the
    current ranking, which is zero for pairs already buried below rank 5 and
    large for a negative sitting at rank 0. Weights are normalised per list and
    averaged over lists, so a 200-impression user cannot drown a 3-impression
    one -- both metrics average per user.
    """
    z = np.asarray(z, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    grad = np.zeros_like(z)
    total = 0.0
    n_lists = 0

    ALPHA = 0.5          # constant (GAUC-aligned) share of each pair's weight
    CUT = 5              # nDCG truncation, matching the scored metric

    order, cuts = _group_bounds(groups)
    for a, b in zip(cuts[:-1], cuts[1:]):
        idx = order[a:b]
        yg = y[idx]
        n = idx.size
        npos = int((yg > 0.5).sum())
        if npos == 0 or npos == n:
            continue                      # no pair, and no nDCG signal either

        zg = z[idx]
        rank = np.empty(n, dtype=np.int64)
        rank[np.argsort(-zg, kind="stable")] = np.arange(n)
        disc = np.where(rank < CUT, 1.0 / np.log2(rank + 2.0), 0.0)

        m = min(CUT, n)
        ideal = np.sort(yg)[::-1][:m]
        idcg = float((ideal / np.log2(np.arange(m) + 2.0)).sum())
        if idcg <= 0.0:
            continue

        p = np.flatnonzero(yg > 0.5)
        q = np.flatnonzero(yg <= 0.5)
        d = zg[p][:, None] - zg[q][None, :]
        dn = np.abs(disc[p][:, None] - disc[q][None, :]) / idcg
        w = (ALPHA + (1.0 - ALPHA) * dn) / float(d.size)

        sig = 1.0 / (1.0 + np.exp(np.clip(d, -30.0, 30.0)))   # sigmoid(-d)
        total += float((w * np.logaddexp(0.0, -d)).sum())
        ws = w * sig
        grad[idx[p]] += -ws.sum(axis=1)
        grad[idx[q]] += ws.sum(axis=0)
        n_lists += 1

    if n_lists:
        total /= n_lists
        grad /= n_lists
    return float(total), grad.astype(np.float32)


CONFIG = {
    "loss": "lambda_pairwise_ndcg5_v1",
    "group_by": "user_id+date",
    "lr": 0.003,
    "max_epochs": 30,
    "patience": 5,
}
