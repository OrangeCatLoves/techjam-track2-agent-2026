import numpy as np
from harness.losses import register_loss

# Blend weight: 0.5 keeps the dense pointwise signal that every row provides
# while adding pairwise, nDCG-weighted pressure on the lists that have pairs.
ALPHA = 0.5


@register_loss("hybrid_pointwise_lambda_pairwise_v1", kind="pairwise")
def hybrid_pointwise_lambda_pairwise(z, y, groups):
    """(1-a) * pointwise logloss  +  a * LambdaRank-weighted BPR within lists.

    Pure BPR gives no gradient at all on a list that is all-positive or
    all-negative, and the measured list profile says most lists are tiny
    (median 3 under user_id+date) with ~30% of users all-negative.  The
    pointwise half supplies signal on those rows; the pairwise half is
    weighted by |delta nDCG| of swapping the pair, which concentrates the
    ranking gradient at the top of each list where nDCG@5 is decided.
    """
    z64 = np.asarray(z, dtype=np.float64).ravel()
    y64 = np.asarray(y, dtype=np.float64).ravel()
    n = z64.shape[0]

    p = 1.0 / (1.0 + np.exp(-np.clip(z64, -30.0, 30.0)))
    point_loss = float(-(y64 * np.log(p + 1e-9)
                         + (1.0 - y64) * np.log(1.0 - p + 1e-9)).mean())
    grad = (1.0 - ALPHA) * (p - y64) / max(1, n)

    order = np.argsort(groups, kind="stable")
    gsorted = np.asarray(groups)[order]
    if n > 0:
        edges = np.flatnonzero(
            np.concatenate(([True], gsorted[1:] != gsorted[:-1], [True])))
    else:
        edges = np.array([0])

    pair_loss = 0.0
    pair_grad = np.zeros(n, dtype=np.float64)
    n_pairs = 0

    for a, b in zip(edges[:-1], edges[1:]):
        idx = order[a:b]
        m = idx.shape[0]
        if m < 2:
            continue
        yy = y64[idx]
        loc_pos = np.flatnonzero(yy > 0.5)
        loc_neg = np.flatnonzero(yy <= 0.5)
        if loc_pos.size == 0 or loc_neg.size == 0:
            continue

        zz = z64[idx]
        rank = np.empty(m, dtype=np.float64)
        rank[np.argsort(-zz, kind="stable")] = np.arange(1.0, m + 1.0)
        disc = 1.0 / np.log2(1.0 + rank)

        ideal_n = min(loc_pos.size, 5)
        idcg = float(np.sum(1.0 / np.log2(np.arange(2.0, ideal_n + 2.0))))
        if idcg <= 0.0:
            continue

        diff = zz[loc_pos][:, None] - zz[loc_neg][None, :]
        w = np.abs(disc[loc_pos][:, None] - disc[loc_neg][None, :]) / idcg
        s = 1.0 / (1.0 + np.exp(np.clip(diff, -30.0, 30.0)))  # sigmoid(-diff)

        pair_loss += float(np.sum(w * np.logaddexp(0.0, -diff)))
        ws = w * s
        pair_grad[idx[loc_pos]] -= ws.sum(axis=1)
        pair_grad[idx[loc_neg]] += ws.sum(axis=0)
        n_pairs += loc_pos.size * loc_neg.size

    if n_pairs > 0:
        pair_loss /= n_pairs
        grad += ALPHA * pair_grad / n_pairs

    total = (1.0 - ALPHA) * point_loss + ALPHA * pair_loss
    return float(total), grad.astype(np.float32)


CONFIG = {
    "loss": "hybrid_pointwise_lambda_pairwise_v1",
    "group_by": "user_id+date",
    "max_epochs": 20,
    "patience": 4,
}
