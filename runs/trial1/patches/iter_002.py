import numpy as np
from harness.losses import register_loss

ALPHA = 0.35  # constant floor: plain BPR component, keeps GAUC pressure below rank 5
CUT = 5       # nDCG truncation used by the metric


@register_loss("lambdarank_ndcg5_v1", kind="pairwise")
def lambdarank_ndcg5(z, y, groups):
    """LambdaRank over within-group positive/negative pairs.

    For each list (a group in `groups`) every (positive, negative) pair gets a
    logistic BPR term softplus(-(z_pos - z_neg)) weighted by
    ALPHA + |dNDCG@5| where |dNDCG@5| is the change in truncated nDCG@5 that
    swapping the two items would cause under the current predicted ranking.
    Pairs are built strictly inside a group, so permuting `groups` changes both
    the pair set and the loss.
    """
    z = np.asarray(z, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    groups = np.asarray(groups).ravel()
    grad = np.zeros(z.shape[0], dtype=np.float64)
    total = 0.0
    wsum = 0.0

    order = np.argsort(groups, kind="stable")
    gs = groups[order]
    if gs.shape[0] == 0:
        return 0.0, grad.astype(np.float32)
    edges = np.flatnonzero(np.concatenate(([True], gs[1:] != gs[:-1], [True])))

    for a, b in zip(edges[:-1], edges[1:]):
        idx = order[a:b]
        n = idx.shape[0]
        if n < 2:
            continue
        yy = y[idx]
        pos = np.flatnonzero(yy > 0.5)
        neg = np.flatnonzero(yy <= 0.5)
        if pos.size == 0 or neg.size == 0:
            continue
        zz = z[idx]

        # predicted ranks (0-based), ties broken stably
        rank = np.empty(n, dtype=np.int64)
        rank[np.argsort(-zz, kind="stable")] = np.arange(n)
        disc = np.where(rank < CUT, 1.0 / np.log2(rank.astype(np.float64) + 2.0), 0.0)

        k = min(CUT, n)
        npos = min(int(pos.size), k)
        idcg = float(np.sum(1.0 / np.log2(np.arange(npos, dtype=np.float64) + 2.0)))
        if idcg <= 0.0:
            continue

        dz = zz[pos][:, None] - zz[neg][None, :]
        dzc = np.clip(dz, -30.0, 30.0)
        sig = 1.0 / (1.0 + np.exp(dzc))              # sigmoid(-(z_pos - z_neg))
        softplus = np.log1p(np.exp(-np.abs(dzc))) + np.maximum(-dzc, 0.0)

        dndcg = np.abs(disc[pos][:, None] - disc[neg][None, :]) / idcg
        w = ALPHA + dndcg

        total += float(np.sum(w * softplus))
        wsum += float(np.sum(w))

        ws = w * sig
        np.add.at(grad, idx[pos], -ws.sum(axis=1))
        np.add.at(grad, idx[neg], ws.sum(axis=0))

    if wsum > 0.0:
        grad /= wsum
        total /= wsum
    return float(total), grad.astype(np.float32)


CONFIG = {
    "loss": "lambdarank_ndcg5_v1",
    "group_by": "user_id+date",
    "k": 16,
    "lr": 0.003,
    "l2": 1e-6,
    "batch": 8192,
    "max_epochs": 40,
    "patience": 5,
}
