import numpy as np
from harness.losses import register_loss

NAME = "lambdarank_ndcg5_v1"


def _lambdarank(z, y, groups):
    """LambdaRank: pairwise logistic loss weighted by |delta nDCG|.

    For every (positive, negative) pair inside one impression list, the
    RankNet gradient is scaled by how much the list's nDCG would change if
    the two items swapped positions under the current scores. Position
    discounts 1/log2(rank+2) make top-of-list mistakes dominate, which is
    what nDCG@5 measures, while deep pairs still carry weight, which is what
    GAUC measures. Normalised by the ideal DCG truncated at 5.
    """
    z = np.asarray(z, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    g = np.asarray(groups).ravel()
    grad = np.zeros_like(z)
    if z.size == 0:
        return 0.0, grad.astype(np.float32)

    order = np.argsort(g, kind="stable")
    gs = g[order]
    edges = np.flatnonzero(
        np.concatenate(([True], gs[1:] != gs[:-1], [True]))
    )

    total = 0.0
    n_lists = 0
    for a, b in zip(edges[:-1], edges[1:]):
        idx = order[a:b]
        n = idx.size
        if n < 2:
            continue
        yy = y[idx]
        n_pos = int(round(float(yy.sum())))
        if n_pos <= 0 or n_pos >= n:
            continue  # no orderable pair; contributes nothing to either metric
        zz = z[idx]

        rank = np.empty(n, dtype=np.float64)
        rank[np.argsort(-zz, kind="stable")] = np.arange(n, dtype=np.float64)
        disc = 1.0 / np.log2(rank + 2.0)

        top = min(n_pos, 5)
        idcg = float(np.sum(1.0 / np.log2(np.arange(top) + 2.0)))
        if idcg <= 0.0:
            continue

        pos = np.flatnonzero(yy > 0.5)
        neg = np.flatnonzero(yy <= 0.5)

        dn = np.abs(disc[pos][:, None] - disc[neg][None, :]) / idcg
        diff = np.clip(zz[pos][:, None] - zz[neg][None, :], -30.0, 30.0)

        total += float(np.sum(dn * np.logaddexp(0.0, -diff)))
        lam = dn / (1.0 + np.exp(diff))  # dn * sigmoid(-(z_pos - z_neg))

        grad[idx[pos]] -= lam.sum(axis=1)
        grad[idx[neg]] += lam.sum(axis=0)
        n_lists += 1

    if n_lists > 0:
        inv = 1.0 / float(n_lists)
        total *= inv
        grad *= inv
    return float(total), grad.astype(np.float32)


def _register(fn):
    last = None
    for kw in ({"kind": "listwise"}, {"kind": "pairwise"},
               {"kind": "pointwise"}, {}):
        try:
            return register_loss(NAME, **kw)(fn)
        except Exception as exc:  # unknown kind label; try the next spelling
            last = exc
    raise last


lambdarank_ndcg5 = _register(_lambdarank)

CONFIG = {"loss": NAME, "group_by": "user_id+date", "max_epochs": 40,
          "patience": 5}
