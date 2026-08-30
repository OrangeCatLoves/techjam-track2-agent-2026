import numpy as np
from harness.losses import register_loss

_DISC = 1.0 / np.log2(np.arange(2, 8194, dtype=np.float64))


@register_loss("lambdarank_anchored_v1", kind="listwise")
def lambdarank_anchored(z, y, groups):
    """LambdaRank pairwise term (|dNDCG@5| weights) + pointwise logloss anchor.

    Purely relative objectives (BPR, listwise softmax) get zero gradient from
    lists that are all-positive or all-negative, so a large share of rows never
    updates the user_id x video_id cross.  The anchor term restores that signal
    for every row; the lambda term supplies metric-shaped ordering pressure,
    weighting each (pos, neg) pair by the nDCG@5 change a swap would cause, so
    pairs near the top of the list dominate and pairs below rank 5 are ignored.
    """
    z = np.asarray(z, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    n = max(1, z.shape[0])
    alpha = 0.4  # weight on the pointwise anchor

    # --- pointwise anchor over every row -------------------------------
    p = 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))
    loss_pt = float(-(y * np.log(p + 1e-9) + (1.0 - y) * np.log(1.0 - p + 1e-9)).sum()) / n
    g_pt = (p - y) / n

    # --- lambda-weighted pairwise term ---------------------------------
    g_pw = np.zeros(n, dtype=np.float64)
    loss_pw = 0.0
    order = np.argsort(groups, kind="stable")
    gs = np.asarray(groups)[order]
    if gs.shape[0] > 0:
        starts = np.flatnonzero(np.concatenate(([True], gs[1:] != gs[:-1])))
        bounds = np.concatenate((starts, [gs.shape[0]]))
    else:
        bounds = np.array([0])

    for b in range(bounds.shape[0] - 1):
        idx = order[bounds[b]:bounds[b + 1]]
        m = idx.shape[0]
        if m < 2:
            continue
        yy = y[idx]
        npos = int(yy.sum())
        if npos == 0 or npos == m:
            continue  # no orderable pair; the anchor already covers these rows
        zz = z[idx]
        rank = np.empty(m, dtype=np.int64)
        rank[np.argsort(-zz, kind="stable")] = np.arange(m)
        disc = np.zeros(m, dtype=np.float64)
        top = rank < 5
        disc[top] = _DISC[rank[top]]
        idcg = _DISC[:min(npos, 5)].sum()
        if idcg <= 0.0:
            continue
        pos = np.flatnonzero(yy > 0.5)
        neg = np.flatnonzero(yy <= 0.5)
        d = zz[pos][:, None] - zz[neg][None, :]
        w = np.abs(disc[pos][:, None] - disc[neg][None, :]) / idcg
        # a floor keeps deep pairs contributing a little, which GAUC rewards
        w = w + 0.05 / max(1, pos.shape[0] * neg.shape[0])
        dc = np.clip(d, -30.0, 30.0)
        loss_pw += float((w * np.logaddexp(0.0, -dc)).sum())
        lam = w / (1.0 + np.exp(dc))  # w * sigmoid(-d)
        g_pw[idx[pos]] -= lam.sum(axis=1)
        g_pw[idx[neg]] += lam.sum(axis=0)

    loss_pw /= n
    g_pw /= n

    loss = alpha * loss_pt + (1.0 - alpha) * loss_pw
    grad = alpha * g_pt + (1.0 - alpha) * g_pw
    if not np.all(np.isfinite(grad)):
        grad = np.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
    return float(loss), grad.astype(np.float32)


CONFIG = {
    "loss": "lambdarank_anchored_v1",
    "group_by": "user_id+date",
    "max_epochs": 30,
    "patience": 5,
}
