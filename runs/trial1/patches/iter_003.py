import numpy as np
from harness.losses import register_loss

ALPHA = 1.0      # weight on the pairwise term relative to pointwise
CAP = 16         # negatives sampled per positive, per list


@register_loss("hybrid_point_bpr_user_v1", kind="pairwise")
def hybrid_point_bpr(z, y, groups):
    """Pointwise logloss on every row + sampled BPR over pairs inside a list.

    A pure pairwise loss contributes nothing on lists that are all-positive,
    all-negative or single-item, which is a large share of short lists. The
    pointwise term keeps those rows training the shared embeddings while the
    BPR term aligns the model with within-list ordering. Pairs are built only
    from rows sharing a `groups` value, so permuting `groups` changes the loss.
    """
    z = np.asarray(z, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    n = z.shape[0]
    if n == 0:
        return 0.0, np.zeros(0, dtype=np.float32)

    # ---- pointwise term -------------------------------------------------
    p = 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))
    loss_point = float(-(y * np.log(p + 1e-9) + (1.0 - y) * np.log(1.0 - p + 1e-9)).mean())
    g_point = (p - y) / n

    # ---- pairwise term, strictly within a group -------------------------
    order = np.argsort(groups, kind="stable")
    gs = np.asarray(groups)[order]
    edges = np.flatnonzero(np.concatenate(([True], gs[1:] != gs[:-1], [True])))

    g_pair = np.zeros(n, dtype=np.float64)
    loss_pair = 0.0
    total_pairs = 0
    rng = np.random.default_rng(12345)

    for i in range(edges.shape[0] - 1):
        idx = order[edges[i]:edges[i + 1]]
        if idx.shape[0] < 2:
            continue
        yy = y[idx]
        pos = idx[yy > 0.5]
        neg = idx[yy <= 0.5]
        if pos.shape[0] == 0 or neg.shape[0] == 0:
            continue
        if neg.shape[0] > CAP:
            sel = rng.integers(0, neg.shape[0], size=(pos.shape[0], CAP))
            negs = neg[sel]
        else:
            negs = np.repeat(neg[None, :], pos.shape[0], axis=0)
        d = z[pos][:, None] - z[negs]
        s = 1.0 / (1.0 + np.exp(-np.clip(d, -30.0, 30.0)))
        loss_pair += float(-np.log(s + 1e-9).sum())
        w = s - 1.0                      # dL/d(z_pos - z_neg)
        np.add.at(g_pair, pos, w.sum(axis=1))
        np.add.at(g_pair, negs.ravel(), -w.ravel())
        total_pairs += d.size

    if total_pairs > 0:
        loss_pair /= total_pairs
        g_pair /= total_pairs

    loss = loss_point + ALPHA * loss_pair
    grad = g_point + ALPHA * g_pair
    return float(loss), grad.astype(np.float32)


CONFIG = {
    "loss": "hybrid_point_bpr_user_v1",
    "group_by": "user_id",
    "k": 16,
    "lr": 0.001,
    "l2": 1e-6,
    "max_epochs": 30,
    "patience": 4,
}
