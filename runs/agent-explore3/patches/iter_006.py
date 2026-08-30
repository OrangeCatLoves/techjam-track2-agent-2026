import numpy as np
from harness.losses import register_loss


@register_loss("topk_plackett_luce_v1", kind="listwise")
def topk_plackett_luce(z, y, groups):
    """Top-k Plackett-Luce (ListMLE-style) over a user's full impression list.

    For each list, positives are placed sequentially: the t-th positive must beat
    every item still remaining after the earlier positives are removed. Unlike a
    single softmax cross-entropy over the list, this enforces an explicit ordering
    among the positives themselves and re-normalises the negative competition set
    at every step, so a list with p positives yields p distinct contrastive terms
    instead of one averaged one. Truncated at K to mirror nDCG@5's top-heavy
    weighting and to bound cost on very long lists.

    Pure lists (all-positive or all-negative) carry no within-list ordering
    information and are skipped, so no gradient mass is spent on them.
    """
    z = np.asarray(z, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    g_all = np.asarray(groups).ravel()

    grad = np.zeros_like(z)
    total = 0.0
    n_lists = 0
    K = 8

    order = np.argsort(g_all, kind="stable")
    gs = g_all[order]
    if gs.size == 0:
        return 0.0, grad.astype(np.float32)
    starts = np.flatnonzero(np.concatenate(([True], gs[1:] != gs[:-1])))
    bounds = np.concatenate((starts, [gs.size]))

    for i in range(bounds.size - 1):
        idx = order[bounds[i]:bounds[i + 1]]
        n = idx.size
        if n < 2:
            continue
        yy = y[idx]
        npos = int((yy > 0.5).sum())
        if npos == 0 or npos == n:
            continue

        zz = z[idx]
        zz = zz - zz.max()
        e = np.exp(np.clip(zz, -30.0, 30.0))

        pos = np.flatnonzero(yy > 0.5)
        # deterministic placement order: current model confidence, easiest first
        pos = pos[np.argsort(-zz[pos], kind="stable")]

        alive = np.ones(n, dtype=bool)
        gl = np.zeros(n, dtype=np.float64)
        steps = 0
        for t in range(min(K, pos.size)):
            p = pos[t]
            denom = float(e[alive].sum())
            if denom <= 1e-12:
                break
            probs = np.where(alive, e / denom, 0.0)
            total += -(zz[p] - np.log(denom))
            gl += probs
            gl[p] -= 1.0
            alive[p] = False
            steps += 1
        if steps == 0:
            continue
        grad[idx] += gl / steps
        n_lists += 1

    if n_lists == 0:
        return 0.0, grad.astype(np.float32)

    scale = 1.0 / n_lists
    grad *= scale
    if not np.all(np.isfinite(grad)):
        grad = np.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
    return float(total * scale), grad.astype(np.float32)


CONFIG = {
    "loss": "topk_plackett_luce_v1",
    "group_by": "user_id",
    "max_epochs": 25,
    "patience": 5,
}
