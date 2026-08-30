import numpy as np
from harness.losses import register_loss

TOP_K = 5
MARGIN = 1.0


@register_loss("topk_hard_margin_warp_v1", kind="pairwise")
def topk_hard_margin_warp(z, y, groups):
    """Head-focused hinge ranking loss with within-list hard-negative mining.

    For each user's list, only negatives that the model currently places in the
    top-K positions are eligible partners (if none are, the single
    highest-scoring negative is used, so every discriminative list keeps
    contributing). A (positive, eligible-negative) pair is active only when it
    violates the margin; correctly ordered, well-separated pairs receive exactly
    zero gradient. Each list is normalised by its own active-pair count and each
    list is weighted equally, matching nDCG@5's per-user weighting and stopping
    long training lists from dominating short ones.

    Homogeneous lists (all-positive or all-negative) are skipped: they can
    change neither GAUC nor nDCG@5.
    """
    z = np.asarray(z, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    g = np.asarray(groups).ravel()

    grad = np.zeros(z.shape[0], dtype=np.float64)
    order = np.argsort(g, kind="stable")
    gs = g[order]
    edges = np.flatnonzero(
        np.concatenate(([True], gs[1:] != gs[:-1], [True]))
    )

    total = 0.0
    n_lists = 0

    for a, b in zip(edges[:-1], edges[1:]):
        idx = order[a:b]
        if idx.size < 2:
            continue
        yy = y[idx]
        pos = idx[yy > 0.5]
        neg = idx[yy <= 0.5]
        if pos.size == 0 or neg.size == 0:
            continue

        zz = z[idx]
        head = idx[np.argsort(-zz, kind="stable")[:TOP_K]]
        elig = np.intersect1d(head, neg)
        if elig.size == 0:
            elig = neg[np.argmax(z[neg]):np.argmax(z[neg]) + 1]

        viol = MARGIN - (z[pos][:, None] - z[elig][None, :])
        act = viol > 0.0
        n_act = int(act.sum())
        if n_act == 0:
            continue

        w = 1.0 / n_act
        total += float(viol[act].sum()) * w
        np.add.at(grad, pos, -w * act.sum(axis=1))
        np.add.at(grad, elig, w * act.sum(axis=0))
        n_lists += 1

    if n_lists > 0:
        scale = 1.0 / n_lists
        grad *= scale
        total *= scale

    return float(total), grad.astype(np.float32)


CONFIG = {
    "loss": "topk_hard_margin_warp_v1",
    "group_by": "user_id",
    "lr": 0.002,
    "max_epochs": 30,
    "patience": 5,
}
