import numpy as np
from harness.losses import register_loss


@register_loss("warp_rank_weighted_hinge_v1", kind="pairwise")
def warp_rank_weighted_hinge(z, y, groups):
    """WARP (Weston et al.) rank-weighted margin hinge, built within each list.

    For every positive in a user's list we count how many negatives violate the
    margin.  That count is an estimate of the positive's rank, and the pair loss
    is scaled by log(1 + rank) / rank, so a positive sitting near the top of the
    list gets a small push and one buried under many negatives gets a large one.
    Unlike logistic / softmax objectives the hinge does not saturate: it keeps
    demanding an absolute score gap of `margin`, which is the pressure the ID
    embeddings need in order to grow away from their measured near-zero norms.

    Lists are defined solely by `groups`; a list with no positive or no negative
    contributes nothing, and each contributing list is normalised to unit weight.
    """
    z = np.asarray(z, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    grad = np.zeros_like(z)
    margin = 1.0
    total = 0.0
    n_lists = 0

    order = np.argsort(groups, kind="stable")
    g_sorted = np.asarray(groups)[order]
    if g_sorted.size == 0:
        return 0.0, grad.astype(np.float32)
    boundary = np.concatenate(([True], g_sorted[1:] != g_sorted[:-1]))
    starts = np.flatnonzero(boundary)
    ends = np.concatenate((starts[1:], [g_sorted.size]))

    for s, e in zip(starts, ends):
        idx = order[s:e]
        if idx.size < 2:
            continue
        yy = y[idx]
        pmask = yy > 0.5
        pos = idx[pmask]
        neg = idx[~pmask]
        n_pos = pos.size
        n_neg = neg.size
        if n_pos == 0 or n_neg == 0:
            continue
        n_lists += 1

        diff = margin - (z[pos][:, None] - z[neg][None, :])
        viol = diff > 0.0
        n_viol = viol.sum(axis=1).astype(np.float64)
        active = n_viol > 0.0
        if not np.any(active):
            continue

        w = np.zeros(n_pos, dtype=np.float64)
        w[active] = np.log1p(n_viol[active]) / n_viol[active]
        scale = 1.0 / float(n_pos)

        hinge = np.where(viol, diff, 0.0)
        total += scale * float((w[:, None] * hinge).sum())

        gpos = -scale * w * n_viol
        gneg = scale * (w[:, None] * viol).sum(axis=0)
        grad[pos] += gpos
        grad[neg] += gneg

    denom = float(max(1, n_lists))
    total /= denom
    grad /= denom
    return float(total), grad.astype(np.float32)


CONFIG = {
    "loss": "warp_rank_weighted_hinge_v1",
    "group_by": "user_id",
    "max_epochs": 45,
    "patience": 6,
}
