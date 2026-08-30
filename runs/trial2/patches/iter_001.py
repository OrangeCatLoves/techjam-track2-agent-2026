import numpy as np
from harness.losses import register_loss


@register_loss("bpr_within_list_v1", kind="pairwise")
def bpr_within_list(z, y, groups):
    """Pairwise BPR over all positive/negative pairs inside each group.

    For every list g (defined by `groups`) we take every ordered pair
    (i positive, j negative) drawn from that list only, and optimise
        L_g = mean_{i,j} -log sigmoid(z_i - z_j)
    Lists with no positives or no negatives contribute nothing, which is
    correct: they carry no within-list ordering information and they are the
    all-negative / all-positive users that GAUC itself excludes.

    Each list is normalised by its own pair count and then all lists are
    weighted equally, so a 200-impression list does not drown out a
    3-impression one. Evaluation lists average under six items, so equal
    per-list weight is the weighting the metric actually uses.

    Gradient. With d = z_i - z_j and s = sigmoid(d),
        dL/dz_i = -(1 - s) / npairs,  dL/dz_j = +(1 - s) / npairs
    summed over the pairs each row takes part in. Computed with an outer
    difference per list, which is cheap because lists are small.
    """
    z = np.asarray(z, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    groups = np.asarray(groups).ravel()

    grad = np.zeros_like(z)
    total_loss = 0.0
    n_lists = 0

    # Contiguous blocks per group via a sort; keeps everything vectorised
    # inside a list and avoids a boolean mask scan per unique group.
    order = np.argsort(groups, kind="stable")
    gz = z[order]
    gy = y[order]
    gg = groups[order]
    if gg.size == 0:
        return 0.0, grad.astype(np.float32)

    boundaries = np.flatnonzero(gg[1:] != gg[:-1]) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [gg.size]))

    for s, e in zip(starts, ends):
        yl = gy[s:e]
        pos = np.flatnonzero(yl > 0.5)
        if pos.size == 0:
            continue
        neg = np.flatnonzero(yl <= 0.5)
        if neg.size == 0:
            continue

        zl = gz[s:e]
        d = zl[pos][:, None] - zl[neg][None, :]
        d = np.clip(d, -30.0, 30.0)
        sig = 1.0 / (1.0 + np.exp(-d))

        npairs = float(pos.size * neg.size)
        total_loss += float(-np.log(sig + 1e-9).sum()) / npairs

        # coefficient (1 - sigmoid) per pair
        c = (1.0 - sig) / npairs
        idx = order[s:e]
        np.add.at(grad, idx[pos], -c.sum(axis=1))
        np.add.at(grad, idx[neg], c.sum(axis=0))
        n_lists += 1

    if n_lists > 0:
        inv = 1.0 / float(n_lists)
        total_loss *= inv
        grad *= inv

    return float(total_loss), grad.astype(np.float32)


CONFIG = {
    "loss": "bpr_within_list_v1",
    "group_by": "user_id+date",
    "k": 16,
    "lr": 0.01,
    "l2": 1e-6,
    "batch": 8192,
    "max_epochs": 40,
    "patience": 5,
}
