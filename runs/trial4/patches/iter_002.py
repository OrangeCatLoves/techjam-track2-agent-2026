import numpy as np
from harness.losses import register_loss


@register_loss("bpr_all_pairs_v1", kind="pairwise")
def bpr_all_pairs(z, y, groups):
    """Exhaustive within-list BPR: every (positive, negative) pair inside one
    group contributes -log(sigmoid(z_pos - z_neg)).

    Pairs are built strictly within a group, so permuting `groups` changes the
    pair set and therefore the loss. Each list is normalised by its own pair
    count and lists are averaged equally, so a 200-impression user cannot swamp
    a 4-impression user. Lists that are all-positive or all-negative produce no
    pairs and no gradient -- they carry no within-user ordering information.
    """
    z = np.asarray(z, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    g = np.asarray(groups).ravel()

    grad = np.zeros_like(z)
    total = 0.0
    n_lists = 0

    order = np.argsort(g, kind="stable")
    gs = g[order]
    if gs.shape[0] == 0:
        return 0.0, grad.astype(np.float32)
    edges = np.flatnonzero(np.concatenate(([True], gs[1:] != gs[:-1], [True])))

    for a, b in zip(edges[:-1], edges[1:]):
        idx = order[a:b]
        if idx.shape[0] < 2:
            continue
        yy = y[idx]
        pos = idx[yy > 0.5]
        neg = idx[yy <= 0.5]
        np_, nn = pos.shape[0], neg.shape[0]
        if np_ == 0 or nn == 0:
            continue
        n_lists += 1
        w = 1.0 / float(np_ * nn)
        d = np.clip(z[pos][:, None] - z[neg][None, :], -30.0, 30.0)
        s = 1.0 / (1.0 + np.exp(-d))
        total += -np.log(s + 1e-9).sum() * w
        c = (1.0 - s) * w
        np.add.at(grad, pos, -c.sum(axis=1))
        np.add.at(grad, neg, c.sum(axis=0))

    if n_lists > 0:
        inv = 1.0 / float(n_lists)
        total *= inv
        grad *= inv

    return float(total), grad.astype(np.float32)


CONFIG = {
    "loss": "bpr_all_pairs_v1",
    "group_by": "user_id",
    "k": 16,
    "lr": 0.002,
    "l2": 1e-6,
    "batch": 8192,
    "max_epochs": 30,
    "patience": 5,
}
