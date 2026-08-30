import numpy as np
from harness.losses import register_loss


@register_loss("listwise_softmax_v1", kind="listwise")
def listwise_softmax(z, y, groups):
    """Softmax cross-entropy within each impression list.

    For each group (a user's list under the configured grouping) we softmax the
    logits over the list members and take cross-entropy against a target
    distribution that spreads mass uniformly over that list's positives.

    Lists with no positives, or with no negatives, carry no ordering
    information: their target equals nothing learnable, so they are dropped
    (zero gradient). That concentrates all gradient on discriminative lists,
    which is exactly the population GAUC is averaged over.

    Gradient wrt z is (softmax_prob - target) / n_lists, which is the standard
    listwise softmax gradient. It depends on `groups` through the partition
    function, so permuting the grouping changes the loss.
    """
    z = np.asarray(z, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    g = np.asarray(groups).ravel()

    grad = np.zeros_like(z)
    total_loss = 0.0

    # contiguous-ish grouping via sort, so we touch each row once
    order = np.argsort(g, kind="stable")
    gs = g[order]
    # boundaries of each group in the sorted order
    if gs.shape[0] == 0:
        return 0.0, grad.astype(np.float32)
    change = np.empty(gs.shape[0], dtype=bool)
    change[0] = True
    change[1:] = gs[1:] != gs[:-1]
    starts = np.flatnonzero(change)
    ends = np.append(starts[1:], gs.shape[0])

    n_used = 0
    for s, e in zip(starts, ends):
        idx = order[s:e]
        if idx.shape[0] < 2:
            continue
        yy = y[idx]
        npos = yy.sum()
        if npos <= 0.0 or npos >= idx.shape[0]:
            continue  # no within-list ordering signal
        zz = z[idx]
        m = zz.max()
        ex = np.exp(np.clip(zz - m, -30.0, 30.0))
        p = ex / ex.sum()
        t = yy / npos
        total_loss += float(-(t * np.log(p + 1e-12)).sum())
        grad[idx] = p - t
        n_used += 1

    if n_used == 0:
        return 0.0, grad.astype(np.float32)

    inv = 1.0 / n_used
    return total_loss * inv, (grad * inv).astype(np.float32)


CONFIG = {
    "loss": "listwise_softmax_v1",
    "group_by": "user_id+date",
    "k": 16,
    "lr": 0.01,
    "l2": 1e-6,
    "batch": 8192,
    "max_epochs": 40,
    "patience": 5,
}
