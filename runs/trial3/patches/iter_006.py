import numpy as np
from harness.losses import register_loss, sigmoid

# Deterministic given a fixed harness seed: the module is imported once per run
# and the call sequence is fixed by the trainer's own seeded batching.
_RNG = np.random.default_rng(20260831)

# Evaluation lists average 5.58 rows (user_id, valid). Train sub-lists are cut
# to this size so the objective sees the list length it is scored on.
_M = 6


@register_loss("eval_len_sublist_softmax_v1", kind="listwise")
def eval_len_sublist_softmax(z, y, groups):
    """Listwise softmax over evaluation-length sub-lists of each user's list.

    Each group is randomly permuted and chunked into blocks of _M rows. Blocks
    that are all-positive or all-negative are dropped entirely: their internal
    ordering cannot move GAUC (undefined) or nDCG@5 (pinned at 1 or 0), so any
    gradient spent there is capacity spent off-metric. Surviving blocks get
    cross-entropy against the normalised label distribution, which concentrates
    the gradient on the top of a short list.
    """
    z = np.asarray(z, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    grad = np.zeros_like(z)

    order = np.argsort(groups, kind="stable")
    gs = np.asarray(groups)[order]
    if gs.size == 0:
        return 0.0, grad.astype(np.float32)
    edges = np.flatnonzero(np.concatenate(([True], gs[1:] != gs[:-1], [True])))

    loss = 0.0
    total = 0.0
    for a, b in zip(edges[:-1], edges[1:]):
        idx = order[a:b]
        n = idx.size
        if n < 2:
            continue
        yg = y[idx]
        s = yg.sum()
        if s <= 0.0 or s >= n:
            continue
        idx = idx[_RNG.permutation(n)]
        for st in range(0, n, _M):
            blk = idx[st:st + _M]
            if blk.size < 2:
                continue
            yb = y[blk]
            sb = yb.sum()
            if sb <= 0.0 or sb >= blk.size:
                continue
            zb = z[blk]
            zb = zb - zb.max()
            e = np.exp(np.clip(zb, -30.0, 30.0))
            p = e / e.sum()
            t = yb / sb
            loss += float(-(t * np.log(p + 1e-12)).sum())
            grad[blk] += (p - t)
            total += blk.size

    if total <= 0.0:
        # Degenerate batch (no mixed sub-list survived): fall back to pointwise
        # logloss so training never stalls on a bad shard.
        p = sigmoid(np.clip(z, -30.0, 30.0))
        n = max(1, z.size)
        pl = float(-(y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9)).sum()) / n
        return pl, ((p - y) / n).astype(np.float32)

    grad /= total
    loss /= total
    return float(loss), grad.astype(np.float32)


CONFIG = {
    "loss": "eval_len_sublist_softmax_v1",
    "group_by": "user_id",
    "k": 16,
    "lr": 0.002,
    "l2": 1e-5,
    "batch": 8192,
    "max_epochs": 20,
    "patience": 4,
}
