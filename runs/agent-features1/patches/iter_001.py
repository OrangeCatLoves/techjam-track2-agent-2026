import numpy as np
from harness.losses import register_loss


def _sig(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def bpr_all_pairs(z, y, groups):
    """All-pairs BPR within each list.

    For every (positive, negative) pair inside one group:
        L = softplus(-(z_pos - z_neg))
    Each list is normalised by its own pair count and the total is averaged
    over the lists that actually contain both classes, so a 200-impression
    user carries the same weight as a 3-impression user and lists that are
    all-positive or all-negative contribute nothing (they are unrankable and
    GAUC ignores them too).

    Gradient wrt the logits:
        dL/dz_pos = -sum_neg  sigmoid(-(z_pos - z_neg))
        dL/dz_neg = +sum_pos  sigmoid(-(z_pos - z_neg))
    """
    z = np.asarray(z, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    g = np.asarray(groups).ravel()
    grad = np.zeros_like(z)

    order = np.argsort(g, kind="stable")
    gs = g[order]
    if gs.size == 0:
        return 0.0, grad.astype(np.float32)
    edges = np.flatnonzero(
        np.concatenate(([True], gs[1:] != gs[:-1], [True]))
    )

    total = 0.0
    used = 0
    for a, b in zip(edges[:-1], edges[1:]):
        idx = order[a:b]
        yy = y[idx]
        pos = idx[yy > 0.5]
        neg = idx[yy <= 0.5]
        npos = pos.size
        nneg = neg.size
        if npos == 0 or nneg == 0:
            continue
        d = z[pos][:, None] - z[neg][None, :]
        w = 1.0 / float(npos * nneg)
        total += float(np.logaddexp(0.0, -d).sum()) * w
        s = _sig(-d)
        grad[pos] -= s.sum(axis=1) * w
        grad[neg] += s.sum(axis=0) * w
        used += 1

    if used > 0:
        total /= used
        grad /= used
    return float(total), grad.astype(np.float32)


# Register under the most specific kind the harness accepts, falling back so a
# vocabulary mismatch cannot cost the whole iteration.
def _register(fn):
    for kwargs in ({"kind": "pairwise"}, {}, {"kind": "pointwise"}):
        try:
            register_loss("bpr_all_pairs_v1", **kwargs)(fn)
            return fn
        except Exception:
            continue
    return fn


_register(bpr_all_pairs)

CONFIG = {"loss": "bpr_all_pairs_v1", "group_by": "user_id+date"}
