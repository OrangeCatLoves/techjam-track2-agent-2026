import numpy as np
from harness.losses import register_loss

ALPHA = 0.5  # weight on the listwise term; (1-ALPHA) on the pointwise term


@register_loss("hybrid_softmax_pointwise_v1", kind="listwise")
def hybrid_softmax_pointwise(z, y, groups):
    """Convex blend of a within-list softmax CE and plain pointwise logloss.

    Pure ranking losses (BPR, lambdarank, listwise softmax) emit no gradient for
    lists that are entirely positive or entirely negative, and none for
    singletons. Under (user_id, date) grouping those lists are the majority, so
    a large share of the training rows contribute nothing at all. The pointwise
    term keeps them in the objective, supplying absolute propensity that the
    shift-invariant softmax cannot represent; the softmax term supplies the
    within-list ordering signal that the metric actually scores.
    """
    z = np.asarray(z, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = z.shape[0]
    if n == 0:
        return 0.0, np.zeros(0, dtype=np.float32)

    zc = np.clip(z, -30.0, 30.0)

    # ---- pointwise term: every row, including degenerate lists ----
    p = 1.0 / (1.0 + np.exp(-zc))
    loss_pt = float(-(y * np.log(p + 1e-9) + (1.0 - y) * np.log(1.0 - p + 1e-9)).mean())
    grad_pt = (p - y) / n

    # ---- listwise term: mixed lists only, each list weighted equally ----
    _, inv = np.unique(groups, return_inverse=True)
    n_groups = int(inv.max()) + 1
    size = np.bincount(inv, minlength=n_groups).astype(np.float64)
    npos = np.bincount(inv, weights=y, minlength=n_groups)
    mixed = (npos > 0.0) & (npos < size)
    n_mixed = int(mixed.sum())

    loss_ls = 0.0
    grad_ls = np.zeros(n, dtype=np.float64)

    if n_mixed > 0:
        row_mixed = mixed[inv]
        zmax = np.full(n_groups, -1e30, dtype=np.float64)
        np.maximum.at(zmax, inv, zc)
        e = np.exp(zc - zmax[inv])
        e = np.where(row_mixed, e, 0.0)
        denom = np.bincount(inv, weights=e, minlength=n_groups)
        denom = np.where(denom > 0.0, denom, 1.0)
        sm = e / denom[inv]
        pos_per_row = np.where(npos[inv] > 0.0, npos[inv], 1.0)
        target = np.where(row_mixed, y / pos_per_row, 0.0)
        loss_ls = float(-(target * np.log(sm + 1e-12)).sum() / n_mixed)
        grad_ls = (sm - target) / n_mixed

    loss = (1.0 - ALPHA) * loss_pt + ALPHA * loss_ls
    grad = (1.0 - ALPHA) * grad_pt + ALPHA * grad_ls
    return float(loss), grad.astype(np.float32)


CONFIG = {"loss": "hybrid_softmax_pointwise_v1", "group_by": "user_id+date"}
