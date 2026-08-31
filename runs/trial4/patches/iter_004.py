import numpy as np
from harness.losses import register_loss

W_POINT = 1.0
W_RANK = 1.0


@register_loss("hybrid_pointwise_disc_listwise_v1", kind="pairwise")
def hybrid_pointwise_disc_listwise(z, y, groups):
    """Pointwise logloss on all rows + softmax CE on discriminative lists only.

    Pure listwise/pairwise objectives emit zero gradient for lists that are
    all-positive or all-negative, which starves the sparse id embeddings.  The
    pointwise term restores that signal on every row; the listwise term shapes
    the ordering inside the lists where the metric is decided.
    """
    z = np.asarray(z, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    zc = np.clip(z, -30.0, 30.0)
    n = zc.shape[0]

    # ---- pointwise part (all rows) ----
    p = 1.0 / (1.0 + np.exp(-zc))
    loss_pt = float(-(y * np.log(p + 1e-9) + (1.0 - y) * np.log(1.0 - p + 1e-9)).sum() / max(1, n))
    grad_pt = (p - y) / max(1, n)

    # ---- listwise part (discriminative lists only) ----
    _, inv = np.unique(np.asarray(groups).ravel(), return_inverse=True)
    g = int(inv.max()) + 1 if inv.size else 0
    grad_rk = np.zeros_like(zc)
    loss_rk = 0.0
    if g > 0:
        cnt = np.bincount(inv, minlength=g).astype(np.float64)
        pos = np.bincount(inv, weights=y, minlength=g)
        disc = (pos > 0.0) & (pos < cnt)
        n_disc = float(disc.sum())
        if n_disc > 0.0:
            e = np.exp(zc)
            den = np.bincount(inv, weights=e, minlength=g)
            den = np.maximum(den, 1e-300)
            s = e / den[inv]
            lse = np.log(den)
            posr = np.maximum(pos, 1.0)[inv]
            rowmask = disc[inv]
            # softmax CE against the uniform distribution over a list's positives
            terms = np.where(rowmask, y * (zc - lse[inv]) / posr, 0.0)
            loss_rk = float(-terms.sum() / n_disc)
            grad_rk = np.where(rowmask, s - y / posr, 0.0) / n_disc

    loss = W_POINT * loss_pt + W_RANK * loss_rk
    grad = W_POINT * grad_pt + W_RANK * grad_rk
    return float(loss), grad.astype(np.float32)


CONFIG = {
    "loss": "hybrid_pointwise_disc_listwise_v1",
    "group_by": "user_id+date",
    "batch": 16384,
    "max_epochs": 40,
    "patience": 5,
}
