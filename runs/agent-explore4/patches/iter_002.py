import numpy as np
from harness.losses import register_loss, sigmoid


@register_loss("softmax_listwise_anchored_v1", kind="listwise")
def softmax_listwise_anchored(z, y, groups):
    """Listwise softmax cross-entropy + a pointwise anchor.

    Listwise part: within each user's list, softmax the logits and take
    cross-entropy against the label distribution y / sum(y). Lists with no
    positive, or with a single row, carry no ranking information and are
    excluded from this term.

    Pointwise part: plain logloss over every row. Its job is to keep the
    user_id / video_id / author_id embeddings training on the homogeneous
    lists that the ranking term discards. The previous pairwise run left
    those embeddings near their initialisation while dur_bucket and tab
    absorbed the signal; that is the failure this term is meant to prevent.
    """
    z = np.asarray(z, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = max(1, z.size)
    W = 0.5

    # --- pointwise anchor -------------------------------------------------
    p_sig = 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))
    loss_point = float(-(y * np.log(p_sig + 1e-9)
                         + (1.0 - y) * np.log(1.0 - p_sig + 1e-9)).mean())
    g_point = (p_sig - y) / n

    # --- listwise softmax cross-entropy -----------------------------------
    g_soft = np.zeros_like(z)
    loss_soft = 0.0
    if z.size:
        _, idx = np.unique(np.asarray(groups), return_inverse=True)
        idx = idx.astype(np.int64)
        ng = int(idx.max()) + 1

        gmax = np.full(ng, -np.inf, dtype=np.float64)
        np.maximum.at(gmax, idx, z)
        e = np.exp(np.clip(z - gmax[idx], -30.0, 30.0))
        s = np.bincount(idx, weights=e, minlength=ng)
        p = e / np.maximum(s[idx], 1e-12)

        pos = np.bincount(idx, weights=y, minlength=ng)
        size = np.bincount(idx, minlength=ng)
        valid = (pos[idx] > 0.0) & (pos[idx] < size[idx]) & (size[idx] > 1)
        t = np.where(valid, y / np.maximum(pos[idx], 1e-12), 0.0)

        nv = float(max(1, int(valid.sum())))
        loss_soft = float(-(t * np.log(p + 1e-12) * valid).sum() / nv)
        g_soft = ((p - t) * valid) / nv

    loss = (1.0 - W) * loss_point + W * loss_soft
    grad = (1.0 - W) * g_point + W * g_soft
    return float(loss), grad.astype(np.float32)


CONFIG = {"loss": "softmax_listwise_anchored_v1",
          "group_by": "user_id",
          "batch": 32768,
          "lr": 0.003,
          "max_epochs": 40,
          "patience": 5}
