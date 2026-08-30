import numpy as np
from harness.losses import register_loss


@register_loss("list_centered_logistic_v1", kind="listwise")
def list_centered_logistic(z, y, groups):
    """Binary logloss on list-centred logits.

    d_i = z_i - mean_{g(i)} z, then standard logistic loss on d.

    Why this is not another softmax variant: probabilities are never
    normalised across the items of a list, so a list with zero positives or
    with several positives is still perfectly well defined. Only the
    per-list additive offset -- the part of the score that cannot change
    within-user ordering, and therefore cannot change GAUC or nDCG@5 -- is
    removed from what the model is asked to fit. The chain rule through the
    centering makes the returned gradient the list-centred residual, which
    sums to zero inside every list: capacity is spent entirely on contrast.

    Lists that carry no ordering information (all-positive or all-negative)
    are down-weighted rather than dropped, since they still shape the item
    embeddings that generalise to other lists.
    """
    z = np.asarray(z, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    _, inv = np.unique(groups, return_inverse=True)
    n = np.bincount(inv).astype(np.float64)
    n = np.maximum(n, 1.0)

    mean_z = np.bincount(inv, weights=z) / n
    d = z - mean_z[inv]
    p = 1.0 / (1.0 + np.exp(-np.clip(d, -30.0, 30.0)))

    pos = np.bincount(inv, weights=y)
    degenerate = (pos <= 0.0) | (pos >= n)

    # equal contribution per list, degenerate lists at quarter weight
    w_list = np.where(degenerate, 0.25, 1.0) / n
    w = w_list[inv]
    s = w.sum()
    if s <= 0.0:
        s = 1.0
    w = w / s

    loss = float(-(w * (y * np.log(p + 1e-9)
                        + (1.0 - y) * np.log(1.0 - p + 1e-9))).sum())

    r = w * (p - y)
    g = r - (np.bincount(inv, weights=r) / n)[inv]
    return loss, g.astype(np.float32)


CONFIG = {"loss": "list_centered_logistic_v1",
          "group_by": "user_id+date",
          "max_epochs": 40,
          "patience": 5}
