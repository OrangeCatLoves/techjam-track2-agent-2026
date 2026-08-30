import numpy as np
from harness.losses import register_loss

HOMOGENEOUS_LIST_WEIGHT = 0.15


@register_loss("list_mixedness_balanced_logloss_v1", kind="pointwise")
def list_mixedness_balanced_logloss(z, y, groups):
    """Pointwise logloss with sampling-style reweighting.

    Within each list (groups = user_id+date):
      * label-mixed lists get total weight 1.0, split 0.5 across their
        positives and 0.5 across their negatives, so a list with one
        positive among nine negatives still pushes that positive up
        instead of being drowned by the majority class;
      * all-positive / all-negative lists cannot change any within-list
        ordering, so they keep only HOMOGENEOUS_LIST_WEIGHT of a list's
        weight -- enough to keep the global item-side prior and cold-start
        behaviour calibrated, but no longer dominant.
    Weights are normalised to sum to 1, matching the reference pointwise
    gradient scale (mean over the batch).
    """
    z = np.asarray(z, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    _, inv = np.unique(np.asarray(groups), return_inverse=True)
    n_rows = np.bincount(inv).astype(np.float64)
    n_pos = np.bincount(inv, weights=y).astype(np.float64)
    n_neg = n_rows - n_pos
    mixed = (n_pos > 0.0) & (n_neg > 0.0)

    pos_w = np.zeros_like(n_rows)
    neg_w = np.zeros_like(n_rows)
    flat_w = np.zeros_like(n_rows)
    pos_w[mixed] = 0.5 / n_pos[mixed]
    neg_w[mixed] = 0.5 / n_neg[mixed]
    homo = ~mixed
    flat_w[homo] = HOMOGENEOUS_LIST_WEIGHT / n_rows[homo]

    is_pos = y > 0.5
    weight = np.where(mixed[inv],
                      np.where(is_pos, pos_w[inv], neg_w[inv]),
                      flat_w[inv])
    total = weight.sum()
    if not np.isfinite(total) or total <= 0.0:
        weight = np.full_like(z, 1.0 / max(1, z.size))
    else:
        weight = weight / total

    p = 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))
    loss = float(-(weight * (y * np.log(p + 1e-9)
                             + (1.0 - y) * np.log(1.0 - p + 1e-9))).sum())
    grad = (weight * (p - y)).astype(np.float32)
    return loss, grad


CONFIG = {"loss": "list_mixedness_balanced_logloss_v1",
          "group_by": "user_id+date"}
