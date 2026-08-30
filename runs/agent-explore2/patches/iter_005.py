import numpy as np
from harness.losses import register_loss


@register_loss("within_list_centered_logloss_v1", kind="listwise")
def within_list_centered_logloss(z, y, groups):
    """Logistic loss on scores centred within each list.

    GAUC and nDCG@5 are invariant to adding a constant to every score in a
    user's list, so any effort the model spends on the list-level offset is
    wasted. Centring z by its list mean makes the objective share that
    invariance: the gradient at each row is its own pointwise residual minus
    the list-mean residual, so only within-list contrast is rewarded.

    Unlike a softmax listwise loss this places no sum-to-one constraint on the
    list, so a list with several positives (the common case here: mean list
    size ~5.8, positive rate ~0.31) is not forced to pick a single winner.
    Each list is weighted equally, matching the per-user averaging of both
    metrics. Singleton lists centre to zero and contribute no gradient, which
    is correct -- they carry no within-list ordering information.
    """
    z = np.asarray(z, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()

    # Dense list index, vectorised (many thousands of lists per batch).
    _, idx = np.unique(np.asarray(groups).ravel(), return_inverse=True)
    n = np.bincount(idx).astype(np.float64)
    n = np.maximum(n, 1.0)

    # Centre scores within each list.
    mean_z = np.bincount(idx, weights=z) / n
    zc = z - mean_z[idx]

    p = 1.0 / (1.0 + np.exp(-np.clip(zc, -30.0, 30.0)))

    # Every list contributes equally, matching per-user metric averaging.
    w = 1.0 / n[idx]
    total = w.sum()
    if total <= 0.0:
        return 0.0, np.zeros_like(z, dtype=np.float32)
    w = w / total

    loss = float(-(w * (y * np.log(p + 1e-9)
                        + (1.0 - y) * np.log(1.0 - p + 1e-9))).sum())

    # dL/dz_j = a_j - mean_over_list(a),  a_i = w_i * (p_i - y_i)
    a = w * (p - y)
    mean_a = np.bincount(idx, weights=a) / n
    grad = a - mean_a[idx]

    if not np.all(np.isfinite(grad)):
        grad = np.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)

    return loss, grad.astype(np.float32)


CONFIG = {"loss": "within_list_centered_logloss_v1", "group_by": "user_id+date"}
