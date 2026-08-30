import numpy as np
from harness.losses import register_loss


@register_loss("list_centered_logloss_v1", kind="listwise")
def list_centered_logloss(z, y, groups):
    """Pointwise logloss on logits centered within each impression list.

    Both metrics rank inside one user's list, so any part of the score that is
    constant across that list is invisible to them.  Fitting s_i = z_i - mean_g(z)
    is conditional / fixed-effects logistic regression: the per-list intercept is
    profiled out, so the global bias and every user-side first-order term get zero
    net gradient, and the remaining gradient is spent purely on within-list order.

    Unlike BPR / softmax / hinge, every row keeps a dense individual gradient and
    all-negative and all-positive lists still contribute (they push apart-scored
    items back together), so no training data is discarded.

    A small quadratic penalty on each list mean pins the direction the centering
    made unidentifiable, preventing free drift of the embedding scale.
    """
    z = np.asarray(z, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    n = max(1, z.shape[0])

    _, inv = np.unique(np.asarray(groups).ravel(), return_inverse=True)
    cnt = np.bincount(inv).astype(np.float64)
    cnt = np.maximum(cnt, 1.0)
    mean_z = np.bincount(inv, weights=z) / cnt

    s = z - mean_z[inv]
    p = 1.0 / (1.0 + np.exp(-np.clip(s, -30.0, 30.0)))

    loss = float(-(y * np.log(p + 1e-9) + (1.0 - y) * np.log(1.0 - p + 1e-9)).sum() / n)

    g = (p - y) / n
    g_mean = np.bincount(inv, weights=g) / cnt
    grad = g - g_mean[inv]

    lam = 0.01
    loss += float(lam * (cnt * mean_z * mean_z).sum() / n)
    grad = grad + (2.0 * lam / n) * mean_z[inv]

    return loss, grad.astype(np.float32)


CONFIG = {"loss": "list_centered_logloss_v1", "group_by": "user_id"}
