import numpy as np
from harness.losses import register_loss


@register_loss("list_centered_logistic_v1", kind="listwise")
def list_centered_logistic(z, y, groups):
    """Logistic loss on list-centered logits.

    Both scored metrics are invariant to adding a constant to every score in a
    user's list, so any component of the model that is constant within a list is
    pure wasted capacity. Pointwise logloss does not know that, and BPR/softmax
    only remove it implicitly. Here the logits are explicitly centered within
    each list before the sigmoid, and the gradient is chained through that
    centering, so a per-list shift receives exactly zero gradient.

    Lists with no positives or no negatives carry no within-list ranking
    information (their nDCG@5 is fixed and they are excluded from GAUC), so they
    are given zero weight. Remaining lists are weighted equally.
    """
    z = np.asarray(z, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()

    _, inv = np.unique(groups, return_inverse=True)
    n_list = np.bincount(inv).astype(np.float64)
    n_row = n_list[inv]

    mean_z = np.bincount(inv, weights=z) / n_list
    zc = np.clip(z - mean_z[inv], -30.0, 30.0)
    p = 1.0 / (1.0 + np.exp(-zc))

    npos = np.bincount(inv, weights=y)
    discriminative = (npos > 0.0) & (npos < n_list)
    w = discriminative[inv].astype(np.float64) / n_row
    total = w.sum()
    if total <= 0.0:
        return 0.0, np.zeros_like(z, dtype=np.float32)
    w /= total

    loss = float(-(w * (y * np.log(p + 1e-9)
                        + (1.0 - y) * np.log(1.0 - p + 1e-9))).sum())

    r = w * (p - y)
    # d zc_j / d z_i = delta_ij - 1 / n_g  for i, j in the same list g
    r_mean = np.bincount(inv, weights=r) / n_list
    grad = r - r_mean[inv]
    return loss, grad.astype(np.float32)


CONFIG = {
    "loss": "list_centered_logistic_v1",
    "group_by": "user_id+date",
    "k": 16,
    "lr": 0.001,
    "max_epochs": 30,
    "patience": 4,
    "ensemble": 3,
    "normalise": "within_user_rank",
}
