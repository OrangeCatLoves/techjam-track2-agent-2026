import numpy as np
from harness.losses import register_loss

GAMMA = 1.0


@register_loss("hard_example_focal_logloss_v1", kind="pointwise")
def hard_example_focal_logloss(z, y, groups):
    """Pointwise logloss with focal down-weighting of already-fit rows.

    L_i = -(1 - p_t)^gamma * log(p_t),  p_t = p if y=1 else 1-p.

    Rows the model already gets right (large p_t) contribute almost nothing.
    On this data those are the rows explained by duration bucket and tab, which
    is exactly where the current model concentrates its capacity. Suppressing
    them redirects gradient onto rows whose ordering needs the user/video/author
    crosses.

    Gradient, with s = p(1-p) and sigma = 2y-1 (so dp_t/dz = sigma * s):
        dL/dp_t = gamma*(1-p_t)^(gamma-1)*log(p_t) - (1-p_t)^gamma / p_t
        dL/dz   = dL/dp_t * sigma * s
    At gamma = 0 this reduces to (p - y), the usual logloss gradient.
    """
    z = np.asarray(z, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = max(1, z.shape[0])

    p = 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))
    p = np.clip(p, 1e-7, 1.0 - 1e-7)

    pt = np.where(y > 0.5, p, 1.0 - p)
    pt = np.clip(pt, 1e-7, 1.0 - 1e-7)
    one_minus = 1.0 - pt

    focal = one_minus ** GAMMA
    loss = float(-(focal * np.log(pt)).sum() / n)

    if GAMMA == 0.0:
        dL_dpt = -1.0 / pt
    else:
        dL_dpt = GAMMA * (one_minus ** (GAMMA - 1.0)) * np.log(pt) - focal / pt

    sigma = 2.0 * y - 1.0
    s = p * (1.0 - p)
    grad = dL_dpt * sigma * s / n

    grad = np.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
    return loss, grad.astype(np.float32)


CONFIG = {
    "loss": "hard_example_focal_logloss_v1",
    "group_by": "user_id+date",
    "k": 16,
    "lr": 0.003,
    "l2": 5e-6,
    "batch": 2048,
    "max_epochs": 40,
    "patience": 8,
    "ensemble": 3,
    "normalise": "within_user_rank",
}
