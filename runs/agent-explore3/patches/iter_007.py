import numpy as np
from harness.losses import register_loss

# Soft-rank temperature. Smaller is closer to the true (discontinuous) metric
# but gives weaker gradients; 0.5 is the compromise.
TAU = 0.5
# Temperature of the soft top-k gate.
TAUC = 1.0
# The cutoff the scored metric actually uses.
CUT = 5.0
LOG2 = np.log(2.0)


@register_loss("approx_ndcg5_session_lists_v1", kind="listwise")
def approx_ndcg5_session_lists(z, y, groups):
    """Smooth surrogate for nDCG@5, differentiated exactly.

    Soft rank of item i inside its list:
        r_i = 1 + sum_{j != i} sigmoid((z_j - z_i) / TAU)
    Discount d_i = 1 / log2(1 + r_i), soft top-k gate
        m_i = sigmoid((CUT + 0.5 - r_i) / TAUC)
    DCG = sum_i y_i * d_i * m_i, normalised by the hard ideal DCG@5.
    Loss = -mean over lists of DCG/IDCG.

    Gradient. With P[i, j] = sigmoid((z_j - z_i) / TAU) and P[i, i] = 0,
    dr_i/dz_k = S[i, k] for k != i and -sum_j S[i, j] for k == i, where
    S = P * (1 - P) / TAU. Writing a_i = y_i * d(d_i m_i)/dr_i, the whole
    thing collapses to
        dDCG/dz_k = sum_i S[i, k] * (a_i - a_k)
    which is two matrix reductions per list, no explicit pair loop.

    Lists with zero positives or zero negatives are skipped: they are a
    constant in both GAUC (excluded) and nDCG@5 (always 0 or always 1), so
    their gradient is noise against the scored objective.
    """
    z = np.asarray(z, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    g = np.asarray(groups).ravel()

    grad = np.zeros_like(z)
    total = 0.0
    n_lists = 0

    order = np.argsort(g, kind="stable")
    gs = g[order]
    starts = np.flatnonzero(np.concatenate(([True], gs[1:] != gs[:-1])))
    ends = np.concatenate((starts[1:], [gs.shape[0]]))

    for s, e in zip(starts, ends):
        idx = order[s:e]
        n = idx.shape[0]
        if n < 2:
            continue
        yy = y[idx]
        npos = float(yy.sum())
        if npos <= 0.0 or npos >= n:
            continue

        zz = z[idx]
        # P[i, j] = sigmoid((z_j - z_i) / TAU)
        diff = (zz[None, :] - zz[:, None]) / TAU
        P = 1.0 / (1.0 + np.exp(-np.clip(diff, -30.0, 30.0)))
        np.fill_diagonal(P, 0.0)

        rank = 1.0 + P.sum(axis=1)
        logr = np.log2(1.0 + rank)
        d = 1.0 / logr
        m = 1.0 / (1.0 + np.exp(-np.clip((CUT + 0.5 - rank) / TAUC, -30.0, 30.0)))

        k_ideal = int(min(npos, CUT))
        idcg = float(np.sum(1.0 / np.log2(np.arange(k_ideal) + 2.0)))
        if idcg <= 0.0:
            continue

        dcg = float(np.sum(yy * d * m))
        total += -(dcg / idcg)

        dd = -1.0 / ((1.0 + rank) * LOG2 * logr * logr)
        dm = -m * (1.0 - m) / TAUC
        a = yy * (m * dd + d * dm)

        S = P * (1.0 - P) / TAU
        gk = S.T.dot(a) - a * S.sum(axis=0)

        grad[idx] += -(gk / idcg)
        n_lists += 1

    if n_lists > 0:
        total /= n_lists
        grad /= n_lists

    if not np.all(np.isfinite(grad)):
        grad = np.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)

    return float(total), grad.astype(np.float32)


CONFIG = {
    "loss": "approx_ndcg5_session_lists_v1",
    "group_by": "user_id+date",
}
