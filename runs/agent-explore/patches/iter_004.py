import numpy as np
from harness.losses import register_loss

# Margin past which a correctly ordered pair produces exactly zero gradient.
MARGIN = 1.0
# Weight on the pointwise term applied ONLY to all-positive / all-negative lists,
# which contribute no pairs and would otherwise receive no gradient at all.
DEGEN_WEIGHT = 0.25


def _group_slices(groups):
    order = np.argsort(groups, kind="stable")
    gs = np.asarray(groups)[order]
    if gs.size == 0:
        return order, np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    starts = np.flatnonzero(np.concatenate(([True], gs[1:] != gs[:-1])))
    ends = np.concatenate((starts[1:], [gs.size]))
    return order, starts, ends


@register_loss("metric_weighted_hinge_v1", kind="listwise")
def metric_weighted_hinge(z, y, groups):
    """Squared-hinge pairwise loss weighted to mirror the metric's aggregation.

    Per list with at least one positive and one negative, every (pos, neg) pair
    gets a saturating squared hinge on the score gap. Two weightings are carried
    separately and each normalised over the batch before being mixed 50/50:

      * GAUC-aligned: 1/n_neg per pair. A list's pairs then sum to n_pos, which
        is exactly how GAUC weights a user.
      * nDCG@5-aligned: |delta disc| / IDCG for swapping the pair under the
        current ordering, with the discount zeroed beyond rank 5.

    Degenerate lists (all-positive or all-negative) carry a small pointwise
    logloss so their rows still train the embedding table.
    """
    z = np.asarray(z, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    n = z.size

    g_auc = np.zeros(n, dtype=np.float64)
    g_ndcg = np.zeros(n, dtype=np.float64)
    g_pt = np.zeros(n, dtype=np.float64)
    l_auc = l_ndcg = l_pt = 0.0
    s_auc = s_ndcg = s_pt = 0.0

    order, starts, ends = _group_slices(groups)

    for a, b in zip(starts, ends):
        idx = order[a:b]
        m = idx.size
        if m == 0:
            continue
        zz = z[idx]
        yy = y[idx]
        npos = int((yy > 0.5).sum())
        nneg = m - npos

        if npos == 0 or nneg == 0:
            p = 1.0 / (1.0 + np.exp(-np.clip(zz, -30.0, 30.0)))
            w = 1.0 / m
            l_pt += float(-(w * (yy * np.log(p + 1e-9)
                                 + (1.0 - yy) * np.log(1.0 - p + 1e-9))).sum())
            g_pt[idx] += w * (p - yy)
            s_pt += 1.0
            continue

        P = np.flatnonzero(yy > 0.5)
        N = np.flatnonzero(yy <= 0.5)

        d = zz[P][:, None] - zz[N][None, :]
        slack = np.maximum(MARGIN - d, 0.0)
        base_l = 0.5 * slack * slack
        base_gd = -slack                      # dL/dd, zero once the margin is met

        rank = np.empty(m, dtype=np.int64)
        rank[np.argsort(-zz, kind="stable")] = np.arange(m)
        disc = np.where(rank < 5, 1.0 / np.log2(rank + 2.0), 0.0)
        idcg = float((1.0 / np.log2(np.arange(min(npos, 5)) + 2.0)).sum())
        if idcg <= 0.0:
            idcg = 1.0
        w_ndcg = np.abs(disc[P][:, None] - disc[N][None, :]) / idcg
        w_auc = np.full_like(d, 1.0 / nneg)

        l_auc += float((w_auc * base_l).sum())
        s_auc += float(w_auc.sum())
        l_ndcg += float((w_ndcg * base_l).sum())
        s_ndcg += float(w_ndcg.sum())

        ga = w_auc * base_gd
        gn = w_ndcg * base_gd
        np.add.at(g_auc, idx[P], ga.sum(axis=1))
        np.add.at(g_auc, idx[N], -ga.sum(axis=0))
        np.add.at(g_ndcg, idx[P], gn.sum(axis=1))
        np.add.at(g_ndcg, idx[N], -gn.sum(axis=0))

    s_auc = max(s_auc, 1e-12)
    s_ndcg = max(s_ndcg, 1e-12)
    s_pt = max(s_pt, 1e-12)

    grad = (0.5 * (g_auc / s_auc)
            + 0.5 * (g_ndcg / s_ndcg)
            + DEGEN_WEIGHT * (g_pt / s_pt))
    loss = (0.5 * (l_auc / s_auc)
            + 0.5 * (l_ndcg / s_ndcg)
            + DEGEN_WEIGHT * (l_pt / s_pt))

    if not np.all(np.isfinite(grad)):
        grad = np.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)

    return float(loss), grad.astype(np.float32)


CONFIG = {"loss": "metric_weighted_hinge_v1", "group_by": "user_id+date"}
