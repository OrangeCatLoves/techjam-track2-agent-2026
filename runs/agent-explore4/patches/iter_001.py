import numpy as np
from harness.losses import register_loss

_CHUNK = 1024
_ALPHA = 0.1  # weight on the pointwise anchor term


def _sig(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


@register_loss("bpr_allpairs_listnorm_v1", kind="pairwise")
def bpr_allpairs_listnorm(z, y, groups):
    """All-pairs BPR within each list, every list weighted equally.

    For a list with P positives and N negatives, every (pos, neg) pair gets
    weight 1/(P*N), and each list with at least one of each contributes 1/L of
    the total (L = number of such lists in the batch). Lists that are all
    positive or all negative produce no pairs -- they also cannot change GAUC
    or nDCG@5, so that is correct rather than a loss of signal. A small
    pointwise logloss term keeps those rows anchored and stops the embedding
    scale drifting, since BPR alone only constrains within-list differences.

    Pairs are formed with dense masked blocks over group-aligned chunks, so the
    computation is exact (not sampled), fully deterministic, and never builds a
    matrix larger than ~_CHUNK squared.
    """
    z = np.asarray(z, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    g = np.asarray(groups).ravel()
    n = z.shape[0]
    grad = np.zeros(n, dtype=np.float64)
    loss = 0.0
    if n == 0:
        return 0.0, grad.astype(np.float32)

    order = np.argsort(g, kind="stable")
    gs = g[order]
    zs = z[order]
    ys = y[order]

    uniq, starts, counts = np.unique(gs, return_index=True, return_counts=True)
    csum = np.concatenate(([0.0], np.cumsum(ys)))
    npos = csum[starts + counts] - csum[starts]
    nneg = counts.astype(np.float64) - npos
    valid = (npos > 0) & (nneg > 0)
    n_valid = int(valid.sum())

    if n_valid > 0:
        pair_w = np.zeros(uniq.shape[0], dtype=np.float64)
        pair_w[valid] = 1.0 / (npos[valid] * nneg[valid] * float(n_valid))
        row_w = np.repeat(pair_w, counts)
        gid = np.repeat(np.arange(uniq.shape[0]), counts)

        # assign whole groups to chunks of about _CHUNK rows; chunk ids are
        # non-decreasing, so each chunk is a contiguous slice of sorted rows
        offs = np.cumsum(counts) - counts
        chunk_of_group = offs // _CHUNK
        row_chunk = np.repeat(chunk_of_group, counts)
        _, c_start, c_count = np.unique(row_chunk, return_index=True,
                                        return_counts=True)

        for s, c in zip(c_start, c_count):
            e = s + c
            zc = zs[s:e]
            yc = ys[s:e]
            wc = row_w[s:e]
            gc = gid[s:e]
            if wc.max() <= 0.0:
                continue
            same = gc[:, None] == gc[None, :]
            mask = same & (yc[:, None] > 0.5) & (yc[None, :] < 0.5)
            if not mask.any():
                continue
            diff = zc[:, None] - zc[None, :]
            w2 = np.where(mask, wc[:, None], 0.0)
            loss += float((w2 * np.logaddexp(0.0, -diff)).sum())
            sg = w2 * _sig(-diff)
            gc_grad = -sg.sum(axis=1) + sg.sum(axis=0)
            grad[order[s:e]] += gc_grad

    # pointwise anchor
    p = _sig(z)
    loss += _ALPHA * float(-(y * np.log(p + 1e-9)
                             + (1.0 - y) * np.log(1.0 - p + 1e-9)).sum() / n)
    grad += _ALPHA * (p - y) / n

    return loss, grad.astype(np.float32)


CONFIG = {"loss": "bpr_allpairs_listnorm_v1",
          "group_by": "user_id+date",
          "max_epochs": 60,
          "patience": 6}
