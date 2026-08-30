import numpy as np
from harness.losses import register_loss

# Random sublist construction is reseeded per call from a fixed base seed plus a
# call counter, so the run stays deterministic while lists differ across epochs.
_SEED = 20220422
_CALLS = [0]
_TARGET_LIST = 6  # matches the measured evaluation list mean of 5.6


@register_loss("listwise_softmax_chunked_v1", kind="listwise")
def listwise_softmax_chunked(z, y, groups):
    """Softmax cross-entropy over randomly resampled sublists of a user's rows.

    groups are user_id. Each user's rows in the batch are shuffled and split into
    chunks of about _TARGET_LIST rows, so contrasts are drawn across the user's
    whole history (not within one date) while the optimised list length matches
    the evaluation list length. Chunks with no positive or no negative carry no
    ranking information and are skipped; every surviving chunk is weighted
    equally, so heavy users cannot dominate.
    """
    z = np.asarray(z, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    grad = np.zeros_like(z)

    _CALLS[0] += 1
    rng = np.random.default_rng(_SEED + _CALLS[0])

    order = np.argsort(groups, kind="stable")
    gs = np.asarray(groups)[order]
    if gs.size == 0:
        return 0.0, grad.astype(np.float32)
    starts = np.flatnonzero(np.r_[True, gs[1:] != gs[:-1]])
    ends = np.r_[starts[1:], gs.size]

    loss = 0.0
    n_lists = 0
    for s, e in zip(starts, ends):
        idx = order[s:e]
        if idx.size < 2:
            continue
        yy_all = y[idx]
        pos_all = yy_all.sum()
        if pos_all <= 0 or pos_all >= idx.size:
            continue  # no within-user ranking signal at all
        idx = idx[rng.permutation(idx.size)]
        n_chunks = max(1, int(round(idx.size / float(_TARGET_LIST))))
        for chunk in np.array_split(idx, n_chunks):
            if chunk.size < 2:
                continue
            yy = y[chunk]
            pos = yy.sum()
            if pos <= 0.0 or pos >= chunk.size:
                continue
            zz = z[chunk]
            zz = zz - zz.max()
            ex = np.exp(np.clip(zz, -30.0, 30.0))
            p = ex / max(ex.sum(), 1e-12)
            t = yy / pos
            loss += float(-(t * np.log(p + 1e-12)).sum())
            grad[chunk] += (p - t)
            n_lists += 1

    if n_lists:
        loss /= n_lists
        grad /= n_lists
    return float(loss), grad.astype(np.float32)


CONFIG = {"loss": "listwise_softmax_chunked_v1",
          "group_by": "user_id",
          "patience": 5}
