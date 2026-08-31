import numpy as np
from harness.features.registry import register_feature


@register_feature("user_prior_rate")
def build_user_prior_rate(frame, stats):
    """The user's own smoothed historical long_view propensity.

    Constant within a list, so its first-order term is a no-op by construction.
    It is here only to be crossed: user_id is measured to be badly learned
    (norm 0.320 against dur_bucket 1.332) because each user carries ~43 training
    rows, whereas a quantile bucket of this rate carries ~100k. It is the same
    user information at a resolution the data can actually estimate, so the
    user x dur_bucket and user x video_id crosses get a signal they currently
    lack.
    """
    r = np.asarray(stats.label_rate("user_id"), dtype=np.float64)
    return np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)


@register_feature("dur_vs_slate_mean")
def build_dur_vs_slate_mean(frame, stats):
    """Log duration minus the mean log duration of the same user-day slate.

    dur_bucket is a global quantile bucket, so two items in one list that share
    a bucket are indistinguishable to the model on duration. Lists average under
    six items, so that collision is common. This measures duration against the
    slate the item was actually shown in, which resolves inside a bucket and
    lets the cross express 'shortest thing on this screen' rather than 'short in
    absolute terms'. It reads duration only, which is known before the
    impression, and no label.
    """
    users = np.asarray(frame.keys("user_id"), dtype=np.int64)
    dates = np.asarray(frame.date)
    _, dcode = np.unique(dates, return_inverse=True)
    dcode = dcode.astype(np.int64)
    pair = users * (int(dcode.max()) + 1) + dcode
    _, gidx = np.unique(pair, return_inverse=True)
    d = np.log1p(np.asarray(frame.duration_ms, dtype=np.float64))
    cnt = np.bincount(gidx).astype(np.float64)
    tot = np.bincount(gidx, weights=d)
    mean = tot / np.maximum(cnt, 1.0)
    out = d - mean[gidx]
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


CONFIG = {"features": ["user_prior_rate", "dur_vs_slate_mean"],
          "group_by": "user_id+date"}
