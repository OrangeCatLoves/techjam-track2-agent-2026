import numpy as np
from harness.features.registry import register_feature


@register_feature("video_hist_rate")
def video_hist_rate(frame, stats):
    """Smoothed prior long_view rate of this video, from strictly earlier dates.

    The video_id embedding norm (0.140) says the ID field is barely learned:
    7.5k sparse IDs over 1.14M rows, most of the ordering signal absorbed by
    tab and dur_bucket. A history rate is the dense version of the same
    information, and it varies across the items in one user's list, so it can
    change within-user ordering.
    """
    r = np.asarray(stats.label_rate("video_id"), dtype=np.float64)
    g = float(stats.global_rate())
    r = np.where(np.isfinite(r), r, g)
    return r


@register_feature("video_hist_exposure")
def video_hist_exposure(frame, stats):
    """How often this video was shown before today, log-compressed.

    Separate field from the rate: it says how much to trust the rate, and it
    carries popularity/recency of its own. Log then quantile-bucketed, so only
    the ordering survives, which is what the density shift between train and
    evaluation demands of any count.
    """
    c = np.asarray(stats.exposure_count("video_id"), dtype=np.float64)
    c = np.where(np.isfinite(c), c, 0.0)
    return np.log1p(np.maximum(c, 0.0))


@register_feature("author_hist_rate")
def author_hist_rate(frame, stats):
    """Smoothed prior long_view rate of the author.

    Backs off the video rate for rarely-seen or new videos: author_id is the
    natural parent key, and its embedding norm is equally small (0.141).
    """
    r = np.asarray(stats.label_rate("author_id"), dtype=np.float64)
    g = float(stats.global_rate())
    r = np.where(np.isfinite(r), r, g)
    return r


# Pointwise logloss on purpose: pairwise and listwise both scored at or below
# the reference FM, so this isolates the feature change against the stronger
# known objective rather than stacking two unresolved variables.
CONFIG = {
    "features": ["video_hist_rate", "video_hist_exposure", "author_hist_rate"],
    "group_by": "user_id+date",
}
