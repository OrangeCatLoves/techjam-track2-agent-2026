import numpy as np
from harness.features.registry import register_feature

# Shrinkage constant: how many prior impressions a video needs before its own
# residual rate is trusted over its author's.
_SHRINK = 50.0


def _arr(x, n):
    """Coerce a stats return value to a length-n float array."""
    a = np.asarray(x, dtype=np.float64)
    if a.ndim == 0:
        a = np.full(n, float(a), dtype=np.float64)
    return np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)


@register_feature("dur_residual_item_quality")
def build_item_quality(frame, stats):
    """Item quality with the duration prior removed.

    long_view is a completion event, so a video's historical long_view rate is
    dominated by how long the video is -- information the model already holds in
    dur_bucket. Subtracting the duration bucket's own rate leaves the part of a
    video's history that duration does not explain. Thin videos back off toward
    their author's residual.
    """
    n = len(frame)
    dur_rate = _arr(stats.label_rate("dur_bucket"), n)
    vid_rate = _arr(stats.label_rate("video_id"), n)
    aut_rate = _arr(stats.label_rate("author_id"), n)
    vid_n = np.maximum(_arr(stats.exposure_count("video_id"), n), 0.0)

    w = vid_n / (vid_n + _SHRINK)
    q = w * (vid_rate - dur_rate) + (1.0 - w) * (aut_rate - dur_rate)
    return q.astype(np.float64)


@register_feature("dur_residual_author_quality")
def build_author_quality(frame, stats):
    """The author-level residual on its own.

    Coarser and far better populated than the per-video statistic, so it carries
    signal for videos whose own history is too thin to trust. Kept as a separate
    field so the FM can cross it with user_id independently.
    """
    n = len(frame)
    dur_rate = _arr(stats.label_rate("dur_bucket"), n)
    aut_rate = _arr(stats.label_rate("author_id"), n)
    return (aut_rate - dur_rate).astype(np.float64)


CONFIG = {
    "features": ["dur_residual_item_quality", "dur_residual_author_quality"],
    "group_by": "user_id+date",
}
