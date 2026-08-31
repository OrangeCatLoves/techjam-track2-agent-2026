"""The feature registry, and the one reference feature that ships.

THE CONTRACT (CLAUDE.md section 11.1, interface 3)

    @register_feature(name="...")
    def build(frame, stats) -> np.ndarray

    frame   a Frame. Row attributes known before the impression: key codes,
            duration, date. No labels, no outcomes.
    stats   a CausalStats for the same split, already windowed. The only route
            to anything label-derived, which is why the window holds.

    returns float array of length len(frame), one value per row, same order.
            Pure and deterministic: same frame and stats, same output.

    The harness quantile-buckets the returned values into a new FM field. A
    feature therefore only has to be *monotone-meaningful*, not scaled.

WHY ONLY ONE REFERENCE FEATURE SHIPS, AND WHY IT IS A WEAK ONE
    ``harness/losses.py`` ships pointwise logloss and nothing cleverer; the agent
    wrote all nineteen ranking objectives itself. The same line is drawn here.

    ``video_exposure_count`` is a popularity count. It reads no label at all, so
    it is not a target encoding and it is not the idea under test. It exists to
    show the shape of a feature and to give the leakage tests something concrete
    to run against.

    The interesting move -- keying a smoothed historical long-view rate on a
    video, an author, or a user-by-tab cross -- is available through
    ``stats.label_rate`` and is deliberately left unwritten. CLAUDE.md section
    6.4 requires the agent to choose and write its own experiments, and section
    12.2 M3 scores the run on the agent's reasoning. Shipping the target
    encoding would hand over the hypothesis and hollow out the result.
"""
from __future__ import annotations

from typing import Callable, Dict, Tuple

import numpy as np

from harness.features.base import CausalStats, FeatureError, Frame

#: Signature of every feature. See the module docstring.
FeatureFn = Callable[[Frame, CausalStats], np.ndarray]

_REGISTRY: Dict[str, FeatureFn] = {}


def register_feature(name: str) -> Callable[[FeatureFn], FeatureFn]:
    """Register a feature builder under *name*.

    Generated features call this from ``harness/features/gen/``.
    """
    def decorate(fn: FeatureFn) -> FeatureFn:
        if name in _REGISTRY:
            raise FeatureError(f'feature {name!r} is already registered')
        _REGISTRY[name] = fn
        return fn
    return decorate


def get_feature(name_or_fn: str | FeatureFn) -> FeatureFn:
    """Resolve a feature by name, or pass a callable straight through."""
    if callable(name_or_fn):
        return name_or_fn
    try:
        return _REGISTRY[name_or_fn]
    except KeyError:
        raise FeatureError(
            f'unknown feature {name_or_fn!r}; registered: {sorted(_REGISTRY)}') from None


def registered() -> Tuple[str, ...]:
    """Names of every registered feature."""
    return tuple(sorted(_REGISTRY))


def check_feature(fn: FeatureFn, frame: Frame, stats: CausalStats) -> np.ndarray:
    """Run *fn* and hold it to the contract before it reaches a training run.

    Catches the three failures that otherwise produce a plausible number: wrong
    length, non-finite values, and non-determinism. A feature that fails here has
    not cost a training run.
    """
    out = np.asarray(fn(frame, stats), dtype=np.float64)
    if out.shape != (len(frame),):
        raise FeatureError(
            f'feature returned shape {out.shape}, expected ({len(frame)},)')
    if not np.isfinite(out).all():
        raise FeatureError('feature returned non-finite values')
    again = np.asarray(fn(frame, stats), dtype=np.float64)
    if not np.array_equal(out, again):
        raise FeatureError('feature is not deterministic on identical input')
    return out.astype(np.float32)


# --------------------------------------------------------------------------
# the reference feature -- deliberately weak, see the module docstring
# --------------------------------------------------------------------------

@register_feature('video_exposure_count')
def video_exposure_count(frame: Frame, stats: CausalStats) -> np.ndarray:
    """How many times this video was shown before, log-scaled.

    Reads no label, so it is popularity rather than a target encoding. Exposure
    counts are heavy-tailed, hence ``log1p``; the harness buckets by quantile
    afterwards, so the scale only has to be monotone.
    """
    return np.log1p(stats.exposure_count('video_id'))


# --------------------------------------------------------------------------
# the pipeline entry point
# --------------------------------------------------------------------------

def encode_with_features(splits, features=(), *, prior=None, n_buckets=None):
    """``harness.data.encode`` widened by the named features.

    Each feature becomes one extra FM field, quantile-bucketed on train values.
    With no features this is exactly ``hdata.encode``, so the augmented path and
    the plain path cannot diverge when nothing is requested.

    Returns ``(enc, dim, names)``. *names* is the resolved feature list, recorded
    on the result so the submission path can rebuild an identical design matrix
    -- scoring a features model on an un-augmented matrix would not crash, it
    would silently rank by the wrong thing.
    """
    from harness import data as hdata
    from harness.features import base as fbase

    names = [features] if isinstance(features, str) else list(features)
    enc, dim = hdata.encode(splits)
    if not names:
        return enc, dim, []

    frames, stats = fbase.build_stats(
        splits, prior=fbase.DEFAULT_PRIOR if prior is None else prior)
    columns = {}
    for name in names:
        fn = get_feature(name)
        values = {s: check_feature(fn, frames[s], stats[s]) for s in splits}
        columns[name] = fbase.bucketise(
            values, fbase.DEFAULT_BUCKETS if n_buckets is None else n_buckets)
    enc, dim = fbase.augment(enc, dim, columns)
    return enc, dim, names
