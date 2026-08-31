"""Causal feature construction. The fifth pipeline stage, and the only one the
agent has never been able to target.

OWNS
    - the causal window, which is the whole safety argument of this module
    - the frame and stats objects a generated feature is allowed to see
    - bucketisation, which turns a real number into a field the FM can consume

MUST NEVER
    - hand a feature function a row's own label, or any statistic computed from a
      window that includes the row's own date
    - read a label from the hidden test split. ``harness/data.py`` does not hand
      one out, so this is enforced upstream as well as here

WHY THE WINDOW IS THE DESIGN
    Any historical statistic -- a video's past long-view rate, a user's affinity
    for an author -- is a target encoding. Compute it over all of training and
    apply it to training rows and every row's own label leaks into its own
    feature. Validation inflates, test does not move, and the run is wasted.

    The usual defence is to ask the feature's author to be careful. That does not
    survive contact with generated code. So the window is not the author's
    responsibility here: ``CausalStats`` is constructed per split with the
    correct window already applied, and it is the *only* route to a label-derived
    quantity. A generated feature cannot widen it, because it is never given the
    labels to widen it with.

    CLAUDE.md section 7.4:

        train row on date d   ->  statistics from train dates strictly < d
        validation row        ->  statistics from train dates only
        test row              ->  statistics from train dates only

    All three are implemented below by the same expanding accumulator, which is
    why the train and evaluation paths cannot drift apart.

WHY FEATURES BECOME BUCKETS
    The FM consumes integer field codes, not real numbers. A feature is therefore
    quantile-bucketed into a new categorical field, exactly as the organisers'
    own ``dur_bucket`` turns ``duration_ms`` into ten buckets. Edges come from
    train values only.

    This is also the point of the exercise. ``user_id`` spends 17 parameters on
    roughly 44 observations; a 20-bucket encoding spends 17 on roughly 57,000.
    The diagnostics say the ID fields are starved, not useless, and a bucketed
    statistic is the same information at a density the model can actually fit.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from harness import data as hdata

#: Fields a feature may key a statistic on. These are the row's own attributes,
#: all of them known before the impression happens.
KEYABLE = ('user_id', 'video_id', 'author_id', 'tab', 'dur_bucket')

#: Default smoothing weight, matching the organisers' popularity baseline. A key
#: seen twice should not be trusted the way a key seen two thousand times is.
DEFAULT_PRIOR = 20.0

#: Buckets per generated feature. Chosen for density: 20 buckets over 1.14M train
#: rows is ~57,000 rows each, against ~44 per user id.
DEFAULT_BUCKETS = 20

_FIELD_INDEX = {'user_id': hdata.IDX_USER,
                'video_id': hdata.IDX_VIDEO,
                'author_id': hdata.IDX_AUTHOR,
                'tab': hdata.IDX_TAB}


class FeatureError(ValueError):
    """A feature violated the interface, or asked for something out of window."""


def _dur_bucket_codes(splits: Dict[str, list], n: int = 10) -> Dict[str, np.ndarray]:
    """``duration_ms`` quantile-bucketed on **train** edges, per split.

    Mirrors ``starter.data.encode``: edges from train, ``searchsorted`` for every
    split, so an unseen validation duration lands in a train bucket rather than
    inventing one.
    """
    train_dur = np.array([r[hdata.IDX_DURATION] for r in splits['train']])
    edges = np.quantile(train_dur, np.linspace(0, 1, n + 1)[1:-1])
    out = {}
    for name, rows in splits.items():
        dur = np.array([r[hdata.IDX_DURATION] for r in rows])
        out[name] = np.searchsorted(edges, dur).astype(np.int64)
    return out


def _key_codes(splits: Dict[str, list]) -> Dict[str, Dict[str, np.ndarray]]:
    """Integer codes per keyable field per split, on a shared vocabulary.

    The vocabulary is built from train and evaluation rows together. That is safe
    -- it uses identities, never labels -- and it means a video first seen in
    validation gets its own slot instead of colliding with an unrelated one.
    """
    codes: Dict[str, Dict[str, np.ndarray]] = {}
    for field, idx in _FIELD_INDEX.items():
        raw = {name: np.array([r[idx] for r in rows]) for name, rows in splits.items()}
        vocab = {v: i for i, v in enumerate(np.unique(np.concatenate(list(raw.values()))))}
        codes[field] = {name: np.array([vocab[v] for v in arr], dtype=np.int64)
                        for name, arr in raw.items()}
    codes['dur_bucket'] = _dur_bucket_codes(splits)
    return codes


class CausalStats:
    """Label-derived statistics, already windowed for the split they belong to.

    Constructed by :func:`build_stats`. A feature function receives one of these
    and cannot obtain a label any other way, which is what makes the window a
    property of the system rather than of the feature's author.
    """

    def __init__(self, split: str, rate: Dict[str, np.ndarray],
                 count: Dict[str, np.ndarray], global_rate: np.ndarray):
        self.split = split
        self._rate = rate
        self._count = count
        self._global_rate = global_rate

    def label_rate(self, field: str) -> np.ndarray:
        """Smoothed long-view rate of this row's *field* key, from prior dates only.

        For a train row on date ``d`` this is computed from train rows dated
        strictly before ``d``. For a validation or test row it is computed from
        the whole train period. A key with no history falls back to the running
        global rate.
        """
        self._check(field)
        return self._rate[field]

    def exposure_count(self, field: str) -> np.ndarray:
        """How many prior impressions this row's *field* key had, same window.

        Carries no label, so it is a popularity signal rather than a target
        encoding. Useful as a confidence weight on :meth:`label_rate`.
        """
        self._check(field)
        return self._count[field]

    def global_rate(self) -> np.ndarray:
        """The long-view rate over everything in window, per row."""
        return self._global_rate

    def _check(self, field: str) -> None:
        if field not in KEYABLE:
            raise FeatureError(
                f'unknown key field {field!r}; choose from {KEYABLE}')
        if field not in self._rate:
            raise FeatureError(f'no statistics were built for {field!r}')

    def __repr__(self) -> str:
        return f'<CausalStats split={self.split!r} fields={sorted(self._rate)}>'


class Frame:
    """The row attributes a feature may read. Deliberately label-free.

    Everything here is known before the impression happens, so nothing on this
    object can leak an outcome. Watch time, clicks and likes are absent by
    construction -- they are outcomes of the impression and CLAUDE.md section 7.2
    forbids them as inputs.
    """

    def __init__(self, split: str, codes: Dict[str, np.ndarray],
                 duration_ms: np.ndarray, date: np.ndarray):
        self.split = split
        self.n = len(date)
        self._codes = codes
        self.duration_ms = duration_ms
        self.date = date

    def keys(self, field: str) -> np.ndarray:
        """Integer codes for *field*, one per row."""
        if field not in self._codes:
            raise FeatureError(f'unknown field {field!r}; choose from {KEYABLE}')
        return self._codes[field]

    def __len__(self) -> int:
        return self.n

    def __repr__(self) -> str:
        return f'<Frame split={self.split!r} rows={self.n}>'


def _expanding(keys_by_split: Dict[str, np.ndarray], train_dates: np.ndarray,
               train_labels: np.ndarray, n_keys: int, prior: float
               ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray],
                          Dict[str, np.ndarray]]:
    """One expanding-window pass. The single place the window is implemented.

    Walks the train dates in order. Every row of date ``d`` is scored from the
    accumulator *before* date ``d`` is folded in, so a row can never see itself
    or anything else from its own day. Validation and test are scored from the
    accumulator after the whole train period, which is the same object one step
    further on -- so the two paths cannot drift.
    """
    pos = np.zeros(n_keys, dtype=np.float64)
    imp = np.zeros(n_keys, dtype=np.float64)
    tot_pos = 0.0
    tot_imp = 0.0

    def smooth(p, i, g):
        """Smoothed rate, falling back to *g* where there is nothing to smooth.

        With no history and no prior the ratio is 0/0. That is not an edge case
        to shrug at: it is every key on the first train date, and a NaN there
        propagates silently into a design matrix. No history means no evidence,
        so the answer is the global rate.
        """
        denom = i + prior
        return np.where(denom > 0, (p + prior * g) / np.where(denom > 0, denom, 1.0), g)

    tr_keys = keys_by_split['train']
    rate = {s: np.empty(len(k), dtype=np.float32) for s, k in keys_by_split.items()}
    count = {s: np.empty(len(k), dtype=np.float32) for s, k in keys_by_split.items()}
    grate = {s: np.empty(len(k), dtype=np.float32) for s, k in keys_by_split.items()}

    for d in np.unique(train_dates):
        sel = train_dates == d
        k = tr_keys[sel]
        g = np.float32(tot_pos / tot_imp) if tot_imp else np.float32(0.0)
        rate['train'][sel] = smooth(pos[k], imp[k], g)
        count['train'][sel] = imp[k]
        grate['train'][sel] = g
        np.add.at(pos, k, train_labels[sel])
        np.add.at(imp, k, 1.0)
        tot_pos += float(train_labels[sel].sum())
        tot_imp += float(sel.sum())

    g_all = np.float32(tot_pos / tot_imp) if tot_imp else np.float32(0.0)
    for split, k in keys_by_split.items():
        if split == 'train':
            continue
        rate[split] = smooth(pos[k], imp[k], g_all).astype(np.float32)
        count[split] = imp[k].astype(np.float32)
        grate[split] = np.full(len(k), g_all, dtype=np.float32)
    return rate, count, grate


def build_stats(splits: Dict[str, list] | None = None, *,
                prior: float = DEFAULT_PRIOR,
                fields: Sequence[str] = KEYABLE
                ) -> Tuple[Dict[str, Frame], Dict[str, CausalStats]]:
    """Frames and correctly windowed statistics for every split.

    Returns ``({split: Frame}, {split: CausalStats})``. Train labels are the only
    labels read; validation labels are deliberately not used even though this
    process holds them, because a statistic built from validation and applied to
    validation would inflate the one number that decides everything.
    """
    splits = splits if splits is not None else hdata.load()
    for f in fields:
        if f not in KEYABLE:
            raise FeatureError(f'unknown key field {f!r}; choose from {KEYABLE}')

    codes = _key_codes(splits)
    dates = {name: np.array([r[hdata.IDX_DATE] for r in rows])
             for name, rows in splits.items()}
    durations = {name: np.array([r[hdata.IDX_DURATION] for r in rows], dtype=np.float64)
                 for name, rows in splits.items()}
    train_labels = np.array(hdata.labels(splits, 'train'), dtype=np.float64)

    rate: Dict[str, Dict[str, np.ndarray]] = {s: {} for s in splits}
    count: Dict[str, Dict[str, np.ndarray]] = {s: {} for s in splits}
    grate: Dict[str, np.ndarray] = {}
    for field in fields:
        by_split = codes[field]
        n_keys = int(max(a.max() for a in by_split.values())) + 1
        r, c, g = _expanding(by_split, dates['train'], train_labels, n_keys, prior)
        for s in splits:
            rate[s][field] = r[s]
            count[s][field] = c[s]
            grate[s] = g[s]

    frames = {s: Frame(s, {f: codes[f][s] for f in KEYABLE}, durations[s], dates[s])
              for s in splits}
    stats = {s: CausalStats(s, rate[s], count[s], grate[s]) for s in splits}
    return frames, stats


def bucketise(values_by_split: Dict[str, np.ndarray],
              n_buckets: int = DEFAULT_BUCKETS) -> Dict[str, np.ndarray]:
    """Quantile-bucket a real-valued feature into integer codes.

    Edges come from **train values only**, so an evaluation row cannot shift the
    binning. Ties collapse: a feature that is constant over train yields a single
    bucket, which is a correct representation of a feature carrying no
    information rather than an error.
    """
    train = np.asarray(values_by_split['train'], dtype=np.float64)
    edges = np.unique(np.quantile(train, np.linspace(0, 1, n_buckets + 1)[1:-1]))
    return {s: np.searchsorted(edges, np.asarray(v, dtype=np.float64)).astype(np.int32)
            for s, v in values_by_split.items()}


def augment(enc: Dict[str, tuple], dim: int,
            columns: Dict[str, Dict[str, np.ndarray]]) -> Tuple[Dict[str, tuple], int]:
    """Append bucketed feature columns to an encoded design matrix.

    *columns* maps a feature name to ``{split: int codes}``. Each becomes one new
    FM field, offset past everything already allocated. The FM sums over the
    field axis, so widening ``X`` needs no change to the model.

    Returns ``(enc, new_dim)`` with ``enc`` rebuilt rather than mutated.
    """
    out = {s: list(v) for s, v in enc.items()}
    total = dim
    for name in sorted(columns):
        per_split = columns[name]
        width = int(max(a.max() for a in per_split.values())) + 1
        for s, codes in per_split.items():
            X = out[s][0]
            if len(codes) != len(X):
                raise FeatureError(
                    f'feature {name!r} produced {len(codes)} rows for split {s!r}, '
                    f'expected {len(X)}')
            out[s][0] = np.hstack([X, (codes.astype(np.int32) + total)[:, None]])
        total += width
    return {s: tuple(v) for s, v in out.items()}, total
