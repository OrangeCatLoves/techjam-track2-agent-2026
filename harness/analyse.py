"""The agent's eyes. Bounded, autonomous inspection of train and validation.

OWNS
    - the ``analyse(spec)`` tool: the agent asks a question, the harness computes
      the answer, and the answer is a measured fact
    - the enumeration of what can be asked, so the agent can discover its own
      instrument rather than being told what to look at
    - the hard refusal to touch the test split, from any question, in any form

MUST NEVER
    - read the ``test`` split. Every entry point checks the split name and raises;
      ``harness/data.py`` has already removed the labels, and this closes the
      remaining route, which is aggregate statistics over test *features*
    - return a number the agent then reinterprets as something else. Every result
      carries the question that produced it
    - be a dashboard. If this file grows a fixed set of findings that we decided
      were interesting, the agent is reading our analysis rather than doing its
      own, and Innovation is 20% of the grade

DESIGN INTENT
    Broad enough that a curious agent can find something we did not anticipate;
    narrow enough that it cannot reach the test set. The queries are primitives --
    rates by bucket, distributions, drift over time, disagreement between two
    score vectors -- not conclusions. The list-size mismatch (train 43.5,
    validation 5.6) is discoverable through ``list_size_profile``; it is not
    hardcoded anywhere the agent reads.

DEVIATION FROM CLAUDE.md 6.1
    The spec there returns ``pd.DataFrame``. This returns an ``AnalysisResult``
    holding plain rows, because the result goes straight into an LLM prompt and a
    JSON log, and a DataFrame needs converting at both ends while the guard
    screening works on plain structures. ``.to_frame()`` is available for anyone
    who wants pandas.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Sequence

import numpy as np

from harness import data as hdata
from harness import evaluate as hevaluate

#: Splits any question may be asked about. Note what is missing.
ANALYSABLE_SPLITS = ('train', 'valid')

#: Row fields a question may group or bucket by. Same-impression outcomes are not
#: here because the loader never reads them; this is the positive list.
COLUMNS = ('date', 'user_id', 'video_id', 'author_id', 'tab', 'duration_ms')

#: Derived quantities, computed from the permitted columns.
DERIVED = ('user_impressions', 'video_impressions', 'duration_bucket')


class AnalysisError(ValueError):
    """A malformed or forbidden question."""


@dataclass
class AnalysisResult:
    """One answer, carrying the question that produced it."""
    kind: str
    split: str
    question: Dict[str, Any]
    rows: List[Dict[str, Any]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {'kind': self.kind, 'split': self.split, 'question': self.question,
                'rows': self.rows, 'notes': self.notes}

    def to_frame(self):
        """pandas view, for anyone who wants one. Imported lazily."""
        import pandas as pd
        return pd.DataFrame(self.rows)

    def to_markdown(self, max_rows: int = 25) -> str:
        """Compact table, for the prompt and for the human log."""
        if not self.rows:
            return f'_{self.kind} on {self.split}: no rows_'
        columns = list(self.rows[0])
        head = '| ' + ' | '.join(columns) + ' |'
        rule = '|' + '|'.join('---' for _ in columns) + '|'
        body = []
        for row in self.rows[:max_rows]:
            cells = []
            for column in columns:
                value = row[column]
                cells.append(f'{value:.4f}' if isinstance(value, float) else str(value))
            body.append('| ' + ' | '.join(cells) + ' |')
        extra = ([f'_{len(self.rows) - max_rows} more rows._']
                 if len(self.rows) > max_rows else [])
        return '\n'.join([head, rule, *body, *extra, *self.notes])


# --------------------------------------------------------------------------
# guards and helpers
# --------------------------------------------------------------------------

def _check_split(split: str) -> str:
    if split not in ANALYSABLE_SPLITS:
        raise hdata.TestLabelAccessError(
            f'analysis of split {split!r} is refused. Questions may be asked of '
            f'{ANALYSABLE_SPLITS} only; aggregate statistics over the hidden test '
            f'set are still information about the hidden test set. '
            f'See CLAUDE.md section 6.1.')
    return split


def _column(rows: Sequence[tuple], name: str) -> List[Any]:
    index = {'date': hdata.IDX_DATE, 'user_id': hdata.IDX_USER,
             'video_id': hdata.IDX_VIDEO, 'author_id': hdata.IDX_AUTHOR,
             'tab': hdata.IDX_TAB, 'duration_ms': hdata.IDX_DURATION}.get(name)
    if index is not None:
        return [r[index] for r in rows]
    if name == 'user_impressions':
        counts = Counter(r[hdata.IDX_USER] for r in rows)
        return [counts[r[hdata.IDX_USER]] for r in rows]
    if name == 'video_impressions':
        counts = Counter(r[hdata.IDX_VIDEO] for r in rows)
        return [counts[r[hdata.IDX_VIDEO]] for r in rows]
    if name == 'duration_bucket':
        durations = np.asarray([r[hdata.IDX_DURATION] for r in rows], dtype=float)
        edges = np.quantile(durations, np.linspace(0, 1, 11)[1:-1])
        return [int(b) for b in np.searchsorted(edges, durations)]
    raise AnalysisError(
        f'unknown column {name!r}. Available: {COLUMNS + DERIVED}')


def _bucketise(values: Sequence[Any], bins: int) -> List[Any]:
    """Quantile buckets for numeric columns; identity for categorical ones."""
    if not values or not isinstance(values[0], (int, float, np.integer, np.floating)):
        return list(values)
    array = np.asarray(values, dtype=float)
    if len(np.unique(array)) <= bins:
        return [int(v) if float(v).is_integer() else float(v) for v in array]
    edges = np.quantile(array, np.linspace(0, 1, bins + 1)[1:-1])
    return [int(b) for b in np.searchsorted(edges, array)]


# --------------------------------------------------------------------------
# the questions
# --------------------------------------------------------------------------

def _rate_by_bucket(splits, split, *, column: str, bins: int = 10,
                    min_rows: int = 50, **_) -> AnalysisResult:
    """Label rate grouped by a binned column. The workhorse question."""
    rows = splits[split]
    labels = hdata.labels(splits, split)
    buckets = _bucketise(_column(rows, column), bins)
    positives, totals = defaultdict(int), defaultdict(int)
    for bucket, label in zip(buckets, labels):
        totals[bucket] += 1
        positives[bucket] += label
    out = [{'bucket': b, 'rows': totals[b],
            'long_view_rate': positives[b] / totals[b],
            'positives': positives[b]}
           for b in sorted(totals, key=lambda x: (str(type(x)), x))
           if totals[b] >= min_rows]
    overall = sum(labels) / max(1, len(labels))
    return AnalysisResult('rate_by_bucket', split,
                          {'column': column, 'bins': bins, 'min_rows': min_rows},
                          out, [f'_Overall long_view rate on {split}: {overall:.4f}._'])


def _distribution(splits, split, *, column: str, bins: int = 12, **_) -> AnalysisResult:
    """How a column is distributed. Shape before rates."""
    values = _column(splits[split], column)
    buckets = _bucketise(values, bins)
    counts = Counter(buckets)
    total = sum(counts.values())
    out = [{'bucket': b, 'rows': n, 'share': n / total}
           for b, n in sorted(counts.items(), key=lambda kv: (str(type(kv[0])), kv[0]))]
    notes = [f'_{len(counts)} distinct buckets over {total:,} rows._']
    if isinstance(values[0], (int, float, np.integer, np.floating)):
        array = np.asarray(values, dtype=float)
        notes.append(f'_min {array.min():.1f} · median {np.median(array):.1f} · '
                     f'max {array.max():.1f}._')
    return AnalysisResult('distribution', split, {'column': column, 'bins': bins},
                          out[:200], notes)


def _list_size_profile(splits, split, **_) -> AnalysisResult:
    """Impressions per user. The question behind list construction.

    Both scored metrics rank within one user's list, and a training list built by
    a different rule than the evaluation list is a mismatch the agent has to
    choose how to handle. This measures it rather than asserting it.
    """
    rows = splits[split]
    per_user = Counter(r[hdata.IDX_USER] for r in rows)
    per_user_date = Counter((r[hdata.IDX_USER], r[hdata.IDX_DATE]) for r in rows)
    sizes = np.asarray(list(per_user.values()), dtype=float)
    sizes_date = np.asarray(list(per_user_date.values()), dtype=float)
    out = [
        {'grouping': 'user_id', 'lists': len(per_user),
         'mean_size': float(sizes.mean()), 'median_size': float(np.median(sizes)),
         'p10': float(np.quantile(sizes, 0.1)), 'p90': float(np.quantile(sizes, 0.9)),
         'max_size': float(sizes.max())},
        {'grouping': 'user_id+date', 'lists': len(per_user_date),
         'mean_size': float(sizes_date.mean()),
         'median_size': float(np.median(sizes_date)),
         'p10': float(np.quantile(sizes_date, 0.1)),
         'p90': float(np.quantile(sizes_date, 0.9)),
         'max_size': float(sizes_date.max())},
    ]
    return AnalysisResult('list_size_profile', split, {}, out)


def _temporal_drift(splits, split, *, statistic: str = 'long_view_rate',
                    **_) -> AnalysisResult:
    """A statistic per date. Train and evaluation windows differ in density."""
    rows = splits[split]
    labels = hdata.labels(splits, split)
    by_date: Dict[int, List[int]] = defaultdict(list)
    users_by_date: Dict[int, set] = defaultdict(set)
    durations: Dict[int, List[float]] = defaultdict(list)
    for row, label in zip(rows, labels):
        date = row[hdata.IDX_DATE]
        by_date[date].append(label)
        users_by_date[date].add(row[hdata.IDX_USER])
        durations[date].append(row[hdata.IDX_DURATION])
    out = []
    for date in sorted(by_date):
        labs = by_date[date]
        out.append({'date': date, 'rows': len(labs),
                    'long_view_rate': sum(labs) / len(labs),
                    'users': len(users_by_date[date]),
                    'rows_per_user': len(labs) / len(users_by_date[date]),
                    'median_duration_ms': float(np.median(durations[date]))})
    return AnalysisResult('temporal_drift', split, {'statistic': statistic}, out)


def _cold_key_rate(splits, split, **_) -> AnalysisResult:
    """Rows whose user or video was never seen in training.

    A cold key falls into the encoder's UNK slot, so its embedding is shared with
    every other unseen value of that field. How much of the evaluation set that
    covers bounds what any id-based model can do.
    """
    train_users = {r[hdata.IDX_USER] for r in splits['train']}
    train_videos = {r[hdata.IDX_VIDEO] for r in splits['train']}
    train_authors = {r[hdata.IDX_AUTHOR] for r in splits['train']}
    rows = splits[split]
    total = max(1, len(rows))
    cold_user = sum(1 for r in rows if r[hdata.IDX_USER] not in train_users)
    cold_video = sum(1 for r in rows if r[hdata.IDX_VIDEO] not in train_videos)
    cold_author = sum(1 for r in rows if r[hdata.IDX_AUTHOR] not in train_authors)
    cold_either = sum(1 for r in rows
                      if r[hdata.IDX_USER] not in train_users
                      or r[hdata.IDX_VIDEO] not in train_videos)
    out = [{'field': 'user_id', 'cold_rows': cold_user, 'rate': cold_user / total},
           {'field': 'video_id', 'cold_rows': cold_video, 'rate': cold_video / total},
           {'field': 'author_id', 'cold_rows': cold_author, 'rate': cold_author / total},
           {'field': 'user_or_video', 'cold_rows': cold_either,
            'rate': cold_either / total}]
    return AnalysisResult('cold_key_rate', split, {}, out,
                          [f'_Measured against the {len(splits["train"]):,}-row '
                           f'training split._'])


def _user_composition(splits, split, **_) -> AnalysisResult:
    """How users divide into all-negative, all-positive and discriminative.

    This bounds what any model can do. A user with no positives scores nDCG 0
    whatever is predicted; a user with no negatives scores 1. **GAUC is computed
    over the discriminative users alone**, so the size of that group decides how
    much of the metric is even reachable, and training weight spent on the other
    two groups moves nDCG only.

    The composition differs sharply between splits, which is why this is a
    question rather than a constant.
    """
    labels = hdata.labels(splits, split)
    users = hdata.user_ids(splits, split)
    by_user: Dict[Any, List[int]] = defaultdict(list)
    for user, label in zip(users, labels):
        by_user[user].append(label)

    all_negative = all_positive = discriminative = 0
    for values in by_user.values():
        positives = sum(values)
        if positives == 0:
            all_negative += 1
        elif positives == len(values):
            all_positive += 1
        else:
            discriminative += 1

    total = max(1, len(by_user))
    out = [
        {'group': 'all_negative', 'users': all_negative,
         'share': all_negative / total, 'nDCG@5': 'always 0.0',
         'counts_toward_gauc': False},
        {'group': 'all_positive', 'users': all_positive,
         'share': all_positive / total, 'nDCG@5': 'always 1.0',
         'counts_toward_gauc': False},
        {'group': 'discriminative', 'users': discriminative,
         'share': discriminative / total, 'nDCG@5': 'model-dependent',
         'counts_toward_gauc': True},
    ]
    return AnalysisResult(
        'user_composition', split, {}, out,
        [f'_{total:,} users · overall long_view rate '
         f'{sum(labels) / max(1, len(labels)):.4f}._'])


def _segment_metrics(splits, split, *, scores: Sequence[float],
                     column: str = 'user_impressions', bins: int = 5,
                     min_users: int = 30, **_) -> AnalysisResult:
    """The official metric, broken down by a segment.

    Which users a model is bad at is a different question from how good it is.
    Segments are formed per user, because both metrics are computed per user.
    """
    rows = splits[split]
    if len(scores) != len(rows):
        raise AnalysisError(f'{len(scores)} scores for {len(rows)} rows')
    labels = hdata.labels(splits, split)
    users = hdata.user_ids(splits, split)
    buckets = _bucketise(_column(rows, column), bins)

    # One segment per user: the bucket of that user's first row.
    user_segment: Dict[Any, Any] = {}
    for user, bucket in zip(users, buckets):
        user_segment.setdefault(user, bucket)

    grouped: Dict[Any, List[int]] = defaultdict(list)
    for i, user in enumerate(users):
        grouped[user_segment[user]].append(i)

    out = []
    for segment in sorted(grouped, key=lambda x: (str(type(x)), x)):
        idx = grouped[segment]
        segment_users = {users[i] for i in idx}
        if len(segment_users) < min_users:
            continue
        result = hevaluate.evaluate([users[i] for i in idx],
                                    [labels[i] for i in idx],
                                    [scores[i] for i in idx])
        out.append({'segment': segment, 'users': result['users'],
                    'rows': result['rows'], 'gauc': result['GAUC'],
                    'ndcg5': result['nDCG@5'], 'primary': result['primary']})
    return AnalysisResult('segment_metrics', split,
                          {'column': column, 'bins': bins, 'min_users': min_users},
                          out)


def _score_tie_rate(splits, split, *, scores: Sequence[float], **_) -> AnalysisResult:
    """How often a model gives two of a user's items the same score.

    Ties are handled correctly by the evaluator, so they are safe -- but a tie is
    a decision the model declined to make, and a high rate means capacity is going
    unused.
    """
    rows = splits[split]
    if len(scores) != len(rows):
        raise AnalysisError(f'{len(scores)} scores for {len(rows)} rows')
    by_user: Dict[Any, List[float]] = defaultdict(list)
    for row, score in zip(rows, scores):
        by_user[row[hdata.IDX_USER]].append(float(score))
    tied_pairs = total_pairs = users_with_ties = 0
    for values in by_user.values():
        n = len(values)
        if n < 2:
            continue
        total_pairs += n * (n - 1) // 2
        counts = Counter(values)
        ties = sum(c * (c - 1) // 2 for c in counts.values() if c > 1)
        tied_pairs += ties
        users_with_ties += 1 if ties else 0
    out = [{'within_user_pairs': total_pairs, 'tied_pairs': tied_pairs,
            'tie_rate': tied_pairs / total_pairs if total_pairs else 0.0,
            'users_with_any_tie': users_with_ties, 'users': len(by_user)}]
    return AnalysisResult('score_tie_rate', split, {}, out)


def _model_disagreement(splits, split, *, scores: Sequence[float],
                        other_scores: Sequence[float], top_k: int = 1,
                        **_) -> AnalysisResult:
    """Where two score vectors rank a user's list differently.

    The question behind ensembling: two models that agree everywhere cannot help
    each other, and blending is only worth an iteration if they disagree on users
    that matter.
    """
    rows = splits[split]
    if not (len(scores) == len(other_scores) == len(rows)):
        raise AnalysisError('both score vectors must match the split length')
    labels = hdata.labels(splits, split)
    by_user: Dict[Any, List[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        by_user[row[hdata.IDX_USER]].append(i)

    disagreed = both_right = neither = only_a = only_b = 0
    for idx in by_user.values():
        if len(idx) < 2:
            continue
        top_a = max(idx, key=lambda i: scores[i])
        top_b = max(idx, key=lambda i: other_scores[i])
        if top_a == top_b:
            continue
        disagreed += 1
        a_hit, b_hit = labels[top_a] == 1, labels[top_b] == 1
        both_right += a_hit and b_hit
        neither += (not a_hit) and (not b_hit)
        only_a += a_hit and not b_hit
        only_b += b_hit and not a_hit

    comparable = sum(1 for idx in by_user.values() if len(idx) >= 2)
    out = [{'users_compared': comparable, 'disagree_on_top1': disagreed,
            'disagreement_rate': disagreed / comparable if comparable else 0.0,
            'a_right_b_wrong': only_a, 'b_right_a_wrong': only_b,
            'both_right': both_right, 'neither_right': neither}]
    return AnalysisResult('model_disagreement', split, {'top_k': top_k}, out,
                          ['_A high disagreement rate with balanced win counts is '
                           'the case where blending can help._'])


#: The whole instrument. The agent reads this to discover what it can ask.
QUERIES: Dict[str, Callable[..., AnalysisResult]] = {
    'rate_by_bucket': _rate_by_bucket,
    'distribution': _distribution,
    'list_size_profile': _list_size_profile,
    'user_composition': _user_composition,
    'segment_metrics': _segment_metrics,
    'temporal_drift': _temporal_drift,
    'model_disagreement': _model_disagreement,
    'score_tie_rate': _score_tie_rate,
    'cold_key_rate': _cold_key_rate,
}

#: One line per question, for the agent's prompt. Deliberately describes what each
#: question *measures*, never what the answer is expected to be.
CAPABILITIES: Dict[str, str] = {
    'rate_by_bucket': 'long_view rate grouped by a binned column '
                      f'(column: one of {COLUMNS + DERIVED}; bins; min_rows)',
    'distribution': 'how a column is distributed (column; bins)',
    'list_size_profile': 'impressions per user, by user_id and by user_id+date',
    'user_composition': 'how users divide into all-negative, all-positive and '
                        'discriminative, and which of those GAUC is computed over',
    'segment_metrics': 'GAUC/nDCG@5/primary broken down by a segment '
                       '(needs scores; column; bins; min_users)',
    'temporal_drift': 'rows, users, rate and median duration per date',
    'model_disagreement': 'where two score vectors pick a different top item '
                          '(needs scores and other_scores)',
    'score_tie_rate': 'fraction of within-user score pairs that are tied '
                      '(needs scores)',
    'cold_key_rate': 'share of rows whose user, video or author never appears '
                     'in training',
}


def capabilities() -> Dict[str, Any]:
    """What can be asked, and of what. The agent's self-discovery entry point."""
    return {'kinds': CAPABILITIES,
            'splits': list(ANALYSABLE_SPLITS),
            'columns': list(COLUMNS + DERIVED),
            'refused': ['test'],
            'note': 'Every answer is computed by the harness from train or '
                    'validation rows. The hidden test split cannot be asked '
                    'about, including in aggregate.'}


def analyse(kind: str, split: str = 'valid', *,
            splits: Dict[str, list] | None = None, **question: Any) -> AnalysisResult:
    """Ask one question of the data.

    ``analyse('rate_by_bucket', 'valid', column='duration_ms')``

    Raises ``AnalysisError`` for an unknown question and ``TestLabelAccessError``
    for any attempt to ask about the test split.
    """
    if kind not in QUERIES:
        raise AnalysisError(
            f'unknown analysis {kind!r}. Available: {sorted(QUERIES)}')
    _check_split(split)
    splits = splits if splits is not None else hdata.load()
    return QUERIES[kind](splits, split, **question)
