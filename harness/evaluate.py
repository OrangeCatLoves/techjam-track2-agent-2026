"""Scoring wrapper. Delegates to ``starter/evaluate.py`` and adds nothing.

OWNS
    - the single call site for the official metric
    - the refusal to score the hidden test split

MUST NEVER
    - reimplement GAUC, nDCG@5 or the primary mean. ``starter/evaluate.py`` is the
      sole definition of the score; a second implementation is a second answer
    - accept, compute or return a metric for the ``test`` split. ``harness/data.py``
      does not hand out test labels, and this module refuses to be handed them

The organiser conventions, all confirmed in their source and re-stated here only
so nobody re-derives them: GAUC counts users with ``0 < npos < len(labels)`` and
weights by positive count; nDCG@5 includes zero-positive users scored 0.0; gain is
``2**rel - 1``; the grouping key is ``user_id``; primary is the plain mean.
"""
from __future__ import annotations

from typing import Any, Dict, Sequence

from harness import data as hdata

#: Keys returned by ``starter.evaluate.evaluate`` at k=5.
GAUC, NDCG5, PRIMARY = 'GAUC', 'nDCG@5', 'primary'


def evaluate(user_ids: Sequence[Any], labels: Sequence[Any],
             scores: Sequence[float], k: int = 5) -> Dict[str, Any]:
    """The official metric, unmodified.

    Returns ``{'GAUC': .., 'nDCG@5': .., 'primary': .., 'users': .., 'rows': ..}``.
    """
    return hdata.starter_evaluate_module().evaluate(user_ids, labels, scores, k)


def evaluate_split(splits: Dict[str, list], split: str,
                   scores: Sequence[float], k: int = 5) -> Dict[str, Any]:
    """Score *scores* against a named split. ``test`` is refused.

    ``scores`` must be in the split's own row order, which is the order
    ``harness.data.load`` returns and the order ``row_id`` refers to.
    """
    if split not in hdata.LABELLED_SPLITS:
        raise hdata.TestLabelAccessError(
            f'refusing to evaluate split {split!r}. The hidden test set is scored '
            f'once by the organisers; this process must never compute a test '
            f'metric. See CLAUDE.md section 5.')
    rows = splits[split]
    if len(scores) != len(rows):
        raise ValueError(f'{len(scores)} scores for {len(rows)} rows in {split!r}')
    return evaluate(hdata.user_ids(splits, split), hdata.labels(splits, split),
                    scores, k)


def primary(result: Dict[str, Any]) -> float:
    """Pull the primary score out of an ``evaluate`` result."""
    return float(result[PRIMARY])


def format_result(result: Dict[str, Any], prefix: str = 'valid') -> str:
    """One human-readable line, in the kit's own format."""
    return (f'{prefix:5s}  GAUC {result[GAUC]:.4f} | nDCG@5 {result[NDCG5]:.4f} '
            f'| primary {result[PRIMARY]:.4f}')
