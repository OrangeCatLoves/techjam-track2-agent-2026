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

import numpy as np

from harness import data as hdata

#: Keys returned by ``starter.evaluate.evaluate`` at k=5.
GAUC, NDCG5, PRIMARY = 'GAUC', 'nDCG@5', 'primary'


def _normalise_labels(labels: Sequence[Any]) -> Sequence[Any]:
    """Coerce integral labels to Python ints before scoring. Input hygiene only.

    ``starter.evaluate.ndcg_at_k`` computes ``(2 ** t) - 1`` on whatever type it is
    handed and accumulates in that type, so a float32 label array silently drops
    the whole metric to float32 precision. The organisers' own ``run_fm`` passes
    ``y`` straight from ``encode()``, which is float32, while a caller reading
    labels from the split rows passes Python ints — and the two disagree in the
    seventh significant digit for identical predictions.

    The gap is ~7e-7, far below the 0.002 convergence epsilon, so no decision was
    ever at risk. But two spellings of the same number invite an afternoon of
    chasing a phantom regression, and generated code will call this from
    everywhere. Normalising here makes every call site agree exactly.

    Only integral values are converted, so graded relevance would pass through
    untouched. The label is binary by definition (``long_view``), so this is the
    identity on real data.
    """
    array = np.asarray(labels)
    if array.dtype.kind == 'f' and array.size and np.all(array == np.rint(array)):
        return array.astype(np.int64).tolist()
    return labels


def evaluate(user_ids: Sequence[Any], labels: Sequence[Any],
             scores: Sequence[float], k: int = 5) -> Dict[str, Any]:
    """The official metric, unmodified.

    Returns ``{'GAUC': .., 'nDCG@5': .., 'primary': .., 'users': .., 'rows': ..}``.

    The only thing added is label normalisation; see ``_normalise_labels``. The
    metric itself is the organisers' and is not touched.
    """
    return hdata.starter_evaluate_module().evaluate(
        user_ids, _normalise_labels(labels), scores, k)


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
