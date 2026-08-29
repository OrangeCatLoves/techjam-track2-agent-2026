"""Submission wrapper. Delegates to ``starter/submit.py`` and adds two refusals.

OWNS
    - writing a submission CSV in the organisers' format
    - checking a submission against the evaluation split, using their validator
    - scoring a submission, on validation only

MUST NEVER
    - reimplement ``read_submission`` or ``write_submission``. The organisers'
      validator is the definition of a well-formed submission, and patching
      around a rejection is forbidden (CLAUDE.md section 6.3)
    - score, or allow anyone to score, the ``test`` split. ``starter/submit.py``
      exposes ``--score --split test``, which would print a hidden-test metric.
      That path is closed here
    - sort, reindex or deduplicate the evaluation rows. ``row_id`` is the
      positional index into ``harness.data.load()[split]`` and ``(user_id,
      video_id)`` is not a key: 3.06% of test rows are repeats

Format, for reference: header ``row_id,user_id,video_id,score``; ``row_id``
strictly ``0,1,2,...``; one row per evaluation row, in order; score any finite
float, used only for its relative order.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

from harness import data as hdata
from harness import evaluate as hevaluate

HEADER = ['row_id', 'user_id', 'video_id', 'score']


def write(path: str | Path, rows: Sequence[tuple], scores: Sequence[float]) -> Path:
    """Write a submission for *rows* (an evaluation split, in its own order)."""
    if len(scores) != len(rows):
        raise ValueError(f'{len(scores)} scores for {len(rows)} rows')
    hdata.starter_submit_module().write_submission(str(path), rows, scores)
    return Path(path)


def write_split(path: str | Path, splits: Dict[str, list], split: str,
                scores: Sequence[float]) -> Path:
    """Write a submission for a named split of *splits*."""
    return write(path, splits[split], scores)


def check(path: str | Path, split: str = 'test',
          splits: Dict[str, list] | None = None) -> List[float]:
    """Validate format and alignment with the organisers' own reader.

    Returns the parsed scores. Raises ``ValueError`` with their message on any
    header error, row-count mismatch, ``row_id`` gap, misalignment, or NaN/Inf
    score. Never patch around a failure here; fix the producer.
    """
    splits = splits if splits is not None else hdata.load()
    if split not in hdata.SPLITS:
        raise ValueError(f'unknown split {split!r}')
    return hdata.starter_submit_module().read_submission(str(path), splits[split])


def score(path: str | Path, split: str = 'valid',
          splits: Dict[str, list] | None = None) -> Dict[str, Any]:
    """Check *path* and score it. Validation only, by design.

    This is the independent check of our own scoring path: a submission written
    from model scores, re-read from disk, and scored through the organisers'
    validator and metric.
    """
    if split not in hdata.LABELLED_SPLITS:
        raise hdata.TestLabelAccessError(
            f'refusing to score split {split!r}. starter/submit.py would happily '
            f'print a hidden-test metric here; this harness will not. '
            f'See CLAUDE.md section 5.')
    splits = splits if splits is not None else hdata.load()
    scores = check(path, split, splits)
    return hevaluate.evaluate_split(splits, split, scores)


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description='Harness wrapper over starter/submit.py. Scoring is '
                    'validation-only; --score --split test is refused.')
    parser.add_argument('path')
    parser.add_argument('--data_dir', default=None)
    parser.add_argument('--split', default='test', choices=list(hdata.SPLITS))
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--check', action='store_true')
    group.add_argument('--score', action='store_true')
    args = parser.parse_args(argv)

    splits = hdata.load(args.data_dir)
    if args.check:
        scores = check(args.path, args.split, splits)
        print(f'ok: format and alignment pass, {len(scores):,d} rows, '
              f'split={args.split}')
    else:
        result = score(args.path, args.split, splits)
        print(hevaluate.format_result(result, args.split))
    return 0


if __name__ == '__main__':
    sys.exit(_main())
