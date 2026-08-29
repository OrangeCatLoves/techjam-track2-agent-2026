"""Data access for the whole project. The single door to the dataset.

OWNS
    - resolution of the KuaiRand-Pure data directory (env > .env > configs/base.yaml)
    - the wrapping of ``starter.data.load`` and ``starter.data.encode``
    - **stripping the hidden-test label** so that no downstream code can read it
    - repo-relative path helpers and the parsed run configuration

MUST NEVER
    - modify, copy or shadow anything in ``starter/``; it is organiser code
    - return the ``long_view`` label for a row in the ``test`` split, in any form,
      through any function, under any flag
    - reimplement the loader, the split boundaries or the row ordering; row order
      defines ``row_id`` in a submission and belongs to the organisers

WHY THE STRIP MATTERS
    ``starter.data.load`` returns ``out['test']`` as 170,588 seven-tuples with the
    true label at index 6, and ``starter/baseline.py`` prints test metrics to stdout.
    The hidden test set is not hidden from this process. See CLAUDE.md section 5.
    This module makes reading a test label a physical ``IndexError`` rather than a
    matter of discipline. The complementary control for stdout lives in
    ``harness/guards.py``.
"""
from __future__ import annotations

import functools
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

# --- row tuple layout, as produced by starter/data.py -----------------------
IDX_DATE, IDX_USER, IDX_VIDEO, IDX_AUTHOR, IDX_TAB, IDX_DURATION, IDX_LABEL = range(7)

TRAIN, VALID, TEST = 'train', 'valid', 'test'
SPLITS = (TRAIN, VALID, TEST)
#: Splits whose label this harness is permitted to hand out.
LABELLED_SPLITS = (TRAIN, VALID)

#: Tuple width per split after stripping. A test row is six long, on purpose.
ROW_WIDTH = {TRAIN: 7, VALID: 7, TEST: 6}


class TestLabelAccessError(RuntimeError):
    """Raised when code asks this module for a hidden-test label."""


# --- repo layout -----------------------------------------------------------

def repo_root() -> Path:
    """Absolute path to the repository root (the parent of ``harness/``)."""
    return Path(__file__).resolve().parent.parent


def starter_dir() -> Path:
    """Absolute path to the read-only organiser kit."""
    return repo_root() / 'starter'


@functools.lru_cache(maxsize=1)
def load_config(path: str | os.PathLike | None = None) -> Dict[str, Any]:
    """Parse ``configs/base.yaml`` (or *path*) once and cache it."""
    import yaml
    cfg_path = Path(path) if path is not None else repo_root() / 'configs' / 'base.yaml'
    with open(cfg_path, 'r', encoding='utf-8') as fh:
        return yaml.safe_load(fh)


@functools.lru_cache(maxsize=1)
def _dotenv() -> Dict[str, str]:
    """Minimal ``.env`` reader. No dependency, no export, no logging of values."""
    out: Dict[str, str] = {}
    path = repo_root() / '.env'
    if not path.exists():
        return out
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, _, value = line.partition('=')
        out[key.strip()] = value.strip().strip('"').strip("'")
    return out


def data_dir(explicit: str | os.PathLike | None = None) -> Path:
    """Resolve the KuaiRand-Pure ``data`` directory.

    Precedence: *explicit* argument > ``KUAIRAND_DATA_DIR`` environment variable
    > ``.env`` > ``paths.raw_data_dir`` in ``configs/base.yaml``.
    """
    for candidate in (explicit,
                      os.environ.get('KUAIRAND_DATA_DIR') or None,
                      _dotenv().get('KUAIRAND_DATA_DIR') or None,
                      load_config().get('paths', {}).get('raw_data_dir')):
        if candidate:
            return Path(candidate)
    raise FileNotFoundError(
        'No KuaiRand-Pure data directory configured. Set KUAIRAND_DATA_DIR or '
        'paths.raw_data_dir in configs/base.yaml.')


# --- starter kit import ----------------------------------------------------

@functools.lru_cache(maxsize=1)
def _starter():
    """Import the organiser modules without touching the ``starter/`` tree.

    ``starter/`` deliberately has no ``__init__.py`` and its modules import each
    other by bare name (``from data import load``), so the directory goes on
    ``sys.path`` rather than being turned into a package.
    """
    path = str(starter_dir())
    if path not in sys.path:
        sys.path.insert(0, path)
    import data as starter_data
    import evaluate as starter_evaluate
    import submit as starter_submit
    return starter_data, starter_evaluate, starter_submit


def starter_data_module():
    """The organiser ``data`` module. Read-only use."""
    return _starter()[0]


def starter_evaluate_module():
    """The organiser ``evaluate`` module. The sole definition of the score."""
    return _starter()[1]


def starter_submit_module():
    """The organiser ``submit`` module."""
    return _starter()[2]


# --- the load ---------------------------------------------------------------

_CACHE: Dict[str, Dict[str, List[tuple]]] = {}


def _strip_test_labels(rows: Sequence[tuple]) -> List[tuple]:
    """Return *rows* truncated to six fields. Index 6 ceases to exist."""
    return [r[:IDX_LABEL] for r in rows]


def load(path: str | os.PathLike | None = None, use_cache: bool = True
         ) -> Dict[str, List[tuple]]:
    """Load the official splits with the hidden-test label removed.

    Returns ``{'train': [...7-tuples...], 'valid': [...7-tuples...],
    'test': [...6-tuples...]}`` in the organisers' row order, which is what
    ``row_id`` means in a submission. Never sort or reindex these lists.

    Train and valid rows keep their label at index 6. Test rows are six long, so
    ``row[6]`` raises ``IndexError`` instead of leaking the answer.
    """
    resolved = str(data_dir(path))
    if use_cache and resolved in _CACHE:
        return _CACHE[resolved]

    splits = starter_data_module().load(resolved)
    out = {TRAIN: splits[TRAIN], VALID: splits[VALID],
           TEST: _strip_test_labels(splits[TEST])}
    # Drop the only other reference to the labelled test rows.
    splits.clear()
    del splits

    for name in SPLITS:
        width = ROW_WIDTH[name]
        if out[name] and len(out[name][0]) != width:
            raise RuntimeError(
                f'{name} rows are {len(out[name][0])} wide, expected {width}')

    if use_cache:
        _CACHE[resolved] = out
    return out


def clear_cache() -> None:
    """Forget the in-process split cache. Used by tests."""
    _CACHE.clear()


def row_counts(splits: Dict[str, List[tuple]] | None = None) -> Dict[str, int]:
    """``{'train': n, 'valid': n, 'test': n}`` for a contract check."""
    splits = splits if splits is not None else load()
    return {name: len(splits[name]) for name in SPLITS}


def labels(splits: Dict[str, List[tuple]], split: str) -> List[int]:
    """Labels for a *labelled* split. Asking for ``test`` is an error, by design."""
    if split not in LABELLED_SPLITS:
        raise TestLabelAccessError(
            f'labels for split {split!r} are not available to this process; '
            f'only {LABELLED_SPLITS} carry a label. See CLAUDE.md section 5.')
    return [r[IDX_LABEL] for r in splits[split]]


def user_ids(splits: Dict[str, List[tuple]], split: str) -> List[str]:
    """Grouping key for the metrics. Available for every split."""
    return [r[IDX_USER] for r in splits[split]]


# --- encoding ---------------------------------------------------------------

def encode(splits: Dict[str, List[tuple]] | None = None
           ) -> Tuple[Dict[str, tuple], int]:
    """``starter.data.encode`` over label-stripped splits.

    Returns ``(enc, total_dim)`` where ``enc[split] = (X, y, users)``:

    * train and valid carry a real ``y`` (float32 array of labels);
    * **test carries ``y = None``**. The organiser encoder needs a seventh field
      to read, so a placeholder ``0`` is appended to each test row *inside this
      function only* and the resulting column is discarded before returning.
      ``None`` is deliberate: placeholder zeros would silently pass for labels.
    """
    splits = splits if splits is not None else load()
    padded = dict(splits)
    padded[TEST] = [r + (0,) for r in splits[TEST]]
    enc, total_dim = starter_data_module().encode(padded)
    X_test, _unused_placeholder_labels, users_test = enc[TEST]
    enc[TEST] = (X_test, None, users_test)
    return enc, total_dim
