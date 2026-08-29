"""Integrity guards. The three controls that keep the run defensible.

OWNS
    1. the feature column deny-list (``leakage.deny_columns_as_features``)
    2. the starter-stdout filter, which removes any line carrying a hidden-test
       metric before that text can reach the agent, an LLM prompt or a log
    3. the leak canary, which quarantines any configuration scoring above
       ``leakage.canary_primary_threshold`` on validation

MUST NEVER
    - be weakened, bypassed or made optional by generated code; this module is on
      the protected-path list in ``configs/base.yaml``
    - write a redacted line into any machine-readable log. The raw output of a
      starter script goes to a human-only file under ``runs/raw_starter_output/``
      and nowhere else
    - silently pass. Every guard either returns cleanly or raises

WHY
    ``starter/baseline.py`` and ``starter/ablation_features.py`` print hidden-test
    metrics to stdout on every run. ``harness/data.py`` closes the in-memory route
    to the test label; this module closes the textual one. See CLAUDE.md section 5.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from harness import data as hdata


class LeakageError(RuntimeError):
    """A denied column reached a feature path, or a test metric reached a log."""


class LeakCanaryError(RuntimeError):
    """A validation score implausibly close to the oracle ceiling. Quarantined."""


# --------------------------------------------------------------------------
# 1. column deny-list
# --------------------------------------------------------------------------

def denied_columns() -> frozenset:
    """Columns that may never be model inputs. Same-impression outcomes.

    They remain legitimate as *auxiliary training targets* in a multi-task setup.
    This guard governs inputs only.
    """
    cfg = hdata.load_config().get('leakage', {})
    return frozenset(cfg.get('deny_columns_as_features', ()))


def allowed_log_columns() -> frozenset:
    """Log columns known before the impression, so safe as inputs."""
    cfg = hdata.load_config().get('leakage', {})
    return frozenset(cfg.get('allow_log_columns', ()))


def is_denied(column: str) -> bool:
    """True if *column* is a same-impression outcome."""
    return column in denied_columns()


def assert_columns_allowed(columns: Iterable[str], where: str = 'feature frame') -> None:
    """Raise ``LeakageError`` if any of *columns* is on the deny-list."""
    offending = sorted(c for c in columns if is_denied(c))
    if offending:
        raise LeakageError(
            f'{where} contains same-impression outcome column(s) {offending}. '
            f'These are never permitted as model inputs '
            f'(configs/base.yaml leakage.deny_columns_as_features).')


def assert_frame_clean(frame: Any, where: str = 'feature frame') -> None:
    """Deny-list check for anything exposing ``.columns`` or dict keys."""
    columns = getattr(frame, 'columns', None)
    if columns is None:
        columns = frame.keys() if hasattr(frame, 'keys') else frame
    assert_columns_allowed([str(c) for c in columns], where)


# --------------------------------------------------------------------------
# 2. the starter stdout filter
# --------------------------------------------------------------------------

#: Anything that looks like a scored metric.
_METRIC = r'(?:gauc|auc|ndcg(?:@\d+)?|primary|logloss|mrr|recall(?:@\d+)?|hit(?:rate)?(?:@\d+)?)'
#: Anything that names the held-out split, in either language used by the kit.
_TEST_TOKEN = r'(?:\btest\b|\bhidden\b|测试)'

_METRIC_RE = re.compile(_METRIC, re.IGNORECASE)
_TEST_RE = re.compile(_TEST_TOKEN, re.IGNORECASE)
#: A bare ``test  0.5953``-style line, with no metric name at all.
_BARE_TEST_RE = re.compile(r'^\s*test\b[^A-Za-z]*[-+]?\d*\.\d+', re.IGNORECASE)

REDACTION = '[redacted: line carried a hidden-test metric -- see the raw human-only log]'


#: Separators that are word characters to a regex but word breaks to a reader,
#: so that `test_gauc=0.66` and `test-primary` are caught like `test gauc`.
_SEPARATORS = re.compile(r'[_\-=:/.]+')


def contains_test_metric(line: str) -> bool:
    """True if *line* mentions the test split anywhere near a metric.

    Deliberately over-eager. A line naming both the test split and a metric is
    dropped even if the metric belongs to validation; losing a line of organiser
    stdout costs nothing, and seeing a test score costs the run.
    """
    if _BARE_TEST_RE.match(line):
        return True
    spaced = _SEPARATORS.sub(' ', line)
    return bool(_TEST_RE.search(spaced)) and bool(_METRIC_RE.search(spaced))


def filter_stdout(text: str) -> tuple[str, int]:
    """Redact every test-metric line in *text*.

    Returns ``(clean_text, n_redacted)``. Line count and order are preserved so
    that a human comparing against the raw log can see exactly what was removed.
    """
    lines = text.splitlines()
    redacted = 0
    out: List[str] = []
    for line in lines:
        if contains_test_metric(line):
            out.append(REDACTION)
            redacted += 1
        else:
            out.append(line)
    clean = '\n'.join(out)
    if text.endswith('\n') and clean:
        clean += '\n'
    return clean, redacted


def assert_no_test_metrics(text: str, where: str = 'text') -> None:
    """Raise if *text* still carries a test metric. Use before logging."""
    for n, line in enumerate(text.splitlines(), start=1):
        if contains_test_metric(line):
            raise LeakageError(
                f'{where} line {n} carries a hidden-test metric and must not be '
                f'logged or shown to the agent. See CLAUDE.md section 5.')


def assert_record_clean(record: Any, where: str = 'log record') -> None:
    """Deep check that a JSON-serialisable *record* names no test metric.

    Applied to every machine-readable log line. A dict key such as ``test_gauc``
    or a nested ``{'test': {'primary': ...}}`` both trip it.
    """
    def walk(node: Any, path: str) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                key_s = str(key)
                # Separators are word characters to a regex but word breaks to a
                # reader, so `test_primary` and `test-gauc` must match too.
                key_words = re.sub(r'[_\-.]+', ' ', key_s)
                names_test = bool(_TEST_RE.search(key_words))
                if names_test and isinstance(value, dict):
                    if any(_METRIC_RE.search(re.sub(r'[_\-.]+', ' ', str(k)))
                           for k in value):
                        raise LeakageError(
                            f'{where} at {path}.{key_s} holds test metrics.')
                if names_test and _METRIC_RE.search(key_words):
                    raise LeakageError(
                        f'{where} at {path}.{key_s} is a test metric field.')
                walk(value, f'{path}.{key_s}')
        elif isinstance(node, (list, tuple)):
            for i, value in enumerate(node):
                walk(value, f'{path}[{i}]')
        elif isinstance(node, str):
            if contains_test_metric(node):
                raise LeakageError(f'{where} at {path} contains a test metric line.')

    walk(record, where)


@dataclass
class StarterRun:
    """Result of shelling out to an organiser script, already filtered."""
    returncode: int
    stdout: str                 # safe to show the agent
    stderr: str                 # safe to show the agent
    redacted_lines: int
    raw_log_path: Path | None   # human-only; never read this into a prompt
    seconds: float
    argv: List[str] = field(default_factory=list)


def raw_output_dir() -> Path:
    """Human-only directory for unfiltered organiser stdout."""
    cfg = hdata.load_config().get('paths', {})
    return hdata.repo_root() / cfg.get('runs_dir', 'runs') / 'raw_starter_output'


def run_starter_script(script: str,
                       args: Sequence[str] = (),
                       *,
                       data_dir: str | os.PathLike | None = None,
                       timeout: float | None = None,
                       raw_log: str | os.PathLike | None = None,
                       write_raw_log: bool = True) -> StarterRun:
    """Run ``starter/<script>`` in a subprocess and return filtered output.

    The unfiltered combined output is written to a human-only file. Nothing else
    in this process ever sees it. ``--data_dir`` is appended automatically unless
    *args* already supplies one.
    """
    starter = hdata.starter_dir()
    target = starter / script
    if not target.exists():
        raise FileNotFoundError(f'no such starter script: {target}')

    argv = [sys.executable, script, *map(str, args)]
    if '--data_dir' not in argv:
        argv += ['--data_dir', str(hdata.data_dir(data_dir))]

    env = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONUTF8='1')
    started = time.time()
    proc = subprocess.run(argv, cwd=str(starter), env=env, timeout=timeout,
                          capture_output=True, text=True,
                          encoding='utf-8', errors='replace')
    seconds = time.time() - started

    raw_path: Path | None = None
    if write_raw_log:
        raw_path = Path(raw_log) if raw_log is not None else (
            raw_output_dir() / f'{Path(script).stem}-{int(started)}.log')
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(
            f'# HUMAN-ONLY. Unfiltered organiser stdout; may contain test metrics.\n'
            f'# Never read this file into an LLM prompt or a machine-readable log.\n'
            f'# argv: {argv}\n\n'
            f'--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}\n',
            encoding='utf-8')

    clean_out, n_out = filter_stdout(proc.stdout or '')
    clean_err, n_err = filter_stdout(proc.stderr or '')
    return StarterRun(returncode=proc.returncode, stdout=clean_out, stderr=clean_err,
                      redacted_lines=n_out + n_err, raw_log_path=raw_path,
                      seconds=seconds, argv=argv)


# --------------------------------------------------------------------------
# 3. the leak canary
# --------------------------------------------------------------------------

def canary_threshold() -> float:
    """Validation primary above which a result is treated as a leak, not a win.

    The validation oracle ceiling is 0.8484. Anything near it is a bug.
    """
    return float(hdata.load_config().get('leakage', {})
                 .get('canary_primary_threshold', 0.80))


def quarantine_dir() -> Path:
    """Where tripped configurations are parked for human inspection."""
    cfg = hdata.load_config().get('paths', {})
    return hdata.repo_root() / cfg.get('runs_dir', 'runs') / 'quarantine'


def review_threshold() -> float:
    """Validation primary above which a **kept** result is flagged for a human.

    The lower of the two tiers. See ``flag_for_review``.
    """
    return float(hdata.load_config().get('leakage', {})
                 .get('review_primary_threshold', 0.68))


def review_dir() -> Path:
    """Where flagged-but-kept results are recorded for pre-submission review."""
    cfg = hdata.load_config().get('paths', {})
    return hdata.repo_root() / cfg.get('runs_dir', 'runs') / 'review'


def flag_for_review(val_primary: float, *, context: Dict[str, Any] | None = None,
                    threshold: float | None = None,
                    record: bool = True) -> bool:
    """Flag a kept result as implausibly good. Returns True if flagged.

    **This never raises and never rejects.** The result is kept and can still win;
    it is simply marked so a human looks at it before anything is submitted.

    Why this tier exists at all. The canary catches leaks scoring above 0.80. The
    result that actually costs us the competition is its quieter sibling: a
    sub-threshold 0.72 from the same leak path, which does not trip anything, looks
    like a genuine breakthrough, and is kept and submitted. A leak that *never*
    crosses 0.80 is invisible to the canary entirely.

    Unconditional, not triggered by a canary trip. The invisible case has no trip
    to trigger on.
    """
    limit = review_threshold() if threshold is None else threshold
    if not (val_primary > limit):
        return False
    if record:
        directory = review_dir()
        directory.mkdir(parents=True, exist_ok=True)
        payload = {'val_primary': float(val_primary), 'threshold': limit,
                   'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
                   'context': context or {},
                   'verdict': 'KEPT but flagged: a gain this large is implausible '
                              'on this benchmark. A human must inspect this '
                              'checkpoint before it is submitted.'}
        path = directory / f'review-{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}.json'
        path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    return True


def review_flags(directory: Path | None = None) -> List[Dict[str, Any]]:
    """Every result flagged for review in this run, oldest first."""
    directory = review_dir() if directory is None else Path(directory)
    if not directory.exists():
        return []
    out: List[Dict[str, Any]] = []
    for path in sorted(directory.glob('review-*.json')):
        try:
            out.append(json.loads(path.read_text(encoding='utf-8')))
        except (OSError, json.JSONDecodeError):
            continue
    return out


def review_flag_count(directory: Path | None = None) -> int:
    """How many kept results need a human's eye before submission."""
    return len(review_flags(directory))


def audit_kept_results(results: Iterable[Dict[str, Any]],
                       threshold: float | None = None) -> List[Dict[str, Any]]:
    """Retrospective sweep over results already kept, newest risk first.

    Run after a canary trip. A trip says a leak path exists *now*; it says nothing
    about when the path opened. The same leak may have produced a quieter,
    already-banked result several iterations ago, and stopping the run at that
    point would be forward-looking protection against a backward-looking risk.
    This is the part that actually reaches it.
    """
    limit = review_threshold() if threshold is None else threshold
    flagged = [r for r in results
               if r.get('val_primary') is not None and r['val_primary'] > limit]
    return sorted(flagged, key=lambda r: -r['val_primary'])


def quarantined_records(directory: Path | None = None) -> List[Dict[str, Any]]:
    """Every canary trip recorded for this run, oldest first.

    Read off the filesystem rather than held in memory, so the count survives a
    crash-and-restart exactly like the convergence counters do.
    """
    directory = quarantine_dir() if directory is None else Path(directory)
    if not directory.exists():
        return []
    out: List[Dict[str, Any]] = []
    for path in sorted(directory.glob('canary-*.json')):
        try:
            out.append(json.loads(path.read_text(encoding='utf-8')))
        except (OSError, json.JSONDecodeError):
            continue
    return out


def canary_trip_count(directory: Path | None = None) -> int:
    """How many times the canary has fired in this run.

    The escalation policy lives with the caller, but the number lives here.
    See ``docs/M2_CONTRACT.md`` section 6: one trip is recorded and the run
    continues; a second trip stops the run.
    """
    return len(quarantined_records(directory))


def check_canary(val_primary: float,
                 *,
                 context: Dict[str, Any] | None = None,
                 threshold: float | None = None,
                 quarantine: bool = True,
                 raise_on_trip: bool = True) -> bool:
    """Return True if *val_primary* trips the canary.

    A tripped result is written to ``runs/quarantine/`` and, by default, raises.
    The caller must never keep a quarantined checkpoint.
    """
    limit = canary_threshold() if threshold is None else threshold
    if val_primary <= limit:
        return False

    record = {'val_primary': float(val_primary), 'threshold': limit,
              'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
              'context': context or {},
              'verdict': 'quarantined: validation primary above the leak threshold; '
                         'the validation oracle ceiling is 0.8484'}
    if quarantine:
        directory = quarantine_dir()
        directory.mkdir(parents=True, exist_ok=True)
        # A millisecond timestamp alone collides: two trips inside the same
        # millisecond overwrote each other, which silently undercounted exactly
        # the thing the escalation policy depends on.
        path = directory / f'canary-{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}.json'
        path.write_text(json.dumps(record, indent=2), encoding='utf-8')
    if raise_on_trip:
        raise LeakCanaryError(
            f'validation primary {val_primary:.4f} exceeds the leak canary '
            f'threshold {limit:.2f}. Quarantined; this configuration must not be '
            f'kept or submitted.')
    return True
