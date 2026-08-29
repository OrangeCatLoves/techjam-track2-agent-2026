"""Run artefacts: what was tried, what it scored, and which checkpoint won.

OWNS
    - the run directory layout under ``runs/<run_id>/``
    - the append-only record of every iteration
    - **checkpoint promotion and rollback**, which is what makes keep-or-reject
      mean something rather than being a word in a log
    - archiving each generated patch, so ``harness/models/gen/`` never accumulates
      stale code between iterations
    - the retrospective audit input: every result actually kept

MUST NEVER
    - own a second iteration counter or a second tried-set.
      ``harness/convergence.py`` is authoritative for both; two sources of truth
      for "which iteration is this" is how a restart silently double-counts
    - hold a hidden-test metric. Every record passes
      ``guards.assert_record_clean`` before it is written
    - rewrite history. The ledger is append-only: a rejected iteration is recorded
      as rejected, not deleted. A judge reading the log should see the failures

WHAT ROLLBACK ACTUALLY IS
    Model state lives in checkpoint files, never in a mutable global. So rejecting
    an experiment is not an undo: it is simply declining to move the best-pointer.
    That makes rollback total and free, and it is why a failed swing costs nothing
    (CLAUDE.md section 3.4).
"""
from __future__ import annotations

import json
import shutil
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List

from harness import data as hdata
from harness import guards

LEDGER_VERSION = 1

DECISION_KEEP = 'keep'
DECISION_REJECT = 'reject'
DECISION_FAILED = 'failed'
DECISIONS = (DECISION_KEEP, DECISION_REJECT, DECISION_FAILED)


def new_run_id() -> str:
    """Sortable and unique: ``20260830-143022-a1b2c3``."""
    return f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"


@dataclass
class BestRecord:
    """The validation-best checkpoint. This is what gets submitted."""
    iteration: int
    val_primary: float
    checkpoint: str
    flagged_for_review: bool = False
    metrics: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {'iteration': self.iteration, 'val_primary': self.val_primary,
                'checkpoint': self.checkpoint,
                'flagged_for_review': self.flagged_for_review,
                'metrics': self.metrics}


class Ledger:
    """One run's artefacts on disk. Resumes rather than restarts."""

    def __init__(self, run_dir: str | Path, run_id: str | None = None) -> None:
        self.run_dir = Path(run_dir)
        self.run_id = run_id or self.run_dir.name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        for sub in ('patches', 'checkpoints', 'best'):
            (self.run_dir / sub).mkdir(exist_ok=True)
        self._best: BestRecord | None = None
        self._load_best()

    # -- construction ------------------------------------------------------

    @classmethod
    def open(cls, run_dir: str | Path | None = None, *,
             run_id: str | None = None) -> 'Ledger':
        """Open or resume a run.

        With no arguments, creates a new run under ``runs/``. With *run_id*,
        resumes that run if it exists. Resuming is never destructive.
        """
        if run_dir is not None:
            return cls(run_dir, run_id=run_id)
        root = hdata.repo_root() / hdata.load_config().get('paths', {}).get(
            'runs_dir', 'runs')
        chosen = run_id or new_run_id()
        return cls(root / chosen, run_id=chosen)

    @classmethod
    def latest(cls, runs_root: str | Path | None = None) -> 'Ledger | None':
        """The most recent run, for a restart that was not told which run it is."""
        root = Path(runs_root) if runs_root is not None else (
            hdata.repo_root() / hdata.load_config().get('paths', {}).get(
                'runs_dir', 'runs'))
        if not root.exists():
            return None
        candidates = sorted(p for p in root.iterdir()
                            if p.is_dir() and (p / 'ledger.jsonl').exists())
        return cls(candidates[-1]) if candidates else None

    # -- paths -------------------------------------------------------------

    @property
    def ledger_path(self) -> Path:
        return self.run_dir / 'ledger.jsonl'

    @property
    def convergence_path(self) -> Path:
        """Where the tracker keeps its state. The tracker owns the contents."""
        return self.run_dir / 'convergence.json'

    @property
    def best_path(self) -> Path:
        return self.run_dir / 'best' / 'best.json'

    @property
    def submission_path(self) -> Path:
        return self.run_dir / 'submission.csv'

    def checkpoint_path(self, iteration: int) -> Path:
        return self.run_dir / 'checkpoints' / f'iter_{iteration:03d}.npz'

    def archived_patch_path(self, iteration: int) -> Path:
        return self.run_dir / 'patches' / f'iter_{iteration:03d}.py'

    # -- generated code ----------------------------------------------------

    @staticmethod
    def gen_dir() -> Path:
        """The one directory generated model code may live in."""
        return hdata.repo_root() / 'harness' / 'models' / 'gen'

    def new_patch_path(self, iteration: int) -> Path:
        """Where iteration *n*'s patch is written before it runs."""
        return self.gen_dir() / f'iter_{iteration:03d}.py'

    def archive_patch(self, iteration: int, patch_path: str | Path) -> Path | None:
        """Move a used patch out of ``gen/`` and into the run record.

        Kept for the log whether the experiment succeeded or failed -- a judge
        reading the run should see the code that did not work as well as the code
        that did.
        """
        source = Path(patch_path)
        if not source.exists():
            return None
        target = self.archived_patch_path(iteration)
        shutil.move(str(source), str(target))
        return target

    def clean_gen(self) -> int:
        """Remove leftover generated code. Returns how many files went.

        Run between iterations. A stale patch in ``gen/`` is code nobody decided
        to run, sitting where code that will run is expected to be.
        """
        removed = 0
        for path in self.gen_dir().glob('*.py'):
            path.unlink()
            removed += 1
        cache = self.gen_dir() / '__pycache__'
        if cache.exists():
            shutil.rmtree(cache, ignore_errors=True)
        return removed

    # -- records -----------------------------------------------------------

    def record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Append one iteration record. Screened, then written, never rewritten."""
        payload = dict(record)
        payload.setdefault('run_id', self.run_id)
        payload.setdefault('timestamp', time.strftime('%Y-%m-%dT%H:%M:%S'))
        payload.setdefault('ledger_version', LEDGER_VERSION)
        decision = payload.get('decision')
        if decision is not None and decision not in DECISIONS:
            raise ValueError(f'unknown decision {decision!r}; choose from {DECISIONS}')

        guards.assert_record_clean(payload, where='ledger record')
        with open(self.ledger_path, 'a', encoding='utf-8') as fh:
            fh.write(json.dumps(payload, default=str) + '\n')
        return payload

    def records(self) -> List[Dict[str, Any]]:
        """Every record, in order. Survives a restart because it is on disk."""
        if not self.ledger_path.exists():
            return []
        out: List[Dict[str, Any]] = []
        for line in self.ledger_path.read_text(encoding='utf-8').splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue                      # a torn final line after a hard kill
        return out

    def kept_results(self) -> List[Dict[str, Any]]:
        """Results that were kept. The input to the retrospective leak audit."""
        return [r for r in self.records() if r.get('decision') == DECISION_KEEP]

    # -- the best checkpoint ----------------------------------------------

    def _load_best(self) -> None:
        if self.best_path.exists():
            try:
                blob = json.loads(self.best_path.read_text(encoding='utf-8'))
                self._best = BestRecord(**blob)
            except (json.JSONDecodeError, TypeError):
                self._best = None

    def best(self) -> BestRecord | None:
        """The validation-best checkpoint so far, or None before the first keep."""
        return self._best

    def promote(self, iteration: int, checkpoint: str | Path,
                val_primary: float, *, metrics: Dict[str, Any] | None = None,
                flagged_for_review: bool = False) -> BestRecord:
        """Make this iteration's checkpoint the one that would be submitted.

        Copies the checkpoint into ``best/`` rather than pointing at it, so a
        later cleanup of per-iteration checkpoints cannot orphan the winner.
        """
        source = Path(checkpoint)
        if not source.exists():
            raise FileNotFoundError(f'cannot promote a missing checkpoint: {source}')
        target = self.run_dir / 'best' / 'model.npz'
        shutil.copy2(source, target)

        self._best = BestRecord(iteration=iteration, val_primary=float(val_primary),
                                checkpoint=str(target),
                                flagged_for_review=flagged_for_review,
                                metrics=metrics or {})
        guards.assert_record_clean(self._best.as_dict(), where='best record')
        self.best_path.write_text(json.dumps(self._best.as_dict(), indent=2),
                                  encoding='utf-8')
        return self._best

    def would_improve(self, val_primary: float | None, epsilon: float = 0.0) -> bool:
        """Whether *val_primary* beats the current best by more than *epsilon*.

        ``None`` and ``NaN`` are not improvements. That is the whole point: a NaN
        compared with ``>`` returns False and reads as an ordinary
        non-improvement, so the check is made explicit here rather than left to
        the caller to remember.
        """
        if val_primary is None or val_primary != val_primary:      # NaN
            return False
        if self._best is None:
            return True
        return float(val_primary) > self._best.val_primary + epsilon

    # -- reporting ---------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        """Counts a human or a report needs. No test metrics, by construction."""
        records = self.records()
        decisions = [r.get('decision') for r in records]
        best = self.best()
        return {
            'run_id': self.run_id,
            'run_dir': str(self.run_dir),
            'records': len(records),
            'kept': decisions.count(DECISION_KEEP),
            'rejected': decisions.count(DECISION_REJECT),
            'failed': decisions.count(DECISION_FAILED),
            'best_iteration': None if best is None else best.iteration,
            'best_val_primary': None if best is None else best.val_primary,
            'best_flagged_for_review': bool(best and best.flagged_for_review),
            'canary_trips': guards.canary_trip_count(),
            'review_flags': guards.review_flag_count(),
        }

    def audit(self) -> List[Dict[str, Any]]:
        """Kept results that are implausibly good. See D13.

        Run after any canary trip, and once more before submission. A trip says a
        leak exists now; it says nothing about when the path opened, so the sweep
        looks backwards over everything already banked.
        """
        return guards.audit_kept_results(self.kept_results())
