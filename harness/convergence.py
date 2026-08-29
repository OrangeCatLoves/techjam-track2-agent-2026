"""The stopping rule. Implemented literally, and made restart-proof.

OWNS
    - the official convergence test:
        converged = three consecutive iterations each improving validation
                    primary by <= 0.002
                 OR iteration count == 50
                 OR wall clock == 6 hours
    - the iteration counter, the strike counter, the tried-set and the
      validation-best record
    - persistence of all of the above, so that a crash-and-restart resumes rather
      than restarts

MUST NEVER
    - stop voluntarily before the rule fires. ``should_continue`` is the only
      permitted stop signal and ``assert_may_stop`` enforces it
    - continue after the rule fires. ``record_iteration`` raises once converged
    - reset a counter on restart. Resetting would be gaming the rule
    - see, store or compare a hidden-test metric. Every score handled here is a
      validation score

THE OPEN QUESTION (Q3)
    ``baseline_scores.json`` supplies ``epsilon = 0.002`` and ``N = 3`` but not the
    comparison semantics. Two readings are defensible:

    ``per_iteration``  every one of the last N iterations improved the running
                       best by <= epsilon.  **Default.**
    ``block``          the best of the last N minus the best before them is
                       <= epsilon.

    ``per_iteration`` is the stricter reading: whenever ``block`` fires,
    ``per_iteration`` has already fired, because a sum of N gains cannot be
    <= epsilon unless each of them is. It therefore stops no later than ``block``,
    which is the safe side of a rule we are self-enforcing. It is also the reading
    written in CLAUDE.md section 3.4. Both live in ``COMPARISONS`` behind one
    switch, so an organiser ruling is a one-word config change.

FAILED ITERATIONS (Q4)
    Default in force: an abandoned iteration consumes one of the 50 but does not
    count as a non-improving iteration, so it neither adds nor clears a strike.

WALL CLOCK ACROSS RESTARTS (Q5)
    Accumulated *active* agent time is persisted and resumed. Time while the
    process is not running is not charged to the 6-hour budget; a crash at 02:00
    discovered at 09:00 would otherwise exhaust the budget by accident. Recorded
    in docs/OPEN_QUESTIONS.md.
"""
from __future__ import annotations

import json
import math
import time
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence

from harness import data as hdata

STATE_VERSION = 1

STATUS_OK = 'ok'
STATUS_FAILED = 'failed'

REASON_NO_IMPROVEMENT = 'no_improvement'
REASON_MAX_ITERATIONS = 'max_iterations'
REASON_WALL_CLOCK = 'wall_clock'


class ConvergedError(RuntimeError):
    """Raised on an attempt to keep iterating after convergence fired."""


class EarlyStopError(RuntimeError):
    """Raised on an attempt to stop before convergence fired."""


# --------------------------------------------------------------------------
# the switchable comparison (Q3)
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ScoredIteration:
    """One iteration that produced a validation score."""
    iteration: int
    primary: float
    best_before: float | None   # running best before this iteration, None if first

    @property
    def gain(self) -> float:
        """Improvement over the running best. ``inf`` for the first score."""
        if self.best_before is None:
            return math.inf
        return self.primary - self.best_before


def _converged_per_iteration(scored: Sequence[ScoredIteration],
                             epsilon: float, n: int) -> bool:
    """Every one of the last *n* scored iterations gained <= epsilon."""
    if len(scored) < n:
        return False
    return all(s.gain <= epsilon + 1e-12 for s in scored[-n:])


def _converged_block(scored: Sequence[ScoredIteration],
                     epsilon: float, n: int) -> bool:
    """best(last n) - best(before those n) <= epsilon."""
    if len(scored) < n:
        return False
    window = scored[-n:]
    baseline = window[0].best_before
    if baseline is None:
        return False
    return max(s.primary for s in window) - baseline <= epsilon + 1e-12


#: The one switch. Change ``convergence.comparison`` in configs/base.yaml when
#: Q3 is answered; nothing else moves.
COMPARISONS: Dict[str, Callable[[Sequence[ScoredIteration], float, int], bool]] = {
    'per_iteration': _converged_per_iteration,
    'block': _converged_block,
}
DEFAULT_COMPARISON = 'per_iteration'


def trailing_strikes(scored: Sequence[ScoredIteration], epsilon: float) -> int:
    """Number of trailing iterations that improved the best by <= epsilon.

    Reported for every comparison mode as a human-readable progress signal.
    Under ``per_iteration`` it is exactly the convergence counter.
    """
    count = 0
    for s in reversed(scored):
        if s.gain <= epsilon + 1e-12:
            count += 1
        else:
            break
    return count


# --------------------------------------------------------------------------
# status
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ConvergenceStatus:
    """A snapshot. Everything the loop needs to decide whether to go again."""
    converged: bool
    reason: str | None
    iteration: int
    strikes: int
    best_primary: float | None
    best_iteration: int | None
    elapsed_seconds: float
    remaining_iterations: int
    remaining_seconds: float
    scored_iterations: int
    failed_iterations: int

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------
# the tracker
# --------------------------------------------------------------------------

class ConvergenceTracker:
    """Iteration counter, strike counter, tried-set and best checkpoint.

    All state is written to *state_path* after every mutation, so a kill at any
    point resumes from the last completed iteration.
    """

    def __init__(self,
                 state_path: str | Path,
                 *,
                 epsilon: float | None = None,
                 n_consecutive: int | None = None,
                 max_iterations: int | None = None,
                 max_wall_clock_hours: float | None = None,
                 comparison: str | None = None,
                 allow_early_stop: bool | None = None,
                 initial_best: float | None = None,
                 run_id: str | None = None,
                 clock: Callable[[], float] = time.monotonic) -> None:
        cfg = {}
        try:
            cfg = hdata.load_config().get('convergence', {}) or {}
        except Exception:                      # config is optional for unit tests
            cfg = {}

        self.state_path = Path(state_path)
        self.epsilon = float(cfg.get('epsilon', 0.002) if epsilon is None else epsilon)
        self.n_consecutive = int(cfg.get('n_consecutive', 3)
                                 if n_consecutive is None else n_consecutive)
        self.max_iterations = int(cfg.get('max_iterations', 50)
                                  if max_iterations is None else max_iterations)
        self.max_wall_clock_hours = float(cfg.get('max_wall_clock_hours', 6)
                                          if max_wall_clock_hours is None
                                          else max_wall_clock_hours)
        comparison = comparison if comparison is not None else cfg.get(
            'comparison', DEFAULT_COMPARISON)
        if comparison not in COMPARISONS:
            raise ValueError(f'unknown comparison {comparison!r}; '
                             f'choose from {sorted(COMPARISONS)}')
        self.comparison = comparison
        self.allow_early_stop = bool(cfg.get('allow_early_stop', False)
                                     if allow_early_stop is None else allow_early_stop)
        self._clock = clock

        self.run_id: str = run_id or uuid.uuid4().hex[:12]
        self.iteration: int = 0
        self.failed_iterations: int = 0
        self.best_primary: float | None = initial_best
        self.best_iteration: int | None = None
        self.best_ref: Any = None
        self.history: List[Dict[str, Any]] = []
        self.scored: List[ScoredIteration] = []
        self.tried: List[str] = []
        self._elapsed_seconds: float = 0.0
        self._session_anchor: float | None = None
        self._converged_reason: str | None = None
        self._converged_at: int | None = None

    # -- persistence -------------------------------------------------------

    @classmethod
    def open(cls, state_path: str | Path, **kwargs: Any) -> 'ConvergenceTracker':
        """Resume from *state_path* if it exists, otherwise start a fresh run.

        This is the only constructor the agent loop should use. A restart must
        not reset a counter, so resuming is the default rather than an option.
        """
        tracker = cls(state_path, **kwargs)
        if tracker.state_path.exists():
            tracker.reload()
        else:
            tracker.save()
        return tracker

    def to_dict(self) -> Dict[str, Any]:
        return {
            'version': STATE_VERSION,
            'run_id': self.run_id,
            'rule': {'epsilon': self.epsilon, 'n_consecutive': self.n_consecutive,
                     'max_iterations': self.max_iterations,
                     'max_wall_clock_hours': self.max_wall_clock_hours,
                     'comparison': self.comparison,
                     'allow_early_stop': self.allow_early_stop},
            'iteration': self.iteration,
            'failed_iterations': self.failed_iterations,
            'strikes': self.strikes,
            'best_primary': self.best_primary,
            'best_iteration': self.best_iteration,
            'best_ref': self.best_ref,
            'elapsed_seconds': self.elapsed_seconds,
            'converged': self.status().converged,
            'converged_reason': self.status().reason,
            'converged_at_iteration': self._converged_at,
            'history': self.history,
            'tried': self.tried,
        }

    def save(self) -> None:
        """Atomically persist state. Called after every mutation."""
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(self.state_path.suffix + '.tmp')
        tmp.write_text(json.dumps(self.to_dict(), indent=2), encoding='utf-8')
        tmp.replace(self.state_path)

    def reload(self) -> None:
        """Restore state from disk. Counters are taken as-is, never reset."""
        state = json.loads(self.state_path.read_text(encoding='utf-8'))
        if state.get('version') != STATE_VERSION:
            raise ValueError(f'convergence state version {state.get("version")} '
                             f'is not {STATE_VERSION}')
        rule = state.get('rule', {})
        self.epsilon = float(rule.get('epsilon', self.epsilon))
        self.n_consecutive = int(rule.get('n_consecutive', self.n_consecutive))
        self.max_iterations = int(rule.get('max_iterations', self.max_iterations))
        self.max_wall_clock_hours = float(rule.get('max_wall_clock_hours',
                                                   self.max_wall_clock_hours))
        self.comparison = rule.get('comparison', self.comparison)
        self.allow_early_stop = bool(rule.get('allow_early_stop',
                                              self.allow_early_stop))
        self.run_id = state.get('run_id', self.run_id)
        self.iteration = int(state.get('iteration', 0))
        self.failed_iterations = int(state.get('failed_iterations', 0))
        self.best_primary = state.get('best_primary')
        self.best_iteration = state.get('best_iteration')
        self.best_ref = state.get('best_ref')
        self._elapsed_seconds = float(state.get('elapsed_seconds', 0.0))
        self._converged_at = state.get('converged_at_iteration')
        self._converged_reason = state.get('converged_reason')
        self.history = list(state.get('history', []))
        self.tried = list(state.get('tried', []))
        self.scored = [ScoredIteration(h['iteration'], h['primary'], h['best_before'])
                       for h in self.history if h.get('status') == STATUS_OK]
        self._session_anchor = None

    # -- wall clock --------------------------------------------------------

    def start_session(self) -> None:
        """Begin charging wall clock. Idempotent."""
        if self._session_anchor is None:
            self._session_anchor = self._clock()

    def end_session(self) -> None:
        """Stop charging wall clock and persist the accumulated total."""
        if self._session_anchor is not None:
            self._elapsed_seconds += self._clock() - self._session_anchor
            self._session_anchor = None
            self.save()

    @property
    def elapsed_seconds(self) -> float:
        """Accumulated active agent time across all sessions of this run."""
        live = 0.0 if self._session_anchor is None else self._clock() - self._session_anchor
        return self._elapsed_seconds + live

    @property
    def max_wall_clock_seconds(self) -> float:
        return self.max_wall_clock_hours * 3600.0

    # -- tried set ---------------------------------------------------------

    def has_tried(self, content_hash: str) -> bool:
        """True if an experiment with this content hash was already proposed."""
        return content_hash in self.tried

    def mark_tried(self, content_hash: str) -> None:
        """Record a content hash so the same experiment is never re-proposed."""
        if content_hash and content_hash not in self.tried:
            self.tried.append(content_hash)
            self.save()

    # -- the counters ------------------------------------------------------

    @property
    def strikes(self) -> int:
        """Trailing iterations that improved the best by <= epsilon."""
        return trailing_strikes(self.scored, self.epsilon)

    def status(self) -> ConvergenceStatus:
        """Evaluate the rule. Pure; safe to call as often as you like."""
        reason: str | None = None
        if COMPARISONS[self.comparison](self.scored, self.epsilon, self.n_consecutive):
            reason = REASON_NO_IMPROVEMENT
        elif self.iteration >= self.max_iterations:
            reason = REASON_MAX_ITERATIONS
        elif self.elapsed_seconds >= self.max_wall_clock_seconds:
            reason = REASON_WALL_CLOCK
        return ConvergenceStatus(
            converged=reason is not None,
            reason=reason,
            iteration=self.iteration,
            strikes=self.strikes,
            best_primary=self.best_primary,
            best_iteration=self.best_iteration,
            elapsed_seconds=self.elapsed_seconds,
            remaining_iterations=max(0, self.max_iterations - self.iteration),
            remaining_seconds=max(0.0, self.max_wall_clock_seconds - self.elapsed_seconds),
            scored_iterations=len(self.scored),
            failed_iterations=self.failed_iterations,
        )

    def should_continue(self) -> bool:
        """The only permitted stop signal."""
        return not self.status().converged

    def assert_may_stop(self) -> None:
        """Raise unless the rule has fired. Guards property 1."""
        if not self.status().converged and not self.allow_early_stop:
            raise EarlyStopError(
                f'convergence has not fired (iteration {self.iteration}/'
                f'{self.max_iterations}, strikes {self.strikes}/'
                f'{self.n_consecutive}); stopping now would be a voluntary stop, '
                f'which configs/base.yaml forbids.')

    def record_iteration(self,
                         primary: float | None = None,
                         *,
                         status: str = STATUS_OK,
                         content_hash: str | None = None,
                         checkpoint_ref: Any = None,
                         meta: Dict[str, Any] | None = None) -> ConvergenceStatus:
        """Record one completed iteration and re-evaluate the rule.

        *primary* is a **validation** primary score. ``status=STATUS_FAILED``
        records an abandoned iteration: it consumes one of the 50 but is not a
        non-improving iteration, so the strike streak is left untouched (Q4).

        Raises ``ConvergedError`` if the rule has already fired: property 2 says
        never continue after convergence.
        """
        before = self.status()
        if before.converged:
            raise ConvergedError(
                f'convergence already fired at iteration {self._converged_at} '
                f'({before.reason}); recording another iteration would break the '
                f'rule. See CLAUDE.md section 3.4.')

        if status not in (STATUS_OK, STATUS_FAILED):
            raise ValueError(f'unknown iteration status {status!r}')
        if status == STATUS_OK and primary is None:
            raise ValueError('a successful iteration must report a validation primary')
        if status == STATUS_OK and not math.isfinite(float(primary)):
            # This is the one place a score is compared against the best, so it is
            # the one place that must refuse a score it cannot compare.
            #
            # A NaN slipped through here is not merely ignored, it is actively
            # harmful: `nan <= epsilon` is False, so the iteration is recorded as
            # a non-strike and the trailing streak RESETS. A broken objective
            # emitting NaN would clear the strike counter and keep the run going
            # indefinitely -- the exact opposite of the rule.
            #
            # The caller checks `ExperimentResult.usable` before getting here, so
            # reaching this line is a loop bug, and loop bugs are loud.
            raise ValueError(
                f'refusing to record a non-finite validation primary ({primary!r}). '
                f'A NaN would reset the strike streak rather than count against it. '
                f'Check ExperimentResult.usable before recording, and record a '
                f'non-finite score as a failed iteration instead.')

        self.iteration += 1
        record: Dict[str, Any] = {
            'iteration': self.iteration,
            'status': status,
            'primary': None if primary is None else float(primary),
            'best_before': self.best_primary,
            'gain': None,
            'strike': False,
            'content_hash': content_hash,
            'meta': meta or {},
        }

        if status == STATUS_FAILED:
            self.failed_iterations += 1
        else:
            scored = ScoredIteration(self.iteration, float(primary), self.best_primary)
            record['gain'] = None if self.best_primary is None else scored.gain
            record['strike'] = scored.gain <= self.epsilon + 1e-12
            self.scored.append(scored)
            if self.best_primary is None or float(primary) > self.best_primary:
                self.best_primary = float(primary)
                self.best_iteration = self.iteration
                if checkpoint_ref is not None:
                    self.best_ref = checkpoint_ref
                record['new_best'] = True

        if content_hash:
            if content_hash not in self.tried:
                self.tried.append(content_hash)

        self.history.append(record)

        after = self.status()
        if after.converged and self._converged_at is None:
            self._converged_at = self.iteration
            self._converged_reason = after.reason
        self.save()
        return after

    def record_failure(self, *, content_hash: str | None = None,
                       error: str | None = None) -> ConvergenceStatus:
        """Shorthand for an abandoned iteration."""
        return self.record_iteration(None, status=STATUS_FAILED,
                                     content_hash=content_hash,
                                     meta={'error': error} if error else None)

    # -- reporting ---------------------------------------------------------

    def summary(self) -> str:
        """One human-readable block for the run log."""
        st = self.status()
        best = 'none' if st.best_primary is None else f'{st.best_primary:.4f}'
        return (f'run {self.run_id} | iteration {st.iteration}/{self.max_iterations} '
                f'| strikes {st.strikes}/{self.n_consecutive} '
                f'| best valid primary {best} (iteration {st.best_iteration}) '
                f'| elapsed {st.elapsed_seconds / 3600:.2f}h/'
                f'{self.max_wall_clock_hours:.0f}h '
                f'| converged={st.converged}'
                + (f' ({st.reason})' if st.reason else ''))
