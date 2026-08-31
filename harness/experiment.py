"""``run_experiment`` -- the only thing that crosses the harness/agent boundary.

OWNS
    - the frozen return shape (CLAUDE.md section 11.1, interface 1)
    - the sequence: validate the patch, run it sandboxed, read the result, screen it
    - the stub, which is how the agent half is developed without waiting a minute
      per training run

MUST NEVER
    - return a hidden-test metric, in any field, under any flag
    - raise for an experiment failure. A crash, a timeout, a memory breach and a
      rejected patch are all *returned values*: the loop has to record them,
      spend its one repair attempt, and continue. Only a harness bug raises
    - let the agent see anything the guards have not screened

WHY THE STUB IS HERE AND NOT IN THE TESTS
    The agent half develops against it, so it ships with the harness. And it can
    **fail**: a stub that only returns plausible numbers means the recovery path
    is written blind and gets its first real exercise during the scored run, which
    is the one run that must not need a human.
"""
from __future__ import annotations

import json
import math
import tempfile
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence

import numpy as np

from harness import data as hdata
from harness import guards
from harness import patch as hpatch
from harness import sandbox

#: ``error_kind`` vocabulary. Fixed, because it selects the recovery path
#: (CLAUDE.md section 6.3).
ERROR_CODE = 'code'            # generated code raised -> one repair attempt
ERROR_TIMEOUT = 'timeout'      # -> retry once at 30% subsample
ERROR_MEMORY = 'memory'        # -> retry once at float32, half the features
ERROR_EVALUATOR = 'evaluator'  # -> hard failure, never patch around it
ERROR_REJECTED = 'rejected'    # -> patch validation refused it; nothing ran
ERROR_CANARY = 'canary'        # -> scored above the leak threshold; quarantined
ERROR_KINDS = (ERROR_CODE, ERROR_TIMEOUT, ERROR_MEMORY, ERROR_EVALUATOR,
               ERROR_REJECTED, ERROR_CANARY)

#: Kinds that are never repaired and never retried. A repair attempt on these
#: would be an attempt to patch around a guard.
HARD_FAILURES = (ERROR_EVALUATOR, ERROR_REJECTED, ERROR_CANARY)


@dataclass
class ExperimentResult:
    """Everything an experiment is allowed to tell the agent. Nothing else."""
    ok: bool
    val_gauc: float | None = None
    val_ndcg5: float | None = None
    val_primary: float | None = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    checkpoint: str | None = None
    error: str | None = None
    error_kind: str | None = None
    seconds: float = 0.0
    seed: int = 0
    stdout: str = ''
    peak_memory_mb: float = 0.0
    #: Kept, but implausibly good. A human inspects it before anything is
    #: submitted. Not a rejection: this result can still be the winner.
    flagged_for_review: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def usable(self) -> bool:
        """True if this result may be compared against the current best.

        A NaN primary is *not* usable. Left unguarded it would slip through
        ``primary > best`` as ``False`` and look like an ordinary non-improvement,
        which hides a broken objective rather than reporting it.
        """
        return (self.ok and self.val_primary is not None
                and math.isfinite(self.val_primary))


def _screen(result: ExperimentResult, *, quarantine: bool = True
            ) -> ExperimentResult:
    """Last gate before anything reaches the agent.

    A tripped canary is converted into a **failed result**, not an exception. The
    contract says this function never raises for an experiment failure, and a
    leaking experiment is still an experiment: the loop must record it, roll back,
    mark it tried and carry on. Halting instead would require a human to restart,
    which costs us on autonomy for an event the quarantine file already makes
    visible.

    It is a *hard* failure, so it is never repaired and never retried. Repairing a
    canary trip would be patching around a guard.
    """
    guards.assert_record_clean(result.diagnostics, where='experiment diagnostics')
    if result.stdout:
        guards.assert_no_test_metrics(result.stdout, where='experiment stdout')
    if result.error:
        guards.assert_no_test_metrics(result.error, where='experiment error')
    if result.usable:
        try:
            guards.check_canary(float(result.val_primary),
                                quarantine=quarantine,
                                context={'source': 'run_experiment',
                                         'seed': result.seed})
        except guards.LeakCanaryError as exc:
            return ExperimentResult(
                ok=False, error_kind=ERROR_CANARY, seed=result.seed,
                seconds=result.seconds, stdout=result.stdout,
                peak_memory_mb=result.peak_memory_mb,
                diagnostics=result.diagnostics,
                error=(f'{exc} The result is quarantined and must not be kept '
                       f'or submitted.'))
        # The lower tier. This result is KEPT and may still win; it is only
        # marked, because the leak that costs us the competition is the one that
        # never crosses the canary at all.
        result.flagged_for_review = guards.flag_for_review(
            float(result.val_primary), record=quarantine,
            context={'source': 'run_experiment', 'seed': result.seed})
    return result


def _timeout_seconds() -> float:
    minutes = hdata.load_config().get('agent', {}).get(
        'per_iteration_timeout_minutes', 25)
    return float(minutes) * 60.0


def _memory_limit_gb() -> float:
    return float(hdata.load_config().get('agent', {}).get('memory_limit_gb', 8))


def run_experiment(patch_path: str | Path,
                   seed: int = 0,
                   *,
                   checkpoint_path: str | Path | None = None,
                   timeout_s: float | None = None,
                   memory_limit_gb: float | None = None,
                   max_epochs: int | None = None,
                   subsample: float | None = None) -> ExperimentResult:
    """Run one generated patch and return validation facts.

    Never returns a test metric. Never raises for an experiment failure.
    """
    patch_file = Path(patch_path)
    report = hpatch.validate_patch(patch_file)
    if not report.ok:
        return _screen(ExperimentResult(
            ok=False, error_kind=ERROR_REJECTED, seed=seed,
            error='patch rejected before execution:\n  - '
                  + '\n  - '.join(report.reasons)))

    with tempfile.TemporaryDirectory() as tmp:
        result_json = Path(tmp) / 'result.json'
        args: List[str] = ['-m', 'harness._run_patch', str(patch_file), str(seed),
                           str(result_json)]
        if checkpoint_path is not None:
            args += ['--checkpoint', str(checkpoint_path)]
        if max_epochs is not None:
            args += ['--max_epochs', str(max_epochs)]
        if subsample is not None:
            args += ['--subsample', str(subsample)]

        run = sandbox.run_python(
            args,
            timeout_s=_timeout_seconds() if timeout_s is None else timeout_s,
            memory_limit_gb=(_memory_limit_gb() if memory_limit_gb is None
                             else memory_limit_gb),
            cwd=hdata.repo_root())

        payload: Dict[str, Any] = {}
        if result_json.exists():
            try:
                payload = json.loads(result_json.read_text(encoding='utf-8'))
            except json.JSONDecodeError:
                payload = {}

    if run.timed_out or run.memory_exceeded:
        # The ceiling breach is the story, whatever the child managed to write.
        return _screen(ExperimentResult(
            ok=False, error_kind=run.failure_kind, seed=seed,
            seconds=run.seconds, peak_memory_mb=run.peak_memory_mb,
            stdout=run.stdout,
            error=(f'{run.failure_kind} after {run.seconds:.0f}s '
                   f'(peak {run.peak_memory_mb:.0f} MB)\n{run.tail()}')))

    if not payload:
        return _screen(ExperimentResult(
            ok=False, error_kind=ERROR_CODE, seed=seed, seconds=run.seconds,
            peak_memory_mb=run.peak_memory_mb, stdout=run.stdout,
            error=('the child wrote no result; it probably died before it could.\n'
                   + run.tail())))

    if not payload.get('ok'):
        return _screen(ExperimentResult(
            ok=False, error_kind=payload.get('error_kind') or ERROR_CODE, seed=seed,
            seconds=run.seconds, peak_memory_mb=run.peak_memory_mb,
            stdout=run.stdout, error=payload.get('error') or run.tail()))

    return _screen(ExperimentResult(
        ok=True,
        val_gauc=payload.get('val_gauc'),
        val_ndcg5=payload.get('val_ndcg5'),
        val_primary=payload.get('val_primary'),
        diagnostics=payload.get('diagnostics') or {},
        checkpoint=payload.get('checkpoint'),
        seconds=run.seconds,
        seed=seed,
        stdout=run.stdout,
        peak_memory_mb=run.peak_memory_mb))


# --------------------------------------------------------------------------
# the stub -- and it can fail
# --------------------------------------------------------------------------

#: Every case the loop must handle. The last two are the nastiest and the reason
#: this stub exists at all.
STUB_CASES = ('improvement', 'no_improvement', 'regression', 'code_error',
              'timeout', 'memory_error', 'evaluator_rejection', 'nan_score',
              'canary_trip')


def make_stub_result(case: str, *, seed: int = 0, best_so_far: float = 0.6015,
                     seconds: float = 61.0) -> ExperimentResult:
    """One synthetic ``ExperimentResult`` of a named kind.

    Deterministic given *case*, *seed* and *best_so_far*, so a scripted sequence
    replays identically.
    """
    if case not in STUB_CASES:
        raise ValueError(f'unknown stub case {case!r}; choose from {STUB_CASES}')

    def scored(primary: float) -> ExperimentResult:
        gauc = primary + 0.031
        ndcg = 2 * primary - gauc
        return ExperimentResult(
            ok=True, val_gauc=gauc, val_ndcg5=ndcg, val_primary=primary,
            seconds=seconds, seed=seed, checkpoint=f'stub-checkpoint-{case}-{seed}',
            diagnostics={
                'metrics': {'val_gauc': gauc, 'val_ndcg5': ndcg,
                            'val_primary': primary},
                'fit': {'train_primary': primary + 0.042,
                        'val_primary': primary, 'gap': 0.042,
                        'epochs_run': 11, 'best_epoch': 7},
                'fields': {name: {'mean_abs_w': 0.04, 'mean_v_norm': 0.2,
                                  'n_ids': 1000}
                           for name in ('user_id', 'video_id', 'author_id',
                                        'tab', 'dur_bucket')},
                'lists': {'group_by': 'user_id', 'train_groups': 26210,
                          'mean_train_list_size': 43.5, 'valid_users': 22377,
                          'mean_valid_list_size': 5.58},
                'stub': True,
            })

    def failed(kind: str, message: str) -> ExperimentResult:
        return ExperimentResult(ok=False, error_kind=kind, seed=seed,
                                seconds=seconds, error=message,
                                diagnostics={'stub': True})

    # Routed through the same screen as a real result, so the stub is faithful by
    # construction rather than by anyone remembering to keep the two in sync. It
    # is why the canary case arrives as a failure rather than as a 0.93 score:
    # that is what the loop would really be handed.
    if case == 'improvement':
        return _screen(scored(best_so_far + 0.0100), quarantine=False)
    if case == 'no_improvement':
        return _screen(scored(best_so_far + 0.0005), quarantine=False)
    if case == 'regression':
        return _screen(scored(best_so_far - 0.0400), quarantine=False)
    if case == 'nan_score':
        return _screen(scored(float('nan')), quarantine=False)
    if case == 'canary_trip':
        return _screen(scored(0.93), quarantine=False)
    if case == 'code_error':
        return failed(ERROR_CODE,
                      'Traceback (most recent call last):\n'
                      '  File "harness/models/gen/bpr_v1.py", line 24, in loss\n'
                      '    z_pos = z[positives]\n'
                      'IndexError: index 8192 is out of bounds for axis 0')
    if case == 'timeout':
        return failed(ERROR_TIMEOUT, 'timeout after 1500s (peak 2100 MB)')
    if case == 'memory_error':
        return failed(ERROR_MEMORY, 'memory after 240s (peak 8400 MB)')
    return failed(ERROR_EVALUATOR,
                  'submission rejected: row_id 4211 out of sequence')


class StubRunner:
    """A ``run_experiment`` stand-in that replays a scripted sequence.

    Same call signature as :func:`run_experiment`, so the agent loop cannot tell
    the difference. Records what it was asked, which is what the loop's own tests
    assert against.

    The sequence may name cases or supply ready-made results. When it runs out,
    the *default* case repeats, so a loop that iterates further than the script
    anticipated keeps going instead of raising.
    """

    def __init__(self, sequence: Sequence[str | ExperimentResult],
                 *, default: str = 'no_improvement', best_so_far: float = 0.6015):
        self.sequence = list(sequence)
        self.default = default
        self.best_so_far = best_so_far
        self.calls: List[Dict[str, Any]] = []

    def __call__(self, patch_path: str | Path, seed: int = 0,
                 **kwargs: Any) -> ExperimentResult:
        index = len(self.calls)
        self.calls.append({'patch_path': str(patch_path), 'seed': seed, **kwargs})
        item = self.sequence[index] if index < len(self.sequence) else self.default
        if isinstance(item, ExperimentResult):
            return item
        result = make_stub_result(item, seed=seed, best_so_far=self.best_so_far)
        if result.usable and result.val_primary > self.best_so_far:
            self.best_so_far = result.val_primary
        # Honour checkpoint_path like the real runner does. A stub whose success
        # path cannot be exercised end to end is not doing its job: the loop
        # promotes a checkpoint on a keep, and a promotion of a path that was
        # never written is a failure the stub would otherwise hide until the
        # first real run.
        target = kwargs.get('checkpoint_path')
        if result.usable and target is not None:
            path = Path(target)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(path, V=np.zeros((2, 2), dtype=np.float32),
                     W=np.zeros(2, dtype=np.float32), b=np.float32(0.0))
            result.checkpoint = str(path)
        return result

    @property
    def call_count(self) -> int:
        return len(self.calls)


def stub_patch_source(name: str = 'stub', **config: Any) -> str:
    """A minimal valid patch, for tests that need a real file on disk."""
    return (f'"""Generated patch: {name}."""\n'
            f'CONFIG = {config!r}\n')


def new_patch_path(stem: str | None = None) -> Path:
    """A fresh path inside the generated-models directory."""
    stem = stem or f'patch_{uuid.uuid4().hex[:8]}'
    return hdata.repo_root() / 'harness' / 'models' / 'gen' / f'{stem}.py'
