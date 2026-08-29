"""The harness/agent boundary, the sandbox, and the patch validator.

The boundary is the whole point of M2a: it is the only thing that crosses between
the fixed harness and the LLM-written half, so its shape is frozen in
``docs/M2_CONTRACT.md`` and pinned here.

Two properties matter more than the rest:

* **``run_experiment`` never raises for an experiment failure.** A crash, a
  timeout, a memory breach, a rejected patch and a tripped canary are all returned
  values. If any of them raised, the agent loop would die on its first bad
  generated patch, which is a certainty rather than a risk.
* **The stub can fail.** A stub returning only plausible numbers means the
  recovery path is written blind and first runs for real during the scored run.
"""
from __future__ import annotations

import math

import pytest

from harness import experiment as X
from harness import guards
from harness import patch as P
from harness import sandbox

# --------------------------------------------------------------------------
# the patch validator
# --------------------------------------------------------------------------

GEN = 'harness/models/gen/candidate.py'


def test_a_plain_patch_passes():
    report = P.validate_source('import numpy as np\nCONFIG = {"lr": 0.002}\n', GEN)
    assert report.ok, report.reasons
    assert report.imports == ['numpy']


@pytest.mark.parametrize('path', [
    'harness/data.py', 'harness/guards.py', 'harness/evaluate.py',
    'harness/submit.py', 'harness/convergence.py', 'harness/losses.py',
    'harness/models/runners.py', 'configs/base.yaml', 'starter/baseline.py',
    'starter/evaluate.py', 'tests/test_guards.py', 'scripts/verify_setup.py',
])
def test_protected_paths_are_rejected(path):
    """Generated code that can edit the config can edit its way out of every guard."""
    assert P.is_protected(path)
    assert not P.validate_source('CONFIG = {}\n', path).ok


def test_the_generated_directories_are_writable_and_unprotected():
    for path in ('harness/models/gen/x.py', 'harness/features/gen/y.py'):
        assert P.is_writable(path) and not P.is_protected(path)


def test_a_patch_outside_the_writable_paths_is_rejected():
    assert not P.validate_source('CONFIG = {}\n', 'harness/models/sneaky.py').ok


@pytest.mark.parametrize('source,why', [
    ('import os\nCONFIG={}', 'filesystem access'),
    ('import subprocess\nCONFIG={}', 'process spawning'),
    ('import socket\nCONFIG={}', 'network'),
    ('import pickle\nCONFIG={}', 'deserialisation'),
    ('import torch\nCONFIG={}', 'not on the allowlist'),
    ('from . import sibling\nCONFIG={}', 'relative import'),
])
def test_forbidden_imports_are_rejected(source, why):
    assert not P.validate_source(source, GEN).ok, why


@pytest.mark.parametrize('source', [
    'from baseline import run_fm\nCONFIG={}',
    'import baseline\nCONFIG={}',
    'from data import load\nCONFIG={}',
    'import starter\nCONFIG={}',
    'from evaluate import evaluate\nCONFIG={}',
])
def test_reaching_the_organisers_modules_directly_is_rejected(source):
    """``starter.baseline.run_fm`` computes a hidden-test metric, and
    ``starter.data.load`` returns test rows with their labels attached. Both are
    reached through the harness or not at all."""
    report = P.validate_source(source, GEN)
    assert not report.ok


@pytest.mark.parametrize('source', [
    'CONFIG={}\nx = eval("1+1")',
    'CONFIG={}\nexec("y = 2")',
    'CONFIG={}\nm = __import__("os")',
    'CONFIG={}\nf = open("data.csv")',
    'CONFIG={}\ng = globals()',
    'CONFIG={}\nc = compile("1", "<s>", "eval")',
])
def test_escape_hatches_are_rejected(source):
    """A string search would miss ``__import__('o' + 's')``. The AST does not."""
    assert not P.validate_source(source, GEN).ok


def test_the_harness_loss_registry_stays_importable():
    """A patch must be able to register an objective; that is its whole job."""
    source = ('from harness.losses import register_loss\n'
              '@register_loss("bpr_v1")\n'
              'def bpr(z, y, groups):\n'
              '    return 0.0, z * 0\n'
              'CONFIG = {"loss": "bpr_v1"}\n')
    assert P.validate_source(source, GEN).ok


def test_a_syntax_error_is_a_rejection_not_a_crash():
    report = P.validate_source('def f(:\n', GEN)
    assert not report.ok and 'syntax error' in report.reasons[0]


def test_a_rejected_patch_never_reaches_disk(tmp_path):
    target = tmp_path / 'gen' / 'bad.py'
    with pytest.raises(P.PatchRejected):
        P.write_patch('import os\nCONFIG={}\n', target)
    assert not target.exists(), (
        'a rejected patch on disk could be imported by a later run that skipped '
        'the check')


# --------------------------------------------------------------------------
# the sandbox
# --------------------------------------------------------------------------

def test_sandbox_runs_and_returns_output():
    run = sandbox.run_python(['-c', 'print("hello from the child")'], timeout_s=60)
    assert run.ok and run.returncode == 0
    assert 'hello from the child' in run.stdout
    assert run.failure_kind is None


def test_sandbox_reports_a_crash_as_a_result():
    run = sandbox.run_python(['-c', 'raise SystemExit(3)'], timeout_s=60)
    assert not run.ok and run.returncode == 3
    assert run.failure_kind == 'code'


def test_sandbox_kills_an_infinite_loop():
    run = sandbox.run_python(['-c', 'while True: pass'], timeout_s=2)
    assert run.timed_out and run.failure_kind == 'timeout'
    assert run.seconds < 30, 'the kill must be prompt, not eventual'


def test_sandbox_kills_a_memory_hog():
    run = sandbox.run_python(
        ['-c', 'x = bytearray()\n'
               'while True: x.extend(bytearray(50_000_000))'],
        timeout_s=120, memory_limit_gb=0.4)
    assert run.memory_exceeded and run.failure_kind == 'memory'


def test_sandbox_survives_a_child_that_floods_stdout():
    """Output goes to files, not pipes. A pipe would deadlock here and look
    exactly like an infinite loop."""
    run = sandbox.run_python(
        ['-c', 'print("x" * 200)\n' * 1 + 'for _ in range(50000): print("y" * 200)'],
        timeout_s=120)
    assert run.ok, 'the child filled far more than a pipe buffer and must still exit'
    assert len(run.stdout) > 1_000_000


def test_sandbox_filters_child_output():
    """Generated code can print anything, including a test metric it computed."""
    run = sandbox.run_python(
        ['-c', 'print("test GAUC 0.6610 | primary 0.5946")'], timeout_s=60)
    assert run.redacted_lines >= 1
    assert '0.5946' not in run.stdout
    guards.assert_no_test_metrics(run.stdout, where='sandbox stdout')


# --------------------------------------------------------------------------
# the result type
# --------------------------------------------------------------------------

def test_a_nan_score_is_not_usable():
    """The nastiest case. ``nan > best`` is False, so an unguarded NaN looks like
    an ordinary non-improvement and hides a broken objective."""
    result = X.ExperimentResult(ok=True, val_primary=float('nan'))
    assert result.ok and not result.usable
    assert not (result.val_primary > 0.6015)   # the trap, made explicit


def test_a_missing_score_is_not_usable():
    assert not X.ExperimentResult(ok=True, val_primary=None).usable


def test_a_failed_result_is_not_usable():
    assert not X.ExperimentResult(ok=False, val_primary=0.9, error_kind='code').usable


def test_a_real_score_is_usable():
    assert X.ExperimentResult(ok=True, val_primary=0.6015).usable


# --------------------------------------------------------------------------
# the stub -- and it can fail
# --------------------------------------------------------------------------

def test_every_contract_case_is_implemented():
    assert set(X.STUB_CASES) == {
        'improvement', 'no_improvement', 'regression', 'code_error', 'timeout',
        'memory_error', 'evaluator_rejection', 'nan_score', 'canary_trip'}


@pytest.mark.parametrize('case', X.STUB_CASES)
def test_stub_cases_are_screened_like_real_results(case):
    result = X.make_stub_result(case)
    guards.assert_record_clean(result.diagnostics, where=f'stub {case}')
    if result.error_kind is not None:
        assert result.error_kind in X.ERROR_KINDS
        assert not result.ok and result.error


@pytest.mark.parametrize('case,ok,usable', [
    ('improvement', True, True),
    ('no_improvement', True, True),
    ('regression', True, True),
    ('nan_score', True, False),
    ('code_error', False, False),
    ('timeout', False, False),
    ('memory_error', False, False),
    ('evaluator_rejection', False, False),
    ('canary_trip', False, False),
])
def test_stub_case_shapes(case, ok, usable):
    result = X.make_stub_result(case)
    assert result.ok is ok
    assert result.usable is usable


def test_the_stub_canary_case_arrives_already_converted():
    """A 0.93 score never reaches the loop, because the screen catches it first.

    The stub routes through the same screen as a real result, so it cannot drift
    out of step with what ``run_experiment`` actually returns.
    """
    result = X.make_stub_result('canary_trip')
    assert result.error_kind == X.ERROR_CANARY
    assert result.error_kind in X.HARD_FAILURES, 'never repair a tripped canary'
    assert result.val_primary is None


def test_improvement_and_no_improvement_straddle_epsilon():
    """The stub must be able to produce both sides of the convergence rule."""
    best = 0.6015
    better = X.make_stub_result('improvement', best_so_far=best).val_primary
    barely = X.make_stub_result('no_improvement', best_so_far=best).val_primary
    assert better - best > 0.002
    assert 0 < barely - best <= 0.002


def test_stub_runner_replays_a_sequence_and_records_calls():
    runner = X.StubRunner(['improvement', 'code_error', 'no_improvement'])
    kinds = [runner('patch.py', seed=i).error_kind for i in range(3)]
    assert kinds == [None, X.ERROR_CODE, None]
    assert runner.call_count == 3
    assert [c['seed'] for c in runner.calls] == [0, 1, 2]


def test_stub_runner_keeps_going_past_its_script():
    """A loop that iterates further than the script anticipated must not crash."""
    runner = X.StubRunner(['improvement'], default='no_improvement')
    assert runner('p.py').usable
    for _ in range(5):
        assert runner('p.py').ok
    assert runner.call_count == 6


def test_stub_runner_tracks_the_running_best():
    runner = X.StubRunner(['improvement', 'improvement'])
    first = runner('p.py').val_primary
    second = runner('p.py').val_primary
    assert second > first, 'each improvement must build on the last'


def test_unknown_stub_case_is_rejected():
    with pytest.raises(ValueError):
        X.make_stub_result('everything_works_perfectly')


# --------------------------------------------------------------------------
# run_experiment, against real patches
# --------------------------------------------------------------------------

@pytest.fixture()
def gen_patch(request):
    """Write a patch into the one directory generated code is allowed to live in.

    Not ``tmp_path``: the validator rejects any target outside
    ``agent.writable_paths``, and that rejection is itself under test elsewhere in
    this file. Cleaned up afterwards so the tree stays as the agent will find it.
    """
    written = []

    def write(source: str, stem: str | None = None):
        target = X.new_patch_path(stem or f'test_{abs(hash(request.node.name)):x}')
        written.append(target)
        return P.write_patch(source, target)

    yield write
    for target in written:
        target.unlink(missing_ok=True)


@pytest.mark.data
def test_a_rejected_patch_returns_rather_than_raises(tmp_path):
    """And nothing is executed: no subprocess, no data load."""
    bad = tmp_path / 'bad.py'
    bad.write_text('import os\nCONFIG={}\n', encoding='utf-8')
    result = X.run_experiment(bad, seed=0)
    assert not result.ok
    assert result.error_kind == X.ERROR_REJECTED
    assert result.seconds == 0.0
    assert 'rejected' in result.error


@pytest.mark.slow
def test_generated_code_that_raises_becomes_a_result(gen_patch):
    patch_file = gen_patch(
        'CONFIG = {"max_epochs": 1}\n'
        'raise ValueError("deliberate failure inside a patch")\n')
    result = X.run_experiment(patch_file, seed=0, timeout_s=600)
    assert not result.ok and result.error_kind == X.ERROR_CODE
    assert 'deliberate failure inside a patch' in result.error
    assert not result.usable


@pytest.mark.slow
def test_a_slow_patch_times_out_and_is_reported(gen_patch):
    patch_file = gen_patch(X.stub_patch_source('slow', max_epochs=40))
    result = X.run_experiment(patch_file, seed=0, timeout_s=3)
    assert not result.ok and result.error_kind == X.ERROR_TIMEOUT


@pytest.mark.slow
def test_a_greedy_patch_breaches_the_memory_ceiling(gen_patch):
    patch_file = gen_patch(X.stub_patch_source('big', max_epochs=1))
    result = X.run_experiment(patch_file, seed=0, timeout_s=600,
                              memory_limit_gb=0.05)
    assert not result.ok and result.error_kind == X.ERROR_MEMORY


@pytest.mark.slow
def test_a_working_patch_returns_the_contract_shape(gen_patch):
    patch_file = gen_patch(
        X.stub_patch_source('reference', max_epochs=1, patience=1))
    result = X.run_experiment(patch_file, seed=0, timeout_s=900)

    assert result.ok and result.usable
    assert math.isfinite(result.val_primary)
    assert 0.4 < result.val_primary < 0.8
    assert set(result.diagnostics) >= {'metrics', 'fit', 'fields'}
    guards.assert_record_clean(result.as_dict(), where='experiment result')
    assert result.peak_memory_mb > 0


@pytest.mark.slow
def test_a_patch_can_supply_its_own_objective(gen_patch):
    """The end-to-end proof that a generated loss reaches the trainer."""
    patch_file = gen_patch(
        'from harness.losses import register_loss, pointwise_logloss\n'
        '\n'
        '@register_loss("half_step_v1")\n'
        'def half_step(z, y, groups):\n'
        '    loss, grad = pointwise_logloss(z, y, groups)\n'
        '    return loss, grad * 0.5\n'
        '\n'
        'CONFIG = {"loss": "half_step_v1", "max_epochs": 1, "patience": 1}\n')
    result = X.run_experiment(patch_file, seed=0, timeout_s=900)
    assert result.ok, result.error
    assert result.usable
