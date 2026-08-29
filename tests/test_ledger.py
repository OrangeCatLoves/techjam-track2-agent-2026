"""The ledger: what was tried, and which checkpoint would be submitted.

The load-bearing behaviour is **promotion and rollback**. "Keep or reject" is only
meaningful if rejecting genuinely leaves the previous winner in place, and if a
restart finds the same winner it left.

Rollback is free here by construction: model state lives in checkpoint files, never
in a mutable global, so rejecting is simply declining to move the best-pointer.
These tests pin that it is actually free rather than merely intended to be.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from harness import guards
from harness.ledger import DECISION_FAILED, DECISION_KEEP, DECISION_REJECT, Ledger


@pytest.fixture()
def ledger(tmp_path):
    return Ledger(tmp_path / 'run-test')


def checkpoint(tmp_path, name='model.npz', value=1.0):
    path = tmp_path / name
    np.savez(path, V=np.full((4, 2), value, dtype=np.float32),
             W=np.zeros(4, dtype=np.float32), b=np.float32(value))
    return path


# --------------------------------------------------------------------------
# layout and records
# --------------------------------------------------------------------------

def test_the_run_directory_is_laid_out(ledger):
    for sub in ('patches', 'checkpoints', 'best'):
        assert (ledger.run_dir / sub).is_dir()


def test_records_append_and_survive_reopening(ledger, tmp_path):
    ledger.record({'iteration': 1, 'decision': DECISION_KEEP})
    ledger.record({'iteration': 2, 'decision': DECISION_REJECT})
    reopened = Ledger(tmp_path / 'run-test')
    assert [r['iteration'] for r in reopened.records()] == [1, 2]


def test_a_record_carrying_a_test_metric_is_refused(ledger):
    with pytest.raises(guards.LeakageError):
        ledger.record({'iteration': 1, 'metrics': {'test': {'primary': 0.5946}}})
    assert ledger.records() == [], 'nothing may be written after a refusal'


def test_an_unknown_decision_is_refused(ledger):
    with pytest.raises(ValueError):
        ledger.record({'iteration': 1, 'decision': 'maybe'})


def test_a_torn_final_line_does_not_break_reading(ledger):
    ledger.record({'iteration': 1, 'decision': DECISION_KEEP})
    with open(ledger.ledger_path, 'a', encoding='utf-8') as fh:
        fh.write('{"iteration": 2, "decis')          # killed mid-write
    assert [r['iteration'] for r in ledger.records()] == [1]


def test_history_is_append_only(ledger):
    """A judge should see the failures, not a tidied record of the successes."""
    ledger.record({'iteration': 1, 'decision': DECISION_FAILED,
                   'errors': ['NameError in generated patch']})
    ledger.record({'iteration': 2, 'decision': DECISION_KEEP})
    assert len(ledger.records()) == 2
    assert ledger.records()[0]['decision'] == DECISION_FAILED


# --------------------------------------------------------------------------
# promotion and rollback
# --------------------------------------------------------------------------

def test_nothing_is_best_before_the_first_keep(ledger):
    assert ledger.best() is None
    assert ledger.would_improve(0.5), 'anything beats nothing'


def test_promotion_copies_the_checkpoint_rather_than_pointing_at_it(ledger, tmp_path):
    source = checkpoint(tmp_path, value=7.0)
    best = ledger.promote(3, source, 0.6120)
    source.unlink()                              # per-iteration cleanup
    assert best.iteration == 3
    restored = np.load(best.checkpoint)
    assert restored['V'][0][0] == pytest.approx(7.0), (
        'the winner must survive the loss of the iteration checkpoint')


def test_rejecting_leaves_the_previous_winner_in_place(ledger, tmp_path):
    ledger.promote(1, checkpoint(tmp_path, 'a.npz', 1.0), 0.6120)
    ledger.record({'iteration': 2, 'decision': DECISION_REJECT,
                   'metrics': {'val_primary': 0.6001}})
    assert ledger.best().iteration == 1
    assert ledger.best().val_primary == pytest.approx(0.6120)


def test_would_improve_refuses_none_and_nan(ledger, tmp_path):
    ledger.promote(1, checkpoint(tmp_path), 0.6120)
    assert ledger.would_improve(0.6200)
    assert not ledger.would_improve(0.6100)
    assert not ledger.would_improve(None)
    assert not ledger.would_improve(float('nan')), (
        'NaN compared with > is False and reads as an ordinary non-improvement')


def test_promotion_of_a_missing_checkpoint_fails_loudly(ledger, tmp_path):
    with pytest.raises(FileNotFoundError):
        ledger.promote(1, tmp_path / 'never-written.npz', 0.7)


def test_a_flagged_winner_is_recorded_as_flagged(ledger, tmp_path):
    """A review flag follows the checkpoint, so it cannot be lost at submission."""
    ledger.promote(1, checkpoint(tmp_path), 0.7210, flagged_for_review=True)
    blob = json.loads(ledger.best_path.read_text(encoding='utf-8'))
    assert blob['flagged_for_review'] is True
    assert Ledger(ledger.run_dir).best().flagged_for_review is True


def test_the_best_pointer_survives_a_restart(ledger, tmp_path):
    ledger.promote(4, checkpoint(tmp_path), 0.6180, metrics={'val_gauc': 0.68})
    resumed = Ledger(ledger.run_dir)
    assert resumed.best().iteration == 4
    assert resumed.best().val_primary == pytest.approx(0.6180)
    assert resumed.best().metrics['val_gauc'] == pytest.approx(0.68)


# --------------------------------------------------------------------------
# generated code hygiene
# --------------------------------------------------------------------------

def test_a_used_patch_is_archived_out_of_the_generated_directory(ledger):
    gen = ledger.new_patch_path(5)
    gen.parent.mkdir(parents=True, exist_ok=True)
    gen.write_text('CONFIG = {}\n', encoding='utf-8')
    archived = ledger.archive_patch(5, gen)
    assert archived is not None and archived.exists()
    assert not gen.exists(), 'gen/ must not accumulate code nobody is running'
    assert 'CONFIG' in archived.read_text(encoding='utf-8')


def test_archiving_a_missing_patch_is_not_an_error(ledger):
    assert ledger.archive_patch(9, ledger.new_patch_path(9)) is None


def test_clean_gen_removes_stale_generated_files(ledger):
    gen_dir = ledger.gen_dir()
    gen_dir.mkdir(parents=True, exist_ok=True)
    for name in ('stale_a.py', 'stale_b.py'):
        (gen_dir / name).write_text('CONFIG = {}\n', encoding='utf-8')
    assert ledger.clean_gen() >= 2
    assert not list(gen_dir.glob('*.py'))


# --------------------------------------------------------------------------
# audit and summary
# --------------------------------------------------------------------------

def test_kept_results_feed_the_retrospective_audit(ledger):
    ledger.record({'iteration': 1, 'decision': DECISION_KEEP, 'val_primary': 0.6120})
    ledger.record({'iteration': 2, 'decision': DECISION_REJECT, 'val_primary': 0.7500})
    ledger.record({'iteration': 3, 'decision': DECISION_KEEP, 'val_primary': 0.7210})
    flagged = ledger.audit()
    assert [r['iteration'] for r in flagged] == [3], (
        'only KEPT results are auditable; a rejected one was never banked')


def test_summary_counts_every_decision(ledger, tmp_path):
    ledger.record({'iteration': 1, 'decision': DECISION_KEEP})
    ledger.record({'iteration': 2, 'decision': DECISION_REJECT})
    ledger.record({'iteration': 3, 'decision': DECISION_FAILED})
    ledger.promote(1, checkpoint(tmp_path), 0.6120)
    summary = ledger.summary()
    assert (summary['kept'], summary['rejected'], summary['failed']) == (1, 1, 1)
    assert summary['best_iteration'] == 1
    guards.assert_record_clean(summary, where='ledger summary')


def test_the_convergence_state_lives_in_the_run_directory(ledger):
    """One run, one place. The tracker owns the contents; the ledger owns where."""
    assert ledger.convergence_path.parent == ledger.run_dir
