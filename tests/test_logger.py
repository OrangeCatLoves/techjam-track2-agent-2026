"""The two run logs, and the resource report.

``log.md`` is a graded deliverable: it is the evidence that the agent reasoned
rather than executed a queue. ``log.jsonl`` is what the resource table is built
from. Neither may ever carry a hidden-test metric, and both must survive a
restart without duplicating or dropping anything.
"""
from __future__ import annotations

import json

import pytest

from harness import guards
from harness.logger import (EVENT_INTERVENTION, EVENT_RECOVERY, EVENT_RESTART,
                            RunLogger, write_resource_table)

CHINESE = '✓ 格式与对齐校验通过'


@pytest.fixture()
def logger(tmp_path):
    return RunLogger(tmp_path / 'run-test')


def iteration(n=1, primary=0.6120, decision='keep', **extra):
    return {'iteration': n, 'hypothesis': 'Pointwise logloss is misaligned with '
                                          'within-user ranking metrics.',
            'target_stage': 'objective', 'patch_kind': 'new_loss_function',
            'metrics': {'val_gauc': primary + 0.03, 'val_ndcg5': primary - 0.03,
                        'val_primary': primary},
            'decision': decision, 'reason': f'val_primary {primary:.4f}',
            'wall_clock_s': 63.0, 'tokens': {'input': 8412, 'output': 1903},
            **extra}


# --------------------------------------------------------------------------
# screening
# --------------------------------------------------------------------------

def test_a_test_metric_never_reaches_either_sink(logger):
    with pytest.raises(guards.LeakageError):
        logger.log_iteration({'iteration': 1,
                              'metrics': {'test': {'primary': 0.5946}}})
    assert not logger.jsonl_path.exists()


def test_a_test_metric_in_a_free_text_field_is_caught(logger):
    with pytest.raises(guards.LeakageError):
        logger.log_iteration({'iteration': 1,
                              'hypothesis': 'test GAUC 0.6610 primary 0.5946'})


def test_the_markdown_is_screened_as_text_too(logger):
    """Belt and braces: the record passes, and the rendered page is checked again."""
    logger.log_iteration(iteration())
    guards.assert_no_test_metrics(logger.markdown_path.read_text(encoding='utf-8'),
                                  where='log.md')


# --------------------------------------------------------------------------
# encoding (D10)
# --------------------------------------------------------------------------

def test_non_ascii_content_round_trips_through_both_sinks(logger):
    """The starter kit is bilingual and generated code prints anything.

    A cp1252 sink here turns a successful experiment into a crash report.
    """
    logger.log_iteration(iteration(reason=CHINESE))
    restored = json.loads(logger.jsonl_path.read_text(encoding='utf-8').strip())
    assert restored['reason'] == CHINESE
    assert CHINESE in logger.markdown_path.read_text(encoding='utf-8')


# --------------------------------------------------------------------------
# restart safety
# --------------------------------------------------------------------------

def test_the_markdown_is_regenerated_not_appended(logger, tmp_path):
    """A restart must not produce a duplicated tail.

    log.md is rebuilt from log.jsonl every time, so it cannot drift from the
    machine-readable record either.
    """
    logger.log_iteration(iteration(1))
    logger.log_iteration(iteration(2, 0.6180))
    resumed = RunLogger(tmp_path / 'run-test')
    resumed.log_iteration(iteration(3, 0.6200))
    text = resumed.markdown_path.read_text(encoding='utf-8')
    assert text.count('### Iteration 1') == 1
    assert text.count('### Iteration 3') == 1
    assert len(resumed.iterations()) == 3


def test_an_empty_run_still_renders(logger):
    assert 'No iterations recorded yet' in logger.markdown()


# --------------------------------------------------------------------------
# what a judge reads
# --------------------------------------------------------------------------

def test_the_markdown_shows_the_hypothesis_and_the_decision(logger):
    logger.log_iteration(iteration())
    text = logger.markdown_path.read_text(encoding='utf-8')
    assert 'Hypothesis' in text and 'within-user ranking' in text
    assert 'objective' in text
    assert 'KEPT' in text


def test_failures_are_shown_not_hidden(logger):
    logger.log_iteration(iteration(2, decision='failed',
                                   errors=['IndexError: index 8192 out of bounds']))
    text = logger.markdown_path.read_text(encoding='utf-8')
    assert 'FAILED' in text and 'IndexError' in text


def test_a_flagged_result_is_called_out_in_the_log(logger):
    logger.log_iteration(iteration(1, 0.7210, flagged_for_review=True))
    text = logger.markdown_path.read_text(encoding='utf-8')
    assert 'Flagged for review' in text
    assert 'human' in text.lower()


def test_the_best_score_is_summarised(logger):
    logger.log_iteration(iteration(1, 0.6120))
    logger.log_iteration(iteration(2, 0.6350))
    logger.log_iteration(iteration(3, 0.6200))
    assert '0.6350' in logger.markdown_path.read_text(encoding='utf-8')


# --------------------------------------------------------------------------
# events and the resource report
# --------------------------------------------------------------------------

def test_events_are_kept_apart_from_iterations(logger):
    logger.log_iteration(iteration())
    logger.log_event(EVENT_RESTART, 'process killed at iteration 4, resumed')
    assert len(logger.iterations()) == 1
    assert len(logger.events()) == 1


def test_restarts_and_interventions_are_counted_separately(logger):
    """The definition in force distinguishes them, and the distinction only
    survives if they are recorded apart.

    Restarting a crashed process is operational recovery. A human changing the
    agent's instructions, objective or search space is an intervention, and that
    is the number Impact is scored on.
    """
    logger.log_iteration(iteration())
    logger.log_event(EVENT_RESTART, 'resumed from the ledger')
    logger.log_event(EVENT_RESTART, 'resumed again')
    logger.log_event(EVENT_RECOVERY, 'timeout retried at 30% subsample')
    logger.log_event(EVENT_INTERVENTION, 'a human widened the search space')

    report = logger.resource_report()
    assert report['operational_restarts'] == 2
    assert report['recovery_events'] == 1
    assert report['manual_interventions'] == 1
    assert 'instructions, objective or search space' in report['intervention_definition']


def test_the_resource_report_totals_tokens_and_time(logger):
    logger.log_iteration(iteration(1))
    logger.log_iteration(iteration(2))
    report = logger.resource_report()
    assert report['tokens']['input'] == 8412 * 2
    assert report['tokens']['total'] == (8412 + 1903) * 2
    assert report['wall_clock_hours'] == pytest.approx(126.0 / 3600, abs=1e-6)
    assert report['iterations_used'] == 2
    guards.assert_record_clean(report, where='resource report')


def test_a_zero_intervention_run_reports_zero(logger):
    logger.log_iteration(iteration())
    assert logger.resource_report()['manual_interventions'] == 0


def test_the_resource_table_renders_for_the_readme(logger, tmp_path):
    logger.log_iteration(iteration())
    logger.log_event(EVENT_RESTART, 'resumed')
    path = write_resource_table(logger.resource_report(), tmp_path / 'resources.md')
    text = path.read_text(encoding='utf-8')
    assert 'Manual interventions' in text
    assert 'Operational restarts (not interventions)' in text
    assert 'GPU-hours' in text
    guards.assert_no_test_metrics(text, where='resource table')
