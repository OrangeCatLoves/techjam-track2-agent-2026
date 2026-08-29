"""Convergence rule tests. Hand-built score sequences, exact stop points.

The rule is self-enforced, so it is only as trustworthy as these assertions:

  * strike counting under both readings of Q3
  * the three hard stops: no-improvement, 50 iterations, 6 hours
  * never stopping voluntarily before the rule fires
  * never continuing after it fires
  * a restart resuming iteration count, strike count, best score and tried-set
"""
from __future__ import annotations

import json

import pytest

from harness import convergence as conv
from harness.convergence import (ConvergedError, ConvergenceTracker, EarlyStopError,
                                 STATUS_FAILED, STATUS_OK)

EPS = 0.002


class FakeClock:
    """Injectable monotonic clock, so a six-hour test takes no time."""

    def __init__(self, t: float = 0.0):
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


def make(tmp_path, **kwargs) -> ConvergenceTracker:
    clock = kwargs.pop('clock', FakeClock())
    tracker = ConvergenceTracker.open(
        tmp_path / 'convergence.json',
        epsilon=EPS, n_consecutive=3, max_iterations=50,
        max_wall_clock_hours=6.0, comparison='per_iteration',
        allow_early_stop=False, clock=clock, **kwargs)
    tracker.start_session()
    tracker.clock = clock          # for the tests to advance
    return tracker


def feed(tracker, scores):
    """Record a sequence of validation primaries, returning the final status."""
    status = tracker.status()
    for s in scores:
        status = tracker.record_iteration(s)
    return status


# --------------------------------------------------------------------------
# strike counting
# --------------------------------------------------------------------------

def test_first_score_is_never_a_strike(tmp_path):
    t = make(tmp_path)
    status = feed(t, [0.6015])
    assert status.strikes == 0
    assert status.best_primary == pytest.approx(0.6015)
    assert not status.converged


def test_a_real_improvement_clears_the_streak(tmp_path):
    t = make(tmp_path)
    feed(t, [0.6000, 0.6005, 0.6008])       # two small gains -> two strikes
    assert t.strikes == 2
    feed(t, [0.6100])                        # +0.0092, clears
    assert t.strikes == 0
    assert not t.status().converged


def test_a_gain_exactly_at_epsilon_is_a_strike(tmp_path):
    """'improving by no more than 0.002' includes 0.002 itself."""
    t = make(tmp_path)
    feed(t, [0.6000, 0.6020])
    assert t.strikes == 1


def test_a_gain_just_over_epsilon_is_not_a_strike(tmp_path):
    t = make(tmp_path)
    feed(t, [0.6000, 0.6021])
    assert t.strikes == 0


def test_a_worse_score_is_a_strike_and_does_not_lower_the_best(tmp_path):
    t = make(tmp_path)
    feed(t, [0.6100, 0.5000])
    assert t.strikes == 1
    assert t.best_primary == pytest.approx(0.6100)
    assert t.best_iteration == 1


def test_gain_is_measured_against_the_running_best_not_the_last_score(tmp_path):
    t = make(tmp_path)
    feed(t, [0.6100, 0.5000, 0.6050])   # 0.6050 beats the previous score but not the best
    assert t.strikes == 2
    assert t.best_primary == pytest.approx(0.6100)


# --------------------------------------------------------------------------
# stop point 1: no improvement
# --------------------------------------------------------------------------

def test_three_small_gains_converge_on_the_third(tmp_path):
    t = make(tmp_path)
    assert not feed(t, [0.6000, 0.6010]).converged
    assert not feed(t, [0.6015]).converged
    status = feed(t, [0.6018])
    assert status.converged
    assert status.reason == conv.REASON_NO_IMPROVEMENT
    assert status.iteration == 4
    assert status.strikes == 3


def test_no_convergence_before_three_scored_iterations(tmp_path):
    t = make(tmp_path)
    status = feed(t, [0.60, 0.60])
    assert not status.converged
    assert status.strikes == 1


def test_a_late_breakthrough_prevents_convergence(tmp_path):
    t = make(tmp_path)
    status = feed(t, [0.6000, 0.6005, 0.6008, 0.6300])
    assert not status.converged
    assert status.strikes == 0
    assert status.best_primary == pytest.approx(0.6300)


# --------------------------------------------------------------------------
# stop point 2 and 3: the hard caps
# --------------------------------------------------------------------------

def test_fifty_iterations_stops_exactly_at_fifty(tmp_path):
    t = make(tmp_path)
    # An ever-improving run never strikes, so only the iteration cap can stop it.
    score = 0.50
    for i in range(50):
        score += 0.005
        status = t.record_iteration(score)
        assert status.converged == (i == 49), f'stopped at the wrong iteration {i + 1}'
    assert status.reason == conv.REASON_MAX_ITERATIONS
    assert status.iteration == 50
    assert status.remaining_iterations == 0


def test_six_hours_stops_the_run(tmp_path):
    clock = FakeClock()
    t = make(tmp_path, clock=clock)
    feed(t, [0.60, 0.65])
    clock.advance(5 * 3600 + 3599)
    assert not t.status().converged
    clock.advance(2)                       # crosses six hours
    status = t.status()
    assert status.converged
    assert status.reason == conv.REASON_WALL_CLOCK
    assert status.remaining_seconds == 0.0


# --------------------------------------------------------------------------
# the two behavioural properties
# --------------------------------------------------------------------------

def test_never_continues_after_convergence(tmp_path):
    t = make(tmp_path)
    feed(t, [0.6000, 0.6010, 0.6015, 0.6018])
    assert t.status().converged
    with pytest.raises(ConvergedError):
        t.record_iteration(0.9)
    assert t.iteration == 4, 'a rejected iteration must not advance the counter'


def test_never_stops_voluntarily_before_convergence(tmp_path):
    t = make(tmp_path)
    feed(t, [0.6000, 0.6010])
    assert t.should_continue()
    with pytest.raises(EarlyStopError):
        t.assert_may_stop()
    feed(t, [0.6015, 0.6018])
    assert not t.should_continue()
    t.assert_may_stop()                     # now permitted


# --------------------------------------------------------------------------
# failed iterations (Q4 default in force)
# --------------------------------------------------------------------------

def test_a_failed_iteration_burns_one_of_fifty_but_not_a_strike(tmp_path):
    t = make(tmp_path)
    feed(t, [0.6000, 0.6010])
    assert t.strikes == 1
    status = t.record_failure(error='NameError in generated patch')
    assert status.iteration == 3
    assert status.failed_iterations == 1
    assert status.strikes == 1, 'an abandoned iteration is not a non-improving one'
    assert not status.converged


def test_failures_alone_never_converge_by_no_improvement(tmp_path):
    t = make(tmp_path)
    for _ in range(5):
        status = t.record_failure(error='timeout')
    assert not status.converged
    assert status.strikes == 0
    assert status.scored_iterations == 0


def test_a_successful_iteration_must_report_a_score(tmp_path):
    t = make(tmp_path)
    with pytest.raises(ValueError):
        t.record_iteration(None, status=STATUS_OK)


# --------------------------------------------------------------------------
# restart
# --------------------------------------------------------------------------

def test_restart_resumes_iteration_strikes_best_and_tried_set(tmp_path):
    clock = FakeClock()
    t = make(tmp_path, clock=clock)
    t.mark_tried('hash-bpr-user-lists')
    feed(t, [0.6000, 0.6010])
    t.record_failure(content_hash='hash-broken-patch')
    clock.advance(1234.0)
    t.end_session()                          # process dies here

    clock2 = FakeClock(999999.0)             # a fresh process, unrelated clock origin
    resumed = ConvergenceTracker.open(tmp_path / 'convergence.json', clock=clock2)
    assert resumed.iteration == 3
    assert resumed.failed_iterations == 1
    assert resumed.strikes == 1
    assert resumed.best_primary == pytest.approx(0.6010)
    assert resumed.best_iteration == 2
    assert resumed.has_tried('hash-bpr-user-lists')
    assert resumed.has_tried('hash-broken-patch')
    assert resumed.elapsed_seconds == pytest.approx(1234.0)
    assert resumed.epsilon == EPS and resumed.n_consecutive == 3
    assert resumed.comparison == 'per_iteration'

    # And the streak continues across the restart rather than starting over.
    resumed.start_session()
    resumed.record_iteration(0.6012)
    assert resumed.strikes == 2
    status = resumed.record_iteration(0.6014)
    assert status.strikes == 3 and status.converged


def test_restart_after_convergence_stays_converged(tmp_path):
    t = make(tmp_path)
    feed(t, [0.6000, 0.6010, 0.6015, 0.6018])
    t.end_session()
    resumed = ConvergenceTracker.open(tmp_path / 'convergence.json', clock=FakeClock())
    assert resumed.status().converged
    with pytest.raises(ConvergedError):
        resumed.record_iteration(0.7)


def test_wall_clock_accumulates_across_sessions(tmp_path):
    clock = FakeClock()
    t = make(tmp_path, clock=clock)
    t.record_iteration(0.60)
    clock.advance(2 * 3600)
    t.end_session()

    clock2 = FakeClock(50.0)
    resumed = ConvergenceTracker.open(tmp_path / 'convergence.json', clock=clock2)
    resumed.start_session()
    assert resumed.elapsed_seconds == pytest.approx(2 * 3600)
    clock2.advance(4 * 3600 - 1)
    assert not resumed.status().converged
    clock2.advance(2)
    assert resumed.status().reason == conv.REASON_WALL_CLOCK


def test_state_file_is_readable_and_records_the_rule(tmp_path):
    t = make(tmp_path)
    feed(t, [0.6000, 0.6010])
    state = json.loads((tmp_path / 'convergence.json').read_text(encoding='utf-8'))
    assert state['rule'] == {'epsilon': EPS, 'n_consecutive': 3,
                             'max_iterations': 50, 'max_wall_clock_hours': 6.0,
                             'comparison': 'per_iteration',
                             'allow_early_stop': False}
    assert state['iteration'] == 2
    assert [h['status'] for h in state['history']] == [STATUS_OK, STATUS_OK]


# --------------------------------------------------------------------------
# Q3: the switchable comparison
# --------------------------------------------------------------------------

SEQUENCE = [0.6000, 0.6015, 0.6030, 0.6045]
"""Three consecutive gains of 0.0015 each: under epsilon individually, over it in
total. This is exactly the sequence the two readings of Q3 disagree about."""


def test_per_iteration_is_the_stricter_reading(tmp_path):
    t = make(tmp_path)
    status = feed(t, SEQUENCE)
    assert status.converged, 'each gain was <= epsilon, so the strict rule fires'
    assert status.reason == conv.REASON_NO_IMPROVEMENT


def test_block_reading_keeps_going_on_the_same_sequence(tmp_path):
    t = ConvergenceTracker.open(tmp_path / 'block.json', epsilon=EPS, n_consecutive=3,
                                max_iterations=50, max_wall_clock_hours=6.0,
                                comparison='block', clock=FakeClock())
    t.start_session()
    status = feed(t, SEQUENCE)
    assert not status.converged, 'the block total was 0.0045, over epsilon'
    assert status.strikes == 3, 'strikes are still reported under either reading'


def test_block_reading_fires_when_the_block_total_is_small(tmp_path):
    t = ConvergenceTracker.open(tmp_path / 'block2.json', epsilon=EPS, n_consecutive=3,
                                max_iterations=50, max_wall_clock_hours=6.0,
                                comparison='block', clock=FakeClock())
    t.start_session()
    status = feed(t, [0.6000, 0.6005, 0.6008, 0.6010])
    assert status.converged and status.reason == conv.REASON_NO_IMPROVEMENT


def test_strict_fires_whenever_block_fires(tmp_path):
    """The ordering claim that makes per_iteration the safe default."""
    sequences = [
        [0.60, 0.6005, 0.6008, 0.6010],
        [0.60, 0.60, 0.60, 0.60],
        [0.60, 0.55, 0.58, 0.59],
        [0.60, 0.6300, 0.6301, 0.6302],
    ]
    for seq in sequences:
        scored, best = [], None
        for i, s in enumerate(seq, start=1):
            scored.append(conv.ScoredIteration(i, s, best))
            best = s if best is None else max(best, s)
        strict = conv._converged_per_iteration(scored, EPS, 3)
        block = conv._converged_block(scored, EPS, 3)
        assert not block or strict, f'block fired without strict on {seq}'


def test_unknown_comparison_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        ConvergenceTracker(tmp_path / 'x.json', comparison='vibes')


# --------------------------------------------------------------------------
# the baseline-reproduction iteration must not cost a strike
# --------------------------------------------------------------------------

def test_baseline_reproduction_does_not_burn_a_strike(tmp_path):
    """CLAUDE.md 6.4 requires the agent to reproduce the baseline itself.

    That iteration necessarily scores ~0.6015, which is not an improvement on
    anything. If the tracker treated it as a non-improving iteration, the agent
    would start on strike one having done nothing wrong -- a third of the strike
    budget lost to an off-by-one, discovered during the scored run.

    It does not, because the first scored iteration has no prior best: its gain is
    infinite by definition.
    """
    t = make(tmp_path)
    status = feed(t, [0.6015])
    assert status.strikes == 0
    assert status.best_primary == pytest.approx(0.6015)
    assert status.iteration == 1, 'it does still consume one of the 50'


def test_seeding_the_initial_best_would_burn_a_strike(tmp_path):
    """The trap, made explicit so nobody wires it up by accident.

    Seeding `initial_best` with the published baseline and then recording the
    agent's own reproduction of it produces a gain of exactly zero -- strike one,
    before any experiment has been proposed. The agent's tracker is therefore
    never seeded; it learns the baseline by reproducing it.
    """
    seeded = ConvergenceTracker.open(tmp_path / 'seeded.json', epsilon=EPS,
                                     n_consecutive=3, initial_best=0.6015,
                                     clock=FakeClock())
    seeded.start_session()
    assert seeded.record_iteration(0.6015).strikes == 1

    unseeded = make(tmp_path)
    assert unseeded.record_iteration(0.6015).strikes == 0


def test_the_reproduction_iteration_counts_toward_the_fifty(tmp_path):
    """It is a real experiment cycle -- code written, run, scored -- so it counts.

    Only the strike question was ambiguous; the cap question is not.
    """
    t = make(tmp_path)
    feed(t, [0.6015])
    assert t.status().remaining_iterations == 49
