"""The agent's inspection tool.

Two things are being defended here, and they pull in opposite directions.

**It must not reach the test split.** ``harness/data.py`` removed the labels; the
route this closes is aggregate statistics over test *features*, which are still
information about the hidden set. Every kind is checked, not just the obvious ones.

**It must be broad enough to be an instrument rather than a dashboard.** If the
tool only answers questions whose answers we already decided were interesting, the
agent is reading our analysis rather than doing its own, and Innovation is 20% of
the grade. So the queries are primitives, and this file asserts they compose into
findings nobody hardcoded.
"""
from __future__ import annotations

import numpy as np
import pytest

from harness import analyse as A
from harness import data as hdata
from harness import guards

pytestmark = pytest.mark.data


@pytest.fixture(scope='module')
def scores(splits):
    return np.random.default_rng(0).random(len(splits['valid']))


# --------------------------------------------------------------------------
# the refusal
# --------------------------------------------------------------------------

@pytest.mark.parametrize('kind', sorted(A.QUERIES))
def test_no_question_may_be_asked_of_the_test_split(kind, splits, scores):
    """Aggregate statistics over the hidden set are still information about it."""
    with pytest.raises(hdata.TestLabelAccessError):
        A.analyse(kind, 'test', splits=splits, scores=scores,
                  other_scores=scores, column='duration_ms')


def test_the_refusal_names_test_as_unavailable():
    assert 'test' not in A.ANALYSABLE_SPLITS
    assert A.capabilities()['refused'] == ['test']


def test_an_unknown_question_is_rejected(splits):
    with pytest.raises(A.AnalysisError):
        A.analyse('read_the_test_labels_please', 'valid', splits=splits)


def test_an_unknown_column_is_rejected(splits):
    with pytest.raises(A.AnalysisError):
        A.analyse('distribution', 'valid', splits=splits, column='play_time_ms')


def test_denied_columns_are_not_reachable(splits):
    """The deny-list columns are not merely unlisted; the loader never read them."""
    for column in guards.denied_columns():
        assert column not in A.COLUMNS + A.DERIVED
        with pytest.raises(A.AnalysisError):
            A.analyse('distribution', 'valid', splits=splits, column=column)


# --------------------------------------------------------------------------
# self-discovery
# --------------------------------------------------------------------------

def test_the_agent_can_enumerate_its_own_instrument():
    """Autonomy: the agent discovers what it can ask rather than being told."""
    caps = A.capabilities()
    assert set(caps['kinds']) == set(A.QUERIES)
    assert caps['splits'] == list(A.ANALYSABLE_SPLITS)
    for description in caps['kinds'].values():
        assert description and isinstance(description, str)


def test_capability_descriptions_say_what_is_measured_not_what_to_conclude():
    """A description that states a finding turns the tool into a dashboard."""
    forbidden = ('should', 'best', 'better than', 'you will find', 'proves')
    for kind, description in A.CAPABILITIES.items():
        lowered = description.lower()
        for word in forbidden:
            assert word not in lowered, f'{kind} description editorialises: {word!r}'


def test_every_result_carries_the_question_that_produced_it(splits):
    result = A.analyse('rate_by_bucket', 'valid', splits=splits,
                       column='duration_ms', bins=4)
    assert result.kind == 'rate_by_bucket'
    assert result.split == 'valid'
    assert result.question == {'column': 'duration_ms', 'bins': 4, 'min_rows': 50}


def test_results_are_clean_and_serialisable(splits, scores):
    for kind in ('rate_by_bucket', 'list_size_profile', 'temporal_drift',
                 'cold_key_rate'):
        result = A.analyse(kind, 'valid', splits=splits, column='duration_ms')
        guards.assert_record_clean(result.as_dict(), where=f'analyse {kind}')
        assert result.to_markdown()


# --------------------------------------------------------------------------
# the questions actually answer things
# --------------------------------------------------------------------------

def test_list_size_profile_exposes_the_train_evaluation_mismatch(splits):
    """The finding must be *discoverable*, not hardcoded anywhere the agent reads.

    Measured: train grouped by user_id averages ~43.5 rows against an evaluation
    list of ~5.6 -- while train grouped by (user_id, date) averages ~5.8, which is
    close to the evaluation length. The corpus previously asserted the opposite,
    from a median quoted as a mean. This test pins the shape of the fact, not the
    conclusion the agent should draw from it.
    """
    train = {r['grouping']: r for r in
             A.analyse('list_size_profile', 'train', splits=splits).rows}
    valid = {r['grouping']: r for r in
             A.analyse('list_size_profile', 'valid', splits=splits).rows}

    assert train['user_id']['mean_size'] > 7 * valid['user_id']['mean_size']
    assert train['user_id+date']['mean_size'] == pytest.approx(
        valid['user_id']['mean_size'], abs=1.0), (
        'the option the corpus dismissed is the one that matches')


def test_rate_by_bucket_finds_real_structure(splits):
    """Duration is not flat against the label; a constant column would be."""
    rows = A.analyse('rate_by_bucket', 'valid', splits=splits,
                     column='duration_ms', bins=6).rows
    rates = [r['long_view_rate'] for r in rows]
    assert len(rows) == 6
    assert max(rates) - min(rates) > 0.02
    assert all(0.0 <= r <= 1.0 for r in rates)


def test_rate_by_bucket_drops_thin_buckets(splits):
    rows = A.analyse('rate_by_bucket', 'valid', splits=splits,
                     column='user_id', min_rows=500).rows
    assert all(r['rows'] >= 500 for r in rows)


def test_temporal_drift_reports_one_row_per_date(splits):
    rows = A.analyse('temporal_drift', 'valid', splits=splits).rows
    assert [r['date'] for r in rows] == sorted(r['date'] for r in rows)
    assert len(rows) == 7, 'the validation window is seven days'
    assert all(r['rows'] > 0 and r['users'] > 0 for r in rows)


def test_cold_key_rate_is_zero_on_train_itself(splits):
    """A sanity check on the check: train cannot be cold against train."""
    rows = {r['field']: r for r in
            A.analyse('cold_key_rate', 'train', splits=splits).rows}
    assert rows['user_id']['rate'] == 0.0
    assert rows['video_id']['rate'] == 0.0


def test_cold_key_rate_is_small_but_nonzero_on_validation(splits):
    rows = {r['field']: r for r in
            A.analyse('cold_key_rate', 'valid', splits=splits).rows}
    assert 0.0 < rows['user_or_video']['rate'] < 0.10


def test_distribution_covers_every_row(splits):
    rows = A.analyse('distribution', 'valid', splits=splits, column='tab').rows
    assert sum(r['rows'] for r in rows) == len(splits['valid'])
    assert sum(r['share'] for r in rows) == pytest.approx(1.0)


def test_segment_metrics_break_the_score_down_by_segment(splits, scores):
    rows = A.analyse('segment_metrics', 'valid', splits=splits, scores=scores,
                     column='user_impressions', bins=4).rows
    assert rows
    assert all(0.0 <= r['primary'] <= 1.0 for r in rows)
    assert sum(r['users'] for r in rows) <= len(set(hdata.user_ids(splits, 'valid')))


def test_score_tie_rate_distinguishes_a_tying_model(splits, scores):
    continuous = A.analyse('score_tie_rate', 'valid', splits=splits,
                           scores=scores).rows[0]
    coarse = A.analyse('score_tie_rate', 'valid', splits=splits,
                       scores=np.round(scores, 1)).rows[0]
    assert continuous['tie_rate'] == 0.0
    assert coarse['tie_rate'] > 0.05, (
        'a tie is a decision the model declined to make; the tool must see it')


def test_model_disagreement_sees_two_different_models(splits, scores):
    rng = np.random.default_rng(1)
    other = rng.random(len(scores))
    apart = A.analyse('model_disagreement', 'valid', splits=splits,
                      scores=scores, other_scores=other).rows[0]
    together = A.analyse('model_disagreement', 'valid', splits=splits,
                         scores=scores, other_scores=scores).rows[0]
    assert apart['disagreement_rate'] > 0.5
    assert together['disagreement_rate'] == 0.0, 'a model cannot disagree with itself'


def test_mismatched_score_length_is_rejected(splits):
    for kind, extra in (('score_tie_rate', {}),
                        ('segment_metrics', {}),
                        ('model_disagreement', {'other_scores': [0.0] * 10})):
        with pytest.raises(A.AnalysisError):
            A.analyse(kind, 'valid', splits=splits, scores=[0.0] * 10, **extra)
