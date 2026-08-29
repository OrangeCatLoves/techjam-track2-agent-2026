"""The integrity test. Asserts that the hidden test set stays hidden from us.

The organisers' loader hands this process 170,588 test rows with their true labels
at index 6, and their baseline script prints test metrics to stdout. Nothing stops
us from looking except the two controls asserted here.

What is asserted:
  * ``harness.data.load()['test']`` rows are six long, train and valid seven
  * a deliberate ``row[6]`` on a test row raises ``IndexError``
  * ``harness.data.labels(splits, 'test')`` raises ``TestLabelAccessError``
  * ``harness.evaluate`` and ``harness.submit`` refuse the test split
  * the stdout filter removes the test line from a real captured ``baseline.py`` run
    while leaving the validation line and the row-count line intact
"""
from __future__ import annotations

import pytest

from harness import data as hdata
from harness import evaluate as hevaluate
from harness import guards
from harness import submit as hsubmit

pytestmark = pytest.mark.data


# --------------------------------------------------------------------------
# the in-memory control
# --------------------------------------------------------------------------

def test_test_rows_are_six_wide(splits):
    assert len(splits['test'][0]) == 6
    assert all(len(r) == 6 for r in splits['test'][:1000])


def test_labelled_splits_keep_their_label(splits):
    assert len(splits['train'][0]) == 7
    assert len(splits['valid'][0]) == 7
    assert set(hdata.labels(splits, 'valid')[:1000]) <= {0, 1}


def test_deliberately_reading_a_test_label_fails_loudly(splits):
    """The acceptance criterion: an attempt to read a test label must fail."""
    row = splits['test'][0]
    with pytest.raises(IndexError):
        _ = row[hdata.IDX_LABEL]
    with pytest.raises(IndexError):
        _ = [r[6] for r in splits['test'][:10]]


def test_labels_helper_refuses_test(splits):
    with pytest.raises(hdata.TestLabelAccessError):
        hdata.labels(splits, 'test')


def test_evaluate_refuses_test(splits):
    with pytest.raises(hdata.TestLabelAccessError):
        hevaluate.evaluate_split(splits, 'test', [0.0] * len(splits['test']))


def test_submit_score_refuses_test(tmp_path, splits):
    path = tmp_path / 'sub.csv'
    hsubmit.write_split(path, splits, 'test', [0.0] * len(splits['test']))
    # Checking format is fine: it needs no label.
    hsubmit.check(path, 'test', splits)
    with pytest.raises(hdata.TestLabelAccessError):
        hsubmit.score(path, 'test', splits)


def test_encode_returns_no_test_labels(splits):
    small = {'train': splits['train'][:5000],
             'valid': splits['valid'][:500],
             'test': splits['test'][:500]}
    enc, dim = hdata.encode(small)
    assert dim > 0
    assert enc['test'][1] is None, 'test encoding must carry no label array'
    assert enc['valid'][1] is not None


# --------------------------------------------------------------------------
# the stdout control, on synthetic text
# --------------------------------------------------------------------------

TRANSCRIPT = """loading C:/data ...
{'train': 1141112, 'valid': 124909, 'test': 170588} fields=['user_id']
  epoch  7 | loss 0.4859 | valid GAUC 0.6671 nDCG@5 0.5358 primary 0.6015 | 10.1s
=== fm (seed=0) ===
  valid  GAUC 0.6671 | nDCG@5 0.5358 | primary 0.6015
  test   GAUC 0.6621 | nDCG@5 0.5286 | primary 0.5953
5 domains (current kit)   ( 5) | test GAUC 0.6614 | nDCG@5 0.5285 | primary 0.5950 +/- 0.0003
"""


def test_filter_removes_test_metric_lines():
    clean, n = guards.filter_stdout(TRANSCRIPT)
    assert n == 2
    assert '0.5953' not in clean
    assert '0.5950' not in clean
    assert '0.6614' not in clean


def test_filter_keeps_validation_and_row_counts():
    clean, _ = guards.filter_stdout(TRANSCRIPT)
    assert '0.6015' in clean, 'the validation number must survive'
    assert "'test': 170588" in clean, 'the row-count line is not a metric line'
    assert clean.count('\n') == TRANSCRIPT.count('\n'), 'line count must be preserved'


@pytest.mark.parametrize('line', [
    '  test   GAUC 0.6621 | nDCG@5 0.5286 | primary 0.5953',
    'test primary 0.5946',
    'TEST GAUC: 0.66',
    '  test 0.5953',
    'hidden test set primary 0.59',
    'cfg (13) | test GAUC 0.6601 | nDCG@5 0.5280 | primary 0.5940 +/- 0.0005',
])
def test_lines_that_must_be_redacted(line):
    assert guards.contains_test_metric(line)


@pytest.mark.parametrize('line', [
    '  valid  GAUC 0.6671 | nDCG@5 0.5358 | primary 0.6015',
    "{'train': 1141112, 'valid': 124909, 'test': 170588}",
    'running the contract test suite',
    '  epoch  7 | loss 0.4859 | valid GAUC 0.6671 primary 0.6015',
])
def test_lines_that_must_survive(line):
    assert not guards.contains_test_metric(line)


def test_assert_no_test_metrics_raises():
    with pytest.raises(guards.LeakageError):
        guards.assert_no_test_metrics(TRANSCRIPT, where='transcript')
    clean, _ = guards.filter_stdout(TRANSCRIPT)
    guards.assert_no_test_metrics(clean, where='filtered transcript')


def test_log_records_are_screened():
    with pytest.raises(guards.LeakageError):
        guards.assert_record_clean({'metrics': {'test': {'primary': 0.5946}}})
    with pytest.raises(guards.LeakageError):
        guards.assert_record_clean({'test_primary': 0.5946})
    with pytest.raises(guards.LeakageError):
        guards.assert_record_clean({'note': '  test GAUC 0.66 primary 0.59'})
    guards.assert_record_clean(
        {'metrics': {'val_gauc': 0.667, 'val_primary': 0.6015},
         'note': 'validation primary improved'})


# --------------------------------------------------------------------------
# the stdout control, on a real organiser run
# --------------------------------------------------------------------------

@pytest.mark.slow
def test_captured_baseline_run_is_filtered(data_dir):
    """Run the organisers' own script and confirm the test line never reaches us.

    ``--model random`` is used because it exercises the identical print path as
    ``--model fm`` (both print a valid line and a test line) in seconds rather
    than minutes.
    """
    run = guards.run_starter_script('baseline.py', ['--model', 'random', '--seed', '0'],
                                    data_dir=data_dir, timeout=900)
    assert run.returncode == 0, run.stderr
    assert run.redacted_lines >= 1, 'the organiser script did print a test line'
    guards.assert_no_test_metrics(run.stdout, where='filtered baseline stdout')
    assert 'valid' in run.stdout and '0.4' in run.stdout
    assert guards.REDACTION in run.stdout

    # The raw output is kept for humans, and it does still contain the leak,
    # which is exactly why it must never be read into a prompt or a log.
    assert run.raw_log_path is not None and run.raw_log_path.exists()
    raw = run.raw_log_path.read_text(encoding='utf-8')
    assert 'HUMAN-ONLY' in raw
    with pytest.raises(guards.LeakageError):
        guards.assert_no_test_metrics(raw, where='raw log')
