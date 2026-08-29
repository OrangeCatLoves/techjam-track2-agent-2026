"""The contract test. The gate every harness change must pass.

If any of these move, something broke. Re-run this after touching anything under
``harness/``; movement over 0.001 on the FM row means stop and find out why.

  * split row counts are exactly 1,141,112 / 124,909 / 170,588
  * ``baseline.py --model fm`` gives validation primary 0.6015 +/- 0.001
  * ``baseline.py --model random`` gives validation primary 0.4834 +/- 0.002

The random tolerance is deliberately looser than the FM one: 0.4834 is the
organisers' mean over seeds 0-4 and we run a single seed, which measured 0.4827.

Every organiser script here is run through ``harness.guards.run_starter_script``,
so the test itself never sees the hidden-test metrics those scripts print. That
is asserted too: a contract test that leaked would be worse than no contract test.
"""
from __future__ import annotations

import ast
import re

import pytest

from harness import data as hdata
from harness import guards

pytestmark = pytest.mark.data

#: ``  valid  GAUC 0.6671 | nDCG@5 0.5358 | primary 0.6015``
VALID_LINE = re.compile(
    r'^\s*valid\s+GAUC\s+([0-9.]+)\s*\|\s*nDCG@5\s+([0-9.]+)\s*\|\s*primary\s+([0-9.]+)\s*$',
    re.MULTILINE)
#: ``{'train': 1141112, 'valid': 124909, 'test': 170588} fields=[...]``
COUNTS_LINE = re.compile(r"^\s*(\{'train':.*?\})", re.MULTILINE)

EXPECTED = {'train': 1141112, 'valid': 124909, 'test': 170588}


def parse_valid_scores(stdout: str) -> dict:
    """Pull the final validation line out of filtered organiser stdout."""
    matches = VALID_LINE.findall(stdout)
    assert matches, f'no validation summary line found in:\n{stdout[-2000:]}'
    gauc, ndcg, primary = matches[-1]
    return {'GAUC': float(gauc), 'nDCG@5': float(ndcg), 'primary': float(primary)}


def run_baseline(model: str, data_dir, seed: int = 0) -> guards.StarterRun:
    run = guards.run_starter_script(
        'baseline.py', ['--model', model, '--seed', str(seed)],
        data_dir=data_dir, timeout=3600)
    assert run.returncode == 0, f'baseline.py --model {model} failed:\n{run.stderr}'
    # The contract test must not be the leak. Assert the filter did its job.
    guards.assert_no_test_metrics(run.stdout, where=f'baseline --model {model} stdout')
    assert run.redacted_lines >= 1, (
        'baseline.py always prints a test line; redacting none means the filter '
        'silently stopped working')
    return run


# --------------------------------------------------------------------------
# row counts
# --------------------------------------------------------------------------

def test_split_row_counts(splits):
    assert hdata.row_counts(splits) == EXPECTED


def test_expected_rows_in_config_match_the_contract(expected_rows):
    assert expected_rows == EXPECTED


def test_split_date_boundaries(splits):
    """Every row falls inside its split's date window, and the windows are
    contiguous and disjoint. ``row_id`` is a position in these lists, so a split
    that silently changed shape would invalidate every submission we write.

    Note the train window is ``20220408-20220421`` by the rule but the log's
    first row is dated 20220409: the standard log simply has no 8 April rows.
    The observed extremes are pinned as measured, not as the rule states them.
    """
    windows = {'train': (20220408, 20220421), 'valid': (20220422, 20220428),
               'test': (20220429, 20220508)}
    observed = {'train': (20220409, 20220421), 'valid': (20220422, 20220428),
                'test': (20220429, 20220508)}
    for name, (lo, hi) in windows.items():
        dates = [r[hdata.IDX_DATE] for r in splits[name]]
        assert lo <= min(dates) and max(dates) <= hi
        assert (min(dates), max(dates)) == observed[name]


# --------------------------------------------------------------------------
# the baselines
# --------------------------------------------------------------------------

@pytest.mark.slow
def test_fm_baseline_reproduces(data_dir):
    """The number to beat. Published 0.6016 on validation; measured 0.6015."""
    run = run_baseline('fm', data_dir)
    counts = ast.literal_eval(COUNTS_LINE.search(run.stdout).group(1))
    assert counts == EXPECTED

    scores = parse_valid_scores(run.stdout)
    assert scores['primary'] == pytest.approx(0.6015, abs=0.001)
    assert scores['GAUC'] == pytest.approx(0.6674, abs=0.002)
    assert scores['nDCG@5'] == pytest.approx(0.5357, abs=0.002)


@pytest.mark.slow
def test_random_baseline_is_the_expected_lower_bound(data_dir):
    """The kit's own self-check: a broken harness does not land near 0.483."""
    run = run_baseline('random', data_dir)
    scores = parse_valid_scores(run.stdout)
    assert scores['primary'] == pytest.approx(0.4834, abs=0.002)


def test_item_popularity_rung(data_dir):
    """The trivial baseline. Pure statistics, no training, no seed variance, so
    the tolerance is the tight one and it is fast enough to run every time."""
    scores = parse_valid_scores(run_baseline('pop', data_dir).stdout)
    assert scores['primary'] == pytest.approx(0.5807, abs=0.001)
    assert scores['GAUC'] == pytest.approx(0.6387, abs=0.001)
    assert scores['nDCG@5'] == pytest.approx(0.5227, abs=0.001)


@pytest.mark.slow
def test_the_baseline_ladder_is_in_order(data_dir):
    """random < item popularity < FM, by the published margins.

    A single assertion that catches a scrambled evaluation path: any bug that
    scrambles scores collapses this ordering long before it moves a single number
    by 0.001.
    """
    rnd = parse_valid_scores(run_baseline('random', data_dir).stdout)['primary']
    pop = parse_valid_scores(run_baseline('pop', data_dir).stdout)['primary']
    fm = parse_valid_scores(run_baseline('fm', data_dir).stdout)['primary']
    assert rnd < pop < fm
    assert fm - rnd == pytest.approx(0.6016 - 0.4834, abs=0.004)
    assert fm - pop == pytest.approx(0.6016 - 0.5807, abs=0.002)
