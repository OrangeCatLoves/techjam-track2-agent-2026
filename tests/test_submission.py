"""Submission format tests.

The organisers' ``read_submission`` is the definition of a well-formed
submission, and CLAUDE.md section 6.3 forbids patching around a rejection from
it. These tests pin the five corruptions it must catch, so that a future harness
change cannot quietly loosen the contract.

They also close the loop on our own scoring path: model scores written to disk,
re-read through the organisers' validator, and scored through their metric must
equal the score computed directly in memory.
"""
from __future__ import annotations

import csv
import math

import pytest

from harness import data as hdata
from harness import evaluate as hevaluate
from harness import submit as hsubmit

pytestmark = pytest.mark.data

N_ROWS = 500


@pytest.fixture(scope='module')
def mini(splits):
    """A small stand-in evaluation split, in the real row order."""
    return {'valid': splits['valid'][:N_ROWS], 'test': splits['test'][:N_ROWS]}


@pytest.fixture()
def good(tmp_path, mini):
    """A valid submission over the mini split."""
    path = tmp_path / 'submission.csv'
    scores = [i * 0.001 for i in range(len(mini['valid']))]
    hsubmit.write_split(path, mini, 'valid', scores)
    return path


def rewrite(path, rows):
    with open(path, 'w', newline='', encoding='utf-8') as fh:
        csv.writer(fh).writerows(rows)
    return path


def read_rows(path):
    with open(path, newline='', encoding='utf-8') as fh:
        return list(csv.reader(fh))


# --------------------------------------------------------------------------
# the happy path
# --------------------------------------------------------------------------

def test_a_valid_submission_passes_check(good, mini):
    scores = hsubmit.check(good, 'valid', mini)
    assert len(scores) == N_ROWS


def test_header_and_row_ids_are_what_the_organisers_specify(good):
    rows = read_rows(good)
    assert rows[0] == ['row_id', 'user_id', 'video_id', 'score']
    assert [r[0] for r in rows[1:6]] == ['0', '1', '2', '3', '4']


def test_written_rows_align_with_the_evaluation_split(good, mini):
    rows = read_rows(good)[1:]
    for i in (0, 1, N_ROWS // 2, N_ROWS - 1):
        assert rows[i][1] == mini['valid'][i][hdata.IDX_USER]
        assert rows[i][2] == mini['valid'][i][hdata.IDX_VIDEO]


def test_check_works_on_the_test_split_without_a_label(tmp_path, mini):
    """Format checking must not need the label the harness refuses to hold."""
    path = tmp_path / 'test_submission.csv'
    hsubmit.write_split(path, mini, 'test', [0.5] * N_ROWS)
    assert len(hsubmit.check(path, 'test', mini)) == N_ROWS


@pytest.mark.slow
def test_round_trip_scoring_matches_in_memory_scoring(tmp_path, splits):
    """The independent check of our own scoring path, on the full valid split."""
    import numpy as np
    rng = np.random.default_rng(0)
    scores = rng.random(len(splits['valid']))
    path = tmp_path / 'full_valid.csv'
    hsubmit.write_split(path, splits, 'valid', scores)

    from_disk = hsubmit.score(path, 'valid', splits)
    in_memory = hevaluate.evaluate_split(splits, 'valid', scores)
    # Scores are written at six significant figures, so the agreement is close
    # but not bit-exact.
    assert from_disk['primary'] == pytest.approx(in_memory['primary'], abs=1e-6)
    assert from_disk['rows'] == len(splits['valid'])
    # The published single-seed random reference, as a sanity rung.
    assert from_disk['primary'] == pytest.approx(0.4834, abs=0.002)


# --------------------------------------------------------------------------
# the five corruptions
# --------------------------------------------------------------------------

def test_rejects_wrong_header(good, mini):
    rows = read_rows(good)
    rows[0] = ['row_id', 'user_id', 'video_id', 'pred']
    rewrite(good, rows)
    with pytest.raises(ValueError):
        hsubmit.check(good, 'valid', mini)


def test_rejects_row_count_mismatch(good, mini):
    rows = read_rows(good)
    rewrite(good, rows[:-1])
    with pytest.raises(ValueError):
        hsubmit.check(good, 'valid', mini)


def test_rejects_too_many_rows(good, mini):
    rows = read_rows(good)
    extra = list(rows[-1])
    extra[0] = str(N_ROWS)
    rewrite(good, rows + [extra])
    with pytest.raises(ValueError):
        hsubmit.check(good, 'valid', mini)


def test_rejects_row_id_gap(good, mini):
    rows = read_rows(good)
    rows[101][0] = str(int(rows[101][0]) + 1)      # 99, 101, 101, ...
    rewrite(good, rows)
    with pytest.raises(ValueError):
        hsubmit.check(good, 'valid', mini)


def test_rejects_misalignment(good, mini):
    rows = read_rows(good)
    rows[42][1], rows[43][1] = rows[43][1], rows[42][1]
    if rows[42][1] == rows[43][1]:                  # same user twice: shift instead
        rows[42][1] = rows[42][1] + '9'
    rewrite(good, rows)
    with pytest.raises(ValueError):
        hsubmit.check(good, 'valid', mini)


def test_rejects_nan_score(good, mini):
    rows = read_rows(good)
    rows[7][3] = 'nan'
    rewrite(good, rows)
    with pytest.raises(ValueError):
        hsubmit.check(good, 'valid', mini)


def test_rejects_inf_score(good, mini):
    rows = read_rows(good)
    rows[7][3] = 'inf'
    rewrite(good, rows)
    with pytest.raises(ValueError):
        hsubmit.check(good, 'valid', mini)


def test_rejects_unparseable_score(good, mini):
    rows = read_rows(good)
    rows[7][3] = 'high'
    rewrite(good, rows)
    with pytest.raises(ValueError):
        hsubmit.check(good, 'valid', mini)


def test_writer_refuses_a_score_count_mismatch(tmp_path, mini):
    with pytest.raises(ValueError):
        hsubmit.write(tmp_path / 'x.csv', mini['valid'], [0.0] * (N_ROWS - 1))


def test_nan_scores_never_reach_disk_undetected(tmp_path, mini):
    """A NaN produced by a model is caught by --check, not silently submitted."""
    path = tmp_path / 'nan.csv'
    scores = [0.5] * N_ROWS
    scores[3] = math.nan
    hsubmit.write(path, mini['valid'], scores)
    with pytest.raises(ValueError):
        hsubmit.check(path, 'valid', mini)
