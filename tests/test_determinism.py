"""Determinism of the measurement path.

Everything the agent will be judged on rests on two claims: that the same input
gives the same rows in the same order, and that the same rows give the same
score. Neither is checked anywhere else, and both are silent when they break.

Row order matters more than it looks. ``row_id`` in a submission is the
positional index into ``harness.data.load()[split]``, so a loader that returned
the same rows in a different order would produce a submission that validates
cleanly and scores as noise.

There is no model of ours to be deterministic about yet; that test belongs with
the model runners in Milestone 2. What is pinned here is the loader, the
encoder, the metric, and the organisers' own seeded baseline.
"""
from __future__ import annotations

import numpy as np
import pytest

from harness import data as hdata
from harness import evaluate as hevaluate
from harness import guards

pytestmark = pytest.mark.data


# --------------------------------------------------------------------------
# the loader
# --------------------------------------------------------------------------

def test_load_is_byte_for_byte_repeatable(data_dir):
    """Two independent loads agree on every field of every row."""
    first = hdata.load(data_dir, use_cache=False)
    second = hdata.load(data_dir, use_cache=False)
    assert hdata.row_counts(first) == hdata.row_counts(second)
    for split in hdata.SPLITS:
        assert first[split] == second[split], f'{split} rows differ between loads'


def test_cached_and_uncached_loads_agree(data_dir, splits):
    """The in-process cache must not be a second source of truth."""
    fresh = hdata.load(data_dir, use_cache=False)
    for split in hdata.SPLITS:
        assert fresh[split] == splits[split]


def test_row_order_is_stable_and_is_what_row_id_means(splits):
    """A spot check that order, not just content, is fixed.

    If this ever fails, every submission written before it failed was scored
    against the wrong rows.
    """
    for split in hdata.SPLITS:
        rows = splits[split]
        probe = [0, 1, len(rows) // 3, len(rows) // 2, len(rows) - 1]
        assert [rows[i] for i in probe] == [splits[split][i] for i in probe]


# --------------------------------------------------------------------------
# the encoder
# --------------------------------------------------------------------------

def test_encode_is_repeatable(splits):
    small = {'train': splits['train'][:20000],
             'valid': splits['valid'][:2000],
             'test': splits['test'][:2000]}
    first, dim_first = hdata.encode(small)
    second, dim_second = hdata.encode(small)
    assert dim_first == dim_second
    for split in hdata.SPLITS:
        np.testing.assert_array_equal(first[split][0], second[split][0])
        assert first[split][2] == second[split][2]
    np.testing.assert_array_equal(first['valid'][1], second['valid'][1])
    assert first['test'][1] is None and second['test'][1] is None


def test_encode_vocabulary_is_built_from_train_only(splits):
    """Adding validation rows must not change how training rows encode.

    A vocabulary that grew with the evaluation data would be a mild form of
    leakage and would silently shift every id.
    """
    base = {'train': splits['train'][:20000],
            'valid': splits['valid'][:1000],
            'test': splits['test'][:1000]}
    wider = {'train': base['train'],
             'valid': splits['valid'][:4000],
             'test': splits['test'][:4000]}
    first, dim_first = hdata.encode(base)
    second, dim_second = hdata.encode(wider)
    assert dim_first == dim_second
    np.testing.assert_array_equal(first['train'][0], second['train'][0])


# --------------------------------------------------------------------------
# the metric
# --------------------------------------------------------------------------

def test_evaluate_is_deterministic(splits):
    scores = np.random.default_rng(0).random(len(splits['valid']))
    first = hevaluate.evaluate_split(splits, 'valid', scores)
    second = hevaluate.evaluate_split(splits, 'valid', scores)
    assert first == second


def test_evaluate_is_order_sensitive_within_a_user(splits):
    """A sanity check on the check: shuffling scores must change the score.

    If a permuted score vector scored identically, the metric would not be
    measuring ranking and the determinism tests above would be vacuous.
    """
    rng = np.random.default_rng(0)
    scores = rng.random(len(splits['valid']))
    shuffled = rng.permutation(scores)
    baseline = hevaluate.evaluate_split(splits, 'valid', scores)['primary']
    permuted = hevaluate.evaluate_split(splits, 'valid', shuffled)['primary']
    assert baseline != permuted


# --------------------------------------------------------------------------
# the organisers' seeded baseline
# --------------------------------------------------------------------------

@pytest.mark.slow
def test_same_seed_gives_the_same_baseline_twice(data_dir):
    """The claim the contract test depends on: seeded training is reproducible.

    ``--model random`` is used rather than ``fm`` because it exercises the same
    seeding path in seconds. The FM's reproducibility is pinned separately, and
    to a fixed published number, by test_contract_baseline.py.
    """
    def run(seed):
        result = guards.run_starter_script(
            'baseline.py', ['--model', 'random', '--seed', str(seed)],
            data_dir=data_dir, timeout=1800)
        assert result.returncode == 0, result.stderr
        return result.stdout.split('primary')[-1].split()[0]

    assert run(0) == run(0)
    assert run(0) != run(1), 'different seeds must give different scores'
