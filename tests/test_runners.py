"""The model runner and the loss interface.

The load-bearing assertion is `test_our_trainer_reproduces_the_official_baseline`.
``harness/models/runners.py`` reimplements the organisers' training loop, because
theirs ends by computing a hidden-test metric and so cannot be called. A
reimplementation is only safe if it is provably the same trainer, so this file
pins it to the published number rather than trusting the code review that produced
it.

Also covered: the loss interface and its sign check, list construction, the
diagnostics contract from ``docs/M2_CONTRACT.md`` section 2, and checkpoint
round-tripping, which is what makes keep-or-reject possible at all.
"""
from __future__ import annotations

import numpy as np
import pytest

from harness import data as hdata
from harness import guards
from harness import losses as hlosses
from harness.models import runners as R

pytestmark = pytest.mark.data


# --------------------------------------------------------------------------
# the loss interface
# --------------------------------------------------------------------------

def test_pointwise_loss_satisfies_the_interface():
    report = hlosses.check_loss(hlosses.pointwise_logloss)
    assert np.isfinite(report['loss'])
    assert report['loss_after_step'] <= report['loss']


def test_pointwise_gradient_matches_the_organisers_formula():
    """``starter.baseline.FM.step`` hardcodes ``g = (sigmoid(z) - y) / B``.

    Our trainer gets that number from a loss function instead. If the two ever
    disagree, the baseline reproduction below would drift for reasons that are
    hard to localise, so the equality is asserted directly.
    """
    rng = np.random.default_rng(0)
    z = rng.normal(0, 1, 512)
    y = (rng.random(512) < 0.4).astype(np.float64)
    _, grad = hlosses.pointwise_logloss(z, y, np.zeros(512, dtype=np.int64))
    expected = (hlosses.sigmoid(z) - y) / len(y)
    np.testing.assert_allclose(grad, expected, rtol=1e-6, atol=1e-8)


def test_an_inverted_loss_is_rejected():
    """The most common way a hand-written objective silently trains backwards."""
    def inverted(z, y, groups):
        loss, grad = hlosses.pointwise_logloss(z, y, groups)
        return loss, -grad

    with pytest.raises(hlosses.LossError):
        hlosses.check_loss(inverted)


@pytest.mark.parametrize('broken,reason', [
    (lambda z, y, g: (float('nan'), np.zeros_like(z)), 'non-finite loss'),
    (lambda z, y, g: (0.5, np.zeros(3)), 'wrong gradient shape'),
    (lambda z, y, g: (0.5, np.full_like(z, np.inf)), 'non-finite gradient'),
    (lambda z, y, g: 0.5, 'not a pair'),
])
def test_malformed_losses_are_rejected(broken, reason):
    with pytest.raises(hlosses.LossError):
        hlosses.check_loss(broken)


def test_unknown_loss_name_is_rejected():
    with pytest.raises(hlosses.LossError):
        hlosses.get_loss('bpr_that_nobody_wrote_yet')


def test_only_the_pointwise_reference_ships():
    """Pairwise and listwise objectives are the agent's to write (CLAUDE.md 12.2).

    Shipping them would hand the agent its best idea and hollow out the Innovation
    score. If this fails, someone added a loss the agent was supposed to discover.
    """
    assert hlosses.registered() == ('pointwise_logloss',)


# --------------------------------------------------------------------------
# list construction
# --------------------------------------------------------------------------

def test_grouping_by_user_and_by_user_date_differ_as_documented(splits):
    """The open experiment, measured rather than assumed.

    Train lists under ``user_id`` are ~7x the evaluation list length; under
    ``user_id+date`` they are shorter than it. Neither matches, which is why the
    choice is an experiment for the agent and not a default to guess.
    """
    small = {'train': splits['train'][:200000]}
    by_user = R.build_groups(small, 'train', 'user_id')
    by_user_date = R.build_groups(small, 'train', 'user_id+date')
    assert np.unique(by_user).size < np.unique(by_user_date).size
    size_user = len(by_user) / np.unique(by_user).size
    size_user_date = len(by_user_date) / np.unique(by_user_date).size
    assert size_user > size_user_date


def test_unknown_grouping_is_rejected(splits):
    with pytest.raises(ValueError):
        R.build_groups({'train': splits['train'][:100]}, 'train', 'user_id+phase_of_moon')


# --------------------------------------------------------------------------
# the reproduction -- the reason this module is allowed to exist
# --------------------------------------------------------------------------

@pytest.mark.slow
def test_our_trainer_reproduces_the_official_baseline(splits):
    """Our reimplemented loop must be the organisers' loop.

    Published validation primary is 0.6016; the organisers' own script measured
    0.6015 on this machine. Ours must land on the same number, not merely a good
    one, because every later comparison is made against it.
    """
    result = R.train_fm(splits, seed=0)
    assert result.val_primary == pytest.approx(0.6015, abs=0.001)
    assert result.val_gauc == pytest.approx(0.6674, abs=0.002)
    assert result.val_ndcg5 == pytest.approx(0.5357, abs=0.002)

    # The overfitting signature the organisers' run shows: the best epoch is well
    # before the last, while training loss keeps falling.
    assert result.best_epoch < result.epochs_run
    first, last = result.epoch_history[0], result.epoch_history[-1]
    assert last['loss'] < first['loss']


@pytest.mark.slow
def test_the_same_seed_gives_the_same_model(splits):
    a = R.train_fm(splits, seed=0, max_epochs=2, with_diagnostics=False)
    b = R.train_fm(splits, seed=0, max_epochs=2, with_diagnostics=False)
    assert a.val_primary == b.val_primary
    assert [h['loss'] for h in a.epoch_history] == [h['loss'] for h in b.epoch_history]


@pytest.mark.slow
def test_different_seeds_give_different_models(splits):
    a = R.train_fm(splits, seed=0, max_epochs=2, with_diagnostics=False)
    b = R.train_fm(splits, seed=7, max_epochs=2, with_diagnostics=False)
    assert a.val_primary != b.val_primary


@pytest.mark.slow
def test_a_custom_loss_reaches_the_trainer(splits):
    """A generated objective must change training without the loop changing.

    Halving the gradient halves the effective learning rate, so after one epoch
    the model must differ from the reference. This is the cheapest possible proof
    that the loss is actually plumbed through rather than ignored.
    """
    def half_step(z, y, groups):
        loss, grad = hlosses.pointwise_logloss(z, y, groups)
        return loss, (grad * 0.5).astype(np.float32)

    reference = R.train_fm(splits, seed=0, max_epochs=1, with_diagnostics=False)
    altered = R.train_fm(splits, seed=0, max_epochs=1, loss=half_step,
                         with_diagnostics=False)
    assert altered.val_primary != reference.val_primary


# --------------------------------------------------------------------------
# the diagnostics contract
# --------------------------------------------------------------------------

@pytest.fixture(scope='module')
def short_run(splits):
    """One cheap trained model, shared by the contract tests below."""
    return R.train_fm(splits, seed=0, max_epochs=1, patience=1)


@pytest.mark.slow
def test_diagnostics_have_the_three_required_groups(short_run):
    diagnostics = short_run.diagnostics
    assert set(diagnostics['metrics']) == {'val_gauc', 'val_ndcg5', 'val_primary'}
    assert set(diagnostics['fit']) >= {'train_primary', 'val_primary', 'gap',
                                       'epochs_run', 'best_epoch'}
    assert set(diagnostics['fields']) == set(R.FIELDS)
    for stats in diagnostics['fields'].values():
        assert stats['n_ids'] > 0
        assert np.isfinite(stats['mean_abs_w'])
        assert np.isfinite(stats['mean_v_norm'])


@pytest.mark.slow
def test_diagnostics_carry_no_test_metric(short_run):
    """The contract's hard rule: nothing leaving run_experiment names the test split."""
    guards.assert_record_clean(short_run.diagnostics, where='diagnostics')


@pytest.mark.slow
def test_the_train_validation_gap_is_reported_and_positive(short_run):
    fit = short_run.diagnostics['fit']
    assert fit['train_primary'] is not None
    assert fit['gap'] == pytest.approx(fit['train_primary'] - fit['val_primary'])
    assert fit['gap'] > 0, 'the model should fit train better than validation'


@pytest.mark.slow
def test_list_sizes_show_the_train_evaluation_mismatch(short_run):
    """Documented in CLAUDE.md 9.2 and surfaced here so the agent can see it."""
    lists = short_run.diagnostics['lists']
    assert lists['mean_train_list_size'] > 2 * lists['mean_valid_list_size']
    assert 5.0 < lists['mean_valid_list_size'] < 7.0


# --------------------------------------------------------------------------
# checkpoints -- what makes keep-or-reject possible
# --------------------------------------------------------------------------

@pytest.mark.slow
def test_checkpoint_round_trip_gives_identical_predictions(splits, tmp_path):
    path = tmp_path / 'model.npz'
    result = R.train_fm(splits, seed=0, max_epochs=1, patience=1,
                        checkpoint_path=path, with_diagnostics=False)
    assert result.checkpoint is not None

    enc, dim = hdata.encode(splits)
    restored = R.load_checkpoint(path, dim=dim, k=16)
    scores = R.score_split(restored, splits, 'valid', enc)

    from harness import evaluate as hevaluate
    rescored = hevaluate.evaluate_split(splits, 'valid', scores)
    assert rescored['primary'] == pytest.approx(result.val_primary, abs=1e-9)


@pytest.mark.slow
def test_a_checkpoint_can_score_the_test_split_without_a_label(splits, tmp_path):
    """Producing a submission needs the features, never the answer."""
    path = tmp_path / 'model.npz'
    R.train_fm(splits, seed=0, max_epochs=1, patience=1, checkpoint_path=path,
               with_diagnostics=False)
    enc, dim = hdata.encode(splits)
    assert enc['test'][1] is None
    scores = R.score_split(R.load_checkpoint(path, dim=dim, k=16),
                           splits, 'test', enc)
    assert len(scores) == len(splits['test'])
    assert np.all(np.isfinite(scores))
