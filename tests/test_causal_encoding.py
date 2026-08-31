"""The causal window, tested as a property rather than as a spot check.

CLAUDE.md section 12.2: "A statistic for a row on date d is identical whether or
not rows dated >= d exist in the input."

That is the assertion that matters, because a target encoding computed over the
wrong window does not crash. It produces a *better* validation score, which is
the failure mode nobody notices. So the window is tested by deletion: build the
statistics twice, once on the full data and once with the future removed, and
require the answers to match exactly.
"""
from __future__ import annotations

import numpy as np
import pytest

from harness import data as hdata
from harness.features import base as fbase
from harness.features import registry as freg


# --------------------------------------------------------------------------
# a small synthetic world, so the window can be reasoned about by hand
# --------------------------------------------------------------------------

def _row(date, user, video, author='a1', tab='1', dur=5000.0, label=0):
    return (date, user, video, author, tab, dur, label)


@pytest.fixture
def toy():
    """Three train dates. Video v1 is long-viewed on day 1 and not on day 2.

    Hand-computable: on day 1 v1 has no history, on day 2 it has one impression
    with one positive, on day 3 it has two impressions with one positive.
    """
    train = [
        _row(20220408, 'u1', 'v1', label=1),
        _row(20220409, 'u1', 'v1', label=0),
        _row(20220410, 'u1', 'v1', label=1),
        _row(20220408, 'u2', 'v2', label=0),
        _row(20220409, 'u2', 'v2', label=0),
        _row(20220410, 'u2', 'v2', label=1),
    ]
    valid = [_row(20220422, 'u1', 'v1', label=1), _row(20220422, 'u2', 'v2', label=0)]
    test = [r[:hdata.IDX_LABEL] for r in
            [_row(20220429, 'u1', 'v1'), _row(20220429, 'u2', 'v2')]]
    return {'train': train, 'valid': valid, 'test': test}


# --------------------------------------------------------------------------
# the property that matters
# --------------------------------------------------------------------------

def test_statistic_ignores_rows_from_its_own_date_and_later(toy):
    """Deleting the future must not change a past row's statistic."""
    _, full = fbase.build_stats(toy, prior=0.0)

    trimmed = {'train': [r for r in toy['train'] if r[hdata.IDX_DATE] < 20220410],
               'valid': toy['valid'], 'test': toy['test']}
    _, cut = fbase.build_stats(trimmed, prior=0.0)

    keep = np.array([r[hdata.IDX_DATE] < 20220410 for r in toy['train']])
    np.testing.assert_array_equal(
        full['train'].label_rate('video_id')[keep],
        cut['train'].label_rate('video_id'),
        err_msg='a train row changed when later dates were deleted -- the window leaks')


def test_a_rows_own_label_never_reaches_its_own_statistic(toy):
    """Flipping one row's label must not move that row's own feature."""
    _, before = fbase.build_stats(toy, prior=0.0)

    flipped = dict(toy)
    rows = list(toy['train'])
    r = rows[0]
    rows[0] = r[:hdata.IDX_LABEL] + (1 - r[hdata.IDX_LABEL],)
    flipped['train'] = rows
    _, after = fbase.build_stats(flipped, prior=0.0)

    assert before['train'].label_rate('video_id')[0] == \
        after['train'].label_rate('video_id')[0], \
        'a row saw its own label -- this is the leak the window exists to prevent'


def test_expanding_window_values_are_what_the_rule_says(toy):
    """Hand-computed, so a plausible-but-wrong window cannot pass silently."""
    _, stats = fbase.build_stats(toy, prior=0.0)
    rate = stats['train'].label_rate('video_id')
    count = stats['train'].exposure_count('video_id')

    # v1 rows are at positions 0, 1, 2 on dates 08, 09, 10.
    assert count[0] == 0.0, 'day one should have no history'
    assert count[1] == 1.0, 'day two should see exactly day one'
    assert count[2] == 2.0, 'day three should see days one and two'
    assert rate[1] == pytest.approx(1.0), 'v1 was long-viewed on its only prior day'
    assert rate[2] == pytest.approx(0.5), 'one positive out of two prior impressions'


def test_evaluation_rows_use_the_whole_train_period(toy):
    """Validation and test are scored from all of train, and identically."""
    _, stats = fbase.build_stats(toy, prior=0.0)
    assert stats['valid'].exposure_count('video_id')[0] == 3.0
    assert stats['test'].exposure_count('video_id')[0] == 3.0
    assert stats['valid'].label_rate('video_id')[0] == \
        stats['test'].label_rate('video_id')[0], \
        'valid and test must share a window; a gap between them is a silent bug'


def test_validation_labels_are_not_used_to_build_statistics(toy):
    """Changing a validation label must not move any statistic anywhere.

    Validation labels are in this process, and using them would inflate the one
    number that decides everything. This asserts they are ignored.
    """
    _, before = fbase.build_stats(toy, prior=0.0)
    flipped = dict(toy)
    v = list(toy['valid'])
    v[0] = v[0][:hdata.IDX_LABEL] + (1 - v[0][hdata.IDX_LABEL],)
    flipped['valid'] = v
    _, after = fbase.build_stats(flipped, prior=0.0)
    for split in ('train', 'valid', 'test'):
        np.testing.assert_array_equal(
            before[split].label_rate('video_id'),
            after[split].label_rate('video_id'),
            err_msg=f'{split} statistics moved when a validation label changed')


def test_test_rows_carry_no_label_to_leak(toy):
    """The test frame is built from six-wide rows and nothing asks for a seventh."""
    frames, _ = fbase.build_stats(toy, prior=0.0)
    assert len(frames['test']) == len(toy['test'])
    with pytest.raises(hdata.TestLabelAccessError):
        hdata.labels(toy, 'test')


# --------------------------------------------------------------------------
# the frame a generated feature is handed
# --------------------------------------------------------------------------

def test_frame_exposes_no_outcome_column(toy):
    """A feature must not be able to reach watch time, clicks or the label."""
    frames, _ = fbase.build_stats(toy, prior=0.0)
    frame = frames['train']
    for banned in ('label', 'y', 'long_view', 'play_time_ms', 'is_click', 'is_like'):
        assert not hasattr(frame, banned), f'Frame exposes {banned!r}'


def test_unknown_key_field_is_rejected(toy):
    frames, stats = fbase.build_stats(toy, prior=0.0)
    with pytest.raises(fbase.FeatureError):
        stats['train'].label_rate('play_time_ms')
    with pytest.raises(fbase.FeatureError):
        frames['train'].keys('is_click')


# --------------------------------------------------------------------------
# bucketisation
# --------------------------------------------------------------------------

def test_bucket_edges_come_from_train_only(toy):
    """An evaluation value must not be able to shift the binning."""
    base = {'train': np.arange(100.0), 'valid': np.arange(10.0)}
    wild = {'train': np.arange(100.0), 'valid': np.full(10, 1e9)}
    assert np.array_equal(fbase.bucketise(base)['train'],
                          fbase.bucketise(wild)['train'])


def test_constant_feature_yields_one_bucket():
    """A feature carrying no information is representable, not an error."""
    out = fbase.bucketise({'train': np.ones(50), 'valid': np.ones(5)})
    assert len(np.unique(out['train'])) == 1


# --------------------------------------------------------------------------
# the feature contract
# --------------------------------------------------------------------------

def test_check_feature_rejects_wrong_length(toy):
    frames, stats = fbase.build_stats(toy, prior=0.0)
    with pytest.raises(fbase.FeatureError):
        freg.check_feature(lambda f, s: np.zeros(len(f) + 1), frames['train'], stats['train'])


def test_check_feature_rejects_non_finite(toy):
    frames, stats = fbase.build_stats(toy, prior=0.0)
    with pytest.raises(fbase.FeatureError):
        freg.check_feature(lambda f, s: np.full(len(f), np.nan), frames['train'], stats['train'])


def test_check_feature_rejects_non_determinism(toy):
    frames, stats = fbase.build_stats(toy, prior=0.0)
    rng = np.random.default_rng(0)
    with pytest.raises(fbase.FeatureError):
        freg.check_feature(lambda f, s: rng.random(len(f)), frames['train'], stats['train'])


def test_reference_feature_satisfies_the_contract(toy):
    frames, stats = fbase.build_stats(toy, prior=0.0)
    out = freg.check_feature(freg.get_feature('video_exposure_count'),
                             frames['train'], stats['train'])
    assert out.shape == (len(frames['train']),)


def test_reference_feature_reads_no_label(toy):
    """Flipping every train label must not move a pure exposure count."""
    frames, stats = fbase.build_stats(toy, prior=0.0)
    before = freg.get_feature('video_exposure_count')(frames['train'], stats['train'])

    flipped = dict(toy)
    flipped['train'] = [r[:hdata.IDX_LABEL] + (1 - r[hdata.IDX_LABEL],)
                        for r in toy['train']]
    f2, s2 = fbase.build_stats(flipped, prior=0.0)
    after = freg.get_feature('video_exposure_count')(f2['train'], s2['train'])
    np.testing.assert_array_equal(before, after)


# --------------------------------------------------------------------------
# augmentation
# --------------------------------------------------------------------------

def test_augment_widens_x_and_offsets_past_existing_fields():
    enc = {'train': (np.zeros((4, 5), dtype=np.int32), np.zeros(4), ['u'] * 4),
           'valid': (np.zeros((2, 5), dtype=np.int32), np.zeros(2), ['u'] * 2)}
    cols = {'f': {'train': np.array([0, 1, 2, 0]), 'valid': np.array([1, 0])}}
    out, dim = fbase.augment(enc, 100, cols)
    assert out['train'][0].shape == (4, 6)
    assert dim == 103
    assert out['train'][0][:, 5].min() >= 100, 'new field collides with existing ids'
    np.testing.assert_array_equal(out['train'][0][:, 5], np.array([100, 101, 102, 100]))


def test_augment_rejects_a_length_mismatch():
    enc = {'train': (np.zeros((4, 5), dtype=np.int32), np.zeros(4), ['u'] * 4)}
    with pytest.raises(fbase.FeatureError):
        fbase.augment(enc, 100, {'f': {'train': np.array([0, 1])}})


# --------------------------------------------------------------------------
# the real dataset
# --------------------------------------------------------------------------

@pytest.mark.slow
def test_window_holds_on_the_real_split():
    """The deletion property, on the actual 1.14M train rows."""
    splits = hdata.load()
    cut_at = 20220415
    _, full = fbase.build_stats(splits, fields=('video_id',))
    trimmed = dict(splits)
    trimmed['train'] = [r for r in splits['train'] if r[hdata.IDX_DATE] < cut_at]
    _, cut = fbase.build_stats(trimmed, fields=('video_id',))

    keep = np.array([r[hdata.IDX_DATE] < cut_at for r in splits['train']])
    np.testing.assert_allclose(
        full['train'].label_rate('video_id')[keep],
        cut['train'].label_rate('video_id'),
        rtol=0, atol=0,
        err_msg='the window leaks on real data')
