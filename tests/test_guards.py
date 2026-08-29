"""Guard behaviour that the other suites assume but do not check.

Two jobs here.

The **encoding round trip** (D10). ``run_starter_script`` captures organiser output
as UTF-8, which is correct — the starter kit is bilingual and prints Chinese. But
the Windows console encoding is cp1252 and is independent of how the text was read,
so *printing* or *logging* that captured text still raises ``UnicodeEncodeError``
unless the sink is given an explicit encoding. This was found the expensive way: a
subprocess succeeded, the submission was written correctly, and the line reporting
it crashed.

In Milestone 2 the agent loop prints and logs tool output every iteration, so that
failure would surface as "the experiment crashed" rather than "the print crashed".
These tests exist so that a ``print()`` added in a later milestone cannot
reintroduce it silently.

The **deny-list and canary** behaviours, which until now were exercised only by
``scripts/verify_setup.py`` and not by the test suite.
"""
from __future__ import annotations

import json

import pytest

from harness import guards
from harness import submit as hsubmit

# The organisers' own strings, copied from starter/submit.py lines 99 and 102.
CHINESE_OUTPUT = '✓ 格式与对齐校验通过：124,909 行，split=valid'
CHINESE_WITH_TEST_METRIC = 'CWM 全 13 域 (13) | test GAUC 0.6601 | primary 0.5940'


# --------------------------------------------------------------------------
# encoding: the D10 regression
# --------------------------------------------------------------------------

def test_filter_preserves_non_ascii_text():
    """Filtering must not mangle or drop a legitimate non-ASCII line."""
    clean, redacted = guards.filter_stdout(CHINESE_OUTPUT + '\n')
    assert redacted == 0
    assert CHINESE_OUTPUT in clean


def test_filter_still_redacts_a_non_ascii_test_metric_line():
    """A Chinese line carrying a test metric is redacted like any other."""
    clean, redacted = guards.filter_stdout(CHINESE_WITH_TEST_METRIC + '\n')
    assert redacted == 1
    assert '0.5940' not in clean


def test_non_ascii_output_round_trips_to_a_utf8_log(tmp_path):
    """The M2 log sink. Writing captured output must not raise."""
    path = tmp_path / 'log.md'
    clean, _ = guards.filter_stdout(CHINESE_OUTPUT + '\n')
    path.write_text(clean, encoding='utf-8')
    assert CHINESE_OUTPUT in path.read_text(encoding='utf-8')


def test_non_ascii_output_survives_json_serialisation(tmp_path):
    """The M2 JSONL sink. json.dump defaults to ensure_ascii=True, which escapes
    rather than raises, but the reader must get the original text back."""
    path = tmp_path / 'log.jsonl'
    record = {'stdout': CHINESE_OUTPUT}
    guards.assert_record_clean(record)
    path.write_text(json.dumps(record) + '\n', encoding='utf-8')
    restored = json.loads(path.read_text(encoding='utf-8'))
    assert restored['stdout'] == CHINESE_OUTPUT


def test_a_cp1252_sink_is_the_thing_that_breaks(tmp_path):
    """Pin the actual failure mode, so the fix is not mistaken for luck.

    If this ever stops raising, the environment changed and the explicit
    ``encoding='utf-8'`` on every sink is no longer load-bearing — but until
    then, it is.
    """
    path = tmp_path / 'cp1252.log'
    with pytest.raises(UnicodeEncodeError):
        path.write_text(CHINESE_OUTPUT, encoding='cp1252')


@pytest.mark.data
def test_captured_starter_output_is_non_ascii_and_loggable(tmp_path, splits):
    """End to end against a real organiser run that prints Chinese.

    ``submit.py --check`` is used because it prints a non-ASCII success line and
    needs no training.
    """
    submission = tmp_path / 'valid.csv'
    hsubmit.write_split(submission, splits, 'valid', [0.0] * len(splits['valid']))

    run = guards.run_starter_script(
        'submit.py', ['--check', '--split', 'valid', str(submission)], timeout=1800)
    assert run.returncode == 0, run.stderr
    assert not run.stdout.isascii(), (
        'fixture assumption broken: this organiser path no longer prints '
        'non-ASCII text, so it no longer exercises the encoding round trip')

    # The two sinks Milestone 2 will use, on real captured output.
    (tmp_path / 'run.md').write_text(run.stdout, encoding='utf-8')
    (tmp_path / 'run.jsonl').write_text(
        json.dumps({'stdout': run.stdout}) + '\n', encoding='utf-8')

    # And the human-only raw log, which run_starter_script wrote itself.
    assert run.raw_log_path is not None
    assert run.raw_log_path.read_text(encoding='utf-8')


# --------------------------------------------------------------------------
# the deny-list
# --------------------------------------------------------------------------

def test_denied_columns_come_from_config():
    denied = guards.denied_columns()
    assert 'play_time_ms' in denied
    assert 'is_click' in denied
    assert 'user_id' not in denied
    assert 'duration_ms' not in denied


def test_allowed_columns_pass():
    guards.assert_columns_allowed(['user_id', 'video_id', 'duration_ms', 'tab', 'date'])


@pytest.mark.parametrize('column', [
    'play_time_ms', 'is_click', 'is_like', 'is_follow', 'is_comment',
    'is_forward', 'is_hate', 'is_profile_enter', 'profile_stay_time',
    'comment_stay_time',
])
def test_every_same_impression_outcome_is_rejected(column):
    """`long_view` is a deterministic function of `play_time_ms` and `duration_ms`,
    so these are label proxies, not features."""
    with pytest.raises(guards.LeakageError):
        guards.assert_columns_allowed(['user_id', column])


def test_frame_check_accepts_anything_with_columns():
    class Frame:
        columns = ['user_id', 'play_time_ms']

    with pytest.raises(guards.LeakageError):
        guards.assert_frame_clean(Frame())
    guards.assert_frame_clean({'user_id': [], 'duration_ms': []})


# --------------------------------------------------------------------------
# the canary
# --------------------------------------------------------------------------

def test_canary_ignores_legitimate_scores():
    for score in (0.4834, 0.5807, 0.6015, 0.65, 0.79):
        assert not guards.check_canary(score, quarantine=False, raise_on_trip=False)


def test_canary_fires_above_the_threshold():
    """The validation oracle ceiling is 0.8484. Anything near it is a bug."""
    with pytest.raises(guards.LeakCanaryError):
        guards.check_canary(0.85, quarantine=False)


def test_canary_quarantines_with_context(tmp_path, monkeypatch):
    monkeypatch.setattr(guards, 'quarantine_dir', lambda: tmp_path)
    with pytest.raises(guards.LeakCanaryError):
        guards.check_canary(0.92, context={'iteration': 7, 'patch': 'bpr_loss'})
    files = list(tmp_path.glob('canary-*.json'))
    assert len(files) == 1
    record = json.loads(files[0].read_text(encoding='utf-8'))
    assert record['val_primary'] == 0.92
    assert record['context']['iteration'] == 7
