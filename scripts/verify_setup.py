"""End-to-end Milestone 1 verification. Run this after any harness change.

    python scripts/verify_setup.py            # everything, ~7 minutes
    python scripts/verify_setup.py --fast     # skip the FM baseline, ~40 seconds
    python scripts/verify_setup.py --no-tests # skip the pytest run

Checks, in order: environment, data resolution, split row counts, the test-label
strip, the column deny-list, the stdout filter against a real organiser run, the
leak canary, the submission round trip, the convergence rule, the FM contract,
and the pytest suite.

MUST NEVER print a hidden-test metric. Every organiser script it runs goes
through ``harness.guards.run_starter_script``, and the summary it prints is
screened by ``harness.guards.assert_no_test_metrics`` before it reaches stdout.
"""
from __future__ import annotations

import argparse
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from harness import convergence as hconv          # noqa: E402
from harness import data as hdata                 # noqa: E402
from harness import evaluate as hevaluate         # noqa: E402
from harness import guards                        # noqa: E402
from harness import submit as hsubmit             # noqa: E402

EXPECTED_ROWS = {'train': 1141112, 'valid': 124909, 'test': 170588}
FM_VALID_PRIMARY = 0.6015
RANDOM_VALID_PRIMARY = 0.4834

PASS, FAIL, SKIP = 'PASS', 'FAIL', 'SKIP'


class Report:
    """Collects one line per check and prints a screened summary at the end."""

    def __init__(self) -> None:
        self.rows: List[Tuple[str, str, str, float]] = []

    def run(self, name: str, fn: Callable[[], str], *, skip: bool = False) -> Any:
        if skip:
            self.rows.append((name, SKIP, 'skipped by flag', 0.0))
            print(f'  {SKIP}  {name}')
            return None
        started = time.time()
        try:
            detail = fn() or ''
            status = PASS
        except Exception as exc:                    # a failed check is data, not a crash
            detail = f'{type(exc).__name__}: {exc}'.replace('\n', ' ')[:300]
            status = FAIL
        seconds = time.time() - started
        self.rows.append((name, status, detail, seconds))
        print(f'  {status}  {name}  ({seconds:.1f}s)')
        if detail:
            print(f'        {detail}')
        return status

    @property
    def failed(self) -> int:
        return sum(1 for _, status, _, _ in self.rows if status == FAIL)

    def summary(self) -> str:
        width = max(len(name) for name, _, _, _ in self.rows)
        lines = ['', '=' * (width + 30), 'MILESTONE 1 VERIFICATION SUMMARY',
                 '=' * (width + 30)]
        for name, status, detail, seconds in self.rows:
            lines.append(f'{status:4s}  {name:<{width}}  {seconds:6.1f}s')
            if detail:
                lines.append(f'      {detail}')
        total = sum(seconds for _, _, _, seconds in self.rows)
        n_pass = sum(1 for _, s, _, _ in self.rows if s == PASS)
        n_skip = sum(1 for _, s, _, _ in self.rows if s == SKIP)
        lines += ['-' * (width + 30),
                  f'{n_pass} passed, {self.failed} failed, {n_skip} skipped, '
                  f'{total:.1f}s total']
        return '\n'.join(lines)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--fast', action='store_true',
                        help='skip the FM baseline reproduction (the slow check)')
    parser.add_argument('--no-tests', action='store_true',
                        help='skip the pytest run')
    parser.add_argument('--data_dir', default=None)
    args = parser.parse_args(argv)

    report = Report()
    state: dict = {}

    print('Milestone 1 verification\n')

    # -- environment -------------------------------------------------------
    def check_environment() -> str:
        import numpy
        return (f'python {platform.python_version()} on {platform.system()} '
                f'{platform.release()}, numpy {numpy.__version__}')
    report.run('environment', check_environment)

    def check_data_dir() -> str:
        path = hdata.data_dir(args.data_dir)
        if not path.exists():
            raise FileNotFoundError(f'data directory does not exist: {path}')
        csvs = sorted(p.name for p in path.glob('*.csv'))
        if len(csvs) < 6:
            raise FileNotFoundError(f'expected six CSVs in {path}, found {csvs}')
        state['data_dir'] = path
        return f'{path} ({len(csvs)} CSVs)'
    report.run('data directory resolves', check_data_dir)

    if 'data_dir' not in state:
        print(report.summary())
        return 1

    # -- load and row counts ----------------------------------------------
    def check_load() -> str:
        splits = hdata.load(state['data_dir'])
        state['splits'] = splits
        counts = hdata.row_counts(splits)
        if counts != EXPECTED_ROWS:
            raise AssertionError(f'{counts} != {EXPECTED_ROWS}')
        return ', '.join(f'{k} {v:,d}' for k, v in counts.items())
    report.run('split row counts', check_load)

    if 'splits' not in state:
        print(report.summary())
        return 1
    splits = state['splits']

    # -- control 1: the test label is gone --------------------------------
    def check_strip() -> str:
        widths = {name: len(splits[name][0]) for name in hdata.SPLITS}
        if widths != {'train': 7, 'valid': 7, 'test': 6}:
            raise AssertionError(f'tuple widths {widths} are wrong')
        try:
            _ = splits['test'][0][hdata.IDX_LABEL]
        except IndexError:
            pass
        else:
            raise AssertionError('a test row still yields a label at index 6')
        for refuse in (lambda: hdata.labels(splits, 'test'),
                       lambda: hevaluate.evaluate_split(splits, 'test', []),
                       lambda: hsubmit.score('x.csv', 'test', splits)):
            try:
                refuse()
            except hdata.TestLabelAccessError:
                continue
            raise AssertionError('a test-label accessor did not refuse')
        return ('test rows are 6-tuples; index 6 raises IndexError; '
                'labels/evaluate/score all refuse the test split')
    report.run('test labels stripped and unreachable', check_strip)

    # -- control 2: the deny-list -----------------------------------------
    def check_denylist() -> str:
        denied = guards.denied_columns()
        if 'play_time_ms' not in denied:
            raise AssertionError('play_time_ms is not on the deny-list')
        guards.assert_columns_allowed(['user_id', 'video_id', 'duration_ms', 'tab'])
        try:
            guards.assert_columns_allowed(['user_id', 'play_time_ms'])
        except guards.LeakageError:
            return f'{len(denied)} denied columns; injected play_time_ms rejected'
        raise AssertionError('the deny-list let play_time_ms through')
    report.run('column deny-list', check_denylist)

    # -- control 3: the stdout filter, against a real organiser run -------
    def check_stdout_filter() -> str:
        run = guards.run_starter_script(
            'baseline.py', ['--model', 'random', '--seed', '0'],
            data_dir=state['data_dir'], timeout=1800)
        if run.returncode != 0:
            raise RuntimeError(f'baseline.py --model random failed: {run.stderr[:200]}')
        guards.assert_no_test_metrics(run.stdout, where='filtered stdout')
        if run.redacted_lines < 1:
            raise AssertionError('nothing was redacted; the filter is not working')
        primary = float(run.stdout.split('primary')[-1].split()[0])
        if abs(primary - RANDOM_VALID_PRIMARY) > 0.002:
            raise AssertionError(f'random valid primary {primary:.4f} is not '
                                 f'{RANDOM_VALID_PRIMARY} +/- 0.002')
        state['random_primary'] = primary
        # Careful: the summary is itself screened, so no line here may name the
        # held-out split next to a metric word.
        return (f'{run.redacted_lines} line(s) redacted; '
                f'random valid primary {primary:.4f}' + '\n      ' +
                f'raw organiser output kept human-only at '
                f'{run.raw_log_path.relative_to(REPO_ROOT)}')
    report.run('starter stdout filter (live run)', check_stdout_filter)

    # -- control 4: the leak canary ---------------------------------------
    def check_canary() -> str:
        threshold = guards.canary_threshold()
        if guards.check_canary(0.6015, raise_on_trip=False, quarantine=False):
            raise AssertionError('the canary fired on a legitimate baseline score')
        try:
            guards.check_canary(0.95, context={'check': 'verify_setup'},
                                quarantine=False)
        except guards.LeakCanaryError:
            return (f'threshold {threshold:.2f}; 0.6015 passes, 0.95 quarantined '
                    f'(validation oracle ceiling is 0.8484)')
        raise AssertionError('the canary did not fire at 0.95')
    report.run('leak canary', check_canary)

    # -- the submission round trip ----------------------------------------
    def check_submission() -> str:
        import numpy as np
        workspace = REPO_ROOT / 'workspace'
        workspace.mkdir(exist_ok=True)
        path = workspace / 'verify_valid_submission.csv'
        scores = np.random.default_rng(0).random(len(splits['valid']))
        hsubmit.write_split(path, splits, 'valid', scores)
        result = hsubmit.score(path, 'valid', splits)
        direct = hevaluate.evaluate_split(splits, 'valid', scores)
        if abs(result['primary'] - direct['primary']) > 1e-6:
            raise AssertionError('disk round trip disagrees with in-memory scoring')
        if abs(result['primary'] - RANDOM_VALID_PRIMARY) > 0.002:
            raise AssertionError(f'random submission scored {result["primary"]:.4f}')
        test_path = workspace / 'verify_test_submission.csv'
        hsubmit.write_split(test_path, splits, 'test', [0.0] * len(splits['test']))
        hsubmit.check(test_path, 'test', splits)
        return (f'{hevaluate.format_result(result, "valid")}' + '\n      ' +
                f'test-split submission of {len(splits["test"]):,d} rows '
                f'passes --check')
    report.run('submission write / check / score', check_submission)

    # -- the convergence rule ---------------------------------------------
    def check_convergence() -> str:
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'convergence.json'
            clock = [0.0]
            tracker = hconv.ConvergenceTracker.open(path, clock=lambda: clock[0])
            tracker.start_session()
            for score in (0.6015, 0.6100, 0.6110, 0.6115, 0.6118):
                status = tracker.record_iteration(score)
            if not (status.converged and status.reason == hconv.REASON_NO_IMPROVEMENT):
                raise AssertionError(f'expected convergence, got {status}')
            if status.strikes != 3 or status.iteration != 5:
                raise AssertionError(f'strikes {status.strikes} iteration {status.iteration}')
            tracker.end_session()
            resumed = hconv.ConvergenceTracker.open(path, clock=lambda: 0.0)
            if (resumed.iteration, resumed.strikes) != (5, 3):
                raise AssertionError('restart did not resume the counters')
            try:
                resumed.record_iteration(0.9)
            except hconv.ConvergedError:
                pass
            else:
                raise AssertionError('iteration accepted after convergence')
        return (f'epsilon {tracker.epsilon}, N {tracker.n_consecutive}, '
                f'comparison {tracker.comparison}; stops on strike 3, resumes on '
                f'restart, refuses to continue afterwards')
    report.run('convergence rule', check_convergence)

    # -- the FM contract ---------------------------------------------------
    def check_fm() -> str:
        run = guards.run_starter_script(
            'baseline.py', ['--model', 'fm', '--seed', '0'],
            data_dir=state['data_dir'], timeout=3600)
        if run.returncode != 0:
            raise RuntimeError(f'baseline.py --model fm failed: {run.stderr[:200]}')
        guards.assert_no_test_metrics(run.stdout, where='filtered stdout')
        primary = float(run.stdout.split('primary')[-1].split()[0])
        if abs(primary - FM_VALID_PRIMARY) > 0.001:
            raise AssertionError(f'FM valid primary {primary:.4f} is not '
                                 f'{FM_VALID_PRIMARY} +/- 0.001')
        state['fm_primary'] = primary
        return (f'FM valid primary {primary:.4f} (published 0.6016); '
                f'{run.redacted_lines} line(s) redacted; {run.seconds:.0f}s')
    report.run('FM baseline reproduces (the number to beat)', check_fm,
               skip=args.fast)

    # -- the test suite ----------------------------------------------------
    def check_pytest() -> str:
        argv = [sys.executable, '-m', 'pytest', 'tests/', '-q']
        if args.fast:
            argv += ['-m', 'not slow']
        proc = subprocess.run(argv, cwd=str(REPO_ROOT), capture_output=True,
                              text=True, encoding='utf-8', errors='replace')
        tail = (proc.stdout or '').strip().splitlines()
        summary = tail[-1] if tail else '(no output)'
        if proc.returncode != 0:
            raise AssertionError(f'pytest failed: {summary}')
        return summary
    report.run('pytest suite', check_pytest, skip=args.no_tests)

    summary, redacted = guards.filter_stdout(report.summary())
    if redacted:
        summary += ('\n\n'
                    f'WARNING: {redacted} line(s) of this summary were redacted; '
                    f'a check leaked a hidden-test metric into its own output. '
                    f'Fix that before trusting this run.')
    guards.assert_no_test_metrics(summary, where='verification summary')
    print(summary)

    if report.failed:
        print('\nMilestone 1 is NOT verified. Fix the failures above.')
        return 1
    print('\nMilestone 1 verified. The number to beat is validation primary '
          f'{state.get("fm_primary", FM_VALID_PRIMARY):.4f}.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
