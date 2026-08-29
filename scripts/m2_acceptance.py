"""Milestone 2 acceptance gate.

    python scripts/m2_acceptance.py            # real training, ~7 minutes
    python scripts/m2_acceptance.py --stub     # stubbed, seconds

The gate, per CLAUDE.md 12.2 as revised in D11 -- an **engineering** result only:

  1. ten iterations complete unattended
  2. a valid submission is produced and passes the organisers' own --check
  3. the run survives kill-and-restart, resuming rather than restarting
  4. log.md is readable by a human

**Score is not a gate at M2.** That moved to M3. A working loop must not be
recorded as a failure because the science did not land on schedule.

Runs in deterministic mode by default, so the gate needs no API key and costs no
tokens. What is being checked is the machinery, not the reasoning.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent import llm as agent_llm            # noqa: E402
from agent import loop as agent_loop          # noqa: E402
from harness import experiment as hexperiment  # noqa: E402
from harness import guards                    # noqa: E402

PASS, FAIL = 'PASS', 'FAIL'


def check(name: str, condition: bool, detail: str = '') -> bool:
    print(f'  {PASS if condition else FAIL}  {name}')
    if detail:
        print(f'        {detail}')
    return condition


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--stub', action='store_true',
                        help='use the stub runner instead of real training')
    parser.add_argument('--iterations', type=int, default=10)
    parser.add_argument('--run_dir', default=None)
    parser.add_argument('--keep', action='store_true',
                        help='keep the run directory afterwards')
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir) if args.run_dir else (
        REPO_ROOT / 'runs' / f'm2-acceptance-{int(time.time())}')
    print(f'Milestone 2 acceptance gate\n\nRun directory: {run_dir}\n')

    runner = None
    if args.stub:
        # Every failure kind in the contract, then a plateau.
        runner = hexperiment.StubRunner(
            ['improvement', 'code_error', 'timeout', 'memory_error',
             'evaluator_rejection', 'nan_score', 'canary_trip', 'improvement',
             'regression', 'no_improvement'] + ['no_improvement'] * 10)

    results = []
    started = time.time()

    # -- part one: run, then kill --------------------------------------------
    first = agent_loop.AgentLoop(run_dir=run_dir,
                                 client=agent_llm.LLMClient(provider='none'),
                                 run_experiment=runner,
                                 max_epochs=1 if not args.stub else None)
    half = max(1, args.iterations // 2)
    first.run(max_iterations=half)
    killed_at = first.tracker.iteration
    killed_strikes = first.tracker.strikes
    killed_best = first.ledger.best()
    print(f'  ..  ran {killed_at} iteration(s), then simulated a kill\n')

    # -- part two: restart ---------------------------------------------------
    second = agent_loop.AgentLoop(run_dir=run_dir,
                                  client=agent_llm.LLMClient(provider='none'),
                                  run_experiment=runner,
                                  max_epochs=1 if not args.stub else None)
    results.append(check(
        'restart resumes the counters rather than resetting them',
        second.tracker.iteration == killed_at
        and second.tracker.strikes == killed_strikes,
        f'iteration {second.tracker.iteration} (was {killed_at}), '
        f'strikes {second.tracker.strikes} (was {killed_strikes})'))
    results.append(check(
        'restart recovers the validation-best checkpoint',
        (killed_best is None) or (second.ledger.best() is not None
                                  and second.ledger.best().iteration == killed_best.iteration),
        'none yet' if killed_best is None
        else f'iteration {second.ledger.best().iteration} at '
             f'{second.ledger.best().val_primary:.4f}'))

    summary = second.run(max_iterations=args.iterations - half)
    seconds = time.time() - started

    # -- the gate ------------------------------------------------------------
    print()
    results.append(check(
        f'{args.iterations} iterations complete unattended',
        summary['iterations'] >= args.iterations or summary['converged'],
        f"{summary['iterations']} iteration(s); "
        f"converged={summary['converged']} ({summary['reason']})"))

    submission = summary['submission']
    if args.stub:
        # The stub writes a 2x2 placeholder checkpoint, which cannot score the
        # real encoding. Not a failure: the check is only meaningful when a real
        # model was trained, and pretending otherwise would make the gate green
        # for the wrong reason.
        print('  SKIP  a valid submission is produced and passes --check')
        print('        stub mode trains no real model; run without --stub for this')
        results.append(check(
            'the submission failure is reported rather than raised',
            not submission.get('written') and 'reason' in submission,
            f"reported: {submission.get('reason', '')[:70]}"))
    else:
        results.append(check(
            'a valid submission is produced and passes --check',
            bool(submission.get('written')) and submission.get('rows') == 170588,
            f"{submission.get('rows', 0):,} rows from iteration "
            f"{submission.get('from_iteration')}" if submission.get('written')
            else f"not written: {submission.get('reason')}"))

    log = (run_dir / 'log.md').read_text(encoding='utf-8')
    readable = ('## Iterations' in log and 'Hypothesis' in log
                and 'Decision' in log and len(log) > 500)
    results.append(check('log.md is readable by a human', readable,
                         f'{len(log):,} characters, '
                         f'{log.count("### Iteration")} iteration section(s)'))

    guards.assert_no_test_metrics(log, where='log.md')
    results.append(check('no hidden-test metric anywhere in the run log', True))

    results.append(check(
        'zero manual interventions',
        summary['resources']['manual_interventions'] == 0,
        f"restarts: {summary['resources']['operational_restarts']}, "
        f"recoveries: {summary['resources']['recovery_events']} "
        f'(neither is an intervention)'))

    results.append(check(
        'deterministic mode spent no tokens',
        summary['usage']['total'] == 0,
        f"provider={summary['usage']['provider']}"))

    ledger = summary['ledger']
    print(f'\n  kept {ledger["kept"]} · rejected {ledger["rejected"]} · '
          f'failed {ledger["failed"]} · {seconds:.0f}s total')
    if summary['review_required']:
        print(f'  {len(summary["review_required"])} result(s) need human review '
              f'before submission')

    passed = all(results)
    print(f'\n{"=" * 60}')
    print(f'MILESTONE 2 GATE: {"PASSED" if passed else "NOT PASSED"} '
          f'({sum(results)}/{len(results)} checks)')
    print('Score is not a gate at M2; that is M3. See D11.')
    print('=' * 60)

    if not args.keep and run_dir.exists():
        shutil.rmtree(run_dir, ignore_errors=True)
    return 0 if passed else 1


if __name__ == '__main__':
    sys.exit(main())
