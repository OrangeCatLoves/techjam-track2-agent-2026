"""The deterministic control run. No LLM, no tokens, real training.

    python scripts/control_run.py                    # to convergence
    python scripts/control_run.py --max_iterations 8

Two jobs, and both matter.

**It is the control.** The fallback is a 30-configuration scripted hyperparameter
search. If it scores as well as the agent, the agent was not adding anything, and we
would rather know that than not. "Our agent beat our own scripted search by X" is far
stronger evidence for Innovation and Impact than "our agent scored Y". Recorded as
D20.

**It is the floor.** It produces a complete, valid submission with no API access at
all. If the LLM is unreachable when the scored run is due, this is what we submit,
and it is honest about being a scripted search rather than the agent: every patch it
writes carries `NOT the agent` in its docstring.

The number this prints is the one the agent has to beat, not 0.6015. Beating the
published baseline while losing to our own fallback would not be a result.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent import llm as agent_llm      # noqa: E402
from agent import loop as agent_loop    # noqa: E402

BASELINE = 0.6015


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--max_iterations', type=int, default=50)
    parser.add_argument('--max_epochs', type=int, default=None,
                        help='cap epochs per experiment; omit for full training')
    parser.add_argument('--run_dir', default=None)
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir) if args.run_dir else (
        REPO_ROOT / 'runs' / f'control-{int(time.time())}')

    print('Deterministic control run (no LLM, no tokens)\n')
    print(f'Run directory : {run_dir}')
    print(f'Baseline      : {BASELINE:.4f}\n')

    started = time.time()
    loop = agent_loop.AgentLoop(run_dir=run_dir,
                                client=agent_llm.LLMClient(provider='none'),
                                max_epochs=args.max_epochs)
    summary = loop.run(max_iterations=args.max_iterations)
    elapsed = time.time() - started

    best = summary['best_val_primary']
    submission = summary['submission']
    ledger = summary['ledger']

    print(f'\n{"=" * 62}')
    print('DETERMINISTIC CONTROL RESULT')
    print('=' * 62)
    print(f'  best validation primary : {best:.4f}' if best else
          '  best validation primary : none')
    if best:
        print(f'  vs baseline {BASELINE:.4f}   : {best - BASELINE:+.4f}')
    print(f'  iterations              : {summary["iterations"]}'
          f' (converged={summary["converged"]}, {summary["reason"]})')
    print(f'  kept / rejected / failed: {ledger["kept"]} / '
          f'{ledger["rejected"]} / {ledger["failed"]}')
    print(f'  wall clock              : {elapsed / 60:.1f} min')
    print(f'  tokens                  : {summary["usage"]["total"]}')
    print(f'  submission              : '
          + (f'{submission["rows"]:,} rows, from iteration '
             f'{submission["from_iteration"]}' if submission.get('written')
             else f'NOT WRITTEN: {submission.get("reason")}'))
    print('=' * 62)
    print('\nThis is the number the agent has to beat, not the published baseline.')
    print(f'Artefacts: {run_dir}')

    (run_dir / 'control_result.json').write_text(
        json.dumps({'best_val_primary': best, 'baseline': BASELINE,
                    'delta_vs_baseline': None if best is None else best - BASELINE,
                    'iterations': summary['iterations'],
                    'converged': summary['converged'],
                    'reason': summary['reason'],
                    'wall_clock_minutes': round(elapsed / 60, 2),
                    'submission': submission, 'ledger': ledger}, indent=2),
        encoding='utf-8')
    return 0 if submission.get('written') else 1


if __name__ == '__main__':
    sys.exit(main())
