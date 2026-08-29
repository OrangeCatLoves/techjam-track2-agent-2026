"""ONE real LLM iteration, with everything it did written down for a human to read.

    python scripts/first_contact.py                 # propose only, no training
    python scripts/first_contact.py --train         # also run the experiment
    python scripts/first_contact.py --budget 15000  # tighter ceiling

**This spends real tokens.** It is the first money the project costs, and it exists
because prompt-quality problems are invisible in deterministic mode: cheap to fix at
iteration one, expensive to discover at iteration eight.

It dumps four artefacts into the run directory:

    01_system.txt     the system prompt
    02_prompt.txt     the exact user prompt, including the corpus and the tools
    03_response.txt   the raw text that came back, unparsed
    04_patch.py       the code the agent wrote
    transcript.json   everything, plus token usage

WHAT TO LOOK FOR, in order of how much it tells you:

1. **What did it propose first?** If the opening move is "add more feature fields" or
   "increase the embedding dimension", the corpus is not reaching it: both are
   measured dead ends written into knowledge/methods.md. That is a prompt problem
   and it is the single best early signal of whether the reasoning half works.

2. **Is the hypothesis a claim or a label?** "Pointwise logloss is misaligned with
   within-user ranking, so a pairwise objective should move GAUC more than nDCG" is
   a claim that the next iteration can check. "Try a better loss" is not.

3. **Does the patch actually implement the hypothesis?** A stated pairwise objective
   whose code is pointwise means the agent is describing rather than doing.

4. **Did it ask the data anything?** The analyse tools are offered in the prompt. An
   agent that proposes without ever measuring is guessing more than it needs to.

The budget defaults low on purpose. A single proposal should cost well under 20k
tokens, and a ceiling is cheaper than a surprise.
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

from agent import diagnose as agent_diagnose   # noqa: E402
from agent import llm as agent_llm             # noqa: E402
from agent import propose as agent_propose     # noqa: E402
from harness import analyse as hanalyse        # noqa: E402
from harness import convergence as hconv       # noqa: E402
from harness import data as hdata              # noqa: E402
from harness import experiment as hexperiment  # noqa: E402
from harness import guards                     # noqa: E402
from harness import ledger as hledger          # noqa: E402
from harness import patch as hpatch            # noqa: E402

DEFAULT_BUDGET = 25_000


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--budget', type=int, default=DEFAULT_BUDGET,
                        help=f'token ceiling for this one call (default {DEFAULT_BUDGET})')
    parser.add_argument('--train', action='store_true',
                        help='also run the proposed experiment (adds ~1-2 minutes)')
    parser.add_argument('--max_epochs', type=int, default=None)
    parser.add_argument('--role', default=agent_llm.STRONG,
                        choices=[agent_llm.FAST, agent_llm.STRONG])
    parser.add_argument('--analyse', action='store_true',
                        help='run two analyses first and include them in the prompt')
    parser.add_argument('--run_dir', default=None)
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir) if args.run_dir else (
        REPO_ROOT / 'runs' / f'first-contact-{int(time.time())}')
    run_dir.mkdir(parents=True, exist_ok=True)

    budget = agent_llm.TokenBudget(limit=args.budget, warn_at=args.budget // 2)
    client = agent_llm.LLMClient(budget=budget)
    if not client.enabled:
        print('LLM_PROVIDER is none. Set it in .env to make a real call.')
        return 1

    print(f'First contact\n\nRun directory: {run_dir}')
    print(f'Model role   : {args.role} -> {client.model_for(args.role)}')
    print(f'Token ceiling: {args.budget:,}\n')

    # -- the analyses, if asked for ----------------------------------------
    analyses = []
    if args.analyse:
        print('Running two analyses to put in the prompt...')
        splits = hdata.load()
        for kind, split in (('list_size_profile', 'train'),
                            ('user_composition', 'valid')):
            analyses.append(hanalyse.analyse(kind, split, splits=splits))
        print(f'  {len(analyses)} analysis result(s) included\n')

    # -- the diagnosis, for a run that has not started yet -------------------
    tracker = hconv.ConvergenceTracker.open(run_dir / 'convergence.json')
    diagnosis = agent_diagnose.diagnose(
        hexperiment.ExperimentResult(ok=False, error_kind=None),
        iteration=1, best_primary=None, convergence=tracker.status(), tried=[])

    proposer = agent_propose.Proposer(client)
    prompt = agent_propose.build_prompt(
        diagnosis.to_prompt(), corpus=proposer.corpus,
        capabilities=proposer.capabilities, analyses=analyses or None)

    (run_dir / '01_system.txt').write_text(agent_propose.SYSTEM, encoding='utf-8')
    (run_dir / '02_prompt.txt').write_text(prompt, encoding='utf-8')
    print(f'Prompt built: {len(prompt):,} characters '
          f'(~{agent_llm.estimate_tokens(prompt):,} tokens)\n')

    # -- the call ----------------------------------------------------------
    print('Calling the model...')
    started = time.time()
    try:
        response = client.complete(prompt, system=agent_propose.SYSTEM,
                                   role=args.role, max_tokens=4096)
    except (agent_llm.LLMError, agent_llm.TokenBudgetExceeded) as exc:
        print(f'\nFAILED: {exc}')
        (run_dir / 'transcript.json').write_text(
            json.dumps({'error': str(exc), 'usage': client.usage()}, indent=2),
            encoding='utf-8')
        return 1

    (run_dir / '03_response.txt').write_text(response.text, encoding='utf-8')
    print(f'  {response.input_tokens:,} in / {response.output_tokens:,} out, '
          f'{time.time() - started:.1f}s, attempts {response.attempts}\n')

    # -- parse and validate -------------------------------------------------
    verdict = {'parsed': False, 'valid': False}
    proposal = None
    try:
        proposal = agent_propose.parse(response.text)
        verdict['parsed'] = True
        agent_propose.validate(proposal)
        verdict['valid'] = True
    except agent_propose.ProposalError as exc:
        verdict['error'] = str(exc)
        print(f'REJECTED: {exc}\n')

    if proposal is not None:
        (run_dir / '04_patch.py').write_text(proposal.patch, encoding='utf-8')
        print('--- WHAT IT PROPOSED ---------------------------------------')
        print(f'target stage : {proposal.target_stage}')
        print(f'patch kind   : {proposal.patch_kind}')
        print(f'expected gain: {proposal.expected_gain}')
        print(f'content hash : {proposal.content_hash}')
        print(f'\nhypothesis:\n  {proposal.hypothesis}\n')
        print('patch:')
        for line in proposal.patch.splitlines()[:40]:
            print(f'  {line}')
        if len(proposal.patch.splitlines()) > 40:
            print(f'  ... {len(proposal.patch.splitlines()) - 40} more lines')
        print('------------------------------------------------------------\n')

        # The dead-end check. Cheap, and the best early signal there is.
        lowered = (proposal.hypothesis + ' ' + proposal.patch_kind).lower()
        dead_ends = [phrase for phrase in
                     ('more feature', 'add feature', 'additional feature',
                      'embedding dim', 'larger k', 'increase k', 'more capacity')
                     if phrase in lowered]
        if dead_ends:
            print(f'!! WARNING: the proposal mentions {dead_ends}, which the '
                  f'organisers measured as dead ends and knowledge/methods.md '
                  f'records. The corpus may not be reaching the model.\n')
            verdict['dead_end_warning'] = dead_ends

    # -- optionally run it --------------------------------------------------
    result_summary = None
    if args.train and verdict['valid']:
        print('Running the experiment...')
        ledger = hledger.Ledger(run_dir)
        path = hpatch.write_patch(proposal.patch, ledger.new_patch_path(1))
        result = hexperiment.run_experiment(
            path, 0, checkpoint_path=ledger.checkpoint_path(1),
            max_epochs=args.max_epochs)
        result_summary = {'ok': result.ok, 'usable': result.usable,
                          'val_primary': result.val_primary,
                          'error_kind': result.error_kind,
                          'seconds': round(result.seconds, 1)}
        if result.usable:
            print(f'  val_primary {result.val_primary:.4f} '
                  f'(baseline 0.6015) in {result.seconds:.0f}s\n')
        else:
            print(f'  {result.error_kind}: '
                  f'{(result.error or "")[:200]}\n')
        ledger.archive_patch(1, path)
        ledger.clean_gen()

    transcript = {
        'run_dir': str(run_dir),
        'model': client.model_for(args.role),
        'role': args.role,
        'prompt_chars': len(prompt),
        'response_chars': len(response.text),
        'verdict': verdict,
        'proposal': proposal.as_record() if proposal else None,
        'result': result_summary,
        'usage': client.usage(),
    }
    guards.assert_record_clean(transcript, where='first contact transcript')
    (run_dir / 'transcript.json').write_text(
        json.dumps(transcript, indent=2, default=str), encoding='utf-8')

    usage = client.usage()
    print(f'Tokens: {usage["total"]:,} of {usage["limit"]:,} '
          f'({usage["calls"]} call(s), {usage["failed_calls"]} failed)')
    print(f'\nRead these in order:')
    for name in ('02_prompt.txt', '03_response.txt', '04_patch.py'):
        if (run_dir / name).exists():
            print(f'  {run_dir / name}')
    return 0 if verdict.get('valid') else 1


if __name__ == '__main__':
    sys.exit(main())
