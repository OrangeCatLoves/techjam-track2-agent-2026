"""Build the zip to attach to Devpost.

    python scripts/build_submission_bundle.py     # writes submission_bundle.zip

Collects the things a judge would otherwise have to hunt through the repository
for, with the scored CSV first and an orientation note at the top level. The zip
is a build artefact and is not committed -- rerun this to regenerate it.

Nothing here is a hidden-test metric. runs/raw_starter_output/ holds the
organisers' unfiltered output and is deliberately excluded.
"""
from __future__ import annotations

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BEST = 'agent-explore4'          # the scored run
OUT = ROOT / 'submission_bundle.zip'

README_FIRST = """\
TikTok TechJam 2026, Track 2 - Autonomous ML Research Agent
===========================================================

Repository: https://github.com/OrangeCatLoves/techjam-track2-agent-2026

WHAT TO OPEN FIRST
------------------
  dashboard.html      Open in any browser. No server, no install, no network.
                      Every experiment the agent ran, best first. Click any row
                      to read the agent's own hypothesis, verbatim from its log.

WHAT IS IN HERE
---------------
  submission.csv      The scored submission. 170,588 rows, validated with the
                      organisers' own checker.

  dashboard.html      Browsable results (see above).

  README.md           Full project README: setup, architecture, leakage safety,
                      limitations, and how to verify the zero-intervention claim.

  RESULTS.md          Every number, and the eleven directions we closed with a
                      measurement and a mechanism behind each one.

  run4/               The scored run, complete.
      log.md              Human-readable narrative of all 7 iterations.
      log.jsonl           The same, machine-readable.
      patches/            THE CODE THE AGENT WROTE ITSELF, one file per
                          iteration. This is the evidence of autonomy.
      summary.json        Tokens, wall clock, iterations, interventions.
      resources.md        The resource table.
      convergence.json    The stopping rule's state.
      events.jsonl        Non-experiment events. Both entries here are
                          automatic; neither is a human.

  all_run_logs/       The human-readable log from every run, not just the
                      scored one.

HEADLINE NUMBERS
----------------
  Official baseline (given to us)      0.6015
  Our scripted control, no LLM         0.6025
  Our agent, confirmed over 3 seeds    0.6036

  Iterations used          7 of 50
  Wall clock               0.74 h of 6 h
  GPU-hours                0
  Manual interventions     0

The middle row is a 30-configuration scripted search over the same harness with
no LLM at all. We built it so we could ask honestly whether the agent was
reasoning or whether any search would have found the same thing.

NO HIDDEN-TEST METRIC APPEARS ANYWHERE
--------------------------------------
The organisers' loader returns the test split with its true labels attached.
Ours strips them at load, so test rows are six fields wide and feature code
raises IndexError instead of reading a label. tests/test_no_test_labels.py
holds that in place. There is therefore no test score in this bundle, in the
repository, or in any log - by construction rather than by omission.
"""


def main() -> int:
    best = ROOT / 'runs' / BEST
    if not (best / 'submission.csv').is_file():
        print(f'missing {best / "submission.csv"}')
        return 1

    with zipfile.ZipFile(OUT, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        z.writestr('README_FIRST.txt', README_FIRST)
        z.write(best / 'submission.csv', 'submission.csv')
        z.write(ROOT / 'dashboard.html', 'dashboard.html')
        z.write(ROOT / 'README.md', 'README.md')
        z.write(ROOT / 'docs' / 'RESULTS.md', 'RESULTS.md')

        for name in ('log.md', 'log.jsonl', 'summary.json', 'resources.md',
                     'convergence.json', 'events.jsonl', 'ledger.jsonl'):
            src = best / name
            if src.is_file():
                z.write(src, f'run4/{name}')
        for patch in sorted((best / 'patches').glob('*.py')):
            z.write(patch, f'run4/patches/{patch.name}')

        for run in sorted((ROOT / 'runs').iterdir()):
            log = run / 'log.md'
            if log.is_file():
                z.write(log, f'all_run_logs/{run.name}.md')

    size = OUT.stat().st_size
    with zipfile.ZipFile(OUT) as z:
        names = z.namelist()
    print(f'wrote {OUT}  ({size / 1_048_576:.1f} MB, {len(names)} files)')
    for n in names:
        print(f'  {n}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
