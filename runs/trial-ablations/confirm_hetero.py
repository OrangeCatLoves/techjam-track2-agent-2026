"""Confirm a heterogeneous-batch ensemble across three independent seed sets.

One measurement is not a result. RESULTS.md section 8 requires any replacement
for run 4 to be confirmed the way 0.6036 was: several independent seed sets, not
a single lucky measurement. Single-model seed noise is +/-0.0008, so a +0.001
win on one seed set means nothing on its own.

Set BATCHES to whatever hetero_ensemble.py flagged, then run from the repo root:
    python confirm_hetero.py
"""

import tempfile
import time
from pathlib import Path
from statistics import mean

from harness import data as hdata
from harness import evaluate as hevaluate
from harness.models import runners as R

# <-- set this to the winning tuple from hetero_ensemble.py
BATCHES = (2048, 2048, 2048, 1024, 1024)

SEED_SETS = [
    (11, 23, 37, 53, 71),
    (3, 5, 7, 13, 17),
    (2, 4, 8, 16, 32),
]

print('loading splits...')
splits = hdata.load()
enc, _dim = hdata.encode(splits)
Xva, yva, uva = enc['valid']
groups_va = R.build_groups(splits, 'valid', 'user_id')

print(f'\nconfirming batches {BATCHES}, rank-averaged blend, equal weights\n')

scores = []
for seeds in SEED_SETS:
    member_primaries = []
    member_scores = []
    started = time.time()

    with tempfile.TemporaryDirectory() as scratch:
        for seed, batch in zip(seeds, BATCHES):
            path = Path(scratch) / f'm_{seed}_{batch}.npz'
            member = R.train_fm(splits, seed=seed, batch=batch, patience=5,
                                with_diagnostics=False, checkpoint_path=path)
            model = R.load_checkpoint_state(member)
            member_primaries.append(member.val_primary)
            member_scores.append(model.predict(Xva))

    blended = R.blend(member_scores, groups_va, normalise='within_user_rank')
    final = hevaluate.evaluate(uva, yva, blended)
    primary = float(final['primary'])
    scores.append(primary)

    print(f'  {str(seeds):<24} {primary:.4f}   '
          f'best member {max(member_primaries):.4f}   '
          f'({time.time() - started:.0f}s)')

lo, hi = min(scores), max(scores)
avg = mean(scores)
print(f'\nmean {avg:.4f} | min {lo:.4f} | max {hi:.4f} | spread {hi - lo:.4f}')
print(f'vs run 4 (0.6036): {avg - 0.6036:+.4f}')

if avg > 0.6056:
    print('\nClears the bar: confirmed across three seed sets AND more than 0.002 '
          'above 0.6036.')
    print('Record in docs/RESULTS_teammate.md, then ask A to update RESULTS.md '
          'sections 1 and 8.')
    print('Force-add the submission -- the ignore rules exclude them deliberately:')
    print('  git add -f runs/swetha-2/submission.csv')
elif avg > 0.6036:
    print('\nBeats run 4 but does not clear the 0.002 replacement bar.')
    print('Run 4 stays. Report it as a confirmed but sub-threshold improvement -- '
          'that is\nexactly the case the bar exists to catch, and saying so is '
          'the honest read.')
else:
    print('\nDoes not beat run 4. Run 4 remains the scored submission.')
    print('Record the numbers anyway; a confirmed negative bounds the idea.')

if avg > 0.6036:
    spread = hi - lo
    print(f'\nSpread across seed sets is {spread:.4f} against a single-model seed '
          f'sigma of\n0.0008. A tight spread is itself evidence the variance '
          f'reduction is real.')