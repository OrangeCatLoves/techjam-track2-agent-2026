"""Confirm a candidate configuration across three independent seed sets.

One measurement is not a result. RESULTS.md section 8 requires any replacement
for run 4 to be confirmed the same way 0.6036 was: re-run across several
independent seed sets, not a single lucky measurement.

Edit BATCH to whatever batch_sweep.py flagged, then run from the repo root:
    python confirm_sweep.py
"""

from statistics import mean

from harness import data as d
from harness.models import runners as R

BATCH = 1024        # <-- set this to the candidate from batch_sweep.py

SEED_SETS = [
    (11, 23, 37, 53, 71),
    (3, 5, 7, 13, 17),
    (2, 4, 8, 16, 32),
]

print('loading splits...')
sp = d.load()

print(f'\nconfirming batch {BATCH}, 5-seed rank-averaged blend\n')
scores = []
for seeds in SEED_SETS:
    r = R.train_ensemble(sp, seeds=seeds,
                         normalise='within_user_rank',
                         batch=BATCH, patience=5)
    scores.append(r.val_primary)
    print(f'  {str(seeds):<24} {r.val_primary:.4f}')

lo, hi = min(scores), max(scores)
print(f'\nmean {mean(scores):.4f} | min {lo:.4f} | max {hi:.4f} | '
      f'spread {hi - lo:.4f}')
print(f'vs run 4 (0.6036): {mean(scores) - 0.6036:+.4f}')

if mean(scores) > 0.6056:
    print('\nClears the bar. Both conditions met: confirmed across three seed '
          'sets\nAND more than 0.002 above 0.6036.')
    print('Record in docs/RESULTS_teammate.md, then ask A to update RESULTS.md '
          'sections 1 and 8.')
else:
    print('\nDoes not clear the bar. Run 4 remains the scored submission.')
    print('Record the numbers anyway - a confirmed negative bounds the '
          'mechanism.')