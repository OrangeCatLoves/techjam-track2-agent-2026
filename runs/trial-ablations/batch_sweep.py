"""Batch-size sweep on the confirmed winning configuration.

RESULTS.md section 5 establishes that batch 2048 and 5-seed rank-blending
interact: more Adam steps let sparse ID embeddings grow, the seeds converge to
more different solutions, and the blend exploits that diversity. Only 8192 and
2048 were tested. This asks whether the mechanism keeps paying below 2048.

Hand-run ablation, in the same spirit as RESULTS.md section 5. Not an agent
result and must not be reported as one.

Run from the repo root:
    python batch_sweep.py
"""

import time

from harness import data as d
from harness.models import runners as R

SEEDS = (11, 23, 37, 53, 71)          # run 4's seed set, for comparability
BATCHES = (2048, 1024, 512)           # 2048 reproduces the known 0.6034

print('loading splits...')
sp = d.load()

print(f'\n{"batch":>6}  {"blend":>8}  {"best member":>11}  {"gain":>7}  {"secs":>6}')
print('-' * 48)

results = {}
for batch in BATCHES:
    t0 = time.time()
    r = R.train_ensemble(sp, seeds=SEEDS,
                         normalise='within_user_rank',
                         batch=batch, patience=5)
    secs = time.time() - t0

    diag = r.diagnostics.get('ensemble', {}) or {}
    members = diag.get('members') or diag.get('member_scores') or []
    best_member = max(members) if members else float('nan')
    gain = r.val_primary - best_member if members else float('nan')

    results[batch] = r.val_primary
    print(f'{batch:>6}  {r.val_primary:>8.4f}  {best_member:>11.4f}  '
          f'{gain:>+7.4f}  {secs:>6.0f}')

print('\nreference points from RESULTS.md:')
print('  batch 8192, 5-seed blend   0.6020')
print('  batch 2048, 5-seed blend   0.6034   <- current submission')
print('  replacement bar            0.6056   (and confirmed across seed sets)')

best_batch = max(results, key=results.get)
if results[best_batch] > 0.6056:
    print(f'\nbatch {best_batch} at {results[best_batch]:.4f} clears the bar on ONE '
          f'seed set.\nThat is not a result yet. Run confirm_sweep.py before '
          f'claiming anything.')
else:
    print(f'\nBest was batch {best_batch} at {results[best_batch]:.4f}. '
          f'Nothing clears 0.6056.\nRun 4 stands. Record the numbers anyway - a '
          f'measured ceiling on the\nmechanism is a real finding for the report.')