"""Were the ranking losses failing because their training lists were the wrong shape?

Every ranking objective the agent wrote was trained on user_id lists (43.5 rows,
7.8x too long) or user_id+date lists (5.77 rows, but all from ONE day). A scored
validation list is ~5.6 rows spread across ~3 days. eval_matched is the first
grouping that matches both.

Losses are the agent's own, imported from its run patches rather than reimplemented,
so this measures the grouping and nothing else. Hand-run ablation, private
benchmark -- it does not seed any agent run.
"""
import importlib.util, sys, time
from pathlib import Path
import numpy as np
from harness import data as hdata
from harness.models import runners as R

def load_patch(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m

# Deterministic agent losses only. The BPR at agent-explore/iter_001 resamples
# its negatives on every call, so check_loss -- which calls a loss twice and
# compares -- rejects it as sign-inverted. That is a limitation of the check
# against stochastic losses, not a defect in the loss; it passed in the agent's
# own run by luck of the RNG state.
load_patch('runs/trial4/patches/iter_002.py', 'p_bpr')
load_patch('runs/agent-explore/patches/iter_002.py', 'p_lw')
load_patch('runs/agent-explore2/patches/iter_002.py', 'p_lr')
BPR, LW, LR = 'bpr_all_pairs_v1', 'listwise_softmax_ce_v1', 'lambdarank_ndcg_v1'

splits = hdata.load()
GROUPINGS = ('user_id', 'user_id+date', 'eval_matched')

print('--- how much of each grouping can a ranking loss even learn from? ---')
ytr = np.array(hdata.labels(splits, 'train'), dtype=np.float64)
for gb in GROUPINGS:
    g = R.build_groups(splits, 'train', gb)
    _, idx = np.unique(g, return_inverse=True)
    pos = np.bincount(idx, weights=ytr); size = np.bincount(idx).astype(float)
    usable = (pos > 0) & (pos < size)
    rows_usable = usable[idx].mean()
    print(f'  {gb:14} lists {len(size):>7,}  mixed-class lists {usable.mean():6.1%}  '
          f'rows in a usable list {rows_usable:6.1%}')

print()
print('--- validation primary ---')
t0 = time.time()
base = R.train_fm(splits, seed=0)
print(f'  {"pointwise (control)":<34} {base.val_primary:.4f}', flush=True)
results = {}
for label, loss in (('pairwise BPR', BPR), ('listwise softmax', LW),
                    ('lambdarank nDCG', LR)):
    for gb in GROUPINGS:
        r = R.train_fm(splits, seed=0, loss=loss, group_by=gb)
        results[(label, gb)] = r.val_primary
        print(f'  {label + " x " + gb:<34} {r.val_primary:.4f}  '
              f'({r.val_primary - base.val_primary:+.4f})', flush=True)

print()
print('--- did eval_matched beat the best previously-tried grouping? ---')
for label in ('pairwise BPR', 'listwise softmax', 'lambdarank nDCG'):
    old = max(results[(label, g)] for g in ('user_id', 'user_id+date'))
    new = results[(label, 'eval_matched')]
    print(f'  {label:<20} best old {old:.4f}   eval_matched {new:.4f}   ({new-old:+.4f})')
print(f'\nelapsed {time.time()-t0:.0f}s')
