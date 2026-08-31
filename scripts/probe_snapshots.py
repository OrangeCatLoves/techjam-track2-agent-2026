"""Do snapshot members carry diversity a seed ensemble does not already have?

Trains at run 4's operating point (batch 2048) and keeps every epoch's
validation predictions, so three things can be compared without touching the
harness:

  1. how much consecutive EPOCHS of one run disagree, against how much SEEDS do
  2. whether blending epochs beats that run's own best epoch
  3. whether seeds x snapshots beats seeds alone -- the only comparison that
     decides anything, since run 4 is already a 5-seed blend

Snapshot members are picked as a fixed window around each run's validation-best
epoch. No epoch is selected on its own score, which would be fitting validation
with extra steps.
"""
import sys, json, time
import numpy as np
from harness import data as hdata
from harness import evaluate as hevaluate
from harness.models import runners as R

SEEDS = (0, 1, 2, 3, 4)
BATCH = 2048
WINDOW = 4          # snapshots per seed

splits = hdata.load()
enc, dim = hdata.encode(splits)
Xtr, ytr, _ = enc['train']
Xva, yva, uva = enc['valid']
groups_tr = R.build_groups(splits, 'train', 'user_id')
groups_va = R.build_groups(splits, 'valid', 'user_id')
loss_fn = R.hlosses.get_loss('pointwise_logloss')

def primary(pred):
    r = hevaluate.evaluate(uva, yva, pred)
    return float(r['primary'])

def train_capturing(seed):
    """train_fm's loop, faithfully, but keeping every epoch's val predictions."""
    model = R.PluggableFM(dim, k=16, lr=0.001, l2=1e-6, seed=seed)
    rng = np.random.default_rng(seed)
    preds, scores = [], []
    best, bad = -1.0, 0
    for epoch in range(1, 41):
        order = rng.permutation(len(ytr))
        for i in range(0, len(order), BATCH):
            model.step_with_loss(Xtr[order[i:i+BATCH]], ytr[order[i:i+BATCH]],
                                 groups_tr[order[i:i+BATCH]], loss_fn)
        p = model.predict(Xva)
        s = primary(p)
        preds.append(p); scores.append(s)
        if s > best + 1e-5:
            best, bad = s, 0
        else:
            bad += 1
            if bad >= 4:
                break
    return preds, scores

t0 = time.time()
all_preds, all_scores, best_idx = [], [], []
for seed in SEEDS:
    p, s = train_capturing(seed)
    b = int(np.argmax(s))
    all_preds.append(p); all_scores.append(s); best_idx.append(b)
    print(f'seed {seed}: {len(s)} epochs, best epoch {b+1} at {s[b]:.4f}', flush=True)

def window(seed_i, k):
    """k epochs nearest this seed's best epoch. Fixed window, later on a tie."""
    p, b = all_preds[seed_i], best_idx[seed_i]
    idx = sorted(range(len(p)), key=lambda i: (abs(i - b), -i))[:k]
    return [p[i] for i in sorted(idx)]

print()
print('--- 1. disagreement: epochs within a seed, vs seeds at their best ---')
epoch_corr = [R._mean_rank_corr(window(i, WINDOW), groups_va) for i in range(len(SEEDS))]
seed_best = [all_preds[i][best_idx[i]] for i in range(len(SEEDS))]
seed_corr = R._mean_rank_corr(seed_best, groups_va)
print(f'  epochs within a seed (mean over seeds) : {np.mean(epoch_corr):.4f}')
print(f'  seeds at their best epoch              : {seed_corr:.4f}')
print(f'  -> snapshots are {"MORE" if np.mean(epoch_corr) > seed_corr else "LESS"} redundant than seeds')

print()
print('--- 2. does blending epochs beat the best epoch, within one seed? ---')
for i, seed in enumerate(SEEDS):
    b = all_scores[i][best_idx[i]]
    sb = primary(R.blend(window(i, WINDOW), groups_va))
    print(f'  seed {seed}: best epoch {b:.4f}  snapshot blend {sb:.4f}  ({sb-b:+.4f})')

print()
print('--- 3. the only comparison that matters: vs run 4 ---')
five_seed = primary(R.blend(seed_best, groups_va))
print(f'  5 seeds, best epoch each  ({len(seed_best):>2} members): {five_seed:.4f}')
for k in (2, 3, 4):
    members = [p for i in range(len(SEEDS)) for p in window(i, k)]
    sc = primary(R.blend(members, groups_va))
    print(f'  5 seeds x {k} snapshots     ({len(members):>2} members): {sc:.4f}  ({sc-five_seed:+.4f})')
for k in (2, 3):
    members = [p for i in range(2) for p in window(i, k)]
    sc = primary(R.blend(members, groups_va))
    print(f'  2 seeds x {k} snapshots     ({len(members):>2} members): {sc:.4f}  ({sc-five_seed:+.4f})  [cheap]')
print()
print(f'elapsed {time.time()-t0:.0f}s')
