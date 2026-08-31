"""Would refitting on train + validation actually help? Measured, not guessed.

Q2 asks whether the winning configuration may be retrained on train + validation
before predicting test. We took the conservative reading and disabled it. Nobody
has ruled, but the question of whether it would HELP is answerable with data we
already have -- by shifting the whole protocol one week earlier and running it.

    real     train Apr 8-21   select Apr 22-28   score Apr 29-May 8
    here     train Apr 8-14   select Apr 15-21   score Apr 22-28

Apr 22-28 is the real validation split, and it plays the part of the test set. The
real test set is never touched.

    ARM A, no refit   train on Apr 8-14, early-stop on Apr 15-21, score Apr 22-28
    ARM B, refit      retrain on Apr 8-21 for the epoch count A chose, score the
                      same Apr 22-28

Arm B has no held-out set to stop on, so it reuses A's epoch count -- which is
exactly what a real refit does: keep the selected configuration, add the data.

Both arms share ONE vocabulary, built from Apr 8-21, so the only difference is
which rows carry gradient. That favours arm A slightly, since it is handed ids it
never trained on; if B still wins, the finding is robust in the direction that
matters.

Caveat recorded rather than smoothed over: the period is front-loaded, so here the
refit adds ~28% more rows, where the real one would add ~11%. This measures the
DIRECTION and an upper bound on the size, not the exact gain.
"""
import time
import numpy as np
from harness import data as hdata
from harness import evaluate as hevaluate
from harness.models import runners as R

CUT = 20220415
SEEDS = (0, 1, 2)
BATCH, K, LR, L2, PATIENCE = 8192, 16, 0.001, 1e-6, 4

splits = hdata.load()
enc, dim = hdata.encode(splits)          # vocab from the full train period
Xtr, ytr, _ = enc['train']
Xva, yva, uva = enc['valid']             # Apr 22-28, standing in for test
groups_tr = R.build_groups(splits, 'train', 'user_id')
loss_fn = R.hlosses.get_loss('pointwise_logloss')

dates = np.array([r[hdata.IDX_DATE] for r in splits['train']])
early = np.flatnonzero(dates < CUT)      # Apr 8-14, arm A's training rows
late = np.flatnonzero(dates >= CUT)      # Apr 15-21, arm A's selection rows
allrows = np.arange(len(dates))
print(f'arm A trains on {len(early):,} rows, selects on {len(late):,}')
print(f'arm B trains on {len(allrows):,} rows  (+{len(late)/len(early):.0%})')

def primary(pred, users, labels):
    return float(hevaluate.evaluate(users, labels, pred)['primary'])

sel_users = [splits['train'][i][hdata.IDX_USER] for i in late]
sel_y = ytr[late]

def train(rows, seed, epochs=None, stop_on=None):
    """Train on *rows*. Either early-stop on *stop_on*, or run *epochs* fixed."""
    model = R.PluggableFM(dim, k=K, lr=LR, l2=L2, seed=seed)
    rng = np.random.default_rng(seed)
    best, bad, best_ep, best_state = -1.0, 0, 0, None
    for epoch in range(1, (epochs or 40) + 1):
        order = rng.permutation(len(rows))
        for i in range(0, len(order), BATCH):
            idx = rows[order[i:i + BATCH]]
            model.step_with_loss(Xtr[idx], ytr[idx], groups_tr[idx], loss_fn)
        if stop_on is not None:
            s = primary(model.predict(Xtr[stop_on]), sel_users, sel_y)
            if s > best + 1e-5:
                best, bad, best_ep, best_state = s, 0, epoch, model.state()
            else:
                bad += 1
                if bad >= PATIENCE:
                    break
    if stop_on is not None:
        model.restore(best_state)
        return model, best_ep
    return model, epochs

t0 = time.time()
print()
rows_a, rows_b = [], []
for seed in SEEDS:
    ma, ep = train(early, seed, stop_on=late)
    a = primary(ma.predict(Xva), uva, yva)
    mb, _ = train(allrows, seed, epochs=ep)
    b = primary(mb.predict(Xva), uva, yva)
    rows_a.append(a); rows_b.append(b)
    print(f'  seed {seed}: no refit {a:.4f}   refit {b:.4f}   ({b-a:+.4f})   '
          f'[{ep} epochs]', flush=True)

a, b = np.array(rows_a), np.array(rows_b)
print()
print(f'  ARM A, no refit  mean {a.mean():.4f}  (spread {a.max()-a.min():.4f})')
print(f'  ARM B, refit     mean {b.mean():.4f}  (spread {b.max()-b.min():.4f})')
print(f'  -> refitting is worth {b.mean()-a.mean():+.4f} on this protocol')
print()
print(f'elapsed {time.time()-t0:.0f}s')
