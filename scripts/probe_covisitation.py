"""Item-item co-visitation: is this video like the ones this user already liked?

Method 3 found the first genuinely diverse blend member -- a tree on dense causal
features, agreeing with the FM at 0.79 where seeds agree at 0.90. Co-visitation is
a second source of information neither model currently has: the FM sees a video as
an id with its own embedding, and the tree sees aggregate rates. Neither sees
"users who liked X also liked Y".

    C[i,j] = users who long-viewed both i and j, over the causal window
    score(u,v) = mean over j in u's prior long-views of
                 C[v,j] / sqrt(pop[v] * pop[j])

The cosine-style denominator stops popular videos dominating every user.

CAUSALITY, which is the whole risk here. C and the user histories are built by the
same expanding walk used in harness/features/base.py: rows dated d are scored from
a matrix and a history containing only dates strictly before d, and only then is
date d folded in. Evaluation rows are scored from the whole train period. A row can
never see its own day, so a row's own label can never reach its own feature.
"""
import time
import numpy as np
from harness import data as hdata
from harness import evaluate as hevaluate
from harness.features import base as fbase
from harness.models import runners as R

SEED = 0
splits = hdata.load()
t0 = time.time()

vids = sorted({r[hdata.IDX_VIDEO] for s in splits.values() for r in s})
vmap = {v: i for i, v in enumerate(vids)}
V = len(vids)
users = sorted({r[hdata.IDX_USER] for s in splits.values() for r in s})
umap = {u: i for i, u in enumerate(users)}
print(f'{V:,} videos, {len(users):,} users')

C = np.zeros((V, V), dtype=np.float32)
pop = np.zeros(V, dtype=np.float32)
hist = [[] for _ in range(len(users))]

def score_rows(rows):
    out = np.zeros(len(rows), dtype=np.float32)
    denom_v = np.sqrt(pop + 1.0)
    for n, r in enumerate(rows):
        h = hist[umap[r[hdata.IDX_USER]]]
        if not h:
            continue
        v = vmap[r[hdata.IDX_VIDEO]]
        j = np.asarray(h, dtype=np.int64)
        out[n] = float(np.mean(C[v, j] / (denom_v[v] * denom_v[j])))
    return out

train = splits['train']
dates = np.array([r[hdata.IDX_DATE] for r in train])
labels = np.array([r[hdata.IDX_LABEL] for r in train])
cov_tr = np.zeros(len(train), dtype=np.float32)

for d in np.unique(dates):
    sel = np.flatnonzero(dates == d)
    cov_tr[sel] = score_rows([train[i] for i in sel])          # BEFORE folding d in
    # fold date d in: every new long-view pairs with the user's prior long-views
    by_user = {}
    for i in sel:
        if labels[i]:
            by_user.setdefault(umap[train[i][hdata.IDX_USER]], []).append(
                vmap[train[i][hdata.IDX_VIDEO]])
    for u, new in by_user.items():
        prior = hist[u]
        new_a = np.asarray(new, dtype=np.int64)
        if prior:
            pa = np.asarray(prior, dtype=np.int64)
            C[np.repeat(new_a, len(pa)), np.tile(pa, len(new_a))] += 1.0
            C[np.tile(pa, len(new_a)), np.repeat(new_a, len(pa))] += 1.0
        if len(new_a) > 1:
            for a in range(len(new_a)):
                for b in range(a + 1, len(new_a)):
                    C[new_a[a], new_a[b]] += 1.0
                    C[new_a[b], new_a[a]] += 1.0
        np.add.at(pop, new_a, 1.0)
        hist[u] = prior + new
    print(f'  {d} folded in ({int(sel.size):,} rows)', flush=True)

cov_va = score_rows(splits['valid'])
print(f'co-visitation built in {time.time()-t0:.0f}s')
print(f'  train nonzero {float((cov_tr>0).mean()):.1%}   valid nonzero {float((cov_va>0).mean()):.1%}')

# --- does it help the tree, which is the promising direction? ---
import lightgbm as lgb
frames, stats = fbase.build_stats(splits)
def matrix(split, extra):
    f, s = frames[split], stats[split]
    cols, names = [], []
    for field in fbase.KEYABLE:
        cols.append(s.label_rate(field)); names.append(f'rate_{field}')
        cols.append(np.log1p(s.exposure_count(field))); names.append(f'logexp_{field}')
    cols.append(s.global_rate()); names.append('global_rate')
    d = np.log1p(np.asarray(f.duration_ms, dtype=np.float64))
    cols.append(d); names.append('log_duration')
    u = np.asarray(f.keys('user_id')); dt = np.asarray(f.date)
    _, g = np.unique(u * (int(dt.max()) + 1) + dt, return_inverse=True)
    cnt = np.bincount(g).astype(np.float64); tot = np.bincount(g, weights=d)
    cols.append(d - (tot / np.maximum(cnt, 1))[g]); names.append('dur_vs_slate')
    cols.append(cnt[g]); names.append('slate_size')
    if extra is not None:
        cols.append(extra); names.append('covisitation')
    return np.column_stack(cols).astype(np.float32), names

ytr = np.array(hdata.labels(splits, 'train'), dtype=np.int32)
yva = np.array(hdata.labels(splits, 'valid'), dtype=np.int32)
uva = hdata.user_ids(splits, 'valid')
groups = R.build_groups(splits, 'valid', 'user_id')
def primary(p): return float(hevaluate.evaluate(uva, yva, p)['primary'])

params = dict(objective='binary', learning_rate=0.05, num_leaves=63,
              min_data_in_leaf=200, feature_fraction=0.9, bagging_fraction=0.8,
              bagging_freq=1, verbose=-1, seed=SEED, num_threads=4)
preds = {}
for tag, extra_tr, extra_va in (('without co-visitation', None, None),
                                ('with co-visitation', cov_tr, cov_va)):
    Xt, names = matrix('train', extra_tr); Xv, _ = matrix('valid', extra_va)
    gbm = lgb.train(params, lgb.Dataset(Xt, label=ytr, feature_name=names),
                    num_boost_round=600,
                    valid_sets=[lgb.Dataset(Xv, label=yva)],
                    callbacks=[lgb.early_stopping(50, verbose=False),
                               lgb.log_evaluation(0)])
    p = gbm.predict(Xv, num_iteration=gbm.best_iteration)
    preds[tag] = p
    print(f'  GBDT {tag:<24} {primary(p):.4f}  ({gbm.best_iteration} trees)', flush=True)
    if extra_tr is not None:
        imp = dict(zip(names, gbm.feature_importance('gain')))
        rank = sorted(imp, key=lambda k: -imp[k]).index('covisitation') + 1
        print(f'    covisitation gain rank {rank} of {len(names)}  ({imp["covisitation"]:,.0f})')

import tempfile, os
tmp = tempfile.mkdtemp()
ens = R.train_ensemble(splits, seeds=(0,1,2,3,4), batch=2048,
                       checkpoint_path=os.path.join(tmp,'e.npz'))
p_ens = R.score_split(R.load_checkpoint(ens.checkpoint, dim=0), splits, 'valid')
base = primary(p_ens)
print()
print(f'  5-seed ensemble alone            {base:.4f}')
for tag, p in preds.items():
    sc = primary(R.blend([p_ens, p], groups, weights=[0.7, 0.3]))
    print(f'  ensemble + GBDT {tag:<22} {sc:.4f}  ({sc-base:+.4f})')
print()
print(f'elapsed {time.time()-t0:.0f}s')
