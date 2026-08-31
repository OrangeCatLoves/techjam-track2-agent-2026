"""A gradient-boosted tree on the causal features, alone and blended with the FM.

CLAUDE.md section 9.3 calls a GBDT a trap here, and the reasoning is sound: a tree
cannot split on 27,285 user ids crossed with 7,538 videos, which is exactly what
the FM's embeddings represent. But that argument is about IDs. Now that
harness/features/ exists, the same information is available as DENSE causal
statistics, which is the one representation a tree handles well.

Two things are being measured, and the second is the one that matters:

  1. the tree alone -- expected to be weak, and a useful ablation either way
  2. the tree blended with the FM -- a genuinely different model family, which is
     the "comparable quality, differently wrong" ingredient that model-config
     diversity, heterogeneous batches and snapshots all failed to supply

Every feature is built through harness.features.base, so the causal window is the
one already tested by tests/test_causal_encoding.py: a train row on date d sees
only dates strictly before d, and an evaluation row sees the train period.
"""
import time
import numpy as np
import lightgbm as lgb
from harness import data as hdata
from harness import evaluate as hevaluate
from harness.features import base as fbase
from harness.models import runners as R

SEED = 0
splits = hdata.load()
frames, stats = fbase.build_stats(splits)

def matrix(split):
    f, s = frames[split], stats[split]
    cols, names = [], []
    for field in fbase.KEYABLE:
        cols.append(s.label_rate(field));      names.append(f'rate_{field}')
        cols.append(np.log1p(s.exposure_count(field))); names.append(f'logexp_{field}')
    cols.append(s.global_rate());              names.append('global_rate')
    d = np.log1p(np.asarray(f.duration_ms, dtype=np.float64))
    cols.append(d);                            names.append('log_duration')
    # duration against the slate it was shown in -- resolves inside a duration
    # bucket, and reads only duration, never an outcome
    users = np.asarray(f.keys('user_id')); dates = np.asarray(f.date)
    _, g = np.unique(users * (int(dates.max()) + 1) + dates, return_inverse=True)
    cnt = np.bincount(g).astype(np.float64); tot = np.bincount(g, weights=d)
    cols.append(d - (tot / np.maximum(cnt, 1))[g]); names.append('dur_vs_slate')
    cols.append(cnt[g]);                       names.append('slate_size')
    return np.column_stack(cols).astype(np.float32), names

t0 = time.time()
Xtr, names = matrix('train'); Xva, _ = matrix('valid')
ytr = np.array(hdata.labels(splits, 'train'), dtype=np.int32)
yva = np.array(hdata.labels(splits, 'valid'), dtype=np.int32)
uva = hdata.user_ids(splits, 'valid')
print(f'{Xtr.shape[1]} dense causal features on {Xtr.shape[0]:,} train rows')

def primary(pred):
    return float(hevaluate.evaluate(uva, yva, pred)['primary'])

print()
print('--- 1. the tree alone ---')
params = dict(objective='binary', learning_rate=0.05, num_leaves=63,
              min_data_in_leaf=200, feature_fraction=0.9, bagging_fraction=0.8,
              bagging_freq=1, verbose=-1, seed=SEED, num_threads=4)
ds = lgb.Dataset(Xtr, label=ytr, feature_name=names)
dv = lgb.Dataset(Xva, label=yva, reference=ds)
evals = {}
gbm = lgb.train(params, ds, num_boost_round=600, valid_sets=[dv],
                callbacks=[lgb.early_stopping(50, verbose=False),
                           lgb.log_evaluation(0)])
p_gbdt = gbm.predict(Xva, num_iteration=gbm.best_iteration)
gbdt_score = primary(p_gbdt)
print(f'  GBDT binary            {gbdt_score:.4f}   ({gbm.best_iteration} trees)', flush=True)

print()
print('  top features by gain:')
imp = sorted(zip(names, gbm.feature_importance('gain')), key=lambda t: -t[1])
for n, v in imp[:8]:
    print(f'    {n:<22} {v:>12,.0f}')

print()
print('--- 2. the FM, the blend, and a 5-seed ensemble blend ---')
import tempfile, os
tmp = tempfile.mkdtemp()
fm = R.train_fm(splits, seed=SEED, checkpoint_path=os.path.join(tmp, 'fm.npz'))
fm_model = R.load_checkpoint(fm.checkpoint, dim=0)
p_fm = R.score_split(fm_model, splits, 'valid')
groups = R.build_groups(splits, 'valid', 'user_id')
print(f'  FM alone               {primary(p_fm):.4f}')

agree = R._mean_rank_corr([p_fm, p_gbdt], groups)
print(f'  FM vs GBDT agreement   {agree:.4f}   (5 FM seeds agree at ~0.90)')

print()
best = (None, -1.0)
for w in (0.1, 0.2, 0.3, 0.4, 0.5):
    p = R.blend([p_fm, p_gbdt], groups, weights=[1.0 - w, w])
    sc = primary(p)
    if sc > best[1]: best = (w, sc)
    print(f'  FM + GBDT  (gbdt weight {w:.1f})   {sc:.4f}   ({sc - primary(p_fm):+.4f} vs FM)')

print()
print('--- 3. against run 4: does the tree add to a 5-seed ensemble? ---')
ens = R.train_ensemble(splits, seeds=(0, 1, 2, 3, 4), batch=2048,
                       checkpoint_path=os.path.join(tmp, 'ens.npz'))
ens_model = R.load_checkpoint(ens.checkpoint, dim=0)
p_ens = R.score_split(ens_model, splits, 'valid')
base = primary(p_ens)
print(f'  5-seed ensemble alone  {base:.4f}')
for w in (0.1, 0.2, 0.3):
    sc = primary(R.blend([p_ens, p_gbdt], groups, weights=[1.0 - w, w]))
    print(f'  ensemble + GBDT (w {w:.1f})    {sc:.4f}   ({sc - base:+.4f})')

print()
print(f'elapsed {time.time()-t0:.0f}s')
