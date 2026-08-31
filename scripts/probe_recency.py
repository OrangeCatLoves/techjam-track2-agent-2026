"""Does recent training data generalise better than old training data?

Train is 8-21 April, validation 22-28 April, test 29 April - 8 May. If the signal
drifts, rows nearer the boundary are worth more and weighting by recency should
help. Nothing in five runs has ever tested this: of 56 agent iterations, three
targeted sampling and none of them touched time.

Two parts.

PART A, the diagnostic. Train on the early half of the period, then the late half,
with ROW COUNTS MATCHED so the comparison is about recency and not about volume.
If late does not clearly beat early, no recency method can work and part B is
pointless.

PART B, the method. Resample the full training set with weight 2^(-age/half_life)
per row, keeping the row count identical, and sweep the half-life. half_life=inf
is uniform sampling -- the control -- run through the identical resampling path so
that the only difference is the weighting.
"""
import sys, time
import numpy as np
from harness import data as hdata
from harness.models import runners as R

SEED = 0
splits = hdata.load()
train = splits['train']
dates = np.array([r[hdata.IDX_DATE] for r in train])
uniq = np.unique(dates)
print(f'train dates {uniq.min()} .. {uniq.max()}  ({len(uniq)} days, {len(train):,} rows)')
for d in uniq:
    print(f'   {d}  {int((dates==d).sum()):>7,} rows')

def run(rows, tag):
    r = R.train_fm(dict(splits, train=rows), seed=SEED)
    print(f'  {tag:<40} {r.val_primary:.4f}  (train rows {len(rows):,})', flush=True)
    return r.val_primary

t0 = time.time()
print()
print('--- PART A: early vs late, row-matched ---')
cut = uniq[len(uniq) // 2]
early_i = np.flatnonzero(dates < cut)
late_i = np.flatnonzero(dates >= cut)
n = min(len(early_i), len(late_i))
rng = np.random.default_rng(SEED)
# take the LAST n of early and the FIRST n of late, so each arm stays contiguous
# in time; subsampling at random inside an arm would blur what is being compared
e = [train[i] for i in early_i[-n:]]
l = [train[i] for i in late_i[:n]]
print(f'  cut at {cut}: early {len(early_i):,} rows, late {len(late_i):,} rows, matched to {n:,}')
early_p = run(e, f'early half (dates < {cut})')
late_p = run(l, f'late half  (dates >= {cut})')
print(f'  -> late minus early: {late_p - early_p:+.4f}')

print()
print('--- PART B: recency-weighted resampling, row count held constant ---')
age = (dates.max() - dates).astype(np.float64)   # crude day distance, monotone
age_days = np.array([np.searchsorted(uniq, uniq[-1]) - np.searchsorted(uniq, d)
                     for d in dates], dtype=np.float64)
for half_life in (np.inf, 14.0, 7.0, 3.0, 1.5):
    w = np.ones(len(train)) if np.isinf(half_life) else np.power(2.0, -age_days / half_life)
    p = w / w.sum()
    rng = np.random.default_rng(SEED)
    pick = rng.choice(len(train), size=len(train), replace=True, p=p)
    rows = [train[i] for i in pick]
    mean_age = float(age_days[pick].mean())
    label = 'uniform (control)' if np.isinf(half_life) else f'half-life {half_life:g} days'
    sc = run(rows, f'{label}  [mean age {mean_age:.2f}d]')
print()
print(f'elapsed {time.time()-t0:.0f}s')
