"""Probe: does a watch-time-graded target beat the binary label?  ANSWER: no.

Reproduces the table in docs/RESULTS.md section 5, "Does watch time carry ranking
signal the label throws away?".  Nothing beat the control; the best outcome was a
tie inside the noise band.  Kept so the negative result is reproducible.

    python scripts/probe_watchtime.py            # full sweep, ~12 min
    python scripts/probe_watchtime.py --control  # just the alpha=0 identity check

READ THIS BEFORE IMPORTING ANYTHING HERE
    This module reads ``play_time_ms``, which CLAUDE.md section 7.2 permits as an
    auxiliary training TARGET and forbids as an input feature, and which section 3.1
    says the kit does not load.  Decision D22 reconciles the two.  Three properties
    hold and must keep holding:

      * train rows only -- valid and test are never read for watch time;
      * targets only -- nothing here reaches a feature vector;
      * it lives in scripts/, outside the agent's import surface, so a generated
        patch cannot reach it.

    It is deliberately NOT part of ``harness/``.  Promoting it there would put a
    label-adjacent column one import away from feature code.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness import data as hdata                        # noqa: E402
from harness.models import runners as R                  # noqa: E402

TRAIN_LO, TRAIN_HI = 20220408, 20220421
FILES = ('log_standard_4_08_to_4_21_pure.csv', 'log_standard_4_22_to_5_08_pure.csv')

def _read_train_watch(data_dir):
    """(play_time_ms, duration_ms, user_id, video_id, long_view) in file order."""
    pt, dur, uid, vid, lv = [], [], [], [], []
    for f in FILES:
        with open(f'{data_dir}/{f}', newline='') as fh:
            for r in csv.DictReader(fh):
                d = int(r['date'])
                if TRAIN_LO <= d <= TRAIN_HI:
                    pt.append(float(r['play_time_ms']))
                    dur.append(float(r['duration_ms']))
                    uid.append(r['user_id'])
                    vid.append(r['video_id'])
                    lv.append(1 if r['long_view'] != '0' else 0)
    return (np.array(pt), np.array(dur), uid, vid, np.array(lv, dtype=np.int8))


def load_train_watch(splits=None, data_dir=None):
    """Watch-time arrays aligned position-for-position with splits['train'].

    Raises if alignment fails anywhere. Silent misalignment would poison every
    target in the probe while still producing a plausible score, so this is
    checked exhaustively rather than on a sample.
    """
    splits = splits if splits is not None else hdata.load()
    data_dir = str(hdata.data_dir(data_dir))
    pt, dur, uid, vid, lv = _read_train_watch(data_dir)

    rows = splits['train']
    if len(rows) != len(pt):
        raise RuntimeError(f'row count mismatch: split {len(rows)}, csv {len(pt)}')

    su = np.array([r[hdata.IDX_USER] for r in rows])
    sv = np.array([r[hdata.IDX_VIDEO] for r in rows])
    sd = np.array([r[hdata.IDX_DURATION] for r in rows])
    sl = np.array([r[hdata.IDX_LABEL] for r in rows], dtype=np.int8)

    for name, a, b in (('user_id', su, np.array(uid)),
                       ('video_id', sv, np.array(vid)),
                       ('duration_ms', sd, dur),
                       ('long_view', sl, lv)):
        bad = int((a != b).sum())
        if bad:
            raise RuntimeError(f'{name} misaligned on {bad} of {len(a)} train rows')

    return {'play_time_ms': pt, 'duration_ms': dur, 'long_view': lv}


def threshold(duration_ms):
    """The long_view bar: the video's length, capped at 18 seconds."""
    return np.where(duration_ms <= 18000.0, duration_ms, 18000.0)


def graded_target(watch, alpha: float, cap: float = 0.999):
    """Binary label, with negatives graded by how close they came.

    A positive keeps its 1.0. A negative becomes alpha * (play_time / threshold),
    so a near-miss outranks an instant skip. alpha=0 reproduces the binary label
    exactly, which makes it the control.

    Graded values are capped strictly below 1.0 so a negative can never present
    itself as a positive -- 2.2% of rows have play_time past the bar while still
    being labelled 0, and those must not be promoted.
    """
    y = watch['long_view'].astype(np.float32)
    frac = np.clip(watch['play_time_ms'] / np.maximum(threshold(watch['duration_ms']), 1.0),
                   0.0, cap).astype(np.float32)
    return np.where(y > 0, 1.0, alpha * frac).astype(np.float32)


def retarget(splits, y_new):
    """A copy of *splits* whose train rows carry y_new at index 6.

    valid and test are passed through untouched, so evaluation still uses the
    real long_view labels and the test split stays six fields wide.
    """
    rows = splits['train']
    if len(rows) != len(y_new):
        raise RuntimeError('target length does not match the train split')
    out = dict(splits)
    out['train'] = [r[:hdata.IDX_LABEL] + (float(t),) for r, t in zip(rows, y_new)]
    return out


def graded_target_log(watch, alpha: float, cap: float = 0.999):
    """As graded_target, but the negative's grade is on a log scale.

    Watch time is heavy-tailed, so the linear fraction spends most of its range
    on a handful of long views. log compresses that and spreads the mass where
    the negatives actually sit.
    """
    y = watch['long_view'].astype(np.float32)
    thr = np.maximum(threshold(watch['duration_ms']), 1.0)
    pt = np.clip(watch['play_time_ms'], 0.0, None)
    frac = np.clip(np.log1p(pt) / np.log1p(thr), 0.0, cap).astype(np.float32)
    return np.where(y > 0, 1.0, alpha * frac).astype(np.float32)


def _plans():
    yield 'binary (control)', graded_target, 0.0
    for a in (0.25, 0.5, 1.0):
        yield f'linear a={a}', graded_target, a
    for a in (0.25, 0.5, 1.0):
        yield f'log    a={a}', graded_target_log, a


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--control', action='store_true',
                    help='only check that alpha=0 reproduces the untouched pipeline')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--json', type=Path, default=None)
    args = ap.parse_args(argv)

    splits = hdata.load()
    watch = load_train_watch(splits)
    print(f'watch times aligned to all {len(watch["play_time_ms"]):,} train rows')

    if args.control:
        a = R.train_fm(retarget(splits, graded_target(watch, 0.0)), seed=args.seed)
        b = R.train_fm(splits, seed=args.seed)
        print(f'retargeted alpha=0  {a.val_primary:.4f}')
        print(f'untouched splits    {b.val_primary:.4f}')
        ok = abs(a.val_primary - b.val_primary) < 1e-9
        print('IDENTICAL' if ok else 'MISMATCH -- the retarget path is not a no-op')
        return 0 if ok else 1

    out = []
    for name, fn, alpha in _plans():
        r = R.train_fm(retarget(splits, fn(watch, alpha)), seed=args.seed)
        out.append({'name': name, 'alpha': alpha, 'primary': r.val_primary,
                    'gauc': r.val_gauc, 'ndcg5': r.val_ndcg5})
        print(f'{name:<18} primary {r.val_primary:.4f}  gauc {r.val_gauc:.4f}  '
              f'ndcg5 {r.val_ndcg5:.4f}', flush=True)

    base = out[0]['primary']
    print(f'\n{"":<18} {"primary":>8}  {"vs control":>10}')
    for o in sorted(out, key=lambda d: -d['primary']):
        print(f'{o["name"]:<18} {o["primary"]:>8.4f}  {o["primary"] - base:>+10.4f}')
    if args.json:
        args.json.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
