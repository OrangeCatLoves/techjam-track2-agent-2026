"""Heterogeneous-batch ensembles.

The batch sweep showed two effects pulling against each other:

    batch   best member   blend    gain
     2048      0.6028    0.6034   +0.0006
     1024      0.6016    0.6027   +0.0012
      512      0.6006    0.6023   +0.0017

Blend gain rises as the batch shrinks (members grow more diverse), while member
quality falls. A homogeneous ensemble has to pick one point on that tradeoff.

Members do not have to share a batch size. Two models trained at different batch
sizes differ by more than two models trained at the same batch with different
seeds, so mixing lets high-quality 2048 members carry the score while lower-batch
members act purely as diversity donors.

Equal weights throughout, chosen once. Tuning blend weights against validation is
how you manufacture a result that does not survive the hidden test -- RESULTS.md
section 2 makes the same commitment.

Hand-run ablation. NOT an agent result. See the note at the bottom before
treating anything here as a submission candidate.

Run from the repo root:
    python hetero_ensemble.py
"""

import tempfile
import time
from pathlib import Path

import numpy as np

from harness import data as hdata
from harness import evaluate as hevaluate
from harness.models import runners as R

SEEDS = (11, 23, 37, 53, 71)          # run 4's seed set, for comparability

# (label, per-member batch sizes). One batch per seed, so all five members are
# distinct models and the comparison against run 4 is like for like.
CONFIGS = [
    ('homogeneous 2048 (run 4)', (2048, 2048, 2048, 2048, 2048)),
    ('4x2048 + 1x1024',          (2048, 2048, 2048, 2048, 1024)),
    ('3x2048 + 2x1024',          (2048, 2048, 2048, 1024, 1024)),
    ('3x2048 + 1x1024 + 1x512',  (2048, 2048, 2048, 1024,  512)),
    ('2x2048 + 2x1024 + 1x512',  (2048, 2048, 1024, 1024,  512)),
]


def run_config(splits, enc, Xva, yva, uva, groups_va, batches):
    """Train one member per (seed, batch) pair and blend them."""
    member_primaries = []
    member_scores = []
    started = time.time()

    with tempfile.TemporaryDirectory() as scratch:
        for seed, batch in zip(SEEDS, batches):
            path = Path(scratch) / f'm_{seed}_{batch}.npz'
            member = R.train_fm(splits, seed=seed, batch=batch, patience=5,
                                with_diagnostics=False, checkpoint_path=path)
            model = R.load_checkpoint_state(member)
            member_primaries.append(member.val_primary)
            member_scores.append(model.predict(Xva))

    blended = R.blend(member_scores, groups_va, normalise='within_user_rank')
    final = hevaluate.evaluate(uva, yva, blended)
    return (float(final['primary']), float(final['GAUC']), float(final['nDCG@5']),
            max(member_primaries), member_primaries, time.time() - started)


print('loading splits...')
splits = hdata.load()
enc, _dim = hdata.encode(splits)
Xva, yva, uva = enc['valid']
groups_va = R.build_groups(splits, 'valid', 'user_id')

print(f'\n{"config":<28} {"blend":>8} {"best mem":>9} {"gain":>8} {"secs":>6}')
print('-' * 64)

results = {}
for label, batches in CONFIGS:
    primary, gauc, ndcg, best_member, members, secs = run_config(
        splits, enc, Xva, yva, uva, groups_va, batches)
    results[label] = primary
    print(f'{label:<28} {primary:>8.4f} {best_member:>9.4f} '
          f'{primary - best_member:>+8.4f} {secs:>6.0f}')
    print(f'{"":28} members: '
          + ', '.join(f'{m:.4f}' for m in members))
    print(f'{"":28} GAUC {gauc:.4f} | nDCG@5 {ndcg:.4f}')

print('\nreference points:')
print('  homogeneous 8192, 5-seed blend   0.6020')
print('  homogeneous 2048, 5-seed blend   0.6034   <- current submission')
print('  target                           0.6050')
print('  replacement bar                  0.6056   (and confirmed across seed sets)')

best_label = max(results, key=results.get)
best_score = results[best_label]
print(f'\nbest: {best_label} at {best_score:.4f}')

if best_score > 0.6034:
    print(f'That is +{best_score - 0.6034:.4f} over run 4 on ONE seed set, which is '
          f'inside the\n+/-0.0008 noise band. It is not a result until it is '
          f'confirmed. Set the\nwinning batch tuple in confirm_hetero.py and run '
          f'that before claiming anything.')
else:
    print('Nothing beat the homogeneous 2048 ensemble. Mixing batch sizes does not '
          'help;\nthe diversity gain does not cover the member-quality loss. '
          'Record it -- a\nclean negative on a well-motivated idea is worth a '
          'paragraph in the report.')

print('\nNOTE: this is a hand-designed configuration, not something the agent '
      'found.\nIf it becomes the scored submission, the "the agent discovered '
      'this" claim\nweakens -- and autonomy is 20% of the grade against the '
      'primary metric\'s\nshare of 35%. Discuss with A before swapping the '
      'submission.')