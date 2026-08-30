"""Model training, against label-stripped splits. The engine room.

OWNS
    - the FM training loop, reimplemented so that it never touches the test split
    - the pluggable-objective step, so an agent-written loss drops in without the
      loop changing
    - the diagnostics contract from ``docs/M2_CONTRACT.md`` section 2
    - checkpoint save and restore, which is what makes keep-or-reject possible

MUST NEVER
    - call ``starter.baseline.run_fm``. That function ends with
      ``'test': evaluate(ute, yte, m.predict(Xte))`` and returns a hidden-test
      metric. This module exists because that one line makes the organisers'
      trainer unusable to us (CLAUDE.md section 5, control 3)
    - evaluate any split other than train and valid
    - select on anything except validation primary

WHAT IS REUSED VS REIMPLEMENTED
    The **model** is reused: ``PluggableFM`` subclasses ``starter.baseline.FM`` and
    inherits ``logits`` and ``predict`` unchanged, so the arithmetic cannot drift
    from the organisers'.

    The **trainer** is reimplemented, because theirs computes test metrics and
    because the objective has to become swappable. ``step_with_loss`` duplicates
    twelve lines of their Adam update; that duplication is validated, not assumed,
    by ``tests/test_runners.py``, which asserts this loop reproduces the published
    validation primary of 0.6015 with the pointwise objective.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

from harness import data as hdata
from harness import evaluate as hevaluate
from harness import guards
from harness import losses as hlosses

#: Feature field names, in the column order of the encoded X matrix.
FIELDS: Sequence[str] = ('user_id', 'video_id', 'author_id', 'tab', 'dur_bucket')

#: How many users to sample when measuring the train/validation gap. Scoring all
#: 1.14M train rows costs ~15 s per call for a number used only as a signal.
GAP_SAMPLE_USERS = 4000

#: Measured wall clock for a full reference FM run on this machine, in seconds.
#: Reported alongside every experiment's own cost so that "expensive" is a
#: comparison the agent can make rather than a word it has to interpret.
REFERENCE_FM_SECONDS = 63.0


@dataclass
class TrainResult:
    """Everything a training run is allowed to tell the outside world."""
    val_gauc: float
    val_ndcg5: float
    val_primary: float
    diagnostics: Dict[str, Any]
    checkpoint: str | None
    seconds: float
    seed: int
    epochs_run: int
    best_epoch: int
    epoch_history: List[Dict[str, float]] = field(default_factory=list)

    def as_metrics(self) -> Dict[str, float]:
        return {'val_gauc': self.val_gauc, 'val_ndcg5': self.val_ndcg5,
                'val_primary': self.val_primary}


def _build_pluggable_fm():
    """Subclass the organisers' FM without importing it at module scope.

    ``harness.data`` owns putting ``starter/`` on ``sys.path``, so that side effect
    stays in one place; this call is what triggers it.
    """
    hdata.starter_data_module()
    import baseline as starter_baseline

    class _PluggableFM(starter_baseline.FM):
        """The organisers' FM with a swappable objective.

        ``logits`` and ``predict`` are inherited unchanged. Only the parameter
        update is restated, because the gradient now comes from a loss function
        rather than being hardcoded to logistic regression.
        """

        def step_with_loss(self, X, y, groups, loss_fn) -> float:
            z, E, S = self.logits(X)
            loss, dz = loss_fn(z, y, groups)
            g = np.asarray(dz, dtype=np.float32)
            if g.shape != z.shape:
                raise hlosses.LossError(
                    f'loss returned a gradient of shape {g.shape}, expected {z.shape}')
            if not np.all(np.isfinite(g)):
                raise hlosses.LossError('loss returned a non-finite gradient')

            # --- identical to starter.baseline.FM.step from here down ---
            gV = np.zeros_like(self.V)
            gW = np.zeros_like(self.W)
            np.add.at(gW, X, g[:, None])
            np.add.at(gV, X, g[:, None, None] * (S[:, None, :] - E))
            gV += self.l2 * self.V
            gW += self.l2 * self.W
            self.t += 1
            b1, b2, eps = 0.9, 0.999, 1e-8
            for P, G, M, Vv in ((self.V, gV, self.mV, self.vV),
                                (self.W, gW, self.mW, self.vW)):
                M *= b1
                M += (1 - b1) * G
                Vv *= b2
                Vv += (1 - b2) * (G * G)
                P -= self.lr * (M / (1 - b1 ** self.t)) / (
                    np.sqrt(Vv / (1 - b2 ** self.t)) + eps)
            self.b -= self.lr * g.sum()
            return float(loss)

        def state(self):
            """A deep copy of the trainable parameters."""
            return (self.V.copy(), self.W.copy(), np.float32(self.b))

        def restore(self, state) -> None:
            self.V, self.W, self.b = state[0].copy(), state[1].copy(), np.float32(state[2])

    return _PluggableFM


PluggableFM = _build_pluggable_fm()


# --------------------------------------------------------------------------
# grouping
# --------------------------------------------------------------------------

def build_groups(splits: Dict[str, list], split: str,
                 group_by: str = 'user_id') -> np.ndarray:
    """Integer group id per row, defining the ranking lists for a listwise loss.

    ``user_id`` gives ~42 rows per list on train; ``user_id+date`` gives ~3. The
    evaluation lists are ~6, so neither matches, and which is better is a genuine
    experiment rather than something to guess. See CLAUDE.md section 9.2.
    """
    rows = splits[split]
    if group_by == 'user_id':
        keys = [r[hdata.IDX_USER] for r in rows]
    elif group_by == 'user_id+date':
        keys = [(r[hdata.IDX_USER], r[hdata.IDX_DATE]) for r in rows]
    else:
        raise ValueError(f'unknown group_by {group_by!r}; '
                         f"choose 'user_id' or 'user_id+date'")
    index: Dict[Any, int] = {}
    out = np.empty(len(keys), dtype=np.int64)
    for i, key in enumerate(keys):
        gid = index.get(key)
        if gid is None:
            gid = index[key] = len(index)
        out[i] = gid
    return out


# --------------------------------------------------------------------------
# diagnostics
# --------------------------------------------------------------------------

def field_contributions(model, X: np.ndarray) -> Dict[str, Dict[str, float]]:
    """The FM's equivalent of feature importance, per feature field.

    ``mean_abs_w`` is the first-order pull of the field; ``mean_v_norm`` is how
    much it participates in crosses. A field near zero on both is dead weight,
    which is the measurable form of the organisers' result that pure user-side
    first-order terms contribute exactly zero to within-user ranking.

    Column ``i`` of ``X`` holds only ids belonging to field ``i``, so the ids in
    use are read off the data rather than from encoder offsets the organisers'
    ``encode()`` does not return.
    """
    out: Dict[str, Dict[str, float]] = {}
    for i, name in enumerate(FIELDS[:X.shape[1]]):
        ids = np.unique(X[:, i])
        out[name] = {
            'mean_abs_w': float(np.abs(model.W[ids]).mean()),
            'mean_v_norm': float(np.linalg.norm(model.V[ids], axis=1).mean()),
            'n_ids': int(ids.size),
        }
    return out


def _train_gap_score(model, Xtr, ytr, groups_tr, seed: int) -> float | None:
    """Validation-style primary on a deterministic sample of training users.

    Sampled by user rather than by row, because GAUC is computed per user and a
    row sample would shred the lists it is averaged over.
    """
    unique = np.unique(groups_tr)
    if unique.size == 0:
        return None
    rng = np.random.default_rng(seed)
    take = unique if unique.size <= GAP_SAMPLE_USERS else rng.choice(
        unique, size=GAP_SAMPLE_USERS, replace=False)
    mask = np.isin(groups_tr, take)
    if not mask.any():
        return None
    scores = model.predict(Xtr[mask])
    users = groups_tr[mask].tolist()
    return float(hevaluate.evaluate(users, ytr[mask].tolist(), scores)['primary'])


# --------------------------------------------------------------------------
# the trainer
# --------------------------------------------------------------------------

def train_fm(splits: Dict[str, list] | None = None,
             *,
             k: int = 16,
             lr: float = 0.001,
             l2: float = 1e-6,
             batch: int = 8192,
             max_epochs: int = 40,
             patience: int = 4,
             seed: int = 0,
             loss: str | hlosses.LossFn = 'pointwise_logloss',
             group_by: str = 'user_id',
             checkpoint_path: str | Path | None = None,
             with_diagnostics: bool = True,
             verbose: bool = False) -> TrainResult:
    """Train an FM and return validation results plus diagnostics.

    Selection is on validation primary and nothing else. The returned model state
    is the **validation-best** epoch, restored, matching the competition's
    definition of the scored checkpoint.

    Defaults reproduce ``baseline_scores.json``'s ``fm_official`` config, so
    ``train_fm(seed=0)`` must give validation primary 0.6015.
    """
    splits = splits if splits is not None else hdata.load()
    loss_fn = hlosses.get_loss(loss)

    # Validate the objective on 64 synthetic rows before spending a minute on a
    # real training run. Catches a sign inversion and a grouping-blind pairwise
    # or listwise loss -- both of which train happily and produce a plausible
    # number, which is exactly what makes them expensive to find later.
    loss_report = hlosses.check_loss(loss_fn, kind=hlosses.loss_kind(loss))

    enc, dim = hdata.encode(splits)
    Xtr, ytr, _ = enc['train']
    Xva, yva, uva = enc['valid']

    groups_tr = build_groups(splits, 'train', group_by)

    model = PluggableFM(dim, k=k, lr=lr, l2=l2, seed=seed)
    rng = np.random.default_rng(seed)

    started = time.time()
    best_primary, best_state, best_epoch, bad = -1.0, None, 0, 0
    history: List[Dict[str, float]] = []
    epochs_run = 0

    for epoch in range(1, max_epochs + 1):
        order = rng.permutation(len(ytr))
        epoch_losses = [
            model.step_with_loss(Xtr[order[i:i + batch]],
                                 ytr[order[i:i + batch]],
                                 groups_tr[order[i:i + batch]],
                                 loss_fn)
            for i in range(0, len(order), batch)
        ]
        result = hevaluate.evaluate(uva, yva, model.predict(Xva))
        epochs_run = epoch
        history.append({'epoch': epoch,
                        'loss': float(np.mean(epoch_losses)),
                        'val_gauc': float(result['GAUC']),
                        'val_ndcg5': float(result['nDCG@5']),
                        'val_primary': float(result['primary'])})
        if verbose:
            print(f"  epoch {epoch:2d} | loss {np.mean(epoch_losses):.4f} | "
                  f"valid primary {result['primary']:.4f}")

        # The organisers' early-stop rule, kept identical: a strictly better
        # primary by more than 1e-5 resets patience.
        if result['primary'] > best_primary + 1e-5:
            best_primary, bad, best_epoch = float(result['primary']), 0, epoch
            best_state = model.state()
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is None:
        raise RuntimeError('training produced no validation-best state')
    model.restore(best_state)

    final = hevaluate.evaluate(uva, yva, model.predict(Xva))
    seconds = time.time() - started

    # The leak canary, on the only score that can select anything.
    guards.check_canary(float(final['primary']),
                        context={'runner': 'train_fm', 'seed': seed,
                                 'loss': getattr(loss_fn, '__name__', str(loss)),
                                 'k': k, 'lr': lr, 'group_by': group_by})

    diagnostics: Dict[str, Any] = {}
    if with_diagnostics:
        train_primary = _train_gap_score(model, Xtr, ytr, groups_tr, seed)
        diagnostics = {
            'metrics': {'val_gauc': float(final['GAUC']),
                        'val_ndcg5': float(final['nDCG@5']),
                        'val_primary': float(final['primary'])},
            'fit': {'train_primary': train_primary,
                    'val_primary': float(final['primary']),
                    'gap': (None if train_primary is None
                            else train_primary - float(final['primary'])),
                    'epochs_run': epochs_run,
                    'best_epoch': best_epoch},
            'fields': field_contributions(model, Xtr),
            'lists': {'group_by': group_by,
                      'train_groups': int(np.unique(groups_tr).size),
                      'mean_train_list_size': float(len(groups_tr)
                                                    / max(1, np.unique(groups_tr).size)),
                      'valid_users': int(final['users']),
                      'mean_valid_list_size': float(final['rows'] / max(1, final['users']))},
            # What this experiment cost, with a reference point. "Expensive" has
            # to mean something concrete to the agent rather than be inferred:
            # a full reference FM is ~63 s, so the six-hour ceiling is dominated
            # by LLM latency and overhead, not by training compute.
            'cost': {'seconds': round(seconds, 1),
                     'reference_fm_seconds': REFERENCE_FM_SECONDS,
                     'relative_to_reference': round(seconds / REFERENCE_FM_SECONDS, 2),
                     'timeout_minutes': float(hdata.load_config()
                                              .get('agent', {})
                                              .get('per_iteration_timeout_minutes', 12))},
            'objective': {'kind': loss_report.get('kind'),
                          'name': loss if isinstance(loss, str)
                          else getattr(loss_fn, '__name__', 'callable')},
        }
        guards.assert_record_clean(diagnostics, where='train_fm diagnostics')

    saved: str | None = None
    if checkpoint_path is not None:
        saved = str(save_checkpoint(model, checkpoint_path))

    return TrainResult(val_gauc=float(final['GAUC']),
                       val_ndcg5=float(final['nDCG@5']),
                       val_primary=float(final['primary']),
                       diagnostics=diagnostics,
                       checkpoint=saved,
                       seconds=seconds,
                       seed=seed,
                       epochs_run=epochs_run,
                       best_epoch=best_epoch,
                       epoch_history=history)


# --------------------------------------------------------------------------
# checkpoints -- what makes keep-or-reject possible
# --------------------------------------------------------------------------

def rank_normalise(scores: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Replace each score with its rank inside its own group, scaled to [0, 1].

    On a **single** model this is a no-op for the metrics: it is monotone within
    each list, so it cannot change GAUC or nDCG@5, and the organisers confirm they
    measured that.

    It is not a no-op when **averaging** models. Two models can put a user's items
    in the same order while disagreeing wildly about the size of the gaps, and a
    raw average is then dominated by whichever model happens to use a wider scale.
    Averaging ranks compares like with like (CLAUDE.md 9.5).
    """
    out = np.zeros(len(scores), dtype=np.float64)
    order = np.argsort(groups, kind='stable')
    sorted_groups = groups[order]
    starts = np.flatnonzero(np.r_[True, sorted_groups[1:] != sorted_groups[:-1]])
    ends = np.r_[starts[1:], sorted_groups.size]
    for start, end in zip(starts, ends):
        idx = order[start:end]
        size = len(idx)
        if size == 1:
            out[idx] = 0.5
            continue
        # Average ranks for ties, so a tie stays a tie rather than becoming an
        # arbitrary order that the evaluator would then treat as a real decision.
        values = scores[idx]
        temp = np.argsort(values, kind='stable')
        ranks = np.empty(size, dtype=np.float64)
        ranks[temp] = np.arange(size, dtype=np.float64)
        unique, inverse, counts = np.unique(values, return_inverse=True,
                                            return_counts=True)
        if len(unique) < size:
            sums = np.zeros(len(unique))
            np.add.at(sums, inverse, ranks)
            ranks = (sums / counts)[inverse]
        out[idx] = ranks / (size - 1)
    return out


def blend(score_vectors: Sequence[np.ndarray], groups: np.ndarray,
          weights: Sequence[float] | None = None,
          normalise: str = 'within_user_rank') -> np.ndarray:
    """Combine several models' scores into one ranking.

    ``normalise='within_user_rank'`` rank-normalises each model within each user
    before averaging, which is the only form that means anything here; ``'none'``
    averages raw scores and is offered so the difference can be measured rather
    than asserted.
    """
    if not score_vectors:
        raise ValueError('nothing to blend')
    weights = list(weights) if weights is not None else [1.0] * len(score_vectors)
    if len(weights) != len(score_vectors):
        raise ValueError(f'{len(weights)} weights for {len(score_vectors)} models')
    total = float(sum(weights))
    if total <= 0:
        raise ValueError('blend weights must sum to something positive')

    stacked = np.zeros(len(score_vectors[0]), dtype=np.float64)
    for weight, scores in zip(weights, score_vectors):
        values = np.asarray(scores, dtype=np.float64)
        if normalise == 'within_user_rank':
            values = rank_normalise(values, groups)
        elif normalise != 'none':
            raise ValueError(f'unknown normalise {normalise!r}')
        stacked += weight * values
    return stacked / total


def train_ensemble(splits: Dict[str, list] | None = None,
                   *,
                   seeds: Sequence[int] = (0, 1, 2),
                   normalise: str = 'within_user_rank',
                   weights: Sequence[float] | None = None,
                   checkpoint_path: str | Path | None = None,
                   with_diagnostics: bool = True,
                   **train_kwargs: Any) -> TrainResult:
    """Train one model per seed and blend their validation scores.

    Selection is still on validation primary, and still on the blend as a whole:
    the members are not individually selected, so this is one experiment rather
    than *n*.

    The checkpoint saves every member plus the blend recipe, so the submission can
    reproduce the exact ranking that was scored.
    """
    splits = splits if splits is not None else hdata.load()

    # `seed` is supplied per member from `seeds`, so a caller-level `seed` would
    # collide with it. The child process injects one into every CONFIG, so this
    # is the normal path rather than an edge case: dropping it here is what makes
    # an ensemble expressible in CONFIG at all.
    train_kwargs.pop('seed', None)
    train_kwargs.pop('checkpoint_path', None)

    if not seeds:
        raise ValueError('an ensemble needs at least one seed')

    enc, dim = hdata.encode(splits)
    Xva, yva, uva = enc['valid']
    groups_va = build_groups(splits, 'valid', 'user_id')

    import tempfile

    members: List[TrainResult] = []
    member_scores: List[np.ndarray] = []
    states = []
    started = time.time()

    # Members are checkpointed to a scratch directory so their weights survive
    # long enough to be blended and saved together. Only the ensemble checkpoint
    # is kept; the members are an implementation detail of one experiment.
    with tempfile.TemporaryDirectory() as scratch:
        for seed in seeds:
            member_path = Path(scratch) / f'member_{seed}.npz'
            member = train_fm(splits, seed=seed, with_diagnostics=False,
                              checkpoint_path=member_path, **train_kwargs)
            members.append(member)
            model = load_checkpoint_state(member)
            states.append(model)
            member_scores.append(model.predict(Xva))

    blended = blend(member_scores, groups_va, weights=weights, normalise=normalise)
    final = hevaluate.evaluate(uva, yva, blended)
    seconds = time.time() - started

    guards.check_canary(float(final['primary']),
                        context={'runner': 'train_ensemble', 'seeds': list(seeds),
                                 'normalise': normalise})

    diagnostics: Dict[str, Any] = {}
    if with_diagnostics:
        member_primaries = [m.val_primary for m in members]
        diagnostics = {
            'metrics': {'val_gauc': float(final['GAUC']),
                        'val_ndcg5': float(final['nDCG@5']),
                        'val_primary': float(final['primary'])},
            'fit': {'train_primary': None, 'val_primary': float(final['primary']),
                    'gap': None,
                    'epochs_run': max(m.epochs_run for m in members),
                    'best_epoch': max(m.best_epoch for m in members)},
            'fields': field_contributions(states[0], enc['train'][0]),
            'ensemble': {
                'seeds': list(seeds), 'normalise': normalise,
                'members': [round(p, 6) for p in member_primaries],
                'best_member': round(max(member_primaries), 6),
                'mean_member': round(float(np.mean(member_primaries)), 6),
                'blend_over_best_member': round(
                    float(final['primary']) - max(member_primaries), 6),
            },
            'cost': {'seconds': round(seconds, 1),
                     'reference_fm_seconds': REFERENCE_FM_SECONDS,
                     'relative_to_reference': round(seconds / REFERENCE_FM_SECONDS, 2),
                     'members_trained': len(members)},
            'objective': {'kind': 'ensemble', 'name': f'{len(seeds)}-seed blend'},
        }
        guards.assert_record_clean(diagnostics, where='train_ensemble diagnostics')

    saved: str | None = None
    if checkpoint_path is not None:
        path = Path(checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, ensemble=np.array(1),
            normalise=np.array(normalise), dim=np.array(dim),
            weights=np.array(weights if weights is not None
                             else [1.0] * len(states), dtype=np.float64),
            **{f'V{i}': m.V for i, m in enumerate(states)},
            **{f'W{i}': m.W for i, m in enumerate(states)},
            **{f'b{i}': np.float32(m.b) for i, m in enumerate(states)})
        saved = str(path)

    return TrainResult(val_gauc=float(final['GAUC']),
                       val_ndcg5=float(final['nDCG@5']),
                       val_primary=float(final['primary']),
                       diagnostics=diagnostics, checkpoint=saved,
                       seconds=seconds, seed=int(seeds[0]),
                       epochs_run=max(m.epochs_run for m in members),
                       best_epoch=max(m.best_epoch for m in members),
                       epoch_history=[])


def load_checkpoint_state(member: TrainResult):
    """Rebuild a member model from the checkpoint a train_fm call saved."""
    if not member.checkpoint:
        raise ValueError('ensemble members must be trained with a checkpoint_path')
    blob = np.load(member.checkpoint)
    model = PluggableFM(blob['V'].shape[0], k=blob['V'].shape[1])
    model.V, model.W, model.b = blob['V'], blob['W'], np.float32(blob['b'])
    return model


def save_checkpoint(model, path: str | Path) -> Path:
    """Persist the trainable parameters. Nothing about the test split is stored."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, V=model.V, W=model.W, b=np.float32(model.b))
    return path


class EnsemblePredictor:
    """A blended model, restored from an ensemble checkpoint.

    Scoring needs the group ids as well as the features, because the members are
    combined by within-user rank rather than by raw score. That is the whole
    reason the blend is not a no-op, so it cannot be dropped for convenience.
    """

    def __init__(self, members: List[Any], weights: Sequence[float],
                 normalise: str):
        self.members = members
        self.weights = list(weights)
        self.normalise = normalise

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Raw mean of member logits. Only for callers with no groups.

        This is NOT the ranking the ensemble was scored on. `predict_blended` is.
        """
        return np.mean([m.predict(X) for m in self.members], axis=0)

    def predict_blended(self, X: np.ndarray, groups: np.ndarray) -> np.ndarray:
        """The ranking the ensemble was actually selected on."""
        return blend([m.predict(X) for m in self.members], groups,
                     weights=self.weights, normalise=self.normalise)


def load_checkpoint(path: str | Path, dim: int, k: int = 16, **kwargs):
    """Rebuild a model from a checkpoint, for rollback or for scoring a split.

    Handles both shapes. An ensemble checkpoint holds ``V0/W0/b0 ... Vn/Wn/bn``
    plus the blend recipe; a single model holds ``V/W/b``. Reading only the
    single shape is how a winning ensemble came back as
    ``KeyError: 'V is not a file in the archive'`` and could not be submitted at
    all.
    """
    blob = np.load(Path(path), allow_pickle=False)
    names = set(blob.files)

    if 'ensemble' in names:
        count = sum(1 for n in names if n.startswith('V') and n[1:].isdigit())
        members = []
        for i in range(count):
            member = PluggableFM(blob[f'V{i}'].shape[0], k=blob[f'V{i}'].shape[1])
            member.V, member.W = blob[f'V{i}'], blob[f'W{i}']
            member.b = np.float32(blob[f'b{i}'])
            members.append(member)
        weights = (blob['weights'].tolist() if 'weights' in names
                   else [1.0] * count)
        normalise = (str(blob['normalise']) if 'normalise' in names
                     else 'within_user_rank')
        return EnsemblePredictor(members, weights, normalise)

    model = PluggableFM(blob['V'].shape[0], k=blob['V'].shape[1], **kwargs)
    model.V, model.W, model.b = blob['V'], blob['W'], np.float32(blob['b'])
    return model


def score_split(model, splits: Dict[str, list], split: str,
                enc: Dict[str, tuple] | None = None) -> np.ndarray:
    """Model scores for a split, in the split's own row order.

    Works on ``test``: producing a submission needs the features, never the label.

    An ensemble is scored through its blend, with the group ids it needs. Using
    the raw mean instead would submit a different ranking from the one that was
    selected on validation.
    """
    if enc is None:
        enc, _ = hdata.encode(splits)
    X = enc[split][0]
    if isinstance(model, EnsemblePredictor):
        return model.predict_blended(X, build_groups(splits, split, 'user_id'))
    return model.predict(X)
