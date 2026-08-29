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

def save_checkpoint(model, path: str | Path) -> Path:
    """Persist the trainable parameters. Nothing about the test split is stored."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, V=model.V, W=model.W, b=np.float32(model.b))
    return path


def load_checkpoint(path: str | Path, dim: int, k: int = 16, **kwargs):
    """Rebuild a model from a checkpoint, for rollback or for scoring a split."""
    blob = np.load(Path(path))
    model = PluggableFM(dim, k=k, **kwargs)
    model.V, model.W, model.b = blob['V'], blob['W'], np.float32(blob['b'])
    return model


def score_split(model, splits: Dict[str, list], split: str,
                enc: Dict[str, tuple] | None = None) -> np.ndarray:
    """Model scores for a split, in the split's own row order.

    Works on ``test``: producing a submission needs the features, never the label.
    """
    if enc is None:
        enc, _ = hdata.encode(splits)
    X = enc[split][0]
    return model.predict(X)
