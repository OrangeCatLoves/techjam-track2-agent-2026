"""Objective functions. The primary target of the whole project.

OWNS
    - the loss interface (CLAUDE.md section 11.1, interface 2), designed so that a
      pairwise or listwise objective drops in without touching the training loop
    - the reference pointwise loss, which must reproduce the official baseline
    - the registry the agent's generated losses register into

MUST NEVER
    - be edited to make a generated loss pass. Generated losses live in
      ``harness/models/gen/`` and register themselves; this file holds the
      interface and the reference implementation only
    - see a label from the hidden test split. Losses are called during training,
      on train rows, and ``harness/data.py`` does not hand out test labels

WHY THIS IS THE PRIMARY TARGET
    The official baseline optimises pointwise logloss while both scored metrics are
    within-user ranking metrics. That mismatch is the clearest structural weakness
    in the baseline and the organisers rank it first among untried directions.
    GAUC is itself a pairwise quantity, so a pairwise objective is directly aligned
    with it; a listwise softmax weights the top of each list, which is what nDCG@5
    measures.

    Only the pointwise reference ships. Pairwise and listwise objectives are for
    the agent to propose and write itself (CLAUDE.md section 12.2, M3) -- shipping
    them would hand the agent its best idea and hollow out the Innovation score.
    ``knowledge/methods.md`` describes them; no code here implements them.

THE CONTRACT
    ``loss_and_grad(z, y, groups) -> (loss, dL_dz)``

    z       float array (B,)   model logits for this batch
    y       float array (B,)   labels, 0.0 or 1.0
    groups  int array   (B,)   group id per row, for objectives that need lists.
                               Rows sharing a group id belong to one ranking list.
                               A pointwise loss ignores it.

    loss    float              the batch loss, for logging only
    dL_dz   float32 array (B,) gradient of *loss* with respect to z

    **dL_dz must already be normalised by the batch.** The trainer applies it as-is,
    so a loss that returns an un-normalised gradient trains at B times the intended
    learning rate. The pointwise reference divides by B; anything else must too, or
    must state why not.
"""
from __future__ import annotations

from typing import Callable, Dict, Tuple

import numpy as np

#: Signature of every objective. See the module docstring.
LossFn = Callable[[np.ndarray, np.ndarray, np.ndarray], Tuple[float, np.ndarray]]

_REGISTRY: Dict[str, LossFn] = {}

EPS = 1e-9


class LossError(ValueError):
    """A loss violated the interface. Raised by ``check_loss``."""


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically clipped logistic, matching the organisers' baseline exactly."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def register_loss(name: str) -> Callable[[LossFn], LossFn]:
    """Register an objective under *name*. Re-registering the same name is an error.

    Generated losses call this from ``harness/models/gen/``.
    """
    def decorate(fn: LossFn) -> LossFn:
        if name in _REGISTRY:
            raise LossError(f'loss {name!r} is already registered')
        _REGISTRY[name] = fn
        return fn
    return decorate


def get_loss(name_or_fn: str | LossFn) -> LossFn:
    """Resolve a loss by name, or pass a callable straight through."""
    if callable(name_or_fn):
        return name_or_fn
    try:
        return _REGISTRY[name_or_fn]
    except KeyError:
        raise LossError(
            f'unknown loss {name_or_fn!r}; registered: {sorted(_REGISTRY)}') from None


def registered() -> Tuple[str, ...]:
    """Names of every registered objective."""
    return tuple(sorted(_REGISTRY))


# --------------------------------------------------------------------------
# the reference objective
# --------------------------------------------------------------------------

@register_loss('pointwise_logloss')
def pointwise_logloss(z: np.ndarray, y: np.ndarray,
                      groups: np.ndarray | None = None) -> Tuple[float, np.ndarray]:
    """Mean binary cross-entropy. The official baseline's objective.

    Reproduced here rather than imported so that the trainer has one code path,
    but it is arithmetically identical to ``starter.baseline.FM.step``:
    ``g = (sigmoid(z) - y) / B``. ``tests/test_contract_baseline.py`` and
    ``tests/test_runners.py`` together pin that equality to the published number.

    ``groups`` is ignored: a pointwise objective has no notion of a list, which is
    precisely the weakness the agent is expected to attack.
    """
    p = sigmoid(z)
    batch = len(y)
    loss = -float(np.mean(y * np.log(p + EPS) + (1 - y) * np.log(1 - p + EPS)))
    return loss, ((p - y) / batch).astype(np.float32)


# --------------------------------------------------------------------------
# interface validation, used by the patch validator before running a generated loss
# --------------------------------------------------------------------------

def check_loss(fn: LossFn, *, batch: int = 64, n_groups: int = 8,
               seed: int = 0) -> Dict[str, float]:
    """Assert *fn* satisfies the interface on synthetic data.

    Checked: it returns a pair; the loss is a finite scalar; the gradient is a
    finite float array of shape ``(B,)``; and the gradient's sign is consistent
    with the loss under a small step, which catches a sign error -- the single
    most common way a hand-written objective silently trains backwards.

    Returns a small dict of measurements for the log. Raises ``LossError``
    otherwise.
    """
    rng = np.random.default_rng(seed)
    z = rng.normal(0, 1, batch)
    y = (rng.random(batch) < 0.35).astype(np.float64)
    groups = np.sort(rng.integers(0, n_groups, batch))

    out = fn(z, y, groups)
    if not (isinstance(out, tuple) and len(out) == 2):
        raise LossError('a loss must return (loss, dL_dz)')
    loss, grad = out

    if not np.isscalar(loss) and not isinstance(loss, (float, int, np.floating)):
        raise LossError(f'loss must be a scalar, got {type(loss).__name__}')
    if not np.isfinite(float(loss)):
        raise LossError(f'loss is not finite: {loss}')

    grad = np.asarray(grad)
    if grad.shape != (batch,):
        raise LossError(f'gradient shape {grad.shape} != ({batch},)')
    if not np.all(np.isfinite(grad)):
        raise LossError('gradient contains NaN or Inf')

    # A step along -grad must not increase the loss. Uses a step small enough that
    # first-order behaviour dominates, and tolerates exact ties (a flat objective).
    step = 1e-4 / (np.abs(grad).max() + EPS)
    moved, _ = fn(z - step * grad, y, groups)
    if float(moved) > float(loss) + 1e-8:
        raise LossError(
            f'stepping against the gradient increased the loss '
            f'({loss:.8f} -> {float(moved):.8f}); the sign is probably inverted, '
            f'which would train the model backwards')

    return {'loss': float(loss),
            'grad_abs_mean': float(np.abs(grad).mean()),
            'grad_abs_max': float(np.abs(grad).max()),
            'loss_after_step': float(moved)}
