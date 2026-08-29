# Milestone 2 — the frozen seam

Agreed **before** either half is built, so that the join between M2a and M2b is
mechanical rather than a negotiation.

M2a is the harness side: model runner, `run_experiment`, sandbox, patch validator,
ledger, `analyse`. Testable with a fake agent returning hardcoded patches.

M2b is the agent side: `llm.py`, `diagnose`, `propose`, `loop`. Testable against the
stub in §3.

Nothing below may change once M2b starts without changing both halves in the same
commit.

---

## 1. `run_experiment` — the only thing crossing the boundary

```python
def run_experiment(patch_path: str, seed: int) -> ExperimentResult:
    """Materialise a patch, train, evaluate on validation, return facts.

    NEVER returns a hidden-test metric, under any flag, in any field.
    NEVER raises for an experiment failure: a failure is a returned value with
    `error` set, because the loop must record it and continue.
    Raises only for a harness bug, which must stop the run.
    """
```

Return shape:

```python
{
  "ok":          bool,          # False if error is set
  "val_gauc":    float | None,
  "val_ndcg5":   float | None,
  "val_primary": float | None,  # the only number the keep/reject decision may use
  "diagnostics": dict,          # §2, fixed keys
  "checkpoint":  str | None,    # path to the saved model, for rollback
  "error":       str | None,    # traceback tail, agent-visible, for one repair attempt
  "error_kind":  str | None,    # "code" | "timeout" | "memory" | "evaluator" | None
  "seconds":     float,
  "seed":        int,
}
```

`error_kind` is fixed vocabulary because it selects the recovery path (CLAUDE.md §6.3):

| kind | recovery |
|---|---|
| `code` | one repair attempt, then abandon |
| `timeout` | retry once at 30% subsample |
| `memory` | retry once at float32 with half the features |
| `evaluator` | **hard failure.** Never patch around the evaluator |
| `rejected` | **hard failure.** Patch validation refused it; nothing ran |
| `canary` | **hard failure.** Scored above the leak threshold; quarantined |

`rejected` and `canary` were added during implementation. `canary` matters most: a
tripped canary is returned as a *failed result*, not raised. `run_experiment` never
raises for an experiment failure, and a leaking experiment is still an experiment —
the loop records it, rolls back, marks it tried and continues. Halting would need a
human to restart, which costs autonomy for an event the quarantine file already makes
visible. It is a hard failure because repairing it would be patching around a guard.

---

## 2. `diagnostics` — fixed keys

These feed the agent's next prompt. Three groups, all computed by the harness, never
by the LLM. **The LLM never invents a number.**

```python
{
  # 1. per-metric validation scores, so the agent can see which metric moved.
  #    A pairwise objective should move GAUC more than nDCG@5; that prediction is
  #    only checkable if both are reported separately.
  "metrics": {"val_gauc": float, "val_ndcg5": float, "val_primary": float},

  # 2. train-versus-validation gap, the overfitting signal.
  #    The official FM peaks at epoch 7 and declines to epoch 11 while training loss
  #    keeps falling, so this is a live concern, not a formality.
  "fit": {"train_primary": float, "val_primary": float, "gap": float,
          "epochs_run": int, "best_epoch": int},

  # 3. field contributions -- the FM's equivalent of feature importance.
  #    Per feature field: mean |W| (first-order pull) and mean ||V|| (how much the
  #    field participates in crosses). A field with near-zero values is dead weight,
  #    which is the measurable form of the organisers' "user-side terms contribute
  #    exactly zero" result.
  "fields": {field_name: {"mean_abs_w": float, "mean_v_norm": float}},
}
```

Optional keys may be **added** by a runner; the three above must always be present.
Consumers must tolerate unknown keys.

Every diagnostics dict passes `harness.guards.assert_record_clean` before it leaves
`run_experiment`. A key named `test_*` or a nested `test` block is a hard failure.

---

## 3. The stub must be able to fail

`run_experiment` is expensive (~1–2 min per FM train), so M2b develops against a fake.
**A fake that only returns plausible numbers is worse than useless**: it means the
recovery path is written blind and gets its first real exercise during the scored run,
which is the one run that must not need a human.

The stub must be able to produce, on demand and deterministically by seed:

| Case | Shape returned |
|---|---|
| improvement | `ok=True`, `val_primary` above the current best |
| no improvement | `ok=True`, `val_primary` at or below the best |
| regression | `ok=True`, `val_primary` well below the best |
| code error | `ok=False`, `error_kind="code"`, a realistic traceback tail |
| timeout | `ok=False`, `error_kind="timeout"` |
| memory error | `ok=False`, `error_kind="memory"` |
| evaluator rejection | `ok=False`, `error_kind="evaluator"` |
| NaN score | `ok=True`, `val_primary=float('nan')`, `usable=False` |
| canary trip | `ok=False`, `error_kind="canary"` — already converted by the screen |

The last two are the nastiest and the most important.

A **NaN** must not silently become the new best. `nan > best` is `False`, so an
unguarded NaN reads as an ordinary non-improvement and hides a broken objective.
`ExperimentResult.usable` is the guard: `ok` **and** not `None` **and** finite. Nothing
compares a raw `val_primary` against the best without checking it.

A **canary trip** arrives already converted to a failure, because `run_experiment`
screens it before returning — a 0.93 score never reaches the loop at all. The stub
routes through that same screen, so the two cannot drift apart by anyone forgetting to
keep them in sync.

The loop's acceptance test runs a scripted sequence over the stub covering every row
above, and asserts the ledger, the strike count and the best checkpoint are correct
afterwards.

---

## 4. Interfaces carried forward unchanged from CLAUDE.md §11.1

```python
# Loss function -- the primary target. Must accept pointwise, pairwise and listwise.
def loss_and_grad(z: np.ndarray, y: np.ndarray, groups: np.ndarray) -> tuple[float, np.ndarray]

# Feature function
@register_feature(name="...", needs_dates_before=True)
def build(frame, stats): ...

# Experiment spec, the LLM's output, JSON-schema validated
{"hypothesis": str,
 "target_stage": "objective" | "model" | "features" | "sampling" | "ensemble",
 "patch_kind": str, "expected_gain": float,
 "expected_cost_minutes": float, "patch": str}
```

---

## 5. Already built, do not rebuild

- **The tried-set.** `ConvergenceTracker.has_tried` / `mark_tried`, persisted with the
  counters and tested. The ledger records proposals and patches; it must not own a
  second copy of this or a second iteration number. The tracker stays authoritative.
- **The counters and the stopping rule.** `harness/convergence.py`.
- **Log screening.** `guards.assert_record_clean` — the M2 logger calls it on every
  record before writing, and every sink opens with `encoding='utf-8'` (D10).
