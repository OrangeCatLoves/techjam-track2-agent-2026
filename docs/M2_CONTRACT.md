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


---

## 6. Leak response: two tiers, and the run never halts

**Revised.** An earlier version stopped the run on a second canary trip. That was
wrong; see D13 for the argument that overturned it. The policy in force:

| Validation primary | Response |
|---|---|
| > 0.80 (canary) | **Quarantined.** Never kept, never repaired, never submitted. Returned as `error_kind="canary"` |
| > 0.68 (review) | **KEPT and flagged.** Can still win. A human inspects it before submission |
| otherwise | kept, clean |

**The run never halts on either.** On a canary trip the loop quarantines, records
the patch's content hash in the tried-set, runs a retrospective audit of everything
already kept, and continues. A second trip additionally forces the next proposal to
a different `target_stage`.

**Why the lower tier exists, and why it is unconditional.** The result that costs us
the competition is not the one that trips the canary — that one is caught by
definition. It is the quieter sibling from the same leak path: a 0.72 that trips
nothing, looks like a genuine breakthrough, and is kept and submitted. A leak that
*never* crosses 0.80 is invisible to the canary entirely, and has no trip to trigger
an investigation. So the review flag fires on its own, not in response to a canary.

0.68 is `+0.0785` over the baseline — **31.8% of all remaining headroom consumed in a
single iteration**, on a benchmark whose own authors measured features and capacity
as dead ends. Implausible enough to be worth a look, loose enough not to fire on real
progress.

**Why the audit rather than a stop.** A canary trip says a leak path exists *now*. It
says nothing about when the path opened. Halting is forward-looking protection
against a backward-looking risk: the contaminated checkpoint may already be banked.
`guards.audit_kept_results` is the part that actually reaches it.

Counting survives a restart because both tiers are read off the filesystem
(`runs/quarantine/`, `runs/review/`) rather than held in memory, exactly like the
convergence counters.

---

## 7. Objective declarations

`register_loss(name, kind=...)` takes `pointwise`, `pairwise` or `listwise`. The kind
is a **declaration held to account**, not documentation: `check_loss` permutes the
grouping and requires a declared pairwise or listwise objective to return a different
loss.

This closes a hole the descent check cannot reach. A pairwise loss that builds its
pairs across the whole batch instead of within each user is mathematically valid,
descends properly, produces no NaN, trains and scores — and is not the objective it
claims to be. With train lists averaging 43.5 rows, most of its pairs would compare
rows belonging to different users. It could plausibly still beat 0.6015, for entirely
the wrong reason, and we would carry that into M3 believing we had found something.

The shuffle is of array *positions*, not of group labels: relabelling `0 -> 1,
1 -> 2` leaves the partition identical, so a correct loss would return the same value
and be failed for being correct. Pinned by
`test_relabelling_groups_does_not_fail_a_correct_loss`.

`check_loss` runs inside `train_fm` before the training loop, so a broken objective
costs 64 synthetic rows rather than a minute of training.

**One deviation from the review.** The review asked for `error_kind='rejected'` on
this failure. `rejected` is in `HARD_FAILURES` — never repaired — and a grouping bug
is exactly the kind of thing one repair attempt fixes, given a precise message. It
surfaces as `code` instead, so the agent gets its single repair with the reason
spelled out. The detection is the valuable part; making it terminal throws away a
cheap fix.


---

## 8. M2b requirements, fixed before any of it is written

All three are cheap now and expensive to retrofit.

**Token accounting from the first call, not the first successful one.** Failed
requests, retries and timeouts consume tokens. A counter that only increments on
success reports a Feasibility number wrong by an unknown amount, and Feasibility is
15% of the grade, scored only if we beat the baseline. `llm.py` records usage from
the response object on every call, including calls that raise, and attributes
retries to the iteration that caused them.

**A hard token ceiling per run, enforced in `llm.py`.** 300,000 for development runs,
with a warning at 200,000. A loop that gets stuck re-proposing at 4am should stop,
not drain the account. The ceiling is config; the enforcement is not optional; and
hitting it is a logged event rather than a crash.

**Log which model handled which call.** Recorded per call even when both variables
point at the same model. Doing it now means the fast/strong split is a config change
later rather than an instrumentation project.

Current setting: `LLM_MODEL_FAST=claude-haiku-4-5-20251001` for routine diagnosis and
one-shot repair, `LLM_MODEL_STRONG=claude-opus-5` for proposals and plateau breaks.

---

## 9. Where the score-versus-best comparison lives

Exactly one place: `ConvergenceTracker.record_iteration`, which **refuses** a
non-finite validation primary rather than ignoring it.

Ignoring is not neutral. `nan <= epsilon` is `False`, so a recorded NaN counts as a
non-strike and *resets the trailing streak*: a broken objective emitting NaN would
clear the strike counter and keep the run going indefinitely. Measured before the
fix: three iterations at strike 2, one NaN, strike count back to 0.

`Ledger.would_improve` and `ExperimentResult.usable` guard the same thing at their
own layers, so the loop has no route to the comparison that skips a check. The
tracker's refusal is the backstop: reaching it means the loop failed to check
`usable` first, which is a loop bug, and loop bugs are loud.
