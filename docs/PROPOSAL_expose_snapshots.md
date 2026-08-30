# Exposing the snapshot capability the harness already has

**For contributor A. One string edit in `agent/propose.py`, which is your column.
No harness change. No behaviour change.**

---

## What I found

`harness/models/runners.py` already implements snapshot ensembling, and it has for
some time:

- `train_fm(..., snapshots: int = 1)` — line 228. Keeps a `deque` of the last
  `snapshots + patience + 1` epoch states and returns the window nearest the
  validation-best epoch as `TrainResult.snapshot_states`.
- `train_ensemble(..., snapshots: int = 1)` — line 461. Expands each seed's
  snapshots into separate blend members, reports `snapshots_per_seed` in
  diagnostics, and names the run `"{n}-seed x {m}-snapshot blend"`.
- The selection rule is already the right one: nearest epochs to the peak, later
  preferred on a tie. **Fixed window, no epoch chosen by its own validation
  score.** Whoever wrote it avoided the selection trap deliberately.

And `harness/_run_patch.py` passes a patch's `CONFIG` through with `**config` and
no key whitelist, so this already routes correctly:

```python
CONFIG = {"ensemble": [11, 23, 37, 53, 71], "snapshots": 4, "batch": 2048}
```

**That patch would run today, unchanged.** The plumbing is complete end to end.

## Why no agent has ever used it

`CONFIG_KEYS` in `agent/propose.py` is the only place the agent learns what a
patch may contain — the comment at line 181 says so outright: *"Stated because
the model cannot see train_fm."* It documents `loss`, `group_by`, `k`, `lr`,
`l2`, `batch`, `max_epochs`, `patience`, then `ensemble`, `normalise`, `weights`.

It does not mention `snapshots`. The string has no occurrence of the word.

So across 33 experiments and five runs, no agent has proposed snapshot
ensembling because none of them knew the key existed. This is a documentation
gap, not a capability gap and not a judgement failure. Compare
duration-conditioned recalibration, which *is* in the corpus at char 8,648 of
12,568 — inside the 14,000 cap, never truncated — and which the agent has also
never chosen. That one is a real decision. This one was never on the menu.

## Why it is worth exposing now

Finding 4 established why heterogeneous-batch mixing failed: every way of making
members more different also made them worse, and the two cancelled exactly.
Blend gain rose from +0.0006 to +0.0010 as low-batch members were added, and the
blend score did not move — 0.6031 to 0.6034 across five mixing ratios.

Snapshots are the one source of diversity that does not cost member quality.
Consecutive epochs near the peak sit at roughly equal validation primary but in
different places in parameter space, so they disagree about *which* users they
rank correctly rather than about how good they are. That is exactly the
ingredient Finding 4 showed was missing, and it is the only remaining way to get
it.

It is also nearly free: one training run yields several members, so
`{"ensemble": 5, "snapshots": 4}` costs five training runs, not twenty.

---

## The change

In `agent/propose.py`, in the `CONFIG_KEYS` string, replace:

```
  ensemble  int | list  number of seeds, or an explicit list of seeds
  normalise str         "within_user_rank" (default) or "none"
  weights   list        per-member weights; defaults to equal

  CONFIG = {"ensemble": 3, "normalise": "within_user_rank"}

Cost scales with the number of members: three seeds is three training runs.
```

with:

```
  ensemble  int | list  number of seeds, or an explicit list of seeds
  normalise str         "within_user_rank" (default) or "none"
  weights   list        per-member weights; defaults to equal
  snapshots int         how many epochs of each member to blend (default 1)

  CONFIG = {"ensemble": 3, "normalise": "within_user_rank"}
  CONFIG = {"ensemble": 3, "snapshots": 4}

Cost scales with the number of seeds, not the number of members: one training run
yields as many snapshots as you ask for, so {"ensemble": 3, "snapshots": 4} costs
three training runs and blends twelve members.

Snapshots are the epochs nearest each member's own validation-best epoch, a
window fixed in advance. No epoch is chosen by its own validation score.
```

That is the entire change. `CONFIG_KEYS` is a prompt string with no behaviour
attached, so no test can regress and `verify_setup.py` is unaffected. Worth
running `python -m pytest tests/ -q` anyway to confirm.

## Framing in the writeup

Declare it. The honest sentence is: *"the harness had a snapshot capability that
`CONFIG_KEYS` did not document, so no agent could reach it; we corrected the
omission between run N-1 and run N, and the agent then chose whether to use it."*

Do not fold a post-fix run into run 4's zero-intervention claim — report it as its
own run with its own conditions. But the agent still decides whether to reach for
snapshots, how many, on top of which seed count and batch, and has to justify it
against everything else on the menu. That is the same standing run 4's ensemble
has: you implemented the stage, the agent chose to use it and explained why.

Keep the wording factual. Do not add "this is promising" or "try this next" — the
corpus is explicitly framed as *"background, not a queue to work through"*, and
the result is only worth having if the agent picks it on the evidence.

## Then

Run clean into a fresh directory, no steering, `max_iterations=20` so the
convergence rule is the only thing that stops it.

The agent may still not choose it. From the five runs so far: an ensemble
iteration is reached in roughly three of five runs, and **every iteration that
ever beat 0.6017 was an ensemble iteration** — objectives are 0 for 24. Run 3's
ensemble attempt crashed on the checkpoint bug you have since fixed, so the
current hit rate should be better than that history suggests.