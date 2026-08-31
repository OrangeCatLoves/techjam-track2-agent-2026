# Milestone 2 — what was built, how, and why

A review document, companion to `MILESTONE_1.md`. Written to be handed to a
reviewer with no prior context on this repository.

**Status:** complete. The M2 gate passes 8/8 both stubbed and against real training.
**345 tests** (321 fast, 24 slow), all green. 4,129 lines of new module code across
thirteen modules plus the gate script.

**Scope discipline:** M2 is "the loop works", an engineering result. The score is
explicitly **not** a gate here — that moved to M3 as a deliberate spec revision
(D11, §7 below).

---

## 1. What M2 is, and how it was sequenced

M1 built the safety floor: we cannot see the hidden answers, we can tell whether a
number is real, and we cannot break the competition rules by accident.

M2 builds the thing that stands on it: **a loop where a language model writes real
Python, we run it, read the score, and let it decide what to try next.**

It was split in two, on review:

| Half | What | Testable against |
|---|---|---|
| **M2a** | model runner, `run_experiment`, sandbox, patch validator, ledger, logger, analyse | a fake agent returning hardcoded patches |
| **M2b** | `llm.py`, `diagnose`, `propose`, `loop` | a stub that can fail |

The point of the split is that when the two halves join, **only the seam is new**.
Each side was already exercised on its own.

Before either half was written, the seam itself was frozen (CLAUDE.md section 11.1):
the return shape, the `error_kind` vocabulary, and the diagnostics keys. Building
both halves and *then* agreeing how they talk turns the join into a negotiation
between two things that already exist and neither of which wants to move.

---

## 2. The life of one patch

Every module appears in this path, which is the clearest way to see how they fit.

```
    LLM writes a string of Python
              │
    ┌─────────▼───────────┐
    │ harness/patch.py    │  AST parse; imports, names, target path
    └─────────┬───────────┘
              │  rejected? ──► error_kind="rejected", 0.0s, nothing ran
    ┌─────────▼───────────┐
    │ write to harness/models/gen/iter_007.py
    └─────────┬───────────┘
    ┌─────────▼───────────┐
    │ harness/sandbox.py  │  child process, wall-clock + memory ceilings
    └─────────┬───────────┘
    ┌─────────▼────────────────────────────────────┐
    │ CHILD: harness/_run_patch.py                 │
    │   import patch, read CONFIG                  │
    │   harness/data.py    → label-stripped data   │
    │   harness/losses.py  → objective + check     │
    │   models/runners.py  → train the FM          │
    │   → result.json                              │
    └─────────┬────────────────────────────────────┘
    ┌─────────▼───────────┐
    │ harness/experiment.py│ read JSON, screen, shape
    └─────────┬───────────┘
              │
      ExperimentResult ──► agent/diagnose.py ──► agent/propose.py ──► next patch
              │
      agent/loop.py: keep or reject → ledger.promote / decline
                                    → logger.log_iteration
                                    → tracker.record_iteration
```

Every arrow is a place something can go wrong, and **every one returns rather than
raises.**

---

## 3. M2a — the harness half

### 3.1 `harness/models/runners.py` (386 lines) — the trainer

This module exists because of one line in the organisers' code:

```python
return {'valid': evaluate(uva, yva, m.predict(Xva)),
        'test':  evaluate(ute, yte, m.predict(Xte))}   # ← this
```

`run_fm` computes a hidden-test metric, so it can never be called. The trainer had
to be ours.

**But only the trainer.** `PluggableFM` subclasses their `FM` and inherits `logits`
and `predict` **unchanged**, so the arithmetic cannot drift. Only the loop around it
is reimplemented — because theirs computes test metrics, and because the objective
had to become swappable.

That reimplementation duplicates twelve lines of their Adam update. **Validated, not
assumed:**

| | organisers' recorded run | ours |
|---|---|---|
| epoch 1 loss | 0.6391 | 0.6391 |
| epoch 2 loss | 0.5479 | 0.5479 |
| epoch 1 valid primary | 0.5869 | 0.5869 |
| final valid primary | 0.6015 | **0.6015** |

Digit for digit. That is the licence to use our own trainer.

**Diagnostics produced**, per the frozen contract: per-metric scores; the train/valid
gap (live: the FM peaks at epoch 7 and declines for four more while training loss
falls); per-field contributions (`mean |W|`, `mean ||V||` — the FM's
feature-importance equivalent); list sizes; and measured cost against a reference.

A small implementation note: the organisers' `encode()` does not return field
offsets, so field contributions read the ids straight off the data — column *i* of
`X` only ever contains ids for field *i*.

### 3.2 `harness/losses.py` (253 lines) — the objective

The primary target of the whole project. The baseline optimises pointwise logloss
while both metrics are ranking metrics.

```python
loss_and_grad(z, y, groups) -> (loss, dL_dz)
```

`groups` is the reason the interface exists: a pointwise loss ignores it, a pairwise
or listwise one cannot work without it.

**Only pointwise ships.** No BPR, no listwise softmax — writing those is the agent's
job, and shipping them would hand it its best idea and hollow out the Innovation
score. A test asserts the registry stays a single entry.

**`check_loss` runs before training**, on 64 synthetic rows, and catches two things:

1. **Sign inversion** — steps against the gradient and requires the loss to fall.
   The commonest way a hand-written objective silently trains backwards.
2. **Grouping blindness** — for a *declared* pairwise or listwise loss, permutes the
   grouping and requires the loss to move.

The second was added on review and closes a hole the first cannot reach. A pairwise
loss that builds pairs across the whole batch instead of within each user is
mathematically valid, descends properly, produces no NaN, trains and scores — and is
not the objective it claims to be. With train lists averaging 43.5 rows, most of its
pairs compare rows from different users. It could plausibly beat 0.6015 for entirely
the wrong reason.

**The subtlety that makes it work:** the shuffle is of array *positions*, not group
labels. Relabelling `0→1, 1→2` leaves the partition identical, so a correct
group-aware loss returns the same value and a label-based check would fail correct
code *for being correct*. Pinned by its own test.

`kind` is a **declaration held to account**, not documentation.

### 3.3 `harness/patch.py` (264 lines) — the gate

Four checks on every piece of generated code:

1. **Protected paths.** Two writable directories; everything else refused —
   including `configs/base.yaml`, which holds the protected list itself. Code that
   can edit the config can edit its way out of every guard.
2. **Import allowlist, read off the AST.** A text search for `import os` misses
   `__import__('o' + 's')`. The syntax tree does not.
3. **The organisers' modules refused by name.** `starter.baseline` computes a test
   metric; `starter.data` returns test rows with labels.
4. **Escape hatches.** No `eval`, `exec`, `compile`, `__import__`, `open`, `globals`.

A rejected patch **never reaches disk**, so a later run that skipped validation
cannot import it.

To be precise about what this is: a **correctness and scope gate, not a security
sandbox.** The threat model is a language model writing plausible code that reaches
somewhere it should not. It is not a defence against deliberate escape.

### 3.4 `harness/sandbox.py` (177 lines) — the ceilings

Wall clock and memory, enforced by polling RSS with psutil because Windows has no
`RLIMIT_AS`. A ceiling with a short delay, which is adequate: the goal is ending a
runaway experiment, not preventing an allocation.

**Child output goes to temporary files, not pipes.** A child that writes more than
the ~64 KB pipe buffer while the parent is not reading **blocks forever**, and looks
identical to an infinite loop. A test floods 10 MB of stdout to prove it does not
happen.

All child output is filtered through the M1 guards before return.

### 3.5 `harness/experiment.py` (354 lines) — the boundary and the stub

`run_experiment(patch_path, seed)` is the only thing crossing between harness and
agent. Its most important property: **it never raises for an experiment failure.**
A crash, timeout, memory breach, rejected patch and tripped canary are all returned
values. If any raised, the loop would die on its first bad generated patch — and a
bad generated patch is a certainty, not a risk.

Measured:

| case | result | cost |
|---|---|---|
| working patch | `ok=True`, full diagnostics | 16 s, 542 MB peak |
| rejected patch | `kind=rejected` | **0.0 s** — nothing ran |
| patch that raises | `kind=code`, traceback attached | ~10 s |
| slow patch | `kind=timeout` | killed at 3.1 s |
| memory hog | `kind=memory` | killed at 70 MB |

**The stub can fail.** All nine contract cases, routed through the *same screening
function* as a real result so the two cannot drift apart. Two matter most:

- **NaN.** A test asserts `nan > 0.6015` is `False` — the trap made explicit. An
  unguarded NaN reads as an ordinary non-improvement and hides a broken objective.
- **Canary trip.** Arrives *already converted* to a failure, because that is what
  the loop would really be handed; a 0.93 score never reaches it.

The stub also **honours `checkpoint_path`** and writes a real file. That was added
after the first end-to-end run failed: a stub whose *success* path cannot be
exercised is not doing its job.

### 3.6 `harness/ledger.py` (290 lines) — artefacts and rollback

**Rollback turns out to be free by construction.** Model state lives in checkpoint
files, never in a mutable global, so rejecting an experiment is simply declining to
move the best-pointer. That is why a failed swing costs nothing.

Two details:

- `promote()` **copies** into `best/` rather than pointing at the iteration
  checkpoint, so later cleanup cannot orphan the winner.
- `would_improve()` refuses `None` and `NaN` explicitly, rather than leaving every
  caller to remember `nan > best` is `False`.

It owns **no iteration counter and no tried-set** — `convergence.py` stays
authoritative. Two sources of truth for "which iteration is this" is how a restart
silently double-counts.

It also archives each used patch out of `harness/models/gen/`, so generated code
cannot accumulate where code that is about to run is expected.

### 3.7 `harness/logger.py` (291 lines) — two sinks

`log.jsonl` for the resource table; `log.md` for the judges.

**The markdown is regenerated from the JSONL**, not appended. A restart therefore
cannot produce a duplicated tail, and the two sinks cannot drift.

Restarts, recoveries and manual interventions are **separate event kinds**, because
the definition in force distinguishes them and that only survives if they are
recorded apart. An intervention is a human changing the agent's instructions,
objective or search space; restarting a crashed process is not.

Every record screened; every sink opened with explicit UTF-8 (D10).

### 3.8 `harness/analyse.py` (482 lines) — the agent's eyes

Nine query primitives over train and validation. **Every entry point refuses the
test split** — the route this closes is aggregate statistics over test *features*,
which are still information about the hidden set.

```
rate_by_bucket · distribution · list_size_profile · user_composition
segment_metrics · temporal_drift · model_disagreement · score_tie_rate
cold_key_rate
```

Two design choices aimed at the risk that this becomes a dashboard:

- **`capabilities()`** lets the agent enumerate its own instrument rather than being
  told what to look at.
- A test asserts every capability description says what is *measured*, never what to
  conclude. Matched on advisory **phrases**, not bare words — a substring check on
  "best" would fire on "best epoch" and get weakened rather than heeded.

---

## 4. M2b — the agent half

### 4.1 `agent/llm.py` (377 lines) — the meter

Built to three requirements fixed in the contract *before* a line of it was written,
because all three are cheap now and expensive to retrofit.

**Charged on every call, including failures and retries.** When a request raises,
the true count is unknown, so the meter charges an estimate from prompt length,
marks the call `estimated`, and reports how many were. An approximation that can
only overstate beats a precise number that is wrong. Feasibility is 15% of the
grade.

**Hard per-run ceiling, checked before the call** rather than after, so hitting it
costs nothing. 300K with a warning at 200K, both config.

**Per-call model attribution**, even though both roles point at the same provider
today — so the fast/strong split is later a config change, not an instrumentation
project.

Prompts are screened on the way out: this is the last point at which the text is
still ours.

`LLM_PROVIDER=none` works from day one, and the transport is injectable, so the
entire loop is testable with no network, no key and no spend.

### 4.2 `agent/diagnose.py` (218 lines) — facts before judgement

Rule-based. Deltas, which metric moved, the overfitting signal, cost, run state.

**The LLM never supplies a number.** An LLM asked to both measure and decide will
confidently mis-measure; giving it arithmetic it cannot get wrong and asking only
for the judgement is what makes the reasoning in the log worth reading.

It carries a **noise floor**: a 0.0005 gain is reported as *no change*, because that
is smaller than the jitter between adjacent epochs of the same run. Reporting it as
an improvement would teach the agent to chase noise.

It reports what is true and what has been tried. It never says what to try.

### 4.3 `agent/propose.py` (372 lines) — one experiment at a time

Validated against the frozen spec before anything reaches disk. Tolerant of a fenced
JSON block, because that is the commonest shape a model returns and rejecting it
would spend an iteration on formatting.

**The deterministic fallback is openly a scripted search** — 30 distinct
configurations, the base sequence crossed with six variants. Every patch it writes
carries `NOT the agent` in its docstring, so no reader can mistake one for the
other. It is insurance against an outage **and the honest control**: if it scores as
well as the agent, the agent was not adding anything, and we would rather know.

`knowledge/methods.md` is loaded as *reference material*, and the prompt says
explicitly that it is not a queue and that every estimate in it should be checked
with the tools before being trusted.

### 4.4 `agent/loop.py` (374 lines) — thin on purpose

Diagnose → propose → write → run → decide → log → update the clock.

It **asks** the tracker whether to continue rather than deciding. It checks
`ExperimentResult.usable` before every comparison. It never repairs a hard failure
(`evaluator`, `rejected`, `canary`) — repairing one is patching around a guard. It
writes the submission from the **promoted** checkpoint, not from whatever ran last.

---

## 5. The gate

`scripts/m2_acceptance.py`, run both ways.

```
                                             stub    real
restart resumes counters, not resets          PASS    PASS
restart recovers the best checkpoint          PASS    PASS
ten iterations complete unattended            PASS    PASS
valid submission passing --check              SKIP    PASS  (170,588 rows)
log.md readable by a human                    PASS    PASS
no hidden-test metric anywhere in the log     PASS    PASS
zero manual interventions                     PASS    PASS
deterministic mode spent no tokens            PASS    PASS
                                              8/8     8/8
```

The stub run exercises **every failure kind in the contract** in one pass. The real
run trains actual models and produces an actual submission in 111 s.

**The gate skips the submission check in stub mode rather than failing it.** The
stub trains no real model, so its 2×2 placeholder checkpoint cannot score the real
encoding. A gate that goes green for the wrong reason is worse than one that fails —
so it skips, and still asserts the failure was *reported* rather than raised.

---

## 6. Bugs and errors found while building

Six, five of them found by review questions rather than by tests failing.

**D12 — label dtype changed the metric's precision.** `starter/evaluate.py`
accumulates `(2**t - 1)` in the *label's* dtype, so float32 labels (what `encode`
returns) and int labels (what the submission path passes) gave different precisions
for identical predictions. 7e-7 — four orders of magnitude below epsilon, so nothing
was ever at risk, but two spellings of one number is an afternoon lost to a phantom
regression. Normalised at the single call site.

**D17 — a NaN reset the strike counter.** The worst of the six. `record_iteration`
correctly refused a NaN as a new best, but still recorded the iteration as *scored*,
and `nan <= 0.002` is `False` — so the NaN counted as an improvement and **reset the
trailing streak**:

```
before NaN: strikes = 2
after  NaN: strikes = 0
```

A model emitting NaN would clear the counter every time and prevent the run ever
converging by the no-improvement rule. It now refuses a non-finite primary outright.

**The canary raised out of `run_experiment`**, violating the contract that it never
raises for an experiment failure. Now returned as `error_kind='canary'`.

**The quarantine filename collided.** A millisecond timestamp meant two trips in the
same millisecond overwrote each other, silently undercounting exactly the number the
escalation policy reads. Now uuid-suffixed.

**A credential reached a tracked file.** An API key was pasted into `.env.example` —
the file that *looks* like the one to edit — and a broad `git add -A` on my part
committed it. Caught immediately, nothing pushed, commit amended and the object
purged, key rotated. Two tests now prevent a repeat.

**The corpus contained wrong numbers** — see §8.

---

## 7. Decisions taken and reversed

### D11 — the M2 gate was split

`CLAUDE.md` §12.2 originally gated M2 on "ten iterations *and beats 0.6016*". That
bolts a research result onto an engineering one, and §12.2 predates reading the
starter kit — at that point the plan named LightGBM as the workhorse and assumed
easy early gains. The published ablations killed both. Gating a milestone on an
outcome the strategy section itself calls unlikely means a working loop is recorded
as a failure. M2 is now engineering-only; the score moved to M3.

### D13 — the leak policy, reversed on review

I first decided a second canary trip would **stop the run**. That was wrong, and the
reasoning I gave for it argued against it: if a systematic leak produces kept
sub-threshold results, the contaminated checkpoint **may already be banked before
the first trip**. Stopping is forward-looking protection against a backward-looking
risk. And halting an unattended scored run means it never converges and **nothing is
submitted** — a certain harm traded for a speculative one.

In force:

| Tier | Threshold | Response |
|---|---|---|
| canary | > 0.80 | quarantined; never kept, repaired or submitted |
| review | > 0.68 | **kept and flagged**; a human looks before submission |

The run never halts. On a trip: quarantine, hash into the tried-set, **audit
everything already kept**, continue.

**The review tier is unconditional** — and this is the part both the original design
and the counter-proposal missed. A leak that never crosses 0.80 trips nothing, so
any response *triggered by* a canary trip would never run for the most dangerous
leak of all.

0.68 is +0.0785 over baseline: **31.8% of all remaining headroom in a single
iteration**, on a benchmark whose authors measured features and capacity as dead
ends.

### D15 — timeout tightened

25 → 12 minutes (~11× the measured 63 s reference), so a runaway costs 3% of the
six-hour budget rather than 7%. Measured cost now reaches the agent in
`diagnostics['cost']` with the reference alongside, so "expensive" is a comparison
rather than a word.

---

## 8. The corpus was wrong twice

`knowledge/methods.md` is what the agent retrieves from when proposing. Building
`analyse` found two errors in it, both of which would have actively misled.

**D16 — list sizes.** The corpus said `(user_id, date)` grouping gives "~3" rows and
concluded **"No, roughly half"** — that it does not match evaluation. Measured:

| Grouping | Split | Lists | Mean | Median |
|---|---|---|---|---|
| `user_id` | train | 26,210 | 43.5 | 31 |
| `(user_id, date)` | train | 197,796 | **5.77** | **3** |
| `user_id` | valid | 22,377 | **5.58** | 4 |

Three is the *median*. The mean is 5.77 against an evaluation mean of 5.58 — an
almost exact match. The corpus was steering the agent **away from the option the
measurement favours**, which is worse than saying nothing.

**D18 — user composition.** The corpus quoted the published *test* composition as
though it described the data the agent works with:

| Split | all-neg | all-pos | **discriminative** |
|---|---|---|---|
| train | 5.1% | 2.3% | **92.7%** |
| valid | 30.3% | 11.9% | **57.8%** |
| test (published) | 27.1% | 9.2% | 63.7% |

GAUC is computed over discriminative users alone, so **the three splits do not
measure the metric over comparable populations.** That is a concrete
validation-to-test transfer risk, and a reason train intuitions about GAUC do not
carry.

**Both were removed rather than corrected**, and replaced with the `analyse` calls
that answer them in under a second. A measured number in the corpus becomes a fact
the agent trusts and will not re-check; a tool call is discoverable on demand. The
measured values live in `CLAUDE.md`, which humans read and the agent does not.

**The standing rule this makes concrete:** CLAUDE.md §14 already forbids writing a
number that was not computed by code. The corpus violated it invisibly until a tool
existed to check. Every remaining estimate in `methods.md` is suspect until measured.

---

## 9. What is not built

**Not started:**

- **The agent has never made a real LLM call.** Everything so far ran in
  deterministic mode. The first real run will cost tokens and will surface
  prompt-quality problems no stub can.
- Causal target encoding and the feature registry (`harness/features/`) — the
  writable directory exists and is empty.
- Ensembling, sequence features, multi-task heads. All are the agent's to propose.

**Deliberately absent:** any pairwise or listwise objective. Writing one is the
agent's job and a test asserts the registry stays a single entry.

---

## 10. Known gaps a reviewer should push on

1. **The prompt has never been exercised against a real model.** Everything about
   proposal quality is untested. This is the largest remaining unknown.
2. **The deterministic fallback is a scripted hyperparameter search.** If the API is
   unavailable during the scored run, that is what runs, and it will not beat the
   baseline. It is insurance against producing *nothing*, not against producing a
   weak result.
3. **`check_loss` runs on 64 synthetic rows.** It catches sign inversion and
   grouping blindness, not a subtle numerical error that only appears at scale.
4. **The repair path is one attempt on the cheap model.** Whether Haiku can fix an
   Opus-written objective from a traceback is untested.
5. **`analyse` results are not yet fed into the proposer's prompt automatically.**
   The plumbing exists (`build_prompt` takes `analyses`), but the loop does not yet
   let the agent *request* an analysis before proposing. That is M3 work and it
   matters for Innovation.
6. **The memory ceiling polls at 250 ms.** A very fast allocation could exceed the
   limit between polls.
7. **No multi-seed confirmation yet.** `confirm_seeds: 3` is in config and unused;
   with a 0.0008 seed std, a single-seed gain under ~0.002 is not yet believed.

---

## 11. Test coverage

| File | Tests | Covers |
|---|---|---|
| `test_experiment.py` | 76 | boundary, sandbox, patch validator, stub |
| `test_agent.py` | 49 | meter, diagnosis, proposer, loop |
| `test_guards.py` | 37 | deny-list, filter, canary, review tier, secrets |
| `test_convergence.py` | 32 | stopping rule, strikes, restart, NaN refusal |
| `test_analyse.py` | 31 | nine queries, test refusal, self-discovery |
| `test_runners.py` | 29 | baseline reproduction, losses, diagnostics |
| `test_no_test_labels.py` | 25 | the integrity test |
| `test_ledger.py` | 19 | promotion, rollback, restart |
| `test_submission.py` | 15 | eight corruptions, round trip |
| `test_logger.py` | 15 | both sinks, resource report |
| `test_determinism.py` | 10 | loader, encoder, metric, label dtype |
| `test_contract_baseline.py` | 7 | the regression gate |
| **total** | **345** | 321 fast (~1 min), 24 slow (~7 min) |

---

## 12. How to verify this yourself

```bash
python scripts/verify_setup.py          # M1 foundation, ~5 min
python scripts/m2_acceptance.py --stub  # M2 gate, stubbed, ~20 s
python scripts/m2_acceptance.py         # M2 gate, real training, ~2 min
pytest tests/ -m "not slow"             # 321 tests, ~1 min
pytest tests/                           # everything, ~8 min
```
