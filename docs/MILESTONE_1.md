# Milestone 1 — what was built, how, and why

A review document. Written to be handed to a reviewer (human or model) with no prior
context on this repository.

**Status:** complete and verified. 80 tests passing (6 marked slow),
`scripts/verify_setup.py` green on all 11 checks. Reviewed once; §7 and §8 record what
that review changed.

**Scope discipline:** Milestone 1 is the foundation only. The agent loop, the LLM
integration, the sandbox, the feature registry and the model runners are Milestone 2 and
were deliberately **not** started. Section 9 lists exactly what is absent.

---

## 1. Context

### 1.1 The competition

TikTok TechJam 2026, Track 2: build an **autonomous ML research agent for recommender
systems**. Two things are judged: the agent itself, and the model it produces.

The model's task on KuaiRand-Pure: given a log of short-video impressions, rank the items
**within each user's own logged impression list** by likelihood of `long_view`. Ranking
stage only — no retrieval, no candidate generation.

| Judging criterion | Weight |
|---|---|
| Technical execution (hidden-test delta over baseline) | 35% |
| Innovation & problem insight | 20% |
| Impact & relevance (autonomy, by manual-intervention count) | 20% |
| Feasibility (tokens, wall clock) — **only scored if you beat the baseline** | 15% |
| Presentation | 10% |

### 1.2 The dataset and the splits

KuaiRand-Pure: 27,285 users, ~7,551 videos, 1,436,609 standard-log interactions,
April–May 2022. The organisers supply a starter kit (`starter/`) that is code only; the
data is downloaded separately.

| Split | Dates | Rows |
|---|---|---|
| train | 20220408–20220421 | 1,141,112 |
| validation | 20220422–20220428 | 124,909 |
| hidden test | 20220429–20220508 | 170,588 |

### 1.3 The metric

Both metrics rank within a single user's impression list (~6 items in the evaluation
splits). `primary = mean(GAUC, nDCG@5)`.

| Reference (validation) | primary |
|---|---|
| random | 0.4834 published / 0.4827 measured (seed 0) |
| item popularity | 0.5807 published / 0.5807 measured |
| **FM official baseline** | **0.6016 published / 0.6015 measured** |
| oracle ceiling | 0.8484 |

Scoring reduces to `score = primary_agent - 0.5946` (the test-column baseline). Headroom
to the ceiling is 0.27, not 0.41 — the baseline already captures ~31% of the attainable
range.

### 1.4 The convergence rule

Self-enforced and self-reported:

```
converged = three consecutive iterations each improving validation primary by <= 0.002
         OR iteration count == 50
         OR wall clock == 6 hours
```

The scored submission is the **validation-best checkpoint** at convergence.

### 1.5 Why Milestone 1 exists at all

Milestone 2 is "let an LLM write and run Python for up to six hours, unattended." Three
things must be true before that is safe to start, and none of them were true at the start
of this session:

1. **We cannot accidentally see the hidden-test answers.** They are physically present on
   this machine (§2.1).
2. **We can tell whether a measured number is real.** If the data or scoring path drifts,
   every number the agent produces afterwards is meaningless and nothing announces it.
3. **We cannot break the competition rules by accident.** The stopping rule and the
   submission format are ours to enforce.

Milestone 1 is those three properties, built as code and pinned by tests.

---

## 2. The problem that shaped the whole design

### 2.1 The hidden test set is not hidden from our process

`starter/data.py` `load()` returns `out['test']` as **170,588 rows with their true
`long_view` label at tuple index 6**. The test period lives in the same public CSV as
everything else; the split is by date, not by file.

Worse, `starter/baseline.py` ends `run_fm()` with:

```python
return {'valid': evaluate(uva, yva, m.predict(Xva)),
        'test':  evaluate(ute, yte, m.predict(Xte))}
```

and `__main__` prints both. Observed directly:

```
  valid  GAUC 0.6671 | nDCG@5 0.5358 | primary 0.6015
  test   GAUC 0.6621 | nDCG@5 0.5286 | primary 0.5953
```

`starter/ablation_features.py` prints test scores on every line of its output.

### 2.2 Why this is a design problem and not a discipline problem

The realistic failure is not a human deciding to cheat. It is:

- **Generated code.** In M2 an LLM writes feature transforms and training loops. One
  `for split in splits:` that forgets to exclude test, and labels reach the model. No
  crash, no warning — just a suspiciously good score.
- **Agent context.** The agent reads tool output to decide what to try next. Capturing
  stdout and feeding it to the model is the *natural* implementation, so without a filter
  the default behaviour is that test scores enter the agent's reasoning and it begins
  selecting experiments on test performance. Silently.
- **Judging.** Per-iteration run logs are a graded deliverable. A judge reading logs that
  contain test scores has no way to verify we did not select on them.

So the controls are structural: reading a test label must be impossible, not discouraged.

---

## 3. Architecture

A **hybrid**. A fixed, hand-written harness owns everything that must not break. An LLM
agent (M2) writes real code for the parts that are fair game.

```
  +-----------------------------+        results        +--------------------------+
  |      FIXED HARNESS          | -------------------->  |       AGENT LOOP         |
  |  (never LLM-written)        |                        |   (LLM writes code)      |
  |                             |                        |      -- MILESTONE 2 --   |
  |  data loader + test guard   |                        |                          |
  |  evaluate/submit wrappers   |  <-------------------- |  proposes experiments    |
  |  convergence tracker        |        patch           |  writes loss/model code  |
  |  guards (deny-list, filter, |                        |  reads outcome, reflects |
  |          canary)            |                        |                          |
  +-----------------------------+                        +--------------------------+
```

**Boundary rules, from `configs/base.yaml`:**

- The LLM may modify: loss functions, model builders in `harness/models/gen/`, feature
  transforms in `harness/features/gen/`, training-loop details, hyperparameters, ensemble
  logic.
- The LLM may never modify: `harness/{data,guards,evaluate,submit,convergence,ledger,
  logger,sandbox}.py`, anything in `starter/`, anything in `tests/`.

**`starter/` is read-only.** Everything wraps it. Their `evaluate.py` remains the sole
definition of the score; their `submit.py` remains the sole definition of a well-formed
submission. A second implementation of either would be a second answer.

---

## 4. What was built, component by component

Files, with line counts:

| File | Lines | Owns |
|---|---|---|
| `harness/data.py` | 224 | path resolution, the load, **the test-label strip**, encoding |
| `harness/guards.py` | 302 | deny-list, stdout filter, log screening, leak canary |
| `harness/evaluate.py` | 64 | the one call site for the official metric |
| `harness/submit.py` | 106 | write / check / score |
| `harness/convergence.py` | 467 | the stopping rule, counters, tried-set, restart |
| `scripts/verify_setup.py` | 314 | all of the above, end to end, with a summary |
| `tests/` | 1,048 | 80 tests across five files plus fixtures |

---

### 4.1 `harness/data.py` — the strip

**What.** Wraps `starter.data.load()`. Returns the same dict, except the `test` split is
truncated to **6-tuples**. Index 6 ceases to exist, so `row[6]` raises `IndexError`.

**How.**

```python
IDX_DATE, IDX_USER, IDX_VIDEO, IDX_AUTHOR, IDX_TAB, IDX_DURATION, IDX_LABEL = range(7)
ROW_WIDTH = {TRAIN: 7, VALID: 7, TEST: 6}

def _strip_test_labels(rows):
    return [r[:IDX_LABEL] for r in rows]

def load(path=None, use_cache=True):
    splits = starter_data_module().load(resolved)
    out = {TRAIN: splits[TRAIN], VALID: splits[VALID],
           TEST: _strip_test_labels(splits[TEST])}
    splits.clear()          # drop the only other reference to the labelled rows
    del splits
    # ... width assertion per split, then cache
```

The `splits.clear()` is deliberate: after `load()` returns, no reachable object in the
process holds a labelled test row.

**Other responsibilities:**

- **Path resolution** with explicit precedence: argument > `KUAIRAND_DATA_DIR` env >
  `.env` > `paths.raw_data_dir` in `configs/base.yaml`. The `.env` reader is ~10 lines,
  no dependency, and never logs values.
- **Starter import without touching `starter/`.** That directory deliberately has no
  `__init__.py`, and its modules import each other by bare name (`from data import load`).
  So the directory goes on `sys.path` at position 0 rather than being made a package.
  Position 0 is chosen on purpose: `evaluate` is a real PyPI package name (HuggingFace),
  and starter's internal imports must win.
- **`labels(splits, split)`** raises `TestLabelAccessError` for `test`. A named exception,
  not a generic one, so a caller cannot swallow it by accident.
- **In-process cache.** A full load takes ~5 s, so a module-level dict keyed by resolved
  path is enough; no on-disk cache, no extra format, no cache-invalidation surface.

**The `encode()` decision.** `starter.data.encode()` reads a seventh field per row to
build `y`. The wrapper appends a placeholder `0` to each test row *inside the function*,
calls the organisers' encoder, then **replaces the resulting test `y` with `None`**:

```python
enc[TEST] = (X_test, None, users_test)
```

Placeholder zeros would silently pass for labels in downstream code — `y.mean()` would
return 0.0 and look plausible. `None` fails loudly at first use. This is the general
principle behind every choice in this module: **prefer a crash to a plausible wrong
value.**

**Why it matters.** This is the primary control. Everything else is defence in depth.

---

### 4.2 `harness/guards.py` — three integrity controls

#### (a) Column deny-list

Same-impression outcomes may never be model **inputs**: `play_time_ms`, `is_click`,
`is_like`, `is_follow`, `is_comment`, `is_forward`, `is_hate`, `is_profile_enter`,
`profile_stay_time`, `comment_stay_time`. Sourced from
`configs/base.yaml: leakage.deny_columns_as_features`, so the list is data, not code.

Note the label `long_view = 1` iff `play_time_ms >= duration_ms` (for videos ≤ 18 s), so
`play_time_ms` is a near-exact label proxy. These columns remain legitimate as
**auxiliary training targets** in a multi-task setup; the guard governs inputs only.

`starter/data.py` reads only `date`, `user_id`, `video_id`, `tab`, `duration_ms`,
`long_view`, so there is no leak today. The deny-list governs columns the *agent* might
add in M2.

#### (b) The starter-stdout filter

```python
_METRIC = r'(?:gauc|auc|ndcg(?:@\d+)?|primary|logloss|mrr|recall(?:@\d+)?|hit(?:rate)?(?:@\d+)?)'
_TEST_TOKEN = r'(?:\btest\b|\bhidden\b|测试)'
_SEPARATORS = re.compile(r'[_\-=:/.]+')

def contains_test_metric(line):
    if _BARE_TEST_RE.match(line):        # a bare "test  0.5953" line
        return True
    spaced = _SEPARATORS.sub(' ', line)  # so test_gauc=0.66 reads as two words
    return bool(_TEST_RE.search(spaced)) and bool(_METRIC_RE.search(spaced))
```

**Deliberately over-eager.** A line naming the held-out split anywhere near a metric word
is dropped *even if the metric belongs to validation*. Losing a line of organiser stdout
costs nothing; seeing a test score costs the run.

The Chinese token `测试` is included because the starter kit is bilingual.

`filter_stdout(text)` returns `(clean_text, n_redacted)` and **preserves line count and
order**, replacing each offending line with a marker, so a human diffing against the raw
log can see exactly what was removed.

`run_starter_script()` shells out to an organiser script with `cwd=starter/`, appends
`--data_dir`, forces UTF-8 I/O (the scripts print Chinese; Windows default encoding would
mangle it), writes the **unfiltered** combined output to `runs/raw_starter_output/`
prefixed with a `HUMAN-ONLY` banner, and returns only filtered text. That directory is
gitignored, because it contains the leak by construction.

`assert_no_test_metrics(text)` is the assertion form, for use before logging.
`assert_record_clean(record)` is a deep walk over a JSON-serialisable log record: it trips
on a `test_primary` key, on a nested `{'test': {'primary': ...}}`, and on a string value
that is a test-metric line.

#### (c) The leak canary

The validation oracle ceiling is 0.8484. Anything approaching it is a bug, not a
breakthrough. `check_canary(val_primary)` writes a JSON record to `runs/quarantine/` and
raises `LeakCanaryError` above `leakage.canary_primary_threshold` (0.80).

This is the cheapest and most general control in the repository: four lines of logic that
catch leakage from sources nobody anticipated, including ones introduced by generated code
in M2.

---

### 4.3 `harness/evaluate.py` and `harness/submit.py` — thin wrappers

**No reimplementation.** `starter/evaluate.py` is the sole definition of the score;
`starter/submit.py` is the sole definition of a well-formed submission.

Each wrapper adds exactly one thing: **a refusal to touch the test split.**

- `evaluate.evaluate_split(splits, split, scores)` raises `TestLabelAccessError` for
  `test`.
- `submit.score(path, split)` likewise. This matters because `starter/submit.py` accepts
  `--score --split test` and would print a hidden-test metric. That path is closed.
- `submit.check(path, split)` **does** work on test, because format validation needs no
  label.

`submit.py` also exposes a small CLI (`python -m harness.submit --check --split test
sub.csv`) so the safe path is as convenient as the unsafe one.

The organiser conventions are restated in the module docstring — GAUC counts only users
with `0 < npos < len(labels)` and weights by positive count; nDCG@5 includes zero-positive
users scored 0.0; gain is `2^rel - 1`; grouping key is `user_id` — so nobody re-derives
them, but they are documented as *their* conventions, not reimplemented.

---

### 4.4 `harness/convergence.py` — the stopping rule

The largest module (467 lines) because it carries the most rule risk: the rule is
self-enforced, so it is only as trustworthy as its tests.

**Three properties, enforced in code rather than prose:**

1. **Never stops voluntarily before the rule fires.** `should_continue()` is the only
   permitted stop signal; `assert_may_stop()` raises `EarlyStopError` if called early
   (unless `allow_early_stop`, which is `false`).
2. **Never continues after it fires.** `record_iteration()` raises `ConvergedError`, and
   the rejected iteration does **not** advance the counter.
3. **A restart resumes.** `ConvergenceTracker.open(path)` resumes if the state file
   exists and starts fresh otherwise — resuming is the default, not an option. State is
   saved atomically (`tmp` + `replace`) after every mutation.

**Persisted state:** rule parameters, iteration count, failed-iteration count, best
primary and its iteration, best-checkpoint reference, accumulated wall clock, full
per-iteration history, and the tried-set of content hashes.

**Strike semantics.** Gain is measured against the **running best**, not the previous
score:

```python
gain = primary - best_before          # inf for the first scored iteration
strike = gain <= epsilon + 1e-12      # so a gain of exactly 0.002 is a strike
```

A worse score is a strike and does not lower the best. This means a failed swing costs
nothing — a non-improving iteration cannot lower the validation-best checkpoint — which
is why the design says to take the highest-expected-gain untried structural change on
strike 3 rather than playing safe.

**Q3 — the ambiguity, and why the default is what it is.** `baseline_scores.json` gives
`{epsilon: 0.002, N: 3}` but not the comparison semantics. Two readings:

| Mode | Fires when |
|---|---|
| `per_iteration` (default) | every one of the last 3 scored iterations gained ≤ ε |
| `block` | `best(last 3) − best(before those 3) ≤ ε` |

They disagree on sequences like `0.6000, 0.6015, 0.6030, 0.6045`: three gains of 0.0015,
individually under ε, together 0.0045 over it.

**`per_iteration` is provably the stricter reading.** A sum of N gains cannot be ≤ ε
unless each gain is ≤ ε, so `block` firing implies `per_iteration` has already fired.
`per_iteration` therefore stops no later than `block` — the safe side of a rule we
self-enforce. The ordering claim is pinned by `test_strict_fires_whenever_block_fires`,
which checks it over multiple hand-built sequences rather than asserting it in a comment.

Both live in a single dict, `COMPARISONS`, selected by `convergence.comparison` in the
config. An organiser ruling is a one-word change.

**Q4 — failed iterations.** `record_failure()` advances the iteration counter and
`failed_iterations`, and leaves the strike streak **untouched**: an abandoned candidate
produced no validation score, so it is not a non-improving iteration. This is the standing
default; if it flips, aggressive code generation becomes much more expensive and the
repair policy has to change.

**Q5 — wall clock across restarts.** `elapsed_seconds` accumulates only while the process
is running, via `start_session()` / `end_session()` and an injectable clock. Downtime is
not charged.

The stricter reading would charge downtime, but it makes the budget depend on when a human
happens to notice a crash — a 02:00 failure found at 09:00 would exhaust six hours with no
work done. The rule reads as a compute budget, and the resource report is defined as
"agent wall clock to convergence." Restarts are logged and reported separately, so the
accounting is visible either way. **This is the decision in this milestone I would most
like a reviewer to push on.**

The clock is injected (`clock=time.monotonic` by default) so a six-hour test runs in
microseconds.

---

## 5. The tests

72 tests, 5 marked `slow`. `pytest.ini` registers the markers with `--strict-markers`;
`-m "not slow"` gives a ~20 s fast pass, and CI runs everything (~4 min).
`tests/conftest.py` loads the splits once per session (~5 s, several hundred MB) and skips
cleanly if the dataset is absent.

### 5.1 `test_contract_baseline.py` (7 tests) — the regression gate

The invariant: re-run the organisers' own fixed, deterministic, numpy-only baseline and
demand the number it produced on day one.

| Assertion | Value | Tolerance | Why that tolerance |
|---|---|---|---|
| split row counts | 1,141,112 / 124,909 / 170,588 | exact | — |
| FM valid primary | 0.6015 | ±0.001 | measured; published 0.6016 |
| item popularity | 0.5807 | ±0.001 | trains nothing, no seed variance, reproduces exactly |
| random | 0.4834 | ±0.002 | published figure is a mean over seeds 0–4; we run one seed (0.4827) |
| ladder ordering | random < pop < FM | — | catches a scrambled scoring path |

Item popularity is *not* marked slow (~10 s) so a real baseline check runs on every fast
pass.

Two properties worth noting:

- **The contract test runs the baselines through `guards.run_starter_script`** and asserts
  `redacted_lines >= 1`. A contract test that leaked would be worse than no contract test.
- It is a **smoke alarm, not a diagnosis.** It says *that* something broke, not what — but
  it says so within a minute of the change that caused it. The rule is to run it after
  every harness change; movement over 0.001 means stop.

`test_split_date_boundaries` pins the observed date extremes as **measured**, which is how
a discrepancy surfaced (§7).

### 5.2 `test_no_test_labels.py` (25 tests) — the integrity test

- test rows are 6 wide, train and valid 7
- `test_deliberately_reading_a_test_label_fails_loudly` — the acceptance criterion, an
  explicit `pytest.raises(IndexError)` on `row[6]`
- `labels()`, `evaluate_split()` and `submit.score()` all refuse test
- `encode()['test']` carries `y is None`
- filter behaviour on a synthetic transcript, with parametrised must-redact and
  must-survive line sets (the row-count line `{'train': ..., 'test': 170588}` must
  survive; every metric line naming test must not)
- separator cases: `test_gauc=0.66`, `metrics/test/ndcg@5 0.5282`
- deep log-record screening
- **the live test:** run the organisers' `baseline.py`, assert the test line was redacted
  from what we see, **and assert the human-only raw log still contains it**. Two facts in
  one test — the filter works, and the leak it prevents is real.

### 5.3 `test_convergence.py` (25 tests) — hand-built sequences

Strike counting (a gain of exactly ε is a strike; a gain just over ε is not; a worse score
is a strike and does not lower the best; gain is measured against the running best, not
the last score), the three hard stops (exact stop at iteration 50; six hours with an
injected clock), the two behavioural properties, the Q4 treatment of failures, and both Q3
readings including the ordering proof.

The restart test kills the tracker mid-run, reopens it **with a fresh clock at an
unrelated origin** (999,999 s), and asserts iteration count, strike count, best score,
tried-set and accumulated wall clock all resume — then that the strike streak *continues*
across the restart rather than starting over.

### 5.4 `test_submission.py` (15 tests) — format and round trip

The five required corruptions (wrong header, row-count mismatch, `row_id` gap,
misalignment, NaN) plus too-many-rows, Inf and an unparseable score.

Plus the independent check of our own scoring path: scores written to disk, re-read
through the organisers' validator, scored through their metric, and compared to the
in-memory score — and landing on the published random rung.

`row_id` is the positional index into `load()[split]`; `(user_id, video_id)` is **not** a
key (3.06% of test rows are repeats, up to 12×). The tests pin alignment positionally for
that reason.

---

## 6. `scripts/verify_setup.py` — the acceptance criterion

One command runs everything and prints a pass/fail summary:

```
python scripts/verify_setup.py            # everything, ~5 min
python scripts/verify_setup.py --fast     # skip the FM reproduction, ~20 s
```

11 checks: environment; data resolution; row counts; the strip; the deny-list; the stdout
filter against a **live** organiser run; the canary; the submission round trip; the
convergence rule (including a resume and a post-convergence refusal); the FM
reproduction; the pytest suite.

Each check returns a detail string; a failure is captured as data rather than crashing the
run, so one failure does not hide the other ten.

**The summary is itself screened before printing** — and that caught a real mistake. A
status line reading `... valid GAUC 0.4990 | primary 0.4827; test submission of 170,588
rows passes --check` named the held-out split next to a metric word and tripped the
filter. The guard was right and my wording was wrong. The final screen now uses
`filter_stdout` and appends a warning rather than crashing, so a leak in a check's own
output is reported instead of hidden.

Last full run: **11 passed, 0 failed, 0 skipped, 317 s**, FM at 0.6015.

---

## 7. Findings that contradict what was written down

Recorded as D1–D9 in `docs/OPEN_QUESTIONS.md`.

**The train split's first row is dated 20220409, not 20220408.** The rule and the config
both say the window is `20220408–20220421`; the standard log simply has no 8 April rows.
Row counts still match exactly, so this is a fact rather than a bug — but it was written
as an assertion, it failed, and it is now pinned as measured.

**`requirements.txt` did not describe this machine — now resolved.** The old pins
(`numpy==1.26.4`, `pandas==2.2.2`, ...) predate this interpreter and were never installed.
The full stack has since been installed (numpy 2.4.2, pandas 3.0.5, pyarrow 25.0.1, scipy
1.18.1, scikit-learn 1.9.0, lightgbm 4.7.0) and **the contract test re-run afterwards
still gives 0.6015**. `numpy` was not downgraded by the install. `requirements.txt` now
pins measured versions. `verify_setup.py` prints the interpreter and numpy version on
every run so the record is never guessed.

**Captured organiser stdout crashes a cp1252 console on print.** `run_starter_script`
captures as UTF-8 correctly, but the Windows console encoding is independent of that.
Found while verifying `submit.py --make`: the subprocess succeeded, the submission was
written, and the reporting line raised `UnicodeEncodeError`. In M2 the loop prints and
logs tool output constantly, so every such sink needs an explicit UTF-8 encoding —
otherwise a successful experiment reads as a crashed one.

**FM training wall clock is ~63 s on this machine**, not the ~110 s recorded earlier.
Relevant when sizing the agent's per-iteration timeout in M2.

---

## 8. Known gaps — what a reviewer should push on

Listed deliberately. These are the weakest points.

1. **`configs/base.yaml` is not on `agent.protected_paths`.** The convergence parameters,
   the deny-list and the canary threshold all live in the config, but the config is not
   protected — so generated code could in principle edit the rule it is judged by.
   `configs/` and `scripts/` should join the list when the patch validator is built in M2.
   Recorded rather than changed unilaterally, because that list is also part of the
   README's stated boundary.
2. **The wall-clock decision (D2) is a judgement call**, not a derivation. See §4.4.
3. **The stdout filter is heuristic.** It is over-eager by design, but it is regex over
   text. It cannot catch a test metric that reaches us through a return value rather than
   stdout — which is precisely why the in-memory strip exists as the primary control and
   the filter is defence in depth.
4. **`assert_record_clean` is not yet wired into a logger**, because the logger is M2. It
   is tested in isolation. The M2 logger must call it on every record before writing.
5. ~~No `test_determinism.py`.~~ **Added on review** (8 tests). Pins the loader
   byte-for-byte across independent loads, cache/no-cache agreement, row order at probe
   positions, encoder repeatability, that the vocabulary is built from train only, metric
   determinism, and that a seeded baseline reproduces while different seeds diverge. It
   also asserts the metric is order-sensitive, so the other determinism assertions cannot
   pass vacuously. Model determinism still waits for the M2 runners.
6. **The canary threshold (0.80) is a single scalar** with no per-metric breakdown. A leak
   that improves GAUC but not nDCG@5 could land under 0.80. Cheap to extend.
7. **The in-process split cache is keyed by path only.** If a future caller mutates the
   returned lists, every subsequent caller sees the mutation. No caller does today; a
   defensive copy or a frozen structure would close it.

---

## 9. Explicitly not built

Milestone 2 scope, deliberately untouched:

`harness/sandbox.py` · `harness/ledger.py` · `harness/logger.py` · `harness/analyse.py` ·
`harness/features/registry.py` and `base.py` · `harness/models/runners.py` ·
`agent/loop.py` · `agent/diagnose.py` · `agent/propose.py` · `agent/llm.py` · the patch
validator · causal target encoding · deterministic mode (`LLM_PROVIDER=none`).

`harness/features/gen/` and `harness/models/gen/` exist as empty directories — the
writable surface for generated code.

**No model of our own has been written.** The only models run so far are the organisers'
three baselines, and only to verify the harness. Per the design, any hand-built strong
model would be a private benchmark only; it must not seed the agent's starting state.

---

## 10. Commit history

Small commits, one per numbered deliverable, contract test passing at each.

```
M1.1  harness/data.py: strip the hidden-test label at the loader
M1.2  harness/guards.py: deny-list, stdout filter, leak canary
M1.2b guards: separators must not hide a test metric
M1.3  harness/evaluate.py and harness/submit.py: thin wrappers
M1.4  harness/convergence.py: the stopping rule, restart-proof
M1.5  tests/test_contract_baseline.py: the regression gate
M1.6  tests/test_no_test_labels.py: the integrity test
M1.7  tests/test_convergence.py: hand-built sequences, exact stop points
M1.8  tests/test_submission.py: the five corruptions, plus three more
M1.9  scripts/verify_setup.py, plus the M1 decision record
```

---

## 11. Milestone 1 acceptance, against the brief

| Required | Status |
|---|---|
| `harness/data.py` strips the test label | done; 6-tuples, `IndexError` on index 6 |
| `harness/guards.py`: deny-list, stdout filter, canary | done, all three |
| `harness/evaluate.py`, `harness/submit.py` thin wrappers | done; no reimplementation |
| `harness/convergence.py`, literal rule, switchable Q3 | done; both readings, one switch |
| `test_contract_baseline.py`, marked slow, runs in CI | done; FM, random, pop, ladder |
| `test_no_test_labels.py` | done, 25 tests including a live organiser run |
| `test_convergence.py` incl. resume-after-restart | done, 25 tests |
| `test_submission.py`, five corruptions | done, eight corruptions |
| `pytest tests/` passes | 72 passed |
| a deliberate test-label read fails loudly | `test_deliberately_reading_a_test_label_fails_loudly` |
| `scripts/verify_setup.py` end to end with a summary | done, 11 checks, all green |
| every module documents what it owns and never does | done |
| small commits, one per item | 10 commits |


---

## 12. Review responses

An external review raised twelve points. Recorded here with what was verified or changed.

### Resolved by verification (no change needed)

**`encode()` vs the strip.** The review posed a binary — placeholder labels (tuples stay
7-long, so the `len == 6` assertion is wrong) or reimplement encoding (so
`submit.py --make` breaks). A third path avoids both: the placeholder is appended to a
*temporary copy* inside `harness.data.encode()`, the organisers' encoder runs, and the
resulting test `y` is replaced with `None` before returning. Stored tuples are never
widened. Verified: stored width 6 before and after, `encode()` returns
`X(170588, 5)` with `y is None`, and `starter/submit.py --make --split test` runs to
completion and its 170,588-row output passes `harness.submit.check`. Covered by
`test_encode_returns_no_test_labels` and `test_encode_is_repeatable`.

**Stdout filter aggressiveness.** The row-count line survives, because redaction requires
a metric token *and* a split token. `{'train': 1141112, 'valid': 124909, 'test': 170588}`
has no metric word. Pinned by `test_filter_keeps_validation_and_row_counts`, which asserts
`"'test': 170588"` is still present, and used in anger by
`test_fm_baseline_reproduces`, which parses those counts out of filtered stdout.

**Wall clock across a restart.** Resumes; it does not reset.
`test_wall_clock_accumulates_across_sessions` advances a fake clock two hours, ends the
session, reopens with a fresh clock at an unrelated origin, and asserts the budget picks
up at two hours and fires at six. `elapsed_seconds` is persisted in the same file as the
strike count.

**Convergence edge cases.** First iteration: `best_before is None`, gain is `inf`, never a
strike (`test_first_score_is_never_a_strike`). Fewer than N scored iterations: cannot
converge (`test_no_convergence_before_three_scored_iterations`). Errored iteration:
advances the counter, increments `failed_iterations`, leaves the streak untouched, and
never enters the scored history (`test_a_failed_iteration_burns_one_of_fifty_but_not_a_strike`,
`test_failures_alone_never_converge_by_no_improvement`).

**Where state persists.** `ConvergenceTracker` owns its own JSON file, written atomically
(`tmp` + `replace`) after every mutation, holding the rule parameters, both counters, the
best score and its iteration, the accumulated clock, the full history and the tried-set.
Not temporary. M2's `ledger.py` handles proposals, patches and checkpoint artefacts; the
tracker stays authoritative for the counters, and the two must not both own an iteration
number.

**The item-popularity rung.** Was run. It is in the contract test at 0.5807 (matching the
published figure exactly), untagged as slow so it runs on every fast pass.

### Changed as a result

**`test_determinism.py` added.** See §8.5.

**Dependency stack installed and re-verified.** See §7.

**`requirements.txt` rewritten** to record measured versions rather than aspirational
pins.

**D10 recorded** for the cp1252 console trap.

### Accepted for Milestone 2

The M2 sketch was under-specified. Four additions, one correction and one process change,
all carried into the M2 plan: a model runner (the FM must be reimplemented against
label-stripped splits, because `run_fm` computes test metrics); `agent/llm.py` with token
accounting and `LLM_PROVIDER=none` from the first commit; building the
`run_experiment(patch_path, seed)` boundary first and developing the agent against a stub;
and checkpoint save/restore on keep-or-reject. The correction: the tried-set already
exists (`ConvergenceTracker.has_tried` / `mark_tried`, persisted and tested), so it is not
outstanding work.

**The M2 acceptance criterion was mis-specified** — it bolted an engineering result ("the
loop runs unattended") to a research result ("it beat the baseline"). These should be
separate gates. Note that CLAUDE.md §12.2 states the combined form, so splitting them is a
deviation from the spec and needs an explicit decision rather than a quiet edit.
