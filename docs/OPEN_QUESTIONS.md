# Open questions

Version 3. Q1 and Q6 are resolved from the starter kit source. Seven remain open.
Milestone 1 added nine build decisions at the end of this file (D1-D9); three of them
are the working answers to Q3, Q4 and Q5.

Post the remaining six to the organiser channel. Update this file and the matching
config value in the same commit when an answer arrives.

Status: OPEN / ANSWERED / RESOLVED FROM SOURCE / ASSUMED (no reply, default in force)

---

## RESOLVED FROM SOURCE

### Q1 — Which metric spec is correct?

The Limits row of the PDF Constraints table says `NDCG@10 / Recall@50, click = positive`.

**RESOLVED.** `starter/data.py` line 5 reads `LABEL = 'long_view'`. `starter/evaluate.py`
computes GAUC and nDCG@5 with primary = their mean. `baseline_scores.json` confirms
`"label": "long_view"` and `"metrics": ["GAUC", "nDCG@5"]`. The Limits row is stale
text from an earlier draft. No action needed.

### Q6 — How does `evaluate.py` break ties?

**RESOLVED.** `auc()` implements Mann-Whitney U with proper average-rank tie
correction: equal scores receive the mean of their rank positions. `evaluate()` sorts
each user's list by `-score` before computing nDCG. Ties are handled correctly and
deterministically. Emitting tied scores is safe, though still not useful.

### Conflict B — Is AliCCP required?

A webinar slide says "The full pipeline on AliCCP is required; KuaiRand is an optional
bonus." **RESOLVED.** Stale slide from an earlier version. The starter kit is
KuaiRand-Pure only and the PDF, updated 27 August, never mentions AliCCP.

### Data path

**RESOLVED.** `baseline.py`, `submit.py` and `ablation_features.py` all accept
`--data_dir`, and `data.load(path)` takes the path as an argument. Set
`KUAIRAND_DATA_DIR` in `.env` and pass it through. No filesystem junction required.

---

## STILL OPEN

### Q2 — May the winning configuration be refitted on train + validation?

The rule says the scored submission is the "validation-best checkpoint." A model
refitted on train + validation was never itself scored on validation, so it is not
obviously that checkpoint. Standard practice says refit; a literal reading forbids it.

- **Status:** OPEN
- **Default:** No. `selection.refit_on_train_val: false`. Code path exists, disabled.
- **If it flips:** enable the flag, rerun the final refit, revalidate the submission.
- **Answer:**

---

### Q3 — Exact convergence semantics

`baseline_scores.json` gives `{"epsilon": 0.002, "N": 3}` and the kit README says
"three consecutive iterations where validation primary improves by no more than 0.002."
It does not specify whether the comparison is `best(last 3) - best(before those 3)` or
`max per-iteration gain over the last 3`. These stop at different points.

The convergence rule is **not** implemented anywhere in the starter kit. It is ours to
implement.

- **Status:** OPEN. Implemented, both readings, behind one switch. See **D1** below.
- **Default:** the stricter of the two readings, `per_iteration`
- **Config:** `convergence.epsilon`, `convergence.n_consecutive`, `convergence.comparison`
- **Answer:**

---

### Q4 — Do failed or abandoned iterations count?

If a candidate errors out and is abandoned, does it consume one of the 50, and does it
count as a non-improving iteration for the three-strike window?

- **Status:** OPEN. Default implemented in `record_failure()`. See **D3** below.
- **Default:** counts toward the 50 cap; does NOT count as a non-improving iteration
- **If it flips:** repair policy changes materially. A failed iteration that burns a
  strike makes aggressive code generation much more expensive.
- **Answer:**

---

### Q5 — Does a crash-and-restart affect iteration count or the convergence window?

Per the webinar, restarts and operational recovery are not manual interventions.
Confirming the accounting treatment in writing.

- **Status:** OPEN (the webinar answer is second-hand and load-bearing). Resume is
  implemented and tested; the wall-clock treatment needed a further choice, **D2**.
- **Default:** No effect. State resumes exactly from the ledger. Counters never reset.
  Only *active* agent time is charged to the six hours.
- **Config:** `convergence.reset_state_on_restart: false`
- **Answer:**

---

### Q7 — Are the supplementary KuaiRand files in scope?

`kuairand_video_captions.csv` and `kuairand_video_categories.csv` are published by the
same authors at a separate Zenodo record. The rule says training must rely only on
"the KuaiRand datasets listed below."

- **Status:** OPEN
- **Default:** do not use. `leakage.use_supplementary_files: false`
- **Note:** low marginal value; `video_features_basic_pure.csv` already carries tags.
- **Answer:**

---

### Q8 — May `log_random_4_22_to_5_08_pure.csv` be used?

The randomised-exposure log holds 1.18M impressions but spans 20220422 to 20220508,
overlapping the hidden test window. The starter kit README suggests it as an extra
**unbiased validation set** to check for overfitting to biased traffic, which is a
lower-risk use than training.

- **Status:** OPEN
- **Default:** do not use. `leakage.use_random_exposure_log: false`
- **If allowed:** validation only, strictly dates < 20220429, never training. It
  provides no data before the validation window, so it cannot extend the training
  period at all.
- **Answer:**

---

### Q9 (new) — Are the bonus benchmarks accompanied by splits and baselines?

The starter kit contains only KuaiRand-Pure code and scores. KuaiRand-1k and 27k are
named as bonus benchmarks in the PDF, but no splits, baseline, or scores are provided.

- **Status:** OPEN
- **Default:** do not attempt. 27k is 322M rows and is not feasible on CPU in the window.
- **Answer:**

---

## Definition in force: manual intervention

Per the organiser webinar, a manual intervention is a human changing the agent's
instructions, objective, or search space. Restarting a crashed process, clearing a
lock, or freeing disk is operational recovery and is not an intervention.

Both categories are counted and reported separately in the final resource table.

---

## Decisions taken while building Milestone 1

Nine choices the code now embodies. Each is a conservative reading of something the
organisers have not ruled on, or a measured fact that contradicted a written
assumption. If an answer arrives, change the config value and the row here in the
same commit.

### D1 — Q3 is implemented as `per_iteration`, and here is why that is the stricter reading

`harness/convergence.py` implements both readings behind
`convergence.comparison` in `configs/base.yaml`:

| Mode | Fires when |
|---|---|
| `per_iteration` (default) | every one of the last 3 scored iterations improved the running best by <= 0.002 |
| `block` | `best(last 3) - best(before those 3) <= 0.002` |

A sum of three gains cannot be <= epsilon unless each gain is <= epsilon, so
**`block` firing implies `per_iteration` has already fired**. `per_iteration`
therefore stops no later than `block`, which is the safe side of a rule we
self-enforce. It is also the reading written in CLAUDE.md section 3.4. The ordering
claim is pinned by `test_strict_fires_whenever_block_fires`.

The two readings disagree on sequences like `0.6000, 0.6015, 0.6030, 0.6045`: three
gains of 0.0015 each, individually under epsilon, together 0.0045 over it. Both
behaviours are tested.

### D2 — Wall clock across a restart charges active agent time only (Q5)

`elapsed_seconds` accumulates only while the process is running, and is persisted
and resumed. Time when the process is dead is not charged to the six hours.

Charging downtime would be the stricter reading of "6 hours", but it makes the
budget depend on when a human happens to notice a crash: a 02:00 failure found at
09:00 would exhaust the budget with no work done. The rule is a compute budget, and
CLAUDE.md section 10.2 reports "agent wall clock to convergence". Restarts are
logged and reported separately, so the accounting is visible either way.

### D3 — A failed iteration burns one of the 50 and no strike (Q4)

Implemented exactly as the standing default. `record_failure()` advances the
iteration counter, increments `failed_iterations`, and leaves the strike streak
untouched: an abandoned candidate produced no validation score, so it is not a
non-improving iteration. If the organisers rule that failures burn strikes, aggressive
code generation gets much more expensive and the repair policy must change.

### D4 — The harness refuses `--score --split test` outright

`starter/submit.py` accepts `--score --split test` and would print a hidden-test
metric. `harness/submit.py` and `harness/evaluate.py` raise `TestLabelAccessError`
for the test split instead. Format checking (`--check`) still works on test, because
it needs no label.

### D5 — `harness.data.encode()` returns `y = None` for the test split

The organisers' encoder reads a seventh field per row to build `y`. The wrapper
appends a placeholder `0` internally and discards the resulting column, returning
`None`. Placeholder zeros would silently pass for labels in downstream code; `None`
fails loudly.

### D6 — Gap: `configs/base.yaml` was not on `agent.protected_paths` (RESOLVED)

The convergence parameters, the deny-list and the canary threshold all live in the
config, but the config is not protected, so generated code could in principle edit
the rule it is judged by. **Resolved.** `configs/`, `scripts/`, `harness/losses.py` and
`harness/models/runners.py` are now on the list. Raised in external review, which
also asked for a test asserting a patch touching a protected path is rejected — that
test belongs with the patch validator in M2a and is listed in `docs/M2_CONTRACT.md`.

### D7 — The train split's first row is dated 20220409, not 20220408

The rule and the config both say the train window is `20220408-20220421`. The
standard log simply contains no rows dated 8 April. Not a discrepancy with the
rule, and not a bug: the row count still matches exactly. Pinned as measured in
`test_split_date_boundaries` so that a future loader change cannot move it quietly.

### D8 — `requirements.txt` now records the verified environment (RESOLVED)

The old pins (`numpy==1.26.4`, `pandas==2.2.2`, ...) predated this interpreter and
were never installed here. Resolved by installing the full stack and re-verifying:

| | pinned before | installed and verified |
|---|---|---|
| python | (unstated) | 3.14.0 |
| numpy | 1.26.4 | 2.4.2 |
| pandas | 2.2.2 | 3.0.5 |
| pyarrow | 16.1.0 | 25.0.1 |
| scipy | 1.13.1 | 1.18.1 |
| scikit-learn | 1.5.0 | 1.9.0 |
| lightgbm | 4.3.0 | 4.7.0 |
| PyYAML | 6.0.1 | 6.0.3 |
| pytest | 8.2.0 | 9.1.1 |
| psutil | 5.9.8 | 7.2.2 |

**The contract test was re-run after the install and still passes**: FM validation
primary 0.6015, and the full ladder in order. `numpy` was not downgraded by the
install. `requirements.txt` now pins the measured versions rather than aspirational
ones. `anthropic` is still not installed; that is Milestone 2.

`scripts/verify_setup.py` prints the interpreter and numpy version on every run so
the record is never guessed.

### D9 — Contract tolerances differ by baseline, on purpose

| Baseline | Published valid primary | Tolerance | Why |
|---|---|---|---|
| FM | 0.6016 | 0.001 | measured 0.6015; the number to beat |
| item popularity | 0.5807 | 0.001 | pure statistics, no training, no seed variance; measured 0.5807 exactly |
| random | 0.4834 | 0.002 | the published figure is a mean over seeds 0-4; we run one seed, measured 0.4827 |


### D10 — Captured organiser stdout must never be printed to a cp1252 console

`harness.guards.run_starter_script` captures subprocess output as UTF-8, which is
correct: the starter kit's messages are bilingual and contain Chinese. But printing
that captured text to a Windows console still raises
`UnicodeEncodeError: 'charmap' codec can't encode characters`, because the console
encoding is cp1252 and is independent of how the text was read.

Found while verifying `submit.py --make` end to end: the subprocess succeeded, the
submission was written correctly, and the *reporting* line crashed.

Consequence for Milestone 2: the agent loop will print and log tool output
constantly. Anything that writes captured starter text to stdout or to a file must
set an explicit UTF-8 encoding, and log files must be opened with
`encoding='utf-8'`. A crash here would be read as a failed experiment when the
experiment actually succeeded.

### D11 — The M2 acceptance gate was split (deliberate spec revision)

`CLAUDE.md` §12.2 originally gated Milestone 2 on "a 10-iteration unattended run
produces a valid submission **and beats 0.6016 on validation**." That bolts a research
result onto an engineering one, and the two do not arrive together.

**Why it was a spec bug rather than a standard worth holding.** §12.2 was written before
the starter kit was read. At that point the plan named LightGBM as the workhorse and
assumed a first-pass agent would find easy gains. Reading the kit reversed both: the
organisers' published ablations show static features (0.5940 vs 0.5950) and embedding
capacity (k = 8/16/32, flat) are measured dead ends, and §9.3 now says a GBDT is probably
a trap on 27K x 7.5K IDs. That removed most of the cheap moves. The agent ships at M2
with only a pointwise loss available, so its realistic move set is hyperparameters, or
writing a pairwise objective unprompted. Gating a milestone on an outcome this document's
own strategy section calls unlikely means a working loop gets recorded as a failure
because the science did not land on schedule.

**In force:**

| Milestone | Gate |
|---|---|
| M2 | ten iterations unattended, valid submission passing `--check`, survives kill-and-restart, human-readable log. **Score is not a gate.** |
| M3 | three iterations targeting different pipeline stages with reasoning that changed after evidence, **and beats validation primary 0.6015** |

Nothing about the competition rules changes. The convergence rule, the 50-iteration cap,
the 6-hour ceiling and the validation-best checkpoint are untouched; this is our own
internal milestone accounting.

Raised in external review, accepted, and recorded here as a decision rather than a
slipped target.

### D12 — Label dtype silently changed the metric's precision

`starter/evaluate.py`'s `ndcg_at_k` computes `(2 ** t) - 1` on whatever type it is
handed and accumulates in that type. So the same predictions scored with float32
labels and with Python int labels disagree in the seventh significant digit:

```
same scores, float32 labels -> 0.5869309
same scores, int   labels   -> 0.5869302535796688
```

Both spellings were live in this repo. The organisers' `run_fm` passes `y` straight
from `encode()`, which is float32, and our `evaluate_split` reads Python ints off the
split rows. Found when a checkpoint round-trip test compared the two routes and
failed at 6e-7 despite bit-identical predictions.

**No decision was ever at risk** — 7e-7 is four orders of magnitude below the 0.002
convergence epsilon. But two spellings of one number is a phantom regression waiting
for an afternoon, and generated code will call `evaluate` from everywhere.

**Fix:** `harness.evaluate.evaluate` normalises integral labels to `int` before
delegating. The metric itself is untouched; this is input hygiene at the single call
site. Only integral values convert, so graded relevance would pass through — and
`long_view` is binary by definition, so it is the identity on real data. Pinned by
`test_label_dtype_does_not_change_the_score` and
`test_trainer_and_submission_paths_agree_exactly`.

The published baseline numbers are unaffected at the quoted precision; the FM still
reproduces at 0.6015.

### D13 — Canary escalation, and why the second trip stops the run

One trip: quarantine, record the patch hash, roll back, continue. Two trips in one
run: hard stop.

The canary only catches leaks scoring above 0.80. A systematic leak path also
produces sub-threshold results — 0.72, 0.75 — that look like genuine breakthroughs
and would be **kept and submitted**. The risk of continuing after a second trip is
therefore not wasted iterations but a quarantined result masking a kept one from the
same cause. One trip can be a strange patch; two is a pattern, and a pattern points
at something the harness hands out rather than at any single patch.

This costs a manual intervention. That is the right trade: "we detected a leak and
stopped" beats "we detected a leak twice and kept going". Full reasoning in
`docs/M2_CONTRACT.md` section 6.

### D14 — Baseline reproduction does not burn a strike, but does count toward the 50

Confirmed by test, not by reading. The agent reproduces the baseline itself
(CLAUDE.md 6.4), which scores ~0.6015 and improves on nothing. The tracker treats it
correctly because the first scored iteration has no prior best, so its gain is
infinite by definition.

**The trap that would break this:** seeding `initial_best` with the published 0.6015.
The reproduction then gains exactly zero and is strike one before any experiment has
been proposed — a third of the strike budget lost to an off-by-one, discovered during
the scored run. The agent's tracker is never seeded; it learns the baseline by
reproducing it. Pinned by `test_seeding_the_initial_best_would_burn_a_strike`.

It **does** consume one of the 50. It is a real experiment cycle: code written, run,
scored. Only the strike question was ambiguous.

### D15 — Per-iteration timeout reduced from 25 to 12 minutes

A full reference FM run measures ~63 s on this machine, so 25 minutes was a ~24x
backstop and one runaway experiment would cost 7% of the six-hour budget. Twelve
minutes is ~11x: still generous for a genuinely heavier experiment such as a sequence
or multi-task model, and a runaway now costs 3%.

A timeout is not fatal — the recovery path retries once at a 30% subsample before the
candidate is abandoned.

Related: measured cost is now reported to the agent in `diagnostics['cost']`, with
the reference figure alongside it, so "expensive" is a comparison it can make rather
than a word it has to interpret. At ~63 s per experiment, twenty iterations is about
21 minutes of training — the six-hour ceiling is dominated by LLM latency and
overhead, not by compute.
