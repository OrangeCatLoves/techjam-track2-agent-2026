# Handover — TikTok TechJam 2026, Track 2

**For someone cloning this repository and picking up the work.** Read this first;
it is the only document you need before you can run everything.

**Deadline: 1 September, 02:00.**

---

## 1. Sixty seconds on what this is

Build two things, both graded:

1. **An autonomous ML research agent** — an LLM that inspects data, proposes an
   experiment, writes the Python for it, runs it, reads the score, and decides what
   to try next.
2. **A recommendation model** produced by that agent, scored once on a hidden test
   set.

The task: rank each user's logged video impressions by likelihood of `long_view`.
Metrics are GAUC and nDCG@5, both computed **within each user's own list**; the
primary score is their mean.

**The number to beat is 0.6015** (validation). That is the organisers' Factorization
Machine baseline.

Grading: Technical execution 35%, Innovation 20%, Impact/autonomy 20%, Feasibility
15% *(only scored if you beat the baseline)*, Presentation 10%.

---

## 2. Get it running (about 10 minutes)

### 2.1 The data

KuaiRand-Pure, 194 MB, six CSVs. Not in the repo.

```
https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz
```

Extract it **outside the repo**. You want a directory containing six `.csv` files.

### 2.2 Point the config at it

Edit `configs/base.yaml`:

```yaml
paths:
  raw_data_dir: "/your/path/to/KuaiRand-Pure/data"
```

Or set `KUAIRAND_DATA_DIR` in `.env`, which takes precedence.

### 2.3 Python

Python 3.14.0 is what everything was verified on.

```
pip install -r requirements.txt
```

Those versions are **measured, not aspirational** — the file previously pinned
versions that were never installed and have no 3.14 wheels. If you change anything
in it, re-run `scripts/verify_setup.py` and confirm the baseline still reproduces at
0.6015.

### 2.4 The LLM — read this carefully, it is the least obvious part

**There is no API key, and there must not be one.**

The agent calls the model through the **Claude Code CLI**, which runs on a Claude
Pro/Max *subscription* rather than API credits. Install Claude Code and log in:

```
npm install -g @anthropic-ai/claude-code
claude          # log in interactively once
claude --version
```

Then `.env` (copy from `.env.example`, it is gitignored):

```
LLM_PROVIDER=claude_cli
LLM_MODEL_FAST=claude-haiku-4-5-20251001
LLM_MODEL_STRONG=claude-opus-5
# ANTHROPIC_API_KEY=   <-- LEAVE THIS UNSET. See the warning below.
```

> **⚠ Do not set `ANTHROPIC_API_KEY`.** If it is present in the environment, the
> Claude Code CLI abandons your subscription and bills that API account instead —
> it prints *"ANTHROPIC_API_KEY takes precedence over your claude.ai login"*. We hit
> exactly this: the key had no credit and every call failed. `agent/llm.py` strips
> the variable defensively, but do not rely on that; leave it unset.
>
> `.env.example` is **tracked**; `.env` is not. A key was once pasted into the wrong
> one and committed. It was purged and rotated, and two tests now prevent a repeat
> (`test_env_example_carries_no_filled_secret`,
> `test_no_tracked_file_contains_an_api_key_pattern`). Put real values only in
> `.env`.

To run with no LLM at all, set `LLM_PROVIDER=none`. Everything still works — see §6.

### 2.5 Verify

```
python scripts/verify_setup.py          # ~5 min, the M1 foundation
python scripts/m2_acceptance.py --stub  # ~20 s, the M2 gate, no LLM
pytest tests/ -m "not slow"             # ~1 min
pytest tests/                           # ~8 min, everything
```

If `verify_setup.py` is green you have a working checkout.

---

## 3. The one rule that matters most

**The hidden test set is not hidden from your process, and you must not look.**

`starter/data.py` returns all 170,588 test rows **with their true labels attached**,
and `starter/baseline.py` prints test scores to stdout. Nothing stops you reading
them except the controls in this repo.

Three of them:

| control | where | what it does |
|---|---|---|
| **label strip** | `harness/data.py` | test rows are 6-tuples, not 7. `row[6]` raises `IndexError` |
| **stdout filter** | `harness/guards.py` | any line naming the test split near a metric is redacted before you or the agent see it |
| **leak canary** | `harness/guards.py` | anything scoring >0.80 on validation is quarantined; >0.68 is kept but flagged |

Raw organiser output is kept in `runs/raw_starter_output/` for humans, and is
gitignored **because it contains the leak by construction**. Never read it into a
prompt or a log. Never commit it.

`tests/test_no_test_labels.py` proves all of this, including a live run of the
organisers' own script whose test line is redacted from what we see and still
present in the human-only log.

**If you take one thing from this document:** never remove a guard to make something
pass.

---

## 4. What has been built

Two milestones complete. 356 tests, ~6,600 lines of module code, 33 commits.

### The harness (fixed, never LLM-written)

| module | owns |
|---|---|
| `harness/data.py` | data loading, **the test-label strip**, encoding |
| `harness/guards.py` | deny-list, stdout filter, leak canary, log screening |
| `harness/evaluate.py` | the one call site for the official metric; refuses test |
| `harness/submit.py` | write / check / score; refuses to score test |
| `harness/convergence.py` | the stopping rule, counters, tried-set, restart |
| `harness/models/runners.py` | FM training, losses plugged in, ensembling |
| `harness/losses.py` | the loss interface + `check_loss` validation |
| `harness/patch.py` | validates generated code before it runs |
| `harness/sandbox.py` | subprocess with time and memory ceilings |
| `harness/experiment.py` | `run_experiment` — the harness/agent boundary, and the stub |
| `harness/ledger.py` | run artefacts, checkpoint promotion and rollback |
| `harness/logger.py` | `log.jsonl` + `log.md`, resource report |
| `harness/analyse.py` | 9 data queries the agent can ask; refuses test |

### The agent (the LLM-driven half)

| module | owns |
|---|---|
| `agent/llm.py` | the model call, token metering, the per-run ceiling |
| `agent/diagnose.py` | rule-based facts computed *before* the LLM sees anything |
| `agent/propose.py` | the prompt, proposal validation, the deterministic fallback |
| `agent/loop.py` | diagnose → propose → run → decide → log → repeat |

### Design rules worth knowing before you change anything

- **`starter/` is read-only.** Everything wraps it. `starter/evaluate.py` is the sole
  definition of the score and is never reimplemented.
- **`run_experiment` never raises for an experiment failure.** Crashes, timeouts,
  memory breaches, rejected patches and canary trips are all *returned values*, so
  the loop records them and continues.
- **The LLM never supplies a number.** `diagnose.py` computes the arithmetic; the
  model reasons on top of it. An LLM asked to both measure and decide will
  confidently mis-measure.
- **Generated code can only write to `harness/models/gen/` and
  `harness/features/gen/`.** Everything else is a protected path, enforced by AST
  inspection, not string matching.

---

## 5. Results so far — read this before planning anything

| | validation primary | notes |
|---|---|---|
| random | 0.4827 | sanity floor |
| item popularity | 0.5807 | trivial baseline |
| **FM baseline (the target)** | **0.6015** | reproduced exactly |
| **deterministic control** | **0.6025** | our scripted search; +0.0010 is *inside noise* |
| agent run 1 | 0.6015 | 6 iterations |
| agent run 2 | 0.6010 | 6 iterations |
| agent run 3 | 0.6013 | 8 iterations, contaminated by a harness bug (now fixed) |

| **agent run 4** | **0.6034** | 7 iterations, **beat both baseline and control** |
| **confirmed, mean of 3 seed sets** | **0.6036** | spread 0.0006, +0.0021 over baseline |

### The result, in one paragraph

Run 4's iteration 4 found a **five-seed ensemble of the pointwise FM, blended by
within-user rank, at batch 2048**. It scored 0.6034, and re-running it with two fresh
seed sets gave 0.6040 and 0.6036 — mean **0.6036**, spread 0.0006, which is smaller
than the single-model seed std of 0.0008. Blending beat its own best member in all
three sets (+0.0006 to +0.0015), so averaging is doing real work rather than sampling
until something good appears. Full detail in `docs/RESULTS.md`.

**Zero manual interventions. 7 of 50 iterations. 0.74 of 6 hours. 237k tokens.**

**This is the scored submission** — `runs/agent-explore4/submission.csv`, validated
and committed. Exploration continues, but a later result replaces it only if it is
confirmed across several seed sets *and* better by more than 0.002. Below that
threshold we would be selecting on noise: see `docs/RESULTS.md` §7.

### The honest summary

**The agent beat the baseline on run 4, and only on run 4.** Roughly twenty
experiments across four runs; nineteen of them scored at or below baseline. The one
that worked changed *stage* rather than objective — see §6.

**Do not read that as the agent failing to think.** The reasoning is genuinely good
and it updates on evidence. From run 1, iteration 3:

> *"Both pairwise BPR and listwise softmax landed at 0.5902, and ID embedding norms
> are tiny (user 0.21, video 0.27) while tab/dur_bucket dominate. That is the
> signature of starved gradients: single-class lists (~36% of users) supply exactly
> zero gradient to any purely relative objective."*

That is correct, and it explains the whole pattern. About a third of users saw only
videos they all watched or all skipped. A ranking loss learns from *comparisons*, and
those lists contain none — so switching to a ranking objective throws away a third of
the training signal and loses more than the alignment gains.

---

## 6. What has been tried, and what happened

### Measured dead ends — do not spend iterations here

| tried | result | by whom |
|---|---|---|
| adding static feature fields (13 CWM domains) | no gain, slightly worse | organisers |
| embedding dimension k = 8/16/32 | flat | organisers |
| **pairwise BPR** (4 formulations) | all below baseline | us |
| **listwise softmax** (5 variants) | all below baseline | us |
| lambdarank / nDCG-weighted pairwise | below baseline | us |
| margin hinge, Plackett-Luce, approx-nDCG | below baseline | us |
| within-list centred logistic | ≈ baseline | us |
| hyperparameter tuning (patience, lr, l2, batch, k) | +0.0010, inside noise | us |

The consistent finding across all of it: **the objective is not the bottleneck.** The
baseline FM is well tuned and this dataset resists improvement.

### What is untried, in rough order of promise

1. **More ensembling.** This is what worked (§5). Only one configuration has been
   tried: 5 seeds, rank blend, batch 2048. Untested: more members, unequal weights
   tuned once, blending *different* model families rather than seeds of one, and
   whether the batch-2048 change or the seed-averaging carried the gain — the agent
   flagged that the two were confounded and converged before separating them.
2. **Duration-conditioned recalibration.** CLAUDE.md §9.5 flags this as a *real*
   lever because duration varies within a user's list, so the adjustment is
   non-monotone and can change the ranking. Nobody has tried it.
3. **User history sequences.** Completely unused. Each user has hundreds of training
   interactions and none of that sequence information enters the model. Highest
   ceiling, highest build cost.
4. **Multi-task auxiliary heads.** `is_click`, `is_like` etc. as extra training
   *targets* (never inputs — they are on the deny-list).
5. **Feature engineering with causal encoding.** `harness/features/` is an empty
   directory. The registry was never built.

---

## 7. Things that will bite you

Each of these cost real time to find.

**The agent has no memory across runs, and it shows.** Every one of the four runs
opened with the identical experiment — `pairwise_bpr_loss` — because each run starts
with an empty ledger and empty convergence state. It re-derives the same first idea
every time and rediscovers that it fails. Given runs only get 6–8 productive
iterations, that is expensive.

The obvious fix — writing "BPR does not work here" into `knowledge/methods.md` — is
arguably a **manual intervention**, which CLAUDE.md defines as a human changing the
agent's instructions, objective or search space, and which is how Impact is scored
(20%). Left alone deliberately. If you change it, declare it.

**The agent stops after ~6–8 iterations, every time.** Not a bug. Three rounds
without a >0.002 gain ends the run, however many of the 50 remain. So the agent gets
very few shots, and cautious tuning wastes them. `agent/diagnose.py` tells the agent
this from iteration one.

**A capability that exists but cannot run is worse than one that is absent.** This
bit twice, both times on ensembling, and both times the agent found it by using the
feature rather than a test catching it:

  1. `train_ensemble` shipped with a `seed` collision — `_run_patch` injects a seed
     into every CONFIG and the ensemble supplies its own per member. The agent
     reasoned its way to the ensemble, crashed, and went back to objectives for the
     rest of the run, learning the wrong lesson about the direction.
  2. Once it worked and *won*, the submission could not be written:
     `KeyError: 'V is not a file in the archive'`. An ensemble checkpoint stores
     `V0/W0/b0 ... Vn/Wn/bn`; `load_checkpoint` read only the single-model shape. The
     best result in the project could not be turned into the one artefact the whole
     pipeline exists to produce.

**If you add a capability, exercise the whole path** — propose, validate, train,
checkpoint, reload, score, submit — not just the part you were thinking about. Both
bugs are now regression-tested.

**The method corpus had two wrong numbers**, both of which pointed the agent *away*
from good options. `knowledge/methods.md` claimed `(user_id, date)` grouping gives
"~3" rows per list and did not match evaluation; the mean is actually 5.77 against
5.58 — a near-exact match. It also quoted the *test* set's user composition as if it
described the data we work with. **Treat every remaining estimate in that file as
suspect until you measure it** with `harness/analyse.py`.

**A NaN score used to reset the strike counter.** `nan <= 0.002` is `False`, so a NaN
counted as an improvement. `record_iteration` now refuses non-finite scores outright.
If you add another comparison site, guard it the same way.

**Windows specifics.** No POSIX `resource` module, so memory is enforced by polling
RSS with psutil. Child output goes to *files, not pipes* — a full pipe buffer
deadlocks and looks exactly like an infinite loop. And captured organiser output is
UTF-8 but the console is cp1252, so **every sink needs `encoding='utf-8'`** or a
successful experiment reads as a crash.

**Label dtype changes the metric's precision.** `starter/evaluate.py` accumulates
`(2**t - 1)` in the label's dtype, so float32 and int labels disagree in the seventh
digit. `harness.evaluate` normalises at the single call site. Do not add a second one.

---

## 8. Rules and decisions in force

Full reasoning for every one of these is in `docs/OPEN_QUESTIONS.md` (D1–D21). The
ones that change what you would do:

| decision | in force | why it matters |
|---|---|---|
| convergence semantics | **combined reading** (`comparison: block`) | a team judgement, not an organiser ruling. Makes runs longer than the strict reading |
| failed iterations | consume one of 50, **not** a strike | lets the agent attempt ambitious code |
| refit on train+validation | **disabled** | leaves ~10% of data unused; awaiting a ruling |
| restarts | resume all counters; only *active* time charged | restarting is operational recovery, not an intervention |
| leak response | >0.80 quarantine, >0.68 keep-and-flag, **never halt** | halting an unattended run submits nothing |
| supplementary files / random log | not used | conservative reading |

**Four questions are still unanswered by the organisers** — see
`docs/QUESTIONS_FOR_ORGANISERS.md`, which has a ready-to-paste block. The convergence
one is the most consequential.

---

## 9. How to run things

```bash
# the deterministic control — no LLM, no tokens, ~6 min
python scripts/control_run.py

# ONE real LLM call, with the prompt/response/patch dumped for reading
python scripts/first_contact.py --analyse          # propose only
python scripts/first_contact.py --analyse --train  # and run it

# a full agent run
python -c "from agent.loop import AgentLoop; \
           print(AgentLoop(run_dir='runs/my-run').run(max_iterations=12))"

# validate any submission with the organisers' own checker
python -m harness.submit --check --split test runs/<id>/submission.csv
```

Each run writes to `runs/<id>/`: `log.md` (human-readable, this is what judges read),
`log.jsonl`, `ledger.jsonl`, `convergence.json`, `summary.json`, `resources.md`,
`patches/` (the code the agent wrote), and `submission.csv`.

**Committed:** logs, patches, transcripts. **Not committed:** checkpoints (`.npz`,
2.4 MB each), intermediate submissions (4.4 MB each), and anything under
`raw_starter_output/`. The final submission is added explicitly with `git add -f` so
there is never ambiguity about which file was scored.

---

## 10. What still needs doing

**Required deliverables not yet written:**

- `README.md` — still has `TODO` placeholders in the results, resource and team
  tables
- Devpost description
- The results table with final numbers
- The resource table (tokens, wall clock, iterations, **manual interventions**)

**Where to spend effort if you want the score to move:**

1. **Ensembling**, properly explored. Cheapest credible path, now that it works.
2. **Duration-conditioned recalibration.** Flagged as a real lever, never attempted.
3. Give the agent more iterations by finding one substantial gain early — the strike
   rule kills runs that open with small changes.

**Where not to:** another loss function. Twenty experiments say the objective is not
the bottleneck.

---

## 11. Documents worth reading, in order

| file | what it is |
|---|---|
| `docs/TODO.md` | **what is left and who owns it — start here** |
| `docs/RESULTS.md` | **the numbers, and how the winning experiment was reached** |
| `CLAUDE.md` | the spec. Overrides general ML practice where they conflict |
| `docs/MILESTONE_1.md` | the foundation: safety, measurement, the contract test |
| `docs/MILESTONE_2.md` | the loop: every module, every bug found, every reversal |
| `docs/OPEN_QUESTIONS.md` | D1–D21, every decision and its reasoning |
| `docs/QUESTIONS_FOR_ORGANISERS.md` | what is still unanswered |
| `docs/M2_CONTRACT.md` | the frozen harness/agent interface |
| `knowledge/methods.md` | the agent's method corpus — **contains estimates; verify them** |
| `runs/*/log.md` | what the agent actually did and why |
