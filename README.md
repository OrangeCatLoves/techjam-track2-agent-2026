# Autonomous ML Research Agent — TikTok TechJam 2026, Track 2

An autonomous machine-learning research agent for recommender systems, evaluated on
the KuaiRand-Pure benchmark.

---

## Project overview

An LLM-driven research agent that reads the problem, inspects the data with its own
analysis tools, forms a hypothesis, **writes real Python** for a loss function, a
feature or a model change, runs it in a sandbox, reads the score, and decides what to
try next. It repeats until a literal convergence rule stops it. A fixed harness owns
everything that must not break — the data loader, the guards, the evaluator, the
convergence tracker, the ledger — and the agent owns everything that is fair game.
The scored model is whatever that loop produced.

The part we would point a judge at is not the score. **The loader hands us the hidden
test labels and we refuse to look**: `harness/data.py` strips them before anything
downstream exists, so feature code physically cannot read them, and
`tests/test_no_test_labels.py` proves it. Alongside that, we ran a 30-configuration
scripted search over the same harness as an honest control — it reached 0.6025, so
the agent's contribution is the distance above *that*, not above the published
baseline. And when experiments failed, which was most of the time, we measured *why*
and wrote it down. `docs/RESULTS.md` records eleven closed directions with mechanisms
attached, including the largest single effect we found — worth more than our own
result — which we are not permitted to use.

## Results

| | GAUC | nDCG@5 | Primary | Delta vs baseline |
|---|---|---|---|---|
| Official baseline (hidden test) | 0.6610 | 0.5282 | 0.5946 | — |
| Ours, run 4 iteration 4 (validation) | 0.6700 | 0.5367 | **0.6034** | +0.0018 |
| Ours, confirmed over 3 seed sets (validation) | — | — | **0.6036** | **+0.0021** |
| Ours (hidden test) | *not computed* | *not computed* | *not computed* | — |

**On the blank test row.** We are not permitted to score the hidden test set, and the
harness refuses to: `harness/data.py` strips the test labels at load, so `out['test']`
holds 6-tuples and `row[6]` raises `IndexError`. That row is empty by design, not by
omission. The organisers' own `baseline.py` prints a test metric; our wrapper filters
it before it can reach a log or an LLM prompt, and the raw output is kept in a
human-only file that is never committed.

Scoring formula: `score_dataset = mean(delta(GAUC), delta(nDCG@5))`, which for this
benchmark reduces to `primary_agent − 0.5946`.

### Resource usage

| | |
|---|---|
| Iterations used (of 50) | **7** |
| Agent wall clock to convergence | **0.74 h** of the 6 h cap |
| LLM tokens, input + output | **237,365** (188,312 in / 49,053 out) |
| GPU-hours | **0** |
| GPU-hours | 0 |
| **Manual interventions** | **0** |
| Operational restarts (not interventions) | **0** |
| Recovery events (in-run, automatic) | 1 |

Manual intervention is defined per the organiser webinar as a human changing the
agent's instructions, objective, or search space. Restarting a crashed process,
clearing a lock, or freeing disk is operational recovery and is counted separately.

---

## Setup

### What you need

| | |
|---|---|
| **Python** | **3.14** is what everything here was built and verified on. 3.11+ very likely works, but the pins in `requirements.txt` are 3.14-era wheels — see the note below if your interpreter is older. |
| **GPU** | None. Everything is CPU and numpy. Total GPU-hours for this submission: **0**. |
| **Disk** | ~2 GB for the dataset and cached frames. |
| **RAM** | ~4 GB. Training the FM peaks around 1.5 GB. |
| **LLM access** | **Only needed to run the agent itself.** Everything else — the baseline, the tests, the deterministic control, reproducing our submission — runs with no LLM and no account. See *Running without any LLM access* below. |

> **A note on the pins.** `requirements.txt` records the versions this was *verified*
> on (numpy 2.4.2, pandas 3.0.5, scipy 1.18.1, lightgbm 4.7.0), not aspirational
> minimums. Those are Python 3.14 wheels. On an older interpreter, install
> `numpy`, `pandas`, `scipy`, `scikit-learn`, `lightgbm`, `PyYAML`, `pytest` and
> `psutil` unpinned — the starter kit and the FM baseline are **numpy-only**, so the
> number that matters (validation primary 0.6015) does not depend on the rest.

### 1. Get the data

Download KuaiRand-Pure (194 MB) from
`https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz`
(MD5 `0820331067a3784d9691136f772b35a7`) and extract it **outside this repository**.

You should end up with six CSVs in `<somewhere>/KuaiRand-Pure/data/`. The repository
does not contain the dataset.

### 2. Install

```
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure — one variable

```
cp .env.example .env
```

Set exactly one thing:

```
KUAIRAND_DATA_DIR=/absolute/path/to/KuaiRand-Pure/data
```

That is the whole configuration. `configs/base.yaml` deliberately ships with an
empty `raw_data_dir` so a fresh clone fails with a clear message rather than
pointing at a contributor's machine.

> **Do not set `ANTHROPIC_API_KEY`.** Leave it unset, and unset it if your shell
> already exports one. The agent drives an LLM through the **Claude Code CLI on a
> subscription**; if that variable is present the CLI abandons the subscription and
> bills an API account instead. Nothing in this repository needs an API key, and
> nothing you need to run as a judge uses an LLM at all.

### 4. Verify

```
python scripts/verify_setup.py            # everything, about 10 minutes
python scripts/verify_setup.py --fast     # skips the FM reproduction, about 8 minutes
```

Eleven checks: the environment, the data directory, the split row counts
(1,141,112 / 124,909 / 170,588), the hidden-test label strip, the column deny-list,
the stdout filter against a live organiser run, the leak canary, the submission
round trip, the convergence rule, the FM baseline reproduction, and the pytest
suite. It prints a summary and ends with the number to beat. **If this fails,
nothing downstream is trustworthy.**

The regression gate on its own:

```
pytest tests/test_contract_baseline.py    # row counts and the baseline ladder
pytest tests/ -m "not slow"               # 350 tests, about 8 minutes
```

---

## Results dashboard

One self-contained HTML file, generated from the run artefacts:

```
python scripts/build_dashboard.py     # writes dashboard.html
```

Open it by double-click. No server, no network, no install, no external assets.
It shows the scoreboard, the best result per pipeline stage, and all 58 scored
experiments -- click any row for the agent's own hypothesis, verbatim from its log.

**Every figure on the page is read from `runs/*/log.jsonl`, never typed in**, so it
cannot drift from the runs it describes. It is a view: it trains nothing and scores
nothing. It also cannot show a hidden-test metric, because none exists in the
artefacts it reads.

---

## Running without any LLM access

**A judge does not need an LLM, an API key, or a Claude subscription** to check
anything in this submission. Three things run standalone:

```
python scripts/verify_setup.py         # the full safety and contract suite
python scripts/control_run.py          # the deterministic control, no LLM, no tokens
pytest tests/ -m "not slow"            # 350 tests
```

`scripts/control_run.py` is the deterministic fallback described in CLAUDE.md §13
and decision D20: a 30-configuration scripted search over the same harness, the same
guards, the same logging, and **zero LLM calls**. It produces a complete, valid
submission on its own. It is also the honest control for the agent — it reached
**0.6025**, so the agent's contribution is the distance above that, not above the
published 0.6015.

Every patch it writes carries `NOT the agent` in its docstring so no reader can
mistake one for the other.

---

## Reproducing our results

### The scored submission

Already in the repository, no run required:

```
runs/agent-explore4/submission.csv
python -m harness.submit --check --split test runs/agent-explore4/submission.csv
```

### The agent itself (needs a Claude Code subscription)

```python
from agent.loop import AgentLoop
print(AgentLoop(run_dir='runs/my-run-1').run(max_iterations=12))
```

Install the Claude Code CLI, log in, and leave `ANTHROPIC_API_KEY` unset. A run
costs roughly 25–50 minutes and ~240k tokens against your own subscription.

**The agent has no memory across runs, so a fresh run will not reproduce ours.** It
re-derives its approach each time and usually opens with pairwise BPR, which fails.
That is the design, not a defect — see `docs/RESULTS.md`.

### The individual findings

Each is a standalone script that prints its own table and needs no LLM:

```
python scripts/probe_refit.py          # refit on train+valid is worth +0.0035
python scripts/probe_recency.py        # drift is real; recency weighting cannot exploit it
python scripts/probe_gbdt.py           # a GBDT is the first genuinely diverse blend member
python scripts/probe_watchtime.py      # watch-time targets do not help
python scripts/probe_snapshots.py      # snapshot ensembling does not help
python scripts/probe_list_shape.py     # list construction does not rescue ranking losses
python scripts/probe_multitask.py      # auxiliary heads do not help
python scripts/probe_covisitation.py   # co-visitation is redundant with the FM cross
```

Run logs are at `runs/<run_id>/log.md` (human-readable) and `log.jsonl`
(machine-readable). No log anywhere contains a hidden-test metric, by construction.

---

## Architecture

A hybrid. A fixed, hand-written harness owns everything that must not break. An LLM
agent writes real code for the parts that are fair game.

**The agent may modify:** feature transform functions, model builder functions,
objectives and losses, hyperparameters, sample-weight functions, ensemble logic.

**The agent may never modify:** data loading and splits, the leakage guards, the
sanitised materialisation step, the `evaluate.py` and `submit.py` wrappers, the
convergence tracker, the ledger and logger, the sandbox, or the tests. The patch
validator rejects any diff touching a protected path.

```
  +------------------------------+      results       +------------------------+
  |        FIXED HARNESS         | -----------------> |       AGENT LOOP       |
  |      (never LLM-written)     |                    |    (LLM writes code)   |
  |                              |                    |                        |
  |  data.py   strips test labels|                    |  analyse(spec)  - EDA  |
  |  guards.py deny-list, canary |                    |  diagnose  (rules)     |
  |  losses.py the loss contract |                    |  propose   (JSON)      |
  |  features/ causal window     | <----------------- |  write real Python     |
  |  evaluate.py / submit.py     |       patch        |  read outcome, reflect |
  |  convergence, ledger, logger |                    |                        |
  |  sandbox + patch validator   |                    +------------------------+
  +---------------+--------------+
                  |
                  v
   validation-best checkpoint  ->  submission.csv  ->  scored once on hidden test
```

The boundary is enforced, not just documented: the patch validator rejects any diff
touching a protected path, and `tests/test_patch_validation.py` covers it. Five
pipeline stages are open to the agent — `objective`, `model`, `features`, `sampling`,
`ensemble`.

### The agent loop

Inspect (autonomous EDA via a bounded `analyse` tool) → diagnose from computed facts →
propose one experiment with a stated hypothesis → write the code → validate the patch →
execute in a sandbox → evaluate with the official script → keep or roll back → log →
update the convergence clock.

---

## Leakage safety

The hidden-test period lives in a public file, so integrity is enforced structurally
rather than by convention.

- **Physical isolation.** Materialised parquet frames contain no same-impression
  outcome columns. The test frame contains no label. Generated code is given only
  these paths; the raw CSVs are never on it.
- **Causal encoding.** Every historical statistic uses an expanding window. For a row
  on date `d`, statistics come from dates strictly before `d`. Enforced by the feature
  API, not by developer discipline.
- **Excluded by default.** `video_features_statistic_pure.csv` holds per-video averages
  computed over a month that spans the test window, including near-direct label
  proxies. It is not used.
- **Leak canary.** Any configuration scoring above 0.80 primary on validation is
  quarantined and flagged. Nothing legitimate reaches that.

---

## Competition rules as implemented

- Convergence is implemented literally: three consecutive iterations gaining 0.002 or
  less, or 50 iterations, or 6 hours. The run never stops voluntarily before this and
  never continues after it.
- The scored submission is the literal validation-best checkpoint. Diagnostic temporal
  folds inform the agent's next proposal; they never override the validation winner.
- Refitting on train + validation is implemented but disabled pending an organiser
  ruling (see `docs/OPEN_QUESTIONS.md`, Q2).
- Restarts resume iteration count, strike count, and tried-set from the ledger.
  Counters are never reset.
- No cached LLM responses on the scored run.

---

## Limitations and what we would improve

**The score moved very little, and we can say exactly why.** Run 4 beats the official
baseline by +0.0021 and our own scripted control by +0.0011. Across roughly forty
experiments, eleven directions were closed with measurements: every ranking objective
(43 agent iterations plus a controlled 3x3 ablation, none beating pointwise logloss),
ensembling in four degrees of freedom, hyperparameters, watch-time targets, causal
features, list construction, recency weighting, multi-task heads, and co-visitation.
That is a well-mapped dead end rather than an unexplored space, but it is still a
small delta.

**The largest effect we measured is one we are not allowed to use.** Refitting the
winning configuration on train + validation is worth **+0.0035** on a protocol shifted
one week earlier — every seed, non-overlapping distributions, about four times the
noise floor. Scaled to the real split it is roughly +0.0014 to +0.0035, comparable to
or larger than our entire result. It is disabled behind
`selection.refit_on_train_val: false` pending an organiser ruling that never arrived.
See `docs/RESULTS.md` and `docs/QUESTIONS_FOR_ORGANISERS.md`.

**The agent over-invests in one stage.** 43 of ~56 iterations targeted the objective,
partly because that was the only stage with a rich API until late. Opening the feature
stage changed its behaviour immediately — it derived causal target encoding unprompted
from the field-norm diagnostics — which suggests the corpus and the capability surface
steer it more than we intended.

**No tree search.** Three of the four references in the problem statement (MLE-bench,
AIDE, AI Scientist-v2) describe agents that keep every attempt as a node and choose
which to expand. Ours is a greedy loop with a single incumbent and rollback. AIDE's
draft/debug/improve policy and AI Scientist-v2's replication and aggregation node
types map cleanly onto things we do by hand, and that is the clearest gap between this
submission and the literature it was pointed at.

**Validation is a weaker problem than test.** Validation lists average 5.58 items per
user against test's 7.15, and the oracle ceilings differ (0.8484 against 0.8645), so
the split we select on is not the split we are scored on.

**Single architecture.** Everything is a factorization machine. A GBDT on causal
features turned out to be the first genuinely diverse blend member we found
(agreement 0.79 against 0.90 for seeds), and we found it too late to build on.

---

## Repository layout

| path | what it is |
|---|---|
| `starter/` | The organisers' code. **Read-only.** Never modified, only wrapped. |
| `harness/` | The fixed harness. Data loading with test-label stripping, guards, the loss contract, causal features, evaluation, submission, sandbox, patch validator, convergence, ledger, logger. The agent may never edit these. |
| `harness/models/gen/`, `harness/features/gen/` | Where the agent's generated code lands. |
| `agent/` | The loop: `diagnose.py` (rule-based facts), `propose.py` (prompt and schema), `llm.py` (transport and token metering), `loop.py`. |
| `scripts/` | `verify_setup.py`, `control_run.py` (the no-LLM control), and one self-contained probe per finding. |
| `tests/` | 350 tests. Contract, leakage, submission, causal window, guards, convergence, patch validation, determinism. |
| `runs/` | Per-run artefacts: `log.md`, `log.jsonl`, `patches/`, `submission.csv`. |
| `docs/` | `RESULTS.md` (every number), `HANDOVER.md` (setup and state), `OPEN_QUESTIONS.md` (D1–D22), `QUESTIONS_FOR_ORGANISERS.md`. |

---

## Team contributions

<!-- ONLY REMAINING GAP: replace the two names. The work split is accurate. -->

| Member | Owned |
|---|---|
| *<name 1>* | Harness and safety (`harness/`, `tests/`), the agent loop (`agent/`), the causal feature stage, the experiment probes in `scripts/`, `docs/RESULTS.md` and `docs/OPEN_QUESTIONS.md`. |
| *<name 2>* | Independent agent runs and ablations (`runs/trial*`, `docs/RESULTS_teammate.md`), the ensembling findings on member count, batch size and heterogeneous mixing, and the writeups. |

**Tooling disclosure.** The harness, tests and documentation were written with Claude
Code. The *experiments* — the loss functions, features and model changes that produced
the scored result — were designed and coded by the agent itself, and its reasoning is
quoted verbatim in `docs/RESULTS.md`. The distinction is the substance of this
submission rather than a caveat on it: this is a track about building an autonomous
research agent, so what matters is which decisions the agent made unaided. Every one
of those is in `runs/*/log.jsonl` with its hypothesis, its diff and its outcome.

---

## References

1. Chan et al., "MLE-bench: Evaluating Machine Learning Agents on Machine Learning
   Engineering," OpenAI, 2024. arXiv:2410.07095
2. Jiang et al., "AIDE: AI-Driven Exploration in the Space of Code," 2025.
   arXiv:2502.13138
3. Yamada et al., "The AI Scientist-v2," 2025. arXiv:2504.08066
4. Zhao et al., "Counteracting Duration Bias in Video Recommendation via Counterfactual
   Watch Time," KDD 2024.
5. Gao et al., "KuaiRand: An Unbiased Sequential Recommendation Dataset with Randomly
   Exposed Videos," CIKM 2022.
