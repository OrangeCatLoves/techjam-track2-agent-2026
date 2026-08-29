# Autonomous ML Research Agent — TikTok TechJam 2026, Track 2

An autonomous machine-learning research agent for recommender systems, evaluated on
the KuaiRand-Pure benchmark.

> Fill in every `TODO` before submission. The README is a graded deliverable under
> Demo and Reproducibility.

---

## Project overview

TODO — two paragraphs. What the agent does, and what makes this submission's approach
distinct. Lead with leakage-safe causal feature synthesis: the agent writes feature
code freely, and the harness makes temporal causality structural rather than a rule
the agent has to remember.

---

## Results

| | GAUC | nDCG@5 | Primary | Delta vs baseline |
|---|---|---|---|---|
| Official baseline (hidden test) | 0.6610 | 0.5282 | 0.5946 | — |
| Ours (validation) | TODO | TODO | TODO | TODO |
| Ours (hidden test) | TODO | TODO | TODO | TODO |

Scoring formula: `score_dataset = mean(delta(GAUC), delta(nDCG@5))`, which for this
benchmark reduces to `primary_agent − 0.5946`.

### Resource usage

| | |
|---|---|
| Iterations used (of 50) | TODO |
| Agent wall clock to convergence | TODO |
| LLM tokens, input + output | TODO |
| GPU-hours | 0 |
| **Manual interventions** | TODO |
| Operational restarts (not interventions) | TODO |

Manual intervention is defined per the organiser webinar as a human changing the
agent's instructions, objective, or search space. Restarting a crashed process,
clearing a lock, or freeing disk is operational recovery and is counted separately.

---

## Setup

### Requirements

- Python 3.10 or newer
- No GPU required
- Approximately 2 GB free disk for the dataset and materialised frames

### 1. Get the data

Download KuaiRand-Pure (194 MB) from
`https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz`
(MD5 `0820331067a3784d9691136f772b35a7`) and extract it **outside this repository**.

You should end up with six CSVs in `<somewhere>/KuaiRand-Pure/data/`.

### 2. Install

```
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure

Copy `.env.example` to `.env` and set `KUAIRAND_DATA_DIR` plus your LLM key.
Point `paths.raw_data_dir` in `configs/base.yaml` at the same directory.

### 4. Verify

```
python scripts/verify_setup.py            # everything, about 10 minutes
python scripts/verify_setup.py --fast     # skip the FM reproduction, about 20 seconds
```

This checks the environment, the data, the split row counts, the test-label strip, the
column deny-list, the stdout filter against a live organiser run, the leak canary, the
submission round trip, the convergence rule, the FM baseline reproduction, and the
pytest suite, then prints a summary. If it fails, nothing downstream is trustworthy.

The regression gate on its own:

```
pytest tests/test_contract_baseline.py    # row counts and the baseline ladder
pytest tests/ -m "not slow"               # everything that runs in seconds
```

---

## Reproducing our results

```
python scripts/materialise.py          # build sanitised train/val/test frames
python scripts/run_agent.py            # the full autonomous run
python scripts/run_deterministic.py    # same pipeline, zero LLM calls
```

TODO — confirm these commands match the final scripts and state expected wall clock
for each.

The scored submission is at `runs/<run_id>/submission.csv`. Run logs are at
`runs/<run_id>/log.md` (human-readable) and `log.jsonl` (machine-readable).

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

TODO — insert the architecture diagram.

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

TODO — be specific and honest. Candidates:

- Which open questions were never answered, and the conservative default taken.
- Where the agent's proposals were weakest.
- Validation-to-test transfer risk given 124,909 validation rows.
- What was cut for time.

---

## Repository layout

TODO — short table of the main directories.

---

## Team contributions

| Member | Owned |
|---|---|
| TODO | TODO |

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
