# CLAUDE.md

Operating instructions for Claude Code on **TikTok TechJam 2026, Track 2: Autonomous Machine Learning Research Agent for Recommender Systems**.

Read this completely before writing any code. Where it conflicts with general ML best practice, this file wins, because it encodes competition rules and measured facts that override general practice.

**Version 2.** Rewritten after reading the starter kit source. The strategy changed materially: see §9.

---

## 1. Project goal

Build two things. Both are judged.

1. **An autonomous ML research agent.** An LLM-driven program that reads the problem, inspects the data, engineers features, trains and tunes a model, evaluates, reflects, and iterates, writing the code for each stage itself.
2. **A recommendation model** produced by that agent, scored once on a hidden test set.

The model's task: given a log of short-video impressions, rank the items **within each user's own logged impression list** by likelihood of `long_view`. Ranking stage only. No retrieval, no candidate generation, no catalogue-wide recall.

### Judging criteria

| Category | Weight | Measures |
|---|---|---|
| Technical Execution | 35% | Hidden-test primary delta over baseline, plus failure handling |
| Innovation & Problem Insight | 20% | What the agent targeted and why. Reasoning, not implementation |
| Impact & Relevance | 20% | Autonomy, by number of manual interventions |
| Feasibility & Practicality | 15% | Tokens and wall clock, three tiers, **only scored if you beat the baseline** |
| Presentation & Communication | 10% | Final event only, 11 September |

Autonomy and Innovation together are 40%. An agent executing a human-written experiment queue scores badly on both, however good the model is.

Objective: **beat the baseline reliably, then maximise score within a reasonable resource tier.** Do not add large complexity for small uncertain gains.

---

## 2. Requirements, conflicts, assumptions, open questions

### 2.1 Confirmed requirements

- Required benchmark: **KuaiRand-Pure**, 100% of the primary metric. 1k and 27k are optional bonus.
- Label: `long_view`. Metrics: GAUC and nDCG@5. Primary = mean of the two.
- Splits are fixed and date-based. Develop on train + validation only.
- Convergence: eps = 0.002, N = 3. Hard caps: 50 iterations, 6 hours.
- Scored submission is the **validation-best checkpoint** at convergence, evaluated once on hidden test.
- Any open-source library, paper, public solution, or pretrained weight is allowed.
- **One hard rule: no external training data.** KuaiRand only.
- Deliverables: Devpost description, public GitHub repo with README, per-iteration run logs, manual-intervention count, final submission, results table, resource usage.
- No video required. Without one, a detailed report is expected.

### 2.2 Resolved from the starter kit source

| Was open | Now |
|---|---|
| Q1: metric conflict (`NDCG@10 / Recall@50 / click` in the PDF Limits row) | **RESOLVED.** `data.py` line 5: `LABEL = 'long_view'`. `evaluate.py` computes GAUC and nDCG@5. The Limits row is stale text. |
| Q6: tie handling in `evaluate.py` | **RESOLVED.** `auc()` uses Mann-Whitney U with proper average-rank tie correction. nDCG sorts descending by score. Ties handled correctly, not arbitrarily. Emitting ties is safe but still not useful. |
| Conflict B: AliCCP required (webinar slide) | **RESOLVED.** Stale slide. The kit is KuaiRand-Pure only. |
| Data path | **RESOLVED.** `baseline.py`, `submit.py`, `ablation_features.py` all accept `--data_dir`; `data.load(path)` takes a path argument. No filesystem junction needed. |

### 2.3 Still open — post to organisers, do not silently pick

| # | Question | Default in force |
|---|---|---|
| Q2 | May the winning config be refitted on train + validation before predicting test? | **No.** Flag exists, disabled |
| Q3 | Exact convergence comparison: `best(last 3) - best(before) <= eps`, or `max per-iteration gain over last 3 <= eps`? `baseline_scores.json` gives eps and N but not the comparison rule | Stricter reading |
| Q4 | Do failed or abandoned iterations count toward the 50 cap and the 3-strike window? | Count toward 50; do NOT count as a non-improving iteration |
| Q5 | Does a crash-and-restart affect iteration count or the convergence window? | No. State resumes from the ledger |
| Q7 | Are the supplementary KuaiRand files (captions, categories) in scope? | Do not use |
| Q8 | May `log_random_4_22_to_5_08_pure.csv` be used, restricted to dates < 20220429? | Not for training. The kit README suggests it as an unbiased **validation** set, a different and lower-risk use |

### 2.4 Working assumptions

- The starter kit is code only. KuaiRand-Pure (194 MB) is downloaded separately.
- The 50-iteration and 6-hour limits are self-enforced and self-reported.
- One "iteration" = one agent-defined experiment cycle, not one gradient step.
- **A crash-and-restart or manual failure recovery is not a manual intervention.** Per the organiser webinar, an intervention is a human changing the agent's instructions, objective, or search space. Restarting a crashed process is operational and is reported separately.

---

## 3. Dataset, splits, metrics, convergence, submission

### 3.1 Data

KuaiRand-Pure: 27,285 users, ~7,551 videos, 1,436,609 standard-log interactions from the Kuaishou short-video app, April–May 2022.

Six CSVs. What the kit's `data.py` currently reads:

| File | Used by the kit |
|---|---|
| `log_standard_4_08_to_4_21_pure.csv` | Yes. Train period |
| `log_standard_4_22_to_5_08_pure.csv` | Yes. Validation and test periods |
| `video_features_basic_pure.csv` | **Only `author_id`.** Tags, video type, upload type, music, resolution ignored |
| `user_features_pure.csv` | **Never opened** |
| `video_features_statistic_pure.csv` | Never opened. **Keep it that way, see §7.3** |
| `log_random_4_22_to_5_08_pure.csv` | Never opened |

Columns read per row: `date`, `user_id`, `video_id`, `tab`, `duration_ms`, `long_view`. Everything else in the log is untouched.

**Label:** `long_view = 1` when `play_time_ms >= duration_ms` for videos <= 18,000 ms, or `play_time_ms >= 18,000` for longer ones. Deterministic in watch time and duration. `play_time_ms` is never loaded by the kit; keep it that way.

### 3.2 Splits

| Split | Dates | Rows |
|---|---|---|
| train | 20220408–20220421 | 1,141,112 |
| validation | 20220422–20220428 | 124,909 |
| hidden test | 20220429–20220508 | 170,588 |

Density shift: train averages ~81,500 rows/day, evaluation periods ~17,000/day.

Train has roughly 42 rows per user across 14 days. The evaluation splits have roughly 6 per user. **This matters for list construction in any pairwise or listwise loss.**

### 3.3 Reference numbers (from `baseline_scores.json`, authoritative)

| Model | valid GAUC | valid nDCG@5 | valid primary | test primary |
|---|---|---|---|---|
| random | 0.4993 | 0.4675 | 0.4834 | 0.4753 |
| item popularity | 0.6387 | 0.5227 | 0.5807 | 0.5715 |
| **FM (official baseline)** | **0.6674** | **0.5357** | **0.6016** | **0.5946** |
| oracle ceiling | 1.0000 | 0.6968 | 0.8484 | 0.8645 |

FM 5-seed std on test: 0.0008 on all three figures.

**Use the validation column for local work.** The PDF quotes the test column, which is why earlier drafts had the wrong local targets.

Test composition: 23,875 users. 27.1% all-negative (nDCG always 0), 9.2% all-positive (nDCG always 1), 63.7% discriminative. GAUC is computed over that 63.7% only.

Scoring collapses to `score_dataset = primary_agent - 0.5946`. Headroom above baseline is 0.2699. The baseline already captures 30.7% of the attainable range; judge progress against 0.8645, not 1.0.

### 3.4 Convergence — implement literally

```
converged = three consecutive iterations each improving validation primary by <= 0.002
         OR iteration count == 50
         OR wall clock == 6 hours
```

Three non-negotiable rules:

1. **Never stop voluntarily before convergence fires.** A non-improving iteration cannot lower the validation-best checkpoint, so a failed swing costs nothing. On strike 3, take the highest-expected-gain untried structural change.
2. **Never continue after convergence fires.**
3. **Restarts resume state.** Reload iteration number, strike count, tried-set and best checkpoint from the ledger. Resetting counters on restart would be gaming the rule.

### 3.5 Submission

CSV, header `row_id,user_id,video_id,score`.

`row_id` is the positional index into `data.load(path)[split]`. That order is deterministic: `log_standard_4_08_to_4_21` read first, then `log_standard_4_22_to_5_08`, filtered by date preserving file order. **Never sort or reindex the evaluation list.**

`(user_id, video_id)` is not a key: 3.06% of test rows are repeats, up to 12 times.

`submit.py` gives three modes, all of which the harness should use:

```
python3 submit.py --make  --split test  --data_dir <path> submission.csv
python3 submit.py --check --split test  --data_dir <path> submission.csv
python3 submit.py --score --split valid --data_dir <path> submission.csv
```

`--score` works on validation and gives an independent check of your own scoring path. Use it in the contract test.

---

## 4. Architecture

A **hybrid**: a fixed hand-written harness owns everything that must not break; an LLM agent writes real code for the parts that are fair game.

```
  +-----------------------------+        results        +--------------------------+
  |      FIXED HARNESS          | -------------------->  |       AGENT LOOP         |
  |  (never LLM-written)        |                        |   (LLM writes code)      |
  |                             |                        |                          |
  |  data loader + test guard   |                        |  analyse(spec) - EDA     |
  |  causal feature helpers     |                        |  diagnose last result    |
  |  model runners              |  <-------------------- |  propose experiment      |
  |  evaluate.py wrapper        |        patch           |  write loss/model/feature|
  |  submit.py wrapper          |                        |  read outcome, reflect   |
  |  convergence tracker        |                        |                          |
  |  ledger + run logger        |                        +--------------------------+
  +-------------+---------------+
                |
                v
    validation-best checkpoint  -->  submission.csv  -->  scored once on hidden test
```

### 4.1 What the LLM MAY modify

- **Loss and objective functions** (now the primary target, see §9)
- Model builder functions in `harness/models/gen/`
- Feature transform functions in `harness/features/gen/`
- Training-loop details: batching, list construction, sampling, weighting
- Hyperparameter values
- Ensemble blending logic

### 4.2 What the LLM MAY NEVER modify

- `harness/data.py`, `harness/guards.py`, `harness/evaluate.py`, `harness/submit.py`
- `harness/convergence.py`, `harness/ledger.py`, `harness/logger.py`, `harness/sandbox.py`
- Anything in `starter/` — the organiser's code is read-only
- Anything under `tests/`

The patch validator rejects any diff touching a protected path. State this boundary in the README.

---

## 5. CRITICAL: the test labels are in memory and the kit prints test scores

`data.load()` returns `out['test']` as 170,588 tuples **with their true labels at index 6**. The hidden test set is not hidden from your process. The organisers trust you not to look.

Worse, `baseline.py:run_fm()` ends with:

```python
return {'valid': evaluate(uva, yva, m.predict(Xva)),
        'test':  evaluate(ute, yte, m.predict(Xte))}
```

and `__main__` prints both. **If the agent runs `baseline.py` and reads stdout, it sees the test score.** That contaminates the run and is indefensible if a judge reads your logs.

Three mandatory controls:

1. **`harness/data.py` strips the label from the test split** before anything downstream sees it. `out['test']` becomes 6-tuples, not 7-tuples. Feature code physically cannot read `x[6]` for test rows.
2. **Any stdout from `starter/` scripts is filtered** before reaching the agent's context. Strip lines containing test metrics. Log raw output to a human-only file, never into `log.jsonl` or an LLM prompt.
3. **The harness never calls `run_fm` directly for scoring.** It reimplements the training loop against label-stripped splits, or calls `run_fm` with test disabled.

This is your strongest integrity story: the loader hands you the answers and you refuse to look. Say exactly that in the README, and back it with `test_no_test_labels.py`.

---

## 6. The agent loop

Each iteration:

1. **Inspect.** Call `analyse(spec)` zero or more times. The agent chooses what to ask.
2. **Diagnose.** A rule-based block produces *facts*: per-metric deltas, train/validation gap, segment breakdowns, list-size distribution, runtime. The LLM hypothesises on top. **The LLM never invents numbers.**
3. **Propose.** One experiment as typed JSON. Reject proposals whose content hash is in the tried-set.
4. **Materialise.** Write actual Python into `features/gen/` or `models/gen/`.
5. **Validate the patch.** Protected-path check, import allowlist, determinism on a 10k-row fixture.
6. **Execute.** Sandboxed subprocess, fixed seeds, hard timeout.
7. **Evaluate.** Official `evaluate.py` on validation only.
8. **Decide.** Keep if validation primary improves by more than eps. Otherwise roll back. **Validation is the sole authority.**
9. **Log.** Hypothesis, diff, metrics, tokens, wall clock, errors, decision, reason.
10. **Update the clock.** Increment iteration, update strike count, checkpoint if improved.

### 6.1 Autonomous EDA — the `analyse` tool

```python
def analyse(spec: AnalysisSpec) -> pd.DataFrame:
    """
    spec.kind in {
      "rate_by_bucket",       # label rate grouped by a binned column
      "distribution",         # histogram of a column
      "list_size_profile",    # impressions per user, per split
      "segment_metrics",      # GAUC/nDCG broken down by a segment
      "temporal_drift",       # a statistic per date
      "model_disagreement",   # rows where two saved models rank differently
      "score_tie_rate",       # fraction of within-user tied scores
      "cold_key_rate",        # fraction of rows with unseen user/video keys
    }
    Operates on train and validation only. Never test.
    """
```

This is Figure 1 stage 2. It converts "inspect data" from a human dashboard into real inspection, and it is what Innovation is scored on.

### 6.2 Method corpus

`knowledge/methods.md` is reference material the agent retrieves from when proposing. **It is not a queue of configurations to execute in order.** The problem statement sanctions drawing on published methods; what it forbids is a scripted search.

It now includes the organisers' published dead ends, so the agent does not waste iterations on things already measured to fail.

### 6.3 Failure recovery

Restarts are not interventions, so failure is cheap. Keep recovery simple:

| Failure | Response |
|---|---|
| Code error in generated patch | Traceback back to the LLM, **one** repair attempt, then abandon and mark tried |
| Timeout | Retry once at 30% subsample, then abandon |
| Memory error | Retry once with float32 and half the features, then abandon |
| Evaluator rejects output | Hard failure. Roll back, re-emit last-good submission. Never patch around the evaluator |
| Two consecutive abandons | Force a proposal targeting a different pipeline stage |
| LLM API unavailable | Deterministic mode, logged as a recovery event |

No elaborate escalation ladder. No chaos-testing framework. Two failure-path tests is enough evidence.

### 6.4 Autonomy requirements

Three things the agent must do that a human must not do for it:

1. **Reproduce the official baseline itself.** It reads `baseline.py`, runs it (with test output filtered per §5), and verifies against `baseline_scores.json`.
2. **Choose its own experiments** from the corpus plus its own analysis, not from a config queue.
3. **Write the actual code** for loss, model and feature changes.

Any hand-built strong model exists as a **private benchmark only**. It must not seed the agent's starting state and must not appear on the scored path.

---

## 7. Leakage safety

### 7.1 Test label stripping

See §5. This is the primary control and the one that matters most.

### 7.2 Column deny-list

The kit's loader reads only `date`, `user_id`, `video_id`, `tab`, `duration_ms`, `long_view`, so there is currently no leak. The deny-list governs columns the agent might add.

Never permitted as features: `play_time_ms`, `is_click`, `is_like`, `is_follow`, `is_comment`, `is_forward`, `is_hate`, `is_profile_enter`, `profile_stay_time`, `comment_stay_time`. All are same-impression outcomes.

Permitted, known before the impression: `duration_ms`, `tab`, `hourmin`, `date`, `time_ms`, `is_rand`, and everything in `user_features_pure.csv` and `video_features_basic_pure.csv`.

These outcome columns are legitimate as **auxiliary training targets** in a multi-task setup. They are never inputs.

### 7.3 `video_features_statistic_pure.csv` — excluded

Per-video averages computed over a month that spans the test window, including `long_time_play_cnt`, `valid_play_cnt` and `play_progress`, which are near-direct label proxies. Not used.

### 7.4 Causal target encoding

If the agent builds any historical statistic (video's past long-view rate, user-tag affinity), it is a target encoding and leaks unless the window is causal:

| Rows being featurised | Statistics from |
|---|---|
| Training row on date `d` | dates strictly `< d` |
| Validation row | train dates only (`< 20220422`) |
| Test row | train dates only by default; train + validation if Q2 is answered yes |

The feature API enforces this. A feature reading outside its window raises.

**Note the demotion.** In v1 this was the headline innovation. The organisers have since published that static feature additions produce no gain on this benchmark (see §9), so this is now correctness hygiene rather than a differentiator. Build it, test it, mention it, but do not lead with it.

### 7.5 Leak canary

Quarantine and flag any configuration scoring above 0.80 primary on validation. The validation oracle ceiling is 0.8484, so anything near it is a leak. Test the canary by injecting a `play_time_ms` feature.

---

## 8. Validation strategy

**The official validation split is the sole keep/reject authority, and the scored submission is the literal validation-best checkpoint.** Do not override the validation winner with an internal split.

**Diagnostic folds** inform the agent's next proposal, never selection:

- fit 20220408–20220414 -> score 20220415–20220421
- fit 20220408–20220417 -> score 20220418–20220421

**Measure your own noise floor early.** Run the first non-FM config across 3 seeds. The FM's 0.0008 is the FM's variance; a different objective can have very different variance.

**Train + validation refit** is behind `--refit-on-train-val`, default off, pending Q2.

**Why validation can lie:** selection noise over many decisions on 124,909 rows; temporal drift to a later, longer window; the density shift; a different user mix (validation oracle 0.8484 vs test 0.8645, so the splits differ in composition); silent leakage.

---

## 9. Model and objective strategy — READ BEFORE PLANNING ANY EXPERIMENT

The starter kit README publishes the organisers' own ablations. This changes the priority order substantially from v1.

### 9.1 Measured dead ends — do not repeat

| Tried by the organisers | Result |
|---|---|
| Adding all 13 CWM feature fields (`music_id`, `video_type`, `upload_type`, plus 6 user-side coarse buckets) | primary **0.5940** vs **0.5950** for 5 fields. No gain, marginally worse |
| Embedding dimension k = 8 / 16 / 32 | 0.5895 / 0.5902 / 0.5887. Essentially flat |

Their explanation: the `user_id x video_id` cross already absorbs most of the learnable signal, and 1.14M rows cannot support more capacity. **The bottleneck is neither features nor model size.**

They also confirm, with measurement, that pure user-side first-order terms contribute exactly zero, because anything constant within a user cannot change within-user ordering.

`ablation_features.py` reproduces these numbers. Run it once so the team believes them.

### 9.2 Priority order (the organisers' own, and it is sound)

**1. Change the loss function. Top priority, and the first real experiment.**

The current FM optimises pointwise logloss while both metrics are ranking metrics. That mismatch is the clearest structural weakness in the baseline. Two directions:

- **Pairwise (BPR).** Within a user's impressions, sample positive-negative pairs and optimise `-log(sigmoid(z_pos - z_neg))`.
- **Listwise softmax.** Softmax over a user's impression list, cross-entropy against the labels.

Implementation note: `FM.step()` already computes gradients cleanly, so a pairwise variant is roughly 40 lines inside the existing class. **The open design question is list construction.** Now measured, via `analyse(kind="list_size_profile")`:

| Grouping | Split | Lists | Mean | Median | p90 |
|---|---|---|---|---|---|
| `user_id` | train | 26,210 | **43.5** | 31 | 97 |
| `(user_id, date)` | train | 197,796 | **5.77** | 3 | 14 |
| `user_id` | valid | 22,377 | **5.58** | 4 | 12 |

**This corrects an earlier claim in this file and in `knowledge/methods.md`,** both of which said `(user_id, date)` gives "~3" and concluded it did not match. Three is the *median*; the mean is 5.77, and the evaluation mean is 5.58. So `(user_id, date)` on train matches the evaluation list length almost exactly, while `user_id` alone is 7.8x too long.

That does not settle the experiment — mean list length is one property and the distributions still differ — but the corpus was steering the agent away from the option the measurement favours. The estimate has been removed from `knowledge/methods.md` rather than replaced, so the agent measures it rather than being handed a conclusion.

**2. User history sequences.** Completely unused. Each user has hundreds to thousands of training interactions. DIN/SIM-style interest modelling is untouched. Highest ceiling, highest build cost.

**3. Multi-task.** `is_click`, `is_like`, `is_follow`, `is_comment`, `is_forward`, `play_time_ms` as auxiliary heads over a shared representation, with `long_view` as the main task.

**4. Watch-time modelling.** The CWM censored-regression idea: a completed play means true watch time was truncated by video length, so a one-sided loss beats squared error. Reference only; CWM pins `torch==1.6.0` and rebuilds its own label.

**5. Different architectures** (DeepFM, DCN, xDeepFM). Explicitly deprioritised below 1–4 because capacity is measured not to be the bottleneck.

**6. Time features and drift.** `hourmin`, `date`, train-to-test drift.

**7. Unbiased validation.** `log_random_4_22_to_5_08_pure.csv` as an extra unbiased *validation* set to check for overfitting to biased traffic. Validation, not training. Pending Q8.

### 9.3 Reversal: LightGBM is probably a trap here

v1 of this file named LightGBM the workhorse. That was wrong for this dataset.

The FM works because it learns embeddings for `user_id` and `video_id` and crosses them. A gradient-boosted tree cannot do that: 27K users x 7.5K videos is not splittable. To use a GBDT you would replace IDs with target encodings, which is precisely the static-feature approach the organisers measured as producing no gain.

**Keep the FM as the base architecture and change its objective.** LightGBM becomes a comparison experiment for the ablation table, not the primary path. If the agent proposes it, let it, and let the result speak.

### 9.4 Build order

| Order | What | Notes |
|---|---|---|
| 1 | Reproduce FM baseline | Blocking gate. Match `baseline_scores.json` validation column |
| 2 | Run `ablation_features.py` | Confirm the dead ends yourself, once |
| 3 | **Pairwise BPR loss on the existing FM** | The first real experiment |
| 4 | **Listwise softmax loss** | Second |
| 5 | List-construction variants | `user_id` vs `(user_id, date)` grouping |
| 6 | Multi-task auxiliary heads | Third |
| 7 | Sequence features | Stretch |
| 8 | Ensembling | Only after the above works |

Out of scope entirely: retrieval, candidate generation, negative sampling from a catalogue, diversity re-ranking, full sequential architectures, distributed training, GPU dependencies.

### 9.5 Post-processing arithmetic, so nobody wastes an iteration

- **Per-user score normalisation on a single model is a mathematical no-op.** Monotone within the list, so it cannot change GAUC or nDCG@5. The organisers confirm they measured this.
- **It does matter for blending.** Rank-normalise within each user before averaging models.
- **Duration-conditioned recalibration is a real lever**, because duration varies across a user's items, so the adjustment is non-monotone within the list.
- Tune ensemble weights once. Repeated tuning against validation overfits it.

---

## 10. Logging and resource reporting

### 10.1 Per-iteration JSONL record

```json
{
  "run_id": "...", "iteration": 7, "git_sha": "...", "timestamp": "...",
  "analyses_requested": [{"kind": "list_size_profile"}],
  "hypothesis": "Pointwise logloss is misaligned with within-user ranking metrics. A pairwise BPR objective over each user's impressions should improve GAUC more than nDCG, since GAUC is itself a pairwise measure.",
  "target_stage": "objective",
  "patch_kind": "new_loss_function",
  "diff": "...",
  "metrics": {"val_gauc": 0.681, "val_ndcg5": 0.549, "val_primary": 0.615,
              "diagnostic_folds": {"fold_1": 0.612, "fold_2": 0.609},
              "train_primary": 0.671, "mean_list_size": 6.2},
  "decision": "keep", "reason": "val_primary +0.0134 > eps",
  "strikes_after": 3,
  "errors": [], "recovery_events": [],
  "tokens": {"input": 8412, "output": 1903, "model": "..."},
  "wall_clock_s": 214
}
```

Also emit human-readable `log.md`. That is what judges read.

**No test metrics anywhere in any log the agent can see.** See §5.

### 10.2 Final resource report

Total tokens (input + output), agent wall clock to convergence, iterations used out of 50, GPU-hours (expect 0), and **manual interventions** with the definition stated. Operational restarts listed separately and clearly labelled as non-interventions.

---

## 11. Repository structure and frozen interfaces

```
techjam-track2-agent-2026/
  CLAUDE.md  README.md  requirements.txt  .env.example  .gitignore
  configs/base.yaml
  knowledge/methods.md
  docs/OPEN_QUESTIONS.md  docs/STARTER_KIT_NOTES.md
  starter/                        READ-ONLY. Organiser code
    baseline.py  data.py  evaluate.py  submit.py
    baseline_scores.json  ablation_features.py  README.md
  harness/                        PROTECTED
    data.py            wraps starter data.load, STRIPS TEST LABELS
    guards.py          deny-list, canary, stdout filter
    features/registry.py  base.py  gen/
    models/runners.py  gen/
    analyse.py  evaluate.py  submit.py  sandbox.py
    ledger.py  convergence.py  logger.py
  agent/
    loop.py  diagnose.py  propose.py  llm.py
  scripts/  tests/  workspace/  runs/
```

### 11.1 Frozen interfaces — agree before writing anything

```python
# 1. The only thing crossing the harness/agent boundary
def run_experiment(patch_path: str, seed: int) -> dict:
    """
    Returns:
      val_gauc, val_ndcg5, val_primary : float
      diagnostics : dict
      error : str | None
      seconds : float
    NEVER returns any test metric.
    """

# 2. Loss function -- the primary target
def loss_and_grad(z: np.ndarray, y: np.ndarray, groups: np.ndarray) -> tuple:
    """z: model logits. y: labels. groups: user group id per row.
    Returns (loss: float, dL_dz: np.ndarray)."""

# 3. Feature function
@register_feature(name="...", needs_dates_before=True)
def build(frame, stats):
    """Same index as frame, only new columns. Pure and deterministic."""

# 4. Experiment spec (LLM output, JSON-schema validated)
{"hypothesis": str,
 "target_stage": "objective" | "model" | "features" | "sampling" | "ensemble",
 "patch_kind": str, "expected_gain": float,
 "expected_cost_minutes": float, "patch": str}
```

Interface 2 is new in v2 and is the most important, because the objective is now the primary target. Design it so a pairwise or listwise loss drops in without touching the training loop.

---

## 12. Tests and milestones

### 12.1 Tests, in build order

| Test | Asserts |
|---|---|
| `test_contract_baseline.py` | Split row counts 1,141,112 / 124,909 / 170,588; `--model fm` gives valid primary 0.6016 +/- 0.001; `--model random` gives 0.4834 +/- 0.001; `--model pop` gives 0.5807 +/- 0.001 |
| `test_no_test_labels.py` | `harness.data.load()['test']` tuples carry no label; no stdout reaching the agent contains a test metric |
| `test_submission.py` | `submit.py --check` passes on valid, fails on: wrong header, row-count mismatch, `row_id` gap, misalignment, NaN score |
| `test_causal_encoding.py` | A statistic for a row on date `d` is identical whether or not rows dated `>= d` exist in the input |
| `test_guards.py` | Poisoned frame raises; canary fires on an injected `play_time_ms` feature |
| `test_convergence.py` | Hand-built score sequences give correct strike counts and stop points, including resume-after-restart |
| `test_patch_validation.py` | Patch touching a protected path rejected; banned import rejected |
| `test_loss_interface.py` | A pairwise loss and the existing pointwise loss both satisfy interface 2 and give finite gradients |
| `test_determinism.py` | Same config and seed, identical scores twice |

Regression rule: after any harness change, re-run the contract test and the current best config. Movement over 0.001 means something broke.

### 12.2 Milestones

**M1 Foundation.** Contract test green. Test-label stripping working and tested. `ablation_features.py` run once. Repo pushed, four interfaces frozen.

**M2 Loop works.** Model runner (FM reimplemented against label-stripped splits). `run_experiment` boundary. `analyse` tool. Loss-function interface. Ledger, logger, sandbox, patch validator. Agent loop end to end with real code generation, one repair attempt, rollback, checkpoint save/restore.
*Accept:* **an engineering result only.** Ten iterations complete unattended, a valid submission is produced and passes `--check`, the run survives kill-and-restart, and `log.md` is readable by a human. **Score is not a gate at M2.**

> **Revised.** This gate previously read "and beats 0.6016 on validation", which bolted a research result onto an engineering one. See D11 in `docs/OPEN_QUESTIONS.md`. The agent ships at M2 with only a pointwise loss, and §9.1 records that features and capacity are measured dead ends, so its realistic move set is hyperparameters or writing a pairwise objective unprompted. Gating the milestone on an outcome this file's own strategy calls unlikely was a spec bug. A working loop must not be declared a failure because the science did not land on schedule.

**M3 Agent gets good.** Agent reproduces the baseline itself. Method corpus in use. Pairwise and listwise losses available for it to discover. Noise floor measured. Token accounting complete.
*Accept:* **the research result now lives here.** A run whose log shows at least three iterations targeting different pipeline stages, with reasoning that changed after evidence, **and which beats validation primary 0.6015**.

**M4 Scored run.** Freeze code. One clean, uncached run, started once, recorded. Submission written and `--check` validated. Results and resource tables complete.
*Accept:* zero manual interventions by the stated definition. **Start with at least 8 hours of window remaining.**

**M5 Packaging.** README, Devpost, formatted run logs, open questions and resolutions.
*Accept:* a clean clone reproduces the baseline with no hidden setup.

### 12.3 Fallback scope — drop in this order

1. Video (write the detailed report instead)
2. Sequence modelling
3. Multi-task
4. Bonus datasets
5. Ensembling
6. **Last resort:** deterministic mode. Agent layer replaced by a fixed sequence over the corpus, identical logging and guards.

Never drop: the contract test, test-label stripping, literal convergence and checkpoint rules, submission validation, the run log.

---

## 13. Deterministic mode

`LLM_PROVIDER=none` must run the whole pipeline with no LLM call and produce a valid submission. Test by end of day two. Insurance against an API outage during the scored run, and a judging asset.

---

## 14. Standing rules for Claude Code

- Read `docs/OPEN_QUESTIONS.md` before any decision it affects. When an answer arrives, update the file and the config in the same commit.
- Never modify anything in `starter/`. Wrap it.
- Never reimplement `evaluate.py`. It is the sole definition of the score.
- **Never let a test metric reach the agent's context or any machine-readable log.**
- Never write a number into a log or report that was not computed by code.
- Every commit changing harness behaviour runs the contract test.
- No secrets in source, logs, or git history.
- When something is ambiguous and no ruling exists, take the conservative reading and record it in `OPEN_QUESTIONS.md`.
