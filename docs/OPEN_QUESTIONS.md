# Open questions

Version 2. Q1 and Q6 are now resolved from the starter kit source. Six remain.

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

- **Status:** OPEN
- **Default:** the stricter of the two readings
- **Config:** `convergence.epsilon`, `convergence.n_consecutive`
- **Answer:**

---

### Q4 — Do failed or abandoned iterations count?

If a candidate errors out and is abandoned, does it consume one of the 50, and does it
count as a non-improving iteration for the three-strike window?

- **Status:** OPEN
- **Default:** counts toward the 50 cap; does NOT count as a non-improving iteration
- **If it flips:** repair policy changes materially. A failed iteration that burns a
  strike makes aggressive code generation much more expensive.
- **Answer:**

---

### Q5 — Does a crash-and-restart affect iteration count or the convergence window?

Per the webinar, restarts and operational recovery are not manual interventions.
Confirming the accounting treatment in writing.

- **Status:** OPEN (the webinar answer is second-hand and load-bearing)
- **Default:** No effect. State resumes exactly from the ledger. Counters never reset.
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
