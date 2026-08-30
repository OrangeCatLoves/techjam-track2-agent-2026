# Questions for the organisers

**Status: 4 of 8 decided internally, 4 still want an organiser ruling.**

Deadline confirmed by the team: **submit by 1 September, 02:00.**

A decision made here is a decision about *what we will do*, not a discovery of what
the organisers meant. Where the two could differ, that is said plainly below, because
an assumption recorded as an answer is how a rule gets broken quietly.

---

## Decided — no longer blocking

### 1. Convergence semantics — reading (b), the combined one

Two readings of "three consecutive iterations improving by no more than 0.002":

- **(a) strict:** *each* of the last three improved the running best by ≤ 0.002
- **(b) combined:** best(last three) − best(before those three) ≤ 0.002

**Team decision: (b).** `convergence.comparison: block`.

**What this changes.** (b) fires later than (a), so runs go longer. On three
consecutive gains of 0.0015: (a) stops, (b) continues, because together they are
0.0045. Given the control run converged after only four iterations, this materially
increases how many experiments the agent gets.

**The risk, stated honestly.** This is a judgement about what the organisers meant,
not a ruling from them. If they meant (a), we will have run longer than the rule
allowed. It remains worth confirming, and it is a one-word config change either way —
both readings are implemented and tested.

*(Note: on the control run's actual numbers both readings agreed, so this changes
nothing already measured.)*

### 4. Restarts — resume everything

On a restart we resume iteration count, strike count, best checkpoint and tried-set
from disk. Counters are never reset. Only *active* agent time is charged against the
six hours, so an overnight gap between a crash and its discovery is not charged.

Implemented and tested. Consistent with the webinar's position that restarting a
crashed process is operational recovery, not a manual intervention.

### 6. The randomised-exposure log — not used

`log_random_4_22_to_5_08_pure.csv` is not used at all, for training or validation.
It overlaps the hidden test window, and the safe reading costs us nothing we can
measure.

### 8. Deadline — 1 September, 02:00

Set by the team. Still worth confirming with the organisers in writing, including
the timezone.

---

## Still open — an organiser ruling would change what we do

### 2. Do failed iterations count?

*In plain terms:* we get 50 experiments and a three-strikes rule. Sometimes an
experiment crashes — buggy generated code, or a timeout. Does a crash consume one of
the 50, and does it count as a strike?

**Our default:** it consumes one of the 50, but is **not** a strike. It produced no
validation score, so there is nothing to say it failed to improve.

**Why it matters:** if crashes burned strikes, three buggy experiments in a row would
end the run, and the agent would be punished for attempting ambitious code. Our
reading lets it fail and continue. If the organisers disagree, the repair policy has
to become much more conservative.

### 3. May the winning configuration be refitted on train + validation?

*In plain terms:* we train on chunk 1, pick the winner using chunk 2, and are scored
on chunk 3. Standard practice would retrain the winner on chunks 1 + 2 before
scoring — more data, better model. But the rule says submit "the validation-best
checkpoint", and a refitted model was never itself scored on validation.

**Our default: no.** `selection.refit_on_train_val: false`. Implemented, disabled.

**Cost of the safe reading:** roughly 10% of usable training data left unused.

### 5. Are the supplementary KuaiRand files in scope?

`kuairand_video_captions.csv` and `kuairand_video_categories.csv` are published by
the same authors at a separate record. The rule says training must rely only on "the
KuaiRand datasets listed below", and we cannot resolve what that list contains.

**Our default: not used.** Low stakes either way — `video_features_basic_pure.csv`
already carries tags, so the marginal value is small. Safe under both readings.

### 7. Are the bonus benchmarks accompanied by splits and baselines?

KuaiRand-1k and 27k are named as bonus benchmarks, but the kit ships no splits,
baseline or scores for them.

**Our default: not attempted.** 27k is 322M rows and is not feasible on CPU in the
window. Out of scope regardless of the answer.

---

## Paste this if you get the chance to ask

> Hi — four clarifications on Track 2 (KuaiRand-Pure). We have a conservative default
> for each, so a non-answer is safe; we would rather be right than safe.
>
> **1. Failed iterations.** If a candidate errors out and is abandoned, does it
> consume one of the 50, and does it count as a non-improving iteration for the
> three-strike window? We currently count it toward the 50 but not as a strike, since
> it produced no validation score.
>
> **2. Refitting.** The rule says the scored submission is the "validation-best
> checkpoint". May the winning configuration be refitted on train + validation before
> predicting test? A refitted model was never itself scored on validation, so a
> literal reading forbids it. We have it implemented and disabled.
>
> **3. Supplementary files.** Are `kuairand_video_captions.csv` and
> `kuairand_video_categories.csv` (same authors, separate record) in scope, or does
> "the KuaiRand datasets listed below" exclude them? We are not using them.
>
> **4. Convergence semantics.** `baseline_scores.json` gives `epsilon = 0.002, N = 3`.
> Did you intend (a) *each* of the last three iterations improved the running best by
> ≤ 0.002, or (b) best(last three) − best(before those three) ≤ 0.002? They stop at
> different points on, for example, three consecutive gains of 0.0015.
>
> Also, could you confirm the submission deadline and its timezone?
>
> Thanks.

---

## Where each decision lives in the code

| # | Decision | Config / code | Recorded |
|---|---|---|---|
| 1 | convergence: combined reading | `convergence.comparison: block` | D1, D21 |
| 2 | failures count toward 50, not strikes | `record_failure()` | D3 |
| 3 | no refit on train+valid | `selection.refit_on_train_val: false` | — |
| 4 | restarts resume; active time only | `convergence.py`, D2 | D2 |
| 5 | supplementary files unused | `leakage.use_supplementary_files: false` | — |
| 6 | randomised log unused | `leakage.use_random_exposure_log: false` | — |
| 7 | bonus benchmarks not attempted | — | Q9 |
| 8 | deadline 1 Sept 02:00 | — | — |

**If an answer arrives:** update this file and the matching config value **in the same
commit**, and re-run the affected tests. For #1 that is one word plus
`tests/test_convergence.py`; both readings are already implemented and tested.
