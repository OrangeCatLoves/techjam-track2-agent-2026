# Questions for the organisers — ready to send

Copy the block below into the organiser channel. It is written to be answerable in
one reply, with our default stated for each so a non-answer is still safe.

**Why these two are urgent:** Q3 and Q4 govern code that is already written and
tested. If either comes back different from the reading we implemented, the change
is small now and expensive after the scored run.

---

## Paste this

> Hi — a few clarifications on Track 2 (KuaiRand-Pure). We have implemented a
> conservative default for each, so a non-answer is safe; we would rather be right
> than safe.
>
> **1. Convergence semantics.** `baseline_scores.json` gives `epsilon = 0.002, N = 3`
> and the README says "three consecutive iterations where validation primary
> improves by no more than 0.002". Two readings stop at different points:
>
> - (a) *each* of the last three iterations improved the running best by ≤ 0.002
> - (b) best(last three) − best(before those three) ≤ 0.002
>
> They disagree on, for example, three consecutive gains of 0.0015: individually
> under epsilon, together 0.0045 over it. We have implemented (a), the stricter
> reading, which stops no later. Which did you intend?
>
> **2. Failed iterations.** If a candidate errors out and is abandoned, does it
> consume one of the 50, and does it count as a non-improving iteration for the
> three-strike window? We currently count it toward the 50 but not as a strike,
> since it produced no validation score.
>
> **3. Refitting on train + validation.** The rule says the scored submission is the
> "validation-best checkpoint". May the winning configuration be refitted on train +
> validation before predicting test? A refitted model was never itself scored on
> validation, so a literal reading forbids it. We have it implemented and disabled.
>
> **4. Restarts.** We understand from the webinar that restarting a crashed process
> is operational recovery rather than a manual intervention. Confirming the
> accounting: on a restart we resume iteration count, strike count and the tried-set
> from disk, and we charge only *active* agent time against the six hours (so an
> overnight gap between a crash and its discovery is not charged). Is that right?
>
> **5. Supplementary files.** Are `kuairand_video_captions.csv` and
> `kuairand_video_categories.csv` (published by the same authors, separate Zenodo
> record) in scope, or does "the KuaiRand datasets listed below" exclude them? We
> are not using them.
>
> **6. The randomised-exposure log.** May `log_random_4_22_to_5_08_pure.csv` be used
> as an additional *unbiased validation* set, as the starter README suggests? It
> overlaps the test window, so we are not using it at all, for training or
> validation.
>
> **7. Bonus benchmarks.** KuaiRand-1k and 27k are named as bonus benchmarks but the
> kit ships no splits, baseline or scores for them. Are those published anywhere, or
> is the bonus self-defined?
>
> **8. Deadline.** Could you confirm the submission deadline and its timezone, and
> whether it differs from the 11 September event date?
>
> Thanks.

---

## Our defaults, for our own reference

| # | Question | Default in force | Where |
|---|---|---|---|
| 1 | convergence semantics | stricter, per-iteration | `convergence.comparison: per_iteration`, D1 |
| 2 | failed iterations | count toward 50, not a strike | `record_failure()`, D3 |
| 3 | refit on train+valid | disabled | `selection.refit_on_train_val: false` |
| 4 | restart accounting | resume all counters; charge active time only | D2 |
| 5 | supplementary files | not used | `leakage.use_supplementary_files: false` |
| 6 | randomised log | not used | `leakage.use_random_exposure_log: false` |
| 7 | bonus benchmarks | not attempted | Q9 |
| 8 | deadline | **unknown — this one blocks planning** | — |

## If an answer arrives

Update `docs/OPEN_QUESTIONS.md` and the matching config value **in the same commit**,
and re-run the affected tests. For Q1 that is `convergence.comparison` plus
`tests/test_convergence.py`; both readings are already implemented and tested, so it
is a one-word change.
