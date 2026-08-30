# Results

Every number here was computed by code and is reproducible from the artefacts in
`runs/`. No hidden-test metric appears anywhere in this file, or in any log the agent
can read. See CLAUDE.md §5.

**All scores are validation primary** = mean(GAUC, nDCG@5) on the official validation
split, 124,909 rows, 22,377 users.

---

## 1. The scoreboard

| | validation primary | vs baseline | source |
|---|---|---|---|
| random | 0.4827 | −0.1188 | sanity floor, `--model random` |
| item popularity | 0.5807 | −0.0208 | `--model pop` |
| **FM baseline** | **0.6015** | — | `--model fm`, published 0.6016 |
| deterministic control | 0.6025 | +0.0010 | `runs/control-main` |
| **agent (run 4, iteration 4)** | **0.6034** | **+0.0019** | `runs/agent-explore4` |
| **agent, confirmed mean of 3 seed sets** | **0.6036** | **+0.0021** | §3 below |
| oracle ceiling | 0.8484 | +0.2469 | perfect ranking |

**The agent beat both the published baseline and our own scripted control.**

The deterministic control matters here. It is a 30-configuration scripted
hyperparameter search over the same harness, and it exists precisely so that "the
agent found something" can be distinguished from "any search would have". It
reached 0.6025, which is inside the noise floor — so **the agent's contribution is
the part above 0.6025, not the part above 0.6015.**

---

## 2. The winning experiment

**Run 4, iteration 4** — `runs/agent-explore4/patches/iter_004.py`

```python
CONFIG = {
    "ensemble": [11, 23, 37, 53, 71],
    "normalise": "within_user_rank",
    "batch": 2048,
    "patience": 5,
}
```

A five-seed ensemble of the **pointwise** FM, blended by within-user rank, trained at
batch 2048 instead of the default 8192.

### How the agent got there

Its first three experiments were all objective rewrites and all failed. It then
changed direction, and the reasoning it gave is the substance of the result:

> *"Three objective rewrites moved nothing, and the diagnostics say why: train-valid
> gap is only +0.0065, so the model **underfits rather than overfits**, and ID
> embedding norms (user 0.10, video 0.14) are 5–7× smaller than tab/dur_bucket
> (0.79/0.74) — the user × video cross that carries the ranking signal is barely
> trained, while single-seed noise (~0.001) swamps the small real differences."*

Three separate inferences, each from a measurement the harness supplied:

1. **Underfitting, not overfitting.** From the train/validation gap of +0.0065. A
   sharper objective has little to bite on, which explains why five loss rewrites
   moved nothing.
2. **The ID embeddings are starved.** From the per-field contribution diagnostic. It
   then did the arithmetic itself: *"1.14M rows at batch 8192 is ~139 Adam steps per
   epoch... cutting the batch to 2048 quadruples the step count at the same data
   cost, which is the cheapest way to let sparse ID embeddings grow without adding
   capacity (k is measured flat)."*
3. **The differences are inside the noise.** From its own last three iterations
   moving ≤0.0001. Averaging seeds attacks that directly.

It also re-derived, unprompted, why the blend is not a no-op:

> *"Per-user normalisation of a SINGLE model is a monotone transform inside the list
> and cannot change GAUC or nDCG@5. Rank normalising each member BEFORE averaging is
> different — the average of several within-user rank vectors is not a monotone
> function of any one member."*

And two pieces of discipline it imposed on itself: **"Equal weights, chosen once, so
nothing is tuned against validation"**, and a deliberate refusal to stack another
objective so that the sampling and variance axes stayed separable.

---

## 3. Confirmation — three independent seed sets

The reported +0.0019 sat just under the 0.002 threshold this project set for
"distinguishable from noise on one seed" (D19), so it was re-run with two fresh seed
sets.

| seed set | member scores | best member | **blend** | blend − best member |
|---|---|---|---|---|
| `[11,23,37,53,71]` | 0.6024, 0.6010, 0.6013, 0.6028, 0.6009 | 0.6028 | **0.6034** | +0.0006 |
| `[3,5,7,13,17]` | 0.6014, 0.6019, 0.6025, 0.6022, 0.6021 | 0.6025 | **0.6040** | +0.0015 |
| `[2,4,8,16,32]` | 0.6016, 0.6009, 0.6021, 0.6010, 0.6021 | 0.6021 | **0.6036** | +0.0014 |

```
mean 0.6036 | min 0.6034 | max 0.6040 | spread 0.0006
vs baseline 0.6015: +0.0021
vs control  0.6025: +0.0011
```

**Two things this establishes.**

**The result reproduces.** A spread of 0.0006 across three independent seed sets is
*smaller* than the 0.0008 single-model seed standard deviation the organisers
publish — which is what a five-seed ensemble should do, and evidence that the
variance reduction is real rather than assumed. The mean, +0.0021, clears the 0.002
bar.

**Blending beats picking a lucky seed.** `blend − best member` is positive in all
three sets. This answers a question the agent posed in its own patch docstring and
converged before it could test:

> *"blend vs best member tells me whether averaging added anything beyond picking a
> lucky seed."*

It did. Between +0.0006 and +0.0015, every time.

---

## 4. Run 4 in full

`runs/agent-explore4/` — unattended, converged on the no-improvement rule.

| # | stage | approach | primary | decision |
|---|---|---|---|---|
| 1 | objective | pairwise BPR | 0.5888 | keep (first score) |
| 2 | objective | listwise softmax + pointwise anchor | 0.6004 | keep |
| 3 | objective | lambdarank nDCG, daily lists | 0.5944 | reject |
| **4** | **ensemble** | **5-seed rank blend, batch 2048** | **0.6034** | **keep** |
| 5 | objective | all-pairs RankNet, full user history | 0.6010 | reject |
| 6 | objective | top-k hard-negative margin | 0.5998 | reject |
| 7 | sampling | metric-matched balanced sampling | 0.6013 | reject |

### Resource usage

| | |
|---|---|
| iterations used | 7 of 50 |
| agent wall clock | 0.74 h of 6 h |
| LLM tokens | 237,365 (7 calls, 0 failed) |
| model | `claude-opus-5` via the Claude Code CLI (subscription, no API credits) |
| GPU-hours | 0 |
| **manual interventions** | **0** |
| operational restarts | 0 |
| recovery events | 1 (one repair attempt) |
| canary trips | 0 |
| results flagged for review | 0 |

*Manual intervention is defined per the organiser webinar as a human changing the
agent's instructions, objective or search space. Restarting a crashed process is
operational recovery and is counted separately.*

---

## 5. What did not work — twenty experiments

Across four agent runs, roughly twenty distinct experiments. Everything below scored
**at or under the pointwise baseline**:

| family | variants tried | best result |
|---|---|---|
| pairwise BPR | 4 | 0.5948 |
| listwise softmax | 5 | 0.6007 |
| lambdarank / nDCG-weighted pairwise | 3 | 0.5944 |
| hybrid listwise + pointwise anchor | 2 | 0.6013 |
| margin hinge, Plackett-Luce, approx-nDCG | 3 | 0.6009 |
| within-list centred logistic | 2 | 0.6014 |
| sampling and list construction | 2 | 0.6013 |

**The consistent finding: changing the objective does not help on this benchmark.**
The agent reached that conclusion itself by iteration 6 of run 1 — *"the loss family
is not the bottleneck"* — and the diagnosis in §2 explains why. Roughly 36% of users
have single-class lists that supply zero gradient to any purely relative objective,
so a ranking loss discards about a third of the training signal and loses more than
the metric alignment gains.

This extends the organisers' own published ablations:

| what | result | measured by |
|---|---|---|
| adding static feature fields | no gain | organisers |
| embedding dimension k = 8/16/32 | flat | organisers |
| **the training objective** | **no gain, 19 ways** | **this project** |

---

## 6. Reproducing any of this

```bash
python scripts/verify_setup.py                 # the M1 foundation, ~5 min
python scripts/control_run.py                  # the deterministic control, ~6 min
python -m harness.submit --check --split test runs/agent-explore4/submission.csv
```

The winning experiment, directly:

```python
from harness import data as d
from harness.models import runners as R
r = R.train_ensemble(d.load(), seeds=(11, 23, 37, 53, 71),
                     normalise='within_user_rank', batch=2048, patience=5)
print(r.val_primary, r.diagnostics['ensemble'])
```

Artefacts: `runs/agent-explore4/log.md` is the human-readable run log,
`patches/iter_004.py` is the code the agent wrote, and `summary.json` carries the
resource figures quoted above.

---

## 7. Open item

**Which run is the scored submission** has not been decided. Run 4 was unattended,
converged, and recorded zero manual interventions, and its submission is written and
validated. A fresh run with frozen code would be the by-the-book reading of
CLAUDE.md §12.2 M4, but the agent has no memory across runs, so a fresh run may not
rediscover the ensemble.

Whichever is chosen, this document reports both, and the run that produced the
submitted file will be named explicitly.
