# Results — second contributor

**This file belongs to one person. Nobody else edits it.**

`docs/RESULTS.md` is the master document and is owned by contributor A. Recording
findings here instead means two people can work at the same time without ever
touching the same file, so git has nothing to conflict over.

At the end, A folds anything that matters into `RESULTS.md` §1 and §8. That merge is
a two-minute copy, not a negotiation.

**Naming:** my run directories are `runs/trial1` … `runs/trial4`. They were created
as `runs/swetha-1` … `runs/swetha-4` and renamed afterwards, artefacts included, so
the `run_id` inside each `summary.json` matches its directory. Nothing else about
them changed.

---

## The bar to beat

| | validation primary |
|---|---|
| current submission (run 4) | **0.6036** confirmed |
| **replaces run 4 only if** | **> 0.6056** *and* confirmed across seed sets |

Why 0.002 and not any improvement: validation is 124,909 rows with ±0.0008 seed
noise, so taking the maximum over many experiments manufactures roughly +0.0012 of
luck — about the size of the entire real edge. Full reasoning in `RESULTS.md` §8.

---

## What I did, in one paragraph

Four autonomous agent runs from a clean clone on a second machine, plus two hand-run
ablations extending `RESULTS.md` §5. **Nothing cleared the bar; run 4 remains the
scored submission.** What came out instead was confirmation and closure — an
independent replication of one §5 cell, an independent reproduction of run 4's
central diagnosis, a measured tradeoff curve that upgrades §5's mechanism from
indicative to confirmed, a clean negative that closes the ensembling axis in its last
degree of freedom, and three runs converging on the same configuration. Alongside
that, four defects in the instrumentation that reports to judges: a token
over-count of roughly 4× in every `resources.md`, a usage counter that reports zero
after a restart, a capability the agent has never been able to reach because it is
undocumented, and a guard that aborts a run instead of rejecting a record. The M5
criterion — a clean clone reproducing the baseline with no hidden setup — holds.

---

## My runs

| run | scope | best primary | vs 0.6036 | outcome |
|---|---|---|---|---|
| `runs/trial1` | 6 iterations of 12, converged | 0.6020 | −0.0016 | five objective variants, then a seed ensemble at default batch |
| `runs/trial2` | 5 iterations of 20, converged | 0.6020 | −0.0016 | BPR, then hybrid pointwise + listwise; 1 restart, 3 recoveries |
| `runs/trial3` | 6 iterations of 20, converged | 0.6020 | −0.0016 | converged on the identical configuration to `trial1` |
| `runs/trial4` | crashed at iteration 1 | — | — | `LeakageError` aborted the run — see Defect 3 |
| batch sweep (hand-run) | 3 configs | 0.6034 | −0.0002 | batch 2048 / 1024 / 512 on the confirmed winning config |
| heterogeneous ensembles (hand-run) | 5 configs | 0.6034 | −0.0002 | mixed per-member batch sizes, 0 to 3 low-batch members of 5 |

**Nothing here clears the bar. Run 4 remains the scored submission.** Each run wrote a
valid `submission.csv`, and none is force-added, because all are below run 4.

The two sweeps are hand-run ablations in the same spirit as `RESULTS.md` §5, not agent
results, and must not be reported as such.

### Run conditions

| | `trial1` | `trial2` | `trial3` |
|---|---|---|---|
| iterations used | 6 of 50 | 5 of 50 | 6 of 50 |
| converged | `no_improvement` | `no_improvement` | `no_improvement` |
| agent wall clock | 0.56 h | 0.23 h | 0.39 h |
| LLM tokens (correct) | 192,259 | see Defect 2 | 187,275 |
| manual interventions | 0 | 0 | 0 |
| operational restarts | 0 | 1 | 0 |
| recovery events | 0 | 3 | 0 |
| canary trips | 0 | 0 | 0 |

All runs used `claude-opus-5` via the Claude Code CLI on a **Claude Pro**
subscription — a second machine, a clean clone, and a different subscription tier
from the one that produced run 4. The pipeline reproduced end to end with no code
changes.

---

## Finding 1 — an independent replication of a `RESULTS.md` §5 cell

The §5 ablation reports a 2×2 in which **a five-seed blend at batch 8192 scores
0.6020**, and concludes that blending at the default batch size gains almost nothing.
Each cell of that 2×2 is a single measurement, and §5 says so plainly: *"the
decomposition is indicative rather than confirmed."*

**`trial1` iteration 6 is an independent second measurement of that cell, and it lands
on 0.6020.**

| | batch 8192, 5-seed blend |
|---|---|
| `RESULTS.md` §5, 2×2 cell | 0.6020 |
| `trial1` iteration 6 | 0.6020 |

The agent that produced it had no memory of run 4, no access to the 2×2, and reached
a seed ensemble by its own route after five objective experiments failed. It chose
rank-averaging within each user's list — the same normalisation — and left the batch
size at the default, which is the one variable the 2×2 says carries the interaction.

Two independent measurements agreeing to four decimal places is stronger evidence
than either alone. **The §5 caveat can be narrowed for this cell specifically.**
Findings 3 and 4 extend that to the mechanism as a whole.

### What this says about the mechanism

§5 argues the two ingredients interact: the smaller batch gives sparse ID embeddings
more Adam steps, the seeds therefore converge to more *different* solutions, and
diversity is what a blend exploits. This run is the negative half of that claim,
measured separately. Blending on its own, at the default batch, gained **+0.0003**
over the best single configuration in the same run (0.6017 → 0.6020) — consistent
with the +0.0002 the 2×2 reports, and inside the noise band either way.

One detail the 2×2 does not record. In this run the blend gain was **entirely in
nDCG@5**:

| | GAUC | nDCG@5 | primary |
|---|---|---|---|
| iteration 3 (best single) | 0.6677 | 0.5356 | 0.6017 |
| iteration 6 (5-seed blend) | 0.6677 | 0.5362 | 0.6020 |

GAUC did not move at all to four decimal places. That is what within-user rank
averaging should do when member diversity is low — it can reorder the top of a list
without changing the global pairwise ordering much.

---

## Finding 2 — two agents, same diagnosis, independently

`RESULTS.md` §2 quotes run 4's agent identifying starved ID embeddings from the
per-field norm diagnostic: user 0.10, video 0.14 against tab/dur_bucket 0.79/0.74.

`trial1` iteration 5 reached the same diagnosis unprompted:

> *"Diagnostics show the model is effectively a duration/tab ranker: dur_bucket
> embedding norm 1.027 vs user_id 0.106 and video_id 0.153, so the ID crosses that
> should carry personalisation are near-dead. Every objective tried so far (logistic,
> softmax CE, LambdaRank, BPR) saturates once a pair is ordered, supplying no
> pressure to grow rare ID embeddings, which is why GAUC has not moved."*

Same mechanism, same numbers, different run, no shared memory. This matters for
Innovation & Problem Insight, which is judged on *what the agent identified as worth
trying and why*: a diagnosis two independent runs converge on is evidence of genuine
inference from the diagnostics rather than a plausible story fitted after the fact.

The agent then acted on it differently from run 4 — it proposed WARP rank-weighted
hinge, reasoning that a non-saturating margin would keep applying pressure to grow
the rare embeddings. That scored 0.6012 and was rejected. Correct diagnosis, wrong
remedy, which is itself informative: the embeddings are not starved for *gradient
pressure*, they are starved for *steps*, which is what run 4's batch-size change
supplied.

---

## Finding 3 — the §5 mechanism confirmed, and bounded

§5's 2×2 has only two batch levels, so it cannot separate the diversity effect it
proposes from plain member quality. Extending the sweep below 2048 separates them.

Hand-run on the confirmed winning configuration, same seed set as run 4
`(11,23,37,53,71)`, `normalise='within_user_rank'`, `patience=5`, everything else at
reference.

| batch | best member | blend | blend − best member | secs |
|---|---|---|---|---|
| 8192 | — | 0.6020 | +0.0002 * | — |
| **2048** | **0.6028** | **0.6034** | **+0.0006** | 551 |
| 1024 | 0.6016 | 0.6027 | +0.0012 | 581 |
| 512 | 0.6006 | 0.6023 | +0.0017 | 803 |

\* *from §5's 2×2, measured against a single model rather than best-of-5, so not
strictly comparable to the rows below. Directionally consistent.*

The 2048 row reproduces `RESULTS.md` §3's first seed set exactly — blend 0.6034, best
member 0.6028, gain +0.0006 — on a different machine, different OS install, different
numpy build. **The harness is deterministic across platforms**, which supports the M5
reproducibility claim independently of this result.

### Two effects, moving in opposite directions

**The blend gain rises monotonically as the batch shrinks.** +0.0006 at 2048, +0.0012
at 1024, +0.0017 at 512 — roughly tripling across the range. This is the diversity
half of the §5 mechanism, measured directly rather than inferred.

**Member quality falls at the same time.** 0.6028 at 2048, 0.6016 at 1024, 0.6006 at
512. Below 2048 the members degrade faster than the extra diversity can recover.

The blend score is the product of those two forces, and it **peaks at 2048**. The
curve declines on both sides: 0.6020 at 8192, 0.6034 at 2048, 0.6027 at 1024, 0.6023
at 512.

### What this establishes

**The diversity mechanism is confirmed.** §5's caveat can be narrowed from the
mechanism as a whole to the individual 2×2 cells.

**The mechanism is bounded, and the boundary is located.** More steps is not a free
lever — there is an optimum, not a monotone improvement.

**Run 4's configuration is not a lucky draw.** It sits at the peak of a curve that
declines in both directions. The agent selected the optimum of a tradeoff on its first
attempt, without sweeping it — having reasoned its way there from the train/validation
gap and the embedding-norm diagnostic. That is a stronger claim for the agent than the
raw +0.0021 delta.

### Caveat

Each row is a single measurement on one seed set, and single-model seed noise is
±0.0008. The 1024-vs-512 ordering in particular should not be leaned on. What survives
the noise is the **monotone trend in the gain column** across four levels and the
**direction of the member-quality decline**. Confirming the shape properly would need
three seed sets per row, nine more runs, which was not affordable before the deadline.

---

## Finding 4 — heterogeneous-batch ensembles: the tradeoff cancels exactly

Finding 3 left a question open. Members do not have to *share* a batch size. Two
models trained at different batch sizes differ by more than two trained at the same
batch with different seeds, so mixing should let high-quality 2048 members carry the
score while lower-batch members act purely as diversity donors — quality *and*
diversity, instead of a trade between them.

Every ensemble in both contributors' runs is homogeneous, so this had never been
tested. Five configurations, run 4's seed set, one batch size per seed, equal weights.

| config | best member | blend | blend − best member | GAUC | nDCG@5 | secs |
|---|---|---|---|---|---|---|
| homogeneous 2048 (control) | 0.6028 | **0.6034** | +0.0006 | 0.6700 | 0.5367 | 389 |
| 4×2048 + 1×1024 | 0.6028 | 0.6033 | +0.0005 | 0.6696 | 0.5370 | 404 |
| 3×2048 + 2×1024 | 0.6024 | 0.6033 | +0.0009 | 0.6697 | 0.5368 | 441 |
| 3×2048 + 1×1024 + 1×512 | 0.6024 | **0.6034** | +0.0010 | 0.6700 | 0.5368 | 531 |
| 2×2048 + 2×1024 + 1×512 | 0.6024 | 0.6031 | +0.0007 | 0.6696 | 0.5365 | 548 |

### The control reproduces run 4 member by member

The homogeneous row returned members 0.6024, 0.6010, 0.6013, 0.6028, 0.6009 and a
blend of 0.6034 — **identical to `RESULTS.md` §3's first seed set row, individual
member scores included**, on different hardware and a different Python install.

### The hypothesis was right about the mechanism and wrong about the payoff

**The diversity effect appeared exactly as predicted.** Blend gain rose from +0.0006
to +0.0010 as lower-batch members were mixed in.

**The blend score did not move.** All five configurations land between 0.6031 and
0.6034 — a spread of 0.0003, well inside the ±0.0008 noise band. GAUC varies by 0.0004
across the whole table and nDCG@5 by 0.0005.

Each lower-batch member added raises the blend gain and lowers the best member by
approximately the same amount, and the two cancel. **Diversity costs exactly what it
is worth.**

### What this establishes

This is a sharper negative than "mixing does not help." Finding 3 showed the tradeoff
is unfavourable *below* 2048. Finding 4 shows it is **exactly balanced** — no mixing
ratio exploits it, not merely that the ones tried did not.

Combined with §5's member-count saturation at five, the ensembling axis is bounded in
all three degrees of freedom: **how many members** (saturates at 5), **what batch**
(optimum 2048), and **whether members should differ structurally** (no gain
available).

---

## Finding 5 — three runs, one configuration

`trial1` and `trial3` both converged on **0.6019552430716881** — identical to the last
digit, from different agent seeds and different proposal trajectories. `trial2`
landed at 0.6020388494156208, a different configuration within 0.0001.

| run | agent seed | iterations | best primary |
|---|---|---|---|
| `trial1` | default | 6 | 0.6019552430716881 |
| `trial2` | 7 | 5 | 0.6020388494156208 |
| `trial3` | 13 | 6 | 0.6019552430716881 |

Two independent runs arriving at bit-identical scores means they found the same
configuration by different routes. Together with run 4's member-by-member
reproduction in Finding 4, this is strong evidence that **the harness is fully
deterministic and the agent's search converges on a stable attractor** rather than
wandering.

It also sharpens the limitation below: three runs, fifteen iterations, and the search
lands in the same place every time — which is what you would expect from an agent
with no memory exploring a surface whose productive region it cannot see.

---

## The structural limitation these runs expose

Across `trial1`, `trial2` and `trial3`, thirteen of seventeen iterations went to the
objective. `CLAUDE.md` §9.1 and `knowledge/methods.md` already record objectives as a
swept dead end, and `RESULTS.md` §6 documents nineteen failures — **but none of that
reaches the agent's search**. It rediscovers the dead end from scratch, spends its
convergence budget doing so, and converges on or just after the first iteration that
touches a live lever.

`trial1`'s convergence arithmetic: iterations 4–6 best 0.6020 against 0.6017 before
them, so +0.0003 ≤ ε = 0.002 and the block rule fired at iteration 6. The
`max_iterations` cap was never reached in any run. **Every run stopped within one
iteration of finding the ensemble lever, with no chance to tune it.**

Finding 3 shows what they would have found had they continued: the batch axis they
never reached is where the remaining gain lives, and the optimum sits at a value the
agent would have had to search for. The agent has no cross-run memory, so prior
negative results do not prune its search space, and on a benchmark where the
productive region is narrow it exhausts its convergence budget before reaching it.
Feeding the method corpus's own recorded dead ends into the proposal prompt is the
obvious fix and was not attempted.

---

## Status of every axis

| axis | status | measured by |
|---|---|---|
| objective / loss | ~37 variants, all at or below pointwise | two contributors, nine runs |
| static feature fields | worse — 0.5940 vs 0.5950 | organisers |
| embedding dimension k | flat across 8 / 16 / 32 | organisers |
| ensemble member count | saturates at 5 | `RESULTS.md` §5 |
| batch size | optimum at 2048, declines both sides | Finding 3 |
| heterogeneous member batch | exactly cancelling, no gain | Finding 4 |
| **snapshot ensembling** | **never testable — see Defect 1** | — |
| **duration-conditioned recalibration** | **untried** | — |

**Run 4's 0.6034 is a peak, not a plateau, and the surface around it is mapped in
every direction the agent could reach.**

### Note on provenance

Findings 3 and 4 are hand-designed ablations in the same spirit as `RESULTS.md` §5,
not agent results. Neither beat run 4, so the question of whether a hand-tuned config
should become the scored submission did not arise. Had one cleared the bar it would
have needed A's agreement first: swapping in a human-designed configuration would
weaken the claim that the scored submission is what the autonomous agent produced,
and autonomy carries 20% of the grade against the primary metric's share of 35%.

Equal blend weights were used throughout and never tuned. Weighting the 2048 members
higher would very likely have improved the validation number and would have been
fitting to 124,909 rows of validation noise — the same discipline `RESULTS.md` §2
records run 4's agent imposing on itself.

---

# Instrumentation defects

All four affect what gets reported to judges. All are in contributor A's file areas
and are flagged here, not changed.

## Defect 1 — the agent cannot reach a capability the harness already has

`harness/models/runners.py` implements snapshot ensembling in full:

- `train_fm(..., snapshots: int = 1)` (line 228) keeps a deque of recent epoch states
  and returns the window nearest the validation-best epoch as
  `TrainResult.snapshot_states`.
- `train_ensemble(..., snapshots: int = 1)` (line 461) expands each seed's snapshots
  into separate blend members and reports `snapshots_per_seed` in diagnostics.
- The window rule is already correct: nearest epochs to the peak, later preferred on a
  tie. **Fixed window, no epoch selected by its own validation score.**

`harness/_run_patch.py` forwards a patch's `CONFIG` with `**config` and no key
whitelist, so this patch would run today, unchanged:

```python
CONFIG = {"ensemble": [11, 23, 37, 53, 71], "snapshots": 4, "batch": 2048}
```

**But `CONFIG_KEYS` in `agent/propose.py` never mentions the key.** The string has no
occurrence of the word `snapshots`. That string is the only place the agent learns
what a patch may contain — the comment at line 181 says so outright: *"Stated because
the model cannot see train_fm."*

So across roughly forty experiments and nine runs, no agent has proposed snapshot
ensembling, and none could have. This is a documentation gap, not a capability gap
and not a judgement failure. Compare duration-conditioned recalibration, which *is* in
the corpus at char 8,648 of 12,568 — inside the 14,000 cap, never truncated — and
which the agent has also never chosen. That one is a real decision. This one was never
on the menu.

**Why it matters beyond bookkeeping.** Finding 4 showed that every way of buying
member diversity also costs member quality, and the two cancel exactly. Snapshots are
the one source of diversity that does not: consecutive epochs near the peak sit at
roughly equal validation primary but in different places in parameter space, so they
disagree about *which* users they rank correctly rather than about how good they are.
It is also nearly free — one training run yields several members.

The fix is an eight-line addition to `CONFIG_KEYS`, written out in
`docs/PROPOSAL_expose_snapshots.md`. It is a prompt string with no behaviour attached,
so no test can regress. It should be declared in the writeup as a change to the
agent's search space, and any run after it reported separately from run 4's
zero-intervention claim.

## Defect 2 — `resources.md` over-counts tokens by roughly 4×

`ledger.jsonl` records **cumulative** token totals, not per-iteration ones. Run 4's
input series:

```
26,582 → 53,505 → 80,443 → 107,383 → 134,301 → 161,280 → 188,312
```

Each entry is the running total, rising by about 27k per call. The final entry,
188,312 in + 49,053 out, is **237,365** — exactly `usage.total`.

`resources.md` sums those cumulative snapshots, counting iteration 1's tokens seven
times, iteration 2's six times, and so on, giving 751,806 input and 954,427 total.

**`usage.total` is correct. `resources.tokens.total` is a summing bug.** Corrected
figures:

| run | correct total | `resources.md` claims |
|---|---|---|
| `agent-explore4` | **237,365** | 954,427 |
| `trial1` | **192,259** | 665,889 |
| `trial3` | **187,275** | 638,803 |
| `trial2` | see Defect 3 | 367,937 |

`RESULTS.md` §4's 237,365 was right all along, and the true cost is a quarter of what
the committed artefact claims — the good direction for Feasibility & Practicality,
which is graded in coarse low/medium/high tiers. But every `resources.md` in every run
directory currently overstates it, and deliverable 4 requires this number.

## Defect 3 — the usage counter does not survive a restart

`trial2` reports `usage: {calls: 0, total: 0, by_model: {}}` for a run that completed
five iterations with real LLM calls. Its `log.md` records per-iteration costs
(iteration 1: 27,096 in / 1,720 out), and the hypotheses are plainly diagnosis-driven
rather than from the deterministic fallback queue.

The cause is `operational_restarts: 1`. The usage counter is process-local; when the
run crashed and resumed from the ledger, the new process counted zero calls. All five
iterations had already completed, so nothing was added after.

Since deliverable 4 requires total token consumption, **a restarted run currently
reports none**. The ledger's last cumulative entry is the recoverable figure.

## Defect 4 — a rejected record aborts the whole run

`trial4` crashed at iteration 1:

```
harness.guards.LeakageError: log.jsonl record at log.jsonl record.hypothesis
contains a test metric line.
```

The leak guard fired on the agent's own **hypothesis text** — prose it wrote, which
the pattern matcher read as a test metric — and the exception propagated up through
`logger.log_iteration` → `_write` → `_decide` → `step` → `run`, killing the process
mid-iteration.

The guard is right to refuse the record. Aborting the run is the defect. The
organisers' Robustness criterion, scored under Technical Execution (35%), reads:
*"when a step fails, the agent can recover, retry, or route around it, and long
iterative runs neither crash, stall, nor diverge."* This crashed, on the agent's own
output, at the first iteration.

The obvious handling is to reject the record, log the rejection as a recovery event,
and continue to the next iteration — the machinery for recovery events already exists
and `trial2` used it three times.

---

## Setup findings — a clean clone verifies

`TODO.md` B1 asks for a clean-clone setup following `HANDOVER.md` only. Done, on
Windows 11 / Python 3.14.6 / numpy 2.4.2. `scripts/verify_setup.py` reproduces the FM
baseline at **0.6015** and passes all eleven checks on a clean tree. **A clean clone
verifies.** One safeguard fired correctly during setup, and two documentation
corrections are worth making.

**1. `.env.example` is easy to fill in by mistake — and the guards caught it.**

During setup I pasted an API key into `.env.example` instead of `.env`. The two guard
tests failed immediately:

```
FAILED tests/test_guards.py::test_env_example_carries_no_filled_secret
FAILED tests/test_guards.py::test_no_tracked_file_contains_an_api_key_pattern
2 failed, 328 passed
```

**Nothing was committed.** The tracked file has always carried an empty value; the key
existed only as an uncommitted change in my working tree, and `git checkout --
.env.example` cleared it. `HANDOVER.md` §2.4 records an earlier incident and the two
tests added to prevent a repeat — they work, and they caught a fresh instance of
exactly the mistake they were written for, within minutes, on a new machine.

Reported as evidence the safeguard functions rather than as a defect. It is also a
fair argument for the report: `verify_setup.py` refusing to pass on a working-tree
secret is the correct behaviour, and the M1 gate is doing real work rather than
decorating the repo.

**2. `.env.example` contradicts `HANDOVER.md` §2.4.**

| | `.env.example` | `HANDOVER.md` §2.4 | what run 4 used |
|---|---|---|---|
| `LLM_PROVIDER` | `anthropic` | `claude_cli` | `claude_cli` |
| `LLM_MODEL_FAST` | `claude-opus-5` | `claude-haiku-4-5-20251001` | `claude-haiku-4-5-20251001` |

`summary.json` confirms `HANDOVER.md` is correct. Following `.env.example` as written
would route through an API account — the exact failure §2.4 warns about.

**3. `HANDOVER.md` §2.4's install instruction is outdated.**

It gives `npm install -g @anthropic-ai/claude-code`, which fails on a machine without
Node.js. The current recommended install is a native binary needing no Node:

```powershell
irm https://claude.ai/install.ps1 | iex     # Windows PowerShell
```

Verified working — Claude Code 2.1.251, authenticated on Claude Pro, ran four agent
runs with `claude-opus-5` as the strong model.

---

## What I would do with more time

Ranked by expected value. The **ensembling axis is closed in all three degrees of
freedom** the agent could reach (Findings 3 and 4), so nothing below involves tuning
it further.

1. **Expose `snapshots` in `CONFIG_KEYS`** (Defect 1). Eight lines, no behaviour
   change, no regression risk, and it is the only remaining source of member diversity
   that does not cost member quality — the exact gap Finding 4 identified. The agent
   then chooses whether to use it, as it chose the ensemble stage in run 4.
2. **Fix the run-abort on a rejected record** (Defect 4). Robustness is scored, and
   this is a first-iteration crash on the agent's own output.
2. **Feed recorded dead ends into the proposal prompt.** Roughly thirty-seven
   documented objective failures currently buy the agent nothing; thirteen of
   seventeen iterations across my three completed runs rediscovered them, and every
   run converged within one iteration of first touching a live lever.
3. **Duration-conditioned recalibration** — `CLAUDE.md` §9.5 and
   `knowledge/methods.md` both call it a real lever, and it is the only one the agent
   can already reach and has never chosen. Steering it there costs the
   zero-intervention claim, so it should be a clearly labelled separate run.
4. **Confirm Findings 3 and 4 properly.** Three seed sets per configuration, roughly
   twenty-four runs and five hours.
5. **The bonus benchmarks.** KuaiRand-1k and 27k earn additive points and cost nothing
   from the Pure score if skipped. Nobody has touched them. Given Pure has yielded
   +0.0021 across roughly forty experiments and is measurably saturated on every axis
   the agent can reach, this is plausibly better expected value than squeezing Pure
   further — though 11.7M interactions against a numpy-only pipeline is a real time
   risk.

---

## Working without conflicts

| file | owner |
|---|---|
| `runs/trial*/` | me |
| `docs/RESULTS_teammate.md`, `docs/PROPOSAL_expose_snapshots.md` | me |
| `README.md` and the writeups | me |
| `docs/RESULTS.md`, `docs/TODO.md` | A |
| `harness/`, `agent/`, `tests/`, `.env.example` | A |

- **Never edit a file in the other person's column.** Every defect above is reported,
  not changed.
- **Pull before you push.**

  ```bash
  git pull --rebase origin main
  git push origin main
  ```

---

## For A to action

1. **The token question is resolved.** `usage.total` is correct; `resources.md` sums
   cumulative ledger snapshots and overstates by ~4×. `RESULTS.md` §4's 237,365 stands.
   The aggregator needs fixing, and the usage counter needs to survive a restart.
2. **Fix the run-abort on a rejected log record** (Defect 4). Reject the record, log a
   recovery event, continue.
3. **Expose `snapshots` in `CONFIG_KEYS`** — exact text in
   `docs/PROPOSAL_expose_snapshots.md`.
4. **Fix `.env.example`** to match `HANDOVER.md` §2.4, and update §2.4 to the native
   installer.
5. **For `RESULTS.md` §5:** the blend-at-8192 cell has a second independent
   measurement at 0.6020; Finding 3 extends the batch ablation to 1024 and 512;
   Finding 4 adds the first heterogeneous-batch ensembles. The diversity mechanism is
   confirmed by a monotone gain trend across four batch levels *and* five mixing
   ratios, and the optimum is at 2048 — where run 4 already sits. It turns "we found a
   good config" into "we found the peak of a measured surface."
6. **For `RESULTS.md` §6:** roughly eighteen more objective variants across three
   cold-start runs, all flat.
7. **For `RESULTS.md` §8:** the replacement rule can now lean on "every axis the agent
   can reach has been measured and closed," which is a stronger justification for run 4
   standing than the threshold alone.