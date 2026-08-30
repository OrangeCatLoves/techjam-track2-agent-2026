# TODO — what is left, who owns it

**Deadline: 1 September, 02:00.**

Two people, split by **file area** so nobody hits a merge conflict. Owner A works in
`runs/`, `harness/`, `agent/`. Owner B works in `README.md` and the writeups. Those
sets do not overlap.

**Read first:** `docs/HANDOVER.md` (setup and current state), then `docs/RESULTS.md`
(every number you will need to quote).

---

## Status at a glance

Eight of nine deliverables are done. The one gap is writing: the README, the
Devpost description and the report.

| deliverable | state | owner |
|---|---|---|
| public GitHub repo | done | — |
| per-iteration run logs | done, committed | — |
| manual-intervention count | done — **0** | — |
| results table | done in `RESULTS.md` | — |
| resource usage | measured | — |
| final submission | **decided — run 4**, validated, committed | — |
| **README** | **14 TODO placeholders** | B |
| **Devpost description** | **not started** | B |
| **detailed report** (required, no video) | **not started** | B |

---

## Owner A — code, runs, the submission

### A1. Push — DONE

History was rewritten to strip commit trailers so the repo shows a single author.
Pushed, verified, and the local backup refs have been cleaned up. GitHub holds the
authoritative history.

*(If GitHub's Contributors sidebar still shows two names, that panel is recomputed on
a schedule rather than on push. It clears itself.)*

### A2. Scored submission — DECIDED, no action needed

**Run 4 is the scored submission**: `runs/agent-explore4/submission.csv`, validated,
committed. Confirmed across three seed sets at mean 0.6036.

Owner B can quote it immediately; nothing blocks the results table.

It is replaced only by a result that is (a) confirmed across several seed sets and
(b) better by more than 0.002. See `RESULTS.md` §7 for why that threshold and not a
smaller one.

### A3. Optional — try to move the score

**Two of the three have been done. Neither improved on run 4.** See `RESULTS.md` §5.

- ~~More ensemble members~~ — **done.** It saturates at five: 3 gives 0.6028, 5 gives
  0.6034, 8 gives 0.6033 at 2.4x the cost. No upside left.
- ~~Separate the confounded variables~~ — **done.** The 2x2 shows the smaller batch
  and the blending *interact*: together they are worth about double the sum of their
  parts, and blending at the default batch size gains almost nothing. This confirmed
  the agent's own hypothesis and is now the mechanism reported in `RESULTS.md` §5.
- **Duration-conditioned recalibration — still untried.** The last untested lever
  CLAUDE.md §9.5 calls real: duration varies *within* a user's list, so an adjustment
  conditioned on it is non-monotone and can genuinely reorder. Unknown payoff.

Anything new replaces run 4 only if confirmed across seed sets **and** better by more
than 0.002. See `RESULTS.md` §8.

---

## Owner B — README, Devpost, report

Nothing here needs new research. Every number and paragraph exists already; this is
assembly.

### B1. Set up from a clean clone, following `HANDOVER.md` only ⚠️

Do this **before** the writing, and do it without asking Owner A for help.

```bash
git clone https://github.com/OrangeCatLoves/techjam-track2-agent-2026.git
cd techjam-track2-agent-2026
# then follow docs/HANDOVER.md §2 exactly
python scripts/verify_setup.py
```

If anything in those instructions is wrong, **write it down and fix it**. This is
CLAUDE.md §12.2 M5's acceptance criterion — *"a clean clone reproduces the baseline
with no hidden setup"* — and you are the only person who can test it, because
Owner A's machine already works.

Two known snags to expect:

- The dataset is a separate 194 MB download; it is not in the repo.
- **`ANTHROPIC_API_KEY` must stay unset.** The agent runs on a Claude Code
  subscription, and setting that variable makes the CLI abandon it and bill an API
  account instead. See `HANDOVER.md` §2.4.

### B2. Fill in `README.md` — 14 placeholders

| line | what | source |
|---|---|---|
| 13 | project overview, two paragraphs | `HANDOVER.md` §1 + `RESULTS.md` §2 |
| 25–26 | results table | `RESULTS.md` §1 |
| 35–40 | resource table | figures below |
| 106 | confirm the run commands | `HANDOVER.md` §9 |
| 127 | architecture description | `MILESTONE_2.md` §2 has an ASCII diagram |
| 174 | limitations, honestly | `HANDOVER.md` §7, `RESULTS.md` §5 |
| 185 | repository layout | `HANDOVER.md` §4 |
| 193 | team contributions | you two |

Resource figures for lines 35–40 (run 4, from `runs/agent-explore4/summary.json`):

```
Iterations used                 7 of 50
Agent wall clock                0.74 h of 6 h
LLM tokens (input + output)     237,365  (188,312 in / 49,053 out)
GPU-hours                       0
Manual interventions            0
Operational restarts            0
Recovery events                 1
```

**On the hidden-test row:** leave it blank. We are not permitted to compute a test
metric and the harness refuses to. Say so in the README rather than leaving an
unexplained gap — it is a deliberate integrity control, not an omission.

### B3. Devpost description

Lead with the framing that makes the result meaningful:

> The deterministic control — a 30-configuration scripted hyperparameter search over
> the same harness — reached 0.6025. **The agent's contribution is the part above
> 0.6025, not the part above the published 0.6015.**

That is stronger than quoting a raw score, because it shows we tested whether *any*
search would have found the same thing.

Worth including:

- What the agent found, and the reasoning that got it there (`RESULTS.md` §2 quotes
  it in the agent's own words).
- **Zero manual interventions.**
- The honest negative result: nineteen objective variants failed, and the agent
  worked out why — roughly 36% of users have single-class lists that give zero
  gradient to any purely relative objective.
- The integrity story: the loader hands us the hidden test labels and we refuse to
  look. `tests/test_no_test_labels.py` proves it.

### B4. The detailed report

Required in place of a video, so it carries real weight. Largely a merge of existing
documents:

| section | source |
|---|---|
| problem and approach | `HANDOVER.md` §1, `MILESTONE_1.md` §1 |
| architecture | `MILESTONE_2.md` §2–4 |
| leakage safety | `MILESTONE_1.md` §2, `HANDOVER.md` §3 |
| what was tried | `RESULTS.md` §5 |
| the result | `RESULTS.md` §2–3 |
| what did not work, and why | `RESULTS.md` §5 |
| limitations and future work | `HANDOVER.md` §6, §7, §10 |
| decisions and open questions | `OPEN_QUESTIONS.md` (D1–D21) |

Include a short line on tooling: the harness was written with Claude Code; the
experiments were designed and coded by the agent itself. The distinction is the
substance of the submission and reads as rigour, not as a caveat.

---

## Shared — either owner

- **Send the four open organiser questions.** `docs/QUESTIONS_FOR_ORGANISERS.md` has
  a ready-to-paste block. The convergence-semantics one is the most consequential:
  we took a team decision that has not been ruled on.
- **Final check before submitting:**
  ```bash
  python -m harness.submit --check --split test <the submission>
  pytest tests/ -m "not slow"
  ```

---

## What not to spend time on

- **Another loss function.** Nineteen tried, none worked.
- **Adding features, or a bigger embedding dimension.** The organisers measured both
  as dead ends and published the numbers.
- **Chasing the score at the expense of the writeups.** Presentation is 10% currently
  at zero, and the report is a stated requirement. The score has yielded +0.0021
  across roughly twenty experiments.
