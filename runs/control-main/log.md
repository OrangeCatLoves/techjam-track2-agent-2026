# Run `control-main`

_Autonomous ML research agent, TikTok TechJam 2026 Track 2._

Every number below was computed by the harness. The hypotheses are the agent's; the measurements are not.

No hidden-test metric appears in this file, or in any log the agent can read. See CLAUDE.md section 5.

## Summary

- Iterations recorded: **1**
- Best validation primary: **0.6015** (iteration 1)
- Kept: 1 | rejected: 0 | failed: 0

## Iterations

### Iteration 1 — KEPT

**Hypothesis.** Reproduce the official baseline to confirm the harness, the trainer and the scoring path agree with the published number before anything is changed.

**Target stage.** `model`  
**Patch kind.** `baseline_reproduction`  

| metric | value |
|---|---|
| val_gauc | 0.6671 |
| val_ndcg5 | 0.5358 |
| val_primary | 0.6015 |
| train_primary | 0.6870 |
| gap | 0.0855 |

**Decision.** KEPT — val_primary 0.6015 beats the best

_Cost: 66s._
