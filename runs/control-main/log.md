# Run `control-main`

_Autonomous ML research agent, TikTok TechJam 2026 Track 2._

Every number below was computed by the harness. The hypotheses are the agent's; the measurements are not.

No hidden-test metric appears in this file, or in any log the agent can read. See CLAUDE.md section 5.

## Summary

- Iterations recorded: **4**
- Best validation primary: **0.6025** (iteration 4)
- Kept: 2 | rejected: 2 | failed: 0

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

### Iteration 2 — rejected

**Hypothesis.** The baseline stops on validation primary with patience 4. A longer patience may find a later peak, since the validation curve is noisy at the 0.001 level.

**Target stage.** `model`  
**Patch kind.** `patience`  

| metric | value |
|---|---|
| val_gauc | 0.6671 |
| val_ndcg5 | 0.5358 |
| val_primary | 0.6015 |
| train_primary | 0.6870 |
| gap | 0.0855 |

**Decision.** rejected — val_primary 0.6015 does not beat the best

_Cost: 100s._

### Iteration 3 — rejected

**Hypothesis.** A smaller learning rate may reach a better optimum given that the validation curve peaks early and then declines.

**Target stage.** `model`  
**Patch kind.** `learning_rate`  

| metric | value |
|---|---|
| val_gauc | 0.6670 |
| val_ndcg5 | 0.5358 |
| val_primary | 0.6014 |
| train_primary | 0.6790 |
| gap | 0.0776 |

**Decision.** rejected — val_primary 0.6014 does not beat the best

_Cost: 93s._

### Iteration 4 — KEPT

**Hypothesis.** Stronger L2 may delay the overfitting the epoch curve shows.

**Target stage.** `model`  
**Patch kind.** `regularisation`  

| metric | value |
|---|---|
| val_gauc | 0.6685 |
| val_ndcg5 | 0.5365 |
| val_primary | 0.6025 |
| train_primary | 0.6871 |
| gap | 0.0846 |

**Decision.** KEPT — val_primary 0.6025 beats the best

_Cost: 93s._
