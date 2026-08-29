# Starter kit notes

Version 3. Setup verification complete. All three reference commands run and matched
the published numbers.

Machine: Windows, `C:\Users\Lenovo\track2-techjam\`
Data: `C:\Users\Lenovo\track2-techjam\data\KuaiRand-Pure\data`

---

## 1. File inventory

```
starter/
  baseline.py            three baselines: pop, fm, random
  data.py                loading, official splits, categorical encoding
  evaluate.py            GAUC / nDCG@5. DO NOT MODIFY
  submit.py              generate / check / score submissions
  baseline_scores.json   official published scores, seed variance, convergence params
  ablation_features.py   reproduces the "static features don't help" result
  README.md              the organisers' own guidance. Read it in full
```

---

## 2. Data path — RESOLVED, no junction needed

`data.load(data_dir)` takes the directory as a positional argument. All three runnable
scripts expose `--data_dir`, defaulting to `./KuaiRand-Pure/data`.

Confirmed working value:

```
C:\Users\Lenovo\track2-techjam\data\KuaiRand-Pure\data
```

Set `KUAIRAND_DATA_DIR` in `.env` to this and pass it through. This is what ships,
because it works for anyone cloning the repo.

---

## 3. What `data.load()` returns

A dict keyed `'train'`, `'valid'`, `'test'`. Each value is a list of 7-tuples:

| Index | Field | Type |
|---|---|---|
| 0 | `date` | int |
| 1 | `user_id` | str |
| 2 | `video_id` | str |
| 3 | `author_id` (joined from `video_features_basic_pure.csv`, `'UNK'` if missing) | str |
| 4 | `tab` | str |
| 5 | `duration_ms` | float |
| 6 | **`long_view` label** | int 0/1 |

Row order is deterministic: `log_standard_4_08_to_4_21_pure.csv` read first, then
`log_standard_4_22_to_5_08_pure.csv`, filtered by date preserving file order.
**This ordering defines `row_id`. Never reindex or sort the evaluation list.**

`encode(splits)` maps the five fields to contiguous integer ids with a per-field UNK
slot, vocabularies built from **train only**. Returns `(enc, total_dim)` where
`enc[split] = (X int32 (N,5), y float32 (N,), users list)`.

### What is NOT loaded

`user_features_pure.csv`, `video_features_statistic_pure.csv` and
`log_random_4_22_to_5_08_pure.csv` are never opened. From
`video_features_basic_pure.csv` only `author_id` is taken. From the log, only
`date`, `user_id`, `video_id`, `tab`, `duration_ms`, `long_view` are read.

**No leakage from the loader itself.** `play_time_ms`, `is_click` and the other
same-impression outcomes are never touched.

---

## 4. CRITICAL: test labels are returned, and test scores are printed

`out['test']` is 170,588 tuples **with true labels at index 6**.

Confirmed by observation: running `baseline.py --model fm` printed

```
test   GAUC 0.6621 | nDCG@5 0.5286 | primary 0.5953
```

directly to stdout. `ablation_features.py` prints test scores too, on every line.

Required controls, see CLAUDE.md section 5:

- [ ] `harness/data.py` strips index 6 from the test split
- [ ] `harness/guards.py` filters starter stdout before the agent sees it
- [ ] `test_no_test_labels.py` passes

**Filter target:** any stdout line matching `^\s*test\s` or containing `test ` followed
by a metric name. Log raw output to a human-only file.

---

## 5. Split verification — PASS

```
{'train': 1141112, 'valid': 124909, 'test': 170588}
fields=['user_id', 'video_id', 'author_id', 'tab', 'dur_bucket']
```

| Split | Expected | Actual | Match |
|---|---|---|---|
| train | 1,141,112 | 1,141,112 | yes |
| valid | 124,909 | 124,909 | yes |
| test | 170,588 | 170,588 | yes |

---

## 6. Baseline reproduction — PASS

```
python baseline.py --model fm --data_dir C:\Users\Lenovo\track2-techjam\data\KuaiRand-Pure\data
```

Full stdout:

```
loading C:\Users\Lenovo\track2-techjam\data\KuaiRand-Pure\data ...
{'train': 1141112, 'valid': 124909, 'test': 170588} fields=['user_id', 'video_id', 'author_id', 'tab', 'dur_bucket']
  epoch  1 | loss 0.6391 | valid GAUC 0.6467 nDCG@5 0.5272 primary 0.5869 | 10.0s
  epoch  2 | loss 0.5479 | valid GAUC 0.6589 nDCG@5 0.5323 primary 0.5956 | 10.2s
  epoch  3 | loss 0.5129 | valid GAUC 0.6642 nDCG@5 0.5344 primary 0.5993 | 8.5s
  epoch  4 | loss 0.5004 | valid GAUC 0.6642 nDCG@5 0.5346 primary 0.5994 | 8.6s
  epoch  5 | loss 0.4941 | valid GAUC 0.6661 nDCG@5 0.5360 primary 0.6010 | 8.8s
  epoch  6 | loss 0.4897 | valid GAUC 0.6658 nDCG@5 0.5354 primary 0.6006 | 9.7s
  epoch  7 | loss 0.4859 | valid GAUC 0.6671 nDCG@5 0.5358 primary 0.6015 | 10.1s
  epoch  8 | loss 0.4821 | valid GAUC 0.6665 nDCG@5 0.5359 primary 0.6012 | 9.9s
  epoch  9 | loss 0.4784 | valid GAUC 0.6666 nDCG@5 0.5348 primary 0.6007 | 9.4s
  epoch 10 | loss 0.4744 | valid GAUC 0.6650 nDCG@5 0.5342 primary 0.5996 | 11.8s
  epoch 11 | loss 0.4705 | valid GAUC 0.6640 nDCG@5 0.5341 primary 0.5990 | 12.9s
  early stop at epoch 11
=== fm (seed=0) ===
  valid  GAUC 0.6671 | nDCG@5 0.5358 | primary 0.6015
  [test line redacted -- see section 4]
```

### Against `baseline_scores.json` validation column

| | Published | Measured | Delta |
|---|---|---|---|
| GAUC | 0.6674 | 0.6671 | −0.0003 |
| nDCG@5 | 0.5357 | 0.5358 | +0.0001 |
| **primary** | **0.6016** | **0.6015** | **−0.0001** |

PASS. Well within the 0.001 tolerance.

### Observations worth carrying forward

- Wall clock: ~110 s total, ~10 s per epoch on this machine. Slower than the 40 s the
  README quotes, which is fine; budget ~2 min per FM training run when sizing the
  agent's per-iteration timeout.
- **Best epoch is 7, not 11.** Training loss keeps falling (0.6391 → 0.4705) while
  validation primary peaks at epoch 7 and then declines. Classic overfitting. The
  early-stop patience of 4 caught it correctly and the epoch-7 weights were restored.
- The validation curve is noisy: 0.6010, 0.6006, 0.6015, 0.6012, 0.6007. Swings of
  ~0.0009 between adjacent epochs. **This is the practical noise floor for a single FM
  run and it is very close to the 0.0008 seed std the organisers publish.** Any claimed
  gain under ~0.002 needs multiple seeds before it is believed.

---

## 7. Sanity rung: random scoring — PASS

```
python baseline.py --model random --data_dir C:\Users\Lenovo\track2-techjam\data\KuaiRand-Pure\data
```

```
=== random (seed=0) ===
  valid  GAUC 0.4990 | nDCG@5 0.4663 | primary 0.4827
  [test line redacted]
```

| | Published (mean of seeds 0–4) | Measured (seed 0) | Delta |
|---|---|---|---|
| primary | 0.4834 | 0.4827 | −0.0007 |

PASS. The published figure is a 5-seed mean; we ran one seed, so a delta of this size
is expected. **The contract test should use a tolerance of 0.002 for `--model random`,
not 0.001**, or it should average 5 seeds to match the published methodology.

The kit README states that a random primary far from ~0.483 means the harness is
broken. It is not broken.

---

## 8. Ablation reproduction — PASS, and it matters

```
python ablation_features.py C:\Users\Lenovo\track2-techjam\data\KuaiRand-Pure\data
```

```
{'train': 1141112, 'valid': 124909, 'test': 170588}
5 domains (current kit)   ( 5) | test GAUC 0.6614 | nDCG@5 0.5285 | primary 0.5950 +/- 0.0003
+4 item-side = 9 domains  ( 8) | test GAUC 0.6598 | nDCG@5 0.5281 | primary 0.5940 +/- 0.0004
CWM all 13 domains        (13) | test GAUC 0.6601 | nDCG@5 0.5280 | primary 0.5940 +/- 0.0005
```

Runtime: roughly 15 minutes (3 configs x 3 seeds x ~1 min).

Note: this script reports **test** scores, not validation. It is an organiser-provided
reference, run once by a human. The agent must never run it or see its output.

| Config | Fields | primary | vs 5-field |
|---|---|---|---|
| current kit | 5 | 0.5950 | — |
| + `music_id`, `video_type`, `upload_type` | 8 | 0.5940 | −0.0010 |
| + 6 user-side coarse buckets (CWM full) | 13 | 0.5940 | −0.0010 |

**Adding static features makes it very slightly worse.** The gaps are the same size as
the seed variance, so the honest reading is: no difference.

Reproduced independently. The organisers' claim holds.

### Consequence for the build

Do not spend iterations adding static feature fields. The `user_id x video_id` cross
already absorbs the signal those fields carry. This is written into
`knowledge/methods.md` so the agent does not propose it either.

---

## 9. Convergence rule — not implemented in the kit

`baseline_scores.json` supplies `{"epsilon": 0.002, "N": 3}` and the README describes
the rule in prose, but no code implements it. It is ours to build in
`harness/convergence.py`.

Q3 stays open: the exact comparison semantics are unspecified.

---

## 10. Tie handling in `evaluate.py` — RESOLVED (Q6)

`auc()` is Mann-Whitney U with proper average-rank tie correction. Runs of equal scores
receive the mean of their rank positions. `evaluate()` sorts each user's list by
`-score` before computing nDCG. Deterministic and correct. Emitting ties is safe.

---

## 11. `evaluate.py` conventions, all confirmed in source

| Convention | Confirmed |
|---|---|
| GAUC counts only users with `0 < npos < len(labs)` | Yes |
| GAUC weights by each user's positive count | Yes: `gnum += npos * auc(...)`, `gden += npos` |
| nDCG@5 includes zero-positive users, scored 0 | Yes: `idcg == 0` returns 0.0 and is appended |
| nDCG gain = `2^rel - 1` | Yes |
| Grouping key is `user_id` alone | Yes |
| primary = `(gauc + ndcg) / 2` | Yes |

`evaluate(user_ids, labels, scores, k=5)` is fully model-agnostic. Any model can be
scored by handing it three equal-length arrays.

---

## 12. `submit.py` behaviour

Three mutually exclusive modes: `--make`, `--check`, `--score`. All accept `--data_dir`
and `--split {valid,test}`.

`read_submission` validates in order: header exactly `row_id,user_id,video_id,score`;
four fields per row; `row_id` strictly `0,1,2,...`; row count not exceeding the split;
`user_id` and `video_id` matching the evaluation row positionally; score parseable as
float; score not NaN or Inf; total row count equal to the split.

`--score` works on `valid` and prints GAUC, nDCG@5, primary. Use it in the contract test
as an independent check of our own scoring path.

Corruption tests — TODO, to be built as `tests/test_submission.py`:

| Corruption | Rejected |
|---|---|
| wrong header | |
| row-count mismatch | |
| `row_id` gap | |
| misalignment against the eval split | |
| NaN score | |

---

## 13. Reference scoreboard, for local use

**Validation is the only column we work against.**

| | valid primary |
|---|---|
| random | 0.4827 (measured) / 0.4834 (published) |
| item popularity | 0.5807 (published, not yet run) |
| **FM official baseline** | **0.6015 (measured) / 0.6016 (published)** |
| oracle ceiling | 0.8484 |

Target to beat: **0.6015**. Headroom to the ceiling: 0.247.

---

## 14. Environment

| | |
|---|---|
| OS | Windows |
| Shell | PowerShell |
| Python version | TODO — `python -V` |
| numpy version used for baseline | TODO — `pip show numpy` |
| CPU / cores | TODO |
| RAM | TODO |
| FM training wall clock | ~110 s (11 epochs at ~10 s) |
| Ablation wall clock | ~15 min |

Did the baseline number change after installing pandas / LightGBM? TODO — re-run
`--model fm` after `pip install -r requirements.txt` and confirm 0.6015 still holds.

---

## 15. Setup status

| Step | Status |
|---|---|
| Data downloaded and extracted | done |
| Split row counts verified | done |
| FM baseline reproduced | done, 0.6015 |
| Random sanity rung | done, 0.4827 |
| Ablation reproduced | done, no gain from static features |
| Item popularity rung | not run (optional) |
| Full dependency install and re-verify | TODO |
| Repo committed and pushed | TODO |
| Open questions posted to organisers | TODO |

Setup verification is complete. The build can begin.
