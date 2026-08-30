# Results — second contributor

**This file belongs to one person. Nobody else edits it.**

`docs/RESULTS.md` is the master document and is owned by contributor A. Recording
findings here instead means two people can work at the same time without ever
touching the same file, so git has nothing to conflict over.

At the end, A folds anything that matters into `RESULTS.md` §1 and §8. That merge is
a two-minute copy, not a negotiation.

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

## My runs

Add a row per run. Directory name goes in the first column so anyone can find the
artefacts.

| run directory | iterations | best primary | vs 0.6036 | what it tried |
|---|---|---|---|---|
| `runs/<name>-1` | | | | |

---

## Individual experiments worth noting

Anything interesting, including failures. A failure that explains *why* is worth more
than a success nobody can account for.

### `runs/<name>-1`, iteration N — <what it was>

- **Hypothesis:** what the agent believed was wrong
- **Result:** validation primary, and which metric moved (GAUC vs nDCG@5)
- **Read:** what it means, honestly

---

## If something clears the bar

1. **Confirm it.** Re-run the same configuration with two more independent seed sets.
   One measurement is not a result — that discipline is what made 0.6036 believable.

   ```python
   from harness import data as d
   from harness.models import runners as R
   sp = d.load()
   for seeds in [(2,4,8,16,32), (3,5,7,13,17)]:
       r = R.train_ensemble(sp, seeds=seeds, ...)   # your winning config
       print(seeds, round(r.val_primary, 4))
   ```

2. **Force-add the submission.** The ignore rules exclude them deliberately:

   ```bash
   git add -f runs/<name>-N/submission.csv
   ```

3. **Record the numbers here**, then tell A to update `RESULTS.md` §1 and §8. Do not
   edit `RESULTS.md` yourself — that is the file that would conflict.

---

## Working without conflicts

| file | owner |
|---|---|
| `runs/<name>-*/` | you |
| `docs/RESULTS_teammate.md` | you |
| `README.md` and the writeups | you |
| `docs/RESULTS.md`, `docs/TODO.md` | A |
| `harness/`, `agent/`, `tests/` | A |

Two rules, and conflicts become impossible rather than unlikely:

- **Never edit a file in the other person's column.** If you need a change there, say
  so rather than making it.
- **Pull before you push.**

  ```bash
  git pull --rebase origin main
  git push origin main
  ```

  `--rebase` replays your commits on top of theirs instead of creating a merge
  commit. Since you are working on different files, it applies cleanly every time.

If a pull ever does report a conflict, it means the ownership split was crossed
somewhere. Stop and work out where rather than resolving it by hand.
