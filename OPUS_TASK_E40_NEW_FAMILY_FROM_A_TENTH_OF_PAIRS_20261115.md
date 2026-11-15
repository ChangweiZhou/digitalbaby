# E40: does a learner that has lived generalise a new family from a tenth of its pairs, where a fresh one memorises? (15 November 2026, revision 1)

Evidence level E2. The question and design are the outside reviewer's (its memo, revision 6, section 19.4, on `claude/vibrant-cerf-sje726` at e41d41f3); this specification and the predictions are mine. **Not released: it waits for the reviewer's audit.** One worker, best the E36 worker, whose runner already has the three update paths.

Shell names: `D=origin/codex/minifly-consolidated-20261023; M=MOD97_LOCAL_RULE_20261106`. Host: Linux x86-64, Python 3.13.18, torch 2.15.0+cpu; at most three processes, one thread each; launch outside any wrapper with a time limit. Edit nothing existing.

## Why

Two results stand beside each other and are not like for like.

- **A fresh GRU taught a tenth of the pairs memorises.** E35's arm `G` (backpropagation, mod 89, f = 0.10, a full birth of 24,576 updates, seed 73129): 98% on taught pairs and 19.6% on held-out add and sub at 12 letters. A hand-wired circuit did not change that (E35-B: 19.9 with the module, 18.0 forced through it).
- **A learner with a life behind it learns a new family from few items.** E37: at 131,000 items of a family taught at a quarter of its pairs, lived learners are at 77 to 94 and fresh ones at 7 to 19.

E40 puts the two under one condition: a family no network has met, a tenth of its pairs taught, lived learners beside fresh ones.

## The family

Four tasks at modulus 83, with code bytes no earlier family uses:

```
F4 = [("add", 83, b"e"), ("sub", 83, b"f"), ("max", 83, b"g"), ("min", 83, b"h")]
```

Earlier codes are `p q v w` (birth, mod 89), `r t x n` (F1, mod 97), `s u y z` (F3, mod 73) and `c` (compositions) (`git show $D:$M/exp11_cell.py | sed -n '50,51p'`; `git show $D:$M/exp16_integrated.py | sed -n '546,547p'`). Assert that none of F4's codes or its modulus is among them. Items are rendered by `exp11_cell.render` as every family's are: operation symbols as before, 0 to 12 letters.

- **Taught pairs:** `exp11_cell.split(t, .10, seed)[0]`, 689 of 6,889 pairs per task.
- **Coverage (my count):** with this split every value from 0 to 82 appears in each operand position of each task at least once on all three seeds, the fewest being 1 to 4 times. So a held-out pair never contains a number the task has not shown in that position, but some have been shown there once.

## Runs: 15, of 8,192 updates each

| Kind | Start | Rule | Seeds |
|---|---|---|---|
| Lived S | `exp45/e26/e26_life_S_<seed>.pt` | the sweep | 3 |
| Lived B | `exp45/e26/e26_life_B_<seed>.pt` | backpropagation | 3 |
| Lived K | `exp45/e31/e31_life_K_<seed>.pt` and its `_Bhh.pt` | the sweep with Kolen–Pollack feedback, the step as E36's runner writes it (E36 brief, revision 2, "Arm K's code") | 3 |
| Fresh S, B | `arch26.new_net(seed)` | sweep; backpropagation | 3 each |

- **Update and optimizer:** as E36: `exp26_candidate.step`'s text with the rule's forward; a fresh `exp26_candidate.make_opt(m)` in every run.
- **Data, the same batches for every run of a seed:** batches of 64 consecutive items of `X16.entry_stream(entries, seed + 140)` with `entries = [("op", t, a, b) for t in F4 for a, b in exp11_cell.split(t, .10, seed)[0]]`. The generator seed `seed + 140 + 77` is used by no earlier stream (birth `seed + 77`; lives `seed + 177` to `seed + 183` and `seed + 79` to `seed + 84`; E36 `seed + 113`).
- **Rows:** score and save weights at 0, 512, 1,024, 2,048, 4,096 and 8,192 updates. Commit the weights at 8,192 (15 files); record the sha256 of the others.
- Row 8,192 is 524,288 items, 131,072 per task: each taught pair shown about 190 times.

## Scores at every row

1. **Held out:** `exp11_cell.panel(t, exp11_cell.split(t, .10, seed)[1], seed, ks=(12,), n=300)` for each task of F4.
2. **Taught:** the same call on the taught pairs with `seed + 1`, to see memorising.
3. **All held-out pairs without letters, split by wrap,** for add and sub: a + b ≥ 83, or a − b < 0.
4. **Earlier families at 12 letters,** from `exp16_integrated.panels(seed)[0]`: `add89`, `sub89`, `add97`, `sub97`, `add73`, `sub73`. Reported and not read.

The figure read is the mean of held-out add and sub at 12 letters (score 1). Max and min are tabled beside it.

## Checks before the full runs

1. Row 0 of each lived run has the life-end file's `tensors_sha256`.
2. The batch digest is equal across all runs of a seed at every row.
3. The update path against E26's dry births, as E36's check 4 (79618ab2… under the sweep, adf933d7… under backpropagation).
4. Arm K, as E36's check 3.
5. No presented item has a (task, a, b) that is a held-out pair; no code of F4 appears in any earlier family (asserted).
6. One dry run of 256 updates of each kind, with CPU time and a forecast (the reviewer's estimate: about 4 CPU-hours).

Each runner calls `arch26.install()` first and refuses an existing output path. Outputs go to `exp45/e40/`, named `e40_<lived|fresh>_<rule>_<seed>` with `_row<updates>` for saved rows.

## Reading (fixed)

Lived minus fresh at the same row, which is the same number of items of the family, under the same rule; lived K against fresh sweep. **The verdict is at row 8,192**, per rule, in at least two of three seeds: **saving** if lived is at least 15 points above; **cost** if at least 15 below; **none** if less than 15 either way; otherwise mixed. The same differences at rows 2,048 and 4,096 are tabled beside it and not read.

Reported beside it, as facts and not marks:

- each run's held-out score at the first row where its taught score passes 90, or "taught never passed 90";
- the lived runs' held-out level at row 8,192 beside E35's fresh figure of 19.6 (another modulus, a full birth, one seed);
- the wrap and no-wrap scores at row 8,192;
- the fall of the earlier families from row 0.

## What it cannot show

- One new family, at one modulus.
- Lives made for another purpose; a fresh optimizer in every run; the family is taught alone and not inside a life's stream.
- A lived learner has two other moduli behind it and the fresh one none, so a saving does not say which part of the life gives it.
- The fresh runs here have a third of a birth's updates. E35's arm `G` is the fresh reference at a full birth, on mod 89 and one seed.

## Predictions (mine)

| | Prediction | Chance |
|---|---|---|
| P40.1 | Every fresh run is at or below 30 on held-out add and sub at every row | 80% |
| P40.2 | Lived sweep runs are at or above 60 at row 8,192 in two of three seeds | 45% |
| P40.3 | A saving under each of the three rules | 70% |
| P40.4 | Lived runs pass 90 on taught pairs at an earlier row than their fresh references, in two of three seeds per rule | 65% |
| P40.5 | A verdict that differs between the sweep and backpropagation | 20% |
| P40.6 | At row 8,192 the lived sweep runs' wrap pairs are at least 20 points below their no-wrap pairs in two of three seeds (the new modulus is the hard part) | 50% |

## Done

Runner, check receipts, fifteen runs with receipts, and `E40_REPORT.md` with the curves (run × row: the four score groups), the differences and verdict per rule, the facts listed, and the outcome of each prediction. No readings beyond the above.
