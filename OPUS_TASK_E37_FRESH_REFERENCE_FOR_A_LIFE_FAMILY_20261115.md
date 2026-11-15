# E37 (job C): does a learner that has lived need fewer items for a new family than a fresh one? (15 November 2026, revision 2)

Evidence level E2. The question and design are the outside reviewer's (its memo at a8b5ace2 on `claude/vibrant-cerf-sje726`, section 6); this specification and the predictions are mine. Revision 1 (4dd0b9f7) was audited by the reviewer (`AUDIT_E36_E37_E38_BRIEFS_20261115.md` on `claude/vibrant-cerf-sje726` at 823a3884), who stated that with its required changes made as written no re-audit is needed. Revision 2 makes them and the recommendations; the last section lists them. **Released.** One worker.

Shell names: `D=origin/codex/minifly-consolidated-20261023; M=MOD97_LOCAL_RULE_20261106`. Host: Linux x86-64, Python 3.13.18, torch 2.15.0+cpu; at most three processes, one thread each; launch outside any wrapper with a time limit. Edit nothing existing; do not call or patch existing `dev`, `life` or `step` functions except where a line below names one to call.

## Why

In the nine saved lives, later families reach high held-out scores from a quarter of their pairs in few items: mod 73 reaches 77 to 94 after about 131,000 of its items, where a birth from nothing on half the pairs of mod 89 is at 63 to 84 after 262,000 (the reviewer's read of the receipts). That is not yet a saving, because no fresh network has been taught a quarter of the pairs of one family. E37 supplies that reference with the very items the lives saw.

## Runs: fresh networks on one life family alone

A fresh `arch26.new_net(seed)`, the sweep (`local=True`) and backpropagation (`local=False`), update as `exp26_candidate.step` (lines 84-91), optimizer `exp26_candidate.make_opt(m)`, three seeds.

| Part | Family | Stream | Updates | Rows saved and scored | Runs |
|---|---|---|---|---|---|
| 1 | F3 (mod 73) | `X16.entry_stream(X16.family_entries("F3", seed), seed + 105)` | 4,096 | 1,024, 2,048, 4,096 | 6 |
| 2 | F1 (mod 97) | `X16.entry_stream(X16.family_entries("F1", seed), seed + 101)` | 6,144 | 2,048, 4,096, 6,144 | 6 |

`X16` is `exp16_integrated`. Batches are 64 consecutive items of the stream. These are the lives' own streams (`exp26_candidate.life`, line 243: `X16.entry_stream(entries[fam], seed + 100 + si)`, with `si` = 5 for F3 and 1 for F1), so a fresh network sees each family's items in the order the life did.

- **Part 2 is batch-identical to a life's first block.** In block 1 the joint stream has one family, so the life's 6,144 batches are exactly these. Check 1 uses that.
- **Part 1 is item-identical, not batch-identical.** In blocks 5 and 6 a life draws each item from one of three families at random, so about a third of each batch is F3.

Commit the final weights of each run (12 files); record the other rows' sha256.

## Score

The life receipts' own panels, `exp16_integrated.panels(seed)[0]`, at k = 12 by `exp16_integrated.score`: `add73`, `sub73` (part 1) and `add97`, `sub97` (part 2). The figure is the mean of add and sub; max and min are tabled beside it and not read.

## Checks before the full runs

1. **Part 2's batch digest equals the life's.** After 6,144 updates the `exp26_candidate.Digest` of the batches equals `stages[1].data_digest` of `exp45/e26/e26_life_S_<seed>.json`.
2. **Part 1's items are the life's.** The count of F3 items a life saw in a block is in its receipt (`stages[5].exposure_rows_tokens["joint/F3"][0]` for block 5, `stages[6]`'s for block 6 alone). Report block 5's count beside 131,072 and the sum of blocks 5 and 6 beside 262,144, the items at rows 2,048 and 4,096. (The audit's rebuild gives 131,079, 131,131, 130,950 and 262,120, 261,736, 261,821.)
3. **The update path against E26's dry births.** From `arch26.new_net(73129)` with `make_opt`, 2,048 updates on the birth's own stream (`exp11_cell.batches([(t, exp11_cell.split(t, .5, 73129)[0]) for t in exp11_cell.DEV], 73129, 12)`) give the `tensors_sha256` of `exp45/e26_qual/e26_dev_S_dry2048_73129.json` (79618ab2…) under the sweep and of `e26_dev_B_dry2048_73129.json` (adf933d7…) under backpropagation.
4. One dry run of 256 updates per rule and part, with CPU time and a forecast (the reviewer's estimate: 0.9 CPU-hours for part 1, 1.4 for part 2).

## Reading (fixed; the reviewer's proposal, adopted)

Life minus fresh, at equal items of the family, under the same rule. The K lives are read against the fresh sweep network.

| Comparison | Life's score | Fresh network's row |
|---|---|---|
| F3 at about 131,000 items | after block 5 (`stages[5]`) | 2,048 |
| F3 at about 262,000 items | after block 6 (`stages[6]`) | 4,096 |
| F1 at 393,216 items | after block 1 (`stages[1]`) | 6,144 |

For each comparison and rule, in at least two of three seeds: **saving** if the life is at least 15 points above; **cost** if at least 15 below; **none** if less than 15 either way. Otherwise mixed.

The F1 comparison has a different meaning from the F3 ones and is read separately: the lived learner there has only its birth behind it, and the batches are identical, so it compares a born network with an unborn one on the same 6,144 updates.

Reported beside it: E35-A's arm `G` at f = 0.25 (a fresh GRU on a quarter of the pairs of mod 89 under backpropagation), its held-out add and sub at 12 letters at its 4,096-update row, when that branch has it.


## Common to every run (the audit's recommendations)

- Each runner calls `arch26.install()` first, as `dev` and `life` do; the scorer depends on it.
- Each runner refuses an existing output path (`exp20_reference.refuse_existing`).
- Outputs go to `exp45/e37/`, one receipt and its weight files per run, named `e37_<kind or arm>_<rule>_<seed>` with `_row<updates>` for saved rows.

## What it cannot show

- The lived learner had also seen about 1.6 million items at birth and 1.8 million of other families in its life. A saving is in items of the new family, not in total experience.
- In part 1 the fresh network gives every update to F3 and the life gave a third. That favours the fresh network per update and is neutral per item.
- It does not say what in the lived network gives the saving.

## Predictions (mine)

| | Prediction | Chance |
|---|---|---|
| P37.1 | F3 at 131,000 items: a saving under the sweep | 80% |
| P37.2 | F3 at 131,000 items: a saving under backpropagation | 80% |
| P37.3 | F3 at 262,000 items: a saving under each rule | 55% |
| P37.4 | F1 at 393,216 items: a saving under each rule | 65% |
| P37.5 | A reading that differs between the sweep and backpropagation | 15% |

## Done

Runner, check receipts, twelve runs with receipts, and `E37_REPORT.md` with the table (rule × seed × comparison: life, fresh, difference), the verdict per comparison and rule, and the outcome of each prediction. No readings beyond the table above.

## What changed from revision 1 (the audit)

| | Change | Required? |
|---|---|---|
| E37-1 | A check of the update path against E26's dry births; check 1 covered the data only | yes |
| — | Check 2 compares 262,144 with the sum of blocks 5 and 6; "none" is less than 15 either way; output paths, overwrite refusal, `arch26.install()` | recommended |

The audit rebuilt part 2's 6,144 batches and matched `stages[1].data_digest` in all nine life receipts, so the batch identity claimed above holds.
