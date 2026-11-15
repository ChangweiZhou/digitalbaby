# Audit of the E40 brief at 3aa9dbf3 (15 November 2026)

From the outside reviewer. Read-only: nothing was trained and no code from another branch was executed.

**Verdict.** The family, its code bytes, its modulus, the stream seed, the panels and the split are all sound, and every factual claim I could test holds. The brief cannot go as it stands for one reason: its verdict can be met with nothing learned. Before any update, the nine networks that have lived already score 15 to 52 on the figure the brief reads, where a fresh network scores 0. Two changes are required, both worded to be applied as written: the figure that is read, and two added checks. After them E40 can be released without a re-audit.

**Terms.**

- A *lived* network is one of the nine saved at the end of a life (E26 arms S and B, E31 arm K, three seeds). A *fresh* network is `arch26.new_net(seed)` before any training.
- A *row* is a number of updates at which a run is scored; row 0 is before the first update.
- *Held-out pairs* are the operand pairs of a task that are never taught.
- A *reduction pair* is an operand pair of add or sub whose answer needs the reduction at the modulus (here a + b ≥ 83, or a − b < 0). The brief calls the split "by wrap".
- All scores are whole answers: two digits and the end symbol.

Shell names: `D` and `M` as in the brief; `R=PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109`; `C=$R/e40_brief_checks.py` on `claude/vibrant-cerf-sje726`. `python3 -I $C <letter>` runs one part.

## 1. Code bytes, modulus and stream seed

All unused. `python3 -I $C A B C`

- **Code bytes.** `e`, `f`, `g`, `h` are in no earlier family (`p q v w`, `r t x n`, `s u y z`, `c`) and are not digits, operation symbols, `=`, `;` or upper-case letters. The brief's two line references are right (`git show $D:$M/exp11_cell.py | sed -n '50,51p'`; `git show $D:$M/exp16_integrated.py | sed -n '546,547p'`).
- **Modulus.** 83 is not among 73, 89 and 97.
- **Split.** `exp11_cell.split` seeds its generator with `seed*1000 + round(frac*100) + m*7 + code[0]`. For the new family that is `seed*1000 + 692` to `695`, shared with no earlier split (birth 785 to 792, F1 814 to 824, F3 651 to 658, E35's splits of the birth family at 0.10 and 0.25). `round(.10 * 100)` is 10 and `round(.10 * 6889)` is 689.
- **Coverage.** The coordinator's count is right: every value from 0 to 82 appears in each operand position of each task on all three seeds, the fewest appearances being 1 to 4.
- **Stream.** `X16.entry_stream(entries, seed + 140)` seeds its generator with `seed + 217`. No earlier stream, joint stream or panel of any of the three seeds uses that value (birth `seed + 77`; E36 `seed + 113`; E38's rehearsal `seed + 177`; the lives' families `seed + 178`, `180`, `182`; their joint streams `seed + 79` to `84`; panels `seed + 3`).
- **The new bytes' embeddings.** In the saved files the embedding rows of `e`, `f`, `g`, `h` are bit-equal in the birth and the life-end files of a seed, under both rules and Kolen–Pollack feedback. So are 201 of the 256 byte rows, which is every byte no family has used (256 less 55). No training has touched them, so a lived and a fresh network of a seed start with the same embeddings for the new codes. Section 4 turns this into a check.

## 2. Panels and the split by reduction

Right as written.

- **Held-out panel.** `exp11_cell.panel(t, split(t, .10, seed)[1], seed, ks=(12,), n=300)` is the call the life receipts use for the birth family (`git show $D:$M/exp16_integrated.py | sed -n '710p'`), on 6,200 held-out pairs per task.
- **Taught panel.** The same call on the 689 taught pairs with `seed + 1` draws 300 of them with fresh letters; its generator (`seed + 4`) is used by nothing else.
- **Split.** Of the held-out pairs of add and sub, 49.3 to 49.6% are reduction pairs, about 3,070 per task. The two conditions in the brief are the right ones for modulus 83.
- **Earlier families.** `exp16_integrated.panels(seed)[0]` holds all six named panels at 12 letters.
- My reader builds scores 1, 2 and 3 from the brief's text and gives the row-0 table of section 3. `python3 -I $C D`

## 3. The reading

**3.1 The verdict can be met with nothing learned.** `python3 -I $C D E`

Row 0 of the nine lived runs on the new family, before any update (%):

| Lived run | Held-out at 12 letters: add, sub, max, min | Taught at 12 letters: add, sub, max, min | Held-out, no letters: add without / with reduction | sub without / with reduction |
|---|---|---|---|---|
| S 73129 | 17.0, 29.7, 52.3, 34.0 | 19.3, 29.0, 48.7, 36.0 | 16.7 / 12.2 | 0.0 / 0.0 |
| S 93683 | 11.3, 19.3, 31.0, 63.3 | 7.7, 23.3, 24.3, 60.0 | 12.5 / 3.9 | 2.9 / 0.4 |
| S 116089 | 46.7, 16.0, 85.3, 99.0 | 45.7, 14.3, 86.7, 97.0 | 88.8 / 0.0 | 1.6 / 0.3 |
| B 73129 | 37.3, 48.3, 41.3, 42.3 | 31.7, 50.3, 43.3, 38.3 | 72.6 / 0.0 | 52.9 / 8.8 |
| B 93683 | 13.3, 47.0, 95.0, 97.3 | 9.7, 50.0, 93.0, 98.3 | 1.0 / 0.8 | 97.6 / 0.1 |
| B 116089 | 46.0, 57.3, 95.0, 99.3 | 46.0, 60.7, 96.0, 99.0 | 89.8 / 0.1 | 92.1 / 23.1 |
| K 73129 | 30.0, 42.3, 53.0, 37.0 | 33.0, 48.0, 48.3, 40.0 | 57.0 / 14.2 | 14.5 / 3.1 |
| K 93683 | 10.3, 36.7, 79.7, 81.7 | 14.3, 42.3, 78.7, 85.3 | 6.0 / 4.0 | 48.0 / 1.4 |
| K 116089 | 55.0, 26.7, 26.0, 94.3 | 55.0, 26.0, 30.7, 95.0 | 35.8 / 14.3 | 0.0 / 0.0 |

- The brief reads the mean of held-out add and sub at 12 letters. At row 0 that mean is 23.3, 15.3, 31.3 for the sweep's lived runs, 42.8, 30.2, 51.7 for backpropagation's and 36.2, 23.5, 40.8 for Kolen–Pollack's. A fresh network is at 0.
- So "lived at least 15 points above fresh" holds at row 0 in three of three seeds under every rule. A lived run that learned nothing from the tenth and kept its row-0 level would be read as a saving against a fresh run that ends near 18.
- The cause is the one that undid two of my earlier marks: the operation symbol is in the prompt, so a network that has the shared arithmetic answers many pairs that need no reduction under a code it has never seen.
- On reduction pairs the lived networks are near the floor at row 0: the mean of add and sub is 0.1 to 11.6. What is right there after training can only have come from the tenth.
- A fresh network that has memorised a tenth scores alike on both kinds of pair: E35-A's arm `G` at f = 0.10 (mod 89, seed 73129) is at 18 to 22 on held-out pairs without the reduction and 17 to 19 with it, at 96 to 98 on taught pairs. So the reduction score costs the fresh runs nothing. `python3 -I $C F`

**Required, R1.** Three replacements.

In "Scores at every row", replace the last two sentences ("The figure read is … beside it.") by:

> The figure read is the **reduction score**: the mean, over add and sub, of the share of whole answers right on the held-out pairs that need the reduction at 83 (score 3; about 3,070 pairs per task). A run's **gain** at a row is its reduction score at that row minus its own reduction score at row 0. The mean of held-out add and sub at 12 letters (score 1) is tabled at every row as the level reached, with max and min beside it, and is not read.

In "Reading (fixed)", replace the first paragraph by:

> Lived against fresh at the same row, which is the same number of items of the family, under the same rule; lived K against fresh sweep. **The verdict is at row 8,192**, per rule, in at least two of three seeds: **saving** if the lived run's gain is at least 15 points above the fresh run's gain; **cost** if at least 15 below; **none** if less than 15 either way; otherwise mixed. The same differences at rows 2,048 and 4,096 are tabled beside it and not read.
>
> The verdict is not taken on the held-out mean at 12 letters, because the lived networks answer part of the new family before any update: at row 0 they score 15 to 52 on that mean (the reviewer's figures, in its audit) where a fresh network scores 0, so "15 points above fresh" would hold with nothing learned. On the reduction pairs they score 0 to 12 at row 0.

In the list "Reported beside it, as facts and not marks", add:

> - for every run, the held-out mean at 12 letters at row 0 and at row 8,192, and the same for the held-out pairs that need no reduction (score 3);
> - each run's highest reduction score over the rows, and the row at which it occurs.

P40.3 is then a statement about the gain. P40.2 is a statement about the level and stays; it should be read beside the sweep's row-0 level of 15 to 31.

**3.2 Row 8,192 is an acceptable place for the verdict.**

- E35-A's fresh network at a tenth has memorised by 4,096 updates (taught 95 to 97, held-out 12 to 17) and does not move afterwards. In E40 a taught pair has been shown about 95 times by row 4,096 and 190 by row 8,192, so the last row is past memorising for the fresh runs. `git show origin/claude/nice-rubin-y5de8k:$M/exp45/e35a/e35a_dev_G_f10_73129.json`
- The three outcomes are complete and exclusive per seed, with exactly 15 counted as a saving or a cost.
- One row can miss a rise followed by a fall. The added fact (each run's highest reduction score and its row) covers that.

**Recommended.** Score and save rows 64, 128 and 256 as well. In E36 the modulus moved within 64 to 128 updates; here the first row after 0 is 512.

## 4. The checks

Reusing E36's checks is sound for what they cover: the lived start (row-0 hash), the same batches for every run of a seed, the two update paths against E26's dry births, and the Kolen–Pollack step. Check 5 covers the new entries. Three things are new in E40 and are not covered:

- **The fresh start on seeds 93683 and 116089.** The dry-birth check uses seed 73129 only.
- **The new panels and the split by reduction,** which are new scoring code with one implementation.
- **That lived and fresh runs meet the new codes on equal terms.**

**Required, R2.** Add to "Checks before the full runs":

> 7. **The fresh start and the new code bytes.** At row 0, the rows of `book.weight` for the bytes `e`, `f`, `g` and `h` are bit-equal in all five networks of a seed (three lived, two fresh), and every score of a fresh run is 0. (In the saved files these four rows are bit-equal in the birth and life-end files of a seed, as are 201 of the 256 byte rows: no training has touched them.)
> 8. **Row 0 against the reviewer's reader.** For each of the nine lived runs, scores 1, 2 and 3 at row 0 equal the reviewer's figures to one decimal (the table in its audit; `python3 -I PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/e40_brief_checks.py D` on `claude/vibrant-cerf-sje726`). This stands in for a second implementation of the new panels and of the split by reduction.

## 5. Other points (recommended)

- **Forecast.** At E36's measured 0.104 to 0.117 CPU-seconds per update and about 13 to 14 per scored row, fifteen runs are 3.6 to 4.0 CPU-hours of updates and about 0.4 of scoring: 4.0 to 4.5 in all, with the three extra rows.
- **"What it cannot show."** Add: "Part of a lived learner's level on the new family is there before any update; the verdict excludes it and the level does not."
- **Outputs, refusal to overwrite, `arch26.install()`, predictions on file:** all present.
- **The runner.** `e36_reminder.py` on the default branch already has the three update paths, the start from a life-end file and the fresh start (`git show $D:$M/e36_reminder.py | sed -n '88,93p;116,133p'`).

## 6. Can E40 be released?

Not as it stands. After R1 and R2 as written, yes, without a re-audit. Check 8 is what stands in for one: if the worker's row 0 gives the table of section 3, the panels, the split and the scorer are the ones audited here.

| | Change | Where |
|---|---|---|
| R1 | The figure read is the gain in the reduction score; the level is tabled with its row-0 value | "Scores at every row"; "Reading (fixed)"; the list of facts |
| R2 | Two checks: the fresh start and the new code bytes; row 0 against the reviewer's reader | "Checks before the full runs" |

## 7. E39, revision 2

Confirmed, not re-audited.

- R1 to R7 are in the brief as written, and its closing table lists them. R8 concerned E34's addendum 2, which now carries the sentence.
- The specification file has the harness's symbols: `EOS, BOS = 256, 257` (`git show $D:PROJECT_DOCUMENTS/COORD_CHECKS_20261115/sw_world_spec.py | sed -n '8p'`).
- The regenerated output file gives the six digests of the audit, and the brief's check 1 lists their first eight characters (`git show $D:PROJECT_DOCUMENTS/COORD_CHECKS_20261115/OUTPUT_SW_WORLD_MAC.txt | grep digest`). My implementation gives the same three for the first 16 visits again today (`python3 -I $R/e39_world.py spec stats`).
- **Two leftovers (recommended).**
  - Check 6 still says the buffer "never exceeds its capacity in bytes". Since R5 the capacity is in items: write "in items (14,033 at the default size)".
  - Line 71 lists "an anchor, a slow copy, a step size per weight, reminders, the confidence gain" as later candidates. The memo's revision 7 (section 20, 2759b8b3) withdrew that. Replace the sentence by: "Later arms are added by a one-page roster note; the runner's update is the only place they touch. Under the reviewer's memo, revision 7, section 20, no reminder schedule, slow copy, weighted anchor or second store is a candidate until L2 shows a deficit that the plain store does not cover."

## 8. E35-B's addendum 1 and review 9.136

- **Addendum 1** is consistent with the memo: the forced arm on two more seeds, the stop rule, and nothing briefed after stage 1b.
- **Review 9.136 was written against memo revision 6** (e41d41f3) and carries one sentence that revision 7 withdrew: "So among the controls of the long-life programme, short reminders move ahead of anchors." Replace it by: "The reviewer first read this as putting short reminders ahead of anchors among the long-life programme's controls, and withdrew that in its memo's revision 7 (section 20): where a store is used it is the plain bounded store of raw items, and no retention mechanism is a candidate until L2 shows a deficit that the plain store does not cover."
- Revision 7 also marks item 2 of the memo's section 19.3 as withdrawn and adds a rule for any added mechanism; the review does not yet record either.

## 9. What I did not check

- Anything that needs torch: the fresh networks themselves, the update paths, the cost.
- That `arch26.new_net(seed)` gives the births' starting values on seeds 93683 and 116089; check 7 tests it.
- The predictions, beyond the two that R1 touches.
- E36's and E37's merged reports beyond what I recomputed earlier today; E35-B's figures, which are the worker's.
- The forecast is scaled from E36's receipts and is approximate.
