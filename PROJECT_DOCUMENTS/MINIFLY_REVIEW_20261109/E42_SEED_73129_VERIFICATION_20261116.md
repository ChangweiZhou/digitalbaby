# E42, seed 73129: verification of the worker's results (16 November 2026)

From the outside reviewer. This is the reader check that the brief's check 8 asks of me, and a recomputation of the worker's tables for the one seed that is complete.

**What was read.** The worker's branch `claude/gallant-dijkstra-i9zgsr` at `7fbdbdc2` (stage 0 at `4c9e52d1`): `exp45/e42/E42_REPORT_73129.md`, the fifteen receipts, the fifteen full weights files and their small files, the stage-0 report and check receipts, and the three dry runs of check 8 with their recorded answers. The brief is revision 2 (`07943077` on the default branch).

**How.** Read-only. Nothing was trained, torch was not used, and no code of the worker's or the default branch was run. Files were read with `git show`, one at a time. The forward is my own `e42_cells_reader.py`, written from the brief's text, in the form of revision 2. The panels are rebuilt with `np_gru.py`. Every figure below comes from

    python3 -I PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/e42_seed73129_verify.py [A B C D E F G H]

run from the repository root; the part that gives each figure is named in the section.

**One seed.** Everything here is seed 73129. Seeds 93683 and 116089 are still running, and the brief's descriptions are fixed for a cell of three seeds. Section 5 gives what the fixed rules say for this seed alone; it is not the result of E42.

## 1. The runs, the checks, theta and the cost (the worker's records)

- **Fifteen runs, all complete, 24,576 updates each.** Stage 1, under the sweep and under backpropagation: `G`, `F10`, `A10`, `P8`. Stage 2, under the sweep only: `N5`, `N20`, `W10`, `P4`, `P16`, `P8-last`, `P8-source`. `P8-mean` is deferred, as the audit asked.
- **Stage 0.** The receipts of checks 1, 2, 3, 6, 7, 8, 9, 10 and 12 each record `pass: true`; checks 4 and 5 are assertions inside the runner and inside check 9; check 11 is reported from the full runs. Check 3 passed at a second attempt after two faults in the worker's own check code, and the failed receipt is kept. Check 1: `G` is bit-equal to E26 at the start and after 2,048 updates under both rules. Check 11: `G`'s final `tensors_sha256` equals E26's under both rules. Check 12: the scorer reproduces E41's counts and E30's on the six E26 births.
- **theta = 0.5**, chosen by check 10 as the brief's rule requires: mean share of connections replaced per event over updates 768 to 4,096 was 0.06% at 0.1, 0.67% at 0.25 and 7.99% at 0.5, and 0.5 is nearest 5%.
- **The stop** did not apply: `F10`, `A10` and `P8` reach 50 on A under both rules (rows 8,192 and 5,632 for `F10`, 8,192 and 5,888 for `A10`, 5,376 and 4,864 for `P8`, the sweep first).
- **Cost.** 15.22 CPU-hours for the fifteen runs (updates 12.26, scoring 2.89, replacement 0.07), against a forecast of 14.72 and a limit of 30.

I did not rerun any of these checks; they need torch.

## 2. The reader against the worker's scores

The reader reproduces every figure it was set against, with no difference of even one item.

**The dry runs of check 8 (part A).** The three weights files at 2,048 updates, the length-test panel of 300 items at 12 letters and 300 at 40, answers generated one byte at a time as the harness does:

| Dry run | Answers identical to the worker's, 12 letters | 40 letters |
|---|---|---|
| `F10` | 300 of 300 | 300 of 300 |
| `P8` | 300 of 300 | 300 of 300 |
| `P8-source` | 300 of 300 | 300 of 300 |

These networks are almost untrained (4 to 13 answers right of 300), so the agreement is on 1,800 mostly wrong answers, byte for byte. It covers the masked recurrent matrix, the periods and phases, the weights over the steps of a period, and revision 2's form of `source`.

**The final weights of the fifteen runs (parts B and G).** Answers right, reader / receipt's last row:

| Run | Length test, 12 letters | 40 letters | A: add, sub, max, min (of 300) | R: add, sub (of 1,000) |
|---|---|---|---|---|
| `G` S | 295 / 295 | 288 / 288 | 295, 293, 299, 295: equal | 962, 955: equal |
| `G` B | 295 / 295 | 293 / 293 | 295, 296, 300, 292: equal | 976, 987: equal |
| `F10` S | 275 / 275 | 271 / 271 | 275, 278, 296, 292: equal | 894, 875: equal |
| `F10` B | 274 / 274 | 274 / 274 | 274, 277, 294, 294: equal | 953, 913: equal |
| `A10` S | 284 / 284 | 284 / 284 | 284, 273, 295, 295: equal | 900, 823: equal |
| `A10` B | 279 / 279 | 277 / 277 | 279, 280, 296, 294: equal | 944, 913: equal |
| `P8` S | 288 / 288 | 288 / 288 | 288, 286, 297, 298: equal | 935, 915: equal |
| `P8` B | 292 / 292 | 289 / 289 | 292, 296, 295, 297: equal | 968, 964: equal |
| `N5` S | 273 / 273 | 267 / 267 | 273, 261, 292, 291: equal | 879, 794: equal |
| `N20` S | 271 / 271 | 264 / 264 | 271, 271, 295, 292: equal | 906, 816: equal |
| `W10` S | 268 / 268 | 266 / 266 | 268, 263, 297, 291: equal | 896, 801: equal |
| `P4` S | 289 / 289 | 282 / 282 | 289, 283, 296, 294: equal | 914, 933: equal |
| `P16` S | 284 / 284 | 251 / 251 | 284, 291, 299, 294: equal | 950, 956: equal |
| `P8-last` S | 274 / 274 | 277 / 277 | 274, 282, 299, 296: equal | 913, 912: equal |
| `P8-source` S | 286 / 286 | 285 / 285 | 286, 288, 297, 297: equal | 942, 946: equal |

**The saved structure (part C).** In all fifteen files: every cell has exactly N connections beside itself (255, 5, 10 or 20); every cell reads its own previous value; every connection is inside the cell's neighbourhood (for `W10`, among all other cells); every recurrent weight outside the mask is exactly zero, and in `P8-source` so is every entry of the added matrices outside the mask or outside the cell's period; the weights over the steps are exactly zero outside each cell's period.

## 3. The receipts recomputed (part D)

Thresholds in the three-row version (the first of three scored rows in a row at or above the mark). Weight-items are the receipt's count of weights taking part, times 64, times the updates. The last two columns before the replacements are means over the last three scored rows. Every figure equals the worker's table.

| Run | A80, updates | A80, weight-items | R80, updates | R80, weight-items | 12 letters | 40 letters | Replaced, total | Share replaced per event | Never replaced |
|---|---|---|---|---|---|---|---|---|---|
| `G` S | 5,632 | 4.05e11 | 6,656 | 4.79e11 | 98.44 | 96.44 | 0 | | 1.000 |
| `G` B | 5,120 | 3.68e11 | 6,400 | 4.60e11 | 96.89 | 94.56 | 0 | | 1.000 |
| `F10` S | 13,312 | 7.97e11 | 15,104 | 9.05e11 | 90.11 | 89.11 | 0 | | 1.000 |
| `F10` B | 8,704 | 5.21e11 | 9,472 | 5.67e11 | 92.89 | 92.56 | 0 | | 1.000 |
| `A10` S | 13,824 | 8.28e11 | 16,128 | 9.66e11 | 92.67 | 91.78 | 35,833 | 14.6% | 0.557 |
| `A10` B | 9,472 | 5.67e11 | 9,472 | 5.67e11 | 92.78 | 92.22 | 37,525 | 15.3% | 0.533 |
| `P8` S | 7,680 | 4.60e11 | 8,448 | 5.06e11 | 96.33 | 96.00 | 38,713 | 15.8% | 0.491 |
| `P8` B | 6,400 | 3.84e11 | 7,168 | 4.30e11 | 97.00 | 96.56 | 39,903 | 16.2% | 0.459 |
| `N5` S | 17,152 | 1.02e12 | 19,456 | 1.16e12 | 88.22 | 86.89 | 17,033 | 13.9% | 0.576 |
| `N20` S | 14,336 | 8.66e11 | 17,152 | 1.04e12 | 89.44 | 87.89 | 74,785 | 15.2% | 0.524 |
| `W10` S | 17,408 | 1.04e12 | 20,480 | 1.23e12 | 88.56 | 88.00 | 36,594 | 14.9% | 0.548 |
| `P4` S | 8,192 | 4.91e11 | 9,216 | 5.52e11 | 95.67 | 94.44 | 38,876 | 15.8% | 0.495 |
| `P16` S | 7,936 | 4.76e11 | 9,216 | 5.52e11 | 94.56 | 82.78 | 38,492 | 15.7% | 0.482 |
| `P8-last` S | 11,520 | 6.90e11 | 13,312 | 7.97e11 | 92.00 | 91.33 | 39,678 | 16.1% | 0.466 |
| `P8-source` S | 7,680 | 4.67e11 | 8,448 | 5.13e11 | 94.22 | 93.67 | 34,065 | 13.9% | 0.558 |

"Never replaced" is computed from the saved file (connections whose record of when they were made is zero; part C) and equals the worker's column.

## 4. Connections by period (part F)

For `P8`, the share of all connections by the reader's period (rows) and the source's period (columns), at the start from my own draws and at the end from the saved mask. The file's periods equal the draws. The table equals the worker's.

| `P8` under the sweep, start / end | Source 1 | Source 2 | Source 4 | Source 8 |
|---|---|---|---|---|
| Reader 1 | 0.248 / 0.261 | 0.128 / 0.110 | 0.063 / 0.066 | 0.061 / 0.063 |
| Reader 2 | 0.130 / 0.135 | 0.061 / 0.051 | 0.029 / 0.032 | 0.029 / 0.032 |
| Reader 4 | 0.061 / 0.068 | 0.031 / 0.027 | 0.016 / 0.015 | 0.017 / 0.014 |
| Reader 8 | 0.059 / 0.063 | 0.036 / 0.037 | 0.016 / 0.016 | 0.014 / 0.009 |

By the source's period alone: 0.498, 0.257, 0.124, 0.121 at the start and 0.527, 0.225, 0.129, 0.119 at the end under the sweep (0.518, 0.232, 0.123, 0.127 under backpropagation), where the cells are 0.500, 0.250, 0.125, 0.125. No share moves by more than 0.03.

## 5. What the brief's fixed descriptions give for this seed (part E)

One seed; a cell of the brief's table needs two of three. Updates: *fewer* below 2/3 of the reference, *more* above 1.5, otherwise *about the same*; 40 letters: a difference of 15 points or more. Every description below equals the worker's per-seed description.

| Comparison | A80 updates | A80 weight-items | R80 updates | R80 weight-items | 40 letters |
|---|---|---|---|---|---|
| `F10` against `G`, sweep | more | more | more | more | about the same (−7.3) |
| `A10` against `G`, sweep | more | more | more | more | about the same (−4.7) |
| `P8` against `G`, sweep | about the same | about the same | about the same | about the same | about the same (−0.4) |
| `F10` against `G`, backpropagation | more | about the same | about the same | about the same | about the same (−2.0) |
| `A10` against `G`, backpropagation | more | more | about the same | about the same | about the same (−2.3) |
| `P8` against `G`, backpropagation | about the same | about the same | about the same | about the same | about the same (+2.0) |
| `A10` against `F10`, either rule | about the same | about the same | about the same | about the same | about the same (+2.7, −0.3) |
| `P8` against `A10`, sweep | fewer | fewer | fewer | fewer | about the same (+4.2) |
| `P8` against `A10`, backpropagation | about the same | about the same | about the same | about the same | about the same (+4.3) |
| `N5`, `N20`, `W10` against `A10` | about the same | about the same | about the same | about the same | about the same (−4.9, −3.9, −3.8) |
| `P4` against `P8` | about the same | about the same | about the same | about the same | about the same (−1.6) |
| `P16` against `P8` | about the same | about the same | about the same | about the same | about the same (−13.2) |
| `P8-last` against `P8` | about the same | about the same | more | more | about the same (−4.7) |
| `P8-source` against `P8` | about the same | about the same | about the same | about the same | about the same (−2.3) |
| Sweep against backpropagation, `G` | about the same | about the same | about the same | about the same | about the same (+1.9) |
| Sweep against backpropagation, `F10` | more | more | more | more | about the same (−3.4) |
| Sweep against backpropagation, `A10` | about the same | about the same | more | more | about the same (−0.4) |
| Sweep against backpropagation, `P8` | about the same | about the same | about the same | about the same | about the same (−0.6) |

## 6. My expectations of memo section 22.8, on this seed

| | Expectation | Chance I gave | Seed 73129 |
|---|---|---|---|
| X1 | Check 1 passes under both rules | 85% | Passes, by the worker's receipt |
| X2 | `A10` reaches 80 on A within the dose under the sweep | 65% | Reached, at 13,824 |
| X3 | `A10` needs more updates than `G` to reach 80 on A under the sweep | 60% | More (13,824 against 5,632) |
| X4 | `A10` reaches 80 on A in no more updates than `F10` under the sweep | 55% | No: 13,824 against 13,312 (described as about the same) |
| X5 | `P8` against `A10`, updates to 80 on A under the sweep | fewer 20%, same 40%, more 40% | Fewer (7,680 against 13,824) |
| X6 | `P16` is lower than `A10` at 40 letters | 55% | No by the 15-point mark: 82.78 against 91.78 (−9.0) |
| X7 | `P4` is lower than `A10` at 40 letters | 25% | No: 94.44 against 91.78 |
| X8 | Some stage 1 configuration's description against `G` differs between the rules | 40% | Yes: `F10` (weight-items on A; updates and weight-items on R) and `A10` (R) |

**The period arithmetic** (longer periods: easier training, a harder length test).

- *Training.* Each set of periods reaches 80 on A in fewer updates than `A10` under the sweep (8,192, 7,680 and 7,936 for `P4`, `P8`, `P16` against 13,824; the last two comparisons are outside the brief's list). Among the three sets the description is about the same: there is no ordering by the longest period.
- *The length test.* `P16` is the lowest of all fifteen runs at 40 letters (82.78), while `P4` (94.44) and `P8` (96.00) are not below `A10` (91.78). The direction predicted appears for period 16 only, and its size, 13.2 points below `P8` and 9.0 below `A10`, is under the 15-point mark, so the fixed description is "about the same".

## 7. Observations outside the brief's descriptions

These are things seen in the files, on one seed. None is a reading under the brief.

1. **`P16`'s loss at 40 letters is steady, not one bad row.** Its last six rows at 40 letters are 86.3, 80.3, 84.3, 82.7, 82.0, 83.7; `P8`'s are 94.7 to 97.3 and `P4`'s 93.7 to 95.7 (part D). At 12 letters `P16` is at 94.56, so it loses 11.8 points between 12 and 40 letters where no other run loses more than 2.4.
2. **The replacement rate in the full runs is three times the rate theta was chosen for.** Check 10 chose theta = 0.5 for a rate nearest 5% over updates 768 to 4,096 (it measured 8.0%). Over a whole run the rate is 13.9 to 16.2% per event, and it rises throughout: in `A10` under the sweep from 190 connections per event in the first 4,096 updates to 471 of 2,560 in the last (part H). The brief's rule was followed; the rate it was tuned on is not the rate of the run.
3. **A connection made late almost never lasts, and about four slots in ten hold a connection with next to no weight.** At the end of every run with changing connections (part H): 46 to 58% of connections were never replaced, with a median strength 1.3 to 1.5 times the cell's mean; 35 to 41% were made at one of the last two events and have a median strength of zero (those made 256 updates before the end are at about a tenth of the cell's mean, and at most 0.2% of them are at half); fewer than 0.5% are between 512 and 1,024 updates old; and 6 to 13% are replaced connections that lasted, at 1.6 to 1.7 times the cell's mean. Of the replaced connections that lasted, most were made in the first 8,192 updates (in `A10` under the sweep, 93, 34, 17, 14, 9 and 8 by block of 4,096 updates). A new connection starts at zero and may be given up after 512 updates if it is below half the cell's mean; late in a run nearly all are.
4. **So a cell with changing connections works with about six of its ten.** Connections at or above half the cell's mean strength, per cell: 6.3 (`A10` S), 6.1 (`A10` B), 6.0 (`P8` S), 5.9 (`P8` B), 3.2 of 5 (`N5`), 12.4 of 20 (`N20`), 6.3 (`W10`), against 8.8 and 8.3 of 10 in `F10`. The comparisons of `A10` with `F10` and among `N5`, `A10` and `N20` are therefore between about 6 and 9, and among about 3, 6 and 12 working connections, not 10 and 10 and 5, 10 and 20.
5. **The pattern of who reads whom is not selected by period.** Section 4: no share of connections by reader's and source's period moves by more than 0.03 from the random start.
6. **Where the two rules part.** In `G` and in `P8` the sweep needs 1.04 to 1.20 times backpropagation's updates. In `F10` it needs 1.53 times on A and 1.59 on R, and in `A10` 1.46 and 1.70. The worker's cosine between the sweep's gradient and the exact one on the recurrent matrix falls over training in the limited runs (0.744 to 0.604 in `F10`, 0.754 to 0.482 in `N20`) and not in `G` (0.741 to 0.745). I did not recompute the cosines.

## 8. What I did not check

- **Anything that needs torch.** The training itself, the gradients, the equality of `G` with E26 under training (checks 1 and 11), the faster forward against the specification under both rules (check 9), the cosines, the CPU figures. They stand on the worker's receipts.
- **Rows before the last.** The reader was run on the final weights and on the three dry runs. The thresholds are recomputed from the scores in the receipts, not from weights at those rows, which are not committed.
- **`P8-last`, `P4`, `P16`, `N5`, `N20`, `W10` have no dry-run answers on file,** so for them the reader is checked on the final counts only (all equal).
- **The other two seeds.**
