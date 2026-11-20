# Line `nano/ahead2`: an independent repeat of the look-ahead result of `nano/sparse` (20 November 2026)

**Status.** Development level. One text (two-digit sums), ten seeds, one program of mine. No checker gate, seal or outside review is claimed. The task was the coordinating session's, under Mr. Xu's words of 20 November as that session quoted them (`PLAN_1.md`, head). The readings marked "mine" are this session's.

**What was done.** `PLAN_1.md` was committed (2caf1ac) before any program existed. `ahead.py` was then written from the prose of `nano/sparse/PLAN_1.md`, `PLAN_2.md`, `PLAN_5.md` and `PLAN_10.md`: a plain Python reference with the cells in dictionaries, and a compiled kernel with the cells addressed directly, which agree to 6e-15 on seven cases (`check_output.txt`). The text, the split and the form of the score are those of `loop_small_case/cells2.py`.

**That line's programs.** I did not open any `.py` or `.c` file of `nano/sparse` at any time, before or after my results: the figures agreed, so there was no difference to look for. I did print the first three lines of `nano/sparse/check4_output.txt` (text output of that line's check) while listing files, before my plan; it is disclosed in the plan.

## 1. The result

Never-shown additions, the system writing its own answer bytes, whole answer right, %. Mean of the last three readings (30,000, 35,000, 40,000 lines); each seed; median. Files `res/base_*`, `res/la_*`, `res/ctl_*`; tables `report_main.txt`, `report_2.txt`.

| Arm | reported seeds 73129, 93683, 116089, 406364, 545353 | seeds 2001 to 2005 (used by no line before) |
|---|---|---|
| without look-ahead | 79.1, 78.1, 77.3, 76.3, 77.2; **77.3** | 76.9, 78.1, 78.3, 81.0, 77.1; **78.1** |
| with look-ahead | 88.0, 91.3, 91.9, 92.7, 91.4; **91.4** | 90.4, 92.7, 91.3, 92.1, 90.3; **91.3** |
| control: cells keyed on the guess of the preceding line | 81.5, 79.6, 79.5, 77.5, 78.7; **79.5** | 77.8, 79.6, 79.3, 82.1, 78.1; **79.3** |
| as claimed by `nano/sparse` | 91.4 with (88.0 to 92.7), 77.4 without | none |

By place at the last reading (median; hundreds, tens, units): with look-ahead 99.2, 91.2, 100 and 99.0, 91.8, 100; without 98.4, 77.4, 100 and 98.0, 77.8, 100.

The life `add:40000,sub:20000`, nothing of addition shown during subtraction, last reading of each phase:

| Arm, seeds | before: each seed; median | after: each seed; median | after as % of before: each seed; median | subtraction at the end, clean score |
|---|---|---|---|---|
| with, reported | 88.0, 91.2, 92.6, 92.6, 91.0; **91.2** | 82.6, 89.0, 86.8, 89.0, 86.2; **86.8** | 93.9, 97.6, 93.7, 96.1, 94.7; **94.7** | 94.1, 88.5, 80.0, 89.4, 91.6; **89.4** |
| with, 2001 to 2005 | 89.8, 93.6, 91.8, 92.8, 91.0; **91.8** | 75.0, 86.8, 84.6, 88.4, 88.2; **86.8** | 83.5, 92.7, 92.2, 95.3, 96.9; **92.7** | 92.4, 91.1, 93.1, 90.1, 87.1; **91.1** |
| without, reported | 79.4, 79.0, 77.4, 76.0, 77.0; 77.4 | 78.4, 78.4, 78.2, 74.8, 76.6; 78.2 | 98.7, 99.2, 101.0, 98.4, 99.5; 99.2 | 82.2, 65.6, 77.5, 78.8, 73.5; 77.5 |
| without, 2001 to 2005 | 77.0, 78.8, 77.8, 81.8, 77.8; 77.8 | 76.0, 78.2, 76.6, 76.6, 77.4; 76.6 | 98.7, 99.2, 98.5, 93.6, 99.5; 98.7 | 83.8, 82.2, 84.5, 78.2, 81.7; 82.2 |

The clean score of subtraction is on the held-out pairs whose written line no shown pair writes (80 to 116 pairs a seed). On all 500 held-out pairs, which are mostly lines also shown: 93.6 and 92.8 with, 85.8 and 87.2 without.

The guesses on the 500 never-shown additions. At the first answer byte the guess three ahead (the units digit) is right on 100% on all ten seeds at every point measured. The guess two ahead (the tens digit) is right on 88.6% and 89.0% (medians of the two sets) at 40,000 lines and on 85.8% and 84.8% at the end of the life. At the second answer byte the guess two ahead (the units digit) is right on 100%.

Size at the end of the life (medians): with look-ahead 3,273,190 cells that have counted, 34.6 million counts above zero, 35,086 trusts used, 96 CPU-seconds; without 1,363,005 cells, 6.2 million counts, 7,775 trusts, 21 CPU-seconds.

## 2. Does it repeat, by the measure set beforehand

`PLAN_1.md`, section 5: with look-ahead at least 85%; at least 8 points above my own arm without it; after subtraction at least 85% of its level; each set of five seeds judged alone, and both needed.

| Part | reported seeds | seeds 2001 to 2005 |
|---|---|---|
| (a) at least 85% | 91.4: met | 91.3: met |
| (b) at least 8 points above without | 14.1: met | 13.2: met |
| (c) kept at least 85% of its level | 94.7: met | 92.7: met |

**It repeats**, on both sets of seeds and in all three parts.

On the reported seeds every figure I can set beside that line's is equal to the digit: the five means of the last three readings, the readings before and after subtraction, the five clean scores of subtraction, the guess rates at the end of the life, the kinds with the largest trust, 3.27 million cells, 35,086 trusts used. So that line's prose, with `cells2.py`'s split, determines its program down to the stream of lines; the agreement on the reported seeds is therefore one sample seen twice and not two. **The independent sample is seeds 2001 to 2005**, and it gives the same figures within a point (91.3 against 91.4; 86.8 after subtraction on both).

## 3. The control (mine): is the gain the information in the guess?

Everything as with look-ahead (the same kinds, counts and trusts; the guesses formed and taught as before), except that the cells keyed on a guess are keyed on the guess the system made at the same place of the preceding line (in scoring: of the preceding scored pair). That guess has the same distribution and carries nothing about the present sum.

Result: 79.5 and 79.3 (medians of the two sets), against 91.4 and 91.3 with the system's own present guesses and 77.3 and 78.1 without look-ahead. Of the gain of 13 to 14 points, about 12 go away when the guess is not about the present sum; 1 to 2 points remain (the control is above the arm without on all ten seeds, by 0.9 to 2.4 points; so the remainder is there, and I have not explained it: the cells keyed on a guess from another line still add kinds that hold the place in the line, and the plain cells' look-ahead counts still feed the guesses). **The gain is what it is said to be: cells keyed on the system's own statement about the present line.** The added cells, counts and trusts as such produce at most a seventh of it.

## 4. The open choices, and the two tested

The list is in `PLAN_1.md`, section 3 (C1 to C12). Since my figures equal that line's to the digit, the choices I took are evidently that line's. The two I named beforehand as most likely to matter, under the other choice, five reported seeds, 40,000 lines, mean of the last three readings:

| | each seed; median | main figure |
|---|---|---|
| C1: the level of experience of a cell's second and third counts taken from its first counts | 88.4, 91.1, 92.6, 93.0, 91.3; **91.3** | 91.4 |
| C2: the trusts of the plain cells' second and third counts born at nothing | 87.8, 91.3, 92.4, 92.7, 91.0; **91.3** | 91.4 |

Neither matters (0.1 point at the median; single seeds within 0.7). Not run: C3 (kinds keyed on a guess with up to four delays, 256 kinds in place of 130): in my direct addressing it needs about 12 GB for one run, and the plan asked for it only if the first result differed. C4 and C5 (order at arrival; stored shares) were not varied.

## 5. Predictions (`PLAN_1.md`, section 6)

R1 held (91.4). R2 held (14.1). R3 held (94.7). R4 held (repeats on both sets). R5 held (77.3 against 77.4). R6 held (91.4 against 91.4). R7 held (100). R8 held (control 1.2 to 2.2 above the arm without, 12 below the arm with). R9 and R10 held (0.1). R11 held (89.4 and 91.1). All eleven held; my chances for R4 and R6 (0.50, 0.45) were too low.

## 6. What I set beside the result (readings, mine)

1. **The result stands on seeds no line had used**, from a program written without sight of that line's code.
2. **Look-ahead keeps a little less well than the same cells without it.** Without look-ahead addition after subtraction is at 99% of its level on both sets; with it at 93 to 95%, and on one fresh seed (2001) at 83.5% (75.0 after, from 89.8). The level after is still 8 to 10 points above the arm without. The tens guess is right less often after subtraction (85 against 89).
3. **The remaining errors are the tens digit**, and they follow the tens guess: where the guess two ahead is right on about 89%, the whole answer is right on about 91%.
4. **The cost is as that line states it**: 2.4 times the cells and 5.6 times the counts of the arm without, for a text of 16,000 different lines. This repeat says nothing new about size.
5. Not shown here: the life of 400,000 lines (that line found the cells of four delays to fall with more reading; whether look-ahead falls too I did not run); the answer written units first; any other text.

## 7. Commands

    python3 ahead.py check
    python3 ahead.py run la_2001 seed=2001 look=1          (life add:40000,sub:20000 by default)
    python3 ahead.py run base_2001 seed=2001 look=0
    python3 ahead.py run ctl_2001 seed=2001 look=1 lag=10 plan=add:40000
    python3 ahead.py run c1_73129 seed=73129 look=1 levown=0 plan=add:40000
    python3 ahead.py run c2_73129 seed=73129 look=1 t23=0 plan=add:40000
    python3 report.py base la ctl c1 c2

The job lists are `jobs_main.txt` and `jobs_2.txt`; the raw lines of every run are `out_main.txt` and `out_2.txt`.

## 8. Added later on 20 November: three-digit sums and the depth of the look-ahead (`PLAN_2.md`)

Asked by the coordinating session after the report above; plan committed before its program (802496c); program `ahead3.py` (the depth a number H; checked against its own reference and, at H = 3 on two digits, against `ahead.py` to the digit). Text `aaa+bbb=cccc`, a fifth of the million pairs never shown, 80,000 lines (95% of the lines read are first readings), five reported seeds, mean of the last three readings, median: **no look-ahead 92.3; depth three 99.7; depth four 99.7.** The hypothesis put to me (depth three under 40% with the hundreds digit as the gap; depth four at least 20 points above) is **refuted on both conditions**. The carry is passed one place at a time by the guess of the next digit, so a depth of two or three is enough; and three-digit sums are the easier text for this system, since any four digits are shared with other lines. Full table, predictions and readings: `PLAN_2.md`, section 7; `report_3.txt`.

## 9. NOT RUN: the long life with a second task (asked by the coordinating session, 20 November, 19:18 UTC)

Asked: the look-ahead system with a cap on a cell's weight (a cell that has counted more than 16 times speaks at the weight of 16 counts in the same proportions), addition for 400,000 lines then subtraction for 200,000 with nothing of addition shown again, five reported seeds; whether at least 75% of never-shown addition stands at 400,000 lines and at least 85% of that is kept through the second task. **This was not run by this line.** The budget stated to this session (about two and a half hours, then about one hour for `PLAN_2.md`) was spent when the request arrived, and the request was conditional on budget remaining. No plan, program change or run for it exists here, and no figure of this line bears on it: every life in this folder is 40,000 + 20,000 lines of two-digit sums or 80,000 lines of three-digit sums, without a cap.

What it would take, by my estimate from the runs here (96 CPU-seconds for a life of 60,000 lines with look-ahead): the cap is one change where a cell speaks, to be put in the reference and the kernel and checked; a life of 600,000 lines is about 16 to 20 CPU-minutes a seed, so five seeds are about 25 minutes on four cores for the arm with the cap and as much again for the arm without; with plan, check and report about one hour. The half-length life (200,000 + 100,000) would be about half the run time.
