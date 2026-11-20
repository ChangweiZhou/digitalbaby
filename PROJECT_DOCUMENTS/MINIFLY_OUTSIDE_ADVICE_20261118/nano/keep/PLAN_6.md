# Keeping, sixth plan: cells join when the answering cell hears too few cells. 20 November 2026

Line K1 (`nano/keep`); authorisation and level as in `PLAN_1.md`. Written after the fifth plan's outcome was entered; before any run of this plan.

**Where the fifth plan left it.** The size of the mismatch does not mark the change of task well enough to bring cells in. But under the third plan's rule something else changes at the change of task, and completely: every old cell is silent on the new lines, so the answering cell hears nobody.

**Question.** If cells join while the answering cell hears fewer cells than a set number, do they join in two bursts (at birth and at the change of task) with no clock and no length of life given, is addition kept as in the fourth plan, and is subtraction learned sooner?

**The rule (one change from the fourth plan's candidate: what makes cells join).** The answering cell keeps a running mean A (rate 1e-2 a byte) of the number of cells it hears at a byte (joined, and not silenced at that byte). The common signal is the shortfall X = max(0, 1 - A / K), one number that reaches every waiting cell alike; the waiting cell with index H0+i joins when the running sum of X passes (i+1) theta. One cell at birth; 384 cells available. The rest as in the fourth plan: y = 4,500; constants exp with t = 10,000; no law on the cells.

*A change in the programs' scoring, made with this plan:* a cell that is still in its youth at a reading is now scored as it is treated while learning (not silenced); before, it was scored by its marks alone. With the clock and y = 4,500 no cell is in its youth at any reading of the reported runs, so no reported figure changes; one run of the fourth plan is repeated to confirm it. The readings of the third plan's development runs with y = 18,000 and 90,000 may differ slightly under the new scoring; they were not rerun.

**Hypothesis (one).** The shortfall of cells heard is large at birth and at the change of task and nowhere else. **Refuted if** with no old line the median of five seeds has addition at the end below 75% or subtraction below 70%, or if cells go on joining through addition (more than K + 20 joined by line 400,000, median).

**The learner.** `kage.py` with `H0=1 join=hear K=.. theta=.. young=4500 blaw=exp btau=1e4`; compiled `kage.c`; `check_kage.py` (sixteen cases, two with `join=hear`; agreement to 2e-15). Up to 64,910 numbers that learn; 2 K cells are expected to join.

**Development (seed 1), no old line:** K in {128, 192}, theta in {20, 100} learned bytes: 4 runs. **Chosen:** the largest value of the smaller of the two scores at the end; then seed 1 with one old line in 32.

**Arms on the five reported seeds:** this candidate, no old line and one in 32, beside the fourth plan's and the reference arms; the life continued (`add:400000,sub:200000,add:100000`, no old line). **Measures:** as before; subtraction 50,000 lines into its phase; cells joined by lines 100,000 and 400,000 and by the end.

**Predictions.**
- F1 (0.70): no old line: addition at the end 80% or more (median).
- F2 (0.60): no old line: subtraction at the end 80% or more (median), at or above the fourth plan's 80.0.
- F3 (0.65): subtraction 50,000 lines into its phase 60% or more (median; 9 to 29 in the fourth plan).
- F4 (0.70): cells joined by line 400,000 within K + 20, and between K and K + 40 more join after the change (median).
- F5 (0.25): one old line in 32: addition 60% or more (median). (All late cells are young together when the first old lines come.)
- F6 (0.75): the life continued: subtraction after addition again within 3 points of before (median).

**Cost.** 5 development runs, 15 reported runs, one repetition: 60 to 90 CPU seconds each.

**Not run.** Other rates of the running mean; K below 128 or above 192; subtraction first; a change of task that brings no new line (the old cells would then not fall silent and no cell would join).

## Outcome (entered 20 November, after all runs; `table6_dev.txt`, `table6.txt`, `res/c2_d6_*`, `res/c2_r6_*`, `res/c2_chk4_*`)

**A statement in the plan above was wrong and is corrected here.** I wrote that with the clock and y = 4,500 no cell is in its youth at any reading of the reported runs. The fourth plan's run on seed 73129, repeated under the new scoring, differs from the old at two of twelve readings, both in the middle of a phase, where one cell was in its youth: line 250,000 of addition (subtraction 1.8 for 1.4) and line 100,000 of subtraction (addition 84.6 for 83.4). The readings at the ends of the phases, which are all the figures reported in `PLAN_3.md` to `PLAN_5.md`, are the same. The sentence in `PLAN_4.md` that addition "does not move" during subtraction holds under the old scoring; under the new one a late cell in its youth speaks on addition lines and moves a reading by about a point until its youth is over.

**Development, seed 1, no old line** (addition after its own phase | addition / subtraction at the end; cells joined by line 400,000 / by the end):

| | theta 20 | theta 100 |
|---|---|---|
| K 128 | 84.6 \| 85.0 / 88.6; 128 / 256 | 79.4 \| 79.6 / 92.6; 128 / 256 |
| K 192 | 85.0 \| 85.0 / 89.6; 192 / 384 | 79.0 \| 79.4 / 91.2; 192 / 384 |

The plan's rule gives a tie (smaller score 85.0 at theta 20 for both K). I took K = 128, the smaller system: 256 cells join in all, so that 43,278 numbers learn, as many as in the plain system; the other 128 cells of the pool never join. With one old line in 32 on seed 1: 68.6 / 94.4.

**Five reported seeds** (73129, 93683, 116089, 406364, 545353; median).

| Arm | Cells | Addition after its own phase | Addition at the end | Subtraction at the end |
|---|---|---|---|---|
| plain, no old line | 256 | 84.8 | 2.0, 1.6, 1.8, 1.4, 2.0; **1.8** | 82.0, 89.8, 91.4, 86.0, 86.0; 86.0 |
| plain, one old line in 32 | 256 | 84.8 | 20.8, 12.2, 15.6, 17.2, 9.8; **15.6** | 85.6, 85.8, 90.4, 84.2, 85.8; 85.8 |
| the program's choice (`reserve=128`), no old line | 256 + 128 | 85.4 | 5.0, 3.0, 3.2, 4.8, 4.4; **4.4** | 74.0, 77.2, 78.6, 72.8, 71.6; 74.0 |
| the program's choice, one in 32 | 256 + 128 | 85.4 | 75.2, 62.8, 67.0, 64.4, 67.8; **67.0** | 78.6, 72.8, 75.4, 73.4, 77.6; **75.4** |
| fourth plan (clock), no old line | 384 | 84.2 | 83.4, 85.6, 81.2, 85.4, 86.6; **85.4** | 85.2, 87.2, 80.0, 73.2, 78.8; 80.0 |
| fourth plan (clock), one in 32 | 384 | 84.2 | 68.8, 56.8, 69.0, 51.2, 51.2; **56.8** | 85.0, 84.8, 83.6, 85.8, 74.6; 84.8 |
| this plan (joining when too few are heard), no old line | 128 + 128 | 89.8, 79.2, 79.0, 83.2, 86.8; 83.2 | 89.6, 78.8, 78.6, 85.8, 87.0; **85.8** | 86.8, 85.2, 84.4, 91.6, 93.2; **86.8** |
| this plan, one in 32 | 128 + 128 | the same | 77.8, 60.2, 73.0, 68.4, 71.6; **71.6** | 90.8, 82.6, 91.4, 92.2, 88.0; **90.8** |

Cells joined, every seed and both conditions: 128 by line 100,000 of addition, 128 by line 400,000, 256 by the end. Subtraction 50,000 lines into its phase, no old line: 82.0, 76.2, 75.2, 83.4, 83.8. At the end, no old line: 128 cells may speak on addition lines only and 128 on subtraction lines only, none on both; with one old line in 32: 128 on addition lines only and 128 on both.

1. **The cells join in two bursts, at birth and at the change of task, and at no other time**, with no clock, no length of life and nothing about the task given: exactly 128 and 128 on every seed. The hypothesis is not refuted.
2. **With no old line both tasks are answered at the end: addition 85.8 and subtraction 86.8** (medians; every seed at 78.6 or more on both), with as many cells and numbers as the plain system, which ends at 1.8 and 86.0. Addition is where it was after its own phase (83.2 at the median; seed by seed within 2.6 points).
3. **With one old line in 32, the condition in which the level to reach was set, the level is reached at the median: addition 71.6 against the program's choice's 67.0** (higher on four seeds of five, lower on one: 60.2 against 62.8), **with subtraction at 90.8 against 75.4**. It is still below what the same system keeps with no old line (85.8): the late cells meet some addition lines in their youth and all 128 of them may speak on addition lines.
4. **Subtraction is learned sooner**: 75 to 84% after 50,000 lines, where the clock gave 9 to 29%.
5. **The life continued** (addition 400,000, subtraction 200,000, addition 100,000 lines, no old line); addition / subtraction at the end of each phase:

| seed | after addition | after subtraction | after addition again |
|---|---|---|---|
| 73129 | 89.8 / 0.0 | 89.6 / 86.8 | 89.6 / 86.4 |
| 93683 | 79.2 / 0.0 | 78.8 / 85.2 | 80.6 / 84.2 |
| 116089 | 79.0 / 0.0 | 78.6 / 84.4 | 75.0 / 84.2 |
| 406364 | 83.2 / 0.0 | 85.8 / 91.6 | 81.6 / 91.6 |
| 545353 | 86.8 / 0.0 | 87.0 / 93.2 | 77.2 / 93.6 |

No cell joins in the third phase (256 at the end). Subtraction is unchanged by addition read again (median 86.4 after, 86.8 before). Addition, read again, moves as it moves while it is being learned: medians 85.8 before and 80.6 after; on seed 545353 the readings of addition while it was first learned were 85.0, 89.0, 86.4, 80.2, 86.8, and in the third phase 88.8 and 77.2. So a single reading of a task that is being learned scatters by several points in this system, and the reading of a task that is not being read stands still. I have not looked further into it.

**What this is and is not (my reading).** Three rules, each on a cell's own quantities or on one number that reaches all waiting cells alike, and none referring to the task: (i) a strength from a line is closed unless its line was met in the cell's youth, and a cell with a closed strength on an active line is silent and unchanged; (ii) the answering cell's constants become still with its age; (iii) waiting cells join when the answering cell hears too few cells. Together they write the second task into new cells, at the moment it arrives, and leave the first as it was. The limits are those of `PLAN_4.md`, and they are the larger part of what there is to say: the change of task is seen only because subtraction brings a byte never met (with a change that brought none, the old cells would not fall silent, no cell would join, and I expect the plain system's loss; **not run**); nothing addition formed is used for subtraction (the two sets of cells share only the constants, which are still); the pool of waiting cells is finite and every new kind of line takes K of them; one pair of tasks, one order, five seeds, development level, repeated by no one.

**Predictions.** F1 (0.70) held: 85.8. F2 (0.60) held: 86.8. F3 (0.65) held: 82.0. F4 (0.70) held: 128 and 128. F5 (0.25) held: 71.6; I had thought it unlikely. F6 (0.75) held: 86.4 against 86.8.

**Not run**, as planned: other rates of the running mean; other K; subtraction first; a change of task that brings no new line.
