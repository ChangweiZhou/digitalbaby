# Keeping, seventh plan: a strength opens only if its line was met often enough in the cell's youth. 20 November 2026

Line K1 (`nano/keep`); authorisation and level as in `PLAN_1.md`. Written after the sixth plan's outcome was entered; before any run of this plan.

**Where the sixth plan left it.** With one old line in 32 the candidate keeps 71.6% of addition; with none, 85.8%. The difference is the late cells: each meets some sixteen addition lines in its youth of 500 lines, opens to them, and speaks on addition lines with what subtraction taught it.

**Question.** If a strength opens only when its line was active on at least c of the cell's young bytes, do the late cells stay closed to the old task's rare lines, so that with one old line in 32 addition is kept as with none? And what does it cost: a line of the cell's own task that is rare is then closed too.

**The rule (one change from the sixth plan's candidate).** In its youth a cell counts, for each of its strengths, the bytes at which the line was active (the count stops at 255). After youth a strength is open if its count is at least c. c = 1 is the rule so far. The youth y is lengthened with c so that a line met on one text line in 32 falls well below c and a digit met on one text line in ten well above it; y and c are chosen together.
What I expect to lose: in subtraction (larger number first) some digits are rare in their places (a first number below 10: 1% of lines; a difference of 90 or more: 1.1%; of 80 to 89: 3.1%; a second number of 90 or more: 1%). Strengths from those lines will be closed in most late cells, which are then silent on such lines; of never-shown subtraction lines some 5 to 10% contain one.

**Hypothesis (one).** The loss with old lines is the late cells' opening to them, and a count removes it. **Refuted if** with one old line in 32 the median of five seeds has addition at the end below 78% (the sixth plan: 71.6; with no old line 85.8).

**The learner.** `kage.py` with `H0=1 join=hear K=128 theta=20 young=y meet=c blaw=exp btau=1e4`; compiled; `check_kage.py` has two cases with `meet` (agreement to 1e-15). 256 cells are expected to join; 43,278 numbers that learn; one count (0 to 255) for each strength from a line, made in youth and fixed after it.

**Development (seed 1):** (y, c) in {(4,500, 30), (18,000, 100), (36,000, 180)}: a line must be met on 6%, 5% and 4.5% of the text lines of the cell's youth; one old line in 32 and no old line: 6 runs. **Chosen:** the largest sum over the two conditions of the smaller of the two scores at the end.

**Arms on the five reported seeds:** this candidate at the chosen pair, one old line in 32 and none; beside the sixth plan's candidate and the reference arms. **Measures:** as in the sixth plan.

**Predictions.**
- G1 (0.60): one old line in 32: addition at the end 78% or more (median).
- G2 (0.65): one old line in 32: subtraction at the end below the sixth plan's 90.8 by 4 points or more (median): the price.
- G3 (0.55): one old line in 32: both tasks at 75% or more (median).
- G4 (0.70): no old line: addition within 3 points of the sixth plan's 85.8 (median).
- G5 (0.60): with one old line in 32, at the end no cell that joined at the change may speak on more than half of the never-shown addition lines.

**Cost.** 6 development and 10 reported runs, 60 to 90 CPU seconds each.

**Not run.** A count that goes on after youth; a threshold that is a share of the cell's own busiest line and not a number; more than two conditions of old lines (one in 8, one in 128); a change of task that brings no new line.

## Outcome (entered 20 November, after all runs; `table7_dev.txt`, `table7.txt`, `res/c2_d7_*`, `res/c2_r7_*`)

**Development, seed 1** (addition after its own phase 84.6 in every run; addition / subtraction at the end):

| y, c | no old line | one old line in 32 |
|---|---|---|
| 4,500, 30 | 85.0 / 88.6 | 87.4 / 94.0 |
| 18,000, 100 | 85.0 / 88.6 | 87.4 / 96.4 |
| 36,000, 180 | 85.0 / 88.6 | 87.4 / 95.0 |

The plan's rule gives a three-way tie (85.0 + 87.4). I took y = 4,500, c = 30, which differs from the sixth plan's candidate in c alone.

**Five reported seeds** (73129, 93683, 116089, 406364, 545353; median). 128 cells join at birth and 128 at the change of task on every seed and in both conditions.

| Arm | Addition after its own phase | Addition at the end | Subtraction at the end |
|---|---|---|---|
| plain, one old line in 32 | 84.8 | 20.8, 12.2, 15.6, 17.2, 9.8; **15.6** | 85.6, 85.8, 90.4, 84.2, 85.8; 85.8 |
| the program's choice (`reserve=128`), one in 32 | 85.4 | 75.2, 62.8, 67.0, 64.4, 67.8; **67.0** | 78.6, 72.8, 75.4, 73.4, 77.6; 75.4 |
| sixth plan (met once), one in 32 | 83.2 | 77.8, 60.2, 73.0, 68.4, 71.6; **71.6** | 90.8, 82.6, 91.4, 92.2, 88.0; 90.8 |
| this plan (met on 30 young bytes), one in 32 | 89.8, 79.2, 79.0, 83.2, 86.8; 83.2 | 86.4, 81.6, 80.4, 86.0, 80.4; **81.6** | 86.8, 85.8, 88.4, 85.8, 85.0; **85.8** |
| plain, no old line | 84.8 | 2.0, 1.6, 1.8, 1.4, 2.0; **1.8** | 86.0 |
| the program's choice, no old line | 85.4 | 5.0, 3.0, 3.2, 4.8, 4.4; **4.4** | 74.0 |
| sixth plan, no old line | 83.2 | 89.6, 78.8, 78.6, 85.8, 87.0; **85.8** | 86.8, 85.2, 84.4, 91.6, 93.2; 86.8 |
| this plan, no old line | the same | 89.6, 78.8, 78.6, 85.8, 87.0; **85.8** | 86.8, 85.2, 84.4, 91.6, 93.2; **86.8** |

With no old line the runs are the sixth plan's to the digit, as they must be (no line of the old kind is met, whatever c).

1. **With one old line in 32, addition is kept at 81.6% (80.4 to 86.4 over the seeds) and subtraction learned to 85.8%.** The program's choice: 67.0 and 75.4; the plain system: 15.6 and 85.8. The hypothesis is not refuted (the line was 78%). On every seed addition at the end is within 3.4 points of its level after its own phase on one side or the other, except seed 545353 (86.8, then 80.4).
2. At the end, in both conditions and on every seed, 128 cells may speak on addition lines only and 128 on subtraction lines only, none on both.
3. **The price I expected did not arise, and my account of it in the plan was wrong.** I had reasoned by the role of a digit in the line (a first number below 10 is rare). But a line of the system is a byte value at a delay, whatever its role: the digit 0 five bytes back is met when it is a digit of either number or of an answer of the line or of the line before. So every digit line is met many times in 500 lines by either task, and only the operator's lines are met by one task alone; the late cells speak on every never-shown subtraction line (128.0 of 128 on average). Subtraction with one old line in 32 is nevertheless 5 points lower than in the sixth plan (85.8 against 90.8) and equal to its level with no old line (86.8). My reading, not tested: in the sixth plan the late cells, open to addition lines, were also taught by them, and that helped subtraction; here they are not.
4. The threshold c is tied to how rare the old lines are. One old line in 32 gives about 16 meetings of each operator line in a youth of 500 lines, below 30. With one old line in 8 it would be about 62, above it, and the late cells would open to them. **Not run** here; the next plan runs it.

**Predictions.** G1 (0.60) held: 81.6. G2 (0.65) held as a number (85.8, 5 points below 90.8) and failed in its reason: no late cell is silent on any subtraction line. G3 (0.55) held: 81.6 and 85.8. G4 (0.70) held: 85.8, the same runs. G5 (0.60) held: none.

**Not run**, as planned: a count that goes on after youth; a threshold that is a share and not a number; one old line in 8 or in 128; a change of task that brings no new line.
