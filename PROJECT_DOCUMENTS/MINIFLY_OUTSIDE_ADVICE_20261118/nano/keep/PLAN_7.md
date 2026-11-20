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
