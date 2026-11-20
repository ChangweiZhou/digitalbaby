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
