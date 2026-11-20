# Keeping, fifth plan: cells join on a common signal of mismatch, not on a clock (candidate c, in part); and the life continued. 20 November 2026

Line K1 (`nano/keep`); authorisation and level as in `PLAN_1.md`. Written after the fourth plan's outcome was entered; before any run of this plan.

**Where the fourth plan left it.** Strengths closed until met in youth, with constants that become still, keep addition at its own level with no old line (85.4) and learn subtraction to 80.0. Two things in it are still set from outside: the cells join by a clock whose rate was fixed with the length of the life known, and the cells that can learn the new task arrive one by one whatever happens (one every 21,000 bytes; subtraction is at 9 to 29% after the first 50,000 lines of it).

**Question.** If cells join when a signal that reaches all cells alike says the mismatch is above its usual size, (i) is the keeping the same, (ii) is subtraction learned sooner and as well, with no clock and no length of life given?

**The rule (one change from the fourth plan's candidate).** The answering cell keeps two running means of its own squared mismatch (summed over its branches): a fast one F (rate 1e-3 a byte) and a slow one G (rate 1e-5 a byte). The common signal is the excess X = max(0, F - (1 + delta) G). It is one number, the same for every cell. C is its running sum; the cell with index H0+i joins when C passes (i+1) theta (each waiting cell has its own level, laid down at birth). One cell is present at birth. Everything else is the fourth plan's: y = 4,500; constants with exp, t = 10,000; no law on the cells; 384 cells available in all (a cell that never joins is never used).
This is candidate (c) of the brief only in its second half (a common signal changes how fast the system takes on cells). Its first half, steps that shrink with a cell's accumulated use, is not in this plan: the second plan found that a stiffening law trades keeping against learning, and the third that silence on unmet lines protects without stiffening. The option `use=` exists in the programs and is checked, and is **not run**.

**Hypothesis (one).** The excess of mismatch over its usual size is large at birth and at the change of task and nowhere else, so that cells join in two bursts and the keeping of the fourth plan is unchanged. **Refuted if** with no old line the median of five seeds has addition at the end below 70%, or subtraction below 65%, or if more than 20 cells join between line 100,000 and line 400,000 of addition (median).

**The learner.** `kage.py` with `H0=1 join=sig theta=.. delta=.. young=4500 blaw=exp btau=1e4`; compiled `kage.c`; `check_kage.py` has three cases with `join=sig` (agreement to 2e-15). Up to 64,910 numbers that learn.

**Development (seed 1), no old line:** theta in {150, 250, 400}, delta in {0.25, 0.5}: 6 runs. **Chosen:** the largest value of the smaller of the two scores at the end. The chosen setting is then run on seed 1 with one old line in 32.

**Arms on the five reported seeds:** this candidate, no old line and one in 32; beside the fourth plan's candidate and the reference arms. **Measures:** as before; subtraction 50,000 lines into its phase; the number of cells joined by lines 100,000 and 400,000 of addition and by the end.

**The life continued (a measure on the candidates, no change to them).** `plan=add:400000,sub:200000,add:100000`, no old line, five seeds, for the fourth plan's candidate and for this one: subtraction at the end of the third phase (the plain system: 2.8% after 40,000 lines of addition taught again, `loop_small_case/REPORT.md`, section 11) and addition there.

**Predictions.**
- E1 (0.60): no old line: addition at the end 80% or more (median).
- E2 (0.55): no old line: subtraction at the end 75% or more (median).
- E3 (0.60): subtraction 50,000 lines into its phase is 50% or more (median), against 9 to 29 in the fourth plan.
- E4 (0.50): at the chosen setting between 60 and 200 cells join after the change of task (median), and at most 20 between lines 100,000 and 400,000.
- E5 (0.30): one old line in 32: addition 60% or more (median). (More late cells are young together when the first old lines come; I expect the fourth plan's weakness to remain.)
- E6 (0.80): the life continued, both candidates: after addition is read again for 100,000 lines, subtraction is within 3 points of its level before (median), and addition no lower than before.

**Cost.** 7 development runs, 10 reported runs, 10 runs of the continued life: about 70 to 90 CPU seconds each.

**Not run.** Steps that shrink with accumulated use; other rates of the two running means; a pool larger than 384; subtraction first; a change of task that brings no new line.
