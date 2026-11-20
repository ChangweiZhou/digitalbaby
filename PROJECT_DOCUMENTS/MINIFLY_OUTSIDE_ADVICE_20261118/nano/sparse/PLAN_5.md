# Line `nano/sparse`, fifth plan (E5): is the fall of m = 4 with more reading, and the swinging of its readings, the constant step of the trust? (20 November 2026)

**Authorisation.** Mr. Xu, 20 November 2026, as quoted to this session by the coordinating session: "Keep experimenting with the agents." and "Respect evidence and keep trying new ideas." **Source.** `claude/nano-sparse-8mmhfm` at the commit before this one. `sparse.py` gains the option `shrink=1` (the option of `nano/table/trust.py`, `PLAN_4.md` there; it was in the kernel from the start and is now reachable, with the numpy reference and a line of the check, `check_output.txt`). Written before any run with it. Development level; no checker gate, seal or outside review is claimed.

## 1. The evidence this follows

1. `PLAN_1.md`, outcome: m = 4 has 80.9% of never-shown sums at 40,000 lines and 68.3% at 400,000, and its readings swing by ten points and more from one to the next. Its step, 0.1, is the largest tried.
2. Fourth plan, E4b (`PLAN_3.md`, outcome entered with this commit): with the answer written units first, m = 4's last reading is 95.2% at the median, but single readings of single seeds fall to 0 to 39% and come back, so the mean of the last three readings is 80.6.
3. Two candidates were named for the fall (`PLAN_1.md`, reading 3): the step, or the cells of the whole pair taking over on shown lines.

## 2. The question and the one change

**With a step that shrinks, does m = 4 stop falling and stop swinging?** One change: a trust's step is eta over the square root of one plus the number of times that trust has been used (`shrink=1`). Nothing else: m = 4, eta 0.1, e 2, trusts born at 0.02, nothing chosen again. Runs, five reported seeds each: (a) `plan=add:400000`, read at 5,000, 10,000, 20,000, 40,000 and every 50,000; (b) the answer units first (`rev=1`), 40,000 lines, read every 5,000.

## 3. Hypothesis, what would refute it, predictions (mine, before any run)

*Hypothesis.* The fall is the step: with the shrinking step, m = 4 at 400,000 lines (mean of the last three readings, median of five) is within 5 points of its own reading at 40,000. *Refuted if* it is more than 10 points below; then the fall is not the step alone, and the cells of the whole pair are the next thing to test.

| | Statement | Chance |
|---|---|---|
| S1 | (a) 400,000 within 5 points of 40,000 | 0.50 |
| S2 | (a) the reading at 40,000 is at least 78 (not more than 3 below the constant step's 80.9) | 0.60 |
| S3 | (a) 400,000 above the relay's 84.2 | 0.15 |
| S4 | (b) the mean of the last three readings is at least 90 at the median | 0.55 |
| S5 | (b) no reading of any seed from 10,000 lines on is under 70 | 0.50 |

## 4. Size, cost, what is not run

Size as m = 4 in `PLAN_1.md`. About 15 CPU-minutes for (a) and 2 for (b). Not run: the shrinking step for m = 3 (its readings do not swing); a grid over eta with the shrinking step; removing the cells of the whole pair (the next test if this is refuted).

## 5. Addition 1 (20 November, written before its runs): two more single changes on the same question, at the coordinating session's suggestion

The coordinating session, having read the first report, proposed parting the two candidates for m = 4's fall by three single changes: the shrinking step (this plan as written), the trust of every kind born at nothing, and the cells of the sets that hold all four digits left out. It also reported from `nano/table` (its `PLAN_8.md`, which I have not read) that with many kinds given at once the birth value of the trust decided the result there. In `PLAN_1.md` the trust was born at 0.02 or 0.25 and 0.02 was chosen; nothing was tried.

Two arms are added, each one change from m = 4 of `PLAN_1.md` (eta 0.1, e 2, constant step), `plan=add:400000`, read at the same marks, **development seed 1 first**; the five reported seeds only for an arm that does not fall on seed 1 (400,000 within 5 points of 40,000):
- (c) `T0=0`: every trust born at nothing.
- (d) `drop=3467/2356/4578`: no cells for the three sets of delays that hold all four digits where the tens, the hundreds and the units digit are named (252 kinds). These sets are named by me from knowing the line; the arm is a diagnostic of what takes over, not a system proposed.

Predictions (mine): (c) does not stop the fall (chance 0.65 that 400,000 is more than 5 below 40,000); (d) stops it (0.45). Cost: about 3 CPU-minutes each on seed 1.
