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

## 6. Outcome of the plan as written and of addition 1 (entered 20 November; `res/s_shr4_*`, `s_shr4rev_*`, `s_z4_*`, `s_drop4_*`, `s_base4_seed1.json`; table in `report_e5.txt`)

Never-shown additions, own bytes, whole answer, %. Five reported seeds (73129, 93683, 116089, 406364, 545353) for the shrinking step; development seed 1 for addition 1.

| Arm (m = 4) | 5,000 | 10,000 | 40,000 | 100,000 | 200,000 | 400,000 | last three at the end: each seed; median |
|---|---|---|---|---|---|---|---|
| constant step (`PLAN_1.md`), median of five | 74.6 | 78.0 | 80.0 | 76.4 | 63.6 | 66.8 | 68.4, 68.3, 62.5, 71.9, 65.9; 68.3 |
| (a) shrinking step, median of five | 65.0 | 72.4 | 77.4 | 72.4 | 63.8 | 55.8 | 62.1, 59.2, 57.8, 59.9, 57.7; **59.2** |
| seed 1, unchanged | 72.6 | 81.8 | 81.2 | 79.0 | 64.6 | 55.8 | 63.4 |
| seed 1, (c) trust born at nothing | 73.4 | 81.0 | 80.2 | 76.4 | 66.8 | 53.8 | 62.1 |
| seed 1, (d) no cells for the three sets that hold all four digits | 71.2 | 79.8 | 80.2 | 76.4 | 64.8 | 55.8 | 62.1 |

| Arm (m = 4, answer units first, 40,000 lines) | each seed, last three | median, last three | median, last reading | lowest single reading from 10,000 lines on |
|---|---|---|---|---|
| constant step (E4b) | 86.3, 36.8, 72.2, 96.7, 80.6 | 80.6 | 95.2 | 0.0 |
| (b) shrinking step | 89.9, 89.7, 91.6, 93.4, 91.0 | **91.0** | 91.6 | 85.4 |

**The hypothesis is refuted**: with the shrinking step m = 4 at 400,000 lines is 18 points under its own reading at 40,000 (59.2 against 77.4; more than 10 refutes). The fall is not the step. With the shrinking step the fall is smooth and the same on every seed.

| | Statement | Outcome |
|---|---|---|
| S1 | (a) 400,000 within 5 points of 40,000 | failed (18 under) |
| S2 | (a) at least 78 at 40,000 | failed (77.4) |
| S3 | (a) 400,000 above the relay's 84.2 | failed |
| S4 | (b) last three at least 90 at the median | held (91.0) |
| S5 | (b) no reading from 10,000 lines on under 70 | held (lowest 85.4) |
| addition 1 | (c) does not stop the fall | held (chance given 0.65) |
| addition 1 | (d) stops the fall | failed (chance given 0.45) |

Readings (mine).
1. **The swings and collapses were the constant step; the fall is something else.** With the shrinking step and the answer written units first, 91.0% of never-shown sums are right within 40,000 lines on every seed (89.7 to 93.4), above the relay's 84.2 at 400,000. This holds for the text written units first only.
2. **The fall is not the birth of the trust and not the cells of the whole pair** (seed 1; neither arm differs from the unchanged run by more than 1.3 points at the end). My P6 mechanism of `PLAN_1.md` is therefore wrong as stated.
3. A third candidate, from how a cell speaks (untested as this is written; addition 2): a cell says log((c + e) / (n + 14 e)) for each byte. For a byte it has never been followed by, that is log(e / (n + 14 e)), which goes on falling as the cell counts. A cell that holds two tens digits and one units digit has, for a never-shown pair, not seen that pair; where the pair's own carry is the rarer case for that cell (a units digit of 1 gives a carry only with a 9), the cell may never have seen it at all, and its "never" weighs 3.5 at 40 counts and 5.4 at 400, while the cell that speaks for the right answer says the same at any count. Absence of a count is taken as stronger evidence of absence the longer the reading, and that is wrong exactly for what was never shown.

## 7. Addition 2 (20 November, written before its runs): a cell that has counted much speaks as if it had counted a fixed number of times

One change from m = 4 with the shrinking step (arm (a)): `ncap=N`. A cell that has counted more than N times says log((c N/n + e) / (N + 14 e)): its proportions, at the weight of N counts. Its counts and its level of experience are unchanged. In the program, the numpy reference and the check (`check_output.txt`). Development seed 1, `plan=add:400000`, N in {16, 64}; the better by the mean of the last three readings goes to the five reported seeds.

*Hypothesis.* With the cap, m = 4 at 400,000 lines is within 5 points of its reading at 40,000 (seed 1, then five seeds). *Refuted if* it is more than 10 under. Chances (mine): within 5 points 0.45; the 40,000 reading itself at least 75: 0.60; 400,000 above the relay's 84.2: 0.10.
