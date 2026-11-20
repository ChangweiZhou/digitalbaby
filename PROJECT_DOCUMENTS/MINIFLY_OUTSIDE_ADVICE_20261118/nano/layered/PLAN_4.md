# Line `nano/layered`, fourth plan (E4): a second gate on what a holding cell gives out (20 November 2026)

**Authorisation and standing** as in `PLAN_1.md`. **Source.** This branch at the commit that adds this file; programs unchanged since 83c7eb0 (the option `og=1` was in both programs from the start and is in the check: CHECK PASSED). Written after the third plan's outcome and before any scored run with `og=1` (the check's runs of 300 to 1,000 lines apart). Development level; no checker gate, seal or review is claimed.

## 1. Question, hypothesis, what would refute it

The text where the first three plans left the most to gain is `g2` (two-digit numbers with 0 to 8 letters between them): 8.3% of never-shown sums at the gap of 8 letters, units digit hardly above a guess. Crosswalk `OLD_LISTS_AGAINST_THE_LOOP_20261120.md`, section 5, item 5: the cell has one gate and is read by every reader at every byte, the letters included.

**Does a second gate, on what a cell A gives out, let the layered system answer never-shown sums on `g2`, at equal numbers that learn?**

The gate: o = logistic function of a third set of strengths from the same active lines, plus a threshold; the cell gives out o times its state; its state and its traces are as before. What reaches the cell changes the first two sets of strengths by (that value times o) times each trace, and the third set, from the lines active now, by that value times state times o (1 - o). The gate is taught by the relayed mismatch as everything else is; not by reward.

*Hypothesis.* With the output gate, at least 30% of never-shown sums at the gap of 8 letters on the development seed. *Refuted if* the best arm with the gate is at or below 15% at the gap of 8 letters. (A gain over 8.3% that stays under 15% is refuted as registered and reported as it is.)

## 2. Arms and numbers

One gate, the same numbers: the third plan's chosen run, already made (`res/lay_d3_a1_n32.json`: cells A 160, cells B 118, **40,864**; etaA = 1, etaB = 1, nB = 32; never shown 9.4 / 10.5 / 8.3 at gaps 0 / 4 / 8). Not rerun.

With the output gate: cells A 120, each with three sets of 54 strengths and three thresholds (19,440 + 360); cells B 134, each reading all cells A and the 18 lines of the present byte (16,080 + 2,412 + 134); the answering cell 2,430: **40,856**. Return strengths apart: 18,492. Text `g2`, `chain=2`, gap 8, 2,400,000 lines, development seed 1, etaB = 1, nB = 32. Four runs: the output gate's threshold at birth 1 (mostly open) or -1 (mostly shut), by etaA = 1 or 1/4. *Rule of choice, fixed now:* the highest mean of the last three readings of never-shown sums at the gap of 8 letters.

The two arms differ in the gate and, because the numbers are held equal, in the split between cells A and cells B (160 + 118 against 120 + 134). A difference between them cannot be put on the gate alone; if the gate arm is clearly better I will say that a one-gate run at 120 + 134 is owed.

**Reported seeds** (both arms, five seeds each) only if the chosen gate arm reaches 30% at the gap of 8 letters on the development seed; otherwise not run.

## 3. Predictions (mine, before the runs)

| | Statement | Chance |
|---|---|---|
| O1 | best gate arm: at least 30% of never-shown sums at the gap of 8 letters | 0.15 |
| O2 | best gate arm: above the one-gate arm (8.3) at the gap of 8 letters by at least 5 points | 0.30 |
| O3 | best gate arm, last reading, gap of 8: units digit right on at least 30% of never-shown sums (one gate: 15) | 0.25 |
| O4 | the arm born mostly open does at least as well as the arm born mostly shut (best over the two steps) | 0.60 |
| O5 | best gate arm: never-shown sums (mean over gaps) within 10 points of sums shown | 0.85 |

My expectation, stated beforehand: the third plan's reading was that the first number's units digit is not held, and a gate on what is given out does not by itself make a cell hold; I expect little.

## 4. Cost, and what is not run

Four runs of about 13 CPU-minutes, one batch, about 15 minutes of wall time. **Not run:** the reported seeds unless the line of 30% is reached; a one-gate arm at 120 + 134; the gate on the other texts; the gate in `hold.py` (the crosswalk's own smallest experiment names the cell of REPORT section 12; this plan puts the gate in the layered cell, which is what this line owns).
