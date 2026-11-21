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

## Development outcome (entered 20 November after the four development runs and before any reported seed was read; nothing above was changed)

Development seed 1, `g2`, 2,400,000 lines, %, mean of the last three readings (`res/lay_d4_og_*.json`); by place: never-shown sums at the gap of 8 letters, last reading.

| arm | never shown at gaps 0 / 4 / 8 | shown at gaps 0 / 4 / 8 | by place at 8 letters |
|---|---|---|---|
| one gate, 160 + 118 (third plan's run) | 9.4 / 10.5 / 8.3 | 12.5 / 11.7 / 10.1 | 97 / 70 / 15 |
| output gate born mostly open (threshold 1), etaA = 1 | 73.5 / 73.3 / 70.5 | 74.2 / 73.7 / 72.0 | 98 / 77 / 100 |
| output gate born mostly open, etaA = 1/4 | 76.9 / 76.7 / 75.3 | 81.1 / 80.1 / 78.9 | 99 / 84 / 95 |
| output gate born mostly shut (threshold -1), etaA = 1 | 45.7 / 45.9 / 41.8 | 48.4 / 46.3 / 41.5 | 98 / 82 / 57 |
| output gate born mostly shut, etaA = 1/4 | 85.6 / 85.9 / **84.8** | 87.4 / 87.1 / 86.0 | 97 / 90 / 99 |

**Chosen by the rule: the output gate born mostly shut, etaA = 1/4** (84.8% of never-shown sums at the gap of 8 letters). All four arms with the gate are far above the one-gate arm; each sat at 3 to 17% for 200,000 to 1,600,000 lines and then rose. The line of 30% is passed, so the reported seeds are run, both arms.

Development predictions: O1 (at least 30%; I had given 0.15) **held**, against my expectation; O2 held; O3 held (units digit 99); O4 (born open at least as good as born shut, best over the two steps) **failed** (75.3 against 84.8); O5 held (85.4 against 86.8).

**Run with the reported seeds, as the plan said was owed if the gate arm was clearly better:** one gate at the gate arm's split, 120 + 134 cells (37,256 numbers that learn; fewer than 40,856, because it lacks the third set of strengths), seed 1, etaA = 1, etaB = 1, nB = 32, 2,400,000 lines. It says whether the split, and not the gate, made the difference. *My prediction before it is read:* it stays at or below 15% at the gap of 8 letters, chance 0.80.

**Reported seeds** 73129, 93683, 116089, 406364, 545353, 2,400,000 lines: the chosen gate arm (120 + 134, threshold -1, etaA = 1/4); the one-gate arm at equal numbers (160 + 118, etaA = 1: the third plan's chosen setting). *Predictions before they are read:* O6: gate arm, median of never-shown sums at the gap of 8 letters at least 30%, chance 0.75; O7: at least 60%, chance 0.50; O8: the one-gate arm's median at or below 15%, chance 0.80; O9: gate arm, never-shown (mean over gaps) within 10 points of shown on every seed, chance 0.80. Cost: 11 runs of about 13 CPU-minutes.

## Outcome on the reported seeds (entered 20 November after the runs; nothing above was changed)

*Erratum to the development outcome above:* one gate at 120 + 134 cells has 34,256 numbers that learn, not 37,256 (the program's count).

`report.txt`, `res/lay_r4_*.json`, `res/lay_d4_one_120_134.json`. Text `g2`, two delay lines, 0 to 8 letters, 2,400,000 lines, no crossing; %, mean of the last three readings.

| | 73129 | 93683 | 116089 | 406364 | 545353 | median |
|---|---|---|---|---|---|---|
| **output gate** (120 + 134, 40,856 numbers): never-shown sums at the gap of 8 letters | 41.7 | 64.7 | 74.5 | 60.3 | 79.5 | **64.7** |
| ... never-shown sums, mean over gaps 0, 4, 8 | 48.0 | 69.7 | 78.0 | 63.8 | 80.2 | **69.7** |
| ... sums shown, mean over gaps | 47.9 | 71.9 | 80.0 | 65.6 | 82.3 | **71.9** |
| ... by place at 8 letters, last reading (hundreds/tens/units) | 98/60/86 | 99/72/99 | 98/71/95 | 98/62/97 | 99/86/97 | |
| **one gate** (160 + 118, 40,864 numbers): never-shown sums at the gap of 8 letters | 4.3 | 2.1 | 4.7 | 3.0 | 3.7 | **3.7** |
| ... never-shown sums, mean over gaps | 5.1 | 3.7 | 5.0 | 4.6 | 5.0 | **5.0** |
| ... by place at 8 letters, last reading | 99/51/6 | 98/54/5 | 97/63/6 | 97/56/6 | 98/55/8 | |

Last reading by gap (0 / 4 / 8), median of five seeds: with the gate 75 / 76 / 66; one gate 6 / 4 / 3. About 11.6 CPU-minutes a run with the gate, 12.8 without.

**The control for the split** (one gate at the gate arm's 120 + 134 cells, 34,256 numbers, seed 1): never-shown sums 4.7 / 3.7 / 3.9 at gaps 0 / 4 / 8, flat over the whole run (by place at the end 96 / 56 / 6). So the split between cells A and B is not what made the difference on the development seed; the gate arm at that seed had 84.8. (The control has 6,600 fewer numbers than the gate arm; the equal-numbers one-gate arm is the 160 + 118 row.)

**The hypothesis held as registered, on the development seed and on the reported seeds.** With a gate on what a holding cell gives out, the layered system without a crossing answers 64.7% of never-shown sums at the gap of 8 letters at the median of five seeds (41.7 to 79.5), against 3.7% for one gate at equal numbers (2.1 to 4.7): every seed of the one above every seed of the other. Never-shown sums are answered as often as sums shown (69.7 against 71.9). The gate arm's seeds each sat at 3 to 10% for 600,000 to 1,000,000 lines and then rose within one or two readings; seed 73129 was still rising at the end. What is still missed with the gate is the tens digit (60 to 86% by place); the units digit, which the one-gate arm does not hold (5 to 8%), is right 86 to 99% of the time.

| | Prediction | Outcome |
|---|---|---|
| O1 | best gate arm at least 30% at 8 letters, development seed (0.15) | held (84.8); I had expected little |
| O2 | above the one-gate arm by at least 5 points (0.30) | held |
| O3 | units digit at least 30% (0.25) | held (99) |
| O4 | born mostly open at least as good as born mostly shut (0.60) | **failed** (75.3 against 84.8, development seed) |
| O5 | never-shown within 10 points of shown (0.85) | held |
| control | one gate at 120 + 134 at or below 15% (0.80) | held (3.9) |
| O6 | reported seeds, gate arm: median at 8 letters at least 30% (0.75) | held (64.7) |
| O7 | at least 60% (0.50) | held (64.7) |
| O8 | one-gate arm's median at or below 15% (0.80) | held (3.7) |
| O9 | gate arm: never-shown within 10 points of shown on every seed (0.80) | held (largest difference 2.2) |

Three of my chances were badly placed: O1 to O3 at 0.15 to 0.30 for what happened on all four development arms and all five seeds.

**What this does and does not say** (my reading). It says that at 2,400,000 lines the layered cell with a gate on its output learns this text where the same cell without it, and `hold.py` with one crossing (quoted: 8%), do not. It does not say the one-gate cell cannot: the third plan's addition has it at 50% after 9,600,000 lines. So on this evidence the gate brings the rise forward (to 600,000 to 1,000,000 lines from about 5,000,000 on the development seed) and may raise the level; the level at equal, longer length was not run. Why the gate does this was not examined. One thought, untested: without the gate every cell B reads every cell A's state at every byte, the letters included, so while a cell A is still learning what to hold, its changing state disturbs all answers; with the gate a cell A can be silent except at the bytes where its content is of use. **Not run:** the gate arm beyond 2,400,000 lines; the one-gate arm at 9,600,000 lines on the reported seeds; the gate born open on the reported seeds; the gate on `m1`, `g1`, `d2`; the gate in `hold.py`'s own cell (the crosswalk's item 5 as written); a look inside the cells; a comparator by back-propagation through time; repetition by anyone else.
