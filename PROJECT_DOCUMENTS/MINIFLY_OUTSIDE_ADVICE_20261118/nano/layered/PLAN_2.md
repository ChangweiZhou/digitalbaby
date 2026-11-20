# Line `nano/layered`, second plan (E2): two-digit sums with two delay lines, without a crossing (20 November 2026)

**Authorisation and standing** as in `PLAN_1.md`. **Source.** This branch at the commit that adds this file; programs unchanged since 83c7eb0. Written while the reported seeds of the first plan were running and before any scored run of this system on the text `d2`; I knew the first plan's development outcome (seed 1: `m1` 99.9, `g1` 97.1 at etaA = 1/4, etaB = 1, nB = 32) and one timing run of 20,000 lines of `d2` at the default steps (4% shown, 3% never shown). Development level; no checker gate, seal or review is claimed.

## 1. Question, hypothesis, what would refute it

Text `d2`: lines `aa+bb=ccc`; the present byte and the two before it are on lines (`chain=2`); 2,000 of the 10,000 pairs are never shown; whole answer right on 500 never-shown pairs, mean of the last three readings of 1,600,000 lines (every 200,000). `hold.py` (256 cells, each reading 16 others; quoted from `loop_small_case/REPORT.md` section 12.3 at 0ba9f241, not rerun, the lines and scoring being the same): with the crossing 65.8, 66.6, 67.3, 60.1, 62.1 (median **65.8**), without 16.2, 17.3, 27.4, 23.4, 29.2 (**23.4**).

**Does the layered system, with no crossing, learn two-digit sums that it was never shown, where three of the four digits have to be held for up to five bytes?** Here a gate reads the present byte and the two before it, so a cell A can tell a number's first digit from its second and the first number from the second.

*Hypothesis.* Yes: at least 50% of never-shown sums at the median of the five reported seeds.

*Refuted if* the median is below 35% (nearer to `hold.py` without the crossing than with it).

## 2. The learner and its numbers

`layered.py` as in the first plan, with `chain=2`: cells A 160, each reading 42 lines (14 byte values at three positions); cells B 106, each reading all 160 cells A and the 14 lines of the present byte; the answering cell 14 branches. Numbers that learn (forward): cells A 13,440 from lines and 320 thresholds; cells B 16,960 + 1,484 + 106; answering cell 1,498: **33,808** (`hold.py` on this text: 21,504 + 8,192 + 512 + 3,598 = 33,806). Return strengths apart: 18,444.

## 3. Runs

**Development (seed 1), 1,600,000 lines, compiled:** etaB = 1 (the first plan's choice; it was at the edge of that grid and nB carried the rest); etaA in {1/16, 1/4, 1} by nB in {8, 32, 128}: 9 runs. *Rule of choice, fixed now:* the highest mean of the last three readings of never-shown sums. If the best is at a corner of the grid I may add up to three runs beyond that corner and will say so.

**Reported seeds:** (a) the chosen setting; (b) the same with etaA = 0 (cells A never change).

## 4. Predictions (mine, before any run on this text)

| | Statement | Chance |
|---|---|---|
| D1 | chosen setting: median of never-shown sums at least 50% | 0.50 |
| D2 | chosen setting: median at least 65.8% (the crossing arm of `hold.py`) | 0.30 |
| D3 | chosen setting: median at least 35% (not refuted) | 0.70 |
| D4 | cells A never changing: at least 20 points below the chosen setting at the median | 0.65 |
| D5 | chosen setting: never-shown sums within 10 points of sums shown on every seed | 0.80 |

## 5. Cost, and what is not run

About 5 CPU-minutes a run: development 9 to 12 runs, reported 10 runs, about 100 CPU-minutes, 25 to 30 minutes of wall time. **Not run:** `hold.py` (quoted); one delay line; no delay line; other sizes; K cells A for each cell B; a comparator by back-propagation through time.

## Outcome (entered 20 November after the runs; nothing above was changed)

**Development (seed 1), never-shown sums, %, mean of the last three readings (sums shown in brackets), `res/lay_d2_*.json`:**

| etaA \ nB (etaB = 1) | 8 | 32 | 128 |
|---|---|---|---|
| 1/16 | 5.9 (6.5) | 29.3 (31.9; rose to 69 only at the last reading) | 82.9 (83.5) |
| 1/4 | 58.2 (58.5) | 71.5 (72.9) | 89.6 (90.3) |
| 1 | 61.1 (67.0) | **92.1** (93.1) | 87.6 (89.7) |

**Chosen by the rule: etaA = 1, etaB = 1, nB = 32.** It lies on the edge of the grid in etaA but not at a corner, so by the plan no run was added beyond it.

**Five reported seeds** (`report.txt`, `res/lay_r2_*.json`), 1,600,000 lines, %, mean of the last three readings; `hold.py`'s rows quoted from REPORT section 12.3 at 0ba9f241:

| | 73129 | 93683 | 116089 | 406364 | 545353 | median |
|---|---|---|---|---|---|---|
| layered, no crossing: never-shown sums | 86.3 | 88.3 | 92.1 | 75.9 | 74.5 | **86.3** |
| ... the same runs: sums shown | 84.9 | 89.3 | 92.4 | 76.5 | 78.9 | **84.9** |
| layered, cells A never change: never-shown sums | 0.8 | 0.9 | 0.7 | 0.6 | 0.9 | **0.8** |
| `hold.py`, one crossing: never-shown sums (quoted) | 65.8 | 66.6 | 67.3 | 60.1 | 62.1 | **65.8** |
| ... sums shown (quoted) | 66.4 | 66.8 | 68.0 | 65.2 | 69.9 | **66.8** |
| `hold.py`, no crossing: never-shown sums (quoted) | 16.2 | 17.3 | 27.4 | 23.4 | 29.2 | **23.4** |

By place at the end (hundreds/tens/units), never-shown sums: 99/87/100, 99/90/100, 98/94/100, 100/78/100, 100/74/100; with cells A never changing 80/17/8, 65/16/10, 79/19/10, 71/17/10, 75/14/13. About 5.3 CPU-minutes a run (3.8 for the control).

**The hypothesis held as registered**: 86.3% of never-shown sums at the median, every seed above 74, with no crossing; every seed is above every seed of `hold.py` with the crossing (60 to 67), at equal numbers that learn (33,808 against 33,806; the layered system has 18,444 return strengths besides, `hold.py` 3,598). For comparison only, and not at equal conditions: the full chain of nine delay lines with no holding cells gave 84.2 at 400,000 lines (REPORT section 10). What is still missed is the tens digit (74 to 94% by place); hundreds and units are right 98 to 100%.

| | Prediction | Outcome |
|---|---|---|
| D1 | median of never-shown sums at least 50% (0.50) | held (86.3) |
| D2 | median at least 65.8% (0.30) | held (86.3); I had thought this unlikely |
| D3 | median at least 35% (0.70) | held |
| D4 | cells A never changing at least 20 points below (0.65) | held (0.8 against 86.3) |
| D5 | never-shown within 10 points of shown on every seed (0.80) | held (largest difference 4.4 points, seed 545353) |

My reading: on this text, where a gate can see from its lines which digit is before it, exact traces in cells that read only lines, taught over two steps of return strengths, do more than the crossing did in `hold.py`, and no value differs from one cell read to the next. **Not run:** `hold.py` (quoted); steps beyond the grid; one delay line or none; other sizes; a comparator by back-propagation through time; repetition by anyone else.
