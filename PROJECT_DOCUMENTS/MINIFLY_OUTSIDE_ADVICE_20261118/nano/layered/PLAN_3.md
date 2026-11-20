# Line `nano/layered`, third plan (E3): two-digit numbers with letters between them, without a crossing (20 November 2026)

**Authorisation and standing** as in `PLAN_1.md`. **Source.** This branch at the commit that adds this file; programs unchanged since 83c7eb0. Written while the reported seeds of the second plan were running and before any scored run of this system on the text `g2` (one timing run of 20,000 lines at the default steps: 1% of never-shown sums). I knew the first plan's outcome and the second plan's development grid (seed 1, `d2`: 92.1% of never-shown sums at etaA = 1, etaB = 1, nB = 32; 87.6 to 89.6 at nB = 128; 5.9 to 61.1 at nB = 8). Development level; no checker gate, seal or review is claimed.

## 1. Question, hypothesis, what would refute it

Text `g2`: lines `aa+<0 to 8 letters>bb=ccc`, the present byte and the two before it on lines (`chain=2`); 2,000 of 10,000 pairs never shown; whole answer right on 500 never-shown pairs at gaps of 0, 4 and 8 letters (and on 500 shown pairs), mean of the last three readings of 2,400,000 lines (every 200,000). `hold.py` on the development seed (quoted from `loop_small_case/PLAN_11.md` and REPORT section 12.3 at 0ba9f241; the lines and scoring are the same): one crossing 7.7% (10, 10, 8 at the three gaps; by place 97, 76, 16); no crossing 3.1%. Not learned there.

**Do holding cells that read only lines, with ordinary cells above them and no crossing, hold both digits of the first number exactly across the letters and answer sums never shown?**

*Hypothesis.* Yes: at least 30% of never-shown sums at the gap of 8 letters on the development seed.

*Refuted if* 15% or less at the gap of 8 letters (the eleventh plan's line).

## 2. The learner and its numbers

`layered.py`, `chain=2`: cells A 160, each reading 54 lines (18 byte values at three positions); cells B 118, each reading all cells A and the 18 lines of the present byte; the answering cell 18 branches. Numbers that learn (forward): 17,280 + 320; 18,880 + 2,124 + 118; 2,142: **40,864** (`hold.py` on this text: 27,648 + 8,192 + 512 + 4,626 = 40,978). Return strengths apart: 21,004.

## 3. Runs

**Development seed 1, 2,400,000 lines, compiled, four settings** at etaB = 1: (etaA, nB) = (1, 32) (the second plan's choice), (1, 128), (1/4, 128), (4, 32); and the control with cells A never changing at (0, 32). *Rule of choice, fixed now:* the highest mean of the last three readings of never-shown sums at the gap of 8 letters.

**Reported seeds, only if the chosen setting reaches 30% at the gap of 8 letters on the development seed:** the chosen setting on the five seeds; the control with cells A never changing on the five seeds if time allows, else stated as not run.

## 4. Predictions (mine, before any run on this text)

Never-shown sums throughout; shown sums are reported beside them.

| | Statement | Chance |
|---|---|---|
| G1 | chosen setting, development seed: at least 30% at the gap of 8 letters | 0.55 |
| G2 | the same: at least 50% at the gap of 8 letters | 0.40 |
| G3 | the same: the gap of 8 letters within 15 points of the gap of none | 0.50 |
| G4 | the same, at the gap of 8 letters at the last reading: the units digit right on at least 50% of never-shown sums (`hold.py`: 16) | 0.55 |
| G5 | the same: never-shown sums (mean over gaps) within 10 points of sums shown | 0.80 |
| G6 | cells A never changing, development seed: at most 5% | 0.85 |
| G7 | if the reported seeds are run: median at the gap of 8 letters at least 30% | 0.45 |

## 5. Cost, and what is not run

About 12 CPU-minutes a run: development 5 runs, about 60 CPU-minutes; reported 5 to 10 runs, 60 to 120 CPU-minutes; in all 30 to 50 minutes of wall time. **Not run:** `hold.py` (quoted); longer runs; more cells; gaps beyond 8 in teaching or scoring; the task byte and the modulus of the first family; a comparator by back-propagation through time.
