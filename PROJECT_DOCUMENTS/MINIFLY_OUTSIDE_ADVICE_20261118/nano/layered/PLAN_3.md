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

## Outcome (entered 20 November after the development runs; nothing above was changed)

Development seed 1, 2,400,000 lines, %, mean of the last three readings (`res/lay_d3_*.json`); by place: first/second/third answer byte right on never-shown sums at the gap of 8 letters at the last reading.

| etaA, nB (etaB = 1) | never shown at gaps 0 / 4 / 8 | shown at gaps 0 / 4 / 8 | by place at 8 letters |
|---|---|---|---|
| 1, 32 | 9.4 / 10.5 / **8.3** | 12.5 / 11.7 / 10.1 | 97 / 70 / 15 |
| 1, 128 | 4.0 / 4.4 / 4.2 | 4.4 / 4.4 / 3.3 | 96 / 46 / 2 |
| 1/4, 128 | 4.3 / 2.3 / 0.5 | 3.7 / 1.9 / 0.2 | 93 / 36 / 0 |
| 4, 32 | 5.1 / 5.3 / 5.2 | 4.4 / 5.0 / 4.6 | 98 / 51 / 7 |
| cells A never change (0, 32) | 0.7 / 0.7 / 0.6 | 1.2 / 1.2 / 1.3 | 67 / 14 / 11 |
| `hold.py`, one crossing (quoted, PLAN_11 at 0ba9f241) | 10 / 10 / 8 | not quoted | 97 / 76 / 16 |

Chosen by the rule: etaA = 1, nB = 32, with 8.3% at the gap of 8 letters, rising slowly (never-shown mean over gaps 5% at the first reading, 10% at the last). About 13 CPU-minutes a run.

**The hypothesis is refuted as registered** (8.3% at the gap of 8 letters, where 15% or less refutes). **The reported seeds were not run**, as the plan lays down (the line was 30%).

The layered system without a crossing stands where `hold.py` with one crossing stood on this text, to the point: 8 against 8 at the gap of 8 letters, and the same picture by place (hundreds 97, tens 70 to 76, units 15 to 16). The units digit is hardly above a guess, and it is so at the gap of none as well (9.4% whole answers there), although lines with no letters are a ninth of the text and are the lines of the second plan's text, where this system reached 86%. So the exact units digit of the first number is not held once letters can follow it, by either design, in this time.

| | Prediction | Outcome |
|---|---|---|
| G1 | at least 30% at the gap of 8 letters (0.55) | **failed** (8.3) |
| G2 | at least 50% (0.40) | **failed** |
| G3 | the gap of 8 within 15 points of the gap of none (0.50) | held only in the empty sense that every gap is equally low |
| G4 | units digit right on at least 50% at the gap of 8 letters (0.55) | **failed** (15) |
| G5 | never-shown within 10 points of shown (0.80) | held (9.4 against 11.4), of little weight at this level |
| G6 | cells A never changing at most 5% (0.85) | held (0.6) |
| G7 | reported seeds | **not run** |

I was wrong about this text by a wide margin: the second plan's success had made me expect that a gate which can see which digit is before it would be enough. A reading of mine, not tested: a cell that is to hold the first number's units digit must keep its gate shut on every one of the pairs and triples of bytes that the letters can form (with two delay lines a gate's drive is a sum over three positions, so the letters present many more combinations than the nine of the first plan's texts), and until it does, its trace of that digit has faded by the time the answer is named; in the second plan the same digit had to last six bytes of fixed kinds. Whether longer runs, a higher birth threshold of the gates, or a gate on what a cell gives out (the fourth plan) changes this is not known from these runs. **Not run:** the reported seeds; longer runs; more cells; `hold.py` (quoted); a look inside the cells.
