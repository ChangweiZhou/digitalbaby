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

## Outcome (entered 20 November, after all runs; `table5.txt`, `res/c2_d5_*`, `res/c2_r5_*`)

**Development, seed 1, no old line** (addition / subtraction at the end; cells joined by line 100,000 of addition / by line 400,000 / by the end):

| theta | delta 0.25 | delta 0.5 |
|---|---|---|
| 150 | 86.0 / 24.2; 193 / 193 / 208 | 81.4 / 0.2; 151 / 151 / 157 |
| 250 | 86.4 / 11.4; 118 / 118 / 128 | 82.0 / 0.0; 91 / 91 / 96 |
| 400 | 80.2 / 0.0; 74 / 74 / 81 | 66.2 / 1.6; 57 / 57 / 61 |

Chosen by the plan's rule: theta 150, delta 0.25 (smaller score 24.2). With one old line in 32 on seed 1: 81.2 / 29.8. The setting failed on the development seed; it was run on the reported seeds all the same, so that its size is known.

**Five reported seeds** (73129, 93683, 116089, 406364, 545353; median).

| Arm | Addition after its own phase | Addition at the end | Subtraction at the end |
|---|---|---|---|
| fourth plan (clock), no old line | 84.2 | 83.4, 85.6, 81.2, 85.4, 86.6; **85.4** | 85.2, 87.2, 80.0, 73.2, 78.8; **80.0** |
| joining on the excess of mismatch, no old line | 94.2, 89.2, 93.0, 85.0, 80.8; 89.2 | 94.2, 88.8, 92.0, 83.6, 81.2; **88.8** | 11.0, 9.0, 9.2, 20.2, 17.2; **11.0** |
| joining on the excess of mismatch, one in 32 | the same | 66.4, 68.6, 67.0, 19.4, 55.4; **66.4** | 7.6, 22.2, 12.0, 8.4, 26.4; **12.0** |

Cells joined by line 100,000 / by line 400,000 / by the end, no old line: 193/193/207, 195/195/209, 192/192/204, 196/196/212, 194/194/209. Subtraction 50,000 lines into its phase: 9.2, 13.6, 1.8, 6.8, 10.4.

1. **The hypothesis is refuted in its second half.** The excess of mismatch is large at birth (about 194 cells join within the first 100,000 lines and none in the following 300,000, as hoped) but small at the change of task: 12 to 16 cells join there, and subtraction is not learned (11.0%).
2. **Why (my reading, from the numbers).** The answering cell's mismatch never falls far: four of the nine bytes it names on a line are digits of the two numbers, which nothing can foretell, so its mean squared mismatch stays near 0.4 when everything learnable is learned. At the change the easy bytes of the new lines (the operator, `=`, the leading 0, the end) are learned by the first few cells, and what remains above the usual size is the two hard digits of the answer: about 0.2 on 0.4, less than the slow mean takes up in a few thousand lines. A signal of this kind measures the bytes that cannot be foretold as much as the ones not yet learned.
3. **Addition is kept all the same** with no old line (88.8), and is learned better in its own phase than with the clock (89.2 against 84.2, with about 194 cells that all join early). That part I had not predicted.
4. **The life continued, fourth plan's candidate (clock), no old line**: addition / subtraction at the end of addition, of subtraction, and of 100,000 further lines of addition:

| seed | after addition | after subtraction | after addition again |
|---|---|---|---|
| 73129 | 82.0 / 1.4 | 83.4 / 85.2 | 85.6 / 84.6 |
| 93683 | 84.2 / 0.2 | 85.6 / 87.2 | 85.0 / 87.2 |
| 116089 | 83.4 / 0.0 | 81.2 / 80.0 | 80.6 / 80.2 |
| 406364 | 84.2 / 0.0 | 85.4 / 73.2 | 84.4 / 73.4 |
| 545353 | 85.0 / 0.0 | 86.6 / 78.8 | 79.4 / 79.8 |

Subtraction is unchanged by 100,000 further lines of addition (median 80.2 after, 80.0 before; the plain system: 2.8). Addition does not gain from them: median 84.4 against 85.4, and on one seed it falls by 7 points (86.6 to 79.4), which I have not looked into.
The continued life of this plan's candidate was **not run**, the candidate having failed.

**Predictions.** E1 (0.60) held: 88.8. E2 (0.55) failed: 11.0. E3 (0.60) failed: 9.2. E4 (0.50) failed in its first half (12 to 16 cells join after the change) and held in its second (none between lines 100,000 and 400,000). E5 (0.30) held as written (66.4), in a system that does not learn subtraction; I count it as empty. E6 (0.80), on the fourth plan's candidate only: subtraction held (80.2 against 80.0); "addition no lower than before" failed (84.4 against 85.4).

**Not run**, as planned: steps that shrink with accumulated use; other rates of the two running means; subtraction first.
