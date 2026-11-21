# Keeping, second plan: age (candidate a). 20 November 2026

Line K1 (`nano/keep`); the same authorisation and level as `PLAN_1.md`. Written before any run of this plan; the first plan's runs were in progress and unread.

**Question.** If cells in between join over the whole life at a steady rate, and a cell's steps shrink with its age, do the cells themselves settle where subtraction is written (in the young cells), with the program knowing nothing of when the task changes, and is addition kept as the program's choice keeps it (`reserve=128`: 67.0% with one old line in 32, 4.4% with none)?

**Hypothesis (one).** Steps that shrink with age leave what the old cells wrote in place while the young cells take the new task, so that with one old line in 32 much more of addition is kept than in the plain system (15.6%). **Refuted if**, at the setting chosen on seed 1, the median of five seeds with one old line in 32 has addition below 30% at the end, or subtraction below 50%.

**The learner** (`kage.py`, numpy, the reference; `kage.c` through `fastage.py`, compiled; `check_kage.py` compares every strength of the two on ten cases, agreement to 1e-15, and compares `kage.py` with all cells present and no law against `cells2.py`: 3e-15).
The plain system with 384 cells in between in all (as many as the program's choice has: 256 + 128). `H0` cells are present from the first byte; the others join one by one, cell H0+i at learned byte (i+1) * 5,400,000 / (385 - H0): a steady rate over the 600,000 lines, fixed before the life begins. Nothing in the program refers to the change of task. Before it joins a cell is silent and unchanged; it joins with its birth strengths (answering strength nothing).
A joined cell of age a (learned bytes since it joined) has a rate r(a): `exp`: exp(-a / tau); `hyp`: 1 / (1 + a / tau). The rate multiplies (i) the cell's own steps on its strengths from lines and its threshold, (ii) the answering cell's step on the strength from that cell, (iii) the step of the return strength to that cell, and (iv) the fading of (ii) and (iii). (iv) is needed: a strength that goes on fading at 1e-5 a byte while its steps have stopped is gone after 1.8 million bytes whatever was learned (the error recorded in `loop_small_case/PLAN_4.md`). The answering cell's constants change as in the plain system. Everything else (eta1 1/4, eta2 1/16, lambda 1e-5, the mismatch, the relay over learned return strengths) is the plain system's.
Numbers that learn: 64,910 (53,760 strengths from lines, 384 thresholds, 5,376 answering strengths, 14 constants, 5,376 return strengths), as in the program's-choice arm; 43,278 in the plain system. Each cell has in addition its age, which is not learned.

**One change.** The candidate differs from the plain system in the age mechanism, which has two parts that cannot be separated in one direction (steps that shrink with age without joining stop all learning). The other direction is run as a control: joining with no law (`law=0`).

**Development (seed 1), one old line in 32:** H0 in {1, 128}; law in {exp, hyp}; tau in {30,000, 100,000, 300,000, 1,000,000} learned bytes; and `law=0` for both H0: 18 runs. **Chosen**: the setting with the largest value of the smaller of the two scores at the end. That setting is then run on seed 1 with no old line, and both on the five reported seeds.

**Arms on the five reported seeds** (`plan=add:400000,sub:200000`): plain and the program's choice, with no old line and with one in 32 (run under `PLAN_1.md`); the candidate at the chosen setting, no old line and one in 32; joining with no law at the chosen H0, no old line and one in 32. **Measures:** never-shown addition and subtraction at the end; addition after its own phase.

**Predictions.**
- A1 (0.60): at the chosen setting addition after its own phase is at least 5 points below the plain system's 84.2 (median).
- A2 (0.85): with no old line the candidate ends with addition at 10% or less (median): the young cells speak on addition lines, as the program's new cells do.
- A3 (0.45): with one old line in 32 the candidate ends with addition at 40% or more (median).
- A4 (0.20): with one old line in 32, addition at 60% or more and subtraction at 65% or more (medians): the level of the program's choice within a few points.
- A5 (0.50): joining with no law, one old line in 32, ends with addition below 25% (median).

**Cost.** A run is 5.4 million bytes with up to 384 cells: 80 to 120 CPU seconds expected. Development 19 runs, reported 20 runs: about an hour of CPU, a quarter of that on the clock.

**Not run.** Other laws; a rate that differs between a cell's own steps and the answering cell's; more or fewer cells; a second change of task; silent cells with age; the relay over two steps.

## Outcome (entered 20 November, after all runs; `table2_dev.txt`, `table2.txt`, `res/c2_d2_*`, `res/c2_r2_*`)

`kage.py` with all 256 cells present and no law gives the plain system's readings on seed 1 (71, 74, 72, 77, 75, 79, 82, 83; 81.5 over the last three), compiled.

**Development, seed 1, one old line in 32** (addition after its own phase | addition / subtraction at the end):

| | 128 cells at birth | 1 cell at birth |
|---|---|---|
| joining, no law | 88.6 \| 14.0 / 84.4 | 78.4 \| 12.8 / 87.6 |
| exp, tau 30,000 | 0.0 \| 0.0 / 0.0 | 0.0 \| 0.0 / 0.0 |
| exp, tau 100,000 | 36.0 \| 11.4 / 4.2 | 0.0 \| 0.0 / 0.0 |
| exp, tau 300,000 | 72.6 \| **35.4 / 50.0** | 66.6 \| 18.6 / 38.8 |
| exp, tau 1,000,000 | 81.2 \| 21.4 / 59.4 | 80.4 \| 22.4 / 75.2 |
| hyp, tau 30,000 | 68.2 \| 25.2 / 47.0 | 49.0 \| 6.6 / 24.4 |
| hyp, tau 100,000 | 77.4 \| 18.6 / 61.8 | 77.4 \| 11.6 / 62.0 |
| hyp, tau 300,000 | 81.4 \| 5.8 / 62.2 | 81.4 \| 6.4 / 70.0 |
| hyp, tau 1,000,000 | 82.8 \| 4.8 / 63.6 | 82.4 \| 6.6 / 83.6 |

Chosen by the plan's rule: 128 cells at birth, exp, tau 300,000 (smaller score 35.4). On seed 1 with no old line it gives 7.6 / 43.2; joining with no law gives 1.4 / 88.0.

**Five reported seeds** (73129, 93683, 116089, 406364, 545353; median). Reference arms from `PLAN_1.md`.

| Arm | Addition after its own phase | Addition at the end | Subtraction at the end |
|---|---|---|---|
| plain, no old line | 84.8 | 2.0, 1.6, 1.8, 1.4, 2.0; **1.8** | 86.0 |
| plain, one old line in 32 | 84.8 | 20.8, 12.2, 15.6, 17.2, 9.8; **15.6** | 85.8 |
| the program's choice, no old line | 85.4 | 5.0, 3.0, 3.2, 4.8, 4.4; **4.4** | 74.0 |
| the program's choice, one in 32 | 85.4 | 75.2, 62.8, 67.0, 64.4, 67.8; **67.0** | 75.4 |
| joining, no law, no old line | 78.4, 81.4, 85.2, 82.2, 86.6; 82.2 | 2.0, 1.8, 1.6, 1.6, 2.0; **1.8** | 85.2, 86.2, 91.4, 88.0, 87.4; 87.4 |
| joining, no law, one in 32 | the same | 13.0, 19.2, 12.6, 16.0, 14.2; **14.2** | 86.4, 84.4, 87.2, 86.0, 88.8; 86.4 |
| age (exp, 300,000), no old line | 73.4, 71.4, 68.0, 75.0, 73.8; 73.4 | 7.6, 4.0, 6.2, 5.2, 8.6; **6.2** | 57.4, 50.6, 53.2, 50.6, 54.6; 53.2 |
| age (exp, 300,000), one in 32 | the same | 37.8, 27.8, 19.2, 33.6, 40.2; **33.6** | 58.2, 51.6, 61.8, 51.0, 56.0; 56.0 |

1. **Age keeps twice what the plain system keeps and half of what the program's choice keeps**, with one old line in 32: 33.6% of never-shown addition against 15.6 and 67.0. The level to reach is not reached.
2. **It pays for this three times**: addition is learned less well in its own phase (73.4 against 84.8), subtraction is learned less well (56.0 against 85.8; the program's choice 75.4), and the table of development shows the trade to be general: every law that kept more than 20% of addition held subtraction below 62%, and every law that let subtraction pass 62% kept less than 23% (the laws with tau 30,000 to 100,000 under exp stop learning altogether).
3. **Joining alone does nothing for keeping** (14.2 against the plain system's 15.6) and costs nothing in learning (82.2 and 86.4).
4. **With no old line nothing is kept** under any arm: 6.2 under age.
5. By the plan's own criterion the hypothesis is **not refuted** (addition 33.6, at or above 30; subtraction 56.0, at or above 50), on two of five seeds addition is below 30. I would not rest anything on that margin.

My reading: one clock for each cell protects the old cells only by making every cell stiff soon after it joins; the cells that are young at any moment are then too few to learn a task quickly (one joins every 21,000 bytes), and those that were young at the end of addition are overwritten. The program's choice differs in two things that age does not supply: all 128 new cells are there at the change, and the old ones are exactly still from that moment and not before.

**Predictions.** A1 (0.60) held: 73.4, 11 points below. A2 (0.85) held: 6.2. A3 (0.45) failed: 33.6. A4 (0.20) failed. A5 (0.50) held: 14.2.

**Not run**, as planned: other laws; a rate that differs between a cell's own steps and the answering cell's; other numbers of cells; a second change of task.
