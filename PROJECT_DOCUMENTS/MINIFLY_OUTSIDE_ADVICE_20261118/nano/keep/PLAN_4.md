# Keeping, fourth plan: the answering cell's constants take steps that shrink with its age. 20 November 2026

Line K1 (`nano/keep`); authorisation and level as in `PLAN_1.md`. Written after the third plan's outcome was entered; before any run of this plan.

*A correction of wording in `PLAN_3.md`, outcome, prediction C1:* four seeds of five are at 20% or more (76.8, 30.6, 74.6, 81.2), one below (16.2).

**Where the third plan left it.** With strengths closed until met in youth, the cells sort themselves (299 speak on addition lines only, 85 on subtraction lines only) and nothing addition wrote in them is touched; with no old line addition is kept at 74.6% at the median and at 16 and 31 on two seeds; with the 14 constants of the answering cell put back, addition is at its own level on the three seeds looked at. The constants are the one thing the two tasks still share.

**Question.** If the answering cell's steps on its constants shrink with its own age, is addition kept at its own level with no old line on every seed, and is subtraction still learned (the cells that join can do what the constants did)?

**The rule (one change from the third plan's candidate).** The answering cell's step on each of its 14 constants is multiplied by exp(-n / t), n the number of bytes it has learned from. Nothing else changes: joining with no law (128 of 384 cells at birth), y = 4,500. The rule uses the answering cell's own age and nothing about the task.

**Hypothesis (one).** The constants are the remaining shared writing; once they are still, addition is kept with no old line. **Refuted if** with no old line the median of five seeds has addition at the end below 70%, or any seed below 50%.

**The learner.** `kage.py` with `young=4500 blaw=exp btau=t`; compiled `kage.c`; `check_kage.py` (thirteen cases, two with this option; agreement to 2e-15). 64,910 numbers that learn.

**Development (seed 1):** t in {10,000, 100,000, 1,000,000} learned bytes, no old line and one in 32: 6 runs. **Chosen:** as in the third plan, the largest sum over the two conditions of the smaller of the two scores at the end.

**Arms on the five reported seeds:** the reference arms and the third plan's candidate (run already); this candidate, no old line and one in 32. **Measures:** as in the third plan; and the look inside (`diag3.py`) on the seed with the lowest addition.

**Predictions.**
- D1 (0.65): no old line: addition at the end 70% or more at the median.
- D2 (0.45): no old line: every seed 60% or more.
- D3 (0.55): no old line: addition at the end within 3 points of addition after its own phase, at the median.
- D4 (0.60): no old line: subtraction at the end 65% or more (median).
- D5 (0.35): one old line in 32: addition 60% or more (median). (The late cells that open to addition lines remain; this plan does nothing about them.)
- D6 (0.70): addition after its own phase within 3 points of the third plan's 82.2 (median).

**Cost.** 6 development and 10 reported runs of about 70 CPU seconds; one or two looks inside.

**Not run.** Other laws for the constants; constants removed altogether; a second change of task; the return of addition; a change of task that brings no new line.

## Outcome (entered 20 November, after all runs; `table4_dev.txt`, `table4.txt`, `res/c2_d4_*`, `res/c2_r4_*`, `log/diag4_*.txt`)

**Development, seed 1** (addition after its own phase | addition / subtraction at the end):

| t, learned bytes | no old line | one old line in 32 | sum of the smaller scores |
|---|---|---|---|
| 10,000 | 90.4 \| 90.2 / 81.4 | 62.6 / 91.8 | 144.0 |
| 100,000 | 86.6 \| 86.6 / 72.6 | 62.4 / 88.0 | 135.0 |
| 1,000,000 | 89.8 \| 0.0 / 81.2 | 66.6 / 87.0 | 66.6 |

Chosen by the plan's rule: t = 10,000 (the constants are still after about ten thousand lines).

**Five reported seeds** (73129, 93683, 116089, 406364, 545353; median).

| Arm | Addition after its own phase | Addition at the end | Subtraction at the end |
|---|---|---|---|
| plain, no old line | 84.8 | 2.0, 1.6, 1.8, 1.4, 2.0; **1.8** | 86.0 |
| plain, one old line in 32 | 84.8 | 20.8, 12.2, 15.6, 17.2, 9.8; **15.6** | 85.8 |
| the program's choice (`reserve=128`), no old line | 85.4 | 5.0, 3.0, 3.2, 4.8, 4.4; **4.4** | 74.0 |
| the program's choice, one in 32 | 85.4 | 75.2, 62.8, 67.0, 64.4, 67.8; **67.0** | 75.4 |
| third plan (closed until met in youth), no old line | 82.2 | 76.8, 30.6, 74.6, 16.2, 81.2; **74.6** | 76.2 |
| third plan, one in 32 | 82.2 | 62.8, 50.0, 58.4, 37.4, 54.4; **54.4** | 85.6 |
| this plan (and constants that become still), no old line | 82.0, 84.2, 83.4, 84.2, 85.0; 84.2 | 83.4, 85.6, 81.2, 85.4, 86.6; **85.4** | 85.2, 87.2, 80.0, 73.2, 78.8; **80.0** |
| this plan, one in 32 | the same | 68.8, 56.8, 69.0, 51.2, 51.2; **56.8** | 85.0, 84.8, 83.6, 85.8, 74.6; **84.8** |

Readings of addition during the second phase, no old line, every 50,000 lines: 83, 83, 83, 83 (seed 73129); 86, 86, 86, 86; 81, 81, 81, 81; 85, 85, 85, 85; 87, 87, 87, 87: it does not move. At the end 299 cells may speak on addition lines only and 85 on subtraction lines only on every seed.

1. **With no old line, addition is kept at its own level on every seed** (81.2 to 86.6 at the end; 82.0 to 85.0 after its own phase), **and subtraction is learned to 80.0%** (73.2 to 87.2). The plain system keeps 1.8% and the program's choice 4.4% in that condition; with one old line in 32 the program's choice keeps 67.0% and learns subtraction to 75.4%. The hypothesis is not refuted. For comparison, the earlier report's system that read both tasks mixed from the start for the same 600,000 lines had 75.6 and 81.2 (`loop_small_case/REPORT.md`, section 9).
2. **With one old line in 32 the candidate keeps less (56.8) than with none**, and less than the program's choice (67.0): the level set in that condition is not reached. The look inside (seed 406364, `log/diag4_406364_reh32.txt`): with the cells that joined after the change made silent, addition is 81.0 (at the end 51.2; after its phase 84.2). So the loss is the voice on addition lines of late cells that met a few addition lines in their youth and opened to them; with no old line these cells stay closed to addition and the loss does not arise.
3. In the look inside with no old line (seed 116089) the 2 points between 83.4 and 81.2 return when the answering strengths from the old cells are put back: they are the two bytes at the change, read before the first `-` arrived.
4. Addition after its own phase is no lower than without the rule (84.2; joining with no law 82.2; plain 84.8).

**What this is and is not (my reading).** The two rules together make the system into two sets of cells that share nothing that still changes: which set speaks is settled by a line the old cells never met. Nothing in the program refers to the task or to the moment of change, and the keeping is complete here. But (i) it rests on the second task bringing a line never met before (the operator byte), and a change that brought none would not be seen; not run. (ii) Nothing addition formed is used for subtraction; subtraction is learned from nothing by 85 cells, and reaches 80.0 where the plain system's 256 cells reach 86.0. (iii) The clock of joining was set with the length of the life known (one cell every 21,000 bytes); the next plan removes it. (iv) Five seeds, one pair of tasks, one order, development level, repeated by no one.

**Predictions.** D1 (0.65) held: 85.4. D2 (0.45) held: lowest seed 81.2. D3 (0.55) held: 85.4 against 84.2. D4 (0.60) held: 80.0. D5 (0.35) failed: 56.8. D6 (0.70) held: 84.2 against 82.2.

**Not run**, as planned: other laws for the constants; a second change of task; the return of addition; a change of task that brings no new line.
