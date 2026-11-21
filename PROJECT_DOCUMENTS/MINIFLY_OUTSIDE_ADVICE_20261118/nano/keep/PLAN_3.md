# Keeping, third plan: a strength that is closed until its line is met in the cell's youth (a candidate of my own). 20 November 2026

Line K1 (`nano/keep`); authorisation and level as in `PLAN_1.md`. Written after the second plan's development runs (seed 1) were read and while its five-seed runs were in progress and unread; before any run of this plan (one trial of 150,000 lines, made while the option was being written and before the option had its present form, showed only that the program ran; its figures are not used).

**Where the second plan's development left it** (seed 1, one old line in 32, addition / subtraction at the end): joining with no law 14 / 84; the best age law 35 / 50; the program's choice 67 / 75 (median of five seeds). A law that stiffens the old cells enough to protect them also slows what the system can learn afterwards. And with no old line even the program's choice keeps 4.4%, because the new cells speak on addition lines.

**Question.** Can a rule on a cell's own quantities (i) keep the old cells from being changed by lines of a kind they did not meet when they were young, and (ii) keep the cells that join later silent on lines of a kind they have not met, so that something of addition is kept with no old line at all?

**The rule.** The candidate is joining with no law (`PLAN_2.md`, `law=0`, H0 = 128 of 384 cells, a steady rate) with one change. Each strength from a line is closed at the cell's birth. While the cell is younger than y learned bytes, every strength whose line is active opens (and stays open). After that, at a byte where any closed strength of the cell has its line active, the cell is silent and unchanged by that byte (no step, and no fading of the strengths to and from it, as for a cell that has not joined). The rule uses the cell's age and one mark for each of its own strengths; nothing in it refers to the task. A cell that joined during addition has never met the line "the operator three to five bytes back is `-`", and is silent at the answer positions of subtraction lines; a cell that joined during subtraction and met no addition line while young is silent at the answer positions of addition lines.

**Hypothesis (one).** With this rule the cells settle where subtraction is written (in the cells that join after the change) and addition is kept with no old line. **Refuted if** with no old line the median of five seeds has addition below 20% at the end (the plain system: 1.8; the program's choice: 4.4).

**The learner.** `kage.py` with `young=y` (numpy, the reference), `kage.c` (compiled); `check_kage.py` has two cases with the rule and passes (1e-15). 384 cells in between; 64,910 numbers that learn, as in the second plan; in addition one open-or-closed mark for each of the 53,760 strengths from lines, set once and never by the mismatch, and each cell's age.

**Development (seed 1):** y in {4,500, 18,000, 90,000} learned bytes (500, 2,000 and 10,000 lines), with no old line and with one in 32: 6 runs. **Chosen:** the y with the largest sum, over the two conditions, of the smaller of the two scores at the end.

**Arms on the five reported seeds:** plain; the program's choice; joining with no law (second plan); the candidate at the chosen y; each with no old line and with one in 32. **Measures:** never-shown addition and subtraction at the end; addition after its own phase; and, as a measure only, the number of joined cells that may speak at the answer positions of never-shown addition lines, of subtraction lines, and of both.

**Predictions.**
- C1 (0.55): no old line: addition at the end 20% or more (median).
- C2 (0.30): no old line: addition at the end 50% or more (median).
- C3 (0.65): subtraction at the end is 10 points or more below the program's choice (74.0 and 75.4) in both conditions: the old cells are silent on subtraction lines and the cells that can learn it arrive one by one.
- C4 (0.40): one old line in 32: addition at the end 50% or more (median).
- C5 (0.20): one old line in 32: addition 60% or more and subtraction 65% or more (medians).
- C6 (0.70): addition after its own phase is within 5 points of joining with no law (the rule changes little while there is one task).

What I do not know and the runs will show: how much the answering cell's constants, which subtraction goes on changing, cost addition when nothing else of addition is touched.

**Cost.** 6 development and 10 reported runs of about 60 CPU seconds.

**Not run.** The rule together with an age law; the rule without joining (then nothing could learn subtraction); a strength that opens later than youth; a second change of task; the return of addition when taught again.

*Added at once, before the development runs.* The command that committed this plan also made a second trial, of 30,000 lines (20,000 of addition, 10,000 of subtraction, y = 4,500, no old line), to see that the new measure ran. Its figures, which I saw after the text above was written: addition 51% after its phase and 6% after subtraction, subtraction 8%; 295 cells speaking on addition lines only, 85 on subtraction lines only, 4 on both. So in a short life addition was not kept although the cells were sorted. The predictions above stand as written; this trial lowers my own expectation for C1.

## Outcome (entered 20 November, after all runs; `table3_dev.txt`, `table3.txt`, `res/c2_d3_*`, `res/c2_r3_*`, `log/diag3_*.txt`)

**Development, seed 1** (addition after its own phase 88.6 in every run; addition / subtraction at the end):

| y, learned bytes | no old line | one old line in 32 | sum of the smaller scores |
|---|---|---|---|
| 4,500 | 58.2 / 76.6 | 68.0 / 86.2 | 126.2 |
| 18,000 | 38.2 / 75.8 | 62.0 / 81.0 | 100.2 |
| 90,000 | 6.0 / 69.4 | 47.2 / 59.0 | 53.2 |

Chosen by the plan's rule: y = 4,500 (500 lines).

**Five reported seeds** (73129, 93683, 116089, 406364, 545353; median). The first six arms are those of `PLAN_1.md` and `PLAN_2.md`.

| Arm | Addition after its own phase | Addition at the end | Subtraction at the end |
|---|---|---|---|
| plain, no old line | 84.8 | 2.0, 1.6, 1.8, 1.4, 2.0; **1.8** | 86.0 |
| plain, one old line in 32 | 84.8 | 20.8, 12.2, 15.6, 17.2, 9.8; **15.6** | 85.8 |
| the program's choice, no old line | 85.4 | 5.0, 3.0, 3.2, 4.8, 4.4; **4.4** | 74.0 |
| the program's choice, one in 32 | 85.4 | 75.2, 62.8, 67.0, 64.4, 67.8; **67.0** | 75.4 |
| joining, no law, no old line | 82.2 | 2.0, 1.8, 1.6, 1.6, 2.0; **1.8** | 87.4 |
| joining, no law, one in 32 | 82.2 | 13.0, 19.2, 12.6, 16.0, 14.2; **14.2** | 86.4 |
| closed until met in youth (y = 4,500), no old line | 78.4, 81.4, 85.2, 82.2, 86.6; 82.2 | 76.8, 30.6, 74.6, 16.2, 81.2; **74.6** | 61.0, 61.2, 76.2, 89.4, 86.4; **76.2** |
| closed until met in youth, one in 32 | the same | 62.8, 50.0, 58.4, 37.4, 54.4; **54.4** | 29.4, 84.8, 85.6, 87.2, 87.6; **85.6** |

Cells that may speak at the answer positions of never-shown lines, at the end, every seed: with no old line 299 on addition lines only, 85 on subtraction lines only, none on both; with one old line in 32, 299 on addition lines only and 85 on both.

1. **With no old line the candidate keeps addition at 74.6% at the median, where the plain system keeps 1.8% and the program's choice 4.4%, and learns subtraction to 76.2%.** The hypothesis is not refuted (the line was 20%). But the scatter is wide: 16.2 and 30.6 on two seeds of five, 74.6 to 81.2 on three. The median alone would mislead.
2. **During the second phase addition swings without any addition line being read** (for instance seed 73129: 78, 5, 10, 40, 77 at the readings; seed 406364: 82, 0, 0, 7, 16).
3. **The whole of what is lost with no old line is in the answering cell's constants.** A look inside on three seeds (`diag3.py`; the end state with single parts put back as addition left them): with the 14 constants put back, addition is 88.8 (after its phase 88.6; at the end 58.2), 82.8 (82.2; 16.2) and 81.8 (81.4; 30.6). The old cells' own strengths changed over the second phase by 0.0005 to 0.0008 of their size and the answering strengths from them by 0.007 to 0.008 (the two bytes at the change before the first `-` arrived); the constants changed by 1.0 to 2.0 times their size. So the cells settled where subtraction is written, and nothing that addition wrote in them was touched; the 14 constants are the one thing still shared, and subtraction moves them.
4. **With one old line in 32 the candidate keeps 54.4%, less than with none and less than the program's choice (67.0), though subtraction is higher (85.6 against 75.4).** A cell that joins during subtraction and meets some addition lines in its youth opens some of its strengths to them: all 85 late cells may then speak on addition lines, as the program's new cells do. On one seed (73129) subtraction stayed at 29.4. The level to reach, as the brief defines it (67% with one old line in 32), is **not reached** in that condition; it is passed in the condition with no old line at the median and not on every seed.
5. **A short window is what works.** At y = 18,000 and 90,000 the cells that are young at the change (one and four of them) meet both kinds of line and speak on both; on seed 1 four such cells bring addition to 6%. In a short trial (`PLAN_3.md`, the note above) the same four cells were the whole of the loss.

**What the rule rests on, and its limit (my reading).** It sorts the cells because the two tasks differ in a byte that is on the lines at every learned byte (the operator stays in the ten delay lines through the line and the first two bytes of the next). The rule itself names no byte and no task, but a change of task that brought no line never met before would not be seen by it. That case was not run. The rule is also a strong one: an old cell takes no part at all in a task that brings a new line, so subtraction is learned from nothing by the cells that join later, without anything addition formed; here it still reached 76.2 and 85.6.

**Predictions.** C1 (0.55) held: 74.6 (three seeds of five above 20... four of five: 76.8, 30.6, 74.6, 81.2). C2 (0.30) held at the median (74.6), on three seeds of five. C3 (0.65) failed: subtraction 76.2 and 85.6 against 74.0 and 75.4. C4 (0.40) held, barely: 54.4 (four seeds of five at 50 or more). C5 (0.20) failed: addition 54.4. C6 (0.70) held: 82.2 against 82.2.

**Not run**, as planned: the rule with an age law; a second change of task; the return of addition. **Next** (`PLAN_4.md`): the answering cell's constants.
