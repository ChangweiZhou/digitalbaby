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
