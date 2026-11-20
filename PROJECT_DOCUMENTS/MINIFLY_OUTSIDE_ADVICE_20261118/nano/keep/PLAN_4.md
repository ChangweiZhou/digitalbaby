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
