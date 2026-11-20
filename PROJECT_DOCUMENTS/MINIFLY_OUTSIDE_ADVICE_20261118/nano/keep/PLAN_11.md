# Keeping, eleventh plan: the subtraction score on lines no shown pair writes; means of the last three readings. 20 November 2026

Line K1 (`nano/keep`); authorisation and level as in `PLAN_1.md`. Written after four messages of the coordinating session were read (they had been waiting while the earlier plans ran; I read them only after the report of the line was pushed), and before any run of this plan.

**What the coordinating session told this session** (its words are in its own files; I state what I took from them). (1) A fault in the scoring of subtraction in `cells2.py`, which this line copied unchanged: a subtraction line is written with the larger number first, so a held-out pair (a, b) and the shown pair (b, a) write the same line; about four in five of the 500 held-out subtraction lines are lines that training also writes. Addition is not touched. So every subtraction column of `PLAN_1.md` to `PLAN_10.md` and of `REPORT.md` is a score on mostly shown lines. (2) My figures are single last readings; the mean of the last three readings is to be given beside each. (3) Several errata from two checks of my files. (2) and (3) are answered in `ERRATA.md` from the files that exist; (1) needs runs.

**Question.** On the held-out subtraction pairs whose written line no shown pair writes (about 80 to 100 of 500), is subtraction answered as on all 500, in the plain system, in the program's choice, and in the candidates? For the candidates the question is sharper: cells that open only to lines met in youth might hold shown lines without having learned to subtract.

**What is run** (`clean.py`; no system is changed): the same runs as before, with the same seeds and options, scored at the end on (i) the clean pairs, (ii) the pairs a shown pair also writes, (iii) all 500, and with finer readings early in each phase (`fine=1`), which does not change the lines read. Arms, no old line and one old line in 32: plain; `reserve=128`; the fourth plan's candidate; the seventh plan's candidate (reported seeds; the first, second and fourth also on seeds 1001 to 1005); the other order and the text with no new line of the eighth plan (candidate and plain). 110 runs. Every rerun's end figures must equal the earlier run's; any that do not are reported.

**Hypothesis.** The candidates learned to subtract, not the shown lines. **Refuted if** the seventh plan's candidate's clean subtraction score is more than 10 points below its score on all 500, at the median of the reported seeds, in either condition; or more than 10 points further below than the plain system's is.

**Predictions.**
- K1 (0.65): for every arm the median clean score is within 8 points of the median score on all 500.
- K2 (0.60): the seventh plan's candidate has clean subtraction at 70% or more in both conditions (median, reported seeds).
- K3 (0.60): in the other order (subtraction the first task), the candidate's clean subtraction at the end is within 5 points of the plain system's clean subtraction after its own phase... not available from these runs for the plain system at the end of its first phase; replaced before any run by: the candidate's clean subtraction at the end is 80% or more (median).
- K4 (0.90): every rerun reproduces the earlier end figures to the digit.

**Also from these runs:** the number of lines of the second task read when its never-shown score first reaches 70% at a reading (readings at 2,000-line spacing to 20,000, 10,000 to 100,000, then 50,000), beside the plain system.

**Cost.** 110 runs of 40 to 90 CPU seconds: about 30 minutes on the clock.

**Not run.** The clean score for the arms of the first, second, fifth and ninth plans (their candidates did not keep); a split of the pairs that avoids the fault from the start (it would change every figure of `loop_small_case` and is the coordinating session's to decide).
