# Keeping, first plan: cells that are silent most of the time (candidate b). 20 November 2026

By the agent session that owns line K1 (`nano/keep`), started by the coordinating session under Mr. Xu's words of 20 November as recorded in `ROADMAP_SECOND_REVISION_20261120.md` ("revise the road map and keep doing some experiments. Spin off some agents to speed up the experiment process"). Development level; no checker gate, seal or review is claimed. Written before any run of this plan.

**Copied** from `loop_small_case/` at commit 260c5d7, unchanged: `cells2.py`, `kern.c`, `kern2.c`, `kern3.c`, `fastk.py`, `fastk2.py`, `fastk3.py`, `.gitignore`. The copy gives 81.5 on seed 1 after 400,000 lines of addition (mean of the last three readings; `fast=1`), as the original does.

**Question.** Cells that are silent below a threshold (`clip=`; `loop_small_case/REPORT.md`, section 13) have a fifth to a quarter of the cells active at a byte. After 400,000 lines of addition and 200,000 of subtraction, do they keep more of addition than logistic cells do, with no old line and with one old line in 32, because fewer cells are shared between the two tasks?

**Hypothesis (one).** With silent cells a smaller share of the cells that answer addition is active, and so changed, on subtraction lines; hence less of addition is overwritten at equal rehearsal. **Refuted if** with one old line in 32 the share of addition kept (addition at the end over addition after its own phase, median of five seeds) is not at least 0.10 above the plain system's share (15.6 / 84.2 = 0.19), in both sizes run.

**The learner.** `cells2.py` as copied, the one change from the plain system being the cell: output = drive cut off at 0 and at 1, each cell moving its threshold by 1e-5 a byte towards a mean output of the target. The settings are those of the eighth plan's reported runs, taken as they are: `clip=1/4 cap=1 eta1=1/64 kap=1e-5`, 256 cells and 512 cells, compiled kernel `fast=2`. Numbers that learn: 43,278 with 256 cells (35,840 strengths from lines, 256 thresholds, 3,584 answering strengths, 14 constants, 3,584 return strengths); 86,542 with 512. The program knows of the task change only through the lines it is given.
On the development seed (1) two sparser targets are also run (1/8, 1/16), with and without one old line in 32. A sparser target goes to the reported seeds only if on seed 1 its smaller score of the two tasks at the end, with one old line in 32, exceeds that of the target 1/4 at the same size.

**Arms, five reported seeds (73129, 93683, 116089, 406364, 545353), `plan=add:400000,sub:200000`.**
1. plain (256 logistic cells): no old line; one old line in 32. (A repetition here of `loop_small_case` runs; 1.8 and 15.6 there.)
2. the program's choice, `reserve=128`: no old line; one in 32. (Repetition; 4.4 and 67.0 there.) The level to reach is 67.0.
3. silent cells, 256: no old line; one in 32.
4. silent cells, 512: no old line; one in 32.

**Measures.** Whole answer right on 500 never-shown pairs of each task, at the end; addition after its own phase. Added measure, for the hypothesis: at the end of the run, on the never-shown lines of both tasks at the three answer positions, the number of cells active (output above 0.05 on average) on addition lines, on subtraction lines, and on both (`overlap.py`; for logistic cells every cell is active on both by construction).

**Predictions.**
- B1 (0.85): with no old line, silent cells keep at most 10% of never-shown addition at the median, in both sizes.
- B2 (0.30): with one old line in 32, 256 silent cells end with addition at 30% or more (median).
- B3 (0.10): with one old line in 32, either size of silent cells ends with addition at 55% or more (median).
- B4 (0.60): more than half of the cells active on addition lines are also active on subtraction lines (the two tasks share nine of ten line positions' digits; my expectation is that the sharing stays large).
- B5 (0.75): the hypothesis is refuted as defined above.

**Cost.** Reference arms: 20 runs of 50 to 75 CPU seconds. Silent cells: 20 runs of 15 to 40 CPU seconds; development 12 runs. Under 30 CPU minutes in all, four cores.

**Not run.** Silent cells together with `reserve`; other thresholds' speeds or step sizes; more than 512 cells; the relay over two steps; anything on language.
