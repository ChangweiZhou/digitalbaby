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

## Outcome (entered 20 November, after all 52 runs; `table1.txt`, `res/c2_d1_*`, `res/c2_r1_*`, `res/c2_r_plain_*`, `res/c2_r_res128_*`, `res/ov_*`)

Two faults of my job launcher were found and corrected before any result was read (the label was not passed to the program; a trailing blank joined two job lines); the runs made under them wrote nothing usable and were discarded.

Never-shown pairs, %, the five reported seeds in the order 73129, 93683, 116089, 406364, 545353, and the median.

| Arm | Addition after its own phase | Addition at the end | Subtraction at the end |
|---|---|---|---|
| plain, no old line | 81.8, 81.8, 86.4, 84.8, 87.6; 84.8 | 2.0, 1.6, 1.8, 1.4, 2.0; **1.8** | 82.0, 89.8, 91.4, 86.0, 86.0; 86.0 |
| plain, one old line in 32 | the same | 20.8, 12.2, 15.6, 17.2, 9.8; **15.6** | 85.6, 85.8, 90.4, 84.2, 85.8; 85.8 |
| the program's choice (`reserve=128`), no old line | 83.6, 81.6, 88.0, 90.6, 85.4; 85.4 | 5.0, 3.0, 3.2, 4.8, 4.4; **4.4** | 74.0, 77.2, 78.6, 72.8, 71.6; 74.0 |
| the program's choice, one old line in 32 | the same | 75.2, 62.8, 67.0, 64.4, 67.8; **67.0** | 78.6, 72.8, 75.4, 73.4, 77.6; 75.4 |
| 256 silent cells, no old line | 71.4, 78.8, 56.6, 47.6, 72.4; 71.4 | 0.4, 2.8, 1.2, 0.4, 0.8; **0.8** | 74.6, 65.4, 40.2, 49.0, 32.6; 49.0 |
| 256 silent cells, one old line in 32 | the same | 7.0, 5.4, 6.4, 5.0, 5.4; **5.4** | 71.4, 45.2, 50.6, 44.2, 64.2; 50.6 |
| 512 silent cells, no old line | 62.4, 74.2, 66.4, 74.6, 84.6; 74.2 | 0.8, 1.0, 0.8, 1.2, 1.0; **1.0** | 67.8, 52.8, 78.0, 76.6, 72.4; 72.4 |
| 512 silent cells, one old line in 32 | the same | 8.0, 7.0, 8.6, 8.8, 6.4; **8.0** | 68.4, 62.0, 76.6, 71.4, 80.8; 71.4 |

The four reference arms repeat the end figures of `loop_small_case/REPORT.md`, section 11, to the digit. (The column "after its own phase" is here the single reading at 400,000 lines.)

Development seed, one old line in 32, addition / subtraction at the end: 256 cells: target 1/4: 10.6 / 66.6; 1/8: 7.0 / 32.6; 1/16: 9.8 / 60.6. 512 cells: 6.0 / 71.8; 4.8 / 57.8; 4.6 / 65.8. No sparser target exceeded the target 1/4 by the plan's rule, so none went to the reported seeds.

Cells active on never-shown addition lines / on subtraction lines / on both, at the end (mean output above 0.05 at the answer positions; one old line in 32; five seeds): 256 cells: 165/173/159, 155/157/142, 160/160/150, 162/164/152, 157/167/149. 512 cells: 396/386/360, 393/376/360, 370/363/332, 365/341/319, 369/366/340. About nine in ten of the cells active on addition lines are also active on subtraction lines; on the development seed the share is 0.82 to 0.86 at the targets 1/8 and 1/16.

**The hypothesis is refuted.** With one old line in 32 silent cells keep 5.4% (256) and 8.0% (512) of never-shown addition; as shares of what they had learned, 0.08 and 0.11, below the plain system's 0.18 (15.6 / 84.8). They keep less than logistic cells, not more.

My reading: a cell being silent at most bytes does not make the two tasks use different cells. Which cells are active is decided by the digits on the lines, and addition and subtraction lines carry the same digits in the same places; the one line that differs (the operator) does not sort the cells. So nearly every cell that answers addition is active and changed on subtraction lines.

**Predictions.** B1 (0.85) held: 0.8 and 1.0. B2 (0.30) failed: 5.4. B3 (0.10) failed: 5.4 and 8.0. B4 (0.60) held: about 0.9. B5 (0.75) held: refuted as defined.

**Not run**, as planned: silent cells together with `reserve`; other step sizes or threshold speeds; more than 512 cells.
