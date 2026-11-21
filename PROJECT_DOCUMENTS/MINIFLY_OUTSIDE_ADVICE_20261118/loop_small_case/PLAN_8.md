# Eighth plan: speed. A compiled kernel, and cells that are silent most of the time (roadmap R2) (20 November 2026)

**Authorisation.** Mr. Xu, 20 November 2026, two messages during this work: "Are you using the slow version or the fast version of the code? Previously fable agents optimized the code to run around 2 minutes. Your code seems rather slow." and "The idea is to utilize sparsity to speed up the code as much as possible, and enable fast feedback loop in our research. So see if this can be done. In general, if you got stuck, figure out the root cause first and act." **Source.** This branch at 8e013380. Development level; a small case; no checker gate.

## 1. Why the runs were slow: two causes

**First cause: a Python loop over every byte.** `cells2.py` and `hold.py` were plain numpy, some twenty array operations a byte, each costing microseconds of overhead. Remedy, done before this plan was written: the inner loops in C (`kern.c`, called through `fastk.py`; option `fast=1`), as the agents did for the tissue. `check_fast.py` runs both on the same seed and lines and compares every strength: they agree to 1e-13 or better on every option in use (`check_fast_output.txt`); the numpy programs remain the reference. Measured on this machine while it was busy: the delay-chain program 100,000 lines in 8.5 s against 52 s; the gated cells 100,000 two-digit lines in 19.5 s against about 206 s, with the same readings (74.4 and 89, 83, 100; 5.0, 2.6 and 96, 37, 8). All reported seeds are run with the kernel from here on; the numpy runs already finished are kept (`res/c2_np_r_*.json`) and compared with it.

**Second cause: work that is dense and need not be.** In the compiled delay-chain program a byte costs about 27,000 arithmetic operations. About 2,600 of them depend on the input, and those already use its sparsity (10 active lines of 140). The rest is dense for two reasons: (a) every strength of the answering cell and every return strength is faded at every byte (7,000 operations; removable exactly, with one scale factor); (b) **every cell is active at every byte**, because a logistic output is never zero, so the answering cell's sum, the relay and every change run over all 256 cells. (b) is the larger part, and no rewriting removes it: the cell itself has to be silent most of the time. That is also what a cell is like, and what the tissue's cells are (output zero below a threshold, the threshold moving with the cell's own activity). But it changes the learner, so it has to pass the end-to-end score.

## 2. The question, the hypothesis, what would refute it

**With cells in between that are silent below a threshold, each keeping its mean output near a target by moving its own threshold, does the loop still answer never-shown two-digit sums, and how much faster is a kernel that works only on the active cells?**

The cell (`clip=p`): output = drive cut off at 0 and at 1; its sensitivity is 1 where the drive lies between 0 and 1 and 0 elsewhere; after every byte its threshold moves by `kap` times (its output less the target p). Everything else is the system of the second plan (256 cells, every line read, the relay with learned return connections).

*Hypothesis.* The score holds (at least 70% of never-shown sums at 400,000 lines on the development seed, where the logistic cell has 81.5%) with at most a quarter of the cells active at a byte, and the active-cell kernel (`fast=2`) is at least three times faster than the dense compiled kernel.

*Refuted if* no setting of the grid reaches 40%, or the gain in speed is below 1.5.

*Added before the grid, after a trial of 3,000 lines made only to test the program:* at the target 1/8, 72 cells were active at a byte and only 13 of them lay between the two cut-offs; the rest sat at the upper one, where a cell neither changes nor gives a graded output. The grid therefore runs two forms of the cell: cut off at 0 and at 1 (the tissue's form), and cut off at 0 only (`cap=0`).

Grid on development seed 1, for both forms: eta1 in 1/64, 1/16, 1/4; target p in 1/16, 1/8, 1/4; kap = 1e-3 (and 1e-4 at the best setting). Then the best setting on the five reported seeds beside the logistic cell (`r_plain`).

## 3. Predictions (mine, before any run with these cells)

| | Statement | Chance |
|---|---|---|
| S1 | some setting of the grid reaches at least 70% on the development seed | 0.60 |
| S2 | at the target 1/8 some eta1 reaches at least 60% | 0.50 |
| S3 | the active-cell kernel is at least 3 times faster a byte than the dense compiled kernel at the target 1/8 | 0.70 |
| S4 | 512 such cells at the target 1/16 reach at least the logistic cell's 81.5% | 0.35 |

## 4. Cost, and what is not run

A run of 400,000 lines should take under a minute. Not run in this plan: silent cells together with sparse reading, rewiring, one sign for each cell, or two steps of relay; silent cells in the system without the delay chain (`hold.py`), where the state of a cell that holds would have to stop changing for sparsity to pay.

## 5. Entered on 20 November after the development runs (seed 1), before any reported seed

Never-shown sums, mean of the last three readings of 400,000 lines; cells active at a byte; CPU seconds of the run (`res/c2_d8*.json`, machine busy).

*The registered grid (kap = 1e-3), 256 cells.* Best: cut off at 0 and 1, target 1/4, eta1 = 1/64: 54.7. At the target 1/8: 39.1 at best. In 14 of the 18 settings the score rises early and then falls (for instance 44, 39, 36, 28, 13, 13, 8, 6). **S1 and S2 fail on the registered grid.**

*Root cause of the fall, tested with a second grid added after the first* (eta1 in 1/256, 1/64; kap in 1e-4, 1e-5): the thresholds were moving as fast as the strengths were learning. With kap = 1e-5 the fall is gone. Best with 256 cells: cut off at 0 and 1, target 1/4, eta1 = 1/64, kap = 1e-5: **61.0** (53 cells active a byte, 9.5 s; over 800,000 lines 61.1). Cut off at 0 only, target 1/8: 56.4 (25 cells active, 6.7 s). The logistic cell: 81.5 (all 256 active; about 35 s with the dense compiled kernel, about 300 s in numpy).

*More cells, added after that* (eta1 = 1/64, kap = 1e-5): 512 cells, cut off at 0 and 1: target 1/16: 66.3 (53 active, 12.5 s); target 1/8: 60.9 (68 active); target 1/4: **81.6** (140 active, 22 s; by place 94, 86, 100). 1,024 cells: target 1/8: 78.1 (183 active, 33 s); target 1/16: 64.0 (112 active). **S4 fails at its target (1/16)**; at the target 1/4, 512 silent cells equal the 256 logistic cells.

Reading (mine). Silent cells learn and answer never-shown sums, but the score follows the number of cells that are active, not the number of cells: about 25 active give 56%, 53 give 61 to 66%, 140 give 82%, where 256 logistic cells give 82%. At equal score the active-cell kernel is about 1.6 times faster than the dense compiled kernel; at a score of about 60% it is 4 to 5 times faster. The hypothesis fails as registered (61% and not 70% with 256 cells); S3 holds (about 5 times at the target 1/8).

**Reported seeds** (73129, 93683, 116089, 406364, 545353), beside the logistic cell (`r_plain`): cut off at 0 and 1, target 1/4, eta1 = 1/64, kap = 1e-5, with 256 cells and with 512 cells.

## 6. The reported seeds, and a reading withdrawn (entered 20 November)

Cut off at 0 and 1, target 1/4, eta1 = 1/64, kap = 1e-5, 400,000 lines (`report4.txt`): 256 silent cells 72.2, 71.9, 54.1, 42.0, 71.3 (median **71.3**; about 50 cells active at a byte; 9.6 CPU-seconds a run); 512 silent cells 62.8, 70.0, 69.5, 74.2, 83.2 (median **70.0**; about 140 active; 22.9 s); 256 logistic cells 84.2 (33 s with `fast=1`).

The reading of section 5, that the score follows the number of active cells and that 512 silent cells equal 256 logistic ones, came from the development seed and is **withdrawn**: on the reported seeds the two sizes are level at about 70%, 14 points below the logistic cells, with a wide scatter over seeds. What stands: silent cells learn and answer never-shown sums; the kernel that works on the active cells alone is 3.4 times faster than the first dense kernel and 1.6 times faster than the dense kernel with lazy fading (`fast=3`, added after this plan was written: 15 s), at that cost in score.
