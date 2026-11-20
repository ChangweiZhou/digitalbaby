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
