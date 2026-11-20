# Kinds of cells, second plan (C6, question 2): cells that change at different rates (20 November 2026)

By the agent session that owns line C6 and the folder `nano/kinds/`. Development level; small cases; no checker gate, seal or review is claimed. The readings are mine. Source of the question, authorisation and source of the programs: as in `PLAN_1.md` (the copy of `loop_small_case/` at 260c5d73 in this folder). Written before any run of these arms beyond the 300-line runs of `check_kinds.py`.

## 1. The question

Mr. Xu's "different types of cells that adjusts around different time clocks" is taken here in one form (the reading is mine): each cell in between changes its own strengths at its own rate, fixed at birth. **Does a spread of such rates over the cells change what is learned, against all cells alike at the same geometric mean rate?**

## 2. The law

Option `clock=B`. At birth each cell j draws u_j uniformly from -1 to 1 (its own stream of random numbers, so that no other number at birth changes); the mean of u over the cells is subtracted; the cell's rate is eta1 times B to the power u_j. The geometric mean of the rates is therefore exactly eta1, and the rates run from about eta1/B to eta1 times B. Every change that a cell makes to its own strengths and thresholds is multiplied by its rate: in `hold.py` both what it receives from the answering cell and what the cells that read it send back to it; in `cells2.py` its strengths from the lines and its threshold. The answering cell and the return strengths are unchanged (rate eta2).

A spread at equal geometric mean has a larger arithmetic mean (B = 2: 1.08; B = 4: 1.35; B = 16: 2.87 times eta1). A control (`clock=B clockam=1`, added by me) gives every cell that arithmetic mean, so that a difference can be assigned to the spread or to the larger mean.

## 3. Testbeds and arms

T1: `python3 hold.py <label> text=m1 gap=8 seed=<seed> eta1=1 cross=1 ret=exact lines=960000 every=20000 fast=1` (score and lines to 90% as in `PLAN_1.md`). T2: `python3 cells2.py <label> seed=<seed> fast=3` (two-digit sums with nine delay cells, eta1 = 1/4; never-shown sums, mean of the last three readings of 400,000 lines).

| Arm | Option | T1 | T2 |
|---|---|---|---|
| `base` | none | the five runs `q1_spread_<seed>` of the first plan (the same command in the same copy; the kernel's additions are off) | run |
| `c2` | `clock=2` | run | run |
| `c4` | `clock=4` (the brief's instance) | run | run |
| `c16` | `clock=16` | run | run |
| `am4` | `clock=4 clockam=1` | run | run |
| `am16` | `clock=16 clockam=1` | not run | run |

No setting is chosen on the development seed: the three values of B are all reported, as a dose. The development seed (1) is run first on both testbeds as a check that the arms run.

**The exact learners.** T1: as in `PLAN_1.md` (17,920 numbers that learn in the cells in between, 4,626 at the answering cell, 4,608 return strengths). T2: class `Sys` of `cells2.py`, the plain system: 256 logistic cells reading 140 lines (ten bytes, fourteen byte values); 36,096 numbers that learn in the cells in between, 3,598 at the answering cell, 3,584 return strengths. The rates are 256 numbers fixed at birth; they do not learn.

## 4. Hypothesis, what would refute it, predictions (mine, before the runs)

*Hypothesis.* A spread of rates over the cells, at equal geometric mean, does not help: no `clock` arm is above the baseline by more than the seeds' scatter.

*Refuted if* some `clock` arm has, on a testbed, a median above the baseline's by more than 5 points on T1 or 2 points on T2 **and** is above the baseline on at least four of the five seeds (the seeds are paired: every other number at birth and every line read is the same). It would then be compared with its `am` control before the difference is assigned to the spread.

The runs of `check_kinds.py` showed that with `clock=16` on T1 the numpy and the compiled program agree to 1e-14 for 30 lines and part company by 300: the fastest cells (rates up to 16) make the run sensitive to rounding. I take that as a sign that B = 16 is too wide for T1 at eta1 = 1.

| | Statement | Chance |
|---|---|---|
| B1 | T2 baseline: the median of the five seeds is 84.2 (to 0.1) | 0.90 |
| B2 | T2, `c4`: median within 2 points of the baseline's | 0.60 |
| B3 | T2, `c16`: median at least 2 points below the baseline's | 0.55 |
| B4 | T1, `c4`: median within 5 points of the baseline's | 0.45 |
| B5 | T1, `c16`: median at least 5 points below the baseline's | 0.60 |
| B6 | the hypothesis is not refuted on either testbed | 0.70 |
| B7 | T1, `c4`: median lines to 90% fewer than the baseline's | 0.40 |
| B8 | T2: `am4` within 2 points of `c4` (medians) | 0.60 |

## 5. Cost, what is not run

T1: 4 development runs and 20 runs on the reported seeds, about 3.5 CPU-minutes each, 85 CPU-minutes, about 22 minutes on four cores. T2: 6 development runs and 30 runs, 16 to 20 seconds each, about 3 minutes on four cores. Not run: rates that themselves change; a rate for each strength and not for each cell; a spread in eta2; `am16` and `am2` on T1; `am2` on T2; other shapes of the spread; more than five seeds.
