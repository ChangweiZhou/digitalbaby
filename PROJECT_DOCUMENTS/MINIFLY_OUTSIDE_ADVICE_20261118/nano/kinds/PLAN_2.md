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

## 6. Entered on 20 November during the runs (nothing above is changed)

The development seed (1) ran in all arms: T2 baseline 81.5, `c2` 82.1, `c4` 83.3, `c16` 73.7, `am4` 81.4, `am16` 83.0; T1 (baseline 98.1 from the first plan) `c2` 98.6, `c4` 99.9, `c16` 98.9, `am4` 98.8. Nothing was chosen on it. **Added after the plan, on the coordinating session's rule of 20 November for every line:** where two arms are within about ten points on T1 or five on T2, the ten further seeds 1001 to 1010 are run and the fifteen reported together, the five reported seeds remaining the first table. All arms were run on the further seeds on both testbeds. The never-shown sums of T1 are given as well (not a registered measure).

## 7. Outcome, entered on 20 November after the runs

Result files `res/c2_q2_*`, `res/hold_q2_*` (the T1 baseline's files are copies of `hold_q1_spread_*` under this plan's label); tables as printed by `agg.py` in `table_q2_t2_5.txt`, `table_q2_t2_15.txt`, `table_q2_t1_5.txt`, `table_q2_t1_15.txt`. Seeds in the order 73129, 93683, 116089, 406364, 545353.

**T2, two-digit sums with the chain of delay cells (eta1 = 1/4); never-shown sums.**

| Arm | Five seeds | Median | Fifteen seeds: median | mean | Against the baseline by seed (fifteen) | Median of the differences |
|---|---|---|---|---|---|---|
| baseline | 84.2, 82.7, 86.4, 81.8, 86.7 | **84.2** | 83.7 | 83.2 | | |
| `c2` (rates over a range of 4) | 81.7, 77.1, 86.3, 84.3, 81.7 | **81.7** | 82.2 | 82.4 | 5 above, 10 below | -0.5 |
| `c4` (range of 16) | 72.5, 78.2, 76.1, 83.7, 74.2 | **76.1** | 78.2 | 79.1 | 3 above, 12 below | -3.3 |
| `c16` (range of 256) | 75.2, 73.3, 74.1, 83.1, 82.9 | **75.2** | 79.5 | 77.3 | 3 above, 12 below | -3.7 |
| `am4` (all alike at 1.35 eta1) | 84.7, 84.9, 75.4, 81.1, 88.4 | **84.7** | 83.7 | 82.5 | 9 above, 6 below | +0.1 |
| `am16` (all alike at 2.87 eta1) | 85.2, 86.7, 82.5, 81.6, 82.3 | **82.5** | 82.3 | 81.3 | 6 above, 9 below | -1.1 |

On the five seeds `c4` and `c16` are 8 and 9 points below the baseline at the median and below it on four seeds of five each; on fifteen seeds they are 5.5 and 4.2 below at the median and below on 12 seeds of 15 (a split of 12 to 3 or wider arises by chance about once in thirty). The controls at the arithmetic mean rate are level with the baseline. **My reading: on T2 a spread of rates over the cells of sixteen-fold or more costs about four to eight points of never-shown sums, and the cost belongs to the spread and not to the larger mean rate.** At a four-fold range (`c2`) the difference is within the scatter (10 of 15 below, median difference half a point). The median of lines to 80% is 200 thousand for the baseline and both controls, 400 thousand for `c4` and `c16` on fifteen seeds.

**T1, one digit across a gap of letters (eta1 = 1); sums shown; lines to 90%.**

| Arm | Five seeds | Median | Lines to 90%, thousands (five) | Fifteen seeds: median | mean | Seeds below 90% | Against the baseline by seed | Median lines to 90% (fifteen) |
|---|---|---|---|---|---|---|---|---|
| baseline | 91.2, 81.9, 100.0, 78.1, 100.0 | **91.2** | 640, none, 260, none, 360 | 91.2 | 89.6 | 7 of 15 | | 560 (five do not reach it) |
| `c2` | 82.9, 99.4, 95.3, 95.4, 97.4 | **95.4** | 940, 440, 600, 520, 380 | 97.9 | 95.9 | 2 | 10 above, 5 below | 480 |
| `c4` | 95.7, 93.6, 99.2, 98.5, 94.2 | **95.7** | 240, 480, 300, 340, 500 | 98.5 | 96.4 | 1 | 10 above, 5 below | 380 |
| `c16` | 85.8, 96.7, 83.9, 100.0, 98.8 | **96.7** | 480, 580, 720, 440, 540 | 96.8 | 95.2 | 3 | 9 above, 6 below | 480 |
| `am4` | 99.4, 95.0, 100.0, 99.2, 98.3 | **99.2** | 300, 580, 240, 440, 420 | 98.6 | 98.4 | 0 | 10 above, 4 below, 1 equal | 380 |

Never-shown sums on T1 (20 pairs; 10% is the rate of an answer at random), mean of the same three readings, median of fifteen seeds: baseline 1.7, `c2` 6.7, `c4` 3.3, `c16` 1.7, `am4` 6.7; the highest such mean on any seed is 23.3 (`c4`, seed 1003); the highest single reading of a run is 35 (baseline, seed 1008; `c2`, seed 1009; `am4`, seed 1007). A single reading above 30 occurs in the baseline as well; no arm answers the sums it was not shown.

On the five seeds no arm is separable from the baseline (medians 4 to 8 points above, with two or three seeds of five above). On fifteen seeds every arm has a higher median and mean than the baseline and fewer seeds below 90%, and every arm reaches 90% on all fifteen seeds where the baseline does not on five. The arm that does this most plainly is the control: all cells alike at 1.35 times eta1 has no seed below 95.0, against 7 seeds of 15 below 90% for the baseline (7 against 0 of 15 arises by chance less than once in a hundred). `c4`, which has the same arithmetic mean rate with a spread, is not above its control (median 98.5 against 98.6, mean 96.4 against 98.4, lines to 90% 380 against 380). **My reading: on T1 the baseline's rate (eta1 = 1) is somewhat low, and what the rate arms gain they gain from the larger mean rate; nothing here shows a gain from the spread itself, and nothing shows a loss up to a range of 256.** By seed the pairing carries none of the differences (10 of 15 at most), so this rests on the count of low runs. The note on the baseline of T1 in `PLAN_3.md`, section 7, applies here as well: `c2`, whose arithmetic mean rate is only 1.08, also has 2 low seeds against 7.

**The hypothesis** (a spread of rates at equal geometric mean does not help) **is not refuted** on either testbed by the registered criterion (five seeds), nor on fifteen: on T2 the spread harms; on T1 the arms are above the baseline, but not above the control that has the larger mean rate without the spread.

**Rounding.** At `clock=16` on T1 the numpy and the compiled program part company after about 150 lines (`check_kinds.py`), so a run of that arm is one draw from the runs that agree to rounding, more than in the other arms.

**Predictions scored** (on the five reported seeds, as registered).

| | Statement | Chance | Outcome |
|---|---|---|---|
| B1 | T2 baseline: median of the five seeds 84.2 | 0.90 | **holds** |
| B2 | T2, `c4`: median within 2 points of the baseline's | 0.60 | **fails** (8.1 below) |
| B3 | T2, `c16`: median at least 2 points below the baseline's | 0.55 | **holds** (9.0 below) |
| B4 | T1, `c4`: median within 5 points of the baseline's | 0.45 | **holds** (4.5 above) |
| B5 | T1, `c16`: median at least 5 points below the baseline's | 0.60 | **fails** (5.5 above) |
| B6 | the hypothesis is not refuted on either testbed | 0.70 | **holds** |
| B7 | T1, `c4`: median lines to 90% fewer than the baseline's | 0.40 | **holds** (340 against 640) |
| B8 | T2: `am4` within 2 points of `c4` | 0.60 | **fails** (84.7 against 76.1) |

Five of eight hold. I did not expect the spread to harm on T2 at B = 4, nor the widest spread to do no harm on T1.

**Cost.** T2: 6 development runs and 90 runs, 14 to 20 CPU-seconds each. T1: 4 development runs and 60 runs (and the baseline's 10 on the further seeds), 207 to 301 CPU-seconds each. **Not run:** as listed in section 5; on T1 a baseline at other common rates (which the control now calls for: eta1 at 1.35 and 2 as baselines in their own right, with the spread compared at each); why the spread harms on T2 (which cells, fast or slow, lose their use).
