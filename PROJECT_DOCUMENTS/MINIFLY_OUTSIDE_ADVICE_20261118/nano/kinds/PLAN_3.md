# Kinds of cells, third plan (C6, question 3): one signal that reaches every cell alike (20 November 2026)

By the agent session that owns line C6 and the folder `nano/kinds/`. Development level; small cases; no checker gate, seal or review is claimed. The readings are mine. Source of the question, authorisation and source of the programs: as in `PLAN_1.md`. Written before any run of these arms beyond the 300-line runs of `check_kinds.py`, and before the outcome of the second plan is known.

## 1. The question

Mr. Xu's "ability to output signals (chemicals)" is taken here in one form (the reading is mine): one cell gives out one number that reaches every cell alike, and every cell changes faster or slower by it. **Does a common signal of how wrong the system has lately been, which scales every cell's step, change what is learned, against no such signal at the same mean step?**

## 2. The law

Option `sig=beta sigp=p sigk=k`. At every byte that the system names, the answering cell forms its mismatch m (the byte that arrived, less what was given to each branch) and its size s = the root of the sum of the squares of m. It keeps two numbers: a running mean a of s (a moves towards s by beta times the difference at every byte; it starts at the first s), and the mean A of s over the whole run so far. It sends the one number a/A to every cell in between. Every such cell multiplies the step of that byte by

    M = k times (a/A) to the power p,

with a and A as they stood before that byte's mismatch (what had been sent already). So the steps are larger when the system has lately been more wrong than over the run so far, and smaller when it has been more right. The answering cell's own step and the return strengths are unchanged. In `hold.py` M multiplies eta1 (what a cell receives from the answering cell and what its readers send back to it); in `cells2.py` it multiplies eta1 as well.

**Normalisation.** k is a constant, set on the development seed so that the mean of M over the bytes of the run is 1 on that seed: a first run with k = 1 gives the mean multiplier Mbar; the run is repeated with k = 1/Mbar, and once more if its mean is not within 0.02 of 1. On the reported seeds the same k is used and the mean multiplier of every run is reported; it is equal to the baseline's 1 only as nearly as the seeds resemble the development seed, and a run whose mean is further than 0.05 from 1 is marked.

## 3. Testbeds, settings, arms

T1 and T2 as in `PLAN_2.md`. Settings tried on the development seed (1), on each testbed: beta = 1/16, 1/256, 1/4096 a byte (a line is about 10 bytes), with p = 1 and p = 4: six settings, each normalised as above. Most bytes of both texts cannot be foretold (the digits of the two numbers, the letters), so the size of the mismatch has a floor and a/A may move little; p = 4 is tried for that reason.

Reported on the five seeds, on each testbed: the baseline (T1: the runs `q1_spread_<seed>`; T2: the runs `q2_base_<seed>`; the same commands in the same copy with the options off), and two signal arms: `s1`, the setting with p = 1 that scores highest on the development seed, and `s4`, the same for p = 4.

**The exact learners.** As in `PLAN_2.md`; the signal adds three numbers kept by the answering cell (a, the sum of s, the count) and no number that learns.

## 4. Hypothesis, what would refute it, predictions (mine, before the runs)

*Hypothesis.* The signal makes no difference beyond the seeds' scatter on either testbed, because the size of the mismatch is governed by the bytes that cannot be foretold.

*Refuted if* a signal arm has, on a testbed, a median further from the baseline's than 5 points on T1 or 2 points on T2 **and** lies on that side of the baseline on at least four of the five seeds (paired by seed).

| | Statement | Chance |
|---|---|---|
| C1 | on the development seed with p = 1, a/A at the end of the run is between 0.8 and 1.0 on both testbeds at beta = 1/256 | 0.60 |
| C2 | T2, `s1`: median within 2 points of the baseline's | 0.65 |
| C3 | T2, `s4`: median within 2 points of the baseline's | 0.45 |
| C4 | T1, `s1`: median within 5 points of the baseline's | 0.50 |
| C5 | the hypothesis is not refuted on either testbed, for either arm | 0.60 |
| C6 | if refuted anywhere, the arm is below the baseline (the signal harms) | 0.60 |
| C7 | the mean multiplier is within 0.05 of 1 on all five seeds, in all four arm-and-testbed cells | 0.60 |

## 5. Cost, what is not run

T2: 12 to 18 development runs and 10 runs on the reported seeds, 16 to 20 seconds each. T1: 12 to 18 development runs and 10 runs on the reported seeds, about 3.7 CPU-minutes each: about 100 CPU-minutes, about 26 minutes on four cores. Not run: a signal with a sign (better or worse than expected); a signal that each cell weighs by a strength of its own; a signal sent by any cell but the answering one; a mean A over a fixed span in place of the whole run; a control at each seed's own mean multiplier; more than five seeds.

## 6. Entered on 20 November after the development runs (seed 1), before any run on the reported seeds (nothing above is changed)

`calib_t2.txt`, `calib_t1.txt`, `res/*q3d_*`. Score on the development seed at the constant k finally used; the baseline on that seed: T2 81.5, T1 98.1.

| Testbed | p | beta 1/16 | beta 1/256 | beta 1/4096 |
|---|---|---|---|---|
| T2 | 1 | 84.0 (k 1.0720) | 79.3 (k 1.0714) | 81.7 (k 1.0698) |
| T2 | 4 | 85.2 (k 1.3018) | 79.1 (k 1.3255) | 80.4 (k 1.3064) |
| T1 | 1 | 93.3 (k 1) | 99.9 (k 1.0275) | 97.5 (k 1.0329) |
| T1 | 4 | 99.6 (k 1.0828) | 99.6 (k 1.0992) | 100.0 (k 1.1171) |

With k = 1 the mean multiplier was 0.93 (T2, p = 1), 0.75 to 0.77 (T2, p = 4), 0.97 to 0.98 (T1, p = 1) and 0.90 to 0.92 (T1, p = 4): the mismatch falls over the run, so a/A is mostly below 1. With k set, the mean multiplier of the second run was between 0.990 and 1.015 in every setting; T1 at beta = 1/16 and p = 1 was within 0.02 at k = 1 and was not repeated. The ratio a/A at the end of the run is 0.83 to 0.94 on T2 and 0.96 to 0.98 on T1: the signal moves the step by a few parts in a hundred at p = 1, as expected from the bytes that cannot be foretold. **C1 holds** (0.936 on T2 and 0.970 on T1 at beta = 1/256, p = 1).

**Settings for the reported seeds**, by the rule of section 3 (the highest score on the development seed for each p): T2 `s1`: beta 1/16, k 1.0719875614320444; T2 `s4`: beta 1/16, k 1.3018480343526433; T1 `s1`: beta 1/256, k 1.0275211246368463; T1 `s4`: beta 1/4096, k 1.1171099423560744. On T1 five of the six settings are within 3 points of each other and of the baseline on this seed, at the ceiling, so the choice there is close to arbitrary.

**Added after the plan, on the coordinating session's rule of 20 November for every line** (single runs on the gap text differ by about ten points between programs that agree to rounding): where two arms are within about ten points on T1 or five on T2, the ten further seeds 1001 to 1010 are run and the fifteen reported together, with the five reported seeds as the first table. Both arms are run on the further seeds on both testbeds without waiting to see whether they are close. The never-shown sums are given beside the shown ones on T1 (not a registered measure of this plan).

## 7. Outcome, entered on 20 November after the runs (nothing above is changed)

Result files `res/c2_q3_*`, `res/hold_q3_*` (the baseline's files are copies of `c2_q2_base_*` and `hold_q1_spread_*` under this plan's label); tables as printed by `agg.py` in `table_q3_t2_5.txt`, `table_q3_t2_15.txt`, `table_q3_t1_5.txt`, `table_q3_t1_15.txt`. Seeds in the order 73129, 93683, 116089, 406364, 545353.

**First table: the five reported seeds.**

| Testbed | Arm | Score, five seeds | Median | Against the baseline by seed | Median lines to 80% (T2) or 90% (T1), thousands | Mean multiplier |
|---|---|---|---|---|---|---|
| T2 | baseline | 84.2, 82.7, 86.4, 81.8, 86.7 | **84.2** | | 200 | |
| T2 | `s1` (p 1, beta 1/16) | 83.5, 83.4, 85.9, 84.1, 83.4 | **83.5** | 2 above, 3 below | 200 | 1.000 to 1.004 |
| T2 | `s4` (p 4, beta 1/16) | 80.7, 83.3, 81.9, 77.9, 81.0 | **81.0** | 1 above, 4 below | 300 | 1.003 to 1.016 |
| T1 | baseline | 91.2, 81.9, 100.0, 78.1, 100.0 | **91.2** | | 640 (two seeds do not reach 90%) | |
| T1 | `s1` (p 1, beta 1/256) | 64.0, 96.4, 100.0, 64.0, 99.7 | **96.4** | 1 above, 3 below, 1 equal | 540 (two do not) | 0.997 to 1.015 |
| T1 | `s4` (p 4, beta 1/4096) | 87.4, 98.8, 98.2, 100.0, 99.9 | **98.8** | 2 above, 3 below | 400 (all reach it) | 0.995 to 1.050 |

**Second table: the fifteen seeds together** (the five and 1001 to 1010).

| Testbed | Arm | Median | Mean | Against the baseline by seed | Median of the differences | Seeds below 90% (T1) |
|---|---|---|---|---|---|---|
| T2 | baseline | 83.7 | 83.2 | | | |
| T2 | `s1` | 83.5 | 83.3 | 8 above, 7 below | +0.5 | |
| T2 | `s4` | 81.9 | 82.1 | 5 above, 10 below | -1.7 | |
| T1 | baseline | 91.2 | 89.6 | | | 7 of 15 |
| T1 | `s1` | 99.2 | 93.8 | 9 above, 5 below, 1 equal | +1.7 | 2 of 15 |
| T1 | `s4` | 99.6 | 98.4 | 10 above, 4 below, 1 equal | +3.1 | 1 of 15 |

Never-shown sums on T1 (20 pairs, not a registered measure; 10% is the rate of an answer at random), mean of the same three readings, median of fifteen seeds: baseline 1.7, `s1` 3.3, `s4` 3.3; the highest such mean on any seed is 20.0 (baseline), 16.7 (`s1`), 10.0 (`s4`). The highest single reading of a run is 35 (baseline, seed 1008, and `s1`, seed 1008) and 30 (`s4`, seed 1009); a single reading above 30 occurs in the baseline as well, so no arm moves this figure. These cells do not answer the sums they were not shown, with or without the signal.

**The hypothesis (no difference beyond the scatter).**

- *T2, p = 1:* not refuted. Medians 83.5 against 84.2 on five seeds and 83.5 against 83.7 on fifteen; 8 seeds above and 7 below.
- *T2, p = 4:* **refuted by the registered criterion on the five seeds** (median 3.2 points below, below on four seeds of five). The fifteen seeds do not carry it at that size: median 1.8 points below, below on 10 seeds of 15 (a split of 10 to 5 or wider arises by chance about three times in ten). My reading: with the fourth power the signal does not help on T2 and may cost one to three points; no more is carried.
- *T1, both arms:* not refuted by the criterion on five seeds (`s1`: the median is 5.2 points above but only one seed is above; `s4`: 7.6 above, two seeds above). On fifteen seeds both arms have a higher median (99.2 and 99.6 against 91.2) and fewer seeds below 90% (2 and 1 against 7), with 9 and 10 seeds of 15 above the baseline, which the pairing does not carry (10 of 14 one way arises by chance about once in five or six). See the note below before reading this as an effect of the signal.

**A note on the baseline of T1, which bears on this plan and the second.** On the fifteen seeds the baseline has 7 seeds below 90%; every other arm run on the fifteen in this folder has 1 to 3 (`s1`, `s4`, and the three rate arms of the second plan), though the arms change different things and two of them (`s1` here, `c2` there) change the step by a few parts in a hundred. Either the unchanged system sits where a small disturbance of the steps helps a run out of a state in which it answers about four fifths of the sums, or the baseline drew seven low runs by chance (the coordinating session reports that single runs on this text move by about ten points from rounding alone). I cannot tell the two apart with what was run, and I do not claim that the signal helps on T1. What would tell: the baseline run several times on each seed with a disturbance of rounding size, which is not run here.

**How large the signal is.** With k set, the multiplier's mean over a run is within 0.05 of 1 on every seed (the furthest: 1.0498, T1 `s4`, seed 73129). The ratio a/A at the end of the development runs is 0.83 to 0.94 on T2 and 0.96 to 0.98 on T1; the range of the multiplier within a run was not recorded.

**Predictions scored** (on the five reported seeds, as registered).

| | Statement | Chance | Outcome |
|---|---|---|---|
| C1 | development seed, p = 1, beta = 1/256: a/A at the end between 0.8 and 1.0 on both testbeds | 0.60 | **holds** (0.936, 0.970) |
| C2 | T2, `s1`: median within 2 points of the baseline's | 0.65 | **holds** (0.7 below) |
| C3 | T2, `s4`: median within 2 points of the baseline's | 0.45 | **fails** (3.2 below) |
| C4 | T1, `s1`: median within 5 points of the baseline's | 0.50 | **fails** (5.2 above) |
| C5 | the hypothesis is not refuted on either testbed, for either arm | 0.60 | **fails** (T2, `s4`, on the five seeds) |
| C6 | if refuted anywhere, the arm is below the baseline | 0.60 | **holds** |
| C7 | the mean multiplier is within 0.05 of 1 on all five seeds in all four cells | 0.60 | **holds** (the furthest 1.0498) |

Four of seven hold.

**Cost.** T2: 12 development runs and 30 runs, 14 to 20 CPU-seconds each. T1: 11 development runs and 30 runs, 207 to 301 CPU-seconds each (several queues shared the four cores). **Not run:** as listed in section 5; the range of the multiplier within a run; the baseline repeated on a seed.
