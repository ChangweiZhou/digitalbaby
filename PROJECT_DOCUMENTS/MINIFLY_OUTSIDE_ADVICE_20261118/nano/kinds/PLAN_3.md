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
