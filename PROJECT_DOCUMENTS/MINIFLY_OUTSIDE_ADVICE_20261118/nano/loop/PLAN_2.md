# nano/loop, second plan (roadmap step L2 a): the number of cells (20 November 2026)

**Authorisation and source** as in `PLAN_1.md`; branch `claude/nano-loop-8mmhfm`. Written while the claim runs of the first plan were under way and before any run of this plan. Seen when writing: the development-seed stages of the first plan (arm P, 4 readings: 256 cells 1.8109 nats, 1,024 cells 1.8031), the first two readings of four claim runs, and timings on 8,000 to 20,000 bytes at 4,096 and 16,384 cells. Development level; no checker gate, seal or review is claimed.

**The one variable:** H, the number of cells in between: 256, 1,024, 4,096; 16,384 in part (below). Everything else is the first plan's arm P at its chosen setting: eta1 = 1, eta2 = 1/4, return strengths born at nothing, fading 1e-5, D = 10, 8 readings. (If the first plan's claim runs do not show arm P to be the better answering cell, this plan is void and is rewritten as a new plan; the outcome of the first plan is entered there before these runs start.)

**The exact learner.** `loop.Sys` as in the first plan, run by `kernt.c`: because the chosen arm is born at nothing, its return strengths are the same numbers as the answering cell's strengths at every byte, and this kernel keeps one copy (`check.py`: it agrees with the numpy reference and with `kernp.c` to rounding, 3e-15 to 6e-15). Numbers that learn (the return strengths not counted twice): H x 716 + 65, that is **183,361; 733,249; 2,932,801; 11,731,009**. I do not call the last two small: the largest has more numbers than the roughly ten million the roadmap gives for nanoGPT's character-level example.

**Question.** At fixed settings, does the loss on text never read fall as cells are added?

**Hypothesis.** It falls at every step in size. **Refuted if** 4,096 cells are not below 1,024 cells by at least 0.01 nats on the mean of the seeds (8 readings), or 1,024 not below 256 by as much.

**Runs.** Seeds 73129, 93683, 116089 (three of the five reported seeds; no setting is chosen in this plan except as said under "A stated pre-check").
- 256 and 1,024 cells: the first plan's claim runs of arm P (kernel `kernp.c`).
- 4,096 cells: three seeds, 8 readings (about 4.4 minutes a reading, 35 minutes a run; made in two parts of 4 readings with `resume=1` so that no part passes 30 minutes; a run made in parts gives the same numbers as one made whole, tested on a small case).
- 16,384 cells: seed 73129 only, **2 readings** (about 21 minutes a reading, each reading its own part), compared with the other sizes after 2 readings and not after 8. Readings 3 and 4 only if cores are idle at the end. Eight readings would take about three hours and are not run.

**A stated pre-check, on development seed 1.** The settings were chosen at 256 cells, and at 1,024 cells only eta2 was looked at again. The sum a cell receives over its return strengths may shrink as cells are added, so eta1 = 1 may be too small at the larger sizes. Before the claim figures are read: 1,024 cells, 4 readings, eta1 = 4 and 16 (eta1 = 1 is `s3_P_f1o4`: 1.8031). If one of them is lower by more than 0.01 nats, a second series "eta1 chosen at 1,024 cells" is run at 1,024 and 4,096 cells on seed 73129 and reported beside the fixed-setting series, under its own name; the hypothesis above is scored on the fixed-setting series only.

**Beside every figure:** the unsmoothed tables of the last k bytes (k = 1 to 6, `res/tables.json`); the arm at the earlier birth of the return strengths where the first plan ran it (256 and 1,024 cells; not at 4,096 cells and more: stated as not run).

| | Prediction (mine, before any run of this plan) | Chance |
|---|---|---|
| B1 | 4,096 cells are below 1,024 cells by at least 0.02 nats (mean of seeds, 8 readings) | 0.35 |
| B2 | 4,096 cells reach 1.755 nats or lower (the best unsmoothed table) | 0.25 |
| B3 | the share named right rises at each step, 256 < 1,024 < 4,096 | 0.70 |
| B4 | after 2 readings 16,384 cells are not below 4,096 cells by more than 0.01 nats | 0.60 |
| B5 | the pre-check finds eta1 = 4 or 16 better than eta1 = 1 by more than 0.01 nats at 1,024 cells | 0.40 |

**Cost.** 4,096 cells: 3 x 36 = 108 core-minutes; 16,384 cells: 44; pre-check 8; the second series, if it is run, 8 + 36. About 2.7 to 3.4 core-hours.

**Not run.** 16,384 cells beyond 2 (or 4) readings; settings chosen anew at each size; the earlier birth of the return strengths at 4,096 cells and more; other fading; more than three seeds.

## Entered during the runs, 15:40 UTC, before any of the runs it adds (nothing above is changed)

**Seen.** After one reading, 4,096 cells at the fixed settings stand at 2.0367 and 2.0375 nats (seeds 73129, 93683), where 1,024 cells stood at 1.916 and 256 cells at 1.889 after one reading (development seed). At the fixed settings the larger system learns more slowly. The spread over seeds has been 0.001 to 0.003 nats in every run of this line.

**Changes, with their reasons.**
1. The 4,096-cell runs of seeds 93683 and 116089 are stopped (the first after its first reading, the second before it). The fixed-setting figure at 4,096 cells is therefore on **one seed**, 73129. The cores go to item 2. This is a departure from the plan above and is counted as such.
2. A second stated check on development seed 1, wider than the pre-check above because eta2 may matter as well: 4,096 cells, 2 readings, (eta1, eta2) = (1, 1), (4, 1/4), (16, 1/4), (4, 1); beside them the fixed setting (1, 1/4) after 2 readings on seed 73129 (not seed 1: stated). If the best is lower than the fixed setting by more than 0.01 nats, one run of 8 readings at that setting is made at 4,096 cells on seed 73129 and reported as "4,096 cells, settings chosen at that size", beside and not in place of the fixed-setting series. The hypothesis of this plan is still scored on the fixed-setting series.

**My expectation, before these runs:** at least one of the four is better than the fixed setting by more than 0.01 nats after 2 readings: chance 0.75 (B6). The run of 8 readings at the chosen setting ends below 1,024 cells' 1.768: chance 0.40 (B7).

The queue is now run by `q.sh` (`jobs_3.txt`: these four runs, then the rest of `jobs_2.txt` unchanged).

## Entered 15:58 UTC, as the runs it adds were being started (nothing above is changed)

**The settings check at 4,096 cells** (development seed 1, 2 readings; `res/t_H4096_*.json`): (eta1, eta2) = (16, 1/4) 1.8017 nats, 47.8%; (1, 1) 1.8137, 48.8%; (4, 1/4) 1.8444, 46.6%; (4, 1) 1.8540, 47.2%; the fixed setting (1, 1/4) on seed 73129: 1.9490, 43.3%. B6 held: every one of the four is better than the fixed setting by 0.09 to 0.15 nats. The best, eta1 = 16, is on the edge of what was tried; a larger eta1 was not tried at this size. 4,096 / 256 = 16: the best eta1 found is in proportion to the number of cells, which fits the reason given in the pre-check (the sum a cell receives over its return strengths shrinks as cells are added, because the answering cell's strengths do). This is one point and my reading, not a result.

**Runs added, under their own names ("eta1 in proportion to the cells": eta1 = H / 256, eta2 = 1/4):**
- 4,096 cells, eta1 = 16, seed 73129, 8 readings, and carried on to 16 readings for the third plan (`a16_H4096_s73129`).
- 16,384 cells, eta1 = 64, seed 73129, 2 readings (`a64_H16384_s73129`). **This setting is an extrapolation of the rule above and was checked on no seed at this size.**
- 1,024 cells, eta1 = 4, if the pre-check finds it better than eta1 = 1 by more than 0.01 nats (as the plan above provides).

**Runs stopped, to pay for them:** the fixed-setting run at 16,384 cells after its first reading (2.2201 nats, 36.3%; 20.7 minutes); the fixed-setting run at 4,096 cells after its eighth reading (its readings 9 to 16, which the third plan had asked for, are replaced there by those of the run with eta1 = 16). So the fixed-setting series is: 256 and 1,024 cells on three seeds, 4,096 cells on one seed, 8 readings; 16,384 cells on one seed after one reading only.

**Scoring stays as registered:** the hypothesis and B1 to B4 are scored on the fixed-setting series. B4 (16,384 against 4,096 cells after 2 readings) can no longer be scored on that series as written; it is scored after 1 reading, and also on the series with eta1 in proportion, and both are said.
