# nano/loop, third plan (roadmap step L2 b): the number of readings of the stream (20 November 2026)

**Authorisation and source** as in `PLAN_1.md`. Written with `PLAN_2.md`, before any run of this plan, having seen what that plan lists. Development level; no checker gate, seal or review is claimed.

**The one variable:** the number of times the stream is read: 8, 16, 32. Everything else is the first plan's arm P at its setting (eta1 = 1, eta2 = 1/4, born at nothing, fading 1e-5, D = 10), kernel `kernt.c`. Nothing changes between readings (no step that shrinks, nothing reset), so readings 8 and 16 are read off the run of 32.

**Size.** 1,024 cells (733,249 numbers that learn), three seeds (73129, 93683, 116089), 32 readings: 32 minutes a run, in two parts of 16 readings. And, as the largest size I can afford at more than 8 readings: 4,096 cells, seed 73129, 16 readings (the second plan's run carried on by 8 readings, 35 minutes in two parts). 4,096 cells at 32 readings (2.3 hours a run) are not run.

**Question.** Does the loss on text never read still fall after 8 readings?

**Hypothesis.** It falls from 8 to 16 and from 16 to 32 readings. **Refuted if**, on the mean of the seeds at 1,024 cells, the loss after 32 readings is not below the loss after 8 by at least 0.01 nats, or the loss after 32 is above the loss after 16.

**Also reported:** the loss of the stream itself as it is read (each byte scored before it arrives, learning on) at each reading, beside the loss on the last tenth: their difference is what the system has taken from the particular text and not from its regularities. And reading 8 of these runs beside the first plan's claim runs of the same seeds, which were made with the other kernel: a check that the two kernels give the same figures over eight million bytes.

| | Prediction (mine, before any run of this plan) | Chance |
|---|---|---|
| R1 | 1,024 cells: the loss after 32 readings is below the loss after 16 (mean of seeds) | 0.75 |
| R2 | 1,024 cells, 32 readings: between 1.72 and 1.76 nats | 0.45 |
| R3 | 1,024 cells, 32 readings: the stream as it is read is more than 0.25 nats below the last tenth | 0.55 |
| R4 | 4,096 cells: the loss after 16 readings is below the loss after 8 by at least 0.01 nats | 0.60 |
| R5 | reading 8 of these runs equals the first plan's figure to 0.002 nats on each seed | 0.85 |

**Cost.** 3 x 32 + 36 = 132 core-minutes.

**Not run.** More than 32 readings; 4,096 cells beyond 16 readings; a step or a fading that changes with the reading; 256 cells.

## Entered 15:58 UTC, before the run concerned reached its ninth reading (nothing above is changed)

The 4,096-cell run of this plan is changed, for the reason entered in `PLAN_2.md` at 15:58: at the fixed setting 4,096 cells learn slowly (1.949 nats after 2 readings against 1.80 with eta1 = 16), so a fall from 8 to 16 readings at that setting would mostly show a slow learner catching up. The run of 16 readings is made with eta1 = 16 (`a16_H4096_s73129`) and R4 is scored on it; this is a departure from the plan as written. The 1,024-cell runs are unchanged.

## Outcome (entered about 17:15 UTC; nothing above is changed; every figure copied from `tab_output.txt`, section L2 b)

Arm P, born at nothing, D = 10, eta2 = 1/4. Last tenth, learning off: nats (bits; named right) | the stream as it is read during that reading, nats.

| readings | 1,024 cells, eta1 = 1, mean of seeds 73129, 93683, 116089 | 4,096 cells, eta1 = 16, seed 73129 |
|---|---|---|
| 1 | 1.9174 (2.7662; 44.26%) \| 1.9489 | 1.8509 (2.6703; 46.38%) \| 1.8455 |
| 2 | 1.8528 (2.6730; 46.28%) \| 1.7385 | 1.7999 (2.5967; 47.91%) \| 1.6743 |
| 4 | 1.8043 (2.6031; 47.63%) \| 1.6525 | 1.7632 (2.5438; 48.93%) \| 1.6036 |
| 8 | 1.7687 (2.5517; 48.61%) \| 1.5940 | 1.7379 (2.5072; 49.73%) \| 1.5542 |
| 16 | 1.7429 (2.5144; 49.37%) \| 1.5492 | **1.7189 (2.4798; 50.36%)** \| 1.5153 |
| 32 | **1.7235 (2.4864; 50.05%)** \| 1.5123 | not run |

The three seeds at 32 readings: 1.7249, 1.7225, 1.7230. Beside them: unsmoothed tables of the last 3, 4 bytes 1.796, 1.755 nats (crude), 47.2, 52.5% right.

**The hypothesis is not refuted:** at 1,024 cells the loss falls from 8 to 16 readings by 0.026 nats and from 16 to 32 by 0.019. Each doubling of the readings gives about three quarters of the one before (0.065, 0.049, 0.036, 0.026, 0.019). If that went on, all further readings together would give about 0.05 nats more; that is an extrapolation of mine and not a result.

**Also seen.** The stream as it is read is 0.21 nats below the text never read after 32 readings (0.17 after 8): the system takes in particulars of the stream, slowly, and the loss on text never read has not turned upward by 32 readings. 4,096 cells with eta1 = 16 after 16 readings (1.7189) are 0.005 below 1,024 cells after 32 (1.7235), at about 6.5 times the time a reading (4.7 minutes against 0.7), so at about three times the time in all.

**Predictions scored.** R1 held. R2 held (1.7235). R3 failed (0.211 nats, where I had said more than 0.25). R4 held on the changed run (0.019 nats; the run as first registered, at eta1 = 1, was stopped after 8 readings). R5 held: the two kernels give 1.7689, 1.7687, 1.7686 on the three seeds, the same to the fourth decimal.

**Cost.** 1,024 cells: 43 seconds a reading with `kernt.c`, 23 minutes a run of 32. 4,096 cells: 75 minutes for 16 readings, in four parts. **Not run:** more than 32 readings; 4,096 cells beyond 16; 1,024 cells at eta1 = 4 beyond 8 readings; a step or fading that changes with the reading.
