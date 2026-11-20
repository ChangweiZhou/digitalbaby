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
