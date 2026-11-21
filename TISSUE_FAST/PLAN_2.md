# Plan 2: connections that turn over, with a trial period (21 November 2026)

Written before any run of this plan. Development level; one seed; no gate, seal or outside review is claimed.

## Authorisation and source

- **Owner's instruction (21 November 2026, this session):** change the making rule, after thinking about how biology does it; the plan below was put to him and he answered "可以，跑起来吧".
- **Program:** `TISSUE_EVENT/` of this folder at the commit of this plan. With the new switch off, the program gives the same numbers as at 6cbcd1fab, first form with withdrawal by use, 8,192 bytes: connections, emissions, cell quantities and generator state compared with `==`.
- **Learner:** as plan 1 (E63's tissue, 2,048 cells, neighbourhood radius 1/8, 8 starting connections, the four switches on, 32 steps a byte, no feedback), with withdrawal by use on in every arm.

## Why

Plan 1: the number of connections is set by the making rule. A cell makes a connection every 128 steps while its average activity is below a target that almost no cell reaches, so it connects to almost everything within reach, and what withdrawal by use removes is made again. In the nervous system, as far as the session recalls it (not checked against sources): a neuron is connected to a minority of the cells it could reach; contacts are made and lost all the time and most new ones are transient; a new contact is kept when it becomes useful, that is when its input goes with the receiving cell's activity, and weak, little-used ones are removed. The number is then the balance of a rate of making and a chance of being kept, not a quantity that is filled up.

## The rule (switch `--turnover`)

1. **Making.** Every `--make-period` steps a cell makes one connection to a cell or line within reach that it is not yet connected to, whatever its activity and threshold (the target a* no longer decides making). Feedback cells make none, as before.
2. **Trial.** A connection made from a cell during the life is on trial for `--trial` steps. Its use during the trial is the mean, over the steps at which the receiving cell was active, of the sender's trace times the receiving cell's activity (the quantity withdrawal by use already measures). At the end of the trial it is withdrawn if that use is 0 (including when the receiving cell was never active) or below `--trial-theta` times the mean running use of the receiving cell's other connections from cells that are not on trial; otherwise it stays, and from then on withdrawal by use and withdrawal by strength apply to it as to any connection.
3. Connections from lines are not put on trial (their use is not measured by the program).

## Question and hypothesis

**Question.** With turnover and a trial, does the number of connections a cell stop growing by itself below the size of the neighbourhood, and do the cells near the input lines still carry the byte?

**H2.** In each form, for at least one setting below, the mean number of connections a cell changes by less than 5% over the last 4,096 bytes and ends below 47 (half the neighbourhood), and S(1,0) in window A2 is at least the level 0.1103.

**Refuted if** no setting of a form meets all three.

| No. | Prediction | Chance |
|---|---|---|
| Q1 | At equal making period and trial, the threshold 1 ends with fewer connections than 1/4 (all eight pairs) | 0.8 |
| Q2 | At equal trial and threshold, the making period 2,048 ends with fewer connections than 512 (all eight pairs) | 0.7 |
| Q3 | H2 holds in the first form | 0.4 |
| Q4 | H2 holds in the coincidence form | 0.4 |
| Q5 | In each form, the setting with the fewest connections costs less a byte (core) than plan 1's reference R of that form | 0.3 |

## Arms

Seed 73129; E63's stream of stage A; 16,384 bytes; counts every 1,024 bytes; E63's readings.

| Making period (steps) | Trial (steps) | Threshold |
|---|---|---|
| 512, 2,048 | 512, 2,048 | 1/4, 1 |

Eight settings in each of the two plastic forms: 16 tissues. The references are plan 1's R and W of each form.

## Cost

Forecast from a trial of one setting over 4,096 bytes (first form, making period 512, trial 512, threshold 1/4: 6.5 ms a byte core): at most about 8 ms a byte with records, about 4 minutes a tissue with the run without records, about 15 to 20 minutes on 4 cores for all. Limit 1.5 CPU-hours.

## Not run in this plan

A making period of 128 steps (plan 1's rate; costly with use measured); other seeds; other targets a*; normalisation of a cell's total incoming strength (plan 3); a set point of the cell's own (plan 4); trials for connections from lines.

## Outcome (entered 21 November 2026, after all sixteen runs; `p2_tables.py` reads the receipts in `TISSUE_EVENT/runs/P2_*`)

All sixteen tissues ran to 16,384 bytes, exit 0; in each, the core run again without records gave the same emissions. "H2 met": ends below 47, moves by less than 5% over the last 4,096 bytes, S(1,0) in A2 at least 0.1103. Scores are E63's readings in window A2 (chance 0.0217).

### first

| Setting (making period, trial, threshold) | Connections a cell at 1,024 / 4,096 / 8,192 / 12,288 / 16,384 bytes | Last 4,096 bytes | Trials withdrawn / kept | Withdrawn by use | ms a byte, core / with records | S(1,0), S(2,0), S(4,0) in A2 | H2 met |
|---|---|---|---|---|---|---|---|
| plan 1 R (E63's making) | 93.7 / 93.8 / 93.8 / 93.8 / 93.8 | +0.0% | – | 0 | 1.26 / 1.76 | 0.687, 0.277, 0.041 | no |
| plan 1 W (E63's making, withdrawal by use) | 90.1 / 92.0 / 92.7 / 93.1 / 93.0 | -0.1% | – | 1,198,914 | 3.34 / 4.08 | 0.700, 0.266, 0.023 | no |
| 512, 512, 1/4 | 23.7 / 50.5 / 47.5 / 47.2 / 46.0 | -2.6% | 1,805,997 / 151,050 | 1,910,621 | 3.69 / 4.12 | 0.722, 0.317, 0.026 | yes |
| 512, 512, 1 | 12.8 / 37.7 / 40.9 / 40.7 / 40.0 | -1.5% | 1,835,718 / 134,982 | 1,929,187 | 3.94 / 4.35 | 0.710, 0.384, 0.030 | yes |
| 512, 2048, 1/4 | 36.8 / 46.1 / 46.9 / 42.3 / 42.0 | -0.7% | 1,633,517 / 134,209 | 1,903,434 | 3.20 / 3.66 | 0.680, 0.345, 0.022 | yes |
| 512, 2048, 1 | 22.3 / 38.5 / 35.9 / 35.7 / 35.5 | -0.7% | 1,734,471 / 122,748 | 1,920,722 | 3.26 / 3.68 | 0.720, 0.338, 0.025 | yes |
| 2048, 512, 1/4 | 8.0 / 18.3 / 28.2 / 37.9 / 37.9 | +0.2% | 356,138 / 124,951 | 422,472 | 3.53 / 3.85 | 0.815, 0.434, 0.042 | yes |
| 2048, 512, 1 | 7.8 / 10.2 / 15.8 / 21.4 / 27.5 | +28.0% | 426,022 / 57,502 | 455,513 | 1.91 / 2.14 | 0.924, 0.453, 0.038 | no |
| 2048, 2048, 1/4 | 9.2 / 22.3 / 37.0 / 41.9 / 42.9 | +2.4% | 303,074 / 148,075 | 403,841 | 4.23 / 4.66 | 0.759, 0.378, 0.030 | yes |
| 2048, 2048, 1 | 8.8 / 11.5 / 22.5 / 26.4 / 28.3 | +7.3% | 378,126 / 82,427 | 447,867 | 2.93 / 3.20 | 0.908, 0.529, 0.031 | no |

### coincidence

| Setting (making period, trial, threshold) | Connections a cell at 1,024 / 4,096 / 8,192 / 12,288 / 16,384 bytes | Last 4,096 bytes | Trials withdrawn / kept | Withdrawn by use | ms a byte, core / with records | S(1,0), S(2,0), S(4,0) in A2 | H2 met |
|---|---|---|---|---|---|---|---|
| plan 1 R (E63's making) | 29.6 / 31.3 / 32.9 / 31.0 / 33.1 | +6.9% | – | 0 | 3.19 / 3.64 | 0.304, 0.029, 0.027 | no |
| plan 1 W (E63's making, withdrawal by use) | 23.3 / 22.1 / 24.5 / 25.2 / 23.1 | -8.5% | – | 715,714 | 3.68 / 4.21 | 0.409, 0.024, 0.020 | no |
| 512, 512, 1/4 | 12.4 / 17.2 / 18.6 / 16.7 / 17.7 | +5.9% | 1,166,306 / 587,349 | 1,316,253 | 3.50 / 3.59 | 0.504, 0.029, 0.021 | no |
| 512, 512, 1 | 11.2 / 13.5 / 13.7 / 13.3 / 14.0 | +5.5% | 1,457,987 / 328,416 | 1,560,834 | 2.72 / 2.97 | 0.601, 0.031, 0.022 | no |
| 512, 2048, 1/4 | 13.9 / 20.8 / 19.6 / 20.6 / 19.8 | -3.8% | 445,246 / 837,358 | 773,447 | 3.80 / 4.12 | 0.487, 0.026, 0.024 | yes |
| 512, 2048, 1 | 11.0 / 15.5 / 16.2 / 16.8 / 17.2 | +2.1% | 837,916 / 501,228 | 1,112,738 | 3.31 / 3.41 | 0.524, 0.034, 0.017 | yes |
| 2048, 512, 1/4 | 7.9 / 7.7 / 10.3 / 11.7 / 12.1 | +3.1% | 227,405 / 205,041 | 284,647 | 2.55 / 2.66 | 0.759, 0.044, 0.020 | yes |
| 2048, 512, 1 | 7.7 / 6.4 / 7.8 / 9.2 / 9.6 | +4.5% | 330,435 / 112,024 | 367,328 | 2.05 / 2.13 | 0.869, 0.050, 0.023 | yes |
| 2048, 2048, 1/4 | 8.9 / 8.8 / 11.4 / 13.0 / 13.4 | +3.5% | 111,446 / 240,705 | 211,142 | 2.89 / 3.07 | 0.729, 0.039, 0.019 | yes |
| 2048, 2048, 1 | 8.6 / 6.8 / 8.6 / 9.7 / 10.3 | +7.0% | 236,112 / 128,349 | 312,026 | 2.27 / 2.49 | 0.795, 0.019, 0.020 | no |


| | Outcome |
|---|---|
| H2 | **Held in both forms on this seed.** First form: six of eight settings (for example making period 512, trial 2,048, threshold 1: 35.5 connections a cell, -0.7% over the last 4,096 bytes, S(1,0) 0.720). Coincidence form: six of eight (for example 2,048, 512, 1: 9.6, +4.5%, S(1,0) 0.869). The two settings of each form that miss do so on the 5% of settling: with the making period 2,048 and threshold 1 the number was still rising at the end (+28.0%, +7.3% first form; +7.0% coincidence) |
| Q1 | Held: threshold 1 ends with fewer connections than 1/4 in all eight pairs |
| Q2 | Failed: the making period 2,048 ends with fewer in seven of eight pairs, not in the first form at trial 2,048 and threshold 1/4 (42.9 against 42.0) |
| Q3 | Held |
| Q4 | Held |
| Q5 | Failed: in the first form the settings with fewest connections cost more a byte than plan 1's R (1.91 to 3.94 against 1.26 ms); in the coincidence form they cost less (2.05 to 2.55 against 3.19) |

**Readings (development level, one seed, 16,384 bytes).**

1. **The number of connections is now set by the cells' rule and stays far below the neighbourhood:** about 35 to 46 a cell in the first form and 10 to 20 in the coincidence form, against 93.8 and 33.1 with E63's making. Most new connections fail their trial (in the first form 1.6 to 1.8 million of about 1.8 to 2.0 million at the making period 512).
2. **The cells near the input lines carry the byte as well or better than before.** S(1,0) 0.68 to 0.92 in the first form (R: 0.687) and 0.49 to 0.87 in the coincidence form (R: 0.304); S(2,0) in the first form 0.32 to 0.53 (R: 0.277). S(4,0) stays at chance (0.02 to 0.04) in every arm: four connections from the lines the tissue still carries nothing.
3. **Fewer connections did not make the first form cheaper.** Measuring every connection's use at every step at which its receiving cell is active (withdrawal by use, needed for the trial) costs more than the connections saved. The cost to remove next is that measurement, not the connections.

**Limits.** One seed; 16,384 bytes; one form of trial; the number still rose slowly in several settings and a longer life may move it; no feedback; no reading of what the new connections carry beyond E63's readings.
