# Plan 2b: plan 2's best settings on two seeds not used in choosing them (21 November 2026)

Written before any run of this plan. Development level; no gate, seal or outside review is claimed.

**Authorisation.** Owner, 21 November 2026, this session: "都做吧" to the three items (the use measurement made cheaper; plan 2 repeated on other seeds; plan 3), then "好的，多试一试" to the order revised after an outside reading.

**Program.** `TISSUE_EVENT/` of this folder at the commit of this plan (the same core as plan 2's runs).

**What is repeated.** For each form, the two settings of plan 2 that met H2 on seed 73129 with the highest S(1,0) in A2, chosen by that rule before this plan: first form (making period, trial, threshold) 2,048, 512, 1/4 (S(1,0) 0.815) and 2,048, 2,048, 1/4 (0.759); coincidence form 2,048, 512, 1 (0.869) and 2,048, 512, 1/4 (0.759). Beside each, E63's making (plan 1's R) on the same seed.

**Seeds.** 93683 and 116089 (E63's other two seeds; their streams and births are E63's).

**H2b.** On each new seed, each repeated setting again ends below 47 connections a cell, moves by less than 5% over the last 4,096 bytes, and has S(1,0) in A2 at least 0.1103 and at least that of R on the same seed.

**Refuted if** a setting misses any of the four on either new seed. A miss is recorded for that setting and seed; the outcome states how many of the eight (setting, seed) cases hold.

| No. | Prediction | Chance |
|---|---|---|
| B1 | All eight cases hold | 0.35 |
| B2 | At least six of eight hold | 0.65 |
| B3 | On every new seed, every repeated setting has fewer connections than R of its form | 0.9 |
| B4 | The ordering of plan 2 is kept: in each form the setting with the higher S(1,0) on seed 73129 is higher on both new seeds | 0.4 |

**Arms.** 2 seeds × 2 forms × (2 settings + R) = 12 tissues of 16,384 bytes, counts every 1,024 bytes, E63's readings; withdrawal by use on in the plan-2 settings, off in R (as in plan 1).

**Cost.** About 3 to 5 ms a byte with records, twice that with the run without records: about 3 minutes a tissue, about 12 minutes on 3 cores. Limit 1 CPU-hour.

**Not run.** Other settings; other seeds; longer lives.

## Outcome (entered 21 November 2026; twelve runs, exit 0; `p2b_tables.py`)

### first

| Seed | Setting | Connections a cell at the end | Last 4,096 bytes | S(1,0), S(2,0), S(4,0) in A2 | R on that seed: connections, S(1,0) | ms a byte (core) | Holds |
|---|---|---|---|---|---|---|---|
| 73129 | P2048_T512_th0.25 | 37.9 | +0.2% | 0.815, 0.434, 0.042 | 93.8, 0.687 | 3.53 | yes (chosen on this seed) |
| 73129 | P2048_T2048_th0.25 | 42.9 | +2.4% | 0.759, 0.378, 0.030 | 93.8, 0.687 | 4.23 | yes (chosen on this seed) |
| 93683 | P2048_T512_th0.25 | 29.2 | +4.5% | 0.838, 0.414, 0.031 | 92.3, 0.746 | 2.22 | yes |
| 93683 | P2048_T2048_th0.25 | 34.3 | +3.7% | 0.752, 0.440, 0.033 | 92.3, 0.746 | 2.80 | yes |
| 116089 | P2048_T512_th0.25 | 37.7 | +5.4% | 0.903, 0.380, 0.037 | 93.7, 0.814 | 3.69 | no |
| 116089 | P2048_T2048_th0.25 | 34.9 | -1.0% | 0.877, 0.466, 0.026 | 93.7, 0.814 | 4.08 | yes |

### coincidence

| Seed | Setting | Connections a cell at the end | Last 4,096 bytes | S(1,0), S(2,0), S(4,0) in A2 | R on that seed: connections, S(1,0) | ms a byte (core) | Holds |
|---|---|---|---|---|---|---|---|
| 73129 | P2048_T512_th1 | 9.6 | +4.5% | 0.869, 0.050, 0.023 | 33.1, 0.304 | 2.05 | yes (chosen on this seed) |
| 73129 | P2048_T512_th0.25 | 12.1 | +3.1% | 0.759, 0.044, 0.020 | 33.1, 0.304 | 2.55 | yes (chosen on this seed) |
| 93683 | P2048_T512_th1 | 9.3 | -0.7% | 0.858, 0.036, 0.022 | 33.6, 0.406 | 1.90 | yes |
| 93683 | P2048_T512_th0.25 | 12.0 | -1.9% | 0.739, 0.037, 0.023 | 33.6, 0.406 | 2.42 | yes |
| 116089 | P2048_T512_th1 | 10.6 | +8.3% | 0.918, 0.027, 0.021 | 34.2, 0.352 | 2.01 | no |
| 116089 | P2048_T512_th0.25 | 13.3 | +4.3% | 0.845, 0.027, 0.025 | 34.2, 0.352 | 2.61 | yes |

New-seed cases holding: 6 of 8


| | Outcome |
|---|---|
| H2b | **Not all cases hold: 6 of 8.** Both misses are on seed 116089 and on the 5% of settling only (+5.4% first form, making period 2,048, trial 512; +8.3% coincidence form, threshold 1): the number was still rising slowly at 16,384 bytes. Every case is far below 47 and above R in S(1,0) |
| B1 | Failed (6 of 8) |
| B2 | Held |
| B3 | Held: every repeated setting has fewer connections than R on its seed (29 to 38 against 92 to 94; 9 to 13 against 34) |
| B4 | Held: on both new seeds the setting with the higher S(1,0) on seed 73129 is higher again |

**Reading.** The result of plan 2 repeats on two seeds that were not used to choose the settings: connections settle far below the neighbourhood, and the cells one connection from the lines carry the present byte better than with E63's making on every seed (first form 0.75 to 0.90 against 0.69 to 0.81; coincidence form 0.74 to 0.92 against 0.30 to 0.41). Whether the slow rise seen in some cases stops in a longer life is not known. S(4,0) is at chance everywhere.
