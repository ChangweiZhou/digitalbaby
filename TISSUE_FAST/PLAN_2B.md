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

## Outcome

(to be entered after the runs)
