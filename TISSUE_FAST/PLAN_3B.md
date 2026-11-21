# Plan 3b: plan 3's best arms on seeds 93683 and 116089 (21 November 2026)

Written before any run of this plan. Development level; no gate, seal or outside review is claimed.

**Authorisation.** Owner, 21 November 2026: "都做吧", then "好的，多试一试". **Program:** `TISSUE_EVENT/` at the commit of this plan (unchanged since plan 3's runs).

**Repeated, chosen by rules fixed now:** for each form, the scaled arm with E63's making that has the highest S(1,0) on seed 73129 (N75 in both forms: 0.770, 0.773), and the scaled arm with turnover that has the highest S(2,0) (N75T in both forms: 0.577, 0.758). R of each form on each seed is plan 2b's.

**H3b.** On each new seed: N75 ends below 47 connections a cell, within 5% over the last 4,096 bytes, with S(1,0) at least 0.1103 and at least R's on that seed; N75T has S(1,0) and S(2,0) at least those of N75 on that seed. **Refuted** case by case; the outcome counts the cases that hold (eight).

| No. | Prediction | Chance |
|---|---|---|
| D1 | All four N75 cases hold | 0.7 |
| D2 | All four N75T cases hold | 0.5 |
| D3 | S(4,0) stays at chance (below 0.05) in all eight tissues | 0.85 |

**Arms.** 2 seeds × 2 forms × 2 arms = 8 tissues of 16,384 bytes; settings as in plan 3. **Cost** about 2 to 3.5 ms a byte: about 15 minutes on 4 cores. **Not run:** other arms, seeds, lengths.

## Outcome (entered 21 November 2026; eight runs, exit 0; `p3b_p4_tables.py`)

### first

| Seed | Arm | Connections at the end | Last 4,096 bytes | S(1,0), S(2,0), S(4,0) | R: S(1,0) | ms a byte | Holds |
|---|---|---|---|---|---|---|---|
| 73129 | N75 | 30.4 | -0.1% | 0.770, 0.390, 0.022 | 0.687 | 2.01 | yes (chosen on this seed) |
| 73129 | N75T | 17.5 | +1.2% | 0.992, 0.577, 0.022 | 0.687 | 1.68 | yes (chosen on this seed) |
| 93683 | N75 | 30.4 | +0.0% | 0.766, 0.373, 0.022 | 0.746 | 1.93 | yes |
| 93683 | N75T | 17.2 | +1.7% | 0.895, 0.584, 0.022 | 0.746 | 1.50 | yes |
| 116089 | N75 | 30.9 | +0.2% | 0.897, 0.379, 0.022 | 0.814 | 1.98 | yes |
| 116089 | N75T | 18.0 | +2.6% | 0.958, 0.565, 0.022 | 0.814 | 1.70 | yes |

### coincidence

| Seed | Arm | Connections at the end | Last 4,096 bytes | S(1,0), S(2,0), S(4,0) | R: S(1,0) | ms a byte | Holds |
|---|---|---|---|---|---|---|---|
| 73129 | N75 | 30.1 | -0.0% | 0.773, 0.449, 0.022 | 0.304 | 3.11 | yes (chosen on this seed) |
| 73129 | N75T | 18.8 | +4.1% | 0.982, 0.758, 0.057 | 0.304 | 3.19 | yes (chosen on this seed) |
| 93683 | N75 | 30.2 | -0.1% | 0.766, 0.423, 0.022 | 0.406 | 3.15 | yes |
| 93683 | N75T | 17.6 | +2.8% | 0.891, 0.689, 0.078 | 0.406 | 2.81 | yes |
| 116089 | N75 | 30.6 | +0.2% | 0.903, 0.475, 0.022 | 0.352 | 3.34 | yes |
| 116089 | N75T | 18.7 | +5.0% | 0.985, 0.758, 0.069 | 0.352 | 2.91 | yes |

New-seed cases holding: N75 4 of 4, N75T 4 of 4


| | Outcome |
|---|---|
| H3b | **Held in all eight new-seed cases.** N75 settles at 30.2 to 30.9 connections a cell within 0.2% on both seeds and both forms, with S(1,0) above R; N75T is at or above N75 in both S(1,0) and S(2,0) on every seed |
| D1 | Held |
| D2 | Held |
| D3 | Failed: in the coincidence form N75T has S(4,0) 0.078 (seed 93683) and 0.069 (116089), above 0.05 though below the level 0.1103; with 0.057 on seed 73129, the only arm with anything above chance at four connections |

**Reading.** Plan 3 repeats on two seeds not used to choose its arms, closely: the scaled tissue settles at about 30 connections a cell on every seed, and with turnover added the cells one and two connections from the lines carry the present byte (S(1,0) 0.89 to 0.99, S(2,0) 0.57 to 0.76) on all three seeds. Four connections out, only the coincidence form with both rules shows a trace above chance, on all three seeds.
