# Plan 3: a cell's total incoming strength held (21 November 2026)

Written before any run of this plan. Development level; one seed; no gate, seal or outside review is claimed.

## Authorisation and source

- **Owner, 21 November 2026, this session:** "都做吧" (the three items, plan 3 among them), then "好的，多试一试" to the order revised after an outside reading, which asked that plan 3 be run alone and together with plan 2's turnover, and that the threshold of withdrawal be checked against the total strength and the number of connections.
- **Program:** `TISSUE_EVENT/` of this folder at the commit of this plan. With `--norm` off it gives the same numbers as the program of plans 2 and 2b (checked with `==`, first form with turnover and withdrawal by use, 4,096 bytes).
- **The reviewer's form of 19 November** (`STATUS_OF_STEP_1_20261119.md` on `claude/vibrant-cerf-sje726`, list item 1): a cell conserves the total strength of its incoming connections, excitatory and inhibitory apart, 3/4 and 1/4. Measured there in a tissue without learning of strengths, not in E63's tissue.

## The rule (switch `--norm`)

A cell's learnable incoming connections are scaled, those from excitatory senders to sum to C_E and those from inhibitory senders to sum to C_I (each kind apart; a strength is kept within [0, 1]; a kind with sum 0 is left). The scaling is applied once at the start, after the cell's strengths change at the end of a byte, after it makes a connection, and after it loses one. A connection whose strength falls below 1/128 and is at least 128 steps old is given up, as in E63; the scaling is what makes connections compete for it.

**Check of the threshold (asked by the outside reading).** With C_E = 3/4 the mean excitatory strength reaches 1/128 only at 96 connections, so the scaling alone bounds the number weakly; with C_E = 3/8 at 48. Both are above the numbers plan 2 settled at (10 to 43), so the threshold does not cut connections below what plan 2 found.

## Question and hypothesis

**H3.** Holding the total incoming strength, with E63's making unchanged, brings the number of connections a cell below 47 and keeps it within 5% over the last 4,096 bytes, in both forms, with S(1,0) in A2 at least 0.1103 and at least that of R (plan 1). **Refuted if** neither C_E setting does so in a form.

| No. | Prediction | Chance |
|---|---|---|
| C1 | With E63's making, C_E 3/8 ends with fewer connections than 3/4 in both forms | 0.8 |
| C2 | H3 holds in the first form | 0.5 |
| C3 | H3 holds in the coincidence form | 0.5 |
| C4 | Some arm of the first form has S(4,0) in A2 above the level 0.1103 | 0.2 |
| C5 | The arms with E63's making and the scaling cost less a byte (core) than R of their form | 0.7 |
| C6 | Adding plan 2's turnover to the scaling raises S(2,0) in the first form above both alone | 0.3 |

## Arms

Seed 73129; 16,384 bytes; counts every 1,024 bytes; E63's readings.

| Arm | C_E, C_I | Making |
|---|---|---|
| N75 | 3/4, 1/4 | E63's |
| N38 | 3/8, 1/8 | E63's |
| N75T | 3/4, 1/4 | turnover with trial and withdrawal by use: first form 2,048 / 2,048 / 1/4, coincidence form 2,048 / 512 / 1/4 (the settings that met every condition on all three seeds in plans 2 and 2b) |
| N38T | 3/8, 1/8 | as N75T |

In both forms: 8 tissues. References: plan 1's R and plan 2's runs of the same turnover settings on seed 73129.

## Cost

Forecast from one trial of N75 over 4,096 bytes (first form, 0.52 ms a byte core): at most about 5 ms a byte for the turnover arms; about 15 minutes on 4 cores. Limit 1 CPU-hour.

## Not run in this plan

Other seeds (if H3 holds, a plan 3b repeats it as plan 2b did); other totals; scaling of connections from lines; the set point of plan 4.

## Outcome (entered 21 November 2026; eight runs, exit 0; `p3_tables.py`)

### first

| Arm | Connections a cell at 1,024 / 4,096 / 8,192 / 12,288 / 16,384 | Last 4,096 bytes | Given up / withdrawn by use | ms a byte core / with records | Cells sending a step | S(1,0), S(2,0), S(4,0), S(8,0) in A2 | Informative share d = 1 / 2 / 4 | H3 met |
|---|---|---|---|---|---|---|---|---|
| R (plan 1) | 93.7 / 93.8 / 93.8 / 93.8 / 93.8 | +0.0% | 161,961 / 0 | 1.26 / 1.76 | 34.7 | 0.687, 0.277, 0.041, – | 0.70 / 0.00 / 0.00 | – |
| turnover alone (plan 2) | 9.2 / 22.3 / 37.0 / 41.9 / 42.9 | +2.4% | 433,142 / 403,841 | 4.23 / 4.66 | 84.7 | 0.759, 0.378, 0.030, – | 0.96 / 0.35 / 0.42 | – |
| N75 | 30.3 / 30.5 / 30.5 / 30.5 / 30.4 | -0.1% | 8,021,250 / 0 | 2.01 / 2.29 | 43.3 | 0.770, 0.390, 0.022, – | 1.00 / 0.38 / 0.00 | yes |
| N38 | 18.9 / 19.2 / 19.2 / 19.2 / 19.2 | +0.0% | 8,053,064 / 0 | 1.29 / 1.50 | 24.7 | 0.746, 0.256, 0.022, – | 1.00 / 0.00 / 0.00 | yes |
| N75T | 9.8 / 14.3 / 16.3 / 17.3 / 17.5 | +1.2% | 485,977 / 432,568 | 1.68 / 1.84 | 31.6 | 0.992, 0.577, 0.022, – | 1.00 / 0.38 / 0.00 | – |
| N38T | 9.1 / 11.8 / 13.5 / 14.3 / 14.7 | +2.1% | 491,787 / 453,335 | 0.96 / 1.25 | 20.6 | 0.983, 0.522, 0.022, – | 1.00 / 0.06 / 0.00 | – |

### coincidence

| Arm | Connections a cell at 1,024 / 4,096 / 8,192 / 12,288 / 16,384 | Last 4,096 bytes | Given up / withdrawn by use | ms a byte core / with records | Cells sending a step | S(1,0), S(2,0), S(4,0), S(8,0) in A2 | Informative share d = 1 / 2 / 4 | H3 met |
|---|---|---|---|---|---|---|---|---|
| R (plan 1) | 29.6 / 31.3 / 32.9 / 31.0 / 33.1 | +6.9% | 2,290,745 / 0 | 3.19 / 3.64 | 170.8 | 0.304, 0.029, 0.027, – | 0.57 / 0.99 / 1.00 | – |
| turnover alone (plan 2) | 7.9 / 7.7 / 10.3 / 11.7 / 12.1 | +3.1% | 497,028 / 284,647 | 2.55 / 2.66 | 116.6 | 0.759, 0.044, 0.020, – | 0.98 / 0.97 / 1.00 | – |
| N75 | 29.9 / 30.1 / 30.2 / 30.1 / 30.1 | -0.0% | 8,020,675 / 0 | 3.11 / 3.55 | 51.3 | 0.773, 0.449, 0.022, – | 1.00 / 0.58 / 0.00 | yes |
| N38 | 18.8 / 19.0 / 19.0 / 19.0 / 19.0 | +0.0% | 8,053,029 / 0 | 2.05 / 2.43 | 25.8 | 0.758, 0.291, 0.022, – | 1.00 / 0.00 / 0.00 | yes |
| N75T | 8.4 / 13.2 / 16.5 / 18.0 / 18.8 | +4.1% | 483,319 / 414,637 | 3.19 / 3.45 | 52.3 | 0.982, 0.758, 0.057, – | 1.00 / 0.91 / 0.00 | – |
| N38T | 8.0 / 10.6 / 12.6 / 13.6 / 14.1 | +3.5% | 492,977 / 455,800 | 1.59 / 1.83 | 23.5 | 0.995, 0.605, 0.022, – | 1.00 / 0.38 / 0.00 | – |


"H3 met" is judged for the arms with E63's making only (the hypothesis is about the scaling alone). S(8,0): no cells at that distance in window A2.

| | Outcome |
|---|---|
| H3 | **Held in both forms.** With E63's making unchanged, holding the total incoming strength settles the number at 30.4 (C_E 3/4) and 19.2 (3/8) in the first form and 30.1 and 19.0 in the coincidence form, all within 0.1% over the last 4,096 bytes, with S(1,0) 0.75 to 0.77 (R: 0.687 and 0.304) |
| C1 | Held |
| C2 | Held |
| C3 | Held |
| C4 | Failed: S(4,0) is 0.022 in every arm of the first form (chance 0.0217) |
| C5 | Failed: in the first form the scaled arms with E63's making cost more than R (2.01 and 1.29 against 1.26 ms); in the coincidence form less (3.11 and 2.05 against 3.19) |
| C6 | Held: with turnover added, S(2,0) in the first form is 0.577 (C_E 3/4) and 0.522 (3/8), above the scaling alone (0.390, 0.256) and turnover alone (0.378) |

**Readings (development level, one seed).**

1. **The number of connections is set by competition.** With E63's making a cell still makes one connection every 128 steps, about 500 a byte in the tissue, and about as many are given up (8.0 million over 16,384 bytes): each new connection, scaled in, pushes the weakest below 1/128. The number settles where that happens, 30 at C_E 3/4 and 19 at 3/8, lower than the bound of the threshold check (96, 48) because the strengths are unequal.
2. **Scaling and turnover together give the best near cells measured so far.** S(1,0) 0.98 to 0.995 in all four turnover arms; S(2,0) 0.52 to 0.58 in the first form and 0.61 to 0.76 in the coincidence form, where 91% of the cells two connections from the lines are informative at C_E 3/4.
3. **Nothing reaches four connections.** S(4,0) is at chance in every arm (0.022; 0.057 in the coincidence form at C_E 3/4 with turnover). Step 1 of the roadmap needs the byte carried further; neither rule tried here does it.
4. **Cost.** The cheapest arms are now cheaper than E63's making: first form with C_E 3/8 and turnover 0.96 ms a byte (R 1.26), coincidence form 1.59 (R 3.19).

**Limits.** One seed; 16,384 bytes; the slow rise of plan 2's turnover remains (+1 to +4% in the last 4,096 bytes of the turnover arms).
