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

## Outcome

(to be entered after the runs)
