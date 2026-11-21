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

## Outcome

(to be entered after the runs)
