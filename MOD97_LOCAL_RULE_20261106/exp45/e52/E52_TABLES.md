Receipts read: 8 complete runs in `exp45/e52`. Seed 93683. Written by `exp52_tables.py`.

## Table 1. Every run

Updates to 80 by the three-row rule; “–” is not reached within the dose. Final levels are means of the last three rows. The length test at 40 letters is given at the last row, as the mean of the last three rows, and with its range over the last twelve rows; one row of it is a weak figure. RMS is the root mean square change of R from row to row over the last 16 rows.

| Width, f | Rule | Rate | Decay | Dose | Updates to 80, A | Updates to 80, R | Final A | Final R | Final taught | Length 40: last, mean of 3 (range of 12) | RMS of R | Diverged | Receipt |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| half, 0.50 | S | 0.001 | 0.25 | 40,960 | 10,240 | 13,312 | 91.89 | 91.22 | 93.72 | 85.3, 83.2 (75.7–90.7) | 4.24 | no | `e52_S_h256_f50_lr0.001_93683.json` |
| half, 0.50 | B | 0.001 | 0.25 | 40,960 | 7,424 | 8,704 | 96.56 | 93.97 | 97.11 | 95.7, 95.6 (93.0–98.3) | 2.35 | no | `e52_B_h256_f50_lr0.001_93683.json` |
| half, 0.25 | S | 0.001 | 0.25 | 40,960 | 14,080 | 24,576 | 88.56 | 84.33 | 97.39 | 78.7, 82.3 (74.3–87.7) | 2.45 | no | `e52_S_h256_f25_lr0.001_93683.json` |
| half, 0.25 | B | 0.001 | 0.25 | 40,960 | 6,144 | 7,424 | 97.72 | 96.23 | 98.78 | 98.0, 97.0 (87.0–98.0) | 1.96 | no | `e52_B_h256_f25_lr0.001_93683.json` |
| present, 0.50 | S | 0.001 | 0.25 | 12,288 | 4,864 | 4,864 | 94.56 | 94.18 | 98.44 | 95.7, 91.1 (82.3–95.7) | 3.17 | no | `e52_S_h512_f50_lr0.001_93683.json` |
| present, 0.50 | B | 0.001 | 0.25 | 12,288 | 4,096 | 4,096 | 93.11 | 94.27 | 95.89 | 91.3, 92.0 (91.3–97.7) | 1.77 | no | `e52_B_h512_f50_lr0.001_93683.json` |
| present, 0.25 | S | 0.001 | 0.25 | 12,288 | 9,984 | – | 83.06 | 78.78 | 96.33 | 70.3, 73.2 (68.0–79.7) | 3.14 | no | `e52_S_h512_f25_lr0.001_93683.json` |
| present, 0.25 | B | 0.001 | 0.25 | 12,288 | 3,584 | 4,352 | 92.44 | 90.62 | 97.33 | 88.3, 90.9 (83.7–95.0) | 4.64 | no | `e52_B_h512_f25_lr0.001_93683.json` |

## Table 2. Runs at 0.0003 at E41's dose (their dose is 10/3 of E41's)

| Width, f | Rule | Control | Final A at E41's dose | Final R at E41's dose | Length 40 there: last row, mean of 3 | Final A, R at the end of the run |
|---|---|---|---|---|---|---|

## Table 3. The seven norms (book / l1 / input map / recurrent / lm / l2 / l3)

| Width, f | Rule | Rate | At 80 on A | At 80 on R | At the end | At E41's dose (0.0003 only) |
|---|---|---|---|---|---|---|
| half, 0.50 | S | 0.001 | 45.5 / 3.5 / 25.2 / 22.0 / 22.2 / 6.8 / 41.6 | 45.9 / 3.3 / 24.7 / 21.5 / 21.1 / 6.2 / 38.9 | 49.5 / 3.3 / 22.9 / 20.0 / 18.0 / 5.0 / 37.4 |  |
| half, 0.50 | B | 0.001 | 45.5 / 4.5 / 23.4 / 20.9 / 21.7 / 6.1 / 44.0 | 45.6 / 4.3 / 23.2 / 20.5 / 21.4 / 5.7 / 39.5 | 51.5 / 3.5 / 22.9 / 19.0 / 17.9 / 4.2 / 40.7 |  |
| half, 0.25 | S | 0.001 | 46.0 / 3.4 / 24.8 / 21.8 / 21.2 / 6.4 / 36.6 | 47.8 / 3.4 / 23.6 / 21.0 / 19.8 / 5.6 / 35.0 | 50.6 / 3.4 / 22.9 / 20.3 / 19.0 / 5.5 / 36.7 |  |
| half, 0.25 | B | 0.001 | 45.3 / 4.7 / 21.3 / 20.4 / 22.8 / 6.1 / 46.0 | 45.3 / 4.4 / 21.4 / 19.9 / 22.0 / 5.8 / 40.7 | 50.4 / 3.5 / 22.1 / 17.8 / 17.9 / 4.0 / 46.0 |  |
| present, 0.50 | S | 0.001 | 45.0 / 4.5 / 39.5 / 30.0 / 29.2 / 11.3 / 34.8 | 45.0 / 4.5 / 39.5 / 30.0 / 29.2 / 11.3 / 34.8 | 45.4 / 3.4 / 42.1 / 29.0 / 27.5 / 10.0 / 34.4 |  |
| present, 0.50 | B | 0.001 | 45.2 / 6.0 / 34.5 / 28.5 / 28.9 / 10.8 / 40.5 | 45.2 / 6.0 / 34.5 / 28.5 / 28.9 / 10.8 / 40.5 | 45.8 / 4.3 / 38.3 / 27.3 / 26.1 / 9.1 / 30.9 |  |
| present, 0.25 | S | 0.001 | 45.3 / 3.6 / 39.8 / 29.2 / 25.7 / 10.0 / 27.9 | – | 45.4 / 3.5 / 39.8 / 29.0 / 25.5 / 9.8 / 28.1 |  |
| present, 0.25 | B | 0.001 | 45.1 / 6.5 / 33.6 / 28.3 / 29.2 / 10.6 / 39.0 | 45.1 / 5.8 / 34.0 / 27.6 / 29.1 / 10.5 / 35.2 | 45.6 / 4.2 / 38.3 / 26.2 / 26.0 / 9.6 / 28.7 |  |

## Table 4. Readings for each case (a rule, a width and a taught fraction)

| Width, f | Rule | Best rate on R | Best rate on A | 0.0003 against 0.001: R, A | 0.003 against 0.001: R, A | 0.01 against 0.001: R, A | Gains from its own rate |
|---|---|---|---|---|---|---|---|
| half, 0.50 | S | 0.001 | 0.001 | not run | not run | not run | no (read on R and A) |
| half, 0.50 | B | 0.001 | 0.001 | not run | not run | not run | no (read on R and A) |
| half, 0.25 | S | 0.001 | 0.001 | not run | not run | not run | no (read on R and A) |
| half, 0.25 | B | 0.001 | 0.001 | not run | not run | not run | no (read on R and A) |
| present, 0.50 | S | 0.001 | 0.001 | not run | not run | not run | no (read on R and A) |
| present, 0.50 | B | 0.001 | 0.001 | not run | not run | not run | no (read on R and A) |
| present, 0.25 | S | not reached | 0.001 | not run | not run | not run | no (read on A (no rate reaches 80 on R)) |
| present, 0.25 | B | 0.001 | 0.001 | not run | not run | not run | no (read on R and A) |

## Table 5. The lag (the sweep's updates to 80 over back-propagation's) at each width and fraction

| Width, f | Score | Lag at 0.001 | Sweep's best rate | Back-propagation's best rate | Lag at the rules' own best rates | Reading |
|---|---|---|---|---|---|---|
| half, 0.50 | R | 1.53 | 0.001 | 0.001 | 1.53 | the same within the scatter of one seed |
| half, 0.50 | A | 1.38 | 0.001 | 0.001 | 1.38 | the same within the scatter of one seed |
| half, 0.25 | R | 3.31 | 0.001 | 0.001 | 3.31 | the same within the scatter of one seed |
| half, 0.25 | A | 2.29 | 0.001 | 0.001 | 2.29 | the same within the scatter of one seed |
| present, 0.50 | R | 1.19 | 0.001 | 0.001 | 1.19 | the same within the scatter of one seed |
| present, 0.50 | A | 1.19 | 0.001 | 0.001 | 1.19 | the same within the scatter of one seed |
| present, 0.25 | R | above 2.71 (the sweep not reached) | none | 0.001 | above 2.71 (the sweep reached at no rate; bound from its longest run) | not decided |
| present, 0.25 | A | 2.79 | 0.001 | 0.001 | 2.79 | the same within the scatter of one seed |

## Table 6. The control (rate × decay held at 0.00025) beside the main runs: half width, half of the pairs

| Rule | Rate | Decay | Updates to 80, A | Updates to 80, R | Class against 0.001: R, A | Final A | Final R | Length 40: last, mean of 3 | Norms at the end |
|---|---|---|---|---|---|---|---|---|---|

Further runs at 0.01 called for by the rule (best rate 0.003 on R, or on A where the case is not reached on R; a tie is not a best rate): none.

