# E51 Part B, B1 and B2 on three seeds

Read by `e51_partb_seeds.py read 73129 93683 116089` from the receipts.

## B1. The share of rehearsal

Reduction score at update 4,096 as a percentage of row 0 (kept: within 5 points of row 0); lowest probe; first probe row with the unrelated task's held score at 80%.

| seed | rule | row 0 | 1/2 (addendum) | 1/4 | 1/8 | 1/16 | 1/32 | none (E43) | least share that keeps | PB.1 (1/4 keeps) | PB.2 (1/32 does not keep) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 73129 | S | 96.6 | 97.1 (100.5%, kept) | 89.0 (92.2%) | 86.8 (89.8%) | 63.2 (65.4%) | 44.2 (45.8%) | 0.0 (0.0%) | 1/2 | failed | held |
| 73129 | B | 98.1 | 96.7 (98.6%, kept) | 88.9 (90.6%) | 78.4 (79.9%) | 57.5 (58.6%) | 34.6 (35.3%) | 0.0 (0.0%) | 1/2 | failed | held |
| 93683 | S | 98.7 | 98.9 (100.3%, kept) | 94.9 (96.2%, kept) | 93.3 (94.5%) | 74.0 (75.0%) | 67.5 (68.4%) | 0.0 (0.0%) | 1/4 | held | held |
| 93683 | B | 97.8 | 98.3 (100.5%, kept) | 92.9 (94.9%, kept) | 86.2 (88.1%) | 83.4 (85.3%) | 61.2 (62.6%) | 0.0 (0.0%) | 1/4 | held | held |
| 116089 | S | 99.6 | 96.1 (96.5%, kept) | 92.4 (92.7%) | 85.2 (85.6%) | 70.7 (71.0%) | 2.6 (2.6%) | 0.0 (0.0%) | 1/2 | failed | held |
| 116089 | B | 94.0 | 98.7 (105.0%, kept) | 84.4 (89.8%) | 82.6 (87.9%) | 62.6 (66.6%) | 41.5 (44.2%) | 0.0 (0.0%) | 1/2 | failed | held |

| seed | rule | share | lowest probe (%) | unrelated held at 80%: first probe row | unrelated held at 4,096 (%) |
|---|---|---|---|---|---|
| 73129 | S | 1/4 | 55.50 | 512 | 98.7 |
| 73129 | S | 1/8 | 30.50 | 512 | 99.0 |
| 73129 | S | 1/16 | 19.75 | 448 | 100.0 |
| 73129 | S | 1/32 | 7.00 | 448 | 100.0 |
| 73129 | B | 1/4 | 61.50 | 256 | 100.0 |
| 73129 | B | 1/8 | 40.50 | 256 | 99.7 |
| 73129 | B | 1/16 | 25.75 | 256 | 100.0 |
| 73129 | B | 1/32 | 18.00 | 256 | 100.0 |
| 93683 | S | 1/4 | 68.25 | 448 | 99.0 |
| 93683 | S | 1/8 | 41.50 | 448 | 99.3 |
| 93683 | S | 1/16 | 25.50 | 384 | 100.0 |
| 93683 | S | 1/32 | 5.75 | 384 | 100.0 |
| 93683 | B | 1/4 | 81.75 | 320 | 100.0 |
| 93683 | B | 1/8 | 65.75 | 320 | 100.0 |
| 93683 | B | 1/16 | 44.00 | 256 | 99.7 |
| 93683 | B | 1/32 | 30.75 | 256 | 100.0 |
| 116089 | S | 1/4 | 62.25 | 384 | 100.0 |
| 116089 | S | 1/8 | 15.00 | 384 | 100.0 |
| 116089 | S | 1/16 | 1.25 | 384 | 99.7 |
| 116089 | S | 1/32 | 0.00 | 384 | 100.0 |
| 116089 | B | 1/4 | 67.75 | 256 | 100.0 |
| 116089 | B | 1/8 | 47.25 | 256 | 100.0 |
| 116089 | B | 1/16 | 30.50 | 256 | 100.0 |
| 116089 | B | 1/32 | 14.50 | 256 | 100.0 |

## B2. Rehearsal at intervals

For each cycle: the probe's first-family share at the end of the gap (%) / the review's updates.

| seed | rule | probe at 0 | cycle 1 | cycle 2 | cycle 3 | cycle 4 | cycle 5 | cycle 6 | cycle 7 | cycle 8 | review shortens | more is left | updates | reduction at 0 / end | unrelated held at end |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 73129 | S | 95.25 | 0.00 / 80 | 0.00 / 128 | 3.75 / 96 | 52.50 / 96 | 74.50 / 64 | 65.00 / 48 | 54.25 / 48 | 58.25 / 32 | yes | yes | 2640 | 96.6 / 87.9 | 99.7 |
| 73129 | B | 98.25 | 0.00 / 80 | 0.25 / 64 | 11.75 / 64 | 46.75 / 64 | 70.00 / 32 | 46.50 / 48 | 77.75 / 48 | 72.50 / 48 | no | yes | 2496 | 98.1 / 91.9 | 98.0 |
| 93683 | S | 98.50 | 0.00 / 96 | 0.25 / 112 | 2.00 / 80 | 54.25 / 48 | 56.25 / 32 | 48.50 / 32 | 66.25 / 32 | 71.00 / 32 | yes | yes | 2512 | 98.7 / 90.5 | 99.3 |
| 93683 | B | 97.75 | 0.00 / 64 | 0.25 / 64 | 15.75 / 48 | 60.75 / 48 | 60.75 / 32 | 75.00 / 16 | 46.50 / 16 | 57.50 / 48 | no | yes | 2384 | 97.8 / 93.6 | 99.7 |
| 116089 | S | 99.25 | 0.00 / 128 | 0.50 / 112 | 21.00 / 80 | 70.50 / 48 | 50.50 / 48 | 53.00 / 32 | 68.25 / 32 | 64.00 / 16 | yes | yes | 2544 | 99.6 / 91.7 | 99.3 |
| 116089 | B | 93.75 | 0.00 / 96 | 0.00 / 112 | 0.00 / 64 | 56.75 / 48 | 61.00 / 48 | 39.25 / 32 | 39.25 / 32 | 64.75 / 32 | yes | yes | 2512 | 94.0 / 87.4 | 100.0 |

The brief's sentence for each seed and rule:

- seed 73129, rule S: with rehearsal at intervals the first family becomes cheaper to restore, and more of it survives a gap.
- seed 73129, rule B: with rehearsal at intervals the first family does not become cheaper to restore, and more of it survives a gap.
- seed 93683, rule S: with rehearsal at intervals the first family becomes cheaper to restore, and more of it survives a gap.
- seed 93683, rule B: with rehearsal at intervals the first family does not become cheaper to restore, and more of it survives a gap.
- seed 116089, rule S: with rehearsal at intervals the first family becomes cheaper to restore, and more of it survives a gap.
- seed 116089, rule B: with rehearsal at intervals the first family becomes cheaper to restore, and more of it survives a gap.

Every cycle's review reached its goal: seed 73129 S: True, seed 73129 B: True, seed 93683 S: True, seed 93683 B: True, seed 116089 S: True, seed 116089 B: True.
