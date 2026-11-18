# E52, the other two seeds: the two learning rules at rates 0.001 and 0.003 (18 November 2026)

**Status.** This report covers the runs that the coordinator's decision 19 of 18 November called for. The decision is on the default branch at `0cf288ce`, in `PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/COORDINATION_20261117/COORDINATOR_DECISIONS_20261118.md`. Its words to this worker were:

> "run the other two seeds as your brief's rule prescribes (seeds 73129 and 116089; the half width with both fractions, the present width with a quarter of the pairs). Where the brief leaves the rates open, run 0.003 only, beside E41's runs at 0.001. Start every run with the option to resume."

All 12 runs are complete; none diverged and none was interrupted. The report is the worker's and has not been reviewed. It does not stand in for the programme's independent audit, and it claims no checker gate. The first seed, 93683, is reported in `E52_REPORT.md`, at commit `2d36fcfa` after the reviewer's verification. That report defines the terms used here: the network, the two rules, the scores A and R, updates to 80, a case, the lag, birth and life. Its rules for the readings are applied here unchanged.

- **Worker branch:** `claude/e52-learning-rates-sdga8c`. The driver began at commit `3abe1674`. Each receipt records the commit checked out when its run started. Receipts in progress were committed during the runs, so the 12 receipts name five commits: `3abe1674` (3 runs), `25402e3f` (3), `109793f4` (3), `c23c02b4` (2) and `05655042` (1). Every receipt records no source modified against its commit, and the sources are identical in all five.
- **Host:** Linux x86-64 with 4 cores, Python 3.13.18 and torch 2.15.0+cpu. Each process ran one thread, with at most three processes at once.
- **Runner:** `exp52_resume.py` (sha256 `6531525f8bffec5f…`, unchanged since the first seed), with a checkpoint every 4,096 updates.
- **Driver:** `exp52_drive3.py` (new file, `1401ba576d687ab1…`).
- **Readings:** `exp52_seeds_tables.py` (new file, `115e1fda949da2d7…`), which writes `exp45/e52_seeds/E52_SEEDS_TABLES.md` and `E52_SEEDS_SUMMARY.json`.
- No existing source file was edited.

## 1. What was run

| Factor | Values |
|---|---|
| Rule | S (the sweep), B (back-propagation) |
| Seed | 73129, 116089 |
| Width and taught fraction | half width (H = 256, U = 128) with f = 0.50 and f = 0.25; present width (H = 512, U = 256) with f = 0.25 |
| Learning rate | 0.003, with decay 0.25 on every parameter but the byte table |
| Dose | E41's: 40,960 updates at half width, 12,288 at present width; a scored row every 256 updates |

That gives 2 rules × 2 seeds × 3 widths and fractions = 12 runs. The receipts are in `exp45/e52_seeds/`, apart from the first seed's in `exp45/e52/`, because `exp52_tables.py` keys the first seed's receipts by rule, width, fraction and rate without a seed.

**Which widths and fractions.** The brief's rule asks for the other seeds "for every width and fraction at which either rule gains from its own rate or the lag changes class". On seed 93683 those were:
- half width, f = 0.50 (both rules gain);
- half width, f = 0.25 (the sweep gains; the lag on R is smaller);
- present width, f = 0.25 (the sweep gains; the lag is smaller).

Present width with f = 0.50 did not qualify.

**The runs at 0.001** are E41's receipts of the same rule, width, fraction and seed (`exp45/e41/e41_<rule>_h<H>_f<fp>_<seed>.json`). On seed 93683 `exp52_rate.py` reproduced E41's runs exactly: the same final tensors and the same counts and batch digests at every row (check 1b of `E52_REPORT.md`). E41's receipts do not record the length test or the norms, so those figures exist only at 0.003.

**Checks inside every run, all of which held.**
- The optimizer groups (rate 0.003, decay 0.25, betas (0.9, 0.98), the byte table without decay) were asserted at the start and at the end.
- The batch digest at row 4,096 was asserted equal to E41's receipt of the same rule, width, seed and fraction.
- The length test at 12 letters uses the same 300 items as A's add count, and the two counts were asserted equal at every row.
- No run diverged.

**Interruptions.** None. The launch log `exp45/e52_seeds/logs/LAUNCHES.txt` records 12 starts and 12 exits with code 0, between 14:24:54 and 16:30:14 UTC on 18 November. It records no resumption and no move to `interrupted/`, and no receipt carries a `resumed` record. The evidence that the verification's finding 9 asks for, for each resumed run, is therefore not needed.

**Cost.** 5.94 CPU-hours for the 12 runs (updates and scoring, from the receipts), against a forecast of 6.5 sent to the reviewer before the runs.

## 2. How the readings are made, and the margin in rows

The rules are the brief's, restricted to the two rates run on these seeds:
- **Best rate on a score** is the rate with the fewest updates to 80. Counts within one row (256 updates) are tied, and a tie that includes 0.001 is given to 0.001.
- **Faster / the same / slower** compares the run at 0.003 with the run at 0.001 by `exp41_scale.describe_step`, with the band two thirds to 1.5.
- **The rule gains from its own rate** if it is faster at 0.003 on R and not slower on A; where neither rate reaches 80 on R, if it is faster on A.
- **The lag** at the rules' own best rates is *smaller* (*larger*) below two thirds of (above 1.5 times) the lag at 0.001. It is *gone* if it is smaller and at most 1.0, and otherwise *the same within the scatter of one seed*.
- **Bounds:** where a count is missing, the lag is a bound from the run's last recordable count (its dose less 512). A lower bound is rounded down and an upper bound up.

**The margin of a reading** is the fewest whole rows by which one count to 80 (of one run, on one score) must move for the reading to change. This follows the reviewer's verification of the first seed (`E52_VERIFICATION_20261118.md`, finding 1).
- A count c may move to c ± 256 k, never below 256.
- A count moved past the last recordable row becomes "not reached". A count that was not reached becomes reached at the last recordable row less 256 (k − 1).
- Every reading of the width and fraction is recomputed from the moved counts, best rates included.
- "None within 200" means that no move of a single count by up to 200 rows changes the reading. For example, under back-propagation at present width with f = 0.25 on seed 73129, gaining would need its counts on both R and A to move.

For seed 93683 these margins reproduce the verification's figures: 1, 1, 2 and 3 rows for the readings it names, and 6, 19 and 10 rows for the sweep's gains.

## 3. Every run

The receipt of each run is named in the last column; the final weights of each run at 0.003 are in the `.pt` file of the same name. Final levels are means of the last three rows. One row of the length test is a weak figure: its range over the last twelve rows is 8.7 to 21.7 points in the 12 new runs, and 5.3 to 21.7 points over all 24 runs of this table that have a length test. The largest is seed 73129, the sweep at half width with f = 0.25 (69.3 to 91.0); the smallest is back-propagation at half width with f = 0.50, at 0.001, on seed 93683. The norms are book / l1 / input map / recurrent / lm / l2 / l3, at the last row. E41's receipts (rate 0.001) record neither.


| Seed | Width, f | Rule | Rate | Dose | Updates to 80, A | Updates to 80, R | Final A | Final R | Final taught | Length 40: last, mean of 3 (range of 12) | Norms at the end (book / l1 / input / recurrent / lm / l2 / l3) | Resumed at | Receipt |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 73129 | half, 0.50 | S | 0.001 | 40,960 | 12,288 | 15,872 | 96.17 | 93.82 | 98.11 | not scored (E41) | not recorded (E41) | – | `exp45/e41/e41_S_h256_f50_73129.json` |
| 73129 | half, 0.50 | S | 0.003 | 40,960 | 6,656 | 11,520 | 88.11 | 85.05 | 90.78 | 90.0, 86.1 (78.3–92.7) | 71.4 / 4.2 / 37.7 / 23.3 / 21.1 / 8.9 / 77.9 | – | `exp45/e52_seeds/e52_S_h256_f50_lr0.003_73129.json` |
| 73129 | half, 0.50 | B | 0.001 | 40,960 | 11,520 | 13,056 | 94.11 | 94.50 | 95.17 | not scored (E41) | not recorded (E41) | – | `exp45/e41/e41_B_h256_f50_73129.json` |
| 73129 | half, 0.50 | B | 0.003 | 40,960 | 6,912 | 8,448 | 94.67 | 94.50 | 96.22 | 99.0, 94.7 (87.3–99.0) | 80.1 / 3.7 / 31.3 / 21.6 / 20.6 / 7.1 / 60.5 | – | `exp45/e52_seeds/e52_B_h256_f50_lr0.003_73129.json` |
| 73129 | half, 0.25 | S | 0.001 | 40,960 | 17,408 | 35,840 | 87.94 | 82.83 | 97.00 | not scored (E41) | not recorded (E41) | – | `exp45/e41/e41_S_h256_f25_73129.json` |
| 73129 | half, 0.25 | S | 0.003 | 40,960 | 13,312 | 22,272 | 84.72 | 81.87 | 91.06 | 80.0, 83.8 (69.3–91.0) | 65.0 / 4.7 / 34.3 / 23.1 / 20.9 / 9.9 / 64.3 | – | `exp45/e52_seeds/e52_S_h256_f25_lr0.003_73129.json` |
| 73129 | half, 0.25 | B | 0.001 | 40,960 | 10,240 | 16,896 | 93.44 | 89.80 | 97.17 | not scored (E41) | not recorded (E41) | – | `exp45/e41/e41_B_h256_f25_73129.json` |
| 73129 | half, 0.25 | B | 0.003 | 40,960 | 6,656 | 7,680 | 92.83 | 92.70 | 95.28 | 91.3, 92.9 (88.3–99.0) | 83.5 / 3.9 / 31.8 / 21.9 / 21.2 / 8.9 / 56.1 | – | `exp45/e52_seeds/e52_B_h256_f25_lr0.003_73129.json` |
| 73129 | present, 0.25 | S | 0.001 | 12,288 | 8,704 | – | 85.11 | 80.35 | 97.50 | not scored (E41) | not recorded (E41) | – | `exp45/e41/e41_S_h512_f25_73129.json` |
| 73129 | present, 0.25 | S | 0.003 | 12,288 | 4,096 | 6,656 | 92.72 | 89.62 | 96.50 | 97.3, 93.3 (86.0–97.3) | 43.0 / 6.9 / 54.3 / 39.5 / 33.3 / 21.0 / 39.5 | – | `exp45/e52_seeds/e52_S_h512_f25_lr0.003_73129.json` |
| 73129 | present, 0.25 | B | 0.001 | 12,288 | 4,352 | 7,680 | 85.61 | 83.68 | 95.61 | not scored (E41) | not recorded (E41) | – | `exp45/e41/e41_B_h512_f25_73129.json` |
| 73129 | present, 0.25 | B | 0.003 | 12,288 | 8,192 | 7,424 | 82.56 | 81.57 | 89.50 | 82.7, 80.6 (77.0–92.3) | 47.3 / 4.2 / 46.5 / 37.0 / 31.9 / 20.3 / 29.7 | – | `exp45/e52_seeds/e52_B_h512_f25_lr0.003_73129.json` |
| 116089 | half, 0.50 | S | 0.001 | 40,960 | 10,496 | 20,480 | 90.78 | 88.00 | 94.17 | not scored (E41) | not recorded (E41) | – | `exp45/e41/e41_S_h256_f50_116089.json` |
| 116089 | half, 0.50 | S | 0.003 | 40,960 | 9,216 | 11,008 | 89.78 | 88.87 | 91.83 | 83.3, 88.8 (79.3–94.3) | 53.2 / 4.8 / 35.3 / 23.2 / 20.9 / 7.7 / 78.4 | – | `exp45/e52_seeds/e52_S_h256_f50_lr0.003_116089.json` |
| 116089 | half, 0.50 | B | 0.001 | 40,960 | 7,680 | 8,960 | 97.61 | 94.73 | 98.72 | not scored (E41) | not recorded (E41) | – | `exp45/e41/e41_B_h256_f50_116089.json` |
| 116089 | half, 0.50 | B | 0.003 | 40,960 | 7,424 | 4,352 | 93.28 | 92.43 | 93.94 | 92.3, 93.7 (84.0–96.0) | 102.3 / 3.6 / 32.0 / 21.9 / 19.7 / 7.4 / 57.9 | – | `exp45/e52_seeds/e52_B_h256_f50_lr0.003_116089.json` |
| 116089 | half, 0.25 | S | 0.001 | 40,960 | 24,832 | 30,464 | 84.28 | 82.43 | 94.94 | not scored (E41) | not recorded (E41) | – | `exp45/e41/e41_S_h256_f25_116089.json` |
| 116089 | half, 0.25 | S | 0.003 | 40,960 | 9,216 | 10,240 | 91.72 | 89.97 | 95.33 | 86.0, 86.1 (85.0–93.7) | 47.1 / 5.8 / 32.8 / 23.7 / 20.9 / 8.8 / 58.3 | – | `exp45/e52_seeds/e52_S_h256_f25_lr0.003_116089.json` |
| 116089 | half, 0.25 | B | 0.001 | 40,960 | 7,680 | 9,216 | 94.78 | 91.75 | 98.33 | not scored (E41) | not recorded (E41) | – | `exp45/e41/e41_B_h256_f25_116089.json` |
| 116089 | half, 0.25 | B | 0.003 | 40,960 | 7,168 | 7,168 | 94.56 | 93.53 | 96.61 | 93.0, 93.4 (85.0–96.0) | 73.5 / 3.5 / 30.8 / 22.0 / 21.3 / 7.9 / 59.1 | – | `exp45/e52_seeds/e52_B_h256_f25_lr0.003_116089.json` |
| 116089 | present, 0.25 | S | 0.001 | 12,288 | – | – | 78.56 | 73.92 | 97.78 | not scored (E41) | not recorded (E41) | – | `exp45/e41/e41_S_h512_f25_116089.json` |
| 116089 | present, 0.25 | S | 0.003 | 12,288 | 4,096 | 4,608 | 92.61 | 92.65 | 94.39 | 92.3, 89.1 (82.7–94.3) | 44.2 / 7.3 / 54.0 / 39.8 / 32.4 / 20.3 / 37.3 | – | `exp45/e52_seeds/e52_S_h512_f25_lr0.003_116089.json` |
| 116089 | present, 0.25 | B | 0.001 | 12,288 | 4,864 | 5,888 | 88.17 | 85.95 | 96.56 | not scored (E41) | not recorded (E41) | – | `exp45/e41/e41_B_h512_f25_116089.json` |
| 116089 | present, 0.25 | B | 0.003 | 12,288 | 4,096 | 4,608 | 91.06 | 90.13 | 92.11 | 90.0, 88.4 (86.7–97.0) | 49.7 / 4.2 / 46.1 / 36.3 / 31.8 / 20.5 / 28.1 | – | `exp45/e52_seeds/e52_B_h512_f25_lr0.003_116089.json` |
| 93683 | half, 0.50 | S | 0.001 | 40,960 | 10,240 | 13,312 | 91.89 | 91.22 | 93.72 | 85.3, 83.2 (75.7–90.7) | 49.5 / 3.3 / 22.9 / 20.0 / 18.0 / 5.0 / 37.4 | – | `exp45/e52/e52_S_h256_f50_lr0.001_93683.json` |
| 93683 | half, 0.50 | S | 0.003 | 40,960 | 6,912 | 7,424 | 91.72 | 90.65 | 93.72 | 88.7, 91.4 (80.7–95.7) | 60.8 / 4.2 / 33.5 / 22.9 / 20.2 / 8.6 / 69.7 | – | `exp45/e52/e52_S_h256_f50_lr0.003_93683.json` |
| 93683 | half, 0.50 | B | 0.001 | 40,960 | 7,424 | 8,704 | 96.56 | 93.97 | 97.11 | 95.7, 95.6 (93.0–98.3) | 51.5 / 3.5 / 22.9 / 19.0 / 17.9 / 4.2 / 40.7 | – | `exp45/e52/e52_B_h256_f50_lr0.001_93683.json` |
| 93683 | half, 0.50 | B | 0.003 | 40,960 | 4,864 | 4,864 | 92.50 | 91.25 | 92.89 | 90.3, 89.9 (87.7–98.3) | 114.7 / 3.6 / 31.1 / 21.7 / 19.6 / 7.4 / 63.6 | – | `exp45/e52/e52_B_h256_f50_lr0.003_93683.json` |
| 93683 | half, 0.25 | S | 0.001 | 40,960 | 14,080 | 24,576 | 88.56 | 84.33 | 97.39 | 78.7, 82.3 (74.3–87.7) | 50.6 / 3.4 / 22.9 / 20.3 / 19.0 / 5.5 / 36.7 | – | `exp45/e52/e52_S_h256_f25_lr0.001_93683.json` |
| 93683 | half, 0.25 | S | 0.003 | 40,960 | 7,424 | 11,520 | 94.72 | 94.58 | 96.67 | 94.7, 91.9 (88.3–98.3) | 53.5 / 4.9 / 33.6 / 23.8 / 22.1 / 8.7 / 55.2 | – | `exp45/e52/e52_S_h256_f25_lr0.003_93683.json` |
| 93683 | half, 0.25 | B | 0.001 | 40,960 | 6,144 | 7,424 | 97.72 | 96.23 | 98.78 | 98.0, 97.0 (87.0–98.0) | 50.4 / 3.5 / 22.1 / 17.8 / 17.9 / 4.0 / 46.0 | – | `exp45/e52/e52_B_h256_f25_lr0.001_93683.json` |
| 93683 | half, 0.25 | B | 0.003 | 40,960 | 3,840 | 5,632 | 96.61 | 96.30 | 97.56 | 90.7, 96.3 (90.7–99.7) | 70.3 / 3.7 / 30.0 / 23.5 / 21.8 / 7.8 / 49.9 | – | `exp45/e52/e52_B_h256_f25_lr0.003_93683.json` |
| 93683 | present, 0.25 | S | 0.001 | 12,288 | 9,984 | – | 83.06 | 78.78 | 96.33 | 70.3, 73.2 (68.0–79.7) | 45.4 / 3.5 / 39.8 / 29.0 / 25.5 / 9.8 / 28.1 | – | `exp45/e52/e52_S_h512_f25_lr0.001_93683.json` |
| 93683 | present, 0.25 | S | 0.003 | 12,288 | 4,096 | 5,376 | 96.56 | 95.05 | 97.89 | 92.3, 95.4 (89.7–97.7) | 44.4 / 6.5 / 56.1 / 40.4 / 34.5 / 20.9 / 50.9 | – | `exp45/e52/e52_S_h512_f25_lr0.003_93683.json` |
| 93683 | present, 0.25 | B | 0.001 | 12,288 | 3,584 | 4,352 | 92.44 | 90.62 | 97.33 | 88.3, 90.9 (83.7–95.0) | 45.6 / 4.2 / 38.3 / 26.2 / 26.0 / 9.6 / 28.7 | – | `exp45/e52/e52_B_h512_f25_lr0.001_93683.json` |
| 93683 | present, 0.25 | B | 0.003 | 12,288 | 3,072 | 3,072 | 95.17 | 95.08 | 96.06 | 94.0, 94.9 (83.0–95.3) | 50.6 / 4.5 / 48.8 / 36.6 / 32.0 / 18.9 / 44.4 | – | `exp45/e52/e52_B_h512_f25_lr0.003_93683.json` |

## 4. Readings for each seed, with the margin of each in rows

Each cell: the reading, then in brackets the fewest rows by which one count would have to move to change it, and the reading it would become.

| Seed | Width, f | Best rate on R: S; B | Best rate on A: S; B | 0.003 against 0.001, S: R; A | 0.003 against 0.001, B: R; A | S gains | B gains | Lag on R: at 0.001 → own rates | Lag on A: at 0.001 → own rates |
|---|---|---|---|---|---|---|---|---|---|
| 73129 | half, 0.50 | 0.003 (16 → 0.001); 0.003 (17 → 0.001) | 0.003 (21 → 0.001); 0.003 (17 → 0.001) | the same (4 → faster); faster (6 → the same) | faster (1 → the same); faster (3 → the same) | no (4 → yes) | yes (1 → no) | 1.22 → 1.36: the same within the scatter of one seed (9 → larger) | 1.07 → 0.96: the same within the scatter of one seed (7 → gone) |
| 73129 | half, 0.25 | 0.003 (52 → 0.001); 0.003 (35 → 0.001) | 0.003 (15 → 0.001); 0.003 (13 → 0.001) | faster (7 → the same); the same (7 → faster) | faster (14 → the same); faster (1 → the same) | yes (7 → no) | yes (14 → no) | 2.12 → 2.90: the same within the scatter of one seed (3 → larger) | 1.70 → 2.00: the same within the scatter of one seed (6 → larger) |
| 73129 | present, 0.25 | 0.003 (20 → 0.001); 0.001 (1 → 0.003) | 0.003 (17 → 0.001); 0.001 (17 → 0.003) | faster (5 → not decided); faster (7 → the same) | the same (10 → faster); slower (5 → the same) | yes (5 → no) | no (none within 200) | above 1.53 → 0.87: gone (4 → not decided) | 2.00 → 0.94: gone (2 → smaller) |
| 116089 | half, 0.50 | 0.003 (36 → 0.001); 0.003 (17 → 0.001) | 0.003 (4 → 0.001); 0.001 (1 → 0.003) | faster (11 → the same); the same (9 → faster) | faster (7 → the same); the same (10 → faster) | yes (11 → no) | yes (7 → no) | 2.29 → 2.53: the same within the scatter of one seed (5 → larger) | 1.37 → 1.20: the same within the scatter of one seed (9 → gone) |
| 116089 | half, 0.25 | 0.003 (78 → 0.001); 0.003 (7 → 0.001) | 0.003 (60 → 0.001); 0.003 (1 → 0.001) | faster (40 → the same); faster (29 → the same) | the same (5 → faster); the same (9 → faster) | yes (40 → no) | no (5 → yes) | 3.31 → 1.43: smaller (10 → the same within the scatter of one seed) | 3.23 → 1.29: smaller (8 → gone) |
| 116089 | present, 0.25 | 0.003 (28 → 0.001); 0.003 (4 → 0.001) | 0.003 (30 → 0.001); 0.003 (2 → 0.001) | faster (13 → not decided); faster (15 → not decided) | the same (3 → faster); the same (4 → faster) | yes (13 → no) | no (3 → yes) | above 2.00 → 1.00: gone (1 → smaller) | above 2.42 → 1.00: gone (1 → smaller) |
| 93683 | half, 0.50 | 0.003 (22 → 0.001); 0.003 (14 → 0.001) | 0.003 (12 → 0.001); 0.003 (9 → 0.001) | faster (6 → the same); the same (1 → faster) | faster (4 → the same); faster (1 → the same) | yes (6 → no) | yes (4 → no) | 1.53 → 1.53: the same within the scatter of one seed (7 → larger) | 1.38 → 1.42: the same within the scatter of one seed (6 → larger) |
| 93683 | half, 0.25 | 0.003 (50 → 0.001); 0.003 (6 → 0.001) | 0.003 (25 → 0.001); 0.003 (8 → 0.001) | faster (19 → the same); faster (8 → the same) | the same (3 → faster); faster (1 → the same) | yes (19 → no) | no (3 → yes) | 3.31 → 2.05: smaller (2 → the same within the scatter of one seed) | 2.29 → 1.93: the same within the scatter of one seed (4 → smaller) |
| 93683 | present, 0.25 | 0.003 (25 → 0.001); 0.003 (4 → 0.001) | 0.003 (22 → 0.001); 0.003 (1 → 0.001) | faster (10 → not decided); faster (10 → the same) | the same (1 → faster); the same (3 → faster) | yes (10 → no) | no (1 → yes) | above 2.70 → 1.75: smaller (1 → not decided) | 2.79 → 1.33: smaller (4 → gone) |


## 5. The three seeds side by side

Seed 93683 is read here at the rates 0.001 and 0.003 only, so that all three seeds are read alike. Its full readings, with 0.0003 and 0.01, are in `E52_REPORT.md`; at these two rates its readings are the same as there. Each entry gives the reading, then the margin in rows.

| Width, f | Reading | 73129 | 116089 | 93683 | Count |
|---|---|---|---|---|---|
| half, 0.50 | The sweep gains | no (4) | yes (11) | yes (6) | 2 of 3 |
| half, 0.50 | Back-propagation gains | yes (1) | yes (7) | yes (4) | 3 of 3 |
| half, 0.50 | Lag on R | 1.22 → 1.36, the same (9) | 2.29 → 2.53, the same (5) | 1.53 → 1.53, the same (7) | the same, 3 of 3 |
| half, 0.50 | Lag on A | 1.07 → 0.96, the same (7) | 1.37 → 1.20, the same (9) | 1.38 → 1.42, the same (6) | the same, 3 of 3 |
| half, 0.25 | The sweep gains | yes (7) | yes (40) | yes (19) | 3 of 3 |
| half, 0.25 | Back-propagation gains | yes (14) | no (5) | no (3) | 1 of 3 |
| half, 0.25 | Lag on R | 2.12 → 2.90, the same (3) | 3.31 → 1.43, smaller (10) | 3.31 → 2.05, smaller (2) | smaller, 2 of 3 |
| half, 0.25 | Lag on A | 1.70 → 2.00, the same (6) | 3.23 → 1.29, smaller (8) | 2.29 → 1.93, the same (4) | smaller, 1 of 3 |
| present, 0.25 | The sweep gains | yes (5) | yes (13) | yes (10) | 3 of 3 |
| present, 0.25 | Back-propagation gains | no (none within 200) | no (3) | no (1) | 0 of 3 |
| present, 0.25 | Lag on R | above 1.53 → 0.87, gone (4) | above 2.00 → 1.00, gone (1) | above 2.70 → 1.75, smaller (1) | smaller or gone, 3 of 3 |
| present, 0.25 | Lag on A | 2.00 → 0.94, gone (2) | above 2.42 → 1.00, gone (1) | 2.79 → 1.33, smaller (4) | smaller or gone, 3 of 3 |

**Where the two rules end up at their own rates.** At present width with f = 0.25, the lag at the rules' own rates is 0.87 and 1.00 on R on seeds 73129 and 116089.
- On seed 73129, back-propagation's best rate on R is 0.001: its counts at the two rates are 7,680 and 7,424, which are tied, and the tie goes to 0.001. On A it is slower at 0.003 than at 0.001 (8,192 against 4,352).
- On seed 116089 both rules reach 80 at 0.003 at the same row on both scores (4,608 on R, 4,096 on A).
- In both seeds the sweep does not reach 80 on R at 0.001 within E41's 12,288 updates, so the lag at 0.001 is only a lower bound.

**Gains, in the brief's sense of a case** (a rule at a width and fraction), seed by seed:
- Back-propagation gains at half width with f = 0.50 on all three seeds; on seed 73129 by one row.
- It gains at half width with f = 0.25 on seed 73129 only (14 rows).
- It does not gain at present width with f = 0.25 on any seed; on seed 93683 this rests on one row.
- The sweep gains in both of its cases at a quarter of the pairs on all three seeds, and in its case at half width with f = 0.50 on two seeds (not on seed 73129, four rows).

The "gone" on seed 116089 is one row from "smaller". The "gone" on seed 73129 is four rows from "not decided" on R and two rows from "smaller" on A.

**The brief's sentence, for each seed and each width and fraction** ("at this width and fraction the sweep's lag behind back-propagation is [the same within the scatter of one seed / smaller / gone / larger] when each rule is run at its own best learning rate among those tried"), with "those tried" here meaning 0.001 and 0.003:
- **Half width, half of the pairs:** the same within the scatter of one seed, on R and on A, on all three seeds.
- **Half width, a quarter of the pairs:**
  - seed 93683: smaller on R, the same on A;
  - seed 116089: smaller on R and on A;
  - seed 73129: the same on R and on A. On this seed back-propagation gains as well, so the lag on R rises, from 2.12 to 2.90; it is three rows from "larger".
- **Present width, a quarter of the pairs:** gone on seeds 73129 and 116089 (on R and on A); smaller on seed 93683 (on R and on A).

The brief fixes no rule for combining seeds, so the counts in the last column are counts and not a verdict. For comparison only, E41's tables called a description the seeds' when two of three seeds gave it.

## 6. The explanation under test, seed by seed

The brief states the explanation as follows: the sweep's lag at the half width and at a quarter of the pairs is in part a matter of the learning rate. It is refuted for a seed if the sweep gains at no width and fraction, or if, wherever the sweep gains, back-propagation gains too and the lag is not smaller.
- **Seed 73129:** the sweep gains at half width with f = 0.25 and at present width with f = 0.25.
  - The explanation is not refuted on this seed, but that rests on the present width with f = 0.25 alone, where back-propagation does not gain and the lag is gone.
  - That verdict has a margin of five rows. If the sweep's count on R at 0.003 at present width moved from 6,656 to 7,936, its gain there would end, and the explanation would be refuted on this seed.
  - At the case the explanation names (half width, f = 0.25), back-propagation gains too, and the lag is not smaller. It is the same on R and on A: on R it rises from 2.12 to 2.90, three rows from "larger".
- **Seed 116089:** the sweep gains at all three widths and fractions. At half width with f = 0.25 and at present width with f = 0.25, back-propagation does not gain and the lag is smaller or gone. Not refuted, and no move of a single count changes this verdict.
- **Seed 93683** (from `E52_REPORT.md`): not refuted, and no move of a single count changes this verdict. At half width with f = 0.25 the lag on R is smaller, two rows from "the same".

At the width and fraction that the explanation names, the lag on R is smaller on two seeds (93683 and 116089) and the same on one (73129). E41's three seeds at 0.001 already give lags on R there of 2.12, 3.31 and 3.31.

## 7. What these runs cannot show

- These seeds were run at two rates, 0.001 and 0.003, so "best rate" here means only the better of those two. The first seed's runs at 0.0003 and 0.01 were not repeated.
- The product control was not repeated.
- The runs at 0.001 are E41's, made with `exp41_scale.py`. E52's runner reproduced them exactly on seed 93683, but on these two seeds the two runners were not compared. E41's receipts carry no length test and no norms.
- Of the 108 readings in Section 4, 18 have a margin of one or two rows. Seven of these are gains or lags:
  - back-propagation's gain at half width with f = 0.50 on seed 73129 (1 row);
  - the lag on A at present width on seed 73129 (2);
  - the lags on R and on A at present width on seed 116089 (1 and 1);
  - the lag on R at half width with f = 0.25 on seed 93683 (2);
  - back-propagation's gain and the lag on R at present width on seed 93683 (1 and 1).

  Every margin is in Table 2 of `exp45/e52_seeds/E52_SEEDS_TABLES.md` (Section 4 here) and in Section 5.
- A rate that is faster to 80 is not shown to end higher. On seed 73129, the sweep at half width with f = 0.50 ends at R 85.05 at 0.003 against 93.82 at 0.001. Back-propagation at present width with f = 0.25 ends at R 81.57 at 0.003 against 83.68 at 0.001.
- The brief's predictions were made for seed 93683 and are scored in `E52_REPORT.md`. They are not scored again here.
- Nothing is said about lives. The coordinator's decision 20 leaves the long lives at the rate 0.003 not started.

## 8. Committed files

The new files for decision 19 are all under `MOD97_LOCAL_RULE_20261106/`, and no existing file was modified.
- **Sources:** `exp52_drive3.py` and `exp52_seeds_tables.py`.
- **The report:** `E52_SEEDS_REPORT.md`.
- **Receipts and final weights:** `exp45/e52_seeds/e52_<S|B>_h<256|512>_f<25|50>_lr0.003_<73129|116089>.json`, with the final weights in the `.pt` file of the same name (12 runs, 24 files).
- **Tables:** `exp45/e52_seeds/E52_SEEDS_TABLES.md` and `exp45/e52_seeds/E52_SEEDS_SUMMARY.json`.
- **Logs** (`exp45/e52_seeds/logs/`): one log per run, the launch log `LAUNCHES.txt` and the driver's output `driver3.out`.

The checkpoints (`*.ckpt`) are not committed, and are deleted when a run completes.
