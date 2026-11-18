# E52, the other two seeds: tables and readings at rates 0.001 and 0.003

Written by `exp52_seeds_tables.py` from the receipts named in Table 1. Rate 0.001: E41's receipts for seeds 73129 and 116089 (`exp45/e41/`), E52's for seed 93683 (`exp45/e52/`, equal to E41's by check 1b). Rate 0.003: `exp45/e52_seeds/` for seeds 73129 and 116089, `exp45/e52/` for seed 93683. Seed 93683's readings here use only the rates 0.001 and 0.003; its full readings, with 0.0003 and 0.01, are in `exp45/e52/E52_TABLES.md`.

## Table 1. Every run

| Seed | Width, f | Rule | Rate | Dose | Updates to 80, A | Updates to 80, R | Final A | Final R | Final taught | Length 40: last, mean of 3 (range of 12) | Norms at the end (book / l1 / input / recurrent / lm / l2 / l3) | Resumed at | Receipt |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 73129 | half, 0.50 | S | 0.001 | 40,960 | 12,288 | 15,872 | 96.17 | 93.82 | 98.11 | not scored (E41) | not recorded (E41) | – | `exp45/e41/e41_S_h256_f50_73129.json` |
| 73129 | half, 0.50 | B | 0.001 | 40,960 | 11,520 | 13,056 | 94.11 | 94.50 | 95.17 | not scored (E41) | not recorded (E41) | – | `exp45/e41/e41_B_h256_f50_73129.json` |
| 73129 | half, 0.25 | S | 0.001 | 40,960 | 17,408 | 35,840 | 87.94 | 82.83 | 97.00 | not scored (E41) | not recorded (E41) | – | `exp45/e41/e41_S_h256_f25_73129.json` |
| 73129 | half, 0.25 | B | 0.001 | 40,960 | 10,240 | 16,896 | 93.44 | 89.80 | 97.17 | not scored (E41) | not recorded (E41) | – | `exp45/e41/e41_B_h256_f25_73129.json` |
| 73129 | present, 0.25 | S | 0.001 | 12,288 | 8,704 | – | 85.11 | 80.35 | 97.50 | not scored (E41) | not recorded (E41) | – | `exp45/e41/e41_S_h512_f25_73129.json` |
| 73129 | present, 0.25 | B | 0.001 | 12,288 | 4,352 | 7,680 | 85.61 | 83.68 | 95.61 | not scored (E41) | not recorded (E41) | – | `exp45/e41/e41_B_h512_f25_73129.json` |
| 116089 | half, 0.50 | S | 0.001 | 40,960 | 10,496 | 20,480 | 90.78 | 88.00 | 94.17 | not scored (E41) | not recorded (E41) | – | `exp45/e41/e41_S_h256_f50_116089.json` |
| 116089 | half, 0.50 | B | 0.001 | 40,960 | 7,680 | 8,960 | 97.61 | 94.73 | 98.72 | not scored (E41) | not recorded (E41) | – | `exp45/e41/e41_B_h256_f50_116089.json` |
| 116089 | half, 0.25 | S | 0.001 | 40,960 | 24,832 | 30,464 | 84.28 | 82.43 | 94.94 | not scored (E41) | not recorded (E41) | – | `exp45/e41/e41_S_h256_f25_116089.json` |
| 116089 | half, 0.25 | B | 0.001 | 40,960 | 7,680 | 9,216 | 94.78 | 91.75 | 98.33 | not scored (E41) | not recorded (E41) | – | `exp45/e41/e41_B_h256_f25_116089.json` |
| 116089 | present, 0.25 | S | 0.001 | 12,288 | – | – | 78.56 | 73.92 | 97.78 | not scored (E41) | not recorded (E41) | – | `exp45/e41/e41_S_h512_f25_116089.json` |
| 116089 | present, 0.25 | B | 0.001 | 12,288 | 4,864 | 5,888 | 88.17 | 85.95 | 96.56 | not scored (E41) | not recorded (E41) | – | `exp45/e41/e41_B_h512_f25_116089.json` |
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

## Table 2. Readings, with the margin of each in rows

Each cell: the reading, then in brackets the fewest rows by which one count would have to move to change it, and the reading it would become.

| Seed | Width, f | Best rate on R: S; B | Best rate on A: S; B | 0.003 against 0.001, S: R; A | 0.003 against 0.001, B: R; A | S gains | B gains | Lag on R: at 0.001 → own rates | Lag on A: at 0.001 → own rates |
|---|---|---|---|---|---|---|---|---|---|
| 93683 | half, 0.50 | 0.003 (22 → 0.001); 0.003 (14 → 0.001) | 0.003 (12 → 0.001); 0.003 (9 → 0.001) | faster (6 → the same); the same (1 → faster) | faster (4 → the same); faster (1 → the same) | yes (6 → no) | yes (4 → no) | 1.53 → 1.53: the same within the scatter of one seed (7 → larger) | 1.38 → 1.42: the same within the scatter of one seed (6 → larger) |
| 93683 | half, 0.25 | 0.003 (50 → 0.001); 0.003 (6 → 0.001) | 0.003 (25 → 0.001); 0.003 (8 → 0.001) | faster (19 → the same); faster (8 → the same) | the same (3 → faster); faster (1 → the same) | yes (19 → no) | no (3 → yes) | 3.31 → 2.05: smaller (2 → the same within the scatter of one seed) | 2.29 → 1.93: the same within the scatter of one seed (4 → smaller) |
| 93683 | present, 0.25 | 0.003 (25 → 0.001); 0.003 (4 → 0.001) | 0.003 (22 → 0.001); 0.003 (1 → 0.001) | faster (10 → not decided); faster (10 → the same) | the same (1 → faster); the same (3 → faster) | yes (10 → no) | no (1 → yes) | above 2.70 → 1.75: smaller (1 → not decided) | 2.79 → 1.33: smaller (4 → gone) |

## Table 3. Three seeds side by side (readings only)

| Width, f | Reading | 73129 | 116089 | 93683 |
|---|---|---|---|---|
| half, 0.50 | S gains | – | – | yes |
| half, 0.50 | B gains | – | – | yes |
| half, 0.50 | lag R | – | – | the same within the scatter of one seed |
| half, 0.50 | lag A | – | – | the same within the scatter of one seed |
| half, 0.25 | S gains | – | – | yes |
| half, 0.25 | B gains | – | – | no |
| half, 0.25 | lag R | – | – | smaller |
| half, 0.25 | lag A | – | – | the same within the scatter of one seed |
| present, 0.25 | S gains | – | – | yes |
| present, 0.25 | B gains | – | – | no |
| present, 0.25 | lag R | – | – | smaller |
| present, 0.25 | lag A | – | – | smaller |

Receipts missing or incomplete: `exp45/e52_seeds/e52_S_h256_f50_lr0.003_73129.json`, `exp45/e52_seeds/e52_B_h256_f50_lr0.003_73129.json`, `exp45/e52_seeds/e52_S_h256_f25_lr0.003_73129.json`, `exp45/e52_seeds/e52_B_h256_f25_lr0.003_73129.json`, `exp45/e52_seeds/e52_S_h512_f25_lr0.003_73129.json`, `exp45/e52_seeds/e52_B_h512_f25_lr0.003_73129.json`, `exp45/e52_seeds/e52_S_h256_f50_lr0.003_116089.json`, `exp45/e52_seeds/e52_B_h256_f50_lr0.003_116089.json`, `exp45/e52_seeds/e52_S_h256_f25_lr0.003_116089.json`, `exp45/e52_seeds/e52_B_h256_f25_lr0.003_116089.json`, `exp45/e52_seeds/e52_S_h512_f25_lr0.003_116089.json`, `exp45/e52_seeds/e52_B_h512_f25_lr0.003_116089.json`

