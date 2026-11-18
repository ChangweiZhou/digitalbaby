# E52, the other two seeds: tables and readings at rates 0.001 and 0.003

Written by `exp52_seeds_tables.py` from the receipts named in Table 1. Rate 0.001: E41's receipts for seeds 73129 and 116089 (`exp45/e41/`), E52's for seed 93683 (`exp45/e52/`, equal to E41's by check 1b). Rate 0.003: `exp45/e52_seeds/` for seeds 73129 and 116089, `exp45/e52/` for seed 93683. Seed 93683's readings here use only the rates 0.001 and 0.003; its full readings, with 0.0003 and 0.01, are in `exp45/e52/E52_TABLES.md`.

## Table 1. Every run

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

## Table 2. Readings, with the margin of each in rows

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

## Table 3. Three seeds side by side (readings only)

| Width, f | Reading | 73129 | 116089 | 93683 |
|---|---|---|---|---|
| half, 0.50 | S gains | no | yes | yes |
| half, 0.50 | B gains | yes | yes | yes |
| half, 0.50 | lag R | the same within the scatter of one seed | the same within the scatter of one seed | the same within the scatter of one seed |
| half, 0.50 | lag A | the same within the scatter of one seed | the same within the scatter of one seed | the same within the scatter of one seed |
| half, 0.25 | S gains | yes | yes | yes |
| half, 0.25 | B gains | yes | no | no |
| half, 0.25 | lag R | the same within the scatter of one seed | smaller | smaller |
| half, 0.25 | lag A | the same within the scatter of one seed | smaller | the same within the scatter of one seed |
| present, 0.25 | S gains | yes | yes | yes |
| present, 0.25 | B gains | no | no | no |
| present, 0.25 | lag R | gone | gone | smaller |
| present, 0.25 | lag A | gone | gone | smaller |

