Receipts: seed 73129 from `claude/wonderful-edison-8hx6i6` (exp45/e41), seed 93683 from `claude/e41-seed-93683` @ d4d95951, seed 116089 from `claude/e41-seed-116089` @ 79ea3575; runs: 54.

### Batch digest at row 4,096 (check 2)

| Seed | f | digests equal across the six runs | digest (first 16) |
|---|---|---|---|
| 73129 | 0.10 | yes (6 runs) | e120924e5af546f9 |
| 73129 | 0.25 | yes (6 runs) | a71b1ca10b24a36b |
| 73129 | 0.50 | yes (6 runs) | bbd71ab7ca27b72e |
| 93683 | 0.10 | yes (6 runs) | d16f47eca4284956 |
| 93683 | 0.25 | yes (6 runs) | f7479d130565b50f |
| 93683 | 0.50 | yes (6 runs) | 65b1c7e06b17459a |
| 116089 | 0.10 | yes (6 runs) | b8b188d618187d41 |
| 116089 | 0.25 | yes (6 runs) | bfc1f5de028612a7 |
| 116089 | 0.50 | yes (6 runs) | 1df203fb8f299dc0 |

### Seed 73129: updates and computation to threshold (three-row; single-row in brackets; * where they differ in description)

**Accuracy A**

| Rule | f | H | A50: updates [single] | A80: updates [single] | A90: updates [single] | A80 weight-items | A80 CPU-s | taught at A80 | max A (row) | mean A rows 7,680-8,192 |
|---|---|---|---|---|---|---|---|---|---|---|
| S | 0.10 | 256 | not reached [not reached] | not reached [not reached] | not reached [not reached] | not reached |  |  | 13.50 (38,656) | 6.33 |
| S | 0.10 | 512 | not reached [not reached] | not reached [not reached] | not reached [not reached] | not reached |  |  | 10.17 (10,240) | 7.83 |
| S | 0.10 | 1024 | not reached [not reached] | not reached [not reached] | not reached [not reached] | not reached |  |  | 13.17 (7,168) | 12.00 |
| S | 0.25 | 256 | 6,656 [6,656] | 17,408 [12,800] | not reached [34,304] | 3.54e+11 | 741 | 94.2 | 91.33 (38,656) | 63.33 |
| S | 0.25 | 512 | 3,584 [3,584] | 8,704 [7,168] | not reached [not reached] | 6.26e+11 | 973 | 97.3 | 88.83 (12,032) | 79.72 |
| S | 0.25 | 1024 | 2,048 [2,048] | 6,400 [5,632] | not reached [not reached] | 1.73e+12 | 2370 | 98.0 | 87.00 (8,192) | 85.61 |
| S | 0.50 | 256 | 6,144 [6,144] | 12,288 [11,776] | 18,944 [17,152] | 2.5e+11 | 528 | 90.5 | 96.50 (31,232) | 63.89 |
| S | 0.50 | 512 | 3,840 [3,840] | 5,632 [5,632] | not reached [7,936] | 4.05e+11 | 641 | 90.5 | 94.00 (12,288) | 85.72 |
| S | 0.50 | 1024 | 2,304 [2,304] | 3,328 [3,328] | 5,120 [4,608] | 8.97e+11 | 1220 | 86.5 | 98.00 (8,192) | 96.83 |
| B | 0.10 | 256 | not reached [not reached] | not reached [not reached] | not reached [not reached] | not reached |  |  | 48.17 (40,960) | 25.11 |
| B | 0.10 | 512 | not reached [not reached] | not reached [not reached] | not reached [not reached] | not reached |  |  | 18.83 (11,008) | 16.44 |
| B | 0.10 | 1024 | not reached [not reached] | not reached [not reached] | not reached [not reached] | not reached |  |  | 19.17 (6,656) | 17.72 |
| B | 0.25 | 256 | 7,424 [7,424] | 10,240 [10,240] | 27,648 [15,360] | 2.08e+11 | 407 | 91.0 | 95.00 (40,192) | 62.83 |
| B | 0.25 | 512 | 3,072 [3,072] | 4,352 [4,352] | not reached [8,960] | 3.13e+11 | 467 | 92.7 | 91.17 (8,960) | 85.28 |
| B | 0.25 | 1024 | 2,048 [2,048] | 4,608 [3,840] | not reached [not reached] | 1.24e+12 | 1511 | 97.2 | 86.83 (7,680) | 84.11 |
| B | 0.50 | 256 | 9,216 [9,216] | 11,520 [11,520] | 18,432 [17,152] | 2.34e+11 | 475 | 88.8 | 98.67 (37,888) | 14.94 |
| B | 0.50 | 512 | 4,096 [4,096] | 5,120 [5,120] | 10,240 [7,936] | 3.68e+11 | 544 | 92.8 | 98.33 (10,240) | 89.11 |
| B | 0.50 | 1024 | 2,304 [2,304] | 3,328 [3,328] | 5,888 [4,608] | 8.97e+11 | 1122 | 88.7 | 96.50 (8,192) | 95.06 |

**Accuracy R**

| Rule | f | H | R50: updates [single] | R80: updates [single] | R90: updates [single] | R80 weight-items | R80 CPU-s | taught at R80 | max R (row) | mean R rows 7,680-8,192 |
|---|---|---|---|---|---|---|---|---|---|---|
| S | 0.10 | 256 | not reached [not reached] | not reached [not reached] | not reached [not reached] | not reached |  |  | 10.90 (39,168) | 5.60 |
| S | 0.10 | 512 | not reached [not reached] | not reached [not reached] | not reached [not reached] | not reached |  |  | 8.00 (10,752) | 5.60 |
| S | 0.10 | 1024 | not reached [not reached] | not reached [not reached] | not reached [not reached] | not reached |  |  | 11.00 (6,912) | 10.32 |
| S | 0.25 | 256 | 7,680 [7,680] | 35,840 [25,600] | not reached [not reached] | 7.28e+11 | 1568 | 97.3 | 85.60 (40,704) | 51.62 |
| S | 0.25 | 512 | 3,840 [3,840] | not reached [10,752] | not reached [not reached] | not reached |  |  | 83.85 (11,520) | 74.23 |
| S | 0.25 | 1024 | 2,560 [2,560] | not reached [7,424] | not reached [not reached] | not reached |  |  | 84.15 (8,192) | 80.82 |
| S | 0.50 | 256 | 8,704 [6,656] | 15,872 [12,544] | 28,416 [23,808] | 3.22e+11 | 678 | 93.0 | 95.80 (39,168) | 50.63 |
| S | 0.50 | 512 | 3,840 [3,840] | 6,656 [6,656] | not reached [12,288] | 4.79e+11 | 760 | 88.5 | 90.15 (12,288) | 79.13 |
| S | 0.50 | 1024 | 2,304 [2,304] | 3,840 [3,840] | 6,656 [6,656] | 1.04e+12 | 1414 | 90.0 | 97.15 (7,680) | 95.58 |
| B | 0.10 | 256 | not reached [not reached] | not reached [not reached] | not reached [not reached] | not reached |  |  | 45.15 (40,448) | 25.63 |
| B | 0.10 | 512 | not reached [not reached] | not reached [not reached] | not reached [not reached] | not reached |  |  | 16.85 (10,752) | 13.72 |
| B | 0.10 | 1024 | not reached [not reached] | not reached [not reached] | not reached [not reached] | not reached |  |  | 17.50 (6,912) | 15.98 |
| B | 0.25 | 256 | 7,936 [7,936] | 16,896 [13,056] | 35,840 [23,040] | 3.43e+11 | 684 | 95.2 | 93.75 (36,352) | 53.50 |
| B | 0.25 | 512 | 3,072 [3,072] | 7,680 [5,632] | not reached [not reached] | 5.53e+11 | 805 | 96.0 | 87.95 (11,520) | 83.20 |
| B | 0.25 | 1024 | 2,048 [2,048] | 7,168 [4,608] | not reached [not reached] | 1.93e+12 | 2368 | 95.2 | 84.15 (7,680) | 79.97 |
| B | 0.50 | 256 | 9,472 [9,472] | 13,056 [12,288] | 20,224 [16,384] | 2.65e+11 | 536 | 94.0 | 98.70 (37,632) | 14.82 |
| B | 0.50 | 512 | 3,840 [3,840] | 6,400 [5,120] | 7,936 [7,936] | 4.6e+11 | 679 | 88.7 | 95.20 (10,240) | 89.53 |
| B | 0.50 | 1024 | 2,304 [2,304] | 3,328 [3,328] | 6,400 [4,608] | 8.97e+11 | 1122 | 88.7 | 94.65 (7,168) | 93.60 |

### Seed 93683: updates and computation to threshold (three-row; single-row in brackets; * where they differ in description)

**Accuracy A**

| Rule | f | H | A50: updates [single] | A80: updates [single] | A90: updates [single] | A80 weight-items | A80 CPU-s | taught at A80 | max A (row) | mean A rows 7,680-8,192 |
|---|---|---|---|---|---|---|---|---|---|---|
| S | 0.10 | 256 | not reached [not reached] | not reached [not reached] | not reached [not reached] | not reached |  |  | 15.50 (33,792) | 10.22 |
| S | 0.10 | 512 | not reached [not reached] | not reached [not reached] | not reached [not reached] | not reached |  |  | 15.67 (11,264) | 12.33 |
| S | 0.10 | 1024 | not reached [not reached] | not reached [not reached] | not reached [not reached] | not reached |  |  | 22.00 (7,424) | 19.00 |
| S | 0.25 | 256 | 6,912 [6,912] | 14,080 [12,544] | not reached [35,328] | 2.86e+11 | 587 | 95.0 | 92.00 (38,144) | 63.39 |
| S | 0.25 | 512 | 3,584 [3,584] | 9,984 [8,192] | not reached [not reached] | 7.18e+11 | 1169 | 95.0 | 85.17 (12,032) | 79.78 |
| S | 0.25 | 1024 | 2,048 [2,048] | 4,096 [3,328] | 7,680 [7,680] | 1.1e+12 | 1797 | 96.3 | 93.17 (7,936) | 91.72 |
| S | 0.50 | 256 | 7,168 [7,168] | 10,240 [8,704] | 18,176 [15,872] | 2.08e+11 | 446 | 88.0 | 95.67 (34,048) | 68.67 |
| S | 0.50 | 512 | 3,584 [3,584] | 4,864 [4,352] | 6,400 [6,400] | 3.5e+11 | 552 | 87.3 | 96.50 (12,288) | 92.28 |
| S | 0.50 | 1024 | 2,304 [2,304] | 2,816 [2,816] | 5,376 [3,840] | 7.59e+11 | 1255 | 89.3 | 98.00 (8,192) | 95.61 |
| B | 0.10 | 256 | 17,152 [14,848] | not reached [not reached] | not reached [not reached] | not reached |  |  | 68.50 (40,960) | 38.11 |
| B | 0.10 | 512 | not reached [not reached] | not reached [not reached] | not reached [not reached] | not reached |  |  | 35.00 (11,264) | 28.22 |
| B | 0.10 | 1024 | not reached [not reached] | not reached [not reached] | not reached [not reached] | not reached |  |  | 23.83 (7,168) | 22.06 |
| B | 0.25 | 256 | 4,864 [4,864] | 6,144 [6,144] | 11,520 [8,192] | 1.25e+11 | 250 | 88.2 | 99.33 (39,168) | 89.17 |
| B | 0.25 | 512 | 3,072 [3,072] | 3,584 [3,584] | 7,424 [5,376] | 2.58e+11 | 379 | 90.8 | 95.17 (10,240) | 91.00 |
| B | 0.25 | 1024 | 2,048 [2,048] | 3,584 [3,072] | not reached [7,424] | 9.66e+11 | 1457 | 95.3 | 91.50 (7,424) | 88.78 |
| B | 0.50 | 256 | 6,656 [6,656] | 7,424 [7,424] | 9,728 [9,728] | 1.51e+11 | 302 | 79.2 | 98.33 (36,096) | 82.17 |
| B | 0.50 | 512 | 3,584 [3,584] | 4,096 [4,096] | 4,864 [4,864] | 2.95e+11 | 434 | 90.2 | 97.33 (11,008) | 94.17 |
| B | 0.50 | 1024 | 2,048 [2,048] | 2,816 [2,816] | 4,608 [3,840] | 7.59e+11 | 1149 | 84.0 | 96.67 (7,936) | 94.67 |

**Accuracy R**

| Rule | f | H | R50: updates [single] | R80: updates [single] | R90: updates [single] | R80 weight-items | R80 CPU-s | taught at R80 | max R (row) | mean R rows 7,680-8,192 |
|---|---|---|---|---|---|---|---|---|---|---|
| S | 0.10 | 256 | not reached [not reached] | not reached [not reached] | not reached [not reached] | not reached |  |  | 10.60 (40,960) | 6.25 |
| S | 0.10 | 512 | not reached [not reached] | not reached [not reached] | not reached [not reached] | not reached |  |  | 11.45 (9,728) | 9.73 |
| S | 0.10 | 1024 | not reached [not reached] | not reached [not reached] | not reached [not reached] | not reached |  |  | 16.75 (6,912) | 15.25 |
| S | 0.25 | 256 | 7,424 [7,424] | 24,576 [16,640] | not reached [not reached] | 4.99e+11 | 1023 | 97.0 | 87.35 (38,656) | 54.18 |
| S | 0.25 | 512 | 3,840 [3,840] | not reached [11,264] | not reached [not reached] | not reached |  |  | 81.55 (12,032) | 75.48 |
| S | 0.25 | 1024 | 2,048 [2,048] | 4,096 [3,584] | not reached [7,936] | 1.1e+12 | 1797 | 96.3 | 90.95 (7,936) | 88.35 |
| S | 0.50 | 256 | 7,168 [7,168] | 13,312 [10,240] | 30,208 [18,176] | 2.7e+11 | 574 | 93.3 | 94.10 (29,696) | 65.52 |
| S | 0.50 | 512 | 3,840 [3,840] | 4,864 [4,864] | 7,424 [6,400] | 3.5e+11 | 552 | 87.3 | 96.25 (12,288) | 90.53 |
| S | 0.50 | 1024 | 2,304 [2,304] | 3,072 [3,072] | 5,376 [3,840] | 8.28e+11 | 1369 | 86.3 | 98.10 (8,192) | 94.20 |
| B | 0.10 | 256 | 23,552 [19,200] | not reached [not reached] | not reached [not reached] | not reached |  |  | 63.10 (39,168) | 34.67 |
| B | 0.10 | 512 | not reached [not reached] | not reached [not reached] | not reached [not reached] | not reached |  |  | 30.55 (10,752) | 25.82 |
| B | 0.10 | 1024 | not reached [not reached] | not reached [not reached] | not reached [not reached] | not reached |  |  | 21.05 (5,888) | 20.37 |
| B | 0.25 | 256 | 4,864 [4,864] | 7,424 [6,656] | 17,920 [12,288] | 1.51e+11 | 302 | 94.2 | 98.30 (31,744) | 84.32 |
| B | 0.25 | 512 | 3,072 [3,072] | 4,352 [3,584] | 8,704 [4,864] | 3.13e+11 | 460 | 97.3 | 93.90 (10,496) | 90.32 |
| B | 0.25 | 1024 | 2,048 [2,048] | 3,072 [3,072] | not reached [8,192] | 8.28e+11 | 1248 | 93.5 | 92.35 (8,192) | 89.60 |
| B | 0.50 | 256 | 6,912 [6,912] | 8,704 [8,704] | 15,360 [10,240] | 1.77e+11 | 354 | 88.2 | 97.15 (30,208) | 74.70 |
| B | 0.50 | 512 | 3,584 [3,584] | 4,096 [4,096] | 7,424 [5,120] | 2.95e+11 | 434 | 90.2 | 96.60 (11,008) | 93.92 |
| B | 0.50 | 1024 | 2,048 [2,048] | 2,816 [2,816] | 4,608 [4,608] | 7.59e+11 | 1149 | 84.0 | 95.00 (7,936) | 94.18 |

### Seed 116089: updates and computation to threshold (three-row; single-row in brackets; * where they differ in description)

**Accuracy A**

| Rule | f | H | A50: updates [single] | A80: updates [single] | A90: updates [single] | A80 weight-items | A80 CPU-s | taught at A80 | max A (row) | mean A rows 7,680-8,192 |
|---|---|---|---|---|---|---|---|---|---|---|
| S | 0.10 | 256 | not reached [not reached] | not reached [not reached] | not reached [not reached] | not reached |  |  | 18.00 (32,256) | 10.78 |
| S | 0.10 | 512 | not reached [not reached] | not reached [not reached] | not reached [not reached] | not reached |  |  | 11.00 (11,776) | 9.22 |
| S | 0.10 | 1024 | not reached [not reached] | not reached [not reached] | not reached [not reached] | not reached |  |  | 19.50 (8,192) | 18.11 |
| S | 0.25 | 256 | 9,216 [8,448] | 24,832 [23,552] | not reached [not reached] | 5.05e+11 | 1115 | 95.2 | 89.00 (38,144) | 45.61 |
| S | 0.25 | 512 | 4,096 [4,096] | not reached [12,288] | not reached [not reached] | not reached |  |  | 80.17 (12,288) | 68.39 |
| S | 0.25 | 1024 | 2,048 [2,048] | 6,912 [6,912] | not reached [not reached] | 1.86e+12 | 2806 | 98.3 | 85.00 (8,192) | 82.22 |
| S | 0.50 | 256 | 6,400 [6,400] | 10,496 [8,448] | 24,832 [19,200] | 2.13e+11 | 480 | 90.8 | 93.67 (34,304) | 70.94 |
| S | 0.50 | 512 | 3,072 [3,072] | 5,376 [4,608] | 9,472 [5,376] | 3.87e+11 | 674 | 94.7 | 95.17 (11,776) | 90.39 |
| S | 0.50 | 1024 | 2,048 [2,048] | 2,816 [2,816] | 5,120 [4,352] | 7.59e+11 | 1129 | 88.3 | 94.83 (6,656) | 93.78 |
| B | 0.10 | 256 | 28,672 [19,712] | not reached [not reached] | not reached [not reached] | not reached |  |  | 59.50 (39,936) | 36.67 |
| B | 0.10 | 512 | not reached [not reached] | not reached [not reached] | not reached [not reached] | not reached |  |  | 27.67 (12,288) | 22.22 |
| B | 0.10 | 1024 | not reached [not reached] | not reached [not reached] | not reached [not reached] | not reached |  |  | 25.17 (5,376) | 22.61 |
| B | 0.25 | 256 | 5,632 [5,632] | 7,680 [7,680] | 14,592 [13,056] | 1.56e+11 | 322 | 93.5 | 96.17 (40,704) | 83.33 |
| B | 0.25 | 512 | 2,816 [2,816] | 4,864 [4,864] | not reached [10,496] | 3.5e+11 | 564 | 95.3 | 90.67 (10,496) | 86.44 |
| B | 0.25 | 1024 | 1,792 [1,792] | 4,608 [4,608] | not reached [7,936] | 1.24e+12 | 1680 | 95.8 | 90.33 (7,936) | 89.28 |
| B | 0.50 | 256 | 4,864 [4,864] | 7,680 [7,680] | 14,848 [11,776] | 1.56e+11 | 321 | 89.2 | 98.83 (40,704) | 85.17 |
| B | 0.50 | 512 | 2,304 [2,304] | 4,352 [3,584] | 9,216 [6,912] | 3.13e+11 | 511 | 90.2 | 95.33 (9,472) | 88.83 |
| B | 0.50 | 1024 | 2,048 [2,048] | 3,072 [2,560] | 6,656 [4,864] | 8.28e+11 | 1128 | 90.3 | 93.67 (7,168) | 92.39 |

**Accuracy R**

| Rule | f | H | R50: updates [single] | R80: updates [single] | R90: updates [single] | R80 weight-items | R80 CPU-s | taught at R80 | max R (row) | mean R rows 7,680-8,192 |
|---|---|---|---|---|---|---|---|---|---|---|
| S | 0.10 | 256 | not reached [not reached] | not reached [not reached] | not reached [not reached] | not reached |  |  | 12.25 (37,120) | 8.35 |
| S | 0.10 | 512 | not reached [not reached] | not reached [not reached] | not reached [not reached] | not reached |  |  | 9.30 (11,776) | 6.98 |
| S | 0.10 | 1024 | not reached [not reached] | not reached [not reached] | not reached [not reached] | not reached |  |  | 15.20 (8,192) | 14.35 |
| S | 0.25 | 256 | 10,240 [9,472] | 30,464 [28,160] | not reached [not reached] | 6.19e+11 | 1368 | 96.5 | 86.25 (38,144) | 41.53 |
| S | 0.25 | 512 | 4,096 [4,096] | not reached [not reached] | not reached [not reached] | not reached |  |  | 75.15 (12,032) | 63.93 |
| S | 0.25 | 1024 | 2,560 [2,560] | not reached [7,168] | not reached [not reached] | not reached |  |  | 81.10 (8,192) | 78.63 |
| S | 0.50 | 256 | 7,424 [7,424] | 20,480 [13,568] | not reached [34,304] | 4.16e+11 | 933 | 92.5 | 90.90 (40,960) | 59.70 |
| S | 0.50 | 512 | 3,072 [3,072] | 6,400 [4,608] | 9,472 [8,192] | 4.6e+11 | 800 | 89.2 | 94.85 (10,752) | 87.83 |
| S | 0.50 | 1024 | 2,048 [2,048] | 3,840 [3,072] | 6,400 [5,376] | 1.04e+12 | 1539 | 94.0 | 95.35 (7,936) | 93.22 |
| B | 0.10 | 256 | 31,232 [25,600] | not reached [not reached] | not reached [not reached] | not reached |  |  | 56.10 (39,936) | 33.67 |
| B | 0.10 | 512 | not reached [not reached] | not reached [not reached] | not reached [not reached] | not reached |  |  | 23.15 (10,240) | 20.62 |
| B | 0.10 | 1024 | not reached [not reached] | not reached [not reached] | not reached [not reached] | not reached |  |  | 21.45 (7,168) | 19.37 |
| B | 0.25 | 256 | 6,400 [5,632] | 9,216 [7,936] | 23,296 [17,664] | 1.87e+11 | 385 | 91.7 | 94.45 (39,936) | 79.45 |
| B | 0.25 | 512 | 2,816 [2,816] | 5,888 [4,864] | not reached [10,752] | 4.24e+11 | 680 | 93.2 | 90.90 (10,752) | 86.72 |
| B | 0.25 | 1024 | 1,792 [1,792] | 4,608 [4,096] | not reached [7,936] | 1.24e+12 | 1680 | 95.8 | 90.85 (7,936) | 88.38 |
| B | 0.50 | 256 | 5,120 [5,120] | 8,960 [7,936] | 22,272 [15,616] | 1.82e+11 | 376 | 91.0 | 97.75 (40,704) | 79.13 |
| B | 0.50 | 512 | 2,816 [2,816] | 5,120 [3,584] | 9,216 [6,912] | 3.68e+11 | 600 | 90.5 | 96.25 (12,288) | 89.00 |
| B | 0.50 | 1024 | 2,048 [2,048] | 3,840 [3,072] | 6,656 [5,888] | 1.04e+12 | 1410 | 89.2 | 94.40 (7,424) | 91.78 |

### Descriptions by width step, threshold 80 (three-row version; per-seed descriptions with the seeds' values; * where the single-row version describes the cell differently)

| Threshold | Measure | Unit | Rule | f | Step | Cell | Per seed (description: narrower, wider) |
|---|---|---|---|---|---|---|---|
| 80 | R | updates | S | 0.10 | 256→512 | neither reaches | 73129: neither reaches (nr, nr); 93683: neither reaches (nr, nr); 116089: neither reaches (nr, nr) |
| 80 | R | updates | S | 0.10 | 512→1024 | neither reaches | 73129: neither reaches (nr, nr); 93683: neither reaches (nr, nr); 116089: neither reaches (nr, nr) |
| 80 | R | updates | S | 0.25 | 256→512 | not decided* | 73129: not decided (35,840, nr); 93683: not decided (24,576, nr); 116089: not decided (30,464, nr) |
| 80 | R | updates | S | 0.25 | 512→1024 | neither reaches* | 73129: neither reaches (nr, nr); 93683: falls (nr, 4,096); 116089: neither reaches (nr, nr) |
| 80 | R | updates | S | 0.50 | 256→512 | falls | 73129: falls (15,872, 6,656); 93683: falls (13,312, 4,864); 116089: falls (20,480, 6,400) |
| 80 | R | updates | S | 0.50 | 512→1024 | falls* | 73129: falls (6,656, 3,840); 93683: falls (4,864, 3,072); 116089: falls (6,400, 3,840) |
| 80 | R | updates | B | 0.10 | 256→512 | neither reaches | 73129: neither reaches (nr, nr); 93683: neither reaches (nr, nr); 116089: neither reaches (nr, nr) |
| 80 | R | updates | B | 0.10 | 512→1024 | neither reaches | 73129: neither reaches (nr, nr); 93683: neither reaches (nr, nr); 116089: neither reaches (nr, nr) |
| 80 | R | updates | B | 0.25 | 256→512 | falls | 73129: falls (16,896, 7,680); 93683: falls (7,424, 4,352); 116089: falls (9,216, 5,888) |
| 80 | R | updates | B | 0.25 | 512→1024 | stays | 73129: stays (7,680, 7,168); 93683: stays (4,352, 3,072); 116089: stays (5,888, 4,608) |
| 80 | R | updates | B | 0.50 | 256→512 | falls | 73129: falls (13,056, 6,400); 93683: falls (8,704, 4,096); 116089: falls (8,960, 5,120) |
| 80 | R | updates | B | 0.50 | 512→1024 | stays | 73129: falls (6,400, 3,328); 93683: stays (4,096, 2,816); 116089: stays (5,120, 3,840) |
| 80 | R | weight-items | S | 0.10 | 256→512 | neither reaches | 73129: neither reaches (nr, nr); 93683: neither reaches (nr, nr); 116089: neither reaches (nr, nr) |
| 80 | R | weight-items | S | 0.10 | 512→1024 | neither reaches | 73129: neither reaches (nr, nr); 93683: neither reaches (nr, nr); 116089: neither reaches (nr, nr) |
| 80 | R | weight-items | S | 0.25 | 256→512 | not decided* | 73129: not decided (7.28e+11, nr); 93683: rises (4.99e+11, nr); 116089: not decided (6.19e+11, nr) |
| 80 | R | weight-items | S | 0.25 | 512→1024 | neither reaches* | 73129: neither reaches (nr, nr); 93683: not decided (nr, 1.1e+12); 116089: neither reaches (nr, nr) |
| 80 | R | weight-items | S | 0.50 | 256→512 | stays* | 73129: stays (3.22e+11, 4.79e+11); 93683: stays (2.7e+11, 3.5e+11); 116089: stays (4.16e+11, 4.6e+11) |
| 80 | R | weight-items | S | 0.50 | 512→1024 | rises | 73129: rises (4.79e+11, 1.04e+12); 93683: rises (3.5e+11, 8.28e+11); 116089: rises (4.6e+11, 1.04e+12) |
| 80 | R | weight-items | B | 0.10 | 256→512 | neither reaches | 73129: neither reaches (nr, nr); 93683: neither reaches (nr, nr); 116089: neither reaches (nr, nr) |
| 80 | R | weight-items | B | 0.10 | 512→1024 | neither reaches | 73129: neither reaches (nr, nr); 93683: neither reaches (nr, nr); 116089: neither reaches (nr, nr) |
| 80 | R | weight-items | B | 0.25 | 256→512 | rises | 73129: rises (3.43e+11, 5.53e+11); 93683: rises (1.51e+11, 3.13e+11); 116089: rises (1.87e+11, 4.24e+11) |
| 80 | R | weight-items | B | 0.25 | 512→1024 | rises | 73129: rises (5.53e+11, 1.93e+12); 93683: rises (3.13e+11, 8.28e+11); 116089: rises (4.24e+11, 1.24e+12) |
| 80 | R | weight-items | B | 0.50 | 256→512 | rises* | 73129: rises (2.65e+11, 4.6e+11); 93683: rises (1.77e+11, 2.95e+11); 116089: rises (1.82e+11, 3.68e+11) |
| 80 | R | weight-items | B | 0.50 | 512→1024 | rises | 73129: rises (4.6e+11, 8.97e+11); 93683: rises (2.95e+11, 7.59e+11); 116089: rises (3.68e+11, 1.04e+12) |
| 80 | A | updates | S | 0.10 | 256→512 | neither reaches | 73129: neither reaches (nr, nr); 93683: neither reaches (nr, nr); 116089: neither reaches (nr, nr) |
| 80 | A | updates | S | 0.10 | 512→1024 | neither reaches | 73129: neither reaches (nr, nr); 93683: neither reaches (nr, nr); 116089: neither reaches (nr, nr) |
| 80 | A | updates | S | 0.25 | 256→512 | mixed* | 73129: falls (17,408, 8,704); 93683: stays (14,080, 9,984); 116089: not decided (24,832, nr) |
| 80 | A | updates | S | 0.25 | 512→1024 | falls | 73129: stays (8,704, 6,400); 93683: falls (9,984, 4,096); 116089: falls (nr, 6,912) |
| 80 | A | updates | S | 0.50 | 256→512 | falls | 73129: falls (12,288, 5,632); 93683: falls (10,240, 4,864); 116089: falls (10,496, 5,376) |
| 80 | A | updates | S | 0.50 | 512→1024 | falls | 73129: falls (5,632, 3,328); 93683: falls (4,864, 2,816); 116089: falls (5,376, 2,816) |
| 80 | A | updates | B | 0.10 | 256→512 | neither reaches | 73129: neither reaches (nr, nr); 93683: neither reaches (nr, nr); 116089: neither reaches (nr, nr) |
| 80 | A | updates | B | 0.10 | 512→1024 | neither reaches | 73129: neither reaches (nr, nr); 93683: neither reaches (nr, nr); 116089: neither reaches (nr, nr) |
| 80 | A | updates | B | 0.25 | 256→512 | falls | 73129: falls (10,240, 4,352); 93683: falls (6,144, 3,584); 116089: falls (7,680, 4,864) |
| 80 | A | updates | B | 0.25 | 512→1024 | stays | 73129: stays (4,352, 4,608); 93683: stays (3,584, 3,584); 116089: stays (4,864, 4,608) |
| 80 | A | updates | B | 0.50 | 256→512 | falls | 73129: falls (11,520, 5,120); 93683: falls (7,424, 4,096); 116089: falls (7,680, 4,352) |
| 80 | A | updates | B | 0.50 | 512→1024 | stays | 73129: falls (5,120, 3,328); 93683: stays (4,096, 2,816); 116089: stays (4,352, 3,072) |
| 80 | A | weight-items | S | 0.10 | 256→512 | neither reaches | 73129: neither reaches (nr, nr); 93683: neither reaches (nr, nr); 116089: neither reaches (nr, nr) |
| 80 | A | weight-items | S | 0.10 | 512→1024 | neither reaches | 73129: neither reaches (nr, nr); 93683: neither reaches (nr, nr); 116089: neither reaches (nr, nr) |
| 80 | A | weight-items | S | 0.25 | 256→512 | rises | 73129: rises (3.54e+11, 6.26e+11); 93683: rises (2.86e+11, 7.18e+11); 116089: rises (5.05e+11, nr) |
| 80 | A | weight-items | S | 0.25 | 512→1024 | rises* | 73129: rises (6.26e+11, 1.73e+12); 93683: rises (7.18e+11, 1.1e+12); 116089: not decided (nr, 1.86e+12) |
| 80 | A | weight-items | S | 0.50 | 256→512 | rises | 73129: rises (2.5e+11, 4.05e+11); 93683: rises (2.08e+11, 3.5e+11); 116089: rises (2.13e+11, 3.87e+11) |
| 80 | A | weight-items | S | 0.50 | 512→1024 | rises | 73129: rises (4.05e+11, 8.97e+11); 93683: rises (3.5e+11, 7.59e+11); 116089: rises (3.87e+11, 7.59e+11) |
| 80 | A | weight-items | B | 0.10 | 256→512 | neither reaches | 73129: neither reaches (nr, nr); 93683: neither reaches (nr, nr); 116089: neither reaches (nr, nr) |
| 80 | A | weight-items | B | 0.10 | 512→1024 | neither reaches | 73129: neither reaches (nr, nr); 93683: neither reaches (nr, nr); 116089: neither reaches (nr, nr) |
| 80 | A | weight-items | B | 0.25 | 256→512 | rises | 73129: rises (2.08e+11, 3.13e+11); 93683: rises (1.25e+11, 2.58e+11); 116089: rises (1.56e+11, 3.5e+11) |
| 80 | A | weight-items | B | 0.25 | 512→1024 | rises | 73129: rises (3.13e+11, 1.24e+12); 93683: rises (2.58e+11, 9.66e+11); 116089: rises (3.5e+11, 1.24e+12) |
| 80 | A | weight-items | B | 0.50 | 256→512 | rises | 73129: rises (2.34e+11, 3.68e+11); 93683: rises (1.51e+11, 2.95e+11); 116089: rises (1.56e+11, 3.13e+11) |
| 80 | A | weight-items | B | 0.50 | 512→1024 | rises | 73129: rises (3.68e+11, 8.97e+11); 93683: rises (2.95e+11, 7.59e+11); 116089: rises (3.13e+11, 8.28e+11) |

### Descriptions by width step, thresholds 50, 90 (three-row version; per-seed descriptions with the seeds' values; * where the single-row version describes the cell differently)

| Threshold | Measure | Unit | Rule | f | Step | Cell | Per seed (description: narrower, wider) |
|---|---|---|---|---|---|---|---|
| 50 | R | updates | S | 0.10 | 256→512 | neither reaches | 73129: neither reaches (nr, nr); 93683: neither reaches (nr, nr); 116089: neither reaches (nr, nr) |
| 50 | R | updates | S | 0.10 | 512→1024 | neither reaches | 73129: neither reaches (nr, nr); 93683: neither reaches (nr, nr); 116089: neither reaches (nr, nr) |
| 50 | R | updates | S | 0.25 | 256→512 | falls | 73129: falls (7,680, 3,840); 93683: falls (7,424, 3,840); 116089: falls (10,240, 4,096) |
| 50 | R | updates | S | 0.25 | 512→1024 | falls | 73129: stays (3,840, 2,560); 93683: falls (3,840, 2,048); 116089: falls (4,096, 2,560) |
| 50 | R | updates | S | 0.50 | 256→512 | falls | 73129: falls (8,704, 3,840); 93683: falls (7,168, 3,840); 116089: falls (7,424, 3,072) |
| 50 | R | updates | S | 0.50 | 512→1024 | falls | 73129: falls (3,840, 2,304); 93683: falls (3,840, 2,304); 116089: stays (3,072, 2,048) |
| 50 | R | updates | B | 0.10 | 256→512 | not decided | 73129: neither reaches (nr, nr); 93683: not decided (23,552, nr); 116089: not decided (31,232, nr) |
| 50 | R | updates | B | 0.10 | 512→1024 | neither reaches | 73129: neither reaches (nr, nr); 93683: neither reaches (nr, nr); 116089: neither reaches (nr, nr) |
| 50 | R | updates | B | 0.25 | 256→512 | falls | 73129: falls (7,936, 3,072); 93683: falls (4,864, 3,072); 116089: falls (6,400, 2,816) |
| 50 | R | updates | B | 0.25 | 512→1024 | stays | 73129: stays (3,072, 2,048); 93683: stays (3,072, 2,048); 116089: falls (2,816, 1,792) |
| 50 | R | updates | B | 0.50 | 256→512 | falls | 73129: falls (9,472, 3,840); 93683: falls (6,912, 3,584); 116089: falls (5,120, 2,816) |
| 50 | R | updates | B | 0.50 | 512→1024 | falls | 73129: falls (3,840, 2,304); 93683: falls (3,584, 2,048); 116089: stays (2,816, 2,048) |
| 50 | R | weight-items | S | 0.10 | 256→512 | neither reaches | 73129: neither reaches (nr, nr); 93683: neither reaches (nr, nr); 116089: neither reaches (nr, nr) |
| 50 | R | weight-items | S | 0.10 | 512→1024 | neither reaches | 73129: neither reaches (nr, nr); 93683: neither reaches (nr, nr); 116089: neither reaches (nr, nr) |
| 50 | R | weight-items | S | 0.25 | 256→512 | rises* | 73129: rises (1.56e+11, 2.76e+11); 93683: rises (1.51e+11, 2.76e+11); 116089: stays (2.08e+11, 2.95e+11) |
| 50 | R | weight-items | S | 0.25 | 512→1024 | rises | 73129: rises (2.76e+11, 6.9e+11); 93683: rises (2.76e+11, 5.52e+11); 116089: rises (2.95e+11, 6.9e+11) |
| 50 | R | weight-items | S | 0.50 | 256→512 | rises | 73129: rises (1.77e+11, 2.76e+11); 93683: rises (1.46e+11, 2.76e+11); 116089: stays (1.51e+11, 2.21e+11) |
| 50 | R | weight-items | S | 0.50 | 512→1024 | rises | 73129: rises (2.76e+11, 6.21e+11); 93683: rises (2.76e+11, 6.21e+11); 116089: rises (2.21e+11, 5.52e+11) |
| 50 | R | weight-items | B | 0.10 | 256→512 | mixed* | 73129: neither reaches (nr, nr); 93683: rises (4.79e+11, nr); 116089: not decided (6.35e+11, nr) |
| 50 | R | weight-items | B | 0.10 | 512→1024 | neither reaches | 73129: neither reaches (nr, nr); 93683: neither reaches (nr, nr); 116089: neither reaches (nr, nr) |
| 50 | R | weight-items | B | 0.25 | 256→512 | rises | 73129: stays (1.61e+11, 2.21e+11); 93683: rises (9.88e+10, 2.21e+11); 116089: rises (1.3e+11, 2.03e+11) |
| 50 | R | weight-items | B | 0.25 | 512→1024 | rises | 73129: rises (2.21e+11, 5.52e+11); 93683: rises (2.21e+11, 5.52e+11); 116089: rises (2.03e+11, 4.83e+11) |
| 50 | R | weight-items | B | 0.50 | 256→512 | rises | 73129: stays (1.92e+11, 2.76e+11); 93683: rises (1.4e+11, 2.58e+11); 116089: rises (1.04e+11, 2.03e+11) |
| 50 | R | weight-items | B | 0.50 | 512→1024 | rises | 73129: rises (2.76e+11, 6.21e+11); 93683: rises (2.58e+11, 5.52e+11); 116089: rises (2.03e+11, 5.52e+11) |
| 50 | A | updates | S | 0.10 | 256→512 | neither reaches | 73129: neither reaches (nr, nr); 93683: neither reaches (nr, nr); 116089: neither reaches (nr, nr) |
| 50 | A | updates | S | 0.10 | 512→1024 | neither reaches | 73129: neither reaches (nr, nr); 93683: neither reaches (nr, nr); 116089: neither reaches (nr, nr) |
| 50 | A | updates | S | 0.25 | 256→512 | falls | 73129: falls (6,656, 3,584); 93683: falls (6,912, 3,584); 116089: falls (9,216, 4,096) |
| 50 | A | updates | S | 0.25 | 512→1024 | falls | 73129: falls (3,584, 2,048); 93683: falls (3,584, 2,048); 116089: falls (4,096, 2,048) |
| 50 | A | updates | S | 0.50 | 256→512 | falls | 73129: falls (6,144, 3,840); 93683: falls (7,168, 3,584); 116089: falls (6,400, 3,072) |
| 50 | A | updates | S | 0.50 | 512→1024 | falls | 73129: falls (3,840, 2,304); 93683: falls (3,584, 2,304); 116089: stays (3,072, 2,048) |
| 50 | A | updates | B | 0.10 | 256→512 | not decided | 73129: neither reaches (nr, nr); 93683: not decided (17,152, nr); 116089: not decided (28,672, nr) |
| 50 | A | updates | B | 0.10 | 512→1024 | neither reaches | 73129: neither reaches (nr, nr); 93683: neither reaches (nr, nr); 116089: neither reaches (nr, nr) |
| 50 | A | updates | B | 0.25 | 256→512 | falls | 73129: falls (7,424, 3,072); 93683: falls (4,864, 3,072); 116089: falls (5,632, 2,816) |
| 50 | A | updates | B | 0.25 | 512→1024 | stays | 73129: stays (3,072, 2,048); 93683: stays (3,072, 2,048); 116089: falls (2,816, 1,792) |
| 50 | A | updates | B | 0.50 | 256→512 | falls | 73129: falls (9,216, 4,096); 93683: falls (6,656, 3,584); 116089: falls (4,864, 2,304) |
| 50 | A | updates | B | 0.50 | 512→1024 | falls | 73129: falls (4,096, 2,304); 93683: falls (3,584, 2,048); 116089: stays (2,304, 2,048) |
| 50 | A | weight-items | S | 0.10 | 256→512 | neither reaches | 73129: neither reaches (nr, nr); 93683: neither reaches (nr, nr); 116089: neither reaches (nr, nr) |
| 50 | A | weight-items | S | 0.10 | 512→1024 | neither reaches | 73129: neither reaches (nr, nr); 93683: neither reaches (nr, nr); 116089: neither reaches (nr, nr) |
| 50 | A | weight-items | S | 0.25 | 256→512 | rises | 73129: rises (1.35e+11, 2.58e+11); 93683: rises (1.4e+11, 2.58e+11); 116089: rises (1.87e+11, 2.95e+11) |
| 50 | A | weight-items | S | 0.25 | 512→1024 | rises | 73129: rises (2.58e+11, 5.52e+11); 93683: rises (2.58e+11, 5.52e+11); 116089: rises (2.95e+11, 5.52e+11) |
| 50 | A | weight-items | S | 0.50 | 256→512 | rises | 73129: rises (1.25e+11, 2.76e+11); 93683: rises (1.46e+11, 2.58e+11); 116089: rises (1.3e+11, 2.21e+11) |
| 50 | A | weight-items | S | 0.50 | 512→1024 | rises | 73129: rises (2.76e+11, 6.21e+11); 93683: rises (2.58e+11, 6.21e+11); 116089: rises (2.21e+11, 5.52e+11) |
| 50 | A | weight-items | B | 0.10 | 256→512 | mixed* | 73129: neither reaches (nr, nr); 93683: rises (3.48e+11, nr); 116089: not decided (5.83e+11, nr) |
| 50 | A | weight-items | B | 0.10 | 512→1024 | neither reaches | 73129: neither reaches (nr, nr); 93683: neither reaches (nr, nr); 116089: neither reaches (nr, nr) |
| 50 | A | weight-items | B | 0.25 | 256→512 | rises | 73129: stays (1.51e+11, 2.21e+11); 93683: rises (9.88e+10, 2.21e+11); 116089: rises (1.14e+11, 2.03e+11) |
| 50 | A | weight-items | B | 0.25 | 512→1024 | rises | 73129: rises (2.21e+11, 5.52e+11); 93683: rises (2.21e+11, 5.52e+11); 116089: rises (2.03e+11, 4.83e+11) |
| 50 | A | weight-items | B | 0.50 | 256→512 | rises | 73129: rises (1.87e+11, 2.95e+11); 93683: rises (1.35e+11, 2.58e+11); 116089: rises (9.88e+10, 1.66e+11) |
| 50 | A | weight-items | B | 0.50 | 512→1024 | rises | 73129: rises (2.95e+11, 6.21e+11); 93683: rises (2.58e+11, 5.52e+11); 116089: rises (1.66e+11, 5.52e+11) |
| 90 | R | updates | S | 0.10 | 256→512 | neither reaches | 73129: neither reaches (nr, nr); 93683: neither reaches (nr, nr); 116089: neither reaches (nr, nr) |
| 90 | R | updates | S | 0.10 | 512→1024 | neither reaches | 73129: neither reaches (nr, nr); 93683: neither reaches (nr, nr); 116089: neither reaches (nr, nr) |
| 90 | R | updates | S | 0.25 | 256→512 | neither reaches | 73129: neither reaches (nr, nr); 93683: neither reaches (nr, nr); 116089: neither reaches (nr, nr) |
| 90 | R | updates | S | 0.25 | 512→1024 | neither reaches* | 73129: neither reaches (nr, nr); 93683: neither reaches (nr, nr); 116089: neither reaches (nr, nr) |
| 90 | R | updates | S | 0.50 | 256→512 | falls* | 73129: not decided (28,416, nr); 93683: falls (30,208, 7,424); 116089: falls (nr, 9,472) |
| 90 | R | updates | S | 0.50 | 512→1024 | stays* | 73129: falls (nr, 6,656); 93683: stays (7,424, 5,376); 116089: stays (9,472, 6,400) |
| 90 | R | updates | B | 0.10 | 256→512 | neither reaches | 73129: neither reaches (nr, nr); 93683: neither reaches (nr, nr); 116089: neither reaches (nr, nr) |
| 90 | R | updates | B | 0.10 | 512→1024 | neither reaches | 73129: neither reaches (nr, nr); 93683: neither reaches (nr, nr); 116089: neither reaches (nr, nr) |
| 90 | R | updates | B | 0.25 | 256→512 | not decided* | 73129: not decided (35,840, nr); 93683: falls (17,920, 8,704); 116089: not decided (23,296, nr) |
| 90 | R | updates | B | 0.25 | 512→1024 | neither reaches* | 73129: neither reaches (nr, nr); 93683: not decided (8,704, nr); 116089: neither reaches (nr, nr) |
| 90 | R | updates | B | 0.50 | 256→512 | falls | 73129: falls (20,224, 7,936); 93683: falls (15,360, 7,424); 116089: falls (22,272, 9,216) |
| 90 | R | updates | B | 0.50 | 512→1024 | stays* | 73129: stays (7,936, 6,400); 93683: falls (7,424, 4,608); 116089: stays (9,216, 6,656) |
| 90 | R | weight-items | S | 0.10 | 256→512 | neither reaches | 73129: neither reaches (nr, nr); 93683: neither reaches (nr, nr); 116089: neither reaches (nr, nr) |
| 90 | R | weight-items | S | 0.10 | 512→1024 | neither reaches | 73129: neither reaches (nr, nr); 93683: neither reaches (nr, nr); 116089: neither reaches (nr, nr) |
| 90 | R | weight-items | S | 0.25 | 256→512 | neither reaches | 73129: neither reaches (nr, nr); 93683: neither reaches (nr, nr); 116089: neither reaches (nr, nr) |
| 90 | R | weight-items | S | 0.25 | 512→1024 | neither reaches* | 73129: neither reaches (nr, nr); 93683: neither reaches (nr, nr); 116089: neither reaches (nr, nr) |
| 90 | R | weight-items | S | 0.50 | 256→512 | not decided* | 73129: not decided (5.77e+11, nr); 93683: stays (6.14e+11, 5.34e+11); 116089: not decided (nr, 6.81e+11) |
| 90 | R | weight-items | S | 0.50 | 512→1024 | rises* | 73129: not decided (nr, 1.79e+12); 93683: rises (5.34e+11, 1.45e+12); 116089: rises (6.81e+11, 1.73e+12) |
| 90 | R | weight-items | B | 0.10 | 256→512 | neither reaches | 73129: neither reaches (nr, nr); 93683: neither reaches (nr, nr); 116089: neither reaches (nr, nr) |
| 90 | R | weight-items | B | 0.10 | 512→1024 | neither reaches | 73129: neither reaches (nr, nr); 93683: neither reaches (nr, nr); 116089: neither reaches (nr, nr) |
| 90 | R | weight-items | B | 0.25 | 256→512 | rises* | 73129: not decided (7.28e+11, nr); 93683: rises (3.64e+11, 6.26e+11); 116089: rises (4.73e+11, nr) |
| 90 | R | weight-items | B | 0.25 | 512→1024 | neither reaches* | 73129: neither reaches (nr, nr); 93683: rises (6.26e+11, nr); 116089: neither reaches (nr, nr) |
| 90 | R | weight-items | B | 0.50 | 256→512 | stays* | 73129: stays (4.11e+11, 5.71e+11); 93683: rises (3.12e+11, 5.34e+11); 116089: stays (4.53e+11, 6.63e+11) |
| 90 | R | weight-items | B | 0.50 | 512→1024 | rises | 73129: rises (5.71e+11, 1.73e+12); 93683: rises (5.34e+11, 1.24e+12); 116089: rises (6.63e+11, 1.79e+12) |
| 90 | A | updates | S | 0.10 | 256→512 | neither reaches | 73129: neither reaches (nr, nr); 93683: neither reaches (nr, nr); 116089: neither reaches (nr, nr) |
| 90 | A | updates | S | 0.10 | 512→1024 | neither reaches | 73129: neither reaches (nr, nr); 93683: neither reaches (nr, nr); 116089: neither reaches (nr, nr) |
| 90 | A | updates | S | 0.25 | 256→512 | neither reaches* | 73129: neither reaches (nr, nr); 93683: neither reaches (nr, nr); 116089: neither reaches (nr, nr) |
| 90 | A | updates | S | 0.25 | 512→1024 | neither reaches | 73129: neither reaches (nr, nr); 93683: falls (nr, 7,680); 116089: neither reaches (nr, nr) |
| 90 | A | updates | S | 0.50 | 256→512 | falls* | 73129: not decided (18,944, nr); 93683: falls (18,176, 6,400); 116089: falls (24,832, 9,472) |
| 90 | A | updates | S | 0.50 | 512→1024 | falls* | 73129: falls (nr, 5,120); 93683: stays (6,400, 5,376); 116089: falls (9,472, 5,120) |
| 90 | A | updates | B | 0.10 | 256→512 | neither reaches | 73129: neither reaches (nr, nr); 93683: neither reaches (nr, nr); 116089: neither reaches (nr, nr) |
| 90 | A | updates | B | 0.10 | 512→1024 | neither reaches | 73129: neither reaches (nr, nr); 93683: neither reaches (nr, nr); 116089: neither reaches (nr, nr) |
| 90 | A | updates | B | 0.25 | 256→512 | not decided* | 73129: not decided (27,648, nr); 93683: falls (11,520, 7,424); 116089: not decided (14,592, nr) |
| 90 | A | updates | B | 0.25 | 512→1024 | neither reaches* | 73129: neither reaches (nr, nr); 93683: not decided (7,424, nr); 116089: neither reaches (nr, nr) |
| 90 | A | updates | B | 0.50 | 256→512 | falls | 73129: falls (18,432, 10,240); 93683: falls (9,728, 4,864); 116089: falls (14,848, 9,216) |
| 90 | A | updates | B | 0.50 | 512→1024 | stays | 73129: falls (10,240, 5,888); 93683: stays (4,864, 4,608); 116089: stays (9,216, 6,656) |
| 90 | A | weight-items | S | 0.10 | 256→512 | neither reaches | 73129: neither reaches (nr, nr); 93683: neither reaches (nr, nr); 116089: neither reaches (nr, nr) |
| 90 | A | weight-items | S | 0.10 | 512→1024 | neither reaches | 73129: neither reaches (nr, nr); 93683: neither reaches (nr, nr); 116089: neither reaches (nr, nr) |
| 90 | A | weight-items | S | 0.25 | 256→512 | neither reaches* | 73129: neither reaches (nr, nr); 93683: neither reaches (nr, nr); 116089: neither reaches (nr, nr) |
| 90 | A | weight-items | S | 0.25 | 512→1024 | neither reaches | 73129: neither reaches (nr, nr); 93683: not decided (nr, 2.07e+12); 116089: neither reaches (nr, nr) |
| 90 | A | weight-items | S | 0.50 | 256→512 | stays | 73129: rises (3.85e+11, nr); 93683: stays (3.69e+11, 4.6e+11); 116089: stays (5.05e+11, 6.81e+11) |
| 90 | A | weight-items | S | 0.50 | 512→1024 | rises* | 73129: not decided (nr, 1.38e+12); 93683: rises (4.6e+11, 1.45e+12); 116089: rises (6.81e+11, 1.38e+12) |
| 90 | A | weight-items | B | 0.10 | 256→512 | neither reaches | 73129: neither reaches (nr, nr); 93683: neither reaches (nr, nr); 116089: neither reaches (nr, nr) |
| 90 | A | weight-items | B | 0.10 | 512→1024 | neither reaches | 73129: neither reaches (nr, nr); 93683: neither reaches (nr, nr); 116089: neither reaches (nr, nr) |
| 90 | A | weight-items | B | 0.25 | 256→512 | rises | 73129: rises (5.62e+11, nr); 93683: rises (2.34e+11, 5.34e+11); 116089: rises (2.96e+11, nr) |
| 90 | A | weight-items | B | 0.25 | 512→1024 | neither reaches* | 73129: neither reaches (nr, nr); 93683: rises (5.34e+11, nr); 116089: neither reaches (nr, nr) |
| 90 | A | weight-items | B | 0.50 | 256→512 | rises | 73129: rises (3.74e+11, 7.37e+11); 93683: rises (1.98e+11, 3.5e+11); 116089: rises (3.02e+11, 6.63e+11) |
| 90 | A | weight-items | B | 0.50 | 512→1024 | rises | 73129: rises (7.37e+11, 1.59e+12); 93683: rises (3.5e+11, 1.24e+12); 116089: rises (6.63e+11, 1.79e+12) |

