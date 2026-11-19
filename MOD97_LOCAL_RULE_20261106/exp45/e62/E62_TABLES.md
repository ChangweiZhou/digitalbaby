# E62 tables (seed 73129)

Written by `python3 e62_tables.py` from the receipts under `exp45/e62/` and, for the record and the reference, from the default branch (`origin/codex/minifly-consolidated-20261023`) with `git show`. Updates to a threshold: the first of three consecutive scored rows at or above it, with the first single row in brackets. Levels are means of three scored rows. All scores are percentages.

## The table of the runs

| N | Connections from | Exact gradient: updates to 80 on R | Sweep: updates to 80 on R | Ratio sweep / exact gradient | Exact gradient: to 80 on A | Sweep: to 80 on A | Read by no other cell | Every cell from every cell within | Weights taking part |
|---|---|---|---|---|---|---|---|---|---|
| 2 | all cells | 15,616 [13,824] | 30,976 [26,368] | 1.98 | 16,384 [15,616] | 33,280 [26,880] | 40 | none | 562,432 |
| 4 | all cells | 17,408 [16,640] | not reached | — | 18,688 [17,664] | not reached | 4 | none | 563,968 |
| 8 | all cells | 18,176 [16,640] | not reached | — | 18,176 [15,616] | not reached | 0 | 5 | 567,040 |
| 10 | the neighbourhood (the record) | 15,360 [13,824] | 34,304 [34,304] | 2.23 | 14,080 [12,288] | 35,584 [33,536] | 0 (record) | 5 (record) | 568,576 |
| 10 | all cells | 16,128 [16,128] | not reached | — | 16,128 [14,848] | 47,616 [43,776] | 0 | 4 | 568,576 |
| 16 | all cells | 13,824 [12,288] | 42,752 [39,936] | 3.09 | 13,056 [10,752] | 40,704 [37,120] | 0 | 4 | 573,184 |
| 32 | all cells | 11,520 [11,520] | 38,656 [38,656] | 3.36 | 11,776 [10,752] | 38,656 [29,696] | 0 | 3 | 585,472 |
| 64 | all cells | 7,680 [7,168] | 32,256 [20,736] | 4.20 | 7,680 [7,168] | 28,672 [18,176] | 0 | 2 | 610,048 |
| 128 | all cells | 8,704 [8,704] | 12,288 [10,240] | 1.41 | 8,704 [7,936] | 9,216 [8,704] | 0 | 2 | 659,200 |
| 255 | all cells | 8,704 [8,704] | 7,168 [7,168] | 0.82 | 8,704 [8,704] | 6,144 [6,144] | 0 | 1 | 756,736 |

## Levels, cost to 80 on R, and resumptions

| N | From | Rule | A, R, T at 24,576 | A, R, T at 49,152 | Multiply-adds to 80 on R | CPU-seconds of updates to 80 on R | CPU-hours of the run | Resumed at |
|---|---|---|---|---|---|---|---|---|
| 2 | all | B | 86.94, 88.50, 96.00 | 93.33, 93.07, 98.28 | 562,108,039,168 | 1697 | 1.84 | never |
| 2 | all | S | 75.39, 77.53, 89.33 | 85.11, 85.97, 94.89 | 1,115,001,192,448 | 3956 | 2.10 | never |
| 4 | all | B | 85.28, 85.50, 94.83 | 94.22, 94.38, 98.72 | 628,323,516,416 | 1900 | 1.85 | never |
| 4 | all | S | 61.06, 57.82, 80.83 | 76.39, 76.70, 91.00 | — | — | 2.11 | never |
| 8 | all | B | 84.72, 84.77, 92.06 | 94.50, 94.65, 98.72 | 659,617,218,560 | 2020 | 1.86 | never |
| 8 | all | S | 63.44, 62.60, 84.50 | 78.33, 80.17, 94.11 | — | — | 2.11 | never |
| 10 | near | B | 88.50, 88.87, 94.89 | 93.83, 94.25, 98.33 | 558,932,951,040 | 1726 | 1.90 | never |
| 10 | near | S | 63.61, 65.50, 84.61 | 85.89, 87.22, 94.00 | 1,248,283,590,656 | 4561 | 2.18 | never |
| 10 | all | B | 85.61, 86.47, 93.11 | 94.78, 94.08, 98.89 | 586,879,598,592 | 1788 | 1.85 | never |
| 10 | all | S | 60.56, 53.05, 79.61 | 80.50, 79.82, 91.56 | — | — | 2.11 | never |
| 16 | all | B | 87.22, 88.83, 93.44 | 92.11, 91.92, 96.78 | 507,116,519,424 | 1509 | 1.78 | [16384] |
| 16 | all | S | 64.00, 57.95, 82.56 | 83.61, 84.48, 94.83 | 1,568,304,791,552 | 5503 | 2.11 | never |
| 32 | all | B | 88.17, 88.52, 94.06 | 93.28, 92.90, 96.72 | 431,656,796,160 | 1244 | 1.77 | never |
| 32 | all | S | 68.39, 67.58, 85.00 | 84.89, 85.88, 95.00 | 1,448,448,360,448 | 5051 | 2.11 | never |
| 64 | all | B | 91.39, 92.32, 94.83 | 93.72, 93.17, 96.61 | 299,850,792,960 | 789 | 1.76 | [4096] |
| 64 | all | S | 77.33, 76.57, 84.11 | 83.56, 81.75, 90.11 | 1,259,373,330,432 | 3978 | 2.01 | [8192] |
| 128 | all | B | 91.22, 94.75, 95.11 | 96.39, 96.73, 98.83 | 367,211,315,200 | 962 | 1.87 | never |
| 128 | all | S | 89.28, 86.85, 92.67 | 87.83, 88.10, 93.83 | 518,415,974,400 | 1573 | 2.12 | never |
| 255 | all | B | 95.06, 95.23, 97.39 | 96.78, 97.55, 97.22 | 421,544,329,216 | 806 | 1.53 | never |
| 255 | all | S | 90.67, 89.27, 94.00 | 89.39, 90.88, 94.33 | 347,154,153,472 | 807 | 1.79 | [45056] |

## Reached within k steps (least / mean / most), from the receipts

| N | From | 1 | 2 | 3 | 4 | 6 | 8 | Strongly connected parts (largest) |
|---|---|---|---|---|---|---|---|---|
| 2 | all | 1 / 3.0 / 7 | 1 / 6.9 / 20 | 1 / 14.5 / 50 | 1 / 28.6 / 98 | 1 / 88.6 / 224 | 1 / 165.0 / 256 | 49 (207) |
| 4 | all | 1 / 5.0 / 12 | 1 / 20.4 / 52 | 1 / 71.8 / 172 | 1 / 175.9 / 255 | 1 / 251.4 / 256 | 1 / 252.0 / 256 | 5 (252) |
| 8 | all | 3 / 9.0 / 16 | 14 / 64.9 / 123 | 78 / 223.7 / 256 | 241 / 255.9 / 256 | 256 / 256.0 / 256 | 256 / 256.0 / 256 | 1 (256) |
| 10 | all | 4 / 11.0 / 20 | 35 / 91.8 / 148 | 203 / 249.7 / 256 | 256 / 256.0 / 256 | 256 / 256.0 / 256 | 256 / 256.0 / 256 | 1 (256) |
| 16 | all | 6 / 17.0 / 28 | 75 / 171.0 / 220 | 254 / 256.0 / 256 | 256 / 256.0 / 256 | 256 / 256.0 / 256 | 256 / 256.0 / 256 | 1 (256) |
| 32 | all | 17 / 33.0 / 55 | 232 / 252.9 / 256 | 256 / 256.0 / 256 | 256 / 256.0 / 256 | 256 / 256.0 / 256 | 256 / 256.0 / 256 | 1 (256) |
| 64 | all | 49 / 65.0 / 89 | 256 / 256.0 / 256 | 256 / 256.0 / 256 | 256 / 256.0 / 256 | 256 / 256.0 / 256 | 256 / 256.0 / 256 | 1 (256) |
| 128 | all | 108 / 129.0 / 150 | 256 / 256.0 / 256 | 256 / 256.0 / 256 | 256 / 256.0 / 256 | 256 / 256.0 / 256 | 256 / 256.0 / 256 | 1 (256) |
| 255 | all | 256 / 256.0 / 256 | 256 / 256.0 / 256 | 256 / 256.0 / 256 | 256 / 256.0 / 256 | 256 / 256.0 / 256 | 256 / 256.0 / 256 | 1 (256) |

## Readings

- On seed 73129, with connections drawn from all cells, the exact gradient reached 80 on R after u updates with N connections a cell, for N = 2, 4, 8, 16, 32, 64, 128, 255: 15,616, 17,408, 18,176, 13,824, 11,520, 7,680, 8,704, 8,704; the smallest N of the grid of powers of two that was as fast as the network in which every cell reads every other was 64, and every larger N of the grid was also as fast.
- With 10 connections from all cells the exact gradient reached 80 on R after 16,128 updates, against 15,360 (the exact gradient) with 10 connections from the neighbourhood.
- On seed 73129, with connections drawn from all cells, the sweep reached 80 on R after u updates with N connections a cell, for N = 2, 4, 8, 16, 32, 64, 128, 255: 30,976, not reached, not reached, 42,752, 38,656, 32,256, 12,288, 7,168; no N below 255 was as fast as the network in which every cell reads every other; the smallest N that reached 80 at all was 2.
- With 10 connections from all cells the sweep did not reach 80 on R within 49,152 updates, against 34,304 (the sweep) with 10 connections from the neighbourhood.
- The ratio of the sweep's updates to the exact gradient's was 1.98 at N = 2; 3.09 at N = 16; 3.36 at N = 32; 4.20 at N = 64; 1.41 at N = 128; 0.82 at N = 255, beside the record's 2.23 (N = 10 from the neighbourhood).

Beside the record: decision 31's reference (E42's `G` under the exact gradient, `origin/codex/minifly-consolidated-20261023:MOD97_LOCAL_RULE_20261106/exp45/e42/e42_G_B_73129.json`) reached 80 on R after 6,400 [5,120] updates and on A after 5,120 [5,120].

## Predictions

| | Holds | Figures |
|---|---|---|
| P62.1 | yes | {"note": "each check was run once"} |
| P62.2 | yes | {"updates": {"4": 17408, "8": 18176, "16": 13824, "32": 11520, "64": 7680, "128": 8704, "255": 8704}} |
| P62.3 | yes | {"updates": 15616} |
| P62.4 | yes | {"updates": {"16": 42752, "32": 38656, "64": 32256, "128": 12288, "255": 7168}} |
| P62.5 | no | {"updates": 30976} |
| P62.6 | yes | {"updates": {"S": [7168, 34304], "B": [8704, 15360]}} |
| P62.7 | no | {"ratios": {"S": null, "B": 1.05}} |
| P62.8 | no | {"pairs": {"2": [30976, 15616], "16": [42752, 13824], "32": [38656, 11520], "64": [32256, 7680], "128": [12288, 8704], "255": [7168, 8704]}} |
| P62.9 | no | {"word": "slower"} |
