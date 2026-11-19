# E62 tables (seed 73129)

Written by `python3 e62_tables.py` from the receipts under `exp45/e62/` and, for the record and the reference, from the default branch (`origin/codex/minifly-consolidated-20261023`) with `git show`. Updates to a threshold: the first of three consecutive scored rows at or above it, with the first single row in brackets. Levels are means of three scored rows. All scores are percentages.

## The table of the runs

| N | Connections from | Exact gradient: updates to 80 on R | Sweep: updates to 80 on R | Ratio sweep / exact gradient | Exact gradient: to 80 on A | Sweep: to 80 on A | Read by no other cell | Every cell from every cell within | Weights taking part |
|---|---|---|---|---|---|---|---|---|---|
| 8 | all cells | 18,176 [16,640] | not reached | — | 18,176 [15,616] | not reached | 0 | 5 | 567,040 |
| 10 | the neighbourhood (the record) | 15,360 [13,824] | 34,304 [34,304] | 2.23 | 14,080 [12,288] | 35,584 [33,536] | 0 (record) | 5 (record) | 568,576 |
| 32 | all cells | 11,520 [11,520] | 38,656 [38,656] | 3.36 | 11,776 [10,752] | 38,656 [29,696] | 0 | 3 | 585,472 |
| 255 | all cells | 8,704 [8,704] | 7,168 [7,168] | 0.82 | 8,704 [8,704] | 6,144 [6,144] | 0 | 1 | 756,736 |

## Levels, cost to 80 on R, and resumptions

| N | From | Rule | A, R, T at 24,576 | A, R, T at 49,152 | Multiply-adds to 80 on R | CPU-seconds of updates to 80 on R | CPU-hours of the run | Resumed at |
|---|---|---|---|---|---|---|---|---|
| 8 | all | B | 84.72, 84.77, 92.06 | 94.50, 94.65, 98.72 | 659,617,218,560 | 2020 | 1.86 | never |
| 8 | all | S | 63.44, 62.60, 84.50 | 78.33, 80.17, 94.11 | — | — | 2.11 | never |
| 10 | near | B | 88.50, 88.87, 94.89 | 93.83, 94.25, 98.33 | 558,932,951,040 | 1726 | 1.90 | never |
| 10 | near | S | 63.61, 65.50, 84.61 | 85.89, 87.22, 94.00 | 1,248,283,590,656 | 4561 | 2.18 | never |
| 32 | all | B | 88.17, 88.52, 94.06 | 93.28, 92.90, 96.72 | 431,656,796,160 | 1244 | 1.77 | never |
| 32 | all | S | 68.39, 67.58, 85.00 | 84.89, 85.88, 95.00 | 1,448,448,360,448 | 5051 | 2.11 | never |
| 255 | all | B | 95.06, 95.23, 97.39 | 96.78, 97.55, 97.22 | 421,544,329,216 | 806 | 1.53 | never |
| 255 | all | S | 90.67, 89.27, 94.00 | 89.39, 90.88, 94.33 | 347,154,153,472 | 807 | 1.79 | [45056] |

## Reached within k steps (least / mean / most), from the receipts

| N | From | 1 | 2 | 3 | 4 | 6 | 8 | Strongly connected parts (largest) |
|---|---|---|---|---|---|---|---|---|
| 8 | all | 3 / 9.0 / 16 | 14 / 64.9 / 123 | 78 / 223.7 / 256 | 241 / 255.9 / 256 | 256 / 256.0 / 256 | 256 / 256.0 / 256 | 1 (256) |
| 32 | all | 17 / 33.0 / 55 | 232 / 252.9 / 256 | 256 / 256.0 / 256 | 256 / 256.0 / 256 | 256 / 256.0 / 256 | 256 / 256.0 / 256 | 1 (256) |
| 255 | all | 256 / 256.0 / 256 | 256 / 256.0 / 256 | 256 / 256.0 / 256 | 256 / 256.0 / 256 | 256 / 256.0 / 256 | 256 / 256.0 / 256 | 1 (256) |

## Readings

- On seed 73129, with connections drawn from all cells, the exact gradient reached 80 on R after u updates with N connections a cell, for N = 8, 32, 255: 18,176, 11,520, 8,704; no N below 255 was as fast as the network in which every cell reads every other; the smallest N that reached 80 at all was 8.
- On seed 73129, with connections drawn from all cells, the sweep reached 80 on R after u updates with N connections a cell, for N = 8, 32, 255: not reached, 38,656, 7,168; no N below 255 was as fast as the network in which every cell reads every other; the smallest N that reached 80 at all was 32.
- The ratio of the sweep's updates to the exact gradient's was 3.36 at N = 32; 0.82 at N = 255, beside the record's 2.23 (N = 10 from the neighbourhood).

Beside the record: decision 31's reference (E42's `G` under the exact gradient, `origin/codex/minifly-consolidated-20261023:MOD97_LOCAL_RULE_20261106/exp45/e42/e42_G_B_73129.json`) reached 80 on R after 6,400 [5,120] updates and on A after 5,120 [5,120].

## Predictions

| | Holds | Figures |
|---|---|---|
| P62.1 | yes | {"note": "each check was run once"} |
| P62.6 | yes | {"updates": {"S": [7168, 34304], "B": [8704, 15360]}} |
| P62.9 | no | {"word": "slower"} |
