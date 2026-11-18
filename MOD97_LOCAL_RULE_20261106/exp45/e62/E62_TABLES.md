# E62 tables (seed 73129)

Written by `python3 e62_tables.py` from the receipts under `exp45/e62/` and, for the record and the reference, from the default branch (`origin/codex/minifly-consolidated-20261023`) with `git show`. Updates to a threshold: the first of three consecutive scored rows at or above it, with the first single row in brackets. Levels are means of three scored rows. All scores are percentages.

## The table of the runs

| N | Connections from | Exact gradient: updates to 80 on R | Sweep: updates to 80 on R | Ratio sweep / exact gradient | Exact gradient: to 80 on A | Sweep: to 80 on A | Read by no other cell | Every cell from every cell within | Weights taking part |
|---|---|---|---|---|---|---|---|---|---|
| 10 | the neighbourhood (the record) | 15,360 [13,824] | 34,304 [34,304] | 2.23 | 14,080 [12,288] | 35,584 [33,536] | 0 (record) | 5 (record) | 568,576 |

## Levels, cost to 80 on R, and resumptions

| N | From | Rule | A, R, T at 24,576 | A, R, T at 49,152 | Multiply-adds to 80 on R | CPU-seconds of updates to 80 on R | CPU-hours of the run | Resumed at |
|---|---|---|---|---|---|---|---|---|
| 10 | near | B | 88.50, 88.87, 94.89 | 93.83, 94.25, 98.33 | 558,932,951,040 | 1726 | 1.90 | never |
| 10 | near | S | 63.61, 65.50, 84.61 | 85.89, 87.22, 94.00 | 1,248,283,590,656 | 4561 | 2.18 | never |

## Reached within k steps (least / mean / most), from the receipts

| N | From | 1 | 2 | 3 | 4 | 6 | 8 | Strongly connected parts (largest) |
|---|---|---|---|---|---|---|---|---|

## Readings


Beside the record: decision 31's reference (E42's `G` under the exact gradient, `origin/codex/minifly-consolidated-20261023:MOD97_LOCAL_RULE_20261106/exp45/e42/e42_G_B_73129.json`) reached 80 on R after 6,400 [5,120] updates and on A after 5,120 [5,120].

## Predictions

| | Holds | Figures |
|---|---|---|
| P62.1 | yes | {"note": "each check was run once"} |
