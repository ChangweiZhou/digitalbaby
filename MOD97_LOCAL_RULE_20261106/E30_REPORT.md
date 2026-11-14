# E30: the registered gate measure, read on unseen files. Report

14 November 2026. Opus 5.5 worker. Branch `claude/wizardly-franklin-sr2l6q`, restarted from the default branch at 7287e071 after its earlier work was merged. Brief: `OPUS_TASK_E30_GATE_MEASURE_20261113.md` (6c97998f). This is a read-only measurement: nothing was trained and no existing file was edited; `e30_gates.py` is new. Host: Linux container, Python 3.13.18, torch 2.15.0+cpu, one thread. As the brief asks, this report gives the tables and the four outcomes and no readings beyond them.

## Step 1: calibration on seen files (passed; pushed as c939a88c before step 2)

U on the plus-minus composition panel (1,200 items), for the twelve lives. Ranges are the review's; the tolerance is ±3 points.

| birth × life update | 73129 | 93683 | 116089 | review range |
|---|---|---|---|---|
| SS (E26 life S) | 73.17 | 72.59 | 76.44 | 73–79 |
| BS (E26-X) | 78.30 | 76.45 | 78.82 | 73–79 |
| BB (E26 life B) | 53.04 | 45.24 | 50.39 | 43–53 |
| SB (E26-X) | 48.67 | 42.65 | 49.36 | 43–53 |

All twelve are within tolerance. Three lie just outside the bare range: SS 93683 (−0.41), SB 93683 (−0.35) and BB 73129 (+0.04). Record: `exp45/e30/E30_CALIBRATION.json`.

## Step 2: unseen files

**Sources.**
- **E28:** branch `claude/charming-cori-ljzwz1` at commit 3c06a3d1, the latest commit at reading. The files were extracted with `git show` into a scratch directory and not copied into this branch. Final files are checked against `weights_sha256` and row files against `row_weights[step].sha256`; every one matched.
- **Complete E28 finals at 3c06a3d1:** S-all, B-all, S-bal and B-bal for 73129; S-all and B-all for 93683.
- **NOT_RUN:** S-bal 93683, B-bal 93683 and all four arms of 116089. Stage 2b was still in flight, and their receipts are not complete.
- **E26:** the six births in `exp45/e26/`, each checked against its receipt.

H is accuracy at 12 letters minus accuracy at 40 letters, on mod-89 addition, 300 items each, scored by `exp16_integrated.score`. Accuracy at 12 and at 40 is shown in brackets. Record: `exp45/e30/E30_UNSEEN.json`.

### U (undecided share, %)

| file | arm | seed | row | U, add-89 k = 12 | U, plus-minus |
|---|---|---|---|---|---|
| E28 final | S-all | 73129 | 24576 | 91.66 | 68.37 |
| E28 final | B-all | 73129 | 24576 | 38.60 | 50.25 |
| E28 final | S-bal | 73129 | 24576 | 84.32 | 88.03 |
| E28 final | B-bal | 73129 | 24576 | 35.43 | 48.35 |
| E28 final | S-all | 93683 | 24576 | 91.64 | 72.50 |
| E28 final | B-all | 93683 | 24576 | 42.59 | 56.42 |
| E28 row | S-all | 73129 | 4096 / 8192 / 12288 / 16384 / 20480 | 64.74 / 80.96 / 87.30 / 90.04 / 90.85 | 69.33 / 76.58 / 78.92 / 77.24 / 72.91 |
| E28 row | S-all | 93683 | 4096 / 8192 / 12288 / 16384 / 20480 | 65.51 / 80.01 / 88.23 / 89.95 / 91.55 | 71.88 / 74.27 / 77.14 / 76.15 / 72.96 |
| E26 birth | S | 73129 | final | 48.55 | 65.19 |
| E26 birth | B | 73129 | final | 32.52 | 51.46 |
| E26 birth | S | 93683 | final | 39.04 | 68.46 |
| E26 birth | B | 93683 | final | 30.68 | 47.51 |
| E26 birth | S | 116089 | final | 38.62 | 62.17 |
| E26 birth | B | 116089 | final | 25.68 | 43.32 |

### H (holding loss, points)

| file | arm | seed | row | H (acc at 12 − acc at 40) |
|---|---|---|---|---|
| E28 final | S-all | 73129 | 24576 | 73.33 (75.3 − 2.0) |
| E28 final | B-all | 73129 | 24576 | 9.00 (93.7 − 84.7) |
| E28 final | S-bal | 73129 | 24576 | 85.33 (91.3 − 6.0) |
| E28 final | B-bal | 73129 | 24576 | 7.67 (94.7 − 87.0) |
| E28 final | S-all | 93683 | 24576 | 77.33 (79.7 − 2.3) |
| E28 final | B-all | 93683 | 24576 | 14.67 (96.3 − 81.7) |
| E28 row | S-all | 73129 | 4096 / 8192 / 12288 / 16384 / 20480 | 8.67 (15.7 − 7.0) / 56.00 (56.7 − 0.7) / 61.33 (61.3 − 0.0) / 66.00 (66.3 − 0.3) / 36.33 (37.0 − 0.7) |
| E28 row | S-all | 93683 | 4096 / 8192 / 12288 / 16384 / 20480 | 25.00 (25.7 − 0.7) / 40.33 (42.0 − 1.7) / 51.33 (53.7 − 2.3) / 63.00 (64.0 − 1.0) / 56.67 (58.0 − 1.3) |
| E26 birth | S | 73129 | final | 2.33 (98.3 − 96.0) |
| E26 birth | B | 73129 | final | 0.67 (98.3 − 97.7) |
| E26 birth | S | 93683 | final | 1.67 (98.7 − 97.0) |
| E26 birth | B | 93683 | final | 0.67 (99.3 − 98.7) |
| E26 birth | S | 116089 | final | 0.33 (100.0 − 99.7) |
| E26 birth | B | 116089 | final | 2.33 (94.7 − 92.3) |

## The four outcomes

Rule applied: differences under 5 points in U are not read. Seeds with both E28 S-all and B-all are 73129 and 93683; 116089 is not run.

- **P30.1 holds (2 of 2 available seeds).** U on add-89, S-all against B-all: 91.66 vs 38.60 (+53.06) in 73129, and 91.64 vs 42.59 (+49.05) in 93683.
- **P30.2 holds (2 of 3 seeds).** U on add-89 for the E26 births, S against B: +16.03 in 73129 (yes), +8.36 in 93683 (no; under 10), +12.94 in 116089 (yes).
- **P30.3 holds (2 of 2 available seeds).** H, S-all against B-all: 73.33 vs 9.00 (+64.33) in 73129, and 77.33 vs 14.67 (+62.66) in 93683.
- **P30.4 misses.** On the add-89 panel, U rises from row 4096 to the end in both seeds: 64.74 → 91.66 and 65.51 → 91.64. The brief does not name a panel, so the plus-minus panel is given too: 69.33 → 68.37 (−0.96, under 5, not read) and 71.88 → 72.50. It misses there as well.

P30.1 and P30.3 are judged on two seeds, not three, because the 116089 births were not complete at 3c06a3d1.

## Choices the brief left open

- **"Every position after the first byte"** is read as input positions 1 to L−1 of x = [BOS] + prompt + answer[:−1]; position 0 is the BOS byte. With this choice the calibration passed.
- **"Held half of the split"** is read as `exp11_cell.split(t, .5, seed)[1]`, for the births on add-89. No lives were scored for H in this job.
- **P30.4's "end"** is the final file at row 24576.
