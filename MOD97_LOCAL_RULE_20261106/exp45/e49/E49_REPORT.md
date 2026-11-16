# E49 report, seed 73129: a limited interface (brief `OPUS_TASK_E49_A_LIMITED_INTERFACE_20261116.md`, revision 2)

One seed, a first measurement: a description, not a claim. Nothing here is independently confirmed. The reviewer's reader
(`e49_interface_reader.py` on `claude/vibrant-cerf-sje726`) has not yet been applied to these weights. No formal checker was run.

- **Brief** revision 2 at 74b56ab7 on the default branch `codex/minifly-consolidated-20261023`. Audit of revision 1:
  `AUDIT_E49_LIMITED_INTERFACE_BRIEF_20261116.md` at d6fde19b on `claude/vibrant-cerf-sje726`.
- **Branch** `claude/peaceful-noether-vcvrmf`, which starts at 74b56ab7. The runner was committed in 71e81a0e.
  `exp49_interface.py` sha256 cdf93ce1…; `e49_interface.py` 92f968e0… (the specification's copy, which differs in its import line only).
  E42's runner `exp42_cells.py` 340f9cc5… is imported unchanged; it, `e42_cells.py` and `e42_tables.py` are copied from 4c9e52d1. The tables
  come from `e49_tables.py`.
- **Reference**: E42's `F10` of seed 73129 under each rule, copied from 7fbdbdc2 (`claude/gallant-dijkstra-i9zgsr`) into
  `exp45/e49/reference_e42/`, with the sha256 in `SOURCE.txt`. It was not run again.
- **Host**: Linux x86-64, 4 cores, Python 3.13.18, torch 2.15.0+cpu, one thread per process, at most three processes. Runs were
  launched with `setsid nohup` outside any wrapper with a time limit. This worker is a new session, not E42's worker.
- **Stage 0**: the six checks pass (`exp45/e49_qual/E49_STAGE0_73129.md`). Check 2 is bit-equal under both rules: `tensors_sha256` 837dadc1… (sweep) and 13398546… (backpropagation), with scores equal at all nine rows. The backpropagation reference is the dry run of `F10` that E42's runner made unchanged.
- **Stage 1**: `I16` S, `I16` B and `I64` B ran from 21:00:11Z; `I64` S from 22:01:15Z to 23:02:16Z. All exited with rc=0.
- **Stage 2**: the brief's condition holds under both rules. Under each rule, `I16` does not reach 80 on R within the dose, needs
  *more* updates than `F10`, and is *lower* on all three levels. So `in16` and `out8` ran under both rules without a further release:
  `in16` B and `out8` B from 22:03:10Z to 23:01:56Z, and `in16` S and `out8` S from 23:01Z to 23:57:37Z. All exited with rc=0, and
  the times are in `logs/queue.log`.

## Summary

All eight runs completed. Each run is 24,576 updates on seed 73129.

- **Under backpropagation**, no limited setting reaches 80 on R within the dose. `I64` comes nearest: R 65.2 over the last three rows,
  still rising. `I16` reaches 16.9.
- **Under the sweep**, both limited settings stay at or below 3 on every panel at every scored row. The training loss stops falling
  at about 1.1 after some 2,048 updates.
- **The brief's condition for stage 2 holds under both rules**, so `in16` and `out8` were run under both.
  - Under backpropagation, `in16` (16 input values a cell, dense readout) reaches 80 on R at 20,480 updates. `out8` (dense input, 8
    cells a readout unit) does not reach it within the dose; it ends at 70.8 and is still rising.
  - Under the sweep, `in16` stays at or below 4.2 on every panel, and `out8` reaches 7.8 on R over the last three rows.
- **Against `F10` of seed 73129**, every setting under each rule needs *more* updates to 80 on A and on R.
  - Every setting is *lower* on all three levels, except `in16` under backpropagation, which is *about the same* on A and R and
    *lower* on the length test at 40 letters.
  - Every difference from `F10` lies beyond the spread between `F10`'s seeds now on file: three seeds under the sweep, two under
    backpropagation.

| Setting | Rule | updates to 80 on R (three-row) | levels A, R, length at 40 | R: last three minus the three ending 2,048 earlier | against `F10`: updates to 80 on R; levels |
|---|---|---|---|---|---|
| `F10` (E42's) | B | 9,472 | 93.2, 94.1, 92.6 | +2.07 | reference |
| `I64` | B | not reached | 74.0, 65.2, 61.0 | +7.88, **still rising** | more; lower, lower, lower |
| `I16` | B | not reached | 23.8, 16.9, 16.7 | +3.37 | more; lower, lower, lower |
| `in16` | B | 20,480 | 85.2, 82.5, 66.1 | +0.65 | more; about the same, about the same, lower |
| `out8` | B | not reached | 73.8, 70.8, 68.0 | +8.60, **still rising** | more; lower, lower, lower |
| `F10` (E42's) | S | 15,104 | 91.2, 89.3, 89.1 | +0.68 | reference |
| `I64` | S | not reached | 1.3, 0.9, 1.3 | −0.25 | more; lower, lower, lower |
| `I16` | S | not reached | 1.2, 0.9, 1.1 | −0.02 | more; lower, lower, lower |
| `in16` | S | not reached | 1.9, 1.4, 1.2 | −0.35 | more; lower, lower, lower |
| `out8` | S | not reached | 10.6, 7.8, 7.9 | +2.37 | more; lower, lower, lower |

**Cost.** The eight runs took 7.73 CPU-hours, scoring included (0.90 to 1.02 each). Stage 0 took 0.39 CPU-hours. The brief estimated
about 0.8 a run.

**Weights on the remote** (`git ls-tree -r --name-only HEAD MOD97_LOCAL_RULE_20261106/exp45/e49/ | grep .pt$`): `e49_<setting>_<rule>_73129_start.pt`
and `e49_<setting>_<rule>_73129.pt` for `I64`, `I16`, `in16` and `out8` under `S` and `B`. That is sixteen files, each with the three masks.

## The sentence per rule (the brief's form)

**Backpropagation.**
- **`I64`**: each cell reads 64 of the input's 512 values and each readout unit reads 32 of the cells. This limits a cell's
  connections, not what it can know of the byte. Under this limit the network **does not learn the first family within the dose**.
  It is below 80 on R at update 24,576. Its R over the last three rows is 65.2, 7.9 points above the three rows that end 2,048 updates
  earlier, so it is **still rising at the end of the dose**. It needs **more** updates than with the dense interface (`F10` reaches
  80 on R at 9,472). It ends **lower** on all three levels: A 74.0, R 65.2 and the length test at 40 letters 61.0, against 93.2, 94.1
  and 92.6.
- **`I16`**: each cell reads 16 of the 512 values and each readout unit 8 of the cells, under the same kind of limit. The network
  **does not learn the first family within the dose**. Its R over the last three rows is 16.9, 3.4 points above the three rows
  ending 2,048 updates earlier, which is below the brief's 5 points for "still rising". It needs **more** updates than with the dense
  interface and is **lower** on all three levels: A 23.8, R 16.9 and the length test at 40 letters 16.7.

**Sweep.**
- **`I64`**: with each cell reading 64 values and each readout unit 32 cells, under the same kind of limit, the network
  **does not learn the first family within the dose**. A, R, the taught panel and the length test stay at or below 3.0 at every
  scored row (largest A 1.67, R 1.50, taught 3.00). R over the last three rows is 0.9, and its change against 2,048 updates earlier is
  −0.25. It needs **more** updates than with the dense interface (`F10` reaches 80 on R at 15,104) and is **lower** on all three
  levels.
- **`I16`**: with 16 values and 8 cells, under the same kind of limit, the network **does not learn the first family within the
  dose**. A, R, the taught panel and the length test stay at or below 2.7 at every scored
  row (largest A 2.67, R 1.35). R over the last three rows is 0.9, and its
  change is −0.02. It needs **more** updates than `F10` and is **lower** on all three levels.

In multiply-adds the readings are **not decided** for all eight cells of `I64` and `I16` in the three-row version, as the brief
foresaw. The one exception is the single-row version of `I64` B on A, which is *rises*. In that version the last update that can be
recorded is 24,576, and 24,576 / 8,192 × 0.507 = 1.52, above 1.5. The sentence on learning rests on the reading in updates.

## Stage 2: which side costs (the brief's forms only)

**Backpropagation.**
- **`in16`**: 16 input values a cell, every readout unit reads every cell.
  - It reaches 80 on A at 16,640 updates and 80 on R at 20,480. That is *more* than `F10` in updates (8,704 and 9,472) and *stays* in
    multiply-adds (ratios 1.13 and 1.28).
  - Its levels of A and R are *about the same* (85.2 and 82.5, against 93.2 and 94.1). Its length test at 40 letters is *lower*
    (66.1 against 92.6).
- **`out8`**: every cell reads all 512 values, 8 cells a readout unit.
  - It does not reach 80 on A or R within the dose. That is *more* in updates and *rises* in multiply-adds: at 0.864 of `F10`'s
    multiply-adds a position, 24,064 recordable updates are 2.2 times `F10`'s on R.
  - It is *lower* on all three levels (73.8, 70.8, 68.0). R is still rising at the end (+8.6).
- Of the two, `out8` is the one that does not reach 80 on R within the dose.

**Sweep.**
- **`in16`**: does not reach 80 on A or R within the dose. Every panel stays at or below 4.2 at every scored row (largest A 3.33,
  R 2.25, taught 4.17). R over the last three rows is 1.4 (−0.35). It is *lower* on all three levels and *more* in updates.
- **`out8`**: does not reach 80 within the dose either. R over the last three rows is 7.8, 2.4 points above the three rows ending
  2,048 updates earlier, below the 5 points of "still rising". A is 11.8 at the last row.
  - It is *more* in updates and *lower* on all three levels.
  - In multiply-adds it *rises* on A (three-row; 24,064 × 0.864 / 13,312 = 1.56). On R it is *not decided* in the three-row version
    and *rises* in the single-row version.
- Neither reaches 80 on R. By the mean of R over the last three rows, `in16` is the lower (1.4 against 7.8).

## What a seed alone changes

E42's `F10` is now complete on E42's branch at f2a694eb for seed 93683 under both rules and for seed 116089 under the sweep; `F10`
under backpropagation on 116089 was not on file. The three receipts are copied into `reference_e42/other_seeds/`, with the sha256 in
`SOURCE.txt`.

- Under the sweep, `F10` reaches 80 on R at 14,336 to 15,104 updates over three seeds (ratio 1.05).
- Under backpropagation it reaches 80 on R at 7,936 to 9,472 updates over two seeds (ratio 1.19).
- Its levels differ between seeds by at most 3.6 points on A and R and 7.0 on the length test under the sweep, and by at most 4.4,
  2.4 and 5.1 points under backpropagation.

Every difference of an E49 run from `F10` of seed 73129 lies beyond these spreads, so none is marked; the table is under "Tables".
The smallest are `in16` B's levels of A and R (−8.0 and −11.6 points, against spreads of 4.4 and 2.4). Under backpropagation the spread
rests on two seeds.

## Predictions (the coordinator's), scored

| | Prediction | Chance | Outcome |
|---|---|---|---|
| P49.1 | Checks 1 and 2 pass | 90% | **Right.** Check 1 equals the Mac output line for line. Check 2 is bit-equal under both rules (837dadc1…, 13398546…), with scores equal at every row. |
| P49.2 | `I64`, backpropagation: reaches 80 on R within the dose | 75% | **Wrong.** It does not reach 80 on R: R over the last three rows is 65.2, still rising (+7.9). |
| P49.3 | `I64`, sweep: reaches 80 on R within the dose | 55% | **Wrong.** R stays at or below 1.5 at every row. |
| P49.4 | `I16`, backpropagation: reaches 80 on R within the dose | 50% | **Wrong.** R over the last three rows is 16.9. |
| P49.5 | `I16`, sweep: reaches 80 on R within the dose | 30% | **Wrong.** R stays at or below 1.35 at every row. |
| P49.6 | `I64` under backpropagation is *about the same* as `F10` in updates to 80 on R | 50% | **Wrong.** The reading is *more*: it does not reach 80, and the 24,064 recordable updates are 2.54 times `F10`'s 9,472. |
| P49.7 | Stage 2 runs under at least one rule | 65% | **Right.** It runs under both rules. |
| P49.8 | If stage 2 runs under backpropagation, `in16` is the worse of the two: later to 80 on R, or not reaching it where `out8` does; if neither reaches it, the lower mean of R over the last three rows | 60% | **Wrong.** `in16` B reaches 80 on R at 20,480 updates and `out8` B does not reach it (R 70.8 over the last three rows), so `out8` is the worse of the two by the prediction's own terms. |

Two of eight are right.

## Tables (generated by `python3 MOD97_LOCAL_RULE_20261106/e49_tables.py --tag 73129 --f10-seeds MOD97_LOCAL_RULE_20261106/exp45/e49/reference_e42/other_seeds`)

Receipts read from `exp45/e49` (E49) and `reference_e42/` (E42's `F10`, seed 73129, from 7fbdbdc2). Complete E49 runs: [('I16', 'B'), ('I16', 'S'), ('I64', 'B'), ('I64', 'S'), ('in16', 'B'), ('in16', 'S'), ('out8', 'B'), ('out8', 'S')].

#### Batch digest at row 4,096 (check 4) and the masks at every scored row (check 3)

| Setting | Rule | digest at 4,096 (first 16) | masks held at every row | E42's limit held at every row |
|---|---|---|---|---|
| F10 | B | bbd71ab7ca27b72e | (E42 receipt) | True |
| F10 | S | bbd71ab7ca27b72e | (E42 receipt) | True |
| I16 | B | bbd71ab7ca27b72e | True | True |
| I16 | S | bbd71ab7ca27b72e | True | True |
| I64 | B | bbd71ab7ca27b72e | True | True |
| I64 | S | bbd71ab7ca27b72e | True | True |
| in16 | B | bbd71ab7ca27b72e | True | True |
| in16 | S | bbd71ab7ca27b72e | True | True |
| out8 | B | bbd71ab7ca27b72e | True | True |
| out8 | S | bbd71ab7ca27b72e | True | True |

#### Per run: updates to the thresholds, three-row version [single-row in brackets]

| Setting | Rule | A50 | A80 | A90 | R50 | R80 | R90 | max A (row) | max R (row) | final A, R, taught | CPU-h updates + scoring |
|---|---|---|---|---|---|---|---|---|---|---|---|
| F10 | S | 8,192 [7,168] | 13,312 [12,800] | 20,480 [19,456] | 9,472 [8,960] | 15,104 [13,824] | 23,296 [19,456] | 92.17 (24,576) | 92.30 (24,064) | 92.17, 88.45, 98.17 | 0.72 + 0.16 |
| F10 | B | 5,632 [5,632] | 8,704 [8,192] | 15,360 [15,360] | 6,400 [6,400] | 9,472 [8,960] | 19,200 [15,616] | 96.67 (23,296) | 95.30 (24,064) | 91.83, 93.30, 97.50 | 0.70 + 0.16 |
| I64 | S | not reached [not reached] | not reached [not reached] | not reached [not reached] | not reached [not reached] | not reached [not reached] | not reached [not reached] | 1.67 (1,024) | 1.50 (9,984) | 1.33, 1.00, 1.83 | 0.83 + 0.17 |
| I64 | B | 15,104 [14,336] | not reached [not reached] | not reached [not reached] | 18,176 [16,640] | not reached [not reached] | not reached [not reached] | 77.00 (23,296) | 67.30 (24,320) | 76.67, 62.75, 83.17 | 0.81 + 0.18 |
| I16 | S | not reached [not reached] | not reached [not reached] | not reached [not reached] | not reached [not reached] | not reached [not reached] | not reached [not reached] | 2.67 (18,944) | 1.35 (256) | 1.00, 0.50, 1.00 | 0.85 + 0.17 |
| I16 | B | not reached [not reached] | not reached [not reached] | not reached [not reached] | not reached [not reached] | not reached [not reached] | not reached [not reached] | 26.17 (24,576) | 17.55 (24,576) | 26.17, 17.55, 27.67 | 0.81 + 0.18 |
| in16 | S | not reached [not reached] | not reached [not reached] | not reached [not reached] | not reached [not reached] | not reached [not reached] | not reached [not reached] | 3.33 (17,664) | 2.25 (17,408) | 2.17, 1.30, 3.50 | 0.76 + 0.16 |
| in16 | B | 11,264 [10,496] | 16,640 [16,640] | not reached [not reached] | 11,776 [11,264] | 20,480 [19,456] | not reached [not reached] | 87.67 (23,296) | 84.10 (23,552) | 87.67, 83.20, 93.33 | 0.78 + 0.17 |
| out8 | S | not reached [not reached] | not reached [not reached] | not reached [not reached] | not reached [not reached] | not reached [not reached] | not reached [not reached] | 11.83 (24,576) | 8.90 (24,320) | 11.83, 7.30, 12.00 | 0.75 + 0.16 |
| out8 | B | 11,264 [9,216] | not reached [not reached] | not reached [not reached] | 14,592 [14,080] | not reached [not reached] | not reached [not reached] | 78.83 (24,576) | 71.90 (23,808) | 78.83, 70.40, 87.17 | 0.77 + 0.17 |

#### Per run: multiply-adds to the thresholds (multiply-adds a position x 64 items x updates), three-row version [single-row]

| Setting | Rule | multiply-adds a position | A50 | A80 | A90 | R50 | R80 | R90 |
|---|---|---|---|---|---|---|---|---|
| F10 | S | 930,560 | 4.88e+11 [4.27e+11] | 7.93e+11 [7.62e+11] | 1.22e+12 [1.16e+12] | 5.64e+11 [5.34e+11] | 9e+11 [8.23e+11] | 1.39e+12 [1.16e+12] |
| F10 | B | 930,560 | 3.35e+11 [3.35e+11] | 5.18e+11 [4.88e+11] | 9.15e+11 [9.15e+11] | 3.81e+11 [3.81e+11] | 5.64e+11 [5.34e+11] | 1.14e+12 [9.3e+11] |
| I64 | S | 471,808 | not reached [not reached] | not reached [not reached] | not reached [not reached] | not reached [not reached] | not reached [not reached] | not reached [not reached] |
| I64 | B | 471,808 | 4.56e+11 [4.33e+11] | not reached [not reached] | not reached [not reached] | 5.49e+11 [5.02e+11] | not reached [not reached] | not reached [not reached] |
| I16 | S | 422,656 | not reached [not reached] | not reached [not reached] | not reached [not reached] | not reached [not reached] | not reached [not reached] | not reached [not reached] |
| I16 | B | 422,656 | not reached [not reached] | not reached [not reached] | not reached [not reached] | not reached [not reached] | not reached [not reached] | not reached [not reached] |
| in16 | S | 549,632 | not reached [not reached] | not reached [not reached] | not reached [not reached] | not reached [not reached] | not reached [not reached] | not reached [not reached] |
| in16 | B | 549,632 | 3.96e+11 [3.69e+11] | 5.85e+11 [5.85e+11] | not reached [not reached] | 4.14e+11 [3.96e+11] | 7.2e+11 [6.84e+11] | not reached [not reached] |
| out8 | S | 803,584 | not reached [not reached] | not reached [not reached] | not reached [not reached] | not reached [not reached] | not reached [not reached] | not reached [not reached] |
| out8 | B | 803,584 | 5.79e+11 [4.74e+11] | not reached [not reached] | not reached [not reached] | 7.5e+11 [7.24e+11] | not reached [not reached] | not reached [not reached] |

#### Per run: the three levels (mean of the last three scored rows), the length test at 12, and R at the end against 2,048 updates earlier

| Setting | Rule | A | R | length at 40 | length at 12 | taught | R: last three minus the three ending 2,048 earlier |
|---|---|---|---|---|---|---|---|
| F10 | S | 91.22 | 89.27 | 89.11 | 90.11 | 97.78 | +0.68 |
| F10 | B | 93.22 | 94.08 | 92.56 | 92.89 | 98.39 | +2.07 |
| I64 | S | 1.28 | 0.93 | 1.33 | 1.78 | 1.56 | -0.25 |
| I64 | B | 74.00 | 65.17 | 61.00 | 77.22 | 81.94 | +7.88 |
| I16 | S | 1.17 | 0.90 | 1.11 | 1.33 | 1.17 | -0.02 |
| I16 | B | 23.83 | 16.87 | 16.67 | 22.78 | 26.72 | +3.37 |
| in16 | S | 1.89 | 1.37 | 1.22 | 1.89 | 2.50 | -0.35 |
| in16 | B | 85.22 | 82.53 | 66.11 | 85.33 | 93.61 | +0.65 |
| out8 | S | 10.61 | 7.75 | 7.89 | 9.78 | 12.78 | +2.37 |
| out8 | B | 73.83 | 70.77 | 68.00 | 73.44 | 82.22 | +8.60 |

#### Per run: A and R every 2,048 updates (the receipts' scored rows)

| Setting | Rule | 2,048 | 4,096 | 6,144 | 8,192 | 10,240 | 12,288 | 14,336 | 16,384 | 18,432 | 20,480 | 22,528 | 24,576 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| F10 | S | 1.3, 1.1 | 11.5, 7.2 | 32.0, 25.4 | 59.7, 46.5 | 67.7, 56.5 | 78.0, 70.9 | 83.3, 80.5 | 88.3, 86.0 | 88.8, 87.1 | 90.7, 88.0 | 88.8, 88.5 | 92.2, 88.5 |
| F10 | B | 1.2, 1.5 | 17.7, 12.9 | 51.8, 46.0 | 80.0, 76.0 | 82.8, 82.3 | 86.5, 83.3 | 84.8, 83.7 | 90.3, 89.4 | 85.8, 86.8 | 94.2, 91.6 | 94.2, 92.7 | 91.8, 93.3 |
| I64 | S | 0.7, 0.8 | 0.8, 0.8 | 1.0, 1.0 | 0.5, 0.8 | 1.2, 0.9 | 0.3, 0.8 | 1.0, 0.5 | 0.7, 1.0 | 1.2, 0.9 | 1.0, 0.4 | 0.8, 1.4 | 1.3, 1.0 |
| I64 | B | 1.3, 1.6 | 0.8, 0.8 | 1.3, 1.0 | 3.3, 1.8 | 8.0, 4.5 | 24.7, 18.2 | 51.3, 34.1 | 57.5, 46.6 | 65.5, 51.5 | 65.0, 52.5 | 70.5, 64.5 | 76.7, 62.8 |
| I16 | S | 1.5, 0.7 | 0.5, 1.1 | 0.8, 0.9 | 0.5, 0.7 | 1.0, 1.1 | 1.7, 0.5 | 0.7, 0.6 | 1.0, 1.1 | 0.8, 1.2 | 0.7, 1.1 | 1.8, 1.1 | 1.0, 0.5 |
| I16 | B | 1.7, 1.1 | 0.8, 0.9 | 1.0, 1.1 | 1.5, 0.9 | 1.8, 1.2 | 1.8, 3.0 | 3.5, 3.3 | 5.2, 5.2 | 9.2, 7.2 | 14.7, 11.4 | 16.2, 12.8 | 26.2, 17.6 |
| in16 | S | 0.7, 0.7 | 0.7, 1.1 | 2.0, 0.8 | 1.7, 0.8 | 1.3, 1.1 | 2.7, 1.9 | 1.7, 1.5 | 2.0, 1.9 | 1.8, 1.7 | 1.7, 2.2 | 2.7, 1.9 | 2.2, 1.3 |
| in16 | B | 1.2, 1.5 | 2.7, 3.0 | 9.8, 6.7 | 26.7, 22.6 | 43.0, 36.1 | 63.8, 56.8 | 69.3, 62.2 | 78.8, 72.7 | 81.5, 77.7 | 86.8, 82.0 | 85.5, 81.0 | 87.7, 83.2 |
| out8 | S | 0.5, 1.1 | 0.5, 0.8 | 1.3, 1.1 | 1.5, 0.8 | 2.0, 0.9 | 1.0, 0.9 | 2.2, 1.4 | 1.5, 3.0 | 3.7, 3.5 | 6.2, 4.9 | 6.3, 6.8 | 11.8, 7.3 |
| out8 | B | 0.7, 1.9 | 1.0, 1.4 | 7.3, 6.0 | 28.0, 22.1 | 45.2, 35.5 | 62.7, 48.5 | 56.5, 45.8 | 71.2, 59.5 | 68.3, 58.6 | 74.2, 61.3 | 76.8, 66.6 | 78.8, 70.4 |

#### Per run: mean training loss of the 256 updates before the row (data from the receipts)

| Setting | Rule | 256 | 2,048 | 4,096 | 8,192 | 12,288 | 16,384 | 20,480 | 24,576 |
|---|---|---|---|---|---|---|---|---|---|
| F10 | S | 1.609 | 0.904 | 0.624 | 0.250 | 0.127 | 0.074 | 0.058 | 0.047 |
| F10 | B | 1.604 | 0.832 | 0.486 | 0.161 | 0.099 | 0.073 | 0.057 | 0.045 |
| I64 | S | 1.751 | 1.139 | 1.082 | 1.090 | 1.079 | 1.080 | 1.121 | 1.145 |
| I64 | B | 1.735 | 1.014 | 0.848 | 0.742 | 0.434 | 0.252 | 0.202 | 0.169 |
| I16 | S | 1.885 | 1.329 | 1.164 | 1.135 | 1.141 | 1.168 | 1.163 | 1.200 |
| I16 | B | 1.840 | 1.142 | 0.982 | 0.848 | 0.777 | 0.647 | 0.515 | 0.426 |
| in16 | S | 1.685 | 1.125 | 1.040 | 1.017 | 1.014 | 1.020 | 1.040 | 1.049 |
| in16 | B | 1.675 | 1.057 | 0.842 | 0.427 | 0.219 | 0.140 | 0.107 | 0.084 |
| out8 | S | 1.698 | 1.028 | 0.940 | 0.888 | 0.868 | 0.805 | 0.726 | 0.625 |
| out8 | B | 1.682 | 0.958 | 0.804 | 0.371 | 0.231 | 0.186 | 0.155 | 0.138 |

#### Per run: cosine between the sweep's and the exact gradient of each masked map (first batch; at the start and at the end), and the mean magnitude of the kept weights (start, end)

| Setting | Rule | input map: cosine start, end | lm: cosine start, end | recurrent (E42's): start, end | input map: magnitude start, end | lm: magnitude start, end |
|---|---|---|---|---|---|---|
| I64 | S | 0.732, 0.555 | 0.867, 0.730 | 0.744, 0.566 | 0.0885, 0.0864 | 0.0879, 0.0781 |
| I64 | B | 0.732, 0.505 | 0.867, 0.661 | 0.744, 0.533 | 0.0885, 0.1079 | 0.0879, 0.1255 |
| I16 | S | 0.771, 0.330 | 0.870, 0.591 | 0.777, 0.567 | 0.1765, 0.0986 | 0.1799, 0.1329 |
| I16 | B | 0.771, 0.393 | 0.870, 0.374 | 0.777, 0.430 | 0.1765, 0.1727 | 0.1799, 0.1715 |
| in16 | S | 0.742, 0.619 | 0.865, 0.544 | 0.731, 0.605 | 0.1765, 0.1020 | 0.0312, 0.1035 |
| in16 | B | 0.742, 0.727 | 0.865, 0.752 | 0.731, 0.706 | 0.1765, 0.1981 | 0.0312, 0.0774 |
| out8 | S | 0.763, 0.599 | 0.865, 0.749 | 0.777, 0.666 | 0.0313, 0.0839 | 0.1799, 0.1736 |
| out8 | B | 0.763, 0.402 | 0.865, 0.548 | 0.777, 0.315 | 0.0313, 0.0734 | 0.1799, 0.1592 |

#### Descriptions against E42's `F10` of seed 73129 under the same rule (fixed beforehand). Three-row version; * where the single-row version describes the cell differently

| Setting | Rule | Measure | Description | F10 | Setting |
|---|---|---|---|---|---|
| I64 | S | A80 updates | more | 13,312 | not reached |
| I64 | S | A80 multiply-adds | not decided | 7.93e+11 | not reached |
| I64 | S | R80 updates | more | 15,104 | not reached |
| I64 | S | R80 multiply-adds | not decided | 9e+11 | not reached |
| I64 | S | level of A | lower | 91.22 | 1.28 |
| I64 | S | level of R | lower | 89.27 | 0.93 |
| I64 | S | level of the length test at 40 | lower | 89.11 | 1.33 |
| I64 | B | A80 updates | more | 8,704 | not reached |
| I64 | B | A80 multiply-adds | not decided* | 5.18e+11 | not reached |
| I64 | B | R80 updates | more | 9,472 | not reached |
| I64 | B | R80 multiply-adds | not decided | 5.64e+11 | not reached |
| I64 | B | level of A | lower | 93.22 | 74.00 |
| I64 | B | level of R | lower | 94.08 | 65.17 |
| I64 | B | level of the length test at 40 | lower | 92.56 | 61.00 |
| I16 | S | A80 updates | more | 13,312 | not reached |
| I16 | S | A80 multiply-adds | not decided | 7.93e+11 | not reached |
| I16 | S | R80 updates | more | 15,104 | not reached |
| I16 | S | R80 multiply-adds | not decided | 9e+11 | not reached |
| I16 | S | level of A | lower | 91.22 | 1.17 |
| I16 | S | level of R | lower | 89.27 | 0.90 |
| I16 | S | level of the length test at 40 | lower | 89.11 | 1.11 |
| I16 | B | A80 updates | more | 8,704 | not reached |
| I16 | B | A80 multiply-adds | not decided | 5.18e+11 | not reached |
| I16 | B | R80 updates | more | 9,472 | not reached |
| I16 | B | R80 multiply-adds | not decided | 5.64e+11 | not reached |
| I16 | B | level of A | lower | 93.22 | 23.83 |
| I16 | B | level of R | lower | 94.08 | 16.87 |
| I16 | B | level of the length test at 40 | lower | 92.56 | 16.67 |
| in16 | S | A80 updates | more | 13,312 | not reached |
| in16 | S | A80 multiply-adds | not decided | 7.93e+11 | not reached |
| in16 | S | R80 updates | more | 15,104 | not reached |
| in16 | S | R80 multiply-adds | not decided | 9e+11 | not reached |
| in16 | S | level of A | lower | 91.22 | 1.89 |
| in16 | S | level of R | lower | 89.27 | 1.37 |
| in16 | S | level of the length test at 40 | lower | 89.11 | 1.22 |
| in16 | B | A80 updates | more | 8,704 | 16,640 |
| in16 | B | A80 multiply-adds | stays | 5.18e+11 | 5.85e+11 |
| in16 | B | R80 updates | more | 9,472 | 20,480 |
| in16 | B | R80 multiply-adds | stays | 5.64e+11 | 7.2e+11 |
| in16 | B | level of A | about the same | 93.22 | 85.22 |
| in16 | B | level of R | about the same | 94.08 | 82.53 |
| in16 | B | level of the length test at 40 | lower | 92.56 | 66.11 |
| out8 | S | A80 updates | more | 13,312 | not reached |
| out8 | S | A80 multiply-adds | rises | 7.93e+11 | not reached |
| out8 | S | R80 updates | more | 15,104 | not reached |
| out8 | S | R80 multiply-adds | not decided* | 9e+11 | not reached |
| out8 | S | level of A | lower | 91.22 | 10.61 |
| out8 | S | level of R | lower | 89.27 | 7.75 |
| out8 | S | level of the length test at 40 | lower | 89.11 | 7.89 |
| out8 | B | A80 updates | more | 8,704 | not reached |
| out8 | B | A80 multiply-adds | rises | 5.18e+11 | not reached |
| out8 | B | R80 updates | more | 9,472 | not reached |
| out8 | B | R80 multiply-adds | rises | 5.64e+11 | not reached |
| out8 | B | level of A | lower | 93.22 | 73.83 |
| out8 | B | level of R | lower | 94.08 | 70.77 |
| out8 | B | level of the length test at 40 | lower | 92.56 | 68.00 |

#### The brief's condition for stage 2 (I16 under a rule)

- S: **stage 2 runs**: does not reach 80 on R within the dose; updates to 80 on R: more (rises) against F10; lower than F10 on the level of A; lower than F10 on the level of R; lower than F10 on the level of len40
- B: **stage 2 runs**: does not reach 80 on R within the dose; updates to 80 on R: more (rises) against F10; lower than F10 on the level of A; lower than F10 on the level of R; lower than F10 on the level of len40

#### What a seed alone changes

E42's `F10` beside E49's (three-row version; levels are the means of the last three rows):

| F10 rule | seed | A80 | R80 | level of A | level of R | length at 40 |
|---|---|---|---|---|---|---|
| B | 73129 | 8,704 | 9,472 | 93.22 | 94.08 | 92.56 |
| B | 93683 | 6,912 | 7,936 | 97.61 | 96.45 | 97.67 |
| S | 73129 | 13,312 | 15,104 | 91.22 | 89.27 | 89.11 |
| S | 93683 | 13,056 | 14,336 | 94.61 | 92.87 | 95.22 |
| S | 116089 | 12,544 | 14,336 | 91.17 | 90.48 | 88.22 |

The spread between seeds of `F10` under a rule: for updates the ratio of the most to the fewest, for a level the most minus the least. A difference of an E49 run from `F10` of seed 73129 that is smaller than this spread is marked **within the seeds' spread**. A run that does not reach a threshold differs from `F10` by a ratio of at least 24,064 / `F10`'s updates.

| Rule | seeds of F10 | A80 ratio | R80 ratio | A points | R points | length at 40 points |
|---|---|---|---|---|---|---|
| S | 73129, 93683, 116089 | 1.061 | 1.054 | 3.44 | 3.60 | 7.00 |
| B | 73129, 93683 | 1.259 | 1.194 | 4.39 | 2.37 | 5.11 |

| Setting | Rule | measure | E49 against F10 of 73129 | spread of F10's seeds | mark |
|---|---|---|---|---|---|
| I64 | S | A80 updates | ratio at least 1.808 | 1.061 | beyond the seeds' spread |
| I64 | S | R80 updates | ratio at least 1.593 | 1.054 | beyond the seeds' spread |
| I64 | S | level of A | -89.94 | 3.44 | beyond the seeds' spread |
| I64 | S | level of R | -88.33 | 3.60 | beyond the seeds' spread |
| I64 | S | level of len40 | -87.78 | 7.00 | beyond the seeds' spread |
| I64 | B | A80 updates | ratio at least 2.765 | 1.259 | beyond the seeds' spread |
| I64 | B | R80 updates | ratio at least 2.541 | 1.194 | beyond the seeds' spread |
| I64 | B | level of A | -19.22 | 4.39 | beyond the seeds' spread |
| I64 | B | level of R | -28.92 | 2.37 | beyond the seeds' spread |
| I64 | B | level of len40 | -31.56 | 5.11 | beyond the seeds' spread |
| I16 | S | A80 updates | ratio at least 1.808 | 1.061 | beyond the seeds' spread |
| I16 | S | R80 updates | ratio at least 1.593 | 1.054 | beyond the seeds' spread |
| I16 | S | level of A | -90.06 | 3.44 | beyond the seeds' spread |
| I16 | S | level of R | -88.37 | 3.60 | beyond the seeds' spread |
| I16 | S | level of len40 | -88.00 | 7.00 | beyond the seeds' spread |
| I16 | B | A80 updates | ratio at least 2.765 | 1.259 | beyond the seeds' spread |
| I16 | B | R80 updates | ratio at least 2.541 | 1.194 | beyond the seeds' spread |
| I16 | B | level of A | -69.39 | 4.39 | beyond the seeds' spread |
| I16 | B | level of R | -77.22 | 2.37 | beyond the seeds' spread |
| I16 | B | level of len40 | -75.89 | 5.11 | beyond the seeds' spread |
| in16 | S | A80 updates | ratio at least 1.808 | 1.061 | beyond the seeds' spread |
| in16 | S | R80 updates | ratio at least 1.593 | 1.054 | beyond the seeds' spread |
| in16 | S | level of A | -89.33 | 3.44 | beyond the seeds' spread |
| in16 | S | level of R | -87.90 | 3.60 | beyond the seeds' spread |
| in16 | S | level of len40 | -87.89 | 7.00 | beyond the seeds' spread |
| in16 | B | A80 updates | ratio 1.912 | 1.259 | beyond the seeds' spread |
| in16 | B | R80 updates | ratio 2.162 | 1.194 | beyond the seeds' spread |
| in16 | B | level of A | -8.00 | 4.39 | beyond the seeds' spread |
| in16 | B | level of R | -11.55 | 2.37 | beyond the seeds' spread |
| in16 | B | level of len40 | -26.44 | 5.11 | beyond the seeds' spread |
| out8 | S | A80 updates | ratio at least 1.808 | 1.061 | beyond the seeds' spread |
| out8 | S | R80 updates | ratio at least 1.593 | 1.054 | beyond the seeds' spread |
| out8 | S | level of A | -80.61 | 3.44 | beyond the seeds' spread |
| out8 | S | level of R | -81.52 | 3.60 | beyond the seeds' spread |
| out8 | S | level of len40 | -81.22 | 7.00 | beyond the seeds' spread |
| out8 | B | A80 updates | ratio at least 2.765 | 1.259 | beyond the seeds' spread |
| out8 | B | R80 updates | ratio at least 2.541 | 1.194 | beyond the seeds' spread |
| out8 | B | level of A | -19.39 | 4.39 | beyond the seeds' spread |
| out8 | B | level of R | -23.32 | 2.37 | beyond the seeds' spread |
| out8 | B | level of len40 | -24.56 | 5.11 | beyond the seeds' spread |


## NOT_RUN and open items

- **Seeds 93683 and 116089**: not run. They are not released; the brief releases them for the smallest setting that learns, at the
  coordinator's decision.
- **Continuing a run beyond the dose**: not done; the brief leaves this to the coordinator. The runs described as still rising at the
  end of the dose are `I64` B (+7.88) and `out8` B (+8.60). Each run's final weights are on the remote, so a continuation would start from them, although the
  optimizer's state is not saved.
- **The reviewer's reader** on the saved weights and answers (recommendation b of the audit), and the reviewer's repetition of the
  measurement of its section 4 on the codes the limited networks learned (recommendation g): not done here; both are the reviewer's.
- **`F10` under backpropagation on seed 116089**: not on file when this report was written. The spread between seeds under
  backpropagation rests on two seeds.

## What it cannot show (the brief's list, unchanged)

The masks limit how many values a cell computes with, not what it can know about the byte: 16 values of the byte's code still identify
the byte. Whether the network learns when a cell cannot know the byte needs a fixed input code in which a value stands for few bytes,
which the brief names as the next job. The job also leaves untested more than one seed, an input with geometry, and the byte's code and
the readout's own layers (`l1`, `l2` and `l3` stay dense, and every readout unit still reads all 512 values of the byte's code through
`l2`). It does not vary the number of cells (E47 does), or the settings: the learning rate and decay are E42's, and the scale of the
kept weights at the start is a choice. The kept weights start 2.83 times (`I64`) or 5.66 times (`I16`) larger than the dense ones,
which the table of magnitudes records.

## What I did not check

- **Why the limited settings under the sweep stay low** (`I64`, `I16` and `in16` near chance, `out8` at 7.8 on R): no diagnosis was
  run. The brief allows no reading
  beyond its forms, and none is offered. The data that bear on it are in the tables: the cosines between the sweep's and the exact
  gradient, the mean magnitudes of the kept weights, and the training loss.
- **Whether the start's scale matters**: one scale was run, the brief's.
- **The other seeds' draws**: only seed 73129's draws were used.

## Files

- Runs: `exp45/e49/e49_<setting>_<rule>_73129.json`, with `_start.pt` and `.pt` (the thirteen tensors, the recurrent `mask`, `mask_in`, `mask_out`, `k_in`, `k_out`, periods, phases and `made`) and `_answers.json` (the bytes of every answer on the length-test panel at 12 and 40 letters, at the last row). Logs are in `exp45/e49/logs/`.
- Tables: `exp45/e49/E49_TABLES_73129.md` and `E49_SUMMARY_73129.json`, from `python3 MOD97_LOCAL_RULE_20261106/e49_tables.py`.
- Stage 0: `exp45/e49_qual/`. This holds the checks' receipts, the dry runs of 256 updates with their weights and answers, and E42's
  `F10` dry run under backpropagation by E42's runner (`check2_e42_runner/`).
