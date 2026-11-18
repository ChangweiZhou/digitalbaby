Receipts read from `exp45/e49` (E49) and `reference_e42/` (E42's `F10`, seed 73129, from 7fbdbdc2). Complete E49 runs: [('I16', 'B'), ('I16', 'S'), ('I64', 'B')].

### Batch digest at row 4,096 (check 4) and the masks at every scored row (check 3)

| Setting | Rule | digest at 4,096 (first 16) | masks held at every row | E42's limit held at every row |
|---|---|---|---|---|
| F10 | B | bbd71ab7ca27b72e | (E42 receipt) | True |
| F10 | S | bbd71ab7ca27b72e | (E42 receipt) | True |
| I16 | B | bbd71ab7ca27b72e | True | True |
| I16 | S | bbd71ab7ca27b72e | True | True |
| I64 | B | bbd71ab7ca27b72e | True | True |

### Per run: updates to the thresholds, three-row version [single-row in brackets]

| Setting | Rule | A50 | A80 | A90 | R50 | R80 | R90 | max A (row) | max R (row) | final A, R, taught | CPU-h updates + scoring |
|---|---|---|---|---|---|---|---|---|---|---|---|
| F10 | S | 8,192 [7,168] | 13,312 [12,800] | 20,480 [19,456] | 9,472 [8,960] | 15,104 [13,824] | 23,296 [19,456] | 92.17 (24,576) | 92.30 (24,064) | 92.17, 88.45, 98.17 | 0.72 + 0.16 |
| F10 | B | 5,632 [5,632] | 8,704 [8,192] | 15,360 [15,360] | 6,400 [6,400] | 9,472 [8,960] | 19,200 [15,616] | 96.67 (23,296) | 95.30 (24,064) | 91.83, 93.30, 97.50 | 0.70 + 0.16 |
| I64 | B | 15,104 [14,336] | not reached [not reached] | not reached [not reached] | 18,176 [16,640] | not reached [not reached] | not reached [not reached] | 77.00 (23,296) | 67.30 (24,320) | 76.67, 62.75, 83.17 | 0.81 + 0.18 |
| I16 | S | not reached [not reached] | not reached [not reached] | not reached [not reached] | not reached [not reached] | not reached [not reached] | not reached [not reached] | 2.67 (18,944) | 1.35 (256) | 1.00, 0.50, 1.00 | 0.85 + 0.17 |
| I16 | B | not reached [not reached] | not reached [not reached] | not reached [not reached] | not reached [not reached] | not reached [not reached] | not reached [not reached] | 26.17 (24,576) | 17.55 (24,576) | 26.17, 17.55, 27.67 | 0.81 + 0.18 |

### Per run: multiply-adds to the thresholds (multiply-adds a position x 64 items x updates), three-row version [single-row]

| Setting | Rule | multiply-adds a position | A50 | A80 | A90 | R50 | R80 | R90 |
|---|---|---|---|---|---|---|---|---|
| F10 | S | 930,560 | 4.88e+11 [4.27e+11] | 7.93e+11 [7.62e+11] | 1.22e+12 [1.16e+12] | 5.64e+11 [5.34e+11] | 9e+11 [8.23e+11] | 1.39e+12 [1.16e+12] |
| F10 | B | 930,560 | 3.35e+11 [3.35e+11] | 5.18e+11 [4.88e+11] | 9.15e+11 [9.15e+11] | 3.81e+11 [3.81e+11] | 5.64e+11 [5.34e+11] | 1.14e+12 [9.3e+11] |
| I64 | B | 471,808 | 4.56e+11 [4.33e+11] | not reached [not reached] | not reached [not reached] | 5.49e+11 [5.02e+11] | not reached [not reached] | not reached [not reached] |
| I16 | S | 422,656 | not reached [not reached] | not reached [not reached] | not reached [not reached] | not reached [not reached] | not reached [not reached] | not reached [not reached] |
| I16 | B | 422,656 | not reached [not reached] | not reached [not reached] | not reached [not reached] | not reached [not reached] | not reached [not reached] | not reached [not reached] |

### Per run: the three levels (mean of the last three scored rows), the length test at 12, and R at the end against 2,048 updates earlier

| Setting | Rule | A | R | length at 40 | length at 12 | taught | R: last three minus the three ending 2,048 earlier |
|---|---|---|---|---|---|---|---|
| F10 | S | 91.22 | 89.27 | 89.11 | 90.11 | 97.78 | +0.68 |
| F10 | B | 93.22 | 94.08 | 92.56 | 92.89 | 98.39 | +2.07 |
| I64 | B | 74.00 | 65.17 | 61.00 | 77.22 | 81.94 | +7.88 |
| I16 | S | 1.17 | 0.90 | 1.11 | 1.33 | 1.17 | -0.02 |
| I16 | B | 23.83 | 16.87 | 16.67 | 22.78 | 26.72 | +3.37 |

### Per run: cosine between the sweep's and the exact gradient of each masked map (first batch; at the start and at the end), and the mean magnitude of the kept weights (start, end)

| Setting | Rule | input map: cosine start, end | lm: cosine start, end | recurrent (E42's): start, end | input map: magnitude start, end | lm: magnitude start, end |
|---|---|---|---|---|---|---|
| I64 | B | 0.732, 0.505 | 0.867, 0.661 | 0.744, 0.533 | 0.0885, 0.1079 | 0.0879, 0.1255 |
| I16 | S | 0.771, 0.330 | 0.870, 0.591 | 0.777, 0.567 | 0.1765, 0.0986 | 0.1799, 0.1329 |
| I16 | B | 0.771, 0.393 | 0.870, 0.374 | 0.777, 0.430 | 0.1765, 0.1727 | 0.1799, 0.1715 |

### Descriptions against E42's `F10` of seed 73129 under the same rule (fixed beforehand). Three-row version; * where the single-row version describes the cell differently

| Setting | Rule | Measure | Description | F10 | Setting |
|---|---|---|---|---|---|
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

### The brief's condition for stage 2 (I16 under a rule)

- S: **stage 2 runs**: does not reach 80 on R within the dose; updates to 80 on R: more (rises) against F10; lower than F10 on the level of A; lower than F10 on the level of R; lower than F10 on the level of len40
- B: **stage 2 runs**: does not reach 80 on R within the dose; updates to 80 on R: more (rises) against F10; lower than F10 on the level of A; lower than F10 on the level of R; lower than F10 on the level of len40

### What a seed alone changes

E42's `F10` of seeds 93683 and 116089 is not on file complete. The nearest figure (the brief's) is `G` under the sweep, whose three seeds reach 80 on R at 4,864 to 6,656 updates (a ratio of 1.37).

