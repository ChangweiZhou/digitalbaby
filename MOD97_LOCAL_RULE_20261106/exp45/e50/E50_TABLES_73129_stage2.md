Receipts read from `exp45/e50/` (E50) and `exp45/e50/reference/` (E42's `F10` and E49's `in16`, seed 73129, copied from a36de661). Rules: B is backpropagation, S the sweep. A run of 49,152 updates appears twice: read within its first 24,576 updates (`@24,576`) and at its end (`@end`). Thresholds are in the three-row version (the first of three consecutive scored rows at or above the threshold), with the single-row version in brackets.

### Checks inside the runs: batch digest at row 4,096 (E49's check 4), the masks (E49's check 3), E42's limit, the code (check 2)

| Setting | Rule | digest at 4,096 (first 16) | masks held at every row | E42's limit held at every row | code table, byte table and `l1` unchanged at every row |
|---|---|---|---|---|---|
| F10 | B | bbd71ab7ca27b72e | (E42 receipt) | True | (learned code) |
| in16 | B | bbd71ab7ca27b72e | True | True | (learned code) |
| C-dense | B | bbd71ab7ca27b72e | True | True | True |
| C16-in @24,576 | B | bbd71ab7ca27b72e | True | True | True |
| C16-in @end | B | bbd71ab7ca27b72e | True | True | True |
| F10 | S | bbd71ab7ca27b72e | (E42 receipt) | True | (learned code) |
| in16 | S | bbd71ab7ca27b72e | True | True | (learned code) |
| C-dense | S | bbd71ab7ca27b72e | True | True | True |
| C16-in @24,576 | S | bbd71ab7ca27b72e | True | True | True |
| C16-in @end | S | bbd71ab7ca27b72e | True | True | True |

### Per run: updates to the thresholds on A and R

| Setting | Rule | A50 | A80 | A90 | R50 | R80 | R90 | max A (row) | max R (row) | CPU-h updates + scoring |
|---|---|---|---|---|---|---|---|---|---|---|
| F10 | B | 5,632 [5,632] | 8,704 [8,192] | 15,360 [15,360] | 6,400 [6,400] | 9,472 [8,960] | 19,200 [15,616] | 96.67 (23,296) | 95.30 (24,064) | 0.70 + 0.16 |
| in16 | B | 11,264 [10,496] | 16,640 [16,640] | not reached [not reached] | 11,776 [11,264] | 20,480 [19,456] | not reached [not reached] | 87.67 (23,296) | 84.10 (23,552) | 0.78 + 0.17 |
| C-dense | B | 4,352 [4,352] | 7,680 [6,912] | 12,288 [10,240] | 4,352 [4,352] | 6,912 [6,912] | 12,288 [10,240] | 98.67 (24,576) | 98.15 (23,808) | 0.71 + 0.16 |
| C16-in @24,576 | B | 13,568 [13,568] | not reached [not reached] | not reached [not reached] | 18,176 [16,640] | not reached [not reached] | not reached [not reached] | 73.50 (24,320) | 64.10 (24,320) | 0.72 + 0.16 |
| C16-in @end | B | 13,568 [13,568] | 40,704 [28,928] | not reached [not reached] | 18,176 [16,640] | not reached [47,616] | not reached [not reached] | 85.83 (47,360) | 80.75 (47,616) | 1.44 + 0.32 |
| F10 | S | 8,192 [7,168] | 13,312 [12,800] | 20,480 [19,456] | 9,472 [8,960] | 15,104 [13,824] | 23,296 [19,456] | 92.17 (24,576) | 92.30 (24,064) | 0.72 + 0.16 |
| in16 | S | not reached [not reached] | not reached [not reached] | not reached [not reached] | not reached [not reached] | not reached [not reached] | not reached [not reached] | 3.33 (17,664) | 2.25 (17,408) | 0.76 + 0.16 |
| C-dense | S | 7,936 [7,936] | 13,056 [13,056] | 20,736 [17,152] | 7,424 [7,424] | 12,800 [12,032] | 17,664 [16,384] | 96.17 (24,064) | 96.15 (23,808) | 0.83 + 0.16 |
| C16-in @24,576 | S | not reached [not reached] | not reached [not reached] | not reached [not reached] | not reached [not reached] | not reached [not reached] | not reached [not reached] | 33.67 (23,040) | 30.20 (24,064) | 0.84 + 0.16 |
| C16-in @end | S | 47,104 [47,104] | not reached [not reached] | not reached [not reached] | not reached [not reached] | not reached [not reached] | not reached [not reached] | 54.17 (47,360) | 45.50 (46,336) | 1.68 + 0.32 |

### Per run: the four operations apart (the panel behind A: untaught pairs at 12 letters, 300 items each): updates to 80, and the level (mean of the last three rows)

| Setting | Rule | max: to 80 | min: to 80 | add: to 80 | sub: to 80 | levels max, min, add, sub |
|---|---|---|---|---|---|---|
| F10 | B | 2,048 [2,048] | 2,048 [2,048] | 8,704 [8,704] | 9,472 [7,936] | 97.0, 97.4, 92.9, 93.6 |
| in16 | B | 5,120 [5,120] | 5,376 [5,376] | 15,616 [15,616] | 18,432 [16,640] | 96.4, 96.7, 85.3, 85.1 |
| C-dense | B | 1,024 [1,024] | 1,280 [1,280] | 7,680 [6,912] | 7,680 [6,912] | 97.8, 96.9, 97.3, 98.0 |
| C16-in @24,576 | B | 5,376 [5,376] | 5,632 [5,632] | not reached [not reached] | not reached [not reached] | 95.2, 95.1, 72.8, 63.1 |
| C16-in @end | B | 5,376 [5,376] | 5,632 [5,632] | 34,560 [25,856] | 40,704 [33,280] | 97.9, 96.0, 84.3, 83.2 |
| F10 | S | 3,072 [3,072] | 2,816 [2,816] | 11,776 [11,264] | 15,104 [13,824] | 97.8, 97.8, 90.1, 92.3 |
| in16 | S | not reached [not reached] | not reached [not reached] | not reached [not reached] | not reached [not reached] | 38.8, 40.6, 1.9, 1.9 |
| C-dense | S | 1,280 [1,280] | 1,536 [1,536] | 12,032 [12,032] | 13,824 [13,824] | 97.7, 95.7, 94.0, 93.0 |
| C16-in @24,576 | S | 7,424 [7,424] | 7,680 [7,168] | not reached [not reached] | not reached [not reached] | 95.6, 94.9, 36.4, 29.0 |
| C16-in @end | S | 7,424 [7,424] | 7,680 [7,168] | not reached [not reached] | not reached [not reached] | 96.3, 96.2, 49.4, 49.6 |

### Per run: multiply-adds to the thresholds, three-row version. First count: weights that take part x 64 items x updates (E49's count; for this job without `l1`). Second count: products with a turned-on value x 64 x updates (this job's settings only)

| Setting | Rule | first count a position | A80 | R80 | second count a position | A80 | R80 |
|---|---|---|---|---|---|---|---|
| F10 | B | 930,560 | 5.18e+11 | 5.64e+11 |  |  |  |
| in16 | B | 549,632 | 5.85e+11 | 7.2e+11 |  |  |  |
| C-dense | B | 926,464 | 4.55e+11 | 4.1e+11 | 285,184 | 1.4e+11 | 1.26e+11 |
| C16-in @24,576 | B | 545,536 | not reached | not reached | 277,000 | not reached | not reached |
| C16-in @end | B | 545,536 | 1.42e+12 | not reached | 277,000 | 7.22e+11 | not reached |
| F10 | S | 930,560 | 7.93e+11 | 9e+11 |  |  |  |
| in16 | S | 549,632 | not reached | not reached |  |  |  |
| C-dense | S | 926,464 | 7.74e+11 | 7.59e+11 | 285,184 | 2.38e+11 | 2.34e+11 |
| C16-in @24,576 | S | 545,536 | not reached | not reached | 277,000 | not reached | not reached |
| C16-in @end | S | 545,536 | not reached | not reached | 277,000 | not reached | not reached |

### Per run: the levels (mean of the last three scored rows), and R at the end against 2,048 updates earlier

| Setting | Rule | A | R | length at 40 | length at 12 | taught | R: first answer byte right | R: second answer byte right | R: last three minus the three ending 2,048 earlier |
|---|---|---|---|---|---|---|---|---|---|
| F10 | B | 93.22 | 94.08 | 92.56 | 92.89 | 98.39 | (not recorded) | (not recorded) | +2.07 |
| in16 | B | 85.22 | 82.53 | 66.11 | 85.33 | 93.61 | (not recorded) | (not recorded) | +0.65 |
| C-dense | B | 97.67 | 97.08 | 92.67 | 97.33 | 99.06 | 98.13 | 97.63 | +1.08 |
| C16-in @24,576 | B | 67.94 | 62.12 | 41.56 | 72.78 | 85.28 | 77.13 | 78.03 | +2.73 |
| C16-in @end | B | 83.78 | 79.42 | 56.22 | 84.33 | 96.33 | 84.23 | 92.58 | +2.60 |
| F10 | S | 91.22 | 89.27 | 89.11 | 90.11 | 97.78 | (not recorded) | (not recorded) | +0.68 |
| in16 | S | 1.89 | 1.37 | 1.22 | 1.89 | 2.50 | (not recorded) | (not recorded) | -0.35 |
| C-dense | S | 93.50 | 94.10 | 90.56 | 94.00 | 99.22 | 96.03 | 94.30 | +3.02 |
| C16-in @24,576 | S | 32.72 | 28.97 | 9.56 | 36.44 | 54.61 | 63.10 | 39.57 | +1.65 |
| C16-in @end | S | 49.50 | 43.63 | 13.56 | 49.44 | 74.83 | 71.45 | 53.98 | +1.32 |

### Per run: A, R every 2,048 updates, and for this job's runs the share of R's items with the first and with the second answer byte right

| Setting | Rule | 2,048 | 4,096 | 6,144 | 8,192 | 10,240 | 12,288 | 14,336 | 16,384 | 18,432 | 20,480 | 22,528 | 24,576 | 26,624 | 28,672 | 30,720 | 32,768 | 34,816 | 36,864 | 38,912 | 40,960 | 43,008 | 45,056 | 47,104 | 49,152 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| F10 | B | 1.2, 1.5 | 17.7, 12.9 | 51.8, 46.0 | 80.0, 76.0 | 82.8, 82.3 | 86.5, 83.3 | 84.8, 83.7 | 90.3, 89.4 | 85.8, 86.8 | 94.2, 91.6 | 94.2, 92.7 | 91.8, 93.3 |  |  |  |  |  |  |  |  |  |  |  |  |
| in16 | B | 1.2, 1.5 | 2.7, 3.0 | 9.8, 6.7 | 26.7, 22.6 | 43.0, 36.1 | 63.8, 56.8 | 69.3, 62.2 | 78.8, 72.7 | 81.5, 77.7 | 86.8, 82.0 | 85.5, 81.0 | 87.7, 83.2 |  |  |  |  |  |  |  |  |  |  |  |  |
| C-dense | B | 5.7, 5.0 (42, 13) | 49.5, 44.8 (75, 55) | 76.7, 79.1 (89, 85) | 83.2, 83.8 (89, 89) | 91.8, 90.7 (95, 93) | 91.8, 92.5 (96, 94) | 90.8, 90.5 (94, 94) | 93.3, 95.1 (98, 96) | 92.7, 94.4 (97, 96) | 94.7, 95.0 (97, 96) | 94.8, 96.1 (97, 97) | 98.7, 97.5 (98, 98) |  |  |  |  |  |  |  |  |  |  |  |  |
| C16-in | B | 1.2, 0.9 (9, 11) | 4.2, 3.4 (20, 15) | 12.0, 7.8 (40, 17) | 22.3, 16.9 (58, 27) | 29.0, 23.4 (62, 35) | 40.8, 34.5 (65, 48) | 49.2, 41.8 (72, 55) | 59.3, 49.6 (76, 62) | 65.2, 50.9 (73, 67) | 65.8, 54.6 (73, 73) | 70.5, 60.5 (77, 77) | 63.7, 59.0 (74, 76) | 72.7, 63.2 (79, 76) | 78.2, 69.5 (82, 83) | 78.8, 72.4 (82, 86) | 74.2, 71.0 (82, 85) | 79.0, 74.7 (83, 88) | 79.7, 74.6 (85, 87) | 79.5, 75.2 (84, 88) | 83.2, 74.2 (80, 90) | 84.0, 78.0 (85, 91) | 80.8, 75.7 (82, 90) | 84.0, 79.0 (84, 92) | 84.2, 77.7 (83, 91) |
| F10 | S | 1.3, 1.1 | 11.5, 7.2 | 32.0, 25.4 | 59.7, 46.5 | 67.7, 56.5 | 78.0, 70.9 | 83.3, 80.5 | 88.3, 86.0 | 88.8, 87.1 | 90.7, 88.0 | 88.8, 88.5 | 92.2, 88.5 |  |  |  |  |  |  |  |  |  |  |  |  |
| in16 | S | 0.7, 0.7 | 0.7, 1.1 | 2.0, 0.8 | 1.7, 0.8 | 1.3, 1.1 | 2.7, 1.9 | 1.7, 1.5 | 2.0, 1.9 | 1.8, 1.7 | 1.7, 2.2 | 2.7, 1.9 | 2.2, 1.3 |  |  |  |  |  |  |  |  |  |  |  |  |
| C-dense | S | 4.0, 3.1 (25, 10) | 18.2, 16.8 (60, 26) | 34.8, 34.6 (72, 44) | 54.8, 58.2 (81, 67) | 64.8, 64.6 (83, 73) | 78.3, 81.9 (88, 84) | 84.8, 87.6 (90, 88) | 87.5, 90.0 (92, 91) | 89.3, 91.8 (94, 92) | 87.0, 90.0 (92, 90) | 90.7, 92.2 (95, 93) | 92.3, 93.7 (95, 94) |  |  |  |  |  |  |  |  |  |  |  |  |
| C16-in | S | 0.7, 0.9 (10, 9) | 1.3, 1.5 (14, 12) | 5.8, 4.2 (21, 15) | 9.7, 8.6 (37, 20) | 16.2, 10.2 (45, 21) | 17.8, 16.4 (50, 27) | 21.3, 19.4 (57, 29) | 26.3, 23.7 (60, 34) | 29.0, 24.3 (60, 36) | 29.3, 23.6 (57, 36) | 31.8, 27.6 (62, 38) | 32.7, 28.0 (60, 40) | 33.5, 31.4 (66, 40) | 38.5, 31.8 (63, 42) | 39.5, 35.7 (70, 45) | 41.8, 35.8 (70, 46) | 39.2, 37.4 (68, 48) | 40.7, 38.3 (70, 49) | 42.2, 38.9 (70, 49) | 41.5, 37.3 (69, 47) | 42.8, 39.5 (69, 50) | 46.7, 43.5 (72, 53) | 51.0, 43.6 (73, 53) | 48.7, 44.8 (72, 55) |

### Per run: max, min, add, sub at 12 letters every 4,096 updates

| Setting | Rule | 4,096 | 8,192 | 12,288 | 16,384 | 20,480 | 24,576 | 28,672 | 32,768 | 36,864 | 40,960 | 45,056 | 49,152 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| F10 | B | 92, 93, 19, 17 | 96, 94, 79, 81 | 96, 97, 86, 87 | 98, 98, 91, 90 | 98, 97, 94, 94 | 98, 98, 91, 92 |  |  |  |  |  |  |
| in16 | B | 71, 70, 3, 2 | 92, 92, 33, 21 | 96, 95, 71, 57 | 98, 96, 81, 76 | 96, 98, 86, 87 | 97, 99, 87, 88 |  |  |  |  |  |  |
| C-dense | B | 95, 96, 54, 45 | 95, 94, 83, 83 | 94, 92, 92, 92 | 98, 98, 94, 92 | 97, 98, 94, 95 | 98, 97, 98, 99 |  |  |  |  |  |  |
| C16-in | B | 66, 63, 5, 4 | 89, 88, 27, 18 | 93, 91, 43, 38 | 92, 93, 64, 55 | 95, 95, 69, 63 | 95, 96, 68, 59 | 96, 94, 82, 75 | 96, 96, 74, 74 | 96, 96, 82, 77 | 97, 96, 85, 82 | 97, 97, 82, 80 | 98, 96, 84, 84 |
| F10 | S | 91, 87, 12, 11 | 97, 95, 69, 50 | 97, 95, 83, 73 | 99, 97, 90, 87 | 98, 97, 90, 91 | 99, 97, 92, 93 |  |  |  |  |  |  |
| in16 | S | 42, 39, 1, 0 | 46, 37, 2, 1 | 42, 36, 3, 2 | 41, 40, 2, 2 | 43, 35, 1, 2 | 39, 43, 3, 2 |  |  |  |  |  |  |
| C-dense | S | 95, 93, 21, 15 | 96, 96, 60, 50 | 94, 91, 83, 74 | 98, 97, 91, 84 | 98, 96, 88, 86 | 99, 96, 91, 94 |  |  |  |  |  |  |
| C16-in | S | 46, 45, 1, 1 | 86, 87, 12, 7 | 91, 92, 21, 15 | 95, 94, 30, 23 | 96, 92, 31, 27 | 96, 94, 38, 28 | 96, 96, 39, 38 | 96, 97, 41, 43 | 96, 93, 43, 38 | 95, 93, 44, 39 | 98, 96, 47, 47 | 98, 97, 46, 51 |

### Per run: mean training loss of the 256 updates before the row

| Setting | Rule | 256 | 2,048 | 4,096 | 8,192 | 12,288 | 16,384 | 20,480 | 24,576 | 32,768 | 40,960 | 49,152 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| F10 | B | 1.604 | 0.832 | 0.486 | 0.161 | 0.099 | 0.073 | 0.057 | 0.045 |  |  |  |
| in16 | B | 1.675 | 1.057 | 0.842 | 0.427 | 0.219 | 0.140 | 0.107 | 0.084 |  |  |  |
| C-dense | B | 1.562 | 0.660 | 0.250 | 0.087 | 0.057 | 0.043 | 0.036 | 0.029 |  |  |  |
| C16-in | B | 1.648 | 1.019 | 0.833 | 0.474 | 0.303 | 0.223 | 0.184 | 0.155 | 0.123 | 0.101 | 0.085 |
| F10 | S | 1.609 | 0.904 | 0.624 | 0.250 | 0.127 | 0.074 | 0.058 | 0.047 |  |  |  |
| in16 | S | 1.685 | 1.125 | 1.040 | 1.017 | 1.014 | 1.020 | 1.040 | 1.049 |  |  |  |
| C-dense | S | 1.571 | 0.734 | 0.453 | 0.196 | 0.087 | 0.052 | 0.038 | 0.029 |  |  |  |
| C16-in | S | 1.660 | 1.044 | 0.953 | 0.638 | 0.474 | 0.396 | 0.354 | 0.321 | 0.275 | 0.251 | 0.223 |

### This job's runs: cosine between the sweep's and the exact gradient of each map (first batch; at the start and at the end of the reading), the mean magnitude of the kept weights, and a cell's input for a needed byte (root mean square, without the bias)

| Setting | Rule | input map: cosine start, end | lm: cosine start, end | recurrent: cosine start, end | input map: magnitude start, end | lm: magnitude start, end | recurrent: magnitude start, end | input for a needed byte: start, end |
|---|---|---|---|---|---|---|---|---|
| C-dense | B | 0.757, 0.650 | 0.867, 0.711 | 0.723, 0.639 | 0.0313, 0.0938 | 0.0312, 0.0617 | 0.1509, 0.1436 | 0.389, 5.140 |
| C16-in @24,576 | B | 0.709, 0.325 | 0.866, 0.633 | 0.755, 0.477 | 0.1765, 0.1359 | 0.0312, 0.0773 | 0.1509, 0.2461 | 0.381, 0.742 |
| C16-in @end | B | 0.709, 0.104 | 0.866, 0.664 | 0.755, 0.293 | 0.1765, 0.1398 | 0.0312, 0.0733 | 0.1509, 0.2535 | 0.381, 0.765 |
| C-dense | S | 0.757, 0.721 | 0.867, 0.875 | 0.723, 0.698 | 0.0313, 0.0862 | 0.0312, 0.0698 | 0.1509, 0.1517 | 0.389, 4.769 |
| C16-in @24,576 | S | 0.709, 0.654 | 0.866, 0.689 | 0.755, 0.698 | 0.1765, 0.1107 | 0.0312, 0.0849 | 0.1509, 0.2191 | 0.381, 0.587 |
| C16-in @end | S | 0.709, 0.348 | 0.866, 0.669 | 0.755, 0.431 | 0.1765, 0.1165 | 0.0312, 0.0841 | 0.1509, 0.2301 | 0.381, 0.629 |

### This job's runs: by what a cell knows. Mean magnitude of the kept recurrent weights into the cells that, under 16 values a cell, have input for 0 to 2, for 3 or 4 and for 5 or more of the ten digits (95, 131 and 30 cells); neighbours only [with the cell's reading of itself]. The grouping means nothing in `C-dense` and `C46-one`. The figure describes the size of weights and is not read as use

| Setting | Rule | 0 to 2 digits: start, end | 3 or 4: start, end | 5 or more: start, end |
|---|---|---|---|---|
| C-dense | B | 0.1498, 0.1467 [0.1500, 0.1448] | 0.1515, 0.1440 [0.1514, 0.1430] | 0.1501, 0.1448 [0.1511, 0.1425] |
| C16-in @24,576 | B | 0.1498, 0.2556 [0.1500, 0.2582] | 0.1515, 0.2328 [0.1514, 0.2401] | 0.1501, 0.2268 [0.1511, 0.2339] |
| C16-in @end | B | 0.1498, 0.2652 [0.1500, 0.2671] | 0.1515, 0.2403 [0.1514, 0.2471] | 0.1501, 0.2328 [0.1511, 0.2383] |
| C-dense | S | 0.1498, 0.1501 [0.1500, 0.1512] | 0.1515, 0.1518 [0.1514, 0.1523] | 0.1501, 0.1491 [0.1511, 0.1510] |
| C16-in @24,576 | S | 0.1498, 0.2130 [0.1500, 0.2207] | 0.1515, 0.2087 [0.1514, 0.2187] | 0.1501, 0.2025 [0.1511, 0.2152] |
| C16-in @end | S | 0.1498, 0.2260 [0.1500, 0.2330] | 0.1515, 0.2191 [0.1514, 0.2288] | 0.1501, 0.2132 [0.1511, 0.2266] |

### Descriptions fixed beforehand. * where the single-row version describes the cell differently

| Candidate | Reference | Rule | Measure | Description | Reference | Candidate | Ratio |
|---|---|---|---|---|---|---|---|
| C-dense | F10 | B | A80 updates | about the same | 8,704 | 7,680 | 0.88 |
| C-dense | F10 | B | A80 multiply-adds | stays | 5.18e+11 | 4.55e+11 | 0.88 |
| C-dense | F10 | B | R80 updates | about the same | 9,472 | 6,912 | 0.73 |
| C-dense | F10 | B | R80 multiply-adds | stays | 5.64e+11 | 4.1e+11 | 0.73 |
| C-dense | F10 | B | level of A | about the same | 93.22 | 97.67 | |
| C-dense | F10 | B | level of R | about the same | 94.08 | 97.08 | |
| C-dense | F10 | B | level of the length test at 40 | about the same | 92.56 | 92.67 | |
| in16 | F10 | B | A80 updates | more | 8,704 | 16,640 | 1.91 |
| in16 | F10 | B | A80 multiply-adds | stays | 5.18e+11 | 5.85e+11 | 1.13 |
| in16 | F10 | B | R80 updates | more | 9,472 | 20,480 | 2.16 |
| in16 | F10 | B | R80 multiply-adds | stays | 5.64e+11 | 7.2e+11 | 1.28 |
| in16 | F10 | B | level of A | about the same | 93.22 | 85.22 | |
| in16 | F10 | B | level of R | about the same | 94.08 | 82.53 | |
| in16 | F10 | B | level of the length test at 40 | lower | 92.56 | 66.11 | |
| C16-in @24,576 | C-dense | B | A80 updates | more | 7,680 | not reached |  |
| C16-in @24,576 | C-dense | B | A80 multiply-adds | rises | 4.55e+11 | not reached |  |
| C16-in @24,576 | C-dense | B | A80 multiply-adds, second count | rises | 1.4e+11 | not reached |  |
| C16-in @24,576 | C-dense | B | R80 updates | more | 6,912 | not reached |  |
| C16-in @24,576 | C-dense | B | R80 multiply-adds | rises | 4.1e+11 | not reached |  |
| C16-in @24,576 | C-dense | B | R80 multiply-adds, second count | rises | 1.26e+11 | not reached |  |
| C16-in @24,576 | C-dense | B | level of A | lower | 97.67 | 67.94 | |
| C16-in @24,576 | C-dense | B | level of R | lower | 97.08 | 62.12 | |
| C16-in @24,576 | C-dense | B | level of the length test at 40 | lower | 92.67 | 41.56 | |
| C16-in @end | C-dense | B | A80 updates | more | 7,680 | 40,704 | 5.30 |
| C16-in @end | C-dense | B | A80 multiply-adds | rises | 4.55e+11 | 1.42e+12 | 3.12 |
| C16-in @end | C-dense | B | A80 multiply-adds, second count | rises | 1.4e+11 | 7.22e+11 | 5.15 |
| C16-in @end | C-dense | B | R80 updates | more | 6,912 | not reached |  |
| C16-in @end | C-dense | B | R80 multiply-adds | rises | 4.1e+11 | not reached |  |
| C16-in @end | C-dense | B | R80 multiply-adds, second count | rises | 1.26e+11 | not reached |  |
| C16-in @end | C-dense | B | level of A | about the same | 97.67 | 83.78 | |
| C16-in @end | C-dense | B | level of R | lower | 97.08 | 79.42 | |
| C16-in @end | C-dense | B | level of the length test at 40 | lower | 92.67 | 56.22 | |
| C-dense | F10 | S | A80 updates | about the same | 13,312 | 13,056 | 0.98 |
| C-dense | F10 | S | A80 multiply-adds | stays | 7.93e+11 | 7.74e+11 | 0.98 |
| C-dense | F10 | S | R80 updates | about the same | 15,104 | 12,800 | 0.85 |
| C-dense | F10 | S | R80 multiply-adds | stays | 9e+11 | 7.59e+11 | 0.84 |
| C-dense | F10 | S | level of A | about the same | 91.22 | 93.50 | |
| C-dense | F10 | S | level of R | about the same | 89.27 | 94.10 | |
| C-dense | F10 | S | level of the length test at 40 | about the same | 89.11 | 90.56 | |
| in16 | F10 | S | A80 updates | more | 13,312 | not reached |  |
| in16 | F10 | S | A80 multiply-adds | not decided | 7.93e+11 | not reached |  |
| in16 | F10 | S | R80 updates | more | 15,104 | not reached |  |
| in16 | F10 | S | R80 multiply-adds | not decided | 9e+11 | not reached |  |
| in16 | F10 | S | level of A | lower | 91.22 | 1.89 | |
| in16 | F10 | S | level of R | lower | 89.27 | 1.37 | |
| in16 | F10 | S | level of the length test at 40 | lower | 89.11 | 1.22 | |
| C16-in @24,576 | C-dense | S | A80 updates | more | 13,056 | not reached |  |
| C16-in @24,576 | C-dense | S | A80 multiply-adds | not decided | 7.74e+11 | not reached |  |
| C16-in @24,576 | C-dense | S | A80 multiply-adds, second count | rises | 2.38e+11 | not reached |  |
| C16-in @24,576 | C-dense | S | R80 updates | more | 12,800 | not reached |  |
| C16-in @24,576 | C-dense | S | R80 multiply-adds | not decided | 7.59e+11 | not reached |  |
| C16-in @24,576 | C-dense | S | R80 multiply-adds, second count | rises | 2.34e+11 | not reached |  |
| C16-in @24,576 | C-dense | S | level of A | lower | 93.50 | 32.72 | |
| C16-in @24,576 | C-dense | S | level of R | lower | 94.10 | 28.97 | |
| C16-in @24,576 | C-dense | S | level of the length test at 40 | lower | 90.56 | 9.56 | |
| C16-in @end | C-dense | S | A80 updates | more | 13,056 | not reached |  |
| C16-in @end | C-dense | S | A80 multiply-adds | rises | 7.74e+11 | not reached |  |
| C16-in @end | C-dense | S | A80 multiply-adds, second count | rises | 2.38e+11 | not reached |  |
| C16-in @end | C-dense | S | R80 updates | more | 12,800 | not reached |  |
| C16-in @end | C-dense | S | R80 multiply-adds | rises | 7.59e+11 | not reached |  |
| C16-in @end | C-dense | S | R80 multiply-adds, second count | rises | 2.34e+11 | not reached |  |
| C16-in @end | C-dense | S | level of A | lower | 93.50 | 49.50 | |
| C16-in @end | C-dense | S | level of R | lower | 94.10 | 43.63 | |
| C16-in @end | C-dense | S | level of the length test at 40 | lower | 90.56 | 13.56 | |

### The brief's rules for stages 2 and 3

- B, stage 2 (`C16-in`): **runs**: `C-dense` reaches 80 on R at 6,912 updates
- B, stage 3 (`C46-one`): **runs**: `C16-in` does not reach 80 on R within its dose
- S, stage 2 (`C16-in`): **runs**: `C-dense` reaches 80 on R at 12,800 updates
- S, stage 3 (`C46-one`): **runs**: `C16-in` does not reach 80 on R within its dose

