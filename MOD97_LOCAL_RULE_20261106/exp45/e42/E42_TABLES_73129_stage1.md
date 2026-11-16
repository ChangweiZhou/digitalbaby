Receipts read from `exp45/e42`; seeds requested 73129; complete runs 8.

### Batch digest at row 4,096 (check 2) and the limit at every scored row (check 4)

| Seed | runs | digests equal | digest (first 16) | limit held at every row |
|---|---|---|---|---|
| 73129 | 8 | yes | bbd71ab7ca27b72e | yes |

### Check 11: G's final tensors_sha256 beside E26's receipt (reported, not a failure)

| Rule | Seed | G | E26 | equal |
|---|---|---|---|---|
| S | 73129 | d906412631d18946 | d906412631d18946 | True |
| B | 73129 | efbdf87a6e1e2aeb | efbdf87a6e1e2aeb | True |

### Seed 73129: per run (three-row version; single-row in brackets)

| Config | Rule | A50 | A80 | A90 | R50 | R80 | R90 | A80 weight-items | max A (row) | max R (row) | final A, R, taught | len 12, 40 (last three rows) | replaced | cosine start, end | CPU-h updates + scoring |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| G | S | 3,840 [3,840] | 5,632 [5,632] | 12,800 [7,936] | 3,840 [3,840] | 6,656 [6,656] | 13,824 [12,288] | 4.05e+11 | 98.17 (24,064) | 97.05 (24,320) | 98.00, 95.85, 99.33 | 98.44, 96.44 | 0 | 0.741, 0.745 | 0.70 + 0.15 |
| G | B | 4,096 [4,096] | 5,120 [5,120] | 10,240 [7,936] | 3,840 [3,840] | 6,400 [5,120] | 7,936 [7,936] | 3.68e+11 | 98.50 (24,576) | 98.15 (21,248) | 98.50, 98.15, 98.67 | 96.89, 94.56 | 0 | 0.741, 0.689 | 0.67 + 0.15 |
| F10 | S | 8,192 [7,168] | 13,312 [12,800] | 20,480 [19,456] | 9,472 [8,960] | 15,104 [13,824] | 23,296 [19,456] | 7.97e+11 | 92.17 (24,576) | 92.30 (24,064) | 92.17, 88.45, 98.17 | 90.11, 89.11 | 0 | 0.744, 0.604 | 0.72 + 0.16 |
| F10 | B | 5,632 [5,632] | 8,704 [8,192] | 15,360 [15,360] | 6,400 [6,400] | 9,472 [8,960] | 19,200 [15,616] | 5.21e+11 | 96.67 (23,296) | 95.30 (24,064) | 91.83, 93.30, 97.50 | 92.89, 92.56 | 0 | 0.744, 0.631 | 0.70 + 0.16 |
| A10 | S | 8,192 [7,680] | 13,824 [13,824] | 22,528 [17,408] | 9,728 [9,728] | 16,128 [16,128] | not reached [23,808] | 8.28e+11 | 93.00 (23,040) | 90.85 (23,808) | 92.83, 86.15, 98.83 | 92.67, 91.78 | 35,833 | 0.744, 0.640 | 0.70 + 0.15 |
| A10 | B | 5,888 [5,888] | 9,472 [8,192] | 20,992 [15,616] | 6,400 [6,400] | 9,472 [9,472] | 19,456 [16,384] | 5.67e+11 | 93.33 (24,064) | 94.35 (24,064) | 93.17, 92.85, 97.83 | 92.78, 92.22 | 37,525 | 0.744, 0.563 | 0.68 + 0.16 |
| P8 | S | 5,376 [5,376] | 7,680 [7,680] | 10,496 [9,984] | 6,400 [6,400] | 8,448 [8,448] | 12,288 [12,288] | 4.6e+11 | 97.00 (22,784) | 94.95 (24,064) | 95.67, 92.50, 99.17 | 96.33, 96.00 | 38,713 | 0.727, 0.675 | 0.94 + 0.25 |
| P8 | B | 4,864 [4,864] | 6,400 [6,400] | 8,192 [8,192] | 5,120 [5,120] | 7,168 [7,168] | 8,960 [8,448] | 3.84e+11 | 98.67 (23,552) | 97.85 (19,712) | 98.00, 96.60, 99.50 | 97.00, 96.56 | 39,903 | 0.727, 0.739 | 0.92 + 0.25 |

### Descriptions (fixed beforehand). Updates and weight-items to the 80 threshold, three-row version (* where the single-row version describes the cell differently); the length test at 40 letters, mean of the last three rows

| Candidate | Reference | Rule | Measure | Cell | Per seed: description (reference, candidate) |
|---|---|---|---|---|---|
| F10 | G | S | A80 updates | (seeds missing) | 73129: more (5,632, 13,312) |
| F10 | G | S | A80 weight-items | (seeds missing) | 73129: more (4.05e+11, 7.97e+11) |
| F10 | G | S | R80 updates | (seeds missing) | 73129: more (6,656, 15,104) |
| F10 | G | S | R80 weight-items | (seeds missing) | 73129: more (4.79e+11, 9.05e+11) |
| F10 | G | S | length at 40 | (seeds missing) | 73129: about the same (96.44, 89.11) |
| F10 | G | B | A80 updates | (seeds missing) | 73129: more (5,120, 8,704) |
| F10 | G | B | A80 weight-items | (seeds missing) | 73129: about the same (3.68e+11, 5.21e+11) |
| F10 | G | B | R80 updates | (seeds missing)* | 73129: about the same (6,400, 9,472) |
| F10 | G | B | R80 weight-items | (seeds missing) | 73129: about the same (4.6e+11, 5.67e+11) |
| F10 | G | B | length at 40 | (seeds missing) | 73129: about the same (94.56, 92.56) |
| A10 | G | S | A80 updates | (seeds missing) | 73129: more (5,632, 13,824) |
| A10 | G | S | A80 weight-items | (seeds missing) | 73129: more (4.05e+11, 8.28e+11) |
| A10 | G | S | R80 updates | (seeds missing) | 73129: more (6,656, 16,128) |
| A10 | G | S | R80 weight-items | (seeds missing) | 73129: more (4.79e+11, 9.66e+11) |
| A10 | G | S | length at 40 | (seeds missing) | 73129: about the same (96.44, 91.78) |
| A10 | G | B | A80 updates | (seeds missing) | 73129: more (5,120, 9,472) |
| A10 | G | B | A80 weight-items | (seeds missing)* | 73129: more (3.68e+11, 5.67e+11) |
| A10 | G | B | R80 updates | (seeds missing)* | 73129: about the same (6,400, 9,472) |
| A10 | G | B | R80 weight-items | (seeds missing)* | 73129: about the same (4.6e+11, 5.67e+11) |
| A10 | G | B | length at 40 | (seeds missing) | 73129: about the same (94.56, 92.22) |
| P8 | G | S | A80 updates | (seeds missing) | 73129: about the same (5,632, 7,680) |
| P8 | G | S | A80 weight-items | (seeds missing) | 73129: about the same (4.05e+11, 4.6e+11) |
| P8 | G | S | R80 updates | (seeds missing) | 73129: about the same (6,656, 8,448) |
| P8 | G | S | R80 weight-items | (seeds missing) | 73129: about the same (4.79e+11, 5.06e+11) |
| P8 | G | S | length at 40 | (seeds missing) | 73129: about the same (96.44, 96.00) |
| P8 | G | B | A80 updates | (seeds missing) | 73129: about the same (5,120, 6,400) |
| P8 | G | B | A80 weight-items | (seeds missing) | 73129: about the same (3.68e+11, 3.84e+11) |
| P8 | G | B | R80 updates | (seeds missing) | 73129: about the same (6,400, 7,168) |
| P8 | G | B | R80 weight-items | (seeds missing) | 73129: about the same (4.6e+11, 4.3e+11) |
| P8 | G | B | length at 40 | (seeds missing) | 73129: about the same (94.56, 96.56) |
| A10 | F10 | S | A80 updates | (seeds missing) | 73129: about the same (13,312, 13,824) |
| A10 | F10 | S | A80 weight-items | (seeds missing) | 73129: about the same (7.97e+11, 8.28e+11) |
| A10 | F10 | S | R80 updates | (seeds missing) | 73129: about the same (15,104, 16,128) |
| A10 | F10 | S | R80 weight-items | (seeds missing) | 73129: about the same (9.05e+11, 9.66e+11) |
| A10 | F10 | S | length at 40 | (seeds missing) | 73129: about the same (89.11, 91.78) |
| A10 | F10 | B | A80 updates | (seeds missing) | 73129: about the same (8,704, 9,472) |
| A10 | F10 | B | A80 weight-items | (seeds missing) | 73129: about the same (5.21e+11, 5.67e+11) |
| A10 | F10 | B | R80 updates | (seeds missing) | 73129: about the same (9,472, 9,472) |
| A10 | F10 | B | R80 weight-items | (seeds missing) | 73129: about the same (5.67e+11, 5.67e+11) |
| A10 | F10 | B | length at 40 | (seeds missing) | 73129: about the same (92.56, 92.22) |
| P8 | A10 | S | A80 updates | (seeds missing) | 73129: fewer (13,824, 7,680) |
| P8 | A10 | S | A80 weight-items | (seeds missing) | 73129: fewer (8.28e+11, 4.6e+11) |
| P8 | A10 | S | R80 updates | (seeds missing) | 73129: fewer (16,128, 8,448) |
| P8 | A10 | S | R80 weight-items | (seeds missing) | 73129: fewer (9.66e+11, 5.06e+11) |
| P8 | A10 | S | length at 40 | (seeds missing) | 73129: about the same (91.78, 96.00) |
| P8 | A10 | B | A80 updates | (seeds missing) | 73129: about the same (9,472, 6,400) |
| P8 | A10 | B | A80 weight-items | (seeds missing) | 73129: about the same (5.67e+11, 3.84e+11) |
| P8 | A10 | B | R80 updates | (seeds missing) | 73129: about the same (9,472, 7,168) |
| P8 | A10 | B | R80 weight-items | (seeds missing) | 73129: about the same (5.67e+11, 4.3e+11) |
| P8 | A10 | B | length at 40 | (seeds missing) | 73129: about the same (92.22, 96.56) |
| G | G | S vs B | A80 updates | (seeds missing) | 73129: about the same (5,120, 5,632) |
| G | G | S vs B | A80 weight-items | (seeds missing) | 73129: about the same (3.68e+11, 4.05e+11) |
| G | G | S vs B | R80 updates | (seeds missing) | 73129: about the same (6,400, 6,656) |
| G | G | S vs B | R80 weight-items | (seeds missing) | 73129: about the same (4.6e+11, 4.79e+11) |
| G | G | S vs B | length at 40 | (seeds missing) | 73129: about the same (94.56, 96.44) |
| F10 | F10 | S vs B | A80 updates | (seeds missing) | 73129: more (8,704, 13,312) |
| F10 | F10 | S vs B | A80 weight-items | (seeds missing) | 73129: more (5.21e+11, 7.97e+11) |
| F10 | F10 | S vs B | R80 updates | (seeds missing) | 73129: more (9,472, 15,104) |
| F10 | F10 | S vs B | R80 weight-items | (seeds missing) | 73129: more (5.67e+11, 9.05e+11) |
| F10 | F10 | S vs B | length at 40 | (seeds missing) | 73129: about the same (92.56, 89.11) |
| A10 | A10 | S vs B | A80 updates | (seeds missing)* | 73129: about the same (9,472, 13,824) |
| A10 | A10 | S vs B | A80 weight-items | (seeds missing)* | 73129: about the same (5.67e+11, 8.28e+11) |
| A10 | A10 | S vs B | R80 updates | (seeds missing) | 73129: more (9,472, 16,128) |
| A10 | A10 | S vs B | R80 weight-items | (seeds missing) | 73129: more (5.67e+11, 9.66e+11) |
| A10 | A10 | S vs B | length at 40 | (seeds missing) | 73129: about the same (92.22, 91.78) |
| P8 | P8 | S vs B | A80 updates | (seeds missing) | 73129: about the same (6,400, 7,680) |
| P8 | P8 | S vs B | A80 weight-items | (seeds missing) | 73129: about the same (3.84e+11, 4.6e+11) |
| P8 | P8 | S vs B | R80 updates | (seeds missing) | 73129: about the same (7,168, 8,448) |
| P8 | P8 | S vs B | R80 weight-items | (seeds missing) | 73129: about the same (4.3e+11, 5.06e+11) |
| P8 | P8 | S vs B | length at 40 | (seeds missing) | 73129: about the same (96.56, 96.00) |

### Structure: share of connections never replaced; share of connections by the reader's and the source's period (reader<-source)

| Config | Rule | Seed | never replaced | by period at the start | by period at the end |
|---|---|---|---|---|---|
| A10 | B | 73129 | 0.533 | 1<-1 1.000 | 1<-1 1.000 |
| A10 | S | 73129 | 0.557 | 1<-1 1.000 | 1<-1 1.000 |
| F10 | B | 73129 | 1.000 | 1<-1 1.000 | 1<-1 1.000 |
| F10 | S | 73129 | 1.000 | 1<-1 1.000 | 1<-1 1.000 |
| G | B | 73129 | 1.000 | 1<-1 1.000 | 1<-1 1.000 |
| G | S | 73129 | 1.000 | 1<-1 1.000 | 1<-1 1.000 |
| P8 | B | 73129 | 0.459 | 1<-1 0.248, 1<-2 0.128, 1<-4 0.063, 1<-8 0.061, 2<-1 0.130, 2<-2 0.061, 2<-4 0.029, 2<-8 0.029, 4<-1 0.061, 4<-2 0.031, 4<-4 0.016, 4<-8 0.017, 8<-1 0.059, 8<-2 0.036, 8<-4 0.016, 8<-8 0.014 | 1<-1 0.249, 1<-2 0.115, 1<-4 0.069, 1<-8 0.068, 2<-1 0.137, 2<-2 0.055, 2<-4 0.025, 2<-8 0.034, 4<-1 0.062, 4<-2 0.030, 4<-4 0.018, 4<-8 0.016, 8<-1 0.070, 8<-2 0.032, 8<-4 0.012, 8<-8 0.011 |
| P8 | S | 73129 | 0.491 | 1<-1 0.248, 1<-2 0.128, 1<-4 0.063, 1<-8 0.061, 2<-1 0.130, 2<-2 0.061, 2<-4 0.029, 2<-8 0.029, 4<-1 0.061, 4<-2 0.031, 4<-4 0.016, 4<-8 0.017, 8<-1 0.059, 8<-2 0.036, 8<-4 0.016, 8<-8 0.014 | 1<-1 0.261, 1<-2 0.110, 1<-4 0.066, 1<-8 0.063, 2<-1 0.135, 2<-2 0.051, 2<-4 0.032, 2<-8 0.032, 4<-1 0.068, 4<-2 0.027, 4<-4 0.015, 4<-8 0.014, 8<-1 0.063, 8<-2 0.037, 8<-4 0.016, 8<-8 0.009 |

