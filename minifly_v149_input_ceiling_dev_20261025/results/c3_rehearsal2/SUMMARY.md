# V149 summary: c3_rehearsal2 (edgeworth; taught/calibration/held-out KB [40.0, 5.0, 10.0])

Counting ladder, held-out bits/symbol: uniform 8.006, unigram 4.486, bigram 3.568, ppmc2 3.062, ppmc3 2.748, ppmc4 2.691, ppmc5 2.709

## Readout checks (R1 on evaluator-side codes)

| Code | held bits/symbol | L2 | check |
|---|---:|---:|---|
| onehot_last1 | 3.571 | 1 | PASS |
| onehot_last2 | 2.994 | 1 | PASS |
| const | 4.481 | 1000 | PASS |

## Arms (native KC states of an untaught memory)

| Arm | R1 held | R1 acc | R1 L2 | R2 held | R2 acc | R2 mix |
|---|---:|---:|---:|---:|---:|---:|
| PERM-obs | 3.913 | 0.267 | 30 | 4.320 | 0.179 | 0.003 |
| PERM-x | 3.880 | 0.272 | 100 | 4.329 | 0.179 | 0.003 |
| FE0-x | 3.369 | 0.366 | 30 | 4.199 | 0.192 | 0.003 |
| PERM-last | 3.560 | 0.289 | 30 | 3.782 | 0.276 | 0.001 |

Line-bootstrap SDs: PERM-obs_R1_minus_bigram 0.017, PERM-x_R1_minus_PERM-last_R1 0.015, PERM-obs_R2_minus_R1 0.016

CPU: states 278.1 s, total 4795.7 s; peak memory 1404.6 MB.
