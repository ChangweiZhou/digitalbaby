# V149 summary: c2_dev (buster; taught/calibration/held-out KB [40.0, 5.0, 10.0])

Counting ladder, held-out bits/symbol: uniform 8.006, unigram 4.497, bigram 3.510, ppmc2 2.870, ppmc3 2.527, ppmc4 2.450, ppmc5 2.454

## Readout checks (R1 on evaluator-side codes)

| Code | held bits/symbol | L2 | check |
|---|---:|---:|---|
| onehot_last1 | 3.524 | 1 | PASS |
| onehot_last2 | 2.834 | 1 | PASS |
| const | 4.492 | 300 | PASS |

## Arms (native KC states of an untaught memory)

| Arm | R1 held | R1 acc | R1 L2 | R2 held | R2 acc | R2 mix |
|---|---:|---:|---:|---:|---:|---:|
| PERM-obs | 3.718 | 0.300 | 30 | 4.313 | 0.184 | 0.01 |
| PERM-x | 3.764 | 0.296 | 30 | 4.323 | 0.183 | 0.01 |
| FE0-x | 3.184 | 0.407 | 10 | 4.195 | 0.193 | 0.01 |
| PERM-last | 3.513 | 0.280 | 10 | 3.774 | 0.266 | 0.01 |

Line-bootstrap SDs: PERM-obs_R1_minus_bigram 0.024, PERM-x_R1_minus_PERM-last_R1 0.026, PERM-obs_R2_minus_R1 0.026

CPU: states 298.4 s, total 4629.1 s; peak memory 1415.4 MB.
