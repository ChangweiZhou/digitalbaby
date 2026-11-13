# V149 summary: c3_rehearsal (bryant; taught/calibration/held-out KB [40.0, 5.0, 10.0])

Counting ladder, held-out bits/symbol: uniform 8.006, unigram 4.529, bigram 3.712, ppmc2 3.359, ppmc3 3.198, ppmc4 3.191, ppmc5 3.230

## Readout checks (R1 on evaluator-side codes)

| Code | held bits/symbol | L2 | check |
|---|---:|---:|---|
| onehot_last1 | 3.702 | 1 | PASS |
| onehot_last2 | 3.264 | 1 | PASS |
| const | 4.523 | 100 | PASS |

## Arms (native KC states of an untaught memory)

| Arm | R1 held | R1 acc | R1 L2 | R2 held | R2 acc | R2 mix |
|---|---:|---:|---:|---:|---:|---:|
| PERM-obs | 4.080 | 0.247 | 30 | 4.383 | 0.173 | 0.003 |
| PERM-x | 4.019 | 0.254 | 100 | 4.388 | 0.172 | 0.01 |
| FE0-x | 3.657 | 0.326 | 30 | 4.269 | 0.181 | 0.003 |
| PERM-last | 3.722 | 0.276 | 10 | 3.888 | 0.266 | 0.003 |

Line-bootstrap SDs: PERM-obs_R1_minus_bigram 0.018, PERM-x_R1_minus_PERM-last_R1 0.017, PERM-obs_R2_minus_R1 0.021

CPU: states 272.3 s, total 3397.6 s; peak memory 1385.7 MB.
