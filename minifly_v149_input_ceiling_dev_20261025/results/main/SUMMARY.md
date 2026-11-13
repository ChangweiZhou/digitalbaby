# V149 summary: main (brown; taught/calibration/held-out KB [40, 5, 10])

Counting ladder, held-out bits/symbol: uniform 8.006, unigram 4.376, bigram 3.553, ppmc2 3.052, ppmc3 2.802, ppmc4 2.762, ppmc5 2.783

## Readout checks (R1 on evaluator-side codes)

| Code | held bits/symbol | L2 | check |
|---|---:|---:|---|
| onehot_last1 | 3.555 | 1 | PASS |
| onehot_last2 | 2.982 | 1 | PASS |
| const | 4.370 | 3 | PASS |

## Arms (native KC states of an untaught memory)

| Arm | R1 held | R1 acc | R1 L2 | R2 held | R2 acc | R2 mix |
|---|---:|---:|---:|---:|---:|---:|
| PERM-obs | 3.767 | 0.281 | 100 | 4.234 | 0.182 | 0.001 |
| PERM-x | 3.757 | 0.278 | 100 | 4.241 | 0.182 | 0.001 |
| FE0-x | 3.379 | 0.340 | 30 | 4.137 | 0.187 | 0.001 |
| PERM-last | 3.554 | 0.283 | 10 | 3.757 | 0.241 | 0.001 |

Line-bootstrap SDs: PERM-obs_R1_minus_bigram 0.015, PERM-x_R1_minus_PERM-last_R1 0.016, PERM-obs_R2_minus_R1 0.012

CPU: states 269.4 s, total 4052.2 s; peak memory 1400.5 MB.

## Decision and predictions (design sections 6-7; fixed in LOCK.json before the run)
* readout_checks_all_pass: True
* C_PERM-obs_R1: 3.7668707622632627
* B_bigram: 3.553026976797097
* unigram: 4.375551304633449
* decision: CONTEXT_FIRST
* P1: AS PREDICTED
* P2: AS PREDICTED
* P2_values: {'PERM-last_R1': 3.5535242337082744, 'PERM-x_R1': 3.756549370513998, 'bigram': 3.553026976797097}
* P3: AS PREDICTED
* P3_values: {'PERM-obs_R2': 4.233932730379069, 'PERM-obs_R1': 3.7668707622632627}
