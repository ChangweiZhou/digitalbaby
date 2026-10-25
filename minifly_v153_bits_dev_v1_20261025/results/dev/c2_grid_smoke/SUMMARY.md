# V153 summary: dev / c2_grid_smoke (buster, KB [1.0, 0.5, 0.5])

Ladder (held bits/symbol): uniform 8.006, unigram 4.700, bigram 4.045, ppmc2 3.920, ppmc3 3.914, ppmc4 3.926, ppmc5 3.937

## Block A (R1 on binary KC codes)

| Cell | held | acc | L2 |
|---|---:|---:|---:|
| P95 | 4.318 | 0.192 | 100 |
| P81 | 3.984 | 0.289 | 30 |
| N95 | 4.449 | 0.184 | 100 |
| N81 | 4.077 | 0.268 | 30 |

Bootstrap SDs: P81-N81 0.042, P95-N95 0.046, P95-P81 0.070, N95-N81 0.092

## Block B (FE0 observed KC state)

| Arm | held | acc |
|---|---:|---:|
| R1 | 4.067 | 0.274 |
| ONLINE eta 0.01 pass 1 | 4.479 | 0.236 |
| ONLINE eta 0.01 pass 3 | 4.226 | 0.264 |
| C16 | 6.461 | 0.173 |
| C16Q | 4.181 | 0.279 |
| NATIVE | 6.988 | 0.143 |
| R2 | 4.308 | 0.240 |

Decomposition: online_cost 0.158, port_cost 2.394, bias_recovery 2.280, native_structure_cost 0.527
Bootstrap SDs: online_cost 0.028, port_cost 0.070, native_structure_cost 0.094

## Predictions and decision (rules of design sections 5-6)
* P1: {'result': 'WEAKENED (scrambling effect above 0.10 at some fade)', 'scrambling_effect': {'fade_0.807': -0.09320349043911458, 'fade_0.95': -0.1306991819379899}, 'fade_effect': {'scrambled': 0.33415152635376133, 'unscrambled': 0.37164721785263666}}
* best_cell: P81
* P2: {'result': 'PARTLY (gap after the last pass between 0.15 and 0.30, or first-pass gap above 0.40)', 'gap_pass1': 0.4123445002549957, 'gap_pass3': 0.1584598736479812}
* P3: {'result': 'AS PREDICTED', 'port_cost': 2.3942721908814333}
* P4: {'result': 'AS PREDICTED', 'native_structure_cost': 0.5266693282059638}
* decision: {'costs': {'online': 0.1584598736479812, 'port': 2.3942721908814333, 'native_structure': 0.5266693282059638}, 'next_round_targets': 'port'}

CPU 143.4 s; learner records 68; peak memory 346.0 MB.
