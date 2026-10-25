# V153 summary: dev / c2 (buster, KB [10.0, 2.0, 4.0])

Ladder (held bits/symbol): uniform 8.006, unigram 4.468, bigram 3.539, ppmc2 3.129, ppmc3 3.006, ppmc4 2.993, ppmc5 3.018

## Block A (R1 on binary KC codes)

| Cell | held | acc | L2 |
|---|---:|---:|---:|
| P95 | 3.935 | 0.256 | 100 |
| P81 | 3.178 | 0.403 | 30 |
| N95 | 4.202 | 0.215 | 100 |
| N81 | 3.524 | 0.342 | 30 |

Bootstrap SDs: P81-N81 0.026, P95-N95 0.025, P95-P81 0.031, N95-N81 0.041

## Block B (FE0 observed KC state)

| Arm | held | acc |
|---|---:|---:|
| R1 | 3.486 | 0.351 |
| ONLINE eta 0.01 pass 1 | 3.731 | 0.317 |
| ONLINE eta 0.01 pass 3 | 3.595 | 0.340 |
| C16 | 6.084 | 0.265 |
| C16Q | 3.660 | 0.335 |
| NATIVE | 6.491 | 0.187 |
| R2 | 4.106 | 0.213 |

Decomposition: online_cost 0.109, port_cost 2.598, bias_recovery 2.425, native_structure_cost 0.407
Bootstrap SDs: online_cost 0.012, port_cost 0.053, native_structure_cost 0.066

## Predictions and decision (rules of design sections 5-6)
* P1: {'result': 'WEAKENED (scrambling effect above 0.10 at some fade)', 'scrambling_effect': {'fade_0.807': -0.34643734361748146, 'fade_0.95': -0.2667442264404398}, 'fade_effect': {'scrambled': 0.7569150476008581, 'unscrambled': 0.6772219304238165}}
* best_cell: P81
* P2: {'result': 'AS PREDICTED', 'gap_pass1': 0.24468299890370204, 'gap_pass3': 0.10860872319130621}
* P3: {'result': 'AS PREDICTED', 'port_cost': 2.5983361166838623}
* P4: {'result': 'WEAKENED (native-structure cost below 0.5)', 'native_structure_cost': 0.4070334725431959}
* decision: {'costs': {'online': 0.10860872319130621, 'port': 2.5983361166838623, 'native_structure': 0.4070334725431959}, 'next_round_targets': 'port'}

CPU 1329.4 s; learner records 1640; peak memory 609.2 MB.
