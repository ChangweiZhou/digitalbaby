# V153 summary: rehearsal / c3 (thursday, KB [40.0, 5.0, 10.0])

Ladder (held bits/symbol): uniform 8.006, unigram 4.445, bigram 3.538, ppmc2 2.986, ppmc3 2.667, ppmc4 2.606, ppmc5 2.616

## Block A (R1 on binary KC codes)

| Cell | held | acc | L2 |
|---|---:|---:|---:|
| P95 | 3.833 | 0.267 | 100 |
| P81 | 2.963 | 0.426 | 30 |
| N95 | 4.113 | 0.213 | 100 |
| N81 | 3.321 | 0.357 | 30 |

Bootstrap SDs: P81-N81 0.015, P95-N95 0.014, P95-P81 0.019, N95-N81 0.019

## Block B (FE0 observed KC state)

| Arm | held | acc |
|---|---:|---:|
| R1 | 3.271 | 0.362 |
| ONLINE eta 0.01 pass 1 | 3.537 | 0.322 |
| ONLINE eta 0.01 pass 3 | 3.424 | 0.334 |
| C16 | 5.653 | 0.268 |
| C16Q | 3.528 | 0.344 |
| NATIVE | 6.436 | 0.191 |
| R2 | 4.102 | 0.206 |

Decomposition: online_cost 0.153, port_cost 2.382, bias_recovery 2.125, native_structure_cost 0.783
Bootstrap SDs: online_cost 0.008, port_cost 0.021, native_structure_cost 0.024

## Predictions and decision (rules of design sections 5-6)
* P1: {'result': 'WEAKENED (scrambling effect above 0.10 at some fade)', 'scrambling_effect': {'fade_0.807': -0.3585386066520866, 'fade_0.95': -0.2801081548497786}, 'fade_effect': {'scrambled': 0.8697578115186082, 'unscrambled': 0.7913273597163002}}
* best_cell: P81
* P2: {'result': 'PARTLY (gap after the last pass between 0.15 and 0.30, or first-pass gap above 0.40)', 'gap_pass1': 0.2656715626257662, 'gap_pass3': 0.15278483066662574}
* P3: {'result': 'AS PREDICTED', 'port_cost': 2.38172719036307}
* P4: {'result': 'AS PREDICTED', 'native_structure_cost': 0.7833502733141477}
* decision: {'costs': {'online': 0.15278483066662574, 'port': 2.38172719036307, 'native_structure': 0.7833502733141477}, 'next_round_targets': 'port'}

CPU 4887.6 s; learner records 2820; peak memory 1640.2 MB.
