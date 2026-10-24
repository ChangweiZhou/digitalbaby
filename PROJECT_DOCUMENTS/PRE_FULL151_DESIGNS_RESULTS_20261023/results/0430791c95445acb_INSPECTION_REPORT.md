# BYTE-CORE-1 — inspection report

Generated 2026-09-22 22:54:49 EDT by `package_results.py`. Design and pre-launch amendments: `DESIGN.md`. Full analysis tables: `results/`.

## Run status

- Complete: **True**; stages seen: con, dev, blockE, analyze
- Jobs by namespace-block: {"DEV-D": 80, "CON-E": 256, "CON-C": 3072, "PIL-PIL": 192, "CAL-CAL": 256}
- Failure files: none
- Source lock re-verified at packaging: **True** 
- Analysis run by packager: {"ran": false, "reason": "complete analysis is newer than every job"}
- Qualification verdicts: {"Q10": true, "Q1_F151": true, "Q1_V50": true, "Q2": true, "Q3": true, "Q4": true, "Q5": true, "Q6": true, "Q7": true, "Q8": true, "Q9": true}

## Decisions (DESIGN.md §8)

- **D-B interface efficiency** ρ = η(FULL_151 + FE0) / η(FULL_151 + FE-ID oracle) = 0.0047 / 0.2557 = **0.018** → **FRONT_END_IS_BINDING**
- **D-C streaming vs episode**: NO_MATERIAL_STREAM_LOSS (mean difference -0.0047)
- **D-D feed density**: DENSITY_OPTIONAL {"E-1a": -0.015407907329448974, "E-1b": -0.0018021231386155313}
- **D-E clock**: CLOCK_INSENSITIVE {"E-2a": 0.0, "E-2b": -0.01195747532813132}
- Calibration grid edges: {}

## Milestone table (confirmation worlds; η = bits saved per relevant bit fed)

| arm | rung | fmt | control | n | η ± SE | per-cue acc. | paired acc. (native) | η NLMS same codes | η IDREF |
|---|---|---|---|---:|---|---:|---:|---:|---:|
| F151-E-FE0 | R2 | F1 | normal | 128 | 0.0800 ± 0.0016 | 0.6686 | 0.7704 | 0.3635 | 0.5644 |
| F151-E-FE0 | R2 | F2 | normal | 128 | 0.0061 ± 0.0002 | 0.5091 | 0.6620 | 0.3059 | 0.5644 |
| F151-E-FE0 | R2 | F3 | normal | 128 | 0.0047 ± 0.0002 | 0.5051 | 0.6434 | 0.0788 | 0.5644 |
| F151-E-FE0 | R2 | F3 | unpaired | 128 | -0.0054 ± 0.0001 | 0.5001 | 0.4918 | — | -0.1156 |
| F151-E-FE0-NOLEARN | R2 | F3 | normal | 128 | -0.0000 ± 0.0000 | 0.5000 | 0.4946 | — | 0.5644 |
| F151-E-FE0-USCUT | R2 | F3 | normal | 128 | -0.0037 ± 0.0001 | 0.5000 | 0.4977 | — | 0.5644 |
| F151-E-FEID | R2 | F1 | normal | 128 | 0.2557 ± 0.0030 | 0.7833 | 0.8367 | 0.4004 | 0.5644 |
| F151-E-FEID | R2 | F3 | normal | 128 | 0.2557 ± 0.0030 | 0.7833 | 0.8367 | 0.4004 | 0.5644 |
| F151-E-FETAIL4 | R2 | F3 | normal | 128 | 0.0771 ± 0.0015 | 0.6556 | 0.7739 | 0.3675 | 0.5644 |
| F151-S-FE0 | R2 | F1 | normal | 128 | 0.0000 ± 0.0000 | 0.5000 | 0.5038 | — | 0.5644 |
| F151-S-FE0 | R2 | F3 | normal | 128 | -0.0000 ± 0.0000 | 0.5000 | 0.4988 | — | 0.5644 |
| V50-NOLEARN | R2 | F3 | normal | 128 | 0.0000 ± 0.0000 | 0.5000 | 0.5000 | — | 0.5644 |
| V50-NOLEARN | R24 | F3 | normal | 128 | 0.0000 ± 0.0000 | 0.0411 | — | — | 0.5227 |
| V50-REF | R2 | F1 | normal | 128 | 0.4616 ± 0.0031 | 0.8477 | 0.8974 | 0.4260 | 0.5644 |
| V50-REF | R2 | F2 | normal | 128 | 0.3672 ± 0.0032 | 0.8306 | 0.8500 | 0.2473 | 0.5644 |
| V50-REF | R2 | F3 | normal | 128 | 0.2070 ± 0.0026 | 0.7467 | 0.7816 | 0.0578 | 0.5644 |
| V50-REF | R2 | F3 | unpaired | 128 | -0.1716 ± 0.0030 | 0.4867 | 0.4835 | — | -0.1156 |
| V50-REF | R24 | F1 | normal | 128 | 0.2988 ± 0.0013 | 0.4328 | — | 0.6177 | 0.5227 |
| V50-REF | R24 | F2 | normal | 128 | 0.3110 ± 0.0016 | 0.4522 | — | 0.6492 | 0.5227 |
| V50-REF | R24 | F3 | normal | 128 | 0.1684 ± 0.0014 | 0.2981 | — | 0.5411 | 0.5227 |
| V50-REF | R24 | F3 | unpaired | 128 | -0.1231 ± 0.0007 | 0.0474 | — | — | -0.0298 |
| V50-REF | R8 | F1 | normal | 128 | 0.3093 ± 0.0016 | 0.5389 | — | 0.5833 | 0.5858 |
| V50-REF | R8 | F2 | normal | 128 | 0.3324 ± 0.0016 | 0.5796 | — | 0.6018 | 0.5858 |
| V50-REF | R8 | F3 | normal | 128 | 0.1853 ± 0.0014 | 0.4308 | — | 0.4682 | 0.5858 |

## Primary contrasts (paired over worlds; Holm across the family; materiality δ = 0.02)

| id | a − b | n | mean | 95% CI | p Holm | material | note |
|---|---|---:|---:|---|---:|---|---|
| C-1a | V50-REF|R2|F3|massed|1.0|normal − V50-NOLEARN|R2|F3|massed|1.0|normal | 128 | 0.2070 | [0.2019, 0.2120] | 0.0000 | True |  |
| C-1b | F151-E-FE0|R2|F3|massed|1.0|normal − F151-E-FE0-NOLEARN|R2|F3|massed|1.0|normal | 128 | 0.0048 | [0.0045, 0.0051] | 0.0000 | False |  |
| C-2a | F151-E-FE0|R2|F1|massed|1.0|normal − V50-REF|R2|F1|massed|1.0|normal | 128 | -0.3816 | [-0.3888, -0.3745] | 0.0000 | True |  |
| C-2b | F151-E-FE0|R2|F2|massed|1.0|normal − V50-REF|R2|F2|massed|1.0|normal | 128 | -0.3611 | [-0.3674, -0.3548] | 0.0000 | True |  |
| C-2c | F151-E-FE0|R2|F3|massed|1.0|normal − V50-REF|R2|F3|massed|1.0|normal | 128 | -0.2023 | [-0.2074, -0.1971] | 0.0000 | True |  |
| E-1a | V50-REF interleaved − V50-REF massed | 32 | -0.0154 | [-0.0177, -0.0131] | 0.0000 | False |  |
| E-2a | V50-REF clock x2 − V50-REF clock x0.5 | 32 | 0.0000 | [0.0000, 0.0000] | — | False | structural zero: V50-REF has no time-dependent state (Q5 rest no-op); clock scaling cannot act |
| E-1b | F151-E-FE0 interleaved − F151-E-FE0 massed | 32 | -0.0018 | [-0.0021, -0.0016] | 0.0000 | False |  |
| E-2b | F151-E-FE0 clock x2 − F151-E-FE0 clock x0.5 | 32 | -0.0120 | [-0.0123, -0.0116] | 0.0000 | False |  |

## Selected diagnostics (means over confirmation worlds; definitions in DESIGN.md §5.6)

| metric | F151-E-FE0 R2 F1 | F151-E-FE0 R2 F2 | F151-E-FE0 R2 F3 | F151-E-FE0-NOLEARN R2 F3 | F151-E-FE0-USCUT R2 F3 | F151-E-FEID R2 F1 | F151-E-FEID R2 F3 | F151-E-FETAIL4 R2 F3 | F151-S-FE0 R2 F1 | F151-S-FE0 R2 F3 | V50-NOLEARN R2 F3 | V50-NOLEARN R24 F3 | V50-REF R2 F1 | V50-REF R2 F2 | V50-REF R2 F3 | V50-REF R24 F1 | V50-REF R24 F2 | V50-REF R24 F3 | V50-REF R8 F1 | V50-REF R8 F2 | V50-REF R8 F3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| acc|day_relevant | 0.741 | 0.511 | 0.510 | 0.500 | 0.500 | 0.895 | 0.895 | 0.732 | 0.500 | 0.500 | 0.500 | 0.041 | 0.802 | 0.964 | 0.940 | 0.571 | 0.923 | 0.736 | 0.680 | 0.940 | 0.822 |
| acc|final_trained | 0.571 | 0.501 | 0.501 | 0.500 | 0.500 | 0.643 | 0.643 | 0.566 | 0.500 | 0.500 | 0.500 | 0.040 | 0.622 | 0.756 | 0.706 | 0.177 | 0.343 | 0.227 | 0.304 | 0.467 | 0.362 |
| acc|revision_curve_x1 | 0.526 | 0.500 | 0.500 | 0.500 | 0.500 | 0.535 | 0.535 | 0.520 | 0.500 | 0.500 | 0.500 | 0.038 | 0.409 | 0.319 | 0.410 | 0.060 | 0.059 | 0.078 | 0.143 | 0.130 | 0.169 |
| acc|revision_curve_x4 | 0.638 | 0.503 | 0.502 | 0.500 | 0.500 | 0.751 | 0.751 | 0.620 | 0.500 | 0.500 | 0.500 | 0.038 | 0.592 | 0.876 | 0.716 | 0.171 | 0.329 | 0.211 | 0.308 | 0.523 | 0.361 |
| acc|revision_curve_x12 | 0.785 | 0.522 | 0.514 | 0.500 | 0.500 | 0.898 | 0.898 | 0.774 | 0.500 | 0.500 | 0.500 | 0.038 | 0.803 | 0.999 | 0.988 | 0.582 | 0.963 | 0.742 | 0.691 | 0.990 | 0.859 |
| eta|post2_stable | 0.171 | 0.012 | 0.011 | -0.000 | -0.004 | 0.526 | 0.526 | 0.166 | -0.000 | -0.000 | 0.000 | 0.000 | 0.697 | 0.823 | 0.611 | 0.643 | 0.767 | 0.499 | 0.653 | 0.754 | 0.507 |
| eta|post2_anomaly | 0.088 | 0.004 | 0.005 | -0.000 | -0.004 | 0.295 | 0.295 | 0.086 | 0.000 | -0.000 | 0.000 | 0.000 | 0.539 | 0.638 | 0.420 | 0.554 | 0.674 | 0.410 | 0.560 | 0.662 | 0.419 |
| eta|post2_mislead | 0.088 | -0.001 | 0.002 | 0.000 | -0.003 | 0.283 | 0.283 | 0.092 | 0.000 | 0.000 | 0.000 | 0.000 | 0.549 | 0.658 | 0.406 | 0.431 | 0.588 | 0.320 | 0.464 | 0.619 | 0.369 |
| recovery_acc_x0 | 0.126 | 0.001 | 0.000 | 0.000 | 0.000 | 0.282 | 0.282 | 0.108 | 0.000 | 0.000 | 0.000 | 0.013 | 0.160 | 0.331 | 0.272 | 0.245 | 0.552 | 0.363 | 0.317 | 0.542 | 0.400 |
| recovery_acc_x1 | 0.119 | 0.001 | 0.001 | 0.000 | 0.000 | 0.206 | 0.206 | 0.114 | 0.000 | 0.000 | 0.000 | 0.013 | 0.162 | 0.279 | 0.210 | 0.280 | 0.664 | 0.452 | 0.343 | 0.614 | 0.488 |
| recovery_acc_x4 | 0.098 | 0.003 | 0.000 | 0.000 | 0.000 | 0.121 | 0.121 | 0.083 | 0.000 | 0.000 | 0.000 | 0.013 | 0.089 | 0.051 | 0.145 | 0.362 | 0.587 | 0.529 | 0.344 | 0.421 | 0.462 |
| specificity|neighbour_aversion_gap | 0.019 | 0.002 | 0.002 | 0.000 | 0.000 | -0.004 | -0.004 | 0.016 | 0.000 | 0.000 | 0.000 | — | 0.055 | 0.086 | 0.063 | — | — | — | — | — | — |
| specificity|neighbour_mass_on_source | — | — | — | — | — | — | — | — | — | — | — | 0.000 | — | — | — | 0.015 | 0.028 | 0.016 | 0.030 | 0.043 | 0.027 |
| acc|probe_f4 | — | — | 0.504 | 0.500 | 0.500 | — | 0.895 | 0.732 | — | 0.500 | 0.500 | 0.041 | — | — | 0.633 | — | — | 0.187 | — | — | 0.325 |
| acc|probe_untrained | 0.498 | 0.500 | 0.500 | 0.500 | 0.500 | 0.501 | 0.501 | 0.500 | 0.500 | 0.500 | 0.500 | 0.041 | 0.503 | 0.497 | 0.502 | 0.043 | 0.042 | 0.043 | 0.125 | 0.128 | 0.124 |
| trace|fast_removal_rescues_wrong | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | — | — | — | — | — | — | — | — | — | — | — |
| trace|fast_removal_breaks_correct | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | — | — | — | — | — | — | — | — | — | — | — |
| regularity|framing_bits_saved_per_byte | — | — | — | — | — | — | — | — | — | — | 0.000 | 0.000 | 7.495 | 7.847 | 7.590 | 6.420 | 7.785 | 7.485 | 6.591 | 7.791 | 7.496 |
| regularity|markov_framing_bits_saved_per_byte | — | — | — | — | — | — | — | — | — | — | — | — | 4.314 | 6.599 | 6.288 | 2.538 | 6.062 | 5.688 | 3.058 | 6.358 | 6.007 |

## Front ends (Block B; choice frozen before any Block D result: **FE0**)

| front end | fmt | margin M | retrieval w1 | retrieval w2 (held-out) | stationarity | framing share |
|---|---|---:|---:|---:|---:|---:|
| FE0 | F1 | 17.675 | 1.000 | — | 1.000 | 0.085 |
| FE0 | F2 | 7.377 | 1.000 | — | 1.000 | 0.121 |
| FE0 | F3 | 2.130 | 0.566 | 0.730 | 1.000 | 0.121 |
| FE1 | F1 | 17.481 | 1.000 | — | 1.000 | 0.086 |
| FE1 | F2 | 5.641 | 1.000 | — | 1.000 | 0.221 |
| FE1 | F3 | 1.490 | 0.488 | 0.514 | 1.000 | 0.221 |
| FE2 | F1 | 24.138 | 1.000 | — | 1.000 | 0.000 |
| FE2 | F2 | 2.266 | 1.000 | — | 1.000 | 0.613 |
| FE2 | F3 | 0.167 | 0.039 | 0.045 | 1.000 | 0.613 |
| FE3 | F1 | 24.138 | 1.000 | — | 1.000 | 0.000 |
| FE3 | F2 | 2.364 | 1.000 | — | 1.000 | 0.596 |
| FE3 | F3 | 0.180 | 0.057 | 0.049 | 1.000 | 0.596 |
| FEID | F1 | 29.364 | 1.000 | — | 1.000 | — |
| FEID | F2 | 29.364 | 1.000 | — | 1.000 | — |
| FEID | F3 | 29.364 | 1.000 | 1.000 | 1.000 | — |
| FELAG | F1 | 7.145 | 1.000 | — | 0.280 | 0.231 |
| FELAG | F2 | 6.528 | 1.000 | — | 1.000 | 0.245 |
| FELAG | F3 | 1.814 | 0.467 | 0.555 | 1.000 | 0.245 |
| FETAIL4 | F1 | 18.153 | 1.000 | — | 1.000 | 0.084 |
| FETAIL4 | F2 | 16.793 | 1.000 | — | 1.000 | 0.082 |
| FETAIL4 | F3 | 16.793 | 1.000 | 1.000 | 1.000 | 0.082 |
| V50LAG | F1 | 26.397 | 1.000 | — | 1.000 | 0.073 |
| V50LAG | F2 | 14.481 | 1.000 | — | 1.000 | 0.127 |
| V50LAG | F3 | 2.660 | 0.977 | 0.969 | 1.000 | 0.127 |

## Development screen (Block D, descriptive)

| arm | rung | fmt | n_worlds | eta | accuracy |
|---|---|---|---|---|---|
| F151-E-FE0 | R2 | F3 | 16 | 0.0049 | 0.5054 |
| F151-E-FE1 | R2 | F3 | 16 | 0.0220 | 0.5483 |
| F151-E-FE2 | R2 | F3 | 16 | 0.0001 | 0.5050 |
| F151-E-FE3 | R2 | F3 | 16 | 0.0002 | 0.5065 |
| F151-E-FELAG | R2 | F3 | 16 | 0.0146 | 0.5072 |

## Time factors (Block E)

| arm | density | clock | n_worlds | eta | accuracy |
|---|---|---|---|---|---|
| F151-E-FE0 | interleaved | 1.0 | 32 | 0.0032 | 0.5048 |
| F151-E-FE0 | massed | 0.5 | 32 | 0.0044 | 0.5043 |
| F151-E-FE0 | massed | 1.0 | 32 | 0.0050 | 0.5067 |
| F151-E-FE0 | massed | 2.0 | 32 | -0.0075 | 0.5006 |
| V50-REF | interleaved | 1.0 | 32 | 0.1968 | 0.7320 |
| V50-REF | massed | 0.5 | 32 | 0.2122 | 0.7500 |
| V50-REF | massed | 1.0 | 32 | 0.2122 | 0.7500 |
| V50-REF | massed | 2.0 | 32 | 0.2122 | 0.7500 |

## Calibration (fitted on calibration worlds only, then frozen)

| cell | decoder T | η on CAL | NLMS step |
|---|---:|---:|---:|
| F151-E-FE0|R2|F1 | 14.46 | 0.0749 | 1.0 |
| F151-E-FE0|R2|F2 | 90.22 | 0.0051 | 1.0 |
| F151-E-FE0|R2|F3 | 90.22 | 0.0041 | 0.3 |
| F151-E-FE1|R2|F1 | 9.351 | 0.1001 | 1.0 |
| F151-E-FE1|R2|F2 | 17.21 | 0.0497 | 1.0 |
| F151-E-FE1|R2|F3 | 26.62 | 0.0204 | 0.3 |
| F151-E-FE2|R2|F1 | 9.351 | 0.1065 | 1.0 |
| F151-E-FE2|R2|F2 | 670.2 | 0.0001 | 0.1 |
| F151-E-FE2|R2|F3 | 515.9 | 0.0002 | 0.1 |
| F151-E-FE3|R2|F1 | 9.351 | 0.0981 | 1.0 |
| F151-E-FE3|R2|F2 | 364 | 0.0004 | 0.1 |
| F151-E-FE3|R2|F3 | 472.9 | 0.0002 | 0.1 |
| F151-E-FEID|R2|F1 | 4.266 | 0.2397 | 1.0 |
| F151-E-FEID|R2|F2 | 4.266 | 0.2397 | 1.0 |
| F151-E-FEID|R2|F3 | 4.266 | 0.2397 | 1.0 |
| F151-E-FELAG|R2|F1 | 4.655 | 0.1618 | 1.0 |
| F151-E-FELAG|R2|F2 | 31.69 | 0.0203 | 1.0 |
| F151-E-FELAG|R2|F3 | 29.05 | 0.0152 | 0.3 |
| F151-E-FETAIL4|R2|F1 | 13.25 | 0.0823 | 1.0 |
| F151-E-FETAIL4|R2|F2 | 14.46 | 0.0756 | 1.0 |
| F151-E-FETAIL4|R2|F3 | 14.46 | 0.0756 | 1.0 |
| F151-S-FE0|R2|F1 | 3219 | 0.0000 | — |
| F151-S-FE0|R2|F3 | 9165 | 0.0000 | — |
| V50-REF|R24|F1 | 0.2398 | 0.2996 | 2.0 |
| V50-REF|R24|F2 | 0.2754 | 0.3190 | 2.0 |
| V50-REF|R24|F3 | 0.3389 | 0.1798 | 2.0 |
| V50-REF|R2|F1 | 0.2754 | 0.4756 | 1.0 |
| V50-REF|R2|F2 | 0.257 | 0.3810 | 1.0 |
| V50-REF|R2|F3 | 0.3389 | 0.2200 | 0.3 |
| V50-REF|R8|F1 | 0.2951 | 0.3205 | 2.0 |
| V50-REF|R8|F2 | 0.3389 | 0.3245 | 2.0 |
| V50-REF|R8|F3 | 0.4169 | 0.1832 | 2.0 |

IDREF forgetting factors: {"R2": {"eta_cal": 0.5644089495626329, "lambda": 0.7}, "R24": {"eta_cal": 0.5226950152869337, "lambda": 1.0}, "R8": {"eta_cal": 0.5858658485346755, "lambda": 1.0}}

Confirmation n (pilot rule max(32, ceil((1.96*SD/0.01)^2)), cap 128): **128**; per-contrast required n: C-1a=25, C-1b=1, C-2a=137, C-2b=40, C-2c=25

## Figures

![fig1_milestones.png](results/figures/fig1_milestones.png)
![fig2_acquisition.png](results/figures/fig2_acquisition.png)
![fig3_retention.png](results/figures/fig3_retention.png)
![fig4_recovery.png](results/figures/fig4_recovery.png)
![fig5_interface.png](results/figures/fig5_interface.png)
![fig6_time_factors.png](results/figures/fig6_time_factors.png)

## Scope

- No promotion, minimality or biological-fit claim (DESIGN.md §10).
- FULL_151 at S0 is the unchanged learner with a fixed byte front end and an innate reinforcer detector; it cannot express the 8- or 24-way rungs until stage S2.
- Per-event arrays for every life are kept in `results/jobs/*.npz` (not in the zip) and listed with sha256 in `MANIFEST.json`.
- Lives in this run: 3856.
