# V35 — Shared Solver and Local Ownership Isolate

Status: complete; 64 trajectories. Scientific run.

| Arm | n | A after A | A after D | D current | Whole bpb | Target bpb |
|---|---:|---:|---:|---:|---:|---:|
| H-O | 8 | 100.00% | 7.81% | 99.61% | 2.1907 | 1.1800 |
| H-P | 8 | 100.00% | 10.94% | 100.00% | 1.8645 | 0.7707 |
| H-RLS | 8 | 100.00% | 95.31% | 94.53% | 2.7316 | 3.0860 |
| DICT | 8 | 100.00% | 100.00% | 100.00% | 2.8390 | 0.8176 |
| BASE | 8 | 100.00% | 9.38% | 100.00% | 1.4735 | 1.7078 |
| SHARED-R | 8 | 100.00% | 7.03% | 100.00% | 1.3676 | 0.8454 |
| LOCAL-R | 8 | 100.00% | 29.69% | 100.00% | 1.3605 | 0.5987 |
| LOCAL-V | 8 | 100.00% | 99.22% | 100.00% | 1.2576 | 0.6528 |

Stage I: not_qualified
S1–S6: {"S1":true,"S2":true,"S3":true,"S4":false,"S5":false,"S6":false}

| Contrast | Mean A-after-D difference | Paired 95% CI | Statistical effect | Architectural qualification |
|---|---:|---|---|---|
| C0 | 3.12% | [-0.82%, 7.07%] | favorable_unresolved_precision | not_tested_by_C0 |
| RLS−H-P | 84.38% | [72.86%, 95.89%] | supported_below_qualification | supported_below_qualification |
| R1 | -2.34% | [-7.13%, 2.44%] | no_mean_advantage | no_mean_advantage |
| R2 | 22.66% | [12.23%, 33.08%] | supported_below_qualification | supported_below_qualification |
| R3 | 69.53% | [58.92%, 80.14%] | supported_below_qualification | qualified |
| L4 | 92.19% | [84.93%, 99.44%] | supported_below_qualification | qualified |

Stage II: qualified
L1–L6: {"L1":true,"L2":true,"L3":true,"L4":true,"L5":true,"L6":true}

Raw per-seed endpoints are in SUMMARY.csv. Noncausal O0–O3 fits are in ORACLES.json; they are never live learners.
A positive interval can support an effect below architectural thresholds. A nonsignificant interval is not equivalence.
Stage II is unbounded exact-context storage. V35 cannot qualify a bounded lifelong architecture.
