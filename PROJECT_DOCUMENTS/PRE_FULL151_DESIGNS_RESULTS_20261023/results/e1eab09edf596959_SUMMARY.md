# V33 Position-Bound Byte Memory Isolation

Status: completed; 24/24 trajectories.

Independent covariance contribution: **True**. High-retention covariance candidate: **False**.

| Writer | Whole bpb | Target bpb | A after A | A after D | D current |
|---|---:|---:|---:|---:|---:|
| ordinary | 2.2338 | 1.2492 | 100.00% | 5.47% | 100.00% |
| full | 1.9062 | 0.8451 | 100.00% | 21.88% | 100.00% |
| dictionary | 2.8572 | 0.8176 | 100.00% | 100.00% | 100.00% |

## Primary contrast

- Full minus ordinary, A_after_D: 0.164062 [0.056118, 0.272007].
- Full minus ordinary, D_current: 0.000000 [0.000000, 0.000000].
- Full minus ordinary, target_bpb: -0.404054 [-0.437883, -0.370224].

Gate components: complete_roster=pass, ordinary_acquisition_uncertainty=pass, ordinary_acquisition_mean=pass, full_acquisition_uncertainty=pass, full_acquisition_mean=pass, retention_lower_above_zero=pass, retention_gain_at_least_10pp=pass, D_current_noninferiority=pass, target_loss_noninferiority=pass.

## Revision at four changed labels

| Writer | Corrected rendering | Known alternative | Held-out alias |
|---|---:|---:|---:|
| ordinary | 100.00% | 56.25% | 46.88% |
| full | 100.00% | 56.25% | 46.88% |
| dictionary | 0.00% | 0.00% | 0.00% |

These are reused V32 seeds for a paired representation intervention, not fresh confirmation. Exactly 24 scientific trajectories; no automatic follow-up arms. Dictionary capacity grows and its cumulative counts can delay revision. Uniform-baseline gain includes alphabet learning. Equal lag-block norm is a coding invariant, not evidence that all positions contain equal information. Covariance/cap correlations do not establish a unique cause.

REPORT.json contains paired intervals and all decisions. SUMMARY.csv retains per-seed absolute levels next to gated fractions. REPRESENTATION.json includes same/different-key geometry; COVARIANCE.json includes read-only block energies and disturbance ledgers. REVISION_CURVES.csv and ACQUISITION_CURVES.csv separate renderings and exposure counts.
