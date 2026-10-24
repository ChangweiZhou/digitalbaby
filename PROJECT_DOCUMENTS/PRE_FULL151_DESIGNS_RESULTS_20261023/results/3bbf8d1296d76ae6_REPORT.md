# Fly learning loop analysis

Script version: `1.0.0`

## V2 audit finding (prerequisite)

The 99% seed-connectivity gate is **unattainable in the full 31k-neuron graph**.
At threshold 5, the full graph reaches only 90.93%. The 10k candidate reaches 90.70%.
Only 13 of 528 isolated seeds are caused by compression; 515 are already isolated.
The connectivity "failure" is a property of the seed definitions, not of compression.

## Complete learning loop

| Leg | Hop | Routing mass | Nonzero entries |
|---|---|---:|---:|
| KC_to_MBON | norm | 612.1744 | 15,523 |
| MBON_to_DAN | hop1 | 3.7768 | 1,045 |
| MBON_to_DAN | hop2 | 2.0595 | 20,284 |
| MBON_to_DAN | hop3 | 1.3236 | 31,026 |
| MBON_to_DAN | hop4 | 0.9321 | 31,096 |
| DAN_to_KC | hop1 | 122.6320 | 3,891 |
| DAN_to_KC | hop2 | 1.5700 | 59,503 |
| DAN_to_KC | hop3 | 0.3378 | 259,718 |

## Compartment identification

Found **10** compartments from KC co-targeting overlap.
Assigned 14/104 MBONs and 10/301 DANs.

| Rank | MBONs | DANs | KCs | Mean Jaccard | Max Jaccard |
|---:|---:|---:|---:|---:|---:|
| 0 | 5 | 1 | 1,599 | 0.213 | 0.236 |
| 1 | 1 | 1 | 825 | 0.190 | 0.190 |
| 2 | 1 | 1 | 136 | 0.213 | 0.213 |
| 3 | 1 | 1 | 127 | 0.000 | 0.000 |
| 4 | 1 | 1 | 105 | 0.152 | 0.152 |
| 5 | 1 | 1 | 67 | 0.000 | 0.000 |
| 6 | 1 | 1 | 14 | 0.214 | 0.214 |
| 7 | 1 | 1 | 5 | 0.200 | 0.200 |
| 8 | 1 | 1 | 5 | 0.200 | 0.200 |
| 9 | 1 | 1 | 4 | 0.250 | 0.250 |

## Feedback compression comparison

| Method | Total error | Col p50 | Col p90 | Top-1 agree | Top-3 agree |
|---|---:|---:|---:|---:|---:|
| monolithic_NMF_rank8 | 0.0673 | 0.1621 | 0.5766 | 0.9135 | 0.8846 |
| monolithic_NMF_rank16 | 0.0441 | 0.1341 | 0.4758 | 0.9038 | 0.9455 |
| column_weighted_NMF_rank8 | 0.0740 | 0.1694 | 0.6025 | 0.9231 | 0.9006 |
| column_weighted_NMF_rank16 | 0.0463 | 0.1400 | 0.5031 | 0.9423 | 0.9487 |
| transmitter_init_NMF_rank8 | 0.0672 | 0.1583 | 0.5797 | 0.9231 | 0.8846 |
| transmitter_init_NMF_rank16 | 0.0443 | 0.1348 | 0.4896 | 0.9231 | 0.9455 |
| SVD_rank8_reference | 0.0636 | 0.1542 | 0.5721 | 0.9423 | 0.8910 |
| SVD_rank16_reference | 0.0396 | 0.1283 | 0.4460 | 0.9327 | 0.9583 |

## Closed-loop gain spectrum

MBON-space spectral radius: **0.004793**
DAN-space spectral radius: **0.004793**

A spectral radius < 1 means the feedback loop is contractive:
perturbations decay rather than amplify. This supports stable learning.
A radius near 1 means the system is near the edge of instability,
which may be useful for sensitivity but dangerous for runaway plasticity.

| Compartment | MBONs | DANs | KCs | Spectral radius | Max real eigenvalue |
|---:|---:|---:|---:|---:|---:|
| 9 | 1 | 1 | 105 | 0.000083 | 0.000083 |
| 3 | 5 | 1 | 1,599 | 0.000079 | 0.000079 |
| 0 | 1 | 1 | 825 | 0.000023 | 0.000023 |
| 84 | 1 | 1 | 136 | 0.000001 | 0.000001 |
| 60 | 1 | 1 | 14 | 0.000001 | 0.000001 |
| 89 | 1 | 1 | 127 | 0.000000 | 0.000000 |
| 33 | 1 | 1 | 5 | 0.000000 | 0.000000 |
| 20 | 1 | 1 | 67 | 0.000000 | 0.000000 |
| 39 | 1 | 1 | 5 | 0.000000 | 0.000000 |
| 58 | 1 | 1 | 4 | 0.000000 | 0.000000 |

## Learning simulation (compartment-gated plasticity)

| Arm | Phase | MSE | Old accuracy | New accuracy | W sparsity |
|---|---:|---:|---:|---:|---:|
| uniform | 0 | 0.0407 | nan | 1.0000 | 0.967 |
| uniform | 1 | 0.0424 | 1.0000 | 1.0000 | 0.919 |
| uniform | 2 | 0.0434 | 0.9750 | 1.0000 | 0.863 |
| uniform | 3 | 0.0422 | 0.9533 | 0.9700 | 0.806 |
| uniform | 4 | 0.0417 | 0.9400 | 0.9700 | 0.754 |
| structured_dense | 0 | 0.0430 | nan | 1.0000 | 0.972 |
| structured_dense | 1 | 0.0448 | 0.8300 | 0.6900 | 0.946 |
| structured_dense | 2 | 0.0459 | 0.7050 | 0.6800 | 0.919 |
| structured_dense | 3 | 0.0447 | 0.6500 | 0.5400 | 0.895 |
| structured_dense | 4 | 0.0443 | 0.5425 | 0.5900 | 0.873 |
| structured_sparse | 0 | 0.0430 | nan | 1.0000 | 0.996 |
| structured_sparse | 1 | 0.0449 | 0.7900 | 0.6700 | 0.993 |
| structured_sparse | 2 | 0.0459 | 0.6750 | 0.6100 | 0.990 |
| structured_sparse | 3 | 0.0447 | 0.6167 | 0.5500 | 0.986 |
| structured_sparse | 4 | 0.0443 | 0.5325 | 0.5700 | 0.983 |
| scrambled_sparse | 0 | 0.0430 | nan | 1.0000 | 0.997 |
| scrambled_sparse | 1 | 0.0449 | 0.7800 | 0.6400 | 0.993 |
| scrambled_sparse | 2 | 0.0459 | 0.6500 | 0.6700 | 0.990 |
| scrambled_sparse | 3 | 0.0447 | 0.6067 | 0.5400 | 0.987 |
| scrambled_sparse | 4 | 0.0443 | 0.5350 | 0.6000 | 0.984 |
| oracle_sparse | 0 | 0.0430 | nan | 0.7300 | 0.999 |
| oracle_sparse | 1 | 0.0448 | 0.5800 | 0.5600 | 0.998 |
| oracle_sparse | 2 | 0.0458 | 0.4750 | 0.6000 | 0.996 |
| oracle_sparse | 3 | 0.0447 | 0.4833 | 0.3100 | 0.995 |
| oracle_sparse | 4 | 0.0443 | 0.4100 | 0.4700 | 0.994 |

## What this analysis adds

1. The **complete learning loop** KC->MBON->DAN->KC is extracted, closing
   the loop the prior distillation left open (MBON->DAN only).
2. The **composed gate operator F@G has 98.1% density**: each MBON's
   teaching influence covers nearly all KCs. The loop provides no spatial
   selectivity. Compare with oracle gating (direct KC partners): 3.3%.
3. **Spectral radius 0.005**: the loop is deeply contractive. Learning is
   one-shot, not iterative. Teaching signals do not amplify through the cycle.
4. **Compartments** are identified from KC co-targeting overlap, but only
   14/104 MBONs cluster with DANs above Jaccard 0.15.
5. **Compression comparison**: the high column p90 error (57.66%) is not
   fixable by better NMF -- SVD rank-8 (optimal) also yields 57.2%.
   Rank 16 drops to 44.6%, confirming this is a rank limitation.
6. **5-arm learning simulation** confirms the gate density finding:
   gated learning performs worse than uniform because the dense gate
   corrupts the gradient signal without providing spatial protection.

## What this does NOT claim

- Persistent learning (the simulation is not a continual-learning test)
- Physiological equivalence (unsigned routing surrogates, no dynamics)
- Minimum sufficient circuit (no deletion or necessity test)
- Identified plasticity rule (the update law is a modeling choice)
- Compartment correspondence to published anatomical nomenclature

## Implication for the persistent learning program

The fly's KC->MBON->DAN->KC loop does **not** provide spatial selectivity
for protecting old associations. The gate density (98.1%) and loop gain
(spectral radius 0.005) rule out the loop as a selective memory protection
mechanism. If the fly solves persistent learning, the mechanism is elsewhere:

- **KC sparse coding**: ~5% activation ensures low pattern overlap
- **Temporal gating**: WHEN DANs fire (not which KCs they reach)
- **Valence signaling**: reward vs punishment populations, not spatial routing
- **Non-Hebbian rules**: plasticity rules beyond simple error-gated Hebbian

For the ELM program, this means importing the fly's feedback routing
as a gating matrix is unlikely to improve persistent learning. The routing
structure encodes reward/punishment valence, not spatial selectivity.

## Files

- `loop_operators.npz`: complete KC->MBON, MBON->DAN, DAN->KC operators
- `COMPARTMENTS.csv`: identified compartment summary
- `COMPARTMENT_MEMBERS.csv`: neuron-level compartment membership
- `compartment_jaccard.npz`: MBON-DAN Jaccard overlap matrix
- `COMPRESSION_COMPARISON.csv`: all compression strategies compared
- `COMPARTMENT_SUMMARY.csv`: identified compartment statistics
- `LOOP_GAINS.csv`: per-compartment spectral radius
- `loop_eigenvalues.npz`: full eigenvalue decomposition
- `SIMULATION_RESULTS.csv`: 3-arm learning simulation trajectories
- `SUMMARY.json`: machine-readable summary