# V52 — byte interface decomposition

**Status** `NO_DEPLOYABLE_INTERFACE_CANDIDATE`
**Selected deployable candidate** `None`

## Block A — geometry and stationarity

| arm | deployable | period | J_same | J_diff | ratio | 1-NN | self(stream) |
|---|:-:|---:|---:|---:|---:|---:|---:|
| `V51-LAG` | yes | 14 | 0.1597 | 0.0699 | 2.32 | 0.957 | 1.0000 |
| `PHASE-CENTER-FROZEN` | yes | 14 | 0.1298 | 0.0401 | 3.24 | 0.947 | 1.0000 |
| `PHASE-Z-FROZEN` | yes | 14 | 0.1956 | 0.0399 | 4.91 | 0.979 | 1.0000 |
| `PHASE-Z-SHUFFLED` | no | 14 | 0.1523 | 0.0625 | 2.44 | 0.932 | 0.8188 |
| `XOR-OFF` | no | 14 | 0.2729 | 0.0699 | 3.97 | 1.000 | 1.0000 |
| `PHASE-Z+XOR-OFF` | no | 14 | 0.4723 | 0.0401 | 11.81 | 1.000 | 1.0000 |
| `INVARIANT-TAIL4` | no | 14 | 1.0000 | 0.0465 | 21.56 | 1.000 | 1.0000 |

## Block B — target-only synaptic usability

| arm | U | A after D held-out | D held-out | reset drop |
|---|---:|---:|---:|---:|
| `V51-LAG` | 0.2227 | 0.2305 | 0.3691 | 0.2305 |
| `PHASE-Z-FROZEN` | 0.2539 | 0.2617 | 0.3984 | 0.2617 |
| `XOR-OFF` | 0.4609 | 0.4609 | 0.7500 | 0.4609 |
| `PHASE-Z+XOR-OFF` | 0.8066 | 0.8164 | 0.9160 | 0.8164 |
| `INVARIANT-TAIL4` | 0.9258 | 0.9297 | 0.9941 | 0.9297 |

## Block C — plasticity decomposition (V51-LAG)

| policy | A after D held-out | updates | target bpb |
|---|---:|---:|---:|
| `ALL` | 0.0312 | 15343 | 6.3622 |
| `TARGET-ONLY` | 0.2305 | 1096 | 5.5758 |
| `NO-FRAMING` | 0.0312 | 7672 | 5.9312 |
| `NO-KEY` | 0.0664 | 8767 | 6.1379 |
| `RANDOM-MATCH` | 0.0078 | 1096 | 7.8943 |
| `FIXED-NONTARGET` | 0.0039 | 1096 | 8.0823 |

## Refinements applied to the shipped design

- **A2_stationarity** — Gate moved from target-context self-overlap to ACTUAL-TIME recurrent-context self-overlap. With frozen statistics the same context at the same phase is deterministic, so the target-context test is ~1.0 by construction and gates nothing. The informative failure is a context recurring at a different phase, which is what a period/harmonic error produces.
- **PHASE_Z_SHUFFLED** — Added a nondeployable matched control: identical phase machinery and statistics count, but the phase index is a deterministic pseudo-random function of t rather than t mod P. If it matches PHASE-Z, the gain is per-coordinate rescaling rather than record-structure alignment.
- **FIXED_NONTARGET** — Added to Block C. RANDOM-MATCH varies both write count parity and write position; a fixed non-target position matches count AND regularity, separating "the target position is special" from "a consistent write position is what matters".
- **B3_primary** — Block B primary gate is the paired relative contrast B3. B1/B2 absolute thresholds are reported as usefulness context, because their values sit close to post-hoc V51 audit numbers from the same run family and were not derived independently.
