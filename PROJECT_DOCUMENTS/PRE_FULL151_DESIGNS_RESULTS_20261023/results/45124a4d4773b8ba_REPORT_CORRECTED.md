# V39 results — bounded superposed memory

**Source SHA256** `0f9577b4a06e1443f0ff38753f630a1a3df8bc7c6fca519f0efa997e820a78d0`  
**Stage** `full` · **seeds** 16 · **verdict** **`qualified`**

Label-free: the only teaching signal is the next-byte residual. Workload is V38's, extended with a load axis and byte-identical to V38 at V38's settings.

Primary endpoint `joint_min` = min(retention, held-out transfer), both on the untouched wave-A cohort after wave D, so the two share a delay.

## The frontier

Retention tracks different-key overlap; transfer tracks same-key overlap. A single-component endpoint is maximised by destroying one of them.

| cell | J_same | J_diff | ratio | A initial | retention | held-out | **joint** | payload | ops/byte |
|---|---:|---:|---:|---|---|---|---|---:|---:|
| `S1-lag-c2@32` | 0.730 | 0.492 | 1.49 | +1.000 [+1.000, +1.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | **+0.000 [+0.000, +0.000]** | 8.1 MB | 117,248 |
| `S1-lag-c2@64` | 0.729 | 0.494 | 1.48 | +0.938 [+0.913, +0.962] | +0.035 [+0.005, +0.065] | +0.023 [-0.003, +0.050] | **+0.020 [-0.004, +0.043]** | 8.1 MB | 117,248 |
| `S1-lag-c2@128` | 0.729 | 0.492 | 1.48 | +0.461 [+0.382, +0.539] | +0.023 [+0.003, +0.044] | +0.016 [+0.001, +0.031] | **+0.012 [-0.002, +0.025]** | 8.1 MB | 117,248 |
| `S1-lag-c2@256` | 0.729 | 0.492 | 1.48 | +0.184 [+0.153, +0.215] | +0.051 [+0.031, +0.070] | +0.045 [+0.025, +0.065] | **+0.043 [+0.023, +0.063]** | 8.1 MB | 117,248 |
| `S2-lag-c12@32` | 0.407 | 0.152 | 2.69 | +1.000 [+1.000, +1.000] | +0.984 [+0.962, +1.007] | +0.938 [+0.878, +0.997] | **+0.938 [+0.878, +0.997]** | 8.8 MB | 158,208 |
| `S2-lag-c12@64` | 0.408 | 0.154 | 2.65 | +1.000 [+1.000, +1.000] | +0.922 [+0.881, +0.963] | +0.898 [+0.843, +0.954] | **+0.871 [+0.817, +0.925]** | 8.8 MB | 158,208 |
| `S2-lag-c12@128` | 0.409 | 0.153 | 2.68 | +1.000 [+1.000, +1.000] | +0.645 [+0.580, +0.709] | +0.492 [+0.413, +0.572] | **+0.488 [+0.411, +0.565]** | 8.8 MB | 158,208 |
| `S2-lag-c12@256` | 0.409 | 0.153 | 2.68 | +1.000 [+1.000, +1.000] | +0.197 [+0.163, +0.231] | +0.092 [+0.066, +0.117] | **+0.092 [+0.066, +0.117]** | 8.8 MB | 158,208 |
| `S3-phase-c5@32` | 0.532 | 0.088 | 6.13 | +1.000 [+1.000, +1.000] | +0.367 [+0.247, +0.487] | +0.328 [+0.222, +0.434] | **+0.281 [+0.166, +0.397]** | 8.3 MB | 129,536 |
| `S3-phase-c5@64` | 0.520 | 0.097 | 5.40 | +1.000 [+1.000, +1.000] | +0.410 [+0.316, +0.504] | +0.344 [+0.236, +0.451] | **+0.332 [+0.228, +0.436]** | 8.3 MB | 129,536 |
| `S3-phase-c5@128` | 0.521 | 0.100 | 5.23 | +1.000 [+1.000, +1.000] | +0.203 [+0.158, +0.248] | +0.184 [+0.129, +0.239] | **+0.174 [+0.127, +0.221]** | 8.3 MB | 129,536 |
| `S3-phase-c5@256` | 0.520 | 0.101 | 5.17 | +1.000 [+1.000, +1.000] | +0.068 [+0.050, +0.086] | +0.057 [+0.035, +0.079] | **+0.054 [+0.034, +0.074]** | 8.3 MB | 129,536 |
| `S4-phase-c12@32` | 0.387 | 0.057 | 6.77 | +1.000 [+1.000, +1.000] | +0.539 [+0.413, +0.665] | +0.438 [+0.313, +0.562] | **+0.422 [+0.298, +0.546]** | 8.8 MB | 158,208 |
| `S4-phase-c12@64` | 0.380 | 0.063 | 6.07 | +1.000 [+1.000, +1.000] | +0.555 [+0.484, +0.625] | +0.492 [+0.400, +0.585] | **+0.469 [+0.384, +0.554]** | 8.8 MB | 158,208 |
| `S4-phase-c12@128` | 0.383 | 0.064 | 6.01 | +1.000 [+1.000, +1.000] | +0.232 [+0.176, +0.289] | +0.164 [+0.116, +0.213] | **+0.154 [+0.107, +0.202]** | 8.8 MB | 158,208 |
| `S4-phase-c12@256` | 0.382 | 0.064 | 5.94 | +1.000 [+1.000, +1.000] | +0.023 [+0.014, +0.033] | +0.027 [+0.013, +0.042] | **+0.019 [+0.008, +0.029]** | 8.8 MB | 158,208 |
| `HASH-O@32` | 0.027 | 0.026 | 1.02 | +1.000 [+1.000, +1.000] | +1.000 [+1.000, +1.000] | +0.000 [+0.000, +0.000] | **+0.000 [+0.000, +0.000]** | 8.3 MB | 129,536 |
| `HASH-O@64` | 0.026 | 0.026 | 1.01 | +1.000 [+1.000, +1.000] | +0.812 [+0.769, +0.856] | +0.000 [+0.000, +0.000] | **+0.000 [+0.000, +0.000]** | 8.3 MB | 129,536 |
| `HASH-O@128` | 0.026 | 0.026 | 0.99 | +1.000 [+1.000, +1.000] | +0.002 [-0.002, +0.006] | +0.000 [+0.000, +0.000] | **+0.000 [+0.000, +0.000]** | 8.3 MB | 129,536 |
| `HASH-O@256` | 0.026 | 0.026 | 0.99 | +1.000 [+1.000, +1.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | **+0.000 [+0.000, +0.000]** | 8.3 MB | 129,536 |
| `S5-phase-c12-k82@32` | 0.350 | 0.032 | 11.01 | +1.000 [+1.000, +1.000] | +0.539 [+0.446, +0.632] | +0.578 [+0.443, +0.713] | **+0.477 [+0.362, +0.591]** | 8.8 MB | 95,232 |
| `S5-phase-c12-k82@256` | 0.341 | 0.038 | 8.95 | +1.000 [+1.000, +1.000] | +0.016 [+0.007, +0.024] | +0.006 [-0.001, +0.013] | **+0.004 [-0.002, +0.010]** | 8.8 MB | 95,232 |
| `PRED-V@32` | -- | -- | -- | +1.000 [+1.000, +1.000] | +1.000 [+1.000, +1.000] | +0.969 [+0.923, +1.014] | **+0.969 [+0.923, +1.014]** | 5.1 MB | 68,597 |
| `PRED-V@64` | -- | -- | -- | +1.000 [+1.000, +1.000] | +1.000 [+1.000, +1.000] | +1.000 [+1.000, +1.000] | **+1.000 [+1.000, +1.000]** | 9.1 MB | 117,652 |
| `EXACT-V@32` | -- | -- | -- | +1.000 [+1.000, +1.000] | +1.000 [+1.000, +1.000] | +0.016 [-0.018, +0.049] | **+0.016 [-0.018, +0.049]** | 5.6 MB | 77,532 |
| `EXACT-V@64` | -- | -- | -- | +1.000 [+1.000, +1.000] | +0.996 [+0.988, +1.004] | +0.008 [-0.009, +0.024] | **+0.008 [-0.009, +0.024]** | 10.2 MB | 140,494 |

## Gates

### G0 — acquisition validity (≥ 0.90, at every load) · **pass**

| cell | A initial | valid |
|---|---|:-:|
| `EXACT-V@32` | +1.000 [+1.000, +1.000] | True |
| `EXACT-V@64` | +1.000 [+1.000, +1.000] | True |
| `HASH-O@128` | +1.000 [+1.000, +1.000] | True |
| `HASH-O@256` | +1.000 [+1.000, +1.000] | True |
| `HASH-O@32` | +1.000 [+1.000, +1.000] | True |
| `HASH-O@64` | +1.000 [+1.000, +1.000] | True |
| `PRED-V@32` | +1.000 [+1.000, +1.000] | True |
| `PRED-V@64` | +1.000 [+1.000, +1.000] | True |
| `S1-lag-c2@128` | +0.461 [+0.382, +0.539] | False |
| `S1-lag-c2@256` | +0.184 [+0.153, +0.215] | False |
| `S1-lag-c2@32` | +1.000 [+1.000, +1.000] | True |
| `S1-lag-c2@64` | +0.938 [+0.913, +0.962] | True |
| `S2-lag-c12@128` | +1.000 [+1.000, +1.000] | True |
| `S2-lag-c12@256` | +1.000 [+1.000, +1.000] | True |
| `S2-lag-c12@32` | +1.000 [+1.000, +1.000] | True |
| `S2-lag-c12@64` | +1.000 [+1.000, +1.000] | True |
| `S3-phase-c5@128` | +1.000 [+1.000, +1.000] | True |
| `S3-phase-c5@256` | +1.000 [+1.000, +1.000] | True |
| `S3-phase-c5@32` | +1.000 [+1.000, +1.000] | True |
| `S3-phase-c5@64` | +1.000 [+1.000, +1.000] | True |
| `S4-phase-c12@128` | +1.000 [+1.000, +1.000] | True |
| `S4-phase-c12@256` | +1.000 [+1.000, +1.000] | True |
| `S4-phase-c12@32` | +1.000 [+1.000, +1.000] | True |
| `S4-phase-c12@64` | +1.000 [+1.000, +1.000] | True |
| `S5-phase-c12-k82@256` | +1.000 [+1.000, +1.000] | True |
| `S5-phase-c12-k82@32` | +1.000 [+1.000, +1.000] | True |

Checked at every load, not only the lowest. Cells below the threshold are invalid_acquisition and enter no contrast.

### G1 — separation axis is real (span ≥ 2.0×) · **pass**

Span **4.56×** from `S1-lag-c2` to `S4-phase-c12`; paired gap +5.286 [+4.995, +5.578].

| arm | Jaccard ratio |
|---|---|
| HASH-O | +1.024 [+0.972, +1.075] |
| S1-lag-c2 | +1.486 [+1.447, +1.525] |
| S2-lag-c12 | +2.695 [+2.544, +2.845] |
| S3-phase-c5 | +6.128 [+5.718, +6.539] |
| S4-phase-c12 | +6.773 [+6.472, +7.073] |
| S5-phase-c12-k82 | +11.011 [+10.247, +11.775] |
| EXACT-V | -- |
| PRED-V | -- |

### G2 — capacity pressure exists · **pass**

`S2-lag-c12` (the best arm at the lowest load, so the one with the most to lose) on `joint_min`, load 32 → 256: **-0.8457 [-0.9039, -0.7876]**, required ≤ -0.10 with the upper bound below zero.

If load hurts nobody there is no pressure and every contrast below is void. This is V27's headroom screen applied to capacity.

### G3 — primary: the high-ratio code holds both corners' properties · **pass**

At load **64**, the largest where every compared cell acquires. `S4-phase-c12` must beat each corner on `joint_min` by ≥ 0.10 with the paired lower bound above zero.

| contrast | joint_min difference |
|---|---|
| `S4-phase-c12` − `HASH-O` | +0.4688 [+0.3836, +0.5539] |
| `S4-phase-c12` − `S1-lag-c2` | +0.4492 [+0.3541, +0.5444] |

| cell | retention | held-out | joint |
|---|---|---|---|
| `S4-phase-c12@64` | +0.555 [+0.484, +0.625] | +0.492 [+0.400, +0.585] | +0.469 [+0.384, +0.554] |
| `HASH-O@64` | +0.812 [+0.769, +0.856] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| `S1-lag-c2@64` | +0.035 [+0.005, +0.065] | +0.023 [-0.003, +0.050] | +0.020 [-0.004, +0.043] |

### G3b — capacity curves (descriptive)

`joint_min` by load.

| arm | 32 | 64 | 128 | 256 |
|---|---:|---:|---:|---:|
| EXACT-V | 0.016 | 0.008 | -- | -- |
| HASH-O | 0.000 | 0.000 | 0.000 | 0.000 |
| PRED-V | 0.969 | 1.000 | -- | -- |
| S1-lag-c2 | 0.000 | 0.020 | 0.012 | 0.043 |
| S2-lag-c12 | 0.938 | 0.871 | 0.488 | 0.092 |
| S3-phase-c5 | 0.281 | 0.332 | 0.174 | 0.054 |
| S4-phase-c12 | 0.422 | 0.469 | 0.154 | 0.019 |
| S5-phase-c12-k82 | 0.477 | -- | -- | 0.004 |

### G4 — the two frontier corners (descriptive)

HASH-O destroys input similarity (ratio ~1) and is the retention corner; the lowest-contact code destroys separation and is the transfer corner. Neither is a matched control; both are frontier endpoints.

| corner | ratio | retention | held-out |
|---|---|---|---|
| HASH-O | +1.024 [+0.972, +1.075] | +1.000 [+1.000, +1.000] | +0.000 [+0.000, +0.000] |
| S1-lag-c2 | +1.486 [+1.447, +1.525] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |

## Reference points

| quantity | value |
|---|---:|
| fly KC overlap, same odour (real wiring, 5% coding) | 0.417 |
| fly KC overlap, different odour | 0.032 |
| **fly separation ratio** | **13.5×** |
| ChatGPT V36 preflight gate | ratio ≥ 10.0, J_diff ≤ 0.04 |
| ChatGPT V36 measured (lag96 front end) | 4.01–4.91× |
| V38 PRED-V slots after D (unbounded) | 1504 |
| V38 E1 / E2 held-out transfer | 0.008 / 1.0 |

## Interpretation limits

- G0, G1 and G2 are hard gates in that order. A failure at any level voids the
  levels beneath it; running mechanism contrasts under a void gate is how V27
  and V31 produced uninterpretable results.
- Retention alone is the wrong endpoint. It is maximised by destroying input
  similarity, which HASH-O does by construction, at the cost of all transfer.
  `joint_min` is used precisely so that neither corner can win by giving up the
  other property.
- The superposed arms carry no shared base learner, so their whole-stream bpb is
  not comparable to the slot arms'. Retention and transfer are the endpoints.
- Jaccard on sparse active sets and cosine on dense codes are different metrics.
  They are reported side by side and never compared numerically.
- Revision endpoints are measured shortly after the revision and are therefore
  recency-protected; the primary endpoint deliberately uses the wave-D delay for
  both of its components instead.
- Storage is bounded only for the superposed arms. The slot arms are the
  unbounded reference and run only at the two lowest loads.
- Synthetic stream with controlled renderings. No scaling or language claim.
