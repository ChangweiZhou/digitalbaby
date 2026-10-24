# V48 — Raw-BANC APL-like recurrent KC population controller

**Status:** locked Stage-A motif test before scientific execution  
**Objective:** test one raw-connectome motif only: whether the broad reciprocal KC↔APL structure measured in the supplied BANC v888 graph provides useful population control beyond fixed top-k sparsification.

## Research constraint

The fixed project objective remains a minimal, bounded-state, non-Transformer, online persistent-learning core materially constrained by the raw fly connectome. V48 does **not** test slots, pathways, DAN teaching, replay, factor coding, or a new plasticity rule.

## Raw BANC provenance

The runner embeds the right-side KC↔APL count profile extracted directly from:

- `banc_888_meta.feather` SHA256 `819bbcff476e52702d6f8d8604ce1f12d1d7b11942281df2f49df2a73a6f15a5`
- `banc_888_edgelist_simple_v2.feather` SHA256 `363fdef3813b72a5e45a42f17034cd5a544b654c838929cecf6e1ce5f60625cb`

Measured raw motif used:

- right KCs: 2,267
- KC→APL connected KCs: 2,144
- APL→KC connected KCs: 2,113
- reciprocal KCs: 2,057
- KC→APL recorded-count sum: 10,747
- APL→KC recorded-count sum: 12,228
- APL root: `720575941482622627`

The raw graph does not determine physiological sign or time constants. V48 therefore tests a fixed computational **population-suppression** interpretation, not a claim that the exact equation is biological physiology.

## Scientific arms

1. `TOPK` — existing 51-of-1024 fixed sparse baseline.
2. `APL-RAW` — raw paired KC→APL / APL→KC count profile.
3. `APL-DEGREE-REWIRE` — same marginals, feedback profile permuted across KCs, breaking KC-specific reciprocal correspondence.
4. `APL-WEIGHT-SHUFFLE` — preserves nonzero support/reciprocity but shuffles positive recorded-count magnitudes.

All four arms share:

- the same 96-dimensional raw context representation;
- the same 1024-unit sparse random projection;
- the same projection contacts and weights;
- exactly 51 active slow units per read/write;
- the same `1024×256` learned slow matrix size;
- the same ordinary local delta writer and learning rate 0.6;
- the same BASE-miss persistent write schedule;
- the same FAST and BASE trajectories;
- the same 50:50 FAST/SLOW deployed reader.

No arm changes persistent write count.

## APL controller

For raw projected drive `d`, operate on magnitude `m=|d|` with a scalar controller state internal to each encoding:

1. initialize `a=0`, `m=|d|`;
2. repeat three recurrent steps:
   - weighted population signal from the embedded KC→APL profile;
   - `a ← (1-0.5)a + 0.5 signal`;
   - `m ← max(|d| - 0.5 · APL→KC_weight · a, 0)`;
3. restore the original drive sign;
4. choose the top 51 absolute activities and L2-normalize.

This keeps activity sparsity exactly matched to TOPK. The only scientific variable is how the raw reciprocal motif changes **which** KCs remain active.

## Loads and seeds

- primary load: 256 identities
- secondary load: 128 identities
- 8 acquisition presentations per identity
- fresh scientific seeds: 48801–48816
- engineering seeds: 48791–48793 only

## Primary measurements

- D-cohort trained acquisition
- delayed held-out A recall and bpb after D
- whole-stream bpb
- revision-8 accuracy/bpb
- same-identity and different-identity KC overlap/cosine
- direct per-write interference:
  - same-identity alternate rendering
  - same-outcome different identity
  - different-outcome different identity
- persistent-state bytes

## Ordered qualification gates

1. **Implementation / raw-motif integrity:** complete primary roster; embedded profile reproduces the audited BANC counts.
2. **Acquisition:** `APL-RAW` and `TOPK` retain mean D acquisition ≥0.90, with no seed below 0.80.
3. **Selective interference:** relative to TOPK, `APL-RAW` must improve different-identity/different-outcome interference by at least `0.0005 bpb/write`, paired upper 95% bound <0, without materially worsening same-identity transfer.
4. **Persistent gain:** `APL-RAW − TOPK` delayed held-out bpb ≤−0.05 bit at load 256, paired upper 95% bound <0.
5. **Raw-topology specificity:** `APL-RAW` must beat both `APL-DEGREE-REWIRE` and `APL-WEIGHT-SHUFFLE` by ≥0.03 bit held-out bpb with paired upper bounds <0. If controls match, the special raw topology is not supported.
6. **Prediction safety:** whole-stream cost versus TOPK ≤+0.02 bit mean and ≤+0.04 upper 95% bound.
7. **Bounded state:** qualifying arm state <3 MiB and independent of lifetime load.

No APL gain/step/count sweep is allowed after a failed scientific result. A failed raw-vs-rewire specificity test closes this motif at this abstraction and moves the program to the next raw-BANC motif (MBON recurrence).

## Mac M3 execution

The runner is self-contained. It embeds the raw APL profile and the validated persistent-workload substrate. It uses macOS-safe `spawn` multiprocessing, one BLAS thread per worker, atomic per-job caches, and rolling atomic checkpoints.

```bash
python3 elm_v48_raw_apl_controller.py --self-test
caffeinate -dimsu python3 elm_v48_raw_apl_controller.py --stage all --workers 4 --out-dir v48_results
```

Re-running the same command resumes/reuses matching jobs.
