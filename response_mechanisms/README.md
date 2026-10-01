# Full151 response-mechanism experiments

This new program investigates the response decomposition, own-output homeostasis,
within-cue PN/KC timing plasticity, and separately writable conjunctive eligibility
on actual canonical Full151 models. It is separate from the frozen `minifly_a_v3`
confirmation and does not modify that experiment.

## Status

Two design–independent-audit–pilot–revision cycles are complete. The third-cycle design and code passed39 tests, including a sealed-final-path smoke test on a technical world; its pilot is pending shared-resource availability. Cycle1's
cross-record timing implementation was identified as a mismatch to the intended
within-cue hypothesis and is preserved explicitly as exploratory development.
The final fresh-world run has **not** started. This README will be updated with
the locked roster and audited final outcomes.

## Reproduction

Use Python3.11.15, NumPy2.2.6, SciPy1.14.1, Numba0.61.2, pandas2.2.3.
Restore the upstream immutable package from its already committed archive:

```sh
mkdir -p minifly_a_v3/package
unzip -n minifly_a_v3/input/MINIFLY_MUSE_A_V3_CAUSAL_GATE_20260929.zip -d minifly_a_v3/package
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export NUMBA_CACHE_DIR="$PWD/response_mechanisms/scratch/numba"
python -m pytest response_mechanisms/tests -q
```

Every run verifies upstream imported source hashes against A V3's original
source lock. Pilot receipts include complete fixture, per-record emissions and
literal write ledger, response tensors, component decomposition, state allocation,
source hashes, timings, and peak RSS. A result receipt is immutable once written.
Pilot source snapshots and independent audit dispositions remain under `design/`.

## Scope

This is a four-output, balanced random-pair learning and retention assay with old
and interfering new 4×4 cue grids. It does not establish withheld-pair relation
transfer, general reasoning, or a biological explanation. The historical numeric
claims supplied as motivation could not yet be independently located and are not
accepted thresholds. All numerical conclusions will come from the saved new data.
