# Full151 response-mechanism experiments

This new program investigates the response decomposition, own-output homeostasis,
within-cue PN/KC timing plasticity, and separately writable conjunctive eligibility
on actual canonical Full151 models. It is separate from the frozen `minifly_a_v3`
confirmation and does not modify that experiment.

## Status

All three design–independent-audit–pilot–revision cycles are complete, with all
21 exploratory lives retained and 39 tests passed. No efficacy win is claimed
from these pilots. The final source is locked for 32 fresh worlds × seven arms,
plus seven preselected fresh-process replays. The final run is in progress. Completed-world receipts and progress are checkpointed on this branch.

Cycle 1's cross-record timing implementation was identified as a mismatch to the
intended within-cue hypothesis. Cycle 2 corrected it; cycle 3 strengthened the
complete measurement, validation, resource and publication contract. See the
cycle histories, pilot reports and `SPEC_LOCK.md` for full limitations and methods.

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

## Pilot source history

Each `design/cycleN_source/` directory preserves the exact Python source for that
pilot. To reproduce an older pilot, restore its runner/model files to `src/` and
its `test_*.py` files to `tests/` in a separate checkout. Do not overwrite the
locked final source or existing write-once receipts. Pilot outcomes never enter
the final statistical sample.
