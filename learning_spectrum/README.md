# Causal learning-spectrum preflight

**Complete: all three audited small-test cycles finished; full run blocked by failed theoretical gates.** See [REPORT.md](REPORT.md).

This new program is isolated from frozen Package A and the completed response experiment. It tests one fixed causal centered-interaction rule and an independent causal calibration surrogate. Both fail their respective development gates. No fresh-world behavior was inspected or generated, and no candidate was tuned after failure.

## Reproduce

Use Python3.11.15, NumPy2.2.6, SciPy1.14.1, Numba0.61.2 and pandas2.2.3. Immutable upstream minifly_a_v3/package and src are inherited unchanged from the base branch. If package sources are missing from checkout, restore the existing committed minifly_a_v3/input/MINIFLY_MUSE_A_V3_CAUSAL_GATE_20260929.zip as described in response_mechanisms/README.md; original source hashes must verify.

Run from repository root with OMP_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1, MKL_NUM_THREADS=1, NUMBA_NUM_THREADS=1, PYTHONDONTWRITEBYTECODE=1:

```sh
python -m pytest learning_spectrum/tests -q
CYCLE1_OUTPUT=CYCLE1_NEW_REPLAY.json python learning_spectrum/src/cycle1.py
CYCLE3_OUTPUT=CYCLE3_NEW_REPLAY.json python learning_spectrum/src/cycle3.py
```

Cycle2 writes once to results/CYCLE2_RESULT.json. Reproduce in a fresh copy where that output is absent; preserve original artifacts. Cycle1/3 similarly default to write-once originals, with optional distinct replay filenames. Portable development_history.json.gz includes exactly required fields from prior synthetic response histories, verified against128 original receipt SHA256s. It contains no Package A behavior. `data/PROVENANCE.json` documents source paths and hashes; those original paths record provenance and are not required for portable replay. Source-equivalence tests default to response_mechanisms/src in this repository; RESPONSE_SOURCE_ROOT can point to a separate original source checkout if needed.

The independent audit helper intentionally takes original receipt locations for extraction verification; it is separate from ordinary portable replay. Its historical execution location is recorded in its source and may require path adjustment for that independent provenance check.
