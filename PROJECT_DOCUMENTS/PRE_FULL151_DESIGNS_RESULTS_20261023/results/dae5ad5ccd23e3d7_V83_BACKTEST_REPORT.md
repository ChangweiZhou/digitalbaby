# V83 r3 back-test report

Date: 2026-09-15

The earlier V83 runner was not shipped again after only syntax checking.  r3 was exercised through the following staged back-tests.

## 1. Static checks

- `python -m py_compile minifly_V83_common_world_runner.py` — PASS.
- `bash -n RUN_V83.command` — PASS.

## 2. Source/dependency lock tests

Using a temporary MiniFly-like checkout:

- explicit project-root discovery — PASS;
- extraction of the bundled slim V82E source lock — PASS;
- SHA-256 verification of V82E `model_evo.py`, `common_evo.py`, and `config.json` — PASS;
- V82E champion genome versus delivered `champion_model.json` — PASS;
- V81E hard-coded genome versus delivered V81E source bundle — PASS;
- V82E protocol constants used by V83 (`acquisition_bouts`, common clock, mixed suffix, horizons) — PASS;
- deliberately modified V82E source correctly rejected — PASS.

## 3. Environment-failure test

The sandbox's NumPy 2.3.5 environment was intentionally passed to `environment_preflight()`.  It was rejected before the inherited model import with a readable compatibility error, rather than producing the late vendored-Numba traceback seen on the Mac.

The production launcher creates/uses a pinned NumPy 2.2.6 / Numba 0.61.2 environment when the current interpreter fails this check.

## 4. Full synthetic orchestration back-test

A lightweight synthetic implementation of the MiniFly API was used to exercise the V83 runner without the unavailable raw anatomical assets.

Exercised:

- all **7 V83 arms**;
- **2 independent histories**;
- both loads **48 and 96**;
- all five challenge families (STABLE, ANOMALY1, ANOMALY2, REV, NEW);
- COMMON and MIXED phases;
- day endpoints;
- per-memory output;
- fixed/coupled/actual reader views;
- resource tables;
- sealed receipt/resume path (every job was invoked twice; second invocation read and verified the sealed receipt);
- aggregation;
- shared-memory-identity analysis;
- pairwise contrasts and decision generation;
- report generation;
- secondary continuous-life path through 48 → 72 → 96 associations.

The test initially found two real defects:

1. `shared_identity()` referenced `m` on the right-hand side before assignment and would have crashed final aggregation.
2. The continuous-life panel reused a 48-entry boolean mask on 72/96-entry arrays and would have crashed late in the run.

Both were patched.  The complete synthetic workflow was rerun from the start and finished with:

`SYNTHETIC_FULL_WORKFLOW_PASS`

with the expected 28 `arm × seed × load` history-summary rows.

## 5. What cannot be fully reproduced in this sandbox

The uploaded project artifacts do not include the complete original MiniFly anatomical/data checkout needed by the inherited iteration-5→24 chain (for example the original interface/PN→KC assets).  Therefore the exact numerical V83 scientific trajectories cannot be reproduced here.

The Mac run must still begin with:

1. `--prepare-only` — imports the *real* inherited chain and runs V83 source/model self-tests before launching jobs;
2. `--quick --workers 1` — two fresh histories through the *real* scientific code.

Only after both succeed should the 24-history campaign be run.  This is deliberate: no expensive scientific jobs start before environment, source-lock and model-level checks succeed.
