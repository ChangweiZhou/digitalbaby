# Reproduction and artifact audit

The original native dependency paths remain in the repository and are referenced by `SOURCE_LOCK.json`. Canonical calibration is copied under `assets/`. `ASSET_LOCATION_AUDIT.json` discloses the initial path-coverage omission and verifies contents against locked vendor copies; dormant config differences are document paths, not kernel parameters. Main operative anatomy hashes are also published per history.

The recorded local runtime is `/private/tmp/minifly_a_v3_venv/bin/python`, with NumPy1.24.3, SciPy1.11.1 and Numba0.57.1 loaded from the local Anaconda cache. Numerical threads are one. Source caches are excluded from the package.

`qualify.py` was run twice across the first two development cycles; the first import failure and all counts are retained. `preflight.py` adds the native passive/asset audit. `run.py` executes the frozen addition roster once; **do not run it again merely to read results**. The runtime budget is cumulative across those stages, and caffeinate follows each active supervisor.

For artifact-only verification, run `audit_results.py` and `report.py` with Python3. These use standard libraries and saved decisions; they neither load a learner nor rerun records. `diagnose_saved.py` and `diagnose_marginals.py` are clearly labeled posthoc explanations, not new primary endpoints. `diagnose_addresses.py` performs only controller/address inference from birth/saved weights on eight OLD prompts, without native event calls or teaching; it is not science replay or additional training. `test_boundary_revision.py` checks the API-only cap correction with zero native events.

Files in `cycles/` distinguish the initial implementation, failed test, subsequent integration pass and final addition test/revision. No full native science history is retried. The API-only wrapper is separate from the source-locked main kernel, and its tests do not backfill learning performance. The original design delivery and arithmetic/memory pointers remain unchanged.
