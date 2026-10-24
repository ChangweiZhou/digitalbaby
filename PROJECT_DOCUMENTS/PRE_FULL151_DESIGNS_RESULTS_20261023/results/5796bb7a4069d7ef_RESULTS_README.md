# V77 result reproduction

V77 is iteration22, an unpromoted diagnostic experiment. Parent science and previous releases remain unchanged. The executable dependencies and frozen matrices are referenced in MANIFEST_RESULTS.json rather than redistributed.

From `/Users/pencilbard/Documents/second_try`:

```sh
_omnibus10_env/bin/python minifly/iteration22/runner.py verify
_omnibus10_env/bin/python minifly/iteration22/reporting/analyze77.py
_omnibus10_env/bin/python minifly/iteration22/reporting/supplement77.py
_omnibus10_env/bin/python minifly/iteration22/reporting/audit77.py
.venv/bin/python minifly/iteration22/reporting/figures77.py
```

The main interpreter has NumPy/pandas/Numba/SciPy. The plotting interpreter supplies matplotlib and only reads final CSV tables. Set BLAS threads to1 when independently replaying; runner and audit enforce it. No network is needed. These environments are not bundled.

`runner.py prepare` runs61 software checks, generates32 disjoint input banks and locks scientific files and parent hashes. It reuses the frozen V76 calibration without fitting. Existing preparation is verified, not overwritten. `runner.py run` starts or resumes the32 registered jobs with two subprocesses and caffeinate; only completed, hash-verified atomic receipts are resumed. The sources/config/protocol are frozen once prepared.

`src/model77.py` adds only teaching-reference/objective computation through the V76 all-slow interface. NOFAST removes alpha-fast only in the diagnostic observation clone. `src/assays77.py` implements shared native and own-trained histories, additional between-bout rest, matched clocks/no-write controls and saved old/new vectors. All-slow own-trained TOTAL/NOFAST variants are identical and their redundancy is checked at startup.

The ten declared primary comparisons are in `results/primary_contrasts.csv`; all diagnostic safety criteria are in `results/acceptance_checks.csv`. Summary intervals use independent histories, pooling repeated loads/ages within each history. Final tables retain strata; raw per-bout files, per-cue vectors, generated input arrays and checkpoint files are intentionally external. No promotion follows automatically from any score or guard.

`reporting/package77.py` creates the compact result archive after successful science and audit and refuses to overwrite an existing release. `reporting/verify_bundle.py ARCHIVE_PATH` verifies its exact file set, sizes and every payload hash without loading scientific dependencies.
