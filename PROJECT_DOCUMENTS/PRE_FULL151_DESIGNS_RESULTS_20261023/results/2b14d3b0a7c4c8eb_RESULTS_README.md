# V76 result reproduction

V76 is iteration21. The accepted model remains the full5177 reference from iteration5/bi, instantiated in iteration10, with the existing fixed reader. This release is a diagnostic experiment, not a promoted learning model.

Run from the original project folder using its established Python environment:

```sh
/Users/pencilbard/Documents/second_try/_omnibus10_env/bin/python minifly/iteration21/runner.py verify
/Users/pencilbard/Documents/second_try/_omnibus10_env/bin/python minifly/iteration21/reporting/analyze76.py
/Users/pencilbard/Documents/second_try/_omnibus10_env/bin/python minifly/iteration21/reporting/audit76.py
/Users/pencilbard/Documents/second_try/_omnibus10_env/bin/python minifly/iteration21/reporting/extract76.py
/Users/pencilbard/Documents/second_try/.venv/bin/python minifly/iteration21/reporting/figures76.py
/Users/pencilbard/Documents/second_try/.venv/bin/python minifly/iteration21/reporting/age76.py
```

The analysis interpreter has NumPy, pandas, SciPy and Numba. The plotting interpreter has matplotlib and only reads the final CSV tables. These environments are external, not redistributed. Set BLAS thread counts to1 for standalone analysis if desired; the scientific runner enforces that automatically for its workers.

`runner.py prepare` verifies the118 software checks, deterministically generates the disjoint PN banks, computes one unlabeled geometric calibration, verifies inherited assets and locks the experiment. An existing preparation is verified instead of regenerated. `runner.py run` launches or resumes the32 registered jobs with two processes, capped per-worker execution time, atomic receipts and caffeinate. It verifies completed receipts instead of rerunning them. Never edit a frozen source/configuration or replace a receipt to change scientific outcomes.

The scientific implementation consists of:

- `src/model76.py`: external pair-level residual/label teaching, fixed/adaptive geometric gain, native gamma events and alpha allocation, exact V75 bridge adapter.
- `src/assays76.py`: independent noisy/reversal streams and common-prestate delayed revision/interference tests.
- `src/selftest76.py`: native and legacy parity, information restrictions, update semantics, resource and small integration checks.
- `config.json` and `PROTOCOL.md`: fixed arms, target units, input panels, endpoints, inference and diagnostic transfer guards.

The result tables aggregate only within independent histories before confidence intervals. Confirmatory inference is restricted to the ten predeclared ANOMALY joint-expression contrasts. Scale, allocation, forced-choice and transfer comparisons retain their stated diagnostic scope. Target sign is not substituted for the inherited joint actual/reader threshold.

The compact bundle contains new code, protocol/configuration, final metrics, figures, acceptance evidence and `MANIFEST_RESULTS.json`. It omits the fixed matrices, parent source, PN arrays, per-job raw victim vectors/checkpoints, cache directories and all previous archives. The manifest gives project-relative paths, sizes and SHA-256 values for required external assets. A clean reproduction requires those unchanged parent assets; this bundle is intentionally not a standalone copy of the project.

`data/calibration.json` is small and included because it is a frozen model constant with input provenance. Raw calibration trajectories remain external. Analysis/audit scripts do not alter learning state or choose an arm. `reporting/package76.py` verifies the release and refuses to overwrite an existing archive.

`extract76.py` derives the post-run descriptive stable-memory/no-write breakdown from already saved vectors, verifies threshold and reference agreement, and pools loads within history. It adds no new confirmatory test. `age76.py` renders the prespecified age strata. Verify a delivered archive independently with `reporting/verify_bundle.py ARCHIVE_PATH`.
