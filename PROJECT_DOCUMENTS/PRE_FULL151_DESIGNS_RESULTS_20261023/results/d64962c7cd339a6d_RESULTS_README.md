# V75 compact results and reproduction

Read REPORT.md first. All84 scientific jobs completed under the frozen PROTOCOL.md/config.json. No model is promoted. This bundle contains new science/report scripts, final tables, checks, figures and manifests. Large or unchanged assets are referenced in MANIFEST_RESULTS.json; raw datasets, parent code, inputs, checkpoints, per-assignment victim arrays, caches and old archives are excluded.

From the original project root, with the existing local Python environment:

```sh
_omnibus10_env/bin/python minifly/iteration20/runner.py verify
_omnibus10_env/bin/python minifly/iteration20/reporting/analyze75.py
_omnibus10_env/bin/python minifly/iteration20/reporting/audit75.py
_omnibus10_env/bin/python minifly/iteration20/reporting/gate_audit75.py
```

The runner is resumable: `runner.py run` skips only verified completed jobs. `prepare` verifies an existing lock; it does not regenerate a changed experiment. A failed or interrupted uncommitted job has no completion receipt. Preserve its temporary directory when auditing; do not count it as completed. This release completed without failed scientific jobs.

Figure script: `reporting/figures75.py` requires NumPy, pandas and matplotlib. The local project uses its existing `.venv` for matplotlib and `_omnibus10_env` for scientific libraries. No package download is required. The source runner uses the same inherited NumPy/SciPy/Numba model dependencies as V74.

`reporting/verify_bundle.py` checks the extracted payload. With `--project /absolute/project/root`, it additionally verifies every referenced external asset. The manifest records each asset's original project-relative path, byte size, SHA256 and category. Input and raw-result assets remain available in the same project folder, not inside the ZIP.

Key tables:

- bridge_per_history / bridge_contrasts: the complete16-arm factorial plus anchor scores and all22 declared contrasts per task.
- dose_per_history / dose_pooled_contrasts / dose_gates: paired dose comparisons, both clocks, benefit and safety decisions. The gate count is not a scientific sample size.
- dose_durable_resource: retrospective minimum tested dose;25 means censored beyond24, **not25 presentations tested**.
- time_state_comparisons / time_choice_agreement: state sensitivity and assignment-level choice agreement. ODE1s is not claimed to be fully converged.
- checks/report_audit.json and gate_reconstruction.csv: independent result reconstruction. The original scientific sources remain hash-locked.

Forced bridge choice, main joint1Hz choice, old accuracy, no-write-relative margin damage and direct12-bout-relative victim damage are different endpoints. They must not be silently interchanged. Bridge routing retains an external scalar prediction error and global normalization; it does not reconstruct the native DAN calculation. No physiological compatibility claim or fly-size reduction is made.

License: new model extensions follow the inherited GPL-3.0-or-later project code. See LICENSE and NOTICE; unchanged parent scientific and third-party sources are referenced rather than duplicated.
