# Limited addition development exercise

Three requested cycles are complete: **HOLD_RETENTION_GUARD**. The ordinal U045 candidate shows a positive held-out signal but fails the old-content retention guard. This folder contains DEV evidence only: four worlds,24 histories,76,818 training bytes, zero science worlds.

Read [the three-cycle report](THREE_CYCLE_REPORT.md), [protocol](EXPERIMENT.md), [summary](SUMMARY.json), [qualification](QUALIFICATION.json) and [final audit](FINAL_AUDIT.json). Sources and full compressed receipts are in [cycles](cycles/).

Keep this directory beside `autonomous_observation_v1_20261007` and `r_center_core_v1` in the same repository. Dependencies are imported read-only. Recorded runtime: Python3.11.5, numpy2.2.6, scipy1.14.1, numba0.61.2. Code rejects science IDs and extra DEV worlds. Running `runner.py` on an existing receipt is rejected; publication and formal confirmation have not been authorised. Checkpoint NPZ files in preflight directories are synthetic technical fixtures, including intentional tamper files, not failed scientific trajectories.

The result archive is a development/code-and-evidence bundle, not a standalone runtime: the pinned environment and the parent repository dependency folders are not duplicated. PACKAGE.json records every included payload file; BUNDLE_RECEIPT.json records the finished archive. No formal execution or publication is included.
