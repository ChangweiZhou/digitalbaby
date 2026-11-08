# E16-F2 predictions: the label-free reservoir at 65,536 on the fresh seeds (8 November 2026, written before any E16-F2 life)

Brief: `OPUS_TASK_E16F2_RESERVOIR_FRESH_SEEDS_20261108.md`, as amended at 17:25 EST, together with the E16-F amendment of 17:45 EST in `OPUS_TASK_E16F_FRESH_SEED_CONFIRMATION_20261108.md`. Harness run: commit da9baf76 (it contains 7703d6c and 72d3aca; `exp16_integrated.py`, `summarize_exp16.py` unchanged by this job).

Arm: `E16_REPLAY=total E16_CAP=65536 python3 exp16_integrated.py T <seed>` for seeds 200003, 300007, 400009, on E16-F's births `exp45/e11_dev_M_<seed>.pt`. Receipts `exp45/e16_life_T_tot65536_<seed>.json`; logs `exp45/logs/e16f2_tot65536_<seed>.log`. Twin: E16-F's `exp45/e16_life_D_<seed>.json` (finished, six blocks each).

## Predictions (from the amended brief; not to be changed)

- **P16F2.1** After block 6, the ratio to the twin is at or above 0.80 on all three families in every fresh seed: the mod-97 operations at k = 12, the held compositions (final answer), and the mod-73 operations at k = 12. Standing seeds: operations 0.97 / 0.96 / 0.88, compositions 0.88 / 0.81 / 0.83, later operations 0.99 / 0.98 / 0.95.
- **P16F2.2** Composition loss from block 4 to block 6 is at most 15 points in every seed (standing: 5.4 / 11.9 / 11.8), and the reservoir's distinct count at block 5 is between 27,500 and 29,500.
- **P16F2.3** Acquisition: the mod-97 operations after block 2 at or above 85%, and the held compositions after block 4 at or above 80%, in every seed.

## Launch

Following the E16-F amendment, the three E16-F2 lives run at the same time as the three E16-F reruns (`E16_REPLAY=all_balanced`, relaunched at about 22:24 UTC). That makes six processes on four cores. Each life is detached with `setsid nohup`, so no tool time limit can stop it. Forecast from E16-F's measured pace (arm T 142 ms per update alone; about 29 min of wall per block with six processes on four cores): about 2.9–3.5 hours of wall time per E16-F2 life, ending about 01:30–02:00 UTC on 9 November.

Logs are written to the session's scratch folder while the lives run and copied into `exp45/logs/` at each commit. A `git pull --rebase` replaces a tracked log, which would cut a running process off from the file it writes; E16-F's births lost their logs to this.
