# E16-F3 predictions: the raw-byte reservoir on the fresh seeds (9 November 2026, written before launch)

Brief: `OPUS_TASK_E16F2_RESERVOIR_FRESH_SEEDS_20261108.md`, section E16-F3 (added 19:00 EST). Harness at commit 3b99b700, which contains 7703d6c, where the `raw` mode was added; no harness file is changed by this job. No `exp45/e16_life_T_raw65536_<seed>.json` existed for these seeds before launch.

Arm: `E16_REPLAY=raw E16_CAP=65536 python3 exp16_integrated.py T <seed>` for seeds 200003, 300007, 400009, on E16-F's births. Receipts `exp45/e16_life_T_raw65536_<seed>.json`; logs `exp45/logs/e16f3_raw65536_<seed>.log`. Twin: E16-F's `exp45/e16_life_D_<seed>.json`. Comparator for P16F3.2: E16-F2's `exp45/e16_life_T_tot65536_<seed>.json`.

## Predictions (from the brief; not to be changed)

- **P16F3.1** After block 6, the ratio to the twin is at or above 0.80 on all three families in every seed. Standing seeds: operations 0.98 / 0.96 / 0.87, compositions 0.86 / 0.81 / 0.85, later operations 0.98 / 0.98 / 0.92.
- **P16F3.2** After block 6, within 5 points of this session's E16-F2 (`tot65536`) result on every family in every seed.
- **P16F3.3** At block 5, the reservoir's byte size is between 1.00 and 1.10 MB and its distinct count between 56,000 and 59,000.

## Launch and forecast

The lives are detached with `setsid nohup`, as the E16-F amendment directs. Logs go to the session's scratch folder and are copied into `exp45/logs/` at commits.

The three E16-F4 births run at the same time, on the machine's fourth core. They depend on nothing in E16-F3, and the brief's order "after E16-F3" is kept for E16-F4's lives. That makes six processes on four cores.

Forecast from this machine's pace: the parsed arm cost 5,080 CPU-s per life here against 3,300 on the standing machine, a factor of 1.54. Verbatim replay cost 2,150 CPU-s per life on the standing machine, so here about 3,300 CPU-s per life. With six processes on four cores, that is about 80 minutes of wall time.
