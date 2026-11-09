# E16-F4 predictions: the raw-byte gate result on three more fresh seeds (9 November 2026, written before the births)

Brief: `OPUS_TASK_E16F2_RESERVOIR_FRESH_SEEDS_20261108.md`, section E16-F4 (added 19:50 EST). Harness at commit 3b99b700; no harness file is changed by this job. Seeds 500011, 600013, 700027. No `exp45/e11_dev_M_<seed>.*` and no `exp45/e16_life_*_<seed>.json` existed for any of them before this job (checked 9 November 2026 at commit 3b99b700).

## Predictions (from the brief; not to be changed)

- **P16F4.1 (births)** Each birth reaches at least 85% at k = 12 on all four development operations.
- **P16F4.2 (twin)** After block 6 the twin is at or above 90 / 85 / 90 on family means: the mod-97 operations at k = 12, the held compositions (final answer), and the mod-73 operations at k = 12.
- **P16F4.3 (raw-byte learner)** After block 6 the learner is at or above 0.80 of the twin on all three families in every seed. Standing seeds: 0.98 / 0.96 / 0.87, 0.86 / 0.81 / 0.85, 0.98 / 0.98 / 0.92.
- **P16F4.4 (retention)** Compositions after block 6 are at or above 75% and within 12 points of their block-4 value in every seed. The margin was widened from 10 to 12 after E16-F's one miss at 12.0; the brief states this before launch.
- **P16F4.5 (acquisition)** As in P16F.4: after block 2 the learner is at or above 85% on the mod-97 operations at k = 12, and after block 4 at or above 80% on the held compositions, in every seed.

## Order and forecast

1. **Births:** `python3 exp11_cell.py dev M <seed>`, three in parallel. They start at the same time as E16-F3's three lives, since they share nothing with them. That makes six processes on four cores. Here a birth cost 2,940–3,096 CPU-s, so each takes about 75 minutes of wall time while sharing.
2. **Lives, after E16-F3 and the births:** six in parallel, `E16_REPLAY=raw E16_CAP=65536 python3 exp16_integrated.py T <seed>` and `python3 exp16_integrated.py D <seed>`. Expected cost is about 3,300 CPU-s per T life and 3,840 per D life at this machine's pace. That is about 21,400 CPU-s, or about 1.6 h of wall time on four cores.
3. **Tables:** `E16_SEEDS=500011,600013,700027 E16_SUMMARY=SUMMARY_E16F4.md python3 summarize_exp16.py`.

Every process is detached with `setsid nohup`. Logs go to the session's scratch folder and are copied into `exp45/logs/` at commits. Partial receipts are committed after blocks. The whole job should take about 3 hours of wall time.
