# E16-F5 predictions: the stream without stages on the fresh seeds (9 November 2026, written before launch)

Brief: `OPUS_TASK_E16F5_E16RP_CLOUD_JOBS_20261109.md`, Job 1 (also `OPUS_TASK_E16F2_RESERVOIR_FRESH_SEEDS_20261108.md`, section E16-F5). This session runs Job 1 only. Harness at commit 923b64a0; no harness file is changed by this job. Seeds 200003, 300007, 400009, on E16-F's births `exp45/e11_dev_M_<seed>.pt`. No `exp45/e16_life_T_raw65536_drift_online_<seed>.json` or `exp45/e16_life_D_drift_<seed>.json` existed for these seeds before launch.

## Runs

```
E16_REPLAY=raw E16_CAP=65536 E16_ADMIT=online E16_STREAM=drift python3 exp16_integrated.py T <seed>
E16_STREAM=drift python3 exp16_integrated.py D <seed>
```

Receipts `exp45/e16_life_T_raw65536_drift_online_<seed>.json` and `exp45/e16_life_D_drift_<seed>.json`, with `.pt` final weights. Logs `exp45/logs/e16f5_<arm>_<seed>.log`.

## Predictions (from the brief; not to be changed)

- **P16F5.1** After the last block the learner is at or above 0.80 of the drift twin on all three families in every seed. The families are the mod-97 operations at k = 12, the held compositions by final answer, and the mod-73 operations at k = 12. Standing seeds: 0.97 / 0.93 / 0.84, 0.86 / 0.87 / 0.86, 0.99 / 0.96 / 0.97.
- **P16F5.2** The drift twin is at or above 90 / 85 / 90 on family means after the last block.
- **P16F5.3** The learner's held compositions after the last block are within 6 points of E16-F3's raw-byte result on the same seed (`exp45/e16_life_T_raw65536_<seed>.json`: 81.8 / 79.5 / 82.5 for 200003 / 300007 / 400009). The composition panels were re-keyed on 8 November, so the band includes about 2 points of panel sampling.

## Launch and forecast

The six lives are launched together, each detached with `setsid nohup`. Logs are written to the session's scratch folder and copied into `exp45/logs/` at commits, because on 8 November a `git pull --rebase` cut off logs that were being written inside the repository. Partial receipts are committed after blocks.

Pace on this machine (4 cores, Python 3.11.18, torch 2.15.0+cpu, NumPy 2.4.6), from E16-F3 and E16-F4 with six processes on four cores:
- the raw-byte learner cost 5,090–5,340 CPU-s per life;
- the twin cost 4,430–4,510 CPU-s per life.

So this job should cost about 29,000 CPU-s, or about 2.2 h of wall time. The brief's estimate is 3.5 h.

Push target: earlier jobs from this session were pushed directly to `codex/minifly-consolidated-20261023`, which this session can push to. The brief's branch rule assumes a worker cannot, so this job is pushed the same way, with no other branch created.
