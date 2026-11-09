# E16-F3 report: the raw-byte reservoir at 65,536 on the fresh seeds (9 November 2026)

Brief: `OPUS_TASK_E16F2_RESERVOIR_FRESH_SEEDS_20261108.md`, section E16-F3 (added 19:00 EST). Predictions: `E16F3_PREDICTIONS.md`, written and pushed before launch (commit 26274316). Arm: `E16_REPLAY=raw E16_CAP=65536 python3 exp16_integrated.py T <seed>`, on E16-F's births. The twin is E16-F's `exp45/e16_life_D_<seed>.json`. The comparator for P16F3.2 is E16-F2's `exp45/e16_life_T_tot65536_<seed>.json`. Tables: `exp45/SUMMARY_E16F3.md`.

**Outcome:** P16F3.1 and P16F3.3 hold in every seed. P16F3.2 holds in 200003 and 300007 and misses in 400009. There, the raw-byte arm ends 5.5 points *above* the parsed reservoir on the held compositions, against a band of 5. The gate itself, at least 0.80 of the twin on every family, holds on all three fresh seeds without any parser in the replay path.

## Verdicts

Scores are family means at k = 12 for operations and the mean final-answer accuracy over the four held pairs for compositions, as in E16-F and E16-F2.

### P16F3.1: ratio to the twin at least 0.80 on all three families, holds in every seed

| Seed | mod-97 ops: learner / twin | Held comp: learner / twin | mod-73 ops: learner / twin | Verdict |
|---|---|---|---|---|
| 200003 | 90.4 / 94.9 = 0.95 | 81.8 / 93.7 = 0.87 | 88.9 / 92.6 = 0.96 | holds |
| 300007 | 91.9 / 94.1 = 0.98 | 79.5 / 91.5 = 0.87 | 91.3 / 92.8 = 0.98 | holds |
| 400009 | 87.1 / 95.3 = 0.91 | 82.5 / 94.0 = 0.88 | 91.3 / 94.3 = 0.97 | holds |

The standing seeds gave 0.98 / 0.96 / 0.87 on operations, 0.86 / 0.81 / 0.85 on compositions and 0.98 / 0.98 / 0.92 on later operations.

### P16F3.2: within 5 points of E16-F2 on every family, holds in 200003 and 300007, misses in 400009

| Seed | mod-97 ops: raw / parsed (diff) | Held comp: raw / parsed (diff) | mod-73 ops: raw / parsed (diff) | Verdict |
|---|---|---|---|---|
| 200003 | 90.4 / 91.9 (−1.6) | 81.8 / 80.5 (+1.3) | 88.9 / 91.6 (−2.7) | holds |
| 300007 | 91.9 / 89.0 (+2.9) | 79.5 / 79.8 (−0.3) | 91.3 / 92.0 (−0.7) | holds |
| 400009 | 87.1 / 89.2 (−2.1) | 82.5 / 77.0 (**+5.5**) | 91.3 / 91.0 (+0.3) | **misses** |

The band is two-sided, so a difference of 5.5 in either direction misses it. In 400009 the parsed arm lost 11.8 points on compositions from block 4 to block 6, the most of its three seeds. The raw arm lost 6.7 there. Both arms stood at 88.8–89.2 after block 4. Eight of nine comparisons are within 3 points. That matches the standing seeds, where every comparison was within 3.

### P16F3.3: at block 5, 1.00–1.10 MB and 56,000–59,000 distinct entries, holds in every seed

| Seed | Bytes held at block 5 | Distinct byte strings | By first family, for reading only: mod-97 / comp / mod-73 | Verdict |
|---|---:|---:|---|---|
| 200003 | 1,049,731 (1.050 MB) | 57,715 | 32,340 / 14,713 / 10,662 | holds |
| 300007 | 1,048,747 (1.049 MB) | 58,028 | 32,514 / 14,563 / 10,951 | holds |
| 400009 | 1,047,900 (1.048 MB) | 57,887 | 32,584 / 14,501 / 10,802 | holds |

MB here means 10^6 bytes, as in E16-R3's "1.05 MB". In binary units the reservoir holds 0.999–1.001 MiB. The reservoir held 65,536 entries at every block. Unlike E16-F2's `total` arm, this arm's settings record shows `cap: 65536`.

## Trajectory, for reading only

| Seed | Held comp after blocks 4 / 5 / 6 | Loss, block 4 to 6 | mod-97 ops after blocks 2 / 4 / 6 |
|---|---|---:|---|
| 200003 | 89.2 / 81.5 / 81.8 | 7.3 | 91.9 / 89.2 / 90.4 |
| 300007 | 89.2 / 81.8 / 79.5 | 9.7 | 92.6 / 89.9 / 91.9 |
| 400009 | 89.2 / 82.6 / 82.5 | 6.7 | 88.3 / 84.4 / 87.1 |

All three seeds meet E16-F4's retention clause (at least 75 after block 6, within 12 of block 4) and its acquisition clause (at least 85 after block 2, at least 80 after block 4), although neither clause was predicted for E16-F3. Seed 400009 shows the same dip on the mod-97 operations during the composition blocks as in E16-F and E16-F2, here 75.4 at worst.

## Machine and pace

| Item | Value |
|---|---|
| Machine | as E16-F: 4 cores, Python 3.11.18, torch 2.15.0+cpu, NumPy 2.4.6, one thread per process |
| Harness | commit 26274316; no harness file changed by this job |
| CPU per life | 5,323 / 5,325 / 5,342 s |
| Wall | 00:42:47 to 02:38:41–02:40:18 UTC, about 1 h 57 min; six processes on four cores until about 02:03, when E16-F4's births ended, then three |
| Forecast | 3,300 CPU-s and about 80 minutes; **missed** |

**The cost forecast missed.** On the standing machine, verbatim replay was cheaper than the parsed arm, at 2,150 against 3,300 CPU-s. Here the two cost the same, 5,330 against 5,080. So the saving from skipping the re-rendering does not appear on this machine, and scaling the standing figure by the 1.54 machine factor was wrong. The likely reason is that the per-update cost here is dominated by the learner's update rather than by rendering, but this job did not measure that. This affected only the schedule. No setting was changed.

## Notes

- **Launch:** each life was detached with `setsid nohup`. Nothing was killed or restarted. All three exited 0.
- **Logs:** the logs were written to the session's scratch folder and copied to `exp45/logs/e16f3_raw65536_<seed>.log` at the end. The wall file is `exp45/logs/e16f3_lives_wall.txt`.
- **Shared table:** `exp45/SUMMARY_E16F3.md` holds every arm present for these seeds, so the E16-F and E16-F2 rows appear beside this arm's.
- **E16-F4:** the births ran at the same time as these lives and are reported in the E16-F4 report.

## Files

- **Predictions:** `E16F3_PREDICTIONS.md`.
- **Receipts:** `exp45/e16_life_T_raw65536_{200003,300007,400009}.json`.
- **Logs:** `exp45/logs/e16f3_raw65536_<seed>.log`, `exp45/logs/e16f3_lives_wall.txt`.
- **Tables:** `exp45/SUMMARY_E16F3.md`.
- **Review paragraph:** `E16F3_REVIEW_SECTION_9_43.md`.
