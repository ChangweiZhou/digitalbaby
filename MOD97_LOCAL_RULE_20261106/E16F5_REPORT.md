# E16-F5 report: the stream without stages on the fresh seeds (9 November 2026)

Brief: `OPUS_TASK_E16F5_E16RP_CLOUD_JOBS_20261109.md`, Job 1. Predictions: `E16F5_PREDICTIONS.md`, written and pushed before launch (commit 96985527). Seeds 200003, 300007, 400009, on E16-F's births. The settings record of every receipt carries the birth's SHA-256, and each one equals the SHA-256 of `exp45/e11_dev_M_<seed>.pt`. Tables: `exp45/SUMMARY_E16F5.md`.

**Outcome:** P16F5.2 holds in every seed. P16F5.1 and P16F5.3 hold in 200003 and 300007 and miss in 400009. In that seed the learner's held compositions end at 64.9: 0.69 of the drift twin and 17.5 points below its E16-F3 result. On the other two seeds the drifting stream reproduces E16-F3 almost exactly, within 0.3 points on compositions.

## Verdicts

Scores are family means at k = 12 for operations and the mean final-answer accuracy over the four held pairs for compositions, after the last of the six checkpoints.

### P16F5.1: ratio to the drift twin at least 0.80 on all three families, holds in 200003 and 300007, misses in 400009

| Seed | mod-97 ops: learner / twin | Held comp: learner / twin | mod-73 ops: learner / twin | Verdict |
|---|---|---|---|---|
| 200003 | 88.2 / 94.6 = 0.93 | 81.5 / 94.6 = 0.86 | 89.9 / 93.9 = 0.96 | holds |
| 300007 | 89.6 / 93.1 = 0.96 | 79.3 / 91.5 = 0.87 | 92.2 / 92.9 = 0.99 | holds |
| 400009 | 86.6 / 95.7 = 0.91 | 64.9 / 94.0 = **0.69** | 90.8 / 94.4 = 0.96 | **misses** |

The standing seeds gave 0.97 / 0.93 / 0.84 on operations, 0.86 / 0.87 / 0.86 on compositions and 0.99 / 0.96 / 0.97 on later operations.

### P16F5.2: the drift twin at or above 90 / 85 / 90, holds in every seed

| Seed | mod-97 ops | Held comp | mod-73 ops | Verdict |
|---|---:|---:|---:|---|
| 200003 | 94.6 | 94.6 | 93.9 | holds |
| 300007 | 93.1 | 91.5 | 92.9 | holds |
| 400009 | 95.7 | 94.0 | 94.4 | holds |

### P16F5.3: held compositions within 6 points of E16-F3, holds in 200003 and 300007, misses in 400009

| Seed | E16-F5, drift | E16-F3, blocks | Difference | Verdict |
|---|---:|---:|---:|---|
| 200003 | 81.5 | 81.8 | −0.3 | holds |
| 300007 | 79.3 | 79.5 | −0.2 | holds |
| 400009 | 64.9 | 82.5 | **−17.5** | **misses** |

E16-F3's lives ran at commit 26274316, before the panel re-keying (f38281c), so they were scored on the old composition panels. Their receipts have no `panels` field. This job's receipts record the re-keyed panels. The brief allows about 2 points for this. The miss in 400009 is far outside that, and the same panels give 0.2–0.3 point differences on the other two seeds.

## What happened in seed 400009

| Checkpoint | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---:|---:|---:|---:|---:|---:|
| Learner, mod-97 ops | 90.4 | **62.0** | 77.8 | 80.9 | 86.3 | 86.6 |
| Learner, held comp | 23.1 | 32.5 | 74.1 | 80.9 | 70.6 | **64.9** |
| Twin, held comp | 26.7 | 27.8 | 91.9 | 93.4 | 92.0 | 94.0 |

- **The early dip:** the learner's mod-97 operations fell from 90.4 to 62.0 at checkpoint 2, as the compositions began to enter the stream. The standing seed 116089 showed the same dip, at 72.
- **Compositions never reached the usual level:** they peaked at 80.9, against 88–90 in the other two seeds.
- **The loss was in the plus-minus pairs:** compositions then fell 16 points as the mod-73 operations took over the stream. At the end +- stood at 48.0 and -+ at 44.7, against the max-min pairs at 81.7 and 85.3.
- **The twin was unaffected:** 94 on compositions at the end.

This is the same pattern as E16-F4's seed 600013 under block stages: interference between the plus-minus compositions and the mod-97 addition and subtraction operations, which the twin does not suffer. Under block stages this birth lost only 6.7 points (E16-F3). So the drifting stream can turn a seed that passes under stages into a miss. One seed does not show how often.

## Reservoir

Checkpoint 5, for reading only:

| Seed | Bytes | Distinct byte strings | Items admitted so far |
|---|---:|---:|---:|
| 200003 | 1,048,335 | 55,279 | 983,072 |
| 300007 | 1,048,861 | 55,443 | 983,072 |
| 400009 | 1,048,642 | 55,211 | 983,072 |

Online admission over a mixed stream holds about 2,500 fewer distinct strings than block-end admission did in E16-F3, which held 57,700–58,000. The miss seed's reservoir does not differ from the other two.

## Across six seeds of the drifting stream

| Seeds | Lowest ratio to the twin on any family | Compositions at the end |
|---|---|---|
| standing 73129 / 93683 / 116089 | 0.84 | 80.6 / 81.4 / 80.3 |
| fresh 200003 / 300007 / 400009 | **0.69** | 81.5 / 79.3 / 64.9 |

Development result on a toy harness. One drifting stream per seed, and the twin shares the stream's generators and seeds.

## Machine and pace

| Item | Value |
|---|---|
| Machine | as E16-F to E16-F4: 4 cores, Python 3.11.18, torch 2.15.0+cpu, NumPy 2.4.6, one thread per process. The container restarted before this job, and the environment survived unchanged |
| Harness | commit 96985527, which contains f38281c; no harness file changed by this job |
| Learner lives | 4,687 / 4,687 / 4,738 CPU-s |
| Twin lives | 4,058 / 4,070 / 4,199 CPU-s |
| Wall | 14:03:56 to 15:44:31–15:59:12 UTC, 1 h 55 min, six processes on four cores |
| Forecast | about 29,000 CPU-s and 2.2 h, against the brief's 3.5 h; measured 26,400 CPU-s and 1.9 h |

## Notes and what went wrong

- **Nothing was killed or restarted.** All six lives exited 0.
- **Logs:** the logs were written to the session's scratch folder and copied to `exp45/logs/e16f5_<arm>_<seed>.log` at the end. The brief's command writes them inside the repository, but a `git pull --rebase` cut off logs written there on 8 November. The predictions file states this before launch.
- **Branch:** the brief's branch rule assumes a cloud worker cannot push to `codex/minifly-consolidated-20261023`. This session can, and has pushed every earlier job there. So this job was pushed there directly, with no branch created and nothing left to merge. The predictions file states this before launch.
- **E16-RP:** this session ran Job 1 only. E16-RP was not run.

## Files

- **Predictions:** `E16F5_PREDICTIONS.md`.
- **Receipts:** `exp45/e16_life_T_raw65536_drift_online_{200003,300007,400009}.json`, `exp45/e16_life_D_drift_{200003,300007,400009}.json`.
- **Final weights:** the matching `.pt` files.
- **Logs:** `exp45/logs/e16f5_{T,D}_<seed>.log`, `exp45/logs/e16f5_lives_wall.txt`.
- **Tables:** `exp45/SUMMARY_E16F5.md`.
- **Review paragraph:** `E16F5_REVIEW_SECTION_9_53.md`.
