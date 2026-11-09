# E16-F report: the E16-D finding on three fresh seeds (8–9 November 2026)

Brief: `OPUS_TASK_E16F_FRESH_SEED_CONFIRMATION_20261108.md`, with its amendment of 8 November 17:45 EST. Predictions: `E16F_PREDICTIONS.md`, written before Step 1 and not changed since. Seeds 200003, 300007, 400009; no `exp45/e11_dev_M_<seed>.*` existed for them before this job. Tables: `exp45/SUMMARY_E16F.md`.

**Outcome:** three of the four predictions hold in every seed. P16F.3, the E16-D finding itself, holds in two seeds of three. It misses in seed 300007 by one clause: the held compositions fell 12.0 points from block 4 to block 6, against a limit of 10.

## Verdicts

Scores are exact answers on the sealed panels, as family means at k = 12 for operations and as the mean final-answer accuracy over the four held pairs for compositions. These are the means the summarizer and the standing E16 reports use.

### P16F.1 (births): holds in every seed

Each birth must reach at least 85% at k = 12 on all four development operations.

| Seed | add89 | sub89 | max89 | min89 | Verdict |
|---|---:|---:|---:|---:|---|
| 200003 | 92.0 | 91.0 | 98.7 | 98.7 | holds |
| 300007 | 92.0 | 92.0 | 98.0 | 98.7 | holds |
| 400009 | 90.0 | 92.0 | 99.3 | 99.0 | holds |

### P16F.2 (twin): holds in every seed, on family means

The twin must reach at least 90 on the mod-97 operations, 85 on the held compositions and 90 on the mod-73 operations after block 6.

| Seed | mod-97 ops | Held comp | mod-73 ops | Verdict |
|---|---:|---:|---:|---|
| 200003 | 94.9 | 93.7 | 92.6 | holds |
| 300007 | 94.1 | 91.5 | 92.8 | holds |
| 400009 | 95.3 | 94.0 | 94.3 | holds |

Read per operation instead of per family, a few single cells fall below the bars. They are add73 at 88 in seed 200003, add73 at 86 in seed 300007, and the -+ pair at 84.7 in seed 300007. The brief speaks of "the mod-97 operations" and "the four held compositions" as families, so the verdict above uses family means.

### P16F.3 (learner with every-entry replay): holds in 200003 and 400009, misses in 300007

| Clause | 200003 | 300007 | 400009 |
|---|---|---|---|
| Held comp after block 6 at least 75 | 83.3 | 77.0 | 81.5 |
| Held comp within 10 of block 4 | 88.7 to 83.3, −5.4 | 89.0 to 77.0, **−12.0** | 89.7 to 81.5, −8.2 |
| mod-97 ops within 10 of block 2 | 91.7 to 90.3, −1.4 | 93.8 to 86.4, −7.4 | 85.9 to 84.4, −1.5 |
| Ratio to twin on compositions, at least 0.80 | 0.89 | 0.84 | 0.87 |
| Ratio to twin on mod-73 ops, at least 0.80 | 0.99 | 0.98 | 0.98 |
| Ratio to twin on mod-97 ops | 0.95 | 0.92 | 0.89 |
| **Verdict** | **holds** | **misses** | **holds** |

The mod-97 ratio must reach 0.80 in at least two seeds. It does so in all three, so that clause holds. The standing seeds gave 0.94 / 0.95 / 0.82.

The miss in seed 300007 sits in the plus-minus pairs: +- went from 83.7 to 68.0 and -+ from 81.8 to 62.2 between blocks 4 and 6. The max-min pairs barely moved. The same seed also lost most on sub97, from 90.5 after block 2 to 71.8 after block 6. That pattern matches what the standing seeds lost first.

### P16F.4 (acquisition): holds in every seed

| Seed | mod-97 ops after block 2, at least 85 | Held comp after block 4, at least 80 | Verdict |
|---|---:|---:|---|
| 200003 | 91.7 | 88.7 | holds |
| 300007 | 93.8 | 89.0 | holds |
| 400009 | 85.9 | 89.7 | holds |

The standing seeds gave 94 / 93 / 90 and 90 / 87 / 91.

## Fresh seeds against standing seeds

| Learner, every-entry replay | Held comp, block 4 to 6 | mod-97 ops, block 2 to 6 | Ratios at 6: mod-97 · comp · mod-73 |
|---|---|---|---|
| standing 73129 / 93683 / 116089 | 90.2 to 84.5 (−5.7) · 87.4 to 79.8 (−7.6) · 91.4 to 83.3 (−8.1) | 93.7 to 89.8 · 92.5 to 89.2 · 90.2 to 77.7 | 0.94·0.91·0.99 · 0.95·0.87·0.97 · 0.82·0.89·0.95 |
| fresh 200003 / 300007 / 400009 | 88.7 to 83.3 (−5.4) · 89.0 to 77.0 (−12.0) · 89.7 to 81.5 (−8.2) | 91.7 to 90.3 · 93.8 to 86.4 · 85.9 to 84.4 | 0.95·0.89·0.99 · 0.92·0.84·0.98 · 0.89·0.87·0.98 |

Standing values are computed from the standing receipts `exp45/e16_life_{T_all_balanced,D}_<seed>.json` in the same way. The fresh seeds land in the same range as the standing ones: compositions 77–83 after block 6 against 80–85, and the learner at 0.84–0.99 of the twin on every family against 0.82–0.99. Seed 400009 shows the same dip on the mod-97 operations during the composition blocks that standing seed 116089 showed (85.9 to 71.5 after block 4, recovering to 84.4). The finding transfers. Composition retention is the weakest part of it, and one fresh seed of three falls outside the 10-point margin.

## Summary tables

`exp45/SUMMARY_E16F.md`, written by `E16_SEEDS=200003,300007,400009 E16_SUMMARY=SUMMARY_E16F.md python3 summarize_exp16.py`. The summarizer reads every arm it finds for these seeds, so its first table also shows the E16-F2 lives (`T_tot65536`) as they stood when it ran: after four or five blocks, still running. Those rows are not part of E16-F. The E16-F2 report will carry its own summary.

## Machine and pace

| Item | Value |
|---|---|
| Machine | cloud container, 4 cores, Python 3.11.18, torch 2.15.0+cpu, NumPy 2.4.6, one thread per process |
| Per-update cost measured before launch | arm T 141.8 ms, arm D 115.1 ms; standing machine 36.5 and 31 ms, so about 3.8× slower |
| Births | 2,940 / 2,959 / 3,096 CPU-s; 3,140 s wall for the three in parallel |
| Smoke job | 282 s wall, seven stages, no error |
| Twin lives | 3,841 / 3,836 / 3,918 CPU-s each, about 640 s per block; wall 1 h 55 min to 2 h 00 min, six processes on four cores |
| Learner lives, the reruns | 5,061 / 5,063 / 5,084 CPU-s each, about 845 s per block; wall 2 h 03 min, three processes alone for 15 minutes, then six with the E16-F2 lives |
| Forecast | 900 s per T block and 740 s per D block; measured 845 and 640, so the forecast was about 10% high on CPU and low on wall, because six processes on four cores cost more than two thirds of a core each |

The harness was used unchanged. The births ran at commit b023375 and the twin lives at e58bc35e. The learner reruns ran at commit 109c5afc, which contains 7703d6c; the amendment states that 7703d6c leaves the `all_balanced` path unchanged, and the parity check below confirms it.

## What went wrong

1. **No smoke job before the births.** Birth weights (`*.pt`) are ignored by git, so the standing birth 73129 named in Step 0 was not in this checkout. The smoke job ran on fresh birth 200003 after Step 1 instead, as recorded in `E16F_PREDICTIONS.md` before any run. The births' `.pt` files were added with `git add -f`.
2. **Birth logs cut off by a rebase.** The births wrote their logs inside the repository. Committing them and then running `git pull --rebase` replaced the files while the processes were still writing, so the processes kept writing to deleted files. The logs were recovered from the running processes' open file handles and are complete. All later lives wrote their logs outside the repository and were copied in at the end.
3. **The learner lives were killed in block 6.** The first launch ran the six lives as children of one background tool command, and that command has a two-hour limit. At 2 h 00 min after launch, about 22:21:53 UTC on 8 November, the limit stopped it and every process under it. The twin lives had already exited 0, between 22:16:57 and 22:21:31. The three learner lives had finished five of six blocks. Their partial receipts and logs are kept as `exp45/archive_e16f_killed/e16_life_T_all_balanced_<seed>_killed_after5.json` and `exp45/logs/e16f_T_all_balanced_<seed>_killed.log`.
4. **The learner lives were rerun from scratch with unchanged settings.** These were unfinished lives stopped by the launcher, not crashed or finished ones. Each rerun was detached from the tool with `setsid nohup`, per the amendment, and started at 22:22:41 UTC from the same births with the same command. All three exited 0 between 00:25:24 and 00:26:09 UTC on 9 November.
5. **Parity check passes.** For every seed, the reruns' stages 0 to 5 match the killed receipts exactly: every score on every panel, and every block's training loss. The verdicts above are scored on the reruns, and they equal what the first launch would have given.

## Files

- **Predictions:** `E16F_PREDICTIONS.md`.
- **Receipts:** `exp45/e16_life_T_all_balanced_{200003,300007,400009}.json`, `exp45/e16_life_D_{200003,300007,400009}.json`.
- **Births:** `exp45/e11_dev_M_{200003,300007,400009}.{pt,json}`.
- **Logs:** under `exp45/logs/`:
  - `e16f_birth_<seed>.log` and `e16f_births_wall.txt`;
  - `e16f_dry_all_balanced.log`;
  - `e16f_D_<seed>.log`;
  - `e16f_T_all_balanced_<seed>.log`, the reruns;
  - `e16f_T_all_balanced_<seed>_killed.log`;
  - `e16f_lives_wall_first_launch.txt` and `e16f_lives_wall_rerun.txt`.
- **Smoke receipt:** `exp45/archive_e16f_dry/`.
- **Killed partial receipts:** `exp45/archive_e16f_killed/`.
- **Tables:** `exp45/SUMMARY_E16F.md`.
- **Review paragraph:** `E16F_REVIEW_SECTION_9_32.md`.
