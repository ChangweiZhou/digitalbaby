# Task for the cloud session: E16-F2, the bounded reservoir at 8,192 per family on the fresh seeds (after E16-F)

**Date:** 8 November 2026, 16:20 EST. Same repository, branch, folder and rules as `OPUS_TASK_E16F_FRESH_SEED_CONFIRMATION_20261108.md`; start only after E16-F's six lives and report are committed. **Purpose:** E16-R (`E45_REPORT.md`, section E16-R) found that a per-family reservoir of fixed capacity recovers retention monotonically with capacity on the three standing seeds; at 8,192 per family the learner is at 0.93 / 0.94 / 0.85 of the twin on the mod-97 operations, 0.96–0.97 on the mod-73 operations and 0.82 / 0.79 / 0.81 on the held compositions. Confirm that one arm on the three fresh seeds whose births and twin lives you already have.

## Step 1 (after `git pull`): three lives

```
E16_REPLAY=reservoir E16_CAP=8192 python3 exp16_integrated.py T <seed>
```

for seeds 200003, 300007, 400009. Each writes `exp45/e16_life_T_res8192_<seed>.json`; logs under `exp45/logs/e16f2_res8192_<seed>.log`. The twin lives are E16-F's `e16_life_D_<seed>.json`. Note: the harness was updated at commit 1890015 (receipt `settings` now record the optimizer's actual decay, the arm, the rule and the store flag, and the last block's buffer count); pull before running and record the commit you ran.

## Step 2: tables

```
E16_SEEDS=200003,300007,400009 E16_SUMMARY=SUMMARY_E16F2.md python3 summarize_exp16.py
```

## Predictions (copy into `E16F2_PREDICTIONS.md` before Step 1)

- P16F2.1 After block 6, in every fresh seed, the ratio to the twin is at or above 0.80 on the mod-73 operations at k = 12 and on the mod-97 operations at k = 12, and at or above 0.75 on the held compositions (final answer); standing seeds: 0.96–0.97, 0.85–0.94, 0.79–0.82.
- P16F2.2 Composition retention from block 4 to block 6 loses at most 20 points in every seed (standing: 11.0 / 14.2 / 14.5), and the reservoir's distinct composition count at block 5 is between 6,800 and 7,200.
- P16F2.3 Acquisition: mod-97 operations after block 2 at or above 85% and held compositions after block 4 at or above 80% in every seed.

Report holds / misses per seed with the numbers; no reruns, no tuning. Deliverables: `E16F2_PREDICTIONS.md`, `E16F2_REPORT.md`, the three receipts and logs, `exp45/SUMMARY_E16F2.md`, and one paragraph as `E16F2_REVIEW_SECTION_9_36.md` (do not edit the review). Add files only; pull before every push.

## Budget

Three lives, about 4.5 CPU-hours at your measured pace (T about 142 ms per update), roughly 1.7 hours of wall time on four cores.
