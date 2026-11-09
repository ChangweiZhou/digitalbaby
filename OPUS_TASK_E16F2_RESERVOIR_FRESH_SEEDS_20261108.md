# Task for the cloud session: E16-F2, the label-free reservoir at 65,536 on the fresh seeds (after E16-F)

**Amended 17:25 EST, before you start it:** E16-R2 (`E45_REPORT.md`, section E16-R2) found that a single label-free reservoir of 65,536 entries meets the gate in every standing seed (ratios to the twin: operations 0.97 / 0.96 / 0.88, compositions 0.88 / 0.81 / 0.83, later operations 0.99 / 0.98 / 0.95). Run that arm instead of the per-family 8,192 one. Everywhere below, read `E16_REPLAY=total E16_CAP=65536` for the command, `e16_life_T_tot65536_<seed>.json` for the receipt, `e16f2_tot65536_<seed>.log` for the log, and the standing-seed figures above for the comparators. The predictions become: P16F2.1 ratio to the twin at or above 0.80 on all three families in every fresh seed; P16F2.2 composition loss from block 4 to block 6 at most 15 points (standing: 5.4 / 11.9 / 11.8) and the reservoir's distinct count at block 5 between 27,500 and 29,500; P16F2.3 unchanged. Pull first (the mode exists from commit 72d3aca).


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

## E16-F3 (added 19:00 EST): after E16-F2, the raw-byte arm on the same fresh seeds

E16-R3 (`E45_REPORT.md`, section E16-R3) removed the parser: `E16_REPLAY=raw E16_CAP=65536` stores presented bytes verbatim and meets the gate on the standing seeds (ratios to the twin: operations 0.98 / 0.96 / 0.87, compositions 0.86 / 0.81 / 0.85, later operations 0.98 / 0.98 / 0.92). Run that arm on seeds 200003, 300007, 400009 after the E16-F2 lives finish: three lives, receipts `e16_life_T_raw65536_<seed>.json`, logs `e16f3_raw65536_<seed>.log`, launched detached as amended. Predictions (lock in `E16F3_PREDICTIONS.md` before launch): P16F3.1 ratio to the twin at or above 0.80 on all three families in every seed; P16F3.2 within 5 points of your own E16-F2 (`tot65536`) result on every family in every seed; P16F3.3 the reservoir's byte size at block 5 between 1.00 and 1.10 MB and its distinct count between 56,000 and 59,000. Tables with `E16_SEEDS=200003,300007,400009 E16_SUMMARY=SUMMARY_E16F3.md`. Deliver `E16F3_REPORT.md` and one paragraph as `E16F3_REVIEW_SECTION_9_43.md`. Verbatim replay is faster than the parsed arm (no rendering), about 2,150 CPU-seconds per life on the standing machine.

## E16-F4 (added 19:50 EST): the gate result on three more fresh seeds, after E16-F3

The headline result (E16-R3 / E16-F3: the raw-byte reservoir meets the Phase A gate) should rest on nine seeds, as the composition gate does since E14-S. Three new seeds: **500011, 600013, 700027** (check that no `exp45/e11_dev_M_<seed>.*` exists; record it). Steps, all detached as amended, with partial receipts committed after every block:

1. Births: `python3 exp11_cell.py dev M <seed>` for the three seeds (in parallel; about 50 minutes of CPU each on your machine).
2. Lives, six in parallel: `E16_REPLAY=raw E16_CAP=65536 python3 exp16_integrated.py T <seed>` and `python3 exp16_integrated.py D <seed>`.
3. Tables: `E16_SEEDS=500011,600013,700027 E16_SUMMARY=SUMMARY_E16F4.md python3 summarize_exp16.py`.

Predictions (lock in `E16F4_PREDICTIONS.md` before the births): P16F4.1 births at or above 85% at k = 12 on all four development operations; P16F4.2 the twin at or above 90 / 85 / 90 on family means after block 6; P16F4.3 the raw-byte learner at or above 0.80 of the twin on all three families in every seed (standing: 0.98 / 0.96 / 0.87, 0.86 / 0.81 / 0.85, 0.98 / 0.98 / 0.92); P16F4.4 compositions after block 6 at or above 75% and within 12 points of their block-4 value in every seed (the margin widened from 10 to 12 after E16-F's one miss at 12.0, stated here before launch); P16F4.5 acquisition as in P16F.4. Deliver `E16F4_REPORT.md`, receipts, births (`git add -f` the `.pt` files as before), logs, `exp45/SUMMARY_E16F4.md`, and one paragraph as `E16F4_REVIEW_SECTION_9_45.md`. Budget about 11 CPU-hours on your machine, roughly four hours of wall time with six processes on four cores; forecast from your measured pace.
