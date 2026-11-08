# E16-F predictions: the E16-D finding on fresh seeds (8 November 2026, written before Step 1)

Brief: `OPUS_TASK_E16F_FRESH_SEED_CONFIRMATION_20261108.md`. Harness unchanged: `exp11_cell.py`, `exp16_integrated.py`, `summarize_exp16.py`. Seeds 200003, 300007, 400009; no `exp45/e11_dev_M_<seed>.*` existed for any of them before this job (checked 8 November 2026, worktree of commit b023375).

## Predictions (copied from the brief; not to be changed)

- **P16F.1 (births)** Each fresh birth reaches at least 85% at k = 12 on all four development operations.
- **P16F.2 (twin)** After block 6 the twin is at or above 90% on the mod-97 operations at k = 12, 85% on the four held compositions (final answer), and 90% on the mod-73 operations at k = 12, in every seed.
- **P16F.3 (the learner with every-entry replay, the E16-D finding)** After block 6, in every seed: held compositions at or above 75% and within 10 points of their block-4 value; mod-97 operations within 10 points of their block-2 value; ratio to the twin at or above 0.80 on compositions and on the mod-73 operations. On the mod-97 operations the ratio is at or above 0.80 in at least two seeds (the standing seeds gave 0.94 / 0.95 / 0.82).
- **P16F.4 (acquisition)** After block 2 the learner is at or above 85% on the mod-97 operations at k = 12, and after block 4 at or above 80% on the held compositions, in every seed (standing seeds: 94 / 93 / 90 and 90 / 87 / 91).

## Deviation from Step 0, recorded before any run

The brief's smoke job runs on the existing birth `exp45/e11_dev_M_73129.pt`. Birth weights (`*.pt`) are excluded from the repository by `.gitignore`, so no standing birth is present in this checkout; only the births' `.json` summaries are. The smoke job therefore cannot run before Step 1 here. Order followed instead: Step 1 (the three fresh births) first; then the smoke job with the same command on the first finished fresh birth (`E11_STAGE_UPDATES=4 E16_REPLAY=all_balanced python3 exp16_integrated.py T 200003`), its receipt archived under `exp45/archive_e16f_dry/`; then the forecast, appended below before any life is launched. Nothing else in the brief changes.

## Machine

Cloud container, 4 cores (`nproc`), Python 3.11.18, PyTorch CPU wheel (version recorded in `E16F_REPORT.md`), one thread per process (set by the harness). Six lives will run concurrently on four cores, so per-process wall time is about 1.5× its CPU time.

## Smoke job and forecast (written after Step 1 and the smoke job, before any life)

**Births (Step 1):** done before the smoke job (see the deviation above). 2,940 / 2,959 / 3,096 CPU-s each, 3,140 s wall for the three in parallel on 4 cores.

**Smoke job:** `E11_STAGE_UPDATES=4 E16_REPLAY=all_balanced python3 exp16_integrated.py T 200003`, on the fresh birth 200003 (no standing birth weights in the repository). Finished without error, seven stages, 282 s wall on a 4-core machine with nothing else running. Receipt moved to `exp45/archive_e16f_dry/`. Per-block process CPU at four updates: 31–58 s, which is almost all scoring.

**Why the brief's formula is not used:** "one block ≈ 1,500 × the four-update block's CPU + 20 s" assumes the four-update block is mostly update time. Here that block is mostly scoring (about 30 s), so the formula would give about 13 CPU-hours per block. Instead the update cost was measured directly, with the qualification's own method (20 updates on a mixed F1/F2/F3 batch, fresh birth 200003): **arm T 141.8 ms per update, arm D 115.1 ms**. The standing machine measured 36.5 and 31 ms (`QUALIFY_E16.json`), so this machine is about 3.8× slower per update.

**Forecast:**

| | CPU per block | CPU per life (6 blocks) |
|---|---:|---:|
| T, all_balanced | 6,144 × 0.142 s + about 30 s scoring ≈ 900 s | about 1.5 h |
| D | 6,144 × 0.115 s + about 30 s ≈ 740 s | about 1.25 h |

* **Total:** 3 × (1.5 + 1.25) ≈ 8.2 CPU-hours for the six lives.
* **Wall time:** six processes share four cores. While all six run, each gets about two thirds of a core, so a T life takes about 2.3 h of wall time and a D life about 1.9 h. The job should therefore end about **2.3–2.6 hours** after launch, with a block every 22–25 minutes per T life at first.
