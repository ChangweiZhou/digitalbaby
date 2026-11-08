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
