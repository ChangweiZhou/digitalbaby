# Task for a second cloud session: E16-RC, the capacity curve of the raw-byte reservoir on the standing seeds

**Date:** 8 November 2026, 20:10 EST. **Repository:** `ChangweiZhou/digitalbaby`, branch `codex/minifly-consolidated-20261023`; commit on it, no new branch. **Folder:** `MOD97_LOCAL_RULE_20261106/`. **Needs:** Python 3 with PyTorch on CPU. **Read first:** `OPUS_TASK_E16F_FRESH_SEED_CONFIRMATION_20261108.md` and its amendment (the rules there apply here, in particular: launch every life detached with `setsid nohup`, never in the foreground of a tool command; add files, edit none; pull before every push; no reruns of finished lives, no tuning). Another cloud session is running E16-F3 and E16-F4 on the fresh seeds; do not touch its files, and name yours as below so nothing collides.

**Purpose.** E16-R3 (`E45_REPORT.md`, section E16-R3) showed that one label-free reservoir of 65,536 slots holding presented bytes verbatim meets the Phase A gate on the standing seeds; E16-R showed, for the earlier parsed per-family reservoir, that retention is roughly logarithmic in coverage. The raw arm's curve exists at one point. Measure it at three more capacities, so the claim "about a megabyte of observed experience suffices" has its curve, and so the smallest sufficient budget is known.

## Runs

The standing births are in the repository (`exp45/e11_dev_M_{73129,93683,116089}.pt`, added 8 November). Smoke job first, as the E16-F brief describes (four updates per block, one life; record pace and cores; move the dry receipt to `exp45/archive_e16rc_dry/`). Then nine lives, detached:

```
E16_REPLAY=raw E16_CAP=16384  python3 exp16_integrated.py T <seed>
E16_REPLAY=raw E16_CAP=32768  python3 exp16_integrated.py T <seed>
E16_REPLAY=raw E16_CAP=131072 python3 exp16_integrated.py T <seed>
```

for seeds 73129, 93683, 116089. Receipts `exp45/e16_life_T_raw{16384,32768,131072}_<seed>.json`; logs `exp45/logs/e16rc_raw<cap>_<seed>.log`. Comparators, not rerun: `exp45/e16_life_T_raw65536_<seed>.json` (E16-R3), `exp45/e16_life_T_all_balanced_<seed>.json` (every entry, E16-D), `exp45/e16_life_D_<seed>.json` (twin). Nine lives on four cores: about 2,150 CPU-seconds each on the standing machine, so roughly 2.3 CPU-hours each on yours and about six hours of wall time with nine processes; forecast from the smoke job.

## Tables

`summarize_exp16.py` does not yet list the new arms. Add nothing to it; instead write a small script `e16rc_tables.py` (yours, new file) that reads the twelve raw receipts and the comparators and prints, per seed and capacity: the reservoir's distinct count and byte size at block 5; after block 6 the mod-97 operations at k = 12 (family mean), the held compositions (mean final-answer accuracy over the four held pairs), the mod-73 operations at k = 12, and each as a ratio to the twin; and the composition loss from block 4 to block 6. Save as `exp45/SUMMARY_E16RC.md`.

## Predictions (lock in `E16RC_PREDICTIONS.md` before launch)

- P16RC.1 (monotone) After block 6, held compositions and mod-97 operations rise with capacity in every seed across 16,384 / 32,768 / 65,536 / 131,072.
- P16RC.2 (the gate's lower edge) At 16,384 the ratio to the twin on compositions is below 0.80 in every seed; at 32,768 it is at or above 0.80 in at least two seeds.
- P16RC.3 (saturation) At 131,072 the held compositions are within 3 points of the every-entry arm (84.5 / 79.8 / 83.3) in every seed, and the reservoir holds at most about 24,000 distinct composition byte strings (compositions carry no letters, so their distinct count is bounded by the taught triples).
- P16RC.4 (bytes) The reservoir's byte size at block 5 is 16 ± 1 bytes per slot at every capacity.

Report holds / misses per seed with the numbers. Deliver `E16RC_PREDICTIONS.md`, `E16RC_REPORT.md`, the nine receipts and logs, `e16rc_tables.py`, `exp45/SUMMARY_E16RC.md`, and one paragraph as `E16RC_REVIEW_SECTION_9_46.md` (do not edit the review). Commit partial receipts after every block.
