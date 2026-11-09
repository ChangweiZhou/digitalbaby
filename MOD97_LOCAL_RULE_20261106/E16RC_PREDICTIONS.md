# E16-RC predictions: capacity curve of the raw-byte reservoir on the standing seeds (9 November 2026, written before launch)

Brief: `OPUS_TASK_E16RC_RAW_CAPACITY_CURVE_20261108.md` (commit 7804abe). Harness at commit 7804abe; no harness file is changed by this job. Work is committed on branch `claude/vibrant-gauss-tops5f`, which is this session's assigned branch and is identical to the brief's commit (the brief names `codex/minifly-consolidated-20261023`, whose only later commit is a Codex design note; this session is not permitted to push there). No `exp45/e16_life_T_raw{16384,32768,131072}_<seed>.json` existed for any seed before launch.

Arms: `E16_REPLAY=raw E16_CAP={16384,32768,131072} python3 exp16_integrated.py T <seed>` for seeds 73129, 93683, 116089, on the standing births `exp45/e11_dev_M_<seed>.pt`. Nine lives, all launched together, detached with `setsid nohup`. Receipts `exp45/e16_life_T_raw<cap>_<seed>.json`; logs `exp45/logs/e16rc_raw<cap>_<seed>.log`. Comparators, not rerun: `e16_life_T_raw65536_<seed>.json` (E16-R3), `e16_life_T_all_balanced_<seed>.json` (every entry, E16-D), `e16_life_D_<seed>.json` (twin). Tables: `e16rc_tables.py` → `exp45/SUMMARY_E16RC.md`.

## Predictions (from the brief; not to be changed)

- **P16RC.1 (monotone)** After block 6, held compositions and mod-97 operations rise with capacity in every seed across 16,384 / 32,768 / 65,536 / 131,072.
- **P16RC.2 (the gate's lower edge)** At 16,384 the ratio to the twin on compositions is below 0.80 in every seed; at 32,768 it is at or above 0.80 in at least two seeds.
- **P16RC.3 (saturation)** At 131,072 the held compositions are within 3 points of the every-entry arm (84.5 / 79.8 / 83.3) in every seed, and the reservoir holds at most about 24,000 distinct composition byte strings (compositions carry no letters, so their distinct count is bounded by the taught triples).
- **P16RC.4 (bytes)** The reservoir's byte size at block 5 is 16 ± 1 bytes per slot at every capacity.

Measures as in the standing reports: operations = mean exact accuracy at k = 12 over the family's four operations; held compositions = mean final-answer accuracy over the four held pairs. "Rise" in P16RC.1 is read as strictly greater at each step; ties or drops are misses, reported with their size. The composition count in P16RC.3 is the receipt's `reservoir_by_family.F2` (distinct stored strings whose first byte is `c`) at block 5 and block 6; "about 24,000" is scored against 24,000 exactly, with the number given.

## Smoke job and forecast

`E11_STAGE_UPDATES=4 E16_REPLAY=raw E16_CAP=16384 python3 exp16_integrated.py T 73129`: seven stages, no error, 199 CPU-s, 254 s wall, four cores (`nproc` = 4), 15 GB RAM, Python 3 with PyTorch 2.15.0 (CPU) and NumPy 2.5.3, installed in this session. The receipt and log are in `exp45/archive_e16rc_dry/`. At four updates a block, each block cost about 31 CPU-s, almost all of it scoring, against about 20 s on the standing machine: this machine is about 1.5× slower. A standing raw life cost 2,150 CPU-s, so here about 3,300 CPU-s a life (more at 131,072 only through memory, which is negligible). Nine single-threaded processes on four cores: about 9 × 3,300 / 4 ≈ 2.1 hours of wall time if the cores share cleanly, 3 hours with contention. The brief's figure of six hours is the upper bound used for planning.
