# Task for a cloud session: E16-F, the E16-D finding confirmed on fresh seeds

**Date:** 8 November 2026. **Repository:** `ChangweiZhou/digitalbaby`, branch `codex/minifly-consolidated-20261023`. Commit on that branch; create no branch. **Folder:** `MOD97_LOCAL_RULE_20261106/` (harness `exp11_cell.py`, `exp16_integrated.py`, `summarize_exp16.py`; shared `exp45.py`, `exp3_state.py`). **Needs:** Python 3 with PyTorch on CPU (no GPU is used; each process is single-threaded and the harness sets the thread variables itself). **Purpose:** E16 and E16-D (`E45_REPORT.md`, sections E16 and E16-D; `E45_PREDICTIONS.md`, same names) found that one cell learner under the local rule acquires every family of a three-family stream within 1–9 points of a backprop twin, loses earlier families when the stream changes form, and keeps them when the replay buffer holds every presented item instead of 1,024 per block (compositions 80–85% instead of 46–57%). Every setting in that chain was chosen on the same three seeds (73129, 93683, 116089). This job repeats the decisive arm and the twin on three seeds never used by this learner. It is a compute job on an existing harness: no new mechanisms, tasks, panels or settings, and no tuning.

## Step 0: smoke job, before anything else

From `MOD97_LOCAL_RULE_20261106/`:

```
E11_STAGE_UPDATES=4 E16_REPLAY=all_balanced python3 exp16_integrated.py T 73129
```

This runs the full six-block life with four updates per block on an existing birth (`exp45/e11_dev_M_73129.pt`), writes `exp45/e16_life_T_all_balanced_dry4_73129.json`, and prints one line per block ending in the process CPU time. It must finish without error and the receipt must have seven stages. Record wall time and the machine's core count. Move the receipt to `exp45/archive_e16f_dry/`. Then forecast the whole job from this pace (one block of 6,144 updates costs about 1,500× the four-update block's CPU plus about 20 s of scoring) and write the forecast in `E16F_PREDICTIONS.md` before any further run. If PyTorch is missing or the smoke job fails, report that and stop.

## Step 1: births

Three fresh seeds: **200003, 300007, 400009** (used by Codex for the window net in E14, never for the cell learner; check that no `exp45/e11_dev_M_<seed>.*` exists and record that).

```
python3 exp11_cell.py dev M 200003
python3 exp11_cell.py dev M 300007
python3 exp11_cell.py dev M 400009
```

Each is development by backprop through time on four operations mod 89 with 0–12 distractor letters, 24,576 updates; on the standing seeds it took about 31 CPU-minutes and reached 91–99% at k = 12 on the development operations. The three can run in parallel. Each writes `exp45/e11_dev_M_<seed>.pt` and `.json`.

## Step 2: lives

Six lives, all in parallel if the cores allow:

```
E16_REPLAY=all_balanced python3 exp16_integrated.py T <seed>      # the learner: sign rule + exact trace, decay 0.25, every presented entry replayed, balanced per family
python3 exp16_integrated.py D <seed>                             # the twin: backprop through time on the families seen so far, equal weight
```

for each of the three seeds. Each life is six blocks of 6,144 updates and writes `exp45/e16_life_T_all_balanced_<seed>.json` or `exp45/e16_life_D_<seed>.json` after every block (so a crashed life is visible from its receipt). On the standing machine a block took 13–19 minutes depending on load. Keep the stdout of every run under `exp45/logs/e16f_<arm>_<seed>.log`.

## Step 3: tables

```
E16_SEEDS=200003,300007,400009 E16_SUMMARY=SUMMARY_E16F.md python3 summarize_exp16.py
```

writes `exp45/SUMMARY_E16F.md`. Do not run it without `E16_SEEDS` (that would rewrite the standing summary).

## Predictions to test (copy into `E16F_PREDICTIONS.md` before Step 1; do not change them afterwards)

- P16F.1 (births) Each fresh birth reaches at least 85% at k = 12 on all four development operations.
- P16F.2 (twin) After block 6 the twin is at or above 90% on the mod-97 operations at k = 12, 85% on the four held compositions (final answer), and 90% on the mod-73 operations at k = 12, in every seed.
- P16F.3 (the learner with every-entry replay, the E16-D finding) After block 6, in every seed: held compositions at or above 75% and within 10 points of their block-4 value; mod-97 operations within 10 points of their block-2 value; ratio to the twin at or above 0.80 on compositions and on the mod-73 operations. On the mod-97 operations the ratio is at or above 0.80 in at least two seeds (the standing seeds gave 0.94 / 0.95 / 0.82).
- P16F.4 (acquisition) After block 2 the learner is at or above 85% on the mod-97 operations at k = 12, and after block 4 at or above 80% on the held compositions, in every seed (standing seeds: 94 / 93 / 90 and 90 / 87 / 91).

Report each as holds / misses with the number, every seed separately. A miss is a result, not a problem to fix.

## Rules

- Add files; edit none. The harness files, the standing receipts, `E45_REPORT.md`, `E45_PREDICTIONS.md` and the review are not to be changed. Your deliverables are `E16F_PREDICTIONS.md`, `E16F_REPORT.md` (the four verdicts, the summary tables, the per-seed birth scores, the measured pace and machine, anything that went wrong), the six receipts, the three births (`.pt` and `.json`), the logs, and `exp45/SUMMARY_E16F.md`.
- Write one paragraph for the review as a separate file `E16F_REVIEW_SECTION_9_32.md` (section 9.32 is reserved for this job; do not edit the review itself).
- `git pull` before every push; other sessions push to the same branch. Commit messages say what was run and what was found, in one line each.
- No tuning, no reruns of a finished life, no change of seeds. If a life crashes, keep its partial receipt and log, report it, and do not restart it with changed settings. If a birth fails its 85% prediction, still run its lives: that is a result.
- Do not install anything beyond PyTorch and NumPy. Do not write a memo or a design note.

## Budget

Three births (about 1.5 CPU-hours) and six lives (about 9 CPU-hours at the standing pace); with six or more cores, roughly three hours of wall time after the smoke job. State your own forecast from the smoke job's pace.

## Amendment, 8 November 2026 17:45 EST (after the three learner lives were killed in block 6 by the launcher's time limit)

The partial receipts are kept and are already informative (through block 5 the fresh seeds track the standing ones). To finish without hitting the limit again:

1. **Launch lives detached from the command that starts them**, so the tool's time limit cannot kill them: for each life run `nohup setsid env E16_REPLAY=... E16_CAP=... python3 exp16_integrated.py T <seed> > exp45/logs/<name>.log 2>&1 < /dev/null &` from a short command that returns at once, then poll the logs with short commands (one line per finished block). Never run a life in the foreground of a tool call and never `wait` on it inside one.
2. **Run both remaining sets together, six processes**: the three E16-F2 lives (`E16_REPLAY=total E16_CAP=65536`, the arm that now carries the gate; predictions as amended in `OPUS_TASK_E16F2_RESERVOIR_FRESH_SEEDS_20261108.md`) and the three E16-F reruns (`E16_REPLAY=all_balanced`) from scratch on the same births. Rename the killed receipts and logs with a `_killed` suffix first so the reruns do not overwrite them; the reruns must reproduce the killed receipts' first five blocks exactly (same seed, same code), which is a free parity check: report it.
3. Forecast from your measured pace: six processes on four cores, about 3.5 hours of wall time. Commit partial receipts after every block as you have been doing.
4. Pull first: the harness at commit 7703d6c adds the `total` and `raw` modes and the receipt-settings fix; it does not change the `all_balanced` path.

The E16-F predictions stand unchanged; score them on the reruns. If a rerun is killed again, report the limit you observed and stop rather than restart.
