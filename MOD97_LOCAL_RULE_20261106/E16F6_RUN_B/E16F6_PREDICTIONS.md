# E16-F6 predictions: the plus-minus tail, six more seeds of the drifting stream (9 November 2026, written before any birth or life)

## Source and branch

- **Brief:** `OPUS_TASK_E16M2_E16F6_CLOUD_JOBS_20261109.md`, Job 2 (commit 1470401b).
- **Harness:** `exp16_integrated.py` and `exp11_cell.py` at 1470401b. No harness file is changed by this job.
- **Branch:** `claude/vibrant-gauss-tops5f`, reset to `origin/codex/minifly-consolidated-20261023` at 1470401b. This session cannot push to the consolidated branch.
- **Job choice:** this session first started Job 1 (E16-M2) and ran its four dry runs only partly. They were stopped when the owner said another worker has Job 1. Their dry receipts were deleted uncommitted, and no E16-M2 life was launched.

## Pre-launch checks

- **No births exist for the new seeds.** As of 17:04 UTC, no `exp45/e11_dev_M_<seed>.*` exists for 800011, 900007 or 1000003.
- **No drift receipts exist for the reused seeds.** No `exp45/e16_life_T_raw65536_drift_online_<seed>.json` or `exp45/e16_life_D_drift_<seed>.json` exists for 500011, 600013 or 700027.

## Dry runs

Each passed with seven stages (a full birth curve for the birth) and no error. Receipts and logs are in `exp45/archive_e16f6_dry/`.

| Run | Command | CPU-s | Notes |
|---|---|---:|---|
| Learner | `E11_STAGE_UPDATES=4 E16_REPLAY=raw E16_CAP=65536 E16_ADMIT=online E16_STREAM=drift ... T 500011` | 249 | Receipt settings: stream drift, admit online |
| Twin | `E11_STAGE_UPDATES=4 E16_STREAM=drift ... D 500011` | 257 | |
| Birth | `E11_DEV_UPDATES=4096 python3 exp11_cell.py dev M 800011` | 499 | A sixth of the dose, all curve points written. The resulting `_dry4096` birth is archived and is not used by any life. |

All three ran together, in 518 s of wall time.

## Runs

**First wave**, launched together, detached with `setsid nohup`. Logs are written to the session scratch folder and copied into `exp45/logs/` at commits.
- The three births: `python3 exp11_cell.py dev M <seed>` for 800011, 900007, 1000003.
- Six lives:
  - `E16_REPLAY=raw E16_CAP=65536 E16_ADMIT=online E16_STREAM=drift python3 exp16_integrated.py T <seed>` for 500011, 600013, 700027;
  - `E16_STREAM=drift python3 exp16_integrated.py D <seed>` for the same seeds.

**Second wave.** A detached launcher waits for each birth's `.json` (written after its `.pt`) and its process to exit. It then starts that seed's learner and twin with the same commands, whatever the birth scored.

## Predictions (from the brief; not to be changed)

- **P16F6.1 (births)** Each new birth reaches at least 85% at k = 12 on all four development operations.
- **P16F6.2 (twin)** The drift twin is at or above 90 / 85 / 90 on the three family means after the last checkpoint, in every seed. Seed 500011's twin under stages reached only 84.0 on compositions and was still rising; a repeat is a result.
- **P16F6.3 (rate)** At least four of the six seeds end at or above 0.80 of their drift twin on all three families.
- **P16F6.4 (seed 600013)** It misses the composition clause under drift as it did under stages: below 0.80 of its twin on held compositions. That would make it a seed-level fragility, not a schedule-level one.
- **P16F6.5 (the early signature)** This one covers twelve drift seeds: 73129, 93683, 116089, 200003, 300007, 400009 from E16-S and E16-F5, plus these six.
  - The Spearman rank correlation between the learner's mod-97 family mean at checkpoint 2 and its `+-` pair (final answer) at checkpoint 6 is at least 0.5.
  - Every seed below 70 at checkpoint 2 ends with `+-` below 60.
  - Every seed above 80 at checkpoint 2 ends with `+-` above 65.

## How these are scored

- **Family means** follow the summarizer:
  - operations: mean exact accuracy at k = 12 over the family's four operations;
  - compositions: mean final-answer accuracy over `+-`, `-+`, `><`, `<>`.
- **"Checkpoint 2" and "checkpoint 6"** are receipt stages 2 and 6.
- **Spearman correlation:** average ranks for ties.
- **Panel mix in P16F6.5.** The three standing seeds' drift receipts (E16-S) were scored on the old held panels, where the `+-` triples are shared with the taught `->` pair. The other nine seeds are scored on the re-keyed panels. E16-RP measured the panel difference on the same weights at +1.0 / +6.8 / −0.8 on `+-`. The comparison is used as it stands, with this noted.

## Forecast, from the dry runs and E16-RP's measured pace on this container

**Inputs.**

| Run | CPU-s, full |
|---|---:|
| Learner life (E16-RP, three processes) | about 5,000 |
| Twin life (0.87 of a learner, the ratio on E16-F5's machine) | about 4,400 |
| Birth (six times the dry run) | about 3,000 |

The total is 3 × 3,000 + 6 × 5,000 + 6 × 4,400 ≈ 65,000 CPU-s, a lower bound of 4.5 hours on four cores.

**Waves.**
- **First wave:** nine processes on four cores, about 44% CPU each. The births finish around 1.9 hours after launch.
- **Second wave:** starts then. The first-wave lives end about 3.3 hours after launch, and the second wave about 5.3 hours after launch.

**Forecast:** launch about 17:10 UTC, all lives finished about 22:30 UTC, about 5.3 hours of wall time.
