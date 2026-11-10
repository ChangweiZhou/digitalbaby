# E16-M4 predictions: the replay share on new seeds and where the curve turns (10 November 2026, 17:34 UTC, written before launch)

## Source and branch

- **Brief:** `OPUS_TASK_E16M4_CLOUD_JOB_20261110.md`.
- **Scope:** both halves (a and b) are run by this cloud session. It runs E19 on the same container at normal priority; these lives run under `nice -n 10`. That changes their wall time, not their CPU-seconds or results.
- **Harness:** `exp16_integrated.py` at ab0ae7ac, unchanged.
- **Branch:** `claude/vibrant-gauss-tops5f` (f1f79b65 at launch).

## Run identity

- **Machine:** `Linux vm 6.18.44-fc-v338 #1 SMP x86_64`, 4 cores.
- **Software:** Python 3.13.18, PyTorch 2.15.0 (CPU).

## Pre-launch checks

- **None of the twelve receipts exists.** That is `exp45/e16_life_T_raw65536_drift_online_rs75_<seed>.json` for 1100009, 1200007, 1300021, 1400017, 1500007, 1600033, and `..._rs88_` / `..._rs94_` for 400009, 600013, 300007.
- **Every comparator named in the brief is present:** the share-0.5 lives, the drift twins, and the births.

## Dry runs

`E11_STAGE_UPDATES=4`, run under `nice -n 10`. Each had seven stages, no error, and the right `rshare`, drift stream and online admission. Receipts and logs are in `exp45/archive_e16m4_dry/cloud_20261110/`.

| Shape | Seed | CPU-s |
|---|---|---:|
| Share 0.75 | 1100009 | 180 |
| Share 0.875 | 400009 | 224 |
| Share 0.9375 | 400009 | 235 |

## Runs

Twelve lives, all launched together and detached, at `nice -n 10`. The command is `E16_RSHARE=<share> E16_REPLAY=raw E16_CAP=65536 E16_ADMIT=online E16_STREAM=drift python3 exp16_integrated.py T <seed>`.
- **Half a:** share 0.75 on the six E16-F7 seeds.
- **Half b:** shares 0.875 and 0.9375 on 400009, 600013, 300007.

Logs are written outside the repository and copied to `exp45/logs/e16m4_T_rs<percent>_<seed>.log`.

## Predictions (copied from the brief; not to be changed)

All at the last checkpoint unless a checkpoint is named.

Half a:
- **P16M4.1 (family gate)** all six seeds pass the family gate at share 0.75, as they do at 0.5.
- **P16M4.2 (hard pairs)** at share 0.75 the hard pairs are at least 5 points above the same seed's share-0.5 life in at least five of the six seeds, and at least three of the six reach the hard-pair gate (one does at share 0.5).
- **P16M4.3 (the late loss)** in 1200007 and 1300021 the fall of `+-` from checkpoint 4 to checkpoint 6 (15.0 and 29.7 points at share 0.5) is at most half as large at share 0.75, in both seeds.

Half b:
- **P16M4.4 (still rising at 0.875)** in at least two of the three seeds the hard pairs at share 0.875 are at or above the share-0.75 value minus 3.
- **P16M4.5 (the curve turns by 0.9375)** at share 0.9375 at least two of the three seeds fail the family gate on some family, and in at least two the mod-73 mean is at least 5 points below its share-0.75 value: with four fresh items in a batch of 64 the last family is not acquired in time.

Reading, written before the result. P16M4.1 and P16M4.2 holding make share 0.75 the candidate default for this learner, to be confirmed on all eighteen drift seeds before any gate is quoted at it; it would remain a setting found on three seeds and tested on six. P16M4.2 missing says E16-M3's gain belonged to its three seeds, two of them chosen as failures, and the share is not a general handle. P16M4.3 says whether the late loss of 9.62 is the retention the share buys. P16M4.4 and P16M4.5 together place the turn between 0.875 and 0.9375; P16M4.5 missing says replay of recently admitted items carries acquisition too and the turn is higher still, which would be worth knowing for the cost per event. State which case obtained.

## How these are scored

As in E16-M3:
- **Family means:**
  - operations: mean exact accuracy at k = 12 over the family's four operations;
  - compositions: mean final-answer accuracy over `+-`, `-+`, `><`, `<>`.
- **Hard pairs** = the mean of `+-` and `-+`, final answer.
- **Family gate** = at least 0.80 of the drift twin on all three families.
- **Hard-pair gate** = the hard pairs at least 0.80 of the twin's hard pairs.
- **Checkpoints** are receipt stages.

## Forecast

**Inputs.** Sign-rule lives measured 5,132–5,441 CPU-s each on this container type in E16-M3.

| Item | Forecast |
|---|---|
| Twelve lives | about 64,000 CPU-s |
| E19 alongside | about 38,000 CPU-s |
| Both jobs together | about 102,000 CPU-s, or about 7 hours of wall time on four cores |
| These niced lives finish | about 00:45 UTC on 11 November |
