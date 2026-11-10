# E16-M3 predictions: the replay share, is the reservoir's cost to the rule a dose (10 November 2026, 03:20 UTC, written before launch)

## Source, branch and scope

- **Brief:** `OPUS_TASK_E16F7_E16M3_CLOUD_JOBS_20261109.md`, Job 2 (commit e78aa83e).
- **Halves:** this session runs both halves, so the job is not split.
  - Half a is the nine sign-rule lives.
  - Half b is the five backprop lives.
  - The review paragraph takes the lower number, 9.64.
- **Harness:** `exp16_integrated.py` at e78aa83e, which adds `E16_RSHARE`. No harness file is changed by this job.
- **Branch:** `claude/vibrant-gauss-tops5f` at 691a9959. That is the consolidated branch merged into this session's branch, keeping the consolidated version on every conflict. Every `.py` file is identical to e78aa83e. This session cannot push to the consolidated branch.

## Run identity

- **Machine:** `Linux vm 6.18.44-fc-v338 #1 SMP x86_64`.
- **Cores:** 4.
- **Software:** Python 3.13.18, PyTorch 2.15.0 (CPU).

## Pre-launch check

No receipt `exp45/e16_life_{T,M}_raw65536_drift_online_rs{0,25,75}_<seed>.json` existed before launch, and neither did `exp45/e16_life_M_raw65536_drift_online_600013.json`.

## Dry runs

All six command shapes were dry-run with `E11_STAGE_UPDATES=4`. Each ran seven stages with no error, and each receipt's settings show the intended rshare, drift stream, online admission and rule.

| Command | CPU-s |
|---|---:|
| T at share 0, seed 400009 | 258 |
| T at share 0.25, seed 400009 | 250 |
| T at share 0.75, seed 400009 | 235 |
| M at share 0.25, seed 600013 | 279 |
| M at share 0.75, seed 600013 | 253 |
| M at share 0.5, seed 600013 | 258 |

All six ran together in 512 s of wall time. The receipts and logs are in `exp45/archive_e16m3_dry/cloud_20261110/`. They sit in a subfolder because the coordinating session's own dry runs already occupy the same file names in `exp45/archive_e16m3_dry/`.

## Runs

Fourteen lives on the drifting stream, all launched together and detached. Logs are written outside the repository and copied into `exp45/logs/` at commits and at the end.

- **Half a:** `E16_RSHARE={0,0.25,0.75} E16_REPLAY=raw E16_CAP=65536 E16_ADMIT=online E16_STREAM=drift python3 exp16_integrated.py T <seed>` for seeds 400009, 600013 and 300007.
- **Half b:**
  - `E16_RSHARE={0.25,0.75} ... M <seed>` for seeds 400009 and 600013;
  - `E16_REPLAY=raw E16_CAP=65536 E16_ADMIT=online E16_STREAM=drift python3 exp16_integrated.py M 600013`, at the default share of 0.5.

## Comparators (in the repository, not rerun)

- Share-0.5 learners: `exp45/e16_life_T_raw65536_drift_online_<seed>.json`.
- Twins: `exp45/e16_life_D_drift_<seed>.json`.
- Backprop at share 0.5 for 400009: `exp45/e16_life_M_raw65536_drift_online_400009.json` (E16-M2).

## Predictions (from the brief; not to be changed)

All at the last checkpoint. "Hard pairs" is the mean of `+-` and `-+` by final answer.

- **P16M3.1 (replay does work)** Share 0 ends below 0.80 of the twin on mod-97 operations in all three seeds, below the share-0.5 life by at least 10 points.
- **P16M3.2 (a dose on the rule)** In both failing seeds:
  - the rule's hard pairs at share 0.25 are at least 5 points above share 0.5, and at share 0.75 at least 5 points below share 0.5;
  - mod-97 operations move the other way, each by at least 3 points: 0.25 below 0.5, and 0.75 above 0.5.
- **P16M3.3 (backprop flat)** Every `M` life ends at or above 0.90 of the twin on held compositions, with its hard pairs within 5 points of the `M` life at share 0.5 of the same seed.
- **P16M3.4 (the passing seed)** 300007 passes the family gate at shares 0.25 and 0.75 both, and its hard pairs at 0.25 are above those at 0.75.

**Reading, from the brief:**
- If P16M3.2 and P16M3.3 both hold, the reservoir's cost to the rule is a dose of verbatim replay that backprop tolerates.
- If P16M3.2 misses with the hard pairs flat across shares, the share is not the handle.
- If P16M3.3 misses, the share matters to backprop too.
- The case that obtained will be stated per seed, and both cases if the seeds split.

## How these are scored

- **Family means** follow the summarizer:
  - operations: mean exact accuracy at k = 12 over the family's four operations;
  - compositions: mean final-answer accuracy over `+-`, `-+`, `><`, `<>`.
- **"Below the share-0.5 life by at least 10 points"** in P16M3.1 is measured on the mod-97 family mean, in points.
- **"The family gate"** in P16M3.4 is at least 0.80 of the twin on all three families.
- **The hard pairs at the same seed's share 0.5** in P16M3.3:
  - for 400009, E16-M2's receipt;
  - for 600013, this job's share-0.5 `M` life.
  - That `M` life is itself held to the 0.90 composition clause; its within-5 clause is trivially met.
- **Twins as references.** The drift twins are a joint-stream reference (errata of 9 November), and that is how they are read here.

## Forecast

**Inputs.**
- The dry runs cost 235–279 CPU-s each, the same as E16-F6's learner dry run (249 CPU-s), whose full life cost 4,955–5,119 CPU-s on this container type.
- A `T` life is therefore about 5,000 CPU-s.
- An `M` life is about 4,600 CPU-s: the brief gives 4,500 on E16-M2's container, and the dry run is slightly dearer here.

**Total and wall time.**
- The total is 9 × 5,000 + 5 × 4,600 ≈ 68,000 CPU-s, a lower bound of 4.7 hours on four cores.
- With all fourteen processes sharing the cores, they finish close together.

**Forecast:** launch about 03:25 UTC, all lives finished about 08:30 UTC, about 5 hours of wall time.
