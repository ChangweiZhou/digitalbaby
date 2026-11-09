# E16-M2 predictions: rule or data on the two failing seeds (9 November 2026, written before launch)

Brief: `OPUS_TASK_E16M2_E16F6_CLOUD_JOBS_20261109.md`, Job 1. This session runs Job 1. Harness at commit 1470401b, the commit the brief ships with, in which arm M admits online. No harness file is changed by this job. Before launch, no `exp45/e16_life_M_raw65536_drift_online_400009.json`, `e16_life_T_oracle_drift_400009.json`, `e16_life_M_raw65536_600013.json` or `e16_life_T_oracle_600013.json` existed.

## Runs

Four lives, detached and launched together:

```
E16_REPLAY=raw E16_CAP=65536 E16_ADMIT=online E16_STREAM=drift python3 exp16_integrated.py M 400009
E16_REPLAY=oracle E16_STREAM=drift python3 exp16_integrated.py T 400009
E16_REPLAY=raw E16_CAP=65536 python3 exp16_integrated.py M 600013
E16_REPLAY=oracle python3 exp16_integrated.py T 600013
```

Comparators, all in the repository:
- the failing learners `exp45/e16_life_T_raw65536_drift_online_400009.json` (E16-F5) and `exp45/e16_life_T_raw65536_600013.json` (E16-F4);
- their twins `exp45/e16_life_D_drift_400009.json` and `exp45/e16_life_D_600013.json`.

Logs go to `exp45/logs/e16m2_<arm>_<seed>.log`. They are written outside the repository and copied in at the end.

## Predictions (from the brief; not to be changed)

- **P16M2.1 (the data pipeline is not the cause)** `M` ends at or above 0.90 of the twin on held compositions in both seeds, with its `+-` and `-+` pairs at or above 75.
- **P16M2.2 (replay coverage is not the cause either)** `T` under oracle replay ends at or above 0.80 of the twin on held compositions in both seeds, with the `+-` pair at or above 65.
- **P16M2.3 (the signature belongs to the rule)** The mod-97 family mean dips below 70 when compositions arrive in `T` under oracle replay, and not in `M`. `M` stays above 85 at that checkpoint in both seeds. The checkpoint is checkpoint 2 under drift and block 3 under stages.

**Reading, written in the brief before the result:**
- If P16M2.1 holds and P16M2.2 misses, the failure is the sign rule's interference and not the replay data, and the next arm is on the rule.
- If both hold, the failure lies in what the raw reservoir held or how it was sampled in those seeds, and the next arm is on sampling.
- If `M` misses too, the data pipeline is the cause, whatever the rule.

The report will state which case obtained.

## Dry run

All four commands were run once with `E11_STAGE_UPDATES=4` on this machine, at the same time. Each ran its seven stages without error, in 4 min 52 s to 5 min 12 s of wall time and 215–232 CPU-s. That cost is almost entirely the seven panel evaluations. The dry receipts are in `exp45/archive_e16m2_dry/`, beside the coordinating session's dry runs on seed 73129.

## Forecast

The dry run measures the evaluation cost but not the training cost, so the forecast combines two sources:
- **This machine's measured lives:** the twin cost 4,060–4,510 CPU-s and the raw-byte learner 4,690–5,340.
- **Standing receipts:** on the standing machine `M` cost 1.03× the twin (4,272 / 4,321 against 4,163 / 4,191) and oracle `T` 1.42× the twin (5,907 / 5,904).

Expected here: `M` about 4,300 CPU-s and oracle `T` about 6,100 CPU-s per life, about 21,000 CPU-s in all. With four processes on four cores the wall time is set by the oracle lives, about 1.7 h. The brief's estimate is about 1.5 h.

## Push target

As for E16-F5, this session can push to `codex/minifly-consolidated-20261023`, so it commits there, pulling with `--rebase` before each push and never forcing.
