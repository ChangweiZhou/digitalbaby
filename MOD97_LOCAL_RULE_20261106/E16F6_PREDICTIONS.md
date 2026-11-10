# E16-F6 predictions: the plus-minus tail, six more seeds of the drifting stream (9 November 2026, written before the births)

Brief: `OPUS_TASK_E16M2_E16F6_CLOUD_JOBS_20261109.md`, Job 2. This session ran Job 1 (E16-M2, commit 2af0fa00) and now runs Job 2, as the brief asks when only one worker is available. Harness at commit 2af0fa00, which contains 1470401b; no harness file is changed by this job.

## Before launch

- **Births:** no `exp45/e11_dev_M_<seed>.*` existed for 800011, 900007 or 1000003 (checked 18:13 UTC, 9 November).
- **Lives:** no `exp45/e16_life_T_raw65536_drift_online_<seed>.*` or `exp45/e16_life_D_drift_<seed>.*` existed for any of the six seeds.

## Runs

Phase 1, launched together and detached:
- the three births, `python3 exp11_cell.py dev M <seed>` for 800011, 900007, 1000003;
- the six lives of 500011, 600013 and 700027 (E16-F4's births).

Phase 2, when the births have finished: the same two lives for 800011, 900007 and 1000003, whether or not the births meet P16F6.1.

```
E16_REPLAY=raw E16_CAP=65536 E16_ADMIT=online E16_STREAM=drift python3 exp16_integrated.py T <seed>
E16_STREAM=drift python3 exp16_integrated.py D <seed>
```

Tables: `E16_SEEDS=500011,600013,700027,800011,900007,1000003 E16_SUMMARY=SUMMARY_E16F6.md python3 summarize_exp16.py`. The twelve-seed signature list and its correlation are added by a small read-only script of this job's own (`e16f6_signature.py`); the summarizer is not edited.

Logs are written to the session's scratch folder and copied to `exp45/logs/e16f6_<arm>_<seed>.log` and `exp45/logs/e16f6_birth_<seed>.log`.

## Predictions (from the brief; not to be changed)

- **P16F6.1 (births)** Each new birth reaches at least 85% at k = 12 on all four development operations.
- **P16F6.2 (twin)** The drift twin is at or above 90 / 85 / 90 on the three family means after the last checkpoint in every seed. Seed 500011's twin under stages reached only 84.0 on compositions and was still rising; a repeat is a result.
- **P16F6.3 (rate)** At least four of the six seeds end at or above 0.80 of their drift twin on all three families.
- **P16F6.4 (seed 600013)** misses the composition clause under drift as it did under stages, below 0.80 of its twin on held compositions: a seed-level fragility, not a schedule-level one.
- **P16F6.5 (the early signature)** Over the twelve drift seeds (73129, 93683, 116089, 200003, 300007, 400009 from E16-S and E16-F5, and these six):
  - the Spearman rank correlation between the learner's mod-97 family mean at checkpoint 2 and its `+-` pair by final answer at checkpoint 6 is at least 0.5;
  - every seed below 70 at checkpoint 2 ends with `+-` below 60;
  - every seed above 80 at checkpoint 2 ends with `+-` above 65.

## Dry run

The two commands were run once on seed 500011 with `E11_STAGE_UPDATES=4`, at the same time. Each ran its seven stages without error, in about 4.6 min of wall time:

| Life | CPU-s |
|---|---:|
| Learner | 206 |
| Twin | 199 |

The receipts are in `exp45/archive_e16f6_dry/`.

A first attempt at 18:13 UTC was cut off:
- the learner's dry run was killed after four blocks by a restart of this session's container;
- the twin's dry run never started, because a second detached command in the same tool call lost its working directory.

The partial receipt was deleted, the attempt's log was kept in the scratch folder, and both dry runs were launched again from one script at 18:30 UTC.

## Forecast

The dry run measures the evaluation cost, so the forecast uses this machine's measured lives on the same commands (E16-F5) and births (E16-F4):

| Item | CPU-s |
|---|---:|
| Learner life | 4,690–4,740 |
| Twin life | 4,060–4,200 |
| Birth | about 3,100 (E16-F4's 500011: 3,122) |

**Phase 1:** nine processes on four cores. The births finish after about 7,000 s, about 1.9 h, when each of the six lives has used about 3,100 CPU-s.

**Phase 2:** twelve processes. The remaining work is about 6 × 1,300 + 26,500 ≈ 34,300 CPU-s, about 2.4 h on four cores.

**Expected:**
- about 62,000 CPU-s in all (17 CPU-hours);
- about 4.3 h of wall time, up to 5 h if oversubscription slows the processes;
- the six phase-1 lives finish at about 2.5 h;
- the phase-2 lives finish at about 4.3–5 h.

The brief's estimate is about 5 hours.

## If the container restarts

This session's container restarted once today (above), and a restart kills every process, detached or not. A life killed before it writes its seventh stage has not finished. Such a life will be launched again from its birth with the same command and seed, and the report will say so. Its partial receipt will be kept as `*_killed.json` beside the new one. A birth killed by a restart is treated the same way. No finished life is rerun.

## Push target

As for E16-F5 and E16-M2, this session can push to `codex/minifly-consolidated-20261023`, so it commits there, pulls with `--rebase` before each push and never forces. Partial receipts are pushed as blocks finish.
