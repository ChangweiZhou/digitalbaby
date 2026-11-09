# E16-F4 report: the raw-byte gate on three more fresh seeds (9 November 2026)

Brief: `OPUS_TASK_E16F2_RESERVOIR_FRESH_SEEDS_20261108.md`, section E16-F4 (added 19:50 EST). Predictions: `E16F4_PREDICTIONS.md`, written and pushed before the births (commit 26274316). Seeds 500011, 600013, 700027, new to this repository. Arms:
- **Learner:** `E16_REPLAY=raw E16_CAP=65536 python3 exp16_integrated.py T <seed>`.
- **Twin:** `python3 exp16_integrated.py D <seed>`.

Tables: `exp45/SUMMARY_E16F4.md`.

**Outcome:** only seed 700027 meets every prediction.

| Prediction | Holds in | Misses in |
|---|---|---|
| P16F4.1 (births) | every seed | none |
| P16F4.5 (acquisition) | every seed | none |
| P16F4.2 (twin) | 600013, 700027 | 500011, by 1.0 point on compositions |
| P16F4.3 (gate) | 500011, 700027 | 600013, at 0.69 of the twin on compositions |
| P16F4.4 (retention) | 700027 | 500011, 600013 |

Seed 600013 is the first seed, of nine for the label-free reservoir, where the learner falls below 0.80 of the twin on any family.

## Verdicts

Scores are family means at k = 12 for operations and the mean final-answer accuracy over the four held pairs for compositions, as in E16-F to E16-F3.

### P16F4.1 (births): holds in every seed

| Seed | add89 | sub89 | max89 | min89 | Verdict |
|---|---:|---:|---:|---:|---|
| 500011 | 92.3 | 92.0 | 96.7 | 97.7 | holds |
| 600013 | 91.3 | 89.0 | 96.0 | 97.3 | holds |
| 700027 | 92.7 | 91.7 | 95.3 | 97.0 | holds |

The scores are exact answers out of 300 at k = 12 on the sealed panel, after the last stage of birth.

### P16F4.2 (twin): holds in 600013 and 700027, misses in 500011

| Seed | mod-97 ops, at least 90 | Held comp, at least 85 | mod-73 ops, at least 90 | Verdict |
|---|---:|---:|---:|---|
| 500011 | 94.4 | **84.0** | 92.4 | **misses** |
| 600013 | 94.4 | 93.2 | 94.5 | holds |
| 700027 | 94.8 | 93.2 | 94.1 | holds |

In seed 500011 the twin learned the plus-minus compositions slowly. Its held-composition mean went 66.1 / 74.1 / 80.2 / 84.0 over blocks 3 to 6, and was still rising when the stream ended. The -+ pair stood at 34.8 after block 4 and 62.5 after block 6, and the +- pair at 71.3 and 77.0. The max-min pairs were at 95–99 throughout. In the other five fresh seeds, the twin's compositions were at or above 86 after block 3.

### P16F4.3 (raw-byte learner): holds in 500011 and 700027, misses in 600013

| Seed | mod-97 ops: learner / twin | Held comp: learner / twin | mod-73 ops: learner / twin | Verdict |
|---|---|---|---|---|
| 500011 | 89.1 / 94.4 = 0.94 | 73.4 / 84.0 = 0.87 | 91.7 / 92.4 = 0.99 | holds |
| 600013 | 76.7 / 94.4 = 0.81 | 64.7 / 93.2 = **0.69** | 84.8 / 94.5 = 0.90 | **misses** |
| 700027 | 88.9 / 94.8 = 0.94 | 76.9 / 93.2 = 0.83 | 89.5 / 94.1 = 0.95 | holds |

In 500011 the ratio of 0.87 on compositions rests partly on the twin's low 84.0. Against a twin at 93, as in the other two seeds, the learner's 73.4 would be about 0.79.

In 600013 the learner's weakness is concentrated in addition and subtraction, in both the operations and the compositions.

| 600013 learner | After block 2 | After block 4 | After block 6 |
|---|---|---|---|
| add97 / sub97 | — | 50.0 / 43.2 | 60.5 / 59.0 |
| max97 / min97 | — | 86.5 / 87.2 | 92.2 / 95.0 |
| +- / -+ compositions | — | 65.2 / 76.3 | 49.7 / 53.8 |
| >< / <> compositions | — | 93.5 / 89.8 | 81.3 / 73.8 |
| mod-97 family mean | 85.2 | 52.1 after block 3 | 76.7 |

The learner's mod-97 family fell from 85.2 after block 2 to 52.1 after block 3, when the compositions arrived. That is the deepest dip seen in any seed; the earlier worst was 71.5, in seed 400009 under the every-entry arm. It recovered only to 76.7. The same seed's twin was unaffected: 92.4 after block 3 and 94.4 after block 6.

### P16F4.4 (retention): holds in 700027, misses in 500011 and 600013

| Seed | Held comp after block 4 | After block 6, at least 75 | Loss, at most 12 | Verdict |
|---|---:|---:|---:|---|
| 500011 | 86.7 | **73.4** | **13.3** | **misses**, both clauses |
| 600013 | 81.2 | **64.7** | **16.5** | **misses**, both clauses |
| 700027 | 87.8 | 76.9 | 10.9 | holds |

The margin of 12 points was widened from 10 before launch. Both misses would also miss at 15 on the floor of 75.

### P16F4.5 (acquisition): holds in every seed

| Seed | mod-97 ops after block 2, at least 85 | Held comp after block 4, at least 80 | Verdict |
|---|---:|---:|---|
| 500011 | 91.8 | 86.7 | holds |
| 600013 | 85.2 | 81.2 | holds |
| 700027 | 88.6 | 87.8 | holds |

Seed 600013 clears both bars by the smallest margins seen so far, 0.2 and 1.2 points.

### Reservoir at block 5, for reading only

| Seed | Bytes | Distinct byte strings | By first family: mod-97 / comp / mod-73 |
|---|---:|---:|---|
| 500011 | 1,048,799 | 57,813 | 32,450 / 14,539 / 10,824 |
| 600013 | 1,048,762 | 57,921 | 32,428 / 14,557 / 10,936 |
| 700027 | 1,046,904 | 57,875 | 32,486 / 14,610 / 10,779 |

These match E16-F3 and E16-R3. The weak seed's reservoir holds the same mix as the others, so the 600013 miss is not a coverage difference in the reservoir.

## Across the nine seeds of the raw or label-free reservoir

| Seeds | Arm | Lowest ratio to twin on any family | Composition after block 6 |
|---|---|---|---|
| standing 73129 / 93683 / 116089 | raw (E16-R3) | 0.81 | 80.2 / 74.8 / 80.1 |
| fresh 200003 / 300007 / 400009 | raw (E16-F3) | 0.87 | 81.8 / 79.5 / 82.5 |
| fresh 500011 / 600013 / 700027 | raw (E16-F4) | **0.69** | 73.4 / 64.7 / 76.9 |

Composition retention was already the family with the least headroom. On this batch of seeds it fails in two of three, and in one seed the loss spreads to the mod-97 addition and subtraction operations. The gate in P16F4.3 holds in eight of nine seeds overall. This is a development result on a toy harness with one stream; it is descriptive, not tested.

## Machine and pace

| Item | Value |
|---|---|
| Machine | as E16-F: 4 cores, Python 3.11.18, torch 2.15.0+cpu, NumPy 2.4.6, one thread per process |
| Harness | births at commit 26274316, lives at commit 6433e23a; no harness file changed by this job |
| Births | 00:42:47 to 01:59:48–02:03:15 UTC, run beside E16-F3's three lives; all exit 0 |
| Learner lives | 5,086 / 5,114 / 5,094 CPU-s |
| Twin lives | 4,487 / 4,512 / 4,430 CPU-s |
| Wall for the lives | 02:41:08 to 04:35:32–04:45:27 UTC, about 2 h 04 min |
| Forecast | 3,300 CPU-s per learner life, 3,840 per twin life, 1.6 h of wall; **missed** |

The learner's forecast missed for the reason given in the E16-F3 report: verbatim replay is not cheaper than the parsed arm on this machine. The twin cost about 15% more than E16-F's twins, with six processes sharing four cores throughout. This affected only the schedule.

## Notes

- **Launch:** every process was detached with `setsid nohup`. Nothing was killed or restarted. All nine processes exited 0.
- **Waiters:** two background waiters reached their own time limits while the lives ran. They only read the logs, and a third saw the lives exit.
- **Logs:** the logs were written to the session's scratch folder and copied into `exp45/logs/` at commits.
- **Partial receipts:** these were committed after blocks 3–5.

## Files

- **Predictions:** `E16F4_PREDICTIONS.md`.
- **Births:** `exp45/e11_dev_M_{500011,600013,700027}.{pt,json}`.
- **Receipts:** `exp45/e16_life_T_raw65536_{500011,600013,700027}.json`, `exp45/e16_life_D_{500011,600013,700027}.json`.
- **Logs:** under `exp45/logs/`:
  - `e16f4_birth_<seed>.log` and `e16f4_births_wall.txt`;
  - `e16f4_T_raw65536_<seed>.log`;
  - `e16f4_D_<seed>.log`;
  - `e16f4_lives_wall.txt`.
- **Tables:** `exp45/SUMMARY_E16F4.md`.
- **Review paragraph:** `E16F4_REVIEW_SECTION_9_45.md`.
