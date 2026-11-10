# E16-M2 report: rule or data on the two failing seeds (9 November 2026)

Brief: `OPUS_TASK_E16M2_E16F6_CLOUD_JOBS_20261109.md`, Job 1. Predictions: `E16M2_PREDICTIONS.md`, written and pushed before launch (commit b1c47414). Four lives on the births of the two failing seeds:
- arm `M` (the twin's backprop update on the learner's raw-byte reservoir pipeline);
- arm `T` under oracle replay (the sign rule on the twin's joint stream, no reservoir).

Every receipt's settings record carries the birth's SHA-256, and each equals the SHA-256 of `exp45/e11_dev_M_<seed>.pt`. Every receipt records the re-keyed panels. Tables: `exp45/SUMMARY_E16M2.md`, built by `e16m2_tables.py` (reads only).

**Outcome:**
- **P16M2.1 holds in both seeds.** With the learner's own data pipeline, backprop ends at 0.97 and 0.95 of the twin on held compositions, with the plus-minus pairs at 76.5–86.3.
- **P16M2.2 holds in 400009 and misses in 600013.** Oracle replay lifts the sign rule from 64.9 to 84.5 and from 64.7 to 78.8 on compositions. In 600013 the `+-` pair ends at 62.0, below the 65 bar, though the ratio clause (0.85) holds.
- **P16M2.3 misses in 400009 and holds in 600013.** Under oracle replay on the drifting stream there is no dip at checkpoint 2 (89.3). Under block stages the dip appears at block 3 (65.7). `M` stays above 85 at those checkpoints in both seeds (95.1 and 92.2).

## The reservoir is the same

At checkpoint 5, arm `M` held the same reservoir as the failing learner in each seed:

| Seed | Arm | Bytes | Distinct strings | By family (F1 / F2 / F3) |
|---|---|---:|---:|---|
| 400009 | failing `T` (E16-F5) | 1,048,642 | 55,211 | not recorded |
| 400009 | `M` | 1,048,642 | 55,211 | not recorded |
| 600013 | failing `T` (E16-F4) | 1,048,762 | 57,921 | 32,428 / 14,557 / 10,936 |
| 600013 | `M` | 1,048,762 | 57,921 | 32,428 / 14,557 / 10,936 |

Admission depends only on the stream, so the two arms admitted the same items. The update rule is the only difference between `M` and the failing learner, and backprop recovers the compositions on that reservoir.

## Verdicts

Scores after the last of six checkpoints:
- **Operations:** family mean at k = 12.
- **Compositions:** mean final-answer accuracy over the four held pairs.
- **Ratios:** to the twin of the same stream (E16-F5's `D_drift` for 400009, E16-F4's `D` for 600013).

### P16M2.1: `M` at or above 0.90 of the twin on compositions, with `+-` and `-+` at or above 75 — holds in both seeds

| Seed | `M` comp / twin | Ratio | `+-` | `-+` | Verdict |
|---|---|---:|---:|---:|---|
| 400009 (drift) | 91.3 / 94.0 | 0.97 | 83.0 | 86.3 | holds |
| 600013 (blocks) | 88.7 / 93.2 | 0.95 | 82.0 | 76.5 | holds |

`M`'s ratios on the other two families are 0.99 / 1.01 (400009) and 1.00 / 0.99 (600013).

### P16M2.2: oracle `T` at or above 0.80 of the twin on compositions, with `+-` at or above 65 — holds in 400009, misses in 600013

| Seed | Oracle `T` comp / twin | Ratio | `+-` | `-+` | Verdict |
|---|---|---:|---:|---:|---|
| 400009 (drift) | 84.5 / 94.0 | 0.90 | 74.2 | 77.5 | holds |
| 600013 (blocks) | 78.8 / 93.2 | 0.85 | **62.0** | 71.0 | **misses** (the `+-` clause) |

Oracle replay's composition ratios against the raw learner's:

| Seed | Oracle `T` | Raw `T` |
|---|---:|---:|
| 400009 | 0.90 | 0.69 |
| 600013 | 0.85 | 0.69 |

On operations, oracle `T` stands at 0.94 / 0.95 (400009) and 0.89 / 0.88 (600013). Better data helps the rule a great deal but does not bring it to the twin.

### P16M2.3: the mod-97 dip below 70 when compositions arrive, in oracle `T` and not in `M` — misses in 400009, holds in 600013

| Seed | Checkpoint | Oracle `T` mod-97 mean | `M` mod-97 mean | Failing `T` (comparator) | Verdict |
|---|---|---:|---:|---:|---|
| 400009 (drift) | 2 | **89.3** (no dip) | 95.1 | 62.0 | **misses** |
| 600013 (blocks) | 3 | 65.7 | 92.2 | 52.1 | holds |

**In 400009 under oracle replay**, the mod-97 family is not where the compositions arrive. Two things happen instead:
- **Checkpoint 1:** the family is low at 62.9, with add and sub at 34.8 and 33.5. This is before compositions are on the stream; the raw learner stood at 87.0 / 87.8 there. On the twin's joint stream the drift learner had not yet learned addition and subtraction when the first checkpoint was taken.
- **Checkpoint 3:** the family falls from 89.3 to 79.2, with sub at 62.2, as the compositions enter.

So the rule still loses mod-97 addition and subtraction when compositions arrive, but not below 70 in this seed.

## Which reading obtained

The brief's cases are written for both seeds at once. The seeds split:

- **Seed 600013 (blocks):** P16M2.1 holds and P16M2.2 misses (on the `+-` clause). This is the first case: the sign rule's interference, not the replay data. The signature is there too: the dip at block 3 appears under oracle replay (65.7) and not in `M` (92.2).
- **Seed 400009 (drift):** both hold. By the brief's wording this is the second case: the failure lies in what the raw reservoir held or how it was sampled, and the next arm is on sampling.

What both seeds share:
- **The data pipeline is not the cause in either seed.** `M` on an identical reservoir ends at 0.95–0.97 of the twin, with no dip.
- **The sign rule recovers only part of the gap with ideal data:** 84.5 and 78.8 against the twin's 94.0 and 93.2.
- **What differs between the seeds is how far oracle data closes the gap.** It closes most of it under drift (`+-` at 74.2) and less under stages (`+-` at 62.0).

So in 400009 the raw reservoir's sampling interacts with the rule: the same reservoir that backprop handles well is not enough for the sign rule, and better data rescues it. In 600013 the rule falls short even with oracle data. This is a mixed result on two seeds. It does not single out one account.

## Per-checkpoint tables

Mod-97 family mean (k = 12) and held-composition mean by checkpoint. The full per-operation and per-pair table, every checkpoint, is in `exp45/SUMMARY_E16M2.md`.

### Seed 400009, drifting stream

| Arm | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Failing `T` raw, mod-97 | 37.1 | 90.4 | 62.0 | 77.8 | 80.9 | 86.3 | 86.6 |
| `M` raw, mod-97 | 37.1 | 94.7 | 95.1 | 93.9 | 93.2 | 93.5 | 95.1 |
| Oracle `T`, mod-97 | 37.1 | 62.9 | 89.3 | 79.2 | 85.8 | 83.1 | 89.8 |
| Twin, mod-97 | 37.1 | 94.9 | 95.1 | 94.6 | 93.4 | 94.8 | 95.7 |
| Failing `T` raw, held comp | 24.2 | 23.1 | 32.5 | 74.1 | 80.9 | 70.6 | 64.9 |
| `M` raw, held comp | 24.2 | 27.3 | 43.5 | 94.1 | 93.6 | 91.5 | 91.3 |
| Oracle `T`, held comp | 24.2 | 20.2 | 29.5 | 75.5 | 83.7 | 82.1 | 84.5 |
| Twin, held comp | 24.2 | 26.7 | 27.8 | 91.9 | 93.4 | 92.0 | 94.0 |

Final pairs `+-` / `-+` / `><` / `<>`:

| Arm | `+-` | `-+` | `><` | `<>` |
|---|---:|---:|---:|---:|
| Failing `T` | 48.0 | 44.7 | 81.7 | 85.3 |
| `M` | 83.0 | 86.3 | 97.7 | 98.3 |
| Oracle `T` | 74.2 | 77.5 | 92.8 | 93.3 |
| Twin | 87.8 | 90.3 | 99.3 | 98.5 |

### Seed 600013, block stages

| Arm | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Failing `T` raw, mod-97 | 22.2 | 80.9 | 85.2 | 52.1 | 66.8 | 76.9 | 76.7 |
| `M` raw, mod-97 | 22.2 | 92.3 | 92.8 | 92.2 | 91.7 | 95.2 | 94.4 |
| Oracle `T`, mod-97 | 22.2 | 82.9 | 87.9 | 65.7 | 83.2 | 73.9 | 83.7 |
| Twin, mod-97 | 22.2 | 91.9 | 92.4 | 92.4 | 92.6 | 93.4 | 94.4 |
| Failing `T` raw, held comp | 25.2 | 20.0 | 26.8 | 67.0 | 81.2 | 62.8 | 64.7 |
| `M` raw, held comp | 24.6 | 26.3 | 26.3 | 87.7 | 93.1 | 88.9 | 88.7 |
| Oracle `T`, held comp | 24.6 | 22.8 | 11.1 | 64.0 | 77.9 | 79.1 | 78.8 |
| Twin, held comp | 25.2 | 28.0 | 29.5 | 86.7 | 92.6 | 92.6 | 93.2 |

Final pairs `+-` / `-+` / `><` / `<>`:

| Arm | `+-` | `-+` | `><` | `<>` |
|---|---:|---:|---:|---:|
| Failing `T` | 49.7 | 53.8 | 81.3 | 73.8 |
| `M` | 82.0 | 76.5 | 97.0 | 99.2 |
| Oracle `T` | 62.0 | 71.0 | 91.5 | 90.8 |
| Twin | 87.3 | 88.5 | 98.5 | 98.3 |

Oracle `T` under stages shows a second mod-97 drop at block 5 (73.9, add and sub at 52.2 / 53.8), as the mod-73 operations enter. The failing raw learner did not drop there.

## Panel caveat for seed 600013

- **Which panels:** seed 600013's comparators, the failing learner and the twin from E16-F4, ran before the panel re-keying (f38281c) and were scored on the old composition panels. The two new lives were scored on the re-keyed panels. Operation panels are unaffected (checkpoint 0 is identical, 22.2). The held pairs at birth differ slightly: 25.2 on the old panels, 24.6 on the re-keyed ones.
- **How much it can move:** E16-RP found the re-keying shifts single pairs by up to +6.8 points.
- **What it touches:** the composition ratios to the twin for 600013 (0.95 for `M`, 0.85 for oracle `T`) compare across panels. The absolute pair bars in P16M2.1 and P16M2.2 are read on the new lives' own panels and do not depend on the comparators.
- **Seed 400009:** all four lives are on re-keyed panels.

## Machine and pace

| Item | Value |
|---|---|
| Machine | as E16-F to E16-F5: 4 cores, Python 3.11.18, torch 2.15.0+cpu, NumPy 2.4.6, one thread per process |
| Harness | commit b1c47414 (contains 1470401b, in which `M` admits online); no harness file changed by this job |
| `M` lives | 3,871 (400009) / 3,874 (600013) CPU-s |
| Oracle `T` lives | 4,466 (400009) / 4,406 (600013) CPU-s |
| Wall | 16:55:48 to 18:01:22–18:11:25 UTC, 1 h 16 min, four processes on four cores |

The CPU figures are each log's last block line.

**Forecast against measured:**

| Item | Forecast | Measured |
|---|---|---|
| `M` per life | 4,300 CPU-s | 3,870 CPU-s |
| Oracle `T` per life | 6,100 CPU-s | 4,400–4,470 CPU-s |
| Total | 21,000 CPU-s | 16,617 CPU-s |
| Wall time | 1.7 h (the brief said 1.5 h) | 1.26 h |

The oracle forecast was scaled from the standing machine's 1.42× ratio to the twin. Here the ratio was about 1.1× against this machine's twins, so the oracle lives came in well under the forecast.

## Notes and what went wrong

- **Nothing was killed or restarted.** All four lives exited 0 (`exp45/logs/e16m2_lives_wall.txt`).
- **Logs:** written to the session's scratch folder and copied to `exp45/logs/e16m2_<arm>_<seed>.log` at the end, as the brief asks.
- **Branch:** pushed directly to `codex/minifly-consolidated-20261023`, as the brief allows for a session that can push there.
- **The forecast overestimated the cost by about a quarter.** See above.
- **The brief's reading does not fit a split result.** The two seeds land in different cases. The report states both rather than choosing one.

## Files

- **Predictions:** `E16M2_PREDICTIONS.md`.
- **Receipts:** `exp45/e16_life_M_raw65536_drift_online_400009.json`, `exp45/e16_life_T_oracle_drift_400009.json`, `exp45/e16_life_M_raw65536_600013.json`, `exp45/e16_life_T_oracle_600013.json`.
- **Final weights:** the matching `.pt` files.
- **Dry runs:** `exp45/archive_e16m2_dry/`.
- **Logs:** `exp45/logs/e16m2_{M_raw65536_drift_online,T_oracle_drift}_400009.log`, `exp45/logs/e16m2_{M_raw65536,T_oracle}_600013.log`, `exp45/logs/e16m2_lives_wall.txt`.
- **Tables:** `e16m2_tables.py` and `exp45/SUMMARY_E16M2.md`.
- **Review paragraph:** `E16M2_REVIEW_SECTION_9_58.md`.
