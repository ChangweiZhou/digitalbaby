# E16-F7 report: six more drift seeds and the pre-registered signature S (10 November 2026)

Brief: `OPUS_TASK_E16F7_E16M3_CLOUD_JOBS_20261109.md`, Job 1, run whole (six births, twelve lives). Predictions: `E16F7_PREDICTIONS.md`, committed at d96129c7 before any birth started and not changed since. Harness at e78aa83e, unchanged.

## Verdicts

| Prediction | Verdict | In one line |
|---|---|---|
| P16F7.1 births ≥ 85 at k = 12 | **holds** | lowest 86.3 (sub89, 1400017) |
| P16F7.2 twin ≥ 90 / 85 / 90 | **holds** | lowest twin means 93.7 / 91.2 / 92.2 |
| P16F7.3 ≥ 4 of 6 at ≥ 0.80 on all three families | **holds** | 6 of 6; 1600033 by 0.005 on mod 97 |
| P16F7.4 the signature S | **misses** | threshold clauses hold; Spearman over eighteen seeds 0.109 < 0.35 |
| P16F7.5 hard pairs ≤ 2, family gate ≥ 4 | **holds** | 1 of 6 on `+-`/`-+`, 6 of 6 on the family gate |

**The reading the brief wrote beforehand applies word for word.** The correlation clause misses while the threshold clauses hold, so S separates failures but does not rank the passing seeds, as E16-F6 saw.

The threshold clauses had little to test in the new seeds:
- **Low clause:** no new seed fell below 65, so the six new seeds did not exercise it. Its only support remains the two failures that S was chosen on.
- **High clause:** four new seeds lie at or above 70, and all four pass.
- **Band:** two new seeds fall in the 65–70 band with no clause, and both pass. One of them, 1400017, has the best final `+-` of all eighteen.

## The six seeds

### Births (P16F7.1)

Accuracy (%) at k = 12 from `exp45/e11_dev_M_<seed>.json`, last curve point.

| Seed | add89 | sub89 | max89 | min89 | Lowest | CPU-s |
|---:|---:|---:|---:|---:|---:|---:|
| 1100009 | 92.3 | 90.0 | 97.3 | 99.0 | 90.0 | 3,051 |
| 1200007 | 92.0 | 93.3 | 99.0 | 98.0 | 92.0 | 3,107 |
| 1300021 | 93.0 | 95.0 | 97.7 | 100.0 | 93.0 | 3,055 |
| 1400017 | 90.7 | 86.3 | 96.3 | 97.7 | 86.3 | 3,164 |
| 1500007 | 89.7 | 92.0 | 98.0 | 98.7 | 89.7 | 3,055 |
| 1600033 | 89.3 | 93.0 | 97.0 | 98.7 | 89.3 | 3,126 |

### Final scores and ratios (P16F7.2, P16F7.3)

Family means at checkpoint 6: mod-97 operations at k = 12, held pairs by final answer, mod-73 operations at k = 12.

| Seed | Learner | Twin | Ratios | All three ≥ 0.80 |
|---:|---|---|---|---|
| 1100009 | 84.9 / 78.5 / 90.2 | 93.7 / 93.1 / 93.6 | 0.91 · 0.84 · 0.97 | yes |
| 1200007 | 85.8 / 74.3 / 89.2 | 94.4 / 91.2 / 93.4 | 0.91 · 0.81 · 0.96 | yes |
| 1300021 | 90.4 / 79.1 / 91.6 | 94.8 / 93.6 / 94.2 | 0.95 · 0.84 · 0.97 | yes |
| 1400017 | 80.8 / 83.5 / 89.8 | 94.0 / 92.6 / 93.2 | 0.86 · 0.90 · 0.96 | yes |
| 1500007 | 89.1 / 81.6 / 92.2 | 94.8 / 94.2 / 92.2 | 0.94 · 0.87 · 1.00 | yes |
| 1600033 | 77.3 / 76.0 / 86.4 | 96.1 / 93.3 / 93.8 | 0.805 · 0.81 · 0.92 | yes (by 0.005) |

Every twin is at or above 90 / 85 / 90.

### The hard pairs, apart from the family means (P16F7.5)

`+-` and `-+` pooled, final answer, checkpoint 6, from `e16f7_gate.py` at the receipts' commit:

| Seed | Learner | Twin | Ratio | ≥ 0.80 | Family gate |
|---:|---:|---:|---:|---|---|
| 1100009 | 65.3 | 88.2 | 0.740 | no | pass |
| 1200007 | 57.0 | 88.6 | 0.643 | no | pass |
| 1300021 | 64.0 | 88.0 | 0.727 | no | pass |
| 1400017 | 75.2 | 88.1 | 0.853 | **yes** | pass |
| 1500007 | 71.2 | 89.3 | 0.797 | no | pass |
| 1600033 | 61.8 | 90.1 | 0.685 | no | pass |

So 1 of the 6 seeds reaches 0.80 on the hard pairs, and 6 of 6 pass the family gate.

**All eighteen drift seeds:** 16 pass the family gate as pre-registered, 4 pass on `+-`/`-+` alone and 12 on the whole worked answer. The learner's mean on `+-`/`-+` is 64.8 against the twin's 87.5. On `><`/`<>` it is 90.3 against 98.1.

1200007 passes the family gate with a final `+-` of 49.7, about the level of the failing seed 400009 (48.0). The family gate passes it because `-+` (64.3), `><` and `<>` lift its held mean.

### Every checkpoint

Learner (T) and twin (D) family means, mod-97 / held / mod-73. Checkpoints:

| Checkpoint | Point in the life |
|---:|---|
| 0 | birth |
| 1, 2 | F1.1, F1.2 |
| 3, 4 | F2.1, F2.2 |
| 5, 6 | F3.1, F3.2 |

| Seed | Arm | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---:|---|---|---|---|---|---|---|---|
| 1100009 | T | 31.8/20.9/33.2 | 80.4/23.3/51.0 | 84.5/36.4/44.2 | 80.6/81.9/42.5 | 77.4/82.5/71.9 | 84.2/77.3/87.7 | 84.9/78.5/90.2 |
| | D | 31.8/20.9/33.2 | 93.4/23.8/43.1 | 94.4/19.5/41.2 | 93.8/92.5/22.2 | 95.1/93.7/21.8 | 95.0/93.5/92.5 | 93.7/93.1/93.6 |
| 1200007 | T | 22.1/20.7/24.2 | 88.7/24.1/48.8 | 85.6/27.2/35.7 | 85.0/84.1/39.6 | 81.8/83.5/77.8 | 85.3/67.9/88.3 | 85.8/74.3/89.2 |
| | D | 22.1/20.7/24.2 | 94.2/23.1/27.1 | 95.0/24.8/26.4 | 93.6/88.1/17.6 | 94.1/92.6/23.6 | 95.6/93.6/93.1 | 94.4/91.2/93.4 |
| 1300021 | T | 18.2/11.2/19.8 | 92.1/26.1/32.6 | 83.4/41.4/40.2 | 83.7/86.5/22.9 | 82.1/89.0/78.7 | 89.3/80.9/89.6 | 90.4/79.1/91.6 |
| | D | 18.2/11.2/19.8 | 94.5/15.8/32.7 | 94.9/14.0/21.2 | 91.9/91.0/3.6 | 95.9/93.9/2.9 | 93.9/93.5/92.2 | 94.8/93.6/94.2 |
| 1400017 | T | 26.9/20.9/27.1 | 81.9/16.2/33.6 | 82.2/29.0/17.9 | 65.2/86.1/35.2 | 66.7/87.6/64.6 | 76.2/81.9/84.1 | 80.8/83.5/89.8 |
| | D | 26.9/20.9/27.1 | 92.4/19.2/29.9 | 91.9/19.5/29.2 | 92.2/88.9/46.2 | 92.7/91.6/48.8 | 93.1/91.5/91.6 | 94.0/92.6/93.2 |
| 1500007 | T | 45.4/23.7/51.4 | 86.7/23.7/62.6 | 83.1/32.8/34.1 | 79.8/86.5/26.4 | 81.6/88.1/78.0 | 88.1/81.9/91.7 | 89.1/81.6/92.2 |
| | D | 45.4/23.7/51.4 | 93.2/28.6/57.0 | 94.0/27.2/55.1 | 92.4/92.7/63.7 | 93.8/94.3/70.1 | 93.6/93.5/91.6 | 94.8/94.2/92.2 |
| 1600033 | T | 19.4/14.4/21.6 | 87.2/19.3/33.7 | 82.5/24.5/28.4 | 68.4/80.8/29.4 | 67.0/84.6/64.1 | 75.2/75.8/84.7 | 77.3/76.0/86.4 |
| | D | 19.4/14.4/21.6 | 93.6/17.1/31.2 | 93.8/17.4/28.2 | 92.2/87.9/15.1 | 93.9/93.8/13.3 | 94.8/89.0/92.7 | 96.1/93.3/93.8 |

The learner's `+-` at each checkpoint (`-+` in brackets):

| Seed | 3 | 4 | 5 | 6 | Change from 4 to 6 |
|---:|---|---|---|---|---:|
| 1100009 | 65.3 (70.3) | 72.3 (67.5) | 61.7 (62.7) | 67.2 (63.5) | −5.1 |
| 1200007 | 69.0 (76.8) | 64.7 (76.8) | 35.5 (54.8) | 49.7 (64.3) | −15.0 |
| 1300021 | 76.7 (80.7) | 81.2 (84.8) | 60.0 (79.2) | 51.5 (76.5) | −29.7 |
| 1400017 | 80.2 (76.7) | 82.3 (76.2) | 71.3 (72.2) | 79.0 (71.3) | −3.3 |
| 1500007 | 78.3 (77.0) | 79.7 (80.5) | 73.0 (69.7) | 73.5 (68.8) | −6.2 |
| 1600033 | 66.5 (70.3) | 74.7 (77.5) | 64.0 (65.2) | 61.3 (62.2) | −13.4 |

The full per-operation, per-pair tables are in `exp45/SIGNATURE_E16F7.md` and `exp45/SUMMARY_E16F7.md`.

## The eighteen drift seeds sorted by S (P16F7.4)

From `e16f7_signature.py`, which writes `exp45/SIGNATURE_E16F7.md`.

| Seed | Source | S | Final `+-` | Final `-+` | Ratios to twin | Comp ≥ 0.80 |
|---:|---|---:|---:|---:|---|---|
| 600013 | E16-F6 | 47.2 | 41.7 | 47.8 | 0.77 · 0.71 · 0.91 | **no** |
| 400009 | E16-F5 | 62.0 | 48.0 | 44.7 | 0.90 · 0.69 · 0.96 | **no** |
| 1400017 | **E16-F7** | 65.2 | 79.0 | 71.3 | 0.86 · 0.90 · 0.96 | yes |
| 1000003 | E16-F6 | 66.7 | 68.2 | 61.0 | 0.81 · 0.83 · 0.94 | yes |
| 1600033 | **E16-F7** | 67.0 | 61.3 | 62.2 | 0.80 · 0.81 · 0.92 | yes |
| 800011 | E16-F6 | 69.3 | 63.8 | 63.8 | 0.87 · 0.83 · 0.95 | yes |
| 116089 | E16-S | 70.7 | 72.7 | 71.3 | 0.84 · 0.86 · 0.97 | yes |
| 700027 | E16-F6 | 77.3 | 69.7 | 66.3 | 0.94 · 0.87 · 0.97 | yes |
| 1100009 | **E16-F7** | 77.4 | 67.2 | 63.5 | 0.91 · 0.84 · 0.96 | yes |
| 1500007 | **E16-F7** | 79.8 | 73.5 | 68.8 | 0.94 · 0.87 · 1.00 | yes |
| 500011 | E16-F6 | 80.9 | 64.2 | 70.5 | 0.96 · 0.89 · 1.01 | yes |
| 900007 | E16-F6 | 81.7 | 63.2 | 70.7 | 0.93 · 0.83 · 0.97 | yes |
| 1200007 | **E16-F7** | 81.8 | 49.7 | 64.3 | 0.91 · 0.81 · 0.95 | yes |
| 300007 | E16-F5 | 82.0 | 65.0 | 70.3 | 0.96 · 0.87 · 0.99 | yes |
| 1300021 | **E16-F7** | 82.1 | 51.5 | 76.5 | 0.95 · 0.84 · 0.97 | yes |
| 93683 | E16-S | 86.2 | 69.8 | 70.0 | 0.93 · 0.87 · 0.96 | yes |
| 73129 | E16-S | 87.2 | 65.8 | 77.0 | 0.97 · 0.86 · 0.99 | yes |
| 200003 | E16-F5 | 87.4 | 67.2 | 72.5 | 0.93 · 0.86 · 0.96 | yes |

**Clauses:**
- S below 65 (600013 and 400009, both on file before this job): both end below 0.80 on compositions. **Holds**, but no new seed tested it.
- S at or above 70 (twelve seeds, four of them new): all end at or above 0.80. **Holds.**
- 65 to below 70 (no clause): 1400017, 1000003, 1600033 and 800011 all pass.
- Spearman between S and final `+-` over eighteen seeds: **0.109**, against the 0.35 required. **Misses.** It was 0.399 on the twelve seeds S was chosen on.

**Why the correlation falls.** The six new seeds rank almost in reverse (Spearman −0.714 among the six alone; exploratory, not a prediction):
- 1400017 has the second-lowest S of the new seeds (65.2) and the best final `+-` of all eighteen (79.0).
- 1200007 and 1300021 have S near 82 and the two lowest final `+-` among the passing seeds (49.7 and 51.5).

The per-checkpoint table shows where their `+-` went. Both were at 64.7 and 81.2 at checkpoint 4, the end of the composition blocks. They lost 15 and 30 points during the mod-73 blocks (checkpoints 5 and 6). S looks only at checkpoints 2–4, so it cannot see a loss that happens after them.

**Where the plus-minus loss happens differs by seed** (descriptive, from the receipts):
- In 600013 and 400009 the mod-97 operations dropped as the compositions entered, and the compositions ended low. S catches this.
- In 1200007 and 1300021 the mod-97 operations held, and `+-` was lost later, while the learner trained on the mod-73 family. Their `-+` fell less (12.5 and 8.3 points from checkpoint 4 to 6) and both seeds stay above the family gate. The loss shows in `+-` alone.

This is one stream per seed, and the second pattern is seen in two seeds. It is a description, not a finding about the mechanism.

## Run identity

| Item | Value |
|---|---|
| Machine | `Linux vm 6.18.44-fc-v338 #1 SMP @0 x86_64 GNU/Linux`; 4 cores; Intel Xeon at 2.10 GHz |
| Software | Python 3.11.18, torch 2.15.0+cpu, NumPy 2.4.6, one thread per process |
| Harness | e78aa83e (unchanged) |
| Predictions | d96129c7 |
| Births committed | 229b7a32 |
| Receipts committed | 41f2ba2a |

Each life's environment was checked from `/proc/<pid>/environ` while it ran:
- **Learner:** `E16_REPLAY=raw E16_CAP=65536 E16_ADMIT=online E16_STREAM=drift`.
- **Twin:** `E16_STREAM=drift` only.

### CPU-seconds of every birth and life

From the last `cpu` field of each log in `exp45/logs/e16f7_*`.

| Seed | Birth | Learner | Twin |
|---:|---:|---:|---:|
| 1100009 | 3,051 | 5,337 | 4,722 |
| 1200007 | 3,107 | 5,376 | 4,752 |
| 1300021 | 3,055 | 5,329 | 4,643 |
| 1400017 | 3,164 | 5,371 | 4,780 |
| 1500007 | 3,055 | 5,402 | 4,717 |
| 1600033 | 3,126 | 5,383 | 4,848 |
| **Total** | **18,558** | **32,198** | **28,462** |

All eighteen runs: **79,218 CPU-s**.

### Wall clock

Every process exited 0 (`exp45/logs/e16f7_lives_wall.txt`).

| Event | Time (UTC) |
|---|---|
| Births started | 03:14:26 |
| Births finished | 04:31:01–04:34:58 |
| Twins finished | 08:31:07–08:37:46 |
| Learners finished | 08:49:14–08:51:45 |

Wall time from the first birth to the last life: **5 h 37 min**.

## Forecast against measured

| Item | Forecast | Measured | Measured / forecast |
|---|---:|---:|---:|
| Birth CPU-s (mean) | about 3,050 | 3,093 | 1.01 |
| Learner life CPU-s (mean) | about 5,150 | 5,366 | 1.04 |
| Twin life CPU-s (mean) | about 4,500 | 4,744 | 1.05 |
| Total CPU-s | about 76,000 | 79,218 | 1.04 |
| Births phase | about 1.3 h | 1 h 21 min | about 1.04 |
| Wall time | about 5.3 h | 5 h 37 min | 1.06 |
| Last life done | about 08:45 | 08:51:45 | +7 min |

The brief's estimate was about 5.5 h. Run B's costs on a similar 2.10 GHz Xeon underestimated this container by 4–5% on the lives.

## What went wrong

- **No restarts, no killed runs.** Nothing was rerun or relaunched, and every life ran its seven stages in one go.
- **The forecast was 4–6% low.**
- **Two waiters timed out** before the lives finished. That was only my wait loop. Partial receipts were committed and pushed at 06:17 (88585392) and 08:07 (08eef429).
- **The low clause of P16F7.4 got no new test.** This is not an error, but it limits what the job shows: no new seed fell below 65.

## What this does not claim

These are development results on a toy harness, at evidence level E2, with one stream per seed.

- The "local" status of the sign rule is still open with Mr. Zhou (review section 9.60).
- No learning-core improvement is claimed.
- S is not supported as an early-warning ranking. It does separate the two known failures from the rest, but it was chosen on them.

## Files

| File | What it is |
|---|---|
| `E16F7_PREDICTIONS.md` | the predictions, written before launch |
| `E16F7_REVIEW_SECTION_9_62.md` | the review paragraph |
| `e16f7_gate.py` | a copy of the audit's `gate_by_item_type.py` with the six seeds added and `REF` set to `HEAD` |
| `e16f7_signature.py` | computes S over the eighteen seeds |
| `exp45/SUMMARY_E16F7.md`, `exp45/SIGNATURE_E16F7.md` | the generated tables |
| `exp45/e11_dev_M_<seed>.{json,pt}` | the six births |
| `exp45/e16_life_{T_raw65536_drift_online,D_drift}_<seed>.{json,pt}` | the twelve receipts and final weights |
| `exp45/logs/e16f7_*` | the logs and the wall file |
| `exp45/archive_e16f7_dry/` | the dry run |
