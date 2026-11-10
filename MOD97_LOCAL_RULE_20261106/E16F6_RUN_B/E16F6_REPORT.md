# E16-F6 report: the plus-minus tail, six more seeds of the drifting stream (9 November 2026)

**Sources.**
- Brief: `OPUS_TASK_E16M2_E16F6_CLOUD_JOBS_20261109.md`, Job 2 (commit 1470401b).
- Predictions: `E16F6_PREDICTIONS.md`, committed at 22595016 before any birth or life started, and not changed since.
- Harness: unchanged at 1470401b.
- Branch: `claude/vibrant-gauss-tops5f`. This session cannot push to the consolidated branch.

**Tables.**
- `exp45/SUMMARY_E16F6.md`: the summarizer, six seeds.
- `exp45/E16F6_SIGNATURE.md`: from `e16f6_signature.py`. It holds the per-checkpoint trajectories of learner and twin for the six seeds, and the twelve-seed signature list.

## Outcome

| Prediction | Verdict |
|---|---|
| P16F6.1, births at least 85% | **Holds**, 3 of 3 |
| P16F6.2, twin at or above 90 / 85 / 90 | **Holds**, 6 of 6 |
| P16F6.3, at least four of six seeds at 0.80 on all three families | **Holds**, 5 of 6 |
| P16F6.4, seed 600013 misses compositions under drift | **Holds**: composition ratio 0.705 |
| P16F6.5, the early signature | **Misses** |

**P16F6.5 in detail.**
- The Spearman correlation over twelve seeds is 0.217, against a required 0.5.
- The high-start clause fails in four seeds.
- Only the low-start clause holds, and it rests on a single seed.

**The gate under the drifting stream.** It now stands at 10 of 12 seeds. The two failures are 400009 and 600013. Seed 600013 fails under both schedules, so its fragility belongs to the seed, not the schedule.

## Births (P16F6.1): holds in every seed

| Seed | add89 | sub89 | max89 | min89 | CPU-s | Verdict |
|---|---:|---:|---:|---:|---:|---|
| 800011 | 92.7 | 92.7 | 97.0 | 97.3 | 2,904 | holds |
| 900007 | 89.0 | 91.3 | 98.3 | 96.3 | 2,983 | holds |
| 1000003 | 90.7 | 94.0 | 95.3 | 96.7 | 2,955 | holds |

Each birth ran 24,576 updates. No `exp45/e11_dev_M_<seed>.*` existed for these seeds before the job; that was checked at 17:04 UTC.

## After the last checkpoint

Operations are at k = 12; compositions are final answer over the four held pairs.

| Seed | Twin mod-97 | Twin comp | Twin mod-73 | Learner mod-97 | Learner comp | Learner mod-73 | Ratio mod-97 | Ratio comp | Ratio mod-73 | Learner +- / -+ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 500011 | 91.8 | 88.4 | 90.7 | 87.6 | 78.5 | 91.5 | 0.955 | 0.888 | 1.009 | 64.2 / 70.5 |
| 600013 | 94.1 | 94.8 | 93.8 | 72.8 | 66.8 | 85.8 | **0.774** | **0.705** | 0.914 | 41.7 / 47.8 |
| 700027 | 95.7 | 92.5 | 95.1 | 89.9 | 80.5 | 92.2 | 0.939 | 0.870 | 0.970 | 69.7 / 66.3 |
| 800011 | 94.9 | 91.8 | 94.0 | 82.1 | 75.8 | 89.2 | 0.866 | 0.825 | 0.949 | 63.8 / 63.8 |
| 900007 | 93.1 | 92.9 | 92.7 | 86.2 | 77.5 | 90.0 | 0.927 | 0.834 | 0.971 | 63.2 / 70.7 |
| 1000003 | 94.0 | 91.5 | 93.1 | 76.4 | 75.9 | 87.4 | 0.813 | 0.830 | 0.939 | 68.2 / 61.0 |

- **P16F6.2 (twin at or above 90 / 85 / 90): holds in every seed.**
  - Seed 500011's drift twin reaches 88.4 on compositions. Its twin under stages reached 84.0 and was still rising. The shortfall did not repeat.
  - In 500011 the twin's plus-minus pairs are the weakest of any twin here (78.0 / 78.7).
- **P16F6.3 (at least four of six at 0.80 on all three families): holds, five of six.** Only 600013 misses, on mod-97 operations (0.774) and on compositions (0.705). Seed 1000003 passes narrowly, at 0.813 on mod-97 operations.
- **P16F6.4 (600013 misses compositions under drift): holds.**
  - Its composition ratio is 0.705 under drift; under stages it was 64.7 against a twin at 93.2.
  - The pattern is the same as under stages: `+-` and `-+` end near 42–48 while the max/min pairs stay at 89. The mod-97 operations are 72.8, with add and sub at 52.0 and 47.5.
  - Its twin is unaffected: 94.1 / 94.8 / 93.8.

## The early signature (P16F6.5): misses

The twelve drift seeds, sorted by the learner's mod-97 family mean at checkpoint 2:

| Seed | Source | Panels | Ck-2 mod-97 | Final +- | Final -+ | Ratio comp | All three at 0.80 |
|---|---|---|---:|---:|---:|---:|---|
| 400009 | E16-F5 | new | 62.0 | 48.0 | 44.7 | 0.691 | no |
| 116089 | E16-S | old | 71.5 | 72.7 | 71.3 | 0.864 | yes |
| 1000003 | E16-F6 | new | 75.6 | 68.2 | 61.0 | 0.830 | yes |
| 600013 | E16-F6 | new | 79.9 | 41.7 | 47.8 | 0.705 | no |
| 500011 | E16-F6 | new | 83.6 | 64.2 | 70.5 | 0.888 | yes |
| 800011 | E16-F6 | new | 84.8 | 63.8 | 63.8 | 0.825 | yes |
| 900007 | E16-F6 | new | 86.1 | 63.2 | 70.7 | 0.834 | yes |
| 700027 | E16-F6 | new | 86.9 | 69.7 | 66.3 | 0.870 | yes |
| 200003 | E16-F5 | new | 88.0 | 67.2 | 72.5 | 0.862 | yes |
| 93683 | E16-S | old | 88.4 | 69.8 | 70.0 | 0.868 | yes |
| 73129 | E16-S | old | 92.1 | 65.8 | 77.0 | 0.860 | yes |
| 300007 | E16-F5 | new | 92.6 | 65.0 | 70.3 | 0.867 | yes |

| Clause | Result | Verdict |
|---|---|---|
| Spearman rho at least 0.5 | **0.217**, ties averaged | misses |
| Every seed below 70 at checkpoint 2 ends with `+-` below 60 | Only 400009 is below 70; it ends at 48.0 | holds, on one seed |
| Every seed above 80 at checkpoint 2 ends with `+-` above 65 | Eight seeds are above 80. Four end at or below 65: 500011 at 64.2, 800011 at 63.8, 900007 at 63.2, and 300007 at exactly 65.0, which does not clear "above 65" | misses |

**Panel mix.** The three E16-S seeds were scored on the old held panels, whose `+-` triples are shared with the taught `->` pair. On the nine seeds with re-keyed panels only, rho is 0.450, still below 0.5. E16-RP put the same-weights panel difference on `+-` at +1.0 / +6.8 / −0.8.

**Why the signature fails.** Seed 600013 does not dip at checkpoint 2 (79.9). Under the drifting stream its mod-97 operations fall at checkpoint 3 instead, to 47.2, as the compositions take over the stream. Final `+-` also lies in a narrow band of 63–73 for ten of the twelve seeds, so a rank correlation has little to work with outside the two failures.

**An exploratory reading, not a prediction.** The lowest mod-97 mean over checkpoints 1–6 separates the two failing seeds cleanly: 47.2 for 600013 and 62.0 for 400009. Every passing seed's minimum is 66.7 or above (1000003 at 66.7, 800011 at 69.3, 116089 at 70.7, the rest 77–87). That is a description of these twelve lives, fitted after the fact. It would need its own locked prediction on new seeds before it could be used.

## Machine, pace, forecast against measured

**Machine.** The same four-core container as E16-RC and E16-RP: Intel Xeon at 2.1 GHz, Python 3.13, PyTorch 2.15.0 (CPU), NumPy 2.5.3.

**Dry runs.** These are in `exp45/archive_e16f6_dry/`:
- learner, four updates per block: 249 CPU-s;
- twin, four updates per block: 257 CPU-s;
- birth at 4,096 updates: 499 CPU-s;
- all three together: 518 s of wall time.

| Item | Forecast | Measured |
|---|---|---|
| Birth | about 3,000 CPU-s | 2,904–2,983 |
| Learner life | about 5,000 CPU-s | 4,955–5,119 |
| Twin life | about 4,400 CPU-s | 4,358–4,432 |
| Births done | about 1.9 h after launch | 1.85–1.9 h (18:56–18:59 UTC) |
| All lives done | about 22:30 UTC, 5.3 h | **21:47 UTC, 4.7 h** |

The forecast assumed the first-wave lives would share the cores with the second wave for longer than they did. Once the first wave ended at about 20:50 UTC, the second wave ran six processes on four cores.

**Waves.** The first wave launched at 17:05 UTC: three births and six lives. A detached launcher started each new seed's learner and twin as soon as its birth's `.json` existed and its process had exited, at 18:55:59, 18:57:59 and 18:59:20 UTC.

## What went wrong, or is limited

- **The wrong job, briefly.** This session first started Job 1 (E16-M2) and ran its four dry runs only partly. They were stopped when the owner said another worker has Job 1. Their dry receipts were deleted uncommitted, and nothing of E16-M2 is in this branch.
- **One automatic push failed silently.** The commit landed but the push did not. It was noticed at the next check and pushed by hand, and the commit script now confirms that the remote matches after every push. No data was affected.
- **Logs went to scratch and were copied at each commit.** Each of the fifteen logs has all its lines: six per life, six per birth.
- **One life per seed.** The signature analysis mixes old and new panels, as noted.
- **The exploratory minimum-dip reading is post hoc.**
