# E16-M3 report: the replay share (10 November 2026)

## Sources

- **Brief:** `OPUS_TASK_E16F7_E16M3_CLOUD_JOBS_20261109.md`, Job 2 (commit e78aa83e).
  - Both halves were run by this session: half a is nine sign-rule lives, half b five backprop lives.
- **Predictions:** `E16M3_PREDICTIONS.md`, committed at bec6d07f before launch and not changed since.
- **Harness:** `exp16_integrated.py` at e78aa83e, unchanged.
- **Branch:** `claude/vibrant-gauss-tops5f`. It is the consolidated branch merged in at 691a9959, with every `.py` file identical to e78aa83e.

## Tables

| File | Contents |
|---|---|
| `exp45/E16M3_TABLES.md` (from `e16m3_tables.py`) | Share curves; every-checkpoint trajectories of mod-97 add, sub, max, min and the four held pairs for each share, beside the twin; verdict inputs |
| `exp45/SUMMARY_E16M3.md` | The summarizer, for seeds 400009, 600013, 300007 |

## Run identity

- **Machine:** `Linux vm 6.18.44-fc-v338 #1 SMP x86_64`, 4 cores.
- **Software:** Python 3.13.18, PyTorch 2.15.0 (CPU).
- **Commits:** harness e78aa83e; launch from 691a9959 plus the predictions commit bec6d07f.
- **CPU-seconds per life** (from each log's last line):

| Life | 400009 | 600013 | 300007 |
|---|---:|---:|---:|
| T, share 0 | 5,169 | 5,160 | 5,132 |
| T, share 0.25 | 5,381 | 5,423 | 5,349 |
| T, share 0.75 | 5,441 | 5,355 | 5,308 |
| M, share 0.25 | 4,665 | 4,730 | — |
| M, share 0.5 | (E16-M2) | 4,725 | — |
| M, share 0.75 | 4,669 | 4,733 | — |

- **Forecast against measured:**

| | Forecast | Measured |
|---|---|---|
| T life | about 5,000 CPU-s | 5,132–5,441 (+3 to +9%) |
| M life | about 4,600 CPU-s | 4,665–4,733 (+1 to +3%) |
| All lives finished | about 08:30 UTC (launched 03:19) | 08:21 UTC, 5.0 hours |

## Outcome

**What the job tested.** The prediction was that the replay share acts as a dose that costs the sign rule its hard pairs while backprop stays flat. It points the other way.

**For the rule, more verbatim replay helps the hard pairs, and strongly.**

| Seed | Hard pairs at shares 0.25 / 0.5 / 0.75 |
|---|---|
| 400009 | 43.7 / 46.3 / 73.2 |
| 600013 | 23.3 / 44.8 / 60.2 |
| 300007 | 59.2 / 67.7 / 75.7 |

**Backprop moves the same way, by less.**

| Seed | Hard pairs at shares 0.25 / 0.5 / 0.75 |
|---|---|
| 400009 | 78.2 / 84.7 / 88.1 |
| 600013 | 71.9 / 81.8 / 86.7 |

**Share 0 forgets everything learned before the current mixture.** That is mod-97 operations and the compositions alike: the hard pairs end at about 1%.

| Prediction | Verdict |
|---|---|
| P16M3.1 | **Holds** in all three seeds |
| P16M3.2 | **Misses** in both failing seeds; the hard pairs move opposite to the prediction |
| P16M3.3 | **Misses** at share 0.25 in both seeds; holds at 0.5 and 0.75 |
| P16M3.4 | **Misses**: 300007 fails the gate at 0.25 by 0.004, and its hard pairs at 0.25 are below those at 0.75 |

## Share curves

Last checkpoint. Hard pairs = mean of `+-` and `-+`, final answer. Ratios are to the drift twin, a joint-stream reference. The share-0.5 rows are the comparators on file, except M for 600013, which this job ran.

### Seed 400009

Twin: mod-97 95.7, hard pairs 89.1, compositions 94.0, mod-73 94.4.

| Arm | Share | mod-97 | Hard pairs | +- | -+ | Comp | Ratio mod-97 | Ratio comp | Ratio mod-73 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| T | 0 | 33.2 | 1.2 | 1.5 | 1.0 | 18.2 | 0.347 | 0.194 | 0.948 |
| T | 0.25 | 72.8 | 43.7 | 49.8 | 37.5 | 61.2 | 0.761 | 0.651 | 0.946 |
| T | 0.5 | 86.6 | 46.3 | 48.0 | 44.7 | 64.9 | 0.905 | 0.691 | 0.962 |
| T | 0.75 | 88.8 | **73.2** | 72.0 | 74.5 | 81.7 | 0.928 | **0.869** | 0.967 |
| M | 0.25 | 95.3 | 78.2 | 78.2 | 78.3 | 87.7 | 0.996 | 0.933 | 1.019 |
| M | 0.5 | 95.1 | 84.7 | 83.0 | 86.3 | 91.3 | 0.994 | 0.972 | 1.009 |
| M | 0.75 | 95.8 | 88.1 | 88.7 | 87.5 | 93.0 | 1.001 | 0.990 | 1.005 |

### Seed 600013

Twin: mod-97 94.1, hard pairs 91.2, compositions 94.8, mod-73 93.8.

| Arm | Share | mod-97 | Hard pairs | +- | -+ | Comp | Ratio mod-97 | Ratio comp | Ratio mod-73 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| T | 0 | 30.2 | 1.0 | 0.8 | 1.2 | 17.9 | 0.321 | 0.189 | 0.854 |
| T | 0.25 | 64.3 | 23.3 | 20.3 | 26.3 | 49.7 | 0.684 | 0.524 | 0.941 |
| T | 0.5 | 72.8 | 44.8 | 41.7 | 47.8 | 66.8 | 0.774 | 0.705 | 0.914 |
| T | 0.75 | 76.5 | **60.2** | 57.3 | 63.2 | 75.5 | 0.813 | **0.796** | 0.907 |
| M | 0.25 | 93.1 | 71.9 | 68.8 | 75.0 | 83.5 | 0.989 | 0.881 | 1.005 |
| M | 0.5 | 95.1 | 81.8 | 80.0 | 83.7 | 89.1 | 1.011 | 0.940 | 1.001 |
| M | 0.75 | 94.4 | 86.7 | 86.8 | 86.5 | 91.6 | 1.004 | 0.966 | 1.005 |

### Seed 300007

Twin: mod-97 93.1, hard pairs 85.8, compositions 91.5, mod-73 92.9.

| Arm | Share | mod-97 | Hard pairs | +- | -+ | Comp | Ratio mod-97 | Ratio comp | Ratio mod-73 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| T | 0 | 12.3 | 1.1 | 1.3 | 0.8 | 7.8 | 0.132 | 0.085 | 0.966 |
| T | 0.25 | 84.2 | 59.2 | 58.8 | 59.5 | 72.8 | 0.905 | 0.796 | 0.983 |
| T | 0.5 | 89.6 | 67.7 | 65.0 | 70.3 | 79.3 | 0.962 | 0.867 | 0.992 |
| T | 0.75 | 90.8 | 75.7 | 72.3 | 79.0 | 85.0 | 0.976 | 0.928 | 0.972 |

### Hard-pair ratio to the twin

This is the 9 November audit's harder gate.

| Seed | T at share 0.5 | T at share 0.75 |
|---|---:|---:|
| 400009 | 0.52 | **0.82** |
| 600013 | 0.49 | 0.66 |
| 300007 | 0.79 | **0.88** |

### Trajectories

Mod-97 family mean and hard pairs at checkpoints 0–6. The full per-operation and per-pair tables are in `exp45/E16M3_TABLES.md`.

| Seed | Arm and share | mod-97 | Hard pairs |
|---|---|---|---|
| 400009 | T 0 | 37 88 58 0 31 22 33 | 1 1 12 72 42 1 1 |
| | T 0.25 | 37 85 61 46 56 68 73 | 1 1 5 69 68 44 44 |
| | T 0.5 | 37 90 62 78 81 86 87 | 1 1 3 57 68 52 46 |
| | T 0.75 | 37 90 84 81 84 88 89 | 1 1 1 41 68 70 73 |
| 600013 | T 0 | 22 87 69 0 26 27 30 | 1 1 3 59 53 1 1 |
| | T 0.25 | 22 86 73 36 51 61 64 | 1 1 1 44 46 18 23 |
| | T 0.5 | 22 82 80 47 60 70 73 | 1 0 2 57 69 46 45 |
| | T 0.75 | 22 82 81 69 71 74 76 | 1 1 1 47 62 57 60 |
| 300007 | T 0 | 21 93 89 0 29 13 12 | 1 1 14 85 82 2 1 |
| | T 0.25 | 21 93 88 81 76 79 84 | 1 1 7 83 83 58 59 |
| | T 0.5 | 21 94 93 89 82 88 90 | 1 1 3 76 82 69 68 |
| | T 0.75 | 21 93 91 91 90 91 91 | 1 1 1 68 78 75 76 |

**How the dose works.** It is retention, not acquisition.
- Lower shares learn the hard pairs faster at checkpoint 3, when the compositions arrive. Share 0 reaches 59–85, share 0.75 reaches 41–68.
- They then lose them as the stream drifts on to mod-73.
- By checkpoint 4 every share is near the same peak: 62–78 for shares 0.5–0.75 against 46–83 for 0.25.
- Only the higher shares keep the hard pairs to the end.
- The same holds for mod-97. At share 0.75 its collapse at checkpoint 3 is much smaller: 81 / 69 / 91 against 78 / 47 / 89 at share 0.5.

## Verdicts

### P16M3.1 (replay does work): holds in every seed

| Seed | Share-0 mod-97 ratio (limit < 0.80) | Share 0 − share 0.5 on mod-97 (limit ≤ −10) |
|---|---:|---:|
| 400009 | 0.347 | −53.4 |
| 600013 | 0.321 | −42.6 |
| 300007 | 0.132 | −77.2 |

Share 0 also loses the compositions entirely. The hard pairs end at 1.0–1.2, from peaks of 59–85 at checkpoint 3. Only mod-73, the current family at the end, survives (0.85–0.97).

### P16M3.2 (a dose on the rule): misses in both failing seeds

| Seed | Hard pairs 0.25 − 0.5 (limit ≥ +5) | Hard pairs 0.75 − 0.5 (limit ≤ −5) | mod-97 0.25 − 0.5 (limit ≤ −3) | mod-97 0.75 − 0.5 (limit ≥ +3) | Verdict |
|---|---:|---:|---:|---:|---|
| 400009 | **−2.7** | **+26.9** | −13.8 | **+2.2** | misses |
| 600013 | **−21.4** | **+15.5** | −8.5 | +3.7 | misses |

The mod-97 direction is as predicted: three of the four steps clear the margin, and 400009's 0.75 step falls 0.8 short. The hard pairs move the same way as mod-97, not the opposite way. More replay helps both.

### P16M3.3 (backprop flat): misses at share 0.25 in both seeds

| Seed | M share | Comp ratio (limit ≥ 0.90) | Hard pairs | Share-0.5 M hard pairs | Difference (limit within 5) | Verdict |
|---|---:|---:|---:|---:|---:|---|
| 400009 | 0.25 | 0.933 | 78.2 | 84.7 | **−6.4** | misses |
| 400009 | 0.75 | 0.990 | 88.1 | 84.7 | +3.4 | holds |
| 600013 | 0.25 | **0.881** | 71.9 | 81.8 | **−9.9** | misses |
| 600013 | 0.5 | 0.940 | 81.8 | 81.8 | 0 | holds |
| 600013 | 0.75 | 0.966 | 86.7 | 81.8 | +4.8 | holds |

Backprop is not flat. It rises with the share in the same direction as the rule, by +9.9 and +14.8 points from 0.25 to 0.75. The rule rises by +29.5 and +36.9 over the same range. Backprop's mod-97 operations, unlike the rule's, are flat at 93–96 across all shares.

### P16M3.4 (the passing seed): misses

| Clause | Result |
|---|---|
| Family gate at 0.25 | Minimum ratio **0.796**, compositions just below 0.80: misses |
| Family gate at 0.75 | 0.928: passes |
| Hard pairs at 0.25 above those at 0.75 | **59.2 against 75.7**, the opposite way: misses |

## Which case obtained

None of the brief's three cases fits exactly.
- P16M3.2 misses, but the rule's hard pairs are not flat across shares: they reverse the predicted direction.
- P16M3.3 misses, so the share matters to backprop too, in the same direction and by about a third as much.

**What it means.**
- The reservoir's cost to the rule is not a dose of verbatim replay. Within 0.25–0.75, more verbatim replay is better for both updates, and much better for the rule.
- Both failing seeds respond the same way, so the seeds do not split.
- E16-M2's comparison at share 0.5 was not at a share that favoured backprop; backprop is best at 0.75 too.
- What the rule lacks is retention of the hard pairs once the stream drifts on. That is the gap more replay narrows: the rule's hard pairs at the end of the life rise steeply with the share, while their peak at checkpoint 4 hardly moves.
- At share 0.75, seed 400009 passes the family gate (0.928 / 0.869 / 0.967) and reaches 0.82 of the twin on the hard pairs alone. Seed 600013 comes within 0.004 of the family gate on compositions (0.796) but stays at 0.66 on the hard pairs.

**Follow-up suggested by the curve, not tested here.** The share curve has not flattened by 0.75 for the rule, so a higher share is the obvious next point. Fewer fresh items per batch would slow acquisition: the hard-pair peak at checkpoint 3 already falls from 57 to 41 in 400009 between shares 0.5 and 0.75.

## What went wrong, or is limited

- **One life per seed and share.** Backprop at share 0.5 for 400009 is E16-M2's run on another container. On the evidence of E16-RP and E16-F6, cloud containers reproduce one another bit-for-bit, but that run was not repeated here.
- **Share 0 is not a pure forgetting test.** Under the drifting stream, earlier families keep a small presence in the stream.
- **Twins as references.** The drift twin is a joint-stream reference, so its ratios measure distance from joint training, not from the same data.
- **Branch.** This session's branch carried E16-F6 commits that the consolidated branch had not taken as such. I merged the consolidated branch in rather than force-pushing, keeping its version on every conflict. The harness is identical to e78aa83e, and the only extra files are this session's E16-F6 signature table and dry-run files.
- **Dry runs** are in `exp45/archive_e16m3_dry/cloud_20261110/`, because the coordinating session's dry runs already occupy the same names one folder up.
- **Nothing failed.** Every life ran its six checkpoints with no error, and all fourteen logs are complete.
