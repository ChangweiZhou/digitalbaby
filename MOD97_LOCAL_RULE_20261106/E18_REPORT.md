# E18 report: births with a next-byte target at every bank (10 November 2026)

**Brief:** `OPUS_TASK_E18_LOCAL_HEADS_BIRTHS_20261110.md`, run whole: arms B0–B3, seeds 73129, 93683 and 116089, twelve births.

**Predictions:** `E18_PREDICTIONS.md`, committed at 0ea40628 before any birth started and not changed since.

**Harness:** 8d037ebb, unchanged.

**Tables:** `exp45/E18_TABLES.md`, written by `e18_score.py` (reads only). It holds every birth curve, the first bank's head, the silenced scores and the verdicts.

## Verdicts

| Prediction | Verdict | Seeds 73129 / 93683 / 116089 |
|---|---|---|
| P18.1 B1's first-bank head add89 < 45 in all three | **holds** | 27.3 / 29.7 / 28.3 |
| P18.2 B1 hard mean ≥ 70 in ≥ 2, below the reference in all three | **holds** | 80.7 / 82.2 / 77.2 (references 91.2 / 94.5 / 90.3) |
| P18.3 B0 hard mean ≥ 0.80 of the reference in ≥ 2 | **holds** | 0.96 / 0.96 / 0.94 |
| P18.4 B0 − B1 ≥ 5 in ≥ 2 | **holds** | +6.8 / +8.5 / +7.5 |
| P18.5 \|B2 − B1\| ≤ 10 in ≥ 2 | **misses** | −5.8 / −12.5 / −10.8 (1 of 3) |
| P18.6 B3 hard mean < 40 in all three, and its max/min mean ≥ 20 below B2's | **holds** | hard means 1.2 / 0.8 / 0.8; B2 − B3 on max/min: +95.2 / +94.7 / +91.0 |

**Which case of the brief's reading obtained, seed by seed: neither cleanly; it falls between the two.**

| Seed | B1 below the reference | B0 below the reference | B0 above B1 |
|---:|---:|---:|---:|
| 73129 | 10.5 | 3.7 | 6.8 |
| 93683 | 12.3 | 3.8 | 8.5 |
| 116089 | 13.1 | 5.6 | 7.5 |

- **Against the first case:** B1 does not come "within a few points" of the reference.
- **Against the second case:** B1 does not fall "far below" either. It ends at 0.85–0.88 of the reference, far above the brief's "held at about 55%". Its first bank stays a one-bank learner (head1 hard mean 19.5–20.0, against the one-bank reference's 15.2–21.7).
- **What the numbers say in every seed:**
  - a next-byte target at each bank, with no error passed between them, gets the two-bank network most of the way;
  - the relay between banks is worth a further 7–9 points of hard mean at this dose, under weight signs (B0) against exact backprop inside each bank (B1);
  - B0, with no backprop anywhere, ends 4–6 points below the end-to-end backprop reference, which was made on another machine.

**The cheaper within-bank rules:**
- **Credit through time with weight signs (B2)** costs 6–13 points against exact backprop inside each bank, more than the 10 the brief allowed in two of the three seeds.
- **The trace on the write synapses only (B3)** learns nothing.

**One caution for every comparison.** B1, B2 and B0 were all still rising at the last checkpoint, gaining 3–8 points of hard mean in the final 4,096 updates, while the reference gained 0–4. So the gaps are gaps at a fixed dose of 24,576 updates, not gaps between converged learners. No longer birth was run.

## Every birth, final checkpoint, k = 12 (main head)

| Arm | Seed | add89 | sub89 | max89 | min89 | Hard mean | Max/min mean | Two-bank reference (hard) | Ratio | CPU-s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 | 73129 | 87.7 | 87.3 | 99.0 | 98.7 | **87.5** | 98.8 | 91.2 | 0.96 | 4,307 |
| B0 | 93683 | 91.3 | 90.0 | 97.3 | 98.3 | **90.7** | 97.8 | 94.5 | 0.96 | 4,321 |
| B0 | 116089 | 82.3 | 87.0 | 95.0 | 96.3 | **84.7** | 95.7 | 90.3 | 0.94 | 4,316 |
| B1 | 73129 | 84.3 | 77.0 | 92.3 | 95.3 | **80.7** | 93.8 | 91.2 | 0.88 | 5,231 |
| B1 | 93683 | 84.7 | 79.7 | 95.0 | 96.3 | **82.2** | 95.7 | 94.5 | 0.87 | 5,279 |
| B1 | 116089 | 78.7 | 75.7 | 94.7 | 94.3 | **77.2** | 94.5 | 90.3 | 0.85 | 5,261 |
| B2 | 73129 | 81.3 | 68.3 | 95.7 | 96.3 | **74.8** | 96.0 | 91.2 | 0.82 | 5,702 |
| B2 | 93683 | 76.3 | 63.0 | 97.3 | 96.0 | **69.7** | 96.7 | 94.5 | 0.74 | 5,713 |
| B2 | 116089 | 68.7 | 64.0 | 94.7 | 93.0 | **66.3** | 93.8 | 90.3 | 0.73 | 5,731 |
| B3 | 73129 | 1.0 | 1.3 | 1.3 | 0.3 | **1.2** | 0.8 | 91.2 | 0.01 | 6,400 |
| B3 | 93683 | 0.7 | 1.0 | 2.3 | 1.7 | **0.8** | 2.0 | 94.5 | 0.01 | 6,435 |
| B3 | 116089 | 0.7 | 1.0 | 1.7 | 4.0 | **0.8** | 2.8 | 90.3 | 0.01 | 6,422 |

**References on file** (owner's machine, not rerun; compare levels, not digits). Hard mean and max/min mean, for seeds 73129 / 93683 / 116089:

| Reference | Hard mean | Max/min mean |
|---|---|---|
| Two banks, end-to-end backprop | 91.2 / 94.5 / 90.3 | 96.0 / 95.2 / 95.5 |
| One bank, backprop | 21.7 / 19.3 / 15.2 | 57.2 / 61.0 / 59.3 |

These reproduce the brief's reference table.

## The first bank's own head (local-heads arms), final checkpoint, k = 12

| Arm | Seed | add89 | sub89 | max89 | min89 | Hard mean |
|---|---:|---:|---:|---:|---:|---:|
| B1 | 73129 | 27.3 | 12.3 | 60.3 | 58.7 | 19.8 |
| B1 | 93683 | 29.7 | 10.3 | 55.0 | 59.3 | 20.0 |
| B1 | 116089 | 28.3 | 10.7 | 56.0 | 55.7 | 19.5 |
| B2 | 73129 | 23.3 | 11.7 | 56.3 | 55.7 | 17.5 |
| B2 | 93683 | 24.7 | 8.7 | 55.7 | 61.3 | 16.7 |
| B2 | 116089 | 22.3 | 10.0 | 53.0 | 59.3 | 16.2 |
| B3 | 73129 | 1.0 | 1.0 | 2.0 | 0.3 | 1.0 |
| B3 | 93683 | 0.7 | 1.0 | 2.7 | 1.7 | 0.8 |
| B3 | 116089 | 1.0 | 0.3 | 2.0 | 4.0 | 0.7 |

**What the first bank learns on its own target.** In B1 and B2 the first bank, trained only on its own next-byte prediction, lands at the one-bank reference's level: add89 22–30 and max/min 53–61, against 22–30 and 53–61. On top of that, the second bank builds a learner at 0.73–0.88 of the two-bank reference.

**The silenced score** (main head, k = 12) is 0–1.7% on every operation in every birth. With the stored operands silenced, the learners cannot answer.

## Birth curves

Hard mean / max-min mean at k = 12, main head, at each checkpoint. The loss is in brackets.

| Arm | Seed | 4,096 | 8,192 | 12,288 | 16,384 | 20,480 | 24,576 |
|---|---:|---|---|---|---|---|---|
| B0 | 73129 | 10 / 88 (0.66) | 36 / 96 (0.34) | 60 / 97 (0.20) | 76 / 98 (0.12) | 84 / 98 (0.09) | 88 / 99 (0.07) |
| B0 | 93683 | 9 / 89 (0.63) | 43 / 95 (0.33) | 62 / 95 (0.19) | 78 / 96 (0.12) | 88 / 96 (0.08) | 91 / 98 (0.06) |
| B0 | 116089 | 10 / 88 (0.66) | 37 / 93 (0.32) | 56 / 96 (0.19) | 69 / 96 (0.13) | 78 / 96 (0.09) | 85 / 96 (0.07) |
| B1 | 73129 | 11 / 86 (0.68) | 46 / 94 (0.35) | 56 / 94 (0.23) | 71 / 96 (0.17) | 78 / 97 (0.13) | 81 / 94 (0.11) |
| B1 | 93683 | 6 / 89 (0.72) | 37 / 94 (0.37) | 55 / 96 (0.24) | 65 / 96 (0.18) | 75 / 96 (0.14) | 82 / 96 (0.11) |
| B1 | 116089 | 6 / 83 (0.72) | 30 / 93 (0.42) | 53 / 93 (0.26) | 61 / 95 (0.18) | 70 / 93 (0.15) | 77 / 94 (0.12) |
| B2 | 73129 | 6 / 85 (0.75) | 29 / 94 (0.45) | 47 / 95 (0.28) | 61 / 94 (0.19) | 69 / 96 (0.15) | 75 / 96 (0.13) |
| B2 | 93683 | 2 / 86 (0.80) | 16 / 91 (0.54) | 34 / 96 (0.35) | 53 / 96 (0.24) | 63 / 97 (0.18) | 70 / 97 (0.15) |
| B2 | 116089 | 4 / 80 (0.78) | 20 / 91 (0.50) | 39 / 92 (0.30) | 50 / 94 (0.22) | 58 / 91 (0.17) | 66 / 94 (0.14) |
| B3 | 73129 | 1 / 0 (1.56) | 1 / 2 (1.56) | 1 / 1 (1.56) | 1 / 1 (1.54) | 1 / 1 (1.52) | 1 / 1 (1.51) |
| B3 | 93683 | 1 / 2 (1.48) | 2 / 5 (1.50) | 1 / 3 (1.53) | 2 / 2 (1.53) | 2 / 4 (1.50) | 1 / 2 (1.54) |
| B3 | 116089 | 2 / 1 (1.60) | 0 / 0 (1.56) | 1 / 2 (1.52) | 1 / 1 (1.55) | 1 / 1 (1.53) | 1 / 3 (1.51) |
| Reference | 73129 | 20 / 84 (0.56) | 52 / 93 (0.30) | 78 / 93 (0.18) | 87 / 96 (0.10) | 91 / 97 (0.07) | 91 / 96 (0.06) |
| Reference | 93683 | 17 / 85 (0.54) | 60 / 92 (0.27) | 80 / 92 (0.15) | 87 / 96 (0.10) | 90 / 95 (0.07) | 94 / 95 (0.06) |
| Reference | 116089 | 18 / 88 (0.53) | 52 / 92 (0.25) | 76 / 93 (0.14) | 88 / 94 (0.08) | 88 / 96 (0.06) | 90 / 96 (0.05) |

The full curves (each operation at k = 0, 4, 8, 12 and 20 at every checkpoint, and the first bank's head) are in `exp45/E18_TABLES.md`.

**What the curves show:**
- **Order:** every learning arm follows the same order as at the end (reference > B0 > B1 > B2). The local arms learn the hard operations more slowly; at 12,288 updates B1 and B2 stand 22–46 points behind the reference, and B0 16–20.
- **B3:** its loss stays at 1.5–1.6 from the first checkpoint to the last, and every operation stays near 1%. Its 512-update dry run had max89 and min89 near 20% (and the owner's dry run had 6%), so it picked up a little early and lost it by 4,096 updates.

## Check

`e18_check.py` was run once, before the dry runs. Its output is identical to `exp45/E18_CHECK.json`:
- every number is 0.0: the gradient difference from a3c0b675 with the options unset, the shared initialisation, the four leak terms and the two write-source terms;
- `head1_out_switch` is true.

The output is filed as `exp45/archive_e18_dry/cloud_20261110/e18_check_output.txt`.

## Run identity

| Item | Value |
|---|---|
| Machine | `Linux vm 6.18.44-fc-v338 #1 SMP @0 x86_64 x86_64 x86_64 GNU/Linux`; 4 cores; Intel Xeon at 2.10 GHz |
| Software | Python 3.11.18, torch 2.15.0+cpu, NumPy 2.4.6; one thread per process |
| Harness | 8d037ebb |
| Commits | predictions 0ea40628; B0 2eafb07b; B1 04034a8a; B2 and B3 8600f059 |

Each process's environment was read from `/proc/<pid>/environ` five seconds after launch (`exp45/logs/e18_environ_at_launch.txt`). Every birth had `E16_ARCH=bank2 E16_BOOKDECAY=off`, plus:

| Arm | Extra environment |
|---|---|
| B0 | `E16_BIRTHRULE=T E16_TRACEUP=all` |
| B1 | `E16_HEADS=local` |
| B2 | `E16_HEADS=local E16_BIRTHRULE=T E16_TRACEUP=all` |
| B3 | `E16_HEADS=local E16_BIRTHRULE=T` |

Each arm appeared once per seed, and nothing else was set.

**CPU-seconds per birth:** the last log line's `cpu` field, which is the birth curve. It leaves out the final silenced and first-head scoring.

| Arm | 73129 | 93683 | 116089 | Sum |
|---|---:|---:|---:|---:|
| B0 | 4,307 | 4,321 | 4,316 | 12,944 |
| B1 | 5,231 | 5,279 | 5,261 | 15,771 |
| B2 | 5,702 | 5,713 | 5,731 | 17,146 |
| B3 | 6,400 | 6,435 | 6,422 | 19,257 |
| **All twelve** | | | | **65,118** |

**Wall clock** (UTC; every exit 0, `exp45/logs/e18_wall.txt`):

| Event | Time |
|---|---|
| All twelve births launched together | 16:16:17 |
| B0 finished | 19:54:50–19:55:18 |
| B1 finished | 20:30:40–20:32:17 |
| B2 finished | 20:42:54–20:43:25 |
| B3 finished | 20:54:32–20:55:13 |

**Wall time: 4 h 39 min.**

## Forecast against measured

The forecast came from timing 256 updates per arm directly (`exp45/archive_e18_dry/cloud_20261110/e18_time_updates.py`), because a 512-update dry run is mostly scoring.

| Item | Forecast | Measured | Measured / forecast |
|---|---:|---:|---:|
| B0 birth | about 4,030 | 4,315 (mean) | 1.07 |
| B1 birth | about 5,060 | 5,257 | 1.04 |
| B2 birth | about 5,460 | 5,715 | 1.05 |
| B3 birth | about 6,120 | 6,419 | 1.05 |
| Twelve births, CPU-s | about 62,000 | 65,118 | 1.05 |
| Wall time | about 4.4 h | 4 h 39 min | 1.06 |

The brief's estimate was about 55,000 CPU-s and 3.8 h. On this container the dry runs alone (81–125 CPU-s, less than the owner's 107–143) would have forecast births cheaper than the owner's, which would have been wrong by a factor of about 2.5.

## What went wrong

- **No restarts, no killed or rerun births.** All twelve ran once and exited 0.
- **The forecast was 4–7% low.**
- **B3 learned nothing.** This is a result, not a failure of the run: its loss never moved from about 1.5.
- **A display slip:** my first look at the tables printed some sections twice, because two of my `sed` ranges overlapped. The committed `exp45/E18_TABLES.md` is unaffected.

## What this does not claim

Births are development results on sealed panels, at evidence level E2.

- No life, retention or reuse on never-reinforced inputs is tested.
- No learning-core improvement is claimed.
- The references were made on the owner's machine, so the comparisons are of levels, not digits.
- The gaps are at 24,576 updates, with the local arms still rising.
- Feedback relayed layer by layer is allowed under Mr. Xu's working answers (review 9.67). Decisions remain Mr. Zhou's.

## Files

| File | What it is |
|---|---|
| `E18_PREDICTIONS.md` | the predictions, written before any birth |
| `E18_REVIEW_SECTION_9_71.md` | the review paragraph |
| `e18_score.py` | writes `exp45/E18_TABLES.md` |
| `exp45/e17_dev_bank2_nbd{_bT_tall,_lh,_lh_bT_tall,_lh_bT}_<seed>.{json,pt}` | the twelve births |
| `exp45/logs/e18_birth_<arm>_<seed>.log`, `e18_wall.txt`, `e18_environ_at_launch.txt` | logs, wall file and environments |
| `exp45/archive_e18_dry/cloud_20261110/` | the dry runs, the check output, and the timing script and its output |
