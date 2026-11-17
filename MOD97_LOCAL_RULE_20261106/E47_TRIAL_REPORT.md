# E47 trial report: stage 0 of `OPUS_TASK_E47_LOCALITY_AT_THE_INPUT_AND_THE_OUTPUT_20261116.md`, revision 2

Worker's report, 17 November 2026. Evidence level E2, one seed (73129); every description below is that seed's. Branch `claude/e47-worker-stage-0-3d0ya6`, from the default branch `codex/minifly-consolidated-20261023` at 7da5c453 (the brief at cf96788d). **Stage 0 only: the trial is reported and the worker stops; no roster run was started.** Tables in this report are copied from `exp45/e47/E47_TRIAL_TABLES.md`, computed from the receipts by `e47_trial_read.py`; no reading beyond the brief's fixed forms.

## What was built

- `e47_local.py`: byte-identical copy of `PROJECT_DOCUMENTS/COORD_CHECKS_20261116/e47_local_spec.py` (sha256 031adf65…), imported, not rewritten.
- `exp42_cells.py`, `e42_cells.py`: E42's runner and layer, byte-identical from `claude/gallant-dijkstra-i9zgsr` at 0de93296 (the blobs of 534df8f0). Imported for E42's data, optimizer, panels, scores and check-12 counts.
- `exp47_local.py`, the runner. Its additions:
  - **A faster forward that computes only the cells updated at each position**, one group per period.
    - Period-1 groups go through the specification's `IndexSign`.
    - Groups of period > 1 apply the step weights to the sources' states before the weights. The sweep for them is a custom function: exact gradients to W, b and the step weights c, and sign(W)·scale to the sources.
    - When every cell is updated and all are of period 1 (arm `one`, and 16 a side), the position is computed as the specification computes it.
  - **Scoring** with E42's `score_all` (plain), and with the far cells held at zero and held at their mean, on A, R and the length test at 12 and 40 letters. The means for each scored set come from the plain forward on the prompts followed by the network's own plain answers.
  - **Greedy decoding is incremental:** the prompt is run once, then one position per answer byte. Check 3 shows its answers equal `exp11_cell.respond`'s.
  - **Checkpoint and resume**, every 1,024 updates (added after a container restart; see "Interruptions").
  - **Receipts and weight files.**
- **Optimizer.** E26's (`C26.make_opt`), with `rnn.weight_hh_l0` and `rnn.bias_hh_l0` taken out; `Wc`, `bc`, `bfar` and `a` join the decayed group (0.25). Every receipt records that those two module tensors received no gradient.

## Stage-0 checks (receipts in `exp45/e47_qual/`; checks 1–7 on runner 882b38f9 at commit ef340415, check 8 on 8d80a7dc)

| Check | Result |
|---|---|
| 1. Specification file | **PASS.** The copy is byte-identical. Its self-checks, run in `$K` on this host, print the coordinator's output for checks 2–4: connections at 16 a side equal E42's, counts, periods, time and locality properties, and the audit's draw digests at 16, 32 and 64 a side. The four lines of self-check 1 differ only in rounding residues: logits 1.8e-7 here against 2.4e-7 on the Mac, gradients ≤ 2.6e-7. All are inside 1e-6 / 1e-5; the receipt lists the differing text. |
| 2. 16 a side is E42's `F10` | **PASS.** The start is equal to E42's, built with `from_e42` and `Local(..., structure=...)`. First batch: logits differ by 1.8e-7 under both rules; gradients by 2.6e-7 (S) and 1.8e-7 (B) of each tensor's largest entry. The first 16 losses of the two runners differ by at most 2.4e-7 (S) and 4.8e-7 (B); 0 at the 1st and 16th updates. Nothing needed investigating. A dry run of 2,048 updates under the sweep is beside E42's `e42_F10_S_dry2048_73129` (table below). |
| 3. Faster forward | **PASS.** Equal to the specification's on the first batch at 16 (one) and at 32 and 64 (`one`, `place`, `scatter`), under both rules. It holds at the start and with Wc, bc, bfar and a moved off it, and under both diagnostics. Logits ≤ 1.5e-7, gradients ≤ 3.7e-7 relative (52 cases); `one` is exactly equal. The time properties of self-check 4 hold at 32 and 64 under every arm. Decoding: incremental answers equal `exp11_cell.respond`'s on E26's birth S 73129 and on two moved 1,024-cell networks, plain and under both diagnostics. |
| 4. Locality (runner's forward, 1,024 cells, `one` and `place`, weights moved) | **PASS.** A changed byte at position 9 changes no far cell there and reaches none sooner than its distance allows. The logits up to a position are unchanged when far states at that position are set by hand (positions 5, 9 and 14), and change after it. With `far_off` the far cells are zero. |
| 5. Data | **PASS.** The digest at row 4,096 is bbd71ab7ca27b72e, E26's. Each trial receipt has the same value at row 4,096. |
| 6. Scores | **PASS, equal.** E42's check-12 counts on E26's six births, each loaded by index with `structure=` (sources: itself and the 255 others). The difference is 0 for every count. |
| 7. Cost (serial, idle host, one thread) | **PASS** (figures below). |
| 8. Resume (added) | **PASS.** A 2,048-update run at 1,024 cells, `place`, sweep, was stopped after its checkpoint at 1,024 and resumed from the files. It ends with the same tensors_sha256 (907805f0…), curve, data digest and end cosines as the same run without a stop. |

Check 2's dry run beside E42's (sweep, 2,048 updates; not required to be equal):

| Update | E47 A / R / len12 / len40 / loss | E42 `F10` A / R / len12 / len40 / loss |
|---|---|---|
| 1,024 | 1.2 / 0.9 / 0.3 / 1.0 / 1.0316 | 1.0 / 0.8 / 1.0 / 1.0 / 1.0330 |
| 2,048 | 1.3 / 1.4 / 2.0 / 1.3 / 0.9061 | 1.3 / 1.1 / 2.3 / 1.3 / 0.9040 |

**Check 7 and the forecast.** CPU-seconds per update (sweep / backpropagation), and per scored row with both diagnostics (three generations plus the means' forward):

| Cells, arm | s/update S | s/update B | s/row | forecast for one full run, CPU-h (S / B) | multiply-adds a position |
|---|---|---|---|---|---|
| 256 `one` | 0.196 | 0.195 | 16.4 | 1.8 / 1.8 | 930,560 |
| 1,024 `one` | 0.486 | 0.485 | 42.8 | 4.5 / 4.5 | 955,904 |
| 1,024 `place` | 0.535 | 0.580 | 43.7 | 4.8 / 5.1 | 936,500 |
| 1,024 `scatter` | 0.612 | 0.613 | 50.2 | 5.5 / 5.5 | 936,500 |
| 4,096 `one` | 2.119 | 1.925 | 177.5 | 19.2 / 17.9 | 1,057,280 |
| 4,096 `place` | 1.667 | 1.503 | 158.2 | 15.6 / 14.5 | 960,260 |
| 4,096 `scatter` | 1.604 | 1.498 | 156.0 | 15.2 / 14.4 | 960,260 |

- **One forward and backward on the first batch, under the sweep.** At 1,024 cells the specification takes 0.62 s (`one`), 0.85 s (`place`) and 0.99 s (`scatter`); the faster forward takes 0.49, 0.52 and 0.65 s. At 4,096 cells the specification takes 2.1 to 3.9 s; the faster forward takes 1.5 to 2.1 s.
- **Forecasts.** The trial: 18.9 CPU-h. **The roster as planned now: 48.2 CPU-h.** That is the two `scatter` runs at 1,024 cells (5.5 + 5.5), and `one` at 4,096 cells under backpropagation (17.9) and under the sweep (19.2).
- **On this host, 4,096 `one` costs more CPU-seconds than 4,096 `place` or `scatter`,** although the counted multiply-adds differ by 10%: under `one` every cell is computed at every position.

## The trial (1,024 cells, seed 73129, 24,576 updates, scored every 256)

**Receipts and files** are in `exp45/e47/`. For each of `e47_32_{one,place}_{S,B}_73129` there are:
- `.json`: the receipt, with the curve, both diagnostics at every row, magnitudes, cosines, CPU, provenance and resumes;
- `_start.pt` and `.pt`: the starting and final weights, each with the sources table (cells × 11), periods, phases, interface and far indices, `Wc`, `bc`, `bfar` and `a`;
- `_answers.json`: the bytes of every answer on the length-test panel at 12 and 40 letters at the last row, plain, far cells at zero and far cells at their mean, with prompts and targets.

`one` and `place` each share one start across rules: start tensors 62de40fb… and 318c7841… respectively.

| Run | A80 (3 rows) | R50 | R80 (3 rows) | R90 | final A | final R | len12 | len40 (last 3) | use of far cells | CPU s/update* |
|---|---|---|---|---|---|---|---|---|---|---|
| `one` S | 13,568 | 9,728 | 17,664 | — | 93.3 | 86.3 | 93.0 | 90.8 | partly | 0.397 |
| `one` B | 10,496 | 6,400 | 11,008 | 17,152 | 96.2 | 94.7 | 94.7 | 95.2 | partly | 0.398 |
| `place` S | 15,360 | 10,496 | 19,200 | — | 89.2 | 82.0 | 89.0 | 87.0 | partly | 0.493 |
| `place` B | 10,496 | 6,400 | 11,264 | 15,360 | 95.8 | 95.2 | 97.0 | 96.6 | partly | 0.440 |
| E42 `F10` S | 13,312 | 9,472 | 15,104 | 23,296 | 92.2 | 88.5 | 91.7 | 89.1 | — | 0.105 |
| E42 `F10` B | 8,704 | 6,400 | 9,472 | 19,200 | 91.8 | 93.3 | 91.3 | 92.6 | — | 0.102 |
| E42 `P8` S | 7,680 | 6,400 | 8,448 | 12,288 | 95.7 | 92.5 | 96.0 | 96.0 | — | 0.138 |
| E42 `P8` B | 6,400 | 5,120 | 7,168 | 8,960 | 98.0 | 96.6 | 97.3 | 96.6 | — | 0.135 |
| E42 `G` S | 5,632 | 3,840 | 6,656 | 13,824 | 98.0 | 95.8 | 98.3 | 96.4 | — | 0.103 |
| E42 `G` B | 5,120 | 3,840 | 6,400 | 7,936 | 98.5 | 98.2 | 98.3 | 94.6 | — | 0.097 |

\* E47's CPU per update was measured with up to three E47 processes at once on a 4-core host. E42's figures are from E42's own receipts on its own host; the times are not comparable, and the brief compares in multiply-adds. CPU per scored row: 34.6 (`one` S), 33.6 (`one` B), 41.8 (`place` S) and 39.1 (`place` B) seconds.

### Curves (every 2,048 updates; plain | far cells at zero | far cells at their mean)

| Update | `one` S: R plain / 0 / mean | `one` B | `place` S | `place` B |
|---|---|---|---|---|
| 2,048 | 1.2 / 0.8 / 1.1 | 0.8 / 0.9 / 0.8 | 1.2 / 1.2 / 1.1 | 0.9 / 0.9 / 0.8 |
| 4,096 | 7.6 / 6.2 / 7.8 | 13.3 / 10.3 / 12.7 | 8.2 / 7.8 / 8.2 | 12.8 / 10.9 / 12.2 |
| 6,144 | 23.9 / 21.2 / 23.2 | 49.1 / 35.5 / 45.9 | 24.5 / 23.2 / 24.3 | 44.1 / 39.4 / 44.1 |
| 8,192 | 45.1 / 35.5 / 41.6 | 73.2 / 54.6 / 66.0 | 40.7 / 39.1 / 41.0 | 71.0 / 64.7 / 68.5 |
| 10,240 | 52.1 / 40.9 / 46.5 | 73.7 / 54.0 / 64.8 | 45.9 / 44.5 / 44.4 | 76.7 / 68.0 / 73.3 |
| 12,288 | 67.5 / 53.9 / 61.9 | 84.2 / 63.2 / 76.2 | 61.6 / 55.5 / 61.9 | 87.0 / 79.5 / 83.0 |
| 14,336 | 72.7 / 59.5 / 68.0 | 81.0 / 63.5 / 75.5 | 66.7 / 63.8 / 66.5 | 87.2 / 80.5 / 85.7 |
| 16,384 | 82.2 / 68.3 / 79.3 | 89.0 / 70.2 / 82.6 | 78.0 / 73.8 / 77.3 | 92.8 / 86.3 / 90.8 |
| 18,432 | 84.0 / 68.3 / 79.0 | 92.3 / 73.3 / 85.5 | 77.5 / 73.2 / 75.8 | 94.7 / 89.0 / 93.5 |
| 20,480 | 84.7 / 74.5 / 80.8 | 93.2 / 74.5 / 87.6 | 81.3 / 74.1 / 78.5 | 95.0 / 87.9 / 92.8 |
| 22,528 | 86.7 / 77.8 / 80.8 | 94.4 / 75.0 / 89.5 | 81.3 / 71.5 / 76.7 | 95.3 / 89.7 / 94.0 |
| 24,576 | 86.3 / 77.7 / 80.6 | 94.7 / 78.9 / 88.7 | 82.0 / 73.0 / 78.3 | 95.2 / 89.8 / 93.2 |

Every row (97 a run) with A, taught, len12, len40 and both diagnostics is in the receipts.

### Use of the far cells at the end (the brief's forms, first that applies)

| Run | A plain / zero / mean | R plain / zero / mean | len12 plain / zero / mean | len40 plain / zero / mean | Fall at zero (A, R) | Fall at mean (A, R) | Form |
|---|---|---|---|---|---|---|---|
| `one` S | 93.3 / 77.5 / 84.8 | 86.3 / 77.7 / 80.6 | 93.0 / 84.7 / 86.3 | 90.0 / 79.0 / 80.3 | 15.8, 8.6 | 8.5, 5.7 | **partly** |
| `one` B | 96.2 / 78.8 / 90.2 | 94.7 / 78.9 / 88.7 | 94.7 / 81.3 / 89.7 | 95.3 / 81.0 / 89.0 | 17.4, 15.8 | 6.0, 6.0 | **partly** |
| `place` S | 89.2 / 77.5 / 81.0 | 82.0 / 73.0 / 78.3 | 89.0 / 81.7 / 84.3 | 87.3 / 74.7 / 77.3 | 11.7, 9.0 | 8.2, 3.7 | **partly** |
| `place` B | 95.8 / 88.8 / 93.3 | 95.2 / 89.8 / 93.2 | 97.0 / 88.7 / 94.7 | 96.7 / 88.3 / 94.3 | 7.0, 5.4 | 2.5, 2.0 | **partly** |

No run reaches *used* (a fall of 15 or more at the mean). None is *not used* (both falls under 5 at zero), and none is *a constant is used*.

### Comparisons (E41's four cases; length test: mean of the last three rows at 40 letters, ±15 points)

| Comparison | R80 updates | A80 updates | R80 in multiply-adds | len40 |
|---|---|---|---|---|
| `one` S vs E42 `F10` S | about the same (17,664 / 15,104) | about the same | about the same | +1.7, within 15 |
| `one` B vs E42 `F10` B | about the same (11,008 / 9,472) | about the same | about the same | +2.7, within 15 |
| `place` S vs E42 `F10` S | about the same (19,200 / 15,104) | about the same | about the same | −2.1, within 15 |
| `place` B vs E42 `F10` B | about the same (11,264 / 9,472) | about the same | about the same | +4.0, within 15 |
| `place` S vs E42 `P8` S | more (19,200 / 8,448) | more | — (P8's count not stated) | −9.0, within 15 |
| `place` B vs E42 `P8` B | more (11,264 / 7,168) | more | — | +0.0, within 15 |
| `one` S vs E42 `G` S | more | more | more | −5.7, within 15 |
| `one` B vs E42 `G` B | more | more | about the same | +0.7, within 15 |
| `place` S vs E42 `G` S | more | more | more | −9.4, within 15 |
| `place` B vs E42 `G` B | more | more | about the same | +2.0, within 15 |
| `place` S vs `one` S | about the same (19,200 / 17,664) | about the same | about the same | −3.8, within 15 |
| `place` B vs `one` B | about the same (11,264 / 11,008) | about the same | about the same | +1.3, within 15 |

### Cosines between the sweep's and the exact gradient of `Wc` on the first batch, and magnitudes

| Run | start: interface / far | end: interface / far | share of far cells' recurrent weights with non-zero gradient, end (B / S) | mean \|W\| at the end: interface / far / sweep's scale |
|---|---|---|---|---|
| `one` S | 0.758 / 0.621 | 0.611 / 0.527 | 1.000 / 1.000 | 0.199 / 0.150 / 0.162 |
| `one` B | 0.758 / 0.621 | 0.658 / 0.473 | 1.000 / 1.000 | 0.179 / 0.243 / 0.227 |
| `place` S | 0.758 / 0.626 | 0.603 / 0.537 | 0.899 / 0.900 | 0.201 / 0.041 / 0.081 |
| `place` B | 0.758 / 0.626 | 0.617 / 0.443 | 0.900 / 0.900 | 0.176 / 0.123 / 0.136 |

- **Start.** All runs start at 0.152 / 0.150 / 0.151 (interface / far / scale).
- **Gradient coverage.** The share of non-zero far-cell gradients at the start is 1.00 under `one` and 0.90 under `place`, as the brief's count predicted (10% at 1,024 cells).
- **`place` S.** The far cells' mean |W| falls steadily through the run, from 0.150 to 0.041 (0.110 at 2,048, 0.082 at 8,192, 0.060 at 16,384), and the sweep's scale falls with it, 0.151 to 0.081. This is reported, not read.

## Predictions

| | Prediction | Chance | Outcome |
|---|---|---|---|
| P47.1 | Checks 1 and 2 pass under both rules | 85% | **Yes.** Check 1's self-check-1 lines differ from the Mac's only in rounding residues within tolerance; check 2 is within 1e-6 / 1e-5. |
| P47.2 | Backpropagation, `place`: reaches 80 on R | 70% | **Yes** (three rows from 11,264) |
| P47.3 | Sweep, `place`: reaches 80 on R | 45% | **Yes** (three rows from 19,200; final R 82.0, never 90) |
| P47.4 | Backpropagation, `one`: far cells *used* | 45% | **No** (*partly*: the fall at the mean is 6.0 on A and on R; at zero 17.4 / 15.8) |
| P47.5 | Sweep, `one`: far cells *used* | 30% | **No** (*partly*) |
| P47.10 | *A constant is used* in at least one run | 30% | **No** (all four *partly*) |
| P47.11 | Backpropagation: `place` about the same as `one` in updates to R80 | 50% | **Yes** (11,264 against 11,008) |
| P47.6 | `place`: end cosine lower for far than for interface cells by ≥ 0.1 | 60% | **Split, so no as stated.** Holds under backpropagation (0.617 − 0.443 = 0.174); does not under the sweep (0.603 − 0.537 = 0.066). The brief names one prediction for `place`; the reader counts it as holding only if both runs show it. |
| P47.7 | Backpropagation, `one`: updates to R80 about the same as or more than E42's `F10` | 80% | **Yes** (about the same: 11,008 against 9,472) |

## Interruptions, honest accounting

**The container restarted twice during the trial.**

- **First restart, near 23:15 UTC 16 November.** `one` B, `one` S and `place` B were at 11,264 to 12,544 updates. The runner had no resume and the runs were lost; their receipts and logs are kept in `exp45/e47/lost_at_restart/`. I added checkpoint and resume to `run()` (check 8) and restarted the three from their start.
- **Second restart, near 01:17 UTC 17 November.** The runs resumed from their checkpoints at 13,312 (`one` B), 12,288 (`place` B) and 9,216 (`one` S). The receipts list each resume.
- **The restarts changed nothing.** Over all 33 to 49 rows that the lost receipts share with the completed runs, every value is identical: scores, both diagnostics, losses, magnitudes and data digests. The lost runs were on runner 882b38f9, the completed runs on 8d80a7dc. The runner's diff between them is confined to `run()`'s checkpointing and check 8.
- **CPU.** The completed runs used 11.8 CPU-h of updates and 4.0 CPU-h of scoring, 15.8 CPU-h in all; check 7 forecast 18.9 for this host serial. The lost work was about 6.6 CPU-h, plus up to 1,024 updates a run at the second restart.
- **Earlier runner versions.** Two sets of check receipts were made on earlier runners and replaced; they are kept in `exp45/e47_qual/superseded/`. The first scored with whole-prefix decoding; the second still had the slower `one` path. The trial and the final checks used the runner described above.
- **Host.** Linux x86-64, 4 cores, Python 3.13.18, torch 2.15.0+cpu (installed on this host for the job), one thread per process, at most three processes. Runs were launched with `setsid nohup`, outside any wrapper with a time limit.

## What this trial does not show

The brief's own limits are not restated here beyond what the trial adds:
- **One seed.** Every reading is seed 73129's.
- **Measured time.** E47's CPU per update was measured beside two other E47 runs; E42's on its own host. The comparison is in multiply-adds only.
- **Answers past four bytes.** The far-mean diagnostic holds the far cells at their mean over the scored set at each position. Means at answer positions come from the network's own plain answers, so positions past the four greedy steps are never reached.

**Stopped.** No roster run was started; the roster is released after this report is read.
