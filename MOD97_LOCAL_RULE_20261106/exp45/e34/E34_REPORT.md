# E34 report: sweep births with each byte's error scaled by the network's own confidence (stage 1, seed 73129)

Brief: `OPUS_TASK_E34_ENTROPY_WEIGHTED_ERROR_20261115.md`, revision 2 (95d57fd1 on `codex/minifly-consolidated-20261023`).
Work branch: `claude/vibrant-curie-ivmrt1`. Evidence level E2; a one-seed screen that supports no claim by itself.
Host: Linux x86-64, Python 3.13.18, torch 2.15.0+cpu, numpy 2.5.4. One thread per process; at most three processes, shared with E33 addendum 1.

Code: `exp34_entropy.py` (runner), `e34_read.py` (reader) and `queue_e33a1_e34.sh` (resumable launcher). Nothing existing was edited, and every birth receipt records its sources as unmodified against its commit.

## Checks

| # | Check | Result | Receipt |
|---|---|---|---|
| 1a | U's loss vs `exp28_everybyte.loss_of(..., "all")` on the first three batches (U updates between) | batch 1 bit-equal; batches 2 and 3 differ by 4.8e-7 (relative 9.0e-8 and 9.6e-8), within 1e-6 | `E34_CHECKS.json` |
| 1b | First-batch gradients, U vs S-all's loss (logits and every parameter) | bit-equal | `E34_CHECKS.json` |
| 1c | U dry birth (2,048 updates) `tensors_sha256` vs `exp45/e28_qual/dry/e28_dev_S-all_dry2048_73129.json` | **equal** (b49561cf…); data digest equal | `dry/e34_dev_U_dry2048_73129.json`, `dry/GATE.json` |
| 2 | Logit gradient equals `(softmax - onehot) * w / w.sum()` at real positions (w recomputed independently in float64) and is exactly zero elsewhere | passed at all three temperatures; largest difference 4.4e-9 against a largest entry of 1.0e-3 (974 real, 434 non-real positions) | `E34_CHECKS.json` |
| 3 | At tau 1, no gradient through the gain: parameter gradients equal those with `w` replaced by a constant tensor of the same values | bit-equal | `E34_CHECKS.json` |
| 4 | `data_digest` equals S-all's at every saved row | asserted by the runner at every row of every birth, and rechecked from the receipts | birth receipts |
| 5 | Dry births of T050, T100 and T200 at 2,048 updates | 249, 244 and 267 CPU-s; all losses finite; forecast 2.48 CPU-h for the three full births (actual 2.46) | `E34_DRY.json` |

The dry births' weight diagnostics for their four blocks (distractor-to-answer weight ratio): T050 0.0027, 0.0022, 0.0019, 0.0018; T100 0.078, 0.057, 0.049, 0.047; T200 0.33, 0.26, 0.24, 0.23. These are the same numbers as the full births' first four blocks below. The full births started only after the gate in `dry/GATE.json` opened, which required U's hash to equal S-all's and every dry loss to be finite.

Reader reproduction (`E34_READ_REPRO.json`): `e28_dev_S-all_73129.pt` gives 75.33 / 2.0, and `e26_dev_S_73129.pt` gives 98.33 / 96.0, both exact.

## The three-arm table

Endpoint = mean of rows 16,384, 20,480, 24,576; the final row is in brackets.

| Arm | tau | Acc 12 | Acc 20 | Acc 40 | Keepers 12 / 20 / 40 | Keeps-and-writes 12 / 20 / 40 | Mark |
|---|---|---|---|---|---|---|---|
| T050 | 0.5 | 83.66 (99.33) | 82.56 (99.33) | 79.78 (99.33) | 76.7 / 70.3 / 65.7 (78 / 69 / 63) | 76.7 / 70.3 / 65.7 (78 / 69 / 63) | extrapolates |
| T100 | 1.0 | 95.33 (99.33) | 90.11 (96.33) | 57.56 (47.00) | 67.0 / 53.3 / 31.7 (67 / 53 / 29) | 63.7 / 52.7 / 31.7 (62 / 52 / 29) | intermediate |
| T200 | 2.0 | 86.11 (95.00) | 31.55 (36.00) | 2.00 (2.33) | 36.7 / 10.7 / 0.3 (34 / 10 / 1) | 33.0 / 10.3 / 0.3 (30 / 9 / 1) | fails |

Saved rows at 12 / 40 letters:

| Arm | 4,096 | 8,192 | 12,288 | 16,384 | 20,480 | 24,576 |
|---|---|---|---|---|---|---|
| T050 | 2.33 / 1.33 | 6.33 / 6.67 | 17.00 / 16.00 | 78.33 / 78.00 | 73.33 / 62.00 | 99.33 / 99.33 |
| T100 | 17.67 / 15.00 | 85.00 / 76.67 | 91.67 / 83.00 | 95.00 / 72.67 | 91.67 / 53.00 | 99.33 / 47.00 |
| T200 | 30.67 / 20.33 | 73.33 / 26.33 | 83.33 / 7.00 | 84.33 / 2.33 | 79.00 / 1.33 | 95.00 / 2.33 |

The full rows (12, 20 and 40 letters, keepers) are in `E34_READ_TABLES.md`.

By fine role, last block of the birth (updates 24,065–24,576). Each cell is loss (nats, unweighted CE) / mean entropy (nats) / mean weight / share of the weighted error (%):

| Arm | distractor | operand | code | operator | equals | answer digit | EOS |
|---|---|---|---|---|---|---|---|
| T050 | 3.433 / 3.450 / 0.00101 / 8.9 | 2.773 / 2.534 / 0.00904 / 36.5 | 1.404 / 1.381 / 0.0632 / 38.0 | 0.000 / 0.001 / 0.9987 / 0.0 | 0.000 / 0.000 / 0.9997 / 0.0 | 0.039 / 0.062 / 0.908 / 16.6 | 0.000 / 0.000 / 0.9995 / 0.0 |
| T100 | 3.420 / 3.464 / 0.0313 / 34.1 | 2.708 / 2.538 / 0.0880 / 45.3 | 1.406 / 1.378 / 0.252 / 18.7 | 0.000 / 0.000 / 0.9998 / 0.0 | 0.000 / 0.000 / 1.000 / 0.0 | 0.026 / 0.068 / 0.941 / 1.8 | 0.000 / 0.000 / 0.9999 / 0.0 |
| T200 | 3.415 / 3.469 / 0.177 / 49.0 | 2.700 / 2.542 / 0.289 / 39.6 | 1.401 / 1.384 / 0.501 / 9.5 | 0.000 / 0.002 / 0.9991 / 0.0 | 0.000 / 0.000 / 0.9999 / 0.0 | 0.092 / 0.179 / 0.919 / 1.9 | 0.000 / 0.000 / 0.9998 / 0.0 |

## Distractor-to-answer weight ratio

The ratio is the mean weight of a distractor position over the mean weight of an answer position (answer digits and EOS). The first eight blocks end at 512, 1,024, …, 4,096; the block at each later saved row is the one ending there.

| Arm | 512 | 1,024 | 1,536 | 2,048 | 2,560 | 3,072 | 3,584 | 4,096 | 8,192 | 12,288 | 16,384 | 20,480 | 24,576 | Projection (rows 4,096–24,576 of S-all) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T050 | 0.00269 | 0.00222 | 0.00189 | 0.00179 | 0.00168 | 0.00162 | 0.00156 | 0.00154 | 0.00148 | 0.00137 | 0.00121 | 0.00111 | 0.00108 | 0.0014 to 0.0019 |
| T100 | 0.0784 | 0.0565 | 0.0491 | 0.0469 | 0.0461 | 0.0456 | 0.0447 | 0.0437 | 0.0366 | 0.0338 | 0.0330 | 0.0328 | 0.0326 | 0.039 to 0.049 |
| T200 | 0.328 | 0.260 | 0.240 | 0.234 | 0.230 | 0.225 | 0.221 | 0.217 | 0.199 | 0.192 | 0.190 | 0.188 | 0.187 | 0.20 to 0.23 |

Share of the weighted error by role (%), for the first eight blocks and the end of the birth, is in `E34_READ_TABLES.md`. The first and last blocks, beside the projections (row 24,576 of S-all):

| Arm | | distractor | operand | code | answer digit |
|---|---|---|---|---|---|
| T050 | block to 512 / end / projection | 10.1 / 8.9 / 6 | 29.5 / 36.5 / 26 | 32.0 / 38.0 / 28 | 23.5 / 16.6 / 39 |
| T100 | block to 512 / end / projection | 28.4 / 34.1 / 30 | 34.6 / 45.3 / 42 | 14.7 / 18.7 / 17 | 20.7 / 1.8 / 11 |
| T200 | block to 512 / end / projection | 42.0 / 49.0 / 47 | 32.6 / 39.6 / 40 | 7.9 / 9.5 / 9 | 15.9 / 1.9 / 4 |

## Outcome row

Checked in the brief's order:

- **0** does not apply: check 1 passes.
- **1 applies.** T050 learned in range (83.66) and extrapolates (79.78). As the brief fixes the reading: "The gain rescues extrapolation on this seed at that temperature. A candidate on one seed; nothing is claimed. Stage 2 runs."

Stage 2 has not been run. The brief releases it after stage 1 is read; under outcome 1 it covers every arm that learned in range and extrapolates, which here is T050 only.

## Facts beside the row (not readings)

- **Dose order.** Accuracy at 40 letters for tau 0.5 / 1 / 2 is 79.78 / 57.56 / 2.00. It does not rise with tau.
- **Realised weights.** These are the two tables above. A distractor position's weight relative to an answer position's, in the first block and then at the end: T050 0.0027 → 0.0011 (projection 0.0014 to 0.0019); T100 0.078 → 0.033 (projection 0.039 to 0.049); T200 0.33 → 0.19 (projection 0.20 to 0.23).
- **Range.** Accuracy at 12 letters is 83.66 for T050, 95.33 for T100 and 86.11 for T200, beside the baseline S-all's 59.6 and B-all's 92.8.
- **Distance to the reference.** T050 is the only arm that extrapolates. At 79.78 it is within 15 points of B-all's 82.0 (2.22 below).

## Predictions

| | Prediction | Chance | Outcome |
|---|---|---|---|
| P34.1 | Check 1 passes: U's dry hash equals S-all's | 90% | **true** |
| P34.2 | T200 does not extrapolate | 75% | **true** (2.00) |
| P34.3 | T100 learns in range and extrapolates | 35% | **false**: learns in range (95.33), intermediate at 40 letters (57.56) |
| P34.4 | T050 learns in range and extrapolates | 45% | **true** (83.66; 79.78) |
| P34.5 | Outcome 1 | 55% | **true** |
| P34.6 | Accuracy at 40 letters does not rise with tau across the three arms | 60% | **true** (79.78 / 57.56 / 2.00) |
| P34.7 | All three arms learn in range | 50% | **true** (83.66, 95.33, 86.11) |
| P34.8 | At the end of T100, a distractor position's measured weight is between 0.02 and 0.10 of an answer position's | 65% | **true** (0.0326 in the last block) |

## Run record

- **Births.** Three full births of 24,576 updates, taking 2,977 (T050), 2,993 (T100) and 2,902 (T200) CPU-seconds; 2.46 CPU-hours in all. Four dry births of 2,048 updates. Source commits were 7d9600bc (T050), 789010f1 (T100) and f1dba2a9 (T200). These differ only in result files, and every receipt records its sources as unmodified.
- **Launch.** All ran under `queue_e33a1_e34.sh`, started with `setsid nohup` and outside any tool wrapper, so no wrapper time limit applied. It ran at most three jobs at once, shared with E33 addendum 1. Each birth's rows were read in its own slot right after the birth, then committed and pushed. The launcher is safe to rerun after a container reset: it recreates the venv, skips births that are complete and read, and moves partial outputs to `killed_partial/` before a rerun. No reset happened, and no birth was interrupted or rerun.
- **Committed weights.** For each full arm, the final `.pt` and rows 16,384 and 20,480 (9 files). Rows 4,096–12,288 were measured and their sha256 recorded, but are not committed.
- **NOT_RUN.** Stage 2 (seeds 93683 and 116089), which awaits the brief author's release.
