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

---

# Addendum 1: stage 2 (T050 on seeds 93683 and 116089) and B050 on seed 73129

Brief: addendum 1 of `OPUS_TASK_E34_ENTROPY_WEIGHTED_ERROR_20261115.md` at 4dd0b9f7 on `codex/minifly-consolidated-20261023`. Same host and environment as stage 1.

## Code and checks

- **The B arms.** `exp34_entropy.py` gets two arms, `BU` (tau infinite) and `B050` (tau 0.5). Their only difference is `local = False` in the forward. They use B-all's receipts as the reference (commit 238b75bf). The existing arms' path is unchanged: `step` takes `local` with default `True`, and `dev` sets it from the arm.
- **Reader.** `e34_read.py` gets an optional seed argument and a `tables_a1` command. `python3 e34_read.py tables` still rewrites a byte-identical `E34_READ_TABLES.md` (sha256 23a99351…) and `E34_READ.json`.
- **Check before B050 runs.** The `BU` dry birth at 2,048 updates has `tensors_sha256` 39a50be3…, equal to `exp45/e28_qual/dry/e28_dev_B-all_dry2048_73129.json`. Its data digest is equal and its losses are finite. Receipts: `dry_a1/e34_dev_BU_dry2048_73129.json` and `dry_a1/GATE_B050.json`. B050 ran only after this gate opened.
- **Regression of the unchanged path.** A U dry birth on the edited runner (`dry_a1/e34_dev_U_dry2048_73129.json`) again gives S-all's dry hash, b49561cf….
- **Same data.** Each birth's `data_digest` equals its reference's at every saved row: S-all of the same seed for T050, B-all for B050.

## Per-seed table

Endpoint = mean of rows 16,384, 20,480, 24,576; the final row is in brackets. Marks as in stage 1.

| Seed | Arm | Acc 12 | Acc 20 | Acc 40 | Keepers 12 / 20 / 40 | Keeps-and-writes 12 / 20 / 40 | Mark |
|---|---|---|---|---|---|---|---|
| 73129 | T050 (stage 1) | 83.66 (99.33) | 82.56 (99.33) | 79.78 (99.33) | 76.7 / 70.3 / 65.7 (78 / 69 / 63) | 76.7 / 70.3 / 65.7 (78 / 69 / 63) | extrapolates |
| 93683 | T050 | 69.89 (98.00) | 69.33 (98.00) | 67.22 (96.33) | 96.3 / 90.0 / 78.3 (105 / 98 / 85) | 94.3 / 88.0 / 76.7 (103 / 96 / 83) | not learned in range |
| 116089 | T050 | 92.22 (96.33) | 91.56 (95.67) | 88.89 (93.00) | 90.3 / 86.3 / 79.7 (94 / 89 / 84) | 86.3 / 82.7 / 76.0 (91 / 86 / 81) | extrapolates |
| 73129 | B050 | 65.67 (100.00) | 65.11 (100.00) | 64.11 (100.00) | 91.3 / 88.3 / 83.0 (91 / 88 / 81) | 91.3 / 88.3 / 83.0 (91 / 88 / 81) | not learned in range |

T050 on 93683 is 0.11 below the "learned in range" mark of 70, so, as the marks fix, it is not read for extrapolation.

T050 does not learn in range and extrapolate on both new seeds. The condition for the brief's sentence is therefore not met, and the sentence is not stated.

## The two added facts

**Fact 1: the first saved row at which accuracy at 12 letters reaches 70.** "Plain backpropagation" is E28 B-all and "plain sweep" is E28 S-all, from `E32_READ.json`.

| Seed | Arm | Arm's first row | Plain backpropagation | Plain sweep |
|---|---|---|---|---|
| 73129 | T050 | 16,384 | 4,096 | 24,576 |
| 93683 | T050 | 20,480 | 4,096 | 24,576 |
| 116089 | T050 | 16,384 | 4,096 | 16,384 |
| 73129 | B050 | 20,480 | 4,096 | 24,576 |

**Fact 2: the three rows at 40 letters, beside their mean.**

| Seed | Arm | 16,384 | 20,480 | 24,576 | Mean |
|---|---|---|---|---|---|
| 73129 | T050 | 78.00 | 62.00 | 99.33 | 79.78 |
| 93683 | T050 | 33.00 | 72.33 | 96.33 | 67.22 |
| 116089 | T050 | 87.33 | 86.33 | 93.00 | 88.89 |
| 73129 | B050 | 17.33 | 75.00 | 100.00 | 64.11 |

Every saved row is in `E34_ADDENDUM1_TABLES.md`; the machine-readable form is `E34_ADDENDUM1.json`.

## B050's reading

B050 reaches 65.67 at 12 letters, below 70. The case is **"does not learn in range"**, and the reading is: "Under backpropagation the gain at this temperature prevents in-range learning on this seed."

For reference, B050's endpoint at 40 letters is 64.11 against plain backpropagation's 82.0 (17.89 below) and T050's 79.78. These are not read, because B050 did not learn in range.

## Predictions

| | Prediction | Chance | Outcome |
|---|---|---|---|
| P34.9 | T050 learns in range and extrapolates on both new seeds | 40% | **false** (93683: not learned in range, 69.89) |
| P34.10 | T050 does so on at least one new seed | 75% | **true** (116089) |
| P34.11 | T050 first reaches 70 at 12 letters at row 12,288 or later on both new seeds | 65% | **true** (20,480 and 16,384) |
| P34.12 | B050 is below 70 at 12 letters at row 8,192 | 60% | **true** (8.33) |
| P34.13 | B050 learns in range and is within 15 of 82.0 at 40 letters | 50% | **false** (not learned in range) |

## Run record

- **Births.** Three births of 24,576 updates: T050 on 93683 took 2,866 CPU-seconds, T050 on 116089 2,867, and B050 2,699. All losses were finite.
- **Source commits.** 238b75bf for both T050 births, and 2eade79b for B050, which differs from 238b75bf only in result files. Every receipt records its sources as unmodified.
- **Dry births.** BU and U, 2,048 updates each.
- **Launch.** All ran under `queue_e34a1.sh`, started with `setsid nohup` with at most three processes. Each birth's rows were read in its slot right after it, then committed and pushed. Nothing was interrupted or moved aside.
- **Committed weights.** For each birth, the final `.pt` and rows 16,384 and 20,480 (9 files). Rows 4,096–12,288 were measured and their sha256 recorded, but are not committed.

---

# Addendum 2: two long births (T050 and B-all, seed 73129, 98,304 updates)

Brief: addendum 2 of `OPUS_TASK_E34_ENTROPY_WEIGHTED_ERROR_20261115.md` at 18751457 on `codex/minifly-consolidated-20261023`. Same host and environment. Every statement here carries its horizon: 98,304 updates, one seed.

## Code and checks

- **Code change (commit be01b07c).** `exp34_entropy.py` makes the smallest change the addendum names. For a dose above the full dose it uses the full-dose reference, and it compares the data digest and hash only at rows the reference has. Rows beyond the reference are recorded as skipped (`data_digest_check: "skipped: …"`, `tensors_equal_to_reference: null`).
- **Regression after the change.** A T050 dry birth at 2,048 updates gives `tensors_sha256` 9da02e11…, equal to `exp45/e34/dry/e34_dev_T050_dry2048_73129.json` (`dry_a2/REGRESSION.json`).
- **`exp28_everybyte.py` is unchanged.** It ran as it is with `E11_DEV_UPDATES=98304`. Its digest check against E26's receipts applies at rows 4,096–24,576 and is marked "not asserted" beyond them.
- **Row-24,576 hash check: passed for both.** T050's row 24,576 is dd95a2e8…, equal to `exp45/e34/e34_dev_T050_73129.json`. B-all's is 62a33eb1…, equal to `exp45/e28/e28_dev_B-all_73129.json`. The check was made live during the births, and they would have been stopped on a mismatch.
- **File names.** Both runners tag a non-default dose as `_dry98304`. The files are full births of 98,304 updates despite the tag.

## Rows (accuracy at 12 / 20 / 40 letters)

| Row | T050 | B-all | T050 above at 40 |
|---|---|---|---|
| 4,096 | 2.33 / 1.67 / 1.33 | 73.67 / 70.67 / 57.67 | |
| 8,192 | 6.33 / 6.33 / 6.67 | 89.33 / 87.67 / 78.00 | |
| 12,288 | 17.00 / 17.33 / 16.00 | 95.00 / 93.67 / 87.00 | |
| 16,384 | 78.33 / 78.67 / 78.00 | 91.33 / 90.33 / 83.67 | |
| 20,480 | 73.33 / 69.67 / 62.00 | 93.33 / 91.00 / 77.67 | |
| 24,576 | 99.33 / 99.33 / 99.33 | 93.67 / 94.00 / 84.67 | |
| 28,672 | 98.67 / 98.33 / 95.33 | 96.00 / 96.33 / 89.33 | |
| 32,768 | 91.67 / 90.33 / 87.67 | 97.00 / 97.00 / 90.33 | |
| 36,864 | 98.00 / 98.33 / 95.67 | 95.67 / 93.33 / 88.67 | |
| 40,960 | 96.67 / 97.33 / 96.33 | 99.00 / 99.00 / 96.67 | |
| 45,056 | 94.67 / 94.00 / 91.00 | 95.00 / 95.00 / 92.67 | |
| 49,152 | 98.67 / 98.33 / 98.00 | 97.00 / 97.00 / 94.67 | yes |
| 53,248 | 100.00 / 99.67 / 98.67 | 98.33 / 98.00 / 95.00 | yes |
| 57,344 | 96.00 / 96.33 / 96.67 | 98.00 / 97.00 / 91.33 | yes |
| 61,440 | 95.33 / 95.00 / 91.33 | 97.67 / 97.67 / 95.67 | no |
| 65,536 | 99.67 / 99.67 / 98.00 | 98.00 / 98.67 / 98.67 | no |
| 69,632 | 98.00 / 97.67 / 96.67 | 97.33 / 96.33 / 91.00 | yes |
| 73,728 | 98.67 / 99.00 / 97.67 | 98.67 / 98.33 / 92.33 | yes |
| 77,824 | 99.33 / 99.67 / 99.67 | 99.00 / 97.67 / 93.00 | yes |
| 81,920 | 100.00 / 100.00 / 100.00 | 97.67 / 96.67 / 88.00 | yes |
| 86,016 | 99.67 / 99.67 / 99.33 | 98.33 / 98.33 / 92.67 | yes |
| 90,112 | 99.33 / 99.00 / 98.33 | 98.33 / 97.67 / 90.67 | yes |
| 94,208 | 99.00 / 98.00 / 95.67 | 98.00 / 97.67 / 89.67 | yes |
| 98,304 | 94.67 / 94.00 / 91.33 | 99.00 / 99.00 / 91.67 | no |

Keepers and keeps-and-writes for every row are in `exp45/e34_long/E34_LONG_TABLES.md` and `E34_LONG_READ.json`.

## Reading (addendum 2's rule, at 40 letters over the 13 rows 49,152–98,304)

| Run | Mean acc 40 | Mean acc 20 | Mean acc 12 |
|---|---|---|---|
| T050 | 97.03 | 98.15 | 98.33 |
| B-all | 92.64 | 97.69 | 98.10 |

T050 minus B-all is **+4.38 points**, and T050 is above B-all in **10 of 13 rows**.

The rule's first row needs both a mean at least 10 points above and at least 10 of 13 rows above. Only the second condition holds. The second row ("not above") does not apply either, so the result is **mixed**. As the addendum fixes it: "Mixed; the rows are reported."

## Predictions

| | Prediction | Chance | Outcome |
|---|---|---|---|
| P34.14 | The row-24,576 hash check passes for both runs | 85% | **true** |
| P34.15 | Overtakes and stays | 30% | **false** (mixed: +4.38, 10 of 13) |
| P34.16 | Does not overtake | 35% | **false** (T050's mean is above B-all's) |
| P34.17 | B-all's mean at 40 letters over the 13 rows is at least 10 below its present endpoint of 82.0 | 20% | **false** (92.64) |

## Run record

- **Cost.** T050 took 10,875 CPU-seconds and B-all 10,083: 5.82 CPU-hours.
- **Provenance.** Both started from commit 062bb820, with sources recorded as unmodified. All losses were finite.
- **Launch.** Both ran under `queue_e34a2.sh`, started with `setsid nohup` once addendum 1's queue had ended. The two births ran together, with one reader loop as the third process; it read every row as it appeared, committed the reader file and made the live hash check. Nothing was interrupted or moved aside.
- **Committed weights.** For each run, rows 49,152 and 73,728 and the final row 98,304 (6 files). The other rows were measured, and their sha256 are in the receipts.
- **Next.** B050 at 98,304 updates (the user's request) is running under `queue_e34a3.sh`; its section follows.
