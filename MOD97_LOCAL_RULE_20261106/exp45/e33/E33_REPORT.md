# E33 report: which non-answer targets stop the sweep from extrapolating (stage 1, seed 73129)

Brief: `OPUS_TASK_E33_WHICH_TARGETS_BREAK_EXTRAPOLATION_20261115.md`, revision 2 (214169f8 on `codex/minifly-consolidated-20261023`).
Work branch: `claude/vibrant-curie-ivmrt1`. Evidence level E2; a one-seed screen that supports no claim by itself.
Host: Linux x86-64, Python 3.13.18, torch 2.15.0+cpu, numpy 2.5.4. One thread per process, at most three processes.

Code: `exp33_targets.py` (runner) and `e33_read.py` (reader). Nothing existing was edited. Every birth receipt records sources unmodified against its commit.

## Checks before the full births

| # | Check | Result | Receipt |
|---|---|---|---|
| 1a | A+DOS loss vs `exp28_everybyte.loss_of(..., "balanced")`, first three batches | bit-equal on all three (abs diff 0.0) | `E33_CHECKS.json` |
| 1b | First-batch gradients, A+DOS vs S-bal's loss (logits and every parameter) | bit-equal | `E33_CHECKS.json` |
| 1c | A+DOS dry birth (2,048 updates) `tensors_sha256` vs `exp45/e28_qual/dry/e28_dev_S-bal_dry2048_73129.json` | **equal** (5a7be045…) | `dry/e33_dev_A+DOS_dry2048_73129.json` |
| 1d | Full A+DOS `tensors_sha256` vs S-bal's `curve` at all six saved rows | **equal at all six** (and final) | `e33_dev_A+DOS_73129.json` |
| 2 | Logit gradient exactly zero outside `amask \| sel`; non-zero in every selected class | passed, all eight arms | `E33_CHECKS.json` |
| 3 | `data_digest` equals E28 S-bal's at every saved row | asserted at every row of every arm (dry: against E28's dry receipt) | birth receipts |
| 4 | Dry birth of every arm at 2,048 updates; cost forecast | 249–265 CPU-s each; forecast 6.8 CPU-h; actual 6.37 CPU-h (2,852–2,882 s per birth) | `E33_DRY.json` |

Selected positions on the first batch, by fine role: D = 334 distractor; O = 256 operand; S = 64 code + 64 operator + 64 equals; 782 non-answer positions in all.

Reader reproduction (`E33_READ_REPRO.json`): `e28_dev_S-bal_73129.pt` gives 91.33 / 6.0, and `e26_dev_S_73129.pt` gives 98.33 / 96.0 (12 / 40 letters), both exact.

## The eight-arm table

Endpoint = mean of rows 16,384, 20,480, 24,576; the final row is in brackets. Marks: learned in range acc12 ≥ 70; extrapolates acc40 ≥ 60; fails acc40 ≤ 20; intermediate between.

| Arm | Acc 12 | Acc 20 | Acc 40 | Keepers 12 / 20 / 40 | Keeps-and-writes 12 / 20 / 40 | Mark |
|---|---|---|---|---|---|---|
| A | 99.00 (98.67) | 98.67 (98.67) | 94.78 (94.00) | 102.0 / 91.7 / 73.0 (101 / 95 / 81) | 101.3 / 91.0 / 72.3 (101 / 95 / 81) | extrapolates |
| A+D | 86.11 (85.33) | 73.33 (68.33) | 23.78 (17.33) | 5.3 / 0.7 / 0.0 (3 / 0 / 0) | 5.3 / 0.7 / 0.0 (3 / 0 / 0) | intermediate |
| A+O | 98.33 (98.67) | 96.67 (96.33) | 90.22 (89.67) | 109.0 / 96.0 / 79.0 (109 / 93 / 71) | 102.0 / 89.0 / 72.3 (103 / 87 / 66) | extrapolates |
| A+S | 94.33 (97.67) | 93.44 (97.67) | 88.11 (92.33) | 100.7 / 90.3 / 72.7 (101 / 92 / 73) | 100.7 / 90.3 / 72.7 (101 / 92 / 73) | extrapolates |
| A+DO | 81.11 (82.00) | 47.89 (60.67) | 2.33 (2.00) | 8.0 / 0.0 / 0.0 (4 / 0 / 0) | 7.7 / 0.0 / 0.0 (4 / 0 / 0) | fails |
| A+DS | 83.11 (84.00) | 73.56 (74.67) | 38.11 (31.67) | 11.3 / 0.7 / 0.0 (6 / 0 / 0) | 11.3 / 0.7 / 0.0 (6 / 0 / 0) | intermediate |
| A+OS | 97.67 (98.00) | 97.44 (97.67) | 94.22 (95.67) | 103.3 / 94.7 / 83.0 (102 / 93 / 79) | 102.3 / 93.7 / 82.0 (102 / 93 / 79) | extrapolates |
| A+DOS | 86.55 (91.33) | 54.89 (63.33) | 4.11 (6.00) | 15.3 / 1.0 / 0.0 (12 / 0 / 0) | 15.3 / 1.0 / 0.0 (12 / 0 / 0) | fails |

Loss by fine role, last block of the birth (updates 24,065–24,576; nats per position; every role recorded whether selected or not):

| Arm | distractor | operand | code | operator | equals | answer_digit | eos |
|---|---|---|---|---|---|---|---|
| A | 18.341 | 9.464 | 6.901 | 30.559 | 27.077 | 0.019 | 0.000 |
| A+D | 3.271 | 8.774 | 8.696 | 21.640 | 21.310 | 0.087 | 0.000 |
| A+O | 20.464 | 2.261 | 17.294 | 21.226 | 21.604 | 0.022 | 0.000 |
| A+S | 13.106 | 16.041 | 1.405 | 0.000 | 0.001 | 0.044 | 0.000 |
| A+DO | 3.413 | 2.717 | 19.092 | 23.875 | 23.917 | 0.131 | 0.000 |
| A+DS | 3.272 | 10.814 | 1.397 | 0.000 | 0.000 | 0.082 | 0.000 |
| A+OS | 21.519 | 2.260 | 1.404 | 0.001 | 0.000 | 0.026 | 0.000 |
| A+DOS | 3.414 | 2.707 | 1.397 | 0.004 | 0.000 | 0.107 | 0.000 |

Every saved row of every arm is in `E33_READ_TABLES.md` and `E33_READ.json`.

## Outcome row

Rows are checked in the brief's order:

- **0a** does not apply: A extrapolates (94.78).
- **0b** does not apply: A+DOS fails (4.11), and its six row hashes equal S-bal's.
- **1** does not apply. The failing arms are {A+DO, A+DOS}. No set C of classes gives exactly these with every other arm extrapolating, because A+D and A+DS are intermediate.
- **2** and **3** do not apply. Row 3 needs A+D to extrapolate, and it is intermediate.
- **4 applies: Mixed.** As the brief fixes, the eight-arm table is reported with each class's mean over its four arms beside the mean of the four without it. No class is named.

| Class | Acc 12 with / without | Acc 20 with / without | Acc 40 with / without |
|---|---|---|---|
| D | 84.22 / 97.33 | 62.42 / 96.56 | 17.08 / 91.83 |
| O | 90.92 / 90.64 | 74.22 / 84.75 | 47.72 / 61.19 |
| S | 90.42 / 91.14 | 79.83 / 79.14 | 56.14 / 52.78 |

## Predictions

| | Prediction | Chance | Outcome |
|---|---|---|---|
| P33.1 | Neither 0a nor 0b occurs: A extrapolates and A+DOS fails | 80% | **true** (94.78; 4.11) |
| P33.2 | A+OS extrapolates | 40% | **true** (94.22) |
| P33.3 | A+D fails | 55% | **false**: intermediate (23.78; final row 17.33) |
| P33.4 | Outcome 1 with C of exactly one class | 20% | **false** (outcome 4) |
| P33.5 | Outcome 1 with C of all three classes | 20% | **false** (outcome 4) |
| P33.6 | Outcome 1 with C = {D, O} | 20% | **false** (outcome 4) |
| P33.7 | All eight arms learn in range | 80% | **true** (lowest endpoint acc12: A+DO, 81.11) |

## Run record

- **Source commits.** Births started from 26b62ae1 (A, A+D, A+DOS), 03e3a044 (A+O), 7d8c81b2 (A+S), df70c7b0 (A+DO) and d7e03c49 (A+DS, A+OS). Each of these differs from the others only in result files. `exp33_targets.py` is unchanged across them; the `sources_sha256` values are in each receipt.
- **Final data digest.** The same for every arm (eb49a8e4…), and equal to S-bal's.
- **Interrupted births.** The first A+DS and A+OS births were killed at about row 12,288 by the 2-hour limit on the background shell that launched them, not by the runner. Their partial receipts (`complete: false`, rows to 12,288) and logs are kept in `killed_partial/`. Both births were rerun from scratch; the reported A+DS and A+OS are the reruns. The killed rows 4,096, 8,192 and 12,288 have the same `tensors_sha256` as the reruns' rows (checked). The killed receipts are kept for the record only, and nothing was read from them.
- **Committed files (24 weight files).** For each arm: the final `.pt` and rows 16,384 and 20,480. Rows 4,096–12,288 were measured, and their sha256 values are in the receipts; they are not committed.
- **NOT_RUN.** Stage 2. The brief releases it only after stage 1 is read, and for outcome 4 the arms are to be named from the table before any run.

---

# Addendum 1: reduced stage 2 (A+D and A+OS on seeds 93683 and 116089)

Brief: addendum 1 of `OPUS_TASK_E33_WHICH_TARGETS_BREAK_EXTRAPOLATION_20261115.md`, with its note on row order, at 95d57fd1 on `codex/minifly-consolidated-20261023`. Same host and environment as stage 1. The runner `exp33_targets.py` is unchanged.

## Checks

| # | Check | Result | Receipt |
|---|---|---|---|
| — | `e33_read.py` gets an optional seed argument (default 73129) | `python3 e33_read.py tables` rewrites a byte-identical `E33_READ_TABLES.md` (sha256 829f2216…) and `E33_READ.json` for seed 73129 | commit 285ace55 |
| 1 | The reader reproduces `e28_dev_S-all_93683.pt` | 79.67 / 2.33 at 12 / 40 letters, exact. As references: `e28_dev_S-bal_93683.pt` 90.67 / 82.0 / 16.33 and `e28_dev_S-bal_116089.pt` 87.0 / 61.33 / 2.0 at 12 / 20 / 40 letters | `E33_READ_REPRO_STAGE2.json` |
| 2 | `data_digest` equals S-bal's of the same seed at every saved row | asserted by the runner at all six rows of all four births, and rechecked from the receipts | birth receipts |

## Per-seed table

Endpoint = mean of rows 16,384, 20,480, 24,576; the final row is in brackets.

| Seed | Arm | Acc 12 | Acc 20 | Acc 40 | Keepers 12 / 20 / 40 | Keeps-and-writes 12 / 20 / 40 | Mark |
|---|---|---|---|---|---|---|---|
| 93683 | A+D | 86.00 (86.33) | 76.67 (70.33) | 29.11 (15.33) | 2.0 / 1.0 / 0.0 (1 / 1 / 0) | 2.0 / 1.0 / 0.0 (1 / 1 / 0) | intermediate |
| 93683 | A+OS | 97.44 (97.33) | 97.11 (97.00) | 94.89 (93.67) | 103.3 / 98.7 / 82.0 (104 / 98 / 73) | 99.0 / 95.0 / 79.3 (100 / 94 / 72) | extrapolates |
| 116089 | A+D | 87.89 (82.33) | 75.67 (62.00) | 31.56 (10.00) | 19.0 / 3.0 / 0.0 (11 / 0 / 0) | 16.3 / 2.7 / 0.0 (10 / 0 / 0) | intermediate |
| 116089 | A+OS | 93.11 (95.00) | 91.78 (92.67) | 86.89 (88.33) | 102.7 / 95.3 / 82.3 (103 / 96 / 82) | 100.3 / 93.0 / 80.0 (100 / 93 / 79) | extrapolates |

A+OS minus A+D at 40 letters (endpoints): 65.78 points on 93683, 55.33 on 116089. Every saved row is in `E33_READ_TABLES_93683.md` and `E33_READ_TABLES_116089.md`.

## Rows R1 to R4 (checked in the order R4, R2, R3, R1)

| Seed | R4 (an arm did not learn in range) | R2 (A+OS does not extrapolate) | R3 (A+D extrapolates, or within 40 points of A+OS) | Row that applies |
|---|---|---|---|---|
| 93683 | no (86.00, 97.44) | no (94.89) | no (29.11; 65.78 points below) | **R1**: the contrast replicates on this seed |
| 116089 | no (87.89, 93.11) | no (86.89) | no (31.56; 55.33 points below) | **R1**: the contrast replicates on this seed |

R1 holds on both new seeds, so the addendum's sentence applies: "On three seeds, a sweep birth with answer and distractor targets does not extrapolate to 40 letters, and one with answer, operand and task-code targets does."

## Predictions

| | Prediction | Chance | Outcome |
|---|---|---|---|
| P33.8 | A+OS extrapolates on both new seeds | 75% | **true** (94.89, 86.89) |
| P33.9 | A+D does not extrapolate on either new seed | 85% | **true** (29.11, 31.56) |
| P33.10 | A+D fails (at most 20) on both new seeds | 35% | **false**: intermediate on both (final rows 15.33 and 10.00; the mark is on the endpoint) |
| P33.11 | R1 holds on both new seeds | 65% | **true** |

## Run record

- **Births.** Four births of 24,576 updates each: 2,842–2,948 CPU-seconds per birth, 3.19 CPU-hours in all. Source commits were c8be47ef (93683), 654d6c4c (A+D 116089) and f90a03b2 (A+OS 116089). These differ only in result files, and every receipt records its sources as unmodified.
- **Launch.** All four ran under the resumable queue `queue_e33a1_e34.sh`, started with `setsid nohup` and outside any tool wrapper, so no wrapper time limit applied. They shared at most three process slots with E34. Each birth's rows were read in its own slot right after the birth, then committed and pushed.
- **One slip.** I ran the read-only `tables 93683` command for a few seconds while three births were running, so four processes ran at once for that time.
- **No interruptions.** No birth was interrupted or rerun, and nothing was moved to `killed_partial/` in this stage.
- **Committed weights.** For each birth, the final `.pt` and rows 16,384 and 20,480 (12 files). Rows 4,096–12,288 were measured and their sha256 recorded, but are not committed.
- **NOT_RUN.** The control `A+D-light`, which is not part of this stage.
