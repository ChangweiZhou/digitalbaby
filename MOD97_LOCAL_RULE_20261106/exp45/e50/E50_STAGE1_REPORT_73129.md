# E50, stage 1, seed 73129: the fixed code with the dense interface (`C-dense`) under both rules

Brief: `OPUS_TASK_E50_AN_INPUT_CODE_IN_WHICH_A_VALUE_STANDS_FOR_ONE_BYTE_20261116.md`, revision 2 (d6606454 on this branch). One seed, a first measurement: a description, not a claim. The reviewer has verified this stage from the receipts and the weight files (`PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/E50_STAGE1_VERIFICATION_20261117.md` with `e50_stage1_verify.py`, on `claude/vibrant-cerf-sje726` at 449cf964); no formal checker was run.

**Terms.** The *code* of a byte is the 512 values that feed the input map and `l2`. In this job it is fixed: each of the 46 used bytes turns on 11 values of its own, each equal to 3.2387, and every other value is zero. `C-dense` is E42's `F10` network (256 cells, each reading 10 fixed cells of its neighbourhood, every cell reading all 512 values and every readout unit all 256 cells) with this code in place of the learned one; the byte table and `l1` take no part. `F10` is the same network with the learned code (E42's run, on file). *A* is the share of right answers on untaught pairs of add and sub at 12 letters (300 items each); *R* the share right on 1,000 untaught pairs of each of add and sub without letters whose answer needs the reduction by the modulus; the *length test* is 300 untaught pairs of add at 12 and at 40 letters. A threshold is *reached* at the first of three consecutive scored rows at or above it (rows every 256 updates). The *sweep* (S) is the sign rule; *backpropagation* (B) the exact gradient. A *level* is the mean over the last three scored rows.

**Runs.** `python3 exp50_code.py run C-dense <S|B> 73129`, runner at 0f3ec4bc, 24,576 updates each, launched detached by `exp45/e50/launch_e50.sh`, two processes beside each other, one thread each. Both started at 20:47:21Z; B ended at 21:40:44Z and S at 21:48:16Z (`exp45/e50/logs/queue.log`). The `rc=0` that `logs/queue.log` gives at each end is not the run's exit code (the launch script wrote the status of a command substitution; corrected at b1a516ca). A run is complete by the line `e50 complete` of its log and by `"complete": true` with the final hashes in its receipt, which both runs of this stage have. Neither run was restarted. Cost: 0.87 CPU-hours (B) and 1.00 (S), scoring included. Every figure below is from `exp45/e50/E50_TABLES_73129_stage1.md`, which `python3 e50_tables.py --tag 73129_stage1` writes from the receipts `exp45/e50/e50_C-dense_<rule>_73129.json` and from the reference receipts in `exp45/e50/reference/`.

## Result

| | `C-dense`, B | `F10`, B | `C-dense`, S | `F10`, S |
|---|---|---|---|---|
| Updates to 80 on A | 7,680 | 8,704 | 13,056 | 13,312 |
| Updates to 80 on R | 6,912 | 9,472 | 12,800 | 15,104 |
| Level of A | 97.7 | 93.2 | 93.5 | 91.2 |
| Level of R | 97.1 | 94.1 | 94.1 | 89.3 |
| Level of the length test at 40 letters | 92.7 | 92.6 | 90.6 | 89.1 |
| R: last three rows minus the three ending 2,048 updates earlier | +1.1 | +2.1 | +3.0 | +0.7 |

**The network learns the first family from the fixed code under both rules** (it reaches 80 on R within the dose): at 6,912 updates under backpropagation and at 12,800 under the sweep.

**Against E42's `F10` under the same rule (the brief's forms).** Updates to 80 on A and on R are *about the same* under both rules (ratios 0.88 and 0.73 under backpropagation, 0.98 and 0.85 under the sweep; *fewer* would need a ratio below two thirds). In multiply-adds (first count: 926,464 a position against `F10`'s 930,560) the reading is *stays* in all four cells. The three levels are *about the same* under both rules (differences of +0.1 to +4.8 points, all in favour of `C-dense`; the form needs 15). The single-row version gives the same descriptions. By the second count (products with a turned-on value, 285,184 a position) `C-dense` reaches 80 on R after 1.26e+11 (B) and 2.34e+11 (S) products; None of `F10`'s code values is exactly zero, so its second count equals its first, 930,560 a position; `C-dense` then needs 0.22 (B) and 0.26 (S) of `F10`'s products to 80 on R.

**The four operations apart** (untaught pairs at 12 letters, 300 items each; updates to 80, three-row version).

| | max | min | add | sub | Levels max, min, add, sub |
|---|---|---|---|---|---|
| `C-dense`, B | 1,024 | 1,280 | 7,680 | 7,680 | 97.8, 96.9, 97.3, 98.0 |
| `F10`, B | 2,048 | 2,048 | 8,704 | 9,472 | 97.0, 97.4, 92.9, 93.6 |
| `C-dense`, S | 1,280 | 1,536 | 12,032 | 13,824 | 97.7, 95.7, 94.0, 93.0 |
| `F10`, S | 3,072 | 2,816 | 11,776 | 15,104 | 97.8, 97.8, 90.1, 92.3 |

**The answer's two bytes apart** (R's items; levels): the first byte is right in 98.1% and the second in 97.6% of items under backpropagation, and in 96.0% and 94.3% under the sweep. Early in both runs the first byte is right more often than the second (42 against 13 at 2,048 updates under B; 60 against 26 at 4,096 under S).

**Other measures.** A cell's input for a needed byte (root mean square, without the bias) grows from 0.389 at the start to 5.14 (B) and 4.77 (S). The mean magnitude of a weight of the input map grows from 0.031 to 0.094 (B) and 0.086 (S). The cosine between the sweep's and the exact gradient on the first batch, at the start and at the end: input map 0.757 and 0.650 (B run), 0.721 (S run); `lm` 0.867 and 0.711, 0.875; recurrent map 0.723 and 0.639, 0.698. The figure "by what a cell knows" (the three groups of cells mean nothing in `C-dense`; it is the baseline for `C16-in`): mean magnitude of the kept recurrent weights from neighbours, at the start 0.1498, 0.1515, 0.1501 for the three groups, and at the end 0.1467, 0.1440, 0.1448 (B) and 0.1501, 0.1518, 0.1491 (S).

**Checks inside the runs.** At all 97 scored rows of both runs: the code table has the SHA-256 of the start and the byte table and `l1` are bit-equal to their start; every weight outside a mask is zero and the optimizer's averages there are zero; E42's limit holds. The batch digest at row 4,096 is `bbd71ab7ca27b72e`, E26's. The optimizer holds no state for the byte table or `l1`. The receipts record `byte_table_and_l1_bit_equal_to_start: true`.

**Files for the later job.** Final weights `exp45/e50/e50_C-dense_S_73129.pt`, SHA-256 `056ca79d181f1ec2192ea653aca84f1ad9e739f0a6c90c0b06f24245f8ea78c4`; `exp45/e50/e50_C-dense_B_73129.pt`, SHA-256 `af1315dd2cd048a103238f5fe5517bdae50d1ca397973c1bbe80ccafef3e386a` (both recorded in the receipts under `weights_sha256`); starting network `e50_C-dense_<rule>_73129_start.pt`; answers on the length-test panel `e50_C-dense_<rule>_73129_answers.json`.

## Predictions

| | Prediction | Chance | Outcome |
|---|---|---|---|
| P50.1 | Checks 1 to 4 pass | 85% | true |
| P50.2 | `C-dense`, backpropagation: reaches 80 on R within the dose | 80% | true (6,912 updates) |
| P50.3 | `C-dense`, sweep: reaches 80 on R within the dose | 55% | true (12,800 updates) |
| P50.8 | `C-dense` against `F10`, backpropagation: updates to 80 on R *about the same* or *fewer* | 60% | true (*about the same*, ratio 0.73) |

## What follows

By the brief's rule stage 2 runs under both rules: `C16-in` (16 values a cell, E49's lists; the readout dense), 49,152 updates. With `C-dense` at 6,912 and 12,800 updates to 80 on R, a factor of up to 7.0 (B) and 3.8 (S) can be recorded within 48,640 updates, and E49's factor of 2.16 falls within the first dose under backpropagation (14,930 updates).

## What this stage does not show

One seed. A difference from `F10` of this size (ratios 0.73 to 0.98) is not read as a gain: it is inside the form's *about the same*. The spread between seeds of `F10` (E42's three seeds, receipts on E42's branch at 44f91edc): updates to 80 on A 6,912 to 9,216 and on R 7,936 to 9,728 under backpropagation, and 12,544 to 13,312 on A and 14,336 to 15,104 on R under the sweep. `C-dense` lies inside that range on A and is earlier than every seed of `F10` on R (6,912 and 12,800); with one seed of `C-dense` this is still not read as a gain. `C-dense` differs from `F10` in four things at once (the code is not learned; a value stands for one byte; every byte has the same length; how far one update moves a cell's input), which the job does not separate. Nothing is said here about a limited input.

## Amendments, 18 November 2026

On the reviewer's request after the verification at 449cf964, four passages were changed: the citation of the verification; the end of the runs (the `rc=0` of the queue log is not an exit code); `F10`'s second count of multiply-adds; and the spread between `F10`'s seeds, which the first version said was not read. The figures of the seed spread were checked by the worker from E42's receipts (E49's worker's copies at a36de661 and E42's branch at 44f91edc for `F10` B 116089). No result changed.
