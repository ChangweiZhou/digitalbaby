# E50 report, seed 73129: an input code in which a value stands for one byte

Brief: `OPUS_TASK_E50_AN_INPUT_CODE_IN_WHICH_A_VALUE_STANDS_FOR_ONE_BYTE_20261116.md`, revision 2, entered on this branch at d6606454 from the reviewer's audit (`AUDIT_E50_SPARSE_INPUT_CODE_BRIEF_20261117.md` at ead8dcd8 on `claude/vibrant-cerf-sje726`) in the coordinator's absence, on the reviewer's instructions (`COORDINATION_20261117/WORKER_E50_FIXED_INPUT_CODE.md` there). **One seed, a first measurement: a description, not a claim.** Stages 1 and 2 have been verified by the reviewer from the receipts and the weight files (`PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/E50_STAGE1_VERIFICATION_20261117.md` at 449cf964 and `E50_STAGE2_VERIFICATION_20261118.md` at 4b49ee7e, on `claude/vibrant-cerf-sje726`); stage 3 has not been verified. No formal checker was run (none is installed as runnable software on this branch).

## Provenance

- **Branch** `claude/e50-fixed-input-code-017dSE`, from the default branch `codex/minifly-consolidated-20261023` at 7cabb13a. Every file of the job is new; no existing file was edited (the brief's revision 2 replaces revision 1 at its own path on this branch only).
- **Runner** `MOD97_LOCAL_RULE_20261106/exp50_code.py`, committed at 0f3ec4bc and unchanged through all runs (its SHA-256 is in every receipt). It imports, unchanged: the specification's copy `e50_sparse_code.py` (the default branch's `PROJECT_DOCUMENTS/COORD_CHECKS_20261116/e50_sparse_code_spec.py` with its two import lines pointed at the copies); `e49_interface.py` and `exp49_interface.py`, byte-identical to E49's worker's files at a36de661 (SHA-256 `92f968e0…` and `cdf93ce1…`, `exp45/e50_qual/COPIES.txt`; the runner stops if either differs); and E42's `e42_cells.py` and `exp42_cells.py` of the default branch. Tables: `e50_tables.py` (receipts only).
- **Host**: Linux x86-64, 4 cores, Python 3.13.18, torch 2.15.0+cpu, numpy 2.5.5, in a virtual environment; one thread a process (`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `torch.set_num_threads(1)`); at most three processes at once; runs launched with `setsid nohup` by `exp45/e50/launch_e50.sh`, outside any wrapper with a time limit. The versions are recorded in every receipt.
- **Reference receipts**, not run again: E42's `F10` and E49's `in16` of seed 73129, copied byte for byte from a36de661 into `exp45/e50/reference/` (SHA-256 in `SOURCE.txt` there).
- **Figures**: every figure below is from `exp45/e50/E50_TABLES_73129.md` and `E50_SUMMARY_73129.json`, written by `python3 e50_tables.py --tag 73129` from the receipts, unless another receipt is named. Stage reports: `exp45/e50_qual/E50_STAGE0_73129.md` (checks), `exp45/e50/E50_STAGE1_REPORT_73129.md`, `exp45/e50/E50_STAGE2_REPORT_73129.md`.

## Terms

- The *code* of a byte is the 512 values that feed the input map (`rnn.weight_ih_l0`) and `l2`. Here it is fixed: each of the 46 *used* bytes (the ten digits, `+ - > <`, `=`, the 26 capital letters, `p q v w`, the start symbol) turns on 11 values of its own, each equal to 3.2387, and every other value is zero. The *needed* bytes are the 20 that the task must tell apart (all but the letters). A cell *has input for* a byte if it reads at least one of the byte's 11 values.
- `C-dense`: E42's `F10` layer (256 cells, each reading 10 fixed cells of its neighbourhood and itself) with this code; every cell reads all 512 values, every readout unit all 256 cells. `C16-in`: each cell reads 16 values (E49's lists for K_IN = 16), so that it has input for 1 to 10 of the 20 needed bytes (median 6) and 0 to 7 of the ten digits (median 3). `C46-one` (the audit's control): each cell reads one value of every used byte, 46 values, so that it has input for every byte, each through one weight. In all three the readout is dense. E42's `F10` and E49's `in16` are the same network and the same 16-value lists on the *learned* code, in which every cell can identify every byte.
- *A*: share of right answers on untaught pairs of add and sub at 12 letters (300 each). *R*: share right on 1,000 untaught pairs of each of add and sub without letters whose answer needs the reduction by the modulus. *Length test*: 300 untaught pairs of add at 12 and at 40 letters. The same panel as A holds 300 pairs of each of max and min.
- A threshold is *reached* at the first of three consecutive scored rows at or above it (the three-row version; rows every 256 updates). A *level* is the mean over the last three scored rows. The *sweep* (S) is the sign rule; *backpropagation* (B) the exact gradient. The forms (E41's four cases; *lower* or *higher* at 15 points or more; *still rising* at 5 points or more over 2,048 updates) are the brief's.

## Runs

| Stage | Setting | Rule | Updates | Started | Ended | CPU-hours | Restarted |
|---|---|---|---|---|---|---|---|
| 1 | `C-dense` | B | 24,576 | 17 Nov 20:47:21Z | 21:40:44Z | 0.87 | no |
| 1 | `C-dense` | S | 24,576 | 20:47:21Z | 21:48:16Z | 1.00 | no |
| 2 | `C16-in` | B | 49,152 | 21:49:42Z | 23:37:13Z | 1.76 | no |
| 2 | `C16-in` | S | 49,152 | 21:49:42Z | 23:51:31Z | 2.00 | no |
| 3 | `C46-one` | B | 49,152 | 23:59:50Z | 18 Nov 01:56:28Z | 1.90 | **yes**, from the beginning |
| 3 | `C46-one` | S | 49,152 | 00:00:18Z | 02:13:41Z | 2.18 | **yes**, from the beginning |

Times are from `exp45/e50/logs/queue.log`. **The two runs of stage 3 were restarted.** The host rebooted at about 23:59Z (uptime 0 at 23:59:35Z; the disk was kept) while `C46-one` B was at 9,728 updates and `C46-one` S at 2,560; both were started again from the beginning, and their partial receipts, starting files and logs are kept in `exp45/e50/interrupted_20261117T2359Z/`. A first restart of S at 23:59:50Z was refused by the runner because the partial receipt was still in place; it was moved and S started at 00:00:18Z. On every scored row that the interrupted and the restarted run both have, the two give the same scores, training losses and data digests, from the same starting tensors: under backpropagation the 39 scored rows from update 0 to 9,728, under the sweep the 11 rows from 0 to 2,560 (scores and losses compared from the two logs; data digests and losses from the receipts, which the interrupted runs had last written at 9,216 and 2,048 updates). The other four runs were neither restarted nor paused: the largest wall-clock gap between two scored rows of stage 2, checked at 22:11Z after the reviewer warned that the host might stop idle sessions, was 41 seconds.

**Completion.** Each of the six runs has the log line `e50 complete` and a receipt with `"complete": true`, its final `tensors_sha256` and `weights_sha256`. The `rc=0` that `queue.log` gives at each end is not the run's exit code: the launch script wrote the status of a command substitution (shown with `bash -c "false; echo \"\$(date) rc=\$?\""`, which prints `rc=0`), as its refused start of 23:59:50Z shows. The script was corrected at b1a516ca, after all six runs had been launched. The stage reports' words "exit code 0" rested on those lines; both reports were amended on 18 November to say "complete" by the two criteria above.

**Cost.** The six runs: 9.72 CPU-hours, scoring included. The interrupted pieces of stage 3: 0.44 (0.34 and 0.10, from their logs). The checks: 0.11. In all **10.27 CPU-hours, 0.27 over the brief's limit of 10**; without the interrupted pieces the job is within it (9.83). The reviewer was told when the restart was made.

## Checks before the runs (`exp45/e50_qual/E50_STAGE0_73129.md`)

All pass. Check 1: the specification's self-checks print the coordinator's output exactly. Check 2: at every scored row of all six runs (97 or 193 rows) the code table has the SHA-256 of the start and the byte table and `l1` are bit-equal to their start, and each receipt records `byte_table_and_l1_bit_equal_to_start: true`. Check 3: at every scored row every weight outside a mask is zero and the optimizer's averages there are zero, and E42's limit holds; the batch digest at row 4,096 is `bbd71ab7ca27b72e`, E26's, in all six runs; E49's scorer check on E26's six births gives every count exactly. Check 4: for `C16-in` under backpropagation the specification's forward and a forward written cell by cell differ by at most 2.4e-07 in a logit (tolerance 1e-5). Check 5: six dry runs of 256 updates with their weights, the code table, the owners, the masks and their answers. The runner gives every figure of the audit for seed 73129: the code's digest `12111044f0f8bc70`, the value 3.2387, the digest `6788d5475af8c847` of `C46-one`'s draw, the first-batch losses 5.597, 5.569 and 5.591, both counts of multiply-adds and the groups of cells (95, 131, 30).

## Results

**Under backpropagation.**

| Run | Updates to 80 on A | Updates to 80 on R | Level of A | Level of R | Level at 40 letters | max, min: updates to 80 | add, sub: updates to 80 | R: last three minus three ending 2,048 earlier |
|---|---|---|---|---|---|---|---|---|
| `C-dense` | 7,680 | 6,912 | 97.7 | 97.1 | 92.7 | 1,024; 1,280 | 7,680; 7,680 | +1.1 |
| `C16-in`, first 24,576 | not reached | not reached | 67.9 | 62.1 | 41.6 | 5,376; 5,632 | not reached; not reached | +2.7 |
| `C16-in`, end | 40,704 | not reached | 83.8 | 79.4 | 56.2 | 5,376; 5,632 | 34,560; 40,704 | +2.6 |
| `C46-one`, first 24,576 | 14,080 | 15,360 | 88.5 | 88.9 | 38.2 | 3,584; 3,584 | 14,592; 13,824 | +4.0 |
| `C46-one`, end | 14,080 | 15,360 | 93.8 | 94.25 | 54.3 | 3,584; 3,584 | 14,592; 13,824 | +0.0 |
| E49's `in16` (learned code) | 16,640 | 20,480 | 85.2 | 82.5 | 66.1 | 5,120; 5,376 | 15,616; 18,432 | +0.7 |
| E42's `F10` (learned code) | 8,704 | 9,472 | 93.2 | 94.1 | 92.6 | 2,048; 2,048 | 8,704; 9,472 | +2.1 |

**Under the sweep.**

| Run | Updates to 80 on A | Updates to 80 on R | Level of A | Level of R | Level at 40 letters | max, min: updates to 80 | add, sub: updates to 80 | R: last three minus three ending 2,048 earlier |
|---|---|---|---|---|---|---|---|---|
| `C-dense` | 13,056 | 12,800 | 93.5 | 94.1 | 90.6 | 1,280; 1,536 | 12,032; 13,824 | +3.0 |
| `C16-in`, first 24,576 | not reached | not reached | 32.7 | 29.0 | 9.6 | 7,424; 7,680 | not reached; not reached | +1.7 |
| `C16-in`, end | not reached | not reached | 49.5 | 43.6 | 13.6 | 7,424; 7,680 | not reached; not reached | +1.3 |
| `C46-one`, first 24,576 | not reached | not reached | 63.6 | 65.5 | 23.0 | 5,120; 5,120 | not reached; not reached | +5.6 |
| `C46-one`, end | 35,584 | 34,304 | 85.9 | 87.2 | 27.8 | 5,120; 5,120 | 33,280; 35,584 | +0.0 |
| E49's `in16` (learned code) | not reached | not reached | 1.9 | 1.4 | 1.2 | not reached; not reached | not reached; not reached | -0.4 |
| E42's `F10` (learned code) | 13,312 | 15,104 | 91.2 | 89.3 | 89.1 | 3,072; 2,816 | 11,776; 15,104 | +0.7 |

Read within its first 24,576 updates, `C46-one` under the sweep does not reach 80 on R and is *still rising* (+5.6 points); `C16-in` under either rule is not (+2.7 and +1.7).

## Descriptions against the references (the brief's forms; three-row version)

**`C-dense` against E42's `F10`, same rule.** Updates to 80 on A and on R are *about the same* under both rules (ratios 0.88 and 0.73 under backpropagation, 0.98 and 0.85 under the sweep), multiply-adds *stays*, and the three levels are *about the same*. Changing the learned code for this fixed one costs nothing that the forms can see on this seed. Beside E42's three seeds of `F10` (updates to 80 on A 6,912 to 9,216 and on R 7,936 to 9,728 under backpropagation; 12,544 to 13,312 and 14,336 to 15,104 under the sweep), `C-dense` lies inside their range on A and is earlier than every seed on R; with one seed of `C-dense` this is not read as a gain. By the second count of multiply-adds (`F10`'s equals its first, since none of its code values is exactly zero) `C-dense` needs 0.22 (B) and 0.26 (S) of `F10`'s products to 80 on R.

**`C16-in` against `C-dense`, same rule.** Under both rules: updates to 80 on A and on R *more* (under backpropagation 80 on A at 40,704 updates, 5.30 times, and 80 on R not within 48,640 recordable updates, more than 7.04 times; under the sweep neither). Multiply-adds *rises* by both counts. Levels at the end: under backpropagation A *about the same* (83.8 against 97.7), R and the length test at 40 letters *lower*; under the sweep all three *lower*. Within the first 24,576 updates all three levels are *lower* under both rules.

**Beside it, E49's `in16` against `F10`.** Under backpropagation `in16` needed 2.16 times `F10`'s updates to 80 on R and 1.91 times to 80 on A; under the sweep it learned nothing (R 1.4). Under backpropagation the cost of `C16-in` against `C-dense` is larger than that of `in16` against `F10`, so by the brief it is not described as the cost of the mask. Under the sweep the order is reversed: `C16-in` reaches R 43.6 and max and min 96, where `in16` ended at 1.4 on R and at 39 and 41 on max and min.

**`C46-one` against `C-dense`, same rule.** Under backpropagation: 80 on R at 15,360 updates and on A at 14,080, *more* (ratios 2.22 and 1.83); multiply-adds *stays* by the first count and *rises* by the second; levels at the end: A and R *about the same*, the length test at 40 letters *lower* (54.3 against 92.7). Under the sweep: 80 on R at 34,304 and on A at 35,584, *more* (2.68 and 2.73); multiply-adds *rises*; A and R *about the same* at the end (85.9 and 87.2), the length test *lower* (27.8 against 90.6). Within the first 24,576 updates `C46-one` reaches 80 on R under backpropagation and not under the sweep.

**`C46-one` against `C16-in`, same rule and dose.** Under backpropagation: updates to 80 on A *fewer* (0.35) and on R *fewer* (15,360 against not reached); levels *about the same* (A 93.8 against 83.8, R 94.25 against 79.4, a difference of 14.8; length at 40 letters 54.3 against 56.2). Under the sweep: updates *not decided* (`C16-in` reaches neither threshold; `C46-one` reaches them in the second half of the dose); levels of A and R *higher* (85.9 and 87.2 against 49.5 and 43.6), the length test *about the same*.

## The sentence per rule (the brief's form)

**Backpropagation.** With a code in which a value stands for one byte and each cell reading 16 values of it and every readout unit reading every cell, so that a cell has input for 1 to 10 (median 6) of the 20 bytes it must tell apart, the network **does not learn the first family within the dose** (49,152 updates; R ends at 79.4, with one single row at 80.75), and needs more updates and ends lower on R and the length test than with every cell reading every value. With every cell reading one value of every used byte instead (`C46-one`), it **learns the first family** (80 on R at 15,360 updates). By the brief's rule for this case, **the number of bytes a cell has input for is what costs**. The cost is larger than that of the same mask on the learned code (E49's `in16`), and is not described as the cost of the mask. What the figures support beside that licensed form: `C46-one` itself needs 2.22 times the updates of `C-dense` to 80 on R and is far lower on the length test at 40 letters (54.3 against 92.7); `C46-one` and `C16-in` differ not only in the number of bytes a cell has input for but also in the number of values a cell reads (46 against 16) and in the starting scale of the input weights (√(512/46) = 3.34 against √(512/16) = 5.66); and they differ in speed near the threshold (levels of R 94.25 against 79.42).

**The sweep.** With the same code and the same 16 values a cell, the network **does not learn the first family within the dose** (R ends at 43.6), and needs more updates and ends lower than with every cell reading every value. With one value of every used byte (`C46-one`) it **learns the first family** (80 on R at 34,304 updates, beyond the first 24,576). By the brief's rule, **the number of bytes a cell has input for is what costs**. Beside it, as under backpropagation: `C46-one` itself needs 2.68 times the updates of `C-dense` to 80 on R and is far lower on the length test at 40 letters (27.8 against 90.6), and it differs from `C16-in` also in the number of values a cell reads and in the starting scale of the input weights.

One seed: these are descriptions.

## The explanation under test (the brief, after the predictions)

The explanation: the sweep did not learn with a limited input in E49 because it does not keep a learned code at the size that a cell reading few values requires. Here the code cannot shrink. **`C16-in` under the sweep does not reach 80 on R within its dose while `C-dense` under the sweep does, so by the brief's criterion the explanation is refuted as a sufficient one.** Beside that reading: with the fixed code the sweep gets much further with 16 values a cell than it did with the learned code (R 43.6 against 1.4; max and min learned at 7,424 and 7,680 updates against never), and with one value of every byte it learns. The job does not say what part of that difference is the code's size, since the fixed code differs from the learned one in other respects.

## The descriptions the brief adds

**The four operations apart.** In every run of this job max and min reach 80 at 12 letters: `C-dense` at 1,024 to 1,536 updates, `C16-in` at 5,376 to 7,680, `C46-one` at 3,584 to 5,120. The answer to max or min is one of the operands, so in every run both operands reach the answer; where R does not reach 80 (`C16-in`), what is not learned within the dose is the sums with their reduction. Add and sub reach 80 under backpropagation in `C16-in` at 34,560 and 40,704 updates; under the sweep they end at 49.4 and 49.6.

**The answer's two bytes apart** (R's items, levels at the end): the first byte is right in 98.1, 84.2 and 95.6% of items under backpropagation and in 96.0, 71.5 and 90.2% under the sweep (`C-dense`, `C16-in`, `C46-one`); the second in 97.6, 92.6 and 98.2%, and 94.3, 54.0 and 95.5%. Under backpropagation the second byte ends more often right than the first in both limited settings; under the sweep, in `C16-in`, the first. No failure on the first byte in particular was expected (the brief, revision 2); the bytes are a description.

**By what a cell knows** (mean magnitude of the kept recurrent weights from neighbours into the cells that, at 16 values a cell, have input for 0 to 2, 3 or 4, and 5 or more of the ten digits; 95, 131 and 30 cells). At the start 0.150, 0.151, 0.150. At the end: `C-dense` 0.147, 0.144, 0.145 (B) and 0.150, 0.152, 0.149 (S); `C16-in` 0.265, 0.240, 0.233 (B) and 0.226, 0.219, 0.213 (S); `C46-one` 0.195, 0.206, 0.192 (B) and 0.211, 0.211, 0.208 (S). The grouping means something only in `C16-in`, and there the cells with input for the fewest digits end with the largest weights from their neighbours, by 0.03 (B) and 0.01 (S) over those with the most. In both limited settings the recurrent weights grow, as they do not in `C-dense`. The figure describes the size of weights and is not read as use; the test of use is the reviewer's, from the committed weights.

**Other measures** (the table's last sections). A cell's input for a needed byte (root mean square, without the bias) ends at 5.14 and 4.77 in `C-dense`, 0.77 and 0.63 in `C16-in`, and 0.85 and 0.81 in `C46-one` (B and S), from 0.38 to 0.39 at the start. The cosine between the sweep's and the exact gradient of the input map on the first batch falls during training in the limited settings (`C16-in` B from 0.709 to 0.104; `C46-one` B from 0.755 to 0.555) and less in `C-dense` (0.757 to 0.650).

## Predictions

| | Prediction | Chance | Outcome |
|---|---|---|---|
| P50.1 | Checks 1 to 4 pass | 85% | true |
| P50.2 | `C-dense`, backpropagation: reaches 80 on R within the dose | 80% | true (6,912 updates) |
| P50.3 | `C-dense`, sweep: reaches 80 on R within the dose | 55% | true (12,800) |
| P50.8 | `C-dense` against E42's `F10`, backpropagation: updates to 80 on R *about the same* or *fewer* | 60% | true (*about the same*, ratio 0.73) |
| P50.4' | `C16-in` reaches 80 on R within 49,152 updates under backpropagation | 55% | false in the brief's three-row version (single row at 80.75, level 79.4) |
| P50.5' | `C16-in` reaches 80 on R within 49,152 updates under the sweep | 25% | false (level 43.6) |
| P50.6' | Under backpropagation `C16-in` reaches 80 on R in no more than 3.24 times the updates of `C-dense` | 45% | false |
| P50.7' | In `C16-in`, max and min at 12 letters reach 80 under backpropagation | 90% | true (5,376 and 5,632) |
| P50.8' | In `C16-in`, max and min at 12 letters reach 80 under the sweep | 45% | true (7,424 and 7,680) |

P50.4 to P50.7 of revision 1 were withdrawn with the settings they named.

## What this job cannot show

- **More than one seed.** Every reading is of seed 73129.
- **Why the number of bytes costs.** `C46-one` separates the number of bytes a cell has input for from the number of weights behind an input; it does not say how cells that lack a byte fail to use what their neighbours carry. `C46-one` itself costs against `C-dense` (2.22 and 2.68 times the updates to 80 on R, and the length test at 40 letters *lower*), so an input resting on one weight a byte also costs, though not enough to stop learning within the dose.
- **A longer dose.** `C16-in` under backpropagation ends within a point of 80 on R, rising by 2.6 over the last 2,048 updates (below *still rising*); "does not within the dose" says nothing of a longer one.
- **A cell that stays ignorant.** One position after a byte, at least 241 of the 256 cells are within a connection of a reader of it.
- **A limited readout, a learned sparse code, an input of many tokens, more cells, other settings**, and **the same difference between the rules as in E49** (the code has no gradient here), as revision 2 lists.
- **The length test at 40 letters** falls in both limited settings far more than A and R do (`C46-one` ends at 54.3 and 27.8 against `C-dense`'s 92.7 and 90.6). This is recorded and not explained.

## Rules, and what departed from them

- Only this branch was pushed; no pull request was opened; no existing file was edited; no path was overwritten (interrupted files were moved, not replaced). The brief's revision 2 is a new version of the brief at its own path on this branch only.
- **One commit carries a trailer that the rules for the worker did not allow:** the message of b1a516ca has a `Co-Authored-By` line that names a language model. A rewording needed a force push, which the session's permissions refused; on the reviewer's instruction the commit stays as it is and history is not rewritten. No other commit message and no committed file names one, and no such trailer was added again.
- The cost went 0.27 CPU-hours over the limit because of the reboot (above).
- Where the brief and the audit could both be followed, both were; no conflict between them arose.

## Amendments, 18 November 2026

After the reviewer's verification of stage 3 and of this report (`PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/E50_STAGE3_AND_REPORT_VERIFICATION_20261118.md` on `claude/vibrant-cerf-sje726`, which reproduces all 140 figures of the two result tables and, from the weights, 2,400 of the 2,400 recorded answers): (a) the second figure of "by what a cell knows" at the start is 0.151 (0.151484), not 0.152; (b) for `C46-one` under the sweep, the first group at the end is 0.211 (0.211491), not 0.212; (c) the level of R of `C46-one` under backpropagation is given as 94.25 in the table and the text, which had 94.2 and 94.3; (d) beside the licensed sentence that the number of bytes a cell has input for is what costs, the report now says what else the figures show: `C46-one`'s own cost against `C-dense`, its far lower length test at 40 letters, the two other ways in which `C46-one` and `C16-in` differ, and their difference in speed near the threshold. Earlier, at 1799b2ba, the citations of the verifications of stages 1 and 2, the seed spread of `F10`, its second count and the rows compared for the restart were entered in place. No result changed. Stage 3 is now verified, which supersedes the sentence of the opening that says it has not been.

## Committed files of the job

The files that differ from the default branch at 7cabb13a, listed by `git diff --name-only origin/codex/minifly-consolidated-20261023 015b388d` and each found in `git ls-tree -r --name-only 015b388d` (102 files). This report and its commit come after them.

- `MOD97_LOCAL_RULE_20261106/e49_interface.py`
- `MOD97_LOCAL_RULE_20261106/e50_sparse_code.py`
- `MOD97_LOCAL_RULE_20261106/e50_tables.py`
- `MOD97_LOCAL_RULE_20261106/exp45/e50/E50_STAGE1_REPORT_73129.md`
- `MOD97_LOCAL_RULE_20261106/exp45/e50/E50_STAGE2_REPORT_73129.md`
- `MOD97_LOCAL_RULE_20261106/exp45/e50/E50_SUMMARY_73129.json`
- `MOD97_LOCAL_RULE_20261106/exp45/e50/E50_SUMMARY_73129_stage1.json`
- `MOD97_LOCAL_RULE_20261106/exp45/e50/E50_SUMMARY_73129_stage2.json`
- `MOD97_LOCAL_RULE_20261106/exp45/e50/E50_TABLES_73129.md`
- `MOD97_LOCAL_RULE_20261106/exp45/e50/E50_TABLES_73129_stage1.md`
- `MOD97_LOCAL_RULE_20261106/exp45/e50/E50_TABLES_73129_stage2.md`
- `MOD97_LOCAL_RULE_20261106/exp45/e50/e50_C-dense_B_73129.json`
- `MOD97_LOCAL_RULE_20261106/exp45/e50/e50_C-dense_B_73129.pt`
- `MOD97_LOCAL_RULE_20261106/exp45/e50/e50_C-dense_B_73129_answers.json`
- `MOD97_LOCAL_RULE_20261106/exp45/e50/e50_C-dense_B_73129_start.pt`
- `MOD97_LOCAL_RULE_20261106/exp45/e50/e50_C-dense_S_73129.json`
- `MOD97_LOCAL_RULE_20261106/exp45/e50/e50_C-dense_S_73129.pt`
- `MOD97_LOCAL_RULE_20261106/exp45/e50/e50_C-dense_S_73129_answers.json`
- `MOD97_LOCAL_RULE_20261106/exp45/e50/e50_C-dense_S_73129_start.pt`
- `MOD97_LOCAL_RULE_20261106/exp45/e50/e50_C16-in_B_73129.json`
- `MOD97_LOCAL_RULE_20261106/exp45/e50/e50_C16-in_B_73129.pt`
- `MOD97_LOCAL_RULE_20261106/exp45/e50/e50_C16-in_B_73129_answers.json`
- `MOD97_LOCAL_RULE_20261106/exp45/e50/e50_C16-in_B_73129_at24576.pt`
- `MOD97_LOCAL_RULE_20261106/exp45/e50/e50_C16-in_B_73129_at24576_answers.json`
- `MOD97_LOCAL_RULE_20261106/exp45/e50/e50_C16-in_B_73129_start.pt`
- `MOD97_LOCAL_RULE_20261106/exp45/e50/e50_C16-in_S_73129.json`
- `MOD97_LOCAL_RULE_20261106/exp45/e50/e50_C16-in_S_73129.pt`
- `MOD97_LOCAL_RULE_20261106/exp45/e50/e50_C16-in_S_73129_answers.json`
- `MOD97_LOCAL_RULE_20261106/exp45/e50/e50_C16-in_S_73129_at24576.pt`
- `MOD97_LOCAL_RULE_20261106/exp45/e50/e50_C16-in_S_73129_at24576_answers.json`
- `MOD97_LOCAL_RULE_20261106/exp45/e50/e50_C16-in_S_73129_start.pt`
- `MOD97_LOCAL_RULE_20261106/exp45/e50/e50_C46-one_B_73129.json`
- `MOD97_LOCAL_RULE_20261106/exp45/e50/e50_C46-one_B_73129.pt`
- `MOD97_LOCAL_RULE_20261106/exp45/e50/e50_C46-one_B_73129_answers.json`
- `MOD97_LOCAL_RULE_20261106/exp45/e50/e50_C46-one_B_73129_at24576.pt`
- `MOD97_LOCAL_RULE_20261106/exp45/e50/e50_C46-one_B_73129_at24576_answers.json`
- `MOD97_LOCAL_RULE_20261106/exp45/e50/e50_C46-one_B_73129_start.pt`
- `MOD97_LOCAL_RULE_20261106/exp45/e50/e50_C46-one_S_73129.json`
- `MOD97_LOCAL_RULE_20261106/exp45/e50/e50_C46-one_S_73129.pt`
- `MOD97_LOCAL_RULE_20261106/exp45/e50/e50_C46-one_S_73129_answers.json`
- `MOD97_LOCAL_RULE_20261106/exp45/e50/e50_C46-one_S_73129_at24576.pt`
- `MOD97_LOCAL_RULE_20261106/exp45/e50/e50_C46-one_S_73129_at24576_answers.json`
- `MOD97_LOCAL_RULE_20261106/exp45/e50/e50_C46-one_S_73129_start.pt`
- `MOD97_LOCAL_RULE_20261106/exp45/e50/interrupted_20261117T2359Z/e50_C46-one_B_73129.json`
- `MOD97_LOCAL_RULE_20261106/exp45/e50/interrupted_20261117T2359Z/e50_C46-one_B_73129.log`
- `MOD97_LOCAL_RULE_20261106/exp45/e50/interrupted_20261117T2359Z/e50_C46-one_B_73129_start.pt`
- `MOD97_LOCAL_RULE_20261106/exp45/e50/interrupted_20261117T2359Z/e50_C46-one_S_73129.json`
- `MOD97_LOCAL_RULE_20261106/exp45/e50/interrupted_20261117T2359Z/e50_C46-one_S_73129.log`
- `MOD97_LOCAL_RULE_20261106/exp45/e50/interrupted_20261117T2359Z/e50_C46-one_S_73129_refused_start_0000Z.log`
- `MOD97_LOCAL_RULE_20261106/exp45/e50/interrupted_20261117T2359Z/e50_C46-one_S_73129_start.pt`
- `MOD97_LOCAL_RULE_20261106/exp45/e50/launch_e50.sh`
- `MOD97_LOCAL_RULE_20261106/exp45/e50/logs/e50_C-dense_B_73129.log`
- `MOD97_LOCAL_RULE_20261106/exp45/e50/logs/e50_C-dense_S_73129.log`
- `MOD97_LOCAL_RULE_20261106/exp45/e50/logs/e50_C16-in_B_73129.log`
- `MOD97_LOCAL_RULE_20261106/exp45/e50/logs/e50_C16-in_S_73129.log`
- `MOD97_LOCAL_RULE_20261106/exp45/e50/logs/e50_C46-one_B_73129.log`
- `MOD97_LOCAL_RULE_20261106/exp45/e50/logs/e50_C46-one_S_73129.log`
- `MOD97_LOCAL_RULE_20261106/exp45/e50/logs/queue.log`
- `MOD97_LOCAL_RULE_20261106/exp45/e50/reference/SOURCE.txt`
- `MOD97_LOCAL_RULE_20261106/exp45/e50/reference/e42_F10_B_73129.json`
- `MOD97_LOCAL_RULE_20261106/exp45/e50/reference/e42_F10_S_73129.json`
- `MOD97_LOCAL_RULE_20261106/exp45/e50/reference/e49_in16_B_73129.json`
- `MOD97_LOCAL_RULE_20261106/exp45/e50/reference/e49_in16_S_73129.json`
- `MOD97_LOCAL_RULE_20261106/exp45/e50_qual/COPIES.txt`
- `MOD97_LOCAL_RULE_20261106/exp45/e50_qual/E50_STAGE0_73129.md`
- `MOD97_LOCAL_RULE_20261106/exp45/e50_qual/check1_73129.json`
- `MOD97_LOCAL_RULE_20261106/exp45/e50_qual/check3_73129.json`
- `MOD97_LOCAL_RULE_20261106/exp45/e50_qual/check4_73129.json`
- `MOD97_LOCAL_RULE_20261106/exp45/e50_qual/check5_73129.json`
- `MOD97_LOCAL_RULE_20261106/exp45/e50_qual/check5_73129_runs/e50_C-dense_B_dry256_73129.json`
- `MOD97_LOCAL_RULE_20261106/exp45/e50_qual/check5_73129_runs/e50_C-dense_B_dry256_73129.pt`
- `MOD97_LOCAL_RULE_20261106/exp45/e50_qual/check5_73129_runs/e50_C-dense_B_dry256_73129_answers.json`
- `MOD97_LOCAL_RULE_20261106/exp45/e50_qual/check5_73129_runs/e50_C-dense_B_dry256_73129_start.pt`
- `MOD97_LOCAL_RULE_20261106/exp45/e50_qual/check5_73129_runs/e50_C-dense_S_dry256_73129.json`
- `MOD97_LOCAL_RULE_20261106/exp45/e50_qual/check5_73129_runs/e50_C-dense_S_dry256_73129.pt`
- `MOD97_LOCAL_RULE_20261106/exp45/e50_qual/check5_73129_runs/e50_C-dense_S_dry256_73129_answers.json`
- `MOD97_LOCAL_RULE_20261106/exp45/e50_qual/check5_73129_runs/e50_C-dense_S_dry256_73129_start.pt`
- `MOD97_LOCAL_RULE_20261106/exp45/e50_qual/check5_73129_runs/e50_C16-in_B_dry256_73129.json`
- `MOD97_LOCAL_RULE_20261106/exp45/e50_qual/check5_73129_runs/e50_C16-in_B_dry256_73129.pt`
- `MOD97_LOCAL_RULE_20261106/exp45/e50_qual/check5_73129_runs/e50_C16-in_B_dry256_73129_answers.json`
- `MOD97_LOCAL_RULE_20261106/exp45/e50_qual/check5_73129_runs/e50_C16-in_B_dry256_73129_start.pt`
- `MOD97_LOCAL_RULE_20261106/exp45/e50_qual/check5_73129_runs/e50_C16-in_S_dry256_73129.json`
- `MOD97_LOCAL_RULE_20261106/exp45/e50_qual/check5_73129_runs/e50_C16-in_S_dry256_73129.pt`
- `MOD97_LOCAL_RULE_20261106/exp45/e50_qual/check5_73129_runs/e50_C16-in_S_dry256_73129_answers.json`
- `MOD97_LOCAL_RULE_20261106/exp45/e50_qual/check5_73129_runs/e50_C16-in_S_dry256_73129_start.pt`
- `MOD97_LOCAL_RULE_20261106/exp45/e50_qual/check5_73129_runs/e50_C46-one_B_dry256_73129.json`
- `MOD97_LOCAL_RULE_20261106/exp45/e50_qual/check5_73129_runs/e50_C46-one_B_dry256_73129.pt`
- `MOD97_LOCAL_RULE_20261106/exp45/e50_qual/check5_73129_runs/e50_C46-one_B_dry256_73129_answers.json`
- `MOD97_LOCAL_RULE_20261106/exp45/e50_qual/check5_73129_runs/e50_C46-one_B_dry256_73129_start.pt`
- `MOD97_LOCAL_RULE_20261106/exp45/e50_qual/check5_73129_runs/e50_C46-one_S_dry256_73129.json`
- `MOD97_LOCAL_RULE_20261106/exp45/e50_qual/check5_73129_runs/e50_C46-one_S_dry256_73129.pt`
- `MOD97_LOCAL_RULE_20261106/exp45/e50_qual/check5_73129_runs/e50_C46-one_S_dry256_73129_answers.json`
- `MOD97_LOCAL_RULE_20261106/exp45/e50_qual/check5_73129_runs/e50_C46-one_S_dry256_73129_start.pt`
- `MOD97_LOCAL_RULE_20261106/exp45/e50_qual/checkfigures_73129.json`
- `MOD97_LOCAL_RULE_20261106/exp45/e50_qual/logs/check1.log`
- `MOD97_LOCAL_RULE_20261106/exp45/e50_qual/logs/check3.log`
- `MOD97_LOCAL_RULE_20261106/exp45/e50_qual/logs/check4.log`
- `MOD97_LOCAL_RULE_20261106/exp45/e50_qual/logs/check5.log`
- `MOD97_LOCAL_RULE_20261106/exp45/e50_qual/logs/check_figures.log`
- `MOD97_LOCAL_RULE_20261106/exp49_interface.py`
- `MOD97_LOCAL_RULE_20261106/exp50_code.py`
- `OPUS_TASK_E50_AN_INPUT_CODE_IN_WHICH_A_VALUE_STANDS_FOR_ONE_BYTE_20261116.md`
