# E53: the network with its input and output at a boundary, from a start at which differences between items reach the cells away from the boundary, with and without weight decay

**Status.** This is a first measurement at the evidence level that the coordinator's briefs call E2. It confirms nothing on its own, and no independent audit of it exists.

**Brief and assignment.** The work follows the brief `TASK_E53_THE_NETWORK_WITH_AN_INTERFACE_AT_A_BOUNDARY_WITHOUT_DECAY_AND_FROM_A_LARGER_START_20261117.md`, revision 4. The brief is on `origin/claude/vibrant-cerf-sje726`, unchanged since 2b85dd8b, under `PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/COORDINATION_20261117/`. The job was assigned to this line by the coordinator's decision 17 of 18 November. That decision is in `COORDINATOR_DECISIONS_20261118.md` on the default branch at 39eacc33, and the outside reviewer forwarded it.

**Branch.** The work is on branch `claude/e58-gated-term-kqasnme`, which was created from the default branch at 7cabb13a. No file of E58 was changed.

**Sources of the figures.** All paths are relative to `MOD97_LOCAL_RULE_20261106/`. Every figure in this report comes from one of the following:
- the run receipts `exp45/e53/e53_32_<arm>_<rule>_73129.json`. The command `python3 exp45/e53/e53_tables.py` prints their tables, and its output is saved in `exp45/e53/E53_TABLES.md`;
- the check receipts `exp45/e53_qual/check<n>.json`;
- the trial's receipts and weight files, read from `origin/claude/e47-worker-stage-0-3d0ya6` at 578a877b with `git show` into a folder that is not committed.

**What this report leaves to the reviewer.** The reading of rings 2 and 3, and therefore the statement on the explanation under test, is made by the reviewer from the committed weights, as the brief prescribes (section 6).

## 1. Terms

- **The network.** E47's network at 32 cells a side in its arm `one`: 1,024 cells on a torus.
  - Each cell reads its own previous state and ten cells within three rows and three columns of it.
  - The 256 *interface cells* of one 16 by 16 patch receive the code of the current byte and are read by the readout.
  - The 768 *far cells* receive no input and are not read.
- **Ring.** A far cell's *ring* is its grid distance to the patch, divided by 3 and rounded up. Ring 1 has 228 cells, ring 2 has 300 and ring 3 has 240.
- **Recurrent weights.** `Wc` holds 33 weights for each cell: three gates times eleven weights.
- **Size of the weights.** The *size of the weights* of a group of cells is the mean absolute value of the recurrent weights into its cells.
- **The two rules.** *Back-propagation* (rule B) uses the exact gradient. *The sweep* (rule S) passes the error backward across each connection by the sign of its weight times one scale, the mean size of all recurrent weights.
- **The first family.** Addition, subtraction, the larger and the smaller of two numbers below 89, with 0 to 12 letters between the operands. Half of the pairs of operands are taught.
- **Rows and scores.** The network is scored every 256 updates; the scores of one such point are a *row*.
  - Score **A** is the percentage of right answers on 600 untaught pairs of addition and subtraction with 12 letters.
  - Score **R** is the percentage on 2,000 untaught pairs whose answer needs the reduction modulo 89, without letters.
  - The *length test at 40 letters* is the percentage on 300 untaught additions written with 40 letters.
- **Reaching 80.** A score *reaches 80* at the first of three consecutive rows at or above 80. *Updates to 80* is the number of updates at that row.
- **Level.** The *level* of a score is its mean over the last three rows.
- **Read.** A run *is read* if its levels of A and R are both at or above 80.
- **The two diagnostics.** The network is scored with the far cells held, after every position, at zero, or at their mean over the scored items at that position.
- **The trial.** E47's runs of the arm `one` on seed 73129, with weight decay 0.25 and 24,576 updates.

## 2. Arms and runs

The seed is 73129, and every run makes 24,576 updates.

| Arm | Start | Weight decay of the optimizer's first group | Runs |
|---|---|---|---|
| `one` (the trial, on file) | E47's | 0.25 | 0 |
| `x3` | E47's with every entry of `Wc` multiplied by 3 | 0.25 | S, B |
| `wd0` | E47's | 0 | S, B |
| `x3-wd0` | E47's with every entry of `Wc` multiplied by 3 | 0 | S, B |
| `xf3`, `xf3-wd0` (conditional) | × 3 except the interface cells' weights among themselves | 0.25 or 0 | not made (section 5.4) |

**What the decay covers.** The first group of the optimizer holds thirteen tensors: the weights and biases of `l1`, of the input map, of `lm`, of `l2` and of `l3`, and `Wc`, `bc` and `bfar`. An arm without decay therefore removes the decay from all of these, not from the recurrent weights alone. The byte table has no decay in any arm.

**Restarts.** The host restarted three times while runs were in progress. Every run was started with `--resume`, so each one continued from its last saved state:
- `wd0` S, `wd0` B and `x3-wd0` S were resumed at updates 16,384 and 21,504 (the entry `resumes` of their receipts).
- The rows those runs had logged before each restart and logged again after it are identical (`exp45/e53/logs/`).
- No run was interrupted before its first saved state, so `exp45/e53/interrupted/` is empty.

**Cost.** The six runs took 25.5 CPU-hours: 3.96 to 4.55 each, from `cpu_updates` and `cpu_scoring` in the receipts. The checks took 1.0 CPU-hour. The work lost between a saved state and a restart is not in these sums.

**Software and host.** Python 3.13.18 and torch 2.15.0+cpu, as in the trial; numpy 2.5.5. At most three processes ran at a time, with one thread each.

## 3. The runner

**Files.** The runner is `exp53_local_start.py`, a new file.
- `exp47_local.py` and `e47_local.py` were copied byte for byte from 578a877b, with the SHA-256 values that the brief gives; they are recorded in `E53_COPIED_SOURCES.txt`.
- Every other module that the runner imports has the SHA-256 recorded in the trial's receipts.
- The runner uses `build`, `install`, `make_opt`, `forward`, `step`, `score_row_all`, `magnitudes`, `cosines`, `weights`, `save_weights`, `save_ckpt` and `tsha` from `exp47_local` unchanged.

**The diff.** The difference between the text of `exp47_local.run` and that of the new `run` is in `exp45/e53/E53_RUN_DIFF.txt` (74 lines, from `diff`). It consists of the brief's changes 1 to 8 and two edits that follow from them:
1. **Change 1.** The network is built as `build(32, "one", seed)`, and the starting and structure hashes are asserted.
2. **Change 2.** `multiply` is called before `make_opt`.
3. **Change 3.** `set_decay` and `assert_decay` are called, also after a resume. `opt_record` takes the arm's decay.
4. **Change 4.** The receipt name begins with `e53`, the printed lines begin with `e53`, and the environment variable is `E53_BESIDE`.
5. **Change 5.** The receipt has the fields `experiment="E53"`, `network_arm`, `multiple`, `entries_multiplied`, `entries_multiplied_count`, `weight_decay` and the new `start` text. The provenance adds the runner's hash.
6. **Change 6.** Every row records `ring_magnitudes` and `gates`.
7. **Change 7.** The weights of updates 24,064 and 24,320 are saved.
8. **Change 8.** The call interface (`main`) is new.

The two edits that follow from these changes are:
- The function's signature drops the argument `side`, which is fixed at 32 by change 1.
- One line of the docstring that named E47's check 8 is removed.

I had first added a stop on a non-finite loss and the loss to the printed line. I removed both before any run, because they lay outside changes 1 to 8. A non-finite loss would have been seen in the receipts' `finite_losses`, and every row of every run has `finite_losses` true.

## 4. Checks

1. **With multiple 1 and decay 0.25, the new `run` is the trial's.** Pass (`check1.json`).
   - The arm `one` was run with `--dose 1024` under each rule.
   - At updates 0, 256, 512, 768 and 1,024, these fields equal the trial's exactly: `loss`, `A_counts`, `R_counts`, `taught_counts`, `length_counts`, the counts in `far_off` and `far_mean`, `magnitudes` and `data_digest`.
   - So do `start_tensors_sha256`, `structure_sha256` and `cosine_first_batch_start`.
2. **The starts.** Pass (`check2.json`; four runs of 256 updates under the sweep).
   - `wd0`: every tensor of the starting file is bit-equal to the trial's, including the five tensors of whole numbers.
   - `x3` and `x3-wd0`: every tensor except `Wc` is bit-equal, and `Wc` is bit-equal to the trial's `Wc` × 3 in single precision.
   - `xf3`: 26,925 entries are the trial's × 3 and 6,867 equal the trial's. The 6,867 are exactly the entries in which a cell and its source are both interface cells.
3. **The decay.** Pass (`check3.json`).
   - (a) The decays read back from the optimizer at the start and at the end are 0.25/0 in `x3` and `xf3`, and 0/0 in `wd0` and `x3-wd0`.
   - (b) After one update from E47's start, the largest value of |p(decay 0) − p(decay 0.25) − 0.00025 p(start)| over the thirteen tensors is 4.4e-8 under the sweep and 3.9e-8 under back-propagation. Had the decay not been removed, it would be 1.1e-5 to 8.8e-5. `book.weight` is bit-equal in the two runs.
4. **A run that is killed and resumed equals an uninterrupted one.** Pass at the second attempt (`check4.json`).
   - The first attempt failed in my check's code, not in any comparison. The child process that the check starts and kills inherited the environment variables `E11_*` that the parent's import of the harness sets, and the harness's environment gate refused to start it.
   - I changed the child's environment to that of a shell, as E47's own check 1 does. The first attempt's log is kept in `exp45/e53_qual/check4_first_attempt/`.
   - At the second attempt, `x3-wd0` under the sweep with `--dose 2048` was killed (`SIGKILL`) after its log showed the row of update 1,536, and resumed from the saved state of update 1,024.
   - All 9 rows and every field not beginning with `cpu` or `wall` equal those of the uninterrupted run, as do `tensors_sha256`, `start_tensors_sha256`, `data_digest` and `cosine_first_batch_end`. The decays after the resume are 0/0.
5. **The starts with the multiple, on the first batch.** Pass (`check5.json`).
   - For `x3` and `xf3` under each rule, the loss and every gradient are finite.
   - The share of gate values below 0.05 or above 0.95 is at most 0.002% in every group.
   - The mean absolute state of the interface cells and of rings 1, 2 and 3 is 0.214, 0.095, 0.062 and 0.051 at `x3`, and 0.169, 0.083, 0.059 and 0.050 at `xf3`.
6. **Cost** (no criterion; `check6.json`). From the runs of check 2, a run costs 0.46 CPU-seconds an update and 44 a scored row. The forecast was 4.3 CPU-hours a run and 26.0 for the six runs, below the 30-hour limit. The runs took 25.5.

## 5. Results

### 5.1 The runs

The tables are from `E53_TABLES.md`.

| Rule | Arm | Updates to 80 on A | Updates to 80 on R | Level A | Level R | Level, 40 letters | Read |
|---|---|---|---|---|---|---|---|
| S | one (trial) | 13,568 | 17,664 | 92.3 | 87.8 | 90.8 | read |
| S | x3 | 15,360 | 18,432 | 89.4 | 86.8 | 86.7 | read |
| S | wd0 | 11,264 | 13,312 | 89.2 | 86.3 | 89.6 | read |
| S | x3-wd0 | 15,104 | 22,272 | 83.5 | 78.5 | 82.4 | **not read** |
| B | one (trial) | 10,496 | 11,008 | 95.4 | 94.2 | 95.2 | read |
| B | x3 | 9,728 | 10,240 | 93.8 | 93.5 | 92.3 | read |
| B | wd0 | 8,704 | 11,776 | 90.2 | 88.0 | 91.3 | read |
| B | x3-wd0 | 9,728 | 13,568 | 89.6 | 85.5 | 91.1 | read |

`x3-wd0` under the sweep is not read, because its level of R is 78.5. Its curves are in its receipt and log. No statement about far cells is made for it.

### 5.2 Learning, in the words of `describe`

The words come from `describe`, copied from `e47_trial_read.py`, with 24,064 as the last update at which three rows can begin. The first figure of each pair is the candidate's updates to 80, the second the reference's.

| Rule | Candidate | Reference | A | Words for A | R | Words for R |
|---|---|---|---|---|---|---|
| S | x3 | one | 15,360 / 13,568 | about the same | 18,432 / 17,664 | about the same |
| S | wd0 | one | 11,264 / 13,568 | about the same | 13,312 / 17,664 | about the same |
| S | x3-wd0 | one | 15,104 / 13,568 | about the same | 22,272 / 17,664 | about the same |
| S | x3-wd0 | x3 | 15,104 / 15,360 | about the same | 22,272 / 18,432 | about the same |
| S | x3-wd0 | wd0 | 15,104 / 11,264 | about the same | 22,272 / 13,312 | **more** |
| B | x3 | one | 9,728 / 10,496 | about the same | 10,240 / 11,008 | about the same |
| B | wd0 | one | 8,704 / 10,496 | about the same | 11,776 / 11,008 | about the same |
| B | x3-wd0 | one | 9,728 / 10,496 | about the same | 13,568 / 11,008 | about the same |
| B | x3-wd0 | x3 | 9,728 / 9,728 | about the same | 13,568 / 10,240 | about the same |
| B | x3-wd0 | wd0 | 9,728 / 8,704 | about the same | 13,568 / 11,776 | about the same |

Every run that is read learns the family in *about the same* number of updates as the trial, on A and on R. No new run is *fewer*.

### 5.3 The far cells as a whole

These are the two diagnostics at the last row, read in the forms of E47's brief with `use_form`. Each score is given as plain / far cells at zero / far cells at their mean.

| Rule | Arm | A | R | Fall of A (zero, mean) | Fall of R (zero, mean) | Form |
|---|---|---|---|---|---|---|
| S | one (trial) | 93.3 / 77.5 / 84.8 | 86.3 / 77.7 / 80.6 | 15.8, 8.5 | 8.6, 5.8 | partly |
| S | x3 | 90.2 / 75.3 / 82.2 | 85.7 / 69.4 / 81.3 | 14.8, 8.0 | 16.3, 4.4 | partly |
| S | wd0 | 89.7 / 49.5 / 85.0 | 85.7 / 50.0 / 83.3 | 40.2, 4.7 | 35.7, 2.4 | a constant is used |
| S | x3-wd0 (not read) | 85.7 / 43.5 / 76.3 | 78.8 / 47.0 / 72.5 | 42.2, 9.3 | 31.8, 6.3 | (partly) |
| B | one (trial) | 96.2 / 78.8 / 90.2 | 94.7 / 78.9 / 88.7 | 17.3, 6.0 | 15.8, 6.0 | partly |
| B | x3 | 94.7 / 60.2 / 60.0 | 92.8 / 55.9 / 65.6 | 34.5, 34.7 | 36.9, 27.2 | **used** |
| B | wd0 | 91.2 / 52.2 / 76.2 | 88.6 / 43.8 / 75.7 | 39.0, 15.0 | 44.8, 12.9 | **used** |
| B | x3-wd0 | 90.2 / 39.0 / 48.5 | 86.2 / 38.9 / 50.1 | 51.2, 41.7 | 47.3, 36.1 | **used** |

These are the far cells as a whole: rings 1, 2 and 3 together. The trial's verification found that the whole effect in the trial comes from ring 1. Whether rings 2 and 3 take part in these runs is the reviewer's reading (section 6). This table does not answer it.

### 5.4 The condition of the conditional run

The conditional run is made if, under back-propagation, `x3` or `x3-wd0` is not read, or needs more updates to 80 on A than the trial. Neither does:
- both are read;
- both reach 80 on A at update 9,728, against the trial's 10,496, which is *about the same*.

**The conditional run is therefore not made.** The condition concerns back-propagation only; that `x3-wd0` under the sweep is not read does not enter it.

### 5.5 The size of the weights by ring

The table gives the mean absolute recurrent weight into the cells of each group, from `ring_magnitudes` in the receipts. The trial's start and end are computed in the same way from its weight files; the trial did not record update 4,096.

| Rule | Arm | Update | Interface | Ring 1 | Ring 2 | Ring 3 |
|---|---|---|---|---|---|---|
| S | one (trial) | 0 | 0.152 | 0.150 | 0.151 | 0.150 |
| S | one (trial) | 24,576 | 0.198 | 0.178 | 0.132 | 0.145 |
| S | x3 | 0 | 0.455 | 0.451 | 0.452 | 0.449 |
| S | x3 | 4,096 | 0.243 | 0.201 | 0.176 | 0.169 |
| S | x3 | 24,576 | 0.213 | 0.154 | 0.058 | 0.024 |
| S | wd0 | 0 | 0.152 | 0.150 | 0.151 | 0.150 |
| S | wd0 | 4,096 | 0.262 | 0.238 | 0.204 | 0.178 |
| S | wd0 | 24,576 | 0.404 | 0.346 | 0.287 | 0.237 |
| S | x3-wd0 | 0 | 0.455 | 0.451 | 0.452 | 0.449 |
| S | x3-wd0 | 4,096 | 0.514 | 0.474 | 0.462 | 0.455 |
| S | x3-wd0 | 24,576 | 0.574 | 0.519 | 0.483 | 0.468 |
| B | one (trial) | 0 | 0.152 | 0.150 | 0.151 | 0.150 |
| B | one (trial) | 24,576 | 0.179 | 0.234 | 0.257 | 0.234 |
| B | x3 | 0 | 0.455 | 0.451 | 0.452 | 0.449 |
| B | x3 | 4,096 | 0.241 | 0.217 | 0.188 | 0.180 |
| B | x3 | 24,576 | 0.178 | 0.241 | 0.247 | 0.223 |
| B | wd0 | 0 | 0.152 | 0.150 | 0.151 | 0.150 |
| B | wd0 | 4,096 | 0.267 | 0.264 | 0.255 | 0.259 |
| B | wd0 | 24,576 | 0.409 | 0.387 | 0.361 | 0.348 |
| B | x3-wd0 | 0 | 0.455 | 0.451 | 0.452 | 0.449 |
| B | x3-wd0 | 4,096 | 0.509 | 0.475 | 0.459 | 0.452 |
| B | x3-wd0 | 24,576 | 0.580 | 0.519 | 0.485 | 0.467 |

**The multiple does not survive the decay.** In `x3`, the weights of every group fall from about 0.45 to 0.17–0.24 by update 4,096, under both rules. Decay alone would leave 0.36 of a weight after 4,096 updates. After that the two rules diverge:
- Under the sweep, rings 2 and 3 of `x3` shrink further, to 0.058 and 0.024 at the end: about two fifths and a sixth of the trial's 0.150 at its start.
- Under back-propagation, they end at 0.247 and 0.223, close to the trial's end.

So `x3` ends near the trial in its weights under back-propagation, and far below it in the outer rings under the sweep. Without the decay, `x3-wd0` keeps the multiple, and `wd0` grows from 0.150 to 0.24–0.41.

**Gates.** The receipts also record, at every row, the share of extreme gate values and the mean absolute state of each group on the first batch (`gates`). Under back-propagation with the multiple, from update 4,096 to the end, a large share of the interface cells' gates is below 0.05 or above 0.95. In `x3` the share is 57 to 64% of the reset gates and 43 to 50% of the update gates; in `x3-wd0` it is 58 to 64% and 59 to 65%. In rings 1 to 3 the largest share at any of these rows is 42% in `x3` and 15% in `x3-wd0` (computed from `gates` in the two receipts).

## 6. The explanation under test

The explanation under test is this: the cells beyond the first ring take no part because, from E47's start, differences between items do not reach them. The brief makes the statement for each rule from the reading of rings 2 and 3. That reading is made by the reviewer, from the committed weights of updates 24,064, 24,320 and 24,576, with part F of `e47_trial_verify.py`. This report therefore states only what the brief's conditions already fix:

- **Under the sweep.** `x3-wd0` is not read. Even if rings 2 and 3 were not used in `x3`, the explanation could not be *not supported*, because that would need both `x3` and `x3-wd0` to be read.
  - It is *supported* if rings 2 and 3 are *used* in `x3`.
  - Otherwise it is *not decided*.
- **Under back-propagation.** `x3` and `x3-wd0` are both read, so every outcome of the brief remains open and depends on the reviewer's reading. The far cells as a whole are *used* in `x3`, in `x3-wd0` and also in `wd0` (section 5.3).
  - If rings 2 and 3 are *used* in `wd0`, the brief's extra sentence applies: removing the decay is enough at E47's start.
- **What section 5.3 does not establish.** "The far cells as a whole are used" does not, by the brief's rules, decide the use of rings 2 and 3. Ring 1 alone can produce it.

## 7. Predictions

This report scores P53.1 to P53.5, P53.8 to P53.10 and P53.12. P53.6, P53.7 and P53.11 are scored by the reviewer.

| | Statement | Chance given | Outcome |
|---|---|---|---|
| P53.1 | Checks 1 to 5 pass at the first attempt | 75% | **False.** Check 4 failed at its first attempt through a fault in my check's code (section 4); all checks passed at the second attempt |
| P53.2 | `wd0` reaches 80 on A within the run under both rules | 85% | True (11,264 and 8,704) |
| P53.3 | `wd0` against `one`: updates to 80 on A are *about the same* under both rules | 60% | True (ratios 0.83 and 0.83) |
| P53.4 | `x3-wd0` reaches 80 on A within the run under back-propagation | 60% | True (9,728) |
| P53.5 | `x3-wd0` reaches 80 on A within the run under the sweep | 45% | True (15,104), although the run is not read because its level of R is 78.5 |
| P53.8 | No new run is *fewer* than the trial's run of its rule on updates to 80 on A | 80% | True |
| P53.9 | `x3` reaches 80 on A within the run under back-propagation | 65% | True (9,728) |
| P53.10 | `x3` reaches 80 on A within the run under the sweep | 50% | True (15,360) |
| P53.12 | In `x3` the mean absolute weight into a cell of ring 3 at update 4,096 is at least 0.30 under both rules | 30% | False (0.169 under S, 0.180 under B) |

## 8. What the results cannot show

The limits of the brief apply unchanged:
- **Scope.** There is one seed, one arm of E47 (`one`; the arm `place` is not run) and one size of the grid.
- **Need.** The first family is learned by 256 cells, so the far cells are not needed here. A run in which they stay unused would not show that they could not be used.
- **Starts.** No start that passes differences outward only was run.
- **The sweep's scale.** The sweep has one scale for all recurrent weights, not one for each cell.

Further limits concern this job:
- **The arms without decay remove the decay from the whole network** (thirteen tensors), so their differences from the trial cannot be assigned to the far cells' weights.
- **`x3` does not keep its multiple.** Under the decay, the multiple of `x3` is gone by update 4,096 under both rules (section 5.5). So `x3` tests a larger start only for the first few thousand updates.
- **The far cells as a whole versus rings 2 and 3.** The diagnostics hold all far cells together; whether rings 2 and 3 take part is left to the reviewer's reading.

## 9. Committed files

The list is printed by `git diff --name-only --diff-filter=A fcc79595 HEAD -- MOD97_LOCAL_RULE_20261106` after the commit that adds this report. It is the set of files that E53 adds; fcc79595 is the last commit of E58. No existing file was changed: `git diff --name-only --diff-filter=M fcc79595 HEAD` is empty.

86 files (this report included):

- `E53_COPIED_SOURCES.txt`
- `e47_local.py`
- `exp45/e53/E53_REPORT.md`
- `exp45/e53/E53_RUN_DIFF.txt`
- `exp45/e53/E53_TABLES.md`
- `exp45/e53/e53_32_wd0_B_73129.json`
- `exp45/e53/e53_32_wd0_B_73129.pt`
- `exp45/e53/e53_32_wd0_B_73129_answers.json`
- `exp45/e53/e53_32_wd0_B_73129_start.pt`
- `exp45/e53/e53_32_wd0_B_73129_u24064.pt`
- `exp45/e53/e53_32_wd0_B_73129_u24320.pt`
- `exp45/e53/e53_32_wd0_S_73129.json`
- `exp45/e53/e53_32_wd0_S_73129.pt`
- `exp45/e53/e53_32_wd0_S_73129_answers.json`
- `exp45/e53/e53_32_wd0_S_73129_start.pt`
- `exp45/e53/e53_32_wd0_S_73129_u24064.pt`
- `exp45/e53/e53_32_wd0_S_73129_u24320.pt`
- `exp45/e53/e53_32_x3-wd0_B_73129.json`
- `exp45/e53/e53_32_x3-wd0_B_73129.pt`
- `exp45/e53/e53_32_x3-wd0_B_73129_answers.json`
- `exp45/e53/e53_32_x3-wd0_B_73129_start.pt`
- `exp45/e53/e53_32_x3-wd0_B_73129_u24064.pt`
- `exp45/e53/e53_32_x3-wd0_B_73129_u24320.pt`
- `exp45/e53/e53_32_x3-wd0_S_73129.json`
- `exp45/e53/e53_32_x3-wd0_S_73129.pt`
- `exp45/e53/e53_32_x3-wd0_S_73129_answers.json`
- `exp45/e53/e53_32_x3-wd0_S_73129_start.pt`
- `exp45/e53/e53_32_x3-wd0_S_73129_u24064.pt`
- `exp45/e53/e53_32_x3-wd0_S_73129_u24320.pt`
- `exp45/e53/e53_32_x3_B_73129.json`
- `exp45/e53/e53_32_x3_B_73129.pt`
- `exp45/e53/e53_32_x3_B_73129_answers.json`
- `exp45/e53/e53_32_x3_B_73129_start.pt`
- `exp45/e53/e53_32_x3_B_73129_u24064.pt`
- `exp45/e53/e53_32_x3_B_73129_u24320.pt`
- `exp45/e53/e53_32_x3_S_73129.json`
- `exp45/e53/e53_32_x3_S_73129.pt`
- `exp45/e53/e53_32_x3_S_73129_answers.json`
- `exp45/e53/e53_32_x3_S_73129_start.pt`
- `exp45/e53/e53_32_x3_S_73129_u24064.pt`
- `exp45/e53/e53_32_x3_S_73129_u24320.pt`
- `exp45/e53/e53_early.py`
- `exp45/e53/e53_tables.py`
- `exp45/e53/logs/e53_32_wd0_B_73129.log`
- `exp45/e53/logs/e53_32_wd0_S_73129.log`
- `exp45/e53/logs/e53_32_x3-wd0_B_73129.log`
- `exp45/e53/logs/e53_32_x3-wd0_S_73129.log`
- `exp45/e53/logs/e53_32_x3_B_73129.log`
- `exp45/e53/logs/e53_32_x3_S_73129.log`
- `exp45/e53/logs/scheduler.log`
- `exp45/e53/queues/all.txt`
- `exp45/e53/scheduler.py`
- `exp45/e53_qual/check1.json`
- `exp45/e53_qual/check1.log`
- `exp45/e53_qual/check1_runs/e53_32_one_B_dry1024_73129.json`
- `exp45/e53_qual/check1_runs/e53_32_one_B_dry1024_73129_answers.json`
- `exp45/e53_qual/check1_runs/e53_32_one_S_dry1024_73129.json`
- `exp45/e53_qual/check1_runs/e53_32_one_S_dry1024_73129_answers.json`
- `exp45/e53_qual/check2.json`
- `exp45/e53_qual/check2.log`
- `exp45/e53_qual/check2_runs/e53_32_wd0_S_dry256_73129.json`
- `exp45/e53_qual/check2_runs/e53_32_wd0_S_dry256_73129_answers.json`
- `exp45/e53_qual/check2_runs/e53_32_x3-wd0_S_dry256_73129.json`
- `exp45/e53_qual/check2_runs/e53_32_x3-wd0_S_dry256_73129_answers.json`
- `exp45/e53_qual/check2_runs/e53_32_x3_S_dry256_73129.json`
- `exp45/e53_qual/check2_runs/e53_32_x3_S_dry256_73129_answers.json`
- `exp45/e53_qual/check2_runs/e53_32_xf3_S_dry256_73129.json`
- `exp45/e53_qual/check2_runs/e53_32_xf3_S_dry256_73129_answers.json`
- `exp45/e53_qual/check3.json`
- `exp45/e53_qual/check3.log`
- `exp45/e53_qual/check4.json`
- `exp45/e53_qual/check4.log`
- `exp45/e53_qual/check4_first_attempt/check4.log`
- `exp45/e53_qual/check4_first_attempt/check4_runs/killed/child.log`
- `exp45/e53_qual/check4_first_attempt/check4_runs/plain/e53_32_x3-wd0_S_dry2048_73129.json`
- `exp45/e53_qual/check4_first_attempt/check4_runs/plain/e53_32_x3-wd0_S_dry2048_73129_answers.json`
- `exp45/e53_qual/check4_runs/killed/child.log`
- `exp45/e53_qual/check4_runs/killed/e53_32_x3-wd0_S_dry2048_73129.json`
- `exp45/e53_qual/check4_runs/killed/e53_32_x3-wd0_S_dry2048_73129_answers.json`
- `exp45/e53_qual/check4_runs/plain/e53_32_x3-wd0_S_dry2048_73129.json`
- `exp45/e53_qual/check4_runs/plain/e53_32_x3-wd0_S_dry2048_73129_answers.json`
- `exp45/e53_qual/check5.json`
- `exp45/e53_qual/check5.log`
- `exp45/e53_qual/check6.json`
- `exp47_local.py`
- `exp53_local_start.py`
