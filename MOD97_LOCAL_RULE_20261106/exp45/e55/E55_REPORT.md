# E55, first part: the first family kept without a line of the forward pass that reads the item (seed 73129)

Evidence level E2: a first measurement on one seed, which confirms nothing. Carried out from the brief `TASK_E55_THE_FIRST_FAMILY_KEPT_WITHOUT_A_LINE_OF_THE_PROGRAM_THAT_LOOKS_AT_THE_ITEM_20261118.md`, revision 4, on the reviewer's branch (at commit `2b85dd8b`). The coordinator released it by decision 13 of 18 November 2026 (`COORDINATOR_DECISIONS_20261118.md` on the default branch, `39eacc33`). The work is on E51's worker's branch, and no file of E51 was changed. No formal checker gate, official seal or external review is claimed. The results are to be verified from the receipts and weights by the reviewer's agent before they are read.

Paths are relative to `MOD97_LOCAL_RULE_20261106/`. Host: Linux x86-64, 4 cores, Python 3.13.18, torch 2.15.0+cpu, at most three processes with one thread each. Every receipt records its host, the hashes of its sources and the git state.

## 1. Terms

The terms of the brief hold; those used most are repeated here.

- *The first family* is four operations on two numbers below 89: addition and subtraction modulo 89, and the larger and the smaller of the two.
- The *reduction score* is the percentage of right answers on the 7,832 pairs of addition and subtraction whose answer needs the reduction modulo 89, written without letters.
- *d(g)* is the number of the 31,684 answers of the first family without letters that differ from row 0's at scored row g. The *held-out answers* are the 1,200 answers on held-out pairs written with 12 letters.
- *The new tasks*:
  - `unrelated` is E43's task on bytes that the first family does not use.
  - `conflict` is the family of the same four operations modulo 97, whose items use the first family's bytes except for their first byte.
- The *never-taught score* is the percentage of right answers on a new task's never-taught items. A task is *learned from row g* if this score is at 80 or more at row g and at every later scored row.
- *The cells in reserve* are cells 128 to 255 of the embedded network of E51. *The plain forward pass* is `exp51_reserve.fwd` without its two added lines, which read the item's first byte.
- *The arm `blind`*: all three rows of every cell in reserve, and rows 256 to 511 of `lm` and `l2`, are kept away from the directions in use of the first family, so that these cells receive nothing from the first family's items.
- *The arm `shut`*: only the update-gate rows of the reserve are kept so, and their bias is raised by 10, so that the update gates of the reserve are shut on the first family's items.
- *1 − z* is how far an update gate is open.

## 2. Code

| File | What it is |
|---|---|
| `exp55_reserve.py` | The runner (new). It imports `exp51_reserve.py` as `R51`, not edited, and asserts its SHA-256 `84d0dad7...` at import. It contains the two starts, `Rule55`, `gates_on`, the run and checks 1 to 3, as the brief writes them. |
| `e55_read.py` | The reader (new). It computes the brief's words and tables from the receipts and writes `exp45/e55/E55_READ_all.json` and `E55_TABLES_all.md`. |

The difference between the text of `exp51_reserve.run` and `exp55_reserve.run`, made with Python's `difflib` from the two function bodies, is in section 10.

## 3. The checks

All three passed at the first attempt (`exp45/e55/checks/e55_check_<n>_73129.json`).

1. **The starts.**
   - (a) Every tensor of each start is bit-equal to E51's keyed start outside its four blocks, and the entries named by `zero_blocks` are exactly zero.
   - (b) The largest entry of `D @ Vm` is at most 1.9e-7, against a limit of 1e-5.
   - (c) *blind* (S; B): the largest state of a cell in reserve on the two panels is 2.4e-7; 3.5e-7, and the largest added readout unit 2.4e-7; 3.3e-7. The largest difference of an output from the keyed start under the keyed forward pass is 1.1e-5; 7.6e-6. On the first batch of a new task the reserve state is 0.153; 0.146 (`unrelated`) and 0.070; 0.087 (`conflict`).
   - (c) *shut* (S; B): the largest 1 − z is 4.542e-5 for both rules. The largest reserve state is 5.6e-4; 5.9e-4, the largest added unit 2.9e-4; 3.6e-4, and the largest output difference 2.4e-3; 2.5e-3.
   - (d) Of the 32,884 generated answers, those that differ from the keyed start's: *blind* 0; 0, *shut* 1; 0.
2. **The rule** (eight updates for each rule, task and arm).
   - In all eight cases the old rows' change @ Vm is at most 6.7e-9, the restored tensors are bit-equal, and every matrix has changed.
   - *blind*: after eight updates the whole rows kept blind have `D @ Vm` at most 4.3e-9. The reserve state on the panels is at most 4.2e-7, and the largest output change at most 1.9e-5. On the new task's bytes, the input map of a reserve row has changed by 0.14 to 0.20 in its largest magnitude.
   - *shut*: 1 − z is 4.54e-5 on the panels and the state at most 8.1e-4. The output change is 0.008 to 0.026, which has no limit in this arm. On the new task's bytes, the input to the gate rows has changed by 0.079 to 0.171 in its largest magnitude; this is above 0.01, so there was nothing to ask. The median magnitude of the gate rows' gradient at the first update is 3.8e-9 (S, `unrelated`), 2.0e-10 (S, `conflict`), 4.6e-8 (B, `unrelated`) and 5.5e-9 (B, `conflict`).
3. **The stream and the cost.** The data digests of the four dry runs of 64 updates equal E51 `keyed`'s at row 64. The forecast was 6.1 CPU-hours for the eight runs; they took 6.15.

**Events during the checks.** Check 1 had started when the four dry runs first failed, before any update, on a quoting error in my launch script. Their logs are in `exp45/e55/logs/aborted/`. The dry runs were then made by a helper outside the repository that calls `exp55_reserve.run` with the folder `checks/dry`; check 3 found them and read them. The dry runs cost about 115 CPU-seconds each.

## 4. Results

The full tables of the twelve scored rows of each run, with E51's rows of `keyed`, `open` and `rehearse` beside them, are in `exp45/e55/E55_TABLES_all.md`. No run was interrupted, and none was started again.

| arm | task | rule | first family | largest d(g) | largest held-out differ | reduction row 0 / 16,384 | learned from row | never-taught at 16,384 | with the reserve shut | largest reserve state on the first family | largest output change | 40 letters differ at 16,384 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `blind` | unrelated | S | kept exactly | 0 | 0 | 92.40 / 92.40 | 8192 | 99.0 | 0.0 | 1.31e-05 | 0.00169 | 0 |
| `blind` | unrelated | B | kept on items of up to 22 positions and not on items of 50 | 0 | 0 | 95.25 / 95.25 | 2048 | 100.0 | 0.0 | 0.00121 | 0.2 | 278 |
| `blind` | conflict | S | kept exactly | 0 | 0 | 92.40 / 92.40 | not learned | 69.4 | 34.9 | 3.33e-06 | 9.87e-05 | 0 |
| `blind` | conflict | B | kept exactly | 0 | 0 | 95.25 / 95.25 | not learned | 63.2 | 28.9 | 2.95e-05 | 0.000349 | 0 |
| `shut` | unrelated | S | partly kept | 135 | 17 | 92.40 / 92.56 | not learned | 72.7 | 0.0 | 0.00106 | 2.09 | 5 |
| `shut` | unrelated | B | partly kept | 34 | 1 | 95.25 / 95.29 | 4096 | 99.7 | 0.0 | 0.001 | 0.834 | 4 |
| `shut` | conflict | S | partly kept | 925 | 72 | 92.40 / 93.00 | not learned | 19.8 | 21.9 | 0.00119 | 9.47 | 236 |
| `shut` | conflict | B | partly kept | 244 | 30 | 95.25 / 95.26 | 8192 | 92.9 | 34.4 | 0.00104 | 2.44 | 2 |

The largest state of a cell in reserve on the first family's held-out items, at rows 0, 1,024, 4,096 and 16,384:

| arm | task | rule | rows 0, 1,024, 4,096, 16,384 |
|---|---|---|---|
| blind | unrelated | S | 0: 2.4e-07 | 1024: 5e-06 | 4096: 5.9e-06 | 16384: 1.3e-05 |
| blind | unrelated | B | 0: 3.5e-07 | 1024: 4.3e-05 | 4096: 0.00044 | 16384: 0.0012 |
| blind | conflict | S | 0: 2.4e-07 | 1024: 1.4e-06 | 4096: 2.6e-06 | 16384: 3.3e-06 |
| blind | conflict | B | 0: 3.5e-07 | 1024: 3.2e-06 | 4096: 2e-05 | 16384: 3e-05 |
| shut | unrelated | S | 0: 0.00056 | 1024: 0.001 | 4096: 0.001 | 16384: 0.001 |
| shut | unrelated | B | 0: 0.00059 | 1024: 0.001 | 4096: 0.001 | 16384: 0.001 |
| shut | conflict | S | 0: 0.00056 | 1024: 0.001 | 4096: 0.0011 | 16384: 0.0012 |
| shut | conflict | B | 0: 0.00059 | 1024: 0.001 | 4096: 0.001 | 16384: 0.001 |

The largest eigenvalue of the reserve's map on a byte in use (0.5 I + 0.25 Wn) is 0.605 at row 0 in both arms. At row 16,384 it is 1.098 (blind, `unrelated`, S), 1.434 (blind, `unrelated`, B), 1.058 (blind, `conflict`, S) and 1.197 (blind, `conflict`, B).

### 4.1 The arm `blind` on items of 50 positions, the reserve's map, the arithmetic and the leads

These paragraphs enter findings 1, 2, 3 and 5 of the reviewer's verification of the arm `blind` (`E55_BLIND_VERIFICATION_20261118.md` and `e55_blind_verify.py` on the reviewer's branch, commit `77e65eba`). That verification reproduced all 1,227 figures of the tables, the reading and the message of the four runs from the receipts and the twelve saved networks.

**Items of 50 positions, `unrelated` under back-propagation.** The 40-letter panel holds 300 held-out pairs of addition, each 50 positions long. The number of answers that differ from row 0's does not by itself say what happens to the first family on these items; the right answers do. They fall from 274 of 300 at row 0 to 21 of 300 at row 16,384. On these items the largest state of a cell in reserve reaches 0.9998, and the largest 1 − z of a cell in reserve 0.997, at row 16,384 (`gates_on_first_family_40_letters` in the receipt). The states of the cells in reserve thus reach the limit of the hyperbolic tangent, and some of their update gates open almost fully. On items of 50 positions the first family is lost in this run.

**The reserve's map, by row.** Wn is the block of the candidate rows of the cells in reserve that takes the states of the cells in reserve. The table gives, for all four runs of the arm `blind`, the largest state of a cell in reserve on the 40-letter panel and the largest magnitude of an eigenvalue of 0.5 I + 0.25 Wn; for `unrelated` under back-propagation it also gives the right answers and the largest 1 − z on that panel.

| task | rule | quantity | 0 | 16 | 32 | 64 | 128 | 256 | 512 | 1024 | 2048 | 4096 | 8192 | 16384 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| unrelated | S | largest reserve state, 40 letters | 2.4e-07 | 2.9e-07 | 4.3e-07 | 4.7e-07 | 6e-07 | 5.7e-07 | 5.7e-07 | 6e-05 | 7.2e-06 | 4.6e-05 | 0.00015 | 0.00017 |
| unrelated | S | eigenvalue of 0.5 I + 0.25 Wn | 0.605 | 0.809 | 0.771 | 0.689 | 0.710 | 0.684 | 0.776 | 1.094 | 0.975 | 1.060 | 1.112 | 1.098 |
| unrelated | B | largest reserve state, 40 letters | 3.2e-07 | 4.5e-07 | 7.6e-07 | 9.6e-07 | 1.1e-06 | 1.2e-06 | 1.8e-06 | 0.0024 | 0.3 | 0.33 | 0.95 | 1 |
| unrelated | B | eigenvalue of 0.5 I + 0.25 Wn | 0.605 | 0.716 | 0.734 | 0.907 | 0.905 | 0.837 | 0.864 | 1.193 | 1.324 | 1.342 | 1.385 | 1.434 |
| unrelated | B | right answers of 300, 40 letters | 274 | 274 | 274 | 274 | 274 | 274 | 274 | 274 | 264 | 238 | 101 | 21 |
| unrelated | B | largest 1 − z, 40 letters | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 | 0.502 | 0.875 | 0.913 | 0.982 | 0.997 |
| conflict | S | largest reserve state, 40 letters | 2.4e-07 | 3e-07 | 3.6e-07 | 2.9e-07 | 4e-07 | 2.8e-06 | 5.2e-06 | 1.2e-06 | 2e-06 | 2.1e-06 | 8.6e-06 | 1.1e-05 |
| conflict | S | eigenvalue of 0.5 I + 0.25 Wn | 0.605 | 0.859 | 0.933 | 0.930 | 0.992 | 1.066 | 1.062 | 1.005 | 0.950 | 1.012 | 1.039 | 1.058 |
| conflict | B | largest reserve state, 40 letters | 3.2e-07 | 4.6e-07 | 5.2e-07 | 1.1e-06 | 6.9e-06 | 4.8e-06 | 1.2e-05 | 2.5e-05 | 0.00026 | 0.00097 | 0.0014 | 0.0034 |
| conflict | B | eigenvalue of 0.5 I + 0.25 Wn | 0.605 | 0.709 | 0.847 | 0.969 | 1.068 | 1.043 | 1.064 | 1.103 | 1.161 | 1.170 | 1.180 | 1.197 |

The eigenvalue is above 1 at row 16,384 in all four runs (1.098, 1.434, 1.058, 1.197).
- **A limit of the construction.** The weights among the cells in reserve are left free, so nothing bounds the state of the reserve on longer items or after more updates.
- **What "kept exactly" means here.** It is an observation on items of up to 22 positions.

**The arithmetic.** The 40-letter counts, the small states of the reserve and the changes of the outputs depend on the arithmetic as well as on the weights. On a first-family item a cell in reserve receives only the remainder of the projection of its rows, which single-precision storage leaves at about 1e-8, and the rounding of the arithmetic. At row 4,096 of `unrelated` under back-propagation, the receipt gives 45 changed answers at 40 letters; the reviewer's numpy gives 15 in double precision and 142 in single precision from the same saved weights. The outcomes of the brief's rules are the same in each arithmetic: more than 3 of the 300 answers differ in that run and none in the other three, and the 32,884 scored answers are identical.

**The outputs and the smallest lead.** The *lead* of an answer position is the difference between the largest and the second largest output there. The outputs move by more than the smallest lead without any answer changing on the scored panels. Under the sweep the largest change is 1.69e-3 against a smallest lead of 1.78e-4; under back-propagation it is 0.200 against 2.33e-2. *Kept exactly* is therefore an observation on these panels, not the consequence of a bound.

## 5. The words for each run

**Arm `blind`.**

- `unrelated`, sweep: **kept exactly**; no answer at 40 letters differs at any row. The task is **learned from row 8,192** (99.0 at row 16,384). *The first family was kept while the task was learned, with the plain forward pass.* With the cells in reserve held shut, the never-taught score is 0.0.
- `unrelated`, back-propagation: **kept on items of up to 22 positions and not on items of 50**, the brief's wording for this run.
  - On the scored panels no answer changes at any row.
  - Of the 300 held-out pairs at 40 letters, the answers that differ from row 0's are 21 at row 2,048, 45 at 4,096, 194 at 8,192 and 278 at 16,384. The right answers fall from 274 at row 0 to 21 at row 16,384 (section 4.1).
  - The task is **learned from row 2,048** (100.0). With the cells in reserve held shut, the never-taught score is 0.0.
- `conflict`, sweep: **kept exactly**. *The task was not learned within 16,384 updates*: 69.4 at row 16,384.
- `conflict`, back-propagation: **kept exactly**. *The task was not learned within 16,384 updates*: 63.2 at row 16,384.
- For both runs of `conflict`, the largest 1 − z on the never-taught items at rows 1,024 to 16,384 is 0.93 to 1.00, with 128 cells open at some position at every row. In this arm the gates stand near one half on every item, so these figures carry no information about the use of the reserve.

**Arm `shut`.** In all four runs the largest 1 − z of a cell in reserve on the first family's held-out items stays between 4.5e-5 and 6.5e-5 at every row, and no cell opens there. The gates stayed shut on the first family's items, while the free weights from the cells in reserve, acting on states of about 1e-3, changed answers.

- `unrelated`, sweep: **partly kept**. *The task was not learned within 16,384 updates*: 72.7 at row 16,384.
  - Cells opened on the new task's items: 126 open at some position at row 16,384, 125 at row 8,192 and 16 at row 4,096. The task was not learned although they opened. The largest 1 − z there was 0.002 at row 256, 0.34 at row 1,024 and 1.0 from row 4,096.
  - The first family: d(g) at most 135 (0.43%), at most 17 held-out answers differ, and the reduction score is 92.56 at the last row against 92.40.
  - *Kept* first fails at row 64: d(64) = 42 and 2 held-out answers differ. There, on the first family's items, the largest 1 − z is 4.6e-5, no cell is open, the largest reserve state is 1.0e-3 and the largest added unit 0.0086.
- `unrelated`, back-propagation: **partly kept**; the task is **learned from row 4,096** (99.7). *The first family was kept except for 0.11% of its answers* (largest d(g) 34; at most 1 held-out answer differs; reduction score 95.29 against 95.25).
  - *Kept* first fails at row 1,024: d(1,024) = 34. There the largest 1 − z on the first family's items is 4.6e-5, the largest reserve state 1.0e-3 and the largest added unit 0.044.
  - All 128 cells are open at some position on the new task's items at row 16,384. With the cells in reserve held shut, the never-taught score is 0.0.
- `conflict`, sweep: **partly kept**. *The task was not learned within 16,384 updates*: 19.8 at row 16,384. *Learning did not open the cells in reserve*: no cell is open at some position at any row, and the largest 1 − z on the new task's items is below 0.001 at every row.
  - The first family: d(g) at most 925 (2.92%), at most 72 held-out answers differ, and the reduction score is 93.00 at the last row against 92.40.
  - *Kept* first fails at row 32: d(32) = 37 and 2 held-out answers differ. There the largest 1 − z on the first family's items is 4.5e-5, the largest state 9.7e-4 and the largest added unit 0.0048.
- `conflict`, back-propagation: **partly kept**; the task is **learned from row 8,192** (92.9). *The first family was kept except for 0.77% of its answers* (largest d(g) 244; at most 30 held-out answers differ; reduction score 95.26 against 95.25).
  - *Kept* first fails at row 32, where 3 held-out answers differ. There the largest 1 − z on the first family's items is 4.5e-5, the largest state 9.7e-4 and the largest added unit 0.0040.
  - 58 cells are open at some position on the new task's items at row 16,384. With the cells in reserve held shut, the never-taught score is 34.4.

## 6. The two arms side by side

| task | rule | `blind`: first family (largest d(g)) | `blind`: learned from row | `blind`: reserve shut | `shut`: first family (largest d(g)) | `shut`: learned from row | `shut`: reserve shut |
|---|---|---|---|---|---|---|---|
| unrelated | S | kept exactly (0) | 8,192 | 0.0 | partly kept (135) | not learned (72.7) | 0.0 |
| unrelated | B | kept on items of up to 22 positions and not on items of 50 (0) | 2,048 | 0.0 | partly kept (34) | 4,096 | 0.0 |
| conflict | S | kept exactly (0) | not learned (69.4) | 34.9 | partly kept (925) | not learned (19.8) | 21.9 |
| conflict | B | kept exactly (0) | not learned (63.2) | 28.9 | partly kept (244) | 8,192 | 34.4 |

**`unrelated`.**
- The arm `blind` learns `unrelated` under both rules, with a score of 0.0 when the cells in reserve are held shut.
- **Under the sweep:** *the task was learned and the first family kept exactly in a network whose cells in reserve receive nothing from the first family's items.*
- **Under back-propagation:** the same holds on items of up to 22 positions only; the first family is not kept on items of 50 positions.
- Neither sentence says that these cells sufficed, because the rows of the old cells and of the old readout units change as well.

**`conflict`.**
- In the arm `blind`, the task was not learned within 16,384 updates by a network whose cells in reserve receive only the operation byte and their own states. The cells in reserve raised the never-taught score by 34.5 points under the sweep (69.4 against 34.9 when they are held shut) and by 34.4 points under back-propagation (63.25 against 28.875; 34.375 unrounded).
- Under back-propagation the arm `shut` learns `conflict` and the arm `blind` does not. *On this seed the task was learned by the arm in which a cell in reserve may read what the first family presents, and not by the arm in which it may not.* The arms also differ in that the gates are shut by a bias in the one and stand at one half in the other. Beside this, the arm `shut` lost 244 of the first family's answers (0.77%) and 30 of the 1,200 held-out answers.
- Under the sweep neither arm learns `conflict`, and no comparison is made.

**The number of updates.** The scored rows double from one to the next, so a ratio of 2 is one row. In the arm `blind`, `unrelated` is learned three doublings later than in E51's `keyed` under the sweep (8,192 against 1,024; `open` 512), and two doublings later under back-propagation (2,048 against 512; `open` 512). Each row where a task is learned, beside E51's `keyed` and `open` for the same task and rule:

| run | learned from row | E51 `keyed` | E51 `open` |
|---|---|---|---|
| `blind`, `unrelated`, S | 8,192 | 1,024 | 512 |
| `blind`, `unrelated`, B | 2,048 | 512 | 512 |
| `shut`, `unrelated`, B | 4,096 | 512 | 512 |
| `shut`, `conflict`, B | 8,192 | 512 | 1,024 |

The score with the cells in reserve held shut is given as a figure throughout. The report does not say that a task was "learned in the reserve".

## 7. What remains supplied by the program

As the brief states, this job removes only the reading of the item by the forward pass. The following remain acts of the program:

- The directions in use are computed once, by the program, from 2,080 taught items of the first family, before the new task begins. The network does not keep them during its own learning.
- The rows that the rule protects are named by the program:
  - the old rows;
  - the gate rows of the reserve (arm `shut`) or all rows of the reserve (arm `blind`);
  - every row of `lm` and of `l2`.

  `l1`, the old columns and the bias of `l3`, and the rows of the byte table of the bytes in use are restored by the program after each step.
- The cells in reserve and the added units are put in by the program at the change of task. In the arm `shut` the program also shuts the gates with a bias of 10.
- The change of task is known to the program: the directions are fixed, and a new optimizer is made, at that moment.

Two parts of these acts are named here, at the reviewer's request:
- **The record of the bytes in use.** The 46 bytes of the first family's items are recorded by the program, and the rows of the byte table for these bytes are restored after every update.
- **One use of the keyed forward pass.** It is used once, at the change of task, to compute the directions in use. These were computed for E51 and saved with its keyed start, and E55 reads the saved directions. No forward pass of E55's updates or scoring reads the item.

## 8. The predictions

| | Statement | Chance | Outcome |
|---|---|---|---|
| P55.1 | Checks 1 to 3 pass at the first attempt | 50% | Held: one receipt for each of checks 1 to 3, each with `ok` true. The first launch of the dry runs failed on a quoting error in my launch script before any update, and no check failed. |
| P55.2 | In the arm `shut`, `unrelated` is learned under both rules | 45% | Failed: learned under B only (S: 72.7) |
| P55.2b | In the arm `shut`, `unrelated` is learned under back-propagation | 80% | Held (from row 4,096) |
| P55.3 | In the arm `shut`, `conflict` is learned under both rules | 5% | Failed: learned under B only (from row 8,192); S 19.8 |
| P55.4 | In the arm `shut`, the first family is kept exactly or kept in all four runs | 3% | Failed: partly kept in all four |
| P55.4b | In the arm `shut`, the reduction score is within 2 points of row 0's at every scored row of all four runs | 50% | Held (largest deviation 1.43, `conflict`, S) |
| P55.5 | In the arm `shut`, the largest 1 − z on the first family's held-out items is below 1e-3 at every scored row of every run | 85% | Held (largest 6.5e-5) |
| P55.6 | In the arm `shut`, every run in which the task is learned has at least 32 cells open at some position on the new task's items at row 16,384 | 80% | Held (128 for `unrelated`, B; 58 for `conflict`, B) |
| P55.8 | In the arm `shut`, where `unrelated` is learned, it is learned from row 4,096 or earlier | 55% | Held (B, from row 4,096; not learned under S) |
| P55.9 | In the arm `blind`, the first family is kept exactly in all four runs | 55% | Held by the definition of *kept exactly* on the scored panels. The brief's wording for `unrelated`, B is "kept on items of up to 22 positions and not on items of 50" |
| P55.10 | In the arm `blind`, d(g) is at most 31 at every scored row of all four runs | 80% | Held (0 throughout) |
| P55.11 | In the arm `blind`, the largest state of a cell in reserve on the first family's held-out items is below 1e-4 at every scored row of every run | 70% | Failed: `unrelated`, B is 3.4e-4 at row 2,048 and 1.2e-3 at row 16,384 |
| P55.12 | In the arm `blind`, `unrelated` is learned under both rules | 65% | Held (from rows 8,192 and 2,048) |
| P55.13 | In the arm `blind`, `conflict` is learned under neither rule | 70% | Held (69.4; 63.2) |

## 9. What the job cannot show, deviations and costs

- **What the job cannot show.** The limits in the brief's section of that name apply unchanged. In particular:
  - there is one seed and there are two tasks;
  - no third task is tested;
  - the state of the reserve on long items is not bounded, and is measured on one panel of 40 letters;
  - no arm has both a reserve that reads the first family's content and an exact keeping.
- **The order of the arms.** The four runs of `blind` were reported to the reviewer before the four runs of `shut` were started, as the coordinator's ruling asks.
- **Costs.** The eight runs took 6.15 CPU-hours, 2,536 to 2,992 seconds each, against a forecast of about 1,370 seconds a run. The forecast came from 64-update dry runs; at 16,384 updates the scoring of twelve rows with longer items, and the slower generation once the network answers more, cost more.
- **Interruptions.** No machine restart occurred and no run was interrupted. The folder `exp45/e55/interrupted/` is empty and not committed.
- **Commit messages.** Every commit of this job carries the trailers of `PROJECT_DOCUMENTS/KB/AGENTS_POINTER.md` and no name of a language model.
- **Provenance.**
  - The four starts and checks 1 to 3 were made at commit `2abb0c3f`, while `exp55_reserve.py` was not yet committed. Their receipts record the runner's hash, which equals that of the committed file.
  - The receipts of the four dry runs of check 3 record a description in place of a command, because a helper called `exp55_reserve.run` directly. The helpers and the command behind each dry run are now committed in `helpers/`, with `helpers/README_HELPERS.md`, which also names the aborted launcher's logs.
  - The run of `conflict` under back-propagation in the arm `blind` records commit `4e2a72c2`, and the other three record `d974f048`; the runner is the same file in both.
- **The limits of the arm `blind`.** The weights among the cells in reserve are free, so nothing bounds the state of the reserve on longer items or after more updates (section 4.1).

## 10. The difference between `exp51_reserve.run` and `exp55_reserve.run`

```diff
--- exp51_reserve.run
+++ exp55_reserve.run
@@ -1,30 +1,25 @@
-def run(task, arm, rule, seed, out, cmd, updates=DOSE, start="embed", restore_book=True):
-    local = C.ARMS[rule]; cfg = ARMS[arm]; out = Path(out)
-    tag = f"e51_{task}_{arm}_{rule}_{seed}" + ("_e26birth" if start == "e26" else "") + ("" if updates == DOSE else f"_dry{updates}")
-    path = out / f"{tag}.json"; R.refuse_existing(path); out.mkdir(parents=True, exist_ok=True); (out / "rows_uncommitted").mkdir(exist_ok=True)
-    if start == "e26":                                           # check 6: E26's birth as it is, no embedding, no change of any bias
-        assert arm == "open"
-        install(None)
-        bfile = C.birth_path(E38.E26DIR, rule, seed); m, birth_sha, admission = E38.admit(bfile, rule, seed)
-        m.load_state_dict(T.load(bfile)); in_use = dirs = None
-        start_rec = dict(birth=str(bfile.relative_to(HERE)), birth_sha256=birth_sha, birth_admission=admission)
-    else:
-        m, m0, in_use, dirs, start_rec = load_start(cfg["start"], rule, seed)
+def run(task, arm, rule, seed, out, cmd, updates=4 * R51.DOSE):
+    assert arm in ("blind", "shut")
+    local = R51.C.ARMS[rule]; out = Path(out)
+    tag = f"e55_{task}_{arm}_{rule}_{seed}" + ("" if updates == 4 * R51.DOSE else f"_dry{updates}")
+    path = out / f"{tag}.json"; R51.R.refuse_existing(path); out.mkdir(parents=True, exist_ok=True); (out / "rows_uncommitted").mkdir(exist_ok=True)
+    m, in_use, dirs, start_rec, srec = load_checked(arm, rule, seed)
     bsd = {k: v.clone() for k, v in m.state_dict().items()}
-    start_tsha = R.tensors_sha(bsd)
-    opt = C.make_opt(m)
+    start_tsha = R51.R.tensors_sha(bsd)
+    opt = R51.C.make_opt(m)
     opt.param_groups[0]["weight_decay"] = 0.0
-    groups_read_back = R.opt_groups(m, opt)
-    ruleobj = Rule(m, dirs, in_use, restore_book) if cfg["rule"] else None
+    groups_read_back = R51.R.opt_groups(m, opt)
+    ruleobj = Rule55(m, dirs, in_use, arm)
 
-    P, comp = X16.panels(seed)
+    P, comp = R51.X16.panels(seed)
     held = {n: P[n][12] for n in ("add89", "sub89", "max89", "min89")}
-    s1 = U.first_family_items(); pairs = first_family_pairs()
+    s1 = R51.U.first_family_items(); pairs = R51.first_family_pairs()
     s2_items = [it for n in held for it in held[n]]
-    npan = new_panels(task, seed, P)
+    npan = R51.new_panels(task, seed, P)
     reserve_panel = [it for n in (("add97", "sub97") if task == "conflict" else ("held",)) for it in npan[n]]
-    nxt, stream_note = item_source(task, arm, seed)
-    dg = C.Digest(); counts = {task: 0, "first_family": 0}; losses = []
-    t0 = time.process_time(); cpu_score = [0.0]; rows = []; gen0 = [None]
-    rowset = [r for r in ROWS if r <= updates]
+    nxt, stream_note = R51.item_source(task, "keyed", seed)
+    len40 = R51.E.panel(("add", 89, b"p"), R51.E.split(("add", 89, b"p"), .5, seed)[1], seed, ks=(12, 40), n=300)[40]
+    dg = R51.C.Digest(); counts = {task: 0, "first_family": 0}; losses = []
+    t0 = time.process_time(); cpu_score = [0.0]; rows = []; gen0 = [None]; logits0 = [None]; gen40_0 = [None]
+    rowset = [r for r in ROWS55 if r <= updates]
 
@@ -33,3 +28,3 @@
         r = dict(updates=g, data_digest=dg.value(), items_so_far=dict(counts), loss_since_last_row=(sum(losses) / len(losses)) if losses else None)
-        s1r, s2r, share, classes, gens = score_first_family(m, s1, pairs, held, gen0[0])
+        s1r, s2r, share, classes, gens = R51.score_first_family(m, s1, pairs, held, gen0[0])
         if g == 0:
@@ -38,22 +33,33 @@
         r["share_of_answers_equal_to_row0"] = share; r["wrong_answers_on_reduction_pairs"] = classes
-        if g in GEN_ROWS:
-            gp = out / f"{tag}_reduction_bytes_row{g}.json"; gp.write_text(json.dumps(reduction_bytes(gens, s1)))
-            r["reduction_bytes"] = gp.name; r["reduction_bytes_sha256"] = X16.sha_file(gp)
-        s3 = {n: X16.score(m, it, None) for n, it in npan.items()}
-        r["score3_new_task"] = s3; r["score3_never_taught"] = never_taught(task, s3)
-        if start != "e26":
-            KEY["shut_all"] = True
-            s3s = {n: X16.score(m, it, None) for n, it in npan.items()}
-            KEY["shut_all"] = False
-            r["score3_new_task_reserve_shut"] = s3s; r["score3_never_taught_reserve_shut"] = never_taught(task, s3s)
-            r["reserve"] = reserve_figures(m, reserve_panel, s2_items)
-        mv, split, norms = weights_figures(m, bsd)
+        if g in R51.GEN_ROWS:
+            gp = out / f"{tag}_reduction_bytes_row{g}.json"; gp.write_text(json.dumps(R51.reduction_bytes(gens, s1)))
+            r["reduction_bytes"] = gp.name; r["reduction_bytes_sha256"] = R51.X16.sha_file(gp)
+        s3 = {n: R51.X16.score(m, it, None) for n, it in npan.items()}
+        r["score3_new_task"] = s3; r["score3_never_taught"] = R51.never_taught(task, s3)
+        R51.KEY["shut_all"] = True
+        s3s = {n: R51.X16.score(m, it, None) for n, it in npan.items()}
+        R51.KEY["shut_all"] = False
+        r["score3_new_task_reserve_shut"] = s3s; r["score3_never_taught_reserve_shut"] = R51.never_taught(task, s3s)
+        r["reserve"] = R51.reserve_figures(m, reserve_panel, s2_items)
+        r["gates_on_first_family"] = gates_on(m, s2_items)
+        r["gates_on_new_task"] = gates_on(m, reserve_panel)
+        lg = R51.answer_logits(m, s2_items)
+        if g == 0:
+            logits0[0] = lg
+        r["first_family_largest_change_of_an_output"] = float((lg - logits0[0]).abs().max())
+        r["reserve_map_largest_eigenvalue"] = float(T.linalg.eigvals(0.5 * T.eye(128, dtype=T.float64) + 0.25 * m.rnn.weight_hh_l0[640:768, 128:256].double()).abs().max())
+        r["gates_on_first_family_40_letters"] = gates_on(m, len40)
+        g40 = R51.generate(m, len40)
+        if g == 0:
+            gen40_0[0] = g40
+        r["first_family_40_letters"] = dict(n=len(len40), differ_from_row0=sum(a != b for a, b in zip(g40, gen40_0[0])),
+                                            exact=sum(o == t for o, (_, t) in zip(g40, len40)))
+        mv, split, norms = R51.weights_figures(m, bsd)
         r["score4_movement"] = mv; r["tensor_norms"] = norms
-        if start != "e26":
-            r["score4_movement_old_and_reserve_rows"] = split
+        r["score4_movement_old_and_reserve_rows"] = split
         cpu_score[0] += time.process_time() - cb
-        r["tensors_sha256"] = R.tensors_sha(m.state_dict())
-        if g in KEEP_ROWS:
-            wp = (out if updates == DOSE else out / "rows_uncommitted") / f"{tag}_row{g}.pt"
-            r["weights"] = str(wp.relative_to(out)); R.save_weights(m, wp); r["weights_sha256"] = X16.sha_file(wp)
+        r["tensors_sha256"] = R51.R.tensors_sha(m.state_dict())
+        if g in KEEP55:
+            wp = (out if updates == 4 * R51.DOSE else out / "rows_uncommitted") / f"{tag}_row{g}.pt"
+            r["weights"] = str(wp.relative_to(out)); R51.R.save_weights(m, wp); r["weights_sha256"] = R51.X16.sha_file(wp)
         r["cpu"] = time.process_time() - t0
@@ -61,18 +67,27 @@
         pc = lambda d: round(100 * d["exact"] / d["n"], 1)
-        print(tag, "row", g, "reduction", round(s1r["reduction_score"], 1), "other", round(s1r["add89"]["other"]["pct"], 1), round(s1r["sub89"]["other"]["pct"], 1),
-              "max/min", round(s1r["max89"]["all"]["pct"], 1), round(s1r["min89"]["all"]["pct"], 1), "held k12", {n: pc(v) for n, v in s2r.items()},
-              "same as row 0", round(100 * share["all_pairs_k0"], 2), "new", {n: pc(v) for n, v in s3.items()},
-              "shut", {n: pc(v) for n, v in r.get("score3_new_task_reserve_shut", {}).items()}, "reserve", r.get("reserve"), f"cpu {r['cpu']:.0f}s", flush=True)
+        gf = r["gates_on_first_family"]
+        print(tag, "row", g, "reduction", round(s1r["reduction_score"], 1), "max/min", round(s1r["max89"]["all"]["pct"], 1), round(s1r["min89"]["all"]["pct"], 1),
+              "held k12", {n: pc(v) for n, v in s2r.items()}, "same as row 0", round(100 * share["all_pairs_k0"], 3),
+              "new", r["score3_never_taught"]["pct"], "shut", r["score3_never_taught_reserve_shut"]["pct"],
+              "ff state", f"{gf['largest_reserve_state']:.2e}", "ff 1-z", f"{gf['largest_opening']:.2e}", "out change", f"{r['first_family_largest_change_of_an_output']:.2e}",
+              "eig", round(r["reserve_map_largest_eigenvalue"], 3), "40 letters differ", r["first_family_40_letters"]["differ_from_row0"],
+              "new open at some position", r["gates_on_new_task"]["cells_open_at_some_position"], f"cpu {r['cpu']:.0f}s", flush=True)
 
-    settings = dict(experiment="E51", part="A", task=task, arm=arm, rule=rule, local=local, seed=seed, updates=updates, start=start_rec,
-                    start_form=cfg["start"] if start != "e26" else "E26 birth", start_tensors_sha256=start_tsha, keyed_forward=in_use is not None and cfg["start"] == "keyed",
-                    rule_applied=cfg["rule"], restore_book=restore_book if cfg["rule"] else None, stream=stream_note, decay=0.0,
-                    optimizer_groups_read_back=groups_read_back, model=S41.describe(m), batch=X.BATCH, rows=rowset, committed_rows=list(KEEP_ROWS),
-                    update="exp51_reserve.step (exp26_candidate.step's text; forward exp51_reserve.fwd; the rule's two additions in arm keyed)",
-                    forward="exp51_reserve.fwd (arch26.forward's text; z and z2 held on items whose first byte is in use, in the keyed arms)",
+    settings = dict(experiment="E55", part="first", task=task, arm=arm, rule=rule, local=local, seed=seed, updates=updates, start=start_rec,
+                    start_e55=dict(form=arm, file=srec["weights"], sha256=srec["sha256"], tensors_sha256=srec["tensors_sha256"]),
+                    start_form=arm, start_tensors_sha256=start_tsha, keyed_forward=False,
+                    rule_applied=True, restore_book=True, stream=stream_note, decay=0.0,
+                    rule_rows=("E51's rows (the old rows of the input map and the recurrent matrix, rows 0-255 of lm and l2) by the step, then the whole change "
+                               "from the start projected; " + ("the three rows of each cell in reserve and rows 256-511 of lm and l2 kept blind (the whole row "
+                               "projected after every update)" if arm == "blind" else "the update-gate rows of the reserve (384-511) and rows 256-511 of lm and l2 "
+                               "joined to E51's rows, by the step, then the whole change from the start projected")),
+                    gate_bias=GATE_BIAS if arm == "shut" else None,
+                    optimizer_groups_read_back=groups_read_back, model=R51.S41.describe(m), batch=R51.X.BATCH, rows=rowset, committed_rows=list(KEEP55),
+                    update="exp51_reserve.step (exp26_candidate.step's text; forward exp51_reserve.fwd without a key); the rule exp55_reserve.Rule55",
+                    forward="exp51_reserve.fwd with KEY['in_use'] = None (no added line)",
                     provenance=provenance(cmd))
-    assert KEY["in_use"] is (in_use if (start != "e26" and cfg["start"] == "keyed") else None)
+    assert R51.KEY["in_use"] is None
 
     def write(complete=None):
-        path.write_text(json.dumps(dict(experiment="E51", kind="run", tag=tag, settings=settings, rows=rows, complete=complete), indent=1))
+        path.write_text(json.dumps(dict(experiment="E55", kind="run", tag=tag, settings=settings, rows=rows, complete=complete), indent=1))
 
@@ -81,3 +96,3 @@
         batch = []
-        for _ in range(X.BATCH):
+        for _ in range(R51.X.BATCH):
             kind, it = nxt(); counts[kind] += 1
@@ -89,3 +104,3 @@
         dg.add(batch)
-        losses.append(step(m, opt, batch, local, ruleobj))
+        losses.append(R51.step(m, opt, batch, local, ruleobj))
         g += 1
@@ -93,4 +108,4 @@
             score_row(g); write()
-    write(dict(updates=g, items=counts, optimizer_steps=R.opt_steps(opt), optimizer_groups_read_back=R.opt_groups(m, opt), data_digest=dg.value(),
-               tensors_sha256=R.tensors_sha(m.state_dict()), cpu_total=time.process_time() - t0, cpu_scoring=cpu_score[0],
+    write(dict(updates=g, items=counts, optimizer_steps=R51.R.opt_steps(opt), optimizer_groups_read_back=R51.R.opt_groups(m, opt), data_digest=dg.value(),
+               tensors_sha256=R51.R.tensors_sha(m.state_dict()), cpu_total=time.process_time() - t0, cpu_scoring=cpu_score[0],
                cpu_updates=time.process_time() - t0 - cpu_score[0]))

```

## 11. Committed files

The list is in `E55_COMMITTED_FILES.txt` beside this report, made by `git ls-tree -r --name-only HEAD MOD97_LOCAL_RULE_20261106/exp45/e55 MOD97_LOCAL_RULE_20261106/exp55_reserve.py MOD97_LOCAL_RULE_20261106/e55_read.py` at the commit that adds the list.
