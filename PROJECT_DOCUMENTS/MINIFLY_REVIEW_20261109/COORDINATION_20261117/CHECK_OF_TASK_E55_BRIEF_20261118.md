# Check of the brief of E55, first part (revision 1), before its release

18 November 2026. Written for the outside reviewer by an agent that was given the brief and the files it builds on, and not the author's reasoning. The brief checked is `TASK_E55_THE_FIRST_FAMILY_KEPT_WITHOUT_A_LINE_OF_THE_PROGRAM_THAT_LOOKS_AT_THE_ITEM_20261118.md` in this folder, at commit 8cbcb64c. The check is a reading and a recomputation with numpy. No code of another branch was run and torch was not used.

**Verdict: do not release as it stands; release after the listed changes.** Thirteen defects are listed. Five of them decide what the job would report and must be settled by the author before the release. Check 2 (b) cannot be met, so that the worker would stop before the first run (defect 1). The record's statement that the gates did not open in E51's arm `open` is wrong for the task `unrelated`, and the count on which it rests cannot see a gate that opens at one position (defects 2 and 3). Five of the sentences fixed in "How the result is stated" can be licensed when the runs do not support them (defect 4). The 4,096 updates are carried over from E51, where the gates started half open; in an emulation of this job the gates begin to open between rows 1,024 and 4,096 (defect 5). The construction itself is sound as arithmetic: the limits of check 1 can be met, and the runner can be built from the listed changes once three lines that the list does not cover are dealt with (defect 9).

## Terms and method

The terms are the brief's. *The start `shut`*, *the gate rows of the reserve*, *the directions in use* and *the two added lines* are defined there. *The keyed start* is E51's embedded network as saved before any update, and *the keyed forward* is the forward pass with the two added lines. An update gate is called *shut* when its z is close to 1, and *open by x* when 1 − z is x. *The summed input* of a gate, a reset gate or a candidate is the number to which the sigmoid or the hyperbolic tangent is applied. A cell whose gate is shut still takes a small state on an item, at most the number of positions times 1 − z; this note calls it *the state of a shut cell*.

The script is `check_e55_brief.py` in this folder. It imports the reviewer's verification script of E51's trial (`e51_trial_verify.py`, one folder above) and uses that script's reader of weight files, its items, panels and streams, and its forward pass and generation of answers (`Net`, `answer_logits`, `generate`, `score_first_family`, `score_new`). It reads every file with `git show` and writes nothing, unless the part `emulate` is given a folder outside the repository for its emulated weights. Its parts are named where a figure is given:

| Part | What it computes |
|---|---|
| `record` | the figures of the brief's "What the record says" from the sixteen receipts of E51 |
| `open` | the update gates of the cells in reserve, position by position, in the weights of row 4,096 of E51's arm `open` |
| `start` | the start `shut`, written in numpy from the saved keyed start and the saved directions as the brief's `shut_start` writes it; the figures of the brief's check 1; what the gates, reset gates and candidates of the cells in reserve receive; the part of each new byte's code outside the directions in use |
| `grad` | the gradient at the start `shut` on the first batch of each new task, under both rules, by a backward pass written in the script (the exact gradient agrees with central finite differences of the loss to a relative difference of 6.9e-5 on 24 entries) |
| `check2` | the brief's check 2 after eight emulated updates |
| `residue` | under the sweep, the error that is sent back through entries that the projection brought to zero |
| `emulate` | an emulation of the update of the arm `shut` (the gradient, the rule's two additions, AdamW with rate 0.001, betas 0.9 and 0.98, eps 1e-8 and no decay, the restoring), for up to 4,096 updates |
| `positions` | in the emulated weights of `conflict` under back-propagation at row 4,096, the positions at which the gates of the cells in reserve open |

*The emulation* is the checker's own arithmetic and is not the runner. It computes in double precision and stores the weights in single precision after every step; torch computes in single precision. It was validated against the record: 64 emulated updates of E51's arm `keyed` on `unrelated` give the saved weights of row 64 to within 4.7e-5 in every entry under both rules, where the weights changed by up to 5.9e-2 (part `emulate`, first lines). Its figures after many updates show sizes and the order of events; they are not the figures that the worker's runs will give, because small differences of rounding grow over thousands of updates.

Commands: `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -I PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/COORDINATION_20261117/check_e55_brief.py <part>`, from the repository root. The part `emulate` reads five environment variables, which the first lines of the script describe: `E55_CHECK_UPDATES` (number of updates, 512 if absent), `E55_CHECK_RUNS` (for example `unrelated:B,conflict:S`), `E55_CHECK_VALIDATE` (0 leaves out the validation), `E55_CHECK_VARIANTS` and `E55_CHECK_SAVE`. The table of the emulation below was made with `E55_CHECK_UPDATES=4096`, one run a process; a run of 4,096 updates takes about 21 minutes on one thread. The part `positions` reads the weights that `emulate` writes for `conflict` under back-propagation when `E55_CHECK_SAVE` names a folder; that file was written to the checker's scratch folder and deleted after use.

## The arithmetic of the construction, with the measured figures

This section answers the questions put under point 1 of the request. Figures are given as sweep; back-propagation.

**What the network does on an item made of bytes in use, at the start `shut`** (part `start`; the 1,200 held-out pairs of the first family at 12 letters, 22 positions an item).

| Quantity | Figure |
|---|---|
| What an update gate of the reserve receives from the input map, at every position | 10.0000 to 10.0000; the remainder of the projection is 1.9e-7; 1.3e-7 |
| What it receives from the recurrent matrix | nothing from the old cells and from the bias (entries of at most 2.5e-16); from the cells in reserve the weights are the draws (sum of magnitudes of a row 3.6 to 4.5) times states of at most 5.9e-4, at most 0.003 |
| 1 − z of a cell in reserve | 4.540e-5 in the mean, 4.541e-5 at most (4.542e-5 in single precision) |
| What the reset rows and the candidate rows of a cell in reserve receive | the full input: these rows are not projected and are the draws; the reset gate lies between 0.10 and 0.90, the candidate has a mean magnitude of 0.25; 0.29 and a largest magnitude of 0.96; 0.98 |
| The state of a cell in reserve | at most 5.6e-4; 5.9e-4 (the bound is 22 positions times 4.54e-5, which is 1.0e-3) |
| What reaches the old cells and the old readout units | the states above through weights that are the draws and are free: 384 rows by 128 columns of the recurrent matrix and 512 rows by 128 columns of `lm` |
| An added readout unit at an answer position | at most 2.9e-4; 3.6e-4 (nothing from `l2`, and from `lm` only what the cells in reserve send) |
| Largest difference of an output at an answer position from the keyed start under the keyed forward | 2.4e-3; 2.5e-3 on the held-out pairs and 8.4e-4; 8.1e-4 on the 2,000 items of check 1 |
| Generated answers of the 32,884 that differ from the keyed start's | 1; 0 |

**The limits of check 1 can be met.** (b): 1.9e-7; 1.3e-7 at the input map, 2.3e-9; 2.9e-9 at `l2` and below 4e-16 at the two other blocks, against a limit of 1e-5. (c): 4.541e-5 against 1e-4; 5.6e-4; 5.9e-4 and 2.9e-4; 3.6e-4 against 0.01; 2.4e-3; 2.5e-3 against 0.02. (d): 1; 0 against 5.

**What can change during learning on the new task's items.** The gate rows of the reserve can change only along the part of a code that lies outside the 46 directions in use (input map) and in the columns of the cells in reserve (recurrent matrix). The reset and candidate rows of the cells in reserve, every weight from a cell in reserve to an old cell or to a readout unit, the columns 256 to 511 of `l3` and the rows of the byte table of the bytes not in use are free. On an item of the first family the gate therefore keeps its 10; what changes there is the candidate of a cell in reserve (which is bounded by 1, so that the state stays below 22 times 1 − z) and the weights that multiply that state (which are not bounded).

**Is the claim "no first-family item can open them while learning on other items can" right?** The first half holds as long as the weights of the gate rows from the cells in reserve stay small: a gate of the reserve receives, beside the 10, these 128 weights times states of at most 1e-3. At the start the sum of their magnitudes is 3.6 to 4.5, which gives at most 0.003. The construction sets no bound: steps of 0.001 a weight for 4,096 updates could bring the sum to 530, and 530 times 1e-3 would no longer leave the gate shut. In the emulation the largest 1 − z on the first family's held-out pairs stays between 4.5e-5 and 5.5e-5 at every row of every run. The second half is right for `unrelated` and is right for `conflict` only at the item's first byte:

- `unrelated`: every byte but the start symbol is outside the bytes in use. The part of a byte's code outside the directions in use has a 2-norm of 0.36 to 1.95; 0.52 to 2.23 (of 8.1) and a sum of magnitudes of 5.3 to 28.6; 8.7 to 32.6. To move a gate's summed input from 10 to 2 on one byte, steps of at most 0.001 a weight need at least 280 to 1,507; 245 to 915 updates in which every weight moves in the right direction.
- `conflict`: only the first byte (`r`, `t`, `x` or `n`) is outside the bytes in use; the part of its code outside the directions has a 2-norm of 0.60 to 1.28; 0.84 to 1.73, and the same count gives at least 445 to 910; 309 to 580 updates. At every later position the gate receives 10 from the input map and nothing from the old cells (the rule keeps those weights at zero). It can be held open there only by the weights of the gate rows from the cells in reserve (rows 384 to 511, columns 128 to 255 of the recurrent matrix), acting on the states that cells in reserve took at the first byte; these weights times the states must sum to about −8. A cell that opens at the first byte alone holds a function of the operation byte for the rest of the item and cannot read an operand. The brief does not say this (defect 8).

**Can the gates open by gradient learning from 1 − z = 4.5e-5?** Yes, as far as the record and the emulation go, and slowly.

- *E51's arm `open`.* With free rows the gates opened on `unrelated` under both rules: in the receipts the largest mean opening of a cell rises from 1e-4 at row 0 to 0.13 at row 256 (sweep) and to 0.15 at row 128 (back-propagation); in the weights of row 4,096, 127; 125 of the 128 cells in reserve open by more than 0.1 at some position of an average never-taught item and 102; 75 by more than 0.5 (part `open`; defect 2).
- *The gradient at the start `shut`* (part `grad`). The error at a gate's summed input carries the factor z (1 − z) = 4.5e-5 under both rules: its median magnitude is 3.6e-10; 2.9e-9 on `unrelated` and 1.2e-10; 2.6e-9 on `conflict`. After the rule's projection the entries of the gradient of the gate rows of the input map have a median magnitude of 3.8e-9; 4.6e-8 on `unrelated` and 2.0e-10; 5.6e-9 on `conflict`, against the optimizer's eps of 1e-8. The first step of AdamW is the rate times g / (|g| + eps): its median size at these rows is 0.28; 0.82 of the rate on `unrelated` and 0.02; 0.36 on `conflict`. Under the sweep on `conflict` fewer than 0.5% of the entries of the projected gradient exceed 1e-8, and the largest is 2.5e-8. At the gate rows of the recurrent matrix the projected gradient is at most 1.5e-10; 2.2e-9, because it is an error of that size times a state of 1e-3: these weights, the only ones that can hold a gate open on `conflict` after the first byte, hardly move until a cell in reserve has a large state (defect 7).
- *The emulation of the arm `shut`* (table below). On `unrelated` the largest 1 − z on the new task's items rises from 5.5e-5 to 1.2e-2; 5.0e-3 at row 512 and to 0.14; 0.87 at row 1,024. On `conflict` it is 9.2e-5; 6.5e-3 at row 512 and (back-propagation) 0.23 at row 1,024.

**Is there a path to the first family's answers that the brief does not name?** No path was found that the brief's sentence "it does not bound what the cells in reserve send to the old cells and to the readout" does not cover. What the brief does not state is the size and the growth of that path, and its check 2 (b) assumes that it is below 1e-3 (defect 1). The weights from a cell in reserve receive a gradient that is an error times a state of 1e-3 (median 5.7e-8; 5.2e-7 at the old rows of the recurrent matrix, far above eps), and AdamW moves a weight by up to 0.001 an update whatever the size of its gradient. After g updates an old cell or a readout unit can receive up to 128 × 0.001 g × 1e-3 from the cells in reserve, which is 0.13 for every 1,000 updates. After eight emulated updates the first family's outputs have moved by 0.007 to 0.026, and the largest added readout unit at an answer position grows from 3e-4 to 9e-3 in 64 updates.

**Is there a path by which the new task is learned without the gates opening?** Two. The old cells learn through the part of the new bytes' codes outside the directions in use, as in E51's arm `keyed`, where the old cells alone answered 41% of `conflict`; in the emulation of `shut` on `conflict` the never-taught score rises from 19.0; 8.2 to 35.4; 26.5 at row 512 with every gate still shut, and it is the same with the cells in reserve held shut (35.2; 23.1). And a shut cell is not without effect: its state of about 1e-3 is an input whose outgoing weights are free. The second scoring sets that state to exactly zero, so that a lower score with the cells held shut does not by itself show that a gate opened. The brief keeps the second scoring as a figure, but the sentences it licenses use neither it nor a count of opened gates (defect 4).

**The emulation of the arm `shut`, seed 73129** (part `emulate`; see the caution above). The second count of the fifth column is the number of cells in reserve for which the mean over the never-taught items of the largest 1 − z over the positions of the item exceeds 0.5; such a cell is called *open by more than 0.5 at some position*. The answers that differ are counted against row 0 of the same emulation; the 31,684 answers without letters were scored at rows 0 and 256 to 4,096.

| Run | Row | Loss of the last update | Never-taught score; with the cells in reserve held shut | Cells in reserve open on the new task's items: mean of 1 − z above 0.1; largest 1 − z of an item above 0.5 in the mean | Largest 1 − z on the new task's items | First family, held-out pairs: largest 1 − z; largest added readout unit | Answers that differ from row 0's: of 1,200 held-out; of 31,684 without letters | Reduction score |
|---|---|---|---|---|---|---|---|---|
| `unrelated`, back-propagation | 0 | – | 0.0; 0.0 | 0; 0 | 5.6e-05 | 4.54e-05; 3.6e-04 | 0; 0 | 95.25 |
| `unrelated`, back-propagation | 64 | 6.16 | 0.0; 0.0 | 0; 0 | 1.6e-04 | 4.57e-05; 9.3e-03 | 1; not counted | not counted |
| `unrelated`, back-propagation | 256 | 2.83 | 0.0; 0.0 | 0; 0 | 2.4e-03 | 4.56e-05; 8.2e-03 | 1; 28 | 95.30 |
| `unrelated`, back-propagation | 512 | 2.78 | 0.0; 0.0 | 0; 0 | 5.0e-03 | 4.56e-05; 9.4e-03 | 0; 29 | 95.33 |
| `unrelated`, back-propagation | 1,024 | 2.53 | 0.0; 0.0 | 0; 0 | 0.87 | 4.63e-05; 4.4e-02 | 0; 34 | 95.31 |
| `unrelated`, back-propagation | 2,048 | 0.19 | 72.3; 0.0 | 17; 76 | 1.00 | 4.65e-05; 4.5e-02 | 1; 24 | 95.31 |
| `unrelated`, back-propagation | 4,096 | 0.00 | 99.3; 0.0 | 21; 93 | 1.00 | 4.63e-05; 4.2e-02 | 1; 20 | 95.29 |
| `unrelated`, sweep | 0 | – | 0.0; 0.0 | 0; 0 | 5.5e-05 | 4.54e-05; 2.8e-04 | 0; 0 | 92.40 |
| `unrelated`, sweep | 64 | 7.38 | 0.0; 0.0 | 0; 0 | 1.1e-04 | 4.57e-05; 8.6e-03 | 2; not counted | not counted |
| `unrelated`, sweep | 256 | 3.22 | 0.0; 0.0 | 0; 0 | 2.0e-03 | 4.67e-05; 2.9e-02 | 5; 87 | 92.57 |
| `unrelated`, sweep | 512 | 2.82 | 0.0; 0.0 | 0; 0 | 1.2e-02 | 4.81e-05; 3.9e-02 | 3; 104 | 92.40 |
| `unrelated`, sweep | 1,024 | 2.77 | 0.0; 0.0 | 0; 0 | 0.14 | 5.00e-05; 4.2e-02 | 7; 70 | 92.39 |
| `unrelated`, sweep | 2,048 | 2.50 | 0.0; 0.0 | 0; 3 | 1.00 | 5.14e-05; 5.8e-02 | 6; 99 | 92.63 |
| `unrelated`, sweep | 4,096 | 1.74 | 0.0; 0.0 | 128; 128 | 1.00 | 5.12e-05; 5.1e-02 | 9; 90 | 92.79 |
| `conflict`, back-propagation | 0 | – | 8.2; 8.2 | 0; 0 | 4.9e-05 | 4.54e-05; 3.6e-04 | 0; 0 | 95.25 |
| `conflict`, back-propagation | 64 | 1.74 | 16.9; 16.4 | 0; 0 | 1.2e-04 | 4.56e-05; 9.2e-03 | 11; not counted | not counted |
| `conflict`, back-propagation | 256 | 1.13 | 25.2; 22.4 | 0; 0 | 2.8e-03 | 4.65e-05; 2.9e-02 | 30; 244 | 96.18 |
| `conflict`, back-propagation | 512 | 0.86 | 26.5; 23.1 | 0; 0 | 6.5e-03 | 4.66e-05; 2.4e-02 | 25; 115 | 95.70 |
| `conflict`, back-propagation | 1,024 | 0.65 | 23.1; 25.4 | 0; 0 | 0.23 | 4.76e-05; 1.3e-02 | 11; 39 | 95.38 |
| `conflict`, back-propagation | 2,048 | 0.30 | 33.0; 27.9 | 0; 0 | 0.62 | 4.96e-05; 9.4e-03 | 5; 23 | 95.33 |
| `conflict`, back-propagation | 4,096 | 0.20 | 54.1; 29.5 | 0; 0 | 0.99 | 4.89e-05; 5.1e-03 | 1; 13 | 95.26 |
| `conflict`, sweep | 0 | – | 19.0; 19.0 | 0; 0 | 4.9e-05 | 4.54e-05; 2.8e-04 | 0; 0 | 92.40 |
| `conflict`, sweep | 64 | 1.15 | 26.4; 25.2 | 0; 0 | 6.0e-05 | 4.55e-05; 8.8e-03 | 4; not counted | not counted |
| `conflict`, sweep | 256 | 0.84 | 35.9; 34.1 | 0; 0 | 1.2e-04 | 4.61e-05; 2.6e-02 | 28; 376 | 92.56 |
| `conflict`, sweep | 512 | 0.67 | 35.4; 35.2 | 0; 0 | 9.2e-05 | 4.63e-05; 3.1e-02 | 35; 478 | 92.72 |
| `conflict`, sweep | 1,024 | 0.81 | 34.6; 34.0 | 0; 0 | 8.3e-05 | 4.65e-05; 3.3e-02 | 37; 505 | 92.96 |
| `conflict`, sweep | 2,048 | 0.61 | 34.1; 32.9 | 0; 0 | 9.1e-05 | 5.05e-05; 4.4e-02 | 62; 743 | 93.22 |
| `conflict`, sweep | 4,096 | 0.98 | 31.0; 30.0 | 0; 0 | 8.0e-05 | 5.53e-05; 6.3e-02 | 76; 924 | 93.64 |

Read with the caution above, the emulation says four things.

1. *On the first family's items the gates stay shut.* In all four runs 1 − z of a cell in reserve is at most 5.5e-5 and its state at most 1.1e-3 on the held-out pairs, at every row.
2. *The first family's answers change all the same,* from row 64 on, through the weights from the cells in reserve, which grow while the states stay at 1e-3. Of the 31,684 answers without letters, 20 to 34 differ from row 0's at rows 256 to 4,096 on `unrelated` under back-propagation, 70 to 104 under the sweep, 13 to 244 on `conflict` under back-propagation and 376 to 924 under the sweep (2.9% at row 4,096, with 76 of the 1,200 held-out answers). The reduction score stays within 1.3 points of row 0's. By the brief's words every one of the four runs is *partly kept*.
3. *On `unrelated` the gates open under both rules,* between rows 1,024 and 4,096. Under back-propagation the task is then learned (72.3 at row 2,048, 99.3 at row 4,096, 0.0 with the cells in reserve held shut), with 93 cells open by more than 0.5 at some position and 21 by the brief's count. Under the sweep all 128 cells are open at row 4,096 and the never-taught score is 0.0.
4. *On `conflict` the gates open on a part of the items under back-propagation and not at all under the sweep.* Under back-propagation the largest 1 − z reaches 0.99, and the never-taught score rises to 54.1 at row 4,096 (29.5 with the cells held shut). No cell passes 0.5 in the mean over the items of the largest 1 − z of an item, and 15 cells pass 0.1 (6 at row 2,048; part `emulate` run again for this count). These 15 open at the first byte (800 pairs of an item and a cell have 1 − z above 0.5 there) and are open by about 0.01 at every later position, 200 times the start's 4.5e-5 (part `positions`): the holding open through the cells' own states, which the section above describes, has begun and is far from complete. Under the sweep the largest 1 − z stays below 1.2e-4 and the score stays at what the old cells give (31.0; 30.0 with the cells held shut).

## Defects, the most serious first

Each defect gives the place in the brief, what is wrong, the evidence and a replacement text. Line numbers of `exp51_reserve.py` are those of the file with the SHA-256 that the brief names.

### 1. Check 2 (b) cannot be met, and the movement of the first family's outputs that it limits is the main effect of the construction

**Place.** "Checks before the runs", 2 (b): "In the arm `shut`: on the two panels the largest of 1 − z of a cell in reserve is at most 1e-4, and the largest change of an output at an answer position from the start's is recorded and is at most 1e-3." Also "The change", last paragraph: "How small the effect on the first family is, is the first thing the job measures"; and the predictions P55.1 and P55.4.

**What is wrong.** After eight updates of the arm `shut` the first family's outputs have moved by 7 to 26 times the limit on the held-out pairs and by 2 to 15 times the limit on the 2,000 items. The worker would stop at check 2 and ask, before any run. The cause is the path that the brief names and does not size: the weights from the cells in reserve are free, the gradient of such a weight is an error times a state of about 1e-3, and AdamW moves a weight by up to 0.001 an update whatever the size of its gradient.

**Evidence** (part `check2`, eight emulated updates from the start `shut`; sweep; back-propagation).

| Task | Largest change of an output, held-out pairs | Largest change, 2,000 items | Largest 1 − z on the two panels | Largest state of a cell in reserve |
|---|---|---|---|---|
| `unrelated` | 6.8e-3; 2.1e-2 | 4.7e-3; 1.5e-2 | 4.542e-5; 4.541e-5 | 7.5e-4; 7.8e-4 |
| `conflict` | 2.6e-2; 1.5e-2 | 5.2e-3; 2.3e-3 | 4.542e-5; 4.543e-5 | 8.0e-4; 8.1e-4 |

In the longer emulation (table above) answers of the first family differ from row 0's from row 64 on, and the number that differ among the 31,684 answers without letters is above the brief's limit of 31 for *kept* at some row of every emulated run.

**Replacement** for check 2 (b): "(b) In the arm `shut`, on the two panels: the largest of 1 − z of a cell in reserve lies between 4.4e-5 and 1e-4, and the largest magnitude of a state of a cell in reserve is at most 1.1e-3 (22 positions times 1 − z). The largest change of an output at an answer position from the start's and the largest magnitude of an added readout unit at the answer positions are recorded and have no limit: the weights from the cells in reserve to the old cells and to the readout are free, and the optimizer moves each of them by up to 0.001 an update. The check of this brief found, in an emulation, a change of an output of 0.007 to 0.026 after eight updates."

**Replacement** for the last two sentences of "What follows from the construction": "The construction does not make them zero, and it does not bound what the cells in reserve send to the old cells and to the readout, whose weights from cells in reserve are free. The gradient of such a weight is an error times a state of about 1e-3, and AdamW moves a weight by up to 0.001 an update whatever the size of its gradient, so that after g updates an old cell or a readout unit can receive up to 128 × 0.001 g × 1e-3 from the cells in reserve. In an emulation made for the check of this brief the first family's outputs moved by 0.007 to 0.026 in eight updates, and between 13 and 924 of the 31,684 answers without letters differed from row 0's at rows 256 to 4,096 of four emulated runs. How far this goes in 4,096 updates of the runner is the first thing the job measures."

**For the author to decide before the release.** P55.4 (the first family kept exactly or kept in all four runs, 70%) and P55.1 (the checks pass at the first attempt, 65%) were set without these figures and should be set again. If the author prefers to reduce the state of a shut cell, a larger bias does it at a price: 1 − z and the gradient at the gate fall together by the same factor, the gradient at the gate rows is already near eps at a bias of 10 (defect 7), and in single precision the sigmoid of a summed input above 16.6 is exactly 1, so that the gate could not be opened at all. This check did not test another bias.

### 2. The record says that the gates did not open in E51's arm `open`; on `unrelated` they opened, and the task was not learned in the old cells

**Place.** "What the record says", third bullet: "**In the arm `open` the gates did not open.** The new task was learned there with at most 8 of the 128 cells in reserve open, in the old cells, whose weights were free; the first family was lost. Whether learning opens gates that start shut, when the old cells' weights are protected, is not on record."

**What is wrong.** The count "1; 8 and 0; 0 of 128" is the number of cells whose mean of 1 − z over all positions and items exceeds 0.1 (`reserve_figures`, lines 387 to 399). A cell that opens at one position of an item of 19 positions has a mean of at most 0.053 and is not counted. On `unrelated` most cells in reserve open at the position of a symbol, and the task needs them. For `conflict` the sentence is nearly right.

**Evidence.** Part `open`, weights of row 4,096 of the arm `open`; part `record` for the receipts.

| Run of `open` | Cells with mean of 1 − z above 0.1 (the receipts' count) | Cells for which the mean over items of the largest 1 − z over the positions of the item exceeds 0.1; exceeds 0.5 | Cells whose mean over items exceeds 0.5 at one position, position by position | Never-taught score; the same with the cells in reserve held shut (receipts) |
|---|---|---|---|---|
| `unrelated`, sweep | 1 | 127; 102 | 57 at the first symbol, 10 at the second, 5 at the code byte | 100.0; 0.7 |
| `unrelated`, back-propagation | 8 | 125; 75 | 38 at the first symbol, 9 at the second, 11 at the query byte | 100.0; 5.7 |
| `conflict`, sweep | 0 | 2; 0 | none | 85.0; 83.0 |
| `conflict`, back-propagation | 0 | 16; 0 | none | 94.9; 87.6 |

The receipts agree: at row 4,096 of the four runs the largest state of a cell in reserve on the first family's held-out pairs is 1.00; 1.00 (`unrelated`) and 0.71; 1.00 (`conflict`), where it was 0.0015; 0.0024 at row 0. The verification says of the same runs that the criterion "learned in the reserve" gives "yes" for `open` on the unrelated task (0.7 and 5.7 with the reserve cells shut, its section 8.2).

**Replacement.** "**In the arm `open` the gates opened on `unrelated` and not on `conflict`.** The receipts' count, the number of cells whose mean of 1 − z over all positions and items exceeds 0.1, is 1; 8 on `unrelated` and 0; 0 on `conflict`. It does not count a cell that opens at one position. In the weights of row 4,096 on `unrelated`, 127; 125 of the 128 cells in reserve open by more than 0.1 at some position of an average never-taught item and 102; 75 by more than 0.5, most of them at the position of the first symbol, and with the cells in reserve held shut the never-taught score is 0.7; 5.7 against 100; 100. On `conflict` 2; 16 cells open by more than 0.1 at some position and none by more than 0.5, and the score with the cells held shut is 83.0; 87.6 against 85.0; 94.9: that task was learned mainly in the old cells, whose weights were free. The first family was lost on both tasks. The record therefore shows that learning opens gates that start shut by a bias of 10 when their rows and all other weights are free (the largest mean opening of a cell rises from 1e-4 at row 0 to 0.13 at row 256 under the sweep and to 0.15 at row 128 under back-propagation). It does not show it for gate rows that may change only away from the directions in use, nor when the old cells' weights are protected."

### 3. The count of cells "open by more than 0.1" cannot see what the construction is built to produce

**Place.** "The runner", point 3 (`gates_on`: "the number of cells whose mean of 1 − z over positions and items exceeds 0.1"); "How the result is stated", first paragraph ("the number of cells in reserve open by more than 0.1 on the new task's items") and the third bullet ("with the number of cells open on the new task's items"); P55.5 and P55.6.

**What is wrong.** The figure for the new task's items is E51's `reserve_figures`, a mean over all positions. In E51's arm `keyed` the gates started at 0.5 and the mean was informative. Here every gate starts at 4.5e-5 and can open only where an item presents something outside the directions in use: on `conflict` at one position of 22 (a mean of at most 0.045), on `unrelated` at the positions that learning selects. The count would stay near 0 while the cells are in use. `gates_on` has one sensitive figure (`largest_opening`), but it is applied to the first family's items only.

**Evidence.** Defect 2 (1 and 8 by the mean where 127 and 125 cells are open by more than 0.1 at some position). In the emulation of `shut` on `unrelated` under back-propagation, 17 cells have a mean above 0.1 at row 2,048 and 21 at row 4,096, where 76 and 93 cells are open by more than 0.5 at some position and the never-taught score is 72.3 and 99.3. P55.6 asks for 32 by the mean and would be scored false in the one emulated run that learns its task. In the emulated run of `conflict` under back-propagation at row 4,096 the brief's count is 0 and no cell passes 0.5, while 15 cells are open by more than 0.1 at some position and the score falls from 54.1 to 29.5 when the cells are held shut; for this reason the replacement below sets the mark at 0.1 and records 0.5 beside it.

**Replacement** for point 3 of "The runner": "

    @T.no_grad()
    def gates_on(m, items):
        """Teacher forcing, the plain forward. Over the cells in reserve: the largest of 1 - z over cells, positions and items; the number of
        cells whose mean of 1 - z over positions and items exceeds 0.1; the number of cells for which the mean over items of the largest
        1 - z over the positions of the item exceeds 0.1, and the same number for 0.5; the largest magnitude of a state; and the largest
        magnitude of an added readout unit at the answer positions."""

It groups the items by length as `R51.reserve_figures` does, takes `z`, `h` and `z2` from the second value that `R51.fwd(m, x, False, True)` returns, and returns the six figures as `largest_opening`, `cells_open_by_more_than_0p1`, `cells_open_at_some_position`, `cells_open_by_more_than_0p5_at_some_position`, `largest_reserve_state` and `largest_added_unit_at_answer_positions`." In "The run": "In `score_row`, after the entry `reserve`: `r["gates_on_first_family"] = gates_on(m, s2_items)` and `r["gates_on_new_task"] = gates_on(m, reserve_panel)`." In "Terms": "A cell in reserve is *open at some position* on a set of items if the mean over the items of the largest 1 − z over the positions of the item exceeds 0.1." In "How the result is stated", first paragraph: "the number of cells in reserve open at some position on the new task's never-taught items, the number open by more than 0.1 in the mean, and the largest 1 − z there". P55.6: "In the arm `shut`, every run in which the task is learned has at least 32 cells in reserve open at some position on the new task's items at row 4,096." P55.5: "In the arm `shut`, the largest 1 − z of a cell in reserve on the first family's held-out items is below 1e-3 at every scored row of every run."

### 4. Five sentences fixed in "How the result is stated" can be licensed when the runs do not support them

**(a) Place.** "If the task is not learned: 'learning did not open the cells in reserve within 4,096 updates', with the number of cells open on the new task's items".

**What is wrong.** The sentence is licensed by the never-taught score alone and states something about the gates. In the emulation of `unrelated` under the sweep all 128 cells in reserve are open at row 4,096, by the brief's count and by the count of defect 3, and the never-taught score is 0.0: the brief's sentence would be reported and would be false.

**Replacement.** "If the task is not learned: 'the task was not learned within 4,096 updates', with the never-taught score at row 4,096 and, for every scored row, the largest 1 − z and the number of cells open at some position on the new task's never-taught items. If no cell is open at some position at any row, the report adds 'learning did not open the cells in reserve'. If cells opened, the report gives their number and says that the task was not learned although they opened. The first family's word is then given and is not read as a success of the construction."

**(b) Place.** "*kept* if it is not kept exactly, d(g) is at most 31 at every row (0.1% of the answers) and the reduction score at every row is within 0.5 points of row 0's".

**What is wrong.** d(g) counts the answers without letters only. The state of a shut cell grows with the number of positions (22 for a held-out pair at 12 letters, 10 without letters), and the held-out answers are where it acts first: in the emulation 4; 11 of the 1,200 held-out answers of `conflict` differ at row 64 and 28; 30 at row 256 (2.3%; 2.5%). A run could be called *kept* with several per cent of the held-out answers changed.

**Replacement.** "*kept* if it is not kept exactly, d(g) is at most 31 at every row, at most 1 of the 1,200 held-out answers differs from row 0's at every row (0.1% of each set), and the reduction score at every row is within 0.5 points of row 0's".

**(c) Place.** "If the task is learned and the first family is kept exactly or kept: 'the first family was kept while the task was learned, with the forward pass of the plain network; the separation was carried by weights'."

**What is wrong.** The sentence is read as a success of gates opened by learning, which is the title and the question of the brief, and it does not ask whether a gate opened. A task can be learned in part without it (41% of `conflict` in E51's `keyed` with the cells held shut; 35.4; 26.5 in the emulation at row 512 with every gate shut).

**Replacement.** "If the task is learned and the first family is kept exactly or kept: 'the first family was kept while the task was learned, with the forward pass of the plain network', with the number of cells open at some position on the new task's items and the score with the cells in reserve held shut. If no cell is open at some position, or that score is at 80 or more, the report adds 'the task was learned without the cells in reserve opening', and the run is not read as a success of the gates."

**(d) Place.** "If, for a rule and task, the first family is kept exactly or kept in both arms, the report says that the protection of the gate rows was not needed in that run."

**What is wrong.** If the task is learned in neither arm, or the gates of `shut-gates-free` did not open on the new task's items, nothing was protected against and the sentence is not supported. The two arms also share the start shut by a bias and the protected rows 256 to 511 of `lm` and `l2`, which E51's `keyed` had free and needed for `unrelated` (0.0; 0.0 with the added readout put back, section 8.2 of the verification). If neither arm learns, the two arms do not say which of the three shared conditions is the cause.

**Replacement.** "If, for a rule and task, the task is learned in both arms and the first family is kept exactly or kept in both, the report says that the protection of the gate rows was not needed in that run. If the first family is kept in `shut` and partly kept or lost in `shut-gates-free`, the report says that the protection of the gate rows was needed, with the figures of `gates_on`. If the task is learned in neither arm, the arms are not compared, and the report says that the cause is not identified by this job: both arms share the start shut by a bias, the protected rows 256 to 511 of `lm` and `l2`, and the 4,096 updates."

**(e) Place.** "If the task is learned and the first family is partly kept or lost: 'the weights did not carry the separation', with the row at which d(g) first exceeds 31".

**What is wrong.** *Partly kept* covers every run between 32 changed answers at one row and a reduction score just above half of row 0's, and one negative sentence is licensed for all of them. The state of a shut cell, through the weights that multiply it, puts the runs at the lower end of that range. In the emulation of `unrelated` under back-propagation the task is learned at row 4,096 with 93 cells open by more than 0.5 at some position; d(g) is 28, 29, 34, 24 and 20 at rows 256 to 4,096, at most one held-out answer differs from row 256 on, and the reduction score stays within 0.08 points of row 0's. That run would be reported as "the weights did not carry the separation" because of three answers at one row.

**Replacement.** "If the task is learned and the first family is partly kept, the report gives the largest d(g) over the rows as a number and as a percentage of 31,684, the largest number of held-out answers that differ, and the reduction score at row 4,096 beside row 0's, and says 'the first family was kept except for x% of its answers'. If the first family is lost, the report says 'the first family was not kept'. In both cases it gives the row at which d(g) first exceeds 31 and the figures of `gates_on` at that row, which show whether the gates opened on the first family's items or stayed shut while the weights from the cells in reserve grew."

### 5. The number of updates is carried over from E51, where the gates started half open

**Place.** "The change": "Everything is E51's arm `keyed` (the births, the embedded network, the tasks and their streams, the optimizer, the number of updates, the scored rows, the scores) but three things." "Arms and runs": "eight runs of 4,096 updates". P55.8.

**What is wrong.** In E51's `keyed` the gates of the cells in reserve were the draws (1 − z about 0.5 on the new task's items at row 0), and the tasks were learned from row 512 or 1,024. Here every gate must first be moved from a summed input of 10. At steps of at most 0.001 a weight this takes at least 245 to 1,507 updates for one byte (section "The arithmetic of the construction"), and in the emulation it takes 1,000 to 4,000. The run ends while the outcome is still changing, and "not learned" would be read where the cause is the number of updates. Rule 1 of section 4 of the plan asks that a setting that was carried over be varied before a conclusion rests on it.

**Evidence** (the emulation, table above). `unrelated` under back-propagation: the largest 1 − z on the new task's items is 5.0e-3 at row 512 and 0.87 at row 1,024; the never-taught score is 0.0 at row 1,024, 72.3 at row 2,048 and 99.3 at row 4,096, so that the task is *learned* from the last row only. `unrelated` under the sweep: three cells are open by more than 0.5 at some position at row 2,048 and all 128 at row 4,096, with a score of 0.0 and a loss that still falls (2.50 at row 2,048, 1.74 at row 4,096). `conflict` under back-propagation: the score is 33.0 at row 2,048 and 54.1 at row 4,096.

**Replacement.** In "Arms and runs": "Two arms, two tasks and two rules with seed 73129 give eight runs of 16,384 updates, each network under the rule of its birth. The scored rows are E51's ten and the rows 8,192 and 16,384; the weights of rows 64, 4,096 and 16,384 are committed. The number of updates is four times E51's because the gates must first be opened, which E51's `keyed` did not need." In "The runner", point 4: "`updates=4 * R51.DOSE`; the rows are `R51.ROWS + (8192, 16384)`; the weights of the rows `R51.KEEP_ROWS + (16384,)` are kept; the tag carries `_dry<N>` for N other than 16,384." The costs are then about four times those stated (about 0.5 CPU-hours a run by E51's 0.09 to 0.12 CPU-seconds an update), and the limit of check 3 and the times in "Rules for the worker" are set again. If the author keeps 4,096 updates, the replacement of defect 4 (a) and this sentence in "What it cannot show" are the least that is needed: "The 4,096 updates are E51's. There the gates started half open; here they start shut, and in an emulation made for the check of this brief they began to open between rows 1,024 and 4,096. A task that is not learned at row 4,096 may be learned later."

### 6. Under the sweep, the error is sent back through the signs of rounding remainders

**Place.** "The runner", point 1 (`shut_start`) and point 2 (`Rule55`); "The change", point 2: "an update gate of the reserve receives 10 from the input map and nothing from the recurrent matrix, and an added readout unit receives ... nothing from `lm`."

**What is wrong.** The projection brings two blocks to zero: the weights from the old cells into the gate rows of the reserve (recurrent matrix, rows 384 to 511, columns 0 to 127) and into rows 256 to 511 of `lm`. In single precision they are not zero but remainders of 1e-16 with either sign, and the rule's second addition writes new remainders at every update. The forward pass does not see them. The sweep does: it sends the error back through a map by the sign of each weight times the mean magnitude of the whole matrix (`exp3_state.SignLinear`), and the sign of zero is zero. Under rule S the old cells therefore receive, from the 256 added readout units, an error through 32,768 signs of rounding remainders, at the same magnitude as the error through the birth's weights.

**Evidence.** Part `start`: of 16,384 entries of the first block 471; 446 are exactly zero and the rest are positive and negative in equal shares (largest magnitude 2.5e-16); of 32,768 entries of the second block 955; 963 are exactly zero. Part `residue`, rule S, after eight emulated updates, on the next batch: the loss is the same to six decimal places with the entries as stored and with the entries set to zero; the error arriving at the states of the old cells differs by 80% of its 2-norm on `unrelated` and by 37% on `conflict`; the gradient of the old rows of the input map after the rule's projection differs by 98% and by 26% of its 2-norm (cosine between the two 0.72 and 0.97). Part `emulate` with `E55_CHECK_VARIANTS=zeroed`, one run of `unrelated` under the sweep with the two blocks set to exactly zero after every step, beside the run of the table: at row 4,096, 2 cells in reserve are open by more than 0.5 at some position where the table's run has 128, the loss is 1.85 against 1.74, and the never-taught score is 0.3 against 0.0. Neither run learns the task, and the two take different courses, so that the signs of the remainders take part in what a run under the sweep gives.

**Replacement.** In `shut_start`, after the loop: "`m.rnn.weight_hh_l0[Z_RES, :R51.U0] = 0.0; m.lm.weight[ADDED, :R51.U0] = 0.0`   # the projection leaves remainders of 1e-16 here; the sweep sends the error back by the sign of a weight". In `Rule55`: "

        @T.no_grad()
        def after(self):
            super().after()
            self.m.lm.weight[ADDED, :R51.U0] = 0.0
            if self.arm == "shut":
                self.m.rnn.weight_hh_l0[Z_RES, :R51.U0] = 0.0

with `self.arm = arm` in `__init__`." In check 1 (a) and check 2 (a): "the two blocks that the projection brings to zero (rows 384 to 511, columns 0 to 127 of `rnn.weight_hh_l0`; rows 256 to 511, columns 0 to 127 of `lm.weight`) are exactly zero" (in check 2 (a) the first block in the arm `shut` only).

### 7. Whether the optimizer acts on the gate rows of the arm `shut` is not checked, and under the sweep on `conflict` the gradient there is below eps

**Place.** Check 2 (c): "in the arm `shut-gates-free` the largest magnitude of an entry of `(D_after − D_start) @ Vm` at the gate rows of the reserve ... exceeds 1e-4. If it does not, the worker reports the figure and asks, because the gradient that reaches a shut gate may be too small for the optimizer to act on, and the arm would then not differ from `shut`."

**What is wrong.** The check asks the question for the arm whose gate rows are free. There the gradient is 12 to 130 times larger (in the median) than in the arm `shut`, where it is projected away from the directions in use, and the check passes with a wide margin (0.10 to 0.14 at the input map against 1e-4, part `check2`). For the arm `shut`, on which the main reading rests, nothing is checked or recorded. The outcome "not learned" under the sweep would depend on the optimizer's eps, a constant that has nothing to do with the question.

**Evidence** (part `grad`, first batch, start `shut`; the gradient of the gate rows of the reserve in the input map after the rule's projection).

| Task, rule | Median magnitude of an entry | Largest | Median of g / (g + eps), the size of the first step as a share of the rate |
|---|---|---|---|
| `unrelated`, sweep | 3.8e-9 | 2.7e-7 | 0.28 |
| `unrelated`, back-propagation | 4.6e-8 | 3.3e-6 | 0.82 |
| `conflict`, sweep | 2.0e-10 | 2.5e-8 | 0.02 |
| `conflict`, back-propagation | 5.6e-9 | 3.4e-7 | 0.36 |

In the emulation the largest 1 − z on the items of `conflict` under the sweep is 4.9e-5 at row 0 and 9.2e-5 at row 512.

**Replacement.** Add to check 2: "(d) In the arm `shut`: what an update gate of the reserve receives from the input map on the code of each byte of the new task (the 98 bytes of `unrelated`; `r`, `t`, `x`, `n` of `conflict`), after the eighth update minus at the start, for each of the 128 gates and each byte: the mean magnitude and the largest magnitude of this change are recorded. If the largest magnitude is below 0.01 (the check of this brief found 0.08 to 0.17 in an emulation), the worker reports the figure and asks. The gradient of these rows at the first update, after the rule's projection, is recorded with its median magnitude." Add to "What it cannot show": "The gradient that reaches a shut gate carries the factor z (1 − z) = 4.5e-5. At the gate rows of the arm `shut` it is of the size of the optimizer's eps (1e-8) and under the sweep on `conflict` below it, so that the optimizer's steps there are a fraction of the rate. A task that is not learned within 4,096 updates may be one that a smaller eps or more updates would learn."

### 8. For `conflict`, the brief does not say what can hold a gate open after the first byte

**Place.** "The question": "so that no first-family item can open them while learning on other items can"; "The change", point 3: "what it cannot do is open on an item that presents nothing outside the directions in use"; point 2: "The weights of these rows on other directions keep their draws: from the codes of bytes that are not in use, and from the cells in reserve."

**What is wrong.** An item of `conflict` presents nothing outside the directions in use at the input map after its first byte. The mechanism by which a cell in reserve could read an operand of such an item (a gate opened at the first byte and then held open through the gate rows' weights from the cells in reserve) is not stated, although P55.3 and the reading of a failure on `conflict` depend on it. The second quoted sentence also says more than is the case: on the codes of the bytes not in use the projected rows keep only what acts on the part of a code outside the directions in use, which is 1 to 2% of its squared size.

**Evidence.** Section "The arithmetic of the construction" above. Part `start`: on the new tasks' bytes the projected gate rows give at most 0.19; 0.26 (`unrelated`) and 0.14; 0.17 (`conflict`) beside the 10, where the rows of the keyed start gave up to 1.26; 1.37 and 0.94; 1.05; at the start `shut` the largest 1 − z on the never-taught items is 5.5e-5; 5.7e-5 and 4.9e-5; 4.9e-5. The gates are shut on the new tasks' items as on the first family's. Part `positions`: in the emulated run of `conflict` under back-propagation at row 4,096 the cells in reserve that have opened do so at the first byte, and at the later positions their 1 − z is about 0.01.

**Replacement** for the last sentence of point 3 of "The change": "The reset rows and the candidate rows of the cells in reserve stay free, so that a cell in reserve may read the old cells and the codes of the bytes in use through its candidate; what it cannot do is open on an item that presents nothing outside the directions in use. At the start the gates are shut on the new tasks' items as well (1 − z at most 5.7e-5). In `unrelated` every byte but the start symbol is outside the bytes in use, and a gate can be opened at any position through the part of the byte's code outside the directions in use (2-norm 0.4 to 2.2 of 8.1). In `conflict` this holds for the first byte only (2-norm 0.6 to 1.7). At every later position of an item of `conflict` a gate of the reserve receives 10 from the input map and nothing from the old cells; it can be held open there only by the weights of the gate rows from the cells in reserve (columns 128 to 255 of the recurrent matrix), acting on the states that the cells took at the first byte. A cell that opens at the first byte alone holds a function of the operation byte for the rest of the item." For the sentence of point 2: "On other directions these rows keep the values drawn when the embedded network was made: on the part of the code of a byte not in use that lies outside the directions in use, and on the states of the cells in reserve."

### 9. The runner: three lines of `exp51_reserve.py` and two instructions that the listed changes do not cover

**Place.** "The runner", point 4: "a copy of the text of `R51.run` (lines 452 to 549 of `exp51_reserve.py`) with the changes below and no others".

**What is wrong, with the evidence.**

- Line 528, in `write`: `dict(experiment="E51", kind="run", tag=tag, ...)`. The brief changes `experiment` in the settings (line 518) only, so that every receipt of E55 would carry `"experiment": "E51"` at its top level.
- Line 706, in the part of `exp51_reserve.py` that runs only when the file is called as a program: `T.set_num_threads(1)`. A runner that imports the file does not execute it, and the brief's rule "each with one thread" would rest on the worker's shell.
- Line 518: `start=start_rec`. With `load_shut` this is the record that `R51.load_start` returns, which names the file and the hashes of E51's keyed start. The receipt of a run would name the keyed start as its start and the start `shut` nowhere but in `start_tensors_sha256`.
- Point 1 writes "the hash of its tensors (`R.tensors_sha`)". In the brief `R` is a shell name for a folder; the function is `R51.R.tensors_sha`.
- Check 3's four dry runs write, by the runner's tag, into `$M/exp45/e55` (receipt, files of generated bytes of rows 0 and 64) and into `rows_uncommitted`. "Files committed" and "Done" do not say whether they are committed.

Everything else in lines 452 to 548 was read against the list and is covered: `cfg` (line 453) is used on lines 463, 469, 519 and 520 and 525, each of which the list replaces; `restore_book` (lines 469 and 520) is replaced; the places that test `start` (lines 454, 456, 495, 503, 519 and 525) are named.

**Replacement.** Add to point 4: "In `write`, `experiment="E55"`. The settings add `start_shut=dict(file=..., sha256=..., tensors_sha256=...)` from the receipt of the start `shut`; the entry `start` keeps the record of the keyed start from which it was made. The runner calls `T.set_num_threads(1)` before anything else when it is called as a program, as `exp51_reserve.py` does on its line 706." In point 1: "`R51.R.tensors_sha`". In "Files committed": "the receipts and the files of generated bytes of the four dry runs of check 3; their weights under `rows_uncommitted` are not committed."

### 10. Checks 1 (c) and 2 (b) pass if the two added lines are still installed, and two limits cannot fail

**Place.** Check 1 (c): "the largest of 1 − z of a cell in reserve is at most 1e-4; the largest magnitude of a state of a cell in reserve and the largest magnitude of an added readout unit at the answer positions are recorded and are at most 0.01; the largest difference of an output ... is at most 0.02". Check 1 (d): "is recorded and is at most 5". Check 2 (b).

**What is wrong.** `R51.load_start` installs the keyed forward (through `build_start`, line 258). Check 1 (c) needs the keyed start as its reference, so the worker loads it, after or before `load_shut`; if the key is then still installed when the start `shut` is measured, 1 − z is exactly 0, the states are 0, the added units are 0 and the outputs equal the reference: every limit of 1 (c) and (d) is met by a check that did not test the start. The limits are one-sided. The run asserts `R51.KEY["in_use"] is None`; the checks do not. Separately, the limits of 0.02 and of 5 do not distinguish the start `shut` from a start in which 10 was added to the bias and no projection was made.

**Evidence** (part `start`, held-out pairs; sweep; back-propagation).

| Figure of check 1 | The start `shut` | 10 added to the bias, no projection | The brief's limit |
|---|---|---|---|
| Largest 1 − z | 4.541e-5; 4.541e-5 | 3.8e-4; 3.8e-4 | at most 1e-4 |
| Largest state of a cell in reserve | 5.6e-4; 5.9e-4 | 1.6e-3; 2.6e-3 | at most 0.01 |
| Largest added readout unit | 2.9e-4; 3.6e-4 | 0.90; 0.96 | at most 0.01 |
| Largest difference of an output | 2.4e-3; 2.5e-3 | 2.9e-3; 3.4e-3 | at most 0.02 |
| Answers that differ, of 32,884 | 1; 0 | 1; 0 | at most 5 |

**Replacement** for check 1 (c): "(c) With `R51.KEY["in_use"]` set to `None` and asserted so, on the two panels, by `gates_on`: the largest of 1 − z of a cell in reserve lies between 4.5e-5 and 4.6e-5; the largest magnitude of a state of a cell in reserve lies between 1e-4 and 1.1e-3; the largest magnitude of an added readout unit at the answer positions is above 0 and at most 1e-3. The largest difference of an output at an answer position from that of the keyed start under the keyed forward (`R51.KEY["in_use"]` set to the bytes in use for the reference only) is recorded and is above 0 and at most 0.02." The limit of 5 in (d) may stay; the text should say that it is a record and does not test the projection.

### 11. Small disagreements between "What the record says" and the record

Every other figure of the section was recomputed from the receipts and agrees (see "What was verified").

- "the outputs differ from the birth's by up to 0.0030; 0.0034" and, in check 1 (c), "which gave 0.0030 and 0.0034". The receipt of E51's check 1 has 0.002949 and 0.003405 (the verification writes 0.00295 and 0.00340). **Replacement:** "0.0029; 0.0034".
- Table, arm `rehearse`: "as `open`, with every second item drawn from the first family". Each item is drawn with equal chance from the two streams (`item_source`, lines 434 to 436; 131,138 and 131,006 items in the run on `unrelated`). **Replacement:** "as `open`, with each item drawn with equal chance from the new task's stream or from the first family's taught half".
- The same cell: "of the 31,684 answers, 9.6% and 9.8% differ from row 0's under the sweep and 6.7% and 4.1% under back-propagation". The four figures are right (9.60 and 9.76; 6.65 and 4.14), but the pairs are ordered by task within a rule, against the section's rule "sweep; back-propagation". **Replacement:** "of the 31,684 answers, 9.6%; 6.7% differ from row 0's on `unrelated` and 9.8%; 4.1% on `conflict`".
- "by row 4,096 they have moved by up to 5.3e-4, because the weights are stored in single precision after each projected step". The verification gives this cause in its conclusion and says in its section 3 that the estimate "does not prove that rounding is the only source". **Replacement:** "which the verification attributes to the storing of the weights in single precision after each projected step, by an estimate of size".
- "Terms", the task `unrelated`: "two of 64 symbols are shown, then 0 to 12 fillers, then a query byte". The item begins with a code byte (224) before the two symbols (`e43_task.py`; the verification's `u_render`). **Replacement:** "a code byte, then two of 64 symbols, then 0 to 12 fillers, then a query byte".

### 12. Predictions

- P55.1 (65%) is false as the brief stands, by defect 1.
- P55.5 and P55.6 are decided by the count of defect 3; the replacements are given there. With the brief's wording P55.6 would be scored false in the emulated run of `unrelated` under back-propagation, in which the task is learned and 93 cells are open by more than 0.5 at some position (21 by the brief's count).
- P55.6 and P55.8 are true by their own clause when no run learns its task. With P55.2 at 45% the larger part of their 80% and 75% is this clause. **Replacement** for both clauses: "(the statement is not scored if no run learns its task)".
- The other predictions are decidable from the receipts as worded.
- The emulation gives, for what it shows of sizes: `unrelated` learned under back-propagation only and from row 4,096 (P55.2 and P55.8 not met); `conflict` not learned (P55.3 not met); d(g) above 31 at some row of all four runs (P55.4 not met by the brief's words); the gates shut on the first family's items throughout (P55.5 met). The predictions are the author's and are recorded before any run; the author may wish to set the chances again with these figures in view.

### 13. Language

The brief asks the worker for full sentences, every term defined at first use and no figure of speech. Its own text fails this in the following places.

| Sentence or phrase of the brief | What fails | Replacement |
|---|---|---|
| Title: "without a line of the program that looks at the item" | figure of speech | "without a line of the forward pass that reads the item's first byte" |
| "**Can the weights carry the separation?**"; "the separation was carried by weights"; "the weights did not carry the separation" | figure of speech; "the separation" is not defined | "Is the first family kept and the new task learned when no line of the forward pass tells the two tasks apart?"; "no line of the forward pass told the two tasks apart"; "without the two added lines the first family was not kept" |
| "That separation is an act of the program and not something the network has learned or holds in its weights." | figure of speech | "The two tasks are told apart by two lines of the program; this is not learned and is not in the weights." |
| "when the cells in reserve are silent" | term not defined | "when the states of the cells in reserve are zero" |
| "shut", "open", "open by more than 0.1" of a gate, from the first paragraph on | terms not defined in "Terms" | add to "Terms": "An update gate is *shut* when its z is close to 1 and *open by x* when 1 − z is x." |
| "the forward pass of the plain network", "the forward without a key", "the keyed forward", "the keyed start" | terms not defined in "Terms" | add: "*The plain forward pass* is `exp51_reserve.fwd` without the two added lines (`KEY["in_use"]` is `None`); *the keyed forward pass* is the same with them. *The keyed start* is E51's embedded network as saved before any update." |
| "panel", "stream", "data digest", "letters", "held-out pairs", "teacher forcing" | terms not defined | add one sentence for each to "Terms", from the brief of E51 |
| "a shut gate lets through a share of 4.5e-5 of the candidate" | figure of speech | "a gate with a bias of 10 has 1 − z = 4.5e-5, so that at each position the state takes 4.5e-5 of the candidate" |
| "keep their draws" | coined phrase | "keep the values drawn when the embedded network was made" |
| "Seed 73129; 4,096 updates; no weight decay; a new optimizer at the start of the new task." | not a sentence | "Every run of the record has seed 73129, 4,096 updates, no weight decay and a new optimizer at the start of the new task." |
| "Two arms, two tasks, two rules, seed 73129: eight runs of 4,096 updates, each network under the rule of its birth." | not a sentence | "The two arms are run on two tasks under two rules with seed 73129, which gives eight runs of 4,096 updates; each network is updated under the rule of its birth." |
| "(c) The check can fail: in the arm ..." | not a sentence; the label says nothing | "(c) A case in which a difference must be found: in the arm ..." |
| "as E51's check 3 makes them" | unclear reference | "made as in `exp51_reserve.check3` (lines 599 to 606)" |

## What was verified and found sound

**The record** (part `record`; sweep; back-propagation). Every figure of "What the record says" other than those of defects 2 and 11 was recomputed from the sixteen receipts and agrees.

| Statement of the brief | Recomputed |
|---|---|
| Reduction score at row 0: 92.4; 95.3 | 92.40; 95.25 |
| `keyed`: 92.4; 95.3 on both tasks, no answer of 32,884 changed at any row | 92.40; 95.25 at all ten rows of the four runs; 0 of 31,684 and 0 of 1,200 answers differ at any row |
| `keyed`: learned from row 1,024; 512 on both tasks; 119; 125 and 112; 83 cells | the same |
| `keyed-free`: 0.0; 0.0 and 1.0; 0.2; learned from 512; 512 and 2,048; 1,024 | 0.00; 0.00 and 1.00; 0.23; the same rows |
| `open`: 0.0; 0.0, lost by row 16; 1.3; 0.0; learned from 512; 512 and 1,024; 1,024; 1; 8 and 0; 0 cells | 0.00; 0.01 at row 16; 1.32; 0.00; the same rows; the same counts |
| `rehearse`: 90.0; 92.8 and 88.3; 97.8; 9.6%, 9.8%, 6.7%, 4.1% of the answers differ | 89.95; 92.81 and 88.33; 97.79; 9.60, 9.76, 6.65, 4.14 |
| The sixteen runs cost 2.74 CPU-hours | 2.739 |
| The start of `open`: largest state 0.0015; 0.0024; 1; 0 answers differ | 0.00147; 0.00240 (row 0 of the receipts); 1; 0 (the verification, section 2) |

The sentences attributed to the verification were compared with its text: the two added lines and "none ... at most 0.4%" (its conclusion 2: 0.00% and 0.40%, 0.00%); 5.3e-4 and 1.8e-4 (conclusion 1 and section 3); the rule at the recurrent matrix and at `lm` (section 3); 4.5e-5 (section 2); the directions in use as the codes of the bytes in use, the coordinates of the old cells and the codes of `=` and the ten digits (section 3); the criterion "learned in the reserve" (section 8.2). They agree, with the two exceptions of defect 11. The runner's SHA-256 on the worker's branch (at 0ab99a37) and the commit of the default branch are those the brief names.

**The construction.** The start `shut` does what "The change", point 2, says: on every vector in the span of the directions in use an update gate of the reserve receives 10 from the input map and nothing from the recurrent matrix, and an added readout unit receives nothing from `l2` and from `lm` (remainders 1.9e-7 at most). The limits of check 1 can be met. The brief's statements that the states of the cells in reserve are of the order of 1e-3 on the first family's items and that the gate stays at 1 − 4.5e-5 while those states are small are right. Its statement that a new computation of the directions on the start `shut` would protect every weight from a cell in reserve is right by the sizes: a state of 5e-4 gives an eigenvalue of about 1e-7, and `R51.directions` keeps every eigenvalue above 1e-12 of the largest, which is 13 to 18. On the first family's held-out pairs the gates stay shut through the emulated runs.

**The runner.** `shut_start`, `load_shut` and `Rule55` were read against `R51.away`, `R51.load_start`, `R51.build_start`, `R51.install` and `R51.Rule` and do what the text says. The blocks and the direction matrices fit (128 by 513 against 513 by 46; 128 by 257 against 257 by 129; 256 by 257 against 257 by 129; 256 by 513 against 513 by 11). Each of the two bias vectors of the recurrent layer is appended to its own matrix, as in `R51.Rule`, and the 10 is added to the bias of the input map only. The row sets are index tensors without a repeated entry (384 old rows and 128 gate rows), so that `G[rows] = ...` and `W[rows] = ...` assign every row once. `R51.load_start("keyed", ...)` asserts the file and the tensors of the start and the file of the directions against E51's receipt, returns the saved directions, and installs the keyed forward; `R51.install(None)` then removes the key, and `arch26.install` may be called twice. `Rule55` inherits `before` and `after`, which take the matrices, the directions and the rows from `self.mats`. `gates_on` is specified so that one thing is computed. The stream is that of E51's `keyed`. With the additions of defect 9 a copy of lines 452 to 548 with the listed changes has no line that fails.

**The checks.** Checks 1 (a), 1 (b), 2 (a), 2 (c) and 3 can be carried out as worded and can fail. 1 (b): the blocks of the keyed start give 0.10 to 0.15 where the limit is 1e-5. 2 (a), after eight emulated updates: 5.1e-7 to 8.7e-7 at the input map, at most 1.4e-8 at `l2` and below 1e-25 at the other two, against 1e-5; the largest of these is the rounding of a bias near 10, whose single-precision spacing is 9.5e-7, and the limit is twelve times it. 2 (c): 0.10 to 0.14 at the input map and 0.035 to 0.058 at the recurrent matrix, against 1e-4.

**Rules for the worker, reporting, "Done".** They are consistent with one another and with the list of committed files. The files of E51's Part B on the branch (`exp51_partb.py`, `exp45/e51/partB/`) do not meet the files of this job (`exp55_shut.py`, `exp45/e55/`). `R51.refuse_existing` refuses an existing receipt, which agrees with the rule that an interrupted run's files are moved before it is started again.

## What could not be checked

- *The runner in torch.* torch is not installed here. Nothing of `exp55_shut.py` or of `exp51_reserve.py` was executed; what is said about torch's indexing, about `no_grad` and about the optimizer comes from reading the sources.
- *The emulation against the runner beyond 64 updates.* It reproduces the saved weights of E51's `keyed` at row 64 to 4.7e-5. Its later rows cannot be compared with anything; they depend on rounding. It computes the forward pass in double precision; in single precision 1 − z of a shut gate is a multiple of 6e-8, which the emulation does not have.
- *The arm `shut-gates-free`* was emulated for eight updates only. The four runs of `shut` were emulated once, to 4,096 updates, with the first family's 31,684 answers scored at rows 0 and 256 to 4,096 only; a second run with other rounding would give other figures at the later rows.
- *Costs.* The forecast of check 3 and the CPU-hours of the eight runs.
- *The verification of E51's trial* was read as it stands in the working tree. Its section 8.3 and its last section still hold seven unfilled marks (`@@A@@` to `@@F@@`, `@@TIMES@@`); no sentence that the brief cites is among them.
- *The plan.* The list of `PLAN.md` describes item 11 as "on a birth with E50's code (no device)", waiting for items 4 and 7; the first part uses E41's births and E51's rule. The plan's state table (00:17) records the first part as the brief describes it. Whether the list entry should be amended is the author's decision.
