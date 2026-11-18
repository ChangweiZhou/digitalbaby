# Check of the brief of E55, first part, revision 3, and of two clauses on saving and resuming in the briefs of E57 and E53

18 November 2026, begun 02:50 UTC. Written for the outside reviewer by the agent that checked revision 1 of the brief of E55 (`CHECK_OF_TASK_E55_BRIEF_20261118.md` beside this file). The agent was given the briefs and the files they build on, and not the author's reasoning. The texts checked are those of commit 1c7e515a of the reviewer's branch. The check is a reading and a recomputation with numpy; no code of another branch was run and torch was not used.

**Verdicts.**

- **A, the brief of E55, revision 3: release after the listed changes.** The arm `blind` is sound as arithmetic: every limit of checks 1 and 2 for it can be met and can fail, `Rule55` does what the text says, and in an emulation of 16,384 updates the first family's 32,884 scored answers stay the same in three of four runs and all but one held-out answer at one row in the fourth, while the unrelated task is learned under both rules. Eight defects are listed. Two concern what the main reading can carry. In the emulation the state zero of the reserve becomes unstable in every run, the state on a first-family item then grows by a factor at each position, and under back-propagation on the unrelated task the same network that keeps every scored answer loses 1,055 of 1,200 answers of the first family when the items are written with 40 letters in place of 12; no scored item is longer than 22 positions (defect A-1). And the old rows keep E51's projection of the step for four times E51's updates, so that the rounding which the verification of E51 measured can change an answer without any part of the reserve (defect A-2).
- **B1, the brief of E57, revision 4 (point 7 of the runner, check 8, the rule for an interrupted run): release after the listed changes.** What is saved is enough for a resumed run to equal an uninterrupted one. Six things in the wording and in check 8 should be changed; none touches a run.
- **B2, the brief of E53, revision 3 (check 4): release as it stands, with one sentence corrected.** The check can be carried out and compared as it says, and no file that the killed process leaves can make the resumed run differ.

The script is `check_e55_brief_r3.py` beside this file; it imports the script of the first check (`check_e55_brief.py`) and, through it, the reviewer's verification script of E51's trial. It is run from the repository root as `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -I PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/COORDINATION_20261117/check_e55_brief_r3.py <part>`. Its parts are `blind_start` (the start `blind` and the figures of check 1), `blind_check2` (eight emulated updates and the figures of check 2), `blind_emulate` (the emulation; the table below was made with `E55R3_UPDATES=16384 E55R3_DTYPE=32`, one run a process, about an hour of one thread a run) and `blind_states` (the emulated networks after their last update, which `blind_emulate` writes to a folder named by `E55R3_SAVE`; the four files were written to the checker's scratch folder and deleted after use). Figures are given as sweep; back-propagation.

## A. The brief of E55, revision 3

### A1. The arm `blind` as arithmetic

**The start `blind`** (part `blind_start`; the start written in numpy as the brief's `make_blind` and `load_start55` write it).

- *The constant.* The directions in use of the recurrent matrix and of `lm` contain the constant: the part of the constant outside them has a 2-norm of 1.4e-15; 1.6e-15 and 1.8e-15; 1.5e-15. The projection therefore brings the bias of the rows of the reserve in the recurrent matrix and of rows 256 to 511 of `lm` to zero (at most 2.7e-16 before `zero_blocks`), as the brief says. At the input map and at `l2` the constant is not among the directions (0.77; 0.79 and 0.94; 0.93 of it lie outside), and the biases there are not brought to zero (up to 0.08 and 0.06); the brief does not say that they are.
- *The limits of check 1 for the start `blind` can all be met.*

| Figure of check 1 | Double precision | Single precision | Limit |
|---|---|---|---|
| (b) largest entry of `D @ Vm` | 3.1e-9; 3.7e-9 at the input map, 2.3e-9; 2.9e-9 at `l2`, 0 at the other two | | at most 1e-5 |
| (c) largest state of a cell in reserve, two panels | 6.6e-8; 7.6e-8 | 2.6e-7; 3.5e-7 | at most 1e-5 |
| (c) largest added readout unit at an answer position | 4.9e-8; 6.6e-8 | 2.3e-7; 4.0e-7 | at most 1e-5 |
| (c) largest difference of an output from the keyed start under the keyed forward | 1.7e-7; 1.9e-7 | 1.1e-5; 7.6e-6 | at most 1e-4 |
| (c) largest state of a cell in reserve on the first batch of `unrelated`; of `conflict` | 0.15; 0.15 and 0.07; 0.09 | | above 1e-3 |
| (d) answers of the 32,884 that differ | 0; 0 | 0; 0 | 0 |

- *Each limit can fail when the start is wrong.* With no projection, with the gate rows of the reserve alone made blind, or with the rows of the reserve made blind in the input map and not in the recurrent matrix, the largest state of a cell in reserve on the held-out pairs is 0.59 to 0.93, the largest difference of an output 2.4 to 3.2, and 12 to 45 of the 1,200 held-out answers differ; (b) gives 0.14 to 0.15 at the block that was left out. A start in which the rows of the reserve were set to zero would pass the upper limits and fail the lower limit of the first batch, which is there for that case.
- *At the start the state zero of the reserve is stable.* On a byte in use the gates of a cell in reserve stand at one half (1 − z between 0.500000 and 0.500000), and a small state h of the reserve is taken to (0.5 I + 0.25 Wn) h from one position to the next, with Wn the weights of the candidate rows from the cells in reserve. The largest magnitude of an eigenvalue of that map is 0.605 under both rules, and the largest state does not grow over the 22 positions of a held-out pair (4e-8 at position 1, 6e-8 at position 21).

**The rule of the arm `blind` after eight updates** (part `blind_check2`). The limits of check 2 (a) and (d) can be met.

| Figure of check 2, arm `blind`, four cases | Found | Limit |
|---|---|---|
| (a) old rows, largest entry of `(D_after − D_start) @ Vm` | 1.3e-8 to 1.5e-8 | at most 1e-5 |
| (d) rows kept blind, largest entry of `D_after @ Vm` | 3.1e-9 to 3.8e-9 | at most 1e-5 |
| (d) largest state of a cell in reserve, two panels | 6.3e-8 to 6.9e-8 (2.7e-7 to 3.9e-7 in single precision) | at most 1e-5 |
| (d) largest change of an output at an answer position | 4.0e-7 to 6.7e-7 | at most 1e-4 |
| (d) change of what a row of the reserve receives from the input map on a byte of the new task, largest magnitude | 0.14 to 0.20 | above 0.01 |

The limit of 1e-4 on the change of an output is that of E51's check 3, which gave 1.5e-5 to 2.1e-5 in single precision; in torch the figure will be of that size, because it is the arithmetic of single precision and not a movement of the weights.

**The emulation of the arm `blind`** (part `blind_emulate`) is the checker's own arithmetic and is not the runner. The gradient is computed in single precision, as torch computes it, the projections in double precision, and the weights are stored in single precision after every step. Sixty-four emulated updates of E51's arm `keyed` give the saved weights of row 64 to within 2.4e-5; 2.7e-5 in every entry, where the weights changed by up to 5.9e-2. Its later rows show sizes and the order of events and are not the figures that the runner will give. Each task was emulated under each rule for 16,384 updates, once. The answers are generated in double precision; the states are given in double and in single precision, because what a cell in reserve receives on a byte in use is of the size of the rounding of single-precision arithmetic. The table leaves out the rows 16, 32, 128 and 512; at those rows no answer of the first family differs in any run.

| Run | Row | Never-taught score; with the cells in reserve held shut | First family: answers that differ from row 0's, of 1,200 held-out; of 31,684 without letters | Reduction score | Largest state of a cell in reserve on the held-out pairs, double precision; single precision | The same at positions 1; 11; 21 | Largest eigenvalue of the map of a small state | Largest change of an output on the held-out pairs | Largest entry of `D @ Vm` at the rows kept blind |
|---|---|---|---|---|---|---|---|---|---|
| `unrelated`, back-propagation | 0 | 0.0; 0.0 | 0; 0 | 95.25 | 7.6e-08; 3.5e-07 | 3.3e-08; 6.9e-08; 5.3e-08 | 0.605 | 0.0e+00 | 3.7e-09 |
| `unrelated`, back-propagation | 64 | 0.0; 0.0 | 0; 0 | 95.25 | 1.9e-07; 1.3e-06 | 4.5e-08; 1.5e-07; 1.9e-07 | 0.907 | 4.1e-06 | 4.8e-09 |
| `unrelated`, back-propagation | 256 | 0.0; 0.0 | 0; 0 | 95.25 | 2.0e-07; 1.3e-06 | 5.6e-08; 1.6e-07; 1.5e-07 | 0.837 | 3.4e-06 | 4.8e-09 |
| `unrelated`, back-propagation | 1,024 | 25.3; 0.0 | 0; 0 | 95.25 | 3.1e-05; 7.1e-05 | 1.5e-07; 3.4e-06; 3.2e-05 | 1.193 | 9.1e-04 | 2.1e-08 |
| `unrelated`, back-propagation | 2,048 | 99.0; 0.0 | 0; 0 | 95.25 | 1.1e-04; 3.1e-04 | 1.4e-07; 6.2e-06; 1.1e-04 | 1.316 | 6.0e-03 | 1.8e-08 |
| `unrelated`, back-propagation | 4,096 | 100.0; 0.0 | 0; 0 | 95.25 | 1.7e-04; 8.1e-04 | 1.5e-07; 6.6e-06; 1.7e-04 | 1.366 | 1.3e-02 | 1.9e-08 |
| `unrelated`, back-propagation | 8,192 | 100.0; 0.0 | 0; 0 | 95.25 | 2.7e-04; 1.7e-03 | 1.6e-07; 8.6e-06; 2.7e-04 | 1.408 | 2.8e-02 | 2.0e-08 |
| `unrelated`, back-propagation | 16,384 | 100.0; 0.0 | 0; 0 | 95.25 | 3.5e-04; 8.1e-04 | 1.6e-07; 9.3e-06; 3.5e-04 | 1.429 | 4.0e-02 | 1.9e-08 |
| `unrelated`, sweep | 0 | 0.0; 0.0 | 0; 0 | 92.40 | 6.6e-08; 2.6e-07 | 4.2e-08; 5.5e-08; 5.5e-08 | 0.605 | 0.0e+00 | 3.1e-09 |
| `unrelated`, sweep | 64 | 0.0; 0.0 | 0; 0 | 92.40 | 1.2e-07; 5.8e-07 | 4.5e-08; 1.2e-07; 9.7e-08 | 0.689 | 5.2e-06 | 5.4e-09 |
| `unrelated`, sweep | 256 | 0.0; 0.0 | 0; 0 | 92.40 | 1.2e-07; 5.3e-07 | 5.8e-08; 1.1e-07; 1.0e-07 | 0.684 | 7.4e-06 | 6.4e-09 |
| `unrelated`, sweep | 1,024 | 0.0; 0.0 | 0; 0 | 92.40 | 2.0e-06; 4.3e-06 | 7.1e-08; 1.0e-06; 2.0e-06 | 1.048 | 2.6e-05 | 1.3e-08 |
| `unrelated`, sweep | 2,048 | 6.0; 0.0 | 0; 0 | 92.40 | 2.0e-06; 2.8e-06 | 1.1e-07; 9.1e-07; 2.0e-06 | 1.047 | 9.1e-05 | 2.4e-08 |
| `unrelated`, sweep | 4,096 | 65.3; 0.0 | 0; 0 | 92.40 | 3.1e-06; 6.4e-06 | 1.9e-07; 1.5e-06; 3.1e-06 | 1.086 | 2.3e-04 | 1.9e-08 |
| `unrelated`, sweep | 8,192 | 97.3; 0.0 | 1; 0 | 92.40 | 4.0e-06; 7.5e-06 | 1.5e-07; 1.8e-06; 4.0e-06 | 1.097 | 2.8e-04 | 2.0e-08 |
| `unrelated`, sweep | 16,384 | 99.7; 0.0 | 0; 0 | 92.40 | 4.5e-06; 1.0e-05 | 1.7e-07; 1.7e-06; 4.5e-06 | 1.127 | 7.6e-04 | 2.3e-08 |
| `conflict`, back-propagation | 0 | 8.2; 8.2 | 0; 0 | 95.25 | 7.6e-08; 3.5e-07 | 3.3e-08; 6.9e-08; 5.3e-08 | 0.605 | 0.0e+00 | 3.7e-09 |
| `conflict`, back-propagation | 64 | 21.2; 15.1 | 0; 0 | 95.25 | 2.1e-07; 7.8e-07 | 4.5e-08; 1.3e-07; 2.1e-07 | 0.969 | 3.0e-06 | 5.3e-09 |
| `conflict`, back-propagation | 256 | 24.6; 20.2 | 0; 0 | 95.25 | 9.8e-07; 2.0e-06 | 6.2e-08; 3.4e-07; 9.8e-07 | 1.043 | 1.4e-05 | 7.6e-09 |
| `conflict`, back-propagation | 1,024 | 46.6; 28.8 | 0; 0 | 95.25 | 9.3e-07; 3.3e-06 | 1.2e-07; 2.7e-07; 9.3e-07 | 1.103 | 2.0e-05 | 1.6e-08 |
| `conflict`, back-propagation | 2,048 | 52.6; 31.0 | 0; 0 | 95.25 | 2.3e-06; 8.7e-06 | 1.2e-07; 5.4e-07; 2.3e-06 | 1.161 | 3.3e-05 | 1.8e-08 |
| `conflict`, back-propagation | 4,096 | 61.1; 31.6 | 0; 0 | 95.25 | 3.8e-06; 1.4e-05 | 1.3e-07; 7.3e-07; 3.8e-06 | 1.170 | 7.3e-05 | 2.1e-08 |
| `conflict`, back-propagation | 8,192 | 61.5; 31.9 | 0; 0 | 95.25 | 5.1e-06; 2.4e-05 | 1.9e-07; 9.8e-07; 5.1e-06 | 1.180 | 1.3e-04 | 3.2e-08 |
| `conflict`, back-propagation | 16,384 | 63.2; 28.9 | 0; 0 | 95.25 | 1.2e-05; 4.2e-05 | 1.6e-07; 1.9e-06; 1.2e-05 | 1.197 | 3.1e-04 | 4.5e-08 |
| `conflict`, sweep | 0 | 19.0; 19.0 | 0; 0 | 92.40 | 6.6e-08; 2.6e-07 | 4.2e-08; 5.5e-08; 5.5e-08 | 0.605 | 0.0e+00 | 3.1e-09 |
| `conflict`, sweep | 64 | 24.6; 23.8 | 0; 0 | 92.40 | 8.0e-08; 4.8e-07 | 4.5e-08; 7.8e-08; 7.8e-08 | 0.930 | 2.7e-06 | 4.3e-09 |
| `conflict`, sweep | 256 | 20.9; 31.4 | 0; 0 | 92.40 | 6.5e-07; 9.5e-07 | 7.4e-08; 2.6e-07; 6.5e-07 | 1.064 | 1.3e-05 | 7.4e-09 |
| `conflict`, sweep | 1,024 | 45.9; 35.4 | 0; 0 | 92.40 | 8.1e-07; 1.8e-06 | 7.7e-08; 3.6e-07; 8.1e-07 | 0.991 | 3.2e-05 | 1.0e-08 |
| `conflict`, sweep | 2,048 | 51.9; 37.0 | 0; 0 | 92.40 | 7.9e-07; 2.6e-06 | 1.2e-07; 5.8e-07; 6.8e-07 | 0.959 | 5.8e-05 | 1.6e-08 |
| `conflict`, sweep | 4,096 | 57.2; 37.0 | 0; 0 | 92.40 | 1.0e-06; 2.6e-06 | 1.4e-07; 5.2e-07; 1.0e-06 | 1.038 | 1.1e-04 | 2.3e-08 |
| `conflict`, sweep | 8,192 | 64.1; 35.0 | 0; 0 | 92.40 | 1.1e-06; 3.3e-06 | 1.8e-07; 7.5e-07; 1.1e-06 | 1.045 | 2.3e-04 | 2.2e-08 |
| `conflict`, sweep | 16,384 | 66.6; 34.0 | 0; 0 | 92.40 | 1.9e-06; 5.3e-06 | 2.3e-07; 9.3e-07; 1.9e-06 | 1.099 | 5.2e-04 | 5.8e-08 |

By the brief's words: `unrelated` under back-propagation is *learned from row 2,048* and the first family *kept exactly*; `unrelated` under the sweep is *learned from row 8,192* and the first family *kept* (one held-out answer differs at row 8,192 and none at any other row); `conflict` is *not learned* under either rule (66.6; 63.2 at row 16,384, against 34.0; 28.9 with the cells in reserve held shut) and the first family is *kept exactly*.

The two questions of the request:

- *Does the remainder of the projection stay of the order that the brief states when the whole row is projected?* Yes. The largest entry of `D @ Vm` at the rows kept blind is between 3.1e-9 and 5.8e-8 at every row of every run, and what a cell in reserve receives at position 1 of a first-family item gives a state of 1.6e-7 to 2.3e-7 at row 16,384 (2.3e-6 to 3.2e-6 in single precision), against 3e-8 to 4e-8 at the start.
- *Does the state of the reserve on the first family's items grow over the positions of an item or over the updates?* Over the positions, by a factor that grows over the updates. On a byte in use the gates of a cell in reserve stand at one half, and a small state h of the reserve is taken to (0.5 I + 0.25 Wn) h, with Wn the weights of the candidate rows from the cells in reserve. The largest magnitude of an eigenvalue of this map is 0.605 at the start, passes 1 between rows 64 and 1,024 in all four runs, and is 1.13; 1.43 (`unrelated`) and 1.10; 1.20 (`conflict`) at row 16,384. The state at position 1 stays at the size of the remainder; the state at position 21 is larger by about the 20th power of the eigenvalue: on `unrelated` under back-propagation at row 16,384 it is 1.6e-7 at position 1, 9.3e-6 at position 11 and 3.5e-4 at position 21 (8.1e-4 in single precision), and the first family's outputs have moved by 4.0e-2 (8.9e-2 in single precision) without changing an answer.

**The emulated networks after their last update** (part `blind_states`).

| Run | Largest state of a cell in reserve at position 21 of a held-out pair (double; single precision) | Largest change of an output on the held-out pairs: as emulated; with the cells in reserve held shut; with the change of the old rows along the directions in use taken away; with both | The same 1,200 held-out pairs written with 40 letters (50 positions): largest state of a cell in reserve at the last position (double; single); answers that differ from the start's (double; single) |
|---|---|---|---|
| `unrelated`, back-propagation | 3.5e-4; 8.1e-4 | 4.0e-2; 8.1e-4; 4.0e-2; 2.0e-6 | 1.0; 1.0; 1,055; 1,132 of 1,200 (right answers 141; 66, where the start has 1,144) |
| `unrelated`, sweep | 4.5e-6; 1.0e-5 | 7.6e-4; 6.2e-4; 3.9e-4; 3.9e-6 | 1.5e-4; 3.2e-4; 0; 0 |
| `conflict`, back-propagation | 1.2e-5; 4.2e-5 | 3.1e-4; 2.9e-4; 1.3e-4; 5.3e-6 | 1.8e-3; 6.7e-3; 0; 0 |
| `conflict`, sweep | 1.9e-6; 5.3e-6 | 5.2e-4; 5.2e-4; 8.0e-5; 1.0e-5 | 2.5e-5; 7.7e-5; 0; 0 |

Two paths reach the first family in this arm, and the table separates them. The states of the cells in reserve move the outputs by 4.0e-2 on `unrelated` under back-propagation and by 8e-5 to 4e-4 in the other three runs. The part of the change of the old rows that lies along the directions in use, which E51's rule leaves to accumulate, moves them by 2.9e-4 to 8.1e-4 in all four runs. With both taken away the outputs are within 1e-5 of the start's.

### A2. `Rule55`

The code of `Rule55` for the arm `blind` does what points 5 and 6 of "The change" say. `before` first projects the gradient of the rows kept blind (`W.grad[rows]` with an index tensor gives a copy, and the assignment to `W.grad[rows]` writes it back in place); `R51.Rule.before` then copies each whole gradient through double precision, which leaves those rows as they are, projects the old rows, and takes its snapshot of the weights before the optimizer's step. `R51.Rule.after` projects the step of the old rows from that snapshot and restores `l1`, the old columns and the bias of `l3` and the rows of the byte table; `make_blind` then projects the whole of each row kept blind and `zero_blocks` sets the named entries to zero. The emulation follows this order. The arm `shut` is what the check of revision 1 examined, with three differences that the brief states: the two bias vectors are set to zero as well, the two blocks are set to zero in the start and not only after each update, and the runs have 16,384 updates. Under back-propagation eight emulated updates of the arm `shut` of revision 3 give the weights of the check of revision 1 to 1e-13; under the sweep they differ by up to 1.9e-4 after eight updates, because the start of revision 1 still held the rounding remainders whose signs the sweep reads at the first update (finding 6 of the first check).

### A3 and A4. Defects, the most serious first

**Defect A-1. In the arm `blind` the state of the reserve on a first-family item grows by a factor at each position once learning has made the state zero unstable; the scored items are too short to show what this does, and its cause is not recorded.**

Place: "The change", the paragraph "What follows from the construction of the arm `blind`": "its state is of the order of 1e-7 as long as the weights among the cells in reserve do not enlarge it from one position to the next. The construction does not bound those weights. If learning on the new task makes the state zero of the reserve unstable, a remainder can grow over the 22 positions of an item; the largest state of a cell in reserve on the first family's items is recorded at every scored row for this reason." Also "How the result is stated", first paragraph; "What it cannot show", second point; P55.11.

What is wrong: the brief names as a possibility what the emulation shows in all four runs before row 1,024, and it lacks what is needed to read it. First, the growth is by a factor at every position, so that the state depends on the length of the item, and every scored item of the first family has at most 22 positions. A run can be *kept exactly* on the scored panels and lose the first family on longer items. Second, the quantity that decides the growth, the largest eigenvalue of 0.5 I + 0.25 Wn, is not recorded. Third, the runner records the answers that differ and not how far the outputs have moved; the verification of E51 asked for that figure beside the word *kept*.

Evidence: the two tables of A1. On `unrelated` under back-propagation the eigenvalue is 1.43 at row 16,384; no answer of the 32,884 differs at any row, the outputs have moved by 4.0e-2, and of the same 1,200 held-out pairs written with 40 letters 1,055 answers differ from the start's (1,132 in single precision), with 141 right answers where the start has 1,144. In the three other runs the eigenvalue is 1.10 to 1.20 and no answer at 40 letters differs. P55.11 puts its mark at 1e-4; the state passes it at row 2,048 on `unrelated` under back-propagation (1.1e-4; 3.1e-4 in single precision).

Replacement, in "The run", added to `score_row`: "`r["first_family_largest_change_of_an_output"]`, the largest magnitude of `R51.answer_logits(m, s2_items)` minus the same at row 0; `r["reserve_map_largest_eigenvalue"] = float(T.linalg.eigvals(0.5 * T.eye(128, dtype=T.float64) + 0.25 * m.rnn.weight_hh_l0[640:768, 128:256].double()).abs().max())`; and, with `len40 = R51.E.panel(("add", 89, b"p"), R51.E.split(("add", 89, b"p"), .5, seed)[1], seed, ks=(12, 40), n=300)[40]` (300 held-out pairs of addition at 40 letters, the length test of E52), `r["gates_on_first_family_40_letters"] = gates_on(m, len40)` and the number of the 300 generated answers that differ from row 0's."
Replacement for the last two sentences of the paragraph: "The construction does not bound those weights. On a byte in use the gates of a cell in reserve stand at one half, and a small state h of the reserve is taken to (0.5 I + 0.25 Wn) h from one position to the next, with Wn the weights of the candidate rows from the cells in reserve; if an eigenvalue of this map exceeds 1 in magnitude, the state grows by that factor at every position, and its size depends on the length of the item. In an emulation made for the check of revision 3 the largest eigenvalue rose from 0.61 at the start to 1.10 to 1.43 at row 16,384. No scored answer of the first family changed through the reserve; on the unrelated task under back-propagation the largest state at position 21 of a held-out pair was 3.5e-4, and of the same pairs written with 40 letters 1,055 of 1,200 answers changed. The run records the eigenvalue, the largest state and the largest change of an output at every scored row, and the answers of 300 held-out pairs at 40 letters."
In "How the result is stated", words for the first family: "For the arm `blind` the report gives, beside the word, the number of answers at 40 letters that differ from row 0's at every row; a run that is kept exactly or kept on the scored panels while more than 3 of these 300 answers differ is reported as 'kept on items of up to 22 positions and not on items of 50'."
In "What it cannot show": "An item of the first family with more positions has a larger state of the reserve in the arm `blind`. The job measures this on one panel of 40 letters and does not bound it."
P55.11 is the author's to set again; the emulation contradicts its 70%.

**Defect A-2. The old rows keep E51's projection of the step for four times E51's updates, so that the first family's answers can change through rounding alone.**

Place: "The change", point 6: "The old rows are treated as in E51 (the step is projected)" and, in the same point, "it prevents the remainder from growing with the number of updates, as it does in the old rows of E51's `keyed` (from 6e-7 after 64 updates to 2e-5 to 8e-5 after 4,096)". "What it cannot show", first point. P55.9.

What is wrong: the brief removes the growing remainder at the rows kept blind and leaves it at the old rows, in runs four times as long as the one in which the verification of E51 found it as large as the smallest difference between the two largest outputs of an answer. If an answer of the first family changes in the arm `blind`, the reading "kept exactly" fails for a cause that is not the reserve, and the runner's figures cannot say which cause it was unless the state of the reserve is small.

Evidence: the second table of A1. With the cells in reserve held shut the outputs of the emulated networks of row 16,384 have still moved by 2.9e-4 to 8.1e-4, and by 2e-6 to 1e-5 once the part of the change of the old rows along the directions in use is taken away as well; that part has entries of up to 2.5e-5 at the input map and 1.8e-5 at `l2`. The smallest difference between the two largest outputs is 1.8e-4 under the sweep and 7.8e-4 under back-propagation (the verification of E51, section 3). In the emulation one held-out answer differs at row 8,192 on `unrelated` under the sweep, where the largest state of a cell in reserve is 4.0e-6 and the outputs have moved by 2.8e-4.

Replacement for the first sentence of point 6: "At the old rows the whole change from the start, and not only the step, is projected after every update: with W0 the rows at the start, the rows are set to W0 + away(W − W0), in double precision. This is the change that the verification of E51 proposed for longer runs; it keeps the remainder at the size of one rounding." In `Rule55`: "`__init__` keeps `self.start = [(W[rows].detach().double().clone(), b[rows].detach().double().clone()) for W, b, Vm, rows in self.mats]`, and `after`, directly after `super().after()`, does for each entry of `self.mats` with its start (W0, b0): `D = R51.away(T.cat([W[rows].double() - W0, (b[rows].double() - b0)[:, None]], 1), Vm); W[rows] = (W0 + D[:, :-1]).float(); b[rows] = (b0 + D[:, -1]).float()`." This holds for both arms; in the arm `shut` the start of a gate row contains the bias of 10, and the change from it is what is projected. Check 2 (a) is unchanged. If the author keeps E51's projection of the step, the least that is needed is a record that separates the two causes: "at every scored row the first family's 1,200 held-out answers are generated a second time with `KEY["shut_all"]` set, and the number that differ from row 0's is recorded".
This check did not emulate the replacement; the removal of the same part from the final weights (the second table of A1) is what speaks for it.

**Defect A-3. One sentence of the two arms side by side says what the brief forbids elsewhere, two say more than two runs show, and one outcome has no sentence.**

Place: "How the result is stated", "The two arms side by side": "If the arm `blind` keeps the first family exactly and learns `unrelated`, the report says that cells which receive nothing from the first family's items sufficed for a task on bytes that the first family does not use." And: "If the arm `blind` does not learn `conflict`, the report says that such cells did not suffice for a family whose items are made of bytes in use after their first byte". And: "If the arm `shut` learns a task that the arm `blind` does not, the report says that reading what the first family presents was needed for that task on this seed".

What is wrong: in the arm `blind` the rows of the old cells and of the old readout units also change, away from the directions in use, and the weights from the cells in reserve into the old units are free. The verification of E51 found for `keyed` that the unrelated task is at 0.0 and 0.3 when the incoming connections of the old units are put back. "Sufficed" gives to the cells in reserve what the run does not separate, which is what the point "The score with the cells in reserve held shut" forbids for "learned in the reserve". The first sentence is also licensed when the score with the cells held shut is at 80 or more, and it has no form for a run that is *kept* and not *kept exactly*, which is the outcome of the emulation under the sweep. The second sentence has no form for the outcome of the emulation either: `conflict` is not learned, at 66.6; 63.2, and the cells in reserve add 33 and 34 points to what the network gives with them held shut, so that "did not suffice" would be reported where they did half of what was learned. In the third sentence the two arms differ in more than what a cell in reserve may read (the gates of the arm `shut` start shut by a bias and those of the arm `blind` stand at one half), and one pair of runs shows a difference and not a need. The outcome that the arm `blind` learns `conflict` has no sentence.

Replacement: "If the arm `blind` learns `unrelated`, with a score below 80 when the cells in reserve are held shut, and the first family is kept exactly or kept, the report says that the task was learned and the first family kept (exactly, or with its largest d(g) and the held-out answers that differ) in a network whose cells in reserve receive nothing from the first family's items; it does not say that these cells sufficed, because the rows of the old cells and of the old readout units change as well. For `conflict` in the arm `blind` the report gives the never-taught score at the last row beside the score with the cells held shut and says, if the task is not learned, that it was not learned within 16,384 updates by a network whose cells in reserve receive only the operation byte and their own states, and by how many points the cells in reserve raised the score; if it is learned, it says so with the same two figures. If the arm `shut` learns a task that the arm `blind` does not, the report says that on this seed the task was learned by the arm in which a cell in reserve may read what the first family presents and not by the arm in which it may not, names the other difference between the arms (the gates shut by a bias in the one and at one half in the other), and gives beside it what the arm `shut` lost of the first family."

**Defect A-4. The predictions and the section "For the coordinator" say that no emulation of the arm `blind` exists.**

Place: "Predictions": "P55.9 to P55.13 were set for revision 3; no emulation of the arm `blind` exists." "For the coordinator", point 2: "No emulation of this arm was made", and "The reviewer expects that arm to keep the first family and to learn the unrelated task, and not to learn the family modulo 97 beyond what the old cells give".

What is wrong: one exists now, and the chances P55.9 to P55.13 must say whether they were set before or after it. The five predictions are decidable from the receipts and are consistent with one another (P55.9 implies P55.10, and its chance is the smaller). Against the emulation: P55.9 (kept exactly in all four runs, 55%) is not met, by one held-out answer at one row under the sweep; P55.10 (d(g) at most 31) is met with d(g) = 0 at every row; P55.11 (the state below 1e-4, 70%) is not met, on `unrelated` under back-propagation from row 2,048; P55.12 (`unrelated` learned under both rules) is met; P55.13 (`conflict` learned under neither rule) is met. The expectation in "For the coordinator" that `conflict` is not learned beyond what the old cells give is contradicted: 66.6; 63.2 against 34.0; 28.9 with the cells held shut. The chances are the author's.

Replacement, in "Predictions": "P55.9 to P55.13 were set for revision 3 before any emulation of the arm `blind` existed; the check of revision 3 then emulated that arm to 16,384 updates (`CHECK_OF_TASK_E55_BRIEF_REVISION_3_20261118.md`). The chances were left as they were." If the author sets them again, the last sentence says so instead. In "For the coordinator", point 2, in place of the last sentence: "The check of revision 3 emulated this arm (its part A1): the first family's scored answers were kept, the unrelated task was learned under both rules, the family modulo 97 reached 63 to 67, about 33 points above what the network gives with the cells in reserve held shut, and on first-family items of 40 letters the answers were lost in one run of four."

**Defect A-5. "Done" counts two start files; there are four.**
Place: "Done": "The job is done when the two start files with their receipts".
Evidence: "The runner", point 5: "the four start files (two forms, two rules) with their receipts"; the command `start <blind|shut> <S|B>`.
Replacement: "The job is done when the four start files with their receipts".

**Defect A-6. A run that is partly kept may have no row at which d(g) exceeds 31.**
Place: "How the result is stated": "In both cases it gives the row at which d(g) first exceeds 31".
What is wrong: *kept* has three conditions (d(g), the held-out answers, the reduction score), and a run is partly kept when any one fails.
Replacement: "In both cases it gives the first row at which a condition of *kept* fails, and which condition".

**Defect A-7. Three figures or references are out of date.**
- "What the record says", the emulation of the arm `shut`: "(1 − z at most 5.2e-5)". The check's document has 5.5e-5 (its table, `conflict` under the sweep at row 4,096). Replacement: "(1 − z at most 5.5e-5)".
- The same paragraph: "The reduction score of the first family stays within 1.3 points of row 0's in all four emulated runs; where it moves, it rises." On `unrelated` under the sweep it is 92.39 at row 1,024 against 92.40. Replacement: "where it moves by more than 0.01, it rises".
- "History of the revisions", row 7: "Check 2 (d)". In revision 3 the check of the gate rows of the arm `shut` is 2 (c), and 2 (d) belongs to the arm `blind`. Replacement: "Check 2 (c)".

**Defect A-8. Language.**
- "The price of this form is known beforehand" and, in "For the coordinator", "The price is that a cell in reserve can read only what the first family does not present": figure of speech. Replacement: "The limit of this form is known beforehand" and "The limit is that".
- "In the arm `blind` the gates stand near one half on every item and say nothing": figure of speech. Replacement: "and carry no information about the use of the reserve".
- "Status": "the cells in reserve are blind in all their rows to what the first family's items present" uses *blind* before "Terms" defines it. Replacement: "the cells in reserve receive nothing, in any of their rows, from what the first family's items present".

**Found sound in A, beside A1 and A2.** The counts of runs (eight), of scored rows (twelve) and of checks (three) agree throughout; the runner's name, the commands, the names of the arms and the files of point 5 agree with "Reporting" and, but for the start files, with "Done"; the costs agree (0.55 CPU-hours a run, 4.5 for the eight, about 5 with the checks, a limit of 8). Points 5 and 6 of "The change" state the arithmetic of the arm `blind` rightly, the consequence for `conflict` included. The readings for one arm, rule and task cover every outcome, with the one exception of defect A-6. The arm `blind` shows 128 cells "open at some position" at every row, because its gates stand at one half; the brief says that this figure carries nothing there.

## B1. The brief of E57, revision 4: saving and resuming

The clause was read against `run` of `exp49_interface.py` at a36de661 (lines 164 to 243) and against its model, `exp52_resume.py` at f1d03869; both commits are in this clone.

**Is everything saved that a resumed run needs?** Yes, for the three networks of this job.

| What the run holds | How a resumed run has it |
|---|---|
| The weights | `m.state_dict()` in the saved state. The layer of these networks has one period (the brief's forward asserts `layer.K == 1`), so that `layer.added_parameters()` is empty and no trained tensor lies outside the module; the masks are built again from the seed |
| The optimizer's two running averages and its count of steps | `opt.state_dict()`. torch loads it by the position of each tensor in the groups, and refuses a state whose groups have other sizes. The brief builds the optimizer and applies `no_code_decay` before the state is loaded, which gives the saved order; if `no_code_decay` were left out on a resume, the loading would fail and would not go wrong unnoticed |
| The stream of batches and its digest | drawn again for the first u updates and fed to a new digest, which is asserted against the last scored row |
| The 16 batches of the cosines | drawn again from their own generator |
| The receipt so far, with `curve` and `cosines` | in the saved state |
| The losses since the last scored row | none: line 230 of the copied text empties the list at every scored row, and a state is written only there |
| The counters of CPU time | in the saved state |
| The first batch that E49 kept for its two cosines | not kept in this runner (point 4 of the brief) |

**Defects.**

**B1-1. A finished run can be resumed and rewritten.**
Place: point 7: "It is written at every multiple of `--ckpt` updates (4,096 if the option is absent; 0 writes none)" and "With `--resume` the run does not refuse the paths that exist, and without a saved state the option has no effect."
What is wrong: 24,576 and 49,152 are multiples of 4,096, so that a state is written at the last update, and nothing removes it. The rule for the worker starts every run with `--resume`; the command of a finished run, given again, then finds a saved state, refuses nothing, makes no update and writes the final weights, the answers file and the receipt again, with a new entry in `resumes` and new times. The two sentences on refusing also read against each other when files exist and no saved state does: the first says that nothing is refused, the second that the option has no effect. Both runners that the brief names avoid this: `exp52_resume.py` writes no state at the last update, deletes the state when the run has ended and asserts that the final weights do not exist (lines 138, 152 and 61); `exp47_local.py` does the same (lines 511 and 528) and, on a resume, still refuses the final weights and the answers file (line 432).
Replacement: "It is written at every multiple of `--ckpt` updates below the run's number of updates (4,096 if the option is absent; 0 writes none)" and, in place of the second sentence: "With `--resume` and a saved state present, the run does not refuse the receipt, the starting file and the saved state; it still refuses the final weights and the answers file, so that a finished run is not resumed. With `--resume` and no saved state the run is an ordinary start and refuses every path that exists. The saved state is deleted when the final receipt has been written."

**B1-2. Check 8 does not exercise the case that the long runs have, nor the run whose optimizer was regrouped.**
Place: check 8.
What is wrong: in the runs a state is written every 4,096 updates and the receipt is written, with a set of cosines, every 2,048. A process that dies in the second half of such an interval leaves a receipt on disk that is ahead of the saved state by eight scored rows and one set of cosines. In check 8 the states are 2,048 apart and the process dies at update 3,000: the receipt on disk then equals the receipt inside the state, and no set of cosines lies between the state and the death. A runner that took its receipt from the file and not from the state, or that appended a second entry of `cosines` for the same update, would pass. No state is written between two sets of cosines either, in the check or in the runs, as long as `--ckpt` is a multiple of 2,048 (B1-3). The check also uses the scale `row`; the one run with `--code-decay 0`, in which two tensors have changed their group and which has 49,152 updates, is not resumed in any check. The check can fail for the other things that a resume can get wrong: the weights, the optimizer's averages and steps, the stream, the digest, the generator of the cosines and the losses of the first row after the resume. It cannot fail for the counters of CPU time, which it leaves out of the comparison.
Replacement: "8. **A run that is killed and resumed equals an uninterrupted one.** Two dry runs of 8,192 updates of `in16` under the sweep with the scale `matrix` and `--code-decay 0`, each in a folder of its own: (i) with `--ckpt 0`; (ii) with `--ckpt 4096 --die-after 7000`, which ends with exit code 9, and then the same command with `--resume` in place of `--die-after 7000`, which goes on from the saved state of update 4,096. Before the second call the check records that the receipt on disk has its last scored row at update 6,144 and four entries of `cosines`, so that it is ahead of the saved state. The two finished receipts have the same `tensors_sha256` and the same `start_tensors_sha256`; their scored rows are equal in every field whose name does not begin with `cpu` or `wall`, the batch digest included; each has five entries of `cosines` (updates 0, 2,048, 4,096, 6,144 and 8,192), and they are equal; the optimizer has made 8,192 steps in each; `optimizer_read_back` equals `optimizer` in each, with `book.weight`, `l1.weight` and `l1.bias` in the group without decay; `cpu_updates` of the last row of (ii) is at least 1.5 times that of its row of update 4,096, which shows that the counter was taken over; the receipt of (ii) has one entry in `resumes`, for update 4,096, and that of (i) none."
By E49's receipt of `in16` under the sweep (0.111 CPU-seconds an update, 5.9 a scored row) the check as it stands costs about 30 CPU-minutes and this form about 45.

**B1-3. `--ckpt` is not tied to the rows at which the receipt is written.**
Place: point 7: "It is written at every multiple of `--ckpt` updates ..., at a scored row, after that row and its set of cosines are in the receipt and the receipt is written".
What is wrong: the receipt is written and a set of cosines is taken at multiples of 2,048 only (lines 231 and 232 of the copied text, and point 5). For another value of `--ckpt` the sentence describes nothing.
Replacement: add "The run asserts that `--ckpt` is 0 or a multiple of 2,048."

**B1-4. Point 4 says that the copy of `run` has the listed changes "and no others"; point 7 changes further lines.**
Place: point 4: "with the changes below and no others".
Evidence: point 7 changes line 172 (the refusal), line 190 (the starting file, which a resumed run must read and not write again), line 202 (the counters), line 221 (the scored row of update 0, which a resumed run does not repeat), line 222 (the first update of the loop) and the lines after 232 (the writing of the state, the death asked by `--die-after`).
Replacement: "with the changes below and those of point 7, and no others", and in point 7: "On a resume the starting file is read and not written again, and the scored row of update 0 is not made again."

**B1-5. The time of the checks was not set again.**
Place: "Rules for the worker": "The checks take about half an hour".
Evidence: check 8 alone is 13,240 updates and 54 scored rows, about 30 CPU-minutes by E49's receipt.
Replacement: "The checks take about an hour".

**B1-6. Two small things that `exp52_resume.py` has and the clause does not.**
The model saves the wall time so far, so that `wall` of a resumed receipt counts the whole run, and after loading it asserts the groups of the optimizer against those expected. Replacement, added to point 7: "The saved state also holds the wall time so far. After the two states are loaded the run asserts that `R42.opt_record(m, layer, opt)` equals the entry `optimizer` of the saved receipt, and that `layer.added_parameters()` is empty."

## B2. The brief of E53, revision 3: check 4

The check was read against `run`, `save_ckpt` and `check8` of `exp47_local.py` at 578a877b (lines 409 to 532 and 871 to 884).

- *The check can be carried out as it says.* The runner writes a state at every multiple of 1,024 updates below the run's number of updates, so that a run of 2,048 updates has one saved state, at update 1,024. The scored row of update 1,536 is printed and flushed when it is made; a process killed after that line has made between 1,536 and 2,047 updates, and the second call with `--resume` finds the state of update 1,024, loads the module, the layer's three tensors, the optimizer, the receipt, the counters and the first batch from it, draws the first 1,024 batches again into a new digest and asserts it against the last row. The two receipts can then be compared in their rows, in `tensors_sha256`, `start_tensors_sha256`, `data_digest` and `cosine_first_batch_end`, as the trial's check 8 compares them.
- *What the killed process leaves cannot make the resumed run differ.* The receipt is written at multiples of 4,096 updates and at the end (line 509), so that a run of 2,048 updates that is killed at update 1,536 leaves no receipt at all; it leaves the starting file, the saved state and its log. On a resume the runner never reads the receipt file: it takes the receipt from the saved state (lines 470 to 473) and the starting hash from the starting file. In the long runs the receipt on disk is ahead of the last saved state only if the process dies between the writing of the receipt and the writing of the state at the same update (lines 510 and 512), because the states are 1,024 apart and the receipt 4,096; and the receipt file is not read in that case either. The kill therefore tests what the stop of the trial's check did not: a death at a point where the losses since the last row and 512 or more updates are thrown away.
- *What the check does not reach.* The weights of updates 24,064 and 24,320 (change 7 of the brief) are written only in a run of 24,576 updates. A run that dies after update 24,064 resumes from the state of update 23,552 and writes them again; `save_weights` replaces a file and does not refuse one, so that this is safe by the text of the runner, and no check exercises it.

**Defect B2-1. There is no saved state of update 2,048 in a run of 2,048 updates.**
Place: check 4: "the scored row of update 1,536, which lies between the saved states of updates 1,024 and 2,048".
Evidence: `exp47_local.py`, line 511: `if u % CKPT_EVERY == 0 and u < dose`.
Replacement: "the scored row of update 1,536, which lies after the one saved state of this run, that of update 1,024, and before its end (the runner writes no state at the last update)".

## What could not be checked

- torch is not installed here. Nothing of the runners was executed; what is said about torch's indexing, about assignment to `.grad` and about the loading of an optimizer's state comes from reading.
- The emulation of the arm `blind` was validated as that of the arm `shut` was, on 64 updates of E51's arm `keyed`; its later rows cannot be compared with anything. Each run was emulated once; a second emulation with other rounding would give other figures at the later rows. The arm `shut` was not emulated beyond eight updates in this check, and not to 16,384 updates in the first.
- The replacement of defect A-2 (the whole change of the old rows projected) was not emulated.
- For B1 and B2 no run was made or emulated; the answers rest on the text of the three runners.
