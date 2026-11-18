# Check of the brief of E55, first part, revision 3, and of two clauses on saving and resuming in the briefs of E57 and E53

18 November 2026, begun 02:50 UTC. Written for the outside reviewer by the agent that checked revision 1 of the brief of E55 (`CHECK_OF_TASK_E55_BRIEF_20261118.md` beside this file). The agent was given the briefs and the files they build on, and not the author's reasoning. The texts checked are those of commit 1c7e515a of the reviewer's branch. The check is a reading and a recomputation with numpy; no code of another branch was run and torch was not used.

**Verdicts.**

- **A, the brief of E55, revision 3:** the verdict is not yet given. The reading of the text is complete and its defects are listed below; the emulation of the arm `blind`, on which the first defect rests, has reached row 2,048 of 16,384 and is running.
- **B1, the brief of E57, revision 4 (point 7 of the runner, check 8, the rule for an interrupted run): release after the listed changes.** What is saved is enough for a resumed run to equal an uninterrupted one. Six things in the wording and in check 8 should be changed; none touches a run.
- **B2, the brief of E53, revision 3 (check 4): release as it stands, with one sentence corrected.** The check can be carried out and compared as it says, and no file that the killed process leaves can make the resumed run differ.

The script is `check_e55_brief_r3.py` beside this file; it imports the script of the first check. Its parts are named with the figures. Figures are given as sweep; back-propagation.

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
| (d) answers of the 32,884 that differ | 0; 0 | | 0 |

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

**The emulation of the arm `blind`** (part `blind_emulate`) is the checker's own arithmetic and is not the runner; its later rows show sizes and the order of events. It was started for 16,384 updates of each task under each rule. At row 2,048 it shows the following, and the table is entered here when the runs have ended.

- The remainder of the projection at the rows kept blind does not grow: the largest entry of `D @ Vm` is 1e-8 to 2.4e-8 at every row so far. The brief's statement on this point holds.
- The state of the reserve on the first family's items grows over the positions of an item, and the growth sets in over the updates. The largest eigenvalue of the map above passes 1 between rows 64 and 1,024 in all four runs (1.32 on `unrelated` under back-propagation at row 2,048, 1.16 on `conflict`; 1.05 and 0.96 under the sweep). On `unrelated` under back-propagation at row 2,048 the largest state of a cell in reserve on a held-out pair is 1.4e-7 at position 1, 6.2e-6 at position 11 and 1.1e-4 at position 21 (3.1e-4 in single precision), and the first family's outputs have moved by 6.0e-3. No answer of the 32,884 has changed in any run so far.
- `unrelated` under back-propagation is learned at row 2,048 (99.0; 0.0 with the cells in reserve held shut).

### A2. `Rule55`

The code of `Rule55` for the arm `blind` does what points 5 and 6 of "The change" say. `before` first projects the gradient of the rows kept blind (`W.grad[rows]` with an index tensor gives a copy, and the assignment to `W.grad[rows]` writes it back in place); `R51.Rule.before` then copies each whole gradient through double precision, which leaves those rows as they are, projects the old rows, and takes its snapshot of the weights before the optimizer's step. `R51.Rule.after` projects the step of the old rows from that snapshot and restores `l1`, the old columns and the bias of `l3` and the rows of the byte table; `make_blind` then projects the whole of each row kept blind and `zero_blocks` sets the named entries to zero. The emulation follows this order. The arm `shut` is what the check of revision 1 examined, with three differences that the brief states: the two bias vectors are set to zero as well, the two blocks are set to zero in the start and not only after each update, and the runs have 16,384 updates. Under back-propagation eight emulated updates of the arm `shut` of revision 3 give the weights of the check of revision 1 to 1e-13; under the sweep they differ by up to 1.9e-4 after eight updates, because the start of revision 1 still held the rounding remainders whose signs the sweep reads at the first update (finding 6 of the first check).

### A3 and A4. Defects of the text

(The first defect, on the growth of the state of the reserve in the arm `blind`, is written when the emulation has ended.)

**Defect A-2. "Done" counts two start files; there are four.**
Place: "Done": "The job is done when the two start files with their receipts".
Evidence: "The runner", point 5: "the four start files (two forms, two rules) with their receipts"; the command `start <blind|shut> <S|B>`.
Replacement: "The job is done when the four start files with their receipts".

**Defect A-3. One sentence of the two arms side by side says what the brief forbids elsewhere, and two say more than two runs show.**
Place: "How the result is stated", "The two arms side by side": "If the arm `blind` keeps the first family exactly and learns `unrelated`, the report says that cells which receive nothing from the first family's items sufficed for a task on bytes that the first family does not use." And: "If the arm `shut` learns a task that the arm `blind` does not, the report says that reading what the first family presents was needed for that task on this seed".
What is wrong: in the arm `blind` the rows of the old cells and of the old readout units also change, away from the directions in use, and the weights from the cells in reserve into the old units are free. The verification of E51 found for `keyed` that the unrelated task is at 0.0 and 0.3 when the incoming connections of the old units are put back. "Sufficed" gives to the cells in reserve what the run does not separate, which is what the bullet "The score with the cells in reserve held shut" forbids for "learned in the reserve". The sentence is also licensed when the score with the cells held shut is at 80 or more, and it has no form for a run that is *kept* and not *kept exactly*. In the second sentence the two arms differ in more than what a cell in reserve may read (the gates of the arm `shut` start shut by a bias and those of the arm `blind` stand at one half), and one pair of runs shows a difference and not a need.
Replacement: "If the arm `blind` learns `unrelated`, with a score below 80 when the cells in reserve are held shut, and the first family is kept exactly or kept, the report says that the task was learned and the first family kept (exactly, or with its largest d(g)) in a network whose cells in reserve receive nothing from the first family's items; it does not say that these cells sufficed, because the rows of the old cells and of the old readout units change as well." And: "If the arm `shut` learns a task that the arm `blind` does not, the report says that on this seed the task was learned by the arm in which a cell in reserve may read what the first family presents and not by the arm in which it may not, names the other difference between the arms (the gates shut by a bias in the one and at one half in the other), and gives beside it what the arm `shut` lost of the first family."

**Defect A-4. A run that is partly kept may have no row at which d(g) exceeds 31.**
Place: "How the result is stated": "In both cases it gives the row at which d(g) first exceeds 31".
What is wrong: *kept* has three conditions (d(g), the held-out answers, the reduction score), and a run is partly kept when any one fails.
Replacement: "In both cases it gives the first row at which a condition of *kept* fails, and which condition".

**Defect A-5. Three figures or references are out of date.**
- "What the record says", the emulation of the arm `shut`: "(1 − z at most 5.2e-5)". The check's document has 5.5e-5 (its table, `conflict` under the sweep at row 4,096). Replacement: "(1 − z at most 5.5e-5)".
- The same paragraph: "The reduction score of the first family stays within 1.3 points of row 0's in all four emulated runs; where it moves, it rises." On `unrelated` under the sweep it is 92.39 at row 1,024 against 92.40. Replacement: "where it moves by more than 0.01, it rises".
- "History of the revisions", row 7: "Check 2 (d)". In revision 3 the check of the gate rows of the arm `shut` is 2 (c), and 2 (d) belongs to the arm `blind`. Replacement: "Check 2 (c)".

**Defect A-6. Language.**
- "The price of this form is known beforehand" and, in "For the coordinator", "The price is that a cell in reserve can read only what the first family does not present": figure of speech. Replacement: "The limit of this form is known beforehand" and "The limit is that".
- "In the arm `blind` the gates stand near one half on every item and say nothing": figure of speech. Replacement: "and carry no information about the use of the reserve".
- "Status": "the cells in reserve are blind in all their rows to what the first family's items present" uses *blind* before "Terms" defines it. Replacement: "the cells in reserve receive nothing, in any of their rows, from what the first family's items present".

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
- *What the killed process leaves cannot make the resumed run differ.* The receipt is written at multiples of 4,096 updates and at the end (line 509), so that a run of 2,048 updates that is killed at update 1,536 leaves no receipt at all; it leaves the starting file, the saved state and its log. On a resume the runner never reads the receipt file: it takes the receipt from the saved state (lines 470 to 473) and the starting hash from the starting file. In the long runs the receipt on disk is never ahead of the last saved state, because the states are 1,024 apart and the receipt 4,096. The kill therefore tests what the stop of the trial's check did not: a death at a point where the losses since the last row and 512 or more updates are thrown away.
- *What the check does not reach.* The weights of updates 24,064 and 24,320 (change 7 of the brief) are written only in a run of 24,576 updates. A run that dies after update 24,064 resumes from the state of update 23,552 and writes them again; `save_weights` replaces a file and does not refuse one, so that this is safe by the text of the runner, and no check exercises it.

**Defect B2-1. There is no saved state of update 2,048 in a run of 2,048 updates.**
Place: check 4: "the scored row of update 1,536, which lies between the saved states of updates 1,024 and 2,048".
Evidence: `exp47_local.py`, line 511: `if u % CKPT_EVERY == 0 and u < dose`.
Replacement: "the scored row of update 1,536, which lies after the one saved state of this run, that of update 1,024, and before its end (the runner writes no state at the last update)".

## What could not be checked

- torch is not installed here. Nothing of the runners was executed; what is said about torch's indexing, about assignment to `.grad` and about the loading of an optimizer's state comes from reading.
- The emulation of the arm `blind` was validated as that of the arm `shut` was, on 64 updates of E51's arm `keyed`; its later rows cannot be compared with anything.
- For B1 and B2 no run was made or emulated; the answers rest on the text of the three runners.
