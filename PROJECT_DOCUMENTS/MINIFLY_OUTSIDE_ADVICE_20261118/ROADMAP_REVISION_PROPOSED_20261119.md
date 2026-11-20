# The roadmap of 18 November: what I would keep, what I would change (proposed, 19 November 2026)

By the outside session, at Mr. Xu's question whether the roadmap (`ROADMAP_TO_END_TO_END_PERSISTENT_LEARNING_20261118.md`, the reviewer's) is useful. A proposal; it decides nothing. It rests on the audit and on the small case beside this file (`loop_small_case/REPORT.md`), which are development level.

## 1. Is it useful? Mostly yes

**Keep, as written.**
- Sections 1 and 2: persistent learning *is* the loop (the system gives out what it expects, the byte arrives, it is told, the cells correct themselves), and end to end, persistent and generalising are three aspects of that one loop. This is the best page in the record, and the small case is that page run.
- Section 3, the table of capabilities and what each depends on.
- Section 5, the conditions: one stream; a comparator on every new stream; never-taught instances set aside beforehand; smallest size first; a step not reached is reported and not passed over; the numbers fixed by us listed at every step.
- Steps 2 to 8 in their content: the ends correct themselves, the correction is relayed inward, kept, first generalisation as "a new combination of parts taught in other instances", the rest of the family, a life, cost.

**Change.**
1. **Step 0 and step 1 go.** Step 0 asked for the whole tissue to be defined on paper before anything ran; step 1 asked that it carry bytes with no correction, by a measure that is already complete when nothing changes. Together they took all the work and produced a tissue that does not learn (audit, sections 4.2, 4.3). The first step is the loop itself, at the smallest size.
2. **The tissue's properties are not a start but a list of things to add one at a time** under one end-to-end score: coordinates and neighbourhoods, sparse reading, making and withdrawing connections, one sign for each cell, several answering cells, feedback through dedicated cells, yes or no. Each is kept if the score holds beside the comparator, and reported if it does not.
3. **Generalisation comes before keeping, not after.** The roadmap puts keeping at step 4 and the first generalisation at step 5. The small case reaches the roadmap's own form of generalisation now (88% of never-shown two-digit sums), while keeping fails at the first attempt as it has in every line of the record. Keeping is the hard part and should not stand in front of what can be had.
4. **Every test of generalisation states why an unseen instance could be answered.** Single-digit sums were taken as the test on 19 November; the comparator fails them too.
5. **Keeping is taken in small steps,** each with its own measure, instead of as one step "reached when the first table is still answered": see step R5 below.

## 2. The order I would propose

| | Step | Reached when | State |
|---|---|---|---|
| R1 | The loop at the smallest size: cells in between learn from a mismatch relayed to them | never-shown combinations answered beside the comparator | reached at development level on two-digit sums; to be repeated by another line and verified |
| R2 | The same, as a cell has it, one item at a time: yes or no; sparse reading; one sign a cell; several answering cells; feedback through dedicated cells; connections made and withdrawn | the score of R1 holds within a stated margin after each item | yes or no: 9% at the same length, not reached; sparse reading: 55% with half the lines (one run) |
| R3 | Cells that hold bytes themselves, in place of the delay chain (a trace at each connection, running forward) | the score of R1 on a stream with varying distances | not started |
| R4 | The first family (task byte, letters, the modulus) beside the record's reference | never-taught pairs at the reference's level | not started |
| R5 | Keeping, in parts: (a) what is lost after a second task and how fast it returns; (b) how little rehearsal suffices; (c) rehearsal that the system makes itself; (d) writing a new task where the old one left room | each part has its own number | (a), (b) measured once (`loop_small_case`); (c) two first forms failed; (d) open |
| R6 | A life of several families; then cost beside the comparator | as the roadmap's steps 7 and 8 | not started |

R2 to R4 do not depend on R5 and can run beside it.

## 3. States on 20 November

Entered after the work of 20 November (`loop_small_case/REPORT.md`, sections 10 to 16). The proposal above is unchanged; these are the states of its steps. Development level throughout.

| | Step | State |
|---|---|---|
| R1 | The loop at the smallest size | reached at development level on two-digit sums; to be repeated by another line, which now takes minutes |
| R2 | The same as a cell has it, one item at a time | against 84% with every line read: sparse reading with rewiring 81% with 64 lines and 43% with 32; one sign for each cell 71%; the relay over two steps 84%; cells that are silent most of the time about 70%. Yes or no alone: 9%, not reached. Several answering cells and feedback through dedicated cells: not started |
| R3 | Cells that hold bytes themselves | reached in two of three parts. One digit across a gap of varying length: 91% and 99% with one crossing between cells (five seeds), and as good at gaps never taught. Two-digit sums: 66% of never-shown sums with two delay lines in place of nine and the crossing (five seeds); not learned with none. The two together (two-digit numbers with letters between them): 8% on the development seed, not reached |
| R4 | The first family | not started. Beyond R3 it needs two-digit numbers with letters between them, the task byte, and the reduction by the modulus. Two-digit numbers with letters between them were tried once (development seed): 8% of never-shown sums, not reached |
| R5 | Keeping, in parts | (a) what is lost and how fast it returns, (b) how little rehearsal: measured on five seeds; (c) rehearsal the system makes itself: two forms failed; (d) writing the new task elsewhere: works when the program chooses where, with one old line in 32 (67% kept against 16%); a way for the cells to choose is open |
| R6 | A life; cost | a run costs a fifth to a twelfth of what it did; cost beside the comparator is not measured |

Two things I would add to the proposal in the light of the day. First, the item "cells that are silent most of the time" belongs in R2 and is in the table. Second, R3 was written as one step reached "on a stream with varying distances"; it turned out to have three parts that were reached separately and in a different order than I expected: holding where nothing has to be bridged (a cell's own path is enough); holding across a gap of varying length (needs a signal passed back over one connection between cells); and two-digit numbers (need, besides, the byte or two just past on a cell's lines).
