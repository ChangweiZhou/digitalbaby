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
