# What is to be done, derived from what a cell does (18 November 2026, draft 2, for Mr. Xu)

**Superseded (18 November 2026, about 22:35 UTC).** Mr. Xu: this list does not show how it leads to end-to-end persistent learning with generalisation; what was asked for is a step-by-step route with the dependencies and overlaps of the goals. The route is `ROADMAP_TO_END_TO_END_PERSISTENT_LEARNING_20261118.md`. The reading by surprise proposed below is not the route's reading: from its step 2 on, the tissue is read by the bytes it emits itself.

By the outside reviewer. Derived from Mr. Xu's statement of 18 November of what a cell does (`WHAT_A_NEURON_HAS_20261118.md`, section 8, his words), which is called *the statement* below, and not from the existing network. It starts nothing. Draft 1 is withdrawn: its items 1 to 3 had a cell that receives the answer, which no cell can.

## The statement, in short

There is no separate learner and no objective of the whole. Apart from the byte-level encoder, a cell receives input from other cells; it sends, from that input and its own state, an output to the cells it is connected to; and it keeps adjusting which cells it is connected to. It cannot read bytes that have passed; it has its stored state and the new information that reaches it through other cells. A cell of the encoder reads only scattered small passages of the input, different for each cell. What a cell does is to predict the new information that arrives from other cells.

So there is no output cell, no cell that is given an answer, no error sent back, no window of past bytes, and no item that begins and ends for a cell. Every cell has the same task, and it is its own.

## How anything is read by us

A cell's own quantities may be recorded: what it expected and what arrived. Beyond that, a tissue is read as a living learner is read: by what it does when it is shown something. Since no cell is an output, the tissue's knowledge of an answer is read by **surprise**: the same item is shown once followed by the right answer and once followed by a wrong one, and the mismatch between what the cells expected and what arrived is compared. A tissue that has learned the task is less surprised by the right answer. Taken over the possible answers, the least surprising one is the tissue's answer, which can be set beside the record's scores. Nothing is added to the tissue for this, and nothing is read by an operation that has no meaning for a living learner.

## The order

0. **The tissue written out exactly as the statement says, before any program.** Every cell does the three things of the statement and nothing else. The encoder's cells each read their own scattered passages. Wherever the statement leaves a choice, the choice is listed as ours, with its values on a grid of powers of two: the length of a passage and of a gap; how many cells a cell is connected to at the start; how many exchanges take place between two bytes; what a cell's state consists of; and the forms that "adjusting which cells it is connected to" and "predicting what arrives" can take with only what a cell has. A second reader then goes through the description line by line with two questions: is this in the statement, and can a living cell do it.
1. **Does a cell's prediction of what arrives get better as the stream is read?** For each form of item 0, on the stream of the first family, one byte after another with no reset. Read: each cell's own mismatch over time. If no form improves, nothing that follows can be asked.
2. **Does the tissue come to hold the task?** Read by surprise, on taught pairs and on pairs never taught, starting from the simplest forms of the task (one digit to be recalled; the larger of two; one-digit addition; the first family without letters; with letters). This is where generalisation is first read.
3. **Time and distance.** How item 2 depends on the number of exchanges between two bytes, and on how many connections lie between a cell and the encoder: what Mr. Xu calls relay, measured where it is the only way information reaches a cell.
4. **The connections that the cells arrive at.** Since adjusting its connections is part of what every cell does from the start, their number and choice are an outcome: read from different starts (2 connections a cell and 128), and whether they arrive at the same.
5. **Memory.** A second family after the first, with no store and no act of the program; the first family read again by surprise. One arm with the tissue's own repetition at rest (the tissue runs without input between visits), since Mr. Xu accepts some rehearsal.
6. **Beside the reference.** The tissue's answers by surprise against back-propagation's on the dense network as the record holds them: share right on never-taught pairs, items needed, computation. No new run of the reference.
7. **The numbers we fixed,** listed at every stage, and given to the cell where a cell can set them.

## What becomes of the present work

Nothing further is run with the sweep or the exact gradient, and E61 is not run. E53 and E62 finish, being paid for, and are filed with the earlier line. The reviewer's nine jobs and draft 1 of this file are withdrawn. The record's network, with its readout, its targets and its backward pass, is the reference and nothing else.

## What the reviewer arranges

By Mr. Xu's word the reviewer reviews and arranges the tasks of other agents; runs, verifications and releases are the coordinator's. Item 0 depends on the statement alone and on nothing in this list, so the reviewer has an agent write it now and a second agent check it. The remaining items are for discussion with Mr. Xu.

## History

Draft 1 (18 November 2026, about 22:05 UTC): eight items; items 1 to 3 withdrawn at 22:15 (a cell cannot see the answer). Draft 2 (about 22:25 UTC): rewritten from Mr. Xu's statement of what a cell does.
