# Task, part of E63: the tissue's program written so that it works only where something changes (19 November 2026, 03:51 UTC)

By the outside reviewer, on Mr. Xu's instruction to solve the speed of the program as a problem of program design. For the coordinator's release; the reviewer proposes E57's worker, whose line is free. It has no number of its own. Every figure in it was measured (`PROGRAM_DESIGN_EVENT_DRIVEN_20261119.md`); nothing in it is a forecast unless it says so.

## 1. What is wanted

One program for the tissue that every later step of the roadmap uses, in place of `TISSUE_STEP1/tissue_step1.py` of E63, which visits every connection at every step and costs 0.474 CPU-seconds a byte on the worker's host. The reviewer's kernels, on the real connections of E63's saved state and with shortened cell formulas, cost 0.27 ms a byte. The task is to write the whole tissue in that design and to show two things: that it computes the same tissue as E63's program when its shortcuts are switched off, and what the tissue does and costs when they are switched on.

## 2. What to read

On the reviewer's branch `claude/vibrant-cerf-sje726`, folder `PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/COORDINATION_20261117/`: `PROGRAM_DESIGN_EVENT_DRIVEN_20261119.md` (the design, sections 2 and 5); `event_driven_kernel_20261119/ev2.c` and `ev2_time.py` (the second kernel; a starting point, not a specification); `THE_TISSUE_DEFINED_REVISION_5_20261119.md` with the dated changes at its head (the definition); `TASK_STEP_1_THE_TISSUE_READS_THE_STREAM_20261118.md` (the brief of step 1: the stream, the birth, the readings). On `origin/claude/e63-tissue-step1`: `TISSUE_STEP1/tissue_step1.py` (the reference program), its checks and the records of stage A.

## 3. The program

A compiled core (C, called from Python) with these parts; if the host has no C compiler, report that at once and stop.

1. Connections stored once, sorted by receiving cell, with an index by sending cell; both kept correct when a connection is made or given up.
2. Each cell keeps its arrival from step to step. A sending cell passes on the change of its output since it last sent, when that change exceeds a tolerance. A work list holds the cells whose arrival or state is still moving; no other cell is computed at a step.
3. What only decays while nothing arrives is brought up to date when it is next needed.
4. The making of a connection is looked at when a cell's period has passed, not at every step; a cell whose neighbourhood is full is left out until one of its connections is given up. The draws of the life generator keep the order of E63's program.
5. Four switches, each a parameter: the output floor (0 or a value); the tolerance of point 2 (0 or a value); whether a trace is held for each connection or once for the sending cell; whether strengths are changed at every step or once a byte from a sum kept by the receiving cell.

## 4. Accepted when

**(a) With every switch off** (floor 0, tolerance 0, a trace for each connection, strengths at every step) the program is the definition as E63's program computes it. On seed 73129, for each of the four tissues of stage A over its first 2,048 bytes: the connections (sender, receiver, step of making), the emissions and the state of the life generator are identical to E63's program's; strengths, traces, states, thresholds and averages agree to within 1e-9, and the largest difference is reported. If a connection is made or given up at a different step, report the first such step and stop.

**(b) With the switches on** (floor 1/128, tolerance 1/1024, one trace for a sending cell, strengths once a byte): the first-form tissue, the coincidence tissue and the tissue whose strengths never change, seed 73129, 16,384 bytes each, with the readings of section 11.2 of the brief of step 1. These are set beside E63's stage A for the same tissues. They are not expected to be equal; the report says where they differ.

## 5. Report

At most fifteen lines to the reviewer by message, and a file beside the program: for (a), the result for each tissue; for (b), for each tissue and for both programs, CPU-seconds a byte, connections a cell at the end, the mean number of cells sending at a step, connection visits a byte, and the readings of section 11.2. Each of the four switches turned on alone on the first-form tissue over 4,096 bytes, with its cost and the same readings, so that the effect of each can be told apart. No figure is stated that the program has not measured.

## 6. Limits

2 CPU-hours; ask above it. At most three processes, one thread each. A branch of your own from the head of `origin/claude/e63-tissue-step1`; E63's files are not edited; push only to your own branch; the trailers of `PROJECT_DOCUMENTS/KB/AGENTS_POINTER.md`; no name of a language model in a file. Stage B is not part of this task. If a check fails, send a message at once.
