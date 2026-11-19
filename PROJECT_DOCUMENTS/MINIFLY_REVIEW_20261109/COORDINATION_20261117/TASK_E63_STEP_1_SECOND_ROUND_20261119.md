# Task, part of E63: step 1, second round: one change at a time (19 November 2026, 04:30 UTC)

By the outside reviewer. For the coordinator's release once the tissue's program in the event-driven design (`TASK_E63_THE_TISSUE_PROGRAM_EVENT_DRIVEN_20261119.md`) has been accepted; it runs on that program and on no other. It has no number of its own. Figures of cost are left open and are filled in from that program's measured cost; none is forecast here (rule 5b of `TASK_GRAPH_20261119.md`).

## 1. Why

Stage A of E63 (report at 040d3cf0 on `claude/e63-tissue-step1`; figures as the worker reported them, to be verified by the coordinator) did not reach step 1 of the roadmap with either first form. Under the first form the cells two and four connections from the input lines are constant or silent; under the coincidence form they are active and carry almost nothing of the byte (0.0375 and 0.0264 for the present byte against chance 0.0217 and a level of 0.1103). By the roadmap's rule a step that is not reached is not passed over: one choice of the definition is changed at a time and the step is read again.

## 2. The changes, each alone and then together

Each is a number or a form that is already known, from measurement, to be at fault. Everything not named stays as in stage A.

| | Change | What is known |
|---|---|---|
| F | An output below 1/128 is zero | All thresholds but one are zero and decaying states never reach zero, so half the tissue counts as active by residues (design note, section 1) |
| T | The target of a cell's activity is 1/32 and then 1/128, in place of 1/8 | No cell reaches 1/8; the target only drives every cell to connect to its whole neighbourhood (91.8 connections a cell in stage A) |
| W | A connection is withdrawn when what arrives over it has been of no use to the receiving cell (Mr. Xu, `WHAT_A_NEURON_HAS_20261118.md`, section 19), in a form that is zero for a silent cell: the running product of the input's trace and the receiving cell's activity, brought up to date only when the cell is active; withdrawn when it stays below 1/128 of the cell's largest for 128 steps in which the cell was active | Under the rule by strength alone connections are made and withdrawn at about the same rate with nearly every neighbourhood full |
| R | The strength rule: the first form and the coincidence form, as in stage A | Each failed in its own way in stage A; they are read again under F, T and W |

**The tissues.** Seed 73129, 16,384 bytes each, the stream and the readings of stage A (section 11.2 of the brief of step 1) and the level 0.1103. For each of the two strength rules: stage A's settings on the new program (the reference of this round); F alone; T alone at 1/32 and at 1/128; W alone; F with T at each value; F with W; T with W at each value; F, T and W together at each value of T. That is 13 tissues for a rule and 26 in all, with the one tissue whose strengths never change.

## 3. What is read

For every tissue, at the end and in the two windows of stage A: connections a cell; the share of cells at one, two and four connections from the input lines that are neither silent nor constant; the score for the present byte and for the bytes one and two positions back at those distances, beside what the present byte alone gives; the mean number of cells sending at a step; the cost a byte.

**Step 1 is reached in this round** by a tissue that meets the brief of step 1's rule for stage A's first outcome at both distances two and four. If one or more tissues do, the coordinator is asked to release the full run of step 1 (three seeds, 262,144 bytes) for the simplest of them, the one with the fewest changes. If none does, the report says for each change what it moved, and the next choices are opened in the order of the definition.

## 4. Report and limits

A table of the 27 tissues with the readings above, and at most fifteen lines to the reviewer by message. No figure that the program has not measured. Budget: to be entered from the program's measured cost before release; at the reviewer's kernel's cost the whole round is under one CPU-hour, and that sentence is an expectation and not a measurement. At most four processes, one thread each. Wait in commands of at most two minutes. If a check fails, send a message at once.
