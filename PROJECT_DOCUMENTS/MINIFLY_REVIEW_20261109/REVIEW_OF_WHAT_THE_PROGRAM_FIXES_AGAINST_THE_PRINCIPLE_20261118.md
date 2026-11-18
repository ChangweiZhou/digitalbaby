# What the program fixes, set against what a neuron has: a second review from the principle (18 November 2026, revision 2)

By the outside reviewer, at Mr. Xu's instruction of 18 November (about 20:50 UTC): the places that we fixed and never examined against the principle are all to be examined. It is written for Mr. Xu, Mr. Zhou and the coordinator. It starts no job and takes no decision.

## 0. Why this review exists

On 18 November Mr. Xu asked why a sparse input is not learned although the cells are connected with each other. The answer is in the forward pass: between two bytes of an item the cells exchange what they hold exactly once. The reviewer had read that code many times and had never asked whether it agrees with the principle. Mr. Xu's judgement is that such a question, and not the execution of programs, is what the reviewer is engaged for. He is right, and the fault is the reviewer's.

The earlier reviews (10, 11, 15, 17 and 18 November) asked whether each experiment was run correctly and whether its conclusion followed, and they compared two things with the principle: whether connections were limited, and whether the rule was local. They took the network and the training procedure as given. A structure that every experiment shares could not be found that way, however often the results were reviewed.

This review therefore does not start from the results. It goes through the source of the network and of its training, line by line, records every structure, quantity, schedule and act that the program fixes or supplies, and asks of each: does a neuron have this? Two controls are added. A fresh agent that has seen neither this list nor any document of the project makes its own inventory from the source alone, and the two are compared (section 8). The outside audit of 18 November (`origin/claude/friendly-darwin-8mmhfm` at ae1816b8) is entered where it found what this review also finds or what it missed.

## 1. The principle

In Mr. Xu's words of 15 to 18 November, as the reviewer has recorded them; the numbering is the reviewer's.

- **P1.** A neuron has what arrives over its own sparse connections and its own state, and changes its own connections from these.
- **P2.** Nothing delivers to it a quantity computed from the whole network.
- **P3.** A single global backward pass is prohibited. A signal is relayed between neurons step by step (1 → 50 → 50² → 50³), so that a few passes cover the network.
- **P4.** A code that the learner formed itself is the learner's own information; what the program does by reading the item is ours.
- **P5.** Long-term memory forms in the course of learning and not by an act of the program; some rehearsal is acceptable (requirement A). Slower at first and ahead later is acceptable; fewer samples for a similar result (B1, B2). Learning is local also at the input and at the output (C).
- **P6.** The fewer numbers fixed by us, the better.

## 2. What was read

The network of E50's stage 3 and of E62 (`e50_sparse_code.forward`, with `e42_cells.Layer`, `rec` and `MaskedSign`, `e49_interface.Interface` and `masked`), the dense network from which it derives (`arch26.forward`, `arch20.NetRNN`), the harness (`exp3_state.SignLinear` and `lin`; `exp11_cell.render`, `stream`, `batches`, `respond`; `exp45.batch_tensor`; `exp26_candidate.step` and `make_opt`; `exp16_integrated.adamw16`) and the runner of a life (`smallworld/sw_run.py`: `Life.update`, `batches_with`, `capacity`). All on the default branch. Line numbers are those of the files there.

## 3. The inventory

*Examined* means that the record holds an experiment or a measurement that varied or tested the item, or a stated limit; *never* means that it was taken as given.

### A. Time

| | What the program fixes | Does a neuron have it? | Examined? |
|---|---|---|---|
| A1 | **One exchange between cells for each byte.** At each position the new state of a cell is computed once from the current byte and from the states of the cells it reads at the position before (`e50_sparse_code.py`, lines 56 to 60; the same in `arch26.py` and `e42_cells.py`). What a cell does not read itself can reach it only one cell further at each byte. | No. Between two inputs a neuron and its neighbours exchange signals many times; P3 presupposes it. | **Never,** until Mr. Xu's question of 18 November. |
| A2 | **The answer is due one position after the last digit.** The first answer byte is read out at the position of `=`, directly after the second digit of the second operand (`exp11_cell.py`, line 231; `exp45.py`, line 168). With A1, that digit has made one crossing between cells when the first answer byte is read. | It is the task's format, chosen by us; nothing in the principle sets it. | **Never.** |
| A3 | The error is formed after the item has ended and is passed back over its positions (the sweep's third part). | No. | Yes: named in the chain of steps; E61. |
| A5 | The cell is a gated unit of a fixed form (three gates, fixed equations), and every cell is updated at every byte. | The form is ours (P6). | Named as a limit (memo, 22.10). |
| A4 | The state of every cell is set to zero by the program at the start of every item (`e50_sparse_code.py`, line 55), and the 64 items of an update are 64 separate copies of the network run side by side. Nothing is carried from one item to the next but the weights. | No. A neuron's state is not reset from outside, and it meets one item after another. | Named once as a limit (memo, section 22); **never varied.** |

### B. What a cell reads, and what reads a cell

Weights in use in E50's stage 3: input map 35,328; connections between cells 8,448; the cells' biases 1,536; the readout 526,081 (`l2` 262,144, `lm` 131,072, `l3` 131,584, with their biases). The readout holds 92.1% of the weights in use and is 11.6 times the size of everything that belongs to the cells.

| | What the program fixes | Does a neuron have it? | Examined? |
|---|---|---|---|
| B1 | **The readout is a dense network of two layers.** Each of its 512 hidden units reads all 256 cells at every position (`lm`), and 257 output units read all 512 hidden units (`l3`) (`e50_sparse_code.py`, lines 65 to 67). | No (P1; requirement C names the output). | Stated as a limit in the briefs. One run with 8 cells a readout unit (E49's `out8`: 74 and 71 under the exact gradient, 10.6 and 7.8 under the sweep), in which `l2` and `l3` stayed dense. **Its weight, and what it computes, never.** |
| B2 | **The current byte goes to the readout directly, past the cells.** `z2 = tanh(l2 z1 + lm h)`: all 512 values of the current byte's code enter every hidden unit of the readout through `l2`, the largest map of the network (45.9% of the weights in use). In the experiments on a limited input only the cells' input was limited. | No. | **Never.** It stands in a formula of the memo (section 22) and nowhere else. |
| B3 | The hidden layer of the readout is non-linear and reads everything, so that a part of the computation of an answer can be done there and not in the cells. How large that part is has not been measured. | — | **Never.** |
| B4 | Every cell reads one value of every used byte (stage 3; E61; E62), or the whole code (E42, E54). A byte reaches every cell without any crossing. | No (Mr. Xu, 16 November: a cell cannot see the whole input). | Yes: E49, E50, E57. With 16 of 512 values a cell, a given byte reaches 30% of the cells directly and 98% after one crossing, which by A1 and A2 is the only crossing available for the last digit. |
| B5 | Each cell reads N cells, fixed at the start, from a neighbourhood or from all cells; N = 10. | The form yes; the number is ours (P6). | Yes: E42, E54; E62 is running. |
| B6 | The code and the wiring of stage 3 are drawn for the 46 symbols of the first family (`e50_sparse_code.py`, line 24; `exp50_code.draw_one`). The network has no input for any other byte. | The program's knowledge of the items, built into the wiring (P4). | Found by the outside audit; **never** by the reviewer. |
| B8 | A run is refused if a byte that the program lists as needed by the task reaches no cell (`exp50_code.run`), and the size of a turned-on value of the code is computed from the dense network's starting code. | The program's knowledge of the task (P4). | **Never.** |
| B7 | The 257 output units are normalised together: the error of an output unit contains the sum over all of them (softmax with cross-entropy). | A quantity of the whole output layer (P2), though of one layer only. | **Never.** |

### C. The error

| | What the program fixes | Does a neuron have it? | Examined? |
|---|---|---|---|
| C1 | The right answer byte is given to the output at the three answer positions, and to no other position (`exp3_state.py`, line 162). | It is the task's teaching signal. | A choice of 3 November, not revisited. It fixes the distances that A3 must cover. |
| C2 | While learning, the right previous answer byte is fed as the next input, whatever the network answered (teacher forcing; `exp45.batch_tensor`). | Supplied by us. | **Never** named. |
| C3 | The error returns over the forward connection with the sign of its weight: from a cell to each cell that it reads. | No. Of the 2,560 connections between cells in the record's network, 520 (20%) have a connection in the reverse direction; over the other 80% the error travels where no connection exists. | Named; E31; step 4. **The count never.** |
| C4 | One scale for a whole matrix: the mean magnitude of all its weights (`exp3_state.py`, line 100; `e42_cells.sign_scale`). | No (P2). | Yes: E57 is running. |
| C5 | Through the dense readout the error reaches every cell directly at every answer position, before any relay. | No. | Stated in E61's brief; weighed by the outside audit. |

### D. The change of a weight

| | What the program fixes | Does a neuron have it? | Examined? |
|---|---|---|---|
| D1 | **Each update averages the errors of 64 items** (192 answer positions; 48 items in a life). A weight never changes from one item alone. | No. | **Never varied.** |
| D2 | AdamW: for each weight two running averages; a step of about the rate whatever the size of the gradient. | The averages are a weight's own. The equal step is ours. | Once, on the dense network: plain steps lose the first family as fast (memo, section 24). Not on a network that leaves connections idle. E55 showed the cost: states of 0.001 behind a shut gate changed answers. |
| D3 | Every weight shrinks by the same factor at every update, used or not (0.25 times the rate). | A leak is local; that it is equal for all and blind to use is ours. | Yes: E46, the lives without decay; with it, forgetting is a step (memo, 33.27). A decay that depends on a connection's use: **never.** |
| D5 | The optimizer's correction of its running averages uses one count of updates shared by all weights, and its averages are carried through a whole life (a new optimizer only at the start of a life). | The count is a clock of the whole network. | **Never.** |
| D4 | One rate (0.001) for every weight of every map. | Ours (P6). | The rate: E52. Different rates for different cells or maps: planned (old item 13), **never run.** |

### E. The items and the life

| | What the program fixes | Does a neuron have it? | Examined? |
|---|---|---|---|
| E1 | The pairs of a family are drawn in shuffled order with fresh letters (`exp11_cell.stream`); a life is a sequence of visits of 256 updates to one family. | The world's, ours to set. | The order within a visit: **never.** |
| E2 | The store: the program keeps old items as they were presented and presents them again (half of each batch). | Rehearsal by the program; accepted in measure (P5). | Yes. |
| E3 | Births: a learner starts its life after 24,576 or more updates on the first family, shuffled. | Ours. | Yes (E59, E60). |
| E4 | Cells added, rows named, directions computed and projected at a change of task (E51, E55). | Acts of the program (P4). | Yes: decision 14. |
| E5 | A birth inside a world ends when the program finds a score of 0.8 reached; the rule may be changed between birth and life. | Ours. | E60 for the second; the first **never.** |

### F. Numbers fixed by us (P6)

256 cells; 512 values of the code and 512 hidden units of the readout; N = 10; the neighbourhood of 48; 11 values a byte; 46 values a cell; the rate 0.001; the averages' constants 0.9 and 0.98; decay 0.25; 64 items an update; 0 to 12 letters; the scale of the start (the square root of 256 / (N + 1)); 24,576 or 49,152 updates; 256 updates a visit; half the pairs taught; the store's size and share. Of these the record has varied N (in part), the rate, the decay, the share of pairs and the store.

## 4. The findings, by weight

**1. Relay has no time (A1 with A2). Never examined; in the reviewer's judgement the heaviest.** P3 describes a signal that crosses many cells between two events. In every network of the record a signal crosses one cell for each byte, forward and backward, and the answer is due one byte after the last digit. Three results that the record reads otherwise are what this structure would produce:

- *The sparse input (E49, E50's stage 2).* A cell that does not read a digit can have it one byte later, from a cell that does. For the last digit that is the only crossing before the answer. The record explained the failure by the number of bytes a cell has an input for, and stage 3 gave every cell an input for every byte. That removed the need for relay; it did not test it.
- *The network with an interface at a boundary (E47; E53, running).* A cell two or three crossings from the cells that are read cannot hold the last digit when the answer is due, and what it computes needs as many bytes again to return. The record found the cells beyond the first ring unused "before any training" and explained it by the size of the starting weights; E53 enlarges the start. By A1 and A2 those cells could not serve the last operand at any size of the start.
- *E61.* One step back in position and one step between cells are the same step only because of A1. The outside audit reached the same point from the other side (its section 3.6: one crossing takes one byte, forward as well as backward).

These are deductions from the source and the format, not measurements. The test is direct: let the cells exchange R times for each byte (R = 1, 2, 4, 8; the same weights in every round; the byte held; the readout after the last round), on the input that failed (16 values a cell) and on E47's network. No such run exists.

**2. The network called sparse is a small sparse part inside a large dense one (B1, B2, B3). The weight of this was never examined.** Nine tenths of the weights are a dense network of two layers that reads every cell and, through its largest map, the current byte itself. Every result on "the network with limited connections" (E42, E49, E50, E54, E57, E61 as briefed, E62) is a result on this compound. What the cells compute and what the readout computes has not been separated. Requirement C at the output has one run on record, which failed under the sweep. Read-only measurements on the saved networks would say much at once: the scores with the direct path `l2 z1` replaced by its mean over items; and how far the hidden layer of the readout, and not the cells, combines the operands.

**3. The learner never meets one item after another (A4, D1). Never varied.** Its state is reset by the program, and each change of a weight is an average over 64 items. Both bear on requirement A (what an update for one item does to another; what a state could carry) and on B2 (samples are counted in items, and 64 are spent for one change). The record cannot say whether either matters.

**4. What the outside audit found and the reviewer had not (B6; the audit's 5.2 and 5.3).** The stage-3 network has no input for a byte outside the first family's 46, so that steps 8 to 10 of the chain cannot be run on it. And the reviewer's own memo (section 28.6) already records that the network with 10 connections loses the first family within 64 updates, as the dense one does; step 8 of the chain nevertheless asks "whether sparse connections alone change what is kept" as if it were open. The audit is right on both.

**5. Smaller, and stated for completeness.** The output's normalisation (B7) and teacher forcing (C2) were never named. A decay that depends on use (D3) and rates that differ between cells (D4) were named and never run. The equal step of AdamW (D2) was examined on the dense network only.

**6. The working rule itself, which the record names and Mr. Xu keeps for now.** The independent inventory (section 8) puts first what the record treats as settled: under the sweep as under the exact gradient, the error is one pass backward over all maps and all positions of an item, which P3 prohibits; the sweep differs in the sign and the scale at each crossing. One fact is new: four fifths of that pass's crossings between cells go where no connection exists (C3). Finding 1 bears on this. If the cells exchange several times for each byte, an error could be relayed while the item is read, over connections that exist, and no pass after the item would be needed; that is the form P3 describes. The record holds only an early rule that carried its quantities forward (E19: 0.63 to 0.79), given up on 6 to 10 November for the pass backward (memo, section 33.24). No such rule has been tried on a network with limited connections.

**What was examined and stands as examined:** the sweep's three parts (A3, C3, C4), the number and the choice of connections between cells (B5), the rate, the decay, the share of pairs, the store, the births, and the program's acts at a change of task.

## 5. What becomes of the record's conclusions

| Conclusion on record | It holds under | What it does not show |
|---|---|---|
| A sparse input is not learned unless every cell has an input for every byte (E49, E50) | one exchange a byte and an answer due at once | that relay between cells cannot carry an input |
| The cells beyond the first ring are not used (E47) | the same | that cells away from the interface cannot be brought into use |
| Ten connections a cell cost 2.3 to 2.7 times the updates (E42); fewer pairs are not enough (E54); the curve against N (E62, running) | a dense readout that reads every cell and the current byte, and every cell reading every byte | anything about a network that is local at its input and output |
| The first family is lost within some tens of updates with limited connections as with dense ones (E43) | every item driving every cell; a reset state; updates of 64 items; equal steps | that limited connections cannot help when the input leaves cells idle |
| The sweep is slower than the exact gradient by a factor of about two (E26 to E52) | all of the above | the cost of locality: neither arm is local |

No verified figure changes. What changes is what each may be quoted for.

## 6. What the reviewer proposes

For the coordinator and Mr. Xu to decide. In order.

1. **Give relay time, before anything else is read on sparse inputs or outputs:** the runs with R exchanges for each byte described in finding 1, on the input with 16 values a cell and on E47's network, under both rules, beside the record. The recurrent part is a small share of the cost of an update, so that R = 8 is cheap. The reviewer writes the brief on your word; by decision 45 it would name the point at which the worker waits.
2. **Measure the readout on the saved networks, read-only, now** (finding 2), and add the readout to step 4 of the chain as its third part, as the outside audit also asks.
3. **Hold E61 in either form until item 1 is read.** With several exchanges for each byte a position and a crossing are no longer the same step, and the network of E61 changes. An agent is drafting revision 3 in the audit's form (a cell's own path kept, the crossings limited); it goes back to you, not to a worker.
4. **Let E62 finish.** It is bounded and running; its curve is quoted with the limit of finding 2.
5. **For requirement A,** take the audit's three changes to the chain (its section 5.2): read step 4 also by what an item of another family leaves idle; let step 8 cite E43's second addendum; do not expect steps 2, 3 and 6 to bear on memory. Step 7 (where an experience is written) is read-only and can be made now on the saved networks.
6. **One arm each, when the candidate network exists,** for one item an update with the state carried over, and for plain steps (findings 3 and 5).
7. **Two measurements that need no new code,** named by the independent inventory: the score with the cells silenced (`silence=True` exists in the forward pass), which gives what the direct path alone answers; and, as a second form of item 1 that leaves the network unchanged, m neutral bytes between `=` and the answer (m = 0, 1, 2, 4, 8), with the readout restricted.

## 7. How such an omission is prevented

From now on every brief that the reviewer writes or checks carries a section "What is taken as given", with the rows of section 3 that apply to its network and, for each, whether a neuron has it. A new network is not briefed before its table exists. The check of a brief by a fresh agent includes an inventory made from the source without sight of the brief's own list.

## 8. Limits, and the comparison with the independent inventory

**Limits.** The rows were read from the source by the reviewer; nothing was run but two counts (the shares of B4 and the reciprocal connections of C3). Findings 1 and 2 are deductions and a count; whether either explains a result of the record is not known until the runs and measurements of section 6 exist. The life runner was read in its update and its store, not whole.

**The independent inventory** (`INDEPENDENT_INVENTORY_OF_WHAT_THE_PROGRAM_FIXES_20261118.md` beside this file; made from the source alone by a fresh agent that had seen neither this review nor any document of the project; 80 entries in seven groups).

- *Agreement.* Each of the reviewer's rows of revision 1 is among its entries. Its ten weightiest entries include, independently, the three that this review ranks first: the dense readout with the direct path from the input (its entry 2), one crossing before the answer is due (its entry 3), and the reset state with batches of 64 (its entry 6).
- *What the reviewer had not listed,* now entered above: the count of reciprocal connections (C3: 520 of 2,560, which the reviewer recomputed); the shared count of updates and the optimizer carried through a life (D5); the refusal of a run by the program's list of needed bytes, and the size of the code taken from the dense network (B8); the end of a birth by a score (E5); the fixed form of the cell (A5).
- *Where it ranks otherwise.* It puts first that both rules are one global backward pass (finding 6 above), which the record treats as named and kept. It also ranks the schedule of experience (one family for 256 updates) and the placing of the error by the program (targets only where the program knows an answer to be) among its ten; this review lists them (E1, C1) and does not rank them, since they are the world's and the task's.
- *What it adds as tests:* those of section 6, item 7; a loss at every position and an error for each output unit without the normalisation (for C1 and B7); families interleaved within a visit (for E1); a store filled with the network's own answers in place of the program's copies (for E2).
- *What it could not determine* is listed in its section 3; none of it bears on the rows above.

After the comparison the list has been made by two readers. A structure that both missed is still possible; the rule of section 7 stands for every new brief.

## History

Revision 1 (18 November 2026, 21:05 UTC): the reviewer's inventory of 27 rows and its findings. Revision 2 (18 November 2026): five rows and finding 6 added from the comparison with the independent inventory; section 8 written.
