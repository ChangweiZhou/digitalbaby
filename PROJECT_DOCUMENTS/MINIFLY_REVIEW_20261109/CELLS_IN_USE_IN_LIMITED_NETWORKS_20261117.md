# Cells in use and directions in use in the networks with limited or local connections (17 November 2026)

Reviewer's document, item 2 of `COORDINATION_20261117/PLAN.md`. It is a measurement on saved weights. Nothing was trained, torch was not used, and no code of another branch was run: the weight files were read with `git show`, one at a time. Every figure below is printed by `cells_in_use_limited.py` in this folder; section 7 gives the commands. Every statement is for seed 73129 only.

## 1. The question

In the dense network every cell is in use on every item, including items of an unrelated task made of bytes that the network has never met, and the items of the first family occupy 255 or 256 of the 257 directions of the cells' state (`cells_in_use.py`, `directions_in_use.py`; memo `MEMO_TO_COORDINATOR_LEARNED_STRUCTURE_20261115.md`, section 29). The memo concluded that new learning changes every connection, so that nothing learned earlier is left untouched.

Is the same true in the networks whose connections were limited or made local (E42, E47, E49), or do those networks leave cells, or directions of the cells' state, unused by the first family?

**The answer in three sentences.** In the six networks of 256 cells in which every cell receives the input (E42's `G`, `F10`, `P8`; E49's `in16`) nothing is left unused: every cell's state varies with the item and the family's items occupy 245 to 256 of the 257 directions. In E47's four networks of 1,024 cells, 435 to 628 cells have a state that does not vary with the item (below 0.01; 567 to 697 below 0.05), all of them outside the interface and 419 to 526 of them two or more connections away from it, and the family's items occupy 303 to 334 of the 1,025 directions. These cells are not changed by unrelated items either, and whether new learning would use them cannot be shown by a measurement on fixed weights.

## 2. Terms

- **The first family** is the four tasks add, sub, max and min on two numbers below 89 (add and sub modulo 89), written as a line of bytes with 0 to 12 letters between the operands. The networks measured here were trained on it and on nothing else.
- **The unrelated task** is E43's: copy two symbols after fillers. Its bytes are 128 to 225, none of which occurs in an item of the first family.
- **A cell** is one unit of the recurrent layer. At a position of an item it has an update gate z, a candidate n and a previous state h, and its new state is (1 - z) n + z h (a GRU cell). **A position** is one input symbol of an item; position 0 is the start symbol, and an item's input is the start symbol, the prompt and the answer without its last symbol.
- **Rule S** is the sweep, **rule B** is back-propagation; they are the two learning rules under which each network was trained.
- **The change factor** of a cell at a position is (1 - z)(1 - n²) + z(1 - z)|h - n|, as `cells_in_use.py` defines it. Every change made at that position to the cell's incoming connections is a product that contains this factor, under both rules; where it is zero the cell's incoming connections cannot change. For a cell that is updated only every p-th position it is computed at the positions at which the cell is updated, and nowhere else.
- **The state size** of a cell at a position is the absolute value of its new state. Every change to a connection that leaves the cell is a product that contains the cell's state.
- **A cell's mean** change factor or state size on a set of items is the mean over all pairs (item, position) with position 1 or later (for the change factor: at which the cell is updated).
- **The variation of a cell's state with the item** is the standard deviation of the cell's state across the items of a set at one position (positions counted from the start symbol; at each position, the items long enough to have it), averaged over the positions from 1 to the last. A cell **does not vary with the item** at the mark 0.01 (or 0.05) when this figure is below the mark. Two other forms were computed as a control: the same inside each group of items of one length, and the largest over positions in place of the mean.
- **The marks** 0.01 and 0.05 are those of the request. States lie between -1 and 1.
- **The directions that a set of items occupies** are eigenvectors of the second-moment matrix of the vectors (the state of every cell before a position, followed by a constant 1), taken over all pairs (item, position) of the set. These are the vectors that the items present to the connections between cells; `directions_in_use.py` takes them in the same way for its row "recurrent". There are as many directions as cells, plus one. Four counts are given: the least number of leading eigenvectors that hold 99%, 99.9% and 99.99% of the trace, and the number of eigenvalues above 1e-6 of the largest. These four are relative to the size of the whole vector, which grows with the number of cells. Two counts on an absolute scale are added: the number of eigenvalues above 1e-4 and above 2.5e-3, that is, the number of directions along which the root-mean-square size of the vectors is above 0.01 and above 0.05, the same marks as for single cells.
- **Restricted to the cells a cell reads**: the same counts for the part of the matrix that belongs to one cell's ten sources, the cell itself and the constant (12 directions).
- **Inside the family's directions**: the share of the squared size of a set's vectors that lies in the span of the leading eigenvectors of the family's matrix that hold 99.99% of its trace.
- **The networks.** E42 (256 cells, every cell receives the whole input and is read by the readout): `G`, every cell reads every cell (its tensors are bit-equal to the birth of E26, `E42_THREE_SEEDS_VERIFICATION_20261117.md`, part A); `F10`, each cell reads itself and 10 fixed cells within three rows and columns of it on a 16 x 16 grid; `P8`, as `F10` with connections replaced during training and with 128, 64, 32 and 32 cells updated every 1, 2, 4 and 8 positions. E49 `in16`: `F10` in which each cell reads 16 of the 512 values of the code of a byte. E47 (1,024 cells on a 32 x 32 grid, each reading itself and 10 cells within three rows and columns): the 256 **interface cells** of one 16 x 16 patch receive the input and are read by the readout; the 768 **far cells** receive no input and are not read. A far cell's **ring** is the number of connections between it and the interface: ring 1 holds the 228 far cells within three grid steps of the patch, ring 2 the 300 cells four to six steps away, ring 3 the 240 cells seven or eight steps away (as in `E47_TRIAL_VERIFICATION_20261117.md`). In the arm `one` every cell is updated at every position; in the arm `place` the far cells are updated every 2, 4, 8 or 16 positions, the more rarely the farther from the interface.

## 3. Files, items and method

**Files** (final weights after 24,576 updates, seed 73129):

| Experiment | Branch and commit | Networks | Reader |
|---|---|---|---|
| E42 | `origin/claude/gallant-dijkstra-i9zgsr` at 44f91edc | `G`, `F10`, `P8` under S and B | `e42_cells_reader.py` |
| E47 | `origin/claude/e47-worker-stage-0-3d0ya6` at 578a877b | `one`, `place` under S and B | `e47_local_reader.py` |
| E49 | `origin/claude/peaceful-noether-vcvrmf` at a36de661 | `in16` under B | `e49_interface_reader.py` |

**Items.** (i) The family's taught items as `directions_in_use.py` builds them, with the same generator in the same order: the taught half of each of the four operations, and for each number of letters from 0 to 12, 40 pairs of each operation. That gives 2,080 items in 13 lengths of 10 to 22 positions, and 33,280 pairs (item, position). (ii) The held-out panel of the unrelated task, `e43_unrelated_reader.panel(73129, "held")`: 300 items of 19 positions, 5,700 pairs (item, position). The two sets share no input symbol except the start symbol.

**Method.** The states of the cells after every position come from the reader's own forward computation. The readers do not return the gates, so the script computes z and n of every cell at every position from the reader's state of the position before and the weights. It does this in another arrangement than the readers (dense matrices built from the saved masks or source lists, every cell computed at every position and the updated ones selected afterwards). At every position of every item it checks that (1 - z) n + z h gives the reader's next state for the cells updated there and that the other cells keep their state.

**Checks of the method** (commands in section 7).

1. *The gates.* The largest difference between the state computed from the script's gates and the reader's next state is 0 in the six networks of E42 and between 2.9e-7 and 5.9e-7 in the four of E47 and in `in16`. With one deliberate error put into the script's computation the difference is large: equal step weights in place of the saved ones give 2.0 in `P8` and 0.70 to 0.71 in `place`; an update taken one position early gives 2.0 in `P8` and 1.5 to 1.8 in `place`; the candidate computed without the reset gate gives 1.4 to 2.0 in all eleven networks.
2. *The dense figures of section 29 are reproduced on `G`.* On `G` the script gives 179, 244, 256 (S) and 165, 239, 255 (B) directions at 99%, 99.9% and 99.99%; `directions_in_use.py` prints the same six numbers for the births of seed 73129 in its row "recurrent". On the unrelated panel the script gives a change factor of 0.235 / 0.392 / 0.498 (S) and 0.150 / 0.367 / 0.556 (B) at the 5th, 50th and 95th percentile of cells, and a state size of 0.352 / 0.445 / 0.547 and 0.396 / 0.533 / 0.690; `cells_in_use.py A` prints the same twelve numbers for the births in its row "unrelated, 12 fillers".
3. *The variation of a state in E47 is reproduced.* The script's part X gives, for the standard deviation over the items of the A panel of a cell's state at `=`, averaged over the interface and over rings 1, 2, 3: `one` S 0.398, 0.149, 0.018, 0.002; `one` B 0.458, 0.175, 0.009, 0.000; `place` S 0.370, 0.061, 0.003, 0.000; `place` B 0.441, 0.124, 0.005, 0.000. These are the sixteen numbers of the table in section 6 of `E47_TRIAL_VERIFICATION_20261117.md`.
4. *The structure.* For E47 the sources, periods, phases, interface and far cells saved in each weight file equal the reader's draws; for `in16` the three saved masks equal the reader's draws; the script stops if they do not.
5. *The answers.* Through the reader, the networks answer 95.7% to 99.5% of the 2,080 family items with the whole answer right (table 1) and none of the 300 unrelated items.
6. *Repetition.* Six full runs of the script printed the same figures.

## 4. Tables

In every table "S" and "B" are the two rules, "family" is the set of 2,080 taught items and "unrelated" the panel of 300 items. A pair of numbers "a, b" in a column of cells is the number of cells below 0.01 and the number below 0.05.

### Table 1. Cells below the marks (measures 1 and 2)

| Network | Cells | Family answers right, % | Change factor, family | Change factor, unrelated | State size, family | State size, unrelated | Variation with the item, family | Variation with the item, unrelated |
|---|---|---|---|---|---|---|---|---|
| G S | 256 | 99.5 | 0, 0 | 0, 0 | 0, 0 | 0, 0 | 0, 0 | 0, 0 |
| G B | 256 | 98.9 | 0, 12 | 0, 0 | 0, 0 | 0, 0 | 0, 0 | 0, 0 |
| F10 S | 256 | 98.2 | 0, 3 | 0, 0 | 0, 0 | 0, 0 | 0, 0 | 0, 0 |
| F10 B | 256 | 98.0 | 0, 20 | 0, 0 | 0, 0 | 0, 0 | 0, 0 | 0, 0 |
| P8 S | 256 | 98.4 | 0, 6 | 0, 0 | 0, 0 | 0, 0 | 0, 0 | 0, 0 |
| P8 B | 256 | 99.2 | 5, 38 | 0, 1 | 0, 0 | 0, 0 | 0, 0 | 0, 0 |
| in16 B | 256 | 95.7 | 0, 0 | 0, 0 | 0, 0 | 0, 0 | 0, 0 | 0, 0 |
| one S | 1,024 | 98.3 | 0, 8 | 0, 13 | 41, 72 | 44, 71 | 435, 567 | 464, 586 |
| one B | 1,024 | 98.7 | 4, 201 | 9, 203 | 0, 0 | 0, 0 | 539, 609 | 574, 633 |
| place S | 1,024 | 97.2 | 0, 3 | 0, 3 | 405, 521 | 408, 520 | 628, 697 | 646, 718 |
| place B | 1,024 | 98.9 | 10, 65 | 8, 50 | 210, 243 | 209, 243 | 555, 657 | 599, 685 |

The median over cells of the same three quantities, family / unrelated:

| Network | Change factor | State size | Variation with the item |
|---|---|---|---|
| G S | 0.262 / 0.392 | 0.565 / 0.445 | 0.538 / 0.442 |
| G B | 0.183 / 0.367 | 0.728 / 0.533 | 0.641 / 0.464 |
| F10 S | 0.242 / 0.352 | 0.589 / 0.517 | 0.500 / 0.493 |
| F10 B | 0.178 / 0.359 | 0.688 / 0.535 | 0.603 / 0.474 |
| P8 S | 0.214 / 0.321 | 0.617 / 0.456 | 0.565 / 0.429 |
| P8 B | 0.161 / 0.305 | 0.686 / 0.532 | 0.631 / 0.450 |
| in16 B | 0.449 / 0.540 | 0.454 / 0.357 | 0.461 / 0.219 |
| one S, all 1,024 cells | 0.340 / 0.358 | 0.547 / 0.544 | 0.023 / 0.018 |
| one S, interface cells | 0.242 / 0.309 | 0.567 / 0.542 | 0.501 / 0.514 |
| one B, all 1,024 cells | 0.121 / 0.143 | 0.856 / 0.867 | 0.006 / 0.003 |
| one B, interface cells | 0.176 / 0.352 | 0.652 / 0.509 | 0.594 / 0.438 |
| place S, all 1,024 cells | 0.500 / 0.500 | 0.045 / 0.046 | 0.000 / 0.000 |
| place S, interface cells | 0.253 / 0.322 | 0.581 / 0.528 | 0.472 / 0.492 |
| place B, all 1,024 cells | 0.436 / 0.500 | 0.477 / 0.427 | 0.005 / 0.002 |
| place B, interface cells | 0.175 / 0.344 | 0.642 / 0.505 | 0.587 / 0.441 |

Notes to table 1.

- *The two other forms of the variation.* In the seven networks of 256 cells no cell is below 0.05 in any of the three forms, on either set. In E47 the counts on the family are, inside lengths: `one` S 451, 578; `one` B 543, 611; `place` S 639, 705; `place` B 563, 665; and with the largest over positions: 391, 530; 502, 579; 602, 681; 512, 612.
- *Cells with a small change factor in the networks of 256 cells* (`G` B 12, `F10` B 20, `P8` B 38 below 0.05 on the family) all vary with the item, and all but one of them have a change factor above 0.05 on the unrelated items. They are cells whose state on the family's items lies where the candidate or the update gate is saturated.
- *Cells updated every p-th position.* In `P8` a cell of period 2, 4, 8 is updated 7.52, 3.75, 1.86 times an item on average at positions 1 or later (family). In `place` a cell of period 2, 4, 8, 16 is updated 7.51, 3.71, 1.85, 0.94 times. A cell of period 16 whose first update lies beyond the end of a short item keeps the state zero in that item by construction. This does not produce the counts of `place`: on the unrelated items, which have 19 positions so that every cell is updated at least once, the counts are the same within three cells (state size 408, 520; variation 646, 718).

### Table 2. Directions that the family's items occupy (measure 3) and the overlap with the unrelated items (measure 4)

| Network | Directions | 99% | 99.9% | 99.99% | Above 1e-6 of the largest | Not among the 99.99%; not above 1e-6 | Root-mean-square size above 0.01; above 0.05 | Of the 12 directions of a cell's sources, 99.99%: least / median / most over cells | Cells with all 12 | Correlation over cells of the change factor on the two sets | Unrelated items inside the family's directions (a random direction) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| G S | 257 | 179 | 244 | 256 | 257 | 1; 0 | 257; 257 | not applicable | not applicable | +0.85 | 0.9995 (0.996) |
| G B | 257 | 165 | 239 | 255 | 257 | 2; 0 | 257; 257 | not applicable | not applicable | +0.86 | 0.9990 (0.992) |
| F10 S | 257 | 144 | 230 | 253 | 257 | 4; 0 | 257; 255 | 12 / 12 / 12 | 256 | +0.78 | 0.9956 (0.984) |
| F10 B | 257 | 152 | 232 | 252 | 257 | 5; 0 | 257; 255 | 12 / 12 / 12 | 256 | +0.83 | 0.9932 (0.981) |
| P8 S | 257 | 169 | 235 | 253 | 257 | 4; 0 | 257; 256 | 12 / 12 / 12 | 256 | +0.87 | 0.9972 (0.984) |
| P8 B | 257 | 172 | 236 | 254 | 257 | 3; 0 | 257; 257 | 12 / 12 / 12 | 256 | +0.85 | 0.9975 (0.988) |
| in16 B | 257 | 111 | 204 | 245 | 257 | 12; 0 | 257; 197 | 12 / 12 / 12 | 256 | +0.75 | 0.9994 (0.953) |
| one S | 1,025 | 91 | 230 | 320 | 363 | 705; 662 | 392; 288 | 3 / 8 / 12 | 274 | +0.96 | 0.9978 (0.312) |
| one B | 1,025 | 86 | 232 | 322 | 363 | 703; 662 | 407; 316 | 3 / 7 / 12 | 282 | +0.93 | 0.9981 (0.314) |
| place S | 1,025 | 136 | 251 | 303 | 347 | 722; 678 | 334; 269 | 1 / 6 / 12 | 152 | +0.94 | 0.9969 (0.296) |
| place B | 1,025 | 137 | 269 | 334 | 381 | 691; 644 | 395; 317 | 1 / 10 / 12 | 299 | +0.90 | 0.9984 (0.326) |

Notes to table 2.

- *The relative marks are not on one scale across networks.* The largest eigenvalue is 13.9 to 35.2 in the networks of 256 cells and 49.4 to 486.3 in E47's (`one` B: 486.3, because its far cells hold states near 1). The mark "1e-6 of the largest" therefore corresponds to a root-mean-square size of about 0.005 in `G` S and of about 0.022 in `one` B. The two absolute columns do not have this dependence.
- *With the states after each position* in place of the states before it the count at 99.99% changes by at most 3 (for example `G` S 256, `F10` S 254, `in16` 247, `one` S 323, `place` S 305).
- *The unrelated items' own vectors* occupy, at 99.99%, 256 or 257 directions in `G`, `F10` and `P8`, 235 in `in16`, and 356, 330, 322, 337 in `one` S, `one` B, `place` S, `place` B.
- *The correlation on `G`* is +0.85 and +0.86 here and +0.73 and +0.76 in section 29 of the memo. The memo's figure compares the unrelated panel with additions and subtractions at 12 letters; the figure here uses the 2,080 taught items of four operations at 0 to 12 letters. `cells_in_use.py A` prints +0.73 and +0.76 for these births.
- *The correlation in E47 by group* (interface, ring 1, ring 2, ring 3): `one` S +0.79, +0.96, +1.00, +1.00; `one` B +0.86, +0.98, +0.99, +1.00; `place` S +0.76, +0.96, +0.99, +0.99; `place` B +0.87, +0.97, +1.00, +0.99.
- *Inside the family's directions.* In E47 the family's directions at the 99.99% level are 30 to 33% of all directions and hold 99.69 to 99.84% of the squared size of the unrelated items' vectors.

### Table 3. E47 by ring, and `P8` by period

Columns as in tables 1 and 2. "Directions of the group" are the counts for the part of the family's matrix that belongs to the cells of the group and the constant.

| Network | Group | Cells | Variation with the item, family | Variation, unrelated | State size, family | Change factor, family | Change factor, unrelated | Directions of the group: 99.99%, above 1e-6, of | Root-mean-square size above 0.01; above 0.05 | Of the 12 of a cell's sources, 99.99%: least / median / most |
|---|---|---|---|---|---|---|---|---|---|---|
| one S | interface | 256 | 0, 0 | 0, 0 | 0, 0 | 0, 3 | 0, 0 | 253, 257, 257 | 257; 253 | 9 / 12 / 12 |
| one S | ring 1 | 228 | 16, 60 | 21, 70 | 8, 16 | 0, 0 | 0, 0 | 131, 171, 229 | 144; 72 | 5 / 10 / 12 |
| one S | ring 2 | 300 | 187, 268 | 209, 276 | 30, 47 | 0, 1 | 0, 1 | 24, 47, 301 | 44; 18 | 3 / 6 / 11 |
| one S | ring 3 | 240 | 232, 239 | 234, 240 | 3, 9 | 0, 4 | 0, 12 | 5, 11, 241 | 12; 7 | 3 / 4 / 8 |
| one B | interface | 256 | 0, 0 | 0, 0 | 0, 0 | 0, 24 | 0, 0 | 253, 257, 257 | 257; 255 | 10 / 12 / 12 |
| one B | ring 1 | 228 | 39, 81 | 58, 100 | 0, 0 | 0, 5 | 0, 7 | 126, 162, 229 | 158; 88 | 5 / 10 / 12 |
| one B | ring 2 | 300 | 260, 288 | 276, 293 | 0, 0 | 1, 56 | 2, 61 | 12, 23, 301 | 28; 14 | 3 / 6 / 11 |
| one B | ring 3 | 240 | 240, 240 | 240, 240 | 0, 0 | 3, 116 | 7, 135 | 5, 6, 241 | 6; 5 | 3 / 4 / 6 |
| place S | interface | 256 | 0, 0 | 0, 0 | 0, 0 | 0, 0 | 0, 0 | 251, 257, 257 | 257; 244 | 6 / 12 / 12 |
| place S | ring 1 | 228 | 102, 158 | 113, 178 | 70, 113 | 0, 1 | 0, 1 | 83, 117, 229 | 80; 40 | 2 / 8 / 12 |
| place S | ring 2 | 300 | 286, 299 | 293, 300 | 147, 187 | 0, 2 | 0, 2 | 25, 39, 301 | 29; 14 | 1 / 5 / 10 |
| place S | ring 3 | 240 | 240, 240 | 240, 240 | 188, 221 | 0, 0 | 0, 0 | 15, 18, 241 | 14; 5 | 1 / 3 / 8 |
| place B | interface | 256 | 0, 0 | 0, 0 | 0, 0 | 3, 22 | 0, 1 | 253, 255, 257 | 255; 255 | 11 / 12 / 12 |
| place B | ring 1 | 228 | 48, 121 | 69, 147 | 1, 5 | 1, 12 | 2, 11 | 113, 153, 229 | 140; 71 | 6 / 11 / 12 |
| place B | ring 2 | 300 | 267, 296 | 290, 298 | 34, 47 | 6, 31 | 6, 38 | 21, 40, 301 | 37; 19 | 2 / 8 / 12 |
| place B | ring 3 | 240 | 240, 240 | 240, 240 | 175, 191 | 0, 0 | 0, 0 | 17, 21, 241 | 17; 12 | 1 / 5 / 10 |

The 540 cells of rings 2 and 3 taken together, with the constant (541 directions): the family occupies 19, 10, 26, 22 at 99.99% and 36, 20, 39, 40 above 1e-6 in `one` S, `one` B, `place` S, `place` B. The 768 far cells together (769 directions): 97, 93, 88, 106 at 99.99% and 132, 128, 122, 143 above 1e-6.

In `P8` no cell of any period is below 0.05 in variation or in state size under either rule, every cell's 12 directions are occupied at 99.99%, and the cells of each period with the constant occupy, at 99.99%, 128 of 129 (period 1), 65 of 65 (period 2), 33 of 33 (period 4) and 32 or 33 of 33 (period 8). The cells below 0.05 in change factor on the family are, by period 1, 2, 4, 8: 4, 0, 1, 1 under S and 25, 2, 2, 9 under B.

### Table 4. E47: the cells whose state does not vary with the item (measure 5)

"Cells that vary" are the cells at or above the same mark on the family. A cell's sources are the ten cells it reads beside itself. The mean state "within the mark" means that the cell's mean state at a position on the unrelated items differs from its mean state at the same position on the family by less than the mark, averaged over positions 1 to 18.

| Network | Mark | Cells that do not vary | Interface, ring 1, ring 2, ring 3 | Of these, on unrelated items: variation below the mark; mean state within the mark | Sources that vary, of 10: mean; cells with none | Cells that vary and read them: mean; cells with none | Neither such a source nor such a reader | Read by an interface cell | State size below 0.05; change factor below 0.05 on both sets; neither |
|---|---|---|---|---|---|---|---|---|---|
| one S | 0.01 | 435 | 0, 16, 187, 232 | 434; 428 | 2.12; 111 | 2.18; 120 | 75 | 13 | 66; 5; 364 |
| one S | 0.05 | 567 | 0, 60, 268, 239 | 567; 563 | 1.38; 243 | 1.40; 261 | 189 | 42 | 72; 5; 490 |
| one B | 0.01 | 539 | 0, 39, 260, 240 | 539; 539 | 1.47; 207 | 1.57; 231 | 160 | 23 | 0; 177; 362 |
| one B | 0.05 | 609 | 0, 81, 288, 240 | 608; 609 | 1.22; 306 | 1.26; 325 | 259 | 56 | 0; 177; 432 |
| place S | 0.01 | 628 | 0, 102, 286, 240 | 628; 625 | 1.28; 315 | 1.25; 322 | 260 | 75 | 494; 3; 131 |
| place S | 0.05 | 697 | 0, 158, 299, 240 | 696; 693 | 0.96; 431 | 0.95; 436 | 384 | 123 | 520; 3; 174 |
| place B | 0.01 | 555 | 0, 48, 267, 240 | 555; 553 | 1.49; 230 | 1.55; 246 | 183 | 30 | 242; 42; 271 |
| place B | 0.05 | 657 | 0, 121, 296, 240 | 657; 656 | 1.08; 373 | 1.09; 382 | 320 | 89 | 243; 42; 372 |

The same cells at the mark 0.01, by ring:

| Network | Ring (cells that do not vary) | Sources that vary: mean; cells with none | Cells that vary and read them: mean; cells with none | Change factor on the family, median | State size on the family, median | Mean size of a recurrent weight of these cells |
|---|---|---|---|---|---|---|
| one S | 1 (16) | 4.75; 0 | 6.31; 0 | 0.372 | 0.010 | 0.146 |
| one S | 2 (187) | 2.95; 18 | 3.14; 18 | 0.306 | 0.647 | 0.135 |
| one S | 3 (232) | 1.27; 93 | 1.12; 102 | 0.240 | 0.791 | 0.146 |
| one B | 1 (39) | 3.95; 1 | 5.15; 0 | 0.098 | 0.910 | 0.283 |
| one B | 2 (260) | 2.09; 41 | 2.20; 51 | 0.091 | 0.924 | 0.262 |
| one B | 3 (240) | 0.41; 165 | 0.30; 180 | 0.052 | 0.960 | 0.234 |
| place S | 1 (102) | 4.13; 3 | 3.99; 6 | 0.500 | 0.001 | 0.038 |
| place S | 2 (286) | 1.27; 92 | 1.25; 96 | 0.501 | 0.008 | 0.020 |
| place S | 3 (240) | 0.10; 220 | 0.08; 220 | 0.500 | 0.000 | 0.001 |
| place B | 1 (48) | 4.00; 1 | 5.23; 1 | 0.169 | 0.786 | 0.298 |
| place B | 2 (267) | 2.03; 55 | 2.03; 61 | 0.332 | 0.481 | 0.112 |
| place B | 3 (240) | 0.39; 174 | 0.29; 184 | 0.500 | 0.000 | 0.006 |

The mean size of a recurrent weight over all cells of the interface and of rings 1, 2, 3 is 0.198, 0.178, 0.132, 0.145 in `one` S; 0.179, 0.234, 0.257, 0.234 in `one` B; 0.201, 0.106, 0.024, 0.001 in `place` S; 0.176, 0.259, 0.112, 0.006 in `place` B. These sixteen numbers equal those of the table in section 6 of `E47_TRIAL_VERIFICATION_20261117.md`.

What table 4 shows, in sentences.

- *Where.* No interface cell is among them. At the mark 0.01 they are 7 to 45% of ring 1 (16 to 102 of 228), 62 to 95% of ring 2 (187 to 286 of 300) and 97 to 100% of ring 3 (232 to 240 of 240). Of the 540 cells of rings 2 and 3, 419 (`one` S), 500 (`one` B), 526 (`place` S) and 507 (`place` B) do not vary at the mark 0.01, and 507, 528, 539 and 536 at the mark 0.05.
- *Unrelated items do not change them.* Of the cells that do not vary on the family, all but at most one do not vary on the unrelated items either, and 428 of 435, 539 of 539, 625 of 628 and 553 of 555 hold, at each position, the same mean state on unrelated items as on the family's to within 0.01.
- *They are connected to cells that vary.* At the mark 0.01 such a cell has on average 1.3 to 2.1 sources that vary and is read by 1.3 to 2.2 cells that vary; 75 to 260 of them have neither. The cells without such a connection are those of ring 3. Only 13 to 75 of them are read by an interface cell, and those lie in ring 1 by the geometry.
- *What kind of cells they are differs between the four networks.* In `one` S they hold a constant state that is not near zero (median 0.65 in ring 2 and 0.79 in ring 3) and their change factor is not small (median 0.24 to 0.37); 66 of the 435 have a state size below 0.05. In `one` B they hold a constant state near 1 in size (median 0.91 to 0.96), and 177 of the 539 have a change factor below 0.05 on both sets. In `place` S 494 of the 628 have a state size below 0.05: their recurrent weights have a mean size of 0.001 in ring 3 against 0.201 at the interface, and their change factor is 0.500, the value of a cell whose gates receive nothing. In `place` B the 240 cells of ring 3 are of this last kind (weights 0.006, state 0.000, change factor 0.500), and the cells of rings 1 and 2 are of the kind of `one`.

### E49 `in16`: the input side

The limit of `in16` is at the input, so the same count was made there. The vectors are the code of the byte at each position (the 512 values that `l1` makes of the byte's row of the byte table) and the constant; for one cell, the 16 values it reads and the constant (17 directions).

| | 99% | 99.9% | 99.99% | Above 1e-6 of the largest | Unrelated items' input vectors inside the family's directions (99.99% level) |
|---|---|---|---|---|---|
| The whole code, of 513 | 9 | 17 | 22 | 28 | 0.9891 |
| One cell's 16 values and the constant, of 17: least / median / most over the 256 cells | 6 / 8 / 9 | 9 / 11 / 12 | 12 / 14 / 16 | 16 / 17 / 17 | 0.9977 / 0.9995 / 0.9999 |

The family's bytes occupy 12 to 16 of the 17 input directions of a cell at 99.99% and 16 or 17 above 1e-6, and the codes of the unrelated task's bytes, which the network has never met, lie inside those directions to 99.77% or more for every cell. Reading only 16 input values does not give a new byte an input direction that the family's bytes leave unoccupied. This is the finding of the memo's section 29 for the dense input map (reading 3), found again for each cell's 16 values.

## 5. Conclusions

One sentence for each network: is there a part that the first family leaves unused, how large, and where.

| Network | Conclusion |
|---|---|
| `G` S | There is none: every one of the 256 cells varies with the item and has a change factor and a state size above 0.05 on both sets, and the family's items occupy 256 of the 257 directions at 99.99% and all 257 at the root-mean-square mark 0.05. |
| `G` B | There is none: every cell varies with the item (12 cells have a change factor below 0.05 on the family and above it on unrelated items), and the family occupies 255 of 257 directions at 99.99% and all 257 at the mark 0.05. |
| `F10` S | There is none: every cell varies with the item, the family occupies 253 of 257 directions at 99.99% and 255 at the mark 0.05, and for every cell all 12 directions of the cells it reads are occupied. |
| `F10` B | There is none: every cell varies with the item, the family occupies 252 of 257 directions at 99.99% and 255 at the mark 0.05, and for every cell all 12 directions of the cells it reads are occupied. |
| `P8` S | There is none: every cell of every period varies with the item, the family occupies 253 of 257 directions at 99.99% and 256 at the mark 0.05, and for every cell all 12 directions of the cells it reads are occupied. |
| `P8` B | There is none: every cell of every period varies with the item (5 cells have a change factor below 0.01 on the family and above it on unrelated items), the family occupies 254 of 257 directions at 99.99% and all 257 at the mark 0.05, and for every cell all 12 directions of the cells it reads are occupied. |
| `in16` B | There is none among the cells and none at the input: every cell varies with the item, all 257 directions are occupied at the mark 0.01 (245 at 99.99%, 197 at the mark 0.05, so 60 directions are occupied with a root-mean-square size between 0.01 and 0.05), and of the 17 input directions of a cell the family's bytes occupy 12 to 16 at 99.99%, with the unrelated bytes' codes inside them to 99.77% or more. |
| `one` S | There is one: 435 of the 1,024 cells (567 at the mark 0.05) have a state that does not vary with the item, none at the interface, 16 in ring 1, 187 in ring 2 and 232 in ring 3, and the family occupies 320 of 1,025 directions at 99.99% (392 at the mark 0.01), of which the 540 cells of rings 2 and 3 contribute 19 of their 541. |
| `one` B | There is one: 539 cells (609 at the mark 0.05) do not vary with the item, none at the interface, 39 in ring 1, 260 in ring 2 and all 240 of ring 3, and the family occupies 322 of 1,025 directions at 99.99% (407 at the mark 0.01), of which rings 2 and 3 contribute 10 of their 541. |
| `place` S | There is one: 628 cells (697 at the mark 0.05) do not vary with the item, none at the interface, 102 in ring 1, 286 in ring 2 and all 240 of ring 3, 494 of them with a state size below 0.05 and weights near zero, and the family occupies 303 of 1,025 directions at 99.99% (334 at the mark 0.01), of which rings 2 and 3 contribute 26 of their 541. |
| `place` B | There is one: 555 cells (657 at the mark 0.05) do not vary with the item, none at the interface, 48 in ring 1, 267 in ring 2 and all 240 of ring 3, 242 of them with a state size below 0.05, and the family occupies 334 of 1,025 directions at 99.99% (395 at the mark 0.01), of which rings 2 and 3 contribute 22 of their 541. |

Three readings that the tables support.

1. **Limiting which cells a cell reads, how often it is updated, or which input values it reads does not leave anything unused.** `F10`, `P8` and `in16` have the figures of the dense network: no cell below 0.05 in variation or state size, and 245 to 254 of 257 directions at 99.99% against 255 or 256 in `G`. This agrees with reading 1 of the memo's section 29, which was drawn from the dense network alone.
2. **A part without activity that depends on the item exists only where cells receive no input, and it is the part two or more connections away from the interface.** The 256 interface cells of E47 have the figures of a network of 256 cells (table 1, lower part; table 3: 251 to 253 of 257 directions at 99.99%). Ring 1 lies between. Rings 2 and 3 are the cells that `E47_TRIAL_VERIFICATION_20261117.md` found to take no part in the answers; the present measurement adds the count per cell, the count of directions, and that unrelated items leave them as they are.
3. **By the two quantities of section 29 (change factor and state size) most of these cells are not out of use.** In `one` S 364 of the 435 cells have a state size above 0.05 and a change factor above 0.05, and in `one` B 362 of the 539. A rule that changes the connections of every cell whose change factor and state are not zero would change the connections of these cells and the connections that leave them. What distinguishes them is only that their state carries nothing about the item. In `place` 494 (S) and 242 (B) of them hold a state near zero.

**What this cannot show.**

- **Whether new learning would use that part.** Every figure is a forward computation on fixed weights. The unrelated items were presented to networks that have not learned them. That those items leave the cells of rings 2 and 3 unchanged says that the present weights carry no difference between items to those cells; it does not say what training on a new task would do to them.
- **Whether new learning in that part would leave the first family as it is.** The cells that do not vary are read by cells that vary (table 4), and in `one` they hold constant states of a size of 0.6 to 0.96. If learning changed those states, the cells that read them would receive a changed input. `E47_TRIAL_VERIFICATION_20261117.md` found that holding rings 2 and 3 at their mean leaves the family's scores within 0.7 of a point; that test did not move them to other values.
- **Whether an error reaches those cells under either rule,** which decides whether their connections change at all. This needs the gradients, which need torch.
- **Whether the cells with weights near zero in `place` can be brought back by learning.**
- **Whether a change kept out of the family's directions leaves the family intact in these networks.** The test of `directions_in_use.py` (its part F3: the change projected away from the family's directions) was not made here, because these networks have no saved weights after unrelated learning in the files used.
- **Anything about another seed.**

## 6. For the plan

Item 12 of the plan (E56: an unrelated task learned in E47's network) waits for this item. Three facts of the tables bear on its brief; they are observations, and the design is the coordinator's.

1. The part that the family leaves without item-dependent activity is reached only through cells that the family uses: the input arrives at the interface, ring 1 is read by the interface, and rings 2 and 3 are read only by far cells. Between 13 and 75 of the cells that do not vary are read by an interface cell.
2. In the four networks that part is of different kinds: constant states with gates that are not shut (`one` S), constant states near 1 with 177 cells whose change factor is below 0.05 (`one` B), and states near zero with weights near zero (`place`, ring 3 under both rules and rings 1 and 2 under S).
3. Unrelated items presented to the trained network do not reach that part, so a measurement of whether learning brings it into use has to be made during the learning.

## 7. Commands and run times

From the repository root, with `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1` set; `V=PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/cells_in_use_limited.py`. Times are wall-clock times on a host of four cores that other jobs were using (load average between 1.5 and 6).

| Command | What it prints | Time |
|---|---|---|
| `python3 -I $V` | every figure of sections 3 (checks 1, 4, 5), 4 and 5 for the eleven networks; the tables in the form used here are at the end of its output | 96 to 117 seconds (six runs) |
| `python3 -I $V G_S F10_S` | the same for the named networks only (names: `G_S G_B F10_S F10_B P8_S P8_B one_S one_B place_S place_B in16_B`) | 8 seconds |
| `python3 -I $V X` | check 3: the variation of a state at `=` on the A panel, by ring, in E47's four networks | 8 seconds |
| `python3 -I $V mutate:steps`, `mutate:phase`, `mutate:gate` | check 1: the largest difference with one deliberate error in the script's gates | 70 to 73 seconds each |
| `python3 -I -B PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/directions_in_use.py` | check 2: the dense network's directions (its part F1, row "recurrent", seed 73129) | 35 seconds |
| `python3 -I -B PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/cells_in_use.py A` | check 2: the dense network's change factor and state size on the unrelated panel, and the memo's correlations | 7 seconds |

The script writes no file. It reads the three branches at the commits of section 3, which must be present in the local repository, and the default branch through `np_gru.py` is not needed. The largest array it holds is a matrix of 1,025 x 1,025; a weight file is read into memory and released before the next.

## 8. What was not checked

- **Other seeds, other configurations.** Seeds 93683 and 116089 of E42 exist and were not measured. E49's other settings (`I64`, `I16`, `out8`) and `in16` under the sweep were not measured; the request named `in16` under B. E47 has seed 73129 only.
- **The vectors that a cell of period above 1 receives.** Such a cell reads a weighted mean of its sources' states over the positions of its period. The count "restricted to the cells a cell reads" uses the states themselves at every position. The weighted means are sums of those states, so they lie in the same directions, and the count given is an upper limit for the cells of `P8` and of the far cells of `place`. The count for the weighted means was not computed.
- **Directions at the readout.** The count was made for the vectors presented to the connections between cells, not for those presented to the readout's map `lm` at the answer positions, which `directions_in_use.py` also counts for the dense network.
- **The bit-equality of `G` with the birth of E26** is the statement of `E42_THREE_SEEDS_VERIFICATION_20261117.md` and was not repeated; check 2 shows that the figures on `G` equal the figures that the two earlier scripts print for that birth.
- **The gradients.** The change factor is a factor of every change to a cell's incoming connections; the other factor, the error that arrives at the cell, was not computed.
- **The marks.** 0.01 and 0.05 are fixed by the request. Table 1 gives both, and the notes give the two other forms of the variation; no other mark was tried.
- **The choice of items.** The family's set is one sample of 2,080 taught items and the unrelated set one panel of 300. In E47 the counts of cells that do not vary are of the same size on the two sets (table 1), which are made of different bytes; no further set was tried.
