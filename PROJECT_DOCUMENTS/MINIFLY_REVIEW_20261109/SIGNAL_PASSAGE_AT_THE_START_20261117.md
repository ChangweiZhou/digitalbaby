# Does a difference between two items reach the cells that do not read the input? The starting weights of E47, E49 and E50, and the starting weights changed (17 November 2026)

Reviewer's document, for item 1 of `COORDINATION_20261117/PLAN.md`. Read-only: nothing was trained, torch is not installed, and no code of a worker's branch or of the default branch was run. Weight files were read with `git show`, one at a time, from `claude/e47-worker-stage-0-3d0ya6` at 578a877b, `claude/peaceful-noether-vcvrmf` at a36de661 and `claude/e50-fixed-input-code-017dSE` at c61b1067. Every figure below is printed by `signal_passage.py` in this folder; the part of the script that prints it is named in brackets, and section 9 gives the commands and their run times. Every statement is for seed 73129 and for the weight files named.

## The question

At the starting weights of each of the three local networks, how much of a difference between two items arrives at cells that do not read the input directly, and what change of the starting weights would let it arrive?

The starting sizes of all weights in these networks were taken over from the dense network. Whether, at those sizes, a difference in the input reaches the cells that do not read it had not been measured.

## Terms

- A **byte** is one symbol of an item. An **item** of the first family is a line such as `p 2 1 + <letters> 0 6 =`: a task code, the two digits of the first operand, the operation sign, a number of capital **letters** that carry no information, the two digits of the second operand and `=`. The network then gives the two digits of the answer. A **position** is the place of one byte in the line; the network reads one byte at each position.
- A **cell** is one unit of the recurrent layer. Its **state** h is one number, which changes at each position by the equations of the GRU: r = s(a_r), z = s(a_z), n = tanh(a_n + r c_n), new h = (1 - z) n + z h, where s is the logistic function, a_r, a_z and a_n are sums over the input values the cell reads, and c_n and the recurrent parts of a_r and a_z are sums over the cell's **sources**. r is the **reset gate**, z the **update gate**, n the **candidate**. A cell with z near 1 keeps its state; a cell with z near 0 takes the candidate.
- The **sources** of a cell are the cell itself and ten other cells of its neighbourhood on a grid, its ten **connections**. The **recurrent weights** of a cell are its 33 weights on the eleven sources, eleven for each of r, z and n; the eleven of n are its **candidate rows**. All other entries of the recurrent map are fixed at zero; the 33 are the **kept** entries. The **update-gate bias** is the constant added to the recurrent part of a_z.
- **E47**: 1,024 cells on a 32 x 32 grid. The 256 **interface cells** of one 16 x 16 patch read the input (all 512 values of the byte's code) and are the only cells the readout reads. The other 768 are **far cells**. A far cell's **ring** is the number of connections between it and the interface, as `e47_trial_verify.py` defines it: ring 1 (228 cells) can be read by an interface cell, ring 2 (300 cells) and ring 3 (240 cells) lie further out. In the arm `one` every cell is updated at every position. In the arm `place` far cells are updated every 2, 4, 8 or 16 positions, the more rarely the farther out, from the average of their sources over the positions since their last update.
- **E49**: 256 cells. The **code** of a byte is 512 learned values. In the settings `in16` and `I16` each cell reads 16 of the 512 values; in `in16` the readout reads every cell, in `I16` each readout unit reads 8 cells.
- **E50**, setting `C16-in`: as `in16`, with a fixed code in which each of the 46 bytes of the family turns on 11 values of its own and every other value is zero. A cell that reads 16 values therefore has input for only some bytes.
- **E42's `F10`** is the reference: the same 256 cells and connections as E49, each cell reading all 512 values. It learns the family (E47's verification, section 3).
- The **sweep** (rule S) and **back-propagation** (rule B) are the two learning rules of the runs.
- A **pair** is two items that differ in exactly one byte, the **changed byte**. Two changes are used: the first digit of the first operand (**first digit changed**) and the last digit of the second operand (**last digit changed**). The **difference** of a cell at a position is its state in one item of the pair minus its state in the other. The **later positions** are every position after the changed byte up to the second answer digit, the last position at which an output is read.
- A **group** is a set of cells. In E47: interface, ring 1, ring 2, ring 3. In E50, for a pair: the cells that read at least one value of either of the two bytes at the changed position (the **reading cells**), and the cells that read none. In E49 the code is dense, every value differs between two bytes, every cell reads 16 differing values, and there is one group; for it the distribution over the cells is given.
- The **root mean square** of the difference of a group is taken over the pairs, the cells of the group, and the positions named.
- The **share that arrives** in a group is its root-mean-square difference over that of the first group (interface cells in E47, reading cells in E50), both over every later position. This is the task's definition. Three further figures are given beside it, because the share alone can mislead (section 3): the same ratio at the single position of `=`; the **farthest group at its largest**, which is the largest root-mean-square difference of the farthest group at any single later position over that of the first group at the changed byte; and, in E47, the variation **between items**, which is the standard deviation of a cell's state at `=` over 300 different items, averaged over the cells of a group, over the same for the interface (the measure of E47's verification).
- The **way back** (E47 only): small independent numbers are added to the states of the ring-3 cells at one position, and the run is compared with the unchanged run. The figure is the interface's root-mean-square difference over that of ring 3, over every later position. The readout reads interface cells only, so what a far cell holds can reach an answer only by this way.
- The **mean size of a state** is the mean of |h| over cells, positions and items. **Gate values beyond** is the share of the values of r and z that lie below 0.05 or above 0.95.
- The **factor per position** is the factor by which the root-mean-square difference over all cells changes from one position to the next, measured in items of 40 letters between 10 and 20 and between 20 and 40 positions after the changed byte. A difference **grows** if its size 40 positions after the changed byte is larger than its size at 10 or at 20 positions after.
- The **three conditions** of the task: over every later position of items of 12 letters with the first digit changed, the farthest group has at least a third of the first group's difference; fewer than 20% of gate values lie beyond 0.05 / 0.95; a difference does not grow along 40 letters. The tables say "met at 12 letters" when the three hold, and "met at 0, 4 and 12 letters" when the first also holds at 0 and at 4 letters.
- **Participation** of a distribution over cells is the share of the cells that would carry the whole mean square if all carried equally: 1 when every cell carries the same, 1 / (number of cells) when one cell carries all.

## 1. Summary

1. **At the unchanged start a difference barely leaves the cells that read it in E47, and does leave them in E50.** Over the later positions of items of 12 letters, ring 1 of E47's arm `one` holds 0.187 of the interface's difference, ring 2 0.0207 and ring 3 0.0028; in the arm `place` the three are 0.148, 0.0115 and 0.00033. In E50 the cells that read neither byte hold 0.266 of the reading cells' difference. In E49 there are no cells that do not read the difference: every cell reads 16 values that differ, and the cells carry it about as evenly as in E42's `F10` (participation 0.734 against 0.746).
2. **In every form, and in `F10`, a difference of one byte fades at the start by a factor of 0.64 to 0.66 per position.** Seventeen positions after the first digit, at `=` of an item of 12 letters, the interface of E47 holds 6.0e-05 of an initial 0.246. The share at `=` (ring 3: 0.165) is therefore a ratio of two quantities that have both nearly vanished, and the figure "farthest group at its largest" shows what arrives in absolute terms: ring 3 never holds more than 0.00077 of the size the difference had at the interface.
3. **The arithmetic agrees with the measurement.** One connection crossed in one position multiplies a difference by (1 - z) r (1 - n^2) w, expected 0.0434 at the start and measured 0.0420 to 0.0428 in the three rings of E47 and 0.0436 in E50. For a difference that lasts, the factor from ring to ring is expected at 0.134 to 0.174 and measured at 0.103 to 0.162, the sixfold to tenfold fall of E47's verification.
4. **E50: the smallest change tried that meets the three conditions is the recurrent weights times 1.5** (share 0.390 at 12 letters, 0.388 at 0 letters). The candidate rows alone times 1.5 (0.385) and the update-gate bias lowered by 1 (0.463) also meet them. Every multiple up to 4 meets them; at 6 a difference grows.
5. **E47, arm `one`: none of the uniform changes meets the three conditions.** The share of ring 3 reaches a third only at multiples at which a difference grows: 0.202 at 3.25 (no growth), 0.274 at 3.5 (grows), 0.349 at 3.75 (grows). The largest share without growth is 0.264 (candidate rows times 3.5). Shifting the update-gate bias, alone or together with a multiple, does not change this. The reason is in the wiring and not in the scale: of a far cell's ten connections only 2.4 to 4.0 come from the ring nearer the interface, while growth is decided by all eleven sources (section 6).
6. **E47, arm `one`: a change in one direction meets them.** With only the weights on connections that come from a cell nearer the interface multiplied by 5, ring 3 holds 0.639 of the interface's difference at 12 letters and 0.372 at 0 letters, no gate value lies beyond 0.05 / 0.95, and a difference shrinks by 0.84 per position. This start was not in the task's list; it was tried because the arithmetic pointed to it. It leaves the way back as it is: 0.0025 against 0.0026 at the unchanged start. A uniform multiple raises both ways together (times 3: 0.141 out, 0.149 back).
7. **E47, arm `place`: none of the changes tried meets them,** the directed one included. The far cells' rare updates hold the passage back: without growth ring 3 reaches 0.0300 at most under the uniform changes and 0.202 under the directed change times 8.
8. **Training raised the passage into rings 1 and 2 and not into ring 3.** In the four trained networks of E47 ring 1 holds 0.475 to 0.587 of the interface's difference and ring 2 0.0365 to 0.113; ring 3 holds 0.0152 (`one`, sweep), 0.00058 (`one`, back-propagation), 0.00012 and 0.00040 (`place`). The way back from ring 3 is 0.00046 to 0.0097.
9. **The measurement says nothing about learning.** It shows which starts let a difference of one byte arrive, at the first forward computation. Whether a network started so learns the family, faster or slower, or comes to use its far cells, is not shown (section 8).

The main table. Shares are for items of 12 letters with the first digit changed, over every later position.

| Network form | Farthest group | Share at the unchanged start | Smallest change tried that meets the three conditions | Largest share without growth among the uniform changes tried |
|---|---|---|---|---|
| E47 `one` | ring 3 | 0.0028 | none of the uniform changes; of all tried, connections from nearer the interface times 5 (0.639), with the way back unchanged at 0.0025 | 0.264 (candidate rows times 3.5); 0.202 (recurrent weights times 3.25) |
| E47 `place` | ring 3 | 0.00033 | none | 0.0300 (recurrent weights times 3.5 with the update-gate bias lowered by 2) |
| E49 `in16`, `I16` | none: every cell reads 16 differing values | not applicable | not applicable | not applicable |
| E50 `C16-in` | cells that read neither byte | 0.266 | recurrent weights times 1.5 (0.390); candidate rows times 1.5 (0.385); update-gate bias lowered by 1 (0.463) | 0.892 (recurrent weights times 4) |

## 2. What was measured, and how

**The items** [function `pairs`]. For each of the four operations of the first family (add, sub, max, min modulo 89), 75 held-out pairs of operands were written with 0, 4 and 12 letters by `np_gru.panel`, as the readers build their panels: 300 items at each length. An item's row is the start symbol, the prompt and the two digits of its right answer. Its partner has another digit at one position, drawn among the digits that leave the operand below the modulus, and is given the first item's answer digits, so that the two rows differ in exactly one byte. For the items of 40 letters, 25 pairs of operands of each operation were used: 100 pairs.

**The forward** [class `Layer`, function `forward`]. The three forms are computed by one function written for this measurement: cells with eleven sources each, three rows of weights on them, and for every byte an input sum for each gate. It works in double precision, so that a difference far below the resolution of single precision can be followed; in single precision, in which the networks ran, a difference below about 1e-8 is no difference at all. The structures (connections, rings, periods, masks, the code of E50) come from the reviewer's readers. The loaders are those of `e47_trial_verify.py` and `e49_result_verify.py`, imported and not edited.

**The check of the forward** [part `check`]. On 24 rows of 12 letters, the states of this forward differ from the states of the readers' own forwards by at most 5.7e-07 (E47 `one`, start), 5.6e-07 (E47 `place`, start), 1.1e-06 and 1.7e-06 (two trained networks of E47), 1.8e-07 (E49 `in16`, start), 5.0e-07 (`F10`), 5.5e-07 and 1.0e-07 (the two trained networks of `in16`), and 1.4e-07 (E50, against both forms of its reader). With the weights changed in the same way on both sides (recurrent weights times 3 and the update-gate bias lowered by 1 in E47; candidate rows times 4 in E49; recurrent weights times 2 and the bias lowered by 2 in E50) the largest difference is 1.1e-06 on the starting weights and 6.6e-05 on a trained network. The changes of the start therefore mean in this forward what they would mean in the readers.

**Which starting files hold the same tensors** [part `check`]. The two rules of an arm or setting start from the same tensors in E47, E49 `in16` and E50 `C16-in`. The arms `one` and `place` of E47 differ in the periods, phases and step weights only. `in16` and `I16` differ in the readout's map only, so that their cells compute the same states at the start; one measurement serves both. E50 `C16-in` and E49 `in16` hold the same recurrent weights, biases, input map and input mask; they differ in the code. E49 and E42's `F10` hold the same recurrent weights and biases. The kept recurrent weights have a root mean square of 0.1740 (E47) and 0.1743 (E49, E50) and a largest size of 0.3015, as a uniform draw within 1 / sqrt(11) has; the kept weights of the input map have a root mean square of 0.0361 where a cell reads 512 values and 0.2039 where it reads 16.

**The changes of the start** [method `Layer.changed`]. "Recurrent weights times g" multiplies the 33 kept recurrent weights of every cell, the cell's weight on its own state among them. "Candidate rows only" multiplies the eleven of n. "Update-gate bias -1" subtracts 1 from the bias of z in every cell, so that z falls from about 0.50 to about 0.27 and a cell takes more of the candidate. "Ten connections" multiplies the weights on the ten other cells and leaves the cell's weight on itself. "Connections from nearer the interface" (E47 only) multiplies, in all three gates, the weights on those connections whose source cell is nearer the interface, by grid distance, than the cell that reads it. Nothing else is changed: not the input map, not the biases other than the one named, not the periods.

## 3. The passage at the starting weights (item 1)

**E47, arm `one`: the difference by position** [part `start`]. First digit changed, items of 12 letters, 300 pairs. Root mean square of the difference; an entry of 0 is exactly zero, because a ring can first differ as many positions after the changed byte as it has connections to the interface.

| Positions after the changed byte (the byte read there) | Interface | Ring 1 | Ring 2 | Ring 3 | Ring 1 over interface | Ring 2 over interface | Ring 3 over interface |
|---|---|---|---|---|---|---|---|
| 0 (first operand, digit 1) | 0.24636 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1 (first operand, digit 2) | 0.12811 | 0.01595 | 0 | 0 | 0.1245 | 0 | 0 |
| 2 (operation) | 0.07098 | 0.01616 | 9.9e-04 | 0 | 0.2276 | 0.0139 | 0 |
| 3 (letter) | 0.04070 | 0.01279 | 0.00150 | 7.1e-05 | 0.3143 | 0.0369 | 0.0017 |
| 5 (letter) | 0.01453 | 0.00639 | 0.00138 | 1.8e-04 | 0.4394 | 0.0950 | 0.0125 |
| 8 (letter) | 0.00343 | 0.00186 | 6.3e-04 | 1.5e-04 | 0.5429 | 0.1823 | 0.0441 |
| 12 (letter) | 5.5e-04 | 3.3e-04 | 1.5e-04 | 5.4e-05 | 0.5987 | 0.2681 | 0.0988 |
| 17 (`=`) | 6.0e-05 | 3.8e-05 | 2.0e-05 | 9.9e-06 | 0.6313 | 0.3344 | 0.1650 |
| 19 (answer digit 2) | 2.5e-05 | 1.6e-05 | 8.9e-06 | 4.7e-06 | 0.6429 | 0.3533 | 0.1880 |
| every later position | 0.03556 | 0.00664 | 7.4e-04 | 9.8e-05 | 0.1868 | 0.0207 | 0.0028 |

Three things are to be read from this table. First, the difference at the interface halves at the first position and then falls by about 0.6 per position; nothing renews it, since the two items are identical after the changed byte. Second, the ratio of a ring to the interface rises with the distance from the changed byte, because the far cells lose their difference no faster than the interface does; at `=` ring 3 stands at 0.165 of the interface, but of an interface that holds 6.0e-05. Third, the root mean square over every later position is governed by the first few positions, where the difference is still large; that is why the share that arrives (0.0028 for ring 3) is so much smaller than the ratio at `=`.

**E47, both arms, at the unchanged start** [the rows `unchanged` of part `variants`]. Entries: ring 1, ring 2, ring 3, each over the interface.

| Start | First digit changed, 0 letters | 4 letters | 12 letters | Last digit changed | At `=`, 12 letters | Farthest group at its largest | Between items at `=` | Way back: interface over ring 3; interface at its largest |
|---|---|---|---|---|---|---|---|---|
| `one`, start | 0.186, 0.0198, 0.0023 | 0.186, 0.0206, 0.0027 | 0.187, 0.0207, 0.0028 | 0.169, 0.0116, 4.5e-04 | 0.631, 0.334, 0.165 | 7.7e-04 | 0.162, 0.0167, 0.0024 | 0.0026; 7.1e-04 |
| `place`, start | 0.141, 0.0047, 1.9e-05 | 0.146, 0.0085, 7.0e-05 | 0.148, 0.0115, 3.3e-04 | 0.101, 7.4e-04, 3.7e-07 | 4.343, 2.904, 0.163 | 9.5e-05 | 0.127, 0.0088, 1.7e-04 | 9.7e-04; 0.0011 |

- The share does not depend on the number of letters in the arm `one` (ring 3: 0.0023, 0.0027, 0.0028), for the reason just given.
- With the last digit of the second operand changed there are three later positions: `=` and the two answer digits. Ring 2 first differs at the first answer digit and ring 3 at the second. Whatever the weights, the last digit of the second operand cannot reach ring 2 before the first answer digit is given, and cannot come back from it before the answer ends.
- In the arm `place` the far cells are updated rarely and keep what they received in between, while the interface loses about half at each position. At `=` of an item of 12 letters ring 1 therefore holds 4.3 times and ring 2 2.9 times the interface's difference. The share over the later positions is nevertheless lower than in the arm `one` (ring 3: 0.00033 against 0.0028), because little arrives in the first positions.
- The way back is as weak as the way out: a difference put into ring 3 arrives at the interface at 0.0026 of its size in ring 3 (`one`).
- The variation between items (0.162, 0.0167, 0.0024 in `one`) reproduces the figures of E47's verification on its own panel (0.099, 0.016, 0.0017, 0.0002 as standard deviations): a fall of six- to tenfold from ring to ring.

**E50, `C16-in`** [part `start`]. Of the 256 cells, 67 to 82 read a value of a given digit and 113 to 137 (mean 128.4) a value of either of two digits; a cell has input for 11 to 16 of the 46 bytes (median 14). Of the cells that read neither of two digits, at most one has no connection from a cell that reads one. First digit changed, items of 12 letters:

| Positions after the changed byte (the byte read there) | Cells that read a value of either byte | Cells that read none | Second over first |
|---|---|---|---|
| 0 (first operand, digit 1) | 0.30571 | 0 | 0 |
| 1 (first operand, digit 2) | 0.15745 | 0.02972 | 0.1888 |
| 2 (operation) | 0.08481 | 0.02795 | 0.3296 |
| 3 (letter) | 0.04767 | 0.02128 | 0.4464 |
| 5 (letter) | 0.01630 | 0.01001 | 0.6139 |
| 8 (letter) | 0.00367 | 0.00276 | 0.7533 |
| 12 (letter) | 5.5e-04 | 4.6e-04 | 0.8278 |
| 17 (`=`) | 5.5e-05 | 4.8e-05 | 0.8649 |
| every later position | 0.04319 | 0.01151 | 0.2664 |

The share that arrives is 0.266 at 0, 4 and 12 letters and 0.244 with the last digit changed. The cells that read neither byte never hold more than 0.097 of the size the difference had in the reading cells at the changed byte. They are one connection from a reading cell, with 4.98 reading cells among their ten connections on average (section 6).

**E49, `in16` and `I16`, and `F10`** [part `start`]. There is no group of cells that does not read the difference. The distribution over the 256 cells of a cell's root-mean-square difference, over that of all cells (least, 5%, 25%, median, 75%, 95%, largest; first digit changed, items of 12 letters):

| Network | Later positions | Participation | At the changed byte | Participation |
|---|---|---|---|---|
| E49 `in16` (and `I16`) | 0.182 / 0.494 / 0.733 / 0.942 / 1.140 / 1.475 / 1.759 | 0.734 | 0.199 / 0.497 / 0.766 / 0.935 / 1.169 / 1.418 / 1.851 | 0.747 |
| E42 `F10` | 0.346 / 0.547 / 0.756 / 0.927 / 1.156 / 1.487 / 1.784 | 0.746 | 0.334 / 0.504 / 0.756 / 0.920 / 1.153 / 1.442 / 1.834 | 0.733 |

With 16 values a cell instead of 512 the cells carry a difference as evenly as in `F10`; only the least of the 256 cells is lower (0.18 against 0.35 of the common size). The start of E49 scaled the 16 kept input weights of a cell up by the square root of 32, and the size of a difference at the changed byte is the same in the two networks (0.241 in `in16`, 0.243 in `F10`, in the table below). As far as the passage of a difference goes, the start of E49 has nothing to repair.

**How fast a difference fades, in all forms** [part `start`, items of 40 letters; parts `variants`, rows `unchanged`]. Root-mean-square difference over all cells at the changed byte and 10, 20 and 40 positions after it, and the factor per position between 10 and 20 and between 20 and 40:

| Network | At the changed byte, 10, 20, 40 after | Factor per position | Mean size of a state | Gate values beyond |
|---|---|---|---|---|
| E47 `one` | 0.122, 7.9e-04, 1.1e-05, 2.6e-09 | 0.651, 0.660 | 0.178 (interface), 0.0334 (far) | 0.00% |
| E47 `place` | 0.122, 0.0014, 2.1e-04, 2.9e-05 | 0.825, 0.907 | 0.178 (interface), 0.0168 (far) | 0.00% |
| E49 `in16` | 0.241, 0.0014, 1.5e-05, 2.1e-09 | 0.635, 0.641 | 0.177 | 0.00% |
| E50 `C16-in` | 0.216, 0.0013, 1.4e-05, 2.2e-09 | 0.637, 0.645 | 0.132 | 0.00% |
| E42 `F10` | 0.243, 0.0014, 1.9e-05, 3.7e-09 | 0.648, 0.653 | 0.179 | 0.00% |

The network that learns (`F10`) loses a difference at the start as fast as the networks that learn badly. A start at which a difference lasts is therefore not shown to be needed by these figures; what distinguishes E47 and E50 from `F10` at the start is only where the difference goes.

## 4. The starting weights changed (item 2)

For each network there are two tables. The first gives the passage: the shares over every later position with the first digit changed at 0, 4 and 12 letters; the share with the last digit changed (items without letters, three later positions); the ratio at the single position of `=` in items of 12 letters; the farthest group at its largest (items of 12 letters); in E47 the variation between items and the way back. The second gives the mean size of a state and the gate values beyond 0.05 / 0.95 (both over the runs with the first digit changed at 0, 4 and 12 letters; reset and update gate apart in brackets), the root-mean-square difference over all cells in items of 40 letters at the changed byte and 10, 20 and 40 positions after it, the same for the farthest group, the factor per position between 10 and 20 and between 20 and 40, the factor expected from the arithmetic of section 6, whether a difference grows, and the three conditions. All figures are for 300 pairs at each of 0, 4 and 12 letters and 100 pairs of 40 letters.

### 4.1 E47, arm `one`

The task's list [part `variants`]. Entries of the first table: ring 1, ring 2, ring 3, each over the interface.

| Start | First digit changed, 0 letters | 4 letters | 12 letters | Last digit changed | At `=`, 12 letters | Farthest group at its largest | Between items at `=` | Way back: interface over ring 3; interface at its largest |
|---|---|---|---|---|---|---|---|---|
| unchanged | 0.186, 0.0198, 0.0023 | 0.186, 0.0206, 0.0027 | 0.187, 0.0207, 0.0028 | 0.169, 0.0116, 4.5e-04 | 0.631, 0.334, 0.165 | 7.7e-04 | 0.162, 0.0167, 0.0024 | 0.0026; 7.1e-04 |
| recurrent weights x 1.5 | 0.269, 0.0440, 0.0075 | 0.270, 0.0468, 0.0095 | 0.271, 0.0474, 0.0100 | 0.241, 0.0245, 0.0014 | 0.723, 0.424, 0.256 | 0.0028 | 0.239, 0.0393, 0.0090 | 0.0094; 0.0026 |
| recurrent weights x 2 | 0.343, 0.0767, 0.0172 | 0.348, 0.0851, 0.0240 | 0.350, 0.0874, 0.0267 | 0.302, 0.0404, 0.0031 | 0.784, 0.488, 0.317 | 0.0076 | 0.317, 0.0754, 0.0245 | 0.0258; 0.0069 |
| recurrent weights x 3 | 0.469, 0.162, 0.0525 | 0.505, 0.207, 0.0916 | 0.534, 0.256, 0.141 | 0.394, 0.0750, 0.0087 | 0.817, 0.600, 0.419 | 0.0471 | 0.508, 0.234, 0.132 | 0.149; 0.0499 |
| recurrent weights x 4 | 0.569, 0.258, 0.106 | 0.645, 0.366, 0.213 | 0.734, 0.527, 0.418 | 0.456, 0.108, 0.0174 | 0.869, 0.722, 0.639 | 0.433 | 0.720, 0.521, 0.430 | 0.321; 0.450 |
| recurrent weights x 6 | 0.655, 0.391, 0.216 | 0.723, 0.511, 0.391 | 0.808, 0.671, 0.615 | 0.495, 0.149, 0.0364 | 0.883, 0.798, 0.772 | 1.502 | 0.861, 0.758, 0.695 | 0.489; 4.501 |
| recurrent weights x 8 | 0.676, 0.437, 0.300 | 0.731, 0.548, 0.448 | 0.796, 0.685, 0.614 | 0.495, 0.166, 0.0474 | 0.858, 0.798, 0.744 | 1.939 | 0.853, 0.766, 0.702 | 0.630; 16.393 |
| candidate rows only x 1.5 | 0.270, 0.0441, 0.0075 | 0.271, 0.0468, 0.0095 | 0.272, 0.0474, 0.0100 | 0.241, 0.0245, 0.0014 | 0.728, 0.440, 0.272 | 0.0028 | 0.239, 0.0393, 0.0090 | 0.0092; 0.0025 |
| candidate rows only x 2 | 0.346, 0.0771, 0.0174 | 0.351, 0.0853, 0.0242 | 0.353, 0.0874, 0.0268 | 0.304, 0.0406, 0.0031 | 0.786, 0.506, 0.346 | 0.0076 | 0.317, 0.0751, 0.0245 | 0.0248; 0.0067 |
| candidate rows only x 3 | 0.478, 0.164, 0.0545 | 0.508, 0.207, 0.0942 | 0.531, 0.243, 0.138 | 0.401, 0.0767, 0.0086 | 0.861, 0.612, 0.473 | 0.0434 | 0.498, 0.223, 0.127 | 0.131; 0.0421 |
| candidate rows only x 4 | 0.585, 0.264, 0.113 | 0.653, 0.366, 0.223 | 0.740, 0.505, 0.405 | 0.468, 0.112, 0.0164 | 0.895, 0.702, 0.625 | 0.333 | 0.722, 0.494, 0.397 | 0.305; 0.394 |
| candidate rows only x 6 | 0.707, 0.417, 0.235 | 0.782, 0.544, 0.412 | 0.856, 0.685, 0.630 | 0.522, 0.152, 0.0295 | 0.914, 0.792, 0.766 | 1.480 | 0.877, 0.738, 0.683 | 0.397; 2.585 |
| candidate rows only x 8 | 0.751, 0.465, 0.280 | 0.811, 0.568, 0.441 | 0.851, 0.671, 0.621 | 0.522, 0.155, 0.0331 | 0.880, 0.745, 0.731 | 2.069 | 0.881, 0.753, 0.710 | 0.430; 5.399 |
| update-gate bias -1 | 0.308, 0.0343, 0.0045 | 0.307, 0.0344, 0.0045 | 0.308, 0.0345, 0.0046 | 0.300, 0.0299, 0.0023 | 0.728, 0.442, 0.302 | 0.0010 | 0.271, 0.0279, 0.0040 | 0.0045; 9.3e-04 |
| update-gate bias -2 | 0.453, 0.0528, 0.0071 | 0.452, 0.0528, 0.0071 | 0.452, 0.0530, 0.0072 | 0.449, 0.0513, 0.0057 | 0.878, 0.586, 0.445 | 0.0016 | 0.411, 0.0436, 0.0064 | 0.0071; 0.0014 |

| Start | Mean size of a state: interface, other cells | Gate values beyond (reset, update) | 40 letters, all cells: 0, 10, 20, 40 after | Farthest group: 10, 20, 40 after | Factor per position: 10 to 20, 20 to 40 | Expected factor | Grows | Three conditions |
|---|---|---|---|---|---|---|---|---|
| unchanged | 0.178, 0.0334 | 0.00% (0.00, 0.00) | 0.122, 7.9e-04, 1.1e-05, 2.6e-09 | 9.6e-05, 3.2e-06, 1.2e-09 | 0.651, 0.660 | 0.671 | no | not met |
| recurrent weights x 1.5 | 0.182, 0.0361 | 0.00% (0.00, 0.00) | 0.122, 0.0021, 8.5e-05, 2.1e-07 | 4.5e-04, 3.6e-05, 1.2e-07 | 0.727, 0.741 | 0.756 | no | not met |
| recurrent weights x 2 | 0.189, 0.0405 | 0.00% (0.00, 0.00) | 0.123, 0.0052, 6.0e-04, 1.2e-05 | 0.0015, 2.9e-04, 7.6e-06 | 0.805, 0.822 | 0.841 | no | not met |
| recurrent weights x 3 | 0.213, 0.0611 | 0.00% (0.00, 0.00) | 0.123, 0.0275, 0.0166, 0.0072 | 0.0110, 0.0104, 0.0061 | 0.951, 0.959 | 1.007 | no | not met |
| recurrent weights x 4 | 0.259, 0.115 | 0.04% (0.04, 0.04) | 0.123, 0.0962, 0.129, 0.200 | 0.0503, 0.112, 0.208 | 1.029, 1.022 | 1.149 | yes | not met |
| recurrent weights x 6 | 0.343, 0.230 | 3.43% (3.48, 3.38) | 0.123, 0.285, 0.421, 0.542 | 0.201, 0.385, 0.557 | 1.040, 1.013 | 1.307 | yes | not met |
| recurrent weights x 8 | 0.391, 0.292 | 12.80% (13.03, 12.56) | 0.122, 0.406, 0.544, 0.656 | 0.301, 0.487, 0.638 | 1.030, 1.009 | 1.380 | yes | not met |
| candidate rows only x 1.5 | 0.182, 0.0360 | 0.00% (0.00, 0.00) | 0.122, 0.0020, 7.7e-05, 1.8e-07 | 4.5e-04, 3.4e-05, 1.1e-07 | 0.723, 0.738 | 0.756 | no | not met |
| candidate rows only x 2 | 0.188, 0.0404 | 0.00% (0.00, 0.00) | 0.122, 0.0048, 5.0e-04, 8.7e-06 | 0.0015, 2.7e-04, 5.9e-06 | 0.797, 0.816 | 0.841 | no | not met |
| candidate rows only x 3 | 0.210, 0.0599 | 0.00% (0.00, 0.00) | 0.121, 0.0231, 0.0119, 0.0047 | 0.0104, 0.0083, 0.0043 | 0.936, 0.955 | 1.007 | no | not met |
| candidate rows only x 4 | 0.251, 0.110 | 0.00% (0.00, 0.00) | 0.120, 0.0771, 0.0953, 0.132 | 0.0445, 0.0817, 0.143 | 1.021, 1.016 | 1.156 | yes | not met |
| candidate rows only x 6 | 0.362, 0.262 | 0.00% (0.00, 0.00) | 0.118, 0.264, 0.384, 0.551 | 0.190, 0.354, 0.558 | 1.038, 1.018 | 1.337 | yes | not met |
| candidate rows only x 8 | 0.443, 0.370 | 0.00% (0.00, 0.00) | 0.115, 0.381, 0.548, 0.777 | 0.271, 0.480, 0.781 | 1.037, 1.018 | 1.431 | yes | not met |
| update-gate bias -1 | 0.221, 0.0356 | 0.00% (0.00, 0.00) | 0.173, 6.1e-05, 6.0e-08, 9.1e-14 | 2.0e-05, 2.7e-08, 4.7e-14 | 0.500, 0.512 | 0.520 | no | not met |
| update-gate bias -2 | 0.253, 0.0370 | 0.18% (0.00, 0.37) | 0.206, 6.5e-06, 7.2e-10, 2.2e-17 | 3.1e-06, 4.1e-10, 9.5e-18 | 0.402, 0.421 | 0.422 | no | not met |

- **The share of ring 3 rises with the multiple and passes a third only where a difference grows.** At 12 letters it is 0.0028 unchanged, 0.0100 at 1.5, 0.0267 at 2, 0.141 at 3 and 0.418 at 4. At 4 the difference over all cells is 0.0962, 0.129 and 0.200 at 10, 20 and 40 positions after the changed byte: it grows, by 1.02 to 1.03 per position. At 3 it still shrinks (0.951 and 0.959 per position).
- **The candidate rows carry the passage.** Multiplying them alone gives the same shares within 0.02 (0.0100, 0.0268, 0.138, 0.405) and the same boundary of growth. The gates' rows matter only for the gate values beyond 0.05 / 0.95, which stay at 0.00% when the candidate rows alone are multiplied and reach 12.80% at 8 when all three are. The limit of 20% is not reached by any start tried; growth is reached first.
- **A lower update gate does not help in E47.** With the bias lowered by 1 or by 2 the share of ring 3 is 0.0046 and 0.0072. A cell then takes more of its candidate, but the difference also fades faster (0.50 and 0.40 per position against 0.65), and what is gained per connection is lost in time.
- **The far cells' states grow with the multiple:** their mean size is 0.0334 unchanged, 0.0611 at 3, 0.115 at 4 and 0.292 at 8, against 0.178 to 0.391 in the interface.

Further starts, tried after these results [part `more`]: multiples between 2 and 4, a multiple together with a lowered update-gate bias, and the ten connections multiplied without the cell's weight on its own state.

| Start | First digit changed, 0 letters | 4 letters | 12 letters | Last digit changed | At `=`, 12 letters | Farthest group at its largest | Between items at `=` | Way back: interface over ring 3; interface at its largest |
|---|---|---|---|---|---|---|---|---|
| recurrent weights x 2.5 | 0.409, 0.117, 0.0322 | 0.424, 0.138, 0.0501 | 0.432, 0.150, 0.0632 | 0.352, 0.0576, 0.0056 | 0.829, 0.550, 0.366 | 0.0183 | 0.403, 0.134, 0.0590 | 0.0645; 0.0170 |
| recurrent weights x 3.25 | 0.497, 0.186, 0.0644 | 0.546, 0.247, 0.118 | 0.591, 0.326, 0.202 | 0.412, 0.0835, 0.0106 | 0.811, 0.624, 0.464 | 0.0887 | 0.567, 0.301, 0.191 | 0.206; 0.0967 |
| recurrent weights x 3.5 | 0.523, 0.210, 0.0774 | 0.584, 0.288, 0.148 | 0.645, 0.399, 0.274 | 0.428, 0.0919, 0.0127 | 0.826, 0.656, 0.523 | 0.166 | 0.623, 0.374, 0.265 | 0.258; 0.173 |
| recurrent weights x 3.75 | 0.547, 0.234, 0.0913 | 0.618, 0.329, 0.180 | 0.693, 0.467, 0.349 | 0.443, 0.0999, 0.0149 | 0.847, 0.688, 0.582 | 0.283 | 0.674, 0.449, 0.347 | 0.296; 0.289 |
| candidate rows only x 3.5 | 0.535, 0.214, 0.0814 | 0.586, 0.287, 0.154 | 0.643, 0.377, 0.264 | 0.438, 0.0947, 0.0123 | 0.864, 0.652, 0.545 | 0.128 | 0.613, 0.353, 0.246 | 0.230; 0.145 |
| recurrent weights x 2, update-gate bias -1 | 0.459, 0.112, 0.0315 | 0.459, 0.114, 0.0343 | 0.459, 0.115, 0.0348 | 0.444, 0.0861, 0.0126 | 0.898, 0.614, 0.459 | 0.0093 | 0.430, 0.0999, 0.0318 | 0.0346; 0.0083 |
| recurrent weights x 2.5, update-gate bias -1 | 0.512, 0.166, 0.0594 | 0.517, 0.176, 0.0721 | 0.518, 0.180, 0.0773 | 0.485, 0.114, 0.0203 | 0.889, 0.683, 0.515 | 0.0218 | 0.499, 0.164, 0.0732 | 0.0808; 0.0196 |
| recurrent weights x 3, update-gate bias -1 | 0.564, 0.230, 0.0988 | 0.585, 0.268, 0.140 | 0.597, 0.301, 0.183 | 0.515, 0.140, 0.0292 | 0.833, 0.740, 0.625 | 0.0529 | 0.584, 0.281, 0.177 | 0.198; 0.0529 |
| recurrent weights x 3.5, update-gate bias -1 | 0.612, 0.300, 0.148 | 0.655, 0.379, 0.241 | 0.705, 0.496, 0.404 | 0.538, 0.163, 0.0391 | 0.902, 0.859, 0.832 | 0.213 | 0.693, 0.473, 0.389 | 0.341; 0.221 |
| recurrent weights x 2, update-gate bias -2 | 0.535, 0.136, 0.0407 | 0.535, 0.137, 0.0417 | 0.535, 0.138, 0.0420 | 0.531, 0.121, 0.0239 | 0.941, 0.709, 0.552 | 0.0124 | 0.517, 0.123, 0.0397 | 0.0405; 0.0113 |
| recurrent weights x 2.5, update-gate bias -2 | 0.570, 0.195, 0.0762 | 0.573, 0.200, 0.0839 | 0.573, 0.202, 0.0863 | 0.555, 0.155, 0.0363 | 0.895, 0.760, 0.598 | 0.0276 | 0.567, 0.188, 0.0838 | 0.0857; 0.0239 |
| recurrent weights x 3, update-gate bias -2 | 0.613, 0.270, 0.130 | 0.628, 0.300, 0.166 | 0.634, 0.322, 0.200 | 0.577, 0.187, 0.0503 | 0.894, 0.861, 0.793 | 0.0610 | 0.636, 0.308, 0.198 | 0.201; 0.0564 |
| recurrent weights x 3.5, update-gate bias -2 | 0.658, 0.355, 0.202 | 0.693, 0.429, 0.299 | 0.744, 0.556, 0.475 | 0.593, 0.215, 0.0651 | 1.031, 1.093, 1.102 | 0.223 | 0.745, 0.529, 0.445 | 0.349; 0.240 |
| candidate rows x 3, update-gate bias -1 | 0.575, 0.233, 0.103 | 0.592, 0.264, 0.142 | 0.602, 0.287, 0.175 | 0.531, 0.146, 0.0297 | 0.955, 0.815, 0.713 | 0.0491 | 0.588, 0.268, 0.165 | 0.165; 0.0447 |
| candidate rows x 3, update-gate bias -2 | 0.622, 0.270, 0.134 | 0.634, 0.293, 0.164 | 0.639, 0.308, 0.188 | 0.591, 0.192, 0.0506 | 1.037, 1.004, 0.911 | 0.0574 | 0.639, 0.294, 0.182 | 0.167; 0.0497 |
| ten connections x 3, own weight x 1 | 0.463, 0.156, 0.0518 | 0.482, 0.184, 0.0805 | 0.494, 0.204, 0.105 | 0.400, 0.0778, 0.0091 | 0.891, 0.660, 0.521 | 0.0322 | 0.471, 0.191, 0.102 | 0.108; 0.0303 |
| ten connections x 3.5, own weight x 1 | 0.513, 0.201, 0.0767 | 0.550, 0.252, 0.130 | 0.590, 0.316, 0.204 | 0.435, 0.0959, 0.0132 | 0.934, 0.740, 0.606 | 0.0724 | 0.571, 0.300, 0.198 | 0.207; 0.0773 |
| ten connections x 4, own weight x 1 | 0.557, 0.246, 0.106 | 0.618, 0.328, 0.193 | 0.711, 0.473, 0.358 | 0.463, 0.113, 0.0180 | 0.988, 0.831, 0.709 | 0.218 | 0.686, 0.452, 0.354 | 0.323; 0.236 |

| Start | Mean size of a state: interface, other cells | Gate values beyond (reset, update) | 40 letters, all cells: 0, 10, 20, 40 after | Farthest group: 10, 20, 40 after | Factor per position: 10 to 20, 20 to 40 | Expected factor | Grows | Three conditions |
|---|---|---|---|---|---|---|---|---|
| recurrent weights x 2.5 | 0.199, 0.0480 | 0.00% (0.00, 0.00) | 0.123, 0.0124, 0.0035, 4.2e-04 | 0.0044, 0.0019, 3.0e-04 | 0.882, 0.899 | 0.925 | no | not met |
| recurrent weights x 3.25 | 0.223, 0.0709 | 0.00% (0.00, 0.00) | 0.123, 0.0395, 0.0315, 0.0249 | 0.0169, 0.0218, 0.0224 | 0.978, 0.988 | 1.046 | no | not met |
| recurrent weights x 3.5 | 0.234, 0.0834 | 0.00% (0.01, 0.00) | 0.123, 0.0552, 0.0557, 0.0683 | 0.0251, 0.0425, 0.0686 | 1.001, 1.010 | 1.083 | yes | not met |
| recurrent weights x 3.75 | 0.246, 0.0982 | 0.01% (0.02, 0.01) | 0.123, 0.0745, 0.0886, 0.134 | 0.0363, 0.0734, 0.139 | 1.018, 1.021 | 1.118 | yes | not met |
| candidate rows only x 3.5 | 0.227, 0.0801 | 0.00% (0.00, 0.00) | 0.121, 0.0443, 0.0392, 0.0385 | 0.0228, 0.0309, 0.0401 | 0.988, 0.999 | 1.085 | no | not met |
| recurrent weights x 2, update-gate bias -1 | 0.234, 0.0455 | 0.00% (0.00, 0.00) | 0.172, 0.0023, 9.8e-05, 3.2e-07 | 0.0011, 6.1e-05, 1.9e-07 | 0.730, 0.751 | 0.768 | no | not met |
| recurrent weights x 2.5, update-gate bias -1 | 0.246, 0.0563 | 0.01% (0.00, 0.02) | 0.171, 0.0093, 0.0016, 7.4e-05 | 0.0048, 0.0011, 6.1e-05 | 0.840, 0.857 | 0.889 | no | not met |
| recurrent weights x 3, update-gate bias -1 | 0.265, 0.0771 | 0.04% (0.00, 0.07) | 0.170, 0.0303, 0.0146, 0.0055 | 0.0174, 0.0123, 0.0055 | 0.929, 0.953 | 1.005 | no | not met |
| recurrent weights x 3.5, update-gate bias -1 | 0.291, 0.115 | 0.14% (0.02, 0.26) | 0.169, 0.0753, 0.0733, 0.107 | 0.0502, 0.0724, 0.115 | 0.997, 1.019 | 1.107 | yes | not met |
| recurrent weights x 2, update-gate bias -2 | 0.269, 0.0493 | 0.48% (0.00, 0.96) | 0.204, 0.0012, 2.6e-05, 2.3e-08 | 6.6e-04, 1.7e-05, 1.5e-08 | 0.682, 0.704 | 0.719 | no | not met |
| recurrent weights x 2.5, update-gate bias -2 | 0.283, 0.0627 | 0.78% (0.00, 1.56) | 0.203, 0.0074, 8.9e-04, 2.1e-05 | 0.0043, 6.5e-04, 2.0e-05 | 0.809, 0.830 | 0.864 | no | not met |
| recurrent weights x 3, update-gate bias -2 | 0.304, 0.0885 | 1.29% (0.01, 2.57) | 0.200, 0.0303, 0.0124, 0.0054 | 0.0201, 0.0113, 0.0059 | 0.914, 0.959 | 1.001 | no | not met |
| recurrent weights x 3.5, update-gate bias -2 | 0.332, 0.136 | 2.45% (0.04, 4.85) | 0.198, 0.0859, 0.0861, 0.120 | 0.0664, 0.0892, 0.127 | 1.000, 1.017 | 1.117 | yes | not met |
| candidate rows x 3, update-gate bias -1 | 0.261, 0.0746 | 0.00% (0.00, 0.00) | 0.170, 0.0235, 0.0098, 0.0039 | 0.0150, 0.0087, 0.0041 | 0.916, 0.955 | 1.006 | no | not met |
| candidate rows x 3, update-gate bias -2 | 0.301, 0.0855 | 0.22% (0.00, 0.43) | 0.201, 0.0228, 0.0082, 0.0038 | 0.0166, 0.0078, 0.0043 | 0.903, 0.963 | 1.003 | no | not met |
| ten connections x 3, own weight x 1 | 0.203, 0.0539 | 0.00% (0.00, 0.00) | 0.123, 0.0170, 0.0059, 0.0014 | 0.0077, 0.0041, 1.0e-03 | 0.900, 0.930 | - | no | not met |
| ten connections x 3.5, own weight x 1 | 0.216, 0.0684 | 0.00% (0.00, 0.00) | 0.123, 0.0327, 0.0216, 0.0176 | 0.0167, 0.0165, 0.0126 | 0.960, 0.990 | - | no | not met |
| ten connections x 4, own weight x 1 | 0.234, 0.0909 | 0.01% (0.01, 0.01) | 0.123, 0.0584, 0.0635, 0.0931 | 0.0332, 0.0530, 0.0843 | 1.009, 1.019 | - | yes | not met |

- **Growth begins between 3.25 and 3.5; a third is reached between 3.5 and 3.75.** At 3.25 the share of ring 3 is 0.202 and a difference shrinks slowly (0.978 and 0.988 per position). At 3.5 the share is 0.274 and the difference is 0.0552, 0.0557 and 0.0683 at 10, 20 and 40 positions after: it grows. At 3.75 the share is 0.349.
- **The largest share without growth is 0.264,** with the candidate rows alone times 3.5. There the difference is 0.0443, 0.0392 and 0.0385 at 10, 20 and 40 positions after; by the rule it does not grow, but it has all but stopped shrinking (0.999 per position at the end).
- **A lowered update-gate bias moves the figures a little and not the conclusion.** At a multiple of 3 the share of ring 3 is 0.141 without the shift, 0.183 with the bias lowered by 1 and 0.200 with it lowered by 2, without growth. At 3.5 both shifted starts grow.
- **Leaving out the cell's weight on itself** moves the boundary by about half a unit (0.204 without growth at 3.5, growth at 4) and changes nothing else.
- **The way back rises with the way out** under every uniform change: at a multiple of 3 the interface receives 0.149 of a difference put into ring 3 (0.0026 unchanged), at 3.25 it receives 0.206.

A change in one direction [part `directed`]. Of a far cell's ten connections, 3.76 (ring 1), 3.93 (ring 2) and 5.58 (ring 3) come from a cell that lies nearer the interface by grid distance; 9 of the 768 far cells have none. Only the weights on these connections are multiplied, in all three gates. Interface cells have no such connection and are unchanged.

| Start | First digit changed, 0 letters | 4 letters | 12 letters | Last digit changed | At `=`, 12 letters | Farthest group at its largest | Between items at `=` | Way back: interface over ring 3; interface at its largest |
|---|---|---|---|---|---|---|---|---|
| unchanged | 0.186, 0.0198, 0.0023 | 0.186, 0.0206, 0.0027 | 0.187, 0.0207, 0.0028 | 0.169, 0.0116, 4.5e-04 | 0.631, 0.334, 0.165 | 7.7e-04 | 0.162, 0.0167, 0.0024 | 0.0026; 7.1e-04 |
| connections from nearer the interface x 2 | 0.377, 0.0827, 0.0190 | 0.378, 0.0870, 0.0236 | 0.379, 0.0877, 0.0246 | 0.338, 0.0464, 0.0036 | 1.689, 1.747, 1.706 | 0.0065 | 0.335, 0.0731, 0.0221 | 0.0026; 7.2e-04 |
| connections from nearer the interface x 3 | 0.579, 0.197, 0.0684 | 0.582, 0.212, 0.0908 | 0.584, 0.214, 0.0966 | 0.511, 0.104, 0.0122 | 2.810, 4.202, 5.817 | 0.0240 | 0.525, 0.184, 0.0879 | 0.0025; 7.3e-04 |
| connections from nearer the interface x 4 | 0.795, 0.375, 0.175 | 0.802, 0.414, 0.249 | 0.806, 0.421, 0.273 | 0.686, 0.185, 0.0290 | 3.702, 7.156, 12.344 | 0.0650 | 0.731, 0.366, 0.249 | 0.0025; 7.6e-04 |
| connections from nearer the interface x 5 | 1.027, 0.627, 0.372 | 1.041, 0.711, 0.562 | 1.047, 0.731, 0.639 | 0.863, 0.287, 0.0573 | 4.466, 10.335, 20.696 | 0.146 | 0.950, 0.632, 0.571 | 0.0025; 7.8e-04 |
| connections from nearer the interface x 6 | 1.273, 0.954, 0.691 | 1.298, 1.111, 1.087 | 1.308, 1.160, 1.276 | 1.039, 0.406, 0.100 | 5.136, 13.173, 28.998 | 0.283 | 1.176, 0.971, 1.068 | 0.0025; 8.1e-04 |
| connections from nearer the interface x 8 | 1.778, 1.767, 1.693 | 1.835, 2.133, 2.661 | 1.864, 2.301, 3.281 | 1.380, 0.670, 0.222 | 6.205, 17.542, 40.906 | 0.684 | 1.605, 1.713, 2.152 | 0.0024; 8.7e-04 |
| connections from nearer the interface x 12 | 2.700, 3.312, 3.625 | 2.848, 4.111, 5.459 | 2.958, 4.741, 7.250 | 1.949, 1.146, 0.491 | 7.541, 21.834, 44.951 | 1.364 | 2.273, 2.751, 3.377 | 0.0019; 8.9e-04 |

| Start | Mean size of a state: interface, other cells | Gate values beyond (reset, update) | 40 letters, all cells: 0, 10, 20, 40 after | Farthest group: 10, 20, 40 after | Factor per position: 10 to 20, 20 to 40 | Grows | Three conditions |
|---|---|---|---|---|---|---|---|
| unchanged | 0.178, 0.0334 | 0.00% (0.00, 0.00) | 0.122, 7.9e-04, 1.1e-05, 2.6e-09 | 9.6e-05, 3.2e-06, 1.2e-09 | 0.651, 0.660 | no | not met |
| connections from nearer the interface x 2 | 0.178, 0.0374 | 0.00% (0.00, 0.00) | 0.122, 0.0016, 4.5e-05, 4.4e-08 | 9.7e-04, 5.3e-05, 6.2e-08 | 0.702, 0.707 | no | not met |
| connections from nearer the interface x 3 | 0.178, 0.0445 | 0.00% (0.00, 0.00) | 0.122, 0.0040, 2.4e-04, 8.5e-07 | 0.0043, 3.7e-04, 1.4e-06 | 0.756, 0.754 | no | not met |
| connections from nearer the interface x 4 | 0.178, 0.0561 | 0.00% (0.00, 0.00) | 0.122, 0.0096, 0.0010, 1.1e-05 | 0.0130, 0.0018, 2.0e-05 | 0.800, 0.797 | no | not met |
| connections from nearer the interface x 5 | 0.178, 0.0739 | 0.00% (0.00, 0.00) | 0.122, 0.0206, 0.0035, 1.0e-04 | 0.0317, 0.0062, 1.8e-04 | 0.837, 0.838 | no | met at 0, 4 and 12 letters |
| connections from nearer the interface x 6 | 0.178, 0.0987 | 0.00% (0.00, 0.00) | 0.122, 0.0387, 0.0091, 6.2e-04 | 0.0638, 0.0167, 0.0012 | 0.865, 0.874 | no | met at 0, 4 and 12 letters |
| connections from nearer the interface x 8 | 0.179, 0.158 | 0.15% (0.14, 0.16) | 0.122, 0.0912, 0.0357, 0.0063 | 0.157, 0.0662, 0.0117 | 0.911, 0.917 | no | met at 0, 4 and 12 letters |
| connections from nearer the interface x 12 | 0.180, 0.247 | 3.96% (3.91, 4.01) | 0.122, 0.191, 0.123, 0.0606 | 0.324, 0.222, 0.112 | 0.957, 0.965 | no | met at 0, 4 and 12 letters |

- **At a multiple of 5 the three conditions are met at every length:** ring 3 holds 0.372, 0.562 and 0.639 of the interface's difference at 0, 4 and 12 letters, no gate value lies beyond 0.05 / 0.95, and a difference shrinks by 0.84 per position. At 4 the share is 0.273 at 12 letters and 0.175 at 0 letters, which does not meet the first condition.
- **No multiple up to 12 makes a difference grow,** because the multiplied connections all lead away from the interface and a difference cannot return along them. At 6 and above ring 3 holds a larger difference than the interface (1.276 at 6); the far cells then amplify what they receive, and at 12 a far cell's state (0.247) is larger than an interface cell's (0.180).
- **The way back is not changed:** 0.0025 or 0.0026 at every multiple up to 6, against 0.0026 unchanged. A difference arrives in ring 3 and stays there.

### 4.2 E47, arm `place`

The task's list [part `variants`]. Entries of the first table: ring 1, ring 2, ring 3, each over the interface.

| Start | First digit changed, 0 letters | 4 letters | 12 letters | Last digit changed | At `=`, 12 letters | Farthest group at its largest | Between items at `=` | Way back: interface over ring 3; interface at its largest |
|---|---|---|---|---|---|---|---|---|
| unchanged | 0.141, 0.0047, 1.9e-05 | 0.146, 0.0085, 7.0e-05 | 0.148, 0.0115, 3.3e-04 | 0.101, 7.4e-04, 3.7e-07 | 4.343, 2.904, 0.163 | 9.5e-05 | 0.127, 0.0088, 1.7e-04 | 9.7e-04; 0.0011 |
| recurrent weights x 1.5 | 0.201, 0.0098, 5.8e-05 | 0.211, 0.0183, 2.2e-04 | 0.215, 0.0258, 0.0011 | 0.143, 0.0015, 1.2e-06 | 2.494, 1.810, 0.139 | 3.5e-04 | 0.188, 0.0198, 5.6e-04 | 0.0034; 0.0041 |
| recurrent weights x 2 | 0.251, 0.0159, 1.3e-04 | 0.269, 0.0307, 4.8e-04 | 0.277, 0.0458, 0.0025 | 0.176, 0.0025, 2.6e-06 | 1.568, 1.037, 0.0951 | 9.3e-04 | 0.249, 0.0349, 0.0013 | 0.0086; 0.0106 |
| recurrent weights x 3 | 0.320, 0.0281, 3.3e-04 | 0.364, 0.0573, 0.0013 | 0.405, 0.0993, 0.0074 | 0.222, 0.0046, 7.4e-06 | 0.812, 0.357, 0.0366 | 0.0043 | 0.376, 0.0760, 0.0036 | 0.0346; 0.0473 |
| recurrent weights x 4 | 0.356, 0.0371, 5.6e-04 | 0.417, 0.0746, 0.0020 | 0.494, 0.136, 0.0113 | 0.252, 0.0066, 1.5e-05 | 0.620, 0.216, 0.0214 | 0.0143 | 0.498, 0.121, 0.0067 | 0.0966; 0.145 |
| recurrent weights x 6 | 0.394, 0.0509, 0.0011 | 0.464, 0.0996, 0.0035 | 0.568, 0.189, 0.0198 | 0.289, 0.0105, 4.0e-05 | 0.653, 0.252, 0.0303 | 0.0698 | 0.666, 0.219, 0.0164 | 0.372; 0.656 |
| recurrent weights x 8 | 0.430, 0.0661, 0.0018 | 0.504, 0.126, 0.0056 | 0.612, 0.239, 0.0327 | 0.302, 0.0140, 8.0e-05 | 0.700, 0.322, 0.0510 | 0.168 | 0.737, 0.309, 0.0318 | 0.960; 2.010 |
| candidate rows only x 1.5 | 0.202, 0.0098, 5.8e-05 | 0.212, 0.0184, 2.2e-04 | 0.216, 0.0258, 0.0011 | 0.143, 0.0016, 1.2e-06 | 2.567, 1.879, 0.145 | 3.5e-04 | 0.188, 0.0198, 5.6e-04 | 0.0034; 0.0040 |
| candidate rows only x 2 | 0.254, 0.0161, 1.3e-04 | 0.272, 0.0309, 4.9e-04 | 0.280, 0.0459, 0.0025 | 0.177, 0.0025, 2.6e-06 | 1.657, 1.127, 0.104 | 9.2e-04 | 0.249, 0.0351, 0.0013 | 0.0084; 0.0104 |
| candidate rows only x 3 | 0.329, 0.0290, 3.3e-04 | 0.372, 0.0589, 0.0013 | 0.407, 0.100, 0.0076 | 0.226, 0.0047, 7.5e-06 | 0.873, 0.422, 0.0462 | 0.0042 | 0.374, 0.0763, 0.0037 | 0.0324; 0.0441 |
| candidate rows only x 4 | 0.370, 0.0395, 5.9e-04 | 0.429, 0.0792, 0.0022 | 0.503, 0.142, 0.0129 | 0.256, 0.0068, 1.5e-05 | 0.645, 0.236, 0.0269 | 0.0138 | 0.487, 0.118, 0.0068 | 0.0864; 0.129 |
| candidate rows only x 6 | 0.411, 0.0543, 0.0012 | 0.474, 0.101, 0.0040 | 0.583, 0.187, 0.0219 | 0.295, 0.0109, 4.2e-05 | 0.670, 0.244, 0.0336 | 0.0726 | 0.646, 0.198, 0.0154 | 0.252; 0.419 |
| candidate rows only x 8 | 0.447, 0.0695, 0.0020 | 0.518, 0.129, 0.0067 | 0.630, 0.236, 0.0352 | 0.316, 0.0146, 8.2e-05 | 0.711, 0.302, 0.0527 | 0.177 | 0.733, 0.276, 0.0290 | 0.450; 0.859 |
| update-gate bias -1 | 0.235, 0.0122, 8.1e-05 | 0.236, 0.0173, 2.6e-04 | 0.236, 0.0192, 8.5e-04 | 0.207, 0.0025, 1.8e-06 | 7.180, 13.257, 2.001 | 1.3e-04 | 0.205, 0.0153, 5.1e-04 | 0.0014; 0.0013 |
| update-gate bias -2 | 0.334, 0.0201, 1.8e-04 | 0.333, 0.0256, 5.1e-04 | 0.334, 0.0270, 0.0014 | 0.317, 0.0050, 4.6e-06 | 7.990, 28.215, 6.636 | 1.4e-04 | 0.296, 0.0219, 8.9e-04 | 0.0017; 0.0014 |

| Start | Mean size of a state: interface, other cells | Gate values beyond (reset, update) | 40 letters, all cells: 0, 10, 20, 40 after | Farthest group: 10, 20, 40 after | Factor per position: 10 to 20, 20 to 40 | Grows | Three conditions |
|---|---|---|---|---|---|---|---|
| unchanged | 0.178, 0.0168 | 0.00% (0.00, 0.00) | 0.122, 0.0014, 2.1e-04, 2.9e-05 | 5.2e-06, 2.4e-05, 2.4e-05 | 0.825, 0.907 | no | not met |
| recurrent weights x 1.5 | 0.182, 0.0180 | 0.00% (0.00, 0.00) | 0.122, 0.0030, 5.7e-04, 1.0e-04 | 1.8e-05, 8.7e-05, 9.6e-05 | 0.848, 0.918 | no | not met |
| recurrent weights x 2 | 0.189, 0.0199 | 0.00% (0.00, 0.00) | 0.122, 0.0060, 0.0015, 3.4e-04 | 4.5e-05, 2.3e-04, 2.9e-04 | 0.872, 0.927 | no | not met |
| recurrent weights x 3 | 0.212, 0.0257 | 0.00% (0.00, 0.00) | 0.123, 0.0228, 0.0131, 0.0065 | 1.7e-04, 0.0011, 0.0022 | 0.946, 0.966 | no | not met |
| recurrent weights x 4 | 0.254, 0.0356 | 0.06% (0.06, 0.05) | 0.123, 0.0673, 0.0749, 0.0850 | 4.4e-04, 0.0036, 0.0142 | 1.011, 1.006 | yes | not met |
| recurrent weights x 6 | 0.336, 0.0599 | 2.86% (2.92, 2.80) | 0.122, 0.200, 0.274, 0.346 | 0.0016, 0.0182, 0.0966 | 1.032, 1.012 | yes | not met |
| recurrent weights x 8 | 0.384, 0.0800 | 9.93% (10.08, 9.79) | 0.121, 0.301, 0.394, 0.488 | 0.0038, 0.0436, 0.196 | 1.027, 1.011 | yes | not met |
| candidate rows only x 1.5 | 0.182, 0.0180 | 0.00% (0.00, 0.00) | 0.122, 0.0029, 5.6e-04, 1.0e-04 | 1.8e-05, 8.7e-05, 9.6e-05 | 0.848, 0.918 | no | not met |
| candidate rows only x 2 | 0.188, 0.0198 | 0.00% (0.00, 0.00) | 0.122, 0.0057, 0.0014, 3.2e-04 | 4.5e-05, 2.3e-04, 2.9e-04 | 0.871, 0.928 | no | not met |
| candidate rows only x 3 | 0.209, 0.0256 | 0.00% (0.00, 0.00) | 0.121, 0.0196, 0.0101, 0.0046 | 1.6e-04, 0.0010, 0.0019 | 0.936, 0.961 | no | not met |
| candidate rows only x 4 | 0.246, 0.0352 | 0.00% (0.00, 0.00) | 0.120, 0.0540, 0.0552, 0.0649 | 4.3e-04, 0.0034, 0.0121 | 1.002, 1.008 | yes | not met |
| candidate rows only x 6 | 0.351, 0.0640 | 0.00% (0.00, 0.00) | 0.118, 0.175, 0.264, 0.379 | 0.0017, 0.0178, 0.0941 | 1.042, 1.018 | yes | not met |
| candidate rows only x 8 | 0.434, 0.0931 | 0.00% (0.00, 0.00) | 0.115, 0.274, 0.412, 0.560 | 0.0041, 0.0419, 0.187 | 1.042, 1.015 | yes | not met |
| update-gate bias -1 | 0.221, 0.0200 | 0.00% (0.00, 0.00) | 0.173, 6.7e-04, 1.0e-04, 1.4e-05 | 1.6e-05, 4.4e-05, 2.4e-05 | 0.829, 0.905 | no | not met |
| update-gate bias -2 | 0.253, 0.0219 | 0.43% (0.00, 0.87) | 0.206, 5.2e-04, 7.8e-05, 1.1e-05 | 2.8e-05, 5.6e-05, 2.2e-05 | 0.826, 0.909 | no | not met |

- **No start of the list brings ring 3 above 0.0352,** and the starts with the largest shares (multiples of 4 and more) are starts at which a difference grows. Without growth the share of ring 3 is 0.0074 at most (recurrent weights times 3) and 0.0076 (candidate rows times 3).
- **The ratio at `=` is large and means little.** With the update-gate bias lowered by 2, ring 2 holds 28 times the interface's difference at `=`; the interface has by then lost its own (5.2e-04 over all cells 10 positions after the changed byte), and the far cells, updated rarely, still hold the little they received.
- Among the further starts [part `more`], the largest share of ring 3 without growth is 0.0300 (recurrent weights times 3.5 with the update-gate bias lowered by 2); growth begins between multiples of 3.5 and 3.75 (between 3.25 and 3.5 in the arm `one`).
- Under the change in one direction [part `directed`] the share of ring 3 at 12 letters is 0.0460 at a multiple of 5, 0.202 at 8 and 0.613 at 12; at 12 a difference grows (1.007 and 1.004 per position), and at 0 letters the share is 0.0306 even there. The three conditions are not met by any start tried.

### 4.3 E50, `C16-in`

The task's list [part `variants`]. Entries of the first table: the cells that read neither byte over the cells that read a value of either.

| Start | First digit changed, 0 letters | 4 letters | 12 letters | Last digit changed | At `=`, 12 letters | Farthest group at its largest |
|---|---|---|---|---|---|---|
| unchanged | 0.266 | 0.267 | 0.266 | 0.244 | 0.865 | 0.0972 |
| recurrent weights x 1.5 | 0.388 | 0.390 | 0.390 | 0.354 | 0.918 | 0.146 |
| recurrent weights x 2 | 0.500 | 0.504 | 0.505 | 0.452 | 0.952 | 0.196 |
| recurrent weights x 3 | 0.687 | 0.709 | 0.720 | 0.608 | 0.976 | 0.295 |
| recurrent weights x 4 | 0.820 | 0.863 | 0.892 | 0.714 | 0.948 | 0.403 |
| recurrent weights x 6 | 0.930 | 0.955 | 0.962 | 0.829 | 0.961 | 1.457 |
| recurrent weights x 8 | 0.955 | 0.967 | 0.965 | 0.880 | 0.958 | 2.051 |
| candidate rows only x 1.5 | 0.384 | 0.385 | 0.385 | 0.350 | 0.917 | 0.143 |
| candidate rows only x 2 | 0.493 | 0.496 | 0.497 | 0.444 | 0.943 | 0.188 |
| candidate rows only x 3 | 0.674 | 0.689 | 0.697 | 0.591 | 0.970 | 0.279 |
| candidate rows only x 4 | 0.800 | 0.830 | 0.859 | 0.686 | 0.960 | 0.377 |
| candidate rows only x 6 | 0.905 | 0.937 | 0.953 | 0.763 | 0.958 | 1.150 |
| candidate rows only x 8 | 0.932 | 0.956 | 0.978 | 0.784 | 0.988 | 1.870 |
| update-gate bias -1 | 0.464 | 0.464 | 0.463 | 0.447 | 0.914 | 0.136 |
| update-gate bias -2 | 0.744 | 0.747 | 0.744 | 0.720 | 0.889 | 0.161 |

| Start | Mean size of a state | Gate values beyond (reset, update) | 40 letters, all cells: 0, 10, 20, 40 after | Farthest group: 10, 20, 40 after | Factor per position: 10 to 20, 20 to 40 | Expected factor | Grows | Three conditions |
|---|---|---|---|---|---|---|---|---|
| unchanged | 0.132 | 0.00% (0.00, 0.00) | 0.216, 0.0013, 1.4e-05, 2.2e-09 | 0.0012, 1.3e-05, 2.0e-09 | 0.637, 0.645 | 0.637 | no | not met |
| recurrent weights x 1.5 | 0.138 | 0.00% (0.00, 0.00) | 0.216, 0.0034, 1.0e-04, 1.2e-07 | 0.0031, 9.7e-05, 1.1e-07 | 0.703, 0.715 | 0.705 | no | met at 0, 4 and 12 letters |
| recurrent weights x 2 | 0.147 | 0.00% (0.00, 0.00) | 0.216, 0.0081, 6.1e-04, 4.7e-06 | 0.0077, 6.0e-04, 4.2e-06 | 0.772, 0.784 | 0.772 | no | met at 0, 4 and 12 letters |
| recurrent weights x 3 | 0.173 | 0.01% (0.00, 0.01) | 0.216, 0.0364, 0.0129, 0.0022 | 0.0355, 0.0127, 0.0021 | 0.901, 0.915 | 0.900 | no | met at 0, 4 and 12 letters |
| recurrent weights x 4 | 0.216 | 0.07% (0.06, 0.09) | 0.216, 0.113, 0.110, 0.107 | 0.111, 0.108, 0.105 | 0.997, 0.998 | 1.006 | no | met at 0, 4 and 12 letters |
| recurrent weights x 6 | 0.304 | 4.09% (4.03, 4.14) | 0.215, 0.356, 0.465, 0.542 | 0.351, 0.460, 0.539 | 1.027, 1.008 | 1.120 | yes | not met |
| recurrent weights x 8 | 0.358 | 14.97% (14.84, 15.09) | 0.214, 0.541, 0.644, 0.702 | 0.529, 0.638, 0.692 | 1.018, 1.004 | 1.161 | yes | not met |
| candidate rows only x 1.5 | 0.138 | 0.00% (0.00, 0.00) | 0.216, 0.0033, 9.4e-05, 1.0e-07 | 0.0030, 9.1e-05, 9.1e-08 | 0.701, 0.710 | 0.705 | no | met at 0, 4 and 12 letters |
| candidate rows only x 2 | 0.146 | 0.00% (0.00, 0.00) | 0.216, 0.0077, 5.3e-04, 3.1e-06 | 0.0073, 5.2e-04, 2.9e-06 | 0.765, 0.774 | 0.773 | no | met at 0, 4 and 12 letters |
| candidate rows only x 3 | 0.173 | 0.00% (0.00, 0.00) | 0.215, 0.0328, 0.0098, 0.0011 | 0.0318, 0.0095, 0.0010 | 0.886, 0.895 | 0.902 | no | met at 0, 4 and 12 letters |
| candidate rows only x 4 | 0.217 | 0.00% (0.00, 0.00) | 0.214, 0.0948, 0.0737, 0.0486 | 0.0931, 0.0720, 0.0470 | 0.975, 0.979 | 1.013 | no | met at 0, 4 and 12 letters |
| candidate rows only x 6 | 0.336 | 0.00% (0.00, 0.01) | 0.212, 0.274, 0.360, 0.459 | 0.273, 0.355, 0.455 | 1.028, 1.012 | 1.148 | yes | not met |
| candidate rows only x 8 | 0.424 | 0.00% (0.00, 0.01) | 0.208, 0.384, 0.544, 0.716 | 0.381, 0.538, 0.704 | 1.035, 1.014 | 1.214 | yes | not met |
| update-gate bias -1 | 0.154 | 0.06% (0.00, 0.12) | 0.297, 8.7e-05, 5.3e-08, 2.3e-14 | 8.4e-05, 5.1e-08, 2.2e-14 | 0.477, 0.481 | 0.475 | no | met at 0, 4 and 12 letters |
| update-gate bias -2 | 0.172 | 1.60% (0.00, 3.21) | 0.356, 7.5e-06, 3.2e-10, 1.0e-17 | 7.3e-06, 3.0e-10, 1.0e-17 | 0.366, 0.422 | 0.367 | no | met at 0, 4 and 12 letters |

- **The recurrent weights times 1.5 meet the three conditions at every length** (0.388, 0.390, 0.390), and with the last digit changed the share is 0.354. The candidate rows alone times 1.5 give 0.385, the update-gate bias lowered by 1 gives 0.463.
- **Every multiple up to 4 meets them.** At 4 a difference neither shrinks nor grows to any extent (0.113, 0.110 and 0.107 at 10, 20 and 40 positions after; 0.997 and 0.998 per position). At 6 it grows.
- **In absolute terms the cells that read neither byte receive little at any of these starts.** Their largest difference at any later position, over the reading cells' at the changed byte, is 0.0972 unchanged, 0.146 at 1.5, 0.196 at 2, 0.295 at 3 and 0.403 at 4. The share is larger than these figures because the reading cells themselves keep only half of their difference from one position to the next.
- All 18 further starts [part `more`] meet the three conditions, with shares between 0.613 and 0.950 at 12 letters.

### 4.4 E49, `in16` and `I16`, and E42's `F10`

Every cell reads the difference, so the first table gives the distribution over the cells in place of shares: a cell's root-mean-square difference over the later positions, over that of all cells, at the 5th, 50th and 95th percentile of the 256 cells, the participation over the later positions and at the changed byte, and the largest root-mean-square difference of all cells at any later position over that at the changed byte. Items of 12 letters, first digit changed [part `variants`].

| Start | Cell at 5% | Median cell | Cell at 95% | Participation, later positions | Participation at the changed byte | All cells at their largest |
|---|---|---|---|---|---|---|
| unchanged | 0.494 | 0.942 | 1.475 | 0.734 | 0.747 | 0.525 |
| recurrent weights x 1.5 | 0.508 | 0.941 | 1.492 | 0.720 | 0.745 | 0.547 |
| recurrent weights x 2 | 0.514 | 0.917 | 1.485 | 0.707 | 0.741 | 0.575 |
| recurrent weights x 3 | 0.536 | 0.903 | 1.516 | 0.688 | 0.726 | 0.647 |
| recurrent weights x 4 | 0.509 | 0.897 | 1.576 | 0.646 | 0.704 | 0.733 |
| recurrent weights x 6 | 0.518 | 0.893 | 1.543 | 0.678 | 0.644 | 1.394 |
| recurrent weights x 8 | 0.537 | 0.910 | 1.487 | 0.728 | 0.578 | 2.212 |
| candidate rows only x 1.5 | 0.499 | 0.929 | 1.476 | 0.718 | 0.747 | 0.544 |
| candidate rows only x 2 | 0.529 | 0.909 | 1.510 | 0.701 | 0.746 | 0.570 |
| candidate rows only x 3 | 0.539 | 0.889 | 1.524 | 0.674 | 0.743 | 0.634 |
| candidate rows only x 4 | 0.508 | 0.865 | 1.585 | 0.616 | 0.738 | 0.708 |
| candidate rows only x 6 | 0.493 | 0.876 | 1.597 | 0.628 | 0.724 | 1.031 |
| candidate rows only x 8 | 0.527 | 0.925 | 1.489 | 0.718 | 0.707 | 1.960 |
| update-gate bias -1 | 0.477 | 0.916 | 1.561 | 0.690 | 0.755 | 0.347 |
| update-gate bias -2 | 0.498 | 0.917 | 1.570 | 0.673 | 0.755 | 0.266 |

| Start | Mean size of a state | Gate values beyond (reset, update) | 40 letters, all cells: 0, 10, 20, 40 after | Factor per position: 10 to 20, 20 to 40 | Expected factor | Grows |
|---|---|---|---|---|---|---|
| unchanged | 0.177 | 0.00% (0.00, 0.00) | 0.241, 0.0014, 1.5e-05, 2.1e-09 | 0.635, 0.641 | 0.635 | no |
| recurrent weights x 1.5 | 0.182 | 0.00% (0.00, 0.00) | 0.240, 0.0034, 9.2e-05, 8.9e-08 | 0.698, 0.707 | 0.701 | no |
| recurrent weights x 2 | 0.189 | 0.00% (0.00, 0.00) | 0.240, 0.0076, 5.1e-04, 2.9e-06 | 0.764, 0.771 | 0.765 | no |
| recurrent weights x 3 | 0.214 | 0.00% (0.00, 0.00) | 0.239, 0.0319, 0.0094, 0.0012 | 0.885, 0.900 | 0.886 | no |
| recurrent weights x 4 | 0.253 | 0.13% (0.11, 0.16) | 0.238, 0.0930, 0.0638, 0.0409 | 0.963, 0.978 | 0.981 | no |
| recurrent weights x 6 | 0.327 | 5.27% (5.41, 5.13) | 0.236, 0.294, 0.328, 0.400 | 1.011, 1.010 | 1.091 | yes |
| recurrent weights x 8 | 0.371 | 15.98% (16.20, 15.75) | 0.234, 0.455, 0.535, 0.597 | 1.016, 1.005 | 1.142 | yes |
| candidate rows only x 1.5 | 0.182 | 0.00% (0.00, 0.00) | 0.240, 0.0032, 8.2e-05, 6.4e-08 | 0.693, 0.699 | 0.701 | no |
| candidate rows only x 2 | 0.189 | 0.00% (0.00, 0.00) | 0.240, 0.0069, 4.2e-04, 1.6e-06 | 0.755, 0.757 | 0.765 | no |
| candidate rows only x 3 | 0.213 | 0.00% (0.00, 0.00) | 0.238, 0.0269, 0.0065, 3.5e-04 | 0.868, 0.864 | 0.887 | no |
| candidate rows only x 4 | 0.255 | 0.00% (0.00, 0.00) | 0.236, 0.0751, 0.0458, 0.0141 | 0.952, 0.943 | 0.990 | no |
| candidate rows only x 6 | 0.354 | 0.00% (0.00, 0.00) | 0.231, 0.225, 0.244, 0.289 | 1.008, 1.008 | 1.126 | yes |
| candidate rows only x 8 | 0.428 | 0.00% (0.00, 0.00) | 0.224, 0.337, 0.477, 0.565 | 1.035, 1.009 | 1.207 | yes |
| update-gate bias -1 | 0.221 | 0.00% (0.00, 0.00) | 0.344, 8.6e-05, 4.6e-08, 1.5e-14 | 0.471, 0.473 | 0.471 | no |
| update-gate bias -2 | 0.254 | 0.72% (0.00, 1.44) | 0.413, 6.7e-06, 2.2e-10, 1.2e-17 | 0.356, 0.434 | 0.360 | no |

`F10`, for comparison [part `variants`]:

| Start | Cell at 5% | Median cell | Cell at 95% | Participation, later positions | Participation at the changed byte | All cells at their largest |
|---|---|---|---|---|---|---|
| unchanged | 0.547 | 0.927 | 1.487 | 0.746 | 0.733 | 0.513 |
| recurrent weights x 3 | 0.560 | 0.900 | 1.581 | 0.688 | 0.707 | 0.644 |
| recurrent weights x 4 | 0.524 | 0.877 | 1.595 | 0.607 | 0.688 | 0.742 |
| recurrent weights x 6 | 0.561 | 0.907 | 1.502 | 0.708 | 0.640 | 1.459 |

| Start | Mean size of a state | Gate values beyond (reset, update) | 40 letters, all cells: 0, 10, 20, 40 after | Factor per position: 10 to 20, 20 to 40 | Expected factor | Grows |
|---|---|---|---|---|---|---|
| unchanged | 0.179 | 0.00% (0.00, 0.00) | 0.243, 0.0014, 1.9e-05, 3.7e-09 | 0.648, 0.653 | 0.633 | no |
| recurrent weights x 3 | 0.217 | 0.00% (0.00, 0.00) | 0.243, 0.0392, 0.0135, 0.0016 | 0.899, 0.898 | 0.887 | no |
| recurrent weights x 4 | 0.256 | 0.15% (0.17, 0.14) | 0.243, 0.112, 0.0880, 0.0749 | 0.976, 0.992 | 0.983 | no |
| recurrent weights x 6 | 0.327 | 5.37% (5.34, 5.40) | 0.242, 0.307, 0.377, 0.445 | 1.021, 1.008 | 1.092 | yes |

- **The distribution over the cells does not depend on the start** (cell at the 5th percentile 0.477 to 0.539, participation 0.616 to 0.734 over the fifteen starts). There is nothing here for a change of the start to do.
- **In the networks of 256 cells growth begins between multiples of 4 and 6,** later than in E47 (between 3.25 and 3.5). At 4 a difference shrinks by 0.963 and 0.978 per position in `in16` and by 0.976 and 0.992 in `F10`; at 6 it grows in both. The further starts [part `more`] do not grow up to 3.75, with or without a lowered update-gate bias.

