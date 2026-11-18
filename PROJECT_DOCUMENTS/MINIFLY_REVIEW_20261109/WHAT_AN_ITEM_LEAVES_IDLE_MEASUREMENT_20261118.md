# What an item of another family leaves idle in the networks with a sparse input (18 November 2026, 21:53 UTC)

Reviewer's measurement on saved weights. Nothing was trained, torch was not used, no code of another branch was run, and no weight file was written to disk: each file was read with `git show` into memory, one at a time. Every figure below is printed by `what_an_item_leaves_idle.py` in this folder, on the line whose section number is given; a few ratios are formed from printed figures. Every statement is for seed 73129 only.

## 1. The question

Section 29 of `MEMO_TO_COORDINATOR_LEARNED_STRUCTURE_20261115.md` found that in the dense network every item drives every cell, also an item made of bytes that the network has never met, and `CELLS_IN_USE_IN_LIMITED_NETWORKS_20261117.md` found the same in the networks with ten connections a cell and in E49's `in16`. The networks with a sparse input did not exist then. The question here is whether, in those networks, an item of another family leaves cells, connections between cells, input weights or input directions of the readout idle.

## 2. Terms

**The definitions of section 29.1 of the memo, quoted exactly.**

> *A cell in use.* For a cell at one position of an item, with update gate z, candidate n and previous state h (so that the new state is (1 - z) n + z h), every change made at that position to the cell's incoming connections is a product that contains the factor (1 - z)(1 - n²) + z(1 - z)|h - n|, under back-propagation and under the sweep alike. I call it the change factor. Where it is zero the cell's incoming connections cannot change, whatever the error. Every change to a connection that leaves a cell is a product that contains the cell's state.

> *Directions in use.* A matrix of connections receives input vectors. The directions in use by a set of items are those spanned by the input vectors that the items present to the matrix, with a constant appended for the bias. A change of the matrix that is zero on those directions leaves the matrix's output on those items as it was; if this holds at every matrix, the whole forward pass on those items is as it was.

**How they are applied here, and where they had to be made precise or adapted.**

- **A cell in use at a position**: its change factor at that position is 0.05 or more (the mark of section 29.2 of the memo); the mark 0.01 is given beside it. The change factor never reaches exactly zero, so a mark is needed. **The state size** of a cell at a position is the absolute value of its new state; it decides the changes of the connections that leave the cell. **A cell in use for a kind of items** (the form of the memo's table): its change factor, averaged over all pairs (item, position) of that kind, is 0.05 or more.
- **The size of the change of one weight** (an extension needed to count weights rather than cells). For a weight of gate row g of cell i whose input is u (the state of a source cell before the position, or a value of the byte's code), the derivative of the cell's new state with respect to the weight at one position is f_g · u, with f_r = (1 - z)(1 - n²)|gh_n| r(1 - r), f_z = z(1 - z)|h - n|, f_n = (1 - z)(1 - n²), times r for the recurrent map (gh_n is the recurrent part of the candidate's pre-activation). The memo's change factor is f_n of the input map plus f_z. The size of the change of the weight for an item is the sum over the item's positions of f_g · |u|: the largest change one unit of error at the cell could make. A weight **is changed by an item** when this size is larger than a share (10⁻³; 10⁻² is given as well) of the largest size over the kept weights of the same map for the matched item of kind (a). The scale is thus set by the first family, item by item.
- **Adaptation to the masked maps.** The change factor needs none. Counts of weights are over the kept weights only: in the recurrent map the three gate rows for each of a cell's 11 sources (its ten connections and itself), 8,448 weights; in the input map the three gate rows for each value a cell reads (393,216 in `C-dense`, 35,328 in `C46-one`, 12,288 in `C16-in` and `in16`). A cell **receives input** at a position when at least one value it reads is not zero there. The readout's two maps are not masked in any of the eight networks (`lm` reads all 256 cells, `l2` all 512 values), so the directions in use at the readout needed no adaptation.
- **Directions in use** at a matrix: as `directions_in_use.py` takes them, the eigenvectors of the second-moment matrix of the input vectors with a constant 1 appended, counted at 99.9% and 99.99% of the trace, and as the rank (eigenvalues above 10⁻¹⁰ of the largest, the threshold of that script). For the readout the vectors are those at the three answer positions (inputs `=`, first digit, second digit), as in the memo: for `lm` the cells' states after the position, for `l2` the 512 values of the byte's code. **Inside the directions of kind (a)**: the share of the squared length of another kind's vectors (constant appended) that lies in the span of kind (a)'s directions.

**The networks** (all are E42's `F10`, 256 cells, each reading itself and 10 fixed cells; all update every cell at every position).

| Name | File | Input | Rule; receipt row |
|---|---|---|---|
| `C-dense` S, B | default branch, `exp45/e50/e50_C-dense_<S|B>_73129.pt` | fixed code (each of the 46 symbols turns on 11 of 512 values; any other byte has the zero code); a cell reads all 512 values | 24,576 |
| `C46-one` S, B | same place, `e50_C46-one_…` | fixed code; a cell reads 46 values, one of every used byte | 49,152 |
| `C16-in` S, B | same place, `e50_C16-in_…` | fixed code; a cell reads 16 values drawn at random | 49,152 |
| `in16-row` S | `85417d06`, `exp45/e57/e57_in16_row_S_u49152_73129.pt` | learned code (byte table of 8 values and dense `l1` to 512); a cell reads 16 values; sweep with the scale `row` | 49,152 |
| `in16-none` B | `85417d06`, `exp45/e57/e57_in16_none_B_73129.pt` | as above, back-propagation | 24,576 |

**The items.** 520 items of each kind, matched item by item: for each of the four operations and each number of letters from 0 to 12, ten taught pairs (the taught half of `split(task, 0.5, 73129)`), drawn with `Random("idle|73129")`; 13 lengths of 10 to 22 positions. An item's input is the start symbol, the prompt, and the answer without its end symbol.
- (a) the first family: task byte p, q, v or w, modulo 89;
- (b) the same item with the task byte replaced by r, t, x or n (the second family, modulo 97) and the answer computed modulo 97; all other bytes as in (a). The two differ at position 1 in every item and also in an answer digit in 130 items (`[3]`);
- (c) the item of (a) with each of the 46 symbols replaced, one for one in the order of E50's list, by the bytes 128 to 173. The start symbol is one of the 46 and becomes byte 173, so every byte of a kind-(c) item lies outside the first family's symbols.

## 3. Method and checks

The script computes the forward pass from the tensors of each weight file alone (the code from the file's `code_table`, or from its byte table and `l1`; the three maps as full matrices times the file's masks) and keeps, at every position, r, z, n, the recurrent part of the candidate and the previous state.

1. *The files* (`[3]`). Each file's SHA-256 equals its receipt's `weights_sha256`, each receipt is complete, and every weight outside the three masks is zero. The worker's branch `origin/claude/e50-fixed-input-code-017dSE` is at 85417d06, so it was not fetched again.
2. *A second form of the forward* (`[3]`). In all eight networks the file's three masks equal the draws of the reviewer's readers (`e50_code_reader`, with `OneOfEach` for `C46-one`; `e49_interface_reader` for `in16`), and on 24 items the script's logits differ from those readers' by at most 7.6e-6 (largest logit 27 to 44), with the same largest output at every position.
3. *The local derivatives* (`[3]`). At 40 random kept weights the factors f_g · |u| agree with central finite differences in float64 to a relative difference of at most 1.6e-5 in six networks and 2.6e-4 and 4.4e-4 in `C-dense` S and B (relative to the larger of the derivative and 10⁻⁴).
4. *Repetition.* Two complete runs printed the same figures (84 s each).

## 4. The check against the receipts

The script's forward reproduces every count of the last scored row of each receipt: the panel behind A (300 untaught pairs of each operation at 12 letters), R (1,000 reduction pairs each of add and sub), the taught panel, and the length test at 12 and 40 letters (whole answer right). The R counts by first and second byte were not checked.

| Network | Row | A: add, sub, max, min | R: add, sub | Taught | Length 12, 40 | Equal |
|---|---|---|---|---|---|---|
| `C-dense` S | 24,576 | 273, 281, 298, 289 | 950, 923 | 297, 297, 297, 298 | 273, 263 | yes |
| `C-dense` B | 24,576 | 295, 297, 294, 292 | 980, 970 | 300, 298, 295, 298 | 295, 284 | yes |
| `C46-one` S | 49,152 | 239, 260, 296, 293 | 854, 861 | 274, 284, 296, 297 | 239, 77 | yes |
| `C46-one` B | 49,152 | 279, 289, 297, 293 | 953, 950 | 298, 297, 299, 295 | 279, 181 | yes |
| `C16-in` S | 49,152 | 139, 153, 293, 290 | 547, 349 | 218, 224, 299, 295 | 139, 29 | yes |
| `C16-in` B | 49,152 | 252, 253, 295, 288 | 837, 717 | 284, 288, 298, 298 | 252, 170 | yes |
| `in16-row` S | 49,152 | 248, 258, 294, 288 | 838, 835 | 281, 289, 296, 299 | 248, 126 | yes |
| `in16-none` B | 24,576 | 261, 265, 291, 296 | 885, 779 | 282, 278, 295, 298 | 261, 207 | yes |

The networks answer 451 (`C16-in` S) to 519 (`C-dense` B) of the 520 items of kind (a) with both digits right (`[5.1]`).

## 5. Tables

"l / m / h" is least / median / most over the 520 items (or over all pairs (item, position)).

### 5.1 Cells (`[5.1]`)

| Network | (a): cells in use at a position, l / m / h | (a): at position 1, median | (a): at every position of an item, median | (b): at position 1, median | (c): at a position | Cells in use for (a) and not for (b); not for (c) (by the mean; item by item; at position 1) | At position 1, in use for (b) and not for (a), median (most) | (c): cells with mean state size below 0.05 |
|---|---|---|---|---|---|---|---|---|
| `C-dense` S | 11 / 78 / 179 | 36 | 1 | 256 | 256 at every pair | 0; 0 | 220 (225) | 235 |
| `C-dense` B | 11 / 104 / 197 | 34 | 0 | 256 | 256 at every pair | 0; 0 | 222 (230) | 233 |
| `C46-one` S | 252 / 256 / 256 | 256 | 253 | 256 | 256 at every pair | 0; 0 | 0 (0) | 100 |
| `C46-one` B | 239 / 254 / 256 | 256 | 242 | 256 | 256 at every pair | 0; 0 | 0 (3) | 129 |
| `C16-in` S | 251 / 255 / 256 | 256 | 249 | 256 | 256 at every pair | 0; 0 | 0 (1) | 69 |
| `C16-in` B | 241 / 253 / 256 | 253 | 229 | 256 | 253 / 256 / 256 | 0; 0 | 3 (6) | 16 |
| `in16-row` S | 229 / 253 / 256 | 254 | 231 | 256 | 240 / 256 / 256 | 0; 0 | 2 (18) | 0 |
| `in16-none` B | 216 / 250 / 256 | 256 | 212 | 256 | 224 / 256 / 256 | 0; 0 | 0 (37) | 0 |

Every item of every kind has all 256 cells in use at some position, also at the mark 0.01. The count of cells in use for kind (a) and not for kind (b) or (c) is 0 in all eight networks in all three forms (by the mean change factor, item by item, at position 1) and at both marks. In the six networks with the fixed code all items of kind (c) receive the zero code at every position, so the states of two kind-(c) items of the same length are equal at every position (largest difference 0); in `C-dense` such items leave the state of all but 25 cells below 0.05 in size at the median position, while the change factor of every cell stays at 0.05 or more. In the two `C-dense` networks the first family keeps most cells' gates shut at most positions (78 and 104 cells in use at the median position); an item with the zero code at the task byte's position has all 256 cells in use there.

### 5.2 Connections between cells (`[5.2]`; 8,448 kept recurrent weights; share 10⁻³, with 10⁻² in the last column)

| Network | (a): changed by an item, median; by some item | (b): by an item, median; also changed by the matched (a) item, median share; by some item; by some (b) item and no (a) item | (c): the same four | Changed by some (b), some (c) item and by no (a) item, share 10⁻² |
|---|---|---|---|---|
| `C-dense` S | 5,982; 7,703 | 7,986; 0.745; 8,322; 637 | 2,801; 0.904; 3,026; 66 | 715; 16 |
| `C-dense` B | 6,218; 7,867 | 8,063; 0.767; 8,351; 502 | 2,806; 0.935; 2,984; 34 | 743; 10 |
| `C46-one` S | 8,447; 8,448 | 8,446; 0.9999; 8,448; 0 | 5,096; 1.0000; 5,822; 0 | 0; 0 |
| `C46-one` B | 8,440; 8,448 | 8,428; 0.9996; 8,448; 0 | 4,230; 0.9998; 4,873; 0 | 0; 0 |
| `C16-in` S | 8,420; 8,448 | 8,399; 0.9985; 8,448; 0 | 5,332; 0.9987; 6,274; 0 | 1; 0 |
| `C16-in` B | 8,394; 8,448 | 8,360; 0.9978; 8,448; 0 | 7,190; 0.9980; 7,618; 0 | 5; 0 |
| `in16-row` S | 8,429; 8,448 | 8,429; 0.9989; 8,448; 0 | 8,430; 0.9977; 8,448; 0 | 4; 38 |
| `in16-none` B | 8,393; 8,448 | 8,391; 0.9973; 8,448; 0 | 8,417; 0.9939; 8,448; 0 | 14; 135 |

In `C-dense` the 745 (S) and 581 (B) kept weights that no item of kind (a) changes are all in the reset-gate rows, in 80 and 64 receiving cells; the 637 and 502 that kind (b) changes and kind (a) does not are also all reset-gate weights, in 77 and 62 receiving cells. At the share 10⁻² the few weights changed by (b) or (c) and by no (a) item in the other networks are reset-gate and update-gate weights (no candidate-row weight), in at most 22 receiving cells.

### 5.3 The input map (`[5.3]`)

| Network | Kept input weights | Cells receiving input at position 1, median: (a), (b), (c) | Cells receiving input at some position of the item, median: (b), (c) | Mean input contribution at position 1: (a), (b), (c) | Kept input weights changed by an item, median: (a), (b), (c) | By some item: (a); (b), of which by no (a) item; (c) | (b): also changed by the matched (a) item, median share |
|---|---|---|---|---|---|---|---|
| `C-dense` S | 393,216 | 256, 0, 0 | 256, 0 | 10.42, 0, 0 | 33,924; 32,912; 0 | 178,079; 162,855, 1,573; 0 | 0.936 |
| `C-dense` B | 393,216 | 256, 0, 0 | 256, 0 | 12.00, 0, 0 | 37,142; 36,421; 0 | 182,732; 170,610, 957; 0 | 0.947 |
| `C46-one` S | 35,328 | 256, 0, 0 | 256, 0 | 0.85, 0, 0 | 9,584; 8,862; 0 | 35,182; 32,175, 0; 0 | 0.980 |
| `C46-one` B | 35,328 | 256, 0, 0 | 256, 0 | 1.24, 0, 0 | 9,490; 8,748; 0 | 35,156; 32,153, 2; 0 | 0.980 |
| `C16-in` S | 12,288 | 78, 0, 0 | 254, 0 | 0.46, 0, 0 | 3,215; 3,000; 0 | 11,993; 10,952, 7; 0 | 0.979 |
| `C16-in` B | 12,288 | 78, 0, 0 | 254, 0 | 0.63, 0, 0 | 3,076; 2,862; 0 | 11,888; 10,884, 9; 0 | 0.976 |
| `in16-row` S | 12,288 | 256, 256, 256 | 256, 256 | 1.64, 0.84, 0.77 | 11,519; 11,504; 11,386 | 12,270; 12,270, 2; 12,145, 2 | 0.988 ((c): 0.976) |
| `in16-none` B | 12,288 | 256, 256, 256 | 256, 256 | 2.19, 1.15, 1.12 | 12,176; 12,181; 12,215 | 12,286; 12,286, 0; 12,284, 1 | 0.997 ((c): 0.993) |

Share 10⁻³. The input contribution is the sum over the three gates of |input-map row times code|, bias excluded, averaged over cells. In the six networks with the fixed code, a byte outside the 46 symbols gives no input to any cell: the task byte of kind (b) reaches 0 cells and changes 0 input weights, and an item of kind (c) reaches 0 cells at any position and changes 0 of the kept input weights (520 of 520 items change none, at both shares); only the biases of the input map receive a change from it. The input weights that kind (b) changes are weights on values of the first family's own bytes. In the fixed code 6 of the 512 values belong to no byte; `C-dense` reads them in all 256 cells (4,608 kept weights), `C46-one` in none, `C16-in` in 41 cells (123 kept weights). In the two networks with the learned code, every byte has a code with no value exactly zero, and the codes of r, t, x, n and of the bytes 128 to 173 lie inside the span of the codes of the 46 symbols to 99.26% to 99.79% (r, t, x, n, least) and 98.83% to 99.72% (bytes 128 to 173, least) of their squared length, and inside the nine leading directions of those codes to 92.7% to 97.6% (least).

### 5.4 The readout (`[5.4]`; vectors at the three answer positions, constant appended)

| Network | `lm` (257): directions of (a) at 99.9%, 99.99%, rank | `lm`: (b), (c) inside (a)'s 99.99% directions (random direction) | `lm`: (c)'s squared length without the constant, against (a) | `l2` (513): directions of (a) at 99.9%, 99.99%, rank | `l2`: (b), (c) inside (a)'s directions | `l2`: (c)'s squared length without the constant, against (a) |
|---|---|---|---|---|---|---|
| `C-dense` S | 245, 256, 257 | 0.99972, 0.67174 (0.996) | 0.0036 | 11, 11, 11 | 1.00000, 0.08704 | 0 |
| `C-dense` B | 240, 255, 257 | 0.99989, 0.53290 (0.992) | 0.0030 | 11, 11, 11 | 1.00000, 0.08704 | 0 |
| `C46-one` S | 222, 252, 257 | 0.99983, 0.99685 (0.981) | 0.115 | 11, 11, 11 | 1.00000, 0.08704 | 0 |
| `C46-one` B | 199, 243, 257 | 0.99981, 0.98703 (0.946) | 0.069 | 11, 11, 11 | 1.00000, 0.08704 | 0 |
| `C16-in` S | 203, 246, 257 | 0.99979, 0.99823 (0.957) | 0.151 | 11, 11, 11 | 1.00000, 0.08704 | 0 |
| `C16-in` B | 203, 245, 257 | 0.99978, 0.99929 (0.953) | 0.752 | 11, 11, 11 | 1.00000, 0.08704 | 0 |
| `in16-row` S | 203, 242, 257 | 0.99961, 0.99823 (0.942) | 0.661 | 10, 11, 11 | 1.00000, 0.99000 | 0.520 |
| `in16-none` B | 205, 246, 257 | 0.99961, 0.99912 (0.957) | 0.645 | 11, 11, 11 | 1.00000, 0.94054 | 0.329 |

At `lm` the items of kind (a) have rank 257 in every network, so every input direction of `lm` is in use by the first family and whatever another kind presents lies in their span (share 1.00000 at the rank). At `l2` the first family uses 11 of 513 directions (the codes of `=` and the ten digits, which are the bytes at the answer positions, with the constant). Kind (b) presents the same bytes there (share 1.00000). In the fixed code kind (c) presents only the constant (its code is zero), so it changes no weight of `l2` and only its bias; 8.7% of that constant vector lies in the span of kind (a)'s vectors. In the learned code kind (c) presents codes that lie 94% to 99% inside the 11 directions of kind (a). In `C-dense` the vectors that kind (c) presents to `lm` lie 33% and 47% outside kind (a)'s 99.99% directions, but their part without the constant has 0.3% of the squared length of kind (a)'s, and kind (c)'s vectors in the fixed code occupy only 5 to 12 directions of their own, since all its items have the same states at the same position.

## 6. Plain statements

| Network | What an item of another family leaves idle, and what it does not |
|---|---|
| `C-dense` S | Among cells, none (all 256 are in use at the task byte's position for kinds (b) and (c), against 36 for the family); among connections between cells, none that the family uses: kind (b) changes 637 reset-gate weights that no family item changes and kind (c) only 3,026 of the 8,448; among input weights, a byte outside the 46 symbols drives none (kind (c) changes 0 of 393,216), while the other bytes of a kind-(b) item drive weights of the family's bytes; among readout directions, none at `lm` (rank 257), while at `l2` a kind-(c) item presents only the constant. |
| `C-dense` B | As `C-dense` S: no cell idle (all 256 in use at position 1 for (b) and (c), 34 for the family), 502 reset-gate weights changed by (b) and by no family item, 2,984 recurrent weights changed by (c); 0 input weights and 0 `l2` weights changed by a byte outside the 46 symbols; all 257 directions of `lm` in use by the family. |
| `C46-one` S | No cell and no connection between cells is idle (every recurrent weight that (b) or (c) changes is also changed by the family; (c) changes 5,822 of 8,448); a byte outside the 46 symbols drives no input weight and no `l2` weight; every `lm` direction is in use by the family. |
| `C46-one` B | The same: no cell idle, no recurrent weight changed by (b) or (c) that the family does not change ((c) changes 4,873 of 8,448), no input or `l2` weight driven by a byte outside the 46 symbols, every `lm` direction in use. |
| `C16-in` S | The same: no cell idle, no recurrent weight changed by (b) or (c) and not by the family at the share 10⁻³ ((c) changes 6,274 of 8,448), no input or `l2` weight driven by a byte outside the 46 symbols (the family's task byte itself reaches only 78 cells), every `lm` direction in use. |
| `C16-in` B | The same: no cell idle, no recurrent weight changed by (b) or (c) and not by the family at 10⁻³ ((c) changes 7,618 of 8,448), no input or `l2` weight driven by a byte outside the 46 symbols, every `lm` direction in use. |
| `in16-row` S | Nothing is left idle: every cell is in use, kinds (b) and (c) change all 8,448 recurrent weights and 12,270 and 12,145 of the 12,288 input weights, almost all also changed by the family, and their codes and states lie inside the family's directions at `l2` (99.0%) and `lm`. |
| `in16-none` B | Nothing is left idle: every cell is in use, kinds (b) and (c) change all 8,448 recurrent weights and 12,286 and 12,284 of the 12,288 input weights, and their codes lie 94% inside the family's 11 directions at `l2` and their states inside the family's directions at `lm`. |

**What a network would need in its input for a new byte to drive only weights that no earlier byte drives.** The figures show it for the input map and `l2` only. In the fixed code a byte's code drives exactly the input weights and `l2` columns of the values it turns on; a byte outside the 46 drives none. A new byte would drive only input weights that no earlier byte drives if it had a code on values that no earlier byte turns on, and if cells read those values. In the present code 6 values belong to no byte; `C-dense` reads them in all 256 cells, `C16-in` in 41 cells, `C46-one` in none. In the learned code no such room exists: every byte, met or not, has a code with every value non-zero, lying 98.8% or more inside the span of the family's codes. In none of the eight networks would such a code leave the connections between cells or `lm` idle: an item whose bytes all have the zero code still has every cell in use at every position and changes 2,984 to 7,618 recurrent weights, all but at most 66 of them weights that the family also changes, and the family uses all 257 directions of `lm`.

## 7. What was not measured

- **The errors.** Every weight count uses the largest change one unit of error at the receiving cell could make; whether an error arrives, and its size, needs gradients of a task not yet learned.
- **Learning.** Nothing was trained; what learning the second family would change is not measured.
- **Other seeds**, E57's other `in16` runs (`matrix`, `rowcells`, `rowcol`) and its `F10` and `G` runs, and the files of E50 at update 24,576.
- **The start symbol kept in kind (c).** Kind (c) replaces it by byte 173; a version that keeps it was not run.
- **The R counts by first and second byte** and the recorded answers files of the receipts.
- **Directions at the recurrent map and at `l3`**, and at the readout outside the answer positions.
- **Other marks and shares** than 0.05 and 0.01 for cells and 10⁻³ and 10⁻² for weights.

## 8. Command

From the repository root: `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -I PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/what_an_item_leaves_idle.py [names]` (names: `C-dense_S C-dense_B C46-one_S C46-one_B C16-in_S C16-in_B in16-row_S in16-none_B`; default all). 84 s for all eight on a host of four cores; it writes no file, holds one weight file in memory at a time, and imports `compare_pt_tensors`, `np_gru`, `e42_cells_reader`, `e49_interface_reader` and `e50_code_reader` from this folder. Free disk was 1.2 GB before the first run and after the last.
