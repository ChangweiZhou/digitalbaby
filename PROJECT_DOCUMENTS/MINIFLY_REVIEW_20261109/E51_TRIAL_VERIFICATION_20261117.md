# Verification of the trial of E51, Part A: a new task learned in cells held in reserve (seed 73129)

17 November 2026, begun 23:16 UTC. Written for the outside reviewer by an agent that was given the worker's message and files and not the worker's reasoning. It is a recomputation, and it does not replace the independent audit that the programme's rules ask for.

## Conclusion

**What holds.** Every figure of the worker's message and every cell of its tables is reproduced from the sixteen receipts (2,428 cells of `E51_TABLES_A.md` and 228 values of `E51_READ_A.json`, no difference). A forward pass written for this verification from the brief's text reproduces, from the saved weights, every count to which it was applied: the first family's scores at rows 64 and 4,096 of the four runs of the arm `keyed` and at row 4,096 of the other twelve runs, the new task's scores with the reserve cells open and shut, and the number of reserve cells whose gate is open. In the four `keyed` runs all 32,884 scored answers of the first family (the network's first task: four operations under modulus 89) are, at rows 64 and 4,096, the answers given before the new task, item for item, while the new task is learned: the unrelated task on 300 of 300 never-taught items under both rules, the conflicting family on 758 and 775 of 800. The tensors that the rule leaves unchanged are bit-equal to the start. The nine predictions held. The seven checks before the runs record what the brief asks, and the figures recomputed here agree with them.

**What must be stated more exactly.**

1. *The answers are unchanged; the outputs are not.* By row 4,096 the first family's outputs have moved by up to 5.3e-4 (3.8e-4 and 5.3e-4 in the two unrelated runs, 5.5e-5 and 7.3e-5 in the two conflict runs). The cause is that the weights are stored in single precision after each projected step, so that a small part of the change of the input map's and of `l2`'s old rows, less than four millionths of it at the input map, remains in the directions that the first family uses and grows with the number of updates. The smallest difference between the largest and the next largest output of a first-family answer is 1.8e-4 under the sweep and 7.8e-4 under back-propagation, the same size. No answer changed in this trial, but nothing in the construction bounds this movement over a longer run.
2. *The separation is supplied by the program.* The first family is kept because the forward pass reads each item's first byte and, when that byte is one of 46 listed bytes, sets the reserve cells' gates to shut and the added units of the readout to zero. With these two lines removed, the same weights of row 4,096 answer 0.00% of the first family's reduction pairs after the unrelated task and 0.40% and 0.00% after the conflicting family. The arm `keyed-free` shows the other half: with the two lines and without the rule the reduction score is at 0.0 by row 32 on `unrelated` and at 16.1 and 17.4 at row 64 on `conflict`.
3. *The new task is not learned in the reserve alone.* The brief's reading "learned in the reserve" asks only that the score with the reserve cells shut be below 20%. For the unrelated task it is 0%; but the task is also at 0.0% and 0.3% when the incoming connections of the old cells and old readout units are put back to their starting values, so the connections that carry it include those from reserve cells into the old units. For the conflicting family the old cells and old readout units alone answer 41.8% and 38.9% of the never-taught items (19.0% and 8.2% at the start). Those items are the ones that the first family's operation already answers: with the reserve cells shut the network answers 94.8% and 92.4% of the 344 items whose operands are below 89 and whose answer is the same under both moduli, and 0 of 427 and 2 of 427 of the items whose answer differs between the two moduli. In both tasks the first family nevertheless stays unchanged for the reason the brief gives, which does not depend on where the new task is learned.

**What could not be checked.** The updates themselves were not repeated, so the rule is verified by what it left in the saved weights at rows 64 and 4,096 and not step by step; check 3's figures after eight updates are read from its receipt only. The new entries of the recurrent matrix could be compared with another saved network of the same seed on 5,085 of 147,456 entries only. Nothing here speaks for a second seed, for a third task, or for a run longer than 4,096 updates.

## Terms, material and method

**Terms**, as the brief defines them. *The network* is a byte table (258 rows of 8 numbers), a layer `l1` (8 to H units, with tanh; its output for a byte is the byte's *code*, z1), a recurrent layer of U cells of the GRU kind (for each cell a reset gate r, an update gate z and a candidate n; the new state is (1 − z) n + z h, so that z = 1 keeps the state, which is zero at the start of every item), and a readout z2 = tanh(`l2` z1 + `lm` h) followed by `l3`, which gives 257 *outputs* at each position (called logits in the request); the answer byte is the symbol with the largest output. *Present size* is H = 512, U = 256; *half width* is H = 256, U = 128. *Back-propagation* (rule B) uses the exact gradient; *the sweep* (rule S) sends the error back through each linear map by the signs of its weights times one scale. *The first family* is add, sub, max and min under modulus 89 with the code bytes `p q v w`; its *reduction pairs* are the pairs of add and sub whose answer needs the modulus (3,916 of 7,921 for each), and the *reduction score* is the mean of the two percentages of whole answers right on them, without letters between the operands. *A birth* is a network trained on the first family alone; the *half-width birth* is E41's, with half of the pairs taught. *The embedded network*, or *the start*, is that birth placed inside a network of the present size: cells 0 to 127 are the *old cells*, cells 128 to 255 the *reserve cells*; units 0 to 255 of `l1` and of z2 are *old*, units 256 to 511 *added*. *The bytes in use* are the 46 symbols that occur in 2,080 taught items of the first family (the start symbol included, the end symbol excluded). *The two added lines* of the keyed forward act on every item whose first byte is a byte in use: they set z to 1 for the reserve cells and z2 to 0 for the added units. *The directions in use* at a matrix are the directions spanned by the input vectors that those 2,080 items present to it, a constant 1 appended. *The rule* removes from every change of the old units' rows of four matrices (the input map, the recurrent matrix, `lm`, `l2`) its component along the directions in use, and restores `l1`, the old columns and the bias of `l3`, and the 46 rows of the byte table after every step. The four *arms*: `open` and `rehearse` start from the embedded network with a bias of 10 on the reserve cells' update gates (*the start with a bias*) and change everything, `rehearse` with one item of the first family for each new item on average; `keyed-free` starts from *the keyed start* (no bias, the two added lines) and changes everything; `keyed` is the keyed start with the rule. The two *tasks*: `unrelated` (E43's task on bytes 128 to 225) and `conflict` (the same four operations under modulus 97 with the code bytes `r t x n`, here called the conflicting family). *A row* is a number of updates at which a run is scored (0, 16, 32, 64, 128, 256, 512, 1,024, 2,048, 4,096). *A receipt* is the file of results that a run or a check writes. The *never-taught score* of the new task is the percentage of whole answers right on its held-out panel at 12 letters (300 items for `unrelated`, 800 for `conflict`). The brief's fixed readings: *kept* (the reduction score at row 4,096 within 2 points of row 0), *learned* (the never-taught score reaches 80% at some row and stays there at every later row), *learned in the reserve* (learned, and below 20% at row 4,096 when the reserve cells are held shut on every item). *The lead* of an answer byte, a term of this note, is the largest output at an answer position minus the next largest.

**Material.** The worker's branch `claude/e51-reserve-017dSEpW` at 2aa469c1: the runner `exp51_reserve.py`, sixteen run receipts, 32 weight files (rows 64 and 4,096 of every run), four start files with two files of directions, 48 files of generated bytes, seven check receipts, `E51_TABLES_A.md` and `E51_READ_A.json`. The default branch at 7cabb13a: the two half-width births and E43's receipts and weights. `claude/e50-fixed-input-code-017dSE` at 8b7d3a89: one saved network of the same seed before any update, used in section 2 only.

**Method.** Nothing was trained, torch was not used, and no code of another branch was run; every file was read with `git show` into memory and nothing was written. The script is `e51_trial_verify.py` in this folder. Its forward pass, its generation of answers, its items, streams and hashes are written from the brief's text and from a reading of the sources the brief names; the only imported code is the reviewer's reader of weight files, `compare_pt_tensors.py`. The script is called *the reader* below. It computes in double precision from the saved single-precision weights unless single precision is named. "Part 3" and the like name the part of the script that prints a figure; the command line and the running times are at the end.

## 1. The tables from the receipts (part 1)

The sixteen receipts are complete, have 4,096 updates and the ten rows of the brief, record a weight decay of 0 and a rate of 0.001 in both groups of the optimizer, and record the start, the keyed forward and the rule as each arm requires. Within the 160 rows every percentage follows from its counts, the classes of wrong answers sum to the number of wrong reduction pairs, and the never-taught score follows from its panels (no disagreement).

| What was recomputed | Result |
|---|---|
| `E51_TABLES_A.md`: the table of readings, the table of costs, the table of wrong answers, and the sixteen tables of ten rows | 2,428 cells, 0 differ |
| `E51_READ_A.json` | 228 values, 0 differ |
| The 48 files of generated bytes of the 7,832 reduction pairs (rows 0, 64, 4,096): file hash, right answers, and the six classes of wrong answers assigned again by the reader's own items and its own reading of the brief's order of classes | 0 disagreements with the receipts |
| The same files in the four `keyed` runs: the bytes of row 0 against those of rows 64 and 4,096 | equal in all four runs, both rows |
| 36 weight files (32 of the runs, four starts): SHA-256 of the file and the hash of the tensors, against the receipts; every run's recorded start against the start file | 0 disagreements |

**The worker's message**, recomputed (sweep / back-propagation):

| Claim | Reported | Recomputed from the receipts |
|---|---|---|
| `keyed`, reduction score at rows 0, 64, 4,096 | 92.4 / 95.3 at all three, both tasks | 92.4, 92.4, 92.4 / 95.3, 95.3, 95.3 on `unrelated` and on `conflict`; one set of first-family counts over all ten rows of each run |
| `keyed`, answers equal to row 0's | 100.000% of 31,684 | 31,684 of 31,684 without letters and 1,200 of 1,200 held-out pairs at 12 letters, at every row of all four runs |
| `keyed`, new task at 4,096 | unrelated 100 / 100; conflict 94.8 / 96.9 | 100.0 / 100.0; 94.8 / 96.9 (758 and 775 of 800) |
| `keyed`, learned from row | unrelated 1,024 / 512; conflict 1,024 / 512 | the same |
| `keyed-free`, first family | unrelated 0.0 at 64 and 4,096; conflict 1.0 / 0.2 at 4,096 | 0.0, 0.0 / 0.0, 0.0; 1.0 / 0.2 (16.1 / 17.4 at row 64) |
| `keyed-free`, learned from row | 512 / 512; 2,048 / 1,024 | the same |
| `open`, first family | unrelated 0.0 at 64 and 4,096; conflict 12.4 / 12.7 at 64, 1.3 / 0.0 at 4,096 | the same |
| `open`, learned from row | 512 / 512; 1,024 / 1,024 | the same |
| `rehearse`, reduction score at 4,096 | unrelated 90.0 / 92.8; conflict 88.3 / 97.8 | the same (row 0: 92.4 / 95.3) |
| `keyed`, score with the reserve shut at 4,096 | unrelated 0.0, 0.0; conflict 41.0, 40.4 | the same (0 of 300; 328 and 323 of 800) |
| `keyed`, reserve cells open by more than 0.1 | 119 / 125; 112 / 83 | the same |

**The definition of "learned from row".** The worker's reader (`e51_read.first_learned`) takes the first scored row at which the never-taught score is at 80% or more and every later row is too. This is the brief's wording. The other possible reading, the first row at 80% whatever follows, gives the same row in all sixteen runs, because no run falls below 80% after reaching it; the lowest later value is 81.1% (`open` on `conflict` under the sweep, at the row learned itself). The cost of the rule (the first row learned of `keyed` over that of `keyed-free`) is 1,024/512 = 2 and 512/512 = 1 on `unrelated` and 1,024/2,048 = 0.5 and 512/1,024 = 0.5 on `conflict`; the cost of the device (`keyed` over `open`) is 2, 1, 1 and 0.5. These are ratios of scored rows, which double from one to the next, so a ratio of 2 or 0.5 is one row.

## 2. The construction, from the weights (part 2)

**(a) The embedded start against the half-width birth.** The file and tensors hashes of the two births equal E41's receipts. The reader finds the brief's digest of the 2,080 items (`d22b5733778491b1`) and 46 bytes in use.

| Comparison with the birth, on the first family's items | Sweep | Back-propagation | The worker's check receipts |
|---|---|---|---|
| Keyed start with the two added lines: largest difference of an output at an answer position, on 1,200 held-out pairs at 12 letters, on check 1's 2,000 items, on all 31,684 items without letters and on the 2,080 taught items | 0 on all four sets in double precision, and 0 on the first two in single precision | the same | 0.0 (check 1, on the first two sets) |
| The same: generated answers that differ, of 32,884 | 0 | 0 | 0 (check 4) |
| Keyed start without the two lines: answers that differ, by panel (add, sub, max, min without letters; the same four at 12 letters) | 409, 768, 77, 94; 9, 9, 2, 4 | 200, 389, 72, 61; 21, 17, 2, 5 | the same eight numbers under each rule (check 4) |
| Start with a bias: largest difference of an output, held-out panel and 2,000 items, single precision | 0.00295, 0.00097 | 0.00340, 0.00108 | 0.00295, 0.00097; 0.00340, 0.00108 (check 1) |
| Start with a bias: generated answers that differ, of 32,884 | 1 | 0 | not recorded |
| Start with a bias: largest size of a reserve cell's state on the held-out panel | 0.00147 | 0.00240 | 0.00147, 0.00240 (row 0 of `open`) |

The counts of row 0 are those of the birth: 3,682 + 3,555 of 3,916 + 3,916 reduction pairs (92.40) under the sweep and 3,712 + 3,748 (95.25) under back-propagation, and 292, 283, 290, 290 and 283, 284, 299, 295 of 300 on the held-out panel. Every one of the 32,884 answers of the keyed start is two digits and the end symbol, so that generation presents only bytes in use. The difference from the birth is exactly zero because every old unit receives the same numbers in the same order; it is not a tolerance met.

**(b) Which entries are the birth's, which are zero and which are new.** Of the 1,124,113 parameters of the start, 317,457 are the birth's (all of the half-width network's parameters, each bit-equal to the birth's tensor), 229,632 are zero and 577,024 are new. The table is the same under both rules.

| Tensor | The birth's | Zero | New |
|---|---|---|---|
| byte table | all 258 rows | | |
| `l1` weight and bias | rows 0 to 255 | | rows 256 to 511 |
| input map (`rnn.weight_ih_l0`) | old cells' rows, columns 0 to 255 | old cells' rows, columns 256 to 511 (98,304) | reserve cells' rows, all columns |
| recurrent matrix (`rnn.weight_hh_l0`) | old cells' rows, columns 0 to 127 | | old cells' rows, columns 128 to 255 (from reserve cells into old cells); reserve cells' rows |
| the two bias vectors of the recurrent layer | old cells' rows | | reserve cells' rows |
| `lm` weight | rows 0 to 255, columns 0 to 127 | | rows 0 to 255, columns 128 to 255 (from reserve cells into old units of z2); rows 256 to 511 |
| `lm` bias, `l2` bias | 0 to 255 | | 256 to 511 |
| `l2` weight | rows 0 to 255, columns 0 to 255 | rows 0 to 255, columns 256 to 511 (65,536) | rows 256 to 511 |
| `l3` weight | columns 0 to 255 | columns 256 to 511 (65,792) | |
| `l3` bias | all | | |

*What the new entries hold.* They are the draws of a network of the present size made from the seed before any update. Three observations support this. They are identical in the starts of the two rules. Each block lies within the bound of torch's constructor for a layer of that size and has the standard deviation of a uniform draw (for example the reserve cells' rows of the input map: range −0.0625 to +0.0625, standard deviation 0.0361, where a uniform draw gives 0.0361). And they are bit-equal to the corresponding entries of a network of the same seed that another job saved before any update (`e50_C-dense_S_73129_start.pt`): all new entries of `l1`, of the input map, of the two bias vectors, of `lm` and of `l2`. For the recurrent matrix that file holds the draws on 2,816 entries a gate only, multiplied by one scale; on the 5,085 new entries that can be compared the ratio is 4.82418 throughout, which is the square root of 256/11. The byte table is the birth's: 213 of its rows equal that fresh network's, and the 45 that differ are the bytes in use other than the start symbol, whose row is zero.

The start with a bias differs from the keyed start in 256 numbers only: the bias of the input map is 10 and the bias of the recurrent matrix is 0 at the reserve cells' update-gate rows (384 to 511), which leaves a reserve gate open by about 4.5e-5.

## 3. The keeping, from the weights at rows 64 and 4,096 of the four `keyed` runs (part 3)

**The directions in use.** The saved files of directions have the hashes that the start receipts record, and their list of bytes in use is the reader's. The reader computes the directions again from its own forward pass on the keyed start and finds the same numbers and the same eigenvalues as the receipts: 46 at the input map, 129 at the recurrent matrix, 129 at `lm`, 11 at `l2` (33,280, 33,280, 6,240 and 6,240 vectors; under the sweep the largest eigenvalues are 42.5851, 13.4625, 8.8614 and 17.5122 and the last kept 4.42e-7, 6.20e-3, 4.36e-3 and 3.64e-2; under back-propagation 53.4372, 18.0738, 11.9744, 32.0238 and 2.82e-6, 2.23e-3, 2.83e-3, 1.09e-1; the first eigenvalue dropped is at most 1.1e-14). The saved directions are orthonormal to 3e-15. The projector on them differs from the projector on the reader's directions by at most 4.5e-6 in an entry at the input map, 3.9e-8 at `l2` and 3e-15 at the other two.

The reader also compares the saved directions with the span they must have if the 2,080 items exhaust what the first family can present: at the input map the codes of the 46 bytes in use; at the recurrent matrix and at `lm` the 128 coordinates of the old cells; at `l2` the codes of `=` and of the ten digits; a constant in each case. The projectors agree to the same figures (4.5e-6, 3.9e-8, 3e-15). Two things follow. At the recurrent matrix and at `lm` the rule is the same as leaving the old units' weights from old cells, and their biases, unchanged, and allowing only the weights from reserve cells to change. And the protection does not rest on the sample of 2,080 items: any item whose first byte is in use and whose bytes are all in use presents nothing outside these spans, whatever its pair and its number of letters.

**(a) What the rule does not change.** In all eight files, `l3.weight[:, :256]`, `l3.bias`, `l1.weight`, `l1.bias` and the 46 rows of the byte table in use are bit-equal to the start. So are, although the brief does not ask it, the old-cell columns and the bias of the old cells' rows of the recurrent matrix (49,536 numbers) and of rows 0 to 255 of `lm` (33,024 numbers). What changed beside the projected rows: the byte table's rows of the new task's bytes (98 rows in the unrelated runs, the four rows of `n r t x` in the conflict runs) and all 65,792 entries of `l3.weight[:, 256:]`.

**(b) The change of the constrained rows, projected on the directions in use.** The change is the difference from the start of a matrix with its bias appended, in double precision.

| Run, row | Input map, old cells' rows: 2-norm of the change | Its component along the saved directions: 2-norm; largest entry; share of the change | Largest change of an old cell's summed input on the 33,280 vectors of the 2,080 items | `l2`, rows 0 to 255: 2-norm of the change | Component: 2-norm; largest entry | Largest change of a summed input on the 6,240 vectors |
|---|---|---|---|---|---|---|
| unrelated, sweep, 64 | 6.02 | 6.5e-7; 4.2e-8; 1.1e-7 | 6.0e-7 | 5.06 | 1.7e-7; 1.6e-8 | 1.5e-7 |
| unrelated, sweep, 4,096 | 34.54 | 3.3e-5; 2.0e-6; 9.6e-7 | 2.4e-5 | 10.72 | 3.3e-5; 4.8e-6 | 3.2e-5 |
| conflict, sweep, 64 | 7.31 | 6.6e-7; 4.0e-8; 9.1e-8 | 5.6e-7 | 0.02 | 4.2e-7; 6.3e-8 | 4.2e-7 |
| conflict, sweep, 4,096 | 42.22 | 1.8e-5; 1.0e-6; 4.3e-7 | 1.4e-5 | 0.06 | 1.8e-6; 4.2e-7 | 2.8e-6 |
| unrelated, back-propagation, 64 | 4.77 | 5.9e-7; 3.7e-8; 1.2e-7 | 6.2e-7 | 3.72 | 1.2e-7; 1.0e-8 | 9.6e-8 |
| unrelated, back-propagation, 4,096 | 20.95 | 8.2e-5; 8.0e-6; 3.9e-6 | 1.3e-4 | 4.26 | 3.3e-5; 4.2e-6 | 3.7e-5 |
| conflict, back-propagation, 64 | 9.98 | 6.8e-7; 3.8e-8; 6.8e-8 | 6.0e-7 | 0.09 | 1.5e-7; 2.3e-8 | 3.6e-7 |
| conflict, back-propagation, 4,096 | 47.83 | 2.0e-5; 9.8e-7; 4.2e-7 | 9.7e-6 | 0.26 | 5.2e-6; 6.6e-7 | 7.4e-6 |

At the recurrent matrix and at `lm` the component along the directions is exactly zero in all eight files (the change of the projected rows has a 2-norm of 2.3 to 17.1 and of 1.8 to 10.3, all of it in the columns of the reserve cells). The largest projected change in any file is therefore 8.0e-6 in an entry and 8.2e-5 in 2-norm, at the input map, against a change of 20.9; the rule's requirement is met to a few millionths.

It is not met exactly, and the remainder grows: from 6e-7 at row 64 to 2e-5 to 8e-5 at row 4,096. The runner computes each projected step in double precision and then stores the weights in single precision, so each update leaves a rounding error in every stored weight, and these errors are not confined to the complement of the directions. An estimate shows that rounding alone gives this size: if each of the 384 × 46 components took an independent error of standard deviation s at every update, a 2-norm of 3.3e-5 after 4,096 updates gives s = 3.9e-9, and the rounding of a single-precision number between 0.03 and 0.5 has a standard deviation of 1.1e-9 to 8.6e-9. The estimate does not prove that rounding is the only source.

**(c) The first family through the reader's forward.** In all eight files the generated answers give the receipt's counts (3,682 + 3,555 reduction pairs and 292, 283, 290, 290 held-out answers under the sweep; 3,712 + 3,748 and 283, 284, 299, 295 under back-propagation), and no answer of the 32,884 (all 31,684 items without letters and the 1,200 held-out pairs at 12 letters) differs from the start's.

The outputs are close to the start's and not equal to them:

| Run | Largest change of an output at an answer position, row 64 (31,684 items; held-out panel) | Row 4,096 | Row 4,096, in single precision (held-out panel; check 1's 2,000 items) | Check 3's receipt, after 8 updates, single precision |
|---|---|---|---|---|
| unrelated, sweep | 2.2e-6; 1.7e-6 | 3.8e-4; 3.9e-4 | 3.9e-4; 3.6e-4 | 1.5e-5; 1.7e-5 |
| conflict, sweep | 3.0e-6; 2.6e-6 | 5.5e-5; 5.4e-5 | 5.8e-5; 5.4e-5 | 1.5e-5; 1.9e-5 |
| unrelated, back-propagation | 1.8e-6; 1.4e-6 | 5.3e-4; 4.5e-4 | 4.4e-4; 3.8e-4 | 1.6e-5; 1.7e-5 |
| conflict, back-propagation | 2.5e-6; 2.1e-6 | 7.3e-5; 5.6e-5 | 6.5e-5; 5.7e-5 | 2.0e-5; 2.1e-5 |

In single precision the reader finds 1.4e-5 to 1.9e-5 at row 64, where the change in double precision is 2e-6 to 3e-6. Check 3's figure after eight updates is therefore the arithmetic of single precision and not a movement of the weights; the movement becomes visible above it by row 4,096, where in the two unrelated runs it is four to five times the 1e-4 that check 3(a) allowed after eight updates. The brief set no limit for row 4,096.

How near this came to changing an answer is read from the leads. Under the sweep the smallest lead at the start is 4.06e-4 over the 31,684 items and 1.78e-4 over the held-out panel (32 and 2 leads are below 0.01); under back-propagation 7.75e-4 and 2.33e-2 (10 and none below 0.01). At row 4,096 of the unrelated runs the largest change of any lead is 4.3e-4 (sweep) and 8.5e-4 (back-propagation), which is more than the smallest lead; the leads below 0.01 happened to move less (at most 1.0e-4 and 2.0e-4), and the smallest lead is now 3.83e-4 and 1.62e-4 under the sweep and 5.90e-4 under back-propagation. In the conflict runs the largest change of a lead is 8.1e-5 and 1.0e-4. At no position of any file has another output become the largest.

*Is the remainder along the directions the only path to the first family?* The reader takes the weights of row 4,096, removes in double precision the component of the change of the input map's and of `l2`'s projected rows along its own directions, alters nothing else, and compares again. The largest change of an output from the start's is then 3.6e-13 to 1.1e-12 on the 31,684 items and 1.0e-11 to 2.2e-11 on the held-out panel, in all four runs. Everything else that changed (the reserve cells' rows, the weights from reserve cells into old units, the added rows of `lm` and `l2`, the added columns of `l3`, the byte table's other rows, and the whole of the change of the old rows outside the directions) does not reach a first-family item at all.

**(d) What is guaranteed by the construction and what is found.**

*Guaranteed*, for every item whose first byte is a byte in use and whose other bytes are bytes in use, taught or not, with any number of letters, provided the program does what the brief writes:

- The reserve cells' states are exactly zero at every position and the added units of z2 are exactly zero. This is done by the two added lines of the forward pass, which read the item's first byte; it does not depend on any weight and is not learned.
- The codes of the bytes in use are those of the start, because `l1` and the 46 rows of the byte table are restored after every step.
- Every old cell and every old unit of z2 receives what it received at the start: the change of its weights from the codes is confined to the complement of the codes of the bytes in use, the change of its weights from cells is confined to the reserve cells, whose state is zero, and the old columns and the bias of `l3` are restored; the added columns of `l3` multiply zeros.

The outputs on those items are then the start's, up to the rounding described above. That held-out pairs are kept as well as taught ones is part of this arithmetic and is not a finding about generalisation. Prediction P51.3 says so itself ("by construction: a failure is an error of implementation").

*Not guaranteed, and found here:* that the implementation is right (the bit-equal tensors, the projected change of a few millionths, and the removal test above show it for rows 64 and 4,096); the size of the rounding remainder, and that it changed no answer in 4,096 updates; and everything about the new task, which is the subject of sections 4 and 8. *Not covered at all:* an item whose first byte is not one of the 46, an item that contains a byte not in use, and a first-family answer in which a byte not in use is generated; none occurs in the scored items, in which every answer of the start is two digits and the end symbol.

## 4. The new task, from the weights (part 4)

The items of both tasks begin with a byte that is not in use (224 for `unrelated`; `r` and `t` on the conflict panels), so the two added lines never act on them. The reader's forward gives, for the keyed start and for row 4,096 of each `keyed` run, the receipt's counts of whole answers and of last three bytes on every panel, with the reserve cells open and with them held shut on every item, and the receipt's figures for the reserve's gates.

| Run, row | Reserve open (whole answers right) | Reserve cells shut on every item | Equal to the receipt | Reserve cells open by more than 0.1; mean of 1 − z; largest (reader and receipt agree) |
|---|---|---|---|---|
| unrelated, sweep, 0 | held 0 of 300, taught 0 of 300 | 0, 0 | yes, yes | 128; 0.4986; 0.5948 |
| unrelated, sweep, 4,096 | held 300, taught 300 | 0, 0 | yes, yes | 119; 0.5941; 0.9616 |
| unrelated, back-propagation, 0 | 0, 0 | 0, 0 | yes, yes | 128; 0.4975; 0.5876 |
| unrelated, back-propagation, 4,096 | 300, 300 | 0, 0 | yes, yes | 125; 0.6667; 0.9960 |
| conflict, sweep, 0 | add97 48 of 400, sub97 96 of 400 (18.0%) | 57, 95 (19.0%) | yes, yes | 128; 0.5057; 0.6503 |
| conflict, sweep, 4,096 | 374, 384 (94.8%) | 168, 160 (41.0%) | yes, yes | 112; 0.3476; 0.8798 |
| conflict, back-propagation, 0 | 15, 53 (8.5%) | 9, 57 (8.2%) | yes, yes | 128; 0.5063; 0.6875 |
| conflict, back-propagation, 4,096 | 381, 394 (96.9%) | 161, 162 (40.4%) | yes, yes | 83; 0.2198; 0.8227 |

The second scoring shuts the reserve cells only. The added units of z2, which the brief's terms also count as units in reserve, stay active in it; section 8 separates the two.

## 5. `keyed-free`, `open`, `rehearse` (part 5)

**From the receipts.** The first family's reduction score (%) at the ten rows; in brackets the share of the 31,684 answers equal to row 0's at rows 64 and 4,096.

| Task, arm, rule | 0 | 16 | 32 | 64 | 128 | 256 | 512 | 1,024 | 2,048 | 4,096 |
|---|---|---|---|---|---|---|---|---|---|---|
| unrelated, `keyed-free`, sweep | 92.4 | 0.0 | 0.0 | 0.0 (0.0) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 (0.0) |
| unrelated, `keyed-free`, back-propagation | 95.3 | 2.0 | 0.0 | 0.0 (0.1) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 (0.0) |
| unrelated, `open`, sweep | 92.4 | 0.0 | 0.0 | 0.0 (0.0) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 (0.0) |
| unrelated, `open`, back-propagation | 95.3 | 0.0 | 0.0 | 0.0 (0.0) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 (0.0) |
| unrelated, `rehearse`, sweep | 92.4 | 11.5 | 20.3 | 35.3 (59.0) | 47.8 | 20.5 | 32.2 | 58.9 | 85.5 | 90.0 (90.4) |
| unrelated, `rehearse`, back-propagation | 95.3 | 38.4 | 61.8 | 72.8 (81.5) | 81.4 | 76.0 | 80.4 | 89.8 | 92.8 | 92.8 (93.3) |
| conflict, `keyed-free`, sweep | 92.4 | 45.1 | 36.4 | 16.1 (56.6) | 16.9 | 9.9 | 4.2 | 3.4 | 2.6 | 1.0 (56.0) |
| conflict, `keyed-free`, back-propagation | 95.3 | 26.2 | 20.3 | 17.4 (59.0) | 10.0 | 1.7 | 1.6 | 0.4 | 0.1 | 0.2 (57.5) |
| conflict, `open`, sweep | 92.4 | 25.7 | 19.1 | 12.4 (50.3) | 14.1 | 5.4 | 4.9 | 3.0 | 0.8 | 1.3 (58.1) |
| conflict, `open`, back-propagation | 95.3 | 25.8 | 19.5 | 12.7 (56.0) | 4.5 | 1.2 | 0.5 | 0.0 | 0.0 | 0.0 (60.4) |
| conflict, `rehearse`, sweep | 92.4 | 59.6 | 44.6 | 54.8 (78.0) | 49.4 | 61.7 | 72.8 | 70.3 | 87.4 | 88.3 (90.2) |
| conflict, `rehearse`, back-propagation | 95.3 | 46.6 | 51.6 | 65.5 (81.5) | 70.9 | 72.2 | 78.2 | 89.1 | 94.5 | 97.8 (95.9) |

Three things are read from the table and are in the worker's tables as well. With rehearsal the first family is not held along the way: under the sweep it falls to 11.5 and 59.6 at row 16 and returns to within 2.5 and 4.1 points of its start by row 4,096, and even then 9.6% and 9.8% of the answers are not those of row 0. On the conflicting family the arms without rehearsal lose nearly all reduction pairs while more than half of all answers remain those of row 0; in `open`, max and min, whose answers do not depend on the modulus, are still at 91.4 and 81.1% (sweep) and 97.5 and 80.4% (back-propagation) at row 4,096. And `keyed-free` loses the first family about as `open` does, so holding the reserve shut on the first family's items does not keep it by itself.

**From the weights at row 4,096 of all twelve runs.** The reader's forward (with the two added lines in `keyed-free`, without them in `open` and `rehearse`) gives the receipt's counts in every run: the first family's eight counts (reduction pairs and other pairs of add and sub, max, min, and the four held-out panels), the new task's panels, and the new task's panels with the reserve cells shut. For example `rehearse` on `unrelated` under the sweep: 3,697 + 3,348 reduction pairs (89.95), 281, 292, 283, 296 on the held-out panel, 292 of 300 on the new task; `open` on `conflict` under the sweep: 10 + 93 reduction pairs (1.32), 341 and 339 of 400 on the new task and 331 and 333 with the reserve cells shut.

**`open` on the unrelated task and E43.** The brief's check 6 does not compare the embedded network with E43's network. It starts the runner of `open` from E26's birth of the present size as it is, with no embedding and no change of a bias, for 64 updates, and requires E43's receipt at row 64. The two dry receipts under `check6/` record that start, and at row 64 they have E43's first-family counts, E43's new-task counts, E43's data digest and E43's hash of the tensors, under both rules. The reader computed the same hash again from E43's saved weights of row 64 and finds it equal. This shows that the runner's update and scoring, without the two added lines and without the rule, are E43's.

The trial's own `open` runs are another network, the embedded half-width birth with 128 cells in reserve, and are not expected to equal E43's. Side by side (reduction score; never-taught score of the unrelated task):

| Row | E43, sweep (E26's birth, 256 cells born) | E51 `open`, sweep | E43, back-propagation | E51 `open`, back-propagation |
|---|---|---|---|---|
| 0 | 96.57; 0.0 | 92.40; 0.0 | 98.14; 0.0 | 95.25; 0.0 |
| 64 | 0.10; 0.0 | 0.00; 0.0 | 0.05; 0.0 | 0.00; 0.0 |
| 256 | 0.00; 6.0 | 0.00; 15.3 | 0.00; 98.0 | 0.00; 28.3 |
| 1,024 | 0.00; 100.0 | 0.00; 100.0 | 0.00; 100.0 | 0.00; 100.0 |
| 4,096 | 0.00; 100.0 | 0.00; 100.0 | 0.00; 100.0 | 0.00; 100.0 |

A reserve whose gates start shut by a bias and are free does not protect the first family: it is lost by row 16 (E43's first scored row is 64, where it is lost as well), before the unrelated task has begun to be learned.

## 6. The checks before the runs (part 6)

The runner `exp51_reserve.py` at the worker's head has the SHA-256 `84d0dad7f6e5eeb1…`, and that hash is the one recorded in all seven check receipts, all four start receipts and all sixteen run receipts. Checks 1 to 6 and the starts were made before the runner was committed (the receipts record the file as not yet tracked), with the file that was then committed unchanged.

| Check | What the receipt records | Recomputed here |
|---|---|---|
| 1. The embedding | Largest difference of an output from the birth's on the held-out panel and on the 2,000 items: 0 and 0 for the keyed start under both rules (limit 1e-5); 0.002949 and 0.000974 (sweep), 0.003405 and 0.001085 (back-propagation) for the start with a bias (limit 0.02); row 0's scores equal to the birth's on every panel in the keyed start | 0 and 0; 0.002947 and 0.000971; 0.003399 and 0.001083 in the reader's single precision (the last digits differ because the arithmetic is another library's); the birth's scores and answers in section 2 |
| 2. The directions | The digest of the 2,080 items, 46 bytes in use, 46, 129, 129, 11 directions, with the spectra | The same digest, 46, and 46, 129, 129, 11 (section 3) |
| 3. The rule | (a) After 8 updates of `keyed`: largest change of an output 1.5e-5 to 2.1e-5 on both streams and both rules (limit 1e-4); the five tensors or rows that are not changed bit-equal; (a) with the byte table not restored, on the conflict stream: 41 of the 46 rows changed and the outputs moved by 0.22 to 0.31. (b) After 8 updates of `keyed-free`: outputs moved by 20 to 39. (c) Largest reserve state and largest added unit of z2: 0 and 0 on a batch of the first family; 0.7587 and 0.8764 (unrelated), 0.7652 and 0.9272 (conflict) under the sweep; 0.7274 and 0.9063, 0.9134 and 0.9348 under back-propagation | (a) and (b) cannot be recomputed, because no weights after 8 updates are saved; the corresponding figures at rows 64 and 4,096 are in section 3. (c) recomputed on the first batch of the reader's own three streams, filled as the update fills shorter rows: 0 and 0; 0.7586981 and 0.8763809; 0.7651762 and 0.9271622; 0.7273827 and 0.9062994; 0.9134029 and 0.9347987, equal to the receipt's to six decimal places |
| 4. The scorer | 0 of 32,884 generated answers of row 0 differ from the birth's; without the two added lines 409, 768, 77, 94, 9, 9, 2, 4 (sweep) and 200, 389, 72, 61, 21, 17, 2, 5 (back-propagation) differ | The same 0 and the same sixteen numbers (section 2) |
| 5. The data | The batch digests at update 4,096 of the unrelated stream, the conflict stream and the rehearsed unrelated stream equal E43's receipts and the second addendum's | The reader builds the four streams itself (the fourth is the rehearsed conflict stream, for which no earlier receipt exists) and its own digest. Digests at 4,096: `f60110540a0c32e7`, `60870058e0435e12`, `11fcebdb0e7a36fd`, `e4d6455f4bfbfb2b`. They equal the digests of rows 16 to 4,096 of all sixteen run receipts, with the receipts' counts of items (131,138 unrelated and 131,006 first-family items in `rehearse` on `unrelated`; 130,934 and 131,210 on `conflict`), and E43's and the second addendum's receipts |
| 6. `open` is E43 in another network | First-family counts and the hash of the tensors at row 64 equal E43's, under both rules | Equal; the hash also equals the one the reader computes from E43's saved weights (section 5) |
| 7. Cost | Sixteen dry runs of 64 updates: 0.086 to 0.121 CPU-seconds an update, 10.9 to 23.7 a scored row; forecast 2.76 CPU-hours | The sixteen receipts sum to 2.74 CPU-hours, 0.088 to 0.123 CPU-seconds an update |

The seven receipts record what the brief asks. Two remarks. Check 3(a)'s clause on the byte table was tested as revision 2 requires, on the conflict stream with the restoring switched off. Check 3(a)'s tolerance is met after eight updates, but, as section 3 shows, its figure is the arithmetic of single precision; the quantity it was meant to bound is 2e-6 at row 64 and 5e-5 to 5e-4 at row 4,096.

## 7. The predictions (part 7)

| | Statement of the brief | Figures from the receipts (sweep; back-propagation) | Outcome |
|---|---|---|---|
| P51.1 | `open` loses the first family on `unrelated`: below 5% on reduction pairs by update 64, both rules | 0.00, 0.00, 0.00 at rows 16, 32, 64; 0.01, 0.00, 0.00 | held |
| P51.2 | `open` loses the reduction pairs on `conflict`: below half of row 0 at update 4,096, both rules | 1.32 against 46.20; 0.00 against 47.63 | held |
| P51.3 | `keyed` keeps the first family on both tasks, both rules | change of the reduction score from row 0 to row 4,096: 0.000 in all four runs | held |
| P51.4 | `keyed` learns `unrelated` | learned from row 1,024; from row 512 | held under both rules |
| P51.5 | `keyed` learns `conflict` | learned from row 1,024; from row 512 | held under both rules |
| P51.6 | Where `keyed` learns `unrelated`, the cost of the rule is at most four | 1,024/512 = 2; 512/512 = 1 | held |
| P51.7 | `rehearse` keeps the first family within 5 points on both tasks, both rules | unrelated −2.45; −2.44; conflict −4.07; +2.54 | held; none of the four is within the 2 points of the brief's *kept* |
| P51.8 | `keyed-free` does not keep the first family on `unrelated` under either rule | 0.00 against 92.40; 0.00 against 95.25 | held |
| P51.9 | Where `keyed` learns `unrelated`, it is *learned in the reserve* | 0.0; 0.0 with the reserve cells shut | held, in the brief's reading; see section 8 |

All nine held, as the worker reports. P51.3 was a test of the implementation by its own wording. P51.9 holds by the criterion fixed beforehand; section 8 shows that the criterion establishes less than its name says.
