# Check of the brief of E57, revision 2 (18 November 2026)

Written for the outside reviewer by an agent that has not seen the reviewer's reasoning. The brief is `TASK_E57_THE_SWEEP_WITH_A_SCALE_FOR_EACH_RECEIVING_UNIT_20261117.md` beside this file, revision 2, committed at 7dd813aa. Its SHA-256 was `a5b3a351db8aee10eef5afc35f4b24000a55e31cf2e6b8a732c01d0be6f291a0` when this check began; the last section says whether it had changed when the check ended. This is a second and bounded check. It covers four things: the additions of revision 2 that the first check did not see; whether the replacement texts of the first check were entered faithfully; the consistency and the language of revision 2 as a whole; and the cost and the wall-clock time. The findings of the first check were not derived again. The work was read-only: nothing was trained, torch is not installed, no code of another branch was executed, the files of other branches were read with `git show`, and this check committed nothing. It is not the independent audit that the programme's rules ask for.

**Shell names.** `D`, `W`, `W9`, `V`, `M` and `R` as in the brief. `E50B=origin/claude/e50-fixed-input-code-017dSE`, read at c6c24d9c. "Brief, line n" cites a line of revision 2. *The first check* is `CHECK_OF_TASK_E57_BRIEF_20261117.md` beside this file (SHA-256 `cb26cd41…`), and "first check, line n" cites a line of it. `APP <part>` is the first script of the appendix, run from the repository root as `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -I -B <the script> <part>`.

**Terms.** As in the brief. In addition: *the added run* is the run of the network with a limited input under the sweep with the scale `matrix` in which the code is not decayed; *the run beside it* is the run of the same network with the scale `matrix` and a decayed code; *the standing optimizer* is AdamW with rate 0.001 and a weight decay of 0.25 on every tensor but the byte table; *the padding symbol* is the value 257, which `exp45.batch_tensor` writes at the first position of every item and after the end of every item that is shorter than the longest of its batch.

## Verdict

Release after the listed changes.

The mechanism of the added run is sound. The function `no_code_decay` moves the two tensors of `l1` between the two parameter groups in the way the brief says, at the place the brief names, and no later line of the copied run fails or records something false after the move. The definition of weight decay is right for the optimizer in use. The figures of E50's stage 2 are those of the two receipts. The point on the size of the code in E49 is faithful to its source. No pair of predictions is inconsistent. The replacement texts of the first check are entered word for word in 119 of their 159 lines, and every difference in the other 40 is an adaptation to the added run or to the completed runs of E50; the numbers of runs and of checks are carried through every section. The cost and the wall-clock time follow from the brief's unit costs.

Six defects remain. None changes the roster or the runner's code; five are in what the brief says about the record and about the reading of the added run, and one is in check 7.

1. The reading of the added run is fixed for one outcome of the run beside it only; in the other outcomes two of its three sentences would be false, and one sentence states a share that two runs cannot show.
2. E50's network is described wrongly (its recurrent map is not dense), the brief says that the sweep "learns the taught pairs" where the level is 74.8 and the score never reaches 80, and the ground of the added run assigns to the size of the code what E50's own report declines to assign.
3. Check 7 cannot fail in one case that check 1 allows, names a figure that the runner does not record, and supports "on nothing else" by the list of the groups alone.
4. Two sentences on the weight decay in the record are not true (E46 ran births without weight decay), and E46's result, which bears on the added run more than any other on file, is not cited: without any weight decay `l1` still falls to between a half and two thirds of its starting size.
5. `code_mean_abs` can be computed in two ways, and about 28% of the positions over which it is taken hold the padding symbol, whose code does not depend on any byte.
6. Small points of count and of wording (ten).

Each defect has an exact replacement text.

## Defects, the most serious first

### 1. The reading of the added run is fixed for one outcome of the run beside it only

**The place.** Brief, line 217:

> If it learns the family while the run with a decayed code does not, the report says that, on this seed, the sweep with the scale `matrix` learns the family in the network with a limited input when the code is not decayed, so that the decay of `l1` was necessary for the failure. If it has the word *partly*, the report gives its levels and says that the decay of `l1` accounts for part of the failure. If it has the word *not learned*, the report says that the failure does not need the decay of `l1`.

**What is wrong.**

1. *The second and the third sentence have no condition on the run beside it.* Only the first sentence is conditioned ("while the run with a decayed code does not"). The brief itself gives a chance of 15% that the run beside it does not have the word *not learned* over its 49,152 updates (P57.10, 85%), and the first row of the table of outcomes (brief, line 209) provides for that case. The sentences then read as follows.

   | Word of the run beside it (decayed code) | Word of the added run | What brief line 217 has the report say | Is it supported? |
   |---|---|---|---|
   | *not learned* | *learns the family* | the decay of `l1` was necessary for the failure | yes, for this run on this seed (point 4) |
   | *not learned* | *partly* | the decay accounts for part of the failure | in substance; the word "part" is not (point 3) |
   | *not learned* | *not learned* | the failure does not need the decay | yes |
   | *partly* | *partly* | the decay accounts for part of the failure | no: the two runs have the same word |
   | *partly* or *learns the family* | *not learned* | the failure does not need the decay | no: the run with the decay did not fail, and the run without it did |
   | *learns the family* | *partly* | the decay accounts for part of the failure | no: there was no failure |
   | *learns the family* | *learns the family* | nothing | no sentence is fixed |

2. *Which of the two words is meant is not said.* A run of 49,152 updates has two words, one for the whole run and one for its first 24,576 updates (brief, line 198). The table of outcomes says which it uses (brief, line 205: "the word of each run over its 49,152 updates"). Line 217 does not.

3. *"Accounts for part of the failure" states a share.* The word *partly* means that the level of T at the end is 50 or more while A or R has not reached 80. Two runs show that, without the decay of `l1`, the network reaches those levels and does not learn the family. They do not show what part of the failure the decay accounts for, and the remaining failure may have another form than the one in E49 (for example a code that grows until the tanh saturates).

4. *"Necessary for the failure" is supported, within a limit that the sentence should carry.* The two runs differ in one setting and are deterministic, so the added run is the counterfactual of the run beside it: if it learns the family where that run has the word *not learned*, then that run would not have failed without the decay of `l1`. This holds for one run of one network on one seed. It does not extend to E49's two other networks with a limited input or to another seed, and the brief's "on this seed" should stand in every one of the three sentences and not in the first only.

5. *A code that is not decayed can still shrink.* The brief's ground speaks of "a code that cannot shrink" (line 77). The added run removes the weight decay of `l1`; it does not hold the code at its size. In E46 the dense network was born with a weight decay of 0 on every tensor, and the norm of `l1` still fell from 13.84 to 7.31, 8.45 and 7.87 under the sweep on the three seeds, that is to 0.53 to 0.61 of its starting size, and to 0.61 to 0.69 under back-propagation (`$V:$R/E41_E46_VERIFICATION_20261117.md`, section 9.2, the table "The size of the weights at the end"). The sentence for the word *not learned* is right as it stands, but a reader will take it to say that the failure does not need a shrinking code, and only the two recorded sizes can say whether the code of the added run shrank. The brief has the report set `code_mean_abs` of the two runs side by side; it should say the same of `l1_weight_mean_abs` and say what the comparison is for.

**Replacement.** Replace brief line 217 by:

> **The run in which the code is not decayed** is read beside the run with `matrix` and a decayed code, from which it differs in the decay of `l1` only. Both are read by their word over 49,152 updates, and the reading does not change the word given to the explanation. The three sentences that follow apply only if the run with a decayed code has the word *not learned*. If the run in which the code is not decayed learns the family, the report says that, on this seed, the sweep with the scale `matrix` learns the family in the network with a limited input when `l1` is not decayed, so that the decay of `l1` was necessary for the failure of the run with a decayed code. If it has the word *partly*, the report gives its levels and says that, on this seed, the network without the decay of `l1` reaches these levels and does not learn the family. If it has the word *not learned*, the report says that, on this seed, the network fails without the decay of `l1` as well. If the run with a decayed code has another word, the report gives the levels and the updates to 80 of the two runs and no sentence on the decay. In every case the report sets `l1_weight_mean_abs` and `code_mean_abs` of the two runs side by side at every reading and says whether the code of the run without the decay kept its size: a code that is not decayed can still shrink through the steps of the gradient, as `l1` did in E46. No sentence of the report extends the reading to E49's two other networks with a limited input or to another seed. Two changes can each be sufficient; if this run and a run under another scale both learn the family, the report says so and compares their updates to 80.

### 2. E50's network is described wrongly, and the ground of the added run says more than E50's report

**The places.** Brief, line 61:

> The network has 16 inputs a cell, a dense recurrent map, and a fixed code: the byte table and `l1` are not trained, so that no error has to reach the code.

> Two statements follow for this seed. A limited input alone does not stop the sweep with the scale `matrix`: with a code that is not trained it learns the taught pairs and part of the untaught ones, where E49's network stayed below 5. And the sweep remains far behind back-propagation in a network in which the error that reaches the code plays no part; that distance lies in what the cells and the readout receive.

Brief, line 77: "and in E50 a code that cannot shrink lets the sweep learn the taught pairs". Brief, line 286: "E50's stage 2, which ended after the check, shows that a code that cannot shrink lets the sweep learn the taught pairs."

**What is wrong.**

1. *The recurrent map of `C16-in` is not dense.* `C16-in` is `C-dense` with 16 values a cell (`$E50B:$M/exp45/e50/E50_STAGE2_REPORT_73129.md`, line 5), and `C-dense` is "E42's `F10` network (256 cells, each reading 10 fixed cells of its neighbourhood, every cell reading all 512 values and every readout unit all 256 cells)" (`E50_STAGE1_REPORT_73129.md`, "Terms"). The runner builds every setting with `layer = L42.Layer(m, seed, N=10)` (`$E50B:$M/exp50_code.py`, line 112), and both receipts record `N` 10, `k_in` 16 and `k_out` 256 (`APP e50`). What is dense in `C16-in` is the readout. The network is therefore E49's network with a limited input, with the same lists of 16 values a cell, the same limited connections and the same dense readout, and it differs from that network in the code alone. This makes the comparison with E49's run closer than the brief's sentence allows: a reader of "a dense recurrent map" would take the two networks to differ in the recurrent map as well.

2. *The code is another code, and not E49's code held at its start.* "The byte table and `l1` are not trained" is true and is not the whole difference. In E50 the byte table and `l1` take no part: the code of a byte is read from a fixed table in which each of the 46 symbols in use turns on 11 of the 512 values of its own, each equal to 3.2387, and every other value is zero (`e50_sparse_code.py`, lines 28 to 35 and 49; stage 1 report, "Terms"; the receipts' entry `code`). A value of E49's code lies between −1 and 1 and has a mean magnitude of 0.41 at the start. With 16 values a cell, a cell of `C16-in` has a non-zero input for 1 to 10 of the 20 bytes that the task must distinguish (median 6), where in E49's network every cell reads values that differ between any two bytes (stage 2 report, line 5).

3. *"Learns the taught pairs" says more than the receipt.* The level of T of the sweep's run at update 49,152 is 74.8; the score reaches 50 at update 22,016 and never reaches 80, and its highest single scored row is 79.67 at update 47,616 (`APP e50`). By the brief's own words the run has the word *partly*. The brief defines no sense of "learns" for one score other than "reaches 80".

4. *E50's report declines the inference that the brief draws from it.* The explanation that E50's brief put under test is "that the sweep did not learn with a limited input in E49 because it does not keep a learned code at the size that a cell reading few values requires", and the stage 2 report states: "Here the code cannot shrink, and `C16-in` under the sweep does not reach 80 on R within its dose while `C-dense` under the sweep does. **By the brief's criterion the explanation is refuted as a sufficient one.** It is also the case that with the fixed code the sweep's network gets much further than E49's did (R 43.6 against 1.4; max and min learned); the job does not say what part of that is the code's size, since the fixed code differs from the learned one in other respects" (line 31). Brief lines 77 and 286 assign the gain to the property "cannot shrink". The source says that the gain cannot be assigned, and that a code which cannot shrink was not sufficient for the family. Both statements bear on the added run: the first on why the run is worth making, the second on what it can be expected to show.

5. *The two statements, after the correction.* The first ("a limited input alone does not stop the sweep") is supported, and better than the brief's description suggests, because the two networks have the same masks. The second (the sweep reaches much lower levels than back-propagation where no error that reaches the code is used) is supported as it stands: the forward reads the code from the fixed table, the byte table and `l1` receive no gradient, the receipts record `byte_table_and_l1_bit_equal_to_start` true and an optimizer state for the ten other tensors only, and the gradient of `l3` is exact under both rules, so that the two rules differ only in the gradients of `l2`, `lm`, the input map and the recurrent map (`e50_sparse_code.py`, lines 46 to 66; `APP e50`).

**Replacement.** Replace the point "E50, stage 2" (brief, line 61) by:

> - **E50, stage 2** (`origin/claude/e50-fixed-input-code-017dSE` at c6c24d9c; the worker's report of 23:52 UTC on 17 November, `$M/exp45/e50/E50_STAGE2_REPORT_73129.md`. The reviewer recomputed the levels below from the two receipts; the verification from the weights has not been made). E50's network `C16-in` is E49's network with a limited input (the same 16 values a cell, the same limited connections, every readout unit reading every cell) with a fixed code in place of the learned one. The byte table and `l1` take no part: each of the 46 symbols in use turns on 11 of the 512 values of its own, each equal to 3.2387, and every other value is zero, so that no error has to reach the code and the code cannot change. With the scale `matrix` the sweep's levels are A 32.7, R 29.0 and T 54.6 at update 24,576 and A 49.5, R 43.6 and T 74.8 at update 49,152, where R still rises by 1.3 points in 2,048 updates; T reaches 50 at update 22,016 and does not reach 80, so that the run has the word *partly*. Back-propagation has A 67.9, R 62.1 and T 85.3 at update 24,576 and A 83.8, R 79.4 and T 96.3 at update 49,152; A reaches 80 at update 40,704 and R does not reach it. Three statements follow for this seed. First, a limited input alone does not prevent the sweep with the scale `matrix` from learning part of the family: with the fixed code it reaches 74.8 on T, 49.5 on A and 43.6 on R, where E49's network stayed below 5. Second, E50's report does not assign this gain to the size of the code, because the fixed code differs from the learned one in its form as well (a cell of `C16-in` has an input for 1 to 10 of the 20 bytes that the task must distinguish), and by the criterion of E50's brief a code that cannot shrink is refuted as a sufficient condition: the sweep does not reach 80 on R with it. Third, the sweep reaches much lower levels than back-propagation in a network in which the error that reaches the code is not used; in that network the two rules differ only in the error that the cells and the readout receive.

In brief line 77 replace "It rests on two results: in E49 the code shrinks under the sweep while the kept weights of the input map do not grow to make up for it, and in E50 a code that cannot shrink lets the sweep learn the taught pairs." by:

> It rests on two results: in E49 the code shrinks under the sweep while the kept weights of the input map do not grow to make up for it, and in E50 the same network with a fixed code, which cannot shrink, reaches 74.8 on T under the sweep and does not learn the family. E50's code differs from the learned one in its form as well as in its size, so that E50 does not say what a learned code of constant size would do; the added run comes nearer to that question, although a code that is not decayed can still shrink (E46).

In brief line 286 replace "E50's stage 2, which ended after the check, shows that a code that cannot shrink lets the sweep learn the taught pairs." by:

> E50's stage 2, which ended after the check, shows that the same network with a fixed code, which cannot shrink, reaches 74.8 on T under the sweep, where E49's run stayed below 5.

### 3. Check 7 cannot fail in one case that check 1 allows, names a figure that the runner does not record, and rests "on nothing else" on the list of the groups alone

**The place.** Brief, line 192:

> Its scored row of update 0 and its `start_tensors_sha256` are those of the dry run of check 1 for `in16`, and its `tensors_sha256` is not `88cad76a…`. The check records `l1_weight_mean_abs` at update 0 and after the 256 updates beside the same two figures of the dry run of check 1.

**What is wrong.**

1. *The third condition is compared with a constant, and the constant is right only if check 1 passed bit for bit.* Rule (c) of check 1 lets the job go on when the new crossing equals the old one to rounding and not bit for bit (brief, line 186). The dry run of check 1 for `in16` then has a `tensors_sha256` other than `88cad76a…`, and so has every later run of 256 updates, whether the option acts or not. In that case the condition "is not `88cad76a…`" is met by a run in which the option does nothing. The first condition (the list of the groups) still holds the option to the lists of the optimizer; it does not show that the optimizer used the changed lists. The condition must compare with the hash of the dry run of check 1.

2. *Check 1 has two dry runs for `in16`,* of 256 and of 2,048 updates (brief, line 185). They agree in the start and in the scored row of update 0, but the comparison of the final hash and of the size after 256 updates needs the one of 256 updates.

3. *`l1_weight_mean_abs` after 256 updates is not recorded by any run.* The list `cosines` receives an entry before the first update and after every 2,048 updates (brief, lines 172 and 176). A dry run of 256 updates has the entry of update 0 and no other. The check can compute the figure from the two weight files of each dry run (`_start.pt` and `.pt`), and the brief should say so; as worded, a worker must either change the runner for dry runs or guess the source.

4. *"On nothing else" rests on the list of names.* The entry `optimizer` is read back from the optimizer and lists the tensors of each group with the group's weight decay, rate and betas (`exp42_cells.py`, lines 171 to 179). The check compares the names and the two weight decays. It does not compare the rates and the betas with those of the reference, although they are in the same entry, and it does not show that the eleven other tensors are updated as before. A direct test costs seconds: after one update from the same start on the same batch, the gradients of the two networks are equal, so that every tensor outside `l1` must be equal bit for bit, and for `l1` the value without the decay minus the value with it must equal 0.00025 times the starting value (the optimizer multiplies the weight by 1 − 0.001 × 0.25 and then subtracts a step that does not depend on the weight). This also tests, on the worker's version of torch, the one thing that this check could only read and not run: that a tensor moved between the lists before the first step is treated as a member of its new group (see "Not checked", point 1).

5. *The size after 256 updates is recorded and not judged, and that is right.* Over 256 updates the decay alone multiplies a weight by 0.938 (`APP decay`), and the steps of the gradient can move a weight of `l1` by more than that, so that no direction can be required of the two sizes. The brief asks for the figures only.

**Replacement.** Replace check 7 (brief, line 192) by:

> 7. **The option `--code-decay 0` acts on `l1` and on nothing else.** *The reference* of this check is the dry run of 256 updates of `in16` of check 1 (b).
>    (a) *After one update.* Two networks are built by `R49.build("in16", 73129)`, each with `opt = R42.make_opt(m, layer)`; `no_code_decay` is applied to the second; each makes one update with the runner's `step` under the sweep with the scale `matrix` on the first batch of the stream. Eleven of the thirteen tensors (all but `l1.weight` and `l1.bias`) are equal in the two networks under `torch.equal`. `l1.weight` is not equal, and for `l1.weight` and for `l1.bias` the value in the second network minus the value in the first equals 0.00025 times the starting value, with no entry differing by more than 1e-6.
>    (b) *A dry run of 256 updates* of `in16` under the sweep with the scale `matrix` and `--code-decay 0`. The entry `optimizer` of its receipt lists exactly `book.weight`, `l1.weight` and `l1.bias` in the group with weight decay 0 and the ten other tensors of the network in the group with weight decay 0.25; the rate and the betas of both groups are those of the reference; the entry `optimizer_read_back` equals the entry `optimizer`; the entry `code_decay` is 0. Its `start_tensors_sha256`, the counts of its scored row of update 0 and the batch digest of its scored row of update 256 are those of the reference. Its `tensors_sha256` differs from that of the reference (which is `88cad76a…` if check 1 passed with equal hashes).
>    (c) The check records, without a condition, the mean of the magnitudes of the entries of `l1.weight` in the files `_start.pt` and `.pt` of this dry run and of the reference.

### 4. Two sentences on the weight decay in the record are not true, and the result on file that bears most on the added run is not cited

**The places.** Brief, line 15 (the term *Weight decay*):

> In every run of E26 to E50 it acts on every tensor but the byte table.

Brief, line 23:

> In every run the sweep has used one scale for a whole matrix, and every run has decayed the map that makes the code; no other scale and no other decay of that map was tried in the network with a limited input.

**What is wrong.**

1. *E46 ran births without weight decay.* E46 has eighteen births of the dense network, under both rules and on three seeds, with "three values of the weight decay of the optimizer's first parameter group, 0.25 (the standing value, the reference), 0.05 and 0", the byte table without decay in every run (`$V:$R/E41_E46_VERIFICATION_20261117.md`, section 9.1, the reviewer's own verification). In twelve of these runs `l1` is decayed at 0.05 or not at all. E38 and the second addendum of E43 ran continuations without weight decay (`INVENTORY_DENSE_LINE_20261117.md` beside this file, lines 244 and 330). In E50 `l1` stands in the decayed group of the optimizer and is not changed, because it receives no gradient (`APP e50`: `byte_table_and_l1_bit_equal_to_start` true, no optimizer state for `l1`). The second half of brief line 23 is true: in the network with a limited input no other decay of `l1` was tried.

2. *E46's result is the nearest on file to the added run and is not in the record.* It says two things that the reading of the added run needs. Without any weight decay the dense network learns the first family under both rules; on seed 73129 under the sweep R reaches 80 at update 6,400 (6,656 with the decay of 0.25) and its level is 92.8 (96.3) (section 9.2 of the verification). And without any weight decay the norm of `l1` still falls, from 13.84 to 7.31, 8.45 and 7.87 under the sweep and to 8.45, 9.47 and 9.49 under back-propagation, where the decay of 0.25 brings it to 2.87 to 3.30 under the sweep (the same section, the table of sizes). A code that is not decayed is therefore not a code of constant size, which is the property that the brief's ground names (defect 2) and that the reading must not assume (defect 1, point 5).

**Replacement.** In brief line 15 replace the sentence quoted above by:

> In the runs of E42 and of E49 it acts on every tensor but the byte table. Other values were run in the dense network only: E46 ran births with a weight decay of 0.05 and of 0 on those tensors, and E38 and E43 ran continuations without it. In E50 `l1` takes no part and is not changed.

In brief line 23 replace the sentence quoted above by:

> In every run the sweep has used one scale for a whole matrix. Every run of the network with a limited input has decayed the map that makes the code (`l1`); a weight decay of 0 was run at birth in the dense network only (E46). No other scale and no other decay of `l1` was tried in the network with a limited input.

Add to "What the record says", after the point "The size of the code in E49" (brief, line 33):

> - **E46: births of the dense network without weight decay** (the reviewer's verification, `$V:$R/E41_E46_VERIFICATION_20261117.md`, section 9; three seeds, both rules, 24,576 updates). With a weight decay of 0 on every tensor the dense network learns the first family under both rules; on seed 73129 under the sweep R reaches 80 at update 6,400 (6,656 with the decay of 0.25) and its level is 92.8 (96.3). Without any weight decay the norm of `l1` still falls, from 13.84 to 7.31, 8.45 and 7.87 under the sweep on the three seeds (0.53 to 0.61 of its starting size) and to 8.45, 9.47 and 9.49 under back-propagation; with the decay of 0.25 it falls to 2.87 to 3.30 under the sweep. The steps of the gradient thus produce a part of the fall of `l1`, and a code that is not decayed is not a code of constant size. E46 removed the decay from all tensors at once, in a network with a dense input map; the run of this job removes it from `l1` alone, in the network with a limited input.

### 5. `code_mean_abs` can be computed in two ways, and more than a quarter of its positions hold the padding symbol

**The place.** Brief, line 176:

> The same entry of the list records two sizes: `l1_weight_mean_abs`, the mean of the magnitudes of the entries of `l1.weight`, and `code_mean_abs`, the mean of the magnitudes of the code `tanh(linear(m.book(x), m.l1.weight, m.l1.bias))` over all positions of the 16 batches as `X.batch_tensor` gives them.

**What is wrong.**

1. *`l1_weight_mean_abs` is one thing:* the mean of |w| over the 512 by 8 entries of `l1.weight`.

2. *`code_mean_abs` has two readings.* It can be one mean over all values at all positions of the 16 batches taken together, or the mean over the 16 batches of each batch's mean, which is how the same sentence forms the cosines ("the mean over 16 batches"). The two differ when the batches differ in their number of positions. `batch_tensor` pads a batch to its longest item (`exp45.py`, line 158), and a batch of 64 holds an item with 12 letters with a chance of 0.994 (`APP padding`), so that the difference is rare and small; but the agent that verifies the receipts from the weights needs one definition to recompute the figure.

3. *About 28% of the positions hold the padding symbol.* `batch_tensor` writes the value 257 at the first position of every item and after the end of every item shorter than the longest. An item has 10 to 22 bytes, by its 0 to 12 letters (`exp11_cell.py`, lines 228 to 232), and the input of a batch has 22 positions; on average 15.9 of them hold a byte and 6.1, that is 27.6%, hold the padding symbol (`APP padding`, computed from the text of the two functions). The row of the byte table for this symbol is zero and receives no gradient (`nn.Embedding(258, 8, padding_idx=BOS)`, `arch20.py`, line 43), so that the code at these positions is `tanh(l1.bias)`: it does not depend on the byte table or on `l1.weight`, and it is the code of no byte. A mean over all positions therefore mixes the size of the bytes' codes with a quantity of another kind, in a proportion that the sentence does not state.

4. *Neither size is the figure that the record cites, and the brief does not say so.* The reviewer's verification of E49 gives the size of `l1` as the norm at the end over the norm at the start (0.21 to 0.23) and the size of the code as the mean magnitude over the bytes in use, each counted once (0.41 at the start, 0.05 to 0.07 at the end) (`$V:$R/e49_result_verify.py`, lines 461 to 480). `l1_weight_mean_abs` is a mean of magnitudes, and `code_mean_abs` weights each byte by how often it occurs in the batches. A report that sets the new figures beside 0.41 or 0.21 would compare unlike quantities.

**Replacement.** Replace the sentence quoted above by:

> The same entry of the list records two sizes: `l1_weight_mean_abs`, the mean of the magnitudes of the entries of `l1.weight`, and `code_mean_abs`, the mean of the magnitudes of the code `tanh(linear(m.book(x), m.l1.weight, m.l1.bias))` over the positions of the 16 batches at which `x` is not the padding symbol 257, all 512 values of all such positions of the 16 batches being taken together in one mean. (The padding symbol stands at the first position of every item and after the end of the shorter items of a batch, at about 28% of the positions; its row of the byte table is zero, so that its code is `tanh(l1.bias)` and is the code of no byte.) Neither size is a figure of the reviewer's verification of E49, which gives the norm of `l1` at the end over its norm at the start and the mean magnitude of the code over the bytes in use, each counted once; the report compares each size with its own value at update 0 and not with those figures.

### 6. Small points of count and of wording

| | Brief | Text | What is wrong | Replacement |
|---|---|---|---|---|
| 1 | line 3 | "and resets five chances" | Four chances are reset: P57.1 (75 to 70), P57.2 (35 to 12), P57.3 (30 to 12) and P57.4 (50 to 30) (`APP predictions`; brief, line 288). P57.5 to P57.7 are unchanged, and P57.8 to P57.13 are set for the first time | "and resets four chances" |
| 2 | lines 3 and 266 | "the last section lists every change"; "makes three changes of the reviewer's own" | Revision 2 has further changes of the reviewer's own that the first check did not see and that the last section does not list. This check examined each and found each sound, except where defects 2 and 4 say otherwise | In line 266: "makes changes of the reviewer's own, which are listed below the table". Add to the list at line 288: "- **Other changes.** In "What the record says": the sentence on the training loss of E49's run under the sweep and the point on the size of the code in E49. In "Terms": the term *Weight decay*. In "The question": the sentence on the decay. In "Runs": "(E42, three seeds)" in place of "(E42, three seeds; E50)", and 1.5 CPU-hours for the checks and the cosines. In check 5: the limit of 21 CPU-hours. In "The runner": the commit a36de661 in the command that copies the two files, and the sentence that lets the worker correct the text of the forward. In "What it cannot show": the sentence on E50's network." |
| 3 | line 223 | "Checks 1, 2, 3 and 7 pass at the first attempt, with every hash equal" | Check 7 requires one hash to differ. "Every hash equal" was written for checks 1 and 3, where it means that rule (c) of check 1 is not used | "Checks 1, 2, 3 and 7 pass at the first attempt, and checks 1 and 3 pass without rule (c) of check 1" |
| 4 | line 258 | "the first three runs of the network with a limited input (`matrix`, `row`, `rowcells`)" | The job now has two runs with `matrix` | "the first three runs of the network with a limited input (`matrix`, `row` and `rowcells`, each with a decayed code)" |
| 5 | line 169 | "the printed lines begin with `e57` and give the scale" | The two runs with `matrix` under the sweep then print the same prefix, and their lines cannot be told apart in a log | "the printed lines begin with `e57` and give the scale and, in the run in which the code is not decayed, `cd0`" |
| 6 | line 77 | "The run in which the code is not decayed is the reviewer's addition to the check." | The sentence can be read as an addition to a check of the brief (check 7) or to the first check's findings | "The run in which the code is not decayed was added by the reviewer after the check of revision 1, which did not ask for it." |
| 7 | line 176 | "at the first set the run asserts that `tensors_sha(m.state_dict())` and the optimizer's running averages are the same before and after it" | The first set is taken before the first update, when the optimizer holds no running average (its state is created at the first step), so that the second half of the assertion cannot fail. The text is the first check's; the weakness was not seen there | "at the first two sets (before the first update and after update 2,048) the run asserts that `tensors_sha(m.state_dict())` and the optimizer's running averages are the same before and after the set" |
| 8 | lines 203, 229 and 233 | "the first reading at which"; "at one of the readings of updates 2,048, 4,096 and 6,144"; "at the reading of update 12,288" | *Reading* is not defined; point 5 of the runner speaks of a *set* and of an entry of the list `cosines` | Add to "Terms", in the point that defines the cosine (line 17): "A *reading* is one entry of the list `cosines` of a receipt: the cosines and the two sizes taken before the first update or after a multiple of 2,048 updates (point 5 of the runner)." |
| 9 | line 189 | "whether the entry is true at every row" | "Row" for a scored row, which the first check's defect 13 asked to be written out; the sentence is the first check's own | "whether the entry is true at every scored row" |
| 10 | line 87 | "[--dose N]" | The option is not explained, the word "dose" is not defined, and the brief does not say that the five runs of 49,152 updates are started with it | Add after line 90: "The option `--dose N` sets the number of updates of a run; without it a run has 24,576. The five runs of the network with a limited input under the sweep are started with `--dose 49152`." |

Three phrases of brief line 61 are figures of speech ("does not stop the sweep", "remains far behind back-propagation", "that distance lies in"); the first is the first check's. The replacement under defect 2 removes all three. "Accounts for part of the failure" (line 217) is replaced under defect 1, and "lets the sweep learn" (lines 77 and 286) under defect 2.

## Verified and found sound

### A. The option `--code-decay 0`

- **The optimizer has the two groups that `no_code_decay` expects.** `exp16_integrated.adamw16` (lines 176 to 187) builds, with `E16_BOOKDECAY=off` (required by `exp20_reference.py`, line 24), one group of every parameter but the byte table, which takes the default weight decay, and one group of the byte table with weight decay 0.0, and returns `T.optim.AdamW(groups, lr=lr, weight_decay=wd, betas=(.9, .98))`. `exp26_candidate.make_opt` (lines 72 to 76; `WD = 0.25`, line 40) asserts that the groups read back are these two. E49's receipt of `in16` lists twelve tensors at 0.25 and `book.weight` at 0.0 (`APP e49`); the network has thirteen parameters and no other tensor (`arch20.py`, lines 43 to 50). After the move the groups hold ten and three tensors, as check 7 says. Both assertions of the function hold, and `q is p` finds each tensor once, since `m.l1.weight` is the object that `named_parameters` gave to the optimizer.
- **The move is the operation that E42's runner already uses.** `exp42_cells.make_opt` (lines 161 to 168) selects the decayed group in the same way and extends its list after the optimizer has been built, as the brief says.
- **How AdamW treats the changed lists** (read, not run). A group's `params` is a Python list. At every step the optimizer goes through `param_groups`, takes the weight decay, the rate and the betas from the group, and for each tensor of the group's list that has a gradient creates the tensor's state at its first step and updates it. It keeps no other record of which tensor belongs to which group. A tensor moved before the first step is therefore treated as a member of its new group from the first step on. The receipts of E50, on torch 2.15.0+cpu, agree with the part of this that they can show: `l1` stands in the decayed group, receives no gradient, has no optimizer state after 49,152 steps and is bit-equal to its start, so that the decay is applied tensor by tensor inside the step (`APP e50`).
- **The place in the copied run.** The move follows line 181 and precedes line 195, where `R42.opt_record` writes the entry `optimizer`; the entry therefore shows the groups after the move, as the brief says. Nothing between line 181 and the first update uses the optimizer except `check_masks` at the scored row of update 0, which reads the state of the three masked maps with `opt.state.get` and creates nothing.
- **No later line fails or records something false.** `R42.opt_record` (lines 171 to 179) names each tensor by its identity and lists whatever the groups hold; its first entry `weight_decay` 0.25 is the optimizer's default and remains true. `check_masks` (`exp49_interface.py`, lines 95 to 103) reads the three masked maps only. `R_.opt_steps` (`exp20_reference.py`, lines 112 to 114) returns the set of the step counts, `[49152]`. `optimizer_read_back` (line 239) is the same record at the end. The only assertion on the groups is inside `C26.make_opt`, before the move.
- **The definition of weight decay** (brief, line 15) is right for AdamW: the weight is multiplied by 1 − rate × decay, and the step formed from the two running averages is then subtracted; the decay does not enter the gradient or the averages. It acts on a tensor only at a step at which the tensor has a gradient, which is every step for the thirteen tensors of this job.
- **Check 7 fails when the option does not act,** provided check 1 passed bit for bit: the entry `optimizer` is read back from the optimizer, and the final hash is then `88cad76a…` (`APP e49`). The exception is defect 3.
- **"Necessary for the failure"** follows from the outcome it is attached to, for that run on that seed (defect 1, point 4). The statement of what the run tests (brief, line 77: "whether the decay of `l1` is necessary for the failure of E49's run; it does not test the scale") is sound.
- **P57.12 and P57.13** are decidable from the receipt of the added run.

### B. The record

- **The size of the code in E49** (brief, line 33) is faithful to section 2.7 of the reviewer's verification. Every figure is in the table of that section: `l1` at 0.28, 0.31, 0.26 and 0.23 of its starting size and the input map at 3.29, 3.59, 3.15 and 3.45 in the four runs with a dense input map, with products of 1.14 to 1.74; `l1` at 0.49 to 0.53 and the code at 0.34 to 0.44 in the three runs with a limited input under back-propagation; `l1` at 0.21 to 0.23, the kept input weights at 0.78 to 1.27 and the code at 0.05 to 0.07 in the three under the sweep; the distances 0.055 to 0.08, 0.26 to 0.34 and 0.39 to 0.44.
- **The training loss of E49's run under the sweep** (brief, line 32): from update 4,096 on, the least mean loss of a scored row is 0.9953 (update 10,240) and the greatest 1.0520 (update 21,504) (`APP e49`).
- **The figures of E50's stage 2** (brief, line 61), recomputed from the two receipts at c6c24d9c, each complete with 193 scored rows (`APP e50`):

  | Rule | Levels A, R, T at update 24,576 | Levels A, R, T at update 49,152 | Reaches 80 |
  |---|---|---|---|
  | S | 32.72, 28.97, 54.61 | 49.50, 43.63, 74.83 | none of A, R and T |
  | B | 67.94, 62.12, 85.28 | 83.78, 79.42, 96.33 | A at 40,704; T at 18,944; R not |

  Under the sweep the last three scored rows of R exceed the three that end 2,048 updates earlier by 1.32 points. All of the brief's figures agree.
- **The two statements drawn from E50** are supported once the network is described rightly (defect 2, point 5).

### C. The chances

Four chances are reset against revision 1 and six are set for the first time. Three pairs have a statement that implies another, in practice if not in logic: P57.2 implies P57.4 (12% and 30%), P57.2 implies P57.8 (12% and 30%), and P57.13 implies P57.12 (20% and 45%). In each pair the implying statement has the lower chance (`APP predictions`). No other pair is bound: learning the family within 49,152 updates does not imply a level of T of 50 at update 24,576, and the converse fails as E50's run under the sweep shows. The grounds given for the lower chances of P57.2 to P57.4 (brief, line 288) agree with the record: 24,064 / 20,480 = 1.17, and R at 29.0 at update 24,576 in E50.

### D. The replacement texts of the first check

The first check gives 159 lines of replacement text. 119 are in revision 2 word for word; 40 differ (second script of the appendix). Every difference is one of the following, and none is an error.

- *The added run:* the column "Code decayed" and its row in the table of runs; "five" for "four" runs under the sweep (three places); "the decay of the code in one run"; "a change" for "a scale"; the option, the argument `code_decay` and the ending `_cd0`; "and a decayed code" where a run with `matrix` is named (lines 191, 203, 215, 229 and 232); "concerns the four runs with a decayed code"; the sizes in point 5 of the runner, in the reading and in "Done"; the last sentence of the sixth point of the record; the point on the decay in "What it cannot show", which replaces the first check's.
- *Counts:* eight to nine runs (lines 3, 81, 190, 250, 258 and 262); checks 1, 2, 3 and 7 before a run starts (lines 181, 223 and 258), the command `check <1|2|3|5|7>`, and checks 1, 2, 3, 5, 6 and 7 in "Done"; 11.2 to 13.1 and 12 to 15 CPU-hours; 18 to 21 in check 5; four and a half to five hours (lines 81 and 250).
- *The chances:* P57.1 to P57.4 reset and P57.8 to P57.11 set.
- *E50:* the point of the record rewritten from the completed runs; "(E42, three seeds)" without "E50", which leaves the range 1.04 to 1.89 as it was, since E50's ratio of 1.85 lies inside it; one sentence added in "What it cannot show".
- *Three small improvements:* the commit in place of the branch in the command that copies the files; the import of `exp26_candidate`; the sentence that lets the worker correct the forward.

The instructions that the first check gave outside quoted text are entered as well: score T, the dry run, the levels of E49, the 257 output classes, the sentence on the verification of the result, the reporting points and the terms added under its defect 13.

### E. The counts within revision 2

The table of runs has nine runs, of which five are of the network with a limited input under the sweep and four of these have a decayed code; every later mention agrees. The checks before the runs are 1, 2, 3, 5 and 7 in every place; check 7 follows check 1, whose dry run it uses. The one count that is wrong is "five chances" (defect 6, point 1).

### F. The cost and the wall-clock time

From the brief's unit costs (`APP cost`): five runs at 1.9, one at 0.95 and three at 0.87 CPU-hours are 13.06, the brief's 13.1. The cosines are 5 × 25 + 4 × 13 = 177 sets, which at 10 CPU-seconds are 0.49 CPU-hours, the brief's 0.5. With 1.5 for the checks and the cosines the total is 14.6, the brief's "about 15", and the limit of check 5 is 1.44 times that. The brief gives no unit cost for the checks; its 1.5 leaves 1.0 CPU-hours for them, where the first check computed 0.35 and check 7 adds 0.02, so that the figure errs on the safe side. With three processes at a time and the order of starting that the brief gives, the last run ends after 4.7 hours, and after 4.8 hours with the cosines inside the runs; the brief's "about five hours" is right. The unit costs agree with the receipts: E49's two runs cost 0.918 and 0.956 CPU-hours for 24,576 updates, and E50's run of 49,152 updates under the sweep cost 2.00.

## Not checked

1. Nothing was run in torch. How AdamW treats a list changed before the first step is stated from the source of the library as this check knows it and from two receipts on file; it was not executed on torch 2.15.0. Part (a) of the replacement of check 7 tests it on the worker's installation, and its expectation (eleven tensors equal bit for bit, a difference of 0.00025 times the starting value in `l1`) is derived here and was not run.
2. The figures of E46 are those of the reviewer's verification; they were not recomputed from E46's receipts. The continuations without weight decay of E38 and of E43 are taken from the inventory beside this file.
3. The weights of E50 were not read. Its levels are recomputed from the receipts; the statements quoted from its two reports were not verified.
4. The share of positions with the padding symbol is computed from the text of `exp11_cell.render` and of `exp45.batch_tensor`, not from batches.
5. The findings of the first check, the text of the forward and the argument for equality bit for bit were not examined again.
6. The session named in "Reporting" and the tool by which the worker writes to it were not tried.

## State of the brief when the check ended

The SHA-256 of the brief was `a5b3a351db8aee10eef5afc35f4b24000a55e31cf2e6b8a732c01d0be6f291a0` when the check ended, as when it began. This check did not edit the brief and committed nothing.

## Appendix: the scripts

Both scripts were kept in a folder of this check under the scratch directory and are given here in full. They use the standard library only and were run with Python 3.13.17 from the repository root.

**The first script** (`APP <part>`): `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -I -B e57r2_check.py <part>`. Each part takes less than a second.

```python
"""Second check of the brief of E57 (revision 2): the figures behind CHECK_OF_TASK_E57_BRIEF_REVISION_2_20261117.md.
Read-only; nothing is trained; no line of another branch is executed.

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -I -B <this file> <part>      (from the repository root)

    e50        levels and updates to 80 of E50's stage 2 (C16-in under both rules) from the two receipts at c6c24d9c
    cost       the cost and the wall-clock time of "Runs" from the brief's own unit costs
    decay      the factor by which weight decay alone multiplies a weight over a number of updates
    e49        the statements of revision 2 on E49's two runs of in16 (the training loss, the optimizer's groups, the hashes)
    padding    the share of the positions of a batch that hold the padding symbol
    predictions  the chances of revision 2 against revision 1, and the pairs in which one statement implies another
"""
import json
import subprocess
import sys

M = "MOD97_LOCAL_RULE_20261106"
E50 = "c6c24d9c"


def show(ref, path):
    return subprocess.run(["git", "show", f"{ref}:{path}"], capture_output=True, check=True).stdout


def level(curve, key, upto=None):
    rows = [r for r in curve if upto is None or r["step"] <= upto][-3:]
    return sum(r[key] for r in rows) / 3, [r["step"] for r in rows], [r[key] for r in rows]


def reach(curve, key, thr=80.0, upto=None):
    rows = [r for r in curve if upto is None or r["step"] <= upto]
    for i in range(len(rows) - 2):
        if all(rows[j][key] >= thr for j in (i, i + 1, i + 2)):
            return rows[i]["step"]
    return None


def part_e50():
    for rule in "SB":
        rec = json.loads(show(E50, f"{M}/exp45/e50/e50_C16-in_{rule}_73129.json"))
        c = rec["curve"]
        steps = [r["step"] for r in c]
        print(f"C16-in {rule}: complete {rec.get('complete')}, dose {rec.get('dose')}, rows {len(c)}, first step {steps[0]}, last {steps[-1]}, "
              f"spacing {sorted(set(b - a for a, b in zip(steps, steps[1:])))}, setting {rec.get('setting')}, k_in {rec.get('k_in')}, k_out {rec.get('k_out')}, N {rec.get('N')}")
        for upto in (24576, None):
            out = []
            for k in ("A", "R", "taught"):
                lv, st, vals = level(c, k, upto)
                out.append(f"{k} {lv:.2f} (rows {st}: {[round(v, 2) for v in vals]})")
            print("   level at", upto or steps[-1], ":", "; ".join(out))
            print("   reaches 80 within", upto or steps[-1], ":", {k: reach(c, k, 80.0, upto) for k in ("A", "R", "taught")},
                  "| reaches 50:", {k: reach(c, k, 50.0, upto) for k in ("A", "R", "taught")})
        # R: last three rows minus the three rows ending 2,048 updates earlier
        for upto in (24576, None):
            rows = [r for r in c if upto is None or r["step"] <= upto]
            last = sum(r["R"] for r in rows[-3:]) / 3
            earlier = sum(r["R"] for r in rows[-11:-8]) / 3
            print(f"   R, last three rows minus the three ending 2,048 updates earlier (rows {[r['step'] for r in rows[-11:-8]]}), at {upto or steps[-1]}: {last - earlier:+.2f}")
        print("   max single row of R:", max((r["R"], r["step"]) for r in c), "; max single row of taught:", max((r["taught"], r["step"]) for r in c))
        print("   keys of the receipt that describe the optimizer and the frozen tensors:")
        for k in rec:
            if any(s in k.lower() for s in ("optim", "frozen", "bit_equal", "code", "train")):
                v = rec[k]
                print("     ", k, ":", json.dumps(v)[:600])


def part_cost():
    long_, bp, other = 1.9, 0.95, 0.87
    runs = 5 * long_ + 1 * bp + 3 * other
    print(f"runs: 5 x {long_} + 1 x {bp} + 3 x {other} = {runs:.2f} CPU-hours (the brief: 13.1)")
    sets = 5 * 25 + 4 * 13
    print(f"sets of cosines: 5 x 25 + 4 x 13 = {sets}; at 10 CPU-seconds a set {sets * 10 / 3600:.3f} CPU-hours (the brief: 0.5)")
    print(f"   a set is five passes on 16 batches = 80 passes; at 0.13 CPU-seconds an update that is {80 * 0.13:.1f} CPU-seconds")
    print(f"   the first check's estimate: 4 x 25 + 4 x 13 = {4 * 25 + 4 * 13} sets = {(4 * 25 + 4 * 13) * 10 / 3600:.3f} CPU-hours (it wrote 0.45)")
    print(f"total with 1.5 for the checks and the cosines: {runs + 1.5:.2f} (the brief: about 15)")
    # wall-clock time with three processes: a schedule in the brief's order of starting
    jobs = [("in16 matrix", long_), ("in16 row", long_), ("in16 rowcells", long_), ("in16 rowcol", long_), ("in16 matrix cd0", long_),
            ("in16 B", bp), ("F10 row", other), ("F10 rowcol", other), ("G row", other)]
    slots = [0.0, 0.0, 0.0]; ends = []
    for name, h in jobs:
        i = slots.index(min(slots)); start = slots[i]; slots[i] = start + h; ends.append((name, start, slots[i]))
    for name, s, e in ends:
        print(f"   {name:16s} starts {s:5.2f} h  ends {e:5.2f} h")
    print(f"wall-clock time of the runs, cosines not counted: {max(slots):.2f} h; lower bound (sum / 3): {runs / 3:.2f} h")
    # with the cosines inside the runs: 25 sets of 10 s in a long run, 13 in a short one
    jobs2 = [(n, h + (25 if h == long_ else 13) * 10 / 3600) for n, h in jobs]
    slots = [0.0, 0.0, 0.0]
    for name, h in jobs2:
        i = slots.index(min(slots)); slots[i] += h
    print(f"wall-clock time of the runs with the cosines inside them: {max(slots):.2f} h; lower bound {sum(h for _, h in jobs2) / 3:.2f} h")
    # the eight runs of the first check
    jobs8 = [long_] * 4 + [bp] + [other] * 3
    slots = [0.0, 0.0, 0.0]
    for h in jobs8:
        i = slots.index(min(slots)); slots[i] += h
    print(f"the first check's eight runs: {sum(jobs8):.2f} CPU-hours, {max(slots):.2f} h with three processes (it wrote 11.2 and about four and a half)")
    print(f"check 5's limit: 21 / 15 = {21 / 15:.2f}; the first check's 18 / 12 = {18 / 12:.2f}; with 14.6: {21 / (runs + 1.5):.2f}")


def part_decay():
    f = 1 - 0.001 * 0.25
    for n in (256, 2048, 24576, 49152):
        print(f"(1 - 0.001 x 0.25)^{n} = {f ** n:.6f}")




def part_e49():
    """The statements of revision 2 on E49's two runs of in16 that the first check did not see."""
    W9 = "a36de661"
    for rule in "SB":
        rec = json.loads(show(W9, f"{M}/exp45/e49/e49_in16_{rule}_73129.json"))
        c = rec["curve"]
        ls = [(r["loss"], r["step"]) for r in c if r["step"] >= 4096]
        print(f"in16 {rule}: rows {len(c)}; mean training loss of the scored rows from update 4,096 on: min {min(ls)}, max {max(ls)}")
        ls2 = [(r["loss"], r["step"]) for r in c if r["step"] >= 256]
        print(f"   from update 256 on: min {min(ls2)}, max {max(ls2)}; first rows:", [(r["step"], round(r["loss"], 3)) for r in c[1:17]])
        print("   largest A, R, taught at any scored row:", max(r["A"] for r in c), max(r["R"] for r in c), max(r["taught"] for r in c))
        print("   levels:", {k: round(level(c, k)[0], 2) for k in ("A", "R", "taught")}, "| reaches 80:", {k: reach(c, k) for k in ("A", "R", "taught")},
              "| taught reaches 50:", reach(c, "taught", 50.0))
        print("   optimizer groups:", [(g["weight_decay"], g["params"]) for g in rec["optimizer"]["groups"]])
        print("   tensors_sha256", rec["tensors_sha256"], "start", rec["start_tensors_sha256"], "cpu h", round((rec["cpu_updates"] + rec["cpu_scoring"]) / 3600, 3))
        print("   cpu per update", round(rec["cpu_updates"] / rec["dose"], 4), "s; per scored row", round(rec["cpu_scoring"] / len(c), 2), "s")
    for name in ("e49_in16_S_dry256_73129.json", "e49_in16_B_dry256_73129.json"):
        rec = json.loads(show(W9, f"{M}/exp45/e49_qual/check6_73129_runs/{name}"))
        print(name, rec["tensors_sha256"], "start", rec["start_tensors_sha256"], "rows", [r["step"] for r in rec["curve"]])


PARTS = {"e50": part_e50, "cost": part_cost, "decay": part_decay, "e49": part_e49}


def part_padding():
    """The share of the positions of a batch that hold the padding symbol (257), from the text of exp11_cell.render and exp45.batch_tensor:
    a row is the prompt (7 + k bytes, k letters drawn uniformly from 0 to 12) and the answer (two digits and the end symbol), 10 + k bytes;
    batch_tensor writes the padding symbol first, then the row, pads to the longest row of the batch plus one, and gives all columns but the
    last as the input x."""
    ks = range(13)
    p_full = 1 - (12 / 13) ** 64
    print(f"chance that a batch of 64 holds a row with 12 letters: {p_full:.4f}")
    cols = 22                                             # the longest row has 22 bytes; x has 22 columns
    filled = [min(10 + k, cols - 1) for k in ks]          # positions of a row that hold a byte (the first column is the padding symbol)
    mean_filled = sum(filled) / len(filled)
    print(f"columns of x: {cols}; bytes of a row that x holds, mean over k: {mean_filled:.2f}; share of positions with the padding symbol: "
          f"{1 - mean_filled / cols:.3f}")


def part_predictions():
    """The chances of revision 2 and the pairs of statements of which one implies the other (in practice or in logic)."""
    P = {1: 70, 2: 12, 3: 12, 4: 30, 5: 65, 6: 85, 7: 60, 8: 30, 9: 20, 10: 85, 11: 65, 12: 45, 13: 20}
    r1 = {1: 75, 2: 35, 3: 30, 4: 50, 5: 65, 6: 85, 7: 60}
    print("chances reset against revision 1:", [f"P57.{k}: {r1[k]} -> {P[k]}" for k in r1 if r1[k] != P[k]])
    implies = [(2, 4, "learns within 24,576 under row => T at 24,576 is 50 or more (in practice: T is not below A)"),
               (2, 8, "learns within 24,576 under row => learns within 49,152 (unless the levels fall below 80 afterwards)"),
               (13, 12, "learns within 49,152 without decay => T at 49,152 is 50 or more (in practice)")]
    for a, b, why in implies:
        print(f"P57.{a} ({P[a]}%) implies P57.{b} ({P[b]}%): {'consistent' if P[a] <= P[b] else 'INCONSISTENT'}; {why}")


PARTS.update(padding=part_padding, predictions=part_predictions)


if __name__ == "__main__":
    PARTS[sys.argv[1]]()
```

**The second script** compares the replacement texts of the first check with revision 2: `python3 -I -B compare_r2.py <the first check> <the brief>`. Its last line of output was "lines of replacement text: 159; word for word in revision 2: 119; different: 40".

```python
"""Compare each replacement text of the first check (lines that begin with '> ') with revision 2 of the brief.
For every such line that is not in revision 2 word for word, print the closest line of revision 2 and the words that differ.

    python3 -I -B compare_r2.py <the first check> <revision 2 of the brief>
"""
import difflib
import re
import sys

check = open(sys.argv[1], encoding="utf-8").read().splitlines()
brief = open(sys.argv[2], encoding="utf-8").read().splitlines()
norm = lambda s: re.sub(r"\s+", " ", s.strip())
B = [norm(l) for l in brief]
Bset = set(B)
same = diff = 0
for i, line in enumerate(check, 1):
    if not line.startswith(">"):
        continue
    t = norm(line[1:])
    if not t:
        continue
    if t in Bset:
        same += 1
        continue
    # closest line of the brief
    best = max(range(len(B)), key=lambda j: difflib.SequenceMatcher(None, t, B[j], autojunk=False).quick_ratio() if B[j] else 0)
    ratio = difflib.SequenceMatcher(None, t, B[best], autojunk=False).ratio()
    diff += 1
    print(f"--- check line {i} (ratio {ratio:.3f}) against brief line {best + 1}")
    if ratio < 0.5:
        print("    NO COUNTERPART:", t[:300])
        continue
    a, b = t.split(" "), B[best].split(" ")
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        ctx = " ".join(a[max(0, i1 - 4):i1])
        print(f"    [{tag}] ...{ctx} | CHECK: '{' '.join(a[i1:i2])}' | BRIEF: '{' '.join(b[j1:j2])}'")
print(f"lines of replacement text: {same + diff}; word for word in revision 2: {same}; different: {diff}")
```
