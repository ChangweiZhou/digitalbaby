# Check of the brief E62, the number of connections of a cell (revision 1), 18 November 2026

Checked by an outside checker who had read no other opinion of the brief. Clock at the start of writing: 16:52 UTC, 18 November 2026. The brief is `TASK_E62_THE_NUMBER_OF_CONNECTIONS_OF_A_CELL_20261118.md` beside this file. The figures quoted here are printed by `check_e62_brief.py` beside this file (standard library and numpy; it writes no file; about 15 seconds), run from the repository's root as

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -I PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/COORDINATION_20261117/check_e62_brief.py

The lines of its output are cited below by their letter (A to D).

## 1. What was checked and how

- **The networks.** `e42_cells.neighbourhood` and the drawing of connections in `e42_cells.draws` were copied into the script (generators `random.Random` with string seeds only; the periods and phases use generators of their own and do not affect the connections). The copy reproduces the four digests that `e42_cells.py` itself asserts for seed 73129 (N = 5, 10 and 20 from the neighbourhood and N = 10 from all cells; line A). For seed 73129 and each of the ten settings (N = 2, 4, 8, 10, 16, 32, 64, 128 and 255 from all cells, and N = 10 from the neighbourhood) the script computes the figures of the brief's point 3 of the runner and of its check 2, together with the strongly and weakly connected parts of the graph (lines B).
- **The source.** `exp50_code.py` (`build`, `run`, `counts`, `fixed`, `cosines`, `knows_figure`, `coverage`, `install`, `step`, `fwd`), `e50_sparse_code.forward`, `e42_cells.Layer`, `rec`, `sign_scale` and `draws`, `exp49_interface.kept`, `check_masks`, `magnitudes`, `readers` and `weights_blob`, `exp42_cells.check_limit`, `make_opt` and `opt_record`, `exp3_state.SignLinear` and `lin`, `arch26.Mat`, `exp20_reference.tensors_sha` and its environment gate, and `exp52_resume.py` were read on the default branch at e647e7d7. Nothing from the repository was executed.
- **The record.** The two receipts `$D:$M/exp45/e50/e50_C46-one_<S|B>_73129.json` (byte-identical to the copies given for reading) were read with `git show`, and every figure of the brief's table was recomputed with the brief's definitions (lines C). The memo's table of E42 on three seeds (section 28.3) and its table of E42's second stage on seed 73129 were read for the brief's 2.36 and 1.70.
- **The documents it rests on.** Section 1 and step 2 of `STEPS_FROM_THE_PRINCIPLE_20261118.md`, the coordinator's decisions 30 to 37 on the default branch, and the sister brief E61 for consistency of terms and clauses (E61 itself was not checked).

## 2. The graph of each setting

A step goes from a cell to the cells that read it, as the brief's point 3 states. Because every cell reads itself, the cells reached within k steps are the cells whose state at position t + k can depend on the given cell's state at position t; the counts below include the cell itself. "Every cell from every cell" is the smallest k at which every cell is reached from every cell. "Outside" counts the connections (cell i reads cell j) for which j is not among the 48 cells of i's neighbourhood; "named" counts the different cells j that occur in such connections.

| Setting | Exactly N sources | Read by no other cell | Readers of a cell (least to most) | Reached within 1, 2, 3, 4, 6, 8 steps: least / mean / most | Every cell from every cell | Strongly connected parts (largest) | Connections outside the neighbourhood | Different cells named outside |
|---|---|---|---|---|---|---|---|---|
| N = 2, all | yes | 40 | 0 to 6 | 1/3.0/7; 1/6.9/20; 1/14.5/50; 1/28.6/98; 1/88.6/224; 1/165.0/256 | none | 49 (207) | 431 of 512 | 203 |
| N = 4, all | yes | 4 | 0 to 11 | 1/5.0/12; 1/20.4/52; 1/71.8/172; 1/175.9/255; 1/251.4/256; 1/252.0/256 | none | 5 (252) | 868 of 1,024 | 247 |
| N = 8, all | yes | 0 | 2 to 15 | 3/9.0/16; 14/64.9/123; 78/223.7/256; 241/255.9/256; 256; 256 | 5 | 1 | 1,669 of 2,048 | 255 |
| N = 10, all | yes | 0 | 3 to 19 | 4/11.0/20; 35/91.8/148; 203/249.7/256; 256; 256; 256 | 4 | 1 | 2,049 of 2,560 | 256 |
| N = 16, all | yes | 0 | 5 to 27 | 6/17.0/28; 75/171.0/220; 254/256.0/256; 256; 256; 256 | 4 | 1 | 3,307 of 4,096 | 256 |
| N = 32, all | yes | 0 | 16 to 54 | 17/33.0/55; 232/252.9/256; 256; 256; 256; 256 | 3 | 1 | 6,617 of 8,192 | 256 |
| N = 64, all | yes | 0 | 48 to 88 | 49/65.0/89; 256 from 2 steps on | 2 | 1 | 13,289 of 16,384 | 256 |
| N = 128, all | yes | 0 | 107 to 149 | 108/129.0/150; 256 from 2 steps on | 2 | 1 | 26,620 of 32,768 | 256 |
| N = 255, all | yes (255) | 0 | 255 | 256 from 1 step on | 1 | 1 | 52,992 of 65,280 | 256 |
| N = 10, neighbourhood (the record) | yes | 0 | 3 to 17 | 4/11.0/18; 29/73.9/103; 137/213.3/242; 247/255.9/256; 256; 256 | 5 | 1 | 0 of 2,560 | 0 |

Every setting is one weakly connected part. The digests `dg16(A)` of the ten lists of connections, in the order of the table, are 9622342ea3d0e56f, 7501b501fcd35c94, 10d7ba852a8bbb61, 4a9feeeb77ff57d2, fbe2bf8967eefa58, 3037c7cdde9bfefb, a6f88a4f5ef85709, 2ad32f75b1f251ed, 3b1a3a5516ed1271 and fde78220c57da3d4 (line B). The draws are not nested: the connections of a cell at N = 2 are contained in its connections at N = 4 for one cell of 256, and the same holds for 4 and 8, 8 and 16, and 16 and 32. N = 10 from all cells and N = 10 from the neighbourhood share 112 of their 2,560 connections.

## 3. Defects, in order of weight

### Defect 1. A resumed run recomputes the first batch's figures on the wrong batch and the trained network, and check 3 would not notice

**Place.** The runner, point 4 (saving and resuming), and check 3.

**What is wrong.** In the text of `exp50_code.run` that point 2 copies, the variable `first` holds the run's first batch; it is set at update 1 by `if first is None:`, which also writes `rec["first_batch_loss_start"]` and `rec["cosine_first_batch_start"]`. The same batch is used later for `rec["first_dose"]["cosine_first_batch"]` at update 24,576 and for `rec["cosine_first_batch_end"]`. Point 4 says that a resumed run "draws the first u batches again" and "goes on from update u + 1", but it does not say that `first` is set from the redrawn batches. A copy that follows point 4 literally enters the loop with `first = None`; at update u + 1 it then takes batch u + 1 as `first`, overwrites the two start entries of the saved receipt with figures computed on the trained network, and computes the two later cosines on the wrong batch. Check 3 compares only `tensors_sha256` and the scored rows; these entries are outside the rows, so the check passes. (The update itself is not affected: `cosines` and `first_batch_loss` change no weight and no optimizer state.)

**Evidence.** `exp50_code.py`, lines 395 to 400, 407 to 410 and 413.

**Replacement text.** In point 4, after "loads the two states,": "sets `first` to the first of the redrawn batches and does not compute `first_batch_loss_start` or `cosine_first_batch_start` again (they are in the saved receipt),". In check 3, after "their scored rows are equal in every field whose name does not begin with `cpu` or `wall`;": "their entries `first_batch_loss_start`, `cosine_first_batch_start`, `cosine_first_batch_end`, `data_digest`, `optimizer_steps` and `tensors_sha256` are equal;".

### Defect 2. The assertion on the starting file at a resumption cannot hold as written

**Place.** The runner, point 4: "asserts ... that the starting file has the saved `start_tensors_sha256`".

**What is wrong.** `start_tensors_sha256` is `R_.tensors_sha(m.state_dict())`, the hash of the module's tensors. The starting file is `weights_blob(n)`: the module's tensors together with the recurrent mask, `made`, the periods, the phases, the two masks of the interface, `k_in`, `k_out`, the code table, the owners and the value of a turned-on entry. Its tensors therefore have another hash, and the receipt records the file's own hash separately as `start_sha256`. A worker who follows the sentence literally obtains an assertion that fails at the first resumption (check 3 fails), or must guess what was meant. The brief also does not say that the resumed run must not write the starting file again (the copied text of `run` writes it before the loop).

**Evidence.** `exp50_code.py`, lines 305 to 310, 348 and 363; `exp20_reference.tensors_sha`.

**Replacement text.** "asserts that the rebuilt network's `R_.tensors_sha(m.state_dict())` equals the saved receipt's `start_tensors_sha256` and that the SHA-256 of the starting file on disk equals its `start_sha256`; the starting file is not written again."

### Defect 3. Check 2's condition on cells outside the neighbourhood is ambiguous, nearly vacuous, and does not test the draw

**Place.** Check 2: "with `all`, the connections of the 256 cells together name at least 200 different cells outside the cell's neighbourhood (for N of 4 or more)".

**What is wrong.** (a) "The cell's neighbourhood" has no antecedent when the subject is all 256 cells together; the most natural reading is the number of different cells j such that some cell i reads j and j lies outside i's neighbourhood. (b) Under that reading the figure is 203 at N = 2, 247 at N = 4, 255 at N = 8 and 256 from N = 10 on; the condition is therefore true for every N, and the exclusion of N = 2 has no ground (the threshold happens to lie three cells below the value at N = 2). (c) The condition catches only an implementation that draws from the neighbourhood; an implementation that draws from all cells with another generator, another order or another N passes. The exact digests of the draws are available and test the network exactly. (d) The check omits the record's network, so that its figures, which the report's table needs beside the others, are not produced by the same code. (e) "The kept weights equal those of `arch26.new_net(73129)` times the square root of 256 / (N + 1)" does not say whether equality is bitwise; it is, if the product is formed as in `L42.Layer`.

**Evidence.** Section 2 above; line B of the script; `e42_cells.Layer.__init__`.

**Replacement text.** "2. **The networks are what the brief says.** For each of the ten settings (the nine of the runs and `10 near`), at the start: every cell reads exactly N other cells and itself; the digest `hashlib.sha256(json.dumps(A).encode()).hexdigest()[:16]` of the lists of connections `A` (for cell i, the sorted indices j ≠ i with `layer.mask[i, j]`) equals, for N = 2, 4, 8, 10, 16, 32, 64, 128 and 255 from all cells and N = 10 from the neighbourhood: 9622342ea3d0e56f, 7501b501fcd35c94, 10d7ba852a8bbb61, 4a9feeeb77ff57d2, fbe2bf8967eefa58, 3037c7cdde9bfefb, a6f88a4f5ef85709, 2ad32f75b1f251ed, 3b1a3a5516ed1271, fde78220c57da3d4; the numbers of connections outside the 48 cells of the reading cell's neighbourhood are 431, 868, 1,669, 2,049, 3,307, 6,617, 13,289, 26,620, 52,992 and 0; every weight outside the mask is zero; the kept weights are bit-equal to `arch26.new_net(73129).rnn.weight_hh_l0 * (layer.mask3().float() * (256 / (N + 1)) ** 0.5)` (to the unscaled weights at N = 255); [the rest of the check unchanged]. The check records the figures of point 3 of the runner for each setting and asserts that they equal the reviewer's table in `CHECK_OF_TASK_E62_BRIEF_20261118.md`, section 2."

### Defect 4. At N = 2 and N = 4 the network falls into parts that cannot reach one another, and the brief neither says so nor states the steps correctly for N = 16

**Place.** Terms, "What it cannot show" (second item), and the readings.

**What is wrong.** (a) At N = 2, 40 of the 256 cells are read by no other cell; their states reach only the readout, and only through the readout can the error reach them. The graph has 49 strongly connected parts: one of 207 cells, from which every cell is reached, and 49 cells that cannot reach that part; no cell is reached from every cell, so no number of steps covers the network. At N = 4 the same holds for 4 cells (5 strongly connected parts, the largest of 252). From N = 8 on, the graph is strongly connected. The draw is close to its expectation (256 (1 − N/255)^255 is 34.4 at N = 2 and 4.5 at N = 4), so this is a property of the design and not of the seed. A slow or failed run at N = 2 or N = 4 therefore cannot be read as the effect of the number of connections alone: part of it is that a sixth of the cells (at N = 2) do not take part in the recurrence. (b) The second item of "What it cannot show" states: "a cell that reads 16 or more others reaches every cell within two or three steps". At N = 16 every cell is reached from every cell only within four steps; within three steps the least number reached from a cell is 254. The statement holds at N = 32 (three steps) and at 64 and above (two steps).

**Evidence.** Section 2 above; line B.

**Replacement text.** Add to Terms: "*Reached within k steps:* the cells whose state at position t + k can depend on a given cell's state at position t, the cell itself included (every cell reads itself). With connections from all cells, seed 73129, N = 2 leaves 40 cells that no other cell reads and 49 cells that cannot reach the remaining 207, and N = 4 leaves 4 such cells; from N = 8 on, every cell is reached from every cell, within 5 steps at N = 8, 4 at N = 10 and 16, 3 at N = 32, 2 at N = 64 and 128 and 1 at N = 255 (5 for the record's network)." Replace the second item of "What it cannot show" by: "A number of connections that suits a larger network. With 256 cells, every cell is reached from every cell within four steps at N = 16, three at N = 32 and two at N = 64, so that Mr. Xu's example of several passes (1, 50, 50², 50³) cannot be reproduced at this width; the grid shows the dependence on N at this width and nothing about how N should grow with the number of cells." Add to the report's obligations under "How the result is stated": "The report states for N = 2 and N = 4 the number of cells that no other cell reads and that the network is not one strongly connected part, and does not read their runs as the effect of N alone."

### Defect 5. "A copy of the text with these changes and no other" cannot be carried out literally

**Place.** The runner, points 1 and 2.

**What is wrong.** The text of `exp50_code.run` calls, unqualified, about twenty names of the module `exp50_code` (`build`, `counts`, `coverage`, `digit_groups`, `weights_blob`, `first_batch_loss`, `cosines`, `score_all`, `fixed`, `knows_figure`, `input_size`, `install`, `step`, `versions`, `provenance`, `receipt_path`, `SETTINGS`, `DOSES`, `FIRST_DOSE`, `WEIGHTS_TAKING_PART`, `PRODUCTS_TURNED_ON`, `E42_SCORE_KEYS`, `OUT`). In a new file each must be qualified with `R50.` or defined anew, which point 2 does not list as a change. Three of them cannot be taken unchanged: `build` (the new one, with the arguments N and the choice), `receipt_path` (the new name), and `provenance` (`R50.provenance` writes `runner_sha256` of `exp50_code.py` and its own `sources_sha256`). Further, the receipt already has the entry `N=layer.N`; point 2 calls `N` a "new entry", and adding `N=N` a second time to the call `dict(...)` is a syntax error. Point 1 also does not give the signature of the new `build`.

**Evidence.** `exp50_code.py`, lines 319 to 427 and 431 to 438.

**Replacement text.** Point 1: "... is built by `build(N, choice, seed)`, a copy of the text of `exp50_code.build` for the setting `C46-one` in which the one line ... is replaced by ...". Point 2, first sentence: "The run is a copy of the text of `exp50_code.run` with these changes and no other, apart from calling the functions and constants of `exp50_code` through `R50`: `experiment="E62"`; the new entry `choice` (the entry `N` exists already, as `layer.N`); `build` and `receipt_path` of this file; a function `provenance` of this file that is `R50.provenance`'s text with `exp62_connections.py` as the runner and in `sources_sha256`; ..."

### Defect 6. The record of E42 is quoted for three seeds where the figures are those of one seed, and the run nearest to the bridge is left out

**Place.** "What the record says", the paragraph after the table.

**What is wrong.** (a) "E42 found on three seeds that N = 10 from the neighbourhood needs 2.36 times the updates of the dense network under the sweep and 1.70 times under back-propagation." In the memo's table of section 28.3, 2.36 and 1.70 are the ratios of seed 73129 alone (13,312 / 5,632 and 8,704 / 5,120). On the three seeds the ratios are 2.36, 2.68 and 2.33 under the sweep and 1.70, 1.69 and 2.12 under back-propagation; the memo's own sentence is "2.3 to 2.7 times under the sweep and 1.7 to 2.1 times under back-propagation". The threshold is 80 on A, which the sentence does not say. (b) The paragraph omits E42's `W10`: ten connections drawn from all 255 other cells (replaced during the run, under the sweep), which needed 1.26 times the updates of `A10` (the same from the neighbourhood) on seed 73129 and 0.84 and 1.02 times on the other two seeds. This is the only measured comparison of "from all cells" with "from the neighbourhood", it bears directly on the bridge run and on prediction P62.7, and it shows that one draw against another can differ by about the brief's factor of 1.25.

**Evidence.** Line C of the script; memo, section 28.3 and the table of E42's second stage.

**Replacement text.** "With a dense input and a learned code, E42 found that N = 10 from the neighbourhood needs, to 80 on A, 2.36 times the updates of the dense network under the sweep and 1.70 times under back-propagation on seed 73129 (2.33 to 2.68 and 1.69 to 2.12 on three seeds). E42 ran N = 5, N = 20 and N = 10 from all cells (`W10`) only with connections that change during the run and only under the sweep; `W10` needed 1.26, 0.84 and 1.02 times the updates of the same network from the neighbourhood on the three seeds. No run with this input exists at any other N."

### Defect 7. "The reference" is used in a sense that conflicts with decision 31 and with the chain of steps

**Place.** Terms (the exact gradient's "arm is a reference") and "How the result is stated" ("For each rule, *the reference* is its run with N = 255").

**What is wrong.** Decision 31 fixes "the reference" as back-propagation over the whole item on the dense network, and the chain's term "the dense network" is E26's network, with a learned code and every cell reading the whole byte. The brief uses "a reference" for the exact-gradient arm and "the reference" for each rule's run with N = 255 on the limited input, a network with no run on record. A reader of the report will take "as fast as the reference" under the sweep to mean "as fast as back-propagation on the dense network". The choice of N = 255 with the same limited input as the comparison is sound (it changes only the number of connections); only the name is misleading.

**Evidence.** Decision 31; `STEPS_FROM_THE_PRINCIPLE_20261118.md`, section 2.

**Replacement text.** In "How the result is stated": "For each rule, *the full run* is its run with N = 255, in which every cell reads every other and the input is the same limited code; it is not the dense network of decision 31 (E26's network, with a learned code), and it is never called the reference. An N is *as fast as the full run* if ...". Replace "the reference" by "the full run" in the rest of that section and in P62.9; in Terms keep "its arm is a reference" only as "its arm is a comparison and is never called a candidate".

### Defect 8. "What it cannot show" omits three limits that the design imposes

**Place.** "What it cannot show", and the per-run figure "the CPU-seconds of the updates up to the row at which 80 on R is reached".

**What is wrong.** (a) In this network every cell reads one value of every used byte, and the readout is dense. A cell therefore receives the whole input and, under both rules, an error directly at every answer position without any recurrent connection; N governs only how the cells combine what they hold over positions. The grid says nothing about how many connections a cell needs when input or error must reach it through other cells (the setting of step 4 and of E53). The sister brief E61 states the corresponding consequence for its own variable; E62 does not. (b) Each N is one draw, the draws for different N are not nested (section 2), and N = 10 from all cells shares 112 of 2,560 connections with the record's network; E42's `W10` against `A10` varied from 0.84 to 1.26 over three draws. A difference between neighbouring N, or between the bridge and the record, can therefore be a difference between draws. (c) The masked recurrent map is computed as a dense product (`linear(h, W * mask, b)` in `L42.rec`, and `MaskedSign`'s backward forms `g @ (sign(W * mask) * scale)`), so CPU time hardly depends on N. The CPU-seconds to 80 on R are thus updates times a constant, not the computation of a network with N connections, and the comparison at equal computation (requirement B1) is not made. The count that does depend on N is the row's `multiply_adds` (weights taking part × 64 × updates).

**Evidence.** `e50_sparse_code.forward`; `e42_cells.rec` and `MaskedSign`; the memo's table of E42's second stage; section 2 above.

**Replacement text.** Add to "What it cannot show": "- How many connections a cell needs when its input or its error must come through other cells: here every cell reads one value of every used byte and the readout reads every cell, so that a cell receives input and error directly; N governs only how cells combine what they hold over positions. - Differences smaller than those between draws: each N is one draw of connections, the draws for different N are not nested, and N = 10 from all cells shares 112 of 2,560 connections with the record's network. - Computation: the masked map is computed as a dense product, so CPU time does not fall with N; the report compares the networks by updates and by the multiply-adds of the receipt's rows (`multiply_adds`), and states no saving of computation from CPU time." In "How the result is stated", replace "the CPU-seconds of the updates up to the row at which 80 on R is reached" by "the multiply-adds (`multiply_adds` of the row) and the CPU-seconds of the updates up to the row at which 80 on R is reached, with the statement that CPU time does not depend on N in this implementation".

### Defect 9. Several predictions and the fixed sentence on the smallest N do not say which runs they cover

**Place.** Predictions P62.2, P62.4, P62.6 and P62.8; the first fixed sentence.

**What is wrong.** (a) "Every N from 4 to 255" (P62.2) and "from 16 to 255" (P62.4) do not say whether N = 10 from all cells, which lies in the first range, is included. "Every N at which both rules reach 80 on R" (P62.8) does not say whether the bridge run and the record count. (b) P62.6 does not say how it is scored if a run with N = 255 does not reach 80 on R. (c) "The smallest N that was as fast as" the full run may be N = 10 from all cells, which is not a power of two, and may be followed by a larger N that is not as fast; the sentence would then suggest a threshold that the data do not show. (d) No sentence is fixed for the case, which the brief foresees, in which the full run of a rule does not reach 80 on R.

**Replacement text.** Add before the table of predictions: "In P62.2, P62.4 and P62.8, 'every N' means the eight runs with N = 2, 4, ..., 128 and 255 from all cells within the stated range; the run with N = 10 from all cells and the record are not included. A run that does not reach 80 on R within 49,152 updates counts as not fewer (P62.6) and as outside the band (P62.7)." In the first sentence: "...; the smallest N of the grid of powers of two that was as fast as the network in which every cell reads every other was N*, and every larger N of the grid was [also / not all] as fast." After "no N is called as fast": "and the sentence is: 'On seed 73129 [the rule] did not reach 80 on R within 49,152 updates with every cell reading every other; its levels of R at update 49,152 were, for N = 2, 4, …: …'."

### Defect 10. Check 5 has no criterion of passing and no rule for the forecast of the N that it does not measure

**Place.** Check 5 and P62.1.

**What is wrong.** Every check writes an entry `pass`, and P62.1 is scored on it, but check 5 states no condition under which it passes; the limit of 44 CPU-hours is a condition for asking, not for failing. It also measures N = 2, 32 and 255 and does not say how the forecast for the other six settings is formed, nor how a scored row is timed (how many rows).

**Replacement text.** "5. **Cost.** CPU-seconds an update for N = 2, 32 and 255 under each rule (64 updates each, after 8 that are not counted) and CPU-seconds a scored row (the mean of two rows); the forecast for the eighteen runs is the sum, over the runs, of 49,152 times the cost an update and 193 times the cost a row, each taken, for N = 2, 32 and 255, from the measurement at that N, and for every other N from the larger of the measurements at N = 2 and N = 32 under the same rule. The check passes if every timing is finite and every scored row's assertions hold. If the forecast exceeds 44 CPU-hours, the worker asks before starting."

### Defect 11. The second fixed sentence uses brackets in a sense that the Terms reserve for another figure

**Place.** "How the result is stated", second sentence: "against 15,360 [34,304] with 10 connections from the neighbourhood".

**What is wrong.** Terms define a bracketed figure as the first single row at or above the threshold. Here the bracketed 34,304 is the sweep's three-row figure. A worker may copy the sentence as it stands for each rule.

**Replacement text.** "With 10 connections from all cells [the rule] reached 80 on R after u updates, against 15,360 (exact gradient) or 34,304 (sweep), whichever is the rule's, with 10 connections from the neighbourhood."

### Defect 12. The brief departs from step 2 of the chain in three places without saying so

**Place.** The head and "The question".

**What is wrong.** Step 2 of `STEPS_FROM_THE_PRINCIPLE_20261118.md` (a) names the job E61, together with step 3; (b) asks that "the run with N = 10 from the neighbourhood is repeated once as the bridge to E42", where the brief does not repeat it (check 1 reproduces its first 512 updates) and adds N = 10 from all cells; (c) defines updates to 80 as "the first scored row from which the score stays at or above 80", where the brief uses the first of three consecutive rows. Each choice of the brief is defensible (the record's run exists and check 1 shows that the runner reproduces it; the three-row rule is E50's and E61's), but a reader comparing the two documents will find them inconsistent.

**Replacement text.** Add to the head: "It departs from step 2 in three points, each for a stated reason: the job is E62 and not part of E61; the record's run with N = 10 from the neighbourhood is not repeated, because check 1 shows that the runner reproduces it, and the bridge is instead N = 10 from all cells; updates to a threshold are counted by E50's rule of three consecutive scored rows, not by the chain's first row from which the score stays above."

### Defect 13. The description of the sweep's scale at N = 255 suggests a difference that does not exist, and the added row entries repeat existing ones

**Place.** The runner, point 1 ("under the sweep its scale is then the mean magnitude of the whole matrix; for every other N it is the mean magnitude inside the mask") and point 3 ("At every scored row it records in addition the mean magnitude of the kept recurrent weights and the sweep's scale for the recurrent map").

**What is wrong.** At N = 255 the mask holds every entry, so the mean magnitude of the whole matrix (`SignLinear`'s `weight.abs().mean()`) is the mean inside the mask; the two branches compute the same quantity, and the forward values are the same. The sentence is not false, but it invites a worker to look for a difference. The row's `kept_mean_magnitude["rnn.weight_hh_l0"]` (`R49.magnitudes`) already is the mean magnitude of the kept recurrent weights, and it equals `layer.sign_scale(m)` for every N up to the rounding of a sum taken in another order; the two additions duplicate it.

**Replacement text.** Point 1: "... and the recurrent map is crossed by E26's own call (`exp3_state.lin`); under the sweep its scale is the mean magnitude of the weights inside the mask for every N, which at N = 255 is the whole matrix." Point 3: "**At every scored row** the row's existing entry `kept_mean_magnitude["rnn.weight_hh_l0"]` is the mean magnitude of the kept recurrent weights and, under the sweep, the scale of the recurrent map; nothing is added."

### Defect 14. Smaller matters of language and form

1. "It is step 2 of `STEPS_FROM_THE_PRINCIPLE_20261118.md`": the file is on the reviewer's branch only; write `$V:$R/COORDINATION_20261117/STEPS_FROM_THE_PRINCIPLE_20261118.md`.
2. "as E57's check 8 does": `exp57_scale.py` is not on the default branch, which the worker is told is the source of every file; the clause gives the expression in full and should drop the reference or name the branch.
3. "Level of evidence E2" is not defined in the brief, and "E2" is also the name of an endpoint class in the repository's rules; add "(one seed, no confirmation)" or the reference to the definition.
4. "The 48 cells around it on a grid of 16 by 16": add "(the 7 by 7 square centred on the cell, the grid's edges joined)".
5. `E62_BESIDE`: say what it holds and that the worker sets it at launch (for example the number of other runs in progress).
6. `--ckpt 0 writes none`: the copied condition `u % ckpt_every == 0` divides by zero at 0; say "with 0, no saved state is written (the condition is not evaluated)".
7. "The stream (`E.batches(tp, seed, E.KMAX["M"])`": `tp` is not defined in the brief; add "with `tp = [(t, E.split(t, .5, seed)[0]) for t in E.DEV]`".
8. Point 5 commits three weight files a run with `git add -f`; at 1,124,113 parameters of the module and the code table the eighteen runs add about 270 MB. State the size so that the worker can check the free disk before the runs.
9. With N = 255 the argument `wide=True` has no effect (`draws` takes the branch `N == U - 1` before it looks at the neighbourhood); saying so prevents a search for a difference.
10. The runs' order puts the full runs (N = 255), on which every reading of "as fast" depends, in the second group. If the budget or the host stops the job early, the first group cannot be read. Consider moving N = 255 into the first group (N = 8, 32 and 255 first).

## 4. What was checked and found right

- **The network of point 1.** Replacing `layer = L42.Layer(m, seed, N=10)` in the text of `exp50_code.build` by `L42.Layer(m, seed, N=N, wide=(choice == "all"))` changes only the recurrent mask and the scale of the kept recurrent weights at the start. `L42.Layer` uses no torch generator, and the input map is scaled after it, so the input map, `lm`, `l2`, `l3` and the code table are those of the record. With `N = 10` and `near` the line is the record's (the digest of the draw is the one `e42_cells.py` asserts, fde78220c57da3d4). With N = 255, `draws` takes the branch `N == U - 1`, `self.full` is True, nothing is scaled (the factor would be 1 in any case) and `rec` calls `exp3_state.lin` with `arch26.Mat`, E26's own crossing.
- **The functions used unchanged.** `R50.fwd`, `R50.install`, `R50.step`, `R50.counts`, `R50.fixed` and `R50.cosines` use the layer only through `layer.mask3()`, `layer.full`, `layer.K` and `sign_scale`, all of which are defined for every N and either choice. At a scored row, `R42.check_limit` checks the neighbourhood with `layer.wide` and skips it when `layer.full`; it counts `(mask & ~eye).sum(1) == layer.N`, which holds at N = 255. `R49.check_masks`, `R49.kept`, `R49.magnitudes`, `R49.readers`, `R42.opt_record`, `knows_figure`, `coverage` and `input_size` contain no use of N = 10 or of the neighbourhood. None of them fails or gives a wrong figure with N other than 10, with `wide=True` or with N = 255.
- **The count of weights.** From `exp50_code.counts`: input map 35,328 + `lm` 131,072 + `l2` 262,144 + `l3` 131,584 = 560,128, and the cells' kept weights are `mask3().sum()` = 768 (N + 1); 568,576 at N = 10 is the record's figure, and the formula gives 756,736 at N = 255 (line D).
- **The files a run writes.** With 49,152 updates `mid` is true in `exp50_code.run`, which writes the weights and the answers at update 24,576 beside the starting and final weights, the answers and the receipt; point 5's list is right. `*.pt` and `*.ckpt` are ignored by `.gitignore` on the default branch, hence `git add -f` for the weights.
- **The record's figures.** Every entry of the brief's table was recomputed from the two receipts with the brief's definitions (line C): 15,360 [13,824] and 34,304 [34,304] on R; 14,080 [12,288] and 35,584 [33,536] on A; levels 88.5, 88.9, 94.9 and 63.6, 65.5, 84.6 at 24,576; 93.8, 94.2, 98.3 and 85.9, 87.2, 94.0 at 49,152; 1.90 and 2.18 CPU-hours (updates and scoring); and the ratio 34,304 / 15,360 = 2.23.
- **The cost limit.** At the record's cost a run (2.184 CPU-hours under the sweep, 1.903 under the exact gradient), the eighteen runs need 36.8 CPU-hours; 44 is 1.20 times that. Since CPU time hardly depends on N (defect 8), the forecast is sound; the margin is small if three concurrent processes slow each other more than the record's two did (`processes_beside` 1 in both receipts).
- **The design.** The grid answers the owner's instruction (powers of two from 2 to 128, the effect read) and step 2's question, under both rules, with the full run as the upper end. Drawing from all cells is required for N above 48 and keeps the size of a neighbourhood out of the design; one bridge run at N = 10 measures the choice once, within the limits of one draw (defect 8). The brief states honestly that the start's scale changes with N and lists it among what the result cannot show. The reference network with the same limited input and every connection is the right comparison for "how many connections", provided it is not called the reference of decision 31 (defect 7).
- **Saving and resuming,** apart from defects 1 and 2, follows `exp52_resume.py`: the saved state's contents, the atomic write, saving at a scored row after the receipt, redrawing the stream into a new digest and comparing it with the last row's `data_digest`, deleting the saved state at the end, and `--die-after` with `os._exit(9)`. Saving after the receipt is written is consistent with `exp50_code.run`, which writes the receipt at every multiple of 1,024 updates, and every multiple of 4,096 is one.
- **The environment of the child processes.** The clause removes every variable beginning with `E11_`, `E16_` or `EXP45_`, which covers what `exp20_reference` sets at import (`E16_ARCH`, `E16_BOOKDECAY`, `E16_COMPGAP`) and refuses, and sets the working folder; it meets decision 37 and agrees with `exp50_code.check1` and E61's check 7.
- **Checks 1 and 4** can be carried out as written. Check 1 compares a dry run with `10 near` with the record in the start's tensors, the losses, the counts and the batch digest at three rows, which catches any change of the input, the stream, the optimizer or the update. Check 4 is the assertions of `exp50_code.run`, which hold for every setting (above).
- **The factor 1.25** and the definitions of *as fast*, *slower* and *not reached* can be scored from the receipts without a further decision, once defect 9 is mended.

## 5. What could not be checked

- No run, no forward pass and no gradient: torch is not installed here and was not installed. The statements that the functions of `exp50_code` work with every N rest on reading the source, not on execution. In particular, whether the first-batch loss and the forward values at N = 255 through `exp3_state.lin` equal those of the masked branch with a full mask was not computed; they are the same by construction.
- The CPU cost at N = 2, 32 and 255 on the worker's host, and the effect of three concurrent processes.
- Whether `torch.save` writes a byte-identical starting file on a second write (relevant only if defect 2 is not mended).
- The coordinator's reading of whether drawing from all cells agrees with Mr. Xu's statement of 16 November that "a neuron adjusts only within a pre-defined neighbourhood"; the brief and step 2 read a fixed set of N sources as such a neighbourhood. This is a question for the coordinator, not a defect of the brief.
