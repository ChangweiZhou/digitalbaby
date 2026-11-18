# Audit of the E47 brief and its specification file at 7247682a (16 November 2026)

From the outside reviewer. Read-only: nothing was trained, torch is not installed in the reviewer's container, and no code from another branch was executed. The brief is `OPUS_TASK_E47_LOCALITY_AT_THE_INPUT_AND_THE_OUTPUT_20261116.md`, revision 1. The specification file is `PROJECT_DOCUMENTS/COORD_CHECKS_20261116/e47_local_spec.py`, with the coordinator's output `OUTPUT_E47_SPEC_MAC.txt` beside it; the coordinator's account is review section 9.150. They implement section 26.3 of the reviewer's memo. The job of memo section 27 (connections gated by a code) is E48 and is not part of this audit.

**Verdict.** The network is the one the memo asked for, and the specification file is sound as far as it can be read and checked without torch: a second reader written from the brief's words gives the states and the logits of E42's saved `F10` and `P8` networks at 256 cells, the draws and the counts of the brief are reproduced, and the sweep written by index is the masked rule. Twelve changes are required before release. None alters the layer. The most important are these: check 2 asks for an agreement that two correct programs cannot give, against a dry run that exists under one of the two rules only; the brief's statement that every far cell can be reached and can answer within an item holds for the arm `one` and not for the trial's arm `place`, and at 4,096 cells under `place` nearly all cells of period 16 can take part in no training item; holding the far cells at zero does not separate a constant from information; and inside the patch every cell still reads the whole code of the byte, so the user's requirement is met outside the patch only, which the brief must say.

**Terms.**

- An *interface cell* is one of the 256 cells of the patch; a *far cell* is any other cell. A *position* is one byte of an item; the layer takes one step a position.
- The arms `one`, `place` and `scatter` are the brief's: every cell of period 1; periods 2, 4, 8 and 16 in four equal groups of far cells by distance; the same periods placed at random.
- A far cell *can take part* in an answer when, by the connections, periods and phases alone, some state of the cell depends on a byte of the prompt and can still change a byte of the answer. It is a property of the structure, computed without weights, and an upper limit on use.
- *A* and *R* are E42's scores on untaught pairs and on untaught reduction pairs. A is taken on items of 12 letters and R on items without letters (`git show 7fbdbdc2:$M/exp42_cells.py`, `panels`). Training items have 0 to 12 letters with equal chance, and 10 + k positions for k letters.
- The *sweep* is the sign rule; *backpropagation* is the exact gradient.
- The *reader* is `e47_local_reader.py` beside this audit: the layer and its draws written from the brief's words in numpy, computed group by group of updated cells, which is not how the specification file computes it.

Shell names: `D`, `M` and `K` as in the brief; `R=PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109`; `C=$R/e47_brief_checks.py` on `claude/vibrant-cerf-sje726`. `python3 -I $C <letter>` runs one part (A to K); the whole output is `$R/OUTPUT_E47_BRIEF_CHECKS.txt`.

## 1. The brief against the memo's requirements (section 26.3)

| Item of 26.3 | In the brief | Status |
|---|---|---|
| 1. The interface is a boundary: a fixed number of cells read the byte, a fixed number are read by the readout | One patch of 256 cells does both, at every size | Met as worded. Inside the patch nothing is limited (section 2) |
| 2. Every cell reads at most N cells of its neighbourhood and its own previous value; fixed connections | N = 10 of 48, own value, E42's replacement rule not used | Met |
| 3. One cell a step; reading and read cells the same or neighbours; a variant with extra steps may be added | The same cells; the variant is left out | Met; the omission is allowed (section 3) |
| 4. Periods by place as one variant, scattered as the other | `place`, `scatter`, and `one` beside them; interface cells have period 1 | Met at 1,024 cells; at 4,096 the rule leaves most outer cells unable to take part (section 4) |
| 5. Three numbers of cells at one interface, compared at equal computation and equal weights, beside E42's `G` and `P8` | Three sizes; comparison in multiply-adds and weights against `F10` and `G` | Partly: `P8` is not named; the column "with periods" leaves out part of the arithmetic (section 8) |
| 6. Arithmetic by index; at 256 cells the forward and both rules' gradients equal E42's | `IndexSign`; self-check 1 | Met on the committed output; forward and the rule's algebra confirmed here (section 5) |
| 7. E42's measures, multiply-adds, and scores with the far cells held at zero | All three | Met; the diagnostic needs a second form (section 7) |
| 8. A full-length run on one seed before the roster | The trial | Met; its composition is discussed in section 7 |

## 2. Point 1: is one fixed 16 x 16 patch for both reading and being read the boundary meant?

**It is the boundary the memo meant, and it is acceptable as the first step.** Only the patch reads the byte and only the patch is read, at every size, so the interface costs the same 524,288 multiply-adds a position whether the network has 256 cells or 4,096 (`python3 -I $C C`). At 256 cells every cell is in the patch and the network is E42's `F10`, which anchors the job to a run already on file. Using one patch for both keeps the answer at `=` free of travel, which is the reason the brief gives for leaving the extra steps out.

**It does not meet the user's requirement inside the patch, and the brief does not say so.** The user's statement is that a cell cannot see the whole input. Each of the 256 interface cells reads all 512 values of the byte's code, and each of the 512 readout units reads all 256 interface cells. Those products are 56% of a position's multiply-adds at 256 cells and 50% at 4,096. The patch fixes the cost of the interface; it does not limit what an interface cell reads. The brief lists the size of the interface under what the job cannot show, and does not list this.

**The networks on file rest on the dense interface** (`python3 -I $C J`). In E42's trained `F10` of seed 73129, the input values that hold 90% of a cell's squared input weight number 237 of 512 at the median under the sweep and 279 under backpropagation; the cells that hold 90% of a readout unit's squared weight number 110 and 97 of 256. With all but the largest weights set to zero and nothing retrained, the score on 300 length-test items of 12 letters is:

| Kept | Sweep (91.7 as trained) | Backpropagation (91.3 as trained) |
|---|---|---|
| 256 input values a cell | 83.0 | 72.3 |
| 128 input values a cell | 15.3 | 10.0 |
| 128 cells a readout unit | 84.3 | 88.0 |
| 64 cells a readout unit | 41.0 | 41.7 |

This shows that a limited interface cannot be had by removing weights from a network trained with a dense one. It does not show whether a network trained with a limited interface learns; that has not been tried.

**Judgement.** A limited interface should not be added to E47: it would change two things at once and would end the equality with `F10` on which checks 1 and 2 rest. It should be named in the brief as the next job (required change 10) and can be drafted now. At 256 cells it needs no arithmetic by index: E42's runner with two fixed masks, one on the input map (each cell reads a stated number of the 512 values) and one on `lm` (each readout unit reads a stated number of cells), at about 0.1 CPU-second an update on the cloud host, which is under an hour a run (recommendation 1). A smaller patch is a later question, as the brief says.

## 3. Point 2: the variant with extra steps was dropped; is that acceptable for a first job?

**Yes.** The memo allowed the variant and did not require it, and with one patch for reading and being read the answer needs no travel.

**What the time structure allows** (`python3 -I $C D`). The first digit of the first operand stands 5 + k positions before the first byte of the answer, and the last digit of the second operand stands one position before it. Under `one` a round trip to a cell h connections from the interface takes 2h positions at the least.

- No far cell can use the last digit of the second operand for the first byte of the answer, at any size: the count of far cells that can take part between those two positions is 0, against 256 interface cells. What the far cells can add to that byte depends on earlier bytes only.
- At 1,024 cells under `one` every far cell can take part in every item for some pair of bytes; for the first digit of the first operand and the first answer byte the share is 0.65 without letters and 1.00 from 4 letters.
- At 4,096 cells under `one` the share that can take part for any pair of bytes is 0.32 without letters, 0.59 at 4 letters, 0.92 at 8 and 1.00 at 12.

**What the largest size can and cannot show.** The two scores on which use is read differ in length. A is scored at 12 letters, where every far cell of the arm `one` can take part at 4,096 cells, so a reading of *used* is possible there on A. R is scored without letters, where two thirds of the far cells of that arm cannot take part, so a small fall of R when the far cells are held says little. A reading of *not used* at that size cannot be told apart from "too far to be used in the items that train it", because two thirds of the far cells cannot take part in an item without letters and two fifths cannot in an item of 4 letters. The variant with extra steps is the measurement that separates the two, and is the follow-up if that reading occurs (recommendation 4).

**A statement of the brief to correct.** "A far cell can be reached and can answer back within an item at 1,024 cells in every item" holds for `one`. For the trial's arm, `place`, the share is 0.63 without letters, which is the length at which R is scored, and 0.96 at 12 letters, the length at which A is scored (section 4, required change 5).

## 4. Point 3: periods by place in four equal groups at 2, 4, 8 and 16; is the new rule right?

**The rule as the file has it** (`python3 -I $C A`). Interface cells have period 1. The far cells are ordered by distance from the patch and then by index, and cut into four equal groups with periods 2, 4, 8 and 16: 192 cells each at 1,024 cells, 960 each at 4,096. The mean of 1/period is 0.426 and 0.282. The mean difference in log2 period across a connection is 0.59 and 1.13 (`place`, `scatter`) at 1,024 cells and 0.26 and 1.22 at 4,096, as the brief says; over three seeds `scatter` gives 1.11 to 1.13 and 1.21 to 1.22. The earlier rule (groups of 256, doubling, capped at 16) would have put 3,072 of 4,096 cells at period 16, as section 9.150 says; the change is right.

**At 1,024 cells the rule is acceptable.** The share of far cells that can take part, for any byte of the prompt and any byte of the answer (`python3 -I $C D`, seed 73129; the other two seeds are within 0.03 for `place`):

| Cells, arm | 0 letters | 4 | 8 | 12 | lowest group at 12 letters |
|---|---|---|---|---|---|
| 1,024 `one` | 1.00 | 1.00 | 1.00 | 1.00 | |
| 1,024 `place` | 0.63 | 0.80 | 0.88 | 0.96 | period 16: 0.82 |
| 1,024 `scatter` | 0.63 | 0.83 | 0.93 | 0.98 | period 16: 0.92 |
| 4,096 `one` | 0.32 | 0.59 | 0.92 | 1.00 | |
| 4,096 `place` | 0.24 | 0.37 | 0.50 | 0.61 | period 8: 0.41; period 16: 0.02 |
| 4,096 `scatter` | 0.12 | 0.23 | 0.37 | 0.54 | period 16: 0.36 |

Under `place` at 1,024 cells, then, 0.96 of the far cells can take part in the items on which A is scored and 0.63 in those on which R is scored, and the cells that cannot are of the two longest periods: in an item without letters 0.41 of the cells of period 8 and 0.11 of those of period 16 can take part.

**At 4,096 cells the rule is not right for this task.** The cells of period 8 lie 16 to 20 grid steps from the patch and those of period 16 lie 21 to 24. Of the 960 cells of period 16, 2% can take part in an item of 12 letters, and fewer in shorter items; of those of period 8, 41%. About 1,500 of the 3,840 far cells can therefore take part in no training item. The same is seen in the gradient (`python3 -I $C F`): 27% of the far cells' recurrent weights can receive no gradient from any training item, and 571 far cells have no such weight at all. With decay 0.25 a weight that receives no gradient is multiplied by 0.99975 at every update and ends the dose at 0.2% of its start. A shorter longest period does not cure it (`python3 -I $C K`): with periods 2, 4 and 8 the share at 12 letters is 0.71, with 2 and 4 it is 0.83, with period 2 alone 0.96. At this size and item length the outer cells are too far, whatever the periods. This must be stated before the run (required change 5), and the two `place` runs at 4,096 cells are the first to cut (section 8).

**Self-check 4's figure is confirmed and qualifies the coordinator's account.** The committed output gives 0.90 for the share of far cells' recurrent weights with a non-zero gradient at 1,024 cells under `place`. The structure alone gives 0.90 (`python3 -I $C F`); the weights without gradient belong mostly to cells of period 16, of whose weights 0.64 can receive one. Section 9.150 says that "the far cells' weights all receive gradient under both rules"; the figure is nine in ten (required change 12).

**A cell of period 16 is not always updated.** The brief says such a cell is updated once or twice in an item. Over training items and phases it is not updated at all in 10% (38% of the items without letters), once in 80% and twice in 10% (`python3 -I $C D`).

**E42's first seed on period 16** (`python3 -I $C H`). Under the sweep, E42's `P16` ended at 82.8 on the length test at 40 letters against 96.0 for `P8` and 89.1 for `F10`. In E42 the cells of long period were among the 256 that read the byte and are read; here they are far cells, and the interface keeps period 1. The caution applies to the extent that the far cells are used, and the brief's reading of the length test at 15 points will show it.

**`scatter` is a fair control.** It has the same number of cells at each period, the same weights (971,665) and the same counted computation as `place`, the same start, and the interface at period 1 in both. The two differ in where the periods lie and, as a consequence, in the phases drawn, since a phase is drawn below the cell's period.

## 5. Point 4: is `IndexSign` the masked sign rule correctly written by index?

**Yes, by reading and by a check of the algebra.** The forward is the sum over a cell's 11 sources. The backward sends the error across each connection multiplied by the sign of its weight and by one scale, and gathers it at the source cell; the weight gradient is the exact one; a weight that is exactly zero sends nothing. The scale is the mean magnitude over all entries of the tensor, which equals E42's mean inside the mask because every entry is a connection and no pair of cells occurs twice. Written in numpy from the brief's sentence and compared with the masked matrix form on random values, the two agree to 1.3e-15 in the error sent back and exactly in the weight gradient, at 256 and at 1,024 cells (`python3 -I $C I`).

**The committed figures support the brief's claim.** Self-check 1 prints, for `F10` and for `P8` under both rules, logits that differ from E42's by 2.4e-7 and gradients that differ by at most 3.5e-7 of the tensor's largest entry. The brief asks for 1e-6 and 1e-5. I could not run the file; the forward half is confirmed by the reader, which gives the E42 reader's states to 8.0e-7 and its logits to 7.6e-6, with the same largest output at every position, on E42's saved `F10` and `P8` of seed 73129 under both rules, at 12 and at 40 letters (`python3 -I $C B`).

**One consequence of a single scale at the larger sizes.** The scale is a mean over interface cells and far cells together. If the far cells' weights shrink, through decay where the gradient is absent or weak, the error crossing every connection shrinks with them, the interface's included. At 256 cells this cannot happen. The trial should report the scale and the two mean magnitudes at each scored row (required change 7). A scale taken over a cell's own 11 weights would be a quantity the cell holds, which a mean over all cells is not; it would end the equality with E42 and belongs to a later job (recommendation 5).

## 6. Point 5: is the starting distribution for the cells E26's network does not have acceptable?

**Yes.** The recurrent weights are uniform in ±1/√11, which is the range of E42's start (E26's ±1/16 multiplied by √(256/11)); the biases are uniform in ±1/16, which is the range of the present network's own. The generator depends on the seed and the side only, so the three arms of one size share one start, which is what a comparison of arms needs.

**Four things the words leave open, each a required change.**

1. *Which constructor at 16 a side.* The brief says the 256-cell network is run "from E42's own start". The file's default constructor draws a new start at every side, 16 included; E42's start is obtained only with `from_e42` and the `structure=` argument, as self-check 1 does. The brief must name them (required change 1).
2. *The interface cells at 32 and 64 a side.* Their recurrent weights and biases are newly drawn as well; nothing of E26's recurrent start is used above 256 cells. The brief implies this and should say it (required change 1).
3. *The optimizer's groups.* E26 decays everything but the byte table, and E42's added parameters joined the decayed group (`git show 7fbdbdc2:$M/exp42_cells.py`, `make_opt`). The brief does not say where `Wc`, `bc`, `bfar` and `a` go, nor what becomes of the module's `rnn.weight_hh_l0` and `rnn.bias_hh_l0`, which the layer does not use (required change 2).
4. *Reproduction.* The draws of the new cells come from torch's generator. A second implementation without torch cannot make them, so every run must save its starting weights and its structure (required change 11). The draws by `random.Random` are reproduced; digests are in section 9.

Two remarks that need no change: the string `e42|connections|{seed}` is used at every size, so a cell takes the same places of its sorted neighbourhood at 16 and at 32 a side; and a cell of period 1 still draws a phase, which advances the generator, as in E42.

## 7. Point 6: is the trial the right one to run first?

**The size, the length and the two rules are right; the arm is not sufficient alone.** Under `place` a fall in the scores when the far cells are held cannot be assigned to the boundary or to the periods, and a failure to learn cannot be assigned either. The arm `one` at the same size has no periods, every far cell can take part in every item, and its two runs are in the roster already. The trial should be four runs: `one` and `place` at 1,024 cells under both rules (required change 7). This moves two roster runs forward and adds none.

**Holding the far cells at zero does not separate a constant from information.** A far cell that no byte reaches still has a state, set by its biases and its neighbours, the same in every item. The interface cells that read it learn their weights with that value present. Setting it to zero then shifts those cells' inputs, and the scores can fall although the far cells carry nothing about the item. On untrained recurrent weights at 1,024 cells the far cells' states have a mean size of 0.030 to 0.037, of which 0.004 to 0.008 varies between items, and the logits move 1.8 to 2.8 times as far with the far cells at zero as with the far cells held at their mean over the items (`python3 -I $C E`). The reading *used* must rest on the second form (required change 6).

**The checks.** Checks 1, 3, 5 and 7 are sound. Three need rewording.

- *Check 2 cannot pass as written.* It asks that a dry run of 2,048 updates at 16 a side have the scores of E42's `F10` dry run "exactly" and its tensors to 1e-5, "under each rule". First, E42's branch holds a dry run of `F10` under the sweep only (`python3 -I $C H`). Second, the two programs are not the same arithmetic: self-check 1 shows them 2.4e-7 apart in the logits on the first batch. Under AdamW the first update of an entry is close to the learning rate times the sign of its gradient, whatever the gradient's size, so an entry whose gradient is as small as the difference between the two programs can be moved in opposite directions by them, by up to 0.001 each, which is up to two hundred times the tolerance, after one update. Over 2,048 updates the runs separate, and scores counted on 300 items will not be equal. Third, E42's check 8 compared the reviewer's reader with the worker's answers on one set of saved weights; it set no tolerance for two training runs. P47.1 gives 85% to a check that a correct runner would fail. Required change 1 replaces it with what can hold: equal gradients on the first batch against E42's runner, and the losses and scores reported side by side.
- *Check 4, first clause.* "Changing any input weight of the input map changes no far cell's state at the same position" is true at the first position only: a changed weight changes the interface cells at every position and the far cells one position later. The property meant is about a byte (required change 3).
- *Check 6.* E26's births have a full recurrent matrix. "Through the 256-cell layer with E42's start mapped" must mean every cell reading itself and the 255 others by index, which the default draws cannot give, since a neighbourhood has 48 cells (required change 4).

**What the trial's report must contain** is in required change 7: the scores with the far cells at zero and at their mean at every scored row; the cosines for interface cells and far cells; the scale and the mean magnitudes; the share of far cells' recurrent weights with a gradient; and the measured cost.

## 8. Counts and cost

**Weights and multiply-adds are as the brief states them** (`python3 -I $C C`): 935,953, 965,905 (971,665 with periods) and 1,085,713 (1,114,513) weights; 930,560, 955,904 and 1,057,280 multiply-adds a position with every cell computed at every step. Two figures of the table need correction (required change 8).

- *The present network's row.* On the basis of the other rows, which count products and no biases, the present network has 1,118,720 multiply-adds a position. The brief's 1,122,048 is 3,328 more.
- *The column "with periods".* The brief counts a cell of period p as 33 / p multiply-adds a position. The `step` variant gives every step of a period its own learned weight, so before its 33 products the cell must add up its 11 sources' values with those weights at every step, which is 11 more multiply-adds a position. The least count is 944,948 at 1,024 cells and 1,002,500 at 4,096, against the brief's 936,500 and 960,260. "Sixteen times the cells cost 3% more with periods" becomes 8%, against 14% with every cell at every step.

**Time, on the brief's own figures** (`python3 -I $C G`). A run of 24,576 updates takes 1.9 CPU-hours of updates at 1,024 cells under `one`, 2.7 under `place` or `scatter`, 8.2 at 4,096 cells under `one` and 10.8 under `place`, on the coordinator's machine and without scoring. The trial as drafted is 5.3 CPU-hours; with the two `one` runs, 9.0. The roster's eight further runs are 47 CPU-hours, of which the four at 4,096 cells are 38. That is above the brief's limit of 40, so by its own rule the 4,096-cell runs wait unless the worker's forward is faster.

**The cloud host is slower.** E42's files took 1.8 to 3.3 times as long for an update on the cloud host as for a forward and backward on the coordinator's machine (`G` 0.103 against 0.031, `F10` 0.105 against 0.035, `P8` 0.138 against 0.076, the last after the worker's speed-up). At those ratios the trial as drafted is 10 to 18 CPU-hours of updates, one run at 4,096 cells is 15 to 36, and the eight roster runs are 85 to 156. Scoring is extra: E42's `F10` took 5.9 CPU-seconds a scored row, and a row here is computed two or three times (plain, far cells at zero, far cells at their mean) on a slower layer, which by the same ratios adds about two CPU-hours to a trial run.

**Where the time goes.** With periods the file multiplies the stored inputs of every cell by coefficients for 16 steps at every position: 2 x 16 x 3 x 4,096 = 393,216 multiply-adds a position at 4,096 cells, against 135,168 for the cells' own products. A forward that computes only the cells updated at a position, as the reader does, removes most of this; check 3 covers it.

**What to cut, in order.** First the two `place` runs at 4,096 cells: by structure four in ten of their far cells can take part in no training item, and they are the two most expensive runs. Then the sweep's run at 4,096 cells under `one`, until backpropagation's run at that size has been read. The 1,024-cell runs are all worth their cost.

## 9. Figures for the worker and the coordinator to compare

The brief and the committed output state no digest. These are the reader's, for seed 73129 (`python3 -I $C A`; the other two seeds are in the output file): the first 16 hexadecimal digits of the SHA-256 of `json.dumps` of the list.

| Side | Connections (cells x 10, sorted) | Periods `place` | Periods `scatter` | Phases under `place` |
|---|---|---|---|---|
| 16 | fde78220c57da3d4 | 0cc7c65a16f482a9 | 0cc7c65a16f482a9 | ba1a044f61b89809 |
| 32 | f3361f3823d86f9d | 8394011ba0130f32 | 1f2074a6ca43ab00 | 624270ab87410e73 |
| 64 | a70f0c440580f1bb | 10ef2ba7a17c347c | 772686e61a77fb83 | 80295a954d239c28 |

At 16 a side the connections equal those of the E42 reader for all three seeds. Cells by connections from the interface: 228, 300 and 240 at 1,024 cells; 228, 300, 372, 444, 516, 588, 660 and 732 at 4,096.

## 10. The twelve required changes, as text to apply

**1. Check 2 and the start at 16 a side.** Replace check 2 with:

> 2. **At 16 a side the runner's layer is E42's `F10`.** The runner builds the 256-cell layer from E42's layer of the same seed with `from_e42` and `Local(..., structure=...)`; the default constructor draws a different start and is not used at 16 a side. From that start, on the first batch of seed 73129, the runner's logits equal those of E42's runner to 1e-6 and every gradient, mapped by index, to 1e-5 of the tensor's largest entry, under each rule. The losses of the first 16 updates of the two runners, and the scores of a dry run of 2,048 updates under the sweep beside E42's `e42_F10_S_dry2048_73129`, are reported side by side. Equality of the two runs is not required: the programs add in a different order, AdamW's first update follows the sign of the gradient, and E42 has no dry run of `F10` under backpropagation. A loss that differs by more than 1e-6 at the first update or 1e-3 at the sixteenth is investigated before the trial.

In "Start", replace the last sentence with: "At 16 a side the layer is built from E42's layer of the same seed (`from_e42`, then `Local(..., structure=...)`), so its start is E42's `F10` start exactly. At 32 and 64 a side every cell's recurrent weights and biases are newly drawn, the interface cells' included, and the three arms of one size and seed share one start."

**2. The optimizer.** Add to "Start": "`Wc`, `bc`, `bfar` and `a` are in E26's decayed group (decay 0.25), as E42's added parameters were; the byte table is not decayed. The module's `rnn.weight_hh_l0` and `rnn.bias_hh_l0` take no part: they receive no gradient and are not counted, saved or digested as weights of the network."

**3. Check 4.** Replace its first clause with: "changing the byte at one position changes no far cell's state at that position, and changes a far cell no sooner than its distance allows;".

**4. Check 6.** Replace with: "**Scores:** E42's check 12 on E26's six births, each loaded by index with `structure=`, every cell's sources being itself and the 255 others; the file's own draws are not used, since a neighbourhood has 48 cells."

**5. Time and periods.** Replace the bullets "Time" and "Periods" of the comparison with the record with:

> - **Time.** A connection passes a value one cell a step, and a cell of period p takes values in only at its own updates. From the connections, periods and phases alone (the reviewer's `e47_brief_checks.py`, parts D and F, seed 73129), the share of far cells that can take part in an answer, meaning that some state of the cell depends on a byte of the prompt and can still change a byte of the answer, is, for items of 0, 4, 8 and 12 letters: at 1,024 cells 1.00 throughout under `one`; 0.63, 0.80, 0.88, 0.96 under `place`; 0.63, 0.83, 0.93, 0.98 under `scatter`; at 4,096 cells 0.32, 0.59, 0.92, 1.00 under `one` and 0.24, 0.37, 0.50, 0.61 under `place`. A is scored at 12 letters and R without letters, so the two falls are read with these shares beside them. At 4,096 cells under `place`, 0.41 of the cells of period 8 and 0.02 of those of period 16 can take part at 12 letters, 571 far cells have no recurrent weight that can receive a gradient, and 27% of the far cells' recurrent weights receive none (10% at 1,024 cells); with decay 0.25 such a weight ends the dose at 0.2% of its start. The last digit of the second operand stands one position before the first byte of the answer, so no far cell can use it for that byte. These limits are stated before the run as reasons the far cells may go unused.
> - **Periods.** A cell of period 16 is not updated at all in 10% of training items (38% of those without letters), once in 80% and twice in 10%. The reviewer's count for E42 (a long period lets a cell carry an operand without learning to keep it, and never at 40 letters) applies here to the far cells; E42's `P16` ended at 82.8 at 40 letters on its first seed under the sweep, against 96.0 for `P8`.

In the paragraph after the table of arms, replace "so a period beyond 16 would leave a cell idle through whole items" with "and a period of 16 already leaves a cell idle through one training item in ten".

**6. A second diagnostic, and the reading that rests on it.** After the paragraph on `far_off`, add: "A second diagnostic holds every far cell, after each position, at its mean over the scored items at that position (each scored set has items of one length). It is an argument of the forward beside `far_off`; the runner computes the means from the plain forward on the same items." Replace the bullet "Use of the far cells" with:

> - **Use of the far cells**, at the end of the run, from the fall of A and of R under the two diagnostics; the first form that applies. *Used*: A or R falls by 15 points or more with the far cells held at their mean. *Not used*: A and R both fall by less than 5 with the far cells held at zero. *A constant is used*: A or R falls by 15 points or more at zero and both fall by less than 5 at the mean. Otherwise *partly*. With one seed, a description is that seed's.

P47.4, P47.5 and P47.8 then refer to this definition; their chances may be restated.

**7. The trial.** Replace its first sentence with: "One seed, 73129, at 1,024 cells, with periods `one` and with periods `place`, under the sweep and under backpropagation: four full runs of 24,576 updates, the two `one` runs moved forward from the roster, scored every 256." Add to what is recorded: "at every scored row, E42's scores plain, with the far cells at zero and with the far cells at their mean; the mean magnitude of the recurrent weights of the interface cells and of the far cells, and the sweep's scale; at the first batch and at the end, the cosines for interface cells and for far cells, and the share of the far cells' recurrent weights whose gradient is not zero; CPU-seconds an update and a scored row." In the roster, the further runs become six.

**8. The cost table and the roster's cost.** In the row of the present network replace 1,122,048 with 1,118,720. In the column "with periods" replace 936,500 with 944,948 and 960,260 with 1,002,500, and add under the table: "A cell of period p costs 33 / p multiply-adds a position for its products and 11 for adding up its sources' values with the step weights." Replace "or 3% with periods" with "or 8% with periods". Add to the paragraph on measured time: "On these figures the roster's runs at 4,096 cells take 38 CPU-hours of updates on my machine, and E42's files ran 1.8 to 3.3 times slower on the cloud host; check 7's forecast is made on the worker's host and includes scoring, three forwards a scored row."

**9. `P8` beside `F10` and `G`.** In "How the result is stated", replace the opening line with: "Against E42's `F10` of the same seed and rule (256 cells), against E42's `P8` for the arms with periods, and against E42's `G` (the full network):". P47.7 names `F10`; E42's `P8` reached 80 on R earlier than `F10` on this seed under both rules, so the prediction should say which of the two it is read against.

**10. What it cannot show.** Add the bullet: "**Locality inside the interface.** Each of the 256 interface cells reads all 512 values of the byte's code, and each readout unit reads all 256 interface cells: 524,288 of a position's multiply-adds at every size. The user's requirement is met for the cells outside the patch and not for those inside it. E42's trained `F10` rests on the dense interface (with the smaller half of each cell's input weights removed and nothing retrained, its score at 12 letters falls from 92 to 83 under the sweep and from 91 to 72 under backpropagation). Whether a network trained with a limited interface learns is the next job."

**11. What is saved.** Add to "Done": "Every run saves its starting and its final weights, each with the table of sources (cells x 11), the periods, the phases, `Wc`, `bc`, `bfar` and `a`; and, at the last scored row, the bytes of each answer given on the length-test panel, plain and under each diagnostic. The receipt names the side and the arm."

**12. Section 9.150.** Replace "the far cells' weights all receive gradient under both rules" with "nine in ten of the far cells' recurrent weights receive a gradient under both rules at 1,024 cells with periods by place".

## 11. Recommendations (none holds the release)

1. **Draft the job with a limited interface now.** At 256 cells, E42's runner with two fixed masks: each cell reads a stated number of the 512 input values, each readout unit a stated number of cells, the kept weights scaled at the start as E42 scaled the recurrent ones. Two or three settings under both rules cost a few CPU-hours and can run beside E47's trial. It is the direct test of the user's statement.
2. **Remove the two `place` runs at 4,096 cells from the planned roster,** for the reason in section 4, or replace them by one run with period 2 for every far cell, where 0.96 of the far cells can take part at 12 letters.
3. **Release the 4,096-cell run under `one` for backpropagation first,** and only when check 7 forecasts it at a stated number of CPU-hours; the sweep's run at that size after it has been read.
4. **If the far cells at 4,096 cells read *not used*, run the variant with extra steps** between `=` and the answer as the follow-up; it is the only measurement here that separates distance in time from a failure to learn.
5. **Name a scale for each cell as a later variant of the sweep.** A mean over a cell's own 11 weights is information the cell has; a mean over all cells is not.
6. **Write the sped-up forward so that it computes only the cells updated at a position.** The reader does so and agrees with the E42 reader at 256 cells; section 8 gives the arithmetic it saves.
7. **Report the far-cell diagnostics on the length test at 40 letters as well.** At 40 letters every far cell of every arm can take part at 1,024 cells, and 0.99 or more of them at 4,096 cells, so it is the one panel on which distance in time does not limit use.

## 12. Can E47 be released?

Yes, after the twelve changes. Changes 1 to 4, 8, 9, 11 and 12 are corrections of wording, figures and checks. Changes 5, 6, 7 and 10 alter what the brief states before the run, what it measures and what the trial contains; none alters the layer or its draws. If they are applied as written, no second audit is needed before release. Change 6 adds an argument to the forward; if it is added to the specification file, the commit should be named in the brief, and I will read the difference with the trial's report. A second audit is needed only if the design changes in another way, for example a different rule for periods at 4,096 cells or a limited interface inside this brief.

## 13. What I did not check

- **Anything that needs torch.** The specification file was read and not run. The equality of the gradients with E42's under both rules is the committed output's; the reader confirms the forward, and part I confirms the algebra of the rule, not the file's execution of it.
- **The starting draws of the new cells,** which come from torch's generator.
- **The CPU figures.** The costs in section 8 are arithmetic on the brief's and E42's recorded figures; the ratio between the two hosts mixes a forward and backward with a whole update and, for `P8`, a speed-up.
- **The runner,** which is not written. The structural shares in sections 3 and 4 are upper limits from the structure; what a trained network does within them is the trial's to show.
- **E26's 3,886 to 6,899 updates to 80% on untaught pairs,** which is the record's figure from earlier audits and was not recomputed here.
- **Seeds 93683 and 116089 beyond the draws.** The structural figures at 4,096 cells and the interface figures of section 2 are for seed 73129 only.
