# Verification of the trial of E47 (locality at the input and the output) and a recommendation on its roster (17 November 2026)

Reviewer's document. It concerns stage 0 of `OPUS_TASK_E47_LOCALITY_AT_THE_INPUT_AND_THE_OUTPUT_20261116.md`, revision 2 (cf96788d): four runs at 1,024 cells on seed 73129, reported by the worker in `MOD97_LOCAL_RULE_20261106/E47_TRIAL_REPORT.md` on `claude/e47-worker-stage-0-3d0ya6` at 578a877b. The worker has stopped, as the brief required, and no run of the roster has been started.

Everything below was obtained without training and without running any code of the worker's or of the default branch. Files were read with `git show`, one at a time. The forward computation is the reviewer's numpy reader `e47_local_reader.py`, written from the text of the brief for the audit of 16 November. All figures are produced by `e47_trial_verify.py` in this folder; the part of the script that produces a figure is named in brackets, and section 10 gives the commands. Every statement is for seed 73129 only.

## Terms

- **Interface cells** are the 256 cells of one 16 x 16 patch of a 32 x 32 grid with joined edges. Only they receive the code of the current byte, and only they are read by the readout. **Far cells** are the other 768 cells; they receive no input and are not read. Every cell reads its own previous value and ten cells within three rows and three columns of it.
- A far cell's **ring** is the number of connections between it and the interface: ring 1 holds the 228 far cells within three grid steps of the patch, which an interface cell can read directly; ring 2 the 300 cells four to six steps away; ring 3 the 240 cells seven or eight steps away.
- **Arm `one`**: every cell is updated at every position of an item. **Arm `place`**: interface cells are updated at every position; far cells are updated every 2, 4, 8 or 16 positions, the more rarely the farther they are from the interface.
- **The sweep** (rule S) passes the error backward across each connection by the sign of its weight times one scale shared by all cells; the gradient of each weight given that error is exact. **Back-propagation** (rule B) is the exact gradient.
- **Score A** is the percentage of whole answers that are right on 300 held-out pairs each of addition and subtraction modulo 89, with 12 letters between the operands. **Score R** is the same on 1,000 held-out pairs each of addition and subtraction whose answer requires the reduction by the modulus, with no letters. The **length test** is 300 held-out additions at 12 and at 40 letters; training items have at most 12.
- A score **reaches a mark** at the first scored row from which three consecutive scored rows are at or above it. Rows are scored every 256 updates; a run has 24,576 updates. "A80" and "R80" denote the updates at which A and R reach 80.
- **Multiply-adds a position** is the number of products in the forward computation of one item at one position, biases not counted. **Computation to a mark** is multiply-adds a position times updates to the mark. Every run sees the same items in the same order, so this product is proportional to the total forward computation.
- **The two diagnostics** score a trained network with the far cells held, after every position, either at zero or at their mean over the scored items at that position. The brief reads the fall of A and of R at the last row: *used* if either falls by 15 points or more at the mean; *not used* if both fall by less than 5 at zero; *a constant is used* if either falls by 15 or more at zero and both by less than 5 at the mean; otherwise *partly*.
- **E42's references** on the same seed, each a network of 256 cells in which every cell receives the input and is read by the readout: `G`, the present network (every cell reads every cell); `F10`, ten fixed local connections a cell; `P8`, `F10` with update periods of 1, 2, 4 and 8 positions.

## 1. Summary

1. **The trial is verified.** The reader reproduces, byte for byte, all 7,200 answers that the worker recorded on the length-test panels (four runs, two lengths, three conditions, 300 items), and it reproduces exactly all 72 counts of the last scored row on the panels behind A and R under the three conditions. The thresholds recomputed from the receipts equal the worker's. The eight check receipts exist and pass. The structure saved in each weight file equals the reader's draws, whose digests are those of the audit.
2. **A network whose input and output are confined to a boundary of 256 cells learns the family** under both rules and in both arms. In updates and in computation it is close to E42's `F10` and never faster (1.02 to 1.27 times the updates and 1.05 to 1.29 times the computation in the eight comparisons). Against the present network under back-propagation it needs 1.5 to 2.5 times the computation.
3. **The far cells are used only in the first ring.** The worker's description *partly* is reproduced. My measurement adds where the use lies: holding rings 2 and 3 (540 of the 768 far cells) at their mean, or giving them the states of another item, changes no score by more than 0.7 of a point, and their states hardly vary with the item, which is already so in the starting weights. The whole effect of the far cells comes from the 228 cells that interface cells read directly, which supply a fifth of the interface cells' connections.
4. **Predictions:** five occurred, three did not, and one (P47.6) holds in one of the two runs it covers. I score them as the worker did.
5. **The roster should not be released as written.** Its two questions are, on the evidence of the trial, either answered in advance (cells beyond the first ring are not used at 1,024 cells, so 4,096 cells would add 3,072 cells of the kind that is not used) or of no consequence for the programme's question (where the slow cells are placed among cells that are not used). Section 8 gives the reasons and states what a later brief that uses this wiring should take from the trial.

## 2. What was verified, and how

**Files and checks** [part A]. On the worker's branch at 578a877b each of the four runs has its receipt, its final weights, its starting weights and its answers file. The eight check receipts `exp45/e47_qual/check1.json` to `check8.json` each record `pass: true`. Checks 1 to 7 were made on the runner with digest 882b38f9; the four runs and check 8 used the runner 8d80a7dc. The difference between the two (commit 62b7b16e) is confined to saving a checkpoint every 1,024 updates, resuming from it, and check 8, which shows that a resumed run ends with the same tensors as an uninterrupted one. The worker's `e47_local.py` and the specification file on the default branch are the same blob (fdd00c69). The change of the specification file between the audited revision and the released one (d55e89dc) is the second diagnostic as an argument of the forward and the audit's digests as self-checks, as the audit asked; I read the difference, as the audit said I would.

**What each receipt records** [part A]. All four runs are complete with 97 scored rows. The digest of the data at row 4,096 is bbd71ab7ca27b72e, that of E26's birth for this seed. `Wc`, `bc`, `bfar` (and `a` in the arm `place`) are in the decayed group, the byte table is not decayed, and the module's own recurrent tensors are in no group and received no gradient. The two rules of an arm start from the same tensors [part B].

**The interruption.** A container restart destroyed the first three runs near 12,000 updates; the worker added the checkpoint and restarted them. The receipts of the lost runs are kept, and every one of their 33, 49 and 33 rows is identical, in every recorded value other than processor time, to the same row of the completed run. A second restart was survived by resuming. The restarts therefore changed no result.

**The saved structure** [part B]. In all eight weight files the table of sources, the periods, the phases, the interface cells and the far cells equal the reader's draws. The digests of the draws are those printed in section 9 of the audit (connections f3361f3823d86f9d; periods under `place` 8394011ba0130f32; phases under `place` 624270ab87410e73).

**The reader against the recorded answers** [part C]. For each run the worker saved the bytes of every answer on the two length-test panels at the last row: plain, with the far cells at zero, and with the far cells at their mean. The reader, on the saved final weights, gives the identical answer in 300 of 300 cases in each of the 24 combinations of run, length and condition.

**The reader on the panels behind the thresholds** [part D]. On the four panels of A (300 items each) and the two of R (1,000 each), plain and under both diagnostics, the reader's counts of right answers equal the counts in the receipt's last row in all 72 cases.

**The arrangement used for decoding** [part V]. To decode quickly, the script computes the reader's forward one position at a time. Its logits differ from those of the reader's own forward by 0 in the arm `one` and by 4e-6 in the arm `place`, and its answers are identical on the sample compared.

## 3. The four runs, as verified

Updates to the marks, final scores and the length test [part E]; all equal the worker's table.

| Run | A80 | R50 | R80 | R90 | Final A | Final R | 12 letters | 40 letters, last row (mean of last three rows) |
|---|---|---|---|---|---|---|---|---|
| `one`, sweep | 13,568 | 9,728 | 17,664 | not reached | 93.3 | 86.3 | 93.0 | 90.0 (90.8) |
| `one`, back-propagation | 10,496 | 6,400 | 11,008 | 17,152 | 96.2 | 94.7 | 94.7 | 95.3 (95.2) |
| `place`, sweep | 15,360 | 10,496 | 19,200 | not reached | 89.2 | 82.0 | 89.0 | 87.3 (87.0) |
| `place`, back-propagation | 10,496 | 6,400 | 11,264 | 15,360 | 95.8 | 95.2 | 97.0 | 96.7 (96.6) |
| E42 `F10`, sweep | 13,312 | 9,472 | 15,104 | 23,296 | 92.2 | 88.5 | 91.7 | (89.1) |
| E42 `F10`, back-propagation | 8,704 | 6,400 | 9,472 | 19,200 | 91.8 | 93.3 | 91.3 | (92.6) |
| E42 `P8`, sweep | 7,680 | 6,400 | 8,448 | 12,288 | 95.7 | 92.5 | 96.0 | (96.0) |
| E42 `P8`, back-propagation | 6,400 | 5,120 | 7,168 | 8,960 | 98.0 | 96.6 | 97.3 | (96.6) |
| E42 `G`, sweep | 5,632 | 3,840 | 6,656 | 13,824 | 98.0 | 95.8 | 98.3 | (96.4) |
| E42 `G`, back-propagation | 5,120 | 3,840 | 6,400 | 7,936 | 98.5 | 98.2 | 98.3 | (94.6) |

The arm `place` under the sweep reaches 80 on R narrowly: after the mark at 19,200 one further row is below 80 (77.5 at 20,224), the highest row is 85.8, and the run ends at 82.0.

**Computation** [part E]. Multiply-adds a position, counted from the wiring that ran: 397,824 for the three maps that do not depend on the cells, 524,288 for the input map into 256 interface cells and the readout's map from them, and for the cells 33 each at period 1, or 33 / p + 11 at period p. That gives 955,904 for `one` and 944,948 for `place`. The references: `G` 1,118,720; `F10` 930,560; `P8` 865,708. The last figure is my own count on the same convention, since neither brief states one; the script documents it. It supposes that a cell of period p adds up its 512 input values with its step weights at every position and makes the three gates' products once a period. If instead every cell's input products are made at every position, as E42's program makes them, the count for `P8` is 929,196, and its four ratios to `G` under back-propagation in the table below become 1.25 and 1.04 for A80 and 1.10 and 0.93 for R80.

| Run | Computation to A80, relative to `G` of the same rule | relative to `G` under back-propagation | Computation to R80, relative to `G` of the same rule | relative to `G` under back-propagation |
|---|---|---|---|---|
| `one`, sweep | 2.06 | 2.26 | 2.27 | 2.36 |
| `one`, back-propagation | 1.75 | 1.75 | 1.47 | 1.47 |
| `place`, sweep | 2.30 | 2.53 | 2.44 | 2.53 |
| `place`, back-propagation | 1.73 | 1.73 | 1.49 | 1.49 |
| E42 `F10`, sweep | 1.97 | 2.16 | 1.89 | 1.96 |
| E42 `F10`, back-propagation | 1.41 | 1.41 | 1.23 | 1.23 |
| E42 `P8`, sweep | 1.06 | 1.16 | 0.98 | 1.02 |
| E42 `P8`, back-propagation | 0.97 | 0.97 | 0.87 | 0.87 |
| E42 `G`, sweep | 1.00 | 1.10 | 1.00 | 1.04 |

**One figure of the receipts is low.** For the arm `place` the receipts and the report's cost table record 936,500 multiply-adds a position. That is the count of the specification file's function, which omits the 11 multiply-adds a position by which a cell of period above 1 adds up its sources' values with the step weights. The brief's table was corrected to 944,948 by the audit's change 8; the function was not. The difference is 0.9% and changes no description.

## 4. The audit's twelve changes in what ran

| Change | In what ran |
|---|---|
| 1. Check 2 as a comparison on the first batch, from E42's start | Yes. `check2.json`: logits within 1.8e-7 and gradients within 2.6e-7 of E42's under both rules; the first sixteen losses within 4.8e-7 |
| 2. Which parameters decay; the module's recurrent tensors take no part | Yes, in every receipt (section 2) and in the weight files, which do not hold those tensors |
| 3, 4. Checks 4 and 6 reworded | Yes. `check4.json` tests the two properties named; `check6.json` reproduces E42's counts on E26's six births with a difference of 0 |
| 5. The shares of far cells that can take part | In the brief; the report does not place them beside the falls. Section 6 below does |
| 6. The second diagnostic and the definition of *used* | Yes. Both diagnostics are recorded at every scored row, on A, R and the length test at 12 and 40 letters |
| 7. Four runs, with `one` beside `place`, and the quantities recorded | Yes |
| 8. The cost table | In the brief. The receipts' count for `place` is the uncorrected one (section 3) |
| 9. `P8` beside `F10` and `G` | Yes in updates. The report leaves the comparison in multiply-adds empty because no count for `P8` was stated; section 3 supplies one |
| 10. Locality inside the interface named as not met | In the brief |
| 11. What every run saves | Yes: starting and final weights with the structure, and the answers |
| 12. The sentence of review 9.150 | Corrected by a note in review 9.152; the original sentence stands above the note |

## 5. The reading

**Does the network learn?** Yes. All four runs reach 80 on R within the dose, the sweep's two runs late (17,664 and 19,200 updates) and without reaching 90, back-propagation's two at about 11,000 updates and reaching 90.

**Against E42's `F10`,** which has the same ten connections a cell and a dense interface, the brief's coarse classes give *about the same* in all eight comparisons (two arms, two rules, A and R), in updates and in multiply-adds. Within that class the direction is uniform: the boundary network needs 1.02 to 1.27 times the updates and 1.05 to 1.29 times the computation of `F10`. The 768 additional cells therefore bought nothing on this task and cost a little, as the brief expected ("no fewer updates are expected from more").

**Against the present network,** the boundary network needs 1.7 to 2.9 times the updates. In computation it needs 1.5 to 1.8 times that of the present network under back-propagation when it is itself trained by back-propagation, and 2.3 to 2.5 times when it is trained by the sweep. The programme's question asks for no greater computation to the mark than back-propagation; on this seed the wiring is at about twice that.

**The sweep against back-propagation in this wiring.** The sweep needs 1.6 (`one`) and 1.7 (`place`) times the updates of back-propagation to reach 80 on R and ends 8 and 13 points lower on R. E42's `F10` shows the same ratio on this seed (1.59), and the present network shows none (1.04). The disadvantage of the sweep in sparsely connected cells, found in E42, is carried into this wiring unchanged.

**Periods outside the interface do not reproduce the gain of E42's `P8`.** In E42, update periods removed most of the cost of sparse connections: `P8` reaches 80 on R in 8,448 updates under the sweep against 15,104 for `F10`. Here the arm `place` is no faster than `one` under back-propagation (11,264 against 11,008) and somewhat slower and lower under the sweep (19,200 against 17,664; final R 82.0 against 86.3). The two designs differ in which cells have periods: in `P8` they are cells that receive the input, here they are cells that do not, and all interface cells have period 1. The trial therefore says nothing against periods; it says that periods given to cells that are hardly used (section 6) do nothing.

**The length test** is within 15 points of every reference in every comparison. At 40 letters the four runs end at 87 to 97.

## 6. The far cells

**The worker's statement is reproduced** [parts C, D, E]. Falls at the last row, in points:

| Run | A: at zero, at the mean | R: at zero, at the mean | 40 letters: plain, at zero, at the mean | Form |
|---|---|---|---|---|
| `one`, sweep | 15.8, 8.5 | 8.6, 5.8 | 90.0, 79.0, 80.3 | partly |
| `one`, back-propagation | 17.3, 6.0 | 15.8, 6.0 | 95.3, 81.0, 89.0 | partly |
| `place`, sweep | 11.7, 8.2 | 8.9, 3.7 | 87.3, 74.7, 77.3 | partly |
| `place`, back-propagation | 7.0, 2.5 | 5.5, 2.0 | 96.7, 88.3, 94.3 | partly |

The falls are stable over the last eight scored rows (for example, A at the mean: 6.3 to 9.2, 4.5 to 8.0, 4.7 to 8.5 and 0.8 to 2.7 in the four runs).

**Where the falls come from** [parts F and I]. Four further measurements on the saved weights.

*By the number of letters.* The same 300 held-out pairs each of addition and subtraction, written with 0, 4, 8 and 12 letters. Each entry gives the plain score, the fall with the far cells at zero, and the fall with the far cells at their mean.

| Run | 0 letters | 4 letters | 8 letters | 12 letters |
|---|---|---|---|---|
| `one`, sweep | 89.7; 10.2; 6.8 | 93.5; 13.5; 7.7 | 93.8; 16.2; 9.5 | 93.2; 16.2; 9.0 |
| `one`, back-propagation | 94.7; 17.5; 5.3 | 96.2; 16.3; 6.2 | 96.2; 16.7; 6.3 | 96.2; 17.3; 6.3 |
| `place`, sweep | 82.3; 13.3; 4.7 | 89.2; 10.7; 6.5 | 89.5; 12.2; 7.7 | 89.7; 12.7; 8.8 |
| `place`, back-propagation | 95.0; 6.8; 2.0 | 96.7; 7.7; 3.3 | 96.2; 7.5; 2.5 | 96.0; 7.2; 2.3 |

By structure (the audit's count), the share of far cells that can take part in an answer is 1.00 at every length in the arm `one`, and 0.63, 0.80, 0.88 and 0.96 at 0, 4, 8 and 12 letters in the arm `place`. The falls do not follow these shares.

*By ring, and with the states of another item.* Scores under five conditions: every far cell at zero; every far cell at its mean; rings 2 and 3 at their mean while ring 1 runs freely; every far cell given, at each position, the state it had in the plain run on a different item of the same panel; rings 2 and 3 alone given another item's states.

| Run, panel | Plain | All at zero | All at their mean | Rings 2 and 3 at their mean | All: another item's states | Rings 2 and 3: another item's states |
|---|---|---|---|---|---|---|
| `one`, sweep, A | 93.3 | 77.5 | 84.8 | 93.0 | 66.8 | 92.8 |
| `one`, sweep, R | 86.3 | 77.7 | 80.6 | 86.3 | 75.0 | 86.2 |
| `one`, sweep, 40 letters | 90.0 | 79.0 | 80.3 | 90.7 | 68.0 | 90.3 |
| `one`, back-propagation, A | 96.2 | 78.8 | 90.2 | 96.2 | 78.3 | 96.2 |
| `one`, back-propagation, R | 94.7 | 78.9 | 88.7 | 94.7 | 80.2 | 94.7 |
| `one`, back-propagation, 40 letters | 95.3 | 81.0 | 89.0 | 95.3 | 80.0 | 95.7 |
| `place`, sweep, A | 89.2 | 77.5 | 81.0 | 89.2 | 73.2 | 89.2 |
| `place`, sweep, R | 82.0 | 73.0 | 78.3 | 82.0 | 75.7 | 82.0 |
| `place`, sweep, 40 letters | 87.3 | 74.7 | 77.3 | 87.0 | 73.7 | 87.0 |
| `place`, back-propagation, A | 95.8 | 88.8 | 93.3 | 95.8 | 89.3 | 95.8 |
| `place`, back-propagation, R | 95.2 | 89.8 | 93.2 | 95.2 | 92.2 | 95.2 |
| `place`, back-propagation, 40 letters | 96.7 | 88.3 | 94.3 | 96.7 | 90.0 | 96.7 |

Holding ring 1 alone at its mean gives, in all twelve rows, exactly the score of holding every far cell at its mean. It must, because an interface cell can read no far cell outside ring 1; the agreement is a check of the measurement.

*How much a state varies with the item, and the size of the weights.* The first four columns give the standard deviation, over the items of the A panel, of a cell's state at the position of `=`, averaged over the cells of a ring. The last four give the mean size of a cell's recurrent weights. The rows "start" are computed on the starting weights, before any update.

| Run | Interface | Ring 1 | Ring 2 | Ring 3 | Weights: interface | Ring 1 | Ring 2 | Ring 3 |
|---|---|---|---|---|---|---|---|---|
| `one`, start | 0.099 | 0.016 | 0.0017 | 0.0002 | 0.152 | 0.150 | 0.151 | 0.150 |
| `one`, sweep | 0.398 | 0.149 | 0.018 | 0.002 | 0.198 | 0.178 | 0.132 | 0.145 |
| `one`, back-propagation | 0.458 | 0.175 | 0.009 | 0.000 | 0.179 | 0.234 | 0.257 | 0.234 |
| `place`, start | 0.099 | 0.013 | 0.0008 | 0.0000 | 0.152 | 0.150 | 0.151 | 0.150 |
| `place`, sweep | 0.370 | 0.061 | 0.003 | 0.000 | 0.201 | 0.106 | 0.024 | 0.001 |
| `place`, back-propagation | 0.441 | 0.124 | 0.005 | 0.000 | 0.176 | 0.259 | 0.112 | 0.006 |

In the arm `place` the mean size of a cell's recurrent weights at the end, by period (2, 4, 8, 16), is 0.108, 0.043, 0.012, 0.001 under the sweep and 0.268, 0.172, 0.048, 0.003 under back-propagation, from 0.150 at the start.

*The structure* [part F]. 20.6% of the interface cells' connections come from far cells, necessarily of ring 1, and 151 of the 256 interface cells have at least one such connection. 23.8% of the connections of the cells of ring 1 come from interface cells.

**What each outcome means.**

1. **Rings 2 and 3 take no part in any answer.** Held at their mean, or given the states of another item, these 540 cells change a score by at most 0.7 of a point (two items of 300), and by nothing on A and R in the two runs under back-propagation. Their states hardly vary with the item: the variation of a cell of ring 2 is between a twentieth and less than a hundredth of that of an interface cell, and that of a cell of ring 3 is at most 0.002.
2. **Ring 1 takes part.** With every far cell at its mean the scores fall by 2 to 10 points, and with another item's states they fall by 3 to 27. The second fall is the larger in every row, which shows that interface cells do read what the cells of ring 1 hold about the item: a wrong value misleads them more than an uninformative one. These 228 cells receive a quarter of their connections from interface cells and supply a fifth of the interface cells' connections; they serve as intermediate cells between interface cells.
3. **A large part of the fall at zero is the removal of a constant.** Holding the far cells at zero removes their average level as well as their variation; on A the fall at the mean is between one third and seven tenths of the fall at zero. The average level of an unvarying cell is a bias, not a computation.
4. **Distance in time is not what keeps rings 2 and 3 out.** At 40 letters there are more than forty positions between the first operand and the answer, which is ample for a value to cross three connections, and the two rings still take no part. Nor do the falls at the mean follow the number of letters: they are flat under back-propagation and rise by two to four points from 0 to 12 letters under the sweep.
5. **The attenuation is present before training.** On the starting weights the variation of a state with the item falls by a factor of six to ten with each ring in the arm `one` (0.099, 0.016, 0.0017, 0.0002) and faster in the arm `place`. The starting recurrent weights have a mean size of 0.15 and a cell has eleven sources, so a difference between items shrinks at every connection unless it is renewed by the input, which only interface cells receive. Training raises the variation of the interface and of ring 1 and leaves rings 2 and 3 where they began.
6. **In the arm `place` the slow far cells lose their weights under both rules.** A weight that receives no gradient ends the dose at 0.2% of its starting size, by the decay alone. The cells of period 16 end at about 1% of it under the sweep and 2% under back-propagation, and the cells of period 8 at 8% and 32%. A quarter of the far cells have thus in effect been removed by the decay under both rules, and under the sweep the next quarter is close to it. The worker reported the fall of the far cells' mean weight under the sweep without reading it; under back-propagation the same loss is hidden in the mean by the growth of the weights of the cells of period 2.

**The description that the measurements support** is therefore narrower than the brief's form *partly*: the far cells that interface cells read directly are used, and the far cells two or more connections away are not. That the attenuation at the start is the cause of the second half of this sentence is an explanation, not a finding; with one family nothing requires those cells either. The result that would refute it is a network started so that differences between items do reach rings 2 and 3 which still leaves them unused on a task that the interface alone cannot hold.

## 7. The predictions

| | Prediction (chance) | Outcome |
|---|---|---|
| P47.1 | Checks 1 and 2 pass under both rules (85%) | Yes |
| P47.2 | Back-propagation, `place`: reaches 80 on R (70%) | Yes, from 11,264 |
| P47.3 | Sweep, `place`: reaches 80 on R (45%) | Yes, from 19,200, narrowly (section 3) |
| P47.4 | Back-propagation, `one`: far cells *used* (45%) | No: *partly* (fall at the mean 6.0 on A and on R) |
| P47.5 | Sweep, `one`: far cells *used* (30%) | No: *partly* (8.5 and 5.8) |
| P47.10 | *A constant is used* in at least one run (30%) | No: all four *partly* |
| P47.11 | Back-propagation: `place` about the same as `one` in updates to 80 on R (50%) | Yes: 11,264 against 11,008 |
| P47.6 | `place`: the cosine between the sweep's and the exact gradient is lower for far than for interface cells at the end by 0.1 or more (60%) | Holds in back-propagation's run (0.617 against 0.443), not in the sweep's (0.603 against 0.537). The prediction does not say which run; counted as not occurred |
| P47.7 | Back-propagation, `one`: updates to 80 on R about the same as or more than E42's `F10` (80%) | Yes: about the same (11,008 against 9,472) |

Five occurred, three did not, one is divided. This is the worker's scoring.

**The misses.** P47.4, P47.5 and P47.10 were each given a chance below one half, so their not occurring is the side the coordinator thought more likely. The reason they did not occur is the one section 6 establishes: what the far cells contribute beyond a constant comes from the first ring alone and amounts to two to nine points of A, below the 15 points that *used* requires; and since that contribution is not zero, the case *a constant is used* does not occur either. P47.6 was written for one run and there are two; the cosine of the far cells falls less under the sweep's own training (0.626 to 0.537) than under back-propagation's (0.626 to 0.443). P47.3 occurred against a chance of 45%, but narrowly.

## 8. The roster

**What the brief plans.** On seed 73129: at 1,024 cells the arm `scatter` (the same periods as `place`, assigned to far cells at random) under both rules; at 4,096 cells the arm `one`, under back-propagation first and under the sweep after that run has been read. The other two seeds follow only for arms the first seed shows worth it.

**What it costs** [part H]. The four runs of the trial took 15.8 processor-hours (3.6 to 4.5 a run), plus about 6.6 lost at the first restart; the container restarted twice in about eight hours. The worker's forecast for the roster, measured on an idle host, is 48.2 processor-hours: 5.5 for each `scatter` run, 17.9 and 19.2 for the two runs at 4,096 cells. The trial came in at 0.84 of its forecast, which would make the roster about 40. A run at 4,096 cells is at least 15 hours of one processor without interruption, and a checkpoint is 35 minutes of work.

**What it could show that the trial has not.**

1. **The arm `scatter` at 1,024 cells** (two runs, 11 processor-hours) would show whether it matters that the slow cells are placed far from the interface. On the evidence of section 6 the only far cells whose periods can matter are the 228 of ring 1, because the others take no part. Under `place` these have periods 2 and 4; under `scatter` they would have a mixture of 2 to 16. The two runs would measure the effect of slowing the intermediate cells of ring 1. Periods outside the interface save 1.1% of the computation at this size. I know of no decision that depends on the outcome.
2. **The arm `one` at 4,096 cells** (two runs, 37 processor-hours, 2.3 times the cost of the whole trial) would show whether a network with fifteen far cells for each interface cell still learns, and whether those cells are used (P47.8: not *used*, 65%). The trial answers most of this in advance. At 4,096 cells ring 1 again has 228 cells, and the further 3,612 cells lie in rings 2 to 8. At 1,024 cells rings 2 and 3 take no part, their states hardly vary with the item, and before training the variation already falls sixfold to tenfold with each ring; nothing in the larger network changes any of this. The run would very probably report what the trial reports, the far cells *partly* used through ring 1. The follow-up that the brief plans for a finding of *not used* at that size, extra steps before the answer to separate distance in time from a failure to learn, is also answered in advance by the 40-letter panel (section 6, point 4).

**Recommendation: do not release the roster.** The trial has done what a trial is for: it shows that the roster's two questions are not worth their cost. The reasons, in order of weight:

- **Section 6.** The cells that the roster would add or rearrange are cells that take no part.
- **Cost.** 40 to 48 processor-hours on a host that restarted twice during the trial, for two descriptions that can be given now with fair confidence.
- **The rule of at most three lines of work at once.** The long lives of E44, the retention line (E48) and the interface line (E49, E50) are three, and each bears more directly on the restated question or on the user's requirement than either question of the roster.
- **The brief's own expectation** that more cells bring no benefit on one family.

If the coordinator nevertheless wants one run of the roster, the run at 4,096 cells under back-propagation alone (17.9 processor-hours) is the one to take, read with the ring measurement of part F on its weight file; the `scatter` runs and the sweep's run at 4,096 cells should be dropped in any case.

**What the trial licenses,** as that seed's: a network of 1,024 cells whose input and output are confined to 256 of them learns the first family under both rules, in about the same updates and computation as E42's `F10` and in 1.5 to 2.5 times the computation of the present network under back-propagation; of its 768 far cells, the 228 that interface cells read directly take part in the answers and the other 540 do not; a cell costs 33 multiply-adds a position. The other two seeds are not needed for the decision on the roster. A later brief that builds on this wiring would cover them in its own first rows.

**What follows for a later brief that uses this wiring.** These are requests for that brief, not runs to start now.

1. **Periods belong at the interface.** E42's `P8` is the only sparsely connected arrangement on record whose computation to the mark is near that of back-propagation on the present network (0.87 to 1.25 times, by the two counts of section 3), and its periods are in cells that receive the input. E47 gave periods only to cells that are hardly used. The specification file's first self-check already shows that the layer at 256 cells reproduces E42's `P8`. An arm whose interface cells have `P8`'s periods needs a change of the draws and of the runner's faster forward, which assumes period 1 at the interface, and therefore a short audit; two runs at 1,024 cells would cost about eight processor-hours. Whether it is wanted depends on E42's other two seeds, which another agent is verifying.
2. **No periods for far cells** at sizes where they save about 1% of the computation and where the decay removes the slow cells.
3. **Far cells beyond ring 1 need a reason and a route.** The reason is more than one family to hold, which is the matter of the long-life world and of the retention line and cannot be supplied by a run on one family. The route is a start under which differences between items reach those cells. The second is the untested explanation of section 6, and no design should rest on it before its refuting result has been looked for.
4. **Measured time.** An update costs 0.40 to 0.49 processor-seconds at 1,024 cells and 1.9 to 2.1 at 4,096 (the worker's check 7), against about 0.10 for E42's networks on their host, although the counted computation is lower than the present network's. Arithmetic by index is slow on a processor of this kind, and any long job with this wiring pays that factor: four to five at 1,024 cells and about twenty at 4,096.

## 9. What I could not verify

- **Anything that needs torch.** The gradients, the cosines between the sweep's and the exact gradient, the magnitudes recorded during the runs and the equality with E42's runner on the first batch are the receipts'. The reader confirms the forward computation on the final weights and nothing about the training.
- **The processor times,** which are the receipts'. E47's were measured with up to three processes on four cores, E42's on another host; the two are not comparable, and the comparison above is in counted computation only.
- **The other two seeds.** Nothing of E47 has been run on them. The E42 figures used here are those of the first seed, verified on 16 November; another agent is verifying E42's other seeds, and if `F10` or `P8` differ much between seeds, the comparisons of section 5 carry that uncertainty.
- **The count for `P8`** (865,708 multiply-adds a position) is mine and has not been checked by the coordinator.
- **Why the slow far cells lose their weights** in the arm `place` (section 6, point 6) is not established beyond the arithmetic of the decay; I report the fact.
- **Why rings 2 and 3 are not used.** The attenuation at the start is measured; that it is the cause is an explanation, stated with its refuting result in section 6.

## 10. Commands

From the repository root; `V=PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/e47_trial_verify.py`. The times are those measured on a host that was running several other jobs (load average near 30 on four cores); on an idle host they are several times shorter.

| Command | What it shows | Time |
|---|---|---|
| `python3 -I $V A B E G H` | files, checks, receipts, the saved structure, thresholds, computation, the brief's descriptions, the predictions, cost | 5 seconds |
| `python3 -I $V V` | the arrangement used for decoding against the reader's own forward | 20 seconds |
| `python3 -I $V C D` | the reader against the recorded answers and against the counts of the last row | 7 minutes (391 seconds) |
| `python3 -I $V I` | the variation of a state with the item, by ring, on the starting weights | 25 seconds |
| `python3 -I $V F` | the measurements of section 6 on the four final weight files | 28 minutes in one process; `python3 -I $V F one_S one_B` (16 minutes) and `python3 -I $V F place_S place_B` (12 minutes) can run side by side |

The outputs from which this document was written are those of these commands. The script reads the worker's branch at 578a877b and E42's receipts at 7fbdbdc2; both must be present in the local repository.
