# Check of the brief of E57, revision 1 (17 November 2026)

Written for the outside reviewer by an agent that has not seen the reviewer's reasoning. The brief is `TASK_E57_THE_SWEEP_WITH_A_SCALE_FOR_EACH_RECEIVING_UNIT_20261117.md` beside this file, committed at 8cadfd10. Its SHA-256 was `e39604ff359b97153a69a8cca63e4683cc2d59b27935e92ec4d19f99101e9474` when the check began; the last section of this file says whether it had changed when the check ended. The work was read-only. Nothing was trained, torch is not installed, no code of another branch was executed, and the files of other branches were read with `git show`. The check began at 23:15 UTC on 17 November 2026. It is the check that the brief's status line names. It is not the independent audit that the programme's rules ask for. This check committed nothing; two early states of this file were taken into the reviewer's commits 9c515b1a and 131fe6ee by the staging of those commits.

**Shell names.** `D=origin/codex/minifly-consolidated-20261023` (at 7cabb13a); `W=origin/claude/gallant-dijkstra-i9zgsr` (at 44f91edc); `W9=origin/claude/peaceful-noether-vcvrmf` (at a36de661); `M=MOD97_LOCAL_RULE_20261106`; `R=PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109`. "Brief, line n" cites a line of the brief. "Runner, line n" cites a line of `$W9:$M/exp49_interface.py` (SHA-256 `cdf93ce167d562c8a3765c1ea4f98f74623469492324a85aa1eb34216b7ccbf3`, the hash that E49's receipts record). `APP <part>` is the script of the appendix, run from the repository root as `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -I -B <the script> <part>`.

**Terms.** As in the brief. In addition: *the measurement* is `$R/SWEEP_SCALE_VARIANTS_ALIGNMENT_20261117.md` with `$R/sweep_scale_variants.py`; this check ran the whole script once (258 seconds) and every weight table it printed is in the document word for word (`APP figures`). The measurement's networks are named as it names them: `in16_S` and `in16_B` are the network with a limited input at the end of its run under the sweep and under back-propagation, `F10_S` and `G_S` the network with limited connections and the dense network at the end of their runs under the sweep, `in16_start` the network with a limited input before training. *The two maps of the recurrent layer* are the input map and the recurrent map. `rowcells` is a scale that the repair of defect 1 introduces: the scale `row` at the two maps of the recurrent layer and the scale `matrix` at the other four maps (the measurement's variant V2 "at the two GRU maps only").

## Verdict

Release after the listed changes; do not release as it stands.

Every figure of "What the record says" and of "Runs" was recomputed from the receipts and from the output of the measurement's script. All agree except two cosines that are off by 0.001 and one statement about the dense network (defects 3 and 12). The hashes and paths that the checks name exist and are right. `MaskedSign` accepts a scale of one column or of W's shape, and the brief's function `magnitudes` is the measurement's V0, V2 and V4 on all six maps. The defects are of three kinds.

- **The design would support a wrong conclusion or could not register the right one** (defects 1 to 5). In the network that did not learn, the measurement's own second table shows that the gain of the scale `row` at the two maps of the recurrent layer does not come from the scale at the cells: it comes from the scale of the readout units at `lm`. The roster has no run that separates the cells from the readout, while the explanation under test names the cells. A run of 24,576 updates registers "learns the family" only if the sweep is at most 1.17 times as slow as back-propagation, and on every network on file where the sweep learns it is 1.04 to 1.89 times as slow. The record section leaves out that the network with a limited input which learned under back-propagation has the same low cosines as the one that failed. The run that is meant to tell cause from consequence has no fixed reading, and its one fixed figure cannot vary.
- **The runner and the checks cannot be carried out as written** (defects 6 and 7). A copy of `exp49_interface.run` with the six listed changes stops with a `KeyError` for `G`, calls the brief's new function `magnitudes` with the arguments of E49's function of the same name, and has no scale `exact`, which check 2 needs. The fallback of check 1 cannot be met by a crossing that differs in rounding only, and check 2 cannot load the weight file as worded.
- **Readings, rules and wording that are incomplete** (defects 8 to 13).

Each defect has a replacement text. The replacements of the sections "Runs", "What the record says" (its fourth and fifth points), "How the result is stated", "The runner", "Checks before the runs" and "Predictions" are given once, in full, under defects 2, 3, 5, 6, 7 and 8; the other defects refer to them. With all changes the job has eight runs and costs about 12 CPU-hours in place of six runs and 7.

## Defects, the most serious first

### 1. In the network that did not learn, the gain of the scale `row` comes from three places, and at the two maps of the recurrent layer it does not come from the cells; no run separates them, and the explanation names the cells

**What is wrong.**

1. Brief, line 21: "in the networks with limited connections, a scale for each receiving unit brings the sweep's gradient closer to the exact one, most of all in the network that did not learn". Brief, line 108: the failure arises "because one scale for a whole matrix misstates the sizes of weights that training makes unequal between cells". The brief's table (lines 34 to 45) gives the scale `row` at every map only. The measurement prints a second table for each network, "variant at the two GRU maps only" (document, lines 924 to 932 for `in16_S`), which the brief does not cite. For `in16_S` (`APP figures`):

   | Tensor | `matrix` | `row` at every map | `row` at the two maps of the recurrent layer only |
   |---|---|---|---|
   | byte table | 0.367 | 0.697 | 0.473 |
   | `l1` | 0.298 | 0.624 | 0.527 |
   | input map | 0.603 | 0.676 | 0.594 |
   | recurrent map | 0.651 | 0.762 | 0.654 |
   | `lm`, `l2` | 0.606, 0.643 | 0.694, 0.710 | 0.606, 0.643 |

   With a scale for each gate of each cell and nothing else, the cosine of the input map falls (0.603 to 0.594) and that of the recurrent map does not move (0.651 to 0.654).

2. The scale `row` was then applied at one map at a time, with `matrix` at the others (`APP where`, which uses the measurement's own functions; its rows "at every map", "at the two maps of the recurrent layer" and "rowcol" reproduce the measurement's figures). `in16_S`, mean cosine over the same 16 batches:

   | Scale | byte table | `l1` | input map | recurrent map | `lm` | `l2` |
   |---|---|---|---|---|---|---|
   | `matrix` at every map | 0.367 | 0.298 | 0.603 | 0.651 | 0.606 | 0.643 |
   | `row` at `l3` only | 0.363 | 0.275 | 0.575 | 0.594 | 0.694 | 0.710 |
   | `row` at `lm` only | 0.405 | 0.347 | 0.688 | 0.769 | 0.606 | 0.643 |
   | `row` at `l2` only | 0.381 | 0.302 | 0.603 | 0.651 | 0.606 | 0.643 |
   | `row` at `l1` only | 0.446 | 0.298 | 0.603 | 0.651 | 0.606 | 0.643 |
   | `row` at the input map only | 0.516 | 0.566 | 0.603 | 0.651 | 0.606 | 0.643 |
   | `row` at the recurrent map only | 0.358 | 0.292 | 0.594 | 0.654 | 0.606 | 0.643 |
   | `row` at every map but `l3` | 0.705 | 0.637 | 0.678 | 0.778 | 0.606 | 0.643 |
   | `row` at every map | 0.697 | 0.624 | 0.676 | 0.762 | 0.694 | 0.710 |

   The gain has three sources, and they are three different kinds of unit.
   - The scale at the input map (one for each gate of each cell) raises `l1` from 0.298 to 0.566 and the byte table to 0.516. This is the part that the brief's explanation describes.
   - The scale at `lm` (one for each of the 512 readout units) raises the input map from 0.603 to 0.688 and the recurrent map from 0.651 to 0.769. The whole gain of the two maps of the recurrent layer in the brief's table comes from here.
   - The scale at `l3` (one for each of the 257 output classes) raises `lm` and `l2` and lowers the two maps of the recurrent layer (to 0.575 and 0.594). In `in16_S` the mean magnitudes of the rows of `l3` that belong to the 11 classes that occur as targets range from 0.046 to 0.425 (`APP weights`), so that under `row` the errors of the output classes are weighted against each other by up to nine to one before they reach any other map.
   - The scale at the recurrent map alone changes nothing (0.651 to 0.654).

   In `F10_S` the sources are placed differently: of the rise of the two maps of the recurrent layer from 0.647 and 0.685 to the 0.696 and 0.730 of `row` at every map, the scale at the recurrent map alone gives 0.683 and 0.715, the scale at `lm` alone 0.658 and 0.698, and the scale at `l3` alone 0.647 and 0.686 (`APP where`).

3. The roster has no run that separates these places. If the network with a limited input learns under `row`, brief line 108 writes *supported* for an explanation that names the cells, although the scale of the readout units or of the output classes may have done it. The scale `rowcol` cannot serve as the comparison, because it differs from `row` in two things at once: the form (a product with a column mean) and the place (the recurrent layer only).

4. The change of the gradient's size is not the confound; the place is. Under `row` the norm of the sweep's gradient over that of the exact gradient changes at every tensor behind `l3`. In `in16_S` it falls by a factor of 1.8 to 2.6 (`lm` 1.66 to 0.71, `l2` 1.64 to 0.69, recurrent map 0.44 to 0.20, input map 0.30 to 0.13, `l1` 0.29 to 0.11, byte table 0.18 to 0.10; document, lines 938 and 940); in `F10_S` it rises by a factor of 1.8 to 2.0 at `lm`, `l2` and the two maps of the recurrent layer and of 1.0 to 1.6 at `l1` and the byte table (document, section 6.4, `F10_S`). AdamW divides the step of each weight by the root of the running mean of that weight's squared gradient, so that a factor that is common to a tensor and steady over a run leaves the step unchanged; the gradients are far above the optimizer's constant of 1e-8 (the exact gradient of `in16_S` has norm 3.8 over 555,025 weights). What AdamW does not remove is a change of direction, which is what the cosines measure, and the two tables above show that this change enters at the readout as much as at the cells.

5. A run with the scale `row` at the two maps of the recurrent layer only is therefore needed, and it is worth its cost (0.95 CPU-hours at the brief's length, 1.9 at the length of defect 2): it is the only run that tests the explanation as the brief words it, and the measurement already holds its reference figures for check 2.

**Repair.**

(a) Replace brief line 14 (the term *the scale*) by:

> - *The scale* is the choice of S. In every run of E26 to E50 each entry of S is one number for the whole matrix, the mean of |W| over its kept entries; this is the scale `matrix`. With the scale `row`, S[i, j] is the mean of |W| over the kept entries of row i, at every one of the six maps. Row i of a matrix holds the incoming weights of one *receiving unit*: a unit of the code at `l1`; one gate of one cell at the input map and at the recurrent map, so that a cell has three rows and three scales in each of these two matrices; a readout unit at `l2` and at `lm`; an output class at `l3`. A unit that receives through two maps has one scale at each. With the scale `rowcells`, the input map and the recurrent map have the scale `row` and the other four maps the scale `matrix`. With the scale `rowcol`, at the input map and at the recurrent map S[i, j] is the mean of row i times the mean of column j divided by the mean of the matrix (all over kept entries), and the other four maps keep the scale `matrix`.

(b) Add the run of the network with a limited input under `rowcells` (the section "Runs" is given in full under defect 2).

(c) State the two tables above in "What the record says" (given in full under defect 3), and word the explanation in two parts (given in full under defect 5).

### 2. A run of 24,576 updates cannot register the form of success that the record makes most likely

**What is wrong.**

1. Brief, line 103: a run *learns the family* "if A and R both reach 80 within the run and their levels at the end are both at or above 80". A score reaches 80 at the first of three consecutive rows, so the last update at which it can do so in a run of 24,576 is 24,064. E49's run under back-propagation reaches 80 on R at 20,480 (`APP receipts`). A run under the sweep therefore learns the family, in the brief's sense, only if it needs at most 24,064 / 20,480 = 1.17 times the updates of back-propagation on R.

2. Wherever the sweep learns on file, it needs more than that. Updates to 80 under the sweep over those under back-propagation (`APP receipts`; E50 from `LOG.md`, entry of 22:01):

   | Network | Seed | A | R |
   |---|---|---|---|
   | dense (`G`) | 73129, 93683, 116089 | 1.10, 1.19, 1.24 | 1.04, 1.19, 1.25 |
   | limited connections (`F10`) | 73129, 93683, 116089 | 1.53, 1.89, 1.36 | 1.59, 1.81, 1.47 |
   | fixed code, dense input (E50's `C-dense`) | 73129 | | 1.85 |

   A scale that brought the network with a limited input to where the network with limited connections already is (1.47 to 1.81 on R) would reach 80 on R between updates 30,100 and 37,100 and would be read as *partly*, for which brief line 108 has neither word.

3. The second condition is as narrow. The level of R of E49's run under back-propagation is 82.5, and single rows scatter by 3 to 4 points (`$R/E49_E43ADD2_VERIFICATION_20261117.md`, section 2.5). Back-propagation itself passes the condition by 2.5 points.

4. The length of 24,576 updates is a setting that was fixed for the dense network and carried over (rule 1 of section 4 of the plan). E49's run under the sweep was never continued; the reviewer's verification argues that it is not rising, which is a different statement from a run that shows it. E50 runs its network with 16 inputs a cell for 49,152 updates (`LOG.md`, entry of 22:01).

**Repair.** Replace the section "Runs" (brief, lines 50 to 61) by the text below, replace "six runs" and "about 7 CPU-hours" in brief line 3 by "eight runs" and "about 12 CPU-hours", and replace "fewer than 24,576 updates, made to test the runner" in brief line 17 by "made by a check to test the runner".

> ## Runs
>
> | Network | Scale | Rule | Updates | Runs | Reference on file |
> |---|---|---|---|---|---|
> | limited input (`in16`) | `matrix` | S | 49,152 | 1 | E49's run under the sweep, whose scored rows to update 24,576 this run must repeat; the second half shows whether that run failed or was late |
> | limited input (`in16`) | `row`, `rowcells`, `rowcol` | S | 49,152 | 3 | E49's two runs, and the run with `matrix` of this job |
> | limited input (`in16`) | `none` | B | 24,576 | 1 | E49's run under back-propagation, which this run must repeat; the cosines of the sweep's gradient are recorded on its weights during the run |
> | limited connections (`F10`) | `row`, `rowcol` | S | 24,576 | 2 | E42's `F10` under the sweep and under back-propagation |
> | dense (`G`) | `row` | S | 24,576 | 1 | E42's `G` under the sweep and under back-propagation |
>
> The seed is 73129. Everything but the scale, and the number of updates of four runs, is E49's and E42's: the start, the masks, the optimizer (AdamW, rate 0.001, weight decay 0.25 on every tensor but the byte table), the stream, the sets of items on which the scores are taken, the scored rows every 256 updates. The four runs of the network with a limited input under the sweep have 49,152 updates, because under back-propagation this network reaches 80 on R only at update 20,480 and the sweep, where it learns, needs 1.04 to 1.89 times the updates of back-propagation (E42, three seeds; E50); at 24,576 updates a scale that removed the failure and left the sweep as slow as it is elsewhere would not be seen to learn. The runs with `matrix`, `row` and `rowcells` are started first. The run with `rowcol` and the run under back-propagation follow, and the three other runs take the places that become free.
>
> The cost is about 1.9 CPU-hours for a run of 49,152 updates of the network with a limited input, 0.95 for its run under back-propagation and 0.87 for each of the three others: 11.2 CPU-hours for the eight runs, and about 1 for the checks and for the cosines recorded during the runs (0.35 for the checks; 0.45 for the cosines, of which one set is about 10 CPU-seconds, with 25 sets in a run of 49,152 updates and 13 in a run of 24,576). With three processes at a time the runs take about four and a half hours.

### 3. "What the record says" leaves out three results of the measurement that change what the runs can be expected to show, and one statement about the dense network is not what the measurement found

**What is wrong.**

1. The table has no row for the network with a limited input that learned under back-propagation (`in16_B`), although the measurement has it (document, lines 966 to 985; `APP figures`). Its cosines under the scale `matrix` are as low as those of the network that failed: `l1` 0.309 against 0.298, input map 0.523 against 0.603, recurrent map 0.582 against 0.651. The scale `row` gains 0.04 to 0.05 there at `l1` and at the two maps of the recurrent layer and loses 0.04 at the byte table (0.442 to 0.405). The large gain that brief line 21 cites ("most of all in the network that did not learn") is a property of the weights that the failed run ended in. A run under `row` starts from weights at which no scale changes anything and, if it learns, passes through weights that are more like those of `in16_B` than those of `in16_S`. The reviewer's verification of E49 had already found that the recorded cosines "do not separate the runs that fail from those that learn" (section 2.7); the brief does not cite the sentence.

2. The decomposition of defect 1 is not stated.

3. Brief, line 105: "the measurement found no change of the gradient's direction in the dense network". The measurement found none at the two maps of the recurrent layer (0.788 to 0.791 and 0.819 to 0.821). At the byte table the scale `row` raises the cosine from 0.722 to 0.789 and at `l1` from 0.661 to 0.690 (document, lines 600 and 602). The brief's table gives only the two maps for the dense network.

4. Two figures of the last row of the table (brief, line 45) differ from the script's output by 0.001: the recurrent map under `row` is 0.739 and not 0.740, and the input map under `rowcol` is 0.746 and not 0.745 (`APP figures`). At the start the cosine of the byte table falls under every other scale (0.558 under `matrix`; 0.541, 0.532 and 0.535), which brief line 47 does not say.

5. E50's stage 2 bears on the explanation and is cited only under "What it cannot show" (brief, line 125), without figures. On the running receipts as they stood at commit 8b7d3a89 of `origin/claude/e50-fixed-input-code-017dSE`, the network with 16 inputs a cell and a fixed code is learning under the sweep with the scale `matrix`: at update 16,384 of 49,152, A is at 26.3, R at 23.7 and the taught pairs at 41.2 (back-propagation at the same update: 59.3, 49.6 and 78.3). A limited input alone therefore does not stop the sweep under the scale `matrix`; in E49 the failure goes with a code that is trained. This supports the half of the explanation that concerns the error reaching the code and should stand in the record.

**Repair.** Replace the fourth and fifth points of "What the record says" (brief, lines 32 to 48) by:

> - **The sweep's gradient under other scales** (`$V:$R/SWEEP_SCALE_VARIANTS_ALIGNMENT_20261117.md`, with `sweep_scale_variants.py`; the reviewer ran the script again). Mean cosines over 16 batches of taught items, on saved weights; nothing was trained under any scale. The column `rowcells` is the measurement's variant V2 "at the two GRU maps only".
>
> | Network, weights | Tensor | `matrix` | `row` | `rowcells` | `rowcol` |
> |---|---|---|---|---|---|
> | Limited input, end of the sweep's run (did not learn) | byte table | 0.367 | 0.697 | 0.473 | 0.638 |
> | | `l1` | 0.298 | 0.624 | 0.527 | 0.721 |
> | | input map | 0.603 | 0.676 | 0.594 | 0.714 |
> | | recurrent map | 0.651 | 0.762 | 0.654 | 0.710 |
> | | `lm`, `l2` | 0.606, 0.643 | 0.694, 0.710 | 0.606, 0.643 | 0.606, 0.643 |
> | Limited input, end of the run under back-propagation (learned) | byte table | 0.442 | 0.405 | 0.391 | 0.383 |
> | | `l1` | 0.309 | 0.363 | 0.331 | 0.321 |
> | | input map | 0.523 | 0.567 | 0.537 | 0.562 |
> | | recurrent map | 0.582 | 0.631 | 0.592 | 0.608 |
> | | `lm`, `l2` | 0.700, 0.706 | 0.700, 0.707 | 0.700, 0.706 | 0.700, 0.706 |
> | Limited connections, end of the sweep's run | byte table | 0.677 | 0.794 | 0.716 | 0.778 |
> | | `l1` | 0.537 | 0.631 | 0.620 | 0.693 |
> | | input map | 0.647 | 0.696 | 0.683 | 0.710 |
> | | recurrent map | 0.685 | 0.730 | 0.715 | 0.738 |
> | Dense, end of the sweep's run | byte table, `l1` | 0.722, 0.661 | 0.789, 0.690 | 0.737, 0.687 | 0.783, 0.740 |
> | | input map, recurrent map | 0.788, 0.819 | 0.791, 0.821 | 0.790, 0.822 | 0.800, 0.830 |
> | Limited input, start | byte table, `l1` | 0.558, 0.716 | 0.541, 0.720 | 0.532, 0.718 | 0.535, 0.720 |
> | | input map, recurrent map | 0.745, 0.734 | 0.749, 0.739 | 0.745, 0.734 | 0.746, 0.734 |
>
>   At the start no scale changes a cosine of the two maps of the recurrent layer by more than 0.006, because the starting magnitudes are nearly equal across rows; the cosine of the byte table falls by 0.02 to 0.03 under each of the three other scales. The gain appears in weights that training has made unequal. A scale taken for each column, or the product form at every map, moves the gradients of `lm` and `l2` away from the exact ones (by 0.02 to 0.12), which is why the scale `rowcol` is applied at the recurrent layer only.
>
>   Three limits of these figures bear on the runs. First, in the network that did not learn, the gain of the scale `row` at the two maps of the recurrent layer does not come from the scale at these maps: under `rowcells` their cosines are 0.594 and 0.654, against 0.603 and 0.651 under `matrix`. With the scale `row` at one map only and `matrix` at the others (`COORDINATION_20261117/CHECK_OF_TASK_E57_BRIEF_20261117.md`, defect 1), the scale at `lm` raises them to 0.688 and 0.769; the scale at the input map raises `l1` to 0.566 and the byte table to 0.516; the scale at `l3` raises `lm` and `l2` to 0.694 and 0.710 and lowers the two maps of the recurrent layer to 0.575 and 0.594. Second, the network with a limited input that learned under back-propagation has cosines as low as those of the network that did not learn under the sweep (`l1` 0.309 against 0.298), and there the scale `row` gains 0.04 to 0.05 at `l1` and at the two maps of the recurrent layer and loses 0.04 at the byte table. The large gain is found only on the weights in which the failed run ended. Third, in the dense network the scale `row` leaves the two maps of the recurrent layer where they are (0.003 or less) and raises the byte table by 0.067 and `l1` by 0.029.
> - **E50, stage 2** (running; `origin/claude/e50-fixed-input-code-017dSE` at 8b7d3a89). With a fixed code that is not trained and 16 inputs a cell, the sweep with the scale `matrix` is learning: at update 16,384 of 49,152, A is at 26.3, R at 23.7 and the taught pairs at 41.2 (back-propagation: 59.3, 49.6 and 78.3). A limited input alone does not stop the sweep; in E49 the failure goes with a code that is trained.
> - A cosine nearer to 1 is not a demonstration that a network learns. The low cosines of the network with a limited input were measured at the end of a run that had failed, and the network that learned under back-propagation has them as well; the cosines that E49 recorded do not separate the runs that failed from those that learned (the reviewer's verification, section 2.7). Whether a low cosine is a cause of the failure or a consequence is not known. No run under another scale exists.

### 4. The run that is meant to tell cause from consequence has no fixed reading, its one fixed figure cannot vary, and its cosines are read on one batch

**What is wrong.**

1. Brief, line 106: for the run with the scale `matrix`, "the report gives the update at which the cosine of `l1` first falls below 0.5 and the level of the taught pairs at that update". In E49's run under the sweep the score on the taught pairs is between 1.2 and 4.0 at every multiple of 2,048 updates (`APP receipts`). The second figure is therefore the same wherever the cosine falls. No sentence says what an early or a late fall would mean, and brief line 127 leaves the reading open ("beyond what the cosines during the run show").

2. The end of both courses is on file. Under the scale `matrix` the cosine of `l1` is 0.298 in the network that failed under the sweep and 0.309 in the network that learned under back-propagation (defect 3). The cosine therefore falls below 0.5 on the way to a network that learns as well, and a fall observed in the failing run alone shows nothing. To read it, the same cosine is needed along a run that learns. E49 saved no weights between the start and the end, so this needs one more run: the network with a limited input under back-propagation, with the sweep's cosines recorded on its weights (0.95 CPU-hours). That run also repeats E49's run under back-propagation at full length, where check 3 covers 256 updates.

3. Brief, line 88: the cosines are recorded "on the first batch of the stream". On one batch a cosine scatters widely. In `in16_S` under `matrix` the byte table is 0.275 on the first batch, 0.367 in the mean of the 16 further batches and between 0.144 and 0.498 on single batches; `l1` lies between 0.206 and 0.345 and the input map between 0.462 and 0.729 (document, lines 917 and 918). The cosine of `l1` must pass 0.5 at some reading on its way from 0.72 at the start to 0.30 at the end; with a scatter of 0.14 between batches, which reading is the first below 0.5 (brief, line 120) depends on the batch. Sixteen batches cost about 10 CPU-seconds a set (an update costs 0.13 CPU-seconds, E49's check 6; a set is five passes on 16 batches when all four scales are recorded) and 0.45 CPU-hours for the whole job.

4. Brief, line 88, names "update 0" and "the first batch". In the runner the first batch is drawn inside the loop, before the first step (lines 222 to 226). A worker who draws it before the loop moves the stream by one batch. With 16 batches from a second generator the question does not arise, and the batches are those of check 2 and of the measurement, so that the last set of the two repeated runs must give the measurement's figures.

5. The runner forms a cosine in single precision (line 129). The measurement's Test 2 shows that this moves a cosine of a tensor of 393,216 entries by up to 0.000016. That is harmless for the tables and matters for check 2 (defect 7).

**Repair.** Point 5 of the runner (given under defect 6) records, in every run, the mean cosines over 16 batches under each of the four scales; the reading is fixed in "How the result is stated" (given under defect 5); the run under back-propagation is in "Runs" (defect 2).

### 5. The words *supported* and *not supported*: in four outcomes the report would say something that the runs do not show, or nothing where they show something

**What is wrong.**

1. If the network with a limited input learns under `row`, brief line 108 writes *supported* for a sentence about cells; the runs cannot assign the effect to the cells (defect 1).
2. The words rest on "learns the family" within 24,576 updates. A run that learns at the sweep's usual distance behind back-propagation is *partly*, and for *partly* brief line 108 gives no word (defect 2).
3. Brief, line 108, contradicts itself in one outcome. "It is *not supported* if that run is *not learned*" and "if only the scale `rowcol` learns the family, the report gives the figures without either word" both apply when the run with `row` is *not learned* and the run with `rowcol` learns.
4. No rule covers the run with `matrix`. With a crossing that check 1 has shown equal bit for bit, on the same library and the same kind of processor, that run must repeat the scored rows of E49's run. The brief neither says so nor says what the words mean if it does not, or if, at a greater length, it learns.
5. Where *supported* is licensed, the sentence it is attached to states a cause ("is not learned because one scale … misstates the sizes"). A run that learns under another scale shows that the scale was sufficient on this seed. It does not show what happened in E49's run.
6. The three words for a run (*learns the family*, *partly*, *not learned*) are decidable and cover every outcome: a run in which A reaches 80 and R does not has not learned the family and is *partly* or *not learned* by its taught pairs. But "the level of the taught pairs" is not defined. The brief defines the scores A and R (line 15) and uses a third score from line 30 on without saying what it is. In the receipts it is the entry `taught` of a row: the percentage of right answers on 600 taught pairs of addition and subtraction written with 12 letters (`exp42_cells.py`, lines 225 and 241).
7. `describe` takes four arguments (`e42_tables.py`, line 63). The brief gives the words for the case in which both runs reach 80 and not the two doses that the function needs when one does not. With Dr = Dc = 24,064, a run that does not reach 80 is *more* against E42's references (24,064 / 13,312 = 1.81), which the report must be able to say for the network with limited connections and for the dense one.
8. No measure of the scatter between seeds is given for the two networks in which a number of updates is compared. E42 has three seeds of both (`APP receipts`): under the sweep the network with limited connections reaches 80 on A at 12,544, 13,056 and 13,312 updates and on R at 14,336, 14,336 and 15,104, a range of 768 updates.

**Repair.** Add to brief line 15, after the definition of score R: "Score T is the percentage of right answers on 600 taught pairs of addition and subtraction written with 12 letters (the entry `taught` of a row of a receipt); 'the taught pairs' in this brief means score T." Replace the section "How the result is stated" (brief, lines 101 to 108) by:

> ## How the result is stated (fixed beforehand)
>
> - **Words for one run.** A run has the word *learns the family* if A and R both reach 80 within the run and their levels at the end are both at or above 80. A run that does not learn the family has the word *partly* if the level of T at its end is 50 or more, and the word *not learned* otherwise. Every run has exactly one of the three words. For a run of 49,152 updates the report gives the word twice: for the whole run, and for its first 24,576 updates taken as a run, which is the length of E49's runs (the levels are then the means of the scored rows of updates 24,064, 24,320 and 24,576).
> - **Words for two numbers of updates.** `describe(xr, xc, Dr, Dc)` of `e42_tables.py` (`$W:$M/e42_tables.py`, lines 63 to 72) is called with xr the reference's updates to 80 and xc the run's (`None` where 80 is not reached), and with Dr and Dc the two runs' numbers of updates less 512. Where both reach 80, xc / xr below 2/3 is *fewer*, above 1.5 *more*, otherwise *about the same*; where one does not, the function's words *more*, *fewer*, *not decided* and *neither reaches* are used as it returns them.
> - **The network with a limited input.** For each of the four runs under the sweep: its two words; its updates to 80 on A and on R beside those of E49's run under back-propagation (16,640 and 20,480), in the words of `describe`; its levels of A, R and T at update 24,576 and at the end.
> - **The network with limited connections.** Updates to 80 on A and on R against E42's run under the sweep (13,312 and 15,104) and against its run under back-propagation (8,704 and 9,472), in the words of `describe`; the levels at the end. E42's three seeds give the range within which one seed cannot be read: under the sweep A reaches 80 at 12,544, 13,056 and 13,312 updates and R at 14,336, 14,336 and 15,104. A difference from E42's run under the sweep of 768 updates or less is reported as within the range of the seeds.
> - **The dense network.** The same against E42's runs (5,632 and 6,656; 5,120 and 6,400); the three seeds under the sweep reach 80 on A at 4,864, 5,376 and 5,632 and on R at 4,864, 6,400 and 6,656. This run tests whether the scale `row` does harm where the scale `matrix` works. It is not a control for the explanation.
> - **The cosines during the run,** for every run, as a table by update under each of the four scales on the run's weights. For the run of the network with a limited input with the scale `matrix`, and for its run under back-propagation, the report gives the first reading at which the cosine of `l1` under the scale `matrix` is below 0.5, and the mean training loss of the scored row at that update. If the first such reading of the run under the sweep comes 4,096 updates or more before that of the run under back-propagation, the report says that the cosine fell earlier in the run that fails; otherwise it says that the cosine of `l1` under the scale `matrix` does not separate the run that fails from the run that learns. In neither case does the report say that a low cosine is the cause of the failure; the runs under the other scales are the only test of that.
>
> **The explanation under test** has two parts. Part 1, the cells: under the sweep the network with a limited input does not learn the family because one scale for the whole input map misstates the sizes of the incoming weights of the gates of the cells, which training makes unequal, so that the error that reaches the code is far from the exact error in direction; a scale for each gate of each cell removes this. Part 2, the readout: one scale for the whole of `lm` and one for the whole of `l3` misstate in the same way the sizes of the incoming weights of the readout units and of the output classes, through which the error reaches the cells. The run with `rowcells` tests part 1 alone; the run with `row` tests the two parts together. The table uses the word of each run over its 49,152 updates.
>
> | Outcome in the network with a limited input | What the report says, for seed 73129 |
> |---|---|
> | The run with `matrix` learns the family | E49's run was late and had not failed. No word is given to the explanation; the scales are compared by their updates to 80 |
> | `matrix` does not learn it; `rowcells` learns it | Part 1 is *supported*: a scale for each gate of each cell was sufficient. If the run with `row` does not learn it, the report adds that the scales at the other maps took the gain away |
> | `matrix` and `rowcells` do not learn it; `row` learns it | Part 1 alone is *not supported*. A scale for each receiving unit at every map was sufficient; the runs do not say which of the scales at `lm`, `l3`, `l2` and `l1` is needed beside that of the cells |
> | The runs with `rowcells` and with `row` both have the word *not learned* | The explanation is *not supported*. If the run with `rowcol` learns the family, the report says so beside the word |
> | Any other outcome | The figures, without either word |
>
> A word says that a scale was or was not sufficient on this seed. It does not say that the misstated sizes caused the failure of E49's run. The run with `matrix` must repeat the scored rows of E49's run under the sweep to update 24,576, and the run under back-propagation those of E49's run under back-propagation; if one does not although check 1 passed with equal hashes, the worker reports this first and no word is given. A scale that needs *fewer* updates than the scale `matrix` in the network with limited connections is reported as such; it concerns the larger number of updates that the sweep needs there (requirement B1 of the plan) and not the explanation.

### 6. The runner cannot be built from the six points: the copy stops for `G`, one name stands for two functions, and the scale `exact` does not exist

**What is wrong.** `exp49_interface.run` was read line by line against the brief's points 1 to 6.

1. **`G` is not a setting of E49's runner.** Runner line 165 reads `SETTINGS[setting]` and line 184 asserts `cnt["multiply_adds"] == MA_EXPECTED[setting]`; neither table has `G` (lines 46 and 47), and `main` asserts `setting in SETTINGS` (line 368). Brief line 86 builds the network of `G` and says nothing about these three lines. A copy with "the new forward …, the scale …, `experiment="E57"`, and the name" (brief, line 87) stops with a `KeyError`. The count for `G` is 1,118,720 multiply-adds a position: 397,824 for the code, `l2` and `l3`, 524,288 for the two full maps of the interface and 196,608 for the recurrent map (`e49_interface.py`, lines 49 to 53).
2. **One name, two functions.** The runner defines `magnitudes(m, layer, iface)` at line 106, the mean magnitude of the kept weights of each masked map, and `run` calls it at every scored row (line 213). The brief names its new function `magnitudes(W, maskf, scale)` (line 69). In a file that holds both, the later definition replaces the earlier, and line 213 of the copied text calls the new function with a network, a layer and an interface.
3. **The scale `exact` is a comment.** Brief, line 78: `return row * col / mean  # "rowcol"; "exact" returns A itself and is used by check 2 only`. The code has no branch for `exact`; any name other than `matrix` and `row` reaches the last line. Transcribed to numpy and given `exact`, the brief's code returns magnitudes that differ from |W| by up to 2.78 at `l3` (`APP weights`), so that the last sentence of check 2 would test the scale `rowcol` and fail. The row means are also divided by an unclamped count while the column means are clamped. No row of the three networks is without a kept entry (16 and 11 in every row; `APP weights`), so nothing changes here, but a row without one would give a division of zero by zero and a product of zero with not-a-number in the backward pass.
4. **The recurrent crossing.** `e49_interface.forward` has six crossings and no seventh: `l1` (its line 68), the input map (line 71, two branches), the recurrent map at every position (line 76), `lm` (line 84, two branches), `l2` (line 85) and `l3` (line 86). The brief's "each of the six crossings a call of `cross`" is complete. But line 76 is `E42.rec(layer, m, h, Wh, bh, local, scale)`, and `rec` does two things that the brief does not mention (`e42_cells.py`, lines 128 to 135). It takes a scale that line 73 has computed once for the forward, `layer.sign_scale(m)`; that line has no place in the copy. And for `G`, where `layer.full` is true, it ignores the scale and calls `lin`, that is `SignLinear` with the scale `matrix`. A worker who keeps the call of `rec` and hands it the new magnitudes trains the dense network with the scale `matrix` at the recurrent map under the name `row`, and no check of the brief would show it: check 1 uses the scale `matrix`, and check 2 uses the network with a limited input. The brief should say that `rec` is not called. Its `cross` also forms the magnitudes anew at each of the 22 positions of a batch, where E49's forward forms its scale once; the values are the same and the cost is not. The mask `layer.mask3().float()` has the right shape (768 by 256) and is all ones for `G`, since `Layer` with N = 255 sets every entry of its mask; the clause "or with a mask of ones when `layer.full` is true" changes nothing.
5. **Sentences of the receipt that would be false.** Runner lines 196 and 198 write into every receipt that the update and the forward are `e49_interface.forward`. Line 240 reads the variable `E49_BESIDE`. Lines 176, 218 and 242 print `e49`.
6. **`main`** (lines 362 to 387) has no argument for the scale, and the folders are `e49` and `e49_qual` (lines 41 and 42). The brief gives the names of the receipts and no command.
7. **The cosines.** The runner records `cosine_first_batch_start` and `cosine_first_batch_end` for three tensors through `I49.forward` (lines 112 to 131, 224 to 226 and 233). The brief's point 5 needs six tensors, whose parameter names it does not give (`book.weight`, `l1.weight`, `rnn.weight_ih_l0`, `rnn.weight_hh_l0`, `lm.weight`, `l2.weight`), and does not say what becomes of the two existing entries or where in the loop a set is taken.
8. **The name of a run under back-propagation.** The pattern of brief line 87 has `_S_` and a scale in it; check 3 is a run under back-propagation, where no scale acts.
9. **`e42_tables.py`.** Brief, line 65: "is on `$W` only". It is on `$W9` as well, with the same content (blob 7ef04749); it is not on the default branch.
10. **What the brief assumes rightly.** E49's runner saves the starting weights (line 190, `<name>_start.pt`) and refuses every output path that exists (line 172). It also writes `<name>_answers.json` (line 236), which the brief's list of committed files leaves out.

**Repair.** Replace the section "The runner" (brief, lines 63 to 89) by:

> ## The runner
>
> The runner is a new file, `exp57_scale.py`, in `$M`; no existing source file is edited. `exp49_interface.py` (SHA-256 `cdf93ce167d562c8a3765c1ea4f98f74623469492324a85aa1eb34216b7ccbf3`) and `e49_interface.py` (`92f968e092f2d27eaff4ee7e93ecbbb82b053cc88b6a2d3f967adadf7e3791f1`) are not on the default branch; they are copied byte for byte from `$W9` at a36de661 into `$M` (`git show $W9:$M/<file> > $M/<file>`), with their SHA-256 and source recorded in a text file beside them, and the runner asserts the two hashes when it is imported. `e42_cells.py` and `exp42_cells.py` are on the default branch with the content they have on `$W9` and are used from there. `e42_tables.py` is on `$W` and on `$W9` and not on the default branch; its function `describe` is copied into the script that makes the report's tables, `e57_tables.py`, with its source named. The runner, the two copied files and the text file are committed before the first check is run. The runner imports `exp49_interface` as `R49`, `e49_interface` as `I49`, `exp42_cells` as `R42` and `e42_cells` as `L42`, and is called from `$M` as
>
>     python3 exp57_scale.py run <in16|F10|G> <matrix|row|rowcells|rowcol|none> <S|B> 73129 [--dose N] [--out DIR]
>     python3 exp57_scale.py check <1|2|3|5> [--qual DIR]
>
> The scale `none` is given with rule B and with no other rule.
>
> 1. **The crossing.** `L42.MaskedSign.apply(x, W, b, maskf, S)` computes `linear(x, W * maskf, b)` and, backward, multiplies the error by `(W * maskf).sign() * S`. `S` may be a tensor without dimension, of one column or of the shape of W, since the product is formed entry by entry; the gradients of the weights and of the bias do not use it. The new file defines
>
>        SCALE_AT = {"matrix": ("matrix", "matrix"), "row": ("row", "row"), "rowcells": ("row", "matrix"),
>                    "rowcol": ("rowcol", "matrix"), "exact": ("exact", "exact"), "none": ("matrix", "matrix")}
>        # the scale of a run: (form at the input map and at the recurrent map, form at l1, l2, lm and l3)
>
>        def scale_magnitudes(W, maskf, form):     # W detached; maskf of W's shape, ones where the map is dense
>            A = W.abs() * maskf
>            if form == "exact":                   # check 2 only: the sweep is then back-propagation
>                return A
>            mean = A.sum() / maskf.sum()
>            if form == "matrix":
>                return mean
>            row = A.sum(1, keepdim=True) / maskf.sum(1, keepdim=True).clamp(min=1)
>            if form == "row":
>                return row
>            assert form == "rowcol", form
>            col = A.sum(0, keepdim=True) / maskf.sum(0, keepdim=True).clamp(min=1)
>            return row * col / mean
>
>        def cross(x, W, b, maskf, local, S):
>            return L42.MaskedSign.apply(x, W, b, maskf, S) if local else linear(x, W * maskf, b)
>
>    The function is not named `magnitudes`, because `exp49_interface.py` has a function of that name (its line 106), which the run calls at every scored row.
> 2. **The forward** is `I49.forward` with every crossing through `cross` and with the magnitudes of each map formed once in a forward, as `I49.forward` forms its scale once (its line 73, which is not copied):
>
>        def forward(m, layer, iface, x, local, scale="matrix", silence=False):
>            assert layer.K == 1
>            cells, others = SCALE_AT[scale]
>            mag = lambda W, mk, form: scale_magnitudes(W.detach(), mk, form) if local else None
>            def dense(v, lay):                    # l1, l2, l3: a mask of ones
>                mk = ONES(lay.weight)
>                return cross(v, lay.weight, lay.bias, mk, local, mag(lay.weight, mk, others))
>            z1 = T.tanh(dense(m.book(x), m.l1))
>            B, L = x.shape
>            Wi, bi, mi = m.rnn.weight_ih_l0, m.rnn.bias_ih_l0, iface.mask_in.repeat(3, 1).float()
>            gi = cross(z1, Wi, bi, mi, local, mag(Wi, mi, cells))
>            Wh, bh, mh = m.rnn.weight_hh_l0, m.rnn.bias_hh_l0, layer.mask3().float()      # mh is all ones for G
>            Sh = mag(Wh, mh, cells)
>            h = T.zeros(B, I49.U); hs = []
>            for t in range(L):
>                gh = cross(h, Wh, bh, mh, local, Sh)      # never L42.rec: for G it calls SignLinear and ignores the scale
>                i_r, i_z, i_n = gi[:, t].chunk(3, 1); h_r, h_z, h_n = gh.chunk(3, 1)
>                r = T.sigmoid(i_r + h_r); z = T.sigmoid(i_z + h_z); n = T.tanh(i_n + r * h_n)
>                h = (1 - z) * n + z * h
>                hs.append(h)
>            Hs = T.stack(hs, 1)
>            if silence:
>                Hs = T.zeros_like(Hs)
>            mo = iface.mask_out.float()
>            out = cross(Hs, m.lm.weight, m.lm.bias, mo, local, mag(m.lm.weight, mo, others))
>            z2 = T.tanh(dense(z1, m.l2) + out)
>            return dense(z2, m.l3), hs
>
>    `ONES(W)` returns a tensor of ones of W's shape, made once for each shape. Under back-propagation the forward computes what `I49.forward` computes. This text was written without the training library and has not been run; check 1 is its test.
> 3. **The networks.** `NETWORKS = {"in16": (16, 256), "F10": (512, 256), "G": (512, 256)}`. `in16` and `F10` are built by `R49.build`. `G` is built as `m, layer = R42.build("G", seed)` with `iface = I49.Interface(m, seed, 512, 256)`, whose two masks are full; the run asserts that `tensors_sha(m.state_dict())` is the same before and after the construction of the interface and that `layer.mask3()` is true everywhere.
> 4. **The run** is a copy of the text of `R49.run` (lines 164 to 243 of `exp49_interface.py`) with the changes below and no others; the report shows the difference between the two texts.
>    - Line 164: `run(network, scale, rule, seed, out, cmd, dose=None)`; the run asserts `(scale == "none") == (rule == "B")`.
>    - Line 165: `k_in, k_out = NETWORKS[network]`.
>    - Lines 159 to 161 and 169: the receipt is `$M/exp45/e57/e57_<network>_<scale>_<rule>_73129.json`, with `_u<N>` before the seed for a run of N updates other than 24,576 (for example `e57_in16_row_S_u49152_73129.json`); the other files of a run keep E49's endings (`_start.pt`, `.pt`, `_answers.json`).
>    - Line 174: the network of point 3.
>    - Line 180: the runner's own `install`, which is E49's with the new forward. Scoring calls the forward with `local=False`, so that the scale plays no part in it.
>    - Line 184: the expected multiply-adds a position are `{"in16": 549632, "F10": 930560, "G": 1118720}`.
>    - Lines 191 to 201: `experiment="E57"`; `setting` and `config` hold the network's name; the new entries `scale=scale` and `scale_at=dict(recurrent_layer=SCALE_AT[scale][0], other_maps=SCALE_AT[scale][1])`; the entries `update` and `forward` name `exp57_scale.forward(m, layer, iface, x, local, scale)` in place of `e49_interface.forward`.
>    - Line 213: `kept_mean_magnitude=R49.magnitudes(m, layer, iface)`.
>    - Lines 176, 218 and 242: the printed lines begin with `e57` and give the scale.
>    - Lines 224 to 226 and 233: the entries `cosine_first_batch_start` and `cosine_first_batch_end` are not written and the first batch is not kept; point 5 takes their place, and its first set is taken after the scored row of update 0 (line 221).
>    - Line 228: the runner's own `step`, which is E49's with the new forward and the scale.
>    - Line 231: at every multiple of 2,048 updates the set of cosines of point 5 is appended before the receipt is written.
>    - Line 240: the variable `E57_BESIDE` in place of `E49_BESIDE`.
>    - `provenance` (lines 247 to 253): `sources_sha256` covers `exp57_scale.py` and the files of `R49.SOURCES`; `runner_sha256` is that of `exp57_scale.py`.
>    - `main` (lines 362 to 387): the two commands above; the folders are `X.OUT / "e57"` and `X.OUT / "e57_qual"`.
> 5. **Cosines during the run.** Before the first update and after every 2,048 updates the receipt records, in a list `cosines`, the update and, for each of the four scales `matrix`, `row`, `rowcells` and `rowcol`, the mean over 16 batches of the cosine between the sweep's gradient under that scale and the exact gradient, for the byte table (`book.weight`), `l1` (`l1.weight`), the input map (`rnn.weight_ih_l0`), the recurrent map (`rnn.weight_hh_l0`), `lm` (`lm.weight`) and `l2` (`l2.weight`). Each cosine is taken over all entries of the tensor; outside a mask both gradients are zero, so that this is the cosine over the kept entries. The 16 batches are the second to the seventeenth batch of the stream (the batches of updates 2 to 17). They are drawn once, before the run, from a second generator `E.batches(tp, seed, E.KMAX["M"])`, so that the run's own stream is not advanced; they are the batches of the reviewer's measurement and of check 2. Each cosine is formed in double precision (`cosine_similarity(a.double().flatten(), b.double().flatten(), 0)`). A set is five passes forward and backward on each of the 16 batches. Every gradient is set to `None` before and after a set and no step of the optimizer is taken; at the first set the run asserts that `tensors_sha(m.state_dict())` and the optimizer's running averages are the same before and after it. The CPU time of the sets is recorded in an entry `cpu_cosines`, apart from that of the updates and of the scoring. The run under back-propagation records the same four scales on its weights.
> 6. **Files committed:** the receipts, the starting and the final weights and the answers file of every run (the weights added with `git add -f`, since `.gitignore` excludes `*.pt`), and the logs. The copy of E49's weights that check 2 reads is written outside the repository and is not committed.

### 7. The checks: the fallback of check 1 cannot be met, hashes that are on file are not used, check 2 cannot load its file as worded, and its last tolerance is marginal

**What is wrong.**

1. **The fallback of check 1** (brief, line 95): "If a hash differs while every count and the batch digest agree and every loss agrees to 1e-6, the worker reports … and goes on." Two runs whose crossings differ in rounding move apart from the first update. The reviewer's note of 11 November states the size: "rounding differences from the first update are fully grown" by update 4,096 (`$R/REVIEW_AND_NEXT_STEPS_NOTE_20261111.md`, line 95). At update 2,048 the mean losses of eight rows cannot be expected to agree to 1e-6, nor every count on 600 and on 2,000 items. The fallback can therefore only fail: where the hashes are equal it is not needed, and where they differ its condition is not met. It also does not answer what the worker needs to know at that point, namely whether the crossing differs by rounding or by rule. A comparison of the gradients on one batch at the start answers that, and costs seconds.
2. **Reference hashes on file are not used.** E49's check 6 holds a dry run of 256 updates of the network with a limited input under each rule by E49's runner, with `tensors_sha256` `88cad76a…` (sweep) and `c0eb6548…` (back-propagation), and every receipt of E49 holds `start_tensors_sha256` (`APP receipts`). The brief's check 1 for this network and its check 3 compare losses, counts and digests only. The hash of the network with limited connections is given by its first eight digits.
3. **Check 2 cannot load the file as worded** (brief, line 96: the weights "are loaded into a network built by `exp49_interface.build`"). The file holds 21 tensors: the thirteen of the network and `mask`, `made`, `periods`, `phases`, `mask_in`, `mask_out`, `k_in`, `k_out` (`APP weights`; runner, lines 146 to 151). `load_state_dict` refuses a dictionary with keys that the network does not have. The load is well defined once the thirteen are selected: the masks of the file equal the draws that `build` makes (`APP weights`).
4. **"Batches 2 to 17 of the stream"** counts from 1 in the measurement (`N_FURTHER = 16  # batches 2 to 17 of the stream; batch 1 is the receipts' batch`, script line 48; the tables use `rows[1:]`, line 679). The brief does not say where its count starts, and a worker's list of batches starts at 0. Indices 2 to 17 are 16 other batches, and a mean over them need not lie within 0.002.
5. **"1 to within 1e-5"** for the scale `exact`. With equal gradients a cosine formed in single precision still differs from 1 by the error of two sums of squares. The measurement's Test 2 found such errors of 8.4e-07 to 2.1e-06 for tensors of 131,072 and 196,608 entries and up to 1.6e-05 for 393,216 entries; `l2` has 262,144. The bound is of the size of the error it must exceed. In double precision the bound can be 1e-6.
6. **The tolerance of 0.002 is sound.** The references are rounded to three decimals (an error of up to 0.0005) and single precision adds at most 0.00002 (Test 2). The scales differ by far more at `l1` (0.298, 0.624, 0.527 and 0.721), so a wrong scale fails the check. The eighteen reference figures are those of the script (`APP figures`).
7. **Check 4** is carried out inside every run, while brief line 93 says that each check writes a receipt before the runs and brief line 145 counts "the receipts of the five checks".
8. **Check 5** has a limit of 10 CPU-hours for an estimate of 7. With the changes of this document the estimate is about 12. E54's worker found its host slower than E42's (`LOG.md`, entry of 23:15), so the forecast can exceed the estimate for that reason alone.
9. **An interrupted check.** The dry runs of a check refuse existing paths as the runs do (runner, line 172); brief line 133 gives the rule for an interrupted run only.

**Repair.** Replace the section "Checks before the runs" (brief, lines 91 to 99) by:

> ## Checks
>
> Checks 1, 2, 3 and 5 are made before the runs. Each writes a receipt `$M/exp45/e57_qual/check<n>_73129.json` with an entry `pass`, and its dry runs under `$M/exp45/e57_qual/check<n>_73129_runs/`. Check 4 is made inside every run and check 6 after two of them. No run is started before checks 1 to 3 have passed. If a check fails, the worker stops, reports the first difference and asks; the one exception is stated in check 1.
>
> 1. **With the scale `matrix` the new forward is the old rule.**
>    (a) *At the start, on the first batch of the stream,* for each of `G`, `F10` and `in16` and under both rules: the logits and the gradients of the thirteen tensors through the new forward with the scale `matrix`, against those through the old forward (`R42.forward(m, layer, x, local)` for `G` and `F10`; `I49.forward(m, layer, iface, x, local)` for `F10` and `in16`), compared with `torch.equal`. Where two are not equal, the check records the largest difference of the tensor over the tensor's largest entry.
>    (b) *Dry runs under the sweep with the scale `matrix`.* `G`, 2,048 updates: `tensors_sha256` is `79618ab29390e6ecfdacd49a955afff7a70f282466dd92fd14e6ef3ba1ddd19f` (`DRY2048["S"]` in `exp42_cells.py`), and the scored rows equal those of `$W:$M/exp45/e42_qual/check1_73129_runs/e42_G_S_dry2048_73129.json` in the loss, in every count and in the batch digest. `F10`, 2,048 updates: `start_tensors_sha256` is `dede71d76cb8a9e6af09aa5394eb90c69a45177f6d3dc1861cce0c3b7e35a561` and `tensors_sha256` is `837dadc12e8b1bd702c5d602b80e3fdca82eceda478dd110b480aff57815ea75` (`$W:$M/exp45/e42_qual/check8_73129_runs/e42_F10_S_dry2048_73129.json`, and the run of E49's runner in `$W9:$M/exp45/e49_qual/check2_73129_runs/`). `in16`, 256 updates: `start_tensors_sha256` is `657e185828eb7a4c6e91a3c3a0bae71d6d0c8d38cd84840a66c3924834ff45da` and `tensors_sha256` is `88cad76a4806601be9a859832b9167b95a967fb3cac4a2f7ee8de8b641cb6f34` (`$W9:$M/exp45/e49_qual/check6_73129_runs/e49_in16_S_dry256_73129.json`). `in16`, 2,048 updates: the scored rows equal those of `$W9:$M/exp45/e49/e49_in16_S_73129.json` to update 2,048 in the loss, in every count and in the batch digest.
>    (c) If every comparison of (a) is equal, (b) must hold in full. If (a) is not equal bit for bit but no recorded difference exceeds 1e-6, and a hash of (b) differs while the batch digests agree, the worker reports that the new crossing equals the old one to rounding and not bit for bit, with the differences of (a) and the scored rows of both runs side by side, and goes on; the report then says that the references are not repeated bit for bit and that the run with the scale `matrix` is this job's own reference. Any other difference stops the job. (Two runs that differ in rounding at the first update have moved apart by update 2,048, so the condition is placed on the gradients at the start and not on the losses.)
> 2. **The scales give the gradients of the reviewer's measurement.** A network is built by `R49.build("in16", 73129)`. The file `$W9:$M/exp45/e49/e49_in16_S_73129.pt` is written with `git show` to a place outside the repository and read with `torch.load`. It holds the thirteen tensors of the network and, beside them, `mask`, `made`, `periods`, `phases`, `mask_in`, `mask_out`, `k_in` and `k_out`. The check asserts that `mask`, `mask_in` and `mask_out` equal `layer.mask`, `iface.mask_in` and `iface.mask_out`, and loads the thirteen (`m.load_state_dict({k: f[k] for k in m.state_dict()})`). With the function that records the cosines during the runs (point 5 of the runner: the second to the seventeenth batch of the stream, double precision) the mean cosines are, within 0.002: with `matrix`, byte table 0.367, `l1` 0.298, input map 0.603, recurrent map 0.651, `lm` 0.606, `l2` 0.643; with `row`, 0.697, 0.624, 0.676, 0.762, 0.694, 0.710; with `rowcells`, 0.473, 0.527, 0.594, 0.654, 0.606, 0.643; with `rowcol`, 0.638, 0.721, 0.714, 0.710, 0.606, 0.643. With the scale `exact` the cosine of each of the thirteen tensors of the network is 1 to within 1e-6. The same is done with `$W9:$M/exp45/e49/e49_in16_B_73129.pt` for `matrix` (0.442, 0.309, 0.523, 0.582, 0.700, 0.706) and for `row` (0.405, 0.363, 0.567, 0.631, 0.700, 0.707). (The figures are those of `$V:$R/sweep_scale_variants.py`, a second implementation in numpy and double precision, for its networks `in16_S` and `in16_B`: variant V0, V2 at every map, V2 at the two GRU maps and V4 at the two GRU maps.)
> 3. **Back-propagation is unchanged.** A dry run of 256 updates of `in16` under back-propagation through the new forward has `tensors_sha256` `c0eb654815c62c9a6e3826b50a3e8044b12f776519d9a6278888178913a0b5b0` (`$W9:$M/exp45/e49_qual/check6_73129_runs/e49_in16_B_dry256_73129.json`), and its two scored rows equal the first two of `$W9:$M/exp45/e49/e49_in16_B_73129.json` in the loss, in every count and in the batch digest. If the hash differs, rule (c) of check 1 applies.
> 4. **The masks hold** (E49's check 3) at every scored row of every run: every weight outside a mask is zero, and the optimizer's two running averages there are zero. A failure stops the run. The record is the entry `masks` of every scored row; after the runs the worker writes `check4_73129.json`, which says for each run whether the entry is true at every row.
> 5. **Cost:** CPU-seconds an update, a scored row and a set of cosines, from the dry runs of check 1, and a forecast for the eight runs. If the forecast exceeds 18 CPU-hours, the worker asks before starting.
> 6. **The two repeated runs,** when they have ended. The run of `in16` with the scale `matrix` has, at every scored row to update 24,576, the loss, the counts and the batch digest of E49's run under the sweep. The run under back-propagation ends with `tensors_sha256` `707c85693a98b6b1017ba94abb0754b12758650d8a0d6562d44a28363584ac3e`, E49's. Their cosines at update 24,576 under the scales `matrix` and `row` are the figures of check 2, within 0.002. The worker writes `check6_73129.json`. If check 1 passed with equal hashes and check 6 fails, the worker reports this before any other result.
>
> An interrupted check is made again from its beginning, after its folder of dry runs has been moved to `$M/exp45/e57_qual/interrupted/`.

### 8. The predictions

**What is wrong.**

1. P57.5 and P57.6 are decidable only with the two doses that `describe` needs (defect 5, point 7): a run that does not reach 80 must have a word.
2. P57.7 is read on one batch (defect 4, point 3), and "before update 8,192" leaves open that the readings are those of updates 2,048, 4,096 and 6,144.
3. P57.2 and P57.3 are tied to 24,576 updates. They then predict, at 35% and 30%, that the sweep needs at most 1.17 times the updates of back-propagation on R; on the six pairs of runs of E42 that ratio is 1.04 to 1.81 and is at or below 1.17 only in one seed of the dense network (defect 2). This is the reviewer's forecast and not an error of logic, but the statement that the reviewer most likely means, that the network learns under the scale at all, is not among the predictions.
4. The added runs have no prediction.
5. No prediction contradicts the record or another: P57.2 implies P57.4, and 35% is below 50%.

**Repair.** Replace the table of predictions (brief, lines 112 to 120) by the table below. The chances of P57.1 to P57.7 are the brief's; the others are the reviewer's to set before any run.

> | | Statement | Chance |
> |---|---|---|
> | P57.1 | Checks 1 to 3 pass at the first attempt, with every hash equal | 75% |
> | P57.2 | Under the scale `row` the network with a limited input has the word *learns the family* for its first 24,576 updates | 35% |
> | P57.3 | Under the scale `rowcol` the network with a limited input has the word *learns the family* for its first 24,576 updates | 30% |
> | P57.4 | Under the scale `row` the level of T of the network with a limited input at update 24,576 is 50 or more | 50% |
> | P57.5 | In the network with limited connections, `describe(13312, x, 24064, 24064)` is *about the same*, x being the updates to 80 on A under the scale `row` | 65% |
> | P57.6 | In the dense network, `describe(5632, x, 24064, 24064)` is *about the same*, x being the updates to 80 on A under the scale `row` | 85% |
> | P57.7 | In the run of the network with a limited input under the scale `matrix`, the cosine of `l1` under the scale `matrix` (mean of 16 batches) is below 0.5 at one of the readings of updates 2,048, 4,096 and 6,144 | 60% |
> | P57.8 | Under the scale `row` the network with a limited input has the word *learns the family* for its 49,152 updates | to be set |
> | P57.9 | Under the scale `rowcells` the network with a limited input has the word *learns the family* for its 49,152 updates | to be set |
> | P57.10 | Under the scale `matrix` the network with a limited input has the word *not learned* for its 49,152 updates | to be set |
> | P57.11 | In the run under back-propagation the cosine of `l1` under the scale `matrix` is below 0.5 at the reading of update 12,288 or at an earlier one | to be set |

### 9. One seed and one length are not stated where the brief first makes its claim, and no range between seeds is given

**What is wrong.** Brief, line 21: "Under the sweep the network with a limited input does not learn the first family at all". The statement rests on one run of 24,576 updates on seed 73129, in which A, R and the score on the taught pairs stayed below 5 at every scored row (the largest are 3.33, 2.25 and 4.17; `APP receipts`). "At all" says more than that. The limits stand at brief line 27 ("Every figure is for seed 73129") and line 124, after the question and not in it. Two facts on file bear on the claim and are not cited: E49's two other settings with a limited input (16 and 64 inputs a cell, with a limited readout) failed under the sweep in the same way on that seed, and the sweep's greater number of updates in the network with limited connections holds on each of E42's three seeds (1.36 to 1.89 times on A, 1.47 to 1.81 on R; `APP receipts`). The second gives the range between seeds that a difference on one seed must exceed (defect 5, point 8).

**Repair.** Replace the first two sentences of brief line 21 by:

> In E49, on seed 73129, the network with a limited input did not learn the first family under the sweep within 24,576 updates (A, R and T stayed below 5 at every scored row), while it learned it under back-propagation. E49's two other networks with a limited input failed under the sweep in the same way on that seed. The sweep also needs more updates than back-propagation in the network with limited connections, on each of E42's three seeds (1.36 to 1.89 times the updates to 80 on A).

### 10. Locality: the receiving unit is not named for each map, and the sentence on what the rule uses says more than the scale changes

**What is wrong.** Brief, line 14, calls row i "the receiving unit's own incoming weights", and brief line 23 says that the rule "uses only what a unit has". The statement that a scale taken over a row is more local than one taken over a matrix is correct for each of the six maps, but the brief does not say who the unit is, and at three maps this is not what a reader expects.

- At the input map and at the recurrent map the unit is one gate of one cell (reset, update or candidate), not a cell. Each cell has three rows in each of the two matrices and therefore six scales. In `in16_S` the 768 row means of the input map range from 0.004 to 0.554 (`APP weights`).
- A unit that receives through two maps has a scale at each: a gate through the input map and the recurrent map, a readout unit through `l2` and `lm`. Neither is a mean over all of the unit's incoming weights.
- At `l3` the unit is an output class.

The scale `row` removes the one quantity of the sweep that no unit holds, the mean over a whole matrix. It does not change that the error travels back through the signs of the weights, which the sending side must have under either scale; "uses only what a unit has" reads as if the scale settled that.

The brief describes the scale `rowcol` fairly (line 128): it uses a mean over the sending unit's outgoing weights and the mean over the matrix, and is not presented as more local.

**Repair.** The term is replaced under defect 1 (a). Replace brief line 23 by:

> The question belongs to Mr. Xu's requirement that learning be local, at the input as well (requirement C of the plan), and to his requirement on the number of updates (B1). Under the scale `matrix` the error that a unit sends back is multiplied by the mean over all weights of a matrix, a quantity that no single unit holds. Under the scale `row` it is multiplied by the mean of that unit's own incoming weights. The signs of the weights through which the error then travels are used under either scale; the scale does not change that.

### 11. Rules for the worker, reporting and the end of the job: what a worker in a fresh container lacks

**What is wrong.**

1. The branches are named without a commit for `$W` and `$W9` (brief, line 7), and the fetch is given as a pattern (line 134). A later commit on a worker's branch would change what `git show` returns. The heads are 44f91edc (`$W`) and a36de661 (`$W9`).
2. The SHA-256 of the two files to be copied are not given (defect 6).
3. Brief, line 133: "The six runs take about two and a half hours". The brief does not give the time for which the turn must be held from the first check on, and the worker's host may be slower than E49's (E54's worker found that; `LOG.md`, entry of 23:15). With the changes of this document the runs take about four and a half hours and the checks about half an hour.
4. The rule for an interrupted run is right and sufficient for E49's runner, which refuses existing paths and writes `_start.pt` at its start and the receipt at every 2,048 updates. Nothing is said for an interrupted check (defect 7, point 9).
5. Brief, line 145, counts "the receipts of the five checks" and "the six runs"; line 141 "the six runs finished".
6. Rule 4 of section 4 of the plan asks that a result be "verified from receipts and weights before it is read". The brief commits the weights, which makes this possible, and does not say that it will be done.

Nothing else is missing: `.gitignore` on the default branch excludes `*.pt` and the brief says so; numpy is named; `PROJECT_DOCUMENTS/KB/AGENTS_POINTER.md` is on the default branch and prescribes the trailer; the commands run from `$M`, which works because every path of the harness is taken from the place of its file.

**Repair.**

(a) Replace the first rule (brief, line 132) by:

> - Start a branch of your own from `$D` at 7cabb13a. Fetch the three branches that you read, and no other: `git fetch origin '+refs/heads/claude/gallant-dijkstra-i9zgsr:refs/remotes/origin/claude/gallant-dijkstra-i9zgsr' '+refs/heads/claude/peaceful-noether-vcvrmf:refs/remotes/origin/claude/peaceful-noether-vcvrmf' '+refs/heads/claude/vibrant-cerf-sje726:refs/remotes/origin/claude/vibrant-cerf-sje726'`. Read `$W` at 44f91edc and `$W9` at a36de661 (`git show <commit>:<path>`), so that a later commit on those branches changes nothing. Install torch 2.15.0+cpu from `https://download.pytorch.org/whl/cpu` and numpy in a virtual environment under Python 3.13; every receipt records the versions. Run at most three processes at a time, each with one thread (`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `torch.set_num_threads(1)`), from the folder `$M`.

(b) In the second rule (brief, line 133) replace "The six runs take about two and a half hours in two sets of three." by:

> The checks take about half an hour and the eight runs about four and a half hours with three processes at a time; the turn is held for the whole of that time, and longer if check 5 finds this host slower.

(c) Replace brief line 145 by:

> The job is done when the receipts of checks 1, 2, 3, 5 and 6 and the summary of check 4, the eight runs with their receipts, starting and final weights, answers files and logs, and `E57_REPORT.md` exist; the report holds the table of every run beside its references, the readings in the words fixed above, the tables of the cosines during the runs, the outcome of each prediction, the difference between the runner's `run` and `exp49_interface.run`, and the list of committed files.

(d) In brief line 141 replace "the three runs of the network with a limited input" by "the first three runs of the network with a limited input (`matrix`, `row`, `rowcells`)" and "the six runs finished" by "the eight runs finished". In brief line 3 add after the cost: "The result is verified from the receipts and the weights by an agent of the reviewer before the reviewer reads it (rule 4 of section 4 of the plan)."

(e) In "What it cannot show" (brief, lines 124 to 128) replace the second point by the first sentence below and add the two others:

> - The job does not test the scale in the network with a limited input and a fixed code (E50's `C16-in`; see "What the record says"). If a scale helps here, that network is the next to try it on.
> - In the network with limited connections the scales `row` and `rowcol` differ in form and in place at once; a difference between these two runs cannot be assigned to either.
> - The job does not vary the weight decay, under which `l1` falls to a quarter of its starting size in every run on file (the reviewer's verification of E49, section 2.7); whether the failure under the sweep needs the decay is not tested here.

### 12. Figures and statements about the record (small)

**What is wrong, with the repair in each point.**

1. Brief, line 30: "after 24,576 updates A is at 1.9, R at 1.4 and the taught pairs at 2.5". These are the levels, that is the means of the last three scored rows; the last row has 2.17, 1.30 and 3.50 (`APP receipts`). Replace by: "its levels at the end are 1.9 for A, 1.4 for R and 2.5 for T".
2. Brief, line 45: replace "0.749, 0.740" by "0.749, 0.739" and the last cell "0.745, 0.734" by "0.746, 0.734" (defect 3; the table is given in full there).
3. Brief, line 11: "the 257 possible output bytes". The 257 outputs are the 256 byte values and the end symbol. Replace by: "the 257 output classes (the 256 byte values and the symbol that ends an item)".
4. Brief, line 61: "about 1.5 for the checks and the cosines recorded during the runs". With the brief's own checks and cosines the figure is about 0.3 CPU-hours: three dry runs of 2,048 updates at 0.13 CPU-seconds an update and 8.7 a scored row (E49's check 6) are 0.29, the other checks under 0.02, and 13 sets of three passes on one batch are 5 CPU-seconds a run. The estimate errs on the safe side. The figure for the changed job is in the section "Runs" under defect 2.
5. Brief, line 65: "`e42_tables.py` is on `$W` only" (defect 6, point 9; corrected in the section given there).
6. Brief, line 14: "In every run so far". The inventory of the record supports this for E26 to E50 (`INVENTORY_DENSE_LINE_20261117.md`, row 2.10); earlier experiments used other learners. The replacement under defect 1 (a) says "In every run of E26 to E50".

### 13. Language

The brief is written in complete sentences and defines most of its terms. The sentences below fail the standard that the brief sets for the worker's report (line 136): a term that is not defined at first use, a figure of speech, or a word used in three senses.

| Brief | Text | What fails | Replacement |
|---|---|---|---|
| line 21 | "does not learn the first family at all" | more than the record holds | given under defect 9 |
| line 23 | "a rule that uses only what a unit has" | vague | given under defect 10 |
| line 30 | "the code shrinks and the values that a cell reads come to say little about the byte" | figure of speech; no figure | "the mean magnitude of the code falls from 0.41 to 0.05, and the values that a cell reads then differ little from one byte to another: on the median cell the two closest of the 20 bytes that the task must distinguish lie at 0.055 to 0.08 of the mean distance between two of them, against 0.39 to 0.44 at the start (the three runs with a limited input under the sweep)" |
| line 31 | "the cells carry a difference as evenly as in the network with limited connections. The start has nothing to repair." | figures of speech | "a difference between two bytes is spread over the cells about as evenly as in the network with limited connections: the share of the cells that would hold the whole mean square of the difference, if those cells held it equally, is 0.734 against 0.746. That measurement calls for no change of the starting weights." |
| line 108 | "the network with a limited input is not learned"; "if that run is *not learned*" | a network learns and is not learned; a run is not learned either | given under defect 5: "does not learn the family"; "has the word *not learned*" |
| line 108 | "it bears on the sweep's lag there" | coined use of "lag" | given under defect 5: "it concerns the larger number of updates that the sweep needs there" |
| line 125 | "was still learning at the last reading of its running receipt" | no figure, no date | given under defects 3 and 11 |
| lines 15, 59, 95, 97, 98, 141 | "row" for a point of a receipt, beside "row" of a matrix (line 14) and the scale named `row` | one word in three senses | "scored row" wherever a receipt is meant: in line 15, "the scores of one such point are a *scored row*", "three consecutive scored rows", "the last three scored rows" |
| lines 7, 12 | "E42", "E49", "runner" | not defined at first use | the sentence below |
| lines 59, 88, 95 to 97 | "the stream", "the panels", "every count", "the batch digest", `tensors_sha256` | not defined | the sentence below; "the panels" is replaced in the section "Runs" under defect 2 |
| line 12 | "10 fixed cells of its neighbourhood" | not defined | the sentence below |
| line 133 | "its session's turn" | not defined | add: "(a turn is the time in which the worker acts without waiting for a message)" |

Add to "Terms", after brief line 17:

> - E42, E49 and E50 are earlier experiments of the programme. A *runner* is the script that carries out the runs of one experiment. *The stream* is the sequence of batches of 64 taught items that a run is fed, one batch for each update; it is the same for every run of a seed. A *count* is the number of right answers on one of the sets of items on which the scores are taken, and the *batch digest* is the SHA-256 that a scored row records over all batches fed so far. `tensors_sha256` is the SHA-256 of the thirteen tensors of the network at the end of a run, and `start_tensors_sha256` the same before its first update. The *neighbourhood* of a cell is the 48 cells within three steps of it in each direction on a grid of 16 by 16 cells that closes on itself.

## Verified and found sound

### A. Figures of "What the record says" and of "Runs"

- **E42** (`APP receipts`). Updates to 80 on A and on R: dense 5,632 and 6,656 under the sweep, 5,120 and 6,400 under back-propagation; limited connections 13,312 and 15,104, and 8,704 and 9,472. The four runs cost 0.852, 0.820, 0.873 and 0.855 CPU-hours (updates and scoring), which is the brief's 0.82 to 0.87.
- **E49** (`APP receipts`). Under back-propagation A reaches 80 at 16,640 and R at 20,480; the levels are 85.22 and 82.53. Under the sweep the levels are 1.89, 1.37 and 2.50. The two runs cost 0.956 and 0.918 CPU-hours.
- **The table of cosines.** 37 of its 39 figures equal the output of the script as run in this check, to the third decimal (`APP figures`); the two others are in defect 3. In the measurement V2 at every map is the brief's `row`, V4 at the two GRU maps is its `rowcol`, V0 its `matrix`. The eighteen reference figures of check 2 are the script's. "By more than 0.006" (brief, line 47) and "by 0.02 to 0.12" are the document's own statements (its section 1, points 6 and 4).
- **The two other documents.** The verification of E49 has the shrinking code (0.41 to 0.05) and the small distances between bytes in its section 2.7. The document on the passage of a difference has "every cell reads 16 values that differ" and the two shares 0.734 and 0.746 in its summary, and "No change of the starting weights is called for by this measurement" in its section 7.
- **Cost of the six runs.** Three times 0.95 and three times 0.87 are 5.46 CPU-hours.
- **Shell names and host.** `$D` is at 7cabb13a. `exp49_interface.py` and `e49_interface.py` are on `$W9` and not on `$D`. `e42_cells.py` and `exp42_cells.py` are the same blobs on `$D`, `$W` and `$W9`. The receipts of E42 and E49 record Python 3.13.18 and torch 2.15.0+cpu.

### B. The runner

1. **`MaskedSign` with a tensor as scale** (`e42_cells.py`, lines 50 to 65). The forward passes `scale` to `save_for_backward` as one of its own arguments, which is allowed for a tensor of any shape. The backward forms `(weight * maskf).sign() * scale` entry by entry and then `g @` that matrix, so a scale without dimension, of 768 by 1 or of W's shape all work. The weight gradient `(g.T @ x) * maskf` and the bias gradient `g.sum(0)` do not contain the scale. Nothing on the path of the new forward assumes a number: `Layer.sign_scale` is called by `e42_cells.forward`, by `exp42_cells.forward` and by line 73 of `e49_interface.forward`, none of which the new runner uses, and the receipts of E49 record cosines and mean magnitudes, not the scale.
2. **The brief's `magnitudes` is the measurement's V0, V2 and V4.** Transcribed line for line to numpy and applied to E49's final weights of the network with a limited input under the sweep, it gives the measurement's `scale()` with a largest difference of zero on each of the six maps, masked and dense (`APP weights`). The row mean is over the 768 rows of a stacked matrix one by one, the column mean over all 768 rows, and the product is divided by the mean of the matrix, in both. The measurement clamps the count of a row to 1 and the brief does not; no row or column of the input map or of the recurrent map is without a kept entry (16 and 11 in every row, 3 to 54 and 12 to 54 in a column).
3. **Equality bit for bit with the old rule under the scale `matrix`.** This could not be run; the expectation rests on the code and on two results on file.
   - *A mask of ones against `SignLinear`.* The forward is `linear(x, W * ones, b)` against `linear(x, W, b)`, and a product with 1.0 returns the same number. The backward uses the same expression for the error, and its weight gradient is multiplied by ones. The scale is `(W.abs() * ones).sum() / ones.sum()` against `W.abs().mean()`. The sum of at most 393,216 ones is exact in single precision, and the library forms a mean on the processor as the same sum divided by the number of entries, so the two are expected equal. The specification's self-check 8 gives a difference of exactly zero for `MaskedSign` with a mask of ones and the scale `w.abs().mean()` against `SignLinear` (`$D:PROJECT_DOCUMENTS/COORD_CHECKS_20261116/OUTPUT_E42_SPEC_MAC.txt`, line 23); the form with the sum and the count is the one link that it does not cover.
   - *A masked map against E49's `masked` and against `Layer.sign_scale`.* The brief's `A.sum() / maskf.sum()` with `A = W.abs() * maskf` is the expression of `e49_interface.masked` (its line 60) and of `sign_scale` (`e42_cells.py`, lines 113 to 118), term for term.
   - *One call at every position against E42's faster forward.* E49's check 2 ran E49's forward, which calls `MaskedSign` at every position, for 2,048 updates of `F10` and obtained E42's hash `837dadc1…` (`$W9:$M/exp45/e49_qual/check2_73129.json`: `tensors_equal` true under both rules).
   - *The three dry runs.* For `G` the hash can be expected to equal E42's `79618ab2…`; for `F10` it can be expected to equal `837dadc1…`; for `in16` the scored rows can be expected to equal E49's exactly. All three rest on the first point.
4. **The forward.** The six crossings are listed under defect 6, point 4; there is no seventh. `layer.mask3().float()` is 768 by 256. `layer.full` is true for `G` only.
5. **`G`.** `exp42_cells.build("G", seed)` gives `Layer` with N = 255, for which `full` is true, the mask is true everywhere, K is 1 and no weight is rescaled (`e42_cells.py`, lines 71 to 87). `Interface(m, seed, 512, 256)` makes two full masks and multiplies no weight (`e49_interface.py`, lines 43 to 47). The assertion `layer.K == 1` holds. `check_limit` and `check_masks` pass for full masks. With a crossing equal bit for bit, the copied forward computes what `arch26.forward` computes, and the hash of check 1 for `G` is E42's.
6. **`exp49_interface.run`** saves the starting weights and refuses existing paths; the lines that a copy must change are listed in the section "The runner" under defect 6.
7. **Check 2.** The masks in the weight file equal the masks that `build` draws (`APP weights`). The tolerance of 0.002 is sound (defect 7, point 6).
8. **The cosines during the run.** They need the exact gradient and one gradient for each scale recorded. E49's `map_grads` shows the form: every gradient is set to `None`, one backward pass is made, the gradients are copied and set to `None` again; no step of the optimizer is taken, and `step` calls `opt.zero_grad()` before its own backward pass. The weights and the optimizer are therefore untouched. In E49's run the first batch is available before the first step (lines 222 to 226). On one batch a set costs three passes, about 0.4 CPU-seconds.

### C. The design

- The words *learns the family*, *partly* and *not learned* are decidable from a receipt and cover every outcome (defect 5, point 6).
- The dense network is a meaningful test of whether the scale `row` does harm where the present scale works. It is not a control for the explanation.
- E49's network `I16` is rightly left out: under back-propagation it ends at 17 on R, so that a run under the sweep would have no reference that learns (`$R/E49_E43ADD2_VERIFICATION_20261117.md`, section 1).
- E50's `C16-in` under the scale `row` is rightly deferred. Its two runs were at updates 19,456 (sweep) and 22,528 (back-propagation) of 49,152 at commit 8b7d3a89, its runner is a third file on another branch, and a run costs about 2 CPU-hours.
- The brief's statement on the scale `rowcol` (line 128) is fair.

### D. The checks

- `79618ab29390e6ecfdacd49a955afff7a70f282466dd92fd14e6ef3ba1ddd19f` is `DRY2048["S"]` in `$D:$M/exp42_cells.py` (line 414) and the `tensors_sha256` of `$W:$M/exp45/e42_qual/check1_73129_runs/e42_G_S_dry2048_73129.json`.
- `837dadc1` begins the `tensors_sha256` of `$W:$M/exp45/e42_qual/check8_73129_runs/e42_F10_S_dry2048_73129.json`.
- `$W9:$M/exp45/e49/e49_in16_S_73129.json`, `e49_in16_B_73129.json` and `e49_in16_S_73129.pt` exist; the two receipts are complete, with 97 scored rows.
- Checks 2 and 3 can fail when the thing they are named after is wrong; check 4 is E49's and stops a run by an assertion.

### E. The predictions

Each of P57.1 to P57.7 is decidable from the runs once `describe` has its arguments and the readings of P57.7 are named. None contradicts the record or another.

### F. Rules for the worker

The default branch is at 7cabb13a; its `.gitignore` excludes `*.pt`; the pointer file with the trailers exists; the install line names the version that the receipts record; the rule for an interrupted run fits a runner that refuses existing paths.

## Not checked

1. Nothing was run in torch. The equality bit for bit of section B, point 3, is argued from the code and from two results on file; whether `(W.abs() * ones).sum() / ones.sum()` equals `W.abs().mean()` to the last bit in torch 2.15.0 was not executed. The text of the forward in the repair of defect 6 was written without torch and has not been run.
2. That two runs which differ in rounding fail the fallback of check 1 is argued and supported by the reviewer's note of 11 November; it was not measured.
3. The costs are computed from the CPU times that E42's and E49's receipts record on their hosts. The cost of a set of cosines (about 10 CPU-seconds) is 80 passes at the 0.13 CPU-seconds that E49's check 6 records for an update; it was not measured.
4. No branch was fetched. The branches were read as they stood in this clone: `$W9` at a36de661, `$W` at 44f91edc, E50's branch at 8b7d3a89 (22:39 UTC). What E50's runs show after update 19,456 is not known here.
5. The session named in "Reporting" and the tool by which the worker writes to it were not tried.
6. The three documents of the reviewer were read for the statements that the brief draws from them, and the measurement's tables and tests in full; their other sections were not checked.
7. The table of defect 1, point 2, is a measurement on the final weights of one seed over 16 batches, like the measurement it extends. It says where the gain of a cosine comes from on those weights. It does not say what a run under any scale will do.
8. `exp11_cell.py`, `arch20.py` and `exp20_reference.py` were read only at the places cited (the calls of the forward in scoring, the stream, the refusal of existing paths, the hash of the tensors).

## Appendix: the script behind the figures

The script was kept in a folder of its own under the scratch directory while the check ran; it is given here in full. Its parts were run as `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -I -B check_e57_brief.py <part>` from the repository root, with Python 3.13.17 and numpy 2.5.3. The part `figures` takes the output of the measurement's script, which this check produced with `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -I PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/sweep_scale_variants.py > <file>` (258 seconds). The part `where` takes about 90 seconds; the others take seconds.

```python
"""Check of the brief of E57: the figures behind CHECK_OF_TASK_E57_BRIEF_20261117.md.  Read-only; nothing is trained.

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -I -B <this file> <part> [arguments]      (from the repository root)

    receipts            updates to 80, levels and costs of E42's and E49's runs of seed 73129; E42's three seeds; reference hashes
    figures <file>      the brief's cosines against <file>, the output of sweep_scale_variants.py (run beforehand, whole)
    weights             the tensors of E49's weight file of in16 under the sweep; its masks against the draws; rows and columns
                        without a kept entry; the brief's function `magnitudes` (transcribed to numpy) against the measurement's scale()
    where [networks]    the row scale applied at one map at a time (default: in16_S in16_B F10_S G_S)

Files of other branches are read with `git show`; no line of another branch is executed.  The parts `figures` and `where` use
sweep_scale_variants.py of the reviewer's folder on the current branch (`where` imports it; nothing is written beside it).
Standard library and numpy only.
"""
import sys
sys.dont_write_bytecode = True
import json, os, subprocess
from random import Random
import numpy as np

REPO = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=True).stdout.strip()
RV = os.path.join(REPO, "PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109")
M = "MOD97_LOCAL_RULE_20261106"
W, W9 = "origin/claude/gallant-dijkstra-i9zgsr", "origin/claude/peaceful-noether-vcvrmf"


def show(ref, path, text=True):
    return subprocess.run(["git", "show", f"{ref}:{path}"], capture_output=True, check=True, cwd=REPO, text=text).stdout


def reach(curve, key, thr=80.0):
    """The update of the first of three consecutive scored rows at or above thr."""
    v = [(r["step"], r[key]) for r in curve]
    for i in range(len(v) - 2):
        if all(x[1] >= thr for x in v[i:i + 3]):
            return v[i][0]
    return None


def level(curve, key):
    return sum(r[key] for r in curve[-3:]) / 3


# ---------------------------------------------------------------- receipts
def part_receipts(args):
    for ref, path in ((W, "e42/e42_G_S_73129"), (W, "e42/e42_G_B_73129"), (W, "e42/e42_F10_S_73129"), (W, "e42/e42_F10_B_73129"),
                      (W9, "e49/e49_in16_B_73129"), (W9, "e49/e49_in16_S_73129")):
        d = json.loads(show(ref, f"{M}/exp45/{path}.json")); c = d["curve"]
        cpu = (d["cpu_updates"] + d["cpu_scoring"] + d.get("cpu_replacement", 0.0)) / 3600
        print(f"{path}: rows {len(c)}; updates to 80: A {reach(c, 'A')} R {reach(c, 'R')}; taught to 50: {reach(c, 'taught', 50)};"
              f" levels A {level(c, 'A'):.2f} R {level(c, 'R'):.2f} taught {level(c, 'taught'):.2f}; last row A {c[-1]['A']:.2f} R {c[-1]['R']:.2f}"
              f" taught {c[-1]['taught']:.2f}; CPU-hours {cpu:.3f}; torch {d['provenance']['host']['torch']}; tensors_sha256 {d['tensors_sha256']}")
        if "e49" in path:
            print(f"   start_tensors_sha256 {d['start_tensors_sha256']}; multiply-adds {d['counts']['multiply_adds']}; runner {d['provenance']['runner_sha256']}")
            print("   loss at every 2,048 updates:", " ".join(f"{r['step']}:{r['loss']:.3f}" for r in c if r["step"] % 2048 == 0 and r["step"]))
            print("   taught at every 2,048 updates:", " ".join(f"{r['step']}:{r['taught']:.1f}" for r in c if r["step"] % 2048 == 0))
    print("E42, the three seeds (updates to 80 on A and on R):")
    for cfg in ("G", "F10"):
        for rule in "SB":
            out = []
            for seed in (73129, 93683, 116089):
                c = json.loads(show(W, f"{M}/exp45/e42/e42_{cfg}_{rule}_{seed}.json"))["curve"]
                out.append(f"{seed}: {reach(c, 'A')}, {reach(c, 'R')}")
            print(f"   {cfg} {rule}  " + "; ".join(out))
    for ref, path in ((W, "e42_qual/check8_73129_runs/e42_F10_S_dry2048_73129"), (W, "e42_qual/check1_73129_runs/e42_G_S_dry2048_73129"),
                      (W9, "e49_qual/check2_73129_runs/e49_F10_S_dry2048_73129"), (W9, "e49_qual/check6_73129_runs/e49_in16_S_dry256_73129"),
                      (W9, "e49_qual/check6_73129_runs/e49_in16_B_dry256_73129")):
        d = json.loads(show(ref, f"{M}/exp45/{path}.json"))
        print(f"{path}: complete {d['complete']}; tensors_sha256 {d['tensors_sha256']}; start {d.get('start_tensors_sha256', 'not recorded')};"
              f" CPU-seconds an update {d['cpu_updates'] / d['dose']:.4f}")
    src = show("origin/codex/minifly-consolidated-20261023", f"{M}/exp42_cells.py")
    print("DRY2048 in exp42_cells.py holds 79618ab29390e6ecfdacd49a955afff7a70f282466dd92fd14e6ef3ba1ddd19f:",
          '"S": "79618ab29390e6ecfdacd49a955afff7a70f282466dd92fd14e6ef3ba1ddd19f"' in src)


# ---------------------------------------------------------------- figures
BRIEF = {  # network of the measurement: tensor: (matrix, row, rowcol) as the brief's table and its check 2 give them
    "in16_S": {"byte table": (0.367, 0.697, 0.638), "l1": (0.298, 0.624, 0.721), "input map": (0.603, 0.676, 0.714),
               "recurrent map": (0.651, 0.762, 0.710), "lm": (0.606, 0.694, 0.606), "l2": (0.643, 0.710, 0.643)},
    "F10_S": {"l1": (0.537, 0.631, 0.693), "input map": (0.647, 0.696, 0.710), "recurrent map": (0.685, 0.730, 0.738)},
    "G_S": {"input map": (0.788, 0.791, 0.800), "recurrent map": (0.819, 0.821, 0.830)},
    "in16_start": {"input map": (0.745, 0.749, 0.745), "recurrent map": (0.734, 0.740, 0.734)},
}


def part_figures(args):
    lines = open(args[0]).read().splitlines()

    def rows(net):
        out = {}
        for where, head in (("all", f"{net}, cosine with the exact gradient, variant at every map:"),
                            ("gru", f"{net}, cosine with the exact gradient, variant at the two GRU maps only")):
            j = next(k for k, ln in enumerate(lines) if ln.startswith(head)) + 1
            while not lines[j].startswith("| variant"):
                j += 1
            cols = [c.strip() for c in lines[j].strip("|").split("|")][1:]
            j += 2
            while lines[j].startswith("|"):
                cells = [c.strip() for c in lines[j].strip("|").split("|")]
                out[(cells[0], where)] = {c: float(v.split()[0]) for c, v in zip(cols, cells[1:]) if v != "–"}
                j += 1
        return out
    bad = 0
    for net, d in BRIEF.items():
        r = rows(net)
        for t, want in d.items():
            got = (r[("V0", "all")][t], r[("V2", "all")][t], r[("V4", "gru")][t])
            ok = [abs(a - b) < 5e-4 for a, b in zip(got, want)]
            bad += ok.count(False)
            print(f"{net:11s} {t:14s} brief {want[0]:.3f} {want[1]:.3f} {want[2]:.3f} | script V0 {got[0]:.3f}, V2 at every map {got[1]:.3f}, V4 at the GRU maps {got[2]:.3f} | {ok}")
    print("figures of the brief that differ from the script's output:", bad)
    ts = ("byte table", "l1", "input map", "recurrent map", "lm", "l2")
    print("\nmatrix (V0) | row (V2 at every map) | row at the two GRU maps (V2) | rowcol (V4 at the two GRU maps)")
    for net in ("in16_S", "in16_B", "F10_S", "G_S", "in16_start"):
        r = rows(net)
        for t in ts:
            print(f"{net:11s} {t:14s} " + " | ".join(f"{r[k][t]:.3f}" for k in (("V0", "all"), ("V2", "all"), ("V2", "gru"), ("V4", "gru"))))
    doc = set(ln.rstrip() for ln in open(os.path.join(RV, "SWEEP_SCALE_VARIANTS_ALIGNMENT_20261117.md")).read().splitlines())
    i0 = next(k for k, ln in enumerate(lines) if ln.startswith("===== tables"))
    i1 = next(k for k, ln in enumerate(lines) if ln.startswith("===== summary"))
    tab, keep = [], True
    for ln in lines[i0:i1]:
        if "cosine for the biases" in ln:
            keep = False                                    # the bias tables are not repeated in the document
        elif ln.startswith("**"):
            keep = True
        if keep and ln.startswith("| V"):
            tab.append(ln.rstrip())
    print(f"\nrows of the weight tables printed by this run: {len(tab)}; not found word for word in the document: {sum(ln not in doc for ln in tab)}")


# ---------------------------------------------------------------- weights
def load(ref, path):
    sys.path.insert(0, RV)
    from compare_pt_tensors import tensors
    dt = {"FloatStorage": "<f4", "LongStorage": "<i8", "BoolStorage": "?"}
    raw = show(ref, path, text=False)
    return {n.lstrip("/"): np.frombuffer(b, dtype=dt[st]).reshape(sh).copy() for n, (st, sh, b) in tensors(raw).items()}


def brief_magnitudes(Wt, maskf, scale):
    """The brief's function `magnitudes`, line for line, in numpy.  A name it does not know falls through to the last line."""
    A = np.abs(Wt) * maskf
    mean = A.sum() / maskf.sum()
    if scale == "matrix":
        return mean
    row = A.sum(1, keepdims=True) / maskf.sum(1, keepdims=True)
    if scale == "row":
        return row
    col = A.sum(0, keepdims=True) / np.maximum(maskf.sum(0, keepdims=True), 1)
    return row * col / mean


def measurement_scale(Wt, Mk, variant):
    """sweep_scale_variants.scale for V0, V2, V4 and V5 (lines 285 to 305 of that script)."""
    A = np.abs(Wt) * Mk
    if variant == "V5":
        return A
    mean = A.sum() / Mk.sum()
    row = A.sum(1) / np.maximum(Mk.sum(1), 1)
    col = A.sum(0) / np.maximum(Mk.sum(0), 1)
    one = np.ones(Wt.shape, dtype=Wt.dtype)
    return {"V0": one * mean, "V2": one * row[:, None], "V4": row[:, None] * col[None, :] / mean}[variant]


def part_weights(args):
    T = load(W9, f"{M}/exp45/e49/e49_in16_S_73129.pt")
    print("tensors of e49_in16_S_73129.pt: " + ", ".join(f"{k} {v.shape}" for k, v in T.items()))
    U, D, SIDE, RADIUS, seed = 256, 512, 16, 3, 73129
    ri = Random(f"e49|input|{seed}|16")                                    # e49_interface.draws, lines 26 and 27
    mask_in = np.zeros((U, D), dtype=bool)
    for i in range(U):
        mask_in[i, sorted(ri.sample(range(D), 16))] = True

    def neighbourhood(i):                                                  # e42_cells.neighbourhood, lines 31 and 32
        r, c = divmod(i, SIDE)
        return sorted({((r + dr) % SIDE) * SIDE + (c + dc) % SIDE for dr in range(-RADIUS, RADIUS + 1) for dc in range(-RADIUS, RADIUS + 1)} - {i})
    rc = Random(f"e42|connections|{seed}")                                 # e42_cells.draws, lines 44 and 45, N = 10
    mask = np.eye(U, dtype=bool)
    for i in range(U):
        mask[i, sorted(rc.sample(sorted(neighbourhood(i)), 10))] = True
    print("mask_in of the file equals the draw:", np.array_equal(mask_in, T["mask_in"].astype(bool)), "| recurrent mask equals the draw:",
          np.array_equal(mask, T["mask"].astype(bool)), "| mask_out all ones:", bool(T["mask_out"].all()))
    Mk = {"ih": np.tile(mask_in, (3, 1)), "hh": np.tile(mask, (3, 1))}
    for k, m in Mk.items():
        print(f"{k}: kept {int(m.sum())}; rows without a kept entry {int((m.sum(1) == 0).sum())}; columns without {int((m.sum(0) == 0).sum())};"
              f" kept in a row {sorted(set(m.sum(1).tolist()))}; kept in a column {int(m.sum(0).min())} to {int(m.sum(0).max())}")
    names = {"l1": "l1.weight", "ih": "rnn.weight_ih_l0", "hh": "rnn.weight_hh_l0", "l2": "l2.weight", "lm": "lm.weight", "l3": "l3.weight"}
    for k, n in names.items():
        Wt = T[n].astype(np.float64); m = Mk.get(k, np.ones(Wt.shape, dtype=bool)).astype(np.float64)
        d = [float(np.abs(np.broadcast_to(brief_magnitudes(Wt, m, s), Wt.shape) - measurement_scale(Wt, m, v)).max())
             for s, v in (("matrix", "V0"), ("row", "V2"), ("rowcol", "V4"), ("exact", "V5"))]
        rowm = (np.abs(Wt) * m).sum(1) / m.sum(1)
        print(f"{k:3s} largest difference, the brief's code against the measurement: matrix {d[0]:.1e}, row {d[1]:.1e}, rowcol {d[2]:.1e}; given the name"
              f" 'exact' against V5 {d[3]:.2f}; row means {rowm.min():.4f} to {rowm.max():.4f}, matrix mean {(np.abs(Wt) * m).sum() / m.sum():.4f}")
    A3 = np.abs(T["l3.weight"].astype(np.float64)); rm = A3.mean(1); tg = list(range(48, 58)) + [256]
    rest = [i for i in range(257) if i not in tg]
    print(f"l3, mean magnitude of a row: the 11 rows of the classes that occur as targets {rm[tg].min():.4f} to {rm[tg].max():.4f} (mean {rm[tg].mean():.4f});"
          f" the other 246 rows, mean {rm[rest].mean():.4f}; the matrix {A3.mean():.4f}")


# ---------------------------------------------------------------- where
PLACES = [("matrix at every map", ()), ("row at l3 only", ("l3",)), ("row at lm only", ("lm",)), ("row at l2 only", ("l2",)), ("row at l1 only", ("l1",)),
          ("row at the input map only", ("ih",)), ("row at the recurrent map only", ("hh",)), ("row at the two maps of the recurrent layer", ("ih", "hh")),
          ("row at l1, l2, lm and l3", ("l1", "l2", "lm", "l3")), ("row at every map but l3", ("l1", "ih", "hh", "l2", "lm")),
          ("row at every map", ("l1", "ih", "hh", "l2", "lm", "l3")), ("rowcol at the two maps of the recurrent layer", None)]


def part_where(args):
    sys.path.insert(0, RV)
    import sweep_scale_variants as V
    ts = ("book.weight", "l1.weight", "rnn.weight_ih_l0", "rnn.weight_hh_l0", "lm.weight", "l2.weight")

    def fb(net, maps):
        F = {}
        for k in net.maps:
            v = ("V4" if k in V.GRU_MAPS else "V0") if maps is None else ("V2" if k in maps else "V0")
            F[k] = np.sign(net.Wm(k)) * V.scale(net.P[V.WNAME[k]], net.M[k], v, k in V.GRU_MAPS)
        return F
    for name in args or ["in16_S", "in16_B", "F10_S", "G_S"]:
        net = V.Net(name)
        Fx = V.exact_feedback(net); Fs = [(lab, fb(net, maps)) for lab, maps in PLACES]
        acc = {lab: {t: [] for t in ts} for lab, _ in PLACES}
        for x, y, mask in V.stream_batches(1 + V.N_FURTHER)[1:]:
            c = V.forward(net, x, y, mask)
            ge = V.backward(net, c, Fx)
            for lab, F in Fs:
                g = V.backward(net, c, F)
                for t in ts:
                    acc[lab][t].append(V.cos(g[t], ge[t]))
        print(f"\n{name}: mean cosine with the exact gradient over batches 2 to 17 of the stream")
        print("| scale | " + " | ".join(V.SHORT[t] for t in ts) + " |")
        print("|---|" + "---|" * len(ts))
        for lab, _ in PLACES:
            print(f"| {lab} | " + " | ".join(f"{np.mean(acc[lab][t]):.3f}" for t in ts) + " |")


if __name__ == "__main__":
    {"receipts": part_receipts, "figures": part_figures, "weights": part_weights, "where": part_where}[sys.argv[1]](sys.argv[2:])
```

## The end of the check

The check ended at 23:51 UTC on 17 November 2026, without a restart of the container (the machine had been up since 22:09 UTC). The SHA-256 of the brief was then `e39604ff359b97153a69a8cca63e4683cc2d59b27935e92ec4d19f99101e9474`, as at the beginning: the brief did not change during the check. This file is the only file that the check wrote in the repository; the check wrote no bytecode beside the reviewer's scripts and committed and pushed nothing.

Check complete.
