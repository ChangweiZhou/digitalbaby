# E47: locality at the input and the output — cells that read the byte and are read by the readout form a boundary (16 November 2026, revision 2)

Evidence level E2; a first measurement. The requirements are the outside reviewer's (its memo, revision 18, section 26.3, on `claude/vibrant-cerf-sje726` at 604b29c4), written from the user's statements of 16 November. The specification, the specification file and the predictions are mine. Revision 1 (7247682a) was audited by the reviewer (`AUDIT_E47_LOCAL_INTERFACE_BRIEF_20261116.md` on `claude/vibrant-cerf-sje726` at 58ac93d8), which wrote an independent reader of the layer, required twelve changes, none to the layer or its draws, and stated that after them no re-audit is needed. Revision 2 makes them; the last section lists them. **Released. It runs after E42's other two seeds and E43's additions.** One worker for the trial; the roster's workers are named when it is released.

Shell names: `D=origin/codex/minifly-consolidated-20261023; M=MOD97_LOCAL_RULE_20261106; K=PROJECT_DOCUMENTS/COORD_CHECKS_20261116`. Host: Linux x86-64, Python 3.13.18, torch 2.15.0+cpu; at most three processes, one thread each; launch outside any wrapper with a time limit. Edit nothing existing.

## The requirement (the user's, 16 November)

"Biological cells obviously cannot see the whole input, so locality is essential and our current set up does not scale. Also it makes no sense for a cell to see one million tokens in a future NLP task."

So locality is a requirement of the design and not an option under test. E42 met it between cells and left it unmet in two places: every cell read all 512 values of the input, and every readout unit read every cell. In E42's wiring a cell costs 2,081 multiply-adds a position, of which 2,048 are those two; limiting its connections to other cells saved 17% at most (the reviewer's arithmetic, memo 26.2).

## The comparison with the record (working rule 1)

- **The task and the dose are E26's birth**, which the present network learns: 80% on untaught pairs at 3,886 to 6,899 updates, 96 to 100% at the end (24,576 updates).
- **E42 on its first seed** (the reviewer's read, not yet merged or read by me): with 10 fixed local connections a cell layer of 256 learns the family more slowly than the full network, and the sweep's gradient is less well aligned in a sparse matrix (cosine 0.60 to 0.64 against 0.74 for the full one). E47's 256-cell network **is** E42's `F10` (self-check 1), so E42's three seeds of `F10` are this job's first row and are not run again.
- **Time.** A connection passes a value one cell a step, and a cell of period p takes values in only at its own updates. From the connections, periods and phases alone (the reviewer's `e47_brief_checks.py`, parts D and F, seed 73129), the share of far cells that can take part in an answer, meaning that some state of the cell depends on a byte of the prompt and can still change a byte of the answer, is, for items of 0, 4, 8 and 12 letters: at 1,024 cells 1.00 throughout under `one`; 0.63, 0.80, 0.88, 0.96 under `place`; 0.63, 0.83, 0.93, 0.98 under `scatter`; at 4,096 cells 0.32, 0.59, 0.92, 1.00 under `one` and 0.24, 0.37, 0.50, 0.61 under `place`. A is scored at 12 letters and R without letters, so the two falls are read with these shares beside them. At 4,096 cells under `place`, 0.41 of the cells of period 8 and 0.02 of those of period 16 can take part at 12 letters, 571 far cells have no recurrent weight that can receive a gradient, and 27% of the far cells' recurrent weights receive none (10% at 1,024 cells); with decay 0.25 such a weight ends the dose at 0.2% of its start. The last digit of the second operand stands one position before the first byte of the answer, so no far cell can use it for that byte. These limits are stated before the run as reasons the far cells may go unused.
- **Periods.** A cell of period 16 is not updated at all in 10% of training items (38% of those without letters), once in 80% and twice in 10%. The reviewer's count for E42 (a long period lets a cell carry an operand without learning to keep it, and never at 40 letters) applies here to the far cells; E42's `P16` ended at 82.8 at 40 letters on its first seed under the sweep, against 96.0 for `P8`.

## The network, as code

**`$K/e47_local_spec.py` on the default branch is the specification** (its self-checks run without training; my output is `$K/OUTPUT_E47_SPEC_MAC.txt`). Copy it to `$M/e47_local.py` and import it; do not rewrite it. In words:

- **Cells** stand on a SIDE × SIDE grid with joined edges: 16, 32 or 64 a side, that is 256, 1,024 or 4,096 cells.
- **The interface** is one 16 × 16 patch, rows 0 to 15 and columns 0 to 15: 256 cells at every size. **Only these cells read the byte** (through the present network's input map, 512 values each) **and only these are read by the readout** (through the present `lm`). No other cell touches the input or the output. The byte table, `l1`, `l2` and `l3` are the present network's. The interface does not grow with the number of cells.
- **Every cell reads N = 10 cells of its neighbourhood** (the 48 cells within 3 rows and 3 columns) and its own previous value. Connections are fixed; E42's replacement rule is not used. A cell away from the interface has a bias and no input.
- **By index.** The recurrent weights are a tensor of shape (3, cells, 11) and a table of each cell's 11 sources; no matrix over all pairs of cells is built.
- **Gates** are the GRU's, as in E42. **Periods** enter as in E42's `step` variant (a learned weight for each step of a period, none for a cell of period 1).
- **The sweep** (`IndexSign` in the file): the error crosses each connection by the sign of its weight times one scale, the mean magnitude of the recurrent weights; the weight gradient is exact. This is E42's masked rule written by index.
- **Draws.** Connections by E42's rule and string on the larger grid, so at 16 a side they are E42's (self-check 2). Phases by E42's rule.
- **Start.** The present network's initial values (`arch26.new_net(seed)`) for everything it has. The recurrent weights are drawn uniformly in ±1/√11, the range E42's start has, and the cells' biases in ±1/16, from `T.Generator().manual_seed(seed * 1000 + SIDE)`; the far cells' input biases from the same seed plus one. At 16 a side the layer is built from E42's layer of the same seed (`from_e42`, then `Local(..., structure=...)`), so its start is E42's `F10` start exactly. At 32 and 64 a side every cell's recurrent weights and biases are newly drawn, the interface cells' included, and the three arms of one size and seed share one start. `Wc`, `bc`, `bfar` and `a` are in E26's decayed group (decay 0.25), as E42's added parameters were; the byte table is not decayed. The module's `rnn.weight_hh_l0` and `rnn.bias_hh_l0` take no part: they receive no gradient and are not counted, saved or digested as weights of the network. The starting draws of the new cells come from torch's generator and cannot be reproduced without it, so every run saves its starting weights.

**Three arms of periods** (at 32 and 64 a side):

| Arm | Periods |
|---|---|
| `one` | every cell period 1 |
| `place` | interface cells period 1; the cells away from it, ordered by distance from the interface and then by index, in four equal groups with periods 2, 4, 8 and 16, nearest first. Neighbours then have neighbouring periods. |
| `scatter` | the same periods, placed among the far cells by `Random(f"e47|periods|{seed}|{SIDE}")` |

Across a connection the periods differ by 0.59 (in log2) under `place` and 1.13 under `scatter` at 1,024 cells, 0.26 and 1.22 at 4,096 (self-check 3). I chose four equal groups up to period 16 and no further: an item has at most 20 positions before the answer, and a period of 16 already leaves a cell idle through one training item in ten.

**The diagnostic the memo asks for:** `far_off=True` holds every cell away from the interface at zero at every step. The scores under it say whether those cells are used. **A second diagnostic** holds every far cell, after each position, at its mean over the scored items at that position (each scored set has items of one length). It is the argument `far_mean` of the forward beside `far_off` (added to the specification file at d55e89dc); the runner computes the means from the plain forward on the same items. Holding the far cells at zero also removes a constant the interface has learned with; holding them at their mean does not.

**No extra steps before the answer.** The memo allows a variant with a stated number of steps between `=` and the answer. It is not in this job: the interface cells both read and are read, so the answer at `=` needs no travel, and the question whether the far cells are used at all comes first. At 4,096 cells a finding that the far cells are not used could not be told apart from their being too far away in time; the variant with extra steps is the follow-up if that finding occurs.

## What a cell costs (self-check 3)

| Cells | Weights | Multiply-adds a position, every cell every step | With `place` or `scatter` periods |
|---|---|---|---|
| 256 | 935,953 | 930,560 | — |
| 1,024 | 965,905 (971,665 with periods) | 955,904 | 944,948 |
| 4,096 | 1,085,713 (1,114,513) | 1,057,280 | 1,002,500 |
| the present network, for comparison | 1,124,113 | 1,118,720 | — |

A cell of period p costs 33 / p multiply-adds a position for its products and 11 for adding up its sources' values with the step weights.

So sixteen times the cells cost 14% more multiply-adds, or 8% with periods, and all three sizes stay below the present network. That is the point of the wiring: a cell costs 33 multiply-adds, not 2,081.

**Counted computation is not measured time.** On my Mac the indexed arithmetic of the file is slow: 0.09, 0.27 to 0.39 and 1.2 to 1.6 CPU-seconds for a forward and backward at 256, 1,024 and 4,096 cells, where the multiply-adds differ by 14%. Gathering by index costs far more a multiply-add than a dense product. The job compares in multiply-adds and weights and reports CPU-seconds beside them; and the worker is expected to speed the file's loop up, under check 3, best by computing only the cells updated at a position, as the reviewer's reader does. On these figures the roster's runs at 4,096 cells take 38 CPU-hours of updates on my machine, and E42's files ran 1.8 to 3.3 times slower on the cloud host; check 7's forecast is made on the worker's host and includes scoring, three forwards a scored row.

## Stage 0: checks, and one full-length trial (working rule 2)

Done by one worker, who writes the runner `exp47_local.py` (E42's runner with this layer; E42's data, optimizer, dose of 24,576 updates, and scores).

1. **The specification file.** `$M/e47_local.py` is byte-identical to `$K/e47_local_spec.py`; run on `$K`, its self-checks print what `$K/OUTPUT_E47_SPEC_MAC.txt` has for checks 1 to 4, the draws' digests for seed 73129 among them (the audit's): at 256 cells the same logits as E42's forward to 1e-6 and the same gradients under both rules to 1e-5, for `F10` and for `P8`; the connections at 16 a side equal E42's; the counts and the periods; and the time and locality properties.
2. **At 16 a side the runner's layer is E42's `F10`.** The runner builds the 256-cell layer from E42's layer of the same seed with `from_e42` and `Local(..., structure=...)`; the default constructor draws a different start and is not used at 16 a side. From that start, on the first batch of seed 73129, the runner's logits equal those of E42's runner to 1e-6 and every gradient, mapped by index, to 1e-5 of the tensor's largest entry, under each rule. The losses of the first 16 updates of the two runners, and the scores of a dry run of 2,048 updates under the sweep beside E42's `e42_F10_S_dry2048_73129`, are reported side by side. Equality of the two runs is not required: the programs add in a different order, AdamW's first update follows the sign of the gradient, and E42 has no dry run of `F10` under backpropagation. A loss that differs by more than 1e-6 at the first update or 1e-3 at the sixteenth is investigated before the trial.
3. **A sped-up forward, if the worker writes one,** equals the specification's on the first batch at every size and arm it is used in: logits to 1e-6 and every parameter gradient to 1e-5 of the tensor's largest entry, under both rules; and the time properties of self-check 4 hold on it.
4. **Locality, on the runner's forward:** changing the byte at one position changes no far cell's state at that position, and changes a far cell no sooner than its distance allows; the logits at a position do not change when a far cell's state at that position is changed by hand (they change only through later positions); with `far_off=True` the far cells are zero.
5. **Same data:** the batch digest at row 4,096 is E26's for the seed.
6. **Scores:** E42's check 12 on E26's six births, each loaded by index with `structure=`, every cell's sources being itself and the 255 others; the file's own draws are not used, since a neighbourhood has 48 cells.
7. **Cost:** CPU-seconds per update and per scored row at each size and arm, on an idle host, and a forecast for the roster below.

**The trial.** One seed, 73129, at 1,024 cells, with periods `one` and with periods `place`, under the sweep and under backpropagation: four full runs of 24,576 updates, the two `one` runs moved forward from the roster, scored every 256. Periods by place alone could not separate the effect of the boundary from the effect of the periods. Recorded: at every scored row, E42's scores plain, with the far cells at zero and with the far cells at their mean, on A, R and the length test at 12 and 40 letters (at 40 letters nearly every far cell can take part, so it is the one panel on which distance in time does not limit use); the mean magnitude of the recurrent weights of the interface cells and of the far cells, and the sweep's scale; at the first batch and at the end, the cosines between the sweep's and the exact gradient for interface cells and for far cells, and the share of the far cells' recurrent weights whose gradient is not zero; CPU-seconds an update and a scored row. **The worker reports the trial and stops.** I read it with the reviewer before the roster is released, and the roster may change.

## Stage 1: the roster (released by me after the trial is read)

As planned now, seed 73129 first, six further runs at most: at 1,024 cells the arm `scatter` under both rules; at 4,096 cells the arm `one`, under backpropagation first and only when check 7 forecasts it at a stated number of CPU-hours, and under the sweep after that run has been read. **The two `place` runs at 4,096 cells are not planned** (the audit's recommendation): by structure four in ten of their far cells can take part in no training item, and they are the two most expensive runs. The other two seeds follow only for the arms the first seed shows worth it.

**Measures**, per run: E42's (updates and weight-items to the 50, 80 and 90 thresholds on A and R, three-row version; the length test at 12 and 40 letters), with multiply-adds a position in place of CPU-seconds; the scores with `far_off=True`; the cosines.

## How the result is stated (fixed beforehand)

Against E42's `F10` of the same seed and rule (256 cells), against E42's `P8` for the arms with periods, and against E42's `G` (the full network):

- **Updates to the 80 threshold** on A and on R: E41's table of four cases, read as fewer, more or about the same. The same in multiply-adds.
- **The length test:** the mean at 40 letters over the last three scored rows; *lower* or *higher* at 15 points or more.
- **Use of the far cells**, at the end of the run, from the fall of A and of R under the two diagnostics; the first form that applies. *Used*: A or R falls by 15 points or more with the far cells held at their mean. *Not used*: A and R both fall by less than 5 with the far cells held at zero. *A constant is used*: A or R falls by 15 points or more at zero and both fall by less than 5 at the mean. Otherwise *partly*. With one seed, a description is that seed's.
- `place` against `scatter`, and each against `one`, by the same forms.

**The sentences this job can license**, on the seeds run: that a network whose input and output are confined to a boundary of 256 cells learns the first family (reaches 80 on R within the dose) or does not; that cells beyond the boundary are used, partly used or not used; and what a cell costs in multiply-adds. Nothing about holding many families, which is where more cells should pay and which this job does not test.

## What it cannot show

- **A benefit of more cells.** One family needs a few hundred cells at most; no fewer updates are expected from more. The job shows whether a network wired this way learns at all, whether it uses its far cells, and what a cell costs.
- **Many families and keeping them apart**, which is the long-life world (E44) and the retention question. A network of this kind goes there only after this job.
- **Locality inside the interface.** Each of the 256 interface cells reads all 512 values of the byte's code, and each readout unit reads all 256 interface cells: 524,288 of a position's multiply-adds at every size. The user's requirement is met for the cells outside the patch and not for those inside it. E42's trained `F10` rests on the dense interface (with the smaller half of each cell's input weights removed and nothing retrained, its score at 12 letters falls from 92 to 83 under the sweep and from 91 to 72 under backpropagation; the reviewer's measurement). **Whether a network trained with a limited interface learns is the next job.**
- **One scale for all cells.** The sweep's scale is a mean over every cell's weights, which a cell does not have; a scale from a cell's own 11 weights is a later variant.
- **Measured time.** The arithmetic by index is slow on a CPU; the comparison is in counted computation.
- **Settings.** Learning rate, decay and N are E42's.
- **The form of a cell.** The GRU's equations are kept.

## Predictions (mine)

| | Prediction | Chance |
|---|---|---|
| P47.1 | Stage 0, checks 1 and 2 pass under both rules | 85% |
| P47.2 | Trial, backpropagation, `place`: reaches 80 on R within the dose | 70% |
| P47.3 | Trial, sweep, `place`: reaches 80 on R within the dose | 45% |
| P47.4 | Trial, backpropagation, `one`: the far cells are *used* (a fall of 15 points or more on A or R with them held at their mean) | 45% |
| P47.5 | Trial, sweep, `one`: the far cells are *used* | 30% |
| P47.10 | Trial: in at least one of the four runs the far cells read *a constant is used* | 30% |
| P47.11 | Trial, backpropagation: `place` reaches 80 on R in *about the same* updates as `one` | 50% |
| P47.6 | Trial, `place`: the cosine between the sweep's and the exact gradient is lower for far cells than for interface cells at the end, by 0.1 or more | 60% |
| P47.7 | Trial, backpropagation, `one`: updates to 80 on R are *about the same* as or *more* than E42's `F10` of seed 73129 (not fewer) | 80% |
| P47.8 | Roster, seed 73129: at 4,096 cells under `one` and backpropagation the far cells are not *used* | 65% |
| P47.9 | Roster, seed 73129, 1,024 cells, backpropagation: `place` and `scatter` are *about the same* in updates to 80 on R | 55% |

## Done

Stage 0: the runner, the check receipts, the forecast, and the four trial runs with receipts, weights on the remote (`git add -f`, confirmed by `git ls-tree`), and `E47_TRIAL_REPORT.md` with the curves, the scores under both diagnostics, the cosines and P47.1 to P47.7, P47.10 and P47.11. Every run saves its starting and its final weights, each with the table of sources (cells × 11), the periods, the phases, `Wc`, `bc`, `bfar` and `a`; and, at the last scored row, the bytes of each answer given on the length-test panel, plain and under each diagnostic. The receipt names the side and the arm. No readings beyond the forms above. Then stop.

## What changed from revision 1 (the audit)

Made as the audit wrote them. The specification file gained the second diagnostic as an argument of the forward and the audit's digests of the draws as a self-check, at d55e89dc; the layer and its draws are unchanged, and I reran the self-checks.

| | Change |
|---|---|
| 1 | Check 2 replaced by a first-batch comparison: two programs that agree to seven decimals still drift apart over 2,048 updates, so my check could not pass. The start at 16 a side is E42's, by `from_e42`. |
| 2 | Which parameters decay, and that the module's own recurrent tensors take no part |
| 3, 4 | Checks 4 and 6 reworded so that they test what they name |
| 5 | The shares of far cells that can take part in an answer replace my two bullets on time and periods. At 4,096 cells with periods by place only 2% of the slowest cells can take part even at 12 letters. |
| 6 | A second diagnostic, the far cells held at their mean, and "used" defined on it; holding them at zero also removes a constant |
| 7 | The trial is four runs, with all-period-1 beside periods by place |
| 8 | The cost table corrected (8% with periods, not 3%; I had left out the adding-up of sources with step weights) and the roster's cost stated |
| 9 | E42's `P8` as the reference for the arms with periods |
| 10 | Locality is not yet met inside the interface; a limited interface is the next job |
| 11 | What every run saves |
| 12 | A sentence of review 9.150 corrected (below) |
| — | Recommendations taken: the two `place` runs at 4,096 cells dropped from the plan; the 4,096-cell run under backpropagation first; the diagnostics also at 40 letters; a sped-up forward that computes only updated cells; a per-cell scale named as a later variant |
