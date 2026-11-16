# E47: locality at the input and the output — cells that read the byte and are read by the readout form a boundary (16 November 2026, revision 1)

Evidence level E2; a first measurement. The requirements are the outside reviewer's (its memo, revision 18, section 26.3, on `claude/vibrant-cerf-sje726` at 604b29c4), written from the user's statements of 16 November. The specification, the specification file and the predictions are mine. **Not released: it waits for the reviewer's audit. It runs after E42's other two seeds and E43's additions.** One worker for the trial; the roster's workers are named when it is released.

Shell names: `D=origin/codex/minifly-consolidated-20261023; M=MOD97_LOCAL_RULE_20261106; K=PROJECT_DOCUMENTS/COORD_CHECKS_20261116`. Host: Linux x86-64, Python 3.13.18, torch 2.15.0+cpu; at most three processes, one thread each; launch outside any wrapper with a time limit. Edit nothing existing.

## The requirement (the user's, 16 November)

"Biological cells obviously cannot see the whole input, so locality is essential and our current set up does not scale. Also it makes no sense for a cell to see one million tokens in a future NLP task."

So locality is a requirement of the design and not an option under test. E42 met it between cells and left it unmet in two places: every cell read all 512 values of the input, and every readout unit read every cell. In E42's wiring a cell costs 2,081 multiply-adds a position, of which 2,048 are those two; limiting its connections to other cells saved 17% at most (the reviewer's arithmetic, memo 26.2).

## The comparison with the record (working rule 1)

- **The task and the dose are E26's birth**, which the present network learns: 80% on untaught pairs at 3,886 to 6,899 updates, 96 to 100% at the end (24,576 updates).
- **E42 on its first seed** (the reviewer's read, not yet merged or read by me): with 10 fixed local connections a cell layer of 256 learns the family more slowly than the full network, and the sweep's gradient is less well aligned in a sparse matrix (cosine 0.60 to 0.64 against 0.74 for the full one). E47's 256-cell network **is** E42's `F10` (self-check 1), so E42's three seeds of `F10` are this job's first row and are not run again.
- **Time.** A connection passes a value one cell a step. The prompt has 8 to 20 bytes before the answer, and the farthest cell is 3 connections from the interface at 1,024 cells and 8 at 4,096 (self-check 3). So a far cell can be reached and can answer back within an item at 1,024 cells in every item, and at 4,096 cells only in the longer ones: a round trip of 16 steps there. This is stated before the run, as a reason the far cells of the largest size may go unused.
- **Periods.** A cell of period 16 is updated once or twice in an item. The reviewer's count for E42 (a long period lets a cell carry an operand without learning to keep it, and never at 40 letters) applies here to the far cells.

## The network, as code

**`$K/e47_local_spec.py` on the default branch is the specification** (its self-checks run without training; my output is `$K/OUTPUT_E47_SPEC_MAC.txt`). Copy it to `$M/e47_local.py` and import it; do not rewrite it. In words:

- **Cells** stand on a SIDE × SIDE grid with joined edges: 16, 32 or 64 a side, that is 256, 1,024 or 4,096 cells.
- **The interface** is one 16 × 16 patch, rows 0 to 15 and columns 0 to 15: 256 cells at every size. **Only these cells read the byte** (through the present network's input map, 512 values each) **and only these are read by the readout** (through the present `lm`). No other cell touches the input or the output. The byte table, `l1`, `l2` and `l3` are the present network's. The interface does not grow with the number of cells.
- **Every cell reads N = 10 cells of its neighbourhood** (the 48 cells within 3 rows and 3 columns) and its own previous value. Connections are fixed; E42's replacement rule is not used. A cell away from the interface has a bias and no input.
- **By index.** The recurrent weights are a tensor of shape (3, cells, 11) and a table of each cell's 11 sources; no matrix over all pairs of cells is built.
- **Gates** are the GRU's, as in E42. **Periods** enter as in E42's `step` variant (a learned weight for each step of a period, none for a cell of period 1).
- **The sweep** (`IndexSign` in the file): the error crosses each connection by the sign of its weight times one scale, the mean magnitude of the recurrent weights; the weight gradient is exact. This is E42's masked rule written by index.
- **Draws.** Connections by E42's rule and string on the larger grid, so at 16 a side they are E42's (self-check 2). Phases by E42's rule.
- **Start.** The present network's initial values (`arch26.new_net(seed)`) for everything it has. The recurrent weights are drawn uniformly in ±1/√11, the range E42's start has, and the cells' biases in ±1/16, from `T.Generator().manual_seed(seed * 1000 + SIDE)`; the far cells' input biases from the same seed plus one. At 16 a side the 256-cell network is run as E42's `F10`, from E42's own start.

**Three arms of periods** (at 32 and 64 a side):

| Arm | Periods |
|---|---|
| `one` | every cell period 1 |
| `place` | interface cells period 1; the cells away from it, ordered by distance from the interface and then by index, in four equal groups with periods 2, 4, 8 and 16, nearest first. Neighbours then have neighbouring periods. |
| `scatter` | the same periods, placed among the far cells by `Random(f"e47|periods|{seed}|{SIDE}")` |

Across a connection the periods differ by 0.59 (in log2) under `place` and 1.13 under `scatter` at 1,024 cells, 0.26 and 1.22 at 4,096 (self-check 3). I chose four equal groups up to period 16 and no further: an item has at most 20 positions before the answer, so a period beyond 16 would leave a cell idle through whole items.

**The diagnostic the memo asks for:** `far_off=True` holds every cell away from the interface at zero at every step. The scores under it say whether those cells are used.

**No extra steps before the answer.** The memo allows a variant with a stated number of steps between `=` and the answer. It is not in this job: the interface cells both read and are read, so the answer at `=` needs no travel, and the question whether the far cells are used at all comes first.

## What a cell costs (self-check 3)

| Cells | Weights | Multiply-adds a position, every cell every step | With `place` or `scatter` periods |
|---|---|---|---|
| 256 | 935,953 | 930,560 | — |
| 1,024 | 965,905 (971,665 with periods) | 955,904 | 936,500 |
| 4,096 | 1,085,713 (1,114,513) | 1,057,280 | 960,260 |
| the present network, for comparison | 1,124,113 | 1,122,048 | — |

So sixteen times the cells cost 14% more multiply-adds, or 3% with periods, and all three sizes stay below the present network. That is the point of the wiring: a cell costs 33 multiply-adds, not 2,081.

**Counted computation is not measured time.** On my Mac the indexed arithmetic of the file is slow: 0.09, 0.27 to 0.39 and 1.2 to 1.6 CPU-seconds for a forward and backward at 256, 1,024 and 4,096 cells, where the multiply-adds differ by 14%. Gathering by index costs far more a multiply-add than a dense product. The job compares in multiply-adds and weights and reports CPU-seconds beside them; and the worker is expected to speed the file's loop up, under check 3.

## Stage 0: checks, and one full-length trial (working rule 2)

Done by one worker, who writes the runner `exp47_local.py` (E42's runner with this layer; E42's data, optimizer, dose of 24,576 updates, and scores).

1. **The specification file.** `$M/e47_local.py` is byte-identical to `$K/e47_local_spec.py`; run on `$K`, its self-checks print what `$K/OUTPUT_E47_SPEC_MAC.txt` has for checks 1 to 4: at 256 cells the same logits as E42's forward to 1e-6 and the same gradients under both rules to 1e-5, for `F10` and for `P8`; the connections at 16 a side equal E42's; the counts and the periods; and the time and locality properties.
2. **The 256-cell run is E42's `F10`.** A dry run of 2,048 updates at 16 a side has the `tensors_sha256` structure digests and scores of E42's `F10` dry run of seed 73129 under each rule, to the tolerance E42's own check 8 used (the weights are stored by index here, so the comparison is of the mapped tensors to 1e-5 and of the scores exactly).
3. **A sped-up forward, if the worker writes one,** equals the specification's on the first batch at every size and arm it is used in: logits to 1e-6 and every parameter gradient to 1e-5 of the tensor's largest entry, under both rules; and the time properties of self-check 4 hold on it.
4. **Locality, on the runner's forward:** changing any input weight of the input map changes no far cell's state at the same position; the logits at a position do not change when a far cell's state at that position is changed by hand (they change only through later positions); with `far_off=True` the far cells are zero.
5. **Same data:** the batch digest at row 4,096 is E26's for the seed.
6. **Scores:** E42's check 12, on E26's six births through the 256-cell layer with E42's start mapped.
7. **Cost:** CPU-seconds per update and per scored row at each size and arm, on an idle host, and a forecast for the roster below.

**The trial.** One seed, 73129, at 1,024 cells with periods `place`, under the sweep and under backpropagation: two full runs of 24,576 updates, scored every 256, with E42's scores and, at every scored row, the same scores with `far_off=True`, and the cosine between the sweep's and the exact gradient of the recurrent weights at the first batch and the end, for interface cells and for far cells separately. **The worker reports the trial and stops.** I read it with the reviewer before the roster is released, and the roster may change.

## Stage 1: the roster (released by me after the trial is read)

As planned now, seed 73129 first: at 1,024 cells the arms `one`, `place`, `scatter` under both rules (the two `place` runs are the trial's), and at 4,096 cells `one` and `place` under both rules. Eight further runs. The other two seeds follow only for the arms the first seed shows worth it. If the forecast for the eight exceeds 40 CPU-hours, the 4,096-cell runs wait.

**Measures**, per run: E42's (updates and weight-items to the 50, 80 and 90 thresholds on A and R, three-row version; the length test at 12 and 40 letters), with multiply-adds a position in place of CPU-seconds; the scores with `far_off=True`; the cosines.

## How the result is stated (fixed beforehand)

Against E42's `F10` of the same seed and rule (256 cells), and against E42's `G` (the full network):

- **Updates to the 80 threshold** on A and on R: E41's table of four cases, read as fewer, more or about the same. The same in multiply-adds.
- **The length test:** the mean at 40 letters over the last three scored rows; *lower* or *higher* at 15 points or more.
- **Use of the far cells**, at the end of the run: the fall of A and of R when the far cells are held at zero. *Used* if either falls by 15 points or more; *not used* if both fall by less than 5; otherwise *partly*. With one seed, a description is that seed's.
- `place` against `scatter`, and each against `one`, by the same forms.

**The sentences this job can license**, on the seeds run: that a network whose input and output are confined to a boundary of 256 cells learns the first family (reaches 80 on R within the dose) or does not; that cells beyond the boundary are used, partly used or not used; and what a cell costs in multiply-adds. Nothing about holding many families, which is where more cells should pay and which this job does not test.

## What it cannot show

- **A benefit of more cells.** One family needs a few hundred cells at most; no fewer updates are expected from more. The job shows whether a network wired this way learns at all, whether it uses its far cells, and what a cell costs.
- **Many families and keeping them apart**, which is the long-life world (E44) and the retention question. A network of this kind goes there only after this job.
- **The interface's own size.** It is fixed at 256 cells; a smaller interface is a later question.
- **Measured time.** The arithmetic by index is slow on a CPU; the comparison is in counted computation.
- **Settings.** Learning rate, decay and N are E42's.
- **The form of a cell.** The GRU's equations are kept.

## Predictions (mine)

| | Prediction | Chance |
|---|---|---|
| P47.1 | Stage 0, checks 1 and 2 pass under both rules | 85% |
| P47.2 | Trial, backpropagation: reaches 80 on R within the dose | 70% |
| P47.3 | Trial, sweep: reaches 80 on R within the dose | 45% |
| P47.4 | Trial, backpropagation: the far cells are *used* (a fall of 15 points or more on A or R when they are held at zero) | 55% |
| P47.5 | Trial, sweep: the far cells are *used* | 40% |
| P47.6 | Trial: the cosine between the sweep's and the exact gradient is lower for far cells than for interface cells at the end, by 0.1 or more | 60% |
| P47.7 | Trial, backpropagation: updates to 80 on R are *about the same* as or *more* than E42's `F10` of seed 73129 (not fewer) | 80% |
| P47.8 | Roster, seed 73129: at 4,096 cells under `one` the far cells are *not used* or *partly* under each rule | 60% |
| P47.9 | Roster, seed 73129, 1,024 cells, backpropagation: `place` and `scatter` are *about the same* in updates to 80 on R | 55% |

## Done

Stage 0: the runner, the check receipts, the forecast, and the two trial runs with receipts, weights on the remote (`git add -f`, confirmed by `git ls-tree`), and `E47_TRIAL_REPORT.md` with the curves, the far-off scores, the cosines and P47.1 to P47.7. No readings beyond the forms above. Then stop.
