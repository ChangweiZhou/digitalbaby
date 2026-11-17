# E53: the network with its input and output at a boundary, without weight decay and from a start at which differences between items reach the cells away from the boundary (17 November 2026, revision 1)

Evidence level E2. **Status.** Written by the outside reviewer, acting for the coordinator, who cannot be reached while Mr. Xu is away (his instruction of 17 November: arrange it yourself). It is item 9 of `PLAN.md` beside this file, narrowed to one network form by the measurement of item 1. It is given to a check by an agent of the reviewer that has not seen the reviewer's reasoning before it is released; that check is not the independent audit the programme's rules ask for, and the coordinator has not seen the brief. One seed, four runs, about 15 CPU-hours.

## Shell names and host

`D=origin/codex/minifly-consolidated-20261023` (the default branch, at 7cabb13a); `W7=origin/claude/e47-worker-stage-0-3d0ya6` (E47's worker's branch, at 578a877b: the runner `exp47_local.py` and the trial's receipts and weights under `$M/exp45/e47/`; the runner is not on the default branch); `V=origin/claude/vibrant-cerf-sje726` (the reviewer's branch); `M=MOD97_LOCAL_RULE_20261106`; `R=PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109`. Host: Linux x86-64, Python 3.13, torch 2.15.0+cpu; at most three processes, one thread each.

## Terms

- *The network* is E47's at 32 cells a side in its arm `one` (`$D:OPUS_TASK_E47_LOCALITY_AT_THE_INPUT_AND_THE_OUTPUT_20261116.md`): 1,024 cells on a grid with joined edges, each reading its own previous state and ten cells within three rows and three columns of it, every cell updated at every position of an item. The 256 *interface cells* of one 16 by 16 patch receive the code of the current byte and are read by the readout. The 768 *far cells* receive no input and are not read. A far cell's *ring* is the number of connections between it and the interface: ring 1 has 228 cells, ring 2 has 300, ring 3 has 240.
- *The recurrent weights* are the layer's tensor `Wc`: for each of the three gates and each cell, eleven weights, on the cell's own state and on its ten sources (33,792 numbers). At E47's start they are uniform within 1/√11.
- *Back-propagation* (rule B) is the exact gradient; *the sweep* (rule S) passes the error backward across each connection by the sign of its weight times one scale, the mean size of all recurrent weights.
- *The first family*, scores A and R, the length test, and *updates to 80* are E47's and E42's: A is the share right on untaught pairs of add and sub at 12 letters, R the share right on untaught pairs of add and sub that need the reduction, without letters; a score reaches 80 at the first of three consecutive scored rows at or above 80 (rows every 256 updates). A *level* is the mean over the last three rows.
- *The two diagnostics* are E47's: a trained network is scored with the far cells held, after every position, at zero, or at their mean over the scored items at that position.
- *The share that reaches a ring* is the reviewer's measure (`$R/SIGNAL_PASSAGE_AT_THE_START_20261117.md`): two items that differ in one byte are run through the network, and the root-mean-square difference of the states of a ring's cells, over the later positions, is divided by that of the interface cells.

## The question

E47's trial showed that a network whose input and output are confined to a boundary learns the first family, and that its cells beyond the first ring take no part: holding rings 2 and 3 at their mean changes no score by more than 0.7 of a point (`$R/E47_TRIAL_VERIFICATION_20261117.md`). Two things could keep those cells out, and neither was varied. At the start, differences between items hardly reach them. And the standing weight decay shrinks every weight that learning does not hold up. **Do the cells beyond the first ring come into use, and is the family learned in fewer updates, when the network is trained without weight decay, or without decay and from a start at which a difference between items reaches the third ring at a seventh of its size at the interface?**

The answer decides whether a later job may count on cells away from the interface (item 12 of the plan: a second task learned in them).

## What the record says (working rule 1)

Seed 73129 throughout. The reviewer recomputed the figures of the trial from the receipts (`$W7:$M/exp45/e47/e47_32_one_<rule>_73129.json`).

- **The trial, arm `one`, weight decay 0.25, 24,576 updates.** Under the sweep A reaches 80 at update 13,568 and R at 17,664; the levels are A 92.3, R 87.8, length test at 40 letters 90.8. Under back-propagation A reaches 80 at 10,496 and R at 11,008; the levels are 95.4, 94.2 and 95.2. E42's references of the same seed: the dense network 5,632 and 5,120 updates to 80 on A under the sweep and back-propagation; the network with ten local connections a cell and every cell at the input and the output (`F10`) 13,312 and 8,704. Each run cost 3.6 CPU-hours.
- **The diagnostics at the last row.** With the far cells at zero A and R fall from 93.3 and 86.3 to 77.5 and 77.7 under the sweep and from 96.2 and 94.7 to 78.8 and 78.9 under back-propagation. With the far cells at their mean they fall to 84.8 and 80.6, and to 90.2 and 88.7. The whole effect comes from ring 1.
- **The size of the weights.** The mean size of a recurrent weight is 0.151 for interface cells and 0.150 for far cells at the start. At the end it is 0.198 and 0.150 under the sweep and 0.179 and 0.243 under back-propagation.
- **The passage of a difference at the start** (`$R/SIGNAL_PASSAGE_AT_THE_START_20261117.md`; items of 12 letters, the first digit changed). Ring 1 holds 0.187 of the interface's difference, ring 2 0.0207, ring 3 0.0028. One connection crossed in one position multiplies a difference by 0.043. With the recurrent weights multiplied by 3 the third ring holds 0.141 (0.05 for items without letters), a difference put into ring 3 reaches the interface at 0.149, and a difference shrinks along the item by a factor of 0.95 a position. With 3.25 the share is 0.202 and still shrinks; with 3.5 a difference grows along the item. No uniform change that was tried brings the third ring to a third without growth, because only 2.4 to 4.0 of a far cell's ten connections come from the ring nearer the interface.
- **What training did to the passage.** In the two trained networks of the arm `one` ring 1 holds 0.59 (sweep) and 0.50 (back-propagation) of the interface's difference, ring 2 0.11 and 0.08, and ring 3 0.015 and 0.0006, although the far cells' weights grew under back-propagation. In both trained networks a difference no longer shrinks along the item.
- No run of this network without decay or from another start exists.

## Runs

| Arm | Start | Weight decay | Rules | Updates | Runs |
|---|---|---|---|---|---|
| `one` (the trial; on file) | E47's | 0.25 | S, B | 24,576 | 0 |
| `wd0` | E47's | 0 | S, B | 24,576 | 2 |
| `x3-wd0` | E47's with every entry of `Wc` multiplied by 3 | 0 | S, B | 24,576 | 2 |

Seed 73129. "Weight decay 0" sets the decay of the optimizer's first group to 0 (the group that holds `l1`, the input map, `lm`, `l2`, `l3`, `Wc`, `bc` and `bfar`); the byte table has none in any arm. Everything else is the trial's: the stream, the panels, the two diagnostics at every row, the rows every 256 updates.

A start multiplied by 3 with the standing decay is not run. Over 4,000 updates the decay alone multiplies a weight by 0.37, so that the multiple would be gone before the family is learned, and the run would repeat the trial.

Cost: 3.65 CPU-hours a run in the trial; 14.6 for the four, about 0.4 for the checks.

## The runner

A new file `exp53_local_start.py` in `$M`; no existing source is edited. `exp47_local.py` and `e47_local.py` are copied byte for byte from `$W7` (the second equals `$D:PROJECT_DOCUMENTS/COORD_CHECKS_20261116/e47_local_spec.py`), with their SHA-256 and source recorded in a text file beside them. The new file holds a copy of the text of `exp47_local.run` with the following changes and no others, and the report shows the difference between the two texts.

1. After `build` and before the optimizer is made, in the arm `x3-wd0`, `L.Wc` is multiplied by 3 in place. The starting file is saved after this, so that it holds the start as used.
2. After `make_opt`, in the arms `wd0` and `x3-wd0`, the weight decay of the group that has 0.25 is set to 0. The receipt records the optimizer's groups as read back from the optimizer.
3. The receipt has `experiment="E53"`, the arm's name and the multiple, and the name `$M/exp45/e53/e53_32_<arm>_<rule>_73129.json`; `provenance` adds the hash of `exp53_local_start.py`.
4. At every row the receipt also records the mean size of `Wc` for the interface cells and for each of the three rings apart (the trial records interface and far cells), and the share of gate values beyond 0.05 and 0.95 on the first batch.
5. Saving and resuming are the trial's (a state every 1,024 updates). The final weights, the starting weights and the answers file are committed, as in the trial.

## Checks before the runs (a receipt for each)

1. **With E47's start and decay 0.25 the runner is the trial's.** A dry run of 1,024 updates under each rule: in every row the loss, the counts of every score under the plain scoring and under both diagnostics, and the batch digest equal those of `$W7:$M/exp45/e47/e47_32_one_<rule>_73129.json`.
2. **The starts.** For `wd0` every tensor of the starting file is bit-equal to the one in `$W7:$M/exp45/e47/e47_32_one_<rule>_73129_start.pt`. For `x3-wd0` every tensor but `Wc` is bit-equal, and `Wc` equals the trial's `Wc` multiplied by 3 in single precision, entry for entry.
3. **The decay.** In `wd0` and `x3-wd0` the optimizer read back has a weight decay of 0 in every group. After the first update of `wd0` under each rule, every tensor of the first group differs from the trial's dry run of check 1 after its first update by 0.00025 times the starting tensor, with a largest absolute deviation below 1e-6.
4. **A resumed run equals an uninterrupted one** (the trial's check 8) for the arm `x3-wd0` under the sweep: 2,048 updates, once without interruption and once stopped after update 1,024 and resumed, equal in every row and in the hash of the final tensors.
5. **The start of `x3-wd0` on the first batch:** the loss is finite, and the share of gate values beyond 0.05 and 0.95 is recorded. (The reviewer's measurement finds none beyond these marks at this multiple.)
6. **Cost:** CPU-seconds an update and a scored row, and a forecast.

## How the result is stated (fixed beforehand)

For each rule, `wd0` and `x3-wd0` against the trial's `one`, and against each other.

- **Learning.** Updates to 80 on A and on R, with E42's words and bands (`describe` in `e42_tables.py` on `origin/claude/gallant-dijkstra-i9zgsr`, lines 63 to 72: a ratio below 2/3 is *fewer*, above 1.5 *more*, otherwise *about the same*, with its rules for a run that does not reach 80); the levels of A, of R and of the length test at 40 letters.
- **The far cells as a whole.** The two diagnostics at the last row, read with the words of E47's brief: *used* if A or R falls by 15 points or more with the far cells at their mean; *not used* if both fall by less than 5 with the far cells at zero; *a constant is used* if either falls by 15 or more at zero and both by less than 5 at the mean; otherwise *partly*.
- **Rings 2 and 3.** The reviewer measures this from the committed weights, with the reader and the scripts of the trial's verification (`$R/e47_trial_verify.py`, `$R/signal_passage.py`): the fall of A and of R at the last row when rings 2 and 3 are held at their mean, and the share of a difference that reaches each ring. Rings 2 and 3 *are used* in a run if A or R falls by 5 points or more when they are held at their mean. The worker's report gives the tables of the first two points and leaves this one to the reviewer.
- **The size of the weights** by group at the first and the last row.

The explanation under test: the cells beyond the first ring stay out of the computation because differences between items do not reach them at the start and because the decay shrinks what is not used. It is refuted, for this seed, if rings 2 and 3 are not used in `x3-wd0` under either rule. If they are used in some run, the report says in which, and whether that run learns the family in *fewer*, *about the same* or *more* updates than the trial.

## Predictions (the reviewer's, on file before any run)

| | Statement | Chance |
|---|---|---|
| P53.1 | Checks 1 to 5 pass | 85% |
| P53.2 | `wd0` reaches 80 on A within the run under both rules | 85% |
| P53.3 | `wd0` against `one`: updates to 80 on A are *about the same* under both rules | 70% |
| P53.4 | `x3-wd0` reaches 80 on A within the run under back-propagation | 60% |
| P53.5 | `x3-wd0` reaches 80 on A within the run under the sweep | 45% |
| P53.6 | Rings 2 and 3 are used in `x3-wd0` under at least one rule | 35% |
| P53.7 | Rings 2 and 3 are used in `wd0` under at least one rule | 10% |
| P53.8 | No new run needs *fewer* updates to 80 on A than the trial's run of its rule | 80% |

## What it cannot show

- More than one seed, the arm with periods (`place`), or other sizes of the grid.
- Whether cells away from the interface would be used by a task that the interface cells cannot learn alone. The first family is learned by 256 cells, so that the far cells are not needed here; a run in which they stay unused shows that learning does not draw them in by itself, and not that they could not be used.
- A start that passes differences outward only. The reviewer's measurement found one (the weights on connections from a cell nearer the interface multiplied by 5: 0.64 reaches ring 3), but it leaves the way back to the interface at 0.0025, and the readout reads the interface alone.
- The sweep with a scale taken for each cell (item 18 of the plan). Here the scale is one mean over all cells, and the multiple enlarges it with the weights.

## Rules for the worker

- At most three processes, one thread each (`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `torch.set_num_threads(1)`). torch 2.15.0+cpu from `https://download.pytorch.org/whl/cpu` in a virtual environment; versions recorded in every receipt.
- **This host stops a machine about five minutes after its session's turn ends, with the disk kept, and detached processes die with it.** While runs proceed, keep your turn open with a waiting command (a command that waits on the processes, repeated in pieces within the command's time limit), and commit and push the running receipts between the pieces. After an interruption a run is resumed from its saved state (check 4). The report says which runs were resumed.
- Push only to your own branch. Never push to the default branch or to any other branch, and open no pull request. Other branches are read-only: fetch with `git fetch origin '+refs/heads/<name>:refs/remotes/origin/<name>'` and read with `git show`.
- Commit trailers as `PROJECT_DOCUMENTS/KB/AGENTS_POINTER.md` says. No name of any language model in any committed file or commit message.
- The report, `E53_REPORT.md`, in standard academic English: full sentences, every term defined at first use, every figure with the receipt or the command that gives it; the committed files listed from `git ls-tree`.
- If the brief cannot be followed at some point, stop that part, say what you found, and ask.

## Reporting

Send the reviewer a short message with the tool `send_message` of the claude-code-remote server, to session `session_01YBMaMvtGtQvDFcgppgThrZ`, at each of these points: the checks passed (or which failed); the four runs finished, with their table; the report committed. Each message gives your branch, the commit and the main figures in at most ten lines. If you need a decision, ask in the same way and go on with whatever does not depend on the answer.

## Done

The six checks; the four runs with receipts, starting weights, final weights and answers files; `E53_REPORT.md` with the table of every run beside the trial's, the readings in the words fixed above, the outcome of predictions P53.1 to P53.5 and P53.8 (P53.6 and P53.7 are scored by the reviewer), and the list of committed files.
