# E57: the sweep with a scale for each receiving unit, in the dense network, in the network with limited connections and in the network with a limited input (17 November 2026, revision 1)

**Status.** This brief was written by the outside reviewer, who acts for the coordinator; the coordinator cannot be reached while Mr. Xu is away (his instruction of 17 November: arrange it yourself). It is item 18 of `PLAN.md` beside this file. Before its release it is checked by an agent of the reviewer that has not seen the reviewer's reasoning; that check is not the independent audit that the programme's rules ask for, and the coordinator has not seen the brief. The job is at the evidence level that the coordinator's briefs call E2: a first measurement, which confirms nothing. The job has one seed and six runs and costs about 7 CPU-hours.

## Shell names and host

`D=origin/codex/minifly-consolidated-20261023` (the default branch, at 7cabb13a); `W=origin/claude/gallant-dijkstra-i9zgsr` (E42's worker's branch: E42's receipts and weights under `$M/exp45/e42/`, its checks under `$M/exp45/e42_qual/`); `W9=origin/claude/peaceful-noether-vcvrmf` (E49's worker's branch: `exp49_interface.py`, `e49_interface.py`, `e42_cells.py`, `exp42_cells.py`, and E49's receipts and weights under `$M/exp45/e49/`); `V=origin/claude/vibrant-cerf-sje726` (the reviewer's branch; this brief and the reviewer's documents); `M=MOD97_LOCAL_RULE_20261106`; `R=PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109`. The host is Linux on x86-64 with Python 3.13 and torch 2.15.0+cpu.

## Terms

- *The network* reads one byte at each position of an item. It has a table of 8 numbers for each byte (the *byte table*), a map `l1` from these to the 512 values of the byte's *code*, a recurrent layer of 256 cells, and a readout. Each cell has a reset gate, an update gate and a candidate state. The *input map* (768 by 512) takes the code to the three gates of each cell, and the *recurrent map* (768 by 256) takes the cells' previous states to them. The readout is a map `lm` from the cells' states and a map `l2` from the code to 512 units, and a map `l3` from these to the 257 possible output bytes. A *linear map* is the multiplication by one of these six weight matrices followed by the addition of its bias.
- *The dense network* (`G` in E42) has no restriction. In *the network with limited connections* (`F10`) each cell reads its own state and those of 10 fixed cells of its neighbourhood, so that each row of the recurrent map keeps 11 of its 256 entries; the other entries are zero throughout. In *the network with a limited input* (`in16` in E49) each cell of `F10`, in addition, reads 16 of the 512 values of the code, so that each row of the input map keeps 16 entries. The entries that a matrix keeps are given by its *mask*.
- *Back-propagation* (rule B) carries the error from the output toward the input; where it crosses a linear map backward, it is multiplied by the transpose of the map's weight matrix W. *The sweep* (rule S) differs in the crossing only: the error is multiplied by the transpose of sign(W) times a matrix S of non-negative magnitudes, with sign zero outside the mask. The gradient of a map's own weights is the same product of the arriving error and the map's input under both rules, and the gates are differentiated exactly under both.
- *The scale* is the choice of S. In every run so far each entry of S is one number for the whole matrix, the mean of |W| over its kept entries; this is the scale `matrix`. With the scale `row`, S[i, j] is the mean of |W| over the kept entries of row i, at every one of the six maps: the receiving unit's own incoming weights. With the scale `rowcol`, at the input map and at the recurrent map S[i, j] is the mean of row i times the mean of column j divided by the mean of the matrix (all over kept entries), and the other four maps keep the scale `matrix`.
- *The first family* is four operations on two numbers below 89 (addition and subtraction modulo 89, the larger and the smaller of the two), each written as a line of bytes with 0 to 12 capital letters that carry no information between the operands. Half of the pairs of operands of each operation are taught. Every 256 updates the network is scored; the scores of one such point are a *row*. Score A is the percentage of right answers on 600 untaught pairs of addition and subtraction written with 12 letters. Score R is the percentage on 2,000 untaught pairs of addition and subtraction whose answer needs the reduction modulo 89, written without letters. A score *reaches 80* at the first of three consecutive rows at or above 80, and *updates to 80* is the number of updates at that row. A *level* is the mean over the last three rows.
- *The cosine* of a parameter tensor on a batch is the inner product of the sweep's gradient and the exact gradient of that tensor, divided by the product of their lengths; it is 1 when the two have the same direction.
- A *receipt* is the file that a run or a check writes with its settings, its rows and the hashes of its sources. A *dry run* is a run of fewer than 24,576 updates, made to test the runner.

## The question

Under the sweep the network with a limited input does not learn the first family at all, while it learns it under back-propagation (E49). The sweep is also slower than back-propagation in the network with limited connections. In every run the sweep has used one scale for a whole matrix, and no other was tried. A measurement on the saved weights now shows that, in the networks with limited connections, a scale for each receiving unit brings the sweep's gradient closer to the exact one, most of all in the network that did not learn. **With a scale for each receiving unit, does the sweep learn the first family in the network with a limited input, and does it need fewer updates in the network with limited connections?**

The question belongs to Mr. Xu's requirement that learning be local, at the input as well: a cell that reads part of the input, and a rule that uses only what a unit has. A scale taken from a unit's own incoming weights is more local than one taken over a whole matrix.

## What the record says

Every figure is for seed 73129.

- **E42** (`$W:$M/exp45/e42/`). Updates to 80 on A and on R: the dense network 5,632 and 6,656 under the sweep, 5,120 and 6,400 under back-propagation; the network with limited connections 13,312 and 15,104 under the sweep, 8,704 and 9,472 under back-propagation. Each run cost 0.82 to 0.87 CPU-hours.
- **E49** (`$W9:$M/exp45/e49/`). The network with a limited input under back-propagation: A reaches 80 at 16,640 updates and R at 20,480; levels A 85.2, R 82.5. Under the sweep it does not learn: after 24,576 updates A is at 1.9, R at 1.4 and the taught pairs at 2.5. Each run cost 0.92 to 0.96 CPU-hours. The reviewer's verification (`$V:$R/E49_E43ADD2_VERIFICATION_20261117.md`) found that under the sweep the code shrinks and the values that a cell reads come to say little about the byte.
- **The passage of a difference at the start** (`$V:$R/SIGNAL_PASSAGE_AT_THE_START_20261117.md`). In the network with a limited input every cell reads values that differ between two bytes, and the cells carry a difference as evenly as in the network with limited connections. The start has nothing to repair.
- **The sweep's gradient under other scales** (`$V:$R/SWEEP_SCALE_VARIANTS_ALIGNMENT_20261117.md`, with `sweep_scale_variants.py`; the reviewer ran the script again). Mean cosines over 16 batches of taught items, on saved weights; nothing was trained under any scale.

| Network, weights | Tensor | Scale `matrix` | Scale `row` | Scale `rowcol` |
|---|---|---|---|---|
| Limited input, end of the sweep's run (did not learn) | byte table | 0.367 | 0.697 | 0.638 |
| | `l1` | 0.298 | 0.624 | 0.721 |
| | input map | 0.603 | 0.676 | 0.714 |
| | recurrent map | 0.651 | 0.762 | 0.710 |
| | `lm`, `l2` | 0.606, 0.643 | 0.694, 0.710 | 0.606, 0.643 |
| Limited connections, end of the sweep's run | `l1` | 0.537 | 0.631 | 0.693 |
| | input map | 0.647 | 0.696 | 0.710 |
| | recurrent map | 0.685 | 0.730 | 0.738 |
| Dense, end of the sweep's run | input map, recurrent map | 0.788, 0.819 | 0.791, 0.821 | 0.800, 0.830 |
| Limited input, start | input map, recurrent map | 0.745, 0.734 | 0.749, 0.740 | 0.745, 0.734 |

  At the start no scale changes a cosine of the two maps of the recurrent layer by more than 0.006, because the starting magnitudes are nearly equal across rows. The gain appears in weights that training has made unequal. A scale taken for each column, or the product form at every map, moves the gradients of `lm` and `l2` away from the exact ones (by 0.02 to 0.12), which is why the scale `rowcol` is applied at the recurrent layer only.
- A cosine nearer to 1 is not a demonstration that a network learns. The low cosines of the network with a limited input were measured at the end of a run that had failed; whether they are a cause of the failure or a consequence is not known. No run under another scale exists.

## Runs

| Network | Scale | Rule | Updates | Runs | Reference on file |
|---|---|---|---|---|---|
| limited input (`in16`) | `matrix` | S | 24,576 | 1 | E49's run under the sweep, which this run repeats with the cosines recorded during the run; E49's run under back-propagation |
| limited input (`in16`) | `row`, `rowcol` | S | 24,576 | 2 | the same two |
| limited connections (`F10`) | `row`, `rowcol` | S | 24,576 | 2 | E42's `F10` under the sweep and under back-propagation |
| dense (`G`) | `row` | S | 24,576 | 1 | E42's `G` under the sweep and under back-propagation |

The seed is 73129. Everything but the scale is E49's and E42's: the start, the masks, the optimizer (AdamW, rate 0.001, weight decay 0.25 on every tensor but the byte table), the stream, the panels, the rows every 256 updates. The three runs of the network with a limited input are started first.

The cost is about 0.95 CPU-hours for a run of the network with a limited input and 0.87 for the others, 5.5 CPU-hours for the six runs, and about 1.5 for the checks and the cosines recorded during the runs.

## The runner

The runner is a new file, `exp57_scale.py`, in `$M`; no existing source file is edited. `exp49_interface.py` and `e49_interface.py` are not on the default branch; they are copied byte for byte from `$W9` into `$M`, with their SHA-256 and source recorded in a text file beside them. `e42_cells.py` and `exp42_cells.py` are on the default branch with the content they have on `$W9` and are used from there. `e42_tables.py` is on `$W` only; its function `describe` is copied into the script that makes the report's tables, with its source named.

1. **The crossing.** E42's function `e42_cells.MaskedSign.apply(x, W, b, maskf, scale)` computes `linear(x, W * maskf, b)` and, backward, multiplies the error by `(W * maskf).sign() * scale`. Its argument `scale` may be a tensor of the shape of W or of one column, since the product is formed entry by entry. The new file defines

       def magnitudes(W, maskf, scale):          # W detached; maskf of W's shape, ones where the map is dense
           A = W.abs() * maskf
           mean = A.sum() / maskf.sum()
           if scale == "matrix":
               return mean
           row = A.sum(1, keepdim=True) / maskf.sum(1, keepdim=True)
           if scale == "row":
               return row
           col = A.sum(0, keepdim=True) / maskf.sum(0, keepdim=True).clamp(min=1)
           return row * col / mean               # "rowcol"; "exact" returns A itself and is used by check 2 only

       def cross(x, W, b, maskf, local, scale):
           if not local:
               return linear(x, W * maskf, b)
           return L42.MaskedSign.apply(x, W, b, maskf, magnitudes(W.detach(), maskf, scale))

2. **The forward** is a copy of the text of `e49_interface.forward` in which each of the six crossings is a call of `cross`: `l1`, `l2` and `l3` with a mask of ones; the input map with `iface.mask_in.repeat(3, 1).float()`; `lm` with `iface.mask_out.float()`; the recurrent map, at every position, with `layer.mask3().float()`, or with a mask of ones when `layer.full` is true. With the scale `rowcol` the input map and the recurrent map are given `rowcol` and the other four maps `matrix`. With the scale `row` all six are given `row`. Under back-propagation the forward computes what `e49_interface.forward` computes.
3. **The networks.** `in16` and `F10` are built by `exp49_interface.build`. `G` is built as `m, layer = exp42_cells.build("G", seed)` with `iface = e49_interface.Interface(m, seed, 512, 256)`, whose two masks are full; the run asserts that the tensors of `m` are unchanged by the construction of the interface.
4. **The run** is a copy of the text of `exp49_interface.run` with the new forward in its update, in its scoring (`install`) and in its cosines, with the scale as an argument recorded in the receipt, with `experiment="E57"`, and with the name `$M/exp45/e57/e57_<network>_<scale>_S_73129.json` (`_dry<N>` before the seed for a run of N updates other than 24,576). `provenance` adds the hash of `exp57_scale.py`. The report shows the difference between the two texts.
5. **Cosines during the run.** At update 0 and after every 2,048 updates the receipt records, on the first batch of the stream, the cosine between the sweep's gradient under the run's scale and the exact gradient for the byte table, `l1`, the input map, the recurrent map, `lm` and `l2` (over kept entries), and the same under the scale `matrix`, so that the two can be compared on the same weights. These computations do not touch the optimizer; the run asserts that the tensors and the optimizer's state are unchanged by them.
6. **Files committed:** the receipts, the starting and the final weights of every run (added with `git add -f`, since `.gitignore` excludes `*.pt`), and the logs.

## Checks before the runs

Each check writes a receipt under `$M/exp45/e57_qual/` with an entry `pass`. No run is started before checks 1 to 3 have passed. If a check fails, the worker stops, reports the first difference and asks.

1. **With the scale `matrix` the new forward is the old rule.** Dry runs of 2,048 updates under the sweep with the scale `matrix`: `G`, whose `tensors_sha256` must be E42's (`79618ab29390e6ecfdacd49a955afff7a70f282466dd92fd14e6ef3ba1ddd19f`, `DRY2048["S"]` in `exp42_cells.py`); `F10`, whose `tensors_sha256` must begin `837dadc1` (`$W:$M/exp45/e42_qual/check8_73129_runs/e42_F10_S_dry2048_73129.json`); `in16`, whose rows to update 2,048 must equal those of `$W9:$M/exp45/e49/e49_in16_S_73129.json` in the loss, in every count and in the batch digest. If a hash differs while every count and the batch digest agree and every loss agrees to 1e-6, the worker reports that the new crossing is not bit-equal to the old one, with the largest difference of a tensor, and goes on; any other difference stops the job.
2. **The scales give the gradients of the reviewer's measurement.** The final weights of E49's run of the network with a limited input under the sweep (`$W9:$M/exp45/e49/e49_in16_S_73129.pt`) are loaded into a network built by `exp49_interface.build("in16", 73129)`. On batches 2 to 17 of the stream the mean cosine with the exact gradient is, within 0.002: with `matrix`, byte table 0.367, `l1` 0.298, input map 0.603, recurrent map 0.651, `lm` 0.606, `l2` 0.643; with `row`, 0.697, 0.624, 0.676, 0.762, 0.694, 0.710; with `rowcol`, 0.638, 0.721, 0.714, 0.710, 0.606, 0.643. With the magnitudes `exact` at every map the cosine of every tensor is 1 to within 1e-5. (The figures are those of `$V:$R/sweep_scale_variants.py`, a second implementation in numpy and double precision.)
3. **Back-propagation is unchanged.** A dry run of 256 updates of `in16` under back-propagation through the new forward equals the first rows of `$W9:$M/exp45/e49/e49_in16_B_73129.json` in the loss, in every count and in the batch digest.
4. **The masks hold** (E49's check 3) at every scored row of every run: every weight outside a mask is zero, and the optimizer's two running averages there are zero. A failure stops the run.
5. **Cost:** CPU-seconds an update, a scored row and a set of cosines, from the dry runs, and a forecast. If the forecast exceeds 10 CPU-hours, the worker asks before starting.

## How the result is stated (fixed beforehand)

- **The network with a limited input.** A run *learns the family* if A and R both reach 80 within the run and their levels at the end are both at or above 80. For a run that learns it, updates to 80 on A and on R are set beside those of E49's run under back-propagation (16,640 and 20,480), with the words of `describe` in `e42_tables.py` (`$W:$M/e42_tables.py`, lines 63 to 72): a ratio below 2/3 is *fewer*, above 1.5 *more*, otherwise *about the same*. For a run that does not learn it, the report gives the levels of A, of R and of the taught pairs, and says *partly* if the level of the taught pairs at the end is 50 or more and *not learned* otherwise.
- **The network with limited connections.** Updates to 80 on A and on R against E42's run under the sweep (13,312 and 15,104) and against its run under back-propagation (8,704 and 9,472), in the same words; the levels at the end.
- **The dense network.** The same against E42's runs (5,632 and 6,656; 5,120 and 6,400). This run is a control: the measurement found no change of the gradient's direction in the dense network.
- **The cosines during the run,** for every run, as a table by update: under the run's scale and under the scale `matrix` on the same weights. For the run of the network with a limited input under the scale `matrix`, the report gives the update at which the cosine of `l1` first falls below 0.5 and the level of the taught pairs at that update.

The explanation under test: under the sweep the network with a limited input is not learned because one scale for a whole matrix misstates the sizes of weights that training makes unequal between cells, so that the error that reaches the code points in a wrong direction; a scale for each receiving unit corrects this. It is *supported*, for this seed, if the network with a limited input learns the family under the scale `row`. It is *not supported* if that run is *not learned*. If it is *partly*, or if only the scale `rowcol` learns the family, the report gives the figures without either word. A scale that needs *fewer* updates than the scale `matrix` in the network with limited connections is reported as such; it bears on the sweep's lag there and not on the explanation.

## Predictions (the reviewer's, recorded before any run)

| | Statement | Chance |
|---|---|---|
| P57.1 | Checks 1 to 3 pass at the first attempt | 75% |
| P57.2 | The network with a limited input learns the family under the scale `row` | 35% |
| P57.3 | The network with a limited input learns the family under the scale `rowcol` | 30% |
| P57.4 | Under the scale `row` the level of the taught pairs of the network with a limited input at the end is 50 or more | 50% |
| P57.5 | In the network with limited connections the scale `row` needs *about the same* updates to 80 on A as the scale `matrix` | 65% |
| P57.6 | In the dense network the scale `row` needs *about the same* updates to 80 on A as the scale `matrix` | 85% |
| P57.7 | In the run of the network with a limited input under the scale `matrix`, the cosine of `l1` falls below 0.5 before update 8,192 | 60% |

## What it cannot show

- The job has one seed.
- The job does not test the network with a limited input and a fixed code (E50's `C16-in`), which under the sweep with the scale `matrix` was still learning at the last reading of its running receipt. If the scale `row` helps here, that network is the next to try it on.
- The job does not test the scale in the network with an interface at a boundary (E47; E53), in the networks with periods, or under any learning rate but 0.001.
- The job cannot show why a scale helps or fails beyond what the cosines during the run show.
- A scale for each receiving unit uses only that unit's incoming weights. The scale `rowcol` also uses a mean over the sending unit's outgoing weights and a mean over the matrix; it is included as a comparison and is not a more local rule.

## Rules for the worker

- Start a branch of your own from `$D` at 7cabb13a. Install torch 2.15.0+cpu from `https://download.pytorch.org/whl/cpu` and numpy in a virtual environment; every receipt records the versions. Run at most three processes at a time, each with one thread (`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `torch.set_num_threads(1)`), from the folder `$M`.
- **This host stops a machine about five minutes after its session's turn ends. The disk is kept, and detached processes are terminated.** While runs proceed, do not end your turn: issue a command that waits for the processes, and, since one command may run only for a limited time, repeat it until they have finished; commit and push the logs and the receipts between two such commands. The six runs take about two and a half hours in two sets of three. A run that was interrupted is started again from its beginning, after its incomplete receipt and any file it wrote have been moved to `$M/exp45/e57/interrupted/`; the report says which runs were started again.
- Push only to your own branch. Never push to the default branch or to any other branch, and open no pull request. Do not change any other branch: fetch with `git fetch origin '+refs/heads/<name>:refs/remotes/origin/<name>'` and read with `git show`.
- Every commit carries the trailers that `PROJECT_DOCUMENTS/KB/AGENTS_POINTER.md` prescribes. Apart from the trailers, no name of any language model appears in a committed file or a commit message.
- The report, `$M/exp45/e57/E57_REPORT.md`, is written in standard academic English: full sentences, every term defined at first use, every figure with the receipt or the command that gives it; the committed files are listed from `git ls-tree`.
- If the brief cannot be followed at some point, stop that part, say what you found, and ask.

## Reporting

Send the reviewer a short message with the tool `send_message` of the claude-code-remote server, to session `session_01YBMaMvtGtQvDFcgppgThrZ`, at each of these points: checks 1 to 3 passed (or which failed); the three runs of the network with a limited input have written the row of update 8,192, with their levels of A, of R and of the taught pairs and their cosines; the six runs finished, with their table; the report committed. Each message gives your branch, the commit and the main figures in at most ten lines. If you need a decision, ask in the same way and go on with whatever does not depend on the answer.

## Done

The job is done when the receipts of the five checks, the six runs with their receipts, starting and final weights and logs, and `E57_REPORT.md` exist; the report holds the table of every run beside its references, the readings in the words fixed above, the tables of the cosines during the runs, the outcome of each prediction, the difference between the runner's `run` and `exp49_interface.run`, and the list of committed files.
