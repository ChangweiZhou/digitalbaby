# E52: the two learning rules under other learning rates (17 November 2026, revision 1)

Evidence level E2. **Status.** Written by the outside reviewer, acting for the coordinator, who cannot be reached while Mr. Xu is away (his instruction of 17 November: "我人在外面你自己安排", arrange it yourself). It was checked before release by an agent of the reviewer that had not seen the reviewer's reasoning; that is not the independent audit the programme's rules ask for, and the coordinator has not seen the brief. It is item 8 of `PLAN.md` beside this file. A trial on one seed, about 9 CPU-hours.

Shell names: `D=origin/codex/minifly-consolidated-20261023` (the default branch); `V=origin/claude/vibrant-cerf-sje726` (the reviewer's branch); `M=MOD97_LOCAL_RULE_20261106`; `R=PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109`. Host: Linux x86-64, Python 3.13, torch 2.15.0+cpu; at most three processes, one thread each; launch outside any wrapper with a time limit.

## Terms

- *The network* is `arch20`'s recurrent network at a width (H, U): a byte table, a layer `l1` (8 to H), a GRU of U cells run by `arch26.forward`, and a readout. *Present width*: H = 512, U = 256. *Half width*: H = 256, U = 128. `exp41_scale.new_net(seed, H, U)` builds it.
- *Back-propagation* (rule B) is the exact gradient. *The sweep* (rule S) passes the error backward through each linear map by the signs of its weights times one scale for the map; it is `local=True` in `exp26_candidate.step`.
- *The first family* is add, sub, max and min under modulus 89. *The taught fraction* f is the share of the pairs of operands that the stream teaches; the others are never taught. Score A is the share right on never-taught pairs of add and sub at 12 letters; score R the share right on never-taught pairs of add and sub whose answer needs the modulus, without letters. Both are E41's.
- *Updates to 80* on a score: the first scored row from which three scored rows in a row are at or above 80% (E41's three-row rule; rows every 256 updates).
- *The optimizer* of every experiment since E26 is AdamW with learning rate 0.001, betas (0.9, 0.98) and weight decay 0.25 on every parameter but the byte table (`exp26_candidate.make_opt`). AdamW's decay multiplies a weight by (1 − learning rate × decay) at each update, so the shrinkage a update is 0.00025.

## The question

Every comparison of the sweep with back-propagation in the record was made at one learning rate, 0.001, which the sweep took over from back-propagation. E41 found the sweep behind: 1.1 times the updates at the present width with half of the pairs taught, more at the half width and with a quarter of the pairs, where it reaches 80 on R in 4 of 9 runs. Mr. Xu's question of 15 November is whether our learner, slower at first, overtakes later; that question is put in lives run at the half width, where the sweep is furthest behind. **Is the sweep's lag a property of the rule, or of a learning rate chosen for the other rule?**

## What the record says (working rule 1)

- **E41** (`exp41_scale.py`; three seeds; the reviewer's check `$R/E41_E46_VERIFICATION_20261117.md`). Updates to 80 on A, median of three seeds (range). Sweep, f = 0.50: half width 10,496 (10,240 to 12,288), present 5,376 (4,864 to 5,632). Back-propagation, f = 0.50: 7,680 (7,424 to 11,520) and 4,352 (4,096 to 5,120). Sweep, f = 0.25: half 17,408 (14,080 to 24,832); present 8,704, 9,984 and not reached. Back-propagation, f = 0.25: 7,680 (6,144 to 10,240) and 4,352 (3,584 to 4,864). On R at f = 0.25 the sweep does not reach 80 at the present width on any seed.
- **E46** (the same birth at decay 0, 0.05 and 0.25, present width, f = 0.50, three seeds): the decay does not change the updates to threshold under either rule; under the sweep the final level of R is lower without decay on all three seeds.
- **The learning rate has not been varied** in any run of the present network under either rule that the reviewer has found. E41's brief lists it among its limits: "Learning rate and decay were chosen at one width and are used at all three, so a width may be handicapped by them." An inventory of the record is being made (`PLAN.md`, item 3); if it finds such a run, this section is corrected before the runs are read.

## Runs

The birth of E41 (the first family alone, 0 to 12 letters, batches of 64, `exp41_scale`'s stream, panels and scores), with one thing changed: the learning rate.

| Factor | Values |
|---|---|
| Learning rate | 0.0003 and 0.003 (0.001 is E41's run of the same cell and is not run again) |
| Rule | S, B |
| Width | half, present |
| Taught fraction | 0.50, 0.25 |
| Seed | 73129 |
| Dose | E41's: 40,960 updates at the half width, 12,288 at the present width |

Sixteen runs.

**The decay a update is held fixed.** With the learning rate changed and the decay left at 0.25, the shrinkage a update would change threefold as well, and two things would vary together. The decay is therefore set to 0.25 × 0.001 / (learning rate): 0.8333 at 0.0003 and 0.08333 at 0.003, so that learning rate × decay is 0.00025 in every run, as in E41. The byte table stays without decay.

**One more value where the best is at an end.** For a rule, width and fraction, if the fewest updates to 80 on R (on A where R is not reached in any run of the cell) are at 0.0003, the cell is also run at 0.0001; if at 0.003, also at 0.01; with the decay set by the same rule. At most eight further runs.

## The runner

A new file `exp52_rate.py` in `$M`; no existing source is edited. It is `exp41_scale.run` with the learning rate and the decay as arguments: after `opt = C.make_opt(m)`, every parameter group's `lr` is set to the run's learning rate and group 0's `weight_decay` to the run's decay. The receipt (E41's format) adds `learning_rate`, `weight_decay` and the groups read back from the optimizer at the start and at the end. At the last row the runner also scores the length test that E42 and E46 use (`exp11_cell.panel(("add", 89, b"p"), split(.5, seed)[1], seed, ks=(12, 40), n=300)`; E46's runner on `origin/claude/wonderful-edison-8hx6i6` has it). Receipts in `$M/exp45/e52/e52_<rule>_h<H>_f<25|50>_lr<rate>_<seed>.json`; final weights beside them.

## Checks before the runs (a receipt for each)

1. **The runner is E41's at 0.001.** With learning rate 0.001 and decay 0.25, 2,048 updates at the present width, f = 0.50, seed 73129 give the `tensors_sha256` that E41's check 1 records for that dry run under each rule (`exp41_scale.DRY2048`: 79618ab2 under S, adf933d7 under B, their first eight digits).
2. **Only the two numbers differ.** In a run at 0.003 the optimizer's groups read back learning rate 0.003 in both groups, weight decay 0.08333 in group 0 and 0 for the byte table; the batch digest at row 4,096 equals E41's for the seed and fraction.
3. **Cost.** A dry run of 256 updates at each width with CPU-seconds an update and a scored row, and a forecast.

## How the result is stated (fixed beforehand)

For each rule, width and fraction, from the three runs at 0.0003, 0.001 (E41's) and 0.003, and a fourth where it was run:

- *The best learning rate* of the cell: the one with the fewest updates to 80 on R; where R is not reached at any rate, on A; where neither, the one with the highest mean of R over the last three rows, and the cell is marked *not reached*.
- *The rule gains from its own rate* if its updates to 80 at its best rate are at most 0.8 of its updates at 0.001, or if it reaches 80 at its best rate and does not at 0.001.
- *The lag at each rule's own rate*: the sweep's updates to 80 at its best rate divided by back-propagation's at its best rate, beside the same ratio at 0.001.
- For each cell the final levels of A and R (mean of the last three rows) and the length test at 40 letters are tabled at every rate, so that a rate that is faster to 80 and ends lower is seen.

The sentence the job can license, for one seed: "at this width and fraction the sweep's lag behind back-propagation is [the same / smaller / gone / larger] when each rule is run at its own best learning rate among those tried". Nothing is said about more than one seed until the other seeds are run.

## After the trial

The reviewer reads the trial. The two other seeds are then run for every cell in which either rule gains from its own rate, at 0.001's neighbours that matter, and the long lives' release 2 takes the sweep's best rate at the half width if it gains.

## Predictions (the reviewer's, on file before any run)

| | Statement | Chance |
|---|---|---|
| P52.1 | Under back-propagation the best learning rate is 0.001 or 0.003 in all four cells | 80% |
| P52.2 | The sweep gains from its own rate in at least two of the four cells | 45% |
| P52.3 | At the present width with a quarter of the pairs taught the sweep reaches 80 on R at some rate tried (at 0.001 it does not, on three seeds) | 35% |
| P52.4 | At the half width with half of the pairs taught, the lag at each rule's own rate is smaller than at 0.001 | 45% |
| P52.5 | Back-propagation gains from its own rate in at most one cell | 70% |

The explanation under test, with what refutes it: the sweep's lag at the half width and at a quarter of the pairs is in part a matter of the learning rate. It is refuted for this seed if the sweep gains in no cell, or if back-propagation gains as much wherever the sweep does.

## What it cannot show

- The other numbers of the optimizer (the betas, the epsilon), or another optimizer; the scale inside the sweep.
- A learning rate that changes during a run.
- Widths other than the two; fractions other than the two; more than one seed.
- Anything about a life: the rates are compared on a first learning.

## Rules for the worker

- At most three processes, one thread each (`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `torch.set_num_threads(1)`), launched detached and outside any wrapper with a time limit. torch 2.15.0+cpu from `https://download.pytorch.org/whl/cpu` in a virtual environment; versions recorded in every receipt.
- The container can restart without warning and loses what is not pushed. Commit and push receipts and logs as runs finish; a run interrupted by a restart is started again from its beginning.
- Push only to your own branch. Never push to the default branch or to any other branch, and open no pull request. Other branches are read-only: fetch with `git fetch origin '+refs/heads/<name>:refs/remotes/origin/<name>'` and read with `git show`.
- Commit trailers as `PROJECT_DOCUMENTS/KB/AGENTS_POINTER.md` says. No name of any language model in any committed file or commit message.
- The report, `E52_REPORT.md`, in standard academic English: full sentences, every term defined at first use, every figure with the receipt or the command that gives it; the committed files listed from `git ls-tree`.
- If the brief cannot be followed at some point, stop that part, say what you found, and ask.

## Reporting

Send the reviewer a short message with the tool `send_message` of the claude-code-remote server, to session `session_01YBMaMvtGtQvDFcgppgThrZ`, at each of these points: the checks passed (or which failed); the sixteen runs finished, with the table of best rates; any further runs finished; the report committed. Each message gives your branch, the commit and the main figures in at most ten lines. If you need a decision, ask in the same way and go on with whatever does not depend on the answer.

## Done

The three checks; the sixteen runs and any further runs the rule above calls for; `E52_REPORT.md` with the table of every run (updates to 80 on A and on R, final levels, the length test), the four readings for each cell, the outcome of each prediction, and the list of committed files.
