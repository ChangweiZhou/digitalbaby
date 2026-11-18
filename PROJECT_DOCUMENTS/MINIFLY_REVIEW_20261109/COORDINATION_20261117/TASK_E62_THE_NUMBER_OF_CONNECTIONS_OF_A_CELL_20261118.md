# E62: the number of connections of a cell, from 2 to 128 on a grid of powers of two, in the network with a limited input that both rules learn (18 November 2026, revision 1)

Written by the outside reviewer at Mr. Xu's instruction of 18 November ("如果暂时不行，可以采取log_2的形式，从2, 4, 8, 16, 32, 64, 128等等取一些数字来看效果如何", given after the coordinator's decision 33 was written). It is step 2 of `STEPS_FROM_THE_PRINCIPLE_20261118.md`. The coordinator has not asked for it; it is put to the coordinator for release beside E61, and a worker starts nothing before that. One seed.

## Shell names and host

`D=origin/codex/minifly-consolidated-20261023` (the default branch, at e647e7d7); `V=origin/claude/vibrant-cerf-sje726` (the reviewer's branch); `M=MOD97_LOCAL_RULE_20261106`; `R=PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109`. The host is Linux on x86-64 with four cores, Python 3.13 and torch 2.15.0+cpu. Every source file named below is on the default branch.

## Terms

- *The network* is that of E50's stage 3 (the setting `C46-one` of `exp50_code.py`) with the number and the choice of a cell's connections changed: 256 recurrent cells of the gated kind of E26; the input is a fixed code of 512 values in which each of the 46 used bytes turns on 11 values of its own, and each cell reads 46 of the values, one of every used byte; the readout is dense. The byte table and `l1` take no part.
- *N* is the number of other cells that a cell reads; it also reads itself. The connections are fixed for the run. *From all cells* means that a cell's N connections are drawn from the 255 other cells (`e42_cells.Layer(m, seed, N=N, wide=True)`); *from the neighbourhood* means from the 48 cells around it on a grid of 16 by 16 (`wide=False`). With N = 255 every cell reads every other.
- *The sweep* (rule S, `local=True`) sends the error back through each linear map by the signs of its weights times one scale for the map; *the exact gradient* (rule B, `local=False`) sends it back through the weights. Both are arms, each named for what it is (decision 31). By Mr. Xu's statements of 18 November the exact gradient is outside the principle on any network; its arm is a reference and is never called a candidate. In every run the error is passed back over the whole item, as in the record.
- *The record* is the two runs of E50's stage 3 with N = 10 from the neighbourhood, `$D:$M/exp45/e50/e50_C46-one_<S|B>_73129.json`. They are not repeated.
- A run is scored every 256 updates (a *scored row*). A threshold is *reached* at the first of three consecutive scored rows at or above it; the first single row at or above it is given in brackets. A *level* at an update is the mean of the three scored rows that end there. *A* and *R* are the shares right on never-taught pairs (A: 600 items of addition and subtraction with 12 letters; R: the 2,000 items whose answer needs the reduction by the modulus); *T* is the share right on taught pairs. These are E50's definitions and E50's scorer.

## The question

How does learning depend on the number of cells that a cell reads? Ten was fixed by us. Mr. Xu asks that a number we must fix be varied on a grid of powers of two and the effect read, before a later step asks whether a cell can set the number itself. The connections are drawn from all cells, so that the size of a neighbourhood is not a second number fixed by us; one run with N = 10 from all cells stands beside the record's N = 10 from the neighbourhood, so that the effect of that choice is measured once.

## What the record says

| Network of E50's stage 3, seed 73129, N = 10 from the neighbourhood | Exact gradient | Sweep |
|---|---|---|
| Updates to 80 on R [single row] | 15,360 [13,824] | 34,304 [34,304] |
| Updates to 80 on A [single row] | 14,080 [12,288] | 35,584 [33,536] |
| Levels of A, R, T at update 24,576 | 88.5, 88.9, 94.9 | 63.6, 65.5, 84.6 |
| Levels of A, R, T at update 49,152 | 93.8, 94.2, 98.3 | 85.9, 87.2, 94.0 |
| CPU-hours of the run (updates and scoring) | 1.90 | 2.18 |

(Verified by the reviewer: `$V:$R/E50_STAGE3_AND_REPORT_VERIFICATION_20261118.md`.) With a dense input and a learned code, E42 found on three seeds that N = 10 from the neighbourhood needs 2.36 times the updates of the dense network under the sweep and 1.70 times under back-propagation; E42 ran N = 5 and N = 20 only with connections that change during the run and only under the sweep. No run with this input exists at any other N.

## Runs

Eighteen runs of 49,152 updates on seed 73129: N = 2, 4, 8, 16, 32, 64, 128 and 255 from all cells, and N = 10 from all cells, each under the sweep and under the exact gradient. Order: N = 8, 32 and 128 first (six runs), then N = 2, 255 and 10, then N = 4, 16 and 64. Everything other than the connections is the record's: the input and its start, the stream (`E.batches(tp, seed, E.KMAX["M"])`, half the pairs), batches of 64, AdamW at rate 0.001 with betas 0.9 and 0.98 and weight decay 0.25 (`R42.make_opt`), the update (`R50.step`), the panels and the scorer.

## The runner

A new file, `$M/exp62_connections.py`; no existing source file is edited. It imports `exp50_code` as `R50`, `e42_cells` as `L42`, `exp49_interface` as `R49`, `exp42_cells` as `R42` and `exp26_candidate` as `C26` (first, as in `exp50_code.py`). It is committed before the first check is run.

    python3 exp62_connections.py run <N> <all|near> <S|B> 73129 [--dose N] [--out DIR] [--ckpt N] [--resume] [--die-after N]
    python3 exp62_connections.py check <1|2|3|5> [--qual DIR]

1. **The network** is built by a copy of the text of `exp50_code.build` for the setting `C46-one` in which the one line `layer = L42.Layer(m, seed, N=10)` is replaced by `layer = L42.Layer(m, seed, N=N, wide=(choice == "all"))`. `near` is accepted with N = 10 only (the record's network, for check 1). `L42.Layer` multiplies the kept recurrent weights at the start by the square root of 256 / (N + 1), so that the variance of a cell's recurrent input is kept; with N = 255 nothing is scaled, and the recurrent map is crossed by E26's own call (under the sweep its scale is then the mean magnitude of the whole matrix; for every other N it is the mean magnitude inside the mask). The object returned has the fields of `R50.build`'s, so that `R50.fwd`, `R50.install`, `R50.step`, `R50.counts`, `R50.fixed` and `R50.cosines` are used unchanged.
2. **The run** is a copy of the text of `exp50_code.run` with these changes and no other: `experiment="E62"`; the new entries `N` and `choice`; the network of point 1; the assertion on `WEIGHTS_TAKING_PART` is replaced by the assertion that `counts(n)["weights_taking_part"]` equals 560,128 + 768 (N + 1) (568,576 at N = 10, the record's), and the assertion on `PRODUCTS_TURNED_ON` is dropped, the count being recorded; the entries of point 3; saving and resuming (point 4); the receipt is `$M/exp45/e62/e62_N<N>_<all|near>_<rule>_73129.json` (`_dry<N>` before the seed for a run of other than 49,152 updates), the other files with E50's endings; the variable `E62_BESIDE` in place of `E50_BESIDE`; `sources_sha256` covers `exp62_connections.py` and the files of `R50.SOURCES`. Every assertion of `exp50_code.run` at a scored row is kept. The report shows the difference between the two texts.
3. **What the receipt records once, from the mask:** the number of cells that no other cell reads; the number of cells that are reached from a cell within 1, 2, 3, 4, 6 and 8 steps along the connections (a step goes from a cell to the cells that read it), as the smallest, the mean and the largest over the 256 cells; and the smallest number of steps within which every cell is reached from every cell, or that no such number exists. **At every scored row** it records in addition the mean magnitude of the kept recurrent weights and the sweep's scale for the recurrent map.
4. **Saving and resuming,** as in `$D:$M/exp52_resume.py`. A saved state is `<receipt's stem>.ckpt`: the number of updates made, `m.state_dict()`, `opt.state_dict()`, the receipt so far and the counters of CPU time. It is written at every multiple of `--ckpt` updates below the run's number (4,096 if the option is absent; 0 writes none), at a scored row, after the row is in the receipt and the receipt is written; it is written under another name and renamed with `os.replace`. With `--resume` and a saved state present the run builds the network and the optimizer as at the start, draws the first u batches again into a new digest, asserts that the digest equals the `data_digest` of the saved receipt's last scored row and that the starting file has the saved `start_tensors_sha256`, loads the two states, appends the update and the time to a list `resumes`, and goes on from update u + 1; it then refuses only the final weights and the answers file. With `--resume` and no saved state it is an ordinary start. The saved state is deleted when the final receipt is written. `--die-after N` ends the process with `os._exit(9)` after update N (for check 3 only).
5. **Files committed:** the receipts, the starting and final weights and those of update 24,576 (with `git add -f`), the answers files and the logs of every run and check.

## Checks

Checks 1, 2, 3 and 5 are made before the runs; each writes `$M/exp45/e62_qual/check<n>_73129.json` with an entry `pass`. Check 4 is made inside every run. If a check fails, the worker stops, reports the first difference and asks.

1. **With N = 10 from the neighbourhood the run is the record.** A dry run of 512 updates under each rule with `10 near`: `start_tensors_sha256` is the record's, and the scored rows of updates 0, 256 and 512 equal the record's in the loss, in every count and in the batch digest.
2. **The networks are what the brief says.** For each of the nine settings, at the start: every cell reads exactly N other cells and itself (255 at N = 255); with `all`, the connections of the 256 cells together name at least 200 different cells outside the cell's neighbourhood (for N of 4 or more); every weight outside the mask is zero; the kept weights equal those of `arch26.new_net(73129)` times the square root of 256 / (N + 1) (times 1 at N = 255); the input map, `lm`, `l2`, `l3` and the code table are bit-equal to those of `R50.build("C46-one", 73129)`; the first-batch loss is finite under both rules and the gradients of the recurrent map are zero outside the mask. The check records the figures of point 3 of the runner for each setting.
3. **A run that is killed and resumed equals an uninterrupted one.** Two dry runs of 2,048 updates with `32 all` under the sweep, each in a folder of its own: (i) with `--ckpt 0`; (ii) with `--ckpt 1024 --die-after 1700`, which ends with exit code 9, then the same command with `--resume` in place of `--die-after 1700`. The two finished receipts have the same `tensors_sha256`; their scored rows are equal in every field whose name does not begin with `cpu` or `wall`; (ii) has one entry in `resumes`, for update 1,024. **The environment of the child processes:** the harness sets variables on import that a child must not inherit, or the child is refused (this fault was met in E52, E57 and E53; decision 37). The check starts each child with `env = {k: v for k, v in os.environ.items() if not k.startswith(("E11_", "E16_", "EXP45_"))}` and `cwd` set to `$M`, as E57's check 8 does; the same holds for any script of the worker that starts runs from Python.
4. **Inside every run:** the assertions of `exp50_code.run` at every scored row (the masks, the fixed code, the byte table and `l1` bit-equal to their start). A failure stops the run.
5. **Cost.** CPU-seconds an update and a scored row for N = 2, 32 and 255 under each rule (64 updates each, after 8 that are not counted); a forecast for the eighteen runs. If the forecast exceeds 44 CPU-hours, the worker asks before starting.

An interrupted check is made again from its beginning, after its folder has been moved to `$M/exp45/e62_qual/interrupted/`.

## How the result is stated (fixed beforehand)

For each run the report gives: updates to 80 on R and on A [single row]; the levels of A, R and T at updates 24,576 and 49,152; the number of weights that take part; the CPU-seconds of the updates up to the row at which 80 on R is reached; the figures of point 3 of the runner. One table has a row for each N in rising order, with the record's N = 10 from the neighbourhood in its place, and both rules side by side with the ratio of the sweep's updates to 80 on R to the exact gradient's.

For each rule, *the reference* is its run with N = 255. An N is *as fast as the reference* if its run reaches 80 on R in at most 1.25 times the reference's updates; *slower* if it reaches 80 on R later than that within 49,152 updates; otherwise *not reached*. If the reference itself does not reach 80 on R, the rule's runs are compared by their levels of R at 49,152 and no N is called as fast. (The factor 1.25 is the reviewer's choice; on one seed a smaller difference is not read.) The sentences:

- "On seed 73129, with connections drawn from all cells, [the rule] reached 80 on R after u updates with N connections a cell, for N = 2, 4, …: …; the smallest N that was as fast as the network in which every cell reads every other was N*." If no N below 255 is as fast, the sentence says so and gives the smallest N that reached 80 at all.
- "With 10 connections from all cells [the rule] reached 80 on R after u updates, against 15,360 [34,304] with 10 connections from the neighbourhood."
- "The ratio of the sweep's updates to the exact gradient's was … at N = …", for every N at which both reached 80 on R, beside the record's 2.23.

No sentence says that a number of connections "is enough" or "is needed" without "on seed 73129" and the network. The report states that the result is of one seed (level of evidence E2), that the input is limited and the readout dense, and that the scale of the start changes with N.

## Predictions (the reviewer's, recorded before any run)

| | Statement | Chance |
|---|---|---|
| P62.1 | Checks 1, 2, 3 and 5 pass at the first attempt | 0.65 |
| P62.2 | The exact gradient reaches 80 on R within 49,152 updates at every N from 4 to 255 | 0.75 |
| P62.3 | The exact gradient reaches 80 on R within 49,152 updates at N = 2 | 0.45 |
| P62.4 | The sweep reaches 80 on R within 49,152 updates at every N from 16 to 255 | 0.55 |
| P62.5 | The sweep does not reach 80 on R within 49,152 updates at N = 2 | 0.70 |
| P62.6 | Under each rule the updates to 80 on R at N = 255 are fewer than the record's at N = 10 | 0.70 |
| P62.7 | N = 10 from all cells reaches 80 on R within 0.75 to 1.25 times the record's updates, under each rule | 0.55 |
| P62.8 | At every N at which both rules reach 80 on R, the sweep needs more updates | 0.85 |
| P62.9 | Under the exact gradient, N = 32 is as fast as the reference | 0.50 |

## What it cannot show

- Anything beyond one seed, one family, half the pairs and this network; other seeds follow only on the coordinator's word.
- A number of connections that suits a larger network. With 256 cells, a cell that reads 16 or more others reaches every cell within two or three steps, so that Mr. Xu's example of several passes (1, 50, 50², 50³) cannot be reproduced at this width; the grid shows the dependence on N at this width and nothing about how N should grow with the number of cells.
- The effect of N apart from the scale of the start, which `L42.Layer` changes with N so that a cell's recurrent input keeps its variance.
- Connections that change during the run, or a cell that sets its own number (step 6 of the chain of steps): the connections are fixed here.
- Relay over a limited number of steps (E61): the error passes over the whole item in every run.
- Requirement A, and anything about a life.

## Rules for the worker

- Start a branch of your own from `$D` at e647e7d7; fetch the reviewer's branch by `git fetch origin '+refs/heads/claude/vibrant-cerf-sje726:refs/remotes/origin/claude/vibrant-cerf-sje726'` and no other. Install torch 2.15.0+cpu from `https://download.pytorch.org/whl/cpu` and numpy in a virtual environment under Python 3.13; every receipt records the versions. Run at most three processes at a time, each with one thread (`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `torch.set_num_threads(1)`), from the folder `$M`.
- This host stops a machine about five minutes after its session's turn ends; the disk is kept and detached processes are terminated. While runs proceed, do not end your turn: issue a command that waits for the processes and repeat it until they have finished; commit and push the logs and receipts between two such commands. Start every run with `--resume` from the first time on, with its output appended to a log of its own. A run interrupted before its first saved state is started again after its files have been moved to `$M/exp45/e62/interrupted/`. The report says which runs were resumed, from which update, and which were started again.
- Push only to your own branch; never to the default branch or any other; open no pull request; read other branches with `git show`.
- Every commit carries the trailers that `PROJECT_DOCUMENTS/KB/AGENTS_POINTER.md` prescribes. Apart from the trailers, no name of a language model appears in a committed file or commit message.
- The report, `$M/exp45/e62/E62_REPORT.md`, is in standard academic English: full sentences, every term defined at first use, every figure with the receipt or command that gives it; the committed files are listed from `git ls-tree`. Its tables are made by a script `e62_tables.py` from the receipts.
- If the brief cannot be followed at some point, stop that part, say what you found, and ask.

## Reporting

Send the reviewer a short message with the tool `send_message` of the claude-code-remote server, to session `session_01YBMaMvtGtQvDFcgppgThrZ`, at each of these points: the checks passed (or which failed), with the forecast of check 5; the six first runs have written the scored row of update 24,576, with their levels of A, R and T; the six first runs finished; all runs finished, with the table; the report committed. Each message gives your branch, the commit and the main figures in at most ten lines. If you need a decision, ask in the same way and go on with whatever does not depend on the answer.

## Done

The job is done when the receipts of checks 1, 2, 3 and 5, the eighteen runs with their receipts, weights, answers files and logs, and `E62_REPORT.md` exist; the report holds the table fixed above, the readings in the words fixed above, the outcome of each prediction, the difference between the runner's `run` and `build` and those of `exp50_code.py`, and the list of committed files.

## History of the revisions

Revision 1 (18 November 2026): written at Mr. Xu's instruction of that day, for the coordinator's release.
