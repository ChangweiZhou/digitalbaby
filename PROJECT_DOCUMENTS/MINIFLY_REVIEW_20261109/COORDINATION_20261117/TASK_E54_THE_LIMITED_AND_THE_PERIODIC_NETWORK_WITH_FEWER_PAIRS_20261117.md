# E54: the limited and the periodic network with fewer pairs taught (17 November 2026, revision 1)

Evidence level E2. **Status.** Written by the outside reviewer, acting for the coordinator, who cannot be reached while Mr. Xu is away (his instruction of 17 November: arrange it yourself). It is checked before release by an agent of the reviewer that has not seen the reviewer's reasoning; that is not the independent audit the programme's rules ask for, and the coordinator has not seen the brief. It is item 10 of `PLAN.md` beside this file. One seed, about 20 CPU-hours.

Shell names: `D=origin/codex/minifly-consolidated-20261023` (the default branch); `V=origin/claude/vibrant-cerf-sje726` (the reviewer's branch); `M=MOD97_LOCAL_RULE_20261106`; `R=PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109`. Host: Linux x86-64, Python 3.13, torch 2.15.0+cpu; at most three processes, one thread each; launch outside any wrapper with a time limit.

## Terms

- *The dense network* (`G` in E42) is the network of the present size, 256 cells, each connected to every cell. *The limited network* (`F10`) gives each cell 10 fixed connections to cells of its neighbourhood on a 16 by 16 grid. *The periodic network* (`P8`) is a limited network whose cells update every 1, 2, 4 or 8 positions and whose connections are replaced during learning by E42's rule. All three are configurations of `exp42_cells.py`.
- *Back-propagation* (rule B) is the exact gradient; *the sweep* (rule S) passes the error backward by the signs of the weights.
- *The first family* is add, sub, max and min under modulus 89. *The taught fraction* f is the share of the pairs of operands that the stream teaches. Score A is the share right on never-taught pairs of add and sub at 12 letters; score R the share right on never-taught pairs of add and sub whose answer needs the modulus, without letters; the *length test* is the share right at 40 letters. *Updates to 80*: the first scored row from which three scored rows in a row are at or above 80% (rows every 256 updates).

## The question

Mr. Xu, 17 November: the end result of our learner may be about the same as the conventional one's while it needs far fewer samples; it would then catch up in practice although it learns more slowly and computes more. In the programme's terms a sample is a taught pair, and "fewer samples" means that a smaller taught fraction is enough for the never-taught pairs. E42 compared the limited and the periodic network with the dense one at one taught fraction, a half. **With a quarter and with a tenth of the pairs taught, do the limited and the periodic network generalise better than the dense one, the same, or worse?**

## What the record says (working rule 1)

- **E42, three seeds, f = 0.50** (`$R/E42_THREE_SEEDS_VERIFICATION_20261117.md`). Updates to 80 on A, seeds 73129, 93683, 116089: `G` 5,632, 4,864, 5,376 under the sweep and 5,120, 4,096, 4,352 under back-propagation; `F10` 13,312, 13,056, 12,544 and 8,704, 6,912, 9,216; `P8` 7,680, 8,448, 7,424 and 6,400, 7,168, 7,168. Final levels are close (A 90 to 98).
- **E41, the dense network at the present size with fewer pairs** (the reviewer's reading of the receipts; means of the last three rows at 12,288 updates, three seeds). f = 0.25: under the sweep A 85.1, 83.1, 78.6 and R 80.4, 78.8, 73.9, with R still rising by 2 to 6 points in 2,048 updates; under back-propagation A 85.6, 92.4, 88.2 and R 83.7, 90.6, 86.0. f = 0.10: the taught pairs are learned (97 to 99%) and the never-taught ones are not: under the sweep A 9.0, 14.3, 10.3; under back-propagation A 16.8, 30.1, 25.7; R rises by less than 1 point in 2,048 updates.
- **E41, half width, f = 0.10, 40,960 updates:** under back-propagation A 47.1, 66.6, 56.4 and still rising by 3 to 5 points in 2,048 updates; under the sweep A 12.4, 14.2, 14.8 and flat. So with a tenth of the pairs the never-taught pairs come late, if at all, and a dose of E42's length may be too short to see it.
- **E48** (a hint, not fixed beforehand): a new family learned from a tenth of its pairs in connections present only with its code ended higher than free learning (91 against 86% under the sweep, 97 against 90% under back-propagation at update 1,024).
- No run of a limited or a periodic network at any taught fraction but a half exists.

## Runs

`exp42_cells.py`'s birth with one thing changed: the taught fraction.

| Taught fraction | Configurations | Rules | Dose | Runs |
|---|---|---|---|---|
| 0.25 | `G`, `F10`, `P8` | S, B | 24,576 (E42's) | 6 |
| 0.10 | `G`, `F10`, `P8` | S, B | 49,152 | 6 |

Seed 73129. The runs at f = 0.25 first. Everything else is E42's: the configurations as E42 ran them (`P8` with E42's value of theta and its replacement of connections), the optimizer, the batches of 64, 0 to 12 letters, scoring every 256 updates.

## The runner

A new file `exp54_fraction.py` in `$M`; no existing source is edited. It is `exp42_cells.run` with the taught fraction as an argument: the stream is `E.batches([(t, E.split(t, f, seed)[0]) for t in E.DEV], seed, E.KMAX["M"])`, which is E41's stream for that fraction, and the panels are `exp42_cells.panels(seed, f)`. The receipt (E42's format) adds `fraction`. The length test stays E42's. The four operations on the panel behind A are recorded apart at every row (add, sub, max, min), as the audit of E50 asked for that job. The tensors hash of the network is recorded at update 12,288 as well as at the end. Receipts in `$M/exp45/e54/e54_<config>_<rule>_f<10|25>_<seed>.json`; final weights beside them.

## Checks before the runs (a receipt for each)

1. **At f = 0.50 the runner is E42's.** A dry run of `F10` under the sweep, 256 updates, gives the tensors hash of the same dry run made with `exp42_cells.py` itself on this host.
2. **The stream is E41's.** The batch digest at row 4,096 at f = 0.25 and at f = 0.10 equals the one in E41's receipts of seed 73129 for that fraction.
3. **`G` is E41's network.** `G` at f = 0.25 under each rule has, at update 12,288, the `tensors_sha256` of E41's receipt `e41_<rule>_h512_f25_73129.json`. (If the host's arithmetic differs from E41's host, say so and compare the scores at every row instead.)
4. **Cost.** CPU-seconds a update and a scored row for each configuration, and a forecast.

## How the result is stated (fixed beforehand)

For each rule and taught fraction, `F10` against `G` and `P8` against `G`:

- At f = 0.25: updates to 80 on A and on R, in E42's words (*fewer*, *about the same*, *more*, with E42's bands), and the levels of A, of R and of the length test (means of the last three rows).
- At f = 0.10: the levels of A and of R at updates 12,288, 24,576 and 49,152 (means of three rows), and the rise of R over the last 8,192 updates. A network *generalises from a tenth* if R reaches 80 within the dose; it is *ahead of the dense network* if its level of R at the end is 10 points or more above `G`'s, *behind* if 10 or more below, otherwise *about the same*.
- Max and min apart from add and sub at every reading, since the answers to max and min need no modulus.

The sentence the job can license, for one seed: "with a quarter (a tenth) of the pairs taught, the limited (the periodic) network generalises [better than / as well as / worse than] the dense one under [rule]". "Needs fewer samples" is licensed only by *ahead of the dense network* at f = 0.10 or by reaching 80 on R at f = 0.25 where `G` does not.

## Predictions (the reviewer's, on file before any run)

| | Statement | Chance |
|---|---|---|
| P54.1 | At f = 0.25, `F10` needs *more* updates than `G` to 80 on A, under both rules | 80% |
| P54.2 | At f = 0.25 under the sweep, `P8` reaches 80 on R within its dose | 55% |
| P54.3 | At f = 0.10, neither `F10` nor `P8` is *ahead of the dense network*, under either rule | 75% |
| P54.4 | At f = 0.10 under back-propagation, at least one of the three networks gains 10 points or more of R between updates 24,576 and 49,152 | 40% |
| P54.5 | At f = 0.10 under the sweep, no network ends above 30 on A | 75% |

The explanation under test, with what refutes it: limiting the connections of a cell, or giving cells periods, lowers the number of taught pairs needed. It is refuted for this seed if P54.3 holds and neither limited network reaches 80 on R at f = 0.25 where `G` does not.

## What it cannot show

- More than one seed; widths other than the present one.
- The limited networks with an interface that is itself limited (E49, E50) or at a boundary (E47).
- A periodic network with fixed connections: `P8` carries E42's replacement rule, whose defect the reviewer's memo describes (22.12), and the job does not separate the periods from the replacement.
- The effect of the learning rate, which E52 varies for the dense network only.

## Rules for the worker

- At most three processes, one thread each (`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `torch.set_num_threads(1)`), launched detached and outside any wrapper with a time limit. torch 2.15.0+cpu from `https://download.pytorch.org/whl/cpu` in a virtual environment; versions recorded in every receipt.
- The container can restart without warning and loses what is not pushed. Commit and push receipts and logs as runs finish, and the receipt of a running run at least every two hours; a run interrupted by a restart is started again from its beginning unless you have shown that a resumed run gives the uninterrupted run's hashes.
- Push only to your own branch. Never push to the default branch or to any other branch, and open no pull request. Other branches are read-only: fetch with `git fetch origin '+refs/heads/<name>:refs/remotes/origin/<name>'` and read with `git show`.
- Commit trailers as `PROJECT_DOCUMENTS/KB/AGENTS_POINTER.md` says. No name of any language model in any committed file or commit message.
- The report, `E54_REPORT.md`, in standard academic English: full sentences, every term defined at first use, every figure with the receipt or the command that gives it; the committed files listed from `git ls-tree`.
- If the brief cannot be followed at some point, stop that part, say what you found, and ask.

## Reporting

Send the reviewer a short message with the tool `send_message` of the claude-code-remote server, to session `session_01YBMaMvtGtQvDFcgppgThrZ`, at each of these points: the checks passed (or which failed); the six runs at f = 0.25 finished, with their table; the six runs at f = 0.10 finished; the report committed. Each message gives your branch, the commit and the main figures in at most ten lines. If you need a decision, ask in the same way and go on with whatever does not depend on the answer.

## Done

The four checks; the twelve runs; `E54_REPORT.md` with the table of every run, the readings for each rule and fraction, the outcome of each prediction, and the list of committed files.
