# E54: the limited and the periodic network with fewer pairs taught (17 November 2026, revision 2)

Evidence level E2. **Status.** Written by the outside reviewer, acting for the coordinator, who cannot be reached while Mr. Xu is away (his instruction of 17 November: arrange it yourself). Revision 1 was checked by an agent of the reviewer that had not seen the reviewer's reasoning (`CHECK_OF_TASK_E54_BRIEF_20261117.md` beside this file); the check found eight defects and gave a text for each. Revision 2 enters all eight and adds four things of the reviewer's own, listed in the next section. The additions are given to a second check of the same kind before the brief is released. Neither check is the independent audit that the programme's rules ask for, and the coordinator has not seen the brief. The job is item 10 of `PLAN.md` beside this file, both halves. One seed, about 39 CPU-hours.

## What changed from revision 1

| | Finding of the check, or addition | What revision 2 does |
|---|---|---|
| 1 | A level at a fixed update was read as an end result, so that a network that is only slower would be reported as one that generalises worse | A difference of levels counts as a difference of end results only if the lower curve *has flattened* (defined below). Otherwise the reading is *not decided within the dose*. The ratios of updates stand beside E42's ratios at a half. The job is said not to be able to refute a small lowering of the needed fraction |
| 2 | Under back-propagation the margin of 10 points lay inside the spread between seeds, and seed 73129 is the seed on which the dense network is lowest | The margin is 20 points wherever the record shows a spread of that size or shows none; the spread is stated beside every such difference |
| 3 | The length test held taught pairs (68 of 300 at a quarter, 29 at a tenth) | The length panel is built from the pairs held out at the run's own fraction; the runner asserts that no panel of untaught pairs holds a taught pair; check 5 |
| 4 | No check could detect a wrong periodic network; theta had no value; E42's branch was not named | theta = 0.5 and the branch are given; check 1 has a dry run of `P8` against E42's receipt |
| 5 | Check 3 was not a check before the runs and could not fail | Check 3 is read row by row from the first two runs, stops every run at the first difference, and has no replacement by a comparison of scores |
| 6 | `exp42_cells.run` cannot take a taught fraction | The runner is a copy of an existing text with a list of changes, and the report shows the difference |
| 7 | Three statements about the record and the plan were inexact | Corrected in "What the record says"; the second half of plan item 10 is now part of the job |
| 8 | Smaller points (the four operations are already recorded; "three rows ending at"; restarts; the share replaced; where E42's words are defined) | Entered |
| A1 | Addition. The two fractions of revision 1 lie on either side of the fraction at which the dense network begins to generalise, so that a lowering of that fraction which stops short of a tenth could not be seen (finding 1 f) | A third taught fraction, 0.15, between them |
| A2 | Addition. With the rule of finding 1, a dose of 24,576 leaves most readings *not decided* (at a half, 6 of E42's 18 curves had not flattened by that rule at update 24,576: `G` under the sweep on seed 73129, `F10` under the sweep on all three seeds and under back-propagation on two) | Every run of the three configurations has 49,152 updates |
| A3 | Addition. Runs of two CPU-hours are exposed to restarts of the container | The runner is built on E42's resumable runner, and check 6 shows that a resumed run equals an uninterrupted one |
| A4 | Addition. The record already compares the two rules with fewer pairs in the dense network, and revision 1 did not say so | Stated in "What the record says"; the sweep against back-propagation is a second reading of every pair of runs |

## Shell names and host

`D=origin/codex/minifly-consolidated-20261023` (the default branch); `W=origin/claude/gallant-dijkstra-i9zgsr` (E42's worker's branch, at 44f91edc: E42's receipts under `$M/exp45/e42/`, its checks under `$M/exp45/e42_qual/`, and `e42_tables.py` and `exp42_ckpt.py` under `$M/`, none of which is on the default branch); `V=origin/claude/vibrant-cerf-sje726` (the reviewer's branch); `M=MOD97_LOCAL_RULE_20261106`; `R=PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109`. A line number without a file is a line of `$D:$M/exp42_cells.py`, which is identical on `$W`. Host: Linux x86-64, Python 3.13, torch 2.15.0+cpu; at most three processes, one thread each; launch outside any wrapper with a time limit.

## Terms

- *The dense network* (`G` in E42) is the network of the present size, 256 cells, each connected to every cell. *The limited network* (`F10`) gives each cell 10 fixed connections to cells of its neighbourhood on a 16 by 16 grid. *The periodic network* (`P8`) is a limited network whose cells update every 1, 2, 4 or 8 positions and whose connections are replaced during learning by E42's rule, with E42's value theta = 0.5. All three are configurations of `exp42_cells.py`. *The present width* is this size (`h512` in E41's receipts); *the half width* is E41's dense network of 128 cells (`h256`).
- *Back-propagation* (rule B) is the exact gradient; *the sweep* (rule S) passes the error backward by the signs of the weights.
- *The first family* is add, sub, max and min under modulus 89. *The taught fraction* f is the share of the pairs of operands that the stream teaches; the other pairs are *untaught*. Score A is the share right on untaught pairs of add and sub at 12 letters (300 items each); score R the share right on 1,000 untaught pairs of each of add and sub whose answer needs the reduction by the modulus, without letters; the *length test* is the share right on 300 untaught pairs of add at 40 letters.
- Rows are scored every 256 updates. *Updates to 80* on a score is the first of three consecutive scored rows at or above 80%. A *level* at an update is the mean of a score over the three scored rows that end at that update.
- A curve of R *has flattened* if the mean of R over the nine scored rows that end at the last update of the run is less than 3 points above the mean over the nine scored rows that end 8,192 updates earlier. (The scatter of R from row to row is 2.0 to 4.0 points at a quarter of the pairs in E41; a difference of two means of nine rows then has a scatter of 0.9 to 1.9 points.)

## The question

Mr. Xu, 17 November: the end result of our learner may be about the same as the conventional one's while it needs far fewer samples; it would then catch up in practice although it learns more slowly and computes more. In the programme's terms a sample is a taught pair, and "fewer samples" means that a smaller taught fraction is enough for the untaught pairs. E42 compared the limited and the periodic network with the dense one at one taught fraction, a half. **With a quarter, with 15% and with a tenth of the pairs taught, do the limited and the periodic network generalise better than the dense one, as well, or worse, and is any of the three fractions enough for one of them where it is not enough for the dense network?**

## What the record says (working rule 1)

Seeds are given in the order 73129, 93683, 116089. Every figure of this section was recomputed by the reviewer from the receipts with the script in the appendix of the check.

- **E42, three seeds, f = 0.50** (`$W:$M/exp45/e42/`). Updates to 80 on A: `G` 5,632, 4,864, 5,376 under the sweep and 5,120, 4,096, 4,352 under back-propagation; `F10` 13,312, 13,056, 12,544 and 8,704, 6,912, 9,216; `P8` 7,680, 8,448, 7,424 and 6,400, 7,168, 7,168. On seed 73129 the ratios to `G` are, on A, `F10` 2.36 under the sweep and 1.70 under back-propagation, `P8` 1.36 and 1.25; on R, `F10` 2.27 and 1.48, `P8` 1.27 and 1.12. At the end of the dose (24,576 updates) A is 91 to 99. `F10` under the sweep is 4.5 to 7.4 points of A below `G` on every seed, and its R was still rising by 4.5 to 6.8 points over the last 8,192 updates. The other differences from `G` at the end are under 5 points of R.
- **E41, the dense network at the present width with fewer pairs** (`$D:$M/exp45/e41/`; levels at update 12,288, the end of its dose). f = 0.25: under the sweep A 85.1, 83.1, 78.6 and R 80.4, 78.8, 73.9, with R rising by 1.7, 3.3 and 6.0 points over the last 2,048 updates; under back-propagation A 85.6, 92.4, 88.2 and R 83.7, 90.6, 86.0. Under back-propagation R reaches 80 at updates 7,680, 4,352 and 5,888; under the sweep it does not reach 80 within the dose on any seed (on seed 73129 the last four rows are 83.85, 79.95, 80.75 and 80.35). f = 0.10: the taught pairs are learned (97 to 99%) and the untaught ones are not: under the sweep A 9.0, 14.3, 10.3 and R 6.9, 10.8, 9.0; under back-propagation A 16.8, 30.1, 25.7 and R 14.8, 28.9, 22.2.
- **E41, the half width, 40,960 updates.** f = 0.25: R 82.8, 84.3, 82.4 under the sweep and 89.8, 96.2, 91.8 under back-propagation. f = 0.10 under back-propagation: A 47.1, 66.6, 56.4 and R 44.0, 61.7, 53.0, with R rising by 3.5 to 6.2 points over the last 8,192 updates; under the sweep A 12.4, 14.2, 14.8 and R 10.5, 10.0, 10.3, not rising. So with a tenth of the pairs the untaught pairs come late under back-propagation, if at all, and the run on file ends before it can be said which.
- **The two rules with fewer pairs.** In the dense network the sweep is behind back-propagation at both smaller fractions and both widths: by 3 to 12 points of R at a quarter and, at a tenth, at 7 to 11 against 15 to 62. On the record the sweep therefore needs more taught pairs than back-propagation and not fewer. What is open is whether a limited or a periodic network changes this.
- **The spread between seeds of the dense network alone**, range of the level of R over the three seeds: at a quarter 6.4 under the sweep and 6.9 under back-propagation (present width, update 12,288); at a tenth 3.9 under the sweep and 14.2 under back-propagation (present width, update 12,288), and 12.7 to 20.8 under back-propagation at the half width at updates 12,288, 24,576 and 40,960. Seed 73129 has the lowest dense level at a tenth under back-propagation at both widths and at each of these updates. No run at f = 0.15 exists.
- **E48** (a hint, not fixed beforehand): a new family learned from a tenth of its pairs in connections present only with its code ended higher than free learning (91 against 86% under the sweep, 97 against 90% under back-propagation at update 1,024).
- No run of a limited or a periodic network at any taught fraction but a half exists on any fetched branch.

## Runs

| Part | Taught fraction | Configurations | Rules | Updates | Runs |
|---|---|---|---|---|---|
| 1 | 0.25 | `G`, `F10`, `P8` | S, B | 49,152 | 6 |
| 2 | 0.15 | `G`, `F10`, `P8` | S, B | 49,152 | 6 |
| 3 | 0.10 | `G`, `F10`, `P8` | S, B | 49,152 | 6 |
| 4 | 0.10 | the dense network at the half width | S, B | 122,880 | 2 |

Seed 73129. The parts are run in this order; within parts 1 and 3 the two runs of `G` are launched first (check 3). `P8` is run with theta = 0.5 (`theta` in `$W:$M/exp45/e42/e42_P8_S_73129.json`; `chosen_theta` in `$W:$M/exp45/e42_qual/check10_73129.json`). Everything else in parts 1 to 3 is E42's: the configurations as E42 ran them, the optimizer, the batches of 64, 0 to 12 letters, scoring every 256 updates. Part 4 is E41's own runner, unchanged: `python3 exp41_scale.py run <S|B> 256 10 73129 --dose 122880 --out exp45/e54` (its line 143 puts `_dry122880` into the receipt's name). Part 4 may fill a free process at any time after the checks.

Cost at the rates of E42's receipts of seed 73129 (0.85, 0.82, 0.87, 0.86, 1.20 and 1.18 CPU-hours for `G`, `F10` and `P8` under the sweep and back-propagation at 24,576 updates): 11.6 CPU-hours for each of parts 1 to 3, 3.3 for part 4 (0.040 and 0.044 CPU-seconds an update, 1.5 a scored row, in E41's receipts), about 0.6 for the checks: 38.6 in all, and up to 51 at the rates in the receipts of the other two seeds.

## The runner

A new file `exp54_fraction.py` in `$M`; no existing source is edited. `exp42_cells.run` (lines 308 to 401) takes no fraction and has a half written into lines 325 and 326. E42's worker wrote a resumable form of it, `$W:$M/exp42_ckpt.py`, whose `run_ckpt` is the text of `exp42_cells.run` with saving and resuming added (its lines 63, 64 and 76 are lines 325, 326 and 336), and whose self-test shows resumed dry runs equal to uninterrupted ones in hash, rows and digest (`$W:$M/exp45/e42_qual/ckpt_selftest.json`). The new file is a copy of `$W:$M/exp42_ckpt.py` with the following changes and no others, and the report shows the difference between the two files.

1. The run takes the taught fraction f (0.50, 0.25, 0.15 or 0.10) as an argument. The stream is `E.batches([(t, E.split(t, f, seed)[0]) for t in E.DEV], seed, E.KMAX["M"])`, which is E41's stream for that fraction (`$D:$M/exp41_scale.py` lines 91 to 93), and the `data` string says so.
2. The panels of A, of R and of taught pairs are the first three that `exp42_cells.panels(seed, f)` returns; they are E41's panels at that fraction. Its fourth, the length panel, is not used: line 230 draws it from the pairs held out at a half, of which 68 of 300 are taught at a quarter and 29 at a tenth on this seed. The length panel is `E.panel(LEN_TASK, E.split(LEN_TASK, f, seed)[1], seed, ks=(12, 40), n=300)` with `exp42_cells.LEN_TASK`. Before the first update the runner asserts that no pair of the panels of A, of R and of the length test is among the taught pairs of its operation at this fraction, and that every pair of the taught panel is.
3. Every batch passes E41's assertion that only taught pairs are presented (`taught_keys` and `assert_taught`, `$D:$M/exp41_scale.py` lines 126 to 138, imported or copied as text).
4. The receipt has `experiment="E54"`, a field `fraction`, and the name `e54_<config>_<rule>_f<50|25|15|10>_<seed>.json` under `$M/exp45/e54/`, with `_dry<updates>` before the seed for any number of updates other than 49,152. `provenance` adds the hash of `exp54_fraction.py`. The files of saved state for resuming go to `$M/exp45/e54/ckpt/` and are not committed.
5. The `tensors_sha256` of the network is recorded at updates 12,288 and 24,576 as well as at the end.
6. Nothing inside `exp42_cells`, `e42_cells` or `exp11_cell` is replaced while a run executes.

E42's format already records the four operations on the panel behind A apart at every row (`A_counts` holds `add89`, `sub89`, `max89` and `min89`); nothing is to be added for them.

## Checks before the runs (a receipt for each)

1. **At f = 0.50 the runner is E42's.** Two dry runs of the new runner at f = 0.50: `F10` under the sweep, 2,048 updates, and `P8` under the sweep with theta 0.5, 1,024 updates. Each is compared with the same run made with `exp42_cells.py` on this host and with E42's receipt: `$W:$M/exp45/e42_qual/check8_73129_runs/e42_F10_S_dry2048_73129.json` (`tensors_sha256` begins `837dadc1`) and `$W:$M/exp45/e42_qual/check7_73129_runs/rep0/e42_P8_S_dry1024_73129.json` (begins `1b78f306`; connections replaced at rows 0 to 1,024: 0, 0, 173, 49, 208). Equal are the `tensors_sha256`, the digests of the mask and of `made`, the counts replaced, and the loss, the counts and the batch digest of every row. (A connection can be given up only once it is 512 updates old, so a shorter dry run of `P8` replaces nothing and tests nothing.)
2. **The stream.** The batch digest at row 4,096 begins `a71b1ca10b24a36b` at f = 0.25 and `e120924e5af546f9` at f = 0.10 (E41's receipts of seed 73129), and `a6f5362d355c59fc` at f = 0.15 (no receipt exists at this fraction; the value is from the reviewer's copy of the stream without torch, which gives the other two). It is read at row 4,096 of the first run of each part, and a difference stops the part.
3. **`G` is E41's network.** Beforehand `python3 exp42_cells.py check 1 --qual exp45/e54/qual` passes on this host (2,048 updates of `G` at a half give E26's hashes under both rules; about 9 CPU-minutes). If it does not, stop and ask; no comparison of scores takes its place. The check itself is read from the two runs of `G` at f = 0.25, which are launched first, and no run of part 2 or 3 is launched before it has passed. In every row to update 12,288 the loss, the counts of A, of R and of taught pairs and the batch digest equal those of `$D:$M/exp45/e41/e41_<rule>_h512_f25_73129.json`. The first comparison is at row 256, and a difference stops every run. At update 12,288 the `tensors_sha256` equals the receipt's. The two runs of `G` at f = 0.10 are compared in the same way with `e41_<rule>_h512_f10_73129.json`. In part 4 the rows to update 40,960 equal those of `e41_<rule>_h256_f10_73129.json` in the same quantities.
4. **Cost.** CPU-seconds an update and a scored row for each configuration from the dry runs, and a forecast for each part.
5. **The length panel.** In every scored row `length_counts["12"]` equals `A_counts["add89"]`, as it does in all eighteen receipts of `G`, `F10` and `P8` of E42: built in this way, the length panel at 12 letters is the addition panel behind A. Read at the first scored rows of every run.
6. **A resumed run equals an uninterrupted one.** `P8` under the sweep at f = 0.25, 2,048 updates, once without interruption and once stopped after update 1,024 and resumed: equal in `tensors_sha256`, in every row and in the batch digest.

## How the result is stated (fixed beforehand)

For each rule and each taught fraction, `F10` against `G` and `P8` against `G`. The report gives:

- **Updates to 80** on A and on R, with E42's words and bands (`describe` in `$W:$M/e42_tables.py`, lines 63 to 72: a ratio to `G` below 2/3 is *fewer*, above 1.5 *more*, otherwise *about the same*, with its rules for a run that does not reach 80). Beside each ratio stands the ratio of the same seed at a half, given above.
- **Levels** of A, of R and of the length test at updates 12,288, 24,576 and 49,152; the rise of R by the nine-row rule; and whether the curve has flattened.
- **Whether the fraction is enough.** A network *generalises from the fraction* if R reaches 80 within the run.
- **The comparison of end results.** A network *generalises better* than `G` if its level of R at update 49,152 is above `G`'s by the margin or more and `G`'s curve has flattened; *worse* if it is below by the margin or more and its own curve has flattened; *as well* if the difference is under the margin and both curves have flattened. In every other case the reading is *not decided within the run*, and the report gives both levels and both rises. The margin is 10 points at f = 0.25 under both rules and at f = 0.10 under the sweep, and 20 points at f = 0.10 under back-propagation and at f = 0.15 under both rules (at f = 0.15 no run exists from which a spread could be taken).
- **Max and min apart from add and sub** at every reading, since the answers to max and min need no reduction by the modulus.
- **The share of connections replaced at an event** in `P8`, over the first sixteen and the last sixteen events of each run. theta was chosen on the first 4,096 updates at a half, where this share is 7.6% early and 19.8% late under the sweep on this seed; the runs here are twice as long.
- **The sweep against back-propagation** within each configuration and fraction, as a second reading of the same runs: the difference of the levels of R at update 49,152 with the same margins and the same condition of flattening.
- **Part 4:** whether the dense network at the half width with a tenth of the pairs reaches 80 on R within 122,880 updates, and if not, its level and whether its curve has flattened.

"Needs fewer samples" is licensed, for one seed, only (i) by a limited or periodic network that generalises from a fraction at which `G`'s curve of R has flattened below 80, or (ii) by *generalises better* at any fraction. If the network reaches 80 and `G` neither reaches 80 nor has flattened, the report says "reaches 80 on R at this fraction within the run, and the dense network does not; whether the dense network would arrive is not decided". A ratio of updates to 80 that is smaller at a smaller fraction than at a half is reported as "loses less speed from fewer pairs" and licenses nothing further.

Every sentence of the report about f = 0.10 under back-propagation or about f = 0.15 names the seed and gives the spread between seeds from the record (or says that none is on record) beside the difference found.

## Predictions (the reviewer's, on file before any run)

| | Statement | Chance |
|---|---|---|
| P54.1 | At f = 0.25, `F10` needs *more* updates than `G` to 80 on A, under both rules | 75% |
| P54.2 | At f = 0.25 under the sweep, `P8` reaches 80 on R by update 24,576. (The threshold lies inside the scatter of the dense network's own late level at this fraction, so the outcome says little by itself) | 45% |
| P54.3 | At f = 0.10, neither `F10` nor `P8` has a level of R at update 49,152 above `G`'s by the margin (10 points under the sweep, 20 under back-propagation), under either rule | 85% |
| P54.4 | At f = 0.10 under back-propagation, at least one of the three networks gains 10 points or more of R between updates 24,576 and 49,152 | 45% |
| P54.5 | At f = 0.10 under the sweep, no network has a level of A at update 49,152 above 30 | 85% |
| P54.6 | At f = 0.15 under back-propagation, `G` reaches 80 on R within the run | 45% |
| P54.7 | At f = 0.15 under the sweep, `G` does not reach 80 on R within the run | 85% |
| P54.8 | At no fraction and under neither rule does `F10` or `P8` earn "needs fewer samples" | 80% |
| P54.9 | Part 4 under back-propagation: R reaches 80 within 122,880 updates | 40% |
| P54.10 | Part 4 under the sweep: the level of R at update 122,880 is below 20 | 90% |

The explanation under test: limiting the connections of a cell, or giving cells periods, lowers the number of taught pairs needed. The job can find a lowering of the needed fraction as far as 0.15 or as far as a tenth. It cannot refute a lowering that lies between two of the fractions tested. If no licence is earned, the report says "on this seed no lowering of the needed fraction to 0.15 or to a tenth was found" and lists the readings that were not decided.

## What it cannot show

- More than one seed; widths other than the present one in parts 1 to 3. The dense network alone differs between seeds by 13 to 21 points of R at a tenth under back-propagation (E41), and seed 73129 is its lowest seed there.
- The limited networks with an interface that is itself limited (E49, E50) or at a boundary (E47).
- A periodic network with fixed connections: `P8` carries E42's replacement rule, whose defect the reviewer's memo describes (22.12), and the job does not separate the periods from the replacement.
- The effect of the learning rate, which E52 varies for the dense network only.
- Whether a curve that has not flattened would arrive. Part 4 answers this for one case, the dense network at the half width.

## Rules for the worker

- At most three processes, one thread each (`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `torch.set_num_threads(1)`), launched detached and outside any wrapper with a time limit. torch 2.15.0+cpu from `https://download.pytorch.org/whl/cpu` in a virtual environment; versions recorded in every receipt.
- The container can restart without warning. After a restart, a run of parts 1 to 3 is resumed from its saved state (check 6); a run of part 4 is started again from its beginning, since E41's runner cannot resume. Commit and push receipts and logs as runs finish, and the receipt of a running run at least every two hours (the runner writes it every 4,096 updates).
- Push only to your own branch. Never push to the default branch or to any other branch, and open no pull request. Other branches are read-only: fetch with `git fetch origin '+refs/heads/<name>:refs/remotes/origin/<name>'` and read with `git show`. `exp42_ckpt.py` is taken from `$W` with `git show` and is not executed from that branch.
- Commit trailers as `PROJECT_DOCUMENTS/KB/AGENTS_POINTER.md` says. No name of any language model in any committed file or commit message.
- The report, `E54_REPORT.md`, in standard academic English: full sentences, every term defined at first use, every figure with the receipt or the command that gives it; the committed files listed from `git ls-tree`.
- If the brief cannot be followed at some point, stop that part, say what you found, and ask.

## Reporting

Send the reviewer a short message with the tool `send_message` of the claude-code-remote server, to session `session_01YBMaMvtGtQvDFcgppgThrZ`, at each of these points: checks 1, 4 and 6 passed and the pre-condition of check 3 (or which failed); check 3 passed at f = 0.25; each of parts 1, 2 and 3 finished, with its table; part 4 finished; the report committed. Each message gives your branch, the commit and the main figures in at most ten lines. If you need a decision, ask in the same way and go on with whatever does not depend on the answer.

## Done

The six checks; the twenty runs; `E54_REPORT.md` with the table of every run, the readings for each rule and fraction in the words fixed above, the outcome of each prediction, the difference between `exp54_fraction.py` and `exp42_ckpt.py`, and the list of committed files.
