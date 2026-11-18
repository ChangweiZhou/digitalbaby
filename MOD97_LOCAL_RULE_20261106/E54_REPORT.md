# E54 report: the limited and the periodic network with fewer pairs taught (seed 73129)

**Status.** This report records a first measurement on one seed, at the evidence level that the coordinator's briefs call E2; it confirms nothing. It was written by the worker of E54 from the receipts listed below and was first committed at 87787a54. The reviewer then verified it from the committed files (`PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/E54_VERIFICATION_20261118.md` and `e54_verify.py` on `claude/vibrant-cerf-sje726` at a3827597: 1,289 of 1,295 figures reproduced); this revision corrects the six differing figures and the findings of that verification, changing the report only (no run and no receipt was changed). It has passed no automatic check of the programme's checker (which is not installed as runnable software on this branch), carries no seal or adoption by the owner, and has had no review outside the reviewer's own verification described below. The brief is `PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/COORDINATION_20261117/TASK_E54_THE_LIMITED_AND_THE_PERIODIC_NETWORK_WITH_FEWER_PAIRS_20261117.md`, revision 3, on branch `claude/vibrant-cerf-sje726` (read at commit dc27dbb5). The work is on branch `claude/e54-fewer-pairs-ELsQP`, created from the default branch `codex/minifly-consolidated-20261023` at commit 7cabb13a. Every path below is relative to `MOD97_LOCAL_RULE_20261106/` unless it is stated otherwise.

## 1. The question and the terms

Mr. Xu asked on 17 November whether the programme's learner, while learning more slowly and computing more, might need far fewer samples than the conventional network to reach about the same end result. In the programme's terms a sample is a taught pair of operands, and "fewer samples" means that a smaller taught fraction suffices for the untaught pairs. The brief therefore asks: with a quarter, with 15% and with a tenth of the pairs taught, does the limited network or the periodic network generalise better than the dense network, as well, or worse; and is any of the three fractions enough for one of them where it is not enough for the dense network?

The terms are the brief's. *The dense network* (`G`) has 256 recurrent cells, each connected to every cell. *The limited network* (`F10`) gives each cell 10 fixed connections to cells of its neighbourhood on a 16 by 16 grid. *The periodic network* (`P8`) is a limited network whose cells update every 1, 2, 4 or 8 positions and whose connections are replaced during learning by E42's rule, with theta = 0.5: at every 256th update (an *event*) a cell gives up each connection at least 512 updates old whose strength is below theta times the mean strength of its connections, and takes a new one at random in its neighbourhood. *Back-propagation* (rule B) is the exact gradient; *the sweep* (rule S) passes the error backward by the signs of the weights. The four operations are addition, subtraction, maximum and minimum under modulus 89; *the taught fraction* f is the share of the pairs of operands that are presented during learning, and the other pairs are *untaught*. Score A is the percentage right on 300 untaught pairs each of addition and subtraction with 12 uninformative capital letters between the operands; score R is the percentage right on 1,000 untaught pairs each of addition and subtraction whose answer needs the reduction by the modulus, without letters; the *length test* is the percentage right on 300 untaught pairs of addition with 40 letters. The scores are taken every 256 updates (a *scored row*). *Updates to 80* is the update of the first of three consecutive scored rows at or above 80%. A *level* at an update is the mean of a score over the three scored rows that end at that update. The *last rise* of R is the mean of R over the nine rows that end at the last update less the mean over the nine rows that end 8,192 updates earlier; the *rise before it* is that second mean less the mean over the nine rows that end 16,384 updates before the last update. A curve of R *has flattened* if both rises are smaller than 3 points in size. A network *generalises from the fraction* if R reaches 80 within the run. The readings of end results (*generalises better*, *generalises worse*, *not distinguished*, *not decided within the run*) and their margins are those fixed in the brief's section "How the result is stated"; `e54_tables.py` applies them, and its docstring repeats the definitions.

## 2. What was run

| Part | Taught fraction | Networks | Rules | Updates per run | Runs | Runner |
|---|---|---|---|---|---|---|
| 1 | 0.25 | `G`, `F10`, `P8` | S, B | 49,152 | 6 | `exp54_fraction.py` |
| 2 | 0.15 | `G`, `F10`, `P8` | S, B | 49,152 | 6 | `exp54_fraction.py` |
| 3 | 0.10 | `G`, `F10`, `P8` | S, B | 49,152 | 6 | `exp54_fraction.py` |
| 4 | 0.10 | the dense network at the half width (128 cells) | S, B | 122,880 | 2 | `exp41_scale.py`, unchanged |

The seed is 73129 throughout. The host is Linux on x86-64 with Python 3.13.18, torch 2.15.0+cpu in a virtual environment (torch's version is in every receipt; numpy 2.5.5 was installed beside it, as `pip list` in the environment showed, but numpy's version is in no receipt); at most three processes ran at a time, each with one thread (`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, and `torch.set_num_threads(1)` in the runners). The runs were launched by `e54_driver.py`, which trains nothing itself: it starts each run's own command as a separate process, in the brief's order, and applies the checks that are read from the runs (checks 2 and 3). The driver's log is `exp45/e54/logs/driver.log`, and each run's printed output is `exp45/e54/logs/<run>.log`.

**Interruptions.** The host stopped every process twice, on 18 November: the first time between 00:24:14 and 00:27:01 UTC, the second between 03:48:01 and 03:48:37 UTC (bounded by the last row printed before each stop and the next line of the driver's log, `exp45/e54/logs/driver.log`; the line written there by hand gives about 03:46 for the second stop, which is too early). Each time the driver was started again by hand, and each interrupted run of parts 1 to 3 resumed from its saved state, which the runner writes every 1,024 updates. The receipts record the resumptions under `checkpointing.resumed`: G S and F10 S at f = 0.25 at update 28,672, G B at f = 0.25 at 30,720, G S at f = 0.15 at 25,600, G B at f = 0.15 at 2,048, and F10 S at f = 0.15 at 1,024. Check 6 (below) shows that a resumed run equals an uninterrupted one in every recorded quantity, but only for `P8` under the sweep at f = 0.25, 2,048 updates, stopped by the runner's own option `--stop-after` at update 1,024. The six runs actually resumed were `G` and `F10`, two of them under back-propagation and three at f = 0.15, resumed at updates 1,024 to 30,720 after the host had stopped them. For these six the evidence is of another kind: (a) the logs were appended to, so the rows computed between the last saved state and the stop were printed twice, and all 12 such rows (2 for `G` S, 2 for `G` B and 1 for `F10` S at f = 0.25; 2 for `G` S, 3 for `G` B and 2 for `F10` S at f = 0.15) are identical at the printed precision (`exp45/e54/logs/e54_<config>_<rule>_f<fraction>.log`); (b) at every row after each resumption the batch digest equals that of the uninterrupted run of `P8` under the sweep at the same fraction (the receipts' `data_digest`); and (c) on resuming, the runner asserts that the replayed stream gives the digest stored in the saved state. This is strong evidence that the resumed runs follow their uninterrupted course, not a bitwise proof: no hash of the weights is recorded at the rows after update 12,288, the saved state itself is not hashed, and no hash of the sources is recorded at a resumption (the receipt keeps the provenance written at the first start). These omissions are inherited from `exp42_ckpt.py`; the brief does not ask for more. Part 4 was not interrupted. The CPU-hours of a resumed run count the work up to its last saved state and the work after the resumption; the updates computed between the last saved state and the stop were lost and are not counted.

## 3. The runner

`exp54_fraction.py` is a copy of `exp42_ckpt.py` from branch `claude/gallant-dijkstra-i9zgsr` at 44f91edc, taken with `git show` and never executed from that branch, with the nine changes that the brief lists under "The runner" and with check 5's assertions. In brief: the runner takes the taught fraction (`--fraction <50|25|15|10>`); the stream and the panels of A, of R and of taught pairs are E41's at that fraction; the length panel is drawn from the pairs held out at the run's own fraction; at every start it asserts that no panel of untaught pairs holds a taught pair, that the taught panel holds only taught pairs, and that the length panel at 12 letters is the addition panel behind A; `assert_taught` from `exp41_scale.py` is applied to every batch, in the loop of updates and in the replay on resuming; the default length is 49,152 updates and the receipts are `exp45/e54/e54_<config>_<rule>_f<fraction>_73129.json`; the receipt records `experiment="E54"`, `fraction`, `length_panel_is_addition_panel`, the wrapper's hash and the hashes of `exp54_fraction.py` and `exp41_scale.py`, and the hash of the weights at update 12,288 (`tensors_sha256_at["12288"]`); the self-test of `exp42_ckpt.py` is left out, `--stop-after` stays, and the printed lines say `e54`. At every scored row it asserts that `length_counts["12"]` equals `A_counts["add89"]` (check 5). Nothing inside `exp42_cells`, `e42_cells`, `exp11_cell` or `exp41_scale` is replaced, apart from E42's own `R.install(layer)`. No existing source was edited. The complete difference between the two files is in Appendix A and in `exp45/e54/exp54_fraction_vs_exp42_ckpt.diff`.

Three further files are new: `e54_checks.py` writes the receipts of the checks that are read from receipts, `e54_tables.py` computes every table and reading of this report from the receipts, and `e54_driver.py` launches the runs. None of them trains a network.

## 4. The checks before and during the runs

Every check has a receipt under `exp45/e54/qual/`, and every one passed.

1. **At f = 0.50 the runner is E42's** (`qual/check1.json`). The new runner, `exp42_cells.py` on this host and E42's receipts agree, for `F10` under the sweep at 2,048 updates and for `P8` under the sweep with theta 0.5 at 1,024 updates, in `tensors_sha256` (837dadc1… and 1b78f306…), in the digests of the mask and of `made` at the start and the end, in the counts replaced (0, 0, 173, 49 and 208 at rows 0 to 1,024 for `P8`), and in the loss, the counts and the batch digest of every row. The receipts of the dry runs are under `qual/check1_new/` and `qual/check1_e42/`.
2. **The stream** (`qual/check2_f25.json`, `check2_f15.json`, `check2_f10.json`). The batch digest at row 4,096 of the first run of each part begins `a71b1ca10b24a36b` at f = 0.25, `a6f5362d355c59fc` at f = 0.15 and `e120924e5af546f9` at f = 0.10, as the brief requires.
3. **`G` is E41's network.** The pre-condition, `python3 exp42_cells.py check 1 --qual exp45/e54/qual`, passed on this host (`qual/check1_73129.json`; the two dry runs of `G` at a half over 2,048 updates end on the tensor hashes that E26's network reaches, recorded as `exp42_cells.DRY2048`, under both rules; 675 CPU-seconds). In each of the four runs of `G` at f = 0.25 and f = 0.10, all 49 scored rows to update 12,288 equal E41's receipt `exp45/e41/e41_<rule>_h512_f<fraction>_73129.json` in the loss, the counts of A, of R and of taught pairs and the batch digest, and the hash at update 12,288 equals E41's `tensors_sha256` (`qual/check3_G_<rule>_f<fraction>.json`; the first comparison, at row 256, is in the files ending `_row256.json`; these compare only the printed scores A, R and taught at row 256, to two decimals, with E41's receipt, not the loss, the counts and the batch digest that the brief names; the first full comparison was made from the first saved state, at update 1,024, and every later comparison was full). The driver launched no run of part 2 or 3 before the check had passed at f = 0.25. In part 4 all 161 rows to update 40,960 equal `exp45/e41/e41_<rule>_h256_f10_73129.json` in the same quantities under both rules (`qual/check3_part4_S.json`, `qual/check3_part4_B.json`, and the `_row256.json` files).
4. **Cost** (`qual/check4.json`). The dry runs gave 0.135 and 0.126 CPU-seconds an update for `G` under the sweep and under back-propagation, 0.136 for `F10` under the sweep and 0.181 for `P8` under the sweep, and 7.3 to 11.9 CPU-seconds a scored row. The receipt's forecast was 15.1 CPU-hours for each of parts 1 to 3 and 49.3 for the job, below the limit of 51, so part 1 was launched without asking. The receipt forecast `F10` under back-propagation as `F10` under the sweep times 0.86/0.82, where 0.82 is E42's figure for `G` under back-propagation; with E42's own ratio for `F10` (0.86/0.87) the forecast for the job is 48.9 CPU-hours, and the check still passes.
5. **The length panel.** The assertion at every start and the assertion at every scored row held in every run; otherwise the run would have stopped. Every receipt carries `length_panel_is_addition_panel: true`.
6. **A resumed run equals an uninterrupted one** (`qual/check6.json`). `P8` under the sweep at f = 0.25, 2,048 updates, run once without interruption and once stopped after update 1,024 and started again, gave the same `tensors_sha256` (0cd4ca8e…), the same batch digest and the same nine rows in every field except the CPU and wall times; the second receipt names update 1,024 under `checkpointing.resumed`.

**Cost of the job.** The eighteen runs of parts 1 to 3 used 37.7 CPU-hours of updates, scoring and replacement (the sum of the column "CPU-hours" of the tables below), part 4 used 3.4 CPU-hours, and the dry runs of the checks 0.78 (the sum of the CPU-seconds in their receipts under `exp45/e54/qual/`; 0.6 was the brief's estimate). With the work lost at the two stops (at least 387 CPU-seconds, the updates computed between the last saved state and the stop) and the six replays of the stream on resuming (76 CPU-seconds, `cpu_replay` under `checkpointing.resumed`), the job used about 42.0 CPU-hours, against the corrected forecast of 48.9. In the full runs both rates were lower than in the dry runs: 0.100 to 0.120 CPU-seconds an update for `G` and `F10` and 0.144 to 0.162 for `P8`, and 5.5 to 6.4 and 9.3 to 10.3 CPU-seconds a scored row (`cpu_updates` divided by 49,152 and `cpu_scoring` divided by 193 in each receipt). The reason for the difference has not been examined.

## 5. Results

The tables of this section are written by `python3 e54_tables.py` from the receipts (`exp45/e54/E54_TABLES.md` and `exp45/e54/E54_TABLES.json`). All scores are percentages of items answered exactly, end symbol included. A dash means that the score did not reach 80 within the run. In the tables of updates to 80, the word is E42's (`describe` in `e42_tables.py`): if both networks reach 80, a ratio to `G` below 2/3 is *fewer*, above 1.5 *more*, otherwise *about the same*; the length entered for a run that does not reach 80 is 48,640 updates. Beside each ratio stands the ratio of the same seed at a half from E42. In the first table of each fraction, the columns "Length test (40 letters)", "max" and "min" give levels at the same three updates as A and R (12,288, 24,576 and 49,152); "max" and "min" are the percentages right on the 300 untaught pairs each of maximum and of minimum at 12 letters (`A_counts["max89"]` and `A_counts["min89"]`), whose answers need no reduction by the modulus. "CPU-hours" is the sum of the receipt's `cpu_updates`, `cpu_scoring` and `cpu_replacement`; "Resumed at" lists the updates at which the run was resumed after a stop of the host.

### Taught fraction 0.25

| Network | Rule | Updates to 80, A | Updates to 80, R | A at 12,288 / 24,576 / 49,152 | R at 12,288 / 24,576 / 49,152 | Length test (40 letters) | max | min | Rise before, last rise of R | Curve of R | CPU-hours | Resumed at |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `G` | S | 8,704 | 12,032 | 85.1 / 90.1 / 96.3 | 80.4 / 87.8 / 96.0 | 80.1 / 87.7 / 96.2 | 97.1 / 97.2 / 96.9 | 97.6 / 97.9 / 97.3 | +2.5, +1.4 | flattened | 1.97 | [28672] |
| `F10` | S | 31,744 | 40,960 | 50.7 / 70.7 / 87.1 | 42.9 / 64.8 / 82.8 | 58.1 / 76.3 / 89.3 | 92.6 / 94.2 / 95.7 | 93.6 / 95.2 / 96.0 | +4.7, +2.4 | not flattened | 1.98 | [28672] |
| `P8` | S | 10,496 | 13,568 | 83.3 / 91.4 / 91.8 | 75.8 / 87.8 / 90.7 | 81.1 / 92.2 / 92.6 | 97.0 / 98.0 / 97.9 | 97.6 / 98.8 / 98.3 | +2.1, +0.4 | flattened | 2.52 | never |
| `G` | B | 4,352 | 7,680 | 85.6 / 92.3 / 95.3 | 83.7 / 90.5 / 95.7 | 86.3 / 90.3 / 94.3 | 96.4 / 97.3 / 97.4 | 97.8 / 97.4 / 99.2 | +4.0, +0.5 | not flattened | 1.87 | [30720] |
| `F10` | B | 18,688 | 20,480 | 68.5 / 87.2 / 95.6 | 64.0 / 85.4 / 94.6 | 76.1 / 88.9 / 95.6 | 89.8 / 93.2 / 94.2 | 92.2 / 95.1 / 93.6 | +2.1, +1.0 | flattened | 1.77 | never |
| `P8` | B | 8,448 | 10,240 | 88.2 / 93.4 / 95.4 | 85.1 / 91.1 / 94.0 | 86.4 / 93.2 / 95.4 | 95.9 / 97.3 / 97.7 | 97.1 / 97.3 / 97.4 | +1.2, +0.6 | flattened | 2.47 | never |

Share of the connections of `P8` replaced at an event: rule S: 7.5% over the first sixteen events and 20.8% over the last sixteen; rule B: 7.8% over the first sixteen events and 21.3% over the last sixteen.

| Rule | Network against `G` | Updates to 80 on A: word (ratio; ratio at a half) | Updates to 80 on R: word (ratio; ratio at a half) | R at 49,152: network, `G`, difference | Margin | Rises of the network; of `G` | End result |
|---|---|---|---|---|---|---|---|
| S | `F10` | more (3.65; 2.36) | more (3.40; 2.27) | 82.8, 96.0, -13.2 | 10 | +4.7, +2.4 (not flattened); +2.5, +1.4 (flattened) | not decided within the run |
| S | `P8` | about the same (1.21; 1.36) | about the same (1.13; 1.27) | 90.7, 96.0, -5.3 | 10 | +2.1, +0.4 (flattened); +2.5, +1.4 (flattened) | not distinguished |
| B | `F10` | more (4.29; 1.70) | more (2.67; 1.48) | 94.6, 95.7, -1.1 | 10 | +2.1, +1.0 (flattened); +4.0, +0.5 (not flattened) | not decided within the run |
| B | `P8` | more (1.94; 1.25) | about the same (1.33; 1.12) | 94.0, 95.7, -1.7 | 10 | +1.2, +0.6 (flattened); +4.0, +0.5 (not flattened) | not decided within the run |

| Network | R at 49,152: sweep, back-propagation, difference | Margin | Rises under the sweep; under back-propagation | The sweep against back-propagation |
|---|---|---|---|---|
| `G` | 96.0, 95.7, +0.3 | 10 | +2.5, +1.4 (flattened); +4.0, +0.5 (not flattened) | not decided within the run |
| `F10` | 82.8, 94.6, -11.8 | 10 | +4.7, +2.4 (not flattened); +2.1, +1.0 (flattened) | not decided within the run |
| `P8` | 90.7, 94.0, -3.3 | 10 | +2.1, +0.4 (flattened); +1.2, +0.6 (flattened) | not distinguished |

### Taught fraction 0.15

| Network | Rule | Updates to 80, A | Updates to 80, R | A at 12,288 / 24,576 / 49,152 | R at 12,288 / 24,576 / 49,152 | Length test (40 letters) | max | min | Rise before, last rise of R | Curve of R | CPU-hours | Resumed at |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `G` | S | — | — | 28.9 / 36.6 / 52.7 | 26.2 / 32.1 / 46.6 | 29.4 / 34.6 / 52.6 | 92.1 / 92.6 / 92.9 | 90.4 / 92.4 / 94.9 | +5.0, +3.7 | not flattened | 1.75 | [25600] |
| `F10` | S | — | — | 15.2 / 22.7 / 32.5 | 12.4 / 19.1 / 29.7 | 18.6 / 25.0 / 38.0 | 84.3 / 86.1 / 88.6 | 77.2 / 81.0 / 85.6 | +3.7, +3.0 | not flattened | 1.78 | [1024] |
| `P8` | S | — | — | 36.1 / 48.5 / 62.5 | 31.7 / 42.6 / 56.6 | 40.4 / 51.4 / 66.4 | 92.8 / 93.8 / 95.9 | 92.1 / 92.2 / 92.9 | +5.6, +2.1 | not flattened | 2.59 | never |
| `G` | B | — | — | 58.0 / 64.8 / 77.6 | 53.9 / 60.8 / 73.0 | 59.9 / 67.4 / 81.7 | 94.2 / 96.8 / 96.4 | 93.9 / 94.6 / 96.7 | +4.4, +4.6 | not flattened | 1.67 | [2048] |
| `F10` | B | — | — | 33.1 / 41.8 / 60.9 | 29.9 / 38.4 / 56.8 | 37.7 / 48.8 / 70.3 | 89.6 / 90.0 / 91.3 | 85.2 / 87.1 / 86.0 | +5.5, +7.8 | not flattened | 1.74 | never |
| `P8` | B | 47,872 | — | 43.6 / 57.1 / 78.3 | 38.8 / 51.6 / 72.9 | 45.1 / 59.9 / 79.4 | 93.1 / 94.9 / 95.3 | 89.2 / 91.6 / 94.7 | +8.0, +5.5 | not flattened | 2.56 | never |

Share of the connections of `P8` replaced at an event: rule S: 7.9% over the first sixteen events and 20.5% over the last sixteen; rule B: 7.8% over the first sixteen events and 20.2% over the last sixteen.

| Rule | Network against `G` | Updates to 80 on A: word (ratio; ratio at a half) | Updates to 80 on R: word (ratio; ratio at a half) | R at 49,152: network, `G`, difference | Margin | Rises of the network; of `G` | End result |
|---|---|---|---|---|---|---|---|
| S | `F10` | neither reaches (—; 2.36) | neither reaches (—; 2.27) | 29.7, 46.6, -16.9 | 20 | +3.7, +3.0 (not flattened); +5.0, +3.7 (not flattened) | not decided within the run |
| S | `P8` | neither reaches (—; 1.36) | neither reaches (—; 1.27) | 56.6, 46.6, +10.0 | 20 | +5.6, +2.1 (not flattened); +5.0, +3.7 (not flattened) | not decided within the run |
| B | `F10` | neither reaches (—; 1.70) | neither reaches (—; 1.48) | 56.8, 73.0, -16.2 | 20 | +5.5, +7.8 (not flattened); +4.4, +4.6 (not flattened) | not decided within the run |
| B | `P8` | not decided (—; 1.25) | neither reaches (—; 1.12) | 72.9, 73.0, -0.1 | 20 | +8.0, +5.5 (not flattened); +4.4, +4.6 (not flattened) | not decided within the run |

| Network | R at 49,152: sweep, back-propagation, difference | Margin | Rises under the sweep; under back-propagation | The sweep against back-propagation |
|---|---|---|---|---|
| `G` | 46.6, 73.0, -26.4 | 20 | +5.0, +3.7 (not flattened); +4.4, +4.6 (not flattened) | not decided within the run |
| `F10` | 29.7, 56.8, -27.1 | 20 | +3.7, +3.0 (not flattened); +5.5, +7.8 (not flattened) | not decided within the run |
| `P8` | 56.6, 72.9, -16.4 | 20 | +5.6, +2.1 (not flattened); +8.0, +5.5 (not flattened) | not decided within the run |

### Taught fraction 0.10

| Network | Rule | Updates to 80, A | Updates to 80, R | A at 12,288 / 24,576 / 49,152 | R at 12,288 / 24,576 / 49,152 | Length test (40 letters) | max | min | Rise before, last rise of R | Curve of R | CPU-hours | Resumed at |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `G` | S | — | — | 9.0 / 11.8 / 15.2 | 6.9 / 10.1 / 15.1 | 9.8 / 12.8 / 18.6 | 87.8 / 87.3 / 88.3 | 87.1 / 88.0 / 90.6 | +2.0, +1.6 | flattened | 1.86 | never |
| `F10` | S | — | — | 5.3 / 8.7 / 8.5 | 5.3 / 7.3 / 9.3 | 6.4 / 10.8 / 10.7 | 69.2 / 72.4 / 79.1 | 68.7 / 71.4 / 73.6 | +1.1, -0.1 | flattened | 1.95 | never |
| `P8` | S | — | — | 11.8 / 13.8 / 17.4 | 10.1 / 13.2 / 19.1 | 13.7 / 15.0 / 20.9 | 82.8 / 87.0 / 91.3 | 86.2 / 87.9 / 89.2 | +2.2, +1.8 | flattened | 2.77 | never |
| `G` | B | — | — | 16.8 / 21.4 / 26.5 | 14.8 / 19.5 / 24.6 | 18.0 / 24.0 / 28.4 | 90.2 / 89.9 / 90.2 | 92.4 / 91.2 / 92.7 | +2.5, +1.7 | flattened | 1.82 | never |
| `F10` | B | — | — | 9.2 / 11.9 / 17.1 | 8.3 / 10.7 / 17.3 | 13.3 / 17.2 / 23.2 | 77.4 / 78.3 / 84.3 | 81.6 / 79.4 / 85.1 | +2.1, +2.0 | flattened | 1.90 | never |
| `P8` | B | — | — | 19.1 / 24.6 / 33.3 | 15.5 / 21.2 / 29.1 | 23.9 / 28.9 / 37.7 | 83.7 / 86.2 / 89.0 | 87.2 / 88.8 / 87.6 | +2.8, +2.8 | flattened | 2.73 | never |

Share of the connections of `P8` replaced at an event: rule S: 7.2% over the first sixteen events and 19.4% over the last sixteen; rule B: 8.1% over the first sixteen events and 19.5% over the last sixteen.

| Rule | Network against `G` | Updates to 80 on A: word (ratio; ratio at a half) | Updates to 80 on R: word (ratio; ratio at a half) | R at 49,152: network, `G`, difference | Margin | Rises of the network; of `G` | End result |
|---|---|---|---|---|---|---|---|
| S | `F10` | neither reaches (—; 2.36) | neither reaches (—; 2.27) | 9.3, 15.1, -5.7 | 10 | +1.1, -0.1 (flattened); +2.0, +1.6 (flattened) | neither generalises from this fraction; not distinguished |
| S | `P8` | neither reaches (—; 1.36) | neither reaches (—; 1.27) | 19.1, 15.1, +4.0 | 10 | +2.2, +1.8 (flattened); +2.0, +1.6 (flattened) | neither generalises from this fraction; not distinguished |
| B | `F10` | neither reaches (—; 1.70) | neither reaches (—; 1.48) | 17.3, 24.6, -7.3 | 20 | +2.1, +2.0 (flattened); +2.5, +1.7 (flattened) | neither generalises from this fraction; not distinguished |
| B | `P8` | neither reaches (—; 1.25) | neither reaches (—; 1.12) | 29.1, 24.6, +4.5 | 20 | +2.8, +2.8 (flattened); +2.5, +1.7 (flattened) | neither generalises from this fraction; not distinguished |

| Network | R at 49,152: sweep, back-propagation, difference | Margin | Rises under the sweep; under back-propagation | The sweep against back-propagation |
|---|---|---|---|---|
| `G` | 15.1, 24.6, -9.5 | 20 | +2.0, +1.6 (flattened); +2.5, +1.7 (flattened) | not distinguished |
| `F10` | 9.3, 17.3, -8.0 | 20 | +1.1, -0.1 (flattened); +2.1, +2.0 (flattened) | not distinguished |
| `P8` | 19.1, 29.1, -10.0 | 20 | +2.2, +1.8 (flattened); +2.8, +2.8 (flattened) | not distinguished |

### Part 4: the dense network at the half width, taught fraction 0.10, 122,880 updates

| Rule | Updates to 80, R | Updates to 80, A | A at 12,288 / 24,576 / 49,152 / 122,880 | R at 12,288 / 24,576 / 49,152 / 122,880 | Rise before, last rise of R | Curve of R | CPU-hours |
|---|---|---|---|---|---|---|---|
| S | — | — | 7.0 / 8.0 / 12.9 / 17.7 | 7.0 / 8.1 / 10.1 / 13.4 | +1.4, -0.5 | flattened | 1.77 |
| B | — | 112,384 | 28.6 / 35.6 / 52.7 / 79.8 | 26.8 / 32.9 / 49.9 / 75.0 | +2.5, -0.7 | flattened | 1.65 |

## 6. The readings in the words fixed by the brief

**Taught fraction 0.25** (part 1; tables in section 5).

- *Updates to 80.* Under both rules `F10` needs *more* updates than `G` to reach 80 on A (ratios 3.65 under the sweep and 4.29 under back-propagation, against 2.36 and 1.70 at a half) and on R (3.40 and 2.67, against 2.27 and 1.48). `P8` needs *about the same* number as `G` under the sweep on A and on R (1.21 and 1.13, against 1.36 and 1.27 at a half); under back-propagation it needs *more* on A (1.94, against 1.25) and *about the same* on R (1.33, against 1.12). Under the sweep, the ratios of `P8` are smaller at a quarter than at a half; in the brief's words, `P8` under the sweep "is slowed less by the smaller fraction", which licenses nothing further.
- *Whether the fraction is enough.* All six runs (three networks under two rules) generalise from this fraction: R reaches 80 within the run in every case.
- *End results against `G`.* Under the sweep, `P8` and `G` are *not distinguished*: R at update 49,152 is 90.7 for `P8` and 96.0 for `G`, a difference of −5.3 points against a margin of 10, and both curves have flattened (rises +2.1 and +0.4 for `P8`, +2.5 and +1.4 for `G`). Under the sweep, `F10` against `G` is *not decided within the run*: R is 82.8 against 96.0 (−13.2 points), but the curve of `F10` has not flattened (rises +4.7 and +2.4). Under back-propagation both comparisons are *not decided within the run*, because the curve of `G` has not flattened (rises +4.0 and +0.5): `F10` 94.6 against 95.7 (−1.1) and `P8` 94.0 against 95.7 (−1.7), with flattened curves for `F10` (+2.1, +1.0) and `P8` (+1.2, +0.6).
- *The sweep against back-propagation.* For `P8` the two rules are *not distinguished* (90.7 against 94.0, −3.3 points, margin 10, both curves flattened). For `G` (+0.3 points) and for `F10` (−11.8 points) the reading is *not decided within the run*, since the curve of `G` under back-propagation and the curve of `F10` under the sweep have not flattened.
- *Max and min.* The untaught pairs of maximum and minimum are answered at 89.8 to 99.2% at every level reported (columns "max" and "min").
- *Replacement in `P8`.* 7.5% of the connections were replaced at an event over the first sixteen events and 20.8% over the last sixteen under the sweep, and 7.8% and 21.3% under back-propagation; in E42's run of `P8` under the sweep at a half on this seed, 24,576 updates long, the shares were 7.6% over its first sixteen events and 19.8% over its last sixteen (E42's receipt `exp45/e42/e42_P8_S_73129.json` on `claude/gallant-dijkstra-i9zgsr`).

**Taught fraction 0.15** (part 2). No spread between seeds is on record at this fraction, and this job has one seed (73129); every statement below is for that seed only.

- *Updates to 80.* No network reaches 80 on R within the run, so every reading of updates to 80 on R is *neither reaches*. On A only `P8` under back-propagation reaches 80, at update 47,872; since that is more than 32,256 updates, the reading against `G` is *not decided*. All other readings on A are *neither reaches*.
- *Whether the fraction is enough.* No network generalises from this fraction within the run.
- *End results against `G`.* No curve of R has flattened: every last rise is between +2.1 and +7.8 points and every rise before it between +3.7 and +8.0. Every comparison is therefore *not decided within the run*: under the sweep `F10` 29.7 against `G` 46.6 (−16.9) and `P8` 56.6 against 46.6 (+10.0); under back-propagation `F10` 56.8 against 73.0 (−16.2) and `P8` 72.9 against 73.0 (−0.1); the margin is 20 points.
- *The sweep against back-propagation.* For every network the sweep's level of R is below back-propagation's (by 26.4 points for `G`, 27.1 for `F10` and 16.4 for `P8`), but no curve has flattened, so every reading is *not decided within the run*.
- *Replacement in `P8`.* 7.9% and 20.5% under the sweep, 7.8% and 20.2% under back-propagation.

**Taught fraction 0.10** (part 3).

- *Updates to 80.* No network reaches 80 on A or on R; every reading is *neither reaches*.
- *End results against `G`.* All six curves of R have flattened (every rise is between −0.1 and +2.8 points), and no network generalises from this fraction. Under the sweep, with a margin of 10 points: neither generalises from this fraction; `F10` and `G` are *not distinguished* (9.3 against 15.1, −5.7), and `P8` and `G` are *not distinguished* (19.1 against 15.1, +4.0). Under back-propagation, with a margin of 20 points: neither generalises from this fraction; `F10` and `G` are *not distinguished* (17.3 against 24.6, −7.3, on seed 73129; the dense network's level of R under back-propagation at a tenth differs between E41's three seeds by 14.2 points at the present width (update 12,288) and by 12.7 to 20.8 points at the half width, seed 73129 being its lowest seed there, and no spread of a difference between two networks is on record), and `P8` and `G` are *not distinguished* (29.1 against 24.6, +4.5). For seed 73129 the difference of `P8` under back-propagation is +4.5 points; the dense network's level of R under back-propagation at this fraction differs between the three seeds of E41 by 14.2 points at the present width at update 12,288 and by 12.7 to 20.8 points at the half width, and seed 73129 is its lowest seed there.
- *The sweep against back-propagation.* All three pairs are *not distinguished* (margin 20): `G` 15.1 against 24.6 (−9.5), `F10` 9.3 against 17.3 (−8.0), `P8` 19.1 against 29.1 (−10.0). In each, the sweep's level is the lower one. These differences are for seed 73129; the dense network's level of R under back-propagation at a tenth differs between E41's three seeds by 14.2 points at the present width (update 12,288) and by 12.7 to 20.8 points at the half width, seed 73129 being its lowest seed there, and no spread of a difference between two networks is on record.
- *Max and min.* Even here the untaught pairs of maximum and minimum are answered at 68.7 to 92.7%, whereas A, on addition and subtraction, stays at or below 33.3.
- *Replacement in `P8`.* 7.2% and 19.4% under the sweep, 8.1% and 19.5% under back-propagation.

**Part 4.** The dense network at the half width (128 cells) with a tenth of the pairs taught does not generalise from this fraction within 122,880 updates under either rule, which is three times the length of E41's runs. Under back-propagation its level of R rises from 26.8 at update 12,288 to 49.9 at update 49,152 and 75.0 at update 122,880; one single scored row reaches 83.25 (update 116,992), but R never stays at or above 80 for three consecutive rows. The rise before the last rise is +2.5 points and the last rise −0.7 points, so by the brief's rule its curve has flattened, at a level of about 75. Its score A reaches 80 at update 112,384 and ends at a level of 79.8. The means of nine rows of R were 42.5 at update 40,960, where E41's run ended, 61.8 at 73,728, 69.0 at 90,112, 73.8 at 106,496, 76.3 at 114,688 and 75.5 at 122,880 (computed from the receipt `exp45/e54/e41_B_h256_f10_dry122880_73129.json`). Under the sweep its level of R is 7.0, 8.1, 10.1 and 13.4 at the same four updates, with rises +1.4 and −0.5: it has flattened far below 80. Both statements are for seed 73129, which is the dense network's lowest seed at this fraction under back-propagation in E41; the spread between seeds there is 12.7 to 20.8 points.

**"Needs fewer samples."** No network *generalises better* than `G` at any fraction under either rule, so the report is not licensed to say that the limited or the periodic network needs fewer samples. In the words fixed by the brief: on this seed no lowering of the needed fraction to 0.15 or to a tenth was found. The readings that were not decided are: at f = 0.25, `F10` against `G` under both rules and `P8` against `G` under back-propagation; at f = 0.15, all four comparisons with `G` (differences −16.9, +10.0, −16.2 and −0.1 points, on seed 73129; no spread between seeds is on record at f = 0.15). The readings at f = 0.10 under back-propagation (−7.3 and +4.5 points) are for seed 73129; the dense network's level of R under back-propagation at a tenth differs between E41's three seeds by 14.2 points at the present width (update 12,288) and by 12.7 to 20.8 points at the half width, seed 73129 being its lowest seed there, and no spread of a difference between two networks is on record. Of the twelve readings against `G`, five are decided and seven are not. As the brief states, this job cannot refute a lowering: a lowering that lies between two of the fractions tested is not visible, and at f = 0.15 every reading was left not decided because no curve flattened within 49,152 updates. No case of a network that reaches 80 on R where `G` does not occurred, so neither of the brief's fixed sentences for that case applies.

## 7. The predictions

The reviewer recorded these predictions before any run (brief, section "Predictions"); the outcomes are computed by `e54_tables.py` and listed in `exp45/e54/E54_TABLES.json` under `predictions`.

| | Statement (abbreviated) | Chance given | Outcome | Figures |
|---|---|---|---|---|
| P54.1 | At f = 0.25, `F10` needs *more* updates than `G` to 80 on A, under both rules | 75% | holds | *more* under both (3.65, 4.29) |
| P54.2 | At f = 0.25 under the sweep, `P8` reaches 80 on R by update 24,064 | 45% | holds | 13,568 |
| P54.3 | At f = 0.10, neither `F10` nor `P8` is above `G` by the margin, under either rule | 85% | holds | −5.7, +4.0 (margin 10); −7.3, +4.5 (margin 20); seed 73129; the dense network's level of R under back-propagation at a tenth differs between E41's three seeds by 14.2 points at the present width (update 12,288) and by 12.7 to 20.8 points at the half width, seed 73129 being its lowest seed there, and no spread of a difference between two networks is on record |
| P54.4 | At f = 0.10 under back-propagation, some network gains 10 points of R or more between updates 24,576 and 49,152 | 45% | does not hold | gains 5.1 (`G`), 6.6 (`F10`), 7.9 (`P8`); seed 73129; the dense network's level of R under back-propagation at a tenth differs between E41's three seeds by 14.2 points at the present width (update 12,288) and by 12.7 to 20.8 points at the half width, seed 73129 being its lowest seed there, and no spread of a difference between two networks is on record |
| P54.5 | At f = 0.10 under the sweep, no network has A above 30 at update 49,152 | 85% | holds | 15.2, 8.5, 17.4 |
| P54.6 | At f = 0.15 under back-propagation, `G` reaches 80 on R within the run | 45% | does not hold | does not reach 80; level 73.0; seed 73129; no spread between seeds is on record at f = 0.15 |
| P54.7 | At f = 0.15 under the sweep, `G` does not reach 80 on R within the run | 85% | holds | level 46.6 |
| P54.8 | At no fraction and under neither rule does `F10` or `P8` generalise better than `G` | 80% | holds | no case of *generalises better*; but 7 of its 12 readings are *not decided within the run* (all four at f = 0.15, both of `F10` at f = 0.25, and `P8` under back-propagation at f = 0.25), so no such case was found among the five decided readings |
| P54.9 | Part 4 under back-propagation: R reaches 80 within 122,880 updates | 40% | does not hold | R does not reach 80; level 75.0 at update 122,880, flattened |
| P54.10 | Part 4 under the sweep: R at update 122,880 is below 20 | 90% | holds | level 13.4 |

## 8. What the job cannot show

- The job has one seed and, in parts 1 to 3, one width. The dense network alone differs between seeds by 13 to 21 points of R at a tenth under back-propagation (E41), and seed 73129 is its lowest seed there. Differences of a few points between networks on this seed may not recur on others.
- Every margin rests on the spread between seeds of the dense network alone, at other widths or updates: 14.2 points at the present width at update 12,288, and 12.7, 20.8 and 17.7 points at the half width at updates 12,288, 24,576 and 40,960 (E41's receipts, `exp45/e41/`). No spread of a difference between two networks is on record, and none of `F10` or `P8` at these fractions; at f = 0.15 the margin of 20 points has no spread behind it at all.
- The taught pairs at the three fractions are not subsets of one another: `exp11_cell.split` shuffles the pairs with a seed that includes the fraction, so of the 792 pairs of addition taught at f = 0.10 only 199 are taught at f = 0.25 and 107 at f = 0.15. A comparison across fractions therefore changes which pairs are taught as well as how many.
- The readings at f = 0.10 under back-propagation rest on two verdicts "flattened" that lie close to the boundary of 3 points. For `P8` the larger rise is 2.78 points, so a single scored row of R about 2.0 points higher in the middle window of nine rows would make the curve "not flattened" and the reading against `G` "not decided within the run"; the brief gives the scatter of R from row to row as 2.0 to 4.0 points. For `G` the larger rise is 2.46 points (4.9 points in one row); if that curve had not flattened, both readings under back-propagation at f = 0.10 would be "not decided". In this job's own part 4, on the dense network at the half width under back-propagation at the same fraction, the rule first says "flattened" at update 78,592, at a level of R of 61.1, after which the level of R rose by up to 17.4 points (to the highest level later in the run; 13.8 points to the end). So "flattened" at a tenth is not evidence of a final level.
- At f = 0.15 no curve flattened within 49,152 updates, so the job says nothing about end results at that fraction. Whether a longer run would separate the networks there is open.
- The job does not test limited networks with an interface that is itself limited (E49, E50) or at a boundary (E47), and it does not test the effect of the learning rate (E52).
- `P8` carries E42's replacement rule, whose defect section 22.12 of the reviewer's memo describes; the job does not separate the periods from the replacement, and it does not test a periodic network with fixed connections.
- Part 4 answers for one case only, the dense network at the half width with a tenth of the pairs, whether a curve that had not flattened in E41 reaches 80 with more updates.

## Appendix A. The difference between `exp54_fraction.py` and `exp42_ckpt.py`

Produced by `diff -u` of `git show origin/claude/gallant-dijkstra-i9zgsr:MOD97_LOCAL_RULE_20261106/exp42_ckpt.py` against `exp54_fraction.py`; the same text is in `exp45/e54/exp54_fraction_vs_exp42_ckpt.diff`.

```diff
--- exp42_ckpt.py (claude/gallant-dijkstra-i9zgsr at 44f91edc)
+++ exp54_fraction.py
@@ -1,22 +1,23 @@
-"""E42 checkpointing wrapper around exp42_cells.run (added 2026-11-16 after container restarts; exp42_cells.py is NOT edited).
+"""E54 runner (brief TASK_E54_THE_LIMITED_AND_THE_PERIODIC_NETWORK_WITH_FEWER_PAIRS_20261117.md, revision 3): E42's configurations taught
+a fraction of the pairs other than a half. This file is a copy of exp42_ckpt.py (branch claude/gallant-dijkstra-i9zgsr at 44f91edc) with
+the changes that the brief lists under "The runner" (1 to 9) and the assertion of its check 5; exp42_cells.py is NOT edited.
+
+run_ckpt() is exp42_cells.run's text with saving and resuming (see exp42_ckpt.py), and with:
+  * the taught fraction f: the stream and the panels of A, of R and of taught pairs are E41's at that fraction; the length panel is drawn
+    from the pairs held out at f; assertions at every start that the panels of untaught pairs hold no taught pair, that the taught panel
+    holds only taught pairs, and that the length panel at 12 letters is the addition panel behind A;
+  * exp41_scale.assert_taught on every batch (in the loop of updates and in the replay on resuming);
+  * 49,152 updates by default; receipts e54_<config>_<rule>_f<50|25|15|10>[_dry<updates>]_<seed>.json under exp45/e54/;
+  * the hash of the weights at update 12,288 (tensors_sha256_at["12288"]);
+  * at every scored row, length_counts["12"] == A_counts["add89"].
 
-run_ckpt() is exp42_cells.run's text with two additions, so that a run killed by a container restart continues instead of restarting:
-  * every CK updates (right after a scored row, when the loss list is empty and the replacement count is reset) the whole in-memory
-    state is written atomically to exp45/e42/ckpt/<receipt stem>.ckpt.pt: model state_dict, the layer's mutable state (mask, made,
-    a, Wx), the optimizer state_dict, the replacement Random's state, torch's and Python's global RNG states, the receipt so far
-    (curve included), the CPU counters, and the first batch;
-  * on start, if that checkpoint exists, everything is restored and the data stream is replayed to the checkpoint's update
-    (the batches are regenerated by the same generator and fed to the same digest), then training continues.
-The update loop, scoring, replacement and final writes are otherwise exp42_cells.run's. Correctness is tested, not assumed:
-`python3 exp42_ckpt.py selftest` interrupts dry runs and requires the resumed result bit-equal to the uninterrupted one.
-
-usage: python3 exp42_ckpt.py run <cfg> <S|B> <seed> [--theta T] [--dose D] [--stop-after U] [--out DIR] [--ckdir DIR]
-       sh exp45/e42/selftest_case.sh <i>   (i = 0..3), then python3 exp42_ckpt.py selftest
+usage: python3 exp54_fraction.py run <cfg> <S|B> <seed> --fraction <50|25|15|10> [--theta T] [--dose D] [--stop-after U] [--out DIR] [--ckdir DIR]
 """
 import sys as _sys
 from pathlib import Path as _P
 _sys.path.insert(0, str(_P(__file__).resolve().parent))   # as exp42_cells.py does (python3 -I omits the script folder)
 import exp42_cells as R          # imports the harness in the runner's order (exp26_candidate first)
+from exp41_scale import item_key, taught_keys, assert_taught   # after exp42_cells; none of them uses exp41_scale.FRACS
 import json
 import os
 import random
@@ -28,7 +29,15 @@
 T = R.T
 CK = 1024
 HERE = Path(__file__).resolve().parent
-CKDIR = HERE / "exp45" / "e42" / "ckpt"
+OUT = HERE / "exp45" / "e54"
+CKDIR = OUT / "ckpt"
+DOSE = 49152
+FRACTIONS = (50, 25, 15, 10)
+
+
+def receipt_path(out, config, rule, fp, seed, dose):
+    tag = "" if dose == DOSE else f"_dry{dose}"
+    return Path(out) / f"e54_{config}_{rule}_f{fp}{tag}_{seed}.json"
 
 
 def _save(path, obj):
@@ -38,14 +47,16 @@
     os.replace(tmp, path)
 
 
-def run_ckpt(config, rule, seed, out, cmd, theta=None, dose=None, stop_after=None, ckdir=CKDIR):
+def run_ckpt(config, rule, seed, out, cmd, f, theta=None, dose=None, stop_after=None, ckdir=CKDIR):
     N, ps, variant, wide, changing = R.CONFIGS[config]
     local = R.RULES[rule]
-    dose = dose or R.DOSE
+    dose = dose or DOSE
+    fp = round(f * 100)
+    assert fp in FRACTIONS and f == fp / 100
     assert dose % R.ROW == 0 and CK % R.ROW == 0
     if changing:
         assert theta is not None, "a configuration with changing connections needs --theta (set once by check 10)"
-    rp = R.receipt_path(out, config, rule, seed, dose)
+    rp = receipt_path(out, config, rule, fp, seed, dose)
     paths = dict(receipt=rp, weights=rp.with_suffix(".pt"), structure=rp.with_name(rp.stem + "_structure.json"),
                  small=rp.with_name(rp.stem + "_recurrent.pt"), answers=rp.with_name(rp.stem + "_answers.json"))
     ckp = Path(ckdir) / (rp.stem + ".ckpt.pt")
@@ -60,23 +71,32 @@
     opt = R.make_opt(m, layer)
     wc = R.weight_count(m, layer)
     assert wc == R.WEIGHT_COUNT[config], (config, wc)
-    A, Rs, TA, LP = R.panels(seed)
-    tp = [(t, R.E.split(t, .5, seed)[0]) for t in R.E.DEV]
-    bt = R.E.batches(tp, seed, R.E.KMAX["M"]); dg = R.C26.Digest()
+    A, Rs, TA = R.panels(seed, f)[:3]                     # E41's panels at this fraction
+    LP = R.E.panel(R.LEN_TASK, R.E.split(R.LEN_TASK, f, seed)[1], seed, ks=(12, 40), n=300)   # from the pairs held out at f
+    tk = taught_keys(f, seed)
+    for pan in list(A.values()) + list(Rs.values()) + list(LP.values()):
+        for p, _ in pan:
+            assert item_key(p) not in tk, f"a taught pair is in a panel of untaught pairs: {item_key(p)}"
+    for pan in TA.values():
+        assert_taught(pan, tk)
+    assert LP[12] == A[R.E.NAME(*R.LEN_TASK)], "the length panel at 12 letters is not the addition panel behind A"
+    bt = R.E.batches([(t, R.E.split(t, f, seed)[0]) for t in R.E.DEV], seed, R.E.KMAX["M"]); dg = R.C26.Digest()
     rr = Random(f"e42|replace|{seed}")
     struct0 = R.structure(layer)
-    me = dict(wrapper="exp42_ckpt.py", wrapper_sha256=R.X16.sha_file(HERE / "exp42_ckpt.py"), every=CK, resumed=[])
+    me = dict(wrapper="exp54_fraction.py", wrapper_sha256=R.X16.sha_file(HERE / "exp54_fraction.py"), every=CK, resumed=[])
+    prov = R.provenance(cmd)
+    prov["sources_sha256"].update({s: R.X16.sha_file(HERE / s) for s in ("exp54_fraction.py", "exp41_scale.py")})
     if not resume:
-        rec = dict(experiment="E42", kind="birth", config=config, rule=rule, local=local, seed=seed, dose=dose, row_every=R.ROW,
+        rec = dict(experiment="E54", kind="birth", fraction=f, length_panel_is_addition_panel=True, config=config, rule=rule, local=local, seed=seed, dose=dose, row_every=R.ROW,
                    N=N, period_set=ps, variant=variant, wide_neighbourhood=wide, changing=changing, theta=theta if changing else None,
                    batch=R.X.BATCH, kmax=R.E.KMAX["M"], weight_count=wc, params_module=sum(q.numel() for q in m.parameters()),
                    added_parameters={k: (None if v is None else list(v.shape)) for k, v in (("a", layer.a), ("Wx", layer.Wx))},
                    optimizer=R.opt_record(m, layer, opt),
                    update="exp26_candidate.step's text with exp42_cells.forward(m, layer, x, local) in place of arch26.forward",
-                   data="exp11_cell.batches([(t, split(t, .5, seed)[0]) for t in DEV], seed, 12)",
+                   data=f"exp11_cell.batches([(t, split(t, {f}, seed)[0]) for t in DEV], seed, 12), E41's stream (exp41_scale.stream) for this fraction",
                    forward="exp42_cells.forward (the faster forward; check 9 ties it to e42_cells.forward)",
                    structure_start_sha256=dict(mask=R.tsha(layer.mask), made=R.tsha(layer.made)),
-                   provenance=R.provenance(cmd), complete=False, curve=[], checkpointing=me)
+                   provenance=prov, complete=False, curve=[], checkpointing=me)
         cpu_up = cpu_sc = cpu_rep = 0.0; wall0 = 0.0; first = None; u0 = 0
     else:
         start_mask, start_made = layer.mask.clone(), layer.made.clone()      # the freshly built layer, before restoring
@@ -93,15 +113,17 @@
         rr.setstate(st["rr"]); T.set_rng_state(st["torch_rng"]); random.setstate(st["py_rng"])
         rec = st["rec"]; u0 = st["u"]; first = st["first"]
         cpu_up, cpu_sc, cpu_rep, wall0 = st["cpu_up"], st["cpu_sc"], st["cpu_rep"], st["wall"]
-        assert (rec["config"], rec["rule"], rec["seed"], rec["dose"]) == (config, rule, seed, dose)
+        assert (rec["config"], rec["rule"], rec["seed"], rec["dose"], rec["fraction"]) == (config, rule, seed, dose, f)
         assert rec["structure_start_sha256"] == dict(mask=R.tsha(start_mask), made=start_made), "checkpoint is not from this run's start"
         c = time.process_time()
         for _ in range(u0):                                   # replay the data stream to the checkpoint
-            dg.add(next(bt))
+            items = next(bt)
+            assert_taught(items, tk)
+            dg.add(items)
         assert dg.value() == st["digest"], "replayed data digest differs from the checkpoint's"
         rec["checkpointing"]["resumed"].append(dict(at_update=u0, cpu_replay=time.process_time() - c, when=time.strftime("%FT%TZ", time.gmtime()),
                                                      cmd=cmd))
-        print(f"e42 resume {config} {rule} {seed} at {u0}", flush=True)
+        print(f"e54 resume {config} {rule} f{fp} {seed} at {u0}", flush=True)
     curve = rec["curve"]; w0 = time.time() - wall0; losses = []; n_rep_since = 0
 
     def score_row(u):
@@ -109,6 +131,7 @@
         c = time.process_time()
         row = dict(step=u, **R.score_all(m, A, Rs, TA, LP))
         cpu_sc += time.process_time() - c
+        assert row["length_counts"]["12"] == row["A_counts"]["add89"], f"check 5 failed at row {u}: {row['length_counts']} {row['A_counts']}"
         lim = R.check_limit(m, layer)
         assert all(lim.values()), f"check 4 failed at row {u}: {lim}"
         row.update(replaced_since_previous_row=n_rep_since, limit=lim, cpu_updates=cpu_up, cpu_scoring=cpu_sc, cpu_replacement=cpu_rep,
@@ -117,7 +140,7 @@
                    wall=time.time() - w0)
         n_rep_since = 0
         curve.append(row)
-        print(f"e42 {config} {rule} {seed} {u}: A {row['A']:.2f} R {row['R']:.2f} taught {row['taught']:.2f} len12 {row['len12']:.1f} "
+        print(f"e54 {config} {rule} f{fp} {seed} {u}: A {row['A']:.2f} R {row['R']:.2f} taught {row['taught']:.2f} len12 {row['len12']:.1f} "
               f"len40 {row['len40']:.1f} replaced {row['replaced_since_previous_row']} cpu {cpu_up:.0f}+{cpu_sc:.0f}s", flush=True)
 
     def checkpoint(u):
@@ -133,6 +156,7 @@
         if first is None:
             first = items
             rec["cosine_first_batch_start"] = R.cosines(m, layer, first)
+        assert_taught(items, tk)
         dg.add(items)
         c = time.process_time(); losses.append(R.step(m, layer, opt, items, local)); cpu_up += time.process_time() - c
         if u % R.ROW == 0:
@@ -142,12 +166,14 @@
                 assert R.check_opt_outside(layer, m, opt), f"optimizer averages not zero outside the mask after the event at {u}"
                 cpu_rep += time.process_time() - c
             score_row(u); losses = []
+            if u == 12288:
+                rec.setdefault("tensors_sha256_at", {})["12288"] = R.R_.tensors_sha(m.state_dict())
             if u % 4096 == 0 or u == dose:
                 rp.write_text(json.dumps(rec, indent=1))
             if u % CK == 0 and u < dose:
                 checkpoint(u)
                 if stop_after is not None and u >= stop_after:
-                    print(f"e42 stopped after checkpoint at {u} (test)", flush=True)
+                    print(f"e54 stopped after checkpoint at {u} (test)", flush=True)
                     return None
     rec["cosine_first_batch_end"] = R.cosines(m, layer, first)
     sd = m.state_dict()
@@ -168,65 +194,28 @@
                added_parameters_sha256={k: R.tsha(v) for k, v in (("a", layer.a), ("Wx", layer.Wx)) if v is not None},
                optimizer_steps=R.R_.opt_steps(opt), optimizer_read_back=R.opt_record(m, layer, opt), data_digest=dg.value(),
                cpu_updates=cpu_up, cpu_scoring=cpu_sc, cpu_replacement=cpu_rep, wall=time.time() - w0,
-               processes_beside=os.environ.get("E42_BESIDE", "not recorded"))
+               processes_beside=os.environ.get("E54_BESIDE", "not recorded"))
     rp.write_text(json.dumps(rec, indent=1))
     if ckp.exists():
         ckp.unlink()
-    print("e42 complete", rp.name, rec["tensors_sha256"][:16], flush=True)
+    print("e54 complete", rp.name, rec["tensors_sha256"][:16], flush=True)
     return rec
 
 
-Q = HERE / "exp45" / "e42_qual"
-CASES = [("G", "S", 73129, None, 2048, Q / "check1_73129_runs" / "e42_G_S_dry2048_73129.json"),
-         ("G", "B", 73129, None, 2048, Q / "check1_73129_runs" / "e42_G_B_dry2048_73129.json"),
-         ("P8", "S", 73129, 0.5, 2048, Q / "check8_73129_runs" / "e42_P8_S_dry2048_73129.json"),
-         ("P8-source", "S", 73129, 0.5, 2048, Q / "check8_73129_runs" / "e42_P8-source_S_dry2048_73129.json")]
-VOLATILE = ("wall", "cpu_updates", "cpu_scoring", "cpu_replacement")
-
-
-def selftest(i=None):
-    """Each case: a dry run of 2,048 updates stopped after its checkpoint at 1,024 and resumed in a new process-equivalent call.
-    Required equal to the ORIGINAL runner's receipt of the same dry run (exp42_cells.run, stage-0 checks 1 and 8): final tensors_sha256,
-    the data digest, and every scored row of the curve except CPU and wall times. selftest <i> runs case i; no index collects all."""
-    base = Q / "ckpt_selftest"
-    if i is None:
-        res = {}
-        for j in range(len(CASES)):
-            res.update(json.loads((base / f"case{j}.json").read_text()))
-        ok = all(v["tensors_equal"] and v["curve_equal"] and v["digest_equal"] and v["resumed_at"] == [1024] for v in res.values())
-        out = dict(selftest=res, pass_=ok, wrapper_sha256=R.X16.sha_file(HERE / "exp42_ckpt.py"), runner_sha256=R.X16.sha_file(HERE / "exp42_cells.py"))
-        (Q / "ckpt_selftest.json").write_text(json.dumps(out, indent=1))
-        print("ckpt selftest:", "PASS" if ok else "FAIL", flush=True)
-        return ok
-    cfg, rule, seed, th, dose, orig_p = CASES[i]
-    orig = json.loads(orig_p.read_text())
-    key = f"{cfg} {rule} {seed} dry{dose}"
-    cb = base / f"case{i}"
-    rb = json.loads(R.receipt_path(cb, cfg, rule, seed, dose).read_text())     # written by the two separate processes of selftest_case.sh
-    assert rb["complete"] is True
-    strip = lambda c: [{k: v for k, v in r.items() if k not in VOLATILE} for r in c]
-    res = {key: dict(original_receipt=str(orig_p.relative_to(HERE)), original_tensors=orig["tensors_sha256"], resumed_tensors=rb["tensors_sha256"],
-                     tensors_equal=rb["tensors_sha256"] == orig["tensors_sha256"], curve_equal=strip(rb["curve"]) == strip(orig["curve"]),
-                     digest_equal=rb["data_digest"] == orig["data_digest"], rows=len(rb["curve"]),
-                     resumed_at=[x["at_update"] for x in rb["checkpointing"]["resumed"]])}
-    print(key, res[key], flush=True)
-    (base / f"case{i}.json").write_text(json.dumps(res))
-    return True
-
-
 def main(argv):
     cmd = "python3 " + " ".join(argv)
     a = argv[1:]
 
     def opt(k, d=None):
         return a[a.index(k) + 1] if k in a else d
-    if a[0] == "selftest":
-        sys.exit(0 if selftest(int(a[1]) if len(a) > 1 else None) else 1)
     assert a[0] == "run"
     config, rule, seed = a[1], a[2], int(a[3])
     assert config in R.CONFIGS and rule in R.RULES and seed in R.SEEDS
     th, d, s = opt("--theta"), opt("--dose"), opt("--stop-after")
-    run_ckpt(config, rule, seed, Path(opt("--out", str(R.OUT))), cmd, theta=float(th) if th else None,
+    assert "--fraction" in a, "--fraction <50|25|15|10> is required"
+    fp = int(opt("--fraction"))
+    assert fp in FRACTIONS
+    run_ckpt(config, rule, seed, Path(opt("--out", str(OUT))), cmd, fp / 100, theta=float(th) if th else None,
              dose=int(d) if d else None, stop_after=int(s) if s else None, ckdir=Path(opt("--ckdir", str(CKDIR))))
 
 
```

## Appendix B. Committed files

The files of E54, from `git ls-tree -r -l e718483e -- exp54_fraction.py e54_checks.py e54_tables.py e54_driver.py exp45/e54` run in `MOD97_LOCAL_RULE_20261106/` (size in bytes, path); this report, `E54_REPORT.md`, was committed after that commit. The files of saved state (`exp45/e54/**/ckpt/`) are not committed, as the brief requires.

```
11871 e54_checks.py
9342 e54_driver.py
15066 e54_tables.py
37998 exp45/e54/E54_TABLES.json
10746 exp45/e54/E54_TABLES.md
310 exp45/e54/driver_state.json
325740 exp45/e54/e41_B_h256_f10_dry122880_73129.json
1275306 exp45/e54/e41_B_h256_f10_dry122880_73129.pt
324378 exp45/e54/e41_S_h256_f10_dry122880_73129.json
1275306 exp45/e54/e41_S_h256_f10_dry122880_73129.pt
183905 exp45/e54/e54_F10_B_f10_73129.json
5096585 exp45/e54/e54_F10_B_f10_73129.pt
1383371 exp45/e54/e54_F10_B_f10_73129_recurrent.pt
44046 exp45/e54/e54_F10_B_f10_73129_structure.json
185074 exp45/e54/e54_F10_B_f15_73129.json
5096585 exp45/e54/e54_F10_B_f15_73129.pt
1383371 exp45/e54/e54_F10_B_f15_73129_recurrent.pt
44046 exp45/e54/e54_F10_B_f15_73129_structure.json
185410 exp45/e54/e54_F10_B_f25_73129.json
5096585 exp45/e54/e54_F10_B_f25_73129.pt
1383371 exp45/e54/e54_F10_B_f25_73129_recurrent.pt
44046 exp45/e54/e54_F10_B_f25_73129_structure.json
183574 exp45/e54/e54_F10_S_f10_73129.json
5096585 exp45/e54/e54_F10_S_f10_73129.pt
1383371 exp45/e54/e54_F10_S_f10_73129_recurrent.pt
44046 exp45/e54/e54_F10_S_f10_73129_structure.json
184496 exp45/e54/e54_F10_S_f15_73129.json
5096585 exp45/e54/e54_F10_S_f15_73129.pt
1383371 exp45/e54/e54_F10_S_f15_73129_recurrent.pt
44046 exp45/e54/e54_F10_S_f15_73129_structure.json
184882 exp45/e54/e54_F10_S_f25_73129.json
5096585 exp45/e54/e54_F10_S_f25_73129.pt
1383371 exp45/e54/e54_F10_S_f25_73129_recurrent.pt
44046 exp45/e54/e54_F10_S_f25_73129_structure.json
185244 exp45/e54/e54_G_B_f10_73129.json
5096475 exp45/e54/e54_G_B_f10_73129.pt
1383349 exp45/e54/e54_G_B_f10_73129_recurrent.pt
993684 exp45/e54/e54_G_B_f10_73129_structure.json
185416 exp45/e54/e54_G_B_f15_73129.json
5096475 exp45/e54/e54_G_B_f15_73129.pt
1383349 exp45/e54/e54_G_B_f15_73129_recurrent.pt
993684 exp45/e54/e54_G_B_f15_73129_structure.json
185700 exp45/e54/e54_G_B_f25_73129.json
5096475 exp45/e54/e54_G_B_f25_73129.pt
1383349 exp45/e54/e54_G_B_f25_73129_recurrent.pt
993684 exp45/e54/e54_G_B_f25_73129_structure.json
184504 exp45/e54/e54_G_S_f10_73129.json
5096475 exp45/e54/e54_G_S_f10_73129.pt
1383349 exp45/e54/e54_G_S_f10_73129_recurrent.pt
993684 exp45/e54/e54_G_S_f10_73129_structure.json
186045 exp45/e54/e54_G_S_f15_73129.json
5096475 exp45/e54/e54_G_S_f15_73129.pt
1383349 exp45/e54/e54_G_S_f15_73129_recurrent.pt
993684 exp45/e54/e54_G_S_f15_73129_structure.json
185133 exp45/e54/e54_G_S_f25_73129.json
5096475 exp45/e54/e54_G_S_f25_73129.pt
1383349 exp45/e54/e54_G_S_f25_73129_recurrent.pt
993684 exp45/e54/e54_G_S_f25_73129_structure.json
187391 exp45/e54/e54_P8_B_f10_73129.json
5104957 exp45/e54/e54_P8_B_f10_73129.pt
1391825 exp45/e54/e54_P8_B_f10_73129_recurrent.pt
50081 exp45/e54/e54_P8_B_f10_73129_structure.json
188091 exp45/e54/e54_P8_B_f15_73129.json
5104957 exp45/e54/e54_P8_B_f15_73129.pt
1391825 exp45/e54/e54_P8_B_f15_73129_recurrent.pt
49889 exp45/e54/e54_P8_B_f15_73129_structure.json
188471 exp45/e54/e54_P8_B_f25_73129.json
5104957 exp45/e54/e54_P8_B_f25_73129.pt
1391825 exp45/e54/e54_P8_B_f25_73129_recurrent.pt
50214 exp45/e54/e54_P8_B_f25_73129_structure.json
187521 exp45/e54/e54_P8_S_f10_73129.json
5104957 exp45/e54/e54_P8_S_f10_73129.pt
1391825 exp45/e54/e54_P8_S_f10_73129_recurrent.pt
49792 exp45/e54/e54_P8_S_f10_73129_structure.json
187641 exp45/e54/e54_P8_S_f15_73129.json
5104957 exp45/e54/e54_P8_S_f15_73129.pt
1391825 exp45/e54/e54_P8_S_f15_73129_recurrent.pt
49862 exp45/e54/e54_P8_S_f15_73129_structure.json
188301 exp45/e54/e54_P8_S_f25_73129.json
5104957 exp45/e54/e54_P8_S_f25_73129.pt
1391825 exp45/e54/e54_P8_S_f25_73129_recurrent.pt
49969 exp45/e54/e54_P8_S_f25_73129_structure.json
16219 exp45/e54/exp54_fraction_vs_exp42_ckpt.diff
1370 exp45/e54/logs/check1_e42.log
1434 exp45/e54/logs/check1_new.log
1722 exp45/e54/logs/check3_pre.log
988 exp45/e54/logs/check6_resumed.log
910 exp45/e54/logs/check6_whole.log
9650 exp45/e54/logs/driver.log
9432 exp45/e54/logs/driver.out
19770 exp45/e54/logs/e54_F10_B_f10.log
19864 exp45/e54/logs/e54_F10_B_f15.log
19811 exp45/e54/logs/e54_F10_B_f25.log
19425 exp45/e54/logs/e54_F10_S_f10.log
20079 exp45/e54/logs/e54_F10_S_f15.log
19978 exp45/e54/logs/e54_F10_S_f25.log
19433 exp45/e54/logs/e54_G_B_f10.log
19742 exp45/e54/logs/e54_G_B_f15.log
19682 exp45/e54/logs/e54_G_B_f25.log
19196 exp45/e54/logs/e54_G_S_f10.log
19643 exp45/e54/logs/e54_G_S_f15.log
19692 exp45/e54/logs/e54_G_S_f25.log
20215 exp45/e54/logs/e54_P8_B_f10.log
20194 exp45/e54/logs/e54_P8_B_f15.log
20135 exp45/e54/logs/e54_P8_B_f25.log
20146 exp45/e54/logs/e54_P8_S_f10.log
20198 exp45/e54/logs/e54_P8_S_f15.log
20143 exp45/e54/logs/e54_P8_S_f25.log
34064 exp45/e54/logs/part4_B.log
33817 exp45/e54/logs/part4_S.log
2038 exp45/e54/qual/check1.json
2665 exp45/e54/qual/check1_73129.json
13201 exp45/e54/qual/check1_73129_runs/e42_G_B_dry2048_73129.json
993683 exp45/e54/qual/check1_73129_runs/e42_G_B_dry2048_73129_structure.json
13168 exp45/e54/qual/check1_73129_runs/e42_G_S_dry2048_73129.json
993683 exp45/e54/qual/check1_73129_runs/e42_G_S_dry2048_73129_structure.json
13099 exp45/e54/qual/check1_e42/e42_F10_S_dry2048_73129.json
44045 exp45/e54/qual/check1_e42/e42_F10_S_dry2048_73129_structure.json
9700 exp45/e54/qual/check1_e42/e42_P8_S_dry1024_73129.json
44783 exp45/e54/qual/check1_e42/e42_P8_S_dry1024_73129_structure.json
13617 exp45/e54/qual/check1_new/e54_F10_S_f50_dry2048_73129.json
44045 exp45/e54/qual/check1_new/e54_F10_S_f50_dry2048_73129_structure.json
10223 exp45/e54/qual/check1_new/e54_P8_S_f50_dry1024_73129.json
44783 exp45/e54/qual/check1_new/e54_P8_S_f50_dry1024_73129_structure.json
294 exp45/e54/qual/check2_f10.json
295 exp45/e54/qual/check2_f15.json
295 exp45/e54/qual/check2_f25.json
595 exp45/e54/qual/check3_G_B_f10.json
357 exp45/e54/qual/check3_G_B_f10_row256.json
595 exp45/e54/qual/check3_G_B_f25.json
357 exp45/e54/qual/check3_G_B_f25_row256.json
595 exp45/e54/qual/check3_G_S_f10.json
357 exp45/e54/qual/check3_G_S_f10_row256.json
595 exp45/e54/qual/check3_G_S_f25.json
357 exp45/e54/qual/check3_G_S_f25_row256.json
398 exp45/e54/qual/check3_part4_B.json
327 exp45/e54/qual/check3_part4_B_row256.json
398 exp45/e54/qual/check3_part4_S.json
327 exp45/e54/qual/check3_part4_S_row256.json
1770 exp45/e54/qual/check4.json
624 exp45/e54/qual/check6.json
14332 exp45/e54/qual/check6_resumed/e54_P8_S_f25_dry2048_73129.json
45211 exp45/e54/qual/check6_resumed/e54_P8_S_f25_dry2048_73129_structure.json
14030 exp45/e54/qual/check6_whole/e54_P8_S_f25_dry2048_73129.json
45211 exp45/e54/qual/check6_whole/e54_P8_S_f25_dry2048_73129_structure.json
13171 exp54_fraction.py
```
