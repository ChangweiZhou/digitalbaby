# E62 report: the number of connections of a cell, from 2 to 255, in the network of E50's stage 3 (seed 73129)

**Status.** This report records a first measurement on one seed; it confirms nothing, and its result is not confirmed by any other seed. It was written by the worker of E62 from the receipts listed below, and nobody has reviewed it yet. It has passed no automatic check of the programme's checker (which is not installed as runnable software on this branch) and carries no seal or adoption by the owner. The brief is `PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/COORDINATION_20261117/TASK_E62_THE_NUMBER_OF_CONNECTIONS_OF_A_CELL_20261118.md`, revision 2, on the reviewer's branch `claude/vibrant-cerf-sje726` (unchanged since 4a30269f). The coordinator released it in decision 41 of `COORDINATOR_DECISIONS_20261118.md` on the default branch at 5a27108e ("Released as written, at revision 2, at Mr. Xu's instruction. On E54's line"). The work is on branch `claude/e62-connections-ELsQP`, which was started from the default branch `codex/minifly-consolidated-20261023` at e647e7d7, as the brief says; nothing was merged into it, and no file of E54 was changed. Because the full default branch does not fit on this host's disk, the branch is checked out sparsely (the Python sources of `MOD97_LOCAL_RULE_20261106/`, E62's own folders, the specification file of E50 and the two files of the root that the rules name); the commits contain the whole tree of e647e7d7. Every path below is relative to `MOD97_LOCAL_RULE_20261106/` unless it is stated otherwise.

## 1. The question, and how the two rules are read

Mr. Xu asked on 18 November that a number that we fix by hand, the number of other cells that a cell reads, be varied on a grid of powers of two and the effect be read, before a later step asks whether a cell can set the number itself. The brief therefore asks: how does learning depend on the number of cells that a cell reads?

*The network* is that of E50's stage 3 (the setting `C46-one` of `exp50_code.py`), in which only the number and the choice of a cell's connections are changed. It has 256 recurrent cells of the gated kind of E26. Its input is a fixed code of 512 values in which each of the 46 used bytes turns on 11 values of its own, and each cell reads 46 of these values, one of every used byte; the readout reads every cell. The byte table and the first input layer `l1` take no part. *N* is the number of other cells that a cell reads; every cell also reads itself. The connections are fixed for the whole run. *From all cells* means that a cell's N connections are drawn at random from the 255 other cells; *from the neighbourhood* means that they are drawn from the 48 cells around it on a grid of 16 by 16 cells (a square of 7 by 7 cells centred on the cell, with the grid's edges joined). With N = 255 every cell reads every other, and the choice has no effect. At the start the kept recurrent weights are multiplied by the square root of 256 / (N + 1), so that the variance of a cell's recurrent input does not depend on N; nothing is scaled at N = 255.

*The record* is the two runs of E50's stage 3 with N = 10 from the neighbourhood (`exp45/e50/e50_C46-one_<S|B>_73129.json` on the default branch), which were not repeated. *The full run* of a rule is its run with N = 255. It is the same network with every cell reading every other and the same limited input code. It is not the dense network of decision 31, and it is never called the reference. *The reference* of decision 31 is back-propagation over the whole item on the dense network with a learned code: for seed 73129, E42's configuration `G` under the exact gradient, which reached 80 on R after 6,400 updates and on A after 5,120 (`exp45/e42/e42_G_B_73129.json` on the default branch).

Two rules pass the error backward through each linear map. *The sweep* (rule S) passes it by the signs of the weights times one scale for the map; *the exact gradient* (rule B) passes it through the weights themselves. Coordinator's decision 38 revises decision 31, which the brief follows: on a network with limited connections, the exact gradient is a *reference arm*. It shows what the same network would give if the signal sent back were exact. It is never called a candidate or "the local learner". The sweep is the working rule. It has three parts that a neuron does not have: it uses the sign of the forward connection, it applies one scale to a whole matrix, and it passes the error back over the whole item after the item has ended. In every run of this job the error is passed back over the whole item, as in the record.

**Scores.** The network is scored every 256 updates; one such set of scores is a *scored row*. *A* is the percentage of exactly correct answers on 600 never-taught items of addition and subtraction modulo 89 with 12 uninformative letters between the operands. *R* is the percentage on 2,000 never-taught items whose answer needs the reduction by the modulus, written with no letters. *T* is the percentage on 600 taught items with 12 letters. These are E50's definitions and E50's scorer. A threshold is *reached* at the first of three consecutive scored rows at or above it; the first single row at or above it is given in brackets. A *level* at an update is the mean of the three scored rows that end there. An N is *as fast as the full run* if its run reaches 80 on R in at most 1.25 times the full run's updates; *slower* if it reaches 80 on R later than that within 49,152 updates; otherwise *not reached*. The factor 1.25 is the reviewer's choice.

**Reached within k steps.** A step goes from a cell to the cells that read it. The cells *reached within k steps* from a cell are those whose state at position t + k can depend on that cell's state at position t, the cell itself included. A *strongly connected part* is a largest set of cells in which every cell is reached from every other.

## 2. What was run

Eighteen runs of 49,152 updates on seed 73129: N = 2, 4, 8, 16, 32, 64, 128 and 255 from all cells, and N = 10 from all cells, each under the sweep and under the exact gradient. They ran in the brief's order: N = 8, 32 and 255 first, then N = 2, 128 and 10, then N = 4, 16 and 64, with the sweep before the exact gradient at each N. Everything other than the connections is the record's: the input and its start, the stream of batches of 64 drawn from half of the pairs, AdamW at rate 0.001 with betas 0.9 and 0.98 and weight decay 0.25, the update, the panels and the scorer. The host is Linux on x86-64 with four cores, Python 3.13.18 and torch 2.15.0+cpu with numpy 2.5.5 in a virtual environment (the versions are in every receipt's `versions`). At most three processes ran at a time, each with one thread.

The runs were started by `e62_driver.py`. This launcher trains nothing itself. It starts each run's own command, `python3 exp62_connections.py run <N> all <S|B> 73129 --resume`, as a separate process, with the variable `E62_BESIDE` set to the number of other runs in progress, with the harness's own variables removed from the child's environment (decision 37), and with the output appended to the run's log `exp45/e62/logs/e62_N<N>_all_<rule>_73129.log`. The launcher's own log is `exp45/e62/logs/driver.log`.

**Interruptions.** The host stopped every process twice.

- The first stop came between 00:28:37 UTC on 19 November (the launcher's last line before it) and 00:34. The run with N = 255 under the sweep resumed from its saved state at update 45,056. The runs with N = 2 under both rules had not yet reached their first saved state (update 4,096). Their files were moved to `exp45/e62/interrupted/20261119T003506Z/`, and they were started again from the beginning. Their logs were not moved: each still holds, above the restarted run's lines, the lines of the first attempt up to update 3,072.
- The second stop came between 07:12 and 07:16 UTC. Three runs resumed from their saved states: N = 16 under the exact gradient at update 16,384, N = 64 under the sweep at 8,192, and N = 64 under the exact gradient at 4,096.

The receipts record each resumption under `resumes`, with the CPU-seconds of the replay of the stream. Check 3 below shows, for one setting, that a killed and resumed run equals an uninterrupted one in every recorded quantity. No other run was interrupted.

**Cost.** The eighteen runs used 34.69 CPU-hours of updates and scoring: the sum of `cpu_updates` and `cpu_scoring` over the eighteen receipts, given per run in the second table below. The checks used about 0.25 CPU-hours: 0.08 recorded in the receipts' `cpu` field, plus 0.17 in the two child runs of check 3, whose receipts are under `exp45/e62_qual/check3_73129_runs/`. Check 5 forecast 34.68 CPU-hours for the runs, against the brief's limit of 44.

## 3. The runner

`exp62_connections.py` is new; no existing source file was edited. It imports `exp26_candidate` first, then `exp50_code` as `R50`, `e42_cells` as `L42`, `exp49_interface` as `R49` and `exp42_cells` as `R42`, and calls every function of `exp50_code` other than `build` and `run` unchanged through `R50`. It was committed at c2169666, before the first check ran.

- `build(N, choice, seed)` is the text of `exp50_code.build` for the setting `C46-one`. The one line `layer = L42.Layer(m, seed, N=10)` is replaced by `layer = L42.Layer(m, seed, N=N, wide=(choice == "all"))`, and the module's names are called through `R50`.
- `run` is the text of `exp50_code.run` with the brief's changes:
  - `experiment="E62"` and the entry `choice`;
  - `build` and a `receipt_path` of this file, and a `provenance` of this file;
  - the assertion that the weights taking part number 560,128 + 768 (N + 1); the assertion on the products turned on is dropped, and the count is recorded;
  - the variable `E62_BESIDE`;
  - the figures of point 3 of the brief, recorded once in the receipt's entry `graph`;
  - saving and resuming as in `exp52_resume.py`, with the assertions that the brief's point 4 lists.

Every assertion of `exp50_code.run` at a scored row is kept. The complete difference between the two texts of `build` and `run` is in Appendix A and in `exp45/e62/exp62_build_run_vs_exp50_code.diff`.

**Departures from the brief's list.** There are four, none of which changes what a run computes:

1. The record's optional `--verify-scorer` branch, which compares the scorer with E42's at every row and was not used in the record's runs, is not in the copy.
2. The receipt records three entries that the brief does not name: the digest of the lists of connections (`draw_digests.connections`), the scale of the recurrent start (`start_scale.recurrent`) and the interval of the saved states (`checkpoint_every`).
3. The partial receipt is also written at every update at which a state is saved, so that a state is always saved after the receipt that it holds.
4. On resuming, the hashes of the rebuilt network and of the starting file are compared before the stream is drawn again, not after.

Two further scripts are new and train nothing. `e62_tables.py` writes every table and reading of this report from the receipts, reading the record and the reference with `git show`. `e62_driver.py` is the launcher described above.

## 4. The checks before the runs

Each check wrote `exp45/e62_qual/check<n>_73129.json`, and each passed at the first attempt. The logs are in `exp45/e62_qual/logs/`.

1. **With N = 10 from the neighbourhood the run is the record.** Two dry runs of 512 updates, one under each rule, have the record's `start_tensors_sha256`. Their scored rows at updates 0, 256 and 512 equal the record's in the loss, in every count (A, R, taught, the length test, and the two answer bytes of R) and in the batch digest.
2. **The networks are what the brief says.** For each of the ten settings, at the start:
   - every cell reads exactly N other cells and itself;
   - the digest of the lists of connections equals the brief's (9622342ea3d0e56f to fde78220c57da3d4), and so does the number of connections outside the reading cell's neighbourhood (431 to 52,992, and 0 for the record's network);
   - every weight outside the mask is zero, and the kept weights are bit-equal to E26's start times the brief's scale;
   - the input map, `lm`, `l2`, `l3` and the code table are bit-equal to those of `R50.build("C46-one", 73129)`;
   - the first-batch loss is finite under both rules, and the gradients of the recurrent map are zero outside the mask;
   - every figure of the runner's point 3 equals the table of section 2 of `CHECK_OF_TASK_E62_BRIEF_20261118.md`.
3. **A run that is killed and resumed equals an uninterrupted one.** N = 32 from all cells under the sweep, 2,048 updates, was run twice. Once it ran without saved states. Once it saved a state every 1,024 updates, ended with exit code 9 after update 1,700, and was started again with `--resume`. The two finished receipts have the same `tensors_sha256` (e5092ecc…), the same nine rows in every field whose name does not begin with `cpu` or `wall`, and equal `first_batch_loss_start`, `cosine_first_batch_start`, `cosine_first_batch_end`, `data_digest` and `optimizer_steps`; the second has one resumption, at update 1,024. The children were started with the environment without the variables beginning `E11_`, `E16_` and `EXP45_`, from this folder.
4. **Inside every run**, the assertions of `exp50_code.run` at every scored row held: the masks, the fixed code, the byte table and `l1` stayed bit-equal to their start. Otherwise the run would have stopped.
5. **Cost.** The measured CPU-seconds an update were:

   | N | Sweep | Exact gradient |
   |---|---|---|
   | 2 | 0.124 | 0.115 |
   | 32 | 0.124 | 0.098 |
   | 255 | 0.108 | 0.094 |

   A scored row took 5.4 to 6.4 CPU-seconds. The forecast for the eighteen runs was 34.7 CPU-hours, below the limit of 44.

## 5. Results

The tables of this section are written by `python3 e62_tables.py` (`exp45/e62/E62_TABLES.md` and `exp45/e62/E62_TABLES.json`). CPU time does not depend on N in this implementation: the masked recurrent map is computed as a dense product, so that a run with few connections costs as much as one with many. The networks are therefore compared by updates and by the multiply-adds that the receipt counts (`multiply_adds` of a row: the weights taking part, times 64 items a batch, times the updates), and no saving of computation is read from CPU time.

### The table of the runs

| N | Connections from | Exact gradient: updates to 80 on R | Sweep: updates to 80 on R | Ratio sweep / exact gradient | Exact gradient: to 80 on A | Sweep: to 80 on A | Read by no other cell | Every cell from every cell within | Weights taking part |
|---|---|---|---|---|---|---|---|---|---|
| 2 | all cells | 15,616 [13,824] | 30,976 [26,368] | 1.98 | 16,384 [15,616] | 33,280 [26,880] | 40 | none | 562,432 |
| 4 | all cells | 17,408 [16,640] | not reached | — | 18,688 [17,664] | not reached | 4 | none | 563,968 |
| 8 | all cells | 18,176 [16,640] | not reached | — | 18,176 [15,616] | not reached | 0 | 5 | 567,040 |
| 10 | the neighbourhood (the record) | 15,360 [13,824] | 34,304 [34,304] | 2.23 | 14,080 [12,288] | 35,584 [33,536] | 0 (record) | 5 (record) | 568,576 |
| 10 | all cells | 16,128 [16,128] | not reached | — | 16,128 [14,848] | 47,616 [43,776] | 0 | 4 | 568,576 |
| 16 | all cells | 13,824 [12,288] | 42,752 [39,936] | 3.09 | 13,056 [10,752] | 40,704 [37,120] | 0 | 4 | 573,184 |
| 32 | all cells | 11,520 [11,520] | 38,656 [38,656] | 3.36 | 11,776 [10,752] | 38,656 [29,696] | 0 | 3 | 585,472 |
| 64 | all cells | 7,680 [7,168] | 32,256 [20,736] | 4.20 | 7,680 [7,168] | 28,672 [18,176] | 0 | 2 | 610,048 |
| 128 | all cells | 8,704 [8,704] | 12,288 [10,240] | 1.41 | 8,704 [7,936] | 9,216 [8,704] | 0 | 2 | 659,200 |
| 255 | all cells | 8,704 [8,704] | 7,168 [7,168] | 0.82 | 8,704 [8,704] | 6,144 [6,144] | 0 | 1 | 756,736 |

### Levels, cost to 80 on R, and resumptions

| N | From | Rule | A, R, T at 24,576 | A, R, T at 49,152 | Multiply-adds to 80 on R | CPU-seconds of updates to 80 on R | CPU-hours of the run | Resumed at |
|---|---|---|---|---|---|---|---|---|
| 2 | all | B | 86.94, 88.50, 96.00 | 93.33, 93.07, 98.28 | 562,108,039,168 | 1697 | 1.84 | never |
| 2 | all | S | 75.39, 77.53, 89.33 | 85.11, 85.97, 94.89 | 1,115,001,192,448 | 3956 | 2.10 | never |
| 4 | all | B | 85.28, 85.50, 94.83 | 94.22, 94.38, 98.72 | 628,323,516,416 | 1900 | 1.85 | never |
| 4 | all | S | 61.06, 57.82, 80.83 | 76.39, 76.70, 91.00 | — | — | 2.11 | never |
| 8 | all | B | 84.72, 84.77, 92.06 | 94.50, 94.65, 98.72 | 659,617,218,560 | 2020 | 1.86 | never |
| 8 | all | S | 63.44, 62.60, 84.50 | 78.33, 80.17, 94.11 | — | — | 2.11 | never |
| 10 | near | B | 88.50, 88.87, 94.89 | 93.83, 94.25, 98.33 | 558,932,951,040 | 1726 | 1.90 | never |
| 10 | near | S | 63.61, 65.50, 84.61 | 85.89, 87.22, 94.00 | 1,248,283,590,656 | 4561 | 2.18 | never |
| 10 | all | B | 85.61, 86.47, 93.11 | 94.78, 94.08, 98.89 | 586,879,598,592 | 1788 | 1.85 | never |
| 10 | all | S | 60.56, 53.05, 79.61 | 80.50, 79.82, 91.56 | — | — | 2.11 | never |
| 16 | all | B | 87.22, 88.83, 93.44 | 92.11, 91.92, 96.78 | 507,116,519,424 | 1509 | 1.78 | [16384] |
| 16 | all | S | 64.00, 57.95, 82.56 | 83.61, 84.48, 94.83 | 1,568,304,791,552 | 5503 | 2.11 | never |
| 32 | all | B | 88.17, 88.52, 94.06 | 93.28, 92.90, 96.72 | 431,656,796,160 | 1244 | 1.77 | never |
| 32 | all | S | 68.39, 67.58, 85.00 | 84.89, 85.88, 95.00 | 1,448,448,360,448 | 5051 | 2.11 | never |
| 64 | all | B | 91.39, 92.32, 94.83 | 93.72, 93.17, 96.61 | 299,850,792,960 | 789 | 1.76 | [4096] |
| 64 | all | S | 77.33, 76.57, 84.11 | 83.56, 81.75, 90.11 | 1,259,373,330,432 | 3978 | 2.01 | [8192] |
| 128 | all | B | 91.22, 94.75, 95.11 | 96.39, 96.73, 98.83 | 367,211,315,200 | 962 | 1.87 | never |
| 128 | all | S | 89.28, 86.85, 92.67 | 87.83, 88.10, 93.83 | 518,415,974,400 | 1573 | 2.12 | never |
| 255 | all | B | 95.06, 95.23, 97.39 | 96.78, 97.55, 97.22 | 421,544,329,216 | 806 | 1.53 | never |
| 255 | all | S | 90.67, 89.27, 94.00 | 89.39, 90.88, 94.33 | 347,154,153,472 | 807 | 1.79 | [45056] |

### Reached within k steps (least / mean / most), from the receipts

| N | From | 1 | 2 | 3 | 4 | 6 | 8 | Strongly connected parts (largest) |
|---|---|---|---|---|---|---|---|---|
| 2 | all | 1 / 3.0 / 7 | 1 / 6.9 / 20 | 1 / 14.5 / 50 | 1 / 28.6 / 98 | 1 / 88.6 / 224 | 1 / 165.0 / 256 | 49 (207) |
| 4 | all | 1 / 5.0 / 12 | 1 / 20.4 / 52 | 1 / 71.8 / 172 | 1 / 175.9 / 255 | 1 / 251.4 / 256 | 1 / 252.0 / 256 | 5 (252) |
| 8 | all | 3 / 9.0 / 16 | 14 / 64.9 / 123 | 78 / 223.7 / 256 | 241 / 255.9 / 256 | 256 / 256.0 / 256 | 256 / 256.0 / 256 | 1 (256) |
| 10 | all | 4 / 11.0 / 20 | 35 / 91.8 / 148 | 203 / 249.7 / 256 | 256 / 256.0 / 256 | 256 / 256.0 / 256 | 256 / 256.0 / 256 | 1 (256) |
| 16 | all | 6 / 17.0 / 28 | 75 / 171.0 / 220 | 254 / 256.0 / 256 | 256 / 256.0 / 256 | 256 / 256.0 / 256 | 256 / 256.0 / 256 | 1 (256) |
| 32 | all | 17 / 33.0 / 55 | 232 / 252.9 / 256 | 256 / 256.0 / 256 | 256 / 256.0 / 256 | 256 / 256.0 / 256 | 256 / 256.0 / 256 | 1 (256) |
| 64 | all | 49 / 65.0 / 89 | 256 / 256.0 / 256 | 256 / 256.0 / 256 | 256 / 256.0 / 256 | 256 / 256.0 / 256 | 256 / 256.0 / 256 | 1 (256) |
| 128 | all | 108 / 129.0 / 150 | 256 / 256.0 / 256 | 256 / 256.0 / 256 | 256 / 256.0 / 256 | 256 / 256.0 / 256 | 256 / 256.0 / 256 | 1 (256) |
| 255 | all | 256 / 256.0 / 256 | 256 / 256.0 / 256 | 256 / 256.0 / 256 | 256 / 256.0 / 256 | 256 / 256.0 / 256 | 256 / 256.0 / 256 | 1 (256) |


## 6. The readings in the words fixed by the brief

Every statement in this section is for seed 73129 and for the network of E50's stage 3 with connections drawn from all cells, unless it names the record.

**The exact gradient (the reference arm).** “On seed 73129, with connections drawn from all cells, the exact gradient reached 80 on R after u updates with N connections a cell, for N = 2, 4, 8, 16, 32, 64, 128, 255: 15,616, 17,408, 18,176, 13,824, 11,520, 7,680, 8,704, 8,704; the smallest N of the grid of powers of two that was as fast as the network in which every cell reads every other was 64, and every larger N of the grid was also as fast.” The full run reached 80 on R after 8,704 updates, so the limit for "as fast" is 10,880 updates. N = 64 (7,680 updates) and N = 128 (8,704) were as fast. N = 2, 4, 8, 16 and 32 were slower: 15,616, 17,408, 18,176, 13,824 and 11,520 updates. Every run under the exact gradient reached 80 on R. “With 10 connections from all cells the exact gradient reached 80 on R after 16,128 updates, against 15,360 (the exact gradient) with 10 connections from the neighbourhood.”

**The sweep.** “On seed 73129, with connections drawn from all cells, the sweep reached 80 on R after u updates with N connections a cell, for N = 2, 4, 8, 16, 32, 64, 128, 255: 30,976, not reached, not reached, 42,752, 38,656, 32,256, 12,288, 7,168; no N below 255 was as fast as the network in which every cell reads every other; the smallest N that reached 80 at all was 2.” The full run reached 80 on R after 7,168 updates, so the limit for "as fast" is 8,960 updates. N = 128 reached 80 after 12,288 updates and was slower; N = 2, 16, 32 and 64 were slower still, at 30,976, 42,752, 38,656 and 32,256 updates. Three runs did not reach 80 on R by the rule of three consecutive rows:

- N = 4: R was never at or above 80 in a single row (largest 79.20); its level at update 49,152 was 76.70.
- N = 8: a single row first reached 80 at update 41,984; the level at update 49,152 was 80.17.
- N = 10 from all cells: a single row first reached 80 at update 46,336; the level at update 49,152 was 79.82.

These figures are from the receipts `exp45/e62/e62_N<N>_all_S_73129.json`. Under the sweep, updates to 80 on R did not fall steadily with N: N = 2 reached 80 before N = 16, 32 and 64, and N = 4, 8 and 10 did not reach it. Each N is one draw of connections, and the draws for different N are not nested, so this order is not read as an effect of N. “With 10 connections from all cells the sweep did not reach 80 on R within 49,152 updates, against 34,304 (the sweep) with 10 connections from the neighbourhood.”

**The two rules.** “The ratio of the sweep's updates to the exact gradient's was 1.98 at N = 2; 3.09 at N = 16; 3.36 at N = 32; 4.20 at N = 64; 1.41 at N = 128; 0.82 at N = 255, beside the record's 2.23 (N = 10 from the neighbourhood).” The sweep needed more updates than the exact gradient at N = 2, 16, 32, 64 and 128 (ratios 1.41 to 4.20), and fewer at N = 255 (ratio 0.82: 7,168 against 8,704). At N = 4, 8 and 10 from all cells, only the exact gradient reached 80, so no ratio is given.

**N = 2 and N = 4.** These two networks are not one strongly connected part (receipts' entry `graph`).

- With N = 2, 40 cells are read by no other cell, and the network falls into 49 strongly connected parts, the largest of 207 cells.
- With N = 4, 4 cells are read by no other cell, and there are 5 parts, the largest of 252 cells.

In both, some cells cannot reach others at all, so their runs are not read as the effect of N alone. Every cell still reads one value of every used byte, and the readout reads every cell. A cell's input and its error therefore reach it directly, whatever its recurrent connections. That is the likely reason why N = 2 reached 80 at all, but it was not tested.

**Multiply-adds to 80 on R.** These are the receipt's `multiply_adds` at the row at which 80 is reached.

| N | Exact gradient | Sweep |
|---|---|---|
| 64 | 3.00 × 10^11 | 1.26 × 10^12 |
| 128 | 3.67 × 10^11 | 5.18 × 10^11 |
| 255 (full run) | 4.22 × 10^11 | 3.47 × 10^11 |
| 10 near (record) | 5.59 × 10^11 | 1.25 × 10^12 |

Under the exact gradient, N = 64 reached 80 on R with fewer multiply-adds than the full run; under the sweep, no N did. These counts are the receipt's arithmetic and are not computation measured on this host.

**Beside the record and the reference.** The record (N = 10 from the neighbourhood) reached 80 on R after 15,360 updates under the exact gradient and 34,304 under the sweep (ratio 2.23). Decision 31's reference (E42's dense network with a learned code, under the exact gradient) reached it after 6,400 updates. Every run of this job, the full runs included, needed more updates than the reference under the exact gradient. The input of these networks is the limited code, and the reference's is learned.

**On "enough".** No statement of this report says that a number of connections "is enough" or "is needed" in general. On seed 73129, in this network, under the exact gradient the smallest N of the grid that was as fast as the full run was 64; under the sweep no N below 255 was.

## 7. The predictions

The reviewer recorded these predictions before any run; their outcomes are computed by `e62_tables.py` (`exp45/e62/E62_TABLES.json`, entry `predictions`).

| | Statement (abbreviated) | Chance given | Outcome | Figures |
|---|---|---|---|---|
| P62.1 | Checks 1, 2, 3 and 5 pass at the first attempt | 0.65 | holds | each was run once and passed |
| P62.2 | The exact gradient reaches 80 on R at every N from 4 to 255 | 0.75 | holds | 7,680 to 18,176 updates |
| P62.3 | The exact gradient reaches 80 on R at N = 2 | 0.45 | holds | 15,616 |
| P62.4 | The sweep reaches 80 on R at every N from 16 to 255 | 0.55 | holds | 7,168 to 42,752 |
| P62.5 | The sweep does not reach 80 on R at N = 2 | 0.70 | does not hold | it reached 80 after 30,976 |
| P62.6 | Under each rule, N = 255 needs fewer updates to 80 on R than the record | 0.70 | holds | 7,168 against 34,304 (sweep); 8,704 against 15,360 (exact gradient) |
| P62.7 | N = 10 from all cells reaches 80 on R within 0.75 to 1.25 times the record's updates, under each rule | 0.55 | does not hold | exact gradient 1.05 times; the sweep did not reach 80 |
| P62.8 | At every N at which both rules reach 80 on R, the sweep needs more updates | 0.85 | does not hold | at N = 255 the sweep needed fewer (7,168 against 8,704) |
| P62.9 | Under the exact gradient, N = 32 is as fast as the full run | 0.50 | does not hold | 11,520 updates against a limit of 10,880 (slower) |

## 8. What the job cannot show

- Anything beyond one seed, one family, half of the pairs and this network. The result is of one seed and is not confirmed; other seeds follow only on the coordinator's word.
- How many connections a cell needs when its input or its error must come through other cells. Here the input is limited (each cell reads 46 of 512 values), but every cell reads one value of every used byte, and the readout is dense. A cell therefore receives input and error directly, and N governs only how cells combine what they hold over positions.
- A number of connections that suits a larger network. With 256 cells, every cell is reached from every cell within four steps at N = 16, three at N = 32 and two at N = 64, so Mr. Xu's example of several passes (1, 50, 50², 50³) cannot be reproduced at this width.
- Differences smaller than those between draws. Each N is one draw of connections, the draws for different N are not nested, and N = 10 from all cells shares 112 of its 2,560 connections with the record's network. Under the sweep, the uneven order of updates to 80 across N (section 6) is of this kind.
- The effect of N apart from the scale of the start, which changes with N so that a cell's recurrent input keeps its variance.
- Computation: CPU time does not fall with N in this implementation.
- Connections that change during the run, or a cell that sets its own number: the connections are fixed here.
- Relay over a limited number of steps (E61): the error passes over the whole item in every run.
- Requirement A, and anything about a life.

## Appendix A. The difference between the runner's `build` and `run` and those of `exp50_code.py`

Produced by `difflib.unified_diff` over the source text of each function (one line of context); the same text is in `exp45/e62/exp62_build_run_vs_exp50_code.diff`. The other functions of the runner (the graph figures, saving the state, the checks and `main`) are new and are in `exp62_connections.py`.

```diff
--- exp50_code.py:build
+++ exp62_connections.py:build
@@ -1,10 +1,10 @@
-def build(setting, seed):
-    """arch26.new_net; the value of a turned-on entry from it (S50.amplitude_for) and the code table (S50.code_table); E42's F10 start;
-    then the start of the input map: E49's Interface for C-dense and C16-in, and for C46-one E49's Interface with full masks whose
-    input lists are replaced by draw_one's, the kept weights of the input map scaled by sqrt(512 / 46) and the others set to zero."""
-    k_in, k_out = SETTINGS[setting]
-    n = Net()
+def build(N, choice, seed):
+    """exp50_code.build's text for the setting C46-one, with N connections a cell drawn from all cells or from the neighbourhood."""
+    assert choice in ("all", "near") and (choice == "all" or N == 10)
+    setting = SETTING
+    k_in, k_out = R50.SETTINGS[setting]
+    n = R50.Net()
     m = arch26.new_net(seed)
     with T.no_grad():
-        z1 = T.tanh(m.l1(m.book(T.tensor(USED))))
+        z1 = T.tanh(m.l1(m.book(T.tensor(R50.USED))))
         n.mean_squared_length = float(z1.pow(2).sum(1).mean())
@@ -12,8 +12,8 @@
     n.table, n.owner = S50.code_table(seed, n.amplitude)
-    n.own = draw_code(seed)
-    assert all(n.owner[v] == b for b, vs in n.own.items() for v in vs) and sum(o >= 0 for o in n.owner) == ON * len(USED)
-    layer = L42.Layer(m, seed, N=10)                 # F10: period set "1", variant "step", the local neighbourhood
+    n.own = R50.draw_code(seed)
+    assert all(n.owner[v] == b for b, vs in n.own.items() for v in vs) and sum(o >= 0 for o in n.owner) == R50.ON * len(R50.USED)
+    layer = L42.Layer(m, seed, N=N, wide=(choice == "all"))
     if setting == "C46-one":
         iface = I49.Interface(m, seed, D, U)         # both masks full: nothing is scaled
-        lists = draw_one(seed)
+        lists = R50.draw_one(seed)
         iface.k_in = k_in; iface.full = False
--- exp50_code.py:run
+++ exp62_connections.py:run
@@ -1,7 +1,8 @@
-def run(setting, rule, seed, out, cmd, dose=None, verify_scorer=False):
-    k_in, k_out = SETTINGS[setting]
+def run(N, choice, rule, seed, out, cmd, dose=None, ckpt_every=CKPT, resume=False, die_after=None):
+    setting = SETTING
+    k_in, k_out = R50.SETTINGS[setting]
     local = RULES[rule]
-    dose = dose or DOSES[setting]
-    assert dose % ROW == 0
-    rp = receipt_path(out, setting, rule, seed, dose)
+    dose = dose or DOSE
+    assert dose % ROW == 0 and ckpt_every % ROW == 0
+    rp = receipt_path(out, N, choice, rule, seed, dose); cp = rp.with_suffix(".ckpt")
     paths = dict(receipt=rp, start=rp.with_name(rp.stem + "_start.pt"), weights=rp.with_suffix(".pt"),
@@ -11,17 +12,20 @@
         paths.update(weights_first_dose=rp.with_name(rp.stem + f"_at{FIRST_DOSE}.pt"), answers_first_dose=rp.with_name(rp.stem + f"_at{FIRST_DOSE}_answers.json"))
-    R_.refuse_existing(*paths.values(), *(str(p) + ".tmp" for p in paths.values()))
+    resuming = resume and cp.exists()
+    if resuming:
+        R_.refuse_existing(paths["weights"], paths["answers"], str(paths["weights"]) + ".tmp")
+    else:
+        R_.refuse_existing(*paths.values(), *(str(p) + ".tmp" for p in paths.values()), cp)
     Path(out).mkdir(parents=True, exist_ok=True)
-    n = build(setting, seed); m, layer, iface = n.m, n.layer, n.iface
-    cov = coverage(n); rd = R49.readers(iface)
-    print(f"e50 {setting} {rule} {seed}: value of a turned-on entry {n.amplitude:.4f}; cells without input {cov['cells_without_input']}; "
+    n = build(N, choice, seed); m, layer, iface = n.m, n.layer, n.iface
+    cov = R50.coverage(n); rd = R49.readers(iface)
+    print(f"e62 N{N} {choice} {rule} {seed}: value of a turned-on entry {n.amplitude:.4f}; cells without input {cov['cells_without_input']}; "
           f"a needed byte reaches at least {cov['cells_a_needed_byte_reaches']['min']} cells", flush=True)
     if cov["cells_without_input"] or cov["cells_a_needed_byte_reaches"]["min"] == 0 or rd["cell_min"] == 0:
-        raise SystemExit(f"e50 {setting} {seed}: a cell without input, a needed byte that reaches no cell, or a cell that is not read; stopped")
-    install(n)
+        raise SystemExit(f"e62 N{N} {choice} {seed}: a cell without input, a needed byte that reaches no cell, or a cell that is not read; stopped")
+    R50.install(n)
     opt = R42.make_opt(m, layer)
-    cnt = counts(n)
-    assert cnt["weights_taking_part"] == WEIGHTS_TAKING_PART[setting], (setting, cnt)
-    assert abs(cnt["products_with_a_turned_on_value"] - PRODUCTS_TURNED_ON[setting]) <= 100, (setting, cnt)
+    cnt = R50.counts(n)
+    assert cnt["weights_taking_part"] == weights_taking_part(N), (N, cnt)
     ma = cnt["weights_taking_part"]
-    groups = digit_groups(seed)
+    groups = R50.digit_groups(seed)
     A, Rs, TA, LP = R42.panels(seed)
@@ -29,21 +33,42 @@
     bt = E.batches(tp, seed, E.KMAX["M"]); dg = C26.Digest()
-    R49.save_pt(weights_blob(n), paths["start"])
-    rec = dict(experiment="E50", kind="birth", setting=setting, config=setting, rule=rule, local=local, seed=seed, dose=dose, row_every=ROW,
-               k_in=k_in, k_out=k_out, cells=U, N=layer.N, batch=X.BATCH, kmax=E.KMAX["M"], counts=cnt, multiply_adds_per_position=ma,
-               multiply_adds_per_position_second_count=cnt["products_with_a_turned_on_value"],
-               params_module=sum(q.numel() for q in m.parameters()), readers=rd, coverage=cov,
-               code=dict(digest=dg16(n.owner), amplitude=n.amplitude, mean_squared_length_of_present_code=n.mean_squared_length,
-                         used=len(USED), values_owned=sum(o >= 0 for o in n.owner), table_sha256=n.table_sha,
-                         names_in_weight_files=dict(table="code_table", owners="code_owner", amplitude="code_amplitude")),
-               draw_digests=dict(input=dg16(n.lists), readout=dg16(I49.draws(seed, D, k_out)[1])),
-               start_scale=dict(input=(D / k_in) ** 0.5 if k_in < D else 1.0, lm=1.0),
-               digit_groups=dict(sizes=[len(g) for g in groups], note="cells with input for 0 to 2, 3 or 4, 5 or more of the ten digits under E49's 16 values a cell"),
-               optimizer=R42.opt_record(m, layer, opt),
-               update="exp26_candidate.step's text with e50_sparse_code.forward(m, layer, iface, table, x, local) in place of arch26.forward",
-               data="exp11_cell.batches([(t, split(t, .5, seed)[0]) for t in DEV], seed, 12)",
-               forward="e50_sparse_code.forward (the specification's own forward)",
-               start_file=paths["start"].name, start_sha256=X16.sha_file(paths["start"]), start_tensors_sha256=R_.tensors_sha(m.state_dict()),
-               mask_sha256=dict(recurrent=R42.tsha(layer.mask), input=R42.tsha(iface.mask_in), readout=R42.tsha(iface.mask_out)),
-               versions=versions(), provenance=provenance(cmd), complete=False, curve=[])
-    curve = rec["curve"]; cpu_up = cpu_sc = 0.0; w0 = time.time(); losses = []; first = None
+    start = 0; cpu_up = cpu_sc = 0.0; wall0 = 0.0; first = None
+    if resuming:
+        ck = T.load(cp, weights_only=False)
+        rec = ck["rec"]; start = ck["update"]; cpu_up, cpu_sc, wall0 = ck["cpu_up"], ck["cpu_sc"], ck["wall"]
+        assert R_.tensors_sha(m.state_dict()) == rec["start_tensors_sha256"], "the rebuilt network is not the saved run's start"
+        assert X16.sha_file(paths["start"]) == rec["start_sha256"], "the starting file on disk is not the saved run's"
+        c = time.process_time(); w = time.time()
+        for _ in range(start):
+            items = next(bt)
+            if first is None:
+                first = items
+            dg.add(items)
+        assert dg.value() == rec["curve"][-1]["data_digest"], "the redrawn batch digest differs from the saved receipt's last row"
+        assert rec["curve"][-1]["step"] == start
+        m.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
+        rec["resumes"].append(dict(update=start, at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), replay_cpu=time.process_time() - c,
+                                   replay_wall=time.time() - w, command=cmd))
+        print(f"e62 N{N} {choice} {rule} resumed at {start}", flush=True)
+    else:
+        R49.save_pt(R50.weights_blob(n), paths["start"])
+        rec = dict(experiment="E62", kind="birth", setting=setting, config=setting, choice=choice, rule=rule, local=local, seed=seed, dose=dose, row_every=ROW,
+                   k_in=k_in, k_out=k_out, cells=U, N=layer.N, batch=X.BATCH, kmax=E.KMAX["M"], counts=cnt, multiply_adds_per_position=ma,
+                   multiply_adds_per_position_second_count=cnt["products_with_a_turned_on_value"],
+                   params_module=sum(q.numel() for q in m.parameters()), readers=rd, coverage=cov,
+                   code=dict(digest=dg16(n.owner), amplitude=n.amplitude, mean_squared_length_of_present_code=n.mean_squared_length,
+                             used=len(R50.USED), values_owned=sum(o >= 0 for o in n.owner), table_sha256=n.table_sha,
+                             names_in_weight_files=dict(table="code_table", owners="code_owner", amplitude="code_amplitude")),
+                   draw_digests=dict(input=dg16(n.lists), readout=dg16(I49.draws(seed, D, k_out)[1]), connections=dg16(connections(layer))),
+                   graph=graph(layer),
+                   start_scale=dict(input=(D / k_in) ** 0.5 if k_in < D else 1.0, lm=1.0, recurrent=1.0 if layer.full else (U / (N + 1)) ** 0.5),
+                   digit_groups=dict(sizes=[len(g) for g in groups], note="cells with input for 0 to 2, 3 or 4, 5 or more of the ten digits under E49's 16 values a cell"),
+                   optimizer=R42.opt_record(m, layer, opt),
+                   update="exp26_candidate.step's text with e50_sparse_code.forward(m, layer, iface, table, x, local) in place of arch26.forward",
+                   data="exp11_cell.batches([(t, split(t, .5, seed)[0]) for t in DEV], seed, 12)",
+                   forward="e50_sparse_code.forward (the specification's own forward)",
+                   start_file=paths["start"].name, start_sha256=X16.sha_file(paths["start"]), start_tensors_sha256=R_.tensors_sha(m.state_dict()),
+                   mask_sha256=dict(recurrent=R42.tsha(layer.mask), input=R42.tsha(iface.mask_in), readout=R42.tsha(iface.mask_out)),
+                   checkpoint_every=ckpt_every, resumes=[],
+                   versions=R50.versions(), provenance=provenance(cmd), complete=False, curve=[])
+    curve = rec["curve"]; w0 = time.time() - wall0; losses = []
 
@@ -52,9 +77,5 @@
         c = time.process_time()
-        sc = score_all(m, A, Rs, TA, LP)
+        sc = R50.score_all(m, A, Rs, TA, LP)
         cpu_sc += time.process_time() - c
         row = dict(step=u, **sc)
-        if verify_scorer:
-            ref = R42.score_all(m, A, Rs, TA, LP)
-            row["scorer_equals_e42"] = all(sc[k] == ref[k] for k in E42_SCORE_KEYS)
-            assert row["scorer_equals_e42"], f"this file's scorer differs from E42's at row {u}"
         lim = R42.check_limit(m, layer)
@@ -63,6 +84,6 @@
         assert all(all(v.values()) for v in mk.values()), f"E49's check 3 failed at row {u}: {mk}"
-        fx = fixed(n)
+        fx = R50.fixed(n)
         assert all(fx.values()), f"check 2 failed at row {u}: {fx}"
-        row.update(limit=lim, masks=mk, fixed=fx, kept_mean_magnitude=R49.magnitudes(m, layer, iface), by_what_a_cell_knows=knows_figure(n, groups),
-                   input_size=input_size(n), cpu_updates=cpu_up, cpu_scoring=cpu_sc, multiply_adds=ma * X.BATCH * u, data_digest=dg.value(),
+        row.update(limit=lim, masks=mk, fixed=fx, kept_mean_magnitude=R49.magnitudes(m, layer, iface), by_what_a_cell_knows=R50.knows_figure(n, groups),
+                   input_size=R50.input_size(n), cpu_updates=cpu_up, cpu_scoring=cpu_sc, multiply_adds=ma * X.BATCH * u, data_digest=dg.value(),
                    loss=(sum(losses) / len(losses)) if losses else None, finite_losses=all(v == v and abs(v) != float("inf") for v in losses),
@@ -71,3 +92,3 @@
         o = row["ops"]
-        print(f"e50 {setting} {rule} {seed} {u}: A {row['A']:.2f} R {row['R']:.2f} (bytes {row['R_first_byte']:.1f} {row['R_second_byte']:.1f}) "
+        print(f"e62 N{N} {choice} {rule} {seed} {u}: A {row['A']:.2f} R {row['R']:.2f} (bytes {row['R_first_byte']:.1f} {row['R_second_byte']:.1f}) "
               f"taught {row['taught']:.2f} len12 {row['len12']:.1f} len40 {row['len40']:.1f} max {o['max89']:.0f} min {o['min89']:.0f} "
@@ -75,4 +96,5 @@
 
-    score_row(0)
-    for u in range(1, dose + 1):
+    if start == 0:
+        score_row(0)
+    for u in range(start + 1, dose + 1):
         items = next(bt)
@@ -80,6 +102,6 @@
             first = items
-            rec["first_batch_loss_start"] = first_batch_loss(n, first)
-            rec["cosine_first_batch_start"] = cosines(n, first)
+            rec["first_batch_loss_start"] = R50.first_batch_loss(n, first)
+            rec["cosine_first_batch_start"] = R50.cosines(n, first)
         dg.add(items)
-        c = time.process_time(); losses.append(step(n, opt, items, local)); cpu_up += time.process_time() - c
+        c = time.process_time(); losses.append(R50.step(n, opt, items, local)); cpu_up += time.process_time() - c
         if u % ROW == 0:
@@ -87,3 +109,3 @@
             if mid and u == FIRST_DOSE:
-                R49.save_pt(weights_blob(n), paths["weights_first_dose"])
+                R49.save_pt(R50.weights_blob(n), paths["weights_first_dose"])
                 paths["answers_first_dose"].write_text(json.dumps({str(k): E.respond(m, [p for p, _ in it]) for k, it in LP.items()}))
@@ -91,10 +113,16 @@
                                          tensors_sha256=R_.tensors_sha(m.state_dict()), answers_file=paths["answers_first_dose"].name,
-                                         cosine_first_batch=cosines(n, first))
-            if u % 1024 == 0 or u == dose:
+                                         cosine_first_batch=R50.cosines(n, first))
+            ck_now = ckpt_every and u % ckpt_every == 0 and u < dose
+            if u % 1024 == 0 or u == dose or ck_now:
                 rp.write_text(json.dumps(rec, indent=1))
-    rec["cosine_first_batch_end"] = cosines(n, first)
+            if ck_now:
+                save_ckpt(cp, m, opt, rec, u, cpu_up, cpu_sc, time.time() - w0)
+        if die_after is not None and u == die_after:
+            print(f"e62 dying after update {u} (check 3)", flush=True)
+            os._exit(9)
+    rec["cosine_first_batch_end"] = R50.cosines(n, first)
     sd = m.state_dict()
-    R49.save_pt(weights_blob(n), paths["weights"])
+    R49.save_pt(R50.weights_blob(n), paths["weights"])
     paths["answers"].write_text(json.dumps({str(k): E.respond(m, [p for p, _ in it]) for k, it in LP.items()}))
-    fx = fixed(n)
+    fx = R50.fixed(n)
     rec.update(complete=True, weights=paths["weights"].name, weights_sha256=X16.sha_file(paths["weights"]),
@@ -105,5 +133,7 @@
                optimizer_state_for=sorted(k for k, q in m.named_parameters() if opt.state.get(q)), data_digest=dg.value(),
-               cpu_updates=cpu_up, cpu_scoring=cpu_sc, wall=time.time() - w0, processes_beside=os.environ.get("E50_BESIDE", "not recorded"))
+               cpu_updates=cpu_up, cpu_scoring=cpu_sc, wall=time.time() - w0, processes_beside=os.environ.get("E62_BESIDE", "not recorded"))
     rp.write_text(json.dumps(rec, indent=1))
-    print("e50 complete", rp.name, rec["tensors_sha256"][:16], "weights sha256", rec["weights_sha256"][:16], flush=True)
+    if cp.exists():
+        cp.unlink()
+    print("e62 complete", rp.name, rec["tensors_sha256"][:16], "weights sha256", rec["weights_sha256"][:16], flush=True)
     return rec
```

## Appendix B. Committed files

The files of E62, from `git ls-tree -r -l 3a7cfcd1 -- exp62_connections.py e62_driver.py e62_tables.py exp45/e62 exp45/e62_qual` run in `MOD97_LOCAL_RULE_20261106/` (size in bytes, path). This report, `exp45/e62/E62_REPORT.md`, was committed after that commit. The saved states (`*.ckpt`) are deleted when a run finishes and are not committed. The weights files (`.pt`) were added with `git add -f`.

```
3674 e62_driver.py
13400 e62_tables.py
32384 exp45/e62/E62_TABLES.json
7846 exp45/e62/E62_TABLES.md
446773 exp45/e62/e62_N10_all_B_73129.json
5893085 exp45/e62/e62_N10_all_B_73129.pt
9016 exp45/e62/e62_N10_all_B_73129_answers.json
5893325 exp45/e62/e62_N10_all_B_73129_at24576.pt
9016 exp45/e62/e62_N10_all_B_73129_at24576_answers.json
5893265 exp45/e62/e62_N10_all_B_73129_start.pt
446794 exp45/e62/e62_N10_all_S_73129.json
5893085 exp45/e62/e62_N10_all_S_73129.pt
9016 exp45/e62/e62_N10_all_S_73129_answers.json
5893325 exp45/e62/e62_N10_all_S_73129_at24576.pt
9016 exp45/e62/e62_N10_all_S_73129_at24576_answers.json
5893265 exp45/e62/e62_N10_all_S_73129_start.pt
447139 exp45/e62/e62_N128_all_B_73129.json
5893115 exp45/e62/e62_N128_all_B_73129.pt
9016 exp45/e62/e62_N128_all_B_73129_answers.json
5893355 exp45/e62/e62_N128_all_B_73129_at24576.pt
9016 exp45/e62/e62_N128_all_B_73129_at24576_answers.json
5893295 exp45/e62/e62_N128_all_B_73129_start.pt
447268 exp45/e62/e62_N128_all_S_73129.json
5893115 exp45/e62/e62_N128_all_S_73129.pt
9016 exp45/e62/e62_N128_all_S_73129_answers.json
5893355 exp45/e62/e62_N128_all_S_73129_at24576.pt
9016 exp45/e62/e62_N128_all_S_73129_at24576_answers.json
5893295 exp45/e62/e62_N128_all_S_73129_start.pt
447186 exp45/e62/e62_N16_all_B_73129.json
5893085 exp45/e62/e62_N16_all_B_73129.pt
9016 exp45/e62/e62_N16_all_B_73129_answers.json
5893325 exp45/e62/e62_N16_all_B_73129_at24576.pt
9016 exp45/e62/e62_N16_all_B_73129_at24576_answers.json
5893265 exp45/e62/e62_N16_all_B_73129_start.pt
446954 exp45/e62/e62_N16_all_S_73129.json
5893085 exp45/e62/e62_N16_all_S_73129.pt
9016 exp45/e62/e62_N16_all_S_73129_answers.json
5893325 exp45/e62/e62_N16_all_S_73129_at24576.pt
9016 exp45/e62/e62_N16_all_S_73129_at24576_answers.json
5893265 exp45/e62/e62_N16_all_S_73129_start.pt
448044 exp45/e62/e62_N255_all_B_73129.json
5893115 exp45/e62/e62_N255_all_B_73129.pt
9016 exp45/e62/e62_N255_all_B_73129_answers.json
5893355 exp45/e62/e62_N255_all_B_73129_at24576.pt
9016 exp45/e62/e62_N255_all_B_73129_at24576_answers.json
5893295 exp45/e62/e62_N255_all_B_73129_start.pt
448379 exp45/e62/e62_N255_all_S_73129.json
5893115 exp45/e62/e62_N255_all_S_73129.pt
9016 exp45/e62/e62_N255_all_S_73129_answers.json
5893355 exp45/e62/e62_N255_all_S_73129_at24576.pt
9016 exp45/e62/e62_N255_all_S_73129_at24576_answers.json
5893295 exp45/e62/e62_N255_all_S_73129_start.pt
446096 exp45/e62/e62_N2_all_B_73129.json
5892991 exp45/e62/e62_N2_all_B_73129.pt
9008 exp45/e62/e62_N2_all_B_73129_answers.json
5893295 exp45/e62/e62_N2_all_B_73129_at24576.pt
9016 exp45/e62/e62_N2_all_B_73129_at24576_answers.json
5893235 exp45/e62/e62_N2_all_B_73129_start.pt
445960 exp45/e62/e62_N2_all_S_73129.json
5892991 exp45/e62/e62_N2_all_S_73129.pt
9012 exp45/e62/e62_N2_all_S_73129_answers.json
5893295 exp45/e62/e62_N2_all_S_73129_at24576.pt
9016 exp45/e62/e62_N2_all_S_73129_at24576_answers.json
5893235 exp45/e62/e62_N2_all_S_73129_start.pt
446896 exp45/e62/e62_N32_all_B_73129.json
5893085 exp45/e62/e62_N32_all_B_73129.pt
9016 exp45/e62/e62_N32_all_B_73129_answers.json
5893325 exp45/e62/e62_N32_all_B_73129_at24576.pt
9016 exp45/e62/e62_N32_all_B_73129_at24576_answers.json
5893265 exp45/e62/e62_N32_all_B_73129_start.pt
447067 exp45/e62/e62_N32_all_S_73129.json
5893085 exp45/e62/e62_N32_all_S_73129.pt
9016 exp45/e62/e62_N32_all_S_73129_answers.json
5893325 exp45/e62/e62_N32_all_S_73129_at24576.pt
9016 exp45/e62/e62_N32_all_S_73129_at24576_answers.json
5893265 exp45/e62/e62_N32_all_S_73129_start.pt
446040 exp45/e62/e62_N4_all_B_73129.json
5892991 exp45/e62/e62_N4_all_B_73129.pt
9016 exp45/e62/e62_N4_all_B_73129_answers.json
5893295 exp45/e62/e62_N4_all_B_73129_at24576.pt
9016 exp45/e62/e62_N4_all_B_73129_at24576_answers.json
5893235 exp45/e62/e62_N4_all_B_73129_start.pt
445965 exp45/e62/e62_N4_all_S_73129.json
5892991 exp45/e62/e62_N4_all_S_73129.pt
9016 exp45/e62/e62_N4_all_S_73129_answers.json
5893295 exp45/e62/e62_N4_all_S_73129_at24576.pt
9012 exp45/e62/e62_N4_all_S_73129_at24576_answers.json
5893235 exp45/e62/e62_N4_all_S_73129_start.pt
446969 exp45/e62/e62_N64_all_B_73129.json
5893085 exp45/e62/e62_N64_all_B_73129.pt
9016 exp45/e62/e62_N64_all_B_73129_answers.json
5893325 exp45/e62/e62_N64_all_B_73129_at24576.pt
9016 exp45/e62/e62_N64_all_B_73129_at24576_answers.json
5893265 exp45/e62/e62_N64_all_B_73129_start.pt
446982 exp45/e62/e62_N64_all_S_73129.json
5893085 exp45/e62/e62_N64_all_S_73129.pt
9016 exp45/e62/e62_N64_all_S_73129_answers.json
5893325 exp45/e62/e62_N64_all_S_73129_at24576.pt
9016 exp45/e62/e62_N64_all_S_73129_at24576_answers.json
5893265 exp45/e62/e62_N64_all_S_73129_start.pt
446571 exp45/e62/e62_N8_all_B_73129.json
5892991 exp45/e62/e62_N8_all_B_73129.pt
9016 exp45/e62/e62_N8_all_B_73129_answers.json
5893295 exp45/e62/e62_N8_all_B_73129_at24576.pt
9016 exp45/e62/e62_N8_all_B_73129_at24576_answers.json
5893235 exp45/e62/e62_N8_all_B_73129_start.pt
446622 exp45/e62/e62_N8_all_S_73129.json
5892991 exp45/e62/e62_N8_all_S_73129.pt
9016 exp45/e62/e62_N8_all_S_73129_answers.json
5893295 exp45/e62/e62_N8_all_S_73129_at24576.pt
9016 exp45/e62/e62_N8_all_S_73129_at24576_answers.json
5893235 exp45/e62/e62_N8_all_S_73129_start.pt
15766 exp45/e62/exp62_build_run_vs_exp50_code.diff
36694 exp45/e62/interrupted/20261119T003506Z/e62_N2_all_B_73129.json
5893235 exp45/e62/interrupted/20261119T003506Z/e62_N2_all_B_73129_start.pt
36785 exp45/e62/interrupted/20261119T003506Z/e62_N2_all_S_73129.json
5893235 exp45/e62/interrupted/20261119T003506Z/e62_N2_all_S_73129_start.pt
4497 exp45/e62/logs/driver.log
4222 exp45/e62/logs/driver.out
23969 exp45/e62/logs/e62_N10_all_B_73129.log
23925 exp45/e62/logs/e62_N10_all_S_73129.log
24216 exp45/e62/logs/e62_N128_all_B_73129.log
24194 exp45/e62/logs/e62_N128_all_S_73129.log
25234 exp45/e62/logs/e62_N16_all_B_73129.log
23954 exp45/e62/logs/e62_N16_all_S_73129.log
24199 exp45/e62/logs/e62_N255_all_B_73129.log
24473 exp45/e62/logs/e62_N255_all_S_73129.log
25338 exp45/e62/logs/e62_N2_all_B_73129.log
25177 exp45/e62/logs/e62_N2_all_S_73129.log
23965 exp45/e62/logs/e62_N32_all_B_73129.log
23962 exp45/e62/logs/e62_N32_all_S_73129.log
23761 exp45/e62/logs/e62_N4_all_B_73129.log
23589 exp45/e62/logs/e62_N4_all_S_73129.log
24858 exp45/e62/logs/e62_N64_all_B_73129.log
25847 exp45/e62/logs/e62_N64_all_S_73129.log
23767 exp45/e62/logs/e62_N8_all_B_73129.log
23675 exp45/e62/logs/e62_N8_all_S_73129.log
5181 exp45/e62_qual/check1_73129.json
15949 exp45/e62_qual/check1_73129_runs/B/e62_N10_near_B_dry512_73129.json
9016 exp45/e62_qual/check1_73129_runs/B/e62_N10_near_B_dry512_73129_answers.json
15940 exp45/e62_qual/check1_73129_runs/S/e62_N10_near_S_dry512_73129.json
9016 exp45/e62_qual/check1_73129_runs/S/e62_N10_near_S_dry512_73129_answers.json
17400 exp45/e62_qual/check2_73129.json
4041 exp45/e62_qual/check3_73129.json
29720 exp45/e62_qual/check3_73129_runs/i/e62_N32_all_S_dry2048_73129.json
9016 exp45/e62_qual/check3_73129_runs/i/e62_N32_all_S_dry2048_73129_answers.json
30115 exp45/e62_qual/check3_73129_runs/ii/e62_N32_all_S_dry2048_73129.json
9016 exp45/e62_qual/check3_73129_runs/ii/e62_N32_all_S_dry2048_73129_answers.json
4928 exp45/e62_qual/check5_73129.json
1123 exp45/e62_qual/logs/check1.log
288 exp45/e62_qual/logs/check2.log
18 exp45/e62_qual/logs/check3.log
336 exp45/e62_qual/logs/check5.log
29286 exp62_connections.py
```
