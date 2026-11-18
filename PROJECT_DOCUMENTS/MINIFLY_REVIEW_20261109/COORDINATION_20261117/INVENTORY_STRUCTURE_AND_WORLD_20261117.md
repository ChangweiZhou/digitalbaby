# Inventory of settings: the experiments on structure and on the long-life world (E39, E42, E44, E47, E49, E50), from the record (17 November 2026)

Item 3 of `PLAN.md` beside this file, for the experiments on structure and on the long-life world. The experiments on the dense network (E26 to E38, E40, E41, E43, E46, E48) are another agent's inventory and enter here only where a file of this scope cites them.

**How it was made.** Read-only. Every entry comes from a file that was read with `git show <ref>:<path>`, or from the working tree for the reviewer's own documents. Nothing was trained, no code of any branch was run, and no ref was fetched. Receipts (JSON) were read with `python3 -I` and the standard library. Where the record does not give a thing, the entry says "not found". Nothing in this file is a new measurement, and no number here was computed by me: each is copied from the file named beside it.

**Refs and the commits at which they were read** (17 November 2026, between about 21:05 and 21:30 UTC).

| Short name | Ref | Commit read |
|---|---|---|
| `D` | `origin/codex/minifly-consolidated-20261023` (the default branch) | 7cabb13a |
| `B39` | `origin/claude/gallant-edison-p0qa68` (E39 and E44's calibration; an ancestor of `D`, all its files are on `D`) | a11702e4 |
| `B42` | `origin/claude/gallant-dijkstra-i9zgsr` (E42) | 44f91edc |
| `B47` | `origin/claude/e47-worker-stage-0-3d0ya6` (E47's trial) | 578a877b |
| `B49` | `origin/claude/peaceful-noether-vcvrmf` (E49) | a36de661 |
| `B50` | `origin/claude/e50-fixed-input-code-017dSE` (E50, in progress) | c61b1067 |
| `BL2` | `origin/claude/e44-l2-first-long-lives` (the first long lives, in progress) | 9519302d (the ref stood at f249b419 when the reading began and was moved by another session while it went on; the runner `sw_run.py` is the same file in both commits) |
| `WT` | the working tree of `claude/vibrant-cerf-sje726` (HEAD bc50e87f), folder `PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/` | files as they stood on disk |

Path names: `M` = `MOD97_LOCAL_RULE_20261106`; `K` = `PROJECT_DOCUMENTS/COORD_CHECKS_20261116`; `R` = `PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109`. "Review 9.x" is a section of `MINIFLY_PAST_WORK_REVIEW_FOR_MR_ZHOU_20261106.md` on `D`. "Memo n" is a section of `R/MEMO_TO_COORDINATOR_LEARNED_STRUCTURE_20261115.md` in `WT`.

**Terms.** Rule B is exact back-propagation (`local=False`); rule S is the sweep (`local=True`): every linear map sends its error back through the signs of its weights times one scale for the matrix (`M/exp3_state.py` on `D`, class `SignLinear`). An *update* is one step of the optimizer. *The dense network* is the network of E26: a byte table of 258 rows of 8 numbers, a layer `l1` from 8 to 512, a GRU of 256 cells in which every cell reads every cell, and a readout `lm`, `l2`, `l3` (1,124,113 parameters; `M/arch20.py` on `D`, class `NetRNN`). *The first family* is add, sub, max and min under modulus 89. Score A is the share right on untaught pairs of add and sub at 12 letters; score R the share right on untaught pairs of add and sub whose answer needs the reduction by the modulus, without letters; the *length test* is untaught add at 12 and at 40 letters. A threshold is *reached* at the first of three scored rows in a row at or above it (the "three-row version").

---

## Part 1. The values actually run, experiment by experiment

### 1.1 E39: the small world (jobs L0, L1, L1a; L1b not run)

Sources: brief `OPUS_TASK_E39_SMALL_WORLD_PROTOCOL_L0_L1_20261115.md` on `D` (revision 5); report `M/exp45/e39/E39_REPORT.md` on `D`; checks `M/exp45/e39/L0/E39_L0_CHECKS.md` on `D`; code `M/smallworld/sw_net.py`, `sw_run.py`, `sw_world.py` on `D`; receipts read on `D`: `M/exp45/e39/sw_naive_w1_v64.json`, `M/exp45/e39/sw_sweep_w3_v64_h96_u96_vu512.json`, `M/exp45/e39/L1a/sw_solo_naive_w1_t12_h96.json`, `M/exp45/e39/L0/lives/c4/sw_sweep+replay_w1_v4_rho0.5.json`.

| Setting | Value run | Where read |
|---|---|---|
| What ran | L0: nine checks with lives of 4, 8 and 16 visits. L1: 48 lives (three rungs, two arms, worlds 1 to 8, 64 visits). L1a: 72 runs of one task alone. L1b (24 lives with a birth) and L2 (long lives in this world): not run. | report, "Summary" and "Addendum" |
| Learning rules | `naive` (rule B) and `sweep` (rule S) in L1 and L1a. `replay` and `sweep+replay` exist in the runner and ran only in L0's check lives. | brief, "Arms"; L0 checks 4, 6, 9 |
| Optimizer | AdamW; learning rate 0.001; betas (0.9, 0.98); weight decay 0.25; one optimizer for a whole life. No epsilon is passed (`T.optim.AdamW(groups, lr=lr, weight_decay=wd, betas=(.9, .98))`), so the library's default applies; its number is in no receipt: not found. | `sw_net.make_opt`; receipts, field `optimizer` |
| Exempt from decay | The byte table `book.weight` (decay 0.0); every other parameter decays. | receipts, `optimizer.groups` |
| Batch | 64 items of the stream an update: 48 taught, all learned from; 16 asked, answered before the update and never learned from. | receipts, `ledger` |
| Network | `sw_net.SWNet`: byte table 258 × 8; `l1` 8 → H; GRU H → U; `lm` U → H; `l2` H → H; `l3` H → 257. H = U = 64 (52,625 parameters), 96 (102,353), 192 (349,841; L1a only). | `sw_net.py`; receipts, `model` |
| Cell equations | The GRU's, written out step by step (`arch26.forward`). | `M/arch26.py` on `D` |
| What a cell reads | Every cell reads every cell and the whole output of `l1`; the readout reads every cell. No neighbourhood, no periods, no phases. | `sw_net.py`, `arch26.py` |
| Starting weights | `T.manual_seed(world_seed)`, then the constructors; nothing is rescaled. Reason for the draw: not found. | brief, "The learner"; `sw_net.new_net` |
| Error through time | Through the whole item (`arch26.forward(m, x, local, True)`); the state is zero at the start of every item. | `sw_net.step` |
| Coding of a byte | A learned row of 8 numbers, then `tanh(l1)`. | `sw_net.py` |
| Tasks | 540 tasks: 18 operations (16 linear `(c1·a + c2·b) mod p` with c1, c2 in {1, 2, −1, −2}; max; min) at 30 moduli, 8 to 37; groups of 2; the last 20 groups are probe tasks. | brief, "The world"; `sw_world.py` lines 9 to 20 |
| Fraction taught | 0.5 (`round(p*p/2)`): 32 to 684 taught pairs a task. | `sw_world.py`, `TAUGHT_FRACTION = 0.5`; report |
| Letters | 0 to 4, in taught and in asked items alike; no separate test at another length. | `sw_world.py`, `KMAX = 4` |
| Dose | L1: 64 visits of 256 updates (16,384 updates; rung R0) or of 512 (32,768; rungs vu512 and h96vu512). L1a: 16,384 updates a run. | report, "The learner and the ledger" |
| Seeds | Worlds 1 to 8; a world's number seeds its tasks, its schedule and the network's starting weights. L1a: worlds 1 and 2. | brief; receipts, `settings.world` |
| Visit length; life length | 256 or 512 updates; 64 visits (L1); 4, 8, 16 visits (L0). The 1,024-visit lives the protocol was built for were not run. | report |
| Schedule | Visit 0 is group 0; afterwards a new group with chance 0.2, otherwise a return, the class of absence (1 to 4, 5 to 32, 33 or more visits) drawn first. | brief, "The rules, in full" |
| Store and share of rehearsal | A reservoir of raw items, capacity `(4 × P) // 15` items (14,033 at 64 units). Share ρ: 0.5 in L0's check lives (24 of 48 items from the store), 0 as a check; the brief names 0.25 and 0.5 "for a start". No life of L1 used the store. | brief, "The buffer"; L0 check 6; receipt of the `sweep+replay` check life |
| Scores and thresholds | L1: a task's level is the mean of its last 32 asked items; criterion 0.8. Gate under `naive`, in at least six of eight worlds: both tasks of the first group reach 0.8; the median first-visit level of later tasks lies between 0.10 and 0.90; a life of 1,024 visits is forecast within 6 CPU-hours. L1a: share right at least 0.8 over all of the task's asked items in the last 64 updates, checked every 64 updates; a size qualifies if six of the eight upper-band tasks reach it within 8,192 updates under `naive`. | brief, "Job L1", "The criterion", "Job L1b" |
| Found | "L1's gate fails at all three rungs, always low." (review 9.141). "L1a (E39): no linear task of the small world is learned alone." (review 9.146). | review |

### 1.2 E42: cells with a limited number of local connections and their own periods

Sources: brief `OPUS_TASK_E42_CELLS_WITH_LIMITED_LOCAL_CONNECTIONS_AND_PERIODS_20261116.md` on `D` (revision 2); specification `M/e42_cells.py` on `D` (the copy of `K/e42_cells_spec.py`); runner `M/exp42_cells.py` on `D`; receipt read: `M/exp45/e42/e42_P8_S_73129.json` on `B42`; reports on `B42`: `M/exp45/e42/E42_REPORT_three_seeds.md`, `M/exp45/e42/E42_REPORT_73129.md`, `M/exp45/e42_qual/E42_STAGE0_73129.md`. No file of E42's results is on `D`.

| Setting | Value run | Where read |
|---|---|---|
| What ran | 45 runs: stage 1, configurations `G`, `F10`, `A10`, `P8` under S and under B on three seeds (24); stage 2, `N5`, `N20`, `W10`, `P4`, `P16`, `P8-last`, `P8-source` under S only on three seeds (21). `P8-mean` was deferred and not run. | three-seed report, "Runner and resumes per run"; brief, "Configurations" |
| Optimizer | `exp26_candidate.make_opt(m)`: AdamW, learning rate 0.001, betas (0.9, 0.98), weight decay 0.25; the layer's added parameters (`a`, and `Wx` in `source`) join the decayed group. Epsilon: not passed; number not found. | brief, "Data, dose, optimizer"; receipt, `optimizer` and `optimizer_read_back` |
| Exempt from decay | `book.weight` only. | receipt, `optimizer.groups` |
| Batch | 64 items, all taught. | receipt, `batch` |
| Network | The dense network (feed-forward width 512, 256 cells, 1,124,113 parameters) with its recurrent layer changed; the byte table, `l1`, the input map and the readout are unchanged. | brief, "What changes in the network" |
| Cell equations | The GRU's. With periods a cell is updated at positions f, f + p, …; it combines the steps of its period by variant `step` (a learned weight for each step, starting at 1/p; the default), `last` (the last step only) or `source` (a separate matrix for each step). | brief, "The forward" |
| How many cells a cell reads | `G`: all 255 others. `F10`, `A10`, `W10`, the `P` configurations: N = 10. `N5`: 5. `N20`: 20. Always its own previous value as well. | brief, table "Configurations" |
| How they are chosen | `Random(f"e42\|connections\|{seed}").sample(sorted(B_i), N)` for the cells in index order. Fixed in `F10`; in the "changing" configurations a connection at least 512 updates old whose strength is below theta times the mean strength of its cell's connections is given up every 256 updates and replaced by a random unconnected cell of the neighbourhood, starting at zero. theta = 0.5, chosen in stage 0 among 0.1, 0.25 and 0.5 as the value whose share replaced per event over updates 768 to 4,096 of `A10` under S was nearest 5% (0.06%, 0.67%, 7.99%). | brief, "Changing connections", check 10; stage-0 report, check 10 |
| Neighbourhood | The 48 cells within 3 rows and 3 columns on a 16 × 16 grid with joined edges; in `W10` all 255 other cells. | brief, "Geometry" |
| Periods and phases | Sets: all 1; (1, 2, 4) with 128, 64, 64 cells; (1, 2, 4, 8) with 128, 64, 32, 32; (1, 2, 4, 8, 16) with 128, 64, 32, 16, 16. Which cells: `Random(f"e42\|periods\|{seed}")` shuffle. Phase: `randrange(p_i)` from `Random(f"e42\|phases\|{seed}")`. | brief; `e42_cells.py`, `PERIOD_SETS`, `draws` |
| Which cells read the input; which the readout reads | Every cell reads all 512 values of the current byte's code; every readout unit reads every cell. | brief, "The forward" |
| Starting weights | `arch26.new_net(seed)` unchanged; then the recurrent weights outside the mask are set to zero and those inside are multiplied by sqrt(256 / (N + 1)): 4.82 at N = 10, 1 in `G`. Reason given: "so that a cell's recurrent input keeps its variance". | brief, "The start"; `e42_cells.py`, class `Layer` |
| Error through time | Through the whole item under both rules. Under S the masked recurrent map uses the mean magnitude of the weights inside the mask as its scale; the gates, the products, the weights `a` and the holding of a value are differentiated exactly under both rules. | brief, "The two rules" |
| Coding of a byte | Learned (byte table 258 × 8 and `l1`). | brief |
| Task | E26's birth: the first family (modulus 89), `exp11_cell.batches([(t, split(t, .5, seed)[0]) for t in DEV], seed, 12)`, answer bytes as targets. | brief; receipt, `data` |
| Fraction taught | 0.5. | receipt, `data` |
| Letters | Training 0 to 12 (`kmax` 12). Tests: A and the taught panel at 12; R without letters; the length test at 12 and 40. | receipt, `kmax`; `exp42_cells.py`, `panels` |
| Dose | 24,576 updates, scored every 256. | receipt, `dose`, `row_every` |
| Seeds | 73129, 93683, 116089. | three-seed report |
| Scores and thresholds | A, R, the taught panel, the length test; updates and weight-items to 50, 80 and 90 (three-row version, single-row beside it); a candidate needs *fewer* updates below 2/3 of its reference's, *more* above 1.5; at 40 letters *lower* or *higher* at 15 points on the mean of the last three rows; a cell of the table carries what two of three seeds share. One stop: no limited configuration reaches 50 on A on seed 73129. The cosine between the sweep's and the exact gradient of the recurrent matrix at the first batch and at the end. | brief, "Scores", "How the result is stated" |
| Found | The worker's table of descriptions: "F10 \| G \| S \| A80 updates \| more", "P8 \| A10 \| S \| A80 updates \| fewer", "A10 \| F10 \| S \| A80 updates \| about the same" (`E42_REPORT_three_seeds.md`, "Descriptions"). The coordinator's review has no reading of E42's result: review 9.157 says E42 is "finished and not yet read in full by me". | report; review |

### 1.3 E44: the long-life world in the full-scale format (calibration, job F1)

Sources: brief `OPUS_TASK_E44_LONG_LIFE_WORLD_IN_THE_FULL_SCALE_FORMAT_20261116.md` on `D` (revision 2); world `M/smallworld/lw_world.py` on `D`; runner `M/smallworld/sw_run.py` on `D`; receipt read: `M/exp45/e44/F1/lw_naive_w1_v8_turn.json` on `D`; report `M/exp45/e44/E44_REPORT.md` on `D`.

| Setting | Value run | Where read |
|---|---|---|
| What ran | 96 lives of 8 visits (48 gate lives; 24 `sweep-after-naive`; 24 `naive` with decay 0) and 16 second-lap branches at 256 updates a visit. | report, "Decision record" |
| Learning rules (arms) | `naive`: B's birth, lives by B. `sweep`: S's birth, lives by S. `sweep-after-naive`: B's birth, lives by S. `naive` with decay 0 in the life. The replay arms were not run. | report, "The learner and the ledger" |
| Optimizer | One fresh AdamW for the life (`sw_net.make_opt(m, lr, wd)`): learning rate 0.001, betas (0.9, 0.98), decay 0.25 (0 in the decay-0 set). Epsilon: not passed; number not found. | receipt, `optimizer`; report |
| Exempt from decay | `book.weight`. | receipt, `optimizer.groups` |
| Batch | 64 stream items an update: 48 taught, learned from; 16 asked, answered first. | receipt, `ledger` |
| Network | `sw_net.SWNet(256, 128)`: the dense network at the half width, 317,457 parameters; every cell reads every cell and the whole input; no periods. | receipt, `model`; report |
| Starting weights | Loaded, not drawn: E41's half-width births of seed 73129 at a taught fraction of 0.50 (`M/exp45/e41/e41_<B\|S>_h256_f50_73129.pt`), 40,960 updates on the first family. Nothing is rescaled. | receipt, `learner` |
| Error through time | Through the whole item (`sw_net.step`). | receipt, `update` |
| Coding of a byte | Learned (byte table and `l1`). | `sw_net.py` |
| Tasks | 56 families: add, sub, max, min at each modulus from 41 to 100 except 73, 83, 89, 97 (224 tasks); a task's code is two bytes from 128 to 255; the last four families are probe tasks. The birth family is the first family (modulus 89). | brief, "The world"; `lw_world.py` lines 10 to 22 |
| Fraction taught | 0.10 of a task's pairs (168 to 1,000 pairs); an argument of the world's constructor. The birth was taught 0.50 of its pairs. | `lw_world.py`, `TAUGHT_FRACTION = 0.10`; receipt, `world.taught_share`, `learner.fraction` |
| Letters | 0 to 12 in the stream, taught and asked alike; the fixed list is answered without letters. | `lw_world.py`, `KMAX = 12`; brief, item 5 |
| Dose | 8 visits in turn at 64, 128 or 256 updates a visit (512, 1,024 or 2,048 updates); the second lap 8 more visits at 256. | report |
| Seeds | Worlds 1 to 8; the birth of seed 73129 in every world. | report |
| Visit length; life length; schedule | 64, 128, 256 updates; 8 visits, families 0 to 7 in turn (`--in-turn`). The world's own schedule (a new family with chance 0.05, otherwise a return by class of absence) was not used in the calibration. | brief; receipt, `settings` |
| Store and share of rehearsal | None run. | report |
| Scores and thresholds | First-visit level on reduction pairs: the mean of add's and sub's share right on asked reduction items over the last quarter of a family's first visit. Gate under `naive`, per visit length: in at least six of eight worlds the median over families 1 to 7 lies between 0.20 and 0.90, and a life of 1,024 visits is forecast within 6 CPU-hours. Kept: a family's share right on its 200 list items at a later visit divided by its share at the end of its own visit, with "decay alone" beside it (every parameter but the byte table multiplied by 0.99975 to the power of the updates since). | brief, "Changes to the runner and scorer", "The calibration" |
| Found | "E44: the long-life instrument is calibrated. … The adopted visit length is 256." and "Nothing is kept from one visit to the next." (review 9.156). | review |

### 1.4 E44, job L2, release 1: the first long lives (roster note; in progress)

Sources: roster note `OPUS_TASK_E44_L2_ROSTER_FIRST_LONG_LIVES_20261117.md`, revision 1 on `D` and revision 2 on `BL2` (entered by the worker from the audit, in the coordinator's absence); runner `M/smallworld/sw_run.py` on `BL2`; check receipt `M/exp45/e44/L2/checks/E44_L2_CHECKS.json` on `BL2`. What follows is what the note specifies; at 9519302d the branch holds the checks and two 8-visit lives and no receipt of a life of 256 or of 1,024 visits.

| Setting | Value specified | Where read |
|---|---|---|
| Arms | `naive`; `replay` (ρ = 0.5); `naive-wd0`; `replay-wd0`. All four are rule B. The sweep is named for release 2. | note, "Arms (release 1)" |
| Optimizer | As 1.3; decay 0.25, or 0 in the two `wd0` arms. | note; check receipt, check 6 |
| Learner | E41's half-width births under B at a taught fraction of 0.50; world w uses seed (73129, 93683, 116089)[(w − 1) mod 3]. | note, "Learners and worlds" |
| Visit length; life lengths | 256 updates. 1,024 visits (worlds 1 to 4, `naive` and `replay`); 256 visits (worlds 5 to 8 for those arms; worlds 1 to 8 for the two `wd0` arms); 32 lives. A learning probe of 4 visits from the states after 64, 256 and 1,024 visits. | note, "Runs" |
| Schedule | The world's: a new family with chance 0.05, otherwise a return. | brief of E44 |
| Store and share of rehearsal | Reservoir over every new taught item, capacity `(4 × P) // 23` = 55,209 items (1,269,828 bytes, one copy of the weights in 32-bit numbers); ρ = 0.5: 24 of the 48 learned items of an update come from the store. | note; check receipt, check 4 |
| Scores | The definitions D1 to D15 of the note; *stuck* or *still learning* by 90% intervals (t over worlds) of two trends; between arms *higher*, *lower* or *not distinguished*; the comparison of `naive` and `replay` is *closed* at 1,024 visits by rule D12. | note, "What is recorded", "How the result is stated" |
| Run so far | The checks of the runner change and the note's checks 1 to 7: all pass. Two 8-visit lives with the other two births: median first-visit level 0.573 (world 2, seed 93683) and 0.313 (world 3, seed 116089). Forecast for release 1: 48.2 CPU-hours. | check receipt, `all_passed`, `other_births`, `checks.7` |
| Found | Nothing yet. `R/COORDINATION_20261117/LOG.md` in `WT` records that the trial has been running since 21:04 UTC and that a helper session for the sweep's long lives was started at 21:10; no file of either was on a ref that I read. | `LOG.md` |

### 1.5 E47: locality at the input and the output (the trial)

Sources: brief `OPUS_TASK_E47_LOCALITY_AT_THE_INPUT_AND_THE_OUTPUT_20261116.md` on `D` (revision 2); specification `K/e47_local_spec.py` on `D`; receipt read: `M/exp45/e47/e47_32_place_S_73129.json` on `B47`; report `M/E47_TRIAL_REPORT.md` on `B47`.

| Setting | Value run | Where read |
|---|---|---|
| What ran | Four runs at 1,024 cells: arms `one` and `place`, each under S and under B. The roster (the arm `scatter` at 1,024 cells; 4,096 cells) was not run. | report, "The trial", "Stopped" |
| Optimizer | E26's, with the module's two unused recurrent tensors taken out; `Wc`, `bc`, `bfar`, `a` in the decayed group. AdamW, learning rate 0.001, betas (0.9, 0.98), decay 0.25. Epsilon: number not found. | receipt, `optimizer`; report, "Optimizer" |
| Exempt from decay | `book.weight`. | receipt |
| Batch | 64. | receipt |
| Network | 1,024 cells on a 32 × 32 grid with joined edges; the byte table, `l1` (8 → 512), `l2`, `l3` are the dense network's; weights 965,905 (`one`), 971,665 (`place`). | brief, "What a cell costs"; receipt, `weight_count` |
| Cell equations | The GRU's; periods as E42's variant `step`. | brief |
| What a cell reads | N = 10 cells of its neighbourhood (48 cells, radius 3) and itself; fixed; drawn by E42's rule and string on the larger grid. | brief; `e47_local_spec.py`, `draws` |
| Periods and phases | `one`: every cell 1. `place`: the 256 interface cells 1; the 768 far cells in four groups of 192 with periods 2, 4, 8, 16, nearest the interface first. Phases by E42's rule. | receipt, `cells_by_period`; brief |
| Which cells read the input; which the readout reads | Only the 256 cells of one 16 × 16 patch (the interface); each of them reads all 512 values of the byte's code and is read by every readout unit. A far cell has a bias and no input. | brief, "The network, as code" |
| Starting weights | `arch26.new_net(seed)` for what the dense network has. Every cell's recurrent weights are drawn uniformly in ±1/√11, "the range E42's start has" (the dense ±1/16 times √(256/11), `e47_local_spec.py` comment), and the cells' biases in ±1/16, from `T.Generator().manual_seed(seed * 1000 + SIDE)`; the far cells' input biases from the same seed plus one. The mean size of a recurrent weight at the start is 0.152 (interface) and 0.150 (far). | brief, "Start"; receipt, `start`; report, "Cosines … and magnitudes" |
| Error through time | Through the whole item. Under S the error crosses each connection by the sign of its weight times one scale, the mean magnitude of all recurrent weights (`IndexSign`). | brief, "The sweep" |
| Coding of a byte | Learned. | brief |
| Task, fraction, letters, dose, seed | E26's birth (the first family); 0.5; training 0 to 12 letters, tests as E42; 24,576 updates, scored every 256; seed 73129 only. | receipt, `data`, `kmax`, `dose`, `seed` |
| Scores and thresholds | E42's, with multiply-adds a position in place of weight-items; at every row also with the far cells held at zero and held at their mean. *Used*: A or R falls by 15 points or more with the far cells at their mean. *Not used*: both fall by less than 5 with them at zero. *A constant is used*: 15 or more at zero and less than 5 at the mean. Otherwise *partly*. | brief, "How the result is stated" |
| Found | All four runs reach 80 on R (17,664 and 19,200 updates under S; 11,008 and 11,264 under B; report, table of the trial). "No run reaches *used* (a fall of 15 or more at the mean). None is *not used* (both falls under 5 at zero), and none is *a constant is used*." (report, "Use of the far cells at the end"). The coordinator's review has no reading of the trial. | report |

### 1.6 E49: a limited interface

Sources: brief `OPUS_TASK_E49_A_LIMITED_INTERFACE_20261116.md` on `D` (revision 2); specification `K/e49_interface_spec.py` on `D`; receipt read: `M/exp45/e49/e49_I16_S_73129.json` on `B49`; report `M/exp45/e49/E49_REPORT.md` on `B49`.

| Setting | Value run | Where read |
|---|---|---|
| What ran | Eight runs on seed 73129: `I64`, `I16`, `in16`, `out8`, each under S and under B. The reference, E42's `F10`, was not run again. | report, "Summary" |
| Optimizer | E42's: AdamW, learning rate 0.001, betas (0.9, 0.98), decay 0.25; `book.weight` without decay. Weights outside a mask start at zero, receive no gradient and stay at zero. Epsilon: number not found. | receipt, `optimizer`; brief, "Runner" |
| Batch | 64. | receipt |
| Network | E42's `F10` layer: 256 cells, each reading 10 fixed cells of its neighbourhood of 48 and itself, every cell at period 1; feed-forward width 512. | brief, "The network, as code" |
| Which cells read the input | Cell i reads K_IN of the 512 values of the byte's code, the same for its three gates: `Random(f"e49\|input\|{seed}\|{K_IN}").sample(range(512), K_IN)`. K_IN: 64 (`I64`), 16 (`I16`, `in16`), 512 (`out8`). The values have no positions; the subset is random. | brief |
| Which cells the readout reads | Readout unit j (of the 512 of `lm`) reads K_OUT of the 256 cells: `Random(f"e49\|readout\|{seed}\|{K_OUT}").sample(range(256), K_OUT)`. K_OUT: 32 (`I64`), 8 (`I16`, `out8`), 256 (`in16`). | brief |
| Starting weights | E42's start; then the kept weights of the input map are multiplied by √(512 / K_IN) and those of `lm` by √(256 / K_OUT), "as E42 scaled the kept recurrent weights"; the others are zero; biases untouched. The receipt of `I16` records 5.657 for both. | brief, "The start"; receipt, `start_scale` |
| Error through time; the two rules | Through the whole item; under S, E42's masked sign rule on each masked map (the mean magnitude inside the mask). | brief |
| Coding of a byte | Learned; `l1`, `l2`, `l3` stay dense. | brief, "What it cannot show" |
| Task, fraction, letters, dose, seed | The first family; 0.5; 0 to 12 letters; 24,576 updates; seed 73129. | receipt |
| Scores and thresholds | E42's scores; updates and multiply-adds to 50, 80, 90; the level: the mean of A, of R and of the length test at 40 letters over the last three rows, *lower* or *higher* at 15 points; a run that does not reach 80 on R is *still rising* if its R over the last three rows is 5 points or more above the three rows ending 2,048 updates earlier. | brief, "How the result is stated" |
| Found | "**Under backpropagation**, no limited setting reaches 80 on R within the dose." "**Under the sweep**, both limited settings stay at or below 3 on every panel at every scored row." In stage 2, `in16` under B reaches 80 on R at 20,480 updates (report, "Summary"). The coordinator's review has no reading of E49's result. | report |

### 1.7 E50: an input code in which a value stands for one byte (in progress)

Sources: brief `OPUS_TASK_E50_AN_INPUT_CODE_IN_WHICH_A_VALUE_STANDS_FOR_ONE_BYTE_20261116.md`, revision 1 on `D` ("Not released: it waits for the reviewer's audit") and revision 2 on `B50` (entered by the worker from the audit, in the coordinator's absence); specification `K/e50_sparse_code_spec.py` on `D`; checks `M/exp45/e50_qual/E50_STAGE0_73129.md` on `B50`; receipts read: `M/exp45/e50/e50_C-dense_B_73129.json` and `e50_C-dense_S_73129.json` on `B50`, both `complete = false` with five rows, the last at update 1,024.

| Setting | Value specified in revision 2 (and what the receipts show) | Where read |
|---|---|---|
| Stages | 1: `C-dense` (512 values a cell, 256 cells a readout unit) under S and B, 24,576 updates; running. 2: `C16-in` (16 values a cell, E49's lists; the readout dense), 49,152 updates, under each rule under which `C-dense` reaches 80 on R. 3: `C46-one` (46 values, one of every used byte; the readout dense), 49,152 updates, under each rule under which `C16-in` does not. The settings `C64` and `C16` of revision 1 are withdrawn. | brief revision 2, "Settings and stages" |
| Optimizer | E49's. The receipts list AdamW, learning rate 0.001, betas (0.9, 0.98), decay 0.25, `book.weight` without decay. The byte table and `l1` take no part and receive no gradient. | receipts, `optimizer`, `cosine_first_batch_start` |
| Batch | 64. | receipts |
| Network | E42's `F10` layer (256 cells, N = 10) with E49's two masks. | brief; receipts, `cells`, `N` |
| Coding of a byte | Fixed, not learned: each of the 46 used symbols turns on 11 of the 512 values, its own and no others (`Random(f"e50\|code\|{seed}")` shuffle); 6 values have no owner. The value of a turned-on entry is 3.2387 for seed 73129, set so that a used byte's code has the mean squared length of the dense network's code at the start (115.38). | brief; receipts, `code` |
| Starting weights | E49's start and scaling. `C-dense`: factor 1.0 on the input map and on `lm`. `C16-in`: √(512/16). `C46-one`: √(512 / 46) = 3.336 on the input map. | receipts, `start_scale`; brief |
| Task, fraction, letters, seed | The first family; 0.5; 0 to 12 letters; seed 73129. | receipts, `data`, `kmax`, `seed` |
| Scores | E49's, and: the two answer bytes apart on R's items; the four operations apart; the mean size of the recurrent weights into cells grouped by the number of digits they have input for. | brief, "What is recorded" |
| Found | Nothing yet: at update 1,024 A is 2.0 (B) and 2.2 (S) and R is 1.2 and 1.1. The five checks before the runs pass (`E50_STAGE0_73129.md`). | receipts, `curve`; stage-0 report |

---

## Part 2. Settings with one value in the whole scope

"Taken over from the dense network" means: the value is the one E26's network was run with, reached here through `exp26_candidate.make_opt`, `exp26_candidate.step`, `arch26.forward` or `arch20`'s class.

### 2.1 One value in every experiment of the scope

| Setting | Value | Where it was first chosen, and the reason the record gives | Taken over from the dense network? |
|---|---|---|---|
| Kind of optimizer | AdamW (decoupled weight decay) | `M/mod97_local.py` on `D`, lines 142 to 144, for the arms `bpw…` and `signw…` of the decay sweep on the mod-97 window network; the line's comment gives the reason: "decoupled weight decay (AdamW), lr .001, as in the grokking papers". Review 9.3 reports that sweep. | Yes |
| Learning rate | 0.001 | The same line of `mod97_local.py` (the same comment). Before it, 0.001 was "V181's settings: Adam lr 0.001" (`M/REPORT.md` on `D`). It is carried in `M/exp11_cell.py` line 274, `M/exp20_reference.py` line 53 (`LR, WD, BETAS = .001, .3, (.9, .98)`) and from there in `M/exp26_candidate.py`. The record describes its origin in two ways: the E42 brief, "Learning rate and decay were chosen for the dense GRU"; the reviewer's instruction `R/COORDINATION_20261117/WORKER_E44_L2_SWEEP_LONG_LIVES.md` in `WT`, "was chosen for back-propagation". I found no run in this scope with another value. | Yes |
| Betas | (0.9, 0.98) | The same line of `mod97_local.py`. A reason beyond that line's comment: not found. | Yes |
| Epsilon | The library's default; no file read passes one | Not chosen in any file read; the number is in no receipt: not found. | Yes (the same constructor call) |
| Parameters exempt from decay | The byte table only | `M/exp16_integrated.py` on `D`, function `adamw16`, under `E16_BOOKDECAY=off`, which `exp20_reference.py` sets on import (review 9.139). Reason: not found in the files read. | Yes |
| Items of the stream an update | 64 | `mod97_local.py` line 38 (`BATCH = 64`); the first mod-97 run copied its set-up from an earlier campaign (`M/REPORT.md`, "Setup"). Review 9.24 compared batches of 64, 256 and 1,024 under the sign rule on an earlier learner and review 9.27 adopted 64 "as the toy-scale settings". | Yes |
| Targets | The answer bytes only (answer-masked cross-entropy) | `exp26_candidate.step`. The E39 brief lists it as a limit: "Answer bytes are the only targets." Reason: not found in the files read. | Yes |
| Error through time | Through the whole item; the state is zero at the start of every item | `M/arch26.py`, argument `through_time=True` ("credit through the whole item (both arms of E26)"); review 9.105. | Yes |
| The sweep's form | Signs of the weights times one scale for each (stacked) matrix, the mean magnitude of its weights (of the weights inside the mask where a matrix is masked); gates and products differentiated exactly | `M/exp3_state.py`, `SignLinear`; review 9.105 keeps "one scale per stacked matrix" for E26. The E42 brief calls the scale on a masked matrix a choice ("The sign rule on few entries"). | Yes; the masked form was added in E42 |
| Equations of a cell | The GRU's | `arch20.NetRNN("gru")`. The E42 brief, choice 2: "The GRU's equations are kept in this job." | Yes |
| Number of recurrent layers; the readout's own layers | One layer; `l2` and `l3` dense | `arch20.py`. | Yes |
| Width of a byte's row in the table | 8 numbers (where a table is used; E50 does not use it) | `M/REPORT.md`, "Setup": "a learned 8-dimensional byte embedding", copied from the earlier campaign's model. Reason: not found. | Yes |
| Starting draw of everything the dense network has | The constructors after `T.manual_seed(seed)`; for the GRU and `lm` uniform between −1/16 and 1/16 (`R/AUDIT_E49_LIMITED_INTERFACE_BRIEF_20261116.md`, section 2) | `arch20.new_net`. Reason: not found. | Yes |
| Rule for rescaling weights that are kept when others are removed | Multiply by the square root of (the dense number of sources / the kept number): E42 √(256/(N+1)); E47 the range ±1/√11; E49 and E50 √(512/K_IN) and √(256/K_OUT) | First in the E42 brief ("so that a cell's recurrent input keeps its variance"); E49 "as E42 scaled the kept recurrent weights". No other scale was run (E49 report, "What I did not check": "one scale was run, the brief's"). | It is defined from the dense network's variance |
| Scoring of an answer | Greedy generation; right only if every answer byte and the end symbol are right | `sw_net.answer`; `exp11_cell.respond`. | Yes |
| In the two worlds: split of an update | 48 taught items learned from, 16 asked items answered first | E39 brief, "One stream, read once". Reason: not found. | No (new in E39) |
| In the two worlds: the optimizer over a life | One optimizer, fresh at the start of the life, kept through it | E39 brief; E44 brief ("E41 saved weights only, so the life starts a fresh AdamW", `R/AUDIT_E44_LONG_LIFE_WORLD_BRIEF_20261116.md`, section 2). | No |
| In the two worlds: size of the store | The bytes of one copy of the weights (`4 × P`) | E39 brief, "The buffer". Reason: review 9.135, "a replay buffer the size of one copy of the weights". | No |
| In the two worlds: share of rehearsal | ρ = 0.5 is the only share with a run (ρ = 0 as a check) | L2 roster note: "ρ = 0.5 follows E38's and E43's rehearsal share; it is not tuned, and that is a limit." | From E38 and E43, which ran on the dense network |

### 2.2 One value in every experiment on structure (E42, E47, E49, E50), though another value occurs in a world

| Setting | Value | Where it was first chosen, and the reason given | Taken over from the dense network? |
|---|---|---|---|
| Weight decay | 0.25 | Review 9.149: "the decay of 0.25 was chosen in 9.24, 9.25 and 9.27". Those sections chose it for the sign rule on earlier learners (9.24: "batch 64, decay 0.25" gave the best retention and composition of any local-rule arm; 9.27: adopted "as the toy-scale settings"). For E26 the review says: "Every rule run on file uses 0.25 and every backprop birth 0.30. … Here the backprop arm takes the rule's decay." The E42 brief says of it: "chosen for the dense GRU". Other values in this scope: 0 in E44's decay-0 set and in the `wd0` arms of L2, all under rule B. | Yes |
| Fraction of pairs taught | 0.5 | `exp11_cell.split(t, .5, seed)` in E26's birth; the first mod-97 run already taught half of the pairs (`M/REPORT.md`, "Setup"). Reason: not found. E44's world teaches 0.10 (review 9.146: "taught from a tenth of their pairs"). | Yes |
| Families taught | The first family alone (modulus 89) | E26's birth. | Yes |
| Letters in training | 0 to 12 | `M/exp11_cell.py` on `D`, `KMAX = {"M": 12, "N": 3}`. Review 9.8 (an earlier experiment): "The task puts a run of up to 12 distractor bytes between the two operands". Reason for 12: not found. E39 used 0 to 4. | Yes |
| Dose | 24,576 updates (E50's stages 2 and 3 are specified at 49,152 and have no receipt yet) | `M/exp45.py` line 32, `DEV_UPDATES = 24576`. Reason: not found. The E47 brief: "The task and the dose are E26's birth, which the present network learns: 80% on untaught pairs at 3,886 to 6,899 updates". | Yes |
| Feed-forward width | 512 | `arch20.py` header: "hidden widths are chosen to put the parameter count within 5% of bank2's 1,125,265"; the same header says these networks "are trained by backprop through time only". The worlds used 64 to 192 (E39) and 256 (E44). | Yes |
| Neighbourhood of a cell | 48 cells within 3 rows and columns on a grid with joined edges (`W10` of E42 under S is the one exception) | Memo 22.11, item 3: "The simplest arrangement with no edges." | No (new in E42) |
| Connections a cell reads in E47, E49, E50 | N = 10, fixed | The user's statement of 16 November, review 9.140: "each cell can connect at most N=10 neurons (after all addition is simple), but we can adjust N". E47 brief: "Learning rate, decay and N are E42's." | No |
| Share of cells at period 1 where periods are used in E42 | Half | E42 brief, choice 4. Reason: not found. | No |
| Scoring interval and thresholds | Every 256 updates; 50, 80, 90 in the three-row version | The interval: E42 brief, "Scores, every 256 updates". The thresholds and the three-row version: E41 (review 9.138: "thresholds are 50, 80 and 90"; "a second version of each threshold (held for three consecutive scored rows)"). | The thresholds are E41's, which ran on the dense network |
| Seeds beyond the first | None in E47, E49 and E50 (seed 73129 only); E42 has three | E47, E49 and E50 briefs ("One seed"). | – |

---

## Part 3. Settings that were varied

| Setting | Experiment | Values | Outcome, with source |
|---|---|---|---|
| Learning rule | E39 L1, L1a | `naive` (B), `sweep` (S) | Mean score over 64 visits: `naive` 11.3 / 18.3 / 19.6%, `sweep` 9.6 / 12.4 / 12.3% at the three rungs (E39 report, summary 4); no solo run reaches the criterion under either (addendum, summary 3). |
| Learning rule | E42 stage 1 | S, B in `G`, `F10`, `A10`, `P8` | Updates to 80 on A, S against B: `G` *about the same*, `F10` *more*, `A10` *more* (marked with a star), `P8` *about the same* (three-seed report, "Descriptions"). |
| Learning rule (birth and life) | E44 F1 | `naive`, `sweep`, `sweep-after-naive` | Mean over worlds of the median first-visit level at 256 updates: 0.453, 0.614, 0.196 (E44 report, table of medians). |
| Learning rule | E47 trial | S, B | 80 on R at 17,664 and 19,200 updates under S, 11,008 and 11,264 under B (trial report). |
| Learning rule | E49 | S, B | Under B `in16` reaches 80 on R at 20,480 updates and the other three settings do not; under S none does (E49 report). |
| Width of the network | E39 L1, L1a | H = U = 64, 96, 192 | No size qualifies: 0 of 8 upper-band tasks within 8,192 updates at each (addendum, "The size rule"). |
| Visit length | E39 L1 | 256, 512 updates | The gate fails at both (E39 report, "The gate, rung by rung"). |
| Visit length | E44 F1 | 64, 128, 256 updates | Worlds inside the gate: 0, 4, 8 of 8; 256 adopted (E44 report). |
| Weight decay in the life | E44 F1, under `naive` only | 0.25, 0 | Mean of medians at 256 updates 0.453 against 0.471; kept one visit later 0.021 against 0.031 (E44 report, tables). |
| Weight decay in the life | L2 release 1, under B only | 0.25, 0 (`naive-wd0`, `replay-wd0`) | In progress; no receipt of a life. |
| Store and rehearsal | E39 L0 (check lives only) | none; ρ = 0.5; ρ = 0 | Checks only (with ρ = 0, `replay` equals `naive`); no life of L1 used the store (L0 check 6). |
| Store and rehearsal | L2 release 1, under B only | none; ρ = 0.5 | In progress. |
| Number of connections a cell reads | E42, S only for 5 and 20 | 255 (`G`), 10, 5, 20 | `F10` and `A10` against `G`: *more* updates to 80 on A under both rules; `N5` and `N20` against `A10`: *about the same* (three-seed report). |
| Neighbourhood | E42, S only | 48 cells; all 255 (`W10`) | `W10` against `A10`: *about the same* (three-seed report). |
| Fixed or changing connections | E42, S and B | `F10`, `A10` | *About the same*; the reviewer's verification states that the comparison "compares about six working connections with about nine, not changing connections with fixed ones" (`R/E42_THREE_SEEDS_VERIFICATION_20261117.md`, section 7). |
| theta of the replacement rule | E42 stage 0 (dry runs of 4,096 updates, `A10`, S) | 0.1, 0.25, 0.5 | Share replaced per event 0.06%, 0.67%, 7.99%; 0.5 used everywhere (stage-0 report, check 10). |
| Periods | E42 | all 1; (1, 2, 4, 8) under S and B (`P8`); (1, 2, 4) and (1, 2, 4, 8, 16) under S only (`P4`, `P16`) | `P8` against `A10` under S: *fewer*; under B: *about the same*. `P4` and `P16` against `P8`: *about the same* in updates; `P16` at 40 letters *lower* on one seed of three (three-seed report). |
| How a cell combines the steps of its period | E42, S only | `step`, `last`, `source` (`mean` not run) | `P8-last` and `P8-source` against `P8`: *about the same* (three-seed report). |
| Number of cells | E47 (against E42's `F10`) | 256 (`F10`, on file), 1,024 (trial); 4,096 planned, not run | 1,024 cells against `F10`: *about the same* in updates to 80 on R in all four comparisons (trial report, "Comparisons"). |
| Periods of the far cells | E47 trial | `one`, `place` (`scatter` not run) | `place` against `one`: *about the same* under both rules (trial report). |
| Holding the far cells | E47 trial (a diagnostic at scoring, not a training setting) | plain; at zero; at their mean | All four runs read *partly* (trial report). |
| Input values a cell reads; cells a readout unit reads | E49 | (512, 256) `F10`; (64, 32); (16, 8); (16, 256); (512, 8) | Every limited setting needs *more* updates than `F10` to 80 on A and on R under each rule (E49 report, "Summary"). |
| Coding of a byte | E50 against E42 and E49 | learned; fixed with 11 values a byte | In progress. |
| Dose | E39 L1 (through the visit length) | 16,384; 32,768 updates | The gate fails at both. |
| Dose | E50 | 24,576 (stage 1); 49,152 (stages 2 and 3) | Stages 2 and 3 have not started on the ref read. |
| Seed | E42 | 73129, 93683, 116089 | For `F10`: 80 on R at 14,336 to 15,104 updates under S over three seeds and at 7,936 to 9,472 under B over two (E49 report, "What a seed alone changes"). |
| World (seed of tasks, schedule and, in E39, of the weights) | E39, E44 | 1 to 8 | Per-world figures in both reports. |
| Birth seed of the long-life learner | E44 F1 against L2's two 8-visit lives | 73129; 93683 (world 2); 116089 (world 3) | Medians 0.566 and 0.301 with seed 73129's birth, 0.573 and 0.313 with the other two (`E44_L2_CHECKS.json`, `other_births`). |

Settings that differ between E39's world and E44's world but were changed together, so that no outcome can be assigned to one of them: the fraction taught (0.5; 0.10), the operations (18; 4), the moduli (8 to 37; 41 to 100), the letters (0 to 4; 0 to 12), the chance of a new group (0.2; 0.05), the width (64 to 192; 256 and 128), and whether the learner has been born (E44 brief, "Why the small world is replaced").

---

## Part 4. Comparisons made under a setting chosen for one side

Each row names the comparison, the setting, the evidence that it was chosen for one side or for the dense network, and the evidence that it was not varied for the other side. "Not varied" means: I found no receipt, report or brief in this scope with another value for that arm.

| No. | Comparison | Setting held | Evidence of how it was chosen | Evidence that the other side never had another value |
|---|---|---|---|---|
| 4.1 | A limited or local network against the dense one: `F10`, `A10`, `P8` and the stage-2 configurations against `G` (E42); the boundary network against `F10` and `G` (E47); the limited interface against `F10` (E49) | Learning rate 0.001 and weight decay 0.25 | E42 brief, "What it cannot show": "**Settings.** Learning rate and decay were chosen for the dense GRU." Memo 22.10, the same sentence. E47 brief: "Learning rate, decay and N are E42's." E49 brief: "Learning rate and decay are E42's". | Every receipt read of E42, E47, E49 and E50 records `lr 0.001`, `weight_decay 0.25`. |
| 4.2 | Rule S against rule B, in every experiment of the scope | One learning rate (0.001) and one decay (0.25) for both rules | Decay: review, E26's design: "Every rule run on file uses 0.25 and every backprop birth 0.30. … Here the backprop arm takes the rule's decay"; 0.25 had been chosen for the sign rule on earlier learners (review 9.24, 9.27, 9.149). Learning rate: `mod97_local.py` gave 0.001 to both the back-propagation and the sign arms "as in the grokking papers"; the reviewer's files in `WT` state that it "was chosen for back-propagation" and that the sweep "took over" it (`WORKER_E44_L2_SWEEP_LONG_LIVES.md`, section 4; `TASK_E52_…`, "The question"). | No run of this scope gives the two rules different values, or either rule another learning rate. Decay 0 was run under B only (E44's decay-0 set is `naive`; L2's `wd0` arms are B). |
| 4.3 | A limited network against the dense one (E42, E49, E50) | The size of the kept weights at the start: multiplied by 4.82 (E42, N = 10), 2.83 or 5.66 (E49), 3.336 or 5.66 (E50), so that a cell's input has the dense network's variance | E42 brief, "The start" and choice 5 ("The scale of the kept weights at the start"); E49 brief, "The start"; `R/AUDIT_E49_LIMITED_INTERFACE_BRIEF_20261116.md`, section 2: "The kept weights are 2.83 times (`I64`) or 5.66 times (`I16`) larger than the dense ones. AdamW moves a weight by about 0.001 an update whatever its size, so one update is 3.2% of a weight's mean size in the dense maps, 1.1% at `I64` and 0.57% at `I16`." | E49 report, "What I did not check": "**Whether the start's scale matters**: one scale was run, the brief's." |
| 4.4 | E47: cells away from the boundary, *used* or not; `place` against `one`; the boundary network against `F10` | The starting range of the recurrent weights of every cell, far cells included (±1/√11), and the decay of 0.25 on weights that receive no gradient | The range is "the range E42's start has" (E47 brief), and in E42 every cell reads the input. `R/E47_TRIAL_VERIFICATION_20261117.md`, section 6: on the starting weights the variation of a state with the item "falls by a factor of six to ten with each ring"; "A weight that receives no gradient ends the dose at 0.2% of its starting size, by the decay alone"; "A quarter of the far cells have thus in effect been removed by the decay under both rules". The E47 brief states the decay's effect beforehand ("with decay 0.25 such a weight ends the dose at 0.2% of its start"). | One start and one decay in the four runs (receipts). The verification names the result that is missing: "a network started so that differences between items do reach rings 2 and 3". |
| 4.5 | Rule S against rule B in sparsely connected cells (`F10`, `A10` in E42; E47; E49) | The sweep's single scale for a matrix (the mean magnitude of all its kept weights) | Kept for the dense matrices of E26 (review 9.105). E42 brief: "The scale and the masked weight gradient are choices; the cosine … says how good the approximation is, not whether another choice would do better." E47 brief: "The sweep's scale is a mean over every cell's weights, which a cell does not have; a scale from a cell's own 11 weights is a later variant." | No run with another scale. In E47's `place` run under S the far cells' mean weight falls from 0.150 to 0.041 "and the sweep's scale falls with it, 0.151 to 0.081" (trial report). |
| 4.6 | A limited or local network against the dense one within one dose | 24,576 updates, the dose of the dense network's birth | E47 brief: the dense network reaches 80% on untaught pairs "at 3,886 to 6,899 updates". E49 brief: "The dose is E26's birth, 24,576 updates, which `F10` needs most of on its first seed"; "a setting slower by a factor above 1.59 cannot be told from one that does not learn; under backpropagation the factor is 2.54". | E49 report, "NOT_RUN and open items": "**Continuing a run beyond the dose**: not done" (`I64` and `out8` under B were still rising). E50's revision 2 doubles the dose for its limited settings; no receipt yet. |
| 4.7 | E42: changing against fixed connections, and every changing configuration, under B as under S | theta = 0.5 of the replacement rule | Chosen once, on `A10` under S, over updates 768 to 4,096 (E42 brief, check 10: "The value used everywhere"; stage-0 report). | Used unchanged under B and in `N5`, `N20`, `W10` and the `P` configurations. `R/E42_SEED_73129_VERIFICATION_20261116.md`, section 7: "The replacement rate in the full runs is three times the rate theta was chosen for." |
| 4.8 | E42: the number of connections, the neighbourhood, the period set and the variant | Varied under S only | E42 brief: "**Stage 2**, under the sweep only". | No receipt of `N5`, `N20`, `W10`, `P4`, `P16`, `P8-last` or `P8-source` under B on `B42`. |
| 4.9 | A limited network against the dense one at one fraction of taught pairs | Fraction 0.5 | The fraction of E26's birth. The dense network was also run at 0.25 and 0.10 in E41 (review 9.138). | Every receipt read of E42, E47, E49, E50 has `split(t, .5, seed)`. `PLAN.md` in `WT`, item 10, lists the limited and the periodic network at a quarter and a tenth as not done. |
| 4.10 | A limited network against the dense one at one width | Feed-forward width 512 and 256 cells | The dense network's own size (`arch20.py` header). The dense network was run at three widths in E41; memo 30.1: "What lowers the computation is a smaller network". | E42, E49 and E50 are at 256 cells and width 512; E47 varies the number of cells with the width at 512. |
| 4.11 | Long-life world: `sweep`, `sweep-after-naive` and `replay` against `naive` | Visit length 256 | The gate that adopted it was read "under `naive`" alone (E44 brief, "The gate, per visit length, under `naive`"). `R/AUDIT_E44_L2_ROSTER_NOTE_20261117.md`, section 4.2: under `replay` "A visit therefore teaches 6,144 new items. That is the number the calibration taught in a visit of 128 updates, where the gate failed … The gate was asked of `naive` only." | The roster note keeps 256 for every arm; no life of another length for `replay` or the sweep. |
| 4.12 | Long-life world: the sweep's arm against `naive` | The width of the learner (half width) | E44 brief: half width if it reaches the threshold with the fewest weight-items "in at least two of three seeds under backpropagation". | Every life of E44 is at H = 256, U = 128. (Memo 30.1 states that the half width is the cheaper one "under both rules" in E41; that is the other inventory's scope.) |
| 4.13 | Long-life world: `replay` against `naive` | ρ = 0.5 and a store of one copy of the weights | Roster note: "ρ = 0.5 follows E38's and E43's rehearsal share; it is not tuned, and that is a limit." Audit of the note, section 4.1: "The share ρ = 0.5 is not tuned (the note says so), where the memo's section 17.6 tunes it on worlds set apart." | One value; ρ = 0.25 is named for release 2 only if `replay`'s median first-visit level in the trial is below 0.20. |
| 4.14 | Long-life world: decay 0 against 0.25 | Run under B only | E44 report: the decay-0 set is `naive`. Roster note: release 1's four arms are "under backpropagation"; the sweep's decay-0 arms are named for release 2 "if decay is read as mattering". | No receipt of a sweep life without decay. |
| 4.15 | Small world: `naive` against `sweep` at 64 to 192 units | Learning rate 0.001 and decay 0.25, the defaults taken from E26's network of 512 and 256 | E39 brief, "Optimizer": "defaults 0.001 and 0.25"; the ladder changes only the visit length and the size. | E39 report, "What this does not show": "whether more updates per visit, another rate or decay, or a wider network would change that was not tried beyond the ladder"; addendum: "at the default rate and decay, from one draw of initial values per world; nothing else was tried." |
| 4.16 | E50 (in progress): `C16-in` and `C46-one` against `C-dense`; `C-dense` against `F10` | The value of a turned-on entry of the fixed code, and E49's scaling of the kept weights, under the one learning rate | The value is set to the mean squared length of the dense network's learned code at the start (brief). `R/AUDIT_E50_SPARSE_INPUT_CODE_BRIEF_20261117.md`, section 3: one update moves a cell's input by 0.49% of its size in `C16-in`, 0.83% in `C46-one` and 9.2% in `C-dense`; in `C-dense` one update can move a cell's input for a byte by 0.036, "where in `F10` it can move it by between 0.011 and 0.205". | One value each in the brief; no run yet. |
| 4.17 | E47, E49, E50 against E42's `F10` | One seed (73129) | Briefs of E47, E49, E50. | E49 report, "NOT_RUN": seeds 93683 and 116089 not run; E47 trial report: "One seed. Every reading is seed 73129's." |

---

## Part 5. What the briefs, audits and reports themselves say was not shown

"Taken up" names a later experiment of the record that addressed the item. "None" means that I found no run in the record read. Items that exist only as a plan of 17 November (`PLAN.md` in `WT`) are marked "planned only", with the plan's item number; none of those has a receipt.

### 5.1 E39

| Source | Item | Taken up |
|---|---|---|
| Brief, "What the small world cannot show" | The world is smaller in every dimension than E26's; a result is a claim only when repeated on the E26 families at full scale. | E44 replaced the tasks by families in E26's format; no result of the small world was repeated at full scale. |
| same | No finite life excludes overtaking beyond its horizon. | A standing limit. L2 (in progress) carries a horizon in every statement. |
| same | Supply of tasks: 250 groups last about 1,250 visits; a longer life needs a larger task space. | None. E44's world has 52 families to schedule. |
| same | Answer bytes are the only targets; every-byte targets enter later. | None in this scope. |
| Brief, revisions 3 and 4, "What this cannot show" | That a birth is the right remedy "and not, say, another learning rate; only the birth is tried". | The birth inside the world (L1b) was not run. E44 uses learners already born (E41). Another learning rate: none; planned only (item 8, E52, on the dense network). |
| same | Anything about long lives; that is L2. | L2 in E44's world, in progress. |
| same | What the trend of learning measures after a birth starts from a learner that already has two operations and two moduli. | A description; E44's calibration reports a trend over eight families (negative mean slope in every set, E44 report). |
| Report, "What this does not show" (L0, L1) | Nothing at full scale. | E44 (full-scale format). |
| same | Nothing about long lives. | L2, in progress. |
| same | No claim that the linear operations are unlearnable; "whether more updates per visit, another rate or decay, or a wider network would change that was not tried beyond the ladder". | L1a gave one task alone 16,384 updates at 64, 96 and 192 units: none reached the criterion. Another rate or decay: none. |
| same | The rise of the first-visit level with the order of meeting is not shown to be learning to learn. | E44 F1: later families above earlier ones in 2 of 8 worlds (P44.7 false). L2's D7, in progress. |
| same | No independent re-run; no formal checker or seal. | The reviewer's scorer agreed on two check lives (L0 check 7). A recomputation of L1's tables: not found. |
| Report, addendum, "What this does not show" (L1a) | Nothing at full scale. | See above. |
| same | No claim that these tasks cannot be learned alone; "at the default rate and decay, from one draw of initial values per world; nothing else was tried." | None. |
| same | Nothing about births or lives with a birth; L1b is not run. | None in the small world. |
| same | No independent recomputation: check 14's is the reviewer's. | Not found. Memo 25.1 calls L1a's figures "the worker's figures, not recomputed". |
| Audit of revision 1 (`R/AUDIT_E39_SMALL_WORLD_PROTOCOL_20261115.md`), section 12 | Not checked: anything that needs torch (that a target of 257 raises, the cost per visit, the forward and gradient checks); the runner, network and scorer, which did not exist; whether 0.10 and 0.90 are good bounds for the gate; the coordinator's predictions. | The worker's L0 checks 5 and 8 cover the forward, the gradients and the cost. For E44's world the floor was raised to 0.20 (E44 brief, change 2). The predictions: not found. |
| Audit of revision 3 (`R/AUDIT_E39_REVISION3_BIRTH_PHASE_20261116.md`), section 5 | Not checked: the runner for the two streams and the birth; L1's tables (read, not recomputed); that small-modulus tasks cannot be learned alone ("an inference"); the CPU figures. | Check 10 (streams) passes; checks 11, 12 and 15 were not run. L1a: no task of the band 8 to 13 reaches the criterion (P39.18 true). |

### 5.2 E42

| Source | Item | Taken up |
|---|---|---|
| Brief, "What it cannot show" | A saving in computation: the input map and the readout stay dense. | E47 (trial) and E49. The reviewer's readings state that neither saves computation to the threshold (memo 28.4, "No saving"; `R/E49_E43ADD2_VERIFICATION_20261117.md`, summary: "none saves computation"). |
| same | "Settings. Learning rate and decay were chosen for the dense GRU." | None in this scope. Planned only (item 8, E52) on the dense network at two widths. |
| same | The form of a cell: the GRU's equations are kept. | None. |
| same | One task family, answer targets, one arrangement of cells. | The arrangement: `W10` inside E42. A second task: E43's second addendum ran an unrelated task from E42's `F10` and `P8` of seed 73129 (review 9.156; outside this scope). More families learned by a limited network: none. |
| same | Distances longer than an item: the state is reset at every item. | None. |
| same | The sign rule on few entries: the scale and the masked weight gradient are choices. | None. |
| Brief, "Choices left to the reviewer and to me" | N counts connections between cells only; the GRU's equations; exactly N where the user's limit is "at most N"; half the cells at period 1; the scale of the kept weights and of the sign rule; replacement by a threshold tuned to 5% per event. | N: 5 and 20 inside E42 (S only). The whole byte read by every cell: E49, E50. The others: none. |
| Brief, "Configurations" | `P8-mean` is deferred. | None. |
| Memo 22.10 | "Whether many more cells can be afforded at the same computation needs the input and the readout limited too, which is a later job." | E47 (more cells at a boundary), E49 (a limited interface at 256 cells). More cells with a limited interface: none. |
| Audit (`R/AUDIT_E42_CELLS_BRIEF_20261116.md`), section 11 | Not checked: anything that needs torch; `MaskedSign`'s backward and the clearing of the optimizer's averages (read, not executed); the start with the network's real initial values; the cost after a speed-up; whether Python 3.11 and 3.13 give the same draws. | The worker's stage 0 (checks 3, 6, 7, 9) covers each on the worker's host (`E42_STAGE0_73129.md`). |
| Reviewer's verification of seed 73129, sections 7 and 8; of three seeds, sections 7 and 11 | "E42 therefore still does not say whether changing connections help, on any seed." Not verified: training, gradients, the faster forward against the specification, the cosines, CPU figures; the resumed runs beyond their data; rows before the last; dry-run answers for the two later seeds. | A corrected rule for replacing connections: none (memo 30.4 lists it as resting). |
| Worker's reports | The two reports have no section on limits; `E42_REPORT_73129.md` says only that one seed leaves the description cells empty. | – |

### 5.3 E44 (calibration) and the roster note of the first long lives

| Source | Item | Taken up |
|---|---|---|
| Brief, "What this cannot show" | A life of more than 52 families. | None. |
| same | Whether a family of 168 to 264 taught pairs can be learned by a born network. | The calibration's own table: the family's figure is 0.378 for families of 264 or fewer taught pairs under `naive` at 256 updates (E44 report). The named step (raising the lowest modulus) was not called. |
| same | Whether the gate's level reflects learning the modulus or the code. | None. |
| same | Anything at the present width. | None. The roster note names it for release 2. |
| same | A result here is a claim only when repeated on E26's own lives. | None. |
| Brief, "Not in this brief" | A learner born inside the world on one of its own families. | None. |
| Brief, named further steps | 32 updates a visit; a taught share of 0.05; 512 updates a visit. | Not called, because the gate passed at 256 (E44 report). |
| Report, "What this does not show" | Lives of 8 visits only; no absence beyond 7 visits. | L2, in progress. |
| same | One birth seed (73129) in every world. | L2's two 8-visit lives with the other births (medians 0.573 and 0.313); the long lives use three seeds. |
| same | The two further sets (`sweep-after-naive`; `naive` with decay 0) are descriptions, one run per world. | Decay 0 under B: L2's `wd0` arms, in progress. `sweep-after-naive` in a long life: none. |
| same | Same-author scoring; nothing else recomputed independently. | The reviewer's audit of the roster note reproduces the gate's medians from all 917,504 answers (memo 28.8). |
| Audit of the brief (`R/AUDIT_E44_LONG_LIFE_WORLD_BRIEF_20261116.md`), section 11 | Not checked: anything that needs torch; whether a born network learns a tenth of a family; the CPU rates; the compiled file; the digests under Python 3.11; E41 for seeds 93683 and 116089. | The calibration (the gate passes at 256; measured CPU); the two later births are on `D` and were checked in the audit of the note (section 3). |
| Roster note, revision 2, "What it cannot show" | Anything about the sweep, until release 2. | Brought forward: the sweep's long lives (plan item 6), started 21:10 UTC by `LOG.md`; no receipt read. |
| same | A tuned replay share or buffer, or the best decay; one value each. | None. |
| same | Lives of more than 52 families; anything at full scale. | None. |
| same | Composition, which this world does not teach. | None. |
| same | Decay alone beside what is kept: release 1 does not compute it. | None; the note says the readable form "would need its own check". |
| Audit of the note (`R/AUDIT_E44_L2_ROSTER_NOTE_20261117.md`), section 11 | Not checked: the changed runner; anything that needs torch; the note's statements about E43, E38 and E46; the decay-0 and `sweep-after-naive` sets of the calibration (48 lives); the driver `lw_f1.py`; the streams of the 32 lives. | The runner change: its checks pass (`E44_L2_CHECKS.json`), and `LOG.md` records that the reviewer read the diff at 21:09. The 48 lives of the two further sets: not found. |

### 5.4 E47

| Source | Item | Taken up |
|---|---|---|
| Brief, "What it cannot show" | A benefit of more cells. | None. The run at 4,096 cells was not released (memo 30.2). |
| same | Many families and keeping them apart. | None. Planned only (item 12, E56). |
| same | Locality inside the interface. | E49 (run), E50 (in progress). |
| same | One scale for all cells; a scale from a cell's own 11 weights is a later variant. | None. |
| same | Measured time; the comparison is in counted computation. | A standing limit; the trial reports CPU-seconds beside. |
| same | "Settings. Learning rate, decay and N are E42's." | None. |
| same | The form of a cell. | None. |
| Brief, "No extra steps before the answer" | The variant with a stated number of steps between `=` and the answer. | None. |
| Brief, "Stage 1: the roster" | The arm `scatter` at 1,024 cells; the arm `one` at 4,096 cells; the two `place` runs at 4,096 cells (dropped at the audit). | None; "Not released" (memo 30.2). |
| Audit (`R/AUDIT_E47_LOCAL_INTERFACE_BRIEF_20261116.md`), section 13 | Not checked: anything that needs torch; the starting draws of the new cells; the CPU figures; the runner; E26's figure of 3,886 to 6,899 updates; seeds 93683 and 116089 beyond the draws. | The trial saves its starting weights, and the reviewer's reader reproduces all 7,200 recorded answers (`R/E47_TRIAL_VERIFICATION_20261117.md`, summary 1). The other seeds: none. |
| Trial report, "What this trial does not show" | One seed. Measured time. Answers past four bytes under the far-mean diagnostic. | None. |
| Reviewer's verification, section 9 | Not verified: anything that needs torch; the processor times; the other two seeds; the count of multiply-adds for `P8`; why the slow far cells lose their weights; why rings 2 and 3 are not used ("an explanation, not a finding"). | None. Planned only (items 1 and 9: the passage of differences at the starting weights; E53). |
| Reviewer's verification, section 8 | "Periods belong at the interface": an arm whose interface cells have `P8`'s periods is not on file. | None. |

### 5.5 E49

| Source | Item | Taken up |
|---|---|---|
| Brief, "What it cannot show" | A cell that cannot know the byte. | E50, in progress. |
| same | More than one seed. | None. |
| same | An input with geometry. | None. |
| same | The byte's code and the readout's own layers (`l1`, `l2`, `l3`) stay dense. | E50 replaces the byte table and `l1` by a fixed code; `l2` and `l3` stay dense: none. |
| same | More cells. | E47 varies the number of cells with a dense interface on its patch; with a limited interface: none. |
| same | "Settings. Learning rate and decay are E42's; the scale of the kept weights at the start is a choice." | None. |
| Report, "NOT_RUN and open items" | Seeds 93683 and 116089; continuing a run beyond the dose (`I64` and `out8` under B still rising); the reviewer's reader on the weights; `F10` under B on seed 116089. | The reader: done (`R/E49_E43ADD2_VERIFICATION_20261117.md`). `F10` under B on 116089: on `B42` (9,216 updates to 80 on A). Continuation: "None" (memo 30.2). The seeds: none. |
| Report, "What I did not check" | Why the limited settings under the sweep stay low; whether the start's scale matters ("one scale was run"); the other seeds' draws. | The reviewer measured the end states and states "The cause is not established" (verification, section 2.7). E50's stage 2 under S is specified as a test of one explanation; not run yet. The start's scale: none. |
| Audit (`R/AUDIT_E49_LIMITED_INTERFACE_BRIEF_20261116.md`), section 11 | Not checked: the specification file was not executed; the sweep through the masked maps (read, not computed); the start (measured on stand-in weights); the code of a network trained under the masks; bit-equality with E42's runner over 2,048 updates; `F10` on the other seeds. | Check 2 of E49's stage 0 (bit-equal, E49 report); the learned codes were measured in the audit of E50, section 2.3. |
| Reviewer's verification, section 5 | Not verified: scores at rows before the last (only the final row is reproduced from weights); anything that needs gradients or training; seeds. | None. |

### 5.6 E50

| Source | Item | Taken up |
|---|---|---|
| Brief, revision 2, "What it cannot show" | More than one seed. A learned sparse code. The readout's direct reading of the byte through `l2`. An input of many tokens. More cells, and settings, as in E49. A cell that stays ignorant of a byte (its ignorance lasts one position). A limited readout. The same difference between the two rules as in E49 (the code has no gradient). Why the fixed code costs, if it does (`C-dense` differs from `F10` in four things). Letters (286 of the 512 values belong to the 26 letters). | None; the job is in progress. A limited readout on the learned code is E49's `out8`. |
| Audit (`R/AUDIT_E50_SPARSE_INPUT_CODE_BRIEF_20261117.md`), section 10 | Recommendations not taken into the job: a setting with 64 values a cell and the dense readout (and an `in64` on E49's runner beside it); a draw that gives each cell all the values of a stated number of bytes; a single code shared by the 26 letters; a network in which a byte reaches most cells only after several positions. | None (brief revision 2: recommendation a "is not run"). |
| Audit, section 12 | Not checked: the specification file was not executed; the starting network (taken from two committed files); E49's checks 1 and 2; E47's trial weights; "Why the sweep does not learn with a limited input is not established"; no runner of E50 existed; the chances of the predictions. | The runner exists and its five checks pass (`E50_STAGE0_73129.md`), which also states: "the reviewer's reader has not yet been applied to the dry-run files". |

---

## Part 6. Things that could be tested and have not been

Each line is one thing that the record names, or that follows from a single-valued setting of Part 2, and for which I found no run. The source is in brackets. No order of importance is implied.

**The optimizer and the two rules**

1. A learning rate other than 0.001 has not been run for any network of this scope, under either rule (Part 2.1; row 4.1; E42 brief, "Settings").
2. The sweep has not been run under a learning rate or a decay of its own, different from back-propagation's (row 4.2; `PLAN.md`, item 8).
3. Betas other than (0.9, 0.98), and an epsilon other than the library's default, have not been run (Part 2.1).
4. A decay other than 0.25 has not been run for any limited or local network of E42, E47, E49 or E50 (Part 2.2; E47 brief, "Settings").
5. A life without decay has not been run under the sweep in the long-life world (row 4.14).
6. A choice of which parameters are exempt from decay other than the byte table has not been run (Part 2.1).
7. An optimizer other than AdamW has not been run in this scope (Part 2.1).
8. A batch other than 64 items, and in the worlds a split other than 48 taught and 16 asked, has not been run (Part 2.1).
9. A scale of the sweep other than the mean magnitude of a whole matrix, for example one from a cell's own weights, has not been run (row 4.5; E47 brief, "One scale for all cells").
10. Targets other than the answer bytes have not been run in this scope (E39 brief, "What the small world cannot show").
11. Credit cut in time, and a state carried from one item to the next, have not been run in this scope (Part 2.1; E42 brief, "Distances longer than an item").

**The start**

12. A starting size of the kept weights other than the one that keeps the dense network's variance has not been run in E42, E49 or E50 (row 4.3; E49 report, "What I did not check").
13. A start of the local network at which differences between items reach the cells two or more connections from the boundary has not been run (row 4.4; E47 verification, section 6).
14. A second draw of the starting weights for the same world has not been run in the small world's solo runs (E39 report, addendum, "What this does not show").

**The structure of the network**

15. A cell with equations other than the GRU's has not been run (E42 brief, "The form of a cell").
16. The number of connections, the wide neighbourhood, the period sets and the variants of E42's stage 2 have not been run under back-propagation (row 4.8).
17. The variant `mean` of E42 has not been run (E42 brief, "`P8-mean` is deferred").
18. A replacement rule in which a new connection is judged against its own age has not been run, so that changing connections remain untested (E42 verification, section 7; memo 22.12).
19. A share of cells at period 1 other than one half has not been run (E42 brief, choice 4).
20. A limit of "at most N" connections in place of exactly N has not been run (E42 brief, choice 3).
21. A neighbourhood of another radius or another grid has not been run; only 48 cells and all cells were compared, under the sweep (Part 2.2).
22. The limited or periodic networks have not been run with a quarter or a tenth of the pairs taught (row 4.9; `PLAN.md`, item 10).
23. The limited networks have not been run at the half width or at any width other than 512 (row 4.10).
24. The limited or local networks have not learned more than one family; retention was measured only for E42's `F10` and `P8` of one seed, in E43's addendum (E47 brief, "Many families"; review 9.156).
25. E47's network at 4,096 cells, its arm `scatter`, and its variant with extra steps before the answer have not been run (E47 brief, "Stage 1"; memo 30.2).
26. An arm of E47's wiring whose interface cells have the periods of E42's `P8` has not been run (E47 verification, section 8).
27. A local network run without decay, in which far cells' weights that receive no gradient would not shrink, has not been run (row 4.4).
28. E47, E49 and E50 have not been run on seeds 93683 and 116089 (row 4.17).
29. A limited interface with more than 256 cells has not been run (E49 brief, "More cells").
30. An input whose values have positions, so that a cell's subset is a neighbourhood, has not been run (E49 brief, "An input with geometry").
31. A limit on the readout's own layers `l2` and `l3` has not been run (E49 brief; E50 brief, "The readout's direct reading of the byte").
32. The runs of E49 that were still rising at the end of the dose have not been continued (E49 report, "NOT_RUN and open items").
33. The cause of the sweep's failure with a limited input has not been isolated by a run; E50's stage 2 under the sweep is specified as a test of one explanation and has no receipt (E49 verification, section 2.7; E50 brief).
34. For E50's code: a learned sparse code, a limited readout on the fixed code, a setting with 64 values a cell and a dense readout, a draw that gives each cell all the values of a stated number of bytes, one shared code for the letters, and an input of many tokens have not been run (E50 brief, "What it cannot show"; audit of E50, section 10).

**The dose and the fraction taught**

35. A dose longer than 24,576 updates has not been run for E42's or E47's networks (row 4.6).
36. In the long-life world a taught share other than 0.10 and a lowest modulus other than 41 have not been run (E44 brief, named further steps).

**The worlds**

37. No life of more than 8 visits has a receipt in E44's world on the refs read; the lives of 256 and 1,024 visits are in progress (1.4).
38. A share of rehearsal other than 0.5 and a store of another size have not been run (row 4.13; roster note, "What it cannot show").
39. A visit length calibrated for the `replay` arm or for the sweep, in place of the one calibrated under `naive`, has not been run (row 4.11).
40. The long-life learner at the present width has not been run (E44 brief, "What this cannot show").
41. A learner born inside the long-life world on one of its own families has not been run (E44 brief, "Not in this brief").
42. The arm `sweep-after-naive` has not been run in a life of more than 8 visits (E44 report, "The two further sets are descriptions").
43. A life of more than 52 families has not been run; it needs a larger task space (E44 brief; roster note).
44. Whether the first-visit level in E44's world reflects learning the modulus or the code has not been separated by a run (E44 brief, "What this cannot show").
45. No result of E44's world has been repeated on E26's own lives (E44 brief).
46. Composition is not taught in the long-life world (roster note, "What it cannot show").
47. Decay alone beside what is kept has not been computed in a life with returns (roster note; audit of the note, section 1.3 c as the note cites it).
48. A fresh optimizer at each visit, or an optimizer state carried over from the birth, has not been run; every life starts one fresh optimizer and keeps it (Part 2.1).
49. In the small world, the life with a birth (L1b) and the long lives (L2) were never run, and the linear tasks alone were not tried at another learning rate or decay (E39 report, addendum).
50. The reviewer's recomputation of L1a's updates to criterion from the committed event files (check 14) is not on file (E39 report, addendum, check 14; memo 25.1).

---

## Annex. What I could not find in the record

- The number of AdamW's epsilon: no file read passes one and no receipt records it.
- A reason in the record for the betas (0.9, 0.98) beyond the code comment "as in the grokking papers"; for the dose of 24,576 updates; for the taught fraction of 0.5; for the 8 numbers of a byte's row; for exempting the byte table from decay; for half the cells at period 1; for the split of 48 taught and 16 asked items.
- A section of the coordinator's review that reads the results of E42, of E47's trial or of E49: the review on `D` ends at 9.157, which says these are "finished and not yet read in full by me". The lines "Found" of 1.2, 1.5 and 1.6 are therefore the workers'.
- Any result of E50 (two receipts, incomplete at update 1,024) and any receipt of a long life of E44 (the branch holds checks and two 8-visit lives).
- Any file of the sweep's long lives (plan item 6): they are recorded in `LOG.md` only.
- A recomputation of E39's L1 tables or of L1a's figures by a second implementation.
- A run, anywhere in this scope, with a learning rate other than 0.001.
