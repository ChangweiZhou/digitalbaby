# Inventory of settings in the experiments on the dense network: E26, E31, E33, E34, E35 (A and B), E36, E37, E38, E40, E41, E43 with its addenda, E46, E48 (17 November 2026)

This is item 3 of `PLAN.md` beside this file, for the experiments on the dense network. It was made from the record only: every entry names the file it was read from. Where the record does not say, the entry reads "not found". Nothing was run except `git show`, `git ls-tree` and a standard-library reader of JSON receipts. The experiments E39, E42, E44, E47, E49 and E50 belong to another inventory and appear here only where an experiment of this scope used their files.

## 0. Refs, names and what "confirmed" means

| Name | Meaning |
|---|---|
| D | `origin/codex/minifly-consolidated-20261023` at 7cabb13a, the default branch |
| M | the folder `MOD97_LOCAL_RULE_20261106` |
| W | `origin/claude/wonderful-edison-8hx6i6` at b76f5979: E41's three-seed report and all of E46. It is not an ancestor of D, and D holds no file of E46 |
| T | `origin/claude/sweet-turing-lwi3no` at 2d9d1726: all of E48. It is not an ancestor of D, and D holds no file of E48 |
| P | `origin/claude/peaceful-hopper-8ofq4i` at 58ed6b25: E38 and E43. It is merged into D, and D's copies were read |
| V | the working tree of `claude/vibrant-cerf-sje726`, folder `PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/`, as it stood on 17 November 2026 |
| review 9.x | section 9.x of D:`MINIFLY_PAST_WORK_REVIEW_FOR_MR_ZHOU_20261106.md` |
| memo | V:`MEMO_TO_COORDINATOR_LEARNED_STRUCTURE_20261115.md` (revision 21) |
| brief En | D:`OPUS_TASK_En_*.md` at the root of D |
| rule S, rule B | the sweep (`local=True`) and exact back-propagation (`local=False`) |
| rule K | the sweep with a separate feedback matrix on the recurrent map, kept near the forward matrix (Kolen–Pollack feedback; E31) |

"Confirmed" in a table of Part 1 means that the value named in the brief was also read in the code or in one receipt, which is named. A receipt was read for every experiment; the optimizer's groups in the receipts are the ones the runner read back from the optimizer.

## Part 1. The values actually run

### 1.0 The standing values (the set that E26 fixed and that later experiments name "as E26")

Each later table says "standing" where the value is the one in this table, and gives the value where it differs.

| Setting | Value | Read in | Confirmed in |
|---|---|---|---|
| Optimizer | AdamW, built by `exp16_integrated.adamw16(m, LR, WD, ())` through `exp26_candidate.make_opt`, which asserts the two groups | D:M/`exp26_candidate.py` lines 39 to 40 and 72 to 76; D:M/`exp16_integrated.py` lines 176 to 187 | D:M/`exp45/e26/e26_dev_S_73129.json`, fields `optimizer` and `optimizer_read_back` |
| Learning rate | 0.001 (`LR` is taken from D:M/`exp20_reference.py` line 53) | the same | the same receipt: `lr 0.001` in both groups |
| Betas | (0.9, 0.98), written inside `adamw16` (line 187) | the same | the same receipt |
| Epsilon | No runner sets it, so it is the library's own value. The E34 brief names it as 1e-8 ("AdamW's `eps` of 1e-8"). It is in no receipt | brief E34, section "The rule, as code" | not in any receipt |
| Weight decay | 0.25 on every parameter but the byte table (`WD = 0.25`, `exp26_candidate.py` line 40) | the same | the same receipt: group 0 holds 12 tensors (`l1`, the GRU's two weights and two biases, `lm`, `l2`, `l3`) at 0.25 |
| Exempt from decay | the byte table `book.weight` alone, in a group of its own at 0.0 (`E16_BOOKDECAY=off`, set on import by `exp20_reference.py` lines 24 and 28) | `exp16_integrated.py` lines 181 to 184 | the same receipt: group 1 is `book.weight` at 0.0 |
| State of the optimizer | a fresh optimizer at the start of every birth, life and continuation; no schedule of the rate, no clipping (the E26 brief, revision 4: "there is no clipping or reset in this loop") | `exp26_candidate.py` lines 84 to 91, 129, 205 | receipts record the optimizer's count of steps, which starts at zero in each run (for example `optimizer_steps [36864]` at the end of a life of 36,864 updates) |
| Batch | 64 items | D:M/`exp45.py` line 30 | receipt field `batch: 64` |
| Network | `arch20.NetRNN("gru")`: byte table of 258 rows by 8 (row 257, the start symbol, is the padding row); `l1` from 8 to 512 with tanh; one GRU layer of 256 cells reading the 512 values; readout `z2 = tanh(l2 z1 + lm h)`, `l3` from 512 to 257 outputs; 1,124,113 parameters | D:M/`arch20.py` lines 32 and 36 to 50; D:M/`exp11_cell.py` line 28 (width 512) | receipt field `model.params: 1124113` |
| Cell equations | `r = s(Wir x + Whr h)`, `z = s(Wiz x + Whz h)`, `n = tanh(Win x + r * (Whn h))`, `h' = (1 - z) n + z h`, with `h = 0` at the start of every item | D:M/`arch26.py` lines 9 and 50 to 75 | receipt field `model.forward` names `arch26.forward` |
| Starting weights | drawn by the library's constructors (`nn.Embedding`, `nn.Linear`, `nn.GRU`) after `T.manual_seed(seed)`. No routine sets or rescales any weight. The files read do not state the distributions | `arch20.py` lines 62 to 69 | not in any receipt |
| Rule S | at every linear map (`l1`, the stacked input map, the stacked recurrent map at every step, `l2`, `lm`, `l3`) the error goes back through `sign(W)` times the mean of the absolute weights of the whole matrix; the gates' own nonlinearities and products are differentiated exactly | D:M/`exp3_state.py` lines 91 to 108; `arch26.py` lines 14 to 21 | receipt field `local: true` |
| Error through earlier positions | passed back through all earlier positions: every runner of the scope calls the forward with `through_time=True` (`arch26.py` line 60 detaches the state only when it is `False`) | `exp26_candidate.py` line 86 | receipt field `update`: "credit through the whole item" |
| How a byte is coded | one byte a step, no window; the byte's row of the learned table (8 numbers, no decay), then `l1` (8 to 512) and tanh | `arch20.py` lines 73 to 74; `arch26.py` lines 52 to 53 | — |
| Item | `code a1 a2 op d_1..d_k b1 b2 =` then the answer `c1 c2` and the end symbol; operands and answers are two digits; `d` are upper-case letters | D:M/`exp11_cell.py` lines 228 to 232 | — |
| Letters in training | k drawn uniformly from 0 to 12 for every item | `exp11_cell.py` lines 54 and 244 to 253; `exp16_integrated.py` line 575 | receipt field `kmax: 12` |
| Targets | the answer bytes only (the answer mask of `exp45.batch_tensor`), mean cross entropy | `exp26_candidate.py` lines 84 to 91; `exp45.py` lines 158 to 169 | receipt field `update` |
| Answering and scoring | the prompt is kept and the network is rerun for each output byte, taking the largest output; a whole answer counts only with its end symbol | `exp11_cell.py` lines 188 to 222 | — |
| Seeds | 73129, 93683, 116089 | D:M/`mod97_local.py` line 37 | receipts |
| Host | Linux x86-64, Python 3.13.18, torch 2.15.0+cpu, one thread a process | every brief from E31 on | receipts' `provenance.host` |

### 1.1 E26, with its crossover E26-X

Brief: D:`OPUS_TASK_E26_GRU_UNDER_THE_SWEEP_20261111.md`. Code: D:M/`exp26_candidate.py`, `arch26.py`, `exp26x_cross.py`. Receipts read: D:M/`exp45/e26/e26_dev_S_73129.json`, `e26_dev_B_73129.json`, `e26_life_S_73129.json`. Worker's reports: D:M/`exp45/e26/E26_BIRTHS.md`, `E26_LIVES.md`, `exp45/e26x/E26X_DESIGN.md`, `E26X_RESULTS.md`. The files `E26_REPORT.md` and `E26_REVIEW_SECTION.md` that the brief asks for were not found on D or on `origin/claude/adoring-cori-b07aog`.

| Setting | Value run |
|---|---|
| Rules | S and B, six births and six lives. E26-X: six more lives in which the birth is of one rule and the life of the other (SB, BS) |
| Optimizer | standing. Confirmed in both birth receipts (decay 0.25 under S and under B) and in the life receipt (`complete.optimizer_read_back`) |
| Batch; network; start; error through time; byte code | standing. Life receipt: `batch 64`, `model.params 1124113` |
| Tasks, families, moduli | Birth: add, sub, max, min under modulus 89 (codes `p q v w`). Life: F1, the four operations under modulus 97 (codes `r t x n`); F2, two-step compositions under modulus 97 (code `c`) on 12 taught operator pairs; F3, the four operations under modulus 73 (codes `s u y z`). Schedule F1, F1, F2, F2, F3, F3 (`exp16_integrated.py` lines 546 to 547 and 855; life receipt `schedule`) |
| Taught fraction | Birth: 0.5 of the pairs of each task (`exp26_candidate.py` line 131). F1 and F3: 0.25 (`exp16_integrated.py` line 596). F2: 2,000 operand triples for each of the 12 taught operator pairs, 24,000 entries; four operator pairs are never taught (life receipt `entries`) |
| Letters | Training: 0 to 12. Compositions have no letters (`E16_COMPGAP=0`). Tests: birth panel at 0, 4, 8, 12 and 20 letters, 300 held-out pairs a task; life panels of F1 and F3 at 0, 8, 12, 20 and 40 letters, 400 pairs; the birth family at 12 letters, 300 pairs |
| Dose | Birth 24,576 updates; life six blocks of 6,144, 36,864 in all (life receipt `optimizer_steps [36864]`) |
| Seeds | the three |
| Rehearsal | In a life each item is drawn from one of the families admitted so far, each with equal chance (`exp16_integrated.py` lines 608 to 612). Items of earlier families are therefore none of the stream in blocks 1 and 2, about half in blocks 3 and 4 and about two thirds in blocks 5 and 6 (life receipt: 196,358 of F2 and 196,858 of F1 in block 3; 131,079, 130,903 and 131,234 in block 5). The birth family is never in the stream. No store (`store: false`). Every update |
| Scores and thresholds | Birth: the mean of add and sub under 89 at 12 letters, at six checkpoints; marks 70, and 0.85 of the partner. Life: new instances of taught operator pairs (600 items), mark 85; untaught plus-minus pairs (1,200 items), mark 40; "holds" means two of three seeds. E26-X: an effect is present if it is at least 5 points in two of three seeds and in the mean |
| Found | Review 9.112 E: "born level with backprop; composes from memory at 37 to 54% against 63 to 89%; the registered marks missed; the gap is in the update during the life and not in the birth." |

### 1.2 E31

Brief: D:`OPUS_TASK_E31_ONE_SWITCH_LIVES_20261113.md` with three addenda. Code: D:M/`exp31_switch.py`, `arch31.py` (worker A), `exp31_switch_workerB.py`, `arch31_workerB.py` (worker B). Receipts read: D:M/`exp45/e31/e31_life_K_73129.json`, `e31_life_G_73129.json`. Report: D:M/`exp45/e31/E31_REPORT.md`.

| Setting | Value run |
|---|---|
| Rules | Rule S everywhere with one change on the way back. `G`: exact weights for the reset and update gate rows of the two GRU matrices. `H`: exact weights for the whole recurrent matrix. `K`: the error crosses the recurrent map through a separate matrix `Bhh` (768 by 256), drawn once from U(−1/16, 1/16) by `torch.Generator().manual_seed(seed + 31)`, and after each step `Bhh ← (1 − 0.00025) Bhh + (W1 − (1 − 0.00025) W0)`. `P2a`, `P2b`, `P2c`: rule S plus a term on the update-gate rows, `λ · sum(relu(0.4 − abs(z − 0.5))²)` divided by the number of masked positions times 256, with λ set once so that the term's gradient norm on those rows is 0.25, 0.05 or 0.01 of the task's on the first batch. The references are on file: E26's life S (floor) and E26-X's life SB (ceiling). Arm `P` was withdrawn before any full life; arm `Ks` has a dry run only |
| Optimizer | standing, fresh at the start of the life. Confirmed in both receipts (0.001, 0.25, (0.9, 0.98), byte table 0.0) |
| Batch; network; start; byte code | standing. Every life starts from E26's sweep birth `e26_dev_S_<seed>.pt` |
| Error through time | through the whole item (`arch31.py` line 66: "arch26.forward(m, x, local=True, through_time=True) with the switch") |
| Tasks, fractions, letters, rehearsal | E26's life, unchanged (receipt `schedule`, `store: false`) |
| Dose | 36,864 updates; eighteen lives (six arms, three seeds) |
| Scores and thresholds | Untaught plus-minus pairs at the end of the life. An arm "closes at least half the floor-to-ceiling gap in two of three seeds". K is read against H only. Differences under 15 points between single lives are not read |
| Found | Review 9.128 A: "the recurrent pathway as a whole is where the sweep loses composition, not the gates." and "Kolen–Pollack feedback on the recurrent map is a candidate: a rule inside the line that recovers most of the composition gap on three seeds. It is not replicated on fresh seeds". Review 9.130: "The gate push did not help composition at any strength" |

### 1.3 E33

Brief: D:`OPUS_TASK_E33_WHICH_TARGETS_BREAK_EXTRAPOLATION_20261115.md` (revision 2, addendum 1). Code: D:M/`exp33_targets.py`, importing `masks` and the roles from D:M/`exp28_everybyte.py`. Receipt read: D:M/`exp45/e33/e33_dev_A+D_73129.json`. Report: D:M/`exp45/e33/E33_REPORT.md`.

| Setting | Value run |
|---|---|
| Rules | S only (`local = True`, `exp33_targets.py` line 134) |
| Optimizer; batch; network; start; byte code | standing. Receipt: 0.001, 0.25, (0.9, 0.98), byte table 0.0, `batch 64` |
| Error through time | through the whole item (`arch26.forward(m, x, True, True)`, line 91; receipt `update`) |
| Tasks, fraction | E26's birth data: the four operations under 89, half of the pairs taught. The report states that the final data digest of every arm is eb49a8e4…, which is the digest in E26's two birth receipts of seed 73129 |
| Targets | The answer bytes carry half of the loss. Of the other bytes, the selected classes are targets, each position at the weight it has in E28's balanced loss, `0.5 / other.sum()`; unselected positions are dropped and the rest are not scaled up. Classes: D, the letters; O, the four operand digits; S, the task code, the operator and `=`. Eight arms on seed 73129 (`A`, `A+D`, `A+O`, `A+S`, `A+DO`, `A+DS`, `A+OS`, `A+DOS`); `A+D` and `A+OS` on seeds 93683 and 116089 |
| Letters | Training 0 to 12. Tests at 12, 20 and 40 letters on 300 held-out pairs of addition under 89 (D:M/`e32_read.py` lines 74 to 79) |
| Dose | 24,576 updates; rows saved every 4,096 |
| Seeds | 73129 for the eight arms; all three for two arms |
| Rehearsal | none (a birth) |
| Scores and thresholds | Mean of the last three rows. Learned in range: at least 70 at 12 letters. Extrapolates: at least 60 at 40 letters. Fails: at most 20. On a new seed the contrast replicates if `A+OS` extrapolates and `A+D` does not and is at least 40 points below it |
| Found | Review 9.134: "on three seeds, a sweep birth with answer and distractor targets does not extrapolate to 40 letters (23.8, 29.1, 31.6) and one with answer, operand and task-code targets does (94.2, 94.9, 86.9)." |

### 1.4 E34

Brief: D:`OPUS_TASK_E34_ENTROPY_WEIGHTED_ERROR_20261115.md` (revision 2, addenda 1 and 2). Code: D:M/`exp34_entropy.py`; the plain long birth runs D:M/`exp28_everybyte.py`. Receipts read: D:M/`exp45/e34/e34_dev_T050_73129.json`, `e34_dev_B050_73129.json`, `exp45/e34_long/e34_dev_T050_dry98304_73129.json`. Report: D:M/`exp45/e34/E34_REPORT.md`.

| Setting | Value run |
|---|---|
| Rules | S in arms `T050`, `T100`, `T200` (and `U`, a dry run). B in arm `B050` (and `BU`, a dry run) and in the plain long birth `B-all` |
| Optimizer; batch; network; start; byte code | standing. Confirmed in all three receipts, the 98,304-update one included |
| Error through time | through the whole item (`arch26.forward(m, x, local, True)`, line 91) |
| Tasks, fraction | E26's birth data, as E33 |
| Targets | Every real byte. The error at each position is multiplied by `w = exp(−H / tau)`, where H is the entropy of the network's own prediction there; no gradient passes through `w`; the loss is divided by the sum of `w` over the batch. tau is 0.5, 1 and 2 (infinite in the dry run). Neither the answer mask nor a label of the byte's role enters the loss |
| Letters | Training 0 to 12. Tests at 12, 20 and 40 letters, as E33 |
| Dose | 24,576 updates; 98,304 for the long births of `T050`, `B-all` and `B050` (receipt `dev_updates 98304`) |
| Seeds | 73129 for all arms; `T050` also on 93683 and 116089; `B050` and the long births on 73129 only |
| Rehearsal | none (births) |
| Scores and thresholds | E33's marks. Long births: at 40 letters over the 13 rows from 49,152 to 98,304, "overtakes and stays" if the mean is at least 10 points above and at least 10 of 13 rows are above |
| Found | Review 9.146: "The passing temperature learns in range and extrapolates on one of the two new seeds and does not learn in range on the other (69.9 at 12 letters), so the three-seed sentence is not licensed. Backpropagation with the same gain does not learn in range. Over four times the dose, the reading is *mixed*" |

### 1.5 E35-A and E35-B

Briefs: D:`OPUS_TASK_E35A_INNATE_CIRCUITS_INVENTORY_JIA_20261115.md` and D:`OPUS_TASK_E35B_INNATE_CIRCUITS_INVENTORY_YI_20261115.md` (revision 2, addendum 1 each). Code: D:M/`exp35a_parsed.py`, `exp35a_stream.py`, `exp35b_parsed.py`, `exp35b_stream.py`. Receipts read: D:M/`exp45/e35a/e35a_dev_M_f10_73129.json`, `exp45/e35b/e35b_dev_F_f10_93683.json`. Reports: D:M/`E35A_REPORT.md`, `E35B_REPORT.md`.

| Setting | Value run |
|---|---|
| Rules | B only, in every run ("backpropagation is allowed here and nothing in this brief is a candidate rule") |
| Optimizer, stream stage | standing values, built by `adamw16(m, LR, WD, ())`. The parameters of the added module are in the decayed group at 0.25 (receipts: 22 tensors in group 0 for inventory 甲, 26 for 乙; byte table at 0.0) |
| Optimizer, stage 0 (operands handed over) | E35-A: `T.optim.AdamW(m.parameters(), lr=.003, weight_decay=.01)`, with no betas given and nothing exempt (`exp35a_parsed.py` line 140). E35-B: that optimizer, and also `AdamW(lr=.001, weight_decay=.25, betas=(.9, .98))` on every parameter (`exp35b_parsed.py` lines 174 to 177) |
| Batch | 64 in both stages |
| Network, stream stage | Arm `G`: the standing network. Arm `M`: the same plus a fixed module: two registers written through gates computed from the GRU's state, four fixed operations weighted by a softmax, and a gated read-out added to the logits. Inventory 甲: codes are distributions over 89 positions and the four operations are wired for modulus 89. Inventory 乙: a code is eight unit phasors and a scalar; the operations are phase addition, phase subtraction and a comparator. Arm `X`: the operations replaced by fixed random tables or maps. Arm `F`: arm `M` with the direct read-out's logits for the ten digit bytes removed. The new parameters are created after `arch26.new_net(seed)` under `T.manual_seed(seed + 35)` "with torch's defaults" |
| Network, stage 0 | A small network on parsed input (E10's `RingNet` at modulus 89 for 甲; for 乙 three forms of the code: angles, complex numbers, digit tables) |
| Error through time | through the whole item (`forward(m, x, False, True)`, `exp35a_stream.py` line 166; `exp35b_stream.py` line 74 asserts it) |
| Byte code | standing in the stream stage; in stage 0 the operands are given as digit pairs and the operator as its byte |
| Tasks | the four operations under modulus 89 |
| Taught fraction | 0.10 and 0.25 (0.5 only in the dry check against E26). Stream births were run at 0.10 for every arm and at 0.25 for arm `G` only, since no family had room at 0.25 |
| Letters | Training 0 to 12 (none in stage 0). Tests: held-out panel at 0, 4, 8, 12, 20 and 40 letters, 300 pairs a task; taught panel at 12 |
| Dose | 24,576 updates in the stream stage; 32,768 in stage 0 |
| Seeds | Stage 0: the three (12 runs for 甲, 48 for 乙). Stream stage: 73129 for `G`, `M`, `X`, `F`; `F` (and `G`) at 0.10 on 93683 and 116089 |
| Rehearsal | none (births) |
| Scores and thresholds | Mean of rows 16,384, 20,480 and 24,576, for add and sub and for max and min separately. Generalises: held-out at 12 letters at least 90. Memorises: taught at least 90 and held-out at most 30. Room: `G` below 85. Uses the module: a fall of at least 15 points with the module off. Stop rule: `F` within 15 points of `G` on two of three seeds. Stage 0 gate: at least 90 at fraction 0.25 in two of three seeds |
| Found | Review 9.146: "Both inventories of the innate-circuit line are now closed by their own registered rule, with the same answer: the circuit generalises when its inputs are handed to it, and backpropagation does not learn to hand them over from the byte stream." |

### 1.6 E36

Brief: D:`OPUS_TASK_E36_REMINDER_AFTER_A_LIFE_20261115.md`. Code: D:M/`e36_reminder.py`. Receipt read: D:M/`exp45/e36/e36_reminder_K_73129.json`. Report: D:M/`E36_REPORT.md`.

| Setting | Value run |
|---|---|
| Rules | Reminders under S, B and K, each from the end of the life made under that rule (E26's lives S and B; E31's lives K with their saved `Bhh`). References under S and B from untrained networks. K is read against the reference under S |
| Optimizer | standing: "A fresh `exp26_candidate.make_opt(m)` in every run. The lives saved no optimizer state." Receipt: 0.001, 0.25, (0.9, 0.98), byte table 0.0 |
| Batch; network; byte code | standing |
| Error through time | through the whole item (`exp26_candidate.step`; for K `arch31_workerB.forward(m, x, True, True, switch="K", Bhh=B)`, `e36_reminder.py` line 119) |
| Tasks, fraction | the birth family alone: the taught half of the pairs of the four operations under 89, with a stream seed that is not the birth's (`seed + 36`); the same batches for every run of a seed (receipt `data`) |
| Letters | Training 0 to 12. Tests: the birth family's held-out panels at 12 letters; all 7,921 pairs of add and of sub without letters, split by whether the answer needs the wrap at 89; the life's families at 12 letters |
| Dose | 1,024 updates; rows at 0, 16, 32, 64, 128, 256, 512, 1,024. Fifteen runs |
| Seeds | the three |
| Rehearsal | none; the reminder is given alone |
| Scores and thresholds | Restored: the mean of held-out add and sub under 89 at 12 letters is at least 85 at a row. Readings, in two of three seeds: restored at row 256; or the wrap score at least 30 points above the reference's at row 1,024; or less than 15 points from it |
| Found | Review 9.136: "E36: what a life removes comes back in about a hundred updates." |

### 1.7 E37

Brief: D:`OPUS_TASK_E37_FRESH_REFERENCE_FOR_A_LIFE_FAMILY_20261115.md`. Code: D:M/`exp37_fresh_family.py`. Receipt read: D:M/`exp45/e37/e37_F3_S_73129.json`. Report: D:M/`exp45/e37/E37_REPORT.md`.

| Setting | Value run |
|---|---|
| Rules | S and B on fresh networks. The lived side is read from the receipts of E26's lives (S, B) and E31's lives (K); K is read against fresh S |
| Optimizer; batch; network; start; byte code; error through time | standing (`exp26_candidate.step` and `make_opt` are called). Receipt: 0.001, 0.25, `batch 64`, `params 1124113` |
| Tasks | Part 1: family F3 (modulus 73) alone, the life's own item stream. Part 2: family F1 (modulus 97) alone, batch for batch the life's first block |
| Taught fraction | 0.25 |
| Letters | Training 0 to 12. Tests at 12 letters, 400 held-out pairs a task |
| Dose | 4,096 updates for F3 (rows 1,024, 2,048, 4,096); 6,144 for F1. Twelve runs |
| Seeds | the three |
| Rehearsal | none in the fresh runs. The lived side learned F3 in a joint stream in which about a third of the items were F3 |
| Scores and thresholds | The life's score minus the fresh network's at equal items of the family, mean of add and sub. Saving: at least 15 points above. Cost: at least 15 below. None: less than 15 either way. Two of three seeds |
| Found | Review 9.136: "E37: a lived learner needs far fewer items for a new family." |

### 1.8 E38

Brief: D:`OPUS_TASK_E38_WHAT_ERASES_THE_BIRTH_FAMILY_20261115.md`. Code: D:M/`exp38_erase.py`. Receipts read: D:M/`exp45/e38/e38_anchor10_S_73129.json`, `e38_rehearse_B_73129.json`, `e38_none_S_73129.json`. Report: D:M/`exp45/e38/E38_REPORT.md`.

| Setting | Value run |
|---|---|
| Rules | S from E26's sweep birth and B from its back-propagation birth, five arms each |
| Optimizer | `make_opt`, then by arm. `zero`: decay 0.25, as E26. `none`: decay 0 in group 0 (receipt: groups read back 0.0 and 0.0). `anchor`: decay 0 and, before each step, `p −= 0.001 · 0.25 · (p − p_birth)` on the 1,122,049 decayed weights. `anchor10`: the same at 2.5 (receipt: `decay_target birth`, `decay_strength 2.5`, `stored_birth_values 1122049`). `rehearse`: decay 0.25. The byte table has no decay and no pull in any arm |
| Batch; network; byte code; error through time | standing (`arch26.forward(m, x, local, True)`, `exp38_erase.py` line 84) |
| Tasks | the first two blocks of E26's life: family F1 (modulus 97) alone. In `rehearse` the birth family's taught half joins as a second stream |
| Taught fraction | F1 0.25; the rehearsed birth family 0.5 |
| Letters | Training 0 to 12. Tests: the birth family held out at 12 letters; all pairs without letters split by wrap; add and sub under 97 at 12 letters |
| Dose | 12,288 updates; 17 scored rows (0, 64, 128, 256, 512, 1,024, then every 1,024). Ten lives |
| Seeds | 73129 only |
| Rehearsal | Only in `rehearse`: each item is drawn from the birth family's stream or from F1's with equal chance (receipt, block 1: 196,298 items of the birth family and 196,918 of F1). No store. Every update. The other arms have none |
| Scores and thresholds, at row 12,288 | Keeps: held-out add and sub under 89 at least 80 and wrap pairs at least 70. Loses: wrap pairs at most 20. Acquires: held-out add and sub under 97 at least 85 |
| Found | Review 9.141: "decay is not needed for the loss (reading 2), a uniform pull toward the birth weights does not protect (reading 5), and plain rehearsal keeps and acquires." |

### 1.9 E40

Brief: D:`OPUS_TASK_E40_NEW_FAMILY_FROM_A_TENTH_OF_PAIRS_20261115.md`. Code: D:M/`e40_new_family.py`. Receipt read: D:M/`exp45/e40/e40_lived_K_93683.json`. Report: D:M/`E40_REPORT.md`.

| Setting | Value run |
|---|---|
| Rules | Lived networks under S, B and K, each from the end of its life; fresh networks under S and B. K is read against fresh S |
| Optimizer | standing, fresh in every run (receipt: 0.001, 0.25, (0.9, 0.98), byte table 0.0) |
| Batch; network; byte code; error through time | standing (`e40_new_family.py` line 134 for K) |
| Task | a family no network had met: add, sub, max, min under modulus 83, codes `e f g h`, taught alone |
| Taught fraction | 0.10: 689 of 6,889 pairs a task |
| Letters | Training 0 to 12. Tests: held-out and taught panels at 12 letters; all held-out pairs without letters, split by whether the answer needs the reduction at 83 |
| Dose | 8,192 updates; rows at 0, 64, 128, 256, 512, 1,024, 2,048, 4,096, 8,192. Fifteen runs |
| Seeds | the three |
| Rehearsal | none |
| Scores and thresholds | The reduction score, and each run's gain in it from its own row 0. At row 8,192, in two of three seeds: saving if the lived run's gain is at least 15 points above the fresh run's |
| Found | Review 9.141: "E40: a lived learner learns a new family from a tenth of its pairs. Gain on reduction pairs from each run's own row 0, at row 8,192: lived 71 to 94 points, fresh 6 to 12" |

### 1.10 E41

Brief: D:`OPUS_TASK_E41_COMPUTATION_TO_THRESHOLD_BY_WIDTH_AND_FRACTION_20261116.md`. Code: D:M/`exp41_scale.py`. Receipt read: D:M/`exp45/e41/e41_S_h256_f25_116089.json`. Reports: D:M/`E41_REPORT_73129.md`, `E41_REPORT_93683.md`, `E41_REPORT_116089.md`; the three-seed report is W:M/`E41_REPORT.md`.

| Setting | Value run |
|---|---|
| Rules | S and B |
| Optimizer | standing at all three widths: "learning rate 0.001, decay 0.25, betas (0.9, 0.98), the embedding table of width 8 without decay". Confirmed at the half width in the receipt |
| Batch | 64 |
| Network | the standing class at three widths (feed-forward width, cells): (256, 128) with 317,457 parameters; (512, 256) with 1,124,113; (1,024, 512) with 4,211,985 (`exp41_scale.py` lines 35 to 37; receipt `H 256`, `U 128`). The byte table stays 258 by 8 at every width. Starting weights from the constructors under `T.manual_seed(seed)` |
| Error through time | through the whole item (`exp26_candidate.step`) |
| Tasks | the four operations under 89, from random weights |
| Taught fraction | 0.10, 0.25, 0.50 |
| Letters | Training 0 to 12. Tests: accuracy A on the held-out panel at 12 letters; accuracy R on 1,000 held-out reduction pairs a task without letters; the taught panel at 12 letters |
| Dose | 8,192 updates at the double width, 12,288 at the present width, 40,960 at the half width (receipt `dose 40960`); scored every 256 |
| Seeds | the three; 54 runs |
| Rehearsal | none |
| Scores and thresholds | First row at 50, 80 and 90 on A and on R, in a single-row and a three-row version; computation as parameters times 64 times updates. A step in width "falls" below a ratio of 2/3, "rises" above 1.5, otherwise "stays" |
| Found | Review 9.157, of the coordinator and marked by it "not audited": "the half width is cheapest to threshold at f = 0.50 under backpropagation on all three". The three-seed report (W:M/`E41_REPORT.md`) gives tables and predictions and states "No readings beyond the brief's description"; for example P41.3, fewer weight-items at the double width, "Did not occur, 0 of 3" |

### 1.11 E43 with its two addenda

Brief: D:`OPUS_TASK_E43_IS_A_FAMILY_LOST_TO_UNRELATED_LEARNING_20261116.md`. Code: D:M/`exp43_unrelated.py`, `e43_task.py`, `e43_addendum.py`, `e43_addendum2.py`. Receipts read: D:M/`exp45/e43/e43_unrelated_S_73129.json`, `e43_unrelated-decay_B_73129.json`, `addendum1/e43_unrelated-plain_lr0.03_S_73129_to16384.json`, `addendum1/e43_reminder_S_73129.json`, `addendum2/e43_unrelated-rehearse_S_73129.json`, `addendum2/e43_unrelated-e42P8_S_73129.json`. Report: D:M/`exp45/e43/E43_REPORT.md`.

| Setting | Value run |
|---|---|
| Rules | S from E26's sweep births, B from its back-propagation births |
| Optimizer, main runs | `make_opt`, fresh, then group 0's decay by arm: 0 in `conflict` and `unrelated` (receipt: groups read back 0.0 and 0.0); 0.25 in `conflict-decay` and `unrelated-decay` (receipt: 0.25 and 0.0) |
| Optimizer, addendum 1 | Reminder: standing, decay 0.25 (receipt). `unrelated-plain`: `torch.optim.SGD(m.parameters(), lr)`, no momentum, no decay (receipt: `momentum 0`, `weight_decay 0`). Rates 0.01, 0.03 and 0.1 were run on seed 73129 under B; 0.03 was then used for all six runs |
| Optimizer, addendum 2 | `unrelated-rehearse`: decay 0. `unrelated-rehearse-decay`: 0.25. E42's networks: E42's optimizer with the decayed group set to 0 |
| Batch; byte code; error through time | standing (`arch26.forward(m, x, local, True)`; E42's forward for E42's networks) |
| Network | standing. Addendum 2 also runs E42's `F10` and `P8` of seed 73129, from E42's final weights and structure on `claude/gallant-dijkstra-i9zgsr`, with no replacement event |
| Tasks | `conflict`: family F1 (modulus 97), as the first block of E26's life. `unrelated`: a task on bytes no family uses: 64 symbols (bytes 128 to 191), 32 fillers (192 to 223), a code byte 224 and a query byte 225; the item is `code s1 s2 f_1..f_k query` and the answer `s1 s2` and the end symbol (`e43_task.py`) |
| Taught fraction | F1 0.25. Unrelated task: 2,048 of the 4,096 ordered pairs of symbols |
| Letters | Unrelated task: 0 to 12 fillers in training, 12 in its panels. First family: scored on all pairs without letters and on the held-out panels at 12 letters |
| Dose | 4,096 updates, rows at 0, 64, 256, 1,024, 2,048, 4,096; 24 runs. Addendum 1: reminders of 1,024 updates (six runs); plain runs of 4,096, and the three under S rerun to 16,384. Addendum 2: sixteen runs of 4,096 |
| Seeds | the three; E42's networks on 73129 only |
| Rehearsal | none in the main runs and in addendum 1. Addendum 2: each item is drawn from the unrelated stream or from the first family's taught half with equal chance, by a generator of its own seeded with the string made of `e43`, the seed and `joint` (receipt `stream`). No store. Every update |
| Scores and thresholds | The first family's reduction score as a share of its own value at row 0. Kept: 80% or more. Lost: 20% or less. Read only if the unrelated task's taught panel is at 80% or more in two of three seeds. Under decay the arm is read against "decay alone", computed on the birth with no run |
| Found | Review 9.146: "E43: the registered reading is *lost*, under both rules." Review 9.151: "After unrelated learning the first family is overwritten, not merely unexpressed." Review 9.156: "Rehearsal keeps the first family at no cost under unrelated learning" and "E42's sparse networks lose the family as the dense one does" |

### 1.12 E46

Brief: D:`OPUS_TASK_E46_DOES_LEARNING_THE_FIRST_FAMILY_NEED_WEIGHT_DECAY_20261116.md`. Code: W:M/`exp46_decay.py`. Receipt read: W:M/`exp45/e46/e46_S_wd000_73129.json`. Report: W:M/`E46_REPORT.md`.

| Setting | Value run |
|---|---|
| Rules | S and B |
| Optimizer | standing but for the decay of group 0: 0.25 (the reference, run by the same runner), 0.05 and 0. Receipt at decay 0: 0.001, (0.9, 0.98), group 0 at 0.0, byte table 0.0, field `decay 0.0` |
| Batch; network; start; byte code; error through time | standing, at the present width (receipt `H 512`, `U 256`, `batch 64`) |
| Tasks, fraction | the four operations under 89, half of the pairs taught |
| Letters | Training 0 to 12. Tests: E41's A, R and taught panel; added, addition under 89 at 12 and 40 letters on 300 held-out pairs |
| Dose | 24,576 updates, scored every 256. Eighteen runs |
| Seeds | the three |
| Rehearsal | none |
| Scores and thresholds | Updates to 80 on R (three-row version, E41's four cases); the level of R and the level at 40 letters over rows 24,064 to 24,576, "lower" or "higher" at a difference of 15 points. A cell carries what two of three seeds share |
| Found | W:M/`E46_REPORT.md`: "**The sweep:** learning the first family **does not need** decay." and "**Backpropagation:** learning the first family **does not need** decay." Review 9.157 repeats it and marks it "not audited" |

### 1.13 E48

Brief: D:`OPUS_TASK_E48_A_NEW_FAMILY_IN_CONNECTIONS_GATED_BY_ITS_CODE_20261116.md`. Code: T:M/`exp48_gated_run.py`, T:M/`e48_gated.py`. Receipt read: T:M/`exp45/e48/e48_S_73129_r16.json`. Report: T:M/`exp45/e48/E48_REPORT.md`.

| Setting | Value run |
|---|---|
| Rules | S and B: each of the six lived networks of E26 under the rule it lived under. E31's K lives are not used |
| What may change | Every parameter of the lived network is frozen and is not in the optimizer. Free are the four code rows of the byte table (bytes 101 to 104) and, on the input map and on the recurrent matrix, one added term `up · down` of rank 4, 16 or 64. `up` starts at zero; `down` is drawn uniformly within ±1/√(inputs) from `T.Generator().manual_seed(seed * 1000 + r)`. The terms are present only for items whose first byte is one of the four codes; the harness supplies that flag |
| Optimizer | AdamW, rate 0.001, betas (0.9, 0.98); the terms decay at 0.25; the byte table has no decay (`e48_gated.py` lines 83 to 85; receipt) |
| Batch; byte code | E40's batches of 64; standing byte code |
| Error through time | through the whole item; the gated forward has no cut (`e48_gated.py` lines 55 to 62; the runner asserts `through_time`, T:M/`exp48_gated_run.py` line 57) |
| Task, fraction | E40's family under modulus 83, a tenth of its pairs, E40's stream |
| Letters | as E40 |
| Dose | 1,024 updates; rows at 0, 16, 32, 64, 128, 256, 512, 1,024. Eighteen runs |
| Seeds | the three |
| Rehearsal | none. The earlier families are kept by construction, since nothing they use is changed |
| Scores and thresholds | R, the mean of the reduction score at rows 128, 256 and 512, against the same mean of E40's run of the same seed and rule. No price: within 10 points or above in two of three seeds. A price: 30 points or more below |
| Found | Review 9.157, marked "not audited": "a change confined to gated connections costs nothing at rank 16 or more under either rule, with the gate supplied; at rank 4 some price under the sweep". The worker's table (T:M/`exp45/e48/E48_REPORT.md`) reads the same |

## Part 2. Settings with one value in the whole scope

For each setting the table gives the value, the place where the record shows it first chosen, and the reason the record gives. "Not found" means that the files read give no reason.

| No. | Setting | The one value | First chosen | Reason in the record |
|---|---|---|---|---|
| 2.1 | Learning rate of AdamW on the dense network | 0.001 in every run of the scope that uses AdamW on this network. The only other rates in the scope are with another optimizer (2.2) or on another network (E35's stage 0, 0.003 on a small parsed network) | D:M/`mod97_local.py` lines 142 to 144, the decay sweep of 6 November on the windowed network under both back-propagation and sign feedback (D:M/`REPORT.md`, "Third round": "AdamW, lr 0.001, betas (0.9, 0.98), decoupled decay 0.1, 0.3 or 1.0"). The same file gives 0.001 as the rate of the earlier learner V181 (line 126). Carried through `exp45.py` (line 242), `exp11_cell.py` (line 274), `exp16_integrated.py`, `exp20_reference.py` line 53 and `exp26_candidate.py` line 39 | The code comment at line 142: "decoupled weight decay (AdamW), lr .001, as in the grokking papers". No run that chose the rate on this network or for rule S was found. The audit of E46 declined a second rate ("A second learning rate: no."), and the E46 brief says "only the rate 0.001 is run" |
| 2.2 | Kind of optimizer | AdamW, with one exception: the plain gradient steps of E43's addendum 1 (SGD without momentum) | as 2.1 | as 2.1. The exception was made to test whether AdamW's equal-sized steps cause the loss (brief E43, addendum 1) |
| 2.3 | Betas | (0.9, 0.98) | as 2.1; fixed inside `exp16_integrated.adamw16` (line 187) | "as in the grokking papers" (the same comment). Nothing else found |
| 2.4 | Epsilon | not set by any runner | — | not found |
| 2.5 | Which parameters are exempt from decay | the byte table alone | `exp16_integrated.adamw16` with `E16_BOOKDECAY=off` | Review 9.60 and 9.61: the decay alone had shrunk the byte-table rows that the local rule did not reach, and the table was put in a group without decay as the fix |
| 2.6 | Schedule of the rate; clipping | none | — | not found (the E26 brief states only that the loop has "no clipping or reset") |
| 2.7 | State of the optimizer at the start of a life or a continuation | fresh | E26's life (`exp26_candidate.py` line 205) | Brief E36: "The lives saved no optimizer state." |
| 2.8 | Batch | 64 items | D:M/`mod97_local.py` line 38, following Codex's run of 6 November | Review 9.27 adopts "Decay 0.25, fresh-render replay, batch 64 and the exact trace" as the settings, after E13 had tried batches of 64, 256 and 1,024 on the windowed network under the sign rule (review 9.24) |
| 2.9 | Starting weights of the network | the constructors' own draws under `T.manual_seed(seed)`; nothing rescaled | `arch20.py` lines 62 to 69 | not found |
| 2.10 | The form of rule S | the error crosses each linear map through the signs of its weights times one mean size taken over the whole stacked matrix | `exp3_state.py` lines 91 to 104; for the GRU, the E26 brief, question 4 to the audit | The audit answered "one scale per stacked matrix" as drafted (brief E26, last section; review 9.105). No reason beyond that was found |
| 2.11 | Error through earlier positions | through all of them, under both rules | brief E26, "The update, exactly" | Brief E26, revision 2: the harness's own rule update "cuts credit at every step and calls the bank's trace; it must not be used". Earlier, review 9.8: "removing credit through time costs 20–30 and most of the extrapolation" |
| 2.12 | How a byte is coded | a learned table of 258 rows by 8 without decay, one byte a step, then `l1` and tanh | the table of width 8: D:M/`mod97_local.py` line 104, following Codex's network of 6 November (review 9: "8-byte window × 8-dim embedding"). The layer after it: D:M/`arch20.py` lines 13 to 14 | `arch20.py`: the GRU keeps the earlier network's input layer and readout "so that the only change from arch17 is the recurrent core". No reason for the width 8 was found |
| 2.13 | Kind of cell and depth | one layer of GRU cells in the library's form, zero state at the start of each item | D:M/`arch20.py` (E20-B) | Brief E26: "The network that composes from memory is the GRU, selected in E20-B by a rule fixed beforehand (review 9.101)." |
| 2.14 | Letters in training | uniform, 0 to 12 | D:M/`exp3_state.py` lines 46 to 47 and D:M/`EXP3_REPORT.md` line 11 | not found for the value 12 |
| 2.15 | Targets in every life and continuation | answer bytes only, found with the answer mask. (Births under other targets: E33, E34.) | D:M/`EXP3_REPORT.md` line 17; brief E26 | `EXP3_REPORT.md`: with every position a target "the answer signal was swamped". Brief E26: "Answer targets only. The every-byte question is E24's" |
| 2.16 | Answering and scoring | the largest output at each step with the prompt rerun; whole answers with the end symbol | `exp11_cell.py` lines 188 to 222 | not found |
| 2.17 | Seeds | 73129, 93683, 116089, and no other | D:M/`mod97_local.py` line 37 (Codex's run) | not found |
| 2.18 | The birth from which every continuation starts | E26's six births: modulus 89, half of the pairs, 24,576 updates, decay 0.25, the present width. E31, E36, E37's lived side, E38, E40, E43 and E48 all start from them or from the nine lives grown from them. No continuation in the scope starts from a birth at another fraction, width, decay or dose | brief E26, "Why E20-B's protocol" | "That is the protocol under which this network is known to compose" |
| 2.19 | The life from which every lived network comes | one schedule (F1, F1, F2, F2, F3, F3), blocks of 6,144 updates, families taught at a quarter of their pairs, the joint stream, decay 0.25 | `exp16_integrated.py` line 855; the block length and the quarter of the pairs are E8's (D:M/`E45_REPORT.md` line 185) | as 2.18 |
| 2.20 | Share of rehearsal, where there is any | each item drawn from the old or the new stream with equal chance: one half of the items in E38's `rehearse` and in E43's rehearsed arms; one half and two thirds in a life's later blocks | `exp16_integrated.joint_stream` ("equal weight per family"); brief E38 ("about half"); brief E43, addendum 2 ("with equal chance"); memo 24.8 ("in E38's share") | For the windowed learner the record has a line of runs on the share (review 9.64, closed in 9.68). On the dense network no reason for one half was found |
| 2.21 | Store; schedule of rehearsal | no store (rehearsed items are drawn fresh from the taught pairs); rehearsal in every update, never at intervals | brief E26: "No trace routine, no scan, no local head, no store." | Brief E26: "The learner's own pipeline (a replay store filled as the stream arrives) is the next step and is not run here." |
| 2.22 | Host and arithmetic | one host class, torch 2.15.0+cpu, one thread | briefs | Review 9.112 H (the qualified runtime) |

Three settings named in `PLAN.md` section 3 do not have one value in the scope and are in Part 3: the decay, the taught fraction, and the width. Each of them has one value on part of the scope, and Part 3 says where.

## Part 3. Settings that were varied

| Setting | Experiments | Values | Outcome in one line |
|---|---|---|---|
| The rule | all but E33 (S only) and E35 (B only) | S, B; K in E31, E36, E40 | Births level (review 9.110); lives under S short of B on untaught pairs (9.112); the readings of E36, E38, E40 and E43 are the same under both rules, and in E37 they differ on one comparison (F1: a saving under S, none under B); E41: S needs more updates than B, most at the half width and at a quarter of the pairs (memo 28.1) |
| The way back across the recurrent map | E31 | sweep; exact gate rows (G); exact recurrent matrix (H); a separate matrix (K) | H and K close at least half of the gap on three seeds; G on one (D:M/`exp45/e31/E31_REPORT.md`) |
| A push on the update gates | E31 | strengths 0.25, 0.05, 0.01 | "The gate push did not help composition at any strength" (review 9.130) |
| Weight decay during a continuation | E38 (blocks of F1), E43 and its addendum 2 | 0.25 and 0; E38 also a pull to the birth weights at 0.25 and 2.5 | The first family is lost alike with and without decay; neither pull protects (review 9.141, 9.146) |
| Weight decay at birth | E46 | 0.25, 0.05, 0, at the present width and half of the pairs | "does not need decay" under both rules (W:M/`E46_REPORT.md`) |
| What the decay acts on | E48 against E40 | every weight of the lived network (E40); only the added terms (E48) | Not separated from the other difference between the two (brief E48, "Optimizer") |
| Kind of optimizer and its rate | E43 addendum 1 | SGD without momentum at 0.01, 0.03, 0.1 (B, seed 73129); 0.03 for six runs | The family is lost under plain steps as well (review 9.151) |
| Optimizer of a small parsed network | E35-B stage 0 | AdamW(0.003, decay 0.01) and AdamW(0.001, decay 0.25, betas (0.9, 0.98)) | The angle form passes under both (D:M/`E35B_REPORT.md`) |
| Taught fraction of a birth from random weights | E35 (0.10, 0.25), E41 (0.10, 0.25, 0.50) | — | At 0.10 no run reaches 80 at any width; at 0.25 B reaches it and S in part of the runs (W:M/`E41_REPORT.md`; memo 28.1) |
| Taught fraction of a new family for a lived network | E37 (0.25), E40 and E48 (0.10) | — | Saving at both (review 9.136, 9.141) |
| Width | E41 | half, present, double | Updates to threshold fall with width and computation rises (W:M/`E41_REPORT.md`) |
| Dose | E34 addendum 2 (98,304 against 24,576); E41 (by width); E43 addendum 1 (16,384 for three runs) | — | E34: "mixed" at four times the dose (review 9.146) |
| Targets | E33 (classes of non-answer bytes); E34 (every byte with a gain, tau 0.5, 1, 2) | — | E33: letter targets stop extrapolation on three seeds (review 9.134). E34: at tau 0.5 the sweep learns in range and extrapolates on seeds 73129 and 116089 and does not learn in range on 93683 (D:M/`exp45/e34/E34_REPORT.md`) |
| The start of a continuation | E26-X (birth of the other rule); E36, E37, E40 (lived, born or fresh) | — | The gap of E26 is in the update of the life (9.112); a lived network needs fewer items (9.136, 9.141) |
| Rehearsal, present or absent | E38 (one seed), E43 addendum 2 (three seeds) | none; one half | With it the first family is kept and the new task learned (9.141, 9.156) |
| An added module | E35 | none, wired, scrambled, forced | Not used from the byte stream (9.143, 9.146) |
| Rank of the added term | E48 | 4, 16, 64 | No price from rank 16 under both rules; some price at rank 4 under S |
| The kind of later task | E38 (a contradicting family), E43 (an unrelated task) | — | Lost in both cases |
| The network's structure | E43 addendum 2 | the dense network; E42's `F10` and `P8` (seed 73129) | "E42's sparse networks lose the family as the dense one does" (9.156) |
| Letters in tests | all | 0, 4, 8, 12, 20, 40 | Extrapolation to 40 letters is read in E33, E34, E46 |

Where each of the three settings of `PLAN.md` was not varied:

- **Decay.** It was 0.25 in every six-block life (E26, E26-X, E31), in every run at the half and the double width (E41), in every birth at a tenth or a quarter of the pairs (E35, E41), in every run of E36, E37 and E40, and under the targets of E33 and E34. E46 varied it only at the present width, at half of the pairs, at the rate 0.001.
- **Taught fraction.** Every life teaches its families at a quarter of their pairs, and every continuation starts from a birth at half of the pairs (2.18, 2.19). No life or continuation was run from a birth at another fraction.
- **Width.** Lives and continuations exist at the present width only (brief E41, "What it cannot show": "Networks with a life behind them exist at one width only"). The present sizes were fixed before the scope: the feed-forward width 512 in D:M/`exp11_cell.py` line 28, and the 256 cells in D:M/`arch20.py` line 14, "chosen to put the parameter count within 5% of bank2's 1,125,265".

## Part 4. Comparisons made under a setting chosen for one side

| No. | Comparison | The setting | Evidence that it was chosen for one side or for an earlier network | Was it varied for the other side? |
|---|---|---|---|---|
| 4.1 | S against B, in every experiment that has both (E26, E26-X, E34's `B050`, E36, E37, E38, E40, E41, E43, E46, E48) | the rate 0.001 and betas (0.9, 0.98) of AdamW | Chosen on 6 November on the windowed network in a decay sweep (Part 2, 2.1 and 2.3). Brief E41, "What it cannot show": "Learning rate and decay were chosen at one width and are used at all three, so a width may be handicapped by them." V:`AUDIT_E41_COMPUTATION_TO_THRESHOLD_20261116.md`, section 11, lists as not checked "Whether learning rate 0.001 and decay 0.25 suit the other widths." | No. A search of every runner on D that imports the dense network found another rate only in `e43_addendum.py`, with SGD. V:`AUDIT_E46_WEIGHT_DECAY_BRIEF_20261116.md`, section 4: "A second learning rate: no. It changes every other quantity of the birth and makes E46 another experiment." |
| 4.2 | S against B in E26 and in every life after it | decay 0.25 | Brief E26: "Every rule run on file uses 0.25 and every backprop birth uses 0.30 ... Here both arms use the rule's decay". The value 0.25 was adopted for the sign rule on the windowed and the cell learner (review 9.24, 9.25, 9.27; V:`AUDIT_E46_…`, section 4: "chosen in 9.24 and 9.25 and confirmed in 9.27, for lives of that week's networks"). The memo, 24.6: "The decay was chosen for births ... whether the births here need it is not on file as far as I have found" | Partly. Back-propagation at 0.30 on this pipeline was named "a follow-up, wanted only if the matched reference disappoints" and was not run. E46 later ran 0, 0.05 and 0.25 under both rules for a birth; E38 and E43 ran 0 and 0.25 in short continuations. In six-block lives the decay was not varied under either rule |
| 4.3 | Half, present and double width (E41), and S against B at each | rate, decay and betas | As 4.1: fixed at the present width. The doses also differ by width (40,960, 12,288, 8,192 updates) | No. V:`E41_E46_VERIFICATION_20261117.md`, section 7: the sweep's ratio to back-propagation "is nearest to 1 at the double width and largest at the half width"; section 8, item 6: "the cheapest width is the edge of the range that was run" |
| 4.4 | E26's lives judged by the marks 85 and 40, and S against B over a life | the life's protocol: the schedule, blocks of 6,144 updates, the joint stream | Brief E26: "That is the protocol under which this network is known to compose, with E20-B's marks as absolute marks", and "The marks 85 and 40 are E20-B's, fixed before E20-B ran". E20-B's networks were "trained by backprop through time only" (`arch20.py` line 16) | No. No life with longer blocks, another order or another share of rehearsal was run under either rule |
| 4.5 | A fresh network against a lived one on few pairs (E37, E40), and the fresh references of E36 and of E43's reminder | the standing optimizer and a short dose for the fresh side | Brief E40, "What it cannot show": "The fresh runs here have a third of a birth's updates. E35's arm `G` is the fresh reference at a full birth, on mod 89 and one seed." The record's cost of weak decay "was at 15% of the pairs" (brief E46), and V:`AUDIT_E46_…`, section 4, cites review 9.2 and 9.3 for the windowed network: a decay of 0.3 gave 82 to 89% at 15% of the pairs where 1e-4 gave 65 to 72% | No. A fresh dense network at a tenth of the pairs was run only at decay 0.25 and rate 0.001 (E35's stream stage, E40, E41). Memo 28.1: at the half width under back-propagation "score A is still rising at 40,960 updates (48, 68 and 58%)" |
| 4.6 | Lived K against fresh S (E36, E37, E40) | the rule of the reference | Brief E36: "Arm K is read against the sweep reference." Brief E40: "lived K against fresh sweep." | No fresh network under K exists in the scope |
| 4.7 | K against H and against the floor (E31) | the start of `Bhh` (random, U(−1/16, 1/16)) and its rate of approach (0.00025 an update) | Brief E31, "Arm K's rule". Addendum 2 adds arm `Ks`, started from the sweep's own matrix, "code and dry run now" | No. Report: "Arm Ks: not run at full length." |
| 4.8 | `B050` against plain back-propagation, and against `T050` (E34) | the temperature 0.5 and the division by the sum of the gains | The temperature was selected by the sweep's screen on seed 73129 (brief E34: "says that seed 73129 selected it"). The division was set so that the first-batch gradient is "the same as S-all's" | No. Under B only tau 0.5 was run, on one seed |
| 4.9 | The eight arms of E33 | the weight of a selected position, kept at the balanced loss's `0.5 / other.sum()` | Brief E33: "The weighting is a choice, and it is this one". Dropping a class lowers "both the weight on non-answer targets and the error they carry" | No. The control `A+D-light` was fixed in the brief and not run (report: "NOT_RUN. The control `A+D-light`") |
| 4.10 | The GRU alone against the GRU with a module (E35) | the optimizer of the stream stage | The module's parameters were put in E26's decayed group at 0.25 and rate 0.001. The ring was developed under AdamW at 0.003 with decay 0.01 ("as E10's development"), which the stream stage does not use | For 乙's parsed network both optimizers were run (stage 0). In the stream stage only E26's |
| 4.11 | With rehearsal against without (E38, E43 addendum 2) | the share, one half, at equal updates | Brief E38, table of arms: "about half". Brief E43, addendum 2: "with equal chance". Memo 28.6: "Half of the items of every update are rehearsal". The rehearsed arm therefore receives half as many new items an update | No. No other share, and no rehearsal at intervals. The report adds: "The rows are coarse: nothing is measured between 256 and 1,024." |
| 4.12 | The two pulls of E38 against decay to zero | strengths 0.25 and 2.5, a pull fixed to the birth weights | Brief E38: `anchor` is "decay 0.25 toward each weight's birth value" and `anchor10` "the same at strength 2.5"; "A fixed anchor suits a two-block screen only". The brief gives no reason for the two values | Two values, one seed |
| 4.13 | S against B under plain gradient steps (E43 addendum 1) | the rate 0.03 | Brief E43, addendum 1: "Three runs on seed 73129 under backpropagation at LR = 0.01, 0.03 and 0.1 ... The rate used for all six runs is the smallest at which the taught panel is at 80% or more". Report: under S the condition "fails at 4,096 (taught 0.3, 36.3, 95.3)", the runs were continued to 16,384, and "Read at 8,192, that arm had twice the dose of the backpropagation arm" | No rate was tried under S |
| 4.14 | E48 against E40 | what decays; the rate; the scale of `down`; the decay of the term | Brief E48: "E48 and E40 therefore differ in two things, where the change is made and whether the lived weights decay". V:`E48_RESULT_VERIFICATION_20261117.md`, section 10: "The rate and the form of the change were not varied; part of the delay at the start may belong to them" | No |
| 4.15 | Family F3 in a life against F3 alone in a fresh network (E37, part 1) | the stream | Brief E37: "the fresh network gives every update to F3 and the life gave a third. That favours the fresh network per update and is neutral per item." | E40 teaches the family alone on both sides |
| 4.16 | Decay 0 against 0.25 at birth (E46) | the rate, the width, the fraction | Brief E46: "Decay acts through rate × decay; only the rate 0.001 is run"; "The answer is for the present width"; "this job is at half" of the pairs | No |

## Part 5. What the briefs and reports themselves say was not shown

Each row gives the item, its source, and the later experiment that took it up. "None" means that no experiment of the record, in or out of this scope, was found that did. "Not run" names a job that is written or planned and has no result on file.

### 5.1 E26 and E26-X

| Item | Source | Taken up by |
|---|---|---|
| "Answer targets only. The every-byte question is E24's and is not asked here." | brief E26, "What this cannot say" | For births: E28 (review 9.124), E33, E34. For a life under other targets: none |
| "The joint stream rehearses everything. Nothing here is about retention or about a replay store." | the same | Retention without rehearsal: E36, E38, E43. A replay store on this network: none in the scope |
| "The registered decoder keeps the prompt and reruns the network for each output byte." | the same | none |
| "Three seeds, one size, one host class." | the same | Size: E41, for births only. Other seeds: none |
| The learner's own pipeline, a replay store filled as the stream arrives, "is the next step and is not run here" | brief E26, "Why E20-B's protocol" | none in the scope |
| A back-propagation arm at decay 0.30 on this pipeline | brief E26, revision 2 and "Readings" | none; its condition (P26.5 missing) did not occur |
| One scale per gate in place of one per stacked matrix | brief E26, question 4 | none |
| "Byte-by-byte training and autonomous replay come after attribution" | D:M/`exp45/e26x/E26X_DESIGN.md`, "Not in scope" | Byte-by-byte births: E28, E33, E34. Autonomous replay: none |
| Three crossed lives were interrupted and finished by a resumable runner; "I did not repeat that comparison" | review 9.112 B | not found |
| Finite losses are recorded for 3,072 of the 24,576 updates of each birth; that none was non-finite is "an inference from the record ... recorded as a ruling" | brief E26, revision 4 | Later receipts record a cumulative flag (the same revision) |

### 5.2 E31

The brief has no section on what it cannot show. Its report and the review give these.

| Item | Source | Taken up by |
|---|---|---|
| K "is not replicated on fresh seeds" | review 9.128 A and 9.130 | none (review 9.128, "Held": a replication of K "belongs in the confirmatory run of whichever rule first passes the every-byte condition") |
| "Arm Ks: not run at full length." P31.10 unjudged | D:M/`exp45/e31/E31_REPORT.md` | none |
| Arm `P` "withdrawn before any full life" | the same; brief, addendum 2 | replaced by `P2a` to `P2c` |
| A strength of the push that passed would be "run again on fresh seeds before anything is claimed" | brief, addendum 2 | not needed; none passed |
| The push stops at 0.9, so it "cannot be read as a test of holding across letters" | review 9.125 | E32 (outside this scope) pushed with a limit of 0.99 on births |
| K's gain on new instances of taught pairs "is small" | review 9.128 A | none |
| Feedback of K's kind on all maps, "the candidate" if no class of target is decisive | brief E33, "What follows from it" | none |

### 5.3 E33

| Item | Source | Taken up by |
|---|---|---|
| Kind against amount: letter targets decisive "is exactly what a threshold on error between 12.2 and 20.6 nats would give" | brief, "What stage 1 cannot show" | none; the control `A+D-light` was "NOT_RUN" (report) |
| "One seed." | the same | `A+D` and `A+OS` on two more seeds (addendum 1). The other six arms: none |
| "Why. A decisive class says where the interference enters, not how." | the same | none (brief E34: "neither experiment says how") |
| The sentence on three seeds says "Nothing about why, and nothing about amount against kind." | brief, addendum 1 | none |
| The first `A+DS` and `A+OS` births were killed by a time limit and rerun from the start | report, "Run record" | closed in the report (the killed rows' hashes equal the reruns') |

### 5.4 E34

| Item | Source | Taken up by |
|---|---|---|
| "Amount against kind ... A pass does not say which reduction mattered, nor whether a fixed weight of the same size on distractor targets would do as well." | brief, "What stage 1 cannot show" | none |
| "Why." | the same | none |
| "A like-for-like bound." | the same | `B-all` is cited as the reference; `B050` was run (addendum 1) |
| "One seed." | the same | `T050` on two more seeds; it did not learn in range on 93683. `T100`, `T200`, `B050`: none |
| "Later use. Whether a network born this way then lives and composes under the sweep is not tested here." | the same | none ("The gain therefore does not go into a life until this cost has been measured", addendum 1) |
| "The stream is stationary, so this bears on the shape 'slow start, later lead' and not on continual learning." | brief, addendum 2 | none in the scope |
| The two long runs "then go to seeds 93683 and 116089" if the gain overtakes and stays | brief, addendum 2 | not triggered; the reading was "mixed" |
| "A confirmation on new seeds, and the combination with Kolen–Pollack feedback during a life, need their own brief." | brief, "Stage 2" | none |
| The coordinator had "not yet read" stage 2 and the long births "beyond the prediction tables" | review 9.146 | not found in a later section |

### 5.5 E35-A and E35-B

| Item | Source | Taken up by |
|---|---|---|
| "Nothing about a rule inside the line." | both briefs, "What this brief cannot show" | none; the line was closed (review 9.134, item 4) |
| "Untaught compositions, few-shot acquisition of a new family, and retention are not measured here." | the same | For the plain GRU: E37, E40 (a new family); E36, E38, E43 (retention). For the networks with a module: none |
| "Breadth." | the same | none |
| "Why the controller does or does not use the circuit." | the same | The forced arm `F` tests one cause; otherwise none |
| "A pass-through." | the same | not needed; no arm passed |
| "Thin coverage at f = 0.10." | the same | none as such. E41 ran a tenth of the pairs at three widths |
| Arms `M` and `X` at fraction 0.25 "NOT RUN: no family has room" | both reports | none |
| A rerun at fraction 0.05 if no family has room | both briefs | not triggered |
| "No formal checker gate or external review is claimed." | D:M/`E35A_REPORT.md` | The reviewer "checked the final rows of `G`, `M` and `X` against the receipts" (brief E35-A, addendum 1) |
| That 乙's scrambled control cannot be learned around "is its reasoning, not a numerical result" | review 9.133 | none |

### 5.6 E36

| Item | Source | Taken up by |
|---|---|---|
| "The optimizer state is fresh in reminder and reference alike." | brief, "What it cannot show" | none; no run in the scope carries an optimizer's state |
| "The reminder is given alone, not inside a life's stream ... E38 does." | the same | E38's `rehearse` (continuous, one half, one seed). Reminders at intervals inside a life: none |
| "Nine saved lives designed for another purpose; three seeds." | the same | E43 addendum 1 gives a reminder after unrelated learning, from births |
| One part of check 3 failed as first written and was replaced by the worker | D:M/`E36_REPORT.md`, "Things to note" | closed in the report |

### 5.7 E37

| Item | Source | Taken up by |
|---|---|---|
| "A saving is in items of the new family, not in total experience." | brief, "What it cannot show" | none |
| "In part 1 the fresh network gives every update to F3 and the life gave a third." | the same | E40 teaches the family alone on both sides |
| "It does not say what in the lived network gives the saving." | the same | none by experiment (the memo, section 23, measures saved weights) |
| "no formal checker gate or external review is claimed" | D:M/`exp45/e37/E37_REPORT.md` | The reviewer recomputed the fresh rows from the weights (memo 19.1) |

### 5.8 E38

| Item | Source | Taken up by |
|---|---|---|
| "One seed, and seed 73129 is the one whose E26 lives lost least" | brief, "What stage 1 cannot show" | The arms `zero` and `none` on three seeds for 4,096 updates, as E43's `conflict` arms. The pulls and `rehearse` with a contradicting family on other seeds: none (stage 2 "is not released", review 9.136, addendum) |
| "Why a pull works or fails." | the same | none by experiment (the memo, section 29, on saved weights) |
| A full six-block life under a passing pull; a pull that moves | brief, "Stage 2" | none |
| "Checks 2 and 3 failed as specified under the sweep ... a reviewer should decide whether that suffices." | D:M/`exp45/e38/E38_REPORT.md`, "Checks" and "Limits" | not found: no ruling in the review or in the memo |
| "No independent review of this run has taken place." | the same | Review 9.141: "I recomputed none of them this time". A check of E38 from its weights was not found |

### 5.9 E40

| Item | Source | Taken up by |
|---|---|---|
| Part of a lived learner's level "is there before any update; the verdict excludes it and the level does not." | brief, "What it cannot show" | — |
| "One new family, at one modulus." | the same | E48 uses the same family. Others: none in the scope |
| "Lives made for another purpose; a fresh optimizer in every run; the family is taught alone and not inside a life's stream." | the same | none in the scope |
| "a saving does not say which part of the life gives it" | the same | none |
| "The fresh runs here have a third of a birth's updates." | the same | E41, at a tenth of the pairs under 89, up to 40,960 updates: 80 is not reached |
| Check 8 compares with the audit's table "transcribed into the runner"; "I did not run the reviewer's script." | D:M/`E40_REPORT.md` | Review 9.141: the reviewer's reader "reproduces all fifteen E40 files at row 8,192" |

### 5.10 E41

| Item | Source | Taken up by |
|---|---|---|
| "Settings. Learning rate and decay were chosen at one width and are used at all three" | brief, "What it cannot show" | Decay: E46, at the present width only. Rate: none. Not run: item 8 of `PLAN.md` |
| "Scoring every 256 updates bounds how finely a threshold is located." | the same | none |
| "Experience. Networks with a life behind them exist at one width only" | the same | none in the scope |
| "One task family, from random weights, with answer bytes as targets." | the same | none |
| CPU-seconds "depend on the host" | the same | — |
| The half width "is also the narrowest width that was run, so the cheapest width is not located" | memo 28.1 | none. Not run: item 14 of `PLAN.md` |
| "Whether back-propagation at the half width would reach 80% from a tenth of the pairs with a longer dose is open" | V:`E41_E46_VERIFICATION_20261117.md`, section 7 | none |
| Checks 3 to 6 and 8 "were not run" on the hosts of seeds 93683 and 116089 | D:M/`E41_REPORT_93683.md`, `E41_REPORT_116089.md` | on file for seed 73129 |
| The three-seed report is not on D | V:`E41_E46_VERIFICATION_20261117.md`, section 8, item 7 | confirmed here: W is not an ancestor of D |

### 5.11 E43 and its addenda

| Item | Source | Taken up by |
|---|---|---|
| "A dose of 4,096 updates. A life is 36,864." | brief, "What it cannot show" | none |
| "One unrelated task, a simple one." | the same | none. The second unrelated task "without carrying across a gap" was named only for a reading of kept |
| Long-term memory as "a fact stored once and recalled by a cue" | the same | none |
| "The optimizer. Every run starts a fresh AdamW" | the same | Addendum 1, `unrelated-plain` (SGD): lost as well. A carried state: none |
| "Why." | the same | The reminder of addendum 1; the reviewer's measurements on saved weights (memo 24.8, 29) |
| "Nothing here separates the new task's content from that first-step effect." | report, "Limits" | addendum 1, `unrelated-plain` |
| The plain arm under S was read at 8,192, "twice the dose of the backpropagation arm" | report, addendum 1, "Limits" | none |
| The weights listed as committed "were not" on the remote | report, addendum 1, section 1 | closed in the same addendum |
| "The P8 row files hold the network's `state_dict` only" and cannot be rebuilt | report, addendum 2, "Limits" | not rerun (memo 28.6: "No reported figure depends on them") |
| "The rows are coarse: nothing is measured between 256 and 1,024." | report, addendum 2 | none |
| E42's networks: one seed | brief, addendum 2 | none |
| "Not independently reviewed." | report, all three parts | Main runs: the reviewer "recomputed E43's from its receipts" (review 9.146). Addendum 2: V:`E49_E43ADD2_VERIFICATION_20261117.md`, whose section 5 lists what it could not verify (the two `P8` networks; rows before 4,096; the content of the rehearsal stream) |

### 5.12 E46

| Item | Source | Taken up by |
|---|---|---|
| "A life. Without decay the weights may grow over 36,864 updates" | brief, "What it cannot show" | none in the scope |
| "Composition, which births do not teach." | the same | none |
| "Few taught pairs ... A quarter of the pairs at decay 0 is the cell to add later" | the same | none. Not run: item 8 of `PLAN.md` |
| "Another width ... one half-width cell without decay is needed" | the same | none (memo 28.2: "there is no birth without decay at the half width") |
| "Other rates. Decay acts through rate × decay; only the rate 0.001 is run." | the same | none |
| "That removing decay gives retention." | the same | E43's arms without decay: lost |
| Under S the final level of R is lower without decay on all three seeds; under B two of three seeds are slower at the edge of the band | V:`E41_E46_VERIFICATION_20261117.md`, section 9.3 | none |
| Without decay the size of the recurrent matrix "has not settled at the end of the run" | memo 28.2 | none |
| The report and its receipts are not on D | the same verification, section 8, item 7 | confirmed here |

### 5.13 E48

| Item | Source | Taken up by |
|---|---|---|
| "That a learner can find the gate, or decide when a new term is needed." | brief, "What it cannot show" | none. Not run: item 4 of `PLAN.md` (E51) |
| "Fixed memory ... The fair comparison is with a replay store of equal size" | the same | none |
| "That a term is kept under the standing optimizer." | the same | none |
| "Unrelated tasks, whose answers need output units of their own; the readout is frozen here." | the same | none |
| "Composition across families" | the same | none |
| "A second new family on top of the first" | the same | none |
| E48 and E40 "differ in two things" | brief, "Optimizer" | none |
| "The rate and the form of the change were not varied" | V:`E48_RESULT_VERIFICATION_20261117.md`, section 10 | none |
| The words "as fast as" are not supported for the first 64 updates | memo 28.7 | — |
| "No independent review or checker gate has been run on this report." | T:M/`exp45/e48/E48_REPORT.md` | V:`E48_RESULT_VERIFICATION_20261117.md` |

### 5.14 What the coordinator's review says it did not check

| Statement | Source |
|---|---|
| E38, E40, E35-B: "I recomputed none of them this time and say so." | review 9.141 |
| E43, E35-A's last stage, E34's later stages, E41's first seed: "I recomputed none of them and say so." E41's first seed "not yet read by me" | review 9.146 |
| E43's additions and E44's calibration: "The tables are the workers'; I recomputed none." | review 9.156 |
| E46, E41's two later seeds, E48: "What I read for it, not audited ... finished and not yet read in full by me." | review 9.157 |
| E31 was "Not audited by a second reader"; two workers' independent implementations stood in for the audit | brief E31, first paragraph |

## Part 6. Things that could be tested and have not been

Each line is one thing, with its source in Parts 2, 4 or 5. The order is that of the Parts and carries no ranking.

From Part 2 (one value in the whole scope):

1. Rule S and rule B on the dense network at a learning rate of AdamW other than 0.001 (2.1, 4.1).
2. Either rule with betas other than (0.9, 0.98), or with another epsilon (2.3, 2.4).
3. A life or a continuation that carries the optimizer's state over instead of starting it fresh (2.7; 5.6, 5.9, 5.11).
4. A batch other than 64 items on the dense network (2.8).
5. Starting weights of the dense network at another size than the constructors give, under either rule (2.9).
6. Rule S with one scale for each gate's part of a stacked matrix instead of one for the whole (2.10; 5.1).
7. A byte table of another width than 8 (2.12).
8. Training with another range of letters than 0 to 12 (2.14).
9. A life under targets other than the answer bytes (2.15; 5.1, 5.4).
10. Any result of the scope on seeds other than the three (2.17; 5.1, 5.2).
11. A life or a continuation that starts from a birth at another taught fraction, width, decay or dose than E26's (2.18).
12. A life with another schedule, block length or taught fraction of its families (2.19; 4.4).
13. Rehearsal at a share other than one half, and rehearsal at intervals instead of in every update (2.20, 2.21; 4.11).
14. A store of past items on the dense network, filled as the stream arrives, in place of fresh draws from the taught pairs (2.21; 5.1).

From Part 4 (a setting chosen for one side):

15. Back-propagation at decay 0.30 on E26's pipeline, beside the sweep at 0.25 (4.2).
16. The decay varied in a six-block life with compositions, under either rule (4.2).
17. Rate and decay chosen for the half and for the double width (4.3).
18. A width narrower than the half width (4.3; 5.10).
19. A fresh network at a tenth of the pairs under a stronger decay or a longer dose than those run (4.5).
20. A fresh network trained under rule K, as the reference for the lived K networks (4.6).
21. Arm `Ks` of E31 at full length, and K with another rate of approach (4.7).
22. The gain of E34 under back-propagation at temperatures other than 0.5 and on more than one seed (4.8).
23. The control `A+D-light` of E33, which separates the kind of target from the amount of error (4.9).
24. The GRU with a module under an optimizer chosen for the module's interfaces (4.10).
25. A pull toward earlier weights at other strengths, on more seeds, or one that moves (4.12).
26. Plain gradient steps under the sweep at a rate chosen for the sweep (4.13).
27. E48's term with the lived weights left to decay as in E40, or E40 with the lived weights out of the optimizer, so that the two differences are separated (4.14).
28. E48's term at another rate, another decay or another starting scale of `down` (4.14).
29. The decay varied at a rate other than 0.001, at the half width, or at a quarter of the pairs (4.16; 5.12).

From Part 5 (what the briefs and reports say was not shown):

30. A replication of K's result on composition on fresh seeds (5.2).
31. Feedback of K's kind on all maps, not only the recurrent one (5.2).
32. A network born under E34's gain that then lives and composes (5.4).
33. The gain of E34 on new seeds, and combined with K during a life (5.4).
34. The six arms of E33 that ran on one seed, on the other two seeds (5.3).
35. Compositions, a new family and retention for a network with a wired module (5.5).
36. Reminders given at intervals inside a life's stream (5.6).
37. Whether a life helps with a task of another kind, counted in total experience and not in items of the new family (5.7).
38. Which part of a life gives the saving of E37 and E40 (5.7, 5.9).
39. The pulls and the rehearsal of E38 with a contradicting family on seeds 93683 and 116089 (5.8).
40. A ruling, or a second implementation, for E38's checks 2 and 3, which failed under the sweep (5.8).
41. A new family taught inside a life's stream, not alone (5.9).
42. More than one new family for a lived network (5.9, 5.13).
43. Unrelated learning over more than 4,096 updates (5.11).
44. A harder unrelated task, or one without carrying across a gap (5.11).
45. A fact stored once and recalled by a cue, as the kind of memory tested (5.11).
46. The plain-step arm under the sweep read at the same dose as under back-propagation (5.11).
47. The rehearsed unrelated arm scored between 256 and 1,024 updates (5.11).
48. E42's networks under unrelated learning on more than one seed (5.11).
49. A life without decay, with the sizes of the weights recorded (5.12).
50. A birth without decay at the half width (5.12).
51. A learner that produces E48's gate itself (5.13).
52. E48's terms against a replay store of equal memory (5.13).
53. A term of E48 kept in the standing optimizer without its family's items (5.13).
54. A task whose answers need output units of their own, with E48's frozen readout (5.13).
55. Composition across families with E48's gate (5.13).
56. A threshold located more finely than every 256 updates (5.10).
57. Back-propagation at the half width from a tenth of the pairs with a dose beyond 40,960 updates (5.10).

## What could not be found in the record

- A run that chose the learning rate, the betas or the starting weights on the dense network or for rule S. The values were carried from the windowed network of 6 November.
- A reason for the betas (0.9, 0.98) beyond the code comment "as in the grokking papers"; a reason for the byte table's width 8, for 0 to 12 letters, for the birth's 24,576 updates, for the three seeds, or for the share of one half in rehearsal on this network.
- The distributions from which the starting weights are drawn. The code leaves them to the library, and no file read states them.
- The value of epsilon in any receipt.
- The files `E26_REPORT.md` and `E26_REVIEW_SECTION.md` named in the E26 brief.
- A section "what it cannot show" in the E31 brief, and a section of limits in the per-seed and three-seed reports of E41.
- A ruling on the two checks of E38 that failed under the sweep.
- A section of the coordinator's review that reads E46, E48 or E41's later seeds in full; review 9.157 names them as "not audited". E46 and E48 are on worker branches only.
- Any run in the scope with the error cut off at earlier positions. `through_time=False` appears only as a fixture that must fail, in E26's check 2a.
