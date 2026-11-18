# E55, first part: the first family kept while a new task is learned, without a line of the forward pass that reads the item's first byte; the update gates of the cells in reserve are shut by a bias and can be opened only by learning (18 November 2026, revision 2)

**Status.** This brief was written by the outside reviewer, who acts for the coordinator; the coordinator cannot be reached while Mr. Xu is away (his instruction of 17 November: arrange it yourself). It is the first part of item 11 of `PLAN.md` beside this file, and it is run as an addition to the trial of E51 by E51's worker, on that worker's branch, with E51's births, tasks and runner. Revision 1 was checked by an agent of the reviewer that had not seen the reviewer's reasoning (`CHECK_OF_TASK_E55_BRIEF_20261118.md` beside this file, with `check_e55_brief.py`). The check found thirteen defects and gave a text for each; it also emulated the arm `shut` in its own arithmetic, and the emulation shows that the construction does not keep the first family's answers exactly. Revision 2 enters all thirteen findings and sets the chances again; the last section lists the changes. **The brief is not released by the reviewer: the coordination passes to the coordinator (Mr. Xu's message of 18 November, 00:44 UTC), who decides whether it is run as it stands, with a further arm, or not at all** (the section "For the coordinator" below). The check is not the independent audit that the programme's rules ask for. The job is at the evidence level that the coordinator's briefs call E2: a first measurement, which confirms nothing. It has one seed and eight runs and costs about 5 CPU-hours. The result is verified from the receipts and the weights by an agent of the reviewer before the reviewer reads it (rule 4 of section 4 of the plan).

## Shell names and host

`D=origin/codex/minifly-consolidated-20261023` (the default branch, at 7cabb13a); `W1=origin/claude/e51-reserve-017dSEpW` (E51's worker's branch, the branch of this job: the runner `$M/exp51_reserve.py`, SHA-256 `84d0dad7f6e5eeb15c4518a8c591eb5fc043d72b17926913a3b3fde690d1f34a`, and E51's starts, receipts and weights under `$M/exp45/e51/`); `V=origin/claude/vibrant-cerf-sje726` (the reviewer's branch: this brief, the brief of E51 `$C/TASK_E51_A_NEW_TASK_LEARNED_IN_CELLS_HELD_IN_RESERVE_20261117.md`, and the reviewer's verification of E51's trial `$R/E51_TRIAL_VERIFICATION_20261117.md`); `M=MOD97_LOCAL_RULE_20261106`; `R=PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109`; `C=$R/COORDINATION_20261117`. The host is Linux on x86-64 with Python 3.13 and torch 2.15.0+cpu, as set up for E51.

## Terms

The terms of the brief of E51 hold. Those that this brief uses are repeated here.

- *The network* reads one byte at each position of an item. It has a table of 8 numbers for each byte (the *byte table*), a layer `l1` whose output for a byte is the byte's *code* (512 values), a recurrent layer of 256 *cells*, and a readout: 512 units, each the hyperbolic tangent of a sum formed by a map `l2` from the code and a map `lm` from the cells' states, followed by a map `l3` to the 257 output classes. The *input map* takes the code to the cells and the *recurrent matrix* takes the cells' previous states to them. A cell has a reset gate r, an update gate z and a candidate n, and its new state is (1 − z) n + z h, where h is its previous state; the state is zero at the start of every item, so that a cell whose z is 1 at every position has the state zero. In the two matrices of the recurrent layer, which have 768 rows, rows 0 to 255 belong to the reset gates, rows 256 to 511 to the update gates and rows 512 to 767 to the candidates.
- *Back-propagation* (rule B) uses the exact gradient. *The sweep* (rule S) sends the error back through each linear map by the signs of the map's weights times the mean magnitude of those weights.
- *The first family* is four operations on two numbers below 89 (addition and subtraction modulo 89, the larger and the smaller of the two). *A birth* is a network trained on the first family alone. *The embedded network* of E51 is E41's birth of half the width (256 values of code, 128 cells, 256 readout units) placed inside a network of the present size: cells 0 to 127 are the *old cells* and cells 128 to 255 the *cells in reserve*; units 0 to 255 of `l1` and of the readout are *old* and units 256 to 511 *added*. The *old rows* of a matrix of the recurrent layer are the three rows of each old cell; the *gate rows of the reserve* are rows 384 to 511, the rows of the update gates of the cells in reserve.
- *The bytes in use* are the 46 bytes that occur in the first family's items. *The directions in use* at a matrix are the directions spanned by the input vectors (with a constant 1 appended) that the matrix receives on the first family's items when the states of the cells in reserve are zero: 46 at the input map, 129 at the recurrent matrix and at `lm`, 11 at `l2`. The reviewer's verification found that they are the codes of the bytes in use (input map), the coordinates of the 128 old cells (recurrent matrix and `lm`) and the codes of `=` and of the ten digits (`l2`), each with a constant. A change of a row of a matrix *away from the directions in use* is a change that leaves the row's summed input unchanged for every input vector in their span.
- *The rule* of E51 projects, at every update, the gradient and then the step of chosen rows of the four matrices away from the directions in use, and restores after the step what must not change (`l1`, the old columns and the bias of `l3`, the rows of the byte table of the bytes in use).
- *The two added lines* of E51's forward pass read the first byte of each item and, when it is a byte in use, set the update gates of the cells in reserve to 1 and the added units of the readout to zero.
- *The two tasks* are E51's: `unrelated` (E43's task on 98 bytes that the first family does not use: a code byte, then two of 64 symbols, then 0 to 12 fillers, then a query byte; the answer is the two symbols) and `conflict` (the family of the same four operations modulo 97, whose items are made of bytes in use except for their first byte, which names the operation).
- A *scored row* is the set of scores taken after 0, 16, 32, 64, 128, 256, 512, 1,024, 2,048 and 4,096 updates in E51, and in this job also after 8,192 and 16,384. The *reduction score* is the percentage of right answers on the pairs of addition and subtraction of the first family whose answer needs the reduction modulo 89 (7,832 pairs, without letters). The *never-taught score* of a new task is the percentage of right answers on its items that were never taught (300 for `unrelated`, 800 for `conflict`). A new task is *learned from row g* if its never-taught score is at 80 or more at scored row g and at every later one.
- An update gate is *shut* when its z is close to 1 and *open by x* when 1 − z is x. A cell in reserve is *open at some position* on a set of items if the mean over the items of the largest 1 − z over the positions of the item exceeds 0.1.
- *The plain forward pass* is `exp51_reserve.fwd` without the two added lines (`KEY["in_use"]` is `None`); *the keyed forward pass* is the same with them. *The keyed start* is E51's embedded network as saved before any update.
- A *panel* is a fixed set of items on which a score is taken; the *held-out pairs* of the first family are the 1,200 pairs of its four operations that were never taught, written with 12 *letters* (capital letters between the operands that carry no information). The *stream* of a task is the sequence of its taught items in the order in which the updates receive them, and the *data digest* of a scored row is the SHA-256 over all batches received so far. Under *teacher forcing* the network is given the right answer's earlier bytes while it produces the answer.
- A *receipt* is the file that a run or a check writes with its settings, its scored rows and the hashes of its sources. A *dry run* is a run made by a check.

## The question

Mr. Xu's requirement (17 November) is that long-term memory form by itself during learning and not by an intervention of ours; some rehearsal is acceptable. In the trial of E51 the first family's 32,884 scored answers stayed the same, item for item, while each of two new tasks was learned, without rehearsal. The reviewer's verification states what did this: the forward pass reads each item's first byte and, for a byte in use, holds the cells in reserve and the added readout units at zero; with these two lines removed, the same trained weights answer none of the first family's reduction pairs after `unrelated` and at most 0.4% after `conflict`. The two tasks are told apart by two lines of the program; this is not learned and is not in the weights.

**Is the first family kept and the new task learned when no line of the forward pass tells the two tasks apart? If the update gates of the cells in reserve are shut by a bias, and the rule allows the rows of these gates to change only away from the directions in use, so that no first-family item can open them while learning on other items can, is the first family kept and is each new task learned, with the forward pass of the plain network?**

This removes one of the acts of the program and not all of them; the section "What remains supplied by the program" lists the others.

## What the record says

Every run of the record has seed 73129, 4,096 updates, no weight decay and a new optimizer at the start of the new task. Figures are given as sweep; back-propagation.

- **E51's trial** (`$W1:$M/exp45/e51/`; verified by the reviewer's agent from receipts and weights, `$V:$R/E51_TRIAL_VERIFICATION_20261117.md`). Reduction score at row 0: 92.4; 95.3.

| Arm of E51 | What it is | First family at row 4,096 (reduction score) | `unrelated` learned from row | `conflict` learned from row | Cells in reserve whose mean of 1 − z over all positions and items of the new task's panel exceeds 0.1, at row 4,096 (`unrelated`; `conflict`) |
|---|---|---|---|---|---|
| `keyed` | the two added lines and the rule | 92.4; 95.3 on both tasks; no answer of 32,884 changed at any row | 1,024; 512 | 1,024; 512 | 119; 125 and 112; 83 of 128 |
| `keyed-free` | the two added lines, no rule | `unrelated` 0.0; 0.0. `conflict` 1.0; 0.2 | 512; 512 | 2,048; 1,024 | not used here |
| `open` | no added line, no rule; the update gates of the reserve start shut by a bias of 10 and are free | `unrelated` 0.0; 0.0 (lost by row 16). `conflict` 1.3; 0.0 | 512; 512 | 1,024; 1,024 | 1; 8 and 0; 0 of 128 |
| `rehearse` | as `open`, with each item drawn with equal chance from the new task's stream or from the first family's taught half | `unrelated` 90.0; 92.8. `conflict` 88.3; 97.8; of the 31,684 answers, 9.6%; 6.7% differ from row 0's on `unrelated` and 9.8%; 4.1% on `conflict` | not used here | not used here | not used here |

- **Three statements of the verification** bear on this job. First, in `keyed` the answers are unchanged and the outputs are not: by row 4,096 they have moved by up to 5.3e-4, which the verification attributes to the storing of the weights in single precision after each projected step, by an estimate of size; the smallest difference between the largest and the second largest output of a first-family answer is 1.8e-4 under the sweep. Second, at the recurrent matrix and at `lm` the rule is the same as leaving unchanged the old units' weights from old cells and their biases. Third, a gate with a bias of 10 has 1 − z = 4.5e-5, so that at each position the state takes 4.5e-5 of the candidate: in the start of the arm `open` the largest state of a cell in reserve on the first family's held-out items is 0.0015; 0.0024, the outputs differ from the birth's by up to 0.0029; 0.0034, and 1; 0 of the 32,884 answers differ.
- **In the arm `open` the gates opened on `unrelated` and not on `conflict`.** The receipts' count, the number of cells whose mean of 1 − z over all positions and items exceeds 0.1, is 1; 8 on `unrelated` and 0; 0 on `conflict`. It does not count a cell that opens at one position. In the weights of row 4,096 on `unrelated`, 102; 75 of the 128 cells in reserve open by more than 0.5 at some position of an average never-taught item, most of them at the position of the first symbol, and with the cells in reserve held shut the never-taught score is 0.7; 5.7 against 100; 100. On `conflict` no cell does, and the score with the cells held shut is 83.0; 87.6 against 85.0; 94.9: that task was learned in the old cells, whose weights were free. The first family was lost on both tasks. The record therefore shows that learning opens gates that start shut by a bias of 10 when their rows and all other weights are free (the largest mean opening of a cell rises from 1e-4 at row 0 to 0.13 at row 256 under the sweep and to 0.15 at row 128 under back-propagation). It does not show it for gate rows that may change only away from the directions in use, nor when the old cells' weights are protected. (Revision 1 of this brief said that the gates did not open in this arm; the check found that wrong for `unrelated`, and the reviewer confirmed it from the receipts and by running the check's part `open`.)
- **An emulation of the arm `shut`, made for the check of this brief** (`CHECK_OF_TASK_E55_BRIEF_20261118.md`, part `emulate`; the checker's own arithmetic in double precision, validated on 64 updates of E51's `keyed` to 4.7e-5 an entry; it is not a run of the runner, the reviewer has not repeated it, and its figures show sizes and the order of events only). The gates stay shut on the first family's items throughout (1 − z at most 5.2e-5). On `unrelated` under back-propagation they open on the new task's items after about 1,000 updates, the never-taught score is 72.3 at row 2,048 and 99.3 at row 4,096 with 93 cells open by more than 0.5 at some position, and 20 to 34 of the 31,684 answers of the first family differ from row 0's at the scored rows from 256 on. Under the sweep the same task is at 0.0 at row 4,096, the cells opening only after row 2,048, and 70 to 104 answers differ. On `conflict` the never-taught score at row 4,096 is 31.0 under the sweep, which is what the network gives with the cells in reserve held shut (30.0), and 54.1 under back-propagation (29.5 with the cells held shut), where gates begin to open after row 1,024; 376 to 924 answers differ under the sweep and 13 to 244 under back-propagation. The reduction score of the first family stays within 1.3 points of row 0's in all four emulated runs; where it moves, it rises.

## The change

Everything is E51's arm `keyed` (the births, the embedded network, the tasks and their streams, the optimizer, the scores) but four things.

1. **The forward pass has no added line.** It is `exp51_reserve.fwd` with `KEY["in_use"]` set to `None`, in the update and in the scoring. The second scoring of the new task with the cells in reserve held shut on every item (`KEY["shut_all"]`) is kept as a measurement, as in E51.
2. **The start `shut`.** The keyed start of E51 (`e51_start_keyed_<rule>_73129.pt`, with its saved directions `e51_start_keyed_<rule>_73129_directions.pt`) is changed in four blocks of rows, each with its bias appended as a last column: the gate rows of the reserve in the input map and in the recurrent matrix, and rows 256 to 511 of `lm` and of `l2`. Each block is replaced by its projection away from the directions in use of its matrix (`exp51_reserve.away`, in double precision), and then 10 is added to the bias of the input map at the gate rows of the reserve. After this, for every input vector in the span of the directions in use, an update gate of the reserve receives 10 from the input map and nothing from the recurrent matrix, and an added readout unit receives nothing from `l2` and nothing from `lm`. On other directions these rows keep the values drawn when the embedded network was made: on the part of the code of a byte not in use that lies outside the directions in use, and on the states of the cells in reserve.
3. **The rule with more rows.** E51's rule projects the old rows of the two matrices of the recurrent layer and rows 0 to 255 of `lm` and `l2`. In the arm `shut` it projects, in addition, the gate rows of the reserve in both matrices of the recurrent layer and rows 256 to 511 of `lm` and `l2`. The reset rows and the candidate rows of the cells in reserve stay free, so that a cell in reserve may read the old cells and the codes of the bytes in use through its candidate; what it cannot do is open on an item that presents nothing outside the directions in use. At the start the gates are shut on the new tasks' items as well (1 − z at most 5.7e-5). In `unrelated` every byte but the start symbol is outside the bytes in use, and a gate can be opened at any position through the part of the byte's code outside the directions in use (2-norm 0.4 to 2.2 of 8.1). In `conflict` this holds for the first byte only (2-norm 0.6 to 1.7). At every later position of an item of `conflict` a gate of the reserve receives 10 from the input map and nothing from the old cells; it can be held open there only by the weights of the gate rows from the cells in reserve (columns 128 to 255 of the recurrent matrix), acting on the states that the cells took at the first byte. A cell that opens at the first byte alone holds a function of the operation byte for the rest of the item.

4. **The number of updates is 16,384,** four times E51's, and the scored rows 8,192 and 16,384 are added to E51's ten. In E51's `keyed` the gates of the reserve were about half open on the new task's items from the start, and the tasks were learned from row 512 or 1,024. Here every gate must first be moved from a summed input of 10, which at steps of at most 0.001 a weight takes at least 245 to 1,507 updates for one byte, and in the emulation made for the check the gates began to open between rows 1,024 and 4,096. Rule 1 of section 4 of the plan asks that a setting that was carried over be varied before a conclusion rests on it.

The directions are those saved with E51's keyed start. They are not computed again on the start `shut`: there the cells in reserve have states of the order of 1e-3 on the first family's items, which a new computation would count as directions in use and so protect every weight from a cell in reserve.

What follows from the construction, for every item made of bytes in use only: at every position an update gate of the reserve receives 10 and whatever the cells in reserve send it through the recurrent matrix; as long as the states of the cells in reserve are small, the gate stays at 1 − 4.5e-5 and the states stay small. The construction does not make them zero, and it does not bound what the cells in reserve send to the old cells and to the readout, whose weights from cells in reserve are free. The gradient of such a weight is an error times a state of about 1e-3, and AdamW moves a weight by up to 0.001 an update whatever the size of its gradient, so that after g updates an old cell or a readout unit can receive up to 128 × 0.001 g × 1e-3 from the cells in reserve. In an emulation made for the check of this brief the first family's outputs moved by 0.007 to 0.026 in eight updates, and between 13 and 924 of the 31,684 answers without letters differed from row 0's at the scored rows from 256 to 4,096. How far this goes in the 16,384 updates of the runner is the first thing the job measures.

## Arms and runs

| Arm | Start | Rows projected by the rule | What it shows |
|---|---|---|---|
| `shut` | `shut` | E51's rows, the gate rows of the reserve, rows 256 to 511 of `lm` and `l2` | whether the first family is kept and the new task learned without an added line |
| `shut-gates-free` | `shut` | E51's rows and rows 256 to 511 of `lm` and `l2`; the gate rows of the reserve are free | whether the protection of the gate rows is needed |

Two arms, two tasks and two rules with seed 73129 give eight runs of 16,384 updates, each network under the rule of its birth. The scored rows are E51's ten and the rows 8,192 and 16,384; the weights of rows 64, 4,096 and 16,384 are committed. E51's runs of `keyed`, `open` and `rehearse` of the same task, rule and seed are the references; none is run again. The sixteen runs of E51 cost 2.74 CPU-hours together; a run of this job has four times the updates and two more scored rows and costs about 0.55 CPU-hours, so that the eight runs cost about 4.5 CPU-hours and the checks about 0.4.

## The runner

The runner is a new file, `exp55_shut.py`, in `$M` on the branch `$W1`; no existing file is edited. It imports `exp51_reserve` as `R51` and asserts that file's SHA-256 (above) when it is imported. It is called from `$M` as

    python3 exp55_shut.py start <S|B> 73129
    python3 exp55_shut.py run <unrelated|conflict> <shut|shut-gates-free> <S|B> 73129 [--updates N]
    python3 exp55_shut.py check <1|2|3> 73129

and writes under `$M/exp45/e55/` (checks under `$M/exp45/e55/checks/`).

1. **The start.**

       Z_RES = T.arange(R51.U1 + R51.U0, 2 * R51.U1)        # rows 384 to 511: the update gates of the cells in reserve
       ADDED = T.arange(R51.H0, R51.H1)                     # rows 256 to 511 of lm and of l2
       GATE_BIAS = 10.0

       @T.no_grad()
       def shut_start(m, dirs):
           for W, b, Vm, rows in ((m.rnn.weight_ih_l0, m.rnn.bias_ih_l0, dirs["ih"], Z_RES), (m.rnn.weight_hh_l0, m.rnn.bias_hh_l0, dirs["hh"], Z_RES),
                                  (m.lm.weight, m.lm.bias, dirs["lm"], ADDED), (m.l2.weight, m.l2.bias, dirs["l2"], ADDED)):
               D = R51.away(T.cat([W[rows], b[rows][:, None]], 1).double(), Vm)
               W[rows] = D[:, :-1].float(); b[rows] = D[:, -1].float()
           m.rnn.weight_hh_l0[Z_RES, :R51.U0] = 0.0; m.lm.weight[ADDED, :R51.U0] = 0.0   # the projection leaves remainders of 1e-16 here; the sweep sends the error back by the sign of a weight
           m.rnn.bias_ih_l0[Z_RES] += GATE_BIAS

       def load_shut(rule, seed):
           m, m0, in_use, dirs, rec = R51.load_start("keyed", rule, seed)     # asserts E51's committed keyed start and its saved directions
           R51.install(None)                                                   # the forward without the added lines
           shut_start(m, dirs)
           return m, in_use, dirs, rec

   `start` builds the start with `load_shut`, saves it as `e55_start_shut_<rule>_73129.pt` and writes `e55_start_shut_<rule>_73129.json` with the hash of its tensors (`R51.R.tensors_sha`), the SHA-256 of the file, the record that `R51.load_start` returns, `GATE_BIAS`, and the provenance. Every run and every check builds the start again with `load_shut` and asserts the hash of its tensors against that receipt.
2. **The rule.**

       class Rule55(R51.Rule):
           def __init__(self, m, dirs, in_use, arm):
               super().__init__(m, dirs, in_use, True)
               self.arm = arm
               cells = T.cat([R51.OLD_ROWS, Z_RES]) if arm == "shut" else R51.OLD_ROWS
               units = T.arange(R51.H1)
               self.mats = [(m.rnn.weight_ih_l0, m.rnn.bias_ih_l0, dirs["ih"], cells), (m.rnn.weight_hh_l0, m.rnn.bias_hh_l0, dirs["hh"], cells),
                            (m.lm.weight, m.lm.bias, dirs["lm"], units), (m.l2.weight, m.l2.bias, dirs["l2"], units)]

           @T.no_grad()
           def after(self):
               super().after()
               self.m.lm.weight[ADDED, :R51.U0] = 0.0
               if self.arm == "shut":
                   self.m.rnn.weight_hh_l0[Z_RES, :R51.U0] = 0.0

   `R51.Rule.before` is used as it is and `R51.Rule.after` with the addition above; both act on `self.mats`. The two blocks that the projection brings to zero (the weights from the old cells into the gate rows of the reserve and into rows 256 to 511 of `lm`) are set to exactly zero, because the projection leaves remainders of the order of 1e-16 with either sign, which the forward pass does not see and through whose signs the sweep would send the error back.
3. **The gates on a set of items.**

       @T.no_grad()
       def gates_on(m, items):
           """Teacher forcing, the plain forward. Over the cells in reserve: the largest of 1 - z over cells, positions and items; the number of
           cells whose mean of 1 - z over positions and items exceeds 0.1; the number of cells for which the mean over items of the largest
           1 - z over the positions of the item exceeds 0.1, and the same number for 0.5; the largest magnitude of a state; and the largest
           magnitude of an added readout unit at the answer positions."""

   It groups the items by length as `R51.reserve_figures` does, takes `z`, `h` and `z2` from the second value that `R51.fwd(m, x, False, True)` returns, and returns the six figures as `largest_opening`, `cells_open_by_more_than_0p1`, `cells_open_at_some_position`, `cells_open_by_more_than_0p5_at_some_position`, `largest_reserve_state` and `largest_added_unit_at_answer_positions`.
4. **The run** is a copy of the text of `R51.run` (lines 452 to 549 of `exp51_reserve.py`) with the changes below and no others; every name of `exp51_reserve.py` that the text uses is taken from `R51`. The report shows the difference between the two texts.
   - The signature is `run(task, arm, rule, seed, out, cmd, updates=4 * R51.DOSE)`; the scored rows are `R51.ROWS + (8192, 16384)`, the weights of the rows `R51.KEEP_ROWS + (16384,)` are kept, and wherever the text compares `updates` with `DOSE` it compares with `4 * R51.DOSE`; the run asserts `arm in ("shut", "shut-gates-free")`. The branch for `start == "e26"` is left out, and what the text does `if start != "e26"` is done always.
   - The tag is `e55_<task>_<arm>_<rule>_<seed>`, with `_dry<N>` for a run of N updates other than 16,384; the folder is `$M/exp45/e55`.
   - The start: `m, in_use, dirs, start_rec = load_shut(rule, seed)`, with the assertion of point 1.
   - The rule: `ruleobj = Rule55(m, dirs, in_use, arm)`.
   - The stream: `nxt, stream_note = R51.item_source(task, "keyed", seed)`, the new task's stream alone, which is the stream of E51's arm `keyed`.
   - In `score_row`, after the entry `reserve`: `r["gates_on_first_family"] = gates_on(m, s2_items)` (the 1,200 held-out pairs at 12 letters) and `r["gates_on_new_task"] = gates_on(m, reserve_panel)` (the new task's never-taught items).
   - The variable `cfg` is not used. The settings: `experiment="E55"`, `part="first"`, `start_form="shut"`, `keyed_forward=False`, `rule_applied=True`, `restore_book=True`, the new entries `rule_rows` (the arm's row sets in words) and `gate_bias=GATE_BIAS`; the entries `update` and `forward` say that the forward is `exp51_reserve.fwd` without a key and that the rule is `exp55_shut.Rule55`; `provenance` adds the SHA-256 of `exp55_shut.py`.
   - The assertion before the first row is `assert R51.KEY["in_use"] is None`.
   - In `write`, `experiment="E55"`. The settings add `start_shut=dict(file=..., sha256=..., tensors_sha256=...)` from the receipt of the start `shut`; the entry `start` keeps the record of the keyed start from which it was made. The runner calls `T.set_num_threads(1)` before anything else when it is called as a program, as `exp51_reserve.py` does on its line 706.
5. **Files committed:** the two start files with their receipts, the receipts of the checks and of the eight runs, the weights of rows 64, 4,096 and 16,384 of every run and the files of generated bytes that the run writes, as in E51 (the weights added with `git add -f`), the receipts and the files of generated bytes of the four dry runs of check 3 (their weights under `rows_uncommitted` are not committed), and the logs.

## Checks before the runs

Each check writes a receipt `$M/exp45/e55/checks/e55_check_<n>_73129.json` with an entry `ok`. No run is started before the three checks have passed. If a check fails, the worker stops, reports the first difference and asks. The panels are E51's (`R51.check_panels(73129)`): the 1,200 held-out pairs at 12 letters and the 2,000 items of E51's check 1.

1. **The start,** for each rule.
   (a) Every tensor of the start `shut` is bit-equal to the keyed start's outside the four blocks (rows 384 to 511 of `rnn.weight_ih_l0`, `rnn.bias_ih_l0`, `rnn.weight_hh_l0`, `rnn.bias_hh_l0`; rows 256 to 511 of `lm.weight`, `lm.bias`, `l2.weight`, `l2.bias`). The two blocks that the projection brings to zero (rows 384 to 511, columns 0 to 127 of `rnn.weight_hh_l0`; rows 256 to 511, columns 0 to 127 of `lm.weight`) are exactly zero.
   (b) For each block, with `D` the block and its bias in double precision, and with 10 subtracted from the bias column of the gate rows of the input map, the largest magnitude of an entry of `D @ Vm` is at most 1e-5.
   (c) With `R51.KEY["in_use"]` set to `None` and asserted so, on the two panels, by `gates_on`: the largest of 1 − z of a cell in reserve lies between 4.5e-5 and 4.6e-5; the largest magnitude of a state of a cell in reserve lies between 1e-4 and 1.1e-3; the largest magnitude of an added readout unit at the answer positions is above 0 and at most 1e-3. The largest difference of an output at an answer position from that of the keyed start under the keyed forward pass (`R51.KEY["in_use"]` set to the bytes in use for the reference only) is recorded and is above 0 and at most 0.02 (the limit of E51's check 1 for its start with a bias, which gave 0.0029 and 0.0034).
   (d) The number of the 32,884 generated answers of the first family (the 31,684 items without letters and the 1,200 held-out pairs at 12 letters) that differ from those of the keyed start under the keyed forward is recorded and is at most 5 (E51's start with a bias: 1 and 0). This figure is a record; it does not test the projection.
2. **The rule,** for each rule, each task and each arm: eight updates on the task's stream from the start `shut`, made as in `exp51_reserve.check3` (lines 599 to 606).
   (a) In both arms: `l1.weight`, `l1.bias`, `l3.weight[:, :256]`, `l3.bias` and the rows of the byte table of the bytes in use are bit-equal to the start's; for each of the four matrices and the rows that the arm projects, the largest magnitude of an entry of `(D_after − D_start) @ Vm` is at most 1e-5; each of the four matrices has changed; the two blocks named in check 1 (a) are exactly zero (the first of them in the arm `shut` only).
   (b) In the arm `shut`, on the two panels: the largest of 1 − z of a cell in reserve lies between 4.4e-5 and 1e-4, and the largest magnitude of a state of a cell in reserve is at most 1.1e-3 (22 positions times 1 − z). The largest change of an output at an answer position from the start's and the largest magnitude of an added readout unit at the answer positions are recorded and have no limit: the weights from the cells in reserve to the old cells and to the readout are free, and the optimizer moves each of them by up to 0.001 an update. The check of this brief found, in an emulation, a change of an output of 0.007 to 0.026 after eight updates.
   (c) A case in which a difference must be found: in the arm `shut-gates-free` the largest magnitude of an entry of `(D_after − D_start) @ Vm` at the gate rows of the reserve, in the input map or in the recurrent matrix, exceeds 1e-4. If it does not, the worker reports the figure and asks, because the gradient that reaches a shut gate may be too small for the optimizer to act on, and the arm would then not differ from `shut`.
   (d) In the arm `shut`: what an update gate of the reserve receives from the input map on the code of each byte of the new task (the 98 bytes of `unrelated`; `r`, `t`, `x`, `n` of `conflict`), after the eighth update minus at the start, for each of the 128 gates and each byte: the mean magnitude and the largest magnitude of this change are recorded. If the largest magnitude is below 0.01 (the check of this brief found 0.08 to 0.17 in an emulation), the worker reports the figure and asks. The gradient of these rows at the first update, after the rule's projection, is recorded with its median magnitude.
3. **The stream and the cost.** For each rule and task, a dry run of 64 updates of the arm `shut`: its data digest at row 64 equals that of E51's `keyed` run of the same task and rule at row 64. From the four dry runs: CPU-seconds an update and a scored row, and a forecast for the eight runs. If the forecast exceeds 8 CPU-hours, the worker asks before starting.

## How the result is stated (fixed beforehand)

For each of the eight runs the report gives a table of the twelve scored rows with: the reduction score; the number of the 31,684 answers without letters and of the 1,200 held-out answers at 12 letters that differ from row 0's; the never-taught score of the new task, and the same with the cells in reserve held shut; the number of cells in reserve open at some position on the new task's never-taught items, the number open by more than 0.1 in the mean, and the largest 1 − z there; and the six figures of `gates_on` on the first family's held-out items. Beside it stand the rows of E51's `keyed`, `open` and `rehearse` of the same task and rule.

- **Words for the first family in one run.** With d(g) the number of the 31,684 answers without letters that differ from row 0's at scored row g: the first family is *kept exactly* if d(g) is 0 at every row and no held-out answer differs at any row; *kept* if it is not kept exactly, d(g) is at most 31 at every row, at most 1 of the 1,200 held-out answers differs from row 0's at every row (0.1% of each set), and the reduction score at every row is within 0.5 points of row 0's; *lost* if the reduction score at row 16,384 is below half of row 0's; and *partly kept* otherwise.
- **Words for the new task in one run.** *Learned from row g*, as defined in "Terms"; otherwise *not learned*, with the never-taught score at row 4,096.
- **What the report may say for the arm `shut`, for seed 73129 and one rule and task.** If the task is learned and the first family is kept exactly or kept: "the first family was kept while the task was learned, with the plain forward pass", with the number of cells open at some position on the new task's items and the score with the cells in reserve held shut. If no cell is open at some position, or that score is at 80 or more, the report adds "the task was learned without the cells in reserve opening", and the run is not read as a success of the gates. If the task is learned and the first family is partly kept, the report gives the largest d(g) over the rows as a number and as a percentage of 31,684, the largest number of held-out answers that differ, and the reduction score at the last row beside row 0's, and says "the first family was kept except for x% of its answers". If the first family is lost, the report says "the first family was not kept". In both cases it gives the row at which d(g) first exceeds 31 and the figures of `gates_on` at that row, which show whether the gates opened on the first family's items or stayed shut while the weights from the cells in reserve grew. If the task is not learned: "the task was not learned within 16,384 updates", with the never-taught score at row 16,384 and, for every scored row, the largest 1 − z and the number of cells open at some position on the new task's never-taught items. If no cell is open at some position at any row, the report adds "learning did not open the cells in reserve". If cells opened, the report gives their number and says that the task was not learned although they opened. The first family's word is then given and is not read as a success of the construction.
- **The arm `shut-gates-free`** is read beside `shut`. If, for a rule and task, the task is learned in both arms and the first family is kept exactly or kept in both, the report says that the protection of the gate rows was not needed in that run. If the first family is kept in `shut` and partly kept or lost in `shut-gates-free`, the report says that the protection of the gate rows was needed, with the figures of `gates_on`. If the task is learned in neither arm, the arms are not compared, and the report says that the cause is not identified by this job: both arms share the start shut by a bias, the protected rows 256 to 511 of `lm` and `l2`, and the 4,096 updates.
- **The number of updates.** Where a task is learned in `shut`, the row from which it is learned is set beside that of E51's `keyed` and `open`; the rows double from one to the next, so that a ratio of 2 is one row.
- **The score with the cells in reserve held shut** is given as a figure. The report does not say that a task is "learned in the reserve"; the verification of E51 found that the criterion of that name establishes less than the name says.
- **What remains supplied by the program.** The report repeats the next section in its own summary, so that no reader takes the result for more than it is.

## What remains supplied by the program

- The directions in use are computed once, by the program, from 2,080 taught items of the first family, before the new task begins. The network does not keep them during its own learning.
- The rows that the rule protects are named by the program: the old rows, the gate rows of the reserve, and every row of `lm` and of `l2`. `l1`, the old columns and the bias of `l3`, and the rows of the byte table of the bytes in use are restored by the program after each step.
- The cells in reserve and the added units are put in by the program at the change of task, and the program shuts the gates with a bias of 10.
- The change of task is known to the program: the directions are fixed and a new optimizer is made at that moment.

What the job removes is the reading of the item by the forward pass. If the result is positive, the next parts of item 11 take the other points in turn: a record of the directions in use that the network keeps while it learns, cells in reserve that are present from before the birth, and learning without a marked change of task.

## Predictions (the reviewer's, recorded before any run)

The chances were set again after the check, for runs of 16,384 updates. The reviewer had then seen the check's emulation of the arm `shut` to 4,096 updates, so that the chances of P55.2 to P55.6 and of P55.8 are not independent of it; no run of the runner had been made.

| | Statement | Chance |
|---|---|---|
| P55.1 | Checks 1 to 3 pass at the first attempt | 60% |
| P55.2 | In the arm `shut`, `unrelated` is learned under both rules | 45% |
| P55.2b | In the arm `shut`, `unrelated` is learned under back-propagation | 80% |
| P55.3 | In the arm `shut`, `conflict` is learned under both rules | 5% |
| P55.4 | In the arm `shut`, the first family is kept exactly or kept in all four runs | 3% |
| P55.4b | In the arm `shut`, the reduction score of the first family is within 2 points of row 0's at every scored row of all four runs | 50% |
| P55.5 | In the arm `shut`, the largest 1 − z of a cell in reserve on the first family's held-out items is below 1e-3 at every scored row of every run | 85% |
| P55.6 | In the arm `shut`, every run in which the task is learned has at least 32 cells in reserve open at some position on the new task's items at row 16,384 (the statement is not scored if no run learns its task) | 80% |
| P55.7 | In the arm `shut-gates-free`, the first family is partly kept or lost on `conflict` under both rules | 70% |
| P55.8 | In the arm `shut`, where `unrelated` is learned, it is learned from row 4,096 or earlier (the statement is not scored if it is learned under neither rule) | 55% |

## What it cannot show

- The job has one seed, two tasks and 16,384 updates. In E51's arm `keyed` the first family's outputs moved by up to 5.3e-4 in 4,096 updates through the rounding of the stored weights; here the cells in reserve add their small states. Nothing bounds either over a longer run.
- The job does not test a third task. A second new task would present items outside the directions in use and could open the cells that the first new task uses.
- The job does not test an item of the first family that contains a byte not in use.
- The gradient that reaches a shut gate carries the factor z (1 − z) = 4.5e-5. At the gate rows of the arm `shut` it is of the size of the optimizer's eps (1e-8) and under the sweep on `conflict` below it, so that the optimizer's steps there are a fraction of the rate. A task that is not learned within 16,384 updates may be one that a smaller eps or more updates would learn.
- The points of "What remains supplied by the program".
- The job uses the learned code of E41's births. Whether a code in which a byte has values of its own (E50) lets the directions in use be replaced by a record kept for each input line is the subject of the next part.

## For the coordinator

This brief was not released. Three things should be known before it is.

1. **What the check's emulation indicates.** With the construction as written, the gates of the reserve stay shut on the first family's items, but the first family's answers are not kept exactly: the weights from the cells in reserve into the old cells and the readout are free, the optimizer moves them at its full rate, and they multiply states of about 1e-3. In the emulation the reduction score stays within 1.3 points, 13 to 924 of the 31,684 answers change, and only the unrelated task under back-propagation is learned within 4,096 updates. The reviewer ran the check's parts `open`, `start` and `check2` again and obtained the check's figures; it did not repeat the emulation.
2. **A further arm that the reviewer would add, and has not specified or had checked.** If the reset rows and the candidate rows of the cells in reserve are projected as the gate rows are, and no bias is added, a cell in reserve receives nothing on an item of the first family, its candidate is zero, and its state is zero up to the rounding of the projection, whatever its gate does. The first family would then be kept as in E51's arm `keyed`, without the two added lines and without a gate that must first be opened. The price is that a cell in reserve can then read only what the first family does not present: every byte of the unrelated task, and the operation byte alone of the family modulo 97, so that for that family the old cells would have to do the work under an input from the reserve that is constant within an item.
3. **Why the plan named E50's code for this item.** With the learned code the code of a byte that the first family does not use lies mostly inside the directions in use: the part outside them has a 2-norm of 0.4 to 2.2, where the whole code has about 8.1 (the check's part `start`). The reviewer's reading, which is not measured, is that this follows from all codes being made from a table of 8 numbers a byte. The projection leaves the reserve only that small part of a new byte's code to read, which is why the gates open late. In a code in which each byte has values of its own, a new byte is wholly outside the directions in use, the protected weights are those from the values of the bytes in use, and no projection is needed. That design needs a new birth, with values of the code kept free for later bytes; it is the stronger test and the larger job.

The reviewer's recommendation is to run this brief as it stands only if the result of the actual runner is wanted for the record, and otherwise to have the arm of point 2 specified and checked first; the eight runs of this brief cost about 5 CPU-hours either way.

## Rules for the worker

- Work on your branch `$W1`, after Part B of E51 is finished and reported; this job does not interrupt Part B. Fetch the reviewer's branch to read this brief: `git fetch origin '+refs/heads/claude/vibrant-cerf-sje726:refs/remotes/origin/claude/vibrant-cerf-sje726'`, then `git show origin/claude/vibrant-cerf-sje726:<path>`. Use the environment of E51. Run at most three processes at a time, each with one thread, from the folder `$M`.
- **This host stops a machine about five minutes after its session's turn ends (a turn is the time in which the worker acts without waiting for a message), and it has also restarted a machine while a turn was open. The disk is kept, and processes are terminated.** While runs proceed, do not end your turn: issue a command that waits for the processes and repeat it until they have finished; commit and push the logs and the receipts between two such commands. The checks take about a quarter of an hour and the eight runs about one and a half hours with three processes at a time. A run that was interrupted is started again from its beginning, after its incomplete receipt and any file it wrote have been moved to `$M/exp45/e55/interrupted/`; the report says which runs were started again.
- Push only to your own branch. Never push to the default branch or to any other branch, and open no pull request.
- Every commit carries the trailers that `PROJECT_DOCUMENTS/KB/AGENTS_POINTER.md` prescribes. Apart from the trailers, no name of any language model appears in a committed file or a commit message.
- The report, `$M/exp45/e55/E55_REPORT.md`, is written in standard academic English: full sentences, every term defined at first use, every figure with the receipt or the command that gives it; the committed files are listed from `git ls-tree`.
- If the brief cannot be followed at some point, stop that part, say what you found, and ask.

## Reporting

Send the reviewer a short message with the tool `send_message` of the claude-code-remote server, to session `session_01YBMaMvtGtQvDFcgppgThrZ`, at each of these points: the three checks passed (or which failed), with the figures of check 1 (c) and (d); the four runs of the arm `shut` finished, with each run's words, its largest d(g), its never-taught scores and its numbers of cells open; the eight runs finished and the report committed. Each message gives your branch, the commit and the main figures in at most ten lines. If you need a decision, ask in the same way and go on with whatever does not depend on the answer.

## Done

The job is done when the two start files with their receipts, the receipts of the three checks, the eight runs with their receipts, weights of rows 64, 4,096 and 16,384, files of generated bytes and logs, and `E55_REPORT.md` exist; the report holds the tables of the eight runs beside E51's rows, the words fixed above, the section on what remains supplied by the program, the outcome of each prediction, the difference between the runner's `run` and `exp51_reserve.run`, and the list of committed files.

## History of the revisions

Revision 2 enters the thirteen findings of the check of revision 1 (`CHECK_OF_TASK_E55_BRIEF_20261118.md`), each with the text that the check gave. The check's document was completed after the reviewer's first entry of its findings; the findings numbered 4 (its fifth sentence) and 5 below, and the mark of 0.1 in finding 3, were entered from the completed document on 18 November at about 01:35 UTC.

| | Finding of the check | What was done |
|---|---|---|
| 1 | Check 2 (b) could not be met: the first family's outputs move by 0.007 to 0.026 in eight updates, through the free weights from the cells in reserve | Check 2 (b) rewritten without that limit; the path and its size stated in "The change"; the chances set again |
| 2 | The record said that the gates did not open in E51's arm `open`; on the unrelated task they opened and the task needed them | The point of the record replaced; the emulation entered beside it |
| 3 | The count of cells "open by more than 0.1" is a mean over positions and cannot show a cell that opens at one position | `gates_on` records the cells open at some position (by more than 0.1, and by more than 0.5), on the new task's items as well; the readings and P55.5 and P55.6 use it |
| 4 | Five sentences of "How the result is stated" could be licensed without support | The five rewritten; *kept* also asks for the held-out answers; a run that is partly kept is reported by the share of answers changed |
| 5 | The 4,096 updates were carried over from E51, where the gates started half open; here they must first be opened | 16,384 updates, with the scored rows 8,192 and 16,384; costs, limits and chances set again |
| 6 | Under the sweep the error was sent back through the signs of rounding remainders where the projection gives zero | The two blocks are set to exactly zero at the start and after every update; checks 1 (a) and 2 (a) assert it |
| 7 | Nothing checked that the optimizer acts on the gate rows of the arm `shut`, where the gradient is of the size of its eps | Check 2 (d); a point in "What it cannot show" |
| 8 | The brief did not say what can hold a gate open after the first byte of an item of the family modulo 97 | Stated in "The change" |
| 9 | Lines of the runner were not covered by the listed changes | Entered in the runner section |
| 10 | Checks 1 (c) and 2 (b) passed if the two added lines were still installed, and two limits could not fail | Check 1 (c) rewritten with two-sided limits and an assertion that the key is absent |
| 11 | Five small disagreements with the record | Corrected |
| 12 | Two predictions were true by their own clause when no run learns | "Not scored" in that case |
| 13 | Fourteen sentences or terms below the standard of language | Replaced; the terms defined |

The reviewer's own changes in revision 2: the chances of P55.1 to P55.4, P55.7 and P55.8, with P55.2b and P55.4b added (the note above the table says on what they rest); the section "For the coordinator"; the statement in "Status" that the brief is not released by the reviewer.
