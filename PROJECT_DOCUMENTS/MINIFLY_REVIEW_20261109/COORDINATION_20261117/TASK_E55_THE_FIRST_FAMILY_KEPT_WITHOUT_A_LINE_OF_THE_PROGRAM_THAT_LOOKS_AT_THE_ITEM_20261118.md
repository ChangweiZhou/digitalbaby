# E55, first part: the first family kept while a new task is learned, without a line of the program that looks at the item; the update gates of the cells in reserve are shut by a bias and can be opened only by learning (18 November 2026, revision 1)

**Status.** This brief was written by the outside reviewer, who acts for the coordinator; the coordinator cannot be reached while Mr. Xu is away (his instruction of 17 November: arrange it yourself). It is the first part of item 11 of `PLAN.md` beside this file, and it is run as an addition to the trial of E51 by E51's worker, on that worker's branch, with E51's births, tasks and runner. Before its release it is checked by an agent of the reviewer that has not seen the reviewer's reasoning; that check is not the independent audit that the programme's rules ask for, and the coordinator has not seen the brief. The job is at the evidence level that the coordinator's briefs call E2: a first measurement, which confirms nothing. It has one seed and eight runs and costs about 2 CPU-hours. The result is verified from the receipts and the weights by an agent of the reviewer before the reviewer reads it (rule 4 of section 4 of the plan).

## Shell names and host

`D=origin/codex/minifly-consolidated-20261023` (the default branch, at 7cabb13a); `W1=origin/claude/e51-reserve-017dSEpW` (E51's worker's branch, the branch of this job: the runner `$M/exp51_reserve.py`, SHA-256 `84d0dad7f6e5eeb15c4518a8c591eb5fc043d72b17926913a3b3fde690d1f34a`, and E51's starts, receipts and weights under `$M/exp45/e51/`); `V=origin/claude/vibrant-cerf-sje726` (the reviewer's branch: this brief, the brief of E51 `$C/TASK_E51_A_NEW_TASK_LEARNED_IN_CELLS_HELD_IN_RESERVE_20261117.md`, and the reviewer's verification of E51's trial `$R/E51_TRIAL_VERIFICATION_20261117.md`); `M=MOD97_LOCAL_RULE_20261106`; `R=PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109`; `C=$R/COORDINATION_20261117`. The host is Linux on x86-64 with Python 3.13 and torch 2.15.0+cpu, as set up for E51.

## Terms

The terms of the brief of E51 hold. Those that this brief uses are repeated here.

- *The network* reads one byte at each position of an item. It has a table of 8 numbers for each byte (the *byte table*), a layer `l1` whose output for a byte is the byte's *code* (512 values), a recurrent layer of 256 *cells*, and a readout: 512 units, each the hyperbolic tangent of a sum formed by a map `l2` from the code and a map `lm` from the cells' states, followed by a map `l3` to the 257 output classes. The *input map* takes the code to the cells and the *recurrent matrix* takes the cells' previous states to them. A cell has a reset gate r, an update gate z and a candidate n, and its new state is (1 − z) n + z h, where h is its previous state; the state is zero at the start of every item, so that a cell whose z is 1 at every position has the state zero. In the two matrices of the recurrent layer, which have 768 rows, rows 0 to 255 belong to the reset gates, rows 256 to 511 to the update gates and rows 512 to 767 to the candidates.
- *Back-propagation* (rule B) uses the exact gradient. *The sweep* (rule S) sends the error back through each linear map by the signs of the map's weights times the mean magnitude of those weights.
- *The first family* is four operations on two numbers below 89 (addition and subtraction modulo 89, the larger and the smaller of the two). *A birth* is a network trained on the first family alone. *The embedded network* of E51 is E41's birth of half the width (256 values of code, 128 cells, 256 readout units) placed inside a network of the present size: cells 0 to 127 are the *old cells* and cells 128 to 255 the *cells in reserve*; units 0 to 255 of `l1` and of the readout are *old* and units 256 to 511 *added*. The *old rows* of a matrix of the recurrent layer are the three rows of each old cell; the *gate rows of the reserve* are rows 384 to 511, the rows of the update gates of the cells in reserve.
- *The bytes in use* are the 46 bytes that occur in the first family's items. *The directions in use* at a matrix are the directions spanned by the input vectors (with a constant 1 appended) that the matrix receives on the first family's items when the cells in reserve are silent: 46 at the input map, 129 at the recurrent matrix and at `lm`, 11 at `l2`. The reviewer's verification found that they are the codes of the bytes in use (input map), the coordinates of the 128 old cells (recurrent matrix and `lm`) and the codes of `=` and of the ten digits (`l2`), each with a constant. A change of a row of a matrix *away from the directions in use* is a change that leaves the row's summed input unchanged for every input vector in their span.
- *The rule* of E51 projects, at every update, the gradient and then the step of chosen rows of the four matrices away from the directions in use, and restores after the step what must not change (`l1`, the old columns and the bias of `l3`, the rows of the byte table of the bytes in use).
- *The two added lines* of E51's forward pass read the first byte of each item and, when it is a byte in use, set the update gates of the cells in reserve to 1 and the added units of the readout to zero.
- *The two tasks* are E51's: `unrelated` (E43's task on 98 bytes that the first family does not use: two of 64 symbols are shown, then 0 to 12 fillers, then a query byte; the answer is the two symbols) and `conflict` (the family of the same four operations modulo 97, whose items are made of bytes in use except for their first byte, which names the operation).
- A *scored row* is the set of scores taken after 0, 16, 32, 64, 128, 256, 512, 1,024, 2,048 and 4,096 updates. The *reduction score* is the percentage of right answers on the pairs of addition and subtraction of the first family whose answer needs the reduction modulo 89 (7,832 pairs, without letters). The *never-taught score* of a new task is the percentage of right answers on its items that were never taught (300 for `unrelated`, 800 for `conflict`). A new task is *learned from row g* if its never-taught score is at 80 or more at scored row g and at every later one.
- A *receipt* is the file that a run or a check writes with its settings, its scored rows and the hashes of its sources. A *dry run* is a run made by a check.

## The question

Mr. Xu's requirement (17 November) is that long-term memory form by itself during learning and not by an intervention of ours; some rehearsal is acceptable. In the trial of E51 the first family's 32,884 scored answers stayed the same, item for item, while each of two new tasks was learned, without rehearsal. The reviewer's verification states what did this: the forward pass reads each item's first byte and, for a byte in use, holds the cells in reserve and the added readout units at zero; with these two lines removed, the same trained weights answer none of the first family's reduction pairs after `unrelated` and at most 0.4% after `conflict`. That separation is an act of the program and not something the network has learned or holds in its weights.

**Can the weights carry the separation? If the update gates of the cells in reserve are shut by a bias, and the rule allows the rows of these gates to change only away from the directions in use, so that no first-family item can open them while learning on other items can, is the first family kept and is each new task learned, with the forward pass of the plain network?**

This removes one of the acts of the program and not all of them; the section "What remains supplied by the program" lists the others.

## What the record says

Seed 73129; 4,096 updates; no weight decay; a new optimizer at the start of the new task. Figures are given as sweep; back-propagation.

- **E51's trial** (`$W1:$M/exp45/e51/`; verified by the reviewer's agent from receipts and weights, `$V:$R/E51_TRIAL_VERIFICATION_20261117.md`). Reduction score at row 0: 92.4; 95.3.

| Arm of E51 | What it is | First family at row 4,096 (reduction score) | `unrelated` learned from row | `conflict` learned from row | Cells in reserve open by more than 0.1 on the new task's items at row 4,096 (`unrelated`; `conflict`) |
|---|---|---|---|---|---|
| `keyed` | the two added lines and the rule | 92.4; 95.3 on both tasks; no answer of 32,884 changed at any row | 1,024; 512 | 1,024; 512 | 119; 125 and 112; 83 of 128 |
| `keyed-free` | the two added lines, no rule | `unrelated` 0.0; 0.0. `conflict` 1.0; 0.2 | 512; 512 | 2,048; 1,024 | not used here |
| `open` | no added line, no rule; the update gates of the reserve start shut by a bias of 10 and are free | `unrelated` 0.0; 0.0 (lost by row 16). `conflict` 1.3; 0.0 | 512; 512 | 1,024; 1,024 | 1; 8 and 0; 0 of 128 |
| `rehearse` | as `open`, with every second item drawn from the first family | `unrelated` 90.0; 92.8. `conflict` 88.3; 97.8; of the 31,684 answers, 9.6% and 9.8% differ from row 0's under the sweep and 6.7% and 4.1% under back-propagation | not used here | not used here | not used here |

- **Three statements of the verification** bear on this job. First, in `keyed` the answers are unchanged and the outputs are not: by row 4,096 they have moved by up to 5.3e-4, because the weights are stored in single precision after each projected step; the smallest difference between the largest and the second largest output of a first-family answer is 1.8e-4 under the sweep. Second, at the recurrent matrix and at `lm` the rule is the same as leaving unchanged the old units' weights from old cells and their biases. Third, with a bias of 10 a shut gate lets through a share of 4.5e-5 of the candidate at each position: in the start of the arm `open` the largest state of a cell in reserve on the first family's held-out items is 0.0015; 0.0024, the outputs differ from the birth's by up to 0.0030; 0.0034, and 1; 0 of the 32,884 answers differ.
- **In the arm `open` the gates did not open.** The new task was learned there with at most 8 of the 128 cells in reserve open, in the old cells, whose weights were free; the first family was lost. Whether learning opens gates that start shut, when the old cells' weights are protected, is not on record.

## The change

Everything is E51's arm `keyed` (the births, the embedded network, the tasks and their streams, the optimizer, the number of updates, the scored rows, the scores) but three things.

1. **The forward pass has no added line.** It is `exp51_reserve.fwd` with `KEY["in_use"]` set to `None`, in the update and in the scoring. The second scoring of the new task with the cells in reserve held shut on every item (`KEY["shut_all"]`) is kept as a measurement, as in E51.
2. **The start `shut`.** The keyed start of E51 (`e51_start_keyed_<rule>_73129.pt`, with its saved directions `e51_start_keyed_<rule>_73129_directions.pt`) is changed in four blocks of rows, each with its bias appended as a last column: the gate rows of the reserve in the input map and in the recurrent matrix, and rows 256 to 511 of `lm` and of `l2`. Each block is replaced by its projection away from the directions in use of its matrix (`exp51_reserve.away`, in double precision), and then 10 is added to the bias of the input map at the gate rows of the reserve. After this, for every input vector in the span of the directions in use, an update gate of the reserve receives 10 from the input map and nothing from the recurrent matrix, and an added readout unit receives nothing from `l2` and nothing from `lm`. The weights of these rows on other directions keep their draws: from the codes of bytes that are not in use, and from the cells in reserve.
3. **The rule with more rows.** E51's rule projects the old rows of the two matrices of the recurrent layer and rows 0 to 255 of `lm` and `l2`. In the arm `shut` it projects, in addition, the gate rows of the reserve in both matrices of the recurrent layer and rows 256 to 511 of `lm` and `l2`. The reset rows and the candidate rows of the cells in reserve stay free, so that a cell in reserve may read the old cells and the codes of the bytes in use through its candidate; what it cannot do is open on an item that presents nothing outside the directions in use.

The directions are those saved with E51's keyed start. They are not computed again on the start `shut`: there the cells in reserve have states of the order of 1e-3 on the first family's items, which a new computation would count as directions in use and so protect every weight from a cell in reserve.

What follows from the construction, for every item made of bytes in use only: at every position an update gate of the reserve receives 10 and whatever the cells in reserve send it through the recurrent matrix; as long as the states of the cells in reserve are small, the gate stays at 1 − 4.5e-5 and the states stay small. The construction does not make them zero, and it does not bound what the cells in reserve send to the old cells and to the readout, whose weights from cells in reserve are free. How small the effect on the first family is, is the first thing the job measures.

## Arms and runs

| Arm | Start | Rows projected by the rule | What it shows |
|---|---|---|---|
| `shut` | `shut` | E51's rows, the gate rows of the reserve, rows 256 to 511 of `lm` and `l2` | whether the first family is kept and the new task learned without an added line |
| `shut-gates-free` | `shut` | E51's rows and rows 256 to 511 of `lm` and `l2`; the gate rows of the reserve are free | whether the protection of the gate rows is needed |

Two arms, two tasks, two rules, seed 73129: eight runs of 4,096 updates, each network under the rule of its birth. E51's runs of `keyed`, `open` and `rehearse` of the same task, rule and seed are the references; none is run again. The sixteen runs of E51 cost 2.74 CPU-hours together; the eight runs of this job cost about 1.5 CPU-hours and the checks about 0.4.

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
           m.rnn.bias_ih_l0[Z_RES] += GATE_BIAS

       def load_shut(rule, seed):
           m, m0, in_use, dirs, rec = R51.load_start("keyed", rule, seed)     # asserts E51's committed keyed start and its saved directions
           R51.install(None)                                                   # the forward without the added lines
           shut_start(m, dirs)
           return m, in_use, dirs, rec

   `start` builds the start with `load_shut`, saves it as `e55_start_shut_<rule>_73129.pt` and writes `e55_start_shut_<rule>_73129.json` with the hash of its tensors (`R.tensors_sha`), the SHA-256 of the file, the record that `R51.load_start` returns, `GATE_BIAS`, and the provenance. Every run and every check builds the start again with `load_shut` and asserts the hash of its tensors against that receipt.
2. **The rule.**

       class Rule55(R51.Rule):
           def __init__(self, m, dirs, in_use, arm):
               super().__init__(m, dirs, in_use, True)
               cells = T.cat([R51.OLD_ROWS, Z_RES]) if arm == "shut" else R51.OLD_ROWS
               units = T.arange(R51.H1)
               self.mats = [(m.rnn.weight_ih_l0, m.rnn.bias_ih_l0, dirs["ih"], cells), (m.rnn.weight_hh_l0, m.rnn.bias_hh_l0, dirs["hh"], cells),
                            (m.lm.weight, m.lm.bias, dirs["lm"], units), (m.l2.weight, m.l2.bias, dirs["l2"], units)]

   `R51.Rule.before` and `R51.Rule.after` are used as they are; they act on `self.mats`.
3. **The gates on the first family's items.**

       @T.no_grad()
       def gates_on(m, items):
           """Teacher forcing, the plain forward. Over the cells in reserve: the largest of 1 - z over cells, positions and items; the number of
           cells whose mean of 1 - z over positions and items exceeds 0.1; the largest magnitude of a state; and the largest magnitude of an
           added readout unit at the answer positions."""

   It groups the items by length as `R51.reserve_figures` does, takes `z`, `h` and `z2` from the second value that `R51.fwd(m, x, False, True)` returns, and returns the four figures as `largest_opening`, `cells_open_by_more_than_0p1`, `largest_reserve_state` and `largest_added_unit_at_answer_positions`.
4. **The run** is a copy of the text of `R51.run` (lines 452 to 549 of `exp51_reserve.py`) with the changes below and no others; every name of `exp51_reserve.py` that the text uses is taken from `R51`. The report shows the difference between the two texts.
   - The signature is `run(task, arm, rule, seed, out, cmd, updates=R51.DOSE)`; the run asserts `arm in ("shut", "shut-gates-free")`. The branch for `start == "e26"` is left out, and what the text does `if start != "e26"` is done always.
   - The tag is `e55_<task>_<arm>_<rule>_<seed>`, with `_dry<N>` for a run of N updates other than 4,096; the folder is `$M/exp45/e55`.
   - The start: `m, in_use, dirs, start_rec = load_shut(rule, seed)`, with the assertion of point 1.
   - The rule: `ruleobj = Rule55(m, dirs, in_use, arm)`.
   - The stream: `nxt, stream_note = R51.item_source(task, "keyed", seed)`, the new task's stream alone, which is the stream of E51's arm `keyed`.
   - In `score_row`, after the entry `reserve`: `r["gates_on_first_family"] = gates_on(m, s2_items)` (the 1,200 held-out pairs at 12 letters).
   - The variable `cfg` is not used. The settings: `experiment="E55"`, `part="first"`, `start_form="shut"`, `keyed_forward=False`, `rule_applied=True`, `restore_book=True`, the new entries `rule_rows` (the arm's row sets in words) and `gate_bias=GATE_BIAS`; the entries `update` and `forward` say that the forward is `exp51_reserve.fwd` without a key and that the rule is `exp55_shut.Rule55`; `provenance` adds the SHA-256 of `exp55_shut.py`.
   - The assertion before the first row is `assert R51.KEY["in_use"] is None`.
5. **Files committed:** the two start files with their receipts, the receipts of the checks and of the eight runs, the weights of rows 64 and 4,096 of every run and the files of generated bytes that the run writes, as in E51 (the weights added with `git add -f`), and the logs.

## Checks before the runs

Each check writes a receipt `$M/exp45/e55/checks/e55_check_<n>_73129.json` with an entry `ok`. No run is started before the three checks have passed. If a check fails, the worker stops, reports the first difference and asks. The panels are E51's (`R51.check_panels(73129)`): the 1,200 held-out pairs at 12 letters and the 2,000 items of E51's check 1.

1. **The start,** for each rule.
   (a) Every tensor of the start `shut` is bit-equal to the keyed start's outside the four blocks (rows 384 to 511 of `rnn.weight_ih_l0`, `rnn.bias_ih_l0`, `rnn.weight_hh_l0`, `rnn.bias_hh_l0`; rows 256 to 511 of `lm.weight`, `lm.bias`, `l2.weight`, `l2.bias`).
   (b) For each block, with `D` the block and its bias in double precision, and with 10 subtracted from the bias column of the gate rows of the input map, the largest magnitude of an entry of `D @ Vm` is at most 1e-5.
   (c) With the forward without a key, on the two panels: the largest of 1 − z of a cell in reserve is at most 1e-4; the largest magnitude of a state of a cell in reserve and the largest magnitude of an added readout unit at the answer positions are recorded and are at most 0.01; the largest difference of an output at an answer position from that of the keyed start under the keyed forward is recorded and is at most 0.02 (the limit of E51's check 1 for its start with a bias, which gave 0.0030 and 0.0034).
   (d) The number of the 32,884 generated answers of the first family (the 31,684 items without letters and the 1,200 held-out pairs at 12 letters) that differ from those of the keyed start under the keyed forward is recorded and is at most 5 (E51's start with a bias: 1 and 0).
2. **The rule,** for each rule, each task and each arm: eight updates on the task's stream from the start `shut`, as E51's check 3 makes them.
   (a) In both arms: `l1.weight`, `l1.bias`, `l3.weight[:, :256]`, `l3.bias` and the rows of the byte table of the bytes in use are bit-equal to the start's; for each of the four matrices and the rows that the arm projects, the largest magnitude of an entry of `(D_after − D_start) @ Vm` is at most 1e-5; each of the four matrices has changed.
   (b) In the arm `shut`: on the two panels the largest of 1 − z of a cell in reserve is at most 1e-4, and the largest change of an output at an answer position from the start's is recorded and is at most 1e-3.
   (c) The check can fail: in the arm `shut-gates-free` the largest magnitude of an entry of `(D_after − D_start) @ Vm` at the gate rows of the reserve, in the input map or in the recurrent matrix, exceeds 1e-4. If it does not, the worker reports the figure and asks, because the gradient that reaches a shut gate may be too small for the optimizer to act on, and the arm would then not differ from `shut`.
3. **The stream and the cost.** For each rule and task, a dry run of 64 updates of the arm `shut`: its data digest at row 64 equals that of E51's `keyed` run of the same task and rule at row 64. From the four dry runs: CPU-seconds an update and a scored row, and a forecast for the eight runs. If the forecast exceeds 4 CPU-hours, the worker asks before starting.

## How the result is stated (fixed beforehand)

For each of the eight runs the report gives a table of the ten scored rows with: the reduction score; the number of the 31,684 answers without letters and of the 1,200 held-out answers at 12 letters that differ from row 0's; the never-taught score of the new task, and the same with the cells in reserve held shut; the number of cells in reserve open by more than 0.1 on the new task's items; and the four figures of `gates_on` on the first family's held-out items. Beside it stand the rows of E51's `keyed`, `open` and `rehearse` of the same task and rule.

- **Words for the first family in one run.** With d(g) the number of the 31,684 answers without letters that differ from row 0's at scored row g: the first family is *kept exactly* if d(g) is 0 at every row and no held-out answer differs at any row; *kept* if it is not kept exactly, d(g) is at most 31 at every row (0.1% of the answers) and the reduction score at every row is within 0.5 points of row 0's; *lost* if the reduction score at row 4,096 is below half of row 0's; and *partly kept* otherwise.
- **Words for the new task in one run.** *Learned from row g*, as defined in "Terms"; otherwise *not learned*, with the never-taught score at row 4,096.
- **What the report may say for the arm `shut`, for seed 73129 and one rule and task.** If the task is learned and the first family is kept exactly or kept: "the first family was kept while the task was learned, with the forward pass of the plain network; the separation was carried by weights". If the task is learned and the first family is partly kept or lost: "the weights did not carry the separation", with the row at which d(g) first exceeds 31 and the figures of `gates_on` at that row, which show whether the gates opened on the first family's items or stayed shut while the states of the cells in reserve grew. If the task is not learned: "learning did not open the cells in reserve within 4,096 updates", with the number of cells open on the new task's items; the first family's word is then given and is not read as a success of the construction.
- **The arm `shut-gates-free`** is read beside `shut`. If, for a rule and task, the first family is kept exactly or kept in both arms, the report says that the protection of the gate rows was not needed in that run. If it is kept in `shut` and partly kept or lost in `shut-gates-free`, the report says that the protection of the gate rows was needed, with the figures of `gates_on`.
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

| | Statement | Chance |
|---|---|---|
| P55.1 | Checks 1 to 3 pass at the first attempt | 65% |
| P55.2 | In the arm `shut`, `unrelated` is learned under both rules | 45% |
| P55.3 | In the arm `shut`, `conflict` is learned under both rules | 30% |
| P55.4 | In the arm `shut`, the first family is kept exactly or kept in all four runs | 70% |
| P55.5 | In the arm `shut`, no cell in reserve is open by more than 0.1 on the first family's held-out items at any scored row of any run | 90% |
| P55.6 | In the arm `shut`, every run in which the task is learned has at least 32 cells in reserve open by more than 0.1 on the new task's items at row 4,096 (the statement is true if no run learns its task) | 80% |
| P55.7 | In the arm `shut-gates-free`, the first family is partly kept or lost on `conflict` under both rules | 75% |
| P55.8 | In the arm `shut`, where `unrelated` is learned, it is learned from row 2,048 or earlier (the statement is true if it is learned under neither rule) | 75% |

## What it cannot show

- The job has one seed, two tasks and 4,096 updates. In E51's arm `keyed` the first family's outputs moved by up to 5.3e-4 in that time through the rounding of the stored weights; here the cells in reserve add their small states. Nothing bounds either over a longer run.
- The job does not test a third task. A second new task would present items outside the directions in use and could open the cells that the first new task uses.
- The job does not test an item of the first family that contains a byte not in use.
- The points of "What remains supplied by the program".
- The job uses the learned code of E41's births. Whether a code in which a byte has values of its own (E50) lets the directions in use be replaced by a record kept for each input line is the subject of the next part.

## Rules for the worker

- Work on your branch `$W1`, after Part B of E51 is finished and reported; this job does not interrupt Part B. Fetch the reviewer's branch to read this brief: `git fetch origin '+refs/heads/claude/vibrant-cerf-sje726:refs/remotes/origin/claude/vibrant-cerf-sje726'`, then `git show origin/claude/vibrant-cerf-sje726:<path>`. Use the environment of E51. Run at most three processes at a time, each with one thread, from the folder `$M`.
- **This host stops a machine about five minutes after its session's turn ends (a turn is the time in which the worker acts without waiting for a message), and it has also restarted a machine while a turn was open. The disk is kept, and processes are terminated.** While runs proceed, do not end your turn: issue a command that waits for the processes and repeat it until they have finished; commit and push the logs and the receipts between two such commands. The checks take about a quarter of an hour and the eight runs about three quarters of an hour with three processes at a time. A run that was interrupted is started again from its beginning, after its incomplete receipt and any file it wrote have been moved to `$M/exp45/e55/interrupted/`; the report says which runs were started again.
- Push only to your own branch. Never push to the default branch or to any other branch, and open no pull request.
- Every commit carries the trailers that `PROJECT_DOCUMENTS/KB/AGENTS_POINTER.md` prescribes. Apart from the trailers, no name of any language model appears in a committed file or a commit message.
- The report, `$M/exp45/e55/E55_REPORT.md`, is written in standard academic English: full sentences, every term defined at first use, every figure with the receipt or the command that gives it; the committed files are listed from `git ls-tree`.
- If the brief cannot be followed at some point, stop that part, say what you found, and ask.

## Reporting

Send the reviewer a short message with the tool `send_message` of the claude-code-remote server, to session `session_01YBMaMvtGtQvDFcgppgThrZ`, at each of these points: the three checks passed (or which failed), with the figures of check 1 (c) and (d); the four runs of the arm `shut` finished, with each run's words, its d(4,096), its never-taught scores and its numbers of cells open; the eight runs finished and the report committed. Each message gives your branch, the commit and the main figures in at most ten lines. If you need a decision, ask in the same way and go on with whatever does not depend on the answer.

## Done

The job is done when the two start files with their receipts, the receipts of the three checks, the eight runs with their receipts, weights of rows 64 and 4,096, files of generated bytes and logs, and `E55_REPORT.md` exist; the report holds the tables of the eight runs beside E51's rows, the words fixed above, the section on what remains supplied by the program, the outcome of each prediction, the difference between the runner's `run` and `exp51_reserve.run`, and the list of committed files.
