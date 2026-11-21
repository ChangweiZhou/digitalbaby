# Outside audit of the briefs E61 and E62 and of the chain of steps (18 November 2026)

**For** Mr. Xu, Mr. Zhou, the outside reviewer and the coordinator. **Written by** the outside session that wrote the advice memo beside this file, at the reviewer's suggestion as Mr. Xu relayed it on 18 November: to audit the briefs of E61 and E62 before their release, above all one question of E61's design, and to read the chain of steps independently.

**Status:** proposed. This document decides nothing and releases nothing; the release is the coordinator's. Nothing was trained. The figures marked *computed here* come from forward and backward passes on weights that the record has saved, made by the script `audit_e61_relay.py` beside this file, whose output is `audit_e61_relay_output.txt`. Every reading is mine and is marked as such. Nothing here is a statement of Mr. Xu's or of Mr. Zhou's, and I do not say how the arm under the exact gradient is to be read: decision 31 leaves that to be run, and the reviewer records its own reading as a reading.

## 1. The answers in brief

1. **E61, the design question.** As written, E61 is a sound measurement of one thing and is not a test of the other. It measures how far back in an item credit must be carried when every path is cut after K positions. It does not test relay between neurons, in two respects. (a) Relay is not what brings an error to a cell: the readout holds 92.1% of the weights in use and reads every cell, and before any relay the error is non-zero at all 256 cells under both rules. (b) K counts positions and not crossings between cells: a step of the brief's relay also cuts the cell's own path, which needs no relay. An error covers this network in 4 or 5 crossings; the first operand stands 4 and 5 positions before the answer in the items of R and 16 and 17 in those of A. Whatever K the runs report as sufficient will therefore reflect the length of the scored items (near 5 on R, near 17 on A) and not the number of passes between cells. Set beside "1 → 50 → 50² → 50³: a few passes cover the network", either number would mislead.
2. **A network in which only some cells are read is not the remedy by itself.** In E47's network, which E53 takes up, the 256 cells that are read are the cells that read the byte; around them only the first ring, one crossing away, takes part. A limit on relay would there test one crossing and no more, unless E53 shows that the further cells come into use. The reason is general in this harness: one crossing takes one byte, forward as well as backward, so that cells several crossings from the readout cannot serve an answer that is due one position after the last digit.
3. **What I propose for E61 (mine).** Run the form that the brief names and sets aside: each cell keeps its own path for the whole item, and the number of crossings between cells is limited to X = 0, 1, 2 and 4. It separates what a cell has by itself from what must be relayed, on the same network and beside the same record. Computed here on the record's two trained networks, for items with 12 letters: of the credit that the whole item gives at the first operand, the brief's form delivers none for K of 8 or less and 0.16 to 0.33 at K = 16; the form with crossings delivers 0.02 to 0.04 at X = 0 and 0.46 to 0.47 at X = 4, and at the start of training 0.95 to 0.97 at X = 4. I would run these eight runs first, with K = 0 and K = 8 beside them, inside about the brief's budget by my count (section 3.7).
4. **E62 can be released as it stands.** Its reading is bounded as the brief itself says. Two additions cost seconds a run, and one caution concerns how its result is carried forward (section 4).
5. **The chain of steps.** One statement is not supported by the receipts: E50's 2.68 is the sweep against the sweep with a dense reading of the code, and against back-propagation in the same network the figure is 2.23. The order holds for requirements B and C, with two things missing from the candidate of step 5: a readout that is not dense, and an input for bytes outside the first family. For requirement A the order does not follow from the record. The reviewer's own section 29 and E43's second addendum show that the network with 10 connections a cell loses the first family within 64 updates, as the dense one does, because every item still drives every cell. Nothing that steps 2, 3 and 6 vary is likely to change that; the input (step 4) and the readout can. In my reading step 4 is the hinge of the chain and should be read for what an item leaves idle as well as for whether the family is learned (section 5).

## 2. Scope, independence, and what was executed

**REVIEW_SCOPE.** The reviewer's branch `claude/vibrant-cerf-sje726` as fetched at 8434d095, folder `PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/COORDINATION_20261117/`: the brief of E61 (revision 2) with its check and `check_e61_brief.py`; the brief of E62 (revision 2) with its check and `check_e62_brief.py`; `STEPS_FROM_THE_PRINCIPLE_20261118.md` (revision 2); the reviewer's memo, sections 28.4 to 28.6, 29 and 33.21 to 33.30. The default branch `codex/minifly-consolidated-20261023` at e647e7d7: the coordinator's decisions 30 to 37; the source files `e50_sparse_code.py`, `exp50_code.py`, `e42_cells.py`, `exp42_cells.py`, `e49_interface.py`, `exp3_state.py`, `exp11_cell.py`, `exp45.py`; the receipts and saved weights of E50's stage 3 (`exp45/e50/e50_C46-one_<S|B>_73129`, with the start file), the other four receipts of E50 and four receipts of E42. The claims examined: that relay over K steps tests the principle of relay between neurons (the reviewer's question); the design of E62; the order and the branches of the chain of steps.

**INDEPENDENCE.** Executed, by a separate session on its own branch with its own code: numpy and the standard library, written from the text of the source; it executes no code of the repository and reuses none of the reviewer's scripts. The session is of the same model family as the author of the briefs, so this is not a cross-family review in the sense of the project's review skill, and that review remains pending. I had read the check of E61's brief before I wrote the script. No formal checker gate exists in this branch and none is claimed.

**CHECKS_EXECUTED.** From the repository's root:

| Command | What it showed |
|---|---|
| `python3 PROJECT_DOCUMENTS/MINIFLY_OUTSIDE_ADVICE_20261118/audit_e61_relay.py` (about ten minutes on one core; `quick` as argument gives everything except part 4 in about one minute) | **Validation of the script against the record.** The code's digest and the digest of the first 256 batches of the stream equal the receipt's (12111044f0f8bc70; a921d91607c6421b). The loss of the first batch at the start is 5.590670, the receipt's value. The six cosines between the sweep's and the exact gradient that the receipt stores for the first batch, at the start and at the end, are reproduced to four decimals (0.7554, 0.8658, 0.7453; 0.5152, 0.7950, 0.6899). The exact whole-item gradient agrees with central differences of the loss to 1.5e-07. The form with at most X crossings agrees with central differences of its direct definition to 3.4e-08 for X = 0, 1, 2 and 3. The exact answers of both final networks on the panels A, R and T equal the last scored row of the receipts in all ten counts. **Findings:** parts 1 and 3 to 6, quoted in sections 3 to 5 |
| The reviewer's `check_e61_brief.py`, run again here | Its numerical test passes (the brief's procedure against the direct definition, largest difference 1.1e-15); its statements about the source are present; the distances and shares of the brief's question and the forecast of cost (35.3 CPU-hours in the reference form, 26.6 with the second economy) are printed as its check gives them |
| The reviewer's `check_e62_brief.py`, run again here | Its lines A to C print the figures of the brief and of the check's table: the ten digests of the connections; 40 cells read by no other cell at N = 2 and 4 at N = 4; every cell reached from every cell within 5, 4, 4, 3, 2, 2 and 1 steps at N = 8, 10, 16, 32, 64, 128 and 255; the counts of connections outside the neighbourhood; the record's figures. Its last part reads the reviewer's memo from the working tree, which this checkout does not hold, and was not run |

Not executed: any code in torch (it is not installed here), so the brief's `relayed_grads` was read and not run; any training.

## 3. E61: is relay over K steps a fair test of relay between neurons?

### 3.1 Terms

*The network* is E50's stage 3, as the brief defines it: 256 gated cells, each reading 10 of the 48 cells around it and itself; a fixed input code of which each cell reads 46 values, one of every used byte; a dense readout. *A cell's own path* leads from its state at a position to its own state at the position before, through the term `z * h(t-1)` and through its reading of itself in the recurrent map. *A crossing* leads from a cell's state at a position to the state, at the position before, of another cell that it reads. *Form (a)* is the brief's: every path, a cell's own included, is cut K positions back. *Form (b)* keeps a cell's own path for the whole item and admits the paths with at most X crossings; the brief names it in its section on what one step is and does not run it. *The credit at the first operand* is the error at the states of positions 2 and 3 of an item, where the two digits of the first operand are read.

### 3.2 The error reaches every cell before any relay (*computed here*, parts 1 and 3)

Of the weights in use, the input map has 35,328, the recurrent map 8,448, the cells' biases 1,536 and the readout (`lm`, `l2`, `l3` with biases) 526,081, which is 92.1%. Each of the 512 readout units reads all 256 cells. With K = 0 the error that leaves the readout is non-zero at 256 of 256 cells, at every answer position of every item of the first batch, under both rules. By relay alone an error that starts at one cell reaches every cell after 4 or 5 steps. The brief states this consequence of the dense readout itself, and lists among what the experiment cannot show "relay as the only way an error reaches a cell". The step list calls the dense network "outside the principle under either rule"; by the same reading (the reviewer's, not a ruling) a dense readout is outside it, and no value of K brings this learner inside it.

### 3.3 What K limits

One step of form (a) moves the error one position back along every path at once. It therefore limits two different things together: the crossings between cells, which are relay, and the cell's own path, which is not. In the step list's words a neuron has "what arrives over its own sparse connections and its own state"; its own earlier state needs no relay. Because of the second limit, K must reach the distances that the format of the item fixes, whatever the network: 4 and 5 positions to the first operand in the items of R, which have no letters, and 16 and 17 in the items of A and T, which have 12. The brief computes these distances and builds its predictions on them. My reading: the outcome of the K-grid is largely fixed by the format of the item, and it says little about the number of passes between neurons that the principle speaks of.

What the K-grid does measure, validly: how many positions of its own past quantities a cell would have to keep if every path were cut. That is locality in time, the plan's old item 20, and it is worth an answer of its own.

### 3.4 What each form of limit keeps (*computed here*, part 4)

The whole-item gradient is the sum over all paths; each form keeps a subset of them. The first table gives the cosine between the gradient of the recurrent map under the limit and under the whole item, by the same rule. The second gives the share of the whole-item credit at the first operand that the form delivers (the projection of the form's error there on the whole-item error, as a share of the latter's squared length). Means over batches of 64 items of the stream; "trained" are the record's final weights, each under its own rule.

| Cosine, recurrent map | K = 0 | 1 | 2 | 4 | 8 | 16 | X = 0 | 1 | 2 | 4 |
|---|---|---|---|---|---|---|---|---|---|---|
| Sweep, trained, no letters | 0.74 | 0.87 | 0.94 | 0.99 | 1.00 | 1.00 | 0.83 | 0.96 | 0.99 | 1.00 |
| Sweep, trained, 12 letters | 0.68 | 0.83 | 0.91 | 0.96 | 0.99 | 1.00 | 0.77 | 0.93 | 0.98 | 1.00 |
| Exact gradient, trained, no letters | 0.57 | 0.73 | 0.83 | 0.92 | 1.00 | 1.00 | 0.68 | 0.88 | 0.96 | 1.00 |
| Exact gradient, trained, 12 letters | 0.41 | 0.58 | 0.71 | 0.85 | 0.94 | 0.98 | 0.56 | 0.78 | 0.90 | 0.97 |

| Share of the credit at the first operand | K = 0 | 1 | 2 | 4 | 8 | 16 | X = 0 | 1 | 2 | 4 |
|---|---|---|---|---|---|---|---|---|---|---|
| Sweep, trained, no letters | 0 | 0 | 0 | 0.29 | 1.00 | 1.00 | 0.07 | 0.27 | 0.58 | 0.97 |
| Sweep, trained, 12 letters | 0 | 0 | 0 | 0 | 0 | 0.33 | 0.04 | 0.12 | 0.22 | 0.47 |
| Exact gradient, trained, no letters | 0 | 0 | 0 | 0.13 | 1.00 | 1.00 | 0.04 | 0.20 | 0.49 | 0.93 |
| Exact gradient, trained, 12 letters | 0 | 0 | 0 | 0 | 0 | 0.16 | 0.02 | 0.09 | 0.20 | 0.46 |
| Sweep, at the start, 12 letters | 0 | 0 | 0 | 0 | 0 | 0.49 | 0.05 | 0.29 | 0.65 | 0.97 |
| Exact gradient, at the start, 12 letters | 0 | 0 | 0 | 0 | 0 | 0.45 | 0.03 | 0.18 | 0.49 | 0.95 |

Three readings, each mine.

1. Under form (a) the credit at the first operand is exactly zero until K reaches the distance. For items with 12 letters nothing arrives at K of 8 or less, and at K = 16 a sixth to a third in the trained networks. A run with K of 8 or less can learn such items only by what carries over from shorter ones, as the brief says.
2. Under form (b) some of that credit arrives at every X. At X = 4, about the number of crossings that covers this network, nearly all of it arrives at the start of training (0.95 to 0.97 for 12 letters) and, in the trained networks, nearly all for items without letters (0.93 to 0.97) and about half for items with 12 (0.46 to 0.47).
3. The cosine over all weights flatters both forms, because most of the gradient belongs to the last positions; the second table is the one to read. It also shows a limit of this computation: the record's trained networks carry the first operand through the letters partly by passing it between cells, so that half of its credit on long items needs more than four crossings. A learner limited to X crossings would have to hold more of it in the cells' own paths than these networks do. Whether it does is what a run would show; the tables do not predict it.

### 3.5 The trained networks depend wholly on what passes between cells (*computed here*, part 5)

Forward passes on the record's two final networks, with the project's own panels and the scorer's rule of an exact answer. Each cell keeps its own path; what it receives from the other cells is left intact, removed, or replaced at each position by its mean over the panel's items, which keeps the size of the input and removes what is specific to the item.

| Share of exact answers (%) | Intact (the record's last row) | Nothing from other cells | Only the mean over items |
|---|---|---|---|
| Exact gradient's network: R; A; T | 95.15; 94.67; 99.17 | 0.00; 0.00; 0.00 | 0.70; 0.83; 0.83 |
| Sweep's network: R; A; T | 85.75; 83.17; 93.00 | 0.00; 0.00; 0.00 | 0.75; 0.67; 0.83 |

Every answer of these networks, taught or not, depends on what cells receive from other cells. Credit for what a cell passed to another is therefore credit that this solution needs, and it is exactly what a limit on crossings withholds or grants. This is why I take the number of crossings to be the variable that tests relay between cells in this network.

### 3.6 Would a network in which only some cells are read be the fair test?

Not by that change alone, for three reasons.

1. In E47's network, which E53 takes up, a patch of 16 by 16 cells reads the byte and is read by the readout, among 1,024 cells that each read 10 cells of their neighbourhood (`exp47_local.py`). The cells of the first ring around the patch take part, and an error can reach them only by relay, one crossing from the patch. With the 540 cells of the second and third rings held at their mean, no score changed by more than 0.7 points (the reviewer's memo, section 28.4). A limit on relay would there decide whether the first ring learns, over one crossing. It would not test reach over several crossings, because the cells that lie further are not used; and the patch is by itself a network of the kind that learns the task (E42's `F10`). E53 asks whether the further cells come into use from another start. If they do, that network becomes the place where a limit on crossings tests reach; until then it is not.
2. A readout in which each unit reads few cells does not change this either. In E49's `out8` each of the 512 readout units reads 8 cells drawn at random (`e49_interface.draws`), about 16 readings a cell, so that every cell is still read.
3. The reason is general in this harness. One crossing takes one byte, forward as well as backward (the brief's own section on what one step is). The first answer byte is read out at the position of `=`, one position after the last digit of the second operand; a cell more than one crossing from the cells that receive that digit cannot hold it yet. A network in which relay over several crossings is the only route to cells that the task needs therefore requires several crossings a byte, forward and backward. That is a new network with no record beside it, and I would place it with the step on the readout (section 5.3), not in E61.

### 3.7 What I propose (mine; the specification and the release are the reviewer's and the coordinator's)

**Run form (b) on the brief's network, beside the same record: X = 0, 1, 2 and 4 under each rule.**

- *What it asks.* X = 0: does the network learn when each cell uses only the error that reaches it and its own path, with no credit passed between cells? X = 1, 2, 4: over how many crossings must credit be passed? The whole item is the record.
- *What X = 0 is.* Each connection of a cell needs one running quantity that the cell can update from its own gates at every position, and the error that reaches the cell at the answer positions; nothing is passed between cells and nothing is passed back over positions. In the literature this is learning with eligibility traces and a directly delivered error (section 7). For X of 1 or more, either a cell keeps its quantities of earlier positions and the error is passed back along them, as in the brief's form, or it keeps running quantities for the weights of the cells within X crossings of it. Form (b) is a measurement of how many crossings learning needs; I do not claim that its mechanism for X of 1 or more is what a neuron has.
- *The direct definition, for the brief's checks.* X + 1 copies of the states, equal in value. Copy v takes a cell's own earlier state from itself and the other cells' states from copy v + 1; the last copy takes the other cells' states as constants; the readout reads copy 0; the loss is differentiated once. In torch, with `own(h)` the diagonal of the recurrent map applied to `h` with the same backward value as the crossing, the recurrent input of copy v is `rec(other) + own(hs[v]) - own(other)`, with `other = hs[v + 1]` or, for the last copy, `hs[v].detach()`. The function `loss_levels` of the script is this definition in numpy, and the script's X + 1 backward passes agree with its central differences to 3.4e-08. The torch form is not executed here.
- *Cost.* Each further copy repeats the recurrent part of the forward and backward pass. From the multiply-adds that the reviewer's check prints, I count about 1.0, 1.2, 1.4 and 1.9 times the record's update for X = 0, 1, 2 and 4 under the exact gradient and about 1.1, 1.3, 1.5 and 2.0 under the sweep, or about 24 CPU-hours for the eight runs. This is a count and not a measurement; the brief's check 5 would settle it.
- *Which runs of the K-grid to keep.* K = 0, because K = 0 against X = 0 isolates the cell's own path, and K = 8, because it is the one value that tests the brief's open question (whether credit over short items teaches long ones). By the brief's own count K of 4 or less reaches both digits of the first operand in no training item, and K = 16 reaches them in 12 of 13; these eight runs (K = 1, 2, 4 and 16 under each rule) give the least for their hours. With the second economy the four runs kept cost about 9 CPU-hours, for about 33 in all, inside the brief's limit of 36.

*Predictions, mine, recorded before any run* (80 on R within 49,152 updates on seed 73129; by one seed these score a guess and confirm nothing):

| | Statement | Chance |
|---|---|---|
| A1 | With X = 0, neither rule reaches 80 on R | 0.75 |
| A2 | With X = 4, the exact gradient reaches 80 on R | 0.70 |
| A3 | With X = 4, the exact gradient reaches 80 on A as well | 0.55 |
| A4 | With X = 4, the sweep reaches 80 on R | 0.50 |
| A5 | Under each rule the level of R at update 49,152 rises from X = 0 through 1 and 2 to 4 | 0.65 |

### 3.8 Two smaller points on the brief

1. *The criterion is read on R alone.* The items of R have no letters, so that K of 5 or more reaches the first operand in every one of them, while the items of A have 12. A limit that is "sufficient" on R may leave A far behind; the brief foresees this in P61.10 and does not carry it into the fixed sentence. I would call a limit sufficient only if it also meets the criterion on A (within 1.25 times the record's 14,080 updates under the exact gradient and 35,584 under the sweep), or name the panel in every such sentence.
2. *The name of the variable.* If the K-grid is run, its question and its fixed sentences should call K a limit in positions and say that it is not the number of passes between cells.

## 4. E62

The brief is sound for what it states, and its check's table of the ten networks is reproduced here. Its own limit is the important one: every cell reads a value of every used byte and is read by every readout unit, so that N governs only how cells combine over positions what they hold.

1. **An addition that costs seconds a run.** Part 5 of the script shows that at N = 10 the record's networks answer nothing once a cell receives only the mean from the other cells. At N = 2 and N = 4, where 40 and 4 cells are read by no other cell, a network may reach 80 by another route, each cell alone with the dense readout. The two forward passes of section 3.5 at the end of each run would say which, and "N connections suffice" means different things in the two cases. The function `answered` of the script does this in numpy from a saved file; the mask of each N is in the file.
2. **One draw against a criterion of 1.25.** In E42, ten connections drawn from all cells needed 1.26, 0.84 and 1.02 times the updates of ten from the neighbourhood on three seeds, as the brief records, and the step list says that the later steps "take that N". Before an N is taken, I would repeat the smallest N called as fast, and the one below it, on a second draw of connections (two to four runs), or carry N forward as a range.
3. **The sweep's lag against N is confounded twice.** By the rate: on the dense network E52 found the lag at each rule's better rate nowhere larger and in 9 of its 18 readings smaller or gone (memo, 33.29), and the pair of runs at 0.003 on the record's network is before the coordinator. By the scale of the start, which changes with N, as the brief says. The third outcome of step 2 ("the working rule's shortfall depends on sparsity") should carry both.

## 5. The chain of steps

### 5.1 A statement that the receipts do not support

Step 4 gives as the record: "with a fixed code it learns only when each cell has an input for every byte, 2.68 times slower than back-propagation (E50)". *Computed here* (part 6) from the six receipts of E50: the sweep reaches 80 on R after 34,304 updates with 46 values a cell and after 12,800 with every value, a ratio of 2.68; the exact gradient after 15,360 and 6,912, a ratio of 2.22; the sweep against the exact gradient, both with 46 values a cell, 34,304 / 15,360 = 2.23. The figure 2.68 is the cost of the limited input under the sweep and not the sweep's lag behind back-propagation. The same sentence stands in the table of the reviewer's memo, section 33.23.

### 5.2 Requirement A does not depend on the chain where the chain places it

The chain places memory at steps 7 to 10, after the candidate is assembled, and gives as the outcome of step 8 "whether sparse connections alone change what is kept". The record already answers that question for sparse connections between cells. The following are the reviewer's verified statements, not recomputed here.

- E43's second addendum (memo, 28.6): "E42's networks lose the family as the dense one does: `F10` and `P8` are at or below 1% on reduction pairs by update 64. Limiting the connections between cells does not separate what is learned."
- Section 29.3, reading 1: every item drives every cell, also an item of bytes never met; "the limit removes connections, it does not make any cell idle". Reading 3: with a code in which a value stands for one byte, as in E50, a new byte would drive only connections that no earlier byte drives.
- Section 33.27: on the dense network, without a store and with decay, one intervening visit takes what is answered from about 0.8 to about 0.02 under both rules.

What follows is my reading. Nothing that steps 2, 3 and 6 vary (the number of connections between cells, the limit on relay, a cell's own number) makes a cell idle, so I expect none of them to bear on requirement A; for 10 connections a cell the record shows that it does not. What can bear on it is how the input reaches the cells, which is step 4, and how the readout reads them, which is in no step. The record also shows a tension at exactly this place. In the one limited input that both rules learn (E50's stage 3), every cell reads a value of every used byte, so that, with the same construction extended to its bytes, an item of any family would drive every cell. In the input in which a byte reaches only some cells (16 values a cell drawn at random from 512; a given byte then reaches about three cells in ten directly, by my arithmetic) neither rule reached 80 on R by the rule of three rows. Being learned and leaving cells idle have so far not been had with the same input. E57 bears on this directly.

I would therefore change three things in the chain.

1. Step 4 is read by a second criterion beside "is the family learned": how many cells, and how many connections of the recurrent map and of the readout, an item of another family would leave idle. The scripts of section 29 (`cells_in_use.py`, `directions_in_use.py`) measure this on saved weights with forward passes only; E50's saved networks are on the default branch, and E57's will be on its worker's.
2. Step 8 cites E43's second addendum and says what is new in it: the same protocol on a network whose input leaves cells idle.
3. Steps 2 and 3 are stated as measurements for requirements B and C that are not expected to change the plan for requirement A, so that their 65 to 70 CPU-hours are weighed as that.

This is the thesis of the advice memo (what is kept is decided by where an experience is written) applied to the chain; the chain is ordered by the rule, and the record says that the code decides. The project's working agreement (`AGENTS.md`) asks to "avoid serial prerequisite drift"; in the chain, steps 2 to 6 stand before the first measurement of memory.

### 5.3 Two things that the candidate of step 5 lacks

1. **A readout that is not dense.** Section 1.2 of the list names the output under requirement C. The candidate of step 5 is assembled from the connections between cells, the relay, the input and the rule's form; its readout would remain the dense one, with nine tenths of the weights in use, each readout unit reading every cell. On record is E49's `out8` (8 cells a readout unit; memo, 28.5): under back-propagation 74 on A and 71 on R, 80 not reached; under the sweep 10.6 and 7.8. By section 29's table, in the dense network the family's items use 243 to 248 of the 257 input directions of `lm`, as many as at the recurrent map, so the readout matters for requirement A as well. I would add it to step 4 as a third part.
2. **An input for bytes outside the first family.** The fixed code of E50 is drawn for the 46 symbols that the first family uses, and each cell's wiring is one value of each of those 46 (part 1 of the script). Step 8 asks for "a new task on other bytes", and the world of step 10 gives its 52 families two-byte codes from the bytes 128 to 255. With E50's finding as it stands, a cell would need a value of every byte of the alphabet. The list should say at which step the input for the whole alphabet is fixed, because it changes the network on which steps 2 and 3 are run. Step 9 has the same dependence: the candidate has no learned byte table, so "the rows of its own code bytes" would be the input weights at the values of the family's code bytes, and the step should say so.

### 5.4 Numbers chosen on a network with dense interfaces

Steps 2 and 3 read N and the limit on relay on a network in which every cell receives its input and its error directly, and the later steps "take that N" and "take that k". In the assembled candidate a cell must also obtain input and error through other cells. My expectation, which nothing on file shows, is that the values of steps 2 and 3 are lower bounds. Step 5 should read both again at two or three values around those chosen, and step 11's seeds should come before a value is carried forward, since each is chosen on one seed and one draw.

### 5.5 The branches

- *Step 3.* With form (b) the outcomes become: if X = 0 suffices, no relay between cells is needed in a network whose every cell is read; if a small X suffices, that many crossings; if neither, the network holds what it must remember by passing it between cells, and relay must follow it through the item.
- *Step 4* has no branch for the case that no form on file learns a sparse input. Step 5 then has no input to assemble; the list should say whether the chain continues with the input of one value of every byte, and that the candidate is then not local at its input.
- *Step 6.* "The number of connections is no longer ours" holds only in part: the rule brings numbers of its own (the minimum age, the rate of adding, the measure of weak). The step should count the numbers that we fix before and after.
- *Step 7* is read-only and need not wait for step 5: E62 will save eighteen networks at nine values of N, and the record's two exist now.

## 6. Findings by weight, limits and disposition

**FIRST_FALSE_ASSERTION.** In the briefs of E61 and E62: none found; the statements that I executed hold (section 2). In the chain of steps: step 4's "2.68 times slower than back-propagation" (section 5.1).

**SEVERITY.**

| | Finding | Weight | Ground |
|---|---|---|---|
| 1 | E61 as written does not test relay between cells; its outcome is largely fixed by the format of the item | MAJOR | Sections 3.2 to 3.5 |
| 2 | The chain places requirement A after steps that cannot bear on it, and does not cite the record's answer to step 8's question | MAJOR | Section 5.2 |
| 3 | The candidate of step 5 has a dense readout and an input for 46 symbols only | MAJOR | Section 5.3 |
| 4 | N and the limit on relay are carried forward from a network with dense interfaces, each from one seed and one draw | MINOR | Sections 4 and 5.4 |
| 5 | E61's criterion is read on R alone | MINOR | Section 3.8 |
| 6 | Step 4's figure 2.68 is given the wrong comparison | MINOR | Section 5.1 |
| 7 | Missing branches at steps 4 and 6; step 7 need not wait | NOTE | Section 5.5 |
| 8 | E62: the two forward passes at the end of a run | NOTE | Section 4 |

**GAPS.** No code in torch was run, so the brief's update was read and not executed here; the reviewer's own numerical test of it passes. One seed and one draw of connections throughout. The statements of sections 28.6, 29 and 33.27 of the reviewer's memo are taken as verified by the reviewer and were not recomputed. The reviewer's memo was read in the sections named and not as a whole. This is a same-family review.

**CAUSAL_LIMITS.** The tables of section 3.4 describe gradients in two networks that were trained with the whole item; they do not say what a learner trained under a limit would do, and a cosine is not a learning curve. The removal of section 3.5 shows that the record's solution uses what passes between cells; it does not show that no solution exists without it. The cost of form (b) is a count. The predictions are guesses on one seed. Nothing here bears on requirement A beyond the argument of section 5.2, which rests on the reviewer's measurements.

**DISPOSITION.**

- *E61:* REVISE before release: add the runs of form (b), keep K = 0 and K = 8 beside them, read the criterion on A as well, and name K a limit in positions.
- *E62:* ACCEPT_BOUNDED: release as it stands; the additions of section 4 are advice.
- *The chain of steps:* REVISE: the correction of section 5.1, the three changes of section 5.2, the two additions of section 5.3, and the branches of section 5.5.

This is an outside session's advice. It is not a checker gate, a seal or a cross-family review, and it authorises no run.

## 7. Literature to source-check

Cited from memory; none was opened for this document.

- R. J. Williams and D. Zipser, "A learning algorithm for continually running fully recurrent neural networks", *Neural Computation* 1 (1989): learning in real time, with the influence of every weight on every unit carried forward.
- R. J. Williams and J. Peng, "An efficient gradient-based algorithm for on-line training of recurrent network trajectories", *Neural Computation* 2 (1990): the gradient cut a fixed number of steps back, which is form (a).
- G. Bellec and others, "A solution to the learning dilemma for recurrent networks of spiking neurons", *Nature Communications* 11 (2020): a trace at each connection and an error delivered directly to each unit; with a symmetric readout this is form (b) at X = 0.
- J. M. Murray, "Local online learning in recurrent networks with random feedback", *eLife* 8 (2019): the same family, with random feedback.
- J. Menick and others, "Practical real time recurrent learning with a sparse approximation", ICLR 2021: the influence of a weight kept only on the units that it can reach within n steps; they report that in sparse networks a small n comes close to the full gradient. Form (b) is of this family and is not identical to it.
- O. Marschall, K. Cho and C. Savin, "A unified framework of online learning algorithms for training recurrent neural networks", *Journal of Machine Learning Research* 21 (2020): the place of these rules beside one another.
