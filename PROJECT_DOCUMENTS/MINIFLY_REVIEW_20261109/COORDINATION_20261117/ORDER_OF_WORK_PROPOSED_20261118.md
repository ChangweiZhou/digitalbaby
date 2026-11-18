# A proposed order of work, derived from the principle and from what the program was found to fix (18 November 2026)

**For** Mr. Xu, the project's owner, and Mr. Zhou, its coordinator. **Written by** an agent of the outside reviewer, at the reviewer's request, between 21:19 and 21:27 UTC on 18 November 2026 (clock read with `date -u` at 21:19, 21:23 and 21:26 UTC). **Status:** a proposal. It decides nothing, releases nothing and starts nothing. Every run named here needs a brief written by the reviewer, checked by a fresh agent and released by the coordinator. Nothing was run for this document; no existing file was changed.

## 0. How this was made

The work was done in two stages, so that the order would be derived from the principle and the findings before it could be shaped by anyone's proposed order.

**Stage 1.** I read only what states the principle, what was found to depart from it, and the state of the record: the reviewer's second review (`../REVIEW_OF_WHAT_THE_PROGRAM_FIXES_AGAINST_THE_PRINCIPLE_20261118.md`, sections 0 to 5 and 8, not section 6); the independent inventory (`../INDEPENDENT_INVENTORY_OF_WHAT_THE_PROGRAM_FIXES_20261118.md`, whole); the outside audit (`AUDIT_OF_E61_E62_AND_THE_STEPS_20261118.md` in the reviewer's scratch folder, sections 1 to 4 and 6, not section 5); and the reviewer's memo (`../MEMO_TO_COORDINATOR_LEARNED_STRUCTURE_20261115.md`), items 28 to 42 of its first section and sections 33.23, 33.25 to 33.27, 33.29 and 33.30. I then wrote down my own order (kept as a dated text file in the agent's scratch folder before stage 2 began).

**Stage 2.** I then read the review's section 6, the audit's section 5, the chain of steps (`STEPS_FROM_THE_PRINCIPLE_20261118.md`, revision 3), the coordinator's decisions 38 to 47 (`COORDINATOR_DECISIONS_20261118.md` on the default branch), and the head, terms, question and runs of E61's draft revision 3 with its note. I compared each with my order. Where they were right I changed my order (section 6 says where); where I think they are wrong I kept mine and say why.

To check two names I read the settings of `exp50_code.py` and the head of `exp47_local.py` on the default branch; nothing else outside the list above was read.

## Terms

- **The principle** is the owner's text as quoted in the task of this document and in the review's section 1 (P1 to P6). **Requirement A** (memory formed in the course of learning, some rehearsal acceptable), **B1** (slower at first, ahead later), **B2** (a similar result from far fewer samples) and **C** (learning local at the input and the output) are the owner's.
- **The record's network** is the network of E50's stage 3 (setting `C46-one`): 256 gated cells, each reading 10 cells of its neighbourhood and itself; a fixed input code of 512 values for the 46 bytes of the first family, of which each cell reads 46 values, one of every used byte; and **the readout**, which consists of `lm` (512 hidden units, each reading all 256 cells), **the direct path** `l2` (each hidden unit reading all 512 values of the current byte's code, without passing through any cell), and `l3` (257 output units, each reading all 512 hidden units). The readout holds 92.1% of the weights in use.
- **The input with 16 values a cell** is E50's setting `C16-in`: the same code, each cell reading 16 of the 512 values drawn at random. Neither rule reached 80 on R with it.
- **E47's network** is the network with an interface at a boundary: 1,024 cells, of which one patch of 16 by 16 reads the byte and is read by the readout (`exp47_local.py`, side 32, arm `one`).
- **An exchange** is one application of the recurrent map: each cell's new state is computed from its input and from the states of the cells it reads. The record makes **one exchange for each byte**. **R** is the number of exchanges for each byte; in rounds 2 to R the byte is held, the same weights are used, and the readout reads the state after the last round.
- **The sweep** is the working rule: the error passed backward through each linear map by the signs of its weights times one scale for the map, over the positions of an item. **The exact gradient** is back-propagation. By decision 38 the exact gradient on a network with limited connections is a **reference arm** and never a candidate.
- **A local rule** (item 7) is a rule in which no quantity is passed backward over the positions of an item after it ends: each connection keeps a running quantity computed from its own two ends during the item, and the error reaches a cell only over connections that exist, in a limited number of exchanges.
- **The assembled network** (item 5) is the network that items 2 to 4 select: a sparse input over all 258 byte values, R exchanges for each byte, and a readout limited as item 3 selects.
- **A, R, T, updates to 80**: as in E50 and E61 (a threshold is reached at the first of three consecutive scored rows at or above it). One CPU-hour estimate per run is a count from the facts given for this document (a run of 49,152 updates on the record's network costs about 2 CPU-hours; the recurrent part is under a tenth of an update's cost); each brief's cost check settles it.

## 1. The order in brief

1. **Measurements on the saved networks: what the readout computes, how far the last digit reaches before the answer, and what an item of another family leaves idle.** *Read-only, at once* (the reviewer's agents).
2. **Relay time at the input that failed: R = 1, 2, 4, 8 exchanges for each byte on the input with 16 values a cell, under both rules, with two controls.** *A run that can start as soon as its brief is written.*
3. **The readout made local on the record's network: the direct path removed, and each readout unit reading 256, 32 or 8 cells, or the output units reading 32 cells directly.** *A run that can start as soon as its brief is written.*
4. **Relay time where the error must be relayed: R = 4 and 8 on E47's network, in which only the patch is read.** *A run that can start as soon as its brief is written* (beside 2 and 3).
5. **The assembled network: a code over all 258 byte values with 16 values a cell, R and the readout from items 2 to 4; births under both rules; the number of connections N and of input values a cell varied on powers of two.** *Waits for items 2 and 3* (and reads item 4).
6. **Requirement A, first measurement: a second family, without a store, on the assembled network, with and without weight decay, beside what the record keeps.** *Waits for item 5.*
7. **A local rule on the assembled network: running quantities at each connection, the error relayed over existing connections in a limited number of exchanges, a scale for each receiving cell; this replaces E61.** *Waits for item 5 for its runs; its code and its numerical checks can be written at once.*
8. **What the program's handling of items does: one change of the weights per 64, 8 or 1 items; the state carried from one item to the next; plain steps in place of AdamW.** *Waits for item 5.*
9. **Requirement B: the samples and updates the local rule needs for 80, at half, a quarter and an eighth of the pairs and at rates 0.001, 0.002 and 0.004, beside back-propagation on the dense network.** *Waits for item 7.*
10. **Each cell sets its own number of connections (the chain's step 6), on the assembled network under the local rule.** *Waits for items 5 and 7.*
11. **A life on the assembled network under the local rule, with and without a store, beside the conventional learner with a store of equal memory; the life runner extended.** *Waits for items 6, 7 and 9.*
12. **E61 in either form on the record's network; the limit on relay on E53's network at one exchange a byte (decision 40); any new job on the dense network; the two runs at the rate 0.003 on the record's network now.** *Not to be run.*

## 2. The items

### Item 1. Measurements on the saved networks (read-only, at once)

**Question.** In the networks the record calls sparse, how much of an answer is computed by the readout rather than the cells, how many cells hold the last digit when the first answer byte is due, and how many cells and connections an item of another family leaves idle?

**Why here.** The review's findings 1 and 2 and the inventory's entries 2 and 3 are deductions from the source; every later item is read against them (P1, P3, requirement C). The audit's section 5.2 and the chain's step 7 show that requirement A depends on what an item leaves idle, which can be measured now.

**Networks and measurements.** Saved final weights of E50's stage 3 (`C46-one`, S and B), E50's `C16-in` (S and B), E50's `C-dense`, E42's `F10` and `G` (seed 73129), E47's network, and, as they end, E53's six and E62's eighteen networks. Five measurements, each a forward pass in numpy (no training):

- (a) *The direct path.* Scores of A, R and T (i) as saved; (ii) with `l2 z1` replaced at each position by its mean over the panel's items; (iii) with `l2 z1` removed; (iv) with the cells silenced (the forward's `silence=True`, which sets the cells' contribution to zero), which gives what the direct path alone answers.
- (b) *Where the operands are combined.* A linear map from the cells' states at the three answer positions to the 257 outputs, fitted by least squares on taught items, scored on A and R beside the full readout. A small gap says the cells hold the answer; a large gap says the hidden layer of the readout computes it.
- (c) *Reach of the last digit.* For each item of R, the second digit of the second operand is changed and the cells whose state at `=` changes by more than 1e-3 are counted, split into those that read that byte directly and the others. On `C16-in` the review's shares (30% directly, 98% after one exchange) are the expected values.
- (d) *Idle cells and connections.* With the scripts of the memo's section 29 (`cells_in_use.py`, `directions_in_use.py`), the cells, the recurrent connections and the input directions of `lm` that an item of another family drives: an item of the first family's bytes under another modulus, and an item that contains bytes outside the 46 (which by B6 give no input at all; the count of such bytes in the long-life world's items is given).
- (e) *The audit's two forward passes* (each cell keeps its own path and receives from the other cells nothing, or their mean over items) on each network of E62 as it ends, and the share of the cells of E47's and E53's networks, by ring around the patch, whose replacement by the mean changes a score by more than one point.

**Compared with.** The receipts' last scored rows; the audit's section 3.5 (0.00 to 0.83% with the mean from other cells); the memo's sections 28.4 (rings two and three unused in E47) and 29.3.

**Checks that the measurement computes what is meant.** The digest of each weight file equals its receipt's. The unmodified forward reproduces every scored count of the receipt's last row exactly (the audit did so for `C46-one`). Replacing `l2 z1` by itself (not by its mean) reproduces the receipt. On a small random network, the count of (c) equals a count made by finite differences.

**Fixed wording of the result.** "On [network], with [the direct path replaced by its mean over items / removed / the cells silenced], the share right is A [a], R [r], T [t], against [a0], [r0], [t0] as saved. A linear map from the cells' states answers A [x] and R [y]. At `=`, [n] of 256 cells carry the last digit, [d] of them by reading it directly. An item of [another family] leaves [c] cells and [k] connections undriven. These are measurements on trained networks; they say what these networks use and not what a network without the part would learn."

**Runs and cost.** No training; about 2 to 4 CPU-hours of the reviewer's agents in all, with no file larger than a few megabytes written; the weight files are read one at a time (disk: 1.2 GB free).

**Outcomes.** If (a) shows that the cells silenced still answer much, or that removing the direct path costs little, item 3's arms are read with that in mind. If (b) shows a large gap, the hidden layer of the readout computes part of the answer, and every "result on limited connections" of the record is quoted as a result on cells together with a dense two-layer network. If (d) shows that `C16-in` leaves many cells idle for another family's items, requirement A has a ground on that input, and item 2 decides whether it can be learned.

**What it leaves fixed by us.** Nothing new; it measures structures fixed earlier.

**Depends on / depended on by.** Depends on nothing. Items 2, 3 and 4 are read with it; item 6 uses its method (d).

### Item 2. Relay time at the input that failed (run; brief can be written now)

**Question.** Does the input with 16 values a cell, which neither rule learned with one exchange for each byte, become learnable when the cells exchange R times for each byte?

**Why here.** Finding 1 of the review: with one exchange a byte and the answer due one byte after the last digit, a cell that does not read the last digit can receive it over one exchange only. P3 describes many exchanges between two inputs. Every conclusion of the record on sparse input (E49, E50's stages 2 and 3) holds under this structure, and the sparse input is the condition for both requirement C at the input and requirement A (the audit's section 5.2: being learned and leaving cells idle have so far not been had with the same input). Everything after this item is read on the answer.

**Network and variable.** The record's `C16-in` network (E50's stage 2: 256 cells, 10 connections a cell from the neighbourhood, 16 of 512 code values a cell drawn at random, the dense readout unchanged), seed 73129, half the pairs, 49,152 updates, rate 0.001, the record's start, stream, batches, optimizer, panels and scorer. Only the number of exchanges changes. The variable is R, on the grid 1, 2, 4, 8:

- R = 2, 4, 8 under each rule (six runs). In rounds 2 to R the byte's input term is held, the same weights and biases are used, and the state passes through the full gated update in each round. The readout reads the state after the last round.
- *Control 1, more depth without relay* (one run a rule): R = 4 in which, in rounds 2 to 4, each cell reads only itself (the off-diagonal entries of the recurrent map are not applied). It separates relay between cells from the repeated application of the gated update.
- *Control 2, time before the answer* (one run a rule): R = 1 with four neutral bytes (a byte with a zero code, without target) between `=` and the first answer byte, in training and in scoring. It separates the time between the last digit and the answer from exchanges between every pair of bytes (the inventory's entry 3).
- R = 1 is the record (E50's `C16-in` runs) and is not repeated.

**Compared with.** E50's `C16-in` (R = 1; neither rule reached 80 on R), E50's `C46-one` (every cell reads every byte: 34,304 updates to 80 on R under the sweep, 15,360 under the exact gradient), E57 when read (a scale for each receiving unit, at R = 1), and the dense reference (E42's `G`: 6,400).

**Checks that the runner computes what is meant.** (1) With R = 1 and no neutral bytes, the forward and the loss of the first batch equal the record's bit for bit (5.569, the setting's stored value), under both rules, and the first 256 updates reproduce the `C16-in` receipt's batch and loss digests. (2) With R rounds, a forward written directly as a loop over positions and rounds equals the runner's forward to 1e-6, and under the exact gradient the runner's gradient agrees with central differences on a small network. (3) Control 1's recurrent input in rounds 2 to R equals the diagonal of the masked recurrent map applied to the state. (4) The neutral byte gives zero input to every cell and carries no target. (5) The masks are unchanged at every 256 updates. (6) A cost check over 64 updates for each arm; limit 30 CPU-hours for the ten runs.

**Runs and cost.** Ten runs: about 2.2, 2.6, 3.4 CPU-hours for R = 2, 4, 8, 2.6 for control 1, 2.4 for control 2, per rule; about 26 CPU-hours, about 11 hours on one line. Order: R = 8 and R = 2 under the exact gradient and the sweep first; then R = 4; then the controls.

**Fixed wording of the result.** "On the network of E50's stage 2 (each cell reads 16 of the 512 values of a fixed code; dense readout), with R exchanges between cells for each byte, rule [S/B] reaches 80 on R after [u] updates and on A after [v] [or: does not reach 80 within 49,152 updates; levels at the end A [a], R [r]], against no threshold reached with one exchange (E50). With the same number of rounds and no exchange between cells after the first, it reaches [..]. With one exchange a byte and four neutral bytes before the answer, it reaches [..]. Under the sweep the error passes backward over every round of every position; this is one global backward pass, and the readout reads every cell and the current byte." The words *sufficient* and *not reached* are used as in E61's revision 3, on R and A.

**Outcomes.**
- If some R learns and control 1 does not: relay between cells is what the record's sparse input lacked; the smallest such R is carried to item 5 with the next power of two beside it.
- If control 1 learns as well as R: the gain comes from repeated gated updates and not from relay; item 5 carries R and states this.
- If only control 2 learns: the obstacle is the format of the item (the answer due at once); item 5 takes neutral bytes or the smallest R, and the owner is asked which agrees with the principle.
- If none learns under either rule: the chain's step 4 has no input that leaves cells idle and is learned. Item 5 then continues with an input in which each cell reads one value of every byte value (the construction of `C46-one` over all 258 byte values), stated as not local at the input; requirement A is then tested on the readout and the connections only.
- If the exact gradient learns and the sweep does not: the record's lag of the sweep is again a matter of the rule, and item 7 is given priority over item 6.

**What it leaves fixed by us.** The input code, the 16 values a cell, N = 10, the neighbourhood, the dense readout, the reset of the state at every item, 64 items an update, AdamW, the rate and the decay, the start scaled for one exchange.

**Depends on / depended on by.** Depends on nothing (item 1 informs its reading). Items 5 to 11 depend on it.

### Item 3. The readout made local on the record's network (run; brief can be written now)

**Question.** Does the record's network learn the first family when the byte no longer reaches the readout directly and each readout unit reads only some cells?

**Why here.** Finding 2 of the review: nine tenths of the weights of the network called sparse are a dense readout of two layers that reads every cell and the current byte. Requirement C names the output; P1 forbids a unit that reads every cell. The one run on record (E49's `out8`, 8 cells a readout unit with `l2` and `l3` kept dense) reached 74 and 71 under the exact gradient and 10.6 and 7.8 under the sweep. Every later item needs a readout that is not dense.

**Network and variable.** The record's network (`C46-one`, R = 1), seed 73129, half the pairs, 49,152 updates, the record's start, stream, optimizer and scorer. Four arms under each rule:

- (i) `l2` removed; `lm` dense (each hidden unit reads 256 cells); `l3` dense.
- (ii) `l2` removed; each hidden unit reads 32 cells (drawn as in `e49_interface.draws`); `l3` dense.
- (iii) `l2` removed; each hidden unit reads 8 cells; `l3` dense.
- (iv) No hidden layer: each of the 257 output units reads 32 cells directly through one masked map.

The grid of the number of cells a readout unit reads is 256, 32, 8 (powers of two; 128 and 16 can be added if the curve is steep between).

**Compared with.** The record's `C46-one` runs (with `l2`, dense `lm`); E49's `out8`; the dense reference.

**Checks that the runner computes what is meant.** (1) With `l2` kept and both masks full, the forward and the first-batch loss equal the record's bit for bit (5.590670). (2) In each arm the count of weights in use equals the count computed from the masks, and is stated. (3) Gradients outside the masks are zero; the masks and the zero entries are unchanged at every 256 updates. (4) Under the sweep the scale of each masked readout map is the mean magnitude inside its mask, as `e42_cells.sign_scale` does for the recurrent map. (5) Each cell is read by at least one readout unit in arms (ii) to (iv), and the number of readings per cell is stated. (6) A cost check; limit 14 CPU-hours.

**Runs and cost.** Eight runs; removing `l2` removes 46% of the weights in use, so about 1.2 to 1.5 CPU-hours a run, and about 1 for arm (iv); about 10 CPU-hours, about 4 to 5 hours on one line.

**Fixed wording of the result.** "On the network of E50's stage 3 without the direct path from the byte to the readout, [with each readout unit reading k cells / with each output unit reading 32 cells and no hidden layer], rule [S/B] reaches 80 on R after [u] updates and on A after [v] [or: does not reach 80; levels at the end], against [15,360 / 34,304] with the dense readout and the direct path. In every arm each cell reads one value of every used byte and the error is passed backward over the item."

**Outcomes.** If arm (iii) or (iv) learns under the sweep, that readout is carried to item 5. If only arm (i) learns, the direct path is not needed but reading every cell is; item 5 takes arm (ii)'s 32 cells and the brief of item 5 adds 64 and 128. If no arm without `l2` learns, the record's results on limited connections depended on the direct path, and item 5 must be designed with the readout as its first variable; the owner is told that requirement C at the output fails on the record's network.

**What it leaves fixed by us.** The input of `C46-one` (every cell reads every used byte), one exchange a byte, N = 10, the normalisation of the 257 outputs, the reset, 64 items an update, AdamW, rate and decay.

**Depends on / depended on by.** Depends on nothing. Item 5 depends on it.

### Item 4. Relay time where the error must be relayed (run; brief can be written now)

**Question.** In E47's network, where only the patch reads the byte and is read by the readout, do cells beyond the first ring come into use when the cells exchange R times for each byte?

**Why here.** This is the network in which a cell away from the patch receives the input and the error only by relay (P3), which the coordinator's decision 40 names as the fair test. By the review's finding 1 and the audit's section 3.6, at one exchange a byte such cells cannot serve the last operand at any size of the start, so that the test of decision 40 at R = 1 could show only one exchange. Giving the network time is the condition for that test.

**Network and variable.** E47's network as the record ran it (side 32, arm `one`, seed 73129, its start, decay and dose), with R = 4 and R = 8 under each rule (four runs); R = 1 is E47's record, and E53's runs (larger start, no decay) are the second reference. If E53 shows outer cells in use at R = 1, the coordinator may move the four runs to E53's start.

**Compared with.** E47's runs and the memo's section 28.4 (rings two and three held at their mean change no score by more than 0.7 points); E53's six runs.

**Read.** Updates to 80 and levels; by item 1's method (e), the share of cells of each ring whose replacement by the mean changes a score.

**Runs and cost.** Four runs; by count about 5 CPU-hours at R = 4 and 8 at R = 8 (the recurrent map is four times larger than the record's); about 26 CPU-hours, about 9 to 10 hours on one line. The brief's cost check settles it.

**Outcomes.** If the outer rings come into use at R = 4 or 8, the limit on relay (the number of exchanges over which an error is relayed) has a network on which it tests reach, and it is measured there under the local rule of item 7. If they do not, the reason is not the time for relay, and the network's start or rule is examined before any test of relay over several cells.

**What it leaves fixed by us.** As E47, with R added.

**Depends on / depended on by.** Depends on nothing; E53 adds a reference. Item 7's limit on relay reads it.

### Item 5. The assembled network (waits for items 2 and 3)

**Question.** Does a network with a sparse input over all byte values, R exchanges a byte and a local readout learn the first family, and with how many connections and input values a cell?

**Why here.** The candidate of the chain's step 5 lacks a local readout and an input for bytes outside the first family (the audit's 5.3; the review's B6 and B8). Requirements A, B and C are to be measured on one network that is local at input and output; this is that network under the reference rules.

**Network and variable.** A fixed code over all 258 byte values, drawn without reference to the first family's alphabet (each byte turns on 11 of 512 values, overlaps allowed), each cell reading 16 values drawn at random, without the program's list of needed bytes and without the refusal of a run (B8); R and the readout from items 2 and 3. Varied: N = 4, 16, 64 connections a cell (from all cells) and input values a cell 8, 16, 32, each at the others' middle value; both rules; seed 73129; about 10 runs.

**Compared with.** Items 2 and 3; `C46-one`; the dense reference; E62's curve (read as the curve on the record's network).

**Runs and cost.** About 10 runs, about 25 CPU-hours.

**Outcomes.** A network on which both reference rules learn, with N and values a cell carried as ranges (the audit's 5.4: one draw and one seed do not fix a number); its saved networks are measured by item 1's method (d) at once. If nothing learns, the branch of item 2's last outcome applies.

**Fixed by us.** The cell's form, the code, R, the readout's size, the reset, the batch, AdamW, rate, decay.

**Depends on** items 2 and 3 (and reads 1 and 4). Items 6 to 11 depend on it.

### Item 6. Requirement A, first measurement (waits for item 5)

**Question.** On a network whose input leaves cells idle, is anything of the first family kept after a second family is learned without a store?

**Why here.** The record shows that limiting connections between cells does not change what is kept, because every item drives every cell (memo 28.6 and 29.3; audit 5.2). An input that leaves cells idle is the first change that could. P5: memory forms in the course of learning.

**Network and variable.** The assembled network under both rules, born on the first family; then a second family on other bytes, and one on the same bytes with another modulus; without a store; with decay 0.25 and 0 (decision 42); kept at the first update of a return and over the first 16 updates; about 8 runs of short length (4,096 updates of the second family), about 8 CPU-hours.

**Compared with.** Memo 33.27 (dense network, decay, no store: 0.8 falls to about 0.02 after one visit); E43's second addendum; item 1 (d) on the same networks.

**Outcomes.** Any amount kept at an absence of two visits beyond 0.02 is new (33.27). If nothing is kept although cells are idle, the readout or the shared weights carry the loss, and the readout's share of the change is measured read-only. If something is kept, it is measured under the local rule (item 7) and in a life (item 11). The chain's step 9 (the learner's own code, the outside memo's 5.2) follows this item as decision 43 says.

**Fixed by us.** As item 5; the schedule of one family at a time.

**Depends on** item 5. Item 11 depends on it.

### Item 7. A local rule (code now; runs wait for item 5)

**Question.** Does the assembled network learn the first family when no quantity is passed backward over the item after it ends, and the error reaches a cell only over connections that exist, in a limited number of exchanges?

**Why here.** Finding 6 of the review and the inventory's first entry: both rules on record are one global backward pass (P3), and four fifths of the sweep's crossings go where no connection exists. With R exchanges a byte, an error can be relayed while the item is read; the owner's own description (1 → 50 → 50² → 50³) is this. The audit's form (b) at X = 0 corresponds to a rule with running quantities at each connection and a directly delivered error (its section 3.7 and the literature of its section 7, not source-checked here).

**Rule and variable.** Each connection keeps a running quantity of its own (the influence of its weight on its cell's state, computed forward from the cell's own gates). At each answer position the error formed at the output units travels backward over connections that exist (a cell receives error from the cells it is connected to in the reverse direction, by feedback weights it holds), for E = 1, 2, 4, 8 exchanges (the grid replaces E61's K and X); each weight changes by its running quantity times the error its cell holds; one scale for each receiving cell (E57's form, if E57 supports it). Two forms of the feedback weights: fixed signs and learned (E31). About 8 runs plus two reference runs (the exact gradient and the sweep on the same network); about 20 CPU-hours.

**Checks.** With unlimited E and exact feedback weights, on a small network, the rule's update equals real-time recurrent learning's to 1e-8; no quantity of a later position is read at an earlier one (asserted in the code); the error reaches no cell over a pair without a connection (asserted from the masks).

**Outcomes.** If some E learns: the rule replaces the sweep as the candidate, for the owner to rule. If none does: which of the parts (running quantities, relay of the error, feedback form) fails is read from the reference runs, and the sweep stays the working rule as the owner keeps it.

**Fixed by us.** E, the feedback form, the scale's form, the output normalisation (an arm with an error for each output unit without normalisation is added if the cost allows).

**Depends on** item 5 for its runs; item 4 for the network on which reach is tested. Items 9 to 11 depend on it.

### Item 8. What the program's handling of items does (waits for item 5)

**Question.** Do results change when each change of the weights follows one item rather than 64, when the state is carried from one item to the next, or when AdamW is replaced by plain steps?

**Why here.** Finding 3 of the review (A4, D1): a neuron meets one item after another and is not reset from outside; samples (B2) are counted in items. D2 and D5: AdamW's equal step and its shared counter are ours. Batch size is a number fixed by us (P6) and is varied on powers of two.

**Variable.** Items per change 64, 8, 1 at equal items seen; one arm with the state not reset between items (the start symbol kept); one arm with plain steps; on the assembled network under the sweep and the exact gradient; about 8 runs. Batch 1 runs slowly on four cores; cost about 25 CPU-hours, settled by the brief.

**Outcomes.** If one item a change learns as fast in items, the batch is not a structure the result depends on; if not, every later figure of samples is stated per batch size. The arm without reset says whether a state can carry anything across items.

**Depends on** item 5. Items 9 and 11 read it.

### Item 9. Requirement B (waits for item 7)

**Question.** How many taught pairs and updates does the local rule need to reach 80, against back-propagation on the dense network?

**Variable.** Half, a quarter and an eighth of the pairs; rates 0.001, 0.002 and 0.004 (the rate as a number fixed by us; memo 33.29 shows part of the sweep's lag on the dense network is the rate); the local rule and the dense reference; two more seeds for any reading that carries a conclusion (the chain's step 11). About 15 runs, about 35 CPU-hours.

**Outcomes.** B2 is met where the local rule reaches the reference's level from fewer pairs; B1 at a birth is read here and over a life in item 11.

### Item 10. Each cell sets its own number of connections (waits for items 5 and 7)

The chain's step 6, unchanged in design (dropping and adding separated; starts from 2 and from 128 connections), on the assembled network under the local rule, and with the audit's 5.5 count of the numbers the rule brings. About 4 runs, about 10 CPU-hours.

### Item 11. A life (waits for items 6, 7 and 9)

The chain's step 10 on the assembled network under the local rule, with and without a store, beside the dense network under back-propagation with a store of equal memory, with the probe of new families at 64, 256 and 1,024 visits. The life runner (`smallworld/sw_run.py`) takes the dense network only and must be extended; that work can begin once item 5 is read. Cost to be estimated in its brief; the first long lives cost several hundred CPU-hours.

### Item 12. Not to be run

- **E61, form (a) or form (b), on the record's network.** With the dense readout every cell receives the error directly before any relay (the audit's 3.2), and one exchange a byte makes a position and a crossing the same step (the review's finding 1). Form (b) measures how many crossings of credit the record's solution needs; that question returns, on a network where it means relay, as the variable E of item 7 and on item 4's network.
- **The limit on relay on E53's network at one exchange a byte (decision 40).** By finding 1 it can test one crossing only; item 4 first.
- **Any new job on the dense network** (decision 44 is kept).
- **The two runs of the whole item at the rate 0.003 on the record's network** (decision 39 (ii)). The rate is varied in item 9 on the network that is to carry the result.
- **The chain's step 8 in its stated form** ("whether sparse connections alone change what is kept"): the record answers it (memo 28.6); item 6 asks the open question.

## 3. The running and held jobs, and the chain's steps

| Job | Proposed | Read as |
|---|---|---|
| E53 (ends about 00:20 UTC on 19 November) | Finish. | Whether a larger start without decay brings the outer rings of E47's network into use with one exchange a byte. By finding 1 a negative is expected and says nothing about relay; it is a reference for item 4, whose start it may change. Its networks enter item 1 (e). |
| E57 (nine runs, well advanced) | Finish. | Whether a scale for each receiving unit lets the sweep learn the input with 16 values a cell at one exchange a byte. Read beside item 2; its form of the scale is the default of item 7. |
| E60 (ends about 22:05 UTC) | Finish. | A statement about the sweep and the conventional learner on the dense network, the reference line; no follow-up on the dense network. |
| E62 (eighteen runs, about 35 CPU-hours, started 20:46) | Finish; it is running and does not hold a line that items 2 to 4 need. If a fourth line is wanted before it ends, the coordinator may stop it after the runs in progress. | The number of connections on the record's network, in which every cell reads every used byte and the readout reads every cell and the current byte, so that N governs only how cells combine what they hold over positions. Its N is not carried forward; N is varied again in item 5. Item 1 (e) adds the audit's two forward passes on each saved network. |
| E61 (released earlier, held by the owner; revision 3 drafted, not released) | Do not release revision 3 or revision 2 with amendments. Keep the note and `e61_revision_3_check.py` as verified record of the audit's computation. | Its question of form (b) becomes the variable E of item 7; form (a) is not run. |

**The chain's steps.** Step 1 stands as done, quoted with finding 2's limit. Step 2 (E62): finishes, not carried forward; N moves to item 5. Step 3, first part (E61): dropped (item 12); second part (decision 40): replaced by item 4, then item 7 on item 4's network. Step 4: first part (E57) read beside item 2; second part (feedback learned against signs) moved into item 7; items 2 and 3 are added in front of it as its input and output parts, with the branch the audit's 5.5 asks for (item 2's last outcome). Step 5: becomes item 5, with the readout and the input over all byte values added (audit 5.3). Step 6: item 10. Step 7: moved forward and made read-only at once (item 1 (d)) and repeated on item 5's networks. Step 8: becomes item 6, with its question restated and E43's second addendum cited. Step 9: after item 6, as decision 43 says. Step 10: item 11. Step 11: applied within items 2 to 9 to every reading that carries a conclusion.

## 4. The next 24 hours (from about 22:00 UTC on 18 November)

**Decision points.**
- **D1, now (owner and coordinator):** whether this order replaces the chain as the order of work (decision 44), and whether E61's revision 3 stays unreleased. Nothing below starts without it.
- **D2, at the release of items 2 to 4 (coordinator):** each brief as checked; the cost limits.
- **D3, about 00:30 UTC (coordinator):** E53 read; whether item 4 keeps E47's start or moves to E53's.
- **D4, about 12:00 to 16:00 UTC on 19 November (coordinator, owner informed):** items 2 and 3 read; R and the readout for item 5, or the branch of item 2's last outcome.
- **D5, when the code of item 7 passes its checks (owner):** whether the local rule may be put beside the sweep as a candidate on item 5's network.

**The three free lines.**

| Line | 22:00 to about 00:00 | Then |
|---|---|---|
| 1 | Waits for item 2's brief (reviewer writes about 22:00 to 23:00; fresh agent checks; coordinator releases) | Item 2: checks (about 1 CPU-hour), then ten runs in four waves; ends about 12:00 UTC on 19 November. |
| 2 | Waits for item 3's brief | Item 3: checks, then eight runs; ends about 05:00 to 06:00 UTC. Then, by decision 45, the worker stops; the line waits for item 5's brief (D4). |
| 3 | Waits for item 4's brief | Item 4: checks, then four runs; ends about 10:00 to 14:00 UTC. |

E60's line frees about 22:05 and E53's about 00:20; E57's soon after. The proposal for them: E60's line writes and checks the code of item 7 in numpy and torch on small networks (no long run), and E53's and E57's lines are held for item 5, so that it can start at D4 without waiting for a line.

**The reviewer's own agents, read-only, in the same time.** Item 1 (a) to (d) on the saved networks of E50, E42 and E47 (about 22:00 to 02:00); item 1 (e) on E53's networks after 00:20 and on E62's as each ends; the verification of E60, E53 and E57 as each ends; the "What is taken as given" table (review section 7) for each of the briefs of items 2 to 4; a first draft of item 5's brief, completed at D4.

## 5. Where the requirements are first measured, and what departures remain

| Stage | Requirement first measured | Departures from the principle that remain |
|---|---|---|
| After items 1 to 4 | C at the input (item 2: a byte reaches about three cells in ten directly) and C at the output (item 3); B1 at a birth as updates to 80 against the record, in each item | One global backward pass (both rules; the sweep's three parts); a scale for a whole matrix; in item 2 the dense readout and the direct path; in item 3 every cell reading every used byte; one exchange a byte in item 3; the code drawn by the program for the first family's alphabet; the state reset at every item; 64 items a change; AdamW's equal step and shared counter; uniform decay; the normalisation over 257 outputs; the target placed by the program and the answer fed as input; the fixed form of the cell |
| After item 5 | C at input and output together on one network; B1 at a birth | As above, without the dense readout, the direct path, and the alphabet built into the input; one global backward pass remains |
| After item 6 | A, without a store, under the reference rules | As after item 5; memory is measured with a rule that is not local |
| After item 7 | A, B and C under a rule with no pass backward over the item | The feedback weights' form (fixed or learned) and the scale chosen by us; the normalisation over 257 outputs unless its arm is run; reset and batch unless item 8 removes them |
| After items 8 and 9 | B2 (samples) and B1 at a birth under the local rule | The target given at the answer positions (the task's teaching signal), the form of the cell, and the numbers still fixed by us (R, E, N within ranges, values a cell, rate, decay) |
| After items 10 and 11 | A and B1 over a life | The store, where used, is the program's (rehearsal, accepted in measure by P5); the schedule of families is the world's; the cell's form and the remaining numbers (R, E, rate) |

The learner under test becomes, by the principle, the learner the owner asks for only at item 7 with items 8 and 10 applied: before item 7 every learner on file, the record's and the assembled one, receives its error through one global backward pass. Even then some numbers remain fixed by us (R, E, the rate, the cell's form), on grids of powers of two, and it is not shown before item 11 that the learner's memory forms in the course of a long life.

## 6. Where this order differs from the others

**From the review's section 6.**
- *Agreed and adopted from it:* relay time first; the readout measured read-only now; E61 held; E62 finished; the audit's three changes for requirement A; the arms for one item an update with the state carried, and for plain steps (item 8). Its item 7's neutral bytes before the answer are adopted, but as a control inside item 2 rather than as a second form of it.
- *Different:* the review puts the readout as a read-only measurement and a part of step 4; I make the readout a run that starts at once (item 3), because no later item can satisfy requirement C at the output and the one run on record failed under the sweep. The review asks for one arm with one item an update; I vary the number of items a change on 64, 8, 1, because it is a number fixed by us (P6). The review names no item for the rule itself; I add item 7, because finding 6 (the review's own) concerns the part of the learner most clearly prohibited by P3, and R exchanges make a rule without a pass after the item possible. I add a control without relay to item 2, because repeated gated updates with the byte held could help for another reason than relay.

**From the audit's section 5.**
- *Agreed and adopted:* the correction of 2.68 (5.1); requirement A placed where the input leaves cells idle, not after the rule's steps (5.2); the readout and the input over all byte values added to the candidate (5.3); N and the limit on relay read again on the assembled network as ranges (5.4); a branch where no sparse input is learned, and step 7 read-only at once (5.5).
- *Different:* the audit's section 3.7 (read beside 5) proposes form (b) of E61 on the record's network as the next runs; I do not run it there, because the audit's own sections 3.2 and 3.6 show that on that network every cell is reached by the error directly and one crossing is one byte, so the result would not be about relay as P3 describes it. The audit expects that steps 2, 3 and 6 do not bear on memory; I agree, and I add that the readout (item 3) may, since by its section 5.3 the family's items use nearly all input directions of `lm`.

**From the chain of steps (revision 3).**
- The chain starts with the number of connections (step 2) and the limit on relay (step 3) on the record's network. I put first the two structures that the review found never examined (one exchange a byte; the dense readout), because the chain's steps 2 and 3 are read on them.
- The chain places memory at steps 7 to 10 after the candidate; I place it (item 6) as soon as an input exists that leaves cells idle, before the local rule, following the audit's 5.2.
- The chain has no step for the rule's pass over the item except E61's limit; I give it item 7.
- The chain carries N and K forward from one draw and one seed; I carry ranges.

**From the coordinator's decisions 38 to 47.**
- Decision 38 (the exact gradient as a reference arm) is kept throughout.
- Decision 39 (E61 released with three amendments): I propose not to run E61 in any form now, for the reason given in item 12; the amendments (reading on A as well as R; K named a limit in positions) are carried into the wording of items 2, 3 and 7.
- Decision 40 (relay tested on E53's network when E53 is verified, at one exchange a byte): replaced by item 4, because at one exchange a byte that test can show only one crossing; the line kept free for it is used for item 4.
- Decision 41 (E62 released): kept; it finishes, and its N is not carried forward.
- Decision 42 (no-decay arms into step 8): kept, in item 6.
- Decision 43 (5.2 after step 8; the note of decision 15 not written): kept, after item 6.
- Decision 44 (the chain is the order of work; no new dense job): the second part is kept; the first part is what this proposal asks the owner and coordinator to reconsider (D1).
- Decision 45 (briefs name wait points): kept; item 3's brief names its end as a point where the worker stops.
- Decision 46 (the two free lines kept for decision 40 and step 4's brief): replaced by items 2 to 4 on the three free lines.
- Decision 47 (merges): not affected.

**What I could not judge.**
- Whether R exchanges with the byte held is what the owner means by relay between two inputs, rather than, for example, cells updating at their own times; item 2's control 1 only separates relay from depth.
- The cost of items 4, 7 and 8: E47's network and batch-1 updates were not timed; my figures are counts.
- Whether the gated cell's update repeated R times needs another start scale; the record's start is kept in item 2, which may disadvantage large R.
- Whether a code over all 258 byte values with overlapping values (item 5) is learnable at 16 values a cell; item 2 uses the record's code so that it changes one thing.
- E57's result, which may change the scale used in items 2 and 7.
- The literature that the audit cites for item 7's kind of rule; none of it was opened.
- I read the documents named in section 0 in the parts named, not the whole memo, the briefs of E53, E57, E60 and E62, or the runners; facts about them are as the documents state.
