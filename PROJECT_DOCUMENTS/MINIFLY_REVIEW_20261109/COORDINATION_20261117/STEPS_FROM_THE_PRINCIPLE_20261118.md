# The experiments in order, from the biological principle (18 November 2026, revision 1)

**What this is.** Mr. Xu asked on 17 November for a list of everything that can be done, in order, to be done one item after another, and on 18 November repeated that the list is to be made step by step from the biological principle ("尊重事实不是说之前让你列出来的一步一步做的清单就不要做。把那个做出来。"). The reviewer's plan of 17 November (`PLAN.md` beside this file) was an inventory ordered by requirement and not a chain of steps; this file is the chain. It is written by the outside reviewer for Mr. Xu, Mr. Zhou and the coordinator. It starts no job: a step is run when its brief has been checked and the coordinator has released it. Mr. Xu's rulings of 18 November that it rests on are quoted in section 1; where the list goes beyond them, it is the reviewer's proposal.

## 1. The principle and the rulings

1. **The principle** (Mr. Xu, 15 to 18 November; memo sections 33.21 and 33.22). A neuron has what arrives over its own sparse connections and its own state, and changes its own connections from these. Nothing delivers to it a quantity computed from the whole network. A signal is relayed between neurons step by step ("1 → 50 → 50² → 50³"), so that a few passes cover the network; a single global backward pass is prohibited. A code that the learner has formed by itself is the learner's own information; what the program does by reading the item is ours.
2. **What the learner is to achieve.** Long-term memory that forms in the course of learning and not by an act of the program (requirement A). At equal computation and accuracy, at least the generalisation of back-propagation, perhaps from fewer samples and with more time; slower at first is acceptable (requirements B1 and B2). Local learning also at the input and the output (requirement C).
3. **The working rule is kept for now** (Mr. Xu, 18 November: "我们暂时保留这个，尊重事实而不是我们自己的想法"). *The sweep* passes the error backward through each linear map by the signs of its weights, with one scale for the map, over the positions of an item. Mr. Xu did not define it (memo, 33.24). Three of its parts are not what a neuron has, and each is tested in a step below: the scale of a whole matrix (step 4), the pass backward over all positions of the item (step 3), and the return over the forward connection with its sign (step 4, second part).
4. **As few numbers fixed by us as possible** (Mr. Xu, 18 November). Where a number must be fixed for now, it is varied on a grid of powers of two (2, 4, 8, 16, 32, 64, 128) and the effect is read, and a later step asks whether the learner can set it itself (step 6).
5. **All three of the reviewer's suggestions of 18 November are adopted** (Mr. Xu): the number of steps of relay together with the number of connections (steps 2 and 3); the sparse input (step 4); memory and a life on the sparse network (steps 7 to 10).

## 2. Terms

- *The dense network* is E26's recurrent network of 256 cells in which every cell reads every other; it is outside the principle under either rule and serves as the reference.
- *The network with limited connections* is E42's: the same 256 cells, each reading N other cells (N = 10 in `F10`), drawn from its neighbourhood on a grid of 16 by 16 or, in the wide form, from all cells. N is an argument of E42's layer (`e42_cells.Layer`).
- *Back-propagation* (rule B) is the exact gradient; *the sweep* (rule S) is defined in section 1.
- *A, R*: the shares right on never-taught pairs (A: add and sub with letters; R: pairs whose answer needs the reduction by the modulus). *Updates to 80*: the first scored row from which the score stays at or above 80.
- *A step of relay* is one crossing of the recurrent connections by the error, from a position of the item to the position before it.
- *A birth* is a network trained on the first family; *a life* is a long sequence of visits to many families (E44's world).

## 3. The steps

Each step gives: the question; why the principle puts it there; what the record holds; the experiment; what each outcome leads to; its state. Costs are the reviewer's estimates from the record's 0.8 to 0.9 CPU-hours for a run of 24,576 updates on the network with limited connections (E42) and are settled by each brief's cost check.

### Step 1. Does a network with sparse recurrent connections learn and generalise under the working rule? — done

- **Record.** E42, three seeds, half the pairs: with 10 connections a cell, updates to 80 on A are 2.36 times the dense network's under the sweep and 1.70 times under back-propagation. E54, one seed: with a quarter of the pairs, 3.65 and 4.29 times; with 15% and a tenth nothing is decided or nothing generalises under either network. E42 also ran N = 5 and N = 20 with changing connections under the sweep.
- **Outcome.** It learns, more slowly. The step is passed for half and a quarter of the pairs.

### Step 2. How many connections does a cell need? — to be briefed now (job E61, with step 3)

- **Why here.** The number of connections is the first number that the principle leaves open and that we fixed (10). Mr. Xu's example has about fifty.
- **Experiment.** The network with limited connections, N = 2, 4, 8, 16, 32, 64, 128, and the dense network as reference; connections fixed for the run and drawn from all cells, so that no geometry is given; both rules; half the pairs; seed 73129 first. The run with N = 10 from the neighbourhood is repeated once as the bridge to E42.
- **Read.** Updates to 80 and the level at the end against N, for each rule; the sweep against back-propagation at each N.
- **Outcomes.** If the curve is flat from a small N on, the later steps take that N. If it falls steadily to the dense network, the slowness of step 1 is a matter of the number of connections, and the cost of locality is read from the curve. If the sweep's lag behind back-propagation changes with N, the working rule's shortfall depends on sparsity, which bears on step 4.

### Step 3. Over how many steps must the error be relayed? — to be briefed now (job E61, with step 2)

- **Why here.** The principle allows relay and prohibits one global pass. The working rule passes the error back over every position of the item. How many steps are needed has never been measured (the old plan's item 20).
- **Experiment.** In the same runs' network, the error is cut after k positions, k = 1, 2, 4, 8, 16 and the whole item, at N = 8 and N = 32 first (and at the N that step 2 picks). k = 1 is the form that exists in the code (credit cut at every position).
- **Read.** Updates to 80 and the end level against k, for each rule and N; beside it the smallest k with N to the power k at or above 256, at which relay can first reach every cell.
- **Outcomes.** If a small k suffices, the rule can be made local in time at little cost, and the later steps take that k. If only the whole item suffices, the working rule depends on the long pass, and that dependence is stated as a limit of every later result.

### Step 4. Does it learn when the input, too, is sparse, and with quantities a unit has? — running (E57), then a brief

- **Record.** With 16 inputs a cell the sweep does not learn (E49); with a fixed code it learns only when each cell has an input for every byte, 2.68 times slower than back-propagation (E50). E57, running, takes the sweep's scale for each receiving unit instead of for a whole matrix; E53, running, concerns the network with an interface at a boundary.
- **First part.** Read E57. If the scale for each unit lets the sweep learn the sparse input, that form of the rule is taken from here on.
- **Second part, if it does not.** The return of the error: by the signs of the forward weights against feedback that is learned (E31, three seeds, not replicated) in the network with a sparse input.
- **Outcome.** A form of the rule in which a sparse input is learned, or the statement that no form on file learns it.

### Step 5. The candidate network, assembled and measured against the reference — after steps 2 to 4

- **Experiment.** Sparse recurrent connections (N from step 2), relay over k steps (step 3), a sparse input and the rule's form from step 4; births with half and a quarter of the pairs on three seeds; beside it the dense network under back-propagation.
- **Read.** Requirement B in its plain form: level reached, updates and CPU time to reach it, pairs needed.
- **Outcome.** The size of the gap between the local learner and the reference at a birth, on three seeds. Everything after this step is run on this network.

### Step 6. Can each cell set its own number of connections? — after step 2

- **Why.** Mr. Xu, 18 November: the fewer numbers we fix, the better; the number of connections should become part of learning.
- **Record.** E42's layer replaces a weak connection by a new one, so that the number stays fixed; a new connection starts at zero and is judged against the cell's established connections once it is 512 updates old, which the reviewer found defective: a changing connection is to be judged against its own age, or kept for at least 2,048 updates (memo, section 22.12; old plan, item 22).
- **Experiment.** Dropping and adding are separated. A cell gives up a connection that stays weak against its own other connections after a minimum age; it adds one at a slow fixed rate from the cells it does not read. The number is then an outcome. Start from 2 connections a cell and from 128.
- **Read.** Whether the two starts arrive at a similar number; that number beside the curve of step 2; the score beside the best fixed N.
- **Outcomes.** If they meet, the number of connections is no longer ours. If not, N stays a declared number, taken from step 2.

### Step 7. Where is an experience written in the sparse network? — after step 5, read-only

- **Experiment.** On the saved weights of step 5: which cells and connections an item and a family use and change; how many cells are not in use (the method of the old plan's item 2, which found the cells away from the boundary unused in E47's network).
- **Outcome.** Whether the sparse network leaves room that a new task could take, before any experiment on memory is designed.

### Step 8. A second task, with no act of the program — after step 7

- **Experiment.** E51's protocol on the candidate network: a new task on other bytes and one on the first family's bytes; arms without rehearsal and with rehearsal of a half and a quarter; the first family's score and answers at every row.
- **Read.** What is kept in the sparse network beside the dense one (E51, E55: kept only by the program or by rehearsal).
- **Outcome.** Whether sparse connections alone change what is kept; this decides whether step 9 or a rule that prefers unused connections comes next.

### Step 9. A new family in the learner's own code — after step 8

- **Experiment.** The outside memo's proposal 5.2: births on 1, 4 and 16 families taught together; then a new family learned by the rows of its own code bytes alone, by a small added term, or freely.
- **Outcome.** How small a change that belongs to the new family alone suffices; by Mr. Xu's ruling such a code is the learner's own.

### Step 10. A life on the sparse network — after steps 5 and 8

- **Experiment.** E44's world with the candidate network, with and without a store, beside the dense network under back-propagation with a store of equal memory; the probe of new families at ages 64, 256 and 1,024 visits for every arm. The runner of the lives takes the dense network only and must be extended.
- **Read.** Requirements A and B1 where they are meant: what is kept at a return, the level over the life, and whether the local learner, slower at first, comes level or ahead.

### Step 11. Seeds and sealed confirmations — after each step that carries a conclusion

Two more seeds for steps 2, 3 and 5 where a reading rests on one; the confirmations on tasks sealed in advance, which the coordinator holds.

## 4. The reference line (the dense network): running, and no new job

E60 (the sweep's life from back-propagation's birth), E55 (the first family kept without a line of the program that reads the item), E52's other two seeds, and the verified long lives. They are read as statements about the rule and about the conventional learner. The old plan's items 14, 23 (first half), 24, 25 and 29 belong to this line and are not started unless the coordinator asks.

## 5. Where the old plan's items went

| Old item | Now |
|---|---|
| 20 (the error over a limited number of positions) | step 3 |
| 22 (connections that change during learning) | step 6 |
| 12 (E56: a second task in a local network) | step 8 |
| 16 (two more seeds) and 17 (sealed confirmations) | step 11 |
| 9 (E53), 18 (E57) | step 4, running |
| 13 (cells that learn at different speeds), 21 (facts recalled by a cue) | after step 8, if it calls for them |
| 14, 23, 24, 25, 29 | the reference line (section 4) |

## 6. History

Revision 1 (18 November 2026): written at Mr. Xu's instruction of that day.
