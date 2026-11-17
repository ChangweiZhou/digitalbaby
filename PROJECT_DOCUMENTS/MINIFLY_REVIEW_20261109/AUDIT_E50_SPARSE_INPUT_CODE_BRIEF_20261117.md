# Audit of the E50 brief and its specification file at f8f62df6, with the results of E49 and of E47's trial (17 November 2026)

From the outside reviewer. Read-only: nothing was trained, torch is not installed in the reviewer's container, and no code from another branch was executed. The brief is `OPUS_TASK_E50_AN_INPUT_CODE_IN_WHICH_A_VALUE_STANDS_FOR_ONE_BYTE_20261116.md`, revision 1. No later revision exists: on the default branch at 7cabb13a, `git log` on the brief, the specification file and its output lists the commit f8f62df6 alone. The specification file is `PROJECT_DOCUMENTS/COORD_CHECKS_20261116/e50_sparse_code_spec.py`, with the coordinator's output `OUTPUT_E50_SPEC_MAC.txt` beside it; the coordinator's account is review section 9.155.

A first version of this audit was begun on 16 November and stopped by a restart of the container; its files were committed as work in progress at e50a79c5. This document replaces it. Every figure of that version was computed again, and the brief is now compared with two results that arrived after it was written: E49, complete at a36de661 on `claude/peaceful-noether-vcvrmf`, and the trial of E47, complete at 578a877b on `claude/e47-worker-stage-0-3d0ya6`.

## Verdict: the job should run, with its settings changed

**The specification file does what the brief says.** A second implementation, written from the brief's text, reproduces the digest of the code, every count of the coordinator's output, the value of a turned-on entry (3.2387) and, on the starting network of seed 73129, the three first-batch losses of the coordinator's output to the three decimals printed (5.597, 5.556 and 5.609).

**The settings of revision 1 no longer fit the record.** The settings `C64` and `C16` use E49's two masks, on the same lists. E49 has since run those masks on the learned code, in which every cell can identify every byte, and the network did not learn the first family within the dose under either rule (section 2). A failure of `C64` or `C16` would therefore say nothing about what a cell knows, which is the question of this job. Three results of E49 decide the changes.

- The limit on the readout alone (`out8`) prevents learning within the dose under both rules. It is not what this job is about, and it should leave the job.
- Under backpropagation the network learns with the input alone limited (`in16`), at 2.16 times the updates of the dense interface. A setting that costs as much against the reference of this job would be recorded within the brief's dose only if that reference learned almost as fast as E42's `F10`. The limited settings need a longer dose.
- Under backpropagation every limited setting of E49 learns max and min, which need both operands at the answer; what it does not learn within the dose is add and sub. The score R, which holds add and sub only, cannot say whether the operands arrive. The four operations must be read apart.

**The job as it should run.** Stage 1 is the reference of revision 1, `C-dense`, under both rules. Stage 2 is the brief's conditional setting `C16-in` (16 values a cell, the readout dense), under each rule under which the reference learns, with twice the dose. Stage 3 is a control, defined in this audit, in which every cell reads one value of every used byte; it runs only if `C16-in` does not learn, and it says whether the cause is the number of bytes a cell has input for. The job is then two to six runs and at most 10 CPU-hours.

**Nine changes are required** (section 9, as text to apply). None alters the code, the forward or the reference; the control adds one draw to the specification file. If they are applied as written, no further audit is needed; the reviewer would check the digest of the control's draw only.

**Terms.**

- The *code* of a byte is the 512 values that feed the input map and `l2`. In E42 and E49 it is `tanh(l1(byte table))`, learned. In E50 it is fixed: a *used* byte turns on 11 values of its own, each equal to one number (the *value of a turned-on entry*, 3.2387 for seed 73129), and every other value is zero.
- The *used bytes* are the 46 bytes that the first family puts into the network: the ten digits, the operation signs `+ - > <`, `=`, the 26 capital letters, the task codes `p q v w`, and the start symbol (257). The *needed bytes* are the 20 that the task must tell apart, that is, all but the letters, which stand between the operands and carry no information.
- The *input map* is `rnn.weight_ih_l0` (768 rows, three gates for each of the 256 cells, and 512 columns). `lm` (512 rows, 256 columns) carries the cells' states to the 512 *readout units*. *K_IN* is the number of values of the code that a cell reads, and *K_OUT* the number of cells that a readout unit reads. A cell *has input for* a byte if it reads at least one of the byte's 11 values.
- `F10` is E42's network: 256 cells, each reading 10 fixed cells of its neighbourhood, with the learned code and with K_IN = 512 and K_OUT = 256. E49's settings on the learned code are `I64` (K_IN 64, K_OUT 32), `I16` (16, 8), `in16` (16, 256) and `out8` (512, 8). E50's settings in revision 1 are `C-dense` (512, 256), `C64` (64, 32), `C16` (16, 8) and the conditional `C16-in` (16, 256).
- *A* is the share of right answers on untaught pairs of add and sub at 12 letters (300 items each); the same panel holds 300 items of max and of min, which the runners count and do not include in A. *R* is the share right on 1,000 untaught pairs of each of add and sub, without letters, whose answer requires the reduction by the modulus. The *length test* is 300 untaught pairs of add at 12 and at 40 letters. A threshold is *reached* at the first of three consecutive scored rows at or above it; rows are scored every 256 updates, so within a *dose* of 24,576 updates the last update at which a threshold can be recorded is 24,064.
- The *sweep* is the sign rule (the error is carried backward through a linear map by the signs of its weights times one scale); *backpropagation* is the exact gradient.

**Shell names** used below: `D=origin/codex/minifly-consolidated-20261023`; `K=PROJECT_DOCUMENTS/COORD_CHECKS_20261116`; `M=MOD97_LOCAL_RULE_20261106`; `W=ed4431f2` (E42's worker: `F10`'s records); `S=8b48f76b` (E47's worker: a starting network of seed 73129); `E=a36de661` (E49's worker); `T=578a877b` (E47's trial); `C="PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/e50_brief_checks.py"`, run from the repository root as `PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 python3 -I $C <part>`. The reader is `e50_code_reader.py` in the same folder; it imports the E42 and E49 readers. Section 12 gives the time each part takes.

## 1. The specification file against the brief

**The reader** (`e50_code_reader.py`) is built from the brief's text alone. The specification file looks the code up in a table of 258 × 512 numbers and multiplies it by masked matrices. The reader never forms the 512 values: for every used byte it adds, for each cell, the cell's weights on those of the byte's values that the cell reads, and uses the sum as that cell's input for the byte. The two forms share no structure.

**Reproduced** (`python3 -I $C A`, `B`, `D`).

| | The coordinator's output | The reader |
|---|---|---|
| Digest of the code, seed 73129 | `12111044f0f8bc70` | `12111044f0f8bc70` |
| Used symbols; values each; values owned; a value with two owners | 46; 11 to 11; 506 of 512; none | the same |
| `C64`: bytes a cell has input for, of the 46; of the 20; of the ten digits | 31 to 41 (median 36); 10 to 19 (16); 4 to 10 (8) | the same |
| `C16`: the same three | 11 to 16 (14); 1 to 10 (6); 0 to 7 (3) | the same |
| Cells a needed byte reaches, `C64`; `C16`; cells with no input | 187 to 211; 67 to 87; 0 | the same |
| Value of a turned-on entry | 3.2387 | 3.2387 (mean squared length 115.381) |
| First-batch loss, `C-dense`; `C64`; `C16` | 5.597; 5.556; 5.609 | 5.597; 5.556; 5.609 |

The digests for the other two seeds are `53fb9ffe8aae97fb` (93683) and `f822e27e800c23a6` (116089).

The last two rows need a starting network of seed 73129, which the reviewer cannot construct without torch. Two are on file and are equal tensor by tensor: the start of E47's arm of 256 cells (`git show $S:$M/exp45/e47_qual/check2_runs/dry/e47_16_one_S_dry2048_73129_start.pt`) and the start of `F10` that E49's worker committed. From it the value of a turned-on entry is 3.2387, and the reader's forward, on the first batch of the training stream rebuilt from `exp11_cell.stream`, gives the three losses. This checks the whole chain (the code, the two masks, the scaling of the kept weights, the forward) against the coordinator's forward, to the precision printed. The reader's two forms of the forward, by look-up and by table and masked matrices, differ by at most 3.3e-07 in a logit.

**The symbols are the right ones.** In the first 1,024 batches of seed 73129's stream the input holds 47 symbols: the 46, each of which occurs, and the end symbol, which never stands at or before a scored position (`python3 -I $C A`). Items of a batch are padded with the start symbol, which now has a code of its own (372 positions in the first batch); nothing is scored there or after it.

**The text leaves four things to the specification file** (required change 5).

- *The digest's formula is not in the text.* It is the first 16 hexadecimal digits of the SHA-256 of `json.dumps(owner)`, where `owner` lists for each of the 512 values the byte that turns it on, or -1. Six other readings of the text give six other digests (`python3 -I $C A`).
- *The 46 symbols are named by kind and not listed.* "Four operation symbols" and "four task codes" are `+ - > <` and `p q v w`.
- *The value of a turned-on entry.* The mean is taken over the 46 used bytes with the start symbol among them, whose present code is short (squared length 19.6, against a mean of 117.5 for the other 45). With it the value is 3.2387; without it, 3.268.
- *The masks are E49's own lists.* E49's strings do not name the job, so a setting of E50 with K_IN = 16 reads exactly the input values that E49's `I16` and `in16` read. This is useful, since it makes E49's runs the same lists on the learned code, and the text should say it.

**Three properties of the fixed code that the text does not state** (`python3 -I $C A`, `D`). Every used byte has the same squared length, 115.4, where the present code's lengths run from 19.6 to 204.4. A turned-on value is 3.2387, where a value of the present code is at most 1 in size. And a cell's input for a byte rests on few weights: 11 under the dense interface, mostly one when a cell reads 16 values. The optimizer moves a weight by about its rate of 0.001 in an update, so one update can move such an input by 9.2% of its starting size in `C-dense` and by 0.49% in a cell of `C16-in` that reads one value of the byte. This matters in section 3.

## 2. The record of 17 November: E49 and the trial of E47

### 2.1 E49, recomputed from the receipts and from the weights

E49 ran E50's masks on the learned code: seed 73129, 24,576 updates, eight runs. The table is recomputed from the receipts (`python3 -I $C H`) and agrees with the summary table of the worker's report in every row. Levels are means over the last three scored rows; the rise is the level of R against the three rows ending 2,048 updates earlier.

| Setting (K_IN, K_OUT) | Rule | Updates to 80 on A | Updates to 80 on R | Levels of A, R, 40 letters | Rise of R |
|---|---|---|---|---|---|
| `F10` (512, 256) | backpropagation | 8,704 | 9,472 | 93.2, 94.1, 92.6 | |
| `I64` (64, 32) | backpropagation | not reached | not reached | 74.0, 65.2, 61.0 | +7.9 |
| `I16` (16, 8) | backpropagation | not reached | not reached | 23.8, 16.9, 16.7 | +3.4 |
| `in16` (16, 256) | backpropagation | 16,640 | 20,480 | 85.2, 82.5, 66.1 | +0.7 |
| `out8` (512, 8) | backpropagation | not reached | not reached | 73.8, 70.8, 68.0 | +8.6 |
| `F10` (512, 256) | sweep | 13,312 | 15,104 | 91.2, 89.3, 89.1 | |
| `I64` | sweep | not reached | not reached | 1.3, 0.9, 1.3 | −0.3 |
| `I16` | sweep | not reached | not reached | 1.2, 0.9, 1.1 | 0.0 |
| `in16` | sweep | not reached | not reached | 1.9, 1.4, 1.2 | −0.4 |
| `out8` | sweep | not reached | not reached | 10.6, 7.8, 7.9 | +2.4 |

Under the sweep no setting with a limited input passes 4.2 on any panel at any scored row.

**The runs are confirmed from their final weights** (`python3 -I $C I`). For each of the eight runs the three masks in the weight file are the reader's own draws, every weight outside them is exactly zero, and a decoder that shares no structure with the runner's forward gives every count of the receipt's last row: A for the four operations, R for add and sub, and the length test at 12 and 40 letters. This also completes recommendation b of the E49 audit, which the worker's report lists as open.

### 2.2 The four operations apart

The panel behind A holds 300 untaught pairs of each of the four operations at 12 letters. The share right at the last row, and in brackets the update at which 80 is reached (`python3 -I $C H`):

| Setting | Rule | max | min | add | sub |
|---|---|---|---|---|---|
| `F10` | backpropagation | 98 (2,048) | 98 (2,048) | 91 (8,704) | 92 (9,472) |
| `I64` | backpropagation | 97 (3,840) | 95 (3,584) | 81 (not reached) | 72 (not reached) |
| `I16` | backpropagation | 98 (7,168) | 94 (7,424) | 24 (not reached) | 28 (not reached) |
| `in16` | backpropagation | 97 (5,120) | 99 (5,376) | 87 (15,616) | 88 (18,432) |
| `out8` | backpropagation | 97 (2,560) | 95 (2,560) | 79 (not reached) | 79 (not reached) |
| `F10` | sweep | 99 (3,072) | 97 (2,816) | 92 (11,776) | 93 (15,104) |
| `I64` | sweep | 34 | 25 | 2 | 1 |
| `I16` | sweep | 30 | 33 | 1 | 1 |
| `in16` | sweep | 39 | 43 | 3 | 2 |
| `out8` | sweep | 83 (8,960) | 86 (11,008) | 11 | 12 |

The answer to max or min is one of the two operands, so a network that gives it at 12 letters holds both operands at the answer. Under backpropagation every limited setting does so, the smallest included; what the limited settings do not learn within the dose is the sum and the difference with their reduction. Under the sweep with a limited input the network does not come to give even these answers. The worker's report does not show this, because A and R hold add and sub only.

### 2.3 What E49's networks ended with

The E49 audit asked (recommendation g) for its measurement of the code to be repeated on the codes that the limited networks learn. It is done here on the final weights (`python3 -I $C J`).

| Network | Mean distance between the codes of two needed bytes | Mean size of a kept weight of the input map | A cell's input for a needed byte, root mean square |
|---|---|---|---|
| The start of seed 73129 | 14.4 | 0.031 | 0.39 |
| `F10`, backpropagation; sweep | 7.0; 6.1 | 0.079; 0.083 | 4.5; 4.0 |
| `out8` (dense input), backpropagation; sweep | 6.7; 5.7 | 0.073; 0.084 | 4.5; 6.0 |
| `I64`, `I16`, `in16`, backpropagation | 14.1, 16.2, 14.3 | 0.108, 0.173, 0.198 | 1.95, 1.13, 1.34 |
| `I64`, `I16`, `in16`, sweep | 4.5, 5.7, 3.2 | 0.086, 0.099, 0.102 | 0.77, 0.47, 0.22 |

- **The premise of the brief holds on the trained networks:** the values a cell reads still distinguish the bytes. On no cell of any network do two of the 46 used bytes coincide. Under backpropagation the closest two needed bytes on the values a cell reads are, on the worst cell, at 0.12 to 0.24 of the mean distance between two of them. Under the sweep with a limited input one pair has nearly merged in the code as a whole: `=` and the start symbol are at 0.06 to 0.08 of the mean distance on all 512 values. With the fitted linear rule of the E49 audit (`python3 -I $C J linear`), one rule on a cell's values puts all 46 bytes in their class on all 256 cells in the three limited runs under backpropagation; under the sweep the rule, within its 8,000 steps, leaves one or two of the 46 bytes outside their class on every cell of `I64` and of `in16` and on one cell of `I16`, which is the effect of that pair.
- **With a dense input the code shrinks during training under both rules,** to about half the starting distance between two bytes, and the 512 weights a cell has on it make up for that: a cell's input ends ten or more times larger than at the start.
- **With a limited input the two rules end differently.** Under backpropagation the code keeps its starting size, and a cell's input ends at 1.1 to 1.9. Under the sweep the code shrinks as it does with a dense input, the kept weights do not grow, and a cell's input ends at 0.2 to 0.8, between half and twice its starting size.

These are descriptions of the end of each run. They do not show why the sweep does not learn with a limited input; no run isolates a cause. They do bear on E50 in one way. In E50 the code is fixed, with every two used bytes 15.2 apart, the distance that the limited networks keep under backpropagation and lose under the sweep. A limited setting of E50 under the sweep therefore tests one explanation of E49's result, which section 7 states with the result that would refute it.

### 2.4 The trial of E47

E47's trial has 1,024 cells, of which 256 read the byte's whole learned code and are read by the readout; the other 768 have no input and are not read. On seed 73129 all four runs reach 80 on R (`python3 -I $C H`): at 11,008 and 11,264 updates under backpropagation and at 17,664 and 19,200 under the sweep, which is 1.16 to 1.27 times `F10`'s figure. With the 768 cells held at zero, R at the last row falls by 5 to 16 points; with them held at their mean, by 2 to 6. So a network in which most cells have no input at all learns the first family in about the updates of `F10`, and those cells take some part. E47 does not answer E50's question: its 256 cells that read the input read all of it. E50 remains the only job in which no cell can identify more than a few bytes.

### 2.5 What follows for E50

1. **`C64` and `C16` cannot be read.** They are `I64` and `I16` with the fixed code. On the learned code, where a cell lacks nothing, these masks end at 65.2 and 16.9 on R under backpropagation and at 0.9 under the sweep. The fixed code takes information away from a cell and adds none. If `C64` and `C16` fail, the masks are sufficient to explain it; the job would have spent four runs to learn nothing about what a cell knows.
2. **The limit on the readout should leave the job.** `out8` alone does not reach 80 on R under either rule, while `in16` alone does under backpropagation. The brief itself calls the input "the side this job is about" and holds `C16-in` in reserve for that reason. `C16-in` should be the limited setting of the job.
3. **The comparison must be made beside E49's.** `C16-in` reads the input values that `in16` reads. Against `F10`, `in16` under backpropagation needs 2.16 times the updates to 80 on R and 1.91 times to 80 on A. A cost of `C16-in` against `C-dense` of about that size is the cost of the mask, which E49 has already measured; only a cost beyond it can be laid to what a cell knows, and section 3 shows that even then a control is needed.
4. **The dose of the limited settings is too short.** Within 24,064 recordable updates a setting that needs 2.16 times the reference's updates is recorded only if the reference reaches 80 on R by update 11,130, which is 1.17 times `F10`'s 9,472 (`python3 -I $C H`). If the fixed code costs anything at all, `C16-in` would end as "does not within the dose", and the comparison with `in16` would be undecided. With 49,152 updates the reference may need up to 22,496.
5. **The predictions of the brief on the limited settings were made before these results** and give `C64` and `C16` chances of 65%, 35% and 15% where E49's runs of the same masks ended at 65.2, 16.9 and 0.9 on R. The coordinator's five predictions that a limited setting of E49 would reach 80 on R, or learn in about `F10`'s updates, were all wrong in the same direction.

## 3. Do the settings isolate the one thing varied?

**`C-dense` against `F10`: the right reference, with four differences.** Changing the code is itself a change, so the limited settings are read against `C-dense` and `C-dense` against `F10`; the brief says both. `C-dense` differs from `F10` in four things: the code is not learned, so nothing below the input map adapts; a value stands for one byte; every byte has the same length; and one update can move a cell's input for a byte by 0.036, where in `F10` it can move it by between 0.011 and 0.205 (`python3 -I $C D`). If `C-dense` costs against `F10`, the job can say that this fixed code costs, and not which of the four is responsible.

**The limited settings of revision 1 against `C-dense`: three things vary together.**

- *The readout's mask.* `C64` and `C16` limit the readout as well as the input. Section 2 shows that under backpropagation this limit has the larger cost, and it is not the subject of the job.
- *The number of weights behind a cell's input for a byte.* At K_IN = 16 a cell reads no value of a needed byte in 70% of the pairs of a cell and a needed byte, one value in 25%, two in 4% and three or more in 0.3% (`python3 -I $C B`). Where it reads one, its input for that byte rests on one weight, which starts 5.66 times larger than a dense weight and moves at the same rate.
- *The number of bytes a cell has input for,* which is the subject: a median of 6 of the 20 needed bytes and 3 of the ten digits.

`C16-in` removes the first. It does not separate the second from the third. If `C16-in` learns in about the updates that `in16` needed against `F10`, nothing needs separating: the statement that the network learns although its cells have input for few bytes then stands. If it does not, the result has two possible causes, and the brief's forms would allow the report to name neither.

**A control that separates them** (required change 1, where it is named `C46-one`; the name is the coordinator's to change; `python3 -I $C K`). In it every cell reads one value of every used byte, 46 values a cell, with the readout dense. Such a cell has input for every byte, each through one weight. On the starting network its input has the reference's size (root mean square 0.391, against 0.389 in `C-dense` and 0.382 in `C16-in`), and one update moves an input by 0.83% of its size, the same order as the 0.49% of `C16-in` and far from the 9.2% of `C-dense`. If this control learns and `C16-in` does not, the number of bytes a cell has input for is what costs. If neither learns, the cost is that of an input resting on one weight for a byte, and the job says nothing about what a cell knows. The control is needed only when `C16-in` does not learn, and it is defined now so that it needs no further release.

**The comparison with E49's `in16`** is the second control, and it costs nothing: the same lists on a code in which every cell can identify every byte.

## 4. What a cell can know, and the paragraph "Time to travel"

**Direct input** (`python3 -I $C B`; three seeds). At K_IN = 16 a cell has input for 10 or 11 to 16 of the 46 used bytes (median 14), for 1 or 2 to 10 or 12 of the 20 needed bytes (median 6) and for 0 to 6, 7 or 8 of the ten digits (median 3), depending on the seed. No needed byte is left to few cells: the least-read one has 67, 67 and 64 cells on the three seeds. More than half of what a cell reads belongs to letters: 286 of the 512 values are the letters', and of the 16 values a cell reads about 9 are. When the present byte is one of which a cell reads no value, the cell's input is its bias alone, the same for every such byte. The code is fixed, so training cannot change this. The job therefore does test the user's statement in the sense that E49 could not reach.

**The limit that remains: a cell learns of the other bytes one position later** (`python3 -I $C C`).

| Seed, K_IN = 16 | Needed bytes within one connection of a cell (least, median, most) | Cells a digit has reached at its own position | One position later | Two later |
|---|---|---|---|---|
| 73129 | 18, 20, 20 | 67 to 82 | 241 to 255 | 256 |
| 93683 | 18, 20, 20 | 73 to 88 | 248 to 254 | 256 |
| 116089 | 18, 20, 20 | 73 to 88 | 249 to 254 | 256 |

A cell's ignorance of a byte lasts one position. One position after a byte, the median cell is within one connection of a reader of every needed byte. What it receives is not the byte's code but the states of its 10 neighbours, one number each, each of which also carries everything else that neighbour holds.

**Two statements of the paragraph "Time to travel" do not hold** (required change 2).

- *"Any cell is within 3 connections of any other."* For seed 73129 the state of one cell reaches another within 3 connections for 83.2% of the 65,280 ordered pairs of cells and within 4 for all but 28 pairs; the farthest pairs are 5 apart, and no cell is reached by every other cell within 3. The other seeds give 83.8% and 83.2%. The conclusion the brief draws, that a byte reaches every cell in 3 steps, is true and understates the case, since a byte starts in at least 64 cells and has reached every cell after 2 positions.
- *"A cell without input for that digit cannot use it for that byte, which only the cells that read it can supply."* The second operand's last digit stands one position before `=`, where the answer's first byte is read. A cell's state at `=` is computed from its sources' states at the digit's position, so every cell with a reader among its sources can use the digit at `=`. The forward confirms the count: on the starting network of seed 73129, with the last digit changed in 3,600 items, the cells whose state differs are 113 to 137 at the digit's own position, 255 or 256 at `=`, and 256 one position later; in each of the 90 cases the set of cells is exactly the set within 0, 1 and 2 connections of a reader.

No failure on the answer's first byte in particular follows from the connections. The record now agrees: in E49's `I16` under backpropagation, where a cell reads 16 values and a readout unit 8 cells, max and min at 12 letters are right in 98% and 94% of items, so both operands, the second one's last digit included, are at the answer. Scoring the two answer bytes apart remains a useful description; the stated reason for expecting the first to fail should go, and prediction P50.7 with it.

## 5. The readings, the dose and the programme's rules

**The working rules.** Thresholds are in the three-row version. The job is one seed, read before any further seed is released. The bytes of every answer on the length-test panel are recorded at the last row. Starting and final weights, the code table and the masks are committed and listed from `git ls-tree`. These four rules are met. The rule that a brief is compared with the record is met for the record of 16 November and must now take in E49 and E47's trial (required change 2). The rule that the reference is read before the settings that depend on it is not met: revision 1 launches six runs together and has no rule for the case in which `C-dense` does not learn, which the coordinator's own prediction P50.3 puts at 45% under the sweep (required change 1).

**Against `F10`, every case of `C-dense` can be computed** (`python3 -I $C F`). `F10` reaches 80 on A at 13,312 (sweep) and 8,704 (backpropagation) updates and on R at 15,104 and 9,472. `C-dense` is *fewer* below two thirds of these figures and *more* above one and a half times them; if it does not reach 80 it is *more* in all four cases, since 24,064 divided by `F10`'s figure is at least 1.59.

**Against `C-dense`, the dose decides what can be read** (required change 1). The brief quotes the factors 1.59 and 2.54, which are relative to `F10`. The factor that applies to a limited setting is 24,064 divided by the updates of `C-dense`, which is not known until `C-dense` has run. Within the brief's dose, a limited setting can be read as *more* only if `C-dense` reaches the threshold by update 16,042, and the factor of 2.16 that E49 measured for `in16` can be recorded only if `C-dense` reaches it by update 11,130. With 49,152 updates for the limited settings, *more* can always be decided when `C-dense` has learned, and if `C-dense` needs what `F10` needed a factor of up to 5.1 can be recorded under backpropagation and of up to 3.2 under the sweep.

**The level and "does not within the dose"** are E49's forms and can be computed. For a run of 49,152 updates each reading should be given twice, within the first 24,576 updates and at the end, so that the first is comparable with `C-dense`, with `F10` and with E49.

**What is recorded is sufficient, with two additions** (required change 3).

- *The four operations apart* (section 2.2). The runner already counts them.
- *The diagnostic "by what a cell knows"* is the mean size, at the end, of the recurrent weights into cells that have input for 0 to 2, for 3 or 4 and for 5 or more digits (95, 131 and 30 cells for seed 73129; `python3 -I $C B`). It has no baseline: the figure at the start, and the same three groups of cells in `C-dense`, where the grouping means nothing, are needed to say whether a difference is more than chance. And the size of weights is not use.

**A test of use, which the reviewer computes from the committed files** (`e50_code_reader.patch_test`; mechanics in `python3 -I $C G`). Two items are run that differ only in the second operand's last digit. The first is run again with, at the position of `=`, the states of chosen cells replaced by their states in the second run. If the cells that read neither digit are the ones replaced and the answer's first byte changes to the second run's, those cells carry the digit, and they can have received it only through their neighbours. On the starting network at K_IN = 16, replacing every cell gives the second run's logits exactly and replacing none gives the first run's; before any learning, replacing only the cells that read neither digit moves a logit by 0.18 of the difference between the two runs, against 0.99 for the cells that read one of the two digits. The test needs nothing from the runner beyond the files the brief already has committed.

## 6. Computation per update

**The brief gives no figure of multiply-adds a position for any setting,** and E49's function `counts` would add 4,096 for `l1`, which this job does not compute (required change 6; `python3 -I $C E`, `K`). Counted as E49 counts (every weight of a map inside its mask, the cells' 8,448, `l2`'s 262,144 and `l3`'s 131,584) and without `l1`:

| Setting | Weights that take part | Of `C-dense` | Products with a turned-on value | Of `C-dense` |
|---|---|---|---|---|
| `C-dense` | 926,464 | 1 | 285,184 | 1 |
| `C16-in` | 545,536 | 0.589 | about 277,000 | 0.971 |
| One value of every used byte | 568,576 | 0.614 | 277,504 | 0.973 |
| `C64`, `C16` of revision 1 | 467,712; 418,560 | 0.505; 0.452 | about 163,104; 150,024 | 0.572; 0.526 |

`C-dense` is 0.996 of `F10`'s 930,560 by the first count.

The second count is the one that describes this network. With 11 of the 512 values turned on, the products of the input map with a turned-on value are 8,448 a position under the dense interface, that is 33 for a cell (three gates and 11 values), where the learned code costs a cell 1,536. `l2` has 5,632 such products where it had 262,144. So, for an implementation that skips the products with zero, a code in which a value stands for one byte removes by itself the cost that section 26 of the reviewer's memo attributed to the dense input, and limiting the input afterwards saves 3% of what is left. What remains is `lm` (512 products for a cell, 131,072 in all) and `l3` (131,584), which together are 92% of the products. The job is about what a cell knows and not about computation, and the brief should say which count its multiply-adds are and that a limited input is not expected to save any. By the first count a limited setting can be read as *more* in multiply-adds only above 2.55 times the reference's updates.

## 7. Predictions and explanations

**The predictions.** P50.1, P50.2, P50.3 and P50.8 concern the checks and `C-dense` and can each be scored. P50.4 to P50.7 name `C64` and `C16`; they fall with those settings, and the coordinator should state new ones for `C16-in` under each rule before the runs (required change 8). What the record says of them is in section 2.5, item 5.

**Explanations with the result that would refute them** (working rule 6).

- *The brief's explanation for a failure on the answer's first byte* (that a cell without input for the last digit cannot use it there) is refuted by the connections and by E49 (section 4). It should not stand in the brief.
- *An explanation that this job can test under the sweep.* In E49's three sweep runs with a limited input the code ended small and a cell's input for a needed byte ended at 0.2 to 0.8, where the same settings under backpropagation kept the code at its starting size and ended at 1.1 to 1.9 (section 2.3). One explanation of the sweep's result is that it does not keep a learned code at the size that a cell reading few values requires. In E50 the code cannot shrink. *The explanation is refuted as a sufficient one if `C16-in` under the sweep does not reach 80 on R within its dose while `C-dense` under the sweep does:* the sweep then does not learn with a limited input even when the code keeps its size. If `C16-in` under the sweep learns, the explanation has passed one test and is not thereby shown, since the fixed code differs from the learned one in other respects (section 3). This is the reason to run `C16-in` under the sweep although `in16` under the sweep learned nothing.

## 8. The checks, the cost and "What it cannot show"

**Checks.** Check 1 can pass as written; its statements that need torch (the byte table and `l1` without gradient; gradients zero outside the masks) were read in the code and not run. Check 2, the same SHA-256 of the table at the start and at every scored row, is the right check that the code is fixed. Check 3 refers to E49's checks 3, 4 and 5, and the numbers agree with E49's revision 2. E49's checks 1 and 2, which establish that the runner with both masks full is E42's `F10`, are not in the list; they passed on E49's runner (the worker's stage-0 report, bit-equal under both rules), and this audit confirms the runner's masked forward and its scorer from outside on all eight full runs (section 2.1). E49's check 5 scores E26's births, which have a learned code; it must run through E49's forward, and it then checks the scorer and not E50's forward. Check 4 names `C16` and should name `C16-in`; its tolerance of 1e-6 leaves a factor of three over the difference of 3.3e-07 that the reader's two forms show in float32. Check 5 is sufficient for the reader if the weight files name the code table.

**The update.** E49's runner calls `opt.zero_grad()` and then `loss.backward()` (`git show $E:$M/exp49_interface.py`, function `step`), so the byte table and `l1`, which take no part, have no gradient and are skipped by the optimizer, decay included. They should remain bit-equal to their starting values, and a receipt should say so.

**Cost.** E49's eight runs took 0.90 to 1.02 CPU-hours each, scoring included, against the brief's 0.8. Stage 1 is about 2 CPU-hours, stage 2 about 4, stage 3 at most 4.

**"What it cannot show"** is correct in its five items and lacks five (required change 9): that a cell's ignorance lasts one position; that the two rules differ in three maps here and in six in E49, since the code has no gradient and nothing is carried back through the input map or `l2`; that a cost of `C-dense` against `F10` has four possible sources; that more than half of what a cell reads is letters; and that the readout is not limited in this job.

**"Nothing on file fixes the input code of this learner."** A search of the default branch agrees for this learner. The nearest items are the frozen embedding table of E17 Phase 3, in the two-bank learner during a life (review section 9.66), and the control with identical fixed input codes that section 9.71 defined and did not commission. Their results were not read for this audit.

## 9. The nine required changes, as text to apply

**1. The settings, the stages and the dose.** Replace the section "Settings, seed 73129, each under the sweep and under backpropagation", with its paragraph on `C16-in`, by:

> ## Settings and stages, seed 73129
>
> | Stage | Name | Values a cell reads | Cells a readout unit reads | Run under | Updates |
> |---|---|---|---|---|---|
> | 1 | `C-dense` | 512 | 256 | the sweep and backpropagation | 24,576 |
> | 2 | `C16-in` | 16 (E49's lists for K_IN = 16) | 256 | each rule under which `C-dense` reaches 80 on R within its dose | 49,152 |
> | 3 | `C46-one` | 46, one value of every used byte | 256 | each rule under which `C16-in` does not reach 80 on R within its dose | 49,152 |
>
> **The readout is not limited in this job.** E49 ran the masks of revision 1 on the learned code. With the readout alone limited (`out8`) the network did not reach 80 on R within the dose under either rule; with the input alone limited (`in16`) it reached it under backpropagation at 20,480 updates. The settings `C64` and `C16` of revision 1 are not run.
>
> **Stage 1 is read against E42's `F10` before stage 2 starts.** Under a rule under which `C-dense` does not reach 80 on R within its dose, the report says that the network does not learn the first family from this fixed code within the dose under that rule, gives E49's figures for a run that does not reach the threshold, and makes no statement about a limited input; stages 2 and 3 are not run under that rule, and what follows is my decision.
>
> **`C46-one`** separates the two things that change together between `C-dense` and `C16-in`: the number of bytes a cell has input for, and the number of weights on which a cell's input for a byte rests. With `own[k] = vals[11k : 11k + 11]` as in the code and `rng = Random(f"e50|one|{seed}")`, for cells 0 to 255 in turn and, within a cell, for the 46 symbols in the code's order, the cell reads `own[k][rng.randrange(11)]`. The kept weights of the input map are scaled by √(512 / 46) = 3.336. The digest (the first 16 hexadecimal digits of the SHA-256 of `json.dumps` of the 256 lists, each sorted) is `6788d5475af8c847` for seed 73129 by the reviewer's reader, which also gives a first-batch loss of 5.591 for this setting and 5.569 for `C16-in`. Such a cell has input for every used byte, each through one weight. If `C46-one` reaches 80 on R and `C16-in` does not, the number of bytes a cell has input for is what costs. If neither does, the cost is that of an input resting on one weight for a byte, and nothing is said about what a cell knows.
>
> **Dose.** Stage 1 runs E26's birth, 24,576 updates. Stages 2 and 3 run 49,152 updates, scored every 256, so that a threshold can be recorded up to update 48,640. Every reading of such a run is given twice: within the first 24,576 updates, which is the dose of `C-dense`, of `F10` and of E49, and at the end. The reason is on file: under backpropagation E49's `in16` needed 2.16 times `F10`'s updates to 80 on R, and within 24,064 recordable updates a setting that costs as much against `C-dense` is recorded only if `C-dense` reaches 80 on R by update 11,130.
>
> Two to six runs; about 1 CPU-hour for 24,576 updates (E49's eight runs took 0.90 to 1.02) and 2 for 49,152; at most 10 CPU-hours.

In the table "What a cell is left with", replace the names `C64` and `C16` by "64 values" and "16 values (`C16-in`)", and add the row "`C46-one` (46 values, one of every used byte) | 46 | 20 | 10 | 256".

**2. The comparison with the record.** Replace the bullets "Time to travel" and "Dose" by the five bullets below, and in "What is recorded" delete the sentence "The comparison with the record says the first byte is where a cell without the last digit should fail."

> - **E49: the same lists on the learned code** (seed 73129; the reviewer recomputed every reading from the receipts and every count of the last row from the final weights). Under backpropagation `in16` reaches 80 on R at 20,480 updates, 2.16 times `F10`'s 9,472; `out8` does not reach it (70.8, still rising), nor does `I64` (65.2, still rising) or `I16` (16.9). Under the sweep no setting with a limited input passes 4.2 on any panel at any row, and `out8` ends at 7.8. Under backpropagation max and min at 12 letters reach 80 in every limited setting (at 7,168 and 7,424 updates in `I16`), so both operands are at the answer; what is not learned within the dose is add and sub.
> - **What E49's networks ended with** (the reviewer's measurement on the final weights). With a limited input under backpropagation the code keeps its starting size (the codes of two needed bytes 14.1 to 16.2 apart, 14.4 at the start) and a cell's input for a needed byte ends at 1.1 to 1.9 in root mean square. Under the sweep the code shrinks (3.2 to 5.7 apart) and a cell's input ends at 0.2 to 0.8. In this job the code is fixed, with every two used bytes 15.2 apart.
> - **E47's trial** (1,024 cells, seed 73129): 768 cells have no input and are not read; the first family is learned under both rules in 1.16 to 1.27 times `F10`'s updates to 80 on R, and holding those cells at zero costs 5 to 16 points of R. In E47 the 256 cells that read the byte read its whole learned code. E50 asks what E47 does not: whether the network learns when no cell can identify more than a few bytes.
> - **Time to travel** (the reviewer's figures for seed 73129, `e50_brief_checks.py` part C; the other two seeds alike). A cell reads 10 cells within 3 rows and columns. The state of one cell reaches another within 3 connections for 83% of the ordered pairs of cells and within 5 for all. A byte starts in many cells at once and travels faster: with 16 values a cell a needed byte is read by 67 to 87 cells, one position later it has reached at least 241 of the 256, and two positions later all of them. The second operand's last digit stands one position before `=`, where the answer's first byte is read, so at `=` all but 1 to 15 cells can hold something of that digit. The cells that do not read it hold what the states of their neighbours carried, one number for each connection, and not the digit's code. Whether cells learn to use what arrives in that form is what `C16-in` asks. No failure on the answer's first byte in particular follows from the connections; the two answer bytes are scored apart as a description.
> - **Dose.** `F10` reaches 80 on R at 15,104 updates under the sweep and at 9,472 under backpropagation, so `C-dense`, read against `F10` within 24,064 recordable updates, can be recorded up to 1.59 and 2.54 times `F10`'s figure. The limited settings are read against `C-dense`, and the factor that applies to them is known only when `C-dense` has run; they run twice the dose for that reason.

**3. What is recorded.** Add the first bullet below, extend the bullet "By what a cell knows" by the second, and in that bullet replace "(`C16`)" by "(`C16-in`)".

> - **The four operations apart:** on the panel behind A (untaught pairs at 12 letters, 300 items for each of add, sub, max and min, which the runner already counts), the share right for each operation at every scored row, and the updates to 80 in the three-row version. The answer to max or min is one of the operands, so a setting in which max and min reach 80 and R does not is described as bringing both operands to the answer without learning the sums within the dose.
> - The same figure is given at the start, and for the same three groups of cells (95, 131 and 30 cells for seed 73129) in `C-dense`, where the grouping means nothing. The figure describes the size of weights and is not read as use. The reviewer computes a test of use from the committed files (`e50_code_reader.patch_test`): the states, at `=`, of the cells that read neither of two digits are replaced by their states in a run with the other digit, and the share of items in which the answer's first byte follows them is reported.

**4. How the result is stated.** Replace the first line of the section by: "`C16-in`, and `C46-one` if it runs, against `C-dense` of the same rule; `C-dense` against E42's `F10`; and beside `C16-in` against `C-dense`, E49's `in16` against `F10` under the same rule (2.16 times the updates to 80 on R and 1.91 times to 80 on A under backpropagation; not learned under the sweep)." In "The sentence this job can license", replace "each cell reading K values of it" by "each cell reading 16 values of it and every readout unit reading every cell", and add: "If `C16-in` does not reach 80 on R, the job says why only through `C46-one`. Under backpropagation, a cost of `C16-in` against `C-dense` that is no larger than the cost of `in16` against `F10` is described as the cost of the mask and is not attributed to what a cell knows."

**5. The text of the code** (section "The network, as code").

- In the bullet "The code", after "The 46 symbols that occur as input in the first family", insert: "(the ten digits, `+ - > <`, `=`, the 26 capital letters, `p q v w`, and the start symbol)".
- Add a bullet: "**The digest of the code** is the first 16 hexadecimal digits of the SHA-256 of `json.dumps(owner)` with Python's default separators, where `owner` is the list of 512 entries giving, for each value, the byte that turns it on, or -1 for the six values without an owner. For seed 73129 it is `12111044f0f8bc70`; the reviewer's reader gives `53fb9ffe8aae97fb` for seed 93683 and `f822e27e800c23a6` for seed 116089."
- Replace the bullet "The value of a turned-on entry" by: "**The value of a turned-on entry** is one number for the whole table: the square root of one eleventh of the mean, over the 46 used bytes (the start symbol included, whose present code is the short one), of the squared length of `tanh(l1(byte table))` in the network as constructed for the seed. For seed 73129 the mean squared length is 115.38 and the value 3.2387. The runner computes it before the start and records it. Every used byte then has the squared length 115.4, where the present code's lengths run from 19.6 to 204.4."
- In the bullet "The cells and the two masks are E49's", add: "E49's strings do not name the job, so `C16-in` reads exactly the input values that E49's `I16` and `in16` read."

**6. Multiply-adds.** Add to the table of settings the columns "Weights that take part" (926,464; 545,536; 568,576) and "Products with a turned-on value" (285,184; about 277,000; 277,504), and below it: "The first count is E49's (every weight of a map inside its mask, the cells' 8,448, `l2`'s 262,144 and `l3`'s 131,584) without `l1`'s 4,096, which this job does not compute; E49's function `counts` still adds them and is not used unchanged, and the receipts' count of weights leaves out the byte table and `l1` (6,672 parameters). The second count is of products with a value that is not zero: with 11 of 512 values turned on, the input map has 8,448 of them under the dense interface and `l2` has 5,632. By the second count a limited input saves 3%. The job is about what a cell knows, not about computation; the multiply-adds to a threshold are given by the first count, as in E49, with the second beside them, and by the first count a limited setting is *more* only above 2.55 times the updates of `C-dense`."

**7. Checks.** In check 3, add: "E49's checks 1 and 2 passed on E49's runner at a36de661; E49's check 5 is run through E49's forward, since E26's births have a learned code." In check 4, replace "`C16`" by "`C16-in`" and "1e-6" by "1e-5". In check 5, add: "The weight files hold the code table (258 × 512) and the list of owners under names that the report states, and each receipt says whether the byte table and `l1` are bit-equal at the end to their starting values."

**8. Predictions.** Withdraw P50.4 to P50.7 with the settings they name, and state before the runs: the chance that `C16-in` reaches 80 on R within 49,152 updates, under each rule; the chance that it does so in no more than 1.5 times the factor `in16` needed (backpropagation); and the chance that max and min reach 80 at 12 letters under each rule. Add after the table: "**An explanation under test, with the result that refutes it.** The sweep did not learn with a limited input in E49, and its networks ended with a small code and a small input to the cells. One explanation is that the sweep does not keep a learned code at the size that a cell reading few values requires. Here the code cannot shrink. The explanation is refuted as a sufficient one if `C16-in` under the sweep does not reach 80 on R within its dose while `C-dense` under the sweep does."

**9. "What it cannot show".** Add: "**A cell that stays ignorant of a byte.** A cell's ignorance lasts one position: one position later the byte has reached at least 241 of the 256 cells, through one number for each connection. **A limited readout.** Every readout unit reads every cell; E49's `out8` is the record on that side. **The same difference between the two rules as in E49.** The code has no gradient, so nothing is carried back through the input map or through `l2`; under the sweep the sign rule acts on the recurrent map, on `lm` and on `l3` only. **Why the fixed code costs, if it does.** `C-dense` differs from `F10` in that the code is not learned, in that a value stands for one byte, in that every byte has the same length, and in how far one update can move a cell's input; the job does not separate these. **Letters.** 286 of the 512 values belong to the 26 letters; of the 16 values a cell reads, about 9 are letters' and about 7 belong to the 20 bytes the task must tell apart." In "Done", replace "the six or seven runs" by "the two to six runs".

## 10. Recommendations (none holds the release)

a. **A milder limit, if the coordinator wants two levels:** a setting with 64 values a cell and the dense readout. It has no run of the same lists on the learned code beside it, since E49 ran `I64` with both masks; an `in64` on E49's runner would be needed for the comparison.

b. **In the table "What a cell is left with",** add the figures of section 4 for one position later, so that the table shows both what a cell lacks at a byte's position and how soon it can have it.

c. **P50.7,** if the coordinator keeps a prediction on the two answer bytes, scored on the mean of the last three scored rows; a single row moves by several points between neighbours.

d. **The runs of E49 that were still rising** (`I64` and `out8` under backpropagation) need not be continued for this job. Whether they are continued is a separate decision.

e. **For a later job, not this one.** K_IN counts values, so the number of bytes a cell has input for varies by chance from cell to cell (1 to 10 of the 20 needed bytes). A draw that gives each cell all the values of a stated number of bytes would make that number the setting itself. A single code shared by the 26 letters would leave more than half of a cell's input for the bytes that matter. And a network in which a byte reaches most cells only after several positions, as in E47's geometry, would test ignorance that lasts.

## 11. Run as written, run changed, or wait

**Run changed.** As written, four of the six runs repeat masks that the record shows not to learn on a code that tells every cell more than this one does, and their failure could not be read. Waiting gains nothing: the reference `C-dense` does not depend on E49's result and answers a question that stands by itself, whether this learner learns the first family from a fixed code in which a value stands for one byte. The changed stage 2 is the only limited setting whose counterpart on the learned code learned (`in16` under backpropagation), and under the sweep it tests a stated explanation of E49's result at the cost of one run. The job stays on E49's runner and worker, so it opens no further line of work.

**What the result will and will not say.** If `C16-in` learns under a rule, the sentence of the brief is licensed for that rule and seed: the network learns although each cell has input for 1 to 10 of the 20 bytes it must tell apart. If it does not, `C46-one` says whether the number of bytes is the cause. In neither case does the job say anything about a limited readout, about more cells, or about a learned sparse code.

## 12. The checks behind this audit, and what was not checked

**To repeat the checks,** from the repository root: `PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 python3 -I PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/e50_brief_checks.py <part>`. Measured on the reviewer's container: A 1 second (the code, the digests, the stream, the value of a turned-on entry); B under 1 (what a cell has input for, three seeds); C 57 (travel, from the connections and from the forward); D 3 (the start: the three losses, the sizes); E under 1 (multiply-adds); F under 1 (what can be recorded); G 2 (the test of use, mechanics); H under 1 (E49's and E47's receipts); I 11 (E49's eight runs from their final weights); J 2, or 3 to 7 minutes with the word `linear` added (the codes E49's networks learned); K 2 (`C16-in` and the control at the start, the control's digests).

**Not checked.**

- **The specification file was not executed.** Torch is not installed. Its forward was compared with the reader's through the three first-batch losses of the coordinator's output, to three decimals. Its statements on gradients were read in the code.
- **The starting network** is the file that E47's worker committed for seed 73129, equal tensor by tensor to the start that E49's worker committed. That it is `arch26.new_net(73129)` with E42's start rests on three agreements: its connections are E42's draw, it gives the value 3.2387, and it gives the three losses. The value of a turned-on entry for the other two seeds was not computed.
- **E49's checks 1 and 2** (bit-equality with E42's `F10`) are the worker's statement; the reviewer confirmed the eight full runs from their weights, not those two checks.
- **E47's trial** was read from its receipts; its weights were not opened for this audit.
- **Why the sweep does not learn with a limited input** is not established. Section 2.3 describes the end of each run and nothing more.
- **No runner of E50 exists,** so nothing about one was checked. The reader has not been applied to a trained network of this job; the test of use was checked for its mechanics only.
- **The chances** attached to the predictions were not assessed, and the earlier results on fixed input codes in other learners were not read.
