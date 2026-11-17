# E50: an input code in which a value stands for one byte, so that a cell which reads few values knows few bytes (16 November 2026; revision 2, entered by the worker from the reviewer's audit at ead8dcd8, in the coordinator's absence)

Evidence level E2; one seed, a first measurement. The job is the one the audit of E49 names as next, and the user asked for it to be drafted on 16 November. The specification, the specification file and the predictions P50.1 to P50.3 and P50.8 are the coordinator's ("mine" in the text kept from revision 1 is the coordinator). Revision 1 (f8f62df6) was audited by the reviewer (`PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/AUDIT_E50_SPARSE_INPUT_CODE_BRIEF_20261117.md` on `claude/vibrant-cerf-sje726`, commit ead8dcd8), which required nine changes and stated that, if they are applied as written, no further audit is needed. **Revision 2 enters those nine changes as section 9 of the audit gives them, with the decisions that the reviewer took in the coordinator's place** (`PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/COORDINATION_20261117/WORKER_E50_FIXED_INPUT_CODE.md` on the same branch, section 3). It was entered by the worker of this job, on the worker's branch, because the coordinator cannot be reached; the coordinator has not seen it and will read it on its return. The last section lists the changes. **Run on the reviewer's instruction of 17 November.** One worker; it reuses E49's runner.

Shell names: `D=origin/codex/minifly-consolidated-20261023; M=MOD97_LOCAL_RULE_20261106; K=PROJECT_DOCUMENTS/COORD_CHECKS_20261116`. Host: Linux x86-64, Python 3.13.18, torch 2.15.0+cpu; at most three processes, one thread each; launch outside any wrapper with a time limit. Edit nothing existing.

## What E49 does not test, and this does

The user, 16 November: "biological cells obviously cannot see the whole input, so locality is essential."

E49 limits how many of the input's 512 values a cell reads. The reviewer showed that this does not limit what a cell knows: the present code of a byte is spread over all 512 values, and any 16 of them identify the byte. A cell that reads 16 values there still sees the whole input in the sense that matters.

E50 changes the code so that the limit bites. **Each byte turns on 11 values of its own and no others, and the code is fixed.** A cell that reads 16 of the 512 values then receives a non-zero input for only a few bytes and none at all for the rest. Whatever such a cell needs to know about the other bytes can reach it only through its 10 connections to neighbouring cells.

## The comparison with the record (working rule 1)

- **Reference inside the job:** the same code with the dense interface (`C-dense`), since changing the code is itself a change. E42's `F10` and E49's runs stand beside it.
- **The task needs 20 symbols told apart** (ten digits, four operation symbols, `=`, four task codes, the start symbol); the 26 letters are fillers.
- **What a cell is left with** (self-check 2, seed 73129):

  | Setting | Bytes a cell has input for, of the 46 used | Of the 20 needed | Of the ten digits | Cells a needed byte reaches | Cells a digit has reached at its own position; one position later; two later |
  |---|---|---|---|---|---|
  | `C-dense` (512 values a cell) | 46 | 20 | 10 | 256 | 256; 256; 256 |
  | 64 values | 31 to 41, median 36 | 10 to 19, median 16 | 4 to 10, median 8 | 187 to 211 | not computed (the setting is not run) |
  | 16 values (`C16-in`) | 11 to 16, median 14 | 1 to 10, median 6 | 0 to 7, median 3 | 67 to 87 | 67 to 82; 241 to 255; 256 |
  | `C46-one` (46 values, one of every used byte) | 46 | 20 | 10 | 256 | 256; 256; 256 |

  The last column is the audit's recommendation b (its section 4; the reviewer's figures for seed 73129, `e50_brief_checks.py` part C): it shows how soon a cell that lacks a byte at the byte's own position can have something of it through its connections.

  At 16 values a cell (`C16-in`) a median cell has input for three of the ten digits. No cell is without any input, and every needed byte reaches at least 67 cells, so the information is in the layer and has to travel.
- **E49: the same lists on the learned code** (seed 73129; the reviewer recomputed every reading from the receipts and every count of the last row from the final weights). Under backpropagation `in16` reaches 80 on R at 20,480 updates, 2.16 times `F10`'s 9,472; `out8` does not reach it (70.8, still rising), nor does `I64` (65.2, still rising) or `I16` (16.9). Under the sweep no setting with a limited input passes 4.2 on any panel at any row, and `out8` ends at 7.8. Under backpropagation max and min at 12 letters reach 80 in every limited setting (at 7,168 and 7,424 updates in `I16`), so both operands are at the answer; what is not learned within the dose is add and sub.
- **What E49's networks ended with** (the reviewer's measurement on the final weights). With a limited input under backpropagation the code keeps its starting size (the codes of two needed bytes 14.1 to 16.2 apart, 14.4 at the start) and a cell's input for a needed byte ends at 1.1 to 1.9 in root mean square. Under the sweep the code shrinks (3.2 to 5.7 apart) and a cell's input ends at 0.2 to 0.8. In this job the code is fixed, with every two used bytes 15.2 apart.
- **E47's trial** (1,024 cells, seed 73129): 768 cells have no input and are not read; the first family is learned under both rules in 1.16 to 1.27 times `F10`'s updates to 80 on R, and holding those cells at zero costs 5 to 16 points of R. In E47 the 256 cells that read the byte read its whole learned code. E50 asks what E47 does not: whether the network learns when no cell can identify more than a few bytes.
- **Time to travel** (the reviewer's figures for seed 73129, `e50_brief_checks.py` part C; the other two seeds alike). A cell reads 10 cells within 3 rows and columns. The state of one cell reaches another within 3 connections for 83% of the ordered pairs of cells and within 5 for all. A byte starts in many cells at once and travels faster: with 16 values a cell a needed byte is read by 67 to 87 cells, one position later it has reached at least 241 of the 256, and two positions later all of them. The second operand's last digit stands one position before `=`, where the answer's first byte is read, so at `=` all but 1 to 15 cells can hold something of that digit. The cells that do not read it hold what the states of their neighbours carried, one number for each connection, and not the digit's code. Whether cells learn to use what arrives in that form is what `C16-in` asks. No failure on the answer's first byte in particular follows from the connections; the two answer bytes are scored apart as a description.
- **Dose.** `F10` reaches 80 on R at 15,104 updates under the sweep and at 9,472 under backpropagation, so `C-dense`, read against `F10` within 24,064 recordable updates, can be recorded up to 1.59 and 2.54 times `F10`'s figure. The limited settings are read against `C-dense`, and the factor that applies to them is known only when `C-dense` has run; they run twice the dose for that reason.

## The network, as code

**`$K/e50_sparse_code_spec.py` on the default branch is the specification** (self-checks without training; my output is `$K/OUTPUT_E50_SPEC_MAC.txt`). Copy it beside the runner's copies of E42's and E49's files, with its two imports pointed at them, and otherwise do not rewrite it. In words:

- **The code.** `vals = list(range(512)); Random(f"e50|code|{seed}").shuffle(vals)`. The 46 symbols that occur as input in the first family (the ten digits, `+ - > <`, `=`, the 26 capital letters, `p q v w`, and the start symbol), in increasing order of byte with the start symbol (257) last, take 11 values each in turn: symbol number k turns on `vals[11k : 11k + 11]`. 506 values have one owner each and 6 have none. Every other byte has the zero code; of these only the end symbol ever stands in an input, after an answer, where nothing is scored.
- **The digest of the code** is the first 16 hexadecimal digits of the SHA-256 of `json.dumps(owner)` with Python's default separators, where `owner` is the list of 512 entries giving, for each value, the byte that turns it on, or -1 for the six values without an owner. For seed 73129 it is `12111044f0f8bc70`; the reviewer's reader gives `53fb9ffe8aae97fb` for seed 93683 and `f822e27e800c23a6` for seed 116089.
- **The value of a turned-on entry** is one number for the whole table: the square root of one eleventh of the mean, over the 46 used bytes (the start symbol included, whose present code is the short one), of the squared length of `tanh(l1(byte table))` in the network as constructed for the seed. For seed 73129 the mean squared length is 115.38 and the value 3.2387. The runner computes it before the start and records it. Every used byte then has the squared length 115.4, where the present code's lengths run from 19.6 to 204.4.
- **The code is fixed.** The byte table and `l1` take no part and receive no gradient (self-check 2). The code is what `l1`'s output was: it feeds the cells through the input map and the readout through `l2`.
- **The cells and the two masks are E49's** (E42's `F10` layer; `e49_interface_spec.Interface` with E49's draws and its scaling of the kept weights), and the two rules are E49's. E49's strings do not name the job, so `C16-in` reads exactly the input values that E49's `I16` and `in16` read.
- **The draw of `C46-one`** (below) is not in the specification file, which stays as it is on the default branch; the runner makes it from the text of the next section, and the reviewer checks its digest.

## Settings and stages, seed 73129

| Stage | Name | Values a cell reads | Cells a readout unit reads | Run under | Updates | Weights that take part | Products with a turned-on value |
|---|---|---|---|---|---|---|---|
| 1 | `C-dense` | 512 | 256 | the sweep and backpropagation | 24,576 | 926,464 | 285,184 |
| 2 | `C16-in` | 16 (E49's lists for K_IN = 16) | 256 | each rule under which `C-dense` reaches 80 on R within its dose | 49,152 | 545,536 | about 277,000 |
| 3 | `C46-one` | 46, one value of every used byte | 256 | each rule under which `C16-in` does not reach 80 on R within its dose | 49,152 | 568,576 | 277,504 |

The first count is E49's (every weight of a map inside its mask, the cells' 8,448, `l2`'s 262,144 and `l3`'s 131,584) without `l1`'s 4,096, which this job does not compute; E49's function `counts` still adds them and is not used unchanged, and the receipts' count of weights leaves out the byte table and `l1` (6,672 parameters). The second count is of products with a value that is not zero: with 11 of 512 values turned on, the input map has 8,448 of them under the dense interface and `l2` has 5,632. By the second count a limited input saves 3%. The job is about what a cell knows, not about computation; the multiply-adds to a threshold are given by the first count, as in E49, with the second beside them, and by the first count a limited setting is *more* only above 2.55 times the updates of `C-dense`.

**The readout is not limited in this job.** E49 ran the masks of revision 1 on the learned code. With the readout alone limited (`out8`) the network did not reach 80 on R within the dose under either rule; with the input alone limited (`in16`) it reached it under backpropagation at 20,480 updates. The settings `C64` and `C16` of revision 1 are not run.

**Stage 1 is read against E42's `F10` before stage 2 starts.** Under a rule under which `C-dense` does not reach 80 on R within its dose, the report says that the network does not learn the first family from this fixed code within the dose under that rule, gives E49's figures for a run that does not reach the threshold, and makes no statement about a limited input; stages 2 and 3 are not run under that rule, and what follows is my decision. (The decision, taken by the reviewer in the coordinator's place on 17 November: nothing more is run under that rule, and the reviewer is told.)

**`C46-one`** separates the two things that change together between `C-dense` and `C16-in`: the number of bytes a cell has input for, and the number of weights on which a cell's input for a byte rests. With `own[k] = vals[11k : 11k + 11]` as in the code and `rng = Random(f"e50|one|{seed}")`, for cells 0 to 255 in turn and, within a cell, for the 46 symbols in the code's order, the cell reads `own[k][rng.randrange(11)]`. The kept weights of the input map are scaled by √(512 / 46) = 3.336. The digest (the first 16 hexadecimal digits of the SHA-256 of `json.dumps` of the 256 lists, each sorted) is `6788d5475af8c847` for seed 73129 by the reviewer's reader, which also gives a first-batch loss of 5.591 for this setting and 5.569 for `C16-in`. Such a cell has input for every used byte, each through one weight. If `C46-one` reaches 80 on R and `C16-in` does not, the number of bytes a cell has input for is what costs. If neither does, the cost is that of an input resting on one weight for a byte, and nothing is said about what a cell knows.

**Dose.** Stage 1 runs E26's birth, 24,576 updates. Stages 2 and 3 run 49,152 updates, scored every 256, so that a threshold can be recorded up to update 48,640. Every reading of such a run is given twice: within the first 24,576 updates, which is the dose of `C-dense`, of `F10` and of E49, and at the end. The reason is on file: under backpropagation E49's `in16` needed 2.16 times `F10`'s updates to 80 on R, and within 24,064 recordable updates a setting that costs as much against `C-dense` is recorded only if `C-dense` reaches 80 on R by update 11,130.

Two to six runs; about 1 CPU-hour for 24,576 updates (E49's eight runs took 0.90 to 1.02) and 2 for 49,152; at most 10 CPU-hours.

**Runner:** E49's with this forward; E49's data, optimizer, scores and receipts. Stage 2 starts as soon as stage 1's report is sent, and stage 3 as soon as stage 2 has ended under a rule, without a further release. The audit's recommendation a (a setting with 64 values a cell and the dense readout) is not run.

## Checks before the runs

1. **The specification file.** The runner's copy differs from `$K/e50_sparse_code_spec.py` in its import lines only; run on `$K`, the self-checks print what `$K/OUTPUT_E50_SPEC_MAC.txt` has: 46 used symbols with 11 values each and no value with two owners; the digest of the code (12111044…); for each setting what a cell has input for, that no cell is without input, the byte table and `l1` without gradient, gradients zero outside the masks.
2. **The code is fixed through a run:** the table has the same SHA-256 at the start and at every scored row.
3. **E49's checks 3, 4 and 5** (the masks hold; the batch digest is E26's; the scorer on E26's births). E49's checks 1 and 2 passed on E49's runner at a36de661; E49's check 5 is run through E49's forward, since E26's births have a learned code.
4. **The forward against a second form.** On the first batch, for `C16-in` under backpropagation, the logits equal those of a forward written cell by cell, in which a cell's input is the sum of its weights on the turned-on values it reads, to 1e-5.
5. One dry run of 256 updates per setting and rule, with CPU time and a forecast; the dry runs commit their weights, the code table, the masks and the answers on the length-test panel, for the reviewer's reader. The weight files hold the code table (258 × 512) and the list of owners under names that the report states, and each receipt says whether the byte table and `l1` are bit-equal at the end to their starting values.

## What is recorded

E49's measures (A, R, the taught panel, the length test; updates and multiply-adds to the thresholds in the three-row version; the cosines for the masked maps; the size of the kept weights), and three things for this job:

- **The answer's two digits apart:** on R's items, the share with the first answer byte right and the share with the second right, at every scored row.
- **The four operations apart:** on the panel behind A (untaught pairs at 12 letters, 300 items for each of add, sub, max and min, which the runner already counts), the share right for each operation at every scored row, and the updates to 80 in the three-row version. The answer to max or min is one of the operands, so a setting in which max and min reach 80 and R does not is described as bringing both operands to the answer without learning the sums within the dose.
- **By what a cell knows:** at the end of the run, the mean magnitude of the recurrent weights into cells that have input for 0 to 2, 3 to 4 and 5 or more of the ten digits (`C16-in`). It describes whether cells that know less lean more on their neighbours. The same figure is given at the start, and for the same three groups of cells (95, 131 and 30 cells for seed 73129) in `C-dense`, where the grouping means nothing. The figure describes the size of weights and is not read as use. The reviewer computes a test of use from the committed files (`e50_code_reader.patch_test`): the states, at `=`, of the cells that read neither of two digits are replaced by their states in a run with the other digit, and the share of items in which the answer's first byte follows them is reported.
- **The bytes of each answer given** on the length-test panel at the last row (working rule 3).

Starting and final weights of every run and the starting network of every setting, with the code table, the list of owners and the masks, on the remote (`git add -f`, listed from `git ls-tree`). The births of `C-dense` will be used by a later job, so each `C-dense` receipt records the SHA-256 of its final weight file.

## How the result is stated (fixed beforehand; E49's forms)

`C16-in`, and `C46-one` if it runs, against `C-dense` of the same rule; `C-dense` against E42's `F10`; and beside `C16-in` against `C-dense`, E49's `in16` against `F10` under the same rule (2.16 times the updates to 80 on R and 1.91 times to 80 on A under backpropagation; not learned under the sweep).

- **Updates to the 80 threshold** on A and on R: E41's table of four cases, read as fewer, more or about the same. Multiply-adds beside it, with E49's caution on what that reading can be.
- **The level:** the mean of A, of R and of the length test at 40 letters over the last three scored rows; *lower* or *higher* at 15 points or more.
- **A run that does not reach 80 on R:** E49's rule (the mean of R over the last three rows and its difference from the three rows ending 2,048 updates earlier; *still rising* at 5 points or more).

**The sentence this job can license**, for this seed and each rule: with a code in which a value stands for one byte and each cell reading 16 values of it and every readout unit reading every cell, so that a cell has input for a stated number of the bytes it must tell apart, the network **learns the first family** (reaches 80 on R within the dose) or **does not within the dose**; and whether in about the same updates and to about the same level as with every cell reading every value. If `C16-in` does not reach 80 on R, the job says why only through `C46-one`. Under backpropagation, a cost of `C16-in` against `C-dense` that is no larger than the cost of `in16` against `F10` is described as the cost of the mask and is not attributed to what a cell knows. One seed: a description.

## What it cannot show

- **More than one seed**, until released.
- **A learned sparse code.** The code is given and fixed; whether a learner can form one, or keep one sparse under training, is not asked.
- **The readout's direct reading of the byte.** Each readout unit still reads all 512 values of the current byte's code through `l2`; only its reading of the cells is limited. The answer's digits could in part be carried there.
- **An input of many tokens.** There is still one byte at a time; the code makes a cell ignorant of most bytes, not of most of a long input.
- **More cells**, and **settings**, as in E49.
- **A cell that stays ignorant of a byte.** A cell's ignorance lasts one position: one position later the byte has reached at least 241 of the 256 cells, through one number for each connection.
- **A limited readout.** Every readout unit reads every cell; E49's `out8` is the record on that side.
- **The same difference between the two rules as in E49.** The code has no gradient, so nothing is carried back through the input map or through `l2`; under the sweep the sign rule acts on the recurrent map, on `lm` and on `l3` only.
- **Why the fixed code costs, if it does.** `C-dense` differs from `F10` in that the code is not learned, in that a value stands for one byte, in that every byte has the same length, and in how far one update can move a cell's input; the job does not separate these.
- **Letters.** 286 of the 512 values belong to the 26 letters; of the 16 values a cell reads, about 9 are letters' and about 7 belong to the 20 bytes the task must tell apart.

## Predictions (P50.1 to P50.3 and P50.8 the coordinator's, of revision 1; P50.4' to P50.8' the reviewer's, stated on 17 November before the runs)

| | Prediction | Chance |
|---|---|---|
| P50.1 | Checks 1 to 4 pass | 85% |
| P50.2 | `C-dense`, backpropagation: reaches 80 on R within the dose | 80% |
| P50.3 | `C-dense`, sweep: reaches 80 on R within the dose | 55% |
| P50.8 | `C-dense` against E42's `F10`, backpropagation: updates to 80 on R *about the same* or *fewer* | 60% |

P50.4 to P50.7 of revision 1 are withdrawn with the settings they name (`C64` and `C16`). In their place, the reviewer's chances for `C16-in`, stated before the runs (the worker's instructions, section 3):

| | Statement | Chance |
|---|---|---|
| P50.4' | `C16-in` reaches 80 on R within 49,152 updates under back-propagation | 55% |
| P50.5' | `C16-in` reaches 80 on R within 49,152 updates under the sweep | 25% |
| P50.6' | Under back-propagation `C16-in` reaches 80 on R in no more than 3.24 times the updates of `C-dense` (1.5 times the factor 2.16 that E49's `in16` needed against `F10`) | 45% |
| P50.7' | In `C16-in`, max and min at 12 letters reach 80 under back-propagation | 90% |
| P50.8' | In `C16-in`, max and min at 12 letters reach 80 under the sweep | 45% |

**An explanation under test, with the result that refutes it.** The sweep did not learn with a limited input in E49, and its networks ended with a small code and a small input to the cells. One explanation is that the sweep does not keep a learned code at the size that a cell reading few values requires. Here the code cannot shrink. The explanation is refuted as a sufficient one if `C16-in` under the sweep does not reach 80 on R within its dose while `C-dense` under the sweep does.

## Done

The runner, the check receipts, the two to six runs with receipts and files on the remote, and `E50_REPORT.md` with the tables (setting × rule), the descriptions, the sentence per rule and the outcome of each prediction. No readings beyond the forms above.

## What changed from revision 1 (the audit's nine changes, entered as its section 9 gives them)

| | Change |
|---|---|
| 1 | **The settings, the stages and the dose.** `C64` and `C16` are not run. Stage 1 is `C-dense` under both rules; stage 2 is `C16-in` at 49,152 updates under each rule under which `C-dense` reaches 80 on R; stage 3 is the control `C46-one`, defined here, under each rule under which `C16-in` does not. The table "What a cell is left with" names its rows by the number of values and gains the row of `C46-one` |
| 2 | **The comparison with the record** takes in E49 and E47's trial; the paragraph "Time to travel" is replaced (two of its statements did not hold), the expectation of a failure on the answer's first byte is removed, and the paragraph "Dose" is restated |
| 3 | **What is recorded:** the four operations apart; the figure "by what a cell knows" also at the start and in `C-dense`, and not read as use; the reviewer's test of use |
| 4 | **How the result is stated:** against `C-dense`, with E49's `in16` against `F10` beside it; what the sentence may say when `C16-in` does not learn, and when its cost is no larger than the mask's |
| 5 | **The text of the code:** the 46 symbols listed; the formula of the digest; the value of a turned-on entry defined; `C16-in` reads the input values of E49's `I16` and `in16` |
| 6 | **Multiply-adds:** two counts for each setting, and which one the readings use |
| 7 | **Checks:** E49's checks 1 and 2 are on file and its check 5 runs through E49's forward; check 4 names `C16-in` with a tolerance of 1e-5; the weight files name the code table and the owners, and each receipt says whether the byte table and `l1` are bit-equal to their start |
| 8 | **Predictions:** P50.4 to P50.7 withdrawn; the reviewer's P50.4' to P50.8' stated before the runs; the explanation under test with the result that refutes it |
| 9 | **"What it cannot show":** five items added; "Done" says two to six runs |
| — | **Decisions taken by the reviewer in the coordinator's place** (the worker's instructions, section 3): the chances of P50.4' to P50.8'; under a rule under which `C-dense` does not reach 80 on R nothing more is run and the reviewer is told; the audit's recommendation b is entered (the last column of the table "What a cell is left with") and recommendation a is not run; the final weights of every run and the starting network of every setting are committed, and each `C-dense` receipt records the SHA-256 of its final weight file |
| — | **Entered by the worker and not in the audit's text:** this section and the header; the sentence that the draw of `C46-one` is made by the runner and not added to the specification file, which no worker may edit; the sentence on the runner and on when stages 2 and 3 start. Text kept from revision 1 in which "I", "my" or "mine" occurs is the coordinator's |
