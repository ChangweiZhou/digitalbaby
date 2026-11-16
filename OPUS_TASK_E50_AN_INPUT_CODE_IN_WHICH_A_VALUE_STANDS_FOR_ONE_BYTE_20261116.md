# E50: an input code in which a value stands for one byte, so that a cell which reads few values knows few bytes (16 November 2026, revision 1)

Evidence level E2; one seed, a first measurement. The job is the one the audit of E49 names as next, and the user asked for it to be drafted on 16 November. The specification, the specification file and the predictions are mine. **Not released: it waits for the reviewer's audit.** One worker, best the E49 worker; it reuses E49's runner.

Shell names: `D=origin/codex/minifly-consolidated-20261023; M=MOD97_LOCAL_RULE_20261106; K=PROJECT_DOCUMENTS/COORD_CHECKS_20261116`. Host: Linux x86-64, Python 3.13.18, torch 2.15.0+cpu; at most three processes, one thread each; launch outside any wrapper with a time limit. Edit nothing existing.

## What E49 does not test, and this does

The user, 16 November: "biological cells obviously cannot see the whole input, so locality is essential."

E49 limits how many of the input's 512 values a cell reads. The reviewer showed that this does not limit what a cell knows: the present code of a byte is spread over all 512 values, and any 16 of them identify the byte. A cell that reads 16 values there still sees the whole input in the sense that matters.

E50 changes the code so that the limit bites. **Each byte turns on 11 values of its own and no others, and the code is fixed.** A cell that reads 16 of the 512 values then receives a non-zero input for only a few bytes and none at all for the rest. Whatever such a cell needs to know about the other bytes can reach it only through its 10 connections to neighbouring cells.

## The comparison with the record (working rule 1)

- **Reference inside the job:** the same code with the dense interface (`C-dense`), since changing the code is itself a change. E42's `F10` and E49's runs stand beside it.
- **The task needs 20 symbols told apart** (ten digits, four operation symbols, `=`, four task codes, the start symbol); the 26 letters are fillers.
- **What a cell is left with** (self-check 2, seed 73129):

  | Setting | Bytes a cell has input for, of the 46 used | Of the 20 needed | Of the ten digits | Cells a needed byte reaches |
  |---|---|---|---|---|
  | `C-dense` (512 values a cell) | 46 | 20 | 10 | 256 |
  | `C64` (64) | 31 to 41, median 36 | 10 to 19, median 16 | 4 to 10, median 8 | 187 to 211 |
  | `C16` (16) | 11 to 16, median 14 | 1 to 10, median 6 | 0 to 7, median 3 | 67 to 87 |

  At `C16` a median cell has input for three of the ten digits. No cell is without any input, and every needed byte reaches at least 67 cells, so the information is in the layer and has to travel.
- **Time to travel.** A cell's 10 sources lie within 3 rows and columns on a 16 × 16 grid, so any cell is within 3 connections of any other, and a byte reaches every cell in 3 steps. The second operand's last digit stands one step before the answer's first byte; a cell without input for that digit cannot use it for that byte, which only the cells that read it can supply. This is stated before the run as the reason `C16` may fall short on the answer's first byte.
- **Nothing on file fixes the input code of this learner.** The nearest is the random sparse expansion of October, without learned recurrence, which memorised.
- **Dose:** E26's birth, 24,576 updates. `F10` under the sweep reaches 80 on R at 15,104 of 24,064 recordable updates, so a setting slower by a factor above 1.59 cannot be told from one that does not learn within the dose (2.54 under backpropagation).

## The network, as code

**`$K/e50_sparse_code_spec.py` on the default branch is the specification** (self-checks without training; my output is `$K/OUTPUT_E50_SPEC_MAC.txt`). Copy it beside the runner's copies of E42's and E49's files, with its two imports pointed at them, and otherwise do not rewrite it. In words:

- **The code.** `vals = list(range(512)); Random(f"e50|code|{seed}").shuffle(vals)`. The 46 symbols that occur as input in the first family, in increasing order of byte with the start symbol (257) last, take 11 values each in turn: symbol number k turns on `vals[11k : 11k + 11]`. 506 values have one owner each and 6 have none. Every other byte has the zero code; of these only the end symbol ever stands in an input, after an answer, where nothing is scored.
- **The value of a turned-on entry** is one number for the whole table, chosen once so that the code of a used byte has the mean squared length that the present network's code has over the used bytes at the start: 3.2387 for seed 73129.
- **The code is fixed.** The byte table and `l1` take no part and receive no gradient (self-check 2). The code is what `l1`'s output was: it feeds the cells through the input map and the readout through `l2`.
- **The cells and the two masks are E49's** (E42's `F10` layer; `e49_interface_spec.Interface` with E49's draws and its scaling of the kept weights), and the two rules are E49's.

## Settings, seed 73129, each under the sweep and under backpropagation

| Name | Values a cell reads | Cells a readout unit reads |
|---|---|---|
| `C-dense` | 512 | 256 |
| `C64` | 64 | 32 |
| `C16` | 16 | 8 |

Six runs of 24,576 updates. Runner: E49's with this forward. About 0.8 CPU-hours a run at E42's measured rate for `F10`.

**One more setting, without a further release, if `C16` under a rule does not reach 80 on R within the dose or rises against `C-dense`:** `C16-in` (16 values a cell, the readout dense), to say whether the cost is at the input, which is the side this job is about.

## Checks before the runs

1. **The specification file.** The runner's copy differs from `$K/e50_sparse_code_spec.py` in its import lines only; run on `$K`, the self-checks print what `$K/OUTPUT_E50_SPEC_MAC.txt` has: 46 used symbols with 11 values each and no value with two owners; the digest of the code (12111044…); for each setting what a cell has input for, that no cell is without input, the byte table and `l1` without gradient, gradients zero outside the masks.
2. **The code is fixed through a run:** the table has the same SHA-256 at the start and at every scored row.
3. **E49's checks 3, 4 and 5** (the masks hold; the batch digest is E26's; the scorer on E26's births).
4. **The forward against a second form.** On the first batch, for `C16` under backpropagation, the logits equal those of a forward written cell by cell, in which a cell's input is the sum of its weights on the turned-on values it reads, to 1e-6.
5. One dry run of 256 updates per setting and rule, with CPU time and a forecast; the dry runs commit their weights, the code table, the masks and the answers on the length-test panel, for the reviewer's reader.

## What is recorded

E49's measures (A, R, the taught panel, the length test; updates and multiply-adds to the thresholds in the three-row version; the cosines for the masked maps; the size of the kept weights), and three things for this job:

- **The answer's two digits apart:** on R's items, the share with the first answer byte right and the share with the second right, at every scored row. The comparison with the record says the first byte is where a cell without the last digit should fail.
- **By what a cell knows:** at the end of the run, the mean magnitude of the recurrent weights into cells that have input for 0 to 2, 3 to 4 and 5 or more of the ten digits (`C16`). It describes whether cells that know less lean more on their neighbours.
- **The bytes of each answer given** on the length-test panel at the last row (working rule 3).

Starting and final weights with the code table and the masks, on the remote (`git add -f`, listed from `git ls-tree`).

## How the result is stated (fixed beforehand; E49's forms)

`C64` and `C16` against `C-dense` of the same rule; and `C-dense` against E42's `F10`.

- **Updates to the 80 threshold** on A and on R: E41's table of four cases, read as fewer, more or about the same. Multiply-adds beside it, with E49's caution on what that reading can be.
- **The level:** the mean of A, of R and of the length test at 40 letters over the last three scored rows; *lower* or *higher* at 15 points or more.
- **A run that does not reach 80 on R:** E49's rule (the mean of R over the last three rows and its difference from the three rows ending 2,048 updates earlier; *still rising* at 5 points or more).

**The sentence this job can license**, for this seed and each rule: with a code in which a value stands for one byte and each cell reading K values of it, so that a cell has input for a stated number of the bytes it must tell apart, the network **learns the first family** (reaches 80 on R within the dose) or **does not within the dose**; and whether in about the same updates and to about the same level as with every cell reading every value. One seed: a description.

## What it cannot show

- **More than one seed**, until released.
- **A learned sparse code.** The code is given and fixed; whether a learner can form one, or keep one sparse under training, is not asked.
- **The readout's direct reading of the byte.** Each readout unit still reads all 512 values of the current byte's code through `l2`; only its reading of the cells is limited. The answer's digits could in part be carried there.
- **An input of many tokens.** There is still one byte at a time; the code makes a cell ignorant of most bytes, not of most of a long input.
- **More cells**, and **settings**, as in E49.

## Predictions (mine)

| | Prediction | Chance |
|---|---|---|
| P50.1 | Checks 1 to 4 pass | 85% |
| P50.2 | `C-dense`, backpropagation: reaches 80 on R within the dose | 80% |
| P50.3 | `C-dense`, sweep: reaches 80 on R within the dose | 55% |
| P50.4 | `C64`, backpropagation: reaches 80 on R within the dose | 65% |
| P50.5 | `C16`, backpropagation: reaches 80 on R within the dose | 35% |
| P50.6 | `C16`, sweep: reaches 80 on R within the dose | 15% |
| P50.7 | `C16`, backpropagation, at the end: the first answer byte is right less often than the second, by 10 points or more | 45% |
| P50.8 | `C-dense` against E42's `F10`, backpropagation: updates to 80 on R *about the same* or *fewer* | 60% |

## Done

The runner, the check receipts, the six or seven runs with receipts and files on the remote, and `E50_REPORT.md` with the tables (setting × rule), the descriptions, the sentence per rule and the outcome of each prediction. No readings beyond the forms above.
