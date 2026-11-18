# Verification of stage 3 of E50 and of the final report (seed 73129), 18 November 2026

## Verdict

**Stage 3 is confirmed; the final report is confirmed in its figures, with four small differences of rounding and one reading that says more than its figures support.** Every figure of the final report's two result tables agrees with what I recomputed from the receipts (140 of 140, of which 40 of 40 for the setting `C46-one`), and every row of the tables file for `C46-one` is found in it character for character (80 of 80 rows written from the receipts, 8 of 8 written from the weight files). The ratios 2.22 and 2.68 are right (2.2222 and 2.6800). The saved networks give the recorded scores: the reviewer's forward pass reproduces every count of right answers of the scored rows at updates 24,576 and 49,152 of both runs (4 of 4 rows) and 2,400 of the 2,400 answers that the runs recorded. The masks hold in all six weight files of stage 3: every cell reads exactly 46 values, one of each of the 46 used bytes, and the draw written here from the text of the runner gives the digest that the brief states. The code is fixed: the code table, its list of owners, its value, the byte table and the layer `l1` are bit-equal in the start file, the file at update 24,576 and the final file of both runs. The restart changed nothing that the files show: each completed run equals its interrupted predecessor on every scored row that both have, in every recorded field except the clock and the processor time, and from byte-equal start files.

**Differences.** (1) The final report gives the mean magnitude of the recurrent weights at the start as "0.150, 0.152, 0.150"; the weight files give 0.151 for the second group (0.151484). The stage 2 report was amended to 0.151 at 1799b2ba; the final report was not. (2) For `C46-one` under the sweep the report gives 0.212 for the first group at the end; the weight file gives 0.211491, which is 0.211 (the tables file's 0.2115 rounded a second time). (3) The level of R of `C46-one` under backpropagation is 94.25; the report's table has 94.2 and its text 94.3. (4) The final report itself has no dated section "Amendments", unlike the two stage reports; its changes at 1799b2ba are named only in one sentence of its paragraph "Completion", which mentions the stage reports' amendments and not its own.

**A reading that says more than its figures.** The bold sentence "**the number of bytes a cell has input for is what costs**", given for both rules, is the form that the brief licenses when `C46-one` reaches 80 on R and `C16-in` does not, and that condition is met under both rules. Read as written, it says that this number is the cost. The figures support less: on one seed, giving every cell one value of every used byte turns a run that does not reach 80 on R within 49,152 updates into one that does, but `C46-one` itself needs 2.22 (backpropagation) and 2.68 (the sweep) times the updates of `C-dense` and ends far lower on the length test at 40 letters; and `C46-one` differs from `C16-in` also in the number of values a cell reads (46 against 16) and in the scale of the starting input weights. Section 4 gives the details.

**Not checkable without training**, and therefore still the worker's record: the scores at the scored rows other than those at updates 24,576 and 49,152; the checks that the runner made inside the runs; the cosines between the gradients of the two rules; that each run was trained under the rule it names; the reboot itself; and the processor times.

## Terms, material and method

**Method.** Read-only. Nothing was trained, torch is not installed, and no code of another branch was executed. One script beside this note, `e50_stage3_verify.py`, prints every figure given here; *part N* names the part of its output that gives a figure. The script imports the functions of the verification of stage 2 (`e50_stage2_verify.py`: the reading of receipts, thresholds, levels, the class that evaluates a saved network, the panels, the stream of training items) and points them at 36aa8771. Every file of the worker's branch was read with `git show` into memory; no weight file was written to disk.

**Material.** The worker's branch of E50 at 36aa8771 for the receipts, the weight files, the answers files, the logs, the tables file `E50_TABLES_73129.md` and the summary file `E50_SUMMARY_73129.json`; at 1799b2ba, the head of that branch when fetched and the child of 36aa8771, for the final report `E50_REPORT.md` and the two stage reports. Between the two commits only these three reports changed (part 0); no receipt, weight file, answers file, table, log or source file. The runner `exp50_code.py`, its specification file `e50_sparse_code.py`, the tables script `e50_tables.py` and E49's two files are the same files at 36aa8771 as at c6c24d9c (the commit read by the verification of stage 2) and, for the runner and the specification file, as at 0f3ec4bc, the commit the report names (part 0). The 29 files of stages 1 and 2 and of the references under `exp45/e50/` are unchanged since c6c24d9c (part 0). The brief, the file whose name contains `E50_AN_INPUT_CODE_IN_WHICH_A_VALUE_STANDS_FOR_ONE_BYTE`, revision 2, is the same file at c6c24d9c, 36aa8771 and 1799b2ba.

**Terms.**

- The *code* of a byte is the 512 values that feed the cells and the readout. Here it is fixed: each of the 46 *used bytes* (the bytes that occur as input in the task) turns on 11 values of its own, each equal to 3.2387, and every other value is zero. The *code table* holds the code, and its *owners* are the list that gives, for each value, the byte that turns it on. The *byte table* and the layer `l1` produced the learned code of earlier jobs and take no part here. The 20 *needed bytes* are those the task must tell apart (ten digits, four operation signs, `=`, four task codes and the start symbol). A cell *has input for* a byte if it reads at least one of the byte's values.
- `C-dense` (stage 1) is a network of 256 cells, each reading 10 neighbouring cells and itself, every cell reading all 512 values of the code and every readout unit reading all 256 cells. `C16-in` (stage 2) is the same network with each cell reading 16 of the 512 values; a cell then has input for 1 to 10 of the 20 needed bytes. `C46-one` (stage 3) is the same network with each cell reading one value of every used byte, 46 values; a cell then has input for every byte, each through one value. A *mask* marks the entries of a weight matrix that are kept. `in16` is the earlier job E49's network with the same lists of 16 values on a learned code, and `F10` is the earlier job E42's network with a learned code and every cell reading every value.
- The two *rules* are the *sweep* (S), the local rule of this programme, and *backpropagation* (B), the exact gradient.
- *A* is the share of whole answers right on 300 untaught pairs each of add and sub with 12 letters between the operands; the same panel holds 300 pairs each of max and min. *R* is the share right on 1,000 untaught pairs each of add and sub, without letters, whose answer needs the reduction by the modulus 89. The *length test* is 300 untaught pairs of add at 12 and at 40 letters. The *taught panel* is 300 taught pairs each of add and sub at 12 letters.
- A run is scored every 256 updates; each scoring is a *scored row*, and a run of 49,152 updates has 193 rows. A threshold is *reached* at the first of three consecutive scored rows at or above it (the *three-row version*); the *single-row version* takes the first row at or above it. A *level* is the mean over the last three scored rows. The *rise of R* is the level of R minus the mean of the three rows ending 2,048 updates earlier; *still rising* means a rise of 5 points or more. A run of stages 2 and 3 is given in two *readings*: within its first 24,576 updates (the number of updates of `C-dense`) and at its end.
- A *receipt* is the JSON file that a run writes, with one record for each scored row. The *start file*, the *file at update 24,576* and the *final file* of a run hold its tensors before the first update, after update 24,576 and after the last. The *interrupted* runs are the first runs of stage 3, which a restart of the machine stopped at about 23:59Z on 17 November; their partial files are in `interrupted_20261117T2359Z/`.
- The *reader* is the reviewer's numpy implementation of the network (`e50_code_reader.py` with `e49_interface_reader.py` and `e42_cells_reader.py`).

## 1. The result tables, from the receipts alone: hold (part 1)

In both receipts of `C46-one` the run is complete, with 193 scored rows and 49,152 optimizer steps, and the shares recorded in every row equal the shares computed from the row's counts of right answers (part 0). Every figure below is computed from the counts. Thresholds are in updates, three-row version, with the single-row version in brackets.

| `C46-one` | B, first 24,576 | B, end | S, first 24,576 | S, end |
|---|---|---|---|---|
| Updates to 80 on A | 14,080 [12,288] | 14,080 [12,288] | not reached [not reached] | 35,584 [33,536] |
| Updates to 80 on R | 15,360 [13,824] | 15,360 [13,824] | not reached [not reached] | 34,304 [34,304] |
| Updates to 50 on A | 7,936 [7,936] | 7,936 [7,936] | 15,872 [14,592] | 15,872 [14,592] |
| Updates to 50 on R | 7,936 [7,936] | 7,936 [7,936] | 14,592 [14,592] | 14,592 [14,592] |
| Level of A | 88.50 | 93.83 | 63.61 | 85.89 |
| Level of R | 88.87 | 94.25 | 65.50 | 87.22 |
| Level of the taught panel | 94.89 | 98.33 | 84.61 | 94.00 |
| Level of the length test at 12 and at 40 letters | 87.78; 38.22 | 93.44; 54.33 | 65.56; 23.00 | 84.33; 27.78 |
| Rise of R over the last 2,048 updates | +4.03 | +0.02 | +5.57 (*still rising*) | +0.02 |
| max, min, add, sub at 12 letters: updates to 80 | 3,584 [3,584]; 3,584 [3,584]; 14,592 [11,776]; 13,824 [13,824] | the same | 5,120 [5,120]; 5,120 [5,120]; not reached; not reached | 5,120 [5,120]; 5,120 [5,120]; 33,280 [30,464]; 35,584 [34,560] |
| Levels of max, min, add, sub | 95.78, 96.00, 87.78, 89.22 | 98.78, 97.89, 93.44, 94.22 | 97.11, 96.44, 65.56, 61.67 | 98.78, 98.22, 84.33, 87.44 |

- **The final report's result tables.** All 14 rows of its two tables (seven readings under each rule: `C-dense`, `C16-in` in two readings, `C46-one` in two readings, `in16`, `F10`), ten figures each, equal the recomputed figures to the report's decimal: 140 of 140, among them the 40 of `C46-one`. The rows of a level are updates 24,064 to 24,576 and 48,640 to 49,152; those of the rise are 22,016 to 22,528 and 46,592 to 47,104.
- **The tables file.** Of the rows of `C46-one`, 80 were written here in the file's own formats from the recomputed figures (thresholds at 50, 80 and 90 in both versions, the four operations, multiply-adds, levels, the checks inside the runs, the 54 rows of descriptions against `C-dense` and `C16-in`, A and R every 2,048 updates, the operations every 4,096 updates, the training loss), and all 80 are found in the file character for character. The summary file agrees with the recomputed thresholds, levels and rises of all 14 readings: 440 of 440.
- **The ratios.** Updates to 80 on R: 15,360 / 6,912 = 2.2222 under backpropagation and 34,304 / 12,800 = 2.6800 under the sweep, the report's 2.22 and 2.68. On A: 1.8333 and 2.7255 (reported 1.83 and 2.73). Against `C16-in` under backpropagation on A: 0.3459 (reported 0.35).
- **The words of the brief's forms** (part 1). Against `C-dense`: updates *more* on A and on R under both rules; multiply-adds by the first count *stays* under backpropagation and *rises* under the sweep, by the second count *rises* under both; levels of A and R *about the same* (differences −3.83 and −2.83 under backpropagation, −7.61 and −6.88 under the sweep), the length test at 40 letters *lower* (−38.33 and −62.78). Against `C16-in` at its end: under backpropagation *fewer* updates on A and on R and every level *about the same* (A +10.06, R +14.83, 40 letters −1.89); under the sweep *not decided* on updates, A and R *higher* (+36.39, +43.58), the length test *about the same* (+14.22). All as reported.
- **One figure of the text.** The level of R of `C46-one` at the end under backpropagation is 94.25. The report's table gives 94.2 and its text "R 94.3 against 79.4, a difference of 14.8". The difference, 14.83, is right; the two roundings of 94.25 disagree with each other.
- **The sentence on the first reading.** Within its first 24,576 updates `C46-one` under the sweep has a rise of +5.57 and does not reach 80 on R, so it is *still rising*; `C16-in` has +2.73 and +1.65. As reported.

## 2. From the weights

### (a) The recorded answers and counts are those of the saved weights: holds (part 2a)

The forward pass is the reader's, with the draw of 46 values a cell written in this script; the draws equal the three masks stored in each file. A second form, built from the tensors of the file alone (the file's code table and masks as full matrices), differs from it by at most 5.7e-06 in a logit, where the largest logit is 28.0 to 30.9, and gives the same largest output at every position.

| Counts at the scored row (reader = receipt in every cell) | S, final file, row 49,152 | S, file at 24,576 | B, final file, row 49,152 | B, file at 24,576 |
|---|---|---|---|---|
| The panel behind A: add, sub, max, min (300 each) | 239, 260, 296, 293 | 207, 194, 294, 289 | 279, 289, 297, 293 | 264, 261, 289, 285 |
| R: add, sub (1,000 each) | 854, 861 | 747, 602 | 953, 950 | 879, 881 |
| The taught panel: add, sub, max, min (300 each) | 274, 284, 296, 297 | 267, 246, 296, 299 | 298, 297, 299, 295 | 283, 287, 296, 296 |
| R's items, first byte right: add, sub | 893, 906 | 872, 777 | 963, 966 | 901, 925 |
| R's items, second byte right: add, sub | 929, 940 | 846, 728 | 989, 980 | 974, 942 |
| Length test at 12 and at 40 letters (300 each) | 239, 77 | 207, 64 | 279, 181 | 264, 130 |

**The answers files.** Each of the four files holds 300 recorded answers at 12 letters and 300 at 40. Generating each answer byte by byte (the largest output appended and fed back, until the end symbol or four bytes), the reader gives the recorded answer in 600 of 600 items for each weight file, **2,400 of 2,400 in all**. The SHA-256 of each of the four files equals the one that the receipt records.

### (b) The code was fixed: holds (part 2b)

- In both runs `code_table`, `code_owner`, `code_amplitude`, `book.weight` (the byte table), `l1.weight` and `l1.bias` are bit-equal in the start file, the file at update 24,576 and the final file; so are the three masks and the other structural tensors. The ten trained tensors differ between each two successive files in every kept entry (35,328 of the input map, 8,448 of the recurrent map, all entries of the others).
- The code table equals the reader's table bit for bit (506 entries not zero, all 3.238702), and the list of owners equals the reader's; the reader's digest of the code is `12111044f0f8bc70`, as the receipts record.
- The two start files hold the same tensors. Against the start file of `C-dense`, every tensor is bit-equal except the input map, its mask and the number of values a cell reads, and the input map equals that of `C-dense` times the mask times √(512/46) = 3.3362 with no difference in any bit.
- The SHA-256 of the six files equal those that the receipts record; the final files are `8a2e74962a49d5ef…` (backpropagation) and `1dc6cfb92d3a0aec…` (the sweep), as the logs' last lines state.

### (c) The masks hold, and every cell reads exactly 46 values, one of every used byte: holds (part 2c)

- The draw of the 46 values of each cell, written in this script from the text of the runner (`draw_one`) and of the brief, gives the digest `6788d5475af8c847` that the brief and both receipts state, and equals the reader's earlier `draw_one`.
- In each of the six files the mask of the input map equals that draw. Every one of the 256 cells reads exactly 46 values; for every cell and every used byte the cell reads exactly one of the byte's 11 values (11,776 of 11,776 pairs); no cell reads any of the six values without an owner. The three gates of a cell have weights on the same values.
- Of the 357,888 entries of the input map outside its mask and the 188,160 entries of the recurrent map outside its mask, none differs from zero in any of the six files; no kept weight of the input map is exactly zero; the mask of the readout is full; the recurrent mask equals the reader's draw of E42's connections.
- The multiply-adds a position, recounted from the masks, are 568,576 by the first count and 277,504 by the second, as the brief and the receipts state.

### (d) "By what a cell knows" and "other measures" for `C46-one`: hold, with one difference in the third decimal (part 2d)

Recomputed from the weight files in double precision, all figures equal the receipts' rows 0, 24,576 and 49,152 to the receipts' precision, and the eight rows of `C46-one` in the tables file's two tables of sizes are found character for character.

| `C46-one` | Start | Update 24,576 | End | Reported at the end |
|---|---|---|---|---|
| Mean magnitude of the kept recurrent weights from neighbours, three groups (95, 131, 30 cells), B | 0.1498, 0.1515, 0.1501 | 0.1918, 0.2033, 0.1832 | 0.194760, 0.206270, 0.191637 | 0.195, 0.206, 0.192 |
| The same, S | the same | 0.2055, 0.2056, 0.2076 | 0.211491, 0.211425, 0.208330 | 0.212, 0.211, 0.208 |
| A cell's input for a needed byte, root mean square without the bias, B and S | 0.3904 | 0.8211 and 0.7849 | 0.8498 and 0.8127 | 0.85 and 0.81 |
| Mean magnitude of a kept weight of the input map, B and S | 0.1047 | 0.1197 and 0.1100 | 0.1226 and 0.1125 | (not in the report) |

Of the eight figures of the final report that rest on these weight files, seven equal the weight file's figure rounded to the report's decimals. **The one that differs:** the first group under the sweep at the end is 0.211491, which is 0.211 and not 0.212; the report rounds the four-decimal 0.2115 of the tables file a second time, the same slip that the verification of stage 2 found in two figures of `C16-in`. The cosines between the gradients of the two rules (input map 0.755 to 0.555 under backpropagation and 0.755 to 0.515 under the sweep) are the receipts' and were not recomputed.

## 3. The restart: holds (part 3)

| | Backpropagation | The sweep |
|---|---|---|
| Rows in the interrupted receipt | 37 (0 to 9,216) | 9 (0 to 2,048) |
| Rows equal to the completed run's in every field but the clock and the processor time | 37 of 37 | 9 of 9 |
| Scored rows in the interrupted log | 39 (0 to 9,728) | 11 (0 to 2,560) |
| Log lines equal to the completed log's line in the same place, the processor time set aside | 40 of 40 | 12 of 12 |
| Start file of the interrupted run byte-equal to the completed run's | yes | yes |

- The 25 fields compared in each row are the counts and shares of every panel, the counts of the two answer bytes, the training loss, the batch digest, the checks of the masks, of E42's limit and of the fixed code, and the size figures. The two log comparisons cover the scores, the counts of the two answer bytes, the taught panel, the length test and max and min of every scored row. In the receipts the only top-level entry that differs is `complete`; the starting digests, the first-batch loss and the cosines at the start are equal.
- The interrupted receipts name the commits 67a4313c and 11e76ccd (work-in-progress commits of the same branch, before 36aa8771) where the completed ones name c6c24d9c; the runner and its specification file are the same blobs at all three, and the receipts record the same SHA-256 for every source file and no modified source.
- **The stream of training items.** The running SHA-256 over the batches, rebuilt here from the seed without the runner, equals the recorded one at 193 of 193 rows of each completed run; the digests equal those of `C16-in` at all 193 rows and of `C-dense` at its 97, and the optimizer record is that of `C-dense`.
- The report's statement (amended at 1799b2ba) that the two give the same scores, losses and digests "under backpropagation the 39 scored rows from update 0 to 9,728, under the sweep the 11 rows from 0 to 2,560 … data digests and losses from the receipts, which the interrupted runs had last written at 9,216 and 2,048 updates" is exact.
- The completed runs ran without a pause that the receipts show: the largest interval between two successive scored rows is 41.0 seconds under backpropagation and 45.9 under the sweep (part 0).

## 4. The final report as a whole (read at 1799b2ba; part 4 unless named)

**The runs and the costs.** The start and end times of the table "Runs" are those of `logs/queue.log`. The receipts give 0.874, 0.998, 1.761, 1.999, 1.903 and 2.184 processor-hours, 9.719 in all (reported 9.72; the six rounded figures of the report's table add to 9.71). The interrupted pieces, read at the last scored row of each interrupted log, are 0.338 and 0.099 (reported 0.34 and 0.10); the checks' receipts record 0.114 (reported 0.11); in all 10.271, and 9.833 without the interrupted pieces, as reported. The interrupted pieces ran slightly beyond their last scored row, so 0.44 is a lower bound.

**Completion and the departures.** The report states its three departures. (1) *The `rc=` lines.* Each of the six receipts has `complete: true` and each log ends with `e50 complete` (part 4); `queue.log` writes `rc=0` at every end, including the start at 23:59:50Z that the runner refused (part 0), and the launch script at 36aa8771 writes `exit=` instead; it was changed at b1a516ca (18 November 00:38Z), after the last launch (00:00:18Z). (2) *The commit whose message names a model.* Of the commits from 7cabb13a to 1799b2ba, b1a516ca alone has a co-author line, and that line names a language model and its version (part 0, and a search of the commit messages outside the script). Every commit message of the branch also carries the session lines that the working environment adds, which name the assistant product but no model version; and the brief's file name, set by the coordinator and quoted in the reports, begins with the name of a model family. Apart from these names, no committed file of the job names a language model (a search of the job's files outside the script). (3) *The cost over the limit:* stated, 0.27 processor-hours, as computed above.

**The statements of the summary against the figures of the three stages.** The figures of stages 1 and 2 are those of the two earlier verifications and are reproduced by part 1 of this script.

| Statement of the final report | Figures | Holds |
|---|---|---|
| `C-dense` against `F10`: *about the same* (0.88, 0.73; 0.98, 0.85); inside the range of E42's three seeds on A, earlier than every seed on R | stage 1 verification, its table of seeds | yes |
| `C16-in`: does not learn the first family within the dose under either rule (R 79.4, single row 80.75; R 43.6) | part 1 | yes |
| `C46-one`: learns it under both rules (80 on R at 15,360 and 34,304) | part 1 | yes |
| `C46-one` within the first 24,576 updates reaches 80 on R under backpropagation and not under the sweep, where it is *still rising* | part 1 | yes |
| The cost of `C16-in` against `C-dense` under backpropagation is larger than `in16` against `F10` (2.16) and is not described as the cost of the mask | more than 7.04 against 2.16 | yes |
| Under the sweep the order is reversed (`C16-in` R 43.6, max and min 96; `in16` R 1.4, max and min 39 and 41) | stage 2 verification | yes |
| The explanation under test is refuted as a sufficient one | `C16-in` S does not reach 80 on R, `C-dense` S does at 12,800 | yes |
| max and min reach 80 in every run (`C-dense` 1,024 to 1,536, `C16-in` 5,376 to 7,680, `C46-one` 3,584 to 5,120) | part 4 | yes |
| The two answer bytes (98.1, 84.2, 95.6 and 97.6, 92.6, 98.2 under B; 96.0, 71.5, 90.2 and 94.3, 54.0, 95.5 under S) | part 4: 98.13, 84.23, 95.57; 97.63, 92.58, 98.23; 96.03, 71.45, 90.23; 94.30, 53.98, 95.47 | yes |
| By what a cell knows, at the start "0.150, 0.152, 0.150" | 0.149834, 0.151484, 0.150113 | **no**: 0.151 (amended in the stage 2 report, not here) |
| By what a cell knows, `C46-one` S "0.212, 0.211, 0.208" | 0.211491, 0.211425, 0.208330 | **no** in the first figure: 0.211 |
| The recurrent weights grow in both limited settings and not in `C-dense` | `C46-one` 0.15 to 0.19–0.21; `C16-in` to 0.21–0.27; `C-dense` 0.144–0.152 | yes |
| A cell's input for a needed byte ends at 5.14, 4.77; 0.77, 0.63; 0.85, 0.81, from 0.38 to 0.39 | 5.1399, 4.7685; 0.7650, 0.6290; 0.8498, 0.8127; 0.3814 to 0.3904 | yes |
| The cosine of the input map falls (`C16-in` B 0.709 to 0.104; `C46-one` B 0.755 to 0.555; `C-dense` 0.757 to 0.650) | receipts | yes (receipts only) |
| The length test at 40 letters falls in both limited settings far more than A and R | `C46-one` 54.33 and 27.78, `C16-in` 56.22 and 13.56, against 92.67 and 90.56 | yes |
| `C46-one` against `C16-in` under B: R "94.3 against 79.4" | 94.25 against 79.42 | the table says 94.2; see section 1 |

**The predictions.** P50.2, P50.3 and P50.8 (`C-dense`: 80 on R under backpropagation, under the sweep; *about the same* or *fewer* than `F10`, ratio 0.73) are true; P50.4', P50.5' and P50.6' (`C16-in` reaches 80 on R under backpropagation, under the sweep, within 3.24 times `C-dense`) are false; P50.7' and P50.8' (max and min reach 80 in `C16-in` under each rule) are true. Each outcome is the one the report gives. P50.1 (the checks before the runs pass) rests on checks that execute torch and is the worker's record. No prediction was registered for `C46-one`.

**The reading "the number of bytes a cell has input for is what costs".** The brief states: "If `C46-one` reaches 80 on R and `C16-in` does not, the number of bytes a cell has input for is what costs." The condition is met under both rules, and the report applies the brief's sentence correctly. Three things limit what the sentence can mean on these figures.

1. *`C46-one` also costs.* It needs 2.22 times the updates of `C-dense` to 80 on R under backpropagation and 2.68 times under the sweep, and its level on the length test at 40 letters is 54.33 and 27.78 against 92.67 and 90.56. The number of bytes a cell has input for accounts for the difference between `C16-in` and `C46-one`, not for the whole difference between `C16-in` and `C-dense`. The report says so in its section "What this job cannot show", but its summary sentence does not.
2. *The two settings differ in more than the number of bytes.* In `C16-in` a cell reads 16 values, in `C46-one` 46, so the input map has 12,288 against 35,328 kept weights, and the kept weights start 5.66 times and 3.34 times larger than in `C-dense`. What `C46-one` holds constant, as the brief intends, is the number of weights behind a cell's input for a byte (one in `C46-one`; one in 1,299 of the 1,517 pairs of a cell and a needed byte for which a cell of `C16-in` has input, by the verification of stage 2). The number of bytes a cell has input for is therefore separated from the number of weights behind each, but not from the number of values a cell reads.
3. *Under backpropagation the contrast is one of speed near the threshold.* `C16-in` ended at 79.42 on R, with two rows above 80 and A past 80 at 40,704 updates; `C46-one` ended 14.83 points higher, which the brief's form calls *about the same*. Under the sweep the contrast is large (87.22 against 43.63, *higher*).

What the figures support, on one seed: with one value of every used byte in every cell, the network reaches 80 on R within the dose under both rules (at 15,360 and 34,304 updates), where with 16 values a cell it does not (79.4 and 43.6 at 49,152 updates); the difference between the two settings that the brief names is the number of bytes a cell has input for, beside which the number of values a cell reads also differs; and an input resting on one value a byte still costs 2.2 to 2.7 times the updates of the network reading every value.

**The comparison with `in16` (2.16).** The report compares the cost of `C16-in` against `C-dense` (more than 7.04) with that of `in16` against `F10` (2.16) and, by the brief's rule, does not describe it as the cost of the mask. That holds. The report does not place `C46-one` beside `in16`: under backpropagation `C46-one` costs 2.22 times `C-dense` to 80 on R and 1.83 on A, `in16` 2.16 times `F10` and 1.91 on A. On one seed the two costs are of the same size; a network in which every cell has input for every byte through one value of a fixed code costs about what the 16-value mask cost on the learned code, in which every cell could identify every byte from its 16 values.

**The stage reports at 1799b2ba.** The amended passages of the stage 1 report (the spread between E42's three seeds of `F10`: 6,912 to 9,216 on A and 7,936 to 9,728 on R under backpropagation, 12,544 to 13,312 and 14,336 to 15,104 under the sweep; the second count of `F10`) and of the stage 2 report (0.151, 0.176; the levels of max and min 96.0 to 97.9 at the end, 94.9 to 95.6 within the first 24,576 updates, single rows 89.7 to 99.3; the `rc=` lines) agree with the two earlier verifications. Each has a dated section "Amendments" that lists what changed and says that no result changed.

## 5. The two measurements of the stage 2 verification, for the final networks of `C46-one`

Both are made on the final weights with the forward pass of section 2(a). The figures of `C16-in` and `C-dense` beside them are those of the verification of stage 2 (its section 3), computed in the same way.

### (a) The use of the cells (part 5a)

The items are the 300 items of the length-test panel at 12 letters (untaught pairs of add). For each cell the standard deviation of its state between the 300 items is taken at the position of `=`, where the answer's first byte is read; the magnitude of the state is its absolute value, averaged over items and cells.

| Network | Rule | Cells above 0.05 | Cells above 0.10 | Mean over the 256 cells | Least, median, largest | Mean magnitude of the state |
|---|---|---|---|---|---|---|
| `C46-one` | S | 255 | 250 | 0.4086 | 0.0379, 0.4248, 0.6685 | 0.4739 |
| `C46-one` | B | 247 | 230 | 0.4198 | 0.0129, 0.4171, 0.9369 | 0.5634 |
| `C16-in` (stage 2 verification) | S; B | 256; 249 | 253; 231 | 0.3389; 0.4178 | | 0.3610; 0.5979 |
| `C-dense` (stage 2 verification) | S; B | 254; 246 | 251; 244 | 0.4960; 0.4432 | | 0.5293; 0.5183 |

- In both networks of `C46-one` at least 247 of the 256 cells vary between the items by more than 0.05. No cell has a mean magnitude below 0.05; the cells at or below 0.10 (6 under the sweep, 26 under backpropagation) have mean magnitudes of 0.64 to 0.88 and 0.54 to 0.97: they hold nearly the same value, far from zero, for every item at this position, as in the other networks.
- Under the sweep the mean standard deviation of `C46-one` (0.409) lies between that of `C16-in` (0.339) and that of `C-dense` (0.496); under backpropagation the three are close (0.420, 0.418, 0.443).
- By the three groups of cells of `C16-in` (input for 0 to 2, 3 or 4, 5 or more digits at 16 values a cell) the means are 0.401, 0.412 and 0.418 under the sweep and 0.416, 0.421 and 0.425 under backpropagation; in `C46-one`, where every cell has input for every digit, the grouping has no meaning, and the figures show no difference between the groups.

### (b) The answers given (part 5b)

R's 2,000 reduction items (untaught pairs of add and sub whose answer needs the reduction by 89) are rebuilt from the seed; the counts obtained on them equal the receipts' last rows. An answer is the bytes generated after `=`. The panel behind A is given beside them, since reduction items do not exist for max and min.

| Network | Rule | Items | First byte right | Second byte right | Whole answer right | Different answers given (the set asks for) |
|---|---|---|---|---|---|---|
| `C46-one` | S | R's items, add and sub (2,000) | 90.0% | 93.5% | 85.8% | 90 (89) |
| `C46-one` | S | add and sub, 12 letters (600) | 88.2% | 92.3% | 83.2% | 89 (89) |
| `C46-one` | S | max and min, 12 letters (600) | 99.8% | 98.2% | 98.2% | 89 (89) |
| `C46-one` | B | R's items, add and sub (2,000) | 96.5% | 98.5% | 95.2% | 89 (89) |
| `C46-one` | B | add and sub, 12 letters (600) | 95.8% | 98.0% | 94.7% | 89 (89) |
| `C46-one` | B | max and min, 12 letters (600) | 100.0% | 98.3% | 98.3% | 89 (89) |
| `C16-in` (stage 2 verification) | S; B | R's items (2,000) | 72.0%; 82.9% | 55.0%; 90.9% | 44.8%; 77.7% | 90; 89 (89) |
| `C-dense` (stage 2 verification) | S; B | R's items (2,000) | 95.5%; 98.5% | 94.0%; 97.8% | 93.7%; 97.5% | 89; 89 (89) |

- Under the sweep, on R's items, both bytes are right in 1,715 items, the first alone in 84, the second alone in 154 and neither in 47; under backpropagation 1,903, 26, 66 and 5. In both networks of `C46-one` the second byte is right more often than the first, where in `C16-in` under the sweep the first was right far more often (72.0% against 55.0%).
- Every one of the 2,000 answers is two digits and the end symbol in both networks; the most frequent answer, `81`, is given for 44 items (2.2%) under the sweep and 35 (1.8%) under backpropagation. Neither network answers with a few values.

## 6. What E50 shows and does not show

On seed 73129, a network of 256 cells fed a fixed code in which each of the 46 used bytes turns on 11 values of its own learned the first family (80 on R) under both rules when every cell read every value, in about the same number of updates as the same network with a learned code (6,912 under backpropagation and 12,800 under the sweep). When each cell read 16 of the 512 values, so that it had input for 1 to 10 of the 20 bytes the task must tell apart, neither rule brought R to 80 within 49,152 updates: under backpropagation it ended at 79.4 with two single rows above 80, under the sweep at 43.6; both rules learned max and min, whose answer is one of the operands. Under the sweep this is far more than the 1.4 of the same 16-value lists on a learned code, so the explanation that the sweep failed in E49 only because its learned code shrank is refuted as a sufficient one. When each cell read one value of every used byte instead, both rules reached 80 on R, at 15,360 updates under backpropagation and 34,304 under the sweep, which is 2.22 and 2.68 times the network that reads every value, and both ended far lower than that network on the length test at 40 letters (54.3 and 27.8 against 92.7 and 90.6). In all limited networks almost every cell varies between items at the point where the answer is read, and the networks give as many different answers as the items ask for. E50 therefore shows, for one seed, that cells reading only part of a fixed code can learn the task under either rule when each cell has some input for every byte, and that they do not within this dose when most cells lack input for most needed bytes, although the information reaches all cells within one or two positions through their neighbours; and that one value a byte still costs about two to three times the updates. It does not show how these figures vary between seeds, whether `C16-in` would reach the threshold with more updates (under backpropagation it ended within a point of it, rising by 2.6 points over the last 2,048 updates, below the brief's mark of 5 for *still rising*), which of the differences between `C16-in` and `C46-one` is responsible (the number of bytes a cell has input for, or the number of values a cell reads), why the length test at 40 letters falls so much more than A and R, whether cells that lack a byte use what their neighbours carry, or anything about a limited readout, a learned sparse code or more cells.

## What could not be checked

- **The scored rows other than those at updates 24,576 and 49,152.** Only three states of each run are saved. Every figure of updates to a threshold, the first two rows of each level and the rise of R are the receipts' record. What supports them from outside: the shares equal the counts in every row, the batch digest is right at every row, the logs hold 193 rows in order without a repeat, and the interrupted runs agree with the completed ones on their common rows.
- **That each run was trained under the rule it names**, and the cosines between the gradients of the two rules: no gradient was computed here.
- **The checks inside the runs at the rows between the saved states**, and the checks before the runs, which execute torch (P50.1).
- **The reboot itself** (the uptime at 23:59:35Z) and the processor times, which are the receipts', the logs' and the check receipts' record.
- **The receipts' digests of tensors** (`tensors_sha256`, `table_sha256`), whose formula is the runner's; the files' SHA-256 and the tensors themselves were compared instead.

## How to repeat the checks, and the running time

From the repository root:

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -I PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/e50_stage3_verify.py

With one or more of the names `0 1 2a 2b 2c 2d 3 4 5a 5b` as arguments, only those parts are printed. With the environment variable `E50S3V_SCRATCH` set to a folder, each weight file is written into it, read and deleted before the next; without it the script writes no file. Importing the stage 2 script runs that script's part 1 on its own commit with its output discarded (about two seconds).

Running times on one core of the reviewer's container (18 November 2026), weight files read into memory: part 0, 2 seconds; part 1, under 1 second; part 2a, 210 seconds (49 to 55 seconds for each of the four weight files); parts 2b, 2c and 2d, 1 second or less each; part 3, 20 seconds (the rebuilt stream of 49,152 batches); part 4, under 1 second; part 5a, 4 seconds; part 5b, 94 seconds. Parts 2a, 5a and 5b ran together in 309 seconds; parts 2b, 2c, 2d, 3 and 4 together in 22 seconds; parts 0 and 1 together in 3 seconds.
