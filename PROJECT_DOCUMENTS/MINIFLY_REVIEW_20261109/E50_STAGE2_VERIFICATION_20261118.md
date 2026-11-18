# Verification of the stage-2 report of E50 (the fixed input code with each cell reading 16 of its 512 values, seed 73129), 18 November 2026

## Conclusion

**Confirmed.** Every figure of the result table of the worker's report on stage 2 of the job E50 agrees with what I recomputed from the committed records of the two runs: 88 of 88 figures in its eight columns, among them all 44 figures of the setting `C16-in` (a network of 256 cells in which each cell reads 16 of the 512 values of a fixed input code), read within the first 24,576 updates and at the end of 49,152 updates under each of the two learning rules. The levels of the score on the taught pairs, which the report's text does not give, are 85.28 and 96.33 under backpropagation and 54.61 and 74.83 under the sweep, as the worker's tables file has them. The five predictions are scored correctly, and the brief's condition for calling the explanation under test refuted as a sufficient one is met. The saved networks give the recorded scores: the reviewer's forward pass reproduces every count of right answers of the scored rows at updates 24,576 and 49,152 of both runs, and 2,400 of the 2,400 answers that the runs recorded. The code was fixed in both runs as far as the files show: the code table, its list of owners, its value, the byte table and the layer `l1` are bit-equal in the start file, the file at update 24,576 and the final file. The masks hold in all six files: every cell reads exactly 16 values, the lists are those of the earlier job E49, and no weight outside a mask differs from zero.

**Differences.** Two figures of the report's text differ from the weight files in the third decimal, because a figure already rounded to four decimals was rounded again: the mean magnitude of the recurrent weights of the second group of cells at the start is 0.151 (0.151484), where the report has 0.152, and the mean magnitude of a kept weight of the input map at the start is 0.176 (0.176460), where the report has 0.177. One sentence is looser than its figures: max and min "stay at 96 to 98" holds for the levels at the end (96.00 to 97.89), while the levels within the first 24,576 updates are 94.89 to 95.56 and single scored rows after update 24,576 lie between 89.67 and 99.33. No other figure differs, and no sentence of the report claims more than its figures support.

**Not checkable without training,** and therefore still the worker's record: the scores at the scored rows other than those at updates 24,576 and 49,152, on which every figure of updates to a threshold rests; the checks that the runner made inside the runs; the cosines between the gradients of the two rules; that each run was trained under the rule it names; and the costs and times.

## Terms, material and method

**Method.** Read-only. Nothing was trained, torch is not installed, and no code of another branch was executed. One script beside this note, `e50_stage2_verify.py`, prints every figure given here; *part N* below names the part of its output that gives a figure. Receipts, answers files and logs are read with `git show` into memory. Each weight file was written with `git show` into a scratch folder outside the repository, read, and deleted before the next one was written. The last section gives the command and the running time.

**Material.** The worker's branch at c6c24d9c, which is the head of that branch as fetched: the stage report `E50_STAGE2_REPORT_73129.md`, the tables `E50_TABLES_73129_stage2.md`, the summary `E50_SUMMARY_73129_stage2.json`, and, under `MOD97_LOCAL_RULE_20261106/exp45/e50/`, for each of the two runs the receipt, the start file, the file at update 24,576, the final file, two answers files and the log. The brief (the written specification of the job) is the file whose name contains `E50_AN_INPUT_CODE_IN_WHICH_A_VALUE_STANDS_FOR_ONE_BYTE`, revision 2; it is the same file at c6c24d9c as at d6606454. The runner `exp50_code.py` and its specification file `e50_sparse_code.py` are the same files at c6c24d9c as at 0f3ec4bc, the commit that the report names; the receipts name the commit f54c6db8 with no modified source, and the SHA-256 that they record for the two files equal those computed here (part 0). The files of stage 1 and the four reference receipts (those of `in16` and of `F10`, defined below) are unchanged since b18e563b, the commit that the verification of stage 1 read, and the reference receipts are byte-equal to the originals on the branches of E49 and of E42 (part 0).

**Terms.**

- The *code* of a byte is the 512 values that feed the cells and the readout. In this job it is fixed: each of the 46 *used bytes* (the bytes that occur as input in the task) turns on 11 values of its own, each equal to one number, the *value of a turned-on entry* (3.2387 for this seed), and every other value is zero. The *code table* is the array of 258 rows of 512 values that holds the code, and its *owners* are the list that gives, for each value, the byte that turns it on. The *byte table* and the layer `l1` produced the learned code of the earlier jobs and take no part here. The 20 *needed bytes* are those that the task must tell apart (ten digits, four operation signs, `=`, four task codes and the start symbol).
- `C-dense`, the setting of stage 1, is a network of 256 cells in which each cell reads 10 fixed neighbouring cells and its own previous value (through the *recurrent map*), every cell reads all 512 values of the code (through the *input map*), and every unit of the readout reads all 256 cells (through the map `lm`). `C16-in`, the setting of stage 2, is the same network with each cell reading 16 of the 512 values; a *mask* is the array that marks the entries of a map that are kept. `in16` is the earlier job E49's network with the same lists of 16 values on a learned code, and `F10` is the earlier job E42's network with a learned code and every cell reading every value.
- The two *rules* are the *sweep* (S), the local rule of this programme, and *backpropagation* (B), the exact gradient.
- *A* is the share of whole answers right on 300 untaught pairs each of add and sub with 12 letters between the operands; the same panel holds 300 pairs each of max and min. *R* is the share right on 1,000 untaught pairs each of add and sub, without letters, whose answer needs the reduction by the modulus 89. The *length test* is 300 untaught pairs of add at 12 and at 40 letters. The *taught panel* (entry `taught` of the receipts) is the share right on 300 taught pairs each of add and sub at 12 letters.
- A run is scored every 256 updates; each scoring is a *scored row*, and a run of 49,152 updates has 193 rows. A threshold is *reached* at the first of three consecutive scored rows at or above it (the *three-row version*); the *single-row version* takes the first row at or above it. A *level* is the mean over the last three scored rows. Each run is given in two *readings*: within its first 24,576 updates, which is the number of updates of `C-dense`, and at its end.
- A *receipt* is the JSON file that a run writes, with one record for each scored row. The *start file*, the *file at update 24,576* and the *final file* of a run hold its tensors before the first update, after update 24,576 and after the last.
- The *reader* is the reviewer's numpy implementation of the network (`e50_code_reader.py` with `e49_interface_reader.py` and `e42_cells_reader.py`), written from the brief's text. It draws the code, the 16 values of each cell and the 10 neighbours of each cell from the seed, and takes only the trained weights and the value of a turned-on entry from a weight file.

## 1. The result table, from the receipts alone: holds (part 1)

In both receipts of `C16-in` the shares recorded in every row equal the shares computed from the row's counts of right answers; every figure below is computed from the counts. Each run is complete, with 193 scored rows and 49,152 optimizer steps. Thresholds are in updates, three-row version, with the single-row version in brackets.

| `C16-in` | B, first 24,576 | B, end | S, first 24,576 | S, end |
|---|---|---|---|---|
| Updates to 80 on A | not reached [not reached] | 40,704 [28,928] | not reached [not reached] | not reached [not reached] |
| Updates to 80 on R | not reached [not reached] | not reached [47,616] | not reached [not reached] | not reached [not reached] |
| Updates to 50 on R | 18,176 [16,640] | 18,176 [16,640] | not reached [not reached] | not reached [not reached] |
| Level of A | 67.94 | 83.78 | 32.72 | 49.50 |
| Level of R | 62.12 | 79.42 | 28.97 | 43.63 |
| Level of the length test at 40 letters | 41.56 | 56.22 | 9.56 | 13.56 |
| R: last three rows minus the three ending 2,048 updates earlier | +2.73 | +2.60 | +1.65 | +1.32 |
| max, min at 12 letters: updates to 80 | 5,376 [5,376]; 5,632 [5,632] | the same | 7,424 [7,424]; 7,680 [7,168] | the same |
| add, sub at 12 letters: updates to 80 | not reached; not reached | 34,560 [25,856]; 40,704 [33,280] | not reached; not reached | not reached; not reached |
| Level of the taught panel (not in the report's table) | 85.28 | 96.33 | 54.61 | 74.83 |
| Levels of max, min, add, sub (not in the report's table) | 95.22, 95.11, 72.78, 63.11 | 97.89, 96.00, 84.33, 83.22 | 95.56, 94.89, 36.44, 29.00 | 96.33, 96.22, 49.44, 49.56 |

**Every figure of the first nine rows equals the report's figure,** to the report's one decimal: 44 of 44. The four columns of `C-dense` and of `in16` in the report's table, recomputed in the same way from their receipts, agree as well, 44 of 44. The rows of a level are updates 24,064 to 24,576 for the first reading and 48,640 to 49,152 for the end; the rows of the comparison are updates 22,016 to 22,528 and 46,592 to 47,104. (The note on stage 1 names updates 21,504 to 22,016 for the comparison; that is a slip of its text, and its figures were computed on the rows 22,016 to 22,528.)

**The figures of the report's text (part 1).**

- *Backpropagation, R at the end.* The last three rows are 80.10, 80.45 and 77.70; the highest row is 80.75 at update 47,616. Three rows are at or above 80 (updates 47,616, 48,640 and 48,896), and never three in succession, so the three-row version does not record the threshold and the single-row version records it at 47,616, as reported.
- *Ratios.* 80 on A at 40,704 is 5.30 times the 7,680 of `C-dense`; the last recordable update, 48,640, is 7.04 times the 6,912 of `C-dense` to 80 on R; `in16` against `F10` is 2.16 on R and 1.91 on A. All four are as reported.
- *Descriptions.* At the end under backpropagation `C16-in` minus `C-dense` is −13.89 on A (*about the same* by the brief's mark of 15 points), −17.67 on R and −36.44 on the length test at 40 letters (*lower*); under the sweep −44.00, −50.47 and −77.00 (*lower*). Updates are *more* and multiply-adds *rise* by both counts in every cell at the end, as reported.
- *max and min.* The report states that they "stay at 96 to 98". The levels at the end are 97.89 and 96.00 under backpropagation and 96.33 and 96.22 under the sweep. The levels within the first 24,576 updates are 95.22 and 95.11, and 95.56 and 94.89. Single rows after update 24,576 lie between 93.67 and 99.33 under backpropagation and between 89.67 and 98.67 under the sweep. No row falls below 80 after the threshold. The sentence holds for the levels at the end.
- *The answer's two bytes on R's items.* The levels at the end are 84.23 and 92.58 under backpropagation and 71.45 and 53.98 under the sweep, as reported. Under backpropagation the first byte is right more often in 73 of the 80 rows up to update 20,480 and the second in all 81 rows from update 28,672 on; of the 31 rows between, the first is ahead in 17, the second in 13, and the two are equal in one. The report's "until about 20,480" and "from about 28,672 on" describe this correctly. Under the sweep the first byte is right more often in 187 of 192 rows.
- *Against `in16` under the sweep.* The levels of `in16` are 1.37 on R and 38.78 and 40.56 on max and min; those of `C16-in` at the end are 43.63, 96.33 and 96.22, as reported.
- *The runs.* The receipts record 1.44 + 0.32 = 1.76 CPU-hours under backpropagation and 1.68 + 0.32 = 2.00 under the sweep. The largest interval between two successive scored rows, by the receipts' own clock, is 36.2 seconds under backpropagation and 41.2 under the sweep, over the whole of each run, and each log holds 193 rows in order without a repeat (part 0). The receipts record the code table, the byte table and `l1` as unchanged, every mask as held and E42's limit as held in 193 of 193 rows of both runs.

**The predictions and the rule for stage 3.** P50.4' and P50.5' (80 on R within 49,152 updates under backpropagation, under the sweep) are false; P50.6' (under backpropagation within 3.24 times the updates of `C-dense`, which is update 22,394.9) is false; P50.7' and P50.8' (max and min reach 80 under backpropagation, under the sweep) are true. These are the worker's outcomes. `C16-in` under the sweep does not reach 80 on R within its number of updates and `C-dense` under the sweep does, at 12,800 updates; this is the brief's condition for *refuted as a sufficient one*. Since `C16-in` does not reach 80 on R under either rule, the brief prescribes stage 3 under both rules, as the report states.

**The tables file and the summary file.** Of the rows of `C16-in` in the tables file, 62 were written again here in the tables' own formats from the recomputed figures, and each is found in the file character for character (part 1); the eight rows of its two tables of sizes, written from the weight files, are found as well (part 2d). In the summary file 312 of 312 thresholds, levels and changes of R of the ten readings equal the recomputed figures.

## 2. From the weights

### (a) The recorded answers and counts are those of the saved weights: holds (part 2a)

The forward pass is the reader's, with the reader's own draws; the reader's three masks equal the three masks stored in each file. A second form, built from the tensors of the file alone (the file's code table and the file's masks as full matrices), differs from it by at most 7.6e-06 in a logit, where the largest logit is 27.0 to 31.7, and gives the same largest output at every position.

| Counts at the scored row | S, final file, reader | Receipt, row 49,152 | S, file at 24,576 | Receipt, row 24,576 | B, final file | Receipt, row 49,152 | B, file at 24,576 | Receipt, row 24,576 |
|---|---|---|---|---|---|---|---|---|
| The panel behind A: add, sub, max, min (300 each) | 139, 153, 293, 290 | the same | 113, 83, 287, 282 | the same | 252, 253, 295, 288 | the same | 204, 178, 285, 288 | the same |
| R: add, sub (1,000 each) | 547, 349 | the same | 364, 196 | the same | 837, 717 | the same | 681, 500 | the same |
| The taught panel: add, sub, max, min (300 each) | 218, 224, 299, 295 | the same | 175, 151, 293, 288 | the same | 284, 288, 298, 298 | the same | 249, 238, 296, 298 | the same |
| R's items, first byte right: add, sub | 792, 648 | the same | 641, 557 | the same | 871, 787 | the same | 803, 673 | the same |
| R's items, second byte right: add, sub | 616, 484 | the same | 464, 332 | the same | 949, 869 | the same | 839, 685 | the same |
| Length test at 12 and at 40 letters (300 each) | 139, 29 | the same | 113, 22 | the same | 252, 170 | the same | 204, 122 | the same |

**The answers files.** Each of the four files holds 300 recorded answers at 12 letters and 300 at 40. For every item the reader generated the answer byte by byte (the largest output is appended and fed back, until the end symbol or four bytes). **The generated answer equals the recorded answer in 600 of 600 items for each of the four weight files, 2,400 of 2,400 in all.** The SHA-256 of each of the four weight files equals the one that the receipt records.

### (b) The code was fixed: holds between the three saved states of each run (part 2b)

- **Unchanged tensors.** In both runs `code_table`, `code_owner`, `code_amplitude`, `book.weight` (the byte table), `l1.weight` and `l1.bias` are bit-equal in the start file, the file at update 24,576 and the final file. The three masks and the other structural tensors are bit-equal as well. The ten trained tensors differ between each two successive files in every kept entry (12,288 of the input map, 8,448 of the recurrent map, all entries of the others).
- **The code.** The reader's digest of the code is `12111044f0f8bc70`, as the receipts record. The file's list of owners equals the reader's; the file's code table equals the reader's table bit for bit: 506 entries are not zero, 11 in each of 46 rows, all equal to 3.238702; no value has two owners and six have none.
- **The start.** The two start files hold the same tensors. Against the start file of `C-dense`, every tensor is bit-equal except the input map, its mask and the number of values a cell reads; the input map equals that of `C-dense` times the mask times the square root of 32 with no difference in any bit. `C16-in` therefore begins from the network of `C-dense` with the same code, as the report's "5.66 times larger" states.
- **SHA-256.** The final files have `143520ef71f5a4747645deaaedb174f72aca67d9175f820fecac20772dcae984` (the sweep) and `f9b170926b45326d3d0ed413b1a8ca18f04ca5239cf4fed5afd6a63db99aec4e` (backpropagation), as the receipts and the report's first 16 digits state.
- **The stream of training items (part 4).** The running SHA-256 over the batches, rebuilt here from the seed without the runner, equals the recorded one at 193 of 193 rows of each run, and the first 97 rows equal those of `C-dense` under the same rule; the optimizer record is that of `C-dense`. The two settings were therefore fed the same items in the same order.

### (c) The masks hold, and each cell reads exactly 16 values: holds (part 2c)

In each of the six files the mask of the input map has exactly 16 kept values in each of the 256 rows, equals the reader's draw of E49's lists (digest `0e09b9ff0545e68d`, as the receipts record), and the three gates of a cell have weights on the same values. Of the 380,928 entries of the input map outside the mask and of the 188,160 entries of the recurrent map outside its mask, none differs from zero in any of the six files. The mask of `lm` is full, so no weight lies outside it. No kept weight is exactly zero. From the final file's mask and owners, a cell has input for 1 to 10 of the 20 needed bytes (median 6) and for 0 to 7 of the ten digits (median 3), and a needed byte reaches 67 to 87 cells, as the report and the brief state. Of the 5,120 pairs of a cell and a needed byte, the cell reads no value of the byte in 3,603, one value in 1,299, two in 203 and three in 15. The two counts of multiply-adds a position, recounted from the masks, are 545,536 and 277,000.46, as the brief and the receipts state.

### (d) "By what a cell knows" and "other measures": hold, with two differences in the third decimal (part 2d)

All figures are recomputed from the weight files in double precision and equal the receipts' rows 0, 24,576 and 49,152 to the precision of the receipts. The three groups are the cells with input for 0 to 2, for 3 or 4 and for 5 or more of the ten digits: 95, 131 and 30 cells, as reported.

| Mean magnitude of the kept recurrent weights from neighbours, by group | Start | Update 24,576 | End | Reported at the end |
|---|---|---|---|---|
| `C16-in`, B | 0.1498, 0.1515, 0.1501 | 0.2556, 0.2328, 0.2268 | 0.2652, 0.2403, 0.2328 | 0.265, 0.240, 0.233 |
| `C16-in`, S | the same | 0.2130, 0.2087, 0.2025 | 0.2260, 0.2191, 0.2132 | 0.226, 0.219, 0.213 |
| `C-dense`, B | the same | | 0.1467, 0.1440, 0.1448 | 0.147, 0.144, 0.145 |
| `C-dense`, S | the same | | 0.1501, 0.1518, 0.1491 | 0.150, 0.152, 0.149 |

| Other measures | Start | Update 24,576 | End | Reported |
|---|---|---|---|---|
| A cell's input for a needed byte, root mean square over all cells, gates and needed bytes, without the bias: `C16-in`, B | 0.3814 | 0.7416 | 0.7650 | 0.381; 0.765 |
| The same: `C16-in`, S | 0.3814 | 0.5867 | 0.6290 | 0.381; 0.629 |
| The same: `C-dense`, B and S | 0.389 | | 5.140 and 4.769 | 5.14 and 4.77 |
| Mean magnitude of a kept weight of the input map: `C16-in`, B | 0.1765 | 0.1359 | 0.1398 | 0.177; 0.140 |
| The same: `C16-in`, S | 0.1765 | 0.1107 | 0.1165 | 0.177; 0.117 |

- Of the 25 figures of the two paragraphs that can be recomputed from weights, 23 equal the figure from the weight file rounded to the report's decimals. **The two that differ:** the second group at the start is 0.151484, which is 0.151 and not 0.152; a kept weight of the input map at the start is 0.176460, which is 0.176 and not 0.177. In both the report rounds the four-decimal figure of the tables file (0.1515, 0.1765) a second time.
- The first group minus the third is −0.0003 at the start in both runs, and at the end +0.0325 under backpropagation and +0.0128 under the sweep (reported 0.03 and 0.01).
- **A remark on the definition.** The runner's root mean square is taken over all 5,120 pairs of a cell and a needed byte, and in 70.4% of them the input is zero by the mask. Over the 29.6% of pairs for which the cell reads a value of the byte it is 0.7007 at the start and 1.4054 (backpropagation) and 1.1555 (the sweep) at the end. The comparison with the 5.14 and 4.77 of `C-dense` should be read with this in mind.
- The cosines between the gradients of the two rules are given from the receipts only and are as reported (at the end, input map 0.104 and 0.348, recurrent map 0.293 and 0.431, `lm` 0.664 and 0.669); they were not recomputed.

## 3. Two measurements that the report does not contain

Both are made on the final weights of the two runs of `C16-in` and, for comparison, of the two runs of `C-dense`, with the forward pass of section 2(a).

### (a) The use of the cells (part 3a)

The items are the 300 items of the length-test panel at 12 letters (untaught pairs of add). For each cell the standard deviation of its state between the 300 items is taken at the position of `=`, where the answer's first byte is read. The magnitude of the state is its absolute value, averaged over items and cells.

| Network | Rule | Cells above 0.05 | Cells above 0.10 | Mean over the 256 cells | Least, median, largest | Mean magnitude of the state |
|---|---|---|---|---|---|---|
| `C16-in` | S | 256 | 253 | 0.3389 | 0.0783, 0.3448, 0.7648 | 0.3610 |
| `C16-in` | B | 249 | 231 | 0.4178 | 0.0147, 0.4483, 0.8533 | 0.5979 |
| `C-dense` | S | 254 | 251 | 0.4960 | 0.0152, 0.5001, 0.8522 | 0.5293 |
| `C-dense` | B | 246 | 244 | 0.4432 | 0.0000, 0.4629, 0.8651 | 0.5183 |

- In all four networks at least 246 of the 256 cells vary between the items by more than 0.05. In `C16-in` under the sweep every cell does; its mean standard deviation (0.339) and its mean magnitude (0.361) are the lowest of the four.
- The cells that vary least are not silent. No cell of any of the four networks has a mean magnitude below 0.05. The cells at or below 0.10 (3, 25, 5 and 12 cells in the order of the table) have mean magnitudes of 0.23 to 0.67, 0.60 to 0.97, 0.90 to 0.98 and 0.89 to 1.00: they hold nearly the same value, far from zero, for every item at this position.
- By the three groups of section 2(d), the mean standard deviation in `C16-in` is 0.332, 0.340 and 0.357 under the sweep and 0.431, 0.402 and 0.445 under backpropagation. The cells with input for the fewest digits vary about as much as the others.

### (b) The answers given (part 3b)

Untaught reduction items are available for add and sub: they are R's 2,000 items, rebuilt from the seed, and the counts obtained on them equal the receipts' last rows. Reduction items do not exist for max and min, whose answer needs no reduction; for these the panel behind A is used (untaught pairs at 12 letters), and the same panel is given for add and sub beside R. An answer is the bytes generated after `=`. The figures are those of the final weights, which is the last scored row and not a level.

| Network | Rule | Items | First byte right | Second byte right | Whole answer right | Different answers given | Different right answers in the set |
|---|---|---|---|---|---|---|---|
| `C16-in` | S | add and sub, R (2,000) | 72.0% | 55.0% | 44.8% | 90 | 89 |
| `C16-in` | S | add and sub, 12 letters (600) | 73.2% | 59.0% | 48.7% | 86 | 89 |
| `C16-in` | S | max and min, 12 letters (600) | 99.7% | 97.2% | 97.2% | 89 | 89 |
| `C16-in` | B | add and sub, R (2,000) | 82.9% | 90.9% | 77.7% | 89 | 89 |
| `C16-in` | B | add and sub, 12 letters (600) | 86.8% | 95.3% | 84.2% | 89 | 89 |
| `C16-in` | B | max and min, 12 letters (600) | 99.8% | 97.2% | 97.2% | 89 | 89 |
| `C-dense` | S | add and sub, R (2,000) | 95.5% | 94.0% | 93.7% | 89 | 89 |
| `C-dense` | S | add and sub, 12 letters (600) | 94.2% | 92.3% | 92.3% | 89 | 89 |
| `C-dense` | S | max and min, 12 letters (600) | 99.5% | 97.8% | 97.8% | 89 | 89 |
| `C-dense` | B | add and sub, R (2,000) | 98.5% | 97.8% | 97.5% | 89 | 89 |
| `C-dense` | B | add and sub, 12 letters (600) | 99.2% | 98.7% | 98.7% | 89 | 89 |
| `C-dense` | B | max and min, 12 letters (600) | 99.8% | 97.8% | 97.7% | 89 | 89 |

- **Under the sweep in `C16-in`, on R's items,** the first byte is right in 1,440 of 2,000 items and the second in 1,100; both are right in 896, the first alone in 544, the second alone in 204 and neither in 356. The network gives 90 different answers where the set asks for 89, so it does not answer with a few values: the most frequent answer, `58`, is given for 71 items (3.6%). All but one of the 2,000 answers are two digits and the end symbol.
- On max and min the same network gives the whole answer right in 583 of 600 items and 89 different answers, as `C16-in` under backpropagation and `C-dense` do.
- Under backpropagation in `C16-in` the second byte is right more often than the first (1,818 against 1,658 on R), which is the order that the report describes for the end of that run; under the sweep it is the reverse.

## 4. What stage 2 shows and does not show

On seed 73129, the network in which each cell reads 16 values of the fixed code learned max and min under both rules: both reach 80 within 5,632 updates under backpropagation and within 7,680 under the sweep, and both end at 96 to 98. Under neither rule did it reach 80 on R within 49,152 updates in the three-row version, where the network reading all 512 values reached it at 6,912 updates under backpropagation and at 12,800 under the sweep. Under backpropagation it ended close to the threshold: R stood at 79.4 over the last three rows, two successive rows were above 80, R had risen by 2.6 points over the last 2,048 updates, and A passed 80 at update 40,704, which is 5.30 times the figure of the network reading all values. Under the sweep it ended at 43.6 on R and 49.5 on A, with R rising by 1.3 points over the last 2,048 updates; this is far above the 1.4 on R of the earlier network with the same lists and a learned code, and far below the 94.1 of the network reading all values, so the brief's condition for refuting, as a sufficient explanation, that the sweep fails with a limited input because the learned code shrinks, is met. The measurements of section 3 describe the two networks at the end: almost every cell varies between items in both, the cells with input for the fewest digits about as much as the others, and under the sweep the network gives as many different answers as the items ask for, with the second byte wrong more often than the first. The stage does not show whether more updates would bring either run to the threshold; it does not show which of the two things that differ from the network reading all values is responsible for the larger number of updates, the number of bytes a cell has input for or the number of weights behind a cell's input for a byte, which is the question of stage 3; it says nothing about a limited readout; and with one seed it does not show how much of any difference a seed alone would change, where the verification of stage 1 found differences of up to 2,304 updates and 7 points between three seeds of one network.

## What could not be checked

- **The scored rows other than those at updates 24,576 and 49,152.** Only three states of each run are saved. Every figure of updates to a threshold, the first two rows of each level and the change of R are the receipts' record. What supports them from outside is limited to this: the rows at 24,576 and 49,152 are reproduced exactly from the weights, the shares equal the counts in every row, the digest of the batches is right at every row, and each log holds 193 rows in order without a repeat.
- **That each run was trained under the rule it names.** The receipts record the rule and the two final networks differ; no gradient was computed here. The cosines between the gradients of the two rules are the receipts' record for the same reason.
- **The checks inside the runs at the rows between the saved states** (the code table's SHA-256, the byte table and `l1`, the weights outside the masks, the optimizer's averages there, E42's limit) and the statement that the optimizer holds no state for the byte table or `l1`. The three saved states of each run agree with them.
- **That neither run was restarted or paused.** The receipts' intervals between scored rows (at most 41.2 seconds) and the logs agree with it; the statement itself rests on the worker's clock and the queue log.
- **Costs and times:** the CPU-hours, the start and end times and the exit codes are those of the receipts and of the queue log.
- **Reduction items for max and min,** which do not exist; section 3(b) uses the panel behind A for these two operations.
- **The receipts' digests of tensors** (`tensors_sha256`, `table_sha256`), whose formula is the runner's; the files' SHA-256 and the tensors themselves were compared instead.
- **Stage 3,** which had started when the report was committed and of which no receipt is read here.

## How to repeat the checks, and the running time

From the repository root:

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -I PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/e50_stage2_verify.py

With one or more of the names `0 1 2a 2b 2c 2d 3a 3b 4` as arguments, only those parts are printed. With the environment variable `E50S2V_SCRATCH` set to a folder, each weight file is written into that folder, read and deleted before the next; without it the weight files are read into memory and the script writes no file. The two ways give the same output. The script writes no bytecode.

RUNNING_TIME_PARAGRAPH
