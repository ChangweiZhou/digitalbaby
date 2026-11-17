# Verification of the stage-1 report of E50 (the fixed input code with the dense interface, seed 73129), 17 November 2026

## Conclusion

Every figure of the worker's report on stage 1 of the job E50 (a network of 256 cells fed by a fixed input code in which each byte turns on 11 values of its own, trained on seed 73129 under two learning rules), and every row of its tables for the runs concerned, agrees with what I recomputed from the committed records of the runs (the receipts, which hold one record for each scoring of the network, every 256 updates) and from the weight files; I found no disagreement. The saved final networks give the recorded scores: the reviewer's forward pass, written in numpy from the text of the brief (the written specification of the job), reproduces every count of right answers of the last scoring of both runs and all 1,200 answers that the runs recorded; the fixed code, and the byte table and the layer `l1` that produce the learned code in the comparison network and take no part here, are bit-equal between the start files and the final files; and the sequence of training batches, rebuilt here from the seed, has the recorded SHA-256 digest at every scoring of both runs and of the two comparison runs of the earlier job E42, whose network has a learned code. The four predictions registered for this stage are scored correctly, and the decision to run stage 2 (the same code with each cell reading 16 of its 512 values) under both rules is the one that revision 2 of the brief prescribes; what is on file of stage 2 at the commit read (the launch script, the runner's table of the number of updates of each setting, two log lines for each run) agrees with the brief, but no receipt of stage 2 exists yet, so its number of updates is not yet confirmed from a record of the run. No sentence of the report claims more than its figures support. One sentence says less than the record on file allows: the report states that the spread between the seeds of the comparison network was not read, and when it is read, the fixed code reaches the threshold of 80 on one of the two scores (R, defined below) earlier than each of the three seeds of the comparison network under both rules, by more than those seeds differ among themselves, while on the other score it lies inside their range; with one seed of the fixed code this remains a description and not a demonstrated gain. Not checkable without training, and therefore still the worker's record: the scores at the 96 scorings before the last, on which every figure of updates to a threshold rests; the checks that the runner made inside the runs at those scorings; the cosines between the gradients of the two rules; the checks before the runs that execute torch; and the costs and times.

## Terms, material and method

**Method.** Read-only. Nothing was trained, torch is not installed, and no code of another branch was executed: every file was read with `git show`, one at a time, into memory. One script beside this note, `e50_stage1_verify.py`, prints every figure given here. *Part N* below means the section of that script's output headed by the number N; the parts carry the numbers of the claims that were to be checked, with part 0 for the material and part 12 for further figures of the report. The last section gives the command and the running time.

**Material.** The worker's branch at b18e563b, which is the head of that branch as fetched: the stage report `E50_STAGE1_REPORT_73129.md`, the tables `E50_TABLES_73129_stage1.md`, the summary `E50_SUMMARY_73129_stage1.json`, the receipts, weight files, answers files and logs under `MOD97_LOCAL_RULE_20261106/exp45/e50/`, the checks before the runs under `exp45/e50_qual/`, and the runner `exp50_code.py` with `e50_sparse_code.py` and `e50_tables.py`. The brief is the file whose name contains `E50_AN_INPUT_CODE_IN_WHICH_A_VALUE_STANDS_FOR_ONE_BYTE`, revision 2, at d6606454; it is unchanged at b18e563b. The comparison runs are read from E42's own branch at 44f91edc, which is the head of that branch, and not from the worker's copies. E49's receipts and the starting network of E42 are read at a36de661.

**Terms.**

- The *code* of a byte is the 512 values that feed the cells and the readout. In E42 it is learned: it is `tanh(l1(byte table))`, where the *byte table* holds 8 numbers for each byte and `l1` is a linear layer. In E50 it is fixed: each of the 46 *used bytes* (the bytes that occur as input in the task) turns on 11 values of its own, each equal to one number, the *value of a turned-on entry* (3.2387 for this seed), and every other value is zero. The *code table* is the array of 258 rows of 512 values that holds the fixed code.
- `F10` is E42's network: 256 cells, each reading 10 fixed cells of its neighbourhood and its own previous value, every cell reading all 512 values of the code through the *input map*, and each of the 512 units of the readout reading all 256 cells through the map `lm`. `C-dense` is the same network with the fixed code in place of the learned one. `C16-in`, the setting of stage 2, is `C-dense` with each cell reading 16 of the 512 values.
- The two *rules* are the *sweep* (S), the local rule of this programme, in which the error crosses a linear map by the signs of its weights times one scale, and *backpropagation* (B), the exact gradient.
- The *first family* is the four operations add, sub, max and min under the modulus 89. *A* is the share of whole answers right on 300 untaught pairs each of add and sub with 12 letters between the operands; the same panel holds 300 pairs each of max and min. *R* is the share right on 1,000 untaught pairs each of add and sub, without letters, whose answer needs the reduction by the modulus. The *length test* is 300 untaught pairs of add at 12 and at 40 letters. The *taught panel* is 300 taught pairs of each operation at 12 letters.
- A run is scored every 256 updates; each scoring is a *scored row*, and a run of the *dose* of 24,576 updates has 97 rows (updates 0 to 24,576). A threshold is *reached* at the first of three consecutive scored rows at or above it (the *three-row version*); the *single-row version* takes the first row at or above it. A *level* is the mean over the last three scored rows.
- A *receipt* is the JSON file that a run writes, with one record for each scored row. The *start file* and the *final file* of a run hold its tensors before the first and after the last update.
- The *multiply-adds a position* are the products of the forward pass for one item at one position. The *first count* takes every weight of a map inside its mask; the *second count* takes only the products with a value of the code that is not zero.
- The brief's *descriptions* compare a candidate with a reference. For the updates to a threshold they are those of the earlier job E41: with the ratio of the candidate's figure to the reference's, *fewer* below two thirds, *more* above one and a half, otherwise *about the same*; the same bands in multiply-adds are named *falls*, *rises* and *stays*. For a level they are *lower* or *higher* at a difference of 15 points or more, otherwise *about the same*.
- The *reader* is the reviewer's `e50_code_reader.py`, a numpy implementation written from the brief's text, with `e42_cells_reader.py` and `e49_interface_reader.py`. Its forward pass never forms the 512 values of a byte: for every used byte and every cell it adds the cell's weights on the byte's 11 values and multiplies the sum by the value of a turned-on entry.

**The comparison receipts are E42's own (part 0).** The worker's copies `reference/e42_F10_B_73129.json` (94,914 bytes, SHA-256 `bb908b30…`) and `reference/e42_F10_S_73129.json` (94,957 bytes, `5f633470…`) are byte-equal to E42's receipts at 44f91edc, and the two copies of E49's receipts are byte-equal to the originals at a36de661. In all four receipts of `C-dense` and `F10` the shares recorded in every row equal the shares computed from the row's counts of right answers; every figure below is computed from the counts.

**The runner is the committed one (part 0).** Both receipts name the command `python3 exp50_code.py run C-dense <rule> 73129`, the commit 0f3ec4bc and no modified source; the SHA-256 they record for the runner and for the specification's copy equal those of the files at b18e563b, and the files are unchanged between 0f3ec4bc and b18e563b. The two files copied from E49's worker are identical to the originals at a36de661, and E42's two files are those of the default branch.

## 1. Updates to the 80 threshold: holds (part 1)

| Run | 80 on A, recomputed [single row] | Reported | 80 on R, recomputed [single row] | Reported |
|---|---|---|---|---|
| `C-dense`, B | 7,680 [6,912] | 7,680 | 6,912 [6,912] | 6,912 |
| `F10`, B | 8,704 [8,192] | 8,704 | 9,472 [8,960] | 9,472 |
| `C-dense`, S | 13,056 [13,056] | 13,056 | 12,800 [12,032] | 12,800 |
| `F10`, S | 13,312 [12,800] | 13,312 | 15,104 [13,824] | 15,104 |

| `C-dense` against `F10` | Ratio, three rows | Reported | Description | Ratio, single row | Description | Ratio in multiply-adds (first count) | Description |
|---|---|---|---|---|---|---|---|
| B, 80 on A | 0.8824 | 0.88 | about the same | 0.8438 | about the same | 0.8785 | stays |
| B, 80 on R | 0.7297 | 0.73 | about the same | 0.7714 | about the same | 0.7265 | stays |
| S, 80 on A | 0.9808 | 0.98 | about the same | 1.0200 | about the same | 0.9765 | stays |
| S, 80 on R | 0.8475 | 0.85 | about the same | 0.8704 | about the same | 0.8437 | stays |

The descriptions are those of the report in all eight cells of each version. *Fewer* on R would have needed the threshold at or before update 6,144 under backpropagation and 9,984 under the sweep, three and eleven scored rows earlier than recorded. The thresholds at 50 and at 90 of the tables file are reproduced as well (parts 1 and 12).

## 2. The levels: holds (part 2)

| Run | A, R, 40 letters, recomputed | Reported | R, last three rows minus the three ending 2,048 updates earlier | Reported |
|---|---|---|---|---|
| `C-dense`, B | 97.67, 97.08, 92.67 | 97.7, 97.1, 92.7 | +1.08 | +1.1 |
| `F10`, B | 93.22, 94.08, 92.56 | 93.2, 94.1, 92.6 | +2.07 | +2.1 |
| `C-dense`, S | 93.50, 94.10, 90.56 | 93.5, 94.1, 90.6 | +3.02 | +3.0 |
| `F10`, S | 91.22, 89.27, 89.11 | 91.2, 89.3, 89.1 | +0.68 | +0.7 |

The last three rows are updates 24,064, 24,320 and 24,576, and the three rows of the comparison are updates 21,504 to 22,016. `C-dense` minus `F10` is +4.44, +3.00 and +0.11 points under backpropagation and +2.28, +4.83 and +1.44 under the sweep, which is the report's "+0.1 to +4.8"; all six are *about the same* by the brief's form.

## 3. The four operations apart: holds (part 3)

Updates to 80 in the three-row version, with the single-row version in brackets, and the levels.

| Run | max | min | add | sub | Reported | Levels max, min, add, sub | Reported |
|---|---|---|---|---|---|---|---|
| `C-dense`, B | 1,024 [1,024] | 1,280 [1,280] | 7,680 [6,912] | 7,680 [6,912] | 1,024; 1,280; 7,680; 7,680 | 97.78, 96.89, 97.33, 98.00 | 97.8, 96.9, 97.3, 98.0 |
| `F10`, B | 2,048 [2,048] | 2,048 [2,048] | 8,704 [8,704] | 9,472 [7,936] | 2,048; 2,048; 8,704; 9,472 | 97.00, 97.44, 92.89, 93.56 | 97.0, 97.4, 92.9, 93.6 |
| `C-dense`, S | 1,280 [1,280] | 1,536 [1,536] | 12,032 [12,032] | 13,824 [13,824] | 1,280; 1,536; 12,032; 13,824 | 97.67, 95.67, 94.00, 93.00 | 97.7, 95.7, 94.0, 93.0 |
| `F10`, S | 3,072 [3,072] | 2,816 [2,816] | 11,776 [11,264] | 15,104 [13,824] | 3,072; 2,816; 11,776; 15,104 | 97.78, 97.78, 90.11, 92.33 | 97.8, 97.8, 90.1, 92.3 |

## 4. The answer's two bytes apart: holds (parts 4 and 6)

On R's 2,000 items the levels of the share with the first answer byte right and with the second right are 98.13 and 97.63 under backpropagation (reported 98.1 and 97.6) and 96.03 and 94.30 under the sweep (reported 96.0 and 94.3). The two early rows that the report quotes are 41.95 against 12.55 at update 2,048 under backpropagation (reported 42 against 13) and 60.10 against 25.95 at update 4,096 under the sweep (reported 60 against 26). The first byte is right more often than the second in 86 of the 96 rows after the start under backpropagation and in 94 under the sweep. The rows in which it is not are of two kinds: three rows among the first three of the runs (update 768 under backpropagation, updates 256 and 512 under the sweep), where both shares are near 10 and differ by at most 0.45 points, and nine rows from update 7,936 on under backpropagation, where both shares are above 88 and differ by at most 0.35 points. The report's sentence that the first byte is right more often early in both runs therefore holds for the rise, from update 1,024 on.

The counts of the last row are confirmed from the final weights (part 6): first byte right in 986 and 984 items of add and sub under backpropagation and 963 and 946 under the sweep, second byte right in 981 and 976 and in 953 and 926, as the receipts record. The second byte is counted as the runner counts it, under greedy generation after the byte that the network gave, also where that byte was wrong (30 items under backpropagation and 91 under the sweep).

## 5. The code table, the byte table and `l1`; the masks; the SHA-256 of the weight files; the digest of the code: holds (part 5)

- **SHA-256.** The final file of the run under the sweep has SHA-256 `056ca79d181f1ec2192ea653aca84f1ad9e739f0a6c90c0b06f24245f8ea78c4` and that of the run under backpropagation `af1315dd2cd048a103238f5fe5517bdae50d1ca397973c1bbe80ccafef3e386a`. Both equal the receipts' `weights_sha256` and the report. The SHA-256 of each start file equals the receipt's `start_sha256`.
- **Unchanged tensors.** In both runs `code_table`, `code_owner`, `code_amplitude`, `book.weight` (the byte table), `l1.weight` and `l1.bias` are bit-equal between the start file and the final file, as are the three masks and the other structural tensors. The ten trained tensors differ in every kept entry: the recurrent map in 8,448 of its 196,608 entries, which is the number inside its mask, and the other nine in all entries.
- **The code.** The reader's digest of the code for seed 73129 is `12111044f0f8bc70`, as reported. The file's `code_owner` equals the reader's list of owners and has that digest. The file's `code_table` equals, bit for bit, the reader's table built with the file's value of a turned-on entry: 506 entries are not zero, 11 in each of 46 rows, all equal to 3.238702, and no value has two owners. Computed here from the start file's byte table and `l1`, the mean squared length of the learned code over the 46 used bytes is 115.3811 and the value of a turned-on entry 3.238701 in double precision, against 3.238702 in the file and the receipt.
- **The masks and the weights outside them.** The masks of the input map and of `lm` are full (131,072 entries each), so no weight lies outside them. The recurrent mask equals the reader's draw of E42's ten connections with the cell itself, 11 cells for every cell. Of the 188,160 entries of the recurrent map outside the mask, none is different from zero in the final file or in the start file of either run.
- **The start is E42's.** The thirteen module tensors of each start file, the recurrent mask, the periods and the phases are bit-equal to E42's start of `F10` (the start file of the dry run of E49's check 2 at a36de661), and the two start files of E50 are equal to each other. `C-dense` and `F10` of seed 73129 therefore begin from the same network.
- **Inside the runs (the worker's record).** The receipts record the code table, the byte table and `l1` as unchanged, every mask as held and E42's limit as held in 97 of 97 rows of both runs, and the optimizer as holding state for the ten trained tensors and not for the byte table or `l1`. The comparison above confirms the first statement for the end of each run and not for the rows between.

## 6. The recorded scores are those of the saved weights: holds (part 6)

The forward pass is `e50_code_reader.forward`, applied to the final file of each run with the code built by the reader. Its two forms (by index lists, and by a table with masked matrices) differ on the trained weights by at most 3.8e-06 in a logit, where the largest logit is 25.9 under the sweep and 36.4 under backpropagation, and give the same largest output at every position.

| Counts of right answers at the last scored row | Reader, S | Receipt, S | Reader, B | Receipt, B |
|---|---|---|---|---|
| The panel behind A: add, sub, max, min (300 each) | 273, 281, 298, 289 | the same | 295, 297, 294, 292 | the same |
| R: add, sub (1,000 each) | 950, 923 | the same | 980, 970 | the same |
| The taught panel: add, sub, max, min (300 each) | 297, 297, 297, 298 | the same | 300, 298, 295, 298 | the same |
| R's items, first byte right: add, sub | 963, 946 | the same | 986, 984 | the same |
| R's items, second byte right: add, sub | 953, 926 | the same | 981, 976 | the same |
| Length test at 12 and at 40 letters (300 each) | 273, 263 | the same | 295, 284 | the same |

**The answers files.** Each file holds 300 recorded answers at 12 letters and 300 at 40. I took all of them and not a sample. For each recorded answer, every byte is the largest output of the reader's forward after the prompt and the recorded bytes before it, and the answer ends at the end symbol or at four bytes. **The reader's answers agree with the recorded answers in 600 of 600 items in each run, 1,200 of 1,200 in all.**

The counts are those of the panels that `np_gru.py` builds for the seed, which are the panels of the verifications of E42 and E49; the scores of E50 are therefore taken on E42's panels.

## 7. Multiply-adds a position: holds (part 7)

Recounted from the masks in the final files and from the reader's code, the same in both runs.

| Count | Input map | `l2` | `lm` | Cells | `l3` | Total | Reported |
|---|---|---|---|---|---|---|---|
| First count, `C-dense` | 393,216 | 262,144 | 131,072 | 8,448 | 131,584 | 926,464 | 926,464 |
| First count, `F10` (with the 4,096 of `l1`) | | | | | | 930,560 | 930,560 |
| Second count, `C-dense` | 8,448 (three gates, 256 cells, 11 values; the same for every used byte) | 5,632 | 131,072 | 8,448 | 131,584 | 285,184 | 285,184 |

`C-dense` is 0.9956 of `F10` by the first count. To the 80 threshold on R the first count gives 4.10e+11 and 7.59e+11 multiply-adds for `C-dense` under backpropagation and the sweep against 5.64e+11 and 9.00e+11 for `F10`, and the second count gives 1.26e+11 and 2.34e+11, as reported.

One sentence of the report is inexact without being wrong in its figures: "`F10` has no such count, since its code has no zeros". `F10` has the count, and it equals its first count: in E42's two final files none of the 23,552 values of the codes of the 46 used bytes is exactly zero, so all 930,560 products are with a value that is not zero. By the second count `C-dense` is then 0.306 of `F10` a position and reaches 80 on R with 0.22 (backpropagation) and 0.26 (the sweep) of `F10`'s products. The brief states that the job is not about computation and that the readings use the first count, so the report was not obliged to give this.

## 8. The stream of training items: holds (part 8)

The receipts record, at every scored row, a running SHA-256 over every batch fed to the learner so far. I rebuilt that digest without the runner: the stream of the first family for seed 73129 from `np_gru.py` (the taught halves of the four operations, 0 to 12 letters, 64 items a batch), with each batch as the three arrays that the runner hashes.

- The rebuilt digest after 4,096 batches begins `bbd71ab7ca27b72e`, as reported, and after 24,576 batches `eb49a8e47bfc4368`.
- **The recorded digest equals the rebuilt one at 97 of 97 rows in each of the four receipts:** `C-dense` under both rules and E42's `F10` under both rules. The number of batches recorded in each row equals its update, and each run records 24,576 optimizer steps.
- The receipts of `C-dense` and of `F10` under the same rule record the same optimizer (AdamW, rate 0.001, weight decay 0.25, betas 0.9 and 0.98, the same two groups of parameters), the same batch size, the same largest number of letters, the same dose and the same text for the data.

The items and their order are therefore E42's, batch for batch. With section 5 (the same starting network) and section 6 (the same panels), `C-dense` and `F10` of seed 73129 differ in the code and in what follows from it, and in nothing else that the files show.

## 9. The four predictions: scored correctly (part 9)

| | Prediction | Chance | Recomputed | Worker |
|---|---|---|---|---|
| P50.1 | Checks 1 to 4 pass | 85% | true | true |
| P50.2 | `C-dense`, backpropagation: reaches 80 on R within the dose | 80% | true (6,912 updates) | true (6,912) |
| P50.3 | `C-dense`, sweep: reaches 80 on R within the dose | 55% | true (12,800 updates) | true (12,800) |
| P50.8 | `C-dense` against `F10`, backpropagation: updates to 80 on R *about the same* or *fewer* | 60% | true (*about the same*, ratio 0.73) | true |

The wording and the chances are those of the brief's revision 2. P50.1 rests on receipts that I could confirm only in part.

- *Check 1* (the specification file). Read here: the runner's copy and the specification file have 103 lines each and differ in lines 19 and 20, the two import lines; the specification file at b18e563b is the default branch's; the SHA-256 that the receipt records for both files are those computed here. That the self-checks, when executed, print the coordinator's four lines is the worker's record.
- *Check 2* (the code is fixed through a run). The receipts record it at 97 of 97 rows of both runs and at every row of the six dry runs; section 5 confirms it between start and end.
- *Check 3* (E49's checks 3, 4 and 5). The digest at 4,096 is rebuilt in section 8, and the masks are confirmed on the final files in section 5. The scorer's agreement on the births of an earlier job (largest difference of a count 0) is the worker's record; section 6 confirms the scorer on this job's own networks.
- *Check 4* (the forward against a second form). The worker records a largest difference of 2.4e-07 for `C16-in` under backpropagation against a tolerance of 1e-05, and 1.1e-06 at most over the three settings with the weights moved. The reader's two forms agree to 3.8e-06 on the trained weights of `C-dense`, and section 6 is a stronger form of the check.

## 10. The decision on stage 2: it is what the brief prescribes; what is on file agrees with the brief (part 10)

**The rule.** Revision 2 of the brief runs stage 2, the setting `C16-in` at 49,152 updates, under "each rule under which `C-dense` reaches 80 on R within its dose". `C-dense` reaches it at 6,912 updates under backpropagation and at 12,800 under the sweep, in the three-row version, in complete runs. Stage 2 therefore runs under both rules, as the report states. The brief also requires that stage 1 be read against `F10` before stage 2 starts: the stage report was committed at 21:49:18Z and amended at 21:49:39Z, and the queue log records the start of stage 2 at 21:49:42Z. Whether the report had been pushed by then cannot be read from the commits.

**The report's figures on what stage 2 can record.** Within the 48,640 recordable updates a factor of up to 7.04 (backpropagation) and 3.80 (the sweep) against `C-dense` can be recorded; the report gives 7.0 and 3.8. E49's factor for its setting with 16 values a cell against `F10` under backpropagation is 20,480 / 9,472 = 2.1622; times 6,912 this is update 14,945, or 14,930 with the factor rounded to 2.16 as in the report, and either lies within the 24,064 recordable updates of the first dose.

**Is stage 2 running what the brief specifies?** The evidence at b18e563b is consistent with it, and it is incomplete.

- The queue log has the lines `2026-11-17T21:49:42Z start C16-in S pid 3161` and `start C16-in B pid 3162`: the setting `C16-in`, both rules, two processes.
- `launch_e50.sh` starts each job as its own process with one thread and runs `exp50_code.py run <setting> <rule> 73129` with no option for the dose. The runner, unchanged since 0f3ec4bc, has `DOSES = {"C-dense": 24576, "C16-in": 49152, "C46-one": 49152}` and `SETTINGS` with `"C16-in": (16, 256)`. The dose is therefore 49,152 updates unless the files on the worker's disk differ from the commit.
- Each of the two logs has two lines: the value of a turned-on entry 3.2387, no cell without input, "a needed byte reaches at least 67 cells", and the scored row at update 0. The figure 67 is the reader's for 16 values a cell (67 to 87 cells for a needed byte); under the dense interface it would be 256.
- The setting as this runner builds it is the brief's. In the start files of the dry runs of check 5, which use the same code path and seed, the mask of the input map equals the reader's draw of E49's lists for 16 values a cell (digest `0e09b9ff0545e68d`), the mask of `lm` is full, the input map is the start of `C-dense` multiplied by the mask and by the square root of 32 with no difference in any bit, and every other tensor equals the start of `C-dense`.
- **Not yet on file:** no receipt, start file or weight file of stage 2 exists at b18e563b, because the receipt is first written at update 1,024. The dose recorded by the run, the digest of its lists and its scores can be checked only when a later commit holds them. Whether the two processes are still running cannot be known from the repository.

## 11. The report's wording against its figures; what a seed alone changes (part 11)

**The sentences.** I read every sentence of the report against the recomputed figures. The statements of learning use the brief's own form ("reaches 80 on R within the dose"). The comparison with `F10` is stated as *about the same* in every cell and is expressly "not read as a gain". The paragraph "What this stage does not show" names the single seed and the four things in which `C-dense` differs from `F10`. The further figures of the report are confirmed in part 12: a cell's input for a needed byte (0.389 at the start, 5.140 and 4.769 at the end), the mean magnitude of a kept weight of the input map (0.0313 at the start, 0.0938 and 0.0862 at the end), the three groups of 95, 131 and 30 cells and their mean recurrent weights (recomputed from the weight files to the four decimals printed), and the costs of 0.87 and 1.00 CPU-hours (from the receipts). **I found no sentence that claims more than the figures support.**

**The amendment at f54c6db8 was right.** The first version of the report said that the spread between seeds of `F10` under the sweep "is of that order". For R under the sweep that is false: the three seeds of `F10` differ by a ratio of 1.05, and `C-dense` is at 0.85 of `F10`. The worker removed the statement and wrote that the spread was not read.

**The spread, read from E42's receipts at 44f91edc.** Updates to 80 in the three-row version.

| Rule, score | `F10` 73129 | `F10` 93683 | `F10` 116089 | Range of the three | `C-dense` 73129 | `C-dense` minus `F10` of 73129 | `C-dense` minus the least of the three |
|---|---|---|---|---|---|---|---|
| B, 80 on A | 8,704 | 6,912 | 9,216 | 2,304 | 7,680 | −1,024 | +768 |
| B, 80 on R | 9,472 | 7,936 | 9,728 | 1,792 | 6,912 | −2,560 | −1,024 |
| S, 80 on A | 13,312 | 13,056 | 12,544 | 768 | 13,056 | −256 | +512 |
| S, 80 on R | 15,104 | 14,336 | 14,336 | 768 | 12,800 | −2,304 | −1,536 |

- **On A the difference is within what a seed alone changes.** `C-dense` lies inside the range of the three seeds of `F10` under both rules.
- **On R it is not.** `C-dense` reaches 80 before each of the three seeds of `F10` under both rules, by 4 scored rows under backpropagation and 6 under the sweep against the earliest of them, and its difference from `F10` of the same seed (2,560 and 2,304 updates) is larger than the range of the three seeds (1,792 and 768).
- **The levels.** The three seeds of `F10` differ by 4.39, 3.45 and 5.11 points on A, R and the length test at 40 letters under backpropagation and by 3.44, 3.60 and 7.00 under the sweep. The differences of `C-dense` from `F10` of seed 73129 (+0.11 to +4.83) are of that size. `C-dense` is above the highest of the three seeds on R by 0.63 (backpropagation) and 1.23 (the sweep) and on A under backpropagation by 0.06, and inside the range elsewhere.

**How far this goes.** It does not make the difference on R a gain. `C-dense` has one seed, and its own spread between seeds is not known. If `C-dense` and `F10` had the same distribution of this figure over seeds, one value would lie below the least of three others with a probability of about one quarter. The two rules share the seed, the starting network and the stream, so the two occurrences are not independent, and the probability of both lies between one sixteenth (if they were independent) and one quarter (if one implied the other). The record supports this much: on R the fixed code is earlier than every run of `F10` on file under both rules, on A it is not distinguishable from `F10`, and the form's *about the same* stands for all four cells. The report's caution is therefore correct, and its sentence that the spread "was not read" can be replaced by these figures.

## What the report should change

Nothing in the report is wrong in a figure. Three changes of wording are recommended, for the final report if the stage report is not revised.

1. **The spread between seeds.** Replace "the spread between seeds of `F10` was not read for this stage" by the figures of section 11: "E42's `F10` on its three seeds reaches 80 on R at 7,936 to 9,728 updates under backpropagation and at 14,336 to 15,104 under the sweep, and on A at 6,912 to 9,216 and at 12,544 to 13,312. `C-dense` is inside these ranges on A and below the least of the three on R under both rules, by 1,024 and 1,536 updates. `C-dense` has one seed, so this is not read as a gain."
2. **The second count of `F10`.** Replace "`F10` has no such count, since its code has no zeros" by "for `F10` the second count equals the first, 930,560, since no value of its code is zero"; the ratio to the threshold (0.22 and 0.26 of `F10`'s products to 80 on R) may be given beside it or left out.
3. **"Nothing here is independently confirmed."** The sentence was true when written. The final report can state that the last row of both runs, the recorded answers, the fixed code, the masks, the starting network, the stream of batches and the two counts of multiply-adds are confirmed by this verification, and that the rows before the last, the checks inside the runs and the cosines are not.

## How to repeat the checks, and the running time

From the repository root:

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -I PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/e50_stage1_verify.py

With one or more numbers from 0 to 12 as arguments, only those parts are printed. The whole script ran in 89 seconds on one core of the reviewer's container, which other processes share (started at 22:14:18 UTC on 17 November 2026). Part 6 takes about 80 seconds, part 8 about 8, and every other part about one second or less. The script writes no file and no bytecode.

Part 12 also compares the tables file and the summary file with the recomputed figures. Of the rows of `C-dense`, `F10` and E49's `in16` in the tables file, 80 were written again here in the tables' own formats and each is found in the file character for character: the thresholds, the four operations, the multiply-adds, the levels, A and R every 2,048 updates, the operations every 4,096 updates, the training loss, the checks inside the runs, the 28 rows of descriptions, and for `C-dense` the two tables of sizes, written from the weight files. In the summary file 184 of 184 thresholds, levels and changes of R equal the recomputed figures.

## What could not be checked

- **The rows before the last.** Only the start and the end of each run are saved. The scores at the 96 rows between, and with them every figure of updates to a threshold, the levels' first two rows and the change of R, are the receipts' record. What supports them from outside is limited to this: the last row is reproduced exactly from the final weights, the shares equal the counts in every row, the digest of the batches is right at every row, and the logs hold 97 rows in order without a repeat.
- **That the runs were trained under the rule they name.** The receipts record the rule; the two final networks differ; no gradient was computed here.
- **The checks inside the runs at the rows between** (the code table's SHA-256, the byte table and `l1`, the weights outside the masks, the optimizer's averages there, E42's limit) and the statement that the optimizer holds no state for the byte table or `l1`. The start and the final files agree with them.
- **The cosines** between the sweep's and the exact gradient, and the first-batch loss, which are given from the receipts only.
- **The checks before the runs that execute torch:** the output of the specification file's self-checks (check 1), the scorer on the births of an earlier job (check 3), the worker's two forms of the forward (check 4) and the dry runs (check 5), beyond what sections 5, 6, 8 and 10 confirm of their substance.
- **Costs and times:** the CPU-hours, the start and end times and the exit codes are those of the receipts and of the queue log.
- **Stage 2 beyond its first two log lines,** as section 10 states; and whether the stage report was pushed before stage 2 started.
- **The receipts' two digests of tensors** (`tensors_sha256`, `table_sha256`), whose formula is the runner's; the files' SHA-256 and the tensors themselves were compared instead.
