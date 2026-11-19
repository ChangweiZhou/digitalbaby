# Verification of E57: the sweep with a scale for each receiving unit, and with a code that is not decayed (seed 73129)

Written at 21:44 UTC on 18 November 2026 by an agent of the reviewer that had seen no one's opinion of the result. The object is the worker's branch at commit 85417d06: the report `MOD97_LOCAL_RULE_20261106/exp45/e57/E57_REPORT.md`, the tables `E57_TABLES_73129.md`, the summary `E57_SUMMARY_73129.json`, the receipts, logs and weights under `exp45/e57/`, the checks under `exp45/e57_qual/` and the runner `exp57_scale.py`; and the worker's message, which was treated as a set of claims. The standard is the brief, `COORDINATION_20261117/TASK_E57_THE_SWEEP_WITH_A_SCALE_FOR_EACH_RECEIVING_UNIT_20261117.md`, revision 5.

Every figure below is printed by `e57_verify.py` (beside this file), run from the repository's root as `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -I PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/e57_verify.py`. It takes about one and a half minutes, reads every file with `git show` into memory, writes no file and executes no file of the worker's branch. The numbers in brackets, for example (1.6), name the section of the script's output that prints the figure. By `df -h`, which is outside the script, free disk was 1.2 GB before the first weight file was read and 1.2 GB after the last.

## 1. What was verified and how

- **Receipts and settings** (script section 1). The nine receipts were read field by field against the brief: network, scale and its placement, rule, seed, number of updates, decay of the code, optimizer groups, rate and betas, the optimizer's record read back, finite losses, masks and E42's limit at every scored row, the steps of the scored rows and of the readings, resumption, runner hash, threads. The data digests were compared row by row with E42's and E49's receipts (at 44f91edc and a36de661), and the digest of the first 4,096 batches was computed independently with the stream of `sweep_scale_variants.py`. The runner was read at the five commits that the receipts name, and its versions were compared line by line with `difflib`. The check receipts were read, and check 6 was recomputed from the receipts.
- **Figures** (section 2). Updates to 80, levels at 24,576 and at the end, the rise of R, the words, the words of `describe` (copied from `e42_tables.py` at 44f91edc, lines 63 to 72), the first reading below 0.5, and every cosine, size and loss of the tables of readings were recomputed from the receipts' raw entries. They were compared with every table of the tables file and of the report (parsed from the files), every entry of the summary, 60 figures of the report's text and 63 figures, words and outcomes of the worker's message.
- **Weights** (section 3). For all nine runs, the final and the starting weight files were read one at a time: the file hash and `tensors_sha256` against the receipt, the masks, the weights outside the masks, and the final counts of right answers on the five sets of items, by a forward pass in numpy (float32, masked matrices, whole answers under teacher forcing). For all nine runs, the cosines between the sweep's gradient under each of the four scales and the exact gradient, and the two sizes, were recomputed on the 16 batches of the brief (the second to the seventeenth of the stream) with the float64 forward and backward pass of `sweep_scale_variants.py`, and compared with the receipts' last reading.
- **Wording, predictions and reading** (sections 4 and 5). Each run's words were derived from the brief's definitions. For each word, the margin on which it rests was computed; the predictions were scored and the row of the brief's table of explanations was found.
- **Text files** (section 6). The committed text files of `exp45/e57` and the runner were searched for unfilled marks and for names of language models; the search strings were built from parts.

## 2. Completeness and settings of the nine runs

All nine runs are complete and carry the settings that the brief requires (1.2): five runs of `in16` under the sweep of 49,152 updates (193 scored rows each), and four runs of 24,576 updates (97 rows each). The run without decay of the code has `book.weight`, `l1.weight` and `l1.bias` in the group without weight decay; every other run has `book.weight` alone in that group. Every run made exactly its number of optimizer steps, has an empty list `resumes`, finite losses and masks and E42's limit true at every scored row, a reading every 2,048 updates, and the runner hash of the committed runner (`5409562f…`). Each run's log holds one start, and `queue.log` records nine ends with exit code 0, from 15:09:01Z to 21:23:43Z. The 15 hashes of `sources_sha256` equal the files at 85417d06, and the two copied files of E49 equal those at a36de661 (1.1). The runs of `in16` start at E49's `657e1858…`, those of `F10` at `dede71d7…`, and the run of `G` at the start of check 1's dry run of `G`, whose end is E42's hash `79618ab2…` (1.2).

**The stream** (1.3). The digest of the first 4,096 batches computed here is `bbd71ab7ca27b72e…`, and it equals every run's row of update 4,096. Every scored row of every run has the batch digest of E42's or E49's run of the same network at the same update (97 of 97 rows for each run and each of the two references). The five runs of 49,152 updates agree with one another at all 193 rows.

**The checks** (1.4). Check 1 is equal bit for bit in all eight comparisons of (a), the five hashes of (b) equal the brief's (both in check 1's receipt and in the dry runs' own receipts), the rows of the dry run of `in16` of 2,048 updates equal E49's, and rule (c) was not used. Check 2's largest difference from the brief's figures is 0.00049; with the scale `exact` the least of the 13 cosines is 0.999999999999969. Check 3 ends on `c0eb6548…` with both rows equal. Check 4 holds for all runs. Check 5's forecast is 18.34 CPU-hours against the limit of 21. In check 7, the eleven other tensors are equal after one update, the largest deviation of `l1.weight` is 2.511e-08, and the mean magnitude of `l1.weight` is 0.1773 at the start, 0.1764 at the end without decay and 0.1655 with it. In check 8, the killed run (exit 9, last row 6,144, four readings) and the uninterrupted run end on the same `00cddfc869c516e0…`, no row differs outside the fields of CPU and wall time, the readings are equal, and the CPU ratio is 2.01. Checks 2 and 3 were made with the runner at c55aaa1b (`3b2f58cf…`). Checks 1, 5, 7 and 8 and the nine runs used the runner `5409562f…`. The two versions differ in three lines, all inside the function `check8` (1.1). The first attempt of check 8 stopped because its subprocess failed at the harness's gate on environment variables (1.4).

**Check 6, recomputed** (1.5). Every field common to the two receipts (17 fields, the CPU and wall times excepted) of the 97 scored rows of the run of `in16` with `matrix` and a decayed code equals E49's run under the sweep, to update 24,576. The run under back-propagation equals E49's run in every row and ends on `707c8569…`, E49's hash and the brief's. Its weight file holds tensors equal bit for bit to E49's (3.1). The readings of update 24,576 of both runs are bit for bit the values of check 2, within 0.00049 and 0.00047 of the brief's figures.

**CPU time** (1.6). The nine runs took 17.19 CPU-hours, of which 0.56 were the readings; the worker's figure is right. The check receipts' own entries sum to 0.42 CPU-hours, but check 8's two dry runs ran as subprocesses and are not in its entry (0.89 CPU-hours), and the killed process spent at least 0.15 CPU-hours after its saved state. The checks therefore took at least 1.45 CPU-hours, and the job at least 18.64, against the brief's estimate of 13.1 + 1.5 (finding 1).

## 3. Figures compared and figures that differ

| Source | Compared | Of which figures | Words or outcomes | Differing |
|---|---|---|---|---|
| Tables file | 5,197 | 5,118 | 79 | 0 |
| Summary | 293 | 230 | 63 | 0 |
| Report (its tables) | 5,235 | 5,128 | 107 | 0 |
| Report (figures in its text) | 60 | 60 | 0 | 0 |
| Worker's message | 63 | 35 | 28 | 0 |
| All | 10,848 | 10,571 | 277 | 0 |

(2.7.) Each figure was compared at the precision at which it is printed; the summary was compared to 1e-9. No figure, word or outcome differs. The figure "to within 2.5e-8" of check 7 agrees at two significant digits with the recorded 2.511e-08 (finding 9).

## 4. What the weights reproduce

For all nine runs (3.1): the file hash equals the receipt's `weights_sha256`, and the hash of the thirteen tensors equals `tensors_sha256`. The starting file has `start_tensors_sha256` and the recorded file hash, and the masks of every `in16` start equal those of E49's file. No weight outside a mask is non-zero. The forward pass in numpy gives exactly the receipt's last row in all twelve counts of every run: A (four operations), R (two), T (four) and the length test at 12 and 40 letters. For example, `in16` with `row` has A 248, 258, 294, 288; R 838, 835; T 281, 289, 296, 299; and length 248, 126. `in16` with `matrix` has A 5, 2, 118, 117; R 15, 7; T 8, 7, 117, 111; and length 5, 2.

The cosines of the last reading, recomputed on the saved weights for all nine runs, differ from the receipts by at most 8.49e-07 over the 24 cosines of a run, and the two sizes agree to six decimals (3.2). For the two networks the brief names, at update 49,152 (byte table, `l1`, input map, recurrent map, `lm`, `l2`):

| Run | `matrix` | `row` | `rowcells` | `rowcol` |
|---|---|---|---|---|
| `in16` `row` | 0.220, 0.195, 0.383, 0.389, 0.592, 0.629 | 0.456, 0.417, 0.571, 0.646, 0.592, 0.629 | 0.217, 0.282, 0.400, 0.396, 0.592, 0.629 | 0.234, 0.331, 0.412, 0.408, 0.592, 0.629 |
| `in16` `matrix` | 0.285, 0.291, 0.722, 0.630, 0.656, 0.530 | 0.698, 0.585, 0.807, 0.769, 0.745, 0.812 | 0.383, 0.494, 0.781, 0.687, 0.656, 0.530 | 0.468, 0.416, 0.732, 0.693, 0.656, 0.530 |

The recomputed values and the receipts' values agree in every printed digit.

## 5. Findings, by weight

1. **Error of fact: the CPU time of the checks.** The report and the message give 0.42 CPU-hours for the checks. That is the sum of the check receipts' own entries. It leaves out check 8's two dry runs, which ran as subprocesses (0.89 CPU-hours), and the killed process's work after its saved state (at least 0.15). The checks took at least 1.45 CPU-hours and the job at least 18.64 (1.6). The figure 17.19 for the runs is right.
2. **Omission: a failed check, a changed runner, and checks made with the earlier runner.** The first attempt of check 8 failed: the subprocess stopped at the harness's gate on environment variables, and the parent stopped with `FileNotFoundError`. The brief says that a worker whose check fails stops, reports the first difference and asks. The worker instead changed the runner (66e20486) and made the check again. The report describes this under "Checks", but its section "Rules, and what departed from them" does not list it. The report also does not say that checks 2 and 3 had already been made with the earlier runner (c55aaa1b, `3b2f58cf…`), whereas the brief requires the runner to be committed before the first check. The change is three lines inside `check8` (1.1), so no run and no other check is affected. The worker scored P57.1 false on this ground, which is correct.
3. **Overstatement: "so E49's run had failed and was not late".** The run with `matrix` shows that E49's run did not learn the family within 49,152 updates. It says nothing of a longer run, so "was not late" goes beyond what was measured. The sentence should read "did not learn the family within 49,152 updates".
4. **Overstatement: "The scale `row` does no harm where `matrix` works."** The basis is one run of `G` (and the two runs of `F10`) on one seed. The supported sentence is that, on this seed, in `G` the scale `row` gave *about the same* updates to 80 as E42's runs (6,144 and 6,144 against 5,632 and 6,656) and levels of A 98.89 and R 98.83.
5. **Omission: words that rest on one or two scored rows are not marked** (4.1, 4.2). Against E42's run of `F10` under back-propagation, `F10` with `row` is *about the same* on R at a ratio of 1.459, and `F10` with `rowcol` on A at 1.471. Each word would become *more* if 80 had been reached two scored rows later. The word *partly* of the first 24,576 updates of the run without decay of the code rests on a level of T of 51.83 (rows 50.17, 52.67 and 52.67). The word *learns the family* of `in16` with `row` rests on R reaching 80 at 44,800, 4,352 updates before the end. R reaches 80 on a single row first at 41,216, and 16 of the 18 rows from 44,800 on are at or above 80. The R level is 3.78 points above 80. None of these words is wrong.
6. **Omission: the speed of the run with `row` is not set beside the record.** The report gives 2.17 and 2.19 times the updates of back-propagation in the same network. It does not say that this is beyond the ratios of the sweep to back-propagation in `F10` on E42's three seeds (1.36 to 1.89 on A, 1.47 to 1.81 on R). It also does not say that at 24,576 updates, the length of the run under back-propagation, the run with `row` has A 53.50 and R 44.92 against 85.22 and 82.53 (5.1, 5.2).
7. **Matter of form: an unclear antecedent.** In "Both changes are partial: neither the run without decay nor the runs with `rowcells` and `rowcol` learn the family", the subject "both changes" names two things while the clause lists three runs.
8. **Matter of form: a superseded file.** `E57_SUMMARY_draft.json`, which holds seven runs and differs from the final summary, is committed in the folder of results without a note (6.3).
9. **Matter of form: a rounded bound.** Check 7 is reported as agreeing "to within 2.5e-8", while the recorded largest deviation is 2.511e-08 (1.4). This is immaterial.

The report contains no error in a figure of its tables, its summary, its readings or its predictions. The worker's message contains none apart from finding 1. There are no unfilled marks and no names of language models outside branch names in the 37 committed text files of `exp45/e57` and the runner (6.1). The same holds in the 39 text files of `exp45/e57_qual`, `e57_tables.py` and `E49_COPIES_FOR_E57.txt`. The list of committed files in the report matches the tree: 110 paths, none missing and none extra (6.2).

## 6. The fixed wording and the predictions

**The nine runs as recomputed** (2.1; updates to 80 by the brief's definition; a level is the mean of the last three scored rows):

| Run | Updates | A to 80 | R to 80 | T to 80 | Level A | Level R | Level T | Level at 40 letters | Word |
|---|---|---|---|---|---|---|---|---|---|
| `in16` `matrix` S | 49,152 | not reached | not reached | not reached | 1.22 | 1.18 | 1.89 | 1.11 | not learned |
| same, first 24,576 | 24,576 | not reached | not reached | not reached | 1.89 | 1.37 | 2.50 | 1.22 | not learned |
| `in16` `row` S | 49,152 | 36,096 | 44,800 | 26,368 | 86.06 | 83.78 | 96.06 | 48.11 | learns the family |
| same, first 24,576 | 24,576 | not reached | not reached | not reached | 53.50 | 44.92 | 72.67 | 18.67 | partly |
| `in16` `rowcells` S | 49,152 | not reached | not reached | not reached | 53.67 | 40.23 | 68.33 | 4.22 | partly |
| same, first 24,576 | 24,576 | not reached | not reached | not reached | 31.33 | 24.78 | 43.00 | 2.67 | not learned |
| `in16` `rowcol` S | 49,152 | not reached | not reached | 36,608 | 69.83 | 58.72 | 85.89 | 34.11 | partly |
| same, first 24,576 | 24,576 | not reached | not reached | not reached | 42.83 | 27.27 | 58.83 | 5.67 | partly |
| `in16` `matrix` S, code not decayed | 49,152 | not reached | not reached | not reached | 51.06 | 46.05 | 71.00 | 13.00 | partly |
| same, first 24,576 | 24,576 | not reached | not reached | not reached | 37.78 | 30.38 | 51.83 | 11.67 | partly |
| `in16` B | 24,576 | 16,640 | 20,480 | 15,104 | 85.22 | 82.53 | 93.61 | 66.11 | learns the family |
| `F10` `row` S | 24,576 | 11,520 | 13,824 | 10,496 | 92.94 | 92.30 | 98.72 | 91.89 | learns the family |
| `F10` `rowcol` S | 24,576 | 12,800 | 15,616 | 9,984 | 92.44 | 91.43 | 97.50 | 92.67 | learns the family |
| `G` `row` S | 24,576 | 6,144 | 6,144 | 4,864 | 98.89 | 98.83 | 99.67 | 97.78 | learns the family |

**Words of `describe`** (4.2). Against E49's run under back-propagation (16,640 and 20,480), every run of `in16` under the sweep is *more* on A and on R over 49,152 updates; for `row` the ratios are 2.17 and 2.19. Over its first 24,576 updates every such run is *not decided*, since 24,064 recordable updates are only 1.45 and 1.18 times the reference. Against E42's runs under the sweep, `F10` with `row` (0.865 and 0.915), `F10` with `rowcol` (0.962 and 1.034) and `G` with `row` (1.091 and 0.923) are *about the same*. Against back-propagation, everything is *about the same* except `F10` with `rowcol` on R (1.649, *more*). The differences of `F10` with `row` from E42's run under the sweep (−1,792 and −1,280) exceed the range of E42's three seeds (768); the others (−512, +512; +512, −512) lie within it. Every word that the worker gives is the one the definitions give (section 3), and the dependence of some words on one or two rows is listed in finding 5.

**The first reading at which the cosine of `l1` under `matrix` is below 0.5** is 4,096 in both the failed run and the run under back-propagation (loss of the row 1.040 and 0.842). The brief's sentence is therefore "does not separate", as the report says.

**The table of explanations** (4.4). The brief's table has five rows: (1) `matrix` learns; (2) `matrix` does not, `rowcells` learns; (3) `matrix` and `rowcells` do not, `row` learns; (4) `rowcells` and `row` are both *not learned*; (5) any other outcome. Over 49,152 updates `matrix` is *not learned*, `rowcells` *partly*, `rowcol` *partly* and `row` *learns the family*. The runs therefore select row 3: "Part 1 alone is *not supported*. A scale for each receiving unit at every map was sufficient; the runs do not say which of the scales at `lm`, `l3`, `l2` and `l1` is needed beside that of the cells". The condition of the table holds (section 2, check 6). Since the run with a decayed code is *not learned*, the brief's sentence for the code not decayed is the one for *partly*: "on this seed, the network without the decay of `l1` reaches A 51.06, R 46.05 and T 71.00 and does not learn the family" (4.5).

**Predictions** (4.3). These outcomes agree with the worker's.

| | Outcome | Basis |
|---|---|---|
| P57.1 (65%) | false | check 8 passed at its second attempt; checks 1 and 3 passed without rule (c) |
| P57.2 (12%) | false | `row`, first 24,576: *partly* |
| P57.3 (12%) | false | `rowcol`, first 24,576: *partly* |
| P57.4 (30%) | true | level of T 72.67 |
| P57.5 (65%) | true | x = 11,520, ratio 0.865 |
| P57.6 (85%) | true | x = 6,144, ratio 1.091 |
| P57.7 (60%) | true | cosine of `l1` 0.531, 0.420, 0.369 at 2,048, 4,096, 6,144 |
| P57.8 (30%) | true | *learns the family* |
| P57.9 (20%) | false | *partly* |
| P57.10 (85%) | true | *not learned* |
| P57.11 (65%) | true | 0.450 at 4,096 |
| P57.12 (45%) | true | level of T 71.00 |
| P57.13 (20%) | false | *partly* |

## 7. What the nine runs may be read as showing

On seed 73129, in the network with 16 values of the code a cell, the sweep with the scale `row` learned the family within 49,152 updates (A reaches 80 at 36,096 and R at 44,800; levels A 86.06, R 83.78, T 96.06). The same network with `matrix` did not learn it within 49,152 updates (level of T 1.89), and the run with `matrix` repeated E49's failed run row by row to 24,576. The statement "a scale for each receiving unit lets the sweep learn the input with 16 values a cell" is therefore supported on this one seed, in the brief's sense of a sufficient change, and in no stronger sense. The learning is slow: the run needed 2.17 and 2.19 times the updates of back-propagation in the same network, which is beyond the sweep's ratios in `F10` on E42's three seeds (1.36 to 1.89 on A, 1.47 to 1.81 on R). At E49's length of 24,576 updates it is only *partly* (A 53.50, R 44.92 against 85.22 and 82.53 under back-propagation). Its levels remain below those of the networks with the whole code under the same scale (`F10` A 92.94, R 92.30; `G` 98.89, 98.83). At 40 letters it reaches 48.11, against 66.11 under back-propagation in the same network and 91.89 in `F10`. R met the three-row rule 4,352 updates before the end, so the word rests on the last 18 scored rows. Which map's scale matters is not known: the scale at the two maps of the cells alone (`rowcells`) and the product form there (`rowcol`) gave *partly* (T 68.33 and 85.89). This says only that the scale at the cells' maps was not sufficient on this seed; no run separates `l1`, `l2`, `lm` and `l3`. In `F10` and `G` the scale changed the updates to 80 within the word *about the same*, and the code that is not decayed gave *partly* (A 51.06, R 46.05, T 71.00), so the decay of `l1` was not the whole of the failure. What `row` computes is, for each linear map and each row i, the mean of |W[i, j]| over the entries that row keeps: the mean magnitude of the incoming weights of receiving unit i (8 entries at `l1`, 16 at the input map, 11 at the recurrent map, 256 at `lm`, 512 at `l2` and `l3` in `in16`); the error that unit i sends back to sending unit j is sign(W[i, j]) times that mean times the error at i. It is a function of weights that unit i itself uses in its forward pass, and in that sense a quantity a unit could hold; the sign of every W[i, j] must still be available on the path back to j, and the cosines do not show the mechanism, since at the end the run that learned has, under its own scale, cosines of `l1` and of the input map (0.417 and 0.571) no higher than those of the failed run under `row` (0.585 and 0.807) (3.2, 5.4).

## 8. What was not verified

- Nothing was run with the training library, so the runs themselves were not repeated. The verification rests on the receipts, the hashes, the weight files and an independent numpy forward and backward pass.
- The counts of the scored rows before the last row, the training losses and the readings before the last one were not recomputed from weights, because the intermediate weights are not saved. They were compared only with the tables and with E49's and E42's receipts where those overlap.
- The full stream digest was computed independently to update 4,096 only. Beyond that, the agreement of the five long runs with one another and of all runs with E42's and E49's receipts was checked.
- The equality of `exp42_cells.py` and `e42_cells.py` with the default branch and with a36de661, which the report asserts, was not checked; their hashes were compared with the files at 85417d06 only.
- The difference between the runner's `run` and `exp49_interface.run` (`E57_RUN_DIFF.txt`) was not read line by line against the brief's point 4. Its effect was tested instead through check 1, check 3 and check 6, which reproduce E42's and E49's results bit for bit.
- Whether the worker asked the reviewer before repeating check 8 cannot be seen from the branch.
- One seed: nothing here bears on another seed or on E49's two other networks with a limited input.
