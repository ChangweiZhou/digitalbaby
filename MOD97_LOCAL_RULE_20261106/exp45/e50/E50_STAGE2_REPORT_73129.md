# E50, stage 2, seed 73129: the fixed code with 16 values a cell (`C16-in`) under both rules

Brief: `OPUS_TASK_E50_AN_INPUT_CODE_IN_WHICH_A_VALUE_STANDS_FOR_ONE_BYTE_20261116.md`, revision 2 (d6606454 on this branch). One seed, a first measurement: a description, not a claim. Nothing here is independently confirmed; the reviewer's reader has not yet been applied to these weights, and no formal checker was run. The terms are those of `E50_STAGE1_REPORT_73129.md`.

**The setting.** `C16-in` is `C-dense` with each cell reading 16 of the code's 512 values (E49's lists for K_IN = 16, the lists of E49's `I16` and `in16`; the kept weights of the input map start 5.66 times larger) and every readout unit reading every cell. A cell then has input for 1 to 10 of the 20 bytes the task must tell apart (median 6) and for 0 to 7 of the ten digits (median 3). E49's `in16` is the same lists on the learned code, in which every cell can identify every byte.

**Runs.** `python3 exp50_code.py run C16-in <S|B> 73129`, runner at 0f3ec4bc, 49,152 updates each, two processes beside each other (from 23:37:49Z three, with the first run of stage 3), one thread each. Both started at 21:49:42Z; B ended at 23:37:13Z and S at 23:51:31Z, both with exit code 0 (`exp45/e50/logs/queue.log`). **Neither run was restarted or paused:** the largest wall-clock gap between two scored rows was checked at 22:11Z (41 seconds) after the reviewer warned that the host might have stopped. Cost: 1.76 CPU-hours (B) and 2.00 (S), scoring included. Every figure is from `exp45/e50/E50_TABLES_73129_stage2.md` (`python3 e50_tables.py --tag 73129_stage2`). Each run is read twice: within its first 24,576 updates (the dose of `C-dense`, `F10` and E49) and at its end.

## Result

| | `C-dense`, B | `C16-in`, B, first 24,576 | `C16-in`, B, end | E49's `in16`, B | `C-dense`, S | `C16-in`, S, first 24,576 | `C16-in`, S, end | E49's `in16`, S |
|---|---|---|---|---|---|---|---|---|
| Updates to 80 on A | 7,680 | not reached | 40,704 | 16,640 | 13,056 | not reached | not reached | not reached |
| Updates to 80 on R | 6,912 | not reached | not reached | 20,480 | 12,800 | not reached | not reached | not reached |
| Updates to 50 on R | 4,352 | 18,176 | 18,176 | 11,776 | 7,424 | not reached | not reached | not reached |
| Level of A | 97.7 | 67.9 | 83.8 | 85.2 | 93.5 | 32.7 | 49.5 | 1.9 |
| Level of R | 97.1 | 62.1 | 79.4 | 82.5 | 94.1 | 29.0 | 43.6 | 1.4 |
| Level of the length test at 40 letters | 92.7 | 41.6 | 56.2 | 66.1 | 90.6 | 9.6 | 13.6 | 1.2 |
| R: last three rows minus the three ending 2,048 updates earlier | +1.1 | +2.7 | +2.6 | +0.7 | +3.0 | +1.7 | +1.3 | −0.4 |
| max, min at 12 letters: updates to 80 | 1,024; 1,280 | 5,376; 5,632 | the same | 5,120; 5,376 | 1,280; 1,536 | 7,424; 7,680 | the same | not reached |
| add, sub at 12 letters: updates to 80 | 7,680; 7,680 | not reached | 34,560; 40,704 | 15,616; 18,432 | 12,032; 13,824 | not reached | not reached | not reached |

**Under neither rule does `C16-in` learn the first family within its dose** (it does not reach 80 on R within 49,152 updates in the three-row version).

- **Backpropagation.** R is at 79.4 over the last three rows (80.1, 80.5, 77.7); its highest single row is 80.75 at 47,616 updates, and the single-row version records 80 there. It rises by 2.6 points over the last 2,048 updates, which is below the 5 points of *still rising*. A reaches 80 at 40,704 updates, 5.30 times `C-dense`. Against `C-dense`: updates to 80 on A and on R *more*, multiply-adds *rises* by both counts; at the end the level of A is *about the same* (83.8 against 97.7, a difference of 13.9), and those of R and of the length test at 40 letters are *lower*.
- **The sweep.** R is at 43.6 over the last three rows and A at 49.5; R rises by 1.3 points over the last 2,048 updates. Against `C-dense`: updates *more*, all three levels *lower*.
- **max and min reach 80 under both rules** (5,376 and 5,632 updates under backpropagation; 7,424 and 7,680 under the sweep), and stay at 96 to 98. The answer to max or min is one of the operands, so under both rules the network brings both operands to the answer; what it does not learn within the dose is the sums. This is the brief's description for a setting in which max and min reach 80 and R does not.

**Beside E49's `in16` against `F10`.** Under backpropagation `in16` needed 2.16 times `F10`'s updates to 80 on R and 1.91 times to 80 on A. `C16-in` needs 5.30 times `C-dense`'s updates to 80 on A and does not reach 80 on R within 48,640 recordable updates, which is more than 7.04 times `C-dense`'s 6,912. The cost of `C16-in` against `C-dense` is therefore larger than the cost of the mask that E49 measured, and by the brief it is not described as the cost of the mask. By the brief the job says why only through `C46-one`. Within the first 24,576 updates `C16-in` is below `in16` on every level (R 62.1 against 82.5). Under the sweep the order is the other way: `in16` stayed at 1.4 on R and 39 to 41 on max and min, while `C16-in` reaches 43.6 on R and 96 on max and min.

**The explanation under test** (the brief, after the predictions): that the sweep did not learn with a limited input in E49 because it does not keep a learned code at the size that a cell reading few values requires. Here the code cannot shrink, and `C16-in` under the sweep does not reach 80 on R within its dose while `C-dense` under the sweep does. **By the brief's criterion the explanation is refuted as a sufficient one.** It is also the case that with the fixed code the sweep's network gets much further than E49's did (R 43.6 against 1.4; max and min learned); the job does not say what part of that is the code's size, since the fixed code differs from the learned one in other respects.

**The answer's two bytes apart** (R's items; levels at the end): under backpropagation the first byte is right in 84.2% of items and the second in 92.6%; under the sweep 71.5% and 54.0%. Under backpropagation the first byte is right more often until about 20,480 updates and the second from about 28,672 on.

**By what a cell knows** (mean magnitude of the kept recurrent weights from neighbours into the cells that have input for 0 to 2, for 3 or 4 and for 5 or more digits; 95, 131 and 30 cells). At the start 0.150, 0.152, 0.150. At the end of `C16-in`: 0.265, 0.240, 0.233 under backpropagation and 0.226, 0.219, 0.213 under the sweep. In `C-dense`, where the grouping means nothing: 0.147, 0.144, 0.145 and 0.150, 0.152, 0.149. So in `C16-in` the recurrent weights grow in every group (they do not in `C-dense`), and the cells with input for the fewest digits end with the largest, by 0.03 (B) and 0.01 (S) over the cells with the most. The figure describes the size of weights and is not read as use; the test of use is the reviewer's.

**Other measures.** A cell's input for a needed byte (root mean square over all cells, without the bias): 0.381 at the start, 0.765 (B) and 0.629 (S) at the end, against 5.14 and 4.77 in `C-dense`. Mean magnitude of a kept weight of the input map: 0.177 at the start, 0.140 (B) and 0.117 (S) at the end. Cosine between the sweep's and the exact gradient on the first batch at the end: input map 0.104 (B run) and 0.348 (S run); recurrent map 0.293 and 0.431; `lm` 0.664 and 0.669 (at the start 0.709, 0.755, 0.866).

**Checks inside the runs.** At all 193 scored rows of both runs the code table, the byte table and `l1` are unchanged, the masks hold with the optimizer's averages zero outside them, and E42's limit holds; the digest at row 4,096 is `bbd71ab7ca27b72e`.

**Files.** `exp45/e50/e50_C16-in_<rule>_73129.json` (receipt), `_start.pt`, `.pt` (final), `_at24576.pt` and `_at24576_answers.json` (the first dose), `_answers.json`. Final weights SHA-256 (first 16): B `f9b170926b45326d`, S `143520ef71f5a474`.

## Predictions (the reviewer's, stated before the runs)

| | Statement | Chance | Outcome |
|---|---|---|---|
| P50.4' | `C16-in` reaches 80 on R within 49,152 updates under backpropagation | 55% | false in the three-row version, which is the brief's (R 80.1, 80.5, 77.7 at the last three rows; a single row at 80.75 at 47,616) |
| P50.5' | `C16-in` reaches 80 on R within 49,152 updates under the sweep | 25% | false (level 43.6) |
| P50.6' | Under backpropagation `C16-in` reaches 80 on R in no more than 3.24 times the updates of `C-dense` | 45% | false (not by 22,395 updates; not within the dose) |
| P50.7' | In `C16-in`, max and min at 12 letters reach 80 under backpropagation | 90% | true (5,376 and 5,632) |
| P50.8' | In `C16-in`, max and min at 12 letters reach 80 under the sweep | 45% | true (7,424 and 7,680) |

## What follows

By the brief's rule stage 3 runs under both rules: `C46-one` (every cell reads one value of every used byte, 46 values a cell; the readout dense), 49,152 updates. It started at 23:37:49Z under backpropagation and at 23:52:04Z under the sweep. If `C46-one` reaches 80 on R under a rule under which `C16-in` does not, the number of bytes a cell has input for is what costs; if it does not, the cost is that of an input resting on one weight for a byte, and nothing is said about what a cell knows.

## What this stage does not show

One seed. Under backpropagation the run ends within a point of the threshold, and a single row passes it; that `C16-in` "does not learn within the dose" is the brief's three-row reading at 49,152 updates and says nothing of a longer dose. Until `C46-one` has run, the cost against `C-dense` has two possible causes that this stage does not separate: the number of bytes a cell has input for, and the number of weights on which a cell's input for a byte rests. Nothing is said about a limited readout.
