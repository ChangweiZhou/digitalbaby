# Verification of the result of E48: a new family learned only in connections gated by its own code

17 November 2026. Written by the outside reviewer for the coordinator and for Mr. Zhou.

## 1. What was verified, and how

**The experiment.** E48 starts from the six networks that had lived a whole life in E26 (three seeds, each under two learning rules) and teaches them a new family of four operations under modulus 83, from a tenth of its pairs, as E40 did. In E40 every weight was free. In E48 every weight of the lived network is frozen, and the change is made in two places only: an added *term* on each of the two matrices of the recurrent layer, and the four rows of the byte table that belong to the new family's four code bytes. A term is the product of two matrices, `up · down`, of rank r (4, 16 or 64); it is added to the frozen matrix only when a *gate* is on. The gate is supplied by the harness: it is on when the byte after the start symbol is one of the new family's codes `e f g h`, and off for every other item. The two rules are back-propagation and the rule the programme calls the sweep, in which the error is sent back through each linear map by the signs of its weights times their mean size.

**The scores.** The *reduction score* is E40's: the percentage of whole answers right on every held-out addition and subtraction pair of the new family whose answer needs the reduction at 83, without letters (about 6,140 pairs a seed). *R* is the mean of the reduction score at rows 128, 256 and 512, a row being a number of updates at which the network is scored. The *taught panel* is 300 taught pairs of addition and of subtraction at 12 letters.

**The material.** The worker's branch `claude/sweet-turing-lwi3no` at 2d9d1726: the runner `exp48_gated_run.py`, 18 run receipts, 18 files of generated answers, seven check receipts and 108 committed term files. The default branch at 7cabb13a: the brief (revision 2), the specification file, E40's receipts and the life-end weights. `claude/vibrant-allen-8hziln` at 6922a3a4: E40's weights at rows 64 and 8,192.

**The method.** Nothing was trained and no code of another branch was run. Files were read with `git show`. Every figure below comes from `e48_result_verify.py` in this folder, which uses the numpy reader written for the audit of the brief (`np_gru.py`, `e48_gated_reader.py`): a second implementation of the network and of the gated forward, without torch. The command lines are in section 11.

## 2. Verdict

1. **The result stands as reported.** All 18 receipts are complete and pass their recorded checks; the six required changes of the audit are in what ran. From the life-end weights and each of the 108 committed term files the reader reproduces the recorded scores of that row: 1,943 of 1,944 counts are equal, and the remaining one differs by a single item on which the two largest outputs differ by 0.000003 (a rounding tie between numpy and torch). The bytes generated for the reduction pairs equal the worker's answer files on all 515,396 items compared.
2. **The earlier families are kept exactly, and this is a property of the construction, not a finding.** With the term and the changed code rows loaded, the answers to about 37,500 items of the earlier families in each of the six networks are those of the life-end network item for item. I found no path by which the term or the freed rows reach an item that does not carry the new code.
3. **The finding is the price of that guarantee, and it is small.** By the reading fixed beforehand there is no price at ranks 16 and 64 under either rule nor at rank 4 under back-propagation, and some price at rank 4 under the sweep. The gated term is slower than E40 at the start (below E40 in all 18 runs at row 64) and at or above E40 in 16 of 18 runs at rows 512 and 1,024.
4. **What the result does not show** is how a learner would arrive at this separation by itself. The gate is read by the harness from the task's code. Section 10 lists the limits.

## 3. The receipts and the checks (part A)

- Under `exp45/e48` there are 180 files: 18 run receipts, 18 answer files, seven check receipts, four dry-run files, the report with its tables, and 108 term files of 22,225,164 bytes.
- Every one of the 18 receipts is complete, has the eight rows 0, 16, 32, 64, 128, 256, 512 and 1,024, and records finite losses. In each, row 0 equals row 0 of E40's run of the same seed and rule on every score (check 2). At every row the digest of the answers to the six earlier panels equals row 0's, the six counts equal E40's row 0, and the hash of the lived network's tensors with the four code rows set back equals the life-end file's (check 3). The batch digest at rows 64 to 1,024 equals E40's (check 4): E48 saw E40's batches. No held-out pair, no untaught pair and no item of an earlier code was presented.
- The seven check receipts exist and pass. Check 3a records, at row 64 of the dry run, logits on two batches of earlier families that are bit-equal to the ungated forward's, and logits on a batch of the new family that differ by up to 18.2 (sweep) and 24.6 (back-propagation). Check 5 records a gradient of exactly zero on both `down` matrices at the first update and a non-zero one at the second, under both rules.
- The copy `e48_gated.py` on the worker's branch is the same git blob as the specification file on the default branch. The committed runner and the ten harness sources have the SHA-256 values that the receipts name, and the harness files are identical to those of the brief's commit f8f62df6.
- **The six required changes of the audit are in what ran:** the reading on R; the gate read from `x[:, 1]`, asserted equal for all rows of a call, installed in place of the harness's forward, with `keep_code_rows` between the backward pass and the step, and check 3a; the digests of the earlier answers at every row and the answers written at six rows; the term files as five plain float32 tensors; the comparison with a replay store on stream entries; and the statements on what the decay acts on.

## 4. The reader on the committed files (part B)

- **The life-end files.** For each of the six networks the file's SHA-256 equals the receipt's, and the hash of its tensors, recomputed in numpy by the harness's formula, equals the value recorded at the end of the life. With a term at zero the reader gives row 0 of every run: the 18 counts, the reduction answers item for item, and the digests of the six earlier panels' answers.
- **The 108 term files.** For every file the five tensors have the required shapes; the file's SHA-256 and size, the hash of its tensors, the hash of its code rows and the four norms are those of the receipt. With the term and its code rows placed on the life-end network the reader reproduces the 18 counts and the four summary scores of that row for 107 files. For the 108th (`e48_S_73129_r16_row16.pt`) one count differs by one item: 861 against 862 of 3,131 addition pairs without the reduction. On that item (`e41+40=`) the two largest outputs differ by 2.9 × 10⁻⁶; the next smallest difference on the panel is 2.0 × 10⁻³. The reader's arithmetic is numpy's and the runner's is torch's, and they round differently; the difference is a tie and not a disagreement.
- **The answers.** The bytes that the reader generates for the reduction pairs equal the worker's answer files on 515,396 of 515,396 items, over the 84 rows that have both an answer file and a term file.
- **The earlier panels.** The digests of the bytes generated for the six earlier panels, computed by the reader, equal the receipts' at each of the 108 rows.
- **The gate forced off.** The scores of the new family with the gate off at rows 256 and 1,024 are reproduced in 36 of 36 rows. The four code rows alone carry little: a reduction score of 0.5 to 11.9, near row 0's.
- **The sweep's cosines.** The cosine between each factor of a term and what the sweep uses in its place, recomputed from the term files, equals the receipts' in 216 of 216 cases; from row 64 on it lies between 0.76 and 0.88.

Rows 16, 32 and 64 of the rank-64 runs have no term file (the brief did not ask for one); their scores are taken from the receipts.

## 5. Keeping (part C)

### 5.1 The earlier families are answered exactly as before

For each of the six networks I took the largest term (rank 64, row 1,024) with its code rows and answered a wide set of earlier items: the twelve tasks of the birth family (modulus 89) and of the two families of the life (moduli 97 and 73) at 0, 8, 12, 20 and 40 letters, the composition panels, and every held-out addition and subtraction pair of the three families that needs the reduction, without letters. That is 37,458 to 37,507 items a network in 63 groups.

- The answers with the term loaded and the gate read from the code byte equal the life-end network's item for item, in all six networks. On 3,024 sampled items a network the logits are equal value for value.
- None of the bytes generated is `e`, `f`, `g` or `h`, so no earlier answer ever reads a freed row of the byte table.
- The scores are therefore unchanged: the families of the life stay at 88 to 99% on their panels, the birth family stays where the life left it (38 to 55% on its panels, under 1% on its reduction pairs).

### 5.2 The paths by which a term could reach an earlier item

| Path | Finding |
|---|---|
| The gate depends on the position | No. It is one flag for an item and holds at every position. |
| The gate depends on the hidden state | No. It is computed from the input bytes alone. |
| The gate depends on a byte other than the code | No. The runner reads `x[:, 1]` only. The bytes after the start symbol of every earlier item are `c n p q r s t u v w x y z`. |
| A call mixes items of the new family and others | The runner asserts that all rows of a call agree; a mixed call stops the run. |
| An earlier item reads a freed row of the byte table | The earlier items use 56 byte values, none of them 101 to 104. A generated byte is fed back as input, and no generated byte is one of the four (section 5.1). If one were, the answer would already be wrong at that byte, so no count could change. |
| The optimizer moves a frozen weight | The lived weights are not in the optimizer, except the byte table, whose gradient is zeroed outside the four rows and which has no decay. The receipts' hash at every row, and the reproduction of every row from the life-end file, agree with this. |
| State carried from one item to the next | None exists: the state starts at zero for every item. |

The specification's forward uses the gate in two lines, the two places where a term is added. Keeping is exact by construction.

### 5.3 The gate does the work: the same term applied to earlier items

Applied to earlier items with the gate forced on, the same terms change the answers. The table gives the percentage right for the life-end network (which is also the figure with the term loaded and the gate read from the code) and with the gate forced on.

| Network | Mod 97 panels | Mod 73 panels | Compositions | Mod 97 reduction pairs | Mod 73 reduction pairs | Mod 89 panels |
|---|---|---|---|---|---|---|
| Sweep 73129 | 94.3 → 82.8 | 87.9 → 75.8 | 69.9 → 13.8 | 88.7 → 51.6 | 65.9 → 22.1 | 48.7 → 57.9 |
| Sweep 93683 | 93.2 → 77.6 | 91.2 → 81.9 | 63.2 → 20.8 | 86.9 → 38.2 | 76.6 → 72.8 | 37.6 → 46.6 |
| Sweep 116089 | 95.2 → 65.8 | 93.6 → 82.1 | 62.8 → 11.1 | 92.3 → 1.0 | 80.4 → 61.4 | 42.0 → 53.7 |
| Back-propagation 73129 | 97.0 → 80.2 | 92.0 → 83.3 | 77.8 → 52.9 | 92.9 → 29.2 | 77.5 → 42.6 | 54.5 → 69.9 |
| Back-propagation 93683 | 98.5 → 76.8 | 96.3 → 94.0 | 90.9 → 56.6 | 98.4 → 18.0 | 93.7 → 82.6 | 44.7 → 61.7 |
| Back-propagation 116089 | 94.7 → 83.8 | 94.7 → 84.2 | 76.1 → 67.1 | 91.6 → 40.3 | 86.9 → 46.2 | 43.9 → 57.0 |

Two things follow.

- **The term is not harmless to the families of the life.** It lowers their panels by 2 to 29 points and their reduction pairs by 4 to 91 points. The separation is carried entirely by the gate; a gate that was wrong on some earlier items would cost on those items.
- **The term holds knowledge that an earlier family could use.** In each of the six networks the birth family's panels rise by 9 to 17 points with the gate forced on, and the rise is concentrated in one operation: `max` under modulus 89 goes from 4 to 52%, 27 to 96%, 14 to 94% and 4 to 59% in four networks, and `min` from 0 to 71% and 36 to 89% in the other two (0 or 12 letters). These are operations of the birth family that had fallen to low levels by the end of the life in that network; `max` and `min` do not depend on the modulus, and the term had to learn them for the new family. A gate that is on for one family only withholds this from the others.

### 5.4 The other side: what E40 paid

From E40's receipts, the percentage right on the six earlier panels (2,200 items at 12 letters) at rows 0, 64, 128, 256, 512, 1,024 and 8,192, with every weight free:

| E40 run | 0 | 64 | 128 | 256 | 512 | 1,024 | 8,192 |
|---|---|---|---|---|---|---|---|
| Sweep 73129 | 72.0 | 60.7 | 59.1 | 41.0 | 43.5 | 43.9 | 30.6 |
| Sweep 93683 | 69.0 | 35.7 | 34.0 | 39.6 | 36.2 | 30.3 | 15.7 |
| Sweep 116089 | 71.2 | 44.0 | 46.6 | 40.0 | 41.8 | 38.7 | 36.2 |
| Back-propagation 73129 | 76.0 | 56.0 | 55.8 | 46.4 | 42.8 | 43.5 | 32.7 |
| Back-propagation 93683 | 77.7 | 60.0 | 57.5 | 50.2 | 45.5 | 44.4 | 32.3 |
| Back-propagation 116089 | 76.0 | 54.5 | 54.5 | 49.5 | 47.0 | 46.5 | 40.7 |

On the four panels of the life's families alone, E40 goes from 87 to 97% at row 0 to 42 to 69% at row 64 and 38 to 49% at row 1,024. E48 is at row 0's figures at every row.

## 6. The reading, speed and level (part D)

### 6.1 The reading fixed beforehand

R of each E48 run, R of E40's run of the same seed and rule, and the difference. E40's values, recomputed from its receipts, are those of the brief.

| Rule | Rank | 73129 | 93683 | 116089 | Reading |
|---|---|---|---|---|---|
| Sweep | 4 | 66.89 / 82.65 / −15.75 | 47.72 / 78.93 / −31.22 | 54.10 / 81.45 / −27.36 | some price |
| Sweep | 16 | 80.27 / 82.65 / −2.38 | 70.20 / 78.93 / −8.74 | 73.30 / 81.45 / −8.15 | no price |
| Sweep | 64 | 83.81 / 82.65 / +1.16 | 83.92 / 78.93 / +4.99 | 80.48 / 81.45 / −0.97 | no price |
| Back-propagation | 4 | 73.79 / 78.58 / −4.79 | 83.40 / 91.86 / −8.47 | 91.33 / 91.11 / +0.22 | no price |
| Back-propagation | 16 | 82.44 / 78.58 / +3.86 | 89.95 / 91.86 / −1.91 | 95.63 / 91.11 / +4.52 | no price |
| Back-propagation | 64 | 87.89 / 78.58 / +9.31 | 93.27 / 91.86 / +1.41 | 97.39 / 91.11 / +6.28 | no price |

These are the worker's figures to two decimals.

**The count "16 of 18 by row 1,024"** is not in the worker's report; it is the comparison of single rows: the reduction score of the E48 run against that of E40's run of the same seed and rule at the same row. E48 is at or above E40 in 0 of 18 runs at row 64, 2 at row 128, 11 at row 256, 16 at row 512 and 16 at row 1,024. The two runs below at row 1,024 are the sweep at rank 4 on seeds 93683 (−0.6) and 116089 (−6.3).

**A correction to the wording of the reading.** The brief's sentence for *no price*, which follows my own requirement, says that the confined change "learns the new family as fast as a change to every weight". The single rows do not support "as fast": at row 64 every E48 run is below E40, by 6 to 59 points. The statement that the figures support is that the confined change reaches E40's level within 128 to 512 updates at ranks 16 and 64 and is above it afterwards.

### 6.2 Speed

First scored row at which a run is at or above a level, seeds 73129, 93683 and 116089. E40 has no rows 16 and 32.

| Run | Taught panel 90% | Reduction pairs 80% | Reduction pairs 90% | E40's R of the seed |
|---|---|---|---|---|
| E40, sweep | 64, 256, 64 | 128, 256, 128 | never in 8,192 updates (highest 85.3, 85.5, 88.4) | |
| E48, sweep, rank 4 | 256, 512, 512 | 512, 1,024, 1,024 | 1,024, never, never | 512, 1,024, 1,024 |
| E48, sweep, rank 16 | 256, 256, 256 | 256, 512, 512 | 1,024, 1,024, never (89.5) | 256, 512, 512 |
| E48, sweep, rank 64 | 128, 128, 128 | 256, 256, 256 | 1,024, 512, never (89.0) | 256, 256, 256 |
| E40, back-propagation | 64, 64, 64 | 64, 64, 64 | never (highest 87.2), 64, 64 | |
| E48, back-propagation, rank 4 | 256, 256, 128 | 512, 256, 128 | 1,024, 512, 256 | 512, 512, 512 |
| E48, back-propagation, rank 16 | 128, 128, 64 | 256, 128, 64 | 512, 256, 128 | 256, 256, 128 |
| E48, back-propagation, rank 64 | 64, 64, 32 | 128, 64, 32 | 512, 256, 128 | 128, 256, 128 |

On the reduction pairs at 80% the term is behind E40 by up to three rows (a factor of eight in updates) at rank 4, by up to two rows at rank 16 and by at most one row at rank 64.

The delay is not explained by the size of the change alone. The product is 1 to 3% of the size of the frozen matrix at row 16 and 4 to 10% at row 64, where E40's change of the same two matrices is 8 to 12%. E40 at that row has also changed every other matrix; in my earlier measurement on E40's row 64 (memo section 23) the recurrent layer's share of the change, taken alone, gave 57 to 65% on never-taught reduction pairs under the sweep and 68 to 88% under back-propagation, which is about the level of the rank-64 term at row 64 here (53 to 69% and 74 to 89%). No run varied the learning rate or the form of the term, so how much of the delay belongs to the confinement and how much to this way of writing the change is open.

### 6.3 Level

- **At row 1,024 the confined change is higher than E40.** The mean reduction score over seeds at ranks 4, 16 and 64 is 85.4, 91.4 and 91.1 under the sweep (E40 85.6) and 95.6, 96.6 and 97.0 under back-propagation (E40 90.2).
- **It is also higher than E40's best over 8,192 updates** at ranks 16 and 64 in all six networks (by 3.2 to 8.0 points in five, by 0.6 to 1.1 in the sixth, sweep 116089), and at rank 4 in four of six. The sampling error of one reduction score is about half a point. This comparison was not fixed beforehand, and E40 and E48 differ in two things, where the change is made and whether the lived weights decay, so the gain cannot be attributed to the confinement alone.
- **The curves are even.** From row 128 on, E40's reduction score falls between neighbouring rows in 6 of 18 steps, by up to 19.1 points; E48's falls in 1 of 54 steps, by 0.7.

### 6.4 The ranks and the rules

- **Ranks.** The reduction score rises with the rank in 39 of the 42 cells (rule, seed, row from 16 on). The three exceptions are at rows 512 and 1,024 under the sweep, where ranks 16 and 64 are within 3 points of each other. The advantage of rank is large early (row 64, mean over seeds: 16.4, 39.3 and 58.4 under the sweep; 54.5, 71.0 and 81.3 under back-propagation) and nearly gone by row 1,024.
- **Rules.** Back-propagation is ahead of the sweep at every rank. R under the sweep is 66.9, 47.7, 54.1 at rank 4, 80.3, 70.2, 73.3 at rank 16 and 83.8, 83.9, 80.5 at rank 64; under back-propagation 73.8, 83.4, 91.3, then 82.4, 89.9, 95.6, then 87.9, 93.3, 97.4. Measured against E40, the sweep pays 11 to 28 points more than back-propagation for the confinement at rank 4, 6 to 13 points more at rank 16, and between 8 more and 4 less at rank 64. At row 1,024 the sweep is 5 to 6 points below back-propagation at ranks 16 and 64, which is about the distance between the two rules in E40 itself at that row (85.6 against 90.2).

## 7. The predictions (part E)

| | Prediction | Chance | Outcome | Why a miss |
|---|---|---|---|---|
| P48.1 | Checks 2 and 3 pass for all six networks | 90% | true | |
| P48.2 | Rank 16, back-propagation: no price | 50% | true (+3.86, −1.91, +4.52) | |
| P48.3 | Rank 16, sweep: no price | 35% | true (−2.38, −8.74, −8.15) | |
| P48.4 | Rank 64, back-propagation: no price | 65% | true (+9.31, +1.41, +6.28) | |
| P48.5 | Rank 4: some price or a price under both rules | 65% | **false** | Under back-propagation rank 4 reads *no price* (−4.79, −8.47, +0.22). Rank 4 is slow at row 128 (57.8, 70.2 and 86.1 against E40's 87.2, 94.7 and 92.8) but is above E40 at row 512 in all three seeds, and E40's own row 256 falls to 68.1 in seed 73129; the mean of the three rows hides the early delay. The prediction underrated how little rank back-propagation needs. |
| P48.6 | The difference from E40 is lower under the sweep by 10 points or more, two of three seeds, at two of three ranks | 40% | **false** | It holds at rank 4 (−11.0, −22.8, −27.6) and fails at rank 16 (−6.2, −6.8, −12.7: one seed) and at rank 64 (−8.2, +3.6, −7.3). The sweep's extra price falls with the rank faster than predicted. |
| P48.7 | Mean of rows 512 and 1,024 at rank 16 within 10 points of E40's, two of three seeds, each rule | 55% | true (all six above E40, by 3.7 to 8.6) | |

Five of seven are true, as the worker scored them. The mean squared error of the stated chances is 0.227, against 0.25 for a chance of one half throughout: the chances carried little information.

## 8. Cost, directions used, and the wrong answers (part F)

**Cost in weights.** The family's taught set is 689 pairs an operation, 2,756 entries of three bytes, 8,268 bytes.

| Rank | Numbers that may change | Share of the network's 1,124,113 | Bytes at 4 a number | Times the taught set |
|---|---|---|---|---|
| 4 | 9,248 | 0.82% | 36,992 | 4.5 |
| 16 | 36,896 | 3.28% | 147,584 | 17.9 |
| 64 | 147,488 | 13.12% | 589,952 | 71.4 |

**Directions used.** The trained products use fewer directions than their rank allows. At row 1,024 the number of singular values at or above a tenth of the largest is 3 to 4 at rank 4, 5 to 7 (input map) and 9 to 12 (recurrent matrix) at rank 16, and 6 to 9 and 18 to 24 at rank 64. The size of the product is 0.13 to 0.39 of the frozen matrix's.

**What the wrong answers are** (never-taught reduction pairs, three seeds together, from the worker's answer files).

- At row 0, before any update, 97.2% (sweep) and 94.5% (back-propagation) of the answers are wrong. Of the wrong answers 45% and 15% are not two digits and the end symbol, and 9% and 26% are the right answer under modulus 97, the modulus of the life's first family and of its compositions.
- At row 1,024, 14.6%, 8.6% and 8.9% of the answers are wrong under the sweep at ranks 4, 16 and 64, and 4.4%, 3.4% and 3.0% under back-propagation. Almost all are two digits and the end symbol. Between 71 and 82% of the wrong answers differ from the right answer by exactly 1 or exactly 10, that is, one digit is off by one.
- The errors by 10 go both ways (19 to 34% of the wrong answers towards the value that modulus 73 would give, 25 to 38% the other way). They are therefore errors in the tens digit and not answers under an earlier family's modulus. Answers under modulus 97 are about 1% of the wrong answers or less from row 64 on.

## 9. The gate and the position (part G)

The gate is on from the start symbol, one position before the code byte has been read. A learner that produced its own gate could not have it that early. With nothing retrained, I applied the terms of row 1,024 from the code byte's position on, and from the position after it on. The reduction score, as run / from the code byte / from the position after it:

| Rule, rank | 73129 | 93683 | 116089 |
|---|---|---|---|
| Sweep, 16 | 92.2 / 92.1 / 78.1 | 92.5 / 92.4 / 72.5 | 89.5 / 89.6 / 78.8 |
| Sweep, 64 | 90.8 / 91.0 / 75.1 | 93.5 / 93.5 / 79.5 | 89.0 / 88.9 / 85.3 |
| Back-propagation, 16 | 93.1 / 93.1 / 89.9 | 97.9 / 98.0 / 81.7 | 98.7 / 98.7 / 88.1 |
| Back-propagation, 64 | 94.0 / 94.1 / 91.8 | 98.3 / 98.4 / 86.8 | 98.8 / 98.7 / 95.6 |

The result does not rest on the gate being on before the code is read: leaving the term off at the start symbol changes the score by at most 0.2 points. A gate that is available only one position after the code byte costs 2 to 20 points on these terms, which were trained with the earlier gate.

## 10. What the result shows, and its limits

**What it shows.** E48 does not test whether the learner keeps its earlier knowledge; the construction guarantees it. It tests the price of the guarantee, and three statements follow.

1. In these networks a new family can be learned to E40's level, and above it, without changing any connection that an earlier family uses. E40 learned the same family by changing every weight and lost 38 to 43 points on the four panels of the life's families within 256 updates. That loss was therefore not required for the learning; it is a consequence of where an unconfined rule places the change, not of a lack of room in the network.
2. The frozen lived network is a sufficient base: an addition of 0.8% (back-propagation) to 3.3% (both rules) of the network's numbers is enough, and the trained additions use few directions.
3. The sweep can train such a term, at a price that is large at rank 4 and small from rank 16.

**The limits.**

- **The gate is given.** The harness reads it from the task's code byte. Nothing in the learner decides that an item belongs to a new family, that a new term is needed, or which connections the term should occupy. E48 is an upper bound on what separation can give, not a demonstration that a learner can separate.
- **One new family.** No second family was added. Nothing is said about many families in sequence.
- **Memory is not fixed.** Each family adds a term: 36,896 numbers at rank 16, so that 30 families equal the network, and 9,248 at rank 4 (122 families). At every rank the term takes more memory than the family's taught set (4.5 to 71 times). Under fixed memory the terms would have to be bounded, merged or reused, and E48 does not say how. The fair comparison is with a replay store of equal size, which belongs to the long-life experiment.
- **Families that should share knowledge.** The gate is on for one family at a time. What a term learns is withheld from every other family (section 5.3 measures a case in which another family would gain), and the new family cannot be composed with the others, because composition items carry the code `c` and run with the gate off.
- **Continued learning.** A term is kept exactly only while it is out of the optimizer, as it is here after the last update. Under the standing decay and without its family's items it would shrink.
- **Two differences from E40.** E48 confines the change and also removes the decay from the lived weights. The comparison cannot separate the two.
- **The rate and the form of the change** were not varied; part of the delay at the start may belong to them (section 6.2).
- **The readout is frozen,** so a task whose answers need new output units is out of reach. Three seeds, one family, one item format.

## 11. Commands, durations, and what was not verified

All from the repository root, with `python3 -I`; one thread; the host was shared with other jobs.

| Command | Content | Duration |
|---|---|---|
| `python3 -I PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/e48_result_verify.py A` | receipts, check receipts, sources, the six changes | 2 s |
| `… e48_result_verify.py B` | the reader on the 108 term files | 24 min |
| `… e48_result_verify.py B file=e48_S_73129_r16_row16` | the one file with the rounding tie | 40 s |
| `… e48_result_verify.py B quick` | rows 256 and 1,024 of the six rank-16 runs | about 4 min |
| `… e48_result_verify.py C` | keeping, six networks at rank 64 | 12 min |
| `… e48_result_verify.py C quick` | keeping, two networks at rank 16 | 5 min |
| `… e48_result_verify.py D E` | the reading, speed and level, the predictions | 2 s |
| `… e48_result_verify.py F` | cost, directions, sizes, wrong answers | 5 s |
| `… e48_result_verify.py G` | the gate and the position | 85 s |

**Not verified.**

- Equality with torch's arithmetic bit for bit (checks 1 and 3a of the worker). The reader reproduces generated bytes and counts, not torch's logits to the last bit; among the roughly 1.8 million answers behind the compared counts, one differs through a rounding tie.
- The lived network after training was not saved (the brief did not ask for it). That its tensors are unchanged is recorded by the worker's hash at every row; I confirm it indirectly, since every row is reproduced from the life-end file and the term file.
- The scores of the rank-64 runs at rows 16, 32 and 64, for which no term file was committed, and the CPU times.
- That the stream holds no held-out pair, beyond the receipts' counters and the equality of the batch digests with E40's.
