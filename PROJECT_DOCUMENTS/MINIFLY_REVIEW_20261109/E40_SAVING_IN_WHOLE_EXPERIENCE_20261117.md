# E40: the lived learner's saving, counted in its whole experience (17 November 2026)

Plan item 26 of `PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/COORDINATION_20261117/PLAN.md` (branch `claude/vibrant-cerf-sje726`). A read-only computation from the record. No network was trained, loaded or run, and no weights file was opened. Every file of the record was read with `git show origin/codex/minifly-consolidated-20261023:<path>` at commit `7cabb13adc38753483d34dd5472fc3c84c4de675`, unless another ref is named. The script is `e40_whole_experience.py` beside this file. Sections 4, 5, 6.1, 6.2, 7.1, 7.2, 7.3 and the appendix are its output, unchanged; the other sections are written by hand from those tables and from the files named in them.

## 1. The question

The record states the saving of a learner that has lived in pairs and items of the new family only. E40's report (`MOD97_LOCAL_RULE_20261106/E40_REPORT.md`) gives it at equal experience of the new family: the same taught pairs and the same items for a lived network and for a fresh one. E37's report says so in its limits: "the saving is in items of the new family, not in total experience" (`MOD97_LOCAL_RULE_20261106/exp45/e37/E37_REPORT.md`).

Mr. Xu's question of 17 November (item B2 of the plan; restated in `COORDINATION_20261117/TASK_E54_THE_LIMITED_AND_THE_PERIODIC_NETWORK_WITH_FEWER_PAIRS_20261117.md`) is whether the learner "needs far fewer samples". That file gives the programme's sense of the words: "a sample is a taught pair, and 'fewer samples' means that a smaller taught fraction is enough for the untaught pairs". This document counts the saving in everything a lived network was given, and sets three accounts side by side:

- **(a)** the new family alone, which is what the record states;
- **(b)** the new family and everything before it;
- **(c)** the number of further new families after which the lived learner's whole experience would be smaller than that of fresh networks that each learn one family, if each further family cost what E40's did.

## 2. Terms

- **Family.** Four operations (add, sub, max, min) on two-digit operands under one modulus. Each operation of a family has its own code byte, which opens every item of that operation.
- **Pair.** An ordered pair of operands (a, b), each from 0 to the modulus less one. An operation of a family of modulus m has m × m pairs.
- **Taught, never taught, taught fraction.** A family is taught from a fixed share of the pairs of each operation, the taught fraction. The other pairs are never presented, and the family is scored on them.
- **Distinct taught pair (a sample).** One entry of a family's teaching list: an operation of that family with one pair (a, b). The same operands under the same operation in another family are another sample, because the code byte and the modulus differ. Section 4.2 says how many such operands recur.
- **Composition.** One entry of the second family of a life: two operators and three operands under modulus 97, answered with the intermediate and the final result. Compositions are counted apart from pairs.
- **Item.** One presentation of an entry, rendered with 0 to 12 capital letters between the operands, drawn afresh each time. One sample gives many different items.
- **Update.** One optimizer step on a batch of 64 items. **Presented items** are updates times 64.
- **Row.** A point of a run at which it was scored. A level can only be located at a row.
- **Reduction pair.** A pair whose answer needs the modulus: for add, a + b at or above the modulus; for sub, a − b below zero. They are about half of the pairs of add and of sub: 3,069 of the 6,200 never-taught pairs of add under modulus 83 for seed 73129 (section 5).
- **Reduction score (E40), accuracy R (E41).** The share of whole answers right, with the end symbol, on never-taught reduction pairs of add and of sub rendered without letters, mean of the two operations. E40 scores every such pair; E41 scores a fixed sample of 1,000 of each operation.
- **Reached.** A run reaches a level at the first row at which its score is at or above that level. Where a later row is below it again, the tables show it.
- **Birth.** A network trained from random weights on the first family (modulus 89) alone. **Life.** Six further blocks of updates on other families, started from a birth. **Lived network.** A network at the end of its life. **Fresh network.** A network with random weights, before any update.
- **The two rules.** Back-propagation (B) and the sweep (S), an approximate rule that makes the same forward pass and uses the same loss and optimizer. **Arm K** is the sweep with a learned feedback matrix on the recurrent map; its life starts from the sweep's birth. E40 sets it beside the fresh sweep network, and so does this document.
- **Present width.** Feedforward width 512 and 256 recurrent units, 1,124,113 parameters: the size of every network of E26 and E40.
- **Seed.** 73129, 93683 or 116089. A seed fixes the starting weights, the taught pairs and the order of the items.

## 3. How the figures were obtained

- **The schedule and the taught fractions** are read from the code. The script holds a copy of each rule it uses and stops unless the line that states the rule is in the source file at the commit (appendix).
- **Counts of distinct samples** are computed by the script's copy of the split and of the composition triples. The copy is checked against five things the receipts hold (section 4.2), among them the sha256 of E41's samples of reduction pairs, which depend on the exact split.
- **Updates and presented items** are read from the receipts: the optimizer's step count, the count of batches in the batch digest, and for a life the count of items by family in each block.
- **Levels** are read from the scores at the rows of the receipts. Nothing was rescored.
- **Chain of files.** The script checks by sha256 that each lived run of E40 starts from the life file its receipt names, and each life from its birth file.
- **Not recomputed.** The figures quoted from E44's report in section 7.4 are the worker's.

## 4. What a lived network was given before the new family

### 4.1 Updates and presented items

Receipts: `exp45/e26/e26_dev_<S|B>_<seed>.json` (births), `exp45/e26/e26_life_<S|B>_<seed>.json` and `exp45/e31/e31_life_K_<seed>.json` (lives). Checked here: each life names its birth file by sha256; the sweep's life and the life of arm K (the sweep with learned feedback) both start from the sweep's birth; the batch digest of the birth is equal under both rules, and the batch digest and the count of items by family of every life block are equal under the three rules. So the counts below hold for every rule and differ only by seed.

Rows the receipts have: a birth is scored at 4,096, 8,192, 12,288, 16,384, 20,480 and 24,576 updates; a life before its first block and after each of its six blocks.

| Part | Stream | Updates | Presented items | Items by family, seed 73129 | seed 93683 | seed 116089 |
|---|---|---|---|---|---|---|
| Birth | first family (modulus 89) alone | 24,576 | 1,572,864 | 89: 1,572,864 | 89: 1,572,864 | 89: 1,572,864 |
| Life block 1 (F1.1) | F1 alone | 6,144 | 393,216 | F1: 393,216 | F1: 393,216 | F1: 393,216 |
| Life block 2 (F1.2) | F1 alone | 6,144 | 393,216 | F1: 393,216 | F1: 393,216 | F1: 393,216 |
| Life block 3 (F2.1) | F1 and F2, an item from either with equal chance | 6,144 | 393,216 | F1: 196,858; F2: 196,358 | F1: 196,679; F2: 196,537 | F1: 196,723; F2: 196,493 |
| Life block 4 (F2.2) | as block 3 | 6,144 | 393,216 | F1: 196,472; F2: 196,744 | F1: 196,746; F2: 196,470 | F1: 196,561; F2: 196,655 |
| Life block 5 (F3.1) | F1, F2 and F3, an item from any with equal chance | 6,144 | 393,216 | F1: 131,234; F2: 130,903; F3: 131,079 | F1: 131,455; F2: 130,630; F3: 131,131 | F1: 131,280; F2: 130,986; F3: 130,950 |
| Life block 6 (F3.2) | as block 5 | 6,144 | 393,216 | F1: 130,891; F2: 131,284; F3: 131,041 | F1: 131,235; F2: 131,376; F3: 130,605 | F1: 131,297; F2: 131,048; F3: 130,871 |
| Life, six blocks |  | 36,864 | 2,359,296 | F1: 1,441,887; F2: 655,289; F3: 262,120 | F1: 1,442,547; F2: 655,013; F3: 261,736 | F1: 1,442,293; F2: 655,182; F3: 261,821 |
| **Birth and life** |  | **61,440** | **3,932,160** |  |  |  |

F1 is the family of modulus 97, F2 the compositions, F3 the family of modulus 73. The first family is not in a life's stream.

### 4.2 Distinct taught samples

Counting rule. A family's teaching list is `exp11_cell.split(task, fraction, seed)[0]` for each of its four operations: the first `round(fraction × modulus²)` pairs of a seeded shuffle (Python's `round`, which sends a half to the even number). The compositions' list is 2,000 operand triples for each of the 12 taught operator pairs. The count of distinct samples is the length of these lists. It is the same for every seed and rule. Every entry of every list was presented at least once, because a stream passes through its whole list in shuffled order before it repeats and each stream ran for more items than its list is long (last column).

Checked for seeds 73129, 93683, 116089: this file's split gives E40's count of taught pairs (`exp45/e40/checks/E40_check5.json`), the counts of never-taught pairs with and without the reduction at row 0 of `e40_lived_S_<seed>.json`, the sha256 of E41's two samples of 1,000 reduction pairs at the three fractions (`e41_S_h512_f<10|25|50>_<seed>.json`), and, with the composition triples, the eight `strict_panel_counts` in the life receipts; and E35-A's split count at a tenth (792 taught, 7,129 not).

| Family | Modulus | Pairs per operation | Taught fraction | Taught pairs per operation | Distinct taught samples (four operations) | Items presented (three seeds) | Times each sample was shown, on average |
|---|---|---|---|---|---|---|---|
| First family (birth) | 89 | 7,921 | 0.50 | 3,960 | 15,840 | 1,572,864 | 99.3 |
| F1 (life) | 97 | 9,409 | 0.25 | 2,352 | 9,408 | 1,441,887 to 1,442,547 | 153.3 |
| F2 (life): compositions under modulus 97 | 97 | 97³ = 912,673 triples for each of 12 operator pairs |  | 2,000 triples per operator pair | **24,000 compositions** | 655,013 to 655,289 | 27.3 |
| F3 (life) | 73 | 5,329 | 0.25 | 1,332 | 5,328 | 261,736 to 262,120 | 49.1 to 49.2 |
| **Before the new family** |  |  |  |  | **30,576 pairs and 24,000 compositions** |  |  |
| New family (E40), whole run of 8,192 updates | 83 | 6,889 | 0.10 | 689 | **2,756** | 524,288 | 190.2 |

The same operands under the same operation recur across families (for max and min, and for add and sub without reduction, with the same answer). The accounts below do not merge them. For the record, by seed:

| Seed | Earlier single-operation entries, by family | The same, merged over families by (operation, a, b) | Distinct single-operation steps (operation, a, b) inside the 24,000 compositions | of these, not in F1's taught quarter | New family's taught pairs whose (operation, a, b) an earlier family taught | the same, or a composition step |
|---|---|---|---|---|---|---|
| 73129 | 30,576 | 23,274 | 27,052 | 20,321 | 1,901 of 2,756 | 2,516 of 2,756 |
| 93683 | 30,576 | 23,249 | 27,055 | 20,250 | 1,916 of 2,756 | 2,523 of 2,756 |
| 116089 | 30,576 | 23,315 | 27,102 | 20,371 | 1,908 of 2,756 | 2,495 of 2,756 |

## 5. The new family (E40): updates and items to a level

Receipts: `exp45/e40/e40_<lived|fresh>_<S|B|K>_<seed>.json`. Rows the receipts have: 0, 64, 128, 256, 512, 1,024, 2,048, 4,096 and 8,192 updates (0 to 524,288 items), and no row between them, so a level reached at row 64 was reached at or before 4,096 items. Checked here: each lived run starts from the life file its receipt names (sha256 equal to the life receipt's); every row's presented items equal 64 times its updates and none is a never-taught pair; the batch digest is equal across the five runs of a seed at every row.

A level is *reached* at the first row at which the score is at or above it. Cells give updates, with presented items in brackets. The reduction score is the share of whole answers right on all never-taught reduction pairs of add and of sub (3,055 to 3,078 pairs each), no letters, mean of the two. The taught panel is 300 taught pairs of add and of sub at 12 letters, mean of the two. A lived network answers part of the new family before any update, so its row 0 is not zero.

**On never-taught reduction pairs**

| Run | Seed | Row 0 | 50% | 80% | 90% | Range over the rows after the 80% row | Row 8,192 | Highest (row) |
|---|---|---|---|---|---|---|---|---|
| lived S | 73129 | 6.1 | 64 (4,096) | 128 (8,192) | not reached | 77.0 to 84.5 | 77.0 | 85.3 (128) |
| lived S | 93683 | 2.2 | 64 (4,096) | 256 (16,384) | not reached | 82.1 to 85.5 | 82.1 | 85.5 (4,096) |
| lived S | 116089 | 0.1 | 64 (4,096) | 128 (8,192) | not reached | 79.7 to 88.4 | 83.7 | 88.4 (1,024) |
| lived B | 73129 | 4.4 | 64 (4,096) | 64 (4,096) | not reached | 68.1 to 87.2 | 85.0 | 87.2 (128) |
| lived B | 93683 | 0.5 | 64 (4,096) | 64 (4,096) | 64 (4,096) | 88.8 to 94.7 | 94.5 | 94.7 (128) |
| lived B | 116089 | 11.6 | 64 (4,096) | 64 (4,096) | 64 (4,096) | 89.3 to 92.8 | 89.3 | 95.4 (64) |
| lived K | 73129 | 8.7 | 64 (4,096) | 64 (4,096) | not reached | 79.2 to 89.6 | 79.2 | 89.6 (512) |
| lived K | 93683 | 2.7 | 64 (4,096) | 64 (4,096) | not reached | 80.6 to 89.5 | 89.5 | 89.5 (8,192) |
| lived K | 116089 | 7.2 | 64 (4,096) | 64 (4,096) | not reached | 83.8 to 87.3 | 83.8 | 87.3 (4,096) |
| fresh S | 73129 | 0.0 | not reached | not reached | not reached | — | 6.0 | 6.0 (8,192) |
| fresh S | 93683 | 0.0 | not reached | not reached | not reached | — | 7.6 | 7.6 (8,192) |
| fresh S | 116089 | 0.0 | not reached | not reached | not reached | — | 6.4 | 6.4 (8,192) |
| fresh B | 73129 | 0.0 | not reached | not reached | not reached | — | 7.7 | 7.7 (8,192) |
| fresh B | 93683 | 0.0 | not reached | not reached | not reached | — | 11.8 | 11.8 (8,192) |
| fresh B | 116089 | 0.0 | not reached | not reached | not reached | — | 8.3 | 8.3 (8,192) |

**On the taught panel**

| Run | Seed | Row 0 | 50% | 80% | 90% | Range over the rows after the 80% row | Row 8,192 | Highest (row) |
|---|---|---|---|---|---|---|---|---|
| lived S | 73129 | 24.2 | 64 (4,096) | 64 (4,096) | 64 (4,096) | 91.8 to 99.5 | 99.0 | 99.5 (4,096) |
| lived S | 93683 | 15.5 | 64 (4,096) | 64 (4,096) | 256 (16,384) | 85.5 to 99.5 | 99.5 | 99.5 (8,192) |
| lived S | 116089 | 30.0 | 64 (4,096) | 64 (4,096) | 64 (4,096) | 92.8 to 99.7 | 99.7 | 99.7 (8,192) |
| lived B | 73129 | 41.0 | 64 (4,096) | 64 (4,096) | 64 (4,096) | 91.3 to 99.5 | 98.5 | 99.5 (4,096) |
| lived B | 93683 | 29.8 | 64 (4,096) | 64 (4,096) | 64 (4,096) | 97.3 to 100.0 | 100.0 | 100.0 (8,192) |
| lived B | 116089 | 53.3 | 0 (0) | 64 (4,096) | 64 (4,096) | 97.3 to 99.0 | 99.0 | 99.0 (64) |
| lived K | 73129 | 40.5 | 64 (4,096) | 64 (4,096) | 64 (4,096) | 95.3 to 99.5 | 99.5 | 99.5 (8,192) |
| lived K | 93683 | 28.3 | 64 (4,096) | 64 (4,096) | 64 (4,096) | 94.3 to 100.0 | 100.0 | 100.0 (8,192) |
| lived K | 116089 | 40.5 | 64 (4,096) | 64 (4,096) | 64 (4,096) | 95.5 to 99.7 | 99.2 | 99.7 (4,096) |
| fresh S | 73129 | 0.0 | 2,048 (131,072) | 4,096 (262,144) | 4,096 (262,144) | 97.8 to 97.8 | 97.8 | 97.8 (8,192) |
| fresh S | 93683 | 0.0 | 2,048 (131,072) | 2,048 (131,072) | 4,096 (262,144) | 98.0 to 98.7 | 98.0 | 98.7 (4,096) |
| fresh S | 116089 | 0.0 | 2,048 (131,072) | 2,048 (131,072) | 4,096 (262,144) | 97.0 to 98.8 | 98.8 | 98.8 (8,192) |
| fresh B | 73129 | 0.0 | 2,048 (131,072) | 4,096 (262,144) | 4,096 (262,144) | 98.5 to 98.5 | 98.5 | 98.5 (8,192) |
| fresh B | 93683 | 0.0 | 2,048 (131,072) | 2,048 (131,072) | 4,096 (262,144) | 98.0 to 99.7 | 99.7 | 99.7 (8,192) |
| fresh B | 116089 | 0.0 | 2,048 (131,072) | 4,096 (262,144) | 4,096 (262,144) | 98.8 to 98.8 | 98.8 | 98.8 (8,192) |

The first row after row 0 is at 4,096 items, more than the 2,756 taught pairs of the new family, and the stream shows every taught pair once before it repeats any. So at every row after row 0 the run had been shown all 2,756 distinct taught pairs: the count of distinct pairs to a level is 2,756 for every lived run, and the receipts cannot show whether fewer would do.

Never-taught reduction pairs scored (add, sub): seed 73129: 3,069, 3,074; seed 93683: 3,078, 3,061; seed 116089: 3,070, 3,055.

At row 8,192 the nine lived runs are at 77.0 to 94.5 and the six fresh runs at 6.0 to 11.8 on the reduction score.

The earlier families during the new one (never-taught add and sub at 12 letters, mean of the two, row 0 → row 8,192):

| Run | Seed | Modulus 89 | Modulus 97 | Modulus 73 |
|---|---|---|---|---|
| lived S | 73129 | 33.0 → 25.5 | 92.8 → 21.1 | 80.4 → 43.9 |
| lived S | 93683 | 17.7 → 10.8 | 91.0 → 17.9 | 85.5 → 17.1 |
| lived S | 116089 | 13.5 → 32.2 | 93.8 → 25.6 | 91.9 → 49.9 |
| lived B | 73129 | 37.3 → 31.0 | 95.4 → 30.1 | 85.6 → 36.6 |
| lived B | 93683 | 26.7 → 15.0 | 98.5 → 39.5 | 95.1 → 38.0 |
| lived B | 116089 | 36.3 → 38.2 | 91.4 → 33.5 | 90.3 → 49.7 |
| lived K | 73129 | 31.2 → 37.0 | 95.0 → 32.6 | 83.6 → 48.1 |
| lived K | 93683 | 36.5 → 19.8 | 98.0 → 32.4 | 89.4 → 14.4 |
| lived K | 116089 | 29.5 → 38.8 | 95.1 → 32.4 | 90.4 → 48.0 |

Over the nine lived runs: modulus 97, 91.0 to 98.5 at row 0 and 17.9 to 39.5 at row 8,192; modulus 73, 80.4 to 95.1 at row 0 and 14.4 to 49.9 at row 8,192; modulus 89, 13.5 to 37.3 at row 0 and 10.8 to 38.8 at row 8,192.
The first family (modulus 89) had fallen during the life, in which it is not taught: 93.8 to 99.8 before the life and 13.5 to 37.3 after it on the same panel (life receipts, stages 0 and 6).

## 6. A fresh network that learns one family

### 6.1 The first family at three taught fractions (E41, present width)

Receipts: `exp45/e41/e41_<S|B>_h512_f<50|25|10>_<seed>.json` (feedforward width 512, 256 recurrent units, 1,124,113 parameters: the width of every network of E26 and E40). Rows the receipts have: every 256 updates from 0 to 12,288 (0 to 786,432 items). Accuracy R is the share of whole answers right on a fixed sample of 1,000 never-taught reduction pairs of add and of sub, no letters, mean of the two.

| Taught fraction | Distinct taught pairs | Rule | Seed | R at 50% | R at 80% | R at 90% | R at 80% on three rows running | R at row 12,288 | Highest R | Taught panel at 80% | Taught panel at 90% |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.50 | 15,840 | S | 73129 | 3,840 (245,760) | 6,656 (425,984) | 12,288 (786,432) | 6,656 (425,984) | 90.2 | 90.2 | 4,864 (311,296) | 5,632 (360,448) |
| 0.50 | 15,840 | S | 93683 | 3,840 (245,760) | 4,864 (311,296) | 6,400 (409,600) | 4,864 (311,296) | 96.2 | 96.2 | 4,352 (278,528) | 5,120 (327,680) |
| 0.50 | 15,840 | S | 116089 | 3,072 (196,608) | 4,608 (294,912) | 8,192 (524,288) | 6,400 (409,600) | 92.7 | 94.8 | 4,096 (262,144) | 4,608 (294,912) |
| 0.50 | 15,840 | B | 73129 | 3,840 (245,760) | 5,120 (327,680) | 7,936 (507,904) | 6,400 (409,600) | 92.0 | 95.2 | 5,120 (327,680) | 5,120 (327,680) |
| 0.50 | 15,840 | B | 93683 | 3,584 (229,376) | 4,096 (262,144) | 5,120 (327,680) | 4,096 (262,144) | 94.6 | 96.6 | 3,840 (245,760) | 4,096 (262,144) |
| 0.50 | 15,840 | B | 116089 | 2,816 (180,224) | 3,584 (229,376) | 6,912 (442,368) | 5,120 (327,680) | 96.2 | 96.2 | 3,072 (196,608) | 4,352 (278,528) |
| 0.25 | 7,920 | S | 73129 | 3,840 (245,760) | 10,752 (688,128) | not reached | not reached | 80.3 | 83.8 | 3,840 (245,760) | 4,352 (278,528) |
| 0.25 | 7,920 | S | 93683 | 3,840 (245,760) | 11,264 (720,896) | not reached | not reached | 76.6 | 81.5 | 3,584 (229,376) | 4,352 (278,528) |
| 0.25 | 7,920 | S | 116089 | 4,096 (262,144) | not reached | not reached | not reached | 73.5 | 75.2 | 3,584 (229,376) | 4,864 (311,296) |
| 0.25 | 7,920 | B | 73129 | 3,072 (196,608) | 5,632 (360,448) | not reached | 7,680 (491,520) | 86.8 | 88.0 | 3,328 (212,992) | 4,096 (262,144) |
| 0.25 | 7,920 | B | 93683 | 3,072 (196,608) | 3,584 (229,376) | 4,864 (311,296) | 4,352 (278,528) | 89.2 | 93.9 | 3,584 (229,376) | 3,584 (229,376) |
| 0.25 | 7,920 | B | 116089 | 2,816 (180,224) | 4,864 (311,296) | 10,752 (688,128) | 5,888 (376,832) | 83.2 | 90.9 | 3,328 (212,992) | 4,352 (278,528) |
| 0.10 | 3,168 | S | 73129 | not reached | not reached | not reached | not reached | 7.3 | 8.0 | 2,560 (163,840) | 2,816 (180,224) |
| 0.10 | 3,168 | S | 93683 | not reached | not reached | not reached | not reached | 10.6 | 11.4 | 2,560 (163,840) | 2,816 (180,224) |
| 0.10 | 3,168 | S | 116089 | not reached | not reached | not reached | not reached | 8.3 | 9.3 | 2,304 (147,456) | 2,560 (163,840) |
| 0.10 | 3,168 | B | 73129 | not reached | not reached | not reached | not reached | 15.3 | 16.9 | 2,560 (163,840) | 3,072 (196,608) |
| 0.10 | 3,168 | B | 93683 | not reached | not reached | not reached | not reached | 29.6 | 30.6 | 2,560 (163,840) | 2,816 (180,224) |
| 0.10 | 3,168 | B | 116089 | not reached | not reached | not reached | not reached | 22.6 | 23.1 | 2,304 (147,456) | 2,560 (163,840) |

Checked here: at fraction 0.50 each run's batch digest at row 4,096 equals that of the E26 birth of the same rule and seed, and its counts on E26's own panels at rows 4,096 and 8,192 equal the birth receipt's. So the fresh reference at a half is the lived network's own birth, scored every 256 updates instead of every 4,096.

At a tenth no run reaches 50% on R within 12,288 updates; the highest R of the six runs is 8.0 to 30.6.

### 6.2 Fresh references on other families, and longer ones

| Run | Taught fraction | Distinct taught pairs | Rule | Seed | Never-taught add and sub at 12 letters, mean, by updates (items) | 80% on that panel |
|---|---|---|---|---|---|---|
| E37, F3 (modulus 73) alone | 0.25 | 5,328 | S | 73129 | 1,024 (65,536): 2.5; 2,048 (131,072): 7.2; 4,096 (262,144): 36.5 | not reached |
| E37, F3 (modulus 73) alone | 0.25 | 5,328 | S | 93683 | 1,024 (65,536): 1.6; 2,048 (131,072): 10.9; 4,096 (262,144): 47.1 | not reached |
| E37, F3 (modulus 73) alone | 0.25 | 5,328 | S | 116089 | 1,024 (65,536): 3.2; 2,048 (131,072): 16.9; 4,096 (262,144): 53.5 | not reached |
| E37, F3 (modulus 73) alone | 0.25 | 5,328 | B | 73129 | 1,024 (65,536): 1.6; 2,048 (131,072): 15.2; 4,096 (262,144): 64.2 | not reached |
| E37, F3 (modulus 73) alone | 0.25 | 5,328 | B | 93683 | 1,024 (65,536): 1.4; 2,048 (131,072): 13.4; 4,096 (262,144): 59.6 | not reached |
| E37, F3 (modulus 73) alone | 0.25 | 5,328 | B | 116089 | 1,024 (65,536): 3.5; 2,048 (131,072): 18.5; 4,096 (262,144): 73.5 | not reached |
| E37, F1 (modulus 97) alone | 0.25 | 9,408 | S | 73129 | 2,048 (131,072): 4.2; 4,096 (262,144): 47.1; 6,144 (393,216): 67.0 | not reached |
| E37, F1 (modulus 97) alone | 0.25 | 9,408 | S | 93683 | 2,048 (131,072): 7.1; 4,096 (262,144): 72.5; 6,144 (393,216): 81.4 | 6,144 (393,216) |
| E37, F1 (modulus 97) alone | 0.25 | 9,408 | S | 116089 | 2,048 (131,072): 5.5; 4,096 (262,144): 42.8; 6,144 (393,216): 67.6 | not reached |
| E37, F1 (modulus 97) alone | 0.25 | 9,408 | B | 73129 | 2,048 (131,072): 3.8; 4,096 (262,144): 65.5; 6,144 (393,216): 80.4 | 6,144 (393,216) |
| E37, F1 (modulus 97) alone | 0.25 | 9,408 | B | 93683 | 2,048 (131,072): 9.1; 4,096 (262,144): 73.2; 6,144 (393,216): 83.5 | 6,144 (393,216) |
| E37, F1 (modulus 97) alone | 0.25 | 9,408 | B | 116089 | 2,048 (131,072): 7.4; 4,096 (262,144): 65.5; 6,144 (393,216): 78.9 | not reached |
| E35-A arm G, first family (modulus 89), a whole birth | 0.25 | 7,920 | B | 73129 | 4,096 (262,144): 76.7; 8,192 (524,288): 87.7; 12,288 (786,432): 89.2; 16,384 (1,048,576): 89.5; 20,480 (1,310,720): 87.7; 24,576 (1,572,864): 90.3 | 8,192 (524,288) |
| E35-A arm G, first family (modulus 89), a whole birth | 0.10 | 3,168 | B | 73129 | 4,096 (262,144): 14.7; 8,192 (524,288): 17.0; 12,288 (786,432): 16.8; 16,384 (1,048,576): 19.5; 20,480 (1,310,720): 18.8; 24,576 (1,572,864): 20.5 | not reached |

These runs have the panel at 12 letters only (300 or 400 never-taught pairs of add and of sub, reduction and no-reduction pairs together), not the reduction pairs apart. Receipts: `exp45/e37/e37_<F3|F1>_<S|B>_<seed>.json`, `exp45/e35a/e35a_dev_G_f<25|10>_73129.json`.

A tenth of the first family at E41's other two widths (`exp45/e41/e41_<S|B>_h<256|1024>_f10_<seed>.json`; the lived networks are all at the present width). No run reaches 80% on R. Under back-propagation at the half width, which has E41's longest dose, 2 of 3 seeds pass 50%; no other run does:

| Feedforward width and recurrent units | Rule | Dose in updates (items) | Highest R, seeds 73129, 93683, 116089 | R at the last row |
|---|---|---|---|---|
| 256 and 128 | S | 40,960 (2,621,440) | 10.9, 10.6, 12.2 | 10.2, 10.6, 10.5 |
| 256 and 128 | B | 40,960 (2,621,440) | 45.1, 63.1, 56.1 | 42.4, 62.3, 54.8 |
| 1024 and 512 | S | 8,192 (524,288) | 11.0, 16.8, 15.2 | 11.0, 15.9, 15.2 |
| 1024 and 512 | B | 8,192 (524,288) | 17.5, 21.1, 21.4 | 16.4, 20.6, 21.3 |

E36's references are fresh networks on the first family at a half for 1,024 updates (65,536 items), not on a later family; their reduction score is over all 7,921 pairs, taught ones included. At their last row (`exp45/e36/e36_reference_<S|B>_<seed>.json`):

| Rule | Seed | Reduction score at 1,024 updates | Never-taught add and sub at 12 letters |
|---|---|---|---|
| S | 73129 | 1.1 | 0.8 |
| S | 93683 | 1.3 | 0.8 |
| S | 116089 | 1.1 | 2.0 |
| B | 73129 | 1.1 | 1.0 |
| B | 93683 | 1.4 | 0.8 |
| B | 116089 | 2.5 | 3.7 |

### 6.3 The first family and the new family: what differs, and what it does to the comparison

The fresh networks that reach the level (section 6.1) learned the first family, of modulus 89. The lived networks learned the new family, of modulus 83. Five things differ.

1. **Size.** An operation of the first family has 7,921 pairs and one of the new family 6,889. A tenth is 792 pairs an operation in the first family and 689 in the new one, so the lived network's 2,756 taught pairs are fewer than a tenth of the first family (3,168). At the fractions at which a fresh network reaches the level, the new family would have 6,888 pairs at a quarter and 13,776 at a half, in place of 7,920 and 15,840.
2. **No fresh run reaches the level on the new family.** The only fresh runs on modulus 83 are E40's, at a tenth, and they are at 6.0 to 11.8% after 524,288 items. No fresh network was taught modulus 83 at a quarter or at a half. So the fresh figures "to 80%" are measured on the first family and are used here for a family of this kind; they are not measured on the new family. At a tenth and at the present width the two families agree: no fresh run reaches 50% on either (E40 on modulus 83; E41 on modulus 89, highest R 8.0 to 30.6; E35-A on modulus 89, one seed under back-propagation, 20.5% on the panel at 12 letters after 1,572,864 items). At half the present width, with 2,621,440 items, fresh back-propagation runs at a tenth of modulus 89 reach an R of 45.1, 63.1 and 56.1 at their highest (section 6.2). So what a tenth gives a fresh network depends on the width, the rule and the dose, and no fresh run at a tenth in E40, in E41 at any of its three widths or in E35-A's arm G reaches 80%.
3. **Which way it moves the accounts.** If a fresh network needs a fraction of a family, the smaller family lowers its count of pairs to 6,888 and 13,776, and account (c) in pairs then needs 8 new families in place of 6 at a quarter (pairs only; section 7.1). If it needs a number of pairs, the counts stay as tabled. The record does not say which. The fresh runs at a quarter on other moduli (E37: 5,328 pairs of modulus 73 and 9,408 of modulus 97) were scored on the panel at 12 letters only and stopped at 262,144 and 393,216 items, where 3 of the 12 are at 80% on that panel.
4. **The two scores are of one kind but not the same.** E40 scores every never-taught reduction pair of add and of sub (3,055 to 3,078 each); E41 scores 1,000 of each. E40's first row after row 0 is at 64 updates and its rows then double; E41's rows are 256 updates apart. A lived run that is at 80% at row 64, as all runs of B and K are (`exp45/e40/e40_lived_<B|K>_<seed>.json`, section 5), may have reached it earlier, so its 4,096 items are an upper limit and its ratios "fresh ÷ lived" in section 7.2 are lower limits. A fresh run's first row at 80% is located to within 16,384 items.
5. **What a fresh network learns besides the family.** A fresh network learns with its one family how to read an item: two-digit operands, the operation symbol, the letters to be passed over, the end symbol. A lived network meets the new family with that learned, and with four new code bytes whose rows of the byte table are untrained (check 7 of `MOD97_LOCAL_RULE_20261106/E40_REPORT.md`). This holds for any family a fresh network learns, so it belongs to the cost of "a fresh network on one family" and is not a difference between modulus 89 and modulus 83.

A lived network also answers part of the new family before any update: 0.1 to 11.6% on the reduction score at row 0 (section 5). E40's verdict subtracts that; the levels here do not.

## 7. The accounts

What the record states (E40's verdict) is account (a) at equal experience of the new family: with the same 2,756 taught pairs and the same 524,288 items, the nine lived runs are at 77.0 to 94.5% on never-taught reduction pairs and the six fresh runs at 6.0 to 11.8%. The tables below restate it as the experience needed to reach a level, which is what "needs fewer samples" asks.

### 7.1 In distinct taught samples

| Account | Lived learner | Fresh, a tenth of the new family (E40) | Fresh, a quarter of the first family (E41) | Fresh, a half of the first family (E41) | Lived against fresh at a quarter and at a half |
|---|---|---|---|---|---|
| (a) the new family alone | 2,756 pairs | 2,756 pairs; does not reach the level | 7,920 pairs | 15,840 pairs | 2.9 and 5.7 times fewer for the lived learner |
| (b) the new family and everything before it | 33,332 pairs and 24,000 compositions (57,332 samples) | as (a) | 7,920 pairs | 15,840 pairs | 4.2 and 2.1 times more for the lived learner without the compositions; 7.2 and 3.6 with them |

(c) If each further new family cost the lived learner the tenth that E40's did, and a fresh network is spent on each family: the lived learner's whole count P + n × c falls below the fresh networks' n × F at the smallest whole n above P / (F − c). n counts new families from the modulus-83 family on; the further ones after it are n − 1.

| What is counted before the new family | P | c (lived, each new family) | F (fresh, each family) | P / (F − c) | n, new families in all | Further families after E40's |
|---|---|---|---|---|---|---|
| pairs only | 30,576 | 2,756 | a quarter of modulus 89 (7,920) | 5.92 | 6 | 5 |
| pairs only | 30,576 | 2,756 | a half of modulus 89 (15,840) | 2.34 | 3 | 2 |
| pairs only | 30,576 | 2,756 | a quarter of modulus 83 (6,888; no run) | 7.40 | 8 | 7 |
| pairs only | 30,576 | 2,756 | a half of modulus 83 (13,776; no run) | 2.77 | 3 | 2 |
| pairs and compositions | 54,576 | 2,756 | a quarter of modulus 89 (7,920) | 10.57 | 11 | 10 |
| pairs and compositions | 54,576 | 2,756 | a half of modulus 89 (15,840) | 4.17 | 5 | 4 |
| pairs and compositions | 54,576 | 2,756 | a quarter of modulus 83 (6,888; no run) | 13.21 | 14 | 13 |
| pairs and compositions | 54,576 | 2,756 | a half of modulus 83 (13,776; no run) | 4.95 | 5 | 4 |

### 7.2 In presented items, to 80% on never-taught reduction pairs

The lived learner's figure is the first row of its E40 run at or above 80%; the fresh figure is the first row of the E41 run of the same seed at or above 80% on R. Arm K is set beside the fresh sweep network, as in E40.

| Lived rule | Seed | (a) Lived: items of the new family | Fresh, a tenth of the new family (E40) | Fresh, a quarter of the first family (E41) | Fresh, a half of the first family (E41) | (a) fresh ÷ lived, a quarter | (a) fresh ÷ lived, a half | (b) Lived: birth, life and new family | (b) lived ÷ fresh, a quarter | (b) lived ÷ fresh, a half |
|---|---|---|---|---|---|---|---|---|---|---|
| S | 73129 | 8,192 | not reached in 524,288 (highest 6.0%) | 688,128 | 425,984 | 84 | 52 | 3,940,352 | 5.7 | 9.2 |
| S | 93683 | 16,384 | not reached in 524,288 (highest 7.6%) | 720,896 | 311,296 | 44 | 19 | 3,948,544 | 5.5 | 12.7 |
| S | 116089 | 8,192 | not reached in 524,288 (highest 6.4%) | more than 786,432 | 294,912 | more than 96 | 36 | 3,940,352 | fewer than 5.0 | 13.4 |
| B | 73129 | 4,096 | not reached in 524,288 (highest 7.7%) | 360,448 | 327,680 | 88 | 80 | 3,936,256 | 10.9 | 12.0 |
| B | 93683 | 4,096 | not reached in 524,288 (highest 11.8%) | 229,376 | 262,144 | 56 | 64 | 3,936,256 | 17.2 | 15.0 |
| B | 116089 | 4,096 | not reached in 524,288 (highest 8.3%) | 311,296 | 229,376 | 76 | 56 | 3,936,256 | 12.6 | 17.2 |
| K | 73129 | 4,096 | not reached in 524,288 (highest 6.0%) | 688,128 | 425,984 | 168 | 104 | 3,936,256 | 5.7 | 9.2 |
| K | 93683 | 4,096 | not reached in 524,288 (highest 7.6%) | 720,896 | 311,296 | 176 | 76 | 3,936,256 | 5.5 | 12.6 |
| K | 116089 | 4,096 | not reached in 524,288 (highest 6.4%) | more than 786,432 | 294,912 | more than 192 | 72 | 3,936,256 | fewer than 5.0 | 13.3 |

(c) The same rule as in 7.1, in items: P is the birth and the life, c the lived learner's items to 80% on the new family, F the fresh network's items to 80% on the first family. Cells give n, the new families in all, with the further families after E40's in brackets.

| Lived rule | Seed | P | c | F, a quarter | n (further), a quarter | F, a half | n (further), a half |
|---|---|---|---|---|---|---|---|
| S | 73129 | 3,932,160 | 8,192 | 688,128 | 6 (5) | 425,984 | 10 (9) |
| S | 93683 | 3,932,160 | 16,384 | 720,896 | 6 (5) | 311,296 | 14 (13) |
| S | 116089 | 3,932,160 | 8,192 | more than 786,432 | at most 6 (5) | 294,912 | 14 (13) |
| B | 73129 | 3,932,160 | 4,096 | 360,448 | 12 (11) | 327,680 | 13 (12) |
| B | 93683 | 3,932,160 | 4,096 | 229,376 | 18 (17) | 262,144 | 16 (15) |
| B | 116089 | 3,932,160 | 4,096 | 311,296 | 13 (12) | 229,376 | 18 (17) |
| K | 73129 | 3,932,160 | 4,096 | 688,128 | 6 (5) | 425,984 | 10 (9) |
| K | 93683 | 3,932,160 | 4,096 | 720,896 | 6 (5) | 311,296 | 13 (12) |
| K | 116089 | 3,932,160 | 4,096 | more than 786,432 | at most 6 (5) | 294,912 | 14 (13) |

c is the lived learner's items to the level, not the dose E40 gave. E40 gave 524,288 items of the new family to every run; 9 of the 11 fresh runs at a quarter or a half that reach 80% do so in fewer items than that, so with the dose in place of c the lived learner's whole count would not fall below theirs after any number of families.

The sweep beside back-propagation (checked here, seed by seed): the lived sweep run reaches 80% at the same row as the lived back-propagation run or a later one, and the fresh sweep run reaches it later than the fresh back-propagation run or not at all, at a quarter and at a half, in all three seeds. The taught pairs are the same under both rules.

### 7.3 The three seeds together, by rule

| Lived rule | (a) Lived: items of the new family to 80% | Fresh: items to 80%, a quarter of the first family | Fresh: items to 80%, a half | (a) fresh ÷ lived (a quarter; a half) | (b) Lived: all items to 80% | (b) lived ÷ fresh (a quarter; a half) | (c) n, new families in all (a quarter; a half) |
|---|---|---|---|---|---|---|---|
| S | 8,192 to 16,384 | 688,128 to 720,896; 1 of 3 not reached in 786,432 | 294,912 to 425,984 | 44 to 84; 19 to 52 | 3,940,352 to 3,948,544 | 5.5 to 5.7; 9.2 to 13.4 | 6; 10 to 14 |
| B | 4,096 | 229,376 to 360,448 | 229,376 to 327,680 | 56 to 88; 56 to 80 | 3,936,256 | 10.9 to 17.2; 12.0 to 17.2 | 12 to 18; 13 to 18 |
| K | 4,096 | 688,128 to 720,896; 1 of 3 not reached in 786,432 | 294,912 to 425,984 | 168 to 176; 72 to 104 | 3,936,256 | 5.5 to 5.7; 9.2 to 13.3 | 6; 10 to 14 |

In the quarter columns the ratios and n leave out the seed whose fresh run did not reach 80%, except n, which uses the dose as the least the fresh run could have needed.

### 7.4 Account (c) is an extrapolation that no run supports

Account (c) assumes that each further new family would cost the lived learner what the family of modulus 83 did: a tenth of its pairs and 4,096 to 16,384 items. I found no run on any fetched ref that teaches a lived network a second new family; the searches are in section 9. (E48 starts from the same lived networks and teaches them the same family of modulus 83, in added connections.) So no run supports the assumption, and two results on file speak against relying on it.

- **E40 itself.** The earlier families fall while the new one is learned (last table of section 5): never-taught add and sub of modulus 97 go from 91.0–98.5% to 17.9–39.5%, and of modulus 73 from 80.4–95.1% to 14.4–49.9%. The network that would meet a second new family is not the network that met the first.
- **E44's calibration** (job F1; `MOD97_LOCAL_RULE_20261106/exp45/e44/E44_REPORT.md`; the worker's figures, not recomputed here). Eight new families in turn, each from a tenth of its pairs, were taught to networks with only the birth behind them, at half the present width, with 48 taught items in each update. A family was given 256 updates. Its level on never-taught reduction pairs over the last quarter of those updates, as a median over the second to the eighth family, was 0.301 to 0.636 under back-propagation and 0.537 to 0.737 under the sweep, over eight worlds. The slope of that level against the logarithm of the family's place in turn was −0.101 and −0.039 on average and above zero in 2 of 8 worlds under each rule. That is another world and another learner. In it later families were not learned faster than earlier ones, and the level after 256 updates is set beside E40's nine lived runs, each of which had reached 80% by 256 updates (section 5).

**The run that would test it.** Plan item 24, "New families for a learner that has lived: several in turn", which waits for this item and has no brief on any fetched ref. To test account (c) it has to start from E40's lived networks, whose weights at row 8,192 are committed (`MOD97_LOCAL_RULE_20261106/exp45/e40/e40_lived_<S|B|K>_<seed>_row8192.pt`), and teach them at least as many further families as section 7.2 gives (5 to 17), each of a new modulus with new code bytes and a tenth of its pairs, scored on never-taught reduction pairs at E40's rows. Beside it, a fresh network on each family at a quarter and at a half would supply the reference that modulus 83 lacks. The nearest run in progress is E44's job L2, the first long lives (`OPUS_TASK_E44_L2_ROSTER_FIRST_LONG_LIVES_20261117.md`): lives of 1,024 visits that meet 41 to 52 families, each from a tenth, at the half width, by a learner with one family behind it. Its "trend of learning", the slope of the level at a family's first visit against the logarithm of the number of families met, is the quantity that says whether later families cost less. No report of it is on any fetched ref.

One more count was considered and not used. If the three earlier families are credited to the lived learner as families learned, with a fresh network for each given the same teaching list, the pairs before the new family are equal on both sides and account (b) in pairs becomes account (a) with the 24,000 compositions added. It is not used because at the end of E40 the lived network no longer answers those families (the levels above), where three fresh networks would each still answer theirs.

## 8. In which account the record supports "needs fewer samples"

The record supports it in account (a) and in no other. In account (a) it holds under both rules and under arm K: with the same 2,756 taught pairs a lived network reaches 80% on the never-taught reduction pairs of the new family in 4,096 items under back-propagation (its first row, all three seeds) and in 8,192 to 16,384 items under the sweep, where a fresh network given those pairs is at 6.0 to 11.8% after 524,288 items, and where a fresh network on the first family needs 7,920 or 15,840 pairs and 229,376 to 720,896 items (under the sweep at a quarter, two of three seeds reach 80%, at single rows only). This is also the programme's own sense of the words (section 1), a smaller taught fraction being enough: a tenth is enough for a lived network to reach 80%, and for no fresh network of E40, of E41 at any of its three widths or of E35-A's arm G; the highest fresh R at a tenth is 63.1%, at half the present width under back-propagation. The support is firmer under back-propagation, where two of three lived runs also reach 90% and all three end at 85.0% or more; under the sweep no lived run reaches 90% and one ends at 77.0%. In account (b) the record does not support it under either rule: with its birth and its life counted, a lived network has been given 33,332 distinct pairs and 24,000 compositions against 7,920 or 15,840 pairs, and 3,936,256 to 3,948,544 items against 229,376 to 720,896, which is 2.1 to 4.2 times the pairs (3.6 to 7.2 with the compositions) and 5.5 to 17.2 times the items of a fresh network that learns one family and reaches 80%. In account (c) the record cannot say: the lived learner's whole count would fall below that of fresh networks after 3 to 14 new families counted in pairs and 6 to 18 counted in items, but only if every further family cost what the first did, and no run has taught a lived network a second new family. Between the two rules the record gives the sweep no saving in samples in any account: the taught pairs are the same, and in every seed the sweep reaches 80% at the same row as back-propagation or later, lived and fresh. Three limits hold throughout: the fresh figures for reaching the level are measured on modulus 89, since no fresh network was taught modulus 83 at a quarter or a half; the birth and the life are counted as given and not as needed, since no lived network with a shorter history exists; and a lived run's items are located by rows that start at 64 updates.

## 9. Commands

Run from the repository root. Python 3.13.17, standard library only; numpy is not used.

```
# sections 4 to 7 and the appendix (about 3 seconds; reads 87 files through git show, writes nothing)
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -I PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/e40_whole_experience.py

# the same, with the list of the 87 paths read
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -I PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/e40_whole_experience.py --paths

# the ref and the commit
git rev-parse origin/codex/minifly-consolidated-20261023

# files read by hand (D = origin/codex/minifly-consolidated-20261023, M = MOD97_LOCAL_RULE_20261106)
git show $D:OPUS_TASK_E40_NEW_FAMILY_FROM_A_TENTH_OF_PAIRS_20261115.md
git show $D:OPUS_TASK_E41_COMPUTATION_TO_THRESHOLD_BY_WIDTH_AND_FRACTION_20261116.md
git show $D:OPUS_TASK_E37_FRESH_REFERENCE_FOR_A_LIFE_FAMILY_20261115.md
git show $D:OPUS_TASK_E36_REMINDER_AFTER_A_LIFE_20261115.md
git show $D:OPUS_TASK_E44_LONG_LIFE_WORLD_IN_THE_FULL_SCALE_FORMAT_20261116.md
git show $D:OPUS_TASK_E44_L2_ROSTER_FIRST_LONG_LIVES_20261117.md
git show $D:OPUS_TASK_E48_A_NEW_FAMILY_IN_CONNECTIONS_GATED_BY_ITS_CODE_20261116.md
git show $D:$M/E40_REPORT.md
git show $D:$M/exp45/e37/E37_REPORT.md
git show $D:$M/exp45/e44/E44_REPORT.md
git show $D:$M/e40_new_family.py
git show $D:$M/exp26_candidate.py
git show $D:$M/exp41_scale.py
git show $D:$M/exp16_integrated.py | sed -n '540,625p;700,768p;850,860p'
git show $D:$M/exp11_cell.py | sed -n '40,70p;220,270p'
git ls-tree -r --name-only $D -- $M/exp45/e26/ $M/exp45/e31/ $M/exp45/e36/ $M/exp45/e37/ $M/exp45/e40/ $M/exp45/e41/ $M/exp45/e35a/

# read by hand before they were added to the script: the twelve receipts $M/exp45/e41/e41_{S,B}_h{256,1024}_f10_<seed>.json

# search 1: a report of E44's long lives on any fetched ref (no line printed)
for r in $(git for-each-ref --format='%(refname:short)' refs/remotes/origin); do git ls-tree -r --name-only $r | grep -iE "E44_L2_REPORT" | sed "s|^|$r: |"; done
# search 2: receipt folders that a fetched ref has and the default branch has not (printed: e42, e46, e47, e48, e49, e50 and their _qual folders)
for r in $(git for-each-ref --format='%(refname:short)' refs/remotes/origin); do comm -13 <(git ls-tree -d --name-only $D $M/exp45/ | sort) <(git ls-tree -d --name-only $r $M/exp45/ 2>/dev/null | sort) | sed "s|^|$r: |"; done | awk -F': ' '{print $2}' | sort -u
# search 3: briefs of E42 to E50 that start from a life file (printed: E48 only)
for f in $(git ls-tree --name-only $D | grep -E "^OPUS_TASK_E(4[2-9]|50)"); do echo "$(git show $D:$f | grep -ciE 'e26_life|e31_life') $f"; done | grep -v "^0 "
# search 4: the tasks of 17 November that name a lived network (printed: 0 for E51, E52 and E54)
for f in PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/COORDINATION_20261117/TASK_E5*.md; do echo "$(grep -ciE 'e26_life|e31_life|lived network|has lived' $f) $(basename $f)"; done

# the project's text lint on this document (reads only; exit 0, one advisory warning)
python3 -I -B tools/minifly_check/minifly_check.py lint PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/E40_SAVING_IN_WHOLE_EXPERIENCE_20261117.md
```

The script's own checks, all passed: the source lines of the appendix; the chain of files by sha256; equal batch digests across the rules of a seed in the birth, in every life block and at every row of E40; presented items equal to 64 times the updates at every row of E40, none of them a never-taught pair; the split against five counts in the receipts; and that each E41 run at a half is the E26 birth of the same rule and seed (equal batch digest at row 4,096, equal counts on E26's own panels at rows 4,096 and 8,192).

## 10. What I could not find

1. A fresh network taught the new family (modulus 83) at a quarter or at a half, under either rule. The fresh runs on modulus 83 are E40's six, at a tenth.
2. A row of E40 between 0 and 64 updates. The dry runs (`exp45/e40/dry/`) have rows 0, 64, 128 and 256 only. So the receipts cannot show whether a lived network reaches the level in fewer than 4,096 items, or from fewer than the 2,756 pairs, all of which had been shown by row 64.
3. A lived network taught a second new family, on any fetched ref (searches 1 to 4 of section 9); and a brief for plan item 24.
4. The reduction pairs apart in the fresh runs of E37 and of E35-A. Those runs have the panel at 12 letters only, so they are set beside the reduction score and not on it.
5. A lived network with a shorter birth or a shorter life, or at another width. So the record does not say how much of the 3,932,160 items before the new family the saving needs, and account (b) counts them all.
6. A fresh run at a tenth at the present width under the sweep beyond 12,288 updates, or under back-propagation beyond one seed's 24,576 (E35-A arm G). The longest fresh runs at a tenth are E41's at half the present width, 40,960 updates (section 6.2).
7. A report of E44's long lives (job L2) on any fetched ref.
8. A fresh reference for a later family in E36. Its six references are fresh networks on the first family for 1,024 updates (section 6.2).

## Appendix. The rules, as the code states them

Each line below is in the named file of `MOD97_LOCAL_RULE_20261106/` at commit `7cabb13adc38753483d34dd5472fc3c84c4de675`; the script stops if one is missing.

| File | Line | Text |
|---|---|---|
| `exp45.py` | 30 | `BATCH, BLOCK = 64, 512` |
| `exp45.py` | 32 | `DEV_UPDATES = 24576` |
| `exp11_cell.py` | 56 | `DEV_UPDATES, STAGE_UPDATES = X.DEV_UPDATES, int(os.environ.get("E11_STAGE_UPDATES", "6144"))` |
| `exp11_cell.py` | 50 | `DEV = [("add", 89, b"p"), ("sub", 89, b"q"), ("max", 89, b"v"), ("min", 89, b"w")]` |
| `exp11_cell.py` | 51 | `LIFE = [("add", 97, b"r"), ("sub", 97, b"t"), ("max", 97, b"x"), ("min", 97, b"n")]` |
| `exp11_cell.py` | 238 | `rng = Random(seed * 1000 + round(frac * 100) + m * 7 + code[0])` |
| `exp11_cell.py` | 240 | `k = round(frac * len(pairs))` |
| `exp11_cell.py` | 250 | `rng.shuffle(order)` |
| `exp11_cell.py` | 253 | `yield render(t, a, b, rng.randint(0, kmax), rng)` |
| `exp16_integrated.py` | 546 | `F1 = E.LIFE` |
| `exp16_integrated.py` | 547 | `F3 = [("add", 73, b"s"), ("sub", 73, b"u"), ("max", 73, b"y"), ("min", 73, b"z")]` |
| `exp16_integrated.py` | 550 | `HELD_PAIRS = [("+", "-"), ("-", "+"), (">", "<"), ("<", ">")]` |
| `exp16_integrated.py` | 584 | `rng = Random(seed * 31 + ord(pair[0]) * 7 + ord(pair[1])) if split == "taught"` |
| `exp16_integrated.py` | 595 | `return [("comp", o1, o2, a, b, c) for o1, o2 in TAUGHT_PAIRS for a, b, c in triples((o1, o2), seed, 2000)]` |
| `exp16_integrated.py` | 596 | `return [("op", t, a, b) for t in (F1 if fam == "F1" else F3) for a, b in E.split(t, .25, seed)[0]]` |
| `exp16_integrated.py` | 605 | `yield render(entries[i], rng)` |
| `exp16_integrated.py` | 612 | `yield next(streams[rng.randrange(len(streams))])` |
| `exp16_integrated.py` | 855 | `SCHEDULE = [("F1", 1), ("F1", 2), ("F2", 1), ("F2", 2), ("F3", 1), ("F3", 2)]` |
| `exp26_candidate.py` | 131 | `tp = [(t, E.split(t, .5, seed)[0]) for t in E.DEV]` |
| `exp26_candidate.py` | 142 | `for stop in range(X.BLOCK, E.DEV_DOSE + 1, X.BLOCK):` |
| `exp26_candidate.py` | 214 | `entries = {f: X16.family_entries(f, seed) for f in ("F1", "F2", "F3")}` |
| `exp26_candidate.py` | 243 | `seen_streams.append(X16.entry_stream(entries[fam], seed + 100 + si))` |
| `exp26_candidate.py` | 246 | `for u in range(E.STAGE_UPDATES):` |
| `e40_new_family.py` | 33 | `ROWS = (0, 64, 128, 256, 512, 1024, 2048, 4096, 8192)` |
| `e40_new_family.py` | 37 | `F4 = [("add", 83, b"e"), ("sub", 83, b"f"), ("max", 83, b"g"), ("min", 83, b"h")]` |
| `e40_new_family.py` | 40 | `FRAC = .10` |
| `e40_new_family.py` | 71 | `return a + b >= 83 if op == "add" else a - b < 0` |
| `exp41_scale.py` | 36 | `DOSE = {256: 40960, 512: 12288, 1024: 8192}` |
| `exp41_scale.py` | 40 | `ROW = 256` |
| `exp41_scale.py` | 99 | `return Random(f"e41|{seed}|{round(f * 100)}|{t[0]}").sample(sorted(pairs), 1000)` |
