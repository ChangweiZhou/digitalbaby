# E42 on three seeds: verification of the worker's results, and the first seed's readings tested on all three (17 November 2026)

From the outside reviewer. This document extends `E42_SEED_73129_VERIFICATION_20261116.md` to the two further seeds, and it tests each reading that memo section 22.12 drew from the first seed against all three.

**What was read.** The worker's branch `claude/gallant-dijkstra-i9zgsr` at `44f91edc`: the combined report `exp45/e42/E42_REPORT_three_seeds.md`, its summary and predictions files, the 45 receipts, the 45 structure files, the 45 full weights files and the 45 small files, and the check receipts under `exp45/e42_qual/`. The helper's branch `claude/funny-cori-q54ch0` at `d3b8e823`. E26's six birth files on the default branch. The brief is revision 2 (`07943077` on the default branch).

**How.** Read-only. Nothing was trained, torch was not used, and no code of the worker's, the helper's or the default branch was run. Files were read with `git show`, one at a time. The forward computation is my own `e42_cells_reader.py`, written from the brief's text; the test items are rebuilt with `np_gru.py`. Every figure below comes from

    python3 -I PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/e42_three_seeds_verify.py [A B C D E F G H I]

run from the repository root. The part that gives each figure is named in the section, and section 10 lists the commands and their running times.

**Terms.**

- An *update* is one step of the optimizer on a batch of 64 items. Every run has 24,576 updates and is scored every 256 updates; one such scoring is a *row*.
- *Score A* is the percentage of whole answers right on pairs of operands that were never taught, for addition and subtraction, with 12 letters between the operands (300 items each). *Score R* is the same on 1,000 untaught pairs of each of the two operations whose answer requires a reduction by the modulus, without letters. The *length test* is the percentage right on 300 untaught addition items at 12 letters and on the same 300 at 40 letters; training items have at most 12 letters. The figures at 12 and at 40 letters are means over the last three rows.
- *Updates to 80* (or to 90, or to 50) on a score is the update of the first of three consecutive rows that are all at or above the mark. The brief calls this the three-row version; the single-row version takes the first row at or above the mark.
- *Weight-items* to a mark is the brief's measure of computation: the number of weights that take part in the network, times the 64 items of a batch, times the updates to the mark.
- The two *rules* are backpropagation (exact gradients) and the sweep (each linear map sends its error back through the signs of its weights times one scale for the matrix).
- The brief's *descriptions*: a candidate needs *fewer* updates than its reference if the ratio of the two numbers is below 2/3, *more* if it is above 1.5, and *about the same* otherwise; at 40 letters a candidate is *lower* or *higher* if the difference is 15 points or more. A *cell* of the brief's table carries the description that at least two of the three seeds share.
- The configurations. `G`: the present network, in which each of the 256 cells reads all others. `F10`: each cell reads ten fixed cells among the 48 of its neighbourhood on a 16 by 16 grid. `A10`: as `F10`, but a connection may be given up and replaced by another in the neighbourhood every 256 updates. `W10`: as `A10`, with the ten connections drawn among all 255 other cells. `N5`, `N20`: as `A10` with five and with twenty connections. `P8`: as `A10`, with cells of periods 1, 2, 4 and 8, where a cell of period p changes its value once every p positions and combines what arrived during the p positions with learned weights (the variant `step`). `P4`, `P16`: the period sets 1, 2, 4 and 1, 2, 4, 8, 16. `P8-last`: a cell of period p uses only what arrives at the position of its change. `P8-source`: a separate strength for every connection at every step of the period.

**A note on two labels.** The request for this check named `F10` against `A10` as the comparison for locality and `W10` against `F10` as the comparison for changing connections. In the brief the comparison for locality is `W10` against `A10` (both with changing connections; they differ only in the neighbourhood), and the comparison for changing connections is `A10` against `F10`. Sections 6(b) and 6(e) use the brief's comparisons and give the other figures as well.

## 1. Summary

1. **The worker's figures are confirmed on all three seeds.** The reader reproduces all twelve final counts of every one of the 45 runs. All 720 figures of the worker's summary file that I recomputed from the receipts are equal to mine, all 105 cells of the report's table of descriptions are equal to mine, and my scoring of the eleven predictions agrees with the report's in every case. For all three seeds and both rules, the thirteen tensors of `G`'s final weights file are bit-equal to E26's birth file on the default branch.
2. **Three readings of the first seed hold on three of three seeds.** Ten fixed local connections a cell are enough to learn the task, at 1.7 to 2.1 times the dense network's updates under backpropagation and 2.3 to 2.7 times under the sweep. Under the sweep, periods remove most of that cost: `P8` needs 0.54 to 0.59 of `A10`'s updates. The number of connections, the neighbourhood, changing against fixed connections, the choice among the three period sets, and a separate strength per step all change little.
3. **One reading of the first seed fails, and one is at the margin.** The statement that `P8` is about the same as the dense network on every measure under both rules does not hold: under backpropagation `P8` needs 1.25, 1.75 and 1.65 times `G`'s updates (the cell is *more*), and under the sweep 1.36, 1.74 and 1.38 times (*about the same* on two seeds of three, and *more* if the single-row version of the threshold is used). `P8` reached 90 before the dense network on the first seed only.
4. **Under backpropagation, periods remove little of the cost of sparsity on the two new seeds** (14% and 21% of `A10`'s extra updates over `G`, against 71% on the first seed). What holds on every seed is narrower: with periods, the sweep needs nearly the same number of updates as backpropagation (1.04 to 1.20 times), as in the dense network (1.10 to 1.24), whereas a sparse network without periods needs 1.4 to 1.9 times as many under the sweep.
5. **The period 16 loses accuracy at 40 letters on two seeds of three** (82.8, 94.1 and 67.3, against 96.0, 96.6 and 95.3 for `P8`), and by the brief's mark of 15 points on one.
6. **In computation, no configuration is below the dense network under backpropagation at the 80 threshold on any seed.** `P8` under the sweep costs 1.25, 1.72 and 1.42 times the forward multiply-adds that `G` under backpropagation spends to reach 80 on A.
7. **The defect of the replacement rule is the same on all three seeds** (section 7), so E42 still says nothing about changing connections.

## 2. Inventory, integrity and the checks (part A)

- **Files.** On the worker's branch, each of the 45 runs (15 a seed) has its receipt, its structure file, its full weights file and its small file. The brief asked for full weights for seed 73129 only; they are committed for all three seeds. On the helper's branch there are the same four files for 15, 4 and 14 runs of the three seeds, and each is the identical blob on the worker's branch; six log files are the only files of the helper's branch that are absent from the worker's or differ. The worker's branch at `44f91edc` is therefore the one source.
- **Receipts.** All 45 are complete at 24,576 updates with 97 rows; the limit on connections is recorded as held and the losses as finite at every row; the SHA-256 of each committed weights file equals the receipt's.
- **Same data.** Within each seed the digest of the batches at row 4,096 is one value over the fifteen runs and begins with the brief's value (bbd71ab7, 65b1c7e0, 1df203fb), and the digest of all 24,576 batches is one value over the fifteen runs. The second statement covers the runs that were resumed.
- **Resumed runs.** Seed 73129: all fifteen runs by `exp42_cells.py`, none resumed. Seeds 93683 and 116089: four runs each by `exp42_cells.py` (`G` and `P8` under both rules) and eleven by `exp42_ckpt.py`, the worker's wrapper that saves the whole state every 1,024 updates. Nine runs were resumed from a saved state after a restart of the container: for 93683, `F10` under both rules at 1,024, `A10` under both rules at 18,432 and `P8-source` at 10,240; for 116089, `F10` under the sweep at 1,024 and `F10` under backpropagation and `A10` under both rules at 9,216. This is what the report's table states. The wrapper's own test (four dry runs of 2,048 updates stopped at 1,024 and resumed, with tensors, rows and data digest equal to the unbroken runs) records `pass`; I cannot repeat it.
- **Check receipts for the new seeds.** `check1_93683`, `check1_116089`, `check2_93683`, `check2_116089` and the helper's two for seed 116089 each record `pass: true`. Check 1 has two parts: the thirteen tensors at the start are bit-equal to the present network's for all three seeds, and the dry birth of 2,048 updates has E26's hash under both rules. The second part is run on seed 73129 by every host, as the brief specifies, because E26's dry-birth hashes exist for that seed only.
- **Check 11, done here for every seed.** For each of the three seeds and both rules, I compared the thirteen tensors of `G`'s final weights file with E26's birth file `exp45/e26/e26_dev_<S|B>_<seed>.pt` on the default branch: all thirteen are bit-equal in all six files. `G` run by the new code over 24,576 updates is therefore exactly the birth of E26 on every seed, which is the strongest test of the runner that does not need torch.

## 3. The reader against the receipts, and the saved structure (parts B and C)

**The final weights of the 45 runs (part B).** For every run the reader gives twelve counts of whole answers right: the length test at 12 and at 40 letters, score A for each of the four operations, score R for addition and for subtraction, and the taught panel (300 taught pairs of each of the four operations at 12 letters, which the first seed's check did not include). **All twelve counts equal the receipt's last row in 45 of 45 runs**, with no item different. For seed 73129 this repeats the earlier check at the present commit and adds the taught panel; the tables below give the two new seeds.

A whole answer is counted right here when the network's most probable byte is right at each of the answer's three positions, with the right bytes before it as input. This gives the same verdict for every item as the harness's generation of the answer one byte at a time, because the first wrong byte makes the answer wrong in both. To show it on the new seeds, the length test was also generated one byte at a time for seven runs of each new seed (`F10` and `P8` under both rules, `P16`, `P8-last` and `P8-source`): the two counts are the same in 14 of 14 runs.

**Seed 93683.** Counts of whole answers right by the reader on the final weights; each of the twelve equals the count in the receipt's last row.

| Run | Length test, 12 and 40 letters (of 300) | A: add, sub, max, min (of 300) | R: add, sub (of 1,000) | Taught panel: add, sub, max, min (of 300) | Equal to the receipt |
|---|---|---|---|---|---|
| `G`, sweep | 296, 291 | 296, 293, 295, 297 | 988, 979 | 297, 298, 292, 297 | all twelve |
| `G`, backpropagation | 298, 296 | 298, 294, 297, 293 | 970, 968 | 298, 298, 300, 295 | all twelve |
| `F10`, sweep | 286, 286 | 286, 278, 300, 291 | 935, 900 | 283, 297, 299, 299 | all twelve |
| `F10`, backpropagation | 294, 293 | 294, 292, 290, 291 | 961, 956 | 297, 299, 295, 298 | all twelve |
| `A10`, sweep | 282, 269 | 282, 274, 295, 294 | 930, 883 | 290, 291, 300, 299 | all twelve |
| `A10`, backpropagation | 287, 284 | 287, 288, 281, 288 | 959, 928 | 291, 297, 288, 296 | all twelve |
| `P8`, sweep | 286, 287 | 286, 288, 289, 296 | 965, 944 | 298, 298, 298, 299 | all twelve |
| `P8`, backpropagation | 296, 294 | 296, 291, 293, 297 | 955, 947 | 298, 300, 299, 300 | all twelve |
| `N5`, sweep | 281, 280 | 281, 257, 292, 296 | 896, 838 | 289, 293, 297, 295 | all twelve |
| `N20`, sweep | 280, 274 | 280, 275, 294, 287 | 888, 847 | 285, 286, 300, 291 | all twelve |
| `W10`, sweep | 281, 281 | 281, 269, 294, 295 | 921, 880 | 292, 291, 299, 300 | all twelve |
| `P4`, sweep | 283, 284 | 283, 284, 297, 293 | 922, 904 | 295, 299, 299, 295 | all twelve |
| `P16`, sweep | 285, 283 | 285, 284, 297, 297 | 929, 905 | 295, 295, 297, 300 | all twelve |
| `P8-last`, sweep | 285, 281 | 285, 284, 294, 299 | 900, 923 | 286, 298, 300, 299 | all twelve |
| `P8-source`, sweep | 285, 286 | 285, 285, 296, 298 | 945, 939 | 295, 300, 300, 299 | all twelve |

**Seed 116089.** Counts of whole answers right by the reader on the final weights; each of the twelve equals the count in the receipt's last row.

| Run | Length test, 12 and 40 letters (of 300) | A: add, sub, max, min (of 300) | R: add, sub (of 1,000) | Taught panel: add, sub, max, min (of 300) | Equal to the receipt |
|---|---|---|---|---|---|
| `G`, sweep | 300, 299 | 300, 299, 292, 291 | 993, 996 | 300, 296, 299, 297 | all twelve |
| `G`, backpropagation | 284, 277 | 284, 279, 297, 293 | 955, 895 | 285, 288, 295, 300 | all twelve |
| `F10`, sweep | 274, 264 | 274, 268, 292, 288 | 899, 906 | 289, 296, 296, 298 | all twelve |
| `F10`, backpropagation | 288, 285 | 288, 281, 289, 284 | 931, 926 | 290, 295, 299, 292 | all twelve |
| `A10`, sweep | 275, 261 | 275, 269, 281, 288 | 888, 880 | 289, 289, 294, 294 | all twelve |
| `A10`, backpropagation | 293, 290 | 293, 295, 290, 282 | 953, 965 | 298, 295, 299, 297 | all twelve |
| `P8`, sweep | 285, 284 | 285, 280, 292, 293 | 934, 913 | 295, 293, 298, 299 | all twelve |
| `P8`, backpropagation | 294, 293 | 294, 287, 292, 295 | 939, 937 | 294, 296, 298, 298 | all twelve |
| `N5`, sweep | 277, 263 | 277, 280, 295, 289 | 889, 887 | 294, 297, 300, 297 | all twelve |
| `N20`, sweep | 270, 264 | 270, 264, 284, 293 | 876, 887 | 288, 294, 292, 299 | all twelve |
| `W10`, sweep | 268, 268 | 268, 272, 285, 280 | 860, 896 | 286, 295, 297, 295 | all twelve |
| `P4`, sweep | 281, 279 | 281, 281, 295, 292 | 924, 921 | 293, 295, 299, 297 | all twelve |
| `P16`, sweep | 279, 203 | 279, 278, 294, 293 | 931, 897 | 293, 294, 298, 299 | all twelve |
| `P8-last`, sweep | 271, 257 | 271, 277, 287, 291 | 906, 902 | 289, 293, 298, 297 | all twelve |
| `P8-source`, sweep | 281, 282 | 281, 282, 297, 292 | 918, 921 | 290, 294, 298, 295 | all twelve |

**The saved structure (part C).** In all 45 full weights files:

- every cell has exactly N connections beside itself (255, 5, 10 or 20) and reads its own previous value, and every connection is inside the cell's neighbourhood (for `W10`, among all other cells);
- every recurrent weight outside the mask is exactly zero; in `P8-source` so is every entry of the added matrices outside the mask or outside the cell's period; the weights over the steps of a period are exactly zero outside each cell's period;
- the periods and phases in the file are those that this script draws from the brief's generator strings for the seed, and every connection that the file records as never replaced is among the connections that this script draws for the seed;
- in the six `F10` runs and the six `G` runs the final connections are exactly the drawn ones;
- the tensors of the small file are equal to those of the full file.

Beside the worker's summary file, the share of connections never replaced (0.459 to 0.593 in the 33 runs with changing connections) is equal in 45 of 45 runs, and so are the shares of connections by the reader's period and the source's period, at the start (from my draws) and at the end (from the file). In the runs with periods no such share moves by more than 0.027 between the start and the end, on any seed: as on the first seed, the replacement of connections does not select connections by period.

## 4. The receipts recomputed (part D)

- **Thresholds.** For each of the 45 runs I recomputed the updates to 50, 80 and 90 on A and on R in both versions, the two means of the length test, the number of connections replaced and the count of weights: 720 figures, all equal to the worker's summary file `E42_SUMMARY_three_seeds.json`. The CPU-hours by seed (15.22, 18.66, 18.38; total 52.26) are the sums of the receipts' own figures.
- **Descriptions.** I recomputed the 105 cells of the report's table (21 comparisons, five measures) from the receipts: the cell's description and the three per-seed descriptions are equal in all 105.
- **The mark `*` in the report's table.** The report's heading says the mark stands "where the single-row version describes the cell differently". It is placed where the single-row version gives at least one seed a different description (this reproduces the mark in 105 of 105 cells), which is a wider condition. The description of the cell itself, by two seeds of three, is different with the single-row version in three cells only: `A10` against `G` under backpropagation, weight-items to 80 on A (*more* with three rows, *about the same* with one); and `P8` against `G` under the sweep, updates to 80 on A and updates to 80 on R (*about the same* with three rows, *more* with one). The last two bear on reading (c) below.

## 5. Three seeds: updates to 80 and to 90 on A, and the score at 40 letters (part E)

Seeds in the order 73129, 93683, 116089. "–" means that the mark was not reached within the 24,576 updates. The median counts a run that did not reach the mark as later than any that did.

| Run | Updates to 80 on A | Median (range) | Updates to 90 on A | Median (range) | At 40 letters | Median (range) |
|---|---|---|---|---|---|---|
| `G`, sweep | 5,632, 4,864, 5,376 | 5,376 (4,864 to 5,632) | 12,800, 6,400, 9,472 | 9,472 (6,400 to 12,800) | 96.4, 96.4, 97.1 | 96.4 (96.4 to 97.1) |
| `G`, backpropagation | 5,120, 4,096, 4,352 | 4,352 (4,096 to 5,120) | 10,240, 4,864, 9,216 | 9,216 (4,864 to 10,240) | 94.6, 98.0, 93.2 | 94.6 (93.2 to 98.0) |
| `F10`, sweep | 13,312, 13,056, 12,544 | 13,056 (12,544 to 13,312) | 20,480, 18,944, 21,504 | 20,480 (18,944 to 21,504) | 89.1, 95.2, 88.2 | 89.1 (88.2 to 95.2) |
| `F10`, backpropagation | 8,704, 6,912, 9,216 | 8,704 (6,912 to 9,216) | 15,360, 9,472, 19,456 | 15,360 (9,472 to 19,456) | 92.6, 97.7, 93.7 | 93.7 (92.6 to 97.7) |
| `A10`, sweep | 13,824, 14,336, 13,824 | 13,824 (13,824 to 14,336) | 22,528, 19,968, 22,528 | 22,528 (19,968 to 22,528) | 91.8, 92.6, 86.7 | 91.8 (86.7 to 92.6) |
| `A10`, backpropagation | 9,472, 7,680, 7,936 | 7,936 (7,680 to 9,472) | 20,992, 13,056, 12,544 | 13,056 (12,544 to 20,992) | 92.2, 97.1, 96.1 | 96.1 (92.2 to 97.1) |
| `P8`, sweep | 7,680, 8,448, 7,424 | 7,680 (7,424 to 8,448) | 10,496, 11,008, 12,288 | 11,008 (10,496 to 12,288) | 96.0, 96.6, 95.3 | 96.0 (95.3 to 96.6) |
| `P8`, backpropagation | 6,400, 7,168, 7,168 | 7,168 (6,400 to 7,168) | 8,192, 9,984, 9,472 | 9,472 (8,192 to 9,984) | 96.6, 97.6, 97.1 | 97.1 (96.6 to 97.6) |
| `N5`, sweep | 17,152, 14,336, 14,080 | 14,336 (14,080 to 17,152) | –, 21,760, – | not reached (reached on one seed) | 86.9, 91.8, 86.8 | 86.9 (86.8 to 91.8) |
| `N20`, sweep | 14,336, 13,056, 12,288 | 13,056 (12,288 to 14,336) | –, 19,968, 22,272 | 22,272 (reached on two seeds) | 87.9, 92.3, 88.7 | 88.7 (87.9 to 92.3) |
| `W10`, sweep | 17,408, 12,032, 14,080 | 14,080 (12,032 to 17,408) | –, 16,896, 22,528 | 22,528 (reached on two seeds) | 88.0, 93.7, 90.4 | 90.4 (88.0 to 93.7) |
| `P4`, sweep | 8,192, 9,728, 7,936 | 8,192 (7,936 to 9,728) | 10,496, 13,568, 12,288 | 12,288 (10,496 to 13,568) | 94.4, 95.9, 93.4 | 94.4 (93.4 to 95.9) |
| `P16`, sweep | 7,936, 9,216, 8,192 | 8,192 (7,936 to 9,216) | 11,776, 13,056, 12,800 | 12,800 (11,776 to 13,056) | 82.8, 94.1, 67.3 | 82.8 (67.3 to 94.1) |
| `P8-last`, sweep | 11,520, 9,216, 9,728 | 9,728 (9,216 to 11,520) | 20,224, 13,568, 16,384 | 16,384 (13,568 to 20,224) | 91.3, 93.0, 87.0 | 91.3 (87.0 to 93.0) |
| `P8-source`, sweep | 7,680, 7,936, 7,936 | 7,936 (7,680 to 7,936) | 11,008, 12,032, 11,008 | 11,008 (11,008 to 12,032) | 93.7, 96.0, 95.4 | 95.4 (93.7 to 96.0) |

The first column of each triple is the first seed, and it equals the figures of memo section 22.12 in every entry.

## 6. The readings of the first seed, tested on each seed (part F)

Ratios are of updates to 80 on A unless another measure is named, candidate over reference, in the order of the seeds 73129, 93683, 116089.

**(a) Limiting each cell to ten fixed local connections raises the updates to the threshold. Holds on three of three seeds under both rules.** `F10` over `G`: 2.36, 2.68, 2.33 under the sweep and 1.70, 1.69, 2.12 under backpropagation (to 80 on R: 2.27, 2.95, 2.24 and 1.48, 1.94, 1.90). The cells of the brief's table are *more* for updates under both rules. The sparse network does learn the task on every seed: `F10` reaches 80 and 90 on A in every run, ends at 90 to 94 on A under the sweep and 92 to 98 under backpropagation, and at 40 letters it is 7.3, 1.2 and 8.9 points below `G` under the sweep and within 2.0 points of `G` under backpropagation, in every case under the mark of 15 points.

**(b) Locality adds nothing measurable to the cost of sparsity. Holds on three of three seeds, with no consistent direction.** The brief's comparison is `W10` (ten changing connections drawn among all cells) against `A10` (ten changing connections drawn in the neighbourhood): 1.26, 0.84, 1.02 in updates to 80 on A and 1.27, 0.85, 1.18 on R; at 40 letters `W10` is 3.8 points below `A10` on the first seed and 1.1 and 3.8 points above it on the other two. Every description is *about the same*. On the first seed the network without locality needed a quarter more updates; on the second it needed a sixth fewer. Both configurations share the defect of section 7.

**(c) Periods.**

- *Against the sparse network without periods, under the sweep: holds on three of three.* `P8` over `A10`: 0.56, 0.59, 0.54 (*fewer* on every seed, on A and on R); `P8` over `F10`: 0.58, 0.65, 0.59. Of the updates that `A10` needs beyond `G`, `P8` does not need 75%, 62% and 76% (73%, 56% and 71% with `F10` as the sparse network).
- *Against the sparse network without periods, under backpropagation: the first seed's size does not repeat.* `P8` over `A10`: 0.68, 0.93, 0.90, *about the same* on every seed. The share of `A10`'s extra updates that `P8` does not need is 71% on the first seed and 14% and 21% on the other two (64%, −9% and 42% with `F10`).
- *Against the dense network: the first seed's statement fails under backpropagation and is at the margin under the sweep.* `P8` over `G`: 1.36, 1.74, 1.38 under the sweep, which is *about the same* on two seeds of three with the three-row version and *more* on two of three with the single-row version (section 4); and 1.25, 1.75, 1.65 under backpropagation, where the cell is *more* for updates on A and on R. Memo section 22.12 said of the first seed that `P8` against `G` was about the same on every measure under both rules.
- *`P8` reaches 90 before the dense network: one seed of three.* `P8` over `G` in updates to 90 on A: 0.82, 1.72, 1.30 under the sweep and 0.80, 2.05, 1.03 under backpropagation. The dense network's own figure varies by a factor of two between seeds (6,400 to 12,800 under the sweep), so this comparison is not stable.
- *`P8` equals the dense network at 40 letters: three of three.* The differences are −0.4, +0.1, −1.8 points under the sweep and +2.0, −0.4, +3.9 under backpropagation, and `P8` loses at most 0.6 points between 12 and 40 letters in any of its six runs.
- *At the end of the dose* (score A, mean of the last three rows) `P8` is at 96, 96, 95 under the sweep and 98, 98, 97 under backpropagation, against 98, 99, 99 and 96, 98, 95 for `G`; `F10` and `A10` under the sweep are at 91 to 95.
- *`P8` under the sweep against `G` under backpropagation:* 1.50, 2.06 and 1.71 times the updates to 80 on A (median 1.71), and 1.32, 2.31 and 1.80 on R.
- *A view without a threshold.* Score A at fixed numbers of updates (mean of three rows), in the order of the seeds: at 8,192 updates `G` is at 89, 92, 91 under the sweep and 90, 95, 89 under backpropagation; `P8` at 84, 79, 85 and 91, 88, 90; `F10` at 54, 53, 45 and 79, 87, 80; `A10` at 51, 49, 48 and 78, 85, 82. At 12,288 updates `P8` is at 92, 92, 90 and 95, 92, 92, level with `G` (91, 95, 95 and 91, 94, 93), while `F10` under the sweep is at 78, 76, 79. The three seeds agree closely in this view: `P8` is behind `G` early and level from about 12,000 updates, and the sparse networks without periods are far behind under the sweep and a little behind under backpropagation.
- *The two rules within one configuration: holds on three of three.* The sweep over backpropagation: `G` 1.10, 1.19, 1.24; `P8` 1.20, 1.18, 1.04; `F10` 1.53, 1.89, 1.36; `A10` 1.46, 1.87, 1.74. The brief's cells are *about the same* for `G` and `P8` (every seed) and *more* for `F10` and `A10` (two seeds of three each). This is the part of the periods' effect that is the same on every seed: with periods the sweep is as close to backpropagation as it is in the dense network.
- *The alignment of the sweep's gradient does not explain it.* The cosine between the sweep's gradient and the exact gradient of the recurrent matrix at the end of the sweep's runs (the receipts' figures, which I did not recompute) is 0.74, 0.85, 0.84 for `G`; 0.60, 0.57, 0.75 for `F10`; 0.64, 0.66, 0.67 for `A10`; and 0.67, 0.68, 0.68 for `P8`. `P8` is not better aligned than `A10` by more than 0.03, and `F10` on the third seed is at 0.75. The alignment at the end therefore does not separate `P8`, in which the two rules need the same number of updates, from `A10`, in which they do not; the remark of memo section 22.12 that tied the two is not supported by three seeds.

**(d) The number of connections changes little. Holds on three of three seeds.** Updates to 80 on A under the sweep: `N5` 17,152, 14,336, 14,080; `A10` 13,824, 14,336, 13,824; `N20` 14,336, 13,056, 12,288; for comparison `F10` 13,312, 13,056, 12,544. Every description of `N5` and of `N20` against `A10` is *about the same*. `N5` needs more updates than `N20` on every seed (1.20, 1.10, 1.15 times); the order `N5`, `A10`, `N20` from most to fewest holds on the second seed (with a tie) and the third, and not on the first, where `N20` is above `A10`. `N5` reaches 90 on A on one seed, `N20` on two, `A10` on three. At 40 letters the three are within 5 points of each other on every seed. `N5` and `N20` draw their connections afresh, and all three share the defect of section 7: the comparison is among about 3, 6 and 12 working connections.

**(e) Changing connections against fixed ones. *About the same* on three of three seeds under both rules, and still not a test of changing connections.** `A10` over `F10`: 1.04, 1.10, 1.10 under the sweep and 1.09, 1.11, 0.86 under backpropagation; at 40 letters +2.7, −2.7, −1.6 and −0.3, −0.6, +2.4 points. Under the sweep `A10` is the slower of the two on every seed, by 4 to 10%. The replacement rule's defect repeats on the new seeds (section 7).

**(f) The period sets and the variants.**

- *`P4`, `P8`, `P16`: about the same, three of three.* `P4` over `P8`: 1.07, 1.15, 1.07; `P16` over `P8`: 1.03, 1.09, 1.10. `P8` is the first of the three to reach 80 on every seed, by one to five rows.
- *`last` against `step`: slower on three of three seeds, but "half as many updates again" is the first seed only.* `P8-last` over `P8`: 1.50, 1.09, 1.31 to 80 on A; 1.58, 1.16, 1.19 to 80 on R; 1.93, 1.23, 1.33 to 90 on A. The brief's cell is *about the same*. `P8-last` itself needs 0.83, 0.64 and 0.70 of `A10`'s updates, so on the two new seeds most of what periods give under the sweep is present when a cell uses only what arrives at the position of its change, and the weighted sum over the period adds the smaller part. Memo section 22.12 read the first seed the other way.
- *`source` against `step`: nothing added, three of three.* `P8-source` over `P8`: 1.00, 0.94, 1.07.

**(g) The length test, and period 16.**

- *`P16` at 40 letters:* 82.8, 94.1, 67.3, which is 13.2, 2.4 and 28.0 points below `P8`. By the brief's mark of 15 points it is *lower* on one seed of three (the cell is *about the same*), both against `P8` and against `A10` (−9.0, +1.6, −19.3; prediction P42.7 did not occur). The low score of the first seed therefore repeats on one of the two new seeds, at twice the size, and is absent on the other.
- *`P16` is the only configuration with a large loss between 12 and 40 letters:* −11.8, −1.3 and −26.4 points. The largest loss of any other run is 5.1 (`P8-last` on the third seed; `A10` under the sweep loses 5.0 on the same seed). On the first and third seeds `P16` is the lowest of the fifteen runs at 40 letters; on the second it is not among the lowest three.
- *During training the loss is present on all three seeds.* At 10,240 updates `P16` is at 71.0, 71.7 and 55.0 at 40 letters against 85.7, 84.0 and 89.7 at 12 letters; on the second seed the two meet at about 14,000 updates, on the other two they do not meet. `P8`'s two scores are within about 3 points of each other from 8,000 updates on, on every seed.
- *`P4` and `P8` lose nothing at 40 letters* on any seed (at most 1.9 points below each other and at most 1.2 points between 12 and 40 letters).

**What stands of memo section 22.12, in one table.**

| Reading of the first seed | On three seeds |
|---|---|
| 1. Ten local connections a cell are enough, at 1.7 times the dense network's updates under backpropagation and 2.4 under the sweep | Holds on three of three: 1.69 to 2.12 and 2.33 to 2.68 |
| 2. Periods take most of that cost away; `P8` is about the same as `G` on every measure under both rules; `P8` reaches 90 before `G`; `P8` equals `G` at 40 letters | Under the sweep, against the sparse network: holds on three of three. Against `G`: *about the same* on two of three under the sweep and *more* under backpropagation. Reaching 90 first: one of three. At 40 letters: three of three |
| 3. `P8-last` needs half as many updates again as `P8`; `P8-source` adds nothing | `P8-last` is slower on three of three, by 50%, 9% and 31%; `P8-source` adds nothing on three of three |
| 4. Period 16 costs at 40 letters | Present on two of three seeds (13.2 and 28.0 points below `P8`; 2.4 on the other); one of three by the 15-point mark |
| 5. The number of connections and the neighbourhood change little | Holds on three of three |
| 6. The rules differ only where the cells are sparse and have no periods | The ratios hold on three of three; the supporting remark on the cosines does not |

## 7. The replacement rule on each seed (part G)

The figures come from the final weights files of the 33 runs with changing connections (eleven a seed) and from their receipts. A connection's *strength* is the sum of the magnitudes of its weights over the three gates (in `P8-source` over the steps as well); its *relative strength* is its strength divided by the mean strength of its cell's connections. An *event* is one application of the replacement rule, every 256 updates. Ranges are over the eleven runs of the seed.

| Quantity | Seed 73129 | Seed 93683 | Seed 116089 |
|---|---|---|---|
| Share of connections replaced per event, first 4,096 updates | 6.19 to 8.12% | 6.18 to 8.20% | 6.05 to 8.20% |
| The same, last 4,096 updates | 17.20 to 19.99% | 16.66 to 20.38% | 17.21 to 20.55% |
| The same, whole run | 13.86 to 16.24% | 13.10 to 16.09% | 13.50 to 16.34% |
| Share of the final connections that were never replaced | 0.459 to 0.576 | 0.462 to 0.593 | 0.465 to 0.582 |
| Share made at one of the last two events (their median relative strength is 0.00 in every run) | 0.348 to 0.407 | 0.335 to 0.413 | 0.345 to 0.413 |
| Share replaced more than 1,024 updates before the end and still present | 0.059 to 0.131 | 0.050 to 0.131 | 0.066 to 0.131 |
| Connections per cell at or above half the cell's mean strength: `A10` sweep, `A10` backpropagation, `P8` sweep, `P8` backpropagation, `W10` (of 10) | 6.27, 6.06, 6.02, 5.93, 6.27 | 6.19, 5.92, 6.20, 5.87, 5.98 | 6.20, 5.88, 6.12, 5.87, 6.27 |
| The same, `N5` (of 5); `N20` (of 20) | 3.18; 12.39 | 3.23; 12.18 | 3.24; 12.34 |
| The same in `F10`, whose connections are fixed: sweep, backpropagation (of 10) | 8.79, 8.26 | 8.61, 8.19 | 8.71, 8.26 |

In every one of the 33 runs the connections made 256 updates before the end have a median relative strength of 0.09 to 0.13, and at most 0.2% of them are at half their cell's mean. The connections that were never replaced have a median relative strength of 1.3 to 1.5, and the replaced connections that lasted 1.4 to 1.8.

**The defect repeats without change.** On each seed a new connection starts at zero, reaches about a tenth of its cell's mean strength in 256 updates, and is judged after 512 updates against half that mean, so that nearly every new connection is given up at its first judgement. At the end of a run a third to two fifths of the places for connections hold a connection without weight, and a cell works with about six of its ten connections (5.87 to 6.65 over the 27 runs with N = 10), against 8.19 to 8.79 in `F10`. The share replaced per event in the last 4,096 updates is two to three times the share in the first 4,096, on which theta was chosen.

**What this means for the comparisons.** `A10` against `F10` compares about six working connections with about nine, not changing connections with fixed ones; `N5`, `A10` and `N20` compare about 3, 6 and 12; `W10` against `A10` compares about six with about six, drawn from different sets. E42 therefore still does not say whether changing connections help, on any seed. The comparisons among `A10`, `P8`, `P4`, `P16`, `P8-last` and `P8-source` are between runs that share the defect to the same degree, and the readings on periods are not touched by it.

## 8. Computation (part H)

**The count.** Forward multiply-adds for one item at one position, from the wiring of the brief, not from the time the code took (the code multiplies full matrices by a mask, so its time is no measure). The byte table's layer and the readout's own two layers cost 397,824 whatever the number of cells. Each cell costs 1,536 for its three gates' reading of the 512 input values and 512 for the readout's reading of the cell. Its connections to other cells cost three multiply-adds each: 768 in `G` and 33 at N = 10.

| Configuration | Multiply-adds a position | Share of `G` | With the sums over a period taken first |
|---|---|---|---|
| `G` | 1,118,720 | 1.000 | |
| `F10`, `A10`, `W10` | 930,560 | 0.832 | |
| `N5`; `N20` | 926,720; 938,240 | 0.828; 0.839 | |
| `P4`, `P8`, `P16` (variant `step`), `P8-source` | 931,328 | 0.833 | 0.779, 0.774, 0.772; 0.775 |
| `P8-last` | 798,764 | 0.714 | |

Two statements follow from the wiring and not from any run. First, limiting the connections among cells to ten removes 17% of the multiply-adds, because the input map and the readout are dense (memo section 26). Second, **periods as specified do not lower the count in the variants `step` and `source`**: a cell of period p adds up, with its own weights, what crossed the input map and the recurrent map at each of the p positions, so both maps are crossed at every position, and the weights add a few multiplications. Only `last` crosses the maps at the positions of a cell's change alone. The same function can be computed with the weighted sums over the period taken before the maps are crossed, which gives the last column; I state it because it is a lower count that the same wiring allows, and it is not what the brief specified or what ran.

**Computation to the 80 threshold on A, relative to `G` under backpropagation** (multiply-adds a position times updates to the threshold, each as a ratio; the backward pass crosses the same maps under both rules, so the forward count is taken as proportional to the whole).

| Run | 73129 | 93683 | 116089 | Median |
|---|---|---|---|---|
| `G`, sweep | 1.10 | 1.19 | 1.24 | 1.19 |
| `F10`, sweep | 2.16 | 2.65 | 2.40 | 2.40 |
| `F10`, backpropagation | 1.41 | 1.40 | 1.76 | 1.41 |
| `A10`, sweep | 2.25 | 2.91 | 2.64 | 2.64 |
| `A10`, backpropagation | 1.54 | 1.56 | 1.52 | 1.54 |
| `P8`, sweep | 1.25 | 1.72 | 1.42 | 1.42 |
| `P8`, backpropagation | 1.04 | 1.46 | 1.37 | 1.37 |
| `P4`; `P16`, sweep | 1.33; 1.29 | 1.98; 1.87 | 1.52; 1.57 | 1.52; 1.57 |
| `P8-last`, sweep | 1.61 | 1.61 | 1.60 | 1.61 |
| `P8-source`, sweep | 1.25 | 1.61 | 1.52 | 1.52 |
| `N5`; `N20`; `W10`, sweep | 2.78; 2.35; 2.83 | 2.90; 2.67; 2.44 | 2.68; 2.37; 2.69 | 2.78; 2.37; 2.69 |

**Reading (c) restated in computation.** Under the sweep, the sparse network without periods costs 2.2 to 2.9 times the computation of the dense network under backpropagation to reach 80 on A, and the sparse network with periods 1.25, 1.72 and 1.42 times (median 1.42), against 1.10, 1.19 and 1.24 for the dense network under the sweep. At this threshold no configuration is below the dense network under backpropagation on any seed; the lowest is `P8` under backpropagation on the first seed, at 1.04. With the sums over a period taken first, `P8` under the sweep would be at 1.16, 1.60 and 1.32. At the 90 threshold `P8` under the sweep is at 0.85, 1.88 and 1.11 and under backpropagation at 0.67, 1.71 and 0.86; the values below 1 come from seeds on which the dense network reached 90 late, and the second seed shows the opposite.

## 9. The predictions rescored (part I)

**The brief's eleven predictions.** My scoring from the receipts and the check receipts agrees with the report's file `E42_PREDICTIONS_three_seeds.json` in all eleven: P42.1 to P42.5, P42.9 and P42.10 occurred; P42.6 has the cell *fewer*; P42.7, P42.8 and P42.11 did not occur. For P42.10 I find the same three cells that differ between the rules as the report lists (`F10` weight-items to 80 on A; `P8` updates to 80 on A and on R). **I found no disagreement with the worker's report** in any figure, description or outcome; the one remark is on the wording of the mark `*` (section 4).

**My expectations of memo section 22.8, on three seeds.**

| | Expectation | Chance I gave | Three seeds |
|---|---|---|---|
| X2 | `A10` reaches 80 on A within the dose under the sweep | 65% | Reached on three of three |
| X3 | `A10` needs more updates than `G` to reach 80 on A under the sweep | 60% | More on three of three (2.45, 2.95, 2.57 times) |
| X4 | `A10` reaches 80 on A in no more updates than `F10` under the sweep | 55% | No, on three of three |
| X5 | `P8` against `A10`, updates to 80 on A under the sweep | fewer 20%, same 40%, more 40% | Fewer on three of three |
| X6 | `P16` is lower than `A10` at 40 letters | 55% | By the 15-point mark on one of three (−9.0, +1.6, −19.3) |
| X7 | `P4` is lower than `A10` at 40 letters | 25% | No, on three of three (+2.7, +3.3, +6.8) |
| X8 | Some stage 1 configuration's description against `G` differs between the rules | 40% | Yes (three cells) |

## 10. Commands and running times

All from the repository root. The times are processor times of one thread on the host of this session; while this check ran, about fifteen other processes shared four processors, and the elapsed times were about eight times longer.

| Command | What it gives | Processor time |
|---|---|---|
| `python3 -I PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/e42_three_seeds_verify.py A` | section 2 | 3 seconds |
| `python3 -I …/e42_three_seeds_verify.py D E F H I` | sections 4, 5, 6, 8, 9 | 1 second |
| `OPENBLAS_NUM_THREADS=1 python3 -I -u …/e42_three_seeds_verify.py B C G seed93683` (and `seed116089`, `seed73129`) | sections 3 and 7 for one seed | about 2 minutes a seed |
| `OPENBLAS_NUM_THREADS=1 python3 -I -u …/e42_three_seeds_verify.py C` | the structure of section 3 alone, three seeds | 4 seconds |
| `OPENBLAS_NUM_THREADS=1 python3 -I -u …/e42_three_seeds_verify.py B seed93683 seed116089 cfg=F10 cfg=P8 cfg=P16 cfg=P8-last cfg=P8-source greedy` | the length test generated one byte at a time for 14 runs (section 3) | 2 minutes 21 seconds |

## 11. What I did not verify

- **Anything that needs torch.** The training itself; the gradients; the equality of the faster forward with the specification under both rules (check 9); the cosines quoted in 6(c); the CPU figures. They stand on the worker's receipts. One check that the brief runs with torch could be repeated without it, and was: check 11 (section 2).
- **The resumed runs, beyond their data.** That a run resumed from a saved state continues exactly as an unbroken run rests on the wrapper's own test on four dry runs. What I can see is that the resumed runs received the same batches as the unbroken ones, that their final files satisfy every structural condition, and that the reader reproduces their final scores.
- **Rows before the last.** The reader was run on the final weights. The thresholds are recomputed from the scores in the receipts, not from weights at those rows, which are not committed.
- **The answers of dry runs for the new seeds.** The brief's check 8 (answers recorded byte by byte) exists for seed 73129 only; for the new seeds the reader is checked on the final counts.
