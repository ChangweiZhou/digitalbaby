# Verification of E41 on three seeds and of E46 (17 November 2026)

Written by the outside reviewer. Nothing was trained and no code of another branch was executed. Every figure below comes from one script, `e41_e46_verify.py` in this folder, which reads receipts and weight files with `git show` at fixed commits, recomputes the scores from the counts in the receipts, and passes weight files through the reviewer's own numpy reader (`np_gru.py`). Section 11 gives the commands.

## 1. Summary

1. **E41's tables are right.** All 54 receipts are complete; my thresholds equal the worker's in 648 of 648 values and my descriptions equal the report's in 144 of 144 cells. The 42 committed weight files hash to their receipts, and the reader reproduces the last row of every one of them (section 7).
2. **Widening the network lowers the number of updates to the threshold and raises the computation.** At a quarter and at half of the pairs taught, the half width reaches 80% with the least computation in 24 of 24 cases (two scores, two rules, two fractions, three seeds). The half width is the narrowest width that was run, so the width of least computation has not been located; it may be narrower still.
3. **The sweep's cost relative to back-propagation depends on width and on the taught fraction.** At half of the pairs it needs a median of 1.10 times back-propagation's updates to 80% on held-out pairs and 1.19 times on reduction pairs; at the double width the two rules are equal within 15%. At a quarter of the pairs it needs 1.1 to 3.3 times as many, and at the present width it does not reach 80% on reduction pairs within the dose on any seed.
4. **With a tenth of the pairs taught no run reaches 80%.** Back-propagation at the half width rises throughout its 40,960 updates, to 48, 69 and 58% on the three seeds, and is still rising at the end. The sweep stays between 13 and 16%.
5. **The learners named for the long lives exist and are what their receipts say.** The six half-width births at half of the pairs (both rules, three seeds) are on the default branch, and the reader reproduces all ten counts of each last row.
6. **E46's sentence is reproduced:** by the three cells fixed beforehand, learning the first family does not need weight decay under either rule. Two things lie inside the bands of that sentence and should be carried with it. Under back-propagation two of three seeds need 1.28 and 1.50 times the updates to 80% without decay, which is the boundary of "stays". Under the sweep the final accuracy on reduction pairs is lower without decay on all three seeds, by 1.6 to 5.2 points.
7. **E46 licenses less for the long lives than the roster note's one line suggests.** It was run at the present width, with half of the pairs, for one birth. It says nothing about the half width, about fewer pairs, about retention, or about what the weights do over a life: without decay the weights are still growing at the end of the birth (section 9.5).

## 2. Terms

- **Rules.** S is the sweep, the approximate rule of this programme, which sends the error backward through each linear map with the signs of the weights and one scale per matrix. B is exact back-propagation.
- **Width.** The network has a feed-forward width H and a number U of recurrent units. The three widths are the half (H 256, U 128; 317,457 parameters), the present (512, 256; 1,124,113) and the double (1,024, 512; 4,211,985).
- **Taught fraction f.** The share of the operand pairs of each of the four operations of the first family (addition and subtraction modulo 89, maximum and minimum) that training presents: 0.10, 0.25 or 0.50. The other pairs are never presented.
- **Accuracy A.** The share of whole answers right on 300 never-presented pairs of addition and 300 of subtraction, each item with 12 distracting letters; the mean of the two.
- **Accuracy R.** The same on 1,000 never-presented pairs of addition and 1,000 of subtraction whose answer requires the reduction modulo 89, without letters. These are the pairs a network cannot answer without the modulus.
- **Threshold, three-row version.** The first scored row (rows are 256 updates apart) from which three consecutive rows are at or above the threshold. The single-row version takes the first row at or above it. E41 states its results on the three-row version.
- **Dose.** The number of updates of a run: 40,960 at the half width, 12,288 at the present width, 8,192 at the double width. "Not reached" always means not reached within the dose.
- **Weight-items.** E41's unit of computation: the parameter count times 64 (the batch) times the updates.

## 3. What was read

| Material | Commit | Check |
|---|---|---|
| E41, seed 73129, and the three-seed report `E41_REPORT.md` with `E41_SUMMARY_all.json` | 2722d1c1 (`claude/wonderful-edison-8hx6i6`) | 18 receipts |
| E41, seed 93683 | d4d95951 (`claude/e41-seed-93683`) | 18 receipts |
| E41, seed 116089 | 79ea3575 (`claude/e41-seed-116089`) | 18 receipts |
| The default branch | 7cabb13a | the 54 receipts are the same blobs there; 42 weight files |
| E46 | b76f5979 (`claude/wonderful-edison-8hx6i6`) | 18 receipts, 18 weight files, check receipts |

Every one of the 54 E41 receipts is complete at its full dose, has a row every 256 updates, records the runner hash 559a7ee9978639d7, the rate 0.001, the decay 0.25 and the batch 64, and has finite losses throughout. The batch digest at row 4,096 is the same in the six runs of each seed and fraction, so width and rule saw the same data. A, R and the taught accuracy were recomputed from the counts in every row and equal the stored values (part A of the script).

## 4. E41: updates to the threshold on three seeds

Median and range of the three seeds, three-row version. Where a seed does not reach the threshold the three seeds are listed in the order 73129, 93683, 116089, with "n.r." for not reached. At f = 0.10 no run reaches 80 or 90 on either score, and those rows are left out.

**Updates to 80% on A**

| Rule | f | Half | Present | Double |
|---|---|---|---|---|
| S | 0.25 | 17,408 (14,080 to 24,832) | 8,704; 9,984; n.r. | 6,400 (4,096 to 6,912) |
| S | 0.50 | 10,496 (10,240 to 12,288) | 5,376 (4,864 to 5,632) | 2,816 (2,816 to 3,328) |
| B | 0.25 | 7,680 (6,144 to 10,240) | 4,352 (3,584 to 4,864) | 4,608 (3,584 to 4,608) |
| B | 0.50 | 7,680 (7,424 to 11,520) | 4,352 (4,096 to 5,120) | 3,072 (2,816 to 3,328) |

**Updates to 80% on R**

| Rule | f | Half | Present | Double |
|---|---|---|---|---|
| S | 0.25 | 30,464 (24,576 to 35,840) | n.r.; n.r.; n.r. | n.r.; 4,096; n.r. |
| S | 0.50 | 15,872 (13,312 to 20,480) | 6,400 (4,864 to 6,656) | 3,840 (3,072 to 3,840) |
| B | 0.25 | 9,216 (7,424 to 16,896) | 5,888 (4,352 to 7,680) | 4,608 (3,072 to 7,168) |
| B | 0.50 | 8,960 (8,704 to 13,056) | 5,120 (4,096 to 6,400) | 3,328 (2,816 to 3,840) |

**Updates to 90%**

| Score | Rule | f | Half | Present | Double |
|---|---|---|---|---|---|
| A | S | 0.25 | n.r.; n.r.; n.r. | n.r.; n.r.; n.r. | n.r.; 7,680; n.r. |
| A | S | 0.50 | 18,944 (18,176 to 24,832) | n.r.; 6,400; 9,472 | 5,120 (5,120 to 5,376) |
| A | B | 0.25 | 14,592 (11,520 to 27,648) | n.r.; 7,424; n.r. | n.r.; n.r.; n.r. |
| A | B | 0.50 | 14,848 (9,728 to 18,432) | 9,216 (4,864 to 10,240) | 5,888 (4,608 to 6,656) |
| R | S | 0.25 | n.r.; n.r.; n.r. | n.r.; n.r.; n.r. | n.r.; n.r.; n.r. |
| R | S | 0.50 | 28,416; 30,208; n.r. | n.r.; 7,424; 9,472 | 6,400 (5,376 to 6,656) |
| R | B | 0.25 | 23,296 (17,920 to 35,840) | n.r.; 8,704; n.r. | n.r.; n.r.; n.r. |
| R | B | 0.50 | 20,224 (15,360 to 22,272) | 7,936 (7,424 to 9,216) | 6,400 (4,608 to 6,656) |

The first seed's figures are those I recomputed on 16 November. Two remarks on reading these tables.

- **A "not reached" at the wider widths is often a matter of dose.** The sweep's present-width run at f = 0.25 on seed 116089 is at 80.2% on A at its last row, 12,288; the three-row rule needs two further rows. The present and double widths have 48 and 32 scored rows against the half width's 160.
- **The two versions of the threshold differ often.** In 99 of the 324 cases (run, score, threshold) the single-row version gives an earlier row; in 18 of those the three-row version records no crossing at all, and where both record one the three-row row is later by a median of 1,792 updates and by up to 12,800.

## 5. E41: computation to the threshold

**Counts.** One forward pass uses every matrix entry once per item and position. From the layer shapes: `l1` 8·H, the input map of the recurrent layer 3·U·H, its recurrent matrix 3·U·U, `lm` U·H, `l2` H·H, `l3` 257·H.

| Width | Forward multiply-adds per item and position | Parameters |
|---|---|---|
| Half | 313,600 (2,048 + 98,304 + 49,152 + 32,768 + 65,536 + 65,792) | 317,457 |
| Present | 1,118,720 (4,096 + 393,216 + 196,608 + 131,072 + 262,144 + 131,584) | 1,124,113 |
| Double | 4,203,520 (8,192 + 1,572,864 + 786,432 + 524,288 + 1,048,576 + 263,168) | 4,211,985 |

The ratios half : present : double are 0.2803 : 1 : 3.7574 in multiply-adds and 0.2824 : 1 : 3.7469 in parameters, so E41's weight-items measure the computation to within 1%. An item with k letters has 10 + k input positions and k is uniform on 0 to 12, so a batch of 64 holds 1,024 positions of its own items on average; I count forward multiply-adds per update as 1,024 times the figure in the table. (The harness pads every item to the longest of its batch, which has 12 letters in 99.4% of batches, so about 1,408 positions are in fact computed. This factor, and the factor for the backward pass, are the same at every width and for both rules, whose backward passes use matrices of the same shapes.) In these units the forward multiply-adds are 15.8 to 16.0 times the weight-items.

**Computation to 80%,** median (range) of three seeds, in units of 10^11 weight-items; multiply by 16 for forward multiply-adds.

| Score | Rule | f | Half | Present | Double |
|---|---|---|---|---|---|
| A | S | 0.25 | 3.54 (2.86 to 5.05) | 6.26; 7.18; n.r. | 17.3 (11.0 to 18.6) |
| A | S | 0.50 | 2.13 (2.08 to 2.50) | 3.87 (3.50 to 4.05) | 7.59 (7.59 to 8.97) |
| A | B | 0.25 | 1.56 (1.25 to 2.08) | 3.13 (2.58 to 3.50) | 12.4 (9.66 to 12.4) |
| A | B | 0.50 | 1.56 (1.51 to 2.34) | 3.13 (2.95 to 3.68) | 8.28 (7.59 to 8.97) |
| R | S | 0.25 | 6.19 (4.99 to 7.28) | n.r.; n.r.; n.r. | n.r.; 11.0; n.r. |
| R | S | 0.50 | 3.22 (2.70 to 4.16) | 4.60 (3.50 to 4.79) | 10.4 (8.28 to 10.4) |
| R | B | 0.25 | 1.87 (1.51 to 3.43) | 4.24 (3.13 to 5.53) | 12.4 (8.28 to 19.3) |
| R | B | 0.50 | 1.82 (1.77 to 2.65) | 3.68 (2.95 to 4.60) | 8.97 (7.59 to 10.4) |

**How width acts** (ratios of the wider to the narrower run, per seed, three-row version, 80% on A; the figures on R are in part B of the script and say the same):

| Rule | f | Updates, present / half | Updates, double / present | Weight-items, present / half | Weight-items, double / present |
|---|---|---|---|---|---|
| S | 0.50 | 0.46, 0.47, 0.51 | 0.59, 0.58, 0.52 | 1.62, 1.68, 1.81 | 2.21, 2.17, 1.96 |
| B | 0.50 | 0.44, 0.55, 0.57 | 0.65, 0.69, 0.71 | 1.57, 1.95, 2.01 | 2.44, 2.58, 2.64 |
| S | 0.25 | 0.50, 0.71, n.r. | 0.74, 0.41, n.r. | 1.77, 2.51, n.r. | 2.76, 1.54, n.r. |
| B | 0.25 | 0.42, 0.58, 0.63 | 1.06, 1.00, 0.95 | 1.50, 2.07, 2.24 | 3.97, 3.75, 3.55 |

A doubling of both sizes multiplies the computation per update by 3.6 to 3.8. It divides the updates to 80% on A by 1.4 to 2.4 from the half to the present width; from the present to the double width it divides them by 1.7 to 1.9 under the sweep and by 1.4 to 1.5 under back-propagation at f = 0.50, and by nothing under back-propagation at f = 0.25. The computation to threshold therefore rises at every step in which both runs reach the threshold, on both scores: by 1.1 to 2.5 times at the first step and by 1.5 to 4.0 times at the second.

**The cheapest width.** For 80%, on A and on R, under both rules, at f = 0.25 and 0.50, on each of the three seeds, the half width needs the fewest weight-items: 24 of 24 cases under the three-row version and 24 of 24 under the single-row version. For 90% the half width is the cheapest in 16 of the 19 cases in which any width reaches it. The three exceptions are all under the sweep: on R at f = 0.50 the present width is cheapest on seeds 93683 and 116089 (the half width needs 30,208 updates on the first and does not reach 90% on the second), and at f = 0.25 only one double-width run reaches 90% on A.

Because the cheapest width is the narrowest that was run, E41 does not locate the width of least computation. A quarter-width cell would show whether the computation falls further or turns.

## 6. E41: the description of the width steps

E41 describes each step from a width to the next wider one by a table of four cases: *falls* if the wider run's count is below two thirds of the narrower's, *rises* if above 1.5 times, *stays* in between; if only the narrower run reaches the threshold, *rises* when the wider run's last recordable count is at least 1.5 times the narrower's count and otherwise *not decided*; if only the wider reaches it, *falls* when its count is at most two thirds of the narrower's last recordable count and otherwise *not decided*; *neither reaches* if neither does. A cell carries the description that two of its three seeds share, and is *mixed* otherwise.

Threshold 80, three-row version, the three seeds in the order 73129, 93683, 116089:

| Score | Rule | f | Updates, half to present | Updates, present to double | Weight-items, half to present | Weight-items, present to double |
|---|---|---|---|---|---|---|
| R | S | 0.25 | not decided ×3 | neither reaches, falls, neither reaches | not decided, rises, not decided | neither reaches, not decided, neither reaches |
| R | S | 0.50 | falls ×3 | falls ×3 | stays ×3 | rises ×3 |
| R | B | 0.25 | falls ×3 | stays ×3 | rises ×3 | rises ×3 |
| R | B | 0.50 | falls ×3 | falls, stays, stays | rises ×3 | rises ×3 |
| A | S | 0.25 | falls, stays, not decided | stays, falls, falls | rises ×3 | rises, rises, not decided |
| A | S | 0.50 | falls ×3 | falls ×3 | rises ×3 | rises ×3 |
| A | B | 0.25 | falls ×3 | stays ×3 | rises ×3 | rises ×3 |
| A | B | 0.50 | falls ×3 | falls, stays, stays | rises ×3 | rises ×3 |

At f = 0.10 every cell at threshold 80 is *neither reaches*.

**Against the worker's report.** I parsed the three description tables of `E41_REPORT.md` (thresholds 80, 50 and 90; 36 rows of four cells) and compared them with mine, and compared the twelve thresholds of each of the 54 runs in `E41_SUMMARY_all.json` with mine. The 648 thresholds are identical. The 144 descriptions are identical. One convention differs and is not an error: the report puts an asterisk on a cell when any seed's description differs between the two versions of the threshold (32 cells), whereas the description of the cell itself, the one two of three seeds share, differs between the versions in 19 of them. The brief's sentence admits both readings; the report should say which it uses.

## 7. E41: four statements, and the learners for the long lives

**(a) The sweep against back-propagation.** The ratio of the sweep's updates to threshold to back-propagation's, for the same seed, fraction and width. At a given width this is also the ratio of computation in weight-items. In CPU-seconds an update under the sweep costs 1.07 to 1.11 times one under back-propagation at every width (medians over nine runs: 0.0435 against 0.0402 at the half width, 0.1164 against 0.1092 at the present, 0.4042 against 0.3656 at the double).

| Threshold | f | Half | Present | Double |
|---|---|---|---|---|
| 80 on A | 0.50 | 1.07, 1.38, 1.37 | 1.10, 1.19, 1.24 | 1.00, 1.00, 0.92 |
| 80 on R | 0.50 | 1.22, 1.53, 2.29 | 1.04, 1.19, 1.25 | 1.15, 1.09, 1.00 |
| 80 on A | 0.25 | 1.70, 2.29, 3.23 | 2.00, 2.79, n.r. | 1.39, 1.14, 1.50 |
| 80 on R | 0.25 | 2.12, 3.31, 3.31 | n.r., n.r., n.r. | n.r., 1.33, n.r. |
| 90 on A | 0.50 | 1.03, 1.87, 1.67 | n.r., 1.32, 1.03 | 0.87, 1.17, 0.77 |
| 90 on R | 0.50 | 1.41, 1.97, n.r. | n.r., 1.00, 1.03 | 1.04, 1.17, 0.96 |

In every "n.r." of this table it is the sweep that does not reach the threshold; back-propagation reaches 80% in all 18 runs at f = 0.25 and 0.50 on both scores. Three things hold on three seeds.

1. At half of the pairs the sweep costs a median of 1.10 times back-propagation to 80% on A (range 0.92 to 1.38 over nine cases) and 1.19 times on R (1.00 to 2.29).
2. The ratio is nearest to 1 at the double width and largest at the half width, on both scores and at both fractions.
3. At a quarter of the pairs the sweep costs a median of 1.85 times on A (1.14 to 3.23 over the eight cases in which it reaches 80%). On R it reaches 80% in four of nine runs: at the half width on all three seeds, after 2.1 to 3.3 times back-propagation's updates, and at the double width on one seed. Its accuracy on R at the end of the half-width dose is 79.0, 81.7 and 80.5%, against back-propagation's 93.4, 96.6 and 91.5%.

So the condition of the adopted question, "at no greater computation to threshold than back-propagation", is met by the sweep only at the double width with half of the pairs taught, where the two rules are within 15% of each other. At half of the pairs the sweep is 4 to 25% above back-propagation at the present width and 7 to 129% above it at the half width, which is the width chosen for the long lives. At a quarter of the pairs it is further above at every width. Its cost in CPU-seconds per update adds 7 to 11% to each of these figures.

**(b) Width.** Section 5: updates to threshold fall with width under both rules, computation rises, and the half width is the cheapest to 80% in every case.

**(c) A tenth of the pairs.** No run reaches 80% on A or on R under either version; the highest R at the double width is 11.0, 16.8 and 15.2% under the sweep and 17.5, 21.1 and 21.4% under back-propagation. The only thresholds reached at f = 0.10 are 50% under back-propagation at the half width on seeds 93683 and 116089 (A at 17,152 and 28,672 updates; R at 23,552 and 31,232). The taught pairs are at 96 to 100% in all 18 runs, so the networks have memorised what they were shown.

The rise at the half width under back-propagation, seen on the first seed, holds on all three:

| Updates | A under back-propagation (three seeds) | A under the sweep |
|---|---|---|
| 4,096 | 14.2, 13.5, 22.5 | 4.3, 6.0, 10.2 |
| 8,192 | 24.7, 33.5, 33.7 | 6.8, 10.8, 10.2 |
| 12,288 | 28.8, 41.7, 39.7 | 6.5, 10.3, 11.7 |
| 24,576 | 39.3, 56.5, 47.8 | 6.8, 12.2, 12.7 |
| 40,960 | 48.2, 68.5, 57.7 | 12.8, 14.7, 15.7 |

Over the last 8,192 updates back-propagation is still rising by 0.5 to 0.7 points per 1,000 updates; the sweep by 0.3, 0.1 and 0.0. At equal updates the half width is above the wider ones under back-propagation (mean of A over rows 7,680 to 8,192: 25.1, 38.1, 36.7 at the half width; 16.4, 28.2, 22.2 at the present; 17.7, 22.1, 22.6 at the double), and at equal computation it is far above (48.2, 68.5, 57.7 at its last row against the present width's 16.2, 30.2, 27.7; the two doses cost 1.30 and 1.38 × 10^10 weight-updates). Under the sweep the double width is the highest at equal updates (12.0, 19.0, 18.1 against 6.3, 10.2, 10.8 at the half width) and every level is low. Whether back-propagation at the half width would reach 80% from a tenth of the pairs with a longer dose is open; the sweep shows no sign of it.

**(d) The learners for the long lives.** The roster note of 17 November (`OPUS_TASK_E44_L2_ROSTER_FIRST_LONG_LIVES_20261117.md`, lines 17 and 22) takes its learners from E41's half-width births at f = 0.50, back-propagation's for its first release and the sweep's for the second, world w using seed (73129, 93683, 116089)[(w − 1) mod 3]. Its reason is the width rule of the E44 brief: the half width if it reaches 80% on R at f = 0.50 with the fewest weight-items on at least two of three seeds under back-propagation.

- *The rule's condition holds on three of three seeds under both versions of the threshold.* Weight-items to 80% on R at the half, present and double width: 2.65, 4.60, 8.97 × 10^11 on seed 73129; 1.77, 2.95, 7.59 on 93683; 1.82, 3.68, 10.4 on 116089 (three-row). Single-row: 2.50, 3.68, 8.97; 1.77, 2.95, 7.59; 1.61, 2.58, 8.28.
- *The files are committed.* `git ls-tree` at 7cabb13a shows 42 weight files of E41 on the default branch: all 18 of seed 73129 and the 12 half- and present-width files of each later seed, as the brief asked. Each is the same blob as on its worker's branch, and the sha256 of each file equals its receipt's `weights_sha256` (42 of 42).
- *The reader reproduces the receipts.* For the six half-width births at f = 0.50 the numpy reader gives the ten counts of the receipt's last row exactly (A and the taught panel on four operations, R on two):

| File | A: add, sub of 300 | R: add, sub of 1,000 | Taught: add, sub of 300 | A | R |
|---|---|---|---|---|---|
| `e41_B_h256_f50_73129.pt` | 283, 284 | 951, 942 | 288, 292 | 94.50 | 94.65 |
| `e41_B_h256_f50_93683.pt` | 289, 288 | 945, 940 | 293, 292 | 96.17 | 94.25 |
| `e41_B_h256_f50_116089.pt` | 288, 289 | 917, 943 | 293, 293 | 96.17 | 93.00 |
| `e41_S_h256_f50_73129.pt` | 292, 283 | 938, 901 | 297, 291 | 95.83 | 91.95 |
| `e41_S_h256_f50_93683.pt` | 278, 283 | 918, 933 | 274, 291 | 93.50 | 92.55 |
| `e41_S_h256_f50_116089.pt` | 273, 284 | 907, 911 | 282, 290 | 92.83 | 90.90 |

- *Every committed file.* With the option `files=all` the reader was applied to all 42 committed files, at the three widths, under both rules and at the three fractions: its ten counts equal the receipt's last row in 42 of 42.

**R against A** (the report's P41.8). At the present width and f = 0.50, 80% on R is reached at the same row as on A on seed 93683 under both rules, and 1,024 and 1,024 updates later under the sweep and 1,280 and 768 later under back-propagation on seeds 73129 and 116089.

## 8. Disagreements with the worker's report and with the coordinator's review

There is no disagreement in any tabled number. The following are the differences I found.

1. **The report, P41.8.** It says the rows of R and A on seeds 73129 and 116089 "differ by 768 to 1,024". Under back-propagation on seed 73129 they differ by 1,280 (R at 6,400, A at 5,120). The outcome of the prediction is unchanged.
2. **The report, the asterisk** (section 6): a convention, not an error.
3. **The report, P41.5.** It gives the highest R under back-propagation at the double width and f = 0.10 as 17.5, 21.1, 21.5; the third is 21.45, which I print as 21.4. No difference.
4. **Review 9.156** says of E41 only that the two later seeds have workers, which is right. **Review 9.157 and the roster note** state that the half width is the cheapest to 80% on R at f = 0.50 under back-propagation on all three seeds; that is confirmed here under both versions of the threshold. The report's cost figures (9.90 + 1.55, 10.87 + 1.78 and 10.58 + 1.69 CPU-hours of updates and scoring) equal the sums over the receipts.
5. **Review 9.157 on E46** gives the sentence without its two qualifications (section 9.3). The numbers are not in dispute.
6. **What neither the report nor the review states** and the receipts show: the sweep's ratio to back-propagation grows as the network narrows and as fewer pairs are taught (section 7a); and the cheapest width is the edge of the range that was run (section 5).
7. **Not merged.** The default branch at 7cabb13a holds the 54 receipts, the three per-seed reports and the 42 weight files of E41, but neither the three-seed report (2722d1c1) nor anything of E46 (b76f5979: the report, the 18 receipts and the 18 weight files). `git merge-base --is-ancestor b76f5979 7cabb13a` fails, and so does the same command for 2722d1c1. The roster note cites E46 as a reason, so the branch `claude/wonderful-edison-8hx6i6` should be merged before the note's readers look for it.

## 9. E46: learning the first family without weight decay

### 9.1 The runs

Eighteen births from random weights at the present width with half of the pairs taught, 24,576 updates each: both rules, the three seeds, and three values of the weight decay of the optimizer's first parameter group, 0.25 (the standing value, the reference), 0.05 and 0. The byte table has no decay in any run. All 18 receipts are complete; each records its own decay in the field `decay`, in `optimizer.weight_decay` and in the groups read back at the start and at the end; all carry the runner hash d7e98b3001629589; and all 18 weight files are committed at b76f5979. The complete runs cost 13.96 CPU-hours of updates and 3.25 of scoring.

**The reference is anchored.** In all six runs at decay 0.25 the counts of A, R and the taught panel equal those of E41's present-width run of the same rule and seed at every one of its 49 rows, and the final `tensors_sha256` equals that of E26's birth. The batch digest at row 4,096 is one per seed and is E41's. This holds for the two later seeds as well, whose E41 runs were made on other hosts.

**The reader reproduces the receipts.** For the six births without decay, the numpy reader gives the twelve counts of the last row exactly (A and the taught panel on four operations, R on two, the length test at 12 and at 40 letters), the sha256 of each file equals the receipt's, and the norms of the seven weight groups computed from the file equal the receipt's to 0.1%. With the option `files=all` the same was done for all 18 files: the twelve counts and the seven norms equal the receipts' in 18 of 18.

### 9.2 The figures

Updates to 80% on R and on A (three-row version, single-row in brackets), and levels as the brief defines them: means over rows 24,064, 24,320 and 24,576. The length test is addition modulo 89 on 300 never-presented pairs with 40 letters, where training has at most 12.

| Rule | Decay | Seed | 80 on R | 80 on A | 90 on R | Level R | Level A | Level at 40 letters | 40 letters, last 8 rows |
|---|---|---|---|---|---|---|---|---|---|
| S | 0.25 | 73129 | 6,656 [6,656] | 5,632 [5,632] | 13,824 [12,288] | 96.30 | 97.78 | 96.44 | 91.71 |
| S | 0.25 | 93683 | 4,864 [4,864] | 4,864 [4,352] | 7,424 [6,400] | 98.83 | 99.11 | 96.44 | 97.08 |
| S | 0.25 | 116089 | 6,400 [4,608] | 5,376 [4,608] | 9,472 [8,192] | 98.15 | 98.56 | 97.11 | 97.75 |
| S | 0.05 | 73129 | 5,632 [5,632] | 5,120 [5,120] | 10,496 [8,704] | 96.80 | 97.44 | 98.56 | 97.92 |
| S | 0.05 | 93683 | 4,864 [4,864] | 4,608 [4,608] | 7,168 [6,400] | 99.07 | 99.28 | 99.11 | 97.96 |
| S | 0.05 | 116089 | 5,120 [5,120] | 5,120 [5,120] | 8,448 [7,936] | 96.87 | 97.28 | 96.56 | 96.54 |
| S | 0 | 73129 | 6,400 [6,400] | 5,632 [5,632] | 14,336 [8,960] | 92.83 | 94.94 | 95.56 | 94.75 |
| S | 0 | 93683 | 5,120 [5,120] | 4,864 [4,864] | 7,424 [7,424] | 97.28 | 97.44 | 97.56 | 97.00 |
| S | 0 | 116089 | 5,632 [4,608] | 4,608 [4,608] | 15,360 [10,496] | 92.93 | 94.28 | 94.22 | 94.42 |
| B | 0.25 | 73129 | 6,400 [5,120] | 5,120 [5,120] | 7,936 [7,936] | 95.13 | 95.89 | 94.56 | 93.58 |
| B | 0.25 | 93683 | 4,096 [4,096] | 4,096 [4,096] | 7,424 [5,120] | 97.18 | 98.50 | 98.00 | 97.46 |
| B | 0.25 | 116089 | 5,120 [3,584] | 4,352 [3,584] | 9,216 [6,912] | 94.73 | 94.72 | 93.22 | 94.29 |
| B | 0.05 | 73129 | 7,680 [6,912] | 6,912 [6,912] | 9,984 [9,216] | 97.60 | 97.89 | 98.00 | 97.71 |
| B | 0.05 | 93683 | 4,608 [4,608] | 4,352 [4,352] | 5,888 [5,888] | 98.85 | 98.89 | 99.22 | 99.12 |
| B | 0.05 | 116089 | 4,608 [4,608] | 4,096 [4,096] | 7,680 [6,400] | 97.42 | 97.72 | 97.44 | 96.50 |
| B | 0 | 73129 | 8,192 [8,192] | 7,680 [7,680] | 13,312 [10,752] | 94.72 | 96.06 | 95.44 | 95.54 |
| B | 0 | 93683 | 6,144 [6,144] | 6,144 [6,144] | 8,448 [7,168] | 96.73 | 97.33 | 97.44 | 97.21 |
| B | 0 | 116089 | 4,096 [4,096] | 3,840 [3,840] | 8,960 [6,912] | 96.15 | 96.06 | 95.22 | 95.50 |

These are the worker's figures: the twelve thresholds, the levels and the norm of the recurrent group of each run in `E46_SUMMARY.json` equal mine in 306 of 306 values.

**The size of the weights at the end** (Frobenius norm of each of the seven groups at row 24,576; the three seeds):

| Rule | Group | At the start | Decay 0.25 | Decay 0.05 | Decay 0 | Ratio, decay 0 to 0.25 |
|---|---|---|---|---|---|---|
| S | byte table | 45.11 | 45.22, 46.60, 47.51 | 43.05, 44.23, 44.46 | 43.17, 44.13, 44.27 | 0.95, 0.95, 0.93 |
| S | `l1` | 13.84 | 3.26, 3.30, 2.87 | 3.95, 4.39, 4.15 | 7.31, 8.45, 7.87 | 2.24, 2.56, 2.74 |
| S | input map | 22.66 | 36.30, 40.12, 37.26 | 60.75, 63.77, 61.27 | 89.45, 92.09, 89.65 | 2.46, 2.30, 2.41 |
| S | recurrent | 16.05 | 28.50, 28.29, 27.50 | 44.71, 44.25, 44.98 | 71.99, 70.63, 71.82 | 2.53, 2.50, 2.61 |
| S | `lm` | 13.08 | 26.12, 26.61, 25.51 | 42.96, 43.06, 42.75 | 69.79, 69.56, 70.71 | 2.67, 2.61, 2.77 |
| S | `l2` | 13.06 | 9.30, 9.87, 9.52 | 24.76, 23.86, 25.90 | 42.96, 40.65, 42.79 | 4.62, 4.12, 4.50 |
| S | `l3` | 9.25 | 26.51, 34.45, 26.60 | 25.77, 26.19, 25.75 | 42.29, 39.18, 39.74 | 1.60, 1.14, 1.49 |
| B | byte table | 45.11 | 46.50, 47.44, 47.74 | 44.15, 44.84, 45.08 | 43.86, 44.51, 44.78 | 0.94, 0.94, 0.94 |
| B | `l1` | 13.84 | 3.60, 3.95, 3.76 | 4.59, 5.37, 4.98 | 8.45, 9.47, 9.49 | 2.35, 2.40, 2.53 |
| B | input map | 22.66 | 36.90, 38.69, 38.15 | 55.90, 59.42, 56.67 | 80.36, 84.36, 81.68 | 2.18, 2.18, 2.14 |
| B | recurrent | 16.05 | 26.30, 27.10, 26.75 | 44.03, 43.09, 43.98 | 71.26, 69.97, 70.63 | 2.71, 2.58, 2.64 |
| B | `lm` | 13.08 | 23.89, 25.58, 24.87 | 43.58, 42.30, 42.88 | 71.41, 69.68, 70.84 | 2.99, 2.72, 2.85 |
| B | `l2` | 13.06 | 7.40, 8.53, 9.91 | 23.06, 21.80, 22.21 | 40.98, 39.61, 37.97 | 5.54, 4.64, 3.83 |
| B | `l3` | 9.25 | 22.60, 32.38, 26.86 | 23.66, 24.84, 27.03 | 41.86, 43.70, 47.44 | 1.85, 1.35, 1.77 |

### 9.3 The sentence, and what lies inside its bands

The brief forms three cells per rule from decay 0 against 0.25, each carrying the description two of three seeds share: updates to 80% on R (E41's four cases), the level of R, and the level at 40 letters (*lower* or *higher* if the difference is 15 points or more). Recomputed from the receipts:

| Rule | Updates to 80 on R (reference → decay 0) | Level R | Level at 40 letters | Sentence |
|---|---|---|---|---|
| S | stays: 6,656 → 6,400 (×0.96); 4,864 → 5,120 (×1.05); 6,400 → 5,632 (×0.88) | about the same: −3.47, −1.55, −5.22 | about the same: −0.89, +1.11, −2.89 | does not need decay |
| B | stays: 6,400 → 8,192 (×1.28); 4,096 → 6,144 (×1.50); 5,120 → 4,096 (×0.80) | about the same: −0.42, −0.45, +1.42 | about the same: +0.89, −0.56, +2.00 | does not need decay |

Both sentences are the worker's, and they follow from the rule fixed beforehand. Two facts lie inside the bands and are not carried by the sentence.

1. **Under back-propagation, two of three seeds are slower without decay, at the boundary of the band.** To 80% on R the ratios are 1.28 and 1.50, and to 80% on A they are 1.50 and 1.50 (5,120 → 7,680 and 4,096 → 6,144). The brief counts exactly 1.5 as *stays*; a threshold is located to one row of 256 updates, and one row later on either seed would have read *rises*. Under the single-row version seed 73129 is *rises* (5,120 → 8,192), as the report marks. The third seed is faster without decay (×0.80). To 90% on R the ratios are 1.68, 1.14 and 0.97.
2. **Under the sweep the final accuracy is lower without decay on all three seeds:** by 3.47, 1.55 and 5.22 points on R and by 2.83, 1.67 and 4.28 on A. The direction is the same for the mean of R over the last 4,096 updates (17 rows): −1.35, −0.96 and −5.34 points. Three seeds in one direction is weak evidence, since it would occur one time in four if decay made no difference, so this is a pointer and not a finding. Under back-propagation there is no such difference (−0.42, −0.45, +1.42). To 90% on R the sweep's ratios are 1.04, 1.00 and 1.62.

Decay 0.05 is described as *stays* and *about the same* in every cell under both rules, as the report says. Under back-propagation it ends higher than decay 0.25 on all three seeds: by 2.47, 1.67 and 2.68 points on R and by 3.44, 1.22 and 4.22 at 40 letters. At decay 0.05 the time constant of decay alone is 20,000 updates in place of 4,000.

### 9.4 The six changes of the audit

All six are in revision 2 of the brief and are visible in what was run.

| | Change | Seen in |
|---|---|---|
| 1 | The reference at decay 0.25 run by the same runner | six receipts at 0.25, equal to E41's rows and to E26's final tensors |
| 2 | Check 2 rewritten to estimate the decay from one update | `e46_qual/check2_73129.json`: estimates 0.24998 and 0.05001 in every tensor, the byte table bit-equal, residuals at most 3 × 10^-8 |
| 3 | The sentence names its cells, follows E41's four cases, and has "not decided" as its third outcome; the runs at 0.05 enter no sentence | the report's sentence section |
| 4 | The receipt records the run's own decay | the three fields in all 18 receipts |
| 5 | The dose is a full birth, 24,576 updates | all 18 receipts |
| 6 | Cost | forecast 16.2 CPU-hours, spent 17.2 on the complete runs and 2.9 lost to two container restarts (the report's figure for the loss) |

The recommendations were taken as well: the length test over the last eight rows is tabled, the seven groups are named, every receipt asserts its sample digests against E41's table, and the brief states that the answer is for the present width.

### 9.5 What E46 licenses for the long lives, and what it does not

**It licenses** the statement that a network of the present width, taught half of the pairs from random weights, learns the first family to the same threshold in about the same number of updates and to about the same level without weight decay, under both rules on three seeds, and that its accuracy at 40 letters, far beyond the 12 of training, does not depend on decay (differences between −2.9 and +2.0 points). So neither learning nor generalisation to longer items needs decay in that setting. The six births without decay are on file as learners for a later retention step.

**It does not license** the following, each of which the long lives touch.

1. **The half width.** The long lives take their learners at the half width, and the brief's own limit reads: "If the long-life world takes its learners at the half width, one half-width cell without decay is needed before a life without decay is run there." No birth without decay exists at the half width. The nearest evidence is in E44's calibration, which ran 24 short lives without decay at the half width from back-propagation's births: the first-visit level on reduction pairs at 256 updates is 0.471 without decay against 0.453 with it (means over eight worlds; `E44_REPORT.md`, the table of medians over families 1 to 7; the worker's figures, not recomputed here). For the arms `naive-wd0` and `replay-wd0`, which start from births made with decay, I consider that evidence sufficient to run them.
2. **Fewer pairs.** The record's cost of weak decay was at 15% of the pairs (review 9.2 and 9.3). E41 shows that at a quarter and a tenth the two rules separate strongly; E46 has nothing there.
3. **A life.** Without decay the weights do not settle within the birth. The norm of the recurrent group (mean of three seeds) is 16.0 at the start and, at 6,144, 12,288, 18,432 and 24,576 updates, 49, 59, 66 and 72 under the sweep and 48, 58, 65 and 71 under back-propagation; over the last 6,144 updates it still grows by 0.93 to 0.96 per 1,000 updates. With decay 0.25 it is level at 27 to 29 from 6,144 updates on. A life of 256 visits has 65,536 updates and one of 1,024 visits has 262,144. Whether the units saturate and learning slows at such sizes cannot be read from a birth. Neither the E44 brief nor the roster note records the norms; in the arms without decay they should be recorded at stated visits, which costs nothing.
4. **Retention.** E46 measures none, as its brief says. What is on file without decay is this. (i) In E43's arm of unrelated learning without decay, the first family is at 0 of 3,916 items after 4,096 updates in all six networks and nearly gone after 64. (ii) In E44's calibration without decay, one visit after its own a family keeps a median of 3.1% of its level, against 2.1% with decay, and the birth family falls from 0.947 to 0.099, against 0.084 with decay (`E44_REPORT.md`, the two tables of what is kept; the worker's figures). (iii) In E43's rehearsed arm the first family is kept at 95 to 105% of its starting level with and without decay (review 9.156; the worker's figures). In every case the loss is by overwriting within tens to hundreds of updates and decay plays no part in it. Decay alone removes unused knowledge only after about 2,000 updates, so it can matter only for knowledge that something else has protected for that long. In the roster's terms, the comparison that can show an effect of decay on what is kept is `replay-wd0` against `replay` after long absences; the calibration gives no reason to expect one between `naive-wd0` and `naive`.

## 10. What I could not verify

- The workers' checks that need the training library (the bit-equality of the present-width network with E26's, the forward and gradients at other widths, the dry runs). I verified what the receipts and files carry: completeness, hashes, digests and counts.
- Intermediate rows. The reader was applied to final weights only, since only those are committed; the curves are the receipts'.
- The double-width files of seeds 93683 and 116089, which by the brief were not committed.
- CPU-seconds, which belong to the workers' hosts.
- E44's and E43's figures quoted in section 9.5, which are their workers' and are the subject of other verifications.

## 11. Commands

From the repository root. The numerical library should be held to one thread when other jobs run on the machine.

| Command | What it shows | Time |
|---|---|---|
| `python3 -I PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/e41_e46_verify.py A B C D` | E41: receipts, thresholds, computation, descriptions against the report, the statements | about 10 s (95 s on a first run, while git reads the objects) |
| `OPENBLAS_NUM_THREADS=1 python3 -I PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/e41_e46_verify.py E` | the 42 file hashes; the six half-width births through the reader | about 50 s |
| `python3 -I PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/e41_e46_verify.py F H` | E46: receipts, the reference against E41 and E26, thresholds, levels, norms, cells, sentences, growth | about 3 s |
| `OPENBLAS_NUM_THREADS=1 python3 -I PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/e41_e46_verify.py G` | the six E46 births without decay through the reader | about 2 minutes |
| `OPENBLAS_NUM_THREADS=1 python3 -I PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/e41_e46_verify.py E G files=all` | every committed weight file of E41 (42) and E46 (18) through the reader | about 14 minutes on a loaded machine |
