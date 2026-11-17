# Verification of E49 (a limited interface) and of the second addendum of E43 (rehearsal; E42's networks), 17 November 2026

Read-only. Nothing was trained, and no code of a worker's branch or of the default branch was run. Every file was read with `git show`, one at a time. The forward computations are those of the reviewer's own numpy readers. The two scripts beside this note reproduce every figure in it; section 6 gives the commands.

- **E49** is on `claude/peaceful-noether-vcvrmf` at a36de661: eight runs of seed 73129, with the brief `OPUS_TASK_E49_A_LIMITED_INTERFACE_20261116.md` in revision 2 (74b56ab7).
- **The second addendum of E43** is on `claude/peaceful-hopper-8ofq4i` at 58ed6b25 (runs at 2aecccd2, merged into the default branch at 9d962ac1): sixteen runs.

Terms used below. *The sweep* (rule S) is the local rule of this programme, in which the error crosses each linear map by the signs of its weights times their mean magnitude; *back-propagation* (rule B) is the exact rule. *The first family* is the four operations add, sub, max and min under the modulus 89, which a network learns at its birth. *A* is the share of whole answers right on 300 held-out pairs each of add and sub with 12 letters between the operands; *R* is the same share on 1,000 held-out pairs each of add and sub that need a reduction by the modulus, without letters. A threshold is *reached* at the first scored row from which three successive rows are at or above it. *The dose* is 24,576 updates, scored every 256. *The byte's code* is the 512 values that the layer `l1` makes of a byte's row of the byte table; the cells read it through the input map.

## 1. Summary

**E49.** The worker's report is correct in every figure I recomputed.

- The reviewer's reader reproduces, from the committed final weights of all eight runs, every count of the last scored row (A, R and the taught panel) and every recorded answer of the length-test panel, 600 answers a run.
- The masks in the files are the reader's own draws. Every weight outside a mask is zero. Each start file is E42's start with the kept weights multiplied by the brief's factor, to the last bit.
- The eight changes of our audit are in what ran.

The result itself is one-sided.

- **Under back-propagation every limited setting learns add and sub more slowly than E42's `F10`, and none saves computation.** Only `in16` reaches 80 on R, at 20,480 updates against 9,472, which is 1.28 times `F10`'s multiply-adds. `I64` and `out8` end at 65 and 71 on R and are still rising at about 2 and 1 points per 1,000 updates. `I16` ends at 17.
- **Under the sweep no setting learns add or sub.** R is at or below 2.25 at every row with a limited input and ends at 7.8 with a limited readout. The three runs with a limited input are not rising, and their training loss has not fallen since update 4,096.
- **On continuation: none is worth its cost.** Under back-propagation it could show only that `I64` or `out8` reaches 80 near 32,000 updates, which is already *more* in updates and would most likely be *rises* in multiply-adds. Under the sweep there is nothing to continue. E47's trial does not answer E49's question, because it keeps a dense interface on 256 cells; it makes the question less pressing, because that interface does not grow with the number of cells.
- **An observation outside the brief's forms (section 2.7):** under the sweep with a limited input the network ends in a state in which its cells respond to the input less than they did before any training. The byte's code has shrunk to between a sixth and an eighth of its starting magnitude, and the cells' states at the `=` sign differ between items by a third of what they do before any training. The cause is not established.

**E43, addendum 2.** The worker's report is correct in every figure I recomputed.

- **Rehearsal keeps the first family.** With one item in two taken from the first family's taught half, its reduction score at update 4,096 is at 94.8 to 105.0% of its starting level in all twelve runs, and the unrelated task is at 100% on its taught panel. The reader reproduces all counts of the twelve committed networks exactly.
- **"At no cost" is the registered label and needs two qualifications.** The unrelated task is behind at update 256 in all twelve comparisons, and under back-propagation it reaches 80% one scored row later in two seeds of three; the rows are too coarse to say by how much. Half of every batch is spent on old items.
- **E42's `F10` and `P8` lose the first family as the present network does:** no answer of 31,684 is right from update 1,024 on in any of the four runs.
- **The defect is confirmed:** the two `P8` files at 4,096 hold no period coefficients, so those two networks cannot be rebuilt. No reported figure depends on the files.
- P43.15, P43.16 and P43.18 are right and P43.17 is wrong, as the worker scored them.

## 2. E49

### 2.1 The arms, and what one position costs

Every arm keeps E42's `F10` cell layer: 256 cells, each reading 10 fixed cells of its neighbourhood and its own previous value, every cell updated at every position. The byte table, `l1` and the readout's two last layers `l2` and `l3` stay dense in every arm. The arms differ in two maps only: the input map, through which a cell reads the byte's code, and `lm`, through which each of the 512 units of the readout's first layer reads the cells. The multiply-adds are counted from the reader's own index lists (part C of the script) and equal the receipts'.

| Arm | What a cell reads of the byte's code | What a readout unit reads | Input map | `lm` | Cells | Dense remainder | Total, one item at one position | Of `F10`'s |
|---|---|---|---|---|---|---|---|---|
| `F10` (E42's, the reference) | all 512 values | all 256 cells | 393,216 | 131,072 | 8,448 | 397,824 | 930,560 | 1.000 |
| `I64` | 64 values | 32 cells | 49,152 | 16,384 | 8,448 | 397,824 | 471,808 | 0.507 |
| `I16` | 16 values | 8 cells | 12,288 | 4,096 | 8,448 | 397,824 | 422,656 | 0.454 |
| `in16` | 16 values | all 256 cells | 12,288 | 131,072 | 8,448 | 397,824 | 549,632 | 0.591 |
| `out8` | all 512 values | 8 cells | 393,216 | 4,096 | 8,448 | 397,824 | 803,584 | 0.864 |

The count is of the products of the forward pass. An update is a batch of 64 items, and in the brief's convention its multiply-adds are 64 times the figure of one position, the positions of an item not being counted: 59.6 million for `F10` and 30.2, 27.0, 35.2 and 51.4 million for the four arms.

In the seed's draw an input value is read by 16 to 52 cells at 64 values a cell and by 1 to 18 at 16; a cell is read by 45 to 84 readout units at 32 cells a unit and by 6 to 27 at 8.

### 2.2 The weights through the reader (part A)

For each of the eight runs:

- the three masks in the final file equal the reader's draws (the input lists, the readout lists, and E42's ten connections), and the digests of the draws equal the receipt's and the brief's (`eb1dd941…`, `0e09b9ff…`, `e9c8783e…`, `3d8ba410…`);
- the largest weight outside a mask is exactly zero in the input map, in `lm` and in the recurrent matrix;
- the start file equals E42's start of `F10` (the start file of check 2's dry run, the same under both rules) with the kept weights of the input map multiplied by √(512 / K_IN) and those of `lm` by √(256 / K_OUT), with no difference in any bit; every other tensor is E42's;
- the reader's counts of right answers equal the receipt's last row on A (four operations of 300), on R (two of 1,000) and on the taught panel (four of 300);
- all 600 recorded answers of the length-test panel (300 at 12 letters, 300 at 40) are what the reader's forward gives byte by byte, and the counts of right answers equal the receipt's.

The per-byte evaluation used for speed agrees with `e49_interface_reader.forward` to 4 × 10⁻⁶ in every logit on a sample of each run.

### 2.3 The receipts recomputed (parts B and E)

Updates to a threshold by the three-row rule; "—" is not reached within the dose. Levels are means over the last three scored rows. The last column is the brief's measure for a run that does not reach 80: the mean of R over the last three rows less the mean of the three rows ending 2,048 updates earlier.

| Setting | Rule | 50 on R | 80 on R | 80 on A | Level of A | of R | at 40 letters | R, end against 2,048 earlier | max, min right of 300 at the end |
|---|---|---|---|---|---|---|---|---|---|
| `F10` | B | 6,400 | 9,472 | 8,704 | 93.2 | 94.1 | 92.6 | +2.1 | |
| `I64` | B | 18,176 | — | — | 74.0 | 65.2 | 61.0 | +7.9 | 290, 286 |
| `I16` | B | — | — | — | 23.8 | 16.9 | 16.7 | +3.4 | 293, 282 |
| `in16` | B | 11,776 | 20,480 | 16,640 | 85.2 | 82.5 | 66.1 | +0.65 | 291, 296 |
| `out8` | B | 14,592 | — | — | 73.8 | 70.8 | 68.0 | +8.6 | 292, 286 |
| `F10` | S | 9,472 | 15,104 | 13,312 | 91.2 | 89.3 | 89.1 | +0.7 | |
| `I64` | S | — | — | — | 1.3 | 0.9 | 1.3 | −0.25 | 102, 74 |
| `I16` | S | — | — | — | 1.2 | 0.9 | 1.1 | −0.02 | 91, 98 |
| `in16` | S | — | — | — | 1.9 | 1.4 | 1.2 | −0.35 | 117, 128 |
| `out8` | S | — | — | — | 10.6 | 7.75 | 7.9 | +2.4 | 249, 259 |

All of these equal the worker's tables. Three further points follow from the receipts.

- **A and R count add and sub only.** Max and min are learned in every run under back-propagation (94 to 99% of 300), also in `I16`, where add and sub are at 24 and 28%. Under the sweep with a limited input they are right in a quarter to two fifths of the items, and with a limited readout in five sixths. So "does not learn the first family" means add and sub.
- **In multiply-adds** `in16` under back-propagation *stays* (1.13 times `F10`'s on A, 1.28 on R); `out8` *rises* under back-propagation on both scores and under the sweep on A; the other readings are *not decided*, as the brief foresaw for runs that do not reach the threshold.
- **What a seed alone changes.** E42's `F10` is now on file for three seeds under both rules (E42's branch at 44f91edc). It reaches 80 on R at 14,336 to 15,104 updates under the sweep (ratio 1.05) and at 7,936 to 9,728 under back-propagation (ratio 1.23; the report had two seeds and 1.19). Its levels differ between seeds by at most 3.4, 3.6 and 7.0 points under the sweep and 4.4, 3.5 and 5.1 under back-propagation. No difference of an E49 run from `F10` lies within these spreads. The smallest are `in16`'s levels of A and R under back-propagation, 8.0 and 11.5 points below.

**In the brief's sentence:** with each cell reading 64 or 16 of the input's 512 values and each readout unit 32 or 8 of the cells, a limit on a cell's connections and not on what it can know of the byte, the network does not learn the first family within the dose under either rule. It needs more updates than with the dense interface and ends lower on all three levels. Of the two sides, under back-propagation the limited readout (`out8`) is the one that does not reach 80 on R. Under the sweep the limited input (`in16`) is the lower. One seed: a description.

### 2.4 The checks and the audit's eight changes (part F and section 2.2)

The stage-0 receipts record all six checks as passed. Checks 2, 4 and 5 involve training or the harness and are the worker's record; I did not rerun them. What they assert about the forward is confirmed independently by section 2.2.

| Change of the audit | In what ran |
|---|---|
| 1. What the job tests: a limit on connections, not on what a cell can know | In the brief's revision 2, and the report's sentences use the wording |
| 2. A cell that cannot know the byte is the next job | In the brief; it is E50 |
| 3. Check 2 under both rules | Passed: the runner with both masks full gives E42's tensors (837dadc1… under the sweep, 13398546… under back-propagation) and its scores at all nine rows; the back-propagation reference was made first with E42's runner unchanged |
| 4. A self-check of the masked branches | Four lines of self-check 3 in check 1's output, equal to the coordinator's output; and section 2.2 is a stronger form of it on trained weights |
| 5. Multiply-adds to a threshold, and what the reading can be | As defined; recomputed in section 2.3 |
| 6. The digest of a draw | The receipts' digests equal the brief's four and the reader's |
| 7. What a seed alone changes | Printed in the report for the seeds then on file; completed in section 2.3 |
| 8. "Does not within the dose", and a run still rising | Used in the report; see section 2.5 for how far the measure can be trusted |

The masks hold at every scored row of every run (weights outside zero, the optimizer's two running averages there zero), and the batch digest at row 4,096 is E26's (`bbd71ab7…`) in all eight. The eight runs took 7.73 CPU-hours.

### 2.5 Were the runs still rising, and when would they reach 80? (part D)

The brief's measure compares two means of three rows. Single rows scatter by 3 to 4 points about a straight line under back-propagation, so that measure is noisy: `I64`'s last nine rows of R are 64.5, 62.9, 51.0, 63.8, 62.5, 61.1, 65.5, 67.3 and 62.8, and a line through them rises by 2.1 ± 2.4 points per 1,000 updates, which is not distinguishable from no rise. Over the last 8,192 updates (33 rows) the rise is clear. The table uses that window.

| Run | R, last three rows | Slope of R over the last 8,192 updates, points per 1,000 | Update at which that line reaches 80 (slope one standard error either way) | Slope of A; its date for 80 |
|---|---|---|---|---|
| `I64` B | 65.2 | +1.97 ± 0.27 | 31,800 (30,900 to 33,000) | +1.75 ± 0.22; 27,700 |
| `out8` B | 70.8 | +1.13 ± 0.25 | 32,500 (31,000 to 34,800) | +0.66 ± 0.21; 33,700 |
| `I16` B | 16.9 | +1.51 ± 0.08 | 66,000 (64,000 to 68,200) | +2.22 ± 0.12; 49,600 |
| `out8` S | 7.75 | +0.68 ± 0.05 | 131,000 (124,000 to 139,000) | +0.98 ± 0.08; 95,500 |
| `I64` S, `I16` S, `in16` S | 0.9 to 1.4 | +0.04, 0.00, −0.06 (± 0.02) | no date: the line does not rise | no date |

**How weak these dates are.**

- Each is a straight line continued for a third of the dose or more, on one seed.
- Every curve on file that reaches 80 slows before it. `F10` rises 5 to 10 points per 1,000 updates between 50 and 80 and under 1 afterwards. `in16` under back-propagation rises 3.4 between 50 and 80 and 0.5 ± 0.4 afterwards, at a level of 82.5. `I64` and `out8` under back-propagation are already down to 2.5 and 2.1 points per 1,000 since they passed 50. The dates for these two are therefore earliest dates, and a level that settles below 80 cannot be excluded.
- `I16` under back-propagation is in the early part of its rise, where the rate is still growing (2.0 ± 0.5 over the last 2,048 updates against 1.5 over the last 8,192), so its date may be too late as well as too early.
- The date of `out8` under the sweep is five times the dose and means only that the run is rising slowly.

**The three runs under the sweep with a limited input are not rising.** R is at or below 2.25 at every row of the last third of the dose, and the training loss, which fell to 1.08, 1.16 and 1.04 by update 4,096, is 1.15, 1.20 and 1.05 at the end.

**What a continuation would decide.**

- *Under back-propagation.* It would show whether `I64` and `out8` reach 80 on R, most likely near 32,000 updates if at all. For `I64` the reading in multiply-adds is *stays* only if 80 on R comes by update 28,023 (one and a half times `F10`'s multiply-adds); the line's date is later, so the likely reading is *rises*. `out8` already *rises*. A continuation therefore cannot turn the result into a saving: the best it can show is that a random sparse interface costs about as much computation to the threshold as the dense one and three times the updates.
- *Under the sweep.* Nothing: three runs are flat and one would need a multiple of the dose.
- *Against E47's trial.* E47 keeps the dense interface on one patch of 256 cells and adds cells that touch neither the input nor the readout. Its four runs at 1,024 cells, which another check is verifying, bear on whether cells can be added without the interface growing. They do not bear on whether the interface itself can be thinned, which is E49's question, so E47 has not decided what a continuation of E49 would decide. E47 does remove the reason to care: in its wiring the interface is a fixed 524,288 multiply-adds whatever the number of cells, and E49 shows that thinning it at random saves nothing to the threshold under back-propagation and stops add and sub from being learned under the sweep.

**Verdict: no continuation.** The finding that matters for the programme is the sweep's failure with a limited input, and a longer dose does not address it.

### 2.6 The predictions

Scored from the receipts; the worker's scoring is the same.

| | Prediction | Chance | Outcome |
|---|---|---|---|
| P49.1 | Checks 1 and 2 pass | 90% | Right |
| P49.2 | `I64`, back-propagation, reaches 80 on R | 75% | Wrong (65.2 at the end) |
| P49.3 | `I64`, the sweep, reaches 80 on R | 55% | Wrong (0.9) |
| P49.4 | `I16`, back-propagation, reaches 80 on R | 50% | Wrong (16.9) |
| P49.5 | `I16`, the sweep, reaches 80 on R | 30% | Wrong (0.9) |
| P49.6 | `I64` under back-propagation about the same as `F10` in updates to 80 on R | 50% | Wrong (more) |
| P49.7 | Stage 2 runs under at least one rule | 65% | Right (both) |
| P49.8 | `in16` the worse of the two under back-propagation | 60% | Wrong (`in16` reaches 80, `out8` does not) |

Two of eight. The six misses all lie on one side: the limited interface was expected to cost less than it does.

### 2.7 An observation outside the brief's forms: what the networks became (parts G and H)

The audit's recommendation g announced that the reviewer would look at the byte's code in the saved start and final files. Part H does so, with E42's `F10` of the same seed beside the eight runs.

| Network | Size of `l1` at the end over the start | Size of the kept input weights, end over start | Mean magnitude of the byte's code on the 46 bytes in use | Input map's size times the code's magnitude, against the start | Size of `l3`, end over start | Spread of a cell's state at the `=` sign between items (mean magnitude of the state) |
|---|---|---|---|---|---|---|
| The start (each setting's own start file) | | | 0.41 | 1 | | 0.096 to 0.098 (0.17) |
| `F10` B (E42's) | 0.28 | 3.29 | 0.21 | 1.74 | 3.3 | 0.466 (0.65) |
| `F10` S (E42's) | 0.31 | 3.59 | 0.15 | 1.30 | 3.8 | 0.402 (0.51) |
| `I64` B | 0.49 | 1.52 | 0.44 | 1.67 | 3.7 | 0.328 (0.51) |
| `I16` B | 0.53 | 1.17 | 0.42 | 1.20 | 5.0 | 0.213 (0.61) |
| `in16` B | 0.51 | 1.38 | 0.34 | 1.17 | 3.6 | 0.415 (0.54) |
| `out8` B | 0.26 | 3.15 | 0.19 | 1.45 | 4.6 | 0.279 (0.47) |
| `I64` S | 0.23 | 1.27 | 0.05 | 0.17 | 18.7 | 0.032 (0.07) |
| `I16` S | 0.21 | 0.78 | 0.07 | 0.13 | 21.3 | 0.033 (0.07) |
| `in16` S | 0.23 | 0.81 | 0.05 | 0.10 | 8.6 | 0.038 (0.06) |
| `out8` S | 0.23 | 3.45 | 0.13 | 1.14 | 14.2 | 0.110 (0.24) |

The spread is the standard deviation of a cell's state between the 300 items of the length-test panel at 12 letters, averaged over the 256 cells.

- **With a dense input map the network makes up for a shrinking code.** In `F10` and `out8` under either rule `l1` falls to a quarter or a third of its starting size under the standing decay, the code's magnitude halves or more, and the input map grows three to four times, so that the product stays above its starting value.
- **With a limited input under back-propagation the code is kept large instead.** `l1` stays at half its size, the code at 0.34 to 0.44, and the product at 1.2 to 1.7.
- **With a limited input under the sweep neither happens.** `l1` falls as in the dense settings, the code falls to 0.05 to 0.07, the kept input weights change little (0.78 to 1.27 of their starting size), and the product falls to 0.10 to 0.17 of its starting value. The cells' states then differ between items by 0.03 to 0.04, a third of what they do in the untrained network and a twelfth of `F10`'s, and their mean magnitude is 0.06 to 0.07. The last layer of the readout has grown 9 to 21 times.
- **The values a cell reads say less about the byte.** On each run's own lists, the closest two of the 20 bytes that the task must tell apart are, on the median cell, at 0.055 to 0.08 of the mean distance between two of them under the sweep with a limited input, against 0.26 to 0.34 under back-propagation and 0.39 to 0.44 at the start. No two bytes coincide on any cell in any run.

**What the answers are** (part G, from the worker's answers files, the length-test panel at 12 letters). Under the sweep with a limited input 270 to 300 of 300 answers have the right form, two digits and the end symbol. They are drawn from 45 to 49 different values, where the right answers have 85, and the five most frequent answers make up 43 to 54% of all. Each digit is right in 8 to 16% of the items, which is the rate of guessing among ten. Under back-propagation the same settings give 67 to 84 different answers and each digit right in 35 to 95%.

**What this does and does not establish.** These are measurements on the end states of one seed. They show that the three runs are not slow versions of the runs that learn: the network has moved to a state in which the cells receive a tenth of their starting input and respond to it only slightly, and in that state it produces the form of an answer, part of max and min, and nothing of add and sub. They do not show why the sweep goes there. The recorded cosines between the sweep's and the exact gradient of the masked maps (0.33 to 0.62 for the input map at the end, against 0.39 to 0.73 in the runs under back-propagation) do not separate the runs that fail from those that learn, and no cosine was recorded for `l1` or the byte table, which are the layers whose course differs.

## 3. E43, the second addendum

Further terms. *The unrelated task* is E43's: copy two symbols across up to 12 filler bytes, in bytes that no family uses. *The reduction score* is the mean, over add and sub, of the share of right answers on all pairs of the first family that need a reduction by the modulus (3,916 each), without letters. *Rehearsal* means that items of the first family's taught half are mixed into the stream of the unrelated task.

### 3.1 The rehearsed runs, from the receipts (part A)

Twelve runs of 4,096 updates: the six births (three seeds, each born under the sweep and under back-propagation, each continued under its own rule), without decay (`unrelated-rehearse`) and with the standing decay of 0.25 (`unrelated-rehearse-decay`). Each run's row 0 equals the row 0 of the same birth in the arm without rehearsal, tensors included.

**The share of each batch.** Every item is drawn from the unrelated stream or from the first family's taught half with equal chance. Recounting the brief's generator `Random(f"e43|{seed}|joint")` gives the receipts' counts at every row for all three seeds: at 4,096 updates 131,138 to 131,384 unrelated items and 130,760 to 131,006 items of the first family, so 49.9 to 50.0% of the items, 32 of a batch of 64 on average, are rehearsal. The four runs of a seed share one data digest.

**The two scores at every recorded row** (the first family's reduction score · the unrelated task's taught panel, percent), recomputed from the receipts and equal to the worker's table:

| Run | 0 | 64 | 256 | 1,024 | 2,048 | 4,096 |
|---|---|---|---|---|---|---|
| no decay, S, 73129 | 96.6 · 0 | 79.6 · 0 | 78.1 · 0 | 92.6 · 100 | 90.7 · 100 | 97.1 · 100 |
| no decay, S, 93683 | 98.7 · 0 | 97.9 · 0 | 89.0 · 3.3 | 96.5 · 99.3 | 98.7 · 100 | 98.9 · 100 |
| no decay, S, 116089 | 99.6 · 0 | 92.6 · 0 | 82.7 · 5.0 | 95.6 · 99.7 | 95.0 · 100 | 96.1 · 100 |
| no decay, B, 73129 | 98.1 · 0 | 94.2 · 0 | 83.9 · 74.3 | 91.9 · 100 | 90.6 · 99.7 | 96.7 · 100 |
| no decay, B, 93683 | 97.8 · 0 | 95.8 · 0 | 91.9 · 50.0 | 96.8 · 100 | 95.4 · 100 | 98.3 · 100 |
| no decay, B, 116089 | 94.0 · 0 | 94.4 · 0 | 90.4 · 88.0 | 92.6 · 100 | 95.1 · 100 | 98.7 · 100 |
| decay 0.25, S, 73129 | 96.6 · 0 | 78.1 · 0 | 78.0 · 0 | 88.0 · 98.3 | 90.5 · 99.7 | 93.5 · 100 |
| decay 0.25, S, 93683 | 98.7 · 0 | 97.7 · 0 | 87.6 · 2.3 | 95.2 · 100 | 97.7 · 100 | 98.3 · 100 |
| decay 0.25, S, 116089 | 99.6 · 0 | 91.4 · 0 | 85.4 · 5.0 | 95.7 · 99.7 | 96.5 · 100 | 97.4 · 100 |
| decay 0.25, B, 73129 | 98.1 · 0 | 94.1 · 0 | 81.4 · 75.3 | 87.8 · 100 | 89.1 · 100 | 93.1 · 100 |
| decay 0.25, B, 93683 | 97.8 · 0 | 95.7 · 0 | 93.6 · 50.7 | 95.4 · 100 | 96.2 · 100 | 96.9 · 100 |
| decay 0.25, B, 116089 | 94.0 · 0 | 94.2 · 0 | 87.8 · 88.7 | 93.2 · 99.3 | 95.4 · 100 | 91.3 · 100 |

**The first family is kept.**

| Arm | Rule | Reduction score at 4,096 as a share of its start (three seeds) | Lowest share at any row | Held-out add and sub at 12 letters, of 600: start → 4,096 |
|---|---|---|---|---|
| no decay | S | 100.5, 100.3, 96.5% | 80.9, 90.2, 83.0% (all at update 256) | 588 → 583, 589 → 592, 599 → 567 |
| no decay | B | 98.6, 100.5, 105.0% | 85.5, 94.0, 96.2% (256) | 591 → 570, 592 → 578, 563 → 578 |
| decay 0.25 | S | 96.8, 99.6, 97.8% | 80.8, 88.8, 85.7% (256) | 588 → 555, 589 → 589, 599 → 580 |
| decay 0.25 | B | 94.8, 99.0, 97.2% | 82.9, 95.6, 93.5% (256) | 591 → 558, 592 → 576, 563 → 544 |

- The reduction score counts all pairs, of which half were never taught and are not rehearsed; the held-out panel at 12 letters contains only such pairs. Both are kept, so what is kept is the operation and not a list of rehearsed items. The held-out panel ends 0 to 5.5 points below its start in ten of the twelve runs.
- There is a dip while the unrelated task is being acquired: at update 256 the reduction score is at 81 to 96% of its start, and the held-out panel at 506 to 564 of 600.
- The weights move as far as without rehearsal. Without decay the recurrent matrix changes by 0.88 (sweep) and 0.76 (back-propagation) of its norm at birth with rehearsal, against 0.81 and 0.61 without. The family is kept by the gradient of its own items, not by weights that stand still.

**The unrelated task, and what "at no cost" covers.** At update 4,096 its taught panel is at 100% in all twelve runs and its held-out panel at 99.3 to 100%. By the addendum's table every arm under each rule reads *kept at no cost*. That label rests on the level at the end. On the way there:

| Rule | Taught panel at update 256, with rehearsal / without (three seeds) | First scored row at 80%, with / without |
|---|---|---|
| S, no decay | 0.0 / 8.3; 3.3 / 44.3; 5.0 / 40.0 | 1,024 / 1,024 in all three |
| B, no decay | 74.3 / 98.0; 50.0 / 96.0; 88.0 / 100 | 1,024 / 256; 1,024 / 256; 256 / 256 |
| S, decay | 0.0 / 7.3; 2.3 / 45.0; 5.0 / 41.3 | 1,024 / 1,024 in all three |
| B, decay | 75.3 / 98.0; 50.7 / 96.3; 88.7 / 100 | 1,024 / 256; 1,024 / 256; 256 / 256 |

- The unrelated task is behind at update 256 in all twelve comparisons. Rehearsal therefore delays it.
- The size of the delay is not resolved, because nothing is scored between updates 256 and 1,024. Under the sweep it is less than 768 updates. Under back-propagation the threshold is passed between 64 and 256 updates without rehearsal and between 256 and 1,024 with it in two seeds of three.
- Per item of the unrelated task the rehearsed learner is not behind at the rows on file: at the row where it reaches 80% it has seen 32,755 to 32,911 unrelated items, against 65,536 (sweep) and 16,384 (back-propagation) for the learner without rehearsal at its row. The rows do not allow a finer comparison.
- The cost in computation is the rehearsal itself: one old item is processed for every new one. Only the share of one half was run.

### 3.2 The twelve committed networks through the reader (part B)

For each of the twelve files at 4,096 the sha256 equals the receipt's, and the reader's counts equal the receipt's row exactly on score 1 (all 31,684 items of the first family without letters, in six groups), on score 2 (four held-out panels of 300 at 12 letters) and on score 3 (the unrelated task's held-out and taught panels, built by the reviewer's reader of that task from the brief's words).

Twelve of twelve files pass on each of the three scores. The counts of the last row of every rehearsed run are therefore confirmed from the weights and not only read from the receipts.

### 3.3 E42's networks (part C)

`F10` is E42's layer of 256 cells with 10 fixed local connections a cell. `P8` adds periods of 1, 2, 4 and 8 positions, and its connections are those that E42's replacement rule left at the end of its run; no replacement was made here. Four runs of seed 73129 from E42's final files, no decay, on the unrelated stream of the main arm (the same data digest).

First family: reduction pairs / pairs without reduction / max / min, percent of all pairs, and the count of right answers of all 31,684 items.

| Network | Row 0 | Update 64 | Update 256 | Update 1,024 and after |
|---|---|---|---|---|
| E42 `F10`, S | 91.2 / 93.6 / 98.7 / 96.7 (30,118) | 1.0 / 0.1 / 9.5 / 2.7 (1,049) | 0.3 / 0.0 / 3.2 / 0.1 (281) | 0 |
| E42 `F10`, B | 95.1 / 94.3 / 97.6 / 96.8 (30,397) | 0 (0) | 0.0 / 0.0 / 0.1 / 0.0 (6) | 0 |
| E42 `P8`, S | 94.3 / 95.7 / 98.5 / 98.1 (30,623) | 0.3 / 0.3 / 28.8 / 11.1 (3,216) | 0.0 / 0.0 / 0.9 / 0.7 (132) | 0 |
| E42 `P8`, B | 97.7 / 97.7 / 98.1 / 98.4 (31,034) | 0.0 / 0.0 / 6.9 / 3.2 (800) | 0.0 / 0.0 / 0.0 / 0.1 (4) | 0 |
| The present network, six runs | 94.0 to 99.6 on reduction pairs (30,594 to 31,354) | 0.0 to 4.9 on reduction pairs (6 to 1,207 of all items) | 0 or 1 item | 0 |

- **The reduction pairs are gone by update 64 in all four runs** (1.0% or less), as in the present network.
- **Max and min go somewhat more slowly under the sweep in E42's networks:** at update 64 `P8` still has 28.8% of max and 11.1% of min, and `F10` 9.5% and 2.7%, where the present network has 0 to 3.7%. At update 256 the largest remainder is 3.2%. From update 1,024 nothing is left in any run.
- **The unrelated task is learned in all four** (taught panel 99.3 to 100% at 4,096), so the condition of the reading holds and the reading is *lost* in all four.

**The sentence "limited local connections and periods in the recurrent layer do not protect earlier knowledge" is supported** for these four networks of one seed: the loss is complete by update 1,024 and nearly complete by update 64. The slower loss of max and min at update 64 under the sweep is a difference of a few hundred updates at most and not protection.

**The reader on the files.**

- *The start.* For all four runs E42's final file has the sha256 that the receipt records as its start, and the reader's score 1 on that file equals the receipt's row 0 in all six groups of items. For `P8` this uses the period coefficients of E42's file, and the per-byte evaluation used for speed equals `e42_cells_reader.forward` item by item on 600 items.
- *`F10` at 4,096.* Both files have the receipt's sha256, and their recurrent weights outside E42's mask are exactly zero, so the limit on connections held through the run. The reader gives the receipt's counts: no answer of the first family right, and 289 and 298 of 300 (sweep) and 293 and 300 (back-propagation) on the unrelated task's held-out and taught panels.
- *`P8` at 4,096.* Both files have the receipt's sha256, but they cannot be evaluated as they are (section 3.4).

### 3.4 The missing coefficients

In `P8` a cell of period p combines the p steps of its period with learned coefficients `a` (a table of 8 by 256), which E42's file holds and which the optimizer changed in these runs: by 0.44 of their norm under the sweep and 0.40 under back-propagation at update 4,096, by the receipts. The two `P8` files at 4,096 hold the thirteen tensors of the present network's state and nothing else.

- **What is missing:** the coefficients `a` at 4,096. The mask, the periods and the phases are missing too but did not change, so they can be taken from E42's final file.
- **Does it affect a reported figure? No.** Every score in the receipts was computed in the run, on the network in memory. The receipts' `tensors_sha256` covers the state only.
- **What it prevents:** rebuilding the two networks. The reader cannot confirm the two `P8` rows at 4,096, and no later measurement or run can start from these files.

- **How far the files are from the networks.** With E42's final coefficients put in place of the missing ones, the reader gets 187 and 208 of 300 on the unrelated task's held-out and taught panels under the sweep and 179 and 198 under back-propagation, where the receipts have 300 and 300 under each rule. With the coefficients of the start, one over the period, it gets 19 and 32, and 29 and 42. The first family's score 1 is zero in all six groups with E42's coefficients as well, which agrees with the receipts but does not confirm them.

- **What must be re-run or re-stated.** Nothing must be re-stated in the figures. The report already says under "Limits" that the two networks cannot be rebuilt; its list of committed weights should say the same beside the two files, since "16 weight files at 4,096 checked against receipts" reads as sixteen usable networks. A re-run of the two runs with `a` saved (about 0.4 CPU-hours by the worker's figure) is needed only if those two networks are to be measured or continued; it would also show whether the run repeats bit for bit.

### 3.5 The predictions (part D)

| | Prediction | Chance | Outcome |
|---|---|---|---|
| P43.15 | `unrelated-rehearse`, the sweep: *kept at no cost* | 60% | Right (three of three seeds) |
| P43.16 | `unrelated-rehearse-decay`, the sweep: *kept at no cost* | 65% | Right (three of three) |
| P43.17 | With rehearsal the taught panel reaches 80% no more than twice as many updates later, two of three seeds, under each rule | 60% | Wrong as scored on the rows: under the sweep three of three (the same row), under back-propagation one of three (row 1,024 against row 256 in the other two). The rows cannot tell a factor of two from a factor of four. |
| P43.18 | E42's `F10` and `P8`: *lost* in all four runs | 85% | Right |

The worker's scoring is the same.

## 4. What the two results say together about protection

Protection of the first family was obtained in one way and not in two others. **It was obtained by rehearsal:** when half of the items are old ones, the operation is kept on taught and untaught pairs alike, under both rules, with and without decay, while the weights move as far as they do when the family is lost; the price is one old item for every new one and a delay of the new task that the rows do not resolve. **It was not obtained by the structure of the recurrent layer:** ten local connections a cell, and periods, leave the loss as complete as in the present network, which is what one expects while every cell still reads the whole code of every byte and every readout unit reads every cell, so that each item of the new task changes every cell's input weights and the shared readout. **And E49 shows that thinning the interface at random is not a route to it.** It was not tested for retention, but it does not hide any byte from any cell, since any sixteen values identify the byte, so the new task would still reach every cell; and it makes the first learning itself slower under back-propagation and, under the sweep, stops add and sub from being learned at all. What remains as a structural candidate is a separation that depends on content: connections that are used only for one family (E48), or an input code in which a value stands for one byte, so that a cell receives nothing for the bytes of another task (E50). For the second of these E49 carries a warning: with sixteen values a cell the sweep did not learn in this network, so a job that limits the input must first show, under the sweep, that its own reference learns.

## 5. What I could not verify

- **The two `P8` networks at update 4,096** (section 3.4). Their scores at rows 64 to 4,096 are the worker's record.
- **Scores at rows before 4,096** in all sixteen runs of the addendum: only the files at 4,096 are committed, so the curves are read from the receipts. The same holds for the curves of E49, of which only the final row is confirmed from weights.
- **The content of the rehearsal stream.** I recounted how many items of each kind are drawn, and the reader's unrelated task reproduces the panels exactly. I did not regenerate E38's rehearsal stream of the first family, so that the rehearsed items are the taught half rests on the runner's text and on the receipts.
- **Anything that needs gradients or training:** E49's checks 2, 4 and 5, the cosines between the sweep's and the exact gradient, the training losses, and the optimizer's state. These are the workers' records.
- **Seeds.** E49 is one seed, and so are the four runs from E42's networks.

## 6. Commands

From the repository root; the refs must be present (`claude/peaceful-noether-vcvrmf`, `claude/peaceful-hopper-8ofq4i`, `claude/gallant-dijkstra-i9zgsr`). Times are for this container with other jobs running beside.

| Command | What it prints | Time |
|---|---|---|
| `python3 -I PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/e49_result_verify.py B C D E F G` | E49 from the receipts: thresholds, levels, checks, the arms' counts, slopes and dates, descriptions, seeds, predictions, the answers | 3 seconds |
| `python3 -I PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/e49_result_verify.py A` | E49's eight final files through the reader | about 1.2 CPU-minutes; 3.5 minutes here |
| `python3 -I PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/e49_result_verify.py H` | The maps, the byte's code and the cells' states at the start and the end | about 15 seconds alone; 75 seconds here |
| `python3 -I PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/e43_add2_verify.py A D` | The rehearsed runs from the receipts; the predictions | 3 seconds |
| `python3 -I PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/e43_add2_verify.py B` | The twelve rehearsed networks through the reader | about 4 CPU-minutes; 11 minutes here |
| `python3 -I PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/e43_add2_verify.py C` | E42's networks: receipts beside the present network; the reader on E42's final files and on the files at 4,096 | about 6 CPU-minutes; 13 minutes here |
