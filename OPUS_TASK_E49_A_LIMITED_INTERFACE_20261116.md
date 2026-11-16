# E49: a limited interface — each cell reads a few of the input's values and each readout unit a few cells (16 November 2026, revision 2)

Evidence level E2; one seed, a first measurement. The job is the one the audit of E47 names as next (its recommendation 1), and the user asked for it to be drafted on 16 November. The specification, the specification file and the predictions are mine. Revision 1 (7da5c453) was audited by the reviewer (`AUDIT_E49_LIMITED_INTERFACE_BRIEF_20261116.md` on `claude/vibrant-cerf-sje726` at d6fde19b), which required eight changes, none to the network, the settings or the runs, and stated that after them no re-audit is needed. Revision 2 makes them; the last section lists them. **Released.** One worker, best an E42 worker; it can run beside E47's trial.

Shell names: `D=origin/codex/minifly-consolidated-20261023; M=MOD97_LOCAL_RULE_20261106; K=PROJECT_DOCUMENTS/COORD_CHECKS_20261116`. Host: Linux x86-64, Python 3.13.18, torch 2.15.0+cpu; at most three processes, one thread each; launch outside any wrapper with a time limit. Edit nothing existing.

## The requirement this job addresses, and the part of it that is tested

The user, 16 November: "biological cells obviously cannot see the whole input, so locality is essential and our current set up does not scale."

E42 limited what a cell reads of other cells. E47 confines reading the byte and being read to a boundary of 256 cells. Inside that boundary nothing is limited yet: each of the 256 cells reads all 512 values of the byte's code and each of the 512 readout units reads all 256 cells, 524,288 of a position's multiply-adds. E49 limits both, at 256 cells, and asks whether the network still learns.

**The masks limit the number of values a cell computes with, not what a cell can know about the byte.** The byte's code is a function of the eight numbers of the byte's row, spread over all 512 values, and a small part of it identifies the byte. On the 16 values each cell reads (seed 73129's draw), in E42's trained `F10` under either rule and at a start drawn from the constructors' distributions, no two of the 46 bytes the task uses coincide on any cell; the closest two of the 20 bytes the task must tell apart are at 0.05 to 0.24 of their mean distance on the worst cell and at 0.17 to 0.41 on the median cell, against 0.25 to 0.49 on all 512 values; and one linear rule on a cell's 16 values puts every byte in its class on all 256 cells (the reviewer's measurement, `e49_brief_checks.py`, part E). The user's statement is therefore tested in one sense: whether the network learns when each cell is connected to a small part of the input and each readout unit to a small part of the cells. Whether it learns when the part of the input a cell reads does not say which byte is present is not tested.

## The comparison with the record (working rule 1)

- **The reference is E42's `F10`**: 256 cells, 10 fixed local connections, the dense interface, on E26's birth data and dose. E49 with both masks full is `F10` exactly (self-check 1), so E42's runs of `F10` are the reference and are not run again. On E42's first seed `F10` learns the family more slowly than the full network (the reviewer's read; E42 is not yet merged and I have not read its report).
- **A trained network rests on the dense interface.** With the smaller half of each cell's input weights removed from E42's trained `F10` and nothing retrained, its score at 12 letters falls from 92 to 83 under the sweep and from 91 to 72 under backpropagation (the reviewer's measurement). That says a dense-trained network uses its whole input; it does not say whether a network trained with a limited input learns. With the random masks of any of the four settings put on E42's trained `F10` and nothing retrained, 0 to 4 of 300 answers are right (the reviewer's measurement): the trained network is not a starting point for a limited one.
- **Nothing on file limits the input or the readout of this learner.** The nearest record is the random sparse expansion of October (the mushroom-body store), which memorised; it had no learned recurrence.
- **Coverage, from the draws** (self-check 2, seed 73129): at 16 input values a cell, every input value is read by 1 to 18 cells; at 8 cells a readout unit, every cell is read by 6 to 27 readout units. No input value and no cell is left unread at any setting. One input value read by a single cell is a thin path; the settings at 64 and 32 have 16 to 52 and 45 to 84.
- **The dose is E26's birth, 24,576 updates**, which `F10` needs most of on its first seed.

## The network, as code

**`$K/e49_interface_spec.py` on the default branch is the specification** (self-checks without training; my output is `$K/OUTPUT_E49_SPEC_MAC.txt`). Copy it to `$M/e49_interface.py`, with its import of E42's file pointed at the runner's copy, and otherwise do not rewrite it. In words:

- **The cell layer is E42's `F10`:** 256 cells on E42's grid, each reading 10 cells of its neighbourhood and itself, fixed, every cell at period 1; E42's draws and start.
- **Input mask.** Cell i reads K_IN of the 512 values of the byte's code (the output of `l1`), the same ones for its three gates: `sorted(Random(f"e49|input|{seed}|{K_IN}").sample(range(512), K_IN))`, drawn for cells 0 to 255 in turn from one generator.
- **Readout mask.** Readout unit j (of the 512 of `lm`) reads K_OUT of the 256 cells: `sorted(Random(f"e49|readout|{seed}|{K_OUT}").sample(range(256), K_OUT))`, for units 0 to 511 in turn from one generator.
- **The digest of a draw** is the first 16 hexadecimal digits of the SHA-256 of `json.dumps(lists)` with Python's default separators, `lists` being the 256 sorted lists of the input mask or the 512 of the readout mask. For seed 73129: K_IN 64 `eb1dd941…`, K_IN 16 `0e09b9ff…`, K_OUT 32 `e9c8783e…`, K_OUT 8 `3d8ba410…`. `in16` and `out8` use `I16`'s lists.
- **At the start the runner prints** the least number of cells that read an input value and of readout units that read a cell, and stops if either is zero. For seed 73129 they are 16 and 45 at `I64` and 1 and 6 at `I16`; at K_IN = 16 the chance that a later seed leaves an input value unread is 0.14 (the reviewer's figure).
- **The start.** After E42's start, the kept weights of the input map are multiplied by √(512 / K_IN) and those of `lm` by √(256 / K_OUT), as E42 scaled the kept recurrent weights; the others are zero. Biases are untouched.
- **The two rules.** Backpropagation exactly. The sweep by E42's masked sign rule on each masked map: the error crosses the signs of the weights inside the mask times their mean magnitude; a weight outside a mask has no gradient.
- **No geometry is given to the input's values or to the readout units**, which have none in this network; each reads a random fixed subset. Giving them positions is a later variant.

## Settings, seed 73129, each under the sweep and under backpropagation

| Name | K_IN | K_OUT | Multiply-adds a position | Of which the interface | Run |
|---|---|---|---|---|---|
| `F10` (E42's) | 512 | 256 | 930,560 | 524,288 | the reference, on file |
| `I64` | 64 | 32 | 471,808 | 65,536 | stage 1 |
| `I16` | 16 | 8 | 422,656 | 16,384 | stage 1 |
| `in16` | 16 | 256 | 549,632 | 143,360 | stage 2, only if `I16` falls short |
| `out8` | 512 | 8 | 803,584 | 397,312 | stage 2, only if `I16` falls short |

With the interface limited, the byte's own code and the readout's two layers (397,824 multiply-adds, which do not grow with the cells) are most of the network, and a cell with its share of the interface costs 289 multiply-adds at `I64` and 97 at `I16`, against 2,081 in E42.

**Stage 1:** `I64` and `I16` under both rules, four runs of 24,576 updates. **Stage 2, without a further release:** if `I16` under a rule does not reach 80 on R within the dose, or its updates to 80 on R are *rises* against `F10`, or it is *lower* than `F10` on a level (below), `in16` and `out8` are run under that rule, to say which side costs.

**Runner:** E42's with this forward; E42's data, optimizer (the standing decay, masked weights staying at zero since they have no gradient and start at zero), dose and scores, every 256 updates. About 0.8 CPU-hours a run on E42's measured rate for `F10`; four to eight runs.

## Checks before the runs

1. **The specification file.** The runner's copy differs from `$K/e49_interface_spec.py` in its import line only; run on `$K`, the self-checks print what `$K/OUTPUT_E49_SPEC_MAC.txt` has: with both masks full, logits and gradients bit-equal to E42's `F10` under both rules; for each setting the counts, the coverage, gradients zero outside the masks, and the digests of the draws; and self-check 3: for each of the four settings, after the start, the logits of the masked forward equal those of `E42.forward` on the same network under both rules, bit for bit (every weight outside a mask is zero, so the two compute the same function), and under backpropagation the gradients of the two masked maps are equal inside the masks and the gradients of every other tensor are equal.
2. **With both masks full the runner is E42's `F10`, under both rules.** Under the sweep, a dry run of 2,048 updates has the `tensors_sha256` (837dadc1…) and the scores of E42's `e42_F10_S_dry2048_73129`. Under backpropagation no dry run of `F10` is on file: the worker first makes one with E42's runner unchanged (`F10`, backpropagation, 2,048 updates, seed 73129) and commits its receipt; the E49 runner with both masks full must give the same `tensors_sha256` and scores.
3. **The masks hold through a run:** at every scored row every weight outside a mask is exactly zero, and the optimizer's two running averages there are zero.
4. **Same data:** the batch digest at row 4,096 is E26's for the seed.
5. **Scores:** E42's check 12 on E26's six births, with full masks and every cell connected.
6. One dry run of 256 updates per setting and rule, with CPU time and a forecast. The dry runs commit their weights, the two masks and the answers on the length-test panel, as E42's check 8 did, so that the reviewer's reader can confirm the masked forward before the dose is spent.

## What is recorded

Per run: E42's scores every 256 updates (A, R, the taught panel, the length test at 12 and 40 letters); updates and multiply-adds to the 50, 80 and 90 thresholds on A and R, three-row version (multiply-adds to a threshold are the setting's multiply-adds a position, times 64 items, times the updates to the threshold; positions are not counted, as in E42's weight-items); at the first batch and at the end, the cosine between the sweep's and the exact gradient of each masked map; the mean magnitude of the kept weights of each map at every scored row. At the last row, the bytes of each answer given on the length-test panel (working rule 3). Starting and final weights with the two masks, on the remote (`git add -f`, listed from `git ls-tree`).

## How the result is stated (fixed beforehand)

Against E42's `F10` of seed 73129 under the same rule:

- **Updates to the 80 threshold** on A and on R: E41's table of four cases, read as fewer, more or about the same.
- **The same in multiply-adds.** Against `F10` the ratio is the ratio of updates times 0.507 (`I64`), 0.454 (`I16`), 0.591 (`in16`) or 0.864 (`out8`). At `I64` and `I16` this reading can only be *falls*, *stays* or *not decided*: *rises* would need more updates than can be recorded, and a run that does not reach the threshold is *not decided* in multiply-adds while it *rises* in updates. The report gives the two readings side by side, and the sentence on learning rests on the reading in updates.
- **The level:** the mean of A, of R and of the length test at 40 letters over the last three scored rows; *lower* or *higher* at 15 points or more, otherwise *about the same*.

**The sentence this job can license**, for this seed and each rule: with each cell reading K_IN of the input's 512 values and each readout unit K_OUT of the cells, a limit on a cell's connections and not on what it can know of the byte, the network **learns the first family** (reaches 80 on R within the dose) or **does not within the dose**; and whether it does so in about the same updates and to about the same level as with the dense interface. One seed: a description, not a claim.

**A run that does not reach 80 on R.** Under the sweep `F10` reaches 80 on R at 15,104 of 24,064 recordable updates, so a setting slower by a factor above 1.59 cannot be told from one that does not learn; under backpropagation the factor is 2.54. For such a run the report gives the mean of R over the last three scored rows and its difference from the mean of the three rows ending 2,048 updates earlier; a run in which that difference is 5 points or more is described as still rising at the end of the dose. Whether such a run is continued is my decision.

**What a seed alone changes.** E42's `F10` on seeds 93683 and 116089 is part of E42's stage 1. When it is on file, its updates to 80 on A and R and its three levels for the three seeds are printed beside E49's, and a difference from `F10` of seed 73129 that is smaller than the difference between two seeds of `F10` is marked as such. Until then the nearest figure is `G` under the sweep, whose three seeds reach 80 on R at 4,864 to 6,656 updates (a ratio of 1.37). The other two seeds follow for the smallest setting that learns, by my release.

## What it cannot show

- **A cell that cannot know the byte.** Sixteen values of the byte's code are enough to identify the byte, so no setting of this job makes a cell ignorant of the input. That needs an input code in which a value stands for few bytes, held fixed so that training cannot spread the byte over all values again. With each of the 46 used bytes turning on 11 values of its own, for example, a cell reading 16 values receives a non-zero input for 1 to 10 of the 20 bytes the task must tell apart (median 6) and for 0 to 7 of the ten digits (median 3). **It is the next job after this one**, with its own reference (the same code with the dense interface), its own start and its own audit.
- **More than one seed**, until released.
- **An input with geometry.** The 512 values of a byte's code have no positions; a random subset is not a neighbourhood. A future input with many tokens, which the user's statement looks to, would have one.
- **The byte's code and the readout's own layers** (`l1`, `l2`, `l3`), which stay dense and are now most of the network's computation. In particular each of the 512 readout units still reads all 512 values of the byte's code through `l2` at every setting. Their turn is after this.
- **More cells.** The job is at 256 cells; E47 varies the number.
- **Settings.** Learning rate and decay are E42's; the scale of the kept weights at the start is a choice. The kept weights start 2.83 or 5.66 times larger than the dense ones, so one update of 0.001 is 1.1% or 0.57% of a weight's mean size, against 3.2%. For comparison, in E42's trained `F10` the mean size of an input weight is 0.083 (sweep) and 0.079 (backpropagation) and of a weight of `lm` 0.079 and 0.066, from 0.031 at the start; `I64` starts at 0.088 and `I16` at 0.177 (the reviewer's figures).

## Predictions (mine)

| | Prediction | Chance |
|---|---|---|
| P49.1 | Checks 1 and 2 pass | 90% |
| P49.2 | `I64`, backpropagation: reaches 80 on R within the dose | 75% |
| P49.3 | `I64`, sweep: reaches 80 on R within the dose | 55% |
| P49.4 | `I16`, backpropagation: reaches 80 on R within the dose | 50% |
| P49.5 | `I16`, sweep: reaches 80 on R within the dose | 30% |
| P49.6 | `I64` under backpropagation is *about the same* as `F10` in updates to 80 on R | 50% |
| P49.7 | Stage 2 runs under at least one rule | 65% |
| P49.8 | If stage 2 runs under backpropagation, `in16` is the worse of the two: later to 80 on R, or not reaching it where `out8` does; if neither reaches it, the lower mean of R over the last three rows | 60% |

## Done

The runner, the check receipts, the four to eight runs with receipts and weights on the remote, and `E49_REPORT.md` with the tables (setting × rule: updates and multiply-adds to the thresholds in both versions, the three levels, the cosines), the descriptions against `F10`, the sentence per rule, and the outcome of each prediction. No readings beyond the forms above.

## What changed from revision 1 (the audit)

Made as the audit wrote them. The specification file gained self-check 3, which I ran: for all four settings the masked forward equals E42's dense forward on the same network bit for bit under both rules, and the gradients agree as stated.

| | Change |
|---|---|
| 1 | **What the job tests.** The masks limit how many values a cell computes with, not what it can know: any 16 of the 512 values identify the byte. I had called the job the direct test of the user's statement; it tests it in one sense only, and the brief now says which. |
| 2 | A cell that cannot know the byte needs a fixed input code in which a value stands for few bytes; named as the next job |
| 3 | Check 2 under both rules, with a dry run of `F10` under backpropagation made first by E42's runner |
| 4 | Self-check 3 of the masked branches |
| 5 | How multiply-adds to a threshold are counted, and what that reading can and cannot be at each setting |
| 6 | How a draw's digest is formed, with the four digests |
| 7 | What a seed alone changes, to be printed beside the result |
| 8 | "Does not within the dose", and how a run still rising at the end is described |
| — | Recommendations taken: stage 2 also when `I16` rises; dry-run files for the reader; the runner stops on an unread value; the start's scale under AdamW stated; P49.8's scoring; the masks on the trained network give 0 to 4 of 300 |
