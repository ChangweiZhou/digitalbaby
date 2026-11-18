# Audit of the E49 brief and its specification file at 7da5c453 (16 November 2026)

From the outside reviewer. Read-only: nothing was trained, torch is not installed in the reviewer's container, and no code from another branch was executed. The brief is `OPUS_TASK_E49_A_LIMITED_INTERFACE_20261116.md`, revision 1. The specification file is `PROJECT_DOCUMENTS/COORD_CHECKS_20261116/e49_interface_spec.py`, with the coordinator's output `OUTPUT_E49_SPEC_MAC.txt` beside it; the coordinator's account is review section 9.153. The job is the one that recommendation 1 and check J of `AUDIT_E47_LOCAL_INTERFACE_BRIEF_20261116.md` named as the next.

**Verdict.** The network is the one the recommendation asked for, and the specification file agrees with the brief's text as far as it can be read and checked without torch. A second reader, written from the text and computing every sum over index lists, gives the states and the logits of E42's saved `F10` when both masks are full. The six digests of the draws, the coverage and every count of the brief are reproduced. The scaling of the kept weights keeps the sizes of the drives at the start within 2% of the reference's. Eight changes are required before release. None alters the network, the settings or the runs.

The most important change concerns what the job tests. The brief and review section 9.153 call it the direct test of the user's statement that a cell cannot see the whole input. The statement can be read in two ways, and the job tests one of them. The masks limit how many values a cell computes with. They do not limit what a cell can know about the byte: on the 16 values that each cell reads, every byte the task uses remains distinct, on every cell, both in E42's trained networks and at a start (section 4). The words "the direct test" were first the reviewer's own, in recommendation 1 of the E47 audit, and are corrected here. The second reading needs a different input code and is named as the next job; it should not be an arm of this one.

The other required changes are these. Check 2 covers one of the two rules. No check compares the masked branches of the forward with a second form. The reading in multiply-adds cannot say *rises* at the two main settings, and the brief does not say so. A run that does not reach the threshold may be learning more slowly, because under the sweep the dose leaves room for a slowdown by a factor of 1.59 only. The formula of the digests is in the specification file and not in the text.

**Terms.**

- The *byte's code* is the 512 values `z1 = tanh(l1(book[byte]))`, computed from the eight numbers of the byte's row in the table `book`. The *input map* is `rnn.weight_ih_l0`, of 768 rows (three gates for each of the 256 cells) and 512 columns. `lm` (512 rows, 256 columns) carries the cells' states to the 512 *readout units*; the readout is `z2 = tanh(l2 z1 + lm h)`, and `l3` gives the 257 logits from `z2`.
- *K_IN* is the number of values of the byte's code a cell reads; *K_OUT* is the number of cells a readout unit reads. The settings are `F10` (512, 256: E42's, the reference), `I64` (64, 32), `I16` (16, 8), and the one-sided `in16` (16, 256) and `out8` (512, 8).
- *A* and *R* are E42's scores on untaught pairs at 12 letters and on untaught reduction pairs without letters. The *sweep* is the sign rule; *backpropagation* is the exact gradient. A threshold is *reached* at the first of three consecutive scored rows at or above it, so the last update at which it can be recorded is 24,064.
- The *used bytes* are the 46 bytes the task puts into the network: the start symbol, the four family codes `p q v w`, the ten digits, the four operation signs, `=`, and the 26 letters. The task must tell 20 of them apart. The letters stand between the operands and carry no information, so the network needs to know only that a byte is a letter.
- The *reader* is `e49_interface_reader.py` beside this audit: the draws and the forward written from the brief's text in numpy. For every cell and every readout unit it sums over the listed sources only; the specification file multiplies full matrices by masks.

Shell names: `D`, `M` and `K` as in the brief; `R=PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109`; `C=$R/e49_brief_checks.py` on `claude/vibrant-cerf-sje726`; `W=ed4431f2`, the tip of E42's worker branch `claude/gallant-dijkstra-i9zgsr`. `PYTHONDONTWRITEBYTECODE=1 python3 -I $C <letter>` runs one part (A to G) from the repository root; the whole output is `$R/OUTPUT_E49_BRIEF_CHECKS.txt`.

## 1. Point 1: are the two masks what the recommendation meant, with the same input values for a cell's three gates?

**Yes.** The recommendation was: "each cell reads a stated number of the 512 input values, each readout unit a stated number of cells, the kept weights scaled at the start as E42 scaled the recurrent ones". The brief does this with two fixed masks at 256 cells on E42's `F10`. One list of input values for a cell's three gates is right: a cell is the unit that owns three gate rows, and E42's recurrent mask is likewise one list of cells for the three gates.

**With both masks full the reader gives E42's `F10`** (`python3 -I $C B`). On E42's saved `F10` of seed 73129 (`git show $W:$M/exp45/e42/e42_F10_<S|B>_73129.pt`), 300 length-test items at 12 letters and 300 at 40:

| Network | Largest difference of a state | Largest difference of a logit | Every predicted byte the same | Whole answers right, both readers |
|---|---|---|---|---|
| `F10`, sweep | 2.7e-06 | 1.1e-05 | yes | 275 and 271 of 300 |
| `F10`, backpropagation | 1.8e-06 | 1.1e-05 | yes | 274 and 274 of 300 |

With the masks of each of the four settings, the reader by index and the E42 reader on matrices multiplied by the masks agree to 2.7e-06 in a state and 5.7e-06 in a logit. The reader therefore computes the masked forward of the text.

**What stays dense** is the table lookup, `l1` (4,096 multiply-adds a position), `l2` (262,144) and `l3` (131,584): 397,824 multiply-adds, which is 84% of `I64`'s and 94% of `I16`'s. In particular each of the 512 readout units still reads all 512 values of the byte's code through `l2`. The brief lists the three layers under what the job cannot show; the point about `l2` is added in required change 2.

**The counts are the brief's** (`python3 -I $C C`): 471,808, 422,656, 549,632 and 803,584 multiply-adds a position; 477,201, 428,049, 555,025 and 808,977 weights; 289 and 97 multiply-adds for a cell with its share of the interface, against 2,081.

Two remarks. First, the readout mask fixes how many cells a readout unit reads, not how many readout units read a cell; the second number varies from 6 to 27 at K_OUT = 8 (section 3). Second, the trained `F10` cannot serve as a starting point: with the random masks of any of the four settings put on its weights and nothing retrained, 0 to 4 of 300 answers are right under either rule (part B). This is a different measurement from the one the brief quotes, in which the largest half of each cell's input weights was kept and the score fell from 92 to 83 and from 91 to 72.

## 2. Point 2: is the scale of the kept weights at the start right?

**Yes, for the sizes of the drives.** `nn.GRU` and `nn.Linear(256, 512)` draw every weight uniformly between -1/16 and 1/16, with variance 1/768. A sum over K kept weights, each multiplied by √(512/K) or √(256/K), has the variance of the dense sum. On stand-in weights drawn from the constructors' distributions, with E42's start for the recurrent weights and 300 items of 12 letters (`python3 -I $C D`; mean of three draws):

| Setting | Input drive | Recurrent drive | State | Output of `lm` | Logit | Mean size of a kept weight |
|---|---|---|---|---|---|---|
| `F10` | 0.383 | 0.128 | 0.228 | 0.132 | 0.161 | 0.031 |
| `I64` | 0.379 | 0.129 | 0.227 | 0.131 | 0.159 | 0.088 |
| `I16` | 0.386 | 0.128 | 0.229 | 0.133 | 0.164 | 0.177 |

The entries are root mean squares over items, positions and units. The spread over cells of a cell's own mean squared input drive is 0.30 of the mean at `F10` and `I64` and 0.34 at `I16`, so single cells are not left with unusually weak or strong drives.

**The sign rule's scale follows E42's.** `masked` passes `MaskedSign` the mean magnitude inside the mask, as E42 does for the recurrent map; `MaskedSign` reshapes its inputs, so it accepts the three-dimensional inputs it is given here (`git show 7da5c453:$K/e42_cells_spec.py | sed -n '50,65p'`). The error that crosses a masked map has the dense variance on average. It differs by the number of readers: at `I16` an input value read by one cell receives from the cells 0.12 of the dense error variance and one read by 18 cells 2.25 times it; a cell receives from the readout 0.38 to 1.81 of the dense variance, over the three seeds. The byte's code also receives error through `l2`, which is not masked.

**A consequence of the scale, which the brief should state** (recommendation d). The kept weights are 2.83 times (`I64`) or 5.66 times (`I16`) larger than the dense ones. AdamW moves a weight by about 0.001 an update whatever its size, so one update is 3.2% of a weight's mean size in the dense maps, 1.1% at `I64` and 0.57% at `I16`. The sizes themselves are within the range in which this optimizer has trained weights of this network: E42's kept recurrent weights started near 0.151 and ended at 0.188 (sweep) and 0.178 (backpropagation), and in E42's trained `F10` the mean size of a weight of the dense input map grew from about 0.031 to 0.083 and 0.079, and of `lm` to 0.079 and 0.066. `I64` therefore starts at the size the dense maps reach when trained, and `I16` at the size of E42's kept recurrent weights. The mean magnitudes the brief records at every scored row will show what happens.

## 3. Point 3: are 64 and 32, and 16 and 8, the right settings, and is the rule for the one-sided settings sound?

**The settings are reasonable.** The ratio of two to one makes the mean number of readout units that read a cell (2 K_OUT: 64 and 16) equal to the number of input values the cell reads. `I16` puts a cell's connections to the input at the order of its connections to other cells (10 and itself). The two settings are a factor of four apart. Their total computation is 0.507 and 0.454 of `F10`'s, so `I16` saves little beyond `I64`; its value is as a test of few connections, not as a saving.

**Coverage, from the draws** (`python3 -I $C A`). Cells that read an input value, and readout units that read a cell: least to most, with the median.

| Seed | K_IN 64 | K_IN 16 | K_OUT 32 | K_OUT 8 |
|---|---|---|---|---|
| 73129 | 16 to 52 (32) | 1 to 18 (8) | 45 to 84 (64) | 6 to 27 (16) |
| 93683 | 20 to 47 (32) | 1 to 18 (8) | 42 to 85 (64) | 8 to 26 (16) |
| 116089 | 17 to 50 (32) | 1 to 16 (8) | 43 to 82 (64) | 6 to 29 (16) |

Seed 73129's row is the brief's. No input value and no cell is unread for these three seeds. At K_IN = 16 the chance that a seed leaves at least one input value unread by every cell is 0.14 (recommendation c).

**The rule for stage 2 can be computed and is sound, with one gap.** It needs `I16`'s updates to 80 on R and its three levels, and `F10`'s levels, which are on file (section 6). The one-sided settings reuse `I16`'s own lists, because the string of a draw names only its own side: `in16` has the input digest of `I16` (`0e09b9ff…`) and `out8` its readout digest (`3d8ba410…`). The attribution is therefore made on the same masks. The gap: a run of `I16` that reaches 80 on R much later than `F10`, with levels within 15 points, is a loss by the brief's own reading (*rises*) and does not start stage 2 (recommendation a).

## 4. What the masks limit and what they do not

**The user's statement** is: "biological cells obviously cannot see the whole input, so locality is essential and our current set up does not scale. Also it makes no sense for a cell to see one million tokens in a future NLP task." It can be read in two ways.

1. *Connections.* A cell is connected to a small part of the input. E49 tests this: a cell computes with K_IN of the 512 values.
2. *Information.* What reaches a cell does not determine the input. E49 does not test this. The input at a position is one byte, and its code is a function of eight numbers spread over all 512 values, so a small part of the code still identifies the byte.

**Measurement** (`python3 -I $C E`). For each cell, the vectors of the used bytes were restricted to the values the cell reads (seed 73129's draw). The distance between the closest two of the 20 bytes the task must tell apart is given as a share of the mean distance between two of them; likewise for the ten digits. The first entry is the least over the 256 cells and the second the median.

| Code | All 512 values: the 20; digits | 64 values a cell: the 20 | 64: digits | 16 values a cell: the 20 | 16: digits |
|---|---|---|---|---|---|
| E42's `F10`, sweep, trained | 0.254; 0.638 | 0.147, 0.230 | 0.418, 0.585 | 0.052, 0.170 | 0.145, 0.434 |
| E42's `F10`, backpropagation, trained | 0.294; 0.562 | 0.221, 0.295 | 0.441, 0.555 | 0.130, 0.256 | 0.222, 0.486 |
| A stand-in start, draw 0 | 0.386; 0.517 | 0.289, 0.381 | 0.378, 0.508 | 0.210, 0.352 | 0.269, 0.477 |
| A stand-in start, draw 1 | 0.487; 0.631 | 0.381, 0.467 | 0.454, 0.600 | 0.241, 0.405 | 0.349, 0.523 |

In all four codes and at both settings:

- No two of the 46 used bytes coincide on any cell's values. A rule that assigns a vector to the nearest byte's vector therefore tells every used byte apart on every cell.
- One linear rule on a cell's values, fitted for that cell, puts all 46 bytes in their class (each of the 20 bytes, or "a letter") on all 256 cells.
- On the median cell with 16 values, the closest two digits are at 0.43 to 0.52 of the digits' mean distance, against 0.52 to 0.64 on all 512 values.

How few values are needed before the identity is lost was measured on 1,024 other random sets of each size, for the two trained codes and the first stand-in start. Two of the 20 bytes are closer than 0.05 of the mean distance on 1 (sweep), 0 (backpropagation) and 0 (start) of the sets of 16 values; on 71, 0 and 0 of the sets of 8 values; on 456, 37 and 6 of the sets of 4 values; on 874, 591 and 348 of the sets of 2 values; and on all sets of one value.

**Conclusion.** Sixteen of the 512 values are enough to identify the byte: on every set examined (the 256 of the draw and 1,024 others, for each code) the 20 bytes stay distinct, and the least separation found is 0.045 of the mean distance. E49 limits what a cell computes with, not what a cell can know. The user's statement is tested in the first sense only. The two outcomes do not weigh the same. If the network does not learn at `I16`, the finding is strong: 16 connections are too few for this learner even though they carry the whole byte. If it learns, the finding is that few connections to a dense code suffice; it does not show that the network learns when a cell is ignorant of most of the input, and it does not bear on an input of many tokens.

**The job that tests the second sense** needs an input code in which a value stands for few bytes, so that a cell reading K_IN values receives nothing for most bytes and must obtain the rest from the cells it reads. The code must be fixed; a learned code can spread the byte over all values again. As an illustration, let each of the 46 used bytes turn on 11 values of its own and no other (506 of the 512 values). With the draws of the three seeds, a cell reading 16 values then receives a non-zero input for 1 to 10 of the 20 bytes (median 6) and for 0 to 7 of the ten digits (median 3), and no cell receives something for every digit. A cell reading 64 values receives a non-zero input for 9 to 20 of the 20 bytes (median 15 or 16) and for 3 to 10 digits (median 8); 11 to 20 of the 256 cells receive something for every digit.

**Decision: it belongs in "What it cannot show", with the next job named, and not in E49 as an arm.** Replacing the learned code is a second change beside the masks. It removes `book` and `l1` from training, it changes the mean square of the input, so the start's scale must be set again, and it needs its own reference, which is the same code with the dense interface. That is a job with its own specification file and audit. E49 as drafted stays one change away from `F10` and answers a question that matters for cost. The required changes 1 and 2 state the limit and name the next job.

## 5. Point 4: can the digests of the draws be reproduced from the text?

**The six digests of seed 73129 are reproduced** (`python3 -I $C A`), on Linux with Python 3.13.17, from `random.Random` alone. The draws are fully stated in the text: one generator for each mask, seeded by the stated string; cells 0 to 255, or units 0 to 511, in turn; each sample sorted. A full mask is `list(range(...))` and its digest does not depend on the seed.

**One finding: the digest's formula is not in the text.** It is in the specification file only: the first 16 hexadecimal digits of the SHA-256 of `json.dumps(lists)` with Python's default separators, where `lists` holds one sorted list for each cell or unit. Other natural readings give other digests for K_IN = 16 and seed 73129: `9580b5a0d46b891d` with compact separators and `a158a92278eb88b2` for one flat list, against `0e09b9ff0545e68d`. Required change 6 puts the formula in the text.

Digests for the three seeds, for the brief and for later seeds:

| Seed | K_IN 64 | K_OUT 32 | K_IN 16 | K_OUT 8 |
|---|---|---|---|---|
| 73129 | `eb1dd9417ca1a282` | `e9c8783e0efa8d13` | `0e09b9ff0545e68d` | `3d8ba41006917f42` |
| 93683 | `b1610d882b188d3a` | `462492aedd002fe7` | `c46461aceab7d285` | `7f7cdf0343b5f119` |
| 116089 | `424bf8e12cfd7318` | `e70796a202cad61d` | `235ebf10610593dd` | `e364e5fba0daa95d` |

Full masks, any seed: K_IN 512 `7b0701bb680e83f3`; K_OUT 256 `b27cc82e87e68c52`.

## 6. Point 5: is reading against E42's `F10` of one seed enough for a first job?

**Yes, as a description, with three conditions.** The comparison is within one seed: the same data, the same starting draw, and the reference on file. The readings can be computed in every case in updates. The conditions are stated after the figures.

**The reference** (`python3 -I $C F`; `git show $W:$M/exp45/e42/e42_F10_<S|B>_73129.json`):

| `F10` | Updates to 50, 80, 90 on A | On R | Levels: A, R, 40 letters | A setting is *lower* below |
|---|---|---|---|---|
| Sweep | 8,192; 13,312; 20,480 | 9,472; 15,104; 23,296 | 91.2, 89.3, 89.1 | 76.2, 74.3, 74.1 |
| Backpropagation | 5,632; 8,704; 15,360 | 6,400; 9,472; 19,200 | 93.2, 94.1, 92.6 | 78.2, 79.1, 77.6 |

**The four cases in updates**, for the 80 threshold. With D = 24,064:

| Rule, measure | `F10` | *Falls* below | *Rises* above | Not reached |
|---|---|---|---|---|
| Sweep, A | 13,312 | 8,875 | 19,968 | *rises* (D/x = 1.81) |
| Sweep, R | 15,104 | 10,069 | 22,656 | *rises* (1.59) |
| Backpropagation, A | 8,704 | 5,803 | 13,056 | *rises* (2.76) |
| Backpropagation, R | 9,472 | 6,315 | 14,208 | *rises* (2.54) |

Every case gives a statement. A setting that reaches the threshold later than the third column *rises*; one that does not reach it *rises* too, because D is at least 1.5 times `F10`'s count in all four rows.

**Condition 1: the reading in multiply-adds says less, and the brief should say so.** The brief does not define multiply-adds to a threshold. Taken as the setting's multiply-adds a position times 64 items times the updates, the ratio to `F10` is the ratio of updates times 0.507 (`I64`), 0.454 (`I16`), 0.591 (`in16`) or 0.864 (`out8`). At `I64` and `I16` a run *falls* in multiply-adds whenever its updates are below 1.31 or 1.47 times `F10`'s. *Rises* would need 2.96 or 3.30 times `F10`'s updates, which is beyond update 24,064 in all eight cases (the nearest is `I64` under backpropagation on A, at 25,751). A run that does not reach the threshold is *not decided* in multiply-adds in all eight cases (D times the ratio, over `F10`'s count, is 0.72 to 1.40), although it *rises* in updates. The reading in multiply-adds can therefore only be *falls*, *stays* or *not decided* at the two main settings (required change 5).

**Condition 2: a run that does not reach the threshold may be learning more slowly.** Under the sweep `F10` itself reaches 80 on R at 15,104 of 24,064 recordable updates. A setting slower by a factor above 1.59 on R (1.81 on A) cannot be recorded; under backpropagation the room is 2.54 (2.76). The sentence the brief licenses says the network "learns the first family (reaches 80 on R within the dose) or does not". "Does not" must be "does not within the dose", with the level of R at the end and its change over the last updates stated (required change 8).

**Condition 3: what one seed alone changes.** E42's `F10` on seeds 93683 and 116089 belongs to E42's stage 1 and is not on file at `$W`. For the configurations that are on file for three seeds, the updates to 80 on R are 6,656, 4,864 and 6,400 for `G` under the sweep (largest ratio 1.37) and 8,448, 9,472 and 9,216 for `P8` under the sweep (1.12). The levels of one configuration differ by at most 5 points between seeds. The boundaries of the reading (2/3, 1.5, and 15 points) lie outside these differences, which supports reading one seed as a description. When `F10`'s other two seeds are on file, its three values should be printed beside E49's (required change 7). The brief's own label, "One seed: a description, not a claim", is right and must stay.

## 7. The checks, what is saved, the cost and the predictions

**Check 2 covers one rule** (`python3 -I $C G`). The dry runs of 2,048 updates on E42's worker branch are `e42_F10_S_dry2048_73129`, `e42_G_S`, `e42_G_B`, `e42_P8_S` and `e42_P8-source_S`; there is none of `F10` under backpropagation. The check is attainable under the sweep, unlike E47's check 2: E42's worker found its runner's faster forward bit-equal to the specification's forward for `F10` on the first batch (`git show $W:$M/exp45/e42_qual/E42_STAGE0_73129.md`, check 9), and E49's forward with both masks full makes the same calls. Under backpropagation the runner with full masks is compared with nothing over more than one batch (required change 3).

**No check compares the masked branches with a second form.** When a mask is full the forward takes the dense branch (`lin`), so self-check 1 and check 2 do not execute `masked` at all. Self-check 2 shows that the gradients outside the masks are zero and that the loss is finite. A test is available at no cost: after the start every weight outside a mask is zero, so E42's own forward on the same network computes the same function (required change 4). After the runs the reviewer's reader can be applied to the saved weights and answers, as for E42; it can be applied before the runs if the dry runs of check 6 save them (recommendation b).

**What is saved is sufficient.** Starting and final weights with the two masks allow the reader to recompute the scores and allow the measurement of section 4 to be repeated on the code a limited network learns. Nothing more is asked of the worker for that.

**Cost.** E42's `F10` took 0.87 CPU-hours (sweep) and 0.86 (backpropagation) on seed 73129, scoring included. Four runs are about 3.5 CPU-hours and eight about 7. The specification multiplies full matrices by masks, so the limited settings do not run faster.

**Predictions.** P49.1 to P49.7 can each be scored from the readings. P49.8 does not say how it is scored when neither `in16` nor `out8` reaches 80 on R (recommendation e).

**Nothing else was found to differ from E42's `F10`** in the text or in the specification file: the cell layer, E42's start, the data, the optimizer (AdamW, rate 0.001, decay 0.25 on every tensor but the table), the dose and the scores are E42's. A masked weight starts at zero and has a zero gradient under both rules, so the optimizer's two averages and the decay leave it at zero, which check 3 verifies.

## 8. The eight required changes, as text to apply

1. **What the job tests** (the section now headed "The requirement this tests directly"). Replace the heading by "The requirement this job addresses, and the part of it that is tested". After the second paragraph add: "**The masks limit the number of values a cell computes with, not what a cell can know about the byte.** The byte's code is a function of the eight numbers of the byte's row, spread over all 512 values, and a small part of it identifies the byte. On the 16 values each cell reads (seed 73129's draw), in E42's trained `F10` under either rule and at a start drawn from the constructors' distributions, no two of the 46 bytes the task uses coincide on any cell; the closest two of the 20 bytes the task must tell apart are at 0.05 to 0.24 of their mean distance on the worst cell and at 0.17 to 0.41 on the median cell, against 0.25 to 0.49 on all 512 values; and one linear rule on a cell's 16 values puts every byte in its class on all 256 cells (the reviewer's measurement, `e49_brief_checks.py`, part E). The user's statement is therefore tested in one sense: whether the network learns when each cell is connected to a small part of the input and each readout unit to a small part of the cells. Whether it learns when the part of the input a cell reads does not say which byte is present is not tested." In the sentence the job can license, after "K_OUT of the cells" insert ", a limit on a cell's connections and not on what it can know of the byte,". Review section 9.153's "the direct test of the user's statement" is corrected in the next review section in the same words.

2. **"What it cannot show".** Add: "**A cell that cannot know the byte.** Sixteen values of the byte's code are enough to identify the byte, so no setting of this job makes a cell ignorant of the input. That needs an input code in which a value stands for few bytes, held fixed so that training cannot spread the byte over all values again. With each of the 46 used bytes turning on 11 values of its own, for example, a cell reading 16 values receives a non-zero input for 1 to 10 of the 20 bytes the task must tell apart (median 6) and for 0 to 7 of the ten digits (median 3). It is the next job after this one, with its own reference (the same code with the dense interface), its own start and its own audit." To the item on `l1`, `l2` and `l3` add: "In particular each of the 512 readout units still reads all 512 values of the byte's code through `l2` at every setting."

3. **Check 2, under both rules.** Replace by: "**With both masks full the runner is E42's `F10`, under both rules.** Under the sweep, a dry run of 2,048 updates has the `tensors_sha256` (837dadc1…) and the scores of E42's `e42_F10_S_dry2048_73129`. Under backpropagation no dry run of `F10` is on file: the worker first makes one with E42's runner unchanged (`F10`, backpropagation, 2,048 updates, seed 73129) and commits its receipt; the E49 runner with both masks full must give the same `tensors_sha256` and scores."

4. **A self-check of the masked branches** (specification file and check 1). Add self-check 3: "For each of the four settings, after the start: the logits of `forward(m, layer, iface, x, local)` equal those of `E42.forward(m, layer, x, local)` on the same network under both rules, bit for bit (every weight outside a mask is zero, so the two compute the same function); under backpropagation the gradients of the two masked maps are equal inside the masks and the gradients of every other tensor are equal. A difference is printed with its size." Its output is added to `OUTPUT_E49_SPEC_MAC.txt`.

5. **Multiply-adds to a threshold** ("What is recorded" and "How the result is stated"). Add: "Multiply-adds to a threshold are the setting's multiply-adds a position, times 64 items, times the updates to the threshold; positions are not counted, as in E42's weight-items. Against `F10` the ratio is the ratio of updates times 0.507 (`I64`), 0.454 (`I16`), 0.591 (`in16`) or 0.864 (`out8`). At `I64` and `I16` this reading can only be *falls*, *stays* or *not decided*: *rises* would need more updates than can be recorded, and a run that does not reach the threshold is *not decided* in multiply-adds while it *rises* in updates. The report gives the two readings side by side, and the sentence on learning rests on the reading in updates."

6. **The digest of a draw** (the bullets on the two masks). Add: "The digest of a draw is the first 16 hexadecimal digits of the SHA-256 of `json.dumps(lists)` with Python's default separators, `lists` being the 256 sorted lists of the input mask or the 512 of the readout mask. For seed 73129: K_IN 64 `eb1dd941…`, K_IN 16 `0e09b9ff…`, K_OUT 32 `e9c8783e…`, K_OUT 8 `3d8ba410…`. `in16` and `out8` use `I16`'s lists."

7. **What a seed alone changes** ("How the result is stated"). Add: "E42's `F10` on seeds 93683 and 116089 is part of E42's stage 1. When it is on file, its updates to 80 on A and R and its three levels for the three seeds are printed beside E49's, and a difference from `F10` of seed 73129 that is smaller than the difference between two seeds of `F10` is marked as such. Until then the nearest figure is `G` under the sweep, whose three seeds reach 80 on R at 4,864 to 6,656 updates (a ratio of 1.37)."

8. **A run that does not reach 80 on R** (the sentence the job can license). Replace "or **does not**" by "or **does not within the dose**", and add: "Under the sweep `F10` reaches 80 on R at 15,104 of 24,064 recordable updates, so a setting slower by a factor above 1.59 cannot be told from one that does not learn; under backpropagation the factor is 2.54. For a run that does not reach 80 on R the report gives the mean of R over the last three scored rows and its difference from the mean of the three rows ending 2,048 updates earlier; a run in which that difference is 5 points or more is described as still rising at the end of the dose. Whether such a run is continued is the coordinator's decision."

## 9. Recommendations (none holds the release)

a. **Stage 2 also when `I16` rises.** Start `in16` and `out8` under a rule also when `I16`'s updates to 80 on R are *rises* against `F10`. Otherwise a loss by the brief's own reading has no attribution without a further release.

b. **Dry-run files for the reader.** Let the dry runs of check 6 commit their weights, the two masks and the answers on the length-test panel, as E42's check 8 did. The reviewer's reader can then confirm the masked forward before the dose is spent.

c. **Unread values.** Let the runner print, at the start, the least number of cells that read an input value and of readout units that read a cell, and stop if either is zero. For seed 73129 they are 16 and 45 at `I64` and 1 and 6 at `I16`. At K_IN = 16 the chance that a later seed leaves an input value unread is 0.14.

d. **The start's scale under AdamW.** Add to the item "Settings" of "What it cannot show": "The kept weights start 2.83 or 5.66 times larger than the dense ones, so one update of 0.001 is 1.1% or 0.57% of a weight's mean size, against 3.2%. For comparison, in E42's trained `F10` the mean size of an input weight is 0.083 (sweep) and 0.079 (backpropagation) and of a weight of `lm` 0.079 and 0.066, from 0.031 at the start; `I64` starts at 0.088 and `I16` at 0.177."

e. **P49.8.** State how it is scored when neither `in16` nor `out8` reaches 80 on R, for example by the level of R.

f. **The comparison with the record.** Add beside the quoted measurement: "With the random masks of any of the four settings put on E42's trained `F10` and nothing retrained, 0 to 4 of 300 answers are right. The trained network is not a starting point for a limited one."

g. **The code a limited network learns.** After the runs the reviewer will repeat the measurement of section 4 on the saved starting and final weights of each run. If training under a mask makes a cell's values less informative about the byte, that will be visible there; the report need not compute it.

h. **Readers of a cell.** The readout mask fixes the number of cells a readout unit reads. If a later variant is to fix the number of readout units that read a cell, the lists are drawn for cells and not for readout units. This job should stay as it is.

## 10. Can E49 be released?

**Yes, when the eight changes are applied. A re-audit is not needed if they are applied as written.** They add text, one self-check and one dry run under backpropagation; they change neither the network, nor the settings, nor the runs. The reviewer will read the revision when the worker's stage 0 is on the remote and will apply the reader to the saved weights. If the coordinator prefers to add the fixed-code job to E49 as an arm, in place of naming it as the next job, that arm needs its own specification file and an audit.

## 11. What I did not check

- **The specification file was not executed**, because torch is not installed. Its statements that the full-mask network is bit-equal to E42's `F10` in logits and gradients, that gradients outside the masks are zero, and its first-batch losses (5.561, 5.570, 5.574, 5.569) rest on the committed Mac output. The losses cannot be recomputed here, because the starting weights come from torch's generator.
- **The sweep through the masked maps** was read in `MaskedSign` and not computed.
- **The start** was measured on stand-in weights drawn from the constructors' distributions, not on torch's values for seed 73129.
- **The code of a network trained under E49's masks** does not exist yet. Section 4 measures E42's trained `F10` and stand-in starts; recommendation g covers the rest.
- **Whether the E49 runner and E42's runner stay bit-equal over 2,048 updates** is for check 2 to show; the E49 runner does not exist yet.
- **`F10` on seeds 93683 and 116089** is not on file at `$W`.
- **The division of the 46 used bytes into 20 and 26** is the reviewer's reading of the task: the letters are drawn at random between the operands and do not enter the answer.
