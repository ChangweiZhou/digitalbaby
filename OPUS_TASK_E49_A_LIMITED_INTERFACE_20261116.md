# E49: a limited interface — each cell reads a few of the input's values and each readout unit a few cells (16 November 2026, revision 1)

Evidence level E2; one seed, a first measurement. The job is the one the audit of E47 names as next (its recommendation 1), and the user asked for it to be drafted on 16 November. The specification, the specification file and the predictions are mine. **Not released: it waits for the reviewer's audit.** One worker, best an E42 worker; it can run beside E47's trial.

Shell names: `D=origin/codex/minifly-consolidated-20261023; M=MOD97_LOCAL_RULE_20261106; K=PROJECT_DOCUMENTS/COORD_CHECKS_20261116`. Host: Linux x86-64, Python 3.13.18, torch 2.15.0+cpu; at most three processes, one thread each; launch outside any wrapper with a time limit. Edit nothing existing.

## The requirement this tests directly

The user, 16 November: "biological cells obviously cannot see the whole input, so locality is essential and our current set up does not scale."

E42 limited what a cell reads of other cells. E47 confines reading the byte and being read to a boundary of 256 cells. Inside that boundary nothing is limited yet: each of the 256 cells reads all 512 values of the byte's code and each of the 512 readout units reads all 256 cells, 524,288 of a position's multiply-adds. E49 limits both, at 256 cells, and asks whether the network still learns.

## The comparison with the record (working rule 1)

- **The reference is E42's `F10`**: 256 cells, 10 fixed local connections, the dense interface, on E26's birth data and dose. E49 with both masks full is `F10` exactly (self-check 1), so E42's runs of `F10` are the reference and are not run again. On E42's first seed `F10` learns the family more slowly than the full network (the reviewer's read; E42 is not yet merged and I have not read its report).
- **A trained network rests on the dense interface.** With the smaller half of each cell's input weights removed from E42's trained `F10` and nothing retrained, its score at 12 letters falls from 92 to 83 under the sweep and from 91 to 72 under backpropagation (the reviewer's measurement). That says a dense-trained network uses its whole input; it does not say whether a network trained with a limited input learns.
- **Nothing on file limits the input or the readout of this learner.** The nearest record is the random sparse expansion of October (the mushroom-body store), which memorised; it had no learned recurrence.
- **Coverage, from the draws** (self-check 2, seed 73129): at 16 input values a cell, every input value is read by 1 to 18 cells; at 8 cells a readout unit, every cell is read by 6 to 27 readout units. No input value and no cell is left unread at any setting. One input value read by a single cell is a thin path; the settings at 64 and 32 have 16 to 52 and 45 to 84.
- **The dose is E26's birth, 24,576 updates**, which `F10` needs most of on its first seed.

## The network, as code

**`$K/e49_interface_spec.py` on the default branch is the specification** (self-checks without training; my output is `$K/OUTPUT_E49_SPEC_MAC.txt`). Copy it to `$M/e49_interface.py`, with its import of E42's file pointed at the runner's copy, and otherwise do not rewrite it. In words:

- **The cell layer is E42's `F10`:** 256 cells on E42's grid, each reading 10 cells of its neighbourhood and itself, fixed, every cell at period 1; E42's draws and start.
- **Input mask.** Cell i reads K_IN of the 512 values of the byte's code (the output of `l1`), the same ones for its three gates: `sorted(Random(f"e49|input|{seed}|{K_IN}").sample(range(512), K_IN))`, drawn for cells 0 to 255 in turn from one generator.
- **Readout mask.** Readout unit j (of the 512 of `lm`) reads K_OUT of the 256 cells: `sorted(Random(f"e49|readout|{seed}|{K_OUT}").sample(range(256), K_OUT))`, for units 0 to 511 in turn from one generator.
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

**Stage 1:** `I64` and `I16` under both rules, four runs of 24,576 updates. **Stage 2, without a further release:** if `I16` under a rule does not reach 80 on R within the dose, or is *lower* than `F10` on a level (below), `in16` and `out8` are run under that rule, to say which side costs.

**Runner:** E42's with this forward; E42's data, optimizer (the standing decay, masked weights staying at zero since they have no gradient and start at zero), dose and scores, every 256 updates. About 0.8 CPU-hours a run on E42's measured rate for `F10`; four to eight runs.

## Checks before the runs

1. **The specification file.** The runner's copy differs from `$K/e49_interface_spec.py` in its import line only; run on `$K`, the self-checks print what `$K/OUTPUT_E49_SPEC_MAC.txt` has: with both masks full, logits and gradients bit-equal to E42's `F10` under both rules; for each setting the counts, the coverage, gradients zero outside the masks, and the digests of the draws.
2. **With both masks full the runner is E42's `F10`:** a dry run of 2,048 updates under the sweep has the `tensors_sha256` and scores of E42's `e42_F10_S_dry2048_73129`.
3. **The masks hold through a run:** at every scored row every weight outside a mask is exactly zero, and the optimizer's two running averages there are zero.
4. **Same data:** the batch digest at row 4,096 is E26's for the seed.
5. **Scores:** E42's check 12 on E26's six births, with full masks and every cell connected.
6. One dry run of 256 updates per setting and rule, with CPU time and a forecast.

## What is recorded

Per run: E42's scores every 256 updates (A, R, the taught panel, the length test at 12 and 40 letters); updates and multiply-adds to the 50, 80 and 90 thresholds on A and R, three-row version; at the first batch and at the end, the cosine between the sweep's and the exact gradient of each masked map; the mean magnitude of the kept weights of each map at every scored row. At the last row, the bytes of each answer given on the length-test panel (working rule 3). Starting and final weights with the two masks, on the remote (`git add -f`, listed from `git ls-tree`).

## How the result is stated (fixed beforehand)

Against E42's `F10` of seed 73129 under the same rule:

- **Updates to the 80 threshold** on A and on R: E41's table of four cases, read as fewer, more or about the same. The same in multiply-adds.
- **The level:** the mean of A, of R and of the length test at 40 letters over the last three scored rows; *lower* or *higher* at 15 points or more, otherwise *about the same*.

**The sentence this job can license**, for this seed and each rule: with each cell reading K_IN of the input's 512 values and each readout unit K_OUT of the cells, the network **learns the first family** (reaches 80 on R within the dose) or **does not**; and whether it does so in about the same updates and to about the same level as with the dense interface. One seed: a description, not a claim. The other two seeds follow for the smallest setting that learns, by my release.

## What it cannot show

- **More than one seed**, until released.
- **An input with geometry.** The 512 values of a byte's code have no positions; a random subset is not a neighbourhood. A future input with many tokens, which the user's statement looks to, would have one.
- **The byte's code and the readout's own layers** (`l1`, `l2`, `l3`), which stay dense and are now most of the network's computation. Their turn is after this.
- **More cells.** The job is at 256 cells; E47 varies the number.
- **Settings.** Learning rate and decay are E42's; the scale of the kept weights at the start is a choice.

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
| P49.8 | If stage 2 runs under backpropagation, `in16` is the worse of the two (later to 80 on R, or not reaching it where `out8` does) | 60% |

## Done

The runner, the check receipts, the four to eight runs with receipts and weights on the remote, and `E49_REPORT.md` with the tables (setting × rule: updates and multiply-adds to the thresholds in both versions, the three levels, the cosines), the descriptions against `F10`, the sentence per rule, and the outcome of each prediction. No readings beyond the forms above.
