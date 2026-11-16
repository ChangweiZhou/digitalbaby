# E48: a new family learned only in connections gated by its own code (16 November 2026, revision 1)

Evidence level E2. The requirements are the outside reviewer's (its memo, revision 19, section 27, on `claude/vibrant-cerf-sje726` at 4f8aed6f). **The memo calls this job E47; that number was already taken by the locality job, so it is E48 here.** The specification, the specification file and the predictions are mine. **Not released: it waits for the reviewer's audit.** One worker, best the E40 worker.

Shell names: `D=origin/codex/minifly-consolidated-20261023; M=MOD97_LOCAL_RULE_20261106; K=PROJECT_DOCUMENTS/COORD_CHECKS_20261116`. Host: Linux x86-64, Python 3.13.18, torch 2.15.0+cpu; at most three processes, one thread each; launch outside any wrapper with a time limit. Edit nothing existing.

## Why

The adopted question is whether a learner whose changes are confined to the connections in use can keep what it has learned while learning new things. Three measurements say where the present learner fails at it (the reviewer's, memo sections 23 and 24.8; I have not rerun them):

- what 64 updates learn of a new family is a change of low rank in the recurrent layer;
- the loss of the earlier family is in the same matrices;
- under unrelated learning the earlier family's function is overwritten throughout the network within 64 updates, most of all in the recurrent matrix, with plain gradient steps as with AdamW; putting the readout back restores nothing.

E48 puts the question in its simplest form. The change for a new family is made only in connections that take part when that family's code is present. The earlier families are then untouched **by construction**, and what is measured is the price in learning the new one.

**This is an upper bound in the sense of E35's first stage.** The gate is supplied by the harness from the item's code byte. Whether a learner can produce the gate itself, or decide when a new term is needed, is a later question, and every statement from this job says so.

## The comparison with the record (working rule 1)

- **Setting and reference: E40's own runs.** From the networks that lived under the sweep and under backpropagation, a new family (mod 83, codes `e f g h`) from a tenth of its pairs. With every weight free, lived networks pass 90% on taught pairs by 64 updates in eight of nine runs and are at 77 to 95% on held-out reduction pairs at 8,192; their best reduction scores come as early as 64 to 128 updates. Row 256 of each E40 run is on file and is the row read here.
- **Rank.** The reviewer measured the change that 64 updates make as low rank; the ranks 4, 16 and 64 bracket it.
- **Dose.** 1,024 updates, four times the row that is read, so that a term which learns more slowly than the free network is seen arriving.
- **Starting level.** Row 0 is E40's row 0 exactly (the terms start at zero), so the reading compares levels at the same row from the same start; no starting level has to be subtracted.

## What may change, as code

**`$K/e48_gated_spec.py` on the default branch is the specification** (self-checks without training; my output is `$K/OUTPUT_E48_SPEC_MAC.txt`). Copy it to `$M/e48_gated.py` and import it; do not rewrite it. In words:

- **Frozen:** every parameter of the lived network.
- **Free:** the four code rows of the byte table (bytes 101 to 104); and for each of the two matrices of the recurrent layer one added term of rank r, the product `up · down`: on the input map (768 × 512) and on the recurrent matrix (768 × 256). One pair of terms for the family, shared by its four operations. `up` starts at zero, so the network is unchanged at the start; `down` is drawn uniformly in ±1/√(inputs) from `T.Generator().manual_seed(seed * 1000 + r)`. No bias.
- **The gate** is one flag for an item: on for items of the new family, off for every other. In training every item is of the new family. In scoring, the new family's panels are answered with the gate on and every earlier family's with the gate off. With the gate off the forward is `arch26.forward`, value for value.
- **The two rules.** Each term is two linear maps without bias, each crossed back by the rule of the run: exactly under backpropagation, by weight signs under the sweep (`exp3_state.lin`). With `up` at zero no error reaches `down` at the first update under either rule; it does from the second (self-check).
- **The byte table's gradient** is kept for the four code rows only (`keep_code_rows`, called after `backward`).
- **Optimizer:** `make_opt` in the file: AdamW at rate 0.001, betas (0.9, 0.98); the terms decay at the standing 0.25, the byte table does not.

**Numbers that may change:** 2,304 · r for the terms and 32 for the code rows: 9,248, 36,896 and 147,488 at ranks 4, 16 and 64, against the network's 1,124,113.

**What I checked without training** (E26's lives of seed 73129, both rules, three ranks): with the gate off the logits are bit-equal to `arch26.forward`'s; with the gate on at the start they are too; after an update the only tensor of the network that has changed is the byte table, in rows 101 to 104; and after three gated updates an earlier family's logits with the gate off are bit-equal to the lived network's.

## Runs

The six networks that lived under the sweep and under backpropagation (`exp45/e26/e26_life_<S|B>_<seed>.pt`), each under the rule it lived under, at ranks 4, 16 and 64: 18 runs of 1,024 updates. The Kolen–Pollack lives are not used.

- **Stream, panels, scores:** E40's, from `e40_new_family.py` unchanged (`batches(seed)`, its held-out, taught, plain and earlier-family scores).
- **Rows:** 0, 16, 32, 64, 128, 256, 512 and 1,024.
- **What wrong answers are** (working rule 3): at rows 0, 64, 256 and 1,024 the three bytes generated for every item of the reduction score are written beside the receipt.
- **Committed:** the terms and the four code rows of every run at rows 256 and 1,024, in one small file each (`git add -f`, listed in the report from `git ls-tree`). The lived network itself is not saved again.

About 1.5 CPU-hours with scoring (the reviewer's estimate).

## Checks before the runs

1. **The specification file.** `$M/e48_gated.py` is byte-identical to `$K/e48_gated_spec.py`; run on `$K`, its self-checks print what `$K/OUTPUT_E48_SPEC_MAC.txt` has (the booleans and row lists exactly; the gradients and losses to two digits).
2. **Row 0 equals E40's row 0** on every score, for each of the six networks (`exp45/e40/e40_lived_<S|B>_<seed>.json`).
3. **Keeping, by construction, checked at every row.** With the gate off, every earlier family's answers are the same item for item as at row 0, and the `tensors_sha256` of the lived network's tensors with the four code rows set back to their start equals the life-end file's.
4. **The batch digest** at every row is E40's for the seed.
5. **Under the sweep the terms learn:** at update 2 the gradient of both `down` matrices is not zero.
6. One dry run of 64 updates at rank 16 under each rule, with CPU time and a forecast.

## Reading (fixed beforehand; the reviewer's)

For each rule and rank, the new family's **reduction score** (E40's: held-out add and sub pairs that need the reduction, no letters) **at row 256 against E40's run of the same seed and rule at row 256**:

| Result, in two of three seeds | Reading |
|---|---|
| Within 10 points of E40's, or above it | **No price:** at this rank a change confined to gated connections learns the new family as fast as a change to every weight, on this setting, with the gate supplied. |
| 30 points or more below | **A price.** |
| Otherwise, or the seeds disagree | **Some price**, with the three differences given. |

Tabled beside it and not read: the same at rows 64 and 1,024; the taught panel; the numbers changed, against the network's 1,124,113 and against the number of items a replay store of that size would hold (at 19 bytes an item and 4 bytes a number: about 1,950, 7,770 and 31,050 items at the three ranks).

**What is not a result:** that the earlier families are kept. It is true by construction and is a check.

## What it cannot show

- **That a learner can find the gate,** or decide when a new term is needed. The harness supplies both.
- **Fixed memory.** The terms grow by one a family. The fair comparison is with a replay store of equal size, and it belongs to the long-life world.
- **Unrelated tasks,** whose answers need output units of their own; the readout is frozen here.
- **Composition across families,** which a gate that is on for one family at a time does not allow.
- **A second new family** on top of the first: whether terms of different families interfere when more than one gate could be on.

## Predictions (mine)

| | Prediction | Chance |
|---|---|---|
| P48.1 | Checks 2 and 3 pass for all six networks | 90% |
| P48.2 | Rank 16, backpropagation: *no price* | 50% |
| P48.3 | Rank 16, sweep: *no price* | 35% |
| P48.4 | Rank 64, backpropagation: *no price* | 65% |
| P48.5 | Rank 4: *some price* or *a price* under both rules | 65% |
| P48.6 | Under the sweep the reduction score at row 256 is lower than under backpropagation at the same rank, by 10 points or more, at two of the three ranks (two of three seeds each) | 50% |
| P48.7 | At row 1,024 rank 16 is within 10 points of E40's run at row 1,024, two of three seeds, under each rule | 55% |

## Done

The runner, the check receipts, the 18 runs with receipts, the committed term files, and `E48_REPORT.md` with the curves (run × row: E40's scores), the reading for each rule and rank, the tabled figures, and the outcome of each prediction. No readings beyond the table above.
