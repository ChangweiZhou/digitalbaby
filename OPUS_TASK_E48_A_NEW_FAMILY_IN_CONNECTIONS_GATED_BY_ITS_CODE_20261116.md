# E48: a new family learned only in connections gated by its own code (16 November 2026, revision 2)

Evidence level E2. The requirements are the outside reviewer's (its memo, revision 19, section 27, on `claude/vibrant-cerf-sje726` at 4f8aed6f). **The memo calls this job E47; that number was already taken by the locality job, so it is E48 here.** The specification, the specification file and the predictions are mine. Revision 1 was audited by the reviewer (`AUDIT_E48_GATED_CONNECTIONS_BRIEF_20261116.md` on `claude/vibrant-cerf-sje726` at 68190ef5), which required six changes and stated that after them no re-audit is needed. Revision 2 makes them and the recommendations; the last section lists them. **Released.** One worker, best the E40 worker.

Shell names: `D=origin/codex/minifly-consolidated-20261023; M=MOD97_LOCAL_RULE_20261106; K=PROJECT_DOCUMENTS/COORD_CHECKS_20261116`. Host: Linux x86-64, Python 3.13.18, torch 2.15.0+cpu; at most three processes, one thread each; launch outside any wrapper with a time limit. Edit nothing existing.

## Why

The adopted question is whether a learner whose changes are confined to the connections in use can keep what it has learned while learning new things. Three measurements say where the present learner fails at it (the reviewer's, memo sections 23 and 24.8; I have not rerun them):

- what 64 updates learn of a new family is a change of low rank in the recurrent layer;
- the loss of the earlier family is in the same matrices;
- under unrelated learning the earlier family's function is overwritten throughout the network within 64 updates, most of all in the recurrent matrix, with plain gradient steps as with AdamW; putting the readout back restores nothing.

E48 puts the question in its simplest form. The change for a new family is made only in connections that take part when that family's code is present. The earlier families are then untouched **by construction**, and what is measured is the price in learning the new one.

**This is an upper bound in the sense of E35's first stage.** The gate is supplied by the harness from the item's code byte. Whether a learner can produce the gate itself, or decide when a new term is needed, is a later question, and every statement from this job says so.

## The comparison with the record (working rule 1)

- **Setting and reference: E40's own runs.** From the networks that lived under the sweep and under backpropagation, a new family (mod 83, codes `e f g h`) from a tenth of its pairs. With every weight free, lived networks pass 90% on taught pairs by 64 updates in eight of nine runs and are at 77 to 95% on held-out reduction pairs at 8,192; their best reduction scores come as early as 64 to 128 updates. Rows 128, 256 and 512 of each E40 run are on file, and their mean is read here: a single row of E40 moves by up to 19 points between neighbouring rows (lived B 73129: 87.2, 68.1 and 80.4 at rows 128, 256 and 512), which is more than the 10-point mark.
- **Rank.** The reviewer measured the change that 64 updates make as low rank; the ranks 4, 16 and 64 bracket it.
- **Dose.** 1,024 updates, twice the last row that is read, so that a term which learns more slowly than the free network is seen arriving.
- **Starting level.** Row 0 is E40's row 0 exactly (the terms start at zero), so the reading compares levels at the same row from the same start; no starting level has to be subtracted.

## What may change, as code

**`$K/e48_gated_spec.py` on the default branch is the specification** (self-checks without training; my output is `$K/OUTPUT_E48_SPEC_MAC.txt`). Copy it to `$M/e48_gated.py` and import it; do not rewrite it. In words:

- **Frozen:** every parameter of the lived network.
- **Free:** the four code rows of the byte table (bytes 101 to 104); and for each of the two matrices of the recurrent layer one added term of rank r, the product `up · down`: on the input map (768 × 512) and on the recurrent matrix (768 × 256). One pair of terms for the family, shared by its four operations. `up` starts at zero, so the network is unchanged at the start; `down` is drawn uniformly in ±1/√(inputs) from `T.Generator().manual_seed(seed * 1000 + r)`. No bias.
- **The gate** is one flag for an item: on for items of the new family, off for every other. In training every item is of the new family. In scoring, the new family's panels are answered with the gate on and every earlier family's with the gate off. With the gate off the forward is `arch26.forward`, value for value.
- **How the gate is read.** The runner (`exp48_gated_run.py`; `$M/e48_gated.py` is the copy of the specification file) reads the gate from the item: it is on when the byte after the start symbol, `x[:, 1]`, is one of 101 to 104. It is computed for every row of a call, with an assertion that all rows of the call agree, and it holds for every position of the item, the start symbol's included. After `arch26.install()` the runner sets `exp11_cell.forward` to a function with the harness's signature `(m, x, local, through_time, keep=False, silence=False)` that returns `e48_gated.forward(m, term, x, local, gate, silence)`. E40's `Scorer` and `exp11_cell.respond` then run unchanged; each call of the scorer holds the items of one task. The update is the text of `exp26_candidate.step` with this forward in place of `arch26.forward` and with `keep_code_rows(m)` between `loss.backward()` and `opt.step()`; `freeze(m)` is called once, before the optimizer is made.
- **The two rules.** Each term is two linear maps without bias, each crossed back by the rule of the run: exactly under backpropagation, by weight signs under the sweep (`exp3_state.lin`). With `up` at zero no error reaches `down` at the first update under either rule; it does from the second (self-check).
- **The byte table's gradient** is kept for the four code rows only (`keep_code_rows`, called after `backward`).
- **Optimizer:** `make_opt` in the file: AdamW at rate 0.001, betas (0.9, 0.98); the terms decay at the standing 0.25, the byte table does not. The settings are E40's. What the decay acts on is not. In E40 every weight of the lived network decays (where no gradient opposes it, by a factor 0.938 over 256 updates and 0.774 over 1,024). Here the lived weights are out of the optimizer and do not decay, and the decay acts on the two factors of the term alone. E48 and E40 therefore differ in two things, where the change is made and whether the lived weights decay, and the report says so.

**Numbers that may change:** 2,304 · r for the terms and 32 for the code rows: 9,248, 36,896 and 147,488 at ranks 4, 16 and 64, against the network's 1,124,113.

**What I checked without training** (E26's lives of seed 73129, both rules, three ranks): with the gate off the logits are bit-equal to `arch26.forward`'s; with the gate on at the start they are too; after an update the only tensor of the network that has changed is the byte table, in rows 101 to 104; and after three gated updates an earlier family's logits with the gate off are bit-equal to the lived network's.

## Runs

The six networks that lived under the sweep and under backpropagation (`exp45/e26/e26_life_<S|B>_<seed>.pt`), each under the rule it lived under, at ranks 4, 16 and 64: 18 runs of 1,024 updates. The Kolen–Pollack lives are not used.

- **Stream, panels, scores:** E40's, from `e40_new_family.py` unchanged (`batches(seed)`, its held-out, taught, plain and earlier-family scores).
- **Rows:** 0, 16, 32, 64, 128, 256, 512 and 1,024.
- **What wrong answers are** (working rule 3): at rows 0, 64, 128, 256, 512 and 1,024 the three bytes generated for every item of the reduction score are written beside the receipt.
- **Committed:** for every run at rows 128, 256, 512 and 1,024, one file `exp45/e48/e48_<rule>_<seed>_r<rank>_row<row>.pt`, written with `torch.save` of a dict of five plain float32 tensors (`.detach().clone()`, not parameters): `up_i` (768, r), `down_i` (r, 512), `up_h` (768, r), `down_h` (r, 256) and `code_rows` (4, 8: rows 101 to 104 of the byte table, in that order). A file is about 37 KB, 148 KB and 590 KB at ranks 4, 16 and 64; the 72 files are 18.6 MB. Rows 16, 32 and 64 at ranks 4 and 16 are committed as well (3.3 MB), so that the arrival of the term can be re-read later. Each file's SHA-256 is in the receipt; the files are added with `git add -f` and listed in the report from `git ls-tree`. The lived network itself is not saved again.
- **Two columns that cost no run or almost none** (the audit's recommendations): at rows 256 and 1,024 the new family is scored once more with the gate forced off, which shows what the four code rows carry alone; and at every scored row under the sweep, for each of the four factors, the cosine between the factor and what the sweep uses in its place (the signs times the mean size).

About 1.5 CPU-hours with scoring (the reviewer's estimate).

## Checks before the runs

1. **The specification file.** `$M/e48_gated.py` is byte-identical to `$K/e48_gated_spec.py`; run on `$K`, its self-checks print what `$K/OUTPUT_E48_SPEC_MAC.txt` has (the booleans and row lists exactly; the gradients and losses to two digits).
2. **Row 0 equals E40's row 0** on every score, for each of the six networks (`exp45/e40/e40_lived_<S|B>_<seed>.json`).
3. **Keeping, by construction, checked at every row.** At every row the bytes generated for every item of the six earlier panels are written as the SHA-256 of their JSON; the digest at every row equals row 0's, and the six counts at every row equal E40's row 0. And the `tensors_sha256` of the lived network's tensors with the four code rows set back to their start equals the life-end file's.
3a. **The term is in use where it should be and nowhere else.** In the dry run, at row 64, through the installed forward: the logits on a batch of an earlier family are bit-equal to `arch26.forward`'s on the same network, and the logits on a batch of the new family are not.
4. **The batch digest** at rows 64, 128, 256, 512 and 1,024 is E40's for the seed; at rows 16 and 32, which E40 does not have, it is recorded.
5. **The path through a term is connected:** at update 2 the gradient of both `down` matrices is not zero, under each rule. This tests only that the path is connected, not how close the sweep's path is to the exact one; the cosines above describe that.
6. One dry run of 64 updates at rank 16 under each rule, with CPU time and a forecast.

## Reading (fixed beforehand; the reviewer's)

For each run, let R be the mean of the new family's **reduction score** (E40's: held-out add and sub pairs that need the reduction, no letters) at rows 128, 256 and 512. For each rule and rank, R of the E48 run is compared with R of E40's run of the same seed and rule. E40's values of R are 82.65, 78.93 and 81.45 under the sweep and 78.58, 91.86 and 91.11 under backpropagation (seeds 73129, 93683 and 116089).

- **No price:** in at least two of the three seeds R is within 10 points of E40's, or above it. At this rank a change confined to gated connections learns the new family as fast as a change to every weight, on this setting, with the gate supplied.
- **A price:** in at least two of the three seeds R is 30 points or more below E40's.
- **Some price:** every other case; the three differences are given.

The single rows 64, 256 and 1,024 are tabled beside R and are not read.

Tabled beside it and not read: the taught panel; the numbers changed, against the network's 1,124,113; and the comparison with a replay store (the term at 4 bytes a number is 36,992, 147,584 and 589,952 bytes. An entry of the harness's stream is a task and two operands, 3 bytes; a store of the term's size holds 12,330, 49,194 and 196,650 entries. The new family's whole taught set is 2,756 entries, 8,268 bytes. The taught single-operation pairs of the three earlier families are 30,576 entries, 91,728 bytes. As rendered prompts of 19 bytes, the longest, without the answer, the figures are 1,946, 7,767 and 31,050.)

**The other side of the comparison, at no run:** E40's own earlier-family scores at rows 0, 128, 256 and 512 beside E48's, which are row 0's by construction. In E40's lived sweep run of seed 73129, between row 0 and row 256, addition mod 97 goes from 380 to 109 of 400 and subtraction mod 73 from 302 to 132. That is what E40 paid for its learning.

**What is not a result:** that the earlier families are kept. It is true by construction and is a check.

## What it cannot show

- **That a learner can find the gate,** or decide when a new term is needed. The harness supplies both.
- **Fixed memory.** The terms grow by one a family. The fair comparison is with a replay store of equal size, and it belongs to the long-life world. At every rank the term takes more memory than the new family's whole taught set stored as entries (4.5, 17.9 and 71.4 times), and from rank 16 more than every taught single-operation pair of the life (1.6 times at rank 16).
- **That a term is kept under the standing optimizer.** A term is kept exactly only while it is out of the optimizer, as it is here after the last update. Left in an optimizer with decay 0.25 and without items of its family, each factor shrinks by 0.99975 an update and their product by 0.9995: to 0.36 of its size after 2,048 updates and to 0.02 after 8,192.
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
| P48.6 | The difference from E40 (E48's R minus E40's R) is lower under the sweep than under backpropagation by 10 points or more, at two of the three ranks (two of three seeds each). Restated on differences after the audit: in E40 itself R under backpropagation exceeds R under the sweep by −4.1, +12.9 and +9.7. | 40% |
| P48.7 | The mean of rows 512 and 1,024 at rank 16 is within 10 points of the same mean of E40's run (sweep 83.09, 83.70, 84.05; backpropagation 83.27, 91.31, 90.72), two of three seeds, under each rule | 55% |

## Done

The runner, the check receipts, the 18 runs with receipts, the committed term files, and `E48_REPORT.md` with the curves (run × row: E40's scores), the reading for each rule and rank, the tabled figures, and the outcome of each prediction. No readings beyond the table above.

## What changed from revision 1 (the audit)

Made as the audit wrote them.

| | Change | Required? |
|---|---|---|
| 1 | The reading is on R, the mean of the reduction score at rows 128, 256 and 512, against the same mean of E40's run: single rows of E40 move by up to 19 points, more than the 10-point mark. This corrects the reviewer's own requirement, which I had carried without looking at E40's rows. | yes |
| 2 | How the gate reaches E40's scorer and the update, with check 3a | yes |
| 3 | Digests of the earlier panels' answers at every row; the batch digest where E40 has one; the rows at which answers are written | yes |
| 4 | The term files: what they hold, their names and sizes | yes |
| 5 | The basis of the replay comparison. My figure counted rendered prompts; as stream entries the term costs 4.5, 17.9 and 71.4 times the family's taught set. | yes |
| 6 | What decay acts on here and in E40, and that a term left in the standing optimizer would itself decay away | yes |
| — | E40's own losses on earlier families tabled beside; the gate forced off as a column; the sweep's cosines; P48.6 on differences; the runner's name; early rows of the term committed | recommended |
