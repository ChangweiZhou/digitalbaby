# Audit of the E48 brief: a new family learned only in connections gated by its own code (revision 1, default branch at 7da5c453)

16 November 2026. Outside reviewer. For the coordinator.

## 1. What was audited, and how

Three files on `origin/codex/minifly-consolidated-20261023` at its tip `7da5c453` (the brief entered at `bd9f18fe`):

| File | Lines | Blob |
|---|---|---|
| `OPUS_TASK_E48_A_NEW_FAMILY_IN_CONNECTIONS_GATED_BY_ITS_CODE_20261116.md` | 97 | `60ce7cdb` |
| `PROJECT_DOCUMENTS/COORD_CHECKS_20261116/e48_gated_spec.py` | 121 | `650f9429` |
| `PROJECT_DOCUMENTS/COORD_CHECKS_20261116/OUTPUT_E48_SPEC_MAC.txt` | 7 | `ab8c5ba1` |

The brief implements section 27 of my memo (requirements in 27.2). It lists no points for the audit, so the audit follows the seven questions of section 5 below.

Every file was read with `git show`. No code of another branch was run, torch was not used, and nothing was trained. Two files of mine stand beside this one:

- `e48_gated_reader.py`: the gated forward written in numpy from the text of the brief, and the generation of answers as the harness does it.
- `e48_brief_checks.py`: six parts, A to F; its output is `OUTPUT_E48_BRIEF_CHECKS.txt`. It is run from the repository root with `python3 -I PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/e48_brief_checks.py [A B C D E F]`. Each finding below names the part that shows it.

Terms used below. *The term* is the added product `up · down` of rank r, one on the input map and one on the recurrent matrix. *The gate* is the flag that decides whether the term is added. *The sweep* is the standing rule that sends errors back through the signs of the weights. *The reduction score* is E40's: whole answers on the held-out addition and subtraction pairs that need the reduction at 83, without letters, the mean of the two operations.

## 2. Verdict

The design is the one section 27.2 asked for, item by item, and the specification file does what the brief says. One part of the brief is not sound, and it is my own requirement: the reading compares single rows, and a single row of E40 moves by up to 19 points between neighbouring rows, which is more than the 10-point mark. One part is not stated: how the gate reaches E40's scorer, which the brief says is used unchanged and which has no gate. Four further changes are small.

**The brief can be released after the six required changes of section 3. If they are applied as written, a second audit of the brief is not needed.** The runner does not exist yet; the audit of the results will reproduce the rows of the reading from the committed term files with `e48_gated_reader.py`.

## 3. Required changes (text to apply)

**1. The reading (a correction to my requirement).** In "Reading", replace the sentence before the table and the table by:

> For each run, let R be the mean of the new family's reduction score at rows 128, 256 and 512. For each rule and rank, R of the E48 run is compared with R of E40's run of the same seed and rule. E40's values of R are 82.65, 78.93 and 81.45 under the sweep and 78.58, 91.86 and 91.11 under backpropagation (seeds 73129, 93683 and 116089).
>
> - **No price:** in at least two of the three seeds R is within 10 points of E40's, or above it.
> - **A price:** in at least two of the three seeds R is 30 points or more below E40's.
> - **Some price:** every other case; the three differences are given.
>
> The single rows 64, 256 and 1,024 are tabled beside R and are not read.

In "The comparison with the record", replace "Row 256 of each E40 run is on file and is the row read here" by "Rows 128, 256 and 512 of each E40 run are on file, and their mean is read here: a single row of E40 moves by up to 19 points between neighbouring rows (lived B 73129: 87.2, 68.1 and 80.4 at rows 128, 256 and 512), which is more than the 10-point mark". In "Dose", replace "four times the row that is read" by "twice the last row that is read". In P48.6, replace "the reduction score at row 256" by "R". In P48.7, replace "At row 1,024 rank 16 is within 10 points of E40's run at row 1,024" by "The mean of rows 512 and 1,024 at rank 16 is within 10 points of the same mean of E40's run (sweep 83.09, 83.70, 84.05; backpropagation 83.27, 91.31, 90.72)". P48.2 to P48.5 name readings and keep their wording; their chances may be restated.

**2. How the gate reaches the forward.** After the bullet "The gate", add:

> **How the gate is read.** The runner reads the gate from the item: it is on when the byte after the start symbol, `x[:, 1]`, is one of 101 to 104. It is computed for every row of a call, with an assertion that all rows of the call agree, and it holds for every position of the item, the start symbol's included. After `arch26.install()` the runner sets `exp11_cell.forward` to a function with the harness's signature `(m, x, local, through_time, keep=False, silence=False)` that returns `e48_gated.forward(m, term, x, local, gate, silence)`. E40's `Scorer` and `exp11_cell.respond` then run unchanged; each call of the scorer holds the items of one task. The update is the text of `exp26_candidate.step` with this forward in place of `arch26.forward` and with `keep_code_rows(m)` between `loss.backward()` and `opt.step()`; `freeze(m)` is called once, before the optimizer is made.

Add to "Checks before the runs", after check 3:

> 3a. **The term is in use where it should be and nowhere else.** In the dry run, at row 64, through the installed forward: the logits on a batch of an earlier family are bit-equal to `arch26.forward`'s on the same network, and the logits on a batch of the new family are not.

**3. The checks and what is written.**

- Check 3, replace the first clause by: "At every row the bytes generated for every item of the six earlier panels are written as the SHA-256 of their JSON; the digest at every row equals row 0's, and the six counts at every row equal E40's row 0."
- Check 4, replace by: "The batch digest at rows 64, 128, 256, 512 and 1,024 is E40's for the seed; at rows 16 and 32, which E40 does not have, it is recorded."
- "What wrong answers are": replace "at rows 0, 64, 256 and 1,024" by "at rows 0, 64, 128, 256, 512 and 1,024".

**4. The term files.** Replace the bullet "Committed" by:

> **Committed:** for every run at rows 128, 256, 512 and 1,024, one file `exp45/e48/e48_<rule>_<seed>_r<rank>_row<row>.pt`, written with `torch.save` of a dict of five plain float32 tensors (`.detach().clone()`, not parameters): `up_i` (768, r), `down_i` (r, 512), `up_h` (768, r), `down_h` (r, 256) and `code_rows` (4, 8: rows 101 to 104 of the byte table, in that order). A file is about 37 KB, 148 KB and 590 KB at ranks 4, 16 and 64; the 72 files are 18.6 MB. Each file's SHA-256 is in the receipt; the files are added with `git add -f` and listed in the report from `git ls-tree`. The lived network itself is not saved again.

**5. The comparison with a replay store.** In "Tabled beside it", replace the parenthesis "(at 19 bytes an item and 4 bytes a number: about 1,950, 7,770 and 31,050 items at the three ranks)" by:

> (the term at 4 bytes a number is 36,992, 147,584 and 589,952 bytes. An entry of the harness's stream is a task and two operands, 3 bytes; a store of the term's size holds 12,330, 49,194 and 196,650 entries. The new family's whole taught set is 2,756 entries, 8,268 bytes. The taught single-operation pairs of the three earlier families are 30,576 entries, 91,728 bytes. As rendered prompts of 19 bytes, the longest, without the answer, the figures are 1,946, 7,767 and 31,050.)

In "What it cannot show", add to "Fixed memory": "At every rank the term takes more memory than the new family's whole taught set stored as entries (4.5, 17.9 and 71.4 times), and from rank 16 more than every taught single-operation pair of the life (1.6 times at rank 16)."

**6. Decay.** Add to the bullet "Optimizer":

> The settings are E40's. What the decay acts on is not. In E40 every weight of the lived network decays (where no gradient opposes it, by a factor 0.938 over 256 updates and 0.774 over 1,024). Here the lived weights are out of the optimizer and do not decay, and the decay acts on the two factors of the term alone. E48 and E40 therefore differ in two things, where the change is made and whether the lived weights decay, and the report says so.

Add to "What it cannot show":

> - **That a term is kept under the standing optimizer.** A term is kept exactly only while it is out of the optimizer, as it is here after the last update. Left in an optimizer with decay 0.25 and without items of its family, each factor shrinks by 0.99975 an update and their product by 0.9995: to 0.36 of its size after 2,048 updates and to 0.02 after 8,192.

## 4. Recommendations (not required)

1. **Table the other side of the comparison.** E40's receipts hold the earlier families' scores at every row. In lived S 73129, between row 0 and row 256, addition mod 97 goes from 380 to 109 of 400 and subtraction mod 73 from 302 to 132. A table of E40's earlier-family scores at rows 0, 128, 256 and 512 beside E48's (which are row 0's by construction) shows what E40 paid for its learning and costs no run.
2. **What the code rows carry alone.** At rows 256 and 1,024, score the new family once more with the gate forced off. The difference from the gated score is what the term carries; without this column a reader cannot tell how much of the learning is in the four code rows.
3. **How far the sweep's path through the term is from the exact one.** At every scored row under the sweep, record for each of the four factors the cosine between the factor and what the sweep uses in its place (the signs times the mean size). This is four numbers a row and needs no second backward pass. Check 5 passes under both rules for the reason given in question 2 and tests only that the path is connected; the brief should say that.
4. **P48.6 on differences.** In E40 itself R under backpropagation exceeds R under the sweep by -4.1, +12.9 and +9.7 points in the three seeds. As worded, P48.6 would nearly hold if the term cost nothing under either rule. Stating it on the differences from E40 (E48's R minus E40's R, sweep against backpropagation) makes it a prediction about the term.
5. **Name the runner.** `$M/e48_gated.py` is the copy of the specification file; the runner needs a name of its own in the brief.
6. **Rows 16, 32 and 64 of the term at ranks 4 and 16** can be committed as well (3.3 MB), so that the arrival of the term can be re-read later.

## 5. Findings, by question

### Question 1. The added term

Compared with 27.2 item by item, the brief and the specification file agree with the requirement.

| Requirement (27.2) | Brief and file |
|---|---|
| One term of rank r on the input map (768 by 512) and one on the recurrent matrix (768 by 256), each a product of two matrices | `Term`: `down_i` (r, 512), `up_i` (768, r), `down_h` (r, 256), `up_h` (768, r); `added` is two linear maps without bias (file lines 21 to 42) |
| Applied only on items of the new family | `if gate:` at the two places where the term is added (file lines 51 and 57) |
| Starts at zero | `up_i` and `up_h` are zero; `down` is drawn uniformly in ±1/√(inputs) from a generator seeded with `seed * 1000 + r`, the same for both rules |
| The four code rows free, everything else frozen | `freeze` leaves a gradient on the byte table only; `keep_code_rows` zeroes it outside rows 101 to 104; `make_opt` holds the term and the byte table |
| Ranks 4, 16, 64; each network under its own rule; 1,024 updates; rows 0, 16, 32, 64, 128, 256, 512, 1,024 | as required |

The count is right: r·512 + 768·r + r·256 + 768·r + 32 = 2,304·r + 32, which is 9,248, 36,896 and 147,488 (part A). My memo gave the same figures without the 32.

The reader was checked on the six lived networks of E40 (part B):

- With the term at zero and the gate read from the code byte, the reader reproduces all 18 counts of row 0 of E40's receipt (four held-out panels, four taught panels, four panels without letters, six earlier tasks) for each of the six networks, and the reduction score of each. The row that check 2 of the brief compares with therefore follows from the committed files.
- On the 17,000 items of E40's panels for lived S 73129, the logits with the gate off equal the plain reader's value for value, and so do the logits with the gate on and `up` at zero.
- With a term that holds something (`up` drawn in ±0.05) and with the four code rows moved by a standard normal draw, the earlier tasks' logits are still equal to the lived network's value for value at each rank, and their generated answers are the same item for item. Forced on, the same term moves their logits by 0.8 to 3.8, so the equality is the gate's doing and not an accident of a small term. The new family's logits move by 17 to 33.

Not checked by me: equality with `arch26.forward` in torch's arithmetic. The committed output reports it as bit-equal with the gate off and with the gate on at the start, for both rules and the three ranks.

### Question 2. Which factor starts at zero, and what each rule does with it

`up`, the factor on the output side, starts at zero; `down` is drawn. The following is worked out from `exp3_state.SignLinear` and from AdamW's formula, and part C shows it on stand-in errors.

- **First update.** The gradient of `up` is the error at the term's output times `down` applied to the input; it is the same under both rules and is not zero. The error sent on to `down` is the output error times `up` under backpropagation, and times sign(`up`) × mean|`up`| under the sweep. Both are zero, because the sign of a zero weight is zero. `down` therefore receives no gradient, and AdamW changes it only by the decay factor 0.99975. The term also sends no error back into the network at this update. The committed output agrees: the first-step gradients of `down` are exactly zero under both rules.
- **After the first update** every entry of `up` has moved by the learning rate, 0.001, in the direction of its gradient (AdamW's first step is the rate times the sign, for any gradient well above 10⁻⁸). All entries then have the same size, so sign(`up`) × mean|`up`| is `up` itself to within 3% of the rate in part C.
- **Second update.** The sweep therefore sends into `down` almost exactly the error that backpropagation sends (cosine 1.000000 in part C). The two rules part from the third update on, as the sizes of the entries of `up` spread. `down` moves by 0.74 of the rate an entry at this update; its mean size at the start is 0.022 on the input map and 0.031 on the recurrent matrix.
- **The sign rule is applied to each factor separately,** each with its own scale: `added` is two calls of `lin`. It is not applied to the product. This is the choice consistent with the harness, in which each linear map uses only its own weights, and it is stated exactly enough: the file is the specification and the brief's sentence matches it.
- **The frozen matrices** still pass errors, by signs under the sweep and exactly under backpropagation, and receive no update: `freeze` removes their gradients and the optimizer does not hold them. The rows of the byte table other than 101 to 104 have a zero gradient and no decay, so their step is exactly zero. The committed output lists the byte table as the only tensor changed, in rows 101 to 104 only, under both rules at the three ranks. Check 3 of the brief (the hash of the lived tensors at every row) would catch a slip.

The choice is sound. One consequence should be said in the brief (recommendation 3): check 5, that the gradient of `down` is not zero at update 2 under the sweep, passes under both rules for the same reason and says nothing about later updates.

### Question 3. How the gate is supplied

- **The file's gate is one value for a whole call** (`forward(m, term, x, local, gate, silence=False)`). The brief says that the stream, panels and scores are E40's "unchanged". E40's scorer answers the new family's panels and the earlier families' panels in one call of `Scorer.__call__`, through `exp16_integrated.score`, `exp11_cell.respond` and the forward that `arch26.install` sets, none of which has a gate; and E40's update calls `arch26.forward` by name (part F prints the lines). Neither the brief nor the file says how the gate reaches them: the file has no line that reads the gate from the item and none that replaces the harness's forward. Required change 2 states it.
- **Check 2 does not test the wiring.** At row 0 the term is zero, so the scores are E40's whether the gate is on or off. Check 3 tests that the earlier families are answered with the gate off. Nothing tests that the new family is answered with the gate on; if it were not, the scores would be low and would be read as a price. Check 3a of required change 2 closes this.
- **Earlier families cannot be reached by the term or by the freed rows.** The bytes of every earlier family and of the composition items are the digits, the four operator symbols, `=`, `;`, the letters A to Z and the codes `c n p q r s t u v w x y z`: 55 bytes with values 43 to 122, none of them 101 to 104; in every source file of the harness the codes `e f g h` belong to modulus 83 only (part F). An earlier item's code byte therefore never turns the gate on and its prompt never reads a freed row.
- **One path remains in principle and is closed in practice.** `respond` feeds each generated byte back as input. If a network generated `e`, `f`, `g` or `h` inside an earlier family's answer, the next step would read a freed row. In the six lived networks no generated byte of the six earlier panels is one of the four, nor is any on the life's panels at 0, 8, 12, 20 and 40 letters for lived S 73129 (part B). Because the readout and every weight an earlier item uses are frozen, the first generated byte is the same at every row, and so, step by step, are the others. Check 3, as required change 3 words it, verifies this at every row.
- **E40's scorer does not score the composition panels.** Its fourth score is the six tasks `add89, sub89, add97, sub97, add73, sub73` at 12 letters. The composition code `c` is byte 99, so composition items would be answered with the gate off if they were scored.
- **The gate is on from the first position,** before the code byte has been read; at that position the input is the start symbol, so the term acts there as a different starting state for the family. A learner that produced its own gate could turn it on from the second position at the earliest. Required change 2 says that the gate holds for every position.

### Question 4. The optimizer and decay

E40's reference runs used a fresh AdamW at rate 0.001, betas (0.9, 0.98), decay 0.25 on every tensor but the byte table (the receipts' `optimizer` record). The brief uses the same settings: a fresh AdamW, decay 0.25 on the four matrices of the term, none on the byte table. In settings the comparison is like for like.

In what the decay acts on it is not. In E40 the whole lived network decays while it learns; in E48 the lived weights are out of the optimizer, and only the term decays. The two sides of the comparison differ in this as well as in where the change is made, and the brief does not say so.

Under decay the term shrinks when it is not trained. In E48 nothing is trained after update 1,024 and the term is saved, so what is kept afterwards is the term exactly, and nothing in E48 depends on the point. The brief does not say that this holds only because the term leaves the optimizer: left in an optimizer with decay and without its family's items, the product shrinks to 0.36 of its size in 2,048 updates. Required change 6 adds both statements.

### Question 5. The reading

The requirement in 27.2 compares the reduction score at row 256 with E40's at row 256, with a mark of 10 points. Part D tabulates E40's reduction score from the receipts of the lived runs (each score rests on about 6,100 pairs, so the sampling error of a score is about 0.5 points):

| Run | 64 | 128 | 256 | 512 | 1,024 | Mean of 128, 256, 512 |
|---|---|---|---|---|---|---|
| lived S 73129 | 76.3 | 85.3 | 81.0 | 81.7 | 84.5 | 82.65 |
| lived S 93683 | 69.7 | 70.1 | 83.2 | 83.5 | 83.9 | 78.93 |
| lived S 116089 | 76.1 | 80.8 | 83.8 | 79.7 | 88.4 | 81.45 |
| lived B 73129 | 83.5 | 87.2 | 68.1 | 80.4 | 86.1 | 78.58 |
| lived B 93683 | 92.0 | 94.7 | 92.1 | 88.8 | 93.8 | 91.86 |
| lived B 116089 | 95.4 | 92.8 | 89.8 | 90.6 | 90.8 | 91.11 |

Three of the 42 moves between neighbouring rows from row 64 on, in these six runs, exceed 10 points: 13.0 (lived S 93683, rows 128 to 256), 19.1 and 12.3 (lived B 73129, into and out of row 256). **The reading as I wrote it is not sound.** In lived B 73129 the mark for "within 10 points" at row 256 would be set by a dip of the reference: a term at 60 would be within 10 points of 68.1 while that run's neighbouring rows are at 87.2 and 80.4.

The statistic proposed is R, the mean of rows 128, 256 and 512, on both sides. Its test is E40 against itself with the rows shifted by one: the mean of rows 64, 128 and 256, and the mean of rows 256, 512 and 1,024, each against R.

| Run | Row 128 minus row 256 | Row 512 minus row 256 | R a row early minus R | R a row late minus R |
|---|---|---|---|---|
| lived S 73129 | +4.4 | +0.7 | -1.8 | -0.3 |
| lived S 93683 | -13.0 | +0.3 | -4.6 | +4.6 |
| lived S 116089 | -2.9 | -4.0 | -1.2 | +2.5 |
| lived B 73129 | +19.1 | +12.3 | +1.0 | -0.4 |
| lived B 93683 | +2.6 | -3.3 | +1.1 | -0.3 |
| lived B 116089 | +3.0 | +0.8 | +1.6 | -0.7 |

With single rows, three of the twelve differences exceed the 10-point mark, one seed in each of three of the four cases (a rule and a direction of the shift), so that E40 read against its own curve one row along depends on the two-of-three rule. With R the largest difference is 4.6 points, under half the mark, and every seed is inside it. The best of the three rows is as stable (also 4.6) but selects upward and ignores speed; the mean is preferred. The three runs under Kolen–Pollack, which E48 does not use, move less (largest move 8.4).

The marks are restated on R in required change 1: no price if R is within 10 points of E40's or above it in at least two seeds; a price if it is 30 or more below in at least two seeds; some price otherwise. The three cases exclude one another and cover every outcome. R covers rows at which a term that learns more slowly is still rising; that is intended, since the reading asks whether the term learns as fast as the free network. The level at the end is tabled (rows 512 and 1,024).

### Question 6. Checks, cost, predictions, what is saved

- **Row 0 equal to E40's row 0 on every score:** check 2. The values exist and are reproducible (question 1).
- **The frozen tensors' hash at every row:** check 3, second clause. Sufficient.
- **The earlier families identical item for item with the term off:** check 3, first clause. E40's scorer returns counts and not answers, so the brief must say what is recorded; required change 3 does.
- **The batch digest equal to E40's:** check 4. E40's receipts have a running digest at rows 64, 128, 256, 512 and 1,024; they have no rows 16 and 32, so "at every row" cannot be met as written; required change 3 words it.
- **Cost.** E40's six receipts give 0.114 to 0.119 CPU-seconds an update and 13.7 to 14.4 a scored row on the same host (part E). A run of 1,024 updates with 8 scored rows is about 237 CPU-seconds; 18 runs are 1.2 CPU-hours, and under 1.4 with the earlier tasks answered a second time for the item-for-item check. The brief's 1.5 holds.
- **Predictions.** P48.1 is on the checks. P48.2 to P48.5 each name a reading for one rule and rank and can be scored from the table. P48.6 and P48.7 are on single rows and are restated in required change 1; recommendation 4 concerns P48.6.
- **What is saved.** The brief commits the term at rows 256 and 1,024 without stating the file's format. With the reading on three rows, the rows 128 and 512 are needed as well, and the format must be one the reader can load; required change 4 states both. The generated bytes of the reduction score are written at rows 0, 64, 256 and 1,024; rows 128 and 512 are added in required change 3.

### Question 7. What it cannot show

| Limit | Stated in the brief? |
|---|---|
| The gate is supplied | Yes: "Why" and the first bullet of "What it cannot show" |
| The terms grow by one a family; a replay store of equal size is the fair comparison | Yes; the count given for the store needs its basis (below) |
| Unrelated tasks need output units of their own | Yes: the readout is frozen |
| Composition across families; a second new family | Yes, two limits beyond those I listed |
| A term is kept only while it is out of the optimizer | No; required change 6 |

The count of items for a replay store (part A). The brief's 19 bytes is the longest prompt (12 letters) without its answer. The harness's stream does not store rendered items: it stores entries of a task and two operands and draws the letters afresh, so the unit of a store is the entry, 3 bytes.

| Rank | Numbers | Bytes | Entries of 3 bytes | Rendered prompts of 19 bytes | Times the new family's whole taught set (2,756 entries) |
|---|---|---|---|---|---|
| 4 | 9,248 | 36,992 | 12,330 | 1,946 | 4.5 |
| 16 | 36,896 | 147,584 | 49,194 | 7,767 | 17.9 |
| 64 | 147,488 | 589,952 | 196,650 | 31,050 | 71.4 |

The taught single-operation pairs of the three earlier families are 30,576 entries, 91,728 bytes: the term at rank 4 is 0.40 of that, at rank 16 1.61 times, at rank 64 6.43 times. The comparison is therefore less favourable to the term than the brief's figures suggest, and required change 5 puts these figures in the table.

## 6. What was not checked, and why

- **Everything that needs torch.** The self-checks of the specification file were read from the committed output and not rerun: bit-equality with `arch26.forward`, the gradients at the first two updates, the list of tensors that change, and the fall of the loss after one update. The reader shows the same properties of the forward in numpy arithmetic; the two rules on the two factors are shown in part C on stand-in errors, not on the network's.
- **The values of `down`.** They come from torch's generator and cannot be drawn here.
- **Whether the term learns under either rule,** and how the sweep behaves on it after the second update. Nothing was trained.
- **The runner.** It does not exist yet. The gate's wiring, the item-for-item record and the term files are specified by the required changes and will be checked from the receipts and the committed term files.
- **The cost of the gated forward itself.** The estimate uses E40's receipts; the term adds four small products a step.

## 7. Second audit

Not needed if the six changes are applied as written. If the reading is changed in any other way than required change 1 states, or the gate is supplied otherwise than from the item's code byte, the changed passage should be sent for a look before release.
