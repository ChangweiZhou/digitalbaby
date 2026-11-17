# Check of the brief of E51 before a worker runs it

17 November 2026. The brief checked is `TASK_E51_A_NEW_TASK_LEARNED_IN_CELLS_HELD_IN_RESERVE_20261117.md`, revision 1, in this folder. The code was read on the default branch `origin/codex/minifly-consolidated-20261023` at 7cabb13a with `git show`. Line numbers of the brief are those of revision 1.

**What this check is.** It was made at the request of the brief's author by an agent that worked from the brief and the code alone. It is a check of the same kind as the one the brief's status paragraph describes, and it does not replace the independent audit that the programme's rules ask for. No network was trained and no code of another branch was run. The only updates made are those of part E, in which the readout matrix `l3` alone is moved by the optimizer's arithmetic, for 8 updates and in one run for 256, to reproduce check 3(a). Every figure comes from receipts read through git or from my own numpy code, which re-implements the brief's embedding, forward pass, directions and rule from the brief's text and from `arch26.py`. The script is in the appendix and is run as `python3 -I -B <script> S` and `... B` from the repository root; "part A" to "part G" below name its printed lines. `np_gru.load_weights` was used only to read saved tensors, and `e51_design_checks.py` was read and not used. The two births are those of seed 73129. The fresh weights of the present-size network are numpy's draws at the scales of torch's constructors, because torch's own draws cannot be made without torch; the figures that depend on them (the counts at the input map and at `l3`, the sizes of the leaks) will differ a little in the worker's run. Where two figures are given, the first is for the birth under the sweep and the second for the birth under back-propagation. The kinds named in the headings are those of the request: (a) the worker cannot follow the brief as written, (b) a fragment is wrong for the code, (c) the first family is not kept exactly in `keyed`, (d) arms, (e) checks, (f) the record.

**Verdict.** The embedding fragment, the tensor names and shapes, the gate order and rows, the streams, the digests and the figures cited from E41, E43 and E48 are right. The brief should not go to a worker as it stands, for two reasons. The rule does not keep the first family exactly, so check 3(a) is failed by a correct implementation and the first question of the job is answered by a threshold (defect 1). The job does not measure where the new task is learned, and its statement that only the reserve can learn it is false in the embedded network (defect 2). Five smaller defects follow.

## Defects, most serious first

### 1. The first family is not kept exactly in `keyed`; a correct implementation fails check 3(a); "kept" at update 4,096 is not secured by the construction (kinds c and e)

**What is wrong.** The brief says that the second projection "makes the keeping exact" (line 102), predicts keeping "by construction" (P51.3) and sets a tolerance of 0.001 after 8 updates (check 3a). The rule protects only the directions whose eigenvalue exceeds one millionth of the largest (line 74). At two of the five matrices that criterion leaves out directions that the first family uses.

- **At `l3`** the 6,240 collected vectors have rank 513 of 513: the first family uses every direction of the readout's input. The criterion keeps the 369 to 382 largest (five runs: two births, three draws), and the cut falls where the spectrum has no gap (last eigenvalue kept 5.86e-5, first dropped 5.67e-5; under back-propagation 6.75e-5 and 6.63e-5; part B). Every first-family item, the 2,080 included, therefore has a part of its z2 outside the protected directions. Its 2-norm is up to 0.091 and 0.112 on the 2,080 items and up to 0.131 and 0.145 on held-out pairs and on items with 20 letters; its 1-norm is up to 1.9 and 2.2 (part C). One update of 0.001 an entry can move an output of such an item by up to 0.001 times that 1-norm. There is no direction at `l3` that the first family does not use, for two reasons. The old 256 units of z2 with the constant already have rank 257, of which 178 pass the threshold (sweep birth). And the added units 256 to 511 are not held in reserve: on a first-family item they read z1 and the old cells through fresh weights and are not zero.
- **At the input map** the codes of the 46 bytes in use have rank 46 and the criterion keeps 36 and 39 of them (part B). The part left out is largest for the letters: 2-norm up to 0.024 and 0.031, 1-norm up to 0.35 and 0.43 (part C).
- **At the recurrent matrix, `lm` and `l2`** the criterion keeps everything (129, 129 and 11, each equal to the rank). These three are exact; see the section on what is sound.

**What shows it.**

- *Random changes that the rule allows* (part D). With entries N(0, 0.03²) before the projection, the largest change of a first-family output is 0.011 to 0.015 through `l3` and 0.0002 to 0.005 through the input map. With entries N(0, 4²) it is 1.5 to 2.1 through `l3` and up to 0.6 through the input map, and 5 to 18 answers of each panel change. The size of the leak is the same on the 2,080 items as on held-out pairs and on items with 20 letters, so it comes from the threshold and not from the sample.
- *Check 3(a) itself, for `l3` alone* (part E). I applied the optimizer's arithmetic (AdamW, rate 0.001, betas 0.9 and 0.98, no decay) to `l3` for eight batches of the unrelated stream at the `keyed` start, with the brief's two projections and with every other tensor held at its start. The gradient of `l3` is the same under both rules (`exp3_state.py` line 102). The largest change of an output at an answer position is 3.9e-3 and 4.5e-3 on E43's held-out panel at 12 letters and 3.4e-3 and 4.1e-3 on 2,000 items without letters. It passes 0.001 at the second update. On the conflict stream the same code gives 2.6e-3 to 4.2e-3. The leak through the input map comes on top. With my draws a correct implementation therefore fails the tolerance of 0.001 by a factor of about four, and the worker would have to stop at the checks.
- *Growth.* The change grows in proportion to the number of updates while the readout's gradient keeps its direction: 0.13 after 256 updates of `l3` alone (`... S 256`). The rule bounds it only by 4,096 × 0.001 × 2, about 8 for an output. Of the first family's items without letters, 2.7% and 1.1% have a gap below 0.3 between their two largest outputs, and 9% and 5% a gap below 1 (part E). "Kept within 2 points" is therefore likely but is decided by how the leak grows, which the design does not control, and the share of answers equal to row 0's will not stay at 100%.

**Repair.** There are two ways, and the author has to choose.

*Repair A, which makes the keeping exact (recommended).* Hold the added feed-forward units in reserve as the cells are, and protect every direction that is used. I tested it in numpy: after a random change with entries N(0, 4²) of every entry that may then change, no first-family output moves by more than 1e-9, on taught and held-out pairs with 0 to 20 letters (part G; 0 to 40 letters in a separate run). The changes of wording are these.

1. In "The reserve's update gates", `keyed`, add: "The same forward holds the added units of the readout at zero on those items: after `z2` is computed, `z2 = torch.where(off, torch.zeros_like(z2), z2)`, with `off[b, t, j] = in_use[x[b, 1]] and j >= 256`."
2. In "The bytes in use and the directions in use", replace "exceeds one millionth of the largest" by "exceeds 1e-12 of the largest, that is, every direction the vectors span", and delete the row of `l3` from the table of matrices.
3. In the table of the rule, replace the row "`lm`, `l2`, `l3` and their biases | all | none" by two rows: "`lm`, `l2` and their biases | rows 0 to 255 | rows 256 to 511" and "`l3.weight[:, :256]` and `l3.bias` | not changed at all (restored after the step) | `l3.weight[:, 256:]` is free".
4. In check 2, replace the counts by "46 at the input map, 129 at the recurrent matrix, 129 at `lm` and 11 at `l2`". In check 3(a), replace "0.001" by "1e-4". In check 3(c), add "and every added unit of z2".

*Repair B, which keeps the design and states what it does.*

1. In line 74, replace "exceeds one millionth of the largest" by "exceeds 1e-12 of the largest at the input map, the recurrent matrix, `lm` and `l2` (46, 129, 129 and 11 directions: all that the vectors span), and one millionth of the largest at `l3`, where the vectors span all 513 directions and only the largest are protected".
2. Replace line 102 by: "The second projection keeps the step out of the protected directions. The keeping is then exact at four matrices. At `l3` it is not: each update can move an output of the first family by up to 0.001 times the 1-norm of the part of its z2 outside the protected directions (about 2 in the checker's reader)."
3. In check 3(a), replace "0.001" by "0.02".
4. Add to what is recorded in `keyed` at every scored row: "the largest change from row 0 of an output at an answer position on the panels of check 1, under teacher forcing".
5. In P51.3, replace "By construction" by "Expected; the leak at `l3` is not bounded by the construction".

A smaller threshold at `l3` reduces the leak and the room left for the new task together. At 1e-8 the rule keeps 468 and 491 directions, the eight updates change an output by 1.3e-4 to 2.6e-4, and the part of the new task's z2 that the readout may still use falls from 0.87 and 0.88 to 0.37 and 0.27, of a norm of 12 (parts B and E). At 1e-10 it keeps 495 and 512 of 513.

In either repair, line 13 of "Terms" should say what is computed: the directions are taken from 2,080 taught items, and under repair B at `l3` they are the largest directions and not all that are used.

### 2. The old cells can learn a new response to a new byte, and nothing measures whether the new task is learned in the reserve (kinds d and f)

**What is wrong.** "One limit known beforehand" (line 28) and the second of the "Two properties" (line 113) say that a new byte has no direction of its own at the input map, that the old cells cannot learn a new response to it, and that only the reserve can. The figure of 99.9% is right for a born `l1` and wrong for the embedded network, in which half of `l1`'s units are fresh.

**What shows it** (parts C and F).

- The share of a new byte's code that lies inside the directions in use is 99.97% for the half-width birth alone. In the embedded network it is 98.3% and 97.7% (median over the 98 bytes of the unrelated task). The part outside has a 2-norm of 1.05 and 1.24, where a whole code has about 8, and a 1-norm of 14 and 18, against at most 0.35 and 0.43 for a byte in use.
- One update can therefore move an old cell's gate input on a new byte by up to 0.014 and 0.019. For a reserve cell the figure is 0.14. The old cells are slower by a factor of ten and are not prevented.
- I built a change of the old cells' rows of the input map that lies wholly outside the directions in use (largest entry 0.99 and 0.96). It moves an old cell's gate inputs by 1.0 on the unrelated task's code and query bytes, by at most 0.023 and 0.030 on a byte in use, and a first-family output by at most 0.039 and 0.044.
- Beside the input map the rule leaves free the weights from reserve cells into old cells, `l2` on every byte but eleven, and the free directions of `l3`. The old cells also answer new bytes with their fixed weights, since 98% of a new code lies in directions they read.

**Consequence.** If `keyed` learns the new task, no recorded figure says whether the reserve, the old cells or the readout learned it. The arm table's "whether the new task can be learned in the reserve" (line 121) and the last sentence of "Predictions" cannot be read from the run.

**Repair.**

1. Replace the last two sentences of line 28 by: "That figure is for a born `l1`. In the embedded network half of `l1`'s units are fresh, and a byte that has never occurred has about 2% of its code, a 2-norm of about 1 in 8, outside the directions in use at the input map. The old cells' rows can therefore learn a response to a new byte, about ten times more slowly for each update than a reserve cell's rows. The job does not confine the new task to the reserve; it measures where the task is learned."
2. Delete from line 113 the sentence "And in the present code the old cells cannot learn a new response to a new byte (the limit above); only the reserve can."
3. Add to "What is recorded at every scored row", under the new task: "E43's score 3 a second time with the reserve cells held shut on every item (the added line with `shut[b, i] = i >= 128` for every b), in every arm." Under the weights add: "score 4 separately for the old cells' rows and for the reserve cells' rows of the input map and of the recurrent matrix."
4. Add to "How the result is stated": "*Learned in the reserve*: *learned*, and at update 4,096 the score with the reserve shut is below 20%. Otherwise the report gives both scores."
5. In line 121 replace "whether the new task can be learned in the reserve" by "whether the new task is learned, and how much of it is lost when the reserve is shut".
6. If the author wants the limit to hold by construction, the row of the rule's table for the input map can read "the old cells' rows: not changed at all (restored after the step)". This also removes the leak at the input map of defect 1. Item 3 is needed in either case.

### 3. `keyed` differs from `open` in four things, and the reading of a failed P51.4 does not follow (kind d)

**What is wrong.** `keyed` and `open` differ in the start file (reserve gates at fresh draws against a bias of 10, lines 55 and 56), in the forward pass (the added line), in the projections, and in what is frozen (`l1` and 46 rows of the byte table). The *cost* (line 157) is `keyed`'s row against `open`'s row, so it cannot be assigned to the rule. In `open` the bias of 10 puts 1 − z near 5e-5 for every reserve cell on every item, new items included; in `keyed` the reserve is open on new items from the first update. Check 3(b) is the missing contrast, the forward of `keyed` without the rule, but it runs for 8 updates and is not scored. The sentence "If P51.4 fails under both rules, cells in reserve that read the old cells but may not change them are not enough to learn a new task" (line 173) does not follow either: the weights from reserve cells into old cells are free, so the reserve may change the old cells on new items, and a failure could as well come from the frozen `l1`, of which half is fresh, or from the readout's confinement.

**Repair.** Add a fourth arm, `keyed-free`: the start file and the forward pass of `keyed`, every weight free, nothing projected and nothing restored. It is four more runs (two rules, two tasks), about 40 CPU-minutes by E43's recorded costs. Then state the cost of the rule as `keyed` against `keyed-free`, and the cost of the whole device as `keyed` against `open`. Replace the sentence of line 173 by: "If P51.4 fails under both rules and `keyed-free` learns the task, the rule is what prevents it; if `keyed-free` fails too, the embedding is. Which part of the rule binds is not shown." If the arm is not added, replace line 157's last sentence by "Its *cost* is that first row against the first such row of `open`, a learner without the rule whose reserve starts shut", and delete the sentence of line 173.

### 4. "The classes of E43's second addendum" do not exist (kinds a and f)

**What is wrong.** Line 144 asks for "what the wrong answers on reduction pairs are, in the classes of E43's second addendum". The second addendum has no classes of wrong answers, in its text, its runner, its reading script, its tables or the report.

**What shows it.** This command prints nothing:

```
git grep -n -i -E "wrong|class" origin/codex/minifly-consolidated-20261023 -- MOD97_LOCAL_RULE_20261106/e43_addendum2.py \
  MOD97_LOCAL_RULE_20261106/e43_addendum2_read.py MOD97_LOCAL_RULE_20261106/exp45/e43/addendum2/E43_ADDENDUM2_TABLES.md
```

The nearest things in the record are the observation in `exp45/e43/E43_REPORT.md`, line 53 (the end symbol alone; symbols of the new task), and the classes that the reviewer used for E48 (`E48_RESULT_VERIFICATION_20261117.md`, section 8).

**Repair.** Replace "in the classes of E43's second addendum" by: "in these classes, tried in this order on each wrong answer: (1) the end symbol alone; (2) an answer that contains a byte of 128 or above; (3) two digits and the end symbol that are the right answer under modulus 97; (4) two digits and the end symbol that differ from the right answer by exactly 1 or exactly 10; (5) any other two digits and the end symbol; (6) anything else. The generated bytes of the reduction pairs are written to a file at rows 0, 64 and 4,096."

### 5. One clause of check 3(a) cannot fail on the unrelated stream (kind e)

**What is wrong.** Check 3(a) asks, after 8 updates of the unrelated stream, that "every tensor or row that the table calls not changed is bit-equal to its start". For the 46 rows of the byte table this holds whether or not the worker has written the restoring step. The unrelated task's items contain only bytes of 128 and above (`e43_task.py` lines 9 to 11 and 23; asserted in `exp43_unrelated.py` line 206), the start symbol's row is the padding row and receives no gradient (`exp41_scale.py` line 52, as `arch20.py` line 43), and the byte table has no decay (`exp16_integrated.py` line 183). A row with a gradient of zero at every update is not moved by the optimizer. The record shows it: the 211 rows of bytes that the births never met, and the end symbol's row, are bit-equal in the two half-width births of seed 73129 and in E26's birth of that seed, after 40,960 and 24,576 updates. The clause can fail for `l1` only.

**Repair.** Add to check 3(a): "The same on 8 updates of the conflict stream, whose items contain bytes in use. With the restoring of the byte table switched off, at least one of the 46 rows differs after those 8 updates."

### 6. Places where the worker cannot follow the brief as written (kind a)

1. **Check 6 and the reserve's biases.** "`open` with the embedding replaced by E26's birth" (line 139) leaves open whether the biases of rows 384 to 511 are set to 10 and 0, which line 55 makes part of `open`. With them set, the tensors hash of E43 cannot be reproduced. Replace by: "The runner of `open`, started from E26's birth of the seed as it is, with no embedding and no change of any bias, reproduces …".
2. **How the added line reaches the scorer.** `exp11_cell.respond` calls the module's own name `forward` (line 200), which `arch26.install` sets (line 85); `exp26_candidate.step` calls `arch26.forward` by name (line 86). The copy in the new runner acts in neither unless it is bound. Add to line 56: "After `arch26.install()`, set `exp11_cell.forward` to the runner's copy, and call the copy in the runner's copy of `step`. `exp43_unrelated.eq_states` and `e43_addendum2.rehearse` call `arch26.forward` directly and are not used in `keyed`." Check 4 would catch an unbound scorer (see below), so this saves a round and does not change a result.
3. **The gates are not returned.** `arch26.forward` returns `dict(z1=z1, h=hs)` (line 75). The reserve's figures of line 146 need z. Add to line 56: "and the copy also returns the list of z".
4. **Check 1's items and outputs.** "2,000 of its items without letters" (line 134) names no items. Replace by: "the 2,000 items `Random(f"e51|{seed}|check1").sample(items, 2000)` of the 31,684 items of `exp43_unrelated.first_family_items()`, in its order of tasks". Add: "Outputs are taken under teacher forcing with the right answer."
5. **The score of `conflict` in "learned".** E43's score 3 for `conflict` is two panels, add97 and sub97, of 400 held-out pairs each (`exp43_unrelated.py` line 142). Add to line 157: "for `conflict`, the 800 items of the two panels together".
6. **The weight files are ignored by git.** `.gitignore` line 2 is `*.pt`. E43's first addendum had to repair exactly this. Add to line 150: "The start files and the weights are added with `git add -f`, each after its hash has been compared with its receipt."
7. **The answers themselves.** The share of answers equal to row 0's (line 148) needs the generated bytes, and `exp43_unrelated.score1` returns counts only (lines 74 to 85). Add: "the runner calls `exp11_cell.respond` itself on the items of scores 1 and 2 and keeps the bytes of row 0".
8. **Batches for the directions.** Add to line 74: "one number of letters at a time, so that no row is padded". `exp45.batch_tensor` gives a shorter row the end symbol and then start symbols as inputs (lines 160 to 165), which would add a 47th byte and further states to the directions.
9. **The checkout.** The runner goes into `$M`, which exists on the default branch and not on the reviewer's, while the brief and `$R` exist only on the reviewer's. Add to the line of shell names, as the notes for E44 and E50 in this folder have it: "`D` is your checkout; the brief and `$R` are read from `$V` with `git show`".

### 7. Statements that the record or the code does not support (kind f)

1. **Line 51**, "The weights from reserve cells into old cells are not zero, because otherwise no error would reach the reserve." The readout reads the state of every cell through `lm` (`arch26.py` line 74), and columns 128 to 255 of `lm` keep fresh draws, so the error reaches the reserve at the three answer positions whatever those weights are. Replace the reason by: "they keep the fresh draws, and through them the reserve can act on the old cells on the items of a new task".
2. **Line 27**, "The reviewer's arithmetic for this job". The script's own header (lines 4 to 8) says that it checks another form, in which the reserve's gates are shut by a bias and their rows are projected. Its statement about the scores holds for the `keyed` rule as well at that size of change (part D, entries N(0, 0.03²): no more than two answers of a panel change), but the outputs move by 0.011 to 0.015. Add: "in the form with gates shut by a bias; the `keyed` form was not run".
3. **Line 17**, "255 or 256 of the 257 directions". By this brief's own criterion the two E26 births of seed 73129 give 257 of 257 at the recurrent matrix and at `lm` on the 2,080 items. The memo's figure uses a share of the trace. Say which criterion is meant.
4. **Line 23**, "nearly so at update 64". Five networks are at 0.01 to 0.10% and the sixth, back-propagation of seed 93683, at 4.92% (385 of 7,832). Replace by "at 0.1% or less in five networks and 4.9% in the sixth at update 64".
5. **Line 24**, "95 to 105%". The receipts give 94.8 to 105.0%; the lowest is back-propagation of seed 73129 with decay (93.07 after 98.14).

## What I verified and found sound

**The embedding and the names (kinds a and b).** The brief's fragment, applied line for line in numpy, gives a network whose outputs at the answer positions equal the half-width birth's with a difference of exactly 0 in float64, and also in float32, on the 2,080 items, E43's held-out panel at 12 letters, 2,000 items without letters and held-out pairs at 20 letters (part A). This confirms the tensor names, the shapes, the gate order r, z, n, the rows `g * 256 + i`, and that both bias vectors of the old cells are carried over. Rows 384 to 511 are the reserve's update-gate rows: with the biases of `open` the outputs differ from the birth's by 3.0e-3 and 3.6e-3 on the panel and 7e-4 and 8e-4 on the items without letters, inside check 1's 0.02. My reader gives the counts of E41's receipts on the held-out panel (1,155 = 292 + 283 + 290 + 290 and 1,161 = 283 + 284 + 299 + 295). `exp41_scale.new_net(seed, H, U)` exists (lines 60 and 61), and so do `E.DEV`, `E.split`, `E.render`, `X16.entry_stream` and `X16.family_entries` with the signatures the brief uses.

**The update (kind b).** `exp26_candidate.step` is `zero_grad`, `backward`, `opt.step()` (lines 88 to 90), so the two projections can be placed as the brief says. The shapes of the fragment are right for all five pairs of tensors. Under the sweep each map returns the exact gradient for its own weights (`exp3_state.py` lines 102 and 103), so the projected gradient is outside the directions under both rules. The two bias vectors enter as `arch26.py` lines 64 to 66 show, the second inside the product with r in the candidate; treating each (weight, bias) pair as one map of its own input, as the brief does, is right for both. The loss is taken at the positions whose inputs are `=` and the first two answer bytes (`exp45.py` line 168), which are the brief's three answer positions.

**The arms (kind d).** `open` and `rehearse` start from the same file and differ only in the stream. The stream of `rehearse` on `unrelated` is the second addendum's, with its generator name and seeds (`e43_addendum2.py` lines 128 to 131), and the conflict stream and its panels are E43's (`exp43_unrelated.py` lines 140 and 142). P51.1, P51.2 and P51.7 agree with E43's receipts of seed 73129: the reduction score is 0.10 and 0.05 at update 64 of `unrelated`, 0.05 and 0.00 at update 4,096 of `conflict`, and 0.5 points above and 1.4 points below its start at update 4,096 under rehearsal without decay.

**Paths to the first family in `keyed` (kind c).** Beside the two leaks of defect 1 I found none.

- A random change of any size of the recurrent matrix, `lm`, `l2` and the free rows of the byte table, the end symbol's row included, moves no first-family output by more than 2e-12 (part D). This covers every row, both bias vectors of the old cells, taught and held-out pairs, and 12 and 20 letters. The reason is that the old cells' state fills all 128 directions, so the only free columns are the reserve's, whose state is exactly zero.
- The reserve's own rows are free and cannot reach a first-family item, because the added line makes the state exactly zero.
- Teacher forcing against generation: `exp11_cell.respond` groups prompts by length (line 192), so nothing is padded, and it reads the last position only. At the start every item of the held-out panel and of the 31,684 items without letters generates two digits and the end symbol, so generation presents the same bytes at the same positions as teacher forcing, and the eleven directions of `l2` cover it. A fed-back byte outside the eleven can occur only after an answer is already wrong.
- Padding in the update: a shorter row receives the end symbol and start symbols after its last answer position (`exp45.py` lines 160 to 165), which no loss can reach. The condition reads `x[:, 1]`, which is the code byte in the update and in `respond` (line 196).
- The optimizer: with the decay of group 0 at zero and the byte table's at zero there is no change besides the step, and the second projection covers the momentum. The start symbol's row is zero in both births. Rounding the projected step to single precision costs at most 6e-8 of a weight's size for each update.

**The checks (kind e).** Check 4 can fail as intended: with the `keyed` weights and without the added line, 32 and 48 of the 2,080 items and 33 and 44 of the 1,200 panel items change their answers (part A). Check 3(b) holds with room: 1.9 and 2.1 from `l3` alone. Check 2's counts 129, 129 and 11 are what I find, and its ranges at the input map and at `l3` contain my 36, 39 and 369 to 382.

**The record (kind f).**

- The digests of the 2,080 items are `d22b5733778491b1`, `54bdb3c36ba2dc6e` and `f29bee3306c1a5d1` for the three seeds, and the bytes in use are 46. The 2,080 items hold 480 to 497 distinct pairs of each operation, of 3,960 taught.
- My own re-implementation of the streams and of `exp26_candidate.Digest` gives the receipts' digests at rows 64 to 4,096 for `unrelated`, `conflict` and `unrelated-rehearse`, three seeds, both rules. For seed 73129 at 4,096 they are `f60110540a0c32e7`, `60870058e0435e12` and `11fcebdb0e7a36fd`.
- E43's six receipts of `unrelated` without decay give 0 of 3,916 for add and for sub at update 4,096 and 300 of 300 on both panels of the new task. In E43's `conflict` receipts the new task's two panels are at 93 to 98% at 4,096 and together at 80% or more from row 256 in seed 73129, so *learned* can be reached there.
- E41's six receipts give 92.83 to 96.17 on held-out add and sub at 12 letters and 90.90 to 94.65 on reduction pairs. The file hash and the tensors hash of the two births of seed 73129 equal their receipts'.
- E48's 18 receipts on `origin/claude/sweet-turing-lwi3no` and E40's on the default branch agree with the brief: at ranks 16 and 64 the reduction score is above E40's at update 1,024 in 12 of 12 runs and below it at update 64 in 12 of 12, and the digest of the earlier answers is the same at every row of every run.
- In E43's second addendum the unrelated task is behind the unrehearsed run at update 256 in 12 of 12 comparisons, on both panels.
- E43's runs of seed 73129 took 530 to 650 CPU-seconds each, so about 3 CPU-hours for twelve runs with more rows is plausible.

Some figures of this note come from runs of the same code that the appendix script does not repeat: the float32 comparison, the generation of the 31,684 items, the direction counts of the E26 births, the rank of the old units of z2, the eight updates on the conflict stream and with other draws, the comparison of byte-table rows between births, the stream digests, and the test of repair A at 40 letters.

## What I could not check

- **torch's own draws.** I could not make torch's draws for the fresh weights. The counts at the input map and at `l3` and the sizes of the leaks are for numpy's draws.
- **The tolerance of 1e-5 in check 1.** I could not test it under torch's single-precision arithmetic. numpy in float32 gives a difference of exactly zero, with another library for the matrix products.
- **Anything that needs training.** I do not know how large the leak of defect 1 is at update 4,096, whether `keyed` learns either task, or how fast the reserve opens in `open`. The eight updates of part E move `l3` alone.
- **The leak through the input map under real gradients.** I measured it under random changes and bounded it; I did not form the gradient through the GRU.
- **Check 6 on the worker's host.** E43's receipts were made with torch 2.15.0+cpu on Python 3.13.18. Whether another host gives the same tensors hash at row 64 cannot be known beforehand.
- **Seeds 93683 and 116089.** I checked only the digests of their items and streams. No birth of those seeds was loaded.
- **The memo's section 29.** I checked only the one count in defect 7.
- **The torch lines proposed in the repairs.** They were written from the code and not run; the numpy form of repair A was run (part G).

## Appendix: the script

It writes nothing and reads one birth through git. Saved as a file outside the repository, it runs from the repository root in about one minute.

```python
# Run from the repository root:  python3 -I -B check_e51_brief.py [S|B] [updates in part E, default 8]      (numpy and the standard library; about one minute)
# It reads one half-width birth of seed 73129 through git and writes nothing. No network is trained; part E moves the readout matrix l3 alone.
# The fresh weights are numpy's draws at torch's scales.
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, "PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109")
from random import Random
import numpy as np
from np_gru import load_weights                      # used only to read the saved tensors

RULE, SEED, EOS, BOS = (sys.argv[1] if len(sys.argv) > 1 else "S"), 73129, 256, 257
N_UP = int(sys.argv[2]) if len(sys.argv) > 2 else 8        # updates in part E
SYM = {"add": "+", "sub": "-", "max": ">", "min": "<"}
DEV = [("add", 89, b"p"), ("sub", 89, b"q"), ("max", 89, b"v"), ("min", 89, b"w")]
OLD = np.concatenate([np.arange(g * 256, g * 256 + 128) for g in range(3)])          # the old cells' rows of the two GRU matrices
MATS = {"ih": ("rnn.weight_ih_l0", "rnn.bias_ih_l0"), "hh": ("rnn.weight_hh_l0", "rnn.bias_hh_l0"), "lm": ("lm.weight", "lm.bias"),
        "l2": ("l2.weight", "l2.bias"), "l3": ("l3.weight", "l3.bias")}

def render(task, a, b, k, rng):                      # exp11_cell.render
    op, m, code = task
    d = [rng.choice(list(b"ABCDEFGHIJKLMNOPQRSTUVWXYZ")) for _ in range(k)]
    ans = {"add": (a + b) % m, "sub": (a - b) % m, "max": max(a, b), "min": min(a, b)}[op]
    return list(code) + list(f"{a:02d}".encode()) + [ord(SYM[op])] + d + list(f"{b:02d}".encode()) + [ord("=")], list(f"{ans:02d}".encode()) + [EOS]

def split(task, frac, seed):                         # exp11_cell.split
    op, m, code = task
    pairs = [(a, b) for a in range(m) for b in range(m)]
    Random(seed * 1000 + round(frac * 100) + m * 7 + code[0]).shuffle(pairs)
    return pairs[:round(frac * len(pairs))], pairs[round(frac * len(pairs)):]

def panel(task, pairs, seed, k, n):                  # exp11_cell.panel for one number of letters
    rng = Random(seed + 3); sub = rng.sample(pairs, n)
    return [render(task, a, b, k, rng) for a, b in sub]

def unrelated_batches(seed):                         # e43_task.batches
    pairs = [(a, b) for a in range(128, 192) for b in range(128, 192)]
    Random(f"e43|{seed}|split").shuffle(pairs); taught = pairs[:2048]; rng = Random(f"e43|{seed}|stream")
    def stream():
        while True:
            order = list(range(len(taught))); rng.shuffle(order)
            for i in order:
                a, b = taught[i]; k = rng.randint(0, 12)
                yield [224, a, b] + [rng.choice(list(range(192, 224))) for _ in range(k)] + [225], [a, b, EOS]
    st = stream()
    while True:
        yield [next(st) for _ in range(64)]

def sig(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))

def forward(W, x, in_use=None, zero_added=False):
    """arch26.forward in float64. in_use: the brief's added line (z = 1 for cells 128.. on items whose first byte is in use).
    zero_added: the repair of defect 1 (units 256.. of z2 held at zero on the same items). Returns outputs and input vectors."""
    U = W["rnn.weight_hh_l0"].shape[1]
    z1t = np.tanh(W["book.weight"] @ W["l1.weight"].T + W["l1.bias"]); git = z1t @ W["rnn.weight_ih_l0"].T + W["rnn.bias_ih_l0"]
    h = np.zeros((len(x), U)); key = in_use[x[:, 1]] if in_use is not None else np.zeros(len(x), bool)
    out = []; col = {m: [] for m in MATS}
    for t in range(x.shape[1]):
        z1 = z1t[x[:, t]]; gi = git[x[:, t]]; col["ih"].append(z1); col["hh"].append(h.copy())
        gh = h @ W["rnn.weight_hh_l0"].T + W["rnn.bias_hh_l0"]
        r = sig(gi[:, :U] + gh[:, :U]); z = sig(gi[:, U:2 * U] + gh[:, U:2 * U]); z[key, 128:] = 1.0
        h = (1 - z) * np.tanh(gi[:, 2 * U:] + r * gh[:, 2 * U:]) + z * h
        if t >= x.shape[1] - 3:                      # the three answer positions
            z2 = np.tanh(z1 @ W["l2.weight"].T + W["l2.bias"] + h @ W["lm.weight"].T + W["lm.bias"])
            if zero_added: z2[key, 256:] = 0.0
            col["lm"].append(h.copy()); col["l2"].append(z1); col["l3"].append(z2); out.append(z2 @ W["l3.weight"].T + W["l3.bias"])
    return np.stack(out, 1), {m: np.concatenate(v) for m, v in col.items()}

def run(W, items, in_use=None, zero_added=False):
    """Teacher forcing, x = [start] + prompt + answer[:-1], items grouped by length. Outputs (items, 3, 257), the collected vectors, the targets."""
    out = np.empty((len(items), 3, 257)); col = {m: [] for m in MATS}; groups = {}; tgt = []
    for i, (p, a) in enumerate(items): groups.setdefault(len(p), []).append(i)
    for idx in groups.values():
        o, c = forward(W, np.array([[BOS] + items[i][0] + items[i][1][:-1] for i in idx]), in_use, zero_added); out[idx] = o
        for m in MATS: col[m].append(c[m])
        tgt += [items[i][1][j] for j in range(3) for i in idx]                          # in the order of the vectors collected for lm, l2 and l3
    return out, {m: np.concatenate(v) for m, v in col.items()}, np.array(tgt)

def one(X):
    return np.concatenate([X, np.ones((len(X), 1))], 1)

def away(D, V):
    return D - (D @ V) @ V.T

sd0 = {k: v.astype(np.float64) for k, v in load_weights(f"MOD97_LOCAL_RULE_20261106/exp45/e41/e41_{RULE}_h256_f50_{SEED}.pt")[0].items()}
g = np.random.default_rng(1); u = lambda shape, bound: g.uniform(-bound, bound, shape)
sd = {"book.weight": sd0["book.weight"].copy(), "l1.weight": u((512, 8), 8 ** -.5), "l1.bias": u(512, 8 ** -.5),
      "rnn.weight_ih_l0": u((768, 512), 1 / 16), "rnn.weight_hh_l0": u((768, 256), 1 / 16), "rnn.bias_ih_l0": u(768, 1 / 16), "rnn.bias_hh_l0": u(768, 1 / 16),
      "lm.weight": u((512, 256), 1 / 16), "lm.bias": u(512, 1 / 16), "l2.weight": u((512, 512), 512 ** -.5), "l2.bias": u(512, 512 ** -.5),
      "l3.weight": u((257, 512), 512 ** -.5), "l3.bias": u(257, 512 ** -.5)}
sd["l1.weight"][:256] = sd0["l1.weight"]; sd["l1.bias"][:256] = sd0["l1.bias"]                     # the brief's fragment, line for line
for gate in range(3):
    new, old = slice(gate * 256, gate * 256 + 128), slice(gate * 128, (gate + 1) * 128)
    sd["rnn.weight_ih_l0"][new, :256] = sd0["rnn.weight_ih_l0"][old]; sd["rnn.weight_ih_l0"][new, 256:] = 0.0
    sd["rnn.weight_hh_l0"][new, :128] = sd0["rnn.weight_hh_l0"][old]
    sd["rnn.bias_ih_l0"][new] = sd0["rnn.bias_ih_l0"][old]; sd["rnn.bias_hh_l0"][new] = sd0["rnn.bias_hh_l0"][old]
sd["lm.weight"][:256, :128] = sd0["lm.weight"]; sd["lm.bias"][:256] = sd0["lm.bias"]
sd["l2.weight"][:256, :256] = sd0["l2.weight"]; sd["l2.weight"][:256, 256:] = 0.0; sd["l2.bias"][:256] = sd0["l2.bias"]
sd["l3.weight"][:, :256] = sd0["l3.weight"]; sd["l3.weight"][:, 256:] = 0.0; sd["l3.bias"][:] = sd0["l3.bias"]
op = {k: v.copy() for k, v in sd.items()}; op["rnn.bias_ih_l0"][384:512] = 10.0; op["rnn.bias_hh_l0"][384:512] = 0.0   # the start of `open`

rng = Random(f"e51|{SEED}"); taught_items = []       # the 2,080 items, k = 0 first
by_k = {k: [] for k in range(13)}
for t in DEV:
    taught = split(t, .5, SEED)[0]
    for k in range(13): by_k[k] += [render(t, a, b, k, rng) for a, b in rng.sample(taught, 40)]
for k in range(13): taught_items += by_k[k]
in_use = np.zeros(258, bool); in_use[[BOS] + [b for p, a in taught_items for b in p + a[:-1]]] = True
held12 = [it for t in DEV for it in panel(t, split(t, .5, SEED)[1], SEED, 12, 300)]                 # E43's score 2 (and E41's panel A)
plain = Random(f"audit|{SEED}").sample([render(t, a, b, 0, None) for t in DEV for a in range(89) for b in range(89)], 2000)
held20 = [it for t in DEV for it in panel(t, split(t, .5, SEED)[1], SEED + 11, 20, 300)]
SETS = {"the 2,080 items": taught_items, "held-out pairs, 12 letters": held12, "2,000 items, no letters": plain, "held-out pairs, 20 letters": held20}
ref = {n: run(sd, it, in_use) for n, it in SETS.items()}
right = lambda o, it: (o.argmax(-1) == np.array([a for _, a in it])).all(1)

print(f"A. rule {RULE}, seed {SEED}; bytes in use {int(in_use.sum())}; half-width birth right on the held-out panel at 12 letters: {int(right(run(sd0, held12)[0], held12).sum())} of 1200")
for n, it in SETS.items():
    b = run(sd0, it)[0]; nokey = run(sd, it)[0]; o = run(op, it)[0]
    print(f"   {n}: keyed start against the birth {np.abs(ref[n][0] - b).max():.1e}; `open` start {np.abs(o - b).max():.1e}; keyed weights without the added line "
          f"{np.abs(nokey - b).max():.2f}, answers changed {int((nokey.argmax(-1) != b.argmax(-1)).any(1).sum())} of {len(it)}")

V, spec = {}, {}
for m in MATS:                                       # the brief's directions: eigenvalue above one millionth of the largest
    X = one(ref["the 2,080 items"][1][m]); w, E = np.linalg.eigh(X.T @ X / len(X)); w, E = w[::-1], E[:, ::-1]
    k = int((w > 1e-6 * w[0]).sum()); V[m], spec[m] = E[:, :k], (w, E)
    print(f"B. {m}: {k} directions of {X.shape[1]}; rank of the collected vectors {np.linalg.matrix_rank(X)}; eigenvalues: largest {w[0]:.2e}, last kept {w[k - 1]:.2e}, "
          f"first dropped {w[k] if k < len(w) else 0:.2e}; kept at 1e-8, 1e-10, 1e-12 of the largest: {', '.join(str(int((w > t * w[0]).sum())) for t in (1e-8, 1e-10, 1e-12))}")

for n in SETS:                                       # C. what the first family has outside the protected directions
    r3 = away(one(ref[n][1]["l3"]), V["l3"]); ri = away(one(ref[n][1]["ih"]), V["ih"]); rh = away(one(ref[n][1]["hh"]), V["hh"])
    print(f"C. {n}: part outside the directions, largest 2-norm / 1-norm: l3 {np.linalg.norm(r3, axis=1).max():.3f} / {np.abs(r3).sum(1).max():.2f}; "
          f"input map {np.linalg.norm(ri, axis=1).max():.4f} / {np.abs(ri).sum(1).max():.3f}; recurrent matrix {np.linalg.norm(rh, axis=1).max():.0e}")
codes = one(np.tanh(sd["book.weight"] @ sd["l1.weight"].T + sd["l1.bias"])); rc = away(codes, V["ih"]); new = np.arange(128, 226)
half = one(np.tanh(sd0["book.weight"] @ sd0["l1.weight"].T + sd0["l1.bias"])); wq, Eq = np.linalg.eigh(half[in_use].T @ half[in_use]); rq = away(half, Eq[:, wq > 1e-12 * wq.max()])
print(f"C. a byte of the unrelated task: share of its code's squared norm inside the directions in use: embedded network median {1 - np.median((np.linalg.norm(rc[new], axis=1) / np.linalg.norm(codes[new], axis=1)) ** 2):.4f}, "
      f"half-width birth alone {1 - np.median((np.linalg.norm(rq[new], axis=1) / np.linalg.norm(half[new], axis=1)) ** 2):.4f}; outside part in the embedded network: 2-norm median {np.median(np.linalg.norm(rc[new], axis=1)):.2f}, "
      f"1-norm median {np.median(np.abs(rc[new]).sum(1)):.1f} (a byte in use: at most {np.abs(rc[in_use]).sum(1).max():.2f}; a whole code: {np.median(np.abs(codes[new]).sum(1)):.0f})")

def changed(W, scale, only, Vs, zero_added=False):   # D. a random change of every entry that the table lets change
    gen = np.random.default_rng(100); Wn = {k: v.copy() for k, v in W.items()}
    if "book" in only: Wn["book.weight"][~in_use] += gen.normal(0, scale, (int((~in_use).sum()), 8))
    for m in only:
        if m == "book": continue
        D = gen.normal(0, scale, (W[MATS[m][0]].shape[0], W[MATS[m][0]].shape[1] + 1))
        if not zero_added:
            rows = OLD if m in ("ih", "hh") else np.arange(len(D)); D[rows] = away(D[rows], Vs[m])
        elif m == "l3": D[:, :256] = 0.0; D[:, -1] = 0.0                               # the repair: l3[:, :256] and l3.bias not changed
        else:
            rows = OLD if m in ("ih", "hh") else np.arange(256); D[rows] = away(D[rows], Vs[m])   # rows 256.. of lm and l2 free
        Wn[MATS[m][0]] += D[:, :-1]; Wn[MATS[m][1]] += D[:, -1]
    return Wn
for only in (("l3",), ("ih",), ("hh", "lm", "l2", "book")):
    for scale in (0.03, 4.0):
        Wn = changed(sd, scale, only, V); line = []
        for n, it in SETS.items():
            o = run(Wn, it, in_use)[0]; line.append(f"{n} {np.abs(o - ref[n][0]).max():.1e} ({int((o.argmax(-1) != ref[n][0].argmax(-1)).any(1).sum())} answers)")
        print(f"D. change of {'+'.join(only)}, entries N(0, {scale}^2) and then the rule: largest change of an output: " + "; ".join(line))

bt = unrelated_batches(SEED); feats = []             # E. checks 3(a) and 3(b) for l3 alone: AdamW's arithmetic (rate 0.001, betas 0.9 and 0.98, no decay)
for _ in range(N_UP):
    o, c, y = run(sd, next(bt), in_use); feats.append((one(c["l3"]), y))
W3 = np.concatenate([sd["l3.weight"], sd["l3.bias"][:, None]], 1); w3, E3 = spec["l3"]
Zh, Zp = one(ref["held-out pairs, 12 letters"][1]["l3"]), one(ref["2,000 items, no letters"][1]["l3"])
srt = np.sort(ref["2,000 items, no letters"][0], -1); margin = (srt[..., -1] - srt[..., -2]).min(1)
print(f"E. first family, 2,000 items without letters: share whose smallest gap between the two largest outputs is below 0.1, 0.3, 1.0: {(margin < .1).mean():.3f}, {(margin < .3).mean():.3f}, {(margin < 1).mean():.3f}; "
      f"the unrelated task's first batch at l3: 2-norm of a vector {np.median(np.linalg.norm(feats[0][0], axis=1)):.1f}, of its part outside the brief's directions {np.median(np.linalg.norm(away(feats[0][0], V['l3']), axis=1)):.2f}, "
      f"outside those kept at 1e-8 {np.median(np.linalg.norm(away(feats[0][0], E3[:, :int((w3 > 1e-8 * w3[0]).sum())]), axis=1)):.2f}")
for label, Vp in (("the brief's rule (1e-6)", V["l3"]), ("threshold 1e-8", E3[:, :int((w3 > 1e-8 * w3[0]).sum())]), ("projections off", None)):
    W = W3.copy(); mo = np.zeros_like(W); ve = np.zeros_like(W)
    for step, (X, y) in enumerate(feats, 1):
        lg = X @ W.T; lg -= lg.max(1, keepdims=True); p = np.exp(lg); p /= p.sum(1, keepdims=True); p[np.arange(len(y)), y] -= 1; G = p.T @ X / len(y)
        if Vp is not None: G = away(G, Vp)
        mo = .9 * mo + .1 * G; ve = .98 * ve + .02 * G * G; S = -1e-3 * (mo / (1 - .9 ** step)) / (np.sqrt(ve / (1 - .98 ** step)) + 1e-8)
        W = W + (away(S, Vp) if Vp is not None else S)
        if step in (8, N_UP):
            print(f"E. {step} updates of l3 alone on the unrelated stream, {label}: largest change of an output at an answer position: held-out panel "
                  f"{np.abs(Zh @ (W - W3).T).max():.1e}, 2,000 items without letters {np.abs(Zp @ (W - W3).T).max():.1e}")

D = np.zeros((768, 513)); gen = np.random.default_rng(5)                               # F. an allowed change of the old cells' rows of the input map
for b in (224, 225) + tuple(range(128, 136)): D[OLD] += np.outer(gen.choice([-1.0, 1.0], len(OLD)), rc[b] / (rc[b] @ rc[b]))
assert np.abs(D[OLD] @ V["ih"]).max() < 1e-9
Wn = {k: v.copy() for k, v in sd.items()}; Wn["rnn.weight_ih_l0"] += D[:, :-1]; Wn["rnn.bias_ih_l0"] += D[:, -1]
print(f"F. old cells' rows changed inside what the rule allows (largest entry {np.abs(D).max():.2f}): an old cell's gate input moves by {np.abs(codes[[224, 225]] @ D[OLD].T).mean():.2f} on the new task's code and query bytes "
      f"and by at most {np.abs(codes[in_use] @ D[OLD].T).max():.3f} on a byte in use; largest change of a first-family output {max(np.abs(run(Wn, it, in_use)[0] - ref[n][0]).max() for n, it in SETS.items()):.3f}")

ref2 = {n: run(sd, it, in_use, True) for n, it in SETS.items()}; V2 = {}                # G. the repair of defect 1
for m in ("ih", "hh", "lm", "l2"):
    X = one(ref2["the 2,080 items"][1][m]); w, E = np.linalg.eigh(X.T @ X / len(X)); V2[m] = E[:, w > 1e-12 * w.max()]
Wn = changed(sd, 4.0, ("book", "ih", "hh", "lm", "l2", "l3"), V2, True)
print(f"G. repair (added units of z2 keyed, all directions kept: {', '.join(f'{m} {V2[m].shape[1]}' for m in V2)}; l3[:, :256] and l3.bias fixed): start against the birth "
      f"{max(np.abs(ref2[n][0] - run(sd0, it)[0]).max() for n, it in SETS.items()):.1e}; after a change with entries N(0, 4^2), largest change of an output "
      f"{max(np.abs(run(Wn, it, in_use, True)[0] - ref2[n][0]).max() for n, it in SETS.items()):.1e}")
```
