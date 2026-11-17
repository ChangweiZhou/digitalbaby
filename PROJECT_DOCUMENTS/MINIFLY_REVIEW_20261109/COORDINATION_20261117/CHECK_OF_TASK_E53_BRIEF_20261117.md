# Check of the brief of E53, revision 1 (17 November 2026)

Written for the outside reviewer by an agent that has not seen the reviewer's reasoning. The brief is `TASK_E53_THE_NETWORK_WITH_AN_INTERFACE_AT_A_BOUNDARY_WITHOUT_DECAY_AND_FROM_A_LARGER_START_20261117.md` beside this file, at commit 4582f5fa (its SHA-256 begins `192c6bcb`; it was unchanged when this check ended). The work was read-only. Nothing was trained, torch is not installed, no code of another branch was executed, and the files of other branches were read with `git show`. The check began at 22:34 UTC and ended at END_TIME UTC, without a restart of the container. It is the check that the brief's status line names. It is not the independent audit that the programme's rules ask for. This check committed nothing; two early states of this file were taken into the reviewer's commits 4d1c0380 and 33bc9ffd by the staging of those commits.

**Shell names.** `D=origin/codex/minifly-consolidated-20261023` (at 7cabb13a); `W7=origin/claude/e47-worker-stage-0-3d0ya6` (at 578a877b); `W46=origin/claude/wonderful-edison-8hx6i6` (E46's worker's branch); `M=MOD97_LOCAL_RULE_20261106`; `R=PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109`. "Brief, line n" cites a line of the brief. "Runner, line n" cites a line of `$W7:$M/exp47_local.py` (SHA-256 `8d80a7dc…a427c`). `APP <part>` is the script of the appendix, run from the repository root as `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -I <the script> <part>`.

**Terms.** As in the brief. The *trial* is the two runs of E47's arm `one` on seed 73129. The *size* of a weight is its absolute value. A *fall* is a plain score minus the score under a diagnostic, in points of a percentage. A run *is read* if its levels of A and of R are both at or above 80 (a term introduced in the repair of defect 2).

## Verdict

Release after the listed changes; do not release as it stands.

Every figure of "What the record says" was recomputed and is right, apart from a rounding and one figure of cost (defect 13). The starting weights can be changed as the brief describes without touching anything else in the runner. The defects are of three kinds.

- **The design would support a wrong conclusion** (defects 1, 2, 4, 5). The half of the explanation that concerns the decay is contradicted, for the one arm that is run, by the trial's own receipts; the arm that item 9 of the plan names (the changed start with the decay) is left out on an argument that the same receipts contradict; the rule of refutation calls the explanation refuted when a run has not learned the family at all; and the start times 3 changes what the interface cells compute among themselves, so that a failure to learn can be mistaken for a statement about far cells.
- **The runner and the checks cannot be carried out as written** (defects 3, 6, 7). Taken at its word ("these changes and no others"), the copy of `run` stops at an assertion, and with the most natural repair of that assertion it trains the network of the arm `place` under the name `wd0`. Checks 1 and 3 name runs that the runner as described cannot make; checks 5 and 6 cannot fail.
- **Readings and rules that are incomplete** (defects 8 to 12, 14).

Each defect below has a replacement text. The replacements of the sections "Runs", "The runner", "Checks" and "How the result is stated" are given once, in full, under defects 1, 3, 6 and 2; the other defects refer to them.

## Defects, the most serious first

### 1. The decay is named as a cause against the record of the arm that is run, and the arm that the plan names is left out on an argument that the receipts contradict

**What is wrong.**

1. Brief, line 20: "the standing weight decay shrinks every weight that learning does not hold up". Brief, line 77: the cells stay out "because the decay shrinks what is not used". The trial's receipts say otherwise for the arm `one`, the only arm of this job. The receipt's entry `magnitudes.far` (runner, lines 370 to 372: the mean absolute value of the 33 weights into each far cell) is, by `APP receipts`:

   | Rule | Update 0 | 1,024 | 2,048 | 4,096 | 8,192 | 12,288 | 16,384 | 24,576 |
   |---|---|---|---|---|---|---|---|---|
   | Sweep | 0.1503 | 0.1393 | 0.1320 | 0.1410 | 0.1614 | 0.1603 | 0.1578 | 0.1497 |
   | Back-propagation | 0.1503 | 0.1569 | 0.1806 | 0.2183 | 0.2481 | 0.2471 | 0.2449 | 0.2426 |

   Decay alone would leave 0.368 of a weight after 4,000 updates and 0.0021 after 24,576. By ring (`$R/E47_TRIAL_VERIFICATION_20261117.md`, section 6, table of weights) the cells of rings 2 and 3, which take no part in any answer, end at 0.132 and 0.145 under the sweep and at 0.257 and 0.234 under back-propagation, from 0.151 and 0.150. Every far cell's weights receive a gradient under both rules (receipt, `cosine_first_batch_start` and `cosine_first_batch_end`: `far_nonzero_share_S` and `far_nonzero_share_B` are 1.0). The loss of unused weights that the brief has in mind occurred in the arm `place` (ring 3: 0.001 and 0.006), and the brief does not run that arm. The brief's own line 30 gives 0.150 to 0.150 and 0.150 to 0.243 and draws no consequence.
2. Brief, line 45: "Over 4,000 updates the decay alone multiplies a weight by 0.37, so that the multiple would be gone before the family is learned, and the run would repeat the trial." The arithmetic is right (0.99975 to the power 4,000 is 0.3678). The inference is not a record: at update 4,096 the far cells' weights stood at 0.94 (sweep) and 1.45 (back-propagation) of their starting size, not at 0.36. Whether a start three times larger survives the decay is unknown; it is an outcome of the omitted run, not a reason against it.
3. `PLAN.md`, item 9: "E53: the local networks from a start at which differences pass, with and without decay". The brief has the changed start without decay only. The arm with the changed start and the decay of 0.25 is the one arm that would differ from the trial in the start alone, and the trial is its exact reference.
4. With three of the four cells, the outcome "rings 2 and 3 used in `x3-wd0`, not used in `wd0`" cannot say whether the start alone is enough (defect 2).

**Repair.** (a) Replace the second paragraph of "The question" (brief, line 20, from "Two things could keep" to the end of the bold question) by:

> Two settings could prevent those cells from taking part, and neither was varied. The first is the start: at E47's starting weights a difference between two items hardly reaches them. The second is the weight decay of 0.25. The record supports the first and not the second. In the trial's arm `one` the decay did not shrink the weights of these cells (see "The size of the weights"); it did so in the arm `place`, which this job does not run. The decay is varied all the same, because it has had one value in every run of a local network (rule 1 of section 4 of the plan) and because it is not known whether a start three times larger survives it. **Do the cells beyond the first ring come into use, and is the family learned in fewer updates, when the network is trained from a start at which a difference between items reaches the third ring at a seventh of its size at the interface (a twentieth in items without letters), with the trial's weight decay and without it? Does removing the decay alone change either?**

(b) Replace the bullet "The size of the weights" (brief, line 30) by:

> - **The size of the weights.** The receipt's entry `magnitudes` is the mean absolute value of the weights into the cells of a group: for each cell its 33 recurrent weights (three gates; the weight on its own state and the ten on its sources). It is 0.152 for interface cells and 0.150 for far cells at the start. At the end it is 0.198 and 0.150 under the sweep and 0.179 and 0.243 under back-propagation. By ring (the verification, section 6) the far cells end at 0.178, 0.132 and 0.145 under the sweep and at 0.234, 0.257 and 0.234 under back-propagation, from 0.150. Weight decay alone would have left 0.37 of a weight after 4,000 updates and 0.002 after 24,576; the far cells' mean stood at 0.94 (sweep) and 1.45 (back-propagation) of its starting value at update 4,096. Every far cell's weights receive a gradient under both rules. In this arm, therefore, the decay did not shrink the weights of the cells that take no part. In the arm `place` it did: ring 3 ends at 0.001 (sweep) and 0.006 (back-propagation).

(c) Replace the section "Runs" (brief, lines 35 to 47) by:

> | Arm | Start | Weight decay | Rules | Updates | Runs |
> |---|---|---|---|---|---|
> | `one` (the trial; on file) | E47's | 0.25 | S, B | 24,576 | 0 |
> | `x3` | E47's with every entry of `Wc` multiplied by 3 | 0.25 | S, B | 24,576 | 2 |
> | `wd0` | E47's | 0 | S, B | 24,576 | 2 |
> | `x3-wd0` | E47's with every entry of `Wc` multiplied by 3 | 0 | S, B | 24,576 | 2 |
> | `xf3`, `xf3-wd0` (conditional) | E47's with every entry of `Wc` multiplied by 3 except the weights of an interface cell on its own state and on other interface cells | 0.25 or 0 | B | 24,576 | 0 or 1 |
>
> The seed is 73129. "Weight decay 0" sets to 0 the decay of the optimizer's first group, which holds thirteen tensors: the weights and biases of `l1`, of the input map, of `lm`, of `l2` and of `l3`, and `Wc`, `bc` and `bfar`. The byte table has no decay in any arm. An arm without decay therefore differs from its partner in the decay of the input map, of `l1` and of the readout's three maps as well as in that of the recurrent weights. A difference between the two is the effect of removing all decay, as in E46, and cannot be assigned to the far cells' weights. Everything else is the trial's: the stream, the panels, the two diagnostics at every row, the rows every 256 updates.
>
> The arm `x3` is the only one that differs from the trial in the start alone. Whether the multiple persists under the decay is one of its outcomes; the receipt records the size of the weights of every ring at every row.
>
> The multiple is applied to every cell. The arms `x3` and `x3-wd0` therefore also change what the interface cells compute among themselves (see "What the multiple does to the interface cells"), and, because an update of the optimizer has a fixed absolute size, the recurrent weights change a third as fast in proportion to their size. A run of these arms that does not learn the family, or that needs more updates, cannot be read as a statement about far cells. For that case one conditional run is made. If, under back-propagation, `x3` or `x3-wd0` is *not read* (see "How the result is stated") or needs *more* updates to 80 on A than the trial, the start `xf3` is run under back-propagation with the decay of that arm (of `x3` if both qualify). It is not run under the sweep, whose single scale would be 1.9 times the mean absolute weight into an interface cell at this start.
>
> A run cost 3.65 CPU-hours in the trial. The six runs cost 21.9 CPU-hours, the conditional run 3.65 and the checks about 1.1. If the cost has to stay at four runs, the arm to leave out is `wd0` and not `x3`: the record already shows that the decay does not shrink these cells' weights in this arm, and item 9 of the plan asks for the changed start with and without decay.

(d) In the status line (brief, line 3) replace "One seed, four runs, about 15 CPU-hours." by "The job has one seed and six runs and costs about 23 CPU-hours (27 with the conditional run)."

### 2. The rule of refutation gives a wrong conclusion in several outcomes

**What is wrong.** Brief, line 77: "It is refuted, for this seed, if rings 2 and 3 are not used in `x3-wd0` under either rule. If they are used in some run, the report says in which".

1. *A run that has not learned the family counts as "not used".* Brief, line 74, defines *used* by a fall of 5 points or more. If a run of `x3-wd0` ends with A and R near zero, no fall of 5 points can occur, rings 2 and 3 are "not used", and the explanation is "refuted". The brief's own predictions give this run a chance of 40% (back-propagation) and 55% (sweep) of not reaching 80 on A (P53.4, P53.5). The most likely outcome under the sweep is thus read as a refutation, although it shows only that the network did not learn from this start.
2. *"Under either rule"* can mean under one of the two or under both. The sentence is read here as "under neither rule are they used"; a worker may read it the other way.
3. *Outcomes that the wording does not cover.* (a) Used in `x3-wd0` and not in `wd0`: the brief cannot say whether the start alone would do, because the arm with the changed start and the decay is missing (defect 1). (b) Used in `wd0`: removing the decay is enough, and the start is not shown to be needed; if at the same time `x3-wd0` is not used, the rule calls the explanation refuted although one of its two causes was just shown to act. (c) Used under one rule and not under the other.
4. *The word "refuted" contradicts the brief's own section "What it cannot show"* (line 95: "a run in which they stay unused shows that learning does not draw them in by itself, and not that they could not be used") and the result that the verification names as the refuting one (`$R/E47_TRIAL_VERIFICATION_20261117.md`, section 6: "a network started so that differences between items do reach rings 2 and 3 which still leaves them unused on a task that the interface alone cannot hold"). The first family can be held by the interface alone.

**Repair.** Replace the section "How the result is stated" (brief, lines 68 to 77) by:

> For each rule, each new arm is compared with the trial's `one`; `x3-wd0` is also compared with `x3` and with `wd0`. In every comparison the arm named second is the reference and the other the candidate.
>
> - **Learning.** Updates to 80 on A and on R, in the words of `describe` in `$W7:$M/e47_trial_read.py` (a copy of E42's `e42_tables.describe`). With q the candidate's updates divided by the reference's: *fewer* if q is below 2/3, *more* if q is above 1.5, otherwise *about the same*. If only the reference reaches 80: *more* if 24,064 divided by the reference's updates is at least 1.5, otherwise *not decided*. If only the candidate reaches 80: *fewer* if its updates divided by 24,064 are at most 2/3, otherwise *not decided*. If neither reaches 80: *neither reaches*. The levels of A, of R and of the length test at 40 letters are given beside these words. A run *is read* if its levels of A and of R are both at or above 80. A run that is *not read* is reported with its curves, and none of the statements below about far cells is made for it.
> - **The far cells as a whole.** The two diagnostics at the last row, in the words of E47's brief, the first form that applies: *used* if A or R falls by 15 points or more with the far cells at their mean; *not used* if both fall by less than 5 with the far cells at zero; *a constant is used* if either falls by 15 or more at zero and both by less than 5 at the mean; otherwise *partly*.
> - **Rings 2 and 3.** [the text given under defect 8]
> - **The size of the weights** by ring at the first row, at update 4,096 and at the last row.
>
> **The explanation under test** is that the cells beyond the first ring take no part because, from E47's start, differences between items do not reach them. The decay is tested as a second possible cause, for which the trial gives no support in this arm. The statement is made for each rule apart.
>
> - The explanation is *supported* for a rule if rings 2 and 3 are *used* in `x3` or in `x3-wd0` and *not used* in `wd0`. The report says which: in `x3`, the start alone is enough; in `x3-wd0` and not in `x3`, the start is enough only without the decay.
> - It is *not supported* for a rule if `x3` and `x3-wd0` are both read and rings 2 and 3 are *not used* in both. This means that, with differences reaching these cells, learning this family did not bring them into use on this seed. It does not mean that they cannot be used, because 256 cells are enough for the family (see "What it cannot show").
> - If rings 2 and 3 are *used* in `wd0`, removing the decay is enough at E47's start. The report says so, and the start is then not shown to be needed.
> - In every other case the explanation is *not decided* for that rule: in particular if `x3` or `x3-wd0` is not read, or if a reading of rings 2 and 3 is *not decided*.
>
> For every run that is read, the report says whether it learns the family in *fewer*, *about the same* or *more* updates than the trial.

### 3. The runner cannot be built from the five changes; taken at its word the copy stops, and with the nearest repair it trains another network

**What is wrong.** Brief, line 51: the new file "holds a copy of the text of `exp47_local.run` with the following changes and no others". The five changes do not cover the following lines of `run` and of the functions it calls.

| Runner, line | Text | What happens with an arm named `wd0` or `x3-wd0` |
|---|---|---|
| 434 | `m, L = build(side, arm, seed)` | `build` (line 209) passes the arm's name to `Local(m, seed, side, N, arm)`, and `e47_local.draws` (its line 47: `if periods != "one" and far`) gives every name other than `one` the periods 2, 4, 8 and 16 by place. The network built is that of the arm `place`. |
| 438 | `assert cnt["weights"] == WEIGHT_COUNT[(side, arm)]` | `KeyError`. A worker who adds the two keys with the trial's 965,905 gets an assertion error (the network has 971,665 weights); one who adds them from `L.counts` trains the wrong network without an error. |
| 404 to 406 | `receipt_path` returns `e47_{side}_{arm}_…` | A function outside `run`; change 3 gives the new name and not where it is made. |
| 234 to 238 | `opt_record` writes `weight_decay=C26.WD` | The receipt's `optimizer` and `optimizer_read_back` (lines 451 and 523) carry 0.25 at the top level in every arm, beside groups that read 0.0. Change 2 speaks of the groups only. |
| 446 to 449 | `experiment="E47"`, `start="Local(m, seed, side, 10, arm): …"` | The text of `start` describes E47's draw in the arms with the multiple. |
| 534 to 541 | `provenance` hashes `exp47_local.py` as the runner | Covered by change 3. |
| 890 to 898 | `main` asserts `arm in ARMS`; `OUT = X.OUT / "e47"` (line 56) | No command line for the new arms, for the folder, for `--resume`, or for a shorter run; none is described. |
| 871 to 883 | `check8` runs `run(32, "place", "S", …)` | Check 4 of the brief needs its own function; the brief describes none for any check. |
| 525 | `os.environ.get("E47_BESIDE", …)`; lines 478, 494, 529 print `e47` | Names only. |

Further: (a) `L.Wc` is a parameter that requires a gradient, so that `L.Wc.mul_(3)` raises an error outside `torch.no_grad()`; change 1 does not say so. (b) Check 1 asks for a run "with E47's start and decay 0.25" through the new file; the runner as described has no such arm. (c) The second check of the brief compares every tensor of the starting file with the trial's and would catch the wrong network, because the periods differ; but it is not said from which run its starting files come, nor that a run stops when it fails (defect 6).

What does work as the brief describes it was verified and is listed under "Verified and found sound", section B.

**Repair.** Replace the section "The runner" (brief, lines 49 to 57) by:

> A new file `exp53_local_start.py` in `$M`; no existing source is edited. `exp47_local.py` and `e47_local.py` are copied byte for byte from `$W7` into `$M` (SHA-256 `8d80a7dc33cc68fae6b2eb02b685ba0888ea0b9556335a5745f85a2fea4a427c` and `031adf6592a13bffd4dfe88ded12b4109a12d8bb9e891db61da865e21176acd0`; the second equals `$D:PROJECT_DOCUMENTS/COORD_CHECKS_20261116/e47_local_spec.py`), with their SHA-256 and source recorded in a text file beside them. Every other module that the runner imports is on the default branch with the content the trial used.
>
> The new file imports `exp47_local` and uses its functions unchanged (`build`, `install`, `make_opt`, `forward`, `step`, `score_row_all`, `magnitudes`, `cosines`, `weights`, `save_weights`, `save_ckpt`, `tsha`). It defines its own table of arms, `receipt_path`, `opt_record`, `provenance`, `run`, one function for each check, and `main`. Its `run` is a copy of the text of `exp47_local.run`. The report shows the difference between the two texts, and that difference consists of changes 1 to 8 and nothing else.
>
> The arms are, as (multiple, entries multiplied, weight decay): `one` (1, none, 0.25), used by check 1 only; `x3` (3, all, 0.25); `wd0` (1, none, 0); `x3-wd0` (3, all, 0); `xf3` (3, all but the interface cells' weights among themselves, 0.25) and `xf3-wd0` (the same, 0), for the conditional run.
>
> 1. **The network is always E47's arm `one`.** `build` is called as `build(32, "one", seed)`, and the weight count is asserted against `WEIGHT_COUNT[(32, "one")]` (965,905). The name of an E53 arm is never passed to `build`: `e47_local.draws` gives every arm name other than `one` the periods of the arm `place`. Directly after `build` the run asserts that `R_.tensors_sha(weights(m, L))` is `62de40fb57d41d280b23d3f35c2e3943e13d59a18b183cff48d6f6acc897ef33` and that `tsha(L.idx)`, `tsha(L.p)` and `tsha(L.f)` are `f414eb354435af94dd1489e0f6414a6c1e92bae55bc5ff6fb03c79370acdb63b`, `9aa02009b29c3933c0e1277e60e997e88c80595c4a06ef905fe5e34ac89ec0de` and `57a6e9770cd2bd3d4039e4b3c41aacd8e5e1f694c7b15f3075ff3f5d030404fd` (the trial's `start_tensors_sha256` and `structure_sha256`).
> 2. **The start.** After that assertion and before `make_opt`, under `torch.no_grad()`, the entries named by the arm are multiplied in place: `L.Wc.mul_(3)` for `x3` and `x3-wd0`. For `xf3` and `xf3-wd0`: `far = T.zeros(L.C, dtype=T.bool); far[L.farx] = True; mask = far[L.idx]; mask[L.farx] = True; L.Wc[:, mask] *= 3`, with the assertion `int(mask.sum()) == 8975` (of 11,264 entries a gate; column 0 of `L.idx` is the cell itself). `L.Wc` is not replaced by a new tensor. The starting file is written where the trial's text writes it, after `make_opt`, so that it holds the start as used. On a resume the multiplication is repeated and then overwritten by the saved state; check 4 covers this.
> 3. **The decay.** Directly after `make_opt` the one group whose `weight_decay` equals `C26.WD` is selected (the run asserts that there is exactly one) and its `weight_decay` is set to the arm's decay. The run then asserts that the two groups read back have the arm's decay and 0.0, and repeats the assertion after `opt.load_state_dict` on a resume. `opt_record` of the new file is that of `exp47_local` with the top-level entry `weight_decay` set to the arm's decay.
> 4. **Names.** `receipt_path` gives `e53_32_<arm>_<rule>_73129.json`, with `_dry<N>` before the seed for a run of N updates other than 24,576. The folder of the runs is `$M/exp45/e53/`, that of the checks `$M/exp45/e53_qual/`. Printed lines begin with `e53`. `processes_beside` is read from the environment variable `E53_BESIDE`.
> 5. **The receipt** has `experiment="E53"`, the arm's name, `network_arm="one"`, the multiple, the entries multiplied (as text and as a count), the decay, and for `start` the text "E47's start (Local(m, seed, 32, 10, 'one')) with the named entries of Wc multiplied in single precision". `provenance` adds the hash of `exp53_local_start.py`.
> 6. **Measures at every row,** beside `magnitudes`. (a) `ring_magnitudes`: the mean absolute value of `Wc[:, cells]`, as `magnitudes` computes it, for the interface cells and for each of the three rings. The ring of cell i is `-(-e47_local.distance(i, 32) // e47_local.RADIUS)`: the grid distance to the patch, divided by 3 and rounded up. The run asserts the counts 256, 228, 300 and 240 for rings 0 to 3. (b) `gates`: on the batch `R42.first_batch(73129)` (64 items; every position of `x`, the padding at the end of shorter items included), for the interface cells and for each ring, the share of the values of the reset gate and of the update gate that lie below 0.05 or above 0.95, and the mean absolute state. The gates are recomputed from the states that `forward(m, L, x, False)` returns: with h the state before a position (zero before the first), G the input sum of the position as lines 127 to 148 of `exp47_local.forward` build it, and H = `einsum("bik,gik->bgi", h[:, L.idx], L.Wc) + L.bc`, the gates are r = sigmoid(G[:, 0] + H[:, 0]), z = sigmoid(G[:, 1] + H[:, 1]) and n = tanh(G[:, 2] + r H[:, 2]). The function asserts that (1 − z) n + z h equals the state that `forward` returned, at every position, to 1e-5.
> 7. **The weights of the last three rows.** After the scoring of updates 24,064 and 24,320 the weights are saved with `save_weights` as `<receipt's stem>_u24064.pt` and `<receipt's stem>_u24320.pt`; the receipt lists their names and hashes.
> 8. **`main`.** `python3 exp53_local_start.py run <arm> <S or B> 73129 [--dose N] [--out DIR] [--resume]` and `python3 exp53_local_start.py check <1 to 6>`, from the folder `$M`, with the interpreter of the virtual environment.
>
> Saving and resuming are the trial's (a state every 1,024 updates). For each run the receipt, the starting weights, the final weights, the weights of updates 24,064 and 24,320, the answers file and the log are committed.

The lines of change 2 for `xf3` and the recomputation of change 6 were written from the text of the runner and were not executed in this check, because torch is not installed; checks 2 and 5 of the replacement under defect 6 test them.

### 4. The start times 3 changes what the interface cells compute among themselves; a failure to learn could be read as a statement about far cells

**What is wrong.** Brief, lines 41 and 53: every entry of `Wc` is multiplied, for every cell, all three gates and the cell's weight on its own state. The interface cells must still learn the family, and their own computation is changed at the start. `APP starts` (the functions of `$R/signal_passage.py`, items of 12 letters, first digit changed):

| Start | Share that reaches ring 3 at 0, 4, 12 letters | Way back | Factor a position (10 to 20, 20 to 40 after) | Mean absolute state of an interface cell | The interface's own difference 5, 10, 17 positions after the changed byte |
|---|---|---|---|---|---|
| E47's | 0.0023, 0.0027, 0.0028 | 0.0026 | 0.651, 0.660 | 0.178 | 0.0145, 0.0014, 0.00006 |
| every entry times 3 (the brief's) | 0.0525, 0.0916, 0.141 | 0.149 | 0.951, 0.959 | 0.213 | 0.0727, 0.0428, 0.0267 |
| ten connections times 3, own weight unchanged, every cell (part `more` of `signal_passage.py`) | 0.0518, 0.0805, 0.105 | 0.108 | 0.900, 0.930 | 0.203 | not measured |
| the far cells' 33 weights times 3, the interface cells' unchanged | 0.0822, 0.141, 0.206 | 0.037 | 0.948, 0.982 | 0.178 | 0.0152, 0.0031, 0.0015 |
| every entry times 3 except an interface cell's weights on itself and on interface cells | 0.0811, 0.139, 0.202 | 0.113 | 0.948, 0.975 | 0.180 | 0.0207, 0.0090, 0.0048 |
| the last two at 3.5 | 0.145 to 0.552; 0.140 to 0.514 | 0.053; 0.189 | a difference grows | 0.178; 0.182 | |

1. With the brief's start a difference that the interface cells hold among themselves is 5, 30 and 445 times larger 5, 10 and 17 positions after the changed byte than at E47's start, and the interface cells' states are a fifth larger. These cells learn the family from another start than in the trial, whatever the far cells do.
2. The optimizer's update has a fixed absolute size, so that weights three times larger change a third as fast in proportion. This too slows the interface cells' learning in the arms with the multiple.
3. `signal_passage.py` has a variant that multiplies only the weights on other cells ("ten connections", every cell: 0.105 to ring 3 at a multiple of 3, 0.204 at 3.5 without growth). It has no variant that multiplies the far cells' weights only. The two measured here with its functions separate the two effects: with every entry multiplied by 3 except the interface cells' weights among themselves, ring 3 holds 0.202 of the interface's difference at 12 letters (0.081 without letters), the way back is 0.113, a difference does not grow, and the interface cells' states and their own passage are close to E47's.
4. Under the sweep such a start cannot be read at present: the one scale would be 0.390, which is 1.88 times the mean absolute weight into an interface cell and 0.87 times that into a far cell (`APP scale`). It needs the scale for each cell, item 18 of the plan.

**Repair.** The conditional arm and its reason are in the replacement of "Runs" under defect 1, and its two lines of code in the replacement of "The runner" under defect 3. Add to "What the record says", after the bullet on the passage at the start:

> - **What the multiple does to the interface cells** (the script in the appendix of `CHECK_OF_TASK_E53_BRIEF_20261117.md`, part `starts`). The multiple is applied to every cell. At the start times 3 the mean absolute state of an interface cell is 0.213 against 0.178, and the difference that the interface cells hold 5, 10 and 17 positions after the changed byte is 0.073, 0.043 and 0.027 against 0.015, 0.0014 and 0.00006. A start that multiplies by 3 every recurrent weight except those of an interface cell on its own state and on other interface cells leaves these figures close to E47's (0.180; 0.021, 0.0090, 0.0048), brings 0.202 of a difference to ring 3 at 12 letters (0.081 without letters) and 0.113 back, and does not make a difference grow (0.948 and 0.975 a position). Under the sweep this start cannot be read before item 18 of the plan, because the single scale (0.390) would be 1.9 times the mean absolute weight into an interface cell.

### 5. The arms without decay remove the decay from the whole network, and the record on that is not cited

**What is wrong.** Brief, line 43: the decay is set to 0 in the optimizer's first group. That group holds thirteen tensors (trial's receipt, `optimizer.groups[0].params`: `l1.weight`, `l1.bias`, `rnn.weight_ih_l0`, `rnn.bias_ih_l0`, `lm.weight`, `lm.bias`, `l2.weight`, `l2.bias`, `l3.weight`, `l3.bias`, `Wc`, `bc`, `bfar`). The arm `wd0` therefore differs from the trial in the decay of the input map and of the readout as well as of the recurrent weights, while the explanation under test concerns the far cells' weights alone. The brief does not say this, and it does not cite the one experiment of the record that removed the decay of the same group, E46 (`$W46:$M/E46_REPORT.md`, marked "not audited" in the inventory). `APP e46`, from E46's receipts of seed 73129:

| Dense network, seed 73129 | Updates to 80 on A, decay 0.25 | Decay 0 | Norm of the recurrent matrix at the end, decay 0.25 | Decay 0 |
|---|---|---|---|---|
| Sweep | 5,632 | 5,632 | 28.5 | 72.0 |
| Back-propagation | 5,120 | 7,680 | 26.3 | 71.3 |

Without decay the input map, `lm`, `l2` and `l3` end 1.6 to 5.5 times larger than with it. Under back-propagation the ratio of updates is exactly 1.50, the edge of *about the same*; this bears on P53.3 (defect 11).

**Repair.** The sentence on what "weight decay 0" changes is in the replacement of "Runs" under defect 1. Add to "What the record says":

> - **The dense network without decay** (E46; `origin/claude/wonderful-edison-8hx6i6:$M/E46_REPORT.md`, not audited). With the decay of the same group set to 0 the dense network learns the family under both rules on three seeds. On seed 73129 A reaches 80 at 5,632 updates under the sweep (5,632 with the decay) and at 7,680 under back-propagation (5,120 with the decay). Without decay the recurrent matrix ends with 2.5 to 2.7 times the norm it has with the decay, and the input map and the readout's maps with 1.6 to 5.5 times.

Replace brief, line 33, by: "No run of this network without decay or from another start exists."

### 6. The checks: two cannot be carried out as worded, two cannot fail, and none says what stops

**What is wrong.**

| Check | Can it fail when the thing it is named after is wrong? | Fault |
|---|---|---|
| 1 | Yes, if it is run through the new file. | The runner as described has no arm with E47's start and decay 0.25, so that the check can only be run through `exp47_local.py`, where it tests nothing new. `magnitudes` and the hashes of the start are not among the fields compared, although the trial's receipt records them at every row. Exact equality is achievable: the rows of the trial's lost runs, made on another machine image, equal those of the completed runs in every field but the times (`APP lost`: 49 and 33 rows, 0 differing). |
| 2 | Yes; it would also catch the network of the arm `place` (defect 3), because the starting file holds `periods` and `phases`. | A starting file is written at the start of a run, and the check stands under "before the runs": it is not said which run writes the files compared, nor that a run of the roster stops if its own starting file fails. |
| 3 | First half yes. Second half yes in substance: the difference after one update is 0.00025 times the start in exact arithmetic, because the optimizer multiplies a weight by 1 − 0.001 × 0.25 and then makes a step that does not depend on the decay; rounding adds at most about 1e-7, and E46's check 2 found 2.6e-8 or less. If the decay were not removed the deviation would be 1.1e-5 to 8.8e-5 in each of the thirteen tensors (`APP tensors`), so that the tolerance of 1e-6 can both hold and fail. | "The trial's dry run of check 1 after its first update" does not exist: a dry run of 1,024 updates saves weights at its start and at its end only (runner, lines 511 and 516). The sign of the difference is not given. The byte table, which must be bit-equal, is not mentioned. |
| 4 | Yes. | It needs a function of its own (runner, line 876, runs the arm `place`), which the brief does not provide for. It does not say that the decay read back after the resume is compared. |
| 5 | No: "the loss is finite, and the share … is recorded". | No criterion for the gates. "Gate values beyond 0.05 and 0.95" is not defined (defect 7). |
| 6 | No. | No criterion and no consequence. |

No sentence says that the runs wait for the checks or what the worker does when one fails. The folder of the checks' receipts is not named.

**Repair.** Replace the section "Checks before the runs" (brief, lines 59 to 66) by:

> Each check is a function of `exp53_local_start.py` and writes `$M/exp45/e53_qual/check<n>.json` with an entry `pass`. No run of the roster is started before checks 1 to 5 have passed. If a check fails, the worker stops, reports the first difference found and asks. The trial's files are read with `git show $W7:<path>` into a folder that is not committed.
>
> 1. **With multiple 1 and decay 0.25 the new `run` is the trial's.** The arm `one` of the new file with `--dose 1024`, under each rule. In each of the five rows (updates 0, 256, 512, 768 and 1,024) these fields equal those of `$W7:$M/exp45/e47/e47_32_one_<rule>_73129.json` exactly: `loss`, `A_counts`, `R_counts`, `taught_counts`, `length_counts`, the counts in `far_off` and in `far_mean`, `magnitudes` and `data_digest`. `start_tensors_sha256`, `structure_sha256` and `cosine_first_batch_start` of the receipt equal the trial's. (The rows of the trial's lost runs, made on another machine image of this host, equal those of its completed runs exactly, so that exact equality is expected.)
> 2. **The starts.** One run of 256 updates of each of the arms `x3`, `wd0`, `x3-wd0` and `xf3` under the sweep. For `wd0` every tensor of the starting file is bit-equal (`torch.equal`) to the one in `$W7:$M/exp45/e47/e47_32_one_S_73129_start.pt`, the five tensors of whole numbers (`sources`, `periods`, `phases`, `interface`, `far`) included. For `x3` and `x3-wd0` every tensor but `Wc` is bit-equal, and `Wc` is bit-equal to the trial's `Wc` multiplied by 3 in single precision. For `xf3` every tensor but `Wc` is bit-equal; 26,925 entries of `Wc` equal the trial's multiplied by 3 and 6,867 equal the trial's, and the 6,867 are exactly the entries [g, i, k] in which cell i and cell `sources[i, k]` are both interface cells. The assertion of change 1 repeats the comparison of the unmultiplied start in every run of the roster.
> 3. **The decay.** (a) In the runs of check 2 the groups read back from the optimizer at the start and at the end have the decays 0.25 and 0 in `x3` and `xf3`, and 0 and 0 in `wd0` and `x3-wd0`. (b) From E47's start, one update on the batch `R42.first_batch(73129)`, once with decay 0.25 and once with decay 0, under each rule: `book.weight` is bit-equal between the two, and for each of the thirteen tensors of the first group the largest absolute value of p(decay 0) − p(decay 0.25) − 0.00025 p(start) is below 1e-6. (If the decay were not removed, this value would lie between 1.1e-5 and 8.8e-5 in every one of the thirteen; E46's check 2 found 2.6e-8 or less.)
> 4. **A resumed run equals an uninterrupted one** (the trial's check 8), for the arm `x3-wd0` under the sweep with `--dose 2048`: once without interruption, and once stopped after the saved state of update 1,024 (`stop_at=1024`) and resumed by a second call. The two receipts are equal in every row and in every field whose name does not begin with `cpu` or `wall`, in `tensors_sha256`, `start_tensors_sha256`, `data_digest` and `cosine_first_batch_end`; the groups read back after the resume have the decay 0.
> 5. **The start of the arms with the multiple on the first batch,** for `x3` and `xf3` under each rule: the loss and the gradient of every tensor are finite, and the share of gate values below 0.05 or above 0.95 (change 6) is below 1% in each of the four groups of cells. (The reviewer's measurement in double precision on other items finds 0.00% at both starts.) The mean absolute state of each group is recorded.
> 6. **Cost** (no criterion): CPU-seconds an update and a scored row from the runs of check 2, and a forecast for the six runs. If the forecast exceeds 30 CPU-hours, the worker asks before starting.

### 7. The measures added at every row are not defined: the rings, the gates, the first batch

**What is wrong.**

1. *The rings.* Brief, line 11: "A far cell's *ring* is the number of connections between it and the interface". The runner knows interface and far cells (`L.ifx`, `L.farx`) and no rings. The reviewer's rings (`$R/e47_trial_verify.py`, line 522: `ring = -(-S.dist // 3)`) are the grid distance to the patch divided by 3 and rounded up, which is the least number of connections that the geometry allows. Counted on the connections drawn, 103 of the 768 far cells are farther than their ring says (`APP tensors`: ring 1, 200 cells at one connection, 27 at two, 1 at three; ring 2, 243, 54 and 3 at two, three and four; ring 3, 222 and 18 at three and four). A worker who follows the brief's sentence and counts connections obtains other rings than the reviewer's.
2. *The gates.* Brief, line 56: "the share of gate values beyond 0.05 and 0.95 on the first batch". Which gates (reset and update, or the candidate as well), which cells, which positions (a batch pads shorter items to the longest), and whether "beyond" means outside the interval, are not said. `forward` returns states and no gates (runner, line 197), so that the gates have to be recomputed, and nothing ties the recomputation to the forward.
3. *The first batch at row 0.* In `run` the first batch exists only from the first update (runner, lines 501 to 504), and row 0 is scored before it (line 499).

**Repair.** Change 6 of the replacement of "The runner" under defect 3. In "Terms" (brief, line 11) replace the sentence on the ring by:

> A far cell's *ring* is its grid distance to the patch (the larger of its distances in rows and in columns, with the edges joined), divided by 3 and rounded up; it is the least number of connections by which a value can travel between the cell and the interface. Ring 1 has 228 cells, ring 2 has 300, ring 3 has 240. On the connections drawn, 103 far cells need one or two connections more than their ring.

### 8. The reading of rings 2 and 3 rests on one row and on a threshold that the same kind of quantity crosses from row to row

**What is wrong.** Brief, line 74: rings 2 and 3 "*are used* in a run if A or R falls by 5 points or more when they are held at their mean", measured on the committed weights, which are those of the last row only.

1. The measurement is defined from the committed files alone: the final weights hold the structure, the panels are rebuilt from the seed by the reader, and part F of `$R/e47_trial_verify.py` makes it (rerun here for the trial: with rings 2 and 3 at their mean A and R are 93.0 and 86.3 against 93.3 and 86.3 under the sweep, and unchanged under back-propagation; `F_RERUN_NOTE`). The script's loader has the trial's path fixed (`weights(arm, rule, start)`); the brief should say that the reviewer adapts the path and nothing else.
2. Against zero the threshold is defensible. In the trial the fall for rings 2 and 3 is 0.0 to 0.3 of a point on A and R. Five points are 30 items of A's 600 and 100 of R's 2,000.
3. Against the scatter between rows it is not safe near the line. `APP receipts`, last nine rows of the trial: A changes by up to 1.8 points between neighbouring rows and R by up to 3.9; the fall under the diagnostic that holds all far cells at their mean has a standard deviation of 0.8 to 1.8 points over those rows and, for R under the sweep, lies between 2.35 and 5.95, on both sides of 5. A fall of rings 2 and 3 between about 2 and 8 points would be called *used* or not according to the row at which the run happens to end.
4. A run that has not learned the family gives "not used" by construction (defect 2).

**Repair.** Change 7 of the replacement of "The runner" (the weights of updates 24,064 and 24,320 are saved). Replace the bullet "Rings 2 and 3" (brief, line 74) by:

> - **Rings 2 and 3.** The reviewer measures this from the committed weights of the last three rows (updates 24,064, 24,320 and 24,576), with the reader and part F of `$R/e47_trial_verify.py`, in which only the path of the weight files is changed. On the panel of A (600 items) and on that of R (2,000 items) the network is scored with the cells of rings 2 and 3 held, after every position, at their mean over the items of the scored set at that position, while all other cells run freely. The fall is the plain score minus this score, averaged over the three rows. In a run that is read, rings 2 and 3 are *used* if the fall of A or of R is 5 points or more, *not used* if both falls are below 2 points, and *not decided* otherwise. (In the trial the falls are between 0.0 and 0.3 of a point. Between neighbouring rows at the end of the trial's runs R changes by up to 3.9 points, and the fall of R with all far cells at their mean lay between 2.3 and 6.0 points over the last nine rows of the sweep's run; this is why three rows are averaged and why a fall between 2 and 5 points is not read.) The reviewer also reports the share of a difference that reaches each ring on the starting and on the final weights (`$R/signal_passage.py`), and confirms on the committed starting file of the arm `x3` the 0.141 of his measurement. The worker's report gives the tables of the first two points and leaves this one to the reviewer.

### 9. Nothing cheaper is asked for first, although the first rows of a run show whether it can be read

**What is wrong.** Each run costs 3.65 CPU-hours, and the brief predicts that `x3-wd0` fails to reach 80 on A with a chance of 40% and 55%. No existing weights can answer the question of the start, because no network of this form was trained from another start. The existing weights do answer the question of the decay for this arm (defect 1), which is why `wd0` is the arm to leave out if the cost binds. A shorter run cannot show whether the family is learned, but the first 4,096 to 8,192 updates of a run (0.6 to 1.2 CPU-hours) show three things that decide whether the rest can be read: whether the loss falls as in the trial (sweep 0.628 and 0.256 at updates 4,096 and 8,192, with A at 9.0 and 55.3; back-propagation 0.508 and 0.160, with A at 16.7 and 77.3), whether the multiple is still there (`ring_magnitudes`), and whether the gates have moved to their limits (`gates`). The brief asks for a message only when all runs have finished (line 110).

**Repair.** Add to "Runs":

> **Order and early report.** The two runs of `x3` and the run of `x3-wd0` under back-propagation are started first; the others follow as processes become free. When a run of an arm with the multiple has written the rows of updates 4,096 and 8,192, the worker sends the reviewer, beside the trial's figures of the same rows and rule (sweep: loss 0.628 and 0.256, A 9.0 and 55.3; back-propagation: loss 0.508 and 0.160, A 16.7 and 77.3), the run's loss, A, R, `ring_magnitudes` and `gates`. The run goes on meanwhile. A run whose loss is not finite is stopped and reported.

and in "Reporting" (brief, line 110) insert after "the checks passed (or which failed);": "the rows of updates 4,096 and 8,192 of each run of an arm with the multiple;".

### 10. What the answer decides, and what would make the far cells necessary

**What is wrong.** Brief, line 22: "The answer decides whether a later job may count on cells away from the interface (item 12 of the plan: a second task learned in them)." Only one of the two answers decides anything. The brief's own line 95 says that a run in which the far cells stay unused does not show that they could not be used, since the family is learned by 256 cells. Line 95 is adequate as a statement of the limit; line 22 and line 77 contradict it. A negative result of E53 leaves item 12 where it is.

Making the far cells necessary would be the direct test, and it cannot be had by a change of five lines. It would take one of the following, each a new specification with its own checks. (a) The cells that read the byte and the cells that the readout reads are placed in two patches at least one ring apart, so that every answer has to cross far cells: two sets of indices in the place of `L.ifx` at lines 146 and 193 of the runner's `forward` and at lines 133 and 149 of the specification's, hence a new specification file and E47's checks 3 and 4 again. (b) The patch is made smaller than the family needs: the constant `PATCH` of the specification file, to which the input map (768 by 512) and `lm` (512 by 256) are sized, and a reference run that shows the smaller patch failing alone. (c) A second task, which is item 12 itself. The job as it stands can go ahead if its negative result is worded as line 95 words it.

**Repair.** Replace brief, line 22, by:

> A result that these cells are *used* would let item 12 of the plan (a second task learned in cells away from the interface) begin from that start. A result that they are *not used* does not close item 12: the first family is learned by 256 cells, so that nothing in this job requires the outer cells, and item 12 is then itself the test, on a task that the interface cells cannot hold alone.

Replace the second bullet of "What it cannot show" (brief, line 95) by:

> - The job cannot show whether cells away from the interface would be used by a task that the interface cells cannot learn alone. The first family is learned by 256 cells, so that the far cells are not needed here; a run in which they stay unused shows that learning this family does not by itself bring them into use, and not that they could not be used. A network in which they are necessary needs another specification (the cells that read the byte apart from the cells that are read; or a smaller patch; or a second task, which is item 12) and is not part of this job.

Replace the fourth bullet (brief, line 97) by:

> - The job cannot show the sweep with a scale taken for each cell (item 18 of the plan). The scale is one mean over all recurrent weights, 0.151 at E47's start and 0.452 at the start times 3. Because every weight is multiplied alike, the two rules stand to each other at the start times 3 as at E47's start: with the mean gates of the start, an error is multiplied on its way back by about 0.95 a position under the sweep and by about 1.01 under the exact rule (0.65 and 0.67 at E47's start). The comparison of the rules in the arms `x3` and `x3-wd0` can be read on that footing for as long as the weights into interface cells and into far cells stay of one size; in the trial they ended at 0.198 and 0.150 under the sweep and at 0.179 and 0.243 under back-propagation. The receipt records the scale and the size of the weights of every ring at every row, and the cosine between the two rules' gradients on the first batch at the start and at the end (0.757 at E47's start, 0.621 for the far cells). The start `xf3` is not run under the sweep for this reason.

### 11. The predictions

**What is wrong.**

1. P53.1 ("Checks 1 to 5 pass") cannot be scored as the checks stand: check 5 has no criterion beyond a finite loss, and checks 1 and 3 name runs that do not exist (defect 6).
2. P53.6 and P53.7 are scored "not used" by construction for a run that has not learned the family (defect 2).
3. P53.8 ("No new run needs *fewer* updates") does not say whether *fewer* is the word of `describe` (a ratio below 2/3) or any smaller number, nor how a run that does not reach 80 counts.
4. P53.3 (70%: `wd0` *about the same* as `one` on A under both rules) does not contradict the record, but the record puts it at the edge: on this seed, in the dense network, removing the decay gave a ratio of exactly 1.50 under back-propagation (7,680 against 5,120; `APP e46`), and *more* begins above 1.5. The brief does not cite E46.
5. There is no prediction for the arm `x3`, which the repair of defect 1 adds, nor for whether the multiple survives the decay.

No prediction contradicts the record. P53.2 is supported by E46 (the dense network learns the family without decay on three seeds under both rules). P53.7 at 10% agrees with defect 1. The division of the scoring (P53.1 to P53.5 and P53.8 by the worker, P53.6 and P53.7 by the reviewer; brief, line 114) is clear.

**Repair.** Replace the table of predictions (brief, lines 81 to 90) by the following, the chances of P53.9 to P53.12 to be entered by the reviewer before release:

> | | Statement | Chance |
> |---|---|---|
> | P53.1 | Checks 1 to 5 pass at the first attempt | 85% |
> | P53.2 | `wd0` reaches 80 on A within the run under both rules | 85% |
> | P53.3 | `wd0` against `one`: updates to 80 on A are *about the same* under both rules | 70% |
> | P53.4 | `x3-wd0` reaches 80 on A within the run under back-propagation | 60% |
> | P53.5 | `x3-wd0` reaches 80 on A within the run under the sweep | 45% |
> | P53.6 | Rings 2 and 3 are *used* in `x3-wd0` under at least one rule (a run that is not read, or a reading *not decided*, counts as not *used*) | 35% |
> | P53.7 | Rings 2 and 3 are *used* in `wd0` under at least one rule (counted in the same way) | 10% |
> | P53.8 | No new run is described as *fewer* than the trial's run of its rule on updates to 80 on A (a run that does not reach 80 is not *fewer*) | 80% |
> | P53.9 | `x3` reaches 80 on A within the run under back-propagation | |
> | P53.10 | `x3` reaches 80 on A within the run under the sweep | |
> | P53.11 | Rings 2 and 3 are *used* in `x3` under at least one rule (counted as in P53.6) | |
> | P53.12 | In `x3` the mean absolute weight into a cell of ring 3 at update 4,096 is at least 0.30, twice its value at E47's start, under both rules | |

and in "Done" (brief, line 114) replace "the outcome of predictions P53.1 to P53.5 and P53.8 (P53.6 and P53.7 are scored by the reviewer)" by "the outcome of predictions P53.1 to P53.5, P53.8 to P53.10 and P53.12 (P53.6, P53.7 and P53.11 are scored by the reviewer)".

### 12. Rules for the worker: what a worker in a fresh container lacks

**What is wrong.**

1. *Weight files are ignored by git.* `$D:.gitignore` has the line `*.pt`. The brief asks for starting and final weights to be committed (line 57) and does not say that they have to be added with `git add -f`, nor that the saved state (`…_ckpt.pt`, also ignored) is not to be committed.
2. *An interruption before the first saved state.* The first state is saved after 1,024 updates, about seven minutes of CPU. If the machine stops earlier, the starting file exists and no state does; the run then refuses to start (runner, lines 428 to 430: `resume` is false without a state, and `refuse_existing` finds the starting file). The brief's line 102 says only "a run is resumed from its saved state".
3. *The flag `--resume` from the first start.* The trial's launch script passes `--resume` at every start (`$W7:$M/exp45/e47/launch_trial.sh`); without it a restarted command refuses the existing files. The brief does not say so.
4. *The folder.* The trial's commands were run from `$M` (the same script). The brief names no folder.
5. *"Commit and push the running receipts between the pieces"* (line 102): the receipt is written every 4,096 updates only (runner, lines 509 and 510); between these the log is the only file that changes. The logs and their folder are not named.
6. *`describe` is on a third branch.* Brief, line 72, sends the worker to `origin/claude/gallant-dijkstra-i9zgsr` for ten lines that `$W7:$M/e47_trial_read.py` holds as a copy (lines 31 to 39), together with the forms of E47's brief (`use_form`, lines 52 to 60). The branch `$W7` is fetched in any case.
7. *The time for which the turn has to be held* is not given: six runs in two sets of three take about eight hours of wall time at the trial's speed.

**Repair.** Replace the second bullet of "Rules for the worker" (brief, line 102) by:

> - **This host stops a machine about five minutes after its session's turn ends. The disk is kept, and detached processes are terminated.** While runs proceed, do not end your turn: issue a command that waits for the processes, and, since one command may run only for a limited time, repeat it until they have finished. Six runs in two sets of three take about eight hours. Start every run with `--resume` from the first time on, with its output appended to `$M/exp45/e53/logs/<receipt's stem>.log`; the flag has no effect while no saved state exists, and the same command restarts a run after an interruption. If a run was interrupted before its first saved state (update 1,024), move its starting file into `$M/exp45/e53/interrupted/` and start it again. Between two waiting commands commit and push the logs and the receipts (a receipt is rewritten every 4,096 updates). The report says which runs were resumed.

and add two bullets:

> - Run every command from the folder `$M`. Weight files (`*.pt`) are ignored by the repository's `.gitignore`: add the starting weights, the final weights and the weights of updates 24,064 and 24,320 with `git add -f`. Do not commit the saved states (`…_ckpt.pt`).
> - The words of the comparisons and of E47's forms are the functions `describe` and `use_form` of `$W7:$M/e47_trial_read.py`; copy them into the script that makes the report's tables and name the source.

### 13. Figures and statements about the record

| Brief, line | Text | Finding | Replacement |
|---|---|---|---|
| 30 | "0.151 for interface cells" | The receipt has 0.151506 (`APP receipts`; 0.152 in the verification's table). | "0.152 for interface cells" |
| 47 | "about 0.4 for the checks" | Check 4 alone is two runs of 2,048 updates; E47's check of the same kind cost 0.67 CPU-hours (`$W7:$M/exp45/e47_qual/check8.json`, `cpu` 2,428 s). At the trial's 0.40 s an update and 35 s a row, checks 1 and 4 of the brief cost 0.32 and 0.63 CPU-hours (`APP receipts`). | "about 1.0 for the checks" (1.1 with the replacement of defect 6) |
| 65 | "The reviewer's measurement finds none beyond these marks at this multiple." | The table of `signal_passage.py` prints 0.00%; on the items of 12 letters 0.002% of the interface cells' gate values lie beyond the marks (`APP starts`). | "The reviewer's measurement finds 0.00% beyond these marks at this multiple." |
| 28 | "Each run cost 3.6 CPU-hours" beside line 47 "3.65" | 3.647 and 3.624. | "Each run cost 3.65 CPU-hours" |
| 20 | "reaches the third ring at a seventh of its size at the interface" | 0.141 at 12 letters; 0.0525 without letters, where R is scored. The record's bullet says so; the question does not. | in the replacement of defect 1 |

### 14. Language

The brief asks the report for standard academic English, and the same holds for the brief. The following fail it: a fragment in the place of a sentence, a term not defined at first use, a metaphor, or a label that a reader outside the project cannot understand. Sentences already replaced under defects 1 to 12 are not repeated.

| Brief, line | Text | Fault | Replacement |
|---|---|---|---|
| 3 | "Evidence level E2." | fragment; "E2" not defined | "The evidence level is E2 in the programme's scale." The reviewer adds the file that defines the scale. |
| 3 | "**Status.** Written by the outside reviewer, acting for the coordinator, who cannot be reached while Mr. Xu is away" | fragment | "**Status.** This brief was written by the outside reviewer, who acts for the coordinator; the coordinator cannot be reached while Mr. Xu is away" |
| 3 | "It is given to a check by an agent of the reviewer that has not seen the reviewer's reasoning before it is released" | "before it is released" attaches to the wrong verb | "Before it is released it is checked by an agent of the reviewer that has not seen the reviewer's reasoning" |
| 11 | "1,024 cells on a grid with joined edges" | not a standard expression | "1,024 cells on a square grid whose opposite edges are joined (a torus)" |
| 11, 43 | "the readout", "the code of the current byte", "`l1`", "the input map", "`lm`", "`l2`", "`l3`", "the byte table" | names used and never defined | Add to "Terms": "Beside the cells the network has a table of 8 numbers for each byte (the *byte table*), a map `l1` from these to 512 values (the *code* of the byte), the *input map* from the code to the three gates of each interface cell, and the *readout*: a map `lm` from the interface cells' states and a map `l2` from the code to 512 units, and a map `l3` from these to the 257 possible output bytes." |
| 12 | "At E47's start they are uniform within 1/√11." | imprecise | "At E47's start they are drawn uniformly between −1/√11 and 1/√11." |
| 13 | "the mean size of all recurrent weights" | "size" not defined | Add to "Terms": "The *size* of a weight is its absolute value." |
| 14 | "*The first family*, scores A and R, the length test, and *updates to 80* are E47's and E42's: A is the share right on untaught pairs of add and sub at 12 letters, R the share right on untaught pairs of add and sub that need the reduction, without letters; a score reaches 80 at the first of three consecutive scored rows at or above 80 (rows every 256 updates)." | "the first family", "letters", "add and sub", "the reduction", "the length test" and "row" are not defined | "*The first family* is the task of E26: four operations on two numbers below 89 (addition and subtraction modulo 89, the larger and the smaller of the two), each written as a line of bytes with 0 to 12 capital *letters* that carry no information between the operands. Half of the pairs of operands of each operation are taught; the other half are never shown. Every 256 updates the network is scored; the scores of one such point are a *row*. Score A is the percentage of right answers on 600 untaught pairs of addition and subtraction written with 12 letters. Score R is the percentage on 2,000 untaught pairs of addition and subtraction whose answer needs the reduction modulo 89, written without letters. *The length test* is the percentage on 300 untaught pairs of addition written with 12 and with 40 letters. A score *reaches 80* at the first of three consecutive rows at or above 80, and *updates to 80* is the number of updates at that row." |
| 15 to 16 | (no definition of "the trial", "receipt", "dry run") | terms used later without definition | Add to "Terms": "*The trial* is E47's runs of seed 73129. A *receipt* is the file that a run or a check writes with its settings, its rows and the hashes of its sources. A *dry run* is a run of fewer than 24,576 updates, made to test the runner." |
| 24 | "What the record says (working rule 1)" | "working rule 1" not defined | "What the record says", followed by "The programme's working rules require a brief to state the record on which it rests (working rule 1)." The reviewer adds the file that holds the rules. |
| 26 | "Seed 73129 throughout." | fragment | "Every figure below is for seed 73129." |
| 31, 96 | "the way back" | label not defined in the brief | Add to "Terms": "*The way back* is measured by adding a small difference to the states of the cells of ring 3 at one position; it is the root-mean-square difference that then appears in the interface cells over the later positions, divided by that of ring 3." |
| 56, 65 | "gate values beyond 0.05 and 0.95" | unclear | "values of the reset gate and of the update gate below 0.05 or above 0.95" |
| 66 | "**Cost:** CPU-seconds an update and a scored row, and a forecast." | fragment | in the replacement of defect 6 |
| 94 | "More than one seed, the arm with periods (`place`), or other sizes of the grid." | fragment | "The job has one seed, one arm (the arm with periods, `place`, is not run) and one size of the grid, and cannot show anything beyond them." |
| 96 | "A start that passes differences outward only." | fragment | "The job does not run a start that passes differences outward only." |
| 103 | "Other branches are read-only" | label of computing used loosely | "Do not change any other branch" |
| 104 | "Commit trailers as `PROJECT_DOCUMENTS/KB/AGENTS_POINTER.md` says." | fragment | "Every commit carries the trailers that `PROJECT_DOCUMENTS/KB/AGENTS_POINTER.md` prescribes." |
| 114 | "The six checks; the four runs with receipts, …" | fragment | "The job is done when the following exist: the receipts of the six checks; the six runs (and the conditional run, if its condition is met), each with its receipt, starting weights, final weights, weights of updates 24,064 and 24,320, answers file and log; and `E53_REPORT.md` with …" |

## Verified and found sound

### A. Figures of "What the record says", of "Runs" and of the cost

Recomputed from `$W7:$M/exp45/e47/e47_32_one_{S,B}_73129.json` by `APP receipts`, and from `$R/signal_passage.py` by its parts `start`, `variants`, `more`, `trained` and `arith` restricted to `e47_one` (run times 15, 215, 247, 77 and 4 seconds).

| Brief | Recomputed | |
|---|---|---|
| Sweep: A reaches 80 at 13,568, R at 17,664; levels 92.3, 87.8, 90.8 | 13,568; 17,664; 92.28, 87.80, 90.78 | right |
| Back-propagation: 10,496 and 11,008; levels 95.4, 94.2, 95.2 | 10,496; 11,008; 95.39, 94.15, 95.22 | right |
| E42: dense 5,632 and 5,120; `F10` 13,312 and 8,704 | the same, from `e47_qual/e42_reference/` | right |
| Last row, sweep: 93.3 and 86.3; at zero 77.5 and 77.7; at the mean 84.8 and 80.6 | 93.33, 86.35; 77.50, 77.70; 84.83, 80.60 | right |
| Last row, back-propagation: 96.2 and 94.7; 78.8 and 78.9; 90.2 and 88.7 | 96.17, 94.65; 78.83, 78.90; 90.17, 88.70 | right |
| Weights: 0.151 and 0.150 at the start; 0.198 and 0.150; 0.179 and 0.243 | 0.1515, 0.1503; 0.1985, 0.1497; 0.1789, 0.2426 | right but for the first (defect 13) |
| Cost 3.6 and 3.65 CPU-hours a run; 14.6 for four | 3.647 and 3.624 | right |
| At the start: 0.187, 0.0207, 0.0028; 0.043 a connection | 0.1868, 0.0207, 0.0028; 0.0434 expected, 0.0420 to 0.0428 measured | right |
| Times 3: 0.141 (0.05 without letters); back 0.149; 0.95 a position | 0.141 (0.0525); 0.149; 0.951 and 0.959 | right |
| Times 3.25: 0.202, still shrinks; times 3.5 grows | 0.202, 0.978 and 0.988; 1.001 and 1.010, grows | right |
| 2.4 to 4.0 of ten connections from the ring nearer the interface | 2.38, 2.46, 3.98 | right |
| Trained: ring 1 0.59 and 0.50; ring 2 0.11 and 0.08; ring 3 0.015 and 0.0006; no longer shrinks | 0.587, 0.499; 0.113, 0.0752; 0.0152, 5.8e-04; 1.002 and 1.000 a position | right |
| Directed start times 5: 0.64 to ring 3, way back 0.0025 | 0.639; 0.0025 (the reviewer's document, section 4.1; not rerun) | as documented |
| "0.37 over 4,000 updates" | 0.99975^4000 = 0.3678 | the arithmetic is right; the inference is not (defect 1) |
| "changes no score by more than 0.7 of a point" | F_RERUN_FIGURES | F_RERUN_VERDICT |

`magnitudes` is `L.Wc.detach().abs()[:, cells].mean()` (runner, lines 370 to 372): the mean absolute value of the weights *into* the cells of the group, over the three gates and the eleven sources, the cell's weight on its own state included. Weights that leave a group for another group count with the group that receives them.

### B. The runner (read in full)

1. **Multiplying `L.Wc` in place after `build` and before `make_opt` changes the start and nothing else.** Nothing is derived from `Wc` when the layer is built (`e47_local.Local.__init__`: the tensor is drawn and wrapped; `a` is absent in the arm `one`). The sweep's scale is computed from `L.Wc` in every forward (runner, lines 132 to 134; `IndexSign` receives it), not stored. `plan` (lines 91 to 119) caches indices only. `install` (lines 244 to 257) holds the layer by reference. The module's own `rnn.weight_hh_l0` and `rnn.bias_hh_l0` take no part and are taken out of the optimizer (lines 220 and 221). `L.counts` does not depend on the values.
2. **The starting file holds the multiplied tensor**, because it is written after `make_opt` (line 445).
3. **`Wc` is exactly three gates by 1,024 cells by eleven weights** (33,792 numbers; `APP tensors`), column 0 being the cell's weight on its own state (`e47_local.py`, line 93), drawn within 1/√11 (largest size 0.301507).
4. **"The group that has 0.25" is well defined** in what `make_opt` returns: one group (asserted at line 219), holding the thirteen tensors that the brief lists, and the byte table in a second group with 0.0 (trial's receipt). `run` asserts nothing about the groups; `R_.opt_steps` only reads the step counts. The constant written by `opt_record` is defect 3.
5. **Resuming works for the new arms.** The saved state holds the update number, the module's tensors, `Wc`, `bc`, `bfar`, the optimizer's state with its groups and their decay, the receipt so far, the CPU times and the first batch (lines 412 to 416). On a resume `Wc` is copied from the state (lines 464 to 466) after any second multiplication, the optimizer's groups are restored by `load_state_dict` (line 469), and the hash of the start is taken from the starting file (line 443) and compared with the receipt's (line 471).
6. **Check 2 is achievable.** A starting file holds `book.weight`, the weights and biases of `l1`, of the input map, of `lm`, of `l2` and of `l3`, `Wc`, `bc`, `bfar`, and the five tensors of whole numbers `sources`, `periods`, `phases`, `interface`, `far` (`APP tensors`). The files themselves differ between runs (the trial's two starting files have different file hashes and the same `start_tensors_sha256`); the tensors are bit-equal, and the two rules start from the same tensors. A product by 3 in single precision is correctly rounded and therefore the same on every host (`APP tensors`).
7. **The first-update comparison of check 3 is exact up to rounding,** and its tolerance can both hold and fail (defect 6, row 3).
8. **The copies from `$W7` are sufficient.** `exp47_local.py` imports `exp26_candidate`, `arch26`, `e42_cells`, `e47_local` and `exp42_cells`; all but `e47_local` are on `$D`. The twelve files of `SOURCES` that are on `$D` have there the blobs of `$W7`, and the SHA-256 of `exp42_cells.py`, `e42_cells.py` and `exp26_candidate.py` on `$D` equal those recorded in the trial's receipts. `SPEC` points to `$D:PROJECT_DOCUMENTS/COORD_CHECKS_20261116/e47_local_spec.py`, which exists and equals `e47_local.py`. `E42_SOURCE` is a text and is not checked against anything. No assertion of `run` depends on the branch.
9. **Bit-equal reproduction on this host is on record** (`APP lost`), so that "equal" in checks 1 and 4 is achievable.

### C. The design

1. The comparison of the two rules stays readable at a start in which every weight is multiplied alike (the replacement bullet under defect 10; `APP scale`).
2. The statement of "What it cannot show" on a task that needs the far cells (line 95) is adequate as a statement of the limit.
3. The one-seed trial before more seeds follows rule 4 of section 4 of the plan.
4. E47's forms (*used*, *not used*, *a constant is used*, *partly*) are quoted as E47's brief has them (`$D:OPUS_TASK_E47_…md`, line 88), apart from "the first form that applies". `describe` is at lines 63 to 72 of `e42_tables.py` on `origin/claude/gallant-dijkstra-i9zgsr`, as the brief says.

### D. References

`$D` is at 7cabb13a and `$W7` at 578a877b, as the brief says. `exp47_local.py` is not on `$D`. The trial's receipts, starting files and final files exist under the paths that checks 1 and 2 name. The plan's items 9, 12 and 18 and its requirement C are as the brief cites them.

## Not checked

- Anything that needs torch: that the lines of the replacement under defect 3 run as written (the masked multiplication, the recomputation of the gates, the import of the runner's names into the new file); the optimizer's behaviour in torch 2.15.0 beyond what E46's check 2 records; the worker's cost on its host.
- The reviewer's measurement of the directed start (part `directed` of `signal_passage.py`) and its parts for the arm `place`, E49 and E50 were not rerun.
- The E46 report's statements beyond its receipts of seed 73129 (the two other seeds, its check 2).
- The definition of evidence level E2 and of "working rule 1"; the commit trailers of `AGENTS_POINTER.md` beyond the line that names them; the session named for `send_message`.
- Whether a host with another processor reproduces the trial's rows exactly; the record covers three machine images of one kind.
- The chances of the predictions; they are the reviewer's.

## Appendix: the script behind the figures

APPENDIX_SCRIPT
