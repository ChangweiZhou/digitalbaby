# Audit of the E42 brief and its specification file at 95f2d80f (16 November 2026)

From the outside reviewer. Read-only: nothing was trained, torch is not installed in the reviewer's container, and no code from another branch was executed. The brief is `OPUS_TASK_E42_CELLS_WITH_LIMITED_LOCAL_CONNECTIONS_AND_PERIODS_20261116.md`, revision 1. The specification file is `PROJECT_DOCUMENTS/COORD_CHECKS_20261116/e42_cells_spec.py`, with the coordinator's output `OUTPUT_E42_SPEC_MAC.txt` beside it. They implement section 22 of the reviewer's memo (revision 10, at f68f2b7d).

**Verdict.** The file's forward is section 22.3, the masked sign rule and the replacement rule are the ones meant, and the draws can be reproduced from the brief's text. A second implementation written from the text agrees with the file's committed output wherever the two can be compared without torch. Seven changes are required before release. One corrects the memo's own design of the `source` variant, not the coordinator's reading of it. One corrects the cost figure, which does not agree with the committed output. The other five are checks that are missing or cannot be run as written. No re-audit is needed after them.

**Terms.**

- A *cell* is one of the 256 units of the recurrent layer. Its *connections* are the N other cells it reads; it also reads its own previous value. The *mask* marks which entries of the recurrent matrix belong to a connection or to a cell's reading of itself.
- A cell's *period* p is the number of positions between its updates; its *phase* is the position of its first update. The *steps of its period* at an update at position t are the positions t, t − 1, …, t − p + 1.
- The four *variants* say how a cell combines what arrived during its period: `last` uses the last step only; `mean` gives every step the weight 1/p; `step` learns one weight a[k] for each step k; `source` learns, in addition, a separate matrix of strengths Wh[k] for each step.
- The *sweep* is the project's local rule: an error is sent back through a linear map by the signs of its weights times one scale, the mean magnitude of the matrix.
- The *reader* is `e42_cells_reader.py` beside this audit: a numpy implementation of the layer written from the brief's text. It computes only the cells that are updated at a position, group by group, where the file computes every cell and masks afterwards.
- *Stand-in initial weights* are weights drawn with numpy from the distributions torch uses for a new network. The network's actual initial values need torch's generator, which the reviewer does not have.

Shell names: `D`, `M` and `K` as in the brief; `R=PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109`; `C=$R/e42_brief_checks.py` on `claude/vibrant-cerf-sje726`. `python3 -I $C <letter>` runs one part (A to J). File lines are shown with `git show 95f2d80f:$K/e42_cells_spec.py | sed -n '<lines>p'`.

## 1. The forward, and the four points the coordinator fixed (point 1)

**The file's forward is section 22.3.** Three of the four fixed points are accepted as they stand. The fourth is accepted in part, and the remainder is required change 1.

What was compared, and how:

- **By reading** (lines 136 to 169): the input map is unchanged; the recurrent map reads h(t − 1) through the masked matrix; G and H are sums over the steps of a period with a step before the start symbol contributing nothing; the gate equations are the GRU's; a cell takes its new value only where `t >= f` and `(t - f) % p == 0`; the readout is unchanged. With every cell connected and all periods 1 the file calls `exp3_state.lin` itself and multiplies by coefficients that are exactly 1.
- **By a second implementation.** With every cell connected and all periods 1, the reader gives the same states and logits as the existing GRU reader on E26's birth of seed 73129, with a largest difference of zero at 0, 12 and 40 letters, and it reproduces E30's length-test figures for that file, 98.33 and 96.00. `python3 -I $C B`
- **Against the committed output.** The reader gives the same structural figures as the coordinator's self-checks: a neighbourhood of 48; 4,608, 8,448 and 16,128 recurrent weights inside the mask; networks of 932,113, 935,953 and 943,633; 384, 512 and 640 learned weights a. `python3 -I $C A E`
- **The properties of self-checks 3 and 4 hold in the reader** for all four variants: no state before a changed byte changes; a cell changes only at its own update positions; a cell is zero before its phase. `python3 -I $C D`

The four points:

| Point | Decision | What it changes in the function |
|---|---|---|
| `mean` divides by p always | Accept | A period cut short by the start of an item gets a smaller input, not a rescaled one. It makes `mean` and `step` the same function at the start. Under the other reading (dividing by the steps that exist) the reader finds `mean` differing from `step` by 0.033 in the logits and 0.22 in the states on stand-in initial weights. |
| In `source` the bias enters as a[k] · bh at each step | Accept | With the bias added once, `source` differs from `step` by 0.0029 in the logits at the start; with the fix by 1.5e-7 in the reader (the file reports 2.1e-7). |
| A cell of period 1 has no learned weight | Accept | It is the same cell in every variant, and `G` has no added parameter, so it stays E26. One consequence is noted in section 7: the tensor `a` still holds an entry for each such cell. |
| `Wh[0]` is the existing tensor | Accept the tensor; change its start (required change 1) | See below. |

`python3 -I $C C` gives the figures of the first two rows.

**The start of `source` (the memo's fault).** Section 22.3 says each Wh[k] starts "at the masked Wh divided by p_i", and the file does that (lines 89 to 92). The forward is then right, but the sign rule is not on the same footing as in `step`. The sweep sends an error back through a matrix by one scale for the whole matrix. In `source` the weights of a cell of period 8 are one eighth the size of those of a cell of period 1, by construction, and the one scale is an average over both.

| Reader's period | Mean size of one step's weight in the forward | What the sweep sends back for it in `source` as written | Ratio | In `step` |
|---|---|---|---|---|
| 1 | 0.151 | 0.060 | 0.40 | 1.00 |
| 2 | 0.075 | 0.060 | 0.81 | 1.00 |
| 4 | 0.038 | 0.060 | 1.60 | 1.00 |
| 8 | 0.019 | 0.060 | 3.19 | 1.00 |

`python3 -I $C F` (stand-in initial weights, seed 73129's connections and periods).

So `P8-source` and `P8` compute the same function at the start, but under the sweep they send back errors that differ by a factor of 0.40 through the fast cells and 3.19 through the slowest, an eightfold spread. `P8-source` runs under the sweep only, and its reading is against `P8`. The comparison would then mix two things: separate strengths for each step, which is the question, and a different backward weighting of fast and slow cells, which is not. There is a second, smaller effect under both rules: the optimizer moves every weight by about the learning rate, 0.001, which is 5% of a weight of size 0.019 and 0.7% of one of size 0.151.

The remedy keeps everything the coordinator fixed and moves the factor 1/p from the matrices to the weights a, where `step` already has it:

    H = sum over k of a_i[k] * [ (Wh[k] * mask) h(t - k - 1) + bh ],   Wh[k] starting as the masked, scaled Wh, not divided

In the reader this form agrees with `mean` and `step` at the start to 1.2e-7, and the ratio in the table is 1.00 for every period. `rnn.weight_hh_l0` then has the same values at the start in `P8` and `P8-source`.

## 2. The masked sign rule (point 2)

**`MaskedSign` is the rule meant.** No change to it. One check is missing (required change 3a).

By reading lines 48 to 63 and 109 to 116:

- **Scale.** `sign_scale` divides the summed magnitude of the weights inside the mask by the number of entries inside the mask, and in `source` adds the entries of Wh[1 …] that lie inside the mask and inside the reader's period. That is the mean magnitude inside the mask, one scale for all steps stacked.
- **Back through the map.** The error crosses `sign(weight * mask) * scale`, so an entry outside the mask sends nothing back.
- **Weight gradient.** The usual product of error and input, multiplied by the mask: zero outside it.
- **Every entry inside the mask.** `rec` calls `exp3_state.lin` unchanged when the layer is full (lines 128 and 129), so `G` takes E26's own path and not `MaskedSign`.
- **A new connection.** Its weight is zero, and the sign of zero is zero, so it sends no error back at its first update. Its weight gradient is not zero, so the optimizer moves it at that update, and from then on it sends back like any other. This is the behaviour section 22.3 intends.
- **In `source`** each step's crossing uses the mask narrowed to the cells whose period contains that step, and the same scale.

**What is missing.** Nothing in the file or the brief compares `MaskedSign` with the existing rule. Self-check 1 runs `G`, which bypasses it; self-check 5 shows only that gradients are zero outside the mask and that the two rules differ. A mask with every entry true is the one case where the right answer is on file: `MaskedSign` with that mask and the scale `weight.abs().mean()` must return what `exp3_state.SignLinear` returns.

## 3. The replacement rule (point 3)

**`replace` is section 22.3's rule, clause by clause.** No change to it.

| Clause | In the file (lines 172 to 194) | Agrees |
|---|---|---|
| Strength: summed magnitudes over the three gates and, in `source`, over the steps | `strength`, lines 118 to 123 | yes |
| Given up if at least 512 updates old and below theta times the cell's own mean | `update - made >= MIN_AGE and s < theta * mean`, the mean over the cell's N connections | yes |
| The cell's reading of itself is never given up | `conn` excludes i | yes |
| Candidates: in B_i, not connected at that moment, not given up at this event | `not layer.mask[i, q] and q not in drop`, recomputed before each draw | yes |
| Order: cells in index order, connections in increasing order of source, one `rr.choice(sorted(candidates))` each | as written | yes |
| New weights zero; old weights zero; the optimizer's two averages zero at both | the loop over `(j, new)`, for the three gate rows and, in `source`, every step | yes |

- **"Both ends" is right, and the old end is necessary.** "Both ends" are the entries of the connection given up and of the new one. At the old end the averages must be cleared: the gradient there is zero from then on, but a remaining average would keep moving the weight for many updates, and a weight outside the mask would stop being zero. At the new end the averages are already zero unless the same pair was connected earlier, so clearing them costs nothing.
- **The optimizer's state is addressed correctly**: by the parameter object, `exp_avg` and `exp_avg_sq`, at the three gate rows of cell i and the column of the source.
- **The candidates never run out.** A cell can give up at most N − 1 connections at one event, since not every connection can be below a fraction of their mean. In the hardest case, with all but one connection of every cell at zero strength, the fewest candidates left for a cell's last draw are 30 at N = 10, 10 at N = 20, 40 at N = 5 and 237 without a neighbourhood. The limit, the neighbourhood and "not connected before" hold at every event of the test. `python3 -I $C G`
- **One thing the self-check does not exercise.** Self-check 6 calls `replace` before any optimizer step, when the optimizer has no state, so the lines that clear the averages are not run. The brief's check 4 would catch a failure, because a weight outside the mask would move; recommendation 3 makes it direct.

## 4. The draws (point 4)

**They are reproducible from the text.** I found no place where two readers of the brief could differ. Required change 3b turns that into a check.

- The reader makes the three draws from the brief's three lines and from "the 48 cells whose row and whose column each differ by at most 3, with wrap-around". Its neighbourhood is computed by distance, the file's by offsets; the sets are equal by reading (lines 26 to 30).
- The period sets are filled in increasing order of period from one shuffled order, so the sets are nested: a cell has the same period in `P4`, `P8` and `P16` unless it is among the slowest. `F10`, `A10`, `P4`, `P8`, `P16` and the three variants share one set of connections per seed. `python3 -I $C A`
- For `N5` and `N20` the connections are a different draw, not a subset or superset of `A10`'s: one cell of 256 has its five among its ten. These two configurations therefore differ from `A10` in which cells are read as well as in how many. With changing connections that difference fades during a run; it is stated here so that the report does not call it a one-factor change at the start.

Digests of the draws from the text (`hashlib.sha256(json.dumps(x).encode()).hexdigest()[:16]`, with x the list of periods, the list of phases, or the list of sorted connection lists):

| Seed | Periods 1, 2, 4 | 1, 2, 4, 8 | 1 to 16 | Phases for 1, 2, 4 | 1, 2, 4, 8 | 1 to 16 |
|---|---|---|---|---|---|---|
| 73129 | cdc419757fea4346 | d5d8c69f0af22702 | f8eb80a8bfd8e68c | 868ece46952abfb3 | 40f16c7455e0044e | 4c59a7af5f14339f |
| 93683 | 751146dbc2b41c71 | 5fe0fd0e1a9b6965 | 8b2f574ec8e87baa | 6f6c3432f96493de | f42f73cdb29e14d1 | 0f4a27cf78ed1cb7 |
| 116089 | b4f4dae8defa25dd | 007709688248573e | 1ba63ec1b4fabfde | 487c224b1f4cb59e | e9a53451f04f7cd7 | 8d400c4a2d682efe |

| Seed | Connections, N = 5 | N = 10 | N = 20 | N = 10, no neighbourhood |
|---|---|---|---|---|
| 73129 | ecf98006edb8ad9b | fde78220c57da3d4 | dbe99af8352f428c | 4a9feeeb77ff57d2 |
| 93683 | cc36f3f00a8da08a | 35b553967b0c40ee | 27ec6683c6bef9a1 | e6c342ea94907a45 |
| 116089 | dc9097fe8fbead59 | 70afb39ae3b47296 | a10f50b4bb550930 | 6c1147c9bfab9d16 |

A fixed point for seed 73129: with periods 1, 2, 4, 8 the first twelve cells have periods 1, 2, 1, 8, 2, 1, 1, 1, 4, 1, 2, 1 and phases 0, 0, 0, 2, 1, 0, 0, 0, 1, 0, 1, 0; at N = 10 cell 0 reads cells 13, 14, 16, 45, 51, 61, 209, 222, 242, 243.

The replacement draws depend on trained weights and have no fixed point before a run. The brief's check 8 covers them, since the dry runs' files carry the mask.

## 5. The checks (point 5)

**The ten checks are not sufficient as written.** Four additions and one correction are required (changes 2 to 5). Check 3 is acceptable for what it can show.

- **Check 3, the file's own self-checks.** It is acceptable as a check that a worker's host reproduces the coordinator's run of the same file. Self-check 1 has an outside reference, `arch26.forward`. Self-checks 3 to 6 compare the file with itself. The outside references for the limited configurations are check 6 (the worker's second form), check 8 (the reader) and section 1 of this audit.
- **Check 3 cannot be run on the copy as written (required change 2).** The file finds the harness by `HERE.parents[2] / "MOD97_LOCAL_RULE_20261106"` (line 12). From `$K` that is the harness folder. From the copy at `$M/e42_cells.py` it is a folder of that name beside the repository, which does not exist, and `python3 -I` does not put the script's own folder on the path. So "its self-checks print what the output has" fails at the first import if "its" is the copy. `python3 -I $C I`
- **`MaskedSign` has no reference (required change 3a)**, as section 2 says.
- **The draws have no fixed point (required change 3b)**; section 4 supplies one.
- **The scoring is not checked, and its code is not on the default branch (required change 4).** The brief takes E41's three accuracies "unchanged", but `exp41_scale.py` is on `claude/wonderful-edison-8hx6i6` and not on the default branch at 95f2d80f. `python3 -I $C J`. Whatever the E42 runner uses, nothing in the ten checks tests it: check 1 covers training and check 8 the length test's answers. The figures to test against are on file. E41's check 7(b) gives the counts of A, R and the taught panel for the six E26 births, and E30 gives the length test for the same six files. Configuration `G` run on those files must reproduce both.
- **A sped-up forward is not tied to the specification (required change 5).** Check 9 lets the worker speed up "the runner's own copy of `forward`, provided check 6 and the self-checks still pass on it". The self-checks run the file's forward, not the copy, and check 6 covers backpropagation only. Section 6 shows the speed-up is needed, so its equality with the specification has to be a check, under both rules.
- **Self-check 4 covers `step` only.** `source` reads earlier states through its own code path (the list `hprev_hist`). Recommendation 1.

## 6. Cost, and what to defer (point 6)

**The figure of 85 CPU-hours does not follow from the committed output, and the brief's two ratios do not agree with it (required change 6).**

- The output's last four lines give 0.039, 0.060, 0.180 and 0.307 CPU-seconds for `G`, `A10`, `P8` and `P8-source`. The ratios to `G` are 1.54, 4.62 and 7.87. The brief and review 9.140 state 2.6 and 5.4. `git show 95f2d80f:$K/OUTPUT_E42_SPEC_MAC.txt | tail -4`
- With the cloud host's measured cost for the present network (E41's check 6: 0.120 and 0.111 CPU-seconds per update under the two rules, 5.1 per scored row, about 7.5 with the length test) the 48 runs come to about **151 CPU-hours with the output's ratios and 103 with the brief's**. A ratio of forward-and-backward times overstates the ratio per update, so the true figure for the file as it stands lies below 151; I cannot say by how much.
- **The loop's cost is overhead, not arithmetic.** In the forward, the two sums over a period of 8 add 1.1% to the multiply-adds of a position, and `source` adds seven crossings of the recurrent map, 123%. If the loop cost what its arithmetic costs, the period configurations would cost 5 to 10% more than `A10` and `source` about 2.2 times `A10`, and the 48 runs about **78 CPU-hours**. `python3 -I $C H`

So the speed-up should be expected of the worker, not left optional, and the decision on what to run should use check 9's forecast after it.

**What to defer.** I went in expecting to defer `P8-last`, `P8-mean`, `N5` and `N20`. The costs change that.

| Configuration | Question it answers | CPU-hours for three seeds (after the speed-up, my estimate) | Recommendation |
|---|---|---|---|
| `N5`, `N20` | The user's own axis: "we can adjust N" | 5 each | Keep: the cheapest configurations |
| `W10` | Does the neighbourhood matter at all | 5 | Keep |
| `P4`, `P16` | The prediction that longer periods ease training and hurt the length test | 5 to 6 each | Keep |
| `P8-source` | Each source read at its own distances in time: the user's statement of several earlier moments | 11 | Keep, run last |
| `P8-last` | Does absorbing during the period matter, against reading the last step only | 5 | Keep; first to go if the forecast is still too high |
| `P8-mean` | Does learning the 512 weights a matter, against leaving them at 1/p | 5 | **Defer.** It starts as the same function as `P8` and differs only in whether those weights move. It is the smallest question, and it can be answered later from `P8`'s saved weights a: if they stay near 1/p, `P8-mean` would have told nothing more. |

After the speed-up and without `P8-mean` my estimate is about 73 CPU-hours. If a worker's forecast for its runs still exceeds the brief's bound of 30 CPU-hours, `P8-last` is deferred next. `N5` and `N20` stay: deferring both saves less than deferring one period configuration.

## 7. Other points

- **The count of weights for weight-items is ambiguous (required change 7).** "The count of weights inside the mask" does not say how the added tensors count. As the file makes them, `a` has 2,048 entries in `P8`, of which 512 take part; 128 more are non-zero and belong to cells of period 1, which never use them. In `P8-source` the tensor of Wh[1 …] has 1,376,256 entries, of which 12,672 lie inside the mask and inside the reader's period. A worker counting tensor sizes would give `P8-source` 2.3 million weights and change its description in weight-items. `python3 -I $C E`
- **The data are E26's.** The stream call is E41's at fraction 0.50, the draws use their own generators, and nothing in the layer's construction draws from torch's generator. Check 2's three digests and check 1's two hashes are the ones in the E41 brief; E30's six figures at 40 letters are on file as quoted. `python3 -I $C J`
- **The readings are computable**, with two wordings to tighten (recommendations 5 and 6).
- **The count of carrying cases** quoted from the memo (6%, 44%, 67%, never at 40 letters) is reproduced by `e42_design_checks.py`. `python3 -I $C J`
- **The predictions** can each be scored from the tables the brief asks for.

## 8. The seven required changes, as text to apply

**1. The `source` variant (brief, the table of variants and point 4 of the fixed points; file, lines 89 to 92 and 152 to 156).**

- Table row: H = Σ_k a_i[k] · [ (Wh[k] * mask) h(t − k − 1) + bh ]. Added parameters: Wh[k] for k = 1 … p_i − 1, each starting as a copy of the masked, scaled Wh; a_i as in `step`.
- Fixed point 4: "Wh[0] is the existing tensor `rnn.weight_hh_l0`, unchanged by the variant, and Wh[1 …] are one new tensor of shape (K − 1, 768, 256), a copy of it at the steps inside a cell's period and zero outside."
- File: do not divide the recurrent weights by the period; in the forward multiply each step's crossing, bias included, by the step's coefficient. Self-check 3 must still show `source` within 1e-6 of `mean`, and self-check 5's cosine for `P8-source` is reported again.
- The reason is the memo's: its start for `source` puts the factor 1/p where the sign rule cannot see it (section 1).

**2. Check 3.** "The self-checks are run on `$K/e42_cells_spec.py`, where the file finds the harness; `$M/e42_cells.py` is byte-identical to it and is only imported by the runner." Or make line 12 find the harness from either place.

**3. Three additions to the file's self-checks.**

- (a) `MaskedSign` with a mask of ones and the scale `weight.abs().mean()` returns the same three gradients as `exp3_state.SignLinear` on the same input, to 1e-6 of each tensor's largest entry.
- (b) The digests of the draws for the three seeds equal the two tables of section 4.
- (c) Self-check 4 is run for each of the four variants with periods 1 to 16.

**4. A new check: the scorer.** "With configuration `G`'s layer and the six E26 births `exp45/e26/e26_dev_<S|B>_<seed>.pt`, the runner's scoring gives the counts of E41's check 7(b) for A, R and the taught panel, and E30's accuracies at 12 and 40 letters (`exp45/e30/E30_UNSEEN.json`). A count that differs by one or two items is reported; a larger difference is a failure." The brief also says where the three accuracies' code comes from: `panels`, `reduction_sample` and `score_all` of `exp41_scale.py` at 88b8b7b3 on `claude/wonderful-edison-8hx6i6`, or a rewrite from the E41 brief. This check decides either way.

**5. Check 9, the speed-up.** "The worker speeds up the loop over a period's steps in the runner's forward. For every configuration it is used in, on the first batch of seed 73129, its logits equal the specification's to 1e-6 and every parameter gradient, the added parameters included, equals the specification's to 1e-5 of the tensor's largest entry, under backpropagation and under the sweep."

**6. Cost (brief, check 9; review 9.140).** Replace the two ratios by the committed ones (4.6 and 7.9 for a forward and backward) and the figure of 85 by "about 150 CPU-hours for the file as it stands by those ratios, about 75 to 80 if the loop costs what its arithmetic costs". `P8-mean` is deferred. If a worker's forecast after the speed-up exceeds 30 CPU-hours, `P8-last` is deferred next, and the worker reports before starting.

**7. The count of weights for weight-items.** Add: "The count is the network's 1,124,113 less the 196,608 recurrent weights, plus the recurrent weights inside the mask, plus the weights a of cells of period above 1, plus in `source` the entries of Wh[1 …] inside the mask and inside the reader's period."

| Configuration | Count |
|---|---|
| `G` | 1,124,113 |
| `N5` | 932,113 |
| `F10`, `A10`, `W10`, `P8-last`, `P8-mean` | 935,953 |
| `N20` | 943,633 |
| `P4` | 936,337 |
| `P8` | 936,465 |
| `P16` | 936,593 |
| `P8-source` | 949,137 |

## 9. Recommendations (none holds the release)

1. **Check 5 on the runner's forward for all four variants**, not `step` alone.
2. **Check 7's hash should cover the structure.** `tensors_sha256` covers the thirteen tensors. Add a digest of the mask, of the update at which each connection was made, and of the added parameters, and compare all three between the two runs.
3. **Assert after every event** that the optimizer's two averages are zero at every entry outside the mask.
4. **`G`'s final hash.** At 24,576 updates `G` is E26's birth, so its `tensors_sha256` can be set beside E26's receipt for each rule and seed. A difference is reported, not a failure, since host classes have differed before (review 9.101).
5. **The stop.** Say that "reaches 50 on A" uses the three-row version, as the descriptions do.
6. **P42.5.** "Not *rises*" should say how *not decided* and *neither reaches* count.
7. **Names in the weights file.** Fix the names of the added entries (mask, periods, phases, made, a, Wh[1 …]) in the brief, and write the structure as a small JSON file beside each weights file as well: periods, phases, each cell's connections and the update at which each was made. The structure tables and the two seeds' "small files" then need no torch, and the JSON is a few hundred kilobytes where Wh[1 …] alone is 5.5 MB.
8. **The weights a of cells of period 1.** Either leave them out of the tensor or say in the brief that they take no part; they decay from 1 during a run and should not be read as learned.
9. **Order of stage 2.** The configurations of period 1 first, then `P4`, `P16`, `P8-last`, and `P8-source` last, so that a worker stopped by cost has the cheaper questions answered.
10. **`N5` and `N20`** draw their connections afresh (section 4); the report should say so where it compares them with `A10`.

## 10. Can E42 be released?

Yes, after the seven changes. **No re-audit is needed**: change 1 is three lines of the file and is tested by the file's own self-check 3 and later by check 8, where the reader reads the dry runs; the reader already holds both forms of `source`. Changes 2 to 7 are text and checks.

## 11. What I did not check

- **Anything that needs torch.** The file was not run. Its output is the coordinator's. That `G` is bit-equal to E26 under training, the gradients under either rule, the cosines of self-check 5, and the timings are taken from that output.
- **`MaskedSign`'s backward and the clearing of the optimizer's averages** were read, not executed.
- **The start with the network's real initial values.** The comparisons of variants and the table of section 1 use stand-in initial weights with the same distributions; the figures would differ in the third digit, not in kind.
- **The cost after a speed-up** is my arithmetic, not a measurement.
- **Whether Python's generator gives the same draws** on the coordinator's Python 3.11 and the workers' 3.13. The digests above are from Python 3.13.17; change 3b tests it on each host.
