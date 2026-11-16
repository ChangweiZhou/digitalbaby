# Audit of the E46 brief: does learning the first family need weight decay? (16 November 2026)

From the outside reviewer. Audited: `OPUS_TASK_E46_DOES_LEARNING_THE_FIRST_FAMILY_NEED_WEIGHT_DECAY_20261116.md`, revision 1, at `3126dead` on `codex/minifly-consolidated-20261023`, with review 9.147 and 9.148.

**Verdict.** The job is the right first check and it is small. Six changes are required; they are written in section 7 as text to apply. After them the brief can be released without a second audit.

Everything below is read-only. Nothing was trained, torch was not used, and no code of another branch was run. The commands are parts of `e46_brief_checks.py` beside this file, run from the repository root as `python3 -I PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/e46_brief_checks.py <part>`.

## 1. The reference (point 1)

**What is on file is sound** (part A).

- E41's two present-width cells of seed 73129 at half of the pairs are on the default branch, complete, with their weights; each weights file has the hash its receipt records.
- `exp41_scale.py` on the default branch has the hash recorded in those receipts (559a7ee9…), and none of the eleven recorded sources differs from the default branch's files. So a cell run from the default branch with that file is the cell an E41 worker would have made.
- The thresholds the brief quotes are right: 80 on A at 5,632 and 5,120 updates, 80 on R at 6,656 and 6,400.

**"As an E41 run" is not defined in the brief.** It names no command, no commit, and none of E41's checks for the seed concerned (check 1, check 2, the sample digests of check 7, and check 8, which ties a finished cell to E26's panel at rows 4,096 and 8,192).

**The reference is not scored as the new runs are, and that breaks one comparison.** The brief compares the level at 40 letters as a mean of three rows for the new runs against "the final row alone" for the reference, because E41 saved no weights before its last row. Under the standing decay that test moves widely from row to row at this age (part B; E42's `G` runs of seed 73129, which equal E41's cell at every one of its 49 rows):

| Rule | 40 letters, single rows 9,216 to 12,288 | Three-row means over the same rows | Final row | Mean of the last three |
|---|---|---|---|---|
| Sweep | 72.7 to 89.3 | 77.2 to 82.0 | 89.3 | 82.0 |
| Backpropagation | 67.7 to 96.0 | 81.9 to 91.9 | 86.0 | 86.4 |

A mark of 15 points cannot be read between a three-row mean and one such row. The reference also has no weight norms before its last row, which the brief wants "to show what the weights do without decay".

**The remedy costs two runs.** Four of the six reference cells have to be run by this worker in any case, since E41's other seeds have no worker. If all six are run by the E46 runner at decay 0.25, the three decay values are scored by one code on the same rows, and the two cells of seed 73129 become the strongest form of the brief's check 1: their final `tensors_sha256` must be E41's (7bb47c74… under the sweep, 4d509cd2… under backpropagation) and every row's counts must be E41's. Section 7, change 1.

**Does the runner differ from E41's only in the decay?** By the brief's text, yes: the decay is set on the first group after `make_opt`, which at 0.25 changes nothing. The brief's check 1 asks for the hash of E26's dry birth at 2,048 updates under both rules and for equal scores at one row under backpropagation. It does not ask for the full run's hash. Change 1 supplies it.

## 2. Check 2, the first update (point 2)

**The relation the check asserts does not hold.** AdamW shrinks a weight and then takes its step: p1 = p0 (1 − lr d) − s, where the step s = lr · g / (|g| + eps) is computed from the gradient at p0 and is the same in the two runs. So the run without decay and the run with it differ by lr · d · p0 **added**. The check says the first equals the second **divided by** (1 − lr d), which would be true only where the step is zero.

In a rebuild of the first update in float32 (part C), with a gradient on 98% of the entries:

| Tensor | Entries meeting the brief's relation to 1e-6, decay 0.25 | Largest miss of p1(0) − p1(d) = lr d p0 | Decay estimated from the difference |
|---|---|---|---|
| `l1.weight` | 30.5% | 2.1e-08 | 0.24998 |
| the six others | 2.0 to 2.2% (only the entries with no gradient) | 6.3e-09 to 9.9e-09 | 0.24998 |

At decay 0.05 the estimate is 0.05001. No weight outside the byte table has a zero gradient on a real first batch, so the check as written would fail on nearly every weight of a correct runner.

**What the check should compare.** Two runs of one update from the same start on the first batch, one at decay d and one at 0 (section 7, change 2): the byte table bit-equal, and for every other tensor the decay estimated from the difference, d̂ = ⟨p1(0) − p1(d), p0⟩ / (lr ⟨p0, p0⟩), within 1% of d. The estimate does not depend on the order of shrinking and stepping, so it also holds on a torch whose AdamW should differ in that. The groups read back from the optimizer remain as the brief has them.

## 3. The sentence (point 3)

The brief: "**needs** decay if without it the 80 threshold on R is reached in more updates (*rises* or not reached) or the level on R or at 40 letters is *lower*; **does not need** it if all of those are *stays* or better; otherwise **mixed**".

Enumerated over the six descriptions of the threshold cell and the four of each level cell (part D, 96 combinations):

1. **Every combination gets exactly one sentence if the rule is read on the cells' descriptions**: 51 *needs*, 8 *does not need*, 37 *mixed*.
2. **"Not reached" has two readings.** Read as E41's table reads it, a run that does not reach the threshold is *rises* only when the reference reached it by 7,850 updates (the last recordable row, 11,776, over 1.5), and otherwise *not decided*; if the reference did not reach it either, *neither reaches*. Read as the run's own failure, those cases are *needs*. The two readings give different sentences in 18 of the 96 combinations. On seed 73129 the references are at 6,656 and 6,400, inside the bound; for the other seeds R has never been scored.
3. ***Needs* is given by one cell alone in 39 combinations.** "Needs decay" for the threshold on reduction pairs and "needs decay" for 40 letters at half a birth's dose are different findings. The sentence must name the cell.
4. **The third outcome is not a mixture.** As the rule is built, *mixed* arises only when some cell is undecided (seeds disagree, *not decided*, *neither reaches*) and none is bad. It should be called what it is.
5. **If decay 0 is faster,** the cell is *falls*, which is "*stays* or better": *does not need*. That is right, and the cell says the rest.
6. **The 0.05 runs do not enter the sentence.** The brief should say so.

"Does not need it" requires the threshold on R, the level on R and the level at 40 letters together. That is the right strictness; A adds nothing R does not.

## 4. What to add, and what the record already says about decay (point 4)

**The record has more on decay than the brief cites, and it points away from the cell E46 runs.** The brief says the decay "was chosen on 7 November for the sign rule on another network (review 9.20 to 9.21)". Sections 9.20 and 9.21 are E10 and E9. The decay of 0.25 was chosen in 9.24 and 9.25 and confirmed in 9.27, for lives of that week's networks, where 0.1 to 0.25 was a plateau. What bears on a birth is 9.2 and 9.3, on the same earlier network:

- at **half** of the pairs the sign rule reached 95 to 98% within 16,384 updates, with the decay of that time, which 9.3 gives as 1e-4 (9.2 does not state it; I infer it);
- at **15%** of the pairs the same setting left backpropagation on a plateau of 65 to 72% with the taught pairs at 100%, and the sign rule drifted down to 35 to 45%, where a decay of 0.3 gave 82 to 89%;
- at a decay of 0.1 and below, "held peaks early and drifts while taught stays at 100%", in runs of 98,304 updates.

So the record's expectation for E46's cell, half of the pairs within 12,288 updates, is that decay is not needed, and the two places where the record found a cost of weak decay are few taught pairs and long training. E46 as written looks at neither. `git show 3126dead:MINIFLY_PAST_WORK_REVIEW_FOR_MR_ZHOU_20261106.md | sed -n '318,328p;592,600p'`.

**A decay between 0.05 and 0.25: no.** The record's plateau already covers 0.1 to 0.25. It would matter only if both 0 and 0.05 failed.

**A second learning rate: no.** It changes every other quantity of the birth and makes E46 another experiment.

**The 0.05 arm: keep it.** A weight that nothing pushes on keeps 54% of its size over the dose at 0.05, 5% at 0.25 and all of it at 0 (part E), so the three settings are distinct treatments. Its use is as the fallback if 0 fails: at 0.05 an unused family would last about five times as long as now.

**If one cell is added, a quarter of the pairs at decay 0.** It is where the two rules differ most in E41's first seed (80 on R at 7,680 updates under backpropagation and not within the dose under the sweep) and where storing the pairs is a live alternative to the rule. It is not on the path of the retention step, whose births are at half of the pairs, so it is a recommendation and not a condition (section 8).

## 5. The dose (beyond the four points)

**For the threshold, 12,288 is enough.** The brief's comparison with the record is right for seed 73129: a factor of about 1.8 on R.

**For the level at 40 letters, 12,288 is half a birth, and the reference is still rising there.** At rows 11,776 to 12,288 the standing network is at 82 (sweep) and 86 (backpropagation) at 40 letters; at the end of E26's birth, 24,576 updates, it is at 96 and 95 (part B). A birth that the retention step would start from is the full one.

**The record's cost of weak decay is a late one** (section 4), and the brief's "what it cannot show" names a life but not the second half of a birth.

**At the full dose the reference is anchored bit for bit on all three seeds.** The standing network run again to 24,576 updates with A, R, the taught panel and the length test scored every 256 (E42's `G`) has E26's birth hash under both rules on seed 73129 (d9064126…, efbdf87a…; part G). So the six runs at decay 0.25 of a full-dose E46 must reproduce E26's six `tensors_sha256`, and the no-decay runs are then births that the next job can start from. The cost is in section 6.

Section 7, change 5 gives the dose as the first choice and a restricted sentence as the second.

## 6. Other findings

1. **The receipt would name the wrong decay.** `exp26_candidate.opt_record` writes `weight_decay=WD`, the constant 0.25, beside the groups it reads back (part A). A copy of E41's runner therefore records `optimizer.weight_decay: 0.25` in every arm, at the start and in `optimizer_read_back`. The brief asks for "the decay in a field of its own", which does not remove the wrong one.
2. **Cost.** The brief gives about 2.5 CPU-hours of updates and 1.5 of scoring for the twelve new runs, and the coordinator's message "about 4 CPU-hours with scoring". From E41's own receipts (0.105 and 0.113 CPU-seconds per update) and E42's scoring with the length test (5.7 per row), part F:

   | | 12,288 updates | 24,576 updates |
   |---|---|---|
   | The twelve new runs | 5.4 CPU-hours | 10.8 |
   | With four reference cells for the two missing seeds | 7.2 | 14.4 |
   | All three decays through one runner (change 1) | 8.1 | 16.2 |

3. **The weight groups are not named.** "The norm of each weight group" and P46.8's "the recurrent matrix's norm" need the groups of E43's score 4 (byte table, `l1`, input map, recurrent matrix with its bias, `lm`, `l2`, `l3`).
4. **The sample of reduction pairs depends on a string.** The copy must keep E41's `Random(f"e41|{seed}|…")`; a renamed string gives another sample with no error. Each receipt's `reduction_sample_sha256` should be asserted against E41's check 7 table for half of the pairs, which holds all three seeds.
5. **The statement on E38 is right.** New family at 96.25 and 96.6 without decay against 92.5 and 94.75 with it, from the receipts (part H; `e38_none_*` is the arm without decay, `e38_zero_*` the arm with it).
6. **Nothing is said about lives that the job cannot support.** The sentence excludes them and the limits name them.
7. **The other checks stand:** the batch digest at row 4,096, the length test's code on E26's six files against E30, a dry run per decay. The instruction to force-add the weights and confirm them on the remote answers what went wrong in E43.

## 7. The six required changes, as text to apply

**1. The reference (the table of runs; "Committed"; check 1; the levels).**

- Table of runs, first row: "0.25 | 6: both rules, three seeds | this runner. They are the reference for every comparison."
- Delete "For a seed whose E41 receipts are not there … named as one" and "For the reference, the length test is scored once on E41's committed final weights by this runner's code" and, under the level, "For the reference's length test, the final row alone."
- Check 1 becomes: "**With decay 0.25 the copy is E41 and E26.** (a) A dry run of 2,048 updates at decay 0.25 on seed 73129, with every added score on, has the `tensors_sha256` of E26's dry births (79618ab2… under the sweep, adf933d7… under backpropagation). (b) The two full runs at decay 0.25 of seed 73129 have, at every row up to 12,288, the A, R and taught counts of `e41_<S|B>_h512_f50_73129.json`; at a dose of 12,288 their final `tensors_sha256` is that receipt's (7bb47c74…, 4d509cd2…). (c) For each of the six runs at decay 0.25, E26's panel at rows 4,096 and 8,192 equals the `sealed` counts of `e26_dev_<S|B>_<seed>.json` (E41's check 8). (d) Reported, not a failure if the branch is not at hand: the length-test counts of seed 73129's two runs equal those of `e42_G_<S|B>_73129.json` on `claude/gallant-dijkstra-i9zgsr` at every row."

**2. Check 2.** "**The decay is what is set.** On the first batch, one update from the same start is made at decay d (0.25, and again 0.05) and at decay 0, under backpropagation. (a) The byte table is bit-equal in the two. (b) For every other tensor, with p0 the start: d̂ = ⟨p1(0) − p1(d), p0⟩ / (0.001 ⟨p0, p0⟩) is within 1% of d. (c) Reported: the largest value of |p1(0) − p1(d) − 0.001 d p0| over each tensor, which is a few times 1e-8 when the shrink precedes the step. (d) The groups read back show d and 0.0 at the start and at the end of every run."

**3. The sentence.** "For each rule, three cells are formed from decay 0 against 0.25, each carrying the description two of its three seeds share: the threshold cell (updates to 80 on R, E41's table of four cases, in which a run that does not reach the threshold is *rises* if the reference reached it by the last recordable row over 1.5, and otherwise *not decided*), the level of R, and the level at 40 letters. Learning the first family **needs decay for** each cell that is *rises* or *lower*, and the sentence names those cells. It **does not need** it if the threshold cell is *falls* or *stays* and both level cells are *about the same* or *higher*. Otherwise it is **not decided**, with the cells given. The runs at 0.05 are described in the same three cells and enter no sentence. Nothing is said about lives or about other families." P46.4 and P46.5 take "not decided" for "mixed".

**4. The receipt.** "The `optimizer` and `optimizer_read_back` fields are written with `weight_decay` equal to the run's own decay (`exp26_candidate.opt_record` writes the constant 0.25), and the receipt has `decay` as a field of its own."

**5. The dose, and the record.**

- First choice: "**Dose:** E26's birth, 24,576 updates, for all eighteen runs. Thresholds are read as before, on the three-row version. Levels are means over rows 24,064, 24,320 and 24,576, and are also tabled at rows 11,776 to 12,288. In check 1 (b) the final `tensors_sha256` of the six runs at decay 0.25 are those of E26's births: sweep d9064126…, 1eb83bdd…, 70ae363b…; backpropagation efbdf87a…, 6c4ff9ce…, a5b5ab81… (seeds 73129, 93683, 116089). The final weights of the runs without decay are births for the retention step."
- If the dose stays at 12,288: the sentence of change 3 ends "…, within 12,288 updates, half of a birth", and "What it cannot show" gains: "**The second half of a birth.** At 12,288 updates the standing network is at 82 to 86% at 40 letters and reaches 95 to 96% by 24,576; and on the window network a decay of 0.1 or less let held-out accuracy peak early and drift while the taught pairs stayed at 100% (review 9.3)."
- In either case, "Why", third bullet: replace "(review 9.20 to 9.21)" by "(review 9.24, 9.25 and 9.27, for lives)", and add to the comparison with the record: "On the window network, with the decay of that time (1e-4 by review 9.3), the sign rule reached 95 to 98% at half of the pairs within 16,384 updates; at 15% of the pairs that decay gave 65 to 72% under backpropagation and 35 to 45% under the sign rule against 82 to 89% at a decay of 0.3 (review 9.2 and 9.3). So at half of the pairs the record expects no cost within this dose; its costs of weak decay were at few pairs and late."

**6. Cost.** Replace the paragraph by the figures of section 6, item 2, for the dose chosen: 8.1 CPU-hours at 12,288 updates, 16.2 at 24,576, for eighteen runs.

## 8. Recommendations (none holds the release)

1. **The length test's level over the last eight scored rows,** for every arm. A single row moves by 6 (sweep) to 11 (backpropagation) points from one row to the next; a three-row mean still ranges over 10 points under backpropagation within 3,000 updates.
2. **A quarter of the pairs at decay 0,** if a cell is added: both rules, three seeds, with its reference at 0.25 through the same runner. Twelve runs, 5.4 CPU-hours at 12,288 updates; six under backpropagation alone, 2.7. It can equally be a later job.
3. **Name the seven weight groups** (section 6, item 3), and read P46.8 on the recurrent matrix with its bias.
4. **Assert the sample digests** (section 6, item 4).
5. **Say what width the answer is for.** If E44 takes its learners at the half width, one half-width cell without decay is needed before a life without decay is run at that width.
6. **The first bullet of "Why"** says a learned family goes "to 18 to 52%" after 1,024 updates of decay alone. That is of a family receiving no gradient at all; in a life every weight receives some. The words "with no learning at all" are there and are enough; I note it only because the next bullet draws "no retention without rehearsal is possible" from it.

## 9. Can E46 be released?

Yes, after the six changes. Changes 1 to 4 and 6 are text. Change 5 is a choice between two texts; either is sufficient, and the first serves the plan better. No second audit is needed: check 1 as rewritten ties the runner to E41 and to E26 on every seed, and check 2 tests the one thing the runner adds.

## 10. What I did not check

- **Anything that needs torch.** The order of shrink and step in the host's AdamW is taken from torch's published form; check 2 as rewritten holds under either order. The first update was rebuilt in numpy on stand-in weights and gradients, not on the network's.
- **The runner.** `exp46_decay.py` does not exist yet.
- **The other two seeds' references.** Their thresholds on R have never been scored; E42's `G` receipts for them are not on its branch yet.
- **The review's figures on the window network** (9.2, 9.3, 9.24, 9.27) are quoted from the review, not recomputed.
