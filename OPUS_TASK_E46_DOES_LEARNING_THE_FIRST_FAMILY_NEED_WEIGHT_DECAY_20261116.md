# E46: does learning the first family need weight decay at all? (16 November 2026, revision 2)

Evidence level E2. This is the first check of the plan the user adopted on 16 November (review 9.148): weight decay alone removes whatever the stream stops pushing on, so before any retention experiment it must be known what decay buys. The specification and predictions are mine. Revision 1 (3126dead) was audited by the reviewer (`AUDIT_E46_WEIGHT_DECAY_BRIEF_20261116.md` on `claude/vibrant-cerf-sje726` at 6a208456), which required six changes and stated that after them no re-audit is needed. Revision 2 makes them; the last section lists them. **Released.** One worker, best an E41 worker, whose runner this copies.

Shell names: `D=origin/codex/minifly-consolidated-20261023; M=MOD97_LOCAL_RULE_20261106`. Host: Linux x86-64, Python 3.13.18, torch 2.15.0+cpu; at most three processes, one thread each; launch outside any wrapper with a time limit. Edit nothing existing. (The number E45 is skipped: `exp45` is the harness's own module and folder.)

## Why

- Every birth and life on file uses AdamW at rate 0.001 with decay 0.25. At that setting a weight that receives no gradient shrinks by 0.99975 per update, and that alone takes a learned family's reduction pairs to 18 to 52% of their level after 1,024 updates and to zero by about 2,000, with no learning at all (review 9.145; my rerun on two births).
- So no retention without rehearsal is possible at the standing setting, whatever else is done. The simplest remedy is no decay. Whether that costs anything is not on file for a birth.
- What is on file: a second family is acquired without decay as well as with it (E38, seed 73129, new family at 96.3 and 96.6 against 92.5 and 94.8). The decay of 0.25 was chosen on 7 November for the sign rule on another network (review 9.24, 9.25 and 9.27, for lives) and has been carried since.

## The comparison with the record (the working rule adopted on 16 November)

At decay 0.25, the present width and half of the pairs taught, accuracy A passes 80% between 3,886 and 6,899 updates in the seven births on file, and 90% by 12,288 in five of six (E41 audit). On seed 73129, E41's own runs (three-row version): 80 on A at 5,632 updates under the sweep and 5,120 under backpropagation; 80 on R at 6,656 and 6,400. So a dose of 12,288 locates the 80% threshold at the standing decay with room of about a factor of two, which is what a comparison of "falls, stays, rises" at 1.5 needs.

On the window network, with the decay of that time (1e-4 by review 9.3), the sign rule reached 95 to 98% at half of the pairs within 16,384 updates; at 15% of the pairs that decay gave 65 to 72% under backpropagation and 35 to 45% under the sign rule against 82 to 89% at a decay of 0.3 (review 9.2 and 9.3). So at half of the pairs the record expects no cost within this dose; its costs of weak decay were at few pairs and late. (The audit's reading of the review, which I had not cited.)

For the level at 40 letters the standing network is still rising at 12,288 updates (82 under the sweep and 86 under backpropagation at rows 11,776 to 12,288; 96 and 95 at the end of E26's birth), and single rows there move by 6 to 11 points. So the dose is a full birth.

## Runs

E41's present-width cell at taught fraction 0.50, exactly, with one thing changed: the decay of the optimizer's first group.

| Decay | Runs | Source |
|---|---|---|
| 0.25 | 6: both rules, three seeds | this runner. They are the reference for every comparison. |
| 0.05 | 6 | new |
| 0 | 6 | new |

Eighteen runs through one runner. The reference no longer depends on E41's two missing seeds, and it is scored exactly as the new runs are.

- **Code.** A copy of `exp41_scale.py` named `exp46_decay.py` with one added argument, the decay. After `opt = exp26_candidate.make_opt(m)`, which asserts the standing groups, set `opt.param_groups[0]["weight_decay"] = wd` and record the groups read back. The `optimizer` and `optimizer_read_back` fields are written with `weight_decay` equal to the run's own decay (`exp26_candidate.opt_record` writes the constant 0.25), and the receipt has `decay` as a field of its own. Keep E41's sample string `Random(f"e41|{seed}|50|{t[0]}")` unchanged and assert each receipt's `reduction_sample_sha256` against E41's check 7 table for half of the pairs. The embedding table has no decay in any arm.
- **Dose:** E26's birth, 24,576 updates, for all eighteen runs. Thresholds are read as before, on the three-row version. Levels are means over rows 24,064, 24,320 and 24,576, and are also tabled at rows 11,776 to 12,288. The final weights of the runs without decay are births for the retention step.
- **Data, scores, rows:** E41's at the present width: scored every 256, accuracies A, R and the taught panel, the same panels and sample.
- **Added score, every 256 updates:** the length test of E30 and E42, `exp11_cell.panel(t, exp11_cell.split(t, .5, seed)[1], seed, ks=(12, 40), n=300)` with `t = ("add", 89, b"p")`, since decay may matter for extrapolation and not for accuracy in range.
- **Added record, at every scored row:** the norm of each of the seven weight groups (byte table, `l1`, input map, recurrent matrix with its bias, `lm`, `l2`, `l3`), to show what the weights do without decay.
- **Committed:** final weights of every run (`git add -f`; confirm they are on the remote), receipts in `exp45/e46/e46_<S|B>_wd<000|005|025>_<seed>.json`.

About 16.2 CPU-hours for the eighteen runs at the full dose (the audit's figure from E41's receipts and E42's scoring with the length test).

## Checks before the full runs

1. **With decay 0.25 the copy is E41 and E26.** (a) A dry run of 2,048 updates at decay 0.25 on seed 73129, with every added score on, has the `tensors_sha256` of E26's dry births (79618ab2… under the sweep, adf933d7… under backpropagation). (b) The two full runs at decay 0.25 of seed 73129 have, at every row up to 12,288, the A, R and taught counts of `e41_<S|B>_h512_f50_73129.json`. The final `tensors_sha256` of the six runs at decay 0.25 are those of E26's births: sweep d9064126…, 1eb83bdd…, 70ae363b…; backpropagation efbdf87a…, 6c4ff9ce…, a5b5ab81… (seeds 73129, 93683, 116089). (c) For each of the six runs at decay 0.25, E26's panel at rows 4,096 and 8,192 equals the `sealed` counts of `e26_dev_<S|B>_<seed>.json` (E41's check 8). (d) Reported, not a failure if the branch is not at hand: the length-test counts of seed 73129's two runs equal those of `e42_G_<S|B>_73129.json` on `claude/gallant-dijkstra-i9zgsr` at every row.
2. **The decay is what is set.** On the first batch, one update from the same start is made at decay d (0.25, and again 0.05) and at decay 0, under backpropagation. (a) The byte table is bit-equal in the two. (b) For every other tensor, with p0 the start: d̂ = ⟨p1(0) − p1(d), p0⟩ / (0.001 ⟨p0, p0⟩) is within 1% of d. (c) Reported: the largest value of |p1(0) − p1(d) − 0.001 d p0| over each tensor, which is a few times 1e-8 when the shrink precedes the step. (d) The groups read back show d and 0.0 at the start and at the end of every run.
3. **Same data in every arm:** the batch digest at row 4,096 is E26's for the seed (bbd71ab7…, 65b1c7e0…, 1df203fb…).
4. **The length test's code:** on E26's six final files it gives E30's figures at 12 and 40 letters (`exp45/e30/E30_UNSEEN.json`), as E42's check 12.
5. One dry run of 256 updates per decay, with CPU time and a forecast.

## How the result is stated (fixed beforehand, E41's forms)

For each rule and seed, decay 0 against 0.25 and decay 0.05 against 0.25:

- **Updates to the 80 threshold** on A and on R, three-row version: E41's table of four cases, read as fewer (*falls*, ratio below 2/3), more (*rises*, above 1.5) or about the same (*stays*), with its rules for a run that does not reach the threshold.
- **The level:** the mean of A, of R and of the length test at 40 letters over rows 24,064, 24,320 and 24,576; *lower* or *higher* if the difference is 15 points or more, otherwise *about the same*. Tabled beside it: the same at rows 11,776 to 12,288, and the length test's mean over the last eight scored rows, since a single row there moves by 6 to 11 points.
- A cell carries the description two of its three seeds share; otherwise *mixed*.

**The sentence this job licenses.** For each rule, three cells are formed from decay 0 against 0.25, each carrying the description two of its three seeds share: the threshold cell (updates to 80 on R, E41's table of four cases, in which a run that does not reach the threshold is *rises* if the reference reached it by the last recordable row over 1.5, and otherwise *not decided*), the level of R, and the level at 40 letters. Learning the first family **needs decay for** each cell that is *rises* or *lower*, and the sentence names those cells. It **does not need** it if the threshold cell is *falls* or *stays* and both level cells are *about the same* or *higher*. Otherwise it is **not decided**, with the cells given. The runs at 0.05 are described in the same three cells and enter no sentence. Nothing is said about lives or about other families.

## What it cannot show

- **A life.** Without decay the weights may grow over 36,864 updates in ways a birth of 24,576 does not show; the norms recorded are the first evidence either way.
- **Composition**, which births do not teach.
- **Few taught pairs.** The record's cost of weak decay was at 15% of the pairs; this job is at half. A quarter of the pairs at decay 0 is the cell to add later if it matters (the audit's recommendation).
- **Another width.** The answer is for the present width. If the long-life world takes its learners at the half width, one half-width cell without decay is needed before a life without decay is run there.
- **Other rates.** Decay acts through rate × decay; only the rate 0.001 is run.
- **That removing decay gives retention.** It removes one cause of loss. E43 shows a second, faster one.

## Predictions (mine)

| | Prediction | Chance |
|---|---|---|
| P46.1 | Check 1 passes | 90% |
| P46.2 | Decay 0 reaches 80 on R within the dose in two of three seeds, under backpropagation | 80% |
| P46.3 | The same under the sweep | 65% |
| P46.4 | The sentence under backpropagation: does not need 50%, not decided 30%, needs (some cell) 20% | |
| P46.5 | The sentence under the sweep: does not need 40%, not decided 30%, needs (some cell) 30% | |
| P46.6 | At 40 letters decay 0 is *lower* than 0.25 at the end of the birth, two of three seeds, under the sweep | 25% |
| P46.7 | Decay 0.05 is *about the same* as 0.25 on every description, under each rule | 55% |
| P46.8 | Without decay the norm of the recurrent matrix with its bias at row 24,576 is at least 1.5 times its norm under decay 0.25, two of three seeds, under each rule | 70% |

## Done

`exp46_decay.py`, the check receipts, the eighteen runs with receipts and final weights on the remote, and `E46_REPORT.md` with the tables (rule × seed × decay: updates to 50, 80 and 90 on A and R in both versions; the three levels; the norms at the end), the descriptions, the sentence per rule and the outcome of each prediction. No readings beyond the sentence above.

## What changed from revision 1 (the audit)

Made as the audit wrote them, with the first choice of dose.

| | Change | Required? |
|---|---|---|
| 1 | The reference at decay 0.25 is run by this runner, as a third arm; check 1 anchors it to E41 and to E26's six births | yes |
| 2 | **Check 2 was wrong.** AdamW shrinks a weight and then steps, so two runs differ after one update by rate × decay × the weight added, not by a factor. My relation would have failed on a correct runner. The check now estimates the decay from the difference. | yes |
| 3 | The sentence names its cell; a run that does not reach the threshold follows E41's table; the third outcome is "not decided"; the 0.05 runs enter no sentence | yes |
| 4 | The receipt records the run's own decay; the harness's recording function writes the constant 0.25 | yes |
| 5 | The dose is a full birth, 24,576 updates: at half a birth the standing network is still rising at 40 letters. Two citations to the review corrected, and what the record already says about decay added. | yes |
| 6 | Cost: 16.2 CPU-hours for the eighteen runs | yes |
| — | The length test over the last eight rows; the seven weight groups named; the sample digests asserted; the width the answer is for | recommended |
