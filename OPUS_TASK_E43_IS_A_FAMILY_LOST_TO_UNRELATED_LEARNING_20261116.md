# E43: is a family lost when the learner then learns something unrelated? (16 November 2026, revision 2)

Evidence level E2. The question, the arms' outline and the reading are the outside reviewer's (its memo, revision 14, section 24.5, on `claude/vibrant-cerf-sje726` at 57497920), written after the user's objection of 16 November. This specification, the unrelated task and the predictions are mine. Revision 1 (75613729) was audited by the reviewer (`AUDIT_E43_UNRELATED_LEARNING_BRIEF_20261116.md` on `claude/vibrant-cerf-sje726` at 599ab3b2), which required five changes and stated that after them no re-audit is needed. Revision 2 makes them and the recommendations; the last section lists them. **Released.** One worker, best the E38 worker, whose runner already does the `conflict` arms.

Shell names: `D=origin/codex/minifly-consolidated-20261023; M=MOD97_LOCAL_RULE_20261106; K=PROJECT_DOCUMENTS/COORD_CHECKS_20261116`. Host: Linux x86-64, Python 3.13.18, torch 2.15.0+cpu; at most three processes, one thread each; launch outside any wrapper with a time limit. Edit nothing existing.

## The objection this answers

The user, 16 November: by the three findings so far, Einstein would have to review world geography at intervals in order to recall his life in Germany. Long-term memory exists as a matter of fact; what have we left out?

What has been measured is only the most adverse case. Every life on file consists of families that contradict the earlier ones: the same operands and the same operation symbol, another modulus. There the old family's modulus is gone within 64 to 256 updates (E38), under every rule, with or without decay. The user's example is the other case: later learning that neither repeats nor contradicts the memory. It has never been measured here.

## What is on file

- **E38, seed 73129, the first family's reduction pairs (whole answers, no letters) while the second family is learned:** 97% at the start, 18% after 64 updates, 3% after 256, 0% after 1,024, alike with decay 0.25 (`zero`) and with none (`none`), under the sweep and under backpropagation.
- **At the end of the nine lives** (the reviewer's read of the saved files; I did not rerun it): reduction pairs 0 to 1.7%; pairs without reduction 2 to 93%; max or min at 92% or more in 8 of 18 cases and at 66% or less in the other 10, the surviving operation going with the seed and not with the rule.

## Arms: four, from each of the six E26 births

Starts: `exp45/e26/e26_dev_S_<seed>.pt` continued under the sweep and `e26_dev_B_<seed>.pt` under backpropagation, three seeds, each admitted by `exp26_candidate.admit_e26`. Update: `exp26_candidate.step`'s text; a fresh `exp26_candidate.make_opt(m)`; 4,096 updates. 24 runs.

| Arm | Stream | Decay |
|---|---|---|
| `conflict` | the second family alone (mod 97): `X16.entry_stream(X16.family_entries("F1", seed), seed + 101)`, as the first block of E26's life | none (`weight_decay` 0 in group 0, as E38's arm `none`) |
| `conflict-decay` | the same | 0.25, as E26's life and E38's arm `zero` |
| `unrelated` | the task below | none |
| `unrelated-decay` | the same | 0.25 |

The memo names the two arms without decay. I add the two with decay because 0.25 is the standing setting of every life.

**Decay alone, with no learning, removes the first family** (the audit's main finding). Its reduction score is at 95 to 98% of its starting level after 256 updates, 18 to 49% after 1,024, 0 to 3% after 2,048 and zero after 4,096, in all six births (the reviewer's computation, audit section 3). I reran it on two births with all reduction pairs: 95.2, 17.9, 0.5 and 0.0% for the sweep's birth of seed 73129, and 97.4, 51.9, 4.2 and 0.0% for backpropagation's of seed 93683 (`$K/decay_alone_check.py`, output beside it). So under the standing decay a family that receives no gradient is gone within about 2,000 updates whatever else is or is not learned, and the arm under decay asks whether the new learning holds anything up against that.

A sentence of revision 1 is withdrawn: that decay is the one thing acting on weights the new task does not use. The only weights the new task leaves exactly alone are the byte table's rows of other bytes, and those have no decay; every other weight takes part in every item.

## The unrelated task, as code

**`$K/e43_unrelated_task_spec.py` on the default branch is the specification.** Copy it to `$M/e43_task.py` and import it; do not rewrite it. In words:

- **Bytes.** 64 symbols (bytes 128 to 191), 32 fillers (192 to 223), a code byte (224) and a query byte (225). No family's prompt or answer uses any of them; the families use digits, `+ - > < = ;`, upper-case letters and some lower-case codes, all below 128.
- **Item.** `code s1 s2 f_1 … f_k query`, then the answer `s1 s2` and the end symbol. k is 0 to 12, as in the families. The answer is two bytes and the end symbol, so the harness, the answer mask and the scorer are unchanged.
- **Pairs.** The 4,096 ordered pairs of symbols are shuffled by `Random(f"e43|{seed}|split")`; the first 2,048 are taught, the rest held out.
- **Stream.** `Random(f"e43|{seed}|stream")`: the taught pairs in a fresh shuffled order each pass; for each item `randint(0, 12)` for k, then k fillers by `choice`. Batches of 64 consecutive items.
- **Panels.** `Random(f"e43|{seed}|panel|held")` samples 300 held-out pairs and renders each at k = 12; the same with `taught`.

**What it shares with the first family, stated plainly.** No input byte. It does share: the whole network, since every weight but the embedding rows of other bytes takes part; the start symbol; the end symbol as the third answer byte; the length of items and the format of two answer bytes. It asks for memory across fillers, as the families do across letters. So it is unrelated in content and not in machinery, which is the user's case. **A reading of *kept* would have two explanations** (audit section 1): updating does not disturb what is not contradicted, or the new task holds the old machinery up by using it, since it asks for the same skill of carrying two items across irrelevant bytes.

**What I checked without training** (`$K/OUTPUT_E43_SPEC_MAC.txt`, on E26's sweep birth of seed 73129): the taught and held-out pairs are disjoint; the items pass through `exp45.batch_tensor` and `exp11_cell.score` unchanged; the born network scores 0 of 300 on the held-out panel before any update, so there is no starting level to subtract for the new task; the first-batch loss is 18.2 with finite gradients. The embedding rows of the task's bytes are bit-equal between the sweep's and backpropagation's births of a seed and between each birth and its lives (the audit's check), so no training has moved them. The difference of 1.4e-6 that my Mac output shows against the initial network is a difference between hosts in the initial values, not a movement in training.

## Rows and scores

Score at updates 0, 64, 256, 1,024, 2,048 and 4,096. Save weights at each; commit those at 4,096 (24 files) and, for the reviewer's analysis, rows 64 and 1,024 of seed 73129's `unrelated` arm under each rule; record the sha256 of the rest.

1. **First family, all pairs without letters, split by reduction** (as E36's score 2): add and sub, pairs with a + b ≥ 89 or a − b < 0 against the rest; and max and min on all pairs. **The reduction score** is the mean of the add and sub reduction-pair shares.
2. **First family, held out at 12 letters:** `exp16_integrated.panels(seed)[0]` entries `add89`, `sub89`, `max89`, `min89`.
3. **The new task:** for the `unrelated` arms, the held-out and taught panels above; for the `conflict` arms, `add97` and `sub97` at 12 letters from the same `panels` call.
4. **How much the weights moved:** for each of the groups (embedding, `l1`, input map, recurrent matrix, `lm`, `l2`, `l3`), the norm of the difference from the birth divided by the norm at birth, and the cosine between its weights and the birth's. Under decay the norm of the difference is at least 0.64 of the birth's norm from shrinking alone, so there the cosine is the figure to read.
5. **Decay alone**, no run: the birth file with every parameter but the byte table multiplied by (1 − 0.00025)^t, scored by score 1 at t = 64, 256, 1,024, 2,048 and 4,096. Its reduction scores should be within 3 points of the audit's table.
6. **A functional measure (recommended by the audit):** the mean absolute change of the first family's state at the `=` position between row 0 and each row, on the panel of score 2, divided by the mean absolute state at row 0.

Score 4 describes how much each arm changed the weights. It is not a guard: a change of a few percent of a matrix's size has been enough to teach a family and remove another (memo, section 23). The guard is the condition below.

## Checks before the full runs

1. **The `conflict` arms are E38's.** On seed 73129, `conflict-decay` has E38's `zero` row hashes and `conflict` has E38's `none` row hashes at updates 64, 256, 1,024 and 4,096, under each rule (`tensors_sha256` in `exp45/e38/e38_<zero|none>_<S|B>_73129.json`). They begin:

   | | 64 | 256 | 1,024 | 4,096 |
   |---|---|---|---|---|
   | `zero`, sweep | f3819049 | 95f5de74 | e2d170be | ef1ac53c |
   | `none`, sweep | 5afe5e0a | 166dd91e | 08e44428 | 23f6ddaa |
   | `zero`, backpropagation | 35f0d6c9 | a5cad080 | 676c2fa0 | ce901c57 |
   | `none`, backpropagation | 5abf61c9 | 7789d013 | 8b9c6649 | a423a30b |
2. **Row 0** of every run has the birth file's `tensors_sha256`, and its first-family scores equal E38's row 0 for seed 73129 where E38 has them: add and sub on all pairs, and the four held-out panels.
3. **The specification file.** `$M/e43_task.py` is byte-identical to `$K/e43_unrelated_task_spec.py`. Run on `$K`, the self-checks print the Mac output's three seed lines, its line on shared bytes, its batch-tensor line and its score line exactly. The line on the byte table's rows should read 0.0e+00 on the worker's host: the 1.4e-06 of the Mac output is a difference between hosts in the initial values (audit section 2); report what it reads. The first-batch loss should agree to two decimals. The runner also prints, for the three seeds, `sha256(repr(panel(seed, 'taught')))` and `digest(seed, 4096)`; their first sixteen digits equal the audit's:

   | Seed | Taught panel | Stream, 4,096 updates |
   |---|---|---|
   | 73129 | 3583cd943452ff06 | d3410fc2a8850bc1 |
   | 93683 | 348392b8f4017512 | edbbcc1f0dc94926 |
   | 116089 | 301fd99ddfa152f8 | fe35d695eace801d |
4. **No shared input.** Asserted on every batch of the `unrelated` arms: every prompt byte is 128 or above. And on every first-family panel item: every prompt byte, and every answer byte but the end symbol, is below 128.
5. **The same batches** for the four runs of a seed that share a stream: the batch digest is equal at every row.
6. **`unrelated` has no decay:** the optimizer's groups read back with `weight_decay` 0.0 in both; each arm's decay is recorded in a field of its own.
7. One dry run of 64 updates per arm under the sweep, with CPU time and a forecast (the reviewer's estimate: about 3.5 to 4 CPU-hours with scoring).

Outputs go to `exp45/e43/`, named `e43_<arm>_<S|B>_<seed>` with `_row<updates>`; the runner calls `arch26.install()` first and refuses an existing path.

## Reading (fixed beforehand; the reviewer's, with one condition added)

**The condition.** The reading is taken for a rule only if the unrelated task was learned: in at least two of three seeds the `unrelated` arm's **taught** panel is at 80% or more. The held-out panel is reported beside it. If the condition does not hold at row 4,096, the two `unrelated` arms of that rule continue to 16,384 updates, scored at 8,192 and 16,384, and the reading is taken at the first of those rows at which it holds, with the row stated. If it holds at neither, the report says that the new task was not learned and draws nothing about retention.

**Per rule, on the `unrelated` arm**, with each run's reduction score at row 4,096 (or the later row the condition names) as a share of its own at row 0, in at least two of three seeds:

| Share kept | Reading |
|---|---|
| 80% or more | **Kept.** The fast loss measured so far is loss by contradiction; the learner holds what later learning does not contradict, over this dose. |
| 20% or less | **Lost.** Updating every weight removes even unrelated knowledge; separation of what is learned becomes the first thing to test. |
| Otherwise, or the seeds disagree | **Partly kept.** The shares are given per seed. |

- `conflict` is the reference and is expected at zero by row 1,024.
- `unrelated-decay` is read against decay alone at rows 256, 1,024 and 2,048, per rule, in at least two of three seeds: *held up* if its reduction score is 15 points or more above decay alone's at row 1,024; *not held up* if within 15 points at all three rows; otherwise the differences are given. It is reported as "under the standing decay". Above decay alone means the new learning holds part of the family up; below it, that the new learning removes more than decay does.
- **The next step, named now:** if the reading is *kept* or *partly kept*, a second unrelated task without carrying across a gap follows, to separate "not disturbed" from "held up by use": for example one symbol and the query byte, answered by two symbols fixed by a table drawn once.
- Reported beside the reading, as facts: the same share for pairs without reduction, for max and for min; the share at rows 64, 256 and 1,024; score 4 for all four arms.

## What it cannot show

- **A dose of 4,096 updates.** A life is 36,864. "Kept" here is kept over a ninth of a life.
- **One unrelated task**, a simple one. A harder unrelated task would move the weights more.
- **Long-term memory as the user means it.** Nothing here is a fact stored once and recalled by a cue; the first family is a rule computed by a shared circuit. This job separates loss by contradiction from loss by any further learning, and no more.
- **The optimizer.** Every run starts a fresh AdamW, whose first steps move every weight by about the learning rate whatever the size of its gradient. A loss here may owe as much to that as to the new task.
- **Why.** If the family is kept, the job does not say what protects it; if lost, not which weights carried the loss.

## Predictions (mine)

| | Prediction | Chance |
|---|---|---|
| P43.1 | Check 1 passes under both rules | 85% |
| P43.2 | The unrelated task generalises (held-out panel at 80 or more at row 4,096, two of three seeds) under each rule | 60% |
| P43.3 | `unrelated`, sweep: kept 25%, partly kept 45%, lost 30% | |
| P43.4 | Under `unrelated` the reduction score is above `conflict`'s by at least 30 points at row 1,024, two of three seeds, under each rule | 75% |
| P43.5 | `unrelated-decay`'s reduction score is 15 points or more above decay alone's at row 1,024, two of three seeds, under the sweep (restated after the audit: the old form was settled beforehand by decay alone) | 30% |
| P43.9 | The condition holds at row 4,096 on the taught panel under each rule | 75% |
| P43.6 | The reading is the same under the sweep and under backpropagation | 75% |
| P43.7 | Under `unrelated`, max and min keep a larger share than the reduction pairs, two of three seeds, under the sweep | 60% |
| P43.8 | The recurrent matrix moves less under `unrelated` than under `conflict` at row 4,096 (score 4), two of three seeds, under each rule | 60% |

## Done

Runner, check receipts, 24 runs with receipts and the committed weights, and `E43_REPORT.md` with the curves (run × row: the four scores), whether the condition holds, the reading per rule, the facts listed, and the outcome of each prediction. No readings beyond the table above.

## What changed from revision 1 (the audit)

Made as the audit wrote them. I reran its main finding on two births (decay alone), and it holds.

| | Change | Required? |
|---|---|---|
| 1 | **Decay alone removes the family within about 2,000 updates.** I had written that decay "could" act on unused weights and that E38 could not say because the conflict removed the family first, and I had not computed what decay alone does, which needs no training. The arm under decay is now read against decay alone at rows 256, 1,024 and 2,048; decay alone is scored; row 2,048 is added; P43.5 is restated. | yes |
| 2 | The condition is on the taught panel, and a failed condition continues the `unrelated` arms to 16,384 updates before anything is given up | yes |
| 3 | Check 3 says what the worker's host should print, and adds digests of the taught panel and of the stream to 4,096 updates | yes |
| 4 | Score 4 gains the cosine and is described as a description, not a guard | yes |
| 5 | Cost 3.5 to 4 CPU-hours; check 2 limited to what E38 holds; check 4's wording on the end symbol | yes |
| — | The second explanation of *kept*; the next step named; a functional measure; four more committed files; the optimizer under "cannot show"; P43.2 reworded and P43.9 added | recommended |

---

# Addendum 1 (16 November 2026): E43 read; three additions (the reviewer's memo, revision 16, section 24.7)

## As I read it

Branch `claude/peaceful-hopper-8ofq4i` (ac56179d), merged. The figures are the worker's; the reviewer recomputed them from the 24 receipts. I recomputed none.

- **The registered reading is *lost*, under both rules.** After 4,096 updates on the unrelated task without decay the first family's reduction pairs are at 0.0% of their starting level in all six networks, and so are pairs without reduction, max and min. The condition holds: the new task is at 100% on taught and held-out pairs.
- **The loss is complete after 64 updates**, before the new task scores anything (shares 0.0 to 0.1, one network at 5.0). Under `conflict` at the same point pairs without reduction, max and min are still at 49 to 98%.
- **`unrelated-decay` is below decay alone** at 256, 1,024 and 2,048 updates in all six (by 17 to 51 points at 1,024): the new learning removes more than decay does. Neither of the brief's two readings for that arm applies, so the differences stand as given.
- **Decay alone** reproduces the audit's table within 3 points in all six births.
- P43.1, P43.2, P43.6, P43.8 and P43.9 hold; P43.4, P43.5 and P43.7 miss; P43.3's outcome is *lost*, to which I gave 30%.

## What the reading does not settle

The brief's sentence for *lost* says that updating every weight removes even unrelated knowledge. The figures do not show that the knowledge is removed. The answers become the end symbol alone, which the unrelated task rewards as its third answer byte and which no first-family prompt had needed before its two digits. Three explanations remain (the reviewer's):

1. the knowledge is still in the network and is not expressed, the shared readout having been driven away from digits for every input;
2. the optimizer's first steps: a fresh AdamW moves every weight by about the learning rate whatever the size of its gradient, from a loss of 18;
3. the function is overwritten, as it would be under any update that learns the new task in this network.

I should have asked what the first family's *answers* would turn into, not only how often they would be right. The brief scored whole answers and had no measure that separates a lost computation from a redirected readout.

## The three additions

**1. Push the weight files.** The report lists update 4,096 of every run and updates 64 and 1,024 of seed 73129's `unrelated` runs as committed. They are not on the remote: `exp45/e43/.gitignore` ignores `*.pt`. Add them with `git add -f`, after checking each against the `weights_sha256` in its receipt, and push. The reviewer will read them only.

**2. A reminder, as E36's.** From the six `unrelated` networks at update 4,096 (no decay), 1,024 updates on the first family.

- **Runner:** `e36_reminder.py`'s reminder path with the start file replaced; the rule of the run that made the file; a fresh `exp26_candidate.make_opt(m)`.
- **Data:** E36's, the same batches as its runs of that seed: `exp11_cell.batches([(t, exp11_cell.split(t, .5, seed)[0]) for t in exp11_cell.DEV], seed + 36, 12)`. So E36's fresh references (`exp45/e36/e36_reference_<S|B>_<seed>.json`) are the references here and are not rerun.
- **Rows and scores:** 0, 16, 32, 64, 128, 256 and 1,024 updates; E36's three scores. Commit the weights at 128 and 1,024.
- **Checks:** row 0 has the start file's `tensors_sha256` and E43's row-4,096 first-family scores; the batch digest at row 1,024 equals E36's for that seed.
- **Reading, fixed now, per rule, in at least two of three seeds:**

  | Result | Reading |
  |---|---|
  | The mean of held-out add and sub at 12 letters is at least 85 at row 128 | **Back within about a hundred updates.** What the birth learned was not destroyed by the unrelated task; explanation 1 or 2 holds. E36's lived networks did the same after a life (row 64 to 128). |
  | Not that, and at row 1,024 the wrap score is at least 30 points above the fresh reference's | **Partly back.** |
  | At row 1,024 the wrap score is less than 15 points from the fresh reference's (1.1 to 2.5) | **Not back:** the function was overwritten (explanation 3). |
  | Anything else | Mixed; the curves are given. |

  The networks start at zero on every first-family score and so do the fresh references, so no starting level has to be subtracted.

**3. One arm under plain gradient steps, `unrelated-plain`.** As `unrelated` (no decay), with `torch.optim.SGD(m.parameters(), lr=LR)` and no momentum in place of AdamW, so that each weight moves in proportion to its own gradient.

- **The rate.** Three runs on seed 73129 under backpropagation at LR = 0.01, 0.03 and 0.1, to 4,096 updates. The rate used for all six runs is the smallest at which the taught panel is at 80% or more at row 4,096. If none reaches it, or a loss is not finite, report and stop this addition; nothing about retention is read from it.
- **Runs:** the six births at that rate, each under its own rule (the sweep's gradients with a plain step, or backpropagation's), rows and scores as the `unrelated` arm. Commit the weights at 64 and 4,096.
- **Reading:** the brief's condition and table, unchanged (kept at 80% or more of the starting level, lost at 20% or less), and beside it the share at rows 64, 256 and 1,024. If the first family is kept here and lost under AdamW, the equal-sized step on every weight is the cause. If it is lost here too, it is not.

About 2 CPU-hours for the three additions.

## Predictions (mine)

| | Prediction | Chance |
|---|---|---|
| P43.10 | Reminder: back at row 128 (held-out mean at least 85), two of three seeds, under the sweep | 50% |
| P43.11 | Reminder: at row 1,024 the wrap score is at least 30 above the fresh reference's, two of three seeds, under each rule | 85% |
| P43.12 | Some rate of 0.01, 0.03, 0.1 meets the condition on seed 73129 under backpropagation | 70% |
| P43.13 | `unrelated-plain`, backpropagation: kept 15%, partly kept 30%, lost 55% | |
| P43.14 | Under `unrelated-plain` the share at row 64 is above 50% in two of three seeds, under backpropagation (the loss is slower than under AdamW, whatever its end) | 55% |

## Done

The pushed weight files; the six reminder runs and the plain-step runs with receipts; an addendum to `E43_REPORT.md` with the reminder's curves and reading, the rate chosen, the plain arm's curves and reading, and P43.10 to P43.14. No readings beyond the tables above.

---

# Addendum 2 (16 November 2026): the three additions read; two further additions (the reviewer's memo, revision 19, section 24.8)

## As read so far

Branch `claude/peaceful-hopper-8ofq4i` (722cb621), **not yet merged**: the plain-step runs under the sweep are being continued to 16,384 updates. The figures are the reviewer's read of the receipts and its own measurement on the weight files; I have recomputed none and will read the report when the branch is complete.

- **The reminder.** After 4,096 unrelated updates the first family is not back within a hundred updates of reminding in any of the six networks: reduction pairs at 0.0 to 1.4% after 64 updates and 1.0 to 5.8% after 256. At 1,024 updates they are at 91.3, 10.7 and 69.3% under the sweep and 91.3, 14.4 and 42.0% under backpropagation, where a fresh network is at 1.1 to 2.5%. So something is saved, and the family was not merely unexpressed. Neither of my readings "back within about a hundred updates" (P43.10) holds.
- **Plain gradient steps** at rate 0.03. Under backpropagation the family is at 0.4 to 2.8% after 64 updates and at zero from 256 on. So AdamW's equal-sized steps are not the cause.
- **Where the loss sits** (the reviewer's measurement, seed 73129): with the readout put back to the birth's, 3.7 and 0.7%; the birth with only its recurrent matrix changed is at 34.8 and 30.6% after 64 updates and about 1% after 1,024. The function is overwritten throughout the network, most of all in the recurrent matrix.

Of the three explanations in addendum 1, the third holds.

## Two further additions

**4. The conventional reference: `unrelated-rehearse` and `unrelated-rehearse-decay`.** The `unrelated` arm with items of the first family mixed into the stream, on the six births, without decay and with the standing 0.25: 12 runs of 4,096 updates.

- **Stream.** Each item is drawn from one of two streams with equal chance, by `Random(f"e43|{seed}|joint")`: the unrelated stream of the brief, and the first family's taught half as E38's rehearsal stream, `X16.entry_stream([("op", t, a, b) for t in exp11_cell.DEV for a, b in exp11_cell.split(t, .5, seed)[0]], seed + 100)`. Batches of 64. Record the count of items of each kind.
- **Rows and scores:** as the `unrelated` arm, with row 2,048.
- **Reading, fixed now, per rule and decay, in two of three seeds:** *kept at no cost* if at row 4,096 the first family's reduction score is at 80% or more of its starting level and the unrelated task's taught panel is at 80% or more; *kept at a cost* if the first holds and the second does not; *not kept* if the reduction score is at 20% or less; otherwise the figures are given. Beside it: the row at which the unrelated task's taught panel first reaches 80%, against the `unrelated` arm's, per update and per item of the unrelated task.

**5. E42's networks: the `unrelated` arm from `F10` and `P8` of seed 73129**, under each rule, with E42's forward: 4 runs of 4,096 updates, no decay, from E42's final weights and structure of that seed on `claude/gallant-dijkstra-i9zgsr`. Connections stay as they are at the end of E42's run; no replacement event is made. Rows, scores and reading are the `unrelated` arm's. In these networks every cell still reads the whole byte and the readout is shared, so the expectation is *lost*; the run replaces it by a figure. Row 0's first-family scores must equal the E42 receipt's final row where it has them.

About 2.5 CPU-hours for the sixteen runs.

## Predictions (mine)

| | Prediction | Chance |
|---|---|---|
| P43.15 | `unrelated-rehearse`, sweep: *kept at no cost* | 60% |
| P43.16 | `unrelated-rehearse-decay`, sweep: *kept at no cost* | 65% |
| P43.17 | With rehearsal the unrelated task's taught panel reaches 80% no more than twice as many updates later than without, two of three seeds, under each rule | 60% |
| P43.18 | E42's `F10` and `P8`: *lost* in all four runs | 85% |

## Done

The sixteen runs with receipts and weights at 4,096 on the remote (`git add -f`, listed from `git ls-tree`); an addendum to `E43_REPORT.md` with the curves, the readings, and P43.15 to P43.18. No readings beyond the tables above.
