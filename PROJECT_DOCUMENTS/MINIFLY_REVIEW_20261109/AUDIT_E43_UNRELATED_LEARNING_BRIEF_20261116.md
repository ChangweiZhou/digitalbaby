# Audit of the E43 brief and its specification file at 75613729 (16 November 2026)

From the outside reviewer. Read-only: nothing was trained, torch is not installed in the reviewer's container, and no code from another branch was executed. The brief is `OPUS_TASK_E43_IS_A_FAMILY_LOST_TO_UNRELATED_LEARNING_20261116.md`, revision 1. The specification file is `PROJECT_DOCUMENTS/COORD_CHECKS_20261116/e43_unrelated_task_spec.py`, with the coordinator's output `OUTPUT_E43_SPEC_MAC.txt` beside it. They implement section 24.5 of the reviewer's memo (revision 14, at 57497920).

**Verdict.** The unrelated task is the one the memo asked for, it shares no input byte with any family, and its stream and panels are reproduced by a second reader written from the brief's words. Five changes are required before release. The first is the important one: **weight decay alone, with no learning at all, removes the first family completely within about 2,000 updates**, so the arm under decay cannot be read as the brief reads it. After the five changes no re-audit is needed.

**Terms.**

- The *first family* is the four operations mod 89 that a birth teaches. Its *reduction pairs* are the pairs of add and sub whose answer needs the modulus; the *reduction score* is the mean of add and sub on them.
- *Decay* is the optimizer's weight decay: at every update each parameter but the byte table is multiplied by 1 − 0.001 × 0.25 before the learning step. The *standing* decay is 0.25, the setting of every birth and life on file.
- *Decay alone* is not a run. It is a saved birth with every parameter but the byte table multiplied by (1 − 0.00025)^t, which is exactly what t updates do to a weight that receives no gradient.
- The *reader* is `e43_unrelated_reader.py` beside this audit: the unrelated task written from the brief's words, standard library only.

Shell names: `D`, `M` and `K` as in the brief; `R=PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109`; `C=$R/e43_brief_checks.py` on `claude/vibrant-cerf-sje726`. `python3 -I $C <letter>` runs one part (A to G).

## 1. Is the task unrelated? (point 1)

**No input byte is shared.** I rebuilt the families' bytes from the harness's source, not from the brief: the digits, the four operation symbols and the 26 letters of `exp11_cell.py`, `=`, `;` and the composition code `c` of `exp16_integrated.py`, and every code byte in a task tuple of any `.py` file of the harness folder (`e f g h n p q r s t u v w x y z`). That is 59 bytes, the largest 122. The task's 98 bytes run from 128 to 225. The specification file's own list equals mine. (`python3 -I $C A`)

**The saved files agree.** A row of the byte table moves only when its byte is shown. In each seed the rows that differ between the sweep's birth and backpropagation's birth are exactly the 45 bytes of the first family, the rows a life moves beyond them are exactly `; c n r s t u x y z`, and 203 rows have never moved, among them all of 128 to 225. (`python3 -I $C B`)

**What it shares, and what that means for the reading.** The brief states it correctly: the whole network, the start and end symbols, the format of two answer bytes and the end symbol, and memory across fillers. I add one consequence. The task asks for the same skill the first family needs: to carry two items across irrelevant bytes and give them back. It neither repeats nor contradicts the first family's content, which is Mr. Xu's case. But it exercises machinery the first family uses, as a later family's max and min exercise the first family's (in E38's arm `zero`, max89 and min89 stay between 84 and 100% for 12,288 updates while the second family, which has a max and a min of its own, is learned; `git show $D:$M/exp45/e38/E38_READ_TABLES.md`).

So the three outcomes are not equally clear:

- *Lost* is clear: even learning that reuses the same skill removes the family.
- *Kept* has two explanations: updating does not disturb what is not contradicted, or the new task holds the old machinery up by using it.

A second unrelated task with another structure (no carrying across a gap) would separate the two. It is not needed for release. It should be named in the brief as the next step if the reading is *kept* or *partly kept* (recommendation 1).

## 2. The byte table's rows and the difference of 1.4e-6

The coordinator's output says the task's rows differ by at most 1.4e-6 between the saved birth and the initial network.

- **No training moved them.** In all three seeds rows 128 to 225 are bit-equal between the sweep's birth and backpropagation's birth, and between each birth and its lives (`python3 -I $C B`). Two different trainings cannot move 784 numbers identically; the byte table has no decay; a row whose byte is never shown has a gradient of exactly zero. So the births hold these rows at the initial values of the host that made them.
- **The difference is therefore between hosts.** The coordinator built the initial network on the Mac; the births were made on cloud hosts. The largest entry in these rows is about 3, where one step of a 32-bit float is 2.4e-7, so 1.4e-6 is about six steps: the size expected when two machines draw the same normal values with different arithmetic. The record already has this between the two classes of host (review 9.101 B). I could not draw torch's initial values, so this is an inference from the files, not a reproduction.
- **It does not affect the experiment.** Every run starts from a saved birth on the worker's host. It does affect check 3 as written (section 6).

## 3. Decay: what the arm under decay can and cannot show (point 2)

**The arithmetic is right:** (1 − 0.00025)^4096 = 0.359.

**What it does to the first family, computed without any training** (`python3 -I $C C`; 1,500 pairs of each operation, no letters, whole answers). Reduction score as a share of its level at the start:

| Updates | Factor on every weight | Sweep births (73129, 93683, 116089) | Backpropagation births |
|---|---|---|---|
| 64 | 0.984 | 99.6, 99.9, 99.7 | 99.2, 99.7, 99.3 |
| 256 | 0.938 | 95.2, 97.8, 98.1 | 97.3, 96.7, 96.8 |
| 1,024 | 0.774 | 17.8, 35.3, 26.1 | 39.3, 49.0, 46.0 |
| 2,048 | 0.599 | 0.4, 1.3, 0.3 | 0.4, 3.1, 1.3 |
| 4,096 | 0.359 | 0.0, 0.0, 0.0 | 0.0, 0.0, 0.0 |

Max and min are at 67 to 91% at 1,024, at 3 to 35% at 2,048 and at 1.3% or less at 4,096.

**Consequences.**

1. Under the standing decay a family that receives no gradient is gone after about 2,000 updates, whatever else is or is not learned. This is a fact about the present settings and bears directly on Mr. Xu's question.
2. `unrelated-decay` at row 4,096 can be above zero only if the new task's gradients hold the family's weights up. The brief's sentence "if it differs from `unrelated`, decay alone erodes what nothing contradicts" is settled beforehand: decay alone erodes all of it.
3. So `unrelated-decay` has to be read against decay alone, not against `unrelated`, and at rows where decay alone has not yet reached zero (256, 1,024, 2,048). Above decay alone: the new learning holds part of the family up. On it: nothing does. Below it: the new learning removes more than decay does.

**Which arm carries the reading.** `unrelated` without decay, as the brief has it. It isolates the question E43 asks, whether learning something unrelated removes the family, and the arm under decay cannot answer it. The arm under decay then answers a second question, for the learner as it has actually run: is anything held against a decay that would otherwise remove everything.

**One sentence of the brief to correct.** "Decay is the one thing that could act on weights the new task does not use." The only weights the new task leaves exactly alone are the byte table's rows of other bytes, and those have no decay. Every other weight takes part in every item and receives a gradient (E38's receipts record no parameter tensor without gradient at the first update of the second family; here the same follows from the network's structure). AdamW's step is about the learning rate whatever the gradient's size, once that size is well above 1e-8 (`python3 -I $C D`). So in both `unrelated` arms nearly every weight is pushed by the new task.

## 4. The condition that the new task was learned (point 3)

- **The level, 80%, is reasonable.** On 300 items the sampling error there is about 2 points.
- **The panel should be the taught one.** The condition exists to show that new learning took place. A network that had memorised the 2,048 taught pairs would have learned something unrelated just as well. Requiring 80% on held-out pairs could void the job for a reason outside its question. The held-out panel is reported.
- **Whether it will be met is not known.** The only evidence on file is the span job: a network started from an E26 birth, under backpropagation with every byte a target, learned to copy one of 24 letters across filler, needing 21,333 to 32,000 repeats to reach 50% at distance 4 (review 9.120 B). E43 gives 262,144 items, of which each taught pair is shown 128 times, but it asks for two symbols of 64 on bytes never seen, and half the runs are under the sweep, for which nothing is on file. The brief's own figure is 70%.
- **No further dose is named.** If the condition fails the report "draws nothing", and a second brief would be needed. The runs are cheap, so the brief should say now what happens (change 2).

## 5. Reproducibility (point 4)

`e43_unrelated_reader.py`, written from the brief's words, gives the six digests of the committed output on Python 3.13.17 (the output is the Mac's, which ran Python 3.11.5 in the earlier briefs): stream 34eefb44…, 153cafdc…, 821bcb2a…; held-out panel a6be6e03…, c5415a50…, 7cd7ba49…. (`python3 -I $C E`)

The words alone leave three choices, all settled by the file; since the worker copies the file and does not rewrite it, none is a required change.

| Choice | The file | The other reading gives (seed 73129) |
|---|---|---|
| "A fresh shuffled order each pass" | each pass shuffles the positions 0 to 2,047 afresh | the same first pass, then 3a493292… over 64 updates |
| The order in which the 4,096 pairs are listed before the shuffle | first symbol slowest | 77870106… |
| The order of the pairs a panel's 300 are drawn from | as the split left them | panel 2504c856… |

The digest rule itself is in the file only, which is acceptable for a specification file.

**Two things have no digest on file:** the taught panel, which the condition will now use, and the stream beyond 64 updates. The reader's values:

| Seed | Taught panel | Stream, 4,096 updates |
|---|---|---|
| 73129 | 3583cd943452ff06 | d3410fc2a8850bc1 |
| 93683 | 348392b8f4017512 | edbbcc1f0dc94926 |
| 116089 | 301fd99ddfa152f8 | fe35d695eace801d |

## 6. The checks

- **Check 1 can be run as written.** E38's receipts hold `tensors_sha256` at 64, 256, 1,024 and 4,096 for `zero` and `none` under both rules, and E38's arms are E43's `conflict-decay` and `conflict`: the same birth files, a fresh `make_opt`, the first block of E26's life, group 0's decay set to zero for `none` (`python3 -I $C F`; E38 brief, lines 14 to 34). The sixteen values begin:

  | | 64 | 256 | 1,024 | 4,096 |
  |---|---|---|---|---|
  | `zero`, sweep | f3819049 | 95f5de74 | e2d170be | ef1ac53c |
  | `none`, sweep | 5afe5e0a | 166dd91e | 08e44428 | 23f6ddaa |
  | `zero`, backpropagation | 35f0d6c9 | a5cad080 | 676c2fa0 | ce901c57 |
  | `none`, backpropagation | 5abf61c9 | 7789d013 | 8b9c6649 | a423a30b |

- **Check 2 covers less than it says.** E38's row 0 holds add and sub on all pairs split by the reduction, and the four held-out panels at 12 letters. It holds no max or min on all pairs, so those two figures of score 1 have no reference.
- **Check 3 cannot pass as written.** On the worker's host the line on the byte table's rows should read 0.0e+00, not 1.4e-06 (section 2). That it does is a test of section 2's inference.
- **Check 4's second half** counts the end symbol (256) if answers are included. It should say: every prompt byte, and every answer byte but the end symbol.

## 7. The measure of how far the weights moved (point 5)

- **Under decay it measures the decay.** After 4,096 updates at 0.25 every group's norm is at most 0.36 of its birth value if nothing holds it up, so "the norm of the difference divided by the norm at birth" is at least 0.64 for a pure shrink. In the two arms under decay the measure cannot tell learning from shrinking. A measure of direction is needed beside it (change 4).
- **It cannot be a guard by itself.** In section 23 of the memo, a change of 4 to 12% of each matrix's size taught a new family and removed an old one, and what mattered was a change of low rank in the recurrent layer. A small norm is therefore not a trivial change. The guard against a trivial change is the condition that the new task was learned; the measure describes, and should be worded so.
- **A functional measure would say more** (recommendation 2), and the reviewer can run section 23's analysis on committed weights (recommendation 3).

## 8. Other points

- **The reading is computable in every case.** The reduction score at row 0 is between 92 and 100 in the six births, so the share is defined; *kept* and *lost* cannot both hold.
- **Max and min** are scored on all pairs and at 12 letters, and reported beside the reading.
- **Rows** 0, 64, 256, 1,024 and 4,096 are the memo's.
- **Cost.** 24 runs are 98,304 updates: 3.0 to 3.5 CPU-hours at the rates on file, and about half an hour of scoring. The memo's "24 runs, about 2 CPU-hours" was my slip: its two arms are 12 runs. (`python3 -I $C G`)
- **Prediction P43.5** ("`unrelated-decay`'s share is at least 15 points below `unrelated`'s") is decided by section 3 whenever `unrelated` keeps 15% or more. It should ask what is not known (change 1).
- **Nothing differs silently from section 24.5.** The two added arms and the condition are declared.

## 9. The five required changes, as text to apply

**1. Decay (the arms' paragraph, the reading, the rows, P43.5).**

- Replace "in the unrelated case decay is the one thing that could act on weights the new task does not use … but there the conflict removed the family first" by: "Decay alone, with no learning, removes the first family: its reduction score is at 95 to 98% of its starting level after 256 updates, 18 to 49% after 1,024, 0 to 3% after 2,048 and zero after 4,096, in all six births (the reviewer's computation, audit section 3). The arm under decay asks whether the new learning holds anything up against that."
- Add to the scores: "**Decay alone**, no run: the birth file with every parameter but the byte table multiplied by (1 − 0.00025)^t, scored by score 1 at t = 64, 256, 1,024, 2,048 and 4,096. Its reduction scores should be within 3 points of the audit's table."
- Add row 2,048 to the scored rows of all arms.
- Replace the bullet on `unrelated-decay` in the reading by: "`unrelated-decay` is read against decay alone at rows 256, 1,024 and 2,048, per rule, in at least two of three seeds: *held up* if its reduction score is 15 points or more above decay alone's at row 1,024; *not held up* if within 15 points at all three rows; otherwise the differences are given. It is reported as 'under the standing decay'."
- Replace P43.5 by: "`unrelated-decay`'s reduction score is 15 points or more above decay alone's at row 1,024, two of three seeds, under the sweep", with your chance.

**2. The condition, and what happens if it fails.** "The reading is taken for a rule only if the unrelated task was learned: in at least two of three seeds the `unrelated` arm's **taught** panel is at 80% or more. The held-out panel is reported beside it. If the condition does not hold at row 4,096, the two `unrelated` arms of that rule continue to 16,384 updates, scored at 8,192 and 16,384, and the reading is taken at the first of those rows at which it holds, with the row stated. If it holds at neither, the report says that the new task was not learned and draws nothing about retention."

**3. Check 3.** "Run on `$K`, the self-checks print the Mac output's three seed lines, its line on shared bytes, its batch-tensor line and its score line exactly. The line on the byte table's rows should read 0.0e+00 on the worker's host: the 1.4e-06 of the Mac output is a difference between hosts in the initial values, not a movement in training (audit section 2); report what it reads. The first-batch loss should agree to two decimals. The runner also prints, for the three seeds, `sha256(repr(panel(seed, 'taught')))` and `digest(seed, 4096)`; their first sixteen digits equal the audit's table in section 5."

**4. Score 4.** Add: "and, for each group, the cosine between its weights and the birth's. Under decay the norm of the difference is at least 0.64 of the birth's norm from shrinking alone, so there the cosine is the figure to read." And replace "Score 4 is there because 'kept' would mean little if …" by: "Score 4 describes how much each arm changed the weights. It is not a guard: a change of a few percent of a matrix's size has been enough to teach a family and remove another (memo, section 23). The guard is the condition."

**5. Three small corrections.**

- Cost: "about 3.5 to 4 CPU-hours with scoring" in place of "about 2".
- Check 2: "its first-family scores equal E38's row 0 for seed 73129 where E38 has them: add and sub on all pairs, and the four held-out panels."
- Check 4: "every prompt byte, and every answer byte but the end symbol, is below 128."

## 10. Recommendations (none holds the release)

1. **Name the next step now.** "If the reading is *kept* or *partly kept*, a second unrelated task without carrying across a gap follows, to separate 'not disturbed' from 'held up by use': for example one symbol and the query byte, answered by two symbols fixed by a table drawn once."
2. **A functional measure beside score 4:** the mean absolute change of the first family's state at the `=` position between row 0 and each row, on the panel of score 2, divided by the mean absolute state at row 0.
3. **Commit four more files** for the reviewer's analysis: rows 64 and 1,024 of seed 73129's `unrelated` arm under each rule.
4. **Add to "what it cannot show":** "The optimizer. Every run starts a fresh AdamW, whose first steps move every weight by about the learning rate whatever the size of its gradient. A loss here may owe as much to that as to the new task."
5. **In the task's paragraph**, after "which is the user's case", add: "A reading of *kept* would have two explanations (audit section 1)."

## 11. Can E43 be released?

Yes, after the five changes of section 9, without a re-audit.

## 12. What I did not check

- **Anything that needs torch.** The file's self-checks, the score of 0 of 300 before any update, the first-batch loss and the equality of the `conflict` arms with E38 are the coordinator's output or the worker's checks.
- **Torch's initial values.** Section 2's cause is inferred from the saved files and from the record, not reproduced.
- **E38's runner.** That its arms are E43's `conflict` arms is read from its brief and receipts; check 1 tests it.
- **Whether the task will be learned in the dose.** Section 4 gives the only evidence on file.
- **Decay alone on all pairs.** The table of section 3 uses 1,500 pairs of each operation; the worker's score 1 uses all 7,921, hence the tolerance of 3 points in change 1.
