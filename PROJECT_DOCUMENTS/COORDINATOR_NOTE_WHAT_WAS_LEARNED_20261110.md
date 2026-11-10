# What the coordinating session learned, 9–10 November 2026

A note from the coordinating session (Claude), written at Mr. Xu's request, for Mr. Zhou, Mr. Xu and whoever coordinates next. It covers the E16 follow-ups, E17 Phases 2 to 4, E18 to E20 and two audits. It states what holds, what I got wrong, and the working rules I would follow from the first day if I started again. Sources are the review sections named in brackets in `MINIFLY_PAST_WORK_REVIEW_FOR_MR_ZHOU_20261106.md`. Everything here is a development result on a toy arithmetic harness at evidence level E2, held out by ordered pair; none of it is a learning-core claim.

## 1. The one thing to know first

**No learner on file composes from memory.** A composition prompt is ten bytes and the window learner's window is ten bytes, so every composition that learner has done was done with all operands in view. The windowless learner has to remember them, and under backprop it reaches 49–59% on the compositions it is taught and 1–7% on untaught plus-minus pairs [9.76]. I found this on the last afternoon by reading taught and untaught scores apart. It should have been the first check after the first composition result, and most of the work on learning rules and replay settings in these two days refined learners that lack the computation. E20 tests it directly.

## 2. What holds

| Finding | Where | Limits |
|---|---|---|
| The window learner under the sign rule, nothing frozen, passes the pre-registered family gate in 13 of 15 corrected-panel drift seeds | 9.62, 9.75 | 2 of 15 on the plus-minus pairs alone; 9 of 15 on whole worked answers; composition is in view |
| Under the sign rule the learner retains nothing without replay | 9.64 | one life per seed and share |
| More replay ends with better hard pairs, for the rule more than for backprop | 9.64 | an allocation result: more replay also means fewer fresh items; not a retention mechanism |
| The windowless learner lost its first bank because credit above the write synapses was cut off in time, not because of weight signs | 9.67 | measured as gradient alignment at birth |
| With weight-sign feedback and credit through time at every layer, the windowless learner trains with nothing frozen: 0.92–0.95 of its reference on a new modulus | 9.70 | a backward pass over the item; no published precedent; original skills fall to 11–43; fails at 40 distractors; composes nothing |
| Backprop on the learner's own pipeline reaches the reference, so the remaining 3–7 points belong to weight signs | 9.70 | three seeds |
| For two leaky layers with fixed rates and a linear upper input, the same credit can be carried forward only and exactly; the worker's implementation matches backprop to rounding on a forward pass I wrote independently | 9.72, `exp45/E19_COORDINATOR_CHECK.md` | gradient equality only; learning under it is not shown; the linear write costs about 22 points of birth accuracy under backprop |
| Held panels are held out by ordered pair only (25% reversal overlap, 71–73% sub-step exposure), and this did not inflate scores measurably | 9.75 | strict panels move ratios by at most 0.025 |

## 3. What I got wrong, and the rule each error taught

1. **I read a software artifact as a mechanism, twice.** In E17 Phase 1 the optimizer's decay erased embedding rows that received no gradient, and I explained the collapse as replay onset and then as truncated credit. An outside audit found it. *Rule: before interpreting any collapse, record which parameters and which rows receive gradient on the first update, by symbol role.* This census is now in every receipt.
2. **I turned an implementation's failure into a law.** After Phase 3 I wrote that the first bank "must be born by backprop and frozen". One read-only measurement the next morning, the cosine of each layer's update with the true gradient, showed the layer was simply getting no usable credit. *Rule: measure per-layer alignment with a reference gradient before running lives, and before concluding anything about a rule.*
3. **My directional predictions failed about as often as they held.** Exact feedback against the trace (reversed), the replay dose (reversed), the one-layer trace (missed), the early-warning signature (missed). Writing predictions first is what made these visible. But twice I then re-told the result as a new mechanism ("retention, not acquisition") that the data did not isolate. *Rule: when a direction fails, report the measured curve and stop; a new explanation is a new prediction for a new run.*
4. **I compared arms that differed in more than one thing.** Phase 4's two rule arms differed in feedback scaling as well as time reach; E18's four arms differ in head, parameters, feedback and decay; my E19 brief used different decay for the rule and its ceiling. *Rule: before launch, list every difference between two arms in a table. If the list has more than one row, either add the control or call the arm a package.*
5. **I joined receipts by file name.** The eighteen-seed tally mixed two panel versions; E16-M2 compared arms scored on different panels. *Rule: every receipt carries the hash of its panels and its effective configuration, and comparisons are joined on those fields.* The summarizer now does this.
6. **My "independent" check called the same code on both sides.** *Rule: a check is independent only if the reference was written separately, preferably by hand without autograd, and a check must include a case that is known to fail.* Both are now in place for Phase 4 and E18.
7. **I overclaimed in the wording.** "Keeps its birth" described a learner whose original skills had fallen by half. "Composes" described composition in view. "End of the composition blocks" described a point where composition teaching continues. *Rule: state what was measured, on which panel, in absolute numbers and whole answers, and give the worst subgroup beside the mean.*
8. **Three of four literature claims I made from memory were wrong or out of date.** One relevant paper had appeared the day before. *Rule: a literature claim steers an experiment only after someone has opened the source; commission that check before the design, not after.*
9. **I optimised credit and replay before asking whether the task was computable by the architecture at all.** The windowless reference under backprop was at 1–7% on the hard pairs from Phase 1 onward, and I kept filing "nothing composes" as a footnote. *Rule: a positive reference comes first. If backprop on the same inputs cannot do it, no rule experiment on that architecture is informative about that skill.*
10. **I leaned on the auditor's framing for direction.** Mr. Xu's correction: the auditor audits, the coordinator decides. *Rule: verify every audit claim by running it, adopt what holds, and then decide from the evidence what to do next and say so plainly.*

## 4. Working rules that paid off

- **Predictions before launch, with the reading of each outcome written beforehand.** Every wrong belief above was caught because it had been written down.
- **A dry run of the whole loop for every new command shape, and after any harness change a rerun of an archived dry receipt that must reproduce score for score.** This caught a wrong admission setting and a missing birth file before real runs, and it is how I know the audit fixes changed no behaviour.
- **Cheap read-only diagnostics before expensive lives.** The alignment table took a minute and redirected a day. The taught-against-untaught table took seconds and redirected the programme.
- **Recompute every worker's headline numbers from raw receipts before filing.** All four cloud reports reproduced; I would not have known otherwise. For new code, write an independent reference before the worker's runs report, not after.
- **Briefs that a worker can run without asking:** exact commands, files, seeds, predictions to copy, the reading of each outcome, acceptance tests for new code, cost from measured pace, and the rule that a miss is a result. Workers given this ran clean, on forecast, with no reruns.
- **One producer on the shared machine, with a lock that is created exclusively and released only by its owner, and each job's exit code recorded.** The earlier lock could be overwritten by any process that used the same word.
- **Preserve originals.** Repeats carry their own tag; errata are appended, never edited in.
- **Forecast from measured pace and report forecast against measured.** Wall-time forecasts were within range each time; CPU forecasts for new arms were off by up to 6%.

## 5. On auditors and workers

- Both outside audits were right on every point I could test, including points against my own filed text. An audit that runs nothing still finds real defects by following each loss through time, gradient paths and optimizer groups; the right response is to run its claims, not to argue with them.
- Two findings were real but small when measured (the feedback-map confound, the held-out overlap). Measuring settled both in an hour. Fix the defect anyway.
- A worker's report is a claim until recomputed. A worker's new code is a claim until checked against a reference the worker did not write.
- Do not change a run in flight. Addenda go to the report; controls are new runs.
- Keeping instances busy is not a goal. The useful question for each idle instance is which open question it would close.

## 6. What is open

- **Composition from memory** (E20-A, E20-B): whether the window learner composes when an operand must be remembered, and whether any windowless recurrent network of this size does so under backprop.
- **In flight:** E18 births, E19 Stage B and its decay-matched control, E16-M4, twelve repeats on corrected panels, the multi-timescale literature check.
- **Retention without rehearsal.** Nothing on file retains without replay. Slow and fast weights are the candidate, untested.
- **State across records.** The memory banks reset between items; several decay rates establish nothing across records.
- **Decisions that are the owner's:** whether weight-sign feedback relayed through every layer and back along an item meets the locality constraint; whether births may use backprop; whether a bounded window is an acceptable part of the design; whether to adopt the promotion test in the status note.

## 7. If I started again

1. Write down, for each task, what must be remembered and what is in view, and build one panel that separates them.
2. Get a positive reference under backprop for every skill before touching a rule.
3. Put the census, the hashes and the strict panels in the first receipt.
4. Check alignment per layer before any life.
5. One question per experiment, one difference per comparison.
6. Verify the literature before designing.
7. Then, and only then, compare learning rules.

## 8. Addendum, the same evening: I repeated my own list within hours

Three hours after writing section 3, I shipped the E20-A brief, and Codex's next audit found in it the errors of items 1, 4, 7 and 9: no information audit of the new panel (the first operand was still in view in a ninth of the items, pooled with the rest), a suffix metric, one arm with a lookup store, a prediction that tied existence to the size of a penalty, and a qualifier that failed on the new prompts. The same audit showed that the "strict" held-out panel I had built that morning left most equivalent supervision in, so my statement that the overlap did not inflate scores had no support when I made it. Both are corrected in review 9.79.

What this adds to the rules:

11. **A list of lessons does not apply itself.** Each rule in section 3 needs a check that fails when the rule is broken. Before a brief ships, the assay it relies on must be run against a predictor that must fail (here: one that sees only the last ten bytes) and a case that must pass, and the brief must show both.
12. **Qualify the assay on a reference before spending a roster on it.** Stage the first run so that a cheap arm shows the control works; only then launch the comparison.
13. **A "strict" or "clean" subset is a claim about a classifier.** Test the classifier against every route by which the answer could have been supplied, in every orientation, before reporting what the subset shows; re-score the same items, never a resample.
14. **Speed is where my errors come from.** Every error the audits found was in something I designed and shipped within the hour. The repairs took about an hour each as well. Designing an hour slower costs less than one audit round.
