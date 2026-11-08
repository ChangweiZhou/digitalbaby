# Where the MiniFly program stands: consolidated review, 7 November 2026

This replaces the running log of sections 9.1–9.10 as the thing to read first. Those sections stay as the record; this one says what they add up to.

## 1. The goal, stated so it can be tested

Mr. Zhou's goal is a learner that keeps learning through its life and can reason, not one that memorizes and retrieves. "Reason" has never been defined in the project, which is one source of the confusion, so here is the definition the rest of this document uses. One learner, one continuous life, local plasticity during life, raw bytes in and its own bytes out, that shows five properties:

| | Property | What counts as evidence |
|---|---|---|
| P1 | **Rule acquisition.** Learns a rule from examples and applies it to inputs that were never reinforced | Exact answers on a sealed panel of never-taught inputs, above every shortcut (lookup, nearest neighbour, frequency, structural half-ceilings), with a shuffled-label control |
| P2 | **Working state.** Holds intermediate information to compute multi-step answers | Correct answers when the needed information is no longer in the input window, and on longer gaps than any seen in training |
| P3 | **Retention.** Keeps earlier rules while acquiring later ones | Accuracy on the first tasks after the last tasks have been learned, in one life, without replaying the first tasks' data |
| P4 | **Composition.** Answers untaught combinations of learned rules | Exact answers on two-step problems built from taught one-step rules, where the specific combination was never taught |
| P5 | **Locality.** All of the above during life with local rules | Every life update uses only a synapse's own activity, the sign of its own weight, neighbouring signals and a broadcast error; development before life may use any optimizer. The learner may edit its own local rules during life (learned rates, gated or metaplastic writes) provided the edit itself is local. Clarified 7 November: "local" does not mean a fixed small neighbourhood; which neurons a synapse takes its signals from can itself be adjusted dynamically during life, so learned feedback pathways and dynamically weighted traces are inside P5 |

This is what "persistent learning that can reason" means at a scale we can audit completely. Nothing fly-sized reaches reasoning in the general sense, and nothing here claims to.

## 2. What is established, against that definition

All results below are development results: three seeds, one harness (networks of 60,000 to 440,000 parameters), predictions stated before each launch, a backprop ceiling and shuffled control in every table, and no independent confirmation. Figures are exact answers with the learner's own end-of-sequence byte on sealed never-taught panels.

**P1, rule acquisition: established with local rules.** On modular addition from raw bytes, backprop reaches 95–99% of unseen sums, the sign-symmetric local rule 95–98%, and the project's own random-feedback rule 70–84% at its peak before drifting. Shortcut ceilings are at or below 1%. The fly's sensory path, measured as a residual beside a working shared path, contributes about 1%. Grokking never appeared because on this task the rule is found as fast as the table; it is a signature of a different regime, not a milestone.

**Development before life: established as a sample-efficiency and stability device, not an asymptote device.** A network developed by backprop on one task, then handed to the local rule on a new task, reaches 80–94% at a quarter of the pairs and beats backprop trained from scratch at a tenth of the pairs (where backprop memorizes). Shuffled births are worse than no birth. Breadth of development does not help on related tasks. At four times the width, the local rule from a random birth matches developed births at a quarter of the pairs. A scalar reward signal, the dopamine-alone picture, learned nothing at this budget; the life rule needs a vector error at the output.

**P2, working state: established in development; in life only with credit through time.** A gated recurrent state developed by backprop holds an operand across up to 12 distractor bytes at 84–94% and extrapolates to 20 at 82–88%. An ungated state degrades with distance; a stateless window is at the floor beyond its window. In life, the sign rule trains this state at 68–86% when credit flows through time and collapses to 4–13% when it does not. A per-synapse eligibility trace that follows each unit's own gate does not recover it (2–14%), because on this cell the credit runs through other units. The literature (e-prop, 2020) shows time-local traces do work on cells with a diagonal memory such as the LSTM cell state. That is a result to take from the shelf with one confirming run, not a program.

**P3, retention: not tested.** Nothing in the record learns a second task after a first and then scores the first.

**P4, composition: not tested.** Nothing in the record is scored on an untaught combination of taught rules.

**P5, locality: established for P1, conditional for P2, untested for P3 and P4.**

**The connectome: the compass and steering circuit computes once its gains are fitted.** Codex extracted the FlyWire v630 central complex with signed edges. With hand-set gains the isolated circuits hold and turn a heading and steer, with large margins over a degree-matched rewired control, and the closed loop fails. With 15 gains fitted by gradient descent, the loop closes: 80 of 96 goals on fresh panels with the compass at 10° after four seconds of self-generated turning, against 20–24 for the rewired control fitted identically. This is five cell types, not the central complex; path integration and any learning in that circuit are untested.

**What the old project got wrong, in one paragraph.** The native learner was a random-expansion memory with a local readout and no shared computation, so nothing learned from one input could make another easier. Tasks were smaller than the memory. Gates were set where failure was uninformative, and learner, task and gate changed together, so no negative result localized. Constraints accumulated without a feasibility check, and the one component that most limited generalization, the sensor, was declared immutable. The mechanism catalogues listed ways to manage a store, not ways to compute. None of this was mysterious once a shared computation was put beside the store and measured.

## 3. What is taken from the literature and will not be re-tested

The 2012–2016 recurrent-network results settle what gated cells can compute under backprop through time; the GRU results above are consistent with them and add nothing. E-prop settles that eligibility traces train diagonal-memory cells. Complementary learning systems theory settles why a fast store beside a slow network prevents forgetting in principle. Feedback alignment and sign symmetry settle that backward weights need not be transported. The grokking papers settle when a late rise appears. Re-deriving any of these in this harness is calibration at best. The only reason to run one of them again is a single confirming run so the harness has its own baseline for a comparison that follows.

What the literature does not settle, and what this project can contribute, is the combination: a learner that acquires rules with local plasticity during life, keeps them while acquiring more, and composes them, in one life, on a world small enough to audit every input. Systematic composition was the known failure of the 2012–2016 nets, and sequential retention was the other. Those two are the open questions, and the project's angle on them is specific: a fast associative store of the mushroom-body kind beside a shared path, local rules in life, an optimizer in development.

## 4. Where the confusion came from

- The review became a running log. Each result was appended with its own "next step", and the next step shifted ten times in two days, each shift locally justified and the sequence unreadable. This document is the correction.
- I introduced ideas mid-stream without flagging them as plan changes. The LSTM cell appeared as the interpretation of a negative result, not as a decision put to you first.
- "Reasoning" was never defined, so rule acquisition on unseen sums was treated as progress toward it without saying how far that is from it. Section 1 above is the definition.
- Two lines ran in different vocabularies: my harness in bytes and local rules, Codex's in cell types and gains. They are the same programme, symbolic and embodied, and this document treats them as such.
- One gate I gave Codex was infeasible by construction (a combined-loop target above the steering circuit's own ceiling). It was corrected within the day and the corrected result stands.

## 5. The plan: three experiments to the target demonstration

The remaining properties are P3 and P4, and then all five in one life. Three experiments, each with predictions stated before launch, each about an afternoon on the existing machine.

**E4, retention.** One network with a gated state, developed by backprop on a family of modular operations. In life, four new operations learned in sequence by the local rule, each on its own code byte, 25% of pairs each, no replay of earlier data. Score every operation after every stage. Arms: shared path alone; shared path with a fast associative store of the mushroom-body kind (sparse expansion, local writes at the output, the project's own component); with consolidation between stages (weight decay and self-generated replay from the store); with both; backprop ceiling; shuffled control. Prediction: the shared path alone drops the first operation below 30% by the fourth; store plus consolidation keeps it above 70%. This is the first test of "persistent" in the record, and it is where the project's nine weeks of memory work pay in.

**E5, composition.** Develop and teach one-step operations. Test two-step problems rendered as bytes, for example a sum followed by a subtraction under one modulus, where the specific combination was never taught. Two renderings: direct, where the learner must answer in one pass; and scratchpad, where it emits the intermediate result as bytes and continues. Prediction: direct composition fails (below 30%), consistent with the 2018 systematic-generalization results; scratchpad composition reaches 70% or better if emitting intermediates was taught on other combinations. The difference between the two is the result. Scratchpad composition is reasoning in the only form a small system can show: using a learned step as the input to the next.

**E6, the persistent composer.** One life: operations learned in sequence by the local rule with the store and consolidation from E4, then untaught compositions scored as in E5, then the early operations scored again. Pass means P1 through P5 in one learner. That is the demonstration the project set out to build, at toy scale, and it is reachable this month if E4 and E5 pass. If either fails, its failure localizes to one property with the others held, which is the position the old project never reached.

**The embodied twin, in parallel.** Codex's path-integration brief is written and is the next step on the connectome line. If displacement works, the arena becomes the second life domain, with the same E4 and E5 questions asked about places and goals instead of sums, and the local rule applied at the sites the fly actually uses.

**One shelf check, not a program.** LSTM-type cell with the eligibility-trace rule on the rung-3 task, one run per seed, to confirm the literature result in this harness before E6 relies on it.

## 6. Stop rules and what counts as done

Every experiment keeps the discipline that produced the last two days: predictions before launch, a backprop ceiling and a shuffled control in every table, sealed panels scored at every checkpoint, receipts on the consolidated branch, one harness, no new branches, no catalogues. A negative result ends its line and is written up with its reason; it does not spawn a version. E6 passing on three seeds, with the shortcut audit and the shuffled control, is "done" for this phase. What follows done is scaling and the question of interfaces between the arena and the symbolic world, which are not decisions for this month.

## 7. Next phase, agreed 7 November: composition, de-confounded first

The four composition failures (direct, scratchpad, canonical scratchpad, routed canonical) share a design confound: the two taught operator pairs, (+,+) and (−,−), never let the second operator vary while the first stayed fixed, so the network never saw the factor it was asked to generalize over. The conclusion "composition is not achieved" was too strong; what is established is that a two-factor function is learned as one pattern when its factors are confounded, under either rule. Mr. Zhou has relaxed the no-attention rule as one modest change. The agreed order, each step deciding one thing:

1. **E7, de-confounded composition.** Four single-step operations (addition, subtraction, max, min mod 97) in development; life teaches 12 of the 16 ordered operator pairs in scratchpad form so that every operator appears in both positions, and holds out the four cross pairs. Decides whether P4 was a confound. One afternoon.
2. **If needed: content attention with a copy path.** The rewrite step becomes copying of slots, which generalizes by construction; the operator-to-code legend can be given in context. Decides whether binding needs a mechanism.
3. **The ring with innate operations.** Numbers as bump positions, addition and subtraction as innate shifts, learning confined to the byte-to-displacement and operator-to-direction interfaces, as the fly's central complex composes vectors. Decides whether composition should be innate here. Unites the symbolic and connectome lines.
4. **Meta-learning for compositionality** as the development phase (fresh symbols with meanings given in context, so the network learns to bind at run time), the largest change, saved for last.
5. **The persistent-life demonstration**, independent of the above: one life, eight operations learned in sequence with replay from the store, every earlier one scored after every later one.

Codex's line continues: the path-integration return-leg diagnosis, then arena learning with the eligibility trace.

## Addendum, 7 November 2026 evening (E9, E12, E10, E11)

- Replay is structural, not a scheduling fix: interleaving holds eight operations only while the stream is stationary, and the first distribution change erases every standalone response without replay (E9); the erasure is functional, with no savings on relearning (E12-A). Raviv et al. (*Science*, 5 Nov 2026) reach the same conclusion for a model trained day by day on a child's speech. Replay share buys retention roughly linearly; recency weighting hurts; the buffer saturates near 1,024 prompts per stage (E9, E12).
- Batch 256 at fixed exposure lifts the sign rule to 91–92% retention and 69–73% held composition (E12-B); undiagnosed (decay versus averaging), E13 Part A.
- Composition by wiring: the ring with innate operations composes untaught pairs at 100% zero-shot where its interfaces developed (E10).
- Working state by a local rule: the diagonal cell with an exact time-local trace acquires holding from a no-memory birth (63–90% at k = 12) where the within-step rule reaches 5–32% (E11). In the same learner, retention across sequential operations under the sign rule collapses (E11, P11.5 refuted); P2 against P3 is the open conflict to resolve before scaling.
- Decay 1.0 per update was the dominant cause of the sign rule's forgetting and composition gap: decay 0.25 at batch 64 gives E8 94% retention and 81–85% held composition (E13), and recovers the cell learner's retention from 0–15% to 52–72% on carry operations with fresh-render replay (E11-D). Path integration from the extracted wiring reached its prescribed stop (E13-D). Settings going forward: decay 0.25, fresh-render replay; E14 confirms on fresh seeds and measures the curve.

## Addendum, 8 November 2026 (E13, E14, E15): settings adopted, Phase A has one gate left

- **Adopted toy-scale settings for local arms:** decoupled decay 0.25 per update (E13-A1, E14-A/B: plateau 0.1–0.25, cost at 0.5, collapse at 1.0), fresh-render replay of stored pairs at half of every batch (E11-D), the exact diagonal trace in scan form (E11, E15), batch 64. Decay 1.0, chosen in E8 and carried since, was the largest single handicap on the sign rule in E8, E10, E11 and the E7 composition gap.
- **Standing of the five properties:** P1, P2, P3 and P5 established under a local rule at toy scale with these settings; P4 by wiring on the ring and within 2–3 points of a matched backprop comparator by learning (E14-A on E7). Open: one fresh seed fails one composition pair (E14), retention of holding acquired in life misses its gate by 2–5 points in two seeds (E15-B), one ring seed loses 15 points of carry interface (E15-D).
- **Refuted this round:** the alignment account of the sign rule's decay sensitivity (E14-C); split write and read gates on the cell (E15-C); interleaving as a cure for forgetting at decay 1.0 (E11-D); recency-weighted replay (E12).
- **Plan.** Codex: E14-S (backprop at decay 1.0; the composition gate on nine seeds; the curve of the failing pair). Here: **E16**, the Phase A exit gate, one cell learner at the adopted settings on one stream carrying every toy family, scored per family against a backprop twin, retention measured after the stream moves on; gate 80% of the twin on every family. Then **E17**, the window replaced by a bank of diagonal cells at fixed retention rates fed one byte at a time, matched against E16. Phase B (the synthetic language stream) starts only after E17, with its battery (held-out prediction, segmentation, referent binding, two-word composition, one-shot novel words, retention) written before the stream exists.
