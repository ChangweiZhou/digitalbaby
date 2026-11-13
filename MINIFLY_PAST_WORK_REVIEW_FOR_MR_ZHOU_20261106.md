# MiniFly past-work review for Mr. Zhou

**Date:** 6 November 2026. **Sources:** the local work folder (`second_try_copy`, 214 experiment folders, 33 GB) and the GitHub repository `ChangweiZhou/digitalbaby` (459 commits on the default branch, 7 branches, latest push 6 November). This is a read-only review. No experiment was rerun and no frozen result was changed.

## 1. Bottom line

- **The fly is a very good associative memory and nothing more, and that is by construction.** The mushroom-body design (random sparse expansion, top-5% winner-take-all, a fading byte trace) gives every input its own unique code. That is ideal for "cue → answer" lookup and destroys the similarity structure a learner needs to generalize. Every raw-byte test of rule reuse has confirmed this.
- **No learner in the project, native or reference, has ever learned addition from raw bytes.** Not the native cores (0% taught, 0% held), not small GRUs (100% taught, 0–6% held), not a 1,024-candidate evolutionary search, not backpropagation-through-time at 65,536 records (7 of 1,536 held sums, 0 of 384 carry pairs). The only "addition" successes used an engineered numeric representation supplied by the experimenter.
- **The genuine positive results are narrow and come from different learners.** Durable fact storage, recall under new renderings, a one-bit class rule reused on untaught pairs, memory protection, consequence-driven action choice, and a successor-map operator each worked in its own harness. They have never been shown together in one continuing learner, and the project's own 5 November memos say so.
- **The project is stuck for process reasons as much as scientific ones.** Roughly 220 numbered versions were produced in nine weeks by three AI agents on parallel branches. Most ended "NO_ADVANCE" by a few hundredths of a bit, each followed by a new mechanism rather than a diagnosis. The adopted core has not changed since 11 October. The two clearest diagnoses in the record (17 October and 3 November) were never acted on.
- **The way out is not another mechanism.** Reopen the fixed sensory front end, give the learner an ordered recurrent state and time to compute, let a gradient reference establish that each task is solvable before testing a local rule, and run one learner through one continuous life. Details in section 6.

## 1A. Where the program stands: consolidated review, 7 November 2026

This replaces the running log of sections 9.1–9.10 as the thing to read first. Those sections stay as the record; this one says what they add up to.

### 1. The goal, stated so it can be tested

Mr. Zhou's goal is a learner that keeps learning through its life and can reason, not one that memorizes and retrieves. "Reason" has never been defined in the project, which is one source of the confusion, so here is the definition the rest of this document uses. One learner, one continuous life, local plasticity during life, raw bytes in and its own bytes out, that shows five properties:

| | Property | What counts as evidence |
|---|---|---|
| P1 | **Rule acquisition.** Learns a rule from examples and applies it to inputs that were never reinforced | Exact answers on a sealed panel of never-taught inputs, above every shortcut (lookup, nearest neighbour, frequency, structural half-ceilings), with a shuffled-label control |
| P2 | **Working state.** Holds intermediate information to compute multi-step answers | Correct answers when the needed information is no longer in the input window, and on longer gaps than any seen in training |
| P3 | **Retention.** Keeps earlier rules while acquiring later ones | Accuracy on the first tasks after the last tasks have been learned, in one life, without replaying the first tasks' data |
| P4 | **Composition.** Answers untaught combinations of learned rules | Exact answers on two-step problems built from taught one-step rules, where the specific combination was never taught |
| P5 | **Locality.** All of the above during life with local rules | Every life update uses only a synapse's own activity, the sign of its own weight, neighbouring signals and a broadcast error; development before life may use any optimizer. The learner may edit its own local rules during life provided the edit is local, and, clarified 7 November, "local" does not mean a fixed small neighbourhood: which neurons a synapse takes its signals from can itself be adjusted dynamically, so learned feedback pathways and dynamically weighted traces are inside P5 |

This is what "persistent learning that can reason" means at a scale we can audit completely. Nothing fly-sized reaches reasoning in the general sense, and nothing here claims to.

### 2. What is established, against that definition

All results below are development results: three seeds, one harness (networks of 60,000 to 440,000 parameters), predictions stated before each launch, a backprop ceiling and shuffled control in every table, and no independent confirmation. Figures are exact answers with the learner's own end-of-sequence byte on sealed never-taught panels.

**P1, rule acquisition: established with local rules.** On modular addition from raw bytes, backprop reaches 95–99% of unseen sums, the sign-symmetric local rule 95–98%, and the project's own random-feedback rule 70–84% at its peak before drifting. Shortcut ceilings are at or below 1%. The fly's sensory path, measured as a residual beside a working shared path, contributes about 1%. Grokking never appeared because on this task the rule is found as fast as the table; it is a signature of a different regime, not a milestone.

**Development before life: established as a sample-efficiency and stability device, not an asymptote device.** A network developed by backprop on one task, then handed to the local rule on a new task, reaches 80–94% at a quarter of the pairs and beats backprop trained from scratch at a tenth of the pairs (where backprop memorizes). Shuffled births are worse than no birth. Breadth of development does not help on related tasks. At four times the width, the local rule from a random birth matches developed births at a quarter of the pairs. A scalar reward signal, the dopamine-alone picture, learned nothing at this budget; the life rule needs a vector error at the output.

**P2, working state: established in development; in life only with credit through time.** A gated recurrent state developed by backprop holds an operand across up to 12 distractor bytes at 84–94% and extrapolates to 20 at 82–88%. An ungated state degrades with distance; a stateless window is at the floor beyond its window. In life, the sign rule trains this state at 68–86% when credit flows through time and collapses to 4–13% when it does not. A per-synapse eligibility trace that follows each unit's own gate does not recover it (2–14%), because on this cell the credit runs through other units. The literature (e-prop, 2020) shows time-local traces do work on cells with a diagonal memory such as the LSTM cell state. That is a result to take from the shelf with one confirming run, not a program.

**P3, retention: not tested.** Nothing in the record learns a second task after a first and then scores the first.

**P4, composition: not tested.** Nothing in the record is scored on an untaught combination of taught rules.

**P5, locality: established for P1, conditional for P2, untested for P3 and P4.**

**The connectome: the compass and steering circuit computes once its gains are fitted.** Codex extracted the FlyWire v630 central complex with signed edges. With hand-set gains the isolated circuits hold and turn a heading and steer, with large margins over a degree-matched rewired control, and the closed loop fails. With 15 gains fitted by gradient descent, the loop closes: 80 of 96 goals on fresh panels with the compass at 10° after four seconds of self-generated turning, against 20–24 for the rewired control fitted identically. This is five cell types, not the central complex; path integration and any learning in that circuit are untested.

**What the old project got wrong, in one paragraph.** The native learner was a random-expansion memory with a local readout and no shared computation, so nothing learned from one input could make another easier. Tasks were smaller than the memory. Gates were set where failure was uninformative, and learner, task and gate changed together, so no negative result localized. Constraints accumulated without a feasibility check, and the one component that most limited generalization, the sensor, was declared immutable. The mechanism catalogues listed ways to manage a store, not ways to compute. None of this was mysterious once a shared computation was put beside the store and measured.

### 3. What is taken from the literature and will not be re-tested

The 2012–2016 recurrent-network results settle what gated cells can compute under backprop through time; the GRU results above are consistent with them and add nothing. E-prop settles that eligibility traces train diagonal-memory cells. Complementary learning systems theory settles why a fast store beside a slow network prevents forgetting in principle. Feedback alignment and sign symmetry settle that backward weights need not be transported. The grokking papers settle when a late rise appears. Re-deriving any of these in this harness is calibration at best. The only reason to run one of them again is a single confirming run so the harness has its own baseline for a comparison that follows.

What the literature does not settle, and what this project can contribute, is the combination: a learner that acquires rules with local plasticity during life, keeps them while acquiring more, and composes them, in one life, on a world small enough to audit every input. Systematic composition was the known failure of the 2012–2016 nets, and sequential retention was the other. Those two are the open questions, and the project's angle on them is specific: a fast associative store of the mushroom-body kind beside a shared path, local rules in life, an optimizer in development.

### 4. Where the confusion came from

- The review became a running log. Each result was appended with its own "next step", and the next step shifted ten times in two days, each shift locally justified and the sequence unreadable. This document is the correction.
- I introduced ideas mid-stream without flagging them as plan changes. The LSTM cell appeared as the interpretation of a negative result, not as a decision put to you first.
- "Reasoning" was never defined, so rule acquisition on unseen sums was treated as progress toward it without saying how far that is from it. Section 1 above is the definition.
- Two lines ran in different vocabularies: my harness in bytes and local rules, Codex's in cell types and gains. They are the same programme, symbolic and embodied, and this document treats them as such.
- One gate I gave Codex was infeasible by construction (a combined-loop target above the steering circuit's own ceiling). It was corrected within the day and the corrected result stands.

### 5. The plan: three experiments to the target demonstration

The remaining properties are P3 and P4, and then all five in one life. Three experiments, each with predictions stated before launch, each about an afternoon on the existing machine.

**E4, retention.** One network with a gated state, developed by backprop on a family of modular operations. In life, four new operations learned in sequence by the local rule, each on its own code byte, 25% of pairs each, no replay of earlier data. Score every operation after every stage. Arms: shared path alone; shared path with a fast associative store of the mushroom-body kind (sparse expansion, local writes at the output, the project's own component); with consolidation between stages (weight decay and self-generated replay from the store); with both; backprop ceiling; shuffled control. Prediction: the shared path alone drops the first operation below 30% by the fourth; store plus consolidation keeps it above 70%. This is the first test of "persistent" in the record, and it is where the project's nine weeks of memory work pay in.

**E5, composition.** Develop and teach one-step operations. Test two-step problems rendered as bytes, for example a sum followed by a subtraction under one modulus, where the specific combination was never taught. Two renderings: direct, where the learner must answer in one pass; and scratchpad, where it emits the intermediate result as bytes and continues. Prediction: direct composition fails (below 30%), consistent with the 2018 systematic-generalization results; scratchpad composition reaches 70% or better if emitting intermediates was taught on other combinations. The difference between the two is the result. Scratchpad composition is reasoning in the only form a small system can show: using a learned step as the input to the next.

**E6, the persistent composer.** One life: operations learned in sequence by the local rule with the store and consolidation from E4, then untaught compositions scored as in E5, then the early operations scored again. Pass means P1 through P5 in one learner. That is the demonstration the project set out to build, at toy scale, and it is reachable this month if E4 and E5 pass. If either fails, its failure localizes to one property with the others held, which is the position the old project never reached.

**The embodied twin, in parallel.** Codex's path-integration brief is written and is the next step on the connectome line. If displacement works, the arena becomes the second life domain, with the same E4 and E5 questions asked about places and goals instead of sums, and the local rule applied at the sites the fly actually uses.

**One shelf check, not a program.** LSTM-type cell with the eligibility-trace rule on the rung-3 task, one run per seed, to confirm the literature result in this harness before E6 relies on it.

### 6. Stop rules and what counts as done

Every experiment keeps the discipline that produced the last two days: predictions before launch, a backprop ceiling and a shuffled control in every table, sealed panels scored at every checkpoint, receipts on the consolidated branch, one harness, no new branches, no catalogues. A negative result ends its line and is written up with its reason; it does not spawn a version. E6 passing on three seeds, with the shortcut audit and the shuffled control, is "done" for this phase. What follows done is scaling and the question of interfaces between the arena and the symbolic world, which are not decisions for this month.


### 1A.7 Next phase, agreed 7 November: composition, de-confounded first

The four composition failures (direct, scratchpad, canonical scratchpad, routed canonical) share a design confound: the two taught operator pairs, (+,+) and (−,−), never let the second operator vary while the first stayed fixed, so the network never saw the factor it was asked to generalize over. The conclusion "composition is not achieved" was too strong; what is established is that a two-factor function is learned as one pattern when its factors are confounded, under either rule. Mr. Zhou has relaxed the no-attention rule as one modest change. The agreed order, each step deciding one thing:

1. **E7, de-confounded composition.** Four single-step operations (addition, subtraction, max, min mod 97) in development; life teaches 12 of the 16 ordered operator pairs in scratchpad form so that every operator appears in both positions, and holds out the four cross pairs. Decides whether P4 was a confound. One afternoon.
2. **If needed: content attention with a copy path.** The rewrite step becomes copying of slots, which generalizes by construction; the operator-to-code legend can be given in context. Decides whether binding needs a mechanism.
3. **The ring with innate operations.** Numbers as bump positions, addition and subtraction as innate shifts, learning confined to the byte-to-displacement and operator-to-direction interfaces, as the fly's central complex composes vectors. Decides whether composition should be innate here. Unites the symbolic and connectome lines.
4. **Meta-learning for compositionality** as the development phase (fresh symbols with meanings given in context, so the network learns to bind at run time), the largest change, saved for last.
5. **The persistent-life demonstration**, independent of the above: one life, eight operations learned in sequence with replay from the store, every earlier one scored after every later one.

Codex's line continues: the path-integration return-leg diagnosis, then arena learning with the eligibility trace.

## 2. What was built, in order

| Phase | Dates | What was done | Outcome |
|---|---|---|---|
| Genesis V66 | Mar 2026 | GRPO-trained "liquid brain" on storybook QA; the original DigitalBaby | Self-correction examples; unrelated to the MiniFly line |
| ELM and early byte tasks | 6–14 Sep | Expressive leaky-memory neurons, Fashion-MNIST, v20–v49 byte experiments | Working classifier; later labelled a different code route; ELM excluded 13 Oct |
| Native MiniFly, iterations 3–23 (V66–V78) | 6–14 Sep | FlyWire v630 connectome: 302 PNs, 5,177 KCs, fast/slow/adaptation state, dopamine-style teaching | Real-path writes change later readout; native learner 62.6% vs an NLMS reference 93.3%; no candidate promoted |
| Workstreams A–A10, BKR, FB1, PostV5 | 19–21 Sep | Retention, interference and first-use return of taught PN cues | Causal interference evidence; endpoint narrowed to "old taught cue returns" |
| BYTE_CORE V9–V9.10 | 22–28 Sep | Byte input; V9.2 engineered byte predictor; V9.3–V9.7 taught-pair retention; V9.8–V9.9B held-out relations | First real E3 signal: +17.2 pp (V9.9A) and +9.6 pp (V9.9B) on untaught pairs, explainable by one learned bit per symbol |
| Objective contract and drift audit | 26 Sep | E0–E4 evidence ladder written after an audit found the goal had shrunk to retention | Rule adopted; the 14 October retrospective found it was not consistently applied afterwards |
| Byte arithmetic evolution V2 | 26 Sep | 1,024 candidates × 128 generations, four architecture families, mixed arithmetic | All families extinct; exact answers under 8%, no causal gain over no-write |
| Package A / Package B | 29 Sep–2 Oct | 832 worlds each on Full151 variants | A: residual private-bank write lifts taught recall 50.8% → 85.0%; B's relation controls were invalid |
| R_center core v1, Persistent Core V2 | 3–5 Oct | Eight-store centered-write engineering core; 64-world ERROR study | Taught OLD 97.1%; held Latin completion causal effect −8.2 pp (memorizes, does not infer) |
| V2 addition milestone | 7 Oct | User sets goal: learn addition 0–4 from blurry feedback, answer untaught pairs | Milestone declared; autonomous-observation baseline adopted |
| STABLE, COV-R, COV-SUPPORT (U042) | 8–11 Oct | Addition on an engineered role/magnitude/ordinal representation | Limited addition confirmed on 64 worlds; user adopts COV-SUPPORT as development core (still the adopted core today) |
| Generic byte IO, kernel V3, SOURCE-CELL, gated feedback | 12–14 Oct | New 4,096×257 hash bank; GRU/LSTM kernels; four-operation attempts | Mixed arithmetic fails; user restricts to addition, bans attention/Transformers and ELM |
| Addition lineage comparison | 15 Oct | Native N151/C151/RC151 vs GRU references KEEP/G62 on raw `a+b=` | Native: at most 7.5% taught, 0% held. GRU: 100% taught, 0–5% held |
| NATIVE-BIND V1, V2.1 | 16–17 Oct | 24 and 72 science jobs, 147k records | NO_ADVANCE; partial taught acquisition, poor delayed retention, no held gain |
| Addition-difficulty diagnosis | 17 Oct | 135 GRU runs varying one factor at a time | Memorization is the default; tally input gives 98%; weight decay 44%; native sensor loses byte order |
| CLS, v3–v6 | 18–23 Oct | Complementary-learning-systems replay; generic recurrent controller | CLS best unseen sums 13%, e-prop 0%; v6-R1: 0% everything, 100% truncation, cue drowned by noise |
| Consolidation | 23 Oct | One branch; census of 53 deliveries and 254 mechanism configurations | Bookkeeping only |
| V144–V157 | 24–26 Oct | Readiness, calibration, learned keys (L-PRED), witness readouts | V155 L-PRED failed its own functional fixture; V156: fitted witness 1,536/1,536, native learner 0/64 |
| V171–V187 | 27 Oct–3 Nov | Fully local H/T learners (37k parameters), inhibition, coherence gating, credit, residuals, EOS credit | Every arm negative by ≤0.05 bits; the learner never beat a unigram on held text |
| Research reset | 2 Nov | Memo names "goal substitution plus constraint accumulation" | Documentation; V179 design only |
| External review | 3 Nov | Fixed sensor (decay 0.95) shown to be the bottleneck; decay 0.5 moves the same learner by 1.0–1.4 bits | Archived; sensory birth still pinned in every later design |
| V190–V223, two parallel branches | 3–6 Nov | Memory protection, bAbI QA, context anchor, action–consequence, successor operator, experience scaling, evolution, BPTT reference | Several bounded positives (section 3); raw-byte reasoning still absent; 6 Nov design is engineering-only |

Commit volume by month on the default branch: 4 in March, 168 in September, 227 in October, 60 in the first six days of November.

## 3. What the fly can actually do

Evidence levels follow the project's own contract: E1 taught association, E2 same fact under a new presentation, E3 never-reinforced relation, E4 broad use.

| Capability | Best evidence | Level | Caveat |
|---|---|---|---|
| Store and recall taught facts | R_center taught OLD 97.1%; Package A R3 85.0%; recovered sparse store 91.6% old recall at 512 facts | E1, strong | Exact-key lookup explains all of it |
| Recall under an unseen rendering of the same fact | Sparse store 36.3% → 80.1% when widened | E2 | Key and value preserved; only the wrapper changed |
| Keep still-valid facts while learning new ones | V197: 44.8% → 94.5% with four-gram features plus usage protection | E1/E2 | Harness supplied teaching positions |
| Reuse a learned relation on untaught pairs | V9.9A +17.2 pp, V9.9B +9.6 pp, both persisting after delay and new learning | E3, narrow | A one-bit class per symbol suffices; synthetic relation, two-way choice |
| Reuse a learned operator after the world changes | V193: 191/192 changed queries with frozen readout vs 3/192 frozen | E3, narrow | Operand positions and temporal strides were supplied |
| Bounded addition with an engineered representation | STABLE / COV-R / COV-SUPPORT, 64 worlds | E3 within the engineered task | Role, magnitude and ordinal-output priors supplied by the experimenter; not raw bytes |
| Learn from consequences of own actions | V201: 100% vs 67% with reassigned consequences; V218: 99.4% vs 42.6% shuffled | E1 (control) | One pending action; four-state task solvable by a small table |
| Recover a lost association by retaining context | V206: 95% vs 14% split control | E1 | Write timing supplied |
| Addition from raw bytes | Native ≤7.5% taught, 0% held; GRU 100% taught, 0–6% held; BPTT32 at 65,536 records 7/1,536 held, 0/384 carry | **not achieved by any learner** | |
| Mixed arithmetic | Evolution V2, VALUE-INTERACT, kernel V3, SOURCE-CELL | **not achieved** | Scope withdrawn 14 October |

Two further results are worth keeping as positive controls: V9.2's frozen byte predictor transfers across held-out authors, and V216's experience scaling improves first-response prediction (6/256 → 25/256) but gives 0/128 on counterfactual pairs where a small input change must change the answer.

## 4. Why it memorizes but cannot reason

These causes are all documented in the project's own reports. They compound.

**1. The sensory front end is a pattern separator, and it was never allowed to change.** The native input is an 88-channel receptor trace decaying with τ = 10 s (decay 0.95 per byte in the later learners), projected onto 5,177 KCs with the top 5% firing. The 17 October order probe found that `3+8=` and `8+3=` share 69% of active KCs while `3+8=` and `3+9=` share 39%: reordering costs less than changing the last digit, and early bytes barely register. The 3 November review measured the ceiling of this code with a converged linear readout: 1.7 bits on training text and 4.05 bits on held text, worse than a one-hot of the previous byte. Changing the single decay scalar to 0.5 moved the unchanged 37k-parameter learner from 4.58 to 3.20 held bits and gave transfer to an unseen text. Three weeks of H/T plasticity variants (V175–V184) sat downstream of this bottleneck and each moved the result by at most 0.05 bits. This is the expansion-coding trade-off described by Babadi and Sompolinsky and by Barak, Rigotti and Fusi: sparse random expansion maximizes discrimination at the cost of generalization. The project borrowed the fly's pattern-separation circuit and asked it to do the opposite job.

**2. One-shot answering turns a rule into a table.** From `3+8=` straight to `11`, the "rule" must be a 100-entry function computed in one step, which is a table. The 17 October diagnosis showed the same GRU reaches 98–100% on unseen sums when quantity is in the stream as tally marks, because the step "one more mark" is iterated with shared weights. The native fly has no working state that each byte updates in an order-dependent way, no internal steps before answering, and no way for a retrieved memory to influence the next hidden-state transition. The 26 October roadmap named this "architecture sufficiency" as a lane; it was never run. Open decisions 1 and 2 from 17 October (order-keeping state, time to think) are still open.

**3. The task is small enough to memorize, and the loss gives no reward for a reusable explanation.** Eighty taught facts determine all twenty held sums (design rank 19), so information is not missing. A 12.6k-parameter network still memorizes all eighty and gets 4% held. Unseen accuracy peaks early (21–24%, single runs 60%) and is then overwritten as the last taught facts are fitted. Errors are ±1 near misses with `a+b` and `b+a` agreeing only half the time: the learner gets the shape of the number line but not its precision, and never ties the two operand slots to one meaning. A rule pays off only when the world is bigger than memory.

**4. The learning signal is weak and the update count tiny.** V181 received about 80 optimizer updates per phase; under-training was mistaken for a mechanism failure. Local rules without backpropagation recovered 3.3% of the matched BPTT advantage (V168). But the constraint is not the whole story: even BPTT at 65,536 distinct records answered 7 of 1,536 held sums, with 3.3 bits of loss at answer positions three to five even when the correct prefix was supplied. The bottleneck is representation and task posing first, credit assignment second.

**5. Every success leaned on supplied structure.** COV-R received roles, additivity and ordinal output; V193 received operand positions; V206 received write timing; V217 received candidate state operations; V201 received protocol frames; V203 used a lexical trigger. Remove the scaffold and the score collapses: V209B's unseen-template selector scored 0% with the correct answer always available; V204's changed cues eliminated target writes. The 4 November memo calls this task leakage. The capability being measured was mostly the scaffold's.

**6. Successes were added across learners that share nothing but a name.** The 14 October retrospective lists ten points where the substrate changed under the same label: Full151 → engineered STABLE/COV-R → a new 4,096×257 hash bank → GRU/LSTM kernels → V181 local learner → V201/V208 action-value route → V217 token state machine. V216 uses a Full151-derived sensor with V181's learner; V218 uses neither; R_center is in none of them. The 5 November cross-branch assessment is explicit that these "cannot be added together as the capabilities of one integrated core."

## 5. Why the project is stuck

- **Version count replaced progress.** Version numbers ran from V3 to V223 in nine weeks, 53 named deliveries by 23 October, 254 historical mechanism configurations, a 44-paper bibliography and a 150-row human-learning catalogue. The 2 November reset memo diagnosed "goal substitution plus constraint accumulation": each failed proxy gate spawned a new version instead of a cause.
- **Constraints accumulated without a feasibility check.** No attention, no Transformers, no ELM, no backpropagation in candidates, no supplied arithmetic features, raw 257-class byte output with own EOS, and a hash-pinned Full151 sensory birth. Each was adopted separately. (Mr. Zhou clarified on 6 November that backpropagation and attention are not banned outright; local gradient updates from nearby neurons are acceptable. See section 9.) Nobody showed the target is reachable inside the remaining space, and the one component that most limits generalization (the sensor) was the one declared immutable.
- **The strongest diagnoses were shelved.** The 17 October memo (order-keeping state, time to think, teaching through the stream, protecting learned structure) and the 3 November review (reopen the sensor decay, separate a memorization path from a representation path, measure the convex-readout ceiling before testing any plastic rule) both name concrete next experiments. Neither was run. Work continued on H/T credit variants and then jumped to bAbI QA and action tasks.
- **Three agents, several branches, one owner.** Codex, Claude and Gemini sessions produced `codex/minifly-consolidated-20261023`, `codex/minifly-v39-v40-source`, two `claude/*` branches and a governance branch. Version numbers were allocated and reconciled across branches (V220–V222 needed a reconciliation commit). The 5 November notes were written to make sense of results produced in parallel on different learners.
- **Engineering cycles were counted as science.** Many deliveries are "three-cycle technical qualification, zero science worlds," including the latest (6 November). The active design pointer still reads "THREE_CYCLES_COMPLETE_ACQUISITION_NOT_READY" for V147 from 24 October. The adopted core is still the 11 October engineered addition package, which the record itself says is not a runtime for any later experiment.
- **Scope kept shifting around the stuck point.** Mixed arithmetic → addition only → generic byte IO → autonomous observation → addition with blurry feedback → text prediction → bAbI QA → MiniWorld action learning → scaling study. Each shift was authorized and documented, but the core question (can this architecture reuse what it learned?) was never answered on a fixed learner before moving on.

## 6. What I recommend

In priority order. The first three are cheap and settle questions the record has left open for weeks.

1. **Freeze version allocation and publish a one-page capability ledger.** No new V-number until a single owner signs off that the proposed experiment addresses a named failure from the ledger with a causal control. The 5 November memo is most of this ledger already.
2. **Reopen the sensory birth as an experimental variable.** Run the 3 November proposal as written: same V181 learner, same data and caps, FE decay 0.95 versus 0.5, fresh worlds and seeds, through the full A → B → corrected-A life. This is the only intervention in the record with an effect of more than a bit. Then split the sensor into a long-trace sparse path for episodic memory and a graded-similarity multi-timescale path for prediction, combined by a learned mixture.
3. **Measure the ceiling before testing any rule.** For every task, report the best convex readout of the fixed code and a BPTT reference. If the native learner is near the ceiling, more credit-assignment work cannot help. If BPTT fails too (as in V223), the task posing or representation is wrong and no local rule will fix it.
4. **Give the learner computation, not just memory.** Add an order-dependent recurrent state (generic, byte in, byte out, nothing task-specific) and allow internal steps or own-output bytes before the final answer. Test the four functions from the 26 October lane C: survive distractors, learned gating, retrieved memory influencing later transitions, transfer to new contents. Use disabled-feedback and capacity-matched controls.
5. **Decide the teaching question Mr. Zhou was asked on 17 October.** Worked steps or traces delivered as bytes are teaching, not wiring. Either allow them in the stream or accept that a learner that only sees `3+8=11` is being asked to rediscover arithmetic from a 100-entry table. Tally grounding showed the mechanism works; prediction-based meaning is the scalable version and needs the ordered state from item 4.
6. **Make the world bigger than memory.** A rule is only worth learning when lookup cannot keep up. Use larger operand ranges, curricula that reuse the same quantity in many contexts, and counterfactual pairs where a small input change must change the answer, with the both-correct fraction as the primary score.
7. **One learner, one life, forward transfer as the yardstick.** Pick one learner, keep it fixed across a family of worlds, and score the transfer area between a pre-exposed and a fresh copy on held-out worlds, as the 4 November strategy memo proposes. Report retention of still-valid facts, correction quality, late-life plasticity and compute separately. Stop combining scores from different learners.
8. **Revisit the constraint stack deliberately.** List each prohibition, who adopted it and when, and whether it is a scientific hypothesis or a preference. Keep the ones that define the research question (local learning, raw bytes). Treat the rest, including the fixed sensor, as variables until the target is reached once in the unconstrained space.
9. **Fewer, larger, pre-registered experiments.** Every launched design should run its science worlds. A design with zero science worlds is not a result and should not get a version number.

## 7. Where the evidence is

- Objective contract and drift audit: `MINIFLY_OBJECTIVE_CONTRACT.md`, `V88_TO_V9_7_OBJECTIVE_DRIFT_AUDIT_20260926.md`
- Narrow E3 positives: `BYTE_CORE_V9_9A/REPORT.md`, `BYTE_CORE_V9_9B_RANK/REPORT.md`
- Memorize-but-not-infer: `DIGITALBABY_CLAUDE_B_D4E4578_20260929/r_center_core_v1/REPORT.md`
- Addition lineage, native versus GRU: `PROJECT_DOCUMENTS/ADDITION_LINEAGE_REVIEW_20261015/REVIEW.json`
- Why recurrent learners memorize addition, and the order probe: repository `addition_difficulty_diagnosis_dev_20261017/REPORT.md` and `MEMO.md` (commits c6d4541, f77491f)
- Ten substrate-switch nodes: `PROJECT_DOCUMENTS/ROUTE_RETROSPECTIVE_20261014/REPORT.md`
- Census of 53 deliveries and 254 mechanisms: `PROJECT_DOCUMENTS/FULL151_VERSION_CENSUS_20261023/MEMO.md`
- Research reset: `PROJECT_DOCUMENTS/MINIFLY_RESEARCH_RESET_20261102/MEMO.md`
- Sensor bottleneck review: repository commit a336bff, `MINIFLY_MEMORIZATION_TO_REPRESENTATION_REVIEW.md`
- Latest cross-branch state: `PROJECT_DOCUMENTS/MINIFLY_V216_V218_CROSS_BRANCH_ASSESSMENT_20261105.md`; branch `codex/minifly-v39-v40-source`, `MINIFLY_RESEARCH_MEMO_20261105.md` and `MINIFLY_SCALING_AND_REASONING_20261105.md`
- Active pointer: `PROJECT_DOCUMENTS/CURRENT_ARITHMETIC_EXPERIMENT_DESIGN.json` (V147, acquisition not ready; adopted core unchanged since 11 October)

Relative paths under `PROJECT_DOCUMENTS/` refer to `DIGITALBABY_CLAUDE_B_D4E4578_20260929/PROJECT_DOCUMENTS/`.

## 8. Discussion: six open questions

One diagnosis sits under all six questions. MiniFly has memory but no computation that is shared across inputs. The KC code is a random hash: every input gets its own private support, and learning adjusts the readout of that support alone. A system like that scales in capacity and never in function class. Nothing learned from one input can make another input easier, because nothing is shared. A transformer, a GRU, or the fly's own central complex all have the opposite property: one set of weights processes every input, so a rule found once applies everywhere.

**1. Scaling helped memory, not reasoning.** Scaling improves generalization only under two conditions: similar inputs land near each other in the representation, so a readout can interpolate between them; and the data exceed what the model can store, so the loss has to compress into shared structure. MiniFly violates both. A new record is a new hash row, and every task so far was smaller than the memory. So widening the store raised recall and left counterfactuals at zero, which is what V216 showed. Transformers scale because the same weights serve every position and the corpus is far larger than the parameters, so the cheapest fit becomes a shared rule. The axis to scale is shared computation plus data diversity, in the regime where the model cannot memorize. A cheap check: fix the learner and grow the fact table until taught accuracy falls below 100%. Only past that point does a held-out score say anything about generalization. The project never ran there. Eighty facts against a 12,600-parameter network is the wrong regime by two orders of magnitude.

**2. 99% accuracy but no grokking.** Grokking needs three ingredients: a task where a compact rule and a lookup both fit the training set; parameters shared across examples so the two solutions compete for the same weights; and a pressure, usually weight decay, that makes the rule cheaper than the table during long training past 100%. Nanda and colleagues showed the rule a network finds for modular addition is a Fourier code in the learned embedding, so learnable keys are a fourth requirement. The 99% tasks had none of these. Keys were fixed, each example had near-disjoint support so nothing competed, training stopped at acquisition, and the harness supplied the composition, so there was no rule left to discover. The 17 October GRU runs are the counterpoint: held accuracy peaked early and was then overwritten as the last facts were fitted, which is grokking in reverse, and adding weight decay moved unseen accuracy from 6% to 44% with some seeds at 90%. Those seeds grokked. The biological form of weight decay already exists in the model: rest. Synaptic homeostasis is global downscaling during sleep. So keep training after taught accuracy saturates, apply multiplicative downscaling to the shared pathway at every simulated rest, and plot held-out accuracy across sleep cycles. If a shared learnable layer exists, held-out should climb after the plateau. If only the hash store is plastic it will not, and that negative result pinpoints the problem.

**3. An end-to-end pipeline for persistent learning.** This is buildable on a laptop, and the biggest mistake so far has been building the pipeline and the local learning rule at the same time. Three state variables at three timescales, one chronological stream, one signal that is always available.

- Stream: one continuous byte stream from procedurally generated small worlds, interleaved with small natural text. No phase flags. Held-out worlds come from the same generator with fresh names and fresh transition laws.
- Working state: an order-dependent recurrent state, generic, never reset. This is the missing component.
- Fast memory: the mushroom body as it is. Sparse hash, one-shot writes, decay. It stores episodes and it is good at that. Keep it.
- Slow model: shared predictive weights that read the working state and the fast memory's output and predict the next byte.
- Signal: prequential prediction error, scored before each byte arrives, plus whatever sparse feedback the world gives. No labels needed.
- Consolidation: during rest, replay episodes from fast memory into the slow model, but only those whose replay lowers prediction error on a small held buffer. That is the Go-CLS gate, and it is the memorization-to-representation transfer made mechanical. Downscaling runs in the same rest.
- Scoring: transfer area between a pre-exposed learner and a fresh one on new worlds, retention of still-valid facts, late-life plasticity, compute. The 4 November memo already defines these.

Build it first with a reference learner: truncated backpropagation over a short window for the slow model and working state, local rule for the fast memory. Get the loop to produce positive forward transfer once. Then replace one component with a local-rule version and see what breaks. The constraint reframe that makes this acceptable: biology does not learn PN-to-KC wiring from reward either. It is developmental. Pretraining the slow code once with gradient descent, then freezing it and learning locally during life, is at least as biological as random wiring. "No backprop" should mean no backprop during life, not no optimization ever.

**4. Arithmetic as a substitute for pretraining.** Arithmetic is a test, not a curriculum. At 0–9 the table is memorizable, the vocabulary is tiny, and there is one relation. Used correctly it is a fine grokking testbed, but at the right scale: modular addition mod 97 or 113 with 30–50% of pairs held out, as in the grokking papers. That runs in minutes on a laptop with a one-layer model, so compute is not the constraint. For a curriculum with real structure on a small budget: TinyStories-scale text, where Eldan and Li found one-to-ten-million-parameter models learn grammar and simple reasoning from a few hundred megabytes, trainable on an M3 in hours; synthetic grammars with compositional semantics; and arithmetic with worked steps in the stream. Lee and colleagues found small transformers learn addition with carries when the data include the intermediate steps. The project's own tally result is the same finding from the other side. Scratchpad addition is the cheap substitute. Plain `a+b=c` is not.

**5. Transformers are associative too, so why do they reason?** The premise is right. The difference is four mechanical properties, not a different kind of memory.

| Property | Transformer | MiniFly |
|---|---|---|
| Keys | Learned, trained to put useful neighbors together | Fixed random hash |
| Composition | Each layer's retrieval becomes the next layer's query | One hop, then readout |
| Shared computation | Same weights at every position | Private support per input |
| Serial time | Can write intermediate results to its own input | One-shot answer |
| Pressure | Corpus far larger than parameters | Memory larger than the world |

Transformers fail in MiniFly's way at MiniFly's scale. A transformer trained on 80 addition facts memorizes, and transformers fail length generalization on addition without positional tricks. Their reasoning appears only once the data make the rule cheaper than the table. So the issue is not the associative nature. It is the four rows above plus the training regime. One more point: top-k KC firing is already hard attention over a random projection. The useful part of attention is learned keys and composition in depth, not softmax. The ban removes the wrong thing.

**6. Gluing many fly brains together.** It depends on what the glue is. Parallel copies, more banks, voting: already tried as HASH8, eight stores and sixteen banks, and it buys capacity, not generalization. Modules whose outputs feed each other in loops: that is a recurrent network, and then credit assignment across modules is the problem you were avoiding. A population that communicates through learned routing: that is attention by another name.

The biologically correct version of glue is the fly's own brain. The mushroom body is the memory. Computation lives in the central complex: a ring attractor for heading, and vector operations in the fan-shaped body that implement path integration with on the order of a hundred neurons, in Stone and colleagues' 2017 model. Path integration is vector addition done by a recurrent circuit, with one shared operation applied to every input. Honeybees learn same-versus-different and numerosity including zero with this hardware. So add brain parts, not brains: an attractor-style working state beside the mushroom body. For addition, numbers become bump positions and adding b means moving the bump b steps, with the byte-to-displacement mapping learned from counting experience. That is the tally insight implemented neurally, and it generalizes by construction because the operation is one shared shift. The honest caveat: it is an inductive bias toward a continuous, incrementable state. It is not an adder, but Mr. Zhou should decide whether it crosses his line on arithmetic structure. A ring attractor is as generic as a GRU gate.

### What I would build first

Three experiments, each about a week of laptop work, each with a crisp failure signature.

1. **Development-versus-life split.** Pretrain a small shared encoder with gradient descent on mixed synthetic and text data, freeze it, run local learning on top for the whole life. Compare against the random-hash birth on forward transfer. If frozen learned keys beat random keys, the no-backprop rule has been drawn in the wrong place.
2. **Sleep as weight decay.** Modular addition mod 97, half the pairs held out, a shared layer plus the hash store, training past saturation with downscaling at each rest. Held-out rising after the plateau is grokking. Flat held-out means no shared pathway is doing the work.
3. **Attractor working state with scratchpad teaching.** A ring or line attractor next to the mushroom body, counting experience in the stream, worked steps shown as bytes. Score never-taught sums with the learner's own output and its own stop. This is the one most likely to produce the first E3 on raw bytes, and the one most likely to be contested as too much structure, so run it with the structure as a declared variable.

The literature pointers in this section are from memory and should be checked before quoting: Power et al. 2022 and Nanda et al. 2023 on grokking, Eldan and Li 2023 on TinyStories, Lee et al. 2023 on teaching arithmetic to small transformers, Sun et al. 2023 on Go-CLS, Tononi and Cirelli on synaptic homeostasis, Seelig and Jayaraman 2015 and Stone et al. 2017 on the central complex, Howard et al. 2018 on bee numerosity.

## 9. Codex's runs of 6 November, and what they settle

**Constraint clarification.** On 6 November Mr. Zhou clarified that backpropagation and attention are not banned outright. A neuron that uses information from nearby neurons for a local gradient update is acceptable, and other ways of associating taught things are acceptable. The real concern is the earlier back-and-forth, with generalization never demonstrated. This changes the reading of the earlier sections: V181's fixed-feedback rule is already a local gradient rule, so "does the local-rule learner generalize?" is a well-posed question that can be tested against a backpropagation reference on the same task. Where this review says "no backprop in candidates", it describes what the memos recorded, not the current rule.

**What Codex ran.** Five packages on three branches (`codex/minifly-ordered-front-end-20261106`, `codex/minifly-recurrent-worked-examples-20261106`, `codex/minifly-three-experiments-20261106`), all dated 6 November.

| Package | Learner | Task | Result | Verdict |
|---|---|---|---|---|
| Reasoning scaling matrix | V216/V181 local rule, width 64 vs 128, 4 vs 32 environments | Directed-graph move/stay with a held diamond | Engineering cycles only; scientific roster not run | No evidence yet |
| Ordered byte front end | V181 local rule, 37,313 params; stock, ordered and bag sensors | Graph, 100,000 A events, two births | Taught 187/256. Held quartets 0/32 in every arm. New renderings of the *taught* topologies: 10–12/128. Balanced exact 10–13% against an adaptive shortcut at 83–89% | Negative. The learner is keyed to the literal byte string |
| Recurrent state and worked examples | Gradient-trained GRU, 33,281 params, 2 models × 3 teaching conditions | Graph; held item = same semantic key, different edge-row order | Taught 24/24 histories above 52/64. Held 0/24 above 48/64, best 35/64. No gain from recurrence or worked steps | Negative. 256 semantic keys is a data-starved regime for a combinatorial task; row-order invariance was not learned |
| Frozen pretrained encoder | GRU pretrained on prose and synthetic text, frozen; linear delta-rule readout | Graph A, then forward transfer to B | Aligned 0/384 taught, stock 10/384. Acquisition gate failed in 24/24 histories | Uninformative. The readout could not even memorize, so the hypothesis was never tested |
| Mod-97 downscaling | MLP, 8-byte window × 8-dim embedding → 128 → 128 → 257, plus a linear KC residual; Adam, backprop | `24+81=08` raw bytes with native EOS; 50% of unordered pairs held, half of those sealed | **87.6–93.5% exact on sealed held sums in 3/3 seeds.** Lookup 0, nearest pair ≤1.1%, frequency 1%. Shared path alone 86–92%; KC residual alone 0.9–1.6%. Weight shrinkage hurt | **Positive. The first clear unseen-input generalization in the project** |
| Attractor ring and worked counting | Gradient-trained 32-unit recurrent net, ring vs rewired | 0–9 addition; 75 taught, 25 held = every pair with both operands ≥5 | Taught 75/75; held 0/25 in all six arms | Uninformative. Pure extrapolation from 75 examples; no learner passes that |

**The result that answers the confusion.** The mod-97 run is the finding. Codex reported it as "early transfer, excluded from the delayed-rise claim", because the experiment was registered as a grokking test and the model generalized immediately instead. That framing buries it. The facts:

| Measure (sealed held sums, no-shrinkage arm) | Seed 73129 | Seed 93683 | Seed 116089 |
|---|---:|---:|---:|
| Exact answer with own EOS | 90.5% | 93.5% | 87.6% |
| Both orders of a held pair correct | 84.5% | 88.4% | 79.7% |
| Carry subset | 90.0% | 95.0% | 89.1% |
| Wrap-around subset | 88.3% | 91.4% | 86.4% |
| Exact or canonical lookup | 0 | 0 | 0 |
| Nearest taught pair | 0.4% | 0.9% | 1.1% |
| Shared path alone | 86.3% | 91.0% | 91.6% |
| KC residual alone | 0.9% | 1.6% | 1.0% |

Raw bytes in, raw bytes and EOS out. No parsed operands, no supplied positions. Both orders of every held pair withheld. About 60,000 parameters in the shared path. Thirty-five CPU minutes for the whole roster. The taught split was 4,704 prompts, about half of the 9,409 possible. This is the regime the review's section 8 described: a shared computation with the data too large to store cheaply, so the loss compresses into the rule. The shuffled-label control was not run (it was conditional on a delayed rise), but with lookup and nearest-pair at or below 1% there is no memorization route to 90%.

The KC measurement deserves its own sentence. The fly's sensory path, sitting next to a working shared path with its own trainable readout, contributed about one percent to held sums. Nothing in the earlier record measured this so directly. The random-expansion path is a memory. It is not where the rule lives.

**Why the other four were negative.** Each either used the local-rule learner whose literal keying was already known (the ordered front end answered 93/128 literal teaching prompts and 12/128 fresh renderings of the same graphs), or set a gate that no small learner could pass (extrapolation from 75 examples; 256 keys for a combinatorial graph task; a linear readout on prose features). A gate that a backpropagation reference cannot pass measures the gate, not the mechanism. This is the same pattern as before 6 November: learner, task and gate changed together, so each negative result has three possible causes and settles none of them.

**Why no grokking appeared.** Grokking in Power et al. appears at small training fractions with weight decay applied throughout training. At a 50% fraction the model generalizes during training, which is what happened. Codex applied shrinkage after the plateau in pulses, which is a different intervention, and it hurt. To see the delayed rise: lower the taught fraction to 25–35%, apply weight decay from the start, and train longer. That is a cheap sweep on the same code.

**What to do next: one anchor, one change at a time.** The project now has an anchor result where generalization is demonstrated. Freeze that setup (task, split, model, optimizer, scoring) as the reference, and require every new mechanism to be measured against it before anything else.

1. **Local-gradient version of the same model.** Same data, same 64→128→128→257 net, replace Adam and backpropagation with V181's fixed-feedback rule, and separately with a layer-local rule. One number answers "do we need full backprop, or does a local gradient suffice?" under the clarified constraint. Minutes of compute.
2. **Taught-fraction sweep.** 50%, 35%, 25%, 15%, with and without weight decay, for backprop and for the local rule. This finds the grokking regime and the data efficiency of each rule on the same task.
3. **Decimal addition on the same net.** Variable length, carries, with and without a scratchpad trace in the stream. This is the project's actual target, reached from a working baseline instead of from zero.
4. **Transfer, done properly.** Pretrain the shared net on one of these tasks with gradients, freeze it, and learn the next task with a local rule on top. The 6 November encoder run failed at acquisition because its readout was linear and its encoder was trained on prose. Give the readout one trainable hidden layer and train the encoder on the task family.
5. **Keep the KC path as the fast fact memory.** The one-percent measurement says what it is for. Facts and episodes go there; rules go in the shared path.

**Evidence level.** The mod-97 result is a gradient-trained reference on a finite modular task with fixed two-byte rendering, three development seeds, and no independent confirmation. Within that task it is E3 with a shortcut audit, lacking the prior-teaching control. It does not show that MiniFly's native learner generalizes. It shows that the task is learnable from raw bytes at this scale, that generalization is measurable in this project, and that the sensory path is not the place to look for it.

### 9.1 Executed the same day: local rule versus backprop on mod-97

Step 1 of the ladder above was run on 6 November at Mr. Xu's request. Full report, code and receipts: [MOD97_LOCAL_RULE_20261106/REPORT.md](MOD97_LOCAL_RULE_20261106/REPORT.md). Same task, split, model shape and loss as Codex's run, KC residual omitted, five arms, three seeds, 12,288 updates, about one CPU minute per run.

| Rule | Optimizer settings | Sealed held exact, peak (3 seeds) | Sealed held exact, final | Taught, final |
|---|---|---|---|---|
| Backprop | Adam lr .003 | 97.3% · 99.3% · 97.7% | 95.8% · 98.5% · 95.4% | ≥99.6% |
| V181 local rule (direct feedback) | Adam lr .003 | 74.9% · 83.3% · 77.7% | 57.4% · 74.1% · 67.5% | ≥98.5% |
| V181 local rule, frozen input code | Adam lr .003 | 70.1% · 84.2% · 69.3% | 62.1% · 78.5% · 65.5% | ≥99.5% |
| Backprop | V181 settings (lr .001, wd 1e-4, dropout .1) | 95.8% · 99.7% · 96.2% | 95.5% · 99.7% · 96.1% | ≥99.7% |
| V181 local rule | V181 settings | 59.0% · 38.4% · 45.4% | 33.7% · 10.5% · 28.6% | 14–39% |

Shortcut ceilings on this panel: lookup 0, nearest pair ≤1.1%.

Three things follow. The local rule is not why the project saw zero generalization: on a learnable task with a shared computation path it reuses the rule on 70–84% of never-seen sums at peak, against a native record that never exceeded a 17-point lift. The local rule is weaker and unstable: it gives up 15–25 points to backprop at peak, and after the peak it keeps fitting the taught set while held accuracy falls 10–20 points, memorization overtaking a partial rule. And the project's shipped configuration, V181's rule with V181's hyperparameters, never reaches taught acquisition on this task and ends below 35% held; weight decay removes weights faster than a random-feedback gradient rebuilds them, while the same settings make backprop the most stable arm of all. The next cheap runs are a stabilizer for the local rule (early stopping on the monitoring split, decay on the output layer only), sign-symmetric feedback, which the clarified constraint permits, and the taught-fraction sweep.

### 9.2 Second round: stabilizers, sign-symmetric feedback, taught-fraction sweep

Forty-five further runs, same task and harness, details and receipts in [MOD97_LOCAL_RULE_20261106/REPORT.md](MOD97_LOCAL_RULE_20261106/REPORT.md).

| Rule | 50% of pairs taught, peak | 35% | 25% | 15% |
|---|---|---|---|---|
| Backprop, weight decay + dropout | 96–100% | 94–99% | 91–96% | 82–91% |
| Backprop, no regularizer | 97–99% | 93–98% | 88–96% | 79–88% |
| Sign-symmetric feedback (local) | 95–98% | 91–95% | 88–93% | 61–81% |
| V181 local rule (random feedback) | 75–83% | 65–79% | 52–66% | 36–39% |

Four things follow. Early stopping on the monitoring split recovers the local rule's peak to within one point, so its post-peak decay is drift, not damage; a lower learning rate removes the decay too, without raising the ceiling. Sign-symmetric feedback, where each hidden layer receives the error through the sign of its own forward weights and nothing else, reaches 95–98% at half the pairs, two to three points under backprop, at backprop's speed; under the clarified constraint this is a local rule, and it is the one to carry forward. The random-feedback rule needs roughly twice the data of backprop for the same accuracy and collapses at 15%. And no delayed rise appeared in any of the 48 runs within 16,384 updates: every curve peaks by 7,000 updates with taught accuracy already at 100%, which is the regime where grokking, if it comes, arrives tens of thousands of updates later under weight decay. A 98,304-update probe at 15% is reported in the folder.

The 98,304-update probe at 15% of pairs also showed no delayed rise: backprop with V181's weight decay (1e-4) settles on a flat 65–72% plateau from 16,000 updates onward with taught accuracy at 100%, and sign feedback drifts to 35–45%. The grokking papers use decoupled weight decay near 1.0, four orders of magnitude stronger, so the project's regularizer is far too weak to produce the late rise rather than the task lacking it. A decay sweep in the 0.1–1.0 range at 15–25% of pairs is the six-minute-per-run test of that.

### 9.3 Third round: strong weight decay, and why there is no grokking

Twenty-four runs of 98,304 updates with decoupled weight decay (AdamW) at 0.1, 0.3 and 1.0, at 15% and 25% of pairs. Decay 0.3 is the best regularizer in the study: 94–99% held at 25% of pairs with taught at 99.9%, held for the rest of training, and 82–89% at 15% where the project's decay of 1e-4 gave 65–72%. Decay 1.0 also repairs the sign-feedback local rule, which ends at 84–88% at both fractions instead of drifting to 35–45%.

There is still no grokking, and the sweep shows why. One seed at decay 1.0 sits near 30% held for 40,000 updates and then jumps to 90%, but its taught accuracy sat at 32% through the same stall and jumped with it. That is a late escape from a partial solution, with taught and held moving together, not memorization followed by generalization. At decay 0.3 held tracks taught from the first checkpoint; at decay 0.1 and below held peaks early and drifts while taught stays at 100%. Across 30 long runs there is no regime where this network memorizes first and finds the rule later. Grokking is not a milestone that generalization must pass through. It appears only where memorization outruns rule-finding, and on this task with a shared path the rule is found at least as fast as the table. The project's 80-fact tables with a random-expansion memory are the opposite regime: the table is always cheaper, so nothing ever pushes toward the rule. Full tables and curves: [MOD97_LOCAL_RULE_20261106/REPORT.md](MOD97_LOCAL_RULE_20261106/REPORT.md).

### 9.4 Experiment 1: development-versus-life transfer (171 runs)

Pre-stated hypothesis, arms and decision rule are in the chat record before launch; full report in [MOD97_LOCAL_RULE_20261106/EXP1_REPORT.md](MOD97_LOCAL_RULE_20261106/EXP1_REPORT.md). Development: backprop on mod-97 addition, or a shuffled-target sham. Life: a new task (subtraction mod 97, or addition mod 89) learned by the sign-symmetric local rule from an aligned birth, a shuffled birth, or a random birth, with backprop from scratch as the ceiling and a scalar-reward three-factor rule as the floor.

| Sealed held exact, final, 3 seeds | Subtraction, 10% taught | Subtraction, 25% | Mod-89 addition, 10% | Mod-89 addition, 25% |
|---|---|---|---|---|
| Aligned birth, warm start, local rule | 84 · 91 · 71 | 83 · 94 · 80 | 70 · 88 · 71 | 83 · 90 · 80 |
| Aligned birth, frozen trunk, local rule | 25 · 48 · 14 | 27 · 52 · 13 | 71 · 88 · 73 | 77 · 92 · 80 |
| Shuffled birth, warm start, local rule | 28 · 1 · 52 | 1 · 1 · 76 | 30 · 1 · 61 | 50 · 1 · 63 |
| Random birth, local rule | 63 · 43 · 77 | 78 · 74 · 76 | 42 · 47 · 54 | 65 · 53 · 59 |
| Random birth, backprop (ceiling) | 71 · 62 · 60 | 95 · 96 · 94 | 29 · 41 · 31 | 92 · 95 · 94 |
| Aligned birth, scalar-reward rule | 2 · 2 · 2 | 2 · 3 · 3 | 6 · 3 · 8 | 6 · 5 · 7 |

Five things follow. The development-versus-life split works for the local rule when life is a warm start: 80–94% on a new task at a quarter of the pairs, and at a tenth of the pairs it beats backprop trained from scratch on both tasks, because backprop memorizes at that dose and the aligned birth does not. The gain is from what was learned: shuffled births are worse than no birth, and a frozen random trunk with the same head gives 7–28%. Freezing the trunk is safe for the same operation and costly for a different one under the local rule, so lower layers should stay plastic at a low rate; the diagnostic with backprop on the frozen trunk (72–83% on subtraction) shows the features support the task and the local rule is what fails on fixed features. A scalar reward learned nothing at this budget: the life rule needs a vector error at the output, which dopamine alone does not supply. By the pre-stated rule the result advances on mod-89 addition and falls just short on subtraction (margins over the random birth +4, +20, +4 at 25%), so the next development phase is a mixed family of moduli and operations before adding working state.

### 9.5 Codex's connectome navigation test (6 November, afternoon)

Codex built the premise test proposed in section 9's discussion: extract the central complex and mushroom body from FlyWire v630 with signed edges (6,704 neurons, 417,204 signed edges, 106 role-pair gain groups, zero fitted values), map the 47 EPG compass cells and the EPG–PEN1 offsets from synapse coordinates, and ask whether the wiring computes. Reports on the consolidated branch under `output/minifly-connectome-navigation/` (commit 67b7b2f).

| Probe, fixed hand-set gains | Native wiring | Degree-matched rewired control |
|---|---|---|
| Hold heading two seconds in darkness | 16/16 | 1/16 (0/16 with recurrence removed) |
| Turn in darkness, 96 turns | 8.6° median error, 81/96 within 20° | 54.5° |
| Steer toward a supplied goal, heading clamped | 94/96 correct initial turn | 49/96 (48/96 with goal removed) |
| Converge on the goal, heading clamped | 75/96 within 20° | 14/96 |
| **Combined loop: steer using the compass's own estimate** | **42/96 within 20°**, compass error grows 11° → 21° over four seconds | 11/96 |

The wiring computes. Heading maintenance, turning and goal steering each come out of the connectome with large margins over a control that keeps every degree and sign, which is the strongest structural evidence in the project's record and the first result on a circuit other than the mushroom body. The failure is at the first closed loop, where accumulated compass drift under self-generated turns breaks steering, and in the unqualified plasticity mask (no direct MBON-to-central-complex edges; 61 one-relay candidates still to adjudicate). Path integration and reward learning were not reached.

The reading matters. Every gain in this model was set by hand and frozen; nothing was fitted. Drift in a ring attractor under self-motion is exactly a gain-calibration problem. This is the case for the development-phase optimizer stated in section 9's discussion: fit the 106 gain groups under the connectome constraints on the combined task, as Lappalainen and colleagues did for the visual system, then rerun the closed loop against the same rewired control. If the fitted native circuit closes the loop and the fitted rewired one does not, the connectome's computation is established for navigation and the life-learning question can be asked on it. That fitting run is bounded: about a hundred free parameters, a few hundred simulated trials, and the existing probes as the score.

### 9.6 Experiment 2: mixed-family development (99 runs)

Predictions stated before launch; full report in [MOD97_LOCAL_RULE_20261106/EXP2_REPORT.md](MOD97_LOCAL_RULE_20261106/EXP2_REPORT.md). Development on six tasks (addition and subtraction mod 83, 89, 97, with a task-code byte) against development on addition mod 97 alone; life on a held-out modulus 73 and on decimal addition with carries, by the sign-symmetric local rule.

| Sealed held exact, final, 3 seeds | add73, 25% taught | sub73, 25% | decimal, 25% |
|---|---|---|---|
| Single-task birth, local rule | 86 · 85 · 65 | 85 · 90 · 70 | 87 · 85 · 85 |
| Mixed-family birth, local rule | 78 · 69 · 63 | 73 · 81 · 68 | 87 · 84 · 79 |
| Random birth, local rule | 19 · 32 · 50 | 55 · 1 · 62 | 77 · 80 · 87 |
| Shuffled birth, local rule | 1 · 1 · 1 | 1 · 1 · 1 | 77 · 72 · 72 |
| Backprop from scratch (ceiling) | 91 · 88 · 86 | 91 · 90 · 90 | 98 · 100 · 97 |

The main prediction was refuted: breadth of development cost rather than paid. The mixed birth reaches only 80–97% on its own six tasks where the single birth reaches 96–100% on its one, and that lower quality carried into life, where the single birth beats it by 2–21 points on both mod-73 tasks. The quality of the development representation matters more than its coverage at this capacity, so the next knob is width, not more tasks. Three findings held: shuffled births are useless, developed births beat backprop from scratch by 20–40 points at a tenth of the pairs, and decimal addition with carries is learnable by the local rule from scratch (77–89% at a tenth), so carries are not the hard part at this scale. One warning: the random-birth local rule collapsed in two of six modular cells, a failure mode a warm start from any sensible birth removes. The decision rule was not met; the single-task birth from Experiment 1 remains the live positive.

### 9.7 Width test (72 runs): breadth stops hurting, never helps

Repeating the mixed and single births at hidden widths 256 and 512 confirms the capacity reading. At width 512 the mixed birth masters its six development tasks (92–97%) and its deficit against the single birth closes to within 5 points on both mod-73 tasks, but in no cell at any width does breadth beat a single task by more than 7 points. At width 512 the local rule from a random birth also reaches 91–94% held at a quarter of the pairs, matching the developed births, and its width-128 collapses are gone; development's value is at a tenth of the pairs (developed 63–84%, random 39–53%). Sign-symmetric feedback scales to 438,000 parameters with no change of settings, three points under the width-128 backprop ceiling. Development is a sample-efficiency device at this scale, not an asymptote device, and breadth is not a lever worth pulling further on tasks this similar. The open rungs are the ones development cannot address: working state for multi-step computation, and retention across a sequence of tasks. Details in [MOD97_LOCAL_RULE_20261106/EXP2_REPORT.md](MOD97_LOCAL_RULE_20261106/EXP2_REPORT.md).

### 9.8 Experiment 3, rung 3: working state for multi-step computation (48 runs)

Predictions stated before launch; full report in [MOD97_LOCAL_RULE_20261106/EXP3_REPORT.md](MOD97_LOCAL_RULE_20261106/EXP3_REPORT.md). The task puts a run of up to 12 distractor bytes between the two operands of a modular sum, so the first operand must be held. Three structures were developed by backprop: a stateless window, an ungated recurrent state, and a gated one (GRU). Then life on a held-out modulus with local rules.

| Held exact on mod 73, warm gated birth, 25% taught | k ≤ 12 distractors | k = 20 (never trained) |
|---|---|---|
| Backprop with credit through time (ceiling) | 86–92% | 82–91% |
| Sign rule with credit through time | 68–86% | 49–71% |
| True gradient, no credit through time | 58–69% | 29–40% |
| Sign rule, no credit through time | 4–13% | 3–7% |

Three things follow. The structure to borrow is a gated working state: developed by backprop it holds an operand across every trained run length (84–94%) and extrapolates to twice that length (82–88%); the ungated state degrades with length and the stateless window is at the floor beyond its window by construction. The within-step local rule that sufficed for stateless networks in Experiments 1 and 2 is not enough for a working state: removing the backward weights costs 5–15 points, removing credit through time costs 20–30 and most of the extrapolation, and removing both collapses the rule. Sign feedback with credit through time is a usable life rule at 5–15 points under backprop; what it still needs, to be local in time, is a per-synapse eligibility trace that the error multiplies when it arrives, which is the e-prop construction and the mechanism dopamine acts on in the fly. One process note: the first attempt trained on every byte and was swamped by distractor noise, so every arm was rerun with answer-only loss and the first attempt is archived in the folder.

### 9.9 Codex's central-complex loop closes under fitted gains (6 November, evening)

Codex ran the gain fit twice and then a confirmation on fresh panels (commits 7aeb7aa, 13deb05, f47b2c1 on the consolidated branch, folder `output/minifly-connectome-navigation/`). The fitted model is the EPG, PEN, Δ7, PFL3 and FC2 projections from FlyWire v630 with signed edges, eight role-pair gains and seven adapter scalars, 15 free parameters; the other 98 extracted role pairs are absent from the loop. Gains were fitted by Adam through the full 80-step closed loop (gradient checked against finite differences to 1e-8), the degree-matched rewired control was fitted identically, three seeds, loss plateaued (goal RMS 15.2° flat over the last 512 of 2,048 updates), gains frozen before scoring.

| Fresh panels, three seeds | Native hand-set | Native fitted | Rewired fitted |
|---|---|---|---|
| Hold heading in darkness | 16/16 | 16/16 | 0–16/16 |
| Dark turns, median error | 8.7° | 8.7–8.9° | 61–63° |
| Steer to goal, heading clamped | 75/96, 12.0° | 84/96, 5.6° | 26–27/96, 40–42° |
| **Combined loop, compass's own estimate** | 45/96, 18.7° | **80/96, 9.8–9.9°** | 20–24/96, 29–31° |

The first fit (128 updates) had reached 55–56/96 and was reported as a failure against a gate I had set above the steering circuit's own ceiling; the corrected brief replaced the gate with the gap to that ceiling, and Codex corrected the report. The converged fit closes the loop: 80 of 96 goals, four below the clamped-heading reference, with the compass at 10° after four seconds of self-generated turning, and a 56–60-goal lead over a control that keeps every degree and sign and was fitted with the same budget and data. All three seeds pass every gate including the dark-turn no-regression check. The 12 remaining misses are the same 12 episodes in every seed, all end with a negative signed residual and a third of them start at −120°, which points at one asymmetric steering parameter rather than at the wiring.

This is the premise test from the discussion section, passed for the compass and steering circuit: the connectome's wiring computes, hand-set gains were the only thing missing, and an outer-loop optimizer standing in for development supplied them in minutes. The navigation domain is now a legitimate second task family for the ladder. What it does not yet show is path integration (the fan-shaped body cell types are among the 98 absent pairs) or any lifetime learning, and the rewired control's 16/16 stationary heading in two seeds is a reminder that the adapters alone can hold a bump; the wiring effect is in turning and steering.

### 9.10 Eligibility traces on the GRU: a negative with a reason (6 runs)

The time-local version of the sign rule, with a per-synapse eligibility trace decayed by the unit's own update gate, scores 2–14% held on the rung-3 task from a warm gated birth, the same as the within-step rule and far below the 68–86% of sign feedback with credit through time; two of three seeds drift upward in loss over life. Qualification showed the trace is implemented correctly (exact equality to the within-step rule at zero decay, higher alignment to the through-time gradient with traces on). The reason is structural: the trace carries credit along each unit's own gate only, and on a GRU the credit that teaches holding runs through other units, since the state is remixed across units every step. The next step is a cell with a diagonal memory, the LSTM cell state or a gated slot, where self-path credit is the whole story and the forget gate is the trace decay by construction. Details in [MOD97_LOCAL_RULE_20261106/EXP3_REPORT.md](MOD97_LOCAL_RULE_20261106/EXP3_REPORT.md).

### 9.11 E4 retention and E5 composition (42 runs)

Predictions on file before launch; full report in [MOD97_LOCAL_RULE_20261106/E45_REPORT.md](MOD97_LOCAL_RULE_20261106/E45_REPORT.md). Learner: the window network, developed by backprop, learning by the sign-symmetric local rule in life.

**Retention (P3) is established.** Four operations learned in sequence by the local rule: the shared path alone forgets each one completely at the next stage (96 → 0%), with weight decay low or high, and an episodic store beside it changes nothing on its own. With self-generated replay from that store, half of every batch being earlier prompts answered by the model itself, all four are held at 77–93% on never-taught inputs after the fourth, within 5–15 points of a backprop ceiling that sees every task's ground truth jointly; replay with random answers retains nothing. The fast memory teaches the slow path, and that is what "persistent" costs.

| Sealed held after four stages, 3 seeds | T1 add97 | T2 sub97 | T3 add73 | T4 sub73 |
|---|---|---|---|---|
| Shared path alone, local rule | 0 · 0 · 0 | 3 · 4 · 1 | 0 · 0 · 0 | 92 · 95 · 92 |
| + store + self-generated replay, local rule | 90 · 88 · 78 | 90 · 93 · 82 | 86 · 84 · 77 | 91 · 93 · 86 |
| Backprop on joint ground truth | 98 · 98 · 95 | 97 · 99 · 97 | 93 · 95 · 94 | 94 · 94 · 95 |

**Composition (P4) is half established.** Direct composition of two taught operations does not transfer across operation types under any rule: backprop fits the taught types to 95% and answers 1% of untaught combinations, the 2018 systematic-generalization failure reproduced. Scratchpad composition, where the learner emits the intermediate result as bytes and continues, transfers partly: 41–58% of untaught combinations exact with the local rule, intermediate right in 60–71%, shuffled control at 1%, and better than backprop's 4–36%. The gap to the predicted 70% has a measured cause: teaching compositions erased the single operations being composed (98% → 0–7%), the same forgetting E4 fixed. The next run, E5-R, teaches compositions with replay of the single operations; prediction cross-type ≥70% with single operations retained. That mechanism is E6's.

### 9.12 Codex's eleven-block mechanism sweep (commit eb60ca9): what it adds

Codex ran the survey's eleven circuits as a Colab screen, 99 jobs (eleven blocks × three arms × three seeds), with a design memo and receipts but no scorecard. Scorecard built from the receipts (`exact_pairs` or within-20° counts out of 64, three seeds):

| Block | Candidate | Matched reference | Control | Reading |
|---|---|---|---|---|
| Cortical feedback (mod-97) | sign feedback 60 · 51 · 59 | backprop 61 · 61 · 64 | random feedback 30 · 37 · 36 | **Independent replication** of round 1: sign ≈ backprop, random feedback far below. Advances by the memo's own rule |
| Actor–critic, delayed reward | eligibility trace 64 · 64 · 64 | immediate credit 0 · 0 · 0 | shifted trace 0 · 0 · 0 | **Positive.** A per-synapse trace solves delayed scalar credit to a discrete choice; this is where traces belong, consistent with their failure on state-holding in rung 3 |
| Sleep replay (A, B, A′ fixture) | blocked replay 53 · 41 · 40 | interleaved 50 · 35 · 38 | pause 50 · 36 · 38 | Small gain (+3 to +6), below the memo's 10-point rule; the fixture forgets little (B is 256 presentations), unlike E4's regime |
| Fast/slow two-store | coupled 37 · 37 · 37 | single store 50 · 35 · 38 | uncoupled 42 · 36 · 37 | No gain; the single store is as good or better |
| Mushroom-body output loop | closed 32 · 37 · 37 | open 32 · 37 · 37 | permuted 32 · 37 · 37 | Identical across arms: the implemented MBON→DAN gain changes no outcome. Uninformative about the circuit |
| Central complex (hand-set gains) | native 32 · 40 · 37 | rewired 12 · 13 · 8 | displaced landmark 0 · 0 · 1 | Consistent with the pre-fit loop; superseded by the fitted result (80/96) |
| Grid/head-direction attractor | attractor 0 · 0 · 0 | ordinary integration 56 · 56 · 56 | reset 0 · 0 · 0 | Candidate destroys an integration its control performs perfectly: an implementation fault, unqualified |
| Thalamocortical three-pass loop | loop 0 · 0 · 0 | three-pass feedforward 0 · 0 · 0 | one pass 33 · 24 · 33 | Both three-pass models fail to train; unqualified by the memo's rule (reference at floor) |
| Cerebellar calibration | 2 · 0 · 5 | 1 · 1 · 6 | 4 · 3 · 4 | All arms at floor; unqualified |
| Working-memory gate, scalar reward | learned 0 · 64 · 0 | ungated 0 · 0 · 0 | yoked 59 · 0 · 0 | Bimodal by seed; reward-only learning unstable at this budget, as the scalar-reward arm of Experiment 1 found |
| Attention reference | 64 · 64 · 64 | 64 · 64 · 64 | 64 · 64 · 64 | Task at ceiling for every arm; reference only |

What it adds: an independent replication of the sign-feedback result on a second machine, and one new positive, the eligibility trace for delayed reward on a choice, which fixes where that rule belongs in the design (action selection in the arena, not the working state). What it does not add: the memo's own three-step qualification was not applied before the campaign, so five of eleven blocks ran with a reference at the floor, a candidate below a perfect control, or arms that do not differ, and 45 of the 99 jobs cannot be read. The sweep confirms the ladder's direction and does not change it; the next Codex step remains the path-integration follow-up. Codex's own scorecard, pushed afterwards (commit deb12b9), agrees block for block: the same two positives, the same five unqualified blocks, replay and the two-store model below threshold, the mushroom-body loop without a discriminating effect, and an explicit statement that the memo's per-block qualification was not documented before screening.

### 9.13 E5-R: retaining the parts does not give composition (6 runs)

Scratchpad composition taught with self-generated replay of the single operations keeps the single operations (80–96%) and loses composition: untaught combinations fall from 41–58% to 2–18% with the local rule, and backprop with the parts fully retained still answers only 1–26%. My explanation for the E5 shortfall was wrong, and the pre-stated rule applies: scratchpad composition in this form is the limit at this scale, and E6 is postponed. The reason is specific: at the final step the learner re-reads a ten-byte window that contains the first operator, so an untaught combination is a window it has never seen; the intermediate is emitted correctly and the re-reading fails. The one experiment worth running on this property is a canonical scratchpad in which the learner writes the next step as a taught single-operation prompt and then solves it, so that cross-type transfer reduces to the established rule acquisition. Proposed, not started. Details in [MOD97_LOCAL_RULE_20261106/E45_REPORT.md](MOD97_LOCAL_RULE_20261106/E45_REPORT.md).

### 9.14 E5-C: the canonical scratchpad fails cross-type for a clean reason (15 runs, one archived rerun)

The learner rewrites each step as a taught single-operation prompt and solves it. The first run failed in every arm because my ten-byte window could not hold the operand being copied; the rerun at eighteen bytes is clean (86–100% within-type) and refutes the prediction: untaught operator combinations 0–6% with the local rule and 0–8% with backprop, because the rewriting itself does not generalize. The network learns "(+,+) → r…r" and "(−,−) → t…t" as whole patterns rather than "each operator → its code" as a separable rule, and an untaught pair of operators is a pattern it has never seen. Composition across operation types (P4) is therefore not achieved at this scale in any of three forms, direct, scratchpad or canonical scratchpad, under either rule, and E6 stays postponed. What the three forms share is a stateless window with no mechanism that binds an operator in one slot to its code independently of the rest of the window. The one structure the survey ranked for exactly that, attention as learned routing, is permitted and is the natural next arm, proposed and not run. Details in [MOD97_LOCAL_RULE_20261106/E45_REPORT.md](MOD97_LOCAL_RULE_20261106/E45_REPORT.md).

### 9.15 E6-route: learned-key routing does not give composition either (21 runs)

A four-head positional-key attention layer over the 18-byte window, feeding the output head only the attended bytes (plus, in one variant, a 16-d summary), was developed on single operations and taught the canonical scratchpad. With backprop it learns the taught types to 82–94% and answers 0–4% of untaught operator combinations, writing the correct second-step prompt for 0–5%; the sign rule is unstable on this architecture (within-type 2–84%). Composition across operation types (P4) is therefore not achieved in four forms, direct, scratchpad, canonical scratchpad and routed canonical scratchpad, under either rule. By the rule on file the composition line stops here. What is missing is a mechanism that makes "which operator" a variable independent of its neighbours; every architecture tried binds the operator to its role as part of a whole-window pattern. That is the known boundary of the systematic-generalization literature, now reproduced under local plasticity with full audits. The program's established results stand: rule acquisition, working state, retention, all with local rules. Details in [MOD97_LOCAL_RULE_20261106/E45_REPORT.md](MOD97_LOCAL_RULE_20261106/E45_REPORT.md).

### 9.16 E7: with the factors de-confounded, composition works under backprop (18 runs)

Four single-step operations (add, sub, max, min mod 97) in development; life teaches twelve of the sixteen ordered operator pairs so every operator appears in both positions, and holds out the four cross pairs. With backprop, the plain scratchpad answers 84–97% of two-step problems whose operator pair was never taught, in every seed (held-pair mean 90–93%), against a shuffled control at 1–2%. The four earlier composition failures were the confound in the E5 design, and the principle is on record: a factor to be composed over must vary independently in training. The canonical rewrite is the weaker form here (34–60% with backprop, inconsistent across pairs); emitting the intermediate and continuing is enough. Under the sign rule the result is partial and under-trained (37–38% held, 39–42% within-type at 8,192 updates on a six-times-larger teaching set); by the rule on file it is being tested at width 512 and at double dose. P4 is achieved under backprop in the de-confounded regime; whether it is achieved under the local rule is the open question of the next run. Details in [MOD97_LOCAL_RULE_20261106/E45_REPORT.md](MOD97_LOCAL_RULE_20261106/E45_REPORT.md).

### 9.17 Codex's second sweep, the "mechanism gaps" (commit 74ca554, 108 jobs): what it adds

Twelve blocks aimed at the gaps the ladder left open, three arms and three seeds each, with a scorecard this time. Readings, with Codex's own caveats where they apply:

| Block | Result (three seeds) | What it adds |
|---|---|---|
| Diagonal working state with an online eligibility trace | trace 100 · 100 · 100 = BPTT; no trace 56 · 76 · 93, on a delayed cue with the gap doubled | **The shelf check from the plan, done.** A time-local rule trains a diagonal memory where it failed on the GRU; P2 in life under a local rule is now supported on a single-cue task |
| Grid transport of learned displacements | structured geometry 100%; shuffled geometry 50–54%; reset 0 | Composition of taught moves to unseen pairs is free on an innate geometry. Direct support for step 3 of the plan, composition built into a ring rather than learned |
| Separate write and read gates | 87–98% vs 57–82% tied, 20–58% always open | The PBWM gating structure helps hold a cue across distractors; capacity is confounded, as Codex notes |
| Multi-action delayed credit | traces 100%; immediate 1–25%; shifted 0–20% | Strengthens the earlier trace positive for delayed reward, now over four-action sequences; the arena rule |
| Fan-shaped-body goal writes (engineered proxy) | aligned 100%; permuted 0; frozen 4–6 | Context-addressed local writes work in a proxy; the real plasticity mask is still unqualified |
| Operator binding with attention, register or static readers | 5–10% on held operator pairs; the executor is 100% given a correct prompt; every rewrite wrong | Replicates E5-C and E6-route: attention readers do not fix the canonical rewrite. Consistent with E7, where the plain scratchpad composes and the rewrite is the weaker form |
| Mushroom-body reversal | DAN addressing 71–79% vs permuted 47–55%; APL on = off | Compartment addressing matters in the proxy, APL gain does not |
| Sparse fast/slow replay | 57–62% vs 51–55% without replay | Same direction as E4, smaller because the fixture forgets little |
| Adaptive versus fixed feedback | 96–97% vs 97% | On a small proxy, learned feedback adds nothing over fixed; a caution for the learned-feedback arm, which should be judged on the composition task where the sign rule actually trails backprop |
| Cerebellar timing | localized features RMSE 0.14–0.20 vs 0.73–0.84 | Synthetic; supports the timing role of a sparse expansion |
| Sleep order and scaling | within 1–3 points | No effect |
| Thalamic feedback gate | 0 in every arm | Unqualified again |

Three of these change what the ladder does next. The diagonal-trace result closes the rung-3 loose end in the predicted direction and lets E6 use a diagonal cell with a time-local rule. The grid result is the first evidence inside the project that composition can be innate, which raises step 3 (the ring with innate operations) from a hypothesis to a tested direction. The operator-binding result, together with E7, settles the form of the scratchpad: emit the intermediate and continue; do not rewrite.

### 9.18 E7-b: the local rule composes too, and is not at a ceiling (9 runs)

Plain scratchpad with the sign rule on the de-confounded twelve-pair set: width 512 raises the held-pair mean from 37–38% to 52–68% (max/min pairs 76–89%, add/sub pairs 26–49%), double dose at width 256 to 42–50%, and every curve is still rising at its last checkpoint with within-type accuracy at 60–70% against backprop's 92–94%. Neither pre-stated threshold held in every seed, so by the rule on file the program proceeds to the persistent-life demonstration with compositions reported rather than gated. P4 stands as achieved under backprop and partially achieved under the local rule, with the learned-feedback arm as the designated test of the remaining gap. Details in [MOD97_LOCAL_RULE_20261106/E45_REPORT.md](MOD97_LOCAL_RULE_20261106/E45_REPORT.md).

### 9.19 E8: the persistent-life demonstration (12 runs)

One learner, one life, local plasticity: eight new operations learned in sequence by the sign-symmetric rule with self-generated replay from the episodic store are all held at 75–98% on never-taught inputs after the eighth, where the same learner without replay holds none (0% at every stage). That is properties P1, P3 and P5 in one life, with the window as working state, and it is the persistent-learning demonstration the project set out to show at toy scale. A ninth stage teaching compositions of the learned operations answers untaught operator pairs at 48–55% in the same life (18–26% without replay; 89–91% for backprop on joint ground truth), so P4 is partially present in the same learner. The measured weakness: the composition stage costs the four carry-type operations 20–35 points (to 48–68%) while max and min hold at 89–94%; under the local rule the shared path cannot absorb 24,000 composition items while rehearsing eight operations at this dose. A longer composition stage with a smaller composition share per batch, or wider capacity, is the single pre-registrable repair. Details in [MOD97_LOCAL_RULE_20261106/E45_REPORT.md](MOD97_LOCAL_RULE_20261106/E45_REPORT.md).

### 9.20 E10: composition is free once the operations are innate (27 runs, one archived rerun)

The ring with innate operations: numbers as bumps over 97 positions, add and sub as circular shifts, max and min from cumulative distributions, learning confined to three interfaces (digits to bump, operator byte to operation, bump to digits), parsing by position supplied. In the seeds whose interfaces developed fully, two-step problems on untaught operator pairs are answered at 100% zero-shot from single-step development alone, and at 100% on the confounded cross pairs that defeated every learned architecture in E5, E5-C and E6-route, with a flat ablation at 4–10%. Composition accuracy tracks interface accuracy (62–93% single-step gives 42–88% two-step), so the ring turns composition into an interface-learning problem, which is the project's own idea confirmed at toy scale. Two measured limits: the interfaces are slow and seed-sensitive to develop through a 97-way bump, and life under the sign rule at the usual decay erodes exact interfaces rather than refining them (a lower decay or a write-when-wrong gate is the fix the self-adjusting-rule allowance permits). The supplied parsing is the next thing to learn. Details in [MOD97_LOCAL_RULE_20261106/E45_REPORT.md](MOD97_LOCAL_RULE_20261106/E45_REPORT.md).

## 9.21 Codex's E9, the price of persistence (7 November 2026, assessed)

Codex ran the three compute sweeps of the E9 brief on the unchanged E8 harness (commits 8581c05, 64e6e5f, merged with E10 in 670ba90). The parameterized script reproduces the archived E8 receipts on its default path, and every source file and birth checkpoint is hash-locked (`E9_LOCK.json`, `E9_DEFAULT_PARITY.json`). Fifteen new lives, three seeds each; the report is `MOD97_LOCAL_RULE_20261106/E9_REPORT.md`.

| Schedule, replay share | Mean of eight operations before composition | After composition | Held composition pairs |
|---|---|---|---|
| blocked, 1/2 (E8) | 89–90 | 76–80 | 48–55 |
| blocked, 1/4 | 78–81 | 62–70 | 47–62 |
| blocked, 1/8 | 55–58 | 46–53 | 42–54 |
| blocked, none | 12–13 | 0 | 18–26 |
| interleaved, 1/2 | 91–92 | 75–77 | 44–50 |
| interleaved, none | 91–92 | 0 | 65–70 |

What it settles. Replay share buys retention roughly linearly at this budget: a quarter share costs 9–11 points on the eight-operation mean (my prediction of "within five" missed), an eighth costs 33–37 points on the earliest four. Interleaving the eight operations removes the need for replay while they stay in the stream, but the first distribution shift afterwards, the composition stage, erases every standalone response without replay, in every seed, to exactly 0%. So the forgetting is not a property of blocked presentation; it is a property of any change of stream, and replay (or something that does its job) is a standing requirement of this learner, not a scheduling fix. The composition itself was learned better without replay (65–70% held pairs versus 44–50%), which is the same budget trade-off seen in E8: whatever share of the batch rehearses old skills is not learning the new one. The width-512 plain scratchpad under the sign rule at 16,384 updates reaches 75/62/76% on held pairs, two seeds above the 70% target and one still rising; the gap to the backprop ceiling (84–97%) persists but is closing with dose.

What it does not settle. All zeros are own-output scores on the original prompts; whether the internal arithmetic survives (and could be revived by a short reminder) was not measured, and is the natural next check if replay share must be cut.

## 9.22 Reference: Raviv et al., *Science* 394, eaeh9311 (5 November 2026), "The first 1000 days: modeling language learning from individual children's everyday speech"

Child-specific self-supervised models (DINO-SR encoder over 20-ms speech units, transformer decoder predicting the next unit) trained day by day, in recorded order, on ultradense home recordings of eight children's first 1000 days; nightly replay of recent input with probability decaying in temporal distance; an "error-accumulation window" (minutes of input per weight update) fitted per child; a curriculum of growing daily exposure and a staged decoder learning rate. Phoneme-like units emerge by 10–12 months and word-form vocabularies of thousands track each child's measured vocabulary.

What it gives this program:

- **Independent confirmation of E8/E9 on replay.** With one exposure and no replay the model learned essentially no words by 24 months; replaying each day's events about 25–30 times sufficed and more gave little (their Fig. 5F). "Experience alone was insufficient ... it had to be augmented by memory of past input." That is the E9 conclusion from the other direction: replay is a standing requirement of a learner on a lived stream, not a scheduling fix.
- **A protocol for the language stage (section 1A.7, condition for scaling).** One chronological stream, no epochs or reshuffling, exposure growing with simulated waking time, evaluation each week on the following week's unseen input, and the learner's trajectory compared with a developmental reference rather than a static benchmark. Because the objective is prediction of the stream itself, replay is re-feeding stored recent input, with no self-generated answers: once MiniFly's life task is next-byte prediction on text, the episodic store need only hold raw recent bytes, which removes the fragile regenerate-the-answer step that E4 and E8 depend on.
- **Three knobs we hold fixed that they fitted per child**, each a one-arm addition to the E9 harness: the error-accumulation window (our fixed batch of 64; theirs 4 min to 1 h, shorter better), the replay amount, and recency-weighted rather than uniform replay sampling. E9 showed the earliest operations suffer most when the share is cut, so recency weighting may hurt a sequence of distinct skills; that is the test.

What it does not give: credit assignment and locality (backprop and a transformer; the authors state the architecture is cognitively, not neurally, plausible), and composition or reasoning (syntax and generalization are explicitly deferred). It supports P1 and P3 at scale and is silent on P4 and P5.

## 9.23 E11: a diagonal memory cell with a time-local trace inside the E8 learner (7 November 2026, executed)

Full section: `MOD97_LOCAL_RULE_20261106/E45_REPORT.md`, E11; predictions and two amendments in `E45_PREDICTIONS.md`.

**Established: property P2 by a local rule.** A diagonal memory cell (each unit reads only its own past state; gates driven by the input) makes the per-synapse eligibility trace exact, which the qualification confirms to one part in a million against backprop through time. In life, with the sign rule everywhere and that trace for the cell's synapses, the learner acquires from a birth that never required it the ability to hold an operand across twelve distractor bytes:

| Add mod 97 at k = 12 after its stage | Trace | Within-step | Backprop through time |
|---|---|---|---|
| No-memory birth | 90 / 78 / 63 | 21 / 32 / 5 | 91 / 92 / 94 |
| Memory birth | 92 / 86 / 82 | 60 / 35 / 25 | 94 / 90 / 96 |

At k = 20, never presented, the trace arms are 52–78% (acquired) and 76–88% (developed). Silencing the cell drops every such score to ≤ 6.5%. The trace needs no backward pass through time and no stored activations; it is credit through time without backpropagation through time, which the E3 trace through a full GRU could not do.

**Refuted: retention in this learner.** After four operations in sequence with half-batch episode replay, add and sub are at 0–15% in every trace arm (from 72–92% after their stages) and the birth operations are largely gone after the first life stage; the backprop ceiling on joint data holds every operation at 87–98%. The E8 window net held eight operations at 75–98% under the same rule; adding the cell made forgetting far worse. Not diagnosed; candidates, one arm each: decay 1.0 per update (E12-B says it costs E8 12–15 points on its own; E13 Part A tests it), the cell as an interference channel (add at k = 0 fell faster than at k = 12 in two seeds, so the window path is overwritten first), rote learning of fixed-letter episodes, and blocked presentation.

**Harness record.** Two restarts in one day, both at the seam between the new learner and a mechanism inherited by name: a buffer-fill slip (now caught by a full-loop dry run, `E11_STAGE_UPDATES`), and E8's store-mediated replay, which silently requires a stage's episodes to fit the 65,536-key store and which random distractor letters break (393,000 distinct prompts per stage). E11 therefore replays stored episodes directly, which is what E8's store delivered when its keys fit. The partial launches are archived. The general rule, now applied: before launch, list every carried-over mechanism and the condition it needs in the new task.

**Where this leaves the five properties.** P1, P3 (window net), P5 established; P2 now established by a local rule inside the persistent learner; P4 achieved under backprop when de-confounded, partial under the local rule (52–76%, and E12's batch-256 package may close much of the gap: E13 Part B), free by wiring with the ring. The open conflict is P2 against P3 in one learner: the cell that gives working state under a local rule also makes the sign rule forget faster. That is the next thing to resolve, before any scaling.

## 9.24 Codex's E13: the batch package is mostly decay; it transfers to composition; path integration stopped (7 November 2026, assessed)

Commits 795836b, 3214dc7, merged 85b0e66; report `MOD97_LOCAL_RULE_20261106/E13_REPORT.md`. Eighteen lives plus two default reproductions, three seeds each, hash-locked as before.

| Arm (E8, blocked, share 1/2, sign rule) | Eight-operation mean after composition | Held composition |
|---|---|---|
| batch 64, decay 1.0 (E8) | 76–80 | 48–55 |
| batch 64, decay 0.25 | 94–95 | 81–85 |
| batch 256, decay 1.0 (E12) | 92 | 69–73 |
| batch 256, decay 4.0 | 24–30 | 11–13 |
| batch 256, lr .002 | 91–92 | 75–77 |
| batch 1,024 | 89–91 | 40–42 |
| backprop ceiling, batch 256 | 96 | 84–87 |

What it settles. The decay per unit of exposure is the dominant factor: cutting it four-fold at the original batch gives the best retention and composition of any local-rule arm in the program, 94% and 81–85%, within 2 and 3 points of the batch-256 backprop ceiling's retention and 2 to 6 points of its composition. Raising it four-fold at batch 256 collapses the learner, worse than the baseline, so decay and step count interact rather than cancel. Batch 1,024 keeps retention but halves composition, so bigger is not uniformly better. The E12 package applied to the de-confounded composition test (E7, width 512, fixed exposure) gives 88 / 83 / 88% on held pairs against 75 / 62 / 76 before, two seeds still rising; the historical backprop comparator at that width is 84–97%. So the local rule's composition gap, which this review has carried since E7, is mostly a setting, decay 1.0 per update, not a limit of the rule. The backprop ceiling also moves, by 1–2 points of retention and down a few points of composition at batch 256, so the effect is not exclusive to the local rule.

What it does not settle. Why decay 1.0 hurts the sign rule so much more than backprop (the ceiling at decay 0.3 holds everything) is undiagnosed; the plausible account is that the sign rule's updates are noisier per step, so the same decay erases more signal per unit of learning. E11-D tests decay 0.25 on the cell learner directly.

Path integration. The follow-up ran to its prescribed stop: with the heading clamped to the true heading by an exact inversion of the fixed observer, the return residual is still 0.71 of the outbound distance in every native seed, above the 0.30 gate, so no refit was done. Turn-only compass errors of 38–41° are real but not the whole problem: the extracted PFN→hΔB→PFR wiring supplies an instantaneous displacement-direction signal and the model supplied the accumulator, and that package does not integrate. This line is closed unless the accumulator is taken from the connectome as well.

## 9.25 E11-D: why the trace learner forgot, and the settings that fix most of it (7 November 2026, executed)

Section E11-D of `E45_REPORT.md`. Four one-change arms on the E11 memory-birth trace learner, three seeds.

| Arm | add+sub mean at k = 12 after four operations, per seed |
|---|---|
| E11 baseline (decay 1.0, fixed episodes, blocked) | 13 / 1 / 4 |
| decay 0.25 | 56 / 53 / 49 |
| fresh-render replay | 46 / 13 / 2 |
| interleaved, no replay | 74 / 13 / 14 |
| decay 0.25 + fresh-render replay | 69 / 65 / 56 |
| backprop ceiling | 89 / 90 / 94 |

Decay 1.0 per update was the dominant cause, as Codex's E13 found for the window net the same day; fixed-letter replay episodes a second, smaller one that shows at k = 0 once decay is fixed (the window path memorizes 1,024 fixed prompts rather than keeping the rule); blocked presentation is not the cause, since interleaving fails to acquire the carry operations at all in two seeds. With both fixes the cell learner keeps add and sub at 52–72% and max and min at 87–96% after four operations, against a ceiling of 88–98%. The P2-against-P3 conflict of 9.23 is therefore mostly a setting, not a property of the cell; a 20–35 point gap on the carry-type operations remains, with lower decay, replay share and batch 256 as the candidates (E13-B, E14). Going forward the cell learner runs at decay 0.25 with fresh-render replay.

Program-wide, the lesson of 7 November is that decay 1.0, chosen in E8 for the local rule and carried since, has been the largest single handicap on the sign rule in E8, E10, E11 and the E7 composition gap. The settings question is now in Codex's hands (E14: fresh-seed confirmation, the decay curve, and the alignment diagnostic that should explain why the sign rule is so much more decay-sensitive than backprop).

## 9.26 E15: the fixed settings applied (7 November 2026, executed; Part C pending)

Section E15 of `E45_REPORT.md`. At decay 0.25 with fresh-render replay, three seeds each:

- **Mixed stream.** The sign rule with the trace learns four competing operations from one interleaved stream to 83–98% at k = 12 (k = 20: 83–97%), within 2–10 points of the backprop ceiling; at decay 1.0 two seeds had learned nothing of add and sub. The interleaved failure that would have been a first-order problem for the language stage was decay.
- **Acquisition and retention in one life, no-memory birth.** Holding is acquired at 84–88% in every seed (63–90% and seed-dependent at decay 1.0), and after three further operations add+sub are kept at 35–48% (about 1 at decay 1.0); half the retention of holding developed at birth.
- **Ring under the local rule.** The sign-rule life no longer erodes the innate interfaces in two seeds (100 and 63→67%); in the third 15 points of the carry interfaces are lost (96→81); the comparison interfaces that the confounded life never trains stay at 100 / 47 / 84% instead of decaying to 10–17%. The ring composes by wiring and learns by the local rule in one life.
- **Split write and read gates:** births match the single-gate births; lives pending.

Standing of the five properties after today: P1, P2, P3 and P5 are established under a local rule at toy scale with the settings decay 0.25, fresh-render replay and the exact trace; P4 is achieved by wiring (ring) and is within 5–10 points of backprop by learning (E13-B). The open gaps are retention of holding acquired in life, the carry-type retention gap of the cell learner (E11-D), one seed of ring erosion, and everything about scale. Codex's E14 (fresh seeds, decay curve, alignment diagnostic) is the confirmation step before the settings are adopted for the language stage.

**9.26 addendum, Part C.** Split write and read gates on the cell do not help: after four operations the add+sub mean at k = 12 is 60 / 51 / 56 against the single tied gate's 69 / 65 / 56 at the same settings, and extrapolation to k = 20 is worse. Codex's shelf result does not carry into this learner; the single gate stays. E15 is complete: P15.1 holds, P15.2 holds on acquisition and narrowly misses on retention, P15.3 refuted, P15.4 holds in two seeds of three.

## 9.27 Codex's E14: the decay setting confirmed on fresh seeds, the curve, and a refuted explanation (8 November 2026, assessed)

Forty-six jobs (commit 29b44a7; `MOD97_LOCAL_RULE_20261106/E14_REPORT.md`, predictions and lock written before fitting, exact history parity with the E13 and E8 receipts verified). Three parts.

**A, fresh seeds (200003, 300007, 400009; new births and splits).** E8 under the sign rule at decay 0.25: retention of the eight operations after composition 93.5 / 94.5 / 94.5% (decay 1.0 on the standing seeds: 76–80%); held composition 84.6 / 76.7 / 72.0% against a backprop twin at 89.4 / 88.4 / 90.3%. The third seed misses the predeclared 75% gate, and the miss is one pair: add-then-subtract composes at 18.7% (intermediate 19.8%) while subtract-then-add is at 79.8% and both operations alone are above 93%. That is a routing failure of the E6/E7 kind, not forgetting, and it has happened once in six seeds at decay 0.25. On E7 composition at width 512 the decay-0.25 package reaches 88.7–90.0% held-pair mean, 2.2–3.2 points below a backprop comparator matched in width and dose for the first time; the batch-256 package reaches 83.6–86.8%, 5–8 points below. So the two packages are not equivalent: decay 0.25 is the better one, and the local rule is now within three points of backprop on composition.

**B, the curve (standing seeds).** E8 held composition at decay 0.1 / 0.25 / 0.5 / 1.0: 82 / 82 / 79 / 55, 79 / 81 / 78 / 52, 85 / 85 / 81 / 48. E7 held-pair mean: 92 / 92 / 88 / 75, 90 / 89 / 86 / 62, 92 / 92 / 89 / 76. A plateau from 0.1 to 0.25 (differences of 1–2 points in both directions), a cost of 3–4 points at 0.5, and the collapse only at 1.0. My prediction of a 5-point drop at 0.5 missed in every seed; nothing argues for sweeping lower. **Decay 0.25, fresh-render replay, batch 64 and the exact trace are adopted as the toy-scale settings, on E13, E14 and E15.**

**C, the alignment account is wrong, and its question was badly posed.** The sign-rule update's cosine to the true gradient in the hidden layers is 0.81–0.90 (predicted below 0.5); the Adam step-to-weight ratio at decay 1.0 is 1.3–1.6 times that at 0.25 (predicted at least 2); first-layer weight norms do not fall at decay 1.0. So decay does not strip more signal from a poorly aligned update; the update is well aligned. More to the point, the asymmetry the diagnostic was meant to explain has never been measured: the backprop ceiling has always run at decay 0.3, never at 1.0. The simplest reading of the curve is that decay 1.0 is too much for this learner under either rule. The applied-step cosines of about 0.3 for every arm, backprop included, are a property of Adam's per-coordinate scaling and carry no information about the rule. Codex also notes that the fresh E7 lives give no standalone exact answers after composition training; that is E7's design (no replay of standalone items; it is a composition test, not a persistence demonstration) and is why E8 exists.

**Also delivered:** an inference-only speed-up (final-window evaluation, no redundant store lookups), 1.5× on training and 7–12× on replay generation, qualified exact on one full life and 112 diagnostic checkpoints, deployed for Codex's future jobs only. The cell harness has its own response loop and does not use it.

**What follows.** Codex: E14-S (`CODEX_TASK_E14S_DECAY_CONTROLS_20261108.md`): backprop at decay 1.0 to test whether the asymmetry exists; three more fresh seeds so the composition gate rests on nine; within-stage checkpoints on the composition stage so the failing pair shows whether it never rises or collapses. Here: E16, the Phase A exit gate, one cell learner at the adopted settings on a single stream that carries every toy family (operations with distractors, taught compositions, a later family of operations), scored on every family against a backprop twin on the same stream, with retention measured after the stream moves on.

## 9.28 E16: the integrated toy learner misses the Phase A gate on retention, not on acquisition or composition (8 November 2026, executed)

Section E16 of `E45_REPORT.md`; `exp45/SUMMARY_E16.md`. One cell learner at the adopted settings on one six-block stream (mod-97 operations with distractors, then the twelve taught operator pairs, then mod-73 operations), scored on every family after every block against a backprop twin on the same stream. Three seeds.

- **Acquisition is close to backprop for every family while it is current:** operations with distractors 93 / 92 / 86% at k = 12 (twin 94), the four never-taught operator pairs 87 / 84 / 87% (twin 94 / 91 / 94; the carry pairs 75–84% against a 1–2% ceiling), the later operations 91 / 90 / 82% (twin 94 / 93 / 93). Composition by the local rule inside a learner that also holds operands across distractors is new to the record.
- **Retention fails when the stream changes form.** The operations lose 11–21 points over the four later blocks (the loss comes during the composition blocks and partly recovers when operation-form items return); the compositions lose 30–41 points in the two blocks after them (carry pairs 75–84 → 18–38%), with the earlier family in the replay half of every batch throughout. The twin loses nothing. The gate (80% of the twin on every family) is met on the last family, on the first family in two seeds, and on composition in none.
- **The cell is still the holder** (silenced: 3–7%), extrapolation to k = 20 survives, and the twin's success says the 10-byte window is not the limit.

P3 is therefore the property that fails at the integrated scale, under settings that gave 94% retention in E8 with one form and a window net. Two differences from E8 are the candidates: the replay (1,024 entries per block, uniform over the buffer, so an earlier family's share and diversity both shrink) and the cell learner itself (E11-D already showed it forgets carry operations more than the window net). E16-D runs the three controls that separate them: the local rule on the twin's own data stream (perfect replay), balanced per-family replay from the same buffer, and balanced replay from a buffer that keeps every presented item.

## 9.29 Codex E14S decay controls and nine-seed composition confirmation (8 November 2026)

All 13 jobs finished with no tuning; the full checkpointed seed-400009 life matches E14 exactly apart from timing and the added read-only trajectory key. BP at decay 1.0 retains 85.40 / 83.44 / 81.15% after composition, versus 94.17 / 95.75 / 93.94% at 0.3: higher decay harms BP too. The literal <=85% gate misses by 0.40 points in one seed, and none reaches 92%; the experiment does not establish identical decay sensitivity, since BP and local replay/teaching differ. The three new local 0.25 seeds (500011 / 600013 / 700027) have held composition means of 80.92 / 80.92 / 83.88%, with every pair above 50%; combined with E13/E14, 8 of 9 seeds reach 75%, passing the enlarged gate. The failing seed-400009 `+-` pair stays at 3.33 / 2.83 / 5.00 / 1.17 / 3.83 / 9.17 / 16.33 / 18.67% across 1024-update samples, so no sampled strong acquisition then collapse is observed. `-+` first exceeds 60% at 6144, not by 2048 as predicted; 1024-step sampling and late rising scores do not prove it could never be learned. Report, all per-pair/operation tables, costs and parity provenance: `MOD97_LOCAL_RULE_20261106/E14S_REPORT.md`. Same-source seed confirmation, not an independent-code replication; E16 was untouched.

## 9.30 E16-D: the forgetting was replay diversity, not the local rule (8 November 2026, executed)

Section E16-D of `E45_REPORT.md`. Three controls on the E16 stream, three seeds each. With the twin's own data (fresh items of every seen family, equal weight) the sign rule with the trace keeps the operations within 1.2 points and the compositions within 1.0 point of their peak through the rest of the stream, 3–9 points below backprop in level and flat where E16 fell 11–41 points: the local rule does not forget on its own. Keeping every presented item in the replay buffer, with the same per-family sampling, recovers the compositions to 80–85% (E16: 46–57%) and the operations to 78–90% (E16: 65–83%); re-sampling the same 1,024-item-per-block buffer per family changes nothing, as was written in the predictions file before the result, after Codex's review of the E16-W brief showed that that control had no intervention behind it. So the 1,024 replayed items per block were memorized rather than kept as a rule, the effect E11-D found for distractor letters, now on operands. One residual is exposure share: in the weakest seed the operations still fall to 61% during the composition blocks under full diversity, where the learner's exposure to them is half the twin's.

Two consequences. The Phase A gate is reachable with a replay fix, not a learner change, and E16-D2 (running: an explicit share arm, and backprop on the learner's own batches) finishes the attribution. The fix has to be bounded: a buffer that keeps every item grows with life and fails the scaling requirements, so E16-R will use a per-family reservoir of fixed capacity and find the capacity at which diversity suffices. After that the parser inside the replay, which re-renders stored operands, is the one scaffold left on the learner's side (9.28's discussion), and its replacement by the learner's own generated episodes is the next design step before Phase B.

## 9.31 Codex E16W: the window net also forgets; exact-batch BP does too (8 November 2026)

All twelve amended lives finished (three standing seeds, four arms; eleven new child processes with observed exit zero plus one retained ceiling with unknown numeric exit status), after the default reproduction and revised technical checks passed. The original null balanced arm was stopped after it had started, its partial results/costs preserved, and the revised predictions were locked before the revised launches; this is not a pre-original-launch registration. Uniform local replay loses 45.58 / 46.42 / 48.08 points of held composition after four later mod73 stages, ending at 35.33 / 31.25 / 37.29%. A real 25%-of-batch composition share raises finals to 41.58 / 42.12 / 45.46%, but reduces the loss by only 13.7 / 23.4 / 17.0%, missing the predicted one-third reduction in every seed. BP at the same birth, decay 0.25, AdamW settings and exactly the same recorded input/target batches also loses 44.17 / 42.25 / 38.92 points, ending at 43.42 / 45.29 / 49.88%; all three 57,344-batch streams were independently recounted and matched to both learners. The richer joint-ground-truth BP ceiling retains 90.50 / 90.75 / 91.88%. Old mod97 means already fall during composition (93.33 / 94.92 / 95.04% to 76.00 / 78.88 / 85.33%) and remain below 90% afterward. Thus amended P1 holds and P2/P3/P4 miss in every seed. The loss is not cell-specific or exclusive to the local update, while BP's remaining 8–14-point advantage means the rule is not proved irrelevant. Replay coverage, generated-target quality, store dynamics and their interactions remain unseparated; this is a fixed local-generated experience comparison, not a demonstration that all replay fails. E16-D's structured replay recomputes exact targets, whereas E16W replays self-generated targets, so the studies are not equivalent. Full per-operation/pair/final/full-scratchpad tables, realized exposures, batch streams, hashes, lower-bound costs and interrupted provenance: `MOD97_LOCAL_RULE_20261106/E16W_REPORT.md`. Bounded, parser-free retention remains open.

## 9.33 E16-D2 and E16-W together: coverage, not share, and not mainly the rule (8 November 2026, executed and assessed)

Section E16-D2 of `E45_REPORT.md`; Codex's E16-W in `E16W_REPORT.md` and 9.31. Three results, two harnesses, two learners, the same answer.

- **Share is not the lever.** Doubling the earlier families' exposure in the cell learner leaves the operations' loss during the composition blocks unchanged and trims the compositions' loss by 4–20%; on the window net a 25% composition share buys 6–11 points, 14–23% of the loss. Codex's objection to my first share design was right, and the proper share arm confirms that share was never the cause.
- **The rule matters on poor data, not on rich data.** Backprop on exactly the learner's own batches keeps the operations but still loses 17–28 points of composition (ending 65–77% against the local rule's 46–57% and the local rule on the twin's data at 83–87%). On the window net, matched-batch backprop ends 43–50% against the local rule's 31–37% and a 91% ceiling. So on 1,024 memorized items per block both rules forget, the local rule about 20 points more; with full operand coverage both are flat.
- **Nothing is specific to the cell learner.** The window net's mod-97 operations fall from 93–95% to 76–85% during the composition stage, as the cell learner's did in E16, and its compositions lose 46–48 points over four later stages.

Taken together with 9.30: the integrated learner forgets because its replay covers too few distinct items, not because of exposure share and not primarily because of the sign rule, which pays a secondary penalty on impoverished replay and none on full coverage. The one intervention that worked, replaying every presented item, is unbounded in life. E16-R therefore asks a single question: at what fixed per-family reservoir capacity does operand coverage suffice? That capacity, not a learner change, is what the Phase A gate now turns on. The cross-harness agreement between E16-D2 and E16-W is the first time two learners in two harnesses have confirmed one mechanism in this program.

**9.33 addendum, E16-V (Codex's verification, 8 November 2026).** All 3,124 numbers in the E16 tables match the receipts and the shortcut ceilings are reproduced independently; the discrepancies are in my prose and literal gate reporting, corrected in `E45_REPORT.md` section E16-V: the current-family gap to the twin is 1–11 points, not 1–9; the share arm changed only the second block of each pair's exposure and did not double it; P16D.4 and P16D2.3 miss their literal thresholds (acquisition was better than predicted, not worse); the silenced-cell claim holds for the operations with distractors and the carry compositions, not for max/min mod 73; and every receipt mislabelled arm D's decay as 0.25 (its optimizer runs at 0.3), now fixed in the harness. The new finding is on the window net: E8-style replay drew the same 1,024 composition prompts throughout and 14–21% of its replayed targets were wrong, because E8 replays the learner's own answers. That is a second candidate cause for the window net's forgetting, absent from the cell harness (which replays observed answers), and E16-WR now carries an arm to separate it. Every E8-family result since E8 ran with partly wrong self-generated replay targets late in life, by design; readers of those results should know it.

## 9.34 E16-R: bounded replay recovers most of the retention, monotonically in capacity, and the gate is missed by one point (8 November 2026, executed)

Section E16-R of `E45_REPORT.md`. A per-family reservoir of fixed capacity, filled by Algorithm R (bounded memory, constant work per event), at 2,048 / 4,096 / 8,192, three seeds. Retention rises monotonically with capacity in every seed: composition retention after the later family is 44–64% with about 2,000 distinct compositions held, 68–74% with 3,800, 73–76% with 7,000, and 80–85% with all 24,000 (E16-D); the operations recover likewise. A reservoir of 2,048 reproduces the old 1,024-per-block buffer, so capacity rather than sampling is the variable. At 8,192 the learner stands at 0.93 / 0.94 / 0.85 of the twin on the first operations, 0.96–0.97 on the later ones, and 0.82 / 0.79 / 0.81 on compositions: the gate of 0.80 on every family is met on the operations everywhere and missed by a point on compositions in one seed. Literal predictions: monotonicity holds; the 8,192 arm is 7–8 points short of the unbounded buffer on compositions, so the curve has not saturated; three of eighteen acquisition comparisons exceed the 3-point tolerance, all in the direction of better acquisition.

The shape is the one a scalable mechanism needs, smooth and monotone in a bounded budget, and the remaining gap is coverage of a 24,000-item family with 7,000 slots. Two experiments follow, each one change: one reservoir with a single total capacity and no family labels, at 24,576 (matched to this run) and one larger value, to show the label is unnecessary and find the bend; then verbatim raw-byte replay at the same budget, which removes the parser. Both run here; the 8,192 per-family arm goes to the cloud session for fresh-seed confirmation (E16-F2). Codex's window-net mirror (E16-WR, with the observed-target arm) runs on this machine from 16:30.

## 9.37 Reference: Behrouz, Hashemi, Javanmard and Mirrokni, "Language Models Need Sleep: Learning to Self-Modify and Consolidate Memories" (arXiv 2606.03979v2, July 2026), read 8 November 2026

**What it proposes.** A wake/sleep lifecycle for large language models in place of train/test. In sleep, (1) *memory consolidation*: the model adds a small set of new low-rank parameters to a slower-updating module and distills into them, from a frozen copy of its own pre-update state, the knowledge held in faster-updating modules (on-policy distillation plus an imitation reward), then resets the fast module's extra parameters; modules form a hierarchy of update frequencies (every 1k / 5k / 10k steps, the Continuum Memory System of the same group's Nested Learning paper); (2) *dreaming*: SEAL-style self-generated synthetic data, selected by gradient importance and rewarded by held-out improvement, for self-improvement. Experiments at 1–8B scale: class-incremental intent classification, long-context QA, BABILong to 10M tokens, AIME-type reasoning (73.8→79.2 on Qwen3-8B), SQuAD knowledge incorporation, ARC few-shot. All mechanisms are backprop, LoRA, distillation and RL; nothing is local.

**Where it bears on this program.** It is the same problem, retention under continual acquisition, framed at the other end of the scale, and three of its ideas are cheap to test here under local rules:
- *Replay confined to sleep phases.* Our learner mixes replay into every batch; theirs consolidates only in offline periods with no new input. A schedule arm on the E16 stream, same total replay exposure concentrated into sleep blocks, tests whether offline replay retains as well as interleaved replay. One change, no new mechanism.
- *Synaptic homeostasis.* Slow-wave sleep's global downscaling is weight decay applied only in sleep. Decay is a local rule and E13/E14 showed its level matters; decay in sleep only, against decay every update, is one more schedule arm.
- *Update-frequency tiers.* Their consolidation hierarchy is, at its simplest, different learning rates per module, slow for the shared path and fast for the readout and cell. That is a learner change and waits behind the replay questions, since E16-D showed the learner is not the bottleneck.

**Where it disagrees with our evidence.** The paper states that forgetting is "a problem of insufficient capacity rather than sampling distribution" and answers it by growing parameters each sleep. At toy scale E16-D is a counterexample: at fixed capacity the twin and the local rule on the twin's data retain everything, and E16-R shows retention tracking replay coverage smoothly. Their claim may hold at their scale; it is not what limits us. Growing parameters each sleep also runs against the bounded-memory requirement on file (they recycle the fast tier but the slow tier grows).

**On dreaming.** Their self-generated replay is RL-selected and expensive, and they note themselves that naive iterative self-improvement causes forgetting and collapse (their appendix A.4 cites several failures of on-policy self-distillation). E16-V found 14–21% wrong self-generated replay targets late in E8's life. Both support Codex's ordering in 9.33: verbatim observed experience first, generative replay later and only if it is shown to be diverse and reliable.

**Evidence level for our purposes.** Benchmark results at LLM scale with ablations but no controls of the kind this program runs (no matched-data comparators, no shortcut audits, no seeds reported); an idea source, not evidence. Nothing here changes the current plan; the two schedule arms are added to the list after E16-R2 and the verbatim-replay step.

## 9.38 E16-R2: a label-free, bounded replay memory meets the Phase A gate on the standing seeds (8 November 2026, executed)

Section E16-R2 of `E45_REPORT.md`. One reservoir filled by Algorithm R over every presentation of the life, no family label anywhere in the replay path, replay drawn uniformly from it; total budgets 24,576 and 65,536; three seeds. At the matched budget the label-free reservoir equals the per-family one of E16-R within 5 points on eight of nine comparisons and is better on the operations in every seed, so the family label was doing nothing; the mechanism audit's expectation of the opposite was wrong and is corrected in the report. At 65,536 entries the learner keeps every family at 0.81–0.99 of the backprop twin after the stream has moved on (operations 0.97 / 0.96 / 0.88, compositions 0.88 / 0.81 / 0.83, later operations 0.99 / 0.98 / 0.95): the Phase A gate of 0.80 on every family, met in every seed, with a replay memory that is bounded, label-free and constant-work per event. Retention remains logarithmic in coverage (14,500 distinct compositions give 75–83%, all 24,000 give 80–85%).

What this does and does not settle. It settles that one cell learner under the local rule, at fixed capacity, can acquire three families of different form from one stream and keep them, given a replay memory of a few hundred kilobytes. It does not yet remove the parser (the reservoir stores parsed entries and re-renders them: E16-R3, verbatim raw bytes at the same budget, is prepared), it is not confirmed on fresh seeds (the cloud session's E16-F2 is redirected to this arm), and the label-free allocation has only been tested on a stream whose families have equal durations. After E16-R3, the schedule questions from the Sleep paper (9.37) and E17, the window's removal, follow.

## 9.35 Codex's E16-WR: bounded stage reservoirs and replay targets on the window net (8 November 2026; Codex's section, filed verbatim from `E16WR_REVIEW_SECTION_9_35.md`)



|Prediction|Passes|Total|
|---|---|---|
|P1|0|3|
|P2|0|3|
|P3|1|3|
|P5|3|3|
|P4|10|12|

At self-target capacity16,384, final held composition is 32.08 / 24.88 / 31.58%; observed-target finals are 62.46 / 57.00 / 65.42%. Larger self-target reservoirs are not a monotonic retention repair in these runs. The observed-target comparison must be read through the matched input/coverage checks and its per-seed loss differences, not as a new architecture result.

|Seed|Self loss|Observed loss|Loss reduction pp|P5≥10pp|
|---|---|---|---|---|
|73129|48.9167|16.8333|32.0833|True|
|93683|55.6250|14.0417|41.5833|True|
|116089|52.7917|19.7083|33.0833|True|

All three largest-capacity pairs match actual prompt-order hashes and all nine stage reservoir and draw snapshots exactly. Observed targets were saved with presented examples. This identifies the effect of replacing self-generated replay targets with observed targets at this capacity and schedule, including downstream changes in Store and earlier acquisition. It does not isolate a late-composition-only effect, establish a global capacity optimum, or prove autonomous learning.

Campaign child CPU: 4.4383 hours, including the technical cycles and full default reproduction; controller CPU 1.332s. Campaign wall 46.66min from controller start, excluding prior waiting. Actual peakRSS is recorded per worker in E16WR_FINAL_COSTS.json. Initial forecasts are retained as forecasts, not rewritten to match measured cost.

No final scientific model/optimizer checkpoints were saved under the locked settings. Exact source, hashes, commands, raw receipts, scores, diagnostics and technical trajectory hashes are retained; do not describe these as a complete resumable checkpoint bundle.

Source: E16WR_REPORT.md, E16WR_VERIFICATION.json and exact JOB-selected receipts. Twelve lives, three seeds in each of four arms; one reused-default full reproduction plus separate technical cycles. Existing E16-R review sections were not edited.

## 9.39 E16-WR assessed: self-generated replay targets run away with coverage; observed targets are required (8 November 2026)

Codex's E16-WR (9.35; `E16WR_REPORT.md`, twelve lives, exact batch and reservoir hashes verified) is the window-net mirror of E16-R, and it changes how every E8-family result should be read.

- **Larger self-target reservoirs make retention worse, not better.** With the learner's own answers as replay targets, held composition after four later stages is 32–36% at 1,024 slots per stage, 42–48% at 4,096, and 25–32% at 16,384. The reason is in the exposure records: the fraction of replayed composition targets that are wrong is 17–21% at 1,024 (a small memorized set), 23–38% at 4,096, and climbs through the later stages to 81–91% at 16,384. Wider coverage replays more prompts the learner answers wrongly, those wrong answers are trained in, and the next stage's answers are worse: the positive feedback loop that the Sleep paper (9.37) warns of for iterative self-training, here measured in a 470,000-parameter net.
- **Observed targets break the loop.** Storing the answer as it was presented, at 16,384 slots, gives held composition of 57–65% after four later stages (a loss of 14–20 points from its post-composition value, against 49–56 for self targets at the same capacity) and keeps the mod-97 operations at 95–96% (self targets: 70–77%). That is the same direction as the cell harness, which has always replayed observed answers, and it is why the cell harness's coverage curve (9.34, 9.38) is monotone while the window net's self-target curve is not.
- **The remaining loss with observed targets** (14–20 points on compositions) is coverage: 16,384 per-stage slots hold about 12,000 of 24,000 taught triples, and E16-R2 showed the cell learner at 14,500 distinct compositions losing 5–12 points. The window net's next step is the one the cell learner has taken: one label-free reservoir with a total budget and observed targets.

Consequences for the record. E8's persistent-life demonstration (9.19) and every E8-family result since (E9, E12, E13, E14, E14-S, E16-W) used self-generated replay targets. For the eight operations, learned to about 95%, the targets were mostly right and the 94% retention stands; for compositions, learned to about 80%, one replayed target in five was wrong from the start, and the composition losses reported in E9 and E16-W are partly this loop rather than forgetting alone. The pseudo-rehearsal design, replaying one's own answers, is retired for this program: the episodic store keeps each experience with its observed outcome, which is what the complementary-systems account always said it does. Codex's E16-WR2 (brief written) reruns E8's demonstration and the late-family schedule with observed targets and a total-budget reservoir.

## 9.41 Reference: Codex's source-audited mechanism catalogue (`PROJECT_DOCUMENTS/MINIFLY_MECHANISM_CATALOG_20261108/`, 8 November 2026; 9.40 is reserved for E16-WR2)

Twenty operational records of fly learning and circuit computation with primary sources, a proposed local algorithm each, a parameter and plastic-state accounting, and a readiness table; a 70-row disposition ledger that preserves every number of the old reconstructed supplement (duplicates and one unsourced label named, nothing silently dropped); one retraction found (Okray et al. 2023, retracted 25 March 2026, so cross-modal binding loses its positive support). The catalogue is explicit that it did not inspect the E16 logs, that no source establishes any module as sufficient for rule learning from bytes under the local-rule constraint, and that it is not a revised experiment schedule. That discipline is its value: it is the first catalogue in this project that separates "shown", "inferred" and "proposed", counts the information budget of every teaching signal, and refuses to inflate the count.

What in it bears on the current line. (i) F11, opposing traces instead of overwriting, and F12, bounded post-event teaching reverberation, are the two records that touch retention; both are marked partial on local sufficiency and neither is a replacement for replay, as the catalogue itself says. (ii) F14 (continuous recurrent working state) and F16/F17 (population-vector transforms and goal comparison) are what E11's cell and E10's ring already instantiate; the catalogue's grounding of them is useful for the write-up, not for a new run. (iii) F09, a learned cue becoming a teacher for another association, is the one record that names a learning operation we have not tested and that is directly constrained by circuit work (second-order conditioning); it is a candidate for the language stage, where a learned word must come to teach another, not for Phase A. (iv) F06's accounting rule, that compartment-specific teaching channels cost bandwidth that must be counted, is the right frame for any future learned-feedback arm (Kolen–Pollack), which E14-C made unnecessary for now.

Use: a reference to cite and a checklist for the mechanism audit of future designs, as the program rule in 9.17 already requires; not a source of the next experiment. The next experiments are fixed by the retention results (E16-R3, E16-WR2, the fresh-seed confirmations, then E17), and the catalogue agrees.

**9.41 addendum: Codex's evidence-audit decision memo** (`research/minifly/evidence_audit_2026-11-08/`, commit 7ff4ab9). The companion to the catalogue: a claim-level register (35 sources, 42 claims, each marked full-text, abstract-only or metadata-only), eight source corrections (among them: Bennett et al. 2021 is a computational reward-prediction model, not a fly experiment; the Westeinde 2025 correction amends a genotype and is not a retraction; the Okray retraction is total and supports nothing), a code-level reading of the harness that correctly describes the feedback as dynamic sign-symmetric, not random, and that the cell already implements gated recurrence, and a ranked memo whose first line is the one that matters: the largest measured defect is replay target correctness and coverage under a structured parser, not the absence of a named circuit, and the decisive comparison is raw observed examples against structured re-rendered examples against model-generated targets at matched budget and realized diversity, with the learner, the token count and target access frozen. That is E16-R3 (raw against re-rendered, same entry budget, observed answers in both) together with E16-WR (self-generated against observed), and the memo's falsifier applies: a raw arm that still touched task identities or the answer generator would not count. E16-R3's raw arm stores presented bytes and presented answers only. One refinement is adopted from it: budgets will be reported in bytes as well as entries (the raw receipt records both), since a parsed entry is smaller than the bytes it re-renders. Its ranks 2–5 (F13 re-anchoring, F09 learned teaching transfer, F11 opposing traces, F08 reward-prediction feedback) are marked "design only, not now", and that is where they stay. No new experiment follows from the bundle; it sharpens the reading of the ones running.

## 9.42 E16-R3: the parser is gone, and the gate still holds (8 November 2026, executed)

Section E16-R3 of `E45_REPORT.md`. The same label-free reservoir of 65,536 slots as 9.38, but each slot holds the presented bytes, prompt and answer exactly as they appeared, replayed verbatim: no task recognition, operand extraction, re-rendering or recomputed answer anywhere in the replay path. Three seeds. Verbatim replay is within 3 points of the parsed arm on every family in every seed, and the gate holds: 0.98 / 0.96 / 0.87 of the twin on the first operations, 0.86 / 0.81 / 0.85 on compositions, 0.98 / 0.98 / 0.92 on the later operations. The reservoir is 1.05 MB, extrapolation to k = 20 survives, and the cell is still what holds (silenced 3–7%). E11-D's memorization of 1,024 fixed episodes per stage does not recur at 65,536 verbatim slots over the life: coverage was the variable, not the form of storage.

Standing after 9.38, 9.39 and this section. One cell learner under the local rule, at fixed capacity, with one megabyte of observed experience sampled with constant work per event, acquires three families of different form from one stream and keeps each at 0.81–0.98 of backprop after the stream has moved on; the replay path contains no label, parser or answer generator. Of the scaffold named this morning, the learner's half is removed. The environment's half remains and is next, each as one change on this harness: a continuously drifting stream in place of blocks, and next-byte loss on every byte in place of the answer mask. Then E17, the window. Fresh-seed confirmation of the raw arm is assigned to the cloud session (E16-F3, after E16-F2); Codex's E16-WR2 brings the window net to the same replay.

**9.41 second addendum: the mechanism horizon scan** (`research/minifly/notes/MECHANISM_HORIZON_SCAN_2026-11-08.md`, commit 14430d6; written with ChatGPT's help, no experiments). Five candidate operations, ranked for discussion, each with a primary source, an engineering form, a cost and a falsifier, and a bottom line that matches the program's: reliable teaching targets and scaffold removal first. Where each stands against the record: (1) self-supervised predictive state over the unsegmented stream is what E16-L tests tonight in its simplest form (next-byte loss on every byte, no answer mask), and the scan's own caution applies, that a next-byte head is not a new architecture; if E16-L holds, the predictive-state organization is the follow-up for Phase B, not Phase A. (2) Composable state and binding is the Phase B question (one-shot word-to-referent binding) and the place where a sparse-key associative memory of the mushroom-body kind would be tested against a matched generic memory (9.42's discussion). (3) Neuron-specific dendritic instructive signals are learned feedback by another name; E14-C measured the sign rule's hidden-layer alignment at 0.81–0.90, so the headroom is small and the item stays conditional, as the scan says. (4) Inferred latent context without phase identity: the learner has never received stage identities, and E16-S tonight removes the stages from the stream itself; if retention survives a drifting stream, there is no context-inference problem to solve at this scale. (5) A trainable temporal basis is what E17's bank of diagonal cells at fixed retention rates is, under another name; the fly interval-clock evidence is a good reference for it. The scan's non-negotiable controls (same tokens and observed outcomes; matched plastic parameters, recurrent state, replay bytes, router and parser resources; provenance of every target) are the ones already in force, and its request for replay budgets in bytes is already adopted. Nothing in it changes the queue.

## 9.32 The cloud session's E16-F: the E16-D finding on fresh seeds (8–9 November 2026; the session's section, filed verbatim from `E16F_REVIEW_SECTION_9_32.md`)

E16-F repeated the decisive E16-D arm, the cell learner under the local sign rule with every presented entry replayed and balanced per family, together with its backprop twin, on three seeds never used by this learner (200003, 300007, 400009), with fresh births and no change to the harness or settings. The births reached 90–99% at k = 12 on the development operations (P16F.1 holds), the twin ended at 94–95 / 92–94 / 93–94 on the mod-97 operations, held compositions and mod-73 operations (P16F.2 holds on family means), and the learner acquired each family on time (mod-97 operations 86–94 after block 2, compositions 89–90 after block 4; P16F.4 holds). The retention prediction P16F.3 holds in two seeds and misses in one: after block 6 the learner keeps compositions at 83.3 / 77.0 / 81.5 and stands at 0.84–0.99 of the twin on every family, but in seed 300007 the compositions fall 12.0 points from block 4, beyond the 10-point margin, with the loss concentrated in the plus-minus pairs and in sub97. The standing seeds fell 5.7–8.1 points on the same measure, so the finding transfers in kind and range, and composition retention is its weakest and most seed-dependent part. The first launch's learner lives were stopped in block 6 by the launching tool's two-hour limit. Reruns from scratch with unchanged settings reproduced the first five blocks exactly, every score and loss, so the verdicts are those the first launch would have given. Development result on a toy harness, three fresh seeds; sources `E16F_REPORT.md`, `E16F_PREDICTIONS.md`, `exp45/SUMMARY_E16F.md` and the six receipts.

**9.32 assessed.** The finding transfers. On three seeds the cell learner had never seen, with new births, every-entry replay keeps compositions at 83 / 77 / 82% after the stream has moved on (standing seeds 85 / 80 / 83) and the learner stands at 0.84–0.99 of the twin on every family (standing 0.82–0.99); the births, the twin and acquisition all meet their predictions. The one miss is the retention clause in seed 300007, 12.0 points against a limit of 10, concentrated in the carry pairs and in subtraction, which is where the standing seeds lose first too. The session's process was sound where it mattered: predictions locked before the births, a deviation recorded before any run when the smoke job's birth turned out not to be in the repository, the learner lives killed by the tool's two-hour limit rerun from scratch with unchanged settings and shown to reproduce the killed lives' first five blocks exactly, and a log-file mishap during a rebase reported rather than hidden. Its machine is about 3.8× slower per update than this one, which is why it carries confirmations rather than the critical path. E16-F2 (the label-free 65,536 arm) is at five of six blocks there; E16-F3 (the raw-byte arm) follows. The standing births' weights are now in the repository as well, so the smoke job the brief asked for can be run as written next time.

## 9.36 The cloud session's E16-F2: the label-free reservoir of 65,536 on fresh seeds (9 November 2026; the session's section, filed verbatim from `E16F2_REVIEW_SECTION_9_36.md`)

E16-F2 ran E16-R2's gate arm, one Algorithm-R reservoir of 65,536 entries over every presentation of the life, with the replay half drawn uniformly and no family label anywhere in the replay path, on the three fresh seeds of E16-F (200003, 300007, 400009), using E16-F's births and twin lives and no change to the harness. All three predictions hold in every seed. After block 6 the learner stands at 0.94–0.97 of the twin on the mod-97 operations, 0.82–0.87 on the held compositions and 0.96–0.99 on the mod-73 operations (P16F2.1). Compositions lose 8.8 / 8.0 / 11.8 points from block 4 to block 6, within the 15-point margin, and the reservoir holds 28,298–28,431 distinct entries at block 5, inside the predicted 27,500–29,500 (P16F2.2). Acquisition is on time: 88–93 on the mod-97 operations after block 2 and 88–89 on compositions after block 4 (P16F2.3). Against the every-entry, family-balanced arm on the same births, the reservoir keeps the mod-97 operations better in every seed, by 1.6–4.8 points, leaves compositions level within a few points, and recovers the seed where the every-entry arm missed its retention margin. The Phase A gate for the label-free reservoir therefore holds on six seeds, three standing and three fresh. Composition retention, at 77–81% against the twin's 91–94%, remains the family with the least headroom. Development result on a toy harness; the budget is counted in entries, not bytes. Sources: `E16F2_REPORT.md`, `E16F2_PREDICTIONS.md`, `exp45/SUMMARY_E16F2.md` and the three receipts.

**9.36 assessed.** The gate result of 9.38 transfers to three seeds the learner had never seen: with one label-free reservoir of 65,536 entries the learner stands at 0.97 / 0.95 / 0.94 of the twin on the mod-97 operations, 0.86 / 0.87 / 0.82 on the held compositions and 0.99 / 0.99 / 0.96 on the mod-73 operations (standing seeds 0.97 / 0.96 / 0.88, 0.88 / 0.81 / 0.83, 0.99 / 0.98 / 0.95), with composition losses of 8.8 / 8.0 / 11.8 points from their peak and the reservoir's distinct count at 28,300–28,400 as predicted. All three predictions hold in every seed. Together with 9.32, the Phase A gate as stated in E16 is now met on six seeds with a bounded, label-free replay memory; E16-F3 (the raw-byte arm, parser removed) and E16-F4 (three more seeds) are locked and running there.

## 9.47 Reference: the on-demand temporal read head proposal (`MOD97_LOCAL_RULE_20261106/DESIGN_TEMPORAL_READ_HEAD_20261108.md`, commit 150e27b, design only; read 8 November 2026)

**What it proposes.** An append-only log of raw observed bytes with stable addresses, a sparse index from a key to candidate addresses, a read controller on the current state that decides whether to read and what to ask, a bounded fetch of old bytes re-encoded by the shared encoder and merged into a temporary workspace, and a separate consolidation path. Reading is an action with a cost. The experiment ladder is sound: an oracle-address ceiling before any learned controller, random-address and scrambled-content arms, long-gap factual recall with distractors, then mutation and temporal precedence, then two-hop re-use, then a true streaming life; pre-registered gates with fresh seeds; a locality warning that a backprop-trained controller is a feasibility ceiling, not a local mechanism.

**Prior art.** The document names Neural Turing Machines, Differentiable Neural Computers and retrieval-augmented language models. The lineage is longer: memory networks (Weston et al. 2015; Sukhbaatar et al. 2015), neural episodic control (Pritzel et al. 2017), nearest-neighbour language models (Khandelwal et al. 2020), retrieval-enhanced transformers (Borgeaud et al. 2022), memorizing transformers (Wu et al. 2022), test-time memory modules (Titans, Behrouz et al. 2024) and, on the biological side, hippocampal indexing theory (Teyler and DiScenna 1986), which is this exact proposal: a sparse index over cortical traces, addressed by content, read on demand. The idea has been worked many times; nothing in it is new as a mechanism, and the document says so. What would be new is the controlled version: local training of the controller, byte-level audit of every read, matched budgets, and integration with a learner that consolidates by replay.

**Where it fits.** It answers a different property from the one Phase A measured. Weights plus replay keep rules; a read head recovers specific past episodes, one-shot and at long range, which is the Phase B need (a word bound to a referent after one exposure; a fact stated once and asked later) and the place where a sparse-key associative memory of the mushroom-body kind was always going to be tested (9.42's discussion). The exact-key store that E16-A ablates tonight is the degenerate form of this head (a ten-byte key, a next-byte read, no controller); its result says what the degenerate form was worth. The proposal's Phases 1–3 are a good draft of the episodic part of the Phase B battery and should be folded into it rather than run as a separate line.

**Conditions before it runs.** (i) Phase B's stream and battery exist first, so the head is tested on the stream it is meant for. (ii) The index is the fly-style sparse expansion with a local write, compared against a dense embedding index at matched budget, so the structural question is answered at the same time. (iii) The controller is trained by a local rule from the first science run (a broadcast scalar for the read decision, eligibility on read actions), with the backprop-trained controller reported only as the ceiling. (iv) Log, index and reservoir are three budgets, reported separately in bytes. (v) The uniform-sample reservoir is not an address space; if the log and the reservoir share bytes, eviction must not break addresses, as the document notes. Not for now; filed for Phase B.

## 9.48 Reference: Behrouz, Razaviyayn, Zhong and Mirrokni, "Nested Learning: The Illusion of Deep Learning Architecture" (NeurIPS 2025; arXiv 2512.24695v1, 31 December 2025), read 8 November 2026

**What it is.** The parent of the Sleep paper (9.37). Its claim is a reframing: every component of a trained network, layers, attention, momentum, the optimizer itself, is an associative memory that compresses its own "context flow" by gradient descent, and the components form levels ordered by update frequency (attention updates every token, pre-trained weights never). From the frame it derives (i) Delta Gradient Descent, a weight update whose decay acts only along the current input direction, `W ← W(I − η' x xᵀ) − η δ ⊗ x`, and a delta-rule momentum; (ii) the Continuum Memory System, a chain of feedforward blocks each updated at its own frequency (every step, every thousand, every ten thousand) so that knowledge forgotten at one rate survives at a slower one; (iii) Hope, a self-modifying memory that generates its own keys, values and learning rates, in front of such a chain; (iv) a multi-scale momentum optimizer. Experiments at 760M–1.3B parameters: language-model perplexity, needle-in-a-haystack, BABILong to 10M tokens, class-incremental text classification, continual translation of two unseen languages. The conclusion states that forgetting is not solved and is "a natural consequence of compression" at limited capacity.

**What transfers, and what does not.**
- *The frame.* "Levels are update frequencies" is a useful vocabulary for what this program already has and plans: the reservoir and the weights are two levels (replay every batch, weights every batch at decay 0.25), the E17 bank of diagonal cells at fixed retention rates is a set of levels in the state rather than the weights, and the Sleep paper's tiers (9.37) are the same idea. It costs nothing to adopt the vocabulary.
- *Delta Gradient Descent is a local rule.* Written per synapse, `Δw_ij = −η' x_j (Σ_k w_ik x_k) − η δ_i x_j`: the decay term is the presynaptic activity times the unit's own pre-activation, both available at the synapse (it is the delta rule's forgetting term, Oja-like). Decoupled weight decay was the knob that mattered most this week (E13, E14), and an input-directed decay that forgets only along directions being written is a one-change arm on the E16 stream, inside P5, with a clear prediction either way. This is the one concrete thing in the paper to test here, after the current queue.
- *Frequency tiers for the weights* (the shared path updated every C batches with its accumulated local update, the readout and cell every batch) are also local and one change; lower priority, since E16-D showed the learner is not the bottleneck at this scale.
- *What does not transfer.* Hope and the self-modifying memory are backprop-through-time constructions at LLM scale; the optimizer work (momentum as memory, Muon variants) concerns the development phase, not life; and the capacity account of forgetting is contradicted at toy scale by E16-D (fixed capacity, full-coverage replay, flat retention), as noted for the Sleep paper.

**Evidence level for our purposes.** Benchmark results at scale with ablations, no matched-data comparators or shortcut audits of our kind; an idea source. Nothing changes the queue; the delta-decay arm is added to the list after E16-S, E16-L and E16-A, before E17.

## 9.44 E16-S and E16-L: the stream without stages holds; the answer mask was doing real work (8 November 2026, executed)

Section E16-S and E16-L of `E45_REPORT.md`. Both on the raw-byte reservoir of 65,536 (9.42's arm), three seeds. **No stages.** On a stream whose family mixture drifts smoothly and never announces a boundary, with the reservoir admitting items as they arrive, the learner keeps every family at 0.84–0.99 of a twin trained on the same stream, within 5 points of the staged control on every family; the gate holds in every seed. The block structure of E16 was a convenience for measurement, not a crutch for the learner, and the one environment-side scaffold that could have been hiding a stage signal is gone. **Loss on every byte.** With the answer mask removed, the sign rule does not acquire addition or subtraction at all (3–5% at k = 12 after their blocks; the later operations 32–35%), while the compositions are acquired and then lost. The prompt bytes are near-uniform and unpredictable, and their loss terms swamp the three answer bytes' signal. A twin under the same loss is queued to say whether backprop fails the same way. Either way this is the first environment-side scaffold whose removal costs something, and it names a Phase B design question in advance: a language stream will not point at the answer, so the learner needs either a loss that weights the predictable or a way to find it. One harness slip (the drift twin's receipt overwrote the staged twin's in the working folder) was caught, the originals restored from the repository and the naming fixed.

## 9.40 Codex's E16-WR2: observed targets and a total-budget reservoir on the window net; E8's demonstration rerun with observed targets (8 November 2026; Codex's section, filed verbatim)

Increasing the total reservoir to 65,536 slots raised final held-composition accuracy by 10.17–12.96 percentage points across the three seeds. It still lost 15.67 / 19.00 / 18.46 points after acquisition, in seed order. Final mod-97 accuracy was 95.33% / 95.62% / 96.67%. P1 met both thresholds in 0 of 3 seeds.

At 16,384 total slots, 0 of 3 seeds met both endpoint bands against the prior per-stage reservoir. Across the two Part A capacities, 3 of 6 lives met the acquisition band against E16-W uniform replay. The larger reservoir finished with 21,351–21,516 distinct prompts. In both capacity arms the stage-end design held up to 393,216 pending records in addition to the retained reservoir.

Part B reproduced the E8 demonstration with observed answers. It changed final held-composition accuracy by +0.46 / -0.04 / +3.54 points relative to E13-A1. The eight-operation means met P4 in 3 of 3 seeds. Composition is taught last in Part B, so the endpoint measures acquisition and retention of prior operations. All three full Part B input streams matched prompt hashes independently reconstructed from the original schedule and sampler. The archived E13-A1 runs supply scores, and their original prompt streams were not recorded.

| Job | Post-compose held % | Final held % | Loss pp | Final held full % | Final mod-97 % | Final eight operations % | Weakest carry | Carry final % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A16384_73129 | 85.2500 | 56.0000 | 29.2500 | 50.2083 | 94.2500 | 93.4375 | +- | 21.1667 |
| A16384_93683 | 83.4167 | 50.0833 | 33.3333 | 43.2083 | 95.6250 | 94.6042 | +- | 18.6667 |
| A16384_116089 | 84.5000 | 54.7083 | 29.7917 | 48.5000 | 95.4167 | 95.1250 | +- | 21.1667 |
| A65536_73129 | 81.8333 | 66.1667 | 15.6667 | 59.8333 | 95.3333 | 94.5000 | +- | 37.5000 |
| A65536_93683 | 81.6250 | 62.6250 | 19.0000 | 58.2083 | 95.6250 | 94.6042 | +- | 33.1667 |
| A65536_116089 | 86.1250 | 67.6667 | 18.4583 | 63.2083 | 96.6667 | 95.1458 | +- | 34.3333 |
| B_73129 | 82.7500 | 82.7500 | — | 80.6667 | 95.0417 | 94.2708 | +- | 66.0000 |
| B_93683 | 80.9583 | 80.9583 | — | 78.9583 | 95.8333 | 94.2292 | +- | 69.0000 |
| B_116089 | 88.5417 | 88.5417 | — | 86.6250 | 95.8750 | 94.7292 | +- | 80.6667 |

| Prediction | Passing | Evaluated | Expected |
| --- | --- | --- | --- |
| P1 | 0 | 3 | 3 |
| P2 | 0 | 3 | 3 |
| P3 | 3 | 6 | 6 |
| P4 | 3 | 3 | 3 |
| P5 | 2 | 3 | 3 |

The campaign used 3.4059 child CPU-hours and 28.47 minutes of wall time. The total and per-stage arms differ in retained capacity as well as sampling. The observed targets agree byte for byte with the saved observations.

See [the full report](E16WR2_REPORT.md) and [verification receipt](E16WR2_VERIFICATION.json).

## 9.43 The cloud session's E16-F3: the raw-byte reservoir on fresh seeds (9 November 2026; the session's section, filed verbatim)

E16-F3 ran E16-R3's verbatim arm on the three fresh seeds of E16-F (200003, 300007, 400009), using E16-F's births and twin lives and no change to the harness. The arm is one label-free Algorithm-R reservoir of 65,536 slots holding the presented bytes exactly as they appeared, replayed with no parser, re-rendering or recomputed answer. Two of the three predictions hold in every seed.

- **P16F3.1:** after block 6 the learner stands at 0.91–0.98 of the twin on the mod-97 operations, 0.87–0.88 on the held compositions and 0.96–0.98 on the mod-73 operations.
- **P16F3.3:** at block 5 the reservoir holds 1.048–1.050 MB and 57,715–58,028 distinct byte strings.
- **P16F3.2**, within 5 points of the parsed reservoir of E16-F2 on every family, holds in two seeds. It misses in 400009, where the raw arm keeps 5.5 points *more* of the compositions: 82.5 against 77.0. There the parsed arm lost 11.8 points from block 4 and the raw arm 6.7. The other eight comparisons are within 3 points.

The label-free gate therefore holds without a parser on six seeds, three standing and three fresh. On these seeds, parsing the replayed examples bought nothing that the stored bytes do not give. Composition retention, at 79.5–82.5% against the twin's 91.5–94.0%, is still the family with the least headroom. The cost forecast missed: verbatim replay was not cheaper than the parsed arm on this machine (5,330 against 5,080 CPU-s per life), unlike on the standing machine. Development result on a toy harness; one stream, three seeds. Sources: `E16F3_REPORT.md`, `E16F3_PREDICTIONS.md`, `exp45/SUMMARY_E16F3.md` and the three receipts.

**9.40 and 9.43 assessed together.** Two confirmations and one new gap. *E16-F3:* the parser-free raw-byte reservoir meets the gate on the three fresh seeds (0.91–0.98 of the twin on the first operations, 0.87–0.88 on compositions, 0.96–0.98 on the later operations), within 3 points of the parsed arm on eight of nine comparisons and 5.5 points better on the ninth; the reservoir is 1.05 MB with 57,700–58,000 distinct byte strings, as predicted. With 9.36 and 9.42 the headline mechanism now stands on six seeds in its final, parser-free form, and E16-F4 is adding three more. *E16-WR2, Part B:* E8's eight-operation demonstration rerun with observed replay targets reproduces E13-A1 within 0–3.5 points on composition and within 2 points on every operation, so the program's headline persistence result stands without pseudo-rehearsal. *E16-WR2, Part A, the gap:* on the window net, one label-free reservoir of 65,536 observed-answer slots loses 16–19 points of held composition over the four later stages (per-stage 16,384: 14–20; self-generated targets: 49–56), with the operations at 95–97%; the cell learner on the same replay design loses 5–12 (9.42). The two learners differ in retention under the same kind of replay, and the candidates are the architecture (the cell, window 10 against 18) and the schedule (E8's blocked single-operation stages and 8,192-update composition stage against E16's interleaved families): a matched comparison, one learner on the other's schedule, is the experiment that separates them and is queued behind the loss work. Codex's prediction P1 (loss at most 8) missed in every seed and is recorded as such.

## 9.49 Reference: Codex's roadmap after the reviews (`MOD97_LOCAL_RULE_20261106/ROADMAP_POST_FABLE_REVIEW_20261108.md`, commit ccf5eb5, read 8 November 2026)

Provenance first: the document's "executive decision" was drafted by ChatGPT and relayed through Codex, and it decides nothing; decisions in this program are Mr. Zhou's, and the document is read here as one more synthesis of the record, to be checked against the record. It survives that check because it restates the sequence already in force rather than proposing a new one. A sequencing document: freeze and audit one E16 reference; a genuine streaming learner (a split-call-stable state API, E17 against the window, and the every-byte-loss bottleneck resolved before the learner is called a language predictor); the Phase B episodic battery with sparse and dense locally trained read heads as competing retrievers; consolidation and transfer under a fixed total budget; then a small natural-text pilot against matched recurrent and Transformer baselines. Its guardrails (one primary question per change; predictions and falsifiers first; failed literal thresholds retained; update rule, capacity, replay quality, memory size, objective and exposure never confounded in one inference; teaching bandwidth counted; every memory budgeted separately; a capability ledger of demonstrated / development-only / failed / untested) are the ones this review has been applying, written down in one place. Its immediate priority order coincides with the one in force here, which is the reason it is harmless, not a source of authority. One addition is adopted from it: the capability ledger will be kept as a single table from Phase B on. One scheduling rule is added here after tonight's overlap (E16-A and E16-WR2 started within a minute of each other on the same machine): a producer takes the machine by creating a lock file and releases it by removing it, and no producer launches while another's lock exists.

## 9.51 E16-A: the architecture audit; the last trace of the mushroom-body store does nothing (8 November 2026, executed)

Section E16-A of `E45_REPORT.md`. ChatGPT's critique this evening (9.42's discussion) asked for the ablation that E16-V had already flagged as missing: E16-R3's arm with the exact-key store removed from the learner, everything else identical. The answer is exact: at every checkpoint, on every family, in every seed, the two arms differ by nothing on the operations and by at most one item in 600 on the compositions. The store never entered training and could not score an unseen prompt, so it was inert from E11 onward. The original PN–KC–MBON architecture has left the main model entirely rather than surviving as a module, and the record now says so plainly. The store is retired for new harnesses (E17 and Phase B); the E16 harness keeps the knob so every receipt stays reproducible by name. Where a fast associative memory of the mushroom-body kind may still earn a place is as the sparse-key retriever in the Phase B episodic battery, against a matched generic index, which is the test that was never run because the arithmetic line never posed the task. Standing after today: the learner is fly-constrained (local plasticity, broadcast error, bounded memory) and not fly-structured; the ring in the embodied line remains the one fly circuit that computes.

## 9.50 Codex's E16-WL: the every-byte loss on the window net under both rules (8 November 2026; Codex's section, filed verbatim)

The window net retained operation learning under the every-byte loss. The sign-rule arm reached 92.48–92.98% at stage eight, versus 94.40–95.52% under answer-only loss. Addition and subtraction mod 97 each reached at least 88.5%. P1 missed in all three seeds. Backprop reached 95.98–96.62%, meeting P2 in every seed.

Composition was more sensitive in the sign-rule setup. Held accuracy fell 10.04–24.08 points against its answer-only reference, reaching the predicted 15-point drop in one of three seeds. All three backprop seeds stayed within the predicted loss bound. The two setups use different sampling and Store policies, so their difference cannot isolate the learning rule. The stage losses measure prediction error on prompt and answer positions. They do not measure the relative size or direction of the updates those positions induce.

All six runs passed receipt checks.

Each loss comparison uses its arm’s answer-loss reference. C uses observed replay and a Store. D samples the joint pool of observed examples and has no Store.

| Arm | Seed | Stage 8 mean % | Add97 % | Sub97 % | Final held % | Reference held % | Held loss pp | P1/P2 | P3 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C | 73129 | 92.48 | 88.50 | 89.00 | 72.71 | 82.75 | +10.04 | False | False |
| C | 93683 | 92.98 | 90.17 | 90.00 | 69.33 | 80.96 | +11.62 | False | False |
| C | 116089 | 92.71 | 90.50 | 90.83 | 64.46 | 88.54 | +24.08 | False | True |
| D | 73129 | 96.38 | 94.50 | 92.50 | 89.75 | 89.29 | -0.46 | True | True |
| D | 93683 | 95.98 | 95.83 | 94.83 | 89.00 | 91.25 | +2.25 | True | True |
| D | 116089 | 96.62 | 95.50 | 97.67 | 91.08 | 90.75 | -0.33 | True | True |

The six worker timers sum to 1.720 CPU-hours. Qualification and training took 20.61 minutes after lock acquisition.

## 9.52 ChatGPT's static audit of 8 November, item by item (`PROJECT_DOCUMENTS/MINIFLY_PROJECT_CRITIQUE_AND_STATIC_SOURCE_AUDIT_20261108.md`, commit f0fc6ce), and what was changed tonight

The audit ran no code and read source and receipts. Its findings about this harness are correct in substance, and three of them are fixed as of commit noted below; the rest are recorded. In order:

1. **Held-pair operand collision: correct, fixed, effect bounded.** The held-panel seed `seed + 1` for the +- pair gave `31·seed + 377`, the taught -> pair's seed, so all 600 operand triples of the +- panel were among the -> pair's 2,000 taught triples (verified tonight: 600 of 600). The pairing and the full prompt were still held out, and the shared triples carry a different intermediate and a different final answer under the taught pair, so memorizing them would have hurt +- rather than helped; the receipts agree, with +- never systematically above -+ (E16-R3: 64 / 56 / 69 against 75 / 60 / 68). The other three held pairs did not collide. Fix: held and within panels are now keyed by split name and pair and drawn excluding every taught triple of every pair (operand-level holdout, asserted in qualification); taught streams are unchanged, so every training run is reproducible. New panels apply from E17 on; comparisons of E17 to E16-R3 on compositions are therefore on fresh panels of the same size, and the record says so. The same formula is in E7 and E8 (Codex's harness); Codex is asked to fix it the same way. Old receipts stand with this note; their weights were not saved, so they cannot be re-scored, which is why every life now saves its final weights.
2. **Qualification's "birth" composition score was taken after smoke updates: correct, fixed.** The number in `QUALIFY_E16.json` was post-update; the scientific stage-zero score was always taken before any update and is the one used everywhere. Qualification now scores before the smoke updates and records the post-update number under its own name.
3. **V219's generator and dose labels:** not this harness; recorded for the owners of that line.
4. **The E17 launcher could admit stale births and overwrote the host lock: correct, fixed.** The pipeline now creates the lock only if absent or its own, removes it only if its own, logs every worker's exit code, and admits a birth only if its receipt is complete (24,576 updates), its checkpoint exists, its hash is recorded (`exp45/E17_BIRTHS.json`) and the gate holds; each life records the hash of the birth it loaded. The lock path was already moved to one absolute host-wide file.
5. **E16-WL artifacts can mix modes:** Codex's harness; passed to Codex.
6. **Unsupported configurations and the comparator join: correct, fixed.** The drifting stream with the old uniform replay silently fell back to blocked inputs; that combination is now refused (no published arm used it). The summarizer maps the every-byte-loss arm to the every-byte-loss twin and the drift arm to the drift twin, and reports full-scratchpad exact accuracy beside the final-answer figure for compositions (raw-replay lives: 72–80% full scratchpad against 80–82% final answer).
7–8. **Other lines' restart, scoring and configuration risks:** recorded; not this harness. The store's FIFO eviction note is moot since the store is retired.

The audit's priorities match the queue: freeze the E16 reference (done by receipt and hash from tonight), compare the two objectives under both rules on identical bytes (E16-L, its twin, E16-WL; the one-change loss-weighting arm next), and for composition intervene on the generated intermediate (added to the Phase B battery notes). Its demand that any fly-derived component claimed must be ablated is already met for the store (9.51).

**9.44 addendum: the twin under the every-byte loss (23:15 EST).** Backprop through time on the same stream is unaffected by the objective (operations 92–94% at k = 12 after their blocks and 94–95 at the end; compositions 88–92%; the later operations 90–93%), so the objective is not the limit under exact credit. With Codex's E16-WL (the window net under the sign rule loses little), the failure of E16-L is confined to one cell of the design: the local rule, on the recurrent learner, under the every-byte loss, where the learner stands at a third of its twin on the operations. The suspect is the trace: under the answer mask the state's synapses receive learning signals only at the three answer positions; under the every-byte loss they receive them at every position, most from unpredictable prompt bytes, and the eligibility carries that noise into the synapses that hold operands. The one-change arm is a position-weighted learning signal into the state's synapses (prompt positions at weight 0 and 0.1), which separates the trace's exposure from the shared path's updates. This is the first result in the program where the local rule, not the data, is the limit, and it appears exactly where Phase B lives: a stream that does not point at what to predict.

## 9.55 E17, Phase 0: without the window, one diagonal layer does not learn the operations even under backprop (8 November 2026, executed)

Section E17 Phase 0 of `E45_REPORT.md`. Six births by backprop through time on the two windowless learners. After 24,576 updates addition mod 89 stands at 22–31% and subtraction at 9–17% at k = 12, and no better at k = 0; the comparisons reach 53–73%; the windowed births reached 91–99% at the same dose. The curves are still rising but slowly. P17.0 is refuted for both architectures under exact credit, and the staging worked as designed: no local-rule life was launched on an architecture its own twin cannot learn. The reading is representational: in both learners the write into the state depends on the current byte alone, so the state is a decayed superposition of per-byte features and the two digits of an operand can meet only in the readout's one hidden layer; the window had handed the readout explicit positions. Three one-change variants run overnight as births only: a deeper readout, two stacked banks so that writes can depend on accumulated context, and double dose. If none passes, a byte-at-a-time learner needs state-dependent writes, a non-diagonal recurrence for which no exact local trace exists and the e-prop approximation is the tool; that is a design decision, not a run, and it will be put to Mr. Zhou with the numbers.

## 9.45 The cloud session's E16-F4: the raw-byte gate on three more fresh seeds (9 November 2026; the session's section, filed verbatim)

E16-F4 made three new births (500011, 600013, 700027) and ran the raw-byte learner (one label-free Algorithm-R reservoir of 65,536 verbatim slots) and its backprop twin on each, with no change to the harness. Only seed 700027 meets every prediction.

- **The births and acquisition hold everywhere.** The births reach 89–98% on every development operation (P16F4.1). The learner reaches 85–92 on the mod-97 operations after block 2 and 81–88 on compositions after block 4 (P16F4.5).
- **The twin misses in 500011 (P16F4.2).** Its held compositions reach only 84.0 after block 6, against a bar of 85, and were still rising. The -+ pair is at 62.5.
- **The gate misses in 600013 (P16F4.3).** The learner stands at 0.81 / 0.69 / 0.90 of the twin. This is the first family below 0.80 in nine seeds of the label-free reservoir.
- **Retention misses in 500011 and 600013 (P16F4.4).** Compositions end at 73.4 and 64.7, after losses of 13.3 and 16.5 points from block 4.
- **The failure in 600013 is in addition and subtraction.** The learner's mod-97 family fell from 85.2 to 52.1 when the compositions arrived, the deepest dip yet. It recovered only to 76.7, with add97 and sub97 near 60. The plus-minus compositions fell to about 50, while the twin held 94.
- **The reservoir is not the difference.** It looks the same as in every other seed: 1.05 MB and about 57,900 distinct strings.

The gate therefore holds in eight of nine seeds. Composition retention, already the weakest margin, fails in two of these three seeds. One seed shows that the sign rule's interference between the composition blocks and the earlier plus-minus operations can be large enough to break the 0.80 gate. Development result on a toy harness; one stream, three seeds. Sources: `E16F4_REPORT.md`, `E16F4_PREDICTIONS.md`, `exp45/SUMMARY_E16F4.md`, the births and the six receipts.

**9.45 assessed.** The widening did its job: on seeds 500011, 600013 and 700027 the raw-byte learner stands at 0.94 / 0.81 / 0.94 of the twin on the first operations, 0.87 / 0.69 / 0.83 on compositions and 0.99 / 0.90 / 0.95 on the later operations. The gate (0.80 on every family) holds in 500011 and 700027 and fails in 600013 on compositions (64.7 against the twin's 93.2); the retention clause (compositions within 12 points of their peak) holds only in 700027; acquisition holds everywhere; the twin itself misses its composition bar by a point in 500011. Across all nine seeds the gate now holds on eight (three standing, 200003, 300007, 400009, 500011, 700027) and fails on one, and composition retention is the weakest and most seed-dependent property of the result, as every earlier section said. The headline is corrected accordingly: a bounded, label-free, parser-free replay memory keeps the learner at or above 0.80 of backprop on every family in eight seeds of nine, with one seed at 0.69 on compositions. The failing seed's compositions were acquired normally and lost 24 points over the later blocks, so the mechanism that is weakest is the one E16-R2's capacity curve already named: composition coverage at this budget.

## 9.56 E17, Phase 0b: context-dependent writes are the missing operation (9 November 2026, executed)

Section E17 Phase 0b of `E45_REPORT.md`. Nine births under backprop. A deeper readout on the windowless cell recovers 30–45 points on addition and subtraction and stays short of the gate; double dose on the single bank recovers nothing; two stacked fixed-rate banks, the second written from the first's state, reach the windowed births' level in every seed (addition 93–95%, subtraction 92%, comparisons 95–99% at k = 12). All three predictions hold. So a byte-at-a-time learner needs writes that depend on accumulated context, and one extra diagonal layer supplies them while each bank's own trace stays exact given its input; the known cost is that credit to the first bank through the second bank's memory is not carried (the e-prop truncation). Phase 1, the local-rule life on bank2 against its twin, launched at 09:00, with the truncated trace named in advance as the suspect if the life falls short where the birth did not.

## 9.46 The cloud session's E16-RC: the raw-byte reservoir's capacity curve on the standing seeds (9 November 2026; the session's section, filed verbatim; branch claude/vibrant-gauss-tops5f merged as c8f55e1)


**9.46 assessed.** The curve is flat. From 16,384 slots (0.26 MB) to 131,072 (2.1 MB) retention after the last block moves by less than the seeds differ from one another at a single capacity; the only point below the gate is one seed at the lowest capacity (93683 at 16,384, 0.76 on compositions), and 65,536 remains the smallest budget that met the gate in every seed. The two misses are informative: there is no monotone rise to measure, and the gate's lower edge is at or below 16,384 in two seeds. Set beside E16-R, where the parsed, re-rendered per-family reservoir rose strongly from 2,048 to 8,192 slots per family, the raw-byte reservoir reaches its plateau at a far smaller budget, which is the result that matters for scaling: what is stored verbatim is used more efficiently than what is parsed and regenerated, or the parsed curve's dependence was partly an artifact of per-family allocation. At 131,072 the reservoir holds about 20,000 of the 24,000 taught compositions and still sits 3 points under the every-entry arm in two seeds, so the last few points are not coverage alone; the every-entry arm's balanced per-family draw is the remaining difference. The session could not push to the consolidated branch and said so; its branch was merged by hand, and the standalone brief for the remaining cloud jobs carries that rule.

## 9.53 The cloud session's E16-F5: the stream without stages on the fresh seeds (9 November 2026; the session's section, filed verbatim; pushed to the consolidated branch at 3d3554ca)


E16-F5 ran E16-S1's arm on the three fresh seeds of E16-F (200003, 300007, 400009), on E16-F's births, with the re-keyed panels and no change to the harness. The arm is the raw-byte reservoir of 65,536 slots admitting each item as it arrives, on a stream whose family mixture drifts smoothly with no stage boundaries. Each seed has a twin trained on the same drifting stream.

- **The twin holds 90 / 85 / 90 in every seed (P16F5.2).** It ends at 93–96 on every family.
- **Seeds 200003 and 300007 hold P16F5.1 and P16F5.3.** The learner keeps 0.93 / 0.86 / 0.96 and 0.96 / 0.87 / 0.99 of the twin. Its compositions, 81.5 and 79.3, are within 0.3 points of the same births under block stages in E16-F3.
- **Seed 400009 misses both.** Its compositions end at 64.9: 0.69 of the twin and 17.5 points below its E16-F3 result of 82.5. The learner's mod-97 operations dipped to 62.0 as the compositions entered. Its compositions peaked at only 80.9 and then lost 16 points as the mod-73 operations took over, with the plus-minus pairs ending near 46 and the max-min pairs near 83. The twin stayed at 94.
- **The reservoir is not the difference.** It looks the same as in the other two seeds: 1.05 MB and about 55,300 distinct strings.

This is the interference between plus-minus compositions and mod-97 addition and subtraction that broke E16-F4's seed 600013. Here it appears on a birth that passed under stages, so the drifting stream can turn a passing seed into a miss. The drift gate therefore holds on five of six seeds, three standing and three fresh. Together with E16-F4, composition retention under the sign rule is now the place where the label-free result fails, in two of the fifteen raw-byte lives run so far (nine on block stages, six on the drifting stream). Development result on a toy harness; one stream per seed. Sources: `E16F5_REPORT.md`, `E16F5_PREDICTIONS.md`, `exp45/SUMMARY_E16F5.md`, the six receipts and the final weights.

**9.53 assessed.** The drift gate holds on five of six seeds, and the one miss has the signature of E16-F4's seed 600013: the plus-minus compositions interfere with the mod-97 addition and subtraction under the sign rule, the twin is untouched, and the reservoir is indistinguishable from the passing seeds'. Two of fifteen raw-byte lives now fail, both this way, one under stages and one under drift, and a seed that passed under stages failed under drift. Whether the cause is the rule or the data pipeline is separable on the same births: arm M keeps the learner's pipeline and changes the rule to backprop, and oracle replay keeps the rule and changes the data to the twin's stream. Whether the early dip of the operations predicts the late loss of the pairs is testable across seeds. Both go to the cloud workers as E16-M2 and E16-F6 (`OPUS_TASK_E16M2_E16F6_CLOUD_JOBS_20261109.md`). A harness defect found while writing that brief: arm M under online admission filled the reservoir at block ends only, because admission was gated on the local arm; fixed and dry-run before the brief was pushed; no finished result used that path.

## 9.54 The cloud session's E16-RP: the held-panel collision measured on the same weights, and sign-rule receipts across machines (9 November 2026; the session's section, filed verbatim; branch claude/vibrant-gauss-tops5f merged as c87a7434)


`E16RP_REPORT.md`, `exp45/SUMMARY_E16RP.md`, `exp45/E16RP_SAME_WEIGHTS.json`.

**What was run.** E16-R3's arm (raw reservoir, 65,536 slots) was rerun on the standing seeds at harness 432e90fb, with the re-keyed, operand-disjoint panels.

**The parity test failed, and the cause is the machine.** Each life was identical to E16-R3 at the birth, which has the same hash. From block 1 every operation panel and the training loss differed. On this container, E16-R3's own harness (7703d6c) and the current one are bit-identical over six 64-update blocks. So the divergence comes from the machine or PyTorch build, not the code: the sign rule amplifies rounding from the first update. P16RP.1 misses as locked and P16RP.2 is void. A parity clause is valid only on the comparator's machine, and every cross-machine comparison in the program is between different runs. Across runs, the +- pair moves by up to 16 points with no panel change.

**The clean measurement is on the same weights.** I scored each life's final weights on the old colliding panels, rebuilt exactly and verified against E16-R3's stage-0 counts, and on the new ones.
- The old +- panel's triples are 600 of 600 in the taught -> set; the new panels are disjoint from every taught set.
- On the same weights, the +- difference (new − old) is +1.0 / +6.8 / −0.8, and the mean absolute difference over the four held pairs is 1.08 / 2.37 / 1.38.
- P16RP.3 holds in two seeds and misses in 93683. That miss runs against contamination: the colliding panel scored lower. It is about 2.5 SE of panel sampling.
- The store adds nothing on these panels (no-store equals with-store within 0.1).

**Conclusion.** At block 6 the collision did not inflate held-composition scores, and the E16-family composition figures need no correction on this evidence. Blocks 4–5 cannot be measured, since only final weights were saved.

**9.54 assessed.** Two results. First, the composition figures of the E16 family stand: on the same weights the colliding panel does not score above the clean one in any seed by more than 0.8 points, and the one out-of-band difference, 6.8 points in seed 93683, runs the other way. The collision was a real defect in the panel's construction and had no measurable effect at block 6; blocks 4 and 5 cannot be measured because only final weights were saved, and the figures there carry that caveat. Second, sign-rule receipts reproduce bit for bit on one machine and diverge across machines from the first update, so every cross-machine confirmation in this program is a comparison between different runs, which is what a confirmation should be, and a parity clause is valid only on the comparator's machine. The same finding shows the plus-minus pair moving by up to 16 points between two runs of one seed on two machines with no change in code: the run-to-run spread on that pair is the size of the differences the gate is judged on, which is why the seed count carries the result rather than any one seed, and why E16-F6 adds six. On the branch, the standing-seed receipts `e16_life_T_raw65536_<seed>.json` are now E16-RP's reruns on the re-keyed panels, and E16-R3's originals are `*_oldpanels.json`; the record's E16-R3 figures refer to the originals.

## 9.57 E17, Phase 1 and 1b: the local rule silences the stacked banks, and the stacked banks do not compose even under backprop (9 November 2026, executed; both phases complete)

Section "E17, Phase 1 and 1b" of `E45_REPORT.md`; `exp45/SUMMARY_E17_PHASE1.md`; `exp45/PROBE_E17_bank2_<seed>.json`. Two results from twelve lives on the three standing seeds, one about the architecture and one about the rule.

**The architecture.** The bank2 twin, under full backprop through time, learns the three operation families to 90–96 and learns the two-operator compositions to 16–21 by final answer after six blocks (the max-min pairs 25–30, the plus-minus pairs 2–19, even the taught pairs with new operands 53–60), where the windowed cell twin reached 93 on the same panels in the same two blocks. Phase 0b's gate was on single-operator births and did not test composition. So "context-dependent writes are the missing operation" was true for the operations and is not enough for the compositions: a byte-at-a-time learner built from two stacked fixed-rate banks has no way, in this dose, to parse `c a a o1 b b o2 c c =` into two operations and an intermediate. The windowless architecture question is open again, and it is a backprop question before it is a local-rule question.

**The rule.** The local learner on bank2 falls to 0–2% on every panel within the first block, below its own birth. A read-only probe of the final weights refutes the first account (that the bank's unbounded slow state runs away under constant-size sign steps): the state is silenced instead. The first bank's state sits at a few hundredths of the birth's scale, the second bank's write gates are shut in two seeds and jammed open in the third, and the readout's drive from the state is near zero, while the twin keeps the birth's scale throughout. With the first bank frozen (Phase 1b), the learner acquires the operations in block 1 (71–81 on the family mean, comparisons 86–93) and loses them in block 2 (15–32), the first block in which half of every batch is replay from the reservoir, while the frozen twin holds 93–94 within a point of the unfrozen twin. By the clauses written before launch: P17.1b.1 refuted, P17.1b.2 holds, P17.1b.3's condition met in every seed, so the truncated first-bank trace is not the cause. What the clauses did not foresee is that the rule can learn this architecture and cannot keep what it learned under continued training; the frozen learner ends at 17–20 on the operations after six blocks while its frozen twin ends at 93, and its final weights show the memory path alive (gates open, state at the birth's scale, readout driven by the state) and wrong, unlike the unfrozen learner's silence. Phase 1c, written before launch, separates the two remaining differences from the twin on the frozen architecture: exact feedback in place of sign-symmetric feedback with the trace unchanged, and the twin's update on the learner's own replay pipeline.

**What it means for the program.** The window's removal has turned up two separate costs, and they should not be read as one. Under backprop, the stacked banks do not compose. Under the sign rule, the stacked banks do not hold. The second was already visible in miniature in E16-L (the every-byte loss fails for the sign rule on the cell and not on the window net or the twin): rule × architecture × objective. The honest statement for the record is that no windowless learner has yet met the Phase A gate under either rule, and that the windowed cell learner's result stands as the Phase A result until one does.

## 9.58 The cloud session's E16-M2: rule or data on the two failing seeds (9 November 2026; the session's section, filed verbatim; pushed at 2af0fa00)


E16-M2 separated the two accounts of the composition failures in seeds 400009 (drifting stream) and 600013 (block stages) with two arms on the same births:
- arm `M` keeps the learner's raw-byte reservoir pipeline and swaps in the twin's backprop update;
- oracle replay keeps the sign rule and feeds it the twin's joint stream.

Results:
- **The data pipeline is not the cause (P16M2.1 holds in both seeds).** `M` held a reservoir identical to the failing learner's, the same 1,048,642 and 1,048,762 bytes and the same 55,211 and 57,921 distinct strings. It ends at 0.97 and 0.95 of the twin on held compositions, with the plus-minus pairs at 76.5–86.3, and its mod-97 operations never dip below 91.
- **Oracle replay lifts the sign rule a long way but not to the twin (P16M2.2 holds in 400009 and misses in 600013).** Compositions rise from 64.9 to 84.5 (0.90 of the twin) and from 64.7 to 78.8 (0.85). In 600013 the `+-` pair ends at 62.0, below the 65 bar.
- **The mod-97 dip follows the rule, not the data, under block stages only (P16M2.3 holds in 600013 and misses in 400009).** Under stages the dip below 70 when compositions arrive reappears with oracle data (65.7) and not in `M` (92.2). Under drift, oracle replay shows no dip at checkpoint 2 (89.3). It shows a fall to 79.2 one checkpoint later instead.

By the brief's prewritten reading the seeds split. 600013 is the rule case: the sign rule's interference persists with ideal data. 400009 is the sampling case: the rule fails on the raw reservoir but passes with oracle data. Both seeds agree on what matters for the next step: the same reservoir that backprop handles at the twin's level leaves the sign rule 9.5–14.4 points short on compositions even when the data is ideal. The remaining gap is therefore the rule's, and in 400009 sampling makes it worse. Development result on a toy harness, two seeds, one stream each. Seed 600013's comparators are scored on the pre-re-keying panels. Sources: `E16M2_REPORT.md`, `E16M2_PREDICTIONS.md`, `exp45/SUMMARY_E16M2.md`, the four receipts and final weights.

**9.58 assessed.** The clean result is arm M: on a reservoir identical byte for byte to the failing learner's, backprop ends at 0.95–0.97 of the twin with no dip, so the data pipeline causes nothing on its own in either seed. The second result is that ideal data does not close the rule's gap: with the twin's own stream the sign rule reaches 0.85 and 0.90 of the twin on compositions in these seeds, where on the standing seeds it reached about 0.95 (E16-W's ceiling arm), and under stages the mod-97 dip when compositions arrive reappears with ideal data and not under backprop. I read the split between the seeds differently from the brief's two prewritten cases, which assumed the seeds would agree: in both seeds the rule is short of the twin with ideal data, and in both the raw reservoir's sampling costs the rule a further 14–20 points that it costs backprop nothing. The reservoir's sampling amplifies a weakness of the rule on these births; it does not create it. That is a rule × data interaction, and it points the next arms at the rule first (the learning-signal weighting already queued; the learner-gap comparison) and at label-free sampling second (the raw reservoir carries no family labels, so any sampling change must work from bytes, as admission does). One caveat the report states: seed 600013's comparators were scored on the pre-re-keying panels, so its composition ratios compare across panels, within the few points E16-RP measured. Development result on a toy harness, two seeds.

## 9.59 The cloud session's E16-F6, run B: six more seeds of the drifting stream (9 November 2026; the session's section, filed verbatim; branch claude/vibrant-gauss-tops5f at ac98a824, deliverables copied to `MOD97_LOCAL_RULE_20261106/E16F6_RUN_B/`)


`E16F6_REPORT.md`, `exp45/SUMMARY_E16F6.md`, `exp45/E16F6_SIGNATURE.md`.

**What was run.** Three new births (800011, 900007, 1000003; 89–98% at k = 12, P16F6.1 holds) and twelve drift lives, a learner and a twin on each of six seeds, as in E16-S. The raw reservoir held 65,536 slots with online admission.

**Results.**
- **Twins (P16F6.2): hold.** Every twin is at 90 / 85 / 90 or above. Seed 500011's twin reaches 88.4 on compositions, where its twin under stages had stopped at 84.0.
- **Learners (P16F6.3): hold, five of six.** Five learners stay at or above 0.80 of their twin on all three families, with composition ratios of 0.83–0.89.
- **Seed 600013 (P16F6.4): holds.** It misses again (compositions 0.705, mod-97 operations 0.774) with the same signature as under stages: `+-` and `-+` near 42–48, max/min pairs at 89, add and sub mod 97 near 50, the twin untouched at 94. That makes its fragility a property of the seed, not of the schedule.
- **The early signature (P16F6.5): misses.** Over twelve seeds, the Spearman correlation between the checkpoint-2 mod-97 mean and the final `+-` is 0.217 (0.450 on the nine re-keyed-panel seeds alone), against 0.5. Four seeds above 80 at checkpoint 2 end at or below 65 on `+-`. The reason is that 600013 dips at checkpoint 3 (47.2), not at checkpoint 2 (79.9).

**An exploratory reading, not a prediction.** The minimum mod-97 mean over the life separates the two failing seeds (47.2, 62.0) from the ten passing ones (66.7 and above). That is fitted after the fact, so it would need a locked prediction on new seeds before it could be used. It also suggests that the E16-M2 question (rule or data) is about when compositions displace addition and subtraction, not when that happens on a fixed clock.

**Pace.** 4.7 hours of wall time against a forecast of 5.3. The births, learners and twins each came within 3% of their forecast CPU cost.

**9.59 assessed.** Two sessions ran this job on two machines: this one, on its own branch, finished first; the session that ran E16-M2 is running the same job on the consolidated branch, where its files land under the standard names, so this run's deliverables are filed under `E16F6_RUN_B/`. The first worker's cross-check (8e958da) later showed the two runs bit-identical in all twelve receipts and three births across the two cloud containers, so this is one run and run B is a repeat, not a second sample (9.60). On this run alone: the drift gate stands at 10 of 12 seeds, the two failures are the two already known, and 600013 fails under both schedules, so its fragility belongs to the birth. The pre-registered early signature misses; the report's post-hoc replacement, the minimum of the mod-97 operations over the whole life, separates the two failures from the ten passes cleanly but was fitted after the fact, so it goes into the next brief as a prediction on new seeds and not into the record as a finding. Two things carry forward. The twin's composition shortfall in 500011 under stages did not repeat under drift (88.4 against 84.0): a schedule effect on the twin, not a birth defect. And together with E16-M2, the composition failures are a rule-level weakness of particular births that the reservoir's sampling amplifies, occurring when the compositions displace addition and subtraction, which under drift happens at different checkpoints in different seeds. The per-seed ratios here, 0.83–0.89 on compositions for the five passing seeds, sit where the standing seeds sat.

## 9.57, addendum: E17 Phase 1c, and the trace gap found by measurement (9 November 2026, executed)

Section "E17, Phase 1c" of `E45_REPORT.md`. On the frozen bank2, backprop on the learner's own replay pipeline holds the operations at 92–95 through six blocks in every seed, so the data pipeline is cleared here as E16-M2 cleared it on the windowed learner. Exact feedback in place of sign-symmetric feedback raises the learner from 15–32 to 46–67 after block 2 and does not prevent the collapse, which completes one block later. With both of those excluded, the learner's gradient was compared with backprop through time's directly rather than running the next pre-registered arm: with the first bank frozen and exact feedback, the two agree on every learned synapse except the projection that feeds the second bank's write, where the within-step gradient the rule uses has relative error 0.87; the eligibility scan extended one layer upstream reproduces the exact credit to 1.6e-7, with no new approximation. The "exact trace" of E11 was exact for the write synapses and not for the layer feeding them, and the windowed cell never exposed this because its input window made within-step credit sufficient. Phase 1d, launched with predictions written first, runs the extended trace under both feedbacks on the frozen architecture; if the exact-feedback arm holds, the collapses of Phases 1b and 1c were a trace gap, and the local-rule question for windowless learners becomes how far upstream the trace must reach, which for the unfrozen first bank is the known e-prop truncation.

## 9.60 Fable 5.2's audit memo of 9 November (`PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/MEMO_TO_FABLE_WORKER_20261109.md`, from branch claude/vibrant-cerf-sje726 at 1c7ebee), item by item, and what was done the same evening

The memo is read-only work from committed receipts and weights, with three scripts that reproduce its main claims. Both scripts that bear on the record were re-run here and agree with it. Verdicts item by item.

**Section 3, the E17 embedding defect: confirmed, and it voids the local-arm readings of Phases 1, 1b and 1c.** On the final weights of every local arm (`T_bank2`, `T_bank2_frz1`, `T_bank2_frz1_xfb`, seed 73129 and the others checked), every element of the embedding rows of the operators `+ - > <`, the letters A–Z, the birth codes and the life task codes equals the birth value times 9.9387e-5, which is AdamW's decay alone over 36,864 updates at 0.25 × 0.001; arm M and the twin learned those rows. The cause is as the memo says: the input is one byte, the previous state is detached in time, and the trace credits the write synapses and nothing upstream of them, so a symbol that reaches the loss only through memory never reaches the embedding table, and the decay erases it at 0.215 per block. That schedule alone produces "acquired in block 1, lost in block 2", which I had read as the onset of replay (9.57) and then as the truncated credit to the write projection (9.57 addendum). The in-harness check added tonight shows it on the first update: under the local arm 36 of the 47 byte rows present in the batch receive no gradient (the operators, every letter, the task codes); under backprop, 2 (the padding rows). Consequences: the Phase 1 "silence" and the Phase 1b/1c collapses are not evidence about sign feedback, stacked banks or the trace; the 0.87 relative error of the write projection's within-step gradient stands as a measurement and its causal reading is withdrawn until the rerun; Phase 1d, launched at 18:05 with the same defect, was stopped at 18:12 and its partial files archived. The architecture finding of 9.57 stands, since it is the twin's: the bank2 twin learns operations and not compositions, with embeddings it did learn. The memo's "part that is not a bug" also stands and is sharper than my phrasing of it: with credit cut at the state, this rule cannot learn an input code for a symbol that matters only through memory; the window hid that in every experiment before E17. Fix (knob `E16_BOOKDECAY=off`, tag `_nbd`, old receipts reproducible as before): the embedding table in a no-decay group in lives and births, so rows of symbols unseen at birth keep their initialisation; the frozen arm freezes the table with the first bank; a zero-gradient report on the first update written into every receipt's settings. Dry runs of the birth and four arms pass. The corrected births and backprop twins are running (Phase 2); the sign-rule reruns are written as predictions and wait for Mr. Xu's go, because of section 2.2.

**Section 4, the twin is a ceiling and the gate by item type: confirmed.** In code, arm D under `E16_STREAM=drift` draws from `joint_stream`, equal shares of every family whose drift weight has become positive, and never calls the drift sampler; it also runs at decay 0.3 against the learner's 0.25. Every sentence in the E16-S and E16-F5 record that says "a twin trained on the same drifting stream" is wrong and is corrected in the errata below: the comparator is a backprop reference trained on the joint stream of the families admitted so far, which the staged twin also is, and its block losses at checkpoints 1 and 2 equal the staged twin's. The matched control for the rule is arm M, which exists for the two failing seeds (E16-M2, 0.95–0.97) and for the standing seeds only under the old uniform replay (E16-D2). `gate_by_item_type.py HEAD` reproduces the memo's table: of 21 raw-reservoir lives, 18 pass as pre-registered on family means, 14 on the whole worked answer, 15 on mod-97 addition and subtraction alone, and 4 on the `+-` and `-+` pairs alone; the two comparison pairs have a 69% single-operator shortcut and the learner ends at 89.5 on them against 98.7 for the reference, while on the plus-minus pairs it ends at 64.5 against 86. The pre-registered gate is not rewritten; the record now says, wherever "the Phase A gate is met" appears, that it is met on family means and not met on the plus-minus pairs alone in 17 of 21 lives. The exposure overlaps (40–52% of held operation pairs are the first step of a taught composition; 47 / 44 / 21% of held compositions share a step) are accepted as stated and belong in the exposure audit the objective contract asks for.

**Section 5, corrections: accepted, with one narrowing.** (1) E16-L: the receipts show addition and subtraction at 46.8 / 40.0, 29.8 / 22.5 and 21.5 / 14.2 after block 1, under 12 after block 2 and under 6 after block 3; "does not acquire" becomes "acquires partly in block 1 and loses it from block 2". (2) "`+-` moves up to 16 points between two runs": 16.2 at block 6 includes the panel change, which the same-weights measurement puts at 6.8 in that seed; the run-to-run part is about 9 points. "The frozen twin holds 93 throughout": it ranges 89.4–95.0 by block. E16-RC "flat from 16,384 up": the three-seed composition mean does rise 74.2 → 76.9 → 78.3 → 79.8, within seed spread, which is what 9.46 says in its second sentence; "flat" is kept for the per-seed curves and the mean is now quoted. (3) The held-panel collision is fixed in `exp16_integrated.py` only; the E7, E8 and E10 harnesses still key on `seed + 1`. Those experiments are frozen and will not be rerun; the fix is on Codex's list and the record of each carries the note. (4) E16-F6 was run twice, and the first worker's cross-check (8e958da) shows the two runs bit-identical in all twelve receipts and three births across two cloud containers with different Python and NumPy versions, so there is one E16-F6, run B is a bit-for-bit repeat and adds no statistical weight, and the 9.59 assessment's "two independent runs" is withdrawn. This also narrows E16-RP: the non-reproducibility is between the owner's machine and the cloud containers, not between any two machines. (5) The prediction scorecard (17 of 33 held as written; 1 of 12 mechanism predictions) is accepted as arithmetic, and one process rule follows: a prediction over seeds already on file is computed on them before it is locked; P16F6.5 would have failed that check. (6) The three history items are relayed and not yet verified here; they are listed for checking. (7) The kit note is accepted; no adoption.

**Section 2, the three answers.** They were given by Mr. Xu, who directs this session, in his own words, and are filed as his, not as Mr. Zhou's. On 2.1 and 2.3 the memo's readings are adopted as reporting rules: every receipt will carry CPU-seconds on the reference machine, presentations per distinct item, replay share and reservoir bytes; the retention-against-replay-share curve at fixed reservoir is queued; the one-exposure, no-rehearsal test is the Phase B item the fly-swatter example describes. On 2.2 the memo is right that the plain reading of "full circuit propagation is not allowed" puts the life rule used since Experiment 1 outside the line, since the sign rule relays the output error back through every layer to the embeddings with weight signs in place of weights, and the record should say "sign-symmetric feedback through every layer" wherever it says "local". The six cases go to Mr. Zhou as written, with one added: (G) the eligibility scan extended upstream (`E16_TRACEUP`), which carries the state's credit back through the write synapses' sign-transposes toward the input; in depth it is the same relay as case A and stands or falls with it. Until the ruling: births and backprop references run (case D, as every birth so far); new sign-rule lives are written as predictions and launched only on Mr. Xu's go. If case A is ruled out, the memo's order is the right one: the ladder is climbed again from mod-97 with a rule inside the line, direct feedback (case B) and layer-local objectives (case C) first, each against arm M, stopping at the first rung that fails.

## 9.61 E17 Phase 2, the sign-rule arms with the embeddings kept (9 November 2026, 19:39–22:08 EST; `MOD97_LOCAL_RULE_20261106/exp45/SUMMARY_E17_PHASE2.md`)

**What ran.** Nine sign-rule lives and three frozen backprop twins on the `bank2` architecture (window removed, two stacked fixed-rate banks) with the embeddings in a no-decay group, the fix for the defect the 9 November audit found (9.60). Three seeds, three local arms: the sign rule on both banks (`T_bank2_nbd`), the sign rule with the first bank and the embeddings frozen after birth (`T_bank2_nbd_frz1`), and the frozen arm with exact feedback and the trace extended to the write projection (`T_bank2_nbd_frz1_xfb_tup`), against the frozen twin in the same regime (`D_bank2_nbd_frz1`). Every receipt records what received no gradient on the first update: in the unfrozen local arm 36 of the 47 byte rows in the batch (operators, letters, task codes), in backprop 2 (the two end-of-sequence rows). The embedding rows that receive no gradient now end exactly at their birth values (final/birth = 1.0000 in all three local arms), so the decay artefact is gone from these receipts.

**Verdicts, against the predictions of 19:00 EST.**

| Prediction | Result | Verdict |
|---|---|---|
| P17.2.3 unfrozen sign rule, F1 ≥ 60 after block 1 | 2.8 / 1.6 / 1.5 (Phase 1 with decay: 1.5 / 1.9 / 1.9) | **misses, all three** |
| P17.2.4 frozen sign rule, F1 ≥ 70 after block 2 and ≥ 60 after block 6 | 82.1 / 86.9 / 83.9, then 59.4 / 62.4 / 61.2 (with decay: 21 / 15 / 32, then 20 / 18 / 17) | holds in 93683, 116089; **misses 73129 by 0.6** |
| P17.2.5 frozen, exact feedback, extended trace: F1 ≥ 85 after blocks 2 and 6, within 5 of the frozen twin | 93.7 / 93.6 / 95.0, then 87.5 / 90.1 / 91.9; frozen twin 94.1 / 95.0 / 93.1, then 90.2 / 90.0 / 91.5 | **holds, all three** |
| P17.2.6 compositions, reported | held pairs 13.4 / 14.3 / 15.0 against the frozen twin's 14.9 / 17.8 / 18.1; `+-` 3.5 / 2.7 / 2.8, `-+` 3.2 / 1.2 / 1.3 | reported |

**What the three arms say.**

1. *The unfrozen sign rule destroys the birth in one block.* The development operations fall from 93–95 at birth to 1–2 after block 1 and never return; mod-97 never rises above 3. This is the same silence as Phase 1, so the decayed embeddings were not its cause. The embedding check names a second cause: the rows the rule does reach, the digits and the `=` and `;` separators, it shrinks from a row norm of 4.1 to 0.3 with sign flips (final/birth from −0.40 to +1.56 across the digit rows), while every row it cannot reach stays exactly at birth. The sign rule's own update to the input embeddings, through the write synapses, wrecks the digit code the birth built. Whether the first bank's learning does the same to the recurrent weights is not separated here; an arm with only the embeddings frozen would separate it.
2. *With the first bank and the embeddings frozen, the sign rule survives the stream but keeps a third of the gap.* It holds 82–87 through the two mod-97 blocks, loses about 30 points during the two composition blocks (to 51–54), and recovers to 59–62 by the end; the frozen twin ends at 90–92. Addition and subtraction, which the family mean hides, end at 41 / 24, 46 / 28 and 46 / 23. The Phase 1b collapse to 17–20 was the decayed embeddings; the remaining loss during the composition blocks is the rule's. Ratios to the frozen twin: mod-97 0.66–0.69, mod-73 0.77–0.81, held compositions 0.39–0.52.
3. *Exact feedback with the trace extended to the write projection closes the operation gap entirely.* Mod-97 ends within 3 points of the frozen twin (above it in two seeds), mod-73 2–3 points above it, addition and subtraction at 84–91. On the compositions it ends at 0.81–0.90 of the twin, but the twin itself composes nothing on the hard pairs (`+-` 2.5 / 10.7 / 7.3, `-+` 2.5 / 1.3 / 2.0, shortcut ceilings about 1%); nothing in this architecture composes, with any rule. This arm is also the least local of the three: exact feedback transports the forward weights, which the propagation answer relayed in 9.60 ("full circuit propagation is not allowed") excludes at least as firmly as it excludes the sign rule. Which of the two changes does the work, the feedback or the extended trace, Phase 2 does not separate; Phase 1c (with decay) showed exact feedback alone delaying the collapse without preventing it.

**Reading.** Mr. Xu's question, whether the embedding defect alone explained the Phase 1 results, has a split answer: it explained the Phase 1b collapse of the frozen arm, not the Phase 1 silence of the unfrozen one, and not the whole of the frozen arm's gap. The architecture result of Phase 1 stands unchanged: the twin without a window learns the operations and not the compositions, with or without decay on the embeddings. Evidence level E2 throughout (sealed panels, operand-disjoint from the taught triples); no reuse on never-reinforced inputs is claimed. Phase 3, prepared and not launched: the two changes of arm 3 one at a time, and the unfrozen rule with only the embeddings frozen.

**Cost and pace.** CPU-seconds per life: unfrozen sign rule 7,475–7,515; frozen sign rule 6,853–6,923; exact feedback with extended trace 6,667–6,808; frozen twin 5,611–5,738; unfrozen twin 2,603–2,624 (run earlier, three processes). Wall 19:39 to 22:08 EST, 2 h 29 min for twelve processes on the Mac. Forecast at launch about 23:30 EST, revised at 20:53 to 23:20: the measured wall came in 1 h 15 min under the forecast; the per-block pace of the first two blocks, which score four extra panels at k = 0 / 8 / 20, overstated the rest. The machine lock was released by the launcher at 22:08 with all twelve lives complete.

## 9.66 E17 Phase 3: the two changes one at a time, and the embeddings alone frozen (9–10 November 2026, 23:21–00:57 EST; `MOD97_LOCAL_RULE_20261106/exp45/SUMMARY_E17_PHASE3.md`)

**What ran.** Nine lives on `bank2` with the embeddings out of decay, three seeds, launched on Mr. Xu's go at 23:21 EST: the frozen first bank with exact feedback alone (`T_bank2_nbd_frz1_xfb`), the frozen first bank with the trace extended to the write projection alone (`T_bank2_nbd_frz1_tup`, sign feedback), and the sign rule on both banks with only the embedding table frozen (`T_bank2_nbd_frzb`). Comparators from Phase 2 (9.61): the frozen sign rule (59.4 / 62.4 / 61.2 on mod-97 after block 6), the frozen arm with both changes (87.5 / 90.1 / 91.9) and the frozen twin (90.2 / 90.0 / 91.5).

**Verdicts, against the predictions of 22:40 EST.**

| Prediction | Result | Verdict |
|---|---|---|
| P17.3.1 exact feedback alone, F1 ≥ 80 after block 6, within 8 of the frozen twin | 67.9 / 72.9 / 72.8 | **misses, all three** |
| P17.3.2 trace alone, at least 5 above the frozen sign rule and below the exact-feedback arm | 80.1 / 84.9 / 84.9: 21–24 above the sign rule, **above** the exact-feedback arm by 12 | first clause holds; second **misses in the informative direction** |
| P17.3.3 embeddings alone frozen, development operations ≥ 60 and F1 ≥ 40 after block 1 | development 1.3 / 1.5 / 1.2; F1 1.8 / 1.8 / 1.1 | **misses, all three** |
| P17.3.4 compositions, reported | held pairs 8.8 / 8.3 / 8.8 (feedback), 12.2 / 13.2 / 12.0 (trace); hard pairs 0–4 | reported; none above 25 |

**What the three arms say.**

1. *The extended trace is the larger part of Phase 2's closure, not the feedback.* With sign feedback kept and only the trace extended through time to the projection that feeds each bank's write, the frozen learner ends at 0.89 / 0.94 / 0.93 of the frozen twin on mod-97, 0.97–1.00 on mod-73, with addition and subtraction at 78 / 66, 84 / 77 and 83 / 72; it loses 10–13 points during the composition blocks against the sign rule's 30. Exact feedback alone adds 8–12 points to the sign rule and no more. The two together (Phase 2) are needed to reach the twin, but the order is the reverse of the one predicted. Under 9.60's case G, the extended trace is credit through time with the same sign-of-weight feedback as the rule itself; it stands or falls with the sign rule under the propagation ruling, whereas exact feedback falls separately.
2. *The unfrozen windowless learner is not viable under this rule.* With the embedding table frozen so that the Phase 2 embedding damage cannot occur, the sign rule on both banks still destroys the birth within block 1: development operations 93–95 → 1.3 / 1.5 / 1.2, mod-97 never above 2. The rule's update to the first bank's recurrent and write weights does on its own what its update to the digit rows did in Phase 2. The first bank has to be born by backprop and frozen; under this rule the lifelong part of the learner is the second bank only.
3. *Nothing composes.* Every arm sits at 8–13 on the held pairs, hard pairs at 0–4 against shortcut ceilings of about 1%, the frozen twin at 15–18; the Phase 1 architecture result is unchanged.

**Reading.** The windowless two-bank learner under the most local arm that stands or falls with the sign rule (frozen first bank, sign feedback, extended trace) reaches about nine tenths of its backprop twin on the operations and, like the twin, composes nothing. The two open questions for Mr. Zhou are unchanged by Phase 3 and sharpened by it: whether a sign-of-weight feedback relayed layer by layer counts as "full circuit propagation" (9.60, cases A and G), and whether a learner whose first bank must be frozen after a backprop birth meets the project's aim. Evidence level E2 throughout; no reuse on never-reinforced inputs is claimed. No further Mac run is queued.

**Cost and pace.** CPU-seconds per life: exact feedback alone 4,840–4,878; trace alone 5,129–5,152; embeddings alone frozen 5,574–5,602. Wall 23:21:40 to 00:57:03 EST, 1 h 35 min for nine processes, 40 min under the forecast of 2 h 15 min (made from Phase 2's twelve-process pace). The launcher released the lock at 00:57:03 with all nine lives complete.

## 9.62 E16-F7 (cloud session, 10 November 2026, 03:14–08:52 UTC): six more drift seeds; the signature S separates the known failures and does not rank (`MOD97_LOCAL_RULE_20261106/E16F7_REPORT.md`)

**What ran.** The whole of Job 1 of the 9 November brief in one session: six new births (1100009, 1200007, 1300021, 1400017, 1500007, 1600033), each with a drift learner (sign rule, raw reservoir 65,536, online admission) and a drift twin. Predictions committed at d96129c7 before the births; harness unchanged at e78aa83e; receipts at 41f2ba2a. The numbers below were recomputed here from the eighteen pairs of receipts and agree with the worker's tables to the digit.

| Prediction | Result | Verdict |
|---|---|---|
| P16F7.1 births ≥ 85 at k = 12 | lowest 86.3 (sub89, 1400017) | holds |
| P16F7.2 twin ≥ 90 / 85 / 90 | lowest twin means 93.7 / 91.2 / 92.2 | holds |
| P16F7.3 at least 4 of 6 at ≥ 0.80 on all three families | 6 of 6; 1600033 by 0.005 on mod-97 | holds |
| P16F7.4 the signature S, both clauses | threshold clauses hold; Spearman over eighteen seeds 0.109 against 0.35 | **misses** |
| P16F7.5 hard pairs ≤ 2 of 6, family gate ≥ 4 | 1 of 6 on `+-`/`-+`, 6 of 6 on the family gate | holds |

**What it says.**

1. *The pass rate under drift is 16 of 18 on family means and 4 of 18 on the hard pairs alone.* The learner's mean on `+-`/`-+` over the eighteen seeds is 64.8 against the twin's 87.5. The pre-registered gate is not rewritten; both counts are quoted wherever the gate is.
2. *S is not an early-warning measure.* The threshold clauses hold, but the only seeds below 65 are the two failures S was chosen on; no new seed tested the low clause. Among the six new seeds S ranks almost in reverse (Spearman −0.71, exploratory): 1400017 has S 65.2 and the best final `+-` of all eighteen (79.0); 1200007 and 1300021 have S near 82 and end at 49.7 and 51.5. The reading written in the brief beforehand applies as written: S separates failures and does not rank the passing seeds. It is dropped from later designs.
3. *A second failure pattern, seen in two seeds.* In 600013 and 400009 the mod-97 operations fell as the compositions entered. In 1200007 and 1300021 the operations held and `+-` was lost later, 15 and 30 points during the mod-73 blocks, after the window S looks at. 1200007 passes the family gate with `+-` at 49.7, level with the failing 400009. This is a description of two streams, not a mechanism; 9.64 measures the handle that bears on it.

**Cost.** 79,218 CPU-s over eighteen runs (births 3,051–3,164, learners 5,329–5,402, twins 4,643–4,848), 5 h 37 min of wall on a 4-core 2.10 GHz cloud container, 4% over the worker's forecast; no restarts. Evidence level E2, one stream per seed; no learning-core claim. Section 9.63 is not used (the job was not split).

## 9.64 E16-M3 (cloud session, 10 November 2026, 03:19–08:21 UTC): the replay share is a dose in the opposite direction (`MOD97_LOCAL_RULE_20261106/E16M3_REPORT.md`)

**What ran.** The whole of Job 2 in one session: fourteen drift lives with the raw reservoir of 65,536. The sign rule at replay shares 0 / 0.25 / 0.75 on 400009, 600013 (the two failing seeds) and 300007 (a pass); backprop on the same pipeline at 0.25 / 0.75 on the two failing seeds, and at 0.5 on 600013. Predictions committed at bec6d07f before launch; harness unchanged; the branch `claude/vibrant-gauss-tops5f` was merged here at cf2fea36 (72 files added, none changed). The share curves were recomputed here from the receipts and agree.

Hard pairs (`+-`/`-+` mean, final answer) at the last checkpoint, by replay share:

| Seed | Rule 0 | Rule 0.25 | Rule 0.5 | Rule 0.75 | Backprop 0.25 | Backprop 0.5 | Backprop 0.75 | Twin |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 400009 | 1.2 | 43.7 | 46.3 | 73.2 | 78.2 | 84.7 | 88.1 | 89.1 |
| 600013 | 1.0 | 23.3 | 44.8 | 60.2 | 71.9 | 81.8 | 86.7 | 91.2 |
| 300007 | 1.1 | 59.2 | 67.7 | 75.7 | | | | 85.8 |

| Prediction | Verdict |
|---|---|
| P16M3.1 share 0 ends below 0.80 on mod-97, at least 10 under share 0.5 | holds, all three (ratios 0.35 / 0.32 / 0.13) |
| P16M3.2 less replay helps the rule's hard pairs, more hurts | **misses in both failing seeds, reversed**: 0.75 is 27 and 15 points above 0.5 |
| P16M3.3 backprop flat across shares | **misses at 0.25** (6 and 10 under 0.5); holds at 0.75 |
| P16M3.4 the passing seed passes at both shares, 0.25 above 0.75 | **misses**: compositions 0.796 at 0.25; hard pairs 59.2 against 75.7 |

**What it says.**

1. *My premise was wrong.* I read E16-M2's result (the reservoir costs the rule 14–20 points that it costs backprop nothing) as too much verbatim replay. The receipts say the reverse: within 0.25–0.75, more replay is better for both updates and much better for the rule (29 and 37 points across the range against backprop's 10 and 15).
2. *The share works through retention, not acquisition.* At every share the hard pairs reach a similar peak by checkpoint 4 (the end of the composition blocks); lower shares reach it sooner. Only the higher shares keep them while the stream moves on to mod-73. That is the late-loss pattern of 9.62, item 3, and the same handle bears on it.
3. *At share 0.75 a failing seed passes.* 400009 meets the family gate (0.93 / 0.87 / 0.97) and reaches 0.82 of the twin on the hard pairs alone. 600013 comes to 0.796 on compositions and 0.66 on the hard pairs. The passing seed 300007 rises to 0.88 on the hard pairs. One life per seed and share: a direction, not yet a setting.
4. *Without replay everything before the current mixture is lost.* Share 0 ends at about 1% on the hard pairs from peaks of 59–85, and at 0.13–0.35 of the twin on mod-97. Under the sign rule this learner has no retention of its own; the reservoir is doing all of it. That bears on the project's aim directly and is stated here so that the passing gate is not read as more than it is.

**Not tested.** The curve has not flattened at 0.75, and fewer fresh items per batch slow acquisition (the hard-pair peak at checkpoint 3 in 400009 falls from 57 to 41 between 0.5 and 0.75). Where the curve turns, and whether 0.75 helps seeds it was not found on, are the next cloud job (E16-M4, brief of 10 November).

**Cost.** Rule lives 5,132–5,441 CPU-s, backprop lives 4,665–4,733; 5.0 h of wall on a 4-core cloud container, 3–9% over the worker's forecast on the rule lives; nothing failed. Evidence level E2; no learning-core claim. Section 9.65 is not used.

## 9.67 Mr. Xu's two answers of 10 November, what they change, and E17 Phase 4 prepared (10 November 2026, 10:30 EST)

**The answers, in Mr. Xu's words.** To the two questions that closed 9.66: "Since biological animal does not have first bank frozen, I do not feel this meet the aim. The feedback relayed layer by layer or by some sub-network / hyper-connection etc is fine." As in 9.60, these are filed as Mr. Xu's, who directs this session, and not as Mr. Zhou's; they are adopted as the working rule for design until Mr. Zhou says otherwise.

**What they change.**

1. *Case A and case G of 9.60 are inside the line for design purposes.* Sign-of-weight feedback relayed layer by layer is allowed, and so is a separate feedback pathway (a sub-network or direct long-range connections that carry the error). The stop in 9.60 on new sign-rule lives is lifted for prepared designs; runs on the Mac still launch on Mr. Xu's go. My reading of what stays outside, for correction: an update that needs the exact forward weights on the way back (weight transport), which no arm now depends on, since Phase 3 showed the trace and not exact feedback carries the closure.
2. *A learner with a frozen first bank is not a candidate.* The frozen arms of E17 Phases 1b–3 stand as diagnostics. The claim in 9.66, item 2, that the first bank "has to be born by backprop and frozen" was a statement about the rule as implemented, and the measurement below says which part of the implementation it was about.
3. *The learner of E16 already meets the first answer.* On the window architecture nothing is frozen (`freeze: none`, `frozen: []` in every receipt): every layer learns in life under the sign rule, and it passes the family gate in 16 of 18 drift seeds. The frozen bank arose only in E17, when the window was removed.
4. *One point the answers do not settle, flagged before any run.* In a windowless learner the credit for a byte read early must reach the layers that read it after the item has moved on. Carrying it there is credit relayed back along the item in time as well as layer by layer. In the harness that is a backward pass over the item (bounded by the item's length) with sign-of-weight feedback at each layer; a forward-only form exists for these fixed-rate banks but needs, for every synapse above a write, one trace per bank unit. Plainly said, the Phase 4 arm below is backprop through time over one item with weight signs in place of weights. Whether that is acceptable is Mr. Xu's and Mr. Zhou's call; Phase 4 launches only on the go, and the go is read as accepting it for a diagnostic.

**Why the rule destroys the first bank: measured at birth (`exp45/E17_ALIGN_AT_BIRTH.md`, read-only, no training).** Cosine between each rule's update and the true gradient (backprop through time) on 256 first-block items, per layer, seeds 73129 / 93683 / 116089:

| Layer | Sign rule as run | Trace to the write projection (Phase 3) | Exact feedback, same truncation | Sign feedback, credit through time at every layer |
|---|---|---|---|---|
| embeddings | 0.13 / 0.28 / 0.32 | same | 0.20 / 0.33 / 0.36 | 0.51 / 0.77 / 0.53 |
| first layer | 0.24 / 0.16 / 0.32 | same | 0.32 / 0.24 / 0.41 | 0.47 / 0.60 / 0.45 |
| first bank's projection | 0.09 / −0.06 / 0.05 | 0.54 / 0.61 / 0.34 | 0.08 / −0.01 / 0.26 | 0.81 / 0.78 / 0.55 |
| first bank's write | 0.55–0.67 | same | 0.59–0.88 | 0.71–0.78 |
| second bank's projection | 0.52–0.57 | 0.78–0.80 | 0.57–0.66 | 0.78–0.80 |
| second bank's write, readout | 0.65–0.81 | same | 1.00 | same as the sign rule |

The first bank's projection gets an update uncorrelated with the gradient, and the embeddings and first layer one barely correlated, and exact feedback does not repair it: the damage is the truncation of credit in time, not the weight signs. Under the rule as run, everything above a write learns from the current step only, so a byte that matters later is credited for nothing or for the wrong thing, and the optimizer's per-weight scaling turns those directions into full-size steps. With credit carried through time at every layer and the feedback still by weight signs, every layer is at 0.45–0.81. The window architecture never met this because the window keeps the operands in view within the step.

**E17 Phase 4, prepared, not launched.** New harness option `E16_TRACEUP=all` (tag `_tall`): sign feedback with credit through time at every layer. Checked (`e17p4_check.py`, `exp45/E17P4_CHECK.json`): on the layers that already had through-time credit it reproduces the plain sign rule's gradient to 2e-7; it equals a direct computation exactly; 48.7 ms per update against the plain rule's 54.7 and backprop's 49.4. Nine lives on the Mac, seeds 73129 / 93683 / 116089, `bank2`, embeddings out of decay, nothing frozen, raw reservoir 65,536 at block admission and share 0.5 as in Phases 2–3: the new arm (`T_bank2_nbd_tall`); the Phase 3 trace with nothing frozen (`T_bank2_nbd_tup`, the control for whether one layer of extension is enough); backprop on the learner's own pipeline (`M_bank2_nbd`, the pipeline's ceiling). Comparator: the unfrozen twin `D_bank2_nbd` on file (92.3 / 93.7 / 93.3 on mod-97, 92.6 / 91.6 / 91.9 on mod-73 after block 6). Dry runs of the three commands pass seven stages (`exp45/archive_e17p4_dry/`); under the one-layer trace 36 of 47 embedding rows in the first batch still get no gradient, under the new arm 2 (the padding rows), as under backprop. Predictions P17.4.1–5 are in `E45_PREDICTIONS.md`. Forecast: about 2 h 20 min of wall for nine processes (range 2 h to 2 h 40), from the Phase 2 and Phase 3 paces and the measured cost per update. Launcher `e17p4_lives.sh`.

**The audit session's roadmap note** (`PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/ROADMAP_NOTE_20261110.md`, branch claude/vibrant-cerf-sje726 at 9f0ed126) is on file as a reference and decides nothing. Two of its points bear on today and are noted: it advises no new sign-rule lives until the propagation ruling is in, which Mr. Xu's answer now supplies for design purposes; and it asks whether births may use full backprop (its section 6), which no answer on file addresses and which every birth so far assumes.

## 9.72 Literature check on local credit through time and layer-local targets (cloud session, 10 November 2026; `PROJECT_DOCUMENTS/LIT_CHECK_20261110/LITERATURE_CHECK_LOCAL_CREDIT_IN_TIME.md`)

**What it is.** A read-only memo from a cloud session, about fifty primary sources opened, eight central citations checked by its author against the text. It was commissioned to check four claims I made from memory when answering Mr. Xu's question on delayed feedback, and five design questions. I opened its two newest sources myself (K-prop, arXiv 2611.12530, submitted 9 November 2026; Moon et al., arXiv 2610.04867) and the memo's account of both matches the papers. The rest is the worker's reading, not re-checked here. The memo decides nothing.

**My four claims.**

| Claim as I made it | Verdict |
|---|---|
| Exact online credit within a layer of element-wise recurrent units costs about one trace per parameter; stacked versions approximate the credit through an upper layer's memory | first half confirmed; second half **corrected**: K-prop carries that credit forward, exactly for two leaky layers, at one extra trace per synapse per distinct upper time constant |
| Layer-local objectives match backprop on small problems, fall behind at scale, and are not shown at language scale | **corrected on both halves**: with few blocks they match backprop on ImageNet, and the gap grows with the number of local pieces; a 125M and 400M language-model study exists, never matches end-to-end, and finds that blocks above the first barely improve on what they inherit |
| Sign symmetry stays near backprop at scale | **corrected**: within about 4–11 top-1 points on ImageNet ResNets, and only with a per-layer scale or normalisation; **no study of sign symmetry through time or in recurrent networks was found** |
| Fly dopamine timing sets the sign of plasticity at the output synapses; dopamine neurons carry more than reward | confirmed, narrower: within about a second, in particular compartments |

**What it changes here.**

1. *E17 Phase 4 is outside the published evidence.* Weight-sign feedback carried through time has no published test that the worker could find. Its result stands on our receipts alone. Our feedback already carries a per-layer scale (the mean absolute weight) in every layer except the Phase 3 trace extension, which used raw signs.
2. *E18's local heads have a published analogue with a known failure.* Per-block next-token heads in language models leave the upper blocks nearly redundant. That is the outcome P18.2 hedged against; the memo's remedies (a look-ahead for the lower head, a view of what the upper stage needs) are the candidates if E18 shows it.
3. *The cost objection to Phase 4 has a possible answer.* 9.67 item 4 said the forward-only form of credit through the upper bank needs one trace per synapse per upper unit. For leaky layers with a linear input from the layer below, the count is one per distinct upper rate, and our banks have five. Our second bank does not meet the condition (its write is gated and nonlinear). Whether a two-bank learner that does meet it can still be born is the question of E19 (`OPUS_TASK_E19_FORWARD_ONLY_TWO_BANK_20261110.md`).
4. *On the frozen first stage.* The memo reports adult plasticity at every stage of the fly's olfactory pathway after the receptor neurons, and a receptor layer that is comparatively stable after an early-adult window. Both are reported without argument. Mr. Xu's answer in 9.67 stands as given; the memo neither supports freezing a bank nor contradicts the answer.

Claims 2 and 3 of my answer to Mr. Xu, and the corresponding lines of the audit session's roadmap note, are to be quoted in the corrected form from now on.

## 9.70 E17 Phase 4: with credit through time at every layer, the windowless learner keeps its birth with nothing frozen (10 November 2026, 10:31–12:35 EST; `MOD97_LOCAL_RULE_20261106/exp45/SUMMARY_E17_PHASE4.md`)

**What ran.** Nine lives on `bank2` with the embeddings out of decay and nothing frozen, seeds 73129 / 93683 / 116089, launched on Mr. Xu's go at 10:31:24 EST from the harness at a3c0b675: the new arm (`T_bank2_nbd_tall`: weight-sign feedback, credit carried through time at every layer); the Phase 3 trace with nothing frozen (`T_bank2_nbd_tup`: credit through time to the write projections only); and backprop on the learner's own pipeline (`M_bank2_nbd`: same blocks, reservoir, replay share and decay as the learner, one difference, the update). Comparator: the unfrozen twin `D_bank2_nbd` (backprop on the joint stream, a ceiling). All nine lives loaded the same three births (hashes in the receipts) and ran seven stages; no error, no restart.

**Mod-97 mean after each block, and the end of the life.**

| Arm | Seed | Block 1 | Block 2 | Block 3 | Block 4 | Block 5 | Block 6 | mod-73 at 6 | Ratio to twin, mod-97 · mod-73 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| New arm | 73129 | 92.9 | 90.2 | 82.9 | 80.6 | 86.0 | 87.9 | 90.8 | 0.95 · 0.98 |
| | 93683 | 92.1 | 93.0 | 84.9 | 82.0 | 82.2 | 86.1 | 86.9 | 0.92 · 0.95 |
| | 116089 | 90.7 | 92.4 | 75.5 | 81.6 | 83.9 | 86.9 | 89.4 | 0.93 · 0.97 |
| One-layer trace | 73129 | 65.0 | 65.5 | 35.6 | 31.8 | 41.9 | 44.2 | 49.9 | 0.48 · 0.54 |
| | 93683 | 75.3 | 75.7 | 40.0 | 29.1 | 42.1 | 42.1 | 47.2 | 0.45 · 0.52 |
| | 116089 | 74.9 | 74.2 | 39.2 | 37.6 | 41.4 | 42.6 | 46.9 | 0.46 · 0.51 |
| Backprop, same pipeline | 73129 | 93.6 | 93.2 | 87.6 | 89.4 | 92.2 | 90.9 | 93.5 | 0.98 · 1.01 |
| | 93683 | 94.5 | 93.2 | 90.8 | 86.1 | 92.8 | 92.3 | 93.2 | 0.99 · 1.02 |
| | 116089 | 93.8 | 94.0 | 90.2 | 92.4 | 93.4 | 94.2 | 93.2 | 1.01 · 1.01 |
| Twin | all | 92.2–93.6 | | | | | 92.3–93.7 | 91.6–92.6 | |
| Sign rule as run (Phase 2) | all | 1.5–2.8 | | | | | 0.7–1.2 | 1.1–2.2 | |

**Verdicts, against the predictions of 10:30 EST.**

| Prediction | Result | Verdict |
|---|---|---|
| P17.4.1 the birth survives: new arm at or above 80 on mod-97 after block 1 in two of three, above 50 in all | 92.9 / 92.1 / 90.7 | **holds, all three** |
| P17.4.2 closure: new arm at or above 0.80 of the twin on mod-97 and mod-73 after block 6 in two of three | 0.95 · 0.98, 0.92 · 0.95, 0.93 · 0.97 | **holds, all three** |
| P17.4.3 one layer is not enough: one-layer trace below 50 on mod-97 after block 1 in two of three | 65.0 / 75.3 / 74.9 | **misses as written** |
| P17.4.4 the pipeline's ceiling: backprop on the pipeline at or above 0.90 of the twin on both families in all three | 0.98–1.01 and 1.01–1.02 | **holds, all three** |
| P17.4.5 compositions, reported: none above 25 on held pairs, hard pairs at or under 15 | held 12.8–14.7 (new arm), 7.7–9.0 (one-layer), 14.0–18.5 (backprop); hard pairs 0.8–8.6 | as expected; nothing composes |

**What the three arms say.**

1. *The statement in 9.66 that the first bank must be frozen is withdrawn.* With credit carried through time at every layer and weight-sign feedback kept, the unfrozen learner does not lose its birth: it learns the first family to 91–93 in block 1, where the rule as run fell to 2, and ends at 0.92–0.95 of the twin on mod-97 and 0.95–0.98 on mod-73. It ends above the frozen learner of Phase 3 (80.1 / 84.9 / 84.9) in every seed. What the first bank needed was credit that reaches it through time, as the alignment measurement of 9.67 said beforehand.
2. *What weight signs cost, against the matched control.* Backprop on the same pipeline ends 3–7 points above the new arm on mod-97 and 3–6 on mod-73. The gap is in addition and subtraction (new arm 77–86, backprop 88–93); the comparisons are level (91–94 against 92–97). The pipeline itself costs backprop nothing here (0.98–1.02 of the twin), so that gap belongs to the weight signs.
3. *One layer of extension keeps the learner alive and no more.* P17.4.3 misses as written: with the trace extended to the write projections only, the birth is not destroyed in block 1 (65–75). But neither reading I wrote beforehand obtained. The cheaper trace does not suffice: it ends at 42–44 on mod-97 and 47–50 on mod-73, about half the twin, and loses 30–46 points when the compositions enter. The embeddings and the first layer, still credited within the step, are the difference: under this arm 36 of the 47 embedding rows in a batch get no gradient at all, under the new arm 2.
4. *The rule keeps less of what is not rehearsed.* The development operations (mod 89) are never shown in life and are in no reservoir. After block 6 the new arm holds 10.7 / 22.2 / 43.2 on them; backprop on the same pipeline holds 48.6 / 41.4 / 53.6 and the twin 49.1 / 37.9 / 51.5. This is the E16-M3 finding again in another form (9.64, item 4): under the rule, what replay does not carry is lost faster.
5. *Nothing composes,* in any arm, the twin included. That remains the architecture's open problem (Phase 1) and no rule arm bears on it.

**What this is and is not.** The new arm is, plainly, backprop through time over one item with weight signs in place of weights, computed by a backward pass over the item. Mr. Xu's go was read as accepting that for a diagnostic (9.67 item 4), not as a ruling; whether it meets the project's locality constraint is disputed and is Mr. Zhou's to decide. The literature check found no published test of weight-sign feedback through time (9.72), so this result rests on these nine receipts: three seeds, one life each, evidence level E2 on sealed panels, no reuse on never-reinforced inputs claimed. A forward-only form of the same credit is the question of E19.

**Cost and pace.** CPU-seconds per life: new arm 6,672–6,787 (forecast 6,000–6,800); one-layer trace 7,034–7,068 (forecast 7,500–7,900, under); backprop on the pipeline 6,615–6,732 (forecast 5,500–6,500, over by 2–4%). Wall 10:31:24 to 12:35:27 EST, 2 h 04 min for nine processes against a forecast of 2 h 20 (range 2 h to 2 h 40); the E18 checks and dry runs shared the machine for part of it. The launcher released the lock at 12:35:27.

## 9.75 Codex's source audit of 10 November: every finding checked, errata to 9.62, 9.64, 9.67 and 9.70, and what was fixed (10 November 2026, 15:00 EST)

**The audit.** `PROJECT_DOCUMENTS/MINIFLY_E16_E19_HOSTILE_SOURCE_AUDIT_20261110.md` (Codex, commit c5e2715), from source and receipts, no code run. I ran its claims. Every one I could test holds, several against my own filed text. Measurements: `MOD97_LOCAL_RULE_20261106/exp45/AUDIT_RESPONSE_CHECKS.json`, `E16_AUDIT_TABLES.md`, `E17P4_INDEPENDENT_CHECK.json`, `E18_CHECK.json`. The item-by-item response is `PROJECT_DOCUMENTS/AUDIT_RESPONSE_20261110.md`.

**Errata to the record.**

1. *9.62, the drift pass counts mix two panel versions.* The three E16-S seeds (73129, 93683, 116089) were scored on the original composition panels; the other fifteen on the corrected ones. By stratum: corrected panels, 13 of 15 on family means, 2 of 15 on the plus-minus pairs alone, 9 of 15 on the whole worked answer; original panels, 3 of 3, 2 of 3, 3 of 3. "16 of 18" and "4 of 18" are not one population and are not to be quoted without the strata.
2. *9.62 item 3 and 9.64 item 2, "the end of the composition blocks" is wrong for the drifting stream.* At checkpoint 4 the composition and mod-73 families have equal weight; fresh compositions arrive until three quarters of the life and are replayed after that. The change in `+-` from checkpoint 4 to 6 is net change under continuing supervision, not forgetting after teaching stops. P16M4.3 in the E16-M4 brief inherits this and can only be read that way.
3. *9.64, the replay-share result is an allocation result.* At a fixed number of updates, more replay means fewer fresh items: 1,769,488 admitted at share 0.25, 1,179,680 at 0.5, 589,872 at 0.75 (seed 400009). The share is in rows while the loss is in supervised bytes, three for an operation and six for a composition. "More replay retains the hard pairs" stands as observed; "retention, not acquisition" does not follow from these lives and is withdrawn as a mechanism.
4. *E16-M2, staged seed 600013: the one-factor reading is withdrawn for that seed.* The learner and the reference were scored on the original panel, backprop-on-the-pipeline and the oracle arm on the corrected one (25.2 against 24.6 at checkpoint zero from the same birth). The learner's and reference's final weights were not saved, so no re-score is possible; a repeat of all three arms on one panel and one machine is prepared and waits for the go. The drift comparison for 400009 is on one panel and is unaffected.
5. *9.70, three corrections to the Phase 4 reading.* (a) "Keeps its birth" overstated it. The arm stays trainable: it learns a new modulus to 91–93 in block 1. The birth's own operations fall from 93–95 to 59.2 / 47.0 / 60.2 after block 1 (reference 62.8 / 58.0 / 67.1) and to 10.7 / 22.2 / 43.2 at the end. (b) Beyond the trained gap the skill does not hold: at 40 distractors final addition is 18.2 / 9.2 / 28.2 and subtraction 16.8 / 25.8 / 19.0, against 77–86 at 12. (c) The two rule arms differed in feedback map as well as in time reach: the one-layer trace relayed raw signs, the every-layer arm signs scaled by the mean weight. Measured at the three births, the two maps give projection gradients with cosine 0.991–0.999 to each other and the same alignment with the true gradient (0.55 / 0.66 / 0.37 raw, 0.54 / 0.67 / 0.38 scaled), so the confound is small at birth; it is untested over a life, and `E16_PROJFB=scaled` now exists to hold the map fixed.
6. *9.67, the Phase 4 check was not independent.* `e17p4_check.py` compared the option with a call into the same implementation. `e17p4_independent_check.py` now writes the forward and backward passes by hand, without autograd; the harness matches it to 2e-15 on every parameter at the three births. It also shows that the start-of-sequence row of the embedding never receives a gradient (the table's padding row), which was not on record.
7. *E18, the cheapest arm cannot learn most of its input code.* Measured at initialisation: under B3 only the digit rows and the `=` row of the embedding receive any gradient, 11 of 45 symbols; operators, task codes and letters receive none and stay at their random values. Under B0, B1 and B2 all 45 do. A B3 failure therefore cannot separate "cannot learn its input code" from "cannot learn the operations", and P18.6 is to be read as a statement about the whole package. The four arms also differ in more than the relay (auxiliary head, parameter count, feedback form, decay); `E16_HEADS=aux` (same head, relay kept) and `E16_FIXEDCODES=on` (identical fixed codes) now exist as one-factor controls and have not been run.
8. *E19, my brief confounded decay with the rule.* Stage A runs at decay 0.30 and the rule arms at 0.25, so P19.3's ratio does not isolate weight signs. An addendum asks the worker for a backprop birth at 0.25.
9. *The held panels are held out by ordered pair only.* Measured: 25% of held off-diagonal pairs of mod-97 and mod-73 have their reversal in the taught pool (49–52% at birth, where half is taught), which matters for addition, max and min; and 71–73% of held mod-97 pairs are supervised as a step inside a taught composition. Re-scoring the saved final weights of the fifteen corrected-panel learners and references on strict panels (neither route open) moves the learner's mod-97 mean by −1.2 to +1.7 points (mean +0.2) and every ratio by at most 0.025; one marginal seed, 1600033, moves from 0.805 to 0.798. So the overlap is real and did not inflate these scores measurably. "Evidence level E2" in this line means held out by ordered pair; strict panels are scored in every receipt from now on.
10. *The score-time store.* The learner has it and backprop-on-the-pipeline does not. On held single operations at k = 12 it contributes nothing: receipt and no-store scores are identical in all fifteen seeds. On compositions it is not measured; receipts now carry both.

**Fixed in the harness** (defaults reproduce earlier receipts; two archived dry receipts reproduce score for score and loss for loss): a credit census by symbol role in every life and birth, with embedding movement from initialisation per checkpoint; both head losses logged; strict panels and no-store scores in every receipt; rows and supervised tokens by origin and family per stage; panel hash, replay rows per batch, and a completion record naming the hash of the saved weights; births refuse an existing checkpoint whose receipt is missing or differs, and refuse an every-byte loss; a replay share whose tag does not determine its row count is refused; the isolation check fails on its numbers and includes a fixture that must leak; the summarizer joins arm and comparator on recorded fields and lists mismatches; a lock that is created exclusively with a unique token, and a launcher that records each job's exit code.

**Not fixed, and why.** The E19 items (rate buffers against constants, a streaming inference check, the control arm's memory) are in the worker's files on its branch, mid-run; they are in the addendum and will be applied at merge. Old receipts are not rewritten. Persistent state across records, and the audit's fifth point on durable learning, are design questions, not defects.

## 9.76 A decision on direction: composition from memory comes before more rule and replay variants (10 November 2026, 16:00 EST)

**What the receipts say when taught and untaught compositions are read apart** (end of life, whole worked answers):

| Learner | Taught compositions | Untaught plus-minus pairs | Untaught comparison pairs |
|---|---|---|---|
| Window reference, backprop (200003 / 300007 / 400009) | 92.8 / 91.5 / 92.5 | 88.8 / 85.2 / 88.8 | 98.3 / 96.2 / 98.0 |
| Window learner, sign rule | 77.5 / 84.5 / 82.8 | 67.8 / 66.0 / 44.7 | 89.1 / 88.2 / 76.0 |
| Windowless two-bank reference, backprop (73129 / 93683 / 116089) | 48.7 / 58.5 / 51.7 | 0.8 / 6.8 / 2.9 | 22.2 / 23.1 / 25.4 |
| Windowless, backprop on the learner's pipeline | 36.3 / 42.5 / 40.2 | 0.8 / 3.8 / 1.8 | 19.0 / 20.0 / 26.8 |
| Windowless, weight signs with credit through time | 16.5 / 24.2 / 25.5 | 0.4 / 2.4 / 1.0 | 19.9 / 20.4 / 20.7 |

**The fact that explains the table.** A composition prompt is ten bytes (`cAA+BB-CC=`) and the window learner's window is ten bytes. At `=` the whole prompt is in view; when the final answer is written, the second operator, the third operand and the intermediate result are in view. The window learner has never had to remember an operand in order to compose. The windowless learner must remember all of them, and under backprop it half-learns the compositions it is taught and does not transfer at all; its comparison pairs sit below the 66–69% that ignoring one operator would give.

**The decision, mine.** The project's open problem at this point is not a learning rule and not a replay setting: no learner on file composes from memory. E18, E19 and E16-M4 are left to finish and will be filed, but no further credit-rule or replay-share variant is commissioned until that is answered. Two jobs are briefed (`OPUS_TASK_E20_COMPOSITION_FROM_MEMORY_20261110.md`): E20-A puts distractor letters inside composition prompts so that the window learner, its reference and the windowless reference must carry an operand; E20-B asks whether any ordinary windowless recurrent network of the same size composes from memory under backprop, with one added loop between the banks as the third arm. Backprop is used there as a reference for what can be computed, not as a candidate. The promotion test proposed in the status note of 10 November is not taken up now: an assay of learn, revise and act is premature while the simplest composition from memory has no positive reference; it is the owner's to overrule.

**What this changes in earlier wording.** Wherever this review says the learner "composes", read "composes with every operand in view". The Phase A gate stands as registered and is a gate on that.

## 9.79 Codex's complete audit of 10 November: E20-A withdrawn and rebuilt, the exposure measurement redone, E19 closed (10 November 2026, 17:00 EST)

**The audit** (`PROJECT_DOCUMENTS/MINIFLY_E16_E20_COMPLETE_HOSTILE_AUDIT_20261110.md`, Codex, e97fa270) is again from source and receipts with nothing run. I ran what could be run. Every claim held; one was worse than stated. Item by item: `PROJECT_DOCUMENTS/AUDIT_RESPONSE_2_20261110.md`.

**Errata.**

1. *9.76, the table header "Taught compositions" was wrong.* That panel (`within`) is new operand triples of taught operator pairs. The windowless reference's 49–59% there is transfer to new instances, not failure to fit what it was taught; whether it fits taught instances is not on file. The decision in 9.76 stands on the 1–7% for untaught pairs.
2. *9.76, "the window learner has never had to remember an operand" stands; "composes in view" does not mean it ignores its state.* With the recurrent state zeroed its held compositions fall to zero. That shows dependence on state under an out-of-distribution ablation, nothing more.
3. *9.75 item 9, my strict re-score was not strict and is withdrawn.* The classifier checked composition steps in one orientation and ignored max and min taught under another modulus: 71–73% of the "strict" addition, max and min pairs were still supervised in reversed form, and about 70% of the max and min pairs under another modulus. It also resampled items. Redone on the same registered items with a corrected classifier, fifteen corrected-panel seeds pooled (`exp45/E16_EXPOSURE_STRATA.md`): of 6,000 registered mod-97 items per operation, only 361 (addition), 125 (max) and 136 (min) have no equivalent supervision anywhere; subtraction has 1,656. On those clean items the learner scores 75.6 (addition; 74.2 on all), 73.7 (subtraction; 73.1), 94.9 (min; 95.9) and 89.6 (max; 94.8), with the reference at 90.0, 90.5, 97.8 and 97.6. So for three operations clean items score like the rest; for max the learner is 5 points lower on 125 items, which is suggestive and small. The plain statement is that the registered mod-97 panel is 94–98% items whose relation is supervised somewhere in another form, and it should be described as held out by ordered pair and prompt, not as untaught relations.
4. *The strict tally.* On the first strict panel seed 1600033 fell to 0.798 on mod-97, which would make the corrected-panel family count 12 of 15. I reported the ratio and not the count. The registered count is 13 of 15; the sensitivity is 12 of 15.

**E20-A version 1 is withdrawn, by my decision, before any worker started it.** The audit is right that it could have answered the wrong question. On its panel the first operand was fully in view in 6.6% of items and hidden in 88.7%, pooled; it scored the final suffix; one arm had the store; its one prediction tied existence to the size of a penalty; and the qualifier failed on its prompts. Version 2 (same brief, rewritten): twins that share their last ten bytes at `=` and differ only in the hidden first operand, both intermediate and final results differing, scored both-correct on whole answers with no store in any arm; the same items with the letters leading as a same-length in-view control, taught in distribution; taught instances, new instances of taught pairs and untaught pairs kept apart; a suffix-only predictor asserted to score zero; existence and cost predicted separately (P20.1–6); and two stages, the window reference first, the other nine lives only if the in-view control works. E20-B gains a matched two-bank arm run through the worker's own loop.

**E19 is closed as registered, by my decision.** Recomputed from the nine receipts on the worker's branch. The gate passes: the linear-write network is born by backprop at 69.0 / 70.5 / 69.3 on the hard mean (the gated network: 91–95). P19.3 misses in every seed: the forward-only rule with weight signs reaches 43.2 / 45.8 / 54.5, a ratio of 0.63 / 0.65 / 0.79. P19.4 holds: it is 8.0 / 14.5 / 23.8 points above the same rule without the deep traces, same decay, same network, same dose, which is the cleanest one-factor result of these two days. The worker's report attributes the whole gap to weight signs; that is not established, since the ceiling ran at a different decay and no arm trained with exact forward feedback. The rule costs 220 ms per update against 103 for backprop. The mathematics is sound and independently checked; the learner it yields ends below half on the hard mean at twice the cost. The line is parked: the decay-matched control already requested is accepted if it arrives, and nothing further is commissioned on it while composition from memory is open.

**Fixed in the harness since 9.75:** birth admission by expected hash with a manifest of legacy births; the summarizer withholds ratios across panel or objective mismatches and marks unhashed receipts as legacy; exposure counts follow the actual loss mask; update and scoring CPU are separate; the launcher template returns its jobs' status. A deliberately mismatched birth stops a life.

**What I take from this audit.** I designed E20-A in an afternoon to test the one question I had just decided mattered most, and I made in it the errors I had listed that morning as lessons: no information audit of the new panel, a pooled endpoint, a suffix metric, a compound prediction. The auditor's stopping rule is right and I adopt it: the assay must be shown to detect memory use on a reference before a roster is spent on it.

## 9.71 E18: births of the two-bank learner from nothing under four packages (cloud session, 10 November 2026; `MOD97_LOCAL_RULE_20261106/E18_REPORT.md`)

**What ran.** Twelve births, seeds 73129 / 93683 / 116089, 24,576 updates each, predictions committed before the first birth (0ea40628). I recomputed every number below from the twelve birth receipts and checked each one's final step, head setting, birth rule, trace setting and decay; they are as registered and the worker's verdicts match mine. These receipts predate the hashed format and are legacy evidence in that sense. "Hard mean" is the mean of add89 and sub89 at twelve distractors, whole answers, development panels.

| Arm | What the package is | Hard mean | At 20 distractors | Max/min mean |
|---|---|---|---|---|
| Reference on file | backprop end to end (owner's machine; levels, not digits) | 91.2 / 94.5 / 90.3 | | 96.0 / 95.2 / 95.5 |
| B0 | weight signs at every relay, credit through time at every layer, error relayed between banks; no backprop anywhere; decay 0.25 | 87.5 / 90.7 / 84.7 | 84.5 / 88.0 / 74.7 | 98.8 / 97.8 / 95.7 |
| B1 | a next-byte head at each bank, no gradient between banks, exact backprop inside each; decay 0.30 | 80.7 / 82.2 / 77.2 | 79.2 / 73.3 / 73.5 | 93.8 / 95.7 / 94.5 |
| B2 | as B1 with weight signs and credit through time inside each bank; decay 0.25 | 74.8 / 69.7 / 66.3 | 71.2 / 64.3 / 62.2 | 96.0 / 96.7 / 93.8 |
| B3 | as B2 with the trace on the write synapses only | 1.2 / 0.8 / 0.8 | | 0.8 / 2.0 / 2.8 |

| Prediction | Result | Verdict |
|---|---|---|
| P18.1 the first bank's own head stays a one-bank learner (add89 under 45) | 27.3 / 29.7 / 28.3 | holds, all three |
| P18.2 the second bank builds on it: hard mean at least 70 in two seeds, below the reference in all | 80.7 / 82.2 / 77.2 | holds, all three |
| P18.3 B0 at least 0.80 of the reference in two seeds | 0.96 / 0.96 / 0.94 | holds, all three |
| P18.4 B0 at least 5 points above B1 in two seeds | +6.8 / +8.5 / +7.5 | holds, all three |
| P18.5 B2 within 10 points of B1 in two seeds | −5.8 / −12.5 / −10.8 | **misses** (one of three) |
| P18.6 B3 under 40 in all three, max/min at least 20 under B2's | 1.2 / 0.8 / 0.8; 91–95 under | holds as written |

**What it says, and what it does not.**

1. *A birth with no backprop anywhere.* B0 is born at 0.94–0.96 of the backprop birth, and its curves were still rising at the last checkpoint (84 → 88, 88 → 91, 78 → 85). Every birth before this used backprop. For this skill, under this rule, it is not needed. The rule is the one of 9.70: a backward pass over the item with weight signs, whose standing under the locality constraint is the owner's to decide.
2. *B0 against B1 is a package gap, not a price of the relay.* The two differ in the relay, the extra head and its 657,153 parameters, the feedback form and the decay. P18.4's "the relay is worth more than exact weights" was my wording for a comparison that cannot carry it.
3. *The nearest thing to a one-factor reading among these arms is B0 against B2:* same rule, same feedback, same decay; B2 adds the first bank's head and cuts the gradient between banks. B2 is 12.7 / 21.0 / 18.4 points below. Two differences remain (the head, the cut), so this too is a package gap; the control that keeps the head and the relay exists in the harness (`E16_HEADS=aux`) and has not been run.
4. *A next-byte target at each bank does work, at a cost.* The first bank alone answers 27–30% of additions, as a one-bank network does, and the second bank still reaches 77–82% on top of it. The published failure of this design, an upper stage that adds nothing, does not occur here; the upper bank adds about 60 points on the hard mean (the first bank's own head is at about 20). The package ends about 10 points under the end-to-end reference.
5. *B3 learned nothing, and that says nothing about the bank.* Its loss never moved. As measured before it ran (9.75), it sends gradient to 11 of 45 input symbols; measured now from its saved weights, its operator, letter and task-code embeddings are at their initial values to two decimal places after 24,576 updates, while its digit rows moved 0.7 of a row norm (`exp45/E18_EMBEDDING_MOVEMENT.json`). P18.6 holds as written and is a statement about a package that cannot learn its input code.

**Decision.** The two one-factor controls for this line (the kept relay with the extra head; identical fixed input codes) are defined and not commissioned: by 9.76 no further credit-rule variant runs until composition from memory has an answer. What E18 changes for that work is one thing: a learner can now be born by the rule it lives under.

**Cost.** 65,118 CPU-s for twelve births, 4 h 39 min of wall on a four-core cloud container, 5–6% over the worker's forecast; no restarts. Evidence level E2 on development panels held out by ordered pair; births only: no life, no retention, no composition is tested.

## 9.81 The twelve repeats on the corrected panels (10 November 2026, 14:27–16:45 EST; `MOD97_LOCAL_RULE_20261106/exp45/E16_AUDIT_TABLES.md`)

**What ran.** Twelve window-learner lives on the owner's machine, launched on Mr. Xu's go after the first source audit, receipts tagged `_cp`, originals untouched. Every job exited with code zero; every receipt has seven stages, a completion record and a weight hash that matches its file; every comparison below is between receipts with the same panel hash; and the arms of each staged seed start from the same composition score at checkpoint zero (24.17; 24.58), which the original receipts did not.

**Drift, the three development seeds (73129 / 93683 / 116089), learner against reference.**

| | 73129 | 93683 | 116089 |
|---|---|---|---|
| Ratios to reference, mod-97 · held pairs · mod-73 | 0.97 · 0.86 · 0.99 | 0.93 · 0.87 · 0.96 | 0.84 · 0.86 · 0.97 |
| Plus-minus pairs alone | 0.81 | 0.82 | 0.84 |
| Whole worked answers | 0.84 | 0.84 | 0.83 |

**Staged, seeds 400009 / 600013: learner, backprop on the learner's pipeline, reference; one machine, one panel.**

| | 400009 | 600013 |
|---|---|---|
| Held-pair ratio to reference: sign-rule learner | 0.83 | 0.64 |
| Held-pair ratio: backprop on the learner's pipeline | 0.98 | 0.94 |
| Plus-minus pairs, learner / backprop on pipeline / reference | 67.4 / 86.8 / 90.4 | 46.6 / 80.9 / 89.2 |

| Prediction | Result | Verdict |
|---|---|---|
| P-CP.1 all three drift seeds pass the family gate again | all nine ratios at or above 0.84 | holds |
| P-CP.2 at most one of the three passes on the plus-minus pairs alone | 0.81 / 0.82 / 0.84: all three pass | **misses** |
| P-CP.3 backprop on the pipeline at or above 0.90 on held pairs, the learner at least 0.10 below it, both staged seeds | 0.98 against 0.83; 0.94 against 0.64 | holds, both |
| P-CP.4 the learner's held-pair mean with and without the store differs by under 1 point | identical to two decimals in all five learner lives | holds |

**What it says.**

1. *The eighteen drift seeds are now one population on one panel version:* 16 of 18 pass on family means, 5 of 18 on the plus-minus pairs alone, 12 of 18 on whole worked answers. Those are the counts to quote.
2. *But the three development seeds are not like the other fifteen, and I predicted the opposite.* They pass every gate (3 of 3 on the hard pairs); the fifteen later seeds pass the hard-pair gate in 2 of 15. Two things differ between the groups and are confounded: the reservoir size was chosen on these three seeds as the smallest that met the gate in each of them (9.46), and they were born and run on the owner's machine while the fifteen ran on cloud containers. The honest estimate of how often this learner keeps its hard pairs is the fifteen seeds the settings were not chosen on: 2 of 15. The pooled 5 of 18 flatters it.
3. *E16-M2's conclusion survives a common panel.* With the three arms on one machine and one panel, backprop on the learner's own reservoir and stream reaches 0.94–0.98 of the reference on held compositions and the sign rule 0.64–0.83. What the reservoir pipeline costs the rule it does not cost backprop. The withdrawal in 9.75 item 4 is lifted for this claim, on these new receipts; the original mismatched comparison stays withdrawn.
4. *The score-time store plays no part in held compositions,* nor, as measured earlier, in held single operations. Comparisons between arms with and without it are not affected by it on these panels.
5. *On the owner's machine the lives are deterministic:* the three drift repeats reproduce the original single-operation scores to the digit; only the composition panel changed, and the held-pair means moved by under a point.

**Limits.** The `strict_k12` scores inside these twelve receipts were made by the first strict classifier, which was wrong (9.79); they are not used. Composition here is composition in view (9.76).

**Cost.** Learners 6,275–6,307 CPU-s, backprop on the pipeline 5,718–5,742, references 5,675–5,807; 2 h 17 min of wall for twelve processes against a forecast of 2 h 45 (range 2 h 20 to 3 h 10), just under the range, with the E20 and E18b dry runs sharing the machine. The lock was released at 16:44:44.

## 9.82 The audit session's next-steps note, and what is implemented from it (10 November 2026, 19:00 EST)

**The note.** `PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/NEXT_STEPS_NOTE_20261110.md`, from the cloud session that wrote the memo of 9 November and the roadmap note, copied here unchanged from its branch at its latest commit. It is a read of the record, not an audit. Mr. Xu's instruction on receiving it: "The Mac is idle. We received the next step note. Now implement the steps." What follows is what was implemented, item by item, and where the reading is mine.

| Note section | What it asks | Done |
|---|---|---|
| 3.1 Close the open rulings | accept the backward sweep with weight signs as the working life rule with a declared sweep length; accept a bounded window as a sensory buffer; any optimizer for references, a candidate born by its own rule before any promotion claim; say whose rulings are final | **adopted as working rules, on my reading of Mr. Xu's instruction**; see below |
| 3.2 If the sweep is accepted, drop the fixed-rate banks | let E20-B choose the windowless network; no repairs to the two-bank design meanwhile | adopted as the decision rule for when E20-B reports |
| 3.3 step 1 Backprop at zero replay | three lives | **running** on the owner's machine, with the rule at zero replay repeated beside it on the same machine (six lives; predictions P-Z.1–3) |
| 3.3 steps 2 and 3 | a cost curve with fresh items fixed; one protection mechanism | not started: step 2 needs a new option and waits for step 1; step 3 waits for the literature check |
| 3.4 A life with every next byte as the target | six lives | **running** (predictions P-E.1–4) |
| 3.5 One target test, design only | merge the promotion test, the delayed-observation test and the battery | drafted for audit: `PROJECT_DOCUMENTS/TARGET_TEST_DESIGN_DRAFT_20261110.md`; nothing runs from it |
| 3.6 Two process changes | every brief to Codex before any worker starts; no partial receipts on the default branch | adopted; the waiting briefs carry a status line; partial receipts now go on a session's own branch or stay uncommitted (no new branch is created for them) |
| 3.7 One more week of arithmetic, then close | E20, 3.3 steps 1 and 2, 3.4; then a one-page close by aid removed | adopted as the plan; close date 17 November |

**The rulings, as now recorded.** Until this note, 9.60, 9.67 and 9.70 each carried the locality question forward as the owner's. Working rules from today:

1. *The life rule is a backward sweep over the item with weight signs in place of weights; the sweep length is a declared budget.* What stays outside the line is exact weight transport. The rule is then not what is new in this project; memory, retention and cost are where it has to earn its place.
2. *A bounded window is an allowed sensory buffer.* The defect found in 9.76 was a task that fitted inside it.
3. *References may use any optimizer. A candidate is born by its own rule before any promotion claim.* E18's B0 shows that is possible for single operations.
4. *Whose rulings are final is not mine to write.* Mr. Xu directs this session and his instruction is why these are in force; if Mr. Zhou rules otherwise, items 1 and 2 are the ones that would change, and the consequence is stated in the note (without the sweep the windowless learner has no working rule on file). I have asked Mr. Xu to confirm that these four are his and to say who has the last word.

**Where I depart from the note.** In one place: it suggests three backprop lives at zero replay against the cloud's rule lives; I run the rule again on the same machine beside them, so that the two arms share machine and panel, which is the lesson of 9.75. And its remark on hippocampal reverse replay is, as it says, from memory; it is not relied on.

**Earlier decisions this changes.** 9.76 held all rule variants until E20 reports. E18b (briefed as an exception) is put behind the every-byte lives, as the note asks. The every-byte lives themselves are a second exception, taken on instruction; they use options that already existed and the corrected rule.

**Addendum to 9.82 (10 November 2026, 19:15 EST): the rulings are Mr. Xu's and they stand.** Asked whether the three working rulings were his to set or Mr. Zhou's, Mr. Xu answered: "Mine, they stand." So, as rulings and no longer as my reading: (1) the life rule is a backward sweep over the item with weight signs in place of weights, the sweep length a declared budget, exact weight transport outside the line; (2) a bounded input window is an allowed sensory buffer; (3) references may use any optimizer, and a candidate is born by its own rule before any promotion claim. The phrase "the owner's to decide" in 9.60, 9.67, 9.70, 9.71 and 9.79 is closed for these three questions. Two consequences follow from the note and are now in force: the learning rule is not what is new in this project, and the fixed-rate two-bank design has no claim to stay if E20-B finds a windowless network that composes from memory.

## 9.77 E20-A, stage 1: the window reference composes with an operand in memory, at a cost (cloud session, 10 November 2026; `MOD97_LOCAL_RULE_20261106/exp45/E20A_TABLES.md`)

**What ran.** Stage 1 of E20-A version 2 only: three lives of the window learner's backprop reference (joint stream), seeds 73129 / 93683 / 116089, taught with 0 to 6 letters twice around each composition, inside the prompt or in front of it with equal probability. Predictions were committed before launch (0ea4ce45). I recomputed everything below from the three receipts: each has seven stages, a completion record, a weight hash that matches its file, a birth admitted by the manifest, and for seed 73129 the memory-panel hash of my own qualification record. The worker's verdicts match mine. All scores are whole worked answers with no store, at the end of life.

| | 73129 | 93683 | 116089 |
|---|---|---|---|
| Plus-minus twins, both members right (a recent-bytes responder scores 0 by construction) | 63 of 300, 21.0% | 80 of 300, 26.7% | 130 of 300, 43.3% |
| The same twins, single items right | 42.3% | 47.7% | 66.3% |
| The same items with the letters in front, everything in view | 78.3% | 70.0% | 80.7% |
| Comparison twins, both right | 63.7% | 58.7% | 56.0% |
| Operand hidden, new instances of taught operator pairs | 72.3% | 72.3% | 85.7% |
| Operand hidden, taught triples themselves | 82.7% | 78.3% | 84.3% |
| Twins with the recurrent state zeroed, single items | 0.0% | 0.0% | 0.2% |

| Prediction | Result | Verdict |
|---|---|---|
| P20.1 existence: both right on at least 20% of plus-minus twins in two seeds | 21.0 / 26.7 / 43.3 | holds, all three |
| P20.2 cost: single items at least 15 points below the in-view control in all three | 36.0 / 22.3 / 14.3 | **misses** (two of three) |
| P20.3 the control works: at least 70% on the in-view control in two seeds | 78.3 / 70.0 / 80.7 | holds, all three; one seed exactly on the mark (420 of 600) |

**What it says.**

1. *A learner on file composes from memory.* With the first operand outside its window, the window reference gets both members of a twin right 21–43% of the time, which no responder limited to the recent bytes can do at all. This answers the question of 9.76, for backprop on the window architecture: yes, partly.
2. *Hiding the operand costs 14 to 36 points,* against the same items at the same length with everything in view. The cost is real in every seed; my threshold of 15 was met in two.
3. *The harder step is the untaught operator pair, not the memory.* With the operand hidden the reference still answers 72–86% of new instances of operator pairs it was taught, and 42–66% of single items of the untaught plus-minus pairs.
4. *Teaching with letters costs the in-view task too:* the registered letter-free plus-minus panel ends at 69–81 whole answers, where the reference taught without letters ended at 85–89.

**Stage 2 goes ahead, by my decision,** since the control works: the sign-rule window learner, backprop on its pipeline, and the windowless two-bank reference, nine lives. One caveat is written now, before those results exist. The window learner's rule gives credit through time to the synapses of its memory cell only; the layers above the cell are credited within the step, which is the defect 9.67 found in the windowless learner, and the corrected rule cannot yet run on the window architecture. So if the rule arm misses P20.5, that miss cannot be separated from credit truncated above the cell; the comparison that can carry a conclusion about the pipeline is backprop on it (P20.6), and porting the sweep to the window architecture is the follow-up if the rule arm fails.

Evidence level E2 on panels held out by operand triple and operator pair; one life per seed; backprop is a reference here, not a candidate.

## 9.68 E16-M4, incomplete: stopped at five of six checkpoints; predictions not judged; the replay-share line closes here (10 November 2026, 20:00 EST)

**What happened.** The cloud session ran all twelve lives (share 0.75 on the six E16-F7 seeds; shares 0.875 and 0.9375 on the three E16-M3 seeds) beside E19 on one four-core container. Its container was restarted at about 23:50 UTC with every life inside its sixth and last stretch; the harness cannot resume a life; the session then had to be stopped. Its branch is merged here. The twelve receipts end at checkpoint 5 and are kept under their `_killed` names; there are no final weights. **P16M4.1 to P16M4.5 are all written for the last checkpoint and are not judged.** By my decision the job is not rerun: the next-steps note adopted in 9.82 ends the replay-share line with this job, and the cost-curve run of its section 3.3 replaces it.

**What the five checkpoints show.** Everything below is at checkpoint 5, five sixths of the life, against the comparator lives at the same checkpoint. It is description, not a verdict, and the reading of 9.75 item 3 applies: a higher share is also fewer fresh items.

Share 0.75 on the six seeds it was not found on (plus-minus pairs, final answer):

| Seed | Share 0.5 | Share 0.75 | Reference | Ratio at 0.5 | Ratio at 0.75 |
|---|---:|---:|---:|---:|---:|
| 1100009 | 62.2 | 71.1 | 89.7 | 0.69 | 0.79 |
| 1200007 | 45.2 | 69.5 | 89.9 | 0.50 | 0.77 |
| 1300021 | 69.6 | 76.4 | 88.9 | 0.78 | 0.86 |
| 1400017 | 71.8 | 76.5 | 87.4 | 0.82 | 0.88 |
| 1500007 | 71.3 | 70.2 | 90.2 | 0.79 | 0.78 |
| 1600033 | 64.6 | 75.1 | 81.6 | 0.79 | 0.92 |

- The hard pairs are higher at 0.75 in five of six seeds, by 5 points or more in four. Three of six are at or above 0.80 of the reference, against one at share 0.5. At this checkpoint all six pass the family gate at share 0.75; at share 0.5 two do not (1200007 on held pairs, 0.73; 1600033 on mod-97, 0.79).
- In the two seeds whose `+-` fell late at share 0.5, the fall from checkpoint 4 to 5 was 29.2 and 21.2 points; at 0.75 it is 7.3 and 8.5.

The curve beyond 0.75, three seeds (plus-minus pairs; mod-73 mean):

| Seed | 0.5 | 0.75 | 0.875 | 0.9375 |
|---|---|---|---|---|
| 400009 | 51.5; 89.7 | 69.8; 87.8 | 70.1; 82.4 | 72.2; 82.1 |
| 600013 | 46.2; 80.8 | 56.9; 80.4 | 69.2; 77.4 | 73.9; 78.3 |
| 300007 | 68.8; 89.9 | 75.4; 89.8 | 56.5; 87.5 | 79.8; 88.0 |

- The hard pairs are still rising at 0.9375 in all three seeds, with one unexplained dip at 0.875 (300007). The family being taught at that point, mod-73, is 2 to 6 points lower at the two highest shares: with four fresh rows in a batch of 64 it is acquired more slowly, but at checkpoint 5 no seed has fallen under the family gate. I predicted the curve would turn by 0.9375; at this checkpoint it has not.

**What it adds.** The direction E16-M3 found on three seeds appears on six seeds it was not found on, at one checkpoint short of the end. It does not settle a default share, and nothing is claimed about the end of the life.

**What the loss teaches.** A life that cannot resume should not run on a machine that can restart. Fifteen processes on four cores also stretched the job to six hours, which is what put it in harm's way. Making lives resumable (weights and optimizer state at each checkpoint, the data stream replayed without training up to it, and a test that an interrupted life matches an uninterrupted one exactly) comes before any further life job on a cloud container.

## 9.86 Codex's memo on E21, the interrupted E16-M4 and progress: what holds and what is changed (10 November 2026, 22:00 EST)

**The memo.** `PROJECT_DOCUMENTS/MINIFLY_M4_E21_CONTINUATION_AUDIT_MEMO_20261110.md` (Codex, source and receipts, nothing run). Opus workers received it too. I ran the claims that can be run; all hold.

| Memo point | After running it | Done |
|---|---|---|
| E16-M4 cannot be resumed: its harness saved no weights or optimizer state before the end; a completion would be a fresh full run | agreed; nothing to run | 9.68 stands: closed as incomplete, not reopened. If it is ever reopened it is a new execution of the same twelve cells, counted once per cell, and E21 does not help it |
| E21 needs six acceptance provisions | all six accepted | written into the brief as governing amendments; the brief is now marked audited |
| The exposure ledger files compositions that start with letters under "other" | confirmed: 96,142 such rows at stage 3 of the seed-73129 stage 1 life | the family is now read from the first byte that is not a distractor letter, in the ledger and in the reservoir census |
| The update-CPU field includes the scoring before the first block | confirmed: 718 of "updates" at stage 1 included about 36 of scoring | initial scoring is counted as scoring |
| The second-operand census reads the wrong gap | confirmed: it counted letters after the first operator (26 of 150 plus-minus twins); the second operand is fully out of view in 31 and partly in 27 | corrected census added beside the old field; the panel items and the panel hash are unchanged |
| The E16 summarizer still gives a ratio when panel hashes differ | confirmed, and it was my own exception | removed: differing hashes never get a ratio |
| The worker's E20 reducer accepts a truthy completion field | accepted | stage 2 will be filed from my own recomputation, which checks stages, settings, panel hash and the weight file's hash, as stage 1 was |
| Target-test draft: the teaching graph must identify held outcomes; separate held pairs per phase; match the wider-view arm | accepted | added to the draft as admission conditions for its step 0 |

Assurance after the fixes: the qualifier and the assay check exit 0; the memory-panel hash is the one recorded for stage 1; two dry lives confirm the training path is unchanged (results in `exp45/archive_audit3_dry/`).

**Errata to 9.77.**

1. *"The harder step is the untaught operator pair, not the memory" is withdrawn.* The two panels behind it differ in more than that: the hidden-operand panel of taught pairs mixes twelve operator pairs, the twins are the plus-minus pairs only, and letters inside a prompt change more than whether the first operand is in view. The gaps between those scores do not isolate which component is hardest.
2. *The caveat on stage 2's rule arm had its polarity reversed.* P20.5 predicts that the sign-rule learner is at or below 0.60 of the reference, so a low score makes P20.5 hold, not miss. The caveat should read: if the rule arm is low, which P20.5 predicts, that cannot be separated from credit truncated above its memory cell; and whatever it scores says nothing about the backward-sweep rule on the window architecture, which has not been implemented there.

## 9.87 Codex's audit of E22 and of the E20-B and E18b dispatch: E22 withdrawn and rewritten (10 November 2026, 23:00 EST)

**The audit.** `PROJECT_DOCUMENTS/MINIFLY_E22_DISPATCH_HOSTILE_AUDIT_20261110.md` (Codex, nothing run). I ran what can be run. It holds throughout.

| Audit point | After checking | Done |
|---|---|---|
| E22's parity gate passes when both learners collapse together; the family mean hides addition and subtraction; "within 5" is undefined | accepted | acquisition, absolute final competence and one-sided parity are separate verdicts with literal definitions (P22.1–3) |
| One birth arm per cloud instance confounds host with birth | confirmed from the worker's own note: runs are not bit-identical across machines | version 2 runs all six lives on one host |
| A birth hash shows identity, not training history | confirmed: the manifest held a hash and nothing else | the manifest now records each birth's training configuration and final step, and a life checks them against what it asks for; a rule-born file presented as a backprop birth stops the life, as does a birth at another dose |
| "No backprop anywhere" misnames the rule | accepted | it is a sign-feedback temporal sweep, the same in birth and life |
| Sizing needs a crash policy | accepted | written into the three briefs: keep the partial attempt, stop the cell, report; no restart without authorization |
| E20-B: parity test covers two checkpoints only; networks need their own checks; G3 and G0 on one host; own file names; a collective miss does not show the teaching is the cause; the premise ignores E20-A stage 1 | accepted | seven amendments in the brief |
| E18b: a max/min mark of 30 is below a copy shortcut; partners need the same host | **confirmed by measurement**: copying one operand scores 49–52% on the sealed max and min panels | P18b.2 replaced (at least 66 on max/min and at least 10 on the hard mean); partners rerun on the same host |

**Erratum that reaches beyond E18b: max and min are to be read against 50, not against zero.** On every single-operation panel, answering with one of the two operands is right about half the time. Scores I have quoted without that baseline include the one-bank network's births (max 53–61, min 61), the first bank's own head in E18 (max and min 55–60) and B3's 1–3. The first two are at the level of the shortcut and show no learned operation; the third is far below it, which means malformed answers. Family means average four operations, so a learner that only copies an operand already has about 25 points of a family mean. The hard mean (addition and subtraction, shortcuts of 1–3%) is the figure to trust, as the births have been reported since E18; the family gate of the E16 line is on means that include max and min, and that is one more reason its hard-pair and hard-operation counts are the ones to quote.

**What I take from it.** E22 version 1 took me about twenty minutes, after I had written, the same day, that every defect the audits found was in something I shipped within the hour. The process rule adopted today worked as intended: the brief was audited before any worker started.

## 9.88 The two runs from the next-steps note: backprop also loses everything without replay; the working rule does not survive every-byte targets (10 November 2026, 17:12–19:51 EST)

**What ran.** Twelve lives on the owner's machine, predictions on file before launch (P-Z.1–3, P-E.1–4). All twelve exited with code zero; every receipt has seven stages at the full dose, a completion record, a weight hash that matches its file and a birth admitted by the manifest; each pair of arms shares one panel hash.

**Zero replay, window learner, drifting stream, seeds 400009 / 600013 / 300007.** The two arms differ in the update alone.

| End of life | Sign rule | Backprop on the same pipeline |
|---|---|---|
| mod-97 mean (ratio to the reference on file) | 31.9 / 31.1 / 11.1 (0.33 / 0.33 / 0.12) | 43.0 / 34.9 / 27.6 (0.45 / 0.37 / 0.30) |
| Plus-minus pairs, final answer | 1.6 / 0.9 / 1.5 | 1.0 / 1.0 / 0.8 |
| Plus-minus pairs at their best (checkpoints 3–4) | 61–85 | 91–92 |
| mod-73 mean (the family being taught at the end) | 89.3 / 83.2 / 89.3 | 96.1 / 93.6 / 94.1 |

| Prediction | Verdict |
|---|---|
| P-Z.1 the rule ends under 0.50 on mod-97 and at or under 5 on the plus-minus pairs | holds, all three |
| P-Z.2 backprop ends under 0.60 on mod-97 and at or under 10 on the plus-minus pairs | holds, all three |
| P-Z.3 backprop's mod-97 mean is at least 5 above the rule's in two seeds | holds (+11.1 / +3.8 / +16.6) |

*Reading.* Losing everything without replay belongs to this stream under either update. Backprop learns the plus-minus pairs to 91–92 and ends at 1, exactly as the rule does. The sentence in 9.64 item 4, "under the sign rule this learner has no retention of its own", was true and was not a statement about the rule: nothing on file has retention of its own. Backprop keeps somewhat more of the mod-97 operations (4 to 17 points). One thing in the curves is not understood and is noted without a reading: in both arms the mod-97 score is 0 to 1 at the checkpoint where only compositions are being taught and partly returns afterwards.

**Every next byte as the target, windowless two-bank learner, seeds 73129 / 93683 / 116089.** Each arm differs from its Phase 4 partner in the loss mask alone, and the two arms differ from each other in the update alone.

| | Sign-feedback sweep, every byte | Its answer-loss partner | Backprop on the pipeline, every byte | Its answer-loss partner |
|---|---|---|---|---|
| mod-97 mean at the end | 10.3 / 5.4 / 9.7 | 87.9 / 86.1 / 86.9 | 89.3 / 90.6 / 89.2 | 90.9 / 92.3 / 94.2 |
| Hard mean (add, sub) at the end | 1.4 / 1.2 / 0.9 | | 88.8 / 91.1 / 85.2 | |
| mod-73 mean at the end | 11.9 / 7.7 / 11.9 | 90.8 / 86.9 / 89.4 | 89.2 / 90.8 / 88.1 | 93.5 / 93.2 / 93.2 |
| mod-97 mean after blocks 1 to 6, seed 73129 | 84, 54, 26, 18, 6, 10 | | 91, 90, 86, 88, 89, 89 | |

| Prediction | Verdict |
|---|---|
| P-E.1 backprop tolerates every-byte targets (at least 0.90 of its partner on both families, two seeds) | holds, all three (0.95–0.98) |
| P-E.2 the rule ends within 10 points of its partner on mod-97 in two seeds | **misses, all three** (76 to 81 points below) |
| P-E.3 the rule at or above 0.85 of backprop on mod-97 in two seeds | **misses, all three** (0.06–0.12) |

*Reading, as written beforehand.* The objective is fine and the rule is not: with the stream's own next byte as the only target, backprop on the same pipeline keeps the skill within 2 to 5 points, and the sign-feedback sweep acquires it in the first block (82–84) and then loses all of it, the unrehearsed birth operations included (0–3). This is the pattern E16-L showed on 8 November under the older rule; carrying credit through time at every layer did not cure it. It bears directly on Mr. Zhou's delayed-observation idea: in its plainest form it works under backprop and fails under the working life rule. The single next change named beforehand is predictable distractors.

*A hypothesis, not a finding.* Five sixths of the bytes under this loss are prompt bytes that cannot be predicted. Under the true gradient their error averages to nothing once the learner predicts their frequencies. Under weight-sign feedback the relayed error is not the gradient, so the same irreducible error need not average out and may push the weights steadily. This can be checked without training, at a backprop solution: the mean update over many batches should be near zero for backprop and need not be for the rule. I have not run that check.

**Consequences.** E18b (births under every-byte targets with local heads) stays on hold; its question now reads differently, and the check above comes before it. The retention work of the next-steps note keeps its order: with both updates at the same floor without replay, a protection mechanism is tested under backprop first.

**Cost.** CPU-seconds per life: 6,251–6,313 and 5,503–5,534 (zero replay, rule and backprop); 7,759–7,784 and 7,598–7,755 (every byte). Wall 2 h 38 min for twelve processes against a forecast of 2 h 30 (range 2 h 10 to 3 h). Evidence level E2; one life per seed and arm.

## 9.89 Three small things while E22 runs: a hypothesis that missed, four read-only jobs received, a draft for step 0 (11 November 2026, 00:45 EST)

**1. The hypothesis of 9.88 is not supported.** I had suggested that the working rule collapses under every-byte targets because weight-sign feedback turns the irreducible error of unpredictable prompt bytes into a steady push that the true gradient averages away. The check (`e18c_drift_check.py`, read-only, its prediction committed at 83b6772e before the first run) measured, at the three backprop every-byte solutions, how coherent the prompt-byte part of the gradient is across 128 batches, for backprop and for the sweep. Prediction: the sweep at least twice as coherent as backprop below the second bank's write, in two of three seeds.

| Coherence of the prompt-byte part, below the second bank's write | 73129 / 93683 / 116089 |
|---|---|
| Backprop | 0.34 / 0.31 / 0.48 |
| Sign-feedback sweep | 0.30 / 0.27 / 0.56 |
| Ratio | 0.89 / 0.89 / 1.18 |

**The prediction misses.** The sweep is no more coherent than backprop. One weakness of the measurement is stated with it: backprop's own coherence is 0.3 to 0.5, far above the 0.09 that pure noise would give, so these weights are not at rest under the batch mix I drew, and the check could not see a push appear against a zero background. As registered, it does not support the hypothesis, and the cause of the collapse is open. The next single change remains the one the next-steps note named beforehand: predictable distractors.

**2. Four read-only jobs arrived from one cloud session** (merged at 26a86602; all four are that worker's reading and are audited like any deliverable). I have read two of them so far.

*The evidence ledger* (`PROJECT_DOCUMENTS/EVIDENCE_LEDGER_20261111.md`, with `ledger_recompute.py`) recomputed every number in 9.59–9.82 and lists nineteen disagreements with this review. Fifteen are rounding. Four change a sentence, and these are corrected here:
- *9.66 item 3,* "every arm sits at 8–13 on the held pairs": the arm with only the embeddings frozen is at 1.0–2.0.
- *9.67 item 3,* "`freeze: none`, `frozen: []` in every receipt": only 6 of the 18 drift receipts carry both fields; the older ones predate them. No receipt records any freezing, and the window learner's code has no freezing path, but the sentence as written was false.
- *9.79 item 2,* "with the state zeroed its held compositions fall to zero": the plus-minus pairs fall to 0.0; across all four pairs whole answers reach 6.0% and final answers 28.2%.
- *9.79 item 3,* "the registered mod-97 panel is 94–98% items whose relation is supervised somewhere": that is addition, max and min; subtraction is 72.4%.
Also 9.71 item 4: the local-heads package ends 10.5 / 12.3 / 13.2 points under the reference, not "about 10".

*The storage audit* (`PROJECT_DOCUMENTS/REPO_STORAGE_AUDIT_20261111.md`): the repository is 10.04 GiB packed and a full clone with its checkout is about 21 GiB, which is more than a cloud container has free; 99.4% of it is at the head of the default branch, so history is not the cause; zip bundles from September and October are 4.31 GiB, model weights 0.9 GiB; partial-receipt commits were 39% of commits since 1 November. A tested shallow sparse clone of the working folder without weights is 275 MiB. That option needs no decision and is now written up for workers (`WORKER_CLONE_INSTRUCTIONS.md`). Its other options (commit weights only at completion; move weights out of git; rewrite history) are Mr. Xu's and Mr. Zhou's to choose.

*Not yet read by me:* the multi-timescale literature check and the source check for the delayed-observation memo (`PROJECT_DOCUMENTS/LIT_CHECK_20261110/`).

**3. Step 0 of the target test is drafted for Codex's audit** (`OPUS_TASK_TARGET_TEST_STEP0_20261111.md`): the stream made concrete with proposed sizes, eight checks that each must pass and each must be shown able to fail, and six questions for the audit, one of them whether to generate a predictable-filler version alongside after the every-byte result. No worker starts from it.

## 9.85 E22 version 2: a learner born and living under one rule ends where the backprop-born one does (10 November 2026, 19:52–21:14 EST; `MOD97_LOCAL_RULE_20261106/exp45/E22_VALIDATION.json`)

**What ran.** Six lives of the windowless two-bank learner on the owner's machine, on Mr. Xu's go, seeds 73129 / 93683 / 116089, under the working life rule (a sign-feedback temporal sweep over the item; the longest item is 22 bytes): three started from E18's rule-born births and three, their partners, from the backprop births. Version 2 of the brief, rewritten after Codex's audit of version 1 (9.87); version 2 itself was not audited before running, by Mr. Xu's choice. Predictions and the validator were committed before launch (c6732624).

**Validation, by the script written beforehand.** All six exit codes zero; source files and birth manifest unchanged between launch and scoring; every receipt has stages 0 to 6 at the full dose, a completion record and a matching weight hash; each birth is the pinned file, admitted with its training configuration checked; partners differ in no setting except the birth fields and share one panel hash. No cell was excluded. The backprop-born repeats reproduce the Phase 4 receipts of 9.70 to the digit, which also shows the lives are deterministic on this machine.

| Twelve distractors, whole answers | 73129 | 93683 | 116089 |
|---|---|---|---|
| Rule-born, after block 1: mod-97 mean; hard mean (add, sub) | 90.9; 85.0 | 91.9; 87.2 | 88.7; 82.0 |
| Rule-born, end: mod-97; hard mean; mod-73 | 88.1; 84.5; 90.7 | 84.8; 77.0; 87.5 | 84.3; 76.4; 87.1 |
| Backprop-born partner, end: mod-97; hard mean; mod-73 | 87.9; 82.1; 90.8 | 86.1; 79.2; 86.9 | 86.9; 81.8; 89.4 |
| Partner minus rule-born: mod-97; mod-73 | −0.2; +0.1 | +1.2; −0.6 | +2.6; +2.2 |
| Development operations, never shown in life, at the end: rule-born; partner | 34; 11 | 36; 22 | 29; 43 |

| Prediction | Verdict |
|---|---|
| P22.1 acquisition: after block 1, mod-97 at least 80 and hard mean at least 70, two seeds | holds, all three |
| P22.2 final competence, absolute: mod-97 at least 75, hard mean at least 65, mod-73 at least 75, same seed, two seeds | holds, all three |
| P22.3 parity, one-sided: no more than 5 below the partner on mod-97 and on mod-73, same two seeds | holds, all three (largest gap 2.6) |

In no seed are both partners under the marks of P22.2, so this is not parity after common loss.

**What it says.** For single operations in this network, the birth-lineage condition of Mr. Xu's third ruling can be met: one rule at birth and in life, no backprop at any point, ending within 3 points of the backprop-born learner and above an absolute mark, in every seed. The births differ in level as well as in history, so this is a statement about two birth packages, and three seed pairs describe it; they do not establish equivalence.

**What it does not say.** The learner is answer-supervised and replay-supported; it loses its unrehearsed birth operations however it was born (29–36 against 11–43 at the end, from 90–95); it does not compose (held pairs 13–15 in all six lives); beyond the trained gap it falls (at 40 distractors the mod-97 mean, which includes the max and min shortcut, is 34–43 for the rule-born and 32–49 for the partner); and under every-byte targets the same rule collapses (9.88). The promotion question is separate and untouched.

**Cost.** About 4,400 CPU-s of updates and 300 of scoring per life; six processes, 1 h 22 min of wall against a forecast of 1 h 30 (range 1 h 15 to 1 h 50). The rule-born births cost 4,307–4,321 CPU-s each on a cloud container (9.71). Evidence level E2.


## 9.74 Literature check on learning and memory at several timescales, and the answer to Mr. Zhou's question (cloud session's memo; filed 10 November 2026, 21:45 EST by the clock)

Memo: `PROJECT_DOCUMENTS/LIT_CHECK_20261110/LITERATURE_CHECK_MULTI_TIMESCALE.md`. The worker's own paragraph follows unchanged; my reading is after it.

This was a read-only job. About 25 primary sources were opened in the session, and the central numbers were checked against their raw text. It answers Mr. Zhou's question in three parts.

- **The Hope paper is reading C, not oscillation.**
  - Its "frequencies" are how often blocks of MLP weights are updated (Nested Learning, Sec. 7.1, Eqs. 70–71).
  - Everything is trained by backprop (AdamW).
  - The brain-wave hierarchy is cited as inspiration for ordering the levels; it is not tested.
  - Its continual-learning gains appear only in a figure, on 3B–8B Llama backbones, against ICL, EWC and InCA. There is no replay baseline, no seed count, and no independent replication was found.
- **Brain rhythms do not hold information for 120 s.** They run at about 0.5–150 Hz, and cortical intrinsic timescales are about 50–350 ms. Over seconds to minutes, the documented carriers are replay, ongoing neuromodulator activity and synaptic processes at several rates. In the fly, slow calcium oscillations in the MP1/MV1 dopamine neurons, about 0.05–0.3 Hz, gate long-term memory after spaced training (Plaçais et al. 2012).
- **The five claims.**
  1. Paper: confirmed in outline, corrected in detail.
  2. Reading A: confirmed. The choice of rates matters, and in one controlled ablation (S4D) real decays lost 5–10 points to complex ones.
  3. Reading C: confirmed, narrowed.
     - Benna–Fusi chains (Kaplanis et al. 2018, 2 RL tasks, 3 runs) and a metaplastic hidden weight (Laborieux et al., six permuted MNIST tasks, about 97% against 8–13%, matching EWC without task boundaries) buy retention without rehearsal. Both are synapse-local and need no importance estimate.
     - Nothing was tested on sequences, against a replay-share curve, or with sign-based credit.
  4. Cost of non-element-wise recurrence: confirmed for forward-only credit. Under the backward-sweep rule of the 9.82 addendum, an inner loop costs K times the work per byte instead.
  5. Fly parallel memories: confirmed (9.72). Oscillation's role: narrowed. Theta–gamma ordering is a supported hypothesis, and no function is shown for the fly's 10 Hz mushroom-body oscillation.

Looped models match deeper ones at equal compute (Saunshi et al. 2025), so what an inner loop buys is parameters. No controlled test was found of oscillating units against plain decays for holding ordered items.

**Suggestions, not decisions.**
- A three-variable Benna–Fusi chain per synapse in the window learner, tested at replay shares 0–0.5 against learning-rate-decay and random-consolidation controls. Cost: 2 floats per synapse.
- A shared 512×512 loop run K times per byte in the two-bank learner, against unshared layers at matched compute. Cost: about 262K weights and K× work per byte.
- Complex retentions only if a failure is shown to be about order.

The memo decides nothing.

**My reading (coordinator).** I read the memo's summary, its verdict table and its suggestions. I did not reopen its sources.

*The answer to Mr. Zhou.* Yes, people have worked on it, and under one word they have worked on three different things:

- A. State held at several decay rates, so that something written long ago is still there when something new arrives.
- B. Units that oscillate.
- C. Weights that are updated or consolidated at several rates.

The Hope paper is C, trained by backprop; its brain waves are a motivation, not a mechanism it tests. In the nervous systems the memo covers, information from two minutes ago meets information from now through synapses that change at several rates, through replay and through neuromodulators. The rhythms themselves are too fast to carry it.

*Where this project stands on the three.*

- A is built. The two-bank learner holds its state at five fixed retention rates, 0.5 to 0.999 (`arch17.py`, `RATES`).
- B has no controlled evidence in its favour for holding items in order. It is not on our list and I do not add it.
- C is absent from this line (E16 to E22): no weight in any of its learners has a slow companion. I have not checked the older MiniFly work for it.

*Decisions.*

1. C is the candidate for the protection-mechanism step of the next-steps note (9.82). The order of that note stays: the cost curve of retention against replay share comes first, and it waits for the resumable-lives branch (E21) to be merged. The reason is 9.88: at zero replay backprop loses everything too, so a protection mechanism has to be read against the replay-share curve under both updates, or a gain cannot be told from "a little replay was enough". The memo's first suggestion already has that shape (replay shares 0 to 0.5, two controls). It is not designed tonight.
2. The inner loop (second suggestion) is not adopted now. The network decision waits for E20-B (9.76), and the memo's own finding is that a loop at equal compute buys parameters and nothing else that has been shown.
3. Complex retention rates are not adopted. No failure on file has been shown to be about order.
4. No expectation is borrowed. Nothing the memo found was tested on sequences, against a replay-share curve, or under sign-based credit. A prediction for any of these would be ours alone.

Literature only; no result on file changes. The memo decides nothing and neither does this section beyond the four points above.


## 9.84 Source check of Codex's delayed-observation memo, the reverse-replay remark and sign feedback through time (cloud session's memo; filed 10 November 2026, 21:45 EST by the clock)

Memo: `PROJECT_DOCUMENTS/LIT_CHECK_20261110/SOURCE_CHECK_DELAYED_OBSERVATION.md`. The worker's own paragraph follows unchanged; my reading is after it.

This was a read-only job. Every source cited was opened in the session.

**The memo's seven precedents.** All seven say what the memo says they say. Four need a qualification:
- CLAPP's "later views" are, in its image experiment, successive patches of a static image. Its fully local form was run on images only: 73.6% against CPC's 81.1%.
- GIM's own context module uses backprop through time.
- P-TNCN's 1.70 against 1.38 BPC comes from an arXiv-only paper that reports no seeds.
- Fly sensory preconditioning appeared in behaviour only after 10 near-simultaneous pairings at a 1-s interval. One pairing, or a 30-s interval, did not produce it, and a reviewer called the behavioural effect weak.

**The reverse-replay remark (3.1).** It holds for awake replay at the end of a run, which is compressed into hundreds of milliseconds and whose rate tracks reward in two labs. "Recent" is contested: one study found replay of paths not run for minutes, and of paths never run. No study opened disrupts reverse replay alone. Its role in credit assignment is proposed (Foster & Wilson 2006), modelled (Mattar & Daw 2018) and correlated with learning in humans (Liu et al. 2021), not shown causally. What the sources describe running backward is reactivated state activity, or value in the models; none describes an error signal.

**The sign-feedback finding of 9.72 now reads more narrowly.** No study was found with sign feedback through recurrent weights or in a sequence model. One 2025–2026 preprint (Goupy et al.) applies unscaled sign symmetry in spiking ResNets trained by backprop through time over 4 steps. It falls to chance there (CIFAR-10: 10.0–10.8% against 87–92%; 8 seeds), but comes within about 1–3 points of backprop in the same paper's event-driven networks without time steps. A 2025 sign-symmetry preprint has since been withdrawn as making false claims.

**Prediction of later observations under local rules.** Among the results opened, only Halvagal & Zenke's latent predictive learning reports a control that destroys temporal contiguity. On images it reduces the rule to the level of a purely Hebbian one. None uses the target shuffle within delay and frequency strata that the memo proposes.

The memo decides nothing.

**My reading (coordinator).** I read the memo's summary paragraph above and did not reopen its sources.

1. *Codex's memo.* Its seven precedents say what it says they say. Two of the four qualifications matter for us: the cited greedy model uses backprop through time in its own context module, and the 1.70 against 1.38 comparison comes from a paper with no seeds. Neither changes a decision. The delayed-observation line (E18b) is on hold for a reason of our own: under every-byte targets the working rule collapses (9.88) and the cause is still open (9.89).
2. *A requirement recorded for that line.* If a delayed-observation run is designed again, it carries a control that destroys temporal contiguity, in the form Codex's memo proposes (targets shuffled within delay and frequency strata). Among the published results the worker opened, only one has such a control at all.
3. *Reverse replay is not support for the backward sweep.* What the sources describe running backward is reactivated state, or value in the models, and no study shows it causally assigning credit. The life rule's backward sweep over the item rests on Mr. Xu's working ruling (addendum to 9.82) and on nothing else. No section of this review cites reverse replay for it, and none should.
4. *Sign feedback through time: our result stands alone.* Phase 4 (9.70) and E22 (9.85) carry sign feedback through time at every layer. The worker found no published study of that in a recurrent or sequence model, and one preprint in which unscaled sign feedback through four time steps of a spiking network falls to chance. Ours differs in three ways at once: the feedback is scaled by the layer's mean absolute weight (`exp3_state.py`, `SignLinear`), the units are not spiking, and the sweep covers one item. Which of the three matters is not known. The independent hand-written check (`e17p4_independent_check.py`) shows that the rule is what we say it is; it does not show that it holds elsewhere. Both results are three seeds, one task family, one network, and should be described as ours and unreplicated. A scaled-against-unscaled comparison would be one factor and cheap in compute, but the scaling sits in frozen shared code, so it would need its own switch and its own check. It is not queued.

Literature only; no result on file changes.

**Erratum to the headings filed on 10 November (9.68, 9.75 to 9.89): time-of-day labels.** Where a heading in those sections gives a single time of day rather than a run's interval, I wrote it without reading the clock, and it runs ahead of the commit by up to four hours (for example 9.87 says 23:00 EST and was committed at 19:41; 9.89 says 11 November 00:45 EST and was committed on 10 November at 21:03). The same holds for the addenda to `PROGRAM_STATE_20261107.md` labelled 20:30 EST on 10 November through 01:15 EST on 11 November. Run intervals (9.81, 9.85, 9.88) were read from logs and are right. For everything else the commit times govern. From this section on I read the clock.


## 9.90 Codex's audit of the E20-B amendments: five findings run and confirmed, the brief amended a second time, one hole closed in our own birth admission (10 November 2026, 21:50 EST by the clock)

Audit: `PROJECT_DOCUMENTS/MINIFLY_E20B_AMENDMENTS_AUDIT_20261110.md` (commit 522174bb). Its disposition: go ahead with implementation and a reduced-dose qualification; hold the full dose until the qualification record exists and the rule for choosing a network is fixed. I agree with the disposition and adopt it.

**How I checked it.** Each of its five findings about the inherited source is a statement about what the code does, so I ran each one (`MOD97_LOCAL_RULE_20261106/e20b_audit_checks.py`, output in `exp45/E20B_AUDIT_CHECKS.json`; it trains nothing). All five hold.

| Finding | What the run shows | What changed |
|---|---|---|
| 1. A bare `exp16_integrated.py D <seed>` is not the two-bank reference | Confirmed. With no variables set the runner is the window network with embedding decay on. My amendment 2 named that bare command as the comparator. | Amendment 9: the two exact commands, in a clean environment; parity in two parts, ending in equal weights hashes. |
| 2. Two scoring paths; a `taught` key breaks the shortcut helper | Confirmed. The birth scorer generates at most 4 bytes and a worked answer has 6, so it can never score one whole. Through the six-byte scorer the three fixtures behave as required (64/64 whole and final; 0/64 whole with 64/64 final; 0/64 on both without the end symbol). A `taught` key stops `shortcut_ceilings` with a value error. | Amendment 10. |
| 3. Dispatch at import; embeddings found by name | Confirmed. An unknown architecture name stops at import. A table named `book` gets decay 0.0; the same table under another name gets 0.3 without a word. | Amendment 11, with the causality, row-independence, reset and gradient checks the audit lists. |
| 4. Admission does not look at the seed | Confirmed by fixture: a complete receipt and its weights, copied under another seed's name, were admitted. | Fixed in `admit_birth` (two lines); the same fixture is now refused. Amendment 12 for the worker's own runner. |
| 5. The predictions do not choose a network | Confirmed by reading them: P20.8 and P20.9 can hold for different networks, or for one network in different seeds. | Amendment 13: a selection rule fixed before any result. |

**The hole in admission was never used.** 160 life receipts on file record their birth's hash. They name 27 distinct births, and none is recorded under two seeds. The 33 lives run since admission existed were all admitted through the manifest, which is keyed by file name and was not affected; none went through the receipt path that had the hole. The fix is in `admit_birth` only. I have not touched `life()` or `develop17()`, as promised until the resumable-lives branch (E21) is merged.

**Where I go beyond the audit, and why.**

- *Birth parity as well as life parity.* The worker writes its own birth loop too. Amendment 9 asks that it reproduce the reference birth's weights hash before the life is compared.
- *G3 first.* The audit asks for one gated network to meet both marks in the same two seeds, and a tie rule. I add that if the two-bank network with the loop (G3) qualifies, it is the candidate even when a gated network qualifies too. The next-steps note (3.2) proposed dropping the banks because that design does not compose under backprop. If G3 composes, that reason no longer applies to it, and G3 is one matrix away from the network on which the working rule already trains and is born (9.70, 9.85). A gated cell under the sign sweep is untried ground. This is my decision and it is written into the brief before any result; Codex can contest it with the stage 1 record.
- *Work in progress goes to the worker's own branch.* Containers restarted twice this week, so workers do need to push often. They push to their branch; the default branch gets one commit at completion.

**What the dry run of my own amendment found.** Before sending amendment 9, I ran its two reference commands at a small dose on the Mac, one process under the lock. The first attempt stopped: with a birth dose of 8 updates the birth loop never completes a block and ends in an error, so the life found no birth. The dose must be a multiple of 512, and the amendment now says so. The second attempt (one block of birth, six blocks of four updates of life, 21:46 to 21:50 EST) ran both commands to exit code zero: seven stages, a completion record, and the birth admitted through its own receipt with the new seed check in force. Files are in `exp45/archive_e20b_ref_dry/`, with a note of both attempts. Had I sent the amendment as first written, the worker's first reference run could have stopped on my instruction.

**Sentences withdrawn.** Two sentences of the E20-B brief attributed a miss by all three networks to the teaching. Amendment 6 already narrowed that; amendment 14 now names the two sentences. The same attribution appears in the next-steps note (3.2, last bullet but one); a dated note is appended to our copy of it.

**Limits adopted from the audit's reading (amendment 15).** A G3 gain over G0 is a gain from 32,768 added weights as a package, not evidence about where feedback is placed. "Windowless" describes the network's input: the registered decoder keeps the prompt and reruns the network for every output byte, so nothing here measures single-pass inference. The result will be a development feasibility result at E2.

**Status.** The brief is ready for a worker for stage 1 only (implementation and reduced-dose qualification). Stage 2 waits for Codex's reading of the stage 1 record and my go.

**Erratum to 9.74 and 9.84.** Their headings say "21:45 EST by the clock". The clock read 21:34 when I wrote them and the commit is at 21:36. I wrote the label from an estimate in the very sections that promised to stop doing so. From here on the label is pasted from the `date` command's output.


## 9.91 Codex's progress survey and next experiments: its two corrections to step 0 simulated and adopted, its sequence adopted with one ordering of my own (10 November 2026, 22:10 EST by the clock)

Memo: `PROJECT_DOCUMENTS/MINIFLY_PROGRESS_SURVEY_AND_TARGET_ASSAY_20261110.md` (commit 0f44505f). It surveys what the project has established, corrects the step 0 draft of the target test in two places, and recommends four next experiments.

**The survey.** I checked the figures it quotes from this review's own period against the sections that hold them, and they agree: E18's rule-born births at 0.94 to 0.96 of the backprop birth (9.71); E22's three verdicts and the unrehearsed operations at 29 to 36 (9.85); E19 (closed in 9.79); E20-A's 21 to 43% of twins (9.77); 2 of 15 later seeds on the hard pairs (9.81); backprop near 1% on the plus-minus pairs without replay (9.88); the 90 to 93% held-pair mean of E7. The older figures (Full151, V216, V218) I did not recheck. Its account of the E20-B amendments is accurate.

**Its two corrections to step 0, both simulated** (`PROJECT_DOCUMENTS/target_step0_draft_checks.py`, output `TARGET_STEP0_DRAFT_CHECKS.json`; standard library, no learner).

| Correction | What the simulation shows | What changed in the draft |
|---|---|---|
| The split leaves no pairs for sealed twins | Confirmed by counting: 216 taught and 72 revealed are all 288 pairs of a family. My draft asked for 150 twins per phase whose pairs "never occur in the stream", from a pool of none. | 216 taught, 48 first-encounter, 24 sealed (12 twin pairs) per family, reserved before generation. Results reported over renderings and over distinct pairs. |
| The visibility check cannot fail the way I wrote it | Confirmed. A table memorising forty-byte suffixes scores 0 of 300 sealed twins in three worlds, as does an exact context-and-cue table: the mapping is absent, not out of view. The rule-aware evaluator scores 300 of 300 with the whole episode and 0 with the shared suffix only. | Check 3 now uses the rule-aware evaluator under two inputs. The tables are separate memorisation controls (3b). |

**Its question about the control, answered.** The memo asks whether the control with no prior teaching can still infer the answer through the other family. It can. With shared context values, removing every family-1 teaching episode of a context leaves its family-1 answer fixed by family 2 (a solver over Z_97 says determined). With separate values per family it is undetermined. The draft now gives each family its own context values, and check 9 makes the solver confirm it.

**Two things the simulation showed beyond the memo.**

- *Twins with equal values do occur.* In one world of three, a randomly drawn twin had contexts with the same value, and an evaluator that sees only the suffix then scores it. The draft's check 6 already rejects such twins; the simulation shows the check is needed and not decoration.
- *Twelve distinct twin pairs per family is small.* However many renderings are scored, the claim rests on 24 distinct pairs. This is now a question for the audit (question 7).

**The recommended sequence, and what I decide.**

1. *Probes on the saved E20-A checkpoints.* Adopted. Two probes, both scoring only: the answer when the true intermediate is supplied, against the answer the learner writes alone; and twins with the state reset just before the shared suffix. They say whether a failed twin fails at the intermediate or after it. The receipts already score twins with the state silenced throughout, which is the coarser form of the second probe. I will do these myself on the Mac, with predictions written first, on the three window references of stage 1 and then on stage 2 when all twelve lives are complete. It needs Mr. Xu's go, like any Mac run.
2. *Qualify the target task with a reference before any rule is compared.* Adopted, and written into the design draft as section 7b: the same held cases in a visible and a hidden version; no rule comparison until the reference passes both and the control separates them.
3. *Retention cost from one acquired checkpoint.* Adopted as conditions on the cost-curve design: every arm starts from the same checkpoint, fresh observations and their order are fixed, rehearsal is added as extra updates, and one arm gets the same number of extra updates without rehearsal. It still waits for the resumable-lives branch (E21), which is what makes "from the same checkpoint" possible.
4. *The every-byte failure with one change.* Adopted as conditions on the predictable-filler run: lengths, byte frequencies, answers and target count matched, sign sweep against backprop on one network. Not designed yet.

**My ordering.** The memo says to start with the probes and the step 0 corrections while E20-B qualifies. The step 0 corrections are done tonight. The probes are next. Step 0 version 2 goes back to Codex before any worker starts on it, under the rule of 9.82. Items 3 and 4 keep their place behind E21 and behind the probes.

**What I leave aside.** The memo suggests keeping the native store for a later matched test against a simple index. I agree and do nothing about it now.


## 9.92 E20-P prepared: three probes on the saved E20-A checkpoints, with predictions on file and nothing yet run (10 November 2026, 22:18 EST by the clock)

This is the first item of the sequence adopted in 9.91. Script `MOD97_LOCAL_RULE_20261106/e20_probe.py`, predictions `E20P_PREDICTIONS.md`, launcher `e20p_run.sh`. It is scoring only: no weight is changed.

**The three probes.**

1. *Intermediate.* For every twin item: the answer the learner writes alone, the ending it writes when the true intermediate is supplied, and whether its ending is the right second step of its own intermediate. This says whether a failed twin fails at the first step or after it.
2. *Reset.* The same twins with the state set to zero at two points.
3. *Readout.* A linear readout of each operand and of the intermediate from the state after `=`, fitted on fresh items and tested on the twins, with shuffled-label and untrained-net controls. This is my addition. It separates a learner that did not keep the hidden operand from one that kept it and could not use it.

**What building it showed about Codex's second probe.** Placed exactly, the reset gives zero twins by construction. If the state is cleared at the point after which the two members of a twin are the same input, they get the same answer, and a twin needs two different answers. So that form is a check that the assay has no leak, and the script asserts it. Placed where the memo's words put it, just before the first of the last ten bytes, it is a real measurement for the window learner, because the window still reaches back to the first operand for up to nine steps. Both are run. The receipts already hold the coarser version (state silenced throughout: zero twins in each of the seven complete lives).

**Fixtures, all on untrained nets and stub answers** (`exp45/E20P_FIXTURES_<seed>.json`, three seeds, all pass).

- Scoring: an oracle scores everything; a responder with a wrong intermediate and a consistent ending scores no intermediate, no whole answer and every "follows its own intermediate"; a responder without the end symbol scores no whole answer.
- Reset: with no reset the probe's forward pass equals the harness's exactly (difference 0.0); with one, positions before it are unchanged and later ones differ; after the reset at `=` twin members' outputs are identical.
- Readout: features that contain the operand read out 40 of 40; shuffled labels 1 of 40; features that contain a different operand 0 of 40. The fresh items share no operand triple with the twin panel or with teaching.
- The panels the probe builds have the same hashes as the panels the three lives scored. At run time the script also refuses to continue unless it reproduces each life's own twin count.

**One number from the fixtures worth keeping.** On an untrained net the in-view third operand reads out from the state on 118 to 134 of 160 twin items, and the hidden first operand on 3 of 160. So if a trained net's state gives up the first operand, that is something the life or the birth taught it.

**Scope.** The three window references of stage 1, which are filed (9.77). Four stage 2 checkpoints are complete as well; they are not probed until stage 2 is complete and filed. The two-bank lives need their own state code, to be written then.

**Predictions P-PR.1 to P-PR.7** are in the predictions file with confidences and readings. In short: most failed twins fail at the intermediate (70%); the second step is sound when the intermediate is given (65%); the first operand reads out on at least half the twin items (60%).

**Cost.** About 20 CPU-seconds per checkpoint, one process. It waits for Mr. Xu's go.


## 9.93 E20-P, the probes run: the window reference keeps the hidden operand and loses the twin at the first step (10 November 2026, run 22:48 to 22:49 EST; filed 22:51 EST by the clock)

Script `MOD97_LOCAL_RULE_20261106/e20_probe.py`, scorer `e20p_score.py`, outputs `exp45/E20P_e16_life_D_cm6_<seed>.json`, verdicts `exp45/E20P_VERDICTS.json`. Three checkpoints: the end-of-life weights of the window reference under backprop, E20-A stage 1 (9.77). Scoring only, 12 to 13 CPU-seconds each. Each run reproduced its life's own twin count before going on, and the exact reset gave zero twins, as it must. Evidence level E2, diagnostic. No registered E20-A verdict changes.

**The seven predictions** (seeds 73129 / 93683 / 116089; "holds" is at least two seeds).

| | Prediction | Result | Verdict |
|---|---|---|---|
| P-PR.1 | at least 70% of failed twins have a wrong intermediate | 91.3 / 89.5 / 77.5% | holds, all three |
| P-PR.2 | given the true intermediate, the ending is right on at least 85% of twin items with the second operator in view | 92.5 / 92.0 / 94.8% | holds, all three |
| P-PR.3 | at least 10 points lower where the second operator has left the window | 0.3 / 2.1 / 7.0 points lower | **misses**, all three |
| P-PR.4 | a wrong intermediate is followed by its own correct second step in at least 60% of cases | 88.8 / 88.1 / 89.8% | holds, all three |
| P-PR.5 | the hidden first operand reads out on at least 50% of twin items | 100 / 100 / 100% (1,200 of 1,200 each) | holds, all three |
| P-PR.6 | the intermediate is right 20 points more often where the operand reads out | no item where it does not read out | **cannot be judged**; my expectation that the readout would vary was wrong |
| P-PR.7 | with the state cleared ten bytes before `=`, twins fall to at most half | 0.47 / 0.42 / 0.13 of the ordinary count | holds, all three |

The readout's controls are in order: shuffled labels 0.7 to 0.9%, an untrained net 2.2 to 3.7%.

**What it shows, on the plus-minus pairs,** where there is no copy shortcut.

| Twin items, plus-minus pairs | 73129 | 93683 | 116089 |
|---|---|---|---|
| Intermediate right, first operand hidden | 47.0 | 56.8 | 76.2 |
| Intermediate right, same items with everything in view (leading control) | 83.0 | 79.3 | 90.8 |
| Ending right when the true intermediate is supplied | 89.7 | 84.0 | 87.3 |
| Twins both right, written alone | 21.0 | 26.7 | 43.3 |
| Twins both right with the true intermediate supplied | 81.7 | 73.3 | 76.0 |
| First operand readable from the state at `=` | 100 | 100 | 100 |

1. *The twin is lost at the first step.* Most failed twins have a wrong intermediate, and supplying the true one rescues 76 / 73 / 67% of all failed twins. The second step is sound: it is right about nine times in ten when given the true intermediate, and it is the right continuation of the learner's own wrong intermediate just as often.
2. *The hidden operand is not lost.* A linear readout recovers both its digits from the state on every twin item in every seed, against chance for an untrained net. The second operand reads out on 96 to 100%. So the learner arrives at `=` holding what the first step needs.
3. *It is used less well from memory than from view.* With the same operands in view the first step is right on 79 to 91% of items; from memory, on 47 to 76%. The registered reading for this case applies: the operand is kept and not used, and the limit is the computation from memory, not retention.
4. *The second operator leaving the window costs little* (P-PR.3 misses). The learner holds it.
5. *The state before the last ten bytes matters.* Cleared there, twins survive only where the window still reaches the first operand for six or more steps (eight or more in one seed), and never otherwise.

**Limits.**

- Three seeds, one network, backprop, end-of-life weights. Nothing here is about the working rule.
- I did not read out the first operator, which is also out of view in most twins. "Kept and not used" is shown for the operands. Whether the operator is kept is open.
- The comparison in point 3 is between the same items rendered two ways, which is clean. A comparison with the taught pairs' hidden-operand panel (first step right on 76 / 76 / 90%) is not like for like, because that panel pools all twelve taught pairs, comparison operators included.
- I have not looked at what the wrong intermediates are.

**What I take from it.**

- For composition inside one item, this reference does not need more memory. Mechanisms that add or protect within-item storage would not address what fails here. The retention work of 9.74 concerns forgetting across a life and is unaffected.
- E20-B asks whether a network with no window composes. This result says to expect the same split there: operands held, first step from memory weak. I will run these probes on E20-B's checkpoints myself when they exist; the worker's brief does not change.
- The next question is small and exploratory: what the wrong intermediates are (the other operator, a neighbouring value, a copied operand), and whether the first operator reads out. It is the same script with more written out, and another minute on the Mac.


## 9.94 E20-P2 prepared: the wrong intermediates and the first operator; predictions on file, nothing run (10 November 2026, 23:27 EST by the clock)

The follow-up named at the end of 9.93. Script `MOD97_LOCAL_RULE_20261106/e20_probe2.py`, predictions `E20P2_PREDICTIONS.md`, launcher `e20p2_run.sh`. Scoring only, same three checkpoints. It is exploratory: it was designed after E20-P's results were seen, on the same weights, so it can add detail to 9.93 and cannot confirm it.

**Two parts.**

- *A, what the wrong intermediates are.* Each readable wrong intermediate goes into the first of nine categories it fits (the other operator, operands swapped, another modulus, a copied operand, either operand one digit wrong, off by one, one digit right, other), and the shares are set beside what a uniformly random wrong value would give for the same items. Run on the twins and on the same items with everything in view.
- *B, whether the first operator is kept.* A four-way readout from the state at `=`.

**A confound caught in the design of B.** Among the four held pairs the first operator determines the second. A readout fitted on held pairs could therefore report the first operator while reading the second, which is often in view. The readout is fitted on fresh items of all sixteen pairs instead. The fixture shows the trap is real: given features that hold only the second operator, a readout fitted on the held four "finds" the first operator on 40 of 40 twin items, and one fitted on sixteen pairs gets 8 to 11 of 40.

**Fixtures** (`exp45/E20P2_FIXTURES_<seed>.json`, three seeds, all pass, untrained nets and stub answers only): a hand-worked item lands in each category as expected; a stub that always uses the other operator is counted there whole; the operator readout finds an operator that is present and not one that is absent; fit and test items share no prompt and no taught triple; the run path executes.

**One thing to keep in mind when reading the result.** On an untrained net the first operator reads out on 35 to 42% of items against a chance of 25%. The untrained control is not at chance for operators, so it is reported beside the trained figure.

**Predictions P-P2.1 to P-P2.4** are in the file. The first operator reads out on at least 90% where it is out of view (65%); the other operator accounts for at least a quarter of wrong intermediates (45%); near misses account for at least half (55%); the largest category is the same in view and from memory (50%).

**Cost.** Under a minute, one process. It waits for Mr. Xu's go.


## 9.95 E20-P2, run: the operator is kept too, and the wrong intermediates are near misses of the same kind in view and from memory (10 November 2026, run 23:29 EST; filed 23:31 EST by the clock)

Script `MOD97_LOCAL_RULE_20261106/e20_probe2.py`, scorer `e20p2_score.py`, outputs `exp45/E20P2_e16_life_D_cm6_<seed>.json`, verdicts `exp45/E20P2_VERDICTS.json`. The same three end-of-life checkpoints of the window reference under backprop. Scoring only, 9 CPU-seconds each; each run reproduced its life's own twin item count. Exploratory and diagnostic, E2: designed after 9.93 on the same weights, so it adds detail to that finding and does not confirm it.

**The four predictions** (seeds 73129 / 93683 / 116089).

| | Prediction | Result | Verdict |
|---|---|---|---|
| P-P2.1 | the first operator reads out on at least 90% where it is out of view, and on the sixteen-pair test | 100 / 100 / 100% on both | holds, all three |
| P-P2.2 | "the other operator" is at least 25% of wrong intermediates on plus-minus twins | 6.3 / 2.3 / 0.7% | **misses**, all three |
| P-P2.3 | near misses are at least 50% of them | 78.9 / 86.1 / 93.0%, against 27% for random wrong values | holds, all three |
| P-P2.4 | the largest category is the same from memory and in view | the same in every seed | holds, all three |

Controls for the operator readout: shuffled labels 22 to 27% (chance is 25%); an untrained net 41 to 50% where the operator is out of view. The second operator also reads out on 100%.

**The wrong intermediates on the plus-minus pairs.**

| Share of readable wrong intermediates | From memory (twins) | In view (same items) | Random wrong value |
|---|---|---|---|
| How many are wrong, of 600 items | 318 / 259 / 143 | 102 / 124 / 55 | |
| The value one changed digit of an operand would give | 72.9 / 77.6 / 86.7% | 90.1 / 87.9 / 80.0% | about 23% |
| The other operator | 6.3 / 2.3 / 0.7% | 0.0 / 2.4 / 1.8% | 1% |
| None of the categories | 12.3 / 8.5 / 4.2% | 6.9 / 4.0 / 3.6% | 68% |

1. *Everything the first step needs is in the state.* Both operands (9.93) and both operators read out on every twin item. Nothing was forgotten inside the item.
2. *The learner does not pick the wrong operation.* The other operator accounts for a few per cent of errors, near the random level in two seeds. My guess of a confusion of roles between the two operators was wrong.
3. *The errors are near misses.* Three quarters or more of wrong intermediates are the value that one changed digit of an operand would give.
4. *They are the same errors in view and from memory, two to three times as many.* Hiding the first operand does not bring in a new kind of error. It makes the usual one more frequent (ratios 3.1 / 2.1 / 2.6).

The registered readings that apply: everything is kept and the failure is in computing with it; the first step is the right operation done imprecisely; hiding the operand makes the same errors more frequent.

**What the breakdown cannot say.**

- *Which operand.* For addition and subtraction a changed digit in the first operand and in the second give nearly the same set of values, and the first is checked first. I have added the two rows together above for that reason. The 62 to 78% the scorer files under "first operand" is an artefact of the order.
- *Operand or arithmetic.* A slip in the addition itself can land on the same values. With the operand's digits readable from the state at 100%, the digit is there; whether it is read badly or added badly this does not show.
- *The baseline* assumes random wrong values are uniform. Real errors are not, so "about 23%" is a floor to compare against and nothing more.

**What I take from it, and what I do not.**

- For this reference, composing with an operand in memory is not blocked by storage and not by a wrong rule. It costs precision. That is a narrower and more ordinary problem than the one 9.76 set out to look for.
- I do not conclude that more training would close it. That is a guess this run does not test.
- The comparison that matters for the project is still open: the working rule's learner scores almost no twins in stage 2. Whether it has lost the operand or cannot compute with it is exactly what these two scripts measure. I will run them on the stage 2 checkpoints once stage 2 is complete and filed, and on E20-B's when they exist.
- No new run follows from this tonight.


## 9.96 Overnight, 10 to 11 November: E20-A complete and recomputed; E21 delivered and not yet mergeable; E20-B past the stop I wrote; a job commissioned by Mr. Xu directly (11 November 2026, 08:51 EST by the clock)

Four things arrived between my last filing (23:31 EST) and this morning. I read each and checked what can be checked from receipts. Nothing was run on the Mac for this section.

### A. E20-A, all twelve lives (default branch, commit de869f90)

The worker's report is `MOD97_LOCAL_RULE_20261106/E20A_REPORT.md` and its paragraph `E20A_REVIEW_SECTION_9_77.md`. I recomputed every figure from the twelve receipts with my own script (`e20a_coordinator_check.py`, output `exp45/E20A_COORDINATOR_CHECK.json`). Contract: all twelve have a completion record, seven stages, weights that match the recorded hash, the same memory panel per seed across the four arms, and the registered settings. No problems. The counts agree with the worker's table in every cell.

| Plus-minus twins, end of life (seeds 73129 / 93683 / 116089) | Both right, of 300 | Twin items, of 600 | Same items in view, of 600 |
|---|---|---|---|
| Window reference (backprop) | 63 / 80 / 130 | 254 / 286 / 398 | 470 / 420 / 484 |
| Backprop on the learner's pipeline | 16 / 11 / 68 | 134 / 117 / 269 | 339 / 269 / 396 |
| Sign-rule learner | 0 / 0 / 0 | 6 / 7 / 8 | 121 / 83 / 236 |
| Two-bank reference (backprop, no window) | 0 / 0 / 0 | 7 / 34 / 15 | 6 / 31 / 9 |

- **P20.4 holds** (two-bank reference at most 5% of plus-minus twins in all seeds: 0 of 300 in each).
- **P20.5 holds** (sign-rule learner at or below 0.60 of the reference's twin item accuracy: 0.02 / 0.02 / 0.02).
- **P20.6 misses** (backprop on the pipeline at or above 0.85: 0.53 / 0.41 / 0.68).
- P20.1 to P20.3 were judged at stage 1 (9.77): P20.1 and P20.3 hold, P20.2 misses.

*The worker's reading is right and I adopt it:* P20.4 and P20.5 hold, and neither says anything about memory, because neither arm passes the control with everything in view. The two-bank reference gets 6 / 31 / 9 of 600 plus-minus items with the operands in view. The sign-rule learner gets 121 / 83 / 236. A learner that cannot do the task in view cannot be tested for doing it from memory.

*The relaunches are clean.* A container restart killed four lives. The worker kept the killed receipts and relaunched the same four from their births. Each relaunch reproduced the killed run's scores exactly at every stage the killed run had reached (4, 4, 3 and 3 stages). One killed receipt differs from its relaunch in a logging field added between the two launches; its panel hash and scores are the same. No finished life was rerun.

*What the miss of P20.6 adds.* Backprop on the learner's own data pipeline composes from memory at about half the reference's rate and is also weaker in view. The pipeline (raw reservoir at half share, admission as the stream arrives) costs composition under backprop too. That was already visible in view; it holds from memory.

### B. E21, resumable lives (branch `claude/e21-resumable-lives`, commit 8b3ab4e7)

Delivered: checkpoints at every scored stage of a life and every curve row of a birth, resume by replaying the data stream with the learner off, 47 checks passed on the worker's host, including exact resume after a kill at three points for seven command shapes. Report `E21_RESUME_REPORT.md` on that branch.

It is not mergeable as it stands. The branch starts from 40cb1c20 and was not rebased, and `exp16_integrated.py` conflicts with the default branch (my changes since: the E22 preparations and the seed check in admission). **Not done by me yet:** resolving the conflict and rerunning its 47 checks on the merged file. Until then my promise stands and I do not edit `life()` or `develop17()`.

### C. E20-B (branch `claude/adoring-cori-b07aog`)

*Stage 1 is complete and carefully done* (`exp45/e20b_qual/E20B_STAGE1_RECORD.md`, runner at 0e131a29). By its record: birth parity and seven-stage life parity against my exact reference commands, with identical tensors at the end; the scoring fixtures; the architecture checks for all four networks; G3 bitwise equal to G0 with the added matrix at zero; nine admission refusals. It found and fixed a defect of its own (stage records without scores in a first attempt, kept as defective and not used). I have read this record. I have not rerun its checks.

*Its question to me, answered:* the operational reading at the end of `E20B_PREDICTIONS.md` is correct on all five points (hard mean from the last birth row at k = 12; new instances of taught pairs from `compose.within` at stage 6; plus-minus pooled over the two pairs' 1,200 items; whole answer is `exact`; taught instances kept outside `compose`).

*Stage 2 started without the stop of amendment 8.* The stage 1 record was committed at 04:18 UTC and the stage 2 run script at 04:30 UTC, "released by the coordinator's go". I gave no go, and no reading by Codex is on any branch. If Mr. Xu released it, that is his to do; I record only that the full dose began before the stage 1 record was checked by anyone but its author.

*Then a container restart* killed the three G0 lives after their fourth block of six. The three G0 births are complete. The worker stopped, as amendment 7 requires, and is writing a resumable runner (`exp20_resume.py`, in progress at 13:39 UTC).

*My position on the restart.* Restart the three G0 lives from their births with the runner that was qualified, unchanged, and keep the killed receipts. That is what E20-A did, and there the relaunches reproduced the killed runs exactly. A new resumable runner is a second runner: it may be used for the experiment only after its own kill-and-resume test shows a resumed life equal to an uninterrupted one in scores, losses and final tensors, and after that test has been read.

### D. E18b, E18c and E23 (branch `claude/funny-mendel-34puq7`)

Mr. Xu commissioned this directly in another session on 11 November: finish the every-byte births, then a clean control for the local heads, then targets at different time scales, each after three internal draft-audit-test cycles. His message lifts the hold I had placed on E18b (9.88). The plan is `E18B_E18C_E23_PLAN_20261111.md` on that branch; it marks itself as awaiting Codex.

*E18b is complete.* I read the twelve birth receipts on the branch myself and they agree with the worker's table.

| Birth, hard mean at k = 12 (three seeds) | Answer targets | Every-byte targets |
|---|---|---|
| Local heads with credit through time at every layer (B2) | 74.8 / 69.7 / 66.3 | 0.3 / 0.3 / 0.7 |
| Local heads without it (B3) | 1.2 / 0.8 / 0.8 | 1.0 / 1.7 / 0.2 |

The same-host B2 partner reproduces E18's figures to the decimal. Under every-byte targets the birth collapses, as the life did in 9.88. P18b.1 holds (operator and letter embeddings move); P18b.2, P18b.3 and P18b.4 miss.

*E18c has its births and lives filed and is not yet scored by the worker.* Read from the receipts by me, the relay births (the second bank's error allowed into the first bank, nothing else changed) end at 82.2 / 88.5 / 78.8 against B2's 74.8 / 69.7 / 66.3. That would meet its first prediction. It is the worker's to score and I leave the verdict to its report.

*E23 has four waves still to run.*

### What this leaves

1. **Three versions of the harness.** The default branch, the E21 branch and the E18b/E18c/E23 branch each change `exp16_integrated.py` from a different starting commit. They have to be brought together in one order: E21 onto the current head first, then the options of the E18b/E18c/E23 branch on top. Until that is done, a life started from one branch cannot be compared with a life from another without checking the source it ran.
2. **Probes on the stage 2 checkpoints.** Stage 2 is now complete and filed here. The two probe scripts run as they are on the window lives (sign rule and backprop on the pipeline). The two-bank lives need state code I have not written, and their in-view failure makes a memory probe of little use there.
3. **E20-B needs an answer on the restart** (C above).
4. **My check of E20-B's stage 1** by rerunning its parity on the Mac is still owed.


## 9.97 E18b, E18c and E23 complete: what they settle, one comparison their report does not make, and what I act on now (11 November 2026, 14:51 EST by the clock)

Report: `MOD97_LOCAL_RULE_20261106/E18B_E18C_E23_REPORT_20261111.md` on branch `claude/funny-mendel-34puq7` (commit ed07bf61), run at Mr. Xu's request by another session. Thirty-three births and lives, three seeds, one host, no wave interrupted. It marks itself as awaiting Codex. I read the report and its tables, and reread the birth and life receipts for the figures below. E2, development panels.

**What it settles.**

1. *Every-byte targets break the working rule at birth.* A birth that reaches 66 to 75 with answer targets ends at 0.3 to 0.7 with a loss on every byte. Putting every-byte targets on the first bank's head alone is enough to do it (5.5 to 10.0). With 9.88, where the same change collapsed a life, this now holds at birth and in life, in every seed.
2. *Cutting the error path between the banks costs 7 to 19 points at birth and 12 to 15 after a life.* The relay arm differs from the cut arm in that one path and nothing else.
3. *The question about later targets was not answered.* Its base arm was already broken by point 1. The report says so and names the lesson: a reference cell needs its own full-length result before a contrast is built on it.

**One comparison the report does not make.** The relay learner should be set beside the learner already on file from E22, which has one head at the top and passes its error down through everything. Same life settings, same seeds, read from the receipts with the report's own score (mean of the four mod-97 operations at k = 12):

| End of life, seeds 73129 / 93683 / 116089 | Mod-97 operations | Mod-73 operations |
|---|---|---|
| One head at the top (E22, on the Mac) | 88.1 / 84.8 / 84.3 | 90.7 / 87.5 / 87.1 |
| Local heads with the relay (E18c) | 80.6 / 87.0 / 77.8 | 88.9 / 89.0 / 84.4 |
| Local heads with the path cut (E18c) | 68.1 / 73.4 / 63.0 | 77.1 / 79.9 / 69.2 |

The two were run on different hosts, so this is not a verdict. As far as it goes, the relay learner is no better than the E22 learner: lower in two seeds and higher in one. So E18c shows what the cut costs. It does not show that local heads add anything to the rule we already have. That agrees with Mr. Xu's ruling of 10 November that feedback relayed layer by layer is allowed: the relay is the part that works.

**Where the project stands after these three days.**

- The working rule is sign feedback swept backward over the item, with answer targets and replay. For single operations it is born and lives within a few points of backprop (9.85).
- It has two known limits, and both stand between it and the target test:
  - *It needs answer-only supervision.* The target test has no answer mask: the learner predicts every next byte. As things stand the rule would fail that test at its first step.
  - *It does not compose on untaught operator pairs,* in view or from memory (9.96, A).
- Backprop does not have the first limit in a life (9.88). Whether it has it at birth is not on file.

**Is it time to act? Yes, on what does not depend on E20-B.**

1. *Close the local-heads line.* Local heads add nothing over the top-head rule, and every-byte heads do harm. I do not take up the report's suggested next design (a small-weight extra head) now.
2. *Make the every-byte failure the main question,* because the target test turns on it. The cell that is missing is the smallest one: a backprop birth of the same network under every-byte targets. If backprop also collapses at birth, the cause is the loss and both updates need the same remedy. If it does not, the cause is in the sign feedback. Beside it, one sign-rule birth with the answer bytes weighted to carry half the loss, which tests whether the answer signal is simply outnumbered. Six births, about two and a half hours on one cloud instance. A brief goes to Codex first.
3. *Unify the harness.* E21 onto the current head, then this branch's options. The other branch is final now, so this can be done once.
4. *Probe the stage 2 window checkpoints* (a minute on the Mac, on Mr. Xu's go).

**What waits for E20-B.** The network decision, and with it any work on composition under the rule. G0 and G3 are complete there; G1 is running and G2 is to come.


## 9.98 Codex's audit of 11 November, the harness brought to one version, and two briefs (11 November 2026, 16:02 EST by the clock)

Audit: `PROJECT_DOCUMENTS/MINIFLY_HOSTILE_AUDIT_E20B_STEP0_E18_E21_20261111.md` (commit 6f83bad7), made at revision 376092e, before the merge below existed. It covers E21, the outline of E24, E20-B's stage 1, step 0 version 2, and the E18b/E18c/E23 report.

### A. The harness is one version again

The default branch, the E21 branch and the E18b/E18c/E23 branch each changed `exp16_integrated.py` from a different starting commit. They are merged in that order. There were three textual conflicts, all in that file, each resolved by keeping both sides: the CPU accounting at the start of a life, the birth's configuration record, and one list in the birth loop.

**A defect the merge would have carried in.** On the E18b/E18c/E23 branch the harness cannot be imported with the window architecture, which is the default: one line sets an option on a module that is loaded only for the windowless networks. Every window command on that branch stops at import. Nothing that branch ran used the window learner, so its results are not affected, and its checks did not see it for the same reason. My second smoke test, on the window learner's path, stopped on it. The merge guards the line.

**A mistake of mine in the checks.** In one script I set the resume variables outside a cleaned environment, which strips them. Those runs were ordinary fresh runs and tested nothing about resume. I saw it because the gate test, which must fail, exited zero. Rerun with the variables in the right place.

**What was checked locally on the final file** (`exp45/HARNESS_MERGE_CHECK.json`; one process on the Mac, dry doses).

| Check | Result |
|---|---|
| `e18_check.py` | passes; all 20 values equal the record on file |
| `e23_check.py`, `e20b_audit_checks.py`, `e24_check.py` | pass |
| Two-bank dry birth and life against the pre-merge archive of 10 November | weights hash equal; birth curve and all 7 stage records equal |
| Window dry life (sign rule, raw store, drifting stream, online admission) against the pre-merge harness | weights hash equal; all 7 stage records equal |
| Each of the two lives killed and resumed, against its uninterrupted run | weights hash equal; all 7 stage records equal |
| Resume asked without a qualification record | refused, on both architectures |

**What was not checked.** E21's 47 acceptance checks have not been run on the merged sources. That is two command shapes at one tiny dose above, against seven shapes and three kill points there.

### B. Codex's findings, each checked, and what was done

| Finding | Checked how | Result | Action |
|---|---|---|---|
| Rerun all 47 checks on the merged revision before enabling resume | the merge itself made the case: it carried a defect none of the three branches' own checks had seen | adopted | the harness now refuses `E16_RESUME=1` unless `exp45/RESUME_QUALIFIED.json` names its exact source hashes; brief E21b written |
| A checkpoint's admission omits source hashes, torch version, dtype, a list of supported commands and a qualification gate | read in `run_config`; E21's own report says the torch version is not in the digest | confirmed | source hashes, torch version and dtype added to what a checkpoint is bound to; the gate added; the command list is in E21b |
| A torn final birth receipt can be parsed before the valid checkpoint is loaded | fixture: a receipt cut short beside valid weights and a valid checkpoint, resume asked | confirmed: the run stops with a JSON parse error before any checkpoint is considered | in E21b, with a kill during final publication as a required fixture |
| The far-head shuffle's private generator is not restored on resume | read in the merged birth loop | confirmed | resume is refused for that shape |
| E24: answer-only and every-byte backprop beside the matched sign conditions; bounded readings | wording | adopted | the brief's readings now say "in this network at this dose" and name an interaction between objective and update, not a cause in general |
| E20-B: accept the stage 1 qualification; file the early start as a deviation; verify all 24 completion receipts and weight hashes independently of the driver | to be done on the complete record | adopted | see C |
| Step 0: four corrections | the first is a matter of logic (a moved line is still a true line); the fourth I had seen in my own simulation and handled wrongly, by dropping the twin | all four adopted | the draft is at version 3 |
| E18c's composition table: its entries are percentages of final suffixes; whole answers on the plus-minus pairs are 0 / 2 / 1 and 3 / 5 / 1 of 1,200 | recomputed from the six life receipts | confirmed exactly | recorded here; the report on that branch should carry the correction |
| The cross-host comparison with E22 is descriptive and leaves the value of local heads unresolved | wording of my 9.97 | adopted | below |

**Correction to 9.97.** I wrote that E18c "does not show that local heads add anything". That stands. I also wrote, in the summary, that local heads add nothing. That overstates a comparison across two hosts. The right statement is that their added value is unresolved, and that I close the line because nothing on file argues for continuing it.

### C. E20-B: a deviation, recorded

- **Deviation.** Stage 2 of E20-B began at 04:30 UTC on 11 November, twelve minutes after the stage 1 record was committed, before any review of that record and before a release by this session. Amendment 8 of its brief required both.
- **Stage 1 itself** is accepted on Codex's reading of the committed receipts and mine of the record.
- **Interrupted work.** A container restart killed the three G0 lives after four of six blocks. Mr. Xu authorized their restart from the same births. The interrupted receipts are kept on the worker's branch and their CPU belongs in the cost ledger.
- **Before any outcome is read:** all 24 completion receipts and weight hashes are verified by my own script, independently of the worker's driver, which Codex notes can skip work on a receipt's completion field and does not pass on a child's failure. The weights have to be on the branch for that.

### D. Two briefs and a revised draft

- **E21b, resume qualification on the merged harness** (`OPUS_TASK_E21B_RESUME_QUALIFICATION_20261111.md`). It is what Codex's audit asks for, so a worker may start; its result is audited before resume is enabled.
- **E24, is the every-byte collapse the loss or the rule** (`OPUS_TASK_E24_EVERY_BYTE_CAUSE_20261111.md`). Draft for Codex. It adds one option to the harness, `E16_LOSSW=balanced`, off by default, with its own check and five dry births on file (`exp45/E24_CHECK.json`, `exp45/archive_e24_dry/`). Nine new births and two host-check reruns, about four and a half hours on one cloud instance. No worker starts before the audit.
- **Step 0 of the target test, version 3** (`OPUS_TASK_TARGET_TEST_STEP0_20261111.md`). Draft for Codex.

### E. What is merged along with the code

The E18b/E18c/E23 branch arrives whole: its plan, predictions, report, thirty-three receipts and their weights (about 260 MB). Its report still marks itself as awaiting Codex; Codex's audit above recomputed its 14 verdicts and asks for the corrections listed in B. Being on the default branch does not change that status.


## 9.99 Codex's audit of the E24 brief and of step 0 version 3: every point holds; both documents revised (16:20 EST by the clock, 11 November 2026)

Audit: `PROJECT_DOCUMENTS/MINIFLY_E24_STEP0_V3_AUDIT_20261111.md` (commit e823e169). Its disposition: hold E24's full dose and revise step 0 before any worker. I checked each point and adopt all of them.

### E24

| Finding | Checked how | Result | Change |
|---|---|---|---|
| Both reference baselines in the brief are wrong | reread the six receipts | **confirmed.** The backprop birth is 91.2 / 94.5 / 90.3; the sign-sweep birth with one head at the top is 87.5 / 90.7 / 84.7. I had given the sign-sweep scores to backprop, and to the sign sweep the scores of E18's arm with a head at each bank and backprop inside each (80.7 / 82.2 / 77.2). | corrected; the six original receipts are pinned by hash in the brief, with their weights hashes checked against the manifest |
| The host checks must be a wave of their own, in fresh paths, and shown to have really run | my own smoke test today hit exactly this: a birth "ran" in a folder that already held it, and the harness returned without running | confirmed | the two reruns go first and alone, in a folder with an empty `exp45`; the worker confirms a new receipt was written |
| The two updates also differ in weight decay | read in the harness: 0.30 for the backprop birth, 0.25 for the sign sweep, as in every birth of each kind on file | confirmed | the units compared are the two training configurations; the report gives each configuration's penalty per seed and their difference, as an interaction between objective and configuration |
| A miss of a mark is not a collapse; one seed above 10 misses P24.2 without settling survival; judge the balanced arm against the unbalanced one seed by seed | wording | adopted | the three marks stand unchanged; the readings are narrower |
| The check compares the every-byte path with a hand calculation and not with the harness before the option | read in `e24_check.py` | confirmed: the brief claimed more than the check did | a fixture added. For both updates and both objectives, with the option unset, the loss and every gradient now equal bit for bit those of the harness at the commit before the option existed (`exp45/E24_CHECK.json`) |

The roster is unchanged at nine new births and two host checks. The brief is at revision 2 and goes back to Codex.

**A reading the corrected baselines allow.** The E18b partner births reproduced E18's births exactly on another host and Python (9.96 and that job's report), so birth scores from the two hosts can be set side by side. Hard mean at birth, three seeds: sign sweep with one head at the top 87.5 / 90.7 / 84.7; local heads with the relay 82.2 / 88.5 / 78.8; local heads with the path cut 74.8 / 69.7 / 66.3. The relay learner is below the top-head learner in every seed. That is descriptive, since the arms were not designed as a pair, and it points the same way as the life comparison in 9.97.

### Step 0 of the target test

| Finding | Result | Change in version 4 |
|---|---|---|
| Check 3b still demanded zero first-encounter matches while allowing accidental ones | confirmed: my row contradicted itself | the pass mark is zero twins with both members right; single matches and abstentions are counted separately |
| Check 3 ran the oracle and the prefix-only solver together | confirmed: I described the oracle as solving from the stream | three separate things: the oracle is given the hidden values; the solver is given the bytes; the visibility test uses the solver |
| What the solver knows after the revision was not said | confirmed | the solver is told the change point as evaluator-only metadata; a revision must change the value; a revision probe needs three taught episodes of the changed cue after the change, in three contexts |
| Replacement episodes in the prior-teaching control; reveals and disclosures in that control | confirmed | replacements come from the taught pool; the control is checked against teaching, the other family, reveals and disclosure lines, at every probe |
| A disclosure line moved across the change tells a future value early or a stale one late | confirmed by reasoning: a line names a cue's value, and the value has two eras | lines move only within a phase and never across the change of a value they name; arms are compared on the disclosure ledger probe by probe |
| The parent design's decision rule still treats the moved lines as destroying information | confirmed (row 4 of its section 5) | section 7c added to the design draft: the two comparison streams answer two different questions and are read separately |

Version 4 goes back to Codex. The job stays generator-only.

### What this says about my drafts

Two of these errors are of one kind: a number or a claim written from memory and not reread (the baselines; "what they were at the commit before"). Both were in a brief I wrote within the hour of the merge. The audit caught them before any worker started, which is what the rule of 9.82 is for.


## 9.100 Codex's release audit: the merge released for fresh births, E24 and step 0 released with changes, resume held (16:38 EST by the clock, 11 November 2026)

Audit: `PROJECT_DOCUMENTS/MINIFLY_F4C6AA8_E24_HARNESS_STEP0_AUDIT_20261111.md` (commit 05a967a9), a read-only review at f4c6aa8c.

| Item | Codex's decision | Mine |
|---|---|---|
| Harness merge | released for fresh E24 births; resume held until E21b qualifies it | adopted |
| E24, revision 2 | released with changes, after both isolated host checks pass | adopted; the brief is at revision 3 |
| Step 0, version 4 | released with changes, for generator qualification only | adopted; the file is at version 5 and may go to one worker |

**The merge.** Codex reviewed the three merge commits and found the three resolutions and the window guard sound, and no further break of the default architecture. It confirms the harness hash against my check record.

**Two points about resume that I ran, and both hold.**

- *`mod97_local.py` is missing from the source binding.* `exp45.py` and `exp3_state.py` import it; my list of five files left it out.
- *An explicitly set default changes a checkpoint's configuration.* With `E16_FARHEAD=0` in the environment the configuration has 35 keys against 34 without it, so a run started one way and resumed the other would be refused for a setting that did not change.

Neither touches a run without resume. E24's births are running on this harness, so I am not editing it now. Both go to the E21b worker as governing amendments, with the rest of Codex's list for resume: the gate to check the runtime and the command shape, and to be bound to the hash of the check record it rests on.

**E24, revision 3.** The baselines, receipt hashes and weight hashes in revision 2 all recompute. The changes: a listing of the work folder before the host checks and the fresh completion records after; the reading of P24.3 corrected (reaching 40 shows nothing about the weighting unless the balanced arm is above the unbalanced one in the same seed), with the paired difference reported under every outcome; `E16_RESUME` left unset and checkpoint writing counted in the cost; a receipt without a `lossw` field read as uniform; the six source hashes in the report. In `e24_check.py` the life-refusal fixture now requires the intended assertion and no longer accepts a missing-file error.

**Step 0, version 5.** Check 1 is stated as a requirement on the ordinary stream's prefixes and check 9 on the control stream's; the oracle is given each probe's context and cue; the forty-byte table's key is spelled out; in the control stream a control context's reveals and disclosure lines are replaced as well as its teaching; eligible slots for moving a line are defined and lines that cannot move are counted; the extra-bytes stream is matched on placement. Section 7c of the design draft now carries Codex's three readings in its words and says which older sentences it supersedes. Twelve twin pairs per family are accepted for qualifying the generator and are to be revisited before learners are compared.

**The E24 worker's branch.** The host-check wave is on `claude/charming-clarke-u9g850`. That branch descends from an October commit and not from the default branch; its one commit says the sources are the default branch at f4c6aa8c and that only result files are added there. That is acceptable for results. A birth receipt does not record the hash of the harness that made it, which is why revision 3 asks for the source hashes in the report.

**Where each job stands.**

- E24: host checks running. The nine births start on my release line once both checks reproduce their pinned originals.
- E21b: in progress on `claude/funny-mendel-34puq7`, with the amendments above.
- Step 0: ready for one worker.
- E20-B: G2 lives were running at the worker's last push. Not read.


## 9.101 E20-B verified and read; E24's host check explained; step 0 reproduced; the audit session's six suggestions and what I decide (18:27 EST by the clock, 11 November 2026)

Three jobs reported this evening and the audit session wrote a review with six suggestions (`PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/REVIEW_AND_NEXT_STEPS_NOTE_20261111.md` on `claude/vibrant-cerf-sje726`, commit 78c099fa). I checked each claim I rely on with my own computation. Nothing was trained.

### A. E20-B: a windowless network composes from memory under backprop

The worker's report and review paragraph are on the default branch (commit fff36103). My own check of the 24 receipts (`exp45/E20B_COORDINATOR_CHECK.json`), done independently of the worker's driver as Codex asked (9.98 C): every birth and life has its completion record, every weights file matches its recorded hash, every life has seven stages. No problems.

| End of life, whole answers, seeds 73129 / 93683 / 116089 | Birth, hard mean | New instances of taught pairs | Untaught plus-minus pairs |
|---|---|---|---|
| G0, the two-bank reference | 93.2 / 92.8 / 92.8 | 47.3 / 50.8 / 53.3 | 1.5 / 0.5 / 7.5 |
| G3, two banks with the added loop | 98.8 / 93.2 / 98.0 | 76.5 / 69.3 / 80.3 | 20.2 / 1.1 / 1.0 |
| G1, GRU | 95.3 / 99.8 / 97.3 | 89.8 / 87.0 / 91.2 | 73.6 / 78.8 / 75.1 |
| G2, LSTM | 95.2 / 96.8 / 93.5 | 87.8 / 88.8 / 85.7 | 78.6 / 72.2 / 69.2 |

- **P20.7 holds** (both gated networks are born at 85 or above in every seed). **P20.8 holds. P20.9 holds. P20.10 holds** (the loop adds 29.2 / 18.5 / 27.0 points on new instances of taught pairs; as registered, that measures 32,768 added weights as a package).
- **Selection, by the rule fixed in amendment 13 before any result.** G3 does not qualify. The GRU and the LSTM both qualify in all three seeds. The tie rule takes the higher three-seed mean on the plus-minus pairs: GRU 75.8, LSTM 73.3. **The windowless candidate is the GRU.** A margin of 2.5 points selects; it is not a finding about the two networks.
- **What it says.** Composition from memory is learnable at this size by an ordinary recurrent network under backprop with answer targets. The fixed-rate two-bank design does not do it. By the addendum to 9.82 that design has no claim to stay.
- **What it does not say.** Nothing here was trained by the working rule. The scores are under the registered decoder, which reruns the network for each output byte. E2, development panels.
- **The deviation of 9.98 C stands on the record:** stage 2 began before review and release, and three G0 lives were interrupted and rerun from their births on Mr. Xu's authorization.

### B. E24's host check stopped, correctly, on a difference between host classes

The sign-sweep birth reproduced its original bit for bit (24 of 24 tensors). The backprop birth of seed 73129 differed at every checkpoint, ending at 93.2 against 91.2, and the worker stopped as instructed. The audit session traced it, and I repeated its comparison on the files:

- the worker ran the harness at f4c6aa8c (its recorded source hash equals the file's);
- its backprop rerun equals E20-B's G0 birth of the same seed in the hard mean at all six checkpoints, in the losses, and in **24 of 24 weight tensors**, although the two were made by different programs on different cloud hosts;
- against the pinned original, 1 tensor of 24 is equal, the fixed rate vector;
- the pinned backprop originals were born on the owner's machine (commit fad2cec7, 9 November); the sign-sweep originals on a cloud host (2eafb07b).

So two cloud hosts and two programs give one result, and the owner's machine gives another. The same command and seed differ by +2.0 / −1.7 / +2.5 points between the host classes. I had pinned, as a host check, an original born on a different class of host. **The E24 brief is at revision 4:** backprop baselines are run on the worker's host for all three seeds; the two that remain are added to the first wave and are expected to equal E20-B's G0 births.

What is not yet ruled out is a change in the harness's birth path between 9 November and now. One backprop birth of seed 73129 on the owner's machine with today's harness settles it: if it reproduces the pinned original, the cause is the host.

**Errata that follow.**

- *9.71* put the rule-born births at 0.96 / 0.96 / 0.94 of the backprop births. That divided cloud-born births by births from the owner's machine. Against cloud-born backprop births (93.2 / 92.8 / 92.8) the ratios are 0.94 / 0.98 / 0.91.
- *9.85 (E22)* compared two lives on one machine, which is sound, but the two births came from different host classes. The birth-level difference between host classes is about two points, the size of the gaps E22 reads. Its three verdicts stand as registered; "within 3 points" should be read with that beside it.
- *A rule from now on:* a seed is a pair only within one host class, and differences under about 3 points across host classes are not read.

### C. Step 0 of the target test is delivered and reproduces

`TARGET_TEST_STEP0/` on `claude/target-test-step0` (commit 7005a6af). I ran its check script on the owner's machine: all ten checks pass for the three world seeds, and all 3,722 values of my record equal the worker's. That is determinism across hosts and nothing more; I have not audited the code, and it goes to Codex. The worker raises two points, both from having a disclosure line after every episode: in the moved-lines stream a line names the probed pair before 51 / 47 / 55 of the 96 first-encounter probes, and the disclosure ledger never differs between the aligned and the moved arm.

### D. The audit session's six suggestions

| Suggestion | My decision |
|---|---|
| 1. Release E24 on same-host baselines, with the two missing backprop reruns | **Adopted.** Revision 4. The two reruns may start now; they are baselines, not new arms |
| 2. Before a GRU is born under the rule, measure the sweep's alignment on E20-B's saved weights, by layer and by steps back in time | **Adopted, and it is the next thing I do.** It trains nothing. In every network the rule has worked on, the step back in time crosses no weight matrix; in a GRU it crosses the recurrent weights at every step. I write the GRU and LSTM steps by hand, with the two checks the note names, and measure on G1, G2 and G0 |
| 3. Size the target test for learnability before more audit rounds | **Adopted.** Step 0 shows the answers are determined by the stream, not that a learner can acquire them. A sizing sweep with a backprop GRU, on world seeds that are never scored, registered as a sizing exercise with no verdict. I also take the point that the design must say what a learner may bring, and I expect to make the stated-value lines the base arm if the sweep shows what the note expects |
| 4. One ruling: are learned feedback weights inside the line | **Mr. Xu's to give.** My recommendation is yes: a separate feedback pathway that learns by a local rule and never copies a weight is feedback "relayed by a sub-network", which his answer of 10 November allows. The source the note cites needs checking before it steers a design |
| 5. Make E24 the last experiment on the two-bank network | **Adopted.** After E24, no repair, no predictable-filler stream and no retention curve on that network. Those go to the GRU |
| 6. Route every commission through the coordinating session | **Mr. Xu's to decide.** I support it. The merge of 11 November is what three starting points cost |

**Two things I owed that this changes.**

- *The probes on the stage 2 window checkpoints* (9.97) asked whether the sign rule's window learner lost the operands or cannot compute with them. That learner is on a network we are leaving. I drop the run unless a question about the GRU needs it.
- *My rerun of E20-B's stage 1 parity on the owner's machine* (9.96). Codex accepted stage 1 from the receipts, and full-dose evidence now exists that is stronger than my reduced-dose rerun would be: the harness and E20-B's runner give the same G0 birth, byte for byte, at full dose. I count it discharged.

### E. Order of work from here

1. E24: the two backprop baselines now; the nine births on my release line.
2. One backprop birth on the owner's machine, on Mr. Xu's go.
3. The alignment measurement on E20-B's saved weights: script and its two checks first, then the measurement on Mr. Xu's go.
4. A brief for the sizing sweep, to Codex before a worker.
5. Step 0's code to Codex.
6. Then, in this order and each after the one before it: the GRU under backprop with every-byte targets; the GRU born and living under the sweep against backprop on one host. The second is the run the 17 November close needs: a rule and a network that were born and lived together.


**Addendum to 9.101 (11 November 2026, 18:29 EST by the clock): Mr. Xu's ruling on learned feedback weights.** Asked whether a feedback pathway is inside the line when its weights are separate from the forward weights, learn by a local rule and never copy a forward weight, Mr. Xu answered: "Yes, allowed". This is the fourth working ruling, beside the three of the addendum to 9.82. What it permits is a rule whose backward pathway is learned; exact weight transport stays outside the line, as before. What it does not do is choose a mechanism: the two the audit session names (a weight mirror, and Kolen and Pollack's rule as modified by Akrout and colleagues) are to be source-checked before either steers a design, and nothing is built on them until the alignment measurement of 9.101 D.2 says whether weight signs are enough in a GRU.


## 9.102 E25 prepared: the sweep's alignment in a GRU, an LSTM and the two-bank network; predictions on file, nothing read yet (18:33 EST by the clock, 11 November 2026)

The measurement adopted in 9.101 D.2. Script `MOD97_LOCAL_RULE_20261106/e25_alignment.py`, predictions `E25_PREDICTIONS.md`, launcher `e25_run.sh`. It trains nothing: the file contains no optimizer.

**What it does.** On E20-B's saved births and end-of-life files, it computes each parameter's gradient twice, exactly and by the rule's sign path, and reports the cosine between the two. Under answer targets it also reports the cosine of the error at each state by steps back from `=`. The GRU and LSTM steps are written out by hand so that every linear map can run either way.

**Fixtures, on untrained networks only** (`exp45/E25_FIXTURES_<seed>.json`, three seeds, all pass).

- The written-out GRU and LSTM steps equal the library layers: outputs to 5e-7, every gradient to 2e-6 relative.
- With two gates' recurrent weights swapped, the written-out step no longer matches the library, so the comparison can fail.
- The sign path leaves the forward pass unchanged and the top layer's own gradient unchanged, and changes the recurrent matrix's gradient.
- For the two-bank network the exact path equals the harness's backprop gradient exactly.

**A baseline from the fixtures.** On untrained networks the state-error cosine is about 0.73 at `=` in all three. Twelve steps back it is 0.52 to 0.57 in the GRU, 0.47 to 0.58 in the LSTM, 0.59 to 0.64 in the first bank and 0.73 to 0.76 in the second. So even before training the recurrent networks lose more of the direction with distance than the banks do, and not catastrophically. Trained weights are what matters.

**Predictions P25.1 to P25.5** are in the file with their readings. In short: the GRU loses at least half its alignment over twelve steps (60%); the first bank keeps at least 0.8 of its own (65%); the GRU is below the bank at twelve steps (65%).

**Cost.** Seconds per file. It reads E20-B's 18 saved files for the three networks and waits for Mr. Xu's go, like any run on the owner's machine.


## 9.103 E25, run: in a trained GRU the sweep keeps its alignment through time, as well as or better than in the two-bank network (run 18:36 to 18:37 EST; filed 18:38 EST by the clock, 11 November 2026)

Script `MOD97_LOCAL_RULE_20261106/e25_alignment.py`, scorer `e25_score.py`, outputs `exp45/E25_ALIGNMENT_<seed>.json`, verdicts and tables `exp45/E25_VERDICTS.json`. Eighteen saved files of E20-B (births and end-of-life files of the two-bank network, the GRU and the LSTM, three seeds), each verified against the hash in its receipt before it was read. Six CPU-seconds per seed. Nothing was trained. Diagnostic, E2.

**The five predictions** (end-of-life files, answer targets, mod-97 operations at k = 12; seeds 73129 / 93683 / 116089).

| | Prediction | Result | Verdict |
|---|---|---|---|
| P25.1 | in the GRU the state-error cosine twelve steps back is at most half its value at `=` | 0.74 / 0.75 / 0.82 at `=`; 0.71 / 0.76 / 0.72 twelve steps back | **misses**, all three |
| P25.2 | in the first bank it is at least 0.8 of its value at `=` | 0.56 / 0.53 / 0.55 at `=`; 0.52 / 0.48 / 0.54 twelve back | holds, all three |
| P25.3 | twelve steps back the GRU is below the first bank | GRU 0.71 / 0.76 / 0.72; first bank 0.52 / 0.48 / 0.54 | **misses**, all three: the GRU is above |
| P25.4 | twelve steps back the LSTM's cell state is above the GRU by 0.10 | LSTM 0.64 / 0.72 / 0.71; GRU 0.71 / 0.76 / 0.72 | **misses**, all three |
| P25.5 | for the first layer's weights the GRU is below the two-bank network by 0.10 | GRU 0.72 / 0.51 / 0.60; two-bank 0.77 / 0.41 / 0.51 | **misses**, all three |

Four of five miss, all in the same direction: I expected the sign path to lose the error's direction as it crossed the GRU's recurrent weights again and again, and it does not.

**The error at the state, by steps back from `=`** (cosine between the sweep's error and the true error; end of life, mod-97 operations).

| Steps back | 0 | 4 | 8 | 12 | 16 |
|---|---|---|---|---|---|
| GRU | 0.74 / 0.75 / 0.82 | 0.72 / 0.76 / 0.72 | 0.71 / 0.76 / 0.72 | 0.71 / 0.76 / 0.72 | 0.70 / 0.76 / 0.70 |
| LSTM, cell state | 0.70 / 0.72 / 0.72 | 0.66 / 0.72 / 0.71 | 0.65 / 0.72 / 0.71 | 0.64 / 0.72 / 0.71 | 0.64 / 0.70 / 0.70 |
| Two-bank, second bank | 0.71 / 0.73 / 0.71 | 0.71 / 0.73 / 0.70 | 0.71 / 0.73 / 0.69 | 0.71 / 0.73 / 0.69 | 0.71 / 0.73 / 0.69 |
| Two-bank, first bank | 0.56 / 0.53 / 0.55 | 0.55 / 0.53 / 0.53 | 0.54 / 0.51 / 0.54 | 0.52 / 0.48 / 0.54 | 0.47 / 0.46 / 0.50 |

**The update to each weight matrix** (cosine with the true gradient; end of life, answer targets, mod-97 operations).

| | Recurrent or state-writing matrices | First layer | Embedding table |
|---|---|---|---|
| GRU | recurrent 0.79 / 0.84 / 0.87; input 0.76 / 0.78 / 0.83 | 0.72 / 0.51 / 0.60 | 0.69 / 0.71 / 0.74 |
| LSTM | recurrent 0.76 / 0.72 / 0.77; input 0.76 / 0.67 / 0.76 | 0.59 / 0.42 / 0.40 | 0.59 / 0.47 / 0.43 |
| Two-bank | first bank's write path 0.51 to 0.79; second bank's 0.68 to 0.84 | 0.77 / 0.41 / 0.51 | 0.74 / 0.58 / 0.57 |

1. *On these weights the sweep's update to a GRU is as well aligned as its update to the two-bank network,* on which the rule is known to train (9.70, 9.85). That holds for the recurrent matrix itself, for the layers below it, and twelve to sixteen steps back in time.
2. *A likely reason, not tested here.* In a trained GRU the step back in time has a part that crosses no weights: the gate value that carries a unit's state forward also carries its error back. Units that hold an operand across the letters have that gate near one. The untrained networks of the fixtures, whose gates sit near one half, lost a quarter of their alignment over twelve steps; the trained ones lose almost none. I have not measured the gate values, so this is a reading and not a result.
3. *The LSTM is not clearly better than the GRU.* The candidate stays the GRU.
4. *Under every-byte targets the alignment is not worse.* Over all parameters it is 0.79 / 0.80 / 0.82 in the GRU against 0.67 / 0.64 / 0.74 under answer targets, and 0.83 / 0.78 / 0.82 against 0.71 / 0.68 / 0.68 in the two-bank network. One matrix of the two-bank network (the first bank's projection) is lower in two seeds. So on these weights the every-byte collapse is not a matter of the sweep pointing the wrong way. That is a lead for reading E24 and nothing more.

**The reading registered beforehand applies:** the GRU's alignment at the longest gaps is at or above the two-bank network's, so a GRU birth under the sweep is worth its compute, at a reduced dose first.

**What this does not show**, as the predictions file said before the run. These networks were trained by backprop. Alignment at one point of training is not learning; a high figure is a licence to try a birth under the rule and is not a forecast of its score. Three seeds, one size.

**What follows.** The next job is the one the close of 17 November needs: a GRU born and living under the sweep, against backprop on the same pipeline and host, with the recurrent step written out as it is here. No remedy for the rule is designed before that birth has been tried: on this evidence none is yet called for, and learned feedback weights stay in reserve.


## 9.104 E26 briefed as a draft: a GRU born and living under the sweep, beside backprop on the same pipeline and host (18:42 EST by the clock, 11 November 2026)

Brief: `OPUS_TASK_E26_GRU_UNDER_THE_SWEEP_20261111.md`. Draft for Codex; no worker starts before the audit. It is the run the close of 17 November needs: no rule and network on file were born and lived together on a network that composes.

**The design in four lines.**

- *One difference.* Two arms on E20-B's GRU, same initial values, same batches in the same order, same optimizer at weight decay 0.25: one updated by the sweep, one by backprop. Both use the written-out recurrent step of E25, so that only the path back differs.
- *E20-B's protocol and marks.* The birth and the six-block life on the joint stream under which this network is known to compose, with 85 on new instances of taught pairs and 40 on untaught plus-minus pairs as absolute marks. The learner's own replay pipeline is a later step.
- *Three stages with a stop after each:* implementation and reduced-dose qualification; the six births; the six lives. The lives are released after the births have been read, the lesson of E23's base arm.
- *Five predictions,* with the sweep-born birth at 70 or above held at 50% and the two composition marks at 25% each. A candidate pair exists if the sweep's arm meets both marks in the same two seeds.

**Why decay 0.25 in both arms.** Every rule run on file uses 0.25 and every backprop birth 0.30. E24's audit showed that this makes two configurations and not two updates. Here the backprop arm takes the rule's decay. E20-B's own GRU, at 0.30 with the library layer on another host, stays on file as what the network is known to reach and is not an arm.

**Cost.** Measured on one batch on the owner's machine, an update through the written-out step costs 1.4 times the library layer's on the exact path and 1.6 times under the sweep. That puts the twelve runs near 68,000 CPU-seconds, six to seven hours on three cores, to be replaced by stage 1's measurement.

**Four questions are put to the audit:** whether a backprop arm at 0.30 is also needed; whether the joint stream is the right first pipeline; whether a floor for releasing a sweep-born life should be fixed now; and whether the sweep's scale should be per stacked matrix or per gate.


## 9.105 Codex's audit of the E26 brief and of step 0's code: E26 released for stage 1 with changes; step 0's ordinary stream accepted and its moved-lines arm to be corrected (18:53 EST by the clock, 11 November 2026)

Audit: `PROJECT_DOCUMENTS/MINIFLY_E26_STEP0_IMPLEMENTATION_AUDIT_20261111.md` (commit f5dabbfa). I checked the points that can be checked in the source and adopt all of them.

### E26, now at revision 2

| Finding | Checked how | Result | Change |
|---|---|---|---|
| The harness's rule update would be the wrong one here | read in `exp11_cell.update`: for the rule it runs the forward with credit cut at every step and then calls the trace routine, which `arch20` replaces by a function that stops with an error | **confirmed.** My brief described the way back in a table and never said which loop to run. A worker following E20-B's runner would have reached for that call | the loop is written out in the brief: one loop for both arms, credit through the whole item, only the path back switched. One real optimizer step is tested in each arm, with the harness's rule update as the fixture that must fail |
| The scorer must be qualified on the written-out network | the helper I wrote for E25 has no `silence` argument, which the production scorer passes | confirmed | the network gets the harness's full signature; the three scoring fixtures; and E20-B's saved end-of-life GRU files, loaded into the written-out network, must give the counts in their own receipts |
| Freeze the release of lives before reading the births | my draft left the release of a rule-born life to my judgment of its birth | adopted: that would have let me choose which seeds are compared | all six lives run; the stop before stage 3 checks technical validity only; a life not run is counted as unmeasured |
| Keep feasibility apart from the paired comparison | wording of the readings | adopted | the absolute marks and the difference from the backprop partner are read separately; a missed reference no longer silences the rule's own result; a birth under 70 is reported with its level |
| Keep decay 0.25 in both arms, the joint stream, one scale per stacked matrix | my four questions | answered as drafted | a backprop arm at 0.30 is a follow-up, wanted only if the matched reference disappoints |

Stage 1 may go to a worker. Stages 2 and 3 are released by me in turn.

### Step 0

Codex accepts the ordinary stream as a qualified generator for the three filed worlds. With my rerun on another host (9.101 C) that part is done, and the delivered folder `TARGET_TEST_STEP0/` is merged into the default branch with this section.

Two corrections to the moved-lines arm, both adopted in version 6 of the brief:

- *No line may name a pair before that pair's probe.* The worker had reported 51 / 47 / 55 early disclosures out of 96 itself. The lines of first-encounter episodes now stay in place and only teaching lines move; a deliberately early line must fail the exposure check.
- *The rendered lines are checked as bytes.* I read the delivered check: it compares the symbols named in the lines between the arms and does not parse the values. Each value is now to be parsed from the stream and compared with what its era allows.

Neither touches the ordinary stream. The sizing sweep (9.101 D.3) needs only the ordinary stream and the aligned lines, so it does not wait for these corrections.


## 9.106 Three jobs reported: E24's baselines agree and its nine births are released; E26's stage 1 is accepted and its births released; step 0's moved-lines arm is corrected (20:18 EST by the clock, 11 November 2026)

Each was checked on the files by me before anything was released. On the owner's machine I ran scoring and check scripts only; nothing was trained.

### E24: the baselines are in, and the nine births are released

The worker ran the two remaining backprop baselines on its host (branch `claude/charming-clarke-u9g850`, commit 661e0e84). I loaded both weight files beside E20-B's G0 births of the same seeds: sealed scores equal at all six checkpoints, losses equal, and 24 of 24 tensors equal, for seed 93683 and for seed 116089. With seed 73129 (9.101 B) that makes all three. So E24's backprop baselines are 93.2 / 92.8 / 92.8 and its sign-sweep baselines 87.5 / 90.7 / 84.7, every one born on a cloud host, and the harness and E20-B's runner are shown equal at full dose for all three seeds. The nine new births are released.

### E26: stage 1 accepted with a ruling on one failed check; stage 2 released

Record: `exp45/e26_qual/E26_STAGE1_RECORD.md` (branch `claude/adoring-cori-b07aog`, commit 87648b01), now merged with its sources `arch26.py` and `exp26_candidate.py`.

- **What the worker found.** Every check passes except check 1, which fails in 1 case of 18: on one trained file the written-out step's logits differ from the library layer's by 1.14e-5, against my limit of 1e-5. The worker changed nothing, reported the check as failed, and held stage 2 for a ruling. That is how a failed check should be handled.
- **My ruling: the limit was badly set, and the check is accepted.** I had fixed 1e-5 as an absolute limit from untrained networks, whose logits are below 1. Trained logits reach 25 to 36, where that limit is a few float32 spacings. The difference is 2e-6 of the logit it sits on. The limit is restated as relative, and under it all 18 cases pass. I am changing a criterion after seeing the result, and I record it as that; what makes me willing is the check below, which does not depend on any tolerance.
- **The decisive check holds, on two host classes.** E20-B's saved end-of-life GRU files, loaded into the written-out network and scored by the production scorer, give exactly the counts in their own receipts: all 70 scored blocks on the worker's host, and the five composition panels of all three seeds on the owner's machine (`e26_coordinator_check.py`, `exp45/E26_COORDINATOR_CHECK.json`). On the owner's machine the worker's network also equals my E25 helper bit for bit, in the forward pass and in the sweep's gradients.
- **The check Codex asked for did its work.** One real optimizer step in each arm equals an independent computation tensor for tensor, and the harness's own rule update, run as the fixture, stops at the trace routine; with the trace disabled it differs in 7 tensors, because it cuts credit at every step.
- **Cost, measured.** The sweep costs 1.06 to 1.07 times the exact path on the worker's host. Forecast: about 19,800 CPU-seconds for the six births and 31,200 for the six lives, roughly two and three hours.

At the reduced dose both arms learn about as much as each other, which tests nothing and shows only that the sweep's arm is not dead on arrival.

### Step 0: the moved-lines arm is corrected

Branch `claude/target-step0-moved-lines` (commit 8294fb16), now merged. I reran its check script: every check passes for the three world seeds and all 3,815 values equal the worker's. Of the 61 recorded hashes, 58 are unchanged from the accepted version; the 3 that changed are the moved stream of each seed, as intended. Exact-pair lines before a first-encounter probe are zero in every stream. Both new fixtures fail as required: a line placed ten bytes before its pair's probe, and one wrong value in one line. Step 0 is complete for the dense line schedule; whether the schedule should be sparser waits for the sizing sweep.

### Not finished

E21b is still in progress on `claude/funny-mendel-34puq7`.


## 9.107 The sizing sweep for the target test, briefed as a draft (20:26 EST by the clock, 11 November 2026)

Brief: `OPUS_TASK_TARGET_SIZING_SWEEP_20261111.md`. Draft for Codex; no worker starts before the audit. It is the third suggestion of the audit session's note, adopted in 9.101.

**What it is.** A sizing exercise and nothing else: it registers no prediction for judgment, gives no verdict and concludes nothing about any rule. One learner, a backprop GRU from random weights with every next byte as its target, is run on the step 0 generator's stream at 18 sizes, on world seeds that are never scored.

**The grid.** Modulus 97, 31 or 11; each taught pair shown 4, 32 or 256 times before the first panel; the stream with or without the lines that state the values.

**Two ways of feeding it, because they answer different questions.** Episode by episode with the state reset, which asks whether the hidden values and their addition can be learned from outcomes at all at a given size. And as one continuing stream read once with the state carried, which is how the target test itself runs.

**What its table decides, all three about the design and none about a learner.**

1. Whether the proposed sizes (modulus 97, four presentations) can stay.
2. What a learner must be allowed to bring: nothing, in which case the stream has to be long enough to teach a rule slowly; or a birth on a declared curriculum that shares no pair and no value with the test world.
3. Whether the stated values should be the base arm and their absence the harder one.

The second of these is the decision the design has been missing since its first draft, and it is the coordinator's and Mr. Xu's to make once the table exists.

**Four questions are put to the audit,** the sharpest being whether cutting credit every 256 bytes would make the lines useless for a reason that has nothing to do with the sizes.


## 9.108 Codex's audit of the sizing-sweep brief: held as written; four defects, all mine, all confirmed; the brief revised (20:34 EST by the clock, 11 November 2026)

Audit: `PROJECT_DOCUMENTS/MINIFLY_TARGET_SIZING_SWEEP_AUDIT_20261111.md` (commit 5163a452). Disposition: hold learner training as written. No worker had started. The brief is at revision 2 and goes back to Codex.

| Defect in my draft | Checked how | Result | Change |
|---|---|---|---|
| The modulus would not have reached the evaluator | run: I set the generator's modulus to 31 and asked the other modules. The oracle still held 97 and so did the solver; the checker reads the generator's value once, when it is imported | **confirmed.** Twelve of the eighteen settings, every one at modulus 31 or 11, would have been scored against answers computed modulo 97. My draft said the sizes are "set through its SIZES table (and its modulus constant)", as if there were one constant | one explicit configuration passed to generator, oracle, solver and checker, each still computing for itself; wraparound and wrong-modulus fixtures for addition and subtraction at each modulus; a configuration that does not reach a component stops admission |
| The decision rule for taught pairs could never be met | reasoning: a table keyed by context and cue recalls every taught pair it has seen | **confirmed.** I asked for taught pairs "above the tables" | taught recall is read against that table as its ceiling, on a taught panel frozen beforehand (step 0 has none), with a short-filler version beside the long one. Held behaviour is read separately, against shortcuts built from the same arm's prefix, including one that reads the stated values off the lines |
| A probe inside a training chunk was undefined | reasoning: a chunk can hold a probe and its reveal, and restoring the hidden state does not undo an update that used the reveal | confirmed | the order of events at a probe is written out, with two fixtures on the real training path: changing the reveal and later bytes leaves the response unchanged, and runs with and without probes end with identical weights |
| The cost was wrong twice | arithmetic on the generator's schedule: 3,552 episodes at four repetitions, 112,416 at 256 | **confirmed.** I wrote "about 64 times longer"; it is 31.6 times, because only the first segment's repetitions scale. And I priced 18 cells where the roster is 81 fits | two stages: implementation and a bounded qualification that times the two longest paths; the roster on release, cut beforehand if the forecast passes the budget |

**The readings, restated as Codex asks.** Five classes for a fit, fixed before any fit. An all-negative grid leaves this GRU and this recipe unqualified; it does not show the task is beyond every learner, which is what my draft's first decision said. A negative from random weights leaves the value of a birth curriculum open; that question needs its own paired comparison and is not answered by this sweep, which is what my draft's second decision implied. Ordinary against lines measures the lines as a package; the matched extra-bytes stream is added at one setting to separate content from bytes.

**Step 0's accepted results are not touched by the first defect.** At the default configuration all three components hold 97, and the oracle, the solver and the stream agree in every check on file.

**What I take from it.** This is the third brief in two days in which an audit found a claim I had not run: the E24 baselines written from memory, the E24 check that did not test what the brief said, and now a modulus I assumed lived in one place. The rule of 9.82 caught each before a worker started. The habit to fix is mine: where a brief says a setting "is changed through" something, I now change it and look, before writing the sentence.


## 9.109 Codex's audit of the sizing sweep, revision 2: stage 1 released with changes, the roster held (20:47 EST by the clock, 11 November 2026)

Audit: `PROJECT_DOCUMENTS/MINIFLY_TARGET_SIZING_SWEEP_REV2_AUDIT_20261111.md` (commit ce7619d5). The four findings of the first audit are accepted as addressed. Four further changes before dispatch, all adopted; the brief is at revision 3.

| Change | Checked how | Result |
|---|---|---|
| Fork twins before either prompt; set the boundary before a probe by the last loss target, not the last input | reasoning on my own protocol: I saved the state at the first member's `=`, by which point it has read the first member's context; and a chunk whose last input is `=` has the first answer digit as its target | both confirmed. The twin protocol is rewritten to save the state before the first byte of either member, and the boundary rule is stated on targets, with a fixture that shows the rule can fail |
| The 20-point margin cannot apply to a responder that uses the disclosed values | step 0's own record: in the streams with lines both stated values have been disclosed before 96 of 96 first-encounter probes | confirmed. Such a responder scores near the ceiling, so my class "held success" was unreachable in every line arm. The margin now applies to recitation shortcuts; the direct-use responder is reported beside the learner as a benchmark. The size decision is pinned to the acquisition phase at the first panel |
| Say whether the matched extra-bytes stream is trained on | my draft built it and did not say | it is: three fits at the proposed size, so that content can be told from bytes. 84 fits |
| Bound stage 1 and settle where the parameterized code lives | my draft said both "new files only" and "a small change to the step 0 modules" | copies under `sizing/`, with default parity against the accepted step 0 record; profiling on a world seed that is in no roster, 500 updates on each of four paths, two hours of CPU for the stage |

Stage 1 may go to a worker. The roster is released by me after its record: the modulus fixtures, the causal scoring checks, the dependency across a cut, and the cost forecast.


## 9.110 Overnight results and Codex's overnight memo: the sweep brings a GRU to birth as backprop does; E24 answers its question; E21b goes back; sizing stage 1 finds the learner's span (09:53 EST by the clock, 12 November 2026)

Codex's memo: `PROJECT_DOCUMENTS/MINIFLY_OVERNIGHT_E21B_E26_REVIEW_20261112.md` (commit 3b58b66c). This section was written by the coordinating session running on Claude Opus 5.5, since Fable was not available. Every figure below was recomputed from receipts or rerun, as in earlier sections.

### A. Codex's memo: both of its points hold, and both are acted on

| Point | Checked how | Result | Action |
|---|---|---|---|
| E26's births record finite losses for 3,072 of 24,576 updates each | read in `exp26_candidate.dev`: the list of losses is cleared every 512 updates and flagged only at the six scored rows | confirmed | a coordinator ruling (B below); the lives' loop keeps every loss of a stage, so the lives are covered directly |
| E21b's gate omits `mod97_local.py`, the qualified runtime and the check record's digest, and explicit defaults still change identity | read the gate on the worker's branch | confirmed. Cause: the worker's branch starts at a301ae8e and the amendments that ask for exactly these things were written after it (3c928dfa) | E21b goes back with the amendments and Codex's four refusal tests; resume stays off |

The memo did not find sizing stage 1, which had been pushed overnight (E below).

### B. E26, the births: the sweep brings a GRU to birth as backprop does

Six births, full dose, one cloud host, the same data in the same order, decay 0.25 in both arms, the same written-out step. Recomputed from the receipts. Hard mean, add89 and sub89 at k = 12, whole answers:

| Seed | Arm S, the sweep | Arm B, backprop | S / B |
|---|---|---|---|
| 73129 | 98.0 | 98.5 | 0.995 |
| 93683 | 98.2 | 98.7 | 0.995 |
| 116089 | 99.8 | 93.7 | 1.065 |

- **P26.1 holds** in all three seeds (at least 70; I had put it at 50%).
- **P26.2 holds** in all three (at least 0.85 of the partner; I had put it at 40%).
- Over the six checkpoints the two arms cross several times. The paired difference S − B is −6.7, +1.3, +8.3, −4.0, +20.2 and +5.2 points on average over the seeds. The learning curves are not distinguishable at three seeds.
- For reference only, since host, decay and recurrent code all differ: E20-B's GRU (library layer, decay 0.30) was born at 95.3 / 99.8 / 97.3.

**What this is.** The first birth of a network that composes from memory, made by the working rule, and it is level with backprop on the same pipeline. With E25's alignment it says the sweep's weight signs, crossing the GRU's recurrent matrix at every step back, carry enough of the gradient to learn single operations at full accuracy. **What it is not:** composition. That is the lives' question. Answer targets only. E2.

**Ruling on admission.** Codex is right that the flags do not cover every update. I rule that no update of any birth had a non-finite loss, from the record: a non-finite loss or gradient would enter the weights and both of AdamW's moments and could never leave them, and every birth's last 512 losses are finite and its final weights answer 98 to 100% of 1,200 sealed items. Recorded as an inference, in revision 4 of the brief. **Stage 3 is released.** One condition comes first: the six birth weight files, which the worker's `.gitignore` kept off the branch, are committed before the first life starts.

### C. E24 complete: under every-byte targets the sweep collapses at birth and backprop does not; giving the answer half the loss restores it

Nine new births on one cloud host. Recomputed from the receipts. Hard mean:

| Seed | D-ans | D-all | T-ans | T-all | T-bal | Penalty, backprop | Penalty, sweep | T-bal − T-all |
|---|---|---|---|---|---|---|---|---|
| 73129 | 93.2 | 91.5 | 87.5 | 0.8 | 81.5 | 1.7 | 86.7 | +80.7 |
| 93683 | 92.8 | 89.3 | 90.7 | 1.0 | 79.8 | 3.5 | 89.7 | +78.8 |
| 116089 | 92.8 | 91.2 | 84.7 | 1.2 | 83.3 | 1.6 | 83.5 | +82.1 |

- **P24.1 holds, P24.2 holds, P24.3 holds**, all in all three seeds.
- The penalty for every-byte targets is 1.6 to 3.5 points under backprop and 83.5 to 89.7 under the sweep: an interaction between objective and training configuration, in the two-bank network at this dose.
- The two configurations also differ in decay (0.30 against 0.25). The worker's report says so, and the update is not isolated from decay.
- Giving the answer bytes half the loss brings the sweep back to within 1.4 to 10.9 points of its answer-only level. So in this configuration the harm is a matter of proportion: the answer's share of the error is swamped. Where such a weighting could come from without an answer mask is the open question. The balanced arm uses the mask and is a diagnostic only.
- With E25, which found the sweep's update no worse aligned under every-byte targets on backprop-trained weights, the collapse is not the sweep pointing the wrong way at one moment. It is what the sweep does over training when most of the targets are unpredictable filler. That is a reading, not a result.

E24 is the last experiment on the two-bank network (9.101). Its 98 files were brought onto the default branch by path, without the unrelated October history of the worker's branch.

### D. E21b goes back

See A. The 78 passing checks are kept. The job is finished on the current head with amendments 1 to 4 and the four refusal tests (status note at the end of its brief). Resume stays off.

### E. Sizing stage 1: the pipeline qualifies, and the learner's span is about 12 bytes

Record: `TARGET_TEST_STEP0/sizing/SIZING_STAGE1_REPORT.md`, merged. The worker reported its one registered failure plainly and changed nothing to pass it.

- Items 1 to 4 and 6 pass. I reran the default parity (0 differences in 3,815 values) and the modulus fixtures on the owner's machine. The code needs Python 3.12 or newer.
- **Item 5 failed as registered.** The learner copied a byte 12 positions later across a cut at 99.7%, and learned nothing at 24, 40, 60 or 600 positions, cut or no cut, even at 10,000 updates for 60. The limit is span, not the cut.
- **What that does to the design.** In the line streams a value is stated a median of 411 to 734 bytes before the probe that needs it, so in arm B the lines cannot be read for content. A first-encounter probe in the ordinary stream needs the context letter 12 to 16 bytes back, at the edge of the span.
- **Revision 4 of the brief**, to Codex:
  - the roster stays at 84 fits, with the lines read in arm B as a package only and the base-arm question not decided by this sweep;
  - the copy fixture is added at 16 and 20 bytes before the roster;
  - the worker's alternative cut plan is adopted (keep the 256-byte grid, read forward without an update, replay the chunk), so that training is identical to a run without probes.

This bears on the target test itself, not only on the sweep. The test as drafted asks a learner to use a disclosure hundreds of bytes after it, or to hold a context across a filler. A GRU of E20-B's shape, trained as here, does not carry information 24 bytes over random filler. Either the test's distances are brought within reach, or the learner it is meant for is not this one.

### F. Where this leaves the close of 17 November

- **The candidate pair has been born:** the sweep and the GRU, level with backprop at birth. If the lives meet E20-B's composition marks in the same two seeds, the close can name it, with three conditions attached: answer targets, a joint rehearsing stream, and E2.
- **The next question for the target test is already visible.** The target test has no answer mask. E24 shows the sweep collapsing under every-byte targets on the two-bank network. Whether it does so on the GRU is the bridge from E26 to the target test: E26's two arms, born under every-byte targets. I will brief it after the lives have been read, not before.


## 9.111 The scorer for E26's lives, written before any life exists (09:59 EST by the clock, 12 November 2026)

`MOD97_LOCAL_RULE_20261106/e26_score_lives.py`, with fixtures in `e26_score_lives_fixtures.py`. The practice of E22 (9.85): the reading of a result is fixed in code before the result exists.

**Contract checks**, each of which stops the reading:
- seven stages and a completion record;
- finite losses at every stage, which in the life loop covers every update;
- AdamW read back at decay 0.25 with the embedding table at 0;
- the weights file matching its hash when present;
- each life bound to its own arm's and seed's birth by the hash in that birth's receipt;
- the two arms of a seed fed the same data at every stage, and scored on the same panels.

**What it computes:**
- P26.3, P26.4 and P26.5;
- the candidate-pair reading of the brief: arm S meets both marks, 85 on new instances of taught pairs and 40 on the untaught plus-minus pairs, in the same two or more seeds;
- every endpoint at every stage, each untaught pair separately;
- the paired difference S − B at every stage.

A life that is missing counts as not meeting a mark, and the denominator stays three, as the brief says.

**Tested on the reduced-dose receipts of stage 1:**
- It reproduces the counts in the stage 1 record: new instances of taught pairs 77 and 73 of 600.
- Four corrupted copies are each caught: data that differs between the arms at one stage, a birth hash that is not the birth receipt's, a stage with non-finite losses, and decay 0.30.

At that dose nothing is near a mark, as expected. Nothing here is a result.


## 9.112 E26's lives and the crossover: the sweep composes from memory, under the registered marks; the gap is in the update used during the life (11:33 EST by the clock, 13 November 2026)

This section lands work that had been sitting on worker branches since 12 November, on the audit session's suggestion (its note of 13 November, relayed by Mr. Xu). Sources: `MOD97_LOCAL_RULE_20261106/exp45/e26/E26_LIVES.md`, `exp45/e26x/E26X_DESIGN.md` and `E26X_RESULTS.md`, and the memo on which the crossover was authorised, `PROJECT_DOCUMENTS/MINIFLY_E26_RESULT_AND_NEXT_STEPS_20261112.md`, which was on the owner's machine only and is committed with this section. The crossover was commissioned and run while I was not in the loop; I read its design after the fact.

### A. The six lives, scored by the scorer written before they existed (9.111)

`e26_score_lives.py` reports no contract problem: seven stages each, finite losses at every update, decay 0.25 read back, each life bound to its own birth, both arms of a seed fed the same data and scored on the same panels. End of life, whole answers, seeds 73129 / 93683 / 116089:

| | New instances of taught pairs (mark 85) | Untaught plus-minus pairs (mark 40) | Both marks |
|---|---|---|---|
| Arm S, the sweep | 83.5 / 83.8 / 81.0 | 54.2 / 44.5 / 37.2 | 0 of 3 |
| Arm B, backprop | 84.7 / 93.7 / 87.2 | 63.2 / 89.2 / 66.3 | 2 of 3 |

- **P26.3 misses** (the sweep is under 85 in all three seeds, by 1.2 to 4.0 points).
- **P26.4 holds** (40 or more on untaught pairs in two seeds).
- **P26.5 holds** (backprop meets both marks in two seeds).
- **No candidate pair exists by the registered reading.** The marks are kept as they were fixed before E20-B ran.

**What the result is, with the marks missed.** A network born and living under the working rule now composes from memory on operator pairs it was never taught: 37 to 54% of whole answers, where the two-bank network under either update scored 0 to 8%. That is the first composition by the rule in this project. It is short of backprop on the same pipeline (63 to 89%) by 9, 45 and 29 points.

### B. The crossover (E26-X): the birth is not where the gap comes from

Six further lives on the same seeds, each starting from an existing admitted birth and living under the other update. All six complete, finite, bound to the right births, weights matching their hashes; I recomputed every figure.

| Birth → update during the life | New instances of taught pairs | Untaught plus-minus pairs | Both marks |
|---|---|---|---|
| sweep → sweep | 83.5 / 83.8 / 81.0 | 54.2 / 44.5 / 37.2 | 0 of 3 |
| backprop → sweep | 85.5 / 81.7 / 83.0 | 39.8 / 45.8 / 53.9 | 0 of 3 |
| sweep → backprop | 93.2 / 97.3 / 93.0 | 72.3 / 82.8 / 73.9 | 3 of 3 |
| backprop → backprop | 84.7 / 93.7 / 87.2 | 63.2 / 89.2 / 66.3 | 2 of 3 |

- **A sweep-born GRU that then lives under backprop meets both marks in all three seeds,** better than the backprop-born one. So the sweep's birth is a sound starting point; nothing is missing from it.
- **Either birth, living under the sweep, misses.** The shortfall belongs to the update used while compositions are being learned.
- Three of the crossed lives were interrupted by container restarts and finished by a new resumable runner. The audit session reports that they equal the interrupted receipts at every stage those reached. I did not repeat that comparison.

### C. A pattern across the twelve lives, checked on the receipts

Take every checkpoint from the first composition block on at which a life scores between 75 and 88 on new instances of taught pairs. At that equal level of taught skill, lives updated by the sweep score 43.4 on the untaught pairs on average (21 checkpoints) and lives updated by backprop 63.5 (9 checkpoints). So the sweep is not simply behind on a common path that more training would finish: at the same taught skill its solution carries over less to untaught pairs. This is descriptive, pooled over seeds and checkpoints that are not independent.

### D. Erratum to 9.110

9.110 B gave the paired difference between the arms over the six birth checkpoints as −6.7, +1.3, +8.3, −4.0, +20.2 and +5.2 "points on average over the seeds". Those figures are in the worker's table's units, answers out of 300, and not percentage points. In points they are −2.2, +0.4, +2.8, −1.3, +6.7 and +1.7. The reading there, that the two birth curves are not distinguishable, is unaffected. Found by the audit session.

### E. What I take from it for the close and for the next step

- **The close on 17 November keeps the marks and says what happened:** born level with backprop; composes from memory at 37 to 54% against 63 to 89%; the registered marks missed; the gap is in the update during the life and not in the birth.
- **Next, measurement before training,** as the audit session proposes and as E25 did before E26. One hypothesis would explain both open failures: weight signs carry no magnitudes, so error from unpredictable bytes floods back undamped (the every-byte collapse of E24), and credit does not favour the pathways that already do the work (the weaker carry-over to untaught pairs). Two measurements on saved weights test it without training: the share of filler error reaching each layer under the exact and the sign path, and the sweep's alignment on composition items against single operations, on the sweep-lived and the backprop-lived files. I will write these with predictions first, as E25 was.
- **If it holds, learned feedback weights are the remedy to try,** inside the line by Mr. Xu's ruling of 11 November, after a source check. The crossover gives it bounds on file: lives from the existing sweep-born births score 37 to 54 under the sweep and 72 to 83 under backprop.
- **The target test's distances are not shortened yet.** E20-B's GRU held operands across 40 letters after a birth, while the sizing sweep's GRU from random weights carried nothing past 12 bytes. The copy fixture is to be rerun with mixed delays and from E26's born weights before the design is changed.

### F. Auditing without Codex

Codex is not available for now. Between 10 and 12 November it stopped four runs that would have been wrong as briefed, and I confirmed each. Until an outside auditor returns:

- the audit session reads every brief before a worker starts and every result before it is filed. It is the same model family as the coordinator and the workers, so it is a second reader and not an independent one, and briefs say so;
- every claim about code in a brief carries the command that showed it; every number comes from a script over receipts; each pass mark has one responder that must reach it and one that must not; one real update step per arm is checked against an independent computation;
- new runs flip one switch on qualified code where they can, as the crossover did, in preference to new training loops.

### G. Still missing from the repository

The end-of-life weights of the six E26 lives are only on the worker's container (its `.gitignore` kept them off the branch; the crossover's six are committed). They are needed for the measurements of E. Asked of the worker with this section.

### H. E21b is merged: resume is qualified for seven command shapes on the qualified runtime

The worker requalified after amendments 1 to 4 (branch `claude/funny-mendel-34puq7`, commit deea398c: 97 of 97 checks, the gate now checking the dependency closure with `mod97_local.py`, the runtime and the check record). Before merging I ran my own test of that harness on the owner's machine, one process at a dry dose:

- a birth and a life without resume equal the pre-merge archive of 10 November, weights hash and all seven stage records, so the default path is unchanged;
- a life killed after three checkpoints and resumed with the override equals the uninterrupted one, weights hash and all seven stage records;
- **the production gate refused to resume on the owner's machine,** naming the reason: its torch (2.9.1) is not the qualified one (2.15.0+cpu). That is the refusal Codex asked for, met on a real mismatch and not a fixture.

So resume is available on hosts with the qualified runtime, for the seven listed shapes, and nowhere else. No outside audit has read this revision; Codex's last reading named four refusal tests and the worker's report lists them as passing. The audit session is asked to read the gate.


## 9.113 E27 prepared: two measurements on saved weights for one hypothesis; predictions on file, nothing read (11:39 EST by the clock, 13 November 2026)

Script `MOD97_LOCAL_RULE_20261106/e27_measure.py`, predictions `E27_PREDICTIONS.md`, launcher `e27_run.sh`. It trains nothing. It is the fourth step of the audit session's note of 13 November (9.112 E).

**The hypothesis under test:** the sign path carries no magnitudes, and that one fact explains both the every-byte collapse and the weaker composition.

**A, flood.** Under every-byte targets the gradient splits exactly into the answer's part and the other bytes' part. The script computes both under the exact and the sign path and reports how much larger the other bytes' share is under signs, and how much of the answer's direction survives.

**B, alignment by kind of item.** On the end-of-life GRU files, the sweep's cosine with the true gradient on single operations, on taught pairs and on untaught pairs.

**Fixtures, on untrained networks only** (`exp45/E27_FIXTURES_<seed>.json`, three seeds, all pass): the split is exact to 5e-7 under both paths, and it fails when the two shares are swapped.

**Two things the untrained networks already show.** The amplification is 0.98 to 1.05: before any training the sign path lets through no more filler error than the exact path. And the sweep's cosine is the same on all three kinds of item (0.84 to 0.87 in the two-bank network, 0.92 to 0.93 in the GRU). Neither settles anything about trained weights.

**Five predictions,** each held below one half, because E25 already speaks against the hypothesis in its simple form. The readings say in advance what a miss means: if the flood is not there at a single update, the collapse is something that builds over training, and learned feedback weights lose their main motive for it.

**Not available yet:** the weights of the six E26 lives. Part B runs on the six crossover lives now and is completed when the others are committed.

It waits for Mr. Xu's go, like any run on the owner's machine: seconds per file, one process.


## 9.114 The first hostile audit under the new division of work: E27 held, eleven findings, the main ones run and confirmed (11:49 EST by the clock, 13 November 2026)

**The division of work, from Mr. Xu on 13 November:** all implementation and all runs go to Opus 5.5 workers; the coordinator writes the briefs and has them hostile-audited. I read that as: I start fresh auditor sessions, check their findings by running them, and revise before any worker starts. The auditors are of the same model family as the coordinator and the workers, so they are second readers and not independent ones.

E27 was the first case. I had written its script and predictions myself that morning (9.113). The brief `OPUS_TASK_E27_FLOOD_AND_ALIGNMENT_20261113.md` hands both to a worker as a draft by an interested party, and an auditor read brief, predictions and script before anyone ran them. **Its disposition was to hold.**

| Finding | Run by me | Result |
|---|---|---|
| The all-parameter amplification is diluted by the top layer, whose gradient is the same under both paths | on an untrained two-bank network the top layer's amplification is exactly 1.0; in a control that amplifies the first layer 3.19 times the all-parameter figure is 1.55 | confirmed. P27.1's mark of 1.5 sits on the value a real threefold flood would give. The prediction could miss with the hypothesis true |
| The positive control I suggested in the brief cannot work | one layer scaled to one hundredth: amplification 0.97 | confirmed. The scale cancels. I had written a control without trying it |
| The script drops the GRU's recurrent matrices from its per-tensor output | listed the keys: five tensors, the two recurrent ones absent | confirmed. A name filter of mine |
| The taught-pair panel is a biased slice | counted: five pairs at 50, one at 6, none starting with a comparison | confirmed |
| P27.5 cannot be judged on the files in the repository | the six E26 life weight files are absent | confirmed, and already known; the brief had said to judge it on what exists, which is impossible |

Not run by me and accepted on reading: the predictions for A are measured at weights the every-byte births never visited; "usable" has no baseline; P27.3 compares the sign path with itself and is uninformative; P27.4 mixes composition with item format; one of my readings ("less faithful where composition is learned") claimed something no measurement on these items could show.

**What I did with it.** The five predictions stay as registered, since nothing had been read. An amendment to the predictions file, written before any file is read, says where each is read from, states each one's weakness, registers six supplementary quantities with their readings, and narrows two readings that overclaimed. The brief is at revision 2 with the named defects and the corrected control.

**What it shows about the arrangement.** The auditor found in four minutes what I had not seen in a script I had tested. Five of its eleven findings are of the kind Codex used to find: a claim in a brief that had not been run.


## 9.115 E28 written and audited: the GRU born under every-byte targets (13 November 2026, 12:01 EST by the clock)

**What it asks.** The target test has no answer mask. E24 showed a sign rule collapsing under every-byte targets on the two-bank network; nothing is on file for the sweep on the GRU. E28 is E26's births with the loss mask changed: sweep and backprop, every-byte and balanced targets, three seeds, twelve births and two host checks, about 12.5 CPU-hours. Brief: `OPUS_TASK_E28_GRU_EVERY_BYTE_20261113.md`, revision 2. No run has been made.

**The hostile audit** (a fresh session of the same model family; a second reader, not an independent one) returned nine findings on revision 1. I reran the evidence for seven and adopted all nine. The three that changed the design:

- *E24 is not a like-for-like precedent.* Its sweep cut credit in time and used the trace routine, and its backprop partner ran at decay 0.30. My draft would have read a survival as "the collapse belonged to the two-bank network". The reading now attributes nothing among the three differences.
- *My draft's readings had holes:* a backprop arm at 84 with the sweep at 1 would have been read as "do not single out the rule". Margins and a paired rule are now fixed before the run.
- *A balanced backprop arm was missing,* so a shortfall of the balanced sweep could not have been assigned to the rule. Added.

One measurement from the audit is worth keeping: at E26's born weights the non-answer positions carry over 99% of the every-byte gradient under backprop and under the sweep alike. Proportion alone therefore does not explain why one rule would collapse and not the other.

Also pushed: a read-only brief for a source check on learned feedback weights (`OPUS_TASK_SOURCE_CHECK_LEARNED_FEEDBACK_20261113.md`).


## 9.116 The sizing learner's span: a brief, and a doubt about stage 1's reading (13 November 2026, 12:13 EST by the clock)

**The brief.** `OPUS_TASK_SIZING_SPAN_20261113.md`, revision 2, comes before the held sizing roster: stage 1's copy fixture at fixed distances from 4 to 60 on three seeds, mixed distances, and starts from E26's born weights and from a permuted copy of them. 48 runs, about 9 CPU-hours, backprop only. No run has been made.

**What the hostile audit found that matters beyond the brief.** Stage 1's result, "copies at 12, nothing at 24 or beyond", was not at equal dose. A fixed distance D gives 256/(D+1) repeats per update, so the run at 24 saw about half the repeats of the run at 12, and the run at 12 was itself still at 8% after 13,714 repeats. **The statement that this learner's span is about 12 bytes (9.110, 9.112 E, the sizing brief's revision 4) is therefore unproven: it may be dose.** I repeated that statement in three places without checking the dose. The new brief counts dose in repeats and registers the erratum as a reading if 24 passes.

**A second thing the auditor measured,** which I reran on one birth: on the copy fixture at distance 4, E26's born network learns two to three times slower than random weights. A born start is not a head start on a task of another kind. This bears on "what may a learner bring" in the target test.

Eleven findings in all; all adopted, four rerun by me.


## 9.117 E27, part A: the flood hypothesis is not supported; part B waits on six files (13 November 2026, 14:24 EST by the clock)

**What was done.** An Opus 5.5 worker ran E27's measurements on saved weights, no training (branch `claude/sleepy-knuth-q96dd1`, commit 7515e7cb; not merged, since part B is incomplete). It fixed the four defects in my measurement script that the brief listed and a fifth it found itself: the loss record was overwritten once per path. I read the scorer's output and the three measurement files. The scorer's hash equals the one recorded before the run. Evidence level E2; a measurement on end-of-birth weights.

**Registered predictions.**

| | 73129 | 93683 | 116089 | Result |
|---|---|---|---|---|
| P27.1 amplification on the sweep-born two-bank file, all parameters, at least 1.5 | 0.53 | 1.41 | 0.80 | **misses** |
| P27.2 usable share lower under signs than exact | exact is negative | exact is negative | exact is negative | **void in all three**, as the amendment provides |
| P27.3 collapsed file at least twice the balanced one | 1.10 | 0.73 | 1.65 | **misses** |
| P27.4, P27.5 | | | | not judged: the six E26 life files were not on the worker's container |

**The reading.** The hypothesis was that the sign path lets the unpredictable bytes' error flood the lower layers, and that this is why every-byte targets collapse a sweep birth. It is not supported, and the supplements say why more plainly than the misses do. Amplification below the top layer, by file (three seeds):

| File | Amplification | Outcome of that birth |
|---|---|---|
| untrained, either network | 0.96 to 1.07 | |
| two-bank, backprop, answer targets | 1.93 / 1.81 / 1.81 | healthy |
| two-bank, backprop, every byte | 2.18 / 1.05 / 3.13 | healthy |
| two-bank, sweep, every byte | 1.25 / 1.42 / 3.05 | **collapsed** |
| two-bank, sweep, balanced | 17.2 / 12.4 / 15.7 | healthy |
| GRU, backprop | 2.93 / 2.09 / 1.77 | healthy |
| GRU, sweep | 2.47 / 1.84 / 1.68 | healthy |

Every trained network shows the sign path amplifying the non-answer error, the backprop-born ones included, and the most amplified file of all is the balanced one, which did not collapse. **Amplification is a property of trained weights under a sign backward path, not a mark of the collapse.** The worker's summary leads with the supplements showing amplification of the letters' error; that is true and it does not discriminate.

**Limits.** These are the weights after the collapse, not where it happened; E24 saved no rows. E28 saves every row of its every-byte sweep births for this reason. The "usable" measure is uninformative on answer-trained files, where the other bytes' error is 89 to 605 times the answer's under the exact path too.

**Also checked:** the six E26 life weight files are now committed (branch `claude/adoring-cori-b07aog`, commit ce409d64) and each file's SHA-256 equals `weights_sha256` in its life receipt. The E27 worker is asked to finish part B from them with the script and scorer unchanged.


## 9.118 Slow units and transmission delays: Mr. Xu's question, a brief, and what its audit showed before any run (13 November 2026, 15:09 EST by the clock)

**The question.** Mr. Xu, 13 November: a neuron's decision is shaped by several earlier moments and not by a fixed number of them, through slow neurons that integrate and through finite transmission speed; have we tested this? Not as he means it. The two-bank network had slow units that did not form a circuit. E20-B's G3 had fixed-rate banks with a feedback matrix and scored 20.2 / 1.1 / 1.0. The GRU has learned gates and is born with every unit fast. No experiment here has used transmission delays. He asked for a brief on both mechanisms; it is `OPUS_TASK_E29_SLOW_UNITS_AND_DELAYS_20261113.md`.

**Three things the hostile audit showed, each rerun by me.** None is a result of E29; all are on file already or come from short checks.

1. **Erratum to 9.110, 9.112 E and the sizing brief's revision 4: the GRU's span is not about 12 bytes.** The span job's block B is finished: with distances 4, 12, 24 and 40 mixed in one stream, the plain GRU from random weights passes at every distance at 100% in all three seeds, after about 1,400 to 1,500 repeats of each. Stage 1's failure at a fixed 24 is a plateau of optimisation at that one distance, not a limit on carrying. 9.116 raised the doubt as one of dose; the answer is that mixing distances is enough. The span job's block A, fixed distances at equal dose, is still running, and this erratum does not wait for it: carrying 40 is shown whatever block A gives.
2. **The trained GRU already holds.** In E26's births of seed 73129 (the auditor's preliminary look), a quarter of the units keep their state at 0.99 or more through the distractor letters, and every unit is fully rewritten somewhere in the item. The gate works as a switch. Training makes holding units out of a network born with none.
3. **On a copy at one fixed distance a delay is a shortcut.** A unit delayed by the distance less one hands the key to the output: at distance 12, plain 6.6% and the delay arm 100% after 600 chunks. My draft's delay arm would have measured the luck of the draw.

**What I do.** Part A of E29 is released: a read-only measurement of how the born and lived networks use their gates, sweep against backprop, with the seed already seen reported and not judged. Part B, variants at birth, is withdrawn as I drafted it and is rewritten after block A, on a fixture with no single lag and with the controls the audit asked for. The draft was mine and eleven findings against it held.

**For the target test.** The lines stand hundreds of bytes before their use, and what limits a learner there is now an open question again; "span of 12" is no longer a reason to move them.


## 9.119 E27 complete: all five predictions miss or are void; neither half of the hypothesis is supported (13 November 2026, 15:30 EST by the clock)

**What was done.** The worker reran E27 with the six E26 life weight files, script and scorer unchanged (hashes 51ada865 and ee2dd33e as before the first run), on branch `claude/sleepy-knuth-q96dd1`, commit 7826d1b1, now merged. I read the new scorer output: nothing is absent, P27.1 to P27.3 and supplements S1 to S5 are identical to the first pass (9.117). Evidence level E2; measurements on saved weights.

**The second half of the hypothesis** was that the sign path's alignment with the exact gradient is worse on untaught composition items, and worse in sweep-lived networks.

| | 73129 | 93683 | 116089 | Result |
|---|---|---|---|---|
| P27.4 alignment on untaught pairs at least 0.10 below single operations, both sweep-lived arms | gaps 0.03, 0.08 | -0.04, -0.02 | 0.05, 0.05 | **misses** |
| P27.5 sweep-lived at least 0.05 below backprop-lived, both births | one of two | none | none | **misses** |

Alignment on untaught pairs is between 0.53 and 0.66 in every arm, and in four of six comparisons the sweep-lived network is the better aligned. S6, untaught minus taught, lies between -0.08 and +0.05 everywhere.

**Outcome for E27:** P27.1 misses, P27.2 void, P27.3 misses, P27.4 misses, P27.5 misses. The hypothesis that the sign path's missing magnitudes explain both the every-byte collapse and the weaker composition is not supported in either half. E26-X stands as measured: lives updated by the sweep carry less to untaught pairs than lives updated by backprop. **This measurement leaves that without an explanation.** One-step alignment at the end of a life does not differ between the arms.

**Two corrections to my 9.117, from the worker.** I wrote that every trained network shows amplification below the top layer. Two files do not in every seed: the sweep-born answer-target two-bank file (0.63 and 0.96 in two seeds) and the backprop-born every-byte file in seed 93683 (1.05). The reading is unchanged: the collapsed file sits inside the range of files that did not collapse, and the balanced file is far above all of them. And the worker checked what I had left open: E27's sign path is the gradient E24's sweep births computed (relative difference 0, 0 and 4.6e-7), with the births then applying it through AdamW.

**What it leaves for the next step.** Per-step direction is not where the two updates differ. What has not been measured is what accumulates over a life: thousands of updates, each through AdamW. E28's saved rows, and E29's look at how the networks use their gates, are the measurements on file that can speak to that.


## 9.120 Three jobs read: E28's stage 1 passes and its births are released; the span roster; how the trained GRUs use their gates (13 November 2026, 16:32 EST by the clock)

All three were run by Opus 5.5 workers on cloud hosts (Linux x86-64, Python 3.13.18, torch 2.15.0+cpu) and read by me from their records. The span and E29 branches are complete and merged; E28's is not merged until its births are in.

### A. E28 stage 1: all six items pass; stage 2 released

Branch `claude/charming-cori-ljzwz1`, commit 48a6e563. I checked from the files: both host-check births carry `experiment="E28"`, reproduce E26's `tensors_sha256`, all thirteen tensors bit for bit, and every row's scores and data digest; the cumulative finite-loss flag covers 24,576 of 24,576 updates; the runner has its own loop and never calls `exp26_candidate.dev` or `.step`; the four predictions are identical to the brief's; the balanced loss is checked against the independently weighted form. The first run of the dry-birth check failed through a bug in the check, kept on file and fixed in its own commit. Measured cost 0.127 to 0.134 CPU-seconds per update, about 11.2 CPU-hours for the twelve births. **Released.** Nothing has been read against P28.1 to P28.4.

### B. The span roster: 48 runs, 9.5 CPU-hours

Branch `claude/target-test-sizing-stage-1-drmnlq`, commit cc91ff46. Admission passed, stage 1's counts reproduced exactly. Backprop only; a copy over random filler; no claim about any rule. Class on the last tenth, three seeds (1104 / 1105 / 1106):

| Start | 4 | 12 | 16 | 18 | 20 | 24 | 40 | 60 |
|---|---|---|---|---|---|---|---|---|
| random, fixed distance | pass ×3 | pass ×3 | pass ×3 | pass ×3 | pass ×3 | partial / pass / pass | chance / pass / chance | chance / partial / not held |
| random, mixed {4, 12, 24, 40} | pass ×3 | pass ×3 | | | | pass ×3 | pass ×3 | |
| born (E26, backprop), fixed | pass ×3 | pass ×3 | | | | pass ×3 | chance / partial / chance | |
| born, mixed | pass ×3 | pass ×3 | | | | pass ×3 | pass ×3 | |
| permuted born, fixed | | pass ×3 | | | | pass ×3 | | |

Forecasts: **F1 holds** (18 passes, three of three). **F2 holds** (24 passes, two of three). **F3 misses** (60 passes in none). **F4 holds** (mixed passes at 40 where fixed does not). **F5 holds** (at 12 the born start needs more repeats in two of three pairs, equal in the third). **F6 misses** (born passes at 40 in none).

Readings, as the brief fixed them:

- **Stage 1's "nothing at 24" was dose.** At about 60,000 repeats fixed 24 passes in two of three seeds. The erratum of 9.118 stands and is now complete: the fixed-distance span of this learner at this dose is 24, and with mixed distances it carries 40.
- **Arm B's ordinary stream carries no span caveat** (it needs 13 to 18; the block's span is 24).
- **Mixing distances helps, and by a large factor.** At 40, mixed passes in three of three seeds and fixed in one. Where both pass, mixed reaches 50% after about 1,400 repeats of that distance against 9,000 to 47,000 fixed, more than a factor of two in every seed at 12 and 24.
- **A born start is not a head start here.** At distance 4 it needs two to three times the repeats of random weights (21,333 to 32,000 against 10,666); at 12 and 24 it is level or slower; it passes at 40 no more often. The permuted copy of the born weights does as well as the born weights at 12 and 24, so nothing here is owed to the birth's structure.
- **One run to remember:** fixed 60, seed 1106, sat near 100% for about forty windows and then fell to chance and stayed there. A learned copy was lost under backprop with every byte a target. One run; it is recorded because E28 asks about exactly this kind of loss.

**For the target test.** The lines stand 411 to 734 bytes before their use and nothing here reaches past 60 at a fixed distance; mixed distances beyond 40 were not run. The sizing roster of 84 fits stays held: its brief (revision 4) carries the withdrawn span claim and has never been audited, and it is rewritten before anything runs.

### C. E29 part A: the trained GRUs hold, and the sweep's hold as much

Branch `claude/wizardly-franklin-sr2l6q`, commit 0e982c10. Read-only, 21 files, 33 CPU-seconds. The hand-built check ran first; the predictions were committed before any trained file was opened (1008de25, five minutes before the results). I recomputed the headline figures from the per-unit arrays.

- Untrained: no gated holder in any seed. Born: 41 to 46% of units in the unseen seeds. After a life: 46 to 63%.
- **P29.1 holds:** 44.9, 46.1, 43.8, 41.0%, against a mark of 15%. Training at this task makes holding units out of a network born with none.
- **P29.2 misses:** sweep-born networks have as many holders as backprop-born ones (44.9 against 46.1; 43.8 against 41.0).
- **P29.3 is not judged.** My wording, "the two ... than the two", can be read as each above each (one of three seeds) or as the means (two of three), and the two readings give opposite outcomes. The fault is in the prediction as I wrote it, and I do not choose a reading after seeing the figures. Both are on file.

A descriptive note, not registered: within a birth, the life updated by backprop ends with more units of retention at least one half than the life updated by the sweep in five of six comparisons, by 5 points or more in four (backprop-born: +9.3, +5.1, +8.6; sweep-born: +3.1, -5.5, +7.8). It is a small difference beside the size of what both have.

**The worker's caution, which I adopt:** in every trained file 251 to 256 of 256 units are rewritten somewhere, so "gated holder" is in effect "holds through at least 90% of the letters".

**What this means for Mr. Xu's question.** Integration by slow units is something the GRU already builds for itself, under either update. So a shortage of holding units is not the explanation of the sweep's weaker composition, at this grain. Part B, variants at birth, is rewritten in the light of B above: fixed 40 and 60 are distances where the plain network stalls, and mixing distances is a competitor that any variant has to beat.

### D. Source checks

The memo on learned feedback weights (branch `claude/charming-clarke-u9g850`, merged) and the page on timescales and delays (with E29) are with a hostile auditor. Neither steers a design until read.
