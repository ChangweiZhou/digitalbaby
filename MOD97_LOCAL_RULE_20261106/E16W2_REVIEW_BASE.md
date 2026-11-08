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
