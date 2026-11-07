# MiniFly past-work review for Mr. Zhou

**Date:** 6 November 2026. **Sources:** the local work folder (`second_try_copy`, 214 experiment folders, 33 GB) and the GitHub repository `ChangweiZhou/digitalbaby` (459 commits on the default branch, 7 branches, latest push 6 November). This is a read-only review. No experiment was rerun and no frozen result was changed.

## 1. Bottom line

- **The fly is a very good associative memory and nothing more, and that is by construction.** The mushroom-body design (random sparse expansion, top-5% winner-take-all, a fading byte trace) gives every input its own unique code. That is ideal for "cue → answer" lookup and destroys the similarity structure a learner needs to generalize. Every raw-byte test of rule reuse has confirmed this.
- **No learner in the project, native or reference, has ever learned addition from raw bytes.** Not the native cores (0% taught, 0% held), not small GRUs (100% taught, 0–6% held), not a 1,024-candidate evolutionary search, not backpropagation-through-time at 65,536 records (7 of 1,536 held sums, 0 of 384 carry pairs). The only "addition" successes used an engineered numeric representation supplied by the experimenter.
- **The genuine positive results are narrow and come from different learners.** Durable fact storage, recall under new renderings, a one-bit class rule reused on untaught pairs, memory protection, consequence-driven action choice, and a successor-map operator each worked in its own harness. They have never been shown together in one continuing learner, and the project's own 5 November memos say so.
- **The project is stuck for process reasons as much as scientific ones.** Roughly 220 numbered versions were produced in nine weeks by three AI agents on parallel branches. Most ended "NO_ADVANCE" by a few hundredths of a bit, each followed by a new mechanism rather than a diagnosis. The adopted core has not changed since 11 October. The two clearest diagnoses in the record (17 October and 3 November) were never acted on.
- **The way out is not another mechanism.** Reopen the fixed sensory front end, give the learner an ordered recurrent state and time to compute, let a gradient reference establish that each task is solvable before testing a local rule, and run one learner through one continuous life. Details in section 6.

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
