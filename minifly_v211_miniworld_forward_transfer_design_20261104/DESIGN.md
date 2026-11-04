# MiniFly V211 · MiniWorld v0 forward transfer

**Status** DESIGN_ONLY  
**Date** November 4, 2026  
**Decision** Does prior experience in a continuous sequence of small worlds reduce the number of new-world questions MiniFly gets wrong while it learns from fresh feedback?

## Rationale and executable parent

Round 15 reports that adding local word context and combining evidence in log-odds makes lexical transfer much stronger. With both names and locations changed, the development score reached 100% and ties nearly disappeared. Changing sentence form at the same time dropped it to 21%. On the renamed-word panel, random-location feedback matched true feedback at 100%, while no feedback scored 4%. These numbers support a prospective transfer test. They do not establish that true answer associations, rather than knowledge of what a location answer looks like, explain the gain.

Use the exact Round 15 context-plus-evidence-sum learner for all arms. It writes observed sentences to its existing bounded store, retrieves candidates, and scores word-identity and neighboring-word features with locally learned evidence. Use the same tokenizer, capacities, retrieval, smoothing, arithmetic and feedback update throughout. The parent has engineered word segmentation and a candidate interface. Those are part of the tested package. Resolve and hash the actual Round 15 implementation, corpus or generator, and development receipts before qualification. The attachment describes the algorithm and results but does not contain the executable, and the source was not found in the checked work folders. A reconstruction must be identified as such and independently checked against source-level fixtures or recovered receipts before it can enter this design. V209, V210 and V206 are historical comparators, with no implied inheritance.

The first MiniWorld tests movement and location questions. The memo's possession and carry/drop extension gets a separate admission after this result. Starting with one event family lets us attribute a failure to retrieval, scoring, transfer or continued plasticity before adding a new task. Actions in this experiment are learner-selected answers. World movements are exogenous, so V211 is an experience-driven QA experiment with teaching feedback. V208 remains the reference for independently acting, consequence-driven learning.

[Continual-learning strategy](https://github.com/ChangweiZhou/digitalbaby/blob/41b8c65db397ebb20541c2f162497e00ec58541c/PROJECT_DOCUMENTS/MINIFLY_EXPERIENCE_DRIVEN_CONTINUAL_LEARNING_STRATEGY_20261104.md) · [V210 rehearsal](https://github.com/ChangweiZhou/digitalbaby/blob/475a55603b01827256ad1d15ec96a8686f87ce57/minifly_v210_query_conditioned_event_binding_design_20261104/CYCLE3.md) · [V208 confirmation](https://github.com/ChangweiZhou/digitalbaby/blob/33454cc4d8489ddd8dac9d6190d8a0ca87153912/minifly_v208_consequence_confirmation_design_20261104/official/REPORT.md)

## A small continuous environment

Use 16 independent paired lives. Derive each seed from the first 64 bits of SHA-256 of MINIFLY|V211|LIFE|i for i = 0 through 15. Use separate engineering seeds. Each life visits **32 prior worlds and one target world**. Each world has two people and four places. It keeps a causal location for each person. Movements update that location and emit only the textual observation. The learner sees no simulator state, canonical identity, world number, task tag, or change signal. Across worlds in a life, draw new person and place strings without reuse. All target tokens are unseen in prior experience. The single tokenizer and memory run continuously across the 33 worlds. Store size and eviction or overwrite rules stay fixed. World starts are evaluator bookkeeping only. No memory clearing, decay reset, vocabulary refresh or slot reallocation occurs at a world boundary. Independent lives start from the same frozen birth.

Each observation packet contains two or six movement records, balanced within the world and shuffled prospectively. Both people move in each packet. Their final locations differ. Counterbalance the final mover, current places and question order. Reject immediate redundant moves and guarantee that the other person's current location is a live, same-type alternative. One paired question asks for each person's current location. Generate both answers before providing either feedback, with both queries on the same observed event history. Store the learner's own candidate records for the subsequent local update. Feedback returns the observed answer byte span for that query. Questions, emitted answers and feedback do not become movement records. This paired scoring boundary prevents the first answer from disclosing the second through feedback. It is a disclosed training interface shared by every arm.

Prior worlds have eight such packets each, yielding 16 questions per world. The target world has 32 packets and 64 questions. Use the two fixed query renderings “Where is NAME?” and “Where can NAME be found?” in equal counts, with question order balanced. The main held shift is in the observations. Use a crossed **movement frame × neutral ending** generator with these exact components:

| Frame | Movement observation before the ending |
|---|---|
| F0 | PERSON went to the PLACE |
| F1 | PERSON is now in the PLACE |
| F2 | In the PLACE is where PERSON went |
| F3 | At the PLACE is where PERSON can be found |

Append one of four endings: empty, “ today”, “ for now”, or “ at present”, followed by a period. Index frames and endings 0 through 3. Prior worlds use only pairs with even index sum. Target worlds use only pairs with odd index sum. Both sets contain every frame and ending, both entity orders, and both question renderings. Balance template counts within each phase of a life and use independent random streams for names, locations, template order, movements, questions, label swaps and tie-breaking. Audit realized byte renderings and surface coverage. This tests new combinations of familiar linguistic parts, alongside novel content. It does not ask the learner to infer entirely unseen words for movement or an unknown question operation from no evidence.

At each paired question, score the actual emitted whole answer word before feedback. Exact ties are resolved with a reproducible uniform draw from the tied set sorted by a stable identity, so reordering the list preserves the tied set and distribution. The RNG is fixed by life and question, independent of correctness and candidate order. Report strict unique-correct scores, tie rates, target availability and conditional accuracy among available targets. Record how often prior-world words survive into the target candidate set and win. Permute full candidate records, carrying retrieval votes or other operative candidate evidence with them. Exact scores and selected output at the same tie draw must agree with the unpermuted evaluation. Missing answers count as errors. This measures selection through the parent output port. A generated answer with own EOS is not an endpoint.

## Matched experience arms

| Arm | Prior 32 worlds | Fresh target world |
|---|---|---|
| PRIOR-TRUE | All movement observations and correct answer feedback | Same 64 movement-driven questions and true answer feedback as all arms |
| PRIOR-TYPE-SHAM | Identical observations, with answer assignments swapped within exactly half of the paired questions | True answer feedback |
| PRIOR-OBSERVE | Identical movements, questions and timing. No prior answer bytes or answer-evidence updates. | True answer feedback |
| FRESH | Same birth state. No prior environment exposure. | True answer feedback |

For PRIOR-TYPE-SHAM, a swapped pair exchanges the two correct answer words from the **same packet**. Prelock exactly four swapped pairs in each eight-packet prior world and balance swap assignment across frame and queried-person order across the life. Both feedback values remain current locations in the same observed world. This preserves the exact target-word multiset per packet, answer type and feedback count while eliminating systematic answer assignment in expectation. Audit if the feedback word is missing from the actual retrieved set and report applied updates for all arms. Never inject the feedback word into retrieval before scoring. Both questions run before any teaching in each packet, including under true feedback. The target stream and feedback are identical in all arms even if predictions differ. FRESH and experienced arms get the same number of target observations and question calls at each point. Their lifetime budgets differ by design. Report total prior experience and compute separately from target sample efficiency.

PRIOR-TRUE versus FRESH estimates the overall value of prior life. PRIOR-TRUE versus PRIOR-OBSERVE tests the contribution of prior supervised feedback beyond matched sensory history. PRIOR-TRUE versus PRIOR-TYPE-SHAM tests correct answer assignment beyond type and local context exposure. A matched-history advantage does not identify a single counter or feature family inside the learner. Any state attribution requires a separate intervention.

## Endpoint and decision

For each life and arm compute the average pre-feedback accuracy over **target questions 1–32**, with all 32 counted. The difference between two arms is the normalized area between their early learning curves, or equivalently the number of avoided errors divided by 32. Report differences in both units. Report first-pair accuracy before *any* new-world feedback, fixed bands 1–8, 9–16, 17–32 and 33–64, and the last 16 questions as the plasticity guard. First-pair accuracy is a separate diagnostic. The main endpoint includes online learning and is not a zero-shot score. Treat the 16 paired lives as the independent units. Show paired intervals and raw life-level scores.

**A full forward-transfer pass requires:**

- PRIOR-TRUE improves on FRESH by at least **10 percentage points** in the paired early mean. The paired two-sided 95% lower bound must exceed zero.
- PRIOR-TRUE improves on each matched-history arm by at least **5 points** in the early mean. Require one-sided lower bounds above zero for both, using Bonferroni adjustment across the two matched-history comparisons. Open this inference only after the FRESH contrast passes.
- The lower bound of the paired two-sided 95% interval for PRIOR-TRUE minus FRESH over the last 16 questions is above **−5 points**. An early advantage with a later plasticity deficit does not advance.
- Exact exposure, chronological scoring, capacity and candidate-order audits pass. Report wall time, CPU, stored state, candidate coverage and actual local writes. No tuning after the held schedules are opened.

If prior experience beats FRESH but true and type-sham feedback tie, report a history benefit without evidence that correct answer assignment mattered. If PRIOR-OBSERVE matches PRIOR-TRUE, the contrast supports observation or state carryover as a candidate explanation. If early performance improves and late performance degrades, report the trade-off. If all arms are already at ceiling, classify the result as uninformative for sample efficiency. A target whose answers fail to reach the candidate set is a retrieval failure, reported separately from a readout failure.

**Countermodels.** Report a frozen exact-query table, uniform choice among available candidates, explicit retrieval vote alone, a location-type filter and a latest-location-without-subject rule. Fit any type filter on training data only and score it without order-dependent ties. Run a canonical current-location tracker as an evaluator-only solvability witness. It can solve the task by reading current observations without prior teaching. If a fixed type/recency rule explains performance, state that limitation. This assay measures forward learning transfer on new cue and template combinations under a fixed QA interface. It does not by itself establish reusable semantic representations, autonomous exploration, or general E3/E4 capability.

## Census, limits and execution checks

Each of 16 lives has 512 prior questions and 64 target questions. With three experienced arms and four target arms, the fixed roster yields:

| Quantity | Scheduled total |
|---|---:|
| Prior arm-question applications | 24,576 |
| Target arm-question applications | 4,096 |
| All question applications | 28,672 |
| True or swapped feedback applications | 20,480 |
| Movement-record applications across arms | 57,344 |

Generate each paired movement and question schedule once. Charge every arm application and record exact bytes, prior and target supervision, missing-target writes, candidate-scoring work and any separately recomputed permutation checks. Proposed cumulative ceilings including qualification are 34,000 question applications, 24,000 feedback applications, 68,000 movement-record applications, 600 CPU seconds, 1,200 active wall seconds, 1 GiB peak RSS and 128 MiB retained outputs. The caps are launch limits, not evidence thresholds. Profile the exact production path on a technical roster. Stop at admission if it cannot fit.

1. **Refine → audit → test parent recovery.** Locate the Round 15 executable and pin transitive sources. Reproduce a bounded development row with the original protocol and audit true, type-random and no-feedback paths. Test ties under actual candidate permutations. Charge every fit. If source lineage or bounded behavioral parity cannot be established, stop before any scientific launch.
2. **Refine → audit → test MiniWorld.** Use disjoint engineering seeds. Audit cross-template exposure, observed feedback ordering, current-location witnesses, paired swaps, fixed capacity and lack of world-reset calls. Test tie cases and the reducer on known positive, type-sham-tie, late-regression and incomplete-roster examples.
3. **Refine → audit → test the production entry point.** Run a reduced continuous life through all arms. Verify learning curves from saved receipts, attribution of costs, frozen sources, trace replay and archive readback. Dry-generate and hash the full 16-life roster. Seek official authorization only after admission passes.

**Delivery** Design only. No training or run authorization is inferred from this document.
