# E4 (retention) and E5 (composition): results against the predictions

**7 November 2026 · 6 births, 36 life runs · three seeds · predictions in `E45_PREDICTIONS.md`, written before launch**

Learner: the harness window network (10-byte window, hidden width 256, about 155,000 parameters), stateless. Development by backprop (AdamW lr .001, decay 0.3, 24,576 updates). Life by the sign-symmetric local rule through the sign-feedback autograd path of rung 3, answer-only loss, AdamW lr .001. Sealed held panels (600 pairs per task) scored after every stage. Qualification in `exp45/QUALIFY.json`.

## E4, retention: four operations learned in sequence by the local rule

Sealed held exact (%) on each life task after each stage; also the mean over the four development tasks and the taught T1 panel.

| Arm | After stage 4: add97 (T1) | sub97 (T2) | add73 (T3) | sub73 (T4) | T1 after its own stage |
|---|---|---|---|---|---|
| A shared path alone, decay 1.0 | 0 · 0 · 0 | 3 · 4 · 1 | 0 · 0 · 0 | 92 · 95 · 92 | 96 · 92 · 92 |
| A0 shared path alone, decay 0.1 | 0 · 1 · 1 | 20 · 32 · 31 | 1 · 1 · 1 | 92 · 92 · 94 | 96 · 94 · 95 |
| B + episodic store, no replay | 0 · 0 · 0 | 3 · 4 · 1 | 0 · 0 · 0 | 92 · 95 · 92 | 96 · 92 · 92 |
| **C + store + self-generated replay** | **90 · 88 · 78** | **90 · 93 · 82** | **86 · 84 · 77** | **91 · 93 · 86** | 96 · 92 · 92 |
| C′ + store + replay with random answers | 0 · 1 · 1 | 1 · 3 · 1 | 0 · 1 · 1 | 85 · 78 · 68 | 96 · 92 · 92 |
| D backprop on joint ground truth (ceiling) | 98 · 98 · 95 | 97 · 99 · 97 | 93 · 95 · 94 | 94 · 94 · 95 | 97 · 95 · 95 |

Full stage-by-stage tables are in `exp45/SUMMARY.md`. The development tasks (mod 83 and 89) fall to near zero in every arm including D, because no arm replays or retrains them; they were not part of the life sequence.

| Prediction | Outcome |
|---|---|
| P4.1 A forgets: T1 ≥70 after stage 1, <30 after stage 4 | **Held.** 92–96 → 0. Forgetting is total: each stage erases every other task, including the development tasks, within 8,192 updates. |
| P4.2 A0 forgets less than A but <50 | **Held in direction, barely.** Decay 0.1 leaves 20–32% of the previous task (T2 after stage 4) and 0–1% of T1. Interference, not decay, is the forgetting. |
| P4.3 B's sealed T1 tracks A within 10 | **Held** (0 vs 0). The store alone also failed to protect **taught** T1 items (6–17%): a +6 logit cannot override a shared path that has been driven to a confident wrong answer. The store is useless without replay. |
| P4.4 C keeps T1–T3 ≥70 after stage 4 while learning T4 ≥70 | **Held in every seed.** T1 78–90, T2 82–93, T3 77–86, T4 86–93. Within 15 points of each task's end-of-stage value in every case. |
| P4.5 C′ tracks A | **Held.** Replaying inputs with random answers retains nothing (0–3%) and also slows acquisition of the current task (68–85 vs 92–95). |
| P4.6 D ≥85 everywhere | **Held.** 93–99. |

**What E4 settles.** Under the local rule the shared path forgets completely and immediately; neither weight decay nor an episodic store beside it changes that. What retains is self-generated replay: a buffer of earlier prompts, answered by the model itself (shared path plus store), interleaved at half of every batch. With it, four operations learned in sequence are all held at 77–93% on never-taught inputs, 5–15 points under a backprop ceiling that sees every task's ground truth jointly. The control with random answers shows the content of the replay is what matters, not the rehearsal of inputs. This is the complementary-systems claim in the project's own terms: the fast memory teaches the slow path, and that is what "persistent" costs. Property P3 is established at this scale.

## E5, composition: two-step problems from taught one-step operations

Development on single-step addition and subtraction mod 97. Life teaches compositions of two types, (+,+) and (−,−), 8,000 triples each, 8,192 updates. Held: 600 new triples of each taught type (within-type) and 600 of each untaught type, (+,−) and (−,+) (cross-type). Final-answer exact with own EOS; intermediate correctness reported for scratchpad arms.

| Arm | pp within | mm within | pm cross | mp cross | intermediate right, cross (mean) | single ops after life (add97 · sub97) |
|---|---|---|---|---|---|---|
| Z zero-shot | 0 · 1 · 0 | 0 · 0 · 2 | 0 · 0 · 0 | 0 · 1 · 0 | — | 98 · 98 |
| DIR direct, sign rule | 32 · 16 · 10 | 32 · 21 · 8 | 2 · 0 · 1 | 2 · 1 · 0 | — | 0 · 1 |
| **SCR scratchpad, sign rule** | 86 · 87 · 73 | 85 · 84 · 70 | **42 · 58 · 53** | **41 · 56 · 51** | 60 · 71 · 68 | 0 · 0 |
| SCR-sh scratchpad, shuffled finals | 1 · 1 · 1 | 1 · 1 · 1 | 2 · 2 · 0 | 1 · 2 · 1 | 4 · 2 · 2 | 0 · 0 |
| DIR-bp direct, backprop | 95 · 92 · 89 | 94 · 94 · 88 | 1 · 1 · 1 | 1 · 0 · 1 | — | 7 · 0 |
| SCR-bp scratchpad, backprop | 94 · 95 · 90 | 94 · 91 · 88 | 10 · 36 · 4 | 13 · 19 · 4 | 27 · 45 · 7 | 0 · 0 |

Shortcut floors on every panel: frequency ≤1.2%, answer from the second operation alone ≤1.3%, from the first alone ≤1.5%, exact lookup 0 by construction.

| Prediction | Outcome |
|---|---|
| P5.1 Z under 5 everywhere | **Held.** |
| P5.2 DIR cross-type <30; within-type ≥60 | **Cross-type held** (0–2%). **Within-type failed** (8–32%): the local rule learns a direct three-operand function poorly in 8,192 updates. Backprop learns it (88–95%) and still scores 1% cross-type. Direct composition does not generalize across operation types under either rule. |
| P5.3 SCR cross-type ≥70; within-type ≥80 | **Not met.** Cross-type 41–58%; within-type 70–87%. But far above direct (0–2%), and above the backprop scratchpad (4–36%), with the intermediate right in 60–71% of cross-type cases. |
| P5.4 SCR-sh under 10 | **Held** (0–2%). |
| P5.5 single ops stay ≥80 in every life arm | **Failed in every life arm.** Teaching compositions erased the single-step operations (98 → 0–7%) within the stage, by the same interference E4 measured. |

**What E5 settles.** Direct composition does not transfer across operation types at this scale under any rule: backprop fits the taught types to 95% and answers 1% of untaught combinations, which is the systematic-generalization failure of the 2018 literature reproduced cleanly. Scratchpad composition does transfer, partly: with the local rule, 41–58% of untaught combinations are answered exactly, the intermediate step is right in 60–71%, and the shuffled control is at 1%. The local rule generalizes across types better than backprop here (backprop overfits the taught types: 4–36% cross-type). The gap to the 70% prediction has a measured cause: every life arm erased the single-step operations it was composing (98% → 0–7%), the same forgetting E4 measured and fixed. The final step of a cross-type problem is the single operation the learner just lost.

## Decision

E4 passes its decision rule in all seeds: E6 uses arm C's design, store plus self-generated replay. E5 does not pass P5.3, but its failure is localized to P3 inside one stage, and E4 supplies the remedy. The pre-stated decision for a scratchpad failure was to postpone E6; the measured cause argues instead for one targeted run before E6: **E5-R**, scratchpad composition taught with self-generated replay of the single operations from the store, prediction cross-type ≥70% and single ops retained ≥80%. That run is E6's mechanism applied to E5's question, and it is the next step.

## Evidence level and limits

Development results, three seeds, one harness, no independent confirmation. The store is an exact-key episodic memory (the fully separated limit of a sparse code), chosen at qualification after a random-projection expansion failed to separate windows sharing nine of ten bytes; its replayed answers are therefore mostly correct, which is the complementary-systems premise, not a hidden oracle (arm C′ shows what wrong answers do). The window network has no recurrent state; the ten-byte window stands in for working state at this horizon. Compositions are two steps under one modulus with the intermediate visible in the window; longer chains were not tested. The development tasks were not replayed in any arm and were lost in all, which says the replay buffer must cover everything the learner is meant to keep.

# E5-R: scratchpad composition with replay of the single operations (6 runs)

Prediction on file before launch (P5R.1–P5R.4). Arm SCR-R: as SCR, with the episodic store holding the development single-step items and half of every batch being replayed single-step prompts answered by the model itself; SCR-R-bp the same by backprop.

| Arm | pp within | mm within | pm cross | mp cross | intermediate right, cross | single ops after life (add97 · sub97) |
|---|---|---|---|---|---|---|
| SCR scratchpad, sign rule (no replay) | 86 · 87 · 73 | 85 · 84 · 70 | 42 · 58 · 53 | 41 · 56 · 51 | 60 · 71 · 68 | 0 · 0 |
| **SCR-R with replay, sign rule** | 68 · 58 · 37 | 63 · 59 · 35 | **14 · 5 · 2** | **18 · 10 · 3** | 23 · 11 · 6 | **89 · 80** |
| SCR-R-bp with replay, backprop | 91 · 90 · 82 | 91 · 89 · 83 | 20 · 12 · 3 | 26 · 8 · 1 | 45 · 22 · 3 | 96 · 94 |

| Prediction | Outcome |
|---|---|
| P5R.1 single ops retained ≥80 | **Held** (89, 80 with the sign rule; 96, 94 with backprop). Replay works here as in E4. |
| P5R.2 cross-type ≥70 with intermediates ≥85 | **Refuted, in the wrong direction.** Cross-type fell from 41–58% to 2–18%, intermediates from 60–71% to 6–23%. |
| P5R.3 within-type ≥85 | **Refuted** for the sign rule (35–68%); held for backprop (82–91%). |
| P5R.4 backprop with replay below SCR-R | Not applicable as stated; both are low. Backprop with the parts retained still answers 1–26% of untaught combinations. |

**What this corrects.** My explanation for the E5 shortfall was wrong. Keeping the single operations does not help composition and, under the local rule at this budget, hurts it: half the updates go to the replay, and the two formats compete for the same network. More decisively, backprop with the parts fully retained still fails cross-type (1–26%), so composition across operation types is not limited by retention of the parts. The 41–58% that SCR reached without replay came from partial generalization of the second-step windows across the two taught types, not from the single-step skill.

**Why the window cannot compose across types.** At the final step the learner reads the ten-byte window `+81-07=08;`, which contains the first operator. In the taught types that window always has matching operators; an untaught type's window is a pattern it has never seen, and a stateless network keyed on the whole window has no reason to treat it as "08 minus 07". The intermediate is emitted correctly; it is the re-reading that fails.

**Decision, by the rule on file.** Scratchpad composition in this form is the limit at this scale, and E6 is postponed. The rule said so and it stands. The design-level reading is specific enough to test, and it is the only thing worth testing next on this property: a **canonical scratchpad**, in which the learner writes the next step as a single-operation prompt in the taught format (`08;t08-07=01` rather than `08;01`) and then solves it. The final window is then a taught single-step prompt, so cross-type transfer reduces to the established P1. That is a one-afternoon experiment (E5-C) with the prediction cross-type ≥70% and the shuffled control, and it is proposed, not started.

# E5-C: canonical scratchpad (18-byte window rerun; first run archived)

Prediction on file before launch, amendment on file before the rerun. The learner rewrites each step as a taught single-operation prompt and solves it (`c24+81-07=` → `r24+81=08;t08-07=01`). Development with 0–3 random leading letters on single-operation records, window 18 bytes in every arm. The first run, at the 10-byte window, failed in every arm including backprop because the third operand had left the window by the time the rewrite reached it (teacher-forced accuracy 8–13% at those two positions, near 100% elsewhere); its receipts are kept without the `_w18` tag.

| Arm (18-byte window) | pp within | mm within | pm cross | mp cross | rewritten prompts right, cross (step 1 / step 2) | intermediate right, cross | single ops after life |
|---|---|---|---|---|---|---|---|
| SCR plain scratchpad, sign rule | 87 · 88 · 77 | 86 · 86 · 76 | 8 · 48 · 46 | 8 · 46 · 48 | — | 21 · 67 · 68 | 0 · 0 · 0 |
| **SCRc canonical, sign rule** | 91 · 92 · 86 | 93 · 92 · 87 | **0 · 1 · 6** | **2 · 1 · 1** | 9 · 0 · 0 / 1 · 0 · 0 | 24 · 93 · 35 | 10 · 84 · 5 |
| SCRc-sh canonical, shuffled finals | 0 · 2 · 2 | 2 · 2 · 2 | 1 · 1 · 2 | 2 · 1 · 1 | 0 / 0 | 18 · 51 · 47 | 10 · 0 · 0 |
| SCRc-bp canonical, backprop | 100 · 100 · 99 | 100 · 100 · 100 | 0 · 0 · 1 | 8 · 0 · 2 | 59 · 19 · 0 / 4 · 0 · 0 | 84 · 20 · 17 | 89 · 96 · 95 |

| Prediction | Outcome |
|---|---|
| P5C.1 canonical cross-type ≥70 | **Refuted**: 0–6% with the sign rule, 0–8% with backprop, against 86–100% within-type. |
| P5C.2 rewritten prompts right ≥85 cross-type | **Refuted**: step-1 prompt 0–59%, step-2 prompt 0–4%. The rewriting itself does not generalize across operator positions. |
| P5C.3 shuffled control <10 | Held. |
| P5C.4 plain scratchpad ≤60 cross-type | Held (8–48%, seed-dependent, consistent with E5). |

**What E5-C settles.** The canonical form was the strongest reasonable design for composition on this architecture, and it fails for a clean reason: the learner does not learn "first operator → its code" and "second operator → its code" as two separable rules; it learns "(+,+) → r…r" and "(−,−) → t…t" as whole patterns, and an untaught pair of operators is a pattern it has never seen. Backprop makes the same error (step-1 prompt right in 0–59%). The intermediate is often right (up to 93%), the final step would be solvable if the prompt were written, and the prompt is not written. That is the systematic-generalization failure of the 2018 literature, reproduced at the level of a copy rule, under both the local rule and backprop, with the task's answer bytes and shortcuts fully audited.

**Decision.** Composition across operation types (P4) is not achieved at this scale in any of three forms: direct, scratchpad, canonical scratchpad. E6 stays postponed, by the rule on file. What the three forms share is a stateless window network with no mechanism that binds "the operator in slot k" to "the code for that operator" independently of what else is in the window. The one structure the survey ranked for this, attention as learned routing, is now permitted and is the natural next arm: a single learned-key layer that selects the operator slot before the rewrite. That is a design change, stated here as a proposal and not run.

**Evidence level.** Development results, three seeds, one harness; the window change is a stated amendment with the first run archived. Shortcut floors below 2% on every panel.

# E6-route: learned-key routing on the canonical scratchpad (6 births, 15 runs)

Predictions on file before launch. RouteNet: four attention heads over the 18-byte window (query from the window, keys = slot-position embeddings, value = the slot's byte embedding), output head on the four attended bytes plus a 16-d summary ("att") or the attended bytes alone ("attv"); 65,000–70,000 parameters; all projections through the sign-feedback path in life. Development with left noise reached 92–96% on single operations.

| Arm | pp within | mm within | pm cross | mp cross | rewritten prompts right, cross (step 1 / step 2) | single ops after life |
|---|---|---|---|---|---|---|
| att, sign rule | 84 · 2 · 28 | 74 · 2 · 11 | 2 · 2 · 4 | 1 · 2 · 5 | 5 · 61 · 8 / 0 · 0 · 2 | 22 · 0 · 11 |
| attv, sign rule | 55 · 14 · 15 | 49 · 14 · 16 | 2 · 1 · 0 | 1 · 0 · 0 | 44 · 9 · 3 / 1 · 0 · 0 | 0 · 0 · 0 |
| att, shuffled finals | 2 · 2 · 1 | 2 · 1 · 1 | 1 · 1 · 1 | 2 · 1 · 0 | 31 · 35 · 52 / 1 · 3 · 2 | 0 · 0 · 2 |
| **att, backprop** | 94 · 88 · 90 | 94 · 82 · 85 | **1 · 1 · 4** | **0 · 0 · 4** | 94 · 14 · 2 / 0 · 0 · 5 | 7 · 2 · 21 |
| attv, backprop | 89 · 94 · 94 | 86 · 90 · 90 | 0 · 0 · 0 | 0 · 0 · 0 | 1 · 0 · 1 / 0 · 0 · 0 | 29 · 24 · 0 |

| Prediction | Outcome |
|---|---|
| P6.1 routing with backprop ≥70 cross-type | **Refuted**: 0–4%. |
| P6.2 sign rule within 20 of backprop | Vacuous; both at the floor. The sign rule is also unstable on this architecture (within-type 2–84%). |
| P6.3 values-only ≥ summary | Refuted in direction; values-only is worse within-type and no better cross-type. |
| P6.4 shuffled <10; within ≥85 for sign arms | Shuffled held; within-type failed for the sign arms. |

**What E6-route settles.** A positional-key routing layer does not give the operator-to-code binding. With backprop it learns the taught types to 82–94% and writes the correct second-step prompt for 0–5% of untaught combinations; the first-step prompt is right in one seed only. The heads can select a slot, but the mapping from the attended byte to the code is still learned as part of a pattern that includes the other operator, because nothing forces it to be a function of the attended byte alone, and the summary (or the attention weights themselves) carry the rest. The values-only variant, which was meant to force that invariance, learns the taught types but still writes no untaught prompt correctly: positional keys select the same slots for both types, so the rest of the window still reaches the head through the attended bytes of neighbouring slots.

**Decision, by the rule on file.** Composition across operation types (P4) is not achieved at this scale in four forms: direct, scratchpad, canonical scratchpad, and canonical scratchpad with learned-key routing, under both the local rule and backprop. The composition line stops here. The honest statement for the program is that P1, P2 (in development; in life with credit through time) and P3 are established, P5 holds for those, and P4 is open with a measured boundary: every architecture tried binds an operator to its role as part of a whole-pattern, and a mechanism that makes "which operator" a variable independent of its neighbours is what is missing. That is the known content of the systematic-generalization literature, reproduced here under local plasticity with full audits, and it is where this program's symbolic line ends for this month.
