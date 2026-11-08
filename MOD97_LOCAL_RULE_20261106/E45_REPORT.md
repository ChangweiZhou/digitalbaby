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

# E7: de-confounded composition with four operators (3 births, 15 runs)

Prediction on file before launch. Four single-step operations mod 97 (add `+`/`r`, sub `-`/`t`, max `>`/`x`, min `<`/`n`) in development with left noise (held 94–100%). Life teaches 12 of the 16 ordered operator pairs (2,000 triples each) so every operator appears in both positions, and holds out the four cross pairs (+−, −+, ><, <>). 18-byte window, hidden 256, 8,192 updates.

| Arm | Within (taught pairs, new triples) | +− | −+ | >< | <> | Held mean | Single ops after life |
|---|---|---|---|---|---|---|---|
| **SCR-bp plain scratchpad, backprop** | 92 · 94 · 94 | **90 · 84 · 91** | **87 · 84 · 87** | **94 · 97 · 97** | **94 · 95 · 96** | **91 · 90 · 93** | 0 |
| SCRc-bp canonical, backprop | 95 · 96 · 96 | 94 · 75 · 25 | 16 · 91 · 68 | 95 · 5 · 4 | 36 · 3 · 39 | 60 · 43 · 34 | 27 · 63 · 54 |
| SCR plain scratchpad, sign rule | 42 · 42 · 39 | 7 · 8 · 6 | 9 · 12 · 7 | 66 · 64 · 64 | 70 · 64 · 70 | 38 · 37 · 37 | 0 |
| SCRc canonical, sign rule | 80 · 86 · 87 | 70 · 1 · 2 | 50 · 2 · 2 | 82 · 1 · 82 | 67 · 26 · 4 | 67 · 7 · 23 | 0 |
| SCRc-sh canonical, shuffled finals | 0 · 5 · 2 | 1 · 0 · 1 | 1 · 1 · 1 | 1 · 2 · 0 | 0 · 3 · 1 | 1 · 2 · 1 | 1 |

Frequency floor on every held panel ≤1.2%. Exact lookup 0 by construction (new triples, untaught pairs).

| Prediction | Outcome |
|---|---|
| P7.1 canonical, sign rule ≥70 held mean | **Refuted**: 67 · 7 · 23, seed-unstable; the second-step prompt is written correctly for 58–90% of held cases in the good seed and 1–5% in the others. |
| P7.2 plain, sign rule ≥40 | **Just missed**: 37–38, with the max/min pairs at 64–70% and the add/sub pairs at 6–12%; within-type is only 39–42%, so the local rule is under-trained on the twelve-pair set at this dose. |
| P7.3 shuffled <10 | Held. |
| P7.4 canonical backprop ≥85 | **Refuted for the canonical form** (34–60, with individual pairs from 3% to 95%: the rewriting of the second operator's code generalizes inconsistently). **The plain scratchpad with backprop was not predicted and is the result: 90–93% held mean, every held pair 84–97% in every seed.** |

**What E7 settles.** Composition across untaught operator pairs is achieved once the factors vary independently in training. With backprop, the plain scratchpad answers 84–97% of two-step problems whose operator pair was never taught, from one-step development and twelve taught pairs, with the intermediate emitted as bytes. The four earlier failures were the confound in my E5 design, not an architectural limit, and the design principle is now on record: a factor to be composed over must vary independently in training. The canonical rewrite, which was the stronger idea on paper, is the weaker form here, because writing the second operator's code is itself a generalization that the network does not make consistently; emitting the intermediate and continuing is enough.

Under the local rule the result is partial and under-trained: 37–38% on held pairs and 39–42% within-type at 8,192 updates on a teaching set six times larger than E5's. By the rule on file the next step tests the local rule at width 512 and at double dose (E7-b, predictions on file).

**Evidence level.** Development results, three seeds, one harness; backprop ceiling and shuffled control present; shortcut floors ≤1.2%. Two-step problems under one modulus; longer chains and nesting untested.

# E7-b: the local rule on de-confounded composition, wider and longer (3 births, 6 runs)

Predictions on file before launch. Plain scratchpad, sign rule, same twelve taught pairs and four held pairs as E7.

| Arm | +− | −+ | >< | <> | Held mean | Within-type | Still rising at the end? |
|---|---|---|---|---|---|---|---|
| width 256, 8,192 updates (E7) | 7 · 8 · 6 | 9 · 12 · 7 | 66 · 64 · 64 | 70 · 64 · 70 | 38 · 37 · 37 | 42 · 42 · 39 | yes |
| width 256, 16,384 updates | 13 · 15 · 22 | 17 · 20 · 31 | 66 · 73 · 74 | 74 · 72 · 76 | 42 · 45 · 50 | 46 · 46 · 53 | yes |
| **width 512, 8,192 updates** | 47 · 30 · 46 | 41 · 26 · 49 | 81 · 76 · 86 | 85 · 78 · 89 | **63 · 52 · 68** | 70 · 60 · 69 | yes (peak = final) |
| backprop, width 256, 8,192 (E7) | 90 · 84 · 91 | 87 · 84 · 87 | 94 · 97 · 97 | 94 · 95 · 96 | 91 · 90 · 93 | 92 · 94 · 94 | — |

| Prediction | Outcome |
|---|---|
| P7b.1 width 512 ≥60 held mean in every seed | **Two of three** (63, 68; one seed 52). |
| P7b.2 width 256 at 16,384 ≥50 in every seed | **One of three** (50; 42, 45). |

**What E7-b settles.** The local rule composes across untaught operator pairs once the factors are de-confounded, and it is not at a ceiling: width raises the held mean from 37–38 to 52–68, dose from 37–38 to 42–50, and every curve is still rising at its last checkpoint, with within-type accuracy at 60–70% against backprop's 92–94%. The max/min pairs compose at 76–89% under the local rule; the add/sub pairs, which need the carry, lag at 26–49%. By the rule on file neither prediction held in every seed, so the program proceeds to step 5 (the persistent-life demonstration) with compositions as a reported measure rather than a gate; the obvious remaining knob, width 512 at double dose, is a one-afternoon run that the rule did not pre-register and is left as a proposal.

**Standing of P4.** Achieved under backprop in the de-confounded regime (84–97% on untaught pairs). Partially achieved under the local rule (52–68% at width 512 and still rising), with the gap to backprop open. The learned-feedback arm that Mr. Zhou's clarification permits is the designated next test of that gap, after step 5.

# E8: the persistent-life demonstration (3 births, 9 runs)

Predictions on file before launch. One learner (18-byte window, hidden 512) developed by backprop on four operations mod 89; life by the sign-symmetric rule: eight new operations in sequence (add, sub, max, min mod 97, then mod 73), 6,144 updates and 25% of pairs each, then a ninth stage teaching two-step compositions of the mod-97 operations (the twelve taught operator pairs of E7, plain scratchpad, 8,192 updates). Every operation's sealed panel scored after every stage. Arms: C store + self-generated replay; A no replay; D backprop on joint ground truth. Full stage-by-stage tables in `exp45/SUMMARY_E8.md`.

| Sealed exact (%), three seeds | add97 | sub97 | max97 | min97 | add73 | sub73 | max73 | min73 | Held pairs | Within |
|---|---|---|---|---|---|---|---|---|---|---|
| **C after the eighth operation** | 86 · 86 · 87 | 85 · 83 · 86 | 97 · 96 · 94 | 96 · 98 · 97 | 80 · 78 · 76 | 83 · 75 · 81 | 96 · 97 · 94 | 98 · 98 · 95 | — | — |
| **C after the composition stage** | 83 · 81 · 83 | 68 · 65 · 60 | 89 · 93 · 92 | 90 · 92 · 94 | 65 · 49 · 50 | 59 · 48 · 58 | 90 · 90 · 92 | 93 · 91 · 92 | **55 · 52 · 48** | 61 · 58 · 52 |
| A after the composition stage | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 18 · 26 · 23 | 20 · 30 · 28 |
| D after the composition stage | 93 · 95 · 94 | 92 · 94 · 94 | 98 · 98 · 96 | 98 · 99 · 99 | 88 · 91 · 88 | 88 · 93 · 88 | 98 · 98 · 96 | 98 · 98 · 97 | 89 · 91 · 91 | 91 · 92 · 93 |

| Prediction | Outcome |
|---|---|
| P8.1 C: all eight ≥70 after the ninth stage; A first four <10 | **Held through the eighth stage** (every operation 75–98%, no replay 0). **Missed after the ninth**: the composition stage pulls the four carry-type operations (add and sub, both moduli) to 48–68% while max and min hold at 89–94%. A's first four are 0 at every stage. |
| P8.2 C: held pairs ≥40, within ≥55 | **Held pairs held** (48–55, against 18–26 without replay and 89–91 for backprop). Within-type 52–61, two of three seeds above 55. |
| P8.3 D ≥85 everywhere | Held (88–99; held pairs 89–91). |
| P8.4 development operations lost without replay | As predicted in every arm (23–38% in C and D, 0 in A). |

**What E8 settles.** One learner, one life, local plasticity: eight operations learned in sequence are all held at 75–98% on never-taught inputs after the eighth, where the same learner without replay holds none (0%). That is the persistent-learning demonstration, properties P1, P3 and P5 in one life with the window as working state, and it is the result the project set out to show at toy scale. In the same life, compositions of the learned operations are answered on untaught operator pairs at 48–55%, against 18–26% without replay and 89–91% for the backprop ceiling; P4 is partially present in the same learner. The weakness is specific and measured: the composition stage, which fills half of every batch with two-step problems, costs the carry-type operations 20–35 points while the max/min operations hold. Under the local rule the shared path does not have the slack to absorb 24,000 composition items while rehearsing eight operations at this dose, and the operations that need a carry are the ones that give. The obvious repairs are a longer composition stage with a smaller composition share per batch, or wider capacity, both pre-registrable as a single follow-up; the learned-feedback arm remains the test of the local rule's gap to backprop on composition itself.

**Evidence level.** Development result, three seeds, one harness, no independent confirmation; the store is an exact-key episodic memory whose replayed answers are mostly correct (arm A and E4's random-answer control bound what wrong answers do); two-step compositions under one modulus; shortcut floors at or below 1.2% throughout.

# E10: the ring with innate operations (6 births, 21 runs; first run at a quarter of the development dose archived)

Predictions on file before launch; amendment on file before the rerun. The prompt is parsed by position (supplied). Learned interfaces, shared across slots: two digits → a bump over 97 positions; operator byte → a soft choice among four innate operations on bumps (add = circular convolution, sub = circular correlation, max and min from products of cumulative distributions, all exact on one-hot bumps, `exp45/QUALIFY_E10.json`); bump → two digits. Composition is wiring. A flat ablation keeps the interfaces and replaces the innate operations by a learned mixer. Development by backprop on single-step problems (32,768 updates; the 8,192-update first run did not converge and is archived). Life: 8,192 updates on two-step problems, E7's twelve taught pairs or the confounded (+,+)/(−,−) pair of E5. 20,000 parameters (ring), 58,000 (flat).

Development, single-operation held accuracy after 32,768 updates: ring 100 · 100 · 100 · 100 (seed 73129), 96–97 (116089), 62–93 (93683); flat 3–85.

| Arm | E7 held pairs (+− −+ >< <>) per seed | Confounded cross pairs (+− −+) | Within | Single ops after life |
|---|---|---|---|---|
| **Ring, single-step development only, zero-shot** | **100 100 100 100** · 46 42 84 88 · 86 88 94 95 | — | 100 · 64 · 90 | 100 · 78 · 97 |
| Ring, life on 12 pairs, backprop | 100 100 100 100 · 36 33 88 92 · 60 62 93 95 | — | 100 · 65 · 77 | 100 · 75 · 87 |
| Ring, life on 12 pairs, sign rule | 100 100 100 100 · 14 15 47 50 · 24 22 54 57 | — | 100 · 32 · 37 | 100 · 37 · 46 |
| **Ring, life on the confounded pairs, backprop** | — | **100 100** · 67 62 · 71 68 | 100 · 66 · 68 | 99 · 64 · 84 |
| Ring, life on the confounded pairs, sign rule | — | 100 100 · 22 20 · 24 23 | 100 · 22 · 24 | 58 · 22 · 25 |
| Flat ablation, life on 12 pairs, backprop | 5 4 10 9 · 5 4 15 28 · 3 4 10 10 | — | 8 · 13 · 6 | 7 · 21 · 8 |
| Flat ablation, confounded pairs, backprop | — | 7 6 · 5 5 · 4 6 | 7 · 5 · 6 | 4 · 5 · 5 |

| Prediction | Outcome |
|---|---|
| P10.1 zero-shot ≥90 on every panel | **Held in the two seeds whose development converged** (100 and 91 held-pair mean; 65 in the seed whose single operations were at 62–93%). Composition accuracy tracks interface accuracy. |
| P10.2 confounded pairs: untaught cross pairs ≥90 | **Held in the converged seed under both rules** (100%); 62–71% under backprop and 20–24% under the sign rule in the others. No de-confounding was needed where the interfaces were exact. |
| P10.3 flat ablation <30 confounded; ≥70 on 12 pairs | Confounded held (4–7%). The 12-pair half is unqualified: the flat model's interfaces did not develop (3–85%), so it cannot serve as the E7 reference; E7 itself is that reference (84–97% for the window net). |
| P10.4 sign rule within 5 of backprop | **Refuted in two seeds.** Life under the sign rule at decay 1.0 erodes the interfaces (zero-shot 65 and 91 fall to 32 and 39); backprop life also loses ground (91 → 77). Where the interfaces were exact, nothing moves. |

**What E10 settles.** Composition across untaught operator pairs is free once the operations are innate and the interfaces are exact: zero-shot 100% on every panel from single-step development alone, 100% on the confounded cross pairs that defeated every learned architecture in E5, E5-C and E6-route, with a flat ablation at 4–10%. The ring converts the composition problem into an interface-learning problem, and interface accuracy sets composition accuracy (62–93% single-step gives 42–88% two-step). This is the project's own idea, composition built into geometry as the central complex composes vectors, confirmed at toy scale with the parsing supplied. Two limits are measured: the interfaces are slow to develop through a 97-way bump (four times the dose of the window net) and seed-sensitive; and life under the sign rule at the decay used elsewhere degrades exact interfaces rather than refining them, so a life rule on this architecture needs a lower decay or a gate that writes only when wrong, which the self-adjusting-rule allowance permits.

**Evidence level.** Development result, three seeds, one harness; parsing by position supplied, which is the next thing to learn; mod-97 arithmetic with four operations; shortcut floors not needed for a zero-shot result but the flat ablation and the confounded design serve as the controls.

## E11: a diagonal memory cell with a time-local trace inside the E8 learner (7 November 2026)

Predictions: `E45_PREDICTIONS.md`, section E11 and its two amendments. Script: `exp11_cell.py`; summary: `exp45/SUMMARY_E11.md`; receipts `exp45/e11_*`; qualification `exp45/QUALIFY_E11.json`; shortcut ceilings `exp45/E11_SHORTCUT_CEILINGS.json`. Archived partial launches: `exp45/archive_e11_freshbuffer/`, `exp45/archive_e11_storeoverflow/`.

**Question.** Property P2, working state, under a local rule inside the persistent learner. Exp 3 showed a self-path eligibility trace through a full GRU fails (2–14%) because credit runs through other units; Codex's second sweep found that for a diagonal cell the trace equals backprop through time. E11 adds such a cell to the E8 learner and asks whether the time-local rule (a) re-uses holding developed at birth, (b) acquires holding that birth never required, and (c) keeps it through a sequence of four operations with replay.

**Learner.** E8 window net (hidden 512, 10-byte window) plus a diagonal memory cell of 512 units: `p = tanh(Wp z1)` (64-d), `c = tanh(Wc p + bc)`, `r = sigmoid(Wr p + br)`, `h_t = r ⊙ h_{t−1} + (1 − r) ⊙ c`, readout `z2 = tanh(W2 z1 + Wm h + b2)`. Each unit reads only its own past state and the gates do not read `h`, so the per-synapse trace `e_t = r_t e_{t−1} + ∂h_t/∂w` (one scalar per synapse, carried forward) makes `Σ_t L_t e_t` the exact gradient for the cell's synapses, with `L_t` the sign-feedback learning signal from the readout at step `t`. Qualification: trace versus backprop through time, maximum relative error 1e-6 on all four cell tensors; within-step rule 42–47% off. 800,081 parameters. Cost: trace arm 190 ms per update in the smoke test against 34 (within-step) and 38 (backprop through time); in the runs, about 62 CPU minutes per 6,144-update stage against 14, so about 4.5×, which is Python loop overhead over 25 time steps, not arithmetic.

**Task.** `<code> a a <op> D1..Dk b b = c c EOS` with k uppercase letters between the operator and the second operand, k ∈ 0..12 in life. The first operand is in the window at the answer only for k ≤ 3, so k ≥ 4 needs the cell. Sealed panels at k = 0, 8, 12 and 20 (20 never presented), 400 held pairs each; development operations at k = 12 (300). Shortcut ceiling (the answer equals the visible second operand): max and min 45–54% of items at k ≥ 4, add and sub 0.5–1.5%. So for max and min only scores clearly above half count as memory.

**Births** (backprop through time, 24,576 updates, four operations mod 89): M, memory birth, k ∈ 0..12; N, no-memory birth, k ∈ 0..3.

| Birth | Seed | k=0 | k=12 | k=20 | k=12, cell silenced |
|---|---|---|---|---|---|
| M | 73129 | 90 92 99 98 | 91 93 99 99 | 93 94 99 99 | 1 1 39 25 |
| M | 93683 | 89 89 96 96 | 92 91 97 98 | 91 91 96 97 | 1 0 48 40 |
| M | 116089 | 93 89 95 95 | 92 88 96 96 | 92 89 96 95 | 0 1 43 39 |
| N | 73129 | 93 95 98 98 | 1 2 45 43 | 1 1 44 38 | 1 2 41 35 |
| N | 93683 | 93 93 99 99 | 1 0 50 51 | 1 0 50 51 | 1 1 42 42 |
| N | 116089 | 95 96 99 97 | 1 2 48 39 | 1 2 48 42 | 1 1 36 33 |

Each cell: add sub max min, exact %. P11.1 holds in full: the memory birth holds the operand at k = 12 at 88–99% and extrapolates to k = 20 within 2 points, silencing the cell drops add and sub to 0–1% (max and min to the shortcut ceiling); the no-memory birth is at 93–99% at k = 0 and 1–2% at k = 12 on add and sub.

**Life.** Four operations mod 97 (codes r t x n), a quarter of the pairs each, 6,144 updates per stage, k ∈ 0..12, in sequence. Replay: half of every batch from a buffer of 1,024 episodes per completed stage (prompt and answer as presented). Two amendments on file explain why this and not E8's store-mediated replay: with random letters every presentation is a new exact key, a stage writes ~393,000 distinct prompts into a 65,536-key store, and replay through the store was unanchored; the episode buffer is what E8's store returned when its keys fit. Arms: **T** sign rule everywhere plus the exact trace for the cell's synapses; **S** sign rule everywhere, cell synapses by the within-step rule (state detached); **D** backprop through time on the joint ground truth (ceiling). AdamW lr .001, decoupled decay 1.0 (local) / 0.3 (backprop), as E8. Every evaluation also scores the k = 12 panels with the cell's state forced to zero.

**Results: k = 12 exact % on each mod-97 operation after its own stage → after the fourth stage** (full tables with k = 0, 8, 12, 20, the development operations and the silenced audit at every stage: `exp45/SUMMARY_E11.md`).

| Arm | Seed | add97 | sub97 | max97 | min97 | mean after own | mean final | k=20 final mean |
|---|---|---|---|---|---|---|---|---|
| MT trace, memory birth | 73129 | 92→15 | 84→10 | 90→61 | 87→87 | 88 | 43 | 36 |
| | 93683 | 86→0 | 76→2 | 76→41 | 58→58 | 74 | 25 | 22 |
| | 116089 | 82→3 | 72→4 | 82→44 | 66→66 | 76 | 30 | 24 |
| MS within-step, memory birth | 73129 | 60→4 | 22→2 | 60→49 | 59→59 | 50 | 28 | 26 |
| | 93683 | 35→2 | 21→2 | 62→50 | 55→55 | 43 | 27 | 25 |
| | 116089 | 25→2 | 18→2 | 44→43 | 52→52 | 35 | 25 | 18 |
| MD backprop through time, memory birth | 73129 | 94→89 | 90→88 | 96→96 | 96→96 | 94 | 92 | 92 |
| | 93683 | 90→90 | 92→90 | 95→95 | 96→96 | 93 | 93 | 92 |
| | 116089 | 96→92 | 95→95 | 96→97 | 98→98 | 96 | 95 | 95 |
| NT trace, no-memory birth | 73129 | 90→1 | 34→1 | 52→45 | 55→55 | 58 | 26 | 25 |
| | 93683 | 78→0 | 30→0 | 55→47 | 48→48 | 53 | 24 | 22 |
| | 116089 | 63→1 | 41→1 | 70→42 | 55→55 | 57 | 25 | 22 |
| NS within-step, no-memory birth | 73129 | 21→0 | 1→1 | 48→38 | 52→52 | 30 | 23 | 23 |
| | 93683 | 32→0 | 1→2 | 51→48 | 50→50 | 33 | 25 | 24 |
| | 116089 | 5→1 | 1→0 | 45→41 | 48→48 | 25 | 22 | 21 |
| ND backprop through time, no-memory birth | 73129 | 91→92 | 90→87 | 95→95 | 97→97 | 93 | 93 | 93 |
| | 93683 | 92→90 | 88→92 | 96→96 | 96→96 | 93 | 94 | 94 |
| | 116089 | 94→91 | 94→94 | 95→96 | 97→97 | 95 | 94 | 95 |

Shortcut ceiling for max and min at k ≥ 4 is 45–54% (answer equals the visible second operand); every max/min score of the within-step arms, and every max/min score after the fourth stage except MT 73129's max, sits at that ceiling and is not memory. Add and sub have no shortcut (0.5–1.5%).

**Acquisition of holding, k = 12 on add97 after its stage (the first stage, no replay yet), and k = 20 (never presented):**

| Birth | Trace T | Within-step S | Backprop through time D |
|---|---|---|---|
| Memory birth M, k=12 | 92 / 86 / 82 | 60 / 35 / 25 | 94 / 90 / 96 |
| Memory birth M, k=20 | 88 / 78 / 76 | 50 / 30 / 17 | 93 / 90 / 96 |
| No-memory birth N, k=12 | 90 / 78 / 63 | 21 / 32 / 5 | 91 / 92 / 94 |
| No-memory birth N, k=20 | 78 / 65 / 52 | 14 / 25 / 2 | 91 / 90 / 94 |

Cell silenced at k = 12: ≤ 6.5% in every arm and stage wherever the live score is ≥ 50% (`SUMMARY_E11.md`).

**Predictions.**
- P11.1 (births): holds in full.
- P11.2 (memory birth): MT ≥ 70% after its own stage holds for add, sub and max in every seed and for min in one (87 / 58 / 66); MS at least 15 points below MT holds for add (32–57 points) and sub (54–62) in every seed, for max in two (14 / 38 / 28) and is moot for min, where MS sits at the shortcut ceiling; MD ≥ 85% holds. The memory birth does discriminate the rule.
- P11.3 (no-memory birth, acquisition): NT ≥ 40% at k = 12 holds for add in every seed (90 / 78 / 63) and max in every seed (52 / 55 / 70, only modestly above the ceiling), for sub in one (34 / 30 / 41) and not for min (at the ceiling). NS ≤ 15% misses in two seeds for add (21%, 32%): the within-step rule picks up a little holding, presumably by closing the gate at the '=' byte, where the instantaneous signal can see it. ND ≥ 70% holds (88–97%).
- P11.4 (extrapolation within 10 points at k = 20): holds for both backprop arms (1–3 points) and for the trace arm's add and sub from the memory birth (4–9 points); misses by 1–6 points for acquired holding (NT add: 12 / 13 / 11) and MT max (3 / 11 / 16). Holding developed at birth extrapolates; holding acquired in life extrapolates slightly less.
- P11.5 (retention within 15 points after the fourth stage): **refuted in every seed and both births.** Add and sub fall from 72–92% after their stages to 0–15% after the fourth; the development operations mod 89 fall from 88–99% to 1–41% during the first life stage. The backprop ceiling on joint data keeps every operation at 87–98%, so the task permits retention; this learner under the sign rule, with half-batch episode replay, does not deliver it.
- P11.6 (silenced audit): holds.

**What E11 establishes.** Property P2, working state, by a local rule: a diagonal memory cell with a per-synapse trace carried forward in time acquires, from a birth that never required it, the ability to hold an operand across twelve distractor bytes (63–90% at k = 12, 52–78% at k = 20 never seen), where the within-step rule reaches 5–32% and backprop through time 91–94%; and it re-uses holding developed at birth at 82–92%. The trace is exact for this cell (qualification) and needs no backward pass through time and no stored activations; the cost in this implementation is about 4.5× per update, which is loop overhead that a parallel scan removes.

**What E11 refutes.** That the E8 learner plus this cell keeps what it learns across a sequence of operations under the sign rule with half-batch episode replay. The forgetting is far worse than E8's (E8 held eight operations at 75–98%; E11 holds the current operation and little else), in both births, and the ceiling shows it is not capacity. E11 does not diagnose the cause. The candidates, each one arm: (1) weight decay 1.0 per update, which E12-B suggests costs 12–15 points of E8 retention by itself and which E10 found erodes exact interfaces; (2) the cell as an interference channel, since the new operation's gates change the state statistics the old readout depends on, and add at k = 0 (where the window alone suffices) fell faster than at k = 8–12 in two seeds, so the window path is overwritten before the cell path; (3) fixed-letter episodes being learned by rote rather than maintaining the rule, testable by replaying fresh renders of the stored pairs with the stored answers; (4) blocked presentation, since E9 showed interleaving holds everything while the stream is stationary. E13 Part A (decay versus averaging) bears on (1) directly.

**Harness lessons, on record.** Two restarts: a buffer-fill slip caught only after the first stage (a dry run of the full loop, now `E11_STAGE_UPDATES`, would have caught it in two minutes), and a mechanism dependency that no dry run catches: E8's store-mediated replay needs a stage's episodes to fit the store, which random letters break. Rule for the next experiment: list every mechanism carried over and the condition it needs in the new task, in the predictions file, before launch.

## E11-D: why the trace learner forgets (7 November 2026)

Predictions and mechanism audit: `E45_PREDICTIONS.md`, E11-D. Four one-change arms on the E11 memory-birth trace learner (arm MT), three seeds; receipts `exp45/e11_life_MT_wd0.25_*`, `_fr_*`, `_int_*`, `_wd0.25_fr_*`; tables in `exp45/SUMMARY_E11.md`. The trace gradient was computed by the backward scan that is algebraically identical to the forward trace (qualification: 1e-6), so the updates are E11's; the stage-1 scores reproduce E11's within a point (float-level divergence).

**After the fourth stage, exact % at k = 12 (add sub max min), and the add+sub mean:**

| Arm | 73129 | 93683 | 116089 | add+sub mean |
|---|---|---|---|---|
| E11 baseline: decay 1.0, episode replay, blocked | 15 10 61 87 | 0 2 41 58 | 3 4 44 66 | 13 / 1 / 4 |
| decay 0.25 | 64 48 88 96 | 57 49 72 97 | 45 53 79 93 | 56 / 53 / 49 |
| fresh-render replay (decay 1.0) | 52 39 87 94 | 14 12 74 85 | 4 1 49 69 | 46 / 13 / 2 |
| interleaved, no replay (decay 1.0) | 78 70 96 94 | 15 12 74 64 | 12 16 65 72 | 74 / 13 / 14 |
| decay 0.25 + fresh-render replay | 72 65 92 96 | 68 62 90 96 | 62 52 92 95 | 69 / 65 / 56 |
| backprop ceiling (E11 MD) | 89 88 96 96 | 90 90 95 96 | 92 95 97 98 | 89 / 90 / 94 |

At k = 0 after the fourth stage, add/sub: decay 0.25 alone 15/16, 32/27, 29/46; combined 64/61, 64/58, 60/53; baseline 6/4, 2/4, 6/9. At k = 20: combined 58/54, 63/58, 58/47. Cell silenced at k = 12: ≤ 5% on add and sub in every arm. Shortcut ceiling for max and min, 45–54%, as E11.

**Verdicts.**
- P11D.1 (decay 0.25 the dominant cause): the effect is the largest in the table, from 0–15% to 45–64% on add and sub, but the literal gate of ≥ 50% on both in every seed is met in only one seed (73129 add 64 / sub 48 misses on sub; 93683 57 / 49; 116089 45 / 53). The birth's add89 is held at 44–46% after the first life stage (27–41 before); the other birth operations are not, since nothing replays them. Dominant cause confirmed; gate narrowly missed.
- P11D.2 (fixed-letter episodes): refuted as a cause on its own; +33 points in one seed, +12 and −2 in the others.
- P11D.3 (blocked presentation): refuted. Interleaving all four operations holds everything at 70–96% in one seed and never acquires add or sub (12–16%) in the other two, in any of the four blocks. For this learner the sign rule does not learn four competing carry-type and comparison operations from a mixed stream reliably; E9's window net did, with eight.
- P11D.4 (combined at least as good as the better single arm): holds in every seed on add, and on sub within one point.

**What it means.** Decay 1.0 per update was the main reason the trace learner forgot, as E13 found the same day for the window net; with decay 0.25 the learner keeps add and sub at about half after three further operations instead of nothing. Fixed-letter episode replay is a second, smaller cause that shows once decay is fixed: with fresh-render replay the k = 0 scores are kept (60–64% against 15–32%), so the window path stops memorizing the 1,024 fixed prompts and maintains the rule. Together they recover retention to 52–72% on add and sub, 87–96% on max and min, against a ceiling of 88–98%. The remaining gap of 20–35 points on the carry-type operations is the next target; the candidates are lower decay still (E14 Part B measures the curve on the window net), the replay share, and batch 256 (E13 Part B). Interleaving is not a substitute for replay in this learner. The settings for the cell learner going forward are decay 0.25 and fresh-render replay of stored pairs.

## E15: the fixed settings applied (7 November 2026)

Predictions and mechanism audit: `E45_PREDICTIONS.md`, E15. Settings from E11-D: decay 0.25 on local arms, fresh-render replay of stored pairs, trace by the scan form (identical to the forward trace at 1e-6; with split gates, qualification `exp45/QUALIFY_E11_g2.json`). Three seeds each. Tables: `exp45/SUMMARY_E11.md` (E11-D section) and the receipts named below.

**Part A, interleaved stream at decay 0.25** (`exp45/e11_life_MT_wd0.25_int_*`). The four mod-97 operations presented together in one stream of 4 × 6,144 updates, no replay, memory birth, scored after every block. Exact % at k = 12 (add sub max min) after block 1 and after block 4, and k = 20 after block 4:

| Seed | After block 1 | After block 4 | k = 20 after block 4 | E11-D at decay 1.0, after block 4 |
|---|---|---|---|---|
| 73129 | 89 89 98 97 | 85 89 98 96 | 84 86 96 96 | 78 70 96 94 |
| 93683 | 86 89 95 96 | 87 91 98 97 | 87 89 97 96 | 15 12 74 64 |
| 116089 | 78 83 91 96 | 87 85 94 98 | 86 83 92 97 | 12 16 65 72 |

P15.1 holds in every seed. The mixed-stream failure of E11-D was decay, not the rule or the cell: at decay 0.25 the sign rule with the trace learns four competing operations from one stream to 83–98%, within 2–10 points of the backprop ceiling, and the two seeds that had learned nothing of add and sub at decay 1.0 are at 85–91% after the first block.

**Part B, the no-memory birth at the fixed settings** (`exp45/e11_life_NT_wd0.25_fr_*`): acquisition of holding and its retention in one blocked life with fresh-render replay. Exact % at k = 12:

| Seed | add after its stage (E11 NT at decay 1.0) | After the fourth stage: add sub max min | add+sub mean | k = 20 add sub |
|---|---|---|---|---|
| 73129 | 85 (90) | 44 33 89 96 | 38 | 32 25 |
| 93683 | 88 (78) | 39 31 82 93 | 35 | 29 22 |
| 116089 | 84 (63) | 55 42 87 94 | 48 | 42 38 |

P15.2: acquisition ≥ 60% holds in every seed (84–88%, and no longer seed-dependent as at decay 1.0); the retention gate of add+sub mean ≥ 40% after the fourth stage holds in one seed (48) and misses by 2–5 points in two (38, 35), against about 1 for the same births at decay 1.0. Holding acquired in life is kept at roughly half the rate of holding developed at birth (E11-D combined arm: 69 / 65 / 56).

**Part D, the ring at decay 0.25** (receipts `exp45/e10_life_ring_sign_d32768_wd0.25_*`, `..._ring_conf_sign_d32768_wd0.25_*`; births unchanged). Single-operation interface accuracy (+ − > <, exact %) at birth and at the end of 8,192 updates of sign-rule life on the taught pairs:

| Seed | Birth | Life at decay 1.0 (E10) | Life at decay 0.25 |
|---|---|---|---|
| 73129 | 100 100 100 100 | 100 100 100 100 | 100 100 100 100 |
| 93683 | 63 62 93 93 | 25 26 49 49 | 67 64 92 92 |
| 116089 | 96 96 97 97 | 36 33 55 59 | 81 81 96 95 |

Held-pair composition at the end of life at decay 0.25: 100/100/100/100, 45/42/88/90, 65/65/93/94 (birth zero-shot 100/100/100/100, 46/42/84/88, 86/88/94/95). Confounded-pair arm at decay 0.25: 100/100, 70/66, 72/75 on the cross pairs (decay 1.0: 100/100, 22/20, 24/23), and the comparison interfaces that the confounded life never trains stay at 100, 47/48 and 84/85 (decay 1.0: 17/14, 11/13, 13/10). P15.4 holds in seeds 73129 and 93683 (no erosion; 93683's carry interfaces rise 4 points) and misses in 116089 (15 points lost on the carry interfaces). The ring now composes by wiring and learns by the local rule in one life; the residual erosion in one seed points at a lower decay or a gate that writes only when wrong.

**Part C, split write and read gates:** births complete (92–99% at k = 12, as the single-gate births); the fixed-settings lives are running and are filed as an addendum below when they finish.

**What E15 establishes so far.** With decay 0.25 the local-rule learner (a) learns a mixed stream of four operations to within a few points of backprop, which removes the last reason to doubt that the sign rule can learn from an interleaved stream such as text; (b) acquires working state from a birth that never required it in every seed, at 84–88%; (c) on the ring, learns taught compositions by the local rule without eroding the innate interfaces in two seeds of three. The remaining targets are retention of holding acquired in life (35–48% add+sub after three further operations), and the one seed of ring erosion.

**Part C addendum, split write and read gates** (`exp45/e11_dev_M_g2_*`, `exp45/e11_life_MT_g2_wd0.25_fr_*`). Births by backprop through time reach 92–99% at k = 12, as the single-gate births. The fixed-settings life, exact % at k = 12 after the fourth stage (add sub max min), with the single-gate learner at the same settings (E11-D combined arm) for comparison:

| Seed | Split gates | add+sub mean | k = 20 add sub | Single gate (E11-D) | add+sub mean | k = 20 add sub |
|---|---|---|---|---|---|---|
| 73129 | 59 61 93 96 | 60 | 41 45 | 72 65 92 96 | 69 | 58 54 |
| 93683 | 55 46 87 94 | 51 | 42 31 | 68 62 90 96 | 65 | 63 58 |
| 116089 | 60 53 91 97 | 56 | 42 39 | 62 52 92 95 | 56 | 58 47 |

P15.3 is refuted: the split gates are 9–14 points worse in two seeds and equal in one, and extrapolation to k = 20 drops 14–20 points below k = 12 (single gate: 0–14). Codex's shelf result for separate gates does not carry into this learner at this scale; the single tied gate stays. The silenced audit holds (≤ 5% on add and sub).

**E15 verdicts.** P15.1 holds (mixed stream at decay 0.25, every seed). P15.2 holds on acquisition, misses the retention gate by 2–5 points in two seeds. P15.3 refuted. P15.4 holds in two seeds, misses in one.

## E16: the integrated toy learner, the Phase A exit gate (8 November 2026)

Predictions and mechanism audit: `E45_PREDICTIONS.md`, E16. One E11 cell learner from the memory births at the adopted settings (decay 0.25, fresh-render replay at half of every batch, scan trace), on one stream in six blocks of 6,144 updates: F1 the mod-97 operations with 0–12 distractor letters (interleaved), F2 the twelve taught operator pairs in E8 scratchpad form, F3 the mod-73 operations with distractors. Arm T is the learner, arm D a backprop-through-time twin at decay 0.3 on batches drawn with equal weight from the families seen so far. Three seeds (73129, 93683, 116089). Tables: `exp45/SUMMARY_E16.md`; receipts `exp45/e16_life_{T,D}_{seed}.json`; qualification `exp45/QUALIFY_E16.json`; dry receipts `exp45/archive_e16_dry/`. Wall time 80 minutes for the six lives.

**Each family is acquired by the local rule close to the twin while it is current.**

| Family, scored when its blocks end | T (learner) | D (twin) |
|---|---|---|
| F1 mod-97 operations, k = 12, after block 2 | 93.2 · 92.2 · 86.1 | 93.9 · 93.8 · 94.1 |
| F2 held cross pairs, final answer, after block 4 | 87.0 · 84.3 · 87.0 | 93.8 · 91.2 · 93.7 |
| of which the carry pairs +- / -+ | 82/84 · 75/75 · 77/80 | 88/90 · 86/83 · 91/86 |
| F3 mod-73 operations, k = 12, after block 6 | 90.9 · 90.2 · 82.3 | 93.8 · 92.6 · 93.4 |

The composition result is new: inside one learner that also holds operations with distractors, the sign rule composes the four never-taught operator pairs at 84–87% after two blocks, the carry pairs at 75–84% against shortcut ceilings of 1–2%, within 4–9 points of the twin.

**Earlier families are lost when the stream moves on; the twin loses nothing.**

| After block 6 | T | D | T/D | T after its own blocks |
|---|---|---|---|---|
| F1 operations, k = 12 | 82.6 · 80.2 · 65.3 | 95.4 · 94.1 · 94.4 | 0.87 · 0.85 · 0.69 | 93.2 · 92.2 · 86.1 |
| F1 add / sub, k = 12 | 74/69 · 70/62 · 38/42 | 93/92 · 88/92 · 95/89 | | 90/91 · 86/90 · 77/79 |
| F2 held pairs, final | 56.9 · 53.2 · 46.0 | 93.3 · 92.0 · 93.8 | 0.61 · 0.58 · 0.49 | 87.0 · 84.3 · 87.0 |
| F2 carry pairs +- / -+ | 34/38 · 22/28 · 18/18 | 87/87 · 84/86 · 91/87 | | 82/84 · 75/75 · 77/80 |
| F2 held triples of taught pairs | 64 · 61 · 56 | 94 · 94 · 94 | | 89 · 89 · 87 |
| F3 operations, k = 12 | 90.9 · 90.2 · 82.3 | 93.8 · 92.6 · 93.4 | 0.97 · 0.97 · 0.88 | |

The F1 loss happens during the composition blocks (78.1 · 71.8 · 47.0 after block 4) and partly recovers during F3 (82.6 · 80.2 · 65.3), whose items have F1's form. The F2 loss happens in the two F3 blocks: the carry pairs fall from 75–84% to 18–38%, the taught pairs' held triples from 87–89% to 56–64%, while the comparison pairs stay at 72–85%, close to their 67% visible-only ceiling. In both cases the replay half of every batch carried the earlier family (one third of each batch for F1 during F2; one fifth for F2 during F3) and did not hold it. The cell is still what holds: silencing it at k = 12 leaves 3–7% on the operations and 4–7% on the compositions, and k = 20 is within 2–9 points of k = 12. The development operations (mod 89, not replayed) fall to 9–52% in both arms, as in E8.

**Verdicts.** P16.1 (coexistence gate, 80% of the twin on every family) refuted: F3 passes (0.88–0.97), F1 passes in two seeds (0.85, 0.87) and fails in one (0.69), composition fails in every seed (0.49–0.61). P16.2 (F1 within 10 points of its block-2 value) refuted: −10.6, −12.0, −20.8, with add and sub −16 to −39. P16.3 (composition ≥ 70%, carry pairs ≥ 60% after block 4) holds in every seed. P16.4 (cell still what holds; k = 20 within 10 of k = 12) holds. P16.5 (the twin holds everything) holds: the 10-byte window is not the limit.

**What E16 establishes.** Under a local rule the toy learner acquires every family within 1–9 points of backprop while that family is current, including never-taught compositions inside a learner that also holds operands across distractors. What it does not do is keep an earlier family when the stream moves to a family of a different form: operations lose 11–21 points over four later blocks, compositions 30–41 points over two. P3 is the property that fails at the integrated scale, under the settings that gave 94% retention in E8 (one form, eight operations, window net). Two things differ from E8 and are the candidates: the replay (1,024 entries per block, uniform over the buffer, so an earlier family's share and diversity both shrink as the stream goes on; the twin draws fresh items of every seen family with equal weight) and the learner (the cell learner at window 10, which E11-D already showed forgets carry operations more than the window net). E16-D separates them.

**Evidence level.** Development result, three seeds, one harness; the twin is matched in architecture, window, births and stream content and differs in the learning rule and in how earlier families are sampled; shortcut ceilings 1–2% on the carry pairs, 67% on the comparison pairs (which are therefore not evidence of composition); the store is present and near-inert as in E11/E15. No independent confirmation.
