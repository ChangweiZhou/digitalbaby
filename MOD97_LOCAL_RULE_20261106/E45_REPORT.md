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

## E16-D: why the integrated learner forgot (8 November 2026)

Predictions, mechanism audit and the pre-result amendment on the balanced arm: `E45_PREDICTIONS.md`, E16-D. Three arm-T controls on the E16 stream, same births, seeds and settings; tables `exp45/SUMMARY_E16.md` (retention-across-arms section); receipts `exp45/e16_life_T_{oracle,balanced,all_balanced}_{seed}.json`. Nine lives, 18–19 minutes per block with nine processes, 115 minutes.

| Arm | F1 operations, k = 12, block 2 → 6 | Held compositions, block 4 → 6 | Carry pairs +- / -+ at 6 | F3 at 6 |
|---|---|---|---|---|
| E16 learner (uniform, 1,024/block) | 93.2→82.6 · 92.2→80.2 · 86.1→65.3 | 87.0→56.9 · 84.3→53.2 · 87.0→46.0 | 34/38 · 22/28 · 18/18 | 90.9 · 90.2 · 82.3 |
| balanced per family, same buffer | 92.6→82.1 · 91.5→76.9 · 87.6→69.0 | 89.0→64.8 · 85.8→55.6 · 88.6→49.1 | 38/48 · 28/28 · 25/26 | 91.1 · 89.8 · 85.8 |
| balanced, every presented entry | 93.7→89.8 · 92.5→89.2 · 90.2→77.7 | 90.2→84.5 · 87.4→79.8 · 91.4→83.3 | 76/77 · 68/66 · 74/74 | 92.5 · 89.6 · 88.3 |
| oracle: the twin's stream, local rule | 94.2→93.6 · 92.9→92.8 · 90.8→89.6 | 83.2→82.9 · 85.0→84.0 · 86.9→87.3 | 71/84 · 76/70 · 80/77 | 90.9 · 89.2 · 87.9 |
| twin (backprop, same stream as oracle) | 93.9→95.4 · 93.8→94.1 · 94.1→94.4 | 93.8→93.3 · 91.2→92.0 · 93.7→93.8 | 87/87 · 84/86 · 91/87 | 93.8 · 92.6 · 93.4 |

**The rule is cleared.** On the twin's data the sign rule with the trace loses nothing when the stream moves on: F1 within 1.2 points of its block-2 value and the compositions within 1.0 point of their block-4 value in every seed, 3–9 points below the twin in absolute level and flat where E16 fell 11–41 points. **Operand diversity in the replay is what E16 lacked.** Keeping every presented entry, with the same per-family sampling as the balanced arm, recovers the compositions to 80–85% (within 1.6–4.2 points of the oracle) and the operations to 90 / 89 / 78%; the balanced arm with 1,024 entries per block is indistinguishable from uniform, as the pre-result amendment said it would be in the block that mattered. The 1,024 replayed triples were being memorized rather than kept as a rule, the fixed-episode effect E11-D found for letters, now on operands. **One residual is share, not diversity:** in the weakest seed the operations still fall to 61% during the composition blocks under full diversity (oracle: 75–85%), where the learner's F1 exposure is 25–50% of the batch against the oracle's 50%; the share arm of E16-D2 addresses exactly that block.

**Verdicts.** P16D.1 holds (oracle ≈ twin on retention). P16D.2 withdrawn before the result (null intervention); observed null. P16D.3 holds on compositions and F3 in every seed, misses on F1 in one seed (−11.9 from the oracle). P16D.4 holds within 3–4 points; acquisition is slightly better with diverse replay, not worse.

**What follows.** E16-D2, launched 13:00 EST: the share arm (earlier families 50% of the batch during F2, 25% each during F3, current family never replayed) and arm M (backprop on exactly the learner's uniform-replay batches), which should now forget like the learner if the data are the limit. Then E16-R, the gate attempt with diverse replay under a bounded budget (a per-family reservoir, since a buffer of every entry grows with life and fails the scaling requirements), and the parser-free version of that replay as the next design step.

**Evidence level.** Development result, three seeds, one harness; the oracle arm shares the twin's stream constructor and seeds exactly; shortcut ceilings as in E16. No independent confirmation.

## E16-D2: share and rule separated, and Codex's E16-W read beside it (8 November 2026)

Predictions: `E45_PREDICTIONS.md`, E16-D2. Two arms on the E16 stream, three standing seeds; receipts `exp45/e16_life_T_share_*.json`, `exp45/e16_life_M_*.json`; tables `exp45/SUMMARY_E16.md`. Six lives, 23 minutes per block under a shared machine, 2 h 25 min.

| Arm | F1 operations, k = 12, block 2 → 4 → 6 | Held compositions, block 4 → 6 | Carry pairs +- / -+ at 6 |
|---|---|---|---|
| E16 learner (uniform, 1,024/block) | 93→78→83 · 92→72→80 · 86→47→65 | 87.0→56.9 · 84.3→53.2 · 87.0→46.0 | 34/38 · 22/28 · 18/18 |
| share: earlier families 50% of the batch during F2, 25% each during F3 | 94→79→84 · 93→69→78 · 91→47→66 | 88.5→64.5 · 84.5→54.5 · 86.0→49.5 | 38/54 · 27/28 · 24/20 |
| M: backprop on exactly the learner's batches | 92→88→92 · 93→81→90 · 92→91→90 | 93.6→74.2 · 93.2→65.4 · 93.2→76.7 | 61/53 · 45/43 · 67/69 |
| oracle (E16-D): the twin's data, local rule | 94→92→94 · 93→92→93 · 91→85→90 | 83.2→82.9 · 85.0→84.0 · 86.9→87.3 | 71/84 · 76/70 · 80/77 |
| every-entry replay (E16-D) | 94→88→90 · 93→85→89 · 90→61→78 | 90.2→84.5 · 87.4→79.8 · 91.4→83.3 | 76/77 · 68/66 · 74/74 |

**Share is not the lever.** Doubling the earlier families' exposure leaves the operations' loss during the composition blocks unchanged (15 / 24 / 44 points against 15 / 20 / 39) and trims the compositions' loss during F3 by 4–20% (24 / 30 / 37 points against 30 / 31 / 41). P16D2.1 refuted in every seed. Codex's E16-W found the same on the window net: a 25%-of-batch composition share bought 6–11 points, 14–23% of the loss.

**The rule matters on poor data and not on rich data.** Backprop on the learner's own batches keeps the operations (within 3 points of block 2) but loses 17–28 points of composition, ending 65–77% where the local rule ends 46–57% and the local rule on the twin's data ends 83–87%. P16D2.2 misses on compositions, holds on operations. So with 1,024 memorized items per block, both rules forget compositions and the local rule forgets about 20 points more; with the twin's data, both are flat. Codex's E16-W matched-batch backprop arm shows the same ordering on the window net (ends 43–50% against the local rule's 31–37% and the ceiling's 91%).

**Acquisition is unchanged by either arm** (P16D2.3 holds: share within 2 points on compositions, +0.6 to +5 on operations; arm M higher, as backprop).

**Codex's E16-W, assessed.** Twelve lives, predictions re-locked after the amendment, default parity and exact batch matching verified (`E16W_REPORT.md`; review 9.31). The window net loses 46–48 points of held composition over four later mod-73 stages under E8's replay; its mod-97 operations fall from 93–95% to 76–85% during the composition stage itself, as the cell learner's did in E16. Nothing in E16's forgetting is specific to the cell. Codex's caution stands: the matched arms show the limit of this recorded stream, not of all replay, and the ground-truth ceiling holds 91% under far richer exposure.

**What the three studies establish together (E16-D, E16-D2, E16-W).** The integrated learner forgets because its replay covers too few distinct items (1,024 per block, re-rendered or verbatim), not because the stream has too little share for earlier families, and not primarily because of the learning rule: on data with full operand coverage the local rule retains like backprop, 3–9 points lower in level. The local rule pays an extra 20 points on impoverished replay. The one intervention that worked, replaying every presented entry, is unbounded; E16-R has to find the bounded capacity (a per-family reservoir) at which operand coverage suffices, and that capacity, not a learner change, is what the Phase A gate now turns on.

**Verdicts.** P16D2.1 refuted. P16D2.2 holds on operations, misses on compositions (rule implicated on poor data, by about 20 points). P16D2.3 holds.

**Evidence level.** Development result, three seeds, one harness; arm M shares the learner's batches exactly (same buffer, same draws); E16-W is an independent harness and learner reaching the same conclusion, run by Codex, which is the first cross-harness agreement on a mechanism in this program. No independent code replication.

## E16-V: Codex's independent verification of E16, E16-D and E16-D2, and the corrections it requires (8 November 2026)

`E16V_REPORT.md` (commit f7e92b4), audit only, no reruns. Every numeric table component (3,124) matches the receipts within rounding, the shortcut ceilings are reproduced exactly by independent code, and the oracle/twin and M/learner stream constructions are confirmed identical at source level. The discrepancies are in my prose and in literal gate reporting, and they are corrected here rather than by editing the sections above.

- **"Within 1–9 points of the twin while current" (E16) → 1–11 points.** The F3 gap in seed 116089 is 11.1 points.
- **Exposure shares (E16, E16-D2).** The learner's F1 share during the composition blocks is 1/2 then 1/3 (not "one third"); its F2 share during the mod-73 blocks is 1/4 then 1/5 (not "one fifth"). The share arm moved F1 during F2 from 1/2, 1/3 to 1/2, 1/2 and F2 during F3 from 1/4, 1/5 to 1/4, 1/4: it changed the second block of each pair only and did not double exposure. The first loss-bearing block of each pair was unchanged in every arm but the oracle, one more reason share could not have been the lever.
- **Literal gates.** P16D.4 ("within 3 points") was reported as holding at 3.1–4.3 points; it misses literally. P16D2.3 ("within 3 points in both arms") misses on arm M, whose composition acquisition is more than 3 points higher in every seed (it is backprop), and on one seed of the share arm's F1 (+4.9); the share arm's compositions are within 1.5. Both are recorded as literal misses, with the direction stated: acquisition was better than predicted, not worse.
- **"Balanced indistinguishable from uniform"** → within 2.4–7.8 points at the end; expected shares equal in blocks 3 and 5, different in 4 and 6; no equivalence test was run.
- **Cell audit.** Family means with the cell silenced are 3–7% on F1, but 18 of 168 individual operation scores exceed 10%, up to 32% on max73/min73 in the every-entry arm (comparison operations have a visible-only ceiling near 50% when one operand is hidden). The claim holds for F1 and the carry compositions, not "everywhere". The store was not ablated.
- **Receipt label bug.** Every receipt recorded the shared local decay (0.25) in `settings.wd`, including arm D, whose optimizer runs at 0.3. The harness now records the optimizer's actual decay, the arm, the rule and whether the store is present, and writes the last block's buffer count (verified on a dry run of arm D: wd 0.3). Earlier receipts are left as they are; this note is the correction.
- **Provenance.** The standing births' weights are not in the repository (gitignored); no per-batch digests or realized draw counters exist for E16-D/D2; receipts are reproducible from source and seeds but not attested by run-time hashes. These limit independent replication, not the recorded outcomes.

**The new finding is on the window net.** From the E16-W batch archives: E8-style replay drew the same 1,024 distinct composition prompts throughout the mod-73 stages (4% of the taught triples), and 14–21% of the replayed six-byte targets were mathematically wrong, because E8's replay targets are the learner's own answers. The cell harness replays stored observed answers, so this error source exists only on the window-net side. The window net's forgetting therefore has two candidate causes, coverage and wrong targets, and E16-WR now carries an arm to separate them (observed answers as targets at the largest capacity, with realized draw and target-correctness counters). Every E8-family result since E8 (E9, E12, E13, E14, E14-S, E16-W) ran with self-generated replay targets that are partly wrong late in life; that was the design (pseudo-rehearsal) and should be stated wherever those results are read.

**What stands after the audit.** The acquisition and retention results are as recorded. Richer replay restores retention; coverage is the most effective intervention found; the rule costs 12–31 points by seed (about 20 on average) on impoverished replay and nothing on the twin's data; share is not the lever. Codex's caution is adopted: coverage is not established as the unique cause, and the store, the decay difference between D and the local arms, and the rule are not fully separated.

## E16-R: bounded reservoir replay, the second attempt at the Phase A gate (8 November 2026)

Predictions: `E45_PREDICTIONS.md`, E16-R. Per-family reservoir of fixed capacity filled by Algorithm R over the family's presentations (bounded memory, constant work per event, duplicates allowed), draws balanced per family, fresh-render; capacities 2,048 / 4,096 / 8,192; three standing seeds; nine lives, 17 minutes per block with nine processes, 1 h 40 min. Tables `exp45/SUMMARY_E16.md`; receipts `exp45/e16_life_T_res{2048,4096,8192}_*.json` (each records the reservoirs' distinct counts per block).

| Replay | distinct F2 entries held (of 24,000) | Held compositions, block 4 → 6 | F1 operations k = 12, block 2 → 6 | F3 at 6 |
|---|---|---|---|---|
| balanced, 1,024/block (E16-D) | 2,048 | 89.0→64.8 · 85.8→55.6 · 88.6→49.1 | 92.6→82.1 · 91.5→76.9 · 87.6→69.0 | 91.1 · 89.8 · 85.8 |
| reservoir 2,048 | ≈1,960 | 88.1→63.5 · 87.8→50.6 · 89.4→44.2 | 93.8→83.0 · 92.4→77.9 · 88.4→62.1 | 90.4 · 88.5 · 80.8 |
| reservoir 4,096 | ≈3,790 | 88.2→73.8 · 86.8→72.4 · 88.4→68.2 | 92.9→86.5 · 91.6→85.8 · 88.2→75.2 | 91.2 · 90.5 · 88.4 |
| reservoir 8,192 | ≈7,020 | 87.3→76.3 · 86.9→72.7 · 90.8→76.3 | 93.3→88.3 · 92.4→88.1 · 89.9→79.8 | 90.8 · 90.2 · 90.1 |
| every entry (E16-D) | 24,000 | 90.2→84.5 · 87.4→79.8 · 91.4→83.3 | 93.7→89.8 · 92.5→89.2 · 90.2→77.7 | 92.5 · 89.6 · 88.3 |
| twin (backprop) | | 93.8→93.3 · 91.2→92.0 · 93.7→93.8 | 93.9→95.4 · 93.8→94.1 · 94.1→94.4 | 93.8 · 92.6 · 93.4 |

**Retention rises monotonically with capacity in every seed and has not saturated at 8,192.** Composition retention tracks the number of distinct composition entries the reservoir holds: about 2,000 → 44–64%, 3,800 → 68–74%, 7,000 → 73–76%, 24,000 → 80–85%; roughly a logarithmic return. A reservoir of 2,048 reproduces the fixed 1,024-per-block buffer (within 5 points on compositions in every seed), which says capacity, not the sampling scheme, is the variable. The operations' loss during the composition blocks also shrinks with capacity (seed 116089: 40 → 62 → 67% at block 3 for 2,048 / 4,096 / 8,192, every-entry 61%).

**Ratios to the twin at 8,192:** F1 0.93 / 0.94 / 0.85, F3 0.97 / 0.97 / 0.96, compositions 0.82 / 0.79 / 0.81. The gate (0.80 on every family) is met on the operations in every seed and missed by one point on compositions in one seed.

**Verdicts.** P16R.1: monotone, holds; 2,048 within 5 points of the balanced arm on compositions in every seed (1.3 / 5.0 / 4.9) and on F1 in two (0.9 / 1.0 / 6.9); 8,192 within 5 of every-entry on F1 (1.5 / 1.1 / 2.1) but 7–8 points short on compositions in every seed: partial. P16R.2 (gate at 8,192): misses by 0.01 in one seed on compositions, holds on F1 and F3 everywhere. P16R.3 (4,096 recovers two thirds of the balanced-to-every-entry gap on compositions): 46 / 69 / 56%, misses in two seeds. P16R.4 (acquisition within 3 points of E16): F1 after block 2 +0.6 to +3.8, compositions after block 4 +0.3 to +3.8; misses the literal 3 in three of eighteen comparisons, all in the direction of better acquisition.

**What E16-R establishes.** A bounded replay with constant per-event work recovers most of the retention that the unbounded buffer gave, and the dependence on capacity is smooth and monotone, which is the shape a scalable mechanism should have. What it does not yet do is reach the twin's composition retention at a capacity that is small relative to the family (7,000 of 24,000 triples), and it still uses the family label. The next two steps are therefore fixed: a single reservoir with one total capacity and no labels (Codex's point in 9.33), at a total matched to this run (24,576) and at one larger value, to show the label is unnecessary and to find where the curve bends; and verbatim raw-byte replay at the same budget, which removes the parser. The per-family reservoir at 8,192 goes to the cloud session for fresh-seed confirmation.

**Evidence level.** Development result, three seeds, one harness; comparators from E16-D (same births and stream); shortcut ceilings as in E16; receipts carry distinct counts but, as E16-V noted for the earlier arms, no realized draw counters. Family labels supplied by the harness. No independent confirmation yet (E16-F2 assigned).

## E16-R2: one reservoir, one total budget, no family labels; the Phase A gate is met at 65,536 (8 November 2026)

Predictions: `E45_PREDICTIONS.md`, E16-R2. A single reservoir filled by Algorithm R over every presentation of the life (one counter, no family key anywhere in the replay path), replay half drawn uniformly from it; total capacities 24,576 (equal to E16-R's three per-family reservoirs of 8,192) and 65,536; three standing seeds; six lives, 15 minutes per block, 95 minutes. Receipts `exp45/e16_life_T_tot{24576,65536}_*.json` (with the reservoir's distinct count and, for reading only, its composition by family).

| Replay | distinct entries at block 5 (F1 / F2 / F3) | Held compositions, block 4 → 6 | F1 operations k = 12, block 2 → 6 | F3 at 6 | ratio to twin at 6: F1 · comp · F3 |
|---|---|---|---|---|---|
| per-family 8,192 (E16-R) | 5,500 / 7,000 / 4,200 | 87.3→76.3 · 86.9→72.7 · 90.8→76.3 | 93.3→88.3 · 92.4→88.1 · 89.9→79.8 | 90.8 · 90.2 · 90.1 | 0.93·0.82·0.97, 0.94·0.79·0.97, 0.85·0.81·0.96 |
| one reservoir 24,576, no labels | 6,900 / 7,000 / 2,900 | 88.5→78.0 · 85.1→71.9 · 89.7→70.3 | 93.9→92.8 · 93.2→90.1 · 88.8→80.8 | 91.9 · 90.2 · 88.4 | 0.97·0.84·0.98, 0.96·0.78·0.97, 0.86·0.75·0.95 |
| one reservoir 65,536, no labels | 9,100 / 14,500 / 4,700 | 87.9→82.5 · 86.7→74.8 · 89.7→77.9 | 94.1→92.8 · 93.4→90.8 · 89.5→83.1 | 92.8 · 90.4 · 88.6 | 0.97·0.88·0.99, 0.96·0.81·0.98, 0.88·0.83·0.95 |
| every entry, per family (E16-D) | 9,408 / 24,000 / 5,328 | 90.2→84.5 · 87.4→79.8 · 91.4→83.3 | 93.7→89.8 · 92.5→89.2 · 90.2→77.7 | 92.5 · 89.6 · 88.3 | 0.94·0.91·0.99, 0.95·0.87·0.97, 0.82·0.89·0.95 |
| twin (backprop) | | 93.8→93.3 · 91.2→92.0 · 93.7→93.8 | 93.9→95.4 · 93.8→94.1 · 94.1→94.4 | 93.8 · 92.6 · 93.4 | |

**The family label is unnecessary.** At the matched budget the label-free reservoir is within 5 points of the per-family one on eight of nine family-by-seed comparisons (the ninth, compositions in seed 116089, is 6 points lower) and is better on the operations in every seed (+4.5 / +2.0 / +1.0 at block 6). It allocates slots by presentation share, which in this stream gives each family about a third by the end.

**More total budget keeps buying retention, and at 65,536 the gate is met in every seed.** Ratios to the twin at 65,536: operations 0.97 / 0.96 / 0.88, compositions 0.88 / 0.81 / 0.83, later operations 0.99 / 0.98 / 0.95, all at or above 0.80. The curve is still logarithmic in coverage: 14,500 distinct compositions give 75–83% where all 24,000 give 80–85%.

**Verdicts.** P16R2.1 holds on 8 of 9 comparisons (compositions in seed 116089 −6.0). P16R2.2 partial: compositions within 3 points of every-entry in one seed (−2.0 / −5.0 / −5.4), ratio ≥ 0.85 in one seed (0.88 / 0.81 / 0.83); F1 within 3 points or better in every seed. P16R2.3 misses in direction: the label-free arm's F1 at block 4 is higher than the per-family arm's (90.4 / 88.4 / 74.6 against 86.7 / 83.7 / 64.6), not lower. The mechanism audit was wrong about why: the reservoir is updated at block ends, so during block 3 it holds no compositions at all and during block 4 about one third, which gives F1 a larger replay share during the composition blocks than the per-family arm's fixed thirds, not a smaller one. P16R2.4 holds (acquisition +0.7 to +3.4 on operations, +0.8 to +2.7 on compositions).

**What E16-R2 establishes.** With a bounded, label-free, constant-work replay memory of 65,536 entries (a few bytes each), one cell learner under the local rule, at fixed capacity, acquires three families of different form from one stream and keeps each at 0.81–0.99 of a backprop twin after the stream has moved on. That is the Phase A gate as stated in E16, met on the three standing seeds. Two things it is not yet: confirmed on fresh seeds (the cloud session's E16-F2 is redirected to this arm), and free of the parser (the reservoir stores parsed entries and re-renders them; E16-R3, verbatim raw bytes at the same budget, is prepared). The scaling story also needs a stream with unequal family durations before the label-free allocation is called general.

**Evidence level.** Development result, three seeds, one harness; the budget is in entries, not bytes (an entry re-renders to 7–19 bytes of prompt plus 3–6 of answer); receipts carry distinct counts and composition by family but, as before, no realized draw counters; shortcut ceilings as in E16. Fresh-seed confirmation assigned (E16-F2).

## E16-WR, assessed (Codex, 8 November 2026): self-generated replay targets run away with coverage

`E16WR_REPORT.md`, review 9.35 and 9.39. On the window net with E8-style replay (targets are the learner's own answers), held composition after four later stages is 32–36% at 1,024 reservoir slots per stage, 42–48% at 4,096 and 25–32% at 16,384; the fraction of wrong replayed composition targets is 17–21%, 23–38% and, rising through the later stages, 81–91% respectively. With observed targets at 16,384 slots, held composition is 57–65% (loss 14–20 points) and the mod-97 operations stay at 95–96% (self targets 70–77%). Pseudo-rehearsal is a positive feedback loop whose gain grows with coverage; observed targets break it, and the remaining loss is the coverage limit seen in E16-R2. The cell harness has always replayed observed answers, which is why its coverage curve is monotone. The design is retired for this program; E16-WR2 reruns E8 with observed targets.

## E16-R3: the parser removed; verbatim raw-byte replay meets the gate (8 November 2026)

Predictions: `E45_PREDICTIONS.md`, E16-R3. The same single label-free Algorithm-R reservoir as E16-R2 at 65,536 slots, but each slot holds the presented bytes, prompt and answer exactly as they appeared, and replay returns them verbatim: no task recognition, no operand extraction, no re-rendering, no recomputed answer anywhere in the replay path. Three standing seeds, three lives, 6 minutes per block with three processes on a free machine (36 minutes; verbatim replay also skips the rendering cost, 2,150 s of CPU per life against 3,300 s for the parsed arm). Receipts `exp45/e16_life_T_raw65536_*.json` record the distinct count, the composition by first byte (for reading only) and the reservoir's size in bytes.

| Replay at 65,536 slots | F1 operations k = 12, block 2 → 6 | Held compositions, block 4 → 6 | F3 at 6 | ratio to twin at 6: F1 · comp · F3 | reservoir at block 5 |
|---|---|---|---|---|---|
| parsed entries, re-rendered (E16-R2) | 94.1→92.8 · 93.4→90.8 · 89.5→83.1 | 87.9→82.5 · 86.7→74.8 · 89.7→77.9 | 92.8 · 90.4 · 88.6 | 0.97·0.88·0.99, 0.96·0.81·0.98, 0.88·0.83·0.95 | 28,400 distinct entries |
| raw bytes, verbatim (E16-R3) | 93.6→93.2 · 92.7→89.9 · 90.4→82.2 | 87.6→80.2 · 86.6→74.8 · 89.2→80.1 | 92.2 · 90.7 · 85.6 | 0.98·0.86·0.98, 0.96·0.81·0.98, 0.87·0.85·0.92 | 57,800 distinct byte strings, 1.05 MB |
| twin (backprop) | 93.9→95.4 · 93.8→94.1 · 94.1→94.4 | 93.8→93.3 · 91.2→92.0 · 93.7→93.8 | 93.8 · 92.6 · 93.4 | | |

**The parser was doing nothing that the bytes do not do.** Verbatim replay is within 3 points of the parsed arm on every family in every seed (F1 +0.4 / −0.9 / −0.9; compositions −2.3 / 0.0 / +2.2; F3 −0.6 / +0.3 / −3.0). The gate holds without it: ratios to the twin 0.98 / 0.96 / 0.87 on the first operations, 0.86 / 0.81 / 0.85 on compositions, 0.98 / 0.98 / 0.92 on the later operations, all at or above 0.80. Extrapolation survives (k = 20 within 1.2–4.9 points of k = 12) and the cell is still what holds (silenced 2.6 / 5.5 / 6.8%). The reservoir is 1.05 MB. E11-D's finding that 1,024 fixed episodes per stage are memorized does not carry to 65,536 verbatim slots over the life: coverage, not the form of storage, was the variable.

**Verdicts.** P16R3.1 holds (every comparison within 3 points, tolerance 5). P16R3.2 holds (gate met in every seed). P16R3.3 holds. P16R3.4 holds (1.05 MB against a 1.6 MB bound).

**What E16-R3 establishes, with E16-R2 and E16-WR.** One cell learner under the sign rule with the exact trace, at fixed capacity, with one replay memory of about a megabyte that stores observed experience verbatim and samples it with constant work per event, acquires three families of different form from one stream and keeps each at 0.81–0.98 of a backprop twin after the stream has moved on. The replay path contains no task label, no parser, no answer generator and no family label. Of the scaffold named on 8 November (format, tasks, stages and answers supplied by the program; replay that parses prompts into tasks and operands and regenerates bytes), the replay half is gone. What remains is the environment's side: the stage structure of the stream and the loss masked to the answer bytes, both of which are the next two removals (a continuously drifting stream; next-byte loss on every byte), each one change on this harness.

**Evidence level.** Development result, three seeds, one harness; comparators from E16-R2 and E16 (same births, same stream); shortcut ceilings as in E16; replay targets are observed answers, so E16-V's wrong-target problem does not enter. Fresh-seed confirmation assigned to the cloud session (E16-F3).

## E16-F, assessed (cloud session, 8–9 November 2026): E16-D on fresh seeds

`E16F_REPORT.md`, review 9.32. Seeds 200003, 300007, 400009, new E11 births (90–99% at k = 12), every-entry balanced replay against the twin. After block 6: compositions 83.3 / 77.0 / 81.5 (standing 84.5 / 79.8 / 83.3); ratios to the twin on mod-97 operations 0.95 / 0.92 / 0.89, compositions 0.89 / 0.84 / 0.87, mod-73 operations 0.99 / 0.98 / 0.98. P16F.1, .2 and .4 hold in every seed; P16F.3 holds in two and misses in 300007 by one clause (compositions −12.0 from block 4, limit 10; carry pairs and subtraction). Reruns after a tool time limit reproduced the killed lives' first five blocks exactly. The E16-D finding transfers in kind and range; composition retention remains its weakest, most seed-dependent part.

## E16-F2, assessed (cloud session, 9 November 2026): the label-free reservoir on fresh seeds

`E16F2_REPORT.md`, review 9.36. One reservoir of 65,536 entries, no family label, on seeds 200003 / 300007 / 400009 against E16-F's twin. Ratios to the twin after block 6: mod-97 operations 0.97 / 0.95 / 0.94, held compositions 0.86 / 0.87 / 0.82, mod-73 operations 0.99 / 0.99 / 0.96; composition loss from block 4 to 6: 8.8 / 8.0 / 11.8 points; distinct entries at block 5: 28,300–28,400. P16F2.1–.3 hold in every seed. The Phase A gate is met on six seeds.

## E16-S and E16-L: the stream without stages holds; the loss on every byte does not (8 November 2026)

Predictions: `E45_PREDICTIONS.md`, E16-S and E16-L. Twelve lives on the raw-byte reservoir of 65,536 (E16-R3's arm), three standing seeds: S0 online admission (control), S1 the drifting stream with online admission, S-D the twin on the drifting stream, L the every-byte loss. 25 minutes per block with twelve processes, 2 h 15 min. Receipts `exp45/e16_life_T_raw65536_{online,drift_online,lossall}_*.json`, `exp45/e16_life_D_drift_*.json`.

| Arm | F1 operations k = 12, block 2 → 6 | Held compositions, block 4 → 6 | F3 at 6 | ratio to its twin at 6: F1 · comp · F3 |
|---|---|---|---|---|
| E16-R3 (block admission, blocks) | 93.6→93.2 · 92.7→89.9 · 90.4→82.2 | 87.6→80.2 · 86.6→74.8 · 89.2→80.1 | 92.2 · 90.7 · 85.6 | 0.98·0.86·0.98, 0.96·0.81·0.98, 0.87·0.85·0.92 |
| S0 online admission | 92.9→89.8 · 91.6→89.7 · 89.6→80.5 | 89.6→81.1 · 86.5→75.7 · 90.0→82.2 | 91.6 · 91.4 · 89.4 | 0.94·0.87·0.98, 0.95·0.82·0.99, 0.85·0.88·0.96 |
| S1 drifting stream, no stages | 92.1→92.1 · 88.4→88.9 · 71.5→79.6 | 88.3→80.6 · 85.0→81.4 · 86.6→80.3 | 92.6 · 89.5 · 90.1 | 0.97·0.86·0.99, 0.93·0.87·0.96, 0.84·0.86·0.97 |
| S-D twin on the drifting stream | 94→94 · 94→95 · 94→95 | 93→94 · 91→94 · 94→93 | 93.9 · 93.1 · 93.3 | |
| L loss on every byte | 32.2→33.1 · 41.4→30.9 · 33.0→29.8 | 85.0→55.5 · 80.2→48.2 · 84.5→50.4 | 34.2 · 34.9 · 32.4 | 0.35·0.59·0.36, 0.33·0.52·0.38, 0.32·0.54·0.35 |

**The stages were a convenience, not a crutch.** On a stream whose family mixture drifts smoothly and never announces a boundary, with the reservoir admitting items as they arrive, the learner keeps every family at 0.84–0.99 of a twin trained on the same stream: the gate holds in every seed. Against the online-admission control it is within 5 points on every family (compositions +0.5 / +5.7 / −1.9; F1 +2.3 / −0.8 / −0.9), and online admission is itself within 3.4 points of block-end admission everywhere. Under drift F1's exposure falls to nothing by mid-life, and what the learner keeps of it is what the reservoir and the rule keep. In seed 116089 the first operations dip at the second checkpoint (72%) because their share is already falling as the compositions arrive, and recover to 80 by the end. P16S.0 holds within 3.4 (one comparison 0.4 over the tolerance); P16S.1 holds (one comparison better than the tolerance by 0.7); P16S.2 holds.

**The answer mask was doing real work.** With next-byte loss on every byte, the sign rule does not acquire addition or subtraction at all: at k = 12 after their blocks they stand at 3–5% (max and min at about 60%), at k = 0 the first family ends at 47–50%, and the later operations at 32–35%; the compositions are acquired to 80–85% at block 4 and lose 30–35 points afterwards. P16L.1 and P16L.2 are refuted in every seed. The prompt bytes (operands and random letters, seven to nineteen of them) are near-uniform and unpredictable, so the loss is dominated by terms that carry no rule, and the three answer bytes' signal is swamped under the sign rule. Whether backprop fails the same way under this loss is not yet measured (the L arm's comparator was the answer-loss twin, by design a conservative choice): a twin under the every-byte loss is queued (three lives) so that the objective and the rule are separated. The consequence for Phase B stands either way: a language stream will not point at the answer, so loss weighting, or a learner that finds what is predictable, is a design question, not a detail.

**Two harness notes.** (i) The twin's receipt name omitted the stream tag, so the three drift-twin receipts were written over the standing E16 twin receipts in the working folder; the standing ones were restored from the repository (byte-identical to the E16 commit) and the drift twins renamed `e16_life_D_drift_*`; the harness now carries the stream and loss tags on arm D's receipt. (ii) The drift stream's weights are raised cosines of half-width half the life, centred where the blocks were.

**Evidence level.** Development result, three seeds, one harness; the drift twin shares the stream's family generators and seeds; shortcut ceilings as in E16; L's comparator mismatched in loss (noted). Fresh-seed confirmation of S1 is the natural next cloud job once E16-F3/F4 are in.

## E16-F3 and E16-WR2, assessed (9 November 2026)

*E16-F3 (cloud session):* raw-byte reservoir of 65,536 on seeds 200003 / 300007 / 400009; ratios to the twin 0.95 / 0.98 / 0.91 on mod-97 operations, 0.87 / 0.87 / 0.88 on held compositions, 0.96 / 0.98 / 0.97 on mod-73 operations; within 3 points of the parsed arm on eight of nine comparisons, +5.5 on the ninth; 1.048–1.050 MB, 57,700–58,000 distinct byte strings. P16F3.1 and .3 hold everywhere; P16F3.2 misses once in the better direction. *E16-WR2 (Codex):* window net, observed targets, one total reservoir: at 65,536 slots held composition loses 15.7 / 19.0 / 18.5 points over four later stages (per-stage 16,384 observed: 14–20; self targets: 49–56), operations 95–97%; at 16,384 total, losses 29–33. E8's demonstration with observed targets reproduces E13-A1 (composition +0.5 / 0.0 / +3.5; every operation within 2 points). The window net under this replay retains compositions worse than the cell learner (5–12 points lost); architecture and schedule are the candidates, to be separated by running one learner on the other's schedule.

## E16-A: the architecture audit; the exact-key store does nothing (8 November 2026)

Predictions: `E45_PREDICTIONS.md`, E16-A. E16-R3's arm (raw bytes, 65,536 slots) with the E8 exact-key store absent from the local learner (`E16_STORE=off`: no writes, no logit bias at scoring), three standing seeds, three lives; receipts `exp45/e16_life_T_raw65536_nostore_*.json`. Run alongside Codex's E16-WR2 workers by a scheduling slip (both started within a minute on an idle machine), so 25 minutes per block instead of 6.

**Result.** At every checkpoint, on every family, the no-store arm equals E16-R3: the operations at k = 0 and k = 12 differ by exactly nothing in every seed and every block, the later operations by nothing, and the held compositions by at most one item in 600 (0.04 points, two seeds), which is the store's logit bias changing one answer on one prompt. Training is unaffected because the store never entered the loss; scoring is unaffected because every scored prompt is unseen and the store's keys with distractor letters overflowed within the first block.

**Verdicts.** P16A.1 holds (within 2 points everywhere; the actual differences are 0.00–0.04). P16A.2 holds (k = 0 acquisition identical).

**Consequence for the record.** The exact-key store, the functional descendant of the mushroom-body associative memory that left the computational path on 6 November when its residual measured about 1%, has contributed nothing to any cell-harness result from E11 to E16-S. The original PN–KC–MBON architecture has therefore left the main model entirely; it has not survived as a module. The store is retired for new designs: E17 and the Phase B harness will not carry it. The E16 harness keeps the knob with its default unchanged so that every receipt to date remains reproducible by name; the record, not the default, says it was inert. The one place a fast associative memory of that kind may return is as the sparse-key retriever of the Phase B episodic battery (9.47's conditions), tested against a matched generic index, which is the test the 6 November measurement never had the task to run.

**Evidence level.** Development result, three seeds, one harness; an ablation with everything else identical (same births, stream, reservoir, draws, panels); the equality is exact at the item level, so no further seeds are needed.

## Errata and harness changes after the 8 November static audit

- **Held-pair panels (E16 family, all runs before E17).** The +- held panel's 600 operand triples were identical to the first 600 of the taught -> pair's triples (seed collision `31·seed + 377`); the pairing and prompts were held out, and the shared triples carry different answers under the taught pair. Effect on the reported numbers: bounded by the +- against -+ comparison (no systematic +- advantage in any arm); not re-scorable, since final weights were not saved. From E17 on, held and within panels are keyed by split and pair and operand-disjoint from every taught triple; lives save final weights.
- **`QUALIFY_E16.json`:** `birth_compose_final_before_life` was scored after 20 smoke updates; the stage-zero scores in every receipt are the true birth scores. Qualification now scores before the smoke updates.
- **E16-WL, assessed (Codex).** On the window net the sign rule under the every-byte loss still learns the operations (stage-eight mean 92.5–93.0% against 94.4–95.5% with the answer mask; add and sub 88.5–90.8%) and loses 10–24 points of held composition against its answer-loss reference; backprop is unaffected (96–97%, composition within 2 points). So the every-byte loss costs the window net little and the cell learner under the sign rule everything (E16-L: add and sub 3–5%); the cell learner's backprop twin under the same loss is finishing (at block 4: add and sub 80–93% at k = 12). The failure is specific to the local rule on the recurrent learner under this objective, which points at the trace: under the answer mask the state's synapses receive learning signals only at answer positions; under the every-byte loss they receive them at every position, most of them from unpredictable bytes.

**E16-L, the twin under the every-byte loss (8 November 2026, 23:15 EST).** Backprop through time on the same stream with next-byte loss on every byte is unaffected by the objective: mod-97 operations 92 / 91 / 94% at k = 12 after block 2 (answer-loss twin 94 / 94 / 94) and 95 / 94 / 95 after block 6; held compositions 91.7 / 88.1 / 90.8% after block 4 (answer-loss twin 93.8 / 91.2 / 93.7) and 90.6 / 90.5 / 92.1 after block 6; mod-73 operations 92.4 / 89.9 / 92.9. The prediction written at 21:20 (backprop acquires the first operations to at least 80% at k = 12 after block 2 in every seed) holds. The local learner under the same loss stands at 0.31–0.35 of this twin on the operations and 0.53–0.61 on compositions. Together with E16-WL (the window net under the sign rule loses little under the every-byte loss), the failure is confined to one cell of the design: the local rule, on the recurrent learner, under the every-byte objective. The mechanism to test first is the trace: with the answer mask the state's synapses receive learning signals only at the three answer positions; under the every-byte loss they receive them at every position, most from unpredictable prompt bytes, and the eligibility carries that noise into the holding synapses. Backprop through time computes the same positions' gradients but with exact credit, and the window net has no holding synapses to corrupt. The one-change arm: the every-byte loss for the readout and the shared path, with the learning signal reaching the state's synapses weighted by position (answer bytes 1, prompt bytes λ ∈ {0, 0.1}); if λ = 0 recovers acquisition, the trace's exposure to unpredictable positions is the cause and a learner-side rule for "what is predictable" is the Phase B design; if it does not, the shared path's sign-rule updates are the problem.

## E17, Phase 0: one diagonal layer without a window does not learn the operations under backprop (8 November 2026, 23:35 EST)

Predictions: `E45_PREDICTIONS.md`, E17 (P17.0). Six births by backprop through time, 24,576 updates each, on the two windowless architectures (`exp45/e17_dev_{cell1,bank}_*.json`, hashes in `exp45/E17_BIRTHS.json`; every worker exited 0; 35 minutes for the six).

| Birth, after 24,576 updates, k = 12 | add89 | sub89 | max89 | min89 |
|---|---|---|---|---|
| cell1 (E11 cell, no window), three seeds | 27 / 31 / 23 | 13 / 14 / 17 | 66 / 70 / 62 | 66 / 73 / 64 |
| bank (fixed rates, no window), three seeds | 30 / 29 / 22 | 13 / 10 / 9 | 53 / 61 / 58 | 61 / 61 / 61 |
| E11 windowed births, for reference | 91–92 | 91–92 | 98–99 | 98–99 |

The scores at k = 0 are the same as at k = 12 (cell1 73129: 27% at both), so the failure is not holding across distractors; it is learning the operation at all. The curves are still rising at the end (cell1 73129: add89 18 → 21 → 27% over the last three checkpoints), so a larger dose would help, but at that slope a windowed birth's level is many doses away. **P17.0 is refuted for both architectures, under exact credit.** The local-rule lives were not launched, as the staging required.

**Reading.** In both architectures the write into the state depends on the current byte only (`c_t = tanh(W_c p_t)`, `p_t` a function of the byte), so the state is a gated, decayed superposition of per-byte features, and the only place two digits can interact is the readout's single hidden layer, reading that superposition. The window gave the readout the digits in explicit positions; without it, position is encoded only by each unit's decay pattern, a weak code, and addition mod 89 must be computed from it. Comparison operations, which need less positional structure, reach 53–73%. This is a representation problem, not a rule problem, and it was pre-registered: the clause of P17.0 for a double failure says one diagonal layer is insufficient without a window and a second layer is the next step.

**Next (Phase 0b, births only, under backprop, overnight):** three one-change variants against the same gate, each keeping the trace exact within a layer: a deeper readout (two hidden layers between the state and the logits) on cell1; two stacked fixed-rate banks, the second written from the first's state, so writes can depend on accumulated context; and the single bank at double dose (49,152 updates), to measure the slope. If none passes, the honest conclusion is that a byte-at-a-time learner needs state-dependent writes (a non-diagonal recurrence), for which the exact trace does not exist and the e-prop approximation is the tool, and that is a design decision for Mr. Zhou rather than a run.

**Evidence level.** Development result, three seeds, backprop births; no local-rule result exists for these architectures because the gate stopped the lives.

## E17, Phase 0b: context-dependent writes are the missing operation (9 November 2026, 00:57 EST)

Predictions: `E45_PREDICTIONS.md`, E17 Phase 0b. Nine births by backprop through time (`exp45/E17B_BIRTHS.json`; every worker exited 0; 68 minutes for the nine).

| Birth, k = 12 | add89 | sub89 | max89 | min89 | gate |
|---|---|---|---|---|---|
| cell1 (Phase 0) | 27 / 31 / 23 | 13 / 14 / 17 | 66 / 70 / 62 | 66 / 73 / 64 | no |
| cell1d2, deeper readout | 58 / 67 / 54 | 45 / 59 / 47 | 86 / 86 / 79 | 83 / 90 / 82 | no |
| bank (Phase 0) | 30 / 29 / 22 | 13 / 10 / 9 | 53 / 61 / 58 | 61 / 61 / 61 | no |
| bank, double dose (49,152) | 28 / 29 / — | 14 / 12 / — | 61 / 62 / — | 56 / — / — | no |
| **bank2, two stacked banks** | **93 / 95 / 95** | **92 / 92 / 92** | **99 / 97 / 96** | **98 / 97 / 95** | **yes, every seed** |
| E11 windowed births | 91–92 | 91–92 | 98–99 | 98–99 | |

A deeper readout recovers 30–45 points and stays short; double dose recovers nothing; a second bank whose write is computed from the first bank's state reaches the windowed births' level in every seed. P17.0b.1, .2 and .3 hold. The byte-at-a-time learner needs writes that depend on accumulated context, and one extra diagonal layer supplies them while keeping each bank's own trace exact given its input. The cost is known from qualification: credit from the readout to the first bank through the second bank's memory is not carried by the first bank's trace (64–88% from backprop through time), the e-prop truncation.

**Phase 1 launched** (09:00 EST): bank2 under the local rule on the E16 stream with the raw-byte reservoir, and its twin, three seeds; P17.1–P17.5 as written, plus the clause on the truncated first-bank trace added before launch.

**E16-F4, assessed (cloud session, 9 November 2026).** Raw-byte reservoir on seeds 500011 / 600013 / 700027: ratios to the twin 0.94 / 0.81 / 0.94 on mod-97 operations, 0.87 / 0.69 / 0.83 on compositions, 0.99 / 0.90 / 0.95 on mod-73 operations; the gate holds in two seeds and fails in 600013 on compositions; the retention clause holds in one; acquisition holds in all. Across nine seeds the gate holds on eight. Headline corrected to "eight seeds of nine, one at 0.69 on compositions".

**E16-RC, assessed (cloud session, 9 November 2026; merged from `claude/vibrant-gauss-tops5f`).** Raw-byte reservoir at 16,384 / 32,768 / 131,072 on the standing seeds, with E16-R3's 65,536 and the every-entry arm as comparators. Held compositions after block 6 (16k → 32k → 64k → 128k): 76.7 → 79.2 → 80.2 → 81.5; 69.8 → 78.5 → 74.8 → 78.2; 76.1 → 73.1 → 80.1 → 79.8; mod-97 operations 89.5 → 91.3 → 93.2 → 91.8; 89.7 → 90.4 → 89.9 → 90.6; 78.0 → 78.8 → 82.2 → 80.5. The curve is flat within seed spread; only 93683 at 16,384 falls below the gate (0.76 on compositions); 65,536 is the smallest budget meeting the gate in every seed; 16 bytes per slot throughout; about 20,000 distinct compositions held at 131,072, still 3 points under the every-entry arm in two seeds. P16RC.4 and the composition-count clause hold; P16RC.1 and .2 miss; P16RC.3 misses in one seed.

## E16-F5 and E16-RP, assessed (cloud sessions, 9 November 2026)

**E16-F5** (drifting stream, fresh seeds; `E16F5_REPORT.md`): the drift twin holds 90 / 85 / 90 in every seed; the learner holds 0.93 / 0.86 / 0.96 and 0.96 / 0.87 / 0.99 of it in 200003 and 300007, within 0.3 points of the same births under block stages, and misses in 400009 at 0.69 on compositions (64.9; +- 48.0, -+ 44.7, max-min pairs 82–85; the mod-97 operations dipped to 62.0 as the compositions entered). The drift gate holds on five of six seeds; the two misses of the program (600013 under stages, 400009 under drift) share one signature. **E16-RP** (`E16RP_REPORT.md`): on the same final weights, the old colliding +- panel scores −1.0 / −6.8 / +0.8 relative to the clean one (mean absolute difference over the four held pairs 1.08 / 2.37 / 1.38): the collision did not inflate the E16 composition figures at block 6; blocks 4–5 are unmeasured. Sign-rule receipts are bit-identical on one machine and diverge across machines from the first update, so parity clauses are machine-bound and every cross-machine confirmation is a comparison between runs; the +- pair moved by up to 16 points between two runs of one seed on two machines. Follow-ups briefed to the cloud workers: E16-M2 (arm M and oracle replay on the two failing seeds: rule or data) and E16-F6 (six more drift seeds, the failure rate and the early signature as a predictor). Harness: arm M under online admission filled the reservoir at block ends only; fixed at 1470401b and dry-run; no finished result used that path.

## E17, Phase 1 and 1b: the local rule silences the stacked banks, and the stacked banks do not compose under backprop (9 November 2026; both phases complete)

Predictions: `E45_PREDICTIONS.md`, E17 Phase 1 (P17.1–P17.5 as applied to bank2) and Phase 1b. Twelve lives on the three standing seeds, births `e17_dev_bank2_<seed>.pt` (Phase 0b). Tables: `exp45/SUMMARY_E17_PHASE1.md`. Probe: `e17_probe_state.py`, `exp45/PROBE_E17_bank2_<seed>.json`.

**Phase 1, six lives, complete.** The local learner (`T_bank2_raw65536`) falls to 0–2% on every panel in block 1, including the birth's own mod-89 operations (95 at birth, 1 after block 1), and stays there for six blocks. The twin (`D_bank2`) learns the three operation families and not the compositions:

| Seed | Arm | F1 mean after block 1 / 2 / 6 | Held comp after 4 / 6 | +- / -+ / >< / <> after 6 | within after 6 | F3 after 6 | dev89 after 6 |
|---|---|---|---|---|---|---|---|
| 73129 | T bank2 | 1.5 / 0.0 / 0.0 | 1.3 / 1.2 | 1.2 / 0.2 / 1.5 / 1.8 | 1.0 | 0.0 | 0.0 |
| 73129 | D bank2 | 90.6 / 93.6 / 95.9 | 17.2 / 17.5 | 7.3 / 2.8 / 29.3 / 30.3 | 58.7 | 93.3 | 49.4 |
| 93683 | T bank2 | 1.9 / 0.0 / 0.8 | 1.8 / 0.7 | 1.3 / 0.7 / 0.2 / 0.5 | 0.7 | 0.9 | 1.0 |
| 93683 | D bank2 | 92.5 / 94.3 / 92.3 | 17.3 / 15.9 | 3.8 / 1.7 / 29.8 / 28.3 | 53.3 | 90.3 | 46.4 |
| 116089 | T bank2 | 1.9 / 1.8 / 0.9 | 0.9 / 0.6 | 0.3 / 0.5 / 1.3 / 0.2 | 1.0 | 0.9 | 1.0 |
| 116089 | D bank2 | 94.8 / 93.4 / 94.4 | 21.2 / 20.5 | 18.5 / 9.3 / 29.2 / 25.0 | 60.0 | 93.2 | 47.5 |
| 73129 | E16 windowed twin, for scale | — | — / 93.3 | 87.0 / 86.8 / 99.8 / 99.5 | 93.5 | — | — |

Verdicts. P17.1 refuted in every seed. P17.5's premise fails: the twin is at 16–21 on the held compositions after block 4, not 85, so the clause does not apply and the finding is the twin's. Two stacked fixed-rate banks under backprop through time learn single-operator items of three families to 90–96 and learn the two-operator scratchpad compositions to 16–21 by final answer (the max-min pairs 25–30, the plus-minus pairs 2–19, the within-pair panel of taught pairs 53–60) in the two blocks that took the windowed cell twin to 93. Phase 0b's gate was on single-operator births and never tested composition; the architecture question returns before the rule question is settled. P17.2 and P17.3 are not evaluable (the other windowless learners were not run; the learner holds nothing to extrapolate); P17.4 is void (nothing acquired). The twin's dev89 fell from 95 to 46–49 without replay of the birth's family, as the windowed twin's does.

**The probe (read-only, final weights, a fixed batch of development items).** Under the sign rule the state is silenced, not overdriven. Seed 73129: the first bank's largest |h| is 0.04–0.22 across the five rate groups against 2–6 at birth; the second bank's write gates average 0.03–0.08 with 15–48% of them below 0.01; the readout's drive from the state averages 0.10 against 3.3 at birth. Seed 116089 the same (second-bank gates 0.02–0.03 in the three slow groups, 25–57% shut); seed 93683 the opposite gate failure (second-bank gates 0.74–0.85, 3–8% saturated open) with the state at a third of the birth's scale. The twin keeps the birth's scale in every seed (first bank |h| up to 2–6, gates 0.07–0.64, readout drive 2.1–2.4). Every weight matrix moved by about its own norm in both arms, so the size of the move does not separate them; what does is that under the sign rule the network ends as a feedforward map from the current byte with the memory path closed. The hypothesis the probe was written to test, that the bank's unbounded state (up to 1/(1−R) times the input for the slow units) runs away under constant-size sign steps, is refuted: the slow groups never exceed |h| 0.3 in 73129.

**Phase 1b at block 2, the pre-registered verdicts (lives continue to block 6).** First bank frozen (`l1`, `lp`, `lc`, `lw` at birth values); the second bank and the readout learn.

| Seed | T frozen, F1 mean after block 1 / 2 / 3 | T frozen, add / sub after block 1 → 2 | D frozen, F1 mean after block 1 / 2 / 3 | D unfrozen after block 2 |
|---|---|---|---|---|
| 73129 | 70.7 / 21.3 / 16.1 | 56.5 → 9.5, 50.5 → 6.8 | 93.8 / 92.9 / 89.4 | 93.6 |
| 93683 | 71.4 / 15.2 / 11.9 | 60.5 → 2.0, 51.5 → 2.8 | 93.8 / 94.1 / 91.2 | 94.3 |
| 116089 | 80.6 / 31.8 / 13.7 | 70.2 → 7.0, 66.5 → 7.2 | 95.0 / 94.4 / 91.2 | 93.4 |

P17.1b.1 refuted (under 70 after block 2 in every seed). P17.1b.2 holds (the frozen twin within 1 point of the unfrozen). P17.1b.3's condition is met in every seed (under 40 after block 2 while the frozen twin learns): the truncated first-bank trace is not the cause. Beyond the clause: the frozen learner acquires F1 in block 1 (max/min at 86–93, add/sub at 50–70) and loses it in block 2, the first block in which half of every batch is replay from the reservoir; by block 3 addition and subtraction are at 1–2% while the comparisons hold 20–50%. So the sign rule with the exact second-bank trace can learn the operations on this architecture and cannot keep them under continued training. What differs between the frozen learner and its frozen twin is the feedback path (sign-symmetric against exact), the update through time (trace against backprop; the trace is exact here), the decay (0.25 against 0.3) and the data pipeline (replay from block 2 against the twin's joint stream). Phase 1c separates the first and the last.

**Phase 1b complete (all six lives, six blocks; 9 November 2026, afternoon).** The frozen learner does not recover: F1 after block 6 at 19.9 / 17.6 / 17.2 (add 1.0–3.2, sub 1.8–3.5, the comparisons 30–40), F3 at 18–20, held compositions 4–6, within 3–4. The frozen twin: F1 92.9 / 93.0 / 92.5, F3 89–91, held compositions 17–18, within 48–60, within 1–2 points of the unfrozen twin on every family (P17.1b.2 holds through block 6). The probe on the frozen learner's final weights shows a different end state from the unfrozen collapse: with the first bank fixed, the second bank's gates stay open (0.13–0.17 in the slow groups, none saturated), the state keeps the birth's scale (|h| up to 4–11), and the readout's drive from the state is 2.2–2.6 as in the twin. The memory path is alive and wrong, not silenced. So the unfrozen learner's silence (Phase 1) is produced through the first bank's learning, and the frozen learner's loss is a second failure in the learned path downstream.

**Phase 1c (launched under the lock when Phase 1b ended).** Arm `T_bank2_frz1_xfb_raw65536`: `E16_FEEDBACK=exact` replaces the sign-symmetric feedback by the true transpose through every layer while the trace and the pipeline stay as they are. Arm `M_bank2_frz1_raw65536`: the twin's update on the learner's pipeline. Three seeds each; predictions P17.1c.1–4 in `E45_PREDICTIONS.md`, written before launch. In parallel, the composition question goes back to Phase 0 form: births and twins of a windowless learner on the two-operator prompts under backprop, before any further local-rule life on bank2 is read as a result about the rule.

## E16-M2, assessed (cloud session, 9 November 2026): the data pipeline is not the cause; ideal data does not close the rule's gap

`E16M2_REPORT.md`, `exp45/SUMMARY_E16M2.md`. On the two failing seeds (400009 under drift, 600013 under stages): arm M, backprop on the learner's own reservoir (identical bytes and distinct counts to the failing life's), ends at 0.97 / 0.95 of the twin on held compositions with the plus-minus pairs at 76.5–86.3 and no mod-97 dip (P16M2.1 holds in both). The sign rule on the twin's joint stream (oracle replay) rises from 64.9 to 84.5 (0.90) and from 64.7 to 78.8 (0.85); the +- pair ends at 74.2 and 62.0 (P16M2.2 holds in 400009, misses in 600013 on the pair clause). Under stages the mod-97 dip when compositions arrive reappears with oracle data (65.7) and not in M (92.2); under drift the oracle arm shows no dip at checkpoint 2 and a fall to 79.2 at checkpoint 3 (P16M2.3 holds in 600013, misses in 400009). Reading: in both seeds the rule is short of the twin with ideal data, and in both the raw reservoir's sampling costs the rule a further 14–20 points that costs backprop nothing: a rule × data interaction, with the rule first. Caveat: 600013's comparators are on the pre-re-keying panels. E16-F6 (six more drift seeds, the failure rate, 600013 under drift, the early signature over twelve seeds) is running on the same worker; its three births passed (89.7% or better on the development operations).

## E17, Phase 1c: the pipeline is cleared, exact feedback only delays the loss, and the trace gap is measured (9 November 2026, 18:15 EST)

Predictions and verdicts: `E45_PREDICTIONS.md`, E17 Phase 1c. Six lives on the frozen bank2; tables `exp45/SUMMARY_E17_PHASE1.md`; gradient check `exp45/QUALIFY_E17_bank2_up.json`.

| Seed | Arm | F1 mean after block 1 / 2 / 3 / 6 | F3 after 6 | Held comp after 6 |
|---|---|---|---|---|
| 73129 | M, backprop on the learner's pipeline | 93.1 / 92.4 / 90.2 / 93.0 | 93.4 | 15.8 |
| 73129 | T, exact feedback, trace as before | 82.4 / 45.8 / 10.9 / 15.0 | 14.1 | 4.3 |
| 93683 | M | 94.0 / 95.5 / 89.0 / 92.5 | 92.7 | 15.2 |
| 93683 | T, exact feedback | 78.8 / 49.9 / 10.2 / 17.4 | 16.6 | 5.4 |
| 116089 | M | 95.1 / 95.2 / 90.8 / 92.1 | 92.9 | 17.7 |
| 116089 | T, exact feedback | 85.1 / 66.9 / 10.1 / 16.1 | 15.8 | 4.0 |
| all | T, sign feedback (Phase 1b), for comparison | 71–81 / 15–32 / 12–16 / 17–20 | 18–20 | 4–6 |
| all | D, frozen twin (Phase 1b) | 94–95 / 93–94 / 89–91 / 92.5–93.0 | 89–91 | 17–18 |

Arm M holds in every seed (P17.1c.2): replay from the reservoir does not hurt the exact update, so the learner's data pipeline is cleared on this architecture as it was on the windowed one (E16-M2). Exact feedback raises the learner's standing after block 2 from 15–32 to 46–67 and does not prevent the collapse, which completes one block later (P17.1c.1 refuted; P17.1c.3's condition met one block late). The gradient check then located the remaining difference: with the first bank frozen and exact feedback, the learner's gradient equals backprop through time's on every learned synapse except the projection feeding the second bank's write, where the within-step gradient has relative error 0.87; the scan extended one layer upstream reproduces the exact credit to 1.6e-7 (`arch17.qualify bank2 up`). The trace was exact for the write synapses and not for the layer feeding them, which the windowed cell never exposed because its input window made within-step credit sufficient. **Phase 1d** runs the extended trace under both feedbacks; predictions written before launch.

## Errata and the E17 stop, after the 9 November audit (Fable 5.2 memo; review 9.60)

- **E17 Phases 1, 1b, 1c: the local-arm readings are void.** Under the local arms the embedding rows of every prompt symbol that reaches the loss only through the state (operators, letters, task codes) received no gradient and were erased by AdamW's decay (final/birth = 9.9387e-5 on every element; 0.215 per block). The twin and arm M learned them. The Phase 1 silence and the Phase 1b/1c collapses are therefore not evidence about the rule; the twin's composition failure (16–21 held) stands. The write-projection gradient gap (0.87) stands as a measurement; its causal reading is withdrawn. **Phase 1d** (launched 18:05 EST with the same defect) was stopped at 18:12; partial files in `exp45/archive_e17d_stopped/`. Fix: `E16_BOOKDECAY=off` (tag `_nbd`; embeddings in a no-decay group in lives and births; the frozen arm freezes the table too; a zero-gradient report on the first update in every receipt). **Phase 2** (the corrected E17): births and backprop twins running; sign-rule arms pending Mr. Xu's go (review 9.60, section 2.2).
- **The "drift twin" is not trained on the drifting stream.** Arm D under `E16_STREAM=drift` draws equal shares from every family whose drift weight has become positive (`joint_stream`), as the staged twin does, at decay 0.3. Read "a backprop reference on the joint stream of the families admitted so far" wherever E16-S, E16-F5, E16-F6 or E16-M2 say "a twin trained on the same stream". The matched control for the rule is arm M.
- **The Phase A gate by item type** (`PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/gate_by_item_type.py`): of 21 raw-reservoir lives, 18 pass as pre-registered (family means), 14 on the whole worked answer, 15 on mod-97 add/sub alone, 4 on the `+-`/`-+` pairs alone (learner 64.5 against the reference's 86; the comparison pairs carry a 69% shortcut). The gate stands as pre-registered and the record carries the split.
- **E16-L:** add/sub 46.8 / 40.0, 29.8 / 22.5, 21.5 / 14.2 after block 1, then under 12 and under 6: acquired partly, then lost; not "never acquired".
- **E16-RP:** the run-to-run movement of the `+-` pair is about 9 points once the panel change (6.8) is removed; and E16-F6's two runs being bit-identical across two cloud containers narrows the non-reproducibility to the owner's machine against the containers.
- **E17 Phase 1b:** the frozen twin ranges 89.4–95.0 by block, not "93 throughout".
- **E16-F6:** one run; run B is a bit-for-bit repeat (`E16F6_RUN_B/`).
- **Held-panel keying:** fixed in `exp16_integrated.py`; `exp7_decon.py`, `exp8_life.py`, `exp8_e16w2.py`, `exp8_e16wr.py`, `exp8_e16wr2.py`, `exp10_ring.py` still key on `seed + 1` (frozen experiments; on Codex's list).

## E17, Phase 2: births and twins without embedding decay (9 November 2026, 19:40 EST; sign-rule lives running)

| Seed | Birth `bank2_nbd` add / sub / max / min at k = 12 | Twin `D_bank2_nbd` F1 / F3 after block 6 | Phase 1 twin F1 / F3 | Held comp after 6 (nbd / Phase 1) |
|---|---|---|---|---|
| 73129 | 93.0 / 89.3 / 97.7 / 94.3 | 92.3 / 92.6 | 95.9 / 93.3 | 15.5 / 17.5 |
| 93683 | 94.7 / 94.3 / 94.7 / 95.7 | 93.7 / 91.6 | 92.3 / 90.3 | 21.0 / 15.9 |
| 116089 | 95.7 / 85.0 / 97.0 / 94.0 | 93.3 / 91.9 | 94.4 / 93.2 | 17.8 / 20.5 |

P17.2.1 holds; P17.2.2 holds in two seeds and misses in 73129 by 0.6 point on F1. Keeping the embeddings out of decay changes the backprop reference by at most a few points and leaves its composition failure where it was. The nine sign-rule lives (P17.2.3–5) and the three frozen twins run under the lock; `exp45/SUMMARY_E17_PHASE2.md`.

## E17, Phase 2: the sign-rule arms with the embeddings kept (9 November 2026, 22:10 EST)

Three local arms on `bank2` with the embeddings out of decay, three seeds each, against the frozen twin in the same regime (`D_bank2_nbd_frz1`, added at launch). Family means at k = 12; held compositions by final answer; `exp45/SUMMARY_E17_PHASE2.md`.

| Arm | F1 after block 1 | F1 after block 2 | F1 after block 4 | F1 after block 6 | F3 after 6 | Held comp after 6 | add / sub after 6 |
|---|---|---|---|---|---|---|---|
| `T_bank2_nbd_raw65536` (sign rule, both banks) | 2.8 / 1.6 / 1.5 | 1.1 / 0.6 / 0.8 | 0.5 / 1.1 / 1.7 | 0.7 / 1.1 / 1.2 | 1.1 / 1.1 / 2.2 | 0.7 / 0.6 / 1.0 | 1 / 1, 1 / 1, 1 / 1 |
| `T_bank2_nbd_frz1_raw65536` (sign rule, first bank and embeddings frozen) | 81.4 / 86.6 / 84.2 | 82.1 / 86.9 / 83.9 | 50.8 / 52.0 / 53.8 | 59.4 / 62.4 / 61.2 | 67.8 / 68.2 / 71.6 | 5.8 / 9.2 / 8.5 | 41 / 24, 46 / 28, 46 / 23 |
| `T_bank2_nbd_frz1_xfb_tup_raw65536` (frozen, exact feedback, trace to the projection) | 93.1 / 93.9 / 94.5 | 93.7 / 93.6 / 95.0 | 84.9 / 88.5 / 88.9 | 87.5 / 90.1 / 91.9 | 91.4 / 90.7 / 90.9 | 13.4 / 14.3 / 15.0 | 84 / 85, 90 / 89, 91 / 89 |
| `D_bank2_nbd_frz1` (frozen twin) | — | 94.1 / 95.0 / 93.1 | 85.0 / 89.6 / 91.8 | 90.2 / 90.0 / 91.5 | 88.6 / 88.0 / 88.9 | 14.9 / 17.8 / 18.1 | 90 / 86, 93 / 88, 90 / 91 |

Verdicts: P17.2.3 misses in every seed; P17.2.4 holds in 93683 and 116089 and misses 73129 by 0.6 on the block-6 clause; P17.2.5 holds in every seed; P17.2.6 reported (hard pairs `+-` / `-+` at 1–4 in the local arms and 1–11 in the twin). First-update zero-gradient report: 36 of 47 byte rows in the unfrozen local arm, 2 in backprop; embedding rows without gradient end at exactly their birth values. In the unfrozen local arm the digit and separator rows, the ones the rule reaches, shrink from row norm 4.1 to 0.3 with sign flips during block 1, and the development operations fall from 93–95 to 1–2: the rule's own embedding update destroys the birth. CPU-seconds per life: 7,475–7,515 (unfrozen), 6,853–6,923 (frozen), 6,667–6,808 (exact feedback, extended trace), 5,611–5,738 (frozen twin); wall 2 h 29 min for twelve processes. Review 9.61.

## E17, Phase 3: the two changes one at a time, and the embeddings alone frozen (10 November 2026, 01:05 EST)

Family means at k = 12; held compositions by final answer; three seeds 73129 / 93683 / 116089; `exp45/SUMMARY_E17_PHASE3.md`. Comparators from Phase 2 in the last three rows.

| Arm | F1 after block 1 | F1 after block 2 | F1 after block 4 | F1 after block 6 | F3 after 6 | Held comp after 6 | add / sub after 6 |
|---|---|---|---|---|---|---|---|
| `T_bank2_nbd_frz1_xfb_raw65536` (frozen, exact feedback alone) | 88.8 / 88.9 / 89.0 | 84.9 / 89.8 / 90.4 | 59.1 / 64.5 / 66.8 | 67.9 / 72.9 / 72.8 | 81.8 / 78.8 / 78.6 | 8.8 / 8.3 / 8.8 | 54 / 42, 63 / 50, 64 / 48 |
| `T_bank2_nbd_frz1_tup_raw65536` (frozen, trace to the projection alone) | 90.7 / 93.8 / 91.0 | 90.8 / 93.2 / 92.4 | 78.1 / 80.7 / 84.2 | 80.1 / 84.9 / 84.9 | 87.6 / 88.4 / 86.6 | 12.2 / 13.2 / 12.0 | 78 / 66, 84 / 77, 83 / 72 |
| `T_bank2_nbd_frzb_raw65536` (embeddings alone frozen, sign rule on both banks) | 1.8 / 1.8 / 1.1 | 1.3 / 0.5 / 1.4 | 0.2 / 0.9 / 1.2 | 1.3 / 1.2 / 1.1 | 1.9 / 2.6 / 2.0 | 1.0 / 2.0 / 1.7 | 1 / 2, 1 / 1, 1 / 1 |
| Phase 2 `T_bank2_nbd_frz1_raw65536` (frozen, sign rule) | 81.4 / 86.6 / 84.2 | 82.1 / 86.9 / 83.9 | 50.8 / 52.0 / 53.8 | 59.4 / 62.4 / 61.2 | 67.8 / 68.2 / 71.6 | 5.8 / 9.2 / 8.5 | 41 / 24, 46 / 28, 46 / 23 |
| Phase 2 `T_bank2_nbd_frz1_xfb_tup_raw65536` (both changes) | 93.1 / 93.9 / 94.5 | 93.7 / 93.6 / 95.0 | 84.9 / 88.5 / 88.9 | 87.5 / 90.1 / 91.9 | 91.4 / 90.7 / 90.9 | 13.4 / 14.3 / 15.0 | 84 / 85, 90 / 89, 91 / 89 |
| Phase 2 `D_bank2_nbd_frz1` (frozen twin) | — | 94.1 / 95.0 / 93.1 | 85.0 / 89.6 / 91.8 | 90.2 / 90.0 / 91.5 | 88.6 / 88.0 / 88.9 | 14.9 / 17.8 / 18.1 | 90 / 86, 93 / 88, 90 / 91 |

Verdicts: P17.3.1 misses in every seed; P17.3.2's first clause holds (21–24 above the sign rule) and its second misses, the trace arm ending 12 points above the exact-feedback arm; P17.3.3 misses in every seed (development operations 1.3 / 1.5 / 1.2 after block 1 with the embeddings frozen: the rule's update to the first bank destroys the birth on its own); P17.3.4 reported. Ratios to the frozen twin after block 6, trace alone: mod-97 0.89 / 0.94 / 0.93, mod-73 0.99 / 1.00 / 0.97, held compositions 0.82 / 0.75 / 0.66. First-update zero-gradient report: 47 of 47 byte rows in every arm (the table is frozen in all three). CPU-seconds per life 4,840–5,602; wall 1 h 35 min for nine processes. Review 9.66.

## E16-F7 and E16-M3, the two cloud jobs of 9 November (filed 10 November 2026, 10:30 EST)

Both ran whole in one session each, with predictions committed before launch and the harness unchanged; their reports are `E16F7_REPORT.md` and `E16M3_REPORT.md`, and the headline numbers were recomputed here from the receipts.

**E16-F7.** Six new drift seeds, all passing the family gate (16 of 18 over all drift seeds), one passing on the `+-`/`-+` pairs alone (4 of 18). The signature S, pre-registered this time, misses its correlation clause (Spearman 0.109 over eighteen seeds against 0.35): it separates the two failures it was chosen on and does not rank the rest, and is dropped. Two new seeds lose 15 and 30 points of `+-` during the mod-73 blocks, after S's window, while their mod-97 operations hold.

**E16-M3.** The replay share, 0 to 0.75, on three seeds. The predicted dose is reversed: the rule's hard pairs rise with the share (400009: 43.7 / 46.3 / 73.2 at 0.25 / 0.5 / 0.75; 600013: 23.3 / 44.8 / 60.2; 300007: 59.2 / 67.7 / 75.7), backprop's rise too by a third as much, and with no replay everything before the current mixture is lost (hard pairs about 1%). The share works through retention: the peak by checkpoint 4 hardly moves with the share, the end of the life does. At 0.75 the failing seed 400009 passes both the family gate and the hard-pair gate. One life per seed and share. Follow-up brief: `OPUS_TASK_E16M4_CLOUD_JOB_20261110.md`.

## E17, after Mr. Xu's answers: why the rule loses the first bank, and Phase 4 prepared (10 November 2026, 10:30 EST)

Mr. Xu's answers (review 9.67): a frozen first bank does not meet the aim; feedback relayed layer by layer or by a sub-network is allowed. A read-only measurement at birth (`e17_align_at_birth.py`, `exp45/E17_ALIGN_AT_BIRTH.md`) then asked why the unfrozen learner fails. Under the sign rule as run, the first bank's projection receives an update with cosine 0.09 / −0.06 / 0.05 to the true gradient, the embeddings 0.13–0.32, the first layer 0.16–0.32; exact feedback leaves these where they are; carrying credit through time at every layer, with weight-sign feedback kept, puts every layer at 0.45–0.81. The loss of the birth in Phases 2 and 3 is the truncation of credit in time above the write synapses, not the weight signs. Phase 4 (`E16_TRACEUP=all`; nine lives, nothing frozen; predictions P17.4.1–5) is prepared and waits for the go.

## E18 prepared for a cloud worker: a next-byte target at every bank (10 November 2026, 12:00 EST)

Mr. Xu's question of 10 November (delayed feedback from the stream as a source of local targets) is turned into a first measurement: births of the two-bank learner from nothing under four rules, one with the error relayed between banks by weight signs, three with a next-byte head at each bank and no relay at all. New options `E16_HEADS=local` (tag `_lh`) and `E16_BIRTHRULE=T` (tag `_bT`); `e18_check.py` and `exp45/E18_CHECK.json` show the harness unchanged to the last bit with the options unset, no gradient passing between the banks with local heads under either rule, and the shared layers starting from the plain network's weights. Dry runs of the four births (512 updates) and of two lives from them are in `exp45/archive_e18_dry/`. Brief and predictions P18.1–6: `OPUS_TASK_E18_LOCAL_HEADS_BIRTHS_20261110.md`. References on file: two banks end to end by backprop, 85–98 on the four development operations at k = 12; one bank alone, 22–30 on addition and 9–13 on subtraction. No life is part of this job.

## E17, Phase 4: nothing frozen, credit through time at every layer (10 November 2026, 13:00 EST)

Nine lives, three seeds, launched on Mr. Xu's go at 10:31 EST; tables in `exp45/SUMMARY_E17_PHASE4.md`, verdicts in `E45_PREDICTIONS.md`, reading in review 9.70. With weight-sign feedback and credit carried through time at every layer (`E16_TRACEUP=all`), the unfrozen two-bank learner keeps its birth (mod-97 at 91–93 after block 1, where the rule as run fell to 2) and ends at 0.92–0.95 of the unfrozen twin on mod-97 and 0.95–0.98 on mod-73, above the frozen learner of Phase 3 in every seed. Backprop on the same pipeline ends at 0.98–1.02 of the twin, 3–7 points above the new arm, the gap lying in addition and subtraction. The one-layer trace with nothing frozen survives block 1 at 65–75 and ends at about half the twin. Nothing composes in any arm. The new arm holds less of the unrehearsed development operations than backprop on the same pipeline (11–43 against 41–54). The claim of Phase 3 that the first bank must be frozen is withdrawn. The arm is computed by a backward pass over the item; its standing under the locality constraint is not settled.
