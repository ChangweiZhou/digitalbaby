# E4 (retention) and E5 (composition): design and predictions, stated before launch

**7 November 2026, before any run.** Learner for both: the harness window network (10-byte window, hidden width 256, about 155,000 parameters), stateless. The window stands in for working state at this horizon so that the life rule can stay fully local (sign-symmetric feedback within the step). Development by backprop (AdamW lr .001, decay 0.3, 24,576 updates). Life by the sign rule (AdamW lr .001, decay as stated per arm). Three seeds. Every sealed panel is scored after every stage. Shortcut ladder and shuffled control in every table.

**Addendum, same day, before launch (from Mr. Zhou):** the learner may edit its own local rules during life (learned per-synapse learning rates, gated or metaplastic writes, learned consolidation schedules). This is inside property P5 provided the edit itself uses only local information and broadcast signals. E4 and E5 as designed below do not use it; it is the first thing to add if arm C's self-generated replay proves the weak point (a learned write gate on the store), and it is a candidate component of E6.

## E4, retention

Development family: addition and subtraction mod 83 and mod 89 (codes p, q), half the pairs. Life: four new operations learned **in sequence**, 8,192 updates each, 25% of pairs each, no access to earlier tasks' data: T1 addition mod 97 (code r), T2 subtraction mod 97 (code t), T3 addition mod 73 (code s), T4 subtraction mod 73 (code u). After every stage, score sealed held panels of all four life tasks and of the development tasks.

| Arm | Life learner |
|---|---|
| A | shared path alone, sign rule, decay 1.0 (the setting of Experiments 1–3) |
| A0 | shared path alone, sign rule, decay 0.1 (separates decay-driven forgetting from interference) |
| B | shared path + fast store. **Changed at qualification, before launch:** a random sparse expansion (4,096 units, top 5%) could not separate windows sharing nine of ten bytes (the same overlap problem as the project's KC code), so the store is an exact-key episodic memory on the 10-byte window at each answer position, the perfectly separated limit of such a code (the project's HASH8), with a +6 logit on the majority stored byte and nothing for unseen keys, capacity 65,536, first-in first-out. Logits summed with the shared path; no replay |
| C | B + self-generated replay: a buffer of up to 1,024 earlier taught prompts per task; from stage 2 on, half of every batch is replayed prompts answered by the model's own output (shared path + store), not by ground truth. With an exact store the replayed answers are mostly correct, which is the complementary-systems claim: the fast memory teaches the slow path |
| C′ | C with the replayed answers replaced by random valid answers (rehearsal of inputs without content) |
| D | backprop ceiling: joint training on all tasks seen so far with ground-truth data (no forgetting by construction) |

Predictions:
- P4.1 Arm A forgets: T1 sealed accuracy falls below 30% by the end of stage 4, from at least 70% at the end of stage 1.
- P4.2 Arm A0 forgets less than A (decay is part of the forgetting) but still falls below 50% on T1.
- P4.3 Arm B retains **taught** T1 items (the store answers them by construction) but its **sealed** T1 accuracy tracks A within 10 points: the store protects memories, not the rule.
- P4.4 Arm C keeps T1–T3 sealed accuracy at or above 70% after stage 4, within 15 points of its end-of-stage value, while learning T4 to at least 70%.
- P4.5 Arm C′ tracks A: replaying inputs without content does not retain.
- P4.6 Arm D keeps all tasks at or above 85%.
Decision: E6 uses arm C's design if P4.4 holds in all seeds; if C fails but D passes, the self-generated answers are the weak point and E6 waits for a store that answers better.

## E5, composition

Development: single-step addition and subtraction mod 97 (codes r, t), half the pairs. Two-step problems are written `c a o1 b o2 c =` with code byte `c` for "compose", answer `(a o1 b) o2 c mod 97`. Operation types: (+,+), (−,−), (+,−), (−,+). Taught types in life: (+,+) and (−,−), 8,000 triples each. Held: (i) 600 new triples of the taught types (within-type), (ii) 600 triples of each untaught type (cross-type). Two renderings:
- **Direct**: the target is the final answer bytes and EOS.
- **Scratchpad**: the target is the intermediate result, a separator, then the final answer and EOS (`08;01`). At test the model must emit the intermediate itself; scoring is exact on the final answer with own EOS, intermediate correctness reported separately.

| Arm | Development | Life (8,192 updates, sign rule, decay 1.0) |
|---|---|---|
| Z | single ops | nothing (zero-shot on compositions) |
| DIR | single ops | direct compositions of taught types |
| SCR | single ops | scratchpad compositions of taught types |
| SCR-sh | single ops | scratchpad with shuffled final answers (control) |
| DIR-bp, SCR-bp | single ops | the same teaching by backprop (ceilings) |

Predictions:
- P5.1 Z scores under 5% on every composition panel (format never seen).
- P5.2 DIR cross-type is below 30% in every seed; DIR within-type is at least 60%.
- P5.3 SCR cross-type is at least 70% in every seed, and SCR within-type at least 80%; intermediate correctness at least 85%.
- P5.4 SCR-sh is under 10% on finals.
- P5.5 Single-op panels do not fall below 80% in any life arm (no forgetting of the parts within one stage).
Decision: if P5.3 holds and P5.2 holds, scratchpad composition is the form E6 uses; if DIR cross-type also passes, direct composition is used and the scratchpad becomes the fallback; if SCR fails cross-type, composition is not achieved at this scale and E6 is postponed.

## E5-R, composition with replay of the single operations (added 7 November after E4 and E5, before its launch)

Same as arm SCR, with the episodic store holding the single-step items the learner experienced in development (its taught halves of add97 and sub97) and half of every life batch being replayed single-step prompts answered by the model itself (shared path plus store), exactly as in E4 arm C. Arm SCR-R uses the sign rule; SCR-R-bp the same teaching by backprop. Compositions are never written to the store; composition panels are scored by the shared path alone.

Predictions:
- P5R.1 Single operations retained at or above 80% after life in SCR-R (they fell to 0–7% in every arm without replay).
- P5R.2 SCR-R cross-type final-answer exact at or above 70% in every seed, with the intermediate right in at least 85%.
- P5R.3 SCR-R within-type at or above 85%.
- P5R.4 SCR-R-bp cross-type stays below SCR-R (backprop overfits the taught types even with the parts retained), or, if it does not, the earlier gap was retention alone.
Decision: if P5R.2 holds in all seeds, E6 runs as designed (sequence of operations with replay, then compositions, then the early operations scored again). If it fails with the single operations retained, scratchpad composition itself is the limit at this scale and E6 is postponed.

## E5-C, canonical scratchpad (added 7 November after E5-R, before its launch)

The learner rewrites each step as a taught single-operation prompt and solves it: `c24+81-07=` → `r24+81=08;t08-07=01` EOS. The final step's window is then `08;t08-07=`, a taught subtraction prompt with leading bytes, and contains no first operator. Development for this experiment prefixes each single-operation record with 0–3 random letters so the network learns to ignore leading bytes (a stated design change; bare-prompt development never showed leading bytes). Arms under this development: SCR (plain scratchpad, as E5), SCRc (canonical, sign rule), SCRc-sh (canonical with shuffled finals), SCRc-bp (canonical, backprop). No replay (E5-R showed it hurts composition learning at this budget). Same taught types (+,+) and (−,−), same held panels.

Predictions:
- P5C.1 SCRc cross-type final-answer exact ≥70% in every seed.
- P5C.2 SCRc rewritten step prompts correct ≥85% on cross-type (both steps).
- P5C.3 SCRc-sh under 10%.
- P5C.4 SCR under this development stays ≤60% cross-type (replicating E5).
Decision: if P5C.1 holds in all seeds, P4 is established in canonical-scratchpad form and E6 resumes with that form plus E4's replay. If the rewritten prompts are right (P5C.2) but the finals fail, the leading-byte invariance is the limit; if the rewriting itself fails cross-type, the copy rule does not generalize across operator positions.

**Amendment before the E5-C rerun (same day).** The first E5-C run failed in every arm including the backprop ceiling (0–2% within-type). Teacher-forced accuracy was near 100% at every target position except the third operand of the rewritten second step (8–13%): by the time the learner writes it, the original operand has left the ten-byte window, so the rewrite is impossible by construction, a design error on my part. The rerun uses an 18-byte window in every arm (development and life, plain and canonical), which holds the whole problem plus the step being written. The first run's receipts stay in the folder as `e5c_*` without a tag; the rerun's carry `_w18`. Predictions P5C.1–P5C.4 are unchanged.

## E6-route: learned-key routing on the canonical scratchpad (added 7 November after E5-C, before launch)

RouteNet: four attention heads over the 18-byte window; query from the window, keys = learned slot-position embeddings, value = the slot's byte embedding; output head sees the four attended bytes plus a 16-d window summary ("att") or the attended bytes only ("attv"). All projections go through the sign-feedback path in life. Development with left noise on single operations, as E5-C. Arms: att, attv (sign rule), attsh (att with shuffled finals), attbp, attvbp (backprop). Same canonical scratchpad, same panels.

Predictions:
- P6.1 attbp cross-type final ≥70% in every seed.
- P6.2 att cross-type within 20 points of attbp.
- P6.3 attv cross-type ≥ att cross-type (the values-only head is more invariant).
- P6.4 attsh <10%; within-type ≥85% in att and attv.
Decision: P6.1 and P6.2 both held → P4 established with routing; E6 resumes with routing + canonical scratchpad + E4 replay. attbp fails → routing of this form is insufficient and the composition line stops here.

## E7, de-confounded composition with four operators (added 7 November, before launch)

Operations mod 97 with codes and symbols: add `r` `+`, sub `t` `-`, max `x` `>`, min `n` `<`. Development: all four single-step operations, half the pairs each, with 0–3 random leading letters, 18-byte window, hidden 256, backprop. Life: two-step problems `c a o1 b o2 c =`; the 16 ordered operator pairs are split into 12 taught (2,000 triples each) and 4 held: (+,−), (−,+), (>,<), (<,>). Every operator appears in both positions among the taught pairs. Held panels: 600 new triples per held pair; a within-type panel of 600 new triples pooled over four taught pairs. Arms: SCR plain scratchpad and SCRc canonical scratchpad with the sign rule; SCRc-sh shuffled finals; SCR-bp and SCRc-bp backprop ceilings. 8,192 updates, three seeds, shortcut ladder as before.

Predictions:
- P7.1 SCRc held-pair final-answer exact ≥70% in every seed.
- P7.2 SCR held-pair ≥40% in every seed.
- P7.3 SCRc-sh <10%.
- P7.4 SCRc-bp held-pair ≥85%.
Decision: P7.1 holds → P4 achieved in the de-confounded regime, the four earlier failures are explained by the confound, and step 2 is skipped; the persistent-life demonstration (step 5) then includes compositions. P7.1 fails but P7.4 holds → the local rule is the limit; test it at width 512. Both fail → binding needs a mechanism; step 2.

## E7-b, the local rule at width 512 and at double dose (added 7 November after E7, before launch)

E7 result: backprop plain scratchpad 90–93% held-pair mean (P4 achieved under backprop; the confound explained the earlier failures); sign rule 37–38% plain and 7–67% canonical, with within-type only 39–42% plain, so the local rule is under-trained on the twelve-pair teaching set at 8,192 updates. By the rule on file the local rule is tested at width 512. Two arms, plain scratchpad with the sign rule: (i) width 512, 8,192 updates (new births at 512); (ii) width 256, 16,384 updates. Three seeds each. Backprop at 256 remains the reference.

Predictions:
- P7b.1 width 512, sign rule, plain: held-pair mean ≥60% in every seed.
- P7b.2 width 256 at 16,384 updates, sign rule, plain: held-pair mean ≥50% in every seed.
Decision: either holds → P4 is achieved under the local rule in the de-confounded regime, and step 5 (persistent life) includes compositions. Both fail → the local rule's ceiling on composition is below backprop's at this scale; report it and proceed to step 5 without compositions, leaving steps 2–4 as options.

**Clarification from Mr. Zhou, 7 November, during E7-b.** A local rule does not mean a fixed small neighbourhood of neurons; which neurons a synapse takes its signals from may be adjusted dynamically during life. Consequences for the ladder: (1) learned feedback pathways (feedback weights that adapt during life, as in Kolen–Pollack or weight-mirror rules) are inside P5 and are the natural next arm if the sign rule stays below backprop on composition; (2) credit through time carried by dynamically weighted traces across neurons, rather than each unit's own gate, is inside P5 and is the natural next arm for the working-state rung. Neither is run yet; E7-b (width and dose) completes first.
