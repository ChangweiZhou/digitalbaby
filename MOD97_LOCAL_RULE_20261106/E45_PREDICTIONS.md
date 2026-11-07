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
