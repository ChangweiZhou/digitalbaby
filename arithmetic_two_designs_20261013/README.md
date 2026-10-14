# Two arithmetic-learning designs

13 October 2026. Both are **design only**. No new learner implementation, prototype, training trajectories, three-cycle qualification, background run, adoption or GitHub upload is included in this delivery.

The shared goal is unchanged: the learner discovers arithmetic from actual byte feedback and improves retained exact own-output accuracy on unseen cases per external sample and production compute. Arithmetic-specific information can be supplied as examples in the data, never as a parser, numeric embedding, operator branch, arithmetic function or output renderer in the learner.

| | Design A: RAW-RULE V2 | Design B: STREAM-TRANSFER V1 |
| --- | --- | --- |
| Document | [Experiment](../arithmetic_feedback_rule_discovery_design_v2_20261013/EXPERIMENT.md) | [Experiment](../arithmetic_stream_transfer_design_v1_20261013/EXPERIMENT.md) |
| Immediate question | Repair weak acquisition with coherent joint credit and generic byte access; does retained four-operation E3 improve at the old sample budget? | Does all-byte learning with paid counting/repetition produce retained E3 and addition-to-multiplication transfer in one unchanged core? |
| Learners | GRU and ordered-window MLP; squared/CE objectives, frozen controls, exact RAW/legacy references | One GRU; 2×2 all-byte/answer-only CE × no-decay/decay |
| Main candidate learning | Joint record-level backprop, answer feedback only | Joint record-level backprop; every actual payload byte/EOS, same throughout tasks |
| Data | Existing 0–7 four-operation family split; 512 old records per root | Counting→0–9 addition→multiplication; 35 train/20 held orbits per task, 256 repeated passes |
| Rules in learner | None; legacy numeric reference explicitly not selectable | None; counting/order is data and paid |
| Retention | Paid unrelated byte cohort and delay at each sample checkpoint | Same generic core on unrelated byte cohort; addition also tested after multiplication |
| Planned DEV | 8 worlds×8 configurations =64 jobs | 2 worlds×4 configurations =8 jobs |
| Conditional confirmation | 48 worlds×6 configurations =288 jobs | 16 worlds×3 configurations =48 jobs |
| Conditional maximum | 352 jobs /743,424 executed record applications | 56 jobs /8,107,008 executed record applications |
| What a pass establishes | Bounded four-operation retained E3 improvement under its range/budget | Bounded two-task retained E3, cross-task transfer and improvement over matched answer-only recipe; no four-operation/mixed-expression claim |

A is preserved at its existing canonical location with no changes to its learner/task/budget/gates. B is a distinct alternative data-and-learning regime, not a replacement endpoint or an extra hidden A arm. Different ranges, feedback targets and repetition budgets preclude a causal cross-design winner comparison.

The supplied proposal is reviewed in [REFERENCE_REVIEW.md](../arithmetic_stream_transfer_design_v1_20261013/REFERENCE_REVIEW.md). Its lookup-only diagnosis is false for the current RAW-RULE; its grokking/transfer claims are hypotheses. A successful large-repetition result cannot be reported as few-total-sample efficiency by counting only unique facts or granting free counting/addition.

Both designs link to the complete inventory: 108 historical mechanisms, 136 catalogue rows, 254 version records, the older overlapping 71 subset and first 16 roster. These views are not additive or a compressed survivor count. No whole family is eliminated by this delivery.

No runtime promise is made. Before any future science, measured full-budget technical jobs, independent audits, three-cycle qualification and a projected eight-hour admission/ten-hour hard session boundary are required. There is no authorized prototype in this delivery. Negative/no-advance results would be valid.
