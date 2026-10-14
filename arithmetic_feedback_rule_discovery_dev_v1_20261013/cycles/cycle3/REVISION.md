# Cycle 3 revision

Initial seven-arm testing reproduced weak acquisition/generalization rather than qualifying an arithmetic learner. The initial legal-empty-output unit fixture incorrectly made **all** EOS weights positive, although normalized signed features can sum to a negative value; corrected the fixture to use only the always-positive bias feature. No core change was needed.

Resource audit then found that the initial runner separated teaching CPU from wall time but did not itemize probe, checkpoint and auditor CPU. Added measured CPU categories and independently checked accounting; added actual pre/post component counters to every record. This is required cost reporting, not a learning change. Qualify this final runner on seven complete jobs using fresh technical world 202610130103; preserve initial world 202610130102 and its source snapshot. Re-run affected integrity tests before the final source lock. No reused trajectory, science seed, parameter search or fourth design cycle.

Readiness remains a separate decision. No relaxed percentage gate, arithmetic helper, extra warmup, bank or target manipulation is introduced to improve the score. Both learned FULL recipes must earn their retained unseen-case/sample/compute evidence; current technical tests do not do so. The preparation can be mathematically verified and still not recommended for formal DEV.
