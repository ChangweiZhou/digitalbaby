# RAW-RULE V1: why taught questions fail

This is a read-only review of the saved final technical world 202610130103. No native learning trajectory, candidate training, source edit or scientific selection was executed. Pure checkpoint inference and explicitly requested post-hoc label-oracle algebra are separate from the frozen preparation evidence. It is not a population estimate or causal intervention experiment.

## The failure is answer content, not mainly termination

At 128 examples per operation, the learned depth-1 FULL core answers 41/199 taught questions exactly. Supplying the true earlier answer bytes during evaluation still gives exactly 41/199 completely correct byte/EOS sequences. Depth 4 gives 40/199 in both conditions. All six new configurations predict the actual EOS correctly on **100% of teacher-forced terminal positions**. Depth 1's first byte is correct on 62/199 (31.2%); its teacher-forced answer-payload bytes are only 27.3% correct. Depth 4 scores 27.6% and 24.5%. Final empty outputs number zero and one for the two learned FULL cores.

R1's 137 first-byte errors explain 137 of its 158 failed complete answers (86.7%), before an incorrect generated answer prefix can compound. The EOS mechanism knows how to end a correct observed sequence. A stop-token mask or a longer output cap would not fix those first-byte content failures. Equality of own-output and complete teacher-forced correctness is expected for deterministic greedy inference and is only a consistency check, not independent causal evidence against compounding. See [the interpretation correction](INTERPRETATION_CORRECTION.md). Per-position choices, rather than that equality, support the diagnosis.

## The prescribed head works mathematically, but fits weak features poorly

For the frozen depth-1/depth-4 FULL encoders, the current head satisfies the ridge normal equations at relative residuals 8.39e-15 and 1.44e-15. The confidence inverses satisfy their fixed-feature equations at 6.54e-13 and 2.52e-13. The FULL RLS rule is therefore implementing its intended fixed-feature ridge regression, rather than failing to deliver answer targets to the head. Yet frozen taught recall is only 48/199 and 36/199.

With `P=10I`, the exact fixed-feature objective has ridge coefficient 0.1. Frozen final cue features have true matrix rank 65, but much of their discriminative variation lies in weak directions. All 199 cues are not literally one address, and low effective rank is not low mathematical rank.

Post-hoc oracle fits use the evaluator's known answers after the experiment and never change any stored model. For learned depth 1, fitting **only first-byte targets** to its final features improves first-byte accuracy from 43.7% with ridge 0.1 to 71.9% without ridge. Frozen depth 1 improves from 42.2% to 68.3%. These numbers show that useful weak feature directions are suppressed by the prescribed regularization. They do not prove that an unregularized classifier would generalize.

Jointly fitting every answer byte and EOS to the same final readout is much harder. On unique taught contexts, learned depth 1 improves only from 45/199 (22.6%) completely teacher-forced-correct sequences with ridge 0.1 to 69/199 (34.7%) without ridge. Frozen depth 1 improves from 43/199 to 62/199. Consequently, a ridge change alone does not solve complete taught recall. Shared-phase fitting plus the given representation is a material limitation of this recipe. Least-squares oracle results are not an impossibility proof for every linear classification objective or all nonlinear readouts.

## Common suffixes and extra recurrent processing compress cue distinctions

For learned depth 1, the mean pairwise cue-feature cosine rises from 0.542 before the final `=` byte to 0.824 after it, then 0.929 after the public REPLY marker. The frozen encoder behaves similarly (0.546 → 0.839 → 0.931). These are ordinary sensory/control tokens in the learner; their common content dominates the final recurrent summary.

The additional three tied zero-input transitions in depth 4 further compress distinctions. Learned depth 4's final mean cosine is 0.969, its centered feature effective rank is 2.28 versus depth 1's 4.93, and its first-byte cue matrix condition number is about 26,444 versus 2,751. Both matrices remain full rank 65. There is no exact alias of conflicting answer-first-byte cues: their nearest feature distance minima are 0.0515 and 0.0242, and their median nearest conflicting distances are 0.1034 and 0.0547.

Analytical Jacobians on four-payload-byte questions give depth-1 mean feature sensitivity to the first cue byte, second operand-position byte and REPLY embedding of 0.444, 0.887 and 2.502. The computation uses positions only in the external reviewer, not in the learner. It supports an ordinary recency/common-context bias, not an arithmetic parser bug. Geometry by itself is correlational; it does not prove which repair will succeed.

## Plastic features and the accumulated confidence no longer describe the same regression

The learned depth-1 encoder moves final cue features a mean Euclidean distance 0.203 from birth; depth 4 moves them 0.180. Relative encoder parameter displacement is 11.2% and 14.2%. Historical RLS samples were observed under different feature coordinates. At the current fixed basis, learned heads have ridge normal-equation residuals 0.237 and 0.258, rather than the frozen arms' roundoff residuals. Their corresponding confidence-inverse residuals are 20.60 and 26.13.

This mismatch is a real consequence of the declared learning design, not an unexpected code exception. It prevents treating `P` as the exact posterior of the current learned feature map. It can contribute to acquisition failure, but **cannot be the sole explanation** because frozen arms, where the regression identity is exact, also recall poorly. Learned depth 1 actually scores below its frozen control in this fixture.

## What the next design should change, without arithmetic helpers

Preserve the raw-byte, arbitrary-answer interface and own-output held-family evaluation. The next mechanism comparison should address (1) stable access to early cue bytes and (2) a categorical output objective coordinated with feature learning. Generic causal access/attention or a learned skip route can preserve raw byte identity without specifying digits, operands or operations. Jointly trained categorical encoder/readout learning is a clean alternative to accumulating confidence in a moving feature basis. A fixed-feature version and an objective-only change can distinguish representation access from readout-learning effects.

Do not fix this by supplying operand slots, arithmetic values, relation targets, answer candidates or a numeric formatter. Do not claim memorization recovery equals unseen arithmetic learning. Taught acquisition is a prerequisite and diagnostic; the target remains retained unseen arithmetic per external sample and charged compute.

The next comparison should use the existing sample prefixes and record counts and report payload accuracy, complete own output, termination, feature drift and actual compute. It should not launch another large formal run while the proposed recipe demonstrably fails the same-budget taught acquisition checks. These technical negatives constrain this exact GRU-plus-RLS recipe; they do not scientifically exclude generic recurrence, attention, Hebbian learning or entire inventory families.

Artifacts: `CHECKPOINT_REVIEW.json`, `ANALYTIC_HEAD_DIAGNOSTIC.json`, `review_checkpoints.py`, `analytic_head.py`. Source anchors: `core.py` for normalization, tied processing, RLS and encoder updates; `interface.py` for actual-byte observation and read-only probes; `runner.py` for permitted branches and immutable checkpoint forks.
