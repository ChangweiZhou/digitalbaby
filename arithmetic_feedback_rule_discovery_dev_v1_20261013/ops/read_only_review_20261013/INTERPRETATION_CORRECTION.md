# Interpretation correction: complete teacher forcing and own-output exact

13 October 2026. This note corrects an interpretation in FINDINGS.md and independent initial critiques. The saved counts and checkpoint measurements do not change.

For a deterministic greedy autoregressive model, complete teacher-forced correctness and complete own-output correctness are necessarily equivalent when both evaluate the same tokens/EOS and inference is pure. If every conditional choice under the true prefix is correct, its free rollout follows that prefix by induction. If a conditional choice is wrong, that complete response is already incorrect. Thus the equality 41/199 versus 41/199 cannot independently show that teacher forcing fails to rescue compounding. It is a consistency check, not the causal localization claimed by the initial wording.

The informative measurements are the **per-position** choices:
- R1 first-byte correct: 62/199.
- R1 full own answer correct: 41/199.
- Consequently 137 first-byte errors account for 137 of 158 complete-answer failures, approximately 86.7%. These occur before a generated answer byte can distort the prefix.
- The remaining 21 failures occur after a correct first byte and require the later conditional/error traces for their localization.
- Correct-prefix EOS accuracy is 100%, while conditional payload accuracy is only 27.3%. This separates observed content discrimination from the model's ability to stop on correct prefixes; it does not promise EOS under wrong prefixes.

The V2 design uses this valid first-error evidence. It still measures free rollout, teacher-forced conditional choices and output termination separately. All exact and algebraic data remain diagnostic technical evidence, not formal efficacy or a unique causal repair result. No new trajectory or native learning was performed to make this correction.

