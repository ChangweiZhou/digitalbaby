# MiniFly research objective and evidence boundary

**Effective 26 September 2026.** This is the standing interpretation and experiment-design rule for all MiniFly work in this project, regardless of folder name. It does not alter any frozen design, data, or preregistered result. The historical [V88–V9.7 audit](V88_TO_V9_7_OBJECTIVE_DRIFT_AUDIT_20260926.md) explains why this rule is needed.

## Objective

Build a bounded, biologically informed learner that uses information acquired from earlier byte-level experience when it becomes useful later, despite delay and intervening learning. The eventual test is **reuse of learned information on a relevant, previously unreinforced input or task**, with the answer selected before feedback. The project does not seek only a larger table of rewarded cue–answer pairs. Exact-cue associative learning remains a necessary diagnostic and a legitimate component of memory; it is not sufficient evidence for the intended persistent learning core.

## Evidence levels and permitted claims

| Level | What is actually tested | Claim permitted |
|---|---|---|
| E0 — access | Byte order, sensory coding, readout, numerical and causal interface checks without a learned-value test | The interface works under the stated conditions |
| E1 — taught association | A cue/pair is explicitly reinforced, then the same relation is probed after time or other training | Acquisition, retention, revision, and interference for taught associations |
| E2 — changed presentation | The learned relation is probed on a prespecified, label-preserving byte presentation that was **never reinforced**; first response precedes feedback | Limited retrieval across that presentation change; **not** general rule or new-task transfer |
| E3 — held-out relation | The learner is taught examples of a relation, then must answer prespecified, never-reinforced instances or a compositional downstream task using that relation, before feedback. Re-rendering the *same semantic cue/pair* under a new wrapper remains E2 | Transfer within the tested task family, if prior teaching causally improves performance beyond exposure-independent shortcuts |
| E4 — broader use | Frozen learner succeeds across substantially different, prospectively specified task families or natural streams | Only the breadth directly demonstrated |

The phrase **“persistent learning core” is a project goal**, not a result label earned by E1 or E2. Any promotion claim about reusable information requires an E3 result with delayed and intervening-learning probes, plus a causal state control. An E3 result is still limited to the tested relation family. E0–E2 experiments can be run to diagnose components and may justify further development; they must be reported at their own level.

Fresh random worlds, more seeds, delayed testing, different event order, noisy choice at the final margin, additional banks, or a no-write branch increase confidence in an E1 finding. They do **not** by themselves raise its evidence level. If every scored cue/pair was directly taught its answer, an exact-key table can still explain the outcome. A changed wrapper is E2 only if the scored byte presentation was not itself taught; changed raw bytes alone do not prove a changed internal code. A held-out event is E3 only when its answer was not supplied during the learner's history and prior teaching improves the result beyond exact-trained-key lookup and exposure-independent shortcuts. A compact rule learned from examples is valid E3 evidence, even if that rule is simple.

## Required design and report block

Before allocating a large run, every MiniFly design must state, in plain language:

1. **Claim and level:** the narrowest E0–E4 claim the primary endpoint can support. Distinguish a mechanistic diagnostic from a promotion test.
2. **Exposure audit:** which exact byte forms, cue identities, combinations, labels, and task rules the learner saw with feedback; which scored forms/relations it did not; and whether the first scored choice occurs before feedback or reteaching.
3. **Recitation countermodel:** the performance an exact-key lookup table, a canonicalized-key table, and exposure-independent part/frequency shortcuts could achieve. If one could pass the primary gate, the result stays E1/E2 or the gate is redesigned. A simple feature rule *learned from the prior episode* is legitimate E3 evidence if the causal control supports it.
4. **State, attribution, and causal controls:** what state persists across phases and which actual learner branch produces the scored answer; a positive control known to solve the task; and equal information/feedback budgets for the compared learners. For an E3 claim, compare against a branch with matched sensory input and timing but without the relevant **prior teaching episode claimed to be reused**, while both branches receive no feedback on the current scored instance. A no-write branch only during later interference cannot establish reuse of earlier learning. A successful engineered predictor cannot be counted as a success of the native value-memory branch without an integrated test.
5. **Promotion rule and cost:** prespecified outcome, failure and stopping criteria, resource accounting, and a cheap preflight before a costly confirmation run. The report repeats the level actually earned, including when lower than planned.

This block is required even when reusing an old assay. Reviewers should reject a mismatch between the stated core objective and a primary endpoint that an exact-pair table can pass. They should **not** discard a valid diagnostic just because it is E1; they should prevent its interpretation from silently expanding.

## Current interpretation and next decision

The V9.4–V9.7 “old-memory loss” endpoint is loss of choice or margin on **previously rewarded ordered-byte pairs** after new pairs are taught. “New learning” there is acquisition of other explicitly rewarded pairs. V9.7's HASH8 gain in absolute taught-pair choice and its larger write-associated old-choice penalty are real, separately scoped E1 findings. Neither establishes E3 transfer. The exploratory matched-precorrect-pair reanalysis does not replace V9.7's preregistered comparison.

The next bounded program should put **unseen-input behavioral gates** ahead of another bank/topology search. First, on an engineering world, measure whether prespecified label-preserving byte variants change the KC code or HASH8 address without changing any learning state. Then run a small frozen-architecture E2 test: teach multiple presentations that establish a declared nuisance field, withhold one presentation from all reinforcement, and score that first response before feedback. Keep exact trained-cue performance as an E1 diagnostic. Independently predeclare a small E3 test of never-reinforced combinations or downstream uses of an already taught relation, including the shortcut and prior-teaching controls above; do not let the E2 score stand in for it. Carry delayed and intervening-learning probes into any gate that survives its earliest pre-feedback check. E2 failure localizes an input/address issue but does not rule out E3 on a fixed presentation; E3 failure localizes a relation-extraction/use issue only if its own input and positive controls pass. E2 success alone is not E3 promotion.

Only after this gate identifies the failure should a component lesion, new memory mechanism, topology change, or evolution search be chosen. This ordering is a default research discipline, not a ban on small, clearly labeled E1 diagnostics. It avoids paying for another large confirmation run that cannot answer the project's central question.
