# Human-learning mechanism audit — literature crosswalk (2026-10-24)

> **SUPERSEDED (2026-10-24):** This preliminary memo was written before locating Claude's original 150-entry human-learning catalog and before incorporating V143/V144 results. It is retained for provenance only. **Use [REVISED_AUDIT.md](REVISED_AUDIT.md) as the current, source-resolved assessment.** Do not treat the earlier literature crosswalk as a complete mechanism audit.

**Status:** Research synthesis committed for review; **not an implementation, experimental confirmation, or claim of newly missing mechanisms.** This document audits *human learning* as a source of mechanistic hypotheses, **not Drosophila cognition**, and does not change native MiniFly's experimental gates.

## Scope correction and provenance

The external Deep Research report titled *Audit of “Minifly” Learning Mechanisms (Human-Learning Perspective)* reviewed broad educational psychology principles without retrieving Claude's actual list. It consequently labeled 13 instructional approaches "unspecified (likely missing)." That inference is **invalid**: the repository's [254-row historical mechanism inventory](../FULL151_VERSION_CENSUS_20261023/MECHANISMS_254.md), [full experiment review](../FULL151_VERSION_CENSUS_20261023/MEMO.md), and Claude's fast-memory / CLS work explicitly cover many neighboring mechanisms. The research is useful as a **literature crosswalk and hypothesis generator**, not as a completed line-by-line absence audit.

The critical experimental target is **unseen-addition full autonomous byte/EOS generation with old-task retention and actual CPU/sample efficiency**, using native permitted byte/feedback streams. The latest V6-R1 result fails acquisition and EOS and diagnoses weak cue conditioning relative to noise. Introducing an unbounded replay bank, arithmetic/operand priors, teacher-forced expected-byte updates under scalar-only feedback, or external quiz scheduling is **not** a valid substitute.

## Mechanism versus teaching intervention

| Learning-science topic | Putative internal computational mechanism | Relationship to documented MiniFly work | Next action |
|---|---|---|---|
| Retrieval practice / testing effect | Memory reactivation and retrieval-dependent update; reconstructive readout | Retrieval, replay and readout variants already indexed (e.g. V23/retrieve_hard*, retrieve_soft*, replay_reservoir*) | Test only if architecture can generate endogenous retrieval cues and measure *causal* retention; don't add a quiz UI |
| Spaced repetition | Time-dependent consolidation, decay and reconsolidation under reactivation | V23 fast_slow*, feedback_consolidation and replay; V24 budgeting | Treat spacing schedule as an **environmental intervention**, not a new synapse rule; compare equal exposure and CPU |
| Interleaving | Context discrimination / interference management / task selection | Context routing, stable/split-state and retention experiments | Interleave only allowed task records; isolate selection strategy from implicit task-ID leakage |
| Corrective feedback and error-based learning | Credit assignment, reward-prediction/error-conditioned plasticity | Historical ERROR and Claude fastmem correction/reset explicitly noted in census | Only if acquisition survives native cue-conditioning repair; forbid unavailable target-byte gradients |
| Worked examples / scaffolding | Imitation, curriculum shaping, fading of external supervision | Educational training design, not direct neuronal mechanism; current warmup and feedback protocols | Optional input-policy experiment with total teaching bits and teacher knowledge charged |
| Metacognition / self-regulation | Monitoring uncertainty and controlling learning rates / retrieval / allocation | Importance, uncertainty, surprise and protection variants (e.g. V23/bayes_*, pair_importance) | Specify measurable internal monitoring signal before claiming human metacognition |
| Motivation / reinforcement | Neuromodulatory gain, reward-based plasticity, exploration-exploitation | Native feedback gating and related mechanisms | Distinguish experimental reward gating from human intrinsic motivation; no badges/gamification in core |
| Social learning / observation | Learning from observed action/state trajectories, model imitation | No verified gap from this report; requires a defined observational signal | Defer: would alter information available to native learner |
| Multimodal/dual coding | Complementary representations and cross-modal binding | Not a priority for byte-only interface; generic representational diversity is different | Do not introduce modalities or hand-coded arithmetic features |
| Transfer / analogical reasoning | Binding roles, compositional abstraction, out-of-distribution generalization | Central unresolved scientific objective; engineered additive/ordinal priors previously helped bounded tasks but violated stronger native interpretation | Evaluate withheld relations and controls; do not infer transfer from taught recall |
| Cognitive-load management | Working-memory capacity, gating, interference, resource allocation | Capacity, sparse access, context routing and budgeting thoroughly studied | Specify bounded persistent-state/CPU costs and direct manipulations |

The original research's 13 items overlap considerably; metacognition/self-regulation, error-based learning/feedback and scaffolding/worked examples should **not** be counted as independent novel mechanisms without distinct update rules and ablations.

## Human-learning mechanisms deserving explicit gap assessment (not yet verified absent)

The educational review missed several mechanism-level hypotheses. These are **candidates for targeted code-level comparison**, not blanket endorsements:

1. **Structured compositional binding and variable-role separation:** same relation across exchanged operands, representation changes and previously unseen pairings; prohibit arithmetic hard-coding. Potential falsifier: matched taught success with chance-level unseen outputs.
2. **Successor state / predictive world-model learning:** predict future latent states conditioned on sensory sequence, as distinct from feedback-conditioned response memorization. Must not expose hidden answers.
3. **Complementary learning systems with offline consolidation and replay:** fast episodic acquisition plus slow regularity extraction, capacity-bounded and paid in events/CPU. Existing Claude CLS is an explicit *partial test*, not a missing feature; 13% best unseen sums in a prior setup was not success.
4. **Synaptic tagging, eligibility and delayed credit:** eligibility traces and delayed modulatory plasticity; existing score estimators/e-prop fixtures do not demonstrate native generalization.
5. **Prediction-error driven attention / surprise-weighted encoding:** learn input-dependent selectivity from observable mismatch instead of noise-dominated addressing; compare to non-learning cue-scaling control.
6. **Metaplasticity and stability–plasticity homeostasis:** learned plasticity thresholds, local uncertainty calibration, interference-gated protection; many related historical variants must be checked for exact overlap.
7. **Schema formation and memory integration:** infer shared latent structure across episodes, contrasted against raw/canonical lookup and frequency/length countermodels.
8. **Active hypothesis testing / information-gain selection:** distinct from replay or curriculum; only meaningful if the native agent can choose actions/questions and all additional observations are accounted for. Not enabled by current passive task alone.

For each candidate, a future implementation proposal needs: exact existing-version crosswalk, non-duplicate mathematical update rule, legal information provenance, compute/state budget, matched ablation, preregistered primary endpoint, and failure criteria. Presence/absence cannot be asserted without the implementation audit.

## Recommended sequence, consistent with current project evidence

**P0 — restore reliable sensory conditioning first.** Reproduce the bounded V6 cue-scaling intervention in [NEXT_DESIGN](../../minifly_v6_r1_dev_20261023/NEXT_DESIGN.md), matched to its unchanged baseline and OLD/NEW/held controls; require own byte/EOS behavior, not only improved KC separation.

**P1 — test error-sensitive acquisition only after P0.** Compare a single legal feedback-conditioned native write/update with fixed budget against matched control. Claude fast-memory reset results support *taught* acquisition only, not transfer.

**P2 — measure interference and retention.** If acquisition is real, selectively compare fast/slow protection or bounded replay, rather than stacking mechanisms.

**P3 — test genuine abstract transfer.** For binding/schema/predictive candidates, require held-out relation accuracy above unchanged countermodels and demonstrate output autonomy and counterfactual dependence on training; report uncertainty and total cost.

No mechanism should be described as "implemented" merely because it is recommended here. No new arithmetic experiment was run for this review.

## Literature starting points

- Dunlosky et al. (2013), *Improving Students' Learning With Effective Learning Techniques* (distinguishes practice testing and distributed practice from weaker or conditional interventions): https://www.whz.de/fileadmin/lehre/hochschuldidaktik/docs/dunloskiimprovingstudentlearning.pdf
- Roediger & Butler (2011), retrieval practice; summary: https://psychology.ucsd.edu/undergraduate-program/undergraduate-resources/academic-writing-resources/effective-studying/retrieval-practice.html
- Stanford Teaching Commons, formative assessment and feedback: https://teachingcommons.stanford.edu/teaching-guides/foundations-course-design/feedback-and-assessment/formative-assessment-and-feedback
- Education Endowment Foundation, metacognition and self-regulation: https://educationendowmentfoundation.org.uk/education-evidence/teaching-learning-toolkit/metacognition-and-self-regulation

**Evidence warning:** educational interventions are supported principally in human teaching settings, not as validated architectural improvements to MiniFly. The deep-research report's categorical impact ratings and universal "immediate feedback" prescription do not transfer to the model without controlled tests.
