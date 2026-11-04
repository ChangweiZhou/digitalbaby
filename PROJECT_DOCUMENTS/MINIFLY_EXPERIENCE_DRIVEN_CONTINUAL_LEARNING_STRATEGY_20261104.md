# MiniFly research memo: Experience-driven continual learning and the Bitter Lesson

**Date:** 2026-11-04  
**Status:** Research strategy proposal — not an adopted design, experiment authorization, or claim of established capability  
**Scope:** MiniFly / DigitalBaby, local learning, persistent relevant knowledge, architectural and mechanistic development

## Executive position

MiniFly faces a training-paradigm problem rather than a mere shortage of plasticity-rule variants. Recent successful mechanisms—sparse associative memory, retained-context binding, successor-style temporal representations, and consequence-conditioned action learning—have been demonstrated on bounded tasks. But repeatedly adding task-specific answer structures and control paths risks building an elaborate specialist system rather than a learner that improves with experience.

Sutton's *The Bitter Lesson* is a warning against substituting accumulating human domain knowledge for generally scalable search and learning. It is **not** a theorem that all inductive biases are harmful, that simply increasing byte-prediction compute creates persistent learning, or that local learning without backpropagation is guaranteed to scale. The scientific decision is to keep **general-purpose, reusable mechanisms** while reducing **task-specific rules inside the learner**.

There are meaningful prior results in continual learning, complementary learning systems, replay, reinforcement learning, and representation learning. There is not an accepted turnkey recipe for open-ended general continual learning that MiniFly can simply reproduce.

## What the recent MiniFly experiments actually warrant

- **V193/V196:** Relational readout can be reused after structured observation-only change, but source filtering was externally supplied in the successful mixed-source arm. The representation and task geometry remain highly constrained.
- **V197:** Multiscale associative addresses and importance protection help factual recall in bounded synthetic streams. Longer lifetimes and larger loads remain difficult; do not equate old-memory dilution with failure unless the information remains relevant.
- **V201 byte integration:** Multiple mechanisms can run under one explicit protocol. Protocol dispatch is not autonomously learned task interpretation.
- **V203–V205:** QA1 and QA2 can score highly, but lexical triggers, tokenization, answer-vocabulary masking, supplied candidate operations, and output-control scaffolding are substantial inductive biases. New-word and new-template probes show different generalization limits.
- **V206:** Causally retaining earlier context fixes an information-loss failure under supplied write timing. This is a reusable architectural direction; the model does not yet learn which earlier context deserves retention.
- **V207:** More prose pretraining improves its own held-author next-byte predictions, yet the QA development path never used the predictor fallback. Training a module harder does not help another function without a genuine learned information path.
- **V208:** Learner-chosen actions and correctly paired sampled consequences give bounded reversal learning. It is a one-pending-action assay, with exploration and observable context tightly defined.
- **V209/V209B:** Headline contextual answer-selection scores are confounded by retrieval order and ties; randomly assigned location labels can preserve apparent success. It does not establish learned query-dependent semantic binding.

These are **different candidate mechanisms and task families**, not independent demonstrations that one unchanged learner has acquired open-ended intelligence.

## Core research hypothesis

A viable persistent learner should use its accumulated experience to **acquire useful structure faster in genuinely new environments**, while updating obsolete beliefs, preserving currently relevant facts, and retaining the ability to learn.

The historical research decomposition

`memorization -> representation -> generalization`

should become a *coupled online cycle*, not a strict developmental law:

`experience -> predictive state/representation -> selective associative memory -> inference/action -> new experience`

Representation shapes which experiences can be recalled; memory supplies observations for representation learning; action generates evidence; evidence can revise both memory and action values.

The primary test of this hypothesis is **forward transfer**, not a perfect score on a familiar task or indiscriminate preservation of every old fact.

## Proposed training paradigm: MiniWorld, not another bespoke QA patch

Build a **lightweight, continuously running, procedurally generated environment** that exposes raw causal observations, learner-chosen actions, genuine consequences, and legitimate local teaching feedback. Its dynamics may include people and objects changing locations, possession, delayed effects, partial observations, revisable facts, and environment changes. These rules belong in the **environment**, not as hidden answer shortcuts, hand-coded semantic paths, or simulator-state access inside MiniFly.

Keep the agent architecture fixed over families of worlds. Start with small worlds that are exactly auditable, then progressively increase entity count, compositional event complexity, observation noise, feedback latency, distribution shifts, and concurrency. The learner should not receive latent entity IDs, canonical state tables, correct action labels, future bytes, or privileged phase boundaries unless these are explicitly the *legitimate observed interface* for every competing method.

A teaching cue may indicate that an observation should be learned. It must not secretly convey its desired output, entity identity, action credit assignment, or future relevance.

Three concurrent learning signals are worth evaluating under a single lifetime:

1. **Predictive/structural learning:** predict next observations, state transitions, or future relational consequences from a retained causal state.
2. **Associative learning and revision:** write observed useful associations locally; replace invalidated ones; allocate capacity based on demonstrable future usefulness rather than total historical recall.
3. **Action–consequence learning:** choose actions, receive sampled effects, and assign feedback to the causally responsible earlier state/action, including delayed and eventually multiple pending actions.

These need not use one weight matrix. Multiple memory timescales, sparse support, successor-like state maps, and local three-factor updates are legitimate candidate mechanisms. No backpropagation or attention is required for the MiniFly research track; the restriction should be treated as a scientific constraint to test, not an excuse to avoid competent baselines.

## Experimental progression: hold the learner fixed while experience changes

**Phase A — One persistent life:** Establish an unreset stream combining facts, revisions, structural observations, and action–consequence events. Separate parameters, episodic state, and externally supplied cues. Show that prediction, readout, action, and update all operate on the same chronological experience rather than independent task-specific replays. Measure compute and storage per experience.

**Phase B — Experience diversity rather than first increasing model size:** Add independently generated entities, unfamiliar word forms, reordered event constructions, delayed outcomes, two pending actions, and new combinations of previously experienced relations. Do not silently change the learner for each new task. If a structural limitation is diagnosed, test one causal mechanism with a matched ablation and then freeze it for the next cross-world comparison.

**Phase C — Scaling only after positive transfer:** Vary experience volume, writable support, representation capacity, and replay/consolidation rules. Compare curves, not only endpoints. A bigger stream is justified if gains are reproduced on new worlds with a **fixed architecture**, not merely on the original world generator.

This is a prospective research strategy, not a request to launch Phase A–C immediately. No experiment schedule, code alteration, or new GPU allocation is authorized by this note.

## The primary scientific yardstick

Take two identical initial architectures with matched storage, learning-update and compute budgets. Let **A** complete a pre-exposure lifetime of varied worlds and let **B** be a fresh control. Compare them on *new held-out worlds* where specific keys, objects, answers and surface templates do not overlap with the pre-exposure set.

If `L_A(n)` and `L_B(n)` are their losses after `n` new-world learning opportunities, evaluate:

`Transfer(N) = sum_{n=1}^{N} [L_B(n) - L_A(n)]`.

A positive, replicated transfer area indicates that prior experience makes *new learning* easier. Include confidence intervals across independent world-generating seeds. A frozen test-time comparison alone is insufficient; both groups must have the same opportunities to adapt.

Also report:

- **Relevant-retention accuracy:** facts still true and useful; separate stale-but-superseded associations from errors.
- **Correction quality:** time to replace obsolete knowledge and collateral damage to unrelated valid facts.
- **Plasticity:** improvement on a newly introduced task late in the lifetime, relative to an identically resourced fresh model.
- **Generalization:** unseen roles, rewordings, entity combinations, and environment dynamics; not only held queries from the same templates.
- **Control and causality:** correct action/outcome assignment, online reward or safety *during* learning, not just final greedy preference.
- **Efficiency:** bytes/episodes, learner updates, wall/CPU time, memory capacity, replay expense, and active address traffic.
- **Strong matched baselines:** canonical state-table trackers where appropriate, recency rules, n-gram or Markov predictors, fixed-router variants, no-write controls, random-feedback controls that preserve answer type, and retrieval-ranking-only alternatives.

Never treat the largest number of stored historical facts as the objective: forgetting irrelevant or superseded facts can be correct. Likewise, absolute stability is not success if plasticity collapses.

## Distinguish reusable inductive bias from task leakage

An architectural component deserves expansion when it *increases what can be learned or retained across multiple task families* without encoding a specific answer: e.g., a causal long-term context, multiple temporal scales, writable support expansion, locally learnable routing, or bounded eligibility for delayed outcomes.

Treat proposals skeptically when success requires adding a fresh parser rule for QA1, a hard-wired person/object/location operation for QA2, a parity-specific transition stride, an evaluator-provided source selector, or tie-breaking that happens to exploit candidate presentation order. Such mechanisms can remain positive controls, but should not silently become evidence of autonomous learning.

Before claiming a mechanism, verify the data path from its input through its state update into the scored output; test information removal, frozen readouts, alternative candidate orders, true-versus-scrambled feedback, and simpler source-matched algorithms. Always report both the result and the narrower claim it supports.

## Bitter Lesson guardrail for future iterations

The default response to a failed new task should **not** be to add a task-specific module. Instead ask:

1. Was relevant observable information available to the learner at the decision boundary?
2. Was the information retained in the representation/address, or was it lost before learning/readout?
3. Is the local learning rule capable of discriminating the needed alternatives given those features?
4. Does the failure come from representation, memory, credit assignment, routing, or output control?
5. Can a single proposed mechanism improve **multiple different held-out worlds**, compared with a matched simple baseline, without privileged task knowledge?

If a new mechanism passes such a test, retain it as a candidate generally reusable primitive. If not, record the negative result and resist compensating for it with more benchmark-specific logic.

## Near-term priority and non-goals

**Priority:** Design a *single continuous-life MiniWorld protocol and prospective forward-transfer assay* that reuses the strongest existing local-learning components without adding QA-specific answer pipelines. Keep a concrete read-only ledger of which observations and labels each module actually receives. Validate an end-to-end minimal fixture, then freeze architecture before fresh-world scoring.

**Secondary:** Mechanistically test causally retained context and multiple-pending-action credit assignment—both have a clear relationship to widely reusable learning capacity.

**Defer:** Larger prose pretraining claimed to improve QA without a used interface; extra bAbI-specific wrappers and answer rules; version-number-driven claims of progress; optimizing all-old-fact retention instead of current relevance.

A necessary failure mode to track is **loss of plasticity**: protective weights, increasingly frozen routing and strongly consolidated memories may preserve current accuracy while suppressing later learning. Test new-task acquisition at multiple points during one life.

## Interpretation and references

This is a hypothesis and an experimental strategy, **not** evidence that MiniFly has achieved open-ended lifelong learning. Sutton's *The Bitter Lesson* (2019) is a methodological argument, not a proof that scaling raw text solves the problem. Experience-based agent research provides complementary motivation; continual-learning work identifies both catastrophic forgetting and loss of plasticity as distinct risks.

Selected starting points:
- Richard Sutton (2019), *The Bitter Lesson*: https://bitterlesson.ai/
- David Silver & Richard Sutton (2025), *Welcome to the Era of Experience*: https://theaiinnovator.com/welcome-to-the-era-of-experience/
- McClelland, McNaughton & O'Reilly (1995), *Why There Are Complementary Learning Systems in the Hippocampus and Neocortex*: https://pubmed.ncbi.nlm.nih.gov/7624455/
- Dayan (1993), *The Successor Representation*: https://doi.org/10.1162/neco.1993.5.4.613
- Dohare et al. (2024), *Loss of Plasticity in Deep Continual Learning*: https://www.nature.com/articles/s41586-024-07711-7

**Decision requested in a later turn:** Whether to prioritize an experience-driven, fixed-architecture learning-life experiment over further task-specific QA improvements. This memo does not itself record that decision.
