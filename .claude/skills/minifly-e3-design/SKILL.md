---
name: minifly-e3-design
description: "Use for MiniFly experiment designs or reviews claiming E1 complete answers, E2 persistent useful knowledge, E3 never-taught addition/reuse, transfer or reasoning. Applies to held sets, exact byte/EOS, evidence strata, determinacy, no-prior-teaching and shortcut countermodels, true retention/correction, and avoiding misleading natural-prose exactness."
---

# MiniFly E1–E3 design and claim boundaries — updated 2026-11-02

**Read:** [current research-loop entry](../minifly-research-loop/SKILL.md), [evidence snapshot](../minifly-research-loop/references/CURRENT_EVIDENCE_20261102.md), [original Claude E3 skill](references/ORIGINAL_CLAUDE_SKILL_20261102.md), [rule grammar](references/RULE_GRAMMAR.md), [witness and strata](references/STRATA_WITNESS_UDI.md), [admission/runner design](references/ADMISSION_AND_RUNNER_CONTRACT.md). The Claude protocol has precise declarative S0–S4, witness, shortcut and objective-score concepts; its absent `MFC` implementation **cannot presently certify an experiment**.

## 1. Endpoint is a trained learner's own action

The project requires **small, locally trained, persistent** useful learning, not a trained external evaluator. For deterministic addition:
- Held relations/orbits are fixed before training and not simply equivalent permutations of taught answers.
- Taught units and the witness for held determinacy are made explicit. Exclude trivial S0 and presentation-twin S1 when evaluating E3; in the historical protocol only determined S2–S4 can count.
- Outputs are complete **own bytes and own EOS**, not argmax from a fitted probe, answer-position teacher-forced probabilities, a forced-EOS repair, an answer lookup, a BPTT-only/reference model or digit/carry features coded by the experimenter.
- Compare against no-prior-teaching (same birth, no taught target), chance, unigram/counting, exact-prefix/canonicalization, truncation/constant, replayed-label and other strong shortcuts. The comparator must itself be able to produce a nonzero outcome on the locked test; zero-scoring DIRECT comparators are invalid.
- Retain and quantify previously acquired still-valid OLD answers during NEW learning, but allow legitimately revised obsolete answers to change. OLD-before-teaching acquisition and OLD-after-NEW retention both matter.

A planned component E0/E1 study need not run the full official E3 battery. But **label it** as component or development, and never promote its metric into E3.

## 2. Distinguish distinct forms of learning

The V179 natural-text experiment used teacher-forced full 257-class next-byte/EOS prediction. It found bounded transfer and corrected feedback shifts, while held loss worsened with more repeated B training. That is *predictive learning*, not complete own-response E1/E2/E3.

The V179 synthetic edit mapped frequent two-byte patterns to arbitrary replacement characters, often replacing spaces; its large corrected-target contrast demonstrates changed conditional byte statistics, not correction of semantic facts. For genuine “revision” claims, define new factual feedback, conflicting prior knowledge, retained unrelated facts and an independent answer cue, none of which supply a role/digit parser to the candidate.

For a time-decayed FE-PERM encoder, unchanged last16 bytes **does not imply** identical sensory state. On corrected streams, partition:
- `A_OLD`: original held stream; old distribution.
- `A'_SAME_KC`: same expected byte and **bit-identical entire KC code** under changed environment; strict still-valid retention.
- `A'_ALTERED_KC`: same expected byte but altered sensory state; corrected-environment invariance test.
- `A'_CHANGED`: truly changed supervised target; only new byte counts, compared with same-state NOEDIT feedback.

A future E2 task must specify which knowledge remains relevant, not penalize the model for replacing an obsolete target.

## 3. EOS and whole-output evaluation

Train on genuine next-byte sequences with exactly one EOS per real record; never fabricate EOS at every sampled byte. Distinguish observed-prefix next-byte probabilities from autonomous self-fed generation.

**Do not score arbitrary unseen literary prose by requiring exact reproduction from a three-byte prefix** as a binary ability gate: many continuations are valid and V179's 32-byte generation cap made matching many lines impossible by construction. Report EOS NLL, Brier/calibration and premature-stop rate at real boundaries; length/truncation for free generations; exact natural-text match only in length-feasible disclosed subsets and descriptively. For *deterministic addition/answer tasks*, exact whole-byte+EOS is meaningful and remains mandatory for behavioral claims.

Unmasked 257-class predictive quality, capacity of a fitted ridge/BP readout, and a qualitative sample continuation are three different evidentiary objects.

## 4. Testbed, scope and real compute

Before training, verify a source-trained oracle can *in principle* solve the registered task, the floor/shortcut controls fail where expected, and held answers are determined by taught information. A failed positive reference on an arbitrary architecture (V173/V177) says *that reference regimen did not qualify*, not that local learning was tested. An ungated sparse intermediate failing V178 cannot veto a gated remedy.

Only then run model-native learning. Admit its **entire** footprint: learned weights including 257-way head, fixed sensory/feedback/wiring, all mutable plastic states, optimizer moments, clones, retained checkpoints, data and CPU. V180 resource admission failed at 4.23M parameters/16 lives; V181's 37,313-param contract is **unverified design**, not a validated model. Keep the biological-locality claim separate from whether the training implementation uses minibatches or adaptive optimizer state.

Register independent data worlds and evaluate on a fresh sealed guard *only after* algorithm and thresholds freeze. Previously opened development worlds can still produce useful negative/diagnostic information, never independent confirmation.

## 5. Causal claim clarity

A strong E3 design states: (i) exact taught units, (ii) genuinely unrewarded but determined held relations, (iii) a complete native own response, (iv) still-valid OLD retention, (v) all relevant shortcut comparisons, (vi) distinct exposure and repeated compute costs, (vii) the exact expected result that would invalidate the claimed mechanism. Do not replace a failed conditional claim with a new benchmark in the same version.

The original Claude witness, UDI/leave-units-out and sealed-world G remain valuable candidate *methods*, but the checker required to certify their execution is missing. Use `NOT_VERIFIED` honestly until the executable checker and independent review exist.
