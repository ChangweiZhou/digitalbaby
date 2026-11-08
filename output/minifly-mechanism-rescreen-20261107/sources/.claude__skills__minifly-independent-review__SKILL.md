---
name: minifly-independent-review
description: "Use for hostile reviews of MiniFly results, source code, experimental claims, resource admission, evidence leakage, branch/provenance drift, artifacts and changes to the research skill. Covers cross-agent review obligations, executed source/gradient/data tests, cheapest falsifiers, witness and EOS/whole-response controls, mini-scale accounting and review-debt honesty."
---

# MiniFly independent hostile review — updated 2026-11-02

**Read:** [research-loop skill](../minifly-research-loop/SKILL.md), [current evidence](../minifly-research-loop/references/CURRENT_EVIDENCE_20261102.md), [original independent-review skill](references/ORIGINAL_CLAUDE_SKILL_20261102.md), [old review protocol](references/REVIEW_PROTOCOL.md) and [seal design](references/SEAL_PROTOCOL.md). The original MRK-1 runner/canary/sealing commands are **not currently executable from the integrated repository**. Do not fabricate a REVIEW, PASS, seal or reviewer signature.

## 1. Independence means a different accountable executor

A same-family subagent, the author auditing their own code, or an LLM merely restating the report is **not** independent cross-agent review. Another Codex or Claude family may execute checks under its own branch and commit identity. Verify literal code/outputs rather than accepting self-reported attestation or an invented `Acting-Agent` trailer. If the true reviewer is unavailable, record `REVIEW_PENDING` or `REVIEW_DEBT`—never a fake pass.

When reviewing a skill update itself, challenge the **new instruction's consequences**: whether it catches previous incidents and does not wrongly block a legitimate experiment. An integration that merely imports checker descriptions without the executable must be marked non-operational.

## 2. Minimum executed checks by failure category

**Source/provenance:** inspect active branch and exact sha, original V167/V169/Full151 code, source-lock closure, file authorship and experiment registry. Different branches may reuse version numbers; never merge provenance by number. Verify negative or NOT_RUN status at the actual report. Check version and pointer changes don't adopt a core without approval.

**Scientific decision:** identify one contested assertion, its comparator and expected opposing outcomes. If the next experiment simply diagnoses the previous failed arbitrary topology, ask whether the result has any direct implication for useful learning. A “first failed gate” is not causal attribution to the unique variable.

**Mathematics / learning:** finite-difference or independent formula test for source-local gradients and sign; verify fixed feedback vs learned output path, no hidden BP path, no labels before observation. For an AdamW-gated learner, inspect *applied* weight movement and moment clocks, not merely raw gradient magnitudes. For paired freeze/unfreeze, check clone equality and later plasticity tradeoffs, not only early held score.

**Control/data:** test no-prior-teaching, true equal exposure, zero-writing control liveness, sample/replay provenance, train/held and same-context identity. If an edit changes preceding bytes in a decaying FE-PERM trace, last16 unchanged does not imply full KC identity. A repaired class mask must not secretly change the candidate's logits. Distinguish synthetic conditional-frequency edits from semantic knowledge revision.

**Output:** prove byte/EOS full-class support at 257, real EOS positions rather than fabricated one-byte examples, own continuous generation vs teacher forcing. Require a length-feasible denominator for any exact text comparison. A capped 32-step literary generation test cannot disprove ability to predict a 60-byte unseen sentence.

**Mini/resource:** independently count **all** trainable weights (including output heads and biases), fixed sensory birth, moment tensors, ephemeral buffers, every arm/world/phase checkpoint, probes, filesystem output and process-tree CPU. Refuse any claim that masked dense learned matrices are actually sparse. V180 is the concrete regression: 4.23M scalars, 4.87 GB minimum checkpoint allocation with reserve, scientific run NOT_RUN. V181's compact memory bounds remain to be measured; a static design census is not a runtime pass.

**Outcome:** recompute the signed metric from raw or saved per-site distributions where available; verify same checkpoint train vs held; inspect whether the evaluated output is the candidate's own trained head. A probe advantage or BPTT reference is not local learning. A source-only data audit is not a new confirmation world.

## 3. Report form

Every review should contain:
- `REVIEW_SCOPE`: source@commit, version, exact claim, and assets actually inspected.
- `INDEPENDENCE`: executed cross-family / self-audit / static-only / pending.
- `CHECKS_EXECUTED`: exact commands/fixtures/hashes and observed outputs, or NONE.
- `FIRST_FALSE_ASSERTION`: specific contradicted claim, or NONE FOUND.
- `GAPS`: unavailable raw states, source, data, independence, provenance.
- `SEVERITY`: BLOCKER / MAJOR / MINOR / NOTE, each grounded in evidence.
- `CAUSAL_LIMITS`: which conclusions the test cannot support.
- `DISPOSITION`: REJECT / REVISE / ACCEPT_BOUNDED / INCOMPLETE.

Use `REVIEW_PENDING` instead of claiming the program checker approved an artifact when its CLI is absent. Leave the old original review/seal specification available for future restoration.

## 4. Cheap first, no endless loop

Before a long audit/experiment, run the cheapest contradictory fixture: source count, exact data hash, output rank, sign, target byte/EOS definition, BPTT-to-local path, train/held metric identity, or serialized checkpoint size. When the claim fails for a first-order reason, stop that branch. When it passes, proceed to the smallest causal experiment, not a generic list of new mechanisms.

The adversarial review itself improves recursively only when an incident becomes a tested check, with both a valid-case control and a new held canary. Counting reviews, audit pages or extra flags is not progress.
