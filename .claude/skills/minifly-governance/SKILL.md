---
name: minifly-governance
description: "Use when updating MiniFly research skills, process rules, acceptance criteria, checker code, owner authorizations, protected evidence ledgers or model/experiment pointers. Requires prospective changes, separate commits, explicit owner approval for broadened experiments or changed constraints, review of the skill itself, and honesty about the currently missing MRK-1 executable."
---

# MiniFly research governance — current entry point, 2026-11-02

**Read:** [minifly-research-loop](../minifly-research-loop/SKILL.md), [integration/readiness](../minifly-research-loop/references/SKILL_READINESS_20261102.md), and [Claude's original governance skill](references/ORIGINAL_CLAUDE_SKILL_20261102.md). The detailed old [CHANGE_CLASSES](references/CHANGE_CLASSES.md) and [HELDOUT_CANARIES](references/HELDOUT_CANARIES.md) procedures are preserved as reference designs, **not currently verified executable gates**.

## 1. Protect scientific evidence, not ritual

- A skill change can improve the research process **without** changing an experiment's committed design, official/sealed data, saved outputs, source code or original verdict. Publish a new skill commit, integration provenance and a dated evidence snapshot; never rewrite V175–V181 results.
- The priority is scientific capability: locally learned own output, retained useful learning, corrections and transfer per distinct experience and CPU, with an actual miniature model. Rule count, checker flags, number of agents/cycles and version inflation have zero promotion value.
- Each new requirement must answer **which past failure it prevents** and **how to recognize both false positives and false negatives**. The V178 serial sparse prerequisite was a **bad gate**: failing an ungated intermediate cannot veto testing a gated remedy. Tighten/loosen *ideas* only in a prospective, causal, source-linked way.
- An experimenter's source-local bookkeeping may use phase labels; the candidate learner must not see those labels as neural inputs, hidden gated update signals, task IDs, answer templates or control arms.

## 2. Protected edits / authorization

Treat `.claude/skills/**`, `AGENTS.md`, `PROJECT_DOCUMENTS/KB/**`, experimental acceptance contracts, checker implementations and adoption/current-pointer files as protected. Use a **separate skill-only branch or commit** from ongoing Codex experiments. Record original and new blob hashes, imported branch/commit and exceptions in a manifest. Preserve append-only historical KB facts; additions are **proposals** until authorized and independently reviewed.

Owner request to improve skills grants documentation changes; it **does not grant** scientific training, official confirmation, expanded record/CPU caps, changing protected acceptance thresholds, or adopting a model. Never invent owner words or append a USER_DECISIONS row attributed to a user without the exact provided grant. If a proposal changes old numeric constraints or loosens a scientific safety/validity gate, mark `REQUIRES_USER_DECISION` with old and new numbers and obtain explicit approval before application.

The imported Claude KB is historically provisional; absence of a user decision is **not evidence of approval**. Cross-family review of a skill's own changes is not achieved by the editing agent's self-audit; publish `REVIEW_PENDING` if no independent reviewer executed checks.

## 3. Recursion without self-approval

Use a falsifiable improvement cycle:

1. **Incident:** record the precise observed failure, source path and cost (e.g. V180 checkpoint admission; V179 impossible exact-line cap; V178 false serial prerequisite).
2. **Root cause:** identify which control/checker/skill instruction permitted the failure, distinguishing code defect from mistaken research question.
3. **Repair:** propose one general rule that would prevent **both that failure and close variants**, without vetoing valid experiments.
4. **Counterexamples:** assemble at least one should-fail fixture and one should-pass fixture; do not tune to canary string literals.
5. **Independent challenge:** a separate reviewer attempts a novel counterexample unseen when the fix was written.
6. **Disposition:** keep, revise prospectively or reject; report any inability to execute as `NOT_VERIFIED`. No auto-experiment or autopromotion.

**MRK-1 status:** Claude's old docs describe `python3 -I tools/minifly_check/minifly_check.py`, but the snapshot lacks that dispatcher and many required modules/templates. Do not invoke a missing command, hand-write its PASS artifacts or copy a 26MB corpus merely to make the package look complete. Its original checker spec is available in the [Claude branch](https://github.com/ChangweiZhou/digitalbaby/tree/claude/vibrant-cerf-sje726/tools/minifly_check). Restoration is a separate technical project with its own selftest, test fixtures and independent verification.

## 4. Current process corrections

- **Serial-gate drift:** no arbitrary topology, forced Full151 bridge or unrelated BPTT success as mandatory parent of another candidate.
- **Research proxy drift:** network probe gain/teacher-forced NLL is not own complete output, E2 retention or E3 never-taught addition.
- **Mini drift:** count real learned weights, fixed sensory arrays, optimizer states, all forked lives, checkpoint storage, read-only panels and overhead before training. V181's 37,313 scalars and ≤8MiB deployed sensor+network are a **design contract** pending measured qualification.
- **Evaluation drift:** no exact reproduction of an arbitrary held literary sentence under an impossible token cap; true EOS and length distributions separately. Stable last16 bytes ≠ stable complete FE-PERM/KC code. Synthetic conditional-byte edits ≠ semantic correction.
- **Overfit of the research process:** no new version just because a negative result was obtained; require a decision-relevant discriminating test and preserve fresh holdouts.
- **Compute drift:** one model and one complete real life are the unit of capability. Multiple cloned controls are costed individually; physical learner state and experiment output are both bounded.

## 5. Cross-agent handoff

For a change, provide the exact `source_branch@sha`, new proposed commit SHA, new/old file sha list, what is newly protected, what remains untested, and which results were intentionally left unchanged. Prefer a nonconflicting skill-only tree over modifying Codex's running experiment. A valid GitHub fast-forward with compare-and-swap prevents stale overwrites; a successfully written commit is not a cross-agent scientific review.

**Retain Claude's original protected-change details** in the archived reference. Do not state an MFC governance check passed until the actual runtime, its source closure and an independent reviewer are available.
