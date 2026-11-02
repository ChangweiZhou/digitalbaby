# MiniFly recursive research-loop skill — cross-branch integration

**Date:** 2026-11-02  
**Origin skill:** `claude/vibrant-cerf-sje726@66925b04b92768e0f80272b5d44dfd69d5aeb206`  
**Science base snapshot:** `codex/minifly-consolidated-20261023@51b1f6b36f769c8a204125895970b023c08a6f6e`  
**Disposition:** selective skill/KB integration and updated research instructions; no experimental files or model pointers modified.

## What was merged

- Claude's **five actual skills** and eleven original reference documents, preserved byte-for-byte as input before updating the five entry-point `SKILL.md` files.
- A named copy of each **original Claude SKILL.md** in its own `references/ORIGINAL_CLAUDE_SKILL_20261102.md` so detailed MRK-1 plans/claims are not erased by the update.
- Claude's original historical `PROJECT_DOCUMENTS/KB` tree, preserved unchanged except a new integration-status note (no fabricated owner decision/activation).
- New actionable skills, a [current evidence snapshot](../../.claude/skills/minifly-research-loop/references/CURRENT_EVIDENCE_20261102.md), a [readiness/missing-checker audit](../../.claude/skills/minifly-research-loop/references/SKILL_READINESS_20261102.md), and a cross-agent root [AGENTS.md](../../AGENTS.md).

## Why the original checker is not activated

The skill's old `MFC` interface is fully described in Claude's manuals, but the committed source lacks the **actual minifly_check.py dispatcher and several worker modules/templates**. We therefore do **not** import a partial checker, do not issue fake PASS/REVIEW artifacts, and do not use its proposed tier/constraints as user-approved facts. Full MRK-1 recovery is a distinct coding/testing/review effort.

**This is a functional research-decision skill suite, not a formally runnable official certification system.**

## Lessons incorporated since the skill was written

- V164/V178: known dense BPTT references do **not** establish that a proposed sparse/gated graph will learn; stop false serial prerequisites.
- V167/V169/V172: distinguish learned local features, self-trained output, native value writer, transient prediction and long-lived useful knowledge; a replay benefit without actual forgetting does not demonstrate consolidation.
- V175/V176/V177: interference-code overlap and fitted-probe gains are diagnostics, not operative native success; when a positive reference fails, local candidate is NOT_RUN.
- V179: same-checkpoint training/held divergence, the contrast between freeze-during-B and harm-after-correction, corrected KC identity vs last16 identity, and impossible exactness of arbitrary literary text under a too-short generation cap.
- V180: the **4.23M-parameter / 4.87GB checkpoint** engineering admission failure (technical PASS, science NOT_RUN) is a scale-design failure, not evidence for or against CGLP.
- V181: the **37,313-parameter** compact design and its complete physical resource contract are a design-only proposal, with no source-parity or deployed-memory proof yet.
- Program governance: preserve real statuses, version/branch provenance, complete lives and sample/CPU accounting; no automatic new version, owner-authorization fabrication or promotion from a proxy.

## Independent review pending

This integration was authored as one agent's change. No other family independently validated the changed skill wording, the checker restoration (not performed), or the application of governance rules. Treat the update as **review-pending** while it remains useful as a documented decision aid. Historical KB remains unchanged and explicit mini/operational constraints are not relaxed.

For source blob hashes, imported subtree hashes and resulting file manifest see [INTEGRATION_MANIFEST.json](INTEGRATION_MANIFEST.json).

## What was intentionally NOT merged

- Claude's whole experimental/model branch or its 16,000+-file checker test corpus;
- incomplete checker code published as if executable;
- status promotion, core adoption, user-decision fabrication, E3 confirmation;
- new model code, training data, learned weights, current Codex V181 work, or the live model version registry.
