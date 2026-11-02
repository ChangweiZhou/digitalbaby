# MiniFly skill suite — selective integration from Claude

Five active entry points, updated as of **2026-11-02**:

| Skill | Role |
|---|---|
| [minifly-research-loop](minifly-research-loop/SKILL.md) | Source → falsifier → design → technical admission → lifetime development → hostile review → decision; stops automatic version churn |
| [minifly-claims-knowledge](minifly-claims-knowledge/SKILL.md) | Claims, evidence statuses, cross-branch provenance and append-only knowledge |
| [minifly-governance](minifly-governance/SKILL.md) | Protected edits, owner decisions, cross-agent review, meta-loop anti-Goodhart |
| [minifly-independent-review](minifly-independent-review/SKILL.md) | Executed hostile audit of source, metrics, controls, validity and resource bounds |
| [minifly-e3-design](minifly-e3-design/SKILL.md) | E1/E2/E3 own full byte+EOS, causally valid held tests, no-prior-teaching and shortcuts |

Each updated skill preserves Claude's old `SKILL.md` as `references/ORIGINAL_CLAUDE_SKILL_20261102.md` and leaves original technical references in place.

**Source:** `claude/vibrant-cerf-sje726@66925b04b92768e0f80272b5d44dfd69d5aeb206`. The old five-skill suite and knowledge base are selectively copied rather than merging Claude's entire experimental branch into Codex's active model branch.

**Readiness warning:** The old checker documentation references `tools/minifly_check/minifly_check.py` and supporting modules that are not in the original committed checker runtime. The integrated skills are operational **decision guides**, **not an executable MRK-1 official certification system**. See [readiness](minifly-research-loop/references/SKILL_READINESS_20261102.md) and [dated evidence](minifly-research-loop/references/CURRENT_EVIDENCE_20261102.md). The historical KB at `PROJECT_DOCUMENTS/KB` remains provisional; integration does not authorize threshold changes, activate findings or invent owner decisions.

The new source-only update is **review-pending** until an independent cross-family audit. It does not run or adopt a model.
