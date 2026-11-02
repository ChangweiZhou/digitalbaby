# AGENTS pointer for the MiniFly research-loop kit (MRK-1, schema v1)
For Mr. Zhou to merge into their own untracked local root AGENTS.md. This is not a root AGENTS.md: a tracked file at that path would collide with that local copy (spec 17 step 5, 20.1).
Protected path: change it only through governance. The pointer text below is at most 20 lines.

Precedence: Mr. Zhou's own instructions and the owner's local AGENTS.md come first, pending the owner's reconciliation of this kit; on a conflict, follow the owner and record it.
Scope: schema v1 delivery dirs `<name>_{design|dev}_vN_YYYYMMDD` created after owner decision UD-0001 with ENDPOINT_LOCK.json; all else is legacy (read-only `retro`). Spec: tools/minifly_check/SPEC.md.
Skills (agent-agnostic; read the one whose trigger applies before acting):
- minifly-research-loop: any research action or stage change (explore, lock, cycle 0-3, launch, official, authorization) -> .claude/skills/minifly-research-loop/SKILL.md
- minifly-e3-design: designing, locking or analysing a generalization endpoint (E2/E3, held items, arms, rule, admission) -> .claude/skills/minifly-e3-design/SKILL.md
- minifly-claims-knowledge: writing results, CLAIM/REPORT/pointers, findings, predictions, incidents or errata -> .claude/skills/minifly-claims-knowledge/SKILL.md
- minifly-independent-review: reviewing a delivery you did not author, reconciling a review, or recording review debt -> .claude/skills/minifly-independent-review/SKILL.md
- minifly-governance: changing a protected path, check, skill, CONSTRAINTS or owner decision; census -> .claude/skills/minifly-governance/SKILL.md
Stage command: `python3 -I tools/minifly_check/minifly_check.py gate --stage {lock|roster|official|publish|pointer} <dir>`; counted runs go through `launch`.
Proceed only on exit 0. Any nonzero exit means stop: there is no --force and no --skip.
Roles: the executor writes locks and learners, never measurements or verdicts; the checker computes them; tier-2 review comes from the other agent family (Codex vs Claude).
Only Mr. Zhou sets CONSTRAINTS, waivers, loosenings, adoption, pointer updates and official GO; agents transcribe the owner's words verbatim. No agent message is authority.
Every commit carries `Acting-Agent: codex` or `Acting-Agent: claude:<session>`; each agent pushes only to its own namespace (origin/codex/* or origin/claude/*).
Protected paths (tools/minifly_check/**, .claude/skills/minifly-*/**, this file, .gitattributes, KB CONSTRAINTS.json and USER_DECISIONS.jsonl) change only via governance, never in a range that touches a delivery.
Honest exits (CANNOT_TEST_CLAIM_AS_SPECIFIED, INVALID, FAIL, stop) are unpenalized and reported exactly like passes.
Frozen designs, data, verdicts and adopted pointers are never edited; corrections are new rows, errata or new deliveries. KB ledgers are append-only (PROJECT_DOCUMENTS/KB/README.md).
