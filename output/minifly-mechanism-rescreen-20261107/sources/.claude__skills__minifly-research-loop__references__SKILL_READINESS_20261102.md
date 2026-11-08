# Research-loop skill readiness — 2026-11-02 integration

**Purpose:** Distinguish the usable research **instructions** from the incomplete historical MRK-1 **automatic checker**, so no agent mistakes a documented check for a successfully executed one.

## Source inspection and transfer scope

- Skill source: `claude/vibrant-cerf-sje726@66925b04b92768e0f80272b5d44dfd69d5aeb206`. Five Claude-authored skill entry points and their eleven reference documents under `.claude/skills`.
- Historical knowledge base source: `PROJECT_DOCUMENTS/KB` from that Claude branch. Copied as a **historical/provisional snapshot**, with original LEDGER, failure classes, old analysis and pending owner decisions unchanged. See `PROJECT_DOCUMENTS/KB/INTEGRATION_STATUS_20261102.md`.
- Historical checker source is available in the Claude branch as `tools/minifly_check`, but **not** imported as executable into the consolidated branch because the committed source tree lacks the dispatcher and several core modules.
- Consolidated science anchor before skill sync: `codex/minifly-consolidated-20261023@51b1f6b36f769c8a204125895970b023c08a6f6e` (newer Codex updates may exist, so inspect live HEAD). The skill sync does not merge the Claude model history, clone experiment work folders, or modify any active experiment.

## Executable-checker incompleteness

At Claude source tree `102abf4bed66989f1c3a0c0c36ca68fd27aff25f`, the non-corpus directory `tools/minifly_check` contains some substantive files (`common.py`, `admitcheck.py`, `addition.py`, `claimcheck.py`, `canary.py`, specs, README and rules). But the following paths **do not exist** in that committed directory even though the README describes them:

- `minifly_check.py` (argparse dispatcher);
- `launch.py`, `lockcheck.py`, `seal.py`, `scorer.py`, `decide.py`, `reducercheck.py`, `review.py`;
- `kb.py`, `governance.py`, `retro.py` and the referenced `templates/`.

The separate 26MB historical test corpus contains tiny fake `minifly_check.py` fixtures at corpus-specific paths. Those are *negative test samples*, **not** the real dispatcher, and cannot be used to make the runtime appear present.

**Therefore:** do not run the old `MFC` commands or claim generated GATE reports, sealed-world verification, successful selftest, generated CLAIM, code-independent canary or official release. Even a missing-module-free import of `common.py` is not a complete program. This limitation concerns **checker software**, not the usefulness of the Claude governance principles.

The Claude overview itself reported that its self-improvement build was **not yet independently verifier-accepted**, and that the proposed old owner decision UD-0001 remained pending. Importing documentation does not automatically supply the missing verification or owner approval.

## Current operational modes

| Mode | Allowed claims | Disallowed claims |
|---|---|---|
| `STATIC_SOURCE_INSPECTION_ONLY` (GitHub connector; no local experiment execution) | Exact file SHA/pointers; branch status; source/design flaws; resource arithmetic; dated evidence notes | Runtime tested, source-parity PASS, executed gradients, user consent, independent confirmation |
| `LOCAL_TECHNICAL` (local experiment runner present and explicitly authorized) | Real finite-difference, source parity, output/compute and checkpoint measurements with command/artifact hashes | Formal MRK-1 GATE PASS if the checker is missing; claims of E1/E3 from a technical fixture |
| `BOUNDED_DEV` (prospectively frozen and authorized scientist run) | Genuine operative model lifetime development outcomes, including negatives | Sealed confirmation on opened worlds, autonomous output inferred from fitted probes |
| `FORMAL_RELEASE` | **Unavailable under this imported package** until the full checker, schema, templates, tests, cross-family reviews and owner decisions are restored and independently qualified | Any "official" or "adopted" claim justified only by imported skill prose |

When given only GitHub connector access, source checkout, rerunning code, or verifying raw unpublished model checkpoints may not be possible. State those limitations and do not fabricate missing data.

## Restoration plan — separate technical project, not an automatic action

1. Recover or reimplement the missing original checker code and templates, with original hashes/provenance. **Do not** silently create a weaker “looks the same” script and preserve the old PASS terminology.
2. Create a complete standard-library test harness that actually executes the old schemas, checks, source-timeline tests, lock/launch behavior, metric recomputation, and failure fixtures. Reproduce baseline known validity AND known invalid cases.
3. Add a **new, blinded** corpus of should-fail/should-pass cases that includes the V178 bad-serial-gate case, V179 impossible-natural-text exactness/false stable-code case, V180 4.87 GB checkpoint failure, and V181 design-vs-runtime distinction. Do not tune by the exact old canary strings.
4. Require a **separate agent family** to run/critique the restoration and to evaluate any reduction/loosening of original checks. Preserve historical results.
5. Obtain actual owner approval for constraint thresholds/official release. Record verbatim words, not a guessed USER_DECISIONS entry.
6. Only after pass may `MFC` commands become operative gates in the current SKILL.md.

Until then, the refreshed skills provide **human/agent-guided, auditable decision checklists** and an evidence-oriented recursive research loop, not a certified automation engine.

## Integration validation

A successfully fetched blob and Git tree prove that the **documentation was committed**, not that the source code passes numerical checks. Check five new SKILL.md files, eleven source references, five archived originals, dated evidence, and current KB snapshot exist at the committed branch. Use git/connector SHA verification. Mark independent review of this skill update as `REVIEW_PENDING` until another agent actually challenges it.
