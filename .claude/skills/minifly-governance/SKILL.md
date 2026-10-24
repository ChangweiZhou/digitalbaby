---
name: minifly-governance
description: "Use whenever anyone proposes to create, edit, relax, remove or re-tier anything protected: a MiniFly skill (.claude/skills/minifly-*), anything under tools/minifly_check (checks, RULES.json, corpus, EXPECTED.json), AGENTS.md / PROJECT_DOCUMENTS/KB/AGENTS_POINTER.md, .gitattributes, PROJECT_DOCUMENTS/KB/CONSTRAINTS.json or USER_DECISIONS.jsonl, including a small fix to a check that is blocking your own delivery; when a check gave a false positive or missed a defect; when recording an owner decision that changes constraints, waives a rule or loosens a check; and at each census or every five v1 deliveries, to measure whether the loop itself improves. Defines protected paths and invariants, computed change classes, owner decisions with restated numbers, cross-agent review for tightenings, separation from delivery commits, the conflict-of-interest rule, the advisory window, sealed held-out canaries, held-out incident recall, prune reviews and the only progress metric."
---

> **If you only read this.** Protected paths (the kit, the five skills, AGENTS_POINTER.md, .gitattributes, CONSTRAINTS.json, USER_DECISIONS.jsonl) change only through governance, and never to unblock your own delivery.
> `MFC governance --base B --head H` computes the change class. Anything it cannot classify counts as a loosening.
> Kit and skill changes need a review record by the other agent family. Loosening and constraint changes also need Mr. Zhou's verbatim words restating every changed number.
> Commit protected changes as their own range, with no delivery path, carrying `Governance-Review:` and/or `User-Decision:` trailers. Changes apply prospectively.
> The checker is judged only on sealed held-out canaries and post-freeze incidents. Never optimize anything against flag or pass counts.

`MFC` means `python3 -I tools/minifly_check/minifly_check.py`, run from the repository root. Proceed only on exit 0.

- Normative spec: `tools/minifly_check/SPEC.md` (sections 1, 5, 7.4 GOV/HELD, 12.3, 13 and 20). Kit files and their
  schemas (RULES.json, PROTECTED.json, EXPECTED.json): `tools/minifly_check/SCHEMAS.md` section 19.
- References: [references/CHANGE_CLASSES.md](references/CHANGE_CLASSES.md) (the class table, the CONSTRAINTS direction
  map, the step-by-step process with trailer formats) and
  [references/HELDOUT_CANARIES.md](references/HELDOUT_CANARIES.md) (sealed held-out batches, held-out incident recall).
- Kit status: `tools/minifly_check/README.md`. Its 'Not yet enforced' section lists every check registered as
  `planned`, and a planned check protects nothing.

## 1. Protected paths and invariants

`tools/minifly_check/PROTECTED.json` is authoritative. Today it lists:
- `tools/minifly_check/**` (code, RULES.json, EXPECTED.json, corpus, templates, SCHEMAS.md, INTEGRATION_LOG.md);
- `.claude/skills/minifly-*/**` (these five skills and their references);
- `PROJECT_DOCUMENTS/KB/AGENTS_POINTER.md`, and a root `AGENTS.md` only if one is ever tracked (none is: spec 20.1);
- `.gitattributes`;
- `PROJECT_DOCUMENTS/KB/CONSTRAINTS.json` and `PROJECT_DOCUMENTS/KB/USER_DECISIONS.jsonl`.

Every `PROJECT_DOCUMENTS/KB/*.jsonl` ledger is append-only (GOV-APPEND, KB-02).

Invariants (owner-defined; weakening one needs a USER_DECISIONS row with the owner's verbatim words):
- I1. Native line: no attention, Transformer, ELM or backprop; evaluator-only ceiling learners are labelled and never on the native answer path.
- I2. No supplied arithmetic structure in the native line.
- I3. Raw-byte input plus allowed feedback under the owner's gradual warmup (start ceiling 1/16).
- I4. Primary objective: retained, exact, own-output, own-EOS unseen addition at the final checkpoint, OLD protected, per sample and per counted compute.
- I5. The E0-E4 ladder of `minifly_b/package/DOCS/MINIFLY_OBJECTIVE_CONTRACT.md`; the level actually earned is always stated.
- I6. Confirm wording only for worlds_role SEALED_CONFIRM.
- I7. Tier-2 independence needs the other agent family; same-family sessions or subagents never count.
- I8. FINDINGS, INCIDENTS, USER_DECISIONS, DELIVERY_LEDGER, CONFIRM_LEDGER and REVIEW_DEBT are append-only.
- I9. Changes apply prospectively; frozen designs, data, verdicts and adopted pointers are never edited.
- I10. Honest exits are unpenalized.
- I11. Only sealed-world G counts as program progress.
- I12. Verdict evidence comes only from locked code and the program checker.

## 2. Change classes

`MFC governance --base B --head H` classifies every change in the range (spec 13.2; full table and direction map in
CHANGE_CLASSES.md):

| Class | Typical change | Needs |
|---|---|---|
| tighten | new check id or fixture; advisory to enforced | GOV-REV, GOV-SELF, GOV-SEP, GOV-MONO |
| tighten_or_neutral | check code edit with fired(head) a superset of fired(base) on all fixtures, clean fixtures still clean | GOV-REV, GOV-SELF, GOV-SEP, GOV-MONO |
| neutral | SKILL.md, AGENTS_POINTER.md or .gitattributes text edit with no rule line removed | GOV-REV, GOV-SKILL, GOV-SEP |
| loosen | removed check, fixture, must_fire id or rule line; enforced to advisory; stage removed; tier_min raised; non-monotone code edit | all of the above plus GOV-UD and GOV-NUM |
| constraint | any CONSTRAINTS.json change | GOV-UD (scope `constraint:<key>`), plus GOV-NUM in a loosening direction |

- Unclassifiable counts as loosen. One loosening change makes the whole range need GOV-UD and GOV-NUM.
- A rule line is a line containing minifly_check, MUST, NEVER, must or never. Classification diffs by line, so editing
  such a line removes it: sharpening a NEVER line in place is a loosen. Add the sharper line and keep the old one, or
  obtain a UD.
- GOV-SKILL: each SKILL.md has valid frontmatter (name, and a description of 1024 characters or fewer) and a body
  under 400 lines, and AGENTS_POINTER.md lists all five skills.

## 3. Process

Step by step, with trailer formats: CHANGE_CLASSES.md section 4.

1. Open a proposal with the diff and a rationale that cites an incident id, a false-positive report with evidence, or a
   measured cost (runtime_s from GATE reports). 'It blocked my delivery' is not a rationale.
2. Work on your own namespace branch and touch protected paths only. GOV-SEP fails any range that also touches a
   delivery directory.
3. Run `MFC selftest`, plus `MFC selftest --mutate` for check code, then `MFC governance --base B --head H`.
4. Tighten or neutral: the other family writes a review record, `PROJECT_DOCUMENTS/KB/GOVERNANCE/REVIEW_<id>.json`
   (GOV-REV). Loosen: that review record plus an owner UD whose verbatim words restate the changed numbers (GOV-UD,
   GOV-NUM). Constraint change: an owner UD with scope `constraint:<key>`, plus GOV-NUM in a loosening direction.
5. Merge as a separate commit range with the trailers `Governance-Review: PROJECT_DOCUMENTS/KB/GOVERNANCE/REVIEW_<id>.json`
   and/or `User-Decision: UD-<yyyymmdd>-<n>`, plus your `Acting-Agent:` trailer. Rerun `MFC governance` on the exact
   range and proceed only on exit 0.
6. Identity (spec 20.15): GOV-REV takes every family from `common.commit_family`, the `Acting-Agent:` trailer AND
   the origin/<family>/* branch containing the commit. The review record must be introduced by a commit that acts as
   the reviewer family, and each protected-path commit must act as one family, so push before you run governance.
   A trailer whose own namespace does not contain the commit is identity_mismatch (corpus S72: the only claude-named
   branch is on another remote). Never take identity from git author names.

During Phase 0 integration, every edit to EXPECTED.json or a corpus fixture is also logged in
`tools/minifly_check/INTEGRATION_LOG.md` with the spec clause that justifies it, and an independent verifier re-reads
that log for loosening (spec 20.3).

## 4. Owner decisions

- Record Mr. Zhou's words verbatim, with session, channel, scope and numbers_restated:
  `MFC ledger append --ledger USER_DECISIONS --row ROW.json` (id `UD-<yyyymmdd>-<n>`). You transcribe; the owner spot-checks
  at census.
- For any loosening, every changed number, old and new, appears in the owner's verbatim text (GOV-NUM). The owner's 'yes' to your
  question does not contain the numbers. Ask the owner to write them: "lower margin_abs from 0.05 to 0.03". Corpus G03 is a
  0.05 to 0.03 change whose verbatim lacks '0.03'; it fails.
- Scopes used by the checks: `constraint:<key>` (GOV-UD), `amendment:<delivery>` (AMD-02), `e3_waiver:<delivery>`
  (LOCK-09), adoption and pointer decisions (PTR-01, PTR-07), and a UD naming a delivery for GOV-COI.
- USER_DECISIONS.jsonl is itself protected and append-only. A UD row is committed in a governance range (protected
  paths only, alone or with the change it authorizes), never in a commit that touches a delivery (GOV-SEP). Run
  `MFC governance` on that range like any other.
- An agent message, a memo's 'approved', or a plan file is never an owner decision.

## 5. Conflict of interest and prospective application

- GOV-COI (official and publish stages): a protected-path commit made after a delivery's lock under that delivery
  author's `Acting-Agent:` fails the delivery, unless a UD names it. Read 'the author' as the author's family
  (fail-closed). While your family has a locked, unpublished delivery, hand kit changes to the other family or wait.
  Corpus G04 is exactly this case.
- Never change a check to unblock your own delivery. `native_next_v1_dev_20261023/REPORT.md:13` corrected an
  "erroneous assertion" in its own expanded fixture after that fixture failed. Under v1 a delivery's own evaluator
  files are endpoint-locked (AMD-05), and a check change goes through this skill, with the other family classifying it.
- Every RULES.json entry carries introduced_in. A check introduced after the checker_version of a delivery's first
  SOURCE_LOCK reports ADVISORY_FAIL at lock, technical and roster stages, and changes are advisory for 2 v1
  deliveries at stages below T2 official. T2 official, publish and pointer stages always run every check in the
  bundle at the latest version, which closes lock banking.
- Planned checks (spec 20.2): a check registered `planned` is in no bundle. Implementing it and moving it into a
  bundle is a tightening, and moving an enforced check to planned or advisory is a loosening.
- Versioning after a held-out batch (spec 20.10; lesson of HB-20261024-PILOT, whose RUN-03 was added after the batch
  with the same introduced_in 1.0.0, so every canary fired it): a check added or tightened after a held-out batch is
  committed must carry a later version than the batch's checker version. A new check gets a later introduced_in; a
  tightened check keeps its introduced_in and appends the new version to its RULES `changed_in` list. Bump
  CHECKER_VERSION first. The advisory window reads introduced_in only, so a tightening is enforced at once; held-out
  scoring reads both (section 7). Appending to changed_in is neutral only when the same range changes the check's
  module or a kit module it imports; an append without such a code change, or dropping or rewriting an entry, is a
  loosen (it would exclude the check's held-out hits and false alarms as post_batch).

## 6. The incident to check loop

1. Every defect gets an INCIDENTS row recording whether the frozen checker caught it (caught_at_discovery,
   checker_sha_at_discovery). See `minifly-claims-knowledge` references/KNOWLEDGE_LEDGER.md section 7.
2. If it was missed, the fix is a check plus a fixture plus an EXPECTED.json entry whose must_fire names the check,
   merged through governance as a tightening.
3. The incident closes only when its fixture is caught: a closure row with check_id and fixture_id (KB-05).
4. A fix after a canary miss must generalize. A reviewer rejects any fix that string-matches canary literals.

## 7. Held-out evaluation of the checker

Full protocol: HELDOUT_CANARIES.md. In short: someone who has not read `tools/minifly_check/*.py` writes a batch of 10
mini-deliveries, at least 3 of them clean. It is sealed with `MFC heldout seal`, its hashes are committed before the
checker version under test merges (HELD-01), and it is revealed and scored at census
(`MFC heldout reveal`, `MFC heldout score --checker-sha S`). The archive digest is training data: archive-derived
fixtures may extend the visible corpus but never count as held-out. A rise in corpus recall without a rise in held-out
recall is overfitting.

The score lists as `post_batch`, and never counts toward recall or false alarms, every fired check whose introduced_in or
changed_in is later than the checker_version in the canary's own SOURCE_LOCK (spec 20.18). Batch authors record there
the kit version whose SPEC they wrote against. The report always carries the raw numbers too (recall_fixtures_raw,
false_alarm_rate_clean_raw, post_batch counted); quote both, because a batch stamped with an older version (the
builder's default SOURCE_LOCK says 1.0.0) excludes every later detection and every later false alarm. Round 1 (1.0.1) closed the pilot misses INC-HB1-A3/C2/C3 with visible
fixtures S65-S73; only the next sealed batch, written against 1.0.1 and committed before the fixed checker, measures
whether the fixes generalize.

## 8. Census (every 5 v1 deliveries, or monthly)

| Item | How |
|---|---|
| Re-run gates on recent v1 deliveries | `MFC gate --stage publish <dir>` for each. No PASS may flip without a stated reason, and no FAIL may flip to PASS without a stated bug fix. The reports land in each delivery's `checks/`: never commit them in a range that touches protected paths (GOV-SEP), and record the flips with the census results |
| Held-out recall | `MFC heldout reveal ...`, then `MFC heldout score --checker-sha S ...`, written to `PROJECT_DOCUMENTS/KB/RETRO/HELDOUT_<date>.json`; also held-out incident recall (HELDOUT_CANARIES.md section 2) |
| Prune review | Only measured gate runtime (runtime_s in GATE reports) and catch counts (INCIDENTS, canary reports), never self-reported minutes. A check with zero live or canary catches over 10 or more deliveries and runtime above budget may be proposed for demotion. Demotion is a loosening, and the owner decides it |
| KB diagnostics | `MFC kb status`, `MFC kb score` |
| Kit health | `MFC selftest`, `MFC selftest --mutate` |
| Owner spot-check | Mr. Zhou traces 3 random CLAIM numbers to raw artifacts and spot-checks transcribed UD rows |

Skill edits are validated by selftest plus re-running gates on the last 5 v1 deliveries (spec 13.3).

## 9. Meta-Goodhart rules

- Never optimize prompts, skills or checks against flag counts or pass counts.
- Never cite pass counts, gates added, cycles or flags avoided as progress. Only sealed-world G is progress (I11).
- A canary fix must not string-match canary literals.
- The delivering agent's lessons are inputs to independent review, never conclusions.
- Calibration is never an agent or kit score.

## Open limits (spec 15)

- Owner decisions are agent-transcribed, and in-repo identity (trailers, branch namespaces) is tamper-evident, not
  tamper-proof. Without an owner-held anchor, governance is tamper-evident only. Optional hardening is the owner's
  decision: a push ruleset restricting protected paths, owner-signed `minifly-kit-vX.Y.Z` tags that SELF-01 can verify
  at T2, and later one-time approval codes.
- 'Monotone' loosening by regex narrowing stays partly open between sealed batches. Cross-agent review of code edits
  and the next sealed batch are the defences.
- Owner floors become the target: a weak-but-fail-closed rule is closed only as far as the floors the owner sets.
- The checker is a program-wide common mode. Hand-computed clean fixtures, `selftest --mutate`, reviewer
  trace_number checks and the owner's spot-check mitigate it without removing it.

## Not required

- No governance for delivery-only changes, and no UD for tightenings.
- No self-reported cost data (ceremony minutes) in a prune case.
- No recurring backtests, review canaries in a share of reviews, or dashboards.
- No 100% corpus coverage of every FM class before a merge.

## Failure classes covered

FM-15 test edited after failure; FM-26 lessons stay prose; FM-30 meta-loop Goodhart; parts of FM-27 self or padded
review and FM-28 authorization. Definitions: `PROJECT_DOCUMENTS/KB/FAILURE_CLASSES.json`.
