# Change classes, direction map and the governance process

`MFC` means `python3 -I tools/minifly_check/minifly_check.py`. Normative text: `tools/minifly_check/SPEC.md` sections
13.2 and 7.4 (GOV-*). Protected paths: `tools/minifly_check/PROTECTED.json`. The class is computed by
`MFC governance --base B --head H`, never declared by the proposer.

## 1. Class table (spec 13.2; unclassifiable counts as loosen)

| Change | Class | Requirements |
|---|---|---|
| New check id, a new fixture, or advisory to enforced | tighten | GOV-REV, GOV-SELF, GOV-SEP, GOV-MONO |
| Check code edit where fired(head) is a superset of fired(base) on all fixtures and clean fixtures stay clean | tighten_or_neutral | GOV-REV, GOV-SELF, GOV-SEP, GOV-MONO |
| SKILL.md, AGENTS_POINTER.md (or a tracked AGENTS.md) or .gitattributes text edit with no rule line removed | neutral | GOV-REV, GOV-SKILL, GOV-SEP |
| Removed check, fixture, must_fire id or rule line (containing minifly_check, MUST, NEVER or must/never); enforced to advisory; stage removed; tier_min raised; non-monotone code edit | loosen | Everything above plus GOV-UD and GOV-NUM (the owner's verbatim words restate the numbers) |
| Any CONSTRAINTS.json change | constraint | GOV-UD (scope constraint:<key>). A loosening direction also needs GOV-NUM |

How `governance` decides, and what that means for you:
- RULES.json and EXPECTED.json are diffed by id. A removed must_fire id or check id is a loosen (corpus G02), and so is
  a raised tier_min or a stage dropped from an entry. A new id is a tighten (corpus C00f: a check plus a fixture plus a
  GOV-REV record exits 0).
- Check code is classified by behaviour, not by reading the diff. The base checker is extracted with git archive into
  a temporary directory, and both versions run on both corpora. A code edit is tighten_or_neutral only if every base
  fixture still fires at least what it fired, and every clean fixture stays clean (GOV-MONO). Otherwise it is a loosen.
- SKILL.md and pointer text are diffed by line. Any removed line that contains minifly_check, MUST, NEVER, must or never
  makes the change a loosen (corpus G05), even when the line was only reworded to be stricter. Add new rule lines and
  keep the old ones, or obtain a UD.
- Several changes in one range: the requirements add up, so one loosening change puts the whole range under GOV-UD
  and GOV-NUM. Split unrelated changes into separate ranges.
- Planned checks (spec 20.2): moving a `planned` check into a bundle is a tighten. Moving an enforced check to
  advisory or planned is a loosen.
- A loosening by definition breaks monotonicity, so GOV-MONO reports the non-monotone fixtures. The UD (GOV-UD with
  GOV-NUM) is what authorizes them. Read the `governance` report for how it records that.

## 2. Direction map (CONSTRAINTS.json)

The rule: higher is stricter for floors, except old_max_drop, alpha_max, the tier thresholds, e3_cadence_max,
review_debt_max_open and the default caps, where lower is stricter. A narrower shuf_band is stricter. Draft values
pending UD-0001:

| Key | Draft | Stricter when |
|---|---|---|
| threshold_floors.margin_abs | 0.05 | higher |
| threshold_floors.absolute_floor_above_chance | 0.05 | higher |
| threshold_floors.per_world_floor | -0.05 | higher |
| threshold_floors.min_worlds_nonpilot | 8 | higher |
| threshold_floors.min_items_per_stratum | 8 | higher |
| threshold_floors.input_contrast_min | 2.0 | higher |
| threshold_floors.prompt_id_margin_min | 0.10 | higher |
| threshold_floors.taught_acq_margin_min | 0.05 | higher |
| threshold_floors.min_realized_dose | 0.05 | higher |
| threshold_floors.old_acq_margin | 0.05 | higher |
| threshold_floors.analytic_input_snr_min | 0.5 | higher |
| threshold_floors.oc_min_power | 0.8 | higher |
| threshold_floors.shuf_band | [0.35, 0.65] | narrower |
| threshold_floors.old_max_drop | 0.10 | lower |
| threshold_floors.alpha_max | 0.05 | lower |
| tier_thresholds (t0_max_cpu_h 1, t1_max_cpu_h 4, t1_max_records 20000, family_t1_max_cpu_h 12) | as listed | lower |
| default_grant_caps (max_cpu_h 2, max_records 5000, max_cycles 3) | as listed | lower |
| e3_cadence_max | 2 | lower |
| review_debt_max_open | 3 | lower |

Fail-closed reading for everything else:
- The boolean default caps (official, push, adoption, pointer_update: all false) are stricter when false. Any change to
  true is a loosening.
- Keys outside the map have no computed direction, so treat any change to them as a loosening: native_forbidden,
  supplied_arithmetic_heuristics, evaluator_only_exemptions, io, warmup, objective_targets, required_arms_by_level, the
  testbed grids, e3_cadence_counts_legacy, confirm_alpha_spending and verification_subset_max_mb. The UD's verbatim
  words must state the old and new values.
- Every CONSTRAINTS change, in either direction, needs a UD with scope `constraint:<key>`. Only the owner sets
  constraints.

## 3. What a governance review record holds

The other family writes `PROJECT_DOCUMENTS/KB/GOVERNANCE/REVIEW_<id>.json` on its own branch, with its own
`Acting-Agent:` trailer (GOV-REV). SCHEMAS.md does not yet fix its fields. Until it does, record at least:
- the reviewer's family and session, and utc;
- base and head shas of the range, and the class `MFC governance` computed for each change;
- the commands you ran and their exit codes (`MFC selftest`, `MFC selftest --mutate` for code, `MFC governance`);
- for a tightening, the fixture that now fires and the clean fixture that stays clean;
- for a skill edit, the rule lines added or removed;
- your decision and its reason, including a rejection when a fix string-matches canary literals.

## 4. Step-by-step process

1. **Proposal.** Write the diff and a rationale that cites incident ids, false-positive reports with evidence (the
   GATE report and the artifact that was wrongly flagged), or a measured cost (runtime_s from GATE reports). If your
   family has a locked, unpublished delivery, a protected-path commit under your `Acting-Agent:` fails it at official
   and publish (GOV-COI): hand the change to the other family, or wait.
2. **Branch.** Use your own namespace (`origin/codex/*` or `origin/claude/*`). Touch protected paths only. KB/GOVERNANCE
   records and append-only ledger rows may ride along; no delivery directory may (GOV-SEP, corpus G01).
3. **Measure.** Run:

   ```
   MFC selftest
   MFC selftest --mutate                 # for any check code edit
   MFC governance --base <B> --head <H>  # B = the commit your branch starts from
   ```

4. **Obtain what the class needs.**
   - tighten, tighten_or_neutral, neutral: hand `<commit sha> <path>` to the other family, which writes the review
     record (section 3).
   - loosen: the review record plus an owner UD whose verbatim words restate every changed number, old and new
     (GOV-UD, GOV-NUM). Corpus G03: margin 0.05 to 0.03 with a verbatim lacking '0.03' fails.
   - constraint: an owner UD with scope `constraint:<key>`, plus GOV-NUM for a loosening direction.
   UD rows go in through `MFC ledger append --ledger USER_DECISIONS --row ROW.json`, in a governance range (alone or
   with the change they authorize), never with a delivery.
5. **Commit with trailers.** Add the trailers in the commit message, after a blank line, one per line:

   ```
   Tighten the realized-dose check for lesion arms (INC-...)

   Rationale: <incident id / false-positive evidence / measured cost>

   Acting-Agent: claude:<session>
   Governance-Review: PROJECT_DOCUMENTS/KB/GOVERNANCE/REVIEW_<id>.json
   User-Decision: UD-<yyyymmdd>-<n>
   ```

   `Governance-Review:` is needed for every tighten, tighten_or_neutral, neutral or loosen change; `User-Decision:`
   for every loosen or constraint change. A loosen needs both. The UD's scope must cover the change.
6. **Recheck.** Rerun `MFC governance --base <B> --head <H>` on the exact range that will merge. Proceed only on exit 0.
7. **Merge** the range as it is, with no delivery commits interleaved. Merge, never cherry-pick, so that the trailers
   and the reviewer-namespace commits keep their shas.
8. **Afterwards.** The change applies only to locks built afterwards (introduced_in). It is advisory for 2 v1
   deliveries at stages below T2 official, and T2 official, publish and pointer stages run it at once. For skill edits,
   re-run the gates on the last 5 v1 deliveries at the next census: no FAIL may flip to PASS without a stated bug fix.

## 5. Common mistakes

| Mistake | What fails |
|---|---|
| Fixing a threshold in RULES.json in the same range as delivery results | GOV-SEP, GOV-UD (corpus G01) |
| Removing an EXPECTED must_fire id 'because it was a false positive' without a UD | GOV-CLASS, GOV-UD, GOV-MONO (G02) |
| Owner says 'ok' to an agent-written number change | GOV-NUM (G03) |
| Editing a check after your own delivery's lock | GOV-COI at that delivery's official and publish stages (G04) |
| Rewording a NEVER line in a SKILL.md | GOV-CLASS, GOV-UD (G05) |
| A local, uncommitted checker edit | SELF-01: no gate report is produced, exit 3 |
