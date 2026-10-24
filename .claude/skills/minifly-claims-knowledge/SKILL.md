---
name: minifly-claims-knowledge
description: "Use whenever you write or change anything another agent or Mr. Zhou will read as a MiniFly result, status or lesson: REPORT.md, CLAIM.json, STATUS or SUMMARY files, PROJECT_DOCUMENTS/CURRENT_*.json pointers, catalog or memo rows, or a next-design memo that quotes earlier numbers; when proposing, activating, narrowing or citing a finding in PROJECT_DOCUMENTS/KB/FINDINGS.jsonl; when scoring registered predictions; when filing or closing an incident; when correcting a published verdict; and when exploring what is already known before proposing a design. Enforces the generated headline and closed claim vocabulary, confirm wording only on sealed worlds, pointers as copies of CLAIM.json, findings proposed by the author but activated only by the other agent, errata for changed verdicts, the incident-to-check rule and what never counts as progress. Commands: minifly_check.py claim, gate --stage publish|pointer, ledger append, kb status|score."
---

> **If you only read this.** Never type a result: `MFC claim <dir>` generates CLAIM.json and the headline, and REPORT.md starts with that headline.
> Confirm wording only for worlds_role SEALED_CONFIRM. Soft words (largely, nearly, promising ...) block in the headline and in any sentence naming a gate or the verdict.
> Pointers are byte copies of CLAIM fields, written only at T2 after snapshotting the previous pointer. Quote a narrowed finding only with its narrowing row.
> Propose 0-3 findings (with a reason when 0); only the other agent family activates them. Nothing in the KB is ever edited or deleted.
> Only the sealed-world G series is progress. PASS counts, gates added, cycles and flags are not.

`MFC` means `python3 -I tools/minifly_check/minifly_check.py`, run from the repository root. Exit codes: 0 PASS,
1 FAIL, 2 MISSING_OR_INVALID_INPUT, 3 CHECKER_ERROR. Every nonzero exit blocks; there is no `--force` and no `--skip`.

- Normative spec: `tools/minifly_check/SPEC.md` (sections 7.4, 8, 9.9 and 11). Field-level schemas:
  `tools/minifly_check/SCHEMAS.md` (sections 15-18); this skill never restates them. Examples:
  `tools/minifly_check/templates/CLAIM.example.json`, `templates/POINTER.example.json`.
- Ledgers live in `PROJECT_DOCUMENTS/KB/` (read its `README.md` for the seed rows). They are append-only (I8).
- References: [references/CLAIM_AND_POINTERS.md](references/CLAIM_AND_POINTERS.md) (headline grammar with worked
  headlines, enum tables, banned words, the pointer procedure, legacy pointers) and
  [references/KNOWLEDGE_LEDGER.md](references/KNOWLEDGE_LEDGER.md) (row kinds, fold rules, R-levels, activation,
  prediction scoring formulas, transfer rows, incident closure, errata, the one-shot backtest).
- Sibling skills: `minifly-research-loop` (stages and commands), `minifly-e3-design` (what the endpoint measures),
  `minifly-independent-review` (the T2 review), `minifly-governance` (changing the kit, owner decisions).

## 1. Explore with the KB

- Start from `MFC kb status`. It prints the E3 cadence and deliveries_since_last_E3, open incidents, narrowed findings
  with their caveats, open review debts and the sealed G series.
- Read a finding through its fold, never through its first row: the current status is the most restrictive status among
  the finding and every row that refers to it (KNOWLEDGE_LEDGER.md section 2).
- Cite an F-id (or a catalog id) for every number about prior work, in REPORT, memos and predictions. CLM-16 warns on
  uncited numbers; LOCK-11 fails a prediction whose basis_findings are not in FINDINGS (or 'none').
- Quote a narrowed fact only with its caveat. CATALOG P036 (`PROJECT_DOCUMENTS/FULL151_VERSION_CENSUS_20261023/CATALOG.md:42`)
  still reports STOP-ONLY's conditional stopping gain as +22.10pp [20.47,23.72]. Its finding F-UF-15 is narrowed by
  F-20261024-claude-1: the arm's own stream truncated at one byte (FIXED_1) scores 10.645% against STOP's 11.263% on
  NEW, and beats STOP on OLD (0.977% vs 0.326%). Write the number with that caveat and the narrowing row id, or not at all.
- Find the nearest tested delivery by task family and lineage: DELIVERY_LEDGER rows (task_family, family_id) and that
  delivery's CLAIM.json. The 53 seed rows are legacy census rows (`primary_gate` 'legacy', no task_family): read their
  `legacy_level_text` and REPORT, and treat them as legacy summaries, not v1 claims.
- Base rates are Laplace estimates from DELIVERY_LEDGER (KNOWLEDGE_LEDGER.md section 5); `MFC init` prefills
  base_rate_forecast. Until v1 rows exist per task family there is no informative base rate; say so.
- A fact marked UNVERIFIABLE_IN_REPO (seed F-UF-02, the Full151 V9.9A/B delayed-E3 numbers) cannot be the sole basis
  of a design. Seed rows F-UF-01..29 carry the digest's statuses and start at R0, pending reconciliation by the other
  family: F-UF-16 carries the caveat 'uninformative manipulation' and F-UF-25 is dev only. A seed row is not an
  activated finding.

## 2. Generate the claim

```
MFC score <dir> --role official      # program decide writes results/official/VERDICT.json (official runs only)
MFC claim <dir>                      # writes CLAIM.json and the headline; appends a DELIVERY_LEDGER row (CLM-13)
```

- Run `claim` whatever the outcome. A delivery that never ran official still gets a CLAIM (primary_gate NOT_OPENED,
  with its dev shadow verdict), and an honest exit gets a headline exactly like a PASS.
- Edit only findings_proposed, findings_none_reason and notes. Every other field must equal a fresh regeneration,
  headline included (CLM-02). If an input changed, rerun `claim`; never patch a field. After editing those fields,
  rerun `MFC claim <dir>` too: it keeps your edits and appends the DELIVERY_LEDGER row for the new CLAIM sha (CLM-13).
- The first non-empty line of REPORT.md equals CLAIM.headline (CLM-05). Build it from the file, not by retyping:

```
{ python3 -I -c 'import json,sys; print(json.load(open(sys.argv[1]))["headline"])' <dir>/CLAIM.json; echo; cat "$BODY"; } > <dir>/REPORT.md
```

  Here `$BODY` is your draft prose, kept outside the delivery directory.

- The prose below the headline is yours, but it is linted at publish: confirm wording (CLM-04), soft words (CLM-06),
  native/core/inherited wording (CLM-07), absolute local paths (CLM-12) and uncited prior numbers (CLM-16, WARN).
- Never hand-write a measurement, a verdict or a STATUS boolean. Numbers in the prose are copied from CLAIM, SCORES
  or VERDICT, or computed by a committed script whose path you cite.

## 3. Vocabulary and level rules (CLM-01..09)

| Check | Rule | What you do |
|---|---|---|
| CLM-01 | CLAIM schema and closed enums (tables in CLAIM_AND_POINTERS.md section 2) | Nothing: `claim` writes it. Never add free-text levels |
| CLM-02 | Every field except the agent-editable ones equals a fresh regeneration | Rerun `claim`; never edit generated fields |
| CLM-03 | E2+ needs PASS with gate_type objective. E3 needs pool HELD, counted strata within S2..S4, a no_prior_teaching comparator, ADM-EXPO PASS and headroom over the best shortcut of at least margin_abs. E3 promotion also needs UDI LCB > 0 and SEALED_CONFIRM. More than one declared factor change adds the label integrated_intervention | State the level earned, never the level hoped for (I5) |
| CLM-04 | Any confirm* token in REPORT, STATUS, SUMMARY or the headline requires SEALED_CONFIRM with SEAL-* and WLD-01 PASS | Off sealed worlds, write the measured result and the worlds_role |
| CLM-05 | First non-empty line of REPORT.md equals CLAIM.headline | Use the command in section 2 |
| CLM-06 | Soft words FAIL in the headline sentence and in any sentence containing a gate id or the primary_gate token; WARN elsewhere | Replace the soft word with the number and the gate's result |
| CLM-07 | native, core, inherited or 'same fly' need lineage_parent with inheritance demonstrated (WARN at T1, FAIL at T2) | Name the executable learner and its parent, or drop the word |
| CLM-08 | Every file added after the official lock is listed as POSTHOC and is never a VERDICT input | Post-hoc analysis is welcome; it is labelled, not hidden |
| CLM-09 | worlds_role is computed: fewer than min_worlds gives PILOT; dev-touched worlds give DEV; MAIN needs the official stage and fresh worlds; SEALED_CONFIRM needs SEAL, OC, REVIEW and replay | Never describe worlds as more than their computed role |

- Soft words (CLM-06): largely, mostly, in part, partial, partially, nearly, promising, essentially, approved, met in
  part, substantially, broadly, close to, encouraging. MiniFly-CLS wrote "G1 memory — largely met (84–88% at 24
  passes, still rising; target ≥ 95%)" (`minifly_cls_dev_20261017/REPORT.md:93`). Under v1 that sentence names gate
  G1, so it FAILs; write "G1 memory: 84-88% at 24 passes against the locked target of 95%" and let the gate result speak.
- Confirm wording (CLM-04): LEARNED-BANK headed a section run on one development world "Native confirmation"
  (origin/charming-head `minifly_learned_bank_v1_20261024/cycles/CYCLE2.md:61`). That fails CLM-04 (dev worlds) and
  CLM-07 ('native' without demonstrated inheritance). Read confirm* as any token containing "confirm" in any case,
  review statuses such as CONFIRMED included: refer to a review finding by its id instead. Identifiers count too
  (GAIN_CONFIRMED, axis_confirmed or NativeConfirmation in a SUMMARY.json key or value), and zero-width characters,
  accents and look-alike letters are stripped first; only the whole kit names SEALED_CONFIRM, sealed_confirm,
  CONFIRM_LEDGER, confirm_attempt and confirm_alpha_spending are exempt (corpus S56-S58).
- 'Persistent learning core' is a project goal, not a result label (`minifly_b/package/DOCS/MINIFLY_OBJECTIVE_CONTRACT.md:19`).
- Word lists catch listed words only. Synonyms in body prose are an open loophole (spec 15); the generated headline is
  the real defence, so do not reach for 'well on the way' either.

## 4. Publish gate and the T2 verification subset

```
MFC gate --stage publish <dir>       # S7 exit criterion
MFC lint <files...>                  # T0 text (memos, saved-data reanalysis): CLM-06, CLM-12, CLM-16, PTR-01..07
```

- The publish bundle runs SELF-01, SCHEMA-01, AMD-01..05, MET-01/02, LAD-01, STR-01, DET-01, INERT-01, DEC-01,
  CLM-01..16, ERR-01, KB-01..05 and WLD-01/02, plus RUN-01 and GOV-COI at T2, REV-08 for SEALED_CONFIRM or adoption,
  and DEBT-02 (UDI-01 is planned in checker 1.0.0: not yet enforced). At T0, stamp published outputs DEV_NOT_EVIDENCE.
- CLM-10: every lock prediction has a predictions_scored entry (section 7). CLM-11: 0-3 findings_proposed, each with
  conditions.best_shortcut and a source path:line; zero needs findings_none_reason. CLM-12: no absolute local paths
  (`/Users/`, `/home/`, `/tmp/`, `C:`) and every relative link resolves. CLM-14: each cited GATE report's
  checker_sha256 equals the committed checker at that report's commit.
- CLM-15 (T2): commit the verification subset with the claim: official OUTPUTS for the counted pools (gz or zip
  allowed), TAUGHT manifests, receipts, both locks and the GATE reports. Above CONSTRAINTS.verification_subset_max_mb
  (50 MB, draft) the spec's mode digest_plus_sample applies (per-(world, arm) digests plus a full-row sample whose seed
  the reviewer chooses), but checker 1.0.0 does not implement it: CLM-15 FAILs. Stop and ask the owner
  (REQUIRES_USER_DECISION); never shrink the subset to get under the cap.
- The repo's publications so far excluded raw data on the owner's instruction ("Push the results, not the raw data.",
  `minifly_v6_r1_dev_20261023/PUBLICATION.md:3`; also `native_next_v1_dev_20261023/PUBLICATION.md:3`). A T2 claim
  needs the subset anyway. If the owner's grant excludes it, the conflict is REQUIRES_USER_DECISION: ask the owner, quoting the size,
  before publishing; the size cap is the owner's decision (spec 19, item 12).

## 5. Pointers

Full procedure: CLAIM_AND_POINTERS.md section 4. Checked by `MFC gate --stage pointer --pointer <path> <dir>` (always
T2): SELF-01, PTR-01..07, CLM-02 and DEBT-02.

- Write a pointer only as a minifly.pointer.v1 copy of CLAIM fields (PTR-02: level_earned, primary_gate, gate_type,
  worlds_role and objective_metric byte-equal to the CLAIM named by claim_sha256). Never retype a number.
- Snapshot the previous pointer into `PROJECT_DOCUMENTS/<TRACK>_POINTER_HISTORY/` in the same commit (PTR-01).
  AUTHORIZATION stage pointer_update must be true, or an adoption UD must exist. T1 cannot update pointers.
- Cite the narrowing row id for every quoted finding whose latest status is narrowed or superseded (PTR-04).
- No bare boolean keys matching confirm|pass|passes|passed|success|qualified at any depth (PTR-03), and no absolute
  paths (PTR-05). `PROJECT_DOCUMENTS/CURRENT_EXTERNAL_MEMORY_EXPERIMENT_DESIGN.json:5` carries
  `"confirmation_passes": true` and 22 `/Users/` paths; a v1 pointer like that fails twice.
- PTR-06 (WARN): a newer v1 delivery exists in the same track. Check before you point at an older one.
- A pointer to a legacy delivery needs an owner UD and a KB/RETRO annotation (PTR-07). LEARNED-BANK is legacy: before
  its verdict is cited, it needs `retro`, a Codex executing review recorded in KB/RETRO, and provisional findings (spec 17).

## 6. Findings lifecycle

Detail: KNOWLEDGE_LEDGER.md sections 1-4. Append rows only with
`MFC ledger append --ledger FINDINGS --row ROW.json`; never edit `FINDINGS.jsonl` by hand.

- As the delivering author, propose 0-3 findings at publish: in CLAIM.findings_proposed and as FINDINGS rows with
  status proposed. Zero is fine with a one-sentence findings_none_reason. Each finding carries its full conditions
  (best_shortcut included) and a source `path:line@sha`.
- Only the other agent family activates a finding (KB-03), and only when its source is a T2 delivery with worlds_role
  MAIN or SEALED_CONFIRM and no open review debt (KB-04, DEBT-02). Never activate your own finding: C30 in the corpus
  is exactly that.
- Anyone may append a narrowing, supersession or retraction row. Narrowing is conservative and needs no second agent.
- R-levels R0-R4 are computed by the checker from the rows; never declare one. Only R3 and above may be called
  'transferable'.
- Nothing is deleted. A wrong row is corrected by a new row that refers to it.

## 7. Prediction scoring

- Predictions are registered in ENDPOINT_LOCK for the primary and guards only (LOCK-11); none on admission measurements.
- At publish, `claim` scores each one into predictions_scored (CLM-10): hit (the observation falls inside interval90),
  the Winkler interval score and the Brier score of p_pass. An endpoint that was never opened is scored 'unresolved'.
  Formulas and a worked example: KNOWLEDGE_LEDGER.md section 5.
- Never delete or rewrite a registered prediction. The V6 revision deleted its forecasts instead of scoring them
  ("Delete ungrounded sample-count and window-accuracy forecasts", `minifly_v6_revised_design_20261023/REVIEW.md:11`);
  corpus fixture C26 is built from it and fires CLM-10.
- Append one transfer row {prediction_id, delivery, ok|failed} to each finding a scored prediction cited
  (KNOWLEDGE_LEDGER.md section 6). A finding with 2 or more failed transfers and negative skill is narrowed by an
  appended row.
- Scores are diagnostic only. `MFC kb score` reports hit rate and Brier skill against the Laplace base rate per task
  family, or INSUFFICIENT_N / DEGENERATE_BASE. Calibration is never an agent score or a kit score, so nobody gains by
  predicting pessimistically.

## 8. Incidents

- File every defect you find, old or new, with `MFC ledger append --ledger INCIDENTS --row ROW.json`: fm_class,
  symptom, evidence `path:line@sha`, caught_at_discovery (did the committed checker catch it) and
  checker_sha_at_discovery.
- An incident closes only with a closure row naming a check_id present in RULES.json and a corpus fixture whose
  EXPECTED.must_fire includes that check (KB-05). The check, fixture and EXPECTED entry go through `minifly-governance`.
- Never close an incident with prose, a skill sentence or a memo. 'Lessons stay prose' is FM-26.
- The seed closure rows (INC-SEED-xx-CLn) are pending their corpus fixtures and do not count until KB-05 passes.

## 9. Errata

- Any change to a published primary_gate needs an ERRATA.jsonl row in the delivery, classified BUG_FIX or POST_HOC
  (ERR-01). Commit it together with the regenerated CLAIM; git history keeps the old one.
- POST_HOC can never upgrade a verdict. A BUG_FIX upgrade needs a corpus fixture id and a reviewer sign-off from the
  other family. Treat any change whose new value is PASS as an upgrade, and treat any change you are unsure about as one.
- Never edit VERDICT.json, CLAIM.json or an adopted pointer to make an old result look different. Never rerun official
  to replace a verdict; a follow-up is a new protocol on fresh worlds.

## 10. What never counts as progress

- PASS counts, gates added, cycles completed, flags avoided or fixed, findings recorded and calibration are not
  progress (I11).
- Program progress is the G series on SEALED_CONFIRM worlds only (spec 9.6). MAIN-world G is reported separately as
  'unsealed'. A memo that argues progress from anything else is FM-09 or FM-22.

## Open limits (spec 15)

- Body-prose synonyms of the soft words are not caught (CLM-06 is WARN outside gate sentences).
- Output fabrication is closed by recomputation plus seeded replay for SEALED_CONFIRM or adoption only; it stays open
  for MAIN results.
- Rote predictions remain possible: scoring removes the incentive to game calibration, not the effort to predict well.
- USER_DECISIONS rows are agent-transcribed; without an owner-held anchor a fabricated adoption UD is detectable only
  at the owner's census spot-check.
- CLM-16, WLD-02 and PTR-06 are WARN-only. Any check that RULES.json marks `planned` is listed under 'Not yet
  enforced' in `tools/minifly_check/README.md`; it protects nothing until it is enforced.

## Not required

- No hand-written STATUS booleans, verdict prose or compliance disclaimers.
- No findings when there is nothing durable to record (zero, with a reason).
- No predictions on admission measurements.
- No restatement of CLAIM fields in prose beyond what the reader needs; the headline carries them.
- No recurring backtests, calibration dashboards or agent scores.

## Failure classes covered

FM-07 proxy metric; FM-22 claim inflation; FM-24 narrowings not propagating; FM-25 misremembered facts; FM-26 lessons
stay prose; FM-29 unverifiable publication; parts of FM-09 objective drift and FM-23 lineage and answer-path drift.
Definitions: `PROJECT_DOCUMENTS/KB/FAILURE_CLASSES.json`.
