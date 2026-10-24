---
name: minifly-research-loop
description: "Use at the start of any MiniFly / digitalbaby research action by Codex or Claude and at every stage change: exploring what to try next, freezing a design or ENDPOINT_LOCK.json, running cycle 0 (testbed screen) or test cycles 1-3, launching any roster, pilot, official or confirmation run, recording or relying on Mr. Zhou's authorization, or wanting to change a task, metric, comparator, margin, world or check after seeing an outcome. Applies to every delivery dir <name>_{design|dev}_vN_YYYYMMDD created after the schema-v1 freeze. Gives the stage map of the loop (explore, design, cycle 0, technical, dev, rehearsal, official, audit and search), the computed stake tier, the one blocking command per stage (python3 -I tools/minifly_check/minifly_check.py init|gate|launch|score|cycle), typed cycles with exit criteria, default caps for short grants, what each FAIL allows, the honest-exit verdicts and the forbidden moves. Not needed for read-only analysis that changes no claim, pointer or plan."
---

> **If you only read this.** Find your stage in section 3, run its one blocking command, and proceed only on exit 0.
> Run cycle 0 (the testbed screen) before any learner code exists; rehearse on a fresh world; never rerun an official run.
> Produce every counted result through `launch`; never write ADMISSION, receipts, OUTPUTS, measurements or verdicts by hand.
> Act only inside Mr. Zhou's recorded AUTHORIZATION: the owner's verbatim words, and the default caps for any number the owner did not write.
> Honest exits (CANNOT_TEST_CLAIM_AS_SPECIFIED, INVALID, INCOMPLETE, FAIL, stop) are reported exactly like passes and cost nothing.

`MFC` means `python3 -I tools/minifly_check/minifly_check.py`, run from the repository root. Exit codes: 0 PASS,
1 FAIL, 2 MISSING_OR_INVALID_INPUT, 3 CHECKER_ERROR. Every nonzero exit blocks. There is no `--force` and no `--skip`.

- Normative spec: `tools/minifly_check/SPEC.md`. Field-level schemas: `tools/minifly_check/SCHEMAS.md` (this skill
  never restates them). Templates: `tools/minifly_check/templates/`.
- Scope: schema-v1 delivery dirs `<name>_{design|dev}_vN_YYYYMMDD` created after the owner's freeze decision (UD-0001)
  that contain `ENDPOINT_LOCK.json` with schema `minifly.endpoint_lock.v1`. Everything else is legacy: only
  `MFC retro (--all | <dir>) --out DIR` touches it, read-only, and `gate` refuses it (exit 2).
- Sibling skills: `minifly-e3-design` (what goes into the endpoint), `minifly-claims-knowledge` (CLAIM, REPORT,
  pointers, findings), `minifly-independent-review` (the T2 review), `minifly-governance` (changing the kit).
- Stage checklists with the artifacts read and written: [references/STAGES.md](references/STAGES.md).
  Recording a grant: [references/AUTHORIZATION.md](references/AUTHORIZATION.md).

## 1. Non-negotiables

- I1. Native line: no attention, Transformer, ELM or backprop. Evaluator-only ceiling learners are labelled and never sit on the native answer path.
- I2. No supplied arithmetic structure in the native line.
- I3. Raw-byte input plus allowed feedback, under the owner's gradual warmup (start ceiling 1/16).
- I4. The primary objective is retained, exact, own-output, own-EOS unseen addition at the final checkpoint, OLD protected, reported per sample and per counted compute.
- I5. Use the E0-E4 ladder of `minifly_b/package/DOCS/MINIFLY_OBJECTIVE_CONTRACT.md`; always state the level actually earned.
- I6. Confirm wording only for worlds_role SEALED_CONFIRM.
- I7. Tier-2 independence needs a reviewer from the other agent family (Codex vs Claude). Same-family sessions or subagents never count.
- I8. FINDINGS, INCIDENTS, USER_DECISIONS, DELIVERY_LEDGER, CONFIRM_LEDGER and REVIEW_DEBT are append-only.
- I9. Changes apply prospectively. Never edit frozen designs, data, verdicts or adopted pointers; correct with new rows, errata or new deliveries.
- I10. Honest exits are unpenalized.
- I11. Only sealed-world G counts as program progress. Never PASS counts, cycles, gates added or flags avoided.
- I12. Verdict evidence comes only from locked code and the program checker. Never hand-write a measurement or a verdict.

## 2. Stake tier (computed, never declared down)

`tier = max(declared, tier(lock.budget), tier(measured ATTEMPTS of this delivery), tier(family cumulative CPU), stage floor)`.
TIER-01 computes it and every GATE report prints it with its reasons. Thresholds are owner-set in
`PROJECT_DOCUMENTS/KB/CONSTRAINTS.json` (draft values below, pending UD-0001).

| Tier | When | Required | Cannot |
|---|---|---|---|
| T0 | No learner execution producing scored values (memos, saved-data reanalysis), and at most 1 CPU-h | Nothing new. If you publish text: `MFC lint <files>` (CLM-06, CLM-12, CLM-16, PTR-01..07), and stamp outputs DEV_NOT_EVIDENCE | — |
| T1 | Learner runs with scored values on technical, dev or rehearsal worlds only; delivery at most 4 CPU-h and 20,000 records; lineage family cumulative at most 12 CPU-h | ENDPOINT_LOCK, SOURCE_LOCK, ADMISSION, `launch`, cycles, generated CLAIM (DEV or PILOT). Self-checked; no reviewer | the list below |
| T2 | Above any T1 cap; any official run (MAIN or SEALED_CONFIRM), findings activation, adoption, pointer update, or post-outcome amendment of primary or guard fields | Everything in T1, plus AUTHORIZATION covering the stage, one executing cross-agent review event, witness attestations (WIT-01) and the published verification subset. SEALED_CONFIRM adds the seal, OC-01, a CONFIRM_LEDGER row and a post-run replay | — |

At T1 you cannot activate findings, update pointers, use confirm wording, reset the E3 cadence, adopt, or run official.

Lineage family: two v1 deliveries share a family when the Jaccard index of their native-closure file content hashes
is at least 0.5; families are connected components; family_id is `FAM-` + sha8 of the smallest member name; family CPU
is the measured child CPU in ATTEMPTS summed over v1 members. Splitting work into small deliveries does not lower the
tier, and the T1 cannot-list removes the payoff.

## 3. Stage map

| User step | Stage | What happens | Blocking command | Exit criterion |
|---|---|---|---|---|
| explore | S0 Explore (T0) | Read `MFC kb status`. Read FINDINGS conditions and caveats. Run the cheapest falsifier first (analytic SNR from init formulas, or saved-data reanalysis). Present options neutrally, each with cost and objective distance | none (informational) | Owner AUTHORIZATION recorded for design/implementation |
| design | S1 Design and endpoint lock | `MFC init <dir> [--parent P]`. Fill ENDPOINT_LOCK: contract block, units, split, arms, rule, guards, worlds, predictions. Record AUTHORIZATION | `MFC gate --stage lock <dir>`, then commit | Lock gate PASS; ENDPOINT_LOCK committed before any `results/**` |
| refinement, audit, test run 1 | C0 Testbed screen (before learner code) | `admit.py --testbed` runs evaluator-side ceiling and floor grids on the exact split and budget | `MFC launch --stage testbed <dir> -- python3 -I admit.py --testbed`, then `MFC score --role admission <dir>` | Ceiling passes and floor fails: continue. Otherwise CANNOT_TEST_CLAIM_AS_SPECIFIED: stop, or amend with reason testbed_invalid |
| (test run 1) | C1 Technical | Implement the learner. `MFC lock build <dir>`. One technical record per arm. Directional fixtures. Assemble ADMISSION.json | `MFC launch --stage technical <dir> -- <cmd ...>`, then `MFC gate --stage roster <dir>` | Roster gate PASS (at T2 this includes the review event) |
| audit | T2 review event | The other agent family runs executed checks, the cheapest-shortcut attempt and the canary, and attests the lock sha | REVIEW.json, checked by REV-01..07 and WIT-01 | decision ADMIT, no open BLOCKER or MAJOR |
| test run 2 | C2 Dev | Dev worlds. Outcomes are readable. Learner changes are allowed and counted in k | `MFC launch --stage dev <dir> -- <cmd ...>`, then `MFC score --role dev <dir>` | SCORES dev written, decision recorded |
| test run 3 | C3 Rehearsal | A fresh dev world never used for tuning. Frozen design. One full history per arm, with the official command template | `MFC launch --stage rehearsal <dir> -- <cmd ...>`, then `MFC gate --stage official --dry <dir>` | Dry official gate PASS |
| official run | S6 Official (always T2) | The reviewer acks the final SOURCE_LOCK. AUTHORIZATION stage official is present. SEALED_CONFIRM runs the seal protocol. An interrupted official attempt continues only with `--resume` (identical lock, missing units only, one completed unit replayed bit-identically) | `MFC launch --stage official <dir> -- <cmd ...>` (after an interruption: `MFC launch --stage official --resume <dir> -- <same cmd>`) | Exactly one completed official run (RUN-02; a legal resume chain counts as one run, RUN-04) |
| audit and search | S7 Audit and search | `MFC score --role official <dir>` writes VERDICT.json, then `MFC claim <dir>`. Post-run replay for SEALED_CONFIRM or adoption. Predictions scored. Findings proposed, then activated by the other agent. Incidents filed. Next explore | `MFC gate --stage publish <dir>` (plus `MFC gate --stage pointer --pointer P <dir>`) | Publish gate PASS |

Rules:
- Proceed only on exit 0. Read the GATE report in `checks/` for the failing ids, then use section 7.
- `launch` is the only legal way to produce counted results at T2, and it refuses a dirty closure. It re-verifies
  ENDPOINT_LOCK, SOURCE_LOCK and every closure sha on every invocation, so run one launch per phase and never chain
  phases inside one unlaunched script: a fix between phases then fails loudly at the next launch. Every counted run
  writes its runtime closure (`runlog/RUNTIME_CLOSURE_<attempt_id>.json`, computed from sys.modules, never listed by
  hand); a repo .py file it lists that SOURCE_LOCK lacks is a closure gap (RUN-03; T2 FAIL, T1 WARN).
- Cycle 0 runs before learner code. LEARNED-BANK ran its learner on T-MOD first
  (origin/charming-head `minifly_learned_bank_v1_20261024/cycles/CYCLE2.md:5-15`); only then did the testbed audit
  (`CYCLE2.md:17-36`) show that not even an exact-gradient evaluator network generalises T-MOD from that input code
  (held 0.00 / 0.11). At C0 that is a cheap CANNOT_TEST_CLAIM_AS_SPECIFIED, and because no learner-arm outcome exists
  yet, the redesign needs no UD unless it raises objective_distance (AMD-02).
- Read controls at the first technical record, before spending. Package B (`minifly_b/REPORT.md:12-26`) spent about
  80 core-hours on 832 receipts in which the N_old_rel control performed all 288 old-relation writes and ended
  digest-identical to W. ADM-CTRL-ZERO at the roster gate sees that in one technical receipt.
- Do the cheapest falsifier at explore. V6-R1's cue-to-noise ratio of 0.014
  (`minifly_v6_r1_dev_20261023/REVISION.md:40-52`) follows from its own birth formulas (one active symbol column with
  SD 0.05/sqrt 348 = 0.00268 against content noise SD 0.1), computable before any of its 34,816 records. STOP-ONLY's
  saved streams, re-scored without training, show its own stream truncated at one byte (FIXED_1) at 10.645% against
  STOP's 11.263% on NEW (`minifly_address_capacity_v1_dev_20261023/REPORT.md:7`), after 225,792 records judged
  against a zero-scoring DIRECT comparator.
- Rehearsal runs on a fresh world (CYC-03). One official attempt only (RUN-02). The one exception is an interrupted
  official attempt (nonzero or killed exit, or an incomplete unit set; a launch that died with its container is recorded
  from its in-flight marker by the next launch): `launch --stage official --resume` with the same command runs the
  missing (world, arm) units under the identical lock shas, never rewrites a completed unit, and re-runs the unit
  launch chooses into `results/official_replay/` with a bit-identical canonical row digest (RUN-02 as amended, RUN-04).
  CLAIM.provenance records interrupted, resume_attempts, resumed_units and replay_units.
- A recovered official attempt (recorded from its in-flight marker: exit -1, rusage lost) vouches for its rows only
  through a replay of one of its own units (RUN-04 since 1.0.1, spec 20.17): run `launch --stage official --resume`
  with the same command even when every unit is complete; launch then picks the replay unit among the recovered
  attempt's units (corpus S78 fails without it, S79 passes).
- Honest exits are reported exactly like passes: the generated CLAIM carries them, and nothing penalizes them.

## 4. Typed cycles

Kinds: `testbed` (C0, not counted against the cap), `technical` (C1), `dev` (C2), `rehearsal` (C3). End every cycle with

```
MFC cycle close <dir> --kind {testbed|technical|dev|rehearsal} --decision {continue|revise|stop|escalate} --reason "<one sentence>"
```

It appends one `runlog/CYCLES.jsonl` row, and the checker fills the computed fields: k, native_closure_sha and
params_sha256, the attempts since the last close, the latest gate result, outcomes_readable, and designs_tried_k so far.

When the decision is continue, CYC-02 checks the kind's exit criterion:

| Cycle | Exit criterion |
|---|---|
| testbed | A TESTBED result exists |
| technical | A PASS roster GATE report at the current SOURCE_LOCK |
| dev | `results/dev/SCORES.json` exists |
| rehearsal | A PASS `official --dry` GATE report |

- The test-cycle cap is `AUTHORIZATION.caps.max_cycles` (default 3), counting technical, dev and rehearsal (CYC-01).
  A kind may repeat within the cap. `launch` refuses new technical, dev or rehearsal attempts beyond it. Each stage
  writes only its own results directory and worlds: a testbed attempt (not counted) writes only CEILING_*/FLOOR_* rows
  under results/admission/, so dev work run as `--stage testbed` FAILs CYC-01 for good. Attempts no close covers
  count as an open cycle, so skipping `cycle close` never saves a cycle: `launch` refuses another stage's attempt
  until the open cycle is closed, and the official stage needs every test attempt in a closed cycle (CYC-01).
  runlog/CYCLES.jsonl and ATTEMPTS.jsonl are append-only; never edit or delete a row.
  CYC-04 (checker 1.0.1, spec 20.16) re-checks this at publish, so a row tidied after the official run still fails
  (corpus S77).
- End a cycle on a decision, never on a count: revise, stop and escalate are as legitimate as continue.
- designs_tried_k is computed, never declared: the number of distinct (native_closure_sha, params_sha256) pairs over all
  outcome-bearing attempts and rows. The headline prints `best of k=<k>`.
- Write no per-cycle audit files, plans, predictions or revision taxonomies. The CYCLES row and the reason sentence are the record.

## 5. Authorization

Detail and worked examples: [references/AUTHORIZATION.md](references/AUTHORIZATION.md). Checked by AUT-01..04 at the lock gate.

- Record Mr. Zhou's words verbatim, with UTC time, session and channel. Never paraphrase.
- Record options_shown and, separately, recommendation_shown. No option label contains '(Recommended)' (AUT-03).
- Stage booleans: implementation, technical, dev_cycles, official, push, push_lock, adoption, pointer_update. A stage the owner
  did not grant is false.
- Every numeric cap must appear in the owner's verbatim text (AUT-02). Otherwise the defaults apply: 2 CPU-h, 5,000 records,
  3 cycles, no official, no push, no adoption, no pointer update.
- Write conditionals as predicates over artifact fields, evaluated by the checker or by the non-beneficiary agent
  (AUT-04). Never judge your own conditional grant.
- Answer a pressure phrase ('keep going', 'just finish it') by restating the caps to the owner before acting. It raises nothing.
- A proposal that touches constraints is REQUIRES_USER_DECISION and is never executed in the same iteration.

## 6. Changing things after an outcome

Once ENDPOINT_LOCK is committed, every later committed lock content is an amendment:
- Append an AMENDMENTS row that links the old and new lock shas (AMD-01) and lists results_seen, which must equal the
  computed list of outcome files committed before it (AMD-03). reason is one of ceiling, testbed_invalid,
  measurement_defect or operational; testbed_invalid needs a TESTBED result on the original endpoint showing ceiling FAIL
  or floor PASS.
- A UD (scope `amendment:<delivery>`) is needed whenever primary, guards, gates, arms, units, split or task change after
  any learner-arm outcome, and whenever objective_distance increases on any axis (AMD-02). Such an amendment also keeps
  the original endpoint in coprimary[] and bumps protocol_id. A post-outcome amendment of primary or guard fields
  raises the delivery to T2.
- The original endpoint stays co-primary and is reported in the headline (` | amended: co-primary <gate>`).
- Official worlds stay untouched: their ids appear in no outcome artifact before the latest amendment and in no other
  delivery (AMD-04).
- Editing admit.py or any other file in endpoint_files_sha256 after the lock without an amendment fails AMD-05.

Example: LEARNED-BANK swapped its primary task from T-MOD to T-PAR after learner outcomes on D1
(origin/charming-head `minifly_learned_bank_v1_20261024/cycles/CYCLE2.md:39`). Under schema v1 that needs the owner's
UD, keeps T-MOD co-primary and bumps protocol_id (AMD-02). Never switch task or metric to rescue a run.

## 7. What a FAIL allows

Read the failing ids in the GATE report. Never edit the checker, skip, hand-write numbers or rerun official.

| Failing check | Allowed responses |
|---|---|
| SELF-01, SCHEMA-01 | Commit or revert local checker edits through governance. Use `retro` for legacy dirs |
| LOCK-01..04, LOCK-10..12 | Fix ENDPOINT_LOCK before any results exist. After results exist, write an amendment (section 6) |
| LOCK-05..07 | Fix the code or the closure and rerun `MFC lock build <dir>`. For a forbidden pattern, remove it or move it out of the native answer path |
| LOCK-09 | Open a held-E3 endpoint, or obtain an owner waiver (UD scope `e3_waiver:<delivery>`) |
| AMD-01..05 | Write AMENDMENTS row(s) with results_seen. Obtain a UD where required. Or revert the change |
| ADM-TB-CEIL, ADM-TB-FLOOR | Verdict CANNOT_TEST_CLAIM_AS_SPECIFIED. Redesign through an amendment with reason testbed_invalid |
| ADM-CTRL-ZERO, ADM-CTRL-ISO, ADM-CTRL-DOSE | INVALID with reason TREATMENT_NOT_DELIVERED. Fix the control and repeat the technical cycle on technical worlds only |
| ADM-INPUT, ADM-ACQ | INPUT_STATE_NOT_ADMITTED or NOT_ACQUIRING. Redesign the learner (a new k) or stop |
| ADM-EXPO | Regenerate streams or split under a new protocol_id (amendment) |
| RED-01 | The rule is unreachable. Amend it before any learner-arm outcome; no UD is needed if none exists yet |
| REV-01..07, WIT-01 | Obtain the review, or record REVIEW_DEBT (dev and pilot only) |
| RUN-01, RUN-03 | Re-run the work through `launch` with the runtime-closure helper; a closure gap means the module joins the static closure (or dynamic_imports) and `MFC lock build <dir>` runs again, a new design |
| RUN-02, RUN-04 | Never rerun official. An illegal resume (another lock, a rewritten unit, no or a differing replay) stays FAIL: report it; a follow-up is a new protocol on fresh worlds |
| Verdict FAIL, INVALID, INCOMPLETE | Report it. Never rerun official. A follow-up is a new protocol on fresh worlds |
| CLM-01..16, PTR-01..07 | Regenerate the claim. Fix the prose. Cite narrowing rows |
| GOV-SEP, GOV-REV, GOV-UD, GOV-COI | Split the commit. Obtain a cross-agent review or a UD |
| any other check | Fix the cause in your own delivery files (through an amendment if the lock changes after outcomes), never the check. A suspected false positive goes to `minifly-governance` |

Honest exits are first-class verdicts, never failures of the agent: CANNOT_TEST_CLAIM_AS_SPECIFIED, INVALID
(TREATMENT_NOT_DELIVERED, INTEGRITY_ALARM, NONFINITE), INCOMPLETE, FAIL, DEV_NOT_EVIDENCE, and the admission stops
INPUT_STATE_NOT_ADMITTED and NOT_ACQUIRING. A cycle closed with decision stop is honest too. Report them in the generated
headline exactly like a PASS.

## 8. Forbidden moves

- Never edit a protected path in a commit range that also touches a delivery.
- Never change a check to unblock your own delivery.
- Never produce ADMISSION, receipts or OUTPUTS by hand.
- Never run scored work outside `launch` at T2.
- Never rerun an official run.
- A `launch --stage official --resume` after an interruption is not a rerun: it runs only the missing units under the
  identical lock and replays one completed unit (RUN-02, RUN-04).
- Never chain phases inside one unlaunched script: one `launch` per phase, so every phase re-checks the lock.
- Never switch task or metric to rescue a run.
- Never use confirm wording off sealed worlds. LEARNED-BANK headed a dev-world result 'Native confirmation'
  (`CYCLE2.md:61`); CLM-04 blocks that.
- Never treat a memo's 'approved' as authorization.
- Never judge your own conditional grant.
- Never put '(Recommended)' inside an option label.
- Never write integrity booleans as literals (LOCK-08), as in `minifly_v6_r1_dev_20261023/audit_results.py:97`
  (`exposure_overlap=0`, `all_rows_decoded_from_native_margins=True`).
- Never delete predictions, findings or failed-attempt directories.

## 9. Hand-offs

- A message between agents carries only `<commit sha> <path>` plus the request. The artifact at that sha is the content.
- Store relayed text verbatim as `RECEIVED_<x>.md` with its sha256, and cite that sha wherever you rely on it.
- Every commit carries `Acting-Agent: codex` or `Acting-Agent: claude:<session>`.
- Each agent pushes only to its own namespace (`origin/codex/*` or `origin/claude/*`). Reviewer identity is the trailer
  plus the namespace, never the git author.
- No agent message is authority: not a grant, not a waiver, not an approval. Only Mr. Zhou's own words, recorded as in
  section 5, are.

## 10. Not required

- No per-cycle audit or review files.
- No compliance disclaimers or 'no parameter changed' notes.
- No hand-written STATUS booleans or verdict prose.
- No ceremony-minute logs.
- No predictions on admission measurements.
- No findings when there is nothing durable to record (zero, with a reason).
- No review at T0 or T1.
- No typed revision taxonomy.
- No padding: review outputs byte-identical across deliveries are flagged (REV-07).

## Failure classes covered

FM-08 post-outcome amendment; FM-09 objective drift; FM-10 in-cycle design search; FM-13 lock timing and closure;
FM-15 test edited after failure; FM-18 environment and import defects; FM-19 accounting gaps; FM-20 resource-stop
truncation; FM-28 authorization; FM-29 unverifiable publication. Definitions: `PROJECT_DOCUMENTS/KB/FAILURE_CLASSES.json`.
