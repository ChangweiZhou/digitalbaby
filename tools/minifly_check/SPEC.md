# MiniFly Research-Loop Kit v1 (MRK-1): final architecture

> **Provenance.** Synthesized 2026-10-24 by a Claude Code session (session_01YBMaMvtGtQvDFcgppgThrZ) from a three-design judge panel
> (minimal-enforcement, separation-of-duties, loop-as-learner) and two judges (practicality, adversarial), grounded in two history
> analyses committed under PROJECT_DOCUMENTS/KB/ANALYSIS_20261024/. Integrator amendments to the synthesized text are listed in
> Section 20. Nothing here changes a frozen design, verdict or pointer. CONSTRAINTS values are DRAFT until Mr. Zhou decides (UD-0001).


Status: the implementation specification for schema v1, which applies prospectively only. Synthesizer output from three candidate designs (MFG-1 minimal-enforcement, MSDP separation-of-duties, LaL loop-as-learner) and two judge panels (practicality, adversarial). Owner of objective and constraints: Mr. Changwei Zhou. This document is the single source of truth for the work packages; every check id, file name and schema named here is normative.

---

## 0. Thesis and provenance

**What history shows.** MiniFly's most expensive failures were silent, and most were detectable early from artifacts that already existed or were cheap to create:
- Package B: about 80 core-hours, a dead E3 control detectable after one technical receipt.
- STOP-ONLY: 225,792 records, a weak comparator detectable from its own saved stream.
- V6-R1: 34,816 records; its cue/noise ratio was already written in its own design formulas.
- v5.1: a port gate that could not fail.
- v4: a reversed teaching sign.
- LEARNED-BANK: an untestable task and a fail-open reducer.

Rules written as prose (the E0-E4 contract, the 10-14 continuity rules) existed and recurred as failures anyway.

**What MRK-1 does about it.** It turns the key rules into fail-closed machine checks and places them at four choke points where compute or credibility is spent:
1. LOCK, before the first scored value.
2. ROSTER, before spend beyond a technical allowance.
3. OFFICIAL, before a main or confirmation run.
4. PUBLISH/POINTER, before anything other agents read first.

One program-owned Python-stdlib checker (`tools/minifly_check`) owns everything that decides a verdict:
- the metric;
- the shortcut ladder;
- the strata;
- the decision rule.

A delivery supplies declarations and saved outputs, never verdict code. Cost scales with a computed tier. Honest exits are first-class and unpenalized. The kit's own improvement is governed so that agents cannot pass by editing it. Generalization is measured with numbers at two levels:
- **Learner:** taught units lead to determined, held-out compositions, scored against the strongest shortcut, with causal unit dependence.
- **Research loop:** unit facts and failure classes lead to predictions and checks, scored on future experiments and on sealed canaries.

**Base taken from MFG-1:**
- the four choke points and one stdlib checker;
- the canonical program-wide scorer and shortcut ladder, with the substitution test;
- missing = FAIL, with no `--force` or `--skip`;
- the generated CLAIM and headline, with pointers as byte copies;
- the Phase-0 retro with a measured false-flag rate before freeze;
- GOV-SEP;
- the explicit left-out list;
- the honest exits.

**Changed relative to MFG-1, to close loopholes the judges found:**
- The executor-written `verdict.py` is replaced by a declarative rule in the lock, evaluated by program-owned `decide.py`. This closes fixture-detecting reducers.
- own-EOS is recomputed from `stream_hex`.
- Control receipts use per-channel digests at the first manipulated record: unchanged on lesioned channels, and equal to W elsewhere.
- Tier includes measured, family-cumulative compute from a launch wrapper.
- The E3 cadence counter is computed, and resets only on an admitted, scored, main-world held-E3 primary.
- Every SEALED_CONFIRM uses post-lock two-party entropy.
- At tier 2, lock timing needs an external witness.
- Findings are activated only by the other agent family.
- Owner decisions are required only for loosening and constraints. Tightening needs a cross-agent review.
- Each delivery records 0-3 findings, with a reason when zero.
- Prediction calibration is never an agent or kit score.
- Canary operators are chosen by arm role and are explicitly not called held-out.
- `init` and `lock build` scaffolding are added.

**Grafted from LaL:**
- U2C: units fixed before streams, strata S0-S4 computed from item content, the determinacy witness, and the unit-dependency index (UDI) for E3 promotion only.
- An advisory window plus an `introduced_in` version per check.
- Triggers, computed and used only at T1.
- R0-R4 levels for facts and checks.
- `selftest --mutate`.
- A one-shot KB backtest against a contract-only forecaster.
- The conflict-of-interest rule.
- Sealed canary batches written by someone who did not author the checker.
- Sealed-world G as the only progress metric.
- An owner spot-check of 3 numbers per census.
- A sunset path for checks.

**Grafted from MSDP:**
- cumulative lineage cost;
- REVIEW_DEBT;
- default caps for terse grants;
- conditions evaluated by a non-beneficiary;
- hand-off by commit sha plus path;
- unexecuted findings marked UNVERIFIED;
- the V51 status legacy map;
- the cheapest-shortcut bounty;
- challenge seeds chosen by the reviewer after the lock.

**Dropped as ceremony or impractical** (several are deferred until an incident shows a silent failure they would have caught):
- five-role separation and an evaluator-owner at every T2 delivery;
- governor one-time codes as a requirement (now optional hardening);
- salt escrow, the vault family and the holdout touch registry;
- blind/full hand-off packs, ACK cross-witnessing and SESSION_PROMPTS;
- manifest self-verification on every call, which would brick the kit;
- runtime_guard, instrument.py, the input tap and a separate learner process;
- tripwires and decoys;
- shingle similarity;
- typed CYCLE_PLAN/CYCLE_RECORD revision taxonomies, seeded bugs per cycle and PREMORTEM;
- hash-chained ledgers, which break on cross-branch merge;
- Claude-only hooks as enforcement;
- 10 or 13 skills;
- OC simulation for every design and UDI at every T2;
- review canaries in 1 of 5 reviews, MM-2..10 dashboards and recurring backtests;
- 100% corpus coverage of FM-01..30 before any merge;
- ceremony_minutes;
- mandatory findings.

---

## 1. Scope, protected invariants, non-goals

**Protected invariants** live in `PROJECT_DOCUMENTS/KB/CONSTRAINTS.json` and are owner-defined. Weakening any of them requires a recorded USER_DECISIONS row containing Mr. Zhou's verbatim words.

- I1. Native line: no attention, Transformer, ELM or backprop. Evaluator-only ceiling learners are allowed, are labelled, and never sit on the native answer path.
- I2. No supplied arithmetic structure in the native line.
- I3. Raw-byte input plus allowed feedback, under the owner's gradual warmup (start ceiling 1/16).
- I4. Primary objective: retained, exact, own-output, own-EOS unseen addition at the final checkpoint, OLD protected, reported per sample and per actual counted compute.
- I5. The E0-E4 ladder of `minifly_b/package/DOCS/MINIFLY_OBJECTIVE_CONTRACT.md`. The report always states the level actually earned.
- I6. Confirm wording is allowed only for worlds_role SEALED_CONFIRM.
- I7. Tier-2 independence requires a reviewer from the other agent family (Codex vs Claude). Same-family sessions or subagents never count.
- I8. FINDINGS, INCIDENTS, USER_DECISIONS, DELIVERY_LEDGER, CONFIRM_LEDGER and REVIEW_DEBT are append-only.
- I9. Changes apply prospectively. Frozen designs, data, verdicts and adopted pointers are never edited. Corrections are new rows, errata or new deliveries.
- I10. Honest exits are unpenalized.
- I11. Only sealed-world G (Section 9.6) counts as program progress. Never PASS counts, cycles, gates added or flags avoided.
- I12. Evidence for the verdict comes only from locked code and the program checker. Measurements and verdicts are never hand-written.

**Applicability.** Schema v1 applies to delivery directories `<name>_{design|dev}_vN_YYYYMMDD` created after the owner's freeze decision (UD-0001) and containing `ENDPOINT_LOCK.json` with schema `minifly.endpoint_lock.v1`. All other directories are legacy, and only `retro` (read-only) touches them.

**Non-goals.**
- Tools never train or run learners.
- Tools never write inside legacy directories or into frozen designs.
- Tools never fetch from the network.
- Tools never alter historical verdicts or adopted pointers.

---

## 2. File tree

```
PROJECT_DOCUMENTS/KB/AGENTS_POINTER.md  pointer text (<=20 lines; protected) for Mr. Zhou to paste into their own
                                  untracked root AGENTS.md. A tracked root AGENTS.md is NOT created (it would collide with that local file).
.gitattributes                    PROJECT_DOCUMENTS/KB/*.jsonl merge=union (protected)
README.md                         + 6-line 'Research-loop kit (schema v1)' section
.claude/skills/
  minifly-research-loop/SKILL.md      references/STAGES.md, references/AUTHORIZATION.md
  minifly-e3-design/SKILL.md          references/RULE_GRAMMAR.md, references/ADMISSION_AND_RUNNER_CONTRACT.md, references/STRATA_WITNESS_UDI.md
  minifly-claims-knowledge/SKILL.md   references/CLAIM_AND_POINTERS.md, references/KNOWLEDGE_LEDGER.md
  minifly-independent-review/SKILL.md references/REVIEW_PROTOCOL.md, references/SEAL_PROTOCOL.md
  minifly-governance/SKILL.md         references/CHANGE_CLASSES.md, references/HELDOUT_CANARIES.md
tools/minifly_check/              (all protected)
  __init__.py  minifly_check.py  common.py  launch.py
  lockcheck.py  seal.py
  addition.py  scorer.py  decide.py  reducercheck.py
  admitcheck.py  review.py  canary.py
  claimcheck.py  kb.py
  governance.py  retro.py
  RULES.json  PROTECTED.json  EXPECTED.json  SCHEMAS.md  README.md
  templates/  ENDPOINT_LOCK.template.json AUTHORIZATION.template.json REVIEW.template.json
              ADMISSION.example.json RECEIPT.example.json OUTPUTS.example.jsonl TAUGHT.example.jsonl
              CLAIM.example.json POINTER.example.json admit_skeleton.py
  corpus/     _build/ (generator + base), C00a..C30, G01..G05, K01, heldout/README.md
  tests/      test_<module>.py per module
PROJECT_DOCUMENTS/KB/
  README.md  CONSTRAINTS.json (protected)  USER_DECISIONS.jsonl (protected, append-only)
  FINDINGS.jsonl  INCIDENTS.jsonl  FAILURE_CLASSES.json  DELIVERY_LEDGER.jsonl
  CONFIRM_LEDGER.jsonl  REVIEW_DEBT.jsonl  CANARY_COMMITMENTS.jsonl
  GOVERNANCE/  (cross-agent governance reviews REVIEW_<id>.json)
  RETRO/       (RETRO_<date>.json, HELDOUT_<date>.json, BACKTEST_<date>.json)
<delivery>/   (schema v1)
  ENDPOINT_LOCK.json  SOURCE_LOCK.json  params.json  AUTHORIZATION.json  AMENDMENTS.jsonl
  admit.py (evaluator, endpoint-locked)   run.py, native modules (source-locked)
  ADMISSION.json
  manifests/ HELD.jsonl OLD.jsonl NEW.jsonl TAUGHT/<world>/<arm>.jsonl UDI_VARIANTS.json
  results/receipts/<world>/<arm>.json
  results/{admission,dev,rehearsal,official}/OUTPUTS.jsonl[.gz] (or inside RESULT_BUNDLE*.zip)
  results/admission/FIXTURES.jsonl  results/<role>/SCORES.json  results/official/VERDICT.json
  runlog/ATTEMPTS.jsonl  runlog/CYCLES.jsonl
  review/REVIEW.json  review/*.py + outputs  review/SEAL_COMMIT.json  review/SEAL_REVEAL.json
  checks/GATE_<stage>_<utc>.json  checks/CANARY_<utc>.json
  CLAIM.json  REPORT.md  ERRATA.jsonl (only if a published verdict changes)
```

---

## 3. The loop (the user's framing, with the weak points fixed)

The user's loop is explore -> experiment (design, refinement -> audit -> test run x3 -> official run) -> audit and search. MRK-1 keeps that shape and fixes eight weak points:

1. Cycle 0, a cheapest-falsifier testbed screen, runs before any learner code exists.
2. Cycles are typed, with mechanical exit criteria.
3. Cycles end on a decision, not on a count.
4. The rehearsal runs on a fresh world.
5. Outcome-informed design changes are counted automatically ("best of k").
6. The audit step is an executing cross-agent review at the cheap moment (after admission, before spend) at T2.
7. Official runs are never rerun.
8. Audit and search score predictions and feed findings and incidents back.

### 3.1 Stage map

`MFC` stands for `python3 -I tools/minifly_check/minifly_check.py`.

| User step | MRK stage | What happens | Blocking command | Exit criterion |
|---|---|---|---|---|
| explore | S0 Explore (T0) | Read `MFC kb status`: cadence, open incidents, narrowed findings, review debts, sealed G series. Read FINDINGS conditions and caveats. Run the cheapest falsifier first: analytic SNR from init formulas, or saved-data reanalysis. Present options to the owner neutrally, each with cost and objective distance. | none (informational) | Owner AUTHORIZATION recorded for design/implementation |
| design | S1 Design and endpoint lock | `MFC init <dir> [--parent P]`. Fill ENDPOINT_LOCK: the contract block as fields, units, split, arms, rule, guards, worlds, predictions. Record AUTHORIZATION. | `MFC gate --stage lock <dir>`, then commit | Lock gate PASS. ENDPOINT_LOCK committed before any `results/**`. |
| refinement -> audit -> test run 1 | C0 Testbed screen (before learner code) | `admit.py --testbed` runs evaluator-side ceiling and floor grids on the exact split and budget. | `MFC launch --stage testbed <dir> -- python3 -I admit.py --testbed`, then `MFC score --role admission <dir>` | Ceiling passes and floor fails: continue. Otherwise CANNOT_TEST_CLAIM_AS_SPECIFIED: stop, or amend with reason testbed_invalid. |
| (test run 1) | C1 Technical | Implement the learner. `MFC lock build`. One technical record per arm. Directional fixtures. Assemble ADMISSION.json. | `MFC launch --stage technical ...`, then `MFC gate --stage roster <dir>` | Roster gate PASS. At T2 this includes the review event. |
| audit | T2 review event | The other agent family runs executed checks, the cheapest-shortcut attempt and the canary, and attests the lock sha. | REVIEW.json, checked by REV-* and WIT-01 | decision ADMIT, with no open BLOCKER or MAJOR |
| test run 2 | C2 Dev | Dev worlds. Outcomes are readable. Learner changes are allowed and are counted in k. | `MFC launch --stage dev ...`, then `MFC score --role dev <dir>` | SCORES dev written, decision recorded |
| test run 3 | C3 Rehearsal | Fresh dev world never used for tuning. Frozen design. One full history per arm, with the same command template as official. | `MFC launch --stage rehearsal ...`, then `MFC gate --stage official --dry <dir>` | Dry official gate PASS |
| official run | S6 Official (always T2) | Reviewer acks the final SOURCE_LOCK. AUTHORIZATION stage official is present. For SEALED_CONFIRM, the seal protocol (Section 10) runs. | `MFC launch --stage official <dir> -- ...` | Exactly one completed official attempt (RUN-02) |
| audit and search | S7 Audit and search | `MFC score --role official` produces VERDICT.json, then `MFC claim`. Post-run replay for SEALED_CONFIRM or adoption. Predictions scored. Findings proposed, then activated by the other agent. Incidents filed. Next explore. | `MFC gate --stage publish <dir>` (+ `--stage pointer --pointer P`) | Publish gate PASS |

### 3.2 Typed cycles

Kinds:
- `testbed` (C0, not counted against the cap);
- `technical` (C1);
- `dev` (C2);
- `rehearsal` (C3).

Every cycle ends with `MFC cycle close <dir> --kind K --decision {continue|revise|stop|escalate} --reason "<one sentence>"`. This appends one CYCLES.jsonl row. The checker fills the computed fields:
- k;
- native_closure_sha and params_sha;
- attempts since the last close;
- the latest gate result;
- outcomes_readable;
- designs_tried_k so far.

Exit criteria, checked by CYC-02 whenever the decision is continue:

| Cycle | Exit criterion |
|---|---|
| testbed | A TESTBED result exists |
| technical | A PASS roster GATE report at the current SOURCE_LOCK |
| dev | `results/dev/SCORES.json` exists |
| rehearsal | A PASS `official --dry` GATE report |

Rules:
- The test-cycle cap is `AUTHORIZATION.caps.max_cycles` (default 3), counting technical, dev and rehearsal.
- A cycle kind may repeat within the cap.
- `launch` refuses new dev or rehearsal attempts beyond the cap.
- No per-cycle audit files, plans, predictions or revision taxonomies are written.
- designs_tried_k is computed, never declared: it is the number of distinct (native_closure_sha, params_sha) pairs over all outcome-bearing attempts and rows.

### 3.3 What a FAIL allows

| Failing family | Allowed responses (never: edit the checker, skip, hand-write numbers, rerun official) |
|---|---|
| SELF, SCHEMA | Commit or revert local checker edits through governance. Use `retro` for legacy. |
| LOCK-01..04, 10..12 | Fix ENDPOINT_LOCK before any results exist. After results exist, write an amendment (AMD rules). |
| LOCK-05..07 | Fix code or closure and rerun `lock build`. For a forbidden pattern, remove it or move it out of the native answer path. |
| LOCK-09 | Open a held-E3 endpoint, or obtain an owner waiver (UD scope `e3_waiver:<delivery>`). |
| AMD-* | Write AMENDMENTS row(s) with results_seen. Obtain a UD where required. Or revert the change. |
| ADM-TB | Verdict CANNOT_TEST_CLAIM_AS_SPECIFIED. Redesign through an amendment with reason testbed_invalid. |
| ADM-CTRL | INVALID with reason TREATMENT_NOT_DELIVERED. Fix the control and repeat the technical cycle on technical worlds only. |
| ADM-INPUT, ADM-ACQ | INPUT_STATE_NOT_ADMITTED or NOT_ACQUIRING. Redesign the learner (a new k) or stop. |
| ADM-EXPO | Regenerate streams or split under a new protocol_id (amendment). |
| RED-01 | The rule is unreachable. Amend it before any learner-arm outcome; no UD is needed if none exists yet. |
| REV, WIT | Obtain the review, or record REVIEW_DEBT (dev and pilot only). |
| Verdict FAIL, INVALID, INCOMPLETE | Report it. Never rerun official. A follow-up is a new protocol on fresh worlds. |
| CLM, PTR | Regenerate the claim. Fix the prose. Cite narrowing rows. |
| GOV | Split the commit. Obtain a cross-agent review or a UD. |

### 3.4 Forbidden moves

- Editing any protected path in a commit range that also touches a delivery.
- Changing a check to unblock your own delivery.
- Producing ADMISSION, receipts or OUTPUTS by hand.
- Running scored work outside `launch` at T2.
- Rerunning an official run.
- Switching task or metric to rescue a run.
- Using confirm wording off sealed worlds.
- Treating a memo's 'approved' as authorization.
- Judging your own conditional grant.
- Putting '(Recommended)' inside option labels.
- Writing integrity booleans as literals.
- Deleting predictions, findings or failed-attempt directories.

### 3.5 Not required (anti-ceremony)

- No per-cycle audit or review files.
- No compliance disclaimers or 'no parameter changed' notes.
- No hand-written STATUS booleans or verdict prose.
- No ceremony-minute logs.
- No predictions on admission measurements.
- No findings when there is nothing durable to record (0 with a reason).
- No review at T0 or T1.
- No typed revision taxonomy.
- Byte-identical review outputs across deliveries are flagged as padding (REV-07).

---

## 4. Stake tiers (computed, never declared down)

`tier = max(declared, tier(lock.budget), tier(measured ATTEMPTS of this delivery), tier(family cumulative CPU), stage floor)`. Thresholds are owner-set in CONSTRAINTS; proposed values are shown.

| Tier | When | Required | Cannot |
|---|---|---|---|
| T0 | No learner execution producing scored values (memos, saved-data reanalysis), and at most 1 CPU-h | Nothing new. If text is published: `MFC lint <files>` (CLM-06, CLM-12, CLM-16, PTR-*), and outputs stamped DEV_NOT_EVIDENCE. | — |
| T1 | Learner runs with scored values on technical, dev or rehearsal worlds only. Delivery at most 4 CPU-h and 20,000 records. Lineage family cumulative at most 12 CPU-h. | ENDPOINT_LOCK, SOURCE_LOCK, ADMISSION, launch wrapper, cycles, generated CLAIM (DEV or PILOT). Self-checked; no reviewer. | Activate findings, update pointers, use confirm wording, reset E3 cadence, adopt, run official |
| T2 | Above any T1 cap. Any official run (MAIN or SEALED_CONFIRM), findings activation, adoption, pointer update, or post-outcome amendment of primary or guard fields. | Everything in T1, plus: AUTHORIZATION covering the stage; one executing cross-agent review event; witness attestations (WIT-01); the published verification subset. SEALED_CONFIRM adds the seal, OC-01, a CONFIRM_LEDGER row and a post-run replay. | — |

Lineage family:
- Two v1 deliveries are in the same family when the Jaccard index of their native-closure file content hashes is at least 0.5.
- Families are connected components under that relation.
- family_id = `FAM-` + sha8(lexicographically smallest member name).
- Family cumulative CPU is the sum of `ATTEMPTS.cpu_children_s` over v1 members.

This defeats salami-slicing, and the T1 'cannot' list removes its payoff.

---

## 5. Roles and independence

1. **Executor** (usually Codex; Claude in its own deliveries).
   - Writes ENDPOINT_LOCK, params.json, admit.py, the learner and the runner.
   - Runs everything through `launch` and runs the gates.
   - Trusted at T1 only.
   - Never writes measurements or verdicts.
2. **Program checker** (`tools/minifly_check`).
   - Written once and corpus-tested.
   - Owns: own-EOS recomputation, truth semantics for addition, strata, determinacy, shortcut ladder, decision rule, G, OLD guard, UDI, OC and claim generation.
   - This removes per-delivery common-mode auditors at the root: no delivery code computes the gated metric or the verdict.
3. **Reviewer** (the other agent family).
   - Required once per T2 delivery.
   - Runs executed checks and the cheapest-shortcut attempt, chooses the canary seed after the lock, attests lock shas, holds the seal nonce for SEALED_CONFIRM, and replays official histories for SEALED_CONFIRM or adoption.
   - Identity is derived from the `Acting-Agent:` commit trailer plus the remote branch namespace containing the commit (`origin/claude/*` vs `origin/codex/*`). Never from git author: Codex commits as 'Pencilbard'.
4. **Owner** (Mr. Zhou). Sole authority for:
   - CONSTRAINTS, threshold floors and tier thresholds;
   - waivers;
   - amendments of primary or guards after outcomes;
   - adoption and pointer authorization;
   - loosening any check or skill;
   - official GO.

   Agents transcribe the owner's words verbatim and the owner spot-checks them. At each census the owner traces 3 random CLAIM numbers to raw artifacts.
5. **Held-out author** (the owner, or a fresh session that has not read `tools/minifly_check/*.py`). Writes sealed canary batches from FAILURE_CLASSES definitions and SCHEMAS.md only.

**Single-agent mode.** If the other family is unavailable:
- T2 roster spend may proceed only with a REVIEW_DEBT row, labelled DEV or PILOT.
- SEALED_CONFIRM, findings activation, adoption and pointer updates are blocked until a real cross-family review repays the debt.
- At 3 or more open debts program-wide, no new T2 roster may start.
- A same-family subagent review is debt, not review.

**Stated independence limits.**
- In-repo identity is tamper-evident, not tamper-proof. Codex can write any trailer or branch.
- USER_DECISIONS rows are agent-transcribed.
- The checker is a new program-wide common mode. This is mitigated by:
  - hand-computed clean fixtures;
  - `selftest --mutate`;
  - reviewer trace-a-number checks;
  - the owner spot-check.

---

## 6. Skills (five; agent-agnostic; each SKILL.md under 400 lines)

All five skills share these conventions:
- Field-level schemas live once, in `tools/minifly_check/SCHEMAS.md`. Skills link to it rather than restating fields.
- Commands use `MFC`.
- Each skill opens with a 5-line 'If you only read this' box.
- Each skill ends with 'Not required' and 'Failure classes covered'.

### 6.1 minifly-research-loop

Frontmatter description (999 chars): "Use at the start of any MiniFly / digitalbaby research action by Codex or Claude and at every stage change: exploring what to try next, freezing a design or ENDPOINT_LOCK.json, running cycle 0 (testbed screen) or test cycles 1-3, launching any roster, pilot, official or confirmation run, recording or relying on Mr. Zhou's authorization, or wanting to change a task, metric, comparator, margin, world or check after seeing an outcome. Applies to every delivery dir <name>_{design|dev}_vN_YYYYMMDD created after the schema-v1 freeze. Gives the stage map of the loop (explore, design, cycle 0, technical, dev, rehearsal, official, audit and search), the computed stake tier, the one blocking command per stage (python3 -I tools/minifly_check/minifly_check.py init|gate|launch|score|cycle), typed cycles with exit criteria, default caps for short grants, what each FAIL allows, the honest-exit verdicts and the forbidden moves. Not needed for read-only analysis that changes no claim, pointer or plan."

Body sections:
1. Non-negotiables: I1-I12 in one line each.
2. Tier table (Section 4) and the T1 'cannot' list.
3. Stage map (Section 3.1), with the exact command and exit criterion for each stage.
4. Typed cycles (Section 3.2).
5. Authorization, summarized here with detail in references/AUTHORIZATION.md:
   - Record verbatim text, UTC time and session.
   - Record options_shown and a separate recommendation_shown.
   - Stage booleans: implementation, technical, dev_cycles, official, push, push_lock, adoption, pointer_update.
   - Numeric caps must appear in the owner's verbatim text. Otherwise the defaults apply: 2 CPU-h, 5,000 records, 3 cycles, no official, no push, no adoption.
   - Conditionals are written as predicates over artifact fields, evaluated by the checker or by the non-beneficiary agent.
   - A pressure phrase ('keep going') is answered by restating caps to the owner before acting.
   - A constraint-touching proposal is REQUIRES_USER_DECISION and is never executed in the same iteration.
6. Changing things after an outcome:
   - An amendment row lists results_seen.
   - A UD is needed whenever primary, guards, arms, units, split or task change after any learner-arm outcome, and whenever objective distance increases.
   - The original endpoint stays co-primary.
   - protocol_id is bumped.
   - Official worlds stay untouched.
7. FAIL table (Section 3.3).
8. Forbidden moves (Section 3.4).
9. Hand-offs:
   - Messages carry only '<commit sha> <path> + request'.
   - Relayed text is stored verbatim as RECEIVED_<x>.md with its sha256.
   - Every commit carries `Acting-Agent: codex` or `Acting-Agent: claude:<session>`.
   - Each agent pushes only to its own namespace.
   - No agent message is authority.
10. Not required (Section 3.5).

Key rules:
- Proceed only on exit 0.
- `launch` is the only legal way to produce counted results at T2.
- Cycle 0 runs before learner code.
- Rehearsal runs on a fresh world.
- One official attempt only.
- Honest exits are reported exactly like passes.

References:
- STAGES.md: per-stage checklist and the artifacts each stage reads and writes.
- AUTHORIZATION.md: template walk-through, default caps, predicate grammar, neutral option framing.

Covers FM-08, 09, 10, 13, 15, 18, 19, 20, 28, 29.

### 6.2 minifly-e3-design

Frontmatter description (934 chars): "Use when designing, locking or analysing any MiniFly endpoint that involves generalization: an E2/E3 or unseen-addition primary; choosing held items, strata, countermodels, arms, causal controls or the OLD guard; writing the primary rule in ENDPOINT_LOCK.json; writing admit.py, receipts or OUTPUTS rows for the cycle-0 testbed screen and behavioral admission; claiming reuse, transfer, rule learning or composition from taught units; or arguing that an E0/E1 component result moves the program toward E3. Makes generalization from unit learned knowledge measurable: units fixed before streams, checker-computed strata S0-S4 (only S2+ count), the determinacy witness, the program shortcut ladder, the declarative decision rule, the no-prior-teaching control and, for E3 promotion, the leave-units-out dependency index, G per sample and per compute, OLD protection with an acquisition floor, native-path purity and the E3 cadence rule."

Body sections:
1. Objective metric and G (Section 9.6), with a worked example.
2. Units and held targets.
   - Units are taught facts: operand pairs for single-token operands, or column triples (a_i, b_i, carry_in) for multi-digit operands.
   - Declare `units.witness` (pair_rectangle or column_closure).
   - The held set is fixed in the lock (`split.held_set_sha256`) before any stream exists.
   - The checker computes strata S0-S4 and determinacy. Counted items are determined items in S2+.
   - Each counted stratum needs at least the minimum number of distinct items.
3. Arms by claim level (CONSTRAINTS.required_arms_by_level):
   - E1: candidate, shuf.
   - E2: + positive_control.
   - E3: candidate, no_prior_teaching, shuf, oracle_plumbing.

   Definitions:
   - no_prior_teaching: matched sensory input and timing; the teaching episode claimed to be reused is withheld; teaching_withheld is declared.
   - SHUF: orbit-consistent label permutation agreeing with the truth at about chance. Never a derangement on binary labels.
   - Ceiling and floor learners are evaluator-only and drawn from the CONSTRAINTS grids.
4. The decision rule (references/RULE_GRAMMAR.md).
   - Declarative only. No delivery decision code.
   - Comparator = max(chance, comparator arms, shortcut ladder for the pool).
   - Margins and floors are at least the CONSTRAINTS floors.
   - Aggregators: paired_lcb, k_of_n or all_worlds.
   - A per-world floor and min_worlds apply.
   - A custom reducer needs a UD, an author other than the executor, and property fixtures.
5. OLD protection: guard old_drop (own OLD_END minus own FINAL) plus guard old_acq (OLD_END above the exposure-independent ladder by a margin). This blocks 'protect OLD by never learning OLD'.
6. Cycle 0 testbed screen.
   - admit.py writes CEILING_<cfg> and FLOOR_<cfg> rows on the exact split.
   - The checker scores them with the delivery's own rule.
   - At least one ceiling must PASS, and every floor must FAIL. The floor is the best over the grid.
   - Otherwise CANNOT_TEST_CLAIM_AS_SPECIFIED.
7. Runner contract (references/ADMISSION_AND_RUNNER_CONTRACT.md):
   - OUTPUTS rows carry stream_hex up to gen_cap_bytes, ignoring EOS.
   - TAUGHT manifests are written per arm.
   - Receipts carry per-channel digests at birth, post_first and end, plus write counters incremented at the state-mutation site (`write_counter_site`).
   - `--only-world/--only-arm` flags support replay.
   - Rows are written before any assertion runs.
   - All counted runs go through `launch`.
8. Behavioral admission (the ADM-* list in Section 7.4).
   - Structure meters (shared-operand vs non-shared code similarity, modal fraction, code-overlap drift) are always reported next to distinguishability.
   - Admission can only stop a run. It never supports a claim, because a hash code passes distinguishability.
9. Causal reuse for E3 promotion. Promotion (SEALED_CONFIRM headline, adoption, pointer) requires UDI LCB > 0 from leave-units-out variant worlds (Section 9.7). Without it, the claim is E3 with reuse_specificity not_tested.
10. E3 cadence and the component bridge.
    - After 2 or more deliveries that did not open held E3 (computed), the next lock must open it or carry a waiver.
    - Any E0/E1 work cites its predicted effect on held E3 in predictions.
11. Lineage and purity.
    - lineage.parent is declared and the checker computes the family.
    - The native-closure purity scan applies.
    - answer_path must be native for any counted item.
    - priors_added appear in the headline.
    - More than one declared factor change labels the claim integrated_intervention.
12. OC for confirmation: `MFC oc <dir>`. P(pass | target) must be at least 0.8, otherwise the run cannot be SEALED_CONFIRM.

References:
- RULE_GRAMMAR.md;
- ADMISSION_AND_RUNNER_CONTRACT.md;
- STRATA_WITNESS_UDI.md, with worked addition examples.

Covers FM-02..07, 11, 12, 16, 17, 21, 23, 30 (learner level).

### 6.3 minifly-claims-knowledge

Frontmatter description (948 chars): "Use whenever you write or change anything another agent or Mr. Zhou will read as a MiniFly result, status or lesson: REPORT.md, CLAIM.json, STATUS or SUMMARY files, PROJECT_DOCUMENTS/CURRENT_*.json pointers, catalog or memo rows, or a next-design memo that quotes earlier numbers; when proposing, activating, narrowing or citing a finding in PROJECT_DOCUMENTS/KB/FINDINGS.jsonl; when scoring registered predictions; when filing or closing an incident; when correcting a published verdict; and when exploring what is already known before proposing a design. Enforces the generated headline and closed claim vocabulary, confirm wording only on sealed worlds, pointers as copies of CLAIM.json, findings proposed by the author but activated only by the other agent, errata for changed verdicts, the incident-to-check rule and what never counts as progress. Commands: minifly_check.py claim, gate --stage publish|pointer, ledger append, kb status|score."

Body sections:
1. Explore with the KB.
   - Start from `MFC kb status`.
   - Cite F-ids for every number about prior work.
   - Quote narrowed facts only with their caveat. Example: P036 +22.10pp must appear with FIXED_1.
   - Find the nearest tested delivery by task family and lineage.
   - Use Laplace base rates from DELIVERY_LEDGER.
   - UNVERIFIABLE_IN_REPO facts cannot be the sole basis of a design.
2. Generate the claim.
   - `MFC claim <dir>` writes CLAIM.json and the headline (Section 9.9).
   - The first non-empty line of REPORT.md equals the headline.
   - The prose below it is free.
3. Vocabulary and level rules (CLM-01..09), including the soft-word list and native/core/inherited wording.
4. Publish gate and the T2 verification subset (CLM-15).
5. Pointers.
   - Written only as minifly.pointer.v1 copies of CLAIM fields.
   - The previous pointer is snapshotted into `<TRACK>_POINTER_HISTORY/`.
   - Narrowing rows are cited.
   - A pointer to a legacy delivery needs an owner UD and a RETRO annotation (PTR-07).
6. Findings lifecycle (references/KNOWLEDGE_LEDGER.md).
   - The author proposes 0-3 findings (with a reason when 0).
   - The other agent activates them.
   - Anyone may narrow or retract by appending a row.
   - R-levels are computed.
   - Nothing is deleted.
7. Prediction scoring.
   - Primary and guard predictions only, scored at publish (hit, interval score, Brier).
   - An unopened endpoint is scored unresolved.
   - Transfer rows are added to the findings a prediction cited.
   - Scores are diagnostic only.
8. Incidents. File them with caught_at_discovery and the checker sha. They close only with a check id plus a corpus fixture, through governance.
9. Errata.
   - Any change to a published primary_gate needs an ERRATA row classified BUG_FIX or POST_HOC.
   - POST_HOC can never upgrade a verdict.
   - A BUG_FIX upgrade needs a fixture and a reviewer.
10. What never counts as progress: PASS counts, gates added, cycles, flags. Only the sealed-world G series counts.

References:
- CLAIM_AND_POINTERS.md: headline grammar, enums, banned words, pointer procedure.
- KNOWLEDGE_LEDGER.md: row kinds, fold rules, R-levels, scoring formulas.

Covers FM-07, 22, 24, 25, 26, 29, plus parts of 09 and 23.

### 6.4 minifly-independent-review

Frontmatter description (856 chars): "Use when you review a MiniFly delivery you did not author (required once per tier-2 delivery, after ADMISSION.json exists and before roster spend, plus a lock acknowledgement before the official run, seal duties and a post-run replay for sealed confirmations and adoptions); when you receive a review and must reconcile it; when you find or suspect a defect, shortcut or reward-hacking route in any delivery, old or new; and when the other agent is unavailable and review debt must be recorded. Covers executed checks (rerun admit.py, feed an input that must be rejected, trace a headline number, cheapest-shortcut attempt, replay), the reviewer-seeded canary run, REVIEW.json, the seal nonce, reconciliation statuses with the V51 legacy map, counter-signing refuted blockers, REVIEW_DEBT, and why a same-family subagent review never counts as independent."

Body sections:
1. Preconditions. You belong to the other family, which is checked by REV-01 (trailer plus namespace). A same-family subagent is debt.
2. Timing.
   - One review event after ADMISSION and before the first dev output.
   - A SOURCE_LOCK ack before official (a short delta review of the closure diff).
   - Seal steps for SEALED_CONFIRM.
   - A post-run replay for SEALED_CONFIRM or adoption.
3. Minimum executed checks. Scripts are committed under `review/` with repo-relative paths, CPU caps and output sha256.
   - rerun_admission: rerun at least one technical world and compare digests.
   - reject_input: feed a gate or fixture an input it must reject.
   - trace_number: trace one headline or admission number to its producing line and coefficients.
   - shortcut_attempt: the cheapest non-learning policy that could pass the locked rule. Success means BLOCK.
   - oracle_upstream: required when a stage chain exists.
4. Canary: `MFC canary <dir> --seed <hex chosen after reading the lock>`. All applicable mutants must be caught. A survivor becomes an incident.
5. Judgment questions. Answer one line per FM class triggered by the design, from FAILURE_CLASSES.json. In particular:
   - Does the write counter sit at the mutation site?
   - Does the control cut only its declared path?
   - Are any priors task-shaped?
   - Are there supplied-arithmetic hits (LOCK-07 REVIEW_REQUIRED)?
6. Findings and reconciliation.
   - Severity: BLOCKER, MAJOR, MINOR or NOTE.
   - A finding without an executed check is UNVERIFIED and carries no weight. An UNVERIFIED BLOCKER still pauses the stage until it is executed.
   - Status: CONFIRMED, CONFIRMED_WITH_CORRECTION, REFUTED, NEW_DEFECT or SCOPE_LIMITED.
   - V51 legacy map:
     - CONFIRMED_WITH_QUANTITATIVE_CORRECTION, RISK_CONFIRMED_UNDER_ACTUAL_NATIVE_MASKS and MATCHED_EVENT_SENSITIVITY_SUPPORTS_CALIBRATION_FRAME_DEFECT become CONFIRMED_WITH_CORRECTION.
     - COUNTERFACTUAL_FAILURE_REPRODUCED becomes CONFIRMED.
     - SPECIFICATION_IMPLEMENTATION_MISMATCH and LATENT_INTEGRATION_DEFECT become NEW_DEFECT.
     - CLAUDE_HISTORY_STATEMENT_CORRECTED becomes REFUTED.
   - The author reconciles every finding before launch. A REFUTED BLOCKER needs the reviewer's counter-sign or an owner ruling.
7. Seal duties (references/SEAL_PROTOCOL.md).
8. Post-run replay: `MFC review replay-plan <dir> --seed HEX --n 4`. Replay the chosen (world, arm) histories with the locked command and record output digests.
9. Review debt: when it is allowed, what it blocks, and that it is repaid first when the other family returns.
10. Reviewing history. Legacy deliveries get only RETRO annotations and INCIDENTS rows; they are never edited.
11. The delivering agent's lessons are inputs. List what they omit; for example, LEARNED-BANK's notes omit the fail-open summarize.py, the lock-closure gap and the task's distance from E3.

References:
- REVIEW_PROTOCOL.md: REVIEW.json walk-through, executed-check recipes, canary.
- SEAL_PROTOCOL.md: step list and failure handling.

Covers FM-01, 03, 12, 15, 23, 25, 27, 30.

### 6.5 minifly-governance

Frontmatter description (936 chars): "Use whenever anyone proposes to create, edit, relax, remove or re-tier anything protected: a MiniFly skill (.claude/skills/minifly-*), anything under tools/minifly_check (checks, RULES.json, corpus, EXPECTED.json), AGENTS.md / PROJECT_DOCUMENTS/KB/AGENTS_POINTER.md, .gitattributes, PROJECT_DOCUMENTS/KB/CONSTRAINTS.json or USER_DECISIONS.jsonl, including a small fix to a check that is blocking your own delivery; when a check gave a false positive or missed a defect; when recording an owner decision that changes constraints, waives a rule or loosens a check; and at each census or every five v1 deliveries, to measure whether the loop itself improves. Defines protected paths and invariants, computed change classes, owner decisions with restated numbers, cross-agent review for tightenings, separation from delivery commits, the conflict-of-interest rule, the advisory window, sealed held-out canaries, held-out incident recall, prune reviews and the only progress metric."

Body sections:
1. Protected paths (PROTECTED.json) and invariants I1-I12.
2. Change classes, computed by `MFC governance --base B --head H` (Section 13.2), and the requirements for each class.
3. Process:
   - Open a proposal with the diff and a rationale that cites an incident id, a false-positive report or a measured cost.
   - Run `governance`.
   - Tighten or neutral: a cross-agent review record in `KB/GOVERNANCE/REVIEW_<id>.json`.
   - Loosen or constraint change: an owner UD that restates the changed numbers.
   - Merge as a separate commit range with trailers `User-Decision:` and/or `Governance-Review:`.
4. Owner decisions:
   - verbatim text, session and scope;
   - for loosening, every changed number appears in the verbatim text;
   - the agent transcribes;
   - the owner spot-checks at census.
5. Conflict of interest (GOV-COI) and prospective application:
   - `introduced_in`;
   - advisory for 2 v1 deliveries at stages below T2 official;
   - T2 official, publish and pointer stages always run the latest enforced and advisory checks.
6. The incident -> check loop.
7. Held-out evaluation of the checker (Section 12.3). The archive digest is training data.
8. Census, every 5 v1 deliveries or monthly:
   - re-run gates on recent v1 deliveries (no PASS flips without a reason);
   - held-out recall;
   - prune review, using measured gate runtime and catch counts only;
   - KB diagnostics;
   - owner spot-check.
9. Meta-Goodhart rules:
   - never optimize prompts or skills against flag counts;
   - never cite pass counts as progress;
   - a canary fix must not string-match canary literals;
   - the delivering agent's lessons are inputs to independent review.

References:
- CHANGE_CLASSES.md: classification table and direction map.
- HELDOUT_CANARIES.md: batch authoring, sealing, reveal, scoring, burn.

Covers FM-15, 26 and 30, plus parts of 27 and 28.

---

## 7. The tool: tools/minifly_check

### 7.1 CLI and exit codes

Global options:
- `--repo-root PATH`: default is the git toplevel of the current directory.
- `--git timeline:<fixture_dir>`: selftest only.
- `--json`: print the report to stdout.

```
MFC init <dir> [--parent <dir>] [--claim-level E1|E2|E3] [--task-family F]
MFC lock build <dir>                                   # writes SOURCE_LOCK.json
MFC gate --stage {lock|roster|official|publish|pointer} [--dry] [--pointer PATH] <dir>
MFC launch --stage {testbed|technical|dev|rehearsal|official} <dir> -- <cmd ...>
MFC cycle close <dir> --kind {testbed|technical|dev|rehearsal} --decision {continue|revise|stop|escalate} --reason TEXT
MFC score <dir> --role {admission|dev|rehearsal|official}
MFC oc <dir>
MFC seal {commit-nonce|reveal|derive|verify} <dir> [...]
MFC canary <dir> --seed HEX [--n 20] [--out DIR]
MFC review replay-plan <dir> --seed HEX [--n 4]
MFC claim <dir>
MFC lint <files...>                                    # T0 text/pointer lint
MFC ledger append --ledger {FINDINGS|INCIDENTS|USER_DECISIONS|REVIEW_DEBT|CANARY_COMMITMENTS} --row ROW.json
MFC kb {status|score|backtest-snapshot|backtest-score} [...]
MFC selftest [--mutate] [--only ID]
MFC governance --base REV --head REV
MFC heldout {seal|reveal|score} [...]
MFC retro (--all | <dir>) --out DIR
```

Exit codes:
- 0: PASS.
- 1: FAIL (any enforced check FAIL).
- 2: MISSING_OR_INVALID_INPUT (any MISSING, and no FAIL).
- 3: CHECKER_ERROR (exception in the dispatcher, SELF-01 failure, unknown subcommand).

Every nonzero exit blocks. There is no `--force` and no `--skip`. `--dry` writes `checks/DRY_<stage>_<utc>.json`, which counts only for CYC-02.

Every gate writes `checks/GATE_<stage>_<utc>.json` (schema `minifly.gate_report.v1`). The file records checker_sha256, checker_version, git_head, ENDPOINT_LOCK and SOURCE_LOCK shas, tier and its reasons, and family_id. results[] holds rows of {id, status, detail, evidence[]}, and the file ends with an overall verdict.

Statuses: PASS, FAIL, MISSING, WARN, ADVISORY_FAIL and NOT_APPLICABLE. NOT_APPLICABLE is used only when RULES.json declares the condition (for example SEAL-* when worlds_role is not SEALED_CONFIRM).

### 7.2 Modules and ownership

| Module | Responsibility | WP |
|---|---|---|
| minifly_check.py, common.py, launch.py | Dispatcher, Ctx, registry, GitView, KB view, hashing, bundle reader, tier/family, launch, cycles, SELF/SCHEMA/TIER/RUN/CYC | WP1 |
| lockcheck.py, seal.py | init, lock build, LOCK-*, AMD-*, WIT-01, SEAL-*, CONF-01 | WP2 |
| addition.py, scorer.py, decide.py, reducercheck.py | Task semantics, canonical metric, ladder, strata, witness, G, OLD, UDI, INERT, decision rule, OC, RED, DEC | WP3 |
| admitcheck.py, review.py, canary.py | ADM-*, AUT-*, REV-*, DEBT-*, CAN-*, replay-plan | WP4 |
| claimcheck.py, kb.py | claim generation, CLM-*, ERR-01, WLD-*, PTR-*, lint, ledgers, KB-*, cadence, kb commands | WP5 |
| governance.py, retro.py | GOV-*, selftest, heldout, retro | WP6 |

Implementation constraints:
- Python 3.9 or later, stdlib only (json, hashlib, hmac, ast, zipfile, gzip, subprocess, resource, statistics, math, random, secrets, tempfile, pathlib, re, argparse, dataclasses).
- Startup imports modules lazily. If a module is missing or fails to import, every stage that needs it returns CHECKER_ERROR, which fails closed.
- `minifly_check.py` inserts its own directory into sys.path, so `python3 -I` works.
- Git is called read-only through subprocess, with fixed argv and a 30 s timeout. Allowed commands: rev-parse, log, show, cat-file, ls-files, diff, merge-base --is-ancestor, branch -r --contains, archive (governance only, into a temp dir).
- A shallow repo makes git-dependent checks MISSING.

### 7.3 Shared interfaces (common.py; normative for all WPs)

```python
CHECKER_VERSION = '1.0.0'
EXIT_PASS, EXIT_FAIL, EXIT_MISSING, EXIT_ERROR = 0, 1, 2, 3
class MissingInput(Exception)      # -> status MISSING
class InvalidInput(Exception)      # -> status FAIL with reason
@dataclass class Result: id; status; detail=''; evidence=[]; data={}
CHECKS: dict  # id -> fn(ctx) -> Result | list[Result]
def check(check_id)                # decorator; duplicate id raises
@dataclass class Commit: sha; utc; author; trailers: dict[str, list[str]]; paths: list[str]
class GitView:                     # implemented by RealGit(repo_root) and TimelineGit(fixture_dir)
  head(); is_shallow(); log(paths=None) -> list[Commit] (oldest first)
  first_commit_adding(glob) -> Commit|None; first_commit_where(path, pred_bytes) -> Commit|None
  is_ancestor(a, b) -> bool; file_at(sha, path) -> bytes|None
  remote_branches_containing(sha) -> list[str]
  diff_lines(base, head, path) -> (added, removed); changed_paths(base, head) -> list[str]
  worktree_clean(paths) -> bool
class KB(repo_root):  constraints(); constraints_sha256(); rows(name); failure_classes();
  user_decision(id); finding_status(id)   # rows() dedups identical lines (union merge), raises InvalidInput on duplicate ids with different content
@dataclass class Ctx: repo_root; delivery; git; kb; rules; stage; tier; dry; options
  lock (ENDPOINT_LOCK, cached, MissingInput if absent); source_lock; read_json(rel); read_jsonl(rel)
  iter_rows(pattern)   # *.jsonl, *.jsonl.gz, and members of RESULT_BUNDLE*.zip matching pattern
def canonical_json(obj) -> bytes   # sort_keys, separators (',', ':'), UTF-8, no NaN (NaN -> InvalidInput)
def sha256_bytes/sha256_file/sha256_json; def utc_now() -> 'YYYYMMDDTHHMMSSZ'
def compute_tier(ctx) -> (int, list[str]); def family_of(ctx) -> (family_id, members)
def run_bundle(ctx, stage) -> list[Result]; def write_gate_report(ctx, stage, results) -> Path
def exit_code(results) -> int; def is_v1_delivery(path) -> bool
```

Cross-module functions (lazy imports; a missing function raises MissingInput):
- `kb.deliveries_since_last_e3(kb) -> int`
- `scorer.score_rows(rows, lock, taught_by_world_arm) -> cells`
- `scorer.score(ctx, role) -> dict`
- `decide.evaluate(cells, lock, preconditions) -> dict`
- `decide.oc(lock, dev_summary, sims=2000, seed=0) -> dict`
- `addition.parse/expected/canonical/stratum/determined`

TimelineGit fixture format (also used by the corpus):
- `timeline.json` = {head, commits: [{sha, utc, author, trailers, paths, branches}]}. History is linear, so is_ancestor(a, b) holds iff index(a) <= index(b). branches lists the remote branches containing each commit.
- `repo/` holds the final tree.
- `history/<sha>/<path>` holds the content of a path at the commit that last touched it, whenever that differs from `repo/`.
- file_at(c, p) uses the latest commit at or before c that touched p: history/<that sha>/p if present, else repo/p if that commit is the final toucher of p, else None.

### 7.4 Check registry (RULES.json is authoritative)

Each RULES.json entry is {id, module, stages[], tier_min, introduced_in, status: enforced|advisory, fm_ids[], applies_when}. selftest asserts a one-to-one match between RULES ids and registered functions.

**WP1 (common, launch)**
- SELF-01: Every tracked file under tools/minifly_check equals its HEAD blob (clean worktree). selftest has passed for this checker sha; the result is cached in the system temp dir under the sha, and if absent selftest runs first. On failure: exit 3.
- SCHEMA-01: The delivery is v1. Otherwise exit 2 with 'legacy delivery: use retro'.
- TIER-01: Computes tier and its reasons (Section 4). Informational; drives the bundles.
- RUN-01 (T2 enforced, T1 WARN): Every file under results/{admission,dev,rehearsal,official}/ and results/receipts/ appears, with a matching sha256, in some ATTEMPTS row's produced_paths. That attempt ran from a clean closure.
- RUN-02: At most one official attempt per SOURCE_LOCK produced assessment-bearing rows. A later official attempt is legal only if every earlier one produced zero OUTPUTS rows.
- CYC-01: Closed test cycles are at most caps.max_cycles.
- CYC-02: Each closed cycle with decision continue met its kind's exit criterion.
- CYC-03: Rehearsal world ids appear in no earlier dev or technical row, receipt or attempt argv. The rehearsal argv template equals the official one with world arguments removed.
- CYC-04 (publish; introduced_in 1.0.1, SPEC 20.16): runlog/ATTEMPTS.jsonl and runlog/CYCLES.jsonl are append-only at publish: every committed row survives unchanged in every later commit and in the worktree.

**WP2 (lockcheck, seal)**
- LOCK-01: Schema; required fields non-null; no '__FILL__' values; enums valid.
- LOCK-02: Contract block complete.
  - claim_level_target and intent are set.
  - exposure_streams are declared.
  - causal_control_arm and positive_control_arm exist.
  - The arms required for the claim level are present by role.
  - Arms with role lesion or no_prior_teaching have non-empty zero_channels.
  - no_prior_teaching declares teaching_withheld.
  - A units block is present for E2 and above.
- LOCK-03: primary.metric is exact_own_eos or balanced_exact_own_eos, with unconditional true and output own_bytes_own_eos; then gate_type is objective. Anything else is forced to component.
- LOCK-04: Every threshold meets its CONSTRAINTS.threshold_floors value: margin_abs, absolute floor above chance, per_world_floor, min_worlds (unless PILOT), admission thresholds, old_max_drop and old_acq_margin, alpha. Guards old_drop and old_acq exist whenever an OLD pool exists.
- LOCK-05 (technical and later): SOURCE_LOCK.closure equals a fresh `lock build`. The static AST closure of files.native, files.runner and files.admit is complete. Literal `sys.path.insert/append` targets are resolved. Non-literal importlib or `__import__` calls must be listed in dynamic_imports. Every resolved repo module lies inside the delivery dir, tools/ or files.shared; anything else means a collision and FAIL.
- LOCK-06: Timing.
  - The first commit containing ENDPOINT_LOCK.json precedes the first commit adding results/**.
  - For each native_closure_sha found in receipts or OUTPUTS, the commit whose SOURCE_LOCK has that closure precedes the first commit adding rows with it.
  - Lock and results never share a commit.
- LOCK-07 (technical and later): Native purity, scanned over the closure of files.native and files.runner (evaluator files excluded).
  - CONSTRAINTS.native_forbidden imports, calls or identifiers: FAIL.
  - A files.evaluator module reachable from native or runner: FAIL.
  - supplied_arithmetic_heuristics hits: REVIEW_REQUIRED, which is WARN at T1. At T2 they must be addressed by a REVIEW finding.
- LOCK-08: Any delivery .py that assigns a literal True, False, 0 or 0.0 to a name or key matching `(_ok|_passed|_passes|overlap|_verified|_independent)$` or `^(no_|all_)` fails.
- LOCK-09: E3 cadence. d = kb.deliveries_since_last_e3. If d is at least CONSTRAINTS.e3_cadence_max and the lock is not E3-opening (primary pool HELD, counted strata within S2..S4, claim_level_target E3, official planned with MAIN or SEALED_CONFIRM), a UD with scope `e3_waiver:<delivery>` is required.
- LOCK-10: constraints_sha256 equals KB/CONSTRAINTS.json. Governed fields (io, warmup, native) equal the constraints or cite a UD.
- LOCK-11: Predictions.
  - At least one on the primary (p_pass and/or point plus interval90), and one per guard.
  - base_rate_forecast present.
  - p within [0.02, 0.98].
  - basis_findings exist in FINDINGS, or are 'none'.
- LOCK-12: An analytic input SNR is required when there is no parent or lineage.changes_declared includes frontend, read or init. It must be at least the floor, or 'not_derivable', in which case ADM-INPUT becomes mandatory.
- LOCK-13: Lineage. If another v1 delivery has native-closure Jaccard of at least 0.5 and is neither the declared parent nor its ancestor, FAIL 'undeclared parent'. The family_id is reported.
- AMD-01: Every distinct committed ENDPOINT_LOCK content after the first is linked by an AMENDMENTS row (old sha to new sha).
- AMD-02: An amendment needs a UD (scope `amendment:<delivery>`), the original endpoint in coprimary[] and a bumped protocol_id when either of these holds:
  - it changes primary.*, guards, gates, arms, units, split or task after any learner-arm outcome;
  - it raises objective_distance on any axis.
- AMD-03: reason is one of ceiling, testbed_invalid, measurement_defect or operational. testbed_invalid requires a TESTBED result on the original endpoint showing ceiling FAIL or floor PASS. results_seen must equal the computed list of outcome files committed before the amendment.
- AMD-04: Official world ids, listed or seal-derived, appear in no outcome artifact committed before the latest amendment, and in no other delivery.
- AMD-05: A change to any file in ENDPOINT_LOCK.endpoint_files_sha256 after the lock, without an amendment, fails.
- WIT-01 (T2): External witness.
  - REVIEW.reviewed_endpoint_lock_sha256 equals the current ENDPOINT_LOCK.
  - The commit introducing that attestation is contained in a reviewer-namespace remote branch and is an ancestor of the first dev output commit.
  - For official, review.source_lock_acks contains the final SOURCE_LOCK sha, in a reviewer-namespace commit that is an ancestor of the first official output commit.
- SEAL-01..05 and CONF-01: see Section 10.

**WP3 (addition, scorer, decide, reducercheck)**
- MET-01: Recompute from stream_hex: emitted is the bytes before the first eos within gen_cap; own_eos means an eos was found. The runner's emitted_hex and own_eos must match exactly (two-source agreement). Rows with answer_path other than native never count.
- MET-02: For addition formats, expected_hex is recomputed from prompt_hex. Any mismatch fails (one truth semantics).
- LAD-01: Every (world, arm, pool, checkpoint) cell implied by the lock exists with exactly n_items_per_pool items. Denominators come from the lock.
- LAD-02: Substitution. decide must not PASS when the candidate's rows are replaced by the best shortcut's predictions.
- LAD-03 (WARN): A declared comparator arm below the best shortcut on dev is noted in the headline as WEAK_COMPARATOR.
- LAD-04 (WARN): Modal output fraction above 0.9, or entropy below 0.5 bit, in any pool.
- STR-01: Strata are computed from content. The counted strata are valid for the level (E3 counts only S2..S4). Each counted stratum has at least CONSTRAINTS.min_items_per_stratum items.
- DET-01: Determinacy is computed under units.witness. Counted items are the determined ones. If the determined counted items are fewer than the minimum, FAIL: the testbed cannot test E3.
- INERT-01: INVALID with reason INTEGRITY_ALARM when either holds:
  - two arms declared different have identical stream_hex on every item in every world;
  - the primary contrast has zero between-world variance with identical values (the [0,0] interval).
- UDI-01 (E3 promotion, or a udi block declared): The variants are valid under the witness. Dependent items become undetermined after removal, independent items remain determined, and TAUGHT(variant) = TAUGHT(base) minus the removed set. UDI is computed with a world-level LCB.
- OC-01 (official, when SEALED_CONFIRM or adoption is intended): simulated P(pass | target) must be at least CONSTRAINTS.oc_min_power. Otherwise worlds_role is capped at MAIN.
- DEC-01: VERDICT.json was produced by the program decide, with decide_version, both lock shas and the cells sha. rule_type custom is allowed only with a UD, a REVIEW whose author is not the executor, and 50 reviewer-seeded property fixtures that pass.
- RED-01..11: rule qualification on synthetic cells built from the lock's own arms, worlds, pools and strata.
  - RED-01: a clean positive (candidate 0.9, all else at chance) must PASS. This proves the rule is reachable.
  - RED-02: candidate equal to the strongest comparator must not PASS.
  - RED-03: a missing comparator arm gives INCOMPLETE.
  - RED-04: fewer than min_worlds must not PASS.
  - RED-05: None or NaN gives INVALID.
  - RED-06: all-dev worlds give DEV_NOT_EVIDENCE.
  - RED-07: candidate at comparator + margin - 1e-9 must not PASS.
  - RED-08: a gain only on S0/S1 must not PASS.
  - RED-09: an OLD guard failure must not PASS.
  - RED-10: the shortcut substituted for the candidate must not PASS.
  - RED-11: OLD_END at shortcut level with zero drop must not PASS.

**WP4 (admitcheck, review, canary)**
- ADM-01: ADMISSION.json schema, both lock shas, and producer admit.py in endpoint_files. Worlds are technical only. Committed before the first dev output.
- ADM-TB-CEIL: The checker scores CEILING_* rows with the delivery's rule. At least one config must PASS. Grid coverage must include CONSTRAINTS.testbed_ceiling_grid. Failure gives CANNOT_TEST_CLAIM_AS_SPECIFIED.
- ADM-TB-FLOOR: Every FLOOR_* config, and the best over the CONSTRAINTS.testbed_floor_grid, must FAIL the rule. Failure gives CANNOT_TEST.
- ADM-CTRL-ZERO: For each arm with zero_channels Z, read from raw receipts:
  - writes_by_channel[c] == 0 for every c in Z;
  - channel_digests.post_first[c] == channel_digests.birth[c];
  - in W, writes on each c are above 0 and post_first[c] != birth[c];
  - write_counter_site names a function present in the native closure.

  Failure gives INVALID with reason TREATMENT_NOT_DELIVERED.
- ADM-CTRL-ISO: At post_first, the arm's digests equal W's on every channel that is neither in Z nor in declared_changed. This is the confound check.
- ADM-CTRL-DOSE: realized_dose is at least max(min_realized_dose, the floor).
- ADM-DIR: For every port and sign in FIXTURES.jsonl:
  - target_margin_delta_vs_passive > 0;
  - zero_coef_digest_equals_passive is true;
  - zero_replay_applications == 0.
- ADM-FIX: Fixture params equal the params.json values hashed in SOURCE_LOCK, within the declared tolerances. A production key missing from a fixture is FAIL.
- ADM-GATE: Every plumbing gate in lock.gates has at least one negative fixture with result false and one positive fixture with result true.
- ADM-INPUT: input_contrast.ratio is at least the minimum, and prompt_id_loo_acc is at least chance plus the margin. The structure meters must be present; they are reported, not gated.
- ADM-ACQ: Computed by the checker from admission rows: candidate exact own-EOS on TAUGHT at the prefix checkpoint is at least the best exposure-independent shortcut plus taught_acq_margin. Failure gives NOT_ACQUIRING.
- ADM-SHUF: Computed by the checker from the SHUF arm's TAUGHT manifest against the true answers. Agreement lies within shuf_band, and labels are orbit-consistent.
- ADM-EXPO: Canonical held orbits (exact prompts for E2) have an empty intersection with every feedback-bearing stream. Any TAUGHT or stream file under manifests/ or results/ that is not declared in exposure_streams fails. As amended in 20.11 (changed_in 1.0.1): no row of any exposure stream, feedback-bearing or not, carries a held item's expected answer together with its prompt or orbit, judged by content in every field.
- ADM-BUDGET: lock.budget is within the AUTHORIZATION caps, and schedule.paired_atomic is true.
- ADM-REUSE (T1 only): If the native closure, task and split equal the parent's and the parent's admission passed, the parent ADMISSION sha may be cited to skip ADM-INPUT and LOCK-12.
- AUT-01: AUTHORIZATION.json fields present: verbatim, utc, session, options_shown, recommendation_shown and stages.
- AUT-02: Each cap number appears literally in the verbatim text, or the default caps apply. The stage requested is covered.
- AUT-03: No option label contains '(Recommended)'. Any recommendation is a separate field.
- AUT-04: Every conditional has a predicate, which the checker evaluates (predicate grammar: artifact path, JSON pointer, op, value), or evaluated_by names the non-beneficiary agent together with the result.
- REV-01: reviewer family differs from the author family (lock.author_agent plus trailers). The commit introducing REVIEW.json has an `Acting-Agent:` trailer of the reviewer family and is contained in a remote branch of the reviewer namespace.
- REV-02: executed_checks include the kinds rerun_admission, reject_input, trace_number and shortcut_attempt, plus oracle_upstream when lock.stage_chain is true. Each check has a committed script, cmd, output, matching sha256 values and cpu_cap_s.
- REV-03: A rerun_admission whose match is false has a corresponding finding.
- REV-04: shortcut_attempt is recorded. passed_locked_rule true forces decision BLOCK.
- REV-05: canary.detected == canary.n. The CANARY report commit follows the ENDPOINT_LOCK commit.
- REV-06: REVIEW is committed before the first dev output. decision is ADMIT. No open BLOCKER or MAJOR finding remains. A REFUTED BLOCKER is countersigned_by_reviewer or has a ruling_user_decision.
- REV-07: No executed-check output is byte-identical to an output in another delivery's review/ (padding).
- REV-08 (SEALED_CONFIRM or adoption): postrun.replay digests equal the digests of the official rows for the same (world, arm).
- REV-09 (T2 publish; introduced_in 1.0.1, SPEC 20.20): REVIEW.fm22_prose_adjudications gives a verdict (ok | overclaim, with a note) for every sentence CLM-17P flags, now or in the latest publish/dry GATE report while the sentence is still in the text; the field is reviewer-owned; an overclaim verdict blocks publish until the sentence is gone.
- DEBT-01 (T2 roster without a valid REVIEW): an open REVIEW_DEBT row for this delivery exists, and open debts program-wide are below review_debt_max_open.
- DEBT-02: A delivery with open debt cannot be SEALED_CONFIRM, cannot have findings activated, and fails the pointer stage.
- CAN-01: The unmutated copy reproduces the original statuses.
- CAN-02: Every applicable mutant is caught.
- CAN-03: At least 50% of operators are applicable.

**WP5 (claimcheck, kb)**
- CLM-01: CLAIM.json schema and enums.
- CLM-02: Every CLAIM field except findings_proposed and notes equals a fresh regeneration, including the headline.
- CLM-03: Level rules.
  - E2 or higher requires PASS with gate_type objective.
  - E3 requires pool HELD, counted strata within S2..S4, a no_prior_teaching comparator, ADM-EXPO PASS, and headroom over the best shortcut of at least margin_abs.
  - E3 promotion additionally requires UDI LCB > 0 and SEALED_CONFIRM.
  - More than one declared factor change sets the label integrated_intervention.
- CLM-04: confirm* tokens anywhere in REPORT, STATUS, SUMMARY or the headline require SEALED_CONFIRM with SEAL-* and WLD-01 PASS.
- CLM-05: The first non-empty line of REPORT.md equals CLAIM.headline.
- CLM-06: Soft words (largely, mostly, in part, partial, partially, nearly, promising, essentially, approved, met in part, substantially, broadly, close to, encouraging) are FAIL in the headline sentence and in any sentence containing a gate id or the primary_gate token; WARN elsewhere.
- CLM-07: native, core, inherited or 'same fly' require lineage_parent with inheritance demonstrated. WARN at T1, FAIL at T2.
- CLM-08: Every file added after the official lock is listed as POSTHOC and is never among VERDICT inputs.
- CLM-09: worlds_role is computed. Fewer than min_worlds gives PILOT. Dev-touched worlds give DEV. MAIN requires the official stage and fresh worlds. SEALED_CONFIRM requires SEAL, OC, REVIEW and replay.
- CLM-10: Every lock prediction has an entry in predictions_scored.
- CLM-11: 0-3 findings_proposed, each with conditions.best_shortcut and a source path:line. Zero requires findings_none_reason.
- CLM-12: No absolute local paths (/Users/, /home/, /tmp/, C:). Relative links resolve.
- CLM-13: A DELIVERY_LEDGER row was appended by `claim`.
- CLM-14: Each cited GATE report's checker_sha256 equals the committed checker at that report's commit.
- CLM-15 (T2): Verification subset committed: official OUTPUTS for counted pools (gz/zip allowed), TAUGHT, receipts, both locks and GATE reports. If it exceeds the size cap, use mode digest_plus_sample: per-(world, arm) digests plus a reviewer-seeded full-row sample.
- CLM-16 (WARN): Numbers about prior work in REPORT or memos cite an F-id or catalog id.
- CLM-17 (publish and lint; introduced_in 1.0.1, SPEC 20.13): agent-written text asserts no evidence level above level_earned and, unless primary_gate is PASS, never calls the rule, an endpoint, the result or a failed guard met.
- CLM-17 as amended by 20.20 (round-1 repair part 3): structural. CLAIM.notes, findings_none_reason and findings_proposed[].claim name no E-level token above level_earned in any phrasing, findings_proposed[].level <= level_earned, and the key/value lines, table cells, question/answer pairs and JSON leaves of the REPORT body and the status files carry the CLAIM level and gate or a negation.
- CLM-17P (publish and lint; advisory, WARN; introduced_in 1.0.1, SPEC 20.20): the free-prose clause heuristics of 20.13; every WARN row lists the flagged sentences for the T2 reviewer (REV-09). A lint aid, not a mechanical guarantee.
- CLM-17 and CLM-17P as amended by 20.21 (round-1 repair part 4): CLM-17 reads every string leaf of a proposed finding and judges 'status' and 'decision' values that pair a MET word with an outcome subject; CLM-17P's level side is a token rule (an E-level above level_earned in an unqualified clause is flagged whatever the verb).
- ERR-01: If CLAIM.primary_gate changes between commits after the first publish, an ERRATA row is required (BUG_FIX or POST_HOC). POST_HOC never upgrades. A BUG_FIX upgrade needs a corpus fixture id and a reviewer sign-off.
- WLD-01: Official world ids occur nowhere else in the repo (other deliveries' manifests, JSON, receipts) and in no dev row. Official outputs were first committed after the lock and the reveal.
- WLD-02 (WARN): Overlap of canonical held items with earlier deliveries' held or taught sets. Reported in the headline when greater than 0.
- PTR-01: Schema minifly.pointer.v1. The previous pointer is snapshotted in the same commit. AUTHORIZATION pointer_update is true or an adoption UD exists.
- PTR-02: level_earned, primary_gate, gate_type, worlds_role and objective_metric are byte-equal to the referenced CLAIM (claim_sha256).
- PTR-03: No bare boolean keys matching (confirm|pass|passes|passed|success|qualified) at any depth.
- PTR-04: Every quoted finding whose latest status is narrowed or superseded is cited with its narrowing row id.
- PTR-05: No absolute local paths.
- PTR-06 (WARN): A newer v1 delivery exists in the same track.
- PTR-07: A legacy target requires a UD and a KB/RETRO annotation path.
- KB-01: Ledger row schemas are valid.
- KB-02: Append-only: the git diff of each ledger shows only added lines, and ids are unique.
- KB-03: activation and reconciliation rows are authored by a family different from the finding's author. As amended in 20.15 (changed_in 1.0.1): the activator is the acting family of the commit introducing the row (trailer and namespace, common.commit_family), and the row's author_agent must equal it.
- KB-04: Activation is allowed only for findings from T2 deliveries with worlds_role MAIN or SEALED_CONFIRM and no open debt.
- KB-05: An incident closure row requires a check_id present in RULES and a fixture whose EXPECTED.must_fire includes that check.
- KB-05 as amended in 20.20 (changed_in 1.0.1): a closure row that a later closure row of the same incident names in `supersedes` (with a note), and that later row itself satisfies KB-05, is retired rather than validated. Since 20.21 'later' is checked: only a row appended after the superseded one retires it.

**WP6 (governance)**
- GOV-SELF: selftest passes at head.
- GOV-SEP: A range touching protected paths touches no delivery directory.
- GOV-CLASS: Computed class for each change (Section 13.2).
- GOV-REV: tighten, new or neutral changes require a KB/GOVERNANCE review record by the other family, cited in a `Governance-Review:` trailer.
- GOV-UD: loosen and constraint changes require a `User-Decision:` trailer whose UD scope covers the change.
- GOV-NUM: For a loosen UD, every changed numeric value (old and new) appears in the verbatim text.
- GOV-MONO: For every base fixture, fired(head) is a superset of fired(base), and clean fixtures stay clean.
- GOV-APPEND: Append-only files show only additions.
- GOV-SKILL: Each SKILL.md has valid frontmatter (name, description of 1024 characters or fewer) and a body under 400 lines. PROJECT_DOCUMENTS/KB/AGENTS_POINTER.md (and a root AGENTS.md, if one is ever tracked) lists all five skills.
- GOV-COI (official and publish stages): A protected-path commit after this delivery's lock with the author's Acting-Agent fails, unless a UD names the delivery.
- HELD-01: A heldout commitment row precedes the merge of the checker version under test.
- HELD-02: The revealed bundle and key hashes match the commitment.
- HELD-03: The batch author identity differs from the authors of tools/minifly_check commits in the interval.

### 7.5 Stage bundles (RULES.json `bundles`)

| Stage | Checks |
|---|---|
| lock (also `launch --stage testbed`) | SELF-01, SCHEMA-01, TIER-01, LOCK-01..04, LOCK-08..13, AMD-01..05, AUT-01..04 |
| technical launch | lock bundle + LOCK-05..07, CYC-01 |
| roster (also `launch --stage dev/rehearsal`) | technical bundle + ADM-*, RED-01..11, DEC-01, CYC-02. At T2 add: RUN-01, WIT-01, REV-01..07 (or DEBT-01). |
| official (also `launch --stage official`) | roster bundle + MET-01/02, LAD-01..04, STR-01, DET-01, INERT-01 on dev and rehearsal rows; CYC-03, AMD-04, RUN-02, GOV-COI, WLD-01. When intended: OC-01. For SEALED_CONFIRM: SEAL-01..05, CONF-01. |
| publish | SELF-01, SCHEMA-01, AMD-01..05, MET-01/02, LAD-01, STR-01, DET-01, INERT-01, UDI-01 (if applicable), DEC-01, CLM-01..16, ERR-01, KB-01..05, WLD-01/02. At T2 add RUN-01 and GOV-COI. REV-08 for SEALED_CONFIRM or adoption. DEBT-02. Since 1.0.1 (20.13, 20.16, 20.20): CLM-17, CLM-17P (advisory) and CYC-04, and REV-09 at T2 (RULES.json is authoritative: RUN-01..04 are in the publish bundle too, 20.7/20.8). |
| pointer | SELF-01, PTR-01..07, CLM-02, DEBT-02 |

**Advisory handling.**
- A check whose introduced_in is later than the checker_version recorded in the delivery's first SOURCE_LOCK is ADVISORY_FAIL (non-blocking) at lock, technical and roster stages.
- Official, publish and pointer stages at T2 always enforce every check in the bundle. This closes lock-banking and advisory-window timing.

### 7.6 Fail-closed rules

- Any exception inside a check gives FAIL with the traceback summary.
- MissingInput gives MISSING.
- Unknown schema or enum values give FAIL.
- NaN or None in any gate-relevant field gives FAIL.
- A git failure or a shallow clone gives MISSING.
- A missing receipt for a declared arm, a missing fixture for a declared port or sign, or a missing pool cell gives FAIL or MISSING; never a skip.
- Receipts take precedence over ADMISSION summaries, and any disagreement is FAIL.
- `gate` refuses legacy directories (exit 2) and never writes into them.
- `retro` never imports or executes historical code, and writes only to `--out`, which must lie outside every delivery dir.
- Custom reducers (rule_type custom) run only in a subprocess (`python3 -I`, 20 s timeout, synthetic or canonical cells only). A non-enum return is FAIL.

### 7.7 Self-tests

- `MFC selftest` runs every fixture in `corpus/` with TimelineGit and compares the results with EXPECTED.json:
  - must_fire must be a subset of the fired ids (FAIL or MISSING);
  - must_not_fire must not intersect the fired ids;
  - must_warn ids must have status WARN;
  - expected_exit must match per command;
  - `expect` JSON-pointer assertions on produced artifacts must hold, for example `/verdict`, `/headline`, `/objective_metric/best_shortcut/name`.

  Clean fixtures must fire nothing and exit 0. Target runtime is under 20 s.
- `selftest --mutate` replaces each registered check in turn with always-PASS and then always-FAIL. Some dirty fixture must notice the first replacement, and some clean fixture must notice the second; otherwise the result is 'check not exercised' or 'no clean coverage'.
- Unit tests live in tests/test_<module>.py for each module (`python3 -I -m unittest discover -s tools/minifly_check/tests`).

---

## 8. Artifact schemas (field-level detail in SCHEMAS.md)

**ENDPOINT_LOCK.json (minifly.endpoint_lock.v1)**, written by the executor via `init` and frozen at design. After that it changes only through AMENDMENTS.
- Identity: schema, delivery, protocol_id, created_utc, author_agent (codex|claude), tier_declared, intent (diagnostic|promotion), claim_level_target (E0..E4).
- task {id, family (addition_multi_digit|addition_single_digit|addition_component|other_arithmetic|non_arithmetic), format {kind addition|other, sep_hex, terminator_hex, eos_hex, gen_cap_bytes, normalize_leading_zeros}}.
- objective_distance {task_family, output (own_bytes_own_eos|own_bytes_evaluator_length|classification_label|margin_or_rank), held (unseen_composition|changed_presentation|taught_only), native_only_path, feedback (gradual_warmup|other), old_protection_measured}.
- units {witness pair_rectangle|column_closure, commutative bool}; split {taught_manifest, held_manifest, old_manifest, new_manifest, split_sha256, held_set_sha256}; exposure_streams [{path_glob, feedback_bearing, phase}].
- arms [{id, role candidate|no_prior_teaching|lesion|shuf|frozen|oracle_plumbing|positive_control|other, zero_channels[], declared_changed[], teaching_withheld[], min_realized_dose}]; causal_control_arm; positive_control_arm; ports [{id, signs[]}].
- primary {task_id, metric, unconditional, pool, checkpoint, strata_counted[], candidate_arm, comparator_arms[], ladder auto, chance, margin_abs, absolute_floor, per_world_floor, aggregator {type, alpha|k}, min_worlds, n_items_per_pool{}, rule_type declarative|custom}; gates []; guards [].
- udi null|{variants_manifest, alpha}; checkpoints {prefix, old_end, final}; stage_chain bool.
- worlds {technical[], dev[], rehearsal[], official {mode listed|sealed, ids[], n}, official_role_intended MAIN|SEALED_CONFIRM|PILOT}; schedule {paired_atomic}; budget {max_records, max_cpu_h}.
- admission_thresholds {analytic_input_snr|'not_derivable', input_contrast_min, prompt_id_margin_min, taught_acq_margin_min, shuf_band}; testbed {ceiling_grid[], floor_grid[]}.
- files {admit, evaluator[], generators[], runner[], native[], shared[], assets[], params}; endpoint_files_sha256 {path: sha}.
- lineage {parent '<dir>@<sha>'|null, changes_declared[], priors_added[]}.
- predictions [{id, target primary|guard:<id>, point, interval90, p_pass, basis_findings[], base_rate_forecast {value, source}}]; e3_cadence {waiver_user_decision}; constraints_sha256; user_decisions[]; coprimary[].

**SOURCE_LOCK.json (minifly.source_lock.v1)**, built by `lock build` before each cycle's runs: schema, delivery, endpoint_lock_sha256, built_utc, checker_version, closure_sha256 {path: sha}, dynamic_imports[], env {python, packages {name: version} via importlib.metadata}, native_closure_sha, runner_closure_sha, params_sha256.

**AUTHORIZATION.json (minifly.authorization.v1)**: verbatim, utc, session, channel, options_shown [{label, text, cost, objective_distance}], recommendation_shown, stages {implementation, technical, dev_cycles, official, push, push_lock, adoption, pointer_update}, caps {max_records, max_cpu_h, max_cycles}, defaults_applied, conditionals [{predicate, evaluated_by, result, utc}], recorded_by.

**AMENDMENTS.jsonl (minifly.amendment.v1)**: id, utc, old_lock_sha256, new_lock_sha256, new_protocol_id, fields_changed[], results_seen[], reason, evidence, objective_distance_delta {}, original_endpoint_kept_as_coprimary, user_decision, author_agent.

**params.json**: the learner's production parameters, free-form object. Hashed into SOURCE_LOCK.

**Receipt (minifly.receipt.v1)**, results/receipts/<world>/<arm>.json, written by the runner before any assertion: world, world_role (technical|dev|rehearsal|official), arm, phase, records, taught_presentations, writes_by_channel {}, write_counter_site, channel_digests {birth {}, post_first {}, end {}}, state_digest_end, realized_dose {metric: value}, native_closure_sha, params_sha256, endpoint_lock_sha256, attempt_id.

**OUTPUTS rows (minifly.outputs.v1)**: world, world_role, arm, pool (TAUGHT|OLD|NEW|HELD|PRESENTATION), checkpoint, item_id, prompt_hex, expected_hex, stream_hex (exactly gen_cap_bytes, generated without stopping at EOS), emitted_hex, own_eos, answer_path (native|head|evaluator), native_closure_sha, params_sha256, attempt_id. There is no stratum field; the checker computes strata.

**TAUGHT rows (minifly.taught.v1)**, manifests/TAUGHT/<world>/<arm>.jsonl: world, arm, prompt_hex, answer_hex, phase, feedback_bearing, presentations.

**FIXTURES.jsonl**: port, sign, target_margin_delta_vs_passive, zero_coef_digest_equals_passive, zero_replay_applications, params {}; plus gate fixtures {gate_id, kind negative|positive, params, result}.

**ADMISSION.json (minifly.admission.v1)**: endpoint_lock_sha256, source_lock_sha256, producer {script, sha256}, worlds[], testbed_outputs path, receipts_glob, fixtures path, input_contrast {metric_definition, analytic_snr, same_input_diff_noise, diff_input_same_noise, ratio, prompt_id_loo_acc, chance}, structure_meters {shared_operand_similarity, nonshared_operand_similarity, modal_output_fraction, code_overlap_birth, code_overlap_end}, parent_admission_sha (ADM-REUSE), created_utc. The checker computes everything it can (testbed, acquisition, SHUF, controls) from raw rows rather than trusting summaries.

**ATTEMPTS.jsonl (minifly.attempt.v1)**, written only by `launch`: attempt_id, stage, argv, utc_start, utc_end, git_head, closure_clean, endpoint_lock_sha256, source_lock_sha256, exit_code, wall_s, cpu_children_user_s, cpu_children_sys_s, max_rss_kb, produced_paths [{path, sha256}].

**CYCLES.jsonl (minifly.cycle.v1)**: k, kind, decision, reason, utc, computed {native_closure_sha, params_sha256, attempts[], latest_gate, outcomes_readable, designs_tried_k}.

**REVIEW.json (minifly.review.v1)**: delivery, reviewer_agent, reviewer_session, reviewer_branch, author_agent, reviewed_endpoint_lock_sha256, admission_sha256, utc, executed_checks [{id, kind, script, script_sha256, cmd, output, output_sha256, cpu_cap_s, result, compared_to, match}], canary {seed, n, detected, report}, cheapest_shortcut_attempt {description, script, passed_locked_rule}, fm_judgments {FM-xx: text}, findings [{id, fm_ids[], claim, severity, executed, status, reconciled_by_author {utc, note, evidence}, countersigned_by_reviewer, ruling_user_decision}], source_lock_acks [{sha, utc, note}], seal {nonce_sha256, revealed_nonce, reveal_utc}, postrun {replay [{world, arm, expected_digest, observed_digest}], utc}, decision ADMIT|BLOCK.

**SCORES.json / VERDICT.json**:
- SCORES holds cells [{world, world_role, arm, pool, checkpoint, stratum, determined, n, n_exact}], ladder per (world, pool, checkpoint), best_shortcut, strata counts, determinacy, modal fraction and entropy, G, G_per_sample, G_per_cpu_h, old {end, final, drop, acq}, udi, designs_tried_k, inert_alarms, met_mismatches.
- VERDICT holds verdict (PASS|FAIL|INVALID|INCOMPLETE|DEV_NOT_EVIDENCE|CANNOT_TEST_CLAIM_AS_SPECIFIED), invalid_reason, shadow_verdict, primary {mean_cand, mean_comp, per_world [], lcb, method}, gates {}, guards {}, decide_version, endpoint_lock_sha256, source_lock_sha256, cells_sha256.

**CLAIM.json (minifly.claim.v1)**, generated by `claim`; the agent may edit only findings_proposed, findings_none_reason and notes:
- levels and gates: delivery, protocol_id, level_attempted, level_earned (E0..E4|NONE), reuse_specificity (not_tested|udi_pass|udi_fail), primary_gate (PASS|FAIL|INVALID|INCOMPLETE|NOT_OPENED|CANNOT_TEST_CLAIM_AS_SPECIFIED), dev_shadow_verdict, invalid_reason, stop_reason, gate_type, labels [integrated_intervention|weak_comparator|priors:<x>], worlds_role (DEV|PILOT|MAIN|SEALED_CONFIRM), n_worlds;
- objective_metric {held_exact_own_eos|'not_measured', held_by_stratum, determined_fraction, old_end, old_final, old_drop, new_final, best_shortcut {name, value}, best_other_arm {name, value}, headroom, G, G_per_sample, G_per_cpu_h};
- provenance: designs_tried_k, amendments[], coprimary_results[], posthoc_files[], lineage_parent, family_id, native_path_only, compute {records_total_incl_failed, cpu_h_total_incl_failed, family_cpu_h}, verdict_source {decide_version, verdict_sha256, endpoint_lock_sha256, source_lock_sha256};
- knowledge and review: predictions_scored [{id, observed, hit, interval_score, brier}], findings_proposed[], review {path, decision}, review_debt_id, authorization {path, recommendation_shown}, confirm_attempt {family_id, index, alpha};
- checker_sha256, headline.

**Pointer (minifly.pointer.v1)**: track, delivery, claim, claim_sha256, level_earned, primary_gate, gate_type, worlds_role, objective_metric, narrowing_rows[], updated_utc, previous_snapshot, user_decision.

**ERRATA.jsonl**: id, utc, old_primary_gate, new_primary_gate, classification BUG_FIX|POST_HOC, diff_summary, fixture_id, reviewer_signoff.

**KB ledgers (PROJECT_DOCUMENTS/KB):**
- USER_DECISIONS {id UD-<yyyymmdd>-<n>, utc, verbatim, session, channel, scope[], numbers_restated[], recorded_by}.
- FINDINGS {id F-<yyyymmdd>-<agent>-<n> or seed F-UF-xx, kind finding|activation|narrowing|supersession|retraction|transfer|reconciliation, refers_to, claim, level, conditions {substrate, input_code, priors_supplied, feedback, dose, comparator, best_shortcut, worlds, metric_definition}, effect {value, ci, unit}, source path:line@sha, status, delivery, author_agent, utc, caveat}. The current status and R-level are folds over the rows.
- INCIDENTS {id, kind report|closure, fm_class, delivery, symptom, evidence, found_by_agent, found_utc, caught_at_discovery, checker_sha_at_discovery, heldout_eligible, check_id, fixture_id, status}.
- DELIVERY_LEDGER {delivery, utc, legacy, task_family, level_earned, primary_gate, gate_type, worlds_role, opened_held_E3 (true|false|'unknown'), e3_basis[], objective_metric {held, old_final, G}, family_id, claim_sha256}.
- CONFIRM_LEDGER {family_id, delivery, attempt_index, alpha_allocated, seal_reveal_commit, utc}.
- REVIEW_DEBT {id, delivery, utc, reason, recorded_by, repaid_by}.
- CANARY_COMMITMENTS {batch_id, author_identity, utc, bundle_sha256, key_sha256, n, n_clean, fm_coverage, checker_sha_under_test, revealed_utc}.
- CONSTRAINTS.json (Section 13.1 lists its keys).
- FAILURE_CLASSES.json {FM-xx: {title, definition, check_ids[], fixture_ids[], judgment_question, archive_fc[], incidents[]}}.

**RULES.json, PROTECTED.json, EXPECTED.json**, owned by the checker and protected:
- PROTECTED.json {paths[], append_only[]}.
- EXPECTED.json {fixture_id: {commands [[...]], must_fire[], must_not_fire[], must_warn[], expected_exit[], expect {json_pointer: value}, source_incident, fm_ids[]}}.

---

## 9. Measurement: canonical scorer, ladder, strata, witness, rule, G, OLD, UDI, OC

### 9.1 Canonical metric

An item is exact when all of the following hold:
- own_eos is true, recomputed;
- emitted bytes equal expected bytes, recomputed from the prompt for addition;
- answer_path is native.

balanced_exact_own_eos is the mean of per-expected-answer exact rates. Denominators come from lock.n_items_per_pool.

### 9.2 Shortcut ladder

Ladder members are computed per world, pool and checkpoint from the candidate arm's TAUGHT manifest and its own streams:
- CONST: most frequent taught answer; ties go to the lexicographically smallest.
- LENFREQ: most frequent answer among taught prompts of the same length.
- SUFFIX1..3: most frequent answer among taught prompts sharing the last k bytes.
- EXACT: the taught answer for an identical prompt.
- CANON: the taught answer for the same canonical orbit (sorted, normalized operands).
- NEAREST: the answer of the taught prompt with the most positional byte matches.
- COPY_A, COPY_B: copy one operand.
- ID_ZERO: if an operand is 0, the other operand.
- DIGIT_NO_CARRY: digitwise sum mod 10.
- FIXED_0..8: the arm's own stream_hex truncated to k bytes, with own EOS assumed at k.
- MODE_REPEAT: the arm's modal output byte repeated to its modal emitted length.

The fallback for all members is CONST. Operand members apply to addition only, and skips are recorded.

Pool subsets:
- TAUGHT, OLD and NEW use exposure-independent members only (CONST, LENFREQ, SUFFIXk, COPY_A/B, ID_ZERO, DIGIT_NO_CARRY, FIXED_k, MODE_REPEAT).
- PRESENTATION uses the full ladder except CANON.
- HELD uses the full ladder.

### 9.3 Strata (addition, computed from item content and the taught set)

| Stratum | Definition |
|---|---|
| S0 trivial | An operand is 0, or the answer equals an operand |
| S1 presentation twin | Canonical orbit taught (excluded from E3) |
| S2 interpolation | Each operand value appears in its position in some taught pair; no carry |
| S3 carry | As S2, with carry |
| S4 extrapolation | An operand value is unseen in its position, or the answer is longer than every taught answer |

E3 counts only S2..S4 items that are also determined.

### 9.4 Determinacy witness (evaluator-side; never given to the learner)

- **pair_rectangle:** held (a, b) is determined iff there exist a' != a and b' != b such that (a, b'), (a', b) and (a', b') are all taught (either orientation if commutative). Under additive structure, (a, b) = (a, b') + (a', b) - (a', b').
- **column_closure (multi-digit):** held is determined iff every column triple (a_i, b_i, carry_in_i) it requires occurs in some taught item. These column triples are the multi-digit units.

### 9.5 Declarative decision rule (decide.py)

Preconditions are applied in order:
1. INERT alarm: INVALID with reason INTEGRITY_ALARM.
2. ADM-TB failure: CANNOT_TEST_CLAIM_AS_SPECIFIED.
3. ADM-CTRL failure: INVALID with reason TREATMENT_NOT_DELIVERED.
4. Any missing cell: INCOMPLETE. A shadow verdict is computed on the complete paired worlds.
5. Non-finite values: INVALID with reason NONFINITE.

Then, per world w:
- v_cand = exact rate of the candidate on counted items;
- v_comp_w = max(chance, comparator arms, best shortcut for the pool);
- d_w = v_cand - v_comp_w.

Aggregators:
- **paired_lcb:** LCB = mean(d) - t(1 - alpha, n - 1) * sd / sqrt(n). If sd == 0, use the Hoeffding bound LCB = mean(d) - 2 * sqrt(ln(1/alpha) / (2n)). The primary passes when LCB > margin_abs.
- **k_of_n:** count(d_w > margin_abs) >= k.
- **all_worlds:** every d_w > margin_abs.

The rule also requires:
- mean(v_cand) >= absolute_floor;
- min(d_w) >= per_world_floor;
- every plumbing gate passes;
- guards pass. old_drop: mean(OLD_end - OLD_final) <= max_drop, using the upper bound under paired_lcb. old_acq: mean(OLD_end - best exposure-independent shortcut on OLD) >= margin.

If any world has role technical, dev or rehearsal, the verdict is DEV_NOT_EVIDENCE with a shadow_verdict. Otherwise the verdict is PASS when everything passes, else FAIL.

The Student-t quantile is implemented with the regularized incomplete beta function (Lentz continued fraction) and inverted by bisection, then validated against tabulated values. The alpha used must not exceed the CONFIRM_LEDGER allocation (CONF-01).

### 9.6 G (the objective-aligned generalization score)

```
G = mean_w [ exact(W, HELD, final, counted)
             - max(exact(no_prior_teaching), best shortcut, chance) ]
```

Companions:
- G_per_sample = G per 1,000 taught presentations (from receipts);
- G_per_cpu_h = G per lineage-wide CPU-h, including failed attempts;
- OLD_drop and OLD_acq.

The program progress series is G on SEALED_CONFIRM worlds only. MAIN-world G is shown separately as 'unsealed'.

### 9.7 UDI (required only for E3 promotion)

- For each base world, a variant world removes the witness support of a dependent half D of the counted held items. D is drawn from a seal-derived seed for SEALED_CONFIRM, and the support of the independent items I is kept.
- UDI = mean_w [ (acc_base(D) - acc_var(D)) - (acc_base(I) - acc_var(I)) ], with a world-level LCB.
- Promotion requires LCB > 0.
- Shortcuts and teaching-as-trigger learners give UDI of about 0.

### 9.8 OC (SEALED_CONFIRM and adoption)

- Monte Carlo with 2,000 replications (random.Random(seed)).
- Per world: binomial counts at p_comp and p_comp + target_delta, with between-world SD taken from dev d_w (or declared).
- Reports P(pass | target) and P(pass | null).
- OC-01 requires P(pass | target) >= 0.8.

### 9.9 Headline grammar (generated)

`<PRIMARY_GATE> | earned <E> (attempted <E>) | held exact own-EOS <x>% [S2+ determined, n=<n>] vs best shortcut <name> <y>% | G <g>pp | OLD final <o>% (drop <d>pp) | <n> <ROLE> worlds | best of k=<k>`

Optional suffixes:
- ` | <arm> >= candidate` when any control or shortcut is at least the candidate;
- ` | amended: co-primary <gate>`;
- ` | priors: <list>`;
- ` | WEAK_COMPARATOR`;
- ` | dev shadow <verdict>`.

When nothing held was measured, the headline reads 'held exact own-EOS not_measured'.

---

## 10. Seal protocol (every SEALED_CONFIRM) and confirmation ledger

**Principle.** Entropy arrives only after the lock is fixed and witnessed. Neither party can grind it, and no secret has to be stored long-term.

1. The executor freezes ENDPOINT_LOCK and the final SOURCE_LOCK (shas L1 and L2) and relays '<sha> <path>'.
2. The reviewer (other family) attests L1 and L2 and runs `MFC seal commit-nonce <dir>`. This generates the nonce with secrets.token_hex(32) into a private file outside the repo, then writes and commits `review/SEAL_COMMIT.json` {lock shas, nonce_sha256, reviewer, utc} on the reviewer's branch.
3. The owner gives GO after seeing L1 and L2. The GO is recorded verbatim in AUTHORIZATION.stages.official {granted, verbatim, utc, lock_shas_shown} and committed after SEAL_COMMIT.
4. The reviewer runs `MFC seal reveal`, which writes `review/SEAL_REVEAL.json` {nonce}, committed after GO. If the nonce is lost, steps 2-4 are repeated; no run data is lost.
5. Derivation: key = sha256(nonce || owner_go_verbatim_utf8); world_seed_i = int(HMAC-SHA256(key, L1 || '|' || L2 || '|confirm_worlds|' || i)[:8]). The UDI dependent-set seed and replay seeds use purposes 'udi' and 'replay'.
6. At reveal, `seal reveal` appends a CONFIRM_LEDGER row {family_id, attempt_index j, alpha_allocated}, before any outcome exists.

Checks:
- SEAL-01: SEALED_CONFIRM requires SEAL_COMMIT, GO and SEAL_REVEAL.
- SEAL-02: ordering. SEAL_COMMIT precedes GO, GO precedes REVEAL, and REVEAL precedes the first official output. SEAL_COMMIT is on a reviewer-namespace branch.
- SEAL-03: the nonce hash matches, and the GO verbatim names the lock shas.
- SEAL-04: official world ids equal the derived ones.
- SEAL-05: the CONFIRM_LEDGER row precedes outcomes.
- CONF-01: lock alpha <= alpha_allocated. Default geometric spending is 0.05 * 2^-j per family, pending the owner's decision.

Single-agent mode cannot seal. This is stated, never claimed otherwise.

---

## 11. Knowledge ledger and prediction scoring (research-level generalization)

**Units of research knowledge.**
- **Findings** (FINDINGS.jsonl): scoped unit facts with conditions, best_shortcut, effect with CI, metric_definition, source path:line@sha, status and caveat.
- **Failure classes** (FAILURE_CLASSES.json): FM-01..30, each linked to check ids, fixtures and a judgment question. A failure class is unit knowledge that has been compiled into a check.

**Lifecycle.**
- The delivering author proposes 0-3 findings, appended with status proposed.
- The other family activates a finding with an activation row (KB-03) only when its source is a T2 MAIN or SEALED_CONFIRM delivery with no open debt (KB-04).
- Anyone may append narrowing, supersession or retraction rows; narrowing is conservative and needs no second agent.
- Nothing is deleted. The fold gives current status and R-level.
- Seed rows F-UF-01..29 come from the digest with their statuses: UF-15 narrowed by FIXED_1; UF-16 an uninformative manipulation; UF-02 UNVERIFIABLE_IN_REPO; UF-25 provisional, dev only. They start at R0 pending Codex reconciliation. Archive unit facts enter as provisional.

**R-levels (computed).** The same ladder applies to checks: R1 means it catches its fixture, R2 a variant fixture, R3 a sealed canary or held-out incident.

| Level | Meaning |
|---|---|
| R0 | Recorded |
| R1 | Re-derived from source by a non-author (activation with evidence) |
| R2 | Holds under a re-derived metric or changed presentation |
| R3 | A registered prediction citing it beat its base rate on a later delivery |
| R4 | R3 on a different task family or substrate |

Only R3 and above are called 'transferable'.

**Predictions.**
- They are registered in the lock for the primary and guards only (LOCK-11), and scored at publish (CLM-10):
  - hit means the observation falls inside interval90;
  - Winkler interval score;
  - Brier score for p_pass;
  - an unopened endpoint is scored 'unresolved'.
- A transfer row {prediction_id, delivery, ok|failed} is appended to each finding the prediction cited.
- A finding with 2 or more failed transfers and negative skill is narrowed by appended row.
- Calibration is a KB diagnostic only. It is never an agent score or a kit score, which closes off sandbagging.

**`MFC kb score`** reports:
- hit rate and Brier skill against the Laplace base rate per task family, only where the base rate lies in [0.1, 0.9] and at least 5 scored deliveries exist (otherwise INSUFFICIENT_N or DEGENERATE_BASE);
- per-finding transfer counts.

**One-shot KB backtest (Phase 2).**
1. `MFC kb backtest-snapshot --as-of 2026-10-15T00:00Z --out DIR` runs git archive of the last commit at or before T.
2. A fresh session reads only DIR plus redacted design briefs of the post-T deliveries (results, REPORT and SUMMARY stripped) and writes two forecast files: contract-only (the objective contract alone) and KB-informed.
3. `MFC kb backtest-score` computes Brier skill for each against Laplace, plus KB value = Brier(contract-only) - Brier(KB) with a paired bootstrap CI.
4. The result is always labelled LEAK_RISK, because the container holds full history. It is a falsifiable early signal, not a recurring ritual.

**Explore stage use.**
- `MFC kb status` prints:
  - cadence and deliveries_since_last_E3;
  - open incidents;
  - narrowed findings with caveats;
  - open debts;
  - the sealed G series.
- Designs cite F-ids.
- `init` prefills base rates.

---

## 12. Testing the tests

### 12.1 Pyramid

- **L0 checker self-tests:** unit tests, `selftest`, and `selftest --mutate`, which proves every check is exercised by a dirty fixture and constrained by a clean fixture.
- **L1 regression corpus** (Section 12.2): a synthetic reproduction of each historical failure class, plus clean fixtures that must pass. Blocking legitimate work is a failure too.
- **L2 retro on real history** (Section 12.4): measured recall and false-flag rate before the v1 freeze.
- **L3 per-delivery test of the delivery's own tests:**
  - cycle-0 testbed (a ceiling must pass, floors must fail);
  - RED-01..11 rule qualification on the delivery's actual configuration;
  - ADM-GATE for plumbing gates;
  - LAD-02 substitution;
  - ADM-SHUF and ADM-CTRL from raw rows.
- **L4 reviewer:**
  - executed checks;
  - role-targeted canary mutants on the delivery's own artifacts, with the seed chosen after the lock;
  - post-run replay for SEALED_CONFIRM or adoption.

  These operators are public, so they test checker regression and declaration minimization, not held-out generalization.
- **L5 held-out:** sealed canary batches from a non-tool-author (Section 12.3), plus held-out incident recall on incidents first recorded after the freeze.
- **L6 owner spot-check:** 3 random claim numbers traced to raw artifacts per census.
- **L7 knowledge:** prediction scoring and the one-shot backtest.

### 12.2 Regression corpus (one or more fixtures per failure class)

Fixtures are generated deterministically by `corpus/_build/build_corpus.py` from a hand-built clean base (8 worlds; arms W, NPT, SHUF, FROZEN, ORACLE, plus CEILING and FLOOR; about 10 items per pool). The base has hand-computed SCORES and VERDICT values. The generated output is committed; selftest never runs the builder.

| Id | FM | Built from (source) | Mutation | Expectation |
|---|---|---|---|---|
| C00a | clean | base, T2 MAIN pass | none | All stages exit 0. Hand-computed SCORES and VERDICT values. |
| C00b | clean | base | candidate at shortcut level | verdict FAIL; publish exit 0 |
| C00c | clean | base | all ceilings fail; testbed_invalid amendment on technical-only results, no UD | stop_reason CANNOT_TEST; publish exit 0 |
| C00d | clean | base | complete seal, OC 0.9, replay | SEALED_CONFIRM; confirm wording allowed |
| C00e | clean | base | T1 dev only, no review | primary_gate NOT_OPENED with dev shadow; publish exit 0 |
| C00f | clean | governance | add a check plus fixture, with GOV-REV record | governance exit 0 |
| C01 | FM-01 | minifly_b/REPORT.md:12-26; archive FC08 | emitted_hex from a buggy decoder; expected_hex drops carry | MET-01, MET-02 |
| C02a | FM-02 | minifly_b/REPORT.md:12-26 | N_old_rel writes 288 on old_relation; post_first equal to W; identical streams | ADM-CTRL-ZERO, INERT-01 |
| C02b | FM-02 | minifly_recoverability_v1_r2_dev_20261023/REPORT.md:30-39 | realized_dose 0.00312 | ADM-CTRL-DOSE |
| C03 | FM-03 | minifly_address_capacity_design_v1_20261023/RECEIVED_REVIEW.md:36 | ZERO arm also changes the stop channel at post_first | ADM-CTRL-ISO |
| C04a | FM-04 | V51 AUDIT.json A4 | fixture density 0.02 vs params 0.0354; plumbing gate without a negative fixture | ADM-FIX, ADM-GATE |
| C04b | FM-04 | LEARNED-BANK NOTES section 1 | SHUF derangement on binary labels | ADM-SHUF |
| C05a | FM-05 | LEARNED-BANK CYCLE2.md:10-36 | ceiling grid fails | ADM-TB-CEIL; expect stop_reason CANNOT_TEST |
| C05b | FM-05 | LEARNED-BANK CARRY/CMP | FROZEN floor passes | ADM-TB-FLOOR |
| C05c | FM-05 | persistent_core_v2_trial/results/SUMMARY.json | margin 0.6 with 8 items per world | RED-01 |
| C06a | FM-06 | minifly_address_capacity_v1_dev_20261023/REPORT.md:7 | candidate approximately FIXED_1 of its own stream; DIRECT comparator at 0 | expect /verdict FAIL, headline best shortcut FIXED_1; must_warn LAD-03 |
| C06b | FM-06 | minifly_v6_r1_dev_20261023/v6_data.py:7-34 | gains only on zero-operand items | expect /verdict FAIL; S0 not counted |
| C06c | FM-06 | V6 audit_results.py:91 (NO_OLD guard) | OLD_END at shortcut level, drop 0 | expect /verdict FAIL (old_acq) |
| C07 | FM-07 | minifly_stop_only_confirm_v1_20261023/DESIGN.md:5 | conditional primary; CLAIM edited to gate_type objective | LOCK-03 (forced component), CLM-02 |
| C08a | FM-08 | LEARNED-BANK CYCLE2.md:39-103 | T-MOD to T-PAR after learner outcomes, no UD | AMD-02 |
| C08b | FM-08 | synthetic | margin changed without an AMENDMENTS row | AMD-01 |
| C09a | FM-09 | CATALOG.md:36-43 | 2 non-E3 v1 rows; component lock; no waiver | LOCK-09 |
| C09b | FM-09 | adversarial judge | last 'E3' delivery ended CANNOT_TEST | LOCK-09 (no reset) |
| C10 | FM-10 | LEARNED-BANK CYCLE3.md:46; addition_rule cycles | rehearsal world used in dev; 3 distinct (closure, params) | CYC-03; expect headline 'best of k=3' |
| C11a | FM-11 | minifly_cls_dev_20261017/REPORT.md:3 | official world id present in another delivery | WLD-01, AMD-04 |
| C11b | FM-11 | adversarial judge | SEALED_CONFIRM without a seal | SEAL-01, CLM-04 |
| C12 | FM-12 | minifly_v5_design_review_20261020/REVIEW.md:25-31 | commuted held item in a feedback-bearing calibration stream; undeclared stream file | ADM-EXPO |
| C13a | FM-13 | commit 43d04b5e | lock and results in one commit | LOCK-06 |
| C13b | FM-13 | LEARNED-BANK make_lock.py:12 | native imports bootstrap.py, which is not in SOURCE_LOCK | LOCK-05 |
| C13c | FM-13 | LEARNED-BANK 1ce1eadb to 84c530d4 | T2 dev output before any attestation | WIT-01 |
| C14 | FM-14 | V6 audit_results.py:97; LEARNED-BANK summarize.py | literal exposure_overlap = 0; rule_type custom without UD | LOCK-08, DEC-01 |
| C15 | FM-15 | native_next_v1_dev_20261023/REPORT.md:13,40 | admit.py edited after a FAIL, no amendment | AMD-05 |
| C16 | FM-16 | V6 REVISION.md:40-52 | analytic SNR 0.027; contrast ratio 0.014; taught 0% | LOCK-12, ADM-INPUT, ADM-ACQ |
| C17 | FM-17 | minifly_v4_dev_20261020 ACCOUNTING.json | negative directional delta; zero replay with 32 applications | ADM-DIR |
| C18 | FM-18 | minifly_v6_r1_dev_20261023/cycles/01_REVIEW.md:7-9 | literal sys.path.insert to a sibling delivery's core.py | LOCK-05 |
| C19 | FM-19 | native_next_v1_dev_20261023/REPORT.md:12,46-58 | official OUTPUTS not covered by ATTEMPTS | RUN-01 |
| C20 | FM-20 | minifly_v5_1_r2_dev_20261020/REPORT.md:3-43 | paired_atomic false; one world's controls missing | ADM-BUDGET; expect /verdict INCOMPLETE |
| C21 | FM-21 | V6 EXECUTION_SPEC (2 units) | SEALED_CONFIRM intent; OC 0.41; 4 worlds | OC-01, CLM-09 |
| C22 | FM-22 | CLS REPORT.md:93-98; LEARNED-BANK CYCLE2.md:61 | REPORT first line 'largely met'; 'Native confirmation' on DEV | CLM-05, CLM-06, CLM-04 |
| C23 | FM-23 | ROUTE_RETROSPECTIVE A03/A07; PA-13 | closure Jaccard 0.9 with an earlier delivery, parent null; `.backward(` in a native file | LOCK-13, LOCK-07 |
| C24 | FM-24 | CATALOG P036; CURRENT_EXTERNAL_MEMORY | pointer quotes a narrowed finding; 'confirmation_passes': true; /Users path | PTR-04, PTR-03, PTR-05 |
| C25 | FM-25 | V51 AUDIT A6 | prediction basis F-id absent | LOCK-11 |
| C26 | FM-26 | minifly_v6_revised_design_20261023/REVIEW.md:11 | a scored prediction removed; incident closed without a fixture | CLM-10, KB-05 |
| C27 | FM-27 | multimechanism byte-identical audits | same-family review on the author's branch; no executed checks; copied outputs | REV-01, REV-02, REV-07 |
| C28 | FM-28 | stop_only PLAN.json:4; transformer_external REPORT.md:3 | caps 54,432 absent from the verbatim text; '(Recommended)' label; self-evaluated conditional | AUT-02, AUT-03, AUT-04 |
| C29 | FM-29 | V6 and native_next PUBLICATION.md | T2 claim without the verification subset; /Users path | CLM-15, CLM-12 |
| C30 | FM-30 | LEARNED-BANK NOTES omissions | delivering author activates its own finding | KB-03 |
| G01 | FM-30 | design | RULES threshold edit in the same range as results | GOV-SEP, GOV-UD |
| G02 | FM-30 | design | EXPECTED must_fire id removed without a loosen UD | GOV-CLASS, GOV-UD, GOV-MONO |
| G03 | FM-30 | adversarial judge | CONSTRAINTS margin 0.05 to 0.03; verbatim lacks '0.03' | GOV-NUM |
| G04 | FM-30 | design | author agent edits a check after its own lock | GOV-COI |
| G05 | FM-30 | design | a SKILL.md 'NEVER' line removed without a UD | GOV-CLASS, GOV-UD |
| K01 | FM-26 | design | three synthetic deliveries with known predictions | expect Brier and skill values |

selftest also runs `canary` on C00a with fixed seed 'c0ffee'. All applicable operators must be caught (CAN-01..03).

Role-targeted canary operators. Each operator chooses its target by role or behaviour, not by declaration:
- M01: copy W's post_first channel digests and writes into the causal-control arm receipt.
- M02: negate a directional delta.
- M03: perturb a fixture param by 40%.
- M04: delete a plumbing gate's negative fixture.
- M05: replace candidate rows with the best shortcut's predictions (expect not PASS).
- M06: drop a comparator arm's rows.
- M07: insert a commuted held prompt into a feedback-bearing TAUGHT stream.
- M09: put 'largely met' in the headline sentence.
- M10: set worlds_role SEALED_CONFIRM in a CLAIM copy.
- M11: append `x.backward()` to a native file copy.
- M12: set one emitted_hex to null.
- M13: add a literal 'exposure_overlap = 0'.
- M14: add 'confirmation_passes': true to a pointer copy.
- M15: relabel the causal-control arm's role to 'other' and clear its zero_channels.
- M17: invert SHUF labels.

Git-history operators (amendment, timing) are exercised only in the corpus.

### 12.3 Held-out canary protocol (the checker's analogue of E3)

1. **Author.** The owner, or a fresh session that has read only FAILURE_CLASSES.json, SCHEMAS.md and this spec's Section 8, and has not opened tools/minifly_check/*.py. The author's identity is recorded.
2. **Batch.** 10 mini-deliveries in corpus fixture format, at least 3 of them clean. At least half must not be isomorphic to any visible fixture: a different FM instance shape, different names and layout. Each carries a key {fixture: expected fired classes}.
3. **Seal.**
   - `MFC heldout seal --batch B --in DIR --key-file OWNER_KEY` encrypts with an HMAC-SHA256 counter-mode keystream XOR and writes `corpus/heldout/sealed/B.bin` (ciphertext).
   - It appends a CANARY_COMMITMENTS row {bundle_sha256, key_sha256, n, n_clean, author, utc}.
   - The row must be committed before the checker version under test merges (HELD-01).
   - The key is held by the owner, outside the repo.
4. **Reveal and score.** At census, or before a governance merge, `heldout reveal` (HELD-02) and `heldout score --checker-sha S`. The score uses the frozen checker via git archive at S and reports recall per FM and the false-alarm rate on clean canaries, written to KB/RETRO/HELDOUT_<date>.json.
5. **Burn.** A scored batch moves into the visible corpus through governance and is never held-out again.
6. **Fixes after a miss** must generalize. A reviewer rejects any fix that matches canary literals, and the next sealed batch tests it.
7. **Held-out incident recall.** It counts only incidents first recorded after the v1 freeze, with heldout_eligible true. Their fixture is written by someone other than the check author and committed by hash (a CANARY_COMMITMENTS row with n = 1) before any check is written. The frozen checker's result is recorded as caught_at_discovery.
8. **Archive.** The pre-Full151 archive digest is training data. It is already digested in this design session, and its FC01 is the same defect as C01/C02a. Archive-derived fixtures (A-series) may extend the visible corpus but never count as held-out.

The checker's score is held-out recall per checker version. A rise in corpus recall without a rise in held-out recall counts as overfitting.

### 12.4 Retro acceptance (Phase 0; real history; read-only)

`MFC retro --all --out <scratch>` emits RETRO_ANNOTATION rows (PASS, FLAG or NOT_APPLICABLE, each with a reason). A commit adding 3 or more delivery dirs is NOT_APPLICABLE_BULK_IMPORT for timing; f0cfbcca is an example.

Must FLAG, where artifacts permit:
- R-CTRL: minifly_b/results/science receipts (gz), N_old_rel identical to W.
- R-CLOSURE and R-LITERAL: minifly_v6_r1_dev_20261023 audit_results.py outside SOURCE_LOCK, plus line 97 literals.
- LEARNED-BANK, read via `git show origin/charming-head@f33d8dee:...` and never checked out:
  - R-FAILOPEN: summarize.py 'is not False' and len(verdict);
  - R-CLOSURE: make_lock.py;
  - R-CLAIM: 'Native confirmation';
  - R-LOCKTIME: 1.7 minutes.
- R-CLAIM: minifly_cls_dev_20261017 'largely met'.
- R-POINTER: CURRENT_EXTERNAL_MEMORY_EXPERIMENT_DESIGN.json confirmation_passes.
- R-CATALOG: P036 +22.10pp without FIXED_1, and 40 free-text evidence_level strings.
- R-LOCKTIME: 43d04b5e, 3065bb1a, 43b07e89.
- R-AUTH: 14 of 39 cycle deliveries without AUTHORIZATION.
- R-PATHS: count of /Users paths.

Must stay quiet:
- studies/rcenter-survivor-rounds-20261003 round-1 statistics;
- the addition_sample_generalization_dev_v2_20261011 SELECTION_LOCK-before-confirm ordering;
- PROJECT_DOCUMENTS/V51_CLAUDE_AUDIT_RECONCILIATION_20261020.

Acceptance:
- at least 90% of the artifact-permitting FLAG list is flagged;
- at most 15% of applied checks false-flag on the quiet list;
- results are reported to the owner before UD-0001;
- STOP-ONLY and v5.1 fixture catches are prospective-only (legacy artifacts lack stream_hex and production params), marked NOT_APPLICABLE.

---

## 13. RSI governance and protected invariants

### 13.1 CONSTRAINTS.json keys (owner-set; draft values pending UD-0001)

- **Identity:** schema, owner, version, status (DRAFT_PENDING_OWNER_DECISION), user_decision, objective_ref {path, sha256}.
- **native_forbidden:**
  - imports: torch.autograd, torch.optim, torch.nn.MultiheadAttention, transformers, jax, sklearn.linear_model;
  - calls: .backward(, scaled_dot_product_attention(, numpy.linalg.lstsq(, numpy.linalg.pinv(, torch.linalg.lstsq(;
  - identifiers: transformer, attention, elm, and requires_grad=True.
- **supplied_arithmetic_heuristics:** int() over decoded operand bytes, carry or digit tables, '+' applied to parsed operands.
- **Exemptions and I/O:** evaluator_only_exemptions; io {raw_bytes true, own_eos_required true}; warmup {start_ceiling '1/16', ramp 'gradual'}; objective_targets TARGET_UNSET.
- **threshold_floors:**

  | Key | Value |
  |---|---|
  | margin_abs | 0.05 |
  | absolute_floor_above_chance | 0.05 |
  | per_world_floor | -0.05 |
  | min_worlds_nonpilot | 8 |
  | min_items_per_stratum | 8 |
  | input_contrast_min | 2.0 |
  | prompt_id_margin_min | 0.10 |
  | taught_acq_margin_min | 0.05 |
  | min_realized_dose | 0.05 |
  | shuf_band | [0.35, 0.65] |
  | old_max_drop | 0.10 |
  | old_acq_margin | 0.05 |
  | analytic_input_snr_min | 0.5 |
  | oc_min_power | 0.8 |
  | alpha_max | 0.05 |

- **tier_thresholds:** t0_max_cpu_h 1, t1_max_cpu_h 4, t1_max_records 20000, family_t1_max_cpu_h 12.
- **default_grant_caps:** max_cpu_h 2, max_records 5000, max_cycles 3; official, push, adoption and pointer_update all false.
- **Program rules:** e3_cadence_max 2; e3_cadence_counts_legacy false (pending); review_debt_max_open 3.
- **required_arms_by_level:** E1 [candidate, shuf]; E2 + positive_control; E3 [candidate, no_prior_teaching, shuf, oracle_plumbing].
- **Testbed grids:** testbed_floor_grid FROZEN_RANDOM_FEATURES width [256, 1024, 4096] x readout_lr [0.001, 0.005, 0.02]; testbed_ceiling_grid EXACT_BP and ORACLE_FEATURES configs.
- **Publication and confirmation:** confirm_alpha_spending 'geometric_0.05_2^-j'; verification_subset_max_mb 50.

### 13.2 Change classes (computed by governance.py; unclassifiable counts as loosen)

| Change | Class | Requirements |
|---|---|---|
| New check id, a new fixture, or advisory to enforced | tighten | GOV-REV, GOV-SELF, GOV-SEP, GOV-MONO |
| Check code edit where fired(head) is a superset of fired(base) on all fixtures and clean fixtures stay clean | tighten_or_neutral | GOV-REV, GOV-SELF, GOV-SEP, GOV-MONO |
| SKILL.md, AGENTS_POINTER.md (or a tracked AGENTS.md) or .gitattributes text edit with no rule line removed | neutral | GOV-REV, GOV-SKILL, GOV-SEP |
| Removed check, fixture, must_fire id or rule line (containing minifly_check, MUST, NEVER or must/never); enforced to advisory; stage removed; tier_min raised; non-monotone code edit | loosen | Everything above plus GOV-UD and GOV-NUM (the owner's verbatim words restate the numbers) |
| Any CONSTRAINTS.json change | constraint | GOV-UD (scope constraint:<key>). A loosening direction also needs GOV-NUM. Direction map: higher is stricter for floors (except old_max_drop, alpha_max, tier thresholds, e3_cadence_max, review_debt_max_open and default caps, where lower is stricter); a narrower shuf_band is stricter. |

**Process:**
1. Proposal: diff plus a rationale citing incident ids, false-positive reports with evidence, or a measured cost.
2. Run `MFC governance --base B --head H`, which computes the class. For code changes, the base checker is extracted with git archive into a temp dir and both versions are run on both corpora.
3. Write the review record or the UD.
4. Commit as a separate range: protected paths only, with trailers.
5. The change applies only to locks built afterwards (introduced_in). It is advisory for 2 v1 deliveries at stages below T2 official. T2 official, publish and pointer stages always run the latest checks.

**Meta-Goodhart guards:**
- protected paths;
- computed classes;
- the owner's verbatim words with numbers for any loosening;
- cross-agent review for every other change;
- GOV-SEP (a delivery and its gate never share a range);
- GOV-COI (a fix by the author after its own lock does not apply to its own delivery without a UD);
- SELF-01 (no locally modified checker can produce a gate report);
- CLM-14 (reports carry the committed checker sha);
- sealed held-out canaries written by a non-tool-author and committed before the merge;
- progress measured only as sealed G, never as gates or flags;
- calibration never scored for agents;
- lessons from the delivering agent are inputs, not truth.

**Optional owner hardening** (decision): a GitHub push ruleset restricting the protected paths; owner-signed tags `minifly-kit-vX.Y.Z` that SELF-01 can verify at T2; one-time approval codes later.

### 13.3 The RSI loops

1. **Incident to check.**
   - Every defect gets an INCIDENTS row recording whether the frozen checker caught it.
   - If missed, the fix is a check, a fixture and an EXPECTED entry, merged through governance.
   - An incident closes only when its fixture is caught (KB-05). This turns 'lessons stay prose' (FM-26) into an invariant.
2. **Held-out generalization of the checker:** sealed batches and post-freeze incidents (Section 12.3).
3. **Predictions to findings:** scored transfer and R-levels (Section 11).
4. **Skill edits:** governed. They are validated by selftest plus re-running gates on the last 5 v1 deliveries; no FAIL may flip to PASS without a stated bug fix.
5. **Census prune** (every 5 v1 deliveries or monthly):
   - checks with zero live or canary catches over 10 or more deliveries and measured runtime above budget can be proposed for demotion;
   - demotion is a loosening, decided by the owner;
   - self-reported minutes are never used.

---

## 14. Anti-reward-hacking map

**Learner level:**
- the unconditional own-EOS primary;
- the comparator always includes the full shortcut ladder for the pool, including FIXED_k of the arm's own stream;
- S0 and S1 are excluded from E3, and only determined items count;
- no_prior_teaching, and UDI for promotion;
- the OLD acquisition floor;
- answer_path native, recomputed own-EOS and truth;
- the native purity scan;
- exposure checked by canonical intersection, with undeclared streams rejected;
- structure meters beside distinguishability;
- SHUF validity;
- seal-derived worlds for SEALED_CONFIRM.

**Research-agent level** (literature pattern, then countermeasure):

| Pattern | Countermeasure |
|---|---|
| Scorer patching (METR o3) | Program-owned scorer and decide; SELF-01; CLM-14 |
| Editing time or caps (AI Scientist) | Caps restated in the owner's verbatim words; launch enforces them; ADM-BUDGET |
| Deleting failure markers (Darwin Godel Machine) | Append-only ledgers (KB-02, GOV-APPEND); RUN-01 requires every result to be launch-recorded; failed attempts counted |
| Disabling the sandbox (STOP) | No dependence on hooks; SELF-01 at every gate |
| Modifying tests (ImpossibleBench) | AMD-05 for endpoint and evaluator files; governance for checks; GOV-COI |
| Abort option reduces cheating | Unpenalized honest exits |

Other agent-level defenses:
- self-judged conditionals: AUT-04;
- option framing: AUT-03;
- confirming on tuning worlds: CLM-04, CLM-09, CYC-03;
- fork-and-publish: CONFIRM_LEDGER with family alpha;
- rehearsal reuse: CYC-03;
- lock banking: T2 always runs the latest checks;
- KB poisoning: KB-03 and KB-04;
- sandbagging: calibration is never scored.

---

## 15. Loophole ledger (judge findings)

| Loophole | Closure in MRK-1 | Status |
|---|---|---|
| Salami tier gaming | Family-cumulative measured compute; T1 cannot activate findings, update pointers, confirm or reset cadence | Closed (residual: runs outside launch, below) |
| Self-reported control receipts | Per-channel digests at post_first (unchanged on lesioned channels, equal to W elsewhere); write counter at the mutation site; canary M01; reviewer rerun_admission | Partly open: counters remain executor code; the reviewer reads the write site |
| Self-reported row provenance | own_eos and emitted recomputed from stream_hex; truth recomputed; reviewer replay for confirm | answer_path label partly open (purity scan plus replay) |
| Uncommitted pre-lock runs | launch refuses dirty closures; RUN-01; seal entropy after lock for confirm | Open for non-confirm work (invisible tuning cannot be seen); closed for SEALED_CONFIRM |
| Nominal E3 cadence or reset by honest exit | Cadence computed; reset only by admitted, scored MAIN or SEALED held-E3 with S2+ | Closed |
| Rote predictions or pessimism skill | Primary and guard only; skill only on non-degenerate base rates; contract-only backtest; calibration never scored | Closed as an incentive (rote effort remains) |
| Vacuous or poisoned findings | 0-3 with reason; activation by the other family from T2 MAIN or SEALED only | Closed |
| Soft-word synonyms | Generated headline is the primary defense; expanded list blocks in gate sentences; WARN elsewhere | Partly open in body prose (advisory) |
| Spoofed identity | Trailer plus namespace containment; owner spot-check; optional push ruleset | Open in-repo (tamper-evident only) |
| Fabricated owner decisions | Verbatim with restated numbers; append-only; owner census spot-check; optional push ruleset or signed tags | Open without an owner-held anchor (owner decision) |
| Canaries not held out from the executor | Operators reclassified as regression; role-targeted; true held-out = sealed batches by a non-tool-author | Closed by reclassification |
| Shaped held-out recall | Author is not the check writer; committed by hash before the check; archive is training data | Closed |
| Hash-code distinguishability | Admission can only stop a run; structure meters required; claims rest on G against the ladder | Closed |
| Advisory-window timing and lock banking | T2 official, publish and pointer always enforce the latest checks | Closed |
| Degraded-mode self-review | Same family never counts; debt blocks confirm, activation and pointers; cap of 3 | Closed |
| Weak-but-fail-closed rule or fixture-detecting decide | Program-owned decide; lock parameters at or above owner floors; RED-01 reachability | Closed; floors become the target, and the owner sets them |
| Salt grinding and world pre-screening | Post-lock two-party entropy; worlds derived after GO | Closed for SEALED_CONFIRM; MAIN worlds can be pre-screened, so confirm is banned there |
| Fork-and-publish across deliveries | CONFIRM_LEDGER by computed family, with alpha spending | Closed for SEALED_CONFIRM |
| DAG order is not time order | WIT-01 reviewer attestation in the reviewer namespace; seal for confirm | Partly open at T1 (local order only, DEV-stamped) |
| Teaching-as-trigger | UDI for E3 promotion | Closed for promotion; plain E3 is labelled reuse_specificity not_tested |
| Witness-aware supplied arithmetic | LOCK-07 heuristics plus reviewer; sealed digit relabeling offered as owner decision | Open pending decision |
| Weakened causal control | ADM-CTRL-ISO equal elsewhere at post_first; reviewer rederivation | Closed at the first record; later divergence is judgment |
| Weak floor | Floor = best over the CONSTRAINTS grid; checker-scored; reviewer reruns one config | Partly open: floor learners are executor-run |
| Declaration minimization | Required arms by role; non-empty zero_channels; strata computed; undeclared streams rejected; purity over runner and native; budget measured | Closed for listed fields |
| Self-selected triggers | Fixed bundles at T2; ADM-REUSE only at T1, with computed closure equality | Closed |
| Relabeled science | k computed over (closure, params); any primary or guard change after learner outcomes needs a UD | Closed |
| Self-refuted blockers | Counter-sign or owner ruling | Closed |
| Self-generated canaries | Dropped | Closed |
| Output fabrication | Recompute plus seeded replay for SEALED_CONFIRM or adoption | Closed for confirm; open for MAIN |
| Pruning via self-reported cost | Measured runtime and catches only | Closed |
| Overfit 'monotone' loosening (regex narrowing) | Sealed canaries; code edits need cross-agent review | Partly open between batches |

---

## 16. Ceremony cost

- **T0:** no new files. `MFC lint` takes about 1 minute if text is published.
- **T1:**
  - ENDPOINT_LOCK via `init`: about 20-30 minutes the first time, about 10 with a parent.
  - admit.py and the runner contract (receipts, OUTPUTS with stream_hex, TAUGHT per arm, replay flags): about 2-4 agent-hours once per learner family, then reused.
  - Per run: a `launch` prefix and one `cycle close` line per cycle.
  - CLAIM is generated; findings take 5-10 minutes.
  - Checker runtime: under 10 s per gate, about 10-15 s per 1e6 rows for scoring, under 20 s for selftest.
  - Expected steady-state overhead: under 1 agent-hour per delivery.
- **T2:** about 1.5-2.5 agent-hours in total plus one owner touch (GO and caps). This adds AUTHORIZATION (5 minutes), one review event by the other agent (30-90 minutes), SOURCE_LOCK acks (5 minutes each) and the verification subset (minutes).
- **SEALED_CONFIRM:** the seal adds about 10 minutes across three parties; post-run replay is CPU-capped at 30 minutes or less; OC takes seconds.
- **Governance:** a tightening needs a cross-agent review of about 20 minutes. A loosening needs one owner decision. Census takes about 1 hour of agent time plus about 10 minutes of the owner's.

**What it buys.** Each item below is caught in seconds of checker time from artifacts that should exist anyway:
- Package B: about 80 core-hours, caught by ADM-CTRL at C1.
- STOP-ONLY: 225,792 records, caught by the decide ladder on dev.
- V6-R1: 34,816 records, caught by LOCK-12 at design.
- v5.1: 8,206 applications, caught by ADM-FIX and ADM-GATE.
- v4: 6 invalid histories, caught by ADM-DIR.
- LEARNED-BANK: 11 CPU-h, caught by ADM-TB in C0 and WIT-01.

Gate runtimes are recorded in GATE reports for prune reviews.

---

## 17. Rollout

**Phase 0: build and validate. No agent behaviour changes; T0.**
1. Implement WP1-WP10 on a claude/* branch cut from codex/minifly-consolidated-20261023.
2. Run selftest green (under 20 s) and `selftest --mutate`.
3. Run `retro --all --out <scratch>` and measure recall and false-flag rate against Section 12.4.
4. Codex performs the cross-agent governance review of the whole kit and reconciles it.
5. Do NOT track a root AGENTS.md: Mr. Zhou keeps an untracked local root AGENTS.md (its sha256 is cited in several EVIDENCE.json/SOURCE_LOCK.json files), and a tracked file at the same path would block their pull. Instead, PROJECT_DOCUMENTS/KB/AGENTS_POINTER.md holds the pointer lines for the owner to merge into the owner's own file; Claude Code discovers .claude/skills/ directly.
6. The owner records UD-0001. Verbatim, it restates the CONSTRAINTS numbers and freezes checker 1.0.0, CONSTRAINTS v1, and the seed ledgers as provisional.
7. Commit the retro annotations to KB/RETRO as RETRO_ANNOTATION, not verdicts.

**Phase 1: prospective schema v1.**
- v1 applies to delivery dirs created after UD-0001.
- The first two v1 deliveries (ideally one per agent) run checks below T2 official in advisory mode. T2 official, publish and pointer stages are enforced from day one; owner waivers are logged and count against the waived check.
- Runner families adopt the runner contract.
- Codex adds `Acting-Agent:` trailers and pushes only to codex/*.
- LEARNED-BANK stays legacy. Before its verdict is cited:
  - run `retro` on it;
  - a Codex executing review is recorded in KB/RETRO, covering the fail-open summarize.py, the lock closure, D3 reuse, and amendment A1's distance from E3;
  - its findings enter as provisional;
  - any pointer to it follows PTR-07.

**Phase 2: after two v1 deliveries.**
- Review measured gate runtimes, false positives, catches and canary survivors.
- Flip advisory checks to enforced (tightening, with GOV-REV), or loosen through a UD.
- Run the one-shot KB backtest at T = 2026-10-15.

**Phase 3: first sealed held-out canary batch,** authored by the owner or a fresh session. Commit its hashes before the next checker merge. Reveal it at the next census and report held-out recall for v1.x. Archive-derived A-series fixtures join the visible corpus as training data.

**Phase 4: at 5 or more v1 deliveries.** First `kb score` diagnostics, first prune review, check R-levels updated.

**Deferred** until an incident shows a silent failure they would catch: input tap and runtime guard, separate learner process, evaluator-owner role, typed revision taxonomy, tripwires, recurring backtests, review canaries, one-time codes, a program-owned evaluator learner library.

---

## 18. FM coverage table

| FM | Mechanism (checks) | Residual / judgment |
|---|---|---|
| FM-01 common-mode auditor | Program-owned scorer and decide (MET-01/02, DEC-01). Control expectations come from lock declarations. INERT-01. Reviewer trace_number and rerun_admission. | The checker is a program-wide common mode (mitigated by clean fixtures, --mutate, owner spot-check) |
| FM-02 undelivered manipulation | ADM-CTRL-ZERO, ADM-CTRL-DOSE, INERT-01; INVALID with reason TREATMENT_NOT_DELIVERED distinct from FAIL | Write counter is executor code; reviewer checks the write site; canary M01 |
| FM-03 confounded intervention | ADM-CTRL-ISO at post_first; more than one declared change labels integrated_intervention (CLM-03) | Multi-factor dataflow beyond the first record is review judgment |
| FM-04 non-discriminating gate | Ladder always in the comparator; RED-02/07/10; LAD-02; ADM-GATE; ADM-SHUF; ADM-FIX; LAD-04 | Novel shortcuts outside the ladder become incidents and new ladder members |
| FM-05 untestable testbed | ADM-TB-CEIL/FLOOR in C0 before learner code; RED-01; DET-01; OC-01 for confirm; CANNOT_TEST is unpenalized | Floor learners are executor-run (grid fixed; reviewer reruns one) |
| FM-06 weak comparator or guard | Comparator = max(chance, arms, ladder) in decide; S0/S1 excluded; old_acq guard; LAD-03 in headline | — |
| FM-07 proxy metric | LOCK-03 (otherwise component); objective metric in every headline and pointer; PTR-03; answer_path native; MET-01; CLM-17 (a sub-result never described as the met rule, 20.13) | — |
| FM-08 post-outcome amendment | AMD-01..05; UD for any primary or guard change after learner outcomes or any distance increase; co-primary in headline; ERR-01 | — |
| FM-09 objective drift | LOCK-09 computed cadence with honest-exit-proof reset; objective_distance in lock and claim; UD for distance increase; component bridge predictions | Choosing the most informative step stays judgment |
| FM-10 in-cycle design search | designs_tried_k computed (best of k in headline); typed cycles with exit criteria; CYC-03 fresh rehearsal; CYC-04 append-only runlog at publish (20.16); dev worlds never MAIN | Task-shaped priors: priors_added in headline plus review |
| FM-11 holdout reuse | WLD-01, AMD-04, seal (SEAL-*), CONFIRM_LEDGER, WLD-02 item-universe overlap | MAIN worlds not seal-derived (no confirm wording there) |
| FM-12 exposure leakage | ADM-EXPO canonical intersection plus undeclared-stream rejection, and the content scan of every stream for held answers (20.11); exposed orbits are S1; LOCK-07 evaluator unreachable from native; supplied-arithmetic heuristics to review | No input tap (deferred); digit relabeling is an owner decision |
| FM-13 lock timing and closure | LOCK-05/06, WIT-01, RUN-01, seal ordering, CLM-08 POSTHOC | T1 ordering is local only |
| FM-14 fail-open reducers | No delivery reducers (DEC-01); RED-03..06; LOCK-08; MET-01 two-source | — |
| FM-15 test edited after failure | AMD-05 for endpoint and evaluator files; governance (GOV-SEP, GOV-COI, GOV-MONO) for checks; ERR-01 | Delivery-internal unit assertions are engineering; reviewer |
| FM-16 technical vs behavioral readiness | LOCK-12, ADM-INPUT (with structure meters), ADM-ACQ, ADM-TB, all before roster | — |
| FM-17 directional or fixture mismatch | ADM-DIR, ADM-FIX | Checkpoint resume round-trip is review judgment |
| FM-18 environment and import defects | LOCK-05 resolution outside root fails; launch requires a clean closure; env versions in SOURCE_LOCK | Loud defects are left to the executor by design |
| FM-19 accounting gaps | launch ATTEMPTS (child rusage), RUN-01, CYC-04, compute totals including failed in CLAIM, G_per_cpu_h | Deterministic op counters optional |
| FM-20 resource-stop truncation | ADM-BUDGET paired_atomic; INCOMPLETE on missing pairs; RUN-02; RUN-04 (resume and recovered-attempt replay, 20.8, 20.17); caps restated in verbatim | Mid-run cap changes need a new AUTHORIZATION |
| FM-21 statistics | One decide implementation (t/Hoeffding, per-world floor); min_worlds/PILOT (CLM-09); OC-01; CONF-01 family alpha | — |
| FM-22 claim inflation | Generated headline; enums; CLM-04/05/06/09; CLM-17 structural: no level above level_earned in CLAIM text, status key/value shapes carry the claim (20.20, 20.21); CLM-17P free-prose lint aid (an unqualified E-level above level_earned is flagged whatever the verb, 20.21) adjudicated by the T2 reviewer (REV-09) | Free-prose overclaim is a judgment check, not mechanical: at T2 a sentence CLM-17P flags publishes only with the reviewer's 'ok', and a prose overclaim CLM-17P does not flag (no E-level token, or a met primary without a MET word) publishes unless the reviewer adds it on their own; at T0/T1 CLM-17P only warns |
| FM-23 lineage and answer-path drift | LOCK-13 computed family and parent; LOCK-07; CLM-07; answer_path native; priors in headline | Equation-level diff is review judgment |
| FM-24 narrowings not propagating | FINDINGS fold; PTR-02/04/06; pointers copied from CLAIM; generated views avoided by copy rule | CATALOG and MEMO remain hand-written legacy (retro flags) |
| FM-25 misremembered facts | Predictions cite F-ids (LOCK-11); CLM-16; reviewer trace_number | Prose citation lint is advisory |
| FM-26 lessons stay prose | Incident closure needs check plus fixture (KB-05); FAILURE_CLASSES linked to checks; prediction scoring; R-levels; held-out recall | — |
| FM-27 self or padded review | REV-01 (family plus namespace, common.commit_family 20.15); REV-02 executed checks; UNVERIFIED rule; REV-07 padding; DEBT-* | In-repo identity is spoofable |
| FM-28 authorization | AUT-01..04 (numbers in verbatim, defaults, no '(Recommended)', non-beneficiary conditionals); LOCK-10 | Transcription is trusted; owner spot-check |
| FM-29 unverifiable publication | CLM-12/15; PTR-05; kit and KB in repo; retro marks UNVERIFIABLE_IN_REPO | Size cap is an owner decision |
| FM-30 meta-loop Goodhart | Governance classes, GOV-UD/NUM/REV/SEP/COI/MONO, SELF-01, CLM-14, sealed held-out canaries (post-batch checks excluded, 20.18), progress only sealed G, KB-03 (activator from trailer and namespace, 20.15) | Without an owner-held anchor, governance is tamper-evident only |

---

## 19. Residual risks

1. **Checker common mode.** One scorer bug affects every delivery. Mitigation: hand-computed fixtures, --mutate, reviewer trace, owner spot-check.
2. **Identity and owner-decision authenticity** cannot be proven in-repo while Codex commits as the owner's git identity. Mitigation: optional push ruleset or signed tags; census spot-checks.
3. **Invisible tuning** through scratch scripts outside launch, at T1 and on MAIN worlds. Only SEALED_CONFIRM is immune.
4. **Executor-side instruments** (write counters, answer_path labels, floor learners). Mitigated by review and replay, not eliminated.
5. **Addition-centric ladder and strata.** A non-addition task raises objective_distance and needs a UD. Novel shortcuts are added after the fact.
6. **False positives at rollout.** Mitigated by the Phase-0 measurement, the advisory first two deliveries, and logged waivers that count against checks.
7. **Owner bandwidth** for GO, UD and census. Mitigated by batching and default caps.
8. **Ceremony creep** through the incident-to-check loop. Mitigated by census prune on measured cost and catches.
9. **KB backtest leakage** (labelled LEAK_RISK).
10. **E3 cadence pressure** may push agents to open poorly designed E3 endpoints. Admission and testbed screens make those exits honest and cheap.
11. **AGENTS.md collision** with the owner's local file (Phase 0, step 5).
12. **Verification subset size** versus the raw-data-free publication policy (owner decision).


---

## 20. Integrator amendments (applied before implementation)

1. **No tracked root AGENTS.md.** The pointer lives in `PROJECT_DOCUMENTS/KB/AGENTS_POINTER.md` (protected, <=20 lines) for the owner to merge into the owner's own local AGENTS.md. GOV-SKILL checks that file (and a root AGENTS.md only if one is tracked).
2. **Honest implementation status.** Every check id in Section 7.4 is either implemented and exercised by the corpus, or registered in RULES.json with `status: "planned"`, excluded from every bundle, and listed in tools/minifly_check/README.md under "Not yet enforced". A check function is never stubbed to return PASS. `selftest` fails if a bundle references a planned or unregistered id.
3. **Integration changes to tests are governed even in Phase 0.** When the corpus and the checker disagree during integration, the integrator decides from this spec. Every edit to EXPECTED.json or to a fixture is logged in `tools/minifly_check/INTEGRATION_LOG.md` with the spec clause that justifies it, and an independent verifier re-reads that log for loosening before the kit is committed.
4. **Repo-scale scans are bounded.** Repo-wide scans (WLD-01, REV-07, LOCK-13 family, retro) read only `*.json`, `*.jsonl`, `*.jsonl.gz`, `*.md` and `*.py` files outside `.git/`, skip files over 64 MiB and archive members, and report the scan scope in their detail field. A scan that cannot complete gives MISSING, never PASS.
5. **Confirm wording is matched as identifier tokens (hardening round 4; CLM-04, I6, 3.4).** Before matching, CLM-04
   normalises each text with NFKC, removes Unicode format characters (category Cf: U+200B-U+200F, U+2060, U+FEFF, the
   soft hyphen and the bidi controls) and combining marks, and folds the Cyrillic and Greek letters that look like the
   Latin letters of "confirm". Every case-insensitive occurrence of "confirm" is then a confirm token, taken with its
   enclosing identifier token `[A-Za-z0-9_]+`, so snake_case and CamelCase labels count (GAIN_CONFIRMED,
   axis_confirmed, SCIENTIFIC_CONFIRMATION, NativeConfirmation, dev_confirmed, NOT_CONFIRMED, reconfirms,
   con<U+200B>firmed). The only exempt tokens are the whole kit enum and schema names SEALED_CONFIRM, sealed_confirm
   (official_role_intended / worlds_role values), CONFIRM_LEDGER (as in CONFIRM_LEDGER.jsonl), confirm_attempt and
   confirm_alpha_spending, matched case-sensitively. The scope is unchanged: REPORT.md, STATUS*/SUMMARY* files (JSON
   keys and string values), the headline, notes and proposed finding claims. Corpus S56-S58 (DEV) must fire CLM-04;
   C00d (SEALED_CONFIRM) stays clean. `dev_touched` (CLM-09, WLD-01) fails closed on a receipt that does not parse or
   is not an object instead of skipping it.
6. **The decision comparator includes every declared control arm (amends 9.5; verifier probe 2c).** v_comp_w =
   max(chance, decision-comparator arms, best shortcut for the pool), where the decision comparator is
   primary.comparator_arms plus every declared arm whose role is no_prior_teaching, shuf, frozen or lesion. Arms of
   role oracle_plumbing or positive_control and evaluator-only CEILING_*/FLOOR_* arms are never added, and the cycle-0
   testbed view (decide mode testbed) adds none. This closes declaration minimization: a FROZEN arm identical to the
   candidate on HELD gave PASS because FROZEN was not a declared comparator; now d_w = 0 and the verdict is FAIL (corpus
   S59, clean, honest FAIL). Every decision-comparator arm needs its primary cells (a missing one is INCOMPLETE);
   RED-02 and RED-03 qualify the rule against each of them. VERDICT.primary records comparator_arms and
   implied_comparator_arms; SCORES records decision_comparators, and the headline suffix ` | <arm> >= candidate` names
   any decision-comparator arm at or above the candidate. INERT-01's zero-variance contrast, LAD-03 (WEAK_COMPARATOR),
   CLM-03's no_prior_teaching requirement and canary M06 keep reading the declared comparator_arms, so a failing
   candidate at the level of a FROZEN or SHUF arm is an honest FAIL, not an integrity alarm. G (9.6) is unchanged.
7. **RUN-03 runtime closure (new check; LEARNED-BANK lesson: a hand-listed lock covered 6 files while the runtime
   closure was 55, 44 of them package modules).** The checker still never imports delivery code; the runner records
   what it imported. Every counted run (stage technical, dev, rehearsal or official) writes
   `runlog/RUNTIME_CLOSURE_<attempt_id>.json` = {schema minifly.runtime_closure.v1, attempt_id, files: [{path
   (repo-relative), sha256}]} for every repo file in sys.modules at the end of the run plus the data assets it opened
   from its declared list, with the 12-line stdlib helper of the runner contract (also in templates/admit_skeleton.py),
   called after the imports and again at the end. `launch` sets MINIFLY_REPO_ROOT and MINIFLY_RUNTIME_CLOSURE and
   records the file in produced_paths. RUN-03 (stages roster, official, publish; T2 enforced, T1 WARN like RUN-01):
   for every such attempt that produced any path, the file exists, is in its produced_paths with the current sha256,
   and every listed repo .py file is in the SOURCE_LOCK.closure_sha256 the attempt ran under (found by sha) with the
   same sha256; a file present at runtime but absent from the lock FAILs with "closure gap". Other listed files are data
   assets and must be in files.assets, files.params, endpoint_files_sha256 or SOURCE_LOCK with the same sha256. CLM-15
   adds the official attempts' runtime closures to the verification subset. Corpus: S60 fires (a bootstrap.py reached
   only by importlib); every clean fixture now carries runtime closures.
8. **Resumable official runs with a determinism replay (amends RUN-02 and 3.1 S6; new check RUN-04; LEARNED-BANK
   lesson: the container restarted mid-run, the session resumed only the missing histories under the identical lock and
   re-ran one completed history to show bit-identical determinism).** `MFC launch --stage official --resume <dir> --
   <same cmd>` continues an interrupted official attempt. A resume attempt (ATTEMPTS resume_of) is legal only if (a)
   its ENDPOINT_LOCK and SOURCE_LOCK shas equal those of the interrupted attempt; (b) the interrupted attempt is the
   latest earlier official attempt, its ATTEMPTS row exists, and it ended with a nonzero or killed exit or an
   incomplete (world, arm) unit set; (c) it writes OUTPUTS rows and receipts only for units no earlier official attempt
   has rows for (no overwrite, no duplicates): every earlier attempt's recorded `units` (canonical digests written by
   launch) still match and no other file an earlier official attempt produced is rewritten or removed; and (d) RUN-04:
   when it wrote official rows, it re-ran the unit(s) launch chose at random (at least one unit completed by an earlier
   attempt) into `results/official_replay/`, and their canonical row digest (attempt_id excluded) equals the original.
   Any deviation FAILs. Without --resume, a second official attempt that produces rows still FAILs RUN-02. RUN-02 and
   RUN-04 run at the official and publish stages. The resume uses the same command (CYC-03) and is driven by
   MINIFLY_RESUME_OF, MINIFLY_COMPLETED_UNITS, MINIFLY_REPLAY_UNITS and MINIFLY_REPLAY_DIR. When the launch process
   itself dies, its in-flight marker `runlog/INFLIGHT_<attempt_id>.json` lets the next launch record the attempt
   (exit -1, interrupted, recovered, rusage lost), so (b) can be met after a container restart; RUN-01 FAILs while a
   marker has no ATTEMPTS row. CLAIM gains the generated field provenance {interrupted, resume_attempts, resumed_units,
   replay_units}. Corpus: S61 (legal resume, clean), S62 (a completed unit rewritten: RUN-02), S63 (no replay: RUN-04),
   S64 (replay digest differs: RUN-04). Governance note: RUN-02 now accepts a legal resume that it used to reject; in
   the SPEC 13.2 sense that part is a loosening of RUN-02, bounded by (a)-(d), and it needs the owner's UD at the
   freeze (UD-0001) like every pre-freeze kit change.
9. **One launch per phase (per-phase lock re-check).** Every `launch` invocation re-runs its stage bundle (LOCK-05:
   SOURCE_LOCK.closure equals a fresh lock build, from the technical bundle on) and re-verifies ENDPOINT_LOCK,
   SOURCE_LOCK and every closure, files.* and endpoint file sha before the command starts, and again after it ends
   (closure_clean_after). A multi-phase run therefore calls launch once per phase and never chains phases inside one
   unlaunched script, so a fix between phases fails loudly at the next launch (unit test
   test_launch.TestLaunch.test_every_phase_relaunch_rechecks_the_lock).
10. **Versioning after a held-out batch (self-improvement round 1; governance lesson of HB-20261024-PILOT).** The
    pilot's raw score (recall 3/9, false alarms 3/3) measured schema churn: RUN-03 was added after the batch was authored
    with the same introduced_in 1.0.0, so every canary fired it. Rule: **a check added or tightened after a held-out
    batch is committed carries a later version than the batch's checker version**: a new check gets introduced_in, a
    tightened one keeps introduced_in and appends the version to the RULES field `changed_in` (a list, each entry later
    than introduced_in and not later than CHECKER_VERSION; `selftest` validates it). Round 1 is CHECKER_VERSION 1.0.1:
    CLM-17 and CYC-04 have introduced_in 1.0.1; ADM-EXPO, CLM-04, KB-03, REV-01, GOV-REV, GOV-COI, WIT-01, SEAL-02,
    DEBT-02, HELD-03, DEC-01, RED-02, RED-03 and RUN-04 list changed_in [1.0.1], and since the round-1 repair (20.19)
    also LOCK-02 (role candidate, 20.14) and STR-01, DET-01, ADM-TB-CEIL, ADM-TB-FLOOR and UDI-01, whose results the
    exposure strata of 20.11 change (S65/S66 fire them under 1.0.1, not under 1.0.0). The advisory window (introduced_in later
    than a delivery's first SOURCE_LOCK checker_version gives ADVISORY_FAIL below T2 official/publish/pointer) keys on
    introduced_in only: a tightening of an existing check is enforced at once, so changed_in never relaxes a check.
    Held-out scoring reads both fields (20.18). Governance classifies an appended changed_in entry as neutral when the
    same range changes the check's module or a kit module it imports (an import-closure over the head tree; round-1
    repair: a version recorded for unchanged code would exclude the check's held-out hits and false alarms as
    post_batch), as a loosening without such a change, and a dropped or rewritten entry as a loosening. The corpus keeps its 1.0.0 SOURCE_LOCKs and committed 1.0.0 gate reports
    (the builder lists only checks introduced by 1.0.0 in them), so T1 corpus deliveries see the new checks through
    the advisory window, as a real older delivery would.
11. **Exposure is judged by content, not declarations (amends ADM-EXPO and 9.3; FM-12, INC-HB1-A3).** ADM-EXPO scans
    every declared exposure stream (feedback_bearing true or false), every file under the TAUGHT directory and every
    undeclared TAUGHT/stream/calibration-like file (all of manifests/ except the split manifests; under results/,
    receipts aside, every non-canonical JSONL file and every path part naming taught, stream, calib, curric, expos,
    warm or listen). Records: JSONL rows; for JSON documents the elements of a top-level list, the objects in its
    top-level lists and the rest of the object; for text files each line with its comma/tab/semicolon/pipe fields. Every
    string value and key is read as UTF-8 text, as hex and as base64, and every list of byte values as bytes. A record
    leaks a held item when it carries the item's prompt (E2, exact) or canonical orbit (E3 addition; `a+b`, with or
    without the terminator, in any field) together with its expected answer (a value equal to the answer bytes after
    EOS, whitespace and leading zeros are stripped, or one value holding the prompt followed by the answer). Round-1
    repair (verifier probes A4-A6): a string value that is itself a JSON object or array (as text, hex or base64) is
    parsed and its content is part of the record; inside one value the answer may follow the prompt or orbit after up
    to 32 bytes of non-digit filler ('3+7 equals 10', '"q": "3+7=", "a": "10"'), unless that number is the first
    operand of a following sum, or precede it as 'c = a+b'; and records are read in stream order: when a record's value
    at some field path ends with a held prompt or orbit (only whitespace, '?', '.', ':' and EOS after it) and the next
    record's value at the same path starts, after up to 16 bytes of non-digit filler, with its answer (not a sum's
    first operand), the held item leaked (text lines: the line and each field position are paths). The filler, 'c =
    a+b' and stream-order rules read text only: the raw text of a hex-shaped string (it is the encoding, so '6b...'
    is never the answer 6) and a hex, base64 or byte-list decoding that is not printable UTF-8 (base64 of a hex string
    decodes to noise) keep only the exact rules above. Any leak
    FAILs ADM-EXPO, whatever the stream's or the row's feedback flag; in a feedback-bearing row a held prompt/orbit in
    any field FAILs as before. Non-feedback streams may hold held prompts WITHOUT answers (listening, presentation). A
    stream file that cannot be read as JSONL, JSON or UTF-8 text FAILs (its content cannot be checked). Integer fields
    are not read as answers (counts and indices would collide with small sums): a residual risk the reviewer covers.
    Since the round-1 repair a data file (.jsonl, .json, .txt, .tsv, .csv and their .gz) anywhere else in the delivery
    (outside checks/, runlog/, review/) whose path names one of those stream parts is a candidate too (data/warmup.jsonl).
    Residuals the reviewer covers: an answer two or more records later, or in another field of the next record; files
    listed in files.assets; neutrally named JSON under results/; a string-valued number equal to a held sum next to a
    held prompt counts as an answer (fails closed).
    Strata (9.3): a held orbit that occurs with its answer in any such record is S1 (taught) in every world and is
    never counted; the scorer takes the orbits from the same scan (`admitcheck.exposed_held_orbits`). Corpus: S65 (one
    hex field, prompt then answer, in a listening stream), S66 (text cue and a separate answer field in a priming
    stream), S67 (clean: held prompts without answers in a non-feedback stream); round-1 repair: S84 (a JSON-encoded
    string payload with the answer under another key), S85 (prompt and answer on consecutive lines of a tab-separated
    tape), S86 (clean: JSON payloads with held prompts only, back to back). The round-4 unit test that let a
    presentation-only row carry the answer is tightened.
12. **confirm wording is read as rendered (CLM-04; verifier non-blocking finding).** Before the 20.5 normalisation,
    `claimcheck.markup_skeleton` decodes HTML entities (repeatedly), removes HTML comments and tags, and removes the
    markdown markers `*`, `**`, backticks, `~~` and `__` (single underscores stay: they are identifier characters, and
    intraword `_` is never emphasis in markdown). So `con&#8203;firmed`, `con**firm**ed` and `con<!-- -->firmed` are
    confirm tokens (corpus S74-S76). Round-1 repair (verifier probe D4): each text is read a second time with markdown
    inline links, images and reference links replaced by their bracketed text (`[con](#)firmed`, `con[]()firmed`), and
    a token that reading adds to a line counts too; the first reading keeps link targets, so a target named confirm*
    still counts (corpus S88). retro.py's R-CLAIM reuses `claimcheck.confirm_tokens` (one tokenization); its
    binding rules then read each token as the word 'confirm'. Re-running `retro --all` with the shared tokenization
    gave the same 1234 rows, statuses and reasons as the committed RETRO_20261024.json (acceptance true).
13. **CLM-17: overclaim in agent-written text (new check, introduced_in 1.0.1; FM-22, INC-HB1-C2).** Scope (publish):
    the REPORT.md body below the headline, CLAIM.notes, findings_proposed[].claim and the STATUS*/SUMMARY*/MEMO* files
    of the delivery, and since the round-1 repair its NOTES*/ANALYSIS*/README*/RESULTS* files outside results/ and
    manifests/ (JSON: each string value, and each scalar leaf as the clause '<key words> is <value>', false as 'is
    not true'). Lint (T0): a file inside a v1 delivery with CLAIM.json is judged against that claim; a file outside
    every delivery is judged per sentence (per clause when a sentence names several) and per table row with its header
    against the claim of each v1 delivery of the repo it names, with the name read as 'this delivery'; a file that names
    none is NOT_APPLICABLE. The text is normalised as in 20.12, identifiers are split into words, sentences are cut into clauses
    at ; : | ( ) [ ] dashes and the conjunctions but, while, whereas, although, though, however, yet, because, since,
    so, and, with, whilst, which. Key/value shapes are read like the JSON path (round-1 repair, verifier blocking issue:
    the split cut 'Level earned: E3' into 'Level earned' and 'E3'): every 'key: value' or 'key = value' segment of a
    line (segments cut at ; | and ', '), every short 'key - value' segment (at most 3 words a side) and every data cell
    of a markdown table row (keyed by the row's first cell and, under a header row, by its column header) also gives
    the joined clause '<key> is <value>' (key and value of at most 6 words); a run of two or more key/value lines is
    read line by line, never joined into one sentence. A clause FAILs when (a) it holds an E0..E4 token above
    level_earned together with an achievement word or shape (reached, achieved, earned, attained, met, demonstrated,
    established, cleared, passed, qualifies, secured, obtained, delivered, supported, warranted, substantiated, proven,
    'holds' ending a claim, 'is at', 'at E3', 'level is', 'counts as', 'E3-grade', 'is an E3 learner', 'the result is
    E3', 'E3 ... is the result'; in a joined clause also an affirmative value: yes, true, done, a check mark), unless
    every achievement word is pass/clear, its subject is a process check named before it (lock, roster, admission,
    testbed, rehearsal, review, canary, lint, plumbing, schema, preflight, audit, or a stage word with gate/check/test/
    screen/stage/bundle) and each level token qualifies a process noun (E3 design, E1 plumbing checks) in that subject
    or after a preposition behind the verb ('the roster gate for this E3 design passed'; 'the lock passed E3' still
    asserts), or (b) primary_gate is not PASS and it calls the primary, the (decision) rule or criterion, an endpoint,
    the objective, 'the result' or 'the outcome' met, cleared, passed, satisfied, reached, achieved, fulfilled,
    successful, a success or 'holds' (joined: also an affirmative value); a clause about guards only fails when a guard
    it names (by id or kind word: drop/retention, acquisition), or unnamed any guard, is not PASS in VERDICT. A clause
    with an explicit qualifier (attempted, target, goal, plan, aim, intend, toward(s), not, no, never, neither, nor,
    n't, fail*, below, would, could, might, if, unless, cannot, without, require*, need*, must, should, expected,
    predicted, whether, unearned, short of), or one that opens with the bare form of an achievement verb (an imperative
    or plan: 'Goal: reach E3 next cycle'), is not an assertion; in a joined clause when, whenever, iff, once, provided
    and assuming qualify too ('Decision rule: W passes when ...' defines the rule). Conformity phrases ('as planned',
    'exactly as predicted', 'as expected', 'in line with the plan', 'on target', 'without exception') are removed before
    the qualifier test, and a noun qualifier directly asserted as reached ('the E3 target was reached', 'goal met') does
    not qualify itself. A sentence that cites prior work (an F-id or catalog id together with a prior-work word) is about
    other deliveries and skipped, unless it also points at this delivery (here, now, currently, this delivery/run/
    design/report/...): then it is cut at commas too and only its parts that cite nothing and name no prior work are
    judged. Each text is read twice (20.12: as written, and with markdown links rendered as their text). The round-1
    word 'rung' was dropped from the achievement vocabulary (verifier: it occurs in no kit document, only in a canary),
    and 20.20 removed the 'E3 ... is the result' alternative as well (fitted to canary C2). level_earned and
    primary_gate come from the fresh claim regeneration (the committed CLAIM when regeneration is impossible). Corpus:
    S68 (REPORT prose), S69 (notes, a findings claim, STATUS.md, SUMMARY.json), S70 (clean: honest negations such as
    'E3 was not reached', 'attempted E3', 'the OLD drop guard failed, so the primary FAILs'); round-1 repair: S81 (a
    STATUS key/value block 'evidence_level: E3', 'Decision criterion: satisfied'), S82 (a REPORT quantity/value table
    with '| level earned | E3 |'), S83 (clean: honest key/value lines, table rows, NOTES.md and 'the roster gate for
    this E3 design passed'). Amended by 20.20: the clause heuristics above are the advisory CLM-17P; CLM-17 keeps only
    the structural reading (CLAIM text tokens and status key/value shapes).
14. **The decision comparator includes role 'other' (amends 20.6; relabel loophole).** IMPLIED_COMPARATOR_ROLES is
    no_prior_teaching, shuf, frozen, lesion and other. Relabelling a control as 'other' no longer removes it from the
    comparator; oracle_plumbing, positive_control and CEILING_/FLOOR_ arms are still never added. Corpus S80 (clean,
    honest FAIL: a FROZEN arm relabelled 'other' answers HELD like W). test_decide pinned the old exclusion and is
    tightened (DEC-01, RED-02, RED-03 changed_in 1.0.1). Round-1 repair (verifier blocking issue, probe E2): role
    'candidate' did the same, since only primary.candidate_arm is tested. Now LOCK-02 FAILs any arm of role candidate
    other than primary.candidate_arm (LOCK-02 changed_in 1.0.1), and IMPLIED_COMPARATOR_ROLES also holds 'candidate',
    so such an arm is a decision comparator whenever a lock carrying it is scored anyway. Corpus S87 (dirty: FROZEN
    relabelled 'candidate' answers HELD like W; LOCK-02 fires at lock and official, and the VERDICT is FAIL with
    comparator FROZEN). Residual risk: a copy of the candidate declared positive_control or oracle_plumbing stays out of
    the comparator by design (those roles are meant to be strong); the reviewer checks that such an arm really changes
    what its role says.
15. **Identity is derived from the trailer AND the namespace, by one function (amends 5, KB-03; FM-27/FM-30,
    INC-HB1-C3).** `common.commit_family(git, commit)` returns the acting family of a commit: its `Acting-Agent:` trailer
    must name exactly one family, and an `origin/<family>/*` branch of that family must contain the commit. A trailer
    whose namespace contains nothing (the commit is only in the other family's namespace, or only on branches of other
    remotes such as upstream/claude/*) is identity_mismatch (FAIL); a commit in no origin/claude/* or origin/codex/*
    branch is unpushed (identity not derivable); git author names and row fields never count. Every identity check uses
    it: KB-03 (the activator is the family of the commit introducing the activation row; the row's author_agent must
    equal it; the `kb status` fold ignores an activation whose commit identity fails and gives such a reconciliation no R1/R2 credit), REV-01 (the
    review commit and the lock commit), the REVIEW.json history rule, WIT-01 (attestation and SOURCE_LOCK ack commits),
    SEAL-02 (the SEAL_COMMIT and SEAL_REVEAL commits) and `seal reveal` (the SEAL_COMMIT commit), GOV-REV (the review record's introducing commit and the author
    family of every protected-path commit), GOV-COI, HELD-03 and DEBT-02 (a repayment counts only when the review's
    introducing commit acts as the reviewer family). On "all containing branches must be one namespace": in git a
    commit is contained in every branch that descends from it, so the author's commits are always contained in the
    reviewer's branch (which forks from them) and the reviewer's commits in the author's branch once it merges the
    review. A literal one-namespace rule would make every shared commit a mismatch (it would fail the clean base of the
    corpus and every real two-family history). The implemented rule is the strongest one git can decide: the namespace
    of the trailer's family must contain the commit, and a commit contained only in the other namespace (or only
    outside origin/claude|codex) is a mismatch; a commit in both namespaces is shared history and keeps its trailer's
    family. Residual risk: once any branch of the impersonated family's namespace contains a forged-trailer commit, the
    namespaces cannot tell. That happens when that family merges such a branch AND when it merely forks a new branch from
    the author's tip (verifier probe C6: a forged activation later contained in a forked origin/claude/* branch passes
    KB-03). Reviewers must neither merge nor fork from an author tip that carries a commit with their own family's
    trailer they did not write (minifly-independent-review section 1); a first-parent rule is not implemented.
    Corpus: S71 (a self-activation under a forged 'Acting-Agent: claude' trailer pushed to origin/codex/*), S72 (a
    governance review record whose only claude-named branch is on another remote), S73 (clean: a genuine activation
    from origin/claude/*). Unit tests of REV-01, WIT-01, SEAL and GOV that built reviewer commits without trailers or
    without branches now give those commits the trailer and namespace a real pushed commit has.
16. **CYC-04: append-only runlog at publish (new check, introduced_in 1.0.1).** CYC-01's append-only rule ran only up to
    the official gate, so an earlier ATTEMPTS or CYCLES row edited after the run reached publish unseen. CYC-04 runs the
    same rule (every committed row survives unchanged in every later commit and in the worktree; shallow clone:
    MISSING) at publish. Corpus S77 (the rehearsal ATTEMPTS row rewritten after the report). Like every new check it is
    advisory for a T1 delivery whose first SOURCE_LOCK predates 1.0.1 (20.10) and enforced at every T2 publish.
17. **A recovered official attempt needs a replay of its own units (amends 20.8; RUN-04 changed_in 1.0.1).** An official
    attempt recorded from its in-flight marker (rusage_lost / recovered: the launch process died, so exit code and CPU
    are unknown) vouches for its rows only through a determinism replay: at least one (world, arm) unit it completed
    must be re-run by a later `launch --stage official --resume` into results/official_replay/ with the identical
    canonical digest. Otherwise RUN-04 FAILs at official and publish. `launch --resume` chooses its replay unit among
    the units of unvouched recovered attempts while any exist (a replay-only resume is legal when every unit is
    complete). Corpus: S78 (a recovered attempt that completed every unit, no replay), S79 (clean: S61 with a recovered
    first attempt whose unit is replayed).
18. **Held-out scoring excludes post-batch checks (amends 12.3 step 4; the pilot's RUN-03 drift).** `heldout score` reads
    each revealed canary's checker_version from its own SOURCE_LOCK (the oldest one in its tree) and classifies every
    fired id whose RULES introduced_in or changed_in is later than it as `post_batch`. Post-batch ids are listed per
    fixture and in the report, and never count toward recall or toward a false alarm (a clean canary whose nonzero exits
    only post-batch ids explain is not an alarm). A canary without a SOURCE_LOCK has no version, and nothing is
    excluded. Batch authors record in the SOURCE_LOCK the kit version whose SPEC and SCHEMAS they wrote against (the
    builder's `source_lock_patch` option sets it); a batch written against 1.0.1 is scored on the round-1 checks, while
    re-scoring the burned pilot batch with 1.0.1 excludes them as post_batch. Round-1 repair (verifier): the report also
    carries the raw numbers with post_batch ids counted (recall_fixtures_raw, false_alarm_rate_clean_raw, per fixture
    caught_raw and false_alarm_raw) and `heldout score` prints both, because a batch built with the builder's default
    SOURCE_LOCK (checker_version 1.0.0) excludes every round-1 detection and every round-1 false alarm. Unit tests:
    test_governance.TestHeldoutPostBatch.
19. **Round-1 repair (verifier REVISE after 20.10-20.18; checker version stays 1.0.1, nothing was committed in
    between).** Blocking issues: CLM-17 key/value lines and table rows (20.13), ADM-EXPO JSON-encoded payloads and
    stream order (20.11), role 'candidate' on a control (20.14). Overfitting: 'rung' dropped from the CLM-17 vocabulary
    (20.13). Non-blocking findings addressed: changed_in bookkeeping and the code-change requirement (20.10), raw
    held-out numbers (20.18), CLM-17 conformity phrases, citations about this delivery, vocabulary, process-check false
    alarms, scope and project-level lint (20.13), CLM-04 markdown link syntax (20.12), ADM-EXPO prose answers and
    stream-named data files outside manifests/ and results/ (20.11), the fork case of identity laundering (20.15). New
    corpus fixtures S81-S88 (INTEGRATION_LOG A20); no new check id. Left as residual risks: integer answers and the
    other ADM-EXPO residuals listed in 20.11, positive_control/oracle_plumbing copies of the candidate (20.14), and a
    first-parent identity rule (20.15).
20. **Round-1 repair part 3: FM-22 by mechanism, not vocabulary (amends 20.13 and KB-05; new CLM-17P and REV-09,
    introduced_in 1.0.1; KB-05 changed_in 1.0.1).** The verifier showed that guessing overclaims in free prose with
    regexes both misses paraphrases ('W hits E3', 'crosses the E3 threshold', 'shows E3-level generalisation') and
    false-alarms on honest text ('| E3 reached | no |', 'Did W reach E3? No.'), and that one alternative ('E3 ... is the
    result') was fitted to canary C2's sentence. A larger vocabulary would only continue the arms race, so the check is
    split by what can be judged mechanically.
    (a) CLM-17 (enforced, structural). (i) CLAIM.notes, CLAIM.findings_none_reason and every findings_proposed[].claim
    name no E-level token above level_earned, in any phrasing and even negated. Tokens are read as rendered (the 20.12
    skeleton: entities, markup, format characters, combining marks; look-alike E letters folded; 'E 3', 'E-3',
    'E-three', 'evidence level 3'); an agent states the attempted level only through the generated level_attempted.
    findings_proposed[].level, when present, is E0..E4 or NONE and <= level_earned. (ii) Status surfaces, that is the
    REPORT body and the STATUS*/SUMMARY*/MEMO*/NOTES*/ANALYSIS*/README*/RESULTS* files of 20.13's scope, are read only
    for their key/value shapes: 'key: value' and 'key = value' segments, short 'key - value' segments, markdown table
    cells (keyed by the row's first cell and by the column header of the header row; a first-column cell is a value only
    under an earned-level header), one-line 'question? answer' pairs (or a question line followed by a short answer
    line) and JSON leaves (keyed by their last two path keys). Keys are compared after markup, list markers and context
    words (delivery, final, claim, overall, ...) are removed. A level key (level, evidence level, level earned, earned,
    level reached/achieved/attained, ...) carries level_earned at the head of its value, or a negation; a level column
    header carries a level <= level_earned, or a negation. A gate key (primary gate, gate, verdict, outcome, result,
    status, primary, primary endpoint, endpoint, decision, bottom line, conclusion, objective, ...; as a column header
    only when the row key is the delivery, 'primary' or the candidate arm) carries primary_gate, a negation, or no gate
    word at all; on a non-PASS primary it takes no affirmative value (yes, true, met, reached, pass, passed, achieved, a
    check mark, ...; for status and decision the task words done, ok and complete do not count) and, except status and
    decision, no short (at most 8 words) unqualified achievement phrase. A key naming the primary, the rule, a criterion,
    an endpoint or the objective takes no affirmative value or short unqualified achievement phrase on a non-PASS
    primary. A key about guards takes no PASS or affirmative value while VERDICT says that guard did not PASS. A key that
    holds an E-level above level_earned FAILs with an affirmative value and passes with a negative one (no, false, not
    met, not reached, fail, failed, a cross mark, a dash). A short (at most 6 words) unqualified value that is an E-level
    above level_earned FAILs ('Composition: E3', 'state: E3 unlocked'). Keys naming the attempted level, a target, goal,
    plan, next step, design, protocol, lock, endpoint, objective, scope or prior work are not judged for levels, and
    text citing prior work is skipped. Deviation from the order, stated: a gate key whose value holds no gate word,
    affirmative or achievement verb ('Status: draft', 'Result: see the table') is not judged, because 'status' and
    'result' also head free text; a level key's value is judged strictly, because its vocabulary is closed ('Level
    earned: pending' FAILs). Amended by 20.21: until then a 'status' or 'decision' value with an achievement verb
    was not judged either ('Status: decision rule passed on held items' passed CLM-17), an unstated narrowing that
    20.21 removes.
    (b) CLM-17P (advisory: WARN, never FAIL; publish and lint) runs 20.13's clause heuristics on free prose: the REPORT
    body, the agent-edited CLAIM texts, the status surfaces and every other .md/.txt file of the delivery outside
    checks/, runlog/, review/, results/ and manifests/ (so DISCUSSION.md is read too). Changes against 20.13: the 'E3 ...
    is the result' alternative is removed (fitted to canary C2; nothing in this SPEC or in FAILURE_CLASSES.json justifies
    it); generic shapes are added (a level token as the predicate of a copula unless a design or process noun follows,
    'E3-level', and the achievement verbs hit, exceed, surpass, cross, unlock, grant, accomplish, win, beat, make it to,
    get to, arrive at; on the gate side hit, accomplished, won and positive); questions assert nothing; a key/value
    segment is read only as its joined clause, and not at all when its key or header is qualified; out of reach, short
    of, unmet, missed and lacking qualify. Every WARN row lists each flagged sentence (evidence lines, and data.flagged
    [{label, line, sentence, clause, reason}]).
    (c) REV-09 (T2 publish, enforced): REVIEW.fm22_prose_adjudications [{sentence, verdict ok | overclaim, note}] gives a
    verdict for every sentence CLM-17P flags in the current text, and for every sentence of a CLM-17P WARN row of the
    delivery's latest publish GATE or DRY report that is still in the text. Sentences are compared as rendered (markup
    removed, case and whitespace folded; an adjudication may quote the part of a sentence that holds the flagged
    clause). An overclaim verdict blocks publish while its sentence is still in the agent-written text; the fix is to
    reword or remove it. The reviewer may adjudicate sentences the lint did not flag, and an overclaim verdict on them
    blocks just the same. The field is reviewer-owned: only reviewer-family commits may change it (review_history_problems),
    and the commit introducing REVIEW.json must be the reviewer's. Flagged sentences without a REVIEW.json are MISSING;
    a REVIEW_DEBT does not discharge the duty. MFC review check runs REV-09 too.
    (d) What this guarantees. Free-prose overclaim is a judgment check with a lint aid, not a mechanical guarantee. At
    T2 a sentence CLM-17P flags publishes only if the reviewer adjudicates it ok; a prose overclaim CLM-17P does not
    flag publishes with exit 0 unless the reviewer adds it to fm22_prose_adjudications on their own (an overclaim
    verdict then blocks). At T0/T1 (no review) CLM-17P only warns. The structural part is mechanical within its
    vocabulary (E-level tokens, the key sets above, the table, key/value, question/answer and JSON shapes). A level
    claim worded without an E-level token ('the top rung', 'the composition level') or a status key outside the key
    sets is left to CLM-17P and the reviewer. (Corrected by 20.21: this paragraph first said that at T2 a prose
    overclaim publishes only with the reviewer's 'ok', which held only for flagged sentences.)
    (e) KB-05 and append-only closures. INCIDENTS is append-only, so a closure whose fixture an ordered change re-scoped
    cannot be edited. A closure row may name earlier closures of the same incident in `supersedes`, with a note; KB-05
    then retires the superseded rows, provided the superseding row itself satisfies KB-05 (EXPECTED.json available, its
    check in RULES, its fixture's must_fire holding the check). This narrows what KB-05 validates (a stale closure no
    longer FAILs once a valid closure of the same incident replaces it) and is recorded in INTEGRATION_LOG C58; the
    incident stays closed only through a valid row. INC-HB1-C2-CL1 (CLM-17 on S68) is superseded by INC-HB1-C2-CL3
    (CLM-17 on S89, the structural notes and findings case), whose note records the prose residual; INC-HB1-C2-CL2
    (CLM-17 on S69) stands.
    Corpus (INTEGRATION_LOG A22): S68 re-scoped by order (must_fire CLM-17 became must_warn CLM-17P plus must_fire
    REV-09; its T2 publish still exits 1), S70's CLAIM.notes reworded (it named E3, negated), new S89-S92 (dirty,
    structural: notes and findings, findings_none_reason, 'Level earned: E3', '| E3 reached | yes |'), S93 (clean:
    negative answers and values), S94 and S96 (dirty: REPORT prose without an adjudication; adjudicated overclaim left in
    place), S95 and S97 (clean: a lint false alarm adjudicated ok; an adjudicated overclaim rewritten).
21. **Round-1 repair part 4: CLM-17P by token, status and decision values, findings leaves (amends 20.20; CLM-17,
    CLM-17P and KB-05 stay changed_in 1.0.1; no new check id; verifier REVISE on part 3).** The verifier showed that
    CLM-17P still needed an achievement verb from a list (12 of 30 new paraphrases flagged, 10 of the 23 that carry an
    unqualified E-level token), so REV-09 asked the reviewer about few prose overclaims; that CLM-17 (b) exempted
    'status' and 'decision' values from the achievement-phrase rule; and that 20.20 (d) overstated the T2 guarantee.
    (a) CLM-17P token rule. Every clause of a non-question free-prose sentence, and every joined key/value clause, that
    names an E-level above level_earned is flagged when it carries no qualifier, whatever its verb. Left out: a token on
    a design or process noun (design, lock, plan, attempt, stage, gate, plumbing, checks, tests, roster, admission,
    testbed, review, cycle, bundle, setup, protocol, spec, pipeline, strata, items, worlds, pool, question, hypothesis,
    target, goal, pre-registration: 'the E3 design', 'E1 plumbing checks'); outcome nouns (delivery, endpoint, rule,
    criteria, requirements, conditions, candidate) do not exempt. A key/value shape whose value is a measurement (it
    starts with a number: 'Held E3 composition: 85.2% exact') is read for the levels of its value only, while an
    achievement word on its key ('E3 reached: 8/8 worlds') is still read by the 20.13 shapes. A key/value shape with a
    negative value ('E3 earned: false', 'Reached E3: N', '0/8 worlds') or a qualified key asserts nothing, and a short
    JSON string under a key is read as its joined clause only. The 20.13 achievement shapes stay for a token on a
    process noun ('W passed the E3 gate'). Qualifiers are those of 20.13 plus, for CLM-17P only, requirement, intention,
    plan and deferral words (demand, hypothesis, hope, aspire, next, future, await, wait for); for CLM-17 (b) and
    CLM-17P alike, negative-polarity intensifiers and double negations are removed before the qualifier test ('not
    just', 'no doubt', 'beyond doubt', 'nothing short of', 'never failed to', 'no longer short of'), which only makes
    qualifiers rarer. The gate side (a met primary, rule, criterion, endpoint, objective or guard) still needs a MET
    word; a met primary worded without one ('came out green') is the reviewer's. Measured on the part-3 verifier's 30
    dirty and 20 honest REPORT sentences (visible training data now): 26 of 30 flagged, 0 of 20.
    (b) CLM-17 (b) status and decision values. On a non-PASS primary a 'status' or 'decision' value (context words
    removed: 'Overall status') FAILs when it is a short (at most 8 words) unqualified phrase whose MET word is paired
    with an outcome subject: the primary, the rule, a criterion, an endpoint, the objective or the result, an E-level
    above level_earned, or the candidate arm when no process check is named ('W passes', not 'W passed review'). A
    process check named before the first MET word is that verb's subject ('Status: lock gate passed for the primary
    design', 'Status: admission passed'). A comma segment that continues a gate key's value (or that of a key naming the
    primary, the rule, a criterion, an endpoint or the objective) without a key/value shape of its own is judged like a
    status value, unless the head value is a gate word or a negation: 'Decision: proceed, the rule is met' FAILs,
    'Conclusion: FAIL, although held composition passed the shortcut test' does not; CLM-17P reads the segment as a
    clause, also inside a key/value block. Negative values gain 'nothing' and a plain zero count ('Earned: nothing yet',
    '0/8 worlds', not '0/8 worlds failed'); 'below E3' stays a FAIL for a level key (it is neither the level nor a
    negation). Affirmative answers gain absolutely, indeed, certainly, definitely, surely, sure, yep, yeah, yup,
    affirmed and correct ('Has W earned E3?' / 'Absolutely.').
    (c) CLM-17 (a) reads every string leaf of each findings_proposed entry (its claim, conditions values, extra keys,
    and key names holding a level token), not only its claim. Level tokens also include Roman numerals after a dash
    ('E-III'), 'evidence tier/rung/grade 3', an ordinal before 'evidence level' ('the third evidence level') and a level
    or ordinal rung of 'the (evidence) ladder' ('level 3 of the evidence ladder'); 'rung 3 of the shortcut ladder' is
    not one.
    (d) 20.20 (d) is corrected above: at T2 an unflagged prose overclaim publishes unless the reviewer adds it. The
    FM-22 row of Section 18 and the INC-HB1-C2-CL3 note (an uncommitted row of this range, reworded in place) say the
    same.
    (e) KB-05 supersession checks the order: only a closure row appended after the superseded row retires it.
    Corpus (INTEGRATION_LOG A24): S98 ('Status: W passes on the held split'), S99 ('Decision: go ahead, the locked
    criterion is satisfied on held sums') and S101 (a finding's conditions scope 'held sums at the third evidence
    level'), all must_fire CLM-17; S100 (T2 REPORT 'The held split leaves W with an E3 to its name.', must_warn CLM-17P,
    must_fire REV-09); S102 (clean FAIL: negative answers, a measurement keyed by the E3 task, process-check status
    text, qualified prose) and S103 (clean PASS at E3: 'Status: primary met on held sums', 'Decision: W passes the
    locked rule'). The phrasings are new. Held-out exposure of this range: INTEGRATION_LOG E1.
