# Stage checklists (MRK-1 schema v1)

`MFC` means `python3 -I tools/minifly_check/minifly_check.py`, run from the repository root. `<dir>` is the delivery
dir. Field-level schemas: `tools/minifly_check/SCHEMAS.md`. Normative text: `tools/minifly_check/SPEC.md` sections 3.1,
3.2, 7.1 and 7.5. Proceed past a blocking command only on exit 0; on any nonzero exit, read the failing ids in the
newest `checks/GATE_<stage>_<utc>.json` and use the FAIL table in `../SKILL.md` section 7.

Every gate writes `checks/GATE_<stage>_<utc>.json` with the checker sha, both lock shas, the tier and its reasons, the
family_id and one row per check. Commit it with the work it gates. `--dry` writes `checks/DRY_<stage>_<utc>.json`,
which counts only for CYC-02.

## S0 Explore (T0)

- [ ] Run `MFC kb status`. Read the cadence (deliveries since the last held E3), open incidents, narrowed findings with
      their caveats, open review debts, and the sealed G series.
- [ ] Read the FINDINGS rows you will rely on, with their conditions and caveats; cite F-ids for every number about
      prior work. A narrowed finding is quoted only with its caveat (P036's +22.10pp only with FIXED_1).
- [ ] Find the nearest tested delivery by task family and lineage; note its family_id and verdict.
- [ ] Run the cheapest falsifier before proposing anything that needs learner code:
  - analytic SNR from the init formulas (V6-R1: 0.00268 signal SD against 0.1 noise SD, in
    `minifly_v6_r1_dev_20261023/REVISION.md:40-52`);
  - saved-data reanalysis (STOP-ONLY's saved streams against FIXED_1, in
    `minifly_address_capacity_v1_dev_20261023/REPORT.md:7`).
- [ ] Present options to Mr. Zhou neutrally, each with cost and objective distance
      ([AUTHORIZATION.md](AUTHORIZATION.md)). Include stopping as an option when it is one.
- [ ] If you publish text at T0, run `MFC lint <files>` and stamp the outputs DEV_NOT_EVIDENCE.

Reads: KB ledgers, FINDINGS, DELIVERY_LEDGER, earlier deliveries (read-only). Writes: nothing required.
Exit: Mr. Zhou's AUTHORIZATION for design/implementation is recorded.

## S1 Design and endpoint lock

- [ ] `MFC init <dir> [--parent <dir>] [--claim-level E1|E2|E3] [--task-family F]`. It writes ENDPOINT_LOCK.json from
      the template and prefills base rates.
- [ ] Fill every `__FILL__`: the contract block (claim_level_target, intent, objective_distance), task and format,
      units and witness, split with `held_set_sha256`, exposure_streams, arms with zero_channels, declared_changed and
      teaching_withheld, primary, gates, guards, checkpoints, worlds, budget, admission_thresholds, testbed grids, files,
      lineage and predictions. Content rules: `minifly-e3-design`.
- [ ] Fix the held set (`manifests/HELD.jsonl`, plus OLD and NEW manifests) before any stream exists, and commit the
      manifests with the lock. `MFC init` fills `split.held_set_sha256` and `split_sha256` when the manifests exist
      (formulas: SCHEMAS.md section 1). AMD-05 recomputes both at every stage and at every commit after the lock, so
      editing, re-pinning or deleting a manifest needs an amendment; LOCK-06 requires every TAUGHT stream to be
      committed after the lock; LAD-01 and the scorer accept only manifest items in HELD/OLD/NEW rows.
- [ ] Write `admit.py` (evaluator, endpoint-locked) and list it, the generators and the evaluator files in
      endpoint_files_sha256.
- [ ] Register predictions: at least one on the primary and one per guard, with base_rate_forecast, p within
      [0.02, 0.98], and basis_findings that exist in FINDINGS or 'none' (LOCK-11).
- [ ] Record `AUTHORIZATION.json` from Mr. Zhou's words ([AUTHORIZATION.md](AUTHORIZATION.md)).
- [ ] `MFC gate --stage lock <dir>` (SELF-01, SCHEMA-01, TIER-01, LOCK-01..04, LOCK-08..13, AMD-01..05, AUT-01..04).
- [ ] Commit ENDPOINT_LOCK.json in a commit that adds nothing under `results/` (LOCK-06), with your `Acting-Agent:`
      trailer.

Reads: CONSTRAINTS.json, FINDINGS, the parent delivery. Writes: ENDPOINT_LOCK.json, AUTHORIZATION.json, params.json,
admit.py, manifests, checks/GATE_lock_<utc>.json.
Exit: lock gate PASS, and ENDPOINT_LOCK committed before any `results/**`.

## C0 Testbed screen (before any learner code)

- [ ] `MFC launch --stage testbed <dir> -- python3 -I admit.py --testbed`. `launch` runs the lock bundle first and
      records the attempt in `runlog/ATTEMPTS.jsonl`. A testbed attempt runs CEILING_*/FLOOR_* arms on technical worlds
      only and writes only under `results/admission/` (TAUGHT streams of any world only in cycle 0, before any learner
      attempt). Anything else, such as dev rows or learner-arm receipts, makes launch exit 1 and CYC-01 FAIL for good.
- [ ] `MFC score --role admission <dir>`. The checker scores the CEILING_* and FLOOR_* rows with the delivery's own rule.
- [ ] Commit the results in a commit separate from the lock.
- [ ] `MFC cycle close <dir> --kind testbed --decision {continue|revise|stop|escalate} --reason "<one sentence>"`.

Reads: ENDPOINT_LOCK (split, budget, testbed grids). Writes: `results/admission/OUTPUTS.jsonl` (CEILING_<cfg> and
FLOOR_<cfg> rows), the TESTBED result, `runlog/ATTEMPTS.jsonl`, `runlog/CYCLES.jsonl`.
Exit: at least one ceiling passes and every floor fails: continue. Otherwise the verdict is
CANNOT_TEST_CLAIM_AS_SPECIFIED: stop, or amend with reason testbed_invalid (no UD is needed while no learner-arm
outcome exists, unless the amendment raises objective_distance).

## C1 Technical

- [ ] Implement the native modules and the runner to the runner contract
      (`minifly-e3-design`, references/ADMISSION_AND_RUNNER_CONTRACT.md).
- [ ] `MFC lock build <dir>` writes SOURCE_LOCK.json. Commit it before any run that uses this closure. Rerun it and
      commit again after every closure or params change.
- [ ] `MFC launch --stage technical <dir> -- <cmd ...>`: one technical record per arm, on technical worlds only. The
      technical bundle adds LOCK-05..07 and CYC-01.
- [ ] Produce the directional and gate fixtures (`results/admission/FIXTURES.jsonl`) and ADMISSION.json with admit.py,
      run through `MFC launch --stage technical <dir> -- <cmd ...>`: every file under `results/` must appear in an
      ATTEMPTS row (RUN-01), and ADMISSION.json is assembled from the raw rows, never by hand.
- [ ] `MFC gate --stage roster <dir>`: adds ADM-*, RED-01..11, DEC-01, CYC-02, RUN-01 and RUN-03 (both WARN at T1);
      at T2 also WIT-01 and REV-01..07 (or DEBT-01).
- [ ] Commit ADMISSION.json before the first dev output (ADM-01).
- [ ] `MFC cycle close <dir> --kind technical --decision ... --reason "..."`.

Reads: ENDPOINT_LOCK, params.json, CONSTRAINTS. Writes: SOURCE_LOCK.json, `results/receipts/<world>/<arm>.json`,
`manifests/TAUGHT/<world>/<arm>.jsonl`, admission OUTPUTS rows, FIXTURES.jsonl, ADMISSION.json, ATTEMPTS and CYCLES rows,
checks/GATE_roster_<utc>.json.
Exit: a PASS roster GATE report at the current SOURCE_LOCK. At T2 this includes the review event.

## T2 review event (the other agent family)

- [ ] Relay `<commit sha> <path>` of the lock and ADMISSION to the reviewer. Nothing else is needed in the message.
- [ ] The reviewer runs the executed checks, the cheapest-shortcut attempt and `MFC canary <dir> --seed HEX`, with the
      seed chosen after reading the lock, and commits `review/REVIEW.json` attesting the lock sha on the reviewer's own
      branch (`minifly-independent-review`).
- [ ] Reconcile every finding before launching dev. A REFUTED BLOCKER needs the reviewer's countersign or an owner ruling.
- [ ] If the other family is unavailable: `MFC ledger append --ledger REVIEW_DEBT --row ROW.json`. Debt allows roster
      spend labelled DEV or PILOT only, and blocks SEALED_CONFIRM, findings activation, adoption and pointer updates
      until a real cross-family review repays it. At 3 or more open debts program-wide no new T2 roster starts. A
      same-family subagent review is debt, not review.

Writes (reviewer): review/REVIEW.json, review/*.py and outputs, checks/CANARY_<utc>.json.
Exit: decision ADMIT, no open BLOCKER or MAJOR (REV-01..07, WIT-01).

## C2 Dev

- [ ] `MFC launch --stage dev <dir> -- <cmd ...>`. `launch` runs the roster bundle and refuses attempts beyond the cycle cap.
      The command may name dev worlds only; it writes `results/dev/` and the dev worlds' receipts and TAUGHT only.
- [ ] `MFC score --role dev <dir>`. Outcomes are readable from here on.
- [ ] A learner change is allowed; rerun `MFC lock build <dir>` and commit. Each new (closure, params) pair adds to
      designs_tried_k.
- [ ] `MFC cycle close <dir> --kind dev --decision ... --reason "..."`.

Writes: dev OUTPUTS rows and receipts, `results/dev/SCORES.json`, ATTEMPTS and CYCLES rows.
Exit: `results/dev/SCORES.json` written and the decision recorded.

## C3 Rehearsal

- [ ] Use a fresh dev world that no earlier dev or technical row, receipt or attempt argv mentions (CYC-03).
- [ ] Freeze the design: no closure or params change from here to official.
- [ ] `MFC launch --stage rehearsal <dir> -- <cmd ...>`: one full history per arm, with the argv template of the
      official run minus the world arguments.
- [ ] Optionally `MFC score --role rehearsal <dir>`. If SEALED_CONFIRM or adoption is intended, run `MFC oc <dir>`
      (OC-01 needs P(pass | target) of at least 0.8).
- [ ] `MFC gate --stage official --dry <dir>`.
- [ ] `MFC cycle close <dir> --kind rehearsal --decision ... --reason "..."`.

Writes: rehearsal OUTPUTS rows and receipts, checks/DRY_official_<utc>.json, ATTEMPTS and CYCLES rows.
Exit: a PASS `official --dry` GATE report.

## S6 Official (always T2)

- [ ] The reviewer acks the final SOURCE_LOCK sha in review.source_lock_acks, in a reviewer-namespace commit (WIT-01).
- [ ] AUTHORIZATION stage official is granted in Mr. Zhou's words.
- [ ] SEALED_CONFIRM only, in this order: the reviewer runs `MFC seal commit-nonce <dir>` and commits SEAL_COMMIT;
      Mr. Zhou gives GO after seeing both lock shas, recorded verbatim in AUTHORIZATION.stages.official and committed
      after SEAL_COMMIT; the reviewer runs `MFC seal reveal <dir>` (it appends the CONFIRM_LEDGER row); official world
      ids come from the seal derivation (`MFC seal derive <dir>`, checked with `MFC seal verify <dir>`). Step detail and
      failure handling: `minifly-independent-review`, references/SEAL_PROTOCOL.md. Single-agent mode cannot seal.
- [ ] `MFC launch --stage official <dir> -- <cmd ...>`. `launch` runs the official bundle first.
- [ ] Commit the official outputs. Never rerun. A later official attempt is legal only if every earlier one produced
      zero OUTPUTS rows (RUN-02), or as a resume of an interrupted attempt (next item).
- [ ] Interrupted (nonzero or killed exit, or an incomplete (world, arm) unit set): `MFC launch --stage official
      --resume <dir> -- <same cmd>`. If the launch process itself died with the container, this launch first records
      the lost attempt from its in-flight marker (`runlog/INFLIGHT_<attempt_id>.json`; exit -1, interrupted,
      recovered, rusage lost). The resume runs under the identical ENDPOINT_LOCK and SOURCE_LOCK shas; the runner
      reads MINIFLY_COMPLETED_UNITS and skips those units (never rewrite or duplicate one: RUN-02), and re-runs the
      unit(s) in MINIFLY_REPLAY_UNITS, chosen by launch, into MINIFLY_REPLAY_DIR (`results/official_replay/`) with a
      canonical row digest equal to the original (RUN-04). Keep the command identical (CYC-03 compares the template);
      the resume is driven by the environment. CLAIM.provenance records interrupted, resume_attempts, resumed_units
      and replay_units. A resume that breaks a rule FAILs RUN-02 or RUN-04 for good (ATTEMPTS is append-only).
      A recovered attempt (rusage lost) needs one of its own units replayed with the identical digest before publish
      (RUN-04 since 1.0.1, spec 20.17); a resume replays one even when no unit is missing, and launch chooses it
      among the recovered attempt's units.

Writes: review/SEAL_COMMIT.json and review/SEAL_REVEAL.json (reviewer), official OUTPUTS rows (`.jsonl`, `.jsonl.gz`
or inside RESULT_BUNDLE*.zip) and receipts, checks/GATE_official_<utc>.json, ATTEMPTS row (with the units it wrote and
their canonical digests), the runtime closure runlog/RUNTIME_CLOSURE_<attempt_id>.json; a resume adds
`results/official_replay/OUTPUTS.jsonl`.
Exit: exactly one completed official run (one attempt, or a legal resume chain).

## S7 Audit and search

- [ ] `MFC score --role official <dir>` writes `results/official/SCORES.json` and `results/official/VERDICT.json`.
- [ ] `MFC claim <dir>` writes CLAIM.json with the generated headline and appends the DELIVERY_LEDGER row. The first
      non-empty line of REPORT.md is that headline; the prose below is free (`minifly-claims-knowledge`).
- [ ] SEALED_CONFIRM or adoption: the reviewer runs `MFC review replay-plan <dir> --seed HEX --n 4` and replays those
      (world, arm) histories with the locked command (REV-08).
- [ ] Predictions are scored into CLAIM.predictions_scored; an unopened endpoint is scored unresolved.
- [ ] Propose 0-3 findings in CLAIM.findings_proposed (zero needs findings_none_reason). Only the other family activates them.
- [ ] File incidents: `MFC ledger append --ledger INCIDENTS --row ROW.json`.
- [ ] T2: run `MFC gate --stage publish --dry <dir>` and hand the CLM-17P WARN sentences to the reviewer, who
      adjudicates each in REVIEW.fm22_prose_adjudications (REV-09); reword or remove any sentence judged overclaim.
- [ ] `MFC gate --stage publish <dir>`; for a pointer update also `MFC gate --stage pointer --pointer P <dir>`.
- [ ] Return to S0 with what you learned.

Writes: SCORES.json, VERDICT.json, CLAIM.json, REPORT.md, checks/GATE_publish_<utc>.json, KB ledger rows, ERRATA.jsonl
only if a published verdict changes.
Exit: publish gate PASS.

## Cycle bookkeeping (all test cycles)

- One `MFC cycle close` per cycle; its exit criterion is checked by CYC-02 when the decision is continue (testbed: a
  TESTBED result; technical: a PASS roster report at the current SOURCE_LOCK; dev: `results/dev/SCORES.json`;
  rehearsal: a PASS `official --dry` report).
- The cap `AUTHORIZATION.caps.max_cycles` (default 3) counts technical, dev and rehearsal cycles; testbed is not counted (CYC-01).
  `launch` refuses technical, dev and rehearsal attempts beyond it.
- Stage scope (CYC-01, every tier): each stage writes only its results directory (testbed and technical:
  `results/admission/`; dev, rehearsal, official: their own) and the receipts, TAUGHT and rows of its own worlds; a
  testbed attempt writes CEILING_*/FLOOR_* rows only. An OUTPUTS row names the attempt that wrote it
  (MINIFLY_ATTEMPT_ID); at T2 a row naming no attempt FAILs RUN-01. `launch` refuses a command line that names another
  stage's world.
- A close covers only the attempts of its own kind since the last close. Attempts no close covers form an open cycle
  that counts against the cap; `launch` refuses an attempt of another stage while a cycle is open, and the official
  stage (gate or launch) needs every testbed, technical, dev and rehearsal attempt in a closed cycle (CYC-01).
- runlog/CYCLES.jsonl and runlog/ATTEMPTS.jsonl are append-only: CYC-01 fails when a committed row is edited or removed.
- The same rule runs at publish as CYC-04 (checker 1.0.1, spec 20.16): an ATTEMPTS or CYCLES row edited after the
  official gate still fails the publish gate.
- One launch per phase; never chain phases inside one unlaunched script. Every `launch` re-verifies ENDPOINT_LOCK,
  SOURCE_LOCK and every closure sha before it starts the command (and again after it ends), so a multi-phase run that
  calls launch once per phase turns a mid-run fix into a loud refusal at the next phase instead of a silent change.
- Every counted run (technical, dev, rehearsal, official) writes runlog/RUNTIME_CLOSURE_<attempt_id>.json with the
  runner-contract helper (every repo file in sys.modules plus the declared data assets; call it after the imports and
  again at the end). `launch` records it in produced_paths; RUN-03 (T2 FAIL, T1 WARN) needs every repo .py file in it
  to be in that attempt's SOURCE_LOCK.closure_sha256 with the same sha ('closure gap' otherwise) and every data asset
  declared (files.assets, files.params, endpoint_files_sha256 or SOURCE_LOCK).
- Failed attempts are never deleted. They count in ATTEMPTS, in designs_tried_k and in compute totals.
