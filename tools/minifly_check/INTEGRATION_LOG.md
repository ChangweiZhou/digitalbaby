# MRK-1 integration log (Phase 0, SPEC 20.3)

Every edit to `EXPECTED.json` or to a corpus fixture made while integrating WP1-WP10 is listed here with the
SPEC clause that justifies it, the before/after, and why it is not a loosening. Fixtures are generated: every
fixture change below is a change to `corpus/_build/*.py`, followed by `build_corpus.py` and
`build_corpus.py --check` (byte-identical rebuild). No fixture directory was edited by hand. No `must_fire`
id was removed from any EXPECTED entry, no entry was deleted, and no `planned_must_fire` list was needed.

The independent verifier should re-read sections A (fixture and EXPECTED edits) and B (RULES.json edits)
for loosening. Section C lists checker-side interpretation rulings, which change code rather than tests;
they are recorded so the same verifier can check them against the SPEC as well.

Round 1 integrator: Claude Code session (claude), 2026-10-24.
Round 2 integrator: Claude Code session (claude), 2026-10-24 (verifier REVISE: blocking issues 1-3, sections
A9, A11, A12, B3, C17-C20, D).
Round 3 integrator: Claude Code session (claude), 2026-10-24 (verifier REVISE: blocking issues 1-2 and the
non-blocking findings; sections A9 (recount), A10, A13, B5, C21-C25).
Round 4 (hardening) implementer: Claude Code session (claude), 2026-10-24 (final verifier REVISE: the CLM-04 blocking
issue, probe 2c, the dev_touched finding; plus the RUN-03 runtime closure, resumable official runs with RUN-04 and the
per-phase lock rule; sections A10, A14-A16, B6-B8, C26-C33; SPEC 20.5-20.9).
Self-improvement round 1 fixer: Claude Code session (claude), 2026-10-24 (held-out pilot HB-20261024-PILOT misses
INC-HB1-A3, INC-HB1-C2, INC-HB1-C3 and the open verifier items; CHECKER_VERSION 1.0.1; sections A10, A17-A19, B9-B11,
C34-C46; SPEC 20.10-20.18). The canaries were not read beyond the three CANARY_NOTE.md files the task named; no fix
matches a canary name, field, layout or literal.
Round-1 repair implementer: Claude Code session (claude), 2026-10-24 (verifier REVISE on self-improvement round 1:
blocking issues CLM-17 key/value shapes, ADM-EXPO packaging, role 'candidate'; the 'rung' overfitting item; most
non-blocking findings; sections A20, A21, B12, C47-C56; SPEC 20.10-20.15, 20.18 amended and 20.19). Nothing under
corpus/heldout/revealed/ was read; the verifier's probe files were not copied (fixtures re-implement each probe class
with new names, fields and layouts).

## A. Fixture and EXPECTED.json edits

### A1. Committed CANARY report shape (all full-timeline fixtures)

- Files: `corpus/_build/base_artifacts.py` (`canary_report`), `base_timeline.py` (`c_review` passes the lock
  shas). Affects `checks/CANARY_20261101T143000Z.json` in every fixture built on the full base timeline.
- SPEC: Section 2 (`checks/CANARY_<utc>.json` is the canary report), 7.1 (`MFC canary <dir> --seed HEX`
  writes it), 7.4 REV-05 ("canary.detected == canary.n. The CANARY report commit follows the ENDPOINT_LOCK
  commit").
- Before: `{schema: minifly.canary_report.v1, delivery, seed, utc, n, detected, mutants[]}`.
- After: the same, plus the fields `MFC canary` writes (`n_requested`, `n_mutants`, `checker_sha256` (the
  empty-kit digest, as for the committed gate reports), `checker_version`, `git_mode`, `git_head`, both lock
  shas, `tier`, `decisive_role`, `applicable_fraction`, `operators`, `results` CAN-01..03 PASS, `exit_code` 0,
  `verdict` PASS).
- Why not a loosening: the fixture's report claimed the canary report schema but lacked the fields the
  checker's own report carries. REV-05 reads the report as `MFC canary` writes it, which is stricter than the
  REV-05 text. No EXPECTED entry changed. S19, which exercises REV-05 (REVIEW says 14 of 15 detected), still
  fires it.

### A2. KB overlay narrowing row id (all fixtures with the KB overlay)

- File: `corpus/_build/base_files.py` (`FINDINGS_BASE`); FIXTURE.json mutation texts of C24 and S23
  (`table.py`).
- SPEC: Section 8 KB ledgers, FINDINGS `{id F-<yyyymmdd>-<agent>-<n> or seed F-UF-xx, ...}`; KB-01 (ledger
  row schemas are valid).
- Before: narrowing row id `F-UF-15-N1`, which is neither id form, so KB-01 FAILed on every clean fixture.
- After: `F-20261024-claude-1`, which is the id the real seed KB uses for the same narrowing row. All other
  content is unchanged.
- Why not a loosening: only the id is made SPEC-valid. C24 still fires PTR-04 (F-UF-15 quoted without its
  narrowing row) and S23 still fires GOV-APPEND (the row edited in place).

### A3. K01 DELIVERY_LEDGER `claim_sha256`

- File: `corpus/_build/fixtures_g.py` (`b_k01`).
- SPEC: Section 8 DELIVERY_LEDGER `{..., claim_sha256}`; CLM-13 ("A DELIVERY_LEDGER row was appended by
  `claim`", so the row links its CLAIM by sha); Section 11 `kb score`.
- Before: `claim_sha256` `'0' x 64` in all three rows. No `claim` run can write that, and `kb score` (which
  only scores rows whose CLAIM matches) excluded all three deliveries.
- After: `claim_sha256` = sha256 of each committed CLAIM.json.
- Why not a loosening: K01's EXPECTED entry is unchanged (`/brier_mean` 0.353333, `/skill` contains
  INSUFFICIENT_N). The fixture now carries ledger rows that `claim` could have written.

### A4. Replay digest in C00d and S22

- File: `corpus/_build/fixtures_d.py` (`replay_digest`, used by C00d and S22).
- SPEC: 7.4 REV-08 ("postrun.replay digests equal the digests of the official rows for the same (world,
  arm)"); Section 10 (replay seeds).
- Before: sha256 of the sorted lines `pool|checkpoint|item_id|stream_hex`. This digests one field, not the
  rows.
- After: sha256 over the canonical JSON of each official row, restricted to the OUTPUTS schema fields except
  `attempt_id` (which necessarily differs between attempts), sorted by (pool, checkpoint, item_id), each
  followed by a newline. This equals `review.row_digest`; SCHEMAS.md section 13 now states it.
- Why not a loosening: the digest covers more fields. S22 still fires REV-08 (one observed digest
  `'0' x 64`).

### A5. C00d dev behaviour (OC 0.9)

- File: `corpus/_build/fixtures_d.py` (`b_c00d`).
- SPEC: 12.2 row C00d ("complete seal, OC 0.9, replay"); 9.8 (binomial counts at p_comp and p_comp +
  target_delta, between-world SD from dev d_w); 7.4 OC-01; 13.1 `oc_min_power` 0.8.
- Before: the base dev worlds (W misses 2 and 3 of the 16 held items in w_dev_1 and w_dev_2; mean dev d 0.281).
  Under the SPEC 9.8 procedure that gives P(pass | target) = 0.790 < 0.8, so OC-01 FAILs and the fixture
  cannot be the clean SEALED_CONFIRM row of the table.
- After: in C00d only, W misses one held item fewer in each dev world (w_dev_1 `([1], [3])` -> `([1], [])`,
  w_dev_2 `([5], [6, 7])` -> `([5], [6])`). Mean dev d is 0.344 and P(pass | target) is about 0.94. Official
  worlds keep the base behaviour, so every official number, the VERDICT and the headline are unchanged.
- Why not a loosening: OC-01 and decide.oc are unchanged. C21 (OC 0.41, 4 worlds) still fires OC-01. The
  change makes the fixture match its SPEC row.

### A6. C00d official ATTEMPTS row records the runner-written TAUGHT manifests

- File: `corpus/_build/base_timeline.py` (`stage_run`: TAUGHT files the run writes join `produced_paths`).
  It affects C00d only, the one fixture whose run writes TAUGHT manifests.
- SPEC: 6.2 item 7 (runner contract: TAUGHT manifests are written per arm); Section 8 ATTEMPTS (written only
  by `launch`, `produced_paths`); Section 10 (sealed world ids exist only after the reveal); CLM-08 (files
  added after the official lock are POSTHOC and never VERDICT inputs).
- Before: the derived worlds' TAUGHT manifests appeared in the official commit without any ATTEMPTS record,
  so CLM-08 FAILed on the clean SEALED_CONFIRM fixture. A sealed delivery cannot have them earlier.
- After: `launch` records new or changed `manifests/TAUGHT/**` files (checker change, section C14), and the
  fixture's official ATTEMPTS row lists them, as `launch` would.
- Why not a loosening: CLM-08 is unchanged. S29 (a VERDICT input that no attempt produced) fires it.

### A7. C00f rebuilt as a runnable governance range

- File: `corpus/_build/fixtures_g.py` (`b_c00f`, new `LOCKEXTRA`, `_nested`); `mrk_util.fixture_files`.
- SPEC: 13.2 ("Check code edit where fired(head) is a superset of fired(base) on all fixtures and clean
  fixtures stay clean: tighten_or_neutral"; "For code changes, the base checker is extracted ... and both
  versions are run on both corpora"; heading: "unclassifiable counts as loosen"); 12.2 row C00f ("add a check
  plus fixture, with GOV-REV record: governance exit 0").
- Before: the miniature kit was declared "never executed" (a stub `lockcheck.py`, a three-rule RULES.json,
  nested fixtures with only FIXTURE.json). Governance could not run base or head, so the code edit was
  unclassifiable (= loosen) and GOV-CLASS, GOV-UD, GOV-REV and GOV-MONO fired on a clean fixture.
- After: the base tree holds a full RULES.json copy, a nested EXPECTED.json and a runnable lock-stage nested
  fixture X01 (fires LOCK-08). The head commit adds check LOCK-14 (new module `lockextra.py` plus its RULES
  entry), a runnable nested fixture X02 (fires LOCK-14) and its EXPECTED entry, citing the claude review
  record. Governance overlays these onto the running kit and runs both checkers on both nested corpora.
- Why not a loosening: C00f's EXPECTED entry is unchanged (exit 0; every GOV id must not fire). GOV-MONO now
  observes real fired sets, instead of failing for lack of anything to run. G01-G05 and S23-S25 keep the
  miniature kit and still fire what they expect.

### A8. New supplementary fixtures S26-S45 (EXPECTED entries added)

- Files: `corpus/_build/fixtures_s2.py` (new), `table.py` (entries), `mrk_util.write_fixture` (files beside
  `repo/`, used for the S44/S45 owner key), `base_items.py` (`w_new_const` behaviour key, default off),
  `fixtures_d.b_c00e` (accepts extra options).
- SPEC: 20.2 ("Every check id in Section 7.4 is either implemented and exercised by the corpus, or
  registered in RULES.json with status planned"); 7.7 (`selftest --mutate`: some dirty fixture must notice
  an always-PASS replacement, and some clean fixture an always-FAIL one).
- Why: the first `selftest --mutate` reported 32 ids that were not exercised or had no clean coverage.

| Id | Kind | Defect or configuration | Exercises |
|---|---|---|---|
| S26 | dirty | paired_lcb without alpha (the rule is undecidable) | RED-02..07 |
| S27 | dirty | T1 ADMISSION cites a parent admission, lineage.parent null | ADM-REUSE |
| S28 | dirty | committed CLAIM level_earned E4; publish without `claim` | CLM-03 |
| S29 | dirty | results/official file added after the official run | CLM-08 |
| S30 | dirty | committed gate report with a foreign checker sha | CLM-14 |
| S31 | dirty | CONSTRAINTS without tier_thresholds | TIER-01 |
| S32 | dirty | other-family activation of a finding from a DEV delivery | KB-04 |
| S33 | dirty | governance range with a locally modified kit file | GOV-SELF |
| S34 | dirty | legacy pointer target without a KB/RETRO annotation | PTR-07 |
| S35 | clean | legacy pointer target with UD and RETRO annotation | PTR-01..07, CLM-02, DEBT-02 clean coverage |
| S36 | dirty (WARN) | REPORT quotes a prior-work number without an F-id | CLM-16 (must_warn) |
| S37 | dirty (WARN) | candidate answers one constant on NEW in the dev worlds | LAD-04 (must_warn) |
| S38 | dirty (WARN) | earlier delivery's HELD manifest overlaps 4 held items | WLD-02 (must_warn) |
| S39 | dirty (WARN) | newer v1 delivery in the same track at the pointer stage | PTR-06 (must_warn), PTR-02 |
| S40 | dirty | canary on C02a (the M01 target already blocks) | CAN-02 |
| S41 | dirty | canary on a lock-only delivery | CAN-03 |
| S42 | clean | single-agent T2 dev delivery with a REVIEW_DEBT row (SPEC 5) | DEBT-01 clean coverage |
| S43 | dirty | write-counter site outside the recorded closure | CAN-01 |
| S44 | clean | held-out mechanism: valid commitment, reveal | HELD-01..03 clean coverage |
| S45 | dirty | tampered bundle, late commitment, same author | HELD-01..03 |

- S44/S45 test the commitment, seal and reveal mechanics of SPEC 12.3, using a toy one-fixture batch whose
  plaintext is visible in the builder. They are not held-out canaries and never count toward held-out
  recall. The bundle format is re-stated in the builder (stdlib only, no checker import).
- Why not a loosening: only additions, and every new dirty fixture asserts new must_fire or must_warn ids.

### A9. EXPECTED order-only changes

- C00a, C00b and C00d `must_not_fire`: WIT-01 moved position (it is now listed with the review group, B1).
  The sets are identical.
- The same reordering changes the bytes of every committed gate report the builder writes from the RULES
  bundles (`BA.gate_report`): 259 `checks/GATE_*.json` / `checks/DRY_*.json` files (repo/ and history/ copies)
  differ from the pre-round-1 corpus only in the order of `results[]` (identical result sets, every other
  field equal), and 155 `runlog/CYCLES.jsonl` files differ only in `computed.latest_gate.sha256`, the sha256 of
  the reordered report. Counted in round 2 by comparing the pre-round-1 snapshot with the corpus as rebuilt.
- Round 3 recount (correction): the round-2 counts above were wrong. Rerunning the verifier's comparison script
  (`/tmp/verif/cdiff.py`) against the same pre-round-1 snapshot gives 254 GATE/DRY files that differ only in
  the order of `results[]` and 152 CYCLES files that differ only in `latest_gate.sha256`, as the round-2
  verifier found. Every other differing file is covered by A1-A12. The round-3 corpus gives the same counts,
  and its only additions are the new fixture directories S51-S55 (A13).

### A10. Builder self-tests (`corpus/_build/test_build_corpus.py`)

- `SUPPLEMENTARY` now covers S01-S45.
- A dirty fixture may be advisory-only: it must WARN, and it never blocks.
- The clean-fixture hygiene scan skips nested fixtures of a governance fixture, which carry their own dirty
  kind (C00f X01 holds the LOCK-08 literal on purpose).
- None of these is a fixture or EXPECTED change.
- Round 2: `SUPPLEMENTARY` covers S01-S50.
- Round 3: `SUPPLEMENTARY` covers S01-S55.
- Round 4: `SUPPLEMENTARY` covers S01-S64.
- Self-improvement round 1: `SUPPLEMENTARY` covers S01-S80.

### A11. New supplementary fixtures S46-S50 (round 2; EXPECTED entries added)

- Files: `corpus/_build/fixtures_s3.py` (new), `table.py` (entries). No existing builder changed; every other
  fixture and EXPECTED entry is byte-identical (`build_corpus.py --check`, and a semantic diff of EXPECTED.json
  against the round-1 file shows only the five additions).
- SPEC: 3.2 (every cycle ends with `cycle close`; the cap counts technical, dev and rehearsal; launch refuses
  attempts beyond the cap); 7.4 CYC-01; 3.3 ("Verdict FAIL, INVALID, INCOMPLETE: Report it"); 8 (CLAIM
  primary_gate includes INVALID; invalid_reason, stop_reason); 9.5 step 3 (ADM-CTRL failure gives INVALID
  TREATMENT_NOT_DELIVERED); 9.9 (`<n> <ROLE> worlds`).
- Why: round-2 verifier blocking issues 2 (probe P4: deleting runlog/CYCLES.jsonl left roster and official at
  exit 0) and 3 (probe P2b: an official INVALID was published as 'NOT_OPENED ... 0 DEV worlds').

| Id | Kind | Defect | Commands, exits | Asserts |
|---|---|---|---|---|
| S46 | dirty | committed runlog/CYCLES.jsonl deleted in the worktree (probe P4) | roster 1, official 1 | must_fire CYC-01 |
| S47 | dirty | no cycle ever closed; testbed..official attempts exist | roster 1, official 1 | must_fire CYC-01 |
| S48 | dirty | T1 dev-only delivery (C00e) that never closes a cycle | roster 1 | must_fire CYC-01 |
| S49 | dirty | committed CYCLES row 3 rewritten in a later commit | roster 1 | must_fire CYC-01 |
| S50 | dirty | official-world NPT receipts carry W's writes and digests (control not delivered in the official run only) | score 0, claim 0, publish 0 | `/verdict` INVALID, `/invalid_reason` TREATMENT_NOT_DELIVERED, `/primary_gate` INVALID, `/stop_reason` contains INVALID, `/worlds_role` MAIN, `/n_worlds` 8, `/headline` 'INVALID \| earned NONE (attempted E3) \| ... \| 8 MAIN worlds \| best of k=1' |

- S50 commits the INVALID CLAIM and REPORT the executor would have committed after `claim` (findings_none_reason
  filled), so the publish run tests the honest INVALID publish (SPEC 3.3, I10), not a first-time claim. The
  publish exit 0 is the SPEC outcome: an honest INVALID is reported, never hidden; nothing in the publish bundle
  re-runs ADM-CTRL, and the claim now states INVALID.
- Why not a loosening: only additions; S46-S49 assert a new must_fire. S50 asserts seven claim and verdict
  values; the round-1 checker got five of them wrong (primary_gate NOT_OPENED, stop_reason null, worlds_role
  DEV, n_worlds 0, and the headline), while `/verdict` and `/invalid_reason` were already right.

### A12. Selftest freezes committed fixture content (harness, round 2)

- Files: `common.py` (`TimelineGit.freeze`), `governance.py` (`run_fixture` calls it on each temp copy before
  the first command). No fixture or EXPECTED entry changed.
- SPEC: 7.3 (TimelineGit is a read-only view of history, like RealGit); 7.7 (selftest compares the checker with
  EXPECTED on each fixture).
- Before: a fixture keeps the committed version of a path in repo/ whenever it equals the final tree, so a
  command that rewrote a committed file in place (for example `claim` rewriting CLAIM.json) also rewrote the
  "committed" version. Worktree-vs-HEAD checks (ERR-01, KB-02, CYC-01, dirty paths) could not see such edits in
  selftest, although real git would.
- After: every such path gets its history entry before the commands run, so the committed bytes stay fixed.
- Why not a loosening: the selftest now sees strictly more edits. All 103 fixtures pass unchanged with it.

### A13. New supplementary fixtures S51-S55 (round 3; EXPECTED entries added)

- Files: `corpus/_build/fixtures_s4.py` (new), `table.py` (entries), `test_build_corpus.py` (SUPPLEMENTARY).
  No existing builder changed. A semantic diff of EXPECTED.json against the round-2 file shows only the five
  additions; `build_corpus.py --check` rebuilds all 108 fixtures byte for byte.
- SPEC:
  - 3.1 stage map (C0 testbed: "evaluator-side ceiling and floor grids"; C2 dev worlds; C3 rehearsal);
  - 3.2 ("testbed (C0, not counted against the cap)", "launch refuses new dev or rehearsal attempts beyond the
    cap");
  - 6.2 ("The held set is fixed in the lock (split.held_set_sha256) before any stream exists");
  - 1 I9 (frozen designs and data are never edited);
  - 7.4 AMD-05, LAD-01, CLM-15 ("both locks and GATE reports").
- Why:
  - round-3 verifier blocking issue 1 (probe pD: dev work relabelled as testbed bypassed the cap);
  - blocking issue 2 (probes F and E2: an edited HELD manifest, and out-of-manifest held items, passed every
    gate);
  - the deletion sweep (a deleted committed gate report left publish at exit 0).

| Id | Kind | Defect | Commands, exits | Asserts |
|---|---|---|---|---|
| S51 | dirty | after the rehearsal (cap of 3 used up), a testbed attempt reruns the dev runner on w_dev_1 and rewrites W's dev rows; a testbed CYCLES row covers it (probe pD) | roster 1, official 1 | must_fire CYC-01 |
| S52 | dirty | after the official run, a commit edits item h05 of manifests/HELD.jsonl (probe F) | lock 1, publish 1 | must_fire AMD-05, LAD-01 |
| S53 | dirty | in every official world, item h01 is replaced by the out-of-manifest 2+7= in every arm, with consistent shas (probe E2) | score 1, publish 1 | must_fire LAD-01 |
| S54 | dirty | after the official run, a commit deletes manifests/OLD.jsonl | publish 1 | must_fire AMD-05 |
| S55 | dirty | after the official run, a commit deletes the committed roster gate report (cycle 2's latest_gate) | score 0, claim 0, publish 1 | must_fire CLM-15 |

- In S53, `score` exits 1 because the scorer refuses rows outside the locked manifests (C22). VERDICT is never
  computed from an item the lock did not fix.
- Why not a loosening: these are additions only, and each asserts a new must_fire.

### A14. New supplementary fixtures S56-S64 (round 4; EXPECTED entries added)

- Files: `corpus/_build/fixtures_s5.py` (new), `table.py` (entries), `test_build_corpus.py` (SUPPLEMENTARY S01-S64),
  `base_items.py` (behaviour key `frozen_like_w_held`, default absent), `scores_doc.py` and `mutate.py` (A16).
- A semantic diff of EXPECTED.json against the round-3 file shows only the nine additions; no existing entry changed.
- SPEC: 20.5 (CLM-04 tokens; 7.4 CLM-04; I6; 3.4), 20.6 (9.5 decision comparator), 20.7 (RUN-03), 20.8 (RUN-02 as
  amended, RUN-04), 7.7 (`selftest --mutate`: a dirty fixture must notice an always-PASS replacement, a clean one an
  always-FAIL one).

| Id | Kind | Defect or configuration | Commands, exits | Asserts |
|---|---|---|---|---|
| S56 | dirty | C00e (T1, DEV) + SUMMARY.json {"verdict": "GAIN_CONFIRMED", "axis_confirmed": true} (verifier probe a) | score 0, claim 0, publish 1 | must_fire CLM-04 |
| S57 | dirty | C00e + REPORT body 'NativeConfirmation achieved on dev.' (probe f) | score 0, claim 0, publish 1 | must_fire CLM-04 |
| S58 | dirty | C00e + STATUS.md 'Reuse con<U+200B>firmed on the held set.' (probe g) | score 0, claim 0, publish 1 | must_fire CLM-04 |
| S59 | clean | the undeclared FROZEN arm answers every official HELD item like W (probe 2c) | official 0, score 0, claim 0, publish 0 | `/verdict` FAIL, `/primary_gate` FAIL, `/level_earned` NONE, per-world comparator FROZEN (worlds 0 and 7), `/primary/implied_comparator_arms` [SHUF, FROZEN], headline ending '\| FROZEN >= candidate' |
| S60 | dirty | bootstrap.py committed with the learner, reached only at runtime; the official runtime closure lists it | official 1, score 0, claim 0, publish 1 | must_fire RUN-03 |
| S61 | clean | official run split into a killed attempt (w_off_1..5) and a legal `--resume` (w_off_6..8) with a bit-identical replay of (w_off_1, W) | official 0, score 0, claim 0, publish 0 | `/verdict` PASS, base headline, `/provenance/interrupted` true, `/provenance/resumed_units` (15 units), `/provenance/replay_units` [[w_off_1, W]] |
| S62 | dirty | as S61, and the resume also rewrites the completed unit (w_off_1, NPT) and its receipt | official 1, score 0, claim 0, publish 1 | must_fire RUN-02 |
| S63 | dirty | as S61 without any replay rows | official 1, score 0, claim 0, publish 1 | must_fire RUN-04 |
| S64 | dirty | as S61, the replay answers HELD h02 of (w_off_1, W) with 99 | official 1, score 0, claim 0, publish 1 | must_fire RUN-04 |

- Probes against the round-3 checker (the scratchpad copy of the kit before this round): S56, S57 and S58 exit 0 at
  publish (CLM-04 NOT_APPLICABLE) and S59 gives VERDICT PASS with the headline suffix 'FROZEN >= candidate'; S60 exits 0
  at official and publish. With this round's checker S56-S58 fire CLM-04 only, S59 gives FAIL, S60 fires RUN-03 only,
  S62 fires RUN-02 only and S63/S64 fire RUN-04 only (gate --stage official and publish).
- S59 is a clean fixture: an honest FAIL is reported, never hidden (SPEC 3.3, I10). It is the must-not-PASS fixture
  of the comparator amendment; under the round-3 rule it passed.
- S61 is clean under SPEC 20.8. Under the round-3 RUN-02 it would FAIL (two assessment-bearing official attempts); see
  B7 for why that is recorded as a governed RUN-02 relaxation.
- Why not a loosening of the corpus: only additions; every dirty fixture asserts a new must_fire and every clean one
  asserts values the round-3 checker got wrong (S59) or a configuration it could not express (S61).

### A15. The base writes a runtime closure for every counted attempt (round 4; all full-timeline fixtures)

- Files: `corpus/_build/base_timeline.py` (`runtime_closure`, `add_runtime_closure`; called for the technical run, the
  technical admit attempt, every dev attempt, the rehearsal and the official attempt; options `runtime_extra`,
  `no_runtime_closure`, `runtime_closure_patch`).
- SPEC: 20.7 (every counted run writes `runlog/RUNTIME_CLOSURE_<attempt_id>.json`, launch records it in
  produced_paths); 7.4 RUN-01 (launch-recorded files).
- Before: no runtime closure existed. With RUN-03 enforced at T2 (roster, official, publish) every clean T2 fixture
  would FAIL it.
- After: each counted attempt's commit adds `runlog/RUNTIME_CLOSURE_<attempt_id>.json` (the SOURCE_LOCK
  closure_sha256 entries, which is all the corpus stubs import) and its ATTEMPTS row lists that path and sha256. Counted
  with a scratchpad comparison script (`cdiff4.py`) against a copy of the round-3 kit taken before this round:
  363 new runtime closure files (repo/ and history/ copies), 289 ATTEMPTS
  files whose only difference is the added runtime-closure produced_paths entries, and 75 timeline.json files whose
  only difference is that path in the commits that add it.
- Why not a loosening: the runtime closure is new evidence that RUN-03 verifies; no other byte of those files changed.
  S60 (A14) shows RUN-03 firing on a real gap.

### A16. Committed CLAIM.json carries `provenance`; gate reports carry RUN-02/03/04; reference arithmetic (round 4)

- Files: `corpus/_build/mutate.py` (`claim_doc` writes `provenance` {interrupted false, resume_attempts [],
  resumed_units [], replay_units []}; G now comes from `scores_doc.verdict()['mean_g']`), `fixtures_d.py` (`c00c_claim`
  writes the same `provenance`), `scores_doc.py` (the decision comparator of SPEC 20.6 includes SHUF and FROZEN; G
  keeps NPT, the ladder and chance, SPEC 9.6), `base_artifacts.gate_report` (unchanged code: it reads the RULES
  bundles, B6-B8).
- SPEC: 20.8 (CLAIM gains the generated field provenance; CLM-01 requires it, CLM-02 compares it), 7.5 bundles as
  amended in 20.7/20.8, 9.5 as amended in 20.6, 9.6.
- Counted against the round-3 corpus: 75 CLAIM.json files whose only difference is the added provenance field, 73
  DELIVERY_LEDGER.jsonl files and 2 pointer files whose only difference is claim_sha256 (the sha of that CLAIM), 361
  GATE/DRY reports whose only differences are added `results[]` entries and `counts` (RUN-03 in 223 roster-based
  reports, RUN-03 and RUN-04 in 137 official ones, RUN-02, RUN-03 and RUN-04 in 1 publish report; every earlier entry
  is unchanged), 215 CYCLES.jsonl files whose only difference is
  `computed.latest_gate.sha256` (the sha of the regenerated report), and 2 copies of RULES.json in C00f.
- `scores_doc.verdict` change: in every round-3 fixture SHUF and FROZEN score 0 on counted HELD items, so `d` and G
  are unchanged there (no headline, CLAIM number or EXPECTED value moved); only S59, where FROZEN copies W, differs.
- Why not a loosening: the claim schema gains a field, the gate reports gain checks, and the reference comparator gains
  arms; nothing is removed.

### A17. New supplementary fixtures S65-S80 (self-improvement round 1; EXPECTED entries added)

- Files: `corpus/_build/fixtures_r1.py` (new), `table.py` (entries), `test_build_corpus.py` (SUPPLEMENTARY S01-S80). No
  existing builder function changed behaviour (A18 is a report-content guard that changes no byte).
- SPEC: 20.11 (exposure by content), 20.12 (confirm wording as rendered), 20.13 (CLM-17), 20.14 (comparator role
  other), 20.15 (identity), 20.16 (CYC-04), 20.17 (recovered attempt replay); 12.3 step 6 (fixes after a miss must
  generalize: new names and layouts); 7.7 (`selftest --mutate`: a dirty fixture notices always-PASS, a clean one
  always-FAIL).

| Id | Kind | Defect or configuration | Commands, exits | Asserts |
|---|---|---|---|---|
| S65 | dirty | declared non-feedback `manifests/LISTEN/*.jsonl` (flat file) whose `utterance_hex` holds each commuted held prompt, its answer and EOS | roster 1, official 1 | must_fire ADM-EXPO |
| S66 | dirty | declared non-feedback `manifests/PRIMING/*/*.jsonl`, per official world for W: text `cue` with the held S3 prompt and the answer in a separate `reply_hex` | roster 1, official 1 | must_fire ADM-EXPO |
| S67 | clean | as S65 with held prompts only (hex and text caption; integer fields equal the sum) | roster, official, score, claim, publish 0 | `/verdict` PASS, base headline |
| S68 | dirty | OLD-drop-carried FAIL (one OLD item lost at final in every official world); REPORT body 'sits at evidence level E3', 'satisfies the locked decision rule in every MAIN world' | official 0, score 0, claim 0, publish 1 | must_fire CLM-17 |
| S69 | dirty | the same FAIL; overclaims in CLAIM.notes, findings_proposed[0].claim, STATUS.md and SUMMARY.json | official 0, score 0, claim 0, publish 1 | must_fire CLM-17 |
| S70 | clean | the same FAIL with honest negations ('E3 was not reached', 'attempted E3', 'the OLD drop guard failed, so the primary FAILs', 'The OLD acquisition guard passed') | official, score, claim, publish 0 | `/verdict` FAIL, `/level_earned` NONE, `/guards/old_drop` FAIL, `/guards/old_acq` PASS |
| S71 | dirty | after the report, codex proposes F-20261103-codex-4; a later commit (git author kb-bot, trailer 'Acting-Agent: claude', only in origin/codex/minifly-kb-sync) appends its activation (author_agent claude:s-kb-review) | score 0, claim 0, publish 1, kb status 0 | must_fire KB-03; `/invalid_activations_ignored/0/id` F-20261103-codex-4 |
| S72 | dirty | C00f's range with the review record commit only in origin/codex/minifly-corpus-base and upstream/claude/minifly-gov-review | governance 1 | must_fire GOV-REV |
| S73 | clean | as S71 with the reviewer's own commit (trailer claude:s-kb-review, in origin/claude/minifly-kb-review and the codex branch) | score, claim, publish, kb status 0 | `/invalid_activations_ignored` [] |
| S74 | dirty | C00e + REPORT body 'con&#8203;firmed' | score dev 0, claim 0, publish 1 | must_fire CLM-04 |
| S75 | dirty | C00e + STATUS.md 'con**firm**ed' | score dev 0, claim 0, publish 1 | must_fire CLM-04 |
| S76 | dirty | C00e + SUMMARY.md 'con<!-- -->firmed' | score dev 0, claim 0, publish 1 | must_fire CLM-04 |
| S77 | dirty | after the report, a commit rewrites the rehearsal ATTEMPTS row (wall_s, max_rss_kb) | score 0, claim 0, publish 1 | must_fire CYC-04 |
| S78 | dirty | the only official attempt (every unit complete) recorded from its in-flight marker (exit -1, rusage_lost, recovered), no resume, no replay | official 1, score 0, claim 0, publish 1 | must_fire RUN-04 |
| S79 | clean | S61 with attempt A recovered (rusage lost); the resume replays (w_off_1, W), A's unit | official, score, claim, publish 0 | `/verdict` PASS, base headline, `/provenance/interrupted` true, `/provenance/replay_units` [[w_off_1, W]] |
| S80 | clean | FROZEN relabelled role 'other' (not in comparator_arms) answers every official HELD item like W | official, score, claim, publish 0 | `/verdict` FAIL, `/primary/per_world_detail/0/comparator` FROZEN, `/primary/implied_comparator_arms` [SHUF, FROZEN], S59 headline |

- Probes against the round-0 kit (HEAD, checker 1.0.0, extracted with `git archive` into the scratchpad and run on the
  same fixture copies with `governance.run_fixture`): every dirty fixture S65, S66, S68, S69, S71, S72, S74-S78 exits 0
  on every command (nothing fires); the clean S67, S70, S73 and S79 pass both kits; the clean S80 fails publish under
  1.0.0 with ERR-01 (the old comparator leaves the 'other' arm out, so a fresh VERDICT says PASS). Under 1.0.1 each
  dirty fixture fires its must_fire id and nothing else, except S65/S66, where the exposed orbits also become S1 for the
  scorer and STR-01, DET-01 and ADM-TB-CEIL/FLOOR fire as consequences.
- EXPECTED.json semantic diff against HEAD: the 16 additions, and the must_not_fire lists of the 8 clean fixtures with
  a publish command (C00a, C00b, C00c, C00d, C00e, S42, S59, S61) gain CLM-17 and CYC-04 (the builder derives a clean
  fixture's must_not_fire from its commands' bundles). No entry lost anything; no must_fire id moved. Tightening.
- Changed existing fixture files: only the two copies of RULES.json inside C00f (`repo/` and `history/`), which embed
  the running kit's RULES.json (fixtures_g._real_rules): they now carry checker_version 1.0.1, CLM-17, CYC-04 and the
  changed_in lists. C00f stays clean (governance exit 0).
- Incident closures (KB-05): INC-HB1-A3-CL1/CL2 (ADM-EXPO, S65/S66), INC-HB1-C2-CL1/CL2 (CLM-17, S68/S69),
  INC-HB1-C3-CL1/CL2 (KB-03 S71, GOV-REV S72), appended with `MFC ledger append --ledger INCIDENTS`, which ran KB-05's
  closure_problems against EXPECTED.json; `MFC kb status` no longer lists the three incidents as open.
- Why not a loosening: additions only, every dirty fixture asserts a must_fire, and every clean one asserts values.

### A18. Committed corpus gate reports keep checker 1.0.0's bundles

- File: `corpus/_build/base_artifacts.py` (`REPORT_CHECKER_VERSION = '1.0.0'`; `gate_report` lists only bundle ids whose
  RULES introduced_in is at most that version).
- SPEC: 20.10 (versioning); 19.4 gate report (a report records the checks of the checker that wrote it).
- Before: the builder listed every current bundle id in the committed reports it fabricates, so adding CLM-17 and
  CYC-04 to the publish bundle would have rewritten committed reports that claim checker_version 1.0.0.
- After: the reports a 1.0.0 checker wrote keep their 1.0.0 contents; `build_corpus.py --check` is byte-identical for
  every pre-existing fixture except the C00f RULES copies (A17). Not a loosening: no report, check or EXPECTED value
  is weakened; the reports stay what a 1.0.0 run would have written.

### A19. Unit-test fixtures and assertions changed in round 1 (all tightenings or realism fixes)

- `test_admitcheck.test_presentation_only_rows_do_not_count` asserted PASS for a non-feedback TAUGHT row carrying a
  held orbit WITH its answer. SPEC 20.11 makes that a leak: the test now keeps the PASS for the same row without the
  answer and a new test `test_presentation_only_row_with_the_answer_fails` asserts FAIL with the answer (tightening).
- `test_decide.test_implied_comparators_close_declaration_minimization` pinned implied comparators
  [SHUF, FROZEN, LES] with an OTHER arm present; it now expects [SHUF, FROZEN, LES, OTHER] and that an 'other' arm equal
  to the candidate gives FAIL (SPEC 20.14; tightening).
- `test_common.TestRealRules.test_fields_and_statuses` asserted introduced_in 1.0.0 for every rule; it now pins
  introduced_in 1.0.1 for exactly CLM-17 and CYC-04, changed_in [1.0.1] for exactly the 14 tightened ids, and
  CHECKER_VERSION 1.0.1 (SPEC 20.10). SPEC_IDS gains CLM-17 and CYC-04.
- Identity realism (SPEC 20.15): `test_seal` (three tests) committed the reviewer's SEAL_COMMIT without an Acting-Agent
  trailer; `test_lockcheck.TestWit01` committed the reviewer's attestation and ack without one; the Timeline helpers
  of `test_governance` and `test_claimcheck` (used by test_kb) defaulted to no branches. Under commit_family such
  commits have no derivable identity, so the tests' intended failures (reveal before GO, official outputs exist, wrong
  namespace, same family, ...) were masked by an earlier identity FAIL. The reviewer commits now carry the reviewer
  trailer, branch defaults follow the trailer's family, and the 'wrong namespace' cases carry the reviewer trailer on
  an author-namespace commit (a forged identity), so each test still asserts its original failure and message. REV-01,
  KB-03 and GOV-REV messages keep the substrings the tests read ('lock.author_agent says', 'Acting-Agent codex',
  'origin/claude/*', 'no codex-namespace'). No assertion was removed or relaxed.
- New unit tests: test_admitcheck (5 exposure tests), test_scorer (exposure strata), test_claimcheck (TestOverclaim:
  4 tests; CLM-04 markup), test_common (TestCommitFamily: 2 tests; changed_in validation; bundle placement),
  test_launch (CYC-04; recovered attempt replay), test_governance (TestHeldoutPostBatch: 3 tests), test_seal
  (the SEAL_REVEAL commit must act as the reviewer).

### A20. New supplementary fixtures S81-S88 (round-1 repair; EXPECTED entries added)

- Files: `corpus/_build/fixtures_r1b.py` (new), `table.py` (entries), `test_build_corpus.py` (SUPPLEMENTARY S01-S88). No
  existing builder function changed; `build_corpus.py` regenerated the corpus and `build_corpus.py --check` printed
  "byte-identical: 141 fixtures, base and EXPECTED.json" and "valid: ... FM-01..FM-30 covered". The only rewritten
  pre-existing fixture files are again the two C00f copies of RULES.json (they embed the running kit's RULES.json;
  B12). C00f stays clean.
- SPEC: 20.11 (ADM-EXPO packaging), 20.12 (link syntax), 20.13 (key/value shapes), 20.14 (role candidate), 7.7.

| Id | Kind | Defect or configuration | Commands, exits | Asserts |
|---|---|---|---|---|
| S81 | dirty | OLD-drop-carried FAIL (R1 `_fail_build`); STATUS.md key/value block 'evidence_level: E3', 'Decision criterion: satisfied' | official 0, score 0, claim 0, publish 1 | must_fire CLM-17 |
| S82 | dirty | the same FAIL; REPORT quantity/value table with '\| level earned \| E3 \|', '\| locked rule \| satisfied in 8/8 worlds \|' | official 0, score 0, claim 0, publish 1 | must_fire CLM-17 |
| S83 | clean | the same FAIL; STATUS 'Level earned: NONE', 'Level attempted: E3', 'Primary gate: FAIL', 'Decision rule: W passes when ...', 'Guard old_drop: FAIL'; REPORT table level earned NONE, level attempted E3, guards; NOTES.md 'E3 was not reached'; 'The roster gate for this E3 design passed' | official, score, claim, publish 0 | `/verdict` FAIL, `/level_earned` NONE, `/guards/old_drop` FAIL |
| S84 | dirty | declared non-feedback manifests/SESSION/*.jsonl; each row's `body` is a JSON-encoded string with a held prompt under `text` and its answer under `next_text` | roster 1, official 1 | must_fire ADM-EXPO |
| S85 | dirty | declared non-feedback manifests/TAPE/*.tsv; each commuted held prompt on one tab-separated line, its answer on the next | roster 1, official 1 | must_fire ADM-EXPO |
| S86 | clean | as S84 with held prompts only (`next_text` '?', commuted prompt in `body_hex`), back to back | roster, official, score, claim, publish 0 | `/verdict` PASS, base headline |
| S87 | dirty | FROZEN relabelled role 'candidate' (primary.candidate_arm stays W), answering every official HELD item like W | lock 1, official 1, score 0 | must_fire LOCK-02; `/verdict` FAIL, `/primary/implied_comparator_arms` [SHUF, FROZEN], `/primary/per_world_detail/0/comparator` FROZEN |
| S88 | dirty | C00e (T1, DEV) + SUMMARY.md 'Carry reuse con[]()firmed on the dev worlds.' | score dev 0, claim 0, publish 1 | must_fire CLM-04 |

- Probes against the HEAD kit (checker 1.0.0, `git archive HEAD tools/minifly_check` into the scratchpad, the fixture
  copies run through its CLI with `--git timeline:`): every one of S81-S88 exits 0 on every command and nothing fires.
  Under the repaired kit `selftest --only` passes for each (the dirty ones fire their must_fire id; S84/S85, like
  S65/S66, also make the exposed orbits S1, so STR-01, DET-01 and ADM-TB-CEIL/FLOOR fire as consequences). The
  pre-repair round-1 kit was never committed, so it could not be extracted; the verifier's probes B2/B3 (CLM-17),
  A5/A6 (ADM-EXPO), E2 (role candidate) and D4 (CLM-04) are the classes these fixtures re-implement, and the verifier
  reported each as passing under it.
- EXPECTED.json semantic diff against the pre-repair worktree: the 8 additions only; against HEAD: 24 additions, no
  entry removed, no must_fire, must_warn, commands, expected_exit or expect changed, no must_not_fire id lost.
- Why not a loosening: additions only; every dirty fixture asserts a must_fire, every clean one asserts values.

### A21. Unit tests changed in the round-1 repair (all additions or tightenings)

- `test_common.TestRealRules.test_fields_and_statuses`: the pinned changed_in [1.0.1] set grows from 14 to 20 ids
  (LOCK-02, STR-01, DET-01, ADM-TB-CEIL, ADM-TB-FLOOR, UDI-01; B12). The exact-set assertion stays exact.
- New: test_claimcheck.TestOverclaim (key/value lines and table rows, 17 dirty and 6 clean shapes; conformity phrases,
  citations about this delivery, vocabulary, process checks, 15 dirty and 8 clean; NOTES/ANALYSIS/README/RESULTS
  scope; project document lint naming a delivery), test_claimcheck CLM-04 markup (5 link-syntax shapes that must FAIL,
  one link to a non-confirm target that stays NOT_APPLICABLE), test_admitcheck.ExposureChecks (embedded JSON, filler,
  'c = a+b', stream order in JSONL, JSON list and text files; stream order needs the same field and the right answer;
  stream-named data/ file), test_lockcheck (LOCK-02 role candidate), test_decide (a second candidate arm joins the
  comparator and a copy of W FAILs), test_governance (changed_in append needs a module change; raw held-out numbers).
- No existing assertion was removed or relaxed. The round-1 DIRTY phrase 'The E&#51; rung is the result to build on.'
  still FAILs, now through 'is the result' (C47), not through the dropped word.

## B. RULES.json edits

### B1. WIT-01 joins the review_or_debt alternative group

- SPEC: 3.3 table row "REV, WIT | Obtain the review, or record REVIEW_DEBT (dev and pilot only)"; Section 5
  single-agent mode ("T2 roster spend may proceed only with a REVIEW_DEBT row, labelled DEV or PILOT"); the
  research-loop skill already states "REV-01..07, WIT-01: obtain the review, or record REVIEW_DEBT".
- Before: WIT-01 was a plain T2 entry, so no single-agent T2 roster could pass, contrary to SPEC 5.
- After: WIT-01 is in the group's primary list. The DEBT-01 fallback still never covers a SEALED_CONFIRM
  official stage, and covers a non-dry official stage only for PILOT. `MFC review check` reads the group from
  RULES.json, so the gate and the command cannot drift.
- Why not a loosening relative to the SPEC: it implements the SPEC's own allowance and the debt limits
  stay. C13c (dev output before any attestation) still fires WIT-01, because it has no debt row.

### B2. CLM-02 is NOT_APPLICABLE for a legacy pointer target

- SPEC: 7.4 PTR-07 ("A legacy target requires a UD and a KB/RETRO annotation path"); Section 1 (legacy
  directories have no v1 artifacts and are never written); 7.5 pointer bundle.
- Before: CLM-02 (`applies_when: always`) returned MISSING for every legacy target, so the PTR-07 path could
  never pass.
- After: `applies_when` names v1 targets, and CLM-02 returns NOT_APPLICABLE for a legacy target at the
  pointer stage only (elsewhere a legacy directory is still MISSING).
- Why not a loosening: v1 targets are unchanged, and PTR-07 is what governs legacy targets. S34 fires PTR-07.

### B3. UDI-01 planned

- SPEC: 20.2. UDI-01 is listed in README "Not yet enforced" and removed from the publish bundle.
- Why: no corpus fixture builds UDI variant worlds yet, and SPEC 9.7 does not define the seal-derived draw
  of the dependent set, so the function returns MISSING for SEALED_CONFIRM. The function is kept and fails
  closed. Skills now say UDI-01 is planned. No fixture listed UDI-01 in must_fire, so no `planned_must_fire`
  list was needed.
- Round 2 (verifier blocking issue 1): the planned status no longer leaks into the claim. Before, `claim`
  wrote reuse_specificity `udi_pass` from `scorer.udi()` alone (valid variants and LCB > 0), and CLM-03 could
  report "E3 promotion earned" for an adoption-intended SEALED_CONFIRM delivery, although only UDI-01 can
  certify the UDI result (SPEC 9.7 seal-derived draw; SPEC 7.4 UDI-01; SPEC 7.5 "UDI-01 (if applicable)").
  Now `claimcheck.udi_certifiable` requires UDI-01 to be registered as non-planned and to PASS on the
  delivery; otherwise reuse_specificity is `not_tested` and the CLM-03 detail reads "E3 promotion not
  certifiable: UDI-01 planned (not yet enforced)". A hand-written `udi_pass` in CLAIM.json FAILs CLM-03 (the
  fresh regeneration says not_tested). Unit tests: `test_claimcheck.TestUdiPlanned` (planned gives not_tested
  and no promotion; a hand-written udi_pass FAILs CLM-03; with UDI-01 enforced, udi_pass is claimed only when
  UDI-01 passes). Skills (`CLAIM_AND_POINTERS.md` reuse_specificity row, minifly-e3-design section 9) and the
  README "Not yet enforced" entry say so. This is a tightening: a value the kit cannot certify is no longer
  claimed.

### B4. DEBT-01 applies_when text

- The text now names WIT-01 together with REV-01..07 (follows B1).

### B5. Titles of AMD-05, LAD-01, CYC-01, RUN-01 and LOCK-06 (round 3)

- These are text-only edits to the `title` field, which states what each check now verifies (C21, C22).
- No id, module, stage, tier, status, applies_when or bundle changed.
- The corpus builder does not read titles, so no fixture byte changed.

### B6. RUN-03 registered and bundled (round 4)

- New entry RUN-03 (module launch, stages roster/official/publish, tier_min 0, introduced_in 1.0.0, enforced,
  warn_below_tier 2, fm FM-13/FM-18, applies_when always). Bundles: roster after RUN-01 (so official too, by extends),
  publish after RUN-01. SPEC 20.7. Dirty S60 fires it; every clean T2 fixture (C00a, C00b, C00d, S59, S61, ...)
  constrains it; at T1 (C00e) it would be WARN.
- `tests/test_common.SPEC_IDS` gains RUN-03 and RUN-04, and the bundle test asserts their placement.

### B7. RUN-02 amended and moved into publish; RUN-04 registered (round 4)

- RUN-02: title rewritten for the resume exception; stages official -> official, publish; it joins the publish bundle
  (it used to run at official only, so an illegal resume after the official gate would have reached publish unseen).
- New entry RUN-04 (module launch, stages official/publish, enforced, fm FM-20/FM-19, applies_when 'an official resume
  attempt or results/official_replay rows exist'; NOT_APPLICABLE otherwise). Bundles: official after RUN-02, publish
  after RUN-02. Dirty S63/S64 fire it; S61 constrains it.
- SPEC: 20.8.
- Direction, stated plainly: adding RUN-02 to publish and adding RUN-04 tighten. The RUN-02 amendment itself narrows
  what RUN-02 rejects: a second assessment-bearing official attempt now PASSes when it is a legal resume (identical lock
  shas, the latest interrupted attempt, only units without earlier rows, earlier rows unchanged, plus RUN-04's
  bit-identical replay). That part is a loosening of RUN-02 in the SPEC 13.2 sense, made because the round-4 task
  required it; it is bounded by (a)-(d), no existing fixture's fired set changed (the RUN-02 fixtures still fire), and
  it needs the owner's decision at the freeze (UD-0001) like every pre-freeze kit change. Without --resume a second
  official attempt that produces rows FAILs exactly as before (unit tests TestRun02, TestLaunch.test_official_never_rerun,
  test_official_resume_with_replay).

### B8. RUN-01 scope (round 4)

- No RULES.json change. RUN-01 now also covers `results/official_replay/` and FAILs while an in-flight marker
  (`runlog/INFLIGHT_<attempt_id>.json`) has no ATTEMPTS row (C29). Tightening.

### B9. CHECKER_VERSION 1.0.1 and `changed_in` (self-improvement round 1)

- `RULES.json` checker_version 1.0.0 -> 1.0.1 (also `common.CHECKER_VERSION`, `__init__.CHECKER_VERSION`). New optional
  entry field `changed_in` (list of versions); `validate_rules` checks it (increasing, later than introduced_in, not
  later than CHECKER_VERSION) and rejects an introduced_in later than CHECKER_VERSION. Governance classifies an
  appended changed_in entry as neutral and any other changed_in edit as loosen (it was an unclassifiable field, i.e.
  loosen, before). SPEC 20.10.
- changed_in [1.0.1] on the 14 tightened checks: ADM-EXPO, CLM-04, KB-03, REV-01, GOV-REV, GOV-COI, WIT-01, SEAL-02,
  DEBT-02, HELD-03, DEC-01, RED-02, RED-03, RUN-04. Titles of ADM-EXPO, CLM-04, KB-03, REV-01, WIT-01, SEAL-02, GOV-REV
  and RUN-04 now state the tightened rule; RUN-04's applies_when adds 'a recovered official attempt'. The advisory
  window still keys on introduced_in, so none of these tightenings is advisory for older deliveries.

### B10. CLM-17 registered and bundled

- New entry CLM-17 (module claimcheck, stages publish and lint, tier_min 0, introduced_in 1.0.1, enforced, fm FM-22,
  FM-07, applies_when 'agent-written text exists ...'). Bundles: publish after CLM-16; lint after CLM-16. SPEC 20.13.
  Dirty S68/S69 fire it, clean S70 (and every clean fixture with a publish command) constrains it; mutate reports it
  exercised.

### B11. CYC-04 registered and bundled

- New entry CYC-04 (module launch, stage publish, tier_min 0, introduced_in 1.0.1, enforced, fm FM-10, FM-19,
  applies_when always). Bundle: publish after RUN-04. SPEC 20.16. Dirty S77 fires it; clean publish fixtures constrain
  it. For a T1 delivery whose first SOURCE_LOCK is 1.0.0 it is ADVISORY_FAIL (advisory window), as for every new check.

### B12. Round-1 repair: changed_in bookkeeping, LOCK-02, titles

- changed_in [1.0.1] added to LOCK-02 (role candidate, C51), STR-01, DET-01, UDI-01 (scorer; their results change
  through the exposure strata of SPEC 20.11) and ADM-TB-CEIL, ADM-TB-FLOOR (admitcheck; same cause). Verifier
  non-blocking finding: S65/S66 fire these under 1.0.1 and not under 1.0.0, so without changed_in a batch stamped 1.0.0
  counted them while ADM-EXPO was post_batch. Each append comes with a code change in the module's import closure in
  the same range (C54).
- Titles: LOCK-02 ('...; role candidate only on primary.candidate_arm'), ADM-EXPO (embedded JSON, answer after filler,
  next record in stream order), CLM-04 (link syntax removed), CLM-17 (NOTES/ANALYSIS/README/RESULTS; key/value lines
  and table rows). CLM-17 applies_when names the new files and the lint of files that name a delivery.
- No id, module, stage, tier, status or bundle changed; no new check id. The C00f RULES copies follow (A20).
- Direction: changed_in appends are neutral bookkeeping that make held-out scoring stricter about what counts as
  post-batch; the applies_when edit widens CLM-17's scope (C47). Against HEAD, the range governance sees, CLM-17 is a
  new check id (tighten), so its title and applies_when edits inside this uncommitted round are no separate change;
  the LOCK-02, ADM-EXPO and CLM-04 title edits are neutral, and every changed_in append has its module's code change.

## C. Checker interpretation rulings (code, not tests)

- C1 ADM-CTRL-ZERO/ISO/DOSE, ADM-ACQ and ADM-SHUF require receipts, prefix rows and SHUF manifests in the
  admission worlds (`ADMISSION.worlds`, which ADM-01 cross-checks), not in testbed-only technical worlds.
  SPEC 3.1 C1: "One technical record per arm". Fallback is every lock technical world (fail closed).
- C2 LAD-01: the cells "implied by the lock" are the cells the rule, guards and gates read, plus the fixed
  SPEC measurement points. These are the primary (pool, checkpoint), OLD at old_end and final, TAUGHT at
  prefix, HELD and NEW at final, and every other declared pool at final. It is not the full pool x
  checkpoint product, because SCHEMAS defines no measurement plan. Cells with an undeclared world, arm,
  pool or checkpoint FAIL; extra declared combinations must be complete.
- C3 DEC-01: at pre-run stages (lock/roster/official and the launch stages), official rows without
  VERDICT.json are noted, not MISSING, because `score` writes VERDICT after the run. From publish on,
  VERDICT is required.
- C4 REV-02: compared_to and a boolean match are required for rerun_admission (SPEC 6.4 "compare digests";
  REV-03). For other kinds, a match that is present must be a boolean.
- C5 Headline (SPEC 9.9):
  - `[S2+ determined, ...]` whenever every counted stratum lies within S2..S4.
  - "held exact own-EOS not_measured" replaces the held, shortcut and G segments; OLD is printed when
    measured.
  - `claim` no longer pushes a hand-written REPORT first line down; CLM-05 then FAILs (C22).
  - 0 evaluated worlds and no verdict gives PILOT (CLM-09: "fewer than min_worlds").
- C6 CLM-14 in timeline (corpus) mode: a fixture commit without a kit tree has the empty-kit digest
  (sha256 of b''). In a real repo a commit without a kit never counts.
- C7 CLM-15: without any official run (no official ATTEMPTS row, OUTPUTS or VERDICT), the T2 subset is the
  existing locks, every gate report and the admission/testbed OUTPUTS. This applies to honest exits before
  official, per I10.
- C8 AMD-03: every raw outcome file must be in results_seen. Checker-written score files
  (SCORES/VERDICT/TESTBED) that were committed before the amendment may also be listed. Anything else extra
  FAILs.
- C9 Seal: official world id i = `w_seal_` + 16 hex digits of world_seed_i. SPEC 10 step 5 defines the seed;
  the id format is SCHEMAS.md section 13.
- C10 RED-09 and RED-11 require the lock's own old_drop or old_acq guard to report FAIL on the synthetic
  OLD failure. A rule that rejects everything, or has no such guard, FAILs them (S09).
- C11 `kb score` reads SPEC 8 `predictions_scored` entries (target and p_pass come from the lock prediction
  with that id). It adds top-level `n_primary_predictions`, `brier_mean` (6 decimals) and `skill`.
- C12 `gate` on a legacy directory reports the registered SCHEMA-01 result in `--json` (exit 2, nothing
  written).
- C13 selftest:
  - An expected exit 3 is not a problem.
  - A command that writes no JSON file contributes its `--json` stdout object to `expect`.
  - The SPEC 12.2 canary step uses C00a's own canary command, run after `score` and `claim`, because the
    committed tree alone has a CLAIM without its VERDICT.
  - Fixtures run in forked worker processes.
  - Under `--mutate`, a clean fixture also notices an always-FAIL replacement of an advisory check when that
    id newly WARNs.
- C14 `launch` records new or changed `manifests/TAUGHT/**` files in `produced_paths` (runner contract).
  RUN-01's scope is unchanged.
- C15 GOV-REV: a review record that names no range, but was first committed inside the governed range,
  binds to that range. A record that names another range never binds.
- C16 DEBT-02 reads the lock only to judge a repaid debt, so a legacy pointer target has no debt.
- C17 Cycle accounting (round 2, verifier blocking issue 2). SPEC 3.2: "Every cycle ends with `MFC cycle
  close`", "The test-cycle cap is AUTHORIZATION.caps.max_cycles (default 3), counting technical, dev and
  rehearsal", "`launch` refuses new dev or rehearsal attempts beyond the cap"; 7.6 (a shallow clone gives
  MISSING). Before, the cap, CYC-01 and CYC-02 counted only the CYCLES rows the executor chose to write, so never
  closing a cycle (or deleting committed rows) left the cap unenforced. Now (`launch.cycle_accounting`):
  - Every CYCLES row must carry `computed.attempts` (rows are written by `cycle close`), list only attempt ids
    that ATTEMPTS holds, cover each attempt at most once, and cover only attempts of its own kind. An
    uncovered attempt that precedes a covered one FAILs.
  - Attempts no row covers form one open cycle per stage. CYC-01 counts closed test cycles plus open test
    cycles against the cap, and FAILs when more than one stage is open (a cycle was not closed before the next
    stage).
  - At the non-dry official stage (gate or `launch --stage official`) and once an official attempt exists,
    every testbed, technical, dev and rehearsal attempt must be in a closed cycle. At `gate --stage official
    --dry` the rehearsal cycle may still be open, because its exit criterion is that dry report.
  - runlog/CYCLES.jsonl and runlog/ATTEMPTS.jsonl are append-only: every committed row must survive in every
    later commit and in the worktree. A shallow clone gives MISSING.
  - `launch` refuses an attempt of another stage while a cycle is open, and refuses a dev or rehearsal attempt
    when closed cycles plus the cycle this attempt joins or opens would exceed the cap. An attempt that joins
    the open cycle of its own stage does not open a new one: SPEC 3.2 caps cycles, not attempts.
  - `cycle close --kind K` covers the uncovered attempts of stage K only, and refuses when K has none while
    another stage's cycle is open.
  Unit tests: `test_launch` (cycle cap and close flow in a real git repo; never-closed cycles; open cycles
  against the cap; dry vs non-dry official; row/attempt agreement; append-only over history, worktree deletion
  and shallow MISSING). Corpus: S46-S49 (A11). Skills (research-loop section 4, STAGES.md), SCHEMAS.md section
  11 and README state the rules. Every existing fixture is unchanged and still passes.
- C18 Official precondition verdicts name their worlds (round 2, verifier blocking issue 3). SPEC 9.5 (the
  precondition verdicts INVALID INTEGRITY_ALARM, CANNOT_TEST_CLAIM_AS_SPECIFIED, INVALID
  TREATMENT_NOT_DELIVERED, INVALID NONFINITE); 7.4 CLM-09; 8 (primary_gate INVALID, stop_reason, n_worlds);
  9.9 (`<n> <ROLE> worlds`). Before, `decide.evaluate` returned a precondition verdict before it filled
  `world_roles`, so VERDICT.json carried `world_roles {}`; `claim` then took the delivery as DEV and rewrote
  primary_gate INVALID to NOT_OPENED with dev shadow INVALID, stop_reason null and 0 DEV worlds (probe P2b).
  Now `world_roles` is computed first, so an official INVALID on 8 official worlds publishes as primary_gate
  INVALID, stop_reason INVALID, 8 MAIN worlds. DEC-01 also compares `world_roles` with a fresh decide
  (VERDICT_COMPARED_FIELDS), because CLM-09 reads it. Corpus: S50.
- C19 `claim` reuse_specificity and the CLM-03 promotion note follow UDI-01's status (section B3).
- C20 Selftest freezes committed fixture content before running commands (section A12).
- C21 Stage scope of attempts (round 3, verifier blocking issue 1).
  - SPEC:
    - 3.1 stage map: C0 runs evaluator-side ceiling and floor grids before learner code; C1 is one technical
      record per arm on technical worlds; C2 is dev worlds; C3 is a fresh rehearsal world.
    - 3.2: "testbed (C0, not counted against the cap)"; the cap counts technical, dev and rehearsal;
      "launch refuses new dev or rehearsal attempts beyond the cap".
    - 7.4 CYC-01, RUN-01 (T2 enforced, T1 WARN) and RUN-02.
  - Before: the cap applied only to attempts labelled dev or rehearsal. Nothing tied a results/dev or
    results/rehearsal file, or a row, to an attempt of that stage. So `launch --stage testbed -- <dev runner>`
    produced dev outcomes outside the cap, and `cycle close --kind testbed` covered them.
  - After (`launch.stage_scope_problems`, called from CYC-01 at every tier):
    - Each attempt's produced_paths lie in its stage's results directory: testbed and technical write
      results/admission/, dev, rehearsal and official write their own.
    - Receipts and TAUGHT manifests belong to the stage's worlds. A world list the lock leaves empty admits any
      world no other role lists.
    - A testbed attempt writes only CEILING_*/FLOOR_* receipts and rows. It may write the TAUGHT streams of any
      world only while no learner attempt precedes it (cycle-0 stream generation); after that, only CEILING_*/
      FLOOR_* manifests of technical worlds.
    - Every OUTPUTS row under results/<role>/ whose attempt_id names an ATTEMPTS row must name one of the
      matching stage, on a world of that stage.
    - RESULT_BUNDLE*.zip files outside results/<role>/ are judged by their members.
    - RUN-01 also FAILs (T2; WARN at T1, as the SPEC sets) on OUTPUTS rows whose attempt_id names no ATTEMPTS
      row. Official rows were already covered by RUN-02.
  - `launch` changes:
    - It refuses beforehand a command line that names a lock world of another stage.
    - It exits 1 after an attempt that wrote outside its stage. The ATTEMPTS row is still written; the log is
      append-only, so CYC-01 FAILs for good.
    - It refuses technical attempts beyond the cap too. CYC-01 counts technical cycles against the same cap, so a
      technical attempt beyond it could only end in a CYC-01 FAIL. This is stricter than the SPEC text, which
      names dev and rehearsal.
  - `cycle close` names any out-of-stage attempts of the cycle it closes.
  - Unit tests:
    - `test_launch.test_testbed_cannot_carry_dev_outcomes`: real git, cap 1, the verifier's reproduction. The
      argv refusal holds, launch exits 1, and CYC-01 FAILs at roster, official --dry and official.
    - `TestCycles.test_cyc_01_stage_scope`: each rule, including bundles and cycle-0 TAUGHT.
    - `TestRun01.test_rows_must_name_a_launch_attempt`.
    - `test_cycle_cap`: technical is refused beyond the cap.
  - Corpus: S51.
  - SCHEMAS.md section 10 has the stage-scope table. The research-loop skill (section 4) and STAGES.md (C0, C2
    and cycle bookkeeping) and the runner contract now state the rules. Every existing fixture complied, except
    the dirty C10 (its dev attempt also runs w_reh_1): C10 now also fires CYC-01, and it already expected exit 1.
  - Why not a loosening: every check that held before still holds, and new failure paths are added.
- C22 The held set is fixed in the lock (round 3, verifier blocking issue 2).
  - SPEC: 6.2 ("The held set is fixed in the lock (split.held_set_sha256) before any stream exists"); 1 I9;
    8 (split {held_manifest, old_manifest, new_manifest, split_sha256, held_set_sha256}); 7.4 AMD-05 and LAD-01
    ("cell implied by the lock"); 7.6 (a missing pool cell gives FAIL or MISSING).
  - Before: held_set_sha256 and split_sha256 were format-checked only (LOCK-01). Nothing compared OUTPUTS items
    with the manifests.
  - After:
    1. Formula. `common.split_hashes` defines both hashes: held_set_sha256 = sha256 of the held manifest bytes;
       split_sha256 = sha256(canonical_json({held_manifest, old_manifest, new_manifest: file sha256})). This is the
       corpus convention (corpus/README interpretation 4) and is now in SCHEMAS.md section 1. `MFC init` fills
       both when the manifests exist.
    2. AMD-05 recomputes both from the worktree manifests. At every commit after the first lock commit that
       touches the lock or a split manifest, it recomputes them from that commit's manifests against the lock in
       effect. A manifest that is absent while locked FAILs, and so does a split-hash re-pin without an
       AMENDMENTS row (AMD-01 and AMD-02 still apply to the lock change itself).
    3. The scorer (`scorer.score_role`) raises InvalidInput on any HELD, OLD or NEW row whose item_id is not in
       its pool's manifest, or whose prompt_hex differs from it. Every scorer-based check, and `score` itself,
       therefore fails closed: VERDICT is never computed from an item the lock did not fix.
    4. LAD-01 requires every (world, arm, pool, checkpoint) cell of those pools to hold exactly the manifest's
       item set.
    5. LOCK-06 requires the first commit adding a TAUGHT stream to strictly follow the lock commit ("before any
       stream exists"). The dirty C13a, where lock and testbed share a commit, already fires LOCK-06.
  - Unit tests:
    - `test_lockcheck`: AMD-05 split edit (worktree, committed, reverted), deletion and re-pin; LOCK-06 stream
      timing; `init` fills the split hashes. The fixture builder now writes real manifests and hashes.
    - `test_scorer`: foreign item or prompt raises and fails LAD-01; a cell missing a manifest item fails LAD-01.
  - Corpus: S52, S53 and S54. Every existing fixture's gated delivery already matched its manifests and hashes.
    S12 (a missing item) fires LAD-01 as before.
  - Skills (e3-design section 2, STAGES.md S1, runner contract) say which checks enforce the held set.
  - Why not a loosening: the lock's commitment is now verified, and nothing that held before stops holding.
- C23 CLM-04 confirm wording (round 3, verifier non-blocking finding).
  - SPEC: 7.4 CLM-04 ("confirm* tokens"); 3.4 ("Using confirm wording off sealed worlds").
  - CONFIRM_RE now also matches the re-, un-, dis- and pre- forms ('reconfirms', 'unconfirmed'). They are the
    same confirmation wording.
  - The kit identifiers CONFIRM_LEDGER, confirm_attempt and confirm_alpha_spending stay excluded, and
    SEALED_CONFIRM still never matches.
  - Unit test: `test_claimcheck.test_clm04_native_confirmation_on_dev`. This is a tightening.
- C24 CLM-15 keeps committed reports (round 3, verifier deletion sweep).
  - SPEC: 7.4 CLM-15 ("both locks and GATE reports"); 3.4 ("Deleting predictions, findings or failed-attempt
    directories" is forbidden); 14 ("Deleting failure markers").
  - Before: CLM-15 required only the reports present in the worktree, so a committed deletion of the lock,
    roster, technical, DRY or CANARY report left publish at exit 0.
  - Now:
    - Every checks/*.json report ever committed must still be committed at HEAD and present in the worktree.
    - Every ATTEMPTS.gate_report must exist.
    - Every CYCLES computed.latest_gate must still hold the bytes its sha256 recorded.
    - A shallow clone gives MISSING.
  - Unit test: `test_claimcheck.test_clm15_reports_are_never_deleted`. Corpus: S55. This is a tightening.
- C25 Diagnostics only:
  - `MFC selftest` now prints its preflight problems, which it used to count without naming.
  - `MFC kb --help` lists the backtest-score flags that kb.py parses: --contract, --kb, --outcomes, --seed and
    --boot.
  - corpus/README.md describes S46-S55.

- C26 CLM-04 identifier tokens (round 4, verifier blocking issue; SPEC 20.5, 7.4 CLM-04, I6, 3.4).
  - Before: `CONFIRM_RE = (?<![A-Za-z0-9_])(?:re|un|dis|pre)?confirm\w*` missed every confirm token after '_' or a
    letter (GAIN_CONFIRMED, axis_confirmed, SCIENTIFIC_CONFIRMATION, formal_confirmation, native_confirmation,
    dev_confirmed, NativeConfirmation) and any token split by a zero-width character.
  - After (`claimcheck.confirm_skeleton`, `confirm_tokens`): NFKC, removal of Cf format characters and combining marks,
    folding of Cyrillic/Greek look-alikes, then every case-insensitive 'confirm' with its enclosing `[A-Za-z0-9_]+`
    token is a hit unless the whole token is one of SEALED_CONFIRM, sealed_confirm, CONFIRM_LEDGER, confirm_attempt,
    confirm_alpha_spending (case-sensitive; round 3 compared lowercased tokens). Scope unchanged.
  - Unit test: `test_claimcheck.test_clm04_identifier_tokens_and_format_chars` (all seven verifier probes plus the
    soft hyphen, BOM, fullwidth and Cyrillic forms; the allowlist; SEALED_CONFIRMED and xSEALED_CONFIRM still hit).
    Corpus S56-S58; C00d stays clean. Tightening.
- C27 dev_touched fails closed (round 4, verifier non-blocking finding; SPEC 7.6).
  - Before: `except InvalidInput: continue` skipped an unparseable receipt, so it could not mark its world dev-touched.
  - After: a receipt that does not parse, or is not a JSON object, raises InvalidInput naming it; CLM-09 and WLD-01 FAIL
    and `claim` refuses. Unit test `test_claimcheck.test_dev_touched_unparseable_receipt_fails_closed`. Tightening.
- C28 Decision comparator (round 4, verifier probe 2c; SPEC 20.6, 9.5).
  - `decide.parse_params` returns comparator_arms = declared + implied (`implied_comparator_arms`, roles
    no_prior_teaching/shuf/frozen/lesion, never oracle_plumbing, positive_control or CEILING_/FLOOR_), plus
    `declared_comparator_arms`; `evaluate` uses the decision list except in mode testbed. RED-02/RED-03 and the DEC-01
    property fixtures therefore cover the implied arms. `scorer.inert_alarms`, LAD-03 (`comparator_strength`), CLM-03
    and canary M06 keep the declared list (an all-zero FROZEN arm next to a failing candidate is an honest FAIL, not an
    INTEGRITY_ALARM: a first draft put the implied arms into INERT and turned test_scorer's honest S0-only FAIL into
    INVALID, so that was reverted). SCORES gains decision_comparators; the headline suffix reads them.
  - Unit tests: `test_decide.test_implied_comparators_close_declaration_minimization`, `test_testbed_mode` (the
    testbed view needs no NPT/SHUF cells), `test_reducercheck.test_red_02_03_cover_implied_comparators`.
  - Test change, not a loosening: `test_scorer.test_v6_like_zero_operand_items_are_s0_and_not_counted` declares a SHUF
    arm but wrote no SHUF rows; it now first asserts INCOMPLETE without them (the new requirement) and then writes SHUF
    rows and keeps its FAIL assertion.
- C29 RUN-03 and launch's runtime-closure support (round 4; SPEC 20.7).
  - `launch` sets MINIFLY_REPO_ROOT and MINIFLY_RUNTIME_CLOSURE, snapshots `runlog/RUNTIME_CLOSURE_*.json` with the
    results (so produced_paths records it), and `launch.check_run_03` implements the rule. The SOURCE_LOCK an attempt ran
    under is found by its source_lock_sha256 in the worktree or history (MISSING if absent). Attempts that produced no
    path at all are exempt (nothing to attribute); testbed attempts are not counted runs.
  - The runner contract helper (12 lines) is in templates/admit_skeleton.py and in the e3-design runner contract; both
    copies run under a real `launch` in `test_launch.test_runtime_closure_from_the_template_helper` and
    `test_runtime_closure_contract_snippet`. Synthetic cases: `test_launch.TestRun03`.
  - CLM-15 adds the official attempts' runtime closures and any replay rows to the verification subset. Tightening.
- C30 Resumable official runs (round 4; SPEC 20.8).
  - CLI `launch --stage official --resume <dir> -- <cmd>`; `--resume` with another stage is a usage error (exit 3).
  - `launch.resume_analysis` (RUN-02 as amended) and `replay_analysis` (RUN-04) implement (a)-(d); launch records
    `units` (with canonical digests from `review.row_digest`) for every official attempt, and resume_of,
    replay_requested (one completed unit chosen with secrets.SystemRandom) and replay_units for a resume; it refuses a
    resume that cannot be legal and exits 1 after one that broke a rule.
  - In-flight markers (`runlog/INFLIGHT_<attempt_id>.json`): written before the command, removed after the row; a
    marker whose launch process is gone (pid, boot_id, host) becomes a recovered ATTEMPTS row at the next launch or cycle
    close; a live one refuses. A resume that crashed before writing official rows needs no replay (a later resume
    replays).
  - Stage scope: only official attempts may write `results/official_replay/`; MET-01 (stray rows) and ADM-EXPO
    (undeclared streams) accept `results/official_replay/OUTPUTS.jsonl`, which is never scored.
  - CLAIM.provenance (`claimcheck.official_run_provenance`) {interrupted, resume_attempts, resumed_units,
    replay_units}; CLM-01 checks its shape.
  - Unit tests: `test_launch.TestLaunch.test_official_resume_with_replay`, `test_official_resume_violations`,
    `test_official_resume_replay_mismatch_and_lock_change`, `test_resume_only_for_official`,
    `test_inflight_marker_recovery`; `test_launch.TestResumeChecks`.
  - Residual risk: a recovered row cannot know the dead attempt's exit code or CPU (rusage_lost; compute totals
    undercount it), and like any launched command a hand-planted marker can make hand-written files launch-recorded;
    this is the same power `launch -- cp` already has, and RUN-03/RUN-04 and review replay are what bound it.
- C31 Per-phase lock re-check (round 4; SPEC 20.9). No code change was needed: every `launch` invocation re-runs its
  stage bundle (LOCK-05 from technical on) and `closure_state` (every closure, files.* and endpoint sha and a clean
  worktree for them) before the command and again after it. `test_launch.test_every_phase_relaunch_rechecks_the_lock`
  pins the behaviour (phase 1 runs; a committed fix between phases is refused at phase 2; a new lock build runs).
  The rule 'one launch per phase; never chain phases inside one unlaunched script' is in the research-loop skill
  (sections 3 and 8) and STAGES.md.
- C32 Skills and schemas (round 4): research-loop SKILL.md and STAGES.md (resume, per-phase launch, RUN-03), e3-design
  SKILL.md, RULE_GRAMMAR.md (decision comparator, RED-02/03) and ADMISSION_AND_RUNNER_CONTRACT.md (6.1 runtime
  closure helper, 6.2 resume contract), claims-knowledge SKILL.md and CLAIM_AND_POINTERS.md (CLM-04 tokens,
  provenance, headline suffix), independent-review SKILL.md (resume replay vs REV-08), SCHEMAS.md (7, 10, 10.1-10.3,
  12, 14, 15), README.md and corpus/README.md. Descriptions unchanged; no rule line (must/never/minifly_check) removed;
  every SKILL.md body is under 400 lines.
- C33 Concurrency note (round 4): while this round ran, another session process edited `retro.py` and ran
  `selftest --mutate` and `retro --all` in the same worktree. This round did not touch retro.py; one full unit-test run
  saw `test_retro.test_acceptance_bookkeeping` fail (3 != 2) while that edit was in progress and it passed in every
  later run.

- C34 Versioning (round 1; SPEC 20.10). See B9. The governance lesson is recorded in SPEC 20.10, the governance skill
  (section 5 and 7) and CHANGE_CLASSES.md / HELDOUT_CANARIES.md.
- C35 ADM-EXPO judges content (round 1; SPEC 20.11, 7.4 ADM-EXPO, FM-12; INC-HB1-A3). `admitcheck.exposure_scan`
  (cached per ctx) reads every declared stream, the TAUGHT directory and every undeclared stream-like file, as JSONL
  rows, JSON document records or text lines, decoding every string value and key as text, hex and base64 and every byte
  list as bytes. A record leaks a held item when it carries the prompt (E2) or canonical orbit (E3, `a+b` with or without
  the terminator, any field) with the expected answer (a separate value equal to it after EOS/whitespace/leading zeros
  are stripped, or prompt followed by answer in one value); any leak FAILs whatever the feedback flag. Feedback-bearing
  rows still FAIL on any held prompt/orbit, now in any field. Unreadable stream files (not JSONL, JSON or UTF-8) FAIL.
  `_is_stream_candidate` also treats results/ paths naming calib, curric, expos, warm or listen as stream files (a
  tightening: such a file must be declared). Before: rows with feedback_bearing false were skipped and only prompt_hex
  was read (the A3 route). Unit tests in test_admitcheck.ExposureChecks; corpus S65-S67. Tightening.
- C36 Exposure strata (round 1; SPEC 9.3, 20.11). `scorer.score_rows(..., exposed_orbits=...)`: a held item whose
  canonical orbit `admitcheck.exposed_held_orbits` reports is S1 in every world (S0 stays S0); `score_role` passes the
  orbits (also to the testbed and UDI scoring) and records them under `exposed_orbits`. Only E3 addition locks with a
  held manifest have orbits. Honest deliveries have none, so no existing SCORES, VERDICT or headline moved. Unit test
  test_scorer.test_held_answer_in_any_stream_makes_the_orbit_s1. Tightening.
- C37 CLM-04 markup (round 1; SPEC 20.12; verifier non-blocking finding). `claimcheck.markup_skeleton` runs before the
  20.5 normalisation in `confirm_tokens`: entities decoded (up to 3 rounds), HTML comments and tags removed, `*`, `**`,
  backticks, `~~` and `__` removed. Applied to every CLM-04 text (markdown, text and JSON strings). Unit test
  test_claimcheck.test_clm04_reads_markup_as_rendered; corpus S74-S76; C00d (SEALED_CONFIRM) stays clean. Tightening.
- C38 retro uses the shared tokenization (round 1; SPEC 20.12). `retro.confirm_normalized` rewrites every
  `claimcheck.confirm_tokens` token to the word 'confirm' and `dev_confirm_wording` applies its binding rules to that;
  the local `CONFIRM_RE` is gone. `MFC retro --all --out <scratchpad>` (156 s) gave 1234 rows with statuses and reasons
  identical to the committed RETRO_20261024.json and acceptance true; the committed RETRO files were not rewritten.
- C39 CLM-17 (round 1; SPEC 20.13; INC-HB1-C2). `claimcheck.overclaim_problems` over `agent_text_docs` (REPORT body
  below the headline, CLAIM notes and findings claims, STATUS*/SUMMARY*/MEMO* files; JSON scalars as '<key words> is
  <value>'), with level_earned/primary_gate from the fresh claim (committed CLAIM as fallback) and guard statuses from
  VERDICT. Lint mode judges a file against its delivery's CLAIM.json. Design for low false positives: clause splitting,
  explicit qualifiers, guard clauses checked against VERDICT, prior-work citations skipped; the 14 dirty and 14 clean
  phrases of test_claimcheck.TestOverclaim (none of them a canary sentence) pass. Every pre-existing clean fixture stays
  clean.
- C40 Decision comparator role other (round 1; SPEC 20.14). `decide.IMPLIED_COMPARATOR_ROLES` adds 'other'. No
  pre-existing fixture has an arm of role other, so no fired set changed. Corpus S80. Tightening (DEC-01, RED-02,
  RED-03 changed_in).
- C41 One identity function (round 1; SPEC 20.15, 5; INC-HB1-C3). `common.commit_family` / `commit_is_family` /
  `branch_namespace` / `agent_family`. Callers: kb (KB-03 introducing commit; `activation_problems(...,
  commit_identity=True)` and `row_identity_problems` in the `finding_state` fold, so `kb status` ignores a forged
  activation; reconciliation and R2 credit need a valid identity), review (`is_family_commit`, the lock-commit check of
  REV-01, the REVIEW.json history rule, DEBT-02 repayment), lockcheck WIT-01 (attestation and ack commits: trailer now
  required too), seal (SEAL-02 on the SEAL_COMMIT and, new, the SEAL_REVEAL commit; `seal reveal`; unit test test_seal.test_reveal_commit_must_act_as_the_reviewer), governance (GOV-REV introducing commit and protected-commit authors;
  GOV-COI offenders; HELD-03 kit-commit authors). `ledger append` keeps the field-level check (the row is not committed
  yet). The literal 'all containing branches must be one namespace' was narrowed to a git-decidable rule (SPEC 20.15
  explains why: shared ancestry puts every pre-review author commit in the reviewer branch). Before: KB-03 read the
  trailer only, GOV-REV accepted any `<remote>/claude/` branch, WIT-01 and SEAL-02 read the namespace only, DEBT-02
  read REVIEW.reviewer_agent only, GOV-COI and HELD-03 read trailers only. Tightening.
- C42 CYC-04 (round 1; SPEC 20.16). `launch.check_cyc_04` runs `runlog_append_only_problems` at publish. Unit test
  test_launch.TestCycles.test_cyc_04_append_only_at_publish; corpus S77.
- C43 Recovered official attempts (round 1; SPEC 20.17). `launch.replay_analysis` records which attempt completed each
  unit and which units a resume replayed with the original digest; a recovered official attempt (`attempt_recovered`:
  rusage_lost or recovered) with rows and no such unit FAILs RUN-04 (`recovered_unvouched`). `launch --resume` picks
  its replay unit among unvouched recovered units first. Unit test
  test_launch.TestResumeChecks.test_recovered_attempt_needs_a_replay_of_its_own_units; corpus S78, S79. Tightening.
- C44 Held-out post-batch scoring (round 1; SPEC 20.18). `governance.canary_checker_version` and
  `heldout_score_report(..., batch_versions)`: fired ids newer than the canary's SOURCE_LOCK checker_version (via
  `common.rule_newer_than`) are `post_batch`, listed and not counted. Not re-run on the burned pilot batch (its
  plaintext lies under corpus/heldout/revealed/, which this round does not read beyond the CANARY_NOTE files); unit
  tests test_governance.TestHeldoutPostBatch. This changes a measurement, not a check.
- C45 Builder report contents (A18) and fixture module (A17); the builder still imports no checker code.
- C46 Residual risks recorded in SPEC 20.11/20.13/20.15 and README 'Known limitations': integer answer fields and
  operands split over fields are not read by ADM-EXPO; CLM-17 is lexical; a forged-trailer commit merged by the
  impersonated family is indistinguishable. CYC-04 and CLM-17, like every new check, are advisory for T1 deliveries
  locked under 1.0.0.

- C47 CLM-17 key/value shapes and the rest of the repair (round-1 repair; SPEC 20.13; verifier blocking issue 1 and
  non-blocking findings). `claimcheck._kv_clauses` adds the joined clause '<key> is <value>' for 'key: value',
  'key = value' and short 'key - value' segments and for every data cell of a markdown table row (first cell as key,
  and the column header under a header row); `_kv_blocks_standalone` keeps a run of key/value lines from being joined
  into one sentence. `overclaim_clauses` reads every text twice (as written and with markdown links rendered) and
  returns (line, clause, joined); joined clauses (and the JSON path's '<key> is <value>' docs) also accept an
  affirmative value ('E3: yes') and are qualified by when/whenever/iff/once/provided/assuming. Tightenings: the
  vocabulary (supported, warranted, substantiated, proven, 'holds' ending a claim, 'E3-grade', 'is an E3 learner',
  'the result is E3', 'E3 ... is the result', 'success' unless 'success criteria'), conformity phrases removed before
  the qualifier test ('as planned', 'exactly as predicted', 'as expected', 'in line with the plan', 'on target',
  'without exception'), citation sentences that also point at this delivery judged part by part, the scope
  (NOTES*/ANALYSIS*/README*/RESULTS* outside results/ and manifests/), and lint of files outside a delivery against each
  delivery they name (`_named_delivery_texts`). 'rungs?' was removed from CLM17_ACHIEVE_RE (verifier overfitting
  item: canary vocabulary); 'is the result' carries the round-1 unit phrase instead. Narrowings, stated plainly
  (CLM-17 is new in 1.0.1, so none of this loosens anything HEAD enforces, and no corpus must_fire depends on it): (1)
  the process-check exemption of (a): a clause whose only achievement words are pass/clear, whose subject is a process
  check, and whose level tokens only qualify a process noun ('the roster gate for this E3 design passed', 'E1 plumbing
  checks passed', 'admission passed for the E3 design') is no level claim (verifier non-blocking false alarms); 'the
  lock passed E3', 'W passed the E3 gate' and 'the review passed W for E3' still FAIL; (2) a clause opening with the
  bare form of an achievement verb is a plan ('Goal: reach E3 in the next cycle' was a false alarm through the colon
  split); (3) a key/value run is read line by line, so 'Decision rule: W passes when ...' and the next line are no
  longer one clause. Scratchpad sweep: 42 of 42 paraphrases fire, 0 of 29 honest sentences alarm; every clean fixture
  stays clean.
- C48 CLM-04 markdown links (round-1 repair; SPEC 20.12; verifier probe D4). `markup_skeleton(links=True)` replaces
  inline links, images and reference links by their text; `confirm_tokens` adds the tokens of that reading that the
  first reading (link targets kept) lacks, so nothing the first reading found is lost. retro's R-CLAIM uses
  confirm_tokens and therefore sees the same tokens. Corpus S88. Tightening.
- C49 ADM-EXPO packaging (round-1 repair; SPEC 20.11; verifier blocking issue 2 and probes A4, A8). `_flat_items` lists
  (path, value) pairs and parses string values that are JSON objects or arrays (as text, hex or base64; lenient
  json.loads, so NaN or duplicate keys cannot hide a payload); `HeldIndex.scan_value_full` reads the answer after up
  to 32 bytes of non-digit filler unless it is a following sum's first operand, and 'c = a+b'; `scan_items` returns per
  path the unanswered keys whose prompt or orbit ends the path's last value and the answer forms its first value
  starts with; `exposure_scan` pairs each record with the previous record of the same file at the same path and
  reports '... starts with the answer of held orbit X that <prev> ends with (stream order)'. The filler, 'c = a+b' and
  stream-order rules use text readings only (`_decodings_tagged`): the raw text of a hex-shaped string and a decoding
  that is not printable UTF-8 keep the exact rules only, so the hex text '6b697465' ('kite') is not the answer 6 and
  base64 noise decoded from a hex string never alarms (unit test). Text records keep their
  line and field positions as paths (`TextRecord`). `_is_stream_candidate` adds stream-named data files outside
  manifests/, results/, checks/, runlog/ and review/. The scorer's S1 orbits come from the same scan. Corpus S84-S86.
  Tightening; every pre-existing clean fixture stays clean (S67's back-to-back held prompts start with an operand).
- C50 ADM-EXPO residuals kept (SPEC 20.11): integer answers, answers two or more records later or in another field of
  the next record, files.assets, neutrally named JSON under results/, and the fail-closed false alarm of a string
  number equal to a held sum beside a held prompt (verifier probe A13). Restricting stream order to the same field
  path is what keeps index-like fields of the next record from alarming.
- C51 Role candidate (round-1 repair; SPEC 20.14; verifier blocking issue 3). LOCK-02 FAILs any arm of role candidate
  other than primary.candidate_arm, and `decide.IMPLIED_COMPARATOR_ROLES` adds 'candidate' (the candidate arm itself
  is excluded by `implied_comparator_arms`), so a lock that carries such an arm and is scored anyway still compares
  W with it. No corpus lock has a second candidate arm (checked over every committed ENDPOINT_LOCK.json), so no fired
  set changed. Corpus S87. Tightening.
- C52 positive_control/oracle_plumbing copies of the candidate (verifier non-blocking) stay out of the comparator by
  design (20.6, 20.14); recorded as a residual risk with reviewer guidance (minifly-e3-design RULE_GRAMMAR section 3).
  Requiring declared_changed was considered and not added: it is declared, not verified, so it would only add ceremony.
- C53 Identity laundering through forks (verifier non-blocking, probe C6): no code change. SPEC 20.15 now names the
  fork case and minifly-independent-review section 1 tells the reviewer never to fork or merge an author tip that
  carries a commit with the reviewer's own trailer; a first-parent rule is left as residual.
- C54 changed_in needs a code change (round-1 repair; SPEC 20.10; verifier non-blocking). `governance.classify_rules`
  tags an appended changed_in with the rule's module; `require_module_change` (called by `_analyze`) keeps it neutral
  only when the range also changes the module or a kit module in its import closure (`kit_import_closure`: import
  statements at any depth and lazy('x') / import_module('x') with literal names, over the head tree), else loosen. The
  closure is used rather than the module file alone because a check's behaviour can change through code it calls
  (RED-02/RED-03 in reducercheck changed through decide in round 1). Unit test
  test_governance.test_changed_in_append_needs_a_module_change. Tightening of the classification.
- C55 Held-out raw numbers (round-1 repair; SPEC 20.18). `heldout_score_report` adds recall_fixtures_raw,
  false_alarm_rate_clean_raw and per fixture caught_raw / false_alarm_raw (post_batch ids counted); `heldout score`
  prints both. Reporting only; no batch was re-scored (the revealed batch was not read).
- C56 Verifier items not changed: CLM-17 stays lexical (a paraphrase outside its vocabulary is the residual the
  reviewer covers); A12 (a neutral-named results/ JSON) is left to RUN-01, which fired on the verifier's probe.

## D. Integrator amendments recorded for the verifier (round 2)

- D1 Seed finding ids `F-ARCH-01`..`F-ARCH-12`. SPEC 8 names FINDINGS ids `F-<yyyymmdd>-<agent>-<n>` and seed
  `F-UF-xx`; WP10 seeded the archive-digest facts as `F-ARCH-nn`. KB-01 (`kb.py` id check) and SCHEMAS.md
  section 18 accept `F-ARCH-nn` as a second seed form. It is an extension, not a loosening: the ids are unique,
  seed rows are never activations (KB-03/KB-04), and new rows must use the `F-<yyyymmdd>-<agent>-<n>` form
  (`MFC ledger append` assigns it). KB/README.md states it.
- D2 Seed row `F-UF-16` keeps the digest's recorded status `active` (the KB ledgers are append-only once
  committed; the row is kept as transcribed). The kit's fold ignores a finding row's own `active` without a
  valid activation row (`kb.finding_state`), so `MFC kb status` shows F-UF-16 as provisional with caveat
  "uninformative manipulation". KB/README.md now says so instead of calling it active.

