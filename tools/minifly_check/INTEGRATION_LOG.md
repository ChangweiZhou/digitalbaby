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

