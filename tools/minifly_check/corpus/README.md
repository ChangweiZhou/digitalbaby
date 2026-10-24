# MRK-1 regression corpus

The L1 layer of SPEC Section 12: one or more synthetic reproductions per failure class FM-01..FM-30,
plus clean fixtures that must pass (blocking legitimate work is a failure too). `MFC selftest` runs
every fixture with TimelineGit and compares the results with `tools/minifly_check/EXPECTED.json`;
selftest never runs the builder.

```
corpus/
  _build/                build_corpus.py and its modules (stdlib only; never import checker code)
    base/                the hand-designed clean base (TimelineGit format) + EXPECTED_VALUES.md
  C00a .. C30, G01 .. G05, K01     one directory per SPEC 12.2 row
  S01 .. S64             supplementary fixtures (not SPEC 12.2 rows), see below
    FIXTURE.json         {id, title, fm_ids, source_incident, kind dirty|clean, built_from, mutation, ...}
    timeline.json        {head, dirty, shallow, commits: [{sha, utc, author, trailers, paths, branches, label}]}
    repo/                final tree (worktree)
    history/<sha>/<path> content at a commit whose content differs from repo/
  heldout/README.md      sealed held-out batch protocol (SPEC 12.3); no held-out fixture lives here
```

Regenerate and verify (both must be run from the repository root):

```
python3 -I -B tools/minifly_check/corpus/_build/build_corpus.py          # writes corpus/ and EXPECTED.json
python3 -I -B tools/minifly_check/corpus/_build/build_corpus.py --check  # byte-identical rebuild + validation
```

## The base

`minifly_corpus_base_dev_v1_20261101`: T2, MAIN, E3 target on single-digit addition with the
`pair_rectangle` witness; arms W (candidate), NPT (no_prior_teaching), SHUF, FROZEN, ORACLE
(oracle_plumbing), the CONSTRAINTS ceiling and floor grids as CEILING_*/FLOOR_* testbed arms; 8 listed
MAIN worlds, 2 dev, 1 rehearsal, 8 technical (testbed) worlds of which w_tech_1 carries the native
technical run; 15 taught units, pools TAUGHT 15 / OLD 8 / NEW 7 / HELD 16. Thirteen commits take it
from the KB overlay through lock, cycle 0, learner, technical, cross-family review, roster, dev,
SOURCE_LOCK ack, rehearsal and the official run to REPORT.md. Every number (strata, determinacy, ladder,
best shortcut, G, guards, verdict, headline) is hand-computed in `_build/base/EXPECTED_VALUES.md`.
Native, runner and review `.py` files are tiny stubs: parsed by the checker, never executed.

Each dirty fixture is the base plus the one mutation in its FIXTURE.json `mutation` (some are a
variant first: C22 builds on C00e; lock-stage fixtures end their timeline at the commit the scenario
needs). Executor artifacts that follow from the mutated design (row digests, ATTEMPTS shas, receipts,
REPORT headline, the executor copies of TESTBED.json and dev SCORES.json) are regenerated consistently,
so nothing else differs.

## Supplementary fixtures S01-S64

The SPEC 12.2 table leaves many check ids without a fixture in which they fire, so `selftest --mutate`
would report them as 'check not exercised' (SPEC 7.7, 20.2). S01-S25 add one documented mutation each
for those ids (SELF-01, SCHEMA-01, LOCK-01/02/04/10, AUT-01, AMD-03, RED-08..11, LAD-01/02, STR-01,
DET-01, ADM-01, CYC-01/02, RUN-02, CLM-01/07/11/13, ERR-01, REV-03..06, KB-01/02, PTR-01/02, DEBT-02,
SEAL-02..05, CONF-01, REV-08, GOV-APPEND, GOV-SKILL, GOV-REV; C27 also lists DEBT-01). S26-S45 (integration
round 1) cover the rest: RED-02..07 (S26: an undecidable rule), ADM-REUSE, CLM-03, CLM-08, CLM-14, TIER-01,
KB-04, GOV-SELF, PTR-07, the advisory CLM-16, LAD-04, WLD-02 and PTR-06 (must_warn), CAN-01..03 and
HELD-01..03 (S44/S45: a visible toy batch that tests the held-out mechanism, never held-out recall), and the
clean S35, S42 and S44 constrain the PTR, DEBT-01 and HELD ids. With them `selftest --mutate` reports every
non-planned id exercised and constrained. UDI-01 is registered as planned (README 'Not yet enforced').

S46-S50 (integration round 2) cover input tampering of the cycle accounting and the honest INVALID publish:
S46 deletes the committed runlog/CYCLES.jsonl in the worktree, S47 never closes a cycle, S48 is a T1 dev-only
delivery that never closes one, S49 rewrites a committed cycle row (all fire CYC-01), and S50 publishes an
official INVALID TREATMENT_NOT_DELIVERED on 8 MAIN worlds (the claim states INVALID, never NOT_OPENED).

S51-S54 (integration round 3) cover the testbed and split paths: S51 relabels dev work as a testbed attempt
after the cycle cap is used (CYC-01: a testbed attempt writes only results/admission/ CEILING_*/FLOOR_* rows);
S52 edits a HELD manifest item after the lock and S54 deletes the OLD manifest (AMD-05: the split manifests
hash to split.held_set_sha256 and split_sha256 at every commit after the lock); S53 swaps an out-of-manifest
item into the official HELD rows (LAD-01, and the scorer refuses rows outside the locked manifests).

S56-S64 (hardening round 4, SPEC 20.5-20.9): S56-S58 put confirm wording on DEV inside a snake_case SUMMARY.json
label, a CamelCase REPORT label and behind a zero-width space (CLM-04); S59 (clean) makes the undeclared FROZEN arm
answer HELD exactly like W, an honest FAIL through the decision comparator; S60 imports a bootstrap.py at runtime that
SOURCE_LOCK lacks (RUN-03 closure gap); S61 (clean) splits the official run over an interrupted attempt and a legal
`launch --resume` with a bit-identical replay; S62 rewrites a completed unit in the resume (RUN-02), S63 omits the
replay and S64 replays with a different digest (RUN-04). Since round 4 the base writes
`runlog/RUNTIME_CLOSURE_<attempt_id>.json` for every counted attempt (listed in its produced_paths) and every
committed CLAIM.json carries the generated `provenance` field.

## Adding a fixture from an incident (through governance only)

1. The incident exists first: an INCIDENTS report row (`MFC ledger append --ledger INCIDENTS`) with
   caught_at_discovery and the checker sha. Post-freeze incidents eligible for held-out recall get
   their fixture written by someone other than the check author and committed by hash before any check
   is written (SPEC 12.3 step 7).
2. Write the mutation as a named function in `_build/fixtures_*.py` and one entry in `_build/table.py`
   (id, FM ids, source incident, built-from source, mutation text, commands, must_fire with the exact
   spec check ids, expected_exit per command, expect pointers). Never hand-edit files under a fixture
   directory: the builder owns them.
3. Run the builder, then `--check` (byte-identical and valid). Keep the fixture small.
4. Commit through governance only: a separate commit range touching protected paths only (GOV-SEP),
   classified by `MFC governance --base B --head H`; a new fixture is a tightening and needs a
   cross-agent `KB/GOVERNANCE/REVIEW_<id>.json` cited in a `Governance-Review:` trailer (GOV-REV) and
   GOV-MONO (fired sets only grow; clean fixtures stay clean).
5. Removing a fixture, removing a must_fire id or weakening an expectation is a loosening: it needs an
   owner User-Decision whose verbatim words restate the changed numbers (GOV-UD, GOV-NUM). During Phase 0
   every change to EXPECTED.json or a fixture is also logged in `tools/minifly_check/INTEGRATION_LOG.md`
   with the spec clause that justifies it (SPEC 20.3).
6. Close the incident with a closure row naming the check id and this fixture (KB-05). A fix must not
   string-match fixture or canary literals.

## Interpretations

Choices made where SPEC.md or SCHEMAS.md is silent or ambiguous (stricter reading where there was a
choice). The builder keeps each one in one place so the integrator can change it and rebuild.

1. **Technical worlds.** The lock lists 8 technical worlds so the cycle-0 testbed can pass the locked
   rule as written (paired_lcb, min_worlds 8); the native technical run uses w_tech_1 only ("one
   technical record per arm"). WP7 suggested 1 technical world; with one world no ceiling can pass the
   delivery's own rule.
2. **Testbed scoring.** TESTBED.json (executor copy) scores each config against max(chance, the ladder
   built from W's TAUGHT manifest and the config's own stream); comparator arms have no rows at C0.
3. **Cells.** Native arms write TAUGHT@prefix, OLD@old_end, OLD@final, NEW@final and HELD@final;
   testbed arms write HELD@final, OLD@old_end and OLD@final (the primary and both guards), a superset
   of what the rule reads.
4. **Split hashes.** held_set_sha256 = sha256 of HELD.jsonl; split_sha256 = sha256(canonical_json of
   {held_manifest, old_manifest, new_manifest: file shas}). This is the checker's formula
   (`common.split_hashes`, SCHEMAS.md section 1): AMD-05 recomputes both at every stage and at every commit
   after the lock. TAUGHT streams are committed with the cycle-0 testbed, after the lock fixed the held set
   ("before any stream exists", SPEC 6.2; LOCK-06 checks that order).
5. **Closure hashes.** closure_sha256 keys are repo-relative and include files.admit;
   native_closure_sha = sha256(canonical_json(native_closure_sha256)); runner_closure_sha likewise over
   runner plus native files.
6. **Committed gate reports** (CYC-02 exit criteria and the reports launch wrote) carry git_mode
   `timeline` and checker_sha256 = sha256 of the empty string: fixture trees hold no kit files, so the
   SCHEMAS 19.4 listing is empty (CLM-14).
7. **Checker-written copies.** results/admission/TESTBED.json and results/dev/SCORES.json are executor
   copies from `_build/refcalc.py`; `MFC score` rewrites them. No VERDICT.json or CLAIM.json is
   committed except where an edited CLAIM is the incident (C07, C21, C26).
8. **expect pointers** are bare JSON pointers resolved against the artifacts the fixture's commands
   produce: /verdict and /guards/* in VERDICT.json; /headline, /primary_gate, /level_earned,
   /worlds_role, /stop_reason and /objective_metric/* in CLAIM.json; /designs_tried_k in CLAIM.json or
   SCORES.json (same value); K01's /brier_mean and /skill in the `kb score --json` report.
   `{"contains": s}` is a substring test.
9. **Clean fixtures** list every id of every bundle their commands run in must_not_fire (fired = FAIL
   or MISSING), and expect exit 0 for every command.
10. **Ties** in the ladder go to the lexicographically smallest answer by bytes; the base is designed so
    every number that reaches a headline is tie-free (DIGIT_NO_CARRY is the unique best in every world).
11. **Headline wording** for cases the grammar does not pin down is a best reading: FAIL earns `NONE`
    (C00b); C00c reads `held exact own-EOS not_measured | 0 PILOT worlds | best of k=0`; C00e reports its
    dev numbers with ` | dev shadow FAIL` (2 dev worlds < min_worlds). Numbers use one decimal.
12. **Seal (C00d).** key = sha256(nonce hex string || GO verbatim UTF-8);
    seed_i = first 8 bytes (big-endian) of HMAC-SHA256(key, "L1|L2|confirm_worlds|i") with lowercase
    hex lock shas; world id `w_seal_<16 hex>`; the CONFIRM_LEDGER row is its own commit right after the
    reveal (it names the reveal commit). Replay digest = the REV-08 row digest of SCHEMAS.md section 13
    (canonical rows without attempt_id; integration round 1, INTEGRATION_LOG A4). C00d's dev worlds give W one
    more held item each (mean dev d 0.344) so the SPEC 9.8 OC is about 0.94, as its SPEC 12.2 row says (A5).
13. **C05c** uses margin_abs 0.9 (no synthetic clean positive, 0.9 - 0.05, can clear it); the
    incident's 0.6 is reachable on this base.
14. **C06a** uses FIXED_2: on single-digit addition FIXED_1 can never beat DIGIT_NO_CARRY (both are
    right only on 1-digit answers), so the incident's own-stream truncation is reproduced with k = 2 on
    a HELD set with 12 carry items.
15. **C07** lists LOCK-03 in must_fire: a conditional primary on an E3 objective lock is treated as a
    FAIL, not only a relabel to component (stricter reading).
16. **C27** keeps one executed check (trace_number) so REV-07 has an output to compare; REV-02 fires
    on the three missing kinds.
17. **C21** produces low OC power with weak dev data (W near the comparator on 2 dev worlds); the
    incident's exact 0.41 is not reproduced.
18. **Governance fixtures** G01-G03, G05, S23-S25 and S33 use a miniature protected kit (RULES.json,
    EXPECTED.json, lockcheck.py, five SKILL.md, AGENTS_POINTER.md). C00f instead carries a runnable overlay
    (a full RULES.json copy and nested lock-stage fixtures X01/X02), so GOV-MONO runs the base and head
    checkers on real nested corpora (INTEGRATION_LOG A7).
19. **Size.** A fixture whose timeline ends at the lock is about 28 KiB; a full fixture is about 390 KiB
    (target was about 200 KB): the spec's 12 native worlds x 5 arms of receipts and TAUGHT manifests
    dominate; OUTPUTS are gzip-compressed with a fixed header. 98 fixtures total about 26 MiB.
20. **Commit shas** are sha1("mrk1-corpus|" + label), so the base and its copies share shas; commits
    carry an extra `label` key for readability.
21. **Committed canary report** (`checks/CANARY_20261101T143000Z.json`) has the shape `MFC canary` writes
    (counts, lock shas, CAN-01..03 results, verdict), since REV-05 reads it (INTEGRATION_LOG A1).
22. **Supplementary fixtures S26-S45** (integration round 1, `_build/fixtures_s2.py`) exercise the ids the
    first `selftest --mutate` found without a firing dirty fixture or a constraining clean one: RED-02..07,
    ADM-REUSE, CLM-03, CLM-08, CLM-14, TIER-01, KB-04, GOV-SELF, PTR-07, the advisory CLM-16, LAD-04, WLD-02 and
    PTR-06 (must_warn), CAN-01..03 and HELD-01..03, plus the clean S35 (legacy pointer), S42 (single-agent T2
    with REVIEW_DEBT) and S44 (held-out mechanism). S44/S45 seal a toy one-fixture batch whose plaintext is in
    the builder: they test the mechanism and never count as held-out. The owner key `owner.key` lies beside
    `repo/`. See INTEGRATION_LOG.md A8.
