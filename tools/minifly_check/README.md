# minifly_check: the MRK-1 research-loop checker (schema v1)

One program-owned, standard-library-only checker for MiniFly / digitalbaby deliveries created after
the schema-v1 freeze (UD-0001). It owns the metric, the shortcut ladder, the strata and the decision
rule; a delivery supplies declarations and saved outputs, never verdict code. Normative spec:
[`SPEC.md`](SPEC.md). Field-level schemas: [`SCHEMAS.md`](SCHEMAS.md). Check registry and stage
bundles: [`RULES.json`](RULES.json). Protected paths: [`PROTECTED.json`](PROTECTED.json).

```
MFC = python3 -I tools/minifly_check/minifly_check.py
```

## Exit codes (every nonzero exit blocks; there is no --force and no --skip)

| Exit | Meaning | What you may do |
|---|---|---|
| 0 | PASS (WARN, ADVISORY_FAIL and NOT_APPLICABLE do not block) | proceed |
| 1 | FAIL: an enforced check failed | follow SPEC 3.3 for that check family |
| 2 | MISSING_OR_INVALID_INPUT: something required is absent (and nothing failed); also `legacy delivery: use retro` | supply the input; legacy dirs only get `retro` |
| 3 | CHECKER_ERROR: dispatcher exception, SELF-01 failure, unknown subcommand, usage error, missing module or check function | fix or revert the checker through governance; never work around it |

Global options: `--repo-root PATH` (default: git toplevel of the current directory),
`--git timeline:<fixture_dir>` (selftest and corpus only), `--json` (print the report to stdout).

## Quickstart by stage

| Stage | Commands | Proceed when |
|---|---|---|
| S0 explore (T0) | `MFC kb status`; text you publish: `MFC lint <files>` | owner AUTHORIZATION recorded |
| S1 design and lock | `MFC init <dir> [--parent P] [--claim-level E3] [--task-family addition_single_digit]`, fill every `__FILL__`, write `AUTHORIZATION.json` (template in `templates/`), then `MFC gate --stage lock <dir>` and commit `ENDPOINT_LOCK.json` before any `results/**` | lock gate exit 0 |
| C0 testbed (before learner code) | `MFC launch --stage testbed <dir> -- python3 -I admit.py --testbed`, `MFC score <dir> --role admission`, `MFC cycle close <dir> --kind testbed --decision continue --reason "..."` | a ceiling passes and every floor fails; otherwise CANNOT_TEST_CLAIM_AS_SPECIFIED (stop, or amend with reason testbed_invalid) |
| C1 technical | implement the learner, `MFC lock build <dir>`, commit, `MFC launch --stage technical <dir> -- <runner>`, `MFC launch --stage technical <dir> -- python3 -I admit.py --technical`, `MFC gate --stage roster <dir>`, `MFC cycle close <dir> --kind technical ...` | roster gate exit 0 (at T2 including the cross-agent review event, or a REVIEW_DEBT row) |
| T2 review | the other agent family writes `review/REVIEW.json`, runs `MFC canary <dir> --seed HEX` | decision ADMIT, no open BLOCKER/MAJOR |
| C2 dev | `MFC launch --stage dev <dir> -- <runner>`, `MFC score <dir> --role dev`, `MFC cycle close <dir> --kind dev ...` | `results/dev/SCORES.json` written and a decision recorded |
| C3 rehearsal (fresh world) | `MFC launch --stage rehearsal <dir> -- <same command template as official>`, `MFC gate --stage official --dry <dir>`, `MFC cycle close <dir> --kind rehearsal ...` | dry official gate exit 0 |
| S6 official (always T2) | reviewer acks the final SOURCE_LOCK; for SEALED_CONFIRM the seal protocol (`MFC seal commit-nonce\|reveal\|derive\|verify <dir>`, `MFC oc <dir>`); then `MFC launch --stage official <dir> -- <cmd>` exactly once; after an interruption `MFC launch --stage official --resume <dir> -- <same cmd>` (missing units only, one launch-chosen unit replayed into `results/official_replay/`) | one completed official run (RUN-02; a legal resume chain, RUN-04) |
| S7 audit and publish | `MFC score <dir> --role official`, `MFC claim <dir>`, `MFC gate --stage publish <dir>`; pointers: `MFC gate --stage pointer --pointer P <dir>`; ledgers: `MFC ledger append --ledger FINDINGS --row ROW.json` | publish gate exit 0 |
| Governance | `MFC selftest [--mutate] [--only ID]`, `MFC governance --base B --head H`, `MFC heldout {seal\|reveal\|score}`, `MFC retro (--all \| <dir>) --out DIR` | exit 0 |

Typed cycles: `testbed` (not counted) and `technical`, `dev`, `rehearsal` (counted against
`AUTHORIZATION.caps.max_cycles`, default 3). `cycle close` refuses `--decision continue` until the
kind's exit criterion holds (SCHEMAS.md Section 11). `launch` refuses technical, dev and rehearsal attempts
once the cap is reached (attempts no close covers count as an open cycle), any attempt of another stage while
a cycle is open, a command line naming another stage's world, and any official attempt after one that
produced rows. The official stage needs every test attempt in a closed cycle, every attempt must write only
within its stage (its results directory and worlds; a testbed attempt only CEILING_*/FLOOR_* rows under
results/admission/, SCHEMAS.md Section 10), and runlog/CYCLES.jsonl and ATTEMPTS.jsonl are append-only
(CYC-01).

## What launch does

`MFC launch --stage S <dir> -- <cmd ...>` (S: testbed -> lock bundle, technical -> technical,
dev/rehearsal -> roster, official -> official):
1. runs the stage's gate bundle and writes `checks/GATE_<bundle>_<utc>.json`; refuses on a nonzero exit;
2. refuses when `ENDPOINT_LOCK.json`, `SOURCE_LOCK.json` (required after testbed), `AUTHORIZATION.json`,
   `AMENDMENTS.jsonl` or any closure / `files.*` / endpoint file is uncommitted, modified or differs
   from its locked sha256;
3. applies the cycle cap, the stage's worlds (no world of another stage on the command line) and the
   never-rerun-official rule (`--resume` continues an interrupted official attempt only: identical lock shas, nothing
   rewritten; SCHEMAS.md 10.3);
4. first records any attempt whose launch process died (an in-flight marker `runlog/INFLIGHT_<id>.json` without an
   ATTEMPTS row: exit -1, interrupted, recovered), then writes its own marker and runs the command in the delivery dir
   with inherited stdio and `MINIFLY_ATTEMPT_ID`, `MINIFLY_STAGE`, `MINIFLY_DELIVERY`, `MINIFLY_ENDPOINT_LOCK_SHA256`,
   `MINIFLY_SOURCE_LOCK_SHA256`, `MINIFLY_NATIVE_CLOSURE_SHA`, `MINIFLY_PARAMS_SHA256`, `MINIFLY_REPO_ROOT` and
   `MINIFLY_RUNTIME_CLOSURE` (plus the `MINIFLY_RESUME_OF`, `MINIFLY_COMPLETED_UNITS`, `MINIFLY_REPLAY_UNITS` and
   `MINIFLY_REPLAY_DIR` of a resume) in the environment;
5. appends a `runlog/ATTEMPTS.jsonl` row with RUSAGE_CHILDREN CPU deltas, wall time, peak RSS and every
   new or changed file under `results/` and `manifests/TAUGHT/` (and the runtime closure) with its sha256 - also for
   failed or interrupted commands; official rows record their (world, arm) units with canonical digests - and exits 1
   when the attempt wrote outside its stage (CYC-01 then FAILs) or a resume broke SCHEMAS.md 10.3 (RUN-02/RUN-04).

Every result file must later match an ATTEMPTS row, and every OUTPUTS row must name one (RUN-01, enforced at
T2, WARN at T1). Every counted run writes `runlog/RUNTIME_CLOSURE_<attempt_id>.json` from sys.modules (helper in
`templates/admit_skeleton.py`), and every repo .py file in it must be in that attempt's SOURCE_LOCK closure (RUN-03,
T2 enforced, T1 WARN). Every invocation re-verifies the locks and the closure: run one launch per phase, never
chain phases inside one unlaunched script.

The held set is fixed in the lock: AMD-05 recomputes `split.held_set_sha256` and `split_sha256` from the
manifests at every stage and every commit after the lock, LOCK-06 requires TAUGHT streams to follow the lock
commit, and the scorer and LAD-01 accept only manifest items in HELD/OLD/NEW rows.

## Module map

| Module | Owns | Work package |
|---|---|---|
| `minifly_check.py` | argparse dispatcher, lazy handler imports, exit-code mapping | WP1 |
| `common.py` | Result/CHECKS registry, GitView (RealGit, TimelineGit), KB view, Ctx and row reader, canonical JSON and hashes, tier and lineage family, bundles and advisory handling, gate reports, repo-wide bounded scans; SELF-01, SCHEMA-01, TIER-01 | WP1 |
| `launch.py` | launch wrapper (resume, in-flight markers), cycle close; RUN-01..04, CYC-01..03 | WP1 |
| `lockcheck.py`, `seal.py` | init, lock build, LOCK-*, AMD-*, WIT-01, SEAL-*, CONF-01 | WP2 |
| `addition.py`, `scorer.py`, `decide.py`, `reducercheck.py` | task semantics, canonical scorer, ladder, strata, witness, G, decision rule, OC, RED, DEC | WP3 |
| `admitcheck.py`, `review.py`, `canary.py` | ADM-*, AUT-*, REV-*, DEBT-*, CAN-*, replay-plan | WP4 |
| `claimcheck.py`, `kb.py` | claim, CLM-*, ERR-01, WLD-*, PTR-*, lint, ledgers, KB-* | WP5 |
| `governance.py`, `retro.py` | GOV-*, HELD-*, selftest, heldout, retro | WP6 |
| `corpus/`, `EXPECTED.json` | regression fixtures | WP7 |
| `templates/` | ENDPOINT_LOCK/AUTHORIZATION/REVIEW templates (`__FILL__` = a human decides), examples, `admit_skeleton.py` (never run by the checker) | WP1 |

A missing or broken module fails only the commands and bundles that need it, with CHECKER_ERROR.

## Not yet enforced

Check ids registered in RULES.json with `status: "planned"`: excluded from every bundle and
reported here so nobody mistakes them for live checks (SPEC 20.2). One bullet per planned id, in
the form ``- `ID`: reason``; selftest asserts this list equals the planned ids in RULES.json.

- `UDI-01`: the leave-units-out validity check (E3 promotion only) is implemented in scorer.py and fails
  closed, but no corpus fixture builds UDI variant worlds yet, and SPEC 9.7 does not define the seal-derived
  draw of the dependent set for SEALED_CONFIRM (the function returns MISSING there). It is excluded from the
  publish bundle until both exist; E3 promotion (UDI LCB > 0) is therefore not claimed by any gate. While it
  is planned, `claim` writes reuse_specificity not_tested even when SCORES carries a UDI value, and CLM-03
  reports "E3 promotion not certifiable: UDI-01 planned".

## Known limitations

These parts of enforced checks are incomplete in checker 1.0.0. Each item says what the kit does instead.

- CLM-15 `digest_plus_sample` mode (fails closed): a verification subset above
  `CONSTRAINTS.verification_subset_max_mb` FAILs ("not implemented in checker 1.0.0"). An over-cap
  delivery needs an owner decision (REQUIRES_USER_DECISION) or a later checker.
- REV-08 seal-derived replay seed (not verified): the replayed pairs are checked against
  `MFC review replay-plan` only when `REVIEW.postrun.seed` is recorded, and REV-08 does not check that this
  seed is the seal-derived 'replay' seed of SPEC 10 step 5. Both digests must equal the checker's digest of the
  official rows, and at least
  min(4, available) distinct (world, arm) pairs must be replayed.
- Co-primary endpoints after an amendment (reported, not scored): the scorer does not evaluate
  `lock.coprimary`; `claim` reports each co-primary as NOT_SCORED in `coprimary_results` and in the
  headline suffix ` | amended: co-primary NOT_SCORED`.
- In-flight marker recovery (SCHEMAS.md 10.2) cannot know the exit code or CPU of an attempt whose launch process
  died; the recovered row says so (`rusage_lost`), and compute totals then undercount that attempt.
- A runtime closure (RUN-03) is written by the runner itself; the checker verifies what it lists against the lock but
  cannot see a module the runner hides from sys.modules. A killed run leaves only the closure of its last helper
  call (the contract calls it after the imports and at the end).
- TimelineGit (corpus mode) lists a worktree edit of a committed file only when the fixture carries the
  committed version under `history/`; `selftest` freezes every fixture copy before its commands run, so
  commands that rewrite committed files are seen as worktree edits, as in real git.

## Development

- Python 3.9 or later, standard library only, no network. Kit modules import each other top-level
  (`import common`); `minifly_check.py` and `common.py` put the kit directory first on `sys.path`, so
  `python3 -I` works.
- Git is read-only, through `common.run_git` (fixed argv, 30 s timeout, allowed subcommands only:
  rev-parse, log, show, cat-file, ls-files, diff, merge-base --is-ancestor, branch -r --contains, and
  archive for governance). A shallow clone makes history-dependent checks MISSING.
- The checker never imports or executes delivery or historical code, never trains or runs learners
  itself (launch runs the command you give it), and never writes inside legacy directories.
- Unit tests: `python3 -I -m unittest discover -s tools/minifly_check/tests` (one `test_<module>.py`
  per module). Corpus: `MFC selftest` (fixtures run in forked worker processes; `MINIFLY_SELFTEST_JOBS=N`
  sets the count, default min(4, CPUs)) and `MFC selftest --mutate`. The corpus is generated:
  `python3 -I -B tools/minifly_check/corpus/_build/build_corpus.py [--check]`; every integration-time change
  to EXPECTED.json or a fixture is logged in `INTEGRATION_LOG.md` with the SPEC clause behind it.
- SELF-01 requires the checker worktree to equal HEAD (no modified or untracked kit files) and a
  passing selftest for the current checker sha, cached as
  `<tempdir>/minifly_check_selftest_<sha>.ok`; when the cache is absent the gate runs selftest first.
