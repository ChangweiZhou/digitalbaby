# E21b: resume qualified on the merged harness (report, 12 November 2026)

Brief: `OPUS_TASK_E21B_RESUME_QUALIFICATION_20261111.md`. This is an engineering job. It runs no experiment and makes no claim about learning. It answers the E21 section of `PROJECT_DOCUMENTS/MINIFLY_HOSTILE_AUDIT_E20B_STEP0_E18_E21_20261111.md`.

**Provenance**

| | |
|---|---|
| Start | Default branch `codex/minifly-consolidated-20261023` at `a301ae8e` (review 9.98: harness merge, resume gate) |
| Work branch | `claude/funny-mendel-34puq7`, one commit on top of `6a281e66` (the default branch at delivery). The amendments and status note are at `ffdbabb2`; `6a281e66` adds only the E26 scorer, and the seven harness modules are identical in both. The work began on `a301ae8e` and was rebased; the seven harness modules are byte-identical between `a301ae8e` and `ffdbabb2`, so the 47 checks and the "unchanged" comparisons still refer to `a301ae8e` |
| Machine | cloud container, host name `vm`, 4 vCPU x86_64, Linux 6.18, Python 3.13.18 |
| Software | torch 2.15.0+cpu, float32, one torch thread per run |
| Load | at most three harness processes at a time |

**Review status.** No independent review has been run, and none is claimed. As the brief says, Codex audits this result before resume is switched on. The coordinator's own kill-and-resume test on this branch is also still to come.

**What changed:** the resume code of `exp16_integrated.py` and the check script `e21_resume_check.py`. The full diff against `ffdbabb2` (identical to `a301ae8e` for these files) is `exp45/E21B_HARNESS.diff`. No other module, receipt or birth file was touched.

## Results in one table

| Record | Harness tested | Result |
|---|---|---|
| `exp45/E21B_BASELINE_CHECK.json` | `a301ae8e` as merged, harness sha256 `f4559971…be50ae` | **47 of 47 passed** (10,705 s wall) |
| `exp45/E21B_FIRST_ATTEMPT_CHECK.json` | first version of this change, `7b187600…4f9a`; **withdrawn** (see "What failed first") | 76 of 76 passed |
| `exp45/E21B_SECOND_ATTEMPT_CHECK.json` | this commit's harness, with check-script `b422a99a…`; **withdrawn** (see "What failed first") | 78 of 78 passed |
| `exp45/E21B_PRE_AMENDMENT_CHECK.json` | the harness before the brief's amendments, `1df24530…de20c`, check script `0ead3b97…33e3`; **superseded** by the amendments, not withdrawn | 78 of 78 passed |
| `exp45/E21B_PRE_AMENDMENT_QUALIFICATION_SUPERSEDED.json` | that run's qualification file | names five sources; the gate cannot use it |
| `exp45/E21B_RESUME_CHECK.json` | this commit: harness `c9fcedcf…6350` (`dirty: false`), check script `4a4dc4f9…8406` | **97 of 97 passed** |

The record's `branch_commit` is `1cdde8b9`, the work-in-progress commit the run was made from; it was then squashed into this one commit. The two files it tested are byte-identical here (the hashes above), and so are the other five sources (the qualification's hashes).
| `exp45/RESUME_QUALIFIED.json` | written by that run | names the six source hashes, torch 2.15.0+cpu, float32, the seven shapes, the record's sha256 `51b30b7f…d9ca` and the host |

The 97 checks are:

| Checks | What they are |
|---:|---|
| 47 | E21's original checks, unchanged, run on this harness |
| 25 | E21b fixtures: the 17 before the amendments and 8 for them |
| 19 | "able to fail" checks: each fixture rerun against a deliberately defective harness, where it must fail (12 before the amendments, 7 for them) |
| 2 | default-path checks |
| 4 | static checks for the amendments, each also run against the pre-amendment harness, where it must fail |

## 1. The 47 checks on the merged sources

I ran them first, on `a301ae8e` exactly as merged, before changing anything. The run used a fresh test directory, with `E16_RESUME_UNQUALIFIED=allow` set where a check resumes (`exp45/E21B_BASELINE_CHECK.json`). **All 47 passed.** So nothing had to be reported before the changes.

The only edit made before that run was to the check script. It set the allowance variable inside the cleaned environment of the resuming steps, and set the "unchanged" reference to `a301ae8e`.

The same 47 pass again on the changed harness. The seven "unchanged when not resuming" checks compare the receipt and the weights hash with `a301ae8e`'s harness, and all seven are equal:

| Shape | Weights sha256 |
|---|---|
| T drift online store | `25ed64e5` |
| D reference | `e75d9797` |
| M block admission | `f2ea189a` |
| bank2 Phase 4 | `37e6fb2c` |
| bank2 Phase 4, loss all | `4dba8a76` |
| T compgap 6 | `4c0d9c96` |
| birth rule T, local heads | `2e71e723` |

All seven are the hashes E21's report gives.

## 2. Final publication

- **Final receipts are written by temporary file and rename.** `write_json` writes `<receipt>.tmp`, fsyncs it, and renames it into place. It does this for every receipt write of a life (the three per-stage writes and the final one) and for the final receipt of a birth. The content is byte for byte what `a301ae8e` writes, as the seven unchanged checks confirm.
- **A receipt is a result only if it parses and carries its completion record.** `receipt_state` sorts a receipt into one of four states:

  | State | Meaning |
  |---|---|
  | absent | no receipt file |
  | torn | does not parse |
  | incomplete | parses, but has no completion record |
  | complete | for a birth, `complete: true` and a weights hash; for a life, a `complete` record |

- **Birth.** With `E16_RESUME=1` and a checkpoint of this birth, a torn or incomplete receipt is moved aside to `<receipt>.<state>.<UTC time>`. It is never deleted and never read. The birth then resumes from the checkpoint. A torn receipt in any other case stops the run with that reason instead of a parse error.
- **Life.** With `E16_RESUME=1`, a torn receipt is moved aside the same way and the life resumes from its checkpoint.

### Legacy births stay as they were

Of the 76 birth receipts in `exp45/`, 40 predate the completion record: they have no `complete` field and no weights hash, and no checkpoint. A plain rerun of such a birth still prints "exists with the same configuration; nothing to do", as before, whether resume is off or on. Its receipt and weights are left alone.

This was the first attempt's defect (see "What failed first"). The two default-path checks pin it, on this harness and on `a301ae8e`'s.

## 3. Fixtures, each shown able to fail

All fixtures except the birth ones run on the T drift online store life at 16 updates per stage. A fixture first runs the life and kills it in stage 3, then changes one thing and asks for a resume. Each defective harness ("mutant") is this commit's harness with exactly one guard removed:

- for the configuration guards, the named keys are left out of the comparison;
- for the pairing check, the check is disabled.

| Fixture | This harness | Defective harness, which must fail the fixture |
|---|---|---|
| A source changed after the checkpoint (`arch17.py` appended to) | refused, naming `sources.arch17.py` and both hashes | without the source hashes: resumed (rc 0) |
| A checkpoint from another birth: weights perturbed, admitted by a manifest written for it in the test directory | refused, naming `birth_sha256` | without the birth hash: resumed |
| `E11_STAGE_UPDATES` 16 → 32 (checkpoint renamed to the new stem) | refused, naming `E11_STAGE_UPDATES 16 → 32` | without that key: still refused, but by the replayed-exposure check, without naming the setting. The fixture fails on the naming requirement. |
| `E11_DEV_UPDATES` 8192 → 12288 (a birth killed at update 5096) | refused, naming `E11_DEV_UPDATES 8192 → 12288` | without that key: still refused through `birth.updates`, without naming `E11_DEV_UPDATES` |
| `.ckpt` (stage 4) beside a `.ckpt.prev` from stage 1 of the same run | refused: "not one run's two newest checkpoints … not consecutive" | without the pairing check: resumed |
| `.ckpt.prev` from another run of the same life | refused: "run id … differs" | without the pairing check: resumed |
| torch version edited in the checkpoint (re-sealed; `.prev` removed) | refused, naming `torch_version '0.0.0+edited' → '2.15.0+cpu'` | without the torch version: resumed |
| Kill after the final weights, before the final receipt: life, birth | resumed from the last checkpoint (stage 6; update 8192); receipt and weights equal to the uninterrupted run | not run: no guard to remove (this case already passed in E21) |
| Kill half way through writing the final receipt: life | resumed exact; the receipt on disk at the kill was the earlier stage receipt, intact, and the half-written text was in `.tmp` | `a301ae8e`: the receipt on disk was torn. The fixture fails; the life itself still resumed exactly. |
| Kill half way through writing the final receipt: birth | resumed exact from update 8192 | `a301ae8e`: the receipt on disk was torn, and the resume stopped with `JSONDecodeError` |
| Final receipt cut to half beside valid weights and checkpoint, resume asked: life | set aside (`….json.torn.<time>`), resumed exact | `a301ae8e`: resumed exact but set nothing aside; it overwrote the torn receipt. The fixture fails on the set-aside requirement. |
| The same for a birth | set aside, resumed exact from update 8192 | `a301ae8e`: stopped with `JSONDecodeError`, the gap Codex and the coordinator found |
| Gate: no qualification file | refused: "RESUME_QUALIFIED.json is absent" | — |
| Gate: this directory's sources, this shape listed | resumed exact; `resumed[].qualification = "qualified"` | — |
| Gate: this directory's sources, another shape listed | refused: "not on the qualified list" | `a301ae8e` (sources only): resumed |
| Gate: one source hash wrong | refused, naming `exp45.py` | — |
| Default path: rerun of a legacy birth, resume off, then resume on | "nothing to do", rc 0, files unchanged, nothing set aside | the first-attempt harness fails it, shown by hand (below) |

### What "able to fail" does and does not show here

- **Dose.** For both dose changes, a defective harness still refuses, through another check: the replayed exposure for a life, the birth's own `updates` for a birth. What the fixture shows is the brief's requirement that the setting be named.
- **Life receipts.** For lives, `a301ae8e` already recovered the result from a torn final receipt. It fails the two life fixtures because a torn receipt existed on disk, or because nothing was set aside, not because its result was wrong. The failure that matters, a parse-error stop, is the birth's.

## Amendments 1 to 4 (status note, `ffdbabb2`)

The brief's amendments were written after this branch began. The pre-amendment result (78/78) stands as a record but is superseded; everything below was rerun on the amended harness in a fresh work directory.

1. **Source closure.** `RESUME_SOURCES` now includes `mod97_local.py`, which `exp45.py` and `exp3_state.py` import. The check script walks the imports of `exp16_integrated.py` (Python AST, modules of this folder only, transitively) and requires that set to equal `RESUME_SOURCES` exactly (static check "source closure"). The pre-amendment harness fails it: it lists five sources.
2. **Effective values.** `run_config()` no longer falls back to the raw environment for anything. Every option is recorded by its effective value; the four that were recorded only when set (`E16_FARHEAD`, `E16_FARCTRL`, `E16_HEAD1LOSS`, `E16_LOSSW`) are now always recorded, and `E16_LEGACY_BIRTH` is normalised to `allow`/`off`. A static check reads every `E11_`/`E16_`/`EXP45_` variable the closure's files read and requires each to be a key of `run_config()` (other than `E16_RESUME` and `E16_RESUME_UNQUALIFIED`). The pre-amendment harness fails it, missing the four far-head and loss keys.
3. **Runtime in the gate.** The gate refuses unless the qualification's torch version and default dtype are the running ones. The command shape already held every option that changes the loop; it now includes the four keys above at their effective values.
4. **Check record on disk.** The gate refuses unless the check record the qualification names is on disk in `exp45/` and has the sha256 the qualification names.

Gate order: file absent → sources differ (every differing file named) → runtime → record absent → record hash → shape not listed → qualified. `E16_RESUME_UNQUALIFIED=allow` still overrides a refusal and is recorded.

**The status note's tests, all through the production gate:**

| Fixture | This harness | Defective harness, which must fail the fixture |
|---|---|---|
| A changed dependency (`mod97_local.py` appended to after qualification) | refused: "names other sources: mod97_local.py" | without `mod97_local.py` in the source list: resumed |
| Mismatched runtime: torch version in the qualification `0.0.0+edited` | refused: "the runtime is not the qualified one" | without the runtime check: resumed |
| Mismatched runtime: dtype `torch.float64` in the qualification | refused, the same | without the runtime check: resumed |
| Altered check record (bytes changed after qualification) | refused: "does not have the hash the qualification names" | without the record checks: resumed |
| Check record absent | refused: "not on disk" | without the record checks: resumed |
| Explicit defaults set on resume only (`E16_FARHEAD=0 E16_FARCTRL=none E16_HEAD1LOSS=answer E16_LOSSW=uniform`) | resumed exact from stage 2 | pre-amendment harness: refused, naming `E16_FARCTRL`, `E16_FARHEAD`, … as changed |
| Explicit defaults set on the first run only | resumed exact from stage 2 | pre-amendment harness: refused, the same |
| Explicit defaults through the production gate | resumed exact, `qualification = "qualified"` | — |

The positive controls are the existing ones, rerun on this harness: a valid resume through the gate (`gate_listed`, resumed exact), the seven "unchanged when not resuming" checks (fresh-run parity with `a301ae8e`: weights hashes `25ed64e5`, `e75d9797`, `f2ea189a`, `37e6fb2c`, `4dba8a76`, `4c0d9c96`, `2e71e723`, as before), and the 47 checks' exact resumes.

The mutants for these fixtures are this harness with one guard removed by text replacement, as before: `mut_closure` drops `mod97_local.py` from the list, `mut_runtime` disables the runtime branch, `mut_record` disables both record branches.

## 4. The qualified list and the gate

`resume_shape(arm)` states a run's command shape as the settings that change what the loop does:

> arm, architecture, replay, stream, admission, loss, heads, birth rule, credit through time, feedback, projection feedback, fixed codes, book decay, store, freeze, `E16_RSHARE`, comparison gap, far head, far control, head-1 loss, loss weighting, schedule, window and hidden size.

Doses, seeds and output tags are not part of the shape.

`resume_gate(arm)` passes only if all of these hold:
1. `exp45/RESUME_QUALIFIED.json` exists;
2. it names exactly the current six source hashes;
3. its torch version and dtype are the running ones;
4. the check record it names is in `exp45/` with the sha256 it names;
5. it lists this exact shape.

Otherwise it refuses and gives the reason. `E16_RESUME_UNQUALIFIED=allow` overrides a refusal, and the receipt records that it did.

**Qualified: the seven E21 shapes and nothing else.**

| Name | Arm | Arch | Replay | Stream | Admission | Loss | Heads | Birth rule | Credit through time | Other |
|---|---|---|---|---|---|---|---|---|---|---|
| T_drift_online_store | T | window | raw | drift | online | answer | top | D | none | store on |
| D_reference | D | window | uniform | blocks | block | answer | top | D | none | |
| M_block_admission | M | window | raw | blocks | block | answer | top | D | none | |
| bank2_phase4 | T | bank2 | raw | blocks | block | answer | top | D | all | book decay off |
| bank2_phase4_lossall | T | bank2 | raw | blocks | block | all | top | D | all | book decay off |
| T_compgap6 | T | window | raw | blocks | block | answer | top | D | none | comparison gap 6 |
| birth_rule_T_local_heads | birth | bank2 | uniform | blocks | block | answer | local | T | all | book decay off |

In all seven, freeze is `none`, `E16_RSHARE` is 0.5, there is no far head, and the schedule is `blocked`.

**Not qualified:** every other shape. That includes:
- balanced, share, reservoir, total and oracle replay;
- `uniform` replay outside the two listed shapes. E21 tested it only there: `D_reference` is the default replay, and in the birth the setting is unused;
- `E16_FREEZE`;
- `E16_HEADS=aux` and `relay`;
- `E16_RSHARE` other than 0.5;
- `E16_FARHEAD` and `E16_FARCTRL`. Resume for `E16_FARCTRL=shuffle` is refused outright in any case: its private generator is not in the checkpoint;
- `E16_HEAD1LOSS` and `E16_LOSSW=balanced`.

Because the gate compares the whole shape, a setting is qualified only in a listed combination.

**Checked live** against the pre-amendment file:
- passed: T drift online store, D reference, the birth;
- refused: T with `uniform` replay, bank2 with `E16_FREEZE=bank1`, a birth with relay heads, T drift with `E16_RSHARE=0.25`.

**Checked live again** against this commit's file, in a copy of the folder:
- passed: T drift online store; the same with the four explicit defaults set;
- refused: T drift with `E16_RSHARE=0.25` (shape), the record with one byte appended (hash), the record removed (not on disk), the qualification's torch edited to 2.14.0 (runtime), `mod97_local.py` appended to (sources, naming it);
- with `E16_RESUME_UNQUALIFIED=allow`, the changed dependency resumed and was recorded as "unqualified, allowed".

## 5. The qualification file

`--write-qualification` writes `exp45/RESUME_QUALIFIED.json` only if all of these hold:
- the run passed on the full plan (`--only all`, all seven shapes, not dirty);
- the record's tested harness hash equals the committed `exp16_integrated.py`;
- the record's torch version is the current one.

After the amendments it takes the source hashes from the harness's own `source_hashes()`, so the file and the gate cannot disagree about the list.

The file contains:
- the six source hashes: `exp16_integrated.py c9fcedcf…`, `arch17.py 54a99916…`, `exp11_cell.py 6b94db52…`, `exp45.py c95842e4…`, `exp3_state.py 5c284b4a…`, `mod97_local.py f58c66ed…`;
- `torch 2.15.0+cpu`, `float32`;
- the shape list and shape names;
- the check record's name and sha256;
- the host and the machine.

`e21_resume_check.py --qualify-from RECORD` rebuilds the file from a passed record under the same conditions.

The second attempt's writer listed "uniform replay" as not qualified, which contradicted two listed shapes. Its text was corrected (the `qualified` and `not_qualified` fields) before the final run, and the final run wrote the file itself. Nothing in it was edited by hand.

## What failed first, and why

1. **My check script's kill specification.** The first smoke test of the life fixtures crashed in the child. It read `publish:torn` as numbers, and the receipt-state hook assumed a receipt existed. This was a defect of the check script, not of the harness. I fixed it and reran.
2. **The legacy-receipt default path (withdrawn first attempt).**
   - The first version treated a birth receipt without a completion record as "incomplete", and stopped a plain rerun even without `E16_RESUME`.
   - 40 committed birth receipts are of that kind. A plain rerun of one, for example `E16_ARCH=bank2 E16_BOOKDECAY=off … dev 73129`, stopped with rc 1 where `a301ae8e` prints "nothing to do".
   - None of the 47 checks or 29 fixtures could see it, because they all run in fresh directories. I found it by reading the diff.
   - The first full run had already passed 76/76 (`exp45/E21B_FIRST_ATTEMPT_CHECK.json`). Its qualification file is kept as `exp45/E21B_FIRST_ATTEMPT_QUALIFICATION_WITHDRAWN.json`. It names the defective harness's hash, so the gate cannot use it.
   - The fix: an incomplete receipt is set aside only with `E16_RESUME=1` and a checkpoint. I added the two default-path checks and reran everything in a fresh directory.
3. **Disk full during the second run.** The container's disk allowance ran out four hours into the second run. One birth fixture's first step died, and the runner stopped on the missing receipt.
   - I deleted my earlier work directories; their records were already saved.
   - I discarded the one job that had finished while the disk was full (`gate_listed`) and restarted the runner on the same directory. The rest of the plan finished, with every fixture rerun because of defect 4 below.
   - No result in the final record comes from a run that saw the full disk. I checked every run log in the work directory for "No space left", and none contains it now: the four affected runs were redone.

4. **My check script, again (withdrawn second attempt).** The second attempt passed 78/78 (`exp45/E21B_SECOND_ATTEMPT_CHECK.json`). Checking the restart then showed two defects of the check script:
   - **The fixture cache never hit.** Its key held tuples, which come back from JSON as lists, so the comparison always failed. Every fixture reran after the disk-full restart: wasted time, no wrong result. The 47 original checks use a different cache, and it worked.
   - **A late-bound name.** The `pairs_other_run` fixture built its directory name from the loop variable when it ran, not when it was set up. So it ran in `gate_other_source`'s directory, and the two fixtures may have shared it. Both verdicts were correct, but they cannot be trusted from a shared directory.
   - I withdrew that record and its qualification, and fixed both defects (the key is normalised through JSON; the name is bound when the job is made). I deleted the shared directory and reran in the same work directory. The harness was unchanged, so the 47 checks' runs and the other fixtures' results were kept, as the runner is meant to keep finished jobs. The two affected fixtures and their defective-harness runs reran in their own directories. Pre-amendment record: 78 of 78.

5. **The amendments were missed.** The brief's amendments and status note were added after this branch began at `a301ae8e`, and my first delivery (78/78) did not implement them. I rebased onto `ffdbabb2`, implemented them, and requalified. One defect of my own showed up on the way, in the check script: its "unchanged modules" assertion tripped on unrelated new files of the default branch, and was narrowed to the harness's imported modules.
6. **Container restart during the requalification.** The container restarted at about 20:00 UTC, 4 h 27 min into the run, and killed the runner. The harness and check script were unchanged, so I restarted the runner on the same work directory; it kept the 69 finished jobs and ran the rest. No run log contains "No space left".

## Cost

| Run | Wall time |
|---|---|
| Baseline, 47 checks on `a301ae8e` | 2 h 58 min |
| First attempt, withdrawn | 4 h 26 min |
| Second attempt, withdrawn; its runs are reused by the final record | about 5 h 45 min across two parts (05:23 → about 09:30 UTC, then 09:50 → 11:29), out of about 6 h 6 min elapsed; the gap is the disk-full stop and restart |
| Pre-amendment record: the second attempt's work directory, with the affected fixtures rerun | 636 s (11:35 → 11:46 UTC) |

| Requalification after the amendments (this record), fresh work directory | 15:33 → about 20:00 UTC, stopped by a container restart (last job finished 19:36), then 2,948 s (20:01 → 20:50); about 5 h 17 min of wall time in all |

- The requalification ran 115 jobs, at most three at a time; its work directory holds 2.7 GB.
- The pre-amendment runs used about 48,900 s, 13.6 h, of process time summed over their jobs.
- Each 8,192-update birth takes about 27 minutes on one core. The final plan has 13 of them: 5 from the 47 checks and 8 from the fixtures.
- The brief's forecast came from E21's single run of a little over an hour on its 47 checks. The fixtures and mutants roughly double the plan.

## Limits

- **Doses.** Qualified means tested at the doses E21 used: 16 and 512 updates per stage for lives, 8,192 for the birth, seed 73129. A full-dose resume was not run. E21 estimated replay at full dose from its 512-update figures, and the gate does not look at the dose.
- **Torch.** The gate now checks the torch version and default dtype against the qualification (amendment 3), and each checkpoint's configuration binds them as before. The host is recorded, not checked. Other runtime libraries (numpy, Python) are not checked.
- **Static checks.** The closure walk sees static imports only (`import x`, `from x import y`); a module loaded by name at run time would not be found. The option scan is a regex over the closure's source for `E11_`/`E16_`/`EXP45_` names; an option read under a computed name would not be found.
- **Kill-and-resume by the coordinator** and Codex's audit are still to come, as the status note says.
- **Other birth.** The other-birth fixture fabricates a perturbed birth and a manifest for it inside the test directory. Real births and the real manifest were not touched.
- **Mutants.** Each defective harness removes exactly one guard by text replacement. They show that each fixture can fail; they are not a proof that no other path exists.

## Files

| File | What it is |
|---|---|
| `exp16_integrated.py` | the harness, resume code only |
| `e21_resume_check.py` | the checks |
| `exp45/E21B_HARNESS.diff` | both files against `ffdbabb2` (identical to `a301ae8e` for them) |
| `exp45/E21B_BASELINE_CHECK.json` | the baseline record |
| `exp45/E21B_FIRST_ATTEMPT_CHECK.json` | the first-attempt record |
| `exp45/E21B_FIRST_ATTEMPT_QUALIFICATION_WITHDRAWN.json` | the first attempt's withdrawn qualification |
| `exp45/E21B_SECOND_ATTEMPT_CHECK.json` | the second-attempt record, withdrawn |
| `exp45/E21B_PRE_AMENDMENT_CHECK.json` | the pre-amendment record, 78/78, superseded |
| `exp45/E21B_PRE_AMENDMENT_QUALIFICATION_SUPERSEDED.json` | its qualification |
| `exp45/E21B_RESUME_CHECK.json` | the check record |
| `exp45/RESUME_QUALIFIED.json` | the qualification |
| this report | |
