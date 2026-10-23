# Cycle 3 corrected-source independent re-review

## Decision: rejected pending three bounded integration corrections

The substantial corrections to the initial rejected revision are present in the actual production source. Three remaining integration gaps below still prevent static acceptance of this frozen closure. No reserved Cycle-3 operations job, native pilot/replay, or official learner world was run or authorized by this review. No external mutation was performed. The reviewer changed only these audit artifacts; pure checks used no learner import.

Design: `protocol/ROUND1_DESIGN_CYCLE3.md`, SHA-256 `1a8c4f3d12bac3521d5e734fc704b594c514f8e064f25ffcb11e3317c25f3842`.

Reviewed source: 126 files, exact `integrity.hashes()` digest `a6cdefa94434bd4fba48b574e1cd47b63190b7ab9b2ae6eee7efc535b95e5ba5`. The companion JSON binds the full nonempty map and this report. The original review remains preserved as `CYCLE3_SOURCE_REVIEW_INITIAL` and in the rejected-revision history.

## Corrections verified by source inspection

1. `recovery.py` now supplies genuine production missing-only planning, completed local receipt/part revalidation, historical accepted-source validation, anchored ledger WAL, append-before-replace recovery, conservative 900-second pre-job reservation, immutable quarantine intent, exact-key retry authorization, and operations → pilot → replay ordering. Repeated recovery does not refund an unknown attempt. Consumed reservations are tied to a host token, and complete-receipt crash windows require independent disposition. Tests exercise these functions rather than reproducing their intended behavior as separate ledger arithmetic.
2. `runner.js` now implements the actual active-tool-host sequence: plan, reserve, privately back up/read back the reservation, acknowledge it, start the exact supervisor key, detect and revalidate completion through the planner, invoke actual Library/GitHub transport, establish manifest-bound barriers, then consider the next key. The supervisor refuses an unbacked next world. The keeper provides a separate active-wall counter, session token, boot identity and held coordinator lock; supervisor polling stops native work if this host goes stale. The worker has race-checked SIGKILL PDEATHSIG plus an inherited supervisor-lock descriptor. These are material fixes to the former standalone controller function. An active authorized tool host is still required, and executor reset is not claimed to be survivable by a local daemon.
3. The receipt writer/validator now includes all four eight-store clock boundaries, mandatory probe clocks/store digests, finite nonnegative flush error, final full-state/store/clock linkage, exact runtime/birth/budget/endpoint/tie/acceptance checks, and science-kind/canonical world-path checks. The new semantic tamper tests recompute manifests and cover the previously accepted omissions. The source-backed birth contract's original archive hash, technical record hash, birth records and birth digests were independently checked against the provided archive without executing the learner.
4. Checkpoint identity now includes results and semantic ledger/accounting changes, with heartbeat/ordinary acknowledgements excluded from archive triggering. ZIP writing/hashing/readback are streaming, metadata staging and temporary I/O have stated caps, and verified duplicate readback cleanup retains preserved originals. A fresh isolated restore invokes the restored pure receipt validator. See R1 for the missing production recovery dependencies.
5. The inherited learner/vendor and fixture are byte-identical to the initial reviewed closure. First-pre-feedback probing, coefficients, scales, fixed official worlds and the 11-contrast world-unit inference are unchanged. The added combined-status reporting distinguishes an established nonpositive interval from an inconclusive/gate-failing result. The pinned license/provenance findings from the initial review remain applicable.

The reported 26 recovery-development checks, 15 synthetic semantic checks and streaming-development checks are useful implementation evidence, not the reserved Cycle-3 test acceptance. I did not duplicate that reserved job.

## Remaining bounded corrections

### R1. Restore the production recovery dependencies and test the actual restored planner

`checkpoint.scientific_files()` and `checkpoint.state_files()` omit `operations/cycle1.log` and `operations/cycle2.log`. Both are mandatory historical acceptance inputs: `recovery._historical()` iterates `CYCLE1_ACCEPTED.json`/`CYCLE2_ACCEPTED.json` `input_hashes` and hashes these paths before admitting any new work. An independent pure set comparison found exactly these missing historical input paths. No log file is currently selected for the checkpoint.

Consequently, a restored archive can pass the current receipt-only `restore_verified()` test and still fail its first production plan with missing historical logs. A restored executable source closure is not sufficient when recovery validation also requires accepted evidence outside that closure.

Bounded fix:
- Include the exact historical acceptance dependencies in the private checkpoint. Include the bounded attempt logs/failed-attempt evidence needed for documented recovery, without adding them to the public projection.
- Extend isolated restore qualification to run the restored production historical validation and missing-only planner under an isolated lock, using the restored ledger/source map, in addition to receipt validation. For a live-origin pre-job reservation, expect its conservative interrupted/pending-review disposition, not a restart or refund.
- Keep current Library-version/acknowledgement bootstrap explicit: archive-contained acknowledgement paths must be rebased or resolved inside the consumer root, and a prior version saved inside an archive must not be assumed to be the current remote version. A required read-only reconciliation may stop safely rather than overwrite or clear uncertainty.

This requires no extra native execution.

### R2. Collect yielded transport subprocesses to completion, including reconciliation

`runner.whole()` correctly follows `exec_command` session IDs through `write_stdin`. Both local command wrappers in `controller.js` instead treat an undefined `exit_code` as immediate failure. A long-running checkpoint, ZIP verification or helper can legitimately yield a live session; the controller then returns an error without collecting its completion or distinguishing it from failure. This is especially hazardous when the outstanding helper is updating local transfer state.

A pure fake-tool invocation confirmed the behavior: returning `{session_id:123}` made the actual controller throw `local transfer helper failed`, and its supplied `write_stdin` was never called. No shell process or external API was invoked in this probe.

Bounded fix:
- Use one bounded whole-session collector in both controller and read-only reconciliation paths, with explicit final exit-status checks. Do not rerun the command because it yielded; resume that exact session. On host loss, leave the exact pending phase unresolved for read-only reconciliation.
- Test the collector with fake tools that yield and then succeed, yield and fail, and lose the host. These tests require no real upload/commit and no extra native event.
- Apply the same readback-space preflight used by the normal controller to `reconcileTransport` before it materializes into `backups/reconcile-readback`; currently that path bypasses the pre-transfer projection check. Account for and narrowly clean verified redundant reconciliation readback copies as well.

### R3. Freeze the accounted final-analysis execution path

The design now reserves 900 seconds in its forecast for final analysis, but the executable runner stops when the 64 world keys are complete. `recovery.receipt_path()`, `key_for()` and the supervisor only recognize historical jobs, the three qualification jobs and science world keys. There is no reserved/accounted pure-analysis job or completion barrier for the final result. Calling `analyze()` directly would bypass worker reservation/accounting even if a host keeper counted wall time.

Bounded fix:
- Add exactly one frozen, supervised pure-analysis job after all 64 exact science receipts and their persistence barriers are verified. Invoke the existing frozen `analyze()` function without changing contrasts, worlds, sample size or decision rules.
- Give it a fixed output, full-cap pre-reservation, observed completion accounting, no-overwrite/missing-only recovery, and result backup/publication barrier. Pure structural tests may verify ordering, budget exhaustion, interrupted reservation and completed-result tampering; do not fabricate or execute additional native worlds.
- Retain the separately frozen launch forecast based on the slower pilot/replay duration, 1.5× worker safety factor, two retry ceilings, measured validation/transport overhead and complete projected local storage. Reserving analysis time in prose alone is not an executable accounting gate.

## Acceptance boundary

Preserve this rejected revision, make only the bounded operational corrections, and re-audit the resulting exact closure before consuming the reserved qualification jobs. The learner, fixture, scales, endpoints and inference must remain unchanged.

After corrected static acceptance, qualification is still limited to one pure operations/analysis test job, one full 384-record × 3-branch pilot at 310200, and one independent-process exact replay of 310200, all outside the 64 official worlds. Actual automatic Library saved-version readback, isolated production recovery, GitHub tree/ref acknowledgement and unchanged-call no-op behavior must be demonstrated and independently accepted afterward. Static inspection and the old manual publication do not substitute for that evidence.

Official launch remains blocked by completed Cycle-3 acceptance, verified frozen source/runtime/publication lock, a passing resource forecast and separate parent launch approval. The separate resource-coordination hold on native work is not altered by this review. No scientific conclusion or later-round launch is authorized.
