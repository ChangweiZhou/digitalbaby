# Cycle 3 operations attempt 1: independent failure classification

## Decision

**Confirmed infrastructure/test-instrumentation failure; an exact-key retry is eligible only after the bounded repair, source-transition audit and separate recovery approval. No retry is authorized by this report.** The failed qualification is not accepted. No pilot, replay or official world may be inferred to have run or passed.

The reviewed failed source is the accepted 130-file closure with digest `07786c8d46d9f381b42d15c148102e0fda79e88074aa4bfd0b99436559cde1b0`, design SHA `6e61e0c78f1106a1bf591ede8fc98c1ee35a3b9469283d3e90892951e0868de0`.

## Independently established evidence

- The actual `cycle3/operations` attempt 1 exited with code 1 at the synthetic streaming subprocess's `ru_maxrss < 128 MiB` assertion, before writing the qualification receipt. The preserved log identifies this exact assertion; it does not identify a failed scientific quantity.
- The immutable WAL and anchored ledger reconcile. The observed failed attempt is charged **6.423622839996824 seconds**, with recorded peak **169,107,456 bytes**. Cumulative programme worker time is **58.47810131699225 seconds**. Do not erase or replace that attempt or refund it again.
- The verified private saved-version-3 archive matches its recorded SHA-256 and contains this attempt's exact pre-job reservation, with the full **900-second charge**. The saved pre-job cumulative total is 952.0544784769954 seconds. That is the correct conservative restore position if the later observed-finish evidence is lost. The prebirth backup succeeded; this is not evidence of a completed operations result or GitHub publication.
- An independent pure reproduction used the actual checkpoint writer on a synthetic 32 MiB payload under a padded parent. The fresh executed child reported `ru_maxrss=188194816` bytes, identical to the parent, while `/proc/self/status` reported its own `VmHWM=21368832` and `VmRSS=18571264` bytes. Thus the old 128 MiB assertion can fail solely because the resource counter retains fork/exec history. The actual fresh-process high-water measurement in this reproduction was below the unchanged 128 MiB test bound.
- The host-coordinator lock was independently acquired nonblocking and released, confirming it was free. Nevertheless, the preserved host state still says active and its event chain has only `host_started`. Therefore cleanup did not establish a journalled graceful close. The next startup must conservatively reconcile that stale active session; do not manually mark it closed or lower its charge.
- No native learner module was imported in this independent reproduction. This classification concerns pure test instrumentation and host lifecycle only. The actual supervised entry is the pure operations test; no native pilot/replay has been launched.

Private identity, reservation, exact ledger/host snapshots and input hashes are preserved separately in `CYCLE3_OPERATIONS_FAILURE_PRIVATE.json`, which is excluded from the public projection by its PRIVATE name. The full original failure log remains at its existing attempt path.

## Permitted bounded repairs for re-audit

### 1. Correct the streaming test's process-memory instrument

Record both raw `resource.ru_maxrss` and the fresh executed child's `/proc/self/status` VmHWM/RSS. Apply the existing 128 MiB synthetic streaming criterion to that child's own VmHWM, with explicit Linux units, required fields and failure on missing/invalid measurements. Save the measured values before an assertion can discard them. Add a pure regression that reproduces a large-parent/small-fresh-child case and verifies the correct metric is selected.

Do not raise the 128 MiB streaming test threshold, relax the 768 MiB supervised job cap, modify a learner operation, or change scientific fixtures/parameters. The instrumentation evidence supports a measurement correction, not performance tuning or scientific outcome rejection/replacement.

### 2. Make host closure token-bound and verified

Replace reliance on PTY Ctrl-C for ordinary keeper cleanup with a stop request bound to the exact current host token, followed by collection of that keeper's exact tool session until terminal. The keeper must journal the graceful close and atomically persist inactive/final wall accounting. Verify the matching token, terminal status, final state and released lock; merely sending a request or interrupt is not completion.

Test normal close, stale/wrong-token rejection and keeper disappearance using disposable processes. Host disappearance remains a conservative unknown session; never fabricate a graceful close. Preserve the original job failure if cleanup also fails. Unknown accounting remains charged by the production reconciliation rules.

### 3. Introduce an explicit, narrowly bound source transition

The repair changes the executable closure. Current `validate_attempts()` correctly rejects a modern attempt whose source digest differs from current source, so a successful repair cannot be admitted by overwriting the failed ledger entry or weakening that check globally.

Preserve the exact failed-source snapshot, its full map/design/static review, original attempt/ledger/WAL/log and pre-job backup evidence. A separately audited transition may admit this one historical **failed** `cycle3/operations` attempt 1 under its old source while requiring the new attempt to use the newly accepted source. Bind the transition to:
- old and new full source digests and preserved snapshot hashes;
- this exact job, attempt count and reservation identity;
- original failed-attempt content, failure log and ledger/WAL evidence;
- the instrumentation/lifecycle/recovery-validation changed-file set;
- unchanged scientific learner/vendor, fixture, scales, endpoints, inference and official sample.

The historical failed attempt must retain its old digest, status and observed charge. The new retry has attempt number 2, a fresh reservation and the new accepted digest. Retry counts and cumulative worker time span the transition. The exception must not validate arbitrary mismatched modern attempts, completed native receipts, another job or a new world.

Add pure tests rejecting altered old snapshots/logs, an unapproved new digest, a changed old attempt/charge and attempts to reuse the transition for a different key. Verify that all existing missing-only/barrier/retry-count rules still apply. This is an operational amendment, not a reset of the qualification sequence or its budget.

## Evidence required before exact-key retry

1. Preserve the failure/snapshot evidence above and document the limited design amendment.
2. Independently accept the corrected full source closure and the specific old→new transition, with pure repair/lifecycle/transition checks passing. The prior static acceptance must not be silently relocked.
3. Issue a separate independent recovery approval for disposition `retry`, classification `infrastructure`, exact failed key/attempt/reservation and both failed and newly accepted retry source identities. The current old-source-only approval schema needs to bind the authorized target revision as part of the reviewed correction.
4. Reconcile the stale host session conservatively. Perform fresh memory, wall, worker, output and disk admission.
5. Reserve and privately read back the new attempt's full 900 seconds before execution. Preserve attempt 1 unchanged. At most the originally allowed two infrastructure retries remain available; this decision does not expand that limit.

After that approval, rerun only the same pure operations job. A successful retry still requires independent saved-receipt review. Pilot/replay remain pending and can only follow the registered qualification gates. No numerical/scientific failure may be retried under this infrastructure classification.
