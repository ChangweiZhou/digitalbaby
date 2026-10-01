# A V3 recovery operations

This directory is additive operational tooling. It is **outside** `SOURCE_LOCK.json`
and was not covered by the historical launch qualification. The learner, frozen
driver, auditors, statistical analysis, roster, resource limits, and source lock
remain unchanged. Do not add these files to the locked `src/` or `tests/` closure.

## Connector persistence

The recovery host can clone/fetch the repository, but cannot authenticate a shell
Git push. `connector_persistence.py` supplies only an `Env.persist` transport for
the frozen `Driver`. It stages the science results and emits an exact Git-object
snapshot to `scratch/connector/`. An authorized controller must:

1. Verify the repository/branch and current remote parent match the request.
2. Read the requested blobs from the Git object database, not mutable working files.
3. Create missing blobs through the GitHub connector and check each returned SHA.
4. Create the tree from the expected parent tree; require the exact requested SHA.
5. Create a single-parent commit and perform a **non-force** branch update.
6. Verify the remote commit and write the matching ACK (or a negative ACK on failure).

The adapter then fetches the remote, checks commit, parent, and exact staged tree,
advances the local branch with compare-and-swap, and rechecks remote equality. It
refuses replacement of any existing `.json.gz` receipt. Failed or missing ACKs
return to the frozen driver's infrastructure-stop path. Never fabricate an ACK.
An uncertain publication must be inspected before retrying. No merge, force push,
credential extraction, or source-lock modification is involved.

The driver still validates existing/new receipts, starts only missing world-arms,
uses the original four-worker budget and yoked dependencies, and requests
persistence every four accepted receipts and at stop. Real connector publications
took about 150 seconds and 230 seconds (the latter for eight blobs). ACK waiting
is bounded at 600 seconds to accommodate larger receipt batches; each Git command
is bounded at 10 seconds. These are transport bounds, not enlarged science budgets.
While the driver is inside persistence, a separate
operational watchdog in this adapter checks worker RSS/deadlines, disk, cumulative
wall time, and a conservative worker-time bound. It polls every second during ACK
waiting and before/after every Git command, including local object/index checks
and commands that fail. A breach kills affected work and records the
frozen driver's existing budget failure category before raising. The next driver
poll accounts the wait; the watchdog does not double-charge its counters. A long
wait can still conservatively charge completed-but-uncollected jobs. Do not clear
a budget stop without the existing authorization procedure.

The adapter records each value returned by the original `Env.clock` and returns
it unchanged. Persistence watchdog elapsed time starts at the last driver poll,
so receipt-validation time before persistence is not omitted. The initial worker
snapshot is retained for conservative cumulative accounting. The supplemental
`/proc` process-age guard uses `CLOCK_BOOTTIME` where available, including any
suspension time; this may stop a worker earlier, never extend its allowance.
The frozen Driver's clock, counters, deadlines, and budget values remain unchanged.

Run the adapter only with the exact Python/library versions in `SOURCE_LOCK.json`
and a live authorized controller. Its focused local-remote tests are in
`test_connector_persistence.py`. These transport tests supplement, not replace,
the frozen launch tests and receipt audits.

## Final verification still required

- Full 832-receipt validation under the unchanged source lock
- Literal causal-branch gate over every science receipt
- Independent frozen replay for worlds 190001–190004, all 13 arms
- Final frozen statistical analysis with m=26 and the declared unavailable P3 slots
- Truthful report including null/adverse results, diagnostics, resources, and the
  invalidity of Package B V2 E3 for combined claims

Do not run the final analyzer alongside science workers: it loads all receipts
into memory. Do not replace its estimand, confidence method, controls, or roster.
