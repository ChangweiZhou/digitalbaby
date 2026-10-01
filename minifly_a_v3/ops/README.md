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
persistence every four accepted receipts and at stop. Persistence blocks its
resource polling while workers continue, as with its original Git push. Service
requests promptly; ACK waiting is limited to 60 seconds and each Git command to
15 seconds. A long wait may
conservatively charge completed-but-uncollected jobs or cause a deadline stop.
Do not clear such a stop without the existing authorization procedure.

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
