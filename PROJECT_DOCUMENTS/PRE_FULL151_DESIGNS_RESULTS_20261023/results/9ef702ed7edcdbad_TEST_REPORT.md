# Engineering test report — Round-B amendment v1.2

All tests below used the exact amended source shipped in this package. Fixture runs are software tests only, not MiniFly efficacy evidence.

## Static/unit contract

- pytest: **15/15 passed**.
- Python compile: passed.
- built-in `--self-test`: passed.
- shell syntax for `RUN_ROUND_B_AMENDED.command`: passed.

The suite checks candidate IDs, exact trace delta, gene boxes, prior cue exclusion bundle, world/reactivation semantics, activity override outside the genome, fixed structure, primary denominators, clone-only diagnostics, absence of collateral/churn from Round-A primaries, Round-A hash tamper rejection, incompatible Round-A version rejection, and the amended Round-B anchor contract.

## Actual completed Round-A archive compatibility

The engineering Round-B runs below used the uploaded completed native V88 Round-A archive, including its exact `DECISION.json` with hash:

`897e8f969773ab426fb050a8f2cc6e8e77fbd459ff60d5770779b849a111553e`

The amended validator accepted it and selected `RCE_TRACE_RESTORED`.

## Serial / parallel determinism

Three-history fixture Round B:

- 1 worker: complete, 27/27 result receipts.
- 2 workers: complete, 27/27 result receipts.
- byte-identical: `NORMAL_RESULTS.jsonl.gz`, factorial cell summary, factorial paired contrasts, contextual anchor summary, contextual anchor contrasts, all-arm summary, and `RESULT.json`.

The arm manifest contained exactly:

`Fnative_Anative, Fnative_A3, Fnative_A2p5, Fref_Anative, Fref_A3, Fref_A2p5, REF, V84E, RCE_FULL`.

`FACTORIAL_CELL_SUMMARY.csv` contained only the six causal cells. `CONTEXTUAL_ANCHOR_SUMMARY.csv` contained only REF/V84E/RCE_FULL.

## Receipt-level resume

One completed receipt was deleted from the serial run and the same Round B was rerun with two workers. The missing job was reconstructed; all final analysis files listed above were byte-identical to the uninterrupted serial run.

## Production-size control-flow fixture

A full **24-history × 9-arm = 216 normal-result** fixture Round B completed. The execution harness interrupted the first invocation after 130 receipts; rerunning the same command resumed from sealed receipts and completed all 216. `STATUS.json` is `COMPLETE` and the final arm is RCE_FULL.

This is deliberate evidence that production-size receipt resumption works; it is not biological evidence.
