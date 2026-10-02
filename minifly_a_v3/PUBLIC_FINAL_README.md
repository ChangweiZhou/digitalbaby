# Public final A V3 record

The completed frozen experiment has 832 native science receipts and 52 prescribed replay records. See results/REPORT.md, FINAL_METRICS.json and FINAL_AUDIT.json. No simulations or statistical changes were made during publication.

## Restore the existing frozen inputs

The 108 historical package inputs already exist in the repository's immutable input ZIP. This publication does not duplicate those historical files. Archive: `minifly_a_v3/input/MINIFLY_MUSE_A_V3_CAUSAL_GATE_20260929.zip`; existing commit `6b1345bd95d25e6c818ed85661800b1e049dd1a4`; Git blob `b832e2aee36bc4df8df1164a0c0bc678d6ca5ccd`; SHA256 `c667ab51886e226a9e7f014af634f9e9fdb1e00ad384d0eabcfd64710972ab43`.

From the repository root, run:

    python minifly_a_v3/ops/restore_frozen_inputs.py

The helper checks both archive digests, verifies all 108 member sizes/SHA256s against SOURCE_LOCK, refuses mismatched existing files and unsafe paths, then verifies the complete 146-file frozen closure. It performs no network request or experiment. The source lock is not regenerated.

results/PUBLIC_FINAL_MANIFEST.json contains native public-file hashes plus the exact archived member inventory. All 832 receipts, 52 replay records, frozen runner/tests and numerical metrics are byte-exact. Administrative projections remove private locators only, retain original digests as provenance, and are explicitly not byte-identical to private originals. Historical operational acceptance hashes can refer to those original administrative files; the projection manifest documents differences.

Install the pinned environment from SOURCE_LOCK and run the operational storage qualification in your own environment before reproducing analysis. Private-archive recovery checks in the recorded finalizer require the separately retained private bundle. Full 832-receipt source/audit results and public raw data remain available here.

The historical recovery archive metadata remains a prior 694-record checkpoint. Use the native results/science tree and this final manifest for all 832 records. The ledger's persisted_receipts cache remains829 because frozen Driver.finish did not update it; immutable terminal checkpoint evidence verifies832.
