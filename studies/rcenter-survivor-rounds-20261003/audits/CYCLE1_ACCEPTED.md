# Cycle 1 completion review

## Decision

**Cycle 1 is accepted: revised design → independent static source audit → exact two-record smoke → independent saved-receipt validation.** This closes Cycle 1 only. It does not approve Cycle 2, Cycle 3, any additional native outcome, public publication, the full pilot, or official launch.

The accepted design SHA-256 is `9d8c6b4c8aed261e0ecd21d38fa4339ec62c2fa234f3165342f5ac4adad0dccb`. The source digest is `c63c3b533b318f8da0bb316449ac6b35ba452bab2ceef4d26d3b128ff369ad75`, covering the same 122 files admitted by the static acceptance. All were rehashed after the smoke and remained unchanged.

## Evidence independently checked

- Receipt: `receipts/cycle1/smoke.json`, SHA-256 `09beebb6ad7bd26444ffcfba9427471d1962d30f0ee4d9e859f2a4854c171dc9`.
- The saved fixture identity equals the independently evaluated development seed 310000 fixture. The environment contains 384 planned records, but only its first two were presented to each of W and N_old_relation. The receipt contains exactly four branch-record rows. The two branches derive from the one prescribed newborn base plus clone.
- The independently reviewed source's signed-teacher/no-write assertions passed. The source forbids optimized execution. No claim of a separate relation-store intervention is made.
- All 32 saved store-clock rows were checked: eight stores × two branches × two records. Brain, native elapsed and FE clocks equal 165 seconds after record 1 and 330 seconds after record 2; elapsed_base=0, pending time cleared, byte counts=14/28, teach counts=1/2, and last-byte times match the 13*DT newline slot. Flush error is exactly zero in all four rows.
- Every saved alpha=1, alpha=.5, alpha=0 and private-only score was independently recomputed from raw bank values and frozen scales using standard-library arithmetic. Emitted choices and the full tie sets match the lowest-channel rule. Initial branch predictions match before any old-write intervention, and post-event state identities differ as expected while branch clocks remain identical.
- The supervisor records one completed attempt and no failed/running attempt: 6.004536886997812 seconds charged, peak RSS 283,947,008 bytes (about 270.79 MiB). The receipt's inner work time is 3.3820082879974507 seconds, within the outer charge. RSS is below 512 MiB and elapsed time below 900 seconds. Ledger totals agree with the sole attempt; they are below the total programme limits. This validates this attempt's resource receipt, not full-study feasibility.
- The recorded imported runtime is exactly Python 3.11.15 / numpy 2.2.6 / scipy 1.14.1 / numba 0.61.2. Recorded source and receipt hashes agree with independently recomputed hashes.

The executable saved-evidence checker is `audits/CYCLE1_RECEIPT_CHECK.py`; its result is `audits/CYCLE1_RECEIPT_CHECK.json`. It does not import or run any learner/vendor code. The mutable supervisor ledger at acceptance is preserved byte-for-byte as `audits/CYCLE1_LEDGER_SNAPSHOT.json`; the completion JSON binds this immutable cycle snapshot so that later legitimate ledger growth does not rewrite Cycle 1 evidence.

## Limits retained

This is a minimal constructor/teaching/clock/readout smoke. The `384_records` test label refers to pure fixture generation, not a complete native life. It does not exercise the N_new branch, either 86,400-second delay, held-out probes, long-horizon retention/acquisition, output-intervention state equivalence, anti-leakage adversaries, interruption recovery, off-host backups, publication receipt readback, or analysis/inference implementation. Signed coefficients and no-write magnitudes are checked by the frozen test assertions; their detailed arrays are not retained in this minimal receipt, so later tests need stronger inspectable trace evidence.

No scientific accuracy, E3 success/failure, policy promotion or mechanism-localization conclusion is drawn from these two records. The final endpoint remains **taught-dependent held-out table completion**. Future stronger claims need a prespecified fresh split/control separating row-completion from cross-row reuse.

All open gates listed in `CYCLE1_SOURCE_REVIEW.md` remain open, including three distinct cycle acceptances, independent full-state/leakage tests, full fixed pilot and resource forecast, exact restart/backup/publication-path proofs, license/attribution review, and official lock/launch review. A new independent auditor should review Cycle 2.
