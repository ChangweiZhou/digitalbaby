# Final independent audit and interpretation

Completed 2026-10-02 UTC. The complete 32-world, seven-arm experiment contains
224 primary receipts and seven preselected fresh-process replay receipts. The
prespecified beneficial behavioral criteria were not met by T, H, or J. This is
a negative result for these implementations in this assay, not a general
rejection of the mechanisms or proof of equivalence.

## Verification

- The exact scientific lock remains
  `24943155d4b010c8e17746a265803155e13585a7b0b23fe261f3c0bae95a63fe`.
- All 39 tests passed again in 141.50 seconds. All 224 primary receipts passed
  the locked full-roster reconstruction audit; all seven retained replay
  contents matched their primary receipts after excluding resource fields and
  had distinct saved process IDs.
- Independent calculations checked every saved tensor reduction, all 32 paired
  world contrasts, Student-t intervals/tests, four-test Holm correction,
  undefined ratios, the T preservation floors, H bias criterion, and J controls.
- The complete combined numerical/operational audit passed after the explicitly
  documented one-line auditor arithmetic-order correction below. The original
  auditor failure and both auditor versions are preserved under `audit/`.
- The final simulation coordinator returned exit code 0, with no active jobs;
  the inherited exclusive supervisor lock was independently acquired. The
  corrected combined audit also returned exit code 0.

See the unchanged locked `REPORT.md` and `FINAL_METRICS.json` for all numerical
results. Primary paired accuracy differences in percentage points were:

| Comparison | Mean difference | 95% interval | Holm p |
|---|---:|---:|---:|
| T minus T_OFF | -8.398 | [-11.513, -5.284] | 0.00002049 |
| H minus FE0 | 2.930 | [-2.284, 8.144] | 0.7817 |
| J minus J_ADD | 0.195 | [-0.502, 0.893] | 1.0 |
| J minus J_SHUFFLE | 0.391 | [-1.004, 1.785] | 1.0 |

Timing plasticity reduced final retention accuracy in this locked configuration.
Homeostasis reduced bias, but its accuracy benefit was not established. J had
small positive supportive interaction changes without a demonstrated behavioral
advantage. Mechanistic component tests remain nominal/supportive, not jointly
familywise-confirmed mechanism evidence.

## Transparent numerical-roundoff disposition

The original independent helper evaluated the interaction residual as
`v - F - G - B`; the locked assay uses `v - B - F - G`. These expressions are
algebraically identical but can differ at machine precision. The first complete
audit stopped on the H-minus-FE0 interaction-RMS p-value. A separate review then
found exactly four comparison mismatches, all p-values for theoretically
invariant H-minus-FE0 interaction quantities. No primary endpoint, meaningful
bias effect, or qualification changed under either arithmetic order.

For example, the largest H-minus-FE0 interaction-RMS difference was
2.78e-17 against an interaction magnitude of approximately 0.084. The largest
interaction/main-effect ratio difference was 2.22e-16. A nominal p=0.03247 for
that invariant ratio changed to 0.80087 with the algebraically equivalent
operation order. Such invariant-contrast p-values are numerical artifacts and
must not be interpreted as evidence of a scientific effect. The raw H and FE0
probe values were bitwise identical, and H outputs equaled raw values minus the
recorded homeostatic offset exactly.

The separate reproduction auditor changes only that subtraction order. It
passes all comparisons with the original tolerances. This is an auditor
reproduction correction, not a change to scientific code, receipts, endpoints,
thresholds, statistical plan, `FINAL_METRICS.json`, or the locked renderer's
`REPORT.md`. No p-values were replaced or silently removed.

- Original auditor SHA256:
  `035df3473c4eba2843fe404b873d6e5f5cca61a0e8b07819d26a9b63cfa98fd5`
- Order-aligned auditor SHA256:
  `ecb392e0332f4a833840f42c54dc4b1d826ce9081aa6da2e466f3240e20f84e9`
- Full discrepancy, magnitudes, both computations, and the one-line diff are
  retained in `audit/ROUNDOFF_REVIEW.json` and `audit/ROUNDOFF_HELPER_DIFF.patch`.

## Operational reconciliation and recovery limitations

The final sample is seven preserved original primary receipts plus 217
recomputed receipts from the same fixed roster. The 202 primary lives observed
before workspace loss are historical progress, not extra or recovered data.
The original seven replays were retained. No completed retained receipt was
overwritten and there was no outcome-based selection or additional world.

The documented, explicitly approved operational amendments increased the
cumulative worker cap from 8 to 12 hours and concurrency from one to two. The
original source lock deliberately retains its original limits; compliance with
those superseded limits is not claimed. The original elapsed deadline was
2026-10-02 04:40:04.660057 UTC; completion and this audit preceded it.

- Total conservative ledger charge: 36,539.593530 seconds (10.149887 hours)
- Prior-loss conservative baseline: 17,245.339943 seconds
- Primary receipt-only measured runtime: 15,359.798782 seconds (4.266611 hours)
- Replay receipt-only runtime: 535.371136 seconds
- Successful ledger jobs: exactly 231, without duplicate successful identities
- Maximum observed completed-job runtime: 134.296931 seconds, below 300 seconds
- Maximum saved receipt RSS: 303,550,464 bytes, below 800,000,000 bytes
- Retained parallel reservation intervals: maximum concurrency two
- Result files before audit output: approximately 32.2 MB, below 200 MB

Interrupted attempts were charged conservatively through recovery observation,
including unobserved downtime. Such charges are not measured life runtimes.
Lost pre-reset job durations cannot be reconstructed. Historical single-worker
records lack exact starts, so universal historical concurrency is not
independently established. Userspace RSS polling cannot prove continuous
instantaneous bounds between samples. These limitations are retained in
`OPERATIONAL_AUDIT.json` and the combined audit result.

## Preservation and publication

All 224 primary and seven replay receipts were saved in the private recovery
backup before final auditing. The prior public checkpoint was independently
verified on 2026-10-02 at 01:11 UTC as commit
`48595f1e02cc1bca69ba7304ab051d2c8f40d43f`, containing 42 primary receipts and
seven replays; all 49 remote blob hashes matched local bytes. The final
publication must contain the complete roster and this audit disposition.
Publication is established by a separately verified final remote commit, not
by the existence of a local queue or this report.

This experiment tests taught random pairs and retention under interference. It
does not test withheld-pair relational transfer or general reasoning.
