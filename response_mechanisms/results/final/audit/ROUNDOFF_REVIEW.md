# Supplemental review of invariant-contrast floating-point mismatch

## Finding

The original independent auditor fails reproduction of four H−FE0 mechanistic p-values. It does not fail reconstruction of an appreciably sized response or any primary behavioral test. The failure is caused by a different floating-point subtraction order in an otherwise algebraically identical decomposition:

- Locked assay: `H = v - B - F - G`
- Original independent helper: `H = v - F - G - B`

Across all 224 primary receipts, the original auditor's complete numerical scan finds exactly four fields outside its unchanged 1e−10 absolute/relative comparison tolerances. They are the H−FE0 p-values for g12, g12/g1, g12/g2 and H target projection. All remaining originally checked numerical fields pass; primary tests and all qualifications agree.

The original helper and failure evidence are retained unchanged. The separate `verify_complete_locked_order.py` differs by exactly one line, aligning only the interaction subtraction order with the locked implementation. No scientific source, receipt, metrics file, threshold, roster, dose, operational ledger or locked report was edited. A rerun of the complete numerical comparison with that operation order finds zero mismatches at the original tolerances. This is numerical reproducibility within the original comparison tolerance, not a claim of bit-for-bit identity for independent SciPy statistical calculations.

## Cause and magnitude

H subtracts one channelwise constant homeostatic vector from every cue's raw response. Inspection confirms that H and FE0 raw outputs are bitwise identical at every saved probe, and every H saved output equals its saved raw output minus saved homeostasis exactly. Thus the F, G and interaction components are algebraically invariant to H's offset; observed H−FE0 differences in these components are floating-point cancellation residuals.

For the 32-world final decision-centered learned-response contrasts:

| Quantity | Maximum absolute locked H−FE0 difference | Mean absolute arm magnitude | Locked p | Alternative-order p |
|---|---:|---:|---:|---:|
| g12 | 2.7756e−17 | 0.08408 | 0.374068 | 0.325053 |
| g12/g1 | 2.2204e−16 | 0.95160 | 0.032471 | 0.800868 |
| g12/g2 | 1.1102e−16 | 0.53216 | 0.163227 | 0.609369 |
| H projection | 1.3878e−17 | 0.04715 | 0.500005 | 0.763009 |

For g12, the locked mean contrast is −1.3010e−18 with sample SD 8.1604e−18; the alternative order yields +1.3010e−18 with sample SD 7.3598e−18. Although these absolute differences are negligible, a t statistic divides the small mean by an equally tiny estimated standard error. Consequently its p-value need not be close between operation orders. A tolerance on the p-value cannot characterize numerical agreement of an algebraically zero effect.

The g12/g1 locked p≈0.03247 and nominal interval below zero are particularly important to disclose: they are numerical artifacts, not evidence for an H effect. Aligning the auditor reproduces those saved numbers but does not validate their substantive interpretation. The same caution applies to all H−FE0 invariant-component p-values, including ones that did not mismatch.

## Scientific impact and honest disposition

There is no change to any primary behavioral estimate, primary adjusted p-value, reported bias reduction, or prespecified qualification decision. The invariant-contrast p-values are not used in H qualification; H qualification requires behavioral benefit and bias reduction. The stored qualification checks agree under both decomposition orders.

Recommended disposition: preserve the locked outputs and original failed audit; report that the arithmetic-order-aligned reproduction check passes at its original tolerance; accompany it with this explicit numerical limitation. Treat H's F/G/interaction preservation as an algebraic implementation check, as the locked report already instructs. Do not interpret Student-t significance for contrasts composed solely of floating-point residuals. Do not change thresholds, zero saved results, or regenerate scientific outputs post hoc.

This review covers the helper's complete `numerical()` checks and an additional H/FE0 raw-output/offset check. It does not establish the separate operational audit, scientific replay, publication state, or continuous resource-limit compliance.

## Evidence

- `ROUNDOFF_REVIEW.json`: all four original mismatch records; corrected-order full numerical comparison; both qualification outputs; all 32-world invariant contrasts; raw-output invariance checks
- `ROUNDOFF_HELPER_DIFF.patch`: exact one-line helper-only change
- `LOCKED_ORDER_NUMERICAL_PASS.json`: separate corrected helper's complete numerical check with its original assertions
- Original helper SHA-256: `035df3473c4eba2843fe404b873d6e5f5cca61a0e8b07819d26a9b63cfa98fd5`
- Corrected helper SHA-256: `ecb392e0332f4a833840f42c54dc4b1d826ce9081aa6da2e466f3240e20f84e9`
- Scientific lock SHA-256: `24943155d4b010c8e17746a265803155e13585a7b0b23fe261f3c0bae95a63fe`
- Locked final metrics SHA-256: `8d610a442f611717cfa8efad895a921b60a6fdbd23f93e8b3c42fc3f6c9f3a0c`
