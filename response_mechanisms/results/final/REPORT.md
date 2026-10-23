# Response-mechanism results

Status: PROSPECTIVE FRESH-WORLD FINAL RUN. 32 worlds; all seven arms; 6 repetitions per cohort.

## Primary behavioral endpoint: final old-pair accuracy

| Arm | Accuracy %, mean [95% interval] | W−N_old percentage points |
|---|---:|---:|
| FE0 | 39.453 [35.951, 42.955] | 13.672 [9.730, 17.614] |
| T_OFF | 39.453 [35.951, 42.955] | 13.672 [9.730, 17.614] |
| T | 31.055 [28.953, 33.156] | 6.055 [3.953, 8.156] |
| H | 42.383 [38.266, 46.500] | 17.383 [13.148, 21.617] |
| J_ADD | 39.844 [36.058, 43.630] | 14.258 [10.627, 17.888] |
| J | 40.039 [36.292, 43.786] | 14.258 [10.673, 17.843] |
| J_SHUFFLE | 39.648 [36.092, 43.205] | 14.062 [10.097, 18.028] |

## Prespecified paired primary contrasts

| Comparison | Difference, percentage points | Holm p |
|---|---:|---:|
| T − T_OFF | -8.398 [-11.513, -5.284] | 2.0494162824707616e-05 |
| H − FE0 | 2.930 [-2.284, 8.144] | 0.7816964901133575 |
| J − J_ADD | 0.195 [-0.502, 0.893] | 1.0 |
| J − J_SHUFFLE | 0.391 [-1.004, 1.785] | 1.0 |

## Mechanistic qualification

- T: does not meet prespecified supportive signature
- H: does not meet prespecified supportive signature
- J: does not meet prespecified supportive signature

Mechanistic signature checks are supportive and do not have a joint familywise confirmation guarantee. Full component magnitudes, target-aligned projections of decision-centered W−N_old response, denominator flags, state budgets, exact fixtures, doses and all world-level contrasts are in FINAL_METRICS.json and immutable receipts.

## Interpretation limits

- H preserving F/G/H is an algebraic implementation check; behavioral benefit and bias reduction are independently required
- J comparisons control allocation and learning equation, not exact realized update dose; the bank has extra separately writable synapses relative to FE0
- Birth tensors and N_old controls distinguish preexisting activity from old-learning-induced response; interaction RMS alone is insufficient
- Offline grand-mean centering is reported as a diagnostic, not a deployed or label-tuned learner
- Random mappings are globally balanced but have variable row and column effects, all reported
- This assay tests taught random pairs and retention under interference, not withheld-pair transfer or general reasoning
- A failure to qualify is not proof of no possible benefit; intervals and all negative outcomes are retained

## Verification and resources

All 224 receipts passed receipt-only reconstruction (not independent simulation replay). Total measured worker time 4.267 h; maximum RSS 303.6 MB.

Paired worlds; two-sided Student-t intervals and tests; Holm correction across four primary behavioral contrasts; mechanistic signature gates use nominal95% supportive components and are NOT familywise-confirmed mechanistic claims
