# Response-mechanism results

Status: EXPLORATORY PILOT ONLY. 1 worlds; all seven arms; 4 repetitions per cohort.

## Primary behavioral endpoint: final old-pair accuracy

| Arm | Accuracy %, mean [95% interval] | W−N_old percentage points |
|---|---:|---:|
| FE0 | 25.000 (one pilot world) | 0.000 (one pilot world) |
| T_OFF | 25.000 (one pilot world) | 0.000 (one pilot world) |
| T | 25.000 (one pilot world) | 0.000 (one pilot world) |
| H | 31.250 (one pilot world) | 6.250 (one pilot world) |
| J_ADD | 25.000 (one pilot world) | 0.000 (one pilot world) |
| J | 25.000 (one pilot world) | 0.000 (one pilot world) |
| J_SHUFFLE | 25.000 (one pilot world) | 0.000 (one pilot world) |

## Prespecified paired primary contrasts

| Comparison | Difference, percentage points | Holm p |
|---|---:|---:|
| T − T_OFF | 0.000 (one pilot world) | None |
| H − FE0 | 6.250 (one pilot world) | None |
| J − J_ADD | 0.000 (one pilot world) | None |
| J − J_SHUFFLE | 0.000 (one pilot world) | None |

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

All 7 receipts passed receipt-only reconstruction (not independent simulation replay). Total measured worker time 0.107 h; maximum RSS 221.6 MB.

Paired worlds; two-sided Student-t intervals and tests; Holm correction across four primary behavioral contrasts; mechanistic signature gates use nominal95% supportive components and are NOT familywise-confirmed mechanistic claims
