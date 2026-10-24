# BYTE-CORE V9.2 DEV result summary

Input JSON SHA-256: `b7272a2d7beb9f576cad4f26f0c0a65e97b2c939ebf49628de9c1e3f4e9e1f69`. Controlled cases saved: 4/4; all configured controlled cases saved.
Scores below are copied from the saved experiment result; missing values are shown as —.

**Evidence scope:** DEV is an engineering diagnostic, even if every configured case finishes. It is not a locked confirmation or a capability claim.

Saved acceptance status: `development_diagnostic`. Saved all-world primary pass: `True`.

## Controlled learning curves

BPB is lower-is-better; accuracy is higher-is-better. Slot Markov1/2 are the experiment's target-slot-conditioned `oracle_last1/2` baselines.

| task | world | curve | train bytes | brain BPB | brain acc | no-write BPB | no-write acc | slot M1 BPB | slot M1 acc | slot M2 BPB | slot M2 acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ID | 11 | 10k | 10000 | 6.6810 | 0.0039 | 8.0000 | 0.0039 | 0.8179 | 1.0000 | 0.3004 | 1.0000 |
| ID | 11 | 100k | 100000 | 0.3422 | 0.9824 | 8.0000 | 0.0039 | 0.1063 | 1.0000 | 0.0073 | 1.0000 |
| ID | 11 | full | 196608 | 0.1243 | 0.9976 | 8.0000 | 0.0039 | 0.0551 | 1.0000 | 0.0020 | 1.0000 |
| ID | 12 | 10k | 10000 | 6.6805 | 0.0039 | 8.0000 | 0.0039 | 0.8179 | 1.0000 | 0.3004 | 1.0000 |
| ID | 12 | 100k | 100000 | 0.3405 | 0.9795 | 8.0000 | 0.0039 | 0.1063 | 1.0000 | 0.0073 | 1.0000 |
| ID | 12 | full | 196608 | 0.1247 | 0.9961 | 8.0000 | 0.0039 | 0.0551 | 1.0000 | 0.0020 | 1.0000 |
| ORDER | 11 | 10k | 10000 | 3.7725 | 0.8486 | 8.0000 | 0.0000 | 4.0015 | 0.0625 | 0.9288 | 1.0000 |
| ORDER | 11 | 100k | 100000 | 0.0890 | 1.0000 | 8.0000 | 0.0000 | 4.0000 | 0.0625 | 0.1314 | 1.0000 |
| ORDER | 11 | full | 262144 | 0.0310 | 1.0000 | 8.0000 | 0.0000 | 4.0000 | 0.0625 | 0.0518 | 1.0000 |
| ORDER | 12 | 10k | 10000 | 3.7663 | 0.8506 | 8.0000 | 0.0000 | 4.0015 | 0.0625 | 0.9288 | 1.0000 |
| ORDER | 12 | 100k | 100000 | 0.0879 | 1.0000 | 8.0000 | 0.0000 | 4.0000 | 0.0625 | 0.1314 | 1.0000 |
| ORDER | 12 | full | 262144 | 0.0305 | 1.0000 | 8.0000 | 0.0000 | 4.0000 | 0.0625 | 0.0518 | 1.0000 |

The controlled CSV also includes full-stream BPB for the brain, no-write control, and all-stream Markov2 baseline.

## Route arms and copied-checkpoint lesions

V9.2 predicts only through engineered temporal cells: `temporal_only` is the intact predictor (native readout removed by construction), while `native_only` removes the temporal cells and leaves the learned output bias. These are frozen evaluation lesions, not separate training histories.

| task | world | curve | arm | lesion | target BPB | target acc |
| --- | --- | --- | --- | --- | --- | --- |
| ID | 11 | 10k | full151_ordered | none | 6.6810 | 0.0039 |
| ID | 11 | 10k | full151_ordered | native_only | 8.0228 | 0.0039 |
| ID | 11 | 10k | full151_ordered | temporal_only | 6.6810 | 0.0039 |
| ID | 11 | 10k | full151_ordered_no_predictive_write | none | 8.0000 | 0.0039 |
| ID | 11 | 100k | full151_ordered | none | 0.3422 | 0.9824 |
| ID | 11 | 100k | full151_ordered | native_only | 8.0254 | 0.0039 |
| ID | 11 | 100k | full151_ordered | temporal_only | 0.3422 | 0.9824 |
| ID | 11 | 100k | full151_ordered_no_predictive_write | none | 8.0000 | 0.0039 |
| ID | 11 | full | full151_ordered | none | 0.1243 | 0.9976 |
| ID | 11 | full | full151_ordered | native_only | 8.0310 | 0.0039 |
| ID | 11 | full | full151_ordered | temporal_only | 0.1243 | 0.9976 |
| ID | 11 | full | full151_ordered_no_predictive_write | none | 8.0000 | 0.0039 |
| ID | 12 | 10k | full151_ordered | none | 6.6805 | 0.0039 |
| ID | 12 | 10k | full151_ordered | native_only | 8.0229 | 0.0039 |
| ID | 12 | 10k | full151_ordered | temporal_only | 6.6805 | 0.0039 |
| ID | 12 | 10k | full151_ordered_no_predictive_write | none | 8.0000 | 0.0039 |
| ID | 12 | 100k | full151_ordered | none | 0.3405 | 0.9795 |
| ID | 12 | 100k | full151_ordered | native_only | 8.0253 | 0.0039 |
| ID | 12 | 100k | full151_ordered | temporal_only | 0.3405 | 0.9795 |
| ID | 12 | 100k | full151_ordered_no_predictive_write | none | 8.0000 | 0.0039 |
| ID | 12 | full | full151_ordered | none | 0.1247 | 0.9961 |
| ID | 12 | full | full151_ordered | native_only | 8.0309 | 0.0039 |
| ID | 12 | full | full151_ordered | temporal_only | 0.1247 | 0.9961 |
| ID | 12 | full | full151_ordered_no_predictive_write | none | 8.0000 | 0.0039 |
| ORDER | 11 | 10k | full151_ordered | none | 3.7725 | 0.8486 |
| ORDER | 11 | 10k | full151_ordered | native_only | 7.8407 | 0.0000 |
| ORDER | 11 | 10k | full151_ordered | temporal_only | 3.7725 | 0.8486 |
| ORDER | 11 | 10k | full151_ordered_no_predictive_write | none | 8.0000 | 0.0000 |
| ORDER | 11 | 100k | full151_ordered | none | 0.0890 | 1.0000 |
| ORDER | 11 | 100k | full151_ordered | native_only | 7.6210 | 0.0000 |
| ORDER | 11 | 100k | full151_ordered | temporal_only | 0.0890 | 1.0000 |
| ORDER | 11 | 100k | full151_ordered_no_predictive_write | none | 8.0000 | 0.0000 |
| ORDER | 11 | full | full151_ordered | none | 0.0310 | 1.0000 |
| ORDER | 11 | full | full151_ordered | native_only | 7.6054 | 0.0000 |
| ORDER | 11 | full | full151_ordered | temporal_only | 0.0310 | 1.0000 |
| ORDER | 11 | full | full151_ordered_no_predictive_write | none | 8.0000 | 0.0000 |
| ORDER | 12 | 10k | full151_ordered | none | 3.7663 | 0.8506 |
| ORDER | 12 | 10k | full151_ordered | native_only | 7.8409 | 0.0000 |
| ORDER | 12 | 10k | full151_ordered | temporal_only | 3.7663 | 0.8506 |
| ORDER | 12 | 10k | full151_ordered_no_predictive_write | none | 8.0000 | 0.0000 |
| ORDER | 12 | 100k | full151_ordered | none | 0.0879 | 1.0000 |
| ORDER | 12 | 100k | full151_ordered | native_only | 7.6210 | 0.0000 |
| ORDER | 12 | 100k | full151_ordered | temporal_only | 0.0879 | 1.0000 |
| ORDER | 12 | 100k | full151_ordered_no_predictive_write | none | 8.0000 | 0.0000 |
| ORDER | 12 | full | full151_ordered | none | 0.0305 | 1.0000 |
| ORDER | 12 | full | full151_ordered | native_only | 7.6051 | 0.0000 |
| ORDER | 12 | full | full151_ordered | temporal_only | 0.0305 | 1.0000 |
| ORDER | 12 | full | full151_ordered_no_predictive_write | none | 8.0000 | 0.0000 |

## Ordered-pair swap counterfactual

These metrics exclude the swap-invariant ordered pairs; positive BPB penalty means the swap worsened prediction.

| world | curve | noninvariant n | normal BPB | swap BPB | swap penalty | accuracy drop |
| --- | --- | --- | --- | --- | --- | --- |
| 11 | 10k | 1792 | 3.7691 | 7.5639 | 3.7948 | 0.8482 |
| 11 | 100k | 1792 | 0.0891 | 11.2966 | 11.2075 | 1.0000 |
| 11 | full | 1792 | 0.0310 | 13.0388 | 13.0077 | 1.0000 |
| 12 | 10k | 1792 | 3.7754 | 7.5587 | 3.7833 | 0.8499 |
| 12 | 100k | 1792 | 0.0880 | 11.2913 | 11.2033 | 1.0000 |
| 12 | full | 1792 | 0.0305 | 13.0385 | 13.0079 | 1.0000 |

## Source-separated natural text

Each author is scored separately. Full-stream BPB covers every raw byte; content BPB uses the scorer's Unicode letter/number byte mask. A missing 8-gram value means that baseline was absent from the saved result.

| author | curve | arm | brain full BPB | brain content BPB | M2 full BPB | M2 content BPB | 8gram full BPB | 8gram content BPB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Lewis Carroll | 10k | full151_ordered | 4.5127 | 4.7782 | 3.5218 | 3.5582 | 3.4759 | 3.5349 |
| Lewis Carroll | 10k | full151_ordered_no_predictive_write | 8.0000 | 8.0000 | 3.5218 | 3.5582 | 3.4759 | 3.5349 |
| Lewis Carroll | 100k | full151_ordered | 2.9843 | 3.1129 | 3.0713 | 3.2095 | 2.8805 | 2.9682 |
| Lewis Carroll | 100k | full151_ordered_no_predictive_write | 8.0000 | 8.0000 | 3.0713 | 3.2095 | 2.8805 | 2.9682 |
| Lewis Carroll | full | full151_ordered | 2.8429 | 2.9719 | 3.0153 | 3.1604 | 2.7610 | 2.8329 |
| Lewis Carroll | full | full151_ordered_no_predictive_write | 8.0000 | 8.0000 | 3.0153 | 3.1604 | 2.7610 | 2.8329 |

Machine-readable tables: `controlled_curve.csv`, `route_arms_lesions.csv`, `order_swap.csv`, `natural_text.csv`. Static figure: `FIGURE.svg`.

## Locked text decision

Status: `development_diagnostic`. Two-author pass: `None`. The rule compares each author's full and content-byte BPB with train-only Markov-2 and matched no-write.
