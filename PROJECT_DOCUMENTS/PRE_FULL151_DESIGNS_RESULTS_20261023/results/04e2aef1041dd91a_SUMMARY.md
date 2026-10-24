# BYTE-CORE V9.2 CONFIRM result summary

Input JSON SHA-256: `159b35aa2f53674ae4929170b089d60646f33433ec2ec433069cac22c3aaadba`. Controlled cases saved: 6/6; all configured controlled cases saved.
Scores below are copied from the saved experiment result; missing values are shown as —.

Saved acceptance status: `confirmation`. Saved all-world primary pass: `True`.

## Controlled learning curves

BPB is lower-is-better; accuracy is higher-is-better. Slot Markov1/2 are the experiment's target-slot-conditioned `oracle_last1/2` baselines.

| task | world | curve | train bytes | brain BPB | brain acc | no-write BPB | no-write acc | slot M1 BPB | slot M1 acc | slot M2 BPB | slot M2 acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ID | 101 | 10k | 10000 | 6.7029 | 0.0056 | 8.0000 | 0.0039 | 0.8179 | 1.0000 | 0.3004 | 1.0000 |
| ID | 101 | 100k | 100000 | 0.3398 | 0.9814 | 8.0000 | 0.0039 | 0.1063 | 1.0000 | 0.0073 | 1.0000 |
| ID | 101 | full | 196608 | 0.1225 | 0.9968 | 8.0000 | 0.0039 | 0.0551 | 1.0000 | 0.0020 | 1.0000 |
| ID | 102 | 10k | 10000 | 6.6861 | 0.0056 | 8.0000 | 0.0039 | 0.8179 | 1.0000 | 0.3004 | 1.0000 |
| ID | 102 | 100k | 100000 | 0.3425 | 0.9841 | 8.0000 | 0.0039 | 0.1063 | 1.0000 | 0.0073 | 1.0000 |
| ID | 102 | full | 196608 | 0.1252 | 0.9966 | 8.0000 | 0.0039 | 0.0551 | 1.0000 | 0.0020 | 1.0000 |
| ID | 103 | 10k | 10000 | 6.6598 | 0.0056 | 8.0000 | 0.0039 | 0.8179 | 1.0000 | 0.3004 | 1.0000 |
| ID | 103 | 100k | 100000 | 0.3326 | 0.9871 | 8.0000 | 0.0039 | 0.1063 | 1.0000 | 0.0073 | 1.0000 |
| ID | 103 | full | 196608 | 0.1199 | 0.9966 | 8.0000 | 0.0039 | 0.0551 | 1.0000 | 0.0020 | 1.0000 |
| ORDER | 101 | 10k | 10000 | 3.7756 | 0.8547 | 8.0000 | 0.0000 | 4.0015 | 0.0625 | 0.9288 | 1.0000 |
| ORDER | 101 | 100k | 100000 | 0.0883 | 1.0000 | 8.0000 | 0.0000 | 4.0000 | 0.0625 | 0.1314 | 1.0000 |
| ORDER | 101 | full | 262144 | 0.0307 | 1.0000 | 8.0000 | 0.0000 | 4.0000 | 0.0625 | 0.0518 | 1.0000 |
| ORDER | 102 | 10k | 10000 | 3.7753 | 0.8562 | 8.0000 | 0.0000 | 4.0015 | 0.0625 | 0.9288 | 1.0000 |
| ORDER | 102 | 100k | 100000 | 0.0884 | 1.0000 | 8.0000 | 0.0000 | 4.0000 | 0.0625 | 0.1314 | 1.0000 |
| ORDER | 102 | full | 262144 | 0.0309 | 1.0000 | 8.0000 | 0.0000 | 4.0000 | 0.0625 | 0.0518 | 1.0000 |
| ORDER | 103 | 10k | 10000 | 3.7747 | 0.8567 | 8.0000 | 0.0000 | 4.0015 | 0.0625 | 0.9288 | 1.0000 |
| ORDER | 103 | 100k | 100000 | 0.0886 | 1.0000 | 8.0000 | 0.0000 | 4.0000 | 0.0625 | 0.1314 | 1.0000 |
| ORDER | 103 | full | 262144 | 0.0307 | 1.0000 | 8.0000 | 0.0000 | 4.0000 | 0.0625 | 0.0518 | 1.0000 |

The controlled CSV also includes full-stream BPB for the brain, no-write control, and all-stream Markov2 baseline.

## Route arms and copied-checkpoint lesions

V9.2 predicts only through engineered temporal cells: `temporal_only` is the intact predictor (native readout removed by construction), while `native_only` removes the temporal cells and leaves the learned output bias. These are frozen evaluation lesions, not separate training histories.

| task | world | curve | arm | lesion | target BPB | target acc |
| --- | --- | --- | --- | --- | --- | --- |
| ID | 101 | 10k | full151_ordered | none | 6.7029 | 0.0056 |
| ID | 101 | 10k | full151_ordered | native_only | 8.0230 | 0.0039 |
| ID | 101 | 10k | full151_ordered | temporal_only | 6.7029 | 0.0056 |
| ID | 101 | 10k | full151_ordered_no_predictive_write | none | 8.0000 | 0.0039 |
| ID | 101 | 100k | full151_ordered | none | 0.3398 | 0.9814 |
| ID | 101 | 100k | full151_ordered | native_only | 8.0257 | 0.0039 |
| ID | 101 | 100k | full151_ordered | temporal_only | 0.3398 | 0.9814 |
| ID | 101 | 100k | full151_ordered_no_predictive_write | none | 8.0000 | 0.0039 |
| ID | 101 | full | full151_ordered | none | 0.1225 | 0.9968 |
| ID | 101 | full | full151_ordered | native_only | 8.0316 | 0.0039 |
| ID | 101 | full | full151_ordered | temporal_only | 0.1225 | 0.9968 |
| ID | 101 | full | full151_ordered_no_predictive_write | none | 8.0000 | 0.0039 |
| ID | 102 | 10k | full151_ordered | none | 6.6861 | 0.0056 |
| ID | 102 | 10k | full151_ordered | native_only | 8.0224 | 0.0039 |
| ID | 102 | 10k | full151_ordered | temporal_only | 6.6861 | 0.0056 |
| ID | 102 | 10k | full151_ordered_no_predictive_write | none | 8.0000 | 0.0039 |
| ID | 102 | 100k | full151_ordered | none | 0.3425 | 0.9841 |
| ID | 102 | 100k | full151_ordered | native_only | 8.0241 | 0.0039 |
| ID | 102 | 100k | full151_ordered | temporal_only | 0.3425 | 0.9841 |
| ID | 102 | 100k | full151_ordered_no_predictive_write | none | 8.0000 | 0.0039 |
| ID | 102 | full | full151_ordered | none | 0.1252 | 0.9966 |
| ID | 102 | full | full151_ordered | native_only | 8.0296 | 0.0039 |
| ID | 102 | full | full151_ordered | temporal_only | 0.1252 | 0.9966 |
| ID | 102 | full | full151_ordered_no_predictive_write | none | 8.0000 | 0.0039 |
| ID | 103 | 10k | full151_ordered | none | 6.6598 | 0.0056 |
| ID | 103 | 10k | full151_ordered | native_only | 8.0224 | 0.0039 |
| ID | 103 | 10k | full151_ordered | temporal_only | 6.6598 | 0.0056 |
| ID | 103 | 10k | full151_ordered_no_predictive_write | none | 8.0000 | 0.0039 |
| ID | 103 | 100k | full151_ordered | none | 0.3326 | 0.9871 |
| ID | 103 | 100k | full151_ordered | native_only | 8.0244 | 0.0039 |
| ID | 103 | 100k | full151_ordered | temporal_only | 0.3326 | 0.9871 |
| ID | 103 | 100k | full151_ordered_no_predictive_write | none | 8.0000 | 0.0039 |
| ID | 103 | full | full151_ordered | none | 0.1199 | 0.9966 |
| ID | 103 | full | full151_ordered | native_only | 8.0295 | 0.0039 |
| ID | 103 | full | full151_ordered | temporal_only | 0.1199 | 0.9966 |
| ID | 103 | full | full151_ordered_no_predictive_write | none | 8.0000 | 0.0039 |
| ORDER | 101 | 10k | full151_ordered | none | 3.7756 | 0.8547 |
| ORDER | 101 | 10k | full151_ordered | native_only | 7.8408 | 0.0000 |
| ORDER | 101 | 10k | full151_ordered | temporal_only | 3.7756 | 0.8547 |
| ORDER | 101 | 10k | full151_ordered_no_predictive_write | none | 8.0000 | 0.0000 |
| ORDER | 101 | 100k | full151_ordered | none | 0.0883 | 1.0000 |
| ORDER | 101 | 100k | full151_ordered | native_only | 7.6209 | 0.0000 |
| ORDER | 101 | 100k | full151_ordered | temporal_only | 0.0883 | 1.0000 |
| ORDER | 101 | 100k | full151_ordered_no_predictive_write | none | 8.0000 | 0.0000 |
| ORDER | 101 | full | full151_ordered | none | 0.0307 | 1.0000 |
| ORDER | 101 | full | full151_ordered | native_only | 7.6052 | 0.0000 |
| ORDER | 101 | full | full151_ordered | temporal_only | 0.0307 | 1.0000 |
| ORDER | 101 | full | full151_ordered_no_predictive_write | none | 8.0000 | 0.0000 |
| ORDER | 102 | 10k | full151_ordered | none | 3.7753 | 0.8562 |
| ORDER | 102 | 10k | full151_ordered | native_only | 7.8408 | 0.0000 |
| ORDER | 102 | 10k | full151_ordered | temporal_only | 3.7753 | 0.8562 |
| ORDER | 102 | 10k | full151_ordered_no_predictive_write | none | 8.0000 | 0.0000 |
| ORDER | 102 | 100k | full151_ordered | none | 0.0884 | 1.0000 |
| ORDER | 102 | 100k | full151_ordered | native_only | 7.6209 | 0.0000 |
| ORDER | 102 | 100k | full151_ordered | temporal_only | 0.0884 | 1.0000 |
| ORDER | 102 | 100k | full151_ordered_no_predictive_write | none | 8.0000 | 0.0000 |
| ORDER | 102 | full | full151_ordered | none | 0.0309 | 1.0000 |
| ORDER | 102 | full | full151_ordered | native_only | 7.6054 | 0.0000 |
| ORDER | 102 | full | full151_ordered | temporal_only | 0.0309 | 1.0000 |
| ORDER | 102 | full | full151_ordered_no_predictive_write | none | 8.0000 | 0.0000 |
| ORDER | 103 | 10k | full151_ordered | none | 3.7747 | 0.8567 |
| ORDER | 103 | 10k | full151_ordered | native_only | 7.8408 | 0.0000 |
| ORDER | 103 | 10k | full151_ordered | temporal_only | 3.7747 | 0.8567 |
| ORDER | 103 | 10k | full151_ordered_no_predictive_write | none | 8.0000 | 0.0000 |
| ORDER | 103 | 100k | full151_ordered | none | 0.0886 | 1.0000 |
| ORDER | 103 | 100k | full151_ordered | native_only | 7.6213 | 0.0000 |
| ORDER | 103 | 100k | full151_ordered | temporal_only | 0.0886 | 1.0000 |
| ORDER | 103 | 100k | full151_ordered_no_predictive_write | none | 8.0000 | 0.0000 |
| ORDER | 103 | full | full151_ordered | none | 0.0307 | 1.0000 |
| ORDER | 103 | full | full151_ordered | native_only | 7.6052 | 0.0000 |
| ORDER | 103 | full | full151_ordered | temporal_only | 0.0307 | 1.0000 |
| ORDER | 103 | full | full151_ordered_no_predictive_write | none | 8.0000 | 0.0000 |

## Ordered-pair swap counterfactual

These metrics exclude the swap-invariant ordered pairs; positive BPB penalty means the swap worsened prediction.

| world | curve | noninvariant n | normal BPB | swap BPB | swap penalty | accuracy drop |
| --- | --- | --- | --- | --- | --- | --- |
| 101 | 10k | 3584 | 3.7745 | 7.5578 | 3.7833 | 0.8557 |
| 101 | 100k | 3584 | 0.0884 | 11.2845 | 11.1961 | 1.0000 |
| 101 | full | 3584 | 0.0308 | 13.0281 | 12.9973 | 1.0000 |
| 102 | 10k | 3584 | 3.7777 | 7.5604 | 3.7827 | 0.8597 |
| 102 | 100k | 3584 | 0.0886 | 11.2954 | 11.2068 | 1.0000 |
| 102 | full | 3584 | 0.0309 | 13.0367 | 13.0058 | 1.0000 |
| 103 | 10k | 3584 | 3.7805 | 7.5589 | 3.7784 | 0.8588 |
| 103 | 100k | 3584 | 0.0888 | 11.2945 | 11.2057 | 1.0000 |
| 103 | full | 3584 | 0.0307 | 13.0433 | 13.0126 | 1.0000 |

## Source-separated natural text

Each author is scored separately. Full-stream BPB covers every raw byte; content BPB uses the scorer's Unicode letter/number byte mask. A missing 8-gram value means that baseline was absent from the saved result.

| author | curve | arm | brain full BPB | brain content BPB | M2 full BPB | M2 content BPB | 8gram full BPB | 8gram content BPB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Arthur Conan Doyle | 10k | full151_ordered | 4.4692 | 4.9337 | 3.3685 | 3.7060 | 3.3367 | 3.7288 |
| Arthur Conan Doyle | 10k | full151_ordered_no_predictive_write | 8.0000 | 8.0000 | 3.3685 | 3.7060 | 3.3367 | 3.7288 |
| Robert Louis Stevenson | 10k | full151_ordered | 4.4049 | 4.7718 | 3.3187 | 3.5837 | 3.2749 | 3.5799 |
| Robert Louis Stevenson | 10k | full151_ordered_no_predictive_write | 8.0000 | 8.0000 | 3.3187 | 3.5837 | 3.2749 | 3.5799 |
| Arthur Conan Doyle | 100k | full151_ordered | 2.9556 | 3.2829 | 3.0453 | 3.3730 | 2.7702 | 3.1081 |
| Arthur Conan Doyle | 100k | full151_ordered_no_predictive_write | 8.0000 | 8.0000 | 3.0453 | 3.3730 | 2.7702 | 3.1081 |
| Robert Louis Stevenson | 100k | full151_ordered | 2.8564 | 3.0982 | 2.9644 | 3.2334 | 2.7092 | 2.9845 |
| Robert Louis Stevenson | 100k | full151_ordered_no_predictive_write | 8.0000 | 8.0000 | 2.9644 | 3.2334 | 2.7092 | 2.9845 |
| Arthur Conan Doyle | full | full151_ordered | 2.8096 | 3.1465 | 3.0010 | 3.3300 | 2.6384 | 2.9531 |
| Arthur Conan Doyle | full | full151_ordered_no_predictive_write | 8.0000 | 8.0000 | 3.0010 | 3.3300 | 2.6384 | 2.9531 |
| Robert Louis Stevenson | full | full151_ordered | 2.7214 | 2.9829 | 2.9256 | 3.1971 | 2.6028 | 2.8751 |
| Robert Louis Stevenson | full | full151_ordered_no_predictive_write | 8.0000 | 8.0000 | 2.9256 | 3.1971 | 2.6028 | 2.8751 |

Machine-readable tables: `controlled_curve.csv`, `route_arms_lesions.csv`, `order_swap.csv`, `natural_text.csv`. Static figure: `FIGURE.svg`.

## Locked text decision

Status: `confirmation`. Two-author pass: `True`. The rule compares each author's full and content-byte BPB with train-only Markov-2 and matched no-write.
