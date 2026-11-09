# 9.40 — E16-WR2 observed replay

Increasing the total reservoir to 65,536 slots raised final held-composition accuracy by 10.17–12.96 percentage points across the three seeds. It still lost 15.67 / 19.00 / 18.46 points after acquisition, in seed order. Final mod-97 accuracy was 95.33% / 95.62% / 96.67%. P1 met both thresholds in 0 of 3 seeds.

At 16,384 total slots, 0 of 3 seeds met both endpoint bands against the prior per-stage reservoir. Across the two Part A capacities, 3 of 6 lives met the acquisition band against E16-W uniform replay. The larger reservoir finished with 21,351–21,516 distinct prompts. In both capacity arms the stage-end design held up to 393,216 pending records in addition to the retained reservoir.

Part B reproduced the E8 demonstration with observed answers. It changed final held-composition accuracy by +0.46 / -0.04 / +3.54 points relative to E13-A1. The eight-operation means met P4 in 3 of 3 seeds. Composition is taught last in Part B, so the endpoint measures acquisition and retention of prior operations. All three full Part B input streams matched prompt hashes independently reconstructed from the original schedule and sampler. The archived E13-A1 runs supply scores, and their original prompt streams were not recorded.

| Job | Post-compose held % | Final held % | Loss pp | Final held full % | Final mod-97 % | Final eight operations % | Weakest carry | Carry final % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A16384_73129 | 85.2500 | 56.0000 | 29.2500 | 50.2083 | 94.2500 | 93.4375 | +- | 21.1667 |
| A16384_93683 | 83.4167 | 50.0833 | 33.3333 | 43.2083 | 95.6250 | 94.6042 | +- | 18.6667 |
| A16384_116089 | 84.5000 | 54.7083 | 29.7917 | 48.5000 | 95.4167 | 95.1250 | +- | 21.1667 |
| A65536_73129 | 81.8333 | 66.1667 | 15.6667 | 59.8333 | 95.3333 | 94.5000 | +- | 37.5000 |
| A65536_93683 | 81.6250 | 62.6250 | 19.0000 | 58.2083 | 95.6250 | 94.6042 | +- | 33.1667 |
| A65536_116089 | 86.1250 | 67.6667 | 18.4583 | 63.2083 | 96.6667 | 95.1458 | +- | 34.3333 |
| B_73129 | 82.7500 | 82.7500 | — | 80.6667 | 95.0417 | 94.2708 | +- | 66.0000 |
| B_93683 | 80.9583 | 80.9583 | — | 78.9583 | 95.8333 | 94.2292 | +- | 69.0000 |
| B_116089 | 88.5417 | 88.5417 | — | 86.6250 | 95.8750 | 94.7292 | +- | 80.6667 |

| Prediction | Passing | Evaluated | Expected |
| --- | --- | --- | --- |
| P1 | 0 | 3 | 3 |
| P2 | 0 | 3 | 3 |
| P3 | 3 | 6 | 6 |
| P4 | 3 | 3 | 3 |
| P5 | 2 | 3 | 3 |

The campaign used 3.4059 child CPU-hours and 28.47 minutes of wall time. The total and per-stage arms differ in retained capacity as well as sampling. The observed targets agree byte for byte with the saved observations.

See [the full report](E16WR2_REPORT.md) and [verification receipt](E16WR2_VERIFICATION.json).
