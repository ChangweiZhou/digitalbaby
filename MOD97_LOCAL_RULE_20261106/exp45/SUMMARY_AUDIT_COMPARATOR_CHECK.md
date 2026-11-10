# E16 summary: the integrated toy learner

Scores are exact answers (%) on sealed panels; composition rows give final-answer accuracy. Blocks: F1.1–F1.2 mod-97 operations with distractors, F2.1–F2.2 taught operator pairs, F3.1–F3.2 mod-73 operations.

## Family means by block (k = 12 for operations; held-pair final-answer mean for composition)

| Arm | Seed | Block | F1 mean | F1 add | F1 sub | F3 mean | Held comp | +- | -+ | >< | <> | within | dev89 mean |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| D | 600013 | birth | 22.2 | 6.0 | 2.0 | 24.2 | 25.2 | 1.0 | 1.5 | 41.5 | 56.8 | 15.5 | 93.2 |
| D | 600013 | F1.1 | 91.9 | 88.0 | 87.2 | 30.3 | 28.0 | 1.2 | 1.8 | 49.0 | 60.2 | 19.5 | 48.0 |
| D | 600013 | F1.2 | 92.4 | 89.2 | 86.2 | 31.8 | 29.5 | 1.0 | 1.5 | 53.0 | 62.3 | 21.7 | 35.1 |
| D | 600013 | F2.1 | 92.4 | 87.2 | 87.0 | 38.7 | 86.7 | 77.5 | 78.8 | 94.0 | 96.5 | 88.8 | 40.3 |
| D | 600013 | F2.2 | 92.6 | 90.5 | 86.2 | 42.1 | 92.6 | 87.0 | 87.0 | 98.3 | 98.2 | 93.5 | 45.4 |
| D | 600013 | F3.1 | 93.4 | 89.2 | 89.2 | 93.8 | 92.6 | 86.3 | 86.7 | 98.5 | 98.8 | 94.0 | 50.9 |
| D | 600013 | F3.2 | 94.4 | 90.0 | 90.2 | 94.5 | 93.2 | 87.3 | 88.5 | 98.5 | 98.3 | 95.5 | 57.4 |
| T_oracle | 600013 | birth | 22.2 | 6.0 | 2.0 | 24.2 | 24.6 | 1.2 | 1.0 | 39.2 | 57.0 | 15.8 | 93.2 |
| T_oracle | 600013 | F1.1 | 82.9 | 70.8 | 77.2 | 49.7 | 22.8 | 0.2 | 1.0 | 51.0 | 39.2 | 17.8 | 47.8 |
| T_oracle | 600013 | F1.2 | 87.9 | 81.8 | 79.5 | 37.2 | 11.1 | 0.2 | 0.3 | 31.7 | 12.3 | 9.5 | 35.5 |
| T_oracle | 600013 | F2.1 | 65.7 | 50.0 | 41.5 | 37.7 | 64.0 | 28.7 | 55.8 | 84.2 | 87.5 | 76.7 | 39.2 |
| T_oracle | 600013 | F2.2 | 83.2 | 76.2 | 70.2 | 44.1 | 77.9 | 61.7 | 64.5 | 94.0 | 91.5 | 82.8 | 44.7 |
| T_oracle | 600013 | F3.1 | 73.9 | 52.2 | 53.8 | 77.6 | 79.1 | 67.7 | 62.3 | 95.0 | 91.3 | 84.2 | 46.0 |
| T_oracle | 600013 | F3.2 | 83.7 | 78.0 | 68.0 | 83.2 | 78.8 | 62.0 | 71.0 | 91.5 | 90.8 | 85.0 | 51.2 |
| T_raw65536 | 600013 | birth | 22.2 | 6.0 | 2.0 | 24.2 | 25.2 | 1.0 | 1.5 | 41.5 | 56.8 | 15.5 | 93.2 |
| T_raw65536 | 600013 | F1.1 | 80.9 | 69.5 | 70.0 | 62.2 | 20.0 | 0.7 | 1.5 | 25.2 | 52.5 | 18.7 | 60.9 |
| T_raw65536 | 600013 | F1.2 | 85.2 | 77.8 | 78.0 | 59.8 | 26.8 | 0.8 | 1.7 | 45.5 | 59.3 | 21.2 | 58.5 |
| T_raw65536 | 600013 | F2.1 | 52.1 | 24.8 | 15.2 | 34.6 | 67.0 | 48.8 | 48.7 | 84.2 | 86.5 | 78.7 | 36.7 |
| T_raw65536 | 600013 | F2.2 | 66.8 | 50.0 | 43.2 | 34.6 | 81.2 | 65.2 | 76.3 | 93.5 | 89.8 | 90.0 | 33.8 |
| T_raw65536 | 600013 | F3.1 | 76.9 | 59.5 | 56.8 | 82.8 | 62.8 | 44.2 | 46.8 | 84.3 | 75.7 | 82.8 | 36.3 |
| T_raw65536 | 600013 | F3.2 | 76.7 | 60.5 | 59.0 | 84.8 | 64.7 | 49.7 | 53.8 | 81.3 | 73.8 | 78.2 | 33.6 |
| T_raw65536_drift_online | 600013 | birth | 22.2 | 6.0 | 2.0 | 24.2 | 24.6 | 1.2 | 1.0 | 39.2 | 57.0 | 15.8 | 93.2 |
| T_raw65536_drift_online | 600013 | F1.1 | 82.1 | 67.2 | 73.2 | 55.2 | 21.7 | 0.3 | 0.5 | 27.0 | 58.8 | 19.3 | 59.9 |
| T_raw65536_drift_online | 600013 | F1.2 | 79.9 | 68.2 | 70.8 | 48.7 | 27.3 | 2.5 | 1.8 | 48.3 | 56.7 | 28.5 | 46.0 |
| T_raw65536_drift_online | 600013 | F2.1 | 47.2 | 28.2 | 24.5 | 36.1 | 72.0 | 57.5 | 56.0 | 88.2 | 86.5 | 80.0 | 37.8 |
| T_raw65536_drift_online | 600013 | F2.2 | 60.2 | 34.0 | 31.2 | 62.1 | 80.0 | 63.7 | 73.7 | 92.2 | 90.5 | 84.0 | 47.6 |
| T_raw65536_drift_online | 600013 | F3.1 | 69.8 | 50.2 | 41.2 | 80.8 | 67.1 | 45.2 | 47.2 | 87.5 | 88.5 | 77.2 | 52.8 |
| T_raw65536_drift_online | 600013 | F3.2 | 72.8 | 52.0 | 47.5 | 85.8 | 66.8 | 41.7 | 47.8 | 89.0 | 88.8 | 76.5 | 53.6 |
| D_drift | 600013 | birth | 22.2 | 6.0 | 2.0 | 24.2 | 24.6 | 1.2 | 1.0 | 39.2 | 57.0 | 15.8 | 93.2 |
| D_drift | 600013 | F1.1 | 91.9 | 88.0 | 87.2 | 30.3 | 27.4 | 1.0 | 0.7 | 47.2 | 60.7 | 19.5 | 48.0 |
| D_drift | 600013 | F1.2 | 92.4 | 89.2 | 86.2 | 31.8 | 29.5 | 0.8 | 1.3 | 53.2 | 62.5 | 20.5 | 35.1 |
| D_drift | 600013 | F2.1 | 91.9 | 89.0 | 85.5 | 37.0 | 87.9 | 79.7 | 80.7 | 93.7 | 97.7 | 90.0 | 37.8 |
| D_drift | 600013 | F2.2 | 93.4 | 91.2 | 87.5 | 43.2 | 93.1 | 88.2 | 86.7 | 98.8 | 98.8 | 94.5 | 44.8 |
| D_drift | 600013 | F3.1 | 94.5 | 92.0 | 88.0 | 93.2 | 92.8 | 88.7 | 86.0 | 98.3 | 98.3 | 93.7 | 49.0 |
| D_drift | 600013 | F3.2 | 94.1 | 89.0 | 89.5 | 93.8 | 94.8 | 91.2 | 91.2 | 98.7 | 98.2 | 94.7 | 54.3 |
| T_raw65536_drift_online_rs0 | 600013 | birth | 22.2 | 6.0 | 2.0 | 24.2 | 24.6 | 1.2 | 1.0 | 39.2 | 57.0 | 15.8 | 93.2 |
| T_raw65536_drift_online_rs0 | 600013 | F1.1 | 86.6 | 76.0 | 78.5 | 57.4 | 20.5 | 0.5 | 0.7 | 38.2 | 42.5 | 22.2 | 58.4 |
| T_raw65536_drift_online_rs0 | 600013 | F1.2 | 69.4 | 51.5 | 49.5 | 43.7 | 32.7 | 2.5 | 2.7 | 59.5 | 66.2 | 34.5 | 44.8 |
| T_raw65536_drift_online_rs0 | 600013 | F2.1 | 0.2 | 0.2 | 0.5 | 0.1 | 77.3 | 39.0 | 78.3 | 95.7 | 96.2 | 92.3 | 0.0 |
| T_raw65536_drift_online_rs0 | 600013 | F2.2 | 25.9 | 1.8 | 0.5 | 31.9 | 73.4 | 36.3 | 69.0 | 96.7 | 91.7 | 90.7 | 26.1 |
| T_raw65536_drift_online_rs0 | 600013 | F3.1 | 26.9 | 2.2 | 8.5 | 80.2 | 15.3 | 1.2 | 0.8 | 36.0 | 23.3 | 8.7 | 30.0 |
| T_raw65536_drift_online_rs0 | 600013 | F3.2 | 30.2 | 3.8 | 7.5 | 80.1 | 17.9 | 0.8 | 1.2 | 39.7 | 30.0 | 11.0 | 36.0 |
| T_raw65536_drift_online_rs25 | 600013 | birth | 22.2 | 6.0 | 2.0 | 24.2 | 24.6 | 1.2 | 1.0 | 39.2 | 57.0 | 15.8 | 93.2 |
| T_raw65536_drift_online_rs25 | 600013 | F1.1 | 85.6 | 75.8 | 76.5 | 60.7 | 24.9 | 0.8 | 1.5 | 43.7 | 53.5 | 21.7 | 62.2 |
| T_raw65536_drift_online_rs25 | 600013 | F1.2 | 73.2 | 58.0 | 60.0 | 41.9 | 28.8 | 1.3 | 1.5 | 49.8 | 62.7 | 27.2 | 43.0 |
| T_raw65536_drift_online_rs25 | 600013 | F2.1 | 36.2 | 11.8 | 7.8 | 33.3 | 67.2 | 35.5 | 53.2 | 90.2 | 89.8 | 82.2 | 32.7 |
| T_raw65536_drift_online_rs25 | 600013 | F2.2 | 50.9 | 20.2 | 15.8 | 59.6 | 68.1 | 31.3 | 60.5 | 93.2 | 87.5 | 85.7 | 47.9 |
| T_raw65536_drift_online_rs25 | 600013 | F3.1 | 61.3 | 30.2 | 29.0 | 84.2 | 48.4 | 15.2 | 21.7 | 77.8 | 79.0 | 59.3 | 49.8 |
| T_raw65536_drift_online_rs25 | 600013 | F3.2 | 64.3 | 38.5 | 33.0 | 88.3 | 49.7 | 20.3 | 26.3 | 74.2 | 78.0 | 55.8 | 38.9 |
| T_raw65536_drift_online_rs75 | 600013 | birth | 22.2 | 6.0 | 2.0 | 24.2 | 24.6 | 1.2 | 1.0 | 39.2 | 57.0 | 15.8 | 93.2 |
| T_raw65536_drift_online_rs75 | 600013 | F1.1 | 82.2 | 69.8 | 72.5 | 64.1 | 25.0 | 0.8 | 0.5 | 45.2 | 53.5 | 22.0 | 66.0 |
| T_raw65536_drift_online_rs75 | 600013 | F1.2 | 80.7 | 70.0 | 71.8 | 39.7 | 21.5 | 0.8 | 1.8 | 41.7 | 41.5 | 20.5 | 43.2 |
| T_raw65536_drift_online_rs75 | 600013 | F2.1 | 69.0 | 51.8 | 53.8 | 46.7 | 66.8 | 43.5 | 49.8 | 87.7 | 86.0 | 71.3 | 45.6 |
| T_raw65536_drift_online_rs75 | 600013 | F2.2 | 71.2 | 51.2 | 50.8 | 61.9 | 75.9 | 59.8 | 64.8 | 91.8 | 87.2 | 81.7 | 48.6 |
| T_raw65536_drift_online_rs75 | 600013 | F3.1 | 74.4 | 54.5 | 52.8 | 80.4 | 73.0 | 52.3 | 61.5 | 89.8 | 88.5 | 83.7 | 54.1 |
| T_raw65536_drift_online_rs75 | 600013 | F3.2 | 76.5 | 61.5 | 55.8 | 85.1 | 75.5 | 57.3 | 63.2 | 90.5 | 91.0 | 80.3 | 50.6 |
| M_raw65536_drift_online_rs25 | 600013 | birth | 22.2 | 6.0 | 2.0 | 24.2 | 24.6 | 1.2 | 1.0 | 39.2 | 57.0 | 15.8 | 93.2 |
| M_raw65536_drift_online_rs25 | 600013 | F1.1 | 92.5 | 87.5 | 90.0 | 31.1 | 26.2 | 0.8 | 0.8 | 42.8 | 60.5 | 19.2 | 54.5 |
| M_raw65536_drift_online_rs25 | 600013 | F1.2 | 91.9 | 87.8 | 86.2 | 16.2 | 47.8 | 14.8 | 10.3 | 81.5 | 84.5 | 54.2 | 20.5 |
| M_raw65536_drift_online_rs25 | 600013 | F2.1 | 90.4 | 85.0 | 83.2 | 31.3 | 93.1 | 88.5 | 86.7 | 98.8 | 98.5 | 94.5 | 33.8 |
| M_raw65536_drift_online_rs25 | 600013 | F2.2 | 93.7 | 91.5 | 88.0 | 90.6 | 95.6 | 94.2 | 90.3 | 99.0 | 98.8 | 95.7 | 29.8 |
| M_raw65536_drift_online_rs25 | 600013 | F3.1 | 93.1 | 86.8 | 88.8 | 94.1 | 85.6 | 74.0 | 77.8 | 94.8 | 95.8 | 88.7 | 40.3 |
| M_raw65536_drift_online_rs25 | 600013 | F3.2 | 93.1 | 89.2 | 86.8 | 94.3 | 83.5 | 68.8 | 75.0 | 94.3 | 96.0 | 85.2 | 45.0 |
| M_raw65536_drift_online_rs75 | 600013 | birth | 22.2 | 6.0 | 2.0 | 24.2 | 24.6 | 1.2 | 1.0 | 39.2 | 57.0 | 15.8 | 93.2 |
| M_raw65536_drift_online_rs75 | 600013 | F1.1 | 92.2 | 90.2 | 87.0 | 31.8 | 27.3 | 0.8 | 0.8 | 45.3 | 62.2 | 21.2 | 53.5 |
| M_raw65536_drift_online_rs75 | 600013 | F1.2 | 92.4 | 88.8 | 86.2 | 17.2 | 37.7 | 3.3 | 3.5 | 73.2 | 70.7 | 40.0 | 19.2 |
| M_raw65536_drift_online_rs75 | 600013 | F2.1 | 90.6 | 86.2 | 84.2 | 38.4 | 87.9 | 81.3 | 79.7 | 95.3 | 95.2 | 88.7 | 41.7 |
| M_raw65536_drift_online_rs75 | 600013 | F2.2 | 92.3 | 87.5 | 87.2 | 88.6 | 91.8 | 85.5 | 87.0 | 97.3 | 97.5 | 94.8 | 34.4 |
| M_raw65536_drift_online_rs75 | 600013 | F3.1 | 94.0 | 89.5 | 88.2 | 92.9 | 90.4 | 83.8 | 86.0 | 95.5 | 96.3 | 89.8 | 46.2 |
| M_raw65536_drift_online_rs75 | 600013 | F3.2 | 94.4 | 90.2 | 91.0 | 94.2 | 91.6 | 86.8 | 86.5 | 95.2 | 97.8 | 94.3 | 48.8 |
| M_raw65536_drift_online | 600013 | birth | 22.2 | 6.0 | 2.0 | 24.2 | 24.6 | 1.2 | 1.0 | 39.2 | 57.0 | 15.8 | 93.2 |
| M_raw65536_drift_online | 600013 | F1.1 | 92.8 | 88.8 | 88.5 | 31.4 | 26.3 | 1.5 | 0.8 | 41.8 | 61.0 | 20.0 | 53.5 |
| M_raw65536_drift_online | 600013 | F1.2 | 92.2 | 87.5 | 86.5 | 18.7 | 43.2 | 7.0 | 6.5 | 78.0 | 81.2 | 47.8 | 21.1 |
| M_raw65536_drift_online | 600013 | F2.1 | 92.1 | 88.8 | 86.2 | 37.1 | 90.9 | 85.8 | 81.8 | 98.3 | 97.5 | 91.8 | 40.3 |
| M_raw65536_drift_online | 600013 | F2.2 | 94.0 | 89.2 | 91.8 | 90.9 | 92.1 | 87.8 | 88.7 | 94.0 | 98.0 | 92.2 | 29.4 |
| M_raw65536_drift_online | 600013 | F3.1 | 94.5 | 87.8 | 92.0 | 93.8 | 89.0 | 78.8 | 84.0 | 96.2 | 97.0 | 92.3 | 42.0 |
| M_raw65536_drift_online | 600013 | F3.2 | 95.1 | 90.5 | 92.0 | 93.9 | 89.1 | 80.0 | 83.7 | 95.0 | 97.8 | 89.7 | 51.5 |

## After the last block: learner against twin (gate: T ≥ 80% of D on every family)

| Seed | Arm: family | arm | D | arm/D | F1 after block 2 (T) | T k=20 | T silenced k=12 | Evidence status |
|---|---|---:|---:|---:|---:|---:|---:|---|
| 600013 | T on the twin's stream: F1 ops k=12 | 83.7 | 94.4 | withheld | 87.9 | 77.6 | 4.4 | WITHHELD: panels differ (corrected / original) |
| 600013 | T on the twin's stream: F3 ops k=12 | 83.2 | 94.5 | withheld |  | 79.1 | 5.6 | WITHHELD: panels differ (corrected / original) |
| 600013 | T on the twin's stream: held comp final | 78.8 | 93.2 | withheld |  |  | 11.3 | WITHHELD: panels differ (corrected / original) |
| 600013 | T raw bytes verbatim, 65,536, no parser: F1 ops k=12 | 76.7 | 94.4 | 0.81 | 85.2 | 68.8 | 4.6 | legacy: no completion record or panel hash; not admissible to a new gate |
| 600013 | T raw bytes verbatim, 65,536, no parser: F3 ops k=12 | 84.8 | 94.5 | 0.90 |  | 79.2 | 5.4 | legacy: no completion record or panel hash; not admissible to a new gate |
| 600013 | T raw bytes verbatim, 65,536, no parser: held comp final | 64.7 | 93.2 | 0.69 |  |  | 8.8 | legacy: no completion record or panel hash; not admissible to a new gate |
| 600013 | T raw 65,536, drifting stream, no stages: F1 ops k=12 | 72.8 | 94.1 | 0.77 | 79.9 | 62.4 | 9.0 | legacy: no completion record or panel hash; not admissible to a new gate |
| 600013 | T raw 65,536, drifting stream, no stages: F3 ops k=12 | 85.8 | 93.8 | 0.91 |  | 78.1 | 9.6 | legacy: no completion record or panel hash; not admissible to a new gate |
| 600013 | T raw 65,536, drifting stream, no stages: held comp final | 66.8 | 94.8 | 0.71 |  |  | 9.4 | legacy: no completion record or panel hash; not admissible to a new gate |
| 600013 | E16-M3 T raw 65,536, drift, replay share 0%: F1 ops k=12 | 30.2 | 94.1 | 0.32 | 69.4 | 28.0 | 7.2 | legacy: no completion record or panel hash; not admissible to a new gate |
| 600013 | E16-M3 T raw 65,536, drift, replay share 0%: F3 ops k=12 | 80.1 | 93.8 | 0.85 |  | 76.2 | 8.9 | legacy: no completion record or panel hash; not admissible to a new gate |
| 600013 | E16-M3 T raw 65,536, drift, replay share 0%: held comp final | 17.9 | 94.8 | 0.19 |  |  | 8.0 | legacy: no completion record or panel hash; not admissible to a new gate |
| 600013 | E16-M3 T raw 65,536, drift, replay share 25%: F1 ops k=12 | 64.3 | 94.1 | 0.68 | 73.2 | 59.1 | 3.9 | legacy: no completion record or panel hash; not admissible to a new gate |
| 600013 | E16-M3 T raw 65,536, drift, replay share 25%: F3 ops k=12 | 88.3 | 93.8 | 0.94 |  | 84.1 | 4.5 | legacy: no completion record or panel hash; not admissible to a new gate |
| 600013 | E16-M3 T raw 65,536, drift, replay share 25%: held comp final | 49.7 | 94.8 | 0.52 |  |  | 6.2 | legacy: no completion record or panel hash; not admissible to a new gate |
| 600013 | E16-M3 T raw 65,536, drift, replay share 75%: F1 ops k=12 | 76.5 | 94.1 | 0.81 | 80.7 | 66.4 | 5.3 | legacy: no completion record or panel hash; not admissible to a new gate |
| 600013 | E16-M3 T raw 65,536, drift, replay share 75%: F3 ops k=12 | 85.1 | 93.8 | 0.91 |  | 76.9 | 7.2 | legacy: no completion record or panel hash; not admissible to a new gate |
| 600013 | E16-M3 T raw 65,536, drift, replay share 75%: held comp final | 75.5 | 94.8 | 0.80 |  |  | 11.7 | legacy: no completion record or panel hash; not admissible to a new gate |
| 600013 | E16-M3 M raw 65,536, drift, replay share 25%: F1 ops k=12 | 93.1 | 94.1 | 0.99 | 91.9 | 91.9 | 1.1 | legacy: no completion record or panel hash; not admissible to a new gate |
| 600013 | E16-M3 M raw 65,536, drift, replay share 25%: F3 ops k=12 | 94.3 | 93.8 | 1.01 |  | 94.1 | 0.9 | legacy: no completion record or panel hash; not admissible to a new gate |
| 600013 | E16-M3 M raw 65,536, drift, replay share 25%: held comp final | 83.5 | 94.8 | 0.88 |  |  | 2.2 | legacy: no completion record or panel hash; not admissible to a new gate |
| 600013 | E16-M3 M raw 65,536, drift, replay share 75%: F1 ops k=12 | 94.4 | 94.1 | 1.00 | 92.4 | 94.0 | 1.6 | legacy: no completion record or panel hash; not admissible to a new gate |
| 600013 | E16-M3 M raw 65,536, drift, replay share 75%: F3 ops k=12 | 94.2 | 93.8 | 1.00 |  | 94.2 | 1.9 | legacy: no completion record or panel hash; not admissible to a new gate |
| 600013 | E16-M3 M raw 65,536, drift, replay share 75%: held comp final | 91.6 | 94.8 | 0.97 |  |  | 4.8 | legacy: no completion record or panel hash; not admissible to a new gate |
| 600013 | E16-M3 M raw 65,536, drift, replay share 50%: F1 ops k=12 | 95.1 | 94.1 | 1.01 | 92.2 | 94.4 | 1.6 | legacy: no completion record or panel hash; not admissible to a new gate |
| 600013 | E16-M3 M raw 65,536, drift, replay share 50%: F3 ops k=12 | 93.9 | 93.8 | 1.00 |  | 93.8 | 2.1 | legacy: no completion record or panel hash; not admissible to a new gate |
| 600013 | E16-M3 M raw 65,536, drift, replay share 50%: held comp final | 89.1 | 94.8 | 0.94 |  |  | 4.2 | legacy: no completion record or panel hash; not admissible to a new gate |

## Retention across arms: what each family keeps after the stream moves on (block 6 vs the family's own last block)

| Arm | Seed | F1 k=12 after 2 → 6 | F1 add/sub after 6 | Held comp (final answer) after 4 → 6 | Held comp full scratchpad exact after 6 | +- / -+ after 6 | within after 4 → 6 | F3 after 6 |
|---|---|---|---|---|---|---|---|---|
| D twin (joint ground truth) | 600013 | 92.4 → 94.4 | 90/90 | 92.6 → 93.2 | 92.8 | 87/88 | 94 → 96 | 94.5 |
| T on the twin's stream | 600013 | 87.9 → 83.7 | 78/68 | 77.9 → 78.8 | 76.0 | 62/71 | 83 → 85 | 83.2 |
| T raw bytes verbatim, 65,536, no parser | 600013 | 85.2 → 76.7 | 60/59 | 81.2 → 64.7 | 60.5 | 50/54 | 90 → 78 | 84.8 |
| T raw 65,536, drifting stream, no stages | 600013 | 79.9 → 72.8 | 52/48 | 80.0 → 66.8 | 61.7 | 42/48 | 84 → 76 | 85.8 |
| D twin on the drifting stream | 600013 | 92.4 → 94.1 | 89/90 | 93.1 → 94.8 | 94.4 | 91/91 | 94 → 95 | 93.8 |
| E16-M3 T raw 65,536, drift, replay share 0% | 600013 | 69.4 → 30.2 | 4/8 | 73.4 → 17.9 | 0.0 | 1/1 | 91 → 11 | 80.1 |
| E16-M3 T raw 65,536, drift, replay share 25% | 600013 | 73.2 → 64.3 | 38/33 | 68.1 → 49.7 | 41.0 | 20/26 | 86 → 56 | 88.3 |
| E16-M3 T raw 65,536, drift, replay share 75% | 600013 | 80.7 → 76.5 | 62/56 | 75.9 → 75.5 | 72.8 | 57/63 | 82 → 80 | 85.1 |
| E16-M3 M raw 65,536, drift, replay share 25% | 600013 | 91.9 → 93.1 | 89/87 | 95.6 → 83.5 | 82.2 | 69/75 | 96 → 85 | 94.3 |
| E16-M3 M raw 65,536, drift, replay share 75% | 600013 | 92.4 → 94.4 | 90/91 | 91.8 → 91.6 | 90.9 | 87/86 | 95 → 94 | 94.2 |
| E16-M3 M raw 65,536, drift, replay share 50% | 600013 | 92.2 → 95.1 | 90/92 | 92.1 → 89.1 | 88.2 | 80/84 | 92 → 90 | 93.9 |

## Per-operation exact % at k = 0 / 8 / 12 / 20 after the last block

| Arm | Seed | add97 | sub97 | max97 | min97 | add73 | sub73 | max73 | min73 |
|---|---|---|---|---|---|---|---|---|---|
| D | 600013 | 94/91/90/90 | 92/90/90/88 | 100/99/99/98 | 98/98/98/98 | 88/90/90/89 | 89/90/90/89 | 100/100/99/99 | 98/99/99/98 |
| T_oracle | 600013 | 70/76/78/70 | 67/70/68/58 | 95/96/92/89 | 97/96/96/93 | 66/73/72/66 | 65/76/73/68 | 94/94/94/90 | 94/95/95/94 |
| T_raw65536 | 600013 | 66/64/60/50 | 61/63/59/51 | 98/96/92/84 | 96/96/95/90 | 76/76/73/68 | 75/81/79/70 | 95/94/93/88 | 93/94/94/91 |
| T_raw65536_drift_online | 600013 | 56/57/52/41 | 47/52/48/32 | 96/96/95/88 | 93/96/96/88 | 75/75/73/66 | 78/82/78/63 | 94/96/96/94 | 94/96/95/89 |
| D_drift | 600013 | 91/89/89/89 | 90/89/90/88 | 100/99/99/98 | 100/98/98/98 | 87/87/87/87 | 92/90/91/92 | 100/99/99/99 | 100/99/98/99 |
| T_raw65536_drift_online_rs0 | 600013 | 3/4/4/4 | 4/8/8/5 | 58/50/48/42 | 68/62/62/61 | 65/71/69/64 | 67/72/69/63 | 92/93/92/89 | 93/91/90/89 |
| T_raw65536_drift_online_rs25 | 600013 | 44/42/38/36 | 35/35/33/30 | 94/93/92/83 | 96/95/94/88 | 77/81/80/76 | 85/82/83/79 | 96/97/97/95 | 94/94/93/87 |
| T_raw65536_drift_online_rs75 | 600013 | 64/65/62/47 | 58/62/56/43 | 96/96/94/85 | 93/96/94/91 | 68/74/72/59 | 72/80/77/66 | 95/96/97/92 | 95/96/95/91 |
| M_raw65536_drift_online_rs25 | 600013 | 91/90/89/88 | 85/87/87/85 | 99/99/99/98 | 96/97/98/97 | 86/87/88/87 | 92/93/93/93 | 99/99/99/100 | 98/98/98/97 |
| M_raw65536_drift_online_rs75 | 600013 | 94/89/90/91 | 93/92/91/90 | 99/99/99/98 | 97/98/98/97 | 88/89/88/90 | 90/93/93/92 | 99/98/98/98 | 97/98/98/98 |
| M_raw65536_drift_online | 600013 | 94/91/90/91 | 95/92/92/90 | 98/98/99/98 | 99/99/99/98 | 86/87/86/85 | 93/93/92/93 | 98/98/98/98 | 100/98/98/98 |

## Shortcut ceilings on the held pairs (final answer from a visible-only rule, %)

| Pair | ignore o1, use a | ignore o1, use b | ignore o2 | answer = c |
|---|---:|---:|---:|---:|
| +- | 1.0 | 1.5 | 1.0 | 0.8 |
| -+ | 0.7 | 1.0 | 1.3 | 1.2 |
| >< | 66.2 | 69.2 | 35.3 | 65.7 |
| <> | 69.0 | 67.8 | 30.3 | 70.5 |

## Comparator checks (arm and comparator must share panels and objective)

- `T_oracle` against `D`, seed 600013: {'panels': ('corrected', 'original (pre-fix)')}
- `D_drift` against `D`, seed 600013: {'panels': ('corrected', 'original (pre-fix)')}
