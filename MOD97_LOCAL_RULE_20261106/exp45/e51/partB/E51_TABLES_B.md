# E51 Part B, seed 73129: the readings

## B1. The share of rehearsal

| rule | share | reduction row 0 | at 4,096 | % of row 0 | kept (within 5 points) | lowest probe (%) | unrelated held at 80%: first probe row | unrelated held at 4,096 | source |
|---|---|---|---|---|---|---|---|---|---|
| S | 1/2 (the second addendum) | 96.6 | 97.1 | 100.5 | yes | - | - | 99.3 | e43_unrelated-rehearse_S_73129 |
| S | 1/4 | 96.6 | 89.0 | 92.2 | no | 55.5 | 512 | 98.7 | e51_B1_share_4_S_73129 |
| S | 1/8 | 96.6 | 86.8 | 89.8 | no | 30.5 | 512 | 99.0 | e51_B1_share_8_S_73129 |
| S | 1/16 | 96.6 | 63.2 | 65.4 | no | 19.8 | 448 | 100.0 | e51_B1_share_16_S_73129 |
| S | 1/32 | 96.6 | 44.2 | 45.8 | no | 7.0 | 448 | 100.0 | e51_B1_share_32_S_73129 |
| S | 0 (E43 `unrelated`) | 96.6 | 0.0 | 0.0 | no | - | - | 100.0 | e43_unrelated_S_73129 |
| S | **the least share that keeps** | | | | 1/2 | | | | |
| B | 1/2 (the second addendum) | 98.1 | 96.7 | 98.6 | yes | - | - | 99.3 | e43_unrelated-rehearse_B_73129 |
| B | 1/4 | 98.1 | 88.9 | 90.6 | no | 61.5 | 256 | 100.0 | e51_B1_share_4_B_73129 |
| B | 1/8 | 98.1 | 78.4 | 79.9 | no | 40.5 | 256 | 99.7 | e51_B1_share_8_B_73129 |
| B | 1/16 | 98.1 | 57.5 | 58.6 | no | 25.8 | 256 | 100.0 | e51_B1_share_16_B_73129 |
| B | 1/32 | 98.1 | 34.6 | 35.3 | no | 18.0 | 256 | 100.0 | e51_B1_share_32_B_73129 |
| B | 0 (E43 `unrelated`) | 98.1 | 0.0 | 0.0 | no | - | - | 100.0 | e43_unrelated_B_73129 |
| B | **the least share that keeps** | | | | 1/2 | | | | |

## B2. Rehearsal at intervals

Rule S (e51_B2_intervals_S_73129): probe at update 0 95.25%, the review's goal 85.73%; E43's reduction score 96.6 at update 0 and 87.9 at the end (update 2640); the unrelated task's held-out score at the end 99.7%.

| cycle | probe at the end of the gap: first family (%) | unrelated held (%) | review updates | goal reached | probe at the end of the review: first family (%) | unrelated held (%) |
|---|---|---|---|---|---|---|
| 1 | 0.00 | 6.0 | 80 | yes | 89.50 | 0.0 |
| 2 | 0.00 | 98.7 | 128 | yes | 85.75 | 0.0 |
| 3 | 3.75 | 100.0 | 96 | yes | 88.00 | 35.3 |
| 4 | 52.50 | 100.0 | 96 | yes | 90.25 | 88.3 |
| 5 | 74.50 | 99.7 | 64 | yes | 88.75 | 97.3 |
| 6 | 65.00 | 99.7 | 48 | yes | 87.00 | 99.0 |
| 7 | 54.25 | 100.0 | 48 | yes | 91.00 | 98.3 |
| 8 | 58.25 | 100.0 | 32 | yes | 88.50 | 99.7 |

The review shortens: yes. More is left: yes. Rule S: with rehearsal at intervals the first family becomes cheaper to restore, and more of it survives a gap.

Rule B (e51_B2_intervals_B_73129): probe at update 0 98.25%, the review's goal 88.42%; E43's reduction score 98.1 at update 0 and 91.9 at the end (update 2496); the unrelated task's held-out score at the end 98.0%.

| cycle | probe at the end of the gap: first family (%) | unrelated held (%) | review updates | goal reached | probe at the end of the review: first family (%) | unrelated held (%) |
|---|---|---|---|---|---|---|
| 1 | 0.00 | 98.0 | 80 | yes | 89.00 | 0.3 |
| 2 | 0.25 | 100.0 | 64 | yes | 90.25 | 0.3 |
| 3 | 11.75 | 100.0 | 64 | yes | 91.75 | 69.7 |
| 4 | 46.75 | 100.0 | 64 | yes | 89.25 | 95.7 |
| 5 | 70.00 | 100.0 | 32 | yes | 91.00 | 98.7 |
| 6 | 46.50 | 100.0 | 48 | yes | 92.50 | 91.7 |
| 7 | 77.75 | 100.0 | 48 | yes | 92.25 | 97.0 |
| 8 | 72.50 | 100.0 | 48 | yes | 93.25 | 98.0 |

The review shortens: no. More is left: yes. Rule B: with rehearsal at intervals the first family does not become cheaper to restore, and more of it survives a gap.

## B3. Smaller steps on the new task

| rule | rate | updates | first probe row with the unrelated task at 80% | first-family probe there (%) | last probe row with the first family at half of update 0 or more | unrelated held there (%) | reduction at the end | unrelated held at the end |
|---|---|---|---|---|---|---|---|---|
| S | 0.001 | 4096 | 384 | 0.00 | 0 | 0.0 | 0.0 | 100.0 |
| S | 0.0003 | 13824 | 960 | 0.00 | 16 | 0.0 | 0.0 | 100.0 |
| S | 0.0001 | 40960 | 2304 | 0.00 | 96 | 0.0 | 0.0 | 100.0 |
| S | **smaller steps leave more** | no | | | | | | |
| B | 0.001 | 4096 | 224 | 0.00 | 0 | 0.0 | 0.0 | 100.0 |
| B | 0.0003 | 13824 | 448 | 0.25 | 16 | 0.0 | 0.0 | 100.0 |
| B | 0.0001 | 40960 | 1024 | 0.25 | 48 | 0.0 | 0.0 | 100.0 |
| B | **smaller steps leave more** | no | | | | | | |
