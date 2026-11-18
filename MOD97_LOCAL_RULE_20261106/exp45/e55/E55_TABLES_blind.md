# E55, first part, seed 73129: the runs of the arm(s) blind

Read by `python3 e55_read.py blind` from the receipts.

| arm | task | rule | first family | largest d(g) | largest held-out differ | reduction row 0 / 16,384 | learned from row | never-taught at 16,384 | with the reserve shut | largest reserve state on the first family | largest output change | 40 letters differ at 16,384 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `blind` | unrelated | S | kept exactly | 0 | 0 | 92.40 / 92.40 | 8192 | 99.0 | 0.0 | 1.31e-05 | 0.00169 | 0 |
| `blind` | unrelated | B | kept on items of up to 22 positions and not on items of 50 | 0 | 0 | 95.25 / 95.25 | 2048 | 100.0 | 0.0 | 0.00121 | 0.2 | 278 |
| `blind` | conflict | S | kept exactly | 0 | 0 | 92.40 / 92.40 | not learned | 69.4 | 34.9 | 3.33e-06 | 9.87e-05 | 0 |
| `blind` | conflict | B | kept exactly | 0 | 0 | 95.25 / 95.25 | not learned | 63.2 | 28.9 | 2.95e-05 | 0.000349 | 0 |

Notes (from the reviewer's verification of the arm `blind`, `E55_BLIND_VERIFICATION_20261118.md` at 77e65eba, findings 2, 3 and 5):

- The eigenvalue column is the largest magnitude of an eigenvalue of 0.5 I + 0.25 Wn, Wn the candidate rows of the cells in reserve from the cells in reserve. It is above 1 at row 16,384 in all four runs of the arm `blind`. The construction leaves the weights among the cells in reserve free, so that nothing bounds the state of the reserve on longer items or after more updates; *kept exactly* is an observation on items of up to 22 positions.
- The 40-letter counts, the small states and the output changes depend on the arithmetic as well as on the weights: at row 4,096 of `unrelated` under back-propagation, 45 answers at 40 letters differ in the receipt, 15 in double precision and 142 in single precision in the reviewer's numpy. The outcomes of the brief's rules are the same in each.
- The outputs move by more than the smallest lead without a changed answer on the scored panels (largest change 1.69e-3 against a smallest lead of 1.78e-4 under the sweep; 0.200 against 2.33e-2 under back-propagation), so that *kept exactly* is not the consequence of a bound.

### unrelated, rule S, arm `blind` (`e55_unrelated_blind_S_73129.json`)

| updates | reduction | d(g) of 31,684 | held-out differ of 1,200 | 40 letters differ of 300 | never-taught | never-taught, reserve shut | new task: cells open at some position | new task: open by > 0.1 in the mean | new task: largest 1 − z | first family: largest 1 − z | first family: cells open at some position | first family: largest reserve state | first family: largest added unit | largest output change | eigenvalue | 40 letters: right of 300 | 40 letters: largest reserve state | 40 letters: largest 1 − z |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 92.40 | 0 | 0 | 0 | 0.0 | 0.0 | 128 | 128 | 0.553 | 0.5 | 128 | 2.43e-07 | 2.43e-07 | 0 | 0.605 | 282 | 2.43e-07 | 0.5 |
| 16 | 92.40 | 0 | 0 | 0 | 0.0 | 0.0 | 128 | 128 | 0.593 | 0.5 | 128 | 2.92e-07 | 3.27e-07 | 1.72e-05 | 0.809 | 282 | 2.9e-07 | 0.5 |
| 32 | 92.40 | 0 | 0 | 0 | 0.0 | 0.0 | 128 | 128 | 0.735 | 0.5 | 128 | 3.63e-07 | 4.68e-07 | 1.86e-05 | 0.771 | 282 | 4.29e-07 | 0.5 |
| 64 | 92.40 | 0 | 0 | 0 | 0.0 | 0.0 | 128 | 128 | 0.955 | 0.5 | 128 | 4.83e-07 | 4.56e-07 | 1.72e-05 | 0.689 | 282 | 4.75e-07 | 0.5 |
| 128 | 92.40 | 0 | 0 | 0 | 0.3 | 0.0 | 128 | 128 | 0.99 | 0.5 | 128 | 5.93e-07 | 5.35e-07 | 1.72e-05 | 0.710 | 282 | 5.99e-07 | 0.5 |
| 256 | 92.40 | 0 | 0 | 0 | 0.0 | 0.0 | 128 | 128 | 0.991 | 0.5 | 128 | 5.8e-07 | 5.27e-07 | 1.92e-05 | 0.684 | 282 | 5.73e-07 | 0.5 |
| 512 | 92.40 | 0 | 0 | 0 | 0.0 | 0.0 | 128 | 128 | 0.991 | 0.5 | 128 | 4.96e-07 | 6.4e-07 | 1.62e-05 | 0.776 | 282 | 5.68e-07 | 0.5 |
| 1024 | 92.40 | 0 | 0 | 0 | 0.3 | 0.0 | 128 | 128 | 1 | 0.5 | 128 | 5e-06 | 1.44e-05 | 7.58e-05 | 1.094 | 282 | 6.04e-05 | 0.5 |
| 2048 | 92.40 | 0 | 0 | 0 | 5.0 | 0.0 | 128 | 128 | 1 | 0.5 | 128 | 3.75e-06 | 1.62e-05 | 0.000147 | 0.975 | 282 | 7.23e-06 | 0.5 |
| 4096 | 92.40 | 0 | 0 | 0 | 71.7 | 0.0 | 128 | 128 | 1 | 0.5 | 128 | 5.87e-06 | 3.56e-05 | 0.000498 | 1.060 | 282 | 4.63e-05 | 0.5 |
| 8192 | 92.40 | 0 | 0 | 0 | 96.3 | 0.0 | 128 | 128 | 1 | 0.5 | 128 | 8.63e-06 | 7.08e-05 | 0.000805 | 1.112 | 282 | 0.000152 | 0.5 |
| 16384 | 92.40 | 0 | 0 | 0 | 99.0 | 0.0 | 128 | 128 | 1 | 0.5 | 128 | 1.31e-05 | 9.68e-05 | 0.00169 | 1.098 | 282 | 0.000172 | 0.5 |

E51's rows of the same task and rule (reduction; never-taught; never-taught with the reserve shut):

| updates | `keyed` | `open` | `rehearse` |
|---|---|---|---|
| 0 | 92.4; 0.0; 0.0 | 92.4; 0.0; 0.0 | 92.4; 0.0; 0.0 |
| 16 | 92.4; 0.0; 0.0 | 0.0; 0.0; 0.0 | 11.5; 0.0; 0.0 |
| 32 | 92.4; 0.0; 0.0 | 0.0; 0.0; 0.0 | 20.3; 0.0; 0.0 |
| 64 | 92.4; 0.0; 0.0 | 0.0; 0.0; 0.0 | 35.3; 0.0; 0.0 |
| 128 | 92.4; 0.0; 0.0 | 0.0; 0.0; 0.0 | 47.8; 0.0; 0.0 |
| 256 | 92.4; 0.0; 0.0 | 0.0; 15.3; 0.0 | 20.5; 1.0; 0.0 |
| 512 | 92.4; 43.0; 0.0 | 0.0; 99.0; 0.0 | 32.2; 85.3; 0.0 |
| 1024 | 92.4; 100.0; 0.0 | 0.0; 100.0; 0.0 | 58.9; 97.0; 0.0 |
| 2048 | 92.4; 100.0; 0.0 | 0.0; 99.7; 0.0 | 85.5; 94.7; 0.0 |
| 4096 | 92.4; 100.0; 0.0 | 0.0; 100.0; 0.7 | 90.0; 97.3; 0.0 |

### unrelated, rule B, arm `blind` (`e55_unrelated_blind_B_73129.json`)

| updates | reduction | d(g) of 31,684 | held-out differ of 1,200 | 40 letters differ of 300 | never-taught | never-taught, reserve shut | new task: cells open at some position | new task: open by > 0.1 in the mean | new task: largest 1 − z | first family: largest 1 − z | first family: cells open at some position | first family: largest reserve state | first family: largest added unit | largest output change | eigenvalue | 40 letters: right of 300 | 40 letters: largest reserve state | 40 letters: largest 1 − z |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 95.25 | 0 | 0 | 0 | 0.0 | 0.0 | 128 | 128 | 0.558 | 0.5 | 128 | 3.48e-07 | 3.25e-07 | 0 | 0.605 | 274 | 3.2e-07 | 0.5 |
| 16 | 95.25 | 0 | 0 | 0 | 0.0 | 0.0 | 128 | 128 | 0.654 | 0.5 | 128 | 4.6e-07 | 4.69e-07 | 2.1e-05 | 0.716 | 274 | 4.55e-07 | 0.5 |
| 32 | 95.25 | 0 | 0 | 0 | 0.0 | 0.0 | 128 | 128 | 0.886 | 0.5 | 128 | 7.82e-07 | 8.92e-07 | 1.91e-05 | 0.734 | 274 | 7.63e-07 | 0.5 |
| 64 | 95.25 | 0 | 0 | 0 | 0.0 | 0.0 | 128 | 128 | 0.986 | 0.5 | 128 | 9.57e-07 | 9.74e-07 | 1.72e-05 | 0.907 | 274 | 9.64e-07 | 0.5 |
| 128 | 95.25 | 0 | 0 | 0 | 0.3 | 0.0 | 128 | 128 | 0.99 | 0.5 | 128 | 1.07e-06 | 8.86e-07 | 1.53e-05 | 0.905 | 274 | 1.07e-06 | 0.5 |
| 256 | 95.25 | 0 | 0 | 0 | 0.0 | 0.0 | 128 | 128 | 0.988 | 0.5 | 128 | 1.22e-06 | 7.47e-07 | 1.81e-05 | 0.837 | 274 | 1.18e-06 | 0.5 |
| 512 | 95.25 | 0 | 0 | 0 | 0.0 | 0.0 | 128 | 128 | 0.996 | 0.5 | 128 | 1.87e-06 | 1.46e-06 | 1.72e-05 | 0.864 | 274 | 1.8e-06 | 0.5 |
| 1024 | 95.25 | 0 | 0 | 0 | 25.3 | 0.0 | 128 | 128 | 1 | 0.5 | 128 | 4.25e-05 | 0.000103 | 0.00126 | 1.193 | 274 | 0.00238 | 0.502 |
| 2048 | 95.25 | 0 | 0 | 21 | 99.7 | 0.0 | 128 | 128 | 1 | 0.5 | 128 | 0.000337 | 0.00146 | 0.018 | 1.324 | 264 | 0.297 | 0.875 |
| 4096 | 95.25 | 0 | 0 | 45 | 100.0 | 0.0 | 128 | 128 | 1 | 0.501 | 128 | 0.000443 | 0.00223 | 0.0319 | 1.342 | 238 | 0.332 | 0.913 |
| 8192 | 95.25 | 0 | 0 | 194 | 100.0 | 0.0 | 128 | 127 | 1 | 0.501 | 128 | 0.000468 | 0.00289 | 0.0518 | 1.385 | 101 | 0.952 | 0.982 |
| 16384 | 95.25 | 0 | 0 | 278 | 100.0 | 0.0 | 128 | 127 | 1 | 0.502 | 128 | 0.00121 | 0.00847 | 0.2 | 1.434 | 21 | 1 | 0.997 |

E51's rows of the same task and rule (reduction; never-taught; never-taught with the reserve shut):

| updates | `keyed` | `open` | `rehearse` |
|---|---|---|---|
| 0 | 95.3; 0.0; 0.0 | 95.3; 0.0; 0.0 | 95.3; 0.0; 0.0 |
| 16 | 95.3; 0.0; 0.0 | 0.0; 0.0; 0.0 | 38.4; 0.0; 0.0 |
| 32 | 95.3; 0.0; 0.0 | 0.0; 0.0; 0.0 | 61.8; 0.0; 0.0 |
| 64 | 95.3; 0.0; 0.0 | 0.0; 0.0; 0.0 | 72.8; 0.0; 0.0 |
| 128 | 95.3; 0.0; 0.0 | 0.0; 0.0; 0.0 | 81.4; 0.0; 0.0 |
| 256 | 95.3; 0.0; 0.0 | 0.0; 28.3; 1.3 | 76.0; 4.0; 3.7 |
| 512 | 95.3; 91.3; 0.0 | 0.0; 100.0; 3.3 | 80.4; 92.3; 3.0 |
| 1024 | 95.3; 100.0; 0.0 | 0.0; 100.0; 5.0 | 89.8; 99.0; 1.3 |
| 2048 | 95.3; 100.0; 0.0 | 0.0; 100.0; 6.0 | 92.8; 100.0; 1.3 |
| 4096 | 95.3; 100.0; 0.0 | 0.0; 100.0; 5.7 | 92.8; 100.0; 0.0 |

### conflict, rule S, arm `blind` (`e55_conflict_blind_S_73129.json`)

| updates | reduction | d(g) of 31,684 | held-out differ of 1,200 | 40 letters differ of 300 | never-taught | never-taught, reserve shut | new task: cells open at some position | new task: open by > 0.1 in the mean | new task: largest 1 − z | first family: largest 1 − z | first family: cells open at some position | first family: largest reserve state | first family: largest added unit | largest output change | eigenvalue | 40 letters: right of 300 | 40 letters: largest reserve state | 40 letters: largest 1 − z |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 92.40 | 0 | 0 | 0 | 19.0 | 19.0 | 128 | 128 | 0.519 | 0.5 | 128 | 2.43e-07 | 2.43e-07 | 0 | 0.605 | 282 | 2.43e-07 | 0.5 |
| 16 | 92.40 | 0 | 0 | 0 | 21.4 | 20.6 | 128 | 128 | 0.522 | 0.5 | 128 | 3.12e-07 | 2.66e-07 | 1.67e-05 | 0.859 | 282 | 3.01e-07 | 0.5 |
| 32 | 92.40 | 0 | 0 | 0 | 23.0 | 22.1 | 128 | 128 | 0.528 | 0.5 | 128 | 3.68e-07 | 2.72e-07 | 1.57e-05 | 0.933 | 282 | 3.59e-07 | 0.5 |
| 64 | 92.40 | 0 | 0 | 0 | 24.6 | 23.8 | 128 | 128 | 0.557 | 0.5 | 128 | 5.05e-07 | 2.48e-07 | 1.53e-05 | 0.930 | 282 | 2.95e-07 | 0.5 |
| 128 | 92.40 | 0 | 0 | 0 | 30.8 | 26.2 | 128 | 128 | 0.622 | 0.5 | 128 | 5.91e-07 | 3.63e-07 | 1.38e-05 | 0.992 | 282 | 4.04e-07 | 0.5 |
| 256 | 92.40 | 0 | 0 | 0 | 20.1 | 31.4 | 128 | 128 | 0.979 | 0.5 | 128 | 1.02e-06 | 8.91e-07 | 1.91e-05 | 1.066 | 282 | 2.8e-06 | 0.5 |
| 512 | 92.40 | 0 | 0 | 0 | 35.5 | 35.2 | 128 | 128 | 0.969 | 0.5 | 128 | 1.14e-06 | 1.66e-06 | 2.24e-05 | 1.062 | 282 | 5.17e-06 | 0.5 |
| 1024 | 92.40 | 0 | 0 | 0 | 45.5 | 35.2 | 128 | 128 | 0.929 | 0.5 | 128 | 1.42e-06 | 1.29e-06 | 2.34e-05 | 1.005 | 282 | 1.24e-06 | 0.5 |
| 2048 | 92.40 | 0 | 0 | 0 | 51.4 | 36.8 | 128 | 128 | 0.939 | 0.5 | 128 | 2.57e-06 | 1.6e-06 | 2.67e-05 | 0.950 | 282 | 1.99e-06 | 0.5 |
| 4096 | 92.40 | 0 | 0 | 0 | 58.6 | 37.4 | 128 | 128 | 0.984 | 0.5 | 128 | 2.63e-06 | 4.51e-06 | 5.44e-05 | 1.012 | 282 | 2.07e-06 | 0.5 |
| 8192 | 92.40 | 0 | 0 | 0 | 63.4 | 35.8 | 128 | 128 | 0.995 | 0.5 | 128 | 2.86e-06 | 8.91e-06 | 9.06e-05 | 1.039 | 282 | 8.6e-06 | 0.5 |
| 16384 | 92.40 | 0 | 0 | 0 | 69.4 | 34.9 | 128 | 128 | 0.997 | 0.5 | 128 | 3.33e-06 | 6.88e-06 | 9.87e-05 | 1.058 | 282 | 1.11e-05 | 0.5 |

E51's rows of the same task and rule (reduction; never-taught; never-taught with the reserve shut):

| updates | `keyed` | `open` | `rehearse` |
|---|---|---|---|
| 0 | 92.4; 18.0; 19.0 | 92.4; 19.0; 19.0 | 92.4; 19.0; 19.0 |
| 16 | 92.4; 30.0; 19.8 | 25.7; 31.4; 31.4 | 59.6; 19.9; 19.9 |
| 32 | 92.4; 34.9; 21.4 | 19.1; 30.8; 30.8 | 44.6; 31.5; 31.5 |
| 64 | 92.4; 40.2; 23.8 | 12.4; 33.6; 33.6 | 54.8; 28.1; 28.1 |
| 128 | 92.4; 48.9; 26.8 | 14.1; 52.8; 52.8 | 49.4; 35.9; 35.9 |
| 256 | 92.4; 62.4; 32.8 | 5.4; 64.5; 64.5 | 61.7; 46.8; 46.8 |
| 512 | 92.4; 77.0; 37.5 | 4.9; 75.8; 75.9 | 72.8; 57.2; 57.2 |
| 1024 | 92.4; 90.9; 39.6 | 3.0; 81.1; 80.4 | 70.3; 70.4; 70.1 |
| 2048 | 92.4; 95.2; 41.1 | 0.8; 87.2; 86.9 | 87.4; 84.4; 82.6 |
| 4096 | 92.4; 94.8; 41.0 | 1.3; 85.0; 83.0 | 88.3; 85.9; 83.0 |

### conflict, rule B, arm `blind` (`e55_conflict_blind_B_73129.json`)

| updates | reduction | d(g) of 31,684 | held-out differ of 1,200 | 40 letters differ of 300 | never-taught | never-taught, reserve shut | new task: cells open at some position | new task: open by > 0.1 in the mean | new task: largest 1 − z | first family: largest 1 − z | first family: cells open at some position | first family: largest reserve state | first family: largest added unit | largest output change | eigenvalue | 40 letters: right of 300 | 40 letters: largest reserve state | 40 letters: largest 1 − z |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 95.25 | 0 | 0 | 0 | 8.2 | 8.2 | 128 | 128 | 0.52 | 0.5 | 128 | 3.48e-07 | 3.25e-07 | 0 | 0.605 | 274 | 3.2e-07 | 0.5 |
| 16 | 95.25 | 0 | 0 | 0 | 10.8 | 10.1 | 128 | 128 | 0.555 | 0.5 | 128 | 4.75e-07 | 2.85e-07 | 1.53e-05 | 0.709 | 274 | 4.61e-07 | 0.5 |
| 32 | 95.25 | 0 | 0 | 0 | 16.5 | 12.6 | 128 | 128 | 0.608 | 0.5 | 128 | 5.53e-07 | 3.32e-07 | 1.91e-05 | 0.847 | 274 | 5.23e-07 | 0.5 |
| 64 | 95.25 | 0 | 0 | 0 | 21.2 | 15.1 | 128 | 128 | 0.687 | 0.5 | 128 | 1.09e-06 | 6.19e-07 | 1.43e-05 | 0.969 | 274 | 1.09e-06 | 0.5 |
| 128 | 95.25 | 0 | 0 | 0 | 18.1 | 17.2 | 128 | 128 | 0.803 | 0.5 | 128 | 1.47e-06 | 1.85e-06 | 3.72e-05 | 1.068 | 274 | 6.94e-06 | 0.5 |
| 256 | 95.25 | 0 | 0 | 0 | 24.6 | 20.2 | 128 | 128 | 0.857 | 0.5 | 128 | 1.42e-06 | 1.18e-06 | 2.86e-05 | 1.043 | 274 | 4.8e-06 | 0.5 |
| 512 | 95.25 | 0 | 0 | 0 | 37.1 | 24.6 | 128 | 124 | 0.981 | 0.5 | 128 | 2.25e-06 | 1.11e-06 | 2.57e-05 | 1.064 | 274 | 1.23e-05 | 0.5 |
| 1024 | 95.25 | 0 | 0 | 0 | 46.6 | 28.8 | 128 | 124 | 0.998 | 0.5 | 128 | 3.24e-06 | 1.13e-06 | 3.05e-05 | 1.103 | 274 | 2.47e-05 | 0.5 |
| 2048 | 95.25 | 0 | 0 | 0 | 52.8 | 31.0 | 128 | 120 | 0.998 | 0.5 | 128 | 1.46e-05 | 1.1e-05 | 0.000165 | 1.161 | 274 | 0.000263 | 0.5 |
| 4096 | 95.25 | 0 | 0 | 0 | 61.1 | 31.6 | 128 | 123 | 0.997 | 0.5 | 128 | 1.98e-05 | 2.12e-05 | 0.000257 | 1.170 | 274 | 0.000975 | 0.5 |
| 8192 | 95.25 | 0 | 0 | 0 | 61.5 | 31.9 | 128 | 116 | 0.999 | 0.5 | 128 | 1.55e-05 | 2.11e-05 | 0.000221 | 1.180 | 274 | 0.00145 | 0.502 |
| 16384 | 95.25 | 0 | 0 | 0 | 63.2 | 28.9 | 128 | 115 | 0.999 | 0.5 | 128 | 2.95e-05 | 5.55e-05 | 0.000349 | 1.197 | 274 | 0.00343 | 0.504 |

E51's rows of the same task and rule (reduction; never-taught; never-taught with the reserve shut):

| updates | `keyed` | `open` | `rehearse` |
|---|---|---|---|
| 0 | 95.3; 8.5; 8.2 | 95.3; 8.2; 8.2 | 95.3; 8.2; 8.2 |
| 16 | 95.3; 9.0; 10.0 | 25.8; 11.5; 11.5 | 46.6; 9.5; 9.5 |
| 32 | 95.3; 11.5; 10.9 | 19.5; 17.5; 17.5 | 51.6; 13.8; 13.8 |
| 64 | 95.3; 22.8; 13.4 | 12.7; 28.4; 28.4 | 65.5; 27.0; 27.0 |
| 128 | 95.3; 41.2; 18.1 | 4.5; 50.6; 50.6 | 70.9; 30.2; 30.2 |
| 256 | 95.3; 60.2; 24.0 | 1.2; 64.2; 63.9 | 72.2; 42.5; 42.4 |
| 512 | 95.3; 82.0; 32.2 | 0.5; 71.2; 69.1 | 78.2; 55.1; 54.0 |
| 1024 | 95.3; 91.8; 37.0 | 0.0; 85.4; 81.2 | 89.1; 75.6; 63.8 |
| 2048 | 95.3; 92.5; 38.1 | 0.0; 91.4; 84.9 | 94.5; 82.6; 66.9 |
| 4096 | 95.3; 96.9; 40.4 | 0.0; 94.9; 87.6 | 97.8; 90.9; 66.6 |

