# E16 drift seeds: gates by panel version, and a strict held-out re-score (answer to the 10 November source audit)

## 1. Gates by panel version (from the receipts)

| Seed | Panel | Ratios to reference (mod-97 · held pairs · mod-73) | Family gate | `+-`/`-+` ratio | Hard-pair gate | Worked-answer ratio | Worked-answer gate |
|---:|---|---|---|---:|---|---:|---|
| 73129 | original (pre-fix) | 0.97 · 0.86 · 0.99 | pass | 0.81 | pass | 0.83 | pass |
| 93683 | original (pre-fix) | 0.93 · 0.87 · 0.96 | pass | 0.79 | fail | 0.84 | pass |
| 116089 | original (pre-fix) | 0.84 · 0.86 · 0.97 | pass | 0.83 | pass | 0.83 | pass |
| 200003 | corrected | 0.93 · 0.86 · 0.96 | pass | 0.78 | fail | 0.84 | pass |
| 300007 | corrected | 0.96 · 0.87 · 0.99 | pass | 0.79 | fail | 0.85 | pass |
| 400009 | corrected | 0.90 · 0.69 · 0.96 | **fail** | 0.52 | fail | 0.65 | fail |
| 500011 | corrected | 0.96 · 0.89 · 1.01 | pass | 0.86 | pass | 0.86 | pass |
| 600013 | corrected | 0.77 · 0.71 · 0.91 | **fail** | 0.49 | fail | 0.65 | fail |
| 700027 | corrected | 0.94 · 0.87 · 0.97 | pass | 0.79 | fail | 0.85 | pass |
| 800011 | corrected | 0.87 · 0.83 · 0.95 | pass | 0.74 | fail | 0.79 | fail |
| 900007 | corrected | 0.93 · 0.83 · 0.97 | pass | 0.76 | fail | 0.80 | fail |
| 1000003 | corrected | 0.81 · 0.83 · 0.94 | pass | 0.77 | fail | 0.80 | pass |
| 1100009 | corrected | 0.91 · 0.84 · 0.96 | pass | 0.74 | fail | 0.83 | pass |
| 1200007 | corrected | 0.91 · 0.81 · 0.95 | pass | 0.64 | fail | 0.78 | fail |
| 1300021 | corrected | 0.95 · 0.84 · 0.97 | pass | 0.73 | fail | 0.83 | pass |
| 1400017 | corrected | 0.86 · 0.90 · 0.96 | pass | 0.85 | pass | 0.88 | pass |
| 1500007 | corrected | 0.94 · 0.87 · 1.00 | pass | 0.80 | fail | 0.85 | pass |
| 1600033 | corrected | 0.80 · 0.81 · 0.92 | pass | 0.69 | fail | 0.78 | fail |

| Stratum | Seeds | Family gate | Hard-pair gate | Worked-answer gate |
|---|---:|---:|---:|---:|
| corrected panels | 15 | 13 | 2 | 9 |
| original panels (pre-fix) | 3 | 3 | 2 | 3 |
| all, mixed (as first filed) | 18 | 16 | 4 | 12 |

## 2. Registered against strict held panels, final weights, k = 12, no score-time store

Strict: a held pair is dropped if the operation is commutative and its reversal is taught, or (mod 97) if it is supervised as a step of a taught composition.

| Seed | Learner mod-97: receipt (with store) | registered, no store | strict, no store | Reference mod-97: registered | strict | Ratio registered | Ratio strict | mod-73 ratio registered | strict |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 73129 | final weights not on file | | | | | | | | |
| 93683 | final weights not on file | | | | | | | | |
| 116089 | final weights not on file | | | | | | | | |
| 200003 | 88.2 | 88.2 | 88.5 | 94.6 | 93.8 | 0.932 | 0.944 | 0.958 | 0.959 |
| 300007 | 89.6 | 89.6 | 90.1 | 93.1 | 94.1 | 0.962 | 0.957 | 0.992 | 0.998 |
| 400009 | 86.6 | 86.6 | 86.0 | 95.7 | 96.1 | 0.905 | 0.895 | 0.962 | 0.968 |
| 500011 | 87.6 | 87.6 | 86.7 | 91.8 | 92.3 | 0.955 | 0.939 | 1.009 | 0.997 |
| 600013 | 72.8 | 72.8 | 74.5 | 94.1 | 93.9 | 0.774 | 0.794 | 0.914 | 0.931 |
| 700027 | 89.9 | 89.9 | 89.9 | 95.7 | 95.9 | 0.939 | 0.937 | 0.970 | 0.980 |
| 800011 | 82.1 | 82.1 | 83.7 | 94.9 | 94.8 | 0.866 | 0.883 | 0.949 | 0.956 |
| 900007 | 86.2 | 86.2 | 87.2 | 93.1 | 93.7 | 0.927 | 0.931 | 0.971 | 0.969 |
| 1000003 | 76.4 | 76.4 | 75.8 | 94.0 | 94.4 | 0.813 | 0.803 | 0.939 | 0.964 |
| 1100009 | 84.9 | 84.9 | 85.3 | 93.7 | 95.4 | 0.907 | 0.894 | 0.965 | 0.975 |
| 1200007 | 85.8 | 85.8 | 85.4 | 94.4 | 94.5 | 0.908 | 0.903 | 0.955 | 0.960 |
| 1300021 | 90.4 | 90.4 | 89.6 | 94.8 | 94.2 | 0.954 | 0.952 | 0.971 | 0.988 |
| 1400017 | 80.8 | 80.8 | 81.0 | 94.0 | 93.0 | 0.860 | 0.871 | 0.962 | 0.978 |
| 1500007 | 89.1 | 89.1 | 90.8 | 94.8 | 94.8 | 0.939 | 0.958 | 0.999 | 1.018 |
| 1600033 | 77.3 | 77.3 | 76.1 | 96.1 | 95.4 | 0.805 | 0.798 | 0.921 | 0.913 |
