# The output end on tissue (vi) and on the tissue with no memory (written by outq_tables.py from outq/*.json)

65,536 bytes; inputs: unit v sees the cells of the neighbourhood of the v-th output-line cell ('own'); the reviewer's outq.py rules D (eta 1/64) and Q (eta 1/16, z1). Share right in the last quarter, %.

| Seed | ref_1 | ref_2 | vi D input alone | vi Q input alone | vi D with copies 2,4 | vi Q with copies 2,4 | none D input alone | none Q input alone | none D with copies 2,4 | none Q with copies 2,4 |
|---|---|---|---|---|---|---|---|---|---|---|
| 73129 | 10.55 | 12.34 | 8.36 | 8.10 | 8.93 | 8.33 | 7.69 | 7.62 | 10.88 | 8.79 |
| 93683 | 10.62 | 12.14 | 6.65 | 5.71 | 6.88 | 6.16 | 9.93 | 9.91 | 13.43 | 10.44 |
| 116089 | 10.61 | 12.20 | 8.92 | 8.27 | 9.30 | 8.56 | 10.93 | 11.08 | 14.37 | 13.18 |

## Over the seeds run: smallest / median / largest

| Tissue | inputs | rule | seeds | smallest | median | largest |
|---|---|---|---|---|---|---|
| vi | input alone | D | 3 | 6.65 | 8.36 | 8.92 |
| vi | input alone | Q | 3 | 5.71 | 8.10 | 8.27 |
| vi | with copies 2,4 | D | 3 | 6.88 | 8.93 | 9.30 |
| vi | with copies 2,4 | Q | 3 | 6.16 | 8.33 | 8.56 |
| none | input alone | D | 3 | 7.69 | 9.93 | 10.93 |
| none | input alone | Q | 3 | 7.62 | 9.91 | 11.08 |
| none | with copies 2,4 | D | 3 | 10.88 | 13.43 | 14.37 |
| none | with copies 2,4 | Q | 3 | 8.79 | 10.44 | 13.18 |

## Quarters (%, first to last)

| Tissue | Seed | inputs | D quarters | Q quarters |
|---|---|---|---|---|
| vi | 73129 | input alone | 6.18 / 8.19 / 7.81 / 8.36 | 5.00 / 7.44 / 7.57 / 8.10 |
| vi | 73129 | with copies 2,4 | 5.98 / 8.22 / 8.32 / 8.93 | 5.06 / 7.59 / 7.98 / 8.33 |
| vi | 93683 | input alone | 6.76 / 7.40 / 7.90 / 6.65 | 3.89 / 5.82 / 6.04 / 5.71 |
| vi | 93683 | with copies 2,4 | 6.66 / 7.56 / 8.22 / 6.88 | 3.66 / 6.08 / 6.39 / 6.16 |
| vi | 116089 | input alone | 6.09 / 8.94 / 8.78 / 8.92 | 4.71 / 6.63 / 8.76 / 8.27 |
| vi | 116089 | with copies 2,4 | 5.82 / 9.23 / 9.12 / 9.30 | 4.44 / 6.41 / 7.24 / 8.56 |
| none | 73129 | input alone | 6.98 / 7.29 / 7.30 / 7.69 | 5.68 / 7.29 / 7.29 / 7.62 |
| none | 73129 | with copies 2,4 | 8.18 / 10.05 / 10.47 / 10.88 | 4.99 / 7.40 / 8.72 / 8.79 |
| none | 93683 | input alone | 7.23 / 7.82 / 8.72 / 9.93 | 3.85 / 7.07 / 8.47 / 9.91 |
| none | 93683 | with copies 2,4 | 9.66 / 11.93 / 10.73 / 13.43 | 4.37 / 9.64 / 8.47 / 10.44 |
| none | 116089 | input alone | 5.63 / 8.61 / 10.28 / 10.93 | 5.19 / 7.91 / 10.39 / 11.08 |
| none | 116089 | with copies 2,4 | 12.10 / 13.57 / 14.15 / 14.37 | 6.35 / 12.51 / 13.09 / 13.18 |
