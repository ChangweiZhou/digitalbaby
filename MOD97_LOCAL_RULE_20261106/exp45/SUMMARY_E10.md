
## E10 ring with innate operations: final-answer exact (%) at the end of life, three seeds

| Arm | held pairs (E7: +− −+ >< <> ; confounded: +− −+) | within | single ops after life (mean of 4) |
|---|---|---|---|
| ring, single-step development only (zero-shot) | 13 12 87 90 · 3 4 16 15 · 6 8 39 44 | 49 · 11 · 24 | 55 · 10 · 26 |
| ring, life on 12 pairs, sign rule | 36 34 64 64 · 3 3 24 18 · 10 11 35 34 | 42 · 10 · 18 | 53 · 14 · 25 |
| ring, life on 12 pairs, backprop | 92 92 98 100 · 10 12 25 28 · 25 28 63 63 | 97 · 20 · 43 | 97 · 23 · 50 |
| ring, life on confounded 2 pairs, sign rule | 61 60 · 10 7 · 13 14 | 58 · 10 · 12 | 43 · 8 · 12 |
| ring, life on confounded 2 pairs, backprop | 100 100 · 46 42 · 32 32 | 100 · 40 · 30 | 96 · 33 · 32 |
| flat ablation, life on 12 pairs, backprop | 2 3 6 9 · 4 2 12 14 · 2 3 9 9 | 6 · 5 · 4 | 5 · 10 · 6 |
| flat ablation, life on confounded 2 pairs, backprop | 2 2 · 4 3 · 2 3 | 3 · 6 · 5 | 2 · 4 · 3 |

### E10 predictions (held-pair mean)

- P10.1 ring zero-shot ≥90: [51, 10, 24]
- P10.2 ring confounded ≥90 cross: [60, 8, 14] (sign), [100, 44, 32] (bp)
- P10.3 flat confounded <30: [2, 4, 3]; flat 12-pair ≥70: [5, 8, 6]
- P10.4 ring sign within 5 of bp: [50, 12, 22] vs [95, 19, 45]

## E10 ring with innate operations_d32768: final-answer exact (%) at the end of life, three seeds

| Arm | held pairs (E7: +− −+ >< <> ; confounded: +− −+) | within | single ops after life (mean of 4) |
|---|---|---|---|
| ring, single-step development only (zero-shot) | 100 100 100 100 · 46 42 84 88 · 86 88 94 95 | 100 · 64 · 90 | 100 · 78 · 97 |
| ring, life on 12 pairs, sign rule | 100 100 100 100 · 14 15 47 50 · 24 22 54 57 | 100 · 32 · 37 | 100 · 37 · 46 |
| ring, life on 12 pairs, backprop | 100 100 100 100 · 36 33 88 92 · 60 62 93 95 | 100 · 65 · 77 | 100 · 75 · 87 |
| ring, life on confounded 2 pairs, sign rule | 100 100 · 22 20 · 24 23 | 100 · 22 · 24 | 58 · 22 · 25 |
| ring, life on confounded 2 pairs, backprop | 100 100 · 67 62 · 71 68 | 100 · 66 · 68 | 99 · 64 · 84 |
| flat ablation, life on 12 pairs, backprop | 5 4 10 9 · 5 4 15 28 · 3 4 10 10 | 8 · 13 · 6 | 7 · 21 · 8 |
| flat ablation, life on confounded 2 pairs, backprop | 7 6 · 5 5 · 4 6 | 7 · 5 · 6 | 4 · 5 · 5 |

### E10 predictions (held-pair mean)

- P10.1 ring zero-shot ≥90: [100, 65, 91]
- P10.2 ring confounded ≥90 cross: [100, 21, 24] (sign), [100, 64, 70] (bp)
- P10.3 flat confounded <30: [6, 5, 5]; flat 12-pair ≥70: [7, 13, 7]
- P10.4 ring sign within 5 of bp: [100, 32, 39] vs [100, 62, 77]