## Development births: sealed held exact by k on mod 97 (final), three seeds

| Arch | params | k=0 | k=2 | k=4 | k=6 | k=8 | k=10 | k=12 | k=14 | k=16 | k=20 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| win | 62,097 | 54 · 0 · 69 | 49 · 1 · 65 | 31 · 0 · 37 | 1 · 0 · 1 | 1 · 0 · 1 | 1 · 0 · 1 | 1 · 1 · 2 | 1 · 0 · 2 | 1 · 1 · 1 | 1 · 0 · 2 |
| rnn | 78,609 | 67 · 72 · 64 | 75 · 85 · 69 | 77 · 87 · 72 | 71 · 86 · 70 | 65 · 81 · 68 | 62 · 78 · 66 | 56 · 73 · 65 | 50 · 68 · 61 | 45 · 61 · 59 | 34 · 48 · 49 |
| gru | 142,737 | 84 · 88 · 86 | 88 · 93 · 91 | 89 · 93 · 92 | 88 · 94 · 90 | 88 · 93 · 90 | 88 · 93 · 90 | 87 · 93 · 89 | 86 · 91 · 88 | 86 · 88 · 87 | 84 · 84 · 82 |

## Life on held-out mod 73, 25% taught: sealed held exact by k (final), three seeds

| Arch | Arm | k=0 | k=2 | k=4 | k=6 | k=8 | k=10 | k=12 | k=14 | k=16 | k=20 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| win | warm, sign rule, no credit through time | 9 · 2 · 10 | 5 · 2 · 7 | 6 · 1 · 4 | 1 · 1 · 0 | 1 · 2 · 1 | 1 · 1 · 0 | 1 · 1 · 0 | 1 · 1 · 0 | 1 · 1 · 0 | 1 · 2 · 0 |
| win | random, sign rule, no credit through time | 2 · 2 · 1 | 1 · 2 · 1 | 2 · 1 · 1 | 1 · 1 · 0 | 1 · 1 · 0 | 1 · 1 · 1 | 1 · 1 · 1 | 1 · 1 · 0 | 1 · 1 · 0 | 1 · 0 · 1 |
| win | random, backprop with BPTT | 14 · 11 · 6 | 8 · 8 · 4 | 6 · 6 · 3 | 0 · 0 · 0 | 0 · 0 · 0 | 0 · 0 · 0 | 0 · 0 · 0 | 0 · 0 · 0 | 0 · 0 · 0 | 0 · 0 · 0 |
| rnn | warm, sign rule, no credit through time | 31 · 7 · 6 | 22 · 9 · 9 | 25 · 8 · 7 | 27 · 8 · 8 | 27 · 7 · 8 | 24 · 7 · 7 | 23 · 7 · 7 | 22 · 6 · 6 | 19 · 7 · 6 | 15 · 7 · 7 |
| rnn | warm, sign rule with credit through time | 1 · 52 · 49 | 1 · 66 · 64 | 1 · 70 · 68 | 1 · 69 · 73 | 1 · 67 · 71 | 1 · 64 · 67 | 1 · 58 · 64 | 1 · 54 · 61 | 1 · 50 · 53 | 1 · 38 · 42 |
| rnn | random, sign rule, no credit through time | 1 · 1 · 1 | 1 · 2 · 1 | 1 · 2 · 1 | 0 · 1 · 1 | 0 · 0 · 1 | 0 · 1 · 1 | 0 · 1 · 1 | 0 · 1 · 1 | 0 · 1 · 1 | 0 · 1 · 1 |
| rnn | random, backprop with BPTT | 33 · 72 · 73 | 35 · 75 · 78 | 32 · 74 · 81 | 33 · 73 · 77 | 31 · 72 · 77 | 31 · 70 · 76 | 29 · 66 · 74 | 29 · 62 · 72 | 26 · 58 · 70 | 26 · 47 · 65 |
| gru | warm, sign rule, no credit through time | 13 · 4 · 7 | 11 · 5 · 9 | 11 · 5 · 9 | 12 · 6 · 8 | 11 · 7 · 8 | 10 · 5 · 8 | 11 · 6 · 7 | 11 · 4 · 6 | 10 · 4 · 5 | 7 · 3 · 6 |
| gru | warm, sign rule with eligibility traces (local in time) | 14 · 2 · 3 | 12 · 3 · 5 | 12 · 2 · 4 | 13 · 2 · 4 | 13 · 1 · 4 | 11 · 2 · 3 | 12 · 2 · 4 | 12 · 2 · 4 | 12 · 2 · 4 | 13 · 2 · 3 |
| gru | warm, true gradient, no credit through time | 63 · 60 · 59 | 64 · 61 · 64 | 64 · 66 · 66 | 66 · 68 · 66 | 69 · 67 · 65 | 65 · 64 · 62 | 61 · 58 · 62 | 52 · 53 · 57 | 43 · 50 · 54 | 29 · 40 · 34 |
| gru | warm, sign rule with credit through time | 74 · 70 · 68 | 82 · 80 · 79 | 86 · 83 · 82 | 85 · 81 · 83 | 83 · 82 · 81 | 82 · 80 · 80 | 81 · 78 · 75 | 81 · 78 · 70 | 78 · 75 · 65 | 71 · 71 · 49 |
| gru | warm, backprop with BPTT | 86 · 91 · 88 | 87 · 91 · 88 | 87 · 90 · 87 | 87 · 92 · 87 | 88 · 91 · 87 | 88 · 90 · 87 | 86 · 91 · 86 | 86 · 91 · 86 | 83 · 91 · 85 | 82 · 91 · 84 |
| gru | random, sign rule, no credit through time | 1 · 1 · 1 | 2 · 2 · 2 | 2 · 2 · 1 | 0 · 1 · 2 | 1 · 1 · 2 | 1 · 2 · 3 | 1 · 1 · 2 | 1 · 2 · 3 | 1 · 2 · 2 | 1 · 2 · 1 |
| gru | random, sign rule with eligibility traces | 1 · 1 · 1 | 1 · 1 · 0 | 2 · 2 · 1 | 0 · 1 · 1 | 1 · 1 · 0 | 1 · 2 · 1 | 1 · 2 · 1 | 1 · 2 · 1 | 1 · 1 · 1 | 1 · 2 · 1 |
| gru | random, backprop with BPTT | 85 · 79 · 87 | 88 · 83 · 87 | 90 · 86 · 87 | 89 · 86 · 87 | 90 · 85 · 87 | 89 · 85 · 86 | 87 · 85 · 85 | 87 · 85 · 85 | 86 · 84 · 82 | 85 · 83 · 80 |

Shortcut floors on the sealed panel: frequency 0.7%, second_operand_only 0.0%, no_wrap 51.0%, exact_lookup 0.0%

## Predictions

- P1 window net at k≥6 (first operand out of window): dev-birth life 1%, backprop life 0% (expect floor ≈ no-wrap half or below)
- P2 after development, held at k≤12: gru W 8% (need ≥80 at birth; see dev table), rnn at k≥10 12% (expect <50)
- P3 within-step local rule: gru warm k≤12 8% (need ≥70); gru random k≥6 2% (expect <30); rnn random k≥6 1%
- P4 extrapolation k 14-20: gru W 6% (need ≥50), gru B 84%, rnn W 11%, win W 1%