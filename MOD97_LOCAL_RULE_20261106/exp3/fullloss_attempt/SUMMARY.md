## Development births: sealed held exact by k on mod 97 (final), three seeds

| Arch | params | k=0 | k=2 | k=4 | k=6 | k=8 | k=10 | k=12 | k=14 | k=16 | k=20 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| win | 62,097 | 2 · 1 · 1 | 2 · 0 · 2 | 1 · 1 · 1 | 1 · 1 · 1 | 2 · 1 · 1 | 2 · 1 · 1 | 2 · 1 · 1 | 1 · 1 · 1 | 2 · 1 · 1 | 2 · 1 · 1 |
| rnn | 78,609 | 2 · 0 · 1 | 1 · 0 · 1 | 0 · 0 · 1 | 0 · 0 · 1 | 0 · 0 · 1 | 1 · 0 · 1 | 0 · 0 · 1 | 0 · 0 · 1 | 1 · 0 · 1 | 0 · 0 · 1 |
| gru | 142,737 | 94 · 88 · 92 | 97 · 89 · 94 | 95 · 91 · 93 | 98 · 90 · 94 | 97 · 90 · 93 | 97 · 91 · 94 | 97 · 90 · 93 | 95 · 89 · 91 | 90 · 84 · 91 | 71 · 70 · 88 |

## Life on held-out mod 73, 25% taught: sealed held exact by k (final), three seeds

| Arch | Arm | k=0 | k=2 | k=4 | k=6 | k=8 | k=10 | k=12 | k=14 | k=16 | k=20 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| win | warm start from developed birth, within-step sign rule | 2 · 1 · 0 | 2 · 2 · 0 | 2 · 1 · 0 | 2 · 1 · 1 | 2 · 2 · 1 | 2 · 2 · 1 | 2 · 1 · 1 | 2 · 1 · 1 | 2 · 2 · 1 | 2 · 2 · 1 |
| win | random birth, within-step sign rule | 2 · 2 · 1 | 2 · 2 · 0 | 2 · 2 · 0 | 2 · 2 · 0 | 2 · 1 · 1 | 2 · 1 · 1 | 2 · 1 · 1 | 2 · 2 · 1 | 2 · 1 · 1 | 2 · 2 · 0 |
| win | random birth, backprop with BPTT | 1 · 3 · 6 | 1 · 3 · 4 | 1 · 4 · 5 | 2 · 1 · 0 | 2 · 0 · 0 | 2 · 1 · 0 | 2 · 0 · 0 | 2 · 1 · 0 | 2 · 1 · 0 | 2 · 1 · 0 |
| rnn | warm start from developed birth, within-step sign rule | 2 · 1 · 1 | 2 · 2 · 1 | 2 · 1 · 1 | 2 · 1 · 1 | 2 · 1 · 0 | 2 · 1 · 1 | 2 · 1 · 1 | 2 · 1 · 0 | 2 · 1 · 0 | 2 · 1 · 0 |
| rnn | random birth, within-step sign rule | 2 · 1 · 1 | 2 · 1 · 1 | 2 · 1 · 1 | 2 · 1 · 0 | 2 · 1 · 1 | 2 · 1 · 1 | 2 · 1 · 0 | 2 · 1 · 0 | 2 · 1 · 1 | 2 · 1 · 1 |
| rnn | random birth, backprop with BPTT | 81 · 70 · 52 | 85 · 72 · 59 | 85 · 73 · 57 | 85 · 71 · 55 | 82 · 71 · 54 | 80 · 65 · 50 | 78 · 62 · 50 | 69 · 62 · 43 | 55 · 58 · 38 | 39 · 52 · 30 |
| gru | warm start from developed birth, within-step sign rule | 2 · 2 · 1 | 2 · 2 · 1 | 2 · 1 · 1 | 2 · 2 · 1 | 2 · 1 · 1 | 2 · 1 · 1 | 2 · 1 · 1 | 2 · 2 · 0 | 2 · 2 · 1 | 2 · 2 · 1 |
| gru | random birth, within-step sign rule | 2 · 2 · 1 | 2 · 2 · 0 | 2 · 2 · 0 | 2 · 1 · 0 | 2 · 1 · 1 | 2 · 1 · 1 | 2 · 2 · 0 | 2 · 1 · 0 | 2 · 2 · 1 | 2 · 2 · 0 |
| gru | random birth, backprop with BPTT | 94 · 83 · 88 | 95 · 83 · 89 | 96 · 85 · 90 | 95 · 84 · 91 | 95 · 86 · 91 | 95 · 87 · 92 | 92 · 84 · 90 | 86 · 79 · 87 | 76 · 72 · 90 | 44 · 48 · 87 |

Shortcut floors on the sealed panel: frequency 1.0%, second_operand_only 0.0%, no_wrap 49.3%, exact_lookup 0.0%

## Predictions

- P1 window net at k≥6 (first operand out of window): dev-birth life 2%, backprop life 1% (expect floor ≈ no-wrap half or below)
- P2 after development, held at k≤12: gru W 2% (need ≥80 at birth; see dev table), rnn at k≥10 1% (expect <50)
- P3 within-step local rule: gru warm k≤12 2% (need ≥70); gru random k≥6 1% (expect <30); rnn random k≥6 1%
- P4 extrapolation k 14-20: gru W 2% (need ≥50), gru B 74%, rnn W 1%, win W 2%