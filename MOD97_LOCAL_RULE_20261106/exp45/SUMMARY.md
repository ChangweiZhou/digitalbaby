## E4 retention: sealed held exact (%) on each life task after each stage, three seeds


### A shared path alone, decay 1.0

| After stage | add97 | sub97 | add73 | sub73 | dev tasks (mean) | T1 taught |
|---|---|---|---|---|---|---|
| birth | 14 · 8 · 4 | 6 · 4 · 5 | 14 · 8 · 6 | 8 · 5 · 10 | 96 · 96 · 96 | 12 · 7 · 3 |
| 1 (add97) | 96 · 92 · 92 | 0 · 0 · 0 | 0 · 0 · 2 | 0 · 0 · 0 | 0 · 0 · 0 | 99 · 98 · 97 |
| 2 (sub97) | 0 · 0 · 0 | 93 · 95 · 94 | 0 · 0 · 0 | 0 · 0 · 0 | 0 · 0 · 0 | 0 · 0 · 0 |
| 3 (add73) | 0 · 0 · 1 | 0 · 0 · 0 | 93 · 92 · 87 | 0 · 0 · 0 | 0 · 0 · 1 | 0 · 1 · 0 |
| 4 (sub73) | 0 · 0 · 0 | 3 · 4 · 1 | 0 · 0 · 0 | 92 · 95 · 92 | 2 · 3 · 0 | 0 · 0 · 0 |

### A0 shared path alone, decay 0.1

| After stage | add97 | sub97 | add73 | sub73 | dev tasks (mean) | T1 taught |
|---|---|---|---|---|---|---|
| birth | 14 · 8 · 4 | 6 · 4 · 5 | 14 · 8 · 6 | 8 · 5 · 10 | 96 · 96 · 96 | 12 · 7 · 3 |
| 1 (add97) | 96 · 94 · 95 | 0 · 0 · 0 | 48 · 44 · 22 | 0 · 0 · 0 | 13 · 21 · 23 | 100 · 100 · 100 |
| 2 (sub97) | 0 · 2 · 0 | 95 · 95 · 96 | 0 · 0 · 0 | 3 · 13 · 7 | 13 · 24 · 23 | 0 · 2 · 0 |
| 3 (add73) | 24 · 28 · 17 | 0 · 2 · 2 | 91 · 90 · 93 | 1 · 1 · 1 | 14 · 18 · 14 | 26 · 26 · 19 |
| 4 (sub73) | 0 · 1 · 1 | 20 · 32 · 31 | 1 · 1 · 1 | 92 · 92 · 94 | 12 · 21 · 20 | 0 · 2 · 0 |

### B + episodic store, no replay

| After stage | add97 | sub97 | add73 | sub73 | dev tasks (mean) | T1 taught |
|---|---|---|---|---|---|---|
| birth | 14 · 8 · 4 | 6 · 4 · 5 | 14 · 8 · 6 | 8 · 5 · 10 | 96 · 96 · 96 | 12 · 7 · 3 |
| 1 (add97) | 96 · 92 · 92 | 0 · 0 · 0 | 0 · 0 · 2 | 0 · 0 · 0 | 0 · 0 · 0 | 100 · 100 · 100 |
| 2 (sub97) | 0 · 0 · 0 | 93 · 95 · 94 | 0 · 0 · 0 | 0 · 0 · 0 | 0 · 0 · 0 | 6 · 7 · 3 |
| 3 (add73) | 0 · 0 · 1 | 0 · 0 · 0 | 93 · 92 · 87 | 0 · 0 · 0 | 0 · 0 · 1 | 10 · 17 · 17 |
| 4 (sub73) | 0 · 0 · 0 | 3 · 4 · 1 | 0 · 0 · 0 | 92 · 95 · 92 | 2 · 3 · 0 | 12 · 7 · 9 |

### C + store + self-generated replay

| After stage | add97 | sub97 | add73 | sub73 | dev tasks (mean) | T1 taught |
|---|---|---|---|---|---|---|
| birth | 14 · 8 · 4 | 6 · 4 · 5 | 14 · 8 · 6 | 8 · 5 · 10 | 96 · 96 · 96 | 12 · 7 · 3 |
| 1 (add97) | 96 · 92 · 92 | 0 · 0 · 0 | 0 · 0 · 2 | 0 · 0 · 0 | 0 · 0 · 0 | 100 · 100 · 100 |
| 2 (sub97) | 92 · 92 · 86 | 90 · 94 · 91 | 9 · 9 · 0 | 2 · 0 · 0 | 4 · 3 · 0 | 100 · 100 · 100 |
| 3 (add73) | 95 · 92 · 86 | 83 · 84 · 78 | 91 · 91 · 87 | 0 · 0 · 11 | 0 · 0 · 12 | 100 · 100 · 100 |
| 4 (sub73) | 90 · 88 · 78 | 90 · 93 · 82 | 86 · 84 · 77 | 91 · 93 · 86 | 24 · 0 · 4 | 100 · 100 · 100 |

### C′ + store + replay with random answers

| After stage | add97 | sub97 | add73 | sub73 | dev tasks (mean) | T1 taught |
|---|---|---|---|---|---|---|
| birth | 14 · 8 · 4 | 6 · 4 · 5 | 14 · 8 · 6 | 8 · 5 · 10 | 96 · 96 · 96 | 12 · 7 · 3 |
| 1 (add97) | 96 · 92 · 92 | 0 · 0 · 0 | 0 · 0 · 2 | 0 · 0 · 0 | 0 · 0 · 0 | 100 · 100 · 100 |
| 2 (sub97) | 1 · 1 · 1 | 87 · 87 · 79 | 0 · 0 · 0 | 0 · 0 · 0 | 0 · 0 · 0 | 1 · 1 · 1 |
| 3 (add73) | 1 · 2 · 1 | 2 · 1 · 1 | 85 · 78 · 65 | 0 · 0 · 0 | 0 · 0 · 0 | 0 · 1 · 2 |
| 4 (sub73) | 0 · 1 · 1 | 1 · 3 · 1 | 0 · 1 · 1 | 85 · 78 · 68 | 1 · 0 · 2 | 1 · 1 · 1 |

### D backprop, joint ground truth (ceiling)

| After stage | add97 | sub97 | add73 | sub73 | dev tasks (mean) | T1 taught |
|---|---|---|---|---|---|---|
| birth | 14 · 8 · 4 | 6 · 4 · 5 | 14 · 8 · 6 | 8 · 5 · 10 | 96 · 96 · 96 | 12 · 7 · 3 |
| 1 (add97) | 97 · 95 · 95 | 0 · 1 · 0 | 3 · 15 · 0 | 0 · 1 · 0 | 4 · 15 · 2 | 100 · 100 · 100 |
| 2 (sub97) | 94 · 96 · 96 | 94 · 94 · 97 | 1 · 2 · 0 | 8 · 5 · 6 | 11 · 3 · 3 | 99 · 100 · 99 |
| 3 (add73) | 98 · 98 · 96 | 94 · 97 · 96 | 93 · 94 · 93 | 2 · 5 · 5 | 13 · 10 · 10 | 100 · 100 · 100 |
| 4 (sub73) | 98 · 98 · 95 | 97 · 99 · 97 | 93 · 95 · 94 | 94 · 94 · 95 | 12 · 7 · 11 | 99 · 100 · 99 |

### E4 predictions (sealed T1 = add97 after stage 4 unless stated)

- P4.1 A: T1 after stage 1 [96, 92, 92] (≥70), after stage 4 [0, 0, 0] (<30)
- P4.2 A0: T1 after stage 4 [0, 1, 1] (<50, above A)
- P4.3 B: T1 sealed after stage 4 [0, 0, 0] (within 10 of A)
- P4.4 C: after stage 4, T1 [90, 88, 78], T2 [90, 93, 82], T3 [86, 84, 77] (≥70), T4 [91, 93, 86] (≥70)
- P4.5 C′: T1 after stage 4 [0, 1, 1] (tracks A)
- P4.6 D: after stage 4 T1..T4 [[98, 98, 95], [97, 99, 97], [93, 95, 94], [94, 94, 95]] (≥85)

## E5 composition: final-answer exact (%) on held two-step panels at the end of life, three seeds

| Arm | pp within-type | mm within-type | pm cross-type | mp cross-type | intermediate ok (cross, mean) | single ops after life (add97 · sub97, mean over seeds) |
|---|---|---|---|---|---|---|
| Z zero-shot (no life) | 0 · 1 · 0 | 0 · 0 · 2 | 0 · 0 · 0 | 0 · 1 · 0 | — | 98 · 98 |
| DIR direct, sign rule | 32 · 16 · 10 | 32 · 21 · 8 | 2 · 0 · 1 | 2 · 1 · 0 | — | 0 · 1 |
| SCR scratchpad, sign rule | 86 · 87 · 73 | 85 · 84 · 70 | 42 · 58 · 53 | 41 · 56 · 51 | 60 · 71 · 68 | 0 · 0 |
| SCR-sh scratchpad, shuffled finals | 1 · 1 · 1 | 1 · 1 · 1 | 2 · 2 · 0 | 1 · 2 · 1 | 4 · 2 · 2 | 0 · 0 |
| DIR-bp direct, backprop | 95 · 92 · 89 | 94 · 94 · 88 | 1 · 1 · 1 | 1 · 0 · 1 | — | 7 · 0 |
| SCR-bp scratchpad, backprop | 94 · 95 · 90 | 94 · 91 · 88 | 10 · 36 · 4 | 13 · 19 · 4 | 27 · 45 · 7 | 0 · 0 |

Shortcut floors (percent of panel): pp: frequency 1.2, last_op_only 0.8, first_op_only 1.5; mm: frequency 0.8, last_op_only 1.0, first_op_only 0.8; pm: frequency 0.0, last_op_only 1.3, first_op_only 0.5; mp: frequency 0.0, last_op_only 0.7, first_op_only 0.5

### E5 predictions

- P5.1 Z under 5 everywhere: pp [0, 1, 0], pm [0, 0, 0]
- P5.2 DIR cross-type <30: pm [2, 0, 1], mp [2, 1, 0]; within-type ≥60: pp [32, 16, 10], mm [32, 21, 8]
- P5.3 SCR cross-type ≥70: pm [42, 58, 53], mp [41, 56, 51]; within-type ≥80: pp [86, 87, 73], mm [85, 84, 70]
- P5.4 SCR-sh under 10: pm [2, 2, 0], pp [1, 1, 1]
- ceilings: DIR-bp pm [1, 1, 1], SCR-bp pm [10, 36, 4]