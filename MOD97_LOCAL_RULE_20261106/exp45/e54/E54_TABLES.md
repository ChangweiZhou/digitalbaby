# E54 tables (seed 73129)

Written by `python3 e54_tables.py` from the receipts under `exp45/e54/`. Levels are means of three scored rows; rises are differences of means of nine scored rows; all scores are percentages of items answered correctly.

## Taught fraction 0.25

| Network | Rule | Updates to 80, A | Updates to 80, R | A at 12,288 / 24,576 / 49,152 | R at 12,288 / 24,576 / 49,152 | Length test (40 letters) | max | min | Rise before, last rise of R | Curve of R | CPU-hours | Resumed at |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `G` | S | 8,704 | 12,032 | 85.1 / 90.1 / 96.3 | 80.4 / 87.8 / 96.0 | 80.1 / 87.7 / 96.2 | 97.1 / 97.2 / 96.9 | 97.6 / 97.9 / 97.3 | +2.5, +1.4 | flattened | 1.97 | [28672] |
| `F10` | S | 31,744 | 40,960 | 50.7 / 70.7 / 87.1 | 42.9 / 64.8 / 82.8 | 58.1 / 76.3 / 89.3 | 92.6 / 94.2 / 95.7 | 93.6 / 95.2 / 96.0 | +4.7, +2.4 | not flattened | 1.98 | [28672] |
| `G` | B | 4,352 | 7,680 | 85.6 / 92.3 / 95.3 | 83.7 / 90.5 / 95.7 | 86.3 / 90.3 / 94.3 | 96.4 / 97.3 / 97.4 | 97.8 / 97.4 / 99.2 | +4.0, +0.5 | not flattened | 1.87 | [30720] |

| Rule | Network against `G` | Updates to 80 on A: word (ratio; ratio at a half) | Updates to 80 on R: word (ratio; ratio at a half) | R at 49,152: network, `G`, difference | Margin | Rises of the network; of `G` | End result |
|---|---|---|---|---|---|---|---|
| S | `F10` | more (3.65; 2.36) | more (3.40; 2.27) | 82.8, 96.0, -13.2 | 10 | +4.7, +2.4 (not flattened); +2.5, +1.4 (flattened) | not decided within the run |

| Network | R at 49,152: sweep, back-propagation, difference | Margin | Rises under the sweep; under back-propagation | The sweep against back-propagation |
|---|---|---|---|---|
| `G` | 96.0, 95.7, +0.3 | 10 | +2.5, +1.4 (flattened); +4.0, +0.5 (not flattened) | not decided within the run |
