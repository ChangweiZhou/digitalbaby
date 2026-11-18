# E54 tables (seed 73129)

Written by `python3 e54_tables.py` from the receipts under `exp45/e54/`. Levels are means of three scored rows; rises are differences of means of nine scored rows; all scores are percentages of items answered correctly.

## Taught fraction 0.25

| Network | Rule | Updates to 80, A | Updates to 80, R | A at 12,288 / 24,576 / 49,152 | R at 12,288 / 24,576 / 49,152 | Length test (40 letters) | max | min | Rise before, last rise of R | Curve of R | CPU-hours | Resumed at |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `G` | S | 8,704 | 12,032 | 85.1 / 90.1 / 96.3 | 80.4 / 87.8 / 96.0 | 80.1 / 87.7 / 96.2 | 97.1 / 97.2 / 96.9 | 97.6 / 97.9 / 97.3 | +2.5, +1.4 | flattened | 1.97 | [28672] |
| `F10` | S | 31,744 | 40,960 | 50.7 / 70.7 / 87.1 | 42.9 / 64.8 / 82.8 | 58.1 / 76.3 / 89.3 | 92.6 / 94.2 / 95.7 | 93.6 / 95.2 / 96.0 | +4.7, +2.4 | not flattened | 1.98 | [28672] |
| `P8` | S | 10,496 | 13,568 | 83.3 / 91.4 / 91.8 | 75.8 / 87.8 / 90.7 | 81.1 / 92.2 / 92.6 | 97.0 / 98.0 / 97.9 | 97.6 / 98.8 / 98.3 | +2.1, +0.4 | flattened | 2.52 | never |
| `G` | B | 4,352 | 7,680 | 85.6 / 92.3 / 95.3 | 83.7 / 90.5 / 95.7 | 86.3 / 90.3 / 94.3 | 96.4 / 97.3 / 97.4 | 97.8 / 97.4 / 99.2 | +4.0, +0.5 | not flattened | 1.87 | [30720] |
| `F10` | B | 18,688 | 20,480 | 68.5 / 87.2 / 95.6 | 64.0 / 85.4 / 94.6 | 76.1 / 88.9 / 95.6 | 89.8 / 93.2 / 94.2 | 92.2 / 95.1 / 93.6 | +2.1, +1.0 | flattened | 1.77 | never |
| `P8` | B | 8,448 | 10,240 | 88.2 / 93.4 / 95.4 | 85.1 / 91.1 / 94.0 | 86.4 / 93.2 / 95.4 | 95.9 / 97.3 / 97.7 | 97.1 / 97.3 / 97.4 | +1.2, +0.6 | flattened | 2.47 | never |

Share of the connections of `P8` replaced at an event: rule S: 7.5% over the first sixteen events and 20.8% over the last sixteen; rule B: 7.8% over the first sixteen events and 21.3% over the last sixteen.

| Rule | Network against `G` | Updates to 80 on A: word (ratio; ratio at a half) | Updates to 80 on R: word (ratio; ratio at a half) | R at 49,152: network, `G`, difference | Margin | Rises of the network; of `G` | End result |
|---|---|---|---|---|---|---|---|
| S | `F10` | more (3.65; 2.36) | more (3.40; 2.27) | 82.8, 96.0, -13.2 | 10 | +4.7, +2.4 (not flattened); +2.5, +1.4 (flattened) | not decided within the run |
| S | `P8` | about the same (1.21; 1.36) | about the same (1.13; 1.27) | 90.7, 96.0, -5.3 | 10 | +2.1, +0.4 (flattened); +2.5, +1.4 (flattened) | not distinguished |
| B | `F10` | more (4.29; 1.70) | more (2.67; 1.48) | 94.6, 95.7, -1.1 | 10 | +2.1, +1.0 (flattened); +4.0, +0.5 (not flattened) | not decided within the run |
| B | `P8` | more (1.94; 1.25) | about the same (1.33; 1.12) | 94.0, 95.7, -1.7 | 10 | +1.2, +0.6 (flattened); +4.0, +0.5 (not flattened) | not decided within the run |

| Network | R at 49,152: sweep, back-propagation, difference | Margin | Rises under the sweep; under back-propagation | The sweep against back-propagation |
|---|---|---|---|---|
| `G` | 96.0, 95.7, +0.3 | 10 | +2.5, +1.4 (flattened); +4.0, +0.5 (not flattened) | not decided within the run |
| `F10` | 82.8, 94.6, -11.8 | 10 | +4.7, +2.4 (not flattened); +2.1, +1.0 (flattened) | not decided within the run |
| `P8` | 90.7, 94.0, -3.3 | 10 | +2.1, +0.4 (flattened); +1.2, +0.6 (flattened) | not distinguished |

## Taught fraction 0.15

| Network | Rule | Updates to 80, A | Updates to 80, R | A at 12,288 / 24,576 / 49,152 | R at 12,288 / 24,576 / 49,152 | Length test (40 letters) | max | min | Rise before, last rise of R | Curve of R | CPU-hours | Resumed at |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `G` | S | — | — | 28.9 / 36.6 / 52.7 | 26.2 / 32.1 / 46.6 | 29.4 / 34.6 / 52.6 | 92.1 / 92.6 / 92.9 | 90.4 / 92.4 / 94.9 | +5.0, +3.7 | not flattened | 1.75 | [25600] |
| `F10` | S | — | — | 15.2 / 22.7 / 32.5 | 12.4 / 19.1 / 29.7 | 18.6 / 25.0 / 38.0 | 84.3 / 86.1 / 88.6 | 77.2 / 81.0 / 85.6 | +3.7, +3.0 | not flattened | 1.78 | [1024] |
| `P8` | S | — | — | 36.1 / 48.5 / 62.5 | 31.7 / 42.6 / 56.6 | 40.4 / 51.4 / 66.4 | 92.8 / 93.8 / 95.9 | 92.1 / 92.2 / 92.9 | +5.6, +2.1 | not flattened | 2.59 | never |
| `G` | B | — | — | 58.0 / 64.8 / 77.6 | 53.9 / 60.8 / 73.0 | 59.9 / 67.4 / 81.7 | 94.2 / 96.8 / 96.4 | 93.9 / 94.6 / 96.7 | +4.4, +4.6 | not flattened | 1.67 | [2048] |
| `F10` | B | — | — | 33.1 / 41.8 / 60.9 | 29.9 / 38.4 / 56.8 | 37.7 / 48.8 / 70.3 | 89.6 / 90.0 / 91.3 | 85.2 / 87.1 / 86.0 | +5.5, +7.8 | not flattened | 1.74 | never |
| `P8` | B | 47,872 | — | 43.6 / 57.1 / 78.3 | 38.8 / 51.6 / 72.9 | 45.1 / 59.9 / 79.4 | 93.1 / 94.9 / 95.3 | 89.2 / 91.6 / 94.7 | +8.0, +5.5 | not flattened | 2.56 | never |

Share of the connections of `P8` replaced at an event: rule S: 7.9% over the first sixteen events and 20.5% over the last sixteen; rule B: 7.8% over the first sixteen events and 20.2% over the last sixteen.

| Rule | Network against `G` | Updates to 80 on A: word (ratio; ratio at a half) | Updates to 80 on R: word (ratio; ratio at a half) | R at 49,152: network, `G`, difference | Margin | Rises of the network; of `G` | End result |
|---|---|---|---|---|---|---|---|
| S | `F10` | neither reaches (—; 2.36) | neither reaches (—; 2.27) | 29.7, 46.6, -16.9 | 20 | +3.7, +3.0 (not flattened); +5.0, +3.7 (not flattened) | not decided within the run |
| S | `P8` | neither reaches (—; 1.36) | neither reaches (—; 1.27) | 56.6, 46.6, +10.0 | 20 | +5.6, +2.1 (not flattened); +5.0, +3.7 (not flattened) | not decided within the run |
| B | `F10` | neither reaches (—; 1.70) | neither reaches (—; 1.48) | 56.8, 73.0, -16.2 | 20 | +5.5, +7.8 (not flattened); +4.4, +4.6 (not flattened) | not decided within the run |
| B | `P8` | not decided (—; 1.25) | neither reaches (—; 1.12) | 72.9, 73.0, -0.1 | 20 | +8.0, +5.5 (not flattened); +4.4, +4.6 (not flattened) | not decided within the run |

| Network | R at 49,152: sweep, back-propagation, difference | Margin | Rises under the sweep; under back-propagation | The sweep against back-propagation |
|---|---|---|---|---|
| `G` | 46.6, 73.0, -26.4 | 20 | +5.0, +3.7 (not flattened); +4.4, +4.6 (not flattened) | not decided within the run |
| `F10` | 29.7, 56.8, -27.1 | 20 | +3.7, +3.0 (not flattened); +5.5, +7.8 (not flattened) | not decided within the run |
| `P8` | 56.6, 72.9, -16.4 | 20 | +5.6, +2.1 (not flattened); +8.0, +5.5 (not flattened) | not decided within the run |

## Predictions

| | Holds | Figures |
|---|---|---|
| P54.1 | yes | {"words": {"S": "more", "B": "more"}} |
| P54.2 | yes | {"updates_to_80_R": 13568} |
| P54.6 | no | {"updates_to_80_R": null} |
| P54.7 | yes | {"updates_to_80_R": null} |
