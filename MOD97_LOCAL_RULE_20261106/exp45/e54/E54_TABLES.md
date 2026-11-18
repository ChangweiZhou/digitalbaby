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

## Taught fraction 0.10

| Network | Rule | Updates to 80, A | Updates to 80, R | A at 12,288 / 24,576 / 49,152 | R at 12,288 / 24,576 / 49,152 | Length test (40 letters) | max | min | Rise before, last rise of R | Curve of R | CPU-hours | Resumed at |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `G` | S | — | — | 9.0 / 11.8 / 15.2 | 6.9 / 10.1 / 15.1 | 9.8 / 12.8 / 18.6 | 87.8 / 87.3 / 88.3 | 87.1 / 88.0 / 90.6 | +2.0, +1.6 | flattened | 1.86 | never |
| `F10` | S | — | — | 5.3 / 8.7 / 8.5 | 5.3 / 7.3 / 9.3 | 6.4 / 10.8 / 10.7 | 69.2 / 72.4 / 79.1 | 68.7 / 71.4 / 73.6 | +1.1, -0.1 | flattened | 1.95 | never |
| `P8` | S | — | — | 11.8 / 13.8 / 17.4 | 10.1 / 13.2 / 19.1 | 13.7 / 15.0 / 20.9 | 82.8 / 87.0 / 91.3 | 86.2 / 87.9 / 89.2 | +2.2, +1.8 | flattened | 2.77 | never |
| `G` | B | — | — | 16.8 / 21.4 / 26.5 | 14.8 / 19.5 / 24.6 | 18.0 / 24.0 / 28.4 | 90.2 / 89.9 / 90.2 | 92.4 / 91.2 / 92.7 | +2.5, +1.7 | flattened | 1.82 | never |
| `F10` | B | — | — | 9.2 / 11.9 / 17.1 | 8.3 / 10.7 / 17.3 | 13.3 / 17.2 / 23.2 | 77.4 / 78.3 / 84.3 | 81.6 / 79.4 / 85.1 | +2.1, +2.0 | flattened | 1.90 | never |
| `P8` | B | — | — | 19.1 / 24.6 / 33.3 | 15.5 / 21.2 / 29.1 | 23.9 / 28.9 / 37.7 | 83.7 / 86.2 / 89.0 | 87.2 / 88.8 / 87.6 | +2.8, +2.8 | flattened | 2.73 | never |

Share of the connections of `P8` replaced at an event: rule S: 7.2% over the first sixteen events and 19.4% over the last sixteen; rule B: 8.1% over the first sixteen events and 19.5% over the last sixteen.

| Rule | Network against `G` | Updates to 80 on A: word (ratio; ratio at a half) | Updates to 80 on R: word (ratio; ratio at a half) | R at 49,152: network, `G`, difference | Margin | Rises of the network; of `G` | End result |
|---|---|---|---|---|---|---|---|
| S | `F10` | neither reaches (—; 2.36) | neither reaches (—; 2.27) | 9.3, 15.1, -5.7 | 10 | +1.1, -0.1 (flattened); +2.0, +1.6 (flattened) | neither generalises from this fraction; not distinguished |
| S | `P8` | neither reaches (—; 1.36) | neither reaches (—; 1.27) | 19.1, 15.1, +4.0 | 10 | +2.2, +1.8 (flattened); +2.0, +1.6 (flattened) | neither generalises from this fraction; not distinguished |
| B | `F10` | neither reaches (—; 1.70) | neither reaches (—; 1.48) | 17.3, 24.6, -7.3 | 20 | +2.1, +2.0 (flattened); +2.5, +1.7 (flattened) | neither generalises from this fraction; not distinguished |
| B | `P8` | neither reaches (—; 1.25) | neither reaches (—; 1.12) | 29.1, 24.6, +4.5 | 20 | +2.8, +2.8 (flattened); +2.5, +1.7 (flattened) | neither generalises from this fraction; not distinguished |

| Network | R at 49,152: sweep, back-propagation, difference | Margin | Rises under the sweep; under back-propagation | The sweep against back-propagation |
|---|---|---|---|---|
| `G` | 15.1, 24.6, -9.5 | 20 | +2.0, +1.6 (flattened); +2.5, +1.7 (flattened) | not distinguished |
| `F10` | 9.3, 17.3, -8.0 | 20 | +1.1, -0.1 (flattened); +2.1, +2.0 (flattened) | not distinguished |
| `P8` | 19.1, 29.1, -10.0 | 20 | +2.2, +1.8 (flattened); +2.8, +2.8 (flattened) | not distinguished |

## Part 4: the dense network at the half width, taught fraction 0.10, 122,880 updates

| Rule | Updates to 80, R | Updates to 80, A | A at 12,288 / 24,576 / 49,152 / 122,880 | R at 12,288 / 24,576 / 49,152 / 122,880 | Rise before, last rise of R | Curve of R | CPU-hours |
|---|---|---|---|---|---|---|---|
| S | — | — | 7.0 / 8.0 / 12.9 / 17.7 | 7.0 / 8.1 / 10.1 / 13.4 | +1.4, -0.5 | flattened | 1.77 |
| B | — | 112,384 | 28.6 / 35.6 / 52.7 / 79.8 | 26.8 / 32.9 / 49.9 / 75.0 | +2.5, -0.7 | flattened | 1.65 |

## Predictions

| | Holds | Figures |
|---|---|---|
| P54.1 | yes | {"words": {"S": "more", "B": "more"}} |
| P54.2 | yes | {"updates_to_80_R": 13568} |
| P54.3 | yes | {"differences_and_margins": {"F10 S": [-5.733333333333334, 10], "F10 B": [-7.316666666666663, 20], "P8 S": [3.9833333333333325, 10], "P8 B": [4.466666666666669, 20]}} |
| P54.4 | no | {"gains": {"G": 5.116666666666664, "F10": 6.633333333333335, "P8": 7.916666666666664}} |
| P54.5 | yes | {"levels_A": {"G": 15.166666666666666, "F10": 8.5, "P8": 17.38888888888889}} |
| P54.6 | no | {"updates_to_80_R": null} |
| P54.7 | yes | {"updates_to_80_R": null} |
| P54.8 | yes | {"generalises_better": []} |
| P54.9 | no | {"updates_to_80_R": null} |
| P54.10 | yes | {"level_R": 13.433333333333332} |
