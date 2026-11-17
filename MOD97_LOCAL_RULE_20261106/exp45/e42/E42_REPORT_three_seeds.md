# E42 three-seed report (OPUS_TASK_E42_CELLS_WITH_LIMITED_LOCAL_CONNECTIONS_AND_PERIODS_20261116.md, revision 2)

Assembled by the worker of seed 73129 on branch `claude/gallant-dijkstra-i9zgsr`. No readings beyond the brief's descriptions.

## How the three seeds were run (provenance)

- No separate workers were started for seeds 93683 and 116089; the owner directed this worker to do the whole job (2026-11-16). Stage 0 and seed 73129: `exp45/e42_qual/E42_STAGE0_73129.md`, `exp45/e42/E42_REPORT_73129.md`.
- Checks 1 and 2 for seeds 93683 and 116089 were repeated with runner `exp42_cells.py` sha256 340f9cc5... (commit 4c9e52d1): all PASS on this host (`exp45/e42_qual/check{1,2}_{93683,116089}.json`); check 1 and 2 for 116089 also PASS on the helper host (`exp45/e42_qual/host_b/`).
- Stage 2 for 93683 and 116089 follows seed 73129's stop decision (stage 2 runs); their own stop checks are recorded only: A50 three-row reached by a limited configuration on every seed (see P42.11).
- After two container restarts (16 Nov ~20:35Z and ~20:45Z) lost in-progress runs, later runs used `exp42_ckpt.py`: exp42_cells.run's text plus a full-state checkpoint every 1,024 updates (exp42_cells.py unchanged). Self-test before use: four dry runs (G S, G B, P8 S, P8-source S; 2,048 updates) stopped at 1,024 and resumed in a new process are bit-equal to the original runner's receipts in final tensors, every scored row and the data digest (`exp45/e42_qual/ckpt_selftest.json`, PASS). A live kill-and-resume drill also resumed all three runs at 1,024.
- To halve the wall time the remaining runs were split across two hosts (owner-approved), three processes each: this host ran `exp45/e42/lists/host_a.txt`; a helper session ran `host_b.txt` (seed 116089) on branch `claude/funny-cori-q54ch0`, merged here. Interrupted partial outputs are kept in `exp45/e42/interrupted/`.
- One more container restart on each host (~23:14Z) was resumed automatically from checkpoints.
- 45 of 45 runs complete; every receipt's weights_sha256 matches its committed `.pt`. CPU-h (updates + scoring + replacement): 73129: 15.22, 93683: 18.66, 116089: 18.38; total 52.26.

### Runner and resumes per run

| Seed | Config | Rule | Runner | Resumed at update | tensors_sha256 (16) |
|---|---|---|---|---|---|
| 73129 | P8 | S | exp42_cells.py | - | e391410e6a51a437 |
| 73129 | P8 | B | exp42_cells.py | - | 7fae51d2937eada0 |
| 73129 | G | S | exp42_cells.py | - | d906412631d18946 |
| 73129 | G | B | exp42_cells.py | - | efbdf87a6e1e2aeb |
| 73129 | F10 | S | exp42_cells.py | - | cbf2aefe233ace10 |
| 73129 | F10 | B | exp42_cells.py | - | 0af24abcd60147ed |
| 73129 | A10 | S | exp42_cells.py | - | 7e06217e15f50bfd |
| 73129 | A10 | B | exp42_cells.py | - | 498617c2d1960f13 |
| 73129 | N5 | S | exp42_cells.py | - | 0489847c16ed50cc |
| 73129 | N20 | S | exp42_cells.py | - | 3598c5d5b12d9dc9 |
| 73129 | W10 | S | exp42_cells.py | - | 427909dd9c10f3b9 |
| 73129 | P4 | S | exp42_cells.py | - | 3c207b5829903e54 |
| 73129 | P16 | S | exp42_cells.py | - | 17f044e11c2df968 |
| 73129 | P8-last | S | exp42_cells.py | - | 5cba5d4c6159fbfc |
| 73129 | P8-source | S | exp42_cells.py | - | cd748151fdd56910 |
| 93683 | P8 | S | exp42_cells.py | - | 481d8d0f5618e4a7 |
| 93683 | P8 | B | exp42_cells.py | - | 38c7bbf5064225f7 |
| 93683 | G | S | exp42_cells.py | - | 1eb83bddb92c7912 |
| 93683 | G | B | exp42_cells.py | - | 6c4ff9cee074d90d |
| 93683 | F10 | S | exp42_ckpt.py | 1024 | cfe7d7f4084c08ca |
| 93683 | F10 | B | exp42_ckpt.py | 1024 | daeb66bd04bf37c9 |
| 93683 | A10 | S | exp42_ckpt.py | 18432 | d60a24961d997158 |
| 93683 | A10 | B | exp42_ckpt.py | 18432 | f99eb6e0342f7fee |
| 93683 | N5 | S | exp42_ckpt.py |  | b9686add42cac3a2 |
| 93683 | N20 | S | exp42_ckpt.py |  | 2f5b4e405c3dddd5 |
| 93683 | W10 | S | exp42_ckpt.py |  | c15ebe909b94969c |
| 93683 | P4 | S | exp42_ckpt.py |  | e4ce6d90bde6d6e9 |
| 93683 | P16 | S | exp42_ckpt.py |  | 50738fb1682fc732 |
| 93683 | P8-last | S | exp42_ckpt.py |  | 1bac97672edea641 |
| 93683 | P8-source | S | exp42_ckpt.py | 10240 | fea5cc5567c77714 |
| 116089 | P8 | S | exp42_cells.py | - | 76b3c7e668610233 |
| 116089 | P8 | B | exp42_cells.py | - | c4334b5a136e4043 |
| 116089 | G | S | exp42_cells.py | - | 70ae363b0e2cacb1 |
| 116089 | G | B | exp42_cells.py | - | a5b5ab81e9d701a4 |
| 116089 | F10 | S | exp42_ckpt.py | 1024 | aaf41d51e6a768f6 |
| 116089 | F10 | B | exp42_ckpt.py | 9216 | e3b8764227a42bfe |
| 116089 | A10 | S | exp42_ckpt.py | 9216 | 4d9b8bfa514b7ea0 |
| 116089 | A10 | B | exp42_ckpt.py | 9216 | 238f1f1fa8fce5d0 |
| 116089 | N5 | S | exp42_ckpt.py |  | c45c1c0e14e31ebb |
| 116089 | N20 | S | exp42_ckpt.py |  | d995bf6b114adcc1 |
| 116089 | W10 | S | exp42_ckpt.py |  | 99f1d7bf78b497d2 |
| 116089 | P4 | S | exp42_ckpt.py |  | 5dd3aa606dd84dd3 |
| 116089 | P16 | S | exp42_ckpt.py |  | b612c5061c88b703 |
| 116089 | P8-last | S | exp42_ckpt.py |  | f532e8cebf8d7c1f |
| 116089 | P8-source | S | exp42_ckpt.py |  | a2c6068f17a30bbd |

## Predictions (the brief's, P42.1 to P42.11)

| | Prediction | Chance | Per seed (73129, 93683, 116089) | Outcome |
|---|---|---|---|---|
| P42.1 | Check 1 passes under both rules | 90% | S: tensors and digest equal; B: tensors and digest equal | occurred |
| P42.2 | A10 reaches 80 on A within the dose, two of three seeds, sweep | 60% | A80 (three-row) at 13,824, 14,336, 13,824 | occurred |
| P42.3 | F10 reaches 80 on A within the dose, two of three seeds, backpropagation | 70% | A80 (three-row) at 8,704, 6,912, 9,216 | occurred |
| P42.4 | A10 against G, updates to A80, sweep: rises (more), two of three | 65% | more (5,632 -> 13,824); more (4,864 -> 14,336); more (5,376 -> 13,824) | occurred |
| P42.5 | A10 against F10, updates to A80, sweep: falls or stays (fewer or about the same), two of three | 55% | about the same (13,312 -> 13,824); about the same (13,056 -> 14,336); about the same (12,544 -> 13,824) | occurred |
| P42.6 | P8 against A10, updates to A80, sweep: fewer 20%, about the same 35%, more 45% | fewer 20% / same 35% / more 45% | fewer (13,824 -> 7,680); fewer (14,336 -> 8,448); fewer (13,824 -> 7,424) | cell: fewer |
| P42.7 | P16 lower than A10 at 40 letters, two of three | 55% | about the same (91.78 -> 82.78); about the same (92.56 -> 94.11); lower (86.67 -> 67.33) | did not occur |
| P42.8 | F10 lower than G at 40 letters, sweep, two of three | 45% | about the same (96.44 -> 89.11); about the same (96.44 -> 95.22); about the same (97.11 -> 88.22) | did not occur |
| P42.9 | W10 against A10: about the same on both updates to A80 and the length test | 55% | updates: about the same, about the same, about the same; length: about the same, about the same, about the same | occurred |
| P42.10 | Some stage-1 configuration's description against G differs between the sweep and backpropagation (three-seed cells) | 40% | F10 A80 weight-items: sweep more, backprop about the same; P8 A80 updates: sweep about the same, backprop more; P8 R80 updates: sweep about the same, backprop more | occurred |
| P42.11 | The stop applies (no limited configuration reaches 50 on A on seed 73129) | 10% | A50 three-row reached by a limited configuration on 73129: True | did not occur |

## Tables (generated by `python3 -I e42_tables.py --seeds 73129,93683,116089 --tag three_seeds --out exp45/e42`)

Receipts read from `exp45/e42`; seeds requested 73129, 93683, 116089; complete runs 45.

### Batch digest at row 4,096 (check 2) and the limit at every scored row (check 4)

| Seed | runs | digests equal | digest (first 16) | limit held at every row |
|---|---|---|---|---|
| 73129 | 15 | yes | bbd71ab7ca27b72e | yes |
| 93683 | 15 | yes | 65b1c7e06b17459a | yes |
| 116089 | 15 | yes | 1df203fb8f299dc0 | yes |

### Check 11: G's final tensors_sha256 beside E26's receipt (reported, not a failure)

| Rule | Seed | G | E26 | equal |
|---|---|---|---|---|
| S | 73129 | d906412631d18946 | d906412631d18946 | True |
| B | 73129 | efbdf87a6e1e2aeb | efbdf87a6e1e2aeb | True |
| S | 93683 | 1eb83bddb92c7912 | 1eb83bddb92c7912 | True |
| B | 93683 | 6c4ff9cee074d90d | 6c4ff9cee074d90d | True |
| S | 116089 | 70ae363b0e2cacb1 | 70ae363b0e2cacb1 | True |
| B | 116089 | a5b5ab81e9d701a4 | a5b5ab81e9d701a4 | True |

### Seed 73129: per run (three-row version; single-row in brackets)

| Config | Rule | A50 | A80 | A90 | R50 | R80 | R90 | A80 weight-items | max A (row) | max R (row) | final A, R, taught | len 12, 40 (last three rows) | replaced | cosine start, end | CPU-h updates + scoring |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| G | S | 3,840 [3,840] | 5,632 [5,632] | 12,800 [7,936] | 3,840 [3,840] | 6,656 [6,656] | 13,824 [12,288] | 4.05e+11 | 98.17 (24,064) | 97.05 (24,320) | 98.00, 95.85, 99.33 | 98.44, 96.44 | 0 | 0.741, 0.745 | 0.70 + 0.15 |
| G | B | 4,096 [4,096] | 5,120 [5,120] | 10,240 [7,936] | 3,840 [3,840] | 6,400 [5,120] | 7,936 [7,936] | 3.68e+11 | 98.50 (24,576) | 98.15 (21,248) | 98.50, 98.15, 98.67 | 96.89, 94.56 | 0 | 0.741, 0.689 | 0.67 + 0.15 |
| F10 | S | 8,192 [7,168] | 13,312 [12,800] | 20,480 [19,456] | 9,472 [8,960] | 15,104 [13,824] | 23,296 [19,456] | 7.97e+11 | 92.17 (24,576) | 92.30 (24,064) | 92.17, 88.45, 98.17 | 90.11, 89.11 | 0 | 0.744, 0.604 | 0.72 + 0.16 |
| F10 | B | 5,632 [5,632] | 8,704 [8,192] | 15,360 [15,360] | 6,400 [6,400] | 9,472 [8,960] | 19,200 [15,616] | 5.21e+11 | 96.67 (23,296) | 95.30 (24,064) | 91.83, 93.30, 97.50 | 92.89, 92.56 | 0 | 0.744, 0.631 | 0.70 + 0.16 |
| A10 | S | 8,192 [7,680] | 13,824 [13,824] | 22,528 [17,408] | 9,728 [9,728] | 16,128 [16,128] | not reached [23,808] | 8.28e+11 | 93.00 (23,040) | 90.85 (23,808) | 92.83, 86.15, 98.83 | 92.67, 91.78 | 35,833 | 0.744, 0.640 | 0.70 + 0.15 |
| A10 | B | 5,888 [5,888] | 9,472 [8,192] | 20,992 [15,616] | 6,400 [6,400] | 9,472 [9,472] | 19,456 [16,384] | 5.67e+11 | 93.33 (24,064) | 94.35 (24,064) | 93.17, 92.85, 97.83 | 92.78, 92.22 | 37,525 | 0.744, 0.563 | 0.68 + 0.16 |
| P8 | S | 5,376 [5,376] | 7,680 [7,680] | 10,496 [9,984] | 6,400 [6,400] | 8,448 [8,448] | 12,288 [12,288] | 4.6e+11 | 97.00 (22,784) | 94.95 (24,064) | 95.67, 92.50, 99.17 | 96.33, 96.00 | 38,713 | 0.727, 0.675 | 0.94 + 0.25 |
| P8 | B | 4,864 [4,864] | 6,400 [6,400] | 8,192 [8,192] | 5,120 [5,120] | 7,168 [7,168] | 8,960 [8,448] | 3.84e+11 | 98.67 (23,552) | 97.85 (19,712) | 98.00, 96.60, 99.50 | 97.00, 96.56 | 39,903 | 0.727, 0.739 | 0.92 + 0.25 |
| N5 | S | 9,728 [9,216] | 17,152 [16,384] | not reached [not reached] | 11,776 [11,264] | 19,456 [18,688] | not reached [not reached] | 1.02e+12 | 89.00 (24,576) | 89.40 (24,064) | 89.00, 83.65, 96.00 | 88.22, 86.89 | 17,033 | 0.766, 0.641 | 0.73 + 0.16 |
| N20 | S | 7,680 [7,680] | 14,336 [12,544] | not reached [23,808] | 8,704 [8,704] | 17,152 [16,384] | not reached [not reached] | 8.66e+11 | 90.33 (23,808) | 89.85 (24,064) | 90.33, 86.10, 96.50 | 89.44, 87.89 | 74,785 | 0.754, 0.482 | 0.73 + 0.16 |
| W10 | S | 9,216 [9,216] | 17,408 [16,384] | not reached [not reached] | 11,264 [9,984] | 20,480 [19,456] | not reached [not reached] | 1.04e+12 | 88.83 (23,552) | 88.30 (24,064) | 88.50, 84.85, 96.50 | 88.56, 88.00 | 36,594 | 0.727, 0.593 | 0.73 + 0.16 |
| P4 | S | 6,144 [6,144] | 8,192 [8,192] | 10,496 [9,984] | 6,656 [6,656] | 9,216 [9,216] | 13,824 [13,824] | 4.91e+11 | 96.00 (21,504) | 94.25 (20,736) | 95.33, 92.35, 99.67 | 95.67, 94.44 | 38,876 | 0.729, 0.726 | 0.89 + 0.23 |
| P16 | S | 5,376 [5,376] | 7,936 [7,936] | 11,776 [11,776] | 6,400 [6,400] | 9,216 [9,216] | 13,312 [13,312] | 4.76e+11 | 96.00 (24,064) | 95.90 (21,504) | 95.83, 95.30, 100.00 | 94.56, 82.78 | 38,492 | 0.729, 0.695 | 1.16 + 0.32 |
| P8-last | S | 6,912 [6,912] | 11,520 [10,752] | 20,224 [17,664] | 7,680 [7,168] | 13,312 [11,776] | 20,480 [16,640] | 6.9e+11 | 93.67 (23,296) | 92.35 (24,064) | 92.67, 91.25, 99.00 | 92.00, 91.33 | 39,678 | 0.732, 0.694 | 0.74 + 0.16 |
| P8-source | S | 5,632 [5,632] | 7,680 [7,680] | 11,008 [9,984] | 6,656 [6,656] | 8,448 [8,448] | 11,776 [11,776] | 4.67e+11 | 96.33 (19,712) | 95.90 (21,760) | 95.67, 94.40, 99.17 | 94.22, 93.67 | 34,065 | 0.688, 0.721 | 1.26 + 0.27 |

### Seed 93683: per run (three-row version; single-row in brackets)

| Config | Rule | A50 | A80 | A90 | R50 | R80 | R90 | A80 weight-items | max A (row) | max R (row) | final A, R, taught | len 12, 40 (last three rows) | replaced | cosine start, end | CPU-h updates + scoring |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| G | S | 3,584 [3,584] | 4,864 [4,352] | 6,400 [6,400] | 3,840 [3,840] | 4,864 [4,864] | 7,424 [6,400] | 3.5e+11 | 99.83 (24,320) | 99.20 (23,296) | 98.17, 98.35, 99.17 | 99.33, 96.44 | 0 | 0.740, 0.848 | 1.01 + 0.21 |
| G | B | 3,584 [3,584] | 4,096 [4,096] | 4,864 [4,864] | 3,584 [3,584] | 4,096 [4,096] | 7,424 [5,120] | 2.95e+11 | 99.17 (24,064) | 98.65 (24,064) | 98.67, 96.90, 99.33 | 98.33, 98.00 | 0 | 0.740, 0.822 | 0.89 + 0.20 |
| F10 | S | 7,936 [7,936] | 13,056 [13,056] | 18,944 [15,872] | 8,960 [8,960] | 14,336 [13,312] | 21,760 [19,200] | 7.82e+11 | 95.50 (24,320) | 93.55 (23,040) | 94.00, 91.75, 96.67 | 95.67, 95.22 | 0 | 0.758, 0.569 | 0.95 + 0.20 |
| F10 | B | 4,864 [4,864] | 6,912 [6,912] | 9,472 [8,704] | 5,120 [5,120] | 7,936 [7,936] | 11,776 [11,008] | 4.14e+11 | 98.00 (24,320) | 97.05 (24,064) | 97.67, 95.85, 99.33 | 98.11, 97.67 | 0 | 0.758, 0.636 | 0.92 + 0.20 |
| A10 | S | 8,192 [8,192] | 14,336 [13,824] | 19,968 [19,968] | 9,472 [9,472] | 15,616 [14,848] | 21,504 [19,968] | 8.59e+11 | 95.17 (24,320) | 92.70 (24,320) | 92.67, 90.65, 96.83 | 94.44, 92.56 | 36,470 | 0.758, 0.665 | 0.90 + 0.19 |
| A10 | B | 4,864 [4,864] | 7,680 [7,168] | 13,056 [9,216] | 5,120 [5,120] | 7,936 [7,936] | 13,056 [11,776] | 4.6e+11 | 97.67 (23,552) | 96.80 (23,040) | 95.83, 94.35, 98.00 | 97.44, 97.11 | 38,733 | 0.758, 0.605 | 0.87 + 0.19 |
| P8 | S | 5,632 [5,632] | 8,448 [8,448] | 11,008 [11,008] | 6,400 [6,400] | 9,472 [9,472] | 12,544 [11,776] | 5.06e+11 | 98.00 (18,432) | 95.65 (24,320) | 95.67, 95.45, 99.33 | 96.56, 96.56 | 36,934 | 0.753, 0.676 | 1.31 + 0.33 |
| P8 | B | 4,864 [4,352] | 7,168 [6,656] | 9,984 [8,704] | 4,864 [4,352] | 7,680 [7,168] | 11,520 [11,008] | 4.3e+11 | 98.50 (24,320) | 96.00 (24,320) | 97.83, 95.10, 99.67 | 98.11, 97.56 | 39,512 | 0.753, 0.777 | 1.27 + 0.33 |
| N5 | S | 9,216 [9,216] | 14,336 [13,824] | 21,760 [19,968] | 10,240 [10,240] | 15,616 [15,616] | not reached [22,272] | 8.55e+11 | 92.33 (21,760) | 90.35 (22,272) | 89.67, 86.70, 97.00 | 93.56, 91.78 | 16,859 | 0.749, 0.557 | 0.83 + 0.18 |
| N20 | S | 7,936 [7,424] | 13,056 [12,288] | 19,968 [19,968] | 8,960 [8,960] | 15,872 [15,872] | not reached [not reached] | 7.88e+11 | 92.83 (24,320) | 89.85 (24,320) | 92.50, 86.75, 95.17 | 94.00, 92.33 | 76,774 | 0.749, 0.509 | 0.82 + 0.17 |
| W10 | S | 8,704 [8,704] | 12,032 [12,032] | 16,896 [16,128] | 9,216 [9,216] | 13,312 [13,312] | 21,760 [19,968] | 7.21e+11 | 93.67 (23,552) | 92.10 (21,760) | 91.67, 90.05, 97.17 | 94.00, 93.67 | 38,369 | 0.768, 0.615 | 0.83 + 0.17 |
| P4 | S | 6,400 [6,400] | 9,728 [9,728] | 13,568 [13,568] | 7,168 [7,168] | 11,008 [11,008] | 16,896 [15,872] | 5.83e+11 | 96.00 (24,064) | 92.75 (23,040) | 94.50, 91.30, 99.00 | 95.44, 95.89 | 37,511 | 0.753, 0.726 | 1.00 + 0.24 |
| P16 | S | 5,632 [5,632] | 9,216 [9,216] | 13,056 [13,056] | 7,168 [7,168] | 11,008 [11,008] | 17,408 [17,408] | 5.52e+11 | 96.00 (24,064) | 94.10 (24,320) | 94.83, 91.70, 98.33 | 95.44, 94.11 | 38,166 | 0.752, 0.670 | 1.32 + 0.33 |
| P8-last | S | 6,400 [5,888] | 9,216 [9,216] | 13,568 [13,568] | 6,656 [6,656] | 11,008 [9,728] | 15,616 [15,616] | 5.52e+11 | 96.33 (22,528) | 93.95 (23,296) | 94.83, 91.15, 97.33 | 95.00, 93.00 | 39,552 | 0.754, 0.687 | 0.82 + 0.17 |
| P8-source | S | 5,376 [5,376] | 7,936 [7,936] | 12,032 [11,520] | 6,656 [6,656] | 10,240 [9,728] | 15,360 [13,312] | 4.82e+11 | 97.00 (23,808) | 94.20 (24,064) | 95.00, 94.20, 99.17 | 96.00, 96.00 | 32,192 | 0.733, 0.777 | 1.44 + 0.29 |

### Seed 116089: per run (three-row version; single-row in brackets)

| Config | Rule | A50 | A80 | A90 | R50 | R80 | R90 | A80 weight-items | max A (row) | max R (row) | final A, R, taught | len 12, 40 (last three rows) | replaced | cosine start, end | CPU-h updates + scoring |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| G | S | 3,072 [3,072] | 5,376 [4,608] | 9,472 [5,376] | 3,072 [3,072] | 6,400 [4,608] | 9,472 [8,192] | 3.87e+11 | 99.83 (24,576) | 99.45 (24,576) | 99.83, 99.45, 99.33 | 99.22, 97.11 | 0 | 0.666, 0.844 | 1.01 + 0.21 |
| G | B | 2,304 [2,304] | 4,352 [3,584] | 9,216 [6,912] | 2,816 [2,816] | 5,120 [3,584] | 9,216 [6,912] | 3.13e+11 | 97.33 (22,528) | 98.15 (22,528) | 93.83, 92.50, 95.50 | 94.22, 93.22 | 0 | 0.666, 0.838 | 0.89 + 0.20 |
| F10 | S | 8,704 [8,704] | 12,544 [12,544] | 21,504 [16,896] | 9,728 [9,728] | 14,336 [13,568] | 22,272 [21,248] | 7.51e+11 | 92.67 (19,456) | 91.95 (23,040) | 90.33, 90.25, 97.50 | 90.78, 88.22 | 0 | 0.659, 0.747 | 0.94 + 0.20 |
| F10 | B | 5,120 [5,120] | 9,216 [8,448] | 19,456 [15,872] | 5,632 [5,632] | 9,728 [9,216] | 19,200 [17,664] | 5.52e+11 | 95.00 (22,784) | 93.80 (22,272) | 94.83, 92.85, 97.50 | 94.22, 93.67 | 0 | 0.659, 0.613 | 0.73 + 0.16 |
| A10 | S | 8,448 [8,448] | 13,824 [12,800] | 22,528 [22,528] | 8,960 [8,960] | 14,336 [14,336] | not reached [21,760] | 8.28e+11 | 91.50 (22,784) | 90.60 (23,040) | 90.67, 88.40, 96.33 | 91.67, 86.67 | 36,407 | 0.659, 0.668 | 0.75 + 0.16 |
| A10 | B | 5,632 [5,632] | 7,936 [7,424] | 12,544 [11,264] | 5,632 [5,632] | 9,728 [9,216] | 13,824 [12,800] | 4.75e+11 | 98.17 (22,784) | 96.25 (22,272) | 98.00, 95.90, 98.83 | 96.67, 96.11 | 39,040 | 0.659, 0.526 | 0.74 + 0.16 |
| P8 | S | 5,120 [5,120] | 7,424 [7,424] | 12,288 [11,008] | 5,888 [5,888] | 9,216 [8,192] | 13,824 [13,824] | 4.45e+11 | 96.00 (24,320) | 94.10 (21,248) | 94.17, 92.35, 98.00 | 95.44, 95.33 | 37,856 | 0.665, 0.681 | 1.31 + 0.33 |
| P8 | B | 4,608 [4,608] | 7,168 [6,656] | 9,472 [7,936] | 5,120 [5,120] | 7,936 [7,936] | 12,032 [12,032] | 4.3e+11 | 98.00 (22,528) | 95.85 (23,552) | 96.83, 93.80, 98.33 | 97.44, 97.11 | 40,155 | 0.665, 0.626 | 1.31 + 0.33 |
| N5 | S | 8,448 [8,448] | 14,080 [13,568] | not reached [22,528] | 8,960 [8,960] | 15,616 [14,848] | not reached [not reached] | 8.4e+11 | 92.83 (24,576) | 89.35 (23,040) | 92.83, 88.80, 98.50 | 89.89, 86.78 | 16,590 | 0.670, 0.576 | 0.85 + 0.18 |
| N20 | S | 7,936 [7,936] | 12,288 [11,264] | 22,272 [17,664] | 8,960 [8,960] | 13,312 [13,312] | 22,784 [22,272] | 7.42e+11 | 91.83 (23,296) | 91.30 (23,040) | 89.00, 88.15, 97.00 | 89.89, 88.67 | 74,925 | 0.658, 0.570 | 0.85 + 0.18 |
| W10 | S | 8,704 [8,704] | 14,080 [13,568] | 22,528 [20,480] | 9,728 [9,728] | 16,896 [15,104] | not reached [not reached] | 8.43e+11 | 92.50 (24,320) | 89.95 (24,064) | 90.00, 87.80, 96.83 | 91.56, 90.44 | 36,188 | 0.643, 0.599 | 0.85 + 0.18 |
| P4 | S | 5,376 [5,376] | 7,936 [7,936] | 12,288 [12,288] | 5,632 [5,632] | 8,704 [8,704] | 16,128 [13,056] | 4.76e+11 | 95.83 (24,064) | 93.40 (20,736) | 93.67, 92.25, 98.00 | 93.89, 93.44 | 38,570 | 0.663, 0.683 | 1.06 + 0.25 |
| P16 | S | 5,376 [5,376] | 8,192 [8,192] | 12,800 [10,496] | 6,400 [6,400] | 10,496 [9,728] | 16,896 [16,384] | 4.91e+11 | 95.50 (24,064) | 93.10 (23,808) | 92.83, 91.40, 97.83 | 93.78, 67.33 | 38,565 | 0.665, 0.713 | 1.38 + 0.35 |
| P8-last | S | 6,656 [6,656] | 9,728 [9,728] | 16,384 [15,104] | 7,424 [7,424] | 11,008 [9,984] | 19,712 [16,384] | 5.83e+11 | 94.17 (22,528) | 92.70 (23,296) | 91.33, 90.40, 97.00 | 92.11, 87.00 | 39,356 | 0.664, 0.647 | 0.85 + 0.18 |
| P8-source | S | 5,376 [5,376] | 7,936 [7,424] | 11,008 [11,008] | 6,400 [5,632] | 9,728 [8,704] | 16,128 [13,568] | 4.82e+11 | 97.17 (24,320) | 94.10 (23,296) | 93.83, 91.95, 97.33 | 95.56, 95.44 | 33,608 | 0.658, 0.803 | 1.44 + 0.29 |

### Descriptions (fixed beforehand). Updates and weight-items to the 80 threshold, three-row version (* where the single-row version describes the cell differently); the length test at 40 letters, mean of the last three rows

| Candidate | Reference | Rule | Measure | Cell | Per seed: description (reference, candidate) |
|---|---|---|---|---|---|
| F10 | G | S | A80 updates | more | 73129: more (5,632, 13,312); 93683: more (4,864, 13,056); 116089: more (5,376, 12,544) |
| F10 | G | S | A80 weight-items | more | 73129: more (4.05e+11, 7.97e+11); 93683: more (3.5e+11, 7.82e+11); 116089: more (3.87e+11, 7.51e+11) |
| F10 | G | S | R80 updates | more | 73129: more (6,656, 15,104); 93683: more (4,864, 14,336); 116089: more (6,400, 14,336) |
| F10 | G | S | R80 weight-items | more | 73129: more (4.79e+11, 9.05e+11); 93683: more (3.5e+11, 8.59e+11); 116089: more (4.6e+11, 8.59e+11) |
| F10 | G | S | length at 40 | about the same | 73129: about the same (96.44, 89.11); 93683: about the same (96.44, 95.22); 116089: about the same (97.11, 88.22) |
| F10 | G | B | A80 updates | more | 73129: more (5,120, 8,704); 93683: more (4,096, 6,912); 116089: more (4,352, 9,216) |
| F10 | G | B | A80 weight-items | about the same | 73129: about the same (3.68e+11, 5.21e+11); 93683: about the same (2.95e+11, 4.14e+11); 116089: more (3.13e+11, 5.52e+11) |
| F10 | G | B | R80 updates | more* | 73129: about the same (6,400, 9,472); 93683: more (4,096, 7,936); 116089: more (5,120, 9,728) |
| F10 | G | B | R80 weight-items | more | 73129: about the same (4.6e+11, 5.67e+11); 93683: more (2.95e+11, 4.75e+11); 116089: more (3.68e+11, 5.83e+11) |
| F10 | G | B | length at 40 | about the same | 73129: about the same (94.56, 92.56); 93683: about the same (98.00, 97.67); 116089: about the same (93.22, 93.67) |
| A10 | G | S | A80 updates | more | 73129: more (5,632, 13,824); 93683: more (4,864, 14,336); 116089: more (5,376, 13,824) |
| A10 | G | S | A80 weight-items | more | 73129: more (4.05e+11, 8.28e+11); 93683: more (3.5e+11, 8.59e+11); 116089: more (3.87e+11, 8.28e+11) |
| A10 | G | S | R80 updates | more | 73129: more (6,656, 16,128); 93683: more (4,864, 15,616); 116089: more (6,400, 14,336) |
| A10 | G | S | R80 weight-items | more | 73129: more (4.79e+11, 9.66e+11); 93683: more (3.5e+11, 9.35e+11); 116089: more (4.6e+11, 8.59e+11) |
| A10 | G | S | length at 40 | about the same | 73129: about the same (96.44, 91.78); 93683: about the same (96.44, 92.56); 116089: about the same (97.11, 86.67) |
| A10 | G | B | A80 updates | more | 73129: more (5,120, 9,472); 93683: more (4,096, 7,680); 116089: more (4,352, 7,936) |
| A10 | G | B | A80 weight-items | more* | 73129: more (3.68e+11, 5.67e+11); 93683: more (2.95e+11, 4.6e+11); 116089: more (3.13e+11, 4.75e+11) |
| A10 | G | B | R80 updates | more* | 73129: about the same (6,400, 9,472); 93683: more (4,096, 7,936); 116089: more (5,120, 9,728) |
| A10 | G | B | R80 weight-items | more* | 73129: about the same (4.6e+11, 5.67e+11); 93683: more (2.95e+11, 4.75e+11); 116089: more (3.68e+11, 5.83e+11) |
| A10 | G | B | length at 40 | about the same | 73129: about the same (94.56, 92.22); 93683: about the same (98.00, 97.11); 116089: about the same (93.22, 96.11) |
| P8 | G | S | A80 updates | about the same* | 73129: about the same (5,632, 7,680); 93683: more (4,864, 8,448); 116089: about the same (5,376, 7,424) |
| P8 | G | S | A80 weight-items | about the same* | 73129: about the same (4.05e+11, 4.6e+11); 93683: about the same (3.5e+11, 5.06e+11); 116089: about the same (3.87e+11, 4.45e+11) |
| P8 | G | S | R80 updates | about the same* | 73129: about the same (6,656, 8,448); 93683: more (4,864, 9,472); 116089: about the same (6,400, 9,216) |
| P8 | G | S | R80 weight-items | about the same | 73129: about the same (4.79e+11, 5.06e+11); 93683: more (3.5e+11, 5.68e+11); 116089: about the same (4.6e+11, 5.52e+11) |
| P8 | G | S | length at 40 | about the same | 73129: about the same (96.44, 96.00); 93683: about the same (96.44, 96.56); 116089: about the same (97.11, 95.33) |
| P8 | G | B | A80 updates | more | 73129: about the same (5,120, 6,400); 93683: more (4,096, 7,168); 116089: more (4,352, 7,168) |
| P8 | G | B | A80 weight-items | about the same* | 73129: about the same (3.68e+11, 3.84e+11); 93683: about the same (2.95e+11, 4.3e+11); 116089: about the same (3.13e+11, 4.3e+11) |
| P8 | G | B | R80 updates | more | 73129: about the same (6,400, 7,168); 93683: more (4,096, 7,680); 116089: more (5,120, 7,936) |
| P8 | G | B | R80 weight-items | about the same* | 73129: about the same (4.6e+11, 4.3e+11); 93683: more (2.95e+11, 4.6e+11); 116089: about the same (3.68e+11, 4.76e+11) |
| P8 | G | B | length at 40 | about the same | 73129: about the same (94.56, 96.56); 93683: about the same (98.00, 97.56); 116089: about the same (93.22, 97.11) |
| A10 | F10 | S | A80 updates | about the same | 73129: about the same (13,312, 13,824); 93683: about the same (13,056, 14,336); 116089: about the same (12,544, 13,824) |
| A10 | F10 | S | A80 weight-items | about the same | 73129: about the same (7.97e+11, 8.28e+11); 93683: about the same (7.82e+11, 8.59e+11); 116089: about the same (7.51e+11, 8.28e+11) |
| A10 | F10 | S | R80 updates | about the same | 73129: about the same (15,104, 16,128); 93683: about the same (14,336, 15,616); 116089: about the same (14,336, 14,336) |
| A10 | F10 | S | R80 weight-items | about the same | 73129: about the same (9.05e+11, 9.66e+11); 93683: about the same (8.59e+11, 9.35e+11); 116089: about the same (8.59e+11, 8.59e+11) |
| A10 | F10 | S | length at 40 | about the same | 73129: about the same (89.11, 91.78); 93683: about the same (95.22, 92.56); 116089: about the same (88.22, 86.67) |
| A10 | F10 | B | A80 updates | about the same | 73129: about the same (8,704, 9,472); 93683: about the same (6,912, 7,680); 116089: about the same (9,216, 7,936) |
| A10 | F10 | B | A80 weight-items | about the same | 73129: about the same (5.21e+11, 5.67e+11); 93683: about the same (4.14e+11, 4.6e+11); 116089: about the same (5.52e+11, 4.75e+11) |
| A10 | F10 | B | R80 updates | about the same | 73129: about the same (9,472, 9,472); 93683: about the same (7,936, 7,936); 116089: about the same (9,728, 9,728) |
| A10 | F10 | B | R80 weight-items | about the same | 73129: about the same (5.67e+11, 5.67e+11); 93683: about the same (4.75e+11, 4.75e+11); 116089: about the same (5.83e+11, 5.83e+11) |
| A10 | F10 | B | length at 40 | about the same | 73129: about the same (92.56, 92.22); 93683: about the same (97.67, 97.11); 116089: about the same (93.67, 96.11) |
| P8 | A10 | S | A80 updates | fewer | 73129: fewer (13,824, 7,680); 93683: fewer (14,336, 8,448); 116089: fewer (13,824, 7,424) |
| P8 | A10 | S | A80 weight-items | fewer | 73129: fewer (8.28e+11, 4.6e+11); 93683: fewer (8.59e+11, 5.06e+11); 116089: fewer (8.28e+11, 4.45e+11) |
| P8 | A10 | S | R80 updates | fewer | 73129: fewer (16,128, 8,448); 93683: fewer (15,616, 9,472); 116089: fewer (14,336, 9,216) |
| P8 | A10 | S | R80 weight-items | fewer | 73129: fewer (9.66e+11, 5.06e+11); 93683: fewer (9.35e+11, 5.68e+11); 116089: fewer (8.59e+11, 5.52e+11) |
| P8 | A10 | S | length at 40 | about the same | 73129: about the same (91.78, 96.00); 93683: about the same (92.56, 96.56); 116089: about the same (86.67, 95.33) |
| P8 | A10 | B | A80 updates | about the same | 73129: about the same (9,472, 6,400); 93683: about the same (7,680, 7,168); 116089: about the same (7,936, 7,168) |
| P8 | A10 | B | A80 weight-items | about the same | 73129: about the same (5.67e+11, 3.84e+11); 93683: about the same (4.6e+11, 4.3e+11); 116089: about the same (4.75e+11, 4.3e+11) |
| P8 | A10 | B | R80 updates | about the same | 73129: about the same (9,472, 7,168); 93683: about the same (7,936, 7,680); 116089: about the same (9,728, 7,936) |
| P8 | A10 | B | R80 weight-items | about the same | 73129: about the same (5.67e+11, 4.3e+11); 93683: about the same (4.75e+11, 4.6e+11); 116089: about the same (5.83e+11, 4.76e+11) |
| P8 | A10 | B | length at 40 | about the same | 73129: about the same (92.22, 96.56); 93683: about the same (97.11, 97.56); 116089: about the same (96.11, 97.11) |
| N5 | A10 | S | A80 updates | about the same | 73129: about the same (13,824, 17,152); 93683: about the same (14,336, 14,336); 116089: about the same (13,824, 14,080) |
| N5 | A10 | S | A80 weight-items | about the same | 73129: about the same (8.28e+11, 1.02e+12); 93683: about the same (8.59e+11, 8.55e+11); 116089: about the same (8.28e+11, 8.4e+11) |
| N5 | A10 | S | R80 updates | about the same | 73129: about the same (16,128, 19,456); 93683: about the same (15,616, 15,616); 116089: about the same (14,336, 15,616) |
| N5 | A10 | S | R80 weight-items | about the same | 73129: about the same (9.66e+11, 1.16e+12); 93683: about the same (9.35e+11, 9.32e+11); 116089: about the same (8.59e+11, 9.32e+11) |
| N5 | A10 | S | length at 40 | about the same | 73129: about the same (91.78, 86.89); 93683: about the same (92.56, 91.78); 116089: about the same (86.67, 86.78) |
| N20 | A10 | S | A80 updates | about the same | 73129: about the same (13,824, 14,336); 93683: about the same (14,336, 13,056); 116089: about the same (13,824, 12,288) |
| N20 | A10 | S | A80 weight-items | about the same | 73129: about the same (8.28e+11, 8.66e+11); 93683: about the same (8.59e+11, 7.88e+11); 116089: about the same (8.28e+11, 7.42e+11) |
| N20 | A10 | S | R80 updates | about the same | 73129: about the same (16,128, 17,152); 93683: about the same (15,616, 15,872); 116089: about the same (14,336, 13,312) |
| N20 | A10 | S | R80 weight-items | about the same | 73129: about the same (9.66e+11, 1.04e+12); 93683: about the same (9.35e+11, 9.59e+11); 116089: about the same (8.59e+11, 8.04e+11) |
| N20 | A10 | S | length at 40 | about the same | 73129: about the same (91.78, 87.89); 93683: about the same (92.56, 92.33); 116089: about the same (86.67, 88.67) |
| W10 | A10 | S | A80 updates | about the same | 73129: about the same (13,824, 17,408); 93683: about the same (14,336, 12,032); 116089: about the same (13,824, 14,080) |
| W10 | A10 | S | A80 weight-items | about the same | 73129: about the same (8.28e+11, 1.04e+12); 93683: about the same (8.59e+11, 7.21e+11); 116089: about the same (8.28e+11, 8.43e+11) |
| W10 | A10 | S | R80 updates | about the same | 73129: about the same (16,128, 20,480); 93683: about the same (15,616, 13,312); 116089: about the same (14,336, 16,896) |
| W10 | A10 | S | R80 weight-items | about the same | 73129: about the same (9.66e+11, 1.23e+12); 93683: about the same (9.35e+11, 7.97e+11); 116089: about the same (8.59e+11, 1.01e+12) |
| W10 | A10 | S | length at 40 | about the same | 73129: about the same (91.78, 88.00); 93683: about the same (92.56, 93.67); 116089: about the same (86.67, 90.44) |
| P4 | P8 | S | A80 updates | about the same | 73129: about the same (7,680, 8,192); 93683: about the same (8,448, 9,728); 116089: about the same (7,424, 7,936) |
| P4 | P8 | S | A80 weight-items | about the same | 73129: about the same (4.6e+11, 4.91e+11); 93683: about the same (5.06e+11, 5.83e+11); 116089: about the same (4.45e+11, 4.76e+11) |
| P4 | P8 | S | R80 updates | about the same | 73129: about the same (8,448, 9,216); 93683: about the same (9,472, 11,008); 116089: about the same (9,216, 8,704) |
| P4 | P8 | S | R80 weight-items | about the same | 73129: about the same (5.06e+11, 5.52e+11); 93683: about the same (5.68e+11, 6.6e+11); 116089: about the same (5.52e+11, 5.22e+11) |
| P4 | P8 | S | length at 40 | about the same | 73129: about the same (96.00, 94.44); 93683: about the same (96.56, 95.89); 116089: about the same (95.33, 93.44) |
| P16 | P8 | S | A80 updates | about the same | 73129: about the same (7,680, 7,936); 93683: about the same (8,448, 9,216); 116089: about the same (7,424, 8,192) |
| P16 | P8 | S | A80 weight-items | about the same | 73129: about the same (4.6e+11, 4.76e+11); 93683: about the same (5.06e+11, 5.52e+11); 116089: about the same (4.45e+11, 4.91e+11) |
| P16 | P8 | S | R80 updates | about the same | 73129: about the same (8,448, 9,216); 93683: about the same (9,472, 11,008); 116089: about the same (9,216, 10,496) |
| P16 | P8 | S | R80 weight-items | about the same | 73129: about the same (5.06e+11, 5.52e+11); 93683: about the same (5.68e+11, 6.6e+11); 116089: about the same (5.52e+11, 6.29e+11) |
| P16 | P8 | S | length at 40 | about the same | 73129: about the same (96.00, 82.78); 93683: about the same (96.56, 94.11); 116089: lower (95.33, 67.33) |
| P8-last | P8 | S | A80 updates | about the same | 73129: about the same (7,680, 11,520); 93683: about the same (8,448, 9,216); 116089: about the same (7,424, 9,728) |
| P8-last | P8 | S | A80 weight-items | about the same | 73129: about the same (4.6e+11, 6.9e+11); 93683: about the same (5.06e+11, 5.52e+11); 116089: about the same (4.45e+11, 5.83e+11) |
| P8-last | P8 | S | R80 updates | about the same* | 73129: more (8,448, 13,312); 93683: about the same (9,472, 11,008); 116089: about the same (9,216, 11,008) |
| P8-last | P8 | S | R80 weight-items | about the same* | 73129: more (5.06e+11, 7.97e+11); 93683: about the same (5.68e+11, 6.59e+11); 116089: about the same (5.52e+11, 6.59e+11) |
| P8-last | P8 | S | length at 40 | about the same | 73129: about the same (96.00, 91.33); 93683: about the same (96.56, 93.00); 116089: about the same (95.33, 87.00) |
| P8-source | P8 | S | A80 updates | about the same | 73129: about the same (7,680, 7,680); 93683: about the same (8,448, 7,936); 116089: about the same (7,424, 7,936) |
| P8-source | P8 | S | A80 weight-items | about the same | 73129: about the same (4.6e+11, 4.67e+11); 93683: about the same (5.06e+11, 4.82e+11); 116089: about the same (4.45e+11, 4.82e+11) |
| P8-source | P8 | S | R80 updates | about the same | 73129: about the same (8,448, 8,448); 93683: about the same (9,472, 10,240); 116089: about the same (9,216, 9,728) |
| P8-source | P8 | S | R80 weight-items | about the same | 73129: about the same (5.06e+11, 5.13e+11); 93683: about the same (5.68e+11, 6.22e+11); 116089: about the same (5.52e+11, 5.91e+11) |
| P8-source | P8 | S | length at 40 | about the same | 73129: about the same (96.00, 93.67); 93683: about the same (96.56, 96.00); 116089: about the same (95.33, 95.44) |
| G | G | S vs B | A80 updates | about the same | 73129: about the same (5,120, 5,632); 93683: about the same (4,096, 4,864); 116089: about the same (4,352, 5,376) |
| G | G | S vs B | A80 weight-items | about the same | 73129: about the same (3.68e+11, 4.05e+11); 93683: about the same (2.95e+11, 3.5e+11); 116089: about the same (3.13e+11, 3.87e+11) |
| G | G | S vs B | R80 updates | about the same | 73129: about the same (6,400, 6,656); 93683: about the same (4,096, 4,864); 116089: about the same (5,120, 6,400) |
| G | G | S vs B | R80 weight-items | about the same | 73129: about the same (4.6e+11, 4.79e+11); 93683: about the same (2.95e+11, 3.5e+11); 116089: about the same (3.68e+11, 4.6e+11) |
| G | G | S vs B | length at 40 | about the same | 73129: about the same (94.56, 96.44); 93683: about the same (98.00, 96.44); 116089: about the same (93.22, 97.11) |
| F10 | F10 | S vs B | A80 updates | more | 73129: more (8,704, 13,312); 93683: more (6,912, 13,056); 116089: about the same (9,216, 12,544) |
| F10 | F10 | S vs B | A80 weight-items | more | 73129: more (5.21e+11, 7.97e+11); 93683: more (4.14e+11, 7.82e+11); 116089: about the same (5.52e+11, 7.51e+11) |
| F10 | F10 | S vs B | R80 updates | more | 73129: more (9,472, 15,104); 93683: more (7,936, 14,336); 116089: about the same (9,728, 14,336) |
| F10 | F10 | S vs B | R80 weight-items | more | 73129: more (5.67e+11, 9.05e+11); 93683: more (4.75e+11, 8.59e+11); 116089: about the same (5.83e+11, 8.59e+11) |
| F10 | F10 | S vs B | length at 40 | about the same | 73129: about the same (92.56, 89.11); 93683: about the same (97.67, 95.22); 116089: about the same (93.67, 88.22) |
| A10 | A10 | S vs B | A80 updates | more* | 73129: about the same (9,472, 13,824); 93683: more (7,680, 14,336); 116089: more (7,936, 13,824) |
| A10 | A10 | S vs B | A80 weight-items | more* | 73129: about the same (5.67e+11, 8.28e+11); 93683: more (4.6e+11, 8.59e+11); 116089: more (4.75e+11, 8.28e+11) |
| A10 | A10 | S vs B | R80 updates | more* | 73129: more (9,472, 16,128); 93683: more (7,936, 15,616); 116089: about the same (9,728, 14,336) |
| A10 | A10 | S vs B | R80 weight-items | more* | 73129: more (5.67e+11, 9.66e+11); 93683: more (4.75e+11, 9.35e+11); 116089: about the same (5.83e+11, 8.59e+11) |
| A10 | A10 | S vs B | length at 40 | about the same | 73129: about the same (92.22, 91.78); 93683: about the same (97.11, 92.56); 116089: about the same (96.11, 86.67) |
| P8 | P8 | S vs B | A80 updates | about the same | 73129: about the same (6,400, 7,680); 93683: about the same (7,168, 8,448); 116089: about the same (7,168, 7,424) |
| P8 | P8 | S vs B | A80 weight-items | about the same | 73129: about the same (3.84e+11, 4.6e+11); 93683: about the same (4.3e+11, 5.06e+11); 116089: about the same (4.3e+11, 4.45e+11) |
| P8 | P8 | S vs B | R80 updates | about the same | 73129: about the same (7,168, 8,448); 93683: about the same (7,680, 9,472); 116089: about the same (7,936, 9,216) |
| P8 | P8 | S vs B | R80 weight-items | about the same | 73129: about the same (4.3e+11, 5.06e+11); 93683: about the same (4.6e+11, 5.68e+11); 116089: about the same (4.76e+11, 5.52e+11) |
| P8 | P8 | S vs B | length at 40 | about the same | 73129: about the same (96.56, 96.00); 93683: about the same (97.56, 96.56); 116089: about the same (97.11, 95.33) |

### Structure: share of connections never replaced; share of connections by the reader's and the source's period (reader<-source)

| Config | Rule | Seed | never replaced | by period at the start | by period at the end |
|---|---|---|---|---|---|
| A10 | B | 116089 | 0.504 | 1<-1 1.000 | 1<-1 1.000 |
| A10 | B | 73129 | 0.533 | 1<-1 1.000 | 1<-1 1.000 |
| A10 | B | 93683 | 0.512 | 1<-1 1.000 | 1<-1 1.000 |
| A10 | S | 116089 | 0.551 | 1<-1 1.000 | 1<-1 1.000 |
| A10 | S | 73129 | 0.557 | 1<-1 1.000 | 1<-1 1.000 |
| A10 | S | 93683 | 0.540 | 1<-1 1.000 | 1<-1 1.000 |
| F10 | B | 116089 | 1.000 | 1<-1 1.000 | 1<-1 1.000 |
| F10 | B | 73129 | 1.000 | 1<-1 1.000 | 1<-1 1.000 |
| F10 | B | 93683 | 1.000 | 1<-1 1.000 | 1<-1 1.000 |
| F10 | S | 116089 | 1.000 | 1<-1 1.000 | 1<-1 1.000 |
| F10 | S | 73129 | 1.000 | 1<-1 1.000 | 1<-1 1.000 |
| F10 | S | 93683 | 1.000 | 1<-1 1.000 | 1<-1 1.000 |
| G | B | 116089 | 1.000 | 1<-1 1.000 | 1<-1 1.000 |
| G | B | 73129 | 1.000 | 1<-1 1.000 | 1<-1 1.000 |
| G | B | 93683 | 1.000 | 1<-1 1.000 | 1<-1 1.000 |
| G | S | 116089 | 1.000 | 1<-1 1.000 | 1<-1 1.000 |
| G | S | 73129 | 1.000 | 1<-1 1.000 | 1<-1 1.000 |
| G | S | 93683 | 1.000 | 1<-1 1.000 | 1<-1 1.000 |
| N20 | S | 116089 | 0.533 | 1<-1 1.000 | 1<-1 1.000 |
| N20 | S | 73129 | 0.524 | 1<-1 1.000 | 1<-1 1.000 |
| N20 | S | 93683 | 0.520 | 1<-1 1.000 | 1<-1 1.000 |
| N5 | S | 116089 | 0.582 | 1<-1 1.000 | 1<-1 1.000 |
| N5 | S | 73129 | 0.576 | 1<-1 1.000 | 1<-1 1.000 |
| N5 | S | 93683 | 0.593 | 1<-1 1.000 | 1<-1 1.000 |
| P16 | S | 116089 | 0.468 | 1<-1 0.258, 1<-2 0.122, 1<-4 0.059, 1<-8 0.031, 1<-16 0.030, 2<-1 0.132, 2<-2 0.058, 2<-4 0.032, 2<-8 0.015, 2<-16 0.013, 4<-1 0.066, 4<-2 0.027, 4<-4 0.017, 4<-8 0.007, 4<-16 0.008, 8<-1 0.033, 8<-2 0.013, 8<-4 0.011, 8<-8 0.004, 8<-16 0.002, 16<-1 0.032, 16<-2 0.016, 16<-4 0.008, 16<-8 0.002, 16<-16 0.005 | 1<-1 0.262, 1<-2 0.114, 1<-4 0.062, 1<-8 0.036, 1<-16 0.027, 2<-1 0.152, 2<-2 0.053, 2<-4 0.025, 2<-8 0.014, 2<-16 0.006, 4<-1 0.066, 4<-2 0.029, 4<-4 0.018, 4<-8 0.005, 4<-16 0.006, 8<-1 0.033, 8<-2 0.016, 8<-4 0.009, 8<-8 0.002, 8<-16 0.003, 16<-1 0.033, 16<-2 0.016, 16<-4 0.008, 16<-8 0.003, 16<-16 0.002 |
| P16 | S | 73129 | 0.482 | 1<-1 0.248, 1<-2 0.128, 1<-4 0.063, 1<-8 0.031, 1<-16 0.030, 2<-1 0.130, 2<-2 0.061, 2<-4 0.029, 2<-8 0.013, 2<-16 0.016, 4<-1 0.061, 4<-2 0.031, 4<-4 0.016, 4<-8 0.008, 4<-16 0.009, 8<-1 0.031, 8<-2 0.019, 8<-4 0.007, 8<-8 0.004, 8<-16 0.002, 16<-1 0.028, 16<-2 0.017, 16<-4 0.009, 16<-8 0.005, 16<-16 0.004 | 1<-1 0.253, 1<-2 0.105, 1<-4 0.080, 1<-8 0.034, 1<-16 0.029, 2<-1 0.142, 2<-2 0.054, 2<-4 0.025, 2<-8 0.014, 2<-16 0.014, 4<-1 0.068, 4<-2 0.031, 4<-4 0.018, 4<-8 0.005, 4<-16 0.003, 8<-1 0.034, 8<-2 0.019, 8<-4 0.005, 8<-8 0.000, 8<-16 0.004, 16<-1 0.033, 16<-2 0.018, 16<-4 0.006, 16<-8 0.004, 16<-16 0.001 |
| P16 | S | 93683 | 0.475 | 1<-1 0.242, 1<-2 0.130, 1<-4 0.067, 1<-8 0.034, 1<-16 0.028, 2<-1 0.130, 2<-2 0.059, 2<-4 0.029, 2<-8 0.015, 2<-16 0.017, 4<-1 0.067, 4<-2 0.033, 4<-4 0.014, 4<-8 0.004, 4<-16 0.007, 8<-1 0.034, 8<-2 0.015, 8<-4 0.006, 8<-8 0.005, 8<-16 0.002, 16<-1 0.033, 16<-2 0.018, 16<-4 0.006, 16<-8 0.005, 16<-16 0.001 | 1<-1 0.255, 1<-2 0.120, 1<-4 0.072, 1<-8 0.030, 1<-16 0.023, 2<-1 0.138, 2<-2 0.056, 2<-4 0.031, 2<-8 0.014, 2<-16 0.011, 4<-1 0.070, 4<-2 0.028, 4<-4 0.017, 4<-8 0.005, 4<-16 0.005, 8<-1 0.036, 8<-2 0.016, 8<-4 0.004, 8<-8 0.005, 8<-16 0.002, 16<-1 0.036, 16<-2 0.017, 16<-4 0.004, 16<-8 0.004, 16<-16 0.002 |
| P4 | S | 116089 | 0.492 | 1<-1 0.258, 1<-2 0.122, 1<-4 0.120, 2<-1 0.132, 2<-2 0.058, 2<-4 0.060, 4<-1 0.131, 4<-2 0.055, 4<-4 0.064 | 1<-1 0.262, 1<-2 0.120, 1<-4 0.118, 2<-1 0.138, 2<-2 0.054, 2<-4 0.059, 4<-1 0.138, 4<-2 0.054, 4<-4 0.057 |
| P4 | S | 73129 | 0.495 | 1<-1 0.248, 1<-2 0.128, 1<-4 0.124, 2<-1 0.130, 2<-2 0.061, 2<-4 0.059, 4<-1 0.120, 4<-2 0.067, 4<-4 0.063 | 1<-1 0.259, 1<-2 0.106, 1<-4 0.134, 2<-1 0.137, 2<-2 0.055, 2<-4 0.057, 4<-1 0.130, 4<-2 0.058, 4<-4 0.062 |
| P4 | S | 93683 | 0.492 | 1<-1 0.242, 1<-2 0.130, 1<-4 0.128, 2<-1 0.130, 2<-2 0.059, 2<-4 0.061, 4<-1 0.134, 4<-2 0.066, 4<-4 0.050 | 1<-1 0.260, 1<-2 0.114, 1<-4 0.127, 2<-1 0.139, 2<-2 0.051, 2<-4 0.061, 4<-1 0.133, 4<-2 0.059, 4<-4 0.058 |
| P8-last | S | 116089 | 0.492 | 1<-1 0.258, 1<-2 0.122, 1<-4 0.059, 1<-8 0.061, 2<-1 0.132, 2<-2 0.058, 2<-4 0.032, 2<-8 0.028, 4<-1 0.066, 4<-2 0.027, 4<-4 0.017, 4<-8 0.015, 8<-1 0.065, 8<-2 0.028, 8<-4 0.018, 8<-8 0.014 | 1<-1 0.266, 1<-2 0.121, 1<-4 0.055, 1<-8 0.058, 2<-1 0.137, 2<-2 0.058, 2<-4 0.033, 2<-8 0.022, 4<-1 0.062, 4<-2 0.030, 4<-4 0.017, 4<-8 0.016, 8<-1 0.066, 8<-2 0.028, 8<-4 0.017, 8<-8 0.014 |
| P8-last | S | 73129 | 0.466 | 1<-1 0.248, 1<-2 0.128, 1<-4 0.063, 1<-8 0.061, 2<-1 0.130, 2<-2 0.061, 2<-4 0.029, 2<-8 0.029, 4<-1 0.061, 4<-2 0.031, 4<-4 0.016, 4<-8 0.017, 8<-1 0.059, 8<-2 0.036, 8<-4 0.016, 8<-8 0.014 | 1<-1 0.263, 1<-2 0.117, 1<-4 0.064, 1<-8 0.056, 2<-1 0.141, 2<-2 0.062, 2<-4 0.023, 2<-8 0.025, 4<-1 0.064, 4<-2 0.029, 4<-4 0.016, 4<-8 0.016, 8<-1 0.063, 8<-2 0.035, 8<-4 0.014, 8<-8 0.013 |
| P8-last | S | 93683 | 0.466 | 1<-1 0.242, 1<-2 0.130, 1<-4 0.067, 1<-8 0.061, 2<-1 0.130, 2<-2 0.059, 2<-4 0.029, 2<-8 0.032, 4<-1 0.067, 4<-2 0.033, 4<-4 0.014, 4<-8 0.011, 8<-1 0.068, 8<-2 0.032, 8<-4 0.012, 8<-8 0.013 | 1<-1 0.270, 1<-2 0.118, 1<-4 0.062, 1<-8 0.050, 2<-1 0.138, 2<-2 0.061, 2<-4 0.025, 2<-8 0.026, 4<-1 0.070, 4<-2 0.033, 4<-4 0.010, 4<-8 0.013, 8<-1 0.064, 8<-2 0.032, 8<-4 0.013, 8<-8 0.016 |
| P8-source | S | 116089 | 0.558 | 1<-1 0.258, 1<-2 0.122, 1<-4 0.059, 1<-8 0.061, 2<-1 0.132, 2<-2 0.058, 2<-4 0.032, 2<-8 0.028, 4<-1 0.066, 4<-2 0.027, 4<-4 0.017, 4<-8 0.015, 8<-1 0.065, 8<-2 0.028, 8<-4 0.018, 8<-8 0.014 | 1<-1 0.263, 1<-2 0.115, 1<-4 0.066, 1<-8 0.056, 2<-1 0.142, 2<-2 0.057, 2<-4 0.029, 2<-8 0.022, 4<-1 0.070, 4<-2 0.026, 4<-4 0.017, 4<-8 0.012, 8<-1 0.071, 8<-2 0.029, 8<-4 0.018, 8<-8 0.007 |
| P8-source | S | 73129 | 0.558 | 1<-1 0.248, 1<-2 0.128, 1<-4 0.063, 1<-8 0.061, 2<-1 0.130, 2<-2 0.061, 2<-4 0.029, 2<-8 0.029, 4<-1 0.061, 4<-2 0.031, 4<-4 0.016, 4<-8 0.017, 8<-1 0.059, 8<-2 0.036, 8<-4 0.016, 8<-8 0.014 | 1<-1 0.263, 1<-2 0.105, 1<-4 0.068, 1<-8 0.064, 2<-1 0.139, 2<-2 0.059, 2<-4 0.022, 2<-8 0.030, 4<-1 0.071, 4<-2 0.029, 4<-4 0.014, 4<-8 0.011, 8<-1 0.069, 8<-2 0.036, 8<-4 0.014, 8<-8 0.007 |
| P8-source | S | 93683 | 0.581 | 1<-1 0.242, 1<-2 0.130, 1<-4 0.067, 1<-8 0.061, 2<-1 0.130, 2<-2 0.059, 2<-4 0.029, 2<-8 0.032, 4<-1 0.067, 4<-2 0.033, 4<-4 0.014, 4<-8 0.011, 8<-1 0.068, 8<-2 0.032, 8<-4 0.012, 8<-8 0.013 | 1<-1 0.245, 1<-2 0.115, 1<-4 0.072, 1<-8 0.068, 2<-1 0.131, 2<-2 0.061, 2<-4 0.031, 2<-8 0.027, 4<-1 0.073, 4<-2 0.032, 4<-4 0.014, 4<-8 0.007, 8<-1 0.073, 8<-2 0.035, 8<-4 0.010, 8<-8 0.007 |
| P8 | B | 116089 | 0.465 | 1<-1 0.258, 1<-2 0.122, 1<-4 0.059, 1<-8 0.061, 2<-1 0.132, 2<-2 0.058, 2<-4 0.032, 2<-8 0.028, 4<-1 0.066, 4<-2 0.027, 4<-4 0.017, 4<-8 0.015, 8<-1 0.065, 8<-2 0.028, 8<-4 0.018, 8<-8 0.014 | 1<-1 0.263, 1<-2 0.116, 1<-4 0.063, 1<-8 0.057, 2<-1 0.138, 2<-2 0.056, 2<-4 0.030, 2<-8 0.026, 4<-1 0.069, 4<-2 0.028, 4<-4 0.015, 4<-8 0.014, 8<-1 0.071, 8<-2 0.025, 8<-4 0.020, 8<-8 0.008 |
| P8 | B | 73129 | 0.459 | 1<-1 0.248, 1<-2 0.128, 1<-4 0.063, 1<-8 0.061, 2<-1 0.130, 2<-2 0.061, 2<-4 0.029, 2<-8 0.029, 4<-1 0.061, 4<-2 0.031, 4<-4 0.016, 4<-8 0.017, 8<-1 0.059, 8<-2 0.036, 8<-4 0.016, 8<-8 0.014 | 1<-1 0.249, 1<-2 0.115, 1<-4 0.069, 1<-8 0.068, 2<-1 0.137, 2<-2 0.055, 2<-4 0.025, 2<-8 0.034, 4<-1 0.062, 4<-2 0.030, 4<-4 0.018, 4<-8 0.016, 8<-1 0.070, 8<-2 0.032, 8<-4 0.012, 8<-8 0.011 |
| P8 | B | 93683 | 0.462 | 1<-1 0.242, 1<-2 0.130, 1<-4 0.067, 1<-8 0.061, 2<-1 0.130, 2<-2 0.059, 2<-4 0.029, 2<-8 0.032, 4<-1 0.067, 4<-2 0.033, 4<-4 0.014, 4<-8 0.011, 8<-1 0.068, 8<-2 0.032, 8<-4 0.012, 8<-8 0.013 | 1<-1 0.254, 1<-2 0.111, 1<-4 0.066, 1<-8 0.069, 2<-1 0.141, 2<-2 0.051, 2<-4 0.030, 2<-8 0.028, 4<-1 0.073, 4<-2 0.030, 4<-4 0.011, 4<-8 0.012, 8<-1 0.072, 8<-2 0.030, 8<-4 0.013, 8<-8 0.011 |
| P8 | S | 116089 | 0.490 | 1<-1 0.258, 1<-2 0.122, 1<-4 0.059, 1<-8 0.061, 2<-1 0.132, 2<-2 0.058, 2<-4 0.032, 2<-8 0.028, 4<-1 0.066, 4<-2 0.027, 4<-4 0.017, 4<-8 0.015, 8<-1 0.065, 8<-2 0.028, 8<-4 0.018, 8<-8 0.014 | 1<-1 0.277, 1<-2 0.104, 1<-4 0.061, 1<-8 0.057, 2<-1 0.145, 2<-2 0.052, 2<-4 0.029, 2<-8 0.025, 4<-1 0.068, 4<-2 0.023, 4<-4 0.020, 4<-8 0.014, 8<-1 0.066, 8<-2 0.030, 8<-4 0.018, 8<-8 0.011 |
| P8 | S | 73129 | 0.491 | 1<-1 0.248, 1<-2 0.128, 1<-4 0.063, 1<-8 0.061, 2<-1 0.130, 2<-2 0.061, 2<-4 0.029, 2<-8 0.029, 4<-1 0.061, 4<-2 0.031, 4<-4 0.016, 4<-8 0.017, 8<-1 0.059, 8<-2 0.036, 8<-4 0.016, 8<-8 0.014 | 1<-1 0.261, 1<-2 0.110, 1<-4 0.066, 1<-8 0.063, 2<-1 0.135, 2<-2 0.051, 2<-4 0.032, 2<-8 0.032, 4<-1 0.068, 4<-2 0.027, 4<-4 0.015, 4<-8 0.014, 8<-1 0.063, 8<-2 0.037, 8<-4 0.016, 8<-8 0.009 |
| P8 | S | 93683 | 0.502 | 1<-1 0.242, 1<-2 0.130, 1<-4 0.067, 1<-8 0.061, 2<-1 0.130, 2<-2 0.059, 2<-4 0.029, 2<-8 0.032, 4<-1 0.067, 4<-2 0.033, 4<-4 0.014, 4<-8 0.011, 8<-1 0.068, 8<-2 0.032, 8<-4 0.012, 8<-8 0.013 | 1<-1 0.249, 1<-2 0.126, 1<-4 0.062, 1<-8 0.062, 2<-1 0.135, 2<-2 0.052, 2<-4 0.033, 2<-8 0.030, 4<-1 0.064, 4<-2 0.035, 4<-4 0.015, 4<-8 0.011, 8<-1 0.073, 8<-2 0.034, 8<-4 0.011, 8<-8 0.007 |
| W10 | S | 116089 | 0.547 | 1<-1 1.000 | 1<-1 1.000 |
| W10 | S | 73129 | 0.548 | 1<-1 1.000 | 1<-1 1.000 |
| W10 | S | 93683 | 0.529 | 1<-1 1.000 | 1<-1 1.000 |

