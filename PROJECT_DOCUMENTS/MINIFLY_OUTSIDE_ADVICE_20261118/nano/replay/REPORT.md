# `nano/replay`: keeping by replay from the system's own counting cells. Report of 20 November 2026

By a separate agent session (the one that ran line C in `nano/joint`); the line, its question and its controls were given by the coordinating session, whose idea it is. Development level; repeated by no one; no checker gate, seal or review is claimed. Readings are mine. Source and what was copied: `PLAN_1.md`.

**The system.** Two kinds of cells on the sums text. The relay system of `loop_small_case` (256 cells in between, one answering cell), unchanged. And cells that count: a cell for a run of up to 8 bytes that has occurred in the real lines, counting what follows it. After each real line the system replays r lines: it begins at a line start and draws each next byte, in proportion to its counts, from the longest active counting cell, until the newline; the relay system learns from a replayed line as from a real one; nothing is counted during replay. The program shows no old line and does not tell the system when the task changes.

## 1. Addition, then subtraction (`PLAN_1.md`, `report1.txt`)

400,000 real lines of addition, then 200,000 of subtraction. Never-shown sums, whole answer right, %, at the end; medians of the five reported seeds, and of fifteen seeds (the five, then 1001 to 1010) where run.

| | addition kept (five) | subtraction (five) | addition kept (fifteen) | subtraction (fifteen) |
|---|---|---|---|---|
| plain, no replay | 1.8 | 86.0 | 1.8 | 86.0 |
| replay, one line for each real line | **77.2** | **79.6** | **79.4** | **78.4** |
| replay, one line for two real lines | 73.0 | 82.6 | 74.2 | 81.6 |
| control: the program's old lines, one for each real line | 84.4 | 75.8 | 80.8 | 75.4 |
| control: replay with the cells emptied at the change | 1.8 | 88.2 | | |
| replay from cells of six bytes | 3.2 | 75.4 | | |

1. **The hypothesis holds**: replay keeps 77% of never-shown addition through subtraction (79% on fifteen seeds) with subtraction at 79%, against 1.8% without it.
2. **Replay from the system's own cells keeps as much as old lines chosen by the program**: paired on fifteen seeds, -0.2 points of addition and +1.0 of subtraction.
3. **The ratio trades one task against the other**: half the replay gives 4.4 points less addition and 4.0 more subtraction (twelve and thirteen seeds of fifteen); the sum is the same. Keeping addition costs about 9 points of subtraction at this length, by either route.
4. **It is the old lines that keep** (emptied cells: 1.8), **and they have to be right**: from cells of six bytes 4% of replayed lines are right sums, nothing is kept, and addition is harmed while it is being learned (65.7 after its own phase against 84.2).
5. **What the result rests on, said plainly**: with runs of 8 bytes the counting cells are a complete store of the lines shown (66,050 cells, 78,887 counts above zero, for 16,000 lines; every replayed line is a line that was shown, in proportion to how often). This is rehearsal from a store that the system fills by counting, not keeping from something smaller than the lines; the one arm with a smaller store failed.

## 2. A change of task that brings no new line (`PLAN_2.md`, `report2.txt`)

Addition only: 400,000 real lines of sums whose first number is below 50, then 200,000 of sums whose first number is 50 or above. Never-shown pairs of each task; medians of the five reported seeds.

| | first task kept | second task |
|---|---|---|
| plain, no replay | 63.7 | 88.4 |
| replay, one line for each real line | **92.2** | **88.7** |
| replay, one line for two real lines | 85.0 | 89.0 |
| control: the program's old lines | 84.4 | 84.5 |
| control: replay with the cells emptied at the change | 76.6 | 91.8 |

1. **Replay keeps the first task here too** (92.2 against 63.7; paired +29.6, five seeds of five), without rest on a new line; it ends above where the first task stood after its own phase (89.9).
2. **This change of task is a mild one**: the plain system keeps 63.7% and has 37% of the second task right before it is taught. The two halves share most of what they need, so there is less to keep than in section 1.
3. Replay is ahead of the program's old lines on both tasks on five seeds of five (+6.3 and +4.2); not run on further seeds, and not explained.

## 3. Limits, commands, cost

One testbed; one length; a store that is complete; replay mixed into the stream at a fixed ratio; ratios chosen on one seed. Not run: counting cells that fade or are limited in number; K between 6 and 8; replay in blocks; ratios below 1/2 on the reported seeds (seed 1: 67.6 at 1/4, 46.0 at 1/8); the ten further seeds for section 2.

`python3 check_replay.py`, `python3 check_replay2.py`; `xargs -P 4 -L 1 sh -c 'python3 replay.py "$0" "$@"' < dev1.txt` and likewise `rep1.txt`, `num1.txt`, `add1.txt`; `rep2.txt` with `replay2.py`; `python3 report1.py > report1.txt`, `python3 report2.py > report2.txt`. A run at one replayed line for each real line takes about 60 seconds of one core (27 for the plain system), of which 13 are counting and replay in Python. 81 compiled runs and one numpy run at reduced length for section 1, 30 compiled runs for section 2: about an hour and a half of CPU. Compiled `.so` files are not committed.
