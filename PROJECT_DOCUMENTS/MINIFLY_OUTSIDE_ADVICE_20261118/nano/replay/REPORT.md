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

**An erratum and what it changes** (`PLAN_1.md`, sections 12 and 13; `report_clean.txt`). The coordinating session found that a subtraction line is written with the larger number first, so that about four never-shown subtraction pairs in five write a line that training also writes; addition is not touched. Every subtraction figure in the table above is that mixed score. Scored on the never-shown pairs whose line no shown pair writes (about 400 of the 2,000 never shown), subtraction at the end is, on fifteen seeds: plain 85.8, replay 77.5, replay at half the ratio 81.6, the program's old lines 73.3 (mixed: 86.0, 78.4, 81.6, 75.4). The difference inside a run is half a point at the median. The hypothesis's second clause holds on the clean score (78.9 on the five seeds), and items 1 to 5 stand; on the clean score replay is 4.2 points above the program's old lines on subtraction (ten seeds of fifteen), which I do not claim.

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

## 3. A harsh change of task that brings no new byte (`PLAN_3.md`, `report3.txt`)

Asked by the coordinating session after section 2. Addition (`aa+bb=ccc`) for 400,000 real lines, then subtraction written with the bytes of addition (`aa=bb+ccc` for aa - bb = ccc; the text of `nano/keep`'s `PLAN_8.md`) for 200,000. Never-shown pairs; the second task scored on pairs whose written line no shown pair writes (about 400). Medians.

| | addition kept (five seeds) | second task (five) | addition kept (fifteen) | second task (fifteen) |
|---|---|---|---|---|
| plain, no replay | 1.4 | 82.5 | | |
| replay, one line for each real line | **65.6** | **68.0** | **67.8** | **68.6** |
| control: the program's old lines, one for each real line | 80.8 | 65.8 | 78.6 | 64.1 |
| control: replay with the cells emptied at the change | 0.8 | 88.9 | | |

1. **The hypothesis holds**: replay keeps 65.6% of never-shown addition (every seed above 50) where the plain system keeps 1.4%, through a change that brings no new byte value. Every replayed line is correct.
2. **This change is dearer than a change with a new operator, by either route**: the two scores sum to about 138 to 141 here against about 156 in section 1.
3. **Here replay keeps a little less of the old task than the program's old lines and gives a little more of the new** (paired on fifteen seeds: -4.6, fourteen seeds below; +7.8, ten above); the sums do not differ. A replay draws from everything counted, so three replayed lines in ten are of the second task by the end.
4. Two reservations: at the bytes the system learns from, the second text activates two lines that addition never trained (`=` four bytes back, `+` seven back), so "no new line" holds for byte values and not strictly for lines in this system; and the counting cells are a complete store of the lines shown.

## 4. Dreams: the relay system writes its answers to never-shown sums into the store (`PLAN_4.md`, `report4.txt`)

Asked by the coordinating session. One change from section 1's replay arm: one extra line in ten is a dream. Its two numbers are drawn from short counting cells and paired by chance, so that pairs never shown occur; the relay system writes the answer; the line is counted by the counting cells like a real one; the relay system does not learn from it then, but later replays draw from everything counted. Addition, then subtraction, as in section 1. Medians of five seeds.

| | never-shown addition at the end, the relay system | never-shown addition, the store alone (of 500 held pairs) | clean subtraction, the relay system |
|---|---|---|---|
| replay, no dreams | 77.2 | 0.0 | 78.9 |
| dreams | 77.0 | **84.8** | 80.1 |
| dreams only after 300,000 lines (addition of mine) | 75.2 | 76.0 | 77.2 |
| control: dreams answered by chance | 76.6 | 0.2 | 79.4 |

1. **The hypothesis fails**: dreams do not raise what the relay system keeps (+0.8 paired on five seeds; -2.2 on fifteen, nine seeds below and six above).
2. **The store, by itself, answers 84.8% of the never-shown sums** (83.6% of all 2,000), near the relay system's level at the end of the addition phase (86.9) and above its level at the end of the life (ahead by 4.4 points at the median of fifteen seeds, behind on none); and the store loses nothing through subtraction (83.8 before, 84.8 after). What the relay system generalised is held in a form that does not forget.
3. **Errors do not grow**: 16.5% of the dreamed sums are wrong and stay; wrong sums are 2.2% of replays early and 1.5 to 1.6% from the middle of the addition phase to the end.
4. **The price**: the store grows to hold nearly the whole domain (a line for 99.6% of the never-shown sums beside all shown ones; 84,909 cells against 66,272).
5. Unexpected: dreams answered by chance do not harm the relay system (9% of its replays are then wrong sums, and its score is unchanged), so its score says little about what is in the store; and dreams that begin late do worse than dreams from the start.

## 5. Limits, commands, cost

One testbed; one length; a store that is complete; replay mixed into the stream at a fixed ratio; ratios chosen on one seed. Not run: counting cells that fade or are limited in number; K between 6 and 8; replay in blocks; ratios below 1/2 on the reported seeds (seed 1: 67.6 at 1/4, 46.0 at 1/8); the ten further seeds for section 2.

`python3 check_replay.py`, `python3 check_replay2.py`; `xargs -P 4 -L 1 sh -c 'python3 replay.py "$0" "$@"' < dev1.txt` and likewise `rep1.txt`, `num1.txt`, `add1.txt`; `rep2.txt` with `replay2.py`; `python3 report1.py > report1.txt`, `python3 report2.py > report2.txt`. A run at one replayed line for each real line takes about 60 seconds of one core (27 for the plain system), of which 13 are counting and replay in Python. 81 compiled runs and one numpy run at reduced length for section 1 (and 70 reruns with the clean subtraction score: `rerun.txt`, `python3 report_clean.py > report_clean.txt`), 30 compiled runs for section 2, 44 for section 4 (`dev4.txt`, `rep4.txt` with `replay4.py`; `python3 check_replay4.py`; `python3 report4.py > report4.txt`), 44 for section 3 (`rep3.txt`, `add3.txt` with `replay3.py`; `python3 check_replay3.py`; `python3 report3.py > report3.txt`): about an hour and a half of CPU. Compiled `.so` files are not committed.
