# Keeping by replay from the system's own counting cells: first plan (20 November 2026)

By a separate agent session (the one that ran line C in `nano/joint`). Folder: `nano/replay/`; nothing outside it is changed. Written before any run of this step; the check of the programs was run first (`check_replay_output.txt`; it trains nothing beyond 300 lines). Development level; repeated by no one; no checker gate, seal or review is claimed.

**Source.** Branch `claude/nano-joint-8mmhfm` (based on `claude/friendly-darwin-8mmhfm` at `fe397ca`). Copied from `loop_small_case/` at `fe397ca`: from `cells2.py` the text, the split, the streams of random numbers, the plain system and the scoring (in `replay.py`); `kern3.c` unchanged as `kernr.c`; the build and call of `fastk.py` and `fastk3.py` (in `fastr.py`). Read, not copied, on `claude/friendly-darwin-8mmhfm` at `0a20bb4`: `nano/table/PLAN_1.md` and `table.py`; `loop_small_case/PLAN_13.md`; `OLD_LISTS_AGAINST_THE_LOOP_20261120.md`.

**Authorisation.** The line, its question, its hypothesis and its controls were given to this session by the coordinating session in a message of 20 November, which relays Mr. Xu as having said since: "Keep experimenting with the agents." I have that sentence only through the coordinating session. The idea is the coordinating session's ("mine, yours to improve or refute", in its words); what I changed or added is marked as mine below.

## 1. The evidence this follows (as the coordinating session gave it, with what I checked)

The relay system answers never-shown sums (84.2%) but after 200,000 lines of subtraction keeps 1.8% of addition (2.0, 1.6, 1.8, 1.4, 2.0 on the five seeds; `loop_small_case/report4.txt`, which I read); with one old line in 32 shown by the program, 15.6; with that and 128 new cells, 67.0. `PLAN_13.md` there: the cells in between still carry most of addition; the loss is in the answering cell's reading. `nano/table`: a table made of cells that count holds what it was shown exactly and its counts only grow, but it answers nothing it was not shown.

## 2. The question

Two kinds of cells side by side: the relay system, which generalises and forgets, and cells that count, which do neither. If, between real lines, the system replays lines made from its own counting cells and the relay system learns from them as from real lines, **does it keep addition through subtraction, with no old line shown by the program?**

## 3. The system

*The relay system:* the plain system of `cells2.py`, unchanged (256 cells in between; 43,278 numbers that learn).

*The counting cells.* A cell for a run of 1 to K bytes that has occurred in the real lines; when its run is the last bytes read and a byte arrives, its count for that byte grows by one. K = 8: the line `aa+bb=ccc` needs the eight bytes `aa+bb=cc` to fix its last digit. Counts are made on real lines only.

*Replay.* A replayed line begins at a line start (a newline with nothing before it). Each next byte is drawn, in proportion to its counts, from the longest active cell that has counted; the line stops at the newline. Nothing is counted during replay. After each real line there are r replayed lines (the whole part of r, and one more with a chance of its fractional part). Replay goes on through both tasks alike: the system is not told when the task changes.

*What the program holds, said plainly (mine).* The program holds only the cells that a replay can use: for each byte of a line, the longest cell that does not reach back beyond the newline before the line. Shorter cells, and cells whose run crosses from one line into the next, exist by the definition but are never the longest cell that has counted in a replay that begins at a newline; they are not held and not counted in the cost. And: **with K = 8 the counting cells are a complete store of the lines shown.** Every replayed line is a line that was shown, and lines come back in proportion to how often they were shown (the check: 30,000 of 30,000, shares equal to 0.0006). There is no list of lines in the program, but the counts hold the same information as one, and a replay is, in its effect, a draw of one earlier real line. The question the experiment answers is therefore whether rehearsal from a store the system builds by its own counting keeps the first task, at what ratio and cost; it does not show a store smaller than the lines. The arm with K = 6 (below) is the first step towards a store that is not complete.

## 4. Arms

- **P:** the plain system, no replay (must reproduce 2.0, 1.6, 1.8, 1.4, 2.0 of addition and 82.0, 89.8, 91.4, 86.0, 86.0 of subtraction after the second phase).
- **R:** replay at ratio r, K = 8.
- **O** (control, the upper reference): in place of each replayed line, a real line of the first task drawn by the program from the 8,000 addition lines it shows, at the same ratio, through both phases.
- **Z** (control): as R, with the counting cells emptied by the program when the task changes, so that replay in the second phase holds only the second task. It has R's number of lines.
- **R6** (an addition of mine): as R with K = 6. The cells then do not cover a line: the tens and units of the answer are drawn from cells that do not see the first operand's tens digit, and (the check) about two replayed lines in three are wrong sums, all of them well formed. Does replay of partly wrong lines keep anything?

In R, O, Z and R6 the first phase also has its extra lines, so the four have the same number of lines in both phases; P has fewer. The score of addition after its own phase is reported for every arm.

Plan `add:400000,sub:200000` (real lines), as in the earlier keeping runs. **Measures:** never-shown addition and never-shown subtraction (whole answer right on 500 pairs each) at the last reading of the second phase, as in `report4.txt`, and the mean of its last three readings beside it; by reading of the second phase, how many replayed lines are addition and subtraction, well formed, correct; cells, counts above zero; CPU, and the part of it spent counting and replaying.

## 5. Settings

One number, r, from a grid stated here: {1/8, 1/4, 1/2, 1, 2} on development seed 1, arm R. The hypothesis is the coordinating session's and is stated at r = 1, so r = 1 goes to the five reported seeds whatever the grid says; **beside it the smallest r of the grid whose never-shown addition after subtraction on seed 1 is within 5 points of the grid's best** (the cheapest that works), if it is not 1. O, Z and R6 are run at r = 1. On seed 1: P, the five R, and O, Z, R6 at r = 1.

## 6. Hypothesis and what would refute it (the coordinating session's)

**With one replayed line for each real line, never-shown addition after 200,000 lines of subtraction is at least 50% and never-shown subtraction at least 60%, at the median of the five reported seeds.** Refuted if addition is under 50%.

## 7. Predictions (mine), with chances

- A1 (0.85): P reproduces the ten figures above to the digit.
- A2 (0.8): the hypothesis holds (addition at least 50 and subtraction at least 60 at the median, r = 1).
- A3 (0.9): Z leaves addition under 10% at the median.
- A4 (0.7): R is within 5 points of O on addition at the median of the paired differences.
- A5 (0.95): in R every replayed line is well formed and correct on every seed.
- A6 (0.5): R6 leaves addition at least 20 points below R at the median of the paired differences.
- A7 (0.35): r = 1/8 already keeps at least 50% of addition on seed 1.
- A8 (0.6): at r = 1, R's addition after its own phase is within 3 points of P's at the median (the extra lines of the first phase neither help nor harm).

## 8. Cost, not run

A run at r = 1 reads 1,200,000 lines: about 40 seconds of the kernel and, by the check's timing, some tens of seconds of counting and replay in Python, on one core. Development: 9 runs; reported: 5 or 6 arms on five seeds; in all about 20 minutes on four cores. One numpy run of R at reduced length (`add:40000,sub:20000`) beside the compiled one on seed 1, since a full numpy run of 1,200,000 lines would take about 12 minutes; the full-length figure of P is checked against `cells2.py`'s published figures instead. The ten further seeds 1001 to 1010 are added where two arms are within five points and the difference matters.

Not run in this plan: a change of task that brings no new line (the coordinating session's later step); counting cells that fade or are limited in number; replay in blocks with the real lines paused; replay together with new cells; K between 6 and 8.

---
