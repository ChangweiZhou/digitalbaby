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

## 9. Entered 20 November after the check and the development runs, before any run on the reported seeds

**The check** (`check_replay_output.txt`): passed. The relay system's compiled loop against its numpy loop in every arm over two phases: 1e-14 to 3e-14, the same extra lines. The plain arm against `cells2.py` over two phases (numpy against numpy): 2e-15. The counting cells: with K = 8 (and 10) all of 30,000 replayed lines are lines that were counted, each in proportion to how often it was counted; with K = 6, 9,283 of 30,000 (after 3,000 counted lines).

**Development seed 1** (`res/p_d_*.json`). Never-shown addition after its own phase (mean of the last three readings); then never-shown addition and subtraction after 200,000 lines of subtraction (last reading; in brackets the mean of the last three).

| arm | addition after its phase | addition after subtraction | subtraction | CPU s (of which counting and replay) |
|---|---|---|---|---|
| P | 81.5 | 1.4 (1.9) | 82.2 (78.7) | 32 |
| R, r = 1/8 | 80.0 | 46.0 (48.6) | 85.8 (81.7) | 37 (7) |
| R, r = 1/4 | 89.6 | 67.6 (67.1) | 82.8 (79.6) | 44 (8) |
| R, r = 1/2 | 88.3 | 76.0 (79.0) | 81.0 (77.9) | 48 (10) |
| R, r = 1 | 89.9 | 79.0 (81.0) | 82.2 (76.8) | 61 (14) |
| R, r = 2 | 85.1 | 78.4 (78.9) | 75.2 (69.8) | 106 (25) |
| O, r = 1 | 88.1 | 86.0 (84.1) | 76.4 (73.3) | 54 (4) |
| Z, r = 1 | 89.9 | 1.4 (1.7) | 90.4 (89.3) | 65 (15) |
| R6, r = 1 | 67.7 | 3.4 (2.3) | 68.2 (65.3) | 68 (16) |

Counting cells held at the end in R: 66,050, with 78,887 counts above zero (16,000 different lines were shown). In R every replayed line is correct at every ratio; in R6 23,492 of 600,000 are (3.9%: with all 16,000 lines counted, a cell that does not see the first operand's tens digit has met nearly every continuation, so the share of right sums falls far below the check's third).

**Choice by the rule of section 5:** the grid's best is r = 1 (79.0); the smallest r within 5 points of it is **r = 1/2** (76.0). To the five reported seeds: P; R at r = 1 (the hypothesis) and at r = 1/2; O, Z and R6 at r = 1.

A7 failed on this seed (46.0 at r = 1/8, under 50). Two things I note before the reported seeds, so that they are not made afterwards: with replay the score of addition after its own phase is 6 to 8 points above P's on this seed at r = 1/4 to 1 (the first phase has more lines), and R6's is 14 points below it (replayed wrong sums harm the first task while it is being learned).

## 10. A dated addition, entered 20 November after the five reported seeds were read and before its runs

On the five reported seeds (scored in section 11) R at r = 1 is 7.6 points below O on addition at the median of the paired differences, with single differences from -11.2 to +7.4, and R at r = 1/2 is 4.2 below R at r = 1. Both differences matter for the reading (what replay from the system's own cells costs against old lines chosen by the program; whether half the ratio is enough) and five seeds do not carry them. By the coordinating session's rule the ten further seeds 1001 to 1010 are run for P, R at r = 1, R at r = 1/2 and O, and the fifteen are reported together.
- B1 (0.6): on fifteen seeds R at r = 1 is below O on addition by 2 to 10 points at the median of the paired differences.
- B2 (0.6): R at r = 1/2 is below R at r = 1 on addition on at least nine seeds of fifteen.
- B3 (0.7): R at r = 1 is below P on subtraction on at least eleven seeds of fifteen (keeping addition costs some subtraction).

## 11. Outcome (entered 20 November, after all runs of this plan)

Tables: `report1.txt` (made by `report1.py` from `res/p_*.json`): first the five reported seeds, then the fifteen. The numpy loop beside the compiled one (R, r = 1, seed 1, `add:40000,sub:20000`): the same six readings to the digit.

**The five reported seeds** (73129, 93683, 116089, 406364, 545353; median). Never-shown sums, whole answer right, %, at the last reading after 200,000 lines of subtraction.

| | addition after its own phase | addition after subtraction | subtraction |
|---|---|---|---|
| P, no replay | 84.2 | 2.0, 1.6, 1.8, 1.4, 2.0; **1.8** | 82.0, 89.8, 91.4, 86.0, 86.0; **86.0** |
| R, one replayed line for each real line | 86.3 | 77.2, 85.0, 77.2, 75.6, 79.4; **77.2** | 82.2, 82.6, 74.4, 76.8, 79.6; **79.6** |
| R, one for two real lines | 84.9 | 80.6, 77.4, 73.0, 71.2, 72.0; **73.0** | 75.2, 88.6, 82.8, 81.6, 82.6; **82.6** |
| O, the program's old lines, one for each (control) | 86.1 | 88.4, 77.6, 75.0, 84.4, 87.0; **84.4** | 75.4, 69.0, 76.8, 75.8, 76.0; **75.8** |
| Z, replay, cells emptied at the change (control) | 86.3 | 2.4, 1.4, 1.8, 1.6, 2.4; **1.8** | 86.6, 91.2, 88.0, 89.0, 88.2; **88.2** |
| R6, replay from cells of six bytes | 65.7 | 2.6, 3.2, 2.0, 4.2, 3.6; **3.2** | 79.2, 66.6, 75.4, 81.8, 63.2; **75.4** |

**The hypothesis holds.** With one replayed line for each real line, never-shown addition after 200,000 lines of subtraction is 77.2% (75.6 to 85.0 on the five seeds) and never-shown subtraction 79.6%, where the plain system keeps 1.8% of addition, and the program's one old line in 32 kept 15.6 in the earlier runs. The program shows no old line and is not told when the task changes.

**Fifteen seeds** (the five, then 1001 to 1010), medians: P 1.8 and 86.0; R at r = 1: **79.4 and 78.4**; R at r = 1/2: 74.2 and 81.6; O: 80.8 and 75.4. Paired:

- R at r = 1 less O: addition -0.2 (seven seeds above, eight below), subtraction +1.0 (nine above, six below). **Replay from the system's own counting cells keeps as much as old lines chosen by the program.** The 7.6 points between them on the five reported seeds do not survive the fifteen.
- R at r = 1/2 less R at r = 1: addition -4.4 (below on twelve of fifteen), subtraction +4.0 (above on thirteen). Half the replay keeps a little less of the old task and a little more of the new; the sum of the two scores is the same at the median (156.4 and 155.8; O 156.0). The ratio moves the system along a trade between the two tasks; it does not buy both.
- R at r = 1 less P on subtraction: -9.2 (below on twelve of fifteen); O less P: -10.0 (below on fifteen). Keeping addition costs about 9 points of subtraction at this length, with replay or with the program's old lines alike.

**What else the runs show (my readings).**

1. *It is the old lines that keep.* With the cells emptied at the change (Z), the same number of replayed lines, all of the new task, leaves addition at 1.8%, as with no replay.
2. *Replay from cells that do not cover a line destroys rather than keeps.* With cells of six bytes 3.9% of the replayed lines are right sums (all are well formed); addition after subtraction is 3.2%, and addition after its own phase is 65.7% where the plain system has 84.2: wrong replayed sums harm the first task while it is being learned. A store that is smaller than the lines, made this way, is worse than no store.
3. *The store is complete, and that is what the result rests on.* 66,050 counting cells with 78,887 counts above zero hold the 16,000 lines shown; every replayed line in R is a line that was shown (100.00% correct on every seed). Section 3 said this beforehand: the experiment shows that rehearsal from a store the system fills by counting keeps the first task as well as rehearsal by the program, at one replayed line for each real one; it does not show keeping from anything smaller than the lines themselves. The counting cells are 66,050 against the relay system's 256, and their counts are 1.8 times the relay system's 43,278 numbers.
4. *Cost.* A run at r = 1 takes 60 seconds of one core against 27 for the plain system: 13 seconds of counting and replay in Python, the rest the doubled number of lines.
5. The share of replayed lines that are addition falls through the second phase from 94% to 69% (seed 73129), as the counts of subtraction grow; nothing sets it but the counts.

**Predictions.** A1 held (the ten figures to the digit). A2 held. A3 held (1.8). A4 failed on the five seeds (-7.6), and the fifteen say the two are level (-0.2). A5 held. A6 held (-75.2). A7 failed (46.0 on seed 1). A8 held (-0.4 on five seeds; +1.3 on fifteen). B1 failed (-0.2, not 2 to 10 below). B2 held (twelve of fifteen). B3 held (twelve of fifteen).

**Errors and departures of mine.** Section 10 gives the difference between the two ratios as 4.2; it is 4.4. Z and R6 were not run on the ten further seeds (their differences are 75 points and more). The first phase of R, O, Z and R6 has more lines than P's; on fifteen seeds that is worth +1.3 points of addition after its own phase.

**Not run:** as section 8; r below 1/2 on the reported seeds (seed 1: 67.6 at 1/4, 46.0 at 1/8).

## 12. A dated addition after an erratum of the coordinating session, entered 20 November before its runs

**The erratum** (the coordinating session's message; its file `loop_small_case/ERRATA_20261120.md` on `claude/friendly-darwin-8mmhfm`, which I have not read; the fault itself I checked here). `cells2.py`'s `text()`, which `replay.py` copies, writes a subtraction line with the larger number first, so a never-shown pair (a, b) writes the same line as the pair (b, a), which may be shown. On seeds 1, 73129 and 1001, 407, 399 and 393 of the 500 never-shown subtraction pairs write a line that a shown pair also writes; of the 500 never-shown addition pairs, none. **Every subtraction figure in sections 9 and 11 is therefore mostly a score on lines that were shown; the addition figures, and with them the hypothesis's first clause, are not touched.** The hypothesis's second clause (subtraction at least 60%) was scored in section 11 on the mixed set and has to be scored again.

**The addition.** `replay.py` now scores, beside the present subtraction score (kept under its name, "subtraction, mixed"), the never-shown subtraction pairs whose written line no shown pair writes: those among the 500 ("clean", about 100), and those among all 2,000 never-shown pairs ("clean, wide", about 400, the steadier figure). Nothing else in the program changes; the runs are deterministic, so the reruns must give every earlier figure again, and `report1.py` checks that they do. Rerun: P, R at r = 1 and 1/2, O, Z, R6 on the five reported seeds, and P, R at both ratios and O on 1001 to 1010 (70 runs, about 20 minutes on four cores).

For replay the clean score bears on one thing more, as the coordinating session says: every replayed line is a shown line, so only the clean score says whether the relay system, taught half on replay, answers subtraction it was not shown.

- E1 (0.5): P's clean subtraction (wide) is at least 60% at the median of the five reported seeds.
- E2 (0.6): the second clause of the hypothesis holds on the clean score too: R at r = 1 has at least 60% (clean, wide; median of five).
- E3 (0.6): R at r = 1 is below P on clean subtraction (wide) on at least eleven seeds of fifteen.
