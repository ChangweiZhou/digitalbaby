# Keeping by replay, second plan: a change of task that brings no new line (20 November 2026)

By the same separate agent session as `PLAN_1.md`; same folder, same source. Written before any run of this step and before the outcome of `PLAN_1.md`'s ten further seeds was read; the check of the program was run first (`check_replay2_output.txt`; 300 lines). Development level; repeated by no one; no checker gate, seal or review is claimed.

**Authorisation.** As in `PLAN_1.md`: the coordinating session's message of 20 November named this as "a later step ..., if budget remains": "a change of task that brings no new line (for instance first sums with a below 50, then sums with a of 50 and above)". It told me that another line (`nano/keep`, which I have not read) keeps the first task by a route that rests on the second task bringing a line never met before (the operator), and that replay does not rest on that. This plan tests the second half of that sentence.

## 1. The question

In `PLAN_1.md` the second task brings a line the first never used (the minus sign). Here both tasks are addition: first the sums whose first number is below 50 (3,974 of the 8,000 shown pairs on seed 3; about half on any seed), then those whose first number is 50 or above. Every byte value occurs at every position in both tasks, except that the first byte of a line is 0 to 4 in the first task and 5 to 9 in the second. **Does the plain system lose the first half when taught the second, and does replay from the counting cells keep it?**

## 2. The system and the arms

`replay.py`'s relay system, counting cells (K = 8) and replay, imported unchanged; `replay2.py` changes only the tasks. Split as before (2,000 pairs never shown); the never-shown pairs of a task are all that belong to it (about 1,000 each). Plan `lo:400000,hi:200000` (real lines).

- **P:** no replay. **R:** replay at r = 1, and at r = 1/2 beside it (the two ratios of `PLAN_1.md`; nothing is tuned here). **O** (control): a real line of the first task drawn by the program for each real line. **Z** (control): replay with the cells emptied by the program at the change.

**Measures:** whole answer right on the never-shown pairs of the first task and of the second, at the last reading of the second phase (and the mean of its last three readings); the second task's score at the end of the first phase, before it is taught; in the second phase, how many replayed lines belong to each task; cells; CPU.

## 3. Hypothesis and what would refute it

**With one replayed line for each real line, the never-shown sums of the first task are at least 50% right after 200,000 lines of the second task, at the median of the five reported seeds, and at least 20 points above the plain system.** Refuted if either fails. (If the plain system itself keeps the first task here, the second clause fails and the change of task is not one that needs keeping; that would be a finding about the task, and I would say so.)

## 4. Predictions (mine), with chances

- C1 (0.6): P keeps less than 30% of the first task at the median.
- C2 (0.75): R at r = 1 keeps at least 60% of the first task at the median.
- C3 (0.65): the hypothesis holds.
- C4 (0.6): R at r = 1 is within 8 points of O on the first task at the median of the paired differences.
- C5 (0.85): Z is within 10 points of P on the first task at the median.
- C6 (0.6): at the end of the first phase, before the second task is taught, the plain system has less than 20% of the second task's never-shown sums right.

## 5. Cost, not run

Seed 1 for the five arms as a development pass (nothing is chosen on it), then the five reported seeds: 30 compiled runs of 30 to 60 seconds, about 8 minutes on four cores; the ten further seeds where two arms are within five points and it matters. The numpy loop is the one checked in `PLAN_1.md`; no further full-length numpy run. Not run: other splits of the sums; K = 6; other ratios; the route of `nano/keep` on this change of task (it belongs to that line).

---

## 6. Outcome (entered 20 November, after all runs of this plan)

Tables: `report2.txt` (made by `report2.py` from `res/q_*.json`); the check: `check_replay2_output.txt` (passed, 1e-14 in every arm). Never-shown pairs of each task (about 1,000 each), whole answer right, %; the five reported seeds and the median.

| | first task after its own phase | second task before it is taught | first task after 200,000 lines of the second | second task at the end |
|---|---|---|---|---|
| P, no replay | 88.0 | 36.9 | 60.0, 63.7, 61.9, 68.0, 72.8; **63.7** | 89.8, 84.6, 86.3, 91.1, 88.4; **88.4** |
| R, one replayed line for each real line | 89.9 | 45.2 | 92.2, 93.3, 92.4, 90.5, 89.5; **92.2** | 87.9, 91.2, 88.7, 89.7, 87.7; **88.7** |
| R, one for two real lines | 89.9 | 41.9 | 85.0, 84.8, 88.7, 81.4, 89.8; **85.0** | 89.1, 89.0, 87.3, 87.7, 90.5; **89.0** |
| O, the program's old lines (control) | 87.8 | 45.2 | 88.5, 88.1, 84.4, 84.0, 83.2; **84.4** | 86.6, 83.4, 84.5, 87.5, 75.8; **84.5** |
| Z, replay, cells emptied at the change (control) | 89.9 | 45.2 | 70.5, 77.8, 78.6, 73.6, 76.6; **76.6** | 90.6, 92.8, 91.8, 91.4, 92.1; **91.8** |

**The hypothesis holds.** With one replayed line for each real line the first task is at 92.2% after the second (89.5 to 93.3), 29.6 points above the plain system at the median of the paired differences (16.7 to 32.1, five seeds of five). With replay the first task ends no lower than it stood after its own phase (92.2 against 89.9), and the second task is learned as well as without replay (88.7 against 88.4; paired -0.7).

**What else the runs show (my readings).**

1. *This change of task is a mild one for the plain system.* It keeps 63.7% of the first half where it kept 1.8% of addition through subtraction, and before the second half is taught at all it already has 36.9% of its never-shown sums right (45% in the arms with more lines in the first phase): what was learned on first numbers below 50 carries over in part to 50 and above. I had expected neither (C1 and C6 failed). A change of task that brings no new line is, here, also a change that shares most of what it needs with the first task, so it separates the two routes of keeping less sharply than the coordinating session's message supposed: there is less to keep.
2. *Replay keeps the first task where the task change brings no new line*: 92.2 against 63.7. That is the half of the message's sentence this plan could test; whether the route of `nano/keep` fails here is for that line.
3. *Replay is ahead of the program's old lines on both tasks* on five seeds of five (first task +6.3 at the median, second task +4.2). A reason I have not tested: a replayed line is drawn from all lines counted so far, so a third of the replays in the second phase are lines of the second task (seed 73129: 6%, 16%, 24%, 31% by reading), where the program's old lines are all of the first; and yet the first task is also higher under replay. Not run on further seeds.
4. *Emptying the cells at the change still helps a little here* (76.6 against 63.7; paired +10.5, five seeds of five): Z's replay in the second phase is of second-task lines only, so this is more lines of the second task raising what the two tasks share. In `PLAN_1.md` the same control did nothing (1.8 and 1.8).
5. Half the ratio keeps less (85.0; paired -7.2 against r = 1, four seeds of five below) at the same score on the second task.
6. Cost: 41,211 counting cells and 49,210 counts above zero for the 8,000 lines shown; every replayed line is correct.

**Predictions.** C1 failed (63.7, not under 30). C2 held (92.2). C3 held. C4 held (+6.3; I had not expected R above O). C5 failed, narrowly (+10.5). C6 failed (36.9).

**Not run:** as section 5; the ten further seeds (no pair of arms that matters for the hypothesis is within five points; R against O at +6.3 on five of five would be the one to add).
