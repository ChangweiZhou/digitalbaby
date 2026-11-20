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
