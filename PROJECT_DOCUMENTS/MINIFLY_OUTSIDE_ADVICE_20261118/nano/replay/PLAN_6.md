# Keeping by replay, sixth plan: the replayed line chosen by the relay system's own mismatch (a repeat of another line's rule) (20 November 2026)

By the same separate agent session as `PLAN_1.md` to `PLAN_5.md`; same folder, same source. Written before any run of this step; the check of the program was run first (`check_replay6_output.txt`; lives of 900 lines). Development level; repeated by no one; no checker gate, seal or review is claimed.

**Authorisation and origin.** Asked by the coordinating session in a message of 20 November. The rule is the line `nano/life3`'s (its `PLAN_6.md` and `REPORT.md` on `claude/nano-life3-8mmhfm` at `b96cd74`), which, the coordinating session says, asks to have it repeated by another session before anything is entered. As asked, I have read neither its program nor, so far, its files: I have the rule from the coordinating session's message alone, and the program here (`replay6.py`) is written from that sentence. The coordinating session reports that line's figures as: three tasks in one life, replay as I built it 73.4 / 52.4 / 64.2, with the rule 67.8 / 66.9 / 63.4. I have not rerun those; this plan is on two tasks.

## 1. The rule, as I have built it

Arm R of `PLAN_1.md` (after each real line, one line drawn from the counting cells, K = 8, and learned). **Arm M:** after each real line four lines are drawn from the counting cells exactly as R draws one; the relay system learns only from the one on which its own squared mismatch is largest; the other three are dropped. Nothing names a task.

What the message leaves open and I fix here, stated because the coordinating session asks:

- *The mismatch of a line:* the sum, over the nine bytes of the line from which the system learns and over the fourteen branches of the answering cell, of the squared mismatch, with the line read alone after a newline (the lines before it are not in the delay lines when it is judged, though they are when it is learned).
- *Whose strengths:* the life is read in chunks of 500 real lines; the four candidates of every real line in a chunk are drawn and judged at the start of the chunk's learning, with the relay system's strengths as they then are. So the mismatch is taken with strengths at most 500 real lines old, not those of the moment. The counting cells are of the moment (a candidate is drawn after its real line is counted).
- Judging changes nothing in the relay system: no strength changes when a line is judged.

## 2. The two lives and the arms

(a) `add:400000,sub:200000`, the life of `PLAN_1.md`. (b) `add:400000,swp:200000`, the harsh change of `PLAN_3.md` (subtraction written `aa=bb+ccc`). Arms R and M on each, five reported seeds; seed 1 as a development pass, nothing chosen on it (there is nothing to choose: four candidates, as given). R is made again in this program (the check: the same strengths as the earlier programs' arm R after the same life, difference 0.0), so that both arms carry the same scores.

**Measures:** never-shown addition at the end (the 500 held pairs); the second task at the end on the clean score (never-shown pairs, of all 2,000, whose line no shown pair writes), the mixed score beside it; by reading of the second phase, the share of first-task lines among the lines drawn and among the lines learned from (does the choice move towards a task, and which?); the mean mismatch of the chosen line against the drawn ones; the cost: lines judged for lines learned, and CPU.

## 3. What the coordinating session expects, and what would count against it

Its predictions: (a) on two tasks, M within 3 points of R on both tasks; (b) on the harsh change, a gain of at least 5 points on whichever task is lower. I score both at the median of the paired differences over the five reported seeds; in (b) the lower task is the one that is lower in arm R at the median.

## 4. Predictions (mine), with chances

My expectation of the mechanism, before any run: while the second task is being learned its lines have the larger mismatch, so the rule will choose them more often than they are drawn, the old task will be rehearsed less than under R early in the second phase, and its mismatch will then rise until the choice turns back; the rule should even out the mismatch of the two tasks, not the scores.

- M1 (0.5): (a) holds: M within 3 points of R on both tasks.
- M2 (0.6): in (a), addition at the end is lower under M than under R at the median of the paired differences, and subtraction (clean) higher.
- M3 (0.35): (b) holds: a gain of at least 5 points on the lower task.
- M4 (0.7): in both lives, over the second phase, the share of first-task lines among the lines learned from is lower than among the lines drawn in the first reading (the choice goes to the new task first).
- M5 (0.6): in both lives that share is higher in the last reading than in the first (the choice moves back towards the task that is slipping).
- M6 (0.8): the mean mismatch of the chosen lines is at least 1.5 times that of the drawn ones in the second phase.

## 5. Cost, not run

Twenty-four compiled runs (seed 1 and the five seeds, two lives, two arms). Arm M draws four lines and judges four for each one learned; the judging is a numpy pass over the candidates of a chunk. I expect 90 to 150 seconds a run for M against 65 for R: about 12 minutes on four cores. The ten further seeds where a prediction of section 3 is within its margin by less than five points and time allows; if time does not allow, NOT RUN and said so. Not run: three tasks in one life (that line's own experiment); other numbers of candidates; the mismatch taken with the strengths of the moment; the smallest mismatch, or a choice by chance among four (which is arm R).

---
