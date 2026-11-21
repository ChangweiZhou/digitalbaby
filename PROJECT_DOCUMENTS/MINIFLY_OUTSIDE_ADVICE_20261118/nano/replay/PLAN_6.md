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

## 6. Entered 20 November: a false start, before any result of this plan was read

The 24 runs were started and stopped after ten minutes, for two faults of mine, neither touching the rule. (1) The judging of the candidates in numpy took more than twelve minutes of one core for a run of arm M, where section 5 expected 90 to 150 seconds in all; it is now compiled (`kernm.c` through `fastm.py`: the same arithmetic as the relay system's forward pass, no strength changed), and the check compares it with the numpy function on 3,000 lines (largest difference 1e-15; the same line chosen among every four). (2) Four runs of arm R had finished and failed while writing their result files (a count kept as a numpy value, which the file format refuses); corrected, and the check now writes a result of arm M. No figure of any run had been read; the partial files were deleted and all 24 runs are made again. The plan's sections 1 to 5 stand unchanged.

## 7. A dated addition, entered 20 November after the five reported seeds were read and before its runs

On the five reported seeds (scored in section 8) arm M is above arm R on both tasks in both lives: in (a) by +5.4 on addition and +3.4 on subtraction, in (b) by +8.0 on addition and +0.7 on the second task, with one seed of five the other way on addition in each life. These sit within five points of the margins that matter (3 points in (a), 5 in (b)), so arm M is run on 1001 to 1010 in both lives and the fifteen reported together. For arm R on those seeds I use the earlier runs of the same lives (`res/p_e_R1_*.json` for (a), `res/h_u_R_*.json` for (b)), which the checks show to be the same life as this program's arm R to the last digit, with the same clean score.
- M7 (0.7): on fifteen seeds M is above R on never-shown addition at the median of the paired differences in both lives.
- M8 (0.5): on fifteen seeds, in (a), M is more than 3 points above R on addition.
- M9 (0.5): on fifteen seeds, in (b), M is at least 5 points above R on addition (the lower task).

## 8. Outcome (entered 20 November, after all runs of this plan)

Tables: `report6.txt` (made by `report6.py` from `res/m_y_*.json`; arm R on 1001 to 1010 from the earlier runs of the same lives, which give this program's arm R figures exactly on the five reported seeds). The check: `check_replay6_output.txt` (passed).

Never-shown pairs, whole answer right, %, at the end of the life; the second task on the clean score. Medians of the five reported seeds, then of fifteen (with 1001 to 1010); paired differences M less R.

| | R, five | M, five | M less R, five | R, fifteen | M, fifteen | M less R, fifteen (seeds above / below) |
|---|---|---|---|---|---|---|
| (a) add then sub: addition | 77.2 | 82.6 | +5.4 | 79.4 | 81.6 | **+1.8** (10 / 5) |
| (a): subtraction | 78.9 | 83.3 | +3.4 | 77.5 | 82.7 | **+5.8** (13 / 2) |
| (a): the sum of the two | | | +8.8 | | | +5.8 (12 / 3) |
| (b) add then `aa=bb+ccc`: addition | 65.6 | 73.6 | +8.0 | 67.8 | 73.6 | **+4.8** (10 / 5) |
| (b): the second task | 68.0 | 74.3 | +0.7 | 68.6 | 74.5 | **+1.3** (11 / 4) |
| (b): the sum of the two | | | +8.6 | | | +8.6 (10 / 5) |
| (b): the lower of the two | | | +4.0 | | | +4.0 (11 / 4) |

Five reported seeds, M: (a) addition 82.6, 79.4, 88.4, 79.8, 85.4; subtraction 83.3, 79.7, 88.9, 81.1, 84.5. (b) addition 72.6, 71.8, 76.4, 73.6, 81.2; second task 66.0, 66.9, 74.5, 74.3, 79.9.

**The repeat, in one sentence: with my own program, on two tasks, the rule costs nothing and gains.** In neither life is either task lower under M than under R at the median, on five seeds or on fifteen.

**Against the coordinating session's two predictions.**
- (a) "within 3 points of arm R on both tasks": addition is within (+1.8 on fifteen seeds; +5.4 on five); subtraction is not within, being 5.8 points *above* (thirteen seeds of fifteen). The prediction fails by its letter, in the favourable direction: the rule does not cost where two tasks were already kept.
- (b) "a gain of at least 5 points on whichever task is lower": the lower task under R is addition; the gain is +8.0 on the five reported seeds (holds) and +4.8 on fifteen (0.2 short; ten seeds above, five below). The lower of the two scores, seed by seed, gains +4.0 (eleven of fifteen). A gain of about 4 to 5 points, not safely 5.

**What the rule does** (second phase, medians of the five seeds by reading; `report6.txt`).
- *Which lines it chooses.* Of the lines drawn, 94, 84, 76, 70% are of the first task (the counts); of the lines learned from, 82, 61, 55, 52% in (a) and 85, 71, 65, 60% in (b). So the choice moves towards the **new** task throughout, and more as the phase goes on; it does not move back to the old one (M5 failed). And yet the old task is kept as well or better. So the gain is not more rehearsal of the task that is slipping: on these two-task lives the old task gets fewer replayed lines under M than under R and ends higher.
- *How strongly it chooses.* The chosen line's mismatch is 5.7 to 5.9 against 5.3 to 5.5 for a drawn line in (a) (6.3 to 6.4 against 5.9 to 6.0 in (b)): 8% more, not the factor of 1.5 I expected (M6 failed). Most of a line's mismatch is its two numbers, which nothing can foretell, and is the same for every line; the choice works on the small part that differs.
- *It helps before there is a second task.* Addition after its own phase is 90.8 under M against 86.3 under R (five seeds; four of five above), with replay of addition lines only. The rule is a way of spending each learned line where the relay system is most wrong, within a task as much as between tasks.

My reading: on two tasks the rule is a better use of the replayed line than drawing it by the counts, on both tasks; what `nano/life3` reports for three tasks (the middle task held up) I have not run and do not confirm or contradict here, and its description of the mechanism ("the chosen lines moving by themselves to the task that is slipping", in the coordinating session's words) is not what I see on two tasks, where they move to the task being taught.

**Cost.** Four lines drawn and judged for one learned. CPU 128 seconds a run against 65 for R: 33 seconds of drawing (11 for R) and 40 of judging (compiled; four forward passes of a line, no strength changed). **The mismatch is taken with older strengths, not those of the moment:** those of the start of the chunk, at most 500 real lines (and 500 learned replays) earlier; and with the line read alone after a newline. Whether the strengths of the moment would do better is not run.

**Predictions.** M1 failed (M is more than 3 above R on subtraction; nothing is lower). M2 failed (addition is higher under M, not lower). M3 held on the five seeds (+8.0); on fifteen +4.8. M4 held (82% against 94% in (a), 85% against 94% in (b), first reading). M5 failed (the share falls through the phase). M6 failed (1.08, not 1.5). M7 held (+1.8 and +4.8). M8 failed (+1.8). M9 failed, narrowly (+4.8).

**Errors and departures of mine.** The false start of section 6. Seed 1 was run as a pass and nothing chosen on it (R 79.0 / 84.0, M 83.6 / 84.4 in (a); R 75.0 / 83.0, M 80.4 / 79.0 in (b)).

**Not run:** three tasks in one life; other numbers of candidates; the mismatch with the strengths of the moment; the rule together with dreams or with the whole system's answer of `PLAN_5.md`; arm R on the further seeds in this program (taken from the earlier runs of the same lives).

## 9. A dated addition after a second erratum of the coordinating session, entered 20 November before its runs

**The erratum** (the coordinating session's message; `loop_small_case/ERRATA_20261120.md` section 3 on `claude/friendly-darwin-8mmhfm` at `8ae14f646`, which I have not read; the form of the score is from the message). Every line, mine included, scores a never-shown sum as `cells2.py` does: the held-out line alone, with line ends in the delays that reach before the line. No system meets that past after the first line of its life. The coordinating session reports that, scored after a read line, the plain relay system has 95.5% of never-shown addition at 400,000 lines where the old score gives 84.8. **So every level in this folder (and in `nano/joint`) is measured with the relay system some ten points under what it does in the situation it reads in; differences between arms may or may not carry over. Nothing below changes a finished prediction, which stays scored as written.**

**The addition.** `replay6.py` now gives, at the end of each phase, beside the old score (named "empty past"): never-shown addition scored after a read addition line and after a read line of the second task, and the second task (clean) scored after a read line of the second task. The line before is drawn by chance from the lines the program shows of that task; mean of five draws; for answer byte t a delay d that reaches before the line start takes byte (10 + t - d) of the line before. The check has two cases for it (after a line of line ends the new score is the old one; the delays are those of the stream). The drawing uses its own random numbers, so the lives are unchanged.

**Runs, and what is NOT RUN.** My four hours were spent before this message came; I run only the core, on the five reported seeds: life (a) with no replay (P), arm R and arm M; life (b) with arm R and arm M. NOT RUN for want of budget: the fifteen seeds; `PLAN_1.md` to `PLAN_5.md` re-scored (arm R of this program is arm R of `PLAN_1.md` and of `PLAN_3.md`, so their headline figures are covered on five seeds; the controls O and Z, the dreams, the store's own answer and the whole system's answer are not); `nano/joint`. The store's own answer does not depend on what came before a line (its longest cell begins at the newline before the line or, for the last byte, lies inside the line), so the erratum does not touch it; the whole system's answer of `PLAN_5.md` is touched only in the 0.4% of additions and 61% of subtractions that the relay system answered.
- N1 (0.7): in (a), arm R keeps at least 85% of never-shown addition scored after a read addition line (median of five).
- N2 (0.85): in (a), with no replay under 5% is kept by the new score as by the old.
- N3 (0.6): in (a), M less R on addition has the same sign by the new score as by the old (above).
- N4 (0.6): addition at the end is lower scored after a line of the second task than after an addition line, in both lives under arm R.

## 10. Outcome of the addition of section 9 (entered 20 November)

Tables: `report_past.txt` (made by `report_past.py` from `res/m_z_*.json`). The 20 runs of arms R and M give their earlier end figures again; the lives are unchanged. Five reported seeds; medians. "Empty past" is the score used in every earlier section and plan; "after a read line" is the new one.

| at the end of the life | addition, empty past | addition, after a read addition line | after a line of the second task | second task (clean), empty past | second task, after a read line of it |
|---|---|---|---|---|---|
| (a) P, no replay | 1.8 | 1.8 | 1.8 | 86.0 | 92.7 |
| (a) R | 77.2 | **93.8** (95.0, 93.8, 94.4, 91.7, 92.6) | 93.9 | 78.9 | **88.0** |
| (a) M | 82.6 | 91.6 | 91.0 | 83.3 | 90.7 |
| (b) R | 65.6 | **93.4** (95.8, 94.3, 93.4, 91.6, 91.5) | 93.5 | 68.0 | **88.0** |
| (b) M | 73.6 | 91.4 | 91.2 | 74.3 | 90.2 |

At the end of the addition phase, addition after a read addition line: P 95.3 (the coordinating session reports 95.5), R 97.2, M 96.6; empty past 84.8, 86.8, 90.4.

**What the new score changes, in my own earlier readings. I state each plainly; the finished predictions stay scored as written.**

1. *The level replay keeps.* Scored as the system reads, replay keeps **93.8% of never-shown addition** through 200,000 lines of subtraction (97.2 after the addition phase), not 77.2; with no replay 1.8 by either score. The loss through the second task is 3.4 points, not 9.6.
2. *The harsh change is not dearer* (`PLAN_3.md`, section 8, item 1, and `REPORT.md` section 3, item 2). After a read line replay keeps 93.4% through the change to `aa=bb+ccc` and the second task stands at 88.0: the same as in life (a) (93.8 and 88.0). The 12 points by which the harsh change looked worse (65.6 against 77.2) belong to the empty-past score: under it a held-out line begins with line ends in the delays, a past the system met only on the first line of its life, and after the second text that never-met past is answered worse. I withdraw the reading that the relay system "has to hold apart more" on the harsh change.
3. *The rule of this plan does not gain on both tasks* (section 8). After a read line, M less R on addition is -2.1 in (a) (-5.3, -2.2, -2.0, +1.6, -2.1; four of five below) and -2.0 in (b) (four of five below); on the second task +2.5 (five of five above) and +0.9. So on two tasks the rule trades about two points of the old task for one to two and a half of the new: it moves the system along the same trade as the ratio does, in the direction its choice of lines shows (towards the task being taught). The gain "on both tasks" of section 8 holds only for a line read with an empty past, where M is much better (90.4 against 86.8 already after the addition phase). **Section 8's sentence "the rule costs nothing and gains" is true of the empty-past score and not of the score as the system reads; the coordinating session's prediction (a), "within 3 points on both", holds on the new score** (-2.1 and +2.5), and its (b), a gain of 5 on the lower task, does not (the lower task is the second, +0.9).
4. *Keeping costs the second task about 5 points* (92.7 with no replay, 88.0 with), where the empty-past score said about 7.
5. *Addition is answered equally after an addition line and after a line of the second task* (93.8 and 93.9; 93.4 and 93.5): what the line before is does not matter, only that there is one.

**Predictions.** N1 held (93.8). N2 held (1.8). N3 failed (the sign turns: -2.1). N4 failed (no difference).

**A weakness of the check, mine:** the case "after a line of line ends the new score is the old" was made on a system 900 lines old and compares 0.00 with 0.00; the other case (the delays are those of the stream) is the one that carries. The figure for P after the addition phase, 95.3, beside the coordinating session's 95.5 from its own program, is the working check.

**NOT RUN** (budget spent, as section 9 says): the fifteen seeds; the controls O and Z, the ratio 1/2, the dreams and the whole system's answer on the new score; `nano/joint` on the new score. Until they are, every figure of `PLAN_1.md` to `PLAN_5.md` and of `nano/joint` is an empty-past figure, and items 2 and 3 above show that differences between arms can change sign or vanish on the new score.
