# Line `nano/life3`, fifth plan: is route 2's loss of the middle task its share of the replay? (20 November 2026)

**Authorisation.** Mr. Xu, 20 November 2026, as given to this session by the coordinating session (in Chinese; the coordinator's translation): "You can go on trying boldly; we keep to the facts; besides the ideas I have mentioned, try others as well." And earlier, in English: "Keep experimenting with the agents." This step is not one the coordinating session named; it follows from the first outcome and is mine.

**Source.** `claude/nano-life3-8mmhfm` at commit 6497183; `life3.py` gains the option `flat=1` (below) before this plan is committed; with the option off route 2 is unchanged (seed 73129, 50,000 lines of addition: 75.2, as before). **Development level**; no checker gate, seal or outside review is claimed. Seeds, score, readings and reporting as `PLAN_1.md`. Written while the runs of plans 2 to 4 were in the queue and before any of their results was read.

## 1. What this follows

`PLAN_1.md`, outcome: in life A route 2 ends with addition 73.4, subtraction 52.4, `max` 64.2. In the third phase the replayed lines are 58% additions, 29% subtractions, 14% of the third task. My reading there: subtraction falls because it has 0.29 of a replayed line for each real line; the upper reference, where each old task has 0.5, ends 78.2 / 63.2 / 64.2. That reading has not been tested: the reference differs from route 2 in more than the shares.

## 2. Two arms, each one change from route 2 of life A

- **R2: two replayed lines for each real line** (`r=2`). The shares stay as they are; every task's amount of replay doubles: subtraction then has about 0.58 of a line for each real line, a little more than the reference gives it.
- **Rflat: the replay draws each next byte with equal chance among the bytes that have followed the run, however often** (`flat=1`, r = 1). A rule on the counting cells' own contents that names no task: where a run has been followed by three operators, each is drawn one time in three. On a test of a few thousand lines the replayed lines were 34% / 31% / 36% of the three tasks. This gives subtraction about 0.33 of a line (hardly more than 0.29), takes addition down from 0.58 to 0.33, and gives the third task itself a third of the replay from its first lines.

## 3. One question, one hypothesis

**Is the loss of the middle task the smallness of its share of replay?**

*Hypothesis.* With two replayed lines for each real line, route 2 ends life A with addition, subtraction and the third task each at 60% or more (median of five seeds, last reading). *Refuted* if subtraction is still under 55: then the amount of replay is not what was missing, and my reading of the first outcome is wrong or incomplete.

For Rflat no hypothesis: predictions only. It is the cheap form of "share the replay equally without being told the tasks", and my arithmetic says it cannot help subtraction much, because with three tasks an equal share of one line is a third.

## 4. Predictions (mine)

| | Statement | Chance |
|---|---|---|
| E1 | R2: subtraction 60 or more | 0.50 |
| E2 | R2: subtraction above life A's on five seeds of five | 0.80 |
| E3 | R2: addition 70 or more | 0.75 |
| E4 | R2: the third task under life A's 64.2 (more replay costs the task being learned) | 0.60 |
| E5 | R2: the whole hypothesis | 0.30 |
| E6 | Rflat: subtraction within 6 points of 52.4 | 0.60 |
| E7 | Rflat: addition below life A's on at least four seeds | 0.75 |
| E8 | Rflat: the third task above life A's 64.2 | 0.55 |
| E9 | Rflat: the lowest of its three figures is higher than route 2's lowest in life A (52.4) | 0.50 |

## 5. Cost, and what is not run

Ten lives; R2 reads 2.4 million lines (about 2.5 minutes), Rflat 1.6 million (about 1.5): about 20 CPU-minutes. No run on seed 1: the two values are not chosen from a grid. Not run: other ratios; replay that leaves out the task being read (it would have to know which lines are new); replay chosen by the relay system's own mismatch on the candidate line (a further idea of mine, costlier, for a later plan); the same arms in the life of four tasks.

*A note on what a positive result would and would not mean.* If two lines of replay for each real line suffice for three tasks, the arithmetic says three lines would be wanted for four: the replay a life needs would grow with the number of tasks it has held. That is a bound on "continual", not a rescue.

---

## 6. Outcome (20 November 2026; development level; readings are mine)

Runs: 10 lives, about 18 CPU-minutes. Files: `res/a5_*`, `table5.txt`.

**Life A at its end, route 2 and its two changes.** Never-shown pairs, whole answer, %; median of five seeds, last reading [mean of the last three readings]; subtraction on the clean score. The last two columns: of what the relay system reads in the third phase (real and further lines together), how much is subtraction and how much addition.

| Arm | add | sub | max | subtraction's part of all lines read | addition's part |
|---|---|---|---|---|---|
| R, one replayed line a real line (life A) | 73.4 [75.4] | 52.4 [54.7] | 64.2 [60.6] | 14.4% | 28.8% |
| **R2, two replayed lines** | 80.2 [82.0] | **56.9** [57.5] | 62.4 [57.8] | 19.3% | 38.3% |
| **Rflat, equal chance among the bytes that followed** | 63.6 [63.3] | 54.4 [56.8] | 63.6 [61.5] | 16.5% | 16.8% |
| O, the program's old lines (reference) | 78.2 [77.7] | 63.2 [63.8] | 64.2 [59.5] | 24.9% | 25.2% |

Each seed (73129, 93683, 116089, 406364, 545353): R2 addition 87.0, 83.2, 80.2, 68.4, 79.2; subtraction 64.8, 56.9, 54.1, 46.6, 58.1; `max` 58.4, 62.4, 62.8, 58.8, 62.6. Rflat addition 49.2, 71.6, 44.8, 63.6, 66.2; subtraction 41.1, 61.5, 42.4, 62.9, 54.4; `max` 61.2, 66.6, 63.6, 66.4, 61.0. Paired with route 2 of life A by seed: R2 subtraction +11.6, +4.4, +9.3, -5.8, +7.8 (median +7.8); addition +13.0, +7.2, +8.0, -4.0, +5.8; Rflat subtraction -12.1, +9.1, -2.4, +10.4, +4.0; addition -24.8, -4.4, -27.4, -8.8, -7.2. The replayed lines of Rflat in the third phase are 34 / 33 / 33% of the three tasks; all are shown lines.

**The hypothesis fails**: with two replayed lines for each real line subtraction is at 56.9, not 60 (one seed of five at 60 or more). It is not under the 55 at which I said my reading would be refuted. The honest statement is between: twice the replay gives the middle task about 8 points on four seeds of five and takes a little from the task being learned; it does not restore it.

| | Statement | Chance | Outcome |
|---|---|---|---|
| E1 | R2: subtraction 60 or more | 0.50 | failed (56.9) |
| E2 | R2: subtraction above life A's on five seeds of five | 0.80 | failed (four of five) |
| E3 | R2: addition 70 or more | 0.75 | held (80.2; one seed 68.4) |
| E4 | R2: the third task under life A's 64.2 | 0.60 | held (62.4) |
| E5 | R2: the whole hypothesis | 0.30 | did not hold |
| E6 | Rflat: subtraction within 6 points of 52.4 | 0.60 | held (54.4), with seeds from 41.1 to 62.9 |
| E7 | Rflat: addition below life A's on at least four seeds | 0.75 | held (five of five; by 4 to 27 points) |
| E8 | Rflat: the third task above life A's 64.2 | 0.55 | failed (63.6) |
| E9 | Rflat: its lowest figure above route 2's lowest in life A (52.4) | 0.50 | held by 2 points (54.4), which I do not count as a gain |

**Readings (mine).**

1. **My reading of the first outcome was half right.** More replay helps the middle task, and not enough: with 0.58 of a replayed subtraction line for each real line, a little more than the reference's 0.50, route 2 has subtraction at 56.9 where the reference has 63.2. What the reference has and R2 has not is subtraction as a larger part of everything read: a quarter, against a fifth, because R2 pays for its subtraction lines with twice as many addition lines. Over the four arms subtraction's level follows its part of all lines read: 14.4% gives 52.4, 16.5% gives 54.4, 19.3% gives 56.9, 24.9% gives 63.2. Four points, one life, not a law; it is the regularity I would test next.
2. **Under replay in proportion to counts, more replay cannot raise the middle task's part beyond its part of the counts** (29% here in the limit of endless replay, and then the task being read has almost none of the stream). So the ratio is not the lever. The lever is what is replayed.
3. **Equal chance among the bytes that followed is not that lever.** It makes the three tasks' parts equal with no knowledge of tasks, which is what I built it for, and the equal part is a third of one line: subtraction gains 2 points at the median, addition loses 10, and the seeds scatter much more widely (addition 44.8 to 71.6). A rule that equalises is the wrong aim when the task being read already has every real line: a third of the replay then goes to the one task that does not need it.
4. **What a rule would have to do** (not run; mine): give the replay to the lines the relay system currently gets wrong and withhold it from those it gets right, by its own mismatch on a replayed line before learning from it. That needs no knowledge of tasks, leaves out the task being read as soon as it is learned, and turns to whichever old task is slipping. It is the next plan I would write for this route.
