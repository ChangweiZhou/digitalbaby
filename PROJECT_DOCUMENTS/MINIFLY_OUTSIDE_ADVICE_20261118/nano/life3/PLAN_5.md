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
