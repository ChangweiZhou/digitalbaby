# Line `nano/life3`, third plan: the harsh case, a third task written in the first task's byte values (20 November 2026)

**Authorisation.** Mr. Xu, 20 November 2026, as given to this session by the coordinating session (in Chinese; the coordinator's translation): "You can go on trying boldly; we keep to the facts; besides the ideas I have mentioned, try others as well." And earlier, in English: "Keep experimenting with the agents." The step is one of those the coordinating session named for this line if the budget remained.

**Source.** `claude/nano-life3-8mmhfm` at commit fd16235 (the first plan with its outcome); the programs are those of `PLAN_1.md`, with what is said below added to `life3.py` before this plan was committed. **Development level**; no checker gate, seal or outside review is claimed; readings and predictions marked "mine" are this session's. Seeds, score (subtraction on the clean score), readings every 50,000 real lines, last reading with the mean of the last three beside it, median of the five reported seeds 73129, 93683, 116089, 406364, 545353: all as `PLAN_1.md`.

## 1. The one change

Life C is life A with the third task written otherwise: `aa=bb+ccc` for the larger of aa and bb (task `xam`; the text that `nano/keep` and `nano/replay` used for their harsh change, with `max` in place of subtraction since subtraction is already the second task). It brings **no byte value never met**. The symbol set is kept at the fifteen of life A (the fifteenth byte is never read), so that the first two phases are life A's to the digit for arms P, O, R and K, which is checked.

As `nano/replay` noted of its harsh change: the text brings no new byte value, but it does put byte values at delays where the first task never had them ('=' three bytes after the line's start, '+' six after). For route 1 this is a change announced at some bytes only, which the keeping line found to be its worst case. For subtraction's cells in route 1 a '+' is a line never met at any delay.

*Arms.* The plan is for route 2: P, O and R. Routes 1 and 3 are run beside them (K, S) because they cost little and the roadmap asks which route to carry on; their figures are entered as side results with predictions of their own.

## 2. One question, one hypothesis

**Does replay still keep two earlier tasks when the third brings no new byte?**

*Hypothesis.* At the end of life C route 2 has addition at 50% or more and subtraction at 50% or more (the bar is lower than life A's 60 because life A already put route 2's subtraction at 52.4; the question here is whether the harsh text costs more), and the third task at 50% or more. *Refuted* if an earlier task is under 30%. Beside it: the difference from life A, task by task, paired by seed.

## 3. Predictions (mine), before any run of life C

Disclosure: to see that the new lines of the program run, route 2 was run on seed 1 for 3,000, 2,000 and 2,000 lines (all scores under 7), and route 3 for a few hundred lines (`xam` 63.2).

| | Statement | Chance |
|---|---|---|
| H1 | R: addition 50 or more | 0.65 |
| H2 | R: addition lower than in life A on at least four seeds (the harsh text costs addition) | 0.70 |
| H3 | R: subtraction 50 or more | 0.40 |
| H4 | R: subtraction within 6 points of life A's 52.4 (the harsh text is addition's affair, not subtraction's) | 0.55 |
| H5 | R: the third task 50 or more | 0.55 |
| H6 | R: refuted (an earlier task under 30) | 0.15 |
| H7 | P: addition and subtraction under 10 | 0.90 |
| H8 | O: addition 60 or more, subtraction 50 or more | 0.60 |
| H9 | K: addition under 30 (lost, as the keeping line found for a change announced at some bytes) | 0.85 |
| H10 | K: subtraction kept, within 2 points of its level after its own phase on at least four seeds | 0.80 |
| H11 | K: fewer than 128 cells join at the second change, and the third task ends under 60 | 0.70 |
| H12 | S: addition and subtraction both 60 or more | 0.50 |
| H13 | S: the third task 60 or more | 0.85 |

*Why (mine).* `nano/replay` found the harsh change dearer than a change with a new operator by some 10 points of the old task with two tasks. Here the task that shares bytes with the third is addition, which has the larger share of replay; subtraction has its own operator. Route 1: in the third text some byte always has a '+' on the delay lines, so subtraction's cells should be silent throughout, while addition's cells are silenced at two learned bytes of nine and learn at the others. Route 3: both the '=' and the '+' stand at other delays than in addition, so more of its kinds keep the two tasks apart than in life A.

## 4. Cost, and what is not run

Five arms, five seeds, lives of 800,000 real lines: about 55 CPU-minutes. No run on seed 1: nothing is chosen. Not run: the harsh text as a second task (the other lines did); a harsh text for subtraction; other ratios of replay.
