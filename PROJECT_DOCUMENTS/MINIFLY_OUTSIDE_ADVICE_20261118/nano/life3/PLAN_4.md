# Line `nano/life3`, fourth plan: the other orders of the three tasks, for route 1 (20 November 2026)

**Authorisation.** Mr. Xu, 20 November 2026, as given to this session by the coordinating session (in Chinese; the coordinator's translation): "You can go on trying boldly; we keep to the facts; besides the ideas I have mentioned, try others as well." And earlier, in English: "Keep experimenting with the agents." The step is one of those the coordinating session named for this line if the budget remained.

**Source.** `claude/nano-life3-8mmhfm` at commit fd16235 (the first plan with its outcome); the programs are those of `PLAN_1.md`, with what is said below added to `life3.py` before this plan was committed. **Development level**; no checker gate, seal or outside review is claimed; readings and predictions marked "mine" are this session's. Seeds, score (subtraction on the clean score), readings every 50,000 real lines, last reading with the mean of the last three beside it, median of the five reported seeds 73129, 93683, 116089, 406364, 545353: all as `PLAN_1.md`.

## 1. The one change

Route 1 did best in life A (85.8, 86.2, 90.8). The change is the order of the tasks; the lengths stay with the place in the life (400,000 lines for the first task, 200,000 for each of the others), the fifteen byte values, the splits and the system unchanged (the final candidate of `nano/keep`, 384 cells). The five other orders: add, max, sub; sub, add, max; sub, max, add; max, add, sub; max, sub, add. Route 1 only, five reported seeds. The plain system is not run again: that it keeps nothing is not in doubt.

## 2. One question, one hypothesis

**Does route 1 keep three tasks whatever their order?**

*Hypothesis.* In each of the five orders, at the end of the life, every task is at 60% or more (median of five seeds, last reading). *Refuted for an order* if a task learned earlier is under 30% at the end.

## 3. Predictions (mine), before any run

| | Statement | Chance |
|---|---|---|
| D1 | the hypothesis holds in all five orders | 0.70 |
| D2 | in every order and for both earlier tasks, the level at the end is within 2 points of the level after the task's own phase on at least four seeds | 0.75 |
| D3 | on every seed and in every order 128 cells join in each phase and none waits at the end | 0.80 |
| D4 | the lowest figure of the fifteen (order by task) is addition learned as a second or third task | 0.60 |
| D5 | addition learned in 200,000 lines by 128 fresh cells is 70 or more in all four orders where it is not first | 0.65 |

*Why (mine).* The rule sees a change only through a line never met, and each of the three operators is such a line for the cells of the two other tasks, in any order. What the order changes is how long each task is read and by which cells: a task in second or third place has 200,000 lines and not 400,000. The keeping line found addition at 79.0 as a second task after subtraction.

## 4. Cost, and what is not run

25 lives of 800,000 lines, about 90 seconds each: about 40 CPU-minutes. No run on seed 1: nothing is chosen. Not run: the other orders for routes 2 and 3 (route 2's shares of replay depend on order by arithmetic: the first task has half the lines whichever it is; route 3 costs four times as much); orders of four tasks.
