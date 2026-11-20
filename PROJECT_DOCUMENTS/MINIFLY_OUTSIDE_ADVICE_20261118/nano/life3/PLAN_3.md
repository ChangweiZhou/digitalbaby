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

---

## 5. Outcome (20 November 2026; development level; readings are mine)

Runs: 25 lives of 800,000 real lines, about 50 CPU-minutes. Files: `res/a3_*`, `table3.txt`. **The check:** at the end of the second phase arms P, O, R and K equal life A's files to the digit on every seed.

**Life C at its end** (addition, subtraction, then `aa=bb+ccc` for the larger of the two). Never-shown pairs, whole answer, %; median of five seeds, last reading [mean of the last three readings]; subtraction on the clean score. Beside each, life A (the third task with a new operator byte).

| Arm | add: life C (life A) | sub: life C (life A) | third task: life C (life A) |
|---|---|---|---|
| P plain | 1.6 (2.8) | 1.6 (1.6) | 76.0 [68.6] (70.4) |
| O the program's old lines | 65.2 [66.5] (78.2) | 46.6 [50.9] (63.2) | 56.4 [54.1] (64.2) |
| R route 2, replay | **58.8** [61.7] (73.4) | **36.3** [38.2] (52.4) | **65.2** [61.3] (64.2) |
| K route 1, cells that join | 0.0 [0.0] (85.8) | 86.2 [86.2] (86.2) | 78.8 [76.2] (90.8) |
| S route 3, cells that count | 71.6 [70.6] (65.2) | 58.4 [58.1] (65.0) | 97.0 [96.5] (97.4) |

Each seed at the end (73129, 93683, 116089, 406364, 545353): R addition 52.2, 64.0, 58.0, 61.8, 58.8; subtraction 45.6, 36.3, 30.2, 41.5, 28.6; third 65.2, 67.0, 58.6, 66.0, 65.2. O addition 58.4, 65.2, 74.6, 56.8, 67.4; subtraction 29.8, 46.6, 57.3, 44.9, 53.6. S addition 72.8, 73.8, 70.6, 71.6, 70.0; subtraction 58.4, 55.9, 57.8, 60.0, 62.4. K addition 0.0 on every seed; subtraction 86.2, 76.5, 82.0, 89.8, 92.5.

**The hypothesis fails for route 2 on subtraction** (36.3 against the 50 asked; addition 58.8 and the third task 65.2 pass). By the median route 2 is not refuted (36.3 is above 30); on one seed of five subtraction is under 30 (28.6) and on another at 30.2. Paired with life A by seed, the harsh text costs route 2 between 10.6 and 21.8 points of addition (median 14.2) and between 7.6 and 21.8 of subtraction (median 14.6), and nothing of the third task.

**Predictions scored.**

| | Statement | Chance | Outcome |
|---|---|---|---|
| H1 | R: addition 50 or more | 0.65 | held (58.8) |
| H2 | R: addition lower than in life A on at least four seeds | 0.70 | held (five of five) |
| H3 | R: subtraction 50 or more | 0.40 | failed (36.3) |
| H4 | R: subtraction within 6 points of life A's 52.4 | 0.55 | **failed** (36.3): the harsh text costs subtraction as much as addition |
| H5 | R: the third task 50 or more | 0.55 | held (65.2) |
| H6 | R: refuted (an earlier task under 30) | 0.15 | not by the median; one seed of five |
| H7 | P: addition and subtraction under 10 | 0.90 | held |
| H8 | O: addition 60 or more, subtraction 50 or more | 0.60 | failed on subtraction (65.2; 46.6) |
| H9 | K: addition under 30 | 0.85 | held (0.0 on every seed) |
| H10 | K: subtraction kept within 2 points on at least four seeds | 0.80 | held (0.0, -1.4, -0.5, -0.5, -0.5) |
| H11 | K: fewer than 128 cells join, and the third task under 60 | 0.70 | half: 30 cells join on every seed; the third task is at 78.8 |
| H12 | S: addition and subtraction both 60 or more | 0.50 | failed on subtraction (71.6; 58.4) |
| H13 | S: the third task 60 or more | 0.85 | held (97.0) |

**Readings (mine).**

1. **A third task in the first task's byte values is dear to every system that holds the tasks in one set of cells, and it is the middle task that pays most.** Route 2 ends 58.8 / 36.3 where life A gave 73.4 / 52.4. The upper reference, shown old lines by the program, falls as far (65.2 / 46.6 against 78.2 / 63.2), so this is not a fault of replay from counting cells: replay is 6 and 10 points under the reference here, as it was 5 and 11 under it in life A. I was wrong that subtraction, having its own operator, would be spared (H4): the relay system pays for the new text in everything it holds, and the task with the least rehearsal falls furthest.
2. **Replay does what it did with two tasks, and three tasks make it insufficient.** `nano/replay` kept 65.6 of addition through this text as a second task; here addition, with a subtraction between, is at 58.8, and subtraction at 36.3. The replayed lines are all shown lines and the shares are life A's (58 / 29 / 14).
3. **Route 1 splits exactly as its rule says.** Subtraction's cells never met a '+', are silent throughout and keep subtraction to the digit or within 1.4 points. Addition's cells are silenced at some bytes only, learn the new text at the others and lose addition entirely (0.0 on every seed); 30 cells join, slowly, the last of them 1.28 million learned bytes into the phase, and 98 still wait at the end. The third task is learned to 78.8 by addition's cells and those 30 (158 cells may speak on its lines, and the same 158 on addition's): better than I predicted, and at the price of the whole first task. This is the keeping line's "announced at some bytes only" case seen with a kept task beside it: the rule protects what the new text cannot reach and gives up what it can.
4. **Route 3 keeps addition better here than in life A and subtraction worse** (71.6 and 58.4 against 65.2 and 65.0). I predicted more of its kinds would keep addition and the third task apart, since both the '=' and the '+' stand at other delays; that fits addition's figure. Why subtraction falls 8 points from the end of its phase, where `max` with its own operator cost it 1.4, I have not looked into; it puts route 3 under the 60 of life A's criterion on one task.
5. **No route passes life C whole.** Route 1 loses the first task entirely; route 2 has the second at 36; route 3 is nearest (71.6 / 58.4 / 97.0).
