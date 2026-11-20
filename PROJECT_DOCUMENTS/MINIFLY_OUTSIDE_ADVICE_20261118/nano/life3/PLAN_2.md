# Line `nano/life3`, second plan: a fourth task (20 November 2026)

**Authorisation.** Mr. Xu, 20 November 2026, as given to this session by the coordinating session (in Chinese; the coordinator's translation): "You can go on trying boldly; we keep to the facts; besides the ideas I have mentioned, try others as well." And earlier, in English: "Keep experimenting with the agents." The step is one of those the coordinating session named for this line if the budget remained.

**Source.** `claude/nano-life3-8mmhfm` at commit fd16235 (the first plan with its outcome); the programs are those of `PLAN_1.md`, with what is said below added to `life3.py` before this plan was committed. **Development level**; no checker gate, seal or outside review is claimed; readings and predictions marked "mine" are this session's. Seeds, score (subtraction on the clean score), readings every 50,000 real lines, last reading with the mean of the last three beside it, median of the five reported seeds 73129, 93683, 116089, 406364, 545353: all as `PLAN_1.md`.

## 1. The one change

Life B is life A with a fourth phase: 400,000 lines of addition, 200,000 of subtraction, 200,000 of `max`, then **200,000 lines of a fourth task with a fourth operator byte**. Sixteen byte values. `life3.py` gains the task and gives each further byte value a generator of its own for its birth strengths, so that life B's first three phases are life A's for the relay system and route 1 (a check: at the end of the third phase arms P, O, R and K should equal `res/a1_*` to the digit; route 3's smoothing depends on the number of byte values, so its first three phases are a new sample).

*The fourth task* is `min`: `aa_bb=ccc`, the smaller of the two numbers. Why (mine): of the three candidates of step 0 only `max` could be learned alone within 200,000 lines, so a fourth task with a carry is not available under the rule of the brief; `min` is the nearest task that is not `max`. Step 0, seed 1: the plain system alone on `plan=min:200000` must reach 70% of never-shown pairs at the last reading. If it does not, this plan stops and says so.

*Arms.* P, O (each of the earlier tasks with equal chance, one old line for each real line), R, K (the final candidate unchanged, 384 cells: **the pool is spent before the fourth task**), S; and one further arm, **K512**: route 1 with 512 cells (511 waiting at birth), nothing else changed, to part "the pool is spent" from "four tasks are too many". K512 draws different birth strengths for the answering cell's return connections than K (the generator is drawn in the order of `cells2.py`), so it is a further sample and not the same life.

## 2. One question, one hypothesis

**Does each route still keep, and still learn, when a fourth task comes?**

*Hypothesis.* At the end of life B each of the routes K512, R and S has each of the three earlier tasks at 60% or more and the fourth at 60% or more. *A route is refuted for four tasks* if an earlier task is under 30%. For K with the pool spent the hypothesis is expected to fail and the question is how: by not learning the fourth task, or by losing an earlier one.

## 3. Predictions (mine), before any run of life B

Disclosure: to see that the new lines of the program run, route 1 was run on seed 1 for 3,000, 2,000, 2,000 and 2,000 lines of the four tasks: at the fourth no cell joined, no cell could speak on the fourth task's lines and its score was 1.8. I knew this when I wrote F1 and F2. Route 3 on a few hundred lines of `min` after other tasks read 54.4.

| | Statement | Chance |
|---|---|---|
| M0 | step 0: `min` passes on seed 1 | 0.90 |
| F1 | K (384): no cell joins at the third change, and the fourth task ends under 10 | 0.85 |
| F2 | K (384): the three earlier tasks end within 2 points of their levels at the end of the third phase, on at least four seeds each | 0.85 |
| F3 | K512: 127 or 128 cells join at the third change on every seed; all four tasks 60 or more | 0.70 |
| F4 | R: addition 60 or more | 0.60 |
| F5 | R: subtraction under 50 | 0.75 |
| F6 | R: subtraction under 30 (refuted) | 0.25 |
| F7 | R: `max` under 60 | 0.75 |
| F8 | R: the fourth task 60 or more | 0.40 |
| F9 | R: of the replayed lines of the fourth phase, subtraction's are 18 to 26% and `max`'s 18 to 26% | 0.85 |
| F10 | S: addition 60 or more | 0.50 |
| F11 | S: subtraction 60 or more | 0.55 |
| F12 | S: `max` 60 or more and `min` 60 or more | 0.75 |
| F13 | S: the whole hypothesis | 0.30 |
| F14 | O: addition 60 or more; subtraction under 60 | 0.60 |
| F15 | P: the three earlier tasks under 10; `min` 60 or more | 0.65 |

*Why (mine).* K: with no cell waiting, the lines of a new operator silence every cell and nothing is left to learn with; the old cells are untouched for the same reason. R: in the fourth phase an old task's share of replay is its share of all lines read, about a fifth each for subtraction and `max`; the first outcome showed 0.29 of a line already loses a third of subtraction. S: the first outcome showed addition wearing by about 4 points a phase from 69; another phase may take it under 60; `max` and `min` share every cell that does not hold the operator, which may cost both.

## 4. Cost, and what is not run

A life of 1,000,000 real lines: P about 50 seconds, O and R about 2 minutes, K about 2, S about 7; five seeds, six arms: about 75 CPU-minutes. No run on seed 1 beyond step 0: no setting is chosen. Not run: a fifth task; other fourth tasks; unequal phases; replay with another rule of sharing (the obvious next change for route 2, and a new idea rather than a test: left for a plan of its own).

---

## 5. Outcome (20 November 2026; development level; readings are mine)

Runs: step 0 (one life, seed 1) and 30 lives of 1,000,000 real lines; about 80 CPU-minutes. Files: `res/c0_min_1.json`, `res/a2_*`, `table2.txt` (every seed, every phase end).

**Step 0.** `min` passes: the plain system alone has 88.0 at 200,000 lines on seed 1 (mean of the last three readings 86.3).

**The check.** At the end of the third phase arms P, O, R and K equal life A's files to the digit on every seed (`table1.txt` against `table2.txt`). Route 3's first three phases are a new sample, as said: 65.2 / 64.5 / 97.6 beside life A's 65.2 / 65.0 / 97.4.

**Life B at its end** (after 400,000 lines of addition, 200,000 of subtraction, 200,000 of `max`, 200,000 of `min`). Never-shown pairs, whole answer, %; median of five seeds, last reading; in brackets the mean of the last three readings; subtraction on the clean score.

| Arm | add | sub | max | min (the fourth) |
|---|---|---|---|---|
| P plain | 0.0 [0.1] | 0.5 [0.4] | 3.0 [2.9] | 83.0 [79.1] |
| O the program's old lines | 66.0 [66.1] | 58.3 [56.7] | 62.0 [57.9] | 60.8 [57.7] |
| K route 1, 384 cells (pool spent) | 85.8 [85.8] | 86.2 [86.2] | 91.0 [91.0] | **0.0** [0.0] |
| K512 route 1, 512 cells | 83.0 [83.0] | 90.5 [90.5] | 91.2 [91.2] | **93.2** [91.2] |
| R route 2, replay | 64.0 [63.7] | **42.7** [43.0] | **55.2** [53.4] | 62.4 [58.9] |
| S route 3, cells that count | 63.2 [63.5] | 61.0 [61.7] | 96.8 [96.3] | 96.0 [96.4] |

Each seed at the end (73129, 93683, 116089, 406364, 545353): R addition 67.6, 66.8, 59.4, 64.0, 63.4; subtraction 37.2, 47.3, 37.9, 42.7, 51.1; `max` 62.2, 60.4, 52.0, 55.2, 53.6; `min` 65.2, 62.4, 59.6, 65.4, 61.6. S addition 63.2, 65.8, 60.8, 59.0, 63.8; subtraction 59.6, 61.0, 60.2, 62.4, 61.7. K512 addition 87.8, 82.0, 79.2, 84.0, 83.0; subtraction 90.6, 83.8, 96.3, 89.3, 90.5; `max` 88.6, 91.2, 89.0, 94.0, 91.4; `min` 94.4, 92.2, 93.2, 93.2, 91.6. K `min` 0.0 on every seed.

**The hypothesis: holds for route 1 with 512 cells and for route 3; fails for route 2 on two of its three earlier tasks (subtraction 42.7, `max` 55.2); no route is refuted (no earlier task under 30).** Route 1 with the pool spent keeps everything and learns nothing.

**Predictions scored.**

| | Statement | Chance | Outcome |
|---|---|---|---|
| M0 | `min` passes step 0 | 0.90 | held (88.0) |
| F1 | K (384): no cell joins at the third change; the fourth task under 10 | 0.85 | held (none; 0.0 on every seed) |
| F2 | K (384): the earlier tasks within 2 points of their levels after the third phase | 0.85 | held (addition and subtraction to the digit; `max` moves by +0.8, -0.4, +0.6, +0.2, +0.4) |
| F3 | K512: 127 or 128 join at the third change on every seed; all four 60 or more | 0.70 | held (128; 83.0, 90.5, 91.2, 93.2) |
| F4 | R: addition 60 or more | 0.60 | held (64.0; one seed 59.4) |
| F5 | R: subtraction under 50 | 0.75 | held (42.7; one seed 51.1) |
| F6 | R: subtraction under 30 | 0.25 | did not happen (lowest seed 37.2) |
| F7 | R: `max` under 60 | 0.75 | held (55.2; two seeds above 60) |
| F8 | R: the fourth task 60 or more | 0.40 | held by the last reading (62.4; mean of three 58.9) |
| F9 | R: shares of replay in the fourth phase, subtraction and `max` each 18 to 26% | 0.85 | held (45 / 22 / 22 / 11 on every seed) |
| F10 | S: addition 60 or more | 0.50 | held (63.2; one seed 59.0) |
| F11 | S: subtraction 60 or more | 0.55 | held (61.0; one seed 59.6) |
| F12 | S: `max` and `min` 60 or more | 0.75 | held (96.8, 96.0) |
| F13 | S: the whole hypothesis | 0.30 | held, against my chance again |
| F14 | O: addition 60 or more; subtraction under 60 | 0.60 | held (66.0; 58.3) |
| F15 | P: earlier tasks under 10; `min` 60 or more | 0.65 | held (83.0) |

**Readings (mine).**

1. **Route 1 with its pool spent does not forget and does not learn.** At the fourth operator every cell falls silent on every learned byte, no cell is left to join, and the fourth task stands at 0.0 on every seed for 200,000 lines; the three earlier tasks stay where they were. With 128 more cells in the pool (K512) the fourth task is learned to 93.2 and all four stand at 83 to 93. So for this route the range of continual learning is, exactly, the pool: one task for every 128 waiting cells, each task of this size, every change announced by a new byte. The cells are never given back. What it costs, with sixteen byte values: 161 + 16 + 16 numbers for each cell, 24,704 numbers for each further task, against the plain system's 49,424 for everything.
2. **Route 2 loses more with each task, by the arithmetic of its shares.** In the fourth phase the replay is 45% additions, 22% subtractions, 22% `max`, 11% `min`. Subtraction: 78.9 after its own phase, 52.4 after one more task, 42.7 after two. `max`: 64.2, then 55.2. Addition, the longest task, holds best (77.2, 73.4, 64.0). The upper reference, with a third of a line for each old task, is itself at 66.0 / 58.3 / 62.0 / 60.8: four tasks of this kind are near what one set of 256 cells holds at this rate of rehearsal, however the old lines are chosen. Route 2 is within 2 points of the reference on addition and the fourth task and 16 and 7 points under it on the two middle tasks, which are the two with the smallest shares.
3. **Route 3 still holds at four tasks, on margins that are now 1 to 3 points, and they are wearing.** Addition 69.2, 65.2, 63.2 at the ends of the last three phases; subtraction 66.4, 64.5, 61.0 (one seed under 60 for each at the end). The two new tasks cost each other nothing visible (`max` 97.6 and then 96.8 with `min` at 96.0), though they share every cell that does not hold the operator. By the trend a fifth task would probably put subtraction under 60; I have not run it and do not claim it.
4. **The plain system learns the fourth task better than the third** (83.0 against 70.4) with `max` just before it: something of `max` serves `min` in the cells that then lose `max` (3.0). Route 1 cannot use that, by construction (`min` by fresh cells: 93.2, so it does not need it here).
5. **Taken together, for Mr. Xu's question of whether the system can learn continually within some small range** (my reading, at development level, on tasks that announce themselves by a new byte): route 1, yes, up to the size of its pool and no further, at the price of a separate set of cells for each task; route 3, yes for four tasks at a level of 61 to 63 on the two tasks with a carry and with a slow wear; route 2 at one replayed line for each real line, two tasks well, three with the middle one damaged, four with two damaged.
