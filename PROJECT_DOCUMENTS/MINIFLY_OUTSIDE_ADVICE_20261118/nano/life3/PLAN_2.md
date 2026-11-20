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
