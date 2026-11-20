# Line `nano/life3`, sixth plan: replay chosen by the relay system's own mismatch (20 November 2026)

**Authorisation.** Mr. Xu, 20 November 2026, as given to this session by the coordinating session (in Chinese; the coordinator's translation): "You can go on trying boldly; we keep to the facts; besides the ideas I have mentioned, try others as well." And earlier, in English: "Keep experimenting with the agents." This step is not one the coordinating session named; the idea is mine and follows from the fifth outcome.

**Source.** `claude/nano-life3-8mmhfm` at commit 8d96fc7; `life3.py` gains arm N (below) before this plan is committed; with it unused route 2 is unchanged (seed 73129, 50,000 lines of addition: 75.2). No compiled code is added: the choice among replayed lines is made in numpy, and the relay system's kernel is the one checked in `check_life3_output.txt`. **Development level**; no checker gate, seal or outside review is claimed. Seeds, score, readings and reporting as `PLAN_1.md`.

## 1. What this follows

`PLAN_5.md`, outcome: under replay in proportion to counts the middle task of three falls to 52.4; twice the replay gives it 56.9; equal chances among followers 54.4; the program's own old lines 63.2. Reading 4 there: the replay should go to what the relay system is getting wrong.

## 2. The one change (arm N)

Route 2 of life A, with this difference: after each real line the system draws **four** lines from its counting cells exactly as route 2 draws one, and the relay system learns from **the one on which its own squared mismatch, summed over the line's nine named bytes, is largest**. One further line is learned from for each real line, as in route 2. The mismatch is measured with the line standing alone (line ends before it, as in the scoring) and with the relay system's strengths as they stood at the start of the present block of 500 real lines (so that the four are judged in one pass; the system's own quantity, a little stale). Nothing names a task, an operator or a place in the line; the bytes nothing can foretell (the two numbers) add about the same mismatch to every line.

The two numbers, four candidates and blocks of 500, are set here and not chosen from runs. Disclosure: one run on seed 1 of 6,000, 4,000 and 4,000 lines, to see that the arm runs; in its third phase the chosen lines were about 43% additions, 37% subtractions and 20% of the third task (route 2: 58 / 29 / 14), with every score under 21. I wrote the predictions knowing those shares.

## 3. One question, one hypothesis

**Does choosing the replayed line by the relay system's own mismatch keep the middle task?**

*Hypothesis.* At the end of life A arm N has addition, subtraction and the third task each at 60% or more (median of five seeds, last reading). *Refuted* if subtraction is not at least 3 points above route 2's 52.4 at the median: then the choice does nothing that matters.

## 4. Predictions (mine)

| | Statement | Chance |
|---|---|---|
| N1 | subtraction 60 or more | 0.40 |
| N2 | subtraction above route 2's of life A on at least four seeds | 0.70 |
| N3 | addition 65 or more | 0.65 |
| N4 | the third task 60 or more | 0.55 |
| N5 | the whole hypothesis | 0.25 |
| N6 | of the lines chosen in the third phase, subtraction's are more than 35% | 0.60 |
| N7 | refuted (subtraction under 55.4) | 0.30 |

*What could make it fail (mine).* The largest mismatch among four may pick lines the system cannot yet learn rather than lines it is forgetting, and so feed the hardest sums of the task being read; early in a phase it will pick the new task's lines, which the real lines already supply; and taking the worst of four makes the lines learned from unlike the lines read, which may distort what is kept.

## 5. Cost, and what is not run

Five lives, about 7 CPU-minutes each (the four candidates are judged in numpy). Not run on seed 1 at length: nothing is chosen. Not run: other numbers of candidates; the same arm in the lives of four tasks or with the harsh third text; the choice by mismatch applied to old lines drawn by the program.
