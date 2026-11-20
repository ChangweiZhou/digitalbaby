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

---

## 6. Outcome (20 November 2026; development level; readings are mine)

Runs: five lives of 800,000 real lines. Cost: about 25 CPU-minutes a life, not the 7 I wrote (four lives at once share the memory the numpy pass needs); 2.1 CPU-hours in all. Files: `res/a6_N_*`, `table6.txt`.

**Life A at its end.** Never-shown pairs, whole answer, %; median of five seeds, last reading [mean of the last three readings]; subtraction on the clean score.

| Arm | add | sub | max | lowest of the three |
|---|---|---|---|---|
| R, route 2: one replayed line, drawn in proportion to counts | 73.4 [75.4] | 52.4 [54.7] | 64.2 [60.6] | 52.4 |
| **N: one replayed line, the worst by the system's own mismatch of four drawn** | **67.8** [69.4] | **66.9** [66.7] | **63.4** [59.5] | 63.4 |
| R2, two replayed lines (`PLAN_5.md`) | 80.2 | 56.9 | 62.4 | 56.9 |
| O, the program's old lines (reference) | 78.2 [77.7] | 63.2 [63.8] | 64.2 [59.5] | 63.2 |

Each seed (73129, 93683, 116089, 406364, 545353): addition 64.2, 78.2, 60.2, 70.4, 67.8; subtraction 62.6, 66.9, 68.4, 69.4, 62.4; `max` 62.4, 58.0, 64.8, 68.2, 63.4. Paired with route 2 by seed: subtraction +9.4, +14.5, +23.6, +17.0, +12.0 (five of five; median +14.5); addition -9.8, +2.2, -12.0, -2.0, -5.6 (median -5.6); `max` -2.0, -2.2, +5.6, -5.0, -0.8. Of the lines chosen in the third phase, 37 to 42% are additions, 36 to 40% subtractions, 21 to 25% of the third task (route 2: 58 / 29 / 14); in the second phase 61 to 73% additions. Every chosen line is a line that was shown.

**The hypothesis holds**: addition 67.8, subtraction 66.9, the third task 63.4, each 60 or more at the median on the last reading; addition and subtraction on every seed; the third task on four seeds of five (58.0 on one), and 59.5 as the mean of its last three readings, which is the figure of a task still being learned. Not refuted: subtraction is 14.5 points above route 2's.

| | Statement | Chance | Outcome |
|---|---|---|---|
| N1 | subtraction 60 or more | 0.40 | held (66.9; every seed) |
| N2 | subtraction above route 2's on at least four seeds | 0.70 | held (five of five) |
| N3 | addition 65 or more | 0.65 | held (67.8; two seeds under 65) |
| N4 | the third task 60 or more | 0.55 | held by the last reading (63.4; mean of three 59.5) |
| N5 | the whole hypothesis | 0.25 | held |
| N6 | subtraction's part of the chosen lines above 35% | 0.60 | held (36 to 40%) |
| N7 | refuted | 0.30 | did not happen |

**Readings (mine).**

1. **Choosing the replayed line by the system's own mismatch moves the replay to the task that is slipping, with nothing told about tasks**: subtraction's part of the replay rises from 29% to 38%, and subtraction ends at 66.9 where route 2 left it at 52.4, on every seed. It is paid for in addition (5.6 points at the median, four seeds of five), whose part falls from 58% to 39%. The three tasks end within 5 points of one another; the lowest figure is 63.4 against 52.4.
2. **It matches the program's own old lines without the program**: lowest task 63.4 against the reference's 63.2; subtraction above the reference on three seeds of five, addition under it on four. The reference knows which lines are old and shares them equally; this arm knows neither.
3. **It gives a fifth to a quarter of the replay to the task being read**, more than route 2 does (14%), since a task being learned has large mismatches; the third task does not gain from it (63.4 against 64.2). A rule that set aside lines much like the line just read would spend that part better; not run.
4. **What this does not show.** One life of three tasks, five seeds, no repeat; four candidates and blocks of 500 set once and not varied; the mismatch is measured with the line standing alone and with strengths up to 500 lines old; the cost is four drawn lines judged for one learned from. Whether it holds with four tasks, or with the harsh third text, where the reference itself fails, is not known: by the fourth-task figures of the reference (58 to 66) I would expect about that and no better, since this arm reaches the reference and not beyond it. The store is still every shown line.
