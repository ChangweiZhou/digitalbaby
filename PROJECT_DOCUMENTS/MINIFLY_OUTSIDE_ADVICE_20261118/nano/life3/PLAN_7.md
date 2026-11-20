# Line `nano/life3`, seventh plan: route 3b, cells that count with look-ahead and a cap, on the long life (20 November 2026)

**Authorisation.** Mr. Xu, 20 November 2026, as given to this session by the coordinating session (in Chinese; the coordinator's translation): "You can go on trying boldly; we keep to the facts; besides the ideas I have mentioned, try others as well." And earlier, in English: "Keep experimenting with the agents." This step was asked of this line by the coordinating session in two messages that reached me only after the sixth plan was done (they were queued while I worked); it asked for it before the fourth task, and it comes after, which I say plainly.

**Source.** `claude/nano-life3-8mmhfm` at commit b96cd74. Copied into this folder from `claude/nano-sparse-8mmhfm` at commit f01a472, unchanged: `sparse4.py` (the look-ahead system of that line's `PLAN_10.md`) and that commit's `sparse.py`, which replaces the earlier copy here (it adds the option `ncap`; with the option off its arithmetic is the earlier file's). `life3.py` gains `la=1` and `ncap=` for arm S. `sparse4.py` holds its number of byte values as a module constant, which `life3.py` sets to fifteen. **Development level**; no checker gate, seal or outside review is claimed. Seeds, score, readings every 50,000 real lines, reporting: as `PLAN_1.md`.

## 1. What this follows

`nano/sparse`, as that line reports (its `PLAN_10.md` and the coordinating session's account): cells that also count what arrives two and three bytes after them, with cells keyed on the answering cell's own guesses of those bytes, answer about 91% of never-shown additions at 40,000 lines; with long reading they fall as the plain counting cells do (88 and 91 become 63 and 62 at 400,000 lines on two seeds), and with a cap (a cell that has counted more than 16 times speaks as if it had counted 16 times) 87 and 88 become 82 and 81. The long life with a second task after it was not run there. Here, `PLAN_1.md`: route 3 without look-ahead and without the cap falls from about 76 to 55.8 while addition alone is read.

## 2. The arms (life A: addition 400,000 lines, subtraction 200,000, `max` 200,000; nothing shown again)

- **3b**: `arm=S la=1 ncap=16`: the look-ahead system with that line's settings (at most four delays, eta 0.1, e 2, trusts born at 0.02 and those of the cells keyed on a guess at 0, the step shrinking with use) and the cap of 16.
- **3c**: `arm=S ncap=16`: route 3 of `PLAN_1.md` with the same cap and nothing else changed.
- Beside them, already run: route 3 of `PLAN_1.md` (no look-ahead, no cap).

The first two phases are the long life of two tasks that the coordinating session asked for, with a reading every 50,000 lines; the third phase extends the same runs by the third task. The short life (40,000 lines a task) that the first message asked for is NOT RUN: the second message put the long life first, and the budget does not hold both.

## 3. One question, one hypothesis (the hypothesis is the coordinating session's, registered here as given)

**With the cap, does the look-ahead system hold its level through long reading and keep it through a second task?**

*Hypothesis.* For 3b, never-shown addition is at least 75% at the end of the addition phase, and after subtraction at least 85% of that (median of five seeds, last reading; the mean of the last three readings beside it). *Refuted* if either fails. The same two figures are reported for 3c as predictions, not as its hypothesis.

## 4. Predictions (mine), before any run

Disclosure: before this plan, the compiled look-ahead kernel was checked against its numpy reference with fifteen byte values on a life of 280 lines (scores of a system that has read 280 lines). I have read that line's file for seed 73129 with the cap (82.2 at 400,000 lines with fourteen byte values). And two runs to see that the arms run at their full settings, seed 73129: 3b on 10,000, 2,000 and 2,000 lines of the three tasks (addition 73.6 and 81.2 in its phase, 87.0 after 2,000 lines of subtraction, 65.6 after 2,000 lines of `max`, with `max` at 86.8) and 3c on 10,000 and 2,000 lines (addition 73.0, then 70.4). The chances below were written before those two runs and are left as written. The check's output is `check_la_output.txt`.

| | Statement | Chance |
|---|---|---|
| L1 | 3b: addition 75 or more at the end of its phase | 0.80 |
| L2 | 3b: addition after subtraction at least 85% of that | 0.60 |
| L3 | 3b: subtraction (clean) 60 or more at the end of its phase | 0.60 |
| L4 | 3b: at the end of the life addition, subtraction and `max` each 60 or more | 0.50 |
| L5 | 3b: addition at the end of the life above route 1's 85.8 | 0.10 |
| L6 | 3c: addition 65 or more at the end of its phase (the cap stops most of the fall of `PLAN_1.md`) | 0.60 |
| L7 | 3c: addition 75 or more at the end of its phase | 0.30 |
| L8 | 3c: at the end of the life all three 60 or more | 0.60 |
| L9 | 3b above 3c on addition at the end of the life by 10 points or more | 0.55 |

## 5. Cost, and what is not run

3b: that line reports about 20 CPU-minutes and 3.0 million cells for 400,000 lines; a life of 800,000 lines should cost 40 to 50 CPU-minutes and up to about 1.5 GB a run. 3c: about 6 CPU-minutes. Five seeds each: about 4.5 CPU-hours, about 1.5 hours on four cores, in the background. This is beyond the three hours this line was given; I run it because the coordinating session asked for it and say so. Not run: the short life; 3b without the cap on the long life (that line ran it for one task: it falls); 3b on the life of four tasks or the harsh text; a numpy check at full size (the check is on a short life with at most two delays).

---

## 6. Addition 1 (20 November, written before its runs): the score after a read line, beside the old one

**What it follows.** A message of the coordinating session, received while the runs above were going: a fault of the measuring instrument, its own (`loop_small_case/ERRATA_20261120.md`, section 3, on `claude/friendly-darwin-8mmhfm` at 8ae14f646, which I have not read; the form is taken from the message). Every line, this one included, scores a never-shown pair as `cells2.py` does: the line alone, with line ends in the delays that reach before it (**the empty past**). No system meets that past after the first line of its life. The coordinating session reports that scored **after a read line** the relay system answers 82.9% at 40,000 lines where the old score gives 66.2, and 95.5 at 400,000 where it gives 84.8. So every figure of this line for a system built on the relay system (plain, reference, routes 1 and 2, the choice by mismatch) is a figure with the empty past, and the sizes of what those routes keep are open.

**The score after a read line, as implemented here** (`life3.py`, `past=5`, the default from this commit on): for answer byte t of a held-out line, a delay d that reaches before the line's start holds byte 10 + t - d of a line before it; that line is drawn by chance, for each held-out line, from the shown lines of the task being read in the present phase; the mean of five draws; a generator of its own, so that nothing else in a life changes and the empty-past figures of a run made again must equal the earlier files to the digit (a check). At the answer bytes the delays reach only the last one to four bytes of the line before (its answer digits and its line end), never its operator. For the counting cells the line before is written before the held-out line in what the kernel is given. A test on short lives, seed 73129: plain system at 50,000 lines of addition 68.2 with the empty past and 84.6 after a read line; route 3 at 5,000 lines 67.6 and 67.1.

**Runs** (nothing new but the score): life A again for P, O, K, R (`b1_*`) and for the choice by mismatch (`b6_N_*`), five reported seeds; the lives of this plan again for 3c and for the four seeds of 3b that ended before the score was in the program (`b7_*`; the fifth seed of 3b started after and has both scores in `a7_S3b_545353`). The two-task life is the end of the second phase of the same runs. NOT RUN again with the new score: the lives of four tasks, the harsh text, the orders, `PLAN_5.md`'s two arms, route 3 without the cap (about 4 CPU-hours together).

**No new hypothesis.** The earlier predictions stand as scored. Predictions (mine) for the new figures, at the end of life A: (Q1) the relay-based arms that keep a task read it at least 8 points higher after a read line than with the empty past, for addition, in each of O, K, R, N: chance 0.70; (Q2) the counting cells (3c, 3b) move by under 5 points on every task: 0.70; (Q3) route 2's middle task is still under 60 after a read line, and under the same arm's addition by 15 points or more: 0.55; (Q4) the order of the arms by their lowest task at the end of life A is the same by both scores (route 1, then 3b, then N and the reference, then route 3 capped, then route 2, with ties within 3 points not counted): 0.45; (Q5) the plain system still has both earlier tasks under 10: 0.90.

Cost: about 5.5 CPU-hours (the choice by mismatch 25 CPU-minutes a life, 3b 32), about an hour and a half on four cores, in the background.
