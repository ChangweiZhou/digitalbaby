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
