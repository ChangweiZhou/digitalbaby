# Line `nano/gate2`, first plan: a repeat, from its written description alone, of the second answering cell (20 November 2026)

By a separate session started by the coordinating session (session_017dSEpVed7SthEH3ngGAdNc). **Authorisation**, as quoted to this session by the coordinating session: Mr. Xu (in Chinese): "我们接受bitter lesson, 因此你可以尝试，但避免spot fix, ad hoc tricks等等在其他的任务上无法迁移的手段。多尝试新想法。" and, most recently: "继续大胆尝试。" (the coordinating session's translation: "We accept the bitter lesson, so you may try things, but avoid spot fixes, ad hoc tricks and the like: means that do not transfer to other tasks. Try more new ideas." and "Go on trying boldly."). **Source:** `claude/friendly-darwin-8mmhfm` at a859b2f224f0adca576616c0fc96bf82699ec293; branch `claude/nano-gate2-8mmhfm`; files changed only in this folder. **Status:** development level; no checker gate, seal or review is claimed. Written before the program of this plan was run on anything.

## 1. What is repeated, and from what

The result is the coordinating session's (`../table/PLAN_15.md`): a second answering cell that hears the same senders as the first and the first cell's branches, with a part of the trust for each delay line indexed by (kind of sender, the sender's level of counts, the byte at that delay), is reported some 4 to 7 points above the first cell on never-shown two-digit sums with classes, and a control whose parts read a constant gives nothing.

**What I have read:** `AGENTS.md`; the roadmap's sections 5, 8, 9, 10; `../table/PLAN_15.md` (sections 2 and 4 are the description; section 5 for step 2); `loop_small_case/ERRATA_20261120.md`; `../table/past_cells.py` (the form of the score after a read line); `../merge/merge.py` and `../merge/REPORT.md`; `loop_small_case/cells2.py`. **What I have not opened and will not open until my figures on the five reported seeds exist:** `../table/gate.py`, `../table/gate_all.py`, `../table/kern_gate.c`, the files under `../table/res` and `../table/out15`. (I listed the names of the `past_cells_*` files in `../table/res` while looking for the score's form and opened none.) The time of opening and what differed will be entered below.

## 2. The exact system

*The base*, used as it is by importing `../merge/merge.py` (its class `Sys`, its text, its split, its search for classes): two-digit sums, 14 byte values, nine delay lines, a kind of cell for every set of at most four delays (256 kinds), e = 2.0, step of the trust 0.1 over the square root of one plus the uses of that entry, trusts born 0.02; classes at that line's frozen setting (32 pairs, eJ = 0.02, found anew whenever the bytes read have doubled, first at 10,000 bytes), and the same without classes.

*The addition, as I read the description.* For every byte:

1. The first cell is merge.py's, unchanged: branches a1(v) = sum over speaking senders k of T[k][level of k] s_k(v), p1 = softmax(a1); its trusts change as in merge.py.
2. The second cell: a2(v) = a1(v) + sum over speaking senders k of w_k s_k(v), with w_k = sum over delays j = 1..9 of G[k][level of k][j][the byte at delay j]; p2 = softmax(a2).
3. Every entry of G that was read changes by (0.1 over the square root of one plus the uses of that entry) times (s_k(y) less the sum over v of p2(v) s_k(v)), and its count of uses rises by one. G is born at nothing. The updates of both cells use the probabilities computed before any update at that byte; the cells count afterwards, as in merge.py.
4. When classes are found anew, the parts (and their counts of uses) of a class kind move with the first cell's trust for it: kept where the pair (kind, refined delays) is kept, nothing where it is new.
5. *The control:* the same cell with every part reading byte value 0 whatever stands at its delay (so nine equal parts for each kind and level, each with its own count). It runs in the same run as the second cell, from the same first cell, and neither touches the other or the first.

Choices the description leaves to me, fixed here before any run: the count of uses is kept for each entry of G separately; the level of a sender is the one the first cell uses at that byte; a silent sender (not yet counted) contributes nothing and its parts do not change; at the score each cell continues its own answer bytes.

*Size.* The second cell: 9 x 288 x 40 x 14 = 1,451,520 parts with classes (1,290,240 without), each with a count of uses; the number actually used is reported. The base: about 1.2 million cells that count (as that line reports). The control the same again.

*Programs.* `gate2.py` (the runner and a numpy reference of the loop) and its compiled loop (text inside `gate2.py`, written out as `build/kern_gate2.c`). Checks (`gate2.py check`, output committed): the compiled loop against the numpy reference on a prefix; the first cell of my loop against merge.py's compiled loop on the same input; my runner's first cell against `merge.run_sums` reading by reading on a short run with classes; and, in the runs themselves, my first cell's old score against that line's stored files for the five seeds.

## 3. Runs and measures

Seeds 73129, 93683, 116089, 406364, 545353; 40,000 lines; with classes and without; first cell, second cell and control from one run. Then seeds 3001 to 3005 (used by no line, by the coordinating session's word), with classes. Readings every 5,000 lines. At each reading, the 500 never-shown additions, whole answer right, each cell continuing its own answer bytes: **after a read line** (the bytes before each held-out line are those of a shown line; five draws from `default_rng([seed, 7])`, as `past_cells.py`; the mean of the five) and **with an empty past** (line ends before the line). The last reading and the mean of the last three, for both scores.

*Cost.* One run about two to four CPU-minutes with classes; fifteen runs; under one CPU-hour. Four cores, at most three runs at once.

## 4. Question, hypothesis, what refutes it, predictions (mine, before any run)

*Question.* Does the second answering cell, written from PLAN_15.md's sections 2 and 4 by someone who has not seen its program, give the gain reported?

*Hypothesis.* It does: on the five reported seeds with classes, scored after a read line at the last reading, the second cell's median is at least 2 points above the first cell's, and the control's gain is under half of the second cell's. *Refuted if* the median gain is under 1 point, or the control gives as much as the second cell. Between: not shown.

| | Statement | Chance |
|---|---|---|
| R1 | with classes, five reported seeds, after a read line, last reading: the second cell's median at least 2 points above the first cell's | 0.75 |
| R2 | the same: the second cell's median at least 95.5 | 0.55 |
| R3 | on seeds 73129 and 406364 my second cell is within 1.5 points of the figures given to me (97.4 and 96.8) | 0.50 |
| R4 | with classes: the control's median at most 1 point above the first cell's | 0.75 |
| R5 | without classes: the second cell's median at least 3 points above the first cell's | 0.65 |
| R6 | without classes: the control gives more than half of the second cell's gain (medians) | 0.55 |
| R7 | seeds 3001 to 3005, with classes: the second cell's median at least 2 points above the first cell's | 0.70 |
| R8 | my first cell's old score (empty past, last reading) equals that line's stored figure on all five reported seeds, with classes and without | 0.90 |
| R9 | on every one of the ten seeds with classes the second cell is at or above the first (after a read line, mean of the last three) | 0.60 |

## 5. Not run

Prose (the description's other half; the coordinating session reports it, and it is not this line's task). The side arm of PLAN_15.md's section 4. Any number other than the description's. Long reading, the second task and three-digit sums are steps 2 to 4, each with its own plan before its runs.

## 6. Seen before this plan was committed

The check (`check_output.txt`) includes one short run of the whole runner on development seed 1 (4,000 lines, 8 pairs of classes; needed to compare my first cell with `merge.run_sums`), and its one reading was printed: first cell 71.2 after a read line and 68.4 with an empty past, second cell 66.4 and 70.6, control 61.6 and 61.6. Nothing was chosen on it and the predictions above were written before it.
