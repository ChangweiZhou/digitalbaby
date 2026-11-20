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

## 7. Outcome (20 November; the runs of section 3; `res/s40_*.json`, `out/s40_*.txt`, `report_1.txt`)

Never-shown two-digit additions, whole answer right, %, each cell continuing its own answer bytes; 40,000 lines; first cell, second cell and control from one run. Size: 1.23 million cells that count with classes (1.21 million without); second cell 1,451,520 parts with classes, of which about 708,000 were ever used (control: 70,000); 1,290,240 without, 664,000 used. CPU: 77 to 89 s a run with classes, 41 to 46 s without (one core).

| Setting, seeds | Cell | After a read line: last reading by seed; median | The same, mean of the last three; median | Empty past, last; median |
|---|---|---|---|---|
| with classes; 73129, 93683, 116089, 406364, 545353 | first | 93.2, 95.2, 92.4, 89.6, 92.9; **92.9** | 92.9, 94.0, 92.1, 89.5, 93.0; 92.9 | 92.2, 94.2, 93.6, 87.6, 92.2; 92.2 |
| | second | 97.4, 98.2, 96.4, 96.8, 97.4; **97.4** | 97.4, 98.0, 96.6, 96.2, 97.1; 97.1 | 97.0, 98.6, 96.8, 96.8, 96.6; 96.8 |
| | control | 92.2, 93.8, 92.8, 88.5, 93.6; **92.8** | 92.5, 93.0, 92.4, 88.3, 93.2; 92.5 | 89.4, 92.0, 91.6, 87.2, 93.2; 91.6 |
| without classes; the same seeds | first | 82.0, 80.9, 80.1, 78.6, 77.8; **80.1** | 82.6, 80.0, 79.4, 78.5, 78.5; 79.4 | 79.2, 79.0, 77.6, 76.0, 77.2; 77.6 |
| | second | 88.1, 84.8, 86.3, 84.6, 85.6; **85.6** | 87.9, 85.7, 86.9, 84.5, 85.3; 85.7 | 87.8, 85.6, 86.0, 82.4, 86.6; 86.0 |
| | control | 83.9, 83.4, 83.8, 80.0, 82.5; **83.4** | 85.8, 84.1, 83.5, 81.2, 83.6; 83.6 | 81.8, 81.2, 82.2, 79.0, 82.0; 81.8 |
| with classes; 3001 to 3005 (new) | first | 89.7, 92.5, 90.1, 90.8, 90.9; **90.8** | 90.0, 92.4, 89.7, 90.3, 90.9; 90.3 | 87.4, 91.8, 88.4, 90.6, 89.4; 89.4 |
| | second | 94.6, 97.3, 96.2, 95.8, 97.4; **96.2** | 95.2, 97.1, 96.0, 95.9, 96.8; 96.0 | 93.2, 96.6, 96.4, 96.8, 97.0; 96.6 |
| | control | 89.0, 93.1, 91.2, 90.6, 91.3; **91.2** | 89.8, 92.7, 90.0, 90.1, 90.7; 90.1 | 87.6, 92.6, 90.2, 88.8, 90.0; 90.0 |

Second cell less first cell, seed by seed, after a read line, mean of the last three: with classes +4.5, +4.0, +4.5, +6.7, +4.1 (reported seeds) and +5.2, +4.7, +6.3, +5.6, +5.9 (new seeds); without classes +5.4, +5.7, +7.5, +6.1, +6.7. The control less the first cell: with classes -0.4, -1.0, +0.3, -1.1, +0.2 and -0.2, +0.3, +0.3, -0.2, -0.2; without classes +3.2, +4.1, +4.1, +2.7, +5.1.

By place of the answer (after a read line, last reading, median of the five reported seeds): the whole of the difference is at the tens digit. With classes: first cell 98.6, 93.0, 100 (hundreds, tens, units); second cell 98.5, 97.6, 100; control 98.6, 93.1, 100. Without classes: 98.6, 80.1, 100; 98.7, 85.6, 100; 98.7, 83.4, 100.

**It repeats** (my reading). Written from the description alone, the second cell stands 4.4 points above the first at the median of the five reported seeds with classes (97.4 against 92.9; ten seeds of ten above, by 4.0 to 6.7 at the mean of the last three readings), and 5.4 points above on five seeds no line had used (96.2 against 90.8). The control gives nothing with classes (92.8 and 91.2). Without classes the second cell gains 5.6 points and the control 3.3: more than half of that gain is not from the bytes, as the coordinating session said. About one error in three at the tens digit remains with classes (97.6% of tens digits right).

**The check of my first cell.** On all five reported seeds, with classes and without, every reading of my first cell at the old score equals the corresponding reading in `../merge/res/s40_cls_seed*.json` and `s40_none_seed*.json` (ten files, eight readings each; `report_1.txt`).

**Predictions.** R1 held (+4.4). R2 held (97.4). R3 held (97.4 and 96.8: the same to the digit). R4 held (control 92.8 against 92.9). R5 held (+5.6). R6 held (the control gives 3.3 of 5.6 at the last reading, 4.3 of 6.3 at the mean of the last three). R7 held (+5.4). R8 held (all ten). R9 held (ten of ten). Nine of nine: but I had been given the first figures before I wrote them, so they test the description and the program, and not my foresight.

## 8. When I opened the coordinating session's files, and what differed

Opened at 20:25 UTC on 20 November, after the fifteen runs above had ended, their table had been written (`report_1.txt`) and both had been committed and pushed (d4ce887): `../table/gate.py` (the kernel's text is inside it; there is no `kern_gate.c` in this checkout) and `../table/check15_output.txt`. The result files of that plan are not in this checkout; I read them from `claude/friendly-darwin-8mmhfm` at 681f168 by fetching the branch. In doing so I also saw that commit's title and the two result files of 400,000 lines (second cell 75.5 and 90.4 after a read line at the end, on seeds 73129 and 93683), after `PLAN_2.md` had been committed and its runs started.

*Figures.* 20 of that line's result files have a counterpart here (seeds 73129, 93683, 116089, 406364, 545353; with classes and without; the second cell and the control). In all 20, every one of the eight readings is the same to the last digit for both cells and both scores.

*Programs.* The arithmetic is the same. Differences, none of which changes a figure: (1) that program runs the control as a separate run in place of the second cell; mine runs the second cell and the control side by side from one first cell; (2) it stores the parts as [delay][kind][level][byte], mine as [kind][level][delay][byte]; (3) it changes the first cell's trust and the second cell's parts sender by sender in one loop, mine in two loops (the same values: both use probabilities computed before any change); (4) it skips a sender whose parts sum to exactly nothing; (5) it wraps `merge.run_sums` and scores at that program's readings, mine re-writes the loop of readings (checked against `merge.run_sums`); (6) it has a third arm (the side arm of PLAN_15.md) that I did not write. Every choice I listed in section 2 as left open by the description (a count of uses for each entry; the first cell's level; a silent sender changes nothing; each cell continues its own bytes; parts and counts both follow the trust across a finding) turned out to be the choice that program made. So the description in PLAN_15.md, sections 2 and 4, with `past_cells.py` for the form of the score, was enough to rebuild the result exactly.

*A limit of this repeat.* It is a repeat of the program from its description on the same base system, the same text, the same split and (for five seeds) the same random draws; it shows that the result is not an accident of that program and holds on five further seeds. It does not show that it holds on another base system or another text: that is what steps 2 to 4 ask.
