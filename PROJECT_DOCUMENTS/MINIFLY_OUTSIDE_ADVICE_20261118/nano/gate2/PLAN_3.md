# Line `nano/gate2`, third plan: a second task (addition, then subtraction with nothing of addition shown) (20 November 2026)

**Authorisation**, as quoted to this session by the coordinating session (session_017dSEpVed7SthEH3ngGAdNc): Mr. Xu (in Chinese): "我们接受bitter lesson, 因此你可以尝试，但避免spot fix, ad hoc tricks等等在其他的任务上无法迁移的手段。多尝试新想法。" and, most recently: "继续大胆尝试。" (the coordinating session's translation: "We accept the bitter lesson, so you may try things, but avoid spot fixes, ad hoc tricks and the like: means that do not transfer to other tasks. Try more new ideas." and "Go on trying boldly."). **Source:** `claude/friendly-darwin-8mmhfm` at a859b2f224f0adca576616c0fc96bf82699ec293; this branch at the commit that carries this file. Development level; no gate, seal or review is claimed. Written before any run of a life of two tasks, except a smoke test of the program on development seed 1 (2,000 lines of addition, then 1,000 of subtraction; I saw its two printed readings: it runs, and nothing was chosen on it).

## 1. The evidence, and the question

`nano/merge` has no life of two tasks. Other lines found that cells that count keep 92 to 100% of addition through a short life of subtraction, and the relay system 2%. The second answering cell (`PLAN_1.md`: 97.4 against 92.9 at 40,000 lines) adds 1.45 million learned parts that go on changing through the second task. *Question:* what does each answering cell keep of never-shown addition through subtraction, and what does each learn of subtraction?

## 2. The system, the life and the measures

The system of `PLAN_1.md` unchanged, the same program and numbers (`gate2.py`; first cell, second cell and control from one run). The text of subtraction is `loop_small_case/cells2.py`'s (the larger number first; three answer digits) and the split is that program's (the second permutation of the same random stream, which `merge.run_sums` already draws and discards): 500 pairs held out, 8,000 shown. Classes go on being found by `merge.py`'s rule (whenever the bytes read have doubled) through the whole life; nothing tells the system that the task has changed.

*The life.* Addition 40,000 lines, then subtraction 20,000 lines with no addition line. Seeds 73129, 93683, 116089, 406364, 545353; with classes, and without. If cost allows: addition 400,000 lines, then subtraction 200,000 lines, with classes, seeds 73129 and 93683.

*Readings:* every 5,000 lines of each task, and 1,000 and 2,000 lines after the second task starts (every 50,000 in the long life, with 1,000, 2,000, 5,000, 10,000, 20,000, 40,000 after the change). *At each reading, for each cell, each continuing its own answer bytes:* the 500 never-shown additions after a read addition line (five draws), after a read subtraction line (five draws; from the second task on this is the past the system has), and with an empty past; the never-shown subtractions on the clean score (the held-out pairs whose written line no shown pair writes, about 90 to 100 of the 500; `ERRATA_20261120.md`, section 1) after a subtraction line, after an addition line and with an empty past. (With nine delays the bytes of the line before that an answer byte can see are its last one to three: two answer digits and the line end. The operator of the line before is never seen.)

*Kept* is defined here, before the runs: for a cell, its never-shown additions after a read subtraction line, mean of the last three readings of the second task (10,000, 15,000, 20,000 lines), over its never-shown additions after a read addition line, mean of the last three readings of the first task. (Each is the past the system has at that time.) The same ratio with an addition line before in both, and with an empty past in both, is given beside it.

*Size.* As `PLAN_1.md`; the cells that count grow with the second task (reported at every reading). *Cost.* About two CPU-minutes a run with classes and one without; ten runs. The long life about 25 to 35 CPU-minutes a run.

## 3. Hypothesis, what refutes it, predictions

*Hypothesis (the coordinating session's, given to me to refute):* the second cell keeps at least 90% of its level on addition through the second task, because its parts are indexed by the byte at each delay and the operator stands at one of those delays when the answer is named, so what it learned for lines with '+' is touched little by lines with '-'. *Refuted if* it keeps under 75% (median of the five seeds, with classes, "kept" as defined above). Between 75 and 90: not shown.

*My view before the runs.* The argument covers one part in nine. For a sender and its level, the second cell's trust is a sum of nine parts; at an answer byte one of them is indexed by the operator, and the other eight are indexed by digits, '=' and line ends, which lines with '-' use as lines with '+' do. So subtraction lines move eight ninths of what an addition line reads. Against that: the senders that carry addition are cells keyed on sets of delays, and every kind whose delays include the operator's has separate cells for the two tasks, so what those cells hold is not touched; and subtraction's mismatch pushes the shared parts only where the same kind at the same level and the same byte is wrong for subtraction. I expect the second cell to lose a few points more than the first and to stay above it.

| | Statement (medians of five seeds, with classes, unless said) | Chance |
|---|---|---|
| K1 | the second cell keeps at least 90% | 0.50 |
| K2 | the second cell keeps at least 75% (the hypothesis is not refuted) | 0.85 |
| K3 | the first cell keeps at least 90% | 0.65 |
| K4 | the share the second cell keeps is smaller than the share the first cell keeps | 0.60 |
| K5 | at the end of the life the second cell is still above the first on never-shown additions (after a subtraction line, mean of the last three) | 0.70 |
| K6 | never-shown subtractions on the clean score at the end (after a subtraction line, mean of the last three): the second cell at least 2 points above the first | 0.55 |
| K7 | the control keeps a share within 3 points of the first cell's share | 0.60 |
| K8 | without classes: the second cell keeps at least 90% | 0.55 |
| K9 | the lowest reading of the second cell on addition in the second task is one of the first two (1,000 or 2,000 lines after the change), on at least three seeds of five | 0.40 |

## 4. Not run

Replay, a cap, or any means of keeping; a third task; subtraction first; the two tasks mixed. The long life without classes.

## 5. Outcome of the short life (20 November; `res/life_cls_seed*.json`, `res/life_none_seed*.json`, `out/life_*.txt`, `report_3.txt`)

Addition 40,000 lines, then subtraction 20,000 lines; five reported seeds; first cell, second cell and control from one run. Size at the end: 1.38 million cells that count with classes; second cell 1,451,520 parts, about 803,000 used. CPU: 123 to 131 s a run with classes, about 80 s without. 91 to 101 clean subtractions a seed. "Before": never-shown additions after a read addition line, mean of the last three readings of the first task. "End": never-shown additions after a read subtraction line, mean of the last three readings of the second task. Kept = end over before. Medians of five seeds, with the five values.

| Setting | Cell | Addition before | Addition at the end | Kept, % | Lowest reading in the second task (median; when) | Subtraction at the end, clean score, after a subtraction line |
|---|---|---|---|---|---|---|
| with classes | first | 92.9 | 75.4 (81.4, 78.5, 73.0, 70.6, 75.4) | **81.1** (87.7, 83.5, 79.3, 78.9, 81.1) | 50.2; 1,000 lines after the change on four seeds | 80.4 (83.0, 73.5, 85.3, 80.4, 78.9) |
| | second | 97.1 | 86.2 (79.5, 87.6, 87.9, 83.9, 86.2) | **88.8** (81.7, 89.4, 91.0, 87.2, 88.8) | 63.2; 1,000 lines after on all five | 89.4 (91.4, 87.0, 91.5, 85.8, 89.4) |
| | control | 92.5 | 61.3 (65.6, 67.2, 44.9, 51.1, 61.3) | 65.8 (70.9, 72.3, 48.6, 57.8, 65.8) | 18.6 | 83.1 |
| without classes | first | 79.4 | 76.3 | **96.1** (96.1, 100.1, 95.6, 95.7, 97.1) | 74.3; at the end | 80.4 |
| | second | 85.7 | 82.6 | **95.8** (94.6, 94.8, 95.8, 97.7, 96.2) | 73.3; 1,000 to 5,000 lines after | 87.2 (86.5, 87.1, 87.2, 87.8, 88.4) |
| | control | 83.6 | 71.6 | 83.4 | 66.8 | 83.1 |

The same share with an addition line before the held-out line at both ends: 81.1, 88.8, 65.1 with classes; 96.7, 95.5, 83.3 without. With an empty past at both ends: 87.6, 92.7, 78.9 with classes; 100.9, 96.0, 95.1 without. So what stands before the line matters little for the first and the second cell (as section 2 expected: the operator of the line before is out of reach), and the empty past, which silences the kinds that reach before the line, flatters all three.

The course of never-shown additions through the second task (after a subtraction line; median of five seeds; at 1,000, 2,000, 5,000, 10,000, 15,000, 20,000 lines of subtraction). With classes: first cell 50.2, 60.6, 67.6, 72.6, 80.1, 78.4; second cell 63.2, 74.8, 76.1, 83.7, 87.6, 88.1; control 23.8, 27.2, 36.9, 38.0, 68.2, 70.6. Without classes: first 80.4, 81.1, 79.8, 78.0, 77.0, 74.3; second 74.4, 75.7, 74.7, 79.7, 83.6, 83.6. By place at the end, with classes (hundreds, tens, units): first 96.8, 80.1, 100; second 97.4, 88.5, 100; control 83.3, 80.0, 100.

**What I take from it (mine).**

1. *The coordinating session's hypothesis is neither held nor refuted.* With classes the second cell keeps 88.8% at the median (one seed of five at 90 or more, none under 75): between the two lines drawn. Without classes it keeps 95.8%.
2. *The second cell keeps more than the first, not less, where classes are present* (88.8% against 81.1%; it ends 10.8 points above the first cell where it started 4.2 above), and it learns the second task better (89.4 against 80.4 on the clean subtractions, 87.2 against 80.4 without classes). My expectation that it would lose more was wrong.
3. *The loss is sudden and then mends, while only subtraction is read.* Addition is lowest 1,000 lines after the change (first cell 50, second 63, control 24, with classes) and climbs back for the rest of the second task, with no addition line shown. That is the shape of a reading that was upset and re-fits itself, and not of a store that is overwritten: the cells that hold addition are still there, the trusts and parts that weigh them are shared with the new task and move at once. In the control, whose nine equal parts make its step nine times the first cell's, the fall is to 19 to 24%: a larger step on a shared trust is what loses most.
4. *With classes the first cell keeps far less (81%) than without (96%)*, and less than other lines found for cells that count (92 to 100%). No finding of classes falls inside the second task here (the last is at 32,000 lines, the next would be at 64,000), so this is the class cells and their trusts under the new task, and not a new search. Not looked into further here.
5. *The hundreds digit is touched as well as the tens* (96.8 to 97.4 right at the end against 98.5 before; 83% in the control), which never happened in the one-task runs.

**Predictions.** K1 failed, narrowly (88.8, 90 asked). K2 held (88.8; no seed under 75). K3 failed (the first cell keeps 81.1). K4 failed (the second cell keeps the larger share). K5 held (86.2 against 75.4; four seeds of five, seed 73129 has 79.5 against 81.4). K6 held (89.4 against 80.4). K7 failed (the control keeps 65.8 against 81.1). K8 held (95.8). K9 held (five seeds of five). Five of nine.

## 6. Outcome of the long life (20 November; `res/lifelong_cls_seed*.json`, `out/lifelong_cls_seed*.txt`, `report_3long.txt`)

Addition 400,000 lines, then subtraction 200,000 lines; with classes; seeds 73129 and 93683 (two seeds: a reading, not a claim). 1.40 million cells that count at the end. CPU 672 and 702 s a run. "Kept" as section 2 defines it (mean of the last three readings of each task: 300,000 to 400,000 lines of addition; 100,000 to 200,000 lines of subtraction).

| Seed | Cell | Addition before | Addition at the end | Kept, % | Addition through the second task, after a subtraction line, at 1,000, 5,000, 20,000, 40,000, 100,000, 200,000 lines | Subtraction, clean score: best reading; at the end (mean of the last three) |
|---|---|---|---|---|---|---|
| 73129 | first | 78.8 | 73.9 | 93.8 | 7.5, 20.7, 29.6, 79.7, 78.0, 70.8 | 82.4; 68.9 |
| | second | 76.5 | 13.9 | **18.2** | 16.7, 13.2, 11.5, 23.7, 14.7, 11.6 | 89.5; 78.5 |
| | control | 83.6 | 79.5 | 95.1 | 0.7, 4.6, 4.2, 83.4, 85.0, 73.4 | 83.6; 64.5 |
| 93683 | first | 82.9 | 72.1 | 87.0 | 69.3, 79.6, 82.3, 83.4, 80.4, 67.4 | 74.2; 60.5 |
| | second | 91.9 | 71.4 | **77.7** | 56.5, 69.6, 75.2, 83.8, 84.4, 64.1 | 93.8; 77.2 |
| | control | 85.4 | 66.3 | 77.7 | 34.4, 60.2, 60.3, 87.0, 76.6, 59.7 | 83.1; 59.2 |

By place of the addition answer (hundreds, tens, units), seed 73129, second cell: 99.6, 21.6, 79.1 at 1,000 lines of subtraction and 96.2, 13.2, 97.8 at the end; first cell 99.5, 7.5, 100 at 1,000 lines and 95.4, 71.3, 100 at the end.

**What I take from it (mine; two seeds).**

1. *After a long first task the second cell can lose addition and not recover it.* On seed 73129 it stands at 12 to 24% for the whole of the second task, while the first cell and the control, which fall as low at first (7.5 and 0.7 at 1,000 lines), are back at 80 and 83 by 40,000 lines. On seed 93683 the second cell keeps 77.7%. So over the long life the coordinating session's hypothesis is refuted on one seed of two (18.2%, under its line of 75) and not held on the other. The short life's "keeps 88.8%" does not carry to the long one.
2. *The sudden loss is far deeper after a long first task* (7.5% and 69% for the first cell 1,000 lines after the change, against 50% at the median after the short one), and mending takes 20,000 to 40,000 lines in place of 5,000 to 15,000: the trusts' steps have shrunk with 400,000 lines of use.
3. *The second cell learns the second task better at every reading on both seeds* (clean subtractions 78.5 and 77.2 at the end against 68.9 and 60.5 for the first cell), and every cell's subtraction falls with reading after its best (89.5 to 78.5; 82.4 to 68.9), as addition did in step 2.
4. This was on seeds 73129 and 93683 only, with classes only, and the second cell on seed 73129 had already fallen to the first cell's level before the change (step 2). I do not know from these runs whether the lasting loss on that seed comes from the state the long reading left the parts in, or from the change of task alone.

**Predictions.** None were written for the long life apart from the hypothesis itself; it is scored in point 1.
