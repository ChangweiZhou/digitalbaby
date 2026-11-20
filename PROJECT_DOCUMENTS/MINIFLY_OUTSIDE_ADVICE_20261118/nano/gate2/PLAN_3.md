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
