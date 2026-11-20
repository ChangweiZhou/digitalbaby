# Line `nano/gate2`, fourth plan: three-digit sums, nothing chosen there (20 November 2026)

**Authorisation**, as quoted to this session by the coordinating session (session_017dSEpVed7SthEH3ngGAdNc): Mr. Xu (in Chinese): "我们接受bitter lesson, 因此你可以尝试，但避免spot fix, ad hoc tricks等等在其他的任务上无法迁移的手段。多尝试新想法。" and, most recently: "继续大胆尝试。" (the coordinating session's translation: "We accept the bitter lesson, so you may try things, but avoid spot fixes, ad hoc tricks and the like: means that do not transfer to other tasks. Try more new ideas." and "Go on trying boldly."). **Source:** `claude/friendly-darwin-8mmhfm` at a859b2f224f0adca576616c0fc96bf82699ec293; this branch at the commit that carries this file. Development level; no gate, seal or review is claimed. Written before any run on three-digit sums except a smoke test of the program on development seed 1 (2,000 lines, without classes; one printed reading seen: first 40.8, second 54.8, control 47.4; nothing chosen on it).

## 1. The evidence, and the question

`nano/merge` on three-digit sums, 80,000 lines, with nothing chosen there: 92.3 without classes and 94.1 with (median of five seeds, old score); a look-ahead system has 99.7. In that line's stored files the errors are at the second answer byte (the hundreds: 96% right) and a little at the third. The second answering cell was found and repeated on two-digit sums only. *Question:* with the same program and the same numbers, is the second cell above the first on three-digit sums?

## 2. The system, runs and measures

The system of `PLAN_1.md` unchanged (`gate2.py` with `nd=3`): `merge.py`'s three-digit text and split (a fifth of the million pairs never shown, the first 500 of them scored), nine delays, the same kinds, classes at the frozen setting. 80,000 lines; seeds 73129, 93683, 116089, 406364, 545353; with classes. Readings every 10,000 lines; first cell, second cell and control from one run; whole answer of four bytes, each cell continuing its own bytes. With nine delays the first answer byte sees one byte of the line before, its line end, and the later ones none: so the score after a read line and the score with an empty past are the same number here, and one is reported. My first cell is compared with `../merge/res/s3_cls_seed*.json` reading by reading.

*Size.* About 2.0 million cells that count (as that line's files); the second cell 1,451,520 parts (the number used is reported). *Cost.* About four to five CPU-minutes a run (that line: 150 to 160 s without the second cell); five runs. If time remains, the same without classes.

## 3. Hypothesis, what refutes it, one prediction (mine, before the runs)

*Hypothesis.* The second cell is above the first on three-digit sums: its median over the five seeds (mean of the last three readings) is at least 1 point above the first cell's and the control's gain is under half of it. *Refuted if* the second cell's median is not above the first cell's, or the control gives as much.

*The prediction.* The second cell's median is at least 1.5 points above the first cell's (mean of the last three readings): chance 0.60. My reasons: on two-digit sums the whole gain was at the one place where a kind that is exact elsewhere says chance things (the tens digit), and three-digit sums have two such places; but there the first cell is already at 94 with errors at one place, and that place (the hundreds digit) needs four operand digits and a carry from the units, which no kind of four delays holds, so part of the loss is in what the cells hold and no reading can repair it. Beside it, with smaller weight: the second cell's median at least 97 (0.20); the control within 1 point of the first cell (0.65).

## 4. Not run

More than 80,000 lines; kinds of more than four delays or more than nine delay lines; look-ahead; four-digit sums.

## 5. Outcome (20 November; `res/s3_cls_seed*.json`, `out/s3_cls_seed*.txt`, `report_4.txt`)

Never-shown three-digit additions, whole answer of four bytes right, %, each cell continuing its own bytes; 80,000 lines; with classes; five reported seeds; first cell, second cell and control from one run. Size: 2.03 million cells that count; second cell 1,451,520 parts, 722,000 to 744,000 used. CPU 209 to 232 s a run. (After a read line and with an empty past are the same number here, as section 2 said; the program confirms it reading by reading.)

| Cell | Last reading, by seed; median | Mean of the last three; median | By place at the end, median (thousands, hundreds, tens, units) |
|---|---|---|---|
| first | 95.6, 94.2, 94.2, 94.6, 96.2; 94.6 | 94.9, 93.0, 94.0, 94.1, 95.0; **94.1** | 99.6, 95.6, 99.4, 100 |
| second | 97.0, 97.8, 96.0, 97.0, 97.4; 97.0 | 97.1, 96.9, 96.1, 96.5, 97.7; **96.9** | 99.6, 97.0, 100, 100 |
| control | 94.6, 94.6, 93.8, 94.0, 95.6; 94.6 | 94.3, 93.3, 93.5, 93.3, 95.1; 93.5 | 99.6, 95.6, 98.8, 100 |

Second cell less first cell, seed by seed, mean of the last three: +2.1, +3.9, +2.1, +2.4, +2.7. Control less first: -0.6, +0.3, -0.5, -0.8, +0.1. The second cell is above the first at every reading of every seed (from 5,000 lines: 88.8 to 94.0 against 83.6 to 90.2). Every reading of my first cell equals `../merge/res/s3_cls_seed*.json`.

**The hypothesis holds** (my reading): with the program and numbers of two-digit sums and nothing chosen here, the second cell is 2.9 points above the first at the median (96.9 against 94.1), on five seeds of five, and the control gives nothing. It removes every error at the tens digit (99.4 to 100) and a third of those at the hundreds digit (95.6 to 97.0). What remains is at the one place whose rule needs more than any kind of four delays holds (four operand digits and a carry from the units), as section 3 expected: it is a loss in what the cells hold, and the look-ahead system's 99.7 comes from holding it.

**Predictions.** The prediction held (+2.9 at the mean of the last three, 1.5 asked; chance given 0.60). Beside it: "the second cell's median at least 97" failed by a tenth at the mean of the last three (96.9; 97.0 at the last reading); "the control within 1 point of the first cell" held (-0.5).
