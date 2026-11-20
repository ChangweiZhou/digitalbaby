# Line `nano/gate2`, second plan: the second answering cell over a long reading (20 November 2026)

**Authorisation**, as quoted to this session by the coordinating session (session_017dSEpVed7SthEH3ngGAdNc): Mr. Xu (in Chinese): "我们接受bitter lesson, 因此你可以尝试，但避免spot fix, ad hoc tricks等等在其他的任务上无法迁移的手段。多尝试新想法。" and, most recently: "继续大胆尝试。" (the coordinating session's translation: "We accept the bitter lesson, so you may try things, but avoid spot fixes, ad hoc tricks and the like: means that do not transfer to other tasks. Try more new ideas." and "Go on trying boldly."). **Source:** `claude/friendly-darwin-8mmhfm` at a859b2f224f0adca576616c0fc96bf82699ec293; this branch at the commit that carries this file. Development level; no gate, seal or review is claimed. Written while the last runs of the first plan were finishing (three of the five reported seeds with classes had ended and I had read them: second cell 96.4 to 98.2 after a read line at 40,000 lines against 92.4 to 95.2 for the first) and before any run longer than 40,000 lines; I have not opened the coordinating session's program or result files.

## 1. The evidence, and the question

Every counting system of this programme loses with long reading: `nano/merge`'s frozen setting has 78.9 and 81.3 at 400,000 lines (seeds 73129 and 93683, old score) where it has 92 and 94 at 40,000. The relay system gains (95.5% after a read line at 400,000 lines). *Question:* does the second answering cell hold its level of 40,000 lines over 400,000 lines, with nothing added, and where in the reading and at which place of the answer does it not?

## 2. The system and the runs

The system of `PLAN_1.md` unchanged (`gate2.py`, the same program and numbers; with classes; first cell, second cell and control from one run). 400,000 lines; seeds 116089, 406364, 545353 first (the coordinating session runs 73129 and 93683), then 73129 and 93683 for figures of my own to set beside the coordinating session's. Readings at 5,000, 10,000, 20,000, 40,000, 50,000 and every 50,000 lines; both scores (after a read line, five draws; an empty past); the three cells; by place of the answer. The result file is rewritten at every reading, so a stopped run keeps what it has.

*Size.* As `PLAN_1.md`: 1,451,520 parts in the second cell (the number used is reported at every reading); the base grows with the reading (reported at every reading). *Cost.* About fifteen to twenty CPU-minutes a run by the first plan's rate (80 CPU-seconds for 40,000 lines); five runs; three at a time.

## 3. Hypothesis, what refutes it, predictions (mine, before the runs)

*Hypothesis* (the coordinating session's question, put as a statement so that it can fail): the second cell holds its level: on the median of the seeds run, its mean of the last three readings at 400,000 lines, after a read line, is within 3 points of its own reading at 40,000 lines in the same run. *Refuted if* it is more than 6 points below. Between: not shown. I give the hypothesis 0.35: the parts know the place in the line, which is one known cause of the fall (cells keyed on digits alone go on counting followers of other places), but the cells' counts themselves go on mixing places and nothing caps what a long-counted cell says.

| | Statement | Chance |
|---|---|---|
| L1 | the hypothesis as stated (median within 3 points of the level at 40,000 lines) | 0.35 |
| L2 | on at least two of the first three seeds the second cell's mean of the last three readings, after a read line, is at least 90 | 0.45 |
| L3 | on every seed run the second cell is at least 8 points above the first at the end (mean of the last three, after a read line) | 0.60 |
| L4 | the first cell falls below 85 at the end on every seed run (after a read line, mean of the last three) | 0.75 |
| L5 | the control ends within 3 points of the first cell on every seed run (either side) | 0.55 |
| L6 | where the second cell loses, the loss is at the tens digit (the middle place of the answer): at the end the hundreds and units are each right at 98% or more on every seed | 0.60 |

## 4. Not run

The system without classes over the long reading; a cap on long-counted cells or any other addition; more than 400,000 lines.

## 5. Addition 1 (20 November; written after the three runs of section 2 on seeds 116089, 406364 and 545353 had ended and I had read them, `report_2.txt`; before the runs it describes): where the second cell falls

*The evidence* (my three runs, with classes, never-shown additions after a read line; the whole table is in `report_2.txt`). The second cell does not hold: at 400,000 lines it ends at 74.8, 73.1 and 91.6 (first cell 76.0, 73.5, 80.3; control 84.7, 79.9, 82.4), where it had 96.4, 96.8 and 97.4 at 40,000 lines. It does not fall evenly. Seed 116089: 96.4 at 50,000 lines, 87.0 at 100,000, 90.6, 88.2, 89.9 at 150,000 to 250,000, then 76.4 at 300,000 and it stays there. Seed 406364: 90.8 at 250,000 and 76.5 at 300,000. Seed 545353: 95.6 at 50,000, 84.7 at 100,000, back to 93.8 at 150,000, and no fall at 300,000. `merge.py` finds classes anew whenever the bytes read have doubled: in this life at 64,000, 128,000 and 256,000 lines, and each large fall lies between the readings on either side of one of those. The first cell falls more evenly (92 to 76 to 80), and the control ends *above* the first cell (by 7.0, 6.2 and 1.0 at the mean of the last three readings).

*Hypothesis (mine).* The second cell's large falls happen at the findings of classes, and not between them. Across a finding the first cell's trust and the second cell's parts for a kept pair of (kind, refined delays) are carried over, but the class cells themselves are made anew: another grouping, other pooled counts, other levels. The second cell holds nine times as many parts for each class kind as the first cell holds trusts, each of them fitted to the old cells, and its step has shrunk with use, so it cannot follow. *Refuted if*, on both seeds below, the second cell's readings 1,000 lines before and 1,000 lines after the finding at 256,000 lines differ by under 3 points (then the fall is elsewhere between 250,000 and 300,000 lines).

*Runs.* No mechanism is changed. (a) Seeds 116089 and 406364, with classes, 400,000 lines again with further readings (they change nothing that is learned) at 63,000, 65,000, 70,000, 127,000, 129,000, 135,000, 255,000, 257,000 and 265,000 lines; at every reading also the same score with only the class kinds speaking and with only the plain kinds speaking (as `merge.run_sums` does at the end of its runs); the list of kept pairs at every finding. (b) The system without classes over 400,000 lines, seeds 116089, 406364, 545353 (taken out of "not run" in section 4, because it separates the two accounts: without classes nothing is found anew). Cost: about eight CPU-minutes a run for (a), four for (b).

| | Statement | Chance |
|---|---|---|
| D1 | seed 116089: the second cell at 257,000 lines is at least 8 points below itself at 255,000 lines (after a read line) | 0.60 |
| D2 | the same on seed 406364 | 0.55 |
| D3 | with only the class kinds speaking, the second cell falls by at least 8 points across the finding at 256,000 lines, on at least one of the two seeds | 0.50 |
| D4 | without classes, the second cell ends at least 3 points above the first on all three seeds (mean of the last three, after a read line) | 0.60 |
| D5 | without classes, the second cell at the end is within 5 points of its own reading at 40,000 lines on at least two seeds of three | 0.30 |
| D6 | without classes, no reading-to-reading fall of the second cell after 50,000 lines is larger than 5 points, on all three seeds | 0.55 |

*Not run.* Any repair (parts born anew at each finding, a step that does not shrink, a cap): each would be a new mechanism and needs its own plan.

## 6. Outcome of the five long runs (section 2; `res/s400_cls_seed*.json`, `report_2.txt`)

With classes; 400,000 lines; never-shown additions after a read line; 1.24 million cells that count; second cell 1,451,520 parts, about 776,000 used; CPU 410 to 441 s a run.

| Seed | First cell: at 40,000 lines; at the end (last; mean of the last three) | Second cell | Control |
|---|---|---|---|
| 73129 | 93.2; 77.2; 78.8 | 97.4; 75.5; **76.5** | 92.2; 82.6; 83.6 |
| 93683 | 95.2; 81.2; 82.9 | 98.2; 90.4; **91.9** | 93.8; 84.4; 85.4 |
| 116089 | 92.4; 76.0; 77.9 | 96.4; 74.8; **75.4** | 92.8; 84.7; 84.9 |
| 406364 | 89.6; 73.5; 75.5 | 96.8; 73.1; **74.7** | 88.5; 79.9; 81.7 |
| 545353 | 92.9; 80.3; 83.4 | 97.4; 91.6; **92.5** | 93.6; 82.4; 84.4 |
| median | 92.9; 77.2; 78.8 | 97.4; 75.5; 76.5 | 92.8; 82.6; 84.4 |

With an empty past the figures are within two points of these. By place at the end: hundreds 98.3 to 99.5, units 100 in every cell; all of the loss is at the tens digit. On seeds 73129 and 93683 every reading equals the coordinating session's two files (read from its branch at 681f168).

**The second cell does not hold its level** (mine): on three seeds of five it ends at the first cell's level, twenty points under its own at 40,000 lines; on two it ends near 92, five to six points under. The control ends above the first cell on every seed (by 4.8, 2.5, 7.0, 6.2, 1.0).

**Predictions.** L1 refuted (the second cell's mean of the last three less its reading at 40,000 lines: -20.9, -6.3, -21.0, -22.1, -4.9; median -20.9; within 3 asked, refuted beyond 6). L2 failed (one of the first three seeds at 90 or more; two of all five). L3 failed (-2.3, +9.0, -2.5, -0.8, +9.1). L4 held (first cell 75.5 to 83.4). L5 failed (the control is 1.0 to 7.0 above the first cell). L6 held.

## 7. Outcome of addition 1 (`res/s400d_cls_seed*.json`, `report_2d.txt`; `res/s400_none_seed*.json`, `report_2none.txt`)

*(a) Readings on either side of the findings* (seeds 116089 and 406364; CPU 509 and 514 s; the further readings changed nothing: the readings shared with the first runs are the same). Never-shown additions after a read line:

| Seed | Lines | All kinds: first, second | Only the class kinds speaking: first, second | Only the plain kinds: first, second |
|---|---|---|---|---|
| 116089 | 63,000 | 90.8, 96.9 | 91.2, 90.2 | 74.3, 76.6 |
| | 65,000 | 84.6, 87.0 | 33.8, 61.2 | 73.7, 76.8 |
| | 255,000 | 82.8, 89.9 | 87.4, 85.4 | 26.2, 46.0 |
| | 257,000 | 79.0, 77.8 | 71.6, 67.6 | 27.9, 46.7 |
| | 400,000 | 76.0, 74.8 | 73.0, 72.8 | 7.3, 24.2 |
| 406364 | 63,000 | 86.6, 96.4 | 91.6, 92.0 | 72.2, 73.4 |
| | 65,000 | 84.7, 94.9 | 92.4, 93.1 | 72.2, 75.0 |
| | 255,000 | 82.9, 91.2 | 91.3, 94.5 | 30.4, 50.6 |
| | 257,000 | 78.8, 80.2 | 73.2, 75.6 | 30.7, 49.1 |
| | 400,000 | 73.5, 73.1 | 74.0, 75.5 | 20.6, 24.9 |

The fall is at the finding itself: across the finding at 256,000 lines the second cell loses 12.1 and 11.0 points within 2,000 lines, and nothing is regained in the 143,000 lines after. But **it is not the second cell's parts that fail to follow, as I supposed**: with only the class kinds speaking, the *first* cell falls as far (87.4 to 71.6; 91.3 to 73.2). The list of kept pairs shows what happens (`report_2d.txt`): at that finding, on both seeds, every one of the seven pairs of the kind of delays (3, 4, 6, 7) is dropped. That kind holds the two digits of both operands when the tens digit is named; the classes found in it are the carry and the sum modulo ten. On seed 116089 the same kind is dropped at 64,000 lines (class kinds alone: 91 to 34 for the first cell), is kept again at 128,000 lines, born with nothing, and is dropped again at 256,000. `merge.py` searches for classes only in the sixteen kinds of largest trust, and that line's report already names the fault ("a kind is forgotten when its classes succeed") and carries two repairs as options, neither in its frozen setting. So the step falls of the second cell are a fault of the base system's choice of where to search, which the second cell inherits; while the exact kind is kept, the second cell stays 6 to 10 points above the first (89.9 against 82.8, 91.2 against 82.9 at 255,000 lines).

Two more things in the table (mine). (1) The plain kinds alone fall from 79 to between 7 and 21 for the first cell over the long reading, with no finding involved: the trusts of the plain kinds drift to where the plain kinds alone answer almost nothing, and the first cell's answer comes to rest on the class kinds. The second cell's reading of the plain kinds falls less (46 to 51 at 255,000 lines). (2) With all kinds speaking the first cell is *below* the class kinds alone from 200,000 lines on (82.9 against 86.6 and 90.8): the plain kinds by then take away more than they give.

*(b) Without classes* (seeds 116089, 406364, 545353; CPU 344 to 376 s; 1.22 million cells): first cell 80.1, 78.6, 77.8 at 40,000 lines and 62.3, 61.8, 62.4 at the end (mean of the last three); second cell 86.3, 84.6, 85.6 and 68.1, 69.0, 69.1; control 83.8, 80.0, 82.5 and 66.7, 65.5, 63.5. With nothing found anew the second cell falls evenly with the first and keeps its margin over it (+5.8, +7.2, +6.7 at the end; +6.2, +6.0, +7.8 at 40,000 lines).

**So there are two losses with long reading, and the second cell cures neither** (mine): an even one, in which the first and the second cell fall together by 17 to 18 points without classes, and which is the base system's; and step falls where the base system's search lets go of the kind its best classes were found in. What the second cell adds (4 to 9 points) stands through the first and is lost at the second.

**Predictions.** The hypothesis of addition 1 is half right: the falls are at the findings (not refuted: 12.1 and 11.0 points across the finding), but the cause I gave (the second cell's parts cannot follow re-made cells) is wrong, since the first cell's reading of the class kinds falls as far. D1 held (-12.1). D2 held (-11.0). D3 held (-17.8 and -18.9). D4 held (+5.8, +7.2, +6.7). D5 failed, as I thought likely (-18, -16, -17 points from the level at 40,000 lines). D6 failed (seed 545353 has one fall of 5.8 points between readings; the other two seeds none over 4.1).

## 8. Addition 2 (20 November; written after section 7, before the runs it describes): the base line's own repair of where to search

*The evidence.* Section 7: the step falls happen when the kind (3, 4, 6, 7) is dropped from the sixteen kinds searched. `../merge/PLAN_6.md` (addition 2) gives a repair as an option of `merge.py`, `keep=2`, that line's **setting C**: the trust of a kind counts the trust of the kinds made from it, so a kind whose classes succeed keeps its place. That line ran setting C over 400,000 lines for its own answering cell on two seeds and found nothing gained (80.1 and 81.3 against 78.9 and 81.3 at the old score). It is not this line's mechanism and not the frozen setting; it is used here as that program has it, unchanged, as a test of the cause.

*Question.* With setting C, does the second cell escape the step falls?

*Hypothesis (mine).* It does: with `keep=2` the second cell's mean of the last three readings at 400,000 lines, after a read line, is at least 88 on at least two of three seeds (its level between 100,000 and 250,000 lines in the runs above was 87 to 94). *Refuted if* it ends under 82 on two seeds or more.

*Runs.* `gate2.py` with `keep=2`, everything else as before: seeds 116089, 406364, 545353 over 400,000 lines with the list of kept pairs at every finding; and the five reported seeds at 40,000 lines, to see what setting C does to the level of step 1. Cost: about seven CPU-minutes a long run, 80 s a short one.

| | Statement | Chance |
|---|---|---|
| E1 | the hypothesis: the second cell at least 88 at the end on at least two seeds of three | 0.45 |
| E2 | on at least two seeds of three, no finding after 32,000 lines drops every pair of the kind (3, 4, 6, 7) | 0.60 |
| E3 | the second cell ends at least 5 points above the first on all three seeds | 0.50 |
| E4 | at 40,000 lines with setting C the second cell's median over the five seeds is within 1 point of the frozen setting's 97.4 | 0.60 |
| E5 | the first cell ends under 85 on all three seeds (the even loss is untouched) | 0.75 |

*Not run.* `keep=1`; that line's kinds keyed on two classes (`merge6.py`); any change of my own to the search.
