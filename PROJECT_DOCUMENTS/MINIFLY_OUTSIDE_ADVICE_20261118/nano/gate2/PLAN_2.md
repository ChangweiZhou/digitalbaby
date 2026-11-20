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
