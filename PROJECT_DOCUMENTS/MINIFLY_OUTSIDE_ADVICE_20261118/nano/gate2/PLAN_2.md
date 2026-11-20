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
