# Line `nano/gate2`, seventh plan: the steps of this line again on `nano/merge`'s setting C (20 November 2026)

**Authorisation**, as quoted to this session by the coordinating session (session_017dSEpVed7SthEH3ngGAdNc): Mr. Xu (in Chinese): "我们接受bitter lesson, 因此你可以尝试，但避免spot fix, ad hoc tricks等等在其他的任务上无法迁移的手段。多尝试新想法。" and, most recently: "继续大胆尝试。" (the coordinating session's translation: "We accept the bitter lesson, so you may try things, but avoid spot fixes, ad hoc tricks and the like: means that do not transfer to other tasks. Try more new ideas." and "Go on trying boldly."). **Source:** `claude/friendly-darwin-8mmhfm` at a859b2f224f0adca576616c0fc96bf82699ec293; this branch at the commit that carries this file. Development level; no gate, seal or review is claimed. Written after the outcomes of `PLAN_2.md` (section 9), `PLAN_5.md` and the runs of `PLAN_6.md` had been read, and before any run it describes.

## 1. The evidence, and the question

`PLAN_2.md`, sections 7 and 9: the second cell's large falls over a long reading are the base's search letting go of the kind (3, 4, 6, 7); with `merge.py`'s own option `keep=2` (that line's setting C) the second cell ends at 93.0, 90.0, 95.1 after 400,000 lines on three seeds. The long life of two tasks (`PLAN_3.md`, section 6) was run in the frozen setting only, on seeds 73129 and 93683, and on seed 73129 the second cell had been brought down by that fault before the task changed and then lost addition for good (18% kept). *Question:* with setting C as the base (no mechanism of mine; that program's option as it stands), what do this line's steps give: the long reading on the two seeds not yet run, the long and the short life of two tasks, and three-digit sums?

## 2. Runs and measures

`gate2.py` with `keep=2`; first cell, second cell and control from one run; scores as in the plans of each step. (a) 400,000 lines, seeds 73129 and 93683 (with section 9 of `PLAN_2.md` this makes five). (b) Addition 400,000 lines, then subtraction 200,000, seeds 73129 and 93683. (c) Addition 40,000 lines, then subtraction 20,000, five reported seeds. (d) Three-digit sums, 80,000 lines, five reported seeds. Cost: about 75 CPU-minutes in all; three at a time.

## 3. Hypothesis, what refutes it, predictions (mine, before the runs)

*Hypothesis.* The lasting loss of addition by the second cell on seed 73129 in the long life was a product of the state the search fault had left it in: with setting C it keeps at least 50% of never-shown addition on that seed ("kept" as `PLAN_3.md` defines it). *Refuted if* it keeps under 30% there.

| | Statement | Chance |
|---|---|---|
| S1 | long reading, seeds 73129 and 93683: the second cell at 88 or more at the end on both (mean of the last three, after a read line) | 0.60 |
| S2 | long reading, all five seeds with setting C: the second cell's median at 90 or more | 0.65 |
| S3 | long life: the hypothesis (at least 50% kept on seed 73129) | 0.55 |
| S4 | long life: the second cell keeps at least 75% on both seeds (the coordinating session's hypothesis is not refuted there) | 0.35 |
| S5 | short life, five seeds: the second cell keeps at least 85% at the median (frozen setting: 88.8) | 0.60 |
| S6 | short life: the first cell keeps more than in the frozen setting (81.1 at the median) | 0.50 |
| S7 | three-digit sums: the second cell's median at least 1.5 points above the first cell's (mean of the last three) | 0.70 |

## 4. Not run

Setting C without the second cell (that line has it); `keep=1`; anything on prose; the cell of `PLAN_5.md`.
