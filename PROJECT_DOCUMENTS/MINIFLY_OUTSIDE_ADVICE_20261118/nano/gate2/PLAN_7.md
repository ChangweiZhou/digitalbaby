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

## 5. Outcome (20 November; `res/s400_C_seed*.json`, `res/lifelong_C_seed*.json`, `res/life_C_seed*.json`, `res/s3_C_seed*.json`; `report_2C.txt`, `report_7long.txt`, `report_7life.txt`, `report_7s3.txt`)

`gate2.py` on `merge.py`'s setting C (`keep=2`), nothing else changed. CPU: 464 s for 400,000 lines; about 700 s for the long life; about 130 s for the short life; 218 to 241 s for three-digit sums.

*(a) Long reading, now five seeds* (never-shown additions after a read line; the reading at 40,000 lines, then the mean of the last three readings at 400,000 lines):

| Seed | First cell | Second cell |
|---|---|---|
| 73129 | 91.2, 80.2 | 96.9, **91.0** |
| 93683 | 94.3, 83.1 | 98.2, **94.5** |
| 116089 | 92.0, 82.8 | 96.2, **93.0** |
| 406364 | 89.0, 78.3 | 97.0, **90.0** |
| 545353 | 91.8, 83.4 | 96.6, **95.1** |
| median | 91.8, 82.8 | 96.9, **93.0** |

On five seeds of five the second cell ends at 90 or more, 9.9 to 11.7 points above the first cell and 1.5 to 7.0 under its own level at 40,000 lines (frozen setting: 76.5, 91.9, 75.4, 74.7, 92.5).

*(b) The long life* (addition 400,000 lines, then subtraction 200,000; seeds 73129 and 93683). Addition before, at the end, kept; clean subtractions at the end:

| Seed | First cell | Second cell | Control |
|---|---|---|---|
| 73129 | 80.2, 72.7, 90.7%; 69.2 | 91.0, 59.5, **65.4%**; 78.2 | 84.0, 78.6, 93.5%; 64.4 |
| 93683 | 83.1, 72.5, 87.2%; 60.2 | 94.5, 76.4, **80.9%**; 75.3 | 86.0, 78.9, 91.8%; 59.0 |

Seed 73129, never-shown additions through the second task (after a subtraction line; 1,000, 5,000, 20,000, 40,000, 100,000, 200,000 lines): first cell 4.5, 10.2, 20.2, 77.3, 74.7, 70.3; second cell 23.2, 34.6, 46.6, 70.5, 56.7, 56.3. So with setting C the second cell does come back (to 70 at 40,000 lines, where in the frozen setting it stayed at 12 to 24 to the end), and then slips again while the first cell stays near 72 to 77.

*(c) The short life* (40,000 and 20,000 lines, five seeds, medians and the five values): the first cell keeps 83.3% (85.6, 85.2, 71.2, 83.3, 81.7), the second cell 88.8% (74.6, 92.1, 88.8, 91.5, 87.1), the control 71.2%; lowest reading after the change 42.0, 61.2, 15.8; clean subtractions at the end 81.7, 87.1, 83.5. (Frozen setting: 81.1, 88.8, 65.8; 50.2, 63.2, 18.6; 80.4, 89.4, 83.1.)

*(d) Three-digit sums* (80,000 lines, five seeds, mean of the last three): first cell 94.3 (94.9, 93.3, 94.3, 93.9, 94.9), second cell 96.7 (96.7, 96.7, 95.6, 96.1, 97.5), control 94.3; the second cell above the first on five seeds of five (+1.8, +3.5, +1.3, +2.1, +2.5). (Frozen setting: 94.1, 96.9, 93.5.)

**What I take from it (mine).**

1. *Over a long reading, on setting C, the second cell holds 90 to 95 on five seeds of five* (median 93.0 against 82.8 for the first cell). This is the answer to step 2 that I would carry forward: the second cell does not hold in the frozen setting because of the base's search fault; with that line's own repair it loses 2 to 7 points in 360,000 lines where the first cell loses 9 to 12.
2. *The hypothesis of this plan holds*: the lasting loss on seed 73129 was a product of the fallen state (65.4% kept with setting C against 18.2%). But *the second cell does not keep addition well through a long second task either way*: 65.4% and 80.9%, where the first cell keeps 90.7% and 87.2%; at the end of the long life it is below the first cell on one seed (59.5 against 72.7) and little above on the other (76.4 against 72.5), having been 11 points above before the change. The coordinating session's hypothesis (at least 90% kept; refuted under 75%) is refuted on one seed of two in the long life in both settings.
3. *Over the short life setting C changes nothing for the second cell* (88.8% in both) and little for the first (83.3% against 81.1%).
4. *On three-digit sums the gain stands on setting C* (+2.4 at the median).
5. Throughout, the second cell learns the second task better than the first (clean subtractions 78.2 and 75.3 against 69.2 and 60.2 after the long life; 87.1 against 81.7 after the short one).

**Predictions.** S1 held (91.0, 94.5). S2 held (93.0). S3 held (65.4%). S4 failed (65.4% on seed 73129). S5 held (88.8). S6 held, barely (83.3 against 81.1). S7 held (+2.4). Six of seven.
