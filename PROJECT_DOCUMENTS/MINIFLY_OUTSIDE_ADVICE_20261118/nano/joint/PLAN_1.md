# Line C, first plan (step C1): an edge decided by both ends against the one-sided edge (20 November 2026)

By a separate agent session, started by the outside session that keeps `ROADMAP_SECOND_REVISION_20261120.md`. Folder: `nano/joint/`; nothing outside it is changed. Written before any run of this step. Development level; repeated by no one; no checker gate, seal or review is claimed.

**Source.** Branch `claude/friendly-darwin-8mmhfm` at commit `fe397ca`. Copied from `loop_small_case/` at that commit, into the programs of this folder: from `cells2.py` the text, the split, the plain system of class `Sys` (logistic cells, every line read, one step of relay) and the loop and scoring of `run()`; `kern3.c` as the starting point of `kernj.c`; the build and call of `fastk.py` and `fastk3.py` in `fastj.py`; the manner of `check_fast.py` in `check_joint.py`.

**Authorisation.** As relayed to this session by the coordinating session: on 20 November Mr. Xu asked it to "revise the road map and keep doing some experiments. Spin off some agents to speed up the experiment process." I have no word from him directly. His words on connections, from the roadmap file, section 1: "The connection strength (ideally can involve many cells) should be jointly decided by the cells together. For example if one cell wants to drop out, then the connection strength to it should be zero no matter how willing other cells are. Obviously we can simplify the edge to only involve two cells, and perhaps refine even further to one sided edges, with inhibitory / excite signals respectively. But whether this is justified has to be answered by experiment."

## 1. The question

For the 3,584 connections from the 256 cells in between to the 14 branches of the answering cell: does a connection whose strength both ends decide behave differently, on two-digit sums never shown, from the present connection, whose strength the answering cell alone holds and changes? (The connections into the cells in between come from input lines, which are not cells, and are left as they are.)

## 2. The forms

Common to all: the testbed and plain system of `loop_small_case/REPORT.md`, section 2 and section 10 (text `aa+bb=ccc`, 14 byte values, ten delay lines, 140 lines, 256 logistic cells, eta1 1/4, eta2 1/16, fading 1e-5 a byte, return strengths R held by each cell in between, 400,000 lines, one stream). h_j is the output of cell j; m_v the mismatch of branch v; g = eta2 / (h.h + 1) the answering cell's step. The scaled mismatch g m_v is what the plain system already makes present at connection (v, j): the cell's return strength R[j, v] changes by g m_v h_j there. In every form R changes and fades exactly as in the plain system (by g m_v h_j, fading 1e-5), and the cells in between learn exactly as in the plain system. Only the forward strength w[v, j] of the connection differs.

- **F1, one-sided (the plain system).** w = A, held by the answering cell; born 0; A += g m_v h_j; A fades by 1e-5 a byte.
- **F2, a product of two parts.** w = A B. A is held by the answering cell (any sign, born 0, fades by 1e-5 a byte as in F1). B is held by the sending cell, one for each connection it makes; born 1; never below zero (set to zero when a change would take it below); it does not fade. Either part at zero makes the connection zero. Both parts change at the same moment from the values before the change: A += g m_v h_j B; B += etaB g m_v h_j A. Each end changes only its own part and reads, at the connection, the scaled mismatch, the sender's output and the other end's part (for the sending cell this is the strength divided by its own part; at birth A = 0 the answering cell cannot recover B from the strength, so I let each part be readable at the connection by the other end, which cannot change it). etaB is the sending end's step relative to the answering end's: the coordinator's proposal is etaB = 1; with etaB = 0 the form is F1 exactly, which is used as a check of the program.
- **F3, the smaller of the two parts.** w = sign(A) min(|A|, B), with A and B held, born (A = 0, B = b0) and bounded as in F2; A fades, B does not. Only the end whose part is the smaller changes: where |A| < B, A += g m_v h_j; otherwise B += g m_v h_j sign(A), not below zero. Either part at zero makes the connection zero. With b0 so large that no |A| reaches it, F3 is F1 exactly.

Two changes of mine to the coordinator's proposal, with reasons: (a) a relative step etaB for the sending end in F2, because the product makes the size of a change of the strength grow with A squared plus B squared and I do not know beforehand that etaB = 1 is stable; (b) a birth value b0 for B in F3, because with b0 = 1 the smaller part may be A at every connection for the whole run, and the arm would then be F1 under another name.

A consequence I state beforehand: in F1 the return strength and the forward strength of a connection change by the same product and come to agree. In F2 and F3 the forward strength changes by a different amount from R, so they need not agree. I report their cosine at the end. This belongs to the form (R is the cell's own and is left as it is); it is not a second change, but it is a second route by which a difference of score could arise, and I will say so if a difference appears.

**Numbers that learn.** F1: 35,840 strengths from lines, 256 thresholds, 3,584 strengths of the answering cell, 14 constants, 3,584 return strengths: 43,278. F2 and F3: 3,584 more (the parts B): 46,862. The arms are not equal in numbers that learn; the number of connections is equal.

## 3. Measures

1. Whole answer right on 500 never-shown pairs, mean of the last three readings at 50,000-line intervals (the score of `cells2.py`).
2. At the end (400,000 lines): the share of the 3,584 connections whose |w| is below one hundredth of the largest |w|.
3. The score at the end after those connections are set to zero, beside the score at the end with nothing removed (both at the last reading, not the mean of three).
4. Additions of mine, no prediction except P7: the score at the end after the weakest 25, 50, 75 and 90% of connections by |w| are removed; the cosine of R with w; in F3 the share of connections at which the sending end's part is the smaller.

## 4. Hypothesis and what would refute it

**H: on this testbed the one-sided edge is a simplification that costs nothing in the score: F2 at its chosen setting is within 4 points of F1 on never-shown sums.** Refuted if the paired difference F2 less F1 (same seed) has a median larger than 4 points in size and the same sign on at least four of the five reported seeds. (The plain system's five seeds span 81.8 to 86.7, so 4 points is about the spread.) If H is refuted upward, the both-ends edge is the better form here; if downward, the one-sided edge is not merely a simplification but the better form here. If H stands, measures 2 to 4 say whether the two forms differ in what they leave of the connections.

## 5. Predictions (mine), with chances

- P1 (0.9): my program's F1 gives 81.5 on seed 1 and the report's 84.2, 82.7, 86.4, 81.8, 86.7 on the five seeds, to the digit printed.
- P2 (0.6): H stands (F2 within 4 points of F1 by the rule above).
- P3 (0.25): F2's median is more than 2 points above F1's.
- P4 (0.7): the share of connections below one hundredth of the largest is higher in F2 than in F1 on at least four of five seeds.
- P5 (0.8): removing those connections changes the score at the end by less than 2 points at the median, in every arm.
- P6 (0.6): F3 at its chosen setting is within 4 points of F1 at the median.
- P7 (0.5): removing the weakest half of the connections costs F2 fewer points than F1 on at least four of five seeds.

## 6. Order of work and settings

1. `joint.py` (numpy, the reference), then `kernj.c` with `fastj.py`, then `check_joint.py`: the two on the same seed and 300 lines for every form and setting in use, every strength compared; and etaB = 0 and a very large b0 against F1.
2. Development seed 1, compiled, 400,000 lines. F1 once (must give 81.5). F2 with etaB in {1, 1/4, 1/16}. F3 with b0 in {1, 1/2, 1/4, 1/8}. Choice: the highest score (measure 1); within one point, the value nearest the coordinator's proposal (etaB = 1; b0 = 1). For F3, a value of b0 at which the sending end's part is the smaller at no connection at the end is not eligible. Nothing else is tuned. If F2 diverges at every etaB, that is the result and it is reported; no further grid without an entry here first.
3. The five reported seeds 73129, 93683, 116089, 406364, 545353 for F1, F2 and F3 at the chosen settings.
4. One full-length run of each form on seed 1 with the numpy program, to compare its score with the compiled one.

## 7. Cost

A compiled run: about 16 to 30 seconds of one core. Development 8 runs, reported 15 runs, in all under 15 minutes on four cores. Numpy full-length runs: three, about 4 minutes each, in parallel. Checks: under two minutes. No checkpoints are kept; each run writes one small JSON file in `res/`.

## 8. Not run

Connections of more than two cells (C3); the sign of a sender (C4); other rules for A in F1; fading of B; any change to R's rule; other sizes, lengths or texts; the comparator by back-propagation; the withdrawal of a connection for good (C2, the next plan). F3 is run only if F2 is done.

---

## 9. Entered 20 November after the check and the eight development runs, before any further run

**The check** (`check_joint_output.txt`): the compiled kernel agrees with the numpy program to 1e-14 on every form and setting; form 2 with etaB = 0 and form 3 with b0 = 1e9 are form 1; form 1 agrees with `cells2.py`'s plain system to 3e-15 on 300 lines.

**Development seed 1, 400,000 lines, never-shown sums (mean of the last three readings).** F1: 81.5 (the value to match). F2: etaB = 1: 80.5; 1/4: 81.4; 1/16: 81.5. F3 with b0 = 1, 1/2, 1/4, 1/8: 81.5 each, and identical to F1 in every figure.

**What I had not known when I wrote sections 2 and 6: the strengths of the answering cell are small.** At the end of the plain run the largest |A| is 0.111 and the mean |A| is 0.0156. Two consequences. (i) In F2 the sending end's part, born at 1, hardly moves: its change is proportional to A, and B ends between 0.97 and 1.34 at etaB = 1 (mean 1.01). The proposed F2 is therefore close to F1 by its births, whatever the merit of the form. (ii) In F3 no |A| reaches any of the four values of b0, so none is eligible by the rule of section 6 and the arm as planned is F1 under another name, which is the case section 2 (b) foresaw without knowing the scale.

**Choice for F2 by the rule of section 6:** the three scores lie within one point, so etaB = 1, the coordinator's proposal. It goes to the five seeds as planned.

**An error in my runs, without effect on the figures:** the eight runs were started with the label missing from the command, so that `seed=1` was read as the label, the seed fell back to its default (which is 1) and the eight result files overwrote each other. The printed lines are kept in `log/`; the file was removed and F1 rerun with its label (`res/j_d_f1.json`, 81.5).

**Extension, entered here before its runs** (additions; the registered predictions P1 to P7 are untouched and are scored on the arms they were made for).

- *F2 with a sending end that moves* (arm F2b). Without the bound and the fading, etaB A^2 - B^2 keeps its value at a connection, so the two parts are of comparable weight when etaB is near 1 / A^2, some hundreds to some thousands here. Grid on seed 1: etaB in {16, 64, 256, 1024}. Choice: the highest score among the settings that stay finite; within one point, the largest etaB (the one in which the sending end decides most). If none stays finite or all fall below 40, that is reported and the arm goes to the five seeds at the best of them all the same, so that its size is known.
- *F3 at births where the smaller part can be the sender's.* Grid on seed 1: b0 in {1/16, 1/32, 1/64, 1/128}. The rule of section 6 is applied as written (highest score; within one point the largest b0; a value at which the sender's part is the smaller at no connection is not eligible): this is arm F3. Because that rule may pick a value at which the sender's part is the smaller at few connections, a second arm F3b at b0 = 1/64, the mean |A| of the plain run, goes to the five seeds as well, whatever its score.
- What I expect of F3 where the sender's part is the smaller (my reading of the rule, before the runs): such a connection changes as in F1 while the mismatch pushes its strength away from zero, but it cannot cross zero: B stops at zero, A keeps its sign and only fades, and the connection stays at zero until the mismatch pushes it back towards the sign it had. The sending end's part acts as a bound on the sign of that connection.
- Predictions for the additions: P8 (0.5) F2b is within 4 points of F1 at the median of the paired differences. P9 (0.6) F2b has a higher share of connections below one hundredth of the largest than F1 on at least four of five seeds. P10 (0.4) F3b is more than 4 points below F1 at the median of the paired differences. P11 (0.7) the cosine of R with w is below 0.9 in F2b at the median.
- Reported seeds, then: F1, F2 (etaB = 1), F2b, F3, F3b: 25 runs, about 3 minutes on four cores. Numpy full-length runs on seed 1: F1, F2b and F3b (the forms in which B moves), in place of the three of section 6.

## 10. Entered 20 November after the second development grid and the 25 runs on the reported seeds, before the two further sets of runs named here

**Second development grid, seed 1** (`res/j_d_*.json`). F2 with etaB = 16, 64, 256, 1024: 77.7, 78.8, 80.1, 75.5, all finite; share of connections below one hundredth of the largest 20, 30, 48, 55% (F1: 11%); cosine of R with w 0.85, 0.64, 0.42, 0.30. By the rule of section 9, F2b is etaB = 256. F3 with b0 = 1/16, 1/32, 1/64, 1/128: 81.5, 79.5, 80.2, 82.1; the sender's part is the smaller at the end at 0.03, 0.3, 1.1 and 1.5% of connections (it grows where it is the smaller). By the rule of section 6, F3 is b0 = 1/16 (within one point of the highest, the largest b0); F3b is b0 = 1/64 as entered.

**Reported seeds** (full tables in `report1.txt`; scored in section 11). Never-shown sums, medians: F1 84.2; F2 84.0; F2b 67.3; F3 84.2; F3b 85.3. F2b is below F1 on every seed, by 6 to 27 points, where the development seed had it 1.4 below: the development seed misled the choice of etaB = 256 and I say so.

**Two additions, entered before their runs**, to say what the loss of F2b is, since section 2 named two routes and the 25 runs cannot tell them apart:

- *D1: F2b with the return strength following the forward strength* (`form=4`). As form 2 with etaB = 256, except that R[j, v] changes by g m_v h_j (B^2 + etaB A^2), which is the change of w to first order, in place of g m_v h_j; the cell reads both parts at the connection as before. This is a change of R's rule and is run as a diagnostic only, not as an arm of the question: if the score returns to F1's level, the loss of F2b was the disagreement of R with w; if it stays low, the loss lies in how the product changes the forward strengths. Five reported seeds.
- *D2: the size of the loss against etaB.* F2 with etaB = 16 and 64 on the five reported seeds (seed 1 gave 77.7 and 78.8).
- Predictions: P12 (0.5) D1 is still more than 4 points below F1 at the median of the paired differences. P13 (0.7) the medians fall in the order etaB = 1, 16, 64, 256.
- Cost: 15 compiled runs, about 2 minutes; the check is extended to form 4 and to runs of 3,000 lines in which B moves.

## 11. Outcome (entered 20 November, after all runs of this plan)

Tables: `report1.txt` (made by `report1.py` from `res/j_*.json`); the check: `check_joint_output.txt` (passed, 1e-14 to 3e-13, form 4 and runs of 3,000 lines included). The numpy program at full length on seed 1 gives the compiled program's readings to the digit for F1 (81.5), F2b (80.1) and F3b (80.2).

Never-shown sums, whole answer right, %, mean of the last three readings; the five reported seeds and the median; the last column is the median of the paired differences from F1 and the number of seeds below F1.

| | 73129 | 93683 | 116089 | 406364 | 545353 | median | against F1 |
|---|---|---|---|---|---|---|---|
| F1, one-sided (the plain system) | 84.2 | 82.7 | 86.4 | 81.8 | 86.7 | **84.2** | |
| F2, product, etaB = 1 (as proposed) | 84.0 | 81.8 | 86.5 | 80.6 | 86.2 | **84.0** | -0.5; 4 below |
| product, etaB = 16 (addition D2) | 83.9 | 78.4 | 83.5 | 75.7 | 84.9 | **83.5** | -2.9; 5 below |
| product, etaB = 64 (addition D2) | 82.1 | 79.7 | 73.3 | 73.1 | 77.9 | **77.9** | -8.7; 5 below |
| F2b, product, etaB = 256 | 77.9 | 75.9 | 59.4 | 62.6 | 67.3 | **67.3** | -19.2; 5 below |
| etaB = 256 with R following w (diagnostic D1) | 76.4 | 77.6 | 60.4 | 67.1 | 29.9 | **67.1** | -14.7; 5 below |
| F3, smaller part, b0 = 1/16 | 84.2 | 82.5 | 86.9 | 82.1 | 86.6 | **84.2** | 0.0; 2 below |
| F3b, smaller part, b0 = 1/64 | 86.5 | 83.3 | 88.8 | 85.3 | 83.7 | **85.3** | +2.3; 1 below |

What is left of the connections at the end (medians of five seeds):

| | below one hundredth of the largest, % | score at the end | with those removed | weakest half removed | cosine of R with w | B, smallest to largest |
|---|---|---|---|---|---|---|
| F1 | 11.2 | 84.8 | 85.0 | 84.8 | 1.000 | 1 |
| F2, etaB = 1 | 12.6 | 83.0 | 83.2 | 84.6 | 0.995 | 0.97 to 1.29 |
| etaB = 16 | 24.9 | | | | 0.835 | |
| etaB = 64 | 51.0 | | | | 0.593 | |
| F2b, etaB = 256 | 57.5 | 65.8 | 64.4 | 64.8 | 0.379 | 0.000 to 9.0 |
| F3, b0 = 1/16 | 11.2 | 84.4 | 84.6 | 84.4 | 1.000 | 0.058 to 0.108 |
| F3b, b0 = 1/64 | 10.5 | 84.8 | 84.6 | 83.6 | 1.000 | 0.001 to 0.109 |

**The hypothesis.** H stands for F2 as proposed: the paired differences are -0.2, -0.9, +0.1, -1.2, -0.5 (median -0.5), inside the 4 points. But I do not read that as support for the one-sided edge being a harmless simplification of this joint form, because the arm hardly tests it: with B born at 1 and strengths of size 0.02 to 0.1, the sending end's part stays between 0.97 and 1.3, and F2 is F1 to within that. Where the sending end's part is given weight, the score falls with that weight on every seed: -2.9, -8.7 and -19.2 points at etaB = 16, 64 and 256. So, on this testbed and for this rule (a product, each part changed by the mismatch times the sender's output times the other part), the answer to the question of section 1 is, in my reading: the one-sided edge is not a simplification that loses something; the edge decided by both ends is the worse of the two wherever it differs from the one-sided edge, and equal to it where it does not.

**Which of the two routes of section 2.** The diagnostic D1 brings the cosine of R with w back to 0.998 and the score stays at 67.1 (five seeds below F1, one of them at 29.9). The loss is therefore not the disagreement between return and forward strengths; it lies in how the product changes the forward strengths. What in that I have not tested. What I see: the largest strength is 0.73 at the median in F2b against 0.09 in F1, and 58% of the connections end below one hundredth of the largest against 11%; a strength changes in proportion to B^2 + etaB A^2, so strong connections change fast and weak ones slowly, and the answering cell's step g, which is set from the outputs alone, no longer bounds the change of a branch. Whether the loss is the concentration on few connections or steps that are too large is open.

**F3.** At the birth value chosen by the registered rule (1/16) the sender's part is the smaller at 0.03% of connections at the end, and the arm is F1 for practical purposes; it says nothing. At 1/64 the sender's part is the smaller at 1.2% of connections at the end (it grows where it is the smaller, so it does not stay the smaller), and the score is +2.3 at the median, above F1 on four seeds of five (+2.3, +0.6, +2.4, +3.5, -2.9). That is inside the spread of the seeds and of the readings and I claim no difference; it is the one place where a joint form was not below the one-sided one, and five more seeds would say whether it is anything.

**What is left of the connections.** In the plain system 11% of the connections are below one hundredth of the largest and removing them changes nothing; removing the weakest half at the end costs nothing either (84.8 and 84.8 at the median), the weakest three quarters cost about 7 points, the weakest nine tenths nearly everything (20.8). The product form leaves far more connections near zero (58% at etaB = 256) but at a lower score, so it is not a cheaper network of equal worth.

**Predictions.** P1 held (81.5 on seed 1; the five figures to the digit). P2 held (-0.5). P3 failed (F2's median is 0.2 below F1's). P4 held on five seeds of five, by 0.6 to 1.5 percentage points, which is small. P5 held (median change 0.0 in F1, F2 and F3; -0.2 in F2b, where two seeds lose 2.2 and 2.8). P6 held, emptily, the arm being F1. P7 failed (F2 loses fewer points than F1 on three seeds, the same on one, more on one). P8 failed (-19.2). P9 held (53 to 69% against 11 to 13%). P10 failed, with the sign against me (+2.3). P11 held (0.38). P12 held (-14.7). P13 held (84.0, 83.5, 77.9, 67.3).

**Errors and departures of mine.** (1) The development seed misled the choice of etaB = 256 (1.4 below F1 on seed 1, 19.2 below at the median of the reported seeds); D2 was added to see the whole course. (2) The first eight development runs were started without their label (section 9). (3) F3's registered rule picked a setting that is F1; I had foreseen the case in section 2 but not that the rule of section 6 would select it. (4) The grids of sections 9 and 10 are additions after the first runs; each was entered and committed before its runs. (5) My first commit on this branch carries a co-author line naming a model, against the rule given to me; later commits do not, and I have not rewritten the branch.

**Not run:** what in the product's change causes the loss (a step for the two parts that keeps the change of a branch bounded; parts born at equal size); F3b on more seeds; fading of B; everything listed in section 8.

## 12. A note on measure 2 (entered 20 November, found while checking the programs of `PLAN_5.md`)

The branch for the minus sign, a byte value that never occurs in this text, has all 256 of its strengths at exactly zero from birth to the end in every form (its output is born at zero and its mismatch is zero at every byte). Those 256 connections are 7.1% of the 3,584 and are counted in every "share below one hundredth of the largest" of section 11: of F1's 11.2%, 7.1 points are that branch and about 4 points are connections that learned and stayed small. The differences between the forms are untouched; the statement that removing those connections changes nothing is, for 7.1 points of them, trivially true.

## 13. Two dated additions, asked by the coordinating session, entered 20 November before their runs

(The coordinating session's message asking for them was sent before my reports of C5 and C4 and reached me after them.)

**D3: the product with a bounded step** (`form=5`). As F2b (w = A B, etaB = 256), except that both parts of the connections of branch v change with the step eta2 / (sum over j of h_j^2 (B^2 + etaB A^2) + 1) in place of eta2 / (h.h + 1). With this step the change of a branch's output at a byte is, to first order, the same share of its mismatch as in F1, whatever the sizes of the parts. The return strengths and the constants change as in F2b. The quantity in the step is summed over the branch's connections, so it is the answering cell's to form if it can read each sending part at its connection, as section 2 already lets it; the sending end receives the scaled mismatch as before. If the score returns to F1's level, the loss of F2b was steps that were too large; if it stays low, the loss is in which connections the product lets grow. The check is extended to this form (`check_joint_output.txt`: passed; with etaB = 0 it is form 1). Five reported seeds.
- P14 (0.4): D3 is within 4 points of F1 at the median of the paired differences.

**Ten further seeds**, fixed by the coordinating session for every line: 1001 to 1010. Its rule: where two arms are within five points and the difference matters for the reading, run the ten as well and report the fifteen together (median, and how many seeds of fifteen go each way), keeping the five reported seeds as the first table. In this plan that is F3b against F1 (+2.3 on five seeds, four above), and D3 against F1 or against F2b if within five points of either. F1 on the ten seeds is run once, with `joint.py`, and serves as the plain system for the additions to `PLAN_3.md`, `PLAN_4.md` and `PLAN_5.md` too (each of their programs reproduces it, by its check and on the five seeds).
- P15 (0.35): F3b is above F1 on at least eleven seeds of fifteen.

## 14. Outcome of the additions of section 13 (entered 20 November)

Tables: `report_add.txt` (made by `report_add.py`). The plain system on the ten further seeds: 78.0, 83.3, 84.2, 85.3, 77.0, 85.5, 83.7, 88.8, 79.2, 81.4 (median 83.5; of the fifteen 83.7).

**D3, the product with a bounded step.** Five reported seeds: 59.7, 67.7, 79.8, 52.9, 59.2 (median 59.7); paired against F1 -24.5, -15.0, -6.6, -28.9, -27.5 (median -24.5); against F2b -18.2, -8.3, +20.4, -9.7, -8.1 (median -8.3). The difference from F1 is far from five points, so the ten further seeds were not required by the rule; they were queued with the rest before any was read, and I report them: on the fifteen seeds D3 is -15.0 at the median, below F1 on fourteen and above on one (by 1.2). **Bounding the step does not bring the score back; on the five seeds it is lower than with the plain step.** So the loss of the product with weight at the sending end is not steps that are too large. What remains is which connections the product lets grow: 53 to 73% of the connections are below one hundredth of the largest in D3, as in F2b. One reservation: in D3 the cosine of the return strengths with the forward strengths is 0.09 to 0.38 (0.38 in F2b), and the diagnostic D1, which excluded that route, was made on F2b with the plain step and not on D3; I have not run the two together.

**F3b against F1 on fifteen seeds.** Paired differences: +2.3, +0.6, +2.4, +3.5, -2.9 | +3.1, +1.3, +0.3, +0.9, -3.3, +0.3, +2.0, -0.7, +0.9, +1.0. Median +0.9 (of the ten alone +0.9); above F1 on twelve seeds of fifteen, below on three. The smaller-part edge with the sending part born at 1/64 is, on these seeds, a little above the one-sided edge: about one point, with a sign that holds on twelve of fifteen. The +2.3 of the five reported seeds overstated its size. I would call it a small difference in favour of this joint form and not more; the mechanism (section 9: a connection limited by its sending part cannot cross zero) is not tested.

**Predictions.** P14 failed (-24.5). P15 held (twelve of fifteen).
