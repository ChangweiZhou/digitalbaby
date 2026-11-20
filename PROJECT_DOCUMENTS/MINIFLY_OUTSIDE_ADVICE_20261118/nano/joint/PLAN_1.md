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
