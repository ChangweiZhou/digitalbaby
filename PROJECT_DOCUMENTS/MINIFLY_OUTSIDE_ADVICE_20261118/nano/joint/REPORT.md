# Line C (`nano/joint`): connections decided by both ends. Report of 20 November 2026

By a separate agent session started by the outside session; the frame is `../../ROADMAP_SECOND_REVISION_20261120.md`. Development level; repeated by no one; no checker gate, seal or review is claimed. Readings are mine. Source: `claude/friendly-darwin-8mmhfm` at `fe397ca`; what was copied from `loop_small_case/` is named at the head of each program and in `PLAN_1.md`.

## Step C1: an edge decided by both ends against the one-sided edge

Plan, grids, outcome and scored predictions: `PLAN_1.md` (sections 1 to 8 before any run; 9 and 10 entered before the runs they name; 11 the outcome). Programs: `joint.py` (numpy, the reference), `kernj.c` with `fastj.py` (compiled), `check_joint.py` (`check_joint_output.txt`: passed). Tables: `report1.txt`. Result files: `res/j_*.json`; printed lines: `log/`.

Testbed: two-digit sums, 2,000 of the 10,000 pairs never shown, whole answer right on 500 of them, mean of the last three readings of 400,000 lines; 256 cells in between; the forms apply to the 3,584 connections from those cells to the answering cell's branches.

| Form | five seeds | median | against the one-sided edge (paired) |
|---|---|---|---|
| F1 one-sided: w = A, held by the answering cell (reproduces `cells2.py`) | 84.2, 82.7, 86.4, 81.8, 86.7 | 84.2 | |
| F2 product: w = A B, B held by the sending cell, born 1, never below 0; equal steps | 84.0, 81.8, 86.5, 80.6, 86.2 | 84.0 | -0.5 |
| the same, the sending end's step 16 times larger | 83.9, 78.4, 83.5, 75.7, 84.9 | 83.5 | -2.9 |
| 64 times | 82.1, 79.7, 73.3, 73.1, 77.9 | 77.9 | -8.7 |
| 256 times | 77.9, 75.9, 59.4, 62.6, 67.3 | 67.3 | -19.2 |
| 256 times, the return strength made to follow the forward strength (a diagnostic) | 76.4, 77.6, 60.4, 67.1, 29.9 | 67.1 | -14.7 |
| F3 smaller part: w = sign(A) min(|A|, B), B born 1/16 | 84.2, 82.5, 86.9, 82.1, 86.6 | 84.2 | 0.0 |
| the same, B born 1/64 | 86.5, 83.3, 88.8, 85.3, 83.7 | 85.3 | +2.3 |

1. **As proposed (B born at 1, equal steps), the product edge scores as the one-sided edge does, because it is the one-sided edge to within a few per cent**: the strengths here are of size 0.02 to 0.1, the sending end's part changes in proportion to them and stays between 0.97 and 1.3.
2. **Where the sending end's part has weight, the product edge is worse, on every seed, and the more so the more weight it has** (-2.9, -8.7, -19.2 points). The loss is not the return strengths ceasing to agree with the forward ones (the diagnostic restores the agreement and not the score); it lies in how the product changes the forward strengths. What in that is not tested.
3. **The smaller-part edge is level with the one-sided edge** (0.0 and +2.3; the second is inside the spread of the seeds and I claim nothing from it). The sender's part is the smaller at about 1% of connections at the end at most, because it grows where it is the smaller.
4. **Connections near zero.** One-sided: 11% of the connections are below one hundredth of the largest; removing them, or the weakest half, costs nothing; the weakest three quarters cost about 7 points. The product with weight at the sending end leaves 58% below one hundredth, at a score 19 points lower.
5. My answer to the step's question, for these two joint rules on this testbed: the one-sided edge loses nothing against an edge that both ends decide; the product rule loses against it wherever the two differ. Other joint rules are not covered.

