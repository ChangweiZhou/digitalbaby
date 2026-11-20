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

## Step C2: the right to withdraw from a connection

Plan, development grid, outcome and scored predictions: `PLAN_2.md` (sections 1 to 7 before any run; 8 before the reported seeds; 9 the outcome). Programs: `joint2.py` (numpy, the reference; the rule of withdrawal is in it for both paths), `kernw.c` with `fastw.py` (compiled), `check_joint2.py` (`check_joint2_output.txt`: passed). Tables: `report2.txt`. Result files: `res/w_*.json`.

The plain system (the one-sided edge), in which every 720,000 bytes an end may withdraw for good from each of its connections whose strength, as that end holds it, is below one tenth of the largest it holds: the answering cell compares the strengths of one branch; a cell in between compares its own return strengths. A withdrawn connection is at zero in both directions from then on.

| Who may withdraw | five seeds | median | paired against nobody | connections living at the end, of 3,584 |
|---|---|---|---|---|
| nobody (reproduces `cells2.py`) | 84.2, 82.7, 86.4, 81.8, 86.7 | 84.2 | | 3584 |
| the receiving end only | 83.5, 81.2, 87.9, 82.8, 86.2 | 83.5 | -0.5 | 2212 |
| the sending end only | 84.0, 81.4, 87.3, 81.7, 86.5 | 84.0 | -0.1 | 2308 |
| either end | 83.8, 81.2, 87.7, 82.6, 86.5 | 83.8 | -0.1 | 1823 |
| control: as many as either end withdrew, drawn by chance | 75.9, 77.1, 81.5, 81.6, 75.7 | 77.1 | -5.5 | 1823 |

With the threshold at one twentieth and at one fifth (same T, five seeds): receiving 83.3 and 83.9 (2,558 and 1,669 living), sending 83.2 and 83.3 (2,681 and 1,770), either 83.3 and 84.1 (2,259 and 1,182).

1. **In score the right to withdraw costs nothing, whoever holds it**: every arm is within 1.3 points of nobody withdrawing at the median, at every threshold, with half to a third of the connections left. Either end against the receiving end alone: 0.0.
2. **Either end leaves about a fifth fewer connections than one end** at the same score (1,823 against 2,212 and 2,308), since each end withdraws from connections the other would keep (of either end's withdrawals, 828 the receiving end's alone, 422 the sending end's alone, 494 both).
3. **The two ends withdraw from different connections.** The sending end's rule empties, on every seed, one whole branch of the answering cell (on the seed looked at, the branch for a byte value that never occurs in the text), which the receiving end's rule keeps in full; after the sending end's rule 1% of the living connections are below one hundredth of the largest, after the receiving end's 12%, as with nobody withdrawing.
4. **Withdrawing by the ends' own strengths is not the same as having fewer connections**: the same numbers withdrawn by chance cost 6 points on the median, below on five seeds of five.
5. With either end withdrawing at the threshold of one fifth, 4 to 12 of the 256 cells lose every connection: cells that have dropped out, in the sense of Mr. Xu's sentence "if one cell wants to drop out, then the connection strength to it should be zero no matter how willing other cells are", though here by the sum of single withdrawals and not by a rule for a whole cell.

## Step C2b: a whole cell dropping out

Plan, look, outcome and scored predictions: `PLAN_3.md`. Programs: `joint3.py` (numpy, the reference), `kernd.c` with `fastd.py`, `check_joint3.py` (`check_joint3_output.txt`: passed). Tables: `report3.txt`. Result files: `res/c_*.json`, `res/j_cr_B_*.json`.

The plain system, in which every 720,000 bytes a cell in between drops out for good (it sends nothing; all its connections, in and out, are zero from then on) if the sum of the sizes of the return strengths it holds is below c. Beside it: as many cells dropped by chance at the same moments, and the plain system born with the number of cells that are left. Medians of five seeds; nobody dropping out: 84.2.

| c | cells left | drop out by own rule | as many by chance | born at that size | own rule less born (paired; seeds below) |
|---|---|---|---|---|---|
| 0.09 | 217 | 83.4 | 83.4 | 85.7 | -2.3; 4 of 5 |
| 0.13 | 187 | 83.5 | 81.4 | 85.9 | -1.9; 4 of 5 |
| 0.20 | 128 | 82.9 | 83.2 | 85.6 | -4.8; 4 of 5 |
| 0.30 | 70 | 69.6 | 69.6 | 69.3 | +0.7; 1 of 5 |

1. **The plain system is not short of cells at 256**: born with 128 to 217 cells it scores 85.6 to 85.9; with 70 cells 69.3.
2. **Cells dropping out by this rule reach a smaller system that is a little worse than one born at that size** (2 to 5 points, four seeds of five, down to half the cells; the plan's hypothesis that the two are level is refuted narrowly, 4.8 against 4 points) **and no better than dropping as many by chance.** Unlike single connections (C2), a cell's own total return strength does not pick out what the system can spare.
3. The cost against nobody dropping out is small (0.5 to 1.3 points) while half the cells remain. The spread over seeds in the small systems is large (single runs at 53 and 73 where the others are near 85), so differences of 2 points are not fixed by five seeds.

## What the two steps say together (my reading)

For the connections into the answering cell on this testbed, the score does not tell a one-sided edge from an edge that both ends decide: where the joint forms differ from the one-sided edge they are level with it (the smaller part; the right to withdraw) or below it (the product with weight at the sending end). What both ends deciding changes is which connections exist: fewer of them, and none to a branch that names nothing. If the number of connections is to count, that is a reason for the joint form; the score alone gives none.

Limits: one testbed, one size, one length; one development seed for every choice and five reported seeds; a share-of-the-largest rule and one T on the reported seeds; the joint forms tried are two rules of many; nothing here has been repeated by another line.

## Commands and cost

`python3 check_joint.py`, `python3 check_joint2.py` (about a minute each); `xargs -P 4 -L 1 sh -c 'python3 joint.py "$0" "$@"' < rep1.txt` and likewise `dev1.txt` (see `PLAN_1.md`, section 9, on its labels), `dev2.txt`, `rep1b.txt`, `num1.txt` with `joint.py`; `dev3.txt`, `rep2.txt`, then `rep2x.txt`, `num2.txt` with `joint2.py`; `python3 report1.py > report1.txt`, `python3 report2.py > report2.txt`. A compiled run takes 17 to 26 seconds of one core, a numpy run 3.5 to 6 minutes. In all 57 compiled and 3 numpy runs for C1, 83 compiled and 1 numpy run for C2: about 50 minutes of CPU, half an hour of wall time on four cores. No checkpoints are kept.
