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
4. **Connections near zero.** One-sided: 11% of the connections are below one hundredth of the largest (7.1 points of these are the 256 connections of the branch for a byte value that never occurs, which are exactly zero in every form; `PLAN_1.md`, section 12); removing them, or the weakest half, costs nothing; the weakest three quarters cost about 7 points. The product with weight at the sending end leaves 58% below one hundredth, at a score 19 points lower.
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
3. **The two ends withdraw from different connections** (corrected 20 November; see `PLAN_2.md`, section 10: the branch that the receiving end keeps has all its strengths at exactly zero, and it is kept by the strict comparison in my rule, not by a judgement of the answering cell; without that branch the receiving end leaves about 1,956 connections, and either end about 7% fewer than that). The sending end's rule empties, on every seed, one whole branch of the answering cell (on the seed looked at, the branch for a byte value that never occurs in the text), which the receiving end's rule keeps in full; after the sending end's rule 1% of the living connections are below one hundredth of the largest, after the receiving end's 12%, as with nobody withdrawing.
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
3. The cost against nobody dropping out is small (0.5 to 1.3 points) while half the cells remain. The spread over seeds in the small systems is large (single runs at 53 and 73 where the others are near 85) [erratum, 20 November, pointed out by the coordinating session's check: only the run at 72.6 (born with 178 cells) has peers near 85 (85.5 to 90.1); the run at 53.5 (75 cells dropped to by chance, c = 0.30) has peers at 65.6 to 79.1], so differences of 2 points are not fixed by five seeds.

## Step C5: how a strength is fixed

Plan, grids, outcome and scored predictions: `PLAN_4.md`. Programs: `joint5.py` (numpy, the reference), `kern5.c` with `fast5.py`, `check_joint5.py` (`check_joint5_output.txt`: passed). Tables: `report5.txt`. Result files: `res/s_*.json`.

The one-sided edge kept; one rule of change replaced at a time; the return strength receives the forward strength's change in every arm. Medians of five seeds; settings chosen on seed 1.

| Rule | median | against the present rules (paired) |
|---|---|---|
| the present rules | 84.2 | |
| answering cell: no division by the squared size of what it reads; best fixed step | 81.8 | -1.7 (no common sign) |
| answering cell: a change of fixed size, sign of mismatch times output | 0.0 | -84.2 |
| answering cell: the same with the output taken about the middle of its range (addition) | 1.3 | -82.5 |
| answering cell: in proportion to the strength's own size | 64.1 | -17.7 (5 of 5 below) |
| cells in between: a change of fixed size | 0.8 | -83.3 |
| cells in between: in proportion to the strength's own size | 85.5 | -1.1 (no common sign) |

1. **The division is not needed for the score** (-1.7, inside the spread), but the fixed step that replaces it has to be found, and is narrow (an eighth of it: 65; twice it: 74, on seed 1).
2. **A change of fixed size answers nothing**, in either place, at any step tried. For the rule as asked this is by construction (a logistic output is never negative, so every strength of a branch receives the same change); for the other two it is a finding I have not explained.
3. **A change in proportion to the strength's own size loses 18 points in the answering cell** and leaves 86% of the connections near zero: the same picture as the product edge of C1 with weight at the sending end. Two rules in which a strong connection changes faster than a weak one lose alike. In the cells in between the mildest form of the same rule is level with the present one.

## Step C4: the sign, with and without the right to withdraw

Plan, outcome and scored predictions: `PLAN_5.md`. Programs: `joint4.py` (numpy, the reference), `kerns.c` with `fasts.py`, `check_joint4.py` (`check_joint4_output.txt`: passed). Tables: `report4.txt`. Result files: `res/g_*.json`.

One sign for each cell in between (the `dale=1` of `cells2.py`, a quarter of the cells inhibitory), and the right of either end to withdraw (arm E of C2, same setting); each present or absent. Medians of five seeds.

| | never-shown sums | connections living, of 3,584 |
|---|---|---|
| neither (plain) | 84.2 | 3584 |
| the sign | 70.7 | 3584 |
| withdrawal by either end | 83.8 | 1823 |
| both | 68.3 | 771 |

1. **The right to withdraw does not change the cost of the sign**: the cost with withdrawal less the cost without, seed by seed, is -0.6 at the median (-0.1, -1.9, -3.1, -0.6, +6.1). The sign costs 14 to 19 points either way.
2. **Under the sign, withdrawal leaves about a fifth of the connections (771) at the score of the sign alone** (+0.4 paired at the median). The sign makes the strengths unequal (44% below one hundredth of the largest against 4% in the plain system apart from the dead branch), and those are withdrawn at no cost.
3. The sign alone reproduces the earlier 70.7, 73.9, 64.0, 62.1, 72.3 to the digit with a different kernel.

## Additions of 20 November: the product with a bounded step, and fifteen seeds for the close comparisons

Asked by the coordinating session; entered in the plans before their runs (`PLAN_1.md` sections 13 and 14, `PLAN_3.md` 10 and 11, `PLAN_4.md` 9 and 10, `PLAN_5.md` 9 and 10). Tables: `report_add.txt`. The ten further seeds are 1001 to 1010; the five reported seeds stay the first table everywhere above. Plain system on fifteen seeds: median 83.7.

| Comparison (paired, same seed) | five seeds, median | fifteen seeds, median | seeds of fifteen above / below |
|---|---|---|---|
| product with weight at the sender and a bounded step, less one-sided | -24.5 | -15.0 | 1 / 14 |
| smaller part (sender's part born 1/64), less one-sided | +2.3 | +0.9 | 12 / 3 |
| answering cell without the division, less present rule | -1.7 | 0.0 | 7 / 7 |
| cells in between in proportion to own size, less present rule | -1.1 | -1.1 | 7 / 8 |
| cost of the sign with withdrawal less cost without | -0.6 | -0.6 | 5 / 10 |
| the sign with withdrawal, less the sign alone | +0.4 | +0.4 | 8 / 7 |
| withdrawal by either end, less plain | -0.1 | -0.6 | 3 / 12 |
| cells dropping out by own rule, less as many by chance (c = 0.13) | +0.9 | +1.5 | 10 / 4 |

1. **Bounding the step does not rescue the product edge** (-24.5 on five seeds, lower than with the plain step). Its loss is not steps that are too large; what remains is which connections it lets grow. (Reservation in `PLAN_1.md`, section 14.)
2. **Three statements above are sharpened by the fifteen seeds.** The smaller-part edge is about one point above the one-sided edge, on twelve seeds of fifteen: small, but it is the one joint form that is not level or below. The right to withdraw costs about half a point (twelve of fifteen below), not nothing. A cell dropping out by its own rule is about a point and a half better than one dropped by chance (ten of fifteen), not level with it.
3. **The others stand as they were**: the division is not needed; the cells' proportional rule is level; withdrawal does not change the cost of the sign (17 to 19 points, on fifteen seeds of fifteen).

## What the steps say together (my reading)

1. **The score hardly tells a one-sided edge from an edge that both ends decide** (the largest difference in favour of a joint form is one point, for the smaller of two parts, on fifteen seeds), for the connections into the answering cell on this testbed. Where the joint forms differ from the one-sided edge they are level with it (the smaller of two parts; the right to withdraw held by either end, with or without one sign for each cell) or below it (the product with weight at the sending end).
2. **What both ends deciding changes is which connections exist.** Withdrawal by the ends' own strengths removes half of the connections at no cost, and four fifths under the sign, and is 6 points better than removing as many by chance. For whole cells the like holds only faintly: a cell dropping out by its own total return strength does about a point and a half better than a cell dropped by chance (fifteen seeds), and a little worse than a system born at the smaller size (five seeds).
3. **Rules in which a strong connection changes faster than a weak one lose in the answering cell**: the product with weight at the sending end (-19) and the change in proportion to own size (-18), both ending with most connections near zero and a few large ones. The division in the present rule is not needed for the score; the size of the mismatch is (a change of fixed size answers nothing).
4. **The plain system is not short of cells at 256**: 128 cells score as well.

[Added 20 November, last: every score in this report is an "empty past" score (a never-shown line scored alone, with line ends in the delays before it), which the coordinating session has since found to understate the relay system by some ten points and which, in `nano/replay` (`PLAN_6.md`, section 10), changed the sign of one difference between arms and removed another. Nothing in this folder has been re-scored after a read line; its differences between arms are empty-past differences.] Limits: one testbed, one size, one length; one development seed for every choice (it misled in size three times, each recorded) and five reported seeds, whose spread is about 5 points for the plain system and much more for small or constrained systems; one T on the reported seeds; share-of-the-largest and fixed-threshold rules only; the joint forms and rules tried are a few of many; nothing here has been repeated by another line. Two corrections of my own readings are recorded where they arose (`PLAN_2.md`, section 10; `PLAN_1.md`, section 12).

## Commands and cost

`python3 check_joint.py`, `check_joint2.py`, `check_joint3.py`, `check_joint5.py`, `check_joint4.py` (about a minute each). Runs: `xargs -P 4 -L 1 sh -c 'python3 <program> "$0" "$@"' < <list>` with `joint.py` for `dev1.txt` (see `PLAN_1.md`, section 9, on its labels), `dev2.txt`, `rep1.txt`, `rep1b.txt`, `num1.txt`, `rep3b.txt`; `joint2.py` for `dev3.txt`, `rep2.txt`, then `rep2x.txt`, `num2.txt`; `joint3.py` for `dev4.txt`, `rep3.txt`, then `rep3x.txt`; `joint5.py` for `dev5.txt`, `rep5.txt`; `joint4.py` for `rep4.txt`. The lists `rep3x.txt` and `rep3b.txt` are made from the results of `rep3.txt` (the numbers of cells left). Tables: `python3 report1.py > report1.txt` and likewise `report2`, `report3`, `report5`, `report4`. A compiled run takes 17 to 37 seconds of one core, a numpy run 3.5 to 6 minutes. Compiled runs: 57 (C1), 83 (C2), 71 (C2b), 62 (C5), 24 (C4), 105 (the additions: `add_j.txt` with `joint.py`, `add_5.txt` with `joint5.py`, `add_4.txt` with `joint4.py`, `add_3.txt` then `add_3x.txt` with `joint3.py`; `python3 report_add.py > report_add.txt`); numpy runs at full length: 3, 1, 1, 1, 1. About three hours of CPU, an hour and a half of wall time on four cores. No checkpoints are kept; compiled `.so` files are not committed.
