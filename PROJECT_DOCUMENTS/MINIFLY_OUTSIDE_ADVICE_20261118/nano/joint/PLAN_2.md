# Line C, second plan (step C2): the right to withdraw from a connection (20 November 2026)

By the same separate agent session as `PLAN_1.md`; same folder, same source commit (`fe397ca` of `claude/friendly-darwin-8mmhfm`), same authorisation as recorded there (relayed by the coordinating session; no word to me from Mr. Xu directly). Written before any run of this step and after the runs of `PLAN_1.md`. Development level; repeated by no one; no checker gate, seal or review is claimed.

Mr. Xu's words that this step takes up (roadmap file, section 1): "For example if one cell wants to drop out, then the connection strength to it should be zero no matter how willing other cells are."

## 1. The question

In the plain system (the one-sided edge, F1 of `PLAN_1.md`), let an end of a connection withdraw from it for good. Does it matter for never-shown sums, and for how many connections remain, which end holds that right: nobody, the receiving end only (the answering cell), the sending end only (the cell in between), or either?

## 2. The system and the rule

The plain system exactly (as F1 of `PLAN_1.md`, which reproduces `cells2.py`), with one addition: each of the 3,584 connections from a cell in between j to a branch v of the answering cell is living or withdrawn. A withdrawn connection is gone for good: its forward strength A[v, j] and its return strength R[j, v] are zero and never change again (my reading: the forward and the return strength belong to one connection, so the mismatch of branch v no longer reaches cell j over it). Everything else is unchanged; the base is F1, not F2, because `PLAN_1.md` found F2's sending part either idle or harmful and an arm is to differ from the plain system in one thing.

Every T bytes read (all bytes counted; not at the last byte of the run) each end that holds the right looks at its own strengths:

- **The receiving end** (the answering cell, branch by branch) withdraws from every living connection of branch v whose |A[v, j]| is below theta times the largest |A[v, j']| among the living connections of that branch. It compares only strengths it holds.
- **The sending end** (cell j) withdraws from every living connection of its own whose |R[j, v]| is below theta times the largest |R[j, v']| among its living connections. It compares only the return strengths it holds; it does not see A.
- **Either:** both rules at the same moment, from the values before any withdrawal at that moment; a connection is withdrawn if either end withdraws.

An end never withdraws from its strongest connection, so no branch and no cell withdraws from everything by its own rule; a cell can still lose all its connections in the arms where the receiving end withdraws, and I count such cells.

What I know beforehand about the two ends' quantities: R is born uniform in (-1, 1) and A at zero, both fade at 1e-5 a byte, and at the end of a plain run their cosine is 1.000. So late in a run the two ends hold nearly the same number for a connection but compare it with different neighbours (the receiving end across the 256 senders of a branch, the sending end across its own 14 branches); early in a run the sending end's numbers are mostly its birth values (a line is ten bytes, at nine of which the system learns and fades; after 36,000 bytes 72% of a birth value is left, after 180,000 bytes 20%, after 720,000 bytes 0.15%).

**Arms.** N: nobody withdraws (must reproduce F1: 81.5 on seed 1 and the report's five figures). R: the receiving end only. S: the sending end only. E: either. And a control of mine, X: at each moment as many living connections are withdrawn, drawn by chance, as arm E withdrew at that moment on the same seed (so X has E's number of connections at every moment and nobody's rule).

**Numbers that learn:** 43,278 as in F1, less two for each withdrawn connection.

## 3. Measures

1. Never-shown sums, whole answer right on 500 pairs, mean of the last three readings.
2. How many of the 3,584 connections remain at the end; how many cells have none left; in E, how many were withdrawn by the receiving end alone, the sending end alone, and both at the same moment.
3. The share of connections below one hundredth of the largest at the end, of all 3,584 (withdrawn ones are at zero and count) and of the living ones.
4. The score at the end after the living connections below one hundredth of the largest are removed as well, beside the score at the end.

## 4. Settings, chosen on development seed 1

T in {36,000; 180,000; 720,000} bytes (3,600, 18,000 and 72,000 lines; 111, 22 and 5 moments of withdrawal in a run of 4,000,000 bytes) and theta in {0.05, 0.1, 0.2}: nine pairs, each for R, S and E: 27 runs. One pair (T, theta) is chosen for all arms, so that the arms differ only in who holds the right: **the pair at which E leaves the fewest connections while E's score is within 2 points of N's on seed 1; if no pair is within 2 points, the pair with the highest score of E.** The whole grid is reported for seed 1.

Then the five reported seeds 73129, 93683, 116089, 406364, 545353 for N, R, S, E and X at the chosen pair. An addition, declared here: at the chosen T, the other two values of theta for R, S and E on the five seeds as well (30 runs), so that the arms can be compared at like numbers of remaining connections and not only at a like rule.

## 5. Hypothesis and what would refute it

**H: the right to withdraw held by the receiving end alone is a simplification that costs nothing against the right held by either end: R's score is within 4 points of E's.** Refuted if the paired difference E less R (same seed) has a median larger than 4 points in size and the same sign on at least four of five seeds. Beside it I report the numbers of connections remaining, since a like score with fewer connections is also a difference; I put no threshold on it beforehand.

## 6. Predictions (mine), with chances

- Q1 (0.9): N gives 81.5 on seed 1 and 84.2, 82.7, 86.4, 81.8, 86.7 on the five seeds.
- Q2 (0.6): R is within 4 points of N at the median of the paired differences.
- Q3 (0.35): S is more than 4 points below N at the median of the paired differences.
- Q4 (0.6): H stands.
- Q5 (0.65): R leaves fewer connections than S on at least four of five seeds.
- Q6 (0.7): X scores below E on at least four of five seeds.
- Q7 (0.5): E leaves fewer than half of the connections at the median.
- Q8 (0.6): on the development grid, S at T = 36,000 scores more than 4 points below S at T = 720,000 at theta = 0.2 (the sending end withdrawing by its birth values).

## 7. Order of work, cost, not run

`joint2.py` (numpy reference; the rule of withdrawal is in Python, once, for both paths), `kernw.c` (a copy of `kern3.c` with the living connections as a mask) with `fastw.py`, `check_joint2.py` (numpy against compiled with withdrawals taking place, every strength and the set of living connections compared; N against F1 of `joint.py`). One full-length numpy run of E at the chosen pair on seed 1 beside the compiled one. Cost: 27 development runs, 25 + 30 on the reported seeds, each 16 to 25 seconds of one core: about 8 minutes on four cores; the numpy run about 6 minutes.

Not run: withdrawal in F2 or F3; withdrawal of connections from the input lines; a withdrawn connection made again (rewiring); rules that use a running mean of what passed over a connection; other T or theta; a whole cell dropping out by its own rule; C3 to C6.

---

## 8. Entered 20 November after the check and the development grid, before any run on the reported seeds

**The check** (`check_joint2_output.txt`, run before this plan was committed; it trains nothing at length): compiled against numpy with withdrawals taking place, 300 and 3,000 lines, every arm, two values of theta: strengths agree to 1e-14 to 4e-13, the sets of living connections are identical, withdrawn connections are at zero in both directions; nobody withdrawing is form 1 of `joint.py`.

**Development seed 1** (`res/w_wd_*.json`), never-shown sums (mean of the last three readings) and connections living at the end of 3,584. N: 81.5 (the value to match).

| T (bytes) | theta | R | S | E |
|---|---|---|---|---|
| 36,000 | 0.05 | 73.6, 1767 | 79.9, 1913 | 73.6, 1354 |
| 36,000 | 0.1 | 75.1, 1469 | 78.1, 1659 | 74.9, 976 |
| 36,000 | 0.2 | 50.5, 1106 | 77.1, 1265 | 53.2, 581 |
| 180,000 | 0.05 | 77.2, 2258 | 77.5, 2265 | 76.0, 1852 |
| 180,000 | 0.1 | 76.4, 1930 | 77.1, 1887 | 76.4, 1420 |
| 180,000 | 0.2 | 74.7, 1432 | 76.9, 1427 | 75.5, 855 |
| 720,000 | 0.05 | 79.8, 2612 | 81.1, 2738 | 79.5, 2306 |
| 720,000 | 0.1 | 80.1, 2240 | 79.1, 2362 | 80.3, 1854 |
| 720,000 | 0.2 | 77.9, 1638 | 78.1, 1834 | 76.7, 1130 |

**Choice by the rule of section 4:** E is within 2 points of N at (720,000; 0.05) and (720,000; 0.1); the second leaves fewer connections. **T = 720,000 bytes, theta = 0.1** for every arm: five moments of withdrawal, after 72,000, 144,000, 216,000, 288,000 and 360,000 lines.

Q8 failed on this grid: S at T = 36,000 and theta = 0.2 gives 77.1 against 78.1 at T = 720,000, one point and not four. On this seed it is the receiving end that loses by withdrawing early and often (50.5 at T = 36,000, theta = 0.2), not the sending end, against what I expected from the birth values of R.

Runs on the reported seeds, as planned: N, R, S, E, X at the chosen pair; R, S, E at theta = 0.05 and 0.2 with the same T.
