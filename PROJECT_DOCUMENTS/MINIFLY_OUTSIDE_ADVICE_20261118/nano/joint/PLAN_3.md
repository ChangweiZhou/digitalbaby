# Line C, third plan (step C2b): a whole cell dropping out (20 November 2026)

By the same separate agent session as `PLAN_1.md` and `PLAN_2.md`; same folder, same source commit (`fe397ca` of `claude/friendly-darwin-8mmhfm`). Written before any run of this step, except the look named in section 2. Development level; repeated by no one; no checker gate, seal or review is claimed.

**Authorisation.** After C1 and C2 the coordinating session sent this session a message asking for three further steps (C2b, C5, C4) "on the same testbed and under the same rules", within the four hours first given. That message is the coordinating session's, not Mr. Xu's; I have no word from him directly. I take the steps as lying inside what the coordinating session relayed at the start ("revise the road map and keep doing some experiments. Spin off some agents to speed up the experiment process.") and inside track C of the roadmap file; if that is wrong, these runs are small and nothing outside this folder is touched.

Mr. Xu's words that this step takes up (roadmap file, section 1): "For example if one cell wants to drop out, then the connection strength to it should be zero no matter how willing other cells are."

## 1. The question

If a cell in between may drop out for good by a rule that uses only its own quantities, does the system reach a smaller number of cells that scores as a system born with that number scores, better, or worse?

## 2. The system and the rule

The plain system exactly (form 1 of `joint.py`), with one addition: a cell in between is living or has dropped out. A cell that has dropped out sends nothing from then on (its output is zero, where a living logistic cell never sends less than a little), and all its connections are zero and do not change: its strengths from the lines, its threshold, the answering cell's strengths from it, and its return strengths. The other cells cannot keep or restore any of them.

**The rule.** Every T = 720,000 bytes (the T of `PLAN_2.md`; five moments in a run; by the first, 0.15% of a birth value of R is left) a living cell j drops out if rho_j, the sum over the 14 branches of the sizes of the return strengths it holds, is below c. Why this quantity: the return strengths are the one thing a cell holds that says how much the answering cell reads it (a return strength changes by the same product as the forward strength of its connection and the two agree with cosine 1.000 in the plain system), and they are also the size of what is relayed to it; a cell with small rho is little read and little taught. Why a fixed c and not a share of the median over the cells: the median over the cells is not a quantity a cell has. The cost of a fixed c is that it is a number on the scale of this system's strengths, and has to be set from a look.

**The look, made before this plan** (`res/c_look_L.json`: the plain run on seed 1, 81.5, with rho recorded at the five moments and nobody dropping out): the 10th, 25th, 50th and 75th percentiles of rho over the cells are 0.09, 0.13, 0.20 and 0.30 at the first moment and stay within 0.01 of that to the last (smallest 0.03 to 0.05, largest 0.6 to 0.7).

**Arms.** N: nobody drops out (must reproduce F1). D: the rule, at c in {0.09, 0.13, 0.20, 0.30}. X (control): at each moment as many living cells drop out, drawn by chance, as in D on the same seed and c. B: the plain system born with as many cells as D has left at the end on the same seed and c (`joint.py` with H = that number; all else the same), run for the same 400,000 lines.

**Numbers that learn:** 43,278 with 256 cells; 169 for each cell (140 strengths from lines, a threshold, 14 forward and 14 return strengths) and 14 constants: 169 n + 14 for n cells.

## 3. Measures

Never-shown sums (mean of the last three readings); cells living at the end and after each moment; the score of D against the number of cells left, beside B at the same number and X at the same numbers.

## 4. Settings and runs

Nothing is tuned beyond c; all four values go to the five reported seeds for D, X and B, so that the score is seen against the number of cells. One value is named beforehand as the headline: **the largest c at which D is within 2 points of N on development seed 1; if none is, c = 0.09.** Development: N and the four D on seed 1 (X and B on seed 1 are not needed for the choice and are not run). Reported seeds 73129, 93683, 116089, 406364, 545353: N, 4 D, then 4 X and 4 B: 65 runs. One full-length numpy run of D at the headline c on seed 1.

## 5. Hypothesis and what would refute it

**H: a system that comes to n cells by cells dropping out by their own rule scores as a system born with n cells: D is within 4 points of B.** Judged at c = 0.20 (about half of the cells at the first moment), where the two systems differ most in size from N while, I expect, still learning. Refuted if the paired difference D less B has a median larger than 4 points in size and the same sign on at least four of five seeds. Upward: dropping out reaches something a small system does not find from birth. Downward: a cell that drops out takes with it something the remaining cells do not make good.

## 6. Predictions (mine), with chances

- R1 (0.9): N gives 81.5 on seed 1 and the report's five figures.
- R2 (0.6): D at the headline c is within 4 points of N at the median of the paired differences.
- R3 (0.55): at c = 0.20, D is above B on at least four of five seeds.
- R4 (0.75): at c = 0.20, X is below D on at least four of five seeds.
- R5 (0.6): H stands.
- R6 (0.6): at c = 0.20, fewer than 128 cells are left at the end at the median (cells go on dropping out after the first moment).
- R7 (0.6): B with about 190 cells (c = 0.13) is within 4 points of N at the median.

## 7. Cost, not run

Check: about a minute. 5 development and 65 reported compiled runs of at most 20 seconds of one core: about 6 minutes on four cores. One numpy run of about 4 minutes.

Not run: other own quantities (the running size of the relayed sum; the cell's own change); other T; a cell that returns; dropping out together with the withdrawal of single connections; cells born later.

---
