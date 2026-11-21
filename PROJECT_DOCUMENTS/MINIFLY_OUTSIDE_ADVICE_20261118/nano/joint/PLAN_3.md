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

## 8. Entered 20 November after the check and the development runs, before any run on the reported seeds

The check (`check_joint3_output.txt`): passed; compiled against numpy 1e-14 to 4e-13 with cells dropping out, the same sets of living cells, dropped cells at zero; nobody dropping out is form 1 of `joint.py` at 256 and at 100 cells.

Development seed 1, never-shown sums (mean of the last three readings) and cells left: N 81.5, 256. D at c = 0.09: 81.1, 219. c = 0.13: 81.3, 188. c = 0.20: 77.1, 129. c = 0.30: 72.3, 62. Nearly all the dropping out is at the first moment (25, 61, 126, 193 cells; 12, 7, 1, 1 later).

**Headline by the rule of section 4: c = 0.13.** All four values go to the reported seeds as planned.

## 9. Outcome (entered 20 November, after all runs of this plan)

Tables: `report3.txt` (made by `report3.py` from `res/c_*.json` and `res/j_cr_B_*.json`). The numpy program at full length (D, c = 0.13, seed 1) gives the compiled program's readings to the digit (81.3; 188 cells).

Never-shown sums, mean of the last three readings; medians of the five reported seeds, with the five figures beneath. N (256 cells): **84.2** (84.2, 82.7, 86.4, 81.8, 86.7).

| c | cells left (median; range) | D, own rule | X, as many by chance | B, born at that size | D less B (paired median; seeds below) | D less X |
|---|---|---|---|---|---|---|
| 0.09 | 217; 211 to 222 | **83.4** | **83.4** | **85.7** | -2.3; 4 | +0.5 |
| | | 83.4, 80.7, 86.8, 81.3, 86.3 | 84.3, 85.5, 83.4, 80.8, 81.9 | 85.7, 82.8, 85.2, 87.9, 91.1 | | |
| 0.13 (headline) | 187; 178 to 193 | **83.5** | **81.4** | **85.9** | -1.9; 4 | +0.9 |
| | | 83.5, 82.2, 86.9, 79.7, 83.9 | 88.7, 79.1, 85.9, 81.4, 79.4 | 85.5, 88.1, 90.1, 72.6, 85.9 | | |
| 0.20 | 128; 125 to 132 | **82.9** | **83.2** | **85.6** | -4.8; 4 | -1.4 |
| | | 82.9, 82.9, 82.1, 86.0, 80.0 | 84.9, 83.2, 86.8, 82.8, 81.4 | 87.7, 87.9, 85.5, 77.3, 85.6 | | |
| 0.30 | 70; 64 to 75 | **69.6** | **69.6** | **69.3** | +0.7; 1 | -0.7 |
| | | 79.7, 74.5, 65.3, 69.6, 68.4 | 53.5, 75.1, 69.6, 79.1, 65.6 | 79.7, 69.3, 64.6, 60.3, 71.7 | | |

**The hypothesis.** At c = 0.20, where it was to be judged, D less B is -4.8, -5.0, -3.4, +8.7, -5.6: median -4.8, below on four seeds of five. By the rule of section 5 **H is refuted, downward, and narrowly** (4.8 against the 4 points; one seed is 8.7 the other way; a system born at 125 cells gave 77.3 on that seed and 85 to 88 on the others). The same sign shows at c = 0.09 and 0.13 (-2.3 and -1.9, four seeds below each); at c = 0.30, with 70 cells, the three arms are level (69.6, 69.6, 69.3).

**What I read from it.**

1. *The plain system is not short of cells at 256.* Born with 187 or 128 cells it scores 85.9 and 85.6 at the median, against 84.2 with 256 (paired against N: +1.3 and -0.9); with 70 cells 69.3. I had not known this; it bears on every comparison "at equal numbers of cells" in this line.
2. *Dropping out by this rule reaches a smaller system that is a little worse than a system born at that size* (2 to 5 points at the median for 217 to 128 cells, four seeds of five each time), *and no better than dropping the same numbers of cells by chance* (D less X: +0.5, +0.9, -1.4, -0.7). A cell's own total return strength does not single out cells the system can spare better than chance does. This is unlike the single connections of `PLAN_2.md`, where the ends' own rule was 6 points better than chance.
3. *The cost against N is small while half the cells remain* (-0.5, -0.7, -1.3 at the median), so cells can drop out by their own rule at little cost here; but a system born small does as well or better, so nothing is gained by starting large and letting cells drop out, at this length of run.
4. The spread over seeds is large in the small systems (B at 178 cells: 72.6; X at 75 cells: 53.5), and five seeds do not fix differences of 2 points.

A reason for 2 that I have not tested: a cell's return strengths sum what the answering cell reads from it over all 14 branches, most of which name bytes that carry no sum (the operands' digits, the signs); a cell read weakly in total may still carry a digit of the answer. A rule on what the cell receives at the answer's bytes only would test that; it needs the cell to know which bytes those are, which it does not.

**Predictions.** R1 held. R2 held (-0.7). R3 failed (D above B on one seed of five). R4 failed (X below D on one seed of five). R5 failed (H refuted). R6 failed (128 at the median, and nearly all dropping out is at the first moment). R7 held (+1.3).

**Not run:** as section 7; X and B on the development seed; more seeds.

## 10. A dated addition, asked by the coordinating session, entered 20 November before its runs

Ten further seeds (1001 to 1010; the rule is quoted in `PLAN_1.md`, section 13) for the comparison it would settle: D against X at c = 0.13. Reported: the fifteen together.
- W1 (0.6): D less X is within 2 points at the median of fifteen seeds.

## 11. Outcome of the addition of section 10 (entered 20 November)

Tables: `report_add.txt`. c = 0.13, fifteen seeds (the five reported, then 1001 to 1010); cells left: median 186 (176 to 194).

- D less X: -5.2, +3.1, +0.9, -1.7, +4.5 | -3.5, +11.0, +1.5, +5.3, +0.4, +1.6, +3.1, 0.0, -0.6, +2.6. Median of fifteen +1.5; above zero on ten seeds, below on four, equal on one. Medians of the scores: D 83.9, X 81.4.
- Against nobody dropping out: D -0.5 at the median (nine below, six above); X -1.7 (ten below, five above).

So section 9, item 2, was too flat: at c = 0.13 on fifteen seeds, dropping out by the cell's own rule is a little better than dropping as many by chance, by about a point and a half, on ten seeds of fifteen. It is a small advantage and nothing like the 6 points by which the ends' own rule beat chance for single connections. Not rerun on the further seeds: the systems born at that size, and the other values of c.

**Prediction.** W1 held (+1.5).
