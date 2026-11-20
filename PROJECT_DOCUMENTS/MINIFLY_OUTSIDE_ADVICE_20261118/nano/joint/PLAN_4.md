# Line C, fourth plan (step C5): how a strength is fixed (20 November 2026)

By the same separate agent session; same folder, same source commit (`fe397ca` of `claude/friendly-darwin-8mmhfm`); authorisation as recorded in `PLAN_3.md` (asked by the coordinating session after C2; no word to me from Mr. Xu directly). Written before any run of this step. Development level; repeated by no one; no checker gate, seal or review is claimed.

Mr. Xu's words that this step takes up (roadmap file, section 1): "Similarly how the strength is fixed even in this case can be experimented, and you can try different rules."

## 1. The question

With the one-sided edge kept, does the rule by which a strength is changed matter for never-shown sums? Part A: the answering cell's strengths. Part B, after A: the strengths of the cells in between.

## 2. The rules

Common: the plain system (form 1 of `joint.py`). m_v: mismatch of branch v; h_j: output of cell j; g = eta2 / (h.h + 1) with eta2 = 1/16. In every arm the return strength R[j, v] receives the same change as the forward strength A[v, j] (as the coordinating session asked), the constants of the branches change by g m_v as now, and both A and R fade at 1e-5 a byte as now.

**Part A, the answering cell** (the cells in between keep their present rule):

- (i) the present rule: A += g m h.
- (ii) the same without the division: A += p m h, p a fixed step.
- (iii) a change of fixed size in the direction of the product: A += p sign(m h).
- (iii-b), an addition of mine: A += p sign(m (h - 1/2)). The reason is stated before the runs: a logistic output is never below zero, so the sign of m h is the sign of m alone, every strength of a branch receives the same change at every byte, all 256 strengths of a branch stay equal, and the branch cannot tell one cell from another. I expect (iii) to answer nothing; it is run as asked. Taking the output about the middle of its range gives the product a sign that depends on the sending cell.
- (iv) a change in proportion to the strength's own size: A += g m h (|A| + p) / (1/64 + p). 1/64 is the mean |A| at the end of the plain run on seed 1 (0.0156, `PLAN_1.md` section 9), so that a strength of the usual size moves as under (i); p is the floor that lets a strength born at zero move at all: at p = 1/64 a strength at zero moves at half the present rate, at p = 1/1024 at 6% of it.

**Part B, the cells in between** (the answering cell keeps rule (i)). sig_j = (R[j] . m) h_j (1 - h_j) is what cell j receives times its own sensitivity; at present its strengths from the ten active lines and its threshold change by eta1 sig_j with eta1 = 1/4. The present rule has no division, so (i) and (ii) are one rule here.

- (iii) a change of fixed size: the strengths from the active lines and the threshold change by q sign(sig_j). (No degeneracy here: the sign differs from cell to cell.)
- (iv) in proportion to own size: a strength from an active line changes by eta1 sig_j (|W1| + q) / (1/4 + q); the threshold as now. 1/4 is the mean size of a strength from a line at birth.

**Numbers that learn:** 43,278 in every arm.

## 3. Settings, on development seed 1

Part A: (i) once (81.5). (ii) p in {1, 2, 4, 8, 16, 32} x 1e-4. (iii) and (iii-b) p in {1e-6, 3e-6, 1e-5, 3e-5, 1e-4}. (iv) p in {1/64, 1/256, 1/1024}. For each rule the value with the highest score (mean of the last three readings) goes to the five reported seeds; a rule that answers nothing at every value goes at its best value all the same.
Part B: (iii) q in {1e-5, 1e-4, 1e-3, 1e-2}; (iv) q in {1/4, 1/16, 1/64}; the best of each to the five seeds.
Measures: never-shown sums; beside it the largest and mean |A|, the share of connections below one hundredth of the largest, and the cosine of R with A (1 by construction, as a check).

## 4. Hypothesis and what would refute it

**H: the division by the squared size of what the answering cell reads is not needed: (ii) at its best fixed step is within 4 points of (i).** Refuted if the paired difference (ii) less (i) has a median larger than 4 points in size and the same sign on at least four of five seeds.

## 5. Predictions (mine), with chances

- S1 (0.9): (i) reproduces 81.5 and the report's five figures.
- S2 (0.65): H stands.
- S3 (0.9): (iii) is below 5% at the median.
- S4 (0.4): (iii-b) at its best step is within 4 points of (i) at the median of the paired differences.
- S5 (0.5): (iv) at its best p is more than 4 points below (i) at the median of the paired differences.
- S6 (0.7): on seed 1 the score of (iv) falls as p falls from 1/64 to 1/1024 (the smaller the floor, the more the rule resembles the product form of `PLAN_1.md`, which lost).
- S7 (0.5): Part B (iii) at its best step is within 10 points of (i) at the median.
- S8 (0.6): Part B (iv) at its best q is within 4 points of (i) at the median of the paired differences.

## 6. Cost, not run

Check: a minute. Part A: 20 development runs and 25 on the reported seeds; Part B: 7 and 10; each at most 25 seconds of one core: about 7 minutes on four cores. One full-length numpy run of (ii) at its best step on seed 1; the sign rules are discontinuous, so that two orders of summation can part after some thousand lines, and for them I compare levels and not readings.

Not run: rules with a running mean held by a cell; rules for the constants or the thresholds alone; rules for R different from those for A; the fading; two changed rules together.

---

## 7. Entered 20 November after the check and the development runs, before any run on the reported seeds

The check (`check_joint5_output.txt`): passed, 1e-14 to 1e-11 on every rule at 300 and 3,000 lines; the present rules are form 1 of `joint.py`.

Development seed 1, never-shown sums, mean of the last three readings (`res/s_sd_*.json`):

- (i): 81.5.
- (ii), p = 1, 2, 4, 8, 16, 32 x 1e-4: 65.2, 77.5, 79.7, **81.0**, 74.3, 77.6.
- (iii), p = 1e-6 to 1e-4: 0.0 at every value (as expected: the share of connections below one hundredth of the largest is 7.1% at every step, which is one branch of fourteen; inside a branch all strengths are equal).
- (iii-b), the same steps: 0.0, 0.0, 0.0, 1.9, **3.7**.
- (iv), p = 1/64, 1/256, 1/1024: 66.1, 56.5, **71.4**. The largest strength is 0.9 to 1.0 against 0.11 under (i), and 72 to 87% of the connections end below one hundredth of the largest.
- Part B (iii), q = 1e-5 to 1e-2: 0.1, 0.0, 0.4, **0.8** (at the two largest steps the strengths from the lines grow to a mean size of 17 and 37 and the cells saturate).
- Part B (iv), q = 1/4, 1/16, 1/64: **72.8**, 62.1, 69.3.

To the reported seeds, by the rule of section 3: (i); (ii) at 8e-4; (iii) at 1e-5 (every value gives 0.0; I take the middle one); (iii-b) at 1e-4; (iv) at 1/1024; Part B (iii) at 1e-2 and (iv) at 1/4.

S6 failed on this grid: the score of (iv) does not fall with the floor (66.1, 56.5, 71.4).

A remark before the reported seeds, so that it is not made afterwards: the changes of fixed size answer nothing at any step tried, in the answering cell and in the cells in between, and I did not expect that of (iii-b) or of Part B (iii) (S4 at 0.4, S7 at 0.5). Under a change of fixed size with a fading of 1e-5 a byte no strength can exceed p / 1e-5, and a strength pushed one way for long sits at that bound whatever the size of the mismatch; I have not tested whether that, or the loss of the mismatch's size as such, is the cause. No further grid is run for them in this plan.

## 8. Outcome (entered 20 November, after all runs of this plan)

Tables: `report5.txt` (made by `report5.py` from `res/s_*.json`). The numpy program at full length ((ii) at 8e-4, seed 1) gives the compiled program's readings to the digit (81.0).

Never-shown sums, whole answer right, %, mean of the last three readings; five reported seeds, median, and the median of the paired differences from (i).

| | 73129 | 93683 | 116089 | 406364 | 545353 | median | against (i) |
|---|---|---|---|---|---|---|---|
| (i) the present rules | 84.2 | 82.7 | 86.4 | 81.8 | 86.7 | **84.2** | |
| A (ii) no division, fixed step 8e-4 | 86.3 | 82.9 | 81.8 | 80.1 | 79.7 | **81.8** | -1.7; 3 below, 2 above |
| A (iii) fixed size 1e-5, sign of m h | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | **0.0** | -84.2 |
| A (iii-b) fixed size 1e-4, sign of m (h - 1/2) | 4.7 | 0.1 | 0.7 | 2.6 | 1.3 | **1.3** | -82.5 |
| A (iv) in proportion to own size, floor 1/1024 | 67.9 | 71.7 | 43.8 | 64.1 | 60.8 | **64.1** | -17.7; 5 below |
| B (iii) cells: fixed size 1e-2 | 0.9 | 0.8 | 1.2 | 0.7 | 0.4 | **0.8** | -83.3 |
| B (iv) cells: in proportion to own size, floor 1/4 | 89.3 | 80.9 | 78.5 | 87.4 | 85.5 | **85.5** | -1.1; 3 below, 2 above |

**The hypothesis.** H stands: without the division, at the best fixed step, the answering cell scores -1.7 at the median of the paired differences (+2.1, +0.3, -4.6, -1.7, -6.9), inside the 4 points and without a common sign. The division is not what makes the system work. Two things beside that: the fixed step had to be found (65.2 at an eighth of the best step, 74.3 at twice it, on seed 1), where the division needs no such search; and the two largest losses (-4.6, -6.9) are larger than any gain, so a cost of a few points is not excluded by five seeds.

**What I read from the other rules.**

1. *A change of fixed size answers nothing, in the answering cell and in the cells in between* (0.0, 1.3, 0.8). For (iii) as asked that is by construction (section 2). For (iii-b) and for the cells it is a finding at the steps tried: the size of the mismatch is needed by this system, or the bound p / 1e-5 that the fading puts on a strength under such a rule spoils it; I have not separated the two.
2. *A change in proportion to the strength's own size loses 18 points in the answering cell* (five seeds below) and leaves 86% of the connections below one hundredth of the largest, with the largest strength near 0.8 against 0.09. This is the same picture as the product form of `PLAN_1.md` with weight at the sending end (-19.2 points, 58% near zero, largest 0.73): in both, a strong connection changes faster than a weak one. Two different rules with that property lose alike; I take it as support, not proof, that the property is the cause there.
3. *The same proportional rule in the cells in between costs nothing measurable* (85.5; -1.1 at the median, no common sign), with the floor at the birth size of the strengths (1/4); at smaller floors seed 1 gave 62.1 and 69.3 against 72.8, so it is the mildest form that is level, and on seed 1 even that was 8.7 below (i) while the reported seeds are not: the development seed misleads again in size.

**Predictions.** S1 held. S2 held (-1.7). S3 held (0.0). S4 failed (1.3). S5 held (-17.7). S6 failed (66.1, 56.5, 71.4 on seed 1). S7 failed (0.8). S8 held (-1.1).

**Not run:** as section 6; a fixed-size rule without fading or with a dead zone about zero mismatch; (ii) at neighbouring steps on the reported seeds; (iv) at other floors on the reported seeds.

## 9. A dated addition, asked by the coordinating session, entered 20 November before its runs

Ten further seeds (1001 to 1010; the rule is quoted in `PLAN_1.md`, section 13) for the two arms within five points of (i) whose difference matters for the reading: A (ii), no division, at 8e-4; and B (iv), the cells' proportional rule, at 1/4. The plain system on the ten seeds is the F1 run of `joint.py`. Reported: the fifteen together.
- U1 (0.6): A (ii) is below (i) on at least nine seeds of fifteen.
- U2 (0.55): B (iv) is within 2 points of (i) at the median of the fifteen paired differences.
