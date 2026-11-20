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
