# Keeping, ninth plan: steps that shrink with a cell's own accumulated use (candidate c, first half), on the plain system. 20 November 2026

Line K1 (`nano/keep`); authorisation and level as in `PLAN_1.md`. Written after the eighth plan's outcome was entered; before any run of this plan. It is run so that every candidate of the brief has a measured answer; I expect it to fail.

**Question.** In the plain system (256 cells, all present from birth, no rule of the earlier plans), if each cell's rate is 1 / (1 + U / k), U its own accumulated use, is more of addition kept with one old line in 32 than the plain system keeps (15.6), at what cost to subtraction?

**The rule (one change from the plain system).** The rate multiplies what the age law of the second plan multiplied (the cell's own steps, the answering cell's step and fading on the strength from it, the return strength's step and fading). Two readings of "use": `use=sig`, U grows at every byte by the size of the signal the cell acted on (the relayed mismatch times its sensitivity, before the rate); `use=out`, U grows by the cell's output. The second half of candidate (c), a common signal, was the subject of the fifth and sixth plans and is not in this one.

**Hypothesis (one).** Cells that are all used alike stiffen alike, so the rule can only trade keeping against learning, as the age law did and as a strength that stiffens with its own use did (`loop_small_case/REPORT.md`, section 9). **Refuted if** some k on seed 1 gives, with one old line in 32, addition of 50% or more and subtraction of 65% or more at the end.

**The learner.** `kage.py` with `H=256 H0=256 use=sig|out tau=k`; compiled; both options are in `check_kage.py`. 43,278 numbers that learn; one accumulated use for each cell.

**Development (seed 1), one old line in 32:** `use=sig`, k in {1, 10, 100, 1,000, 10,000}; `use=out`, k in {1e3, 1e4, 1e5, 1e6}: 9 runs. **Chosen:** the largest value of the smaller of the two scores at the end. The chosen setting goes to the five reported seeds, no old line and one in 32.

**Predictions.**
- I1 (0.80): the hypothesis is not refuted: no k on seed 1 gives 50 and 65.
- I2 (0.70): at the chosen setting, five seeds, one old line in 32: addition at the end below 40% (median).
- I3 (0.85): with no old line: addition at the end 10% or less (median).

**Cost.** 9 development and 10 reported runs of 40 to 60 CPU seconds.

**Not run.** The rule on the joining frame; a rate that is a product of use and a common signal; other measures of use.

## Outcome (entered 20 November, after all runs; `table9_dev.txt`, `table9.txt`, `res/c2_d9_*`, `res/c2_r9_*`)

**Development, seed 1, one old line in 32** (addition after its own phase | addition / subtraction at the end):

| use | k | |
|---|---|---|
| sig | 1, 10, 100 | 0.0 \| 0.0 / 0.0 to 0.6 (nothing is learned) |
| sig | 1,000 | 73.0 \| 3.0 / 59.6 |
| sig | 10,000 | 81.2 \| 7.6 / 80.2 |
| out | 1e3 | 0.0 \| 0.0 / 0.0 |
| out | 1e4 | 40.0 \| 4.0 / 4.2 |
| out | 1e5 | 76.4 \| 3.8 / 53.8 |
| out | 1e6 | 80.2 \| 4.2 / 69.8 |

No k gives 50 and 65: the hypothesis is not refuted. Chosen by the plan's rule: `use=sig`, k = 10,000 (smaller score 7.6).

**Five reported seeds** (73129, 93683, 116089, 406364, 545353; median).

| Arm | Addition after its own phase | Addition at the end | Subtraction at the end |
|---|---|---|---|
| plain, no old line | 84.8 | **1.8** | 86.0 |
| plain, one old line in 32 | 84.8 | **15.6** | 85.8 |
| steps that shrink with use, no old line | 83.4, 82.6, 88.2, 84.6, 84.4; 84.4 | 2.4, 1.8, 1.8, 1.6, 2.2; **1.8** | 82.0, 80.6, 87.0, 78.2, 82.6; 82.0 |
| steps that shrink with use, one in 32 | the same | 9.6, 14.4, 12.8, 14.6, 6.0; **12.8** | 81.8, 79.0, 84.2, 79.6, 81.6; 81.6 |

**Steps that shrink with a cell's own use keep nothing**: 12.8 with one old line in 32, below the plain system's 15.6, at a small cost to subtraction. Stronger settings stop learning before they protect anything, weaker ones do nothing; there is no value between, as with a strength that stiffens (`loop_small_case/REPORT.md`, section 9). It is weaker than the age law of the second plan (33.6), which at least makes the cells differ.

**Predictions.** I1 (0.80) held. I2 (0.70) held: 12.8. I3 (0.85) held: 1.8.
