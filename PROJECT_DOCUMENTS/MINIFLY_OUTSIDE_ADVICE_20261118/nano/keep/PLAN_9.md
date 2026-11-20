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
