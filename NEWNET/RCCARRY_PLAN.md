# RCCARRY plan: can rule cells form the carry? (written before the runs, 23 November 2026)

Helper branch `claude/newnet-rccarry-sei232`, from `claude/magical-albattani-sei232` @ d0f0a9da. Development level, one machine,
OMP_NUM_THREADS=1. No gate, seal or outside review is claimed. No existing .py is edited.

## Question and hypothesis
Column addition without the carry written (t24 fmt=col: `a b s` per column, lowest digit first) needs a hidden quantity, the carry.
No mechanism so far learned it: net12 lagro=1 relay=1 new-length whole lines 0.031 (ADD_REPORT.md), error-driven recruitment 0.033
(IDEA4_REPORT.md), storage (back-off table) 0.376. Rule cells (net13) are persistent state cells set by events and learned by
competition: the form a carry has. **Hypothesis:** with rule cells (net13; and variants where the rule cells also read match cells or
pair cells - only their input set changes) the network forms a carry-like state and generalises to new lengths.

## Criterion (fixed now)
- **Pass:** fmt=col, full size (struct K8, L=16, N0 = N1 = 8000), seed 1: P1 new-length whole lines right >= **0.30**.
  If it passes, run seeds 2-3 of the passing arm.
- **Carry-like state:** reported for every rule-cell run - the Pearson correlation of each rule cell's normalised activity (at the
  moment s_i is predicted, columns i >= 1) with the true carry into column i; the max |corr|; the number of cells with |corr| > 0.2;
  the held-out purity of the winner at s_(i-1) as a carry code (against the base rate); a held-out linear read of the carry from the
  rule-cell state (a diagnostic probe, not part of the net, not a result about the net). A carry-like state would show max |corr|
  >= 0.5 or held-out purity >= base rate + 0.15. These numbers are descriptive, not the pass criterion.
- **colc not worse:** net13 fmt=colc P1 new-length whole lines not below the net12 lagro=1 relay=1 value 0.739 by more than 0.03.
- Comparisons: net12 lagro=1 relay=1 (col 0.466 / 0.031, colc 0.919 / 0.739, old / new lines) and storage (col 0.799 / 0.376,
  colc 0.967 / 0.867), from add_runs/res.jsonl; small-size references are run here (net12 lagro=1 relay=1, same small settings).

## Arms
Code: `net13rcc.py` (subclass of net13: `rc_match=K` / `rc_pairs=NP` add match / pair cells to the rule cells' inputs only;
`rc_ro=1` the variation below), `t24_add_rcc.py` (t24_add.py importing the net by module name, plus the read-only carry diagnostic).
Checked before the runs: net13rcc with defaults and t24_add_rcc with net13 give byte-identical results to t24_add.py net=13
(L=2, 300 lines).
1. Small (struct K8 L=4 N0=N1=4000, seed 1), fmt=col: net12 lagro=1 relay=1 (reference); net13 (defaults rc=64 rc_k=3 rc_tau=8
   rc_vig=0.4); net13rcc rc_match=4; net13rcc rc_pairs=32768. fmt=colc: net12 reference; net13.
2. Full (L=16, N0=N1=8000, seed 1): net13 col, net13 colc, net13rcc rc_match=4 col, net13rcc rc_pairs=32768 col.
3. **The one variation** (decided by a check made before this plan, and fixed here): persistence. In the 300-line check, the rule
   cell that won at s_(i-1) was still active when s_i was predicted at every s byte (persistence 1.00: with tau = 8 bytes and k = 3,
   the winners of the last three bytes stay on). So persistence over one column is not missing, and the variation is not a
   persistence change. What is missing is a path: in net13 the rule cells only set the relay's gate (which lag to copy), and a carry
   adds one to a + b - nothing a copy gate can produce. **Variation rc_ro=1:** the rule cells' activity is also an input of the direct
   readout, learned by the readout's own delta rule (the same rule as for the lag cells). It is run small (col) first; at full size
   (col) only if it gains >= 0.02 new-length lines over small net13 col, else NOT_RUN.
4. Seeds 2-3: only for an arm that passes the criterion.

## Expectation (stated before running)
Likely fail. The rule cells cluster the 8-lag window by cosine (vigilance 0.4: >= 4 of 8 (lag, byte) pairs shared); the carry is a
function of 3 bytes (a, b, s at lags 2-4 when s_(i-1) is read) with 1,000 combinations, and of no single byte; 64 cells clustering the
window by overlap should not line up with it. Even a perfect carry cell used through second-level conjunctions reached only 0.03-0.15
(IDEA4 oracle). Falsifier of this expectation: P1 new >= 0.30 or max |corr| >= 0.5.
