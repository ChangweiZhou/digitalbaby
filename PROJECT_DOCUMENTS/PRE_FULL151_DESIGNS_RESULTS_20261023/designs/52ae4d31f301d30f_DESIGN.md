# V81E — relaxed-anatomy multi-objective evolution (PREPARED, NOT RUN)

**Status** prepared and self-tested; 0 scientific evaluations started.
**Launch** `./RUN_V81E.command`, or
`_omnibus10_env/bin/python runner.py start --workers 4 --max-hours N`.

## What this relaxes

Every previous evolution run held anatomy fixed and searched only the fast/slow
allocation of an alpha write. V81E opens the model itself. All 17 genes are
neutral at 0, and the all-zero genome reproduces the inherited model bit-for-bit
(certificate `neutral_phenotype_is_reference`).

| domain | genes | effect at the box edge |
|---|---|---|
| plasticity | `b0`, `b_adapt`, `b_fast`, `b_slow`, `g_alpha_gain` | per-coordinate fast/slow split; total alpha write x0.71-1.41 |
| connectivity | `g_teach_scale`, `g_readout` | DAN->KC and KC->MBON magnitudes x0.71-1.41 / x0.76-1.32 |
| connectivity | `g_support` | grants alpha teaching to the 544 KCs that have alpha readout but no direct DAN route (1,226 -> 1,770 writable) |
| connectivity | `g_pnkc_weight`, `g_pnkc_rewire` | lognormal jitter on PN->KC weights; degree-preserving permutation of up to 50% of PN rows' targets |
| recurrent | `g_feedback`, `g_mbon` | gamma->alpha feedback matrix and MBON->MBON matrix x0.5-2 |
| potentials | `g_k0` | baseline offset vector x0.81-1.23 |
| timescales | `g_fast_tau`, `g_slow_tau`, `g_fraction` | fast and slow lifetimes x0.5-2; native split logit-shifted +/-1 |
| coding | `g_sparsity` | KC active fraction 3.5%-7.1% (5% at neutral) |

`g_sparsity` is bounded well inside the 10% that V74 showed drives near-universal
output saturation.

## Why the search is configured this way

**Multi-objective, not feasible-first.** NSGA-II over (macro_choice up, tail_rise down,
old-accuracy loss down, slow expenditure down). V79E ranked infeasible individuals by
violation before score and spent 89% of its budget on constraint repair; 9 of its 14
feasible points were the reference itself.

**Mutation rate 0.15, not 0.5.** V80E run A perturbed about half of 6 genes per
offspring and violated *every* constraint simultaneously - 28-80% of infeasible
candidates per constraint, at median normalised excesses of 10-46. With 17 genes a 0.5
rate would touch ~9 at once. At 0.15 an offspring changes 2-3 genes, so the search can
move along low-dimensional faces where feasibility is attainable.

**Narrow allocation box.** `b0` had rank correlation **+0.345** with total violation in
run A, the largest of any gene, consistent with iteration 6's equal-allocation result.
`g_support` was **-0.011** - essentially uncorrelated with violation. Only 1 of 720 run-A
candidates had all |b*| < 0.35, so the relaxation axes were never actually explored.

**64 x 50 = 3,200 evaluations**, 6 development and 8 untouched confirmation histories.

## What a winner here would and would not be

A qualifying candidate would **not** be a connectome-constrained fly. Relaxing PN->KC
topology, recurrent gains, baseline potentials and kernel timescales voids the inherited
physiology scorecard, which was fitted at the reference values. Any result is a claim
about what this class of model can do, not about *Drosophila*.

The eight confirmation histories are the only protection against 3,200 evaluations of
overfitting, and they are never used for selection. Read the held-out result first and
the development front second.

## Certificates

59 selftests pass, including 12 written for this iteration: neutral phenotype equals the
reference, neutral encoding matches the native encoder bit-for-bit, each gene moves its
own target array, `rewire_preserves_degree` (row pointers and nnz unchanged), and
`weight_jitter_preserves_support` (column indices unchanged, values changed).

## Known limitation

Because PN->KC weights, topology and sparsity are now genes, the encoded cue bank is
candidate-specific and is re-encoded per candidate rather than precomputed. This costs
run time and means two candidates see different KC codes for the same PN pattern - which
is the point of the experiment, but it does mean `macro_choice` differences partly
reflect coding changes rather than plasticity changes. The `pheno_*` audit columns
recorded with every candidate allow that to be separated afterwards.
