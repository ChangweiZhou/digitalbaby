# V79E — concurrent bounded genetic search for autonomous persistent learning

Status: prepared for user launch, alongside the unchanged V79 diagnostic.
The user clarified that BOTH investigations were requested. The proposal's
exclusion of evolution is document content, not a restriction overriding that
request. V79E is a separately identified extension; do not rewrite the frozen
V79 experiment or wait for its results before beginning this search.

## Objective and phenotype

Run a genuine competing population of autonomous MiniFly allocation rules.
Preserve the full 5,177-KC anatomy, native sensory encoder, EVENT equations,
native raw alpha teaching, gamma equations, adaptation/decay constants and frozen
reader. Evolve four immutable coefficients, not neuron count or topology:

`f_i = sigmoid(logit(f0) + b0 + b_adapt*(2*a_i_pre-1)`
`              + b_fast*tanh(alpha_fast_i_pre/1.0)*sign(u_i)`
`              + b_slow*tanh(slow_i_pre/1.0)*sign(u_i))`.

Here u is the current native raw total alpha increment before allocation and
decay. Commit slow=f_i*u and alpha-fast=(1-f_i)*u before their respective native
decays. The current event's native gamma increment and adaptation update remain
unchanged at that same prestate. Future state and teaching may differ; do not
claim identical lifetime gamma trajectories or identical future expenditure.

Only a coordinate's own pre-event adaptation, alpha-fast/slow state and local
teacher sign enter allocation. No case identity, old cue list, future reward,
global error, allocation normalization, memory bank, replay or evaluator budget
enters the phenotype. The four genes lie in [-2,2], rounded to six decimals for
canonical identity; all-zero genes invoke the native code path exactly.
The 1.0 state scales are prespecified model units, not fitted biological values.
Tanh saturation is part of this bounded search family, not a discovered law.

Learned state remains 165,688 bytes including clock/counters. Four immutable
genotype constants and offline search metadata are reported separately; the
optimizer's population is not hidden persistent memory available to a fly.
This is evolution of four coefficients within one local rule family. Failure
does not reject evolutionary architecture search or all other learning rules.
A b0-only winner could be constant allocation retuning; it would not by itself
establish an adaptation-dependent mechanism.

## Inputs, lifetime tasks and measurement

Use four fixed development histories (7911001–7911004) for every generation.
Each comprises 240 unique 8-of-88 PN patterns, balanced binary roles in blocks
of 12. Exclude all prior V76–V79/interface banks, and all other new development
and confirmation patterns. Hash the complete bank before any evaluation.

Each genotype builds its OWN native-size foundation: 48 associations, six
330-s bouts per association. Challenge copies use newest STABLE, REV, ANOMALY1,
ANOMALY2 and NEW. REV flips all 12 labels. Anomalies flip the first one/two then
give correct teaching; these endpoints measure recovery after contradictions,
not immediate immunity to contradictions. NEW is unused pair96, six bouts.
All other cases use12 bouts. No inter-bout adaptation surgery is used.

Pad each challenge to a common23,760-s clock. Probe COMMON at0/day, then teach
12 further unused pairs108–119 for six bouts each with that genotype's OWN rule.
Probe MIXED at0/day. This continuation tests deployability of the phenotype,
unlike V79's fixed-source transplantation. All probes are immutable copies using
the inherited frozen observer and1-Hz joint choice criterion.

The matched no-challenge control receives identical sensory timing with all
challenge gamma/alpha learning disabled, then learns the same MIXED suffix using
the same genotype. Old damage is relative to that control, not a global oracle.
Old eligibility uses the genotype's own initial correct associations, excluding
the focal old association; NEW includes all old associations. Record lost
counts/denominators, unconditional old accuracy, worst actual/joint margin harm,
foundation capacity and subsequent new learning. Undefined conditional loss is
unavailable and disqualifies the affected development eligibility/confirmation
guard; do not delete that history or interpret zero retained memories as safe.

Resource accounting uses raw pre-decay alpha total, slow and alpha-fast L1,
plus gamma L1. Foundation, challenge and MIXED expenditures are logged separately.
The scored lifetime budget is the mean actual foundation+challenge+MIXED path
over cases. Counterfactual baseline computation is excluded. Instantaneous split
conservation does not imply conserved future total dose; both are measured.

## Selection, reproduction and caching

Population16, eight generations (indices0–7). Generation0 contains REF and15
uniform random coefficient vectors in[-1,1], with a fixed RNG seed. Thereafter:

1. Retain the fixed REF and the best nonreference elite from the previous generation.
2. Generate14 more individuals by size-three fitness tournaments, uniform
   coordinate crossover and independent Gaussian mutation, or15% random
   injection from the full[-2,2] bounds.
3. Per-gene mutation probability0.5, force at least one mutated gene, clip to
   frozen bounds. Sigma schedule is frozen in config; reproduction of generation1
   uses0.5, then0.4/0.3/0.25/0.2/0.15/0.1 through generation7.
4. Require distinct canonical genomes within a generation. Reuse an existing
   complete hash-verified evaluation if exactly the same genome recurs.

One candidate evaluation job covers all four development histories. The search
has128 population slots; because REF/elites are reused, this is not128 distinct
evaluations. Generation plans freeze genomes, ancestry, RNG-derived construction
and parent receipt hashes before dispatch. Subsequent populations depend on
measured fitness, not merely a predetermined random sweep. A partial generation
is never used to select parents or authorize confirmation.

Development fitness balances eight choice components equally: foundation/day,
MIXED new-learning/day, all-old/day accuracy, and five case target choices,
each case averaged across COMMON/day and MIXED/day. Average loads within history
(confirmation only), then histories. Clipped target joint margin[-2,2] is only
a deterministic tiebreaker, followed by canonical ID.

Rank feasible individuals first. Development constraints, relative to REF on
identical inputs, are pooled across assigned histories/cases/phases:

- Slow expenditure and total alpha expenditure each at most1.05×REF.
- Foundation and MIXED new accuracy lose at most2 percentage points.
- STABLE target choice loses at most5 points; all-old accuracy at most1 point.
- Conditional old-choice loss rises at most1 point; mean of per-endpoint worst
  actual-margin damage rises at most0.05Hz.

Infeasible individuals rank by sum of normalized positive constraint violations,
then choice score/margin. This supplies a search direction without promoting an
unsafe individual. Undefined metrics receive a large finite violation; they
cannot become eligible. A zero reference budget admits only zero candidate
expenditure, without denominator epsilons. Numerical feasibility tolerance is
1e-10 on the summed normalized violation. Always report feasible population
counts so an empty useful search envelope is distinguishable from no gain.

## Champion lock and untouched confirmation

After all eight development generations, select the best ELIGIBLE genome across
all their records. If REF wins, explicitly report no nonreference winner. Never
take a least-infeasible individual to confirmation. If even REF is ineligible,
stop for an evaluator/undefined-endpoint diagnosis, not a fabricated success.

Write champion.json with exact genes, developmental score, all generation-score
hashes and source lock. Recompute and verify that selection before held-out
dispatch. Only then evaluate this one locked champion and REF on eight new
histories7913001–7913008, at BOTH48 and96 own-foundation loads. No runner-up
substitution, fitness feedback, parameter adjustment or restart using these
outcomes. Interrupted searches resume the same plans; a time limit is not a
reason to peek at confirmation early.

The confirmatory family has77 prespecified checks. Paired differences are
computed by fresh history:

- One pooled eight-component macro-choice gain: mean at least5pp and positive
  family lower bound.
- Two foundation noninferiority checks, one per load.
- Ten MIXED-new checks (each load×case); four STABLE checks (load×phase).
- Sixty old-memory checks: old accuracy, conditional loss and worst damage in
  each load×case×phase. Limits equal the corresponding development margins.

Use10,000 paired history bootstrap draws (seed7919001), report95% percentile
intervals and family intervals at quantiles0.05/(2×77) and1 minus that. These
are conservative approximate bootstrap intervals with only eight histories,
not exact finite-sample guarantees. Missing constituents prevent a passed
check. Also require raw slow AND raw alpha ratios≤1.05 for every held-out
history/load/case/phase day endpoint, not just pooled expenditure.

All checks must pass for the label HELD_OUT_BEHAVIORAL_CANDIDATE. Otherwise
report failures and their strata. No automatic biological/persistent-core
promotion occurs: original temporal physiology is not retested/refitted here.
Keep the accepted parent unchanged. Confirmation is finite and synthetic, and
the genome family contains constant retuning as well as state-dependent terms.

## Independent execution with the V79 diagnostic

The combined launcher lives in sibling `iteration24_joint`. It starts V79 and
V79E immediately under one wall-time/worker budget. Default total2 workers means
1 diagnostic +1 evolution; optional total4 means2+2. Once diagnostics complete
and verify with their lock released, their capacity transfers to evolution.
Evolution receives a launch-token-bound capacity file; at most4 processes.
If evolution finishes first, V79 retains its existing frozen pool.

Default launch budget8 hours. Stop dispatching at deadline; existing jobs drain
to a per-job40-minute hard timeout. Log failures; retain incomplete attempts;
resume only complete, hash-verified receipts. Duplicate independent or joint
starts are blocked. No new population is declared successful after a partial
run. Code, config, protocol, input hash and inherited parents freeze before use;
dynamic plans and results have separate reproducible receipts.

Preparation runs unit, whole-assay and fabricated-table software fixtures only.
The operational suite uses isolated tiny fake evaluators and never reads fly
inputs. The user selected script-only delivery; no campaign is launched in this
turn. A later report bundles new tables/config/figures/checks and parent manifests
under10MB, excluding raw per-genotype/seed output, source data and caches.
