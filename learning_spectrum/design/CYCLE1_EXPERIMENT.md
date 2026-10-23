# Cycle 1: fixed causal geometry and reconstruction

Written before this cycle's test. This is an isolated program; T and Package A are unchanged. Historical response-mechanism histories may be read as development data. Package A behavioral outcomes may not be opened.

## Candidate A, frozen equation (no parameter search)
For every cue byte, let a=R_TABLE[byte] and let r be the recent receptor trace after elapsed-time decay, before adding a. Let mu_a and mu_r be elementwise baselines persisting across cues, born zero. Use a_tilde=a-mu_a(previous), r_tilde=r-mu_r(previous). Decay eligibility, then add outer(r_tilde,a_tilde). Only afterwards set mu_a += (a-mu_a)/16 and mu_r += (r-mu_r)/16, then r += a. Reset r and eligibility at each explicitly supplied record boundary, preserving baselines. Exclude teacher/outcome and newline bytes, exactly as original J's cue-only sensory contract. No byte identity beyond receptor input, task ID, answer, digit location, evaluator grid, or future samples enter this rule. Signed eligibility is retained. Recent and eligibility tau=10 seconds; eta=.25, bank tau=86400 seconds, bound=16, gain=.4237781016501581 all unchanged from J. Normalization max(1,||phi||²), target +/-1 unchanged. The beta=1/16 is a declared engineering choice borrowed from the existing exposure EMA, not fitted or claimed theoretically unique.

This one candidate remains fixed through all three cycles. Later cycles may fix technical errors only, with documented audit and rerun. A failing gate is not permission to tune beta, gain, amplitude, decay, target, or representation.

## Frozen preflight gate definitions
At each recorded history snapshot evaluate a complete old/new cue grid by isolated clones, so evaluator queries cannot update the continuing history. For a panel F (16 rows), z_i=phi_i/sqrt(max(1,||phi_i||²)); K=ZZ^T/16. P_int=(I4-11^T/4) tensor (I4-11^T/4). Report all 9 eigenvalues of Q_int^T K Q_int, all cross-subspace coupling, and zero modes. Gate on the MINIMUM, not largest/average eigenvalue. Require min nu>=1-(.8)^(1/96), divided by .25 (about .00929), at old_end and new_end histories. This compression is a diagnostic, not a finite-horizon theorem for nonstationary features.

The exact ordered update and elapsed-time decay will be computed in cycle 2, including interference. Necessary finite-horizon gates: minimum symmetric task-target formation operator >=.20 at old_end; >=.02 at final after two one-day rests and new teaching. The final threshold is less than .20 because even perfectly learned isolated memory retains exp(-2) across two days; .02 demands a substantive fraction of that ceiling. Also report singular values, signed target-specific projections, and nonnormal coupling. Clipping invalidates a linear forecast unless explicitly simulated; no ignored clipping allowance.

Cycle 2 will compare query feature geometry against native FE0 on an unlabeled 30-cue directed relation panel over six symbols. Both kernel matrices are centered and trace-normalized before measuring mass in the incidence shared-symbol subspace and cross-cue kernel alignment. Require candidate shared-subspace mass >=50% of FE0 mass AND off-diagonal centered-kernel cosine >=.50. This rules out an exact-address diagonal kernel passing by full rank. These criteria are operational preservation tests, not a proof of relation transfer. Exact definitions must be independently audited before the cycle 2 test.

Full launch additionally requires quantitative source/history-derived final margin forecasts with the unchanged native route, a prespecified prospective falsification tolerance, all causal/test/resource checks, and three completed cycles. No forecast means NO QUANTITATIVE THEORY PREDICTION and blocks full run. No unopened Package A behavioral data will inform predictions; all its arms receive NO QUANTITATIVE THEORY PREDICTION unless independently derivable from source.

## This cycle's small test
1. Verify immutable imported source hashes against source lock.
2. Rebuild original J features on known cue grids and energy decomposition; compare quoted quantities without assuming their units/checkpoint definitions.
3. Replay six-bout cue-only input histories from development worlds 300001 and 300002 for the single candidate; save baselines, feature digests, spectrum and common/main/interaction energy at old_end/new_end.
4. Verify original J source equivalence, causality (future suffix cannot affect prefix), absence of labels from API, and pre-update EMA semantics.
5. Recompute historical H quoted accuracy and fix/break counts from all 32 already completed response histories; label evaluator panel centering an oracle diagnostic, never an online implementation.

No canonical Full151 life or fresh-world behavioral outcome is generated in cycle 1. Cap this cycle at 120 CPU seconds, one thread, 1 GiB RSS, 20 MB outputs. Stop safely on violation. Keep negative results. Cycle outcome determines whether the full-run gate fails, never whether to change candidate.
