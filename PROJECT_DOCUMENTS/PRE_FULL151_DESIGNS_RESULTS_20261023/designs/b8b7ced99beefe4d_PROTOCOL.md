# V78 / iteration23 — timing, state relaxation, and slow commitment

This protocol is frozen before evaluation. Sixteen new seed histories (7801001–7801016); 240 unique 8-of-88 PN patterns each; exclude all V76 prior banks plus V76/V77 and original interface patterns. No fitting, arm selection, swarm, or new connectome compression. Anatomical parent: full 5,177-KC model and frozen RBF64 reader. The V76 geometric normalizer, 2 Hz objective, native EVENT convention and time constants stay fixed. Local physiological RPE is not claimed.

## Refinement of the suggestions

The two supplied documents are proposals, not execution instructions. V77 establishes a timing effect in this simulator, not a biological commit barrier. The model contains alpha-fast, gamma-fast, adaptation and slow decay; it has no mitochondrial readiness state. A spacing curve cannot identify an absent molecular process. Counterfactual state freezing is diagnostic, not a realizable fly.

Fewer function calls need not mean less interference: slow writes add linearly. A faithful batching control must preserve the update vectors and compensate their different ages. Comparing six vs twelve online writes alone confounds time, information, magnitude and feedback. We therefore test exact vector replay first, plus a separate half-dose factorial and an online batch rule. Do not interpret injection L1 after cancellation as equal source dose; report source-vector L1 and injection L1 separately.

## Common history and outcomes

Acquire 96 associations, six 330 s bouts per association; save native states at 48 and 96. Each branch starts at that same state. Focal association is middle or newest. STABLE repeats its old label; REV uses the opposite label; ANOMALY1 has one wrong label then eleven old labels. The focal association is excluded from old-memory guards. All challenge arms get 12 physical A30/rest135/B30/rest135 pairs. Labels are revealed current outcomes; no clean labels, task IDs or evaluator outcomes enter the learner.

LAST is the end of each arm's twelfth observation. COMMON pads to 23,760 s after the prestate (12×330 + 11×1800). Both immediate and +86,400 s probes are immutable. Unlike V77, this common clock also covers the 1800 s sensitivity. MIXED starts at COMMON and uses 12 fresh associations × six bouts **with the same native REF learning rule for every arm**. This isolates challenge effects; it is not a repeat of V77's arm-specific future learner. A matched no-write branch receives all sensory events and all specified state interventions, with no gamma or alpha plasticity, and has identical elapsed time/adaptation within each treatment. Partial adaptation freezes are allowed to differ between treatments.

Primary choice is the existing conjunction of actual memory margin ≥1 Hz and fixed-reader margin ≥1 Hz. Report target success, old lost/total old, old lost/initially-correct old, actual mean/worst old damage vs no-write, and new suffix success separately. Fixed no-write branches distinguish relaxation from additional writes. Never count each old association as an independent history.

## A. Native state interventions (REV; two loads × two focal ages)

At every extra 900 s gap perform a 2×2×2 factorial: F = alpha-fast decay, G = gamma-fast decay, A = adaptation recovery. `N_000` through `N_111` describe F/G/A flags. All eight arms advance clocks and naturally decay slow state; disabled state variables are restored after the gap. Normal 330 s bout dynamics are identical. Add `N_FROZEN` (also restores slow state), `N_DENSE` (zero extra gap and matched end padding), and natural 300/1800 s gaps. These 12 conditions identify joint, isolated and leave-one-out effects without assuming additivity. DENSE and FULL900 also run STABLE/ANOMALY1 to anchor B/C safety.

## B. Slow-write timing (all three cases)

`M12`: V77 TOTAL_MARGIN, dense, native gamma, all alpha correction committed to slow after each pair. `M_SPACED`: same, 900 s extra gap. Save the twelve requests u_t from M12 for evaluator-only replay.

`BATCH6`: replay the same twelve u_t; commit every even bout. Queue q <- exp(-330/tau_s)q + u_t, then s <- s+q; q <-0. The first request is aged exactly one bout before release. No new cue/outcome observation at commit. `HALF12` replays u_t/2 after each pair; `HALF_BATCH6` batches those same half vectors. M12/BATCH6 and HALF12/HALF_BATCH6 must match final slow state to 1e-10 absolute coordinate tolerance, although intervening gamma feedback may differ. These are privileged open-loop controls, not deployable learners.

`ONLINE6`: computes its own MARGIN request at each observed pair, accumulates/decays it and commits every second pair. This deterministic pair counter plus 5,177 float64 queue is an engineering temporal rule, not a local physiological clock. Pending requests reflect its own state. Additional mutable queue is 41,416 bytes (plus scalar bookkeeping); no persistent memory IDs. Challenge stops with empty queue. All twelve labels still enter requests; this is delayed commitment, not selectively discarding contradictory evidence. There is no end-of-training extra retrieval or replay.

## C. Finite alpha-fast reference (all cases, dense only)

`P0` = TOTAL_POINT; `P900` replaces only alpha-fast with exp(-900/tau_alpha_fast) times its current value in an evaluator clone, with the SAME current adapted activity, gamma and slow state. `PINF` zeros alpha-fast (V77 NOFAST_POINT). `M900` applies the same finite attenuation with MARGIN; M12 is its current-reference control. This is explicitly a component-specific reference diagnostic, **not a prediction of the full future brain state** and not anatomy-validated implementation. No other horizons, thresholds or tau search.

## D. Trace fate, without claiming engram identity

For every COMMON/MIXED +day endpoint, save additive pre-alpha-clipping slow, alpha-fast and gamma-relay contributions. Record whether an old lost choice is recovered by removing alpha clipping or by zeroing both fast components in an immutable clone; no reader refitting. Report positive slow projection and reduction vs its no-write projection. These predicates overlap. They measure current support and expression constraints, not physical erasure vs newly added opposing engrams: superposed pooled slow weights do not uniquely identify ownership. Do not report a mutually exclusive biological taxonomy.

## Inference fixed before execution

One family of 15 two-sided exact paired sign-flip tests (16 history units, average both loads/ages unless stated), Holm correction; 5,000 fixed-seed percentile bootstrap intervals descriptive, not simultaneous. Native REV COMMON +day target-choice contrasts: FULL900−DENSE, FULL900−000; F-only−000, G-only−000, A-only−000; FULL900−no-F, FULL900−no-G, FULL900−no-A (8). B contrasts BATCH6−M12, HALF12−M12, ONLINE6−M12 each for REV COMMON +day target choice and old loss (6). C P900−P0 STABLE newest COMMON +day target choice (1). Material gain for choice =10 percentage points. All other curves, interactions, dose and fate results are descriptive; no post hoc best-horizon selection.

For M12, M_SPACED and ONLINE6 apply the unchanged 64 V77 diagnostic gates vs native same spacing: COMMON/MIXED +day REV gain mean≥10 pp and lower CI>0; for each load/case old accuracy lower CI≥−2 pp, old loss upper≤2 pp, added no-write worst damage upper≤0.05 Hz, absolute per-history worst direct margin deficit vs reference upper≤0.05 Hz; STABLE/ANOMALY target lower≥−2 pp; MIXED new accuracy lower≥−2 pp. Report failures even when primary effects pass. These gates only assess diagnostic qualification. Own-history and physiological validation are outside this bounded experiment; **no core promotion in V78**.

## Software acceptance and stopping

Prepare tests independent fixtures, verifies immutable parent hashes, then locks scientific sources/config/protocol/inputs/proposals. Two processes, bounded worker timeout, atomic receipt directories, resume only hash-valid complete jobs. Stop on failed checks, nonfinite states, altered hashes or vector-transport error; never silently change a locked arm. Independent report audit reconstructs all endpoints from vectors, recomputes all contrasts/gates, and replays a bounded first-history subset from native prestate using direct event equations and independent batching. Full raw outputs remain external to a compact <10 MB result ZIP. No fitted final model; config declares every diagnostic arm and parent references.
