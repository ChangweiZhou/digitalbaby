# V66 / Iteration 11 — frozen diagnostic protocol

This protocol is written before any new V66 outcome is measured. It refines the supplied V66 design using the already completed, explicitly retrospective `v66_design_review` audit of V65 streams. All 32 fresh streams are retained; there is no redraw, adaptive candidate-pool expansion, or parameter search. The unchanged iteration5/bi parent is instantiated through the V65 full-reference wrapper and observed by the unchanged iteration8 RBF64 head. No smaller model is tested.

## Refinements and interpretation

At one day, the frozen kernel retains 0.991397 of its slow state, 1.74e-11 of alpha fast state and 9.62e-21 of gamma fast state. Slow-footprint cosine is the primary directional diagnostic; averaging the three compartments is descriptive. A footprint is the full four-association checkpoint minus an otherwise identical history with only that focal association's writes disabled. It includes mediated effects on subsequent focal writes; it is not an isolated synapse label.

Realized write-vector features are post-write measurements, though available without probing the focal output. They predict consequences of a realized write, not its generation in advance. A separate cue-only feature block assesses pre-write anatomical/sensory information. Clipping-aware gradients and exact support interventions audit expression identities and are excluded from the primary predictor blocks. Equal adaptation makes actual- and reader-margin *changes* algebraically equal: these are not independent replications. Off-support endpoint interventions address expression, not whether globally pooled feedback generates writes.

The experiment distinguishes causal write effects inside this fixed effective model. It cannot identify a new biological mechanism, establish a minimal fly brain, or validate a physiology repair. The recent Huang source audit found unresolved raw-to-summary transformations and changed published means; this campaign neither refits those data nor claims a new physiology pass.

## Stage 0

Before the fresh campaign, compare the full wrapper with the original parent, reproduce a hash-referenced V65 load-12 fixture, verify exact save/restore and continued evolution, clone-only evaluation, matched write/no-write adaptation and timing, and floating-point-zero no-learning memory. Capture parent and diagnostic-source SHA-256s. The campaign must preserve these hashes and pass numerical tolerances of 1e-10 (exact state reproduction where applicable).

## Stage A: causal atlas

Six bouts per association: A 30 s, rest 135 s, B 30 s, rest 135 s, repeated six times. Sixteen new 96-association streams; orientations balanced within successive 12-association blocks. Four focal associations define a checkpoint and eligibility (both frozen reader and actual-memory margin >= their existing thresholds). Every later association gets write and no-write twins from its identical pre-state. Only the write twin continues the main trajectory. Horizons 0 and 86,400 s and all probes use clones. Keep every focal row, including ineligible ones.

Primary effect E is signed old-association actual-memory margin in W minus N. Record compartment footprint cosines, normalized absolute slow overlap, unsigned cue-support overlap, signed cue-code alignment, original teaching-signature alignment, compartment/global write and state norms, load, cumulative write norm, no-write margin, clipping, and exact/gradient expression reconstructions. Record checkpoint and load-96 delayed focal retention against a matched no-write-later trajectory.

Descriptive per-stream Spearman correlations and stream-bootstrap intervals are secondary. Fixed ridge (alpha 1, standardized and mean-imputed on training streams only) is evaluated leave-one-stream-out on eligible delayed rows. Feature blocks: CUE (unsigned support overlap, signed KC cosine), LOCAL (CUE + three footprint cosines, absolute slow overlap, teaching cosine), GLOBAL (three write norms, three pre-state norms, cumulative write norm, load, no-write margin, clipping), COMBINED (LOCAL + GLOBAL). Report R2, MAE and non-deadband sign accuracy. LOCAL is a realized-write diagnostic; CUE is available before the later association. No coefficient/hyperparameter selection on held-out streams.

## Stage B: prospective role interventions

Sixteen independent fresh four-focal checkpoints. Each gets a fixed pool of 256 fresh cue pairs, simulated at both consequence orientations from the same checkpoint. During selection, compute only write vectors, teaching magnitudes, sensory signatures and pre-state quantities; never probe focal consequences. A pair qualifies when slow-footprint cosines straddle zero and total, each compartment, and teaching-payload RMS norm ratios are all within [0.8, 1.25]. Zero/zero norms count as ratio one. Rank by the smaller absolute slow cosine, descending; candidate index breaks ties. Select up to 24 per focal, never relax calipers, and report shortfalls. Effects are measured only after the selection list is saved.

**Single primary contrast:** at 24 h, mean E(REINFORCE)-E(OPPOSE), averaged over selected candidates, eligible focals, then streams. The primary uncertainty interval resamples the 16 independent streams, not individual pairs. A positive interval is evidence for directional local interference under these controls. Report all individual candidate caliper diagnostics and focal-level counts. Stage A and Stage B use disjoint cues and seeds.

Secondary sequential challenge: replay the same selected cue pairs in candidate-index order under checkpoint-selected OPPOSE or REINFORCE roles; assess nested lengths 8, 16 and 24 at both horizons. A matched no-write branch follows the same OPPOSE sensory/consequence schedule. Also use a LOW_ABS_ALIGNMENT sequence of distinct candidate pairs chosen by smallest absolute initial slow cosine (not called a proven neutral intervention). Record realized cosines as states diverge; initial matching is not guaranteed thereafter. Focals with fewer than 24 qualifying pairs are excluded only from complete 24-pair sequences, with counts retained.

Exploratory magnitude challenge: candidate orientations with |slow cosine| <= .05; high/low pairs require same role, cosine difference <= .02, teacher RMS ratio within [0.8,1.25], active-KC count difference <=5%, pre-cue naive output norm ratio within [0.8,1.25], and write norm ratio >=1.5. Rank by norm ratio, select up to eight disjoint pairs, no caliper relaxation. This cannot independently establish a global-norm cause: cosine-near-zero is not output-gradient orthogonality and local coordinate distributions remain unmatched. Report unsupported/shortfall if controls cannot be constructed.

## Stage C: durable revision

Use the same independent Stage-B four-focal checkpoints; all four focals, including originally ineligible ones, remain in the tables. Reverse each focal separately at doses 0, 6, 12 bouts, pad each to the same 3,960-second endpoint, then measure at 0 and one day. Padding is rest, so dose comparisons intentionally vary sensory exposure; matched no-write twins at each dose isolate the additional write effect. Dose zero is rest-only. Per-bout writes receive the same matched counterfactual and slow-footprint analysis. Durable success requires an eligible old memory at the checkpoint and both new reader and actual-memory criteria at the delayed endpoint. This is a four-association revision challenge, not the different historical serial-probe/load-12 benchmark.

## Decision and packaging

If Stage 0 fails, stop as diagnostic_invalid. If held-out Stage-A slow alignment is reproducible and the primary Stage-B interval is positive, label signed local interference supported *within this model*. Do not label global feedback disproved. Otherwise report mixed/unresolved/negative with explicit caliper and eligibility coverage. No learning model is promoted. Deliver aggregate tables, decisive figures, diagnostic code/configuration and checks; reference unchanged parent assets and exclude raw per-stream outputs, checkpoints, caches and datasets from the <10 MB bundle.
