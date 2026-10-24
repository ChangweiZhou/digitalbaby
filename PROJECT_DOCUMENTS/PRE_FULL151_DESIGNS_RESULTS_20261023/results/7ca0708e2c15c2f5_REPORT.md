# Step 11 — paired byte-input attribution on one Full151 value store

**Result (27 September 2026): FE0+fading pairs was worse than FE0 on the frozen final held-out choice contrast in this single-store interface.** Across 32 fresh paired worlds, the old-teaching effect (W−N) was **+21.88 percentage points** for FE0 and **−6.25 points** for FE0+fading pairs. The prespecified difference of those effects, pair minus FE0, was **−28.13 points** (world-paired two-sided 95% Student-*t* interval **−40.0 to −16.3 points**). Both inputs passed the trained-cue feasibility pilot and learned taught cues during confirmation. The independent receipt reducer passed. This rules out the claim that Step 9's loss of transfer *required* its two-store absolute-output architecture; an input-only swap can depress the final choice signal even with a single native store. It does **not** prove that pair features were the sole cause of Step 9's failure.

## What was actually compared

The [frozen design](DESIGN.md) paired FE0 and Step 6's fixed fading-pair front end on the *same* V9.8 E3 byte worlds. Both used the V9.2 parent checkpoint, one native Full151 store, identical 144 old teaching records, one day of rest, 72 disjoint new-cohort records, another day, and the same read-only native value scorer. The fixed V9.2 byte predictor remained present but did not supply the scored value. W and N saw identical bytes, teaching bits, and clocks; only W allowed the old cohort's native writes. Both allowed new-cohort writes. The comparison therefore concerns the encoder's effect on a learned value signal, with no change to Full151 plasticity, teaching dose, or evaluator geometry.

The six final held-out ordered cues were never reinforced. An exact byte-key or canonicalized pair-key table has no answer for them; symbol and position frequencies are balanced, while a one-bit symbol-class rule inferred from old teaching can solve the relation. A positive result would have been limited **E3 within this synthetic relation family**, not arithmetic or general reasoning. The scorer compares two orientations' native values on discarded clones. The learner does **not** emit a byte in this experiment; this is an input-memory diagnostic, not an output-autonomy test.

## Prespecified stages and results

Technical world 12400 passed FE0 probe equality against the old V9.8 implementation, fading-pair KC activation, clone independence, read-only state purity, aligned clocks, predictor parity, and receipt checks. The eight pilot worlds (12401–12408) computed trained cues only. Both encoders passed all frozen E1 feasibility thresholds, so the 32 untouched confirmation worlds (12501–12532) were run without replacement or optional stopping. The source-lock digest is `242af6d4db6b11159f13768082f1b17ef5308e9373b59475f6e1210a74faa24d`.

| Confirmation, world-level means | FE0 | FE0+fading pairs |
|---|---:|---:|
| Old trained W, after old teaching | 94.79% | 93.75% |
| Old trained W, final | 87.50% | 85.42% |
| Old trained W−N, final | +39.06 pp | +32.81 pp |
| New taught W / N, after new teaching | 82.29% / 90.63% | 96.88% / 100.00% |
| Never-taught final W | **71.88%** | **59.38%** |
| Never-taught final N | 50.00% | **65.63%** |
| Never-taught final W−N | **+21.88 pp** [11.47, 32.28] | **−6.25 pp** [−15.12, 2.62] |

The sole primary comparison is `(PAIR W−N)−(FE0 W−N)` on final held-out choice: **−28.13 pp** [−39.99, −16.26]. It was negative in 19 worlds, zero in 10, and positive in 3. The FE0 within-arm effect is a descriptive fresh-world replication of V9.9A's limited signal, not another separately gated primary finding. The pair route maintained old trained-cue competence, so the final choice loss is not a blanket inability of Full151 to learn from that input.

## Mechanistic caution

The pair route's no-old-write control scored 65.63% on final held-out choice, rather than roughly 50% for FE0. Because N still learned the *disjoint* later cohort, this is a warning that the pair representation, later writes, or fixed input geometry can change the held-out baseline. The causal W−N contrast remains valid for this paired run, but it should not be paraphrased as “pair features erase all reusable information.”

The prespecified choice metric thresholds the sign of each of three native value margins per world. A descriptive analysis of those continuous margins gives final held-out W−N **+0.0700** for FE0 (95% interval +0.0510 to +0.0891) and **+0.0639** for fading pairs (−0.0327 to +0.1605). Their paired difference is **−0.0061** (−0.1029 to +0.0906). This analog diagnostic is unresolved; it shows that the decisive choice contrast cannot alone locate whether lost discrimination, baseline shift, or wider margin variability is responsible. It does not override the frozen choice endpoint.

There is one execution shortfall against the prose design: the code saved old-end *trained* scores but did **not** save old-end held-out scores. The primary final endpoint, pilot gates, source lock, and confirmation roster were executed as frozen, but this omitted descriptive checkpoint means this run cannot distinguish a signal absent immediately after old teaching from one lost during delay/new learning. We will not rerun completed trajectories to fill it in. The report does not use historical Step 9/10 worlds as missing factorial cells.

## Audit, scope, and next decision

The [source lock](results/SOURCE_LOCK.json), [pilot](results/PILOT.json), [summary](results/SUMMARY.json), [independent audit](results/AUDIT.json), and per-world receipts under `results/worlds/` make the result reviewable. The auditor regenerated each source fixture without running the learner, checked score arithmetic from all pair margins, write-flag digests, clocks, roster, pilot gate, primary reduction, and rejection of three deliberately corrupted receipts. All 32 science worlds completed with no terminal failure. Their accumulated learner runtime was about **144 seconds**; the result directory is about **348 KB**. A source-lock serialization mistake occurred before any trajectory ran and is preserved as a resolved setup note in `results/RESOLVED_SETUP_FAILURE.json`.

For the current bridge program, FE0+fading pairs should **not** replace FE0 as the sole native value input on the strength of its four-fact recitation results. FE0 remains the working reference for limited held-out relation reuse; the pair route remains useful for separating crowded taught expressions. The next design decision, if further work is wanted, is whether a single bounded shared/private input test can preserve FE0's reusable signal while adding pair-based separation. This run does not authorize an open-ended architecture search, a new memory mechanism, or a claim that the byte encoder+memory problem is fully solved.
