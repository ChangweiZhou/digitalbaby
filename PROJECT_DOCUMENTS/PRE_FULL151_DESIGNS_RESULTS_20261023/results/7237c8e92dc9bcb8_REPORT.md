# MiniFly V9.10D — the frozen shared head fits taught cycles but does not transfer them

**Outcome:** The V9.10C online shared KC head learned the 18 directly reinforced cyclic relations almost perfectly, yet did not use that experience to answer the nine never-reinforced relations. Across 64 untouched worlds, its first held-out choice was **51.04%** (world-level 95% Student-*t* interval **46.77–55.32%**), versus **50.00%** for the matched no-old-write head. The paired old-teaching effect was **+1.04 percentage points** [−3.23, +5.32]. Both prespecified requirements fail: the absolute interval does not clear the **66.67% scalar-symbol ceiling**, and the causal interval does not clear zero. The full frozen run and independent audit passed. This is a clean negative E3 test for this engineered head on the cyclic family, not a reversal of its earlier positive result on the easier rank family.

![V9.10D key results](results/KEY_FIGURE.png)

## What was tested

Nine randomly selected uppercase bytes were assigned to three hidden classes per world, with A beating B, B beating C, and C beating A. Of 27 cross-class unordered pairs, 18 were taught in both byte orders for eight bouts (288 actual 12-byte records), while the other nine formed three disjoint directed triangles and were never reinforced. Each symbol had four taught neighbors, exactly two taught wins and two taught losses, plus one withheld win and one withheld loss. The same input framing, parent byte/KC encoder, 5,177-weight online head, learning rate, feedback, and timing were frozen from [V9.10C](../BYTE_CORE_V9_10C_ONLINE_SHARED/REPORT.md). The head received only the actual cue's pre-outcome KC vector and feedback; it had no symbol IDs, classes, pair slots, reverse-cue queries during training, or hidden labels.

This fixture makes the countermodels sharp. An exact taught-pair table has no answer on the nine withheld pairs and ties at 50%. A single scalar assigned to each symbol, however it was learned, cannot orient all three edges of any withheld triangle; its best possible held-out score is **6/9**. Exhaustive balanced three-class assignments using *only taught edges* yield three equivalent global rotations that all predict the same nine withheld answers, so the task is identifiable from the examples. These analytic checks are outside the learner. The engineering-world KC codes were distinct for all opposed cues (minimum L2 distance 2.0); an oracle linear head allowed to see **all 27 labels** could fit them, showing representability of the complete labelled set, not that the online learner could infer its missing labels.

`H_W` and `H_N` received identical old bytes, labels, clocks, and sensory/predictor updates; only `H_W` performed old head writes. Contemporaneous native `REF_W` and `REF_Z` similarly differed in old native value writes. All nine held-out pairs were queried read-only before any feedback on them. After the old phase, a one-day probe was followed by a 72-record, directly taught cyclic cohort on three *disjoint* new symbols, with rest (`R`), exposure/no-write (`S`), and write (`L`) forks; a further day preceded final probes. This later cohort measures new E1 acquisition and limited interference, not another E3 transfer test.

## Confirmatory and follow-up results

| Confirmed cell, 64 worlds | Choice [95% interval] |
|---|---:|
| Shared head `H_W`, **taught** old pairs | **97.57%** [96.65, 98.49] |
| Shared head `H_W`, **withheld** old pairs at first probe | **51.04%** [46.77, 55.32] |
| No-old-write head `H_N`, withheld | 50.00% [50.00, 50.00] |
| Native `REF_W`, withheld | 47.40% [43.62, 51.17] |
| Native no-old-write `REF_Z`, withheld | 51.04% [47.49, 54.60] |

The paired native `REF_W−REF_Z` difference was **−3.65 pp** [−9.46, +2.16], so native REF also showed no positive cyclic transfer here. The engineered head exceeded native REF by **+3.65 pp** [−1.25, +8.54], an unresolved descriptive contrast. Mean signed held-out head margin was **+0.00103** [−0.00679, +0.00886], versus +0.19614 [0.18869, 0.20358] on taught pairs. Thus the failure is visible in continuous evidence as well as thresholded choices.

The delayed and later-learning probes did not rescue the old relation. `H_W` was 51.04% after the first day. At the final day, `H_W_L` was **50.17%** [46.51, 53.84] on old withheld pairs; its contrast against `H_N_L` was **−2.95 pp** [−7.15, +1.25]. New head writes changed old held-out choice by −0.87 pp [−3.40, +1.67] relative to the same-prestate sensory/no-write fork; `S−R` was exactly zero in this schedule. The later new cohort was genuinely acquired on its **directly taught** pairs: `H_W_L` reached 100%, versus 48.70% for `H_W_S`, a paired **+51.30 pp** [45.04, 57.57]. These data support new E1 learning, not persistence of an old E3 cyclic rule that was never established.

## Interpretation and limits

The current head can fit presented pair codes and retain/reuse some information in the V9.10C rank task, but its frozen update/readout has no demonstrated ability to infer this cyclic relation from balanced examples. Near-perfect trained performance alongside chance held-out performance is **consistent with pair-specific fitting**; it is not proof that its weights literally implement a lookup table. The negative result also does not prove the byte/KC representation intrinsically lacks the relation: the engineering oracle shows the fully labelled cues are linearly separable, while inference from the 18 taught labels is a different requirement. The test changes task family, so the positive V9.10C rank result remains valid at its narrower E3 level. Nothing here establishes an E4 learner or a persistent core across relation families.

This experiment narrows the next decision to **how a compact learner binds shared symbolic roles or relational structure across pairs**. Increasing write dose or storage capacity solely to improve taught-pair accuracy has little justification from this result. A future mechanism proposal should first predict a positive held-out effect on the same balanced triangles under an old-teaching/no-old-teaching control, then face delayed and intervening-learning probes. This is a proposed criterion, not an additional experiment or claim that a particular new topology will work.

## Execution and audit

Engineering world 9906 passed source/feed equivalence, held-out exposure, 6/9 scalar ceiling, class-rule identifiability, KC representability, write flags, clocks, and code equality. Eight separate pilot worlds (2051–2058) were not pooled with confirmation and did not trigger parameter changes. All 64 prespecified confirmation worlds (2101–2164) ran without outcome-based stopping in **469.0 seconds**. Source code and parent assets were hashed before pilot/confirmation; canonical byte-order margins were frozen before hidden-label scoring. The independent auditor recomputed all 4,096 confirmation cell rows, exposure, scoring directions, world-level intervals, source hashes, and write ledgers. Maximum contemporaneous KC-code gap and clock error were both zero. The compact bundle is below 10 MB and references, rather than duplicates, unchanged checkpoint, datasets, and parent code.

See the [design](DESIGN.md), [key metrics](results/KEY_METRICS.csv), [world scores](results/CONFIRM_WORLD_SCORES.csv), [full summary](results/CONFIRM_SUMMARY.json), [audit](results/AUDIT.json), [source lock](results/SOURCE_LOCK.json), and [manifest](MANIFEST.json).
