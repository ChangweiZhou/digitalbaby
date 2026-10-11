# MiniFly FEEDBACK-EFFICIENCY V1 experiment design

2026-10-11. Status: **DESIGN_V1; implementation and three-cycle qualification pending.** This ships a prospective design, not permission to launch science. Historical experiments and acceptance gates remain unchanged. COV-R is already adopted for its bounded scope; this experiment is an improvement test, not another prerequisite for using it.

## 1. Goal and evidence boundary

Obtain more ability on never-taught finite-addition combinations from a fixed amount of noisy observed feedback, within an explicit computation budget, while retaining previously acquired ability. The primary evidence level is E3 within this limited task family. Taught-key recall is secondary. No NLP, extra arithmetic operators, multi-digit output, unrestricted generation or giant scaling is added.

The predecessor SAMPLE-GEN V2 passed its frozen adoption gates. On its opaque primary worlds, COV-R achieved OLD final unseen accuracy 93.98%. NEW after 80 examples was 80.00% under STANDARD but 41.41% under BLURRY. Immediate/persistent OLD curves were almost identical. This motivates a low-sample, noisy acquisition test; it does not prove residual clipping caused the weakness or that forgetting never occurs.

The learner retains explicit engineering priors: role/identity features, an additive readout, and ordinal output rendering. We test learning unknown numeral contributions and composing them, not discovering the addition operator from an unrestricted hypothesis class. A successful new recipe replaces COV-R only within this scope.

## 2. Mechanisms and fixed roster

There are **six physical configurations, two mechanism families, and five selectable non-reference recipes**. Confidence normalization and diagonal approximation are variants of U045, not distinct biological mechanisms. U047 introduces a different operation: reusing observed experience. Controls remain eligible if they meet the same gates; a superior extra-update control is not hidden.

| ID | Role | Change from the adopted COV-R |
|---|---|---|
| COV-R | Frozen reference | Full correlation state, fixed residual threshold |
| COV-I | U045 candidate | Confidence-normalized residual threshold |
| DIAG-I | U045 candidate | Same normalized threshold, diagonal correlation state |
| DIAG-R | U045 comparator; selectable | Existing diagonal/fixed-threshold recipe |
| COV-REPLAY | U047 candidate | One uniform historical replay per eligible observed target; weights only on replay |
| COV-EXTRA | U047 update-count comparator; selectable | Same extra-update opportunities, but repeats the current observed target |

No confidence-plus-replay combination, capacity/rate sweep, priority replay, new encoder or topology arm enters this V1. A negative result retains COV-R and closes only the tested recipe/use, not its entire mechanism family.

## 3. Shared interface and equations

All arms receive identical bytes, time, and causal permission. Fixed parser: operand, `+`, operand, `=`, optional observed digit, newline. Operand vocabulary `01234ABCDEFGHIJ`; answer vocabulary `012345678`. Emit before the observed digit arrives. No latent map, clean answer, noise class, phase identity or held-out membership enters the core.

The shared feature vector x has dimension 31: 15 left identities, 15 right identities and a constant. Exactly three entries are one. Prediction is v=x^T w. Emit the nearest digit 0–8, with lower digit winning an exact tie. Initially w=0, P=10I for full arms, p=10 for diagonal arms. Weight bounds remain ±4κ, κ=15/sqrt(45). Rest does not decay these states.

For an allowed, present external target y, r=y−v is computed from the cached pre-feedback prediction. For R arms, γ=1 if |r|≤1, otherwise 1/|r|. For I arms, compute **pre-update** u=x^T P x (or its diagonal equivalent), τ=sqrt(1+u), then γ=1 if r=0, otherwise min(1,τ/|r|). No fitted multiplier is added. P is an internal confidence proxy, not a calibrated posterior variance.

Full update: q=P x; d=1+γu; w←clip(w+γr q/d,−4κ,4κ); P←P−γqq^T/d; symmetrize. Diagonal update: q_j=p_jx_j; u=Σp_jx_j²; same w update; p_j←p_j−γq_j²/d. All denominator, finite-state, symmetry and definiteness conditions require qualification. No undisclosed eigenvalue repair, covariance reset or tuning after DEV.

The I hypothesis is that uncertainty permits larger legitimate corrections early, followed by conservative updates after experience. The competing prediction is amplified wrong feedback. Existing results establish neither prediction.

### Bounded reuse and its comparator

COV-REPLAY and COV-EXTRA use the identical COV-R external update. Both maintain an independent FIFO ring of at most 32 permitted, actually observed `(three feature indices, delivered digit)` records, including duplicates. No clean-label replacement, guessed target or missing/forbidden record is stored.

Order on each allowed present target: (1) normal external w/P update; (2) if the pre-insertion ring is nonempty, one extra update; (3) insert the current example and evict the oldest if needed. COV-REPLAY chooses uniformly from that pre-insertion ring using its own recorded RNG. COV-EXTRA draws the same selection RNG for accounting but uses the current example instead. Both make exactly one extra update at every opportunity; opportunities depend only on permitted target count, not error or an oracle phase boundary. RNG can be freshly identical per arm and independent of environment generation.

For the extra update, predict afresh from the post-external w, recompute fixed-threshold r and γ, and use current P to compute q and d. Update **w only**, with the same clipping equation. P remains bit-identical across this extra update: revisiting an observation is not booked as independent new information. Record this as an engineering design choice, not a guarantee of unbiased estimation. Extra updates increment a separate counter, not the external-observation count.

Caches survive delay and intervening learning. Never clear them using a hidden OLD/NEW boundary. N_OLD forbids OLD cache insertion and learning RNG advancement; N_NEW forbids NEW external and extra updates, cache insertion and learning RNG advancement. Query copies never replay. This differs materially from the predecessor's COMPACT-ID replay recipes; their weak results do not validate this new COV/weights-only recipe.

## 4. Task construction and anti-overfitting boundary

Five symbols have an independently sampled hidden bijection to 0–4. OLD uses A–E, NEW uses F–J. The environment's clean answer is their latent sum; the learner receives only the delivered digit. Ordinary guards use OLD 0–4 with identity mapping and independently mapped NEW F–J. All comparisons use the same sampled maps, teaching masks and byte/noise streams.

Of 15 unordered pairs with repetition, teach five semantic pairs and withhold the other ten in **both orientations**, for the complete lifetime. Present both orientations of non-diagonal teaching edges; a diagonal has two identical slots. Each epoch has ten records. Held-out pair accuracy averages both orientations, with each semantic pair equally weighted. Every symbol is taught in both roles.

DEV uses the public PATH_LOOP and STAR_LOOP geometries from the predecessor. Confirmation uses a new fixed curriculum, TRIANGLE_TAIL: edges (0,1), (1,2), (2,0), (2,3), (3,4), uniformly relabelled. Use independent embeddings for OLD and NEW. This connected odd-cycle graph supplies an identifiable clean additive problem without a taught diagonal. Its exact role-design rank and exposure separation must pass the supplied geometry check.

No triangle-tail learner efficacy is inspected before candidate selection/source lock. Geometry-only enumeration and solvability witnesses are permitted; they are not learner trials. Confirmation entropy is generated after selection. New geometry and private maps reduce direct reuse of test instances, but are not an E4 result or a theorem against adaptive benchmark overfitting.

Feedback remains STANDARD (80% correct, 10% wrong, 10% missing) or BLURRY (60/20/20). Draw kinds independently per record. Wrong digits are uniform over the eight alternatives. The learner is never told which regime or record is wrong. Count missing and wrong records in sample totals.

## 5. Chronology, controls and exact execution budget

Reuse the predecessor's three OLD histories: W, N_OLD, SHUFFLED_OLD. Each has a continuing 320-record OLD parent, checkpointed at n=10,20,40,80,160,320. SHUFFLED_OLD permutes present answers within each epoch, preserving missing positions and epoch target histograms. At every checkpoint, a continuation rests one model day, receives the same 80-record NEW master stream, rests another model day and is probed without targets. Parent prefixes are never re-executed or affected by their children.

During each NEW continuation, at m=20,40,80, take an independent query copy, advance it one model day and measure NEW held-out/taught accuracy. This does not advance the teaching continuation or manufacture more training. At m=80, the existing final probe also measures OLD retention.

For the W parent at n=320 only, add two independent 80-record NEW continuations:

- **N_NEW:** identical OLD W state and NEW bytes/time, but forbids every NEW target-induced state change, including P, cache, auxiliary counters and learning RNG.
- **SHUFFLED_NEW:** identical OLD W state; NEW present targets permuted within epochs, with no clean answer retained in learner-visible state.

These additional controls directly test whether NEW ability comes from NEW teaching. N_OLD remains the causal control for reuse of OLD information. Disabling weight writes while allowing P/cache changes is invalid.

One full world-arm job executes **3×320 + 3×6×80 + 2×80 = 2,560 external records**. It contains three shared-prefix OLD parents and twenty continuation trajectories, **23 correlated state trajectories**, not 23 independent lives. Query copies add no training. Each learner/continuation owns independent mutable arrays, ring, RNG and counters; inherited fixed assets may be shared immutably. Model time, not wall-clock sleep, implements the days. Retain the predecessor's DT=30/14 seconds per tick and six-tick external-record slot; a missing answer uses five sensed bytes plus one silent tick. All causal histories share the same event clock, and probes cannot advance their training parent.

## 6. Primary measure and protections

Primary per-world score J is W's delayed NEW held-out curve at OLD n=320:

**J = 0.25 A_NEW(20) + 0.50 A_NEW(40) + 0.25 A_NEW(80).**

This is normalized trapezoidal area over log2 NEW samples 20–80, not ROC AUC. It measures acquisition after the **same fixed 320-example prior history**. Always plot total exposure T=340,360,400 alongside incremental NEW samples; do not advertise 20-sample learning from birth. Report all three points, immediate/delayed differences and countermodels. Primary population is opaque **BLURRY**, reported separately; a pooled STANDARD mean cannot rescue it.

Secondary: OLD persistent curve at T=n+80; NEW curves at all OLD n; held-out accuracy at NEW80; sample-to-75/90% (unreached thresholds stay right-censored); taught E1. OLD curve score S retains the predecessor's normalized trapezoidal area over log2 T at T=90,100,120,160,240,400. The denominator is log2(400)−log2(90); every segment contributes its mean endpoint accuracy times its log2-T width. All the NEW observations from one world are correlated. Production CPU, external/extra update counts, arithmetic, bytes moved, mutable state, peak RSS, audit/IO and total execution cost are reported separately. No posthoc accuracy/CPU composite.

Protection policies, evaluated relative to a fresh COV-R on the same worlds:

- BLURRY OLD sample-curve score and OLD final unseen accuracy each decline by no more than 3 percentage points; OLD final also ≥85%.
- Opaque STANDARD NEW J and OLD curve each decline by no more than 3 points.
- Ordinary guards' OLD curve and NEW J each decline by no more than 3 points.
- Mean measured production CPU per full job ≤1.5×COV-R; mutable payload ≤2×COV-R; peak worker RSS ≤512MiB.

These are prospective engineering budget/protection policies, not fitted biological thresholds or certified statistical noninferiority. A cheap diagonal recipe that is competitive but does not meet the efficacy gate may be reported as a cost tradeoff, without automatic adoption.

## 7. DEV, selection and one confirmation

DEV: 12 fresh opaque worlds, three in each PATH_LOOP/STAR_LOOP × STANDARD/BLURRY stratum; all six configurations; **72 full jobs, 184,320 records**. Both OLD and NEW use the world's assigned family with independent embeddings. BLURRY selection averages its two family strata equally. No previous technical or scientific world enters. All six recipes must technically qualify before efficacy inspection; an integration failure cannot silently delete an arm.

Eligible recipes meet all technical/resource/protection policies, have mean BLURRY J gain over COV-R ≥0.03, and positive NEW J W−N_NEW and W−SHUFFLED_NEW. Choose largest BLURRY J gain; exact ties within 1e−12 use lower measured production CPU, then payload, then roster order. These tiny DEV strata support selection only. No outcome-based retuning, roster addition or rescue of the benchmark is allowed. If none qualifies, package NO_ADVANCE_RETAIN_COV_R.

At most one recipe advances. Confirmation: **128 fresh worlds**, consisting of 64 opaque BLURRY primary, 32 opaque STANDARD protection and 32 ordinary guards (16 per regime); all use TRIANGLE_TAIL. Run COV-R, selected recipe, and at most one fixed mechanism peer: COV-I→DIAG-I; DIAG-I→DIAG-R; DIAG-R→DIAG-I; COV-REPLAY↔COV-EXTRA. Peer comparisons are diagnostic. Ordinary protection means weight its two noise strata equally. Maximum confirmation is 384 jobs/983,040 records; full program maximum **456 jobs/1,167,360 records**, conditional, not mandatory.

Sole confirmatory efficacy contrast: selected−COV-R J over the 64 primary worlds. Adopt only if mean gain ≥0.05, its one-sided 95% paired Student lower bound >0, both NEW causal-J lower bounds >0, and every protection/resource/audit policy passes. The all-required gates form one conjunctive adoption rule. Use world as the independent unit; paired differences across checkpoints are never separate n. The lower bound is mean difference minus t_(0.95,n−1) times its sample standard deviation/sqrt(n). For variance ≤1e−24, use mean difference minus sqrt(2 ln(20)/n), the conservative Hoeffding bound for a difference in [−1,1], rather than a zero-width confidence interval.

OLD E3 reuse additionally requires positive lower bounds for OLD integrated W−N_OLD and W−SHUFFLED_OLD in the primary worlds. Raw ability is always reported; an artificially weak control cannot manufacture success. The selected arm's raw primary J must also exceed the best declared raw/canonical-key fallback, observed-target mode/mean or fixed-rank countermodel by at least 0.05 in mean. This is a point shortcut guard, not a separate efficacy claim. Learned one-operand models are reported descriptively; a simple rule learned from examples is still legitimate narrow E3. All other intervals, multiple curve points and peer effects are explicitly descriptive. Confirmation failure ends this V1; retain COV-R without another candidate or extra worlds.

## 8. Exposure and recitation audit

Before every world: verify OLD/NEW alphabet separation, five taught versus ten withheld semantic pairs, both orientations withheld, role coverage, real delivered-target counts and clean identifiability. Replays can contain taught, delivered targets only. Probes provide neither targets nor correctness bits.

Score strict raw-key and canonical-key tables, observed-label mode/mean, fixed ASCII/decimal-rank sum, fitted one-operand models, birth-state and complete-state-erasure controls. A clean additive fit is a non-learner solvability witness. A learned additive solution is permitted narrow E3; memorization of the five taught pairs cannot supply the ten withheld answers. Missing-key countermodels must have declared fallback policies, not an invented universal chance rate.

An independent auditor reconstructs every output, external update and extra update; verifies P is untouched on reuse; checks ring contents/RNG, independent clone state, source identity, counts, clocks, probe immutability and full causal branches. Use compact witnesses and sealed checkpoints, not full-state JSON arrays on each record. The auditor must not call the candidate's update function as its own reference equation.

## 9. Required three cycles before a launch request

1. **Design–audit–test–revise:** equation and boundary tests, fair six-arm interface, exact-key countermodels, full causal no-write state, geometry/exposure, synthetic noisy data and known clean solvability.
2. **Design–audit–test–revise:** independent replay/RNG audit, scalar-versus-vector/SPD parity, complete erasure/clone independence, deliberate label/permission/state tampering, whole-job checkpoint and permission namespace tests.
3. **Design–audit–test–revise:** reserved technical full jobs on DEV geometries, measured time/CPU/RSS, detached-dispatch/caffeinate and safe pause tests; synthetic completion, reducer, final-audit and package tests. No confirmation efficacy or science namespaces are admitted.

All cycles concern this single design; they are not three mechanism-search rounds. Document revisions, then lock actual scientific sources, baseline provenance and statistical implementation. Only then request/receive explicit execution authorization. Geometry checks shipped here are not completion of these cycles.

## 10. Operations and stopping

Two detached workers initially; no automatic concurrency increase. Caffeinate follows the active supervisor. Each Mac session has a hard 10-hour limit, stopping safely at a whole-job boundary; admission targets ≤8 hours using a conservative estimate from qualified end-to-end jobs plus analysis/packaging. If this cannot be met, stop before science and revise visibly; do not silently reduce n after efficacy inspection.

Resume requires explicit authorization; never automatically restart scientific failure or repeat executed records. Commit a full job only after its independent audit. Partial state/checkpoints and accepted jobs are counted separately.

Fix the predecessor's postprocessing failure by construction: live progress reads cached counters, never decompresses science receipts while holding the shared status lock. One bounded pass creates compact metrics; selection/analysis consume those metrics. Test concurrent monitor/reducer access and package completion before launch. Pure operational recovery cannot change science equations or gates. No GitHub publication occurs automatically unless separately authorized.

Valid outcomes: NO_ADVANCE_RETAIN_COV_R, CONFIRMATION_UNRESOLVED, RETAIN_COV_R, ADOPT_BOUNDED_FEEDBACK_EFFICIENT_CORE. Negative results are completed science, not invitations to broaden the task.

## Provenance and literature limits

- Adopted result and source: [SAMPLE-GEN V2 report](../addition_sample_generalization_dev_v2_20261011/results/REPORT.md), [core equations](../addition_sample_generalization_dev_v2_20261011/cores.py), [prospective review](../addition_sample_generalization_dev_v2_20261011/NEXT_STEP_REVIEW_20261011.md).
- Mechanism identities/status: [71-item project ledger](../PROJECT_DOCUMENTS/MECHANISM_SCREEN_20261007/NEXT_ROUND_PRIORITY.md), supplemented by this version's [mechanism review](MECHANISM_REVIEW.md). Old ledger statuses are not silently rewritten.
- [Fedus et al., 2020](https://proceedings.mlr.press/v119/fedus20a.html) motivates measuring replay capacity/update ratio separately; its Q-learning findings do not establish efficacy of this bounded supervised recipe.
- [Yu et al., 2018/2019](https://arxiv.org/abs/1812.09985) studies robust diffusion RLS with time-dependent update constraints. Our COV-I confidence threshold is a new engineering hypothesis, not a replication of that algorithm or its guarantees.
