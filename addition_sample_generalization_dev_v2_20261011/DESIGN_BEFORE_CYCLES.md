# MiniFly SAMPLE-GEN V2: generalization acquired per external sample

2026-10-11. **Prospective redesign; no learner has been trained under this protocol.** This replaces the proposed, unrun STABLE/WG1/SF1 efficiency experiment as the recommended next design. Historical protocols, results and locks remain unchanged. This document authorizes neither execution nor publication.

本轮的主问题：**在给定外部训练样本数和有限内部计算预算下，核心获得了多少解决未教组合的能力，而且这些能力是否能经过延迟和后续学习保持？** 已教题回忆不参与主评分。原始STABLE的100%是限定输入先验与固定拆分下的结果，不能直接搬入这个新基准。

Status: `SCIENTIFIC_DESIGN_SPECIFIED_IMPLEMENTATION_AND_QUALIFICATION_PENDING`. The accompanying checks validate task geometry and exposure only. They do not qualify any new encoder, learner, operations, statistical implementation or scientific runner.

## 1. Evidence, scope and benchmark revision

The permitted claim is E3 sample-efficient, persistent generalization within finite addition-like numeral systems. E1 acquisition is secondary; this version does not make a new E2 rendering claim. No NLP, additional operations, multi-digit output or general intelligence is added.

The old benchmark repeatedly used 25 fixed expressions, one split, magnitude receptors and an ordinal readout. Fresh noise/order seeds alone do not remove development overfitting to that task construction. This design changes the **latent numeral meaning, training-pair geometry, development/confirmation separation and sample-efficiency endpoint**, while retaining one-byte prompted answers and noisy/missing observed targets.

The primary suite uses five opaque numeral bytes `A`–`E`. Each world independently samples a bijection π_A from these symbols to values 0–4. The environment computes y=π_A(a)+π_A(b), with answer byte `0`–`8`. The learner never receives π_A or the true y; it receives only an actually delivered, possibly corrupted answer byte.

Examples such as `A+B=3` do not establish whether A means 1 or 2. Subsequent examples must determine useful contributions. Every symbol appears in teaching, in both operand roles. No test asks the learner to infer the meaning of a symbol with no teaching evidence.

Ordinary `0`–`4` addition remains a **confirmation protection suite**, using the identity numeral map. It is not pooled into the primary opaque-numeral score. Opaque-symbol performance is not described as understanding unfamiliar literal decimal notation.

Both suites use an intervening alphabet `F`–`J` with an independent bijection π_B to 0–4. This supplies genuinely new numeral meanings under the same finite operation and output alphabet. The disjoint alphabets provide separate sensory identities; the shared output/bias and candidate internal circuitry can still interact. Survival here does not establish retention under every kind of overlapping-context interference.

## 2. Shared sensory and output contract

All configurations receive exactly the same delivered bytes and model times. Public learner inputs are byte, time and the causal experiment's permitted-learning switch. No world ID, latent values, noise kind, phase name, held-out flag or pair graph is supplied.

Operand vocabulary: `01234ABCDEFGHIJ`; observed/output answers: `012345678`; punctuation: `+=\n`. The common fixed parser reads one operand byte, `+`, one operand byte, `=`, an optional delivered answer digit, and newline. It extracts byte identity and role; it performs no arithmetic. The scalar answer is emitted at `=`, before any answer byte arrives.

The sensory basis has D=31 features: 15 left-role identity indicators, 15 right-role identity indicators and one shared constant. Exactly three features are active. **There are no value thresholds, ordinal input magnitudes, pair-key hashes or learned front-end parameters.** This is one shared, label-blind interface replacement, qualified once for all arms; it is not another encoder search.

For STABLE-ID, map each feature to 15 distinct eligible KC coordinates using one fixed, label-blind projection. There are 465 contacted KCs and 45 active KCs per expression. Full inherited state has N=5,177 coordinates. Eligibility and canonical birth assets are fixed before execution and shared immutably; mutable learners are independent.

Every arm emits `argmax_k −0.5(k−v)^2`, k=0…8, with the lowest digit on an exact tie. This common ordinal output prior is retained and disclosed. It does not supply either operand's latent value or their sum. Learning starts from zero content, not fitted arithmetic coefficients.

Several proposed cores have an **additively separable readout by construction**, and TOPO has separate operand-role units. Thus this experiment measures how efficiently an engineered core learns unknown numeral contributions from noisy examples and retains the resulting unseen-combination ability. It does not demonstrate discovery of the addition operator without structural priors. Numeric decoding of an actually observed answer is part of the common feedback interface, not a hidden correctness bit.

## 3. Training geometry and semantic holdout

There are 15 unordered pairs with repetition among five symbols. Teach exactly five semantic pairs. Supply both orientations of every non-diagonal pair. A diagonal occupies two identical slots, so every teaching epoch has exactly ten external records. The other **ten semantic pairs** are permanently held out in both orientations, including all later branches and replay caches.

| Family | Five taught semantic pairs | Development use |
|---|---|---|
| PATH_LOOP | A randomly relabelled five-vertex path plus a loop at a random vertex | Allowed |
| STAR_LOOP | A randomly relabelled four-edge star plus a loop at its center | Allowed |
| ODD_CYCLE | A randomly relabelled five-cycle; no diagonal teaching | **No learning run or endpoint inspection before confirmation** |

The first two families have 300 and 5 distinct labelled training masks respectively; ODD_CYCLE has 12. Geometry fixtures must verify these counts, full symbol/role coverage, semantic disjointness and clean-task identifiability. The independent-role additive design matrix has rank 9 in every mask. A clean-data additive fit is a solvability witness, never an initialized learner or an injected answer function.

π_A and π_B are independent uniform permutations. Draw graph embeddings, permutations, noise and order from separate namespaces. Repeated permutations by chance are retained; no outcome-based rejection or replacement world is allowed. The public geometry families are not claimed to exhaust all possible addition curricula.

## 4. External samples and feedback

Old-training budgets are n∈{10,20,40,80,160,320} **total external records**, not repeats per expression. The 320-record stream consists of 32 independently shuffled ten-record epochs. Each smaller n is the actual prefix of that master stream; no redraw or artificial rebalancing occurs after observing behavior.

Two predeclared feedback regimes:

| Regime | Correct delivered digit | Wrong delivered digit | Missing digit |
|---|---:|---:|---:|
| STANDARD | 0.80 | 0.10 | 0.10 |
| BLURRY | 0.60 | 0.20 | 0.20 |

Kinds are independent draws per record, not exactly balanced within each ten-record epoch. Wrong digits are uniform among the eight alternatives to the true target. Missing answers cause no target-induced learning. Report realized counts; the regime is known to the environment, not to the learner. All compared arms receive the same realized records.

Count every external record, including wrong/missing feedback, in n. Also report real input bytes, present answers, environment-side correct answers, distinct taught pairs and repeats. Correctness metadata is never a learner input. Internal replay adds **zero external samples** but adds internal updates, CPU and cache state; it cannot be advertised as free learning.

## 5. Persistent learning curves without restarting prefixes

For each world/configuration/history, run one continuing 320-record OLD parent. At each n checkpoint:

1. Query the ten OLD held-out semantic pairs on a read-only copy: immediate E3 accuracy.
2. Fork an independent continuation; rest 86,400 model seconds.
3. Teach exactly 80 external records from a separately generated PATH_LOOP cohort on F–J, using π_B and the same feedback regime.
4. Rest another 86,400 model seconds.
5. Query the OLD held-out pairs, OLD taught pairs, and NEW held-out/taught pairs on read-only copies. No target is delivered in any probe.

Use the same 80-record NEW master in every n continuation and arm. The OLD parent does not receive child teaching or probe outputs. Model time advances without wall-clock sleeping. Use DT=30/14 seconds per byte, six-tick slots and a silent sixth tick for missing answers. Different sample budgets intentionally have different training durations; no temporal-decay candidate is included in this roster.

The persistent curve's x-axis is **T=n+80**, namely {90,100,120,160,240,400} external records in that continuation's complete learning history. The immediate curve's x-axis is n. Showing persistent performance only against n would hide the 80 later samples.

Actual execution is 320 OLD records plus six×80 NEW records = **800 records per causal history**, because OLD prefixes are shared rather than re-executed. Distinguish conceptual checkpoint histories from physically executed records. Forks are correlated observations, not independent experimental units.

## 6. Causal histories and state ownership

Three histories per configuration/world:

- **W:** learns from actually delivered OLD and NEW targets.
- **N_OLD:** exactly W's bytes, timing and NEW training; all OLD target-induced content, visits, covariance, utilities, graph changes, replay caches/priorities and RNG advancement tied to learning are forbidden. Fixed sensory processing is permitted.
- **SHUFFLED:** learns from OLD answer digits permuted among present-target positions within each epoch; missing positions and prefix target histograms remain identical. NEW is W's original stream.

The core must never see the original label behind a shuffled or wrong target. No OLD cache survives in N_OLD. Disabling weight writes alone is insufficient. A frozen state-erasure probe removes **all** learned content/auxiliary used by output, routing or replay.

Each query begins from the same checkpoint copy, so preceding test expressions cannot change a later answer. Queries cannot update weights, visits, covariance, topology, replay state or the parent. They may use bounded transient sensory state that is reset with the line.

State erasure means restoration to the arm's recorded birth state, including its graph, random sensory weights, covariance initialization, counters and RNG. It does not mean zeroing a topology arm's fixed birth features. Erasure uses no target or benchmark label to choose what remains.

## 7. Fixed configurations and mechanisms

There are **nine physical configurations**, including controls. Seven non-reference configurations may be selected; a control that works best is not discarded because its name says control. All numbers below are prospective engineering choices, not biological constants or fitted outcomes.

### 7.1 STABLE-ID and COMPACT-ID: two references

STABLE-ID copies the accepted STABLE content equation, now under the shared identity-only sensory basis. On active coordinates: r=y−v, e=clip(r,−1,1), η_i=(1+visits_i)^−1/2, δ_i=η_i e/√45; F_i←clip(F_i+0.2δ_i,−2,2), S_i←clip(S_i+0.8δ_i,−2,2); increment visits, capped at65,535. Rest does not decay content.

COMPACT-ID keeps F_j,S_j,visits_j for each of the 31 feature groups and uses κ=15/√45: v=κΣ_active(F_j+S_j). Its update preserves the two independent bounds and δ_i above. It compresses synchronous duplicate coordinates; it does not merge bounded F/S into an unjustified single state.

Before any learning experiment, prove mathematical group equivalence and qualify output/numeric boundary behavior against STABLE-ID, including clipping and exact/near ties. Float accumulation differences cannot be silently tolerated at the emitted-byte level. These are **new-input references**; their performance is not the historical STABLE's100%.

### 7.2 U045: COV-R and DIAG-R

Both use v=xᵀw, w initially0 and bound ±4κ. Set P_0=10I (or p_0=10 for DIAG-R). On a permitted observed target, compute r=y−v and γ=1 when |r|≤1, otherwise γ=1/|r|.

**COV-R:** q=P x; d=1+γ xᵀq; w←clip(w+γ r q/d,−4κ,4κ); P←P−γ qqᵀ/d, then symmetrize using(P+Pᵀ)/2. P is maintained positive definite; nonfinite values or an unqualified loss of definiteness are technical failures, not grounds for an undisclosed eigenvalue repair.

**DIAG-R:** q_j=p_jx_j, d=1+γΣp_jx_j²; the same weight update; p_j←p_j−γq_j²/d. It deliberately discards off-diagonal correlations.

P/p updates are forbidden during OLD in N_OLD. There is no time forgetting, hidden noise label, learned numeral lookup supplied by the environment or prefit coefficient. These are new engineered residual-learning instances of U045, not claimed biological replications. The pair asks whether tracking input correlations is worth its sample and computation cost.

### 7.3 U043: TOPO-G, TOPO-S and TOPO-R

Use16 hidden units, eight per operand role, with exactly four incoming contacts per unit. Available sensory identities are the corresponding role's15 identity channels plus the common constant. Label-blind initialization covers every available identity at least once in its role; remaining contacts are fixed by birth RNG. θ∼Uniform[−0.5,0.5], outgoing β=0 and scalar bias b=0. All three arms have identical birth connectivity/weights and independent mutable state.

h_j=clip(0.25+Σ_i θ_ji x_i,0,2), v=b+Σ_j β_jh_j. At each target, cache the **pre-update** derivative d_j=1_{0<u_j<2}, r=y−v, e=clip(r,−1,1), and pre-update β. Let t be permitted present targets before this one: η_out=0.1/√(1+t), η_in=0.02/√(1+t). Update β_j←clip(β_j+η_out e h_j,−4,4), b←clip(b+η_out e,−8,8), θ_ji←clip(θ_ji+η_in e β_j d_j x_i,−2,2). No gradient uses newly updated β.

Each role indexes its 15 identities and constant locally as0…15. A unit's inactive frontier contains four distinct features, initialized by cyclic traversal from local index(4j mod16), skipping active edges; the cursor continues after the last chosen index. Utilities start at0. Update each frontier utility u_ji←(7u_ji+|eβ_jd_jx_i|)/8 using only actual past feedback and pre-update state. When a frontier feature grows into an edge, replace its frontier slot by the next cyclic feature that is neither active nor already in the frontier, with utility reset to0. A no-op does not advance this cursor. Static frontiers remain fixed. These rules depend on public feature indices, not world geometry or numeral values.

Every eighth permitted present target, consider one unit in cyclic order. At most one contact is deleted and one grown in that unit. Preserve minimum sensory coverage of every identity and the constant within each role, and fan-in4. Enumerate legal deletion/growth pairs from the active edges and the inactive frontier. TOPO-G chooses the pair by smallest|β_jθ_ji| on deletion, then largest frontier u_ji on growth, then lowest deleted and grown feature indices. TOPO-R chooses uniformly among legal pairs at the same opportunities. New edge weight is0; deleted content is erased; retained weights stay attached to their original feature/unit identity. An empty legal set produces a recorded no-op; actual no-op and mutation counts can differ as graphs diverge. TOPO-S computes the same learning/utility witnesses but never changes connectivity.

This is true weight-learning plus connection replacement, not gain scanning. There are no mixed-role conjunction units or new arithmetic operations. No graph mutation occurs on missing answers or forbidden OLD learning. RNG and opportunity ledgers are audited. Template parity and feasibility must precede science.

### 7.4 U047: REPLAY-P and REPLAY-U

Both start from COMPACT-ID's update. Keep at most32 permitted, actually observed examples `(feature IDs, delivered y, cached residual priority)`. Missing and forbidden targets are never cached. At each permitted external target: perform its normal update, insert it, and, when a prior record exists, replay exactly one earlier cached record, excluding the just-inserted record.

REPLAY-P samples with weights|last measured residual|+0.1; REPLAY-U is uniform. Update a sampled record's priority from its residual before its replay update. Internal RNG is independent of environment latent/noise generation. Replays increment the learner's actual update/visit clock. Query clones never replay. New labels are never generated from the learner's own predictions, and caches cannot include a held-out pair.

The proposed benefit is **external sample efficiency**, not rescuing an already adequate time-retention score. Rate-matched replay separates selection from simply performing more updates; COMPACT-ID supplies the no-replay reference. Extra updates and cache/selection costs are fully recorded.

This is a prospective change from U047's earlier retention-only activation condition: the new hypothesis is improving a low-sample generalization curve from already observed examples. It does not revive an old replay result as positive evidence. The canonical historical mechanism ledger remains unchanged; this exact new use has no efficacy evidence yet.

## 8. What is scored

For a held-out unordered pair, average correctness of its two orientations; score a diagonal once. Every one of the ten semantic pairs has equal weight. The exact emitted answer is scored against the clean environment truth, which is never fed back to the learner. Soft accuracy, AUC of margins, taught-cue recall and hidden-state distances are not the primary endpoint.

Let A_{a,s}(T_k) be W's persistent OLD held-out accuracy at T_k=n_k+80. The per-world sample-efficiency score is the normalized trapezoidal area against log2 total external samples:

S_{a,s}=Σ_{k=1..5}[A(T_k)+A(T_{k+1})]/2 × [log2(T_{k+1})−log2(T_k)] / [log2(400)−log2(90)].

This summarizes a **behavioral generalization learning curve**, not ROC AUC. Report all six raw points and the immediate curve; the summary cannot hide a late collapse. Multiplying accuracy by ten gives expected solved, previously unreinforced semantic cases in this bounded test set.

Compute causal gain curves G_N(T)=A_W(T)−A_N_OLD(T) and G_S(T)=A_W(T)−A_SHUFFLED(T), with the same integration. Raw W performance is primary, so an artificially bad no-write baseline cannot manufacture a candidate win. Positive causal gains are required for a learning claim.

Also report discrete observed sample-to-75% and sample-to90% persistent accuracy. Failure to reach a threshold is right-censored at400; do not extrapolate or convert it into a favorable numeric sample count. These are secondary measures.

Training samples are the primary information budget. Report actual CPU, wall time, arithmetic/visits, bytes moved, persistent/cache bytes and internal updates separately. Data efficiency and compute efficiency are different results. No `accuracy/CPU` ratio or composite score with posthoc weights is used.

## 9. Controls, witnesses and exposure audit

Required environment-side checks: strict raw/canonicalized exact-key table; training-target mode/mean; hardcoded decimal/ASCII-rank addition; fitted one-operand predictors; initial-state and complete learned-state erasure. Report each countermodel's actual behavior on each world, not a blanket1/9 chance claim.

A clean additive fit must solve every clean held-out pair in every geometry. A learned additive model is a **legitimate narrow E3 solution**, not an invalidating shortcut. Conversely, a fixed raw-symbol arithmetic function cannot be credited with learning hidden numeral meanings. Test structural impossibility of cold-symbol inference only as a negative fixture, never as a scored demand.

The exposure ledger seals raw bytes, delivered labels, permitted writes, cache insertions/replays and graph/covariance changes. Independent auditors reconstruct every prediction/update and validate all checkpoint state. No original-truth column may enter learner imports or its public API. Shared mutable arrays, target-dependent OLD auxiliary changes in N_OLD, feedback-before-output, missing-target updates, test writes and off-roster replay are mandatory negative cases.

## 10. Development, sealed confirmation and selection

**DEV:**12 fresh worlds,3 per(PATH_LOOP/STAR_LOOP × STANDARD/BLURRY) stratum, all opaque numeral maps and all nine configurations. Previous technical worlds/results are excluded. The candidate recipes and numeric constants are fixed before DEV. No parameter search or new arm is added after seeing DEV outcomes.

DEV selection uses S on the development worlds, never ODD_CYCLE outcomes. A non-reference arm is eligible if: all scientific audits pass; final OLD held-out W≥0.75; final NEW held-out W≥0.60; integrated W−N_OLD and W−SHUFFLED are positive; mean S−COMPACT-ID≥0.03; and resource qualification passes. These point gates are screening policies, not confirmation findings. Choose the largest mean difference; ties within1e−12 choose lower measured production CPU, then lower persistent bytes, then the declared roster order. If none qualifies, finish with a negative screen; do not modify the benchmark to obtain a winner.

DEV aggregates equally over the four family×noise strata, with three worlds in each. Final accuracy gates and causal score gates are computed on that same balanced population. They are never evaluated on a handpicked noise regime or successful geometry.

**Confirmation:** after candidate/source/gates are locked, generate fresh private confirmation entropy that was unavailable during candidate design. Run96 fresh worlds:

-64 opaque ODD_CYCLE worlds,32 per noise regime: **sole primary population**.
-32 ordinary digit-addition guards:8 per(PATH_LOOP/STAR_LOOP × STANDARD/BLURRY) stratum.

Run STABLE-ID, COMPACT-ID and the selected arm. Include its within-family comparator(s): COV/DIAG peer, or the remaining two TOPO peers, or REPLAY peer. No unselected family advances. Maximum is five configurations; selecting a simpler/control arm is valid. Held-out templates are public for auditing, but no learner performance on ODD_CYCLE is inspected before this one confirmation.

Fresh roots reduce direct test leakage; a different geometry adds one limited curriculum-structure holdout. Neither provides a theorem of immunity to benchmark overfitting. This result remains bounded addition, not E4.

Primary statistic: selected−COMPACT-ID mean paired S on64 ODD_CYCLE worlds, equally weighting the two noise regimes. Use a stratified paired Student lower bound: variance of mean=Σ_h w_h² s_h²/n_h, Welch–Satterthwaite df; one-sided95%. If variance≤1e−24, use the conservative Hoeffding lower bound for a difference in[−1,1], not a zero-width interval. All checkpoint/candidate comparisons remain within the world; neither events nor held-out pairs become independent n.

Adoption requires: primary mean difference≥0.05 and lower bound>0; both integrated causal-gain lower bounds>0; final opaque W≥0.75; final NEW held-out W≥0.60; ordinary-digit mean S and final held-out accuracy each no more than0.05 below COMPACT-ID; and all resource/audit gates. The protection guards are point policies, not statistically certified noninferiority. Mechanism-peer contrasts are diagnostic unless separately preregistered with multiplicity control. Only one selected-versus-reference efficacy claim is confirmatory; all-required adoption conditions form a conjunction.

These margins are prospective engineering minimums, not thresholds derived from old failing cues. n=64 primary worlds gives approximately5.2pp one-sided uncertainty when paired-score SD is0.25; SD is unknown. This is a capped test, not a power guarantee. Unresolved results do not trigger sample extension.

After confirmation, retire the concrete instances from future development. Public release of final reports does not create a reusable untouched test set. Any later benchmark or search version must explicitly record prior exposure and generate new instances.

## 11. Size, resources and operations

One full job=one world×configuration, with all three OLD parents, all18 persistent continuations and all read-only probes audited before atomic commit. There are21 state trajectories with shared prefixes; they are not21 independent replicates. A job executes2,400 external training records. Replay updates are separate.

| Stage | World/config jobs | Executed external training records |
|---|---:|---:|
| DEV12×9 |108|259,200|
| Confirmation96×maximum5 |480|1,152,000|
| Conditional maximum |588|1,411,200|

With six-byte frames this is at most8,467,200 actual training bytes, less when answers are missing. It is a maximum, not a mandatory confirmation roster. No advancement executes DEV only. Model queries and clone/rest work are accounted separately.

Mutable learned state cap for non-reference candidates:128KiB including caches, priorities, covariance, graph utilities and RNG state; bounded sensory-code cache≤50KiB separately disclosed. Core arithmetic allowance≤20,000 conservative scalar operations per externally present target including replay, gates and amortized graph work. A low-level implementation must qualify this count and report actual CPU/memory movement; sparse activity alone does not certify sparse computation. Fixed inherited assets and interpreter overhead are separately measured, not hidden in a tiny mutable-state claim.

Qualification must project the worst permitted roster at≤8 wall-hours and≤2GiB new artifacts including accepted receipts, provisional data, logs and packaging. Otherwise stop with RESOURCE_QUALIFICATION_HOLD **before** DEV; no silent thinning of n or removal of difficult strata. Default two detached workers; source/state parity must qualify any concurrency change. Caffeinate follows the supervisor. A10-hour session cap pauses only at an audited whole-job boundary, or a separately parity-qualified checkpoint. No automatic scientific restart or trajectory reexecution.

Store compact stream/transition witnesses and checkpoint seals rather than all covariance/hidden arrays per record. Auditor reconstruction must still check every output and state equation. Audit/runtime cost is reported separately from production cost and in total experiment expense.

## 12. Readiness and termination

Before science: implement all nine contracts, the common parser/identity projection, independent equations/auditors, sample/clock/fork accounting, negative fixtures, true detached-dispatch and analysis/package failure tests. Perform the user's three design–audit–test cycles on reserved technical namespaces, then source-lock the actual code and private-root admission. Benchmark geometry checks alone cannot waive this gate.

All nine configurations must pass technical qualification before any DEV endpoints are inspected. A technical failure stops admission; it cannot silently remove an arm or alter the winner pool. Any necessary recipe correction requires a visible prospective amendment and renewed qualification, rather than using DEV outcomes as tuning data.

Valid endpoints: NO_CANDIDATE_ADVANCES; CONFIRMATION_UNRESOLVED; RETAIN_REFERENCE; ADOPT_BOUNDED_SAMPLE_EFFICIENT_CORE. A sample-efficiency gain with additional computation is explicitly labelled; compute-efficiency improvement requires its own measured advantage. No outcome authorizes expanded arithmetic, NLP, another mechanism or automatic parameter rescue.

## References and provenance

- Accepted prior result: `../addition_value_retention_confirm_v1_20261011/results/REPORT.md`.
- Previous technical exercise: `../addition_efficiency_dev_v1_20261011/THREE_CYCLE_REPORT.md`; it does not supply new-test efficacy estimates.
- Mechanism IDs preserve the71-item project ledger. New recipes are not proof of entire-family acceptance/elimination.
- [Cawley and Talbot2010](https://www.jmlr.org/papers/v11/cawley10a.html): selection itself can overfit; independent confirmation is necessary here. We do not claim to implement a reusable-holdout privacy guarantee.
- [RigL2020](https://proceedings.mlr.press/v119/evci20a.html): motivates explicit weight-based removal and gradient-based connection growth. TOPO is a separately specified small engineering variant, not a replication of its results.
- [Robust recursive least squares2018/2019](https://arxiv.org/abs/1812.09985): motivation for bounded noisy-feedback estimation, not empirical evidence for this exact COV-R recipe.
