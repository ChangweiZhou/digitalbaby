# V154 — native context, readout and stopping: a bounded bridge diagnosis

25 October 2026. **Pre-execution V154 protocol. The user has authorized one bounded three-cycle exercise under this identity. GitHub publication and formal confirmation are not authorized. Lifecycle authority: minifly_v154_dev_20261025/STATUS.json.** V152 remains completed with `REFERENCE_NOT_READY`. Its historical design/results and the unexecuted V152-R1 proposal remain preserved.

The goal is **better retained complete own-byte/EOS generalization to unseen addition per total feedback and actual compute**. Deployable candidates stay in the native Full151/R_center lineage, with gradual warmup and local writes. No arithmetic feature/rule, attention, Transformer, ELM/reservoir learner, auxiliary language campaign, forced stop or answer-length channel is introduced.

This serial number identifies a diagnostic protocol. It does not announce a new kernel or inherit an earlier version's capability.

## 1. Evidence and the decision it supports

The newly completed V149 Father Brown run reports FE-PERM observed-KC R1 loss 3.767 BPS, FE0 binary-KC R1 3.379, last-byte R1 3.554 and bigram 3.553. Its simple KC-vote readout is weaker. These are bounded text-prediction measurements on untaught states. The three registered predictions P1–P3 were supported and the separately registered decision was `CONTEXT_FIRST`; these are not four independent predictions. [Pinned report](https://github.com/ChangweiZhou/digitalbaby/blob/67e07f4a649b611f9d5706e48048e25e0c0f64c0/minifly_v149_input_ceiling_dev_20261025/REPORT.md).

V149 changed permutation and decay together. Its evaluator R1 also uses a bias, regularization selection and a seen-class/OTHER parameterization. It establishes a comparison within that finite readout family; it does not establish an information-theoretic ceiling or native addition ability. V154 therefore uses matched raw-byte addition streams, a controlled front-end factorial and explicitly separated readout instruments. It does not subtract a text score from an addition score.

V150 showed output-level learning in its own setting. V151 found a valid native one-step response and a dose mismatch. V152's exact-BPTT and local recurrent references all failed complete acquisition, with empty-EOS responses and 0/9,216 immediate exact probes. Native efficacy was never assessed in V152. These distinct interventions do not identify one shared cause.

The next work separates three questions:

| Lane | Question | Result can establish |
|---|---|---|
| C — saved checkpoints | Does additive q control premature termination at the learned V152 state? | A read-only contribution/necessity result in that recurrent reference |
| A — fixed encoding | What changes with permutation, decay and the representation tap? | Accessibility under a declared probe, and any exact code collisions |
| B — readout/native bridge | Can online local categorical writes use the same features; where does the actual native path differ? | Matched readout-learning diagnostics and actual native acquisition/retention; conditional descriptive E3 |

Lane C uses existing recurrent assets. Lanes A/B use native circuitry. A failure in C or in an external optimizer does not veto the native lane. Oja learning, biased sampling, KC clocks and new cell types remain deferred hypotheses, not experimentally excluded families.

## 2. Operative parent and the controlled front-end factorial

Pin native sources through V152's `SOURCE_LOCK.json`, C16 codebook, receptor table, canonical connectivity, 16 independent mutable compartments, native fast/slow/adaptation equations, event timing and signed-write interface. Keep `homeo=False`, fresh adaptation at record start, and the original receptor normalization `rho=p/max(p)` for nonempty p. Do not change top-k, thresholds, input dimensionality or output codebook.

The four front ends use the same receptor table and initial state:

| ID | Permutation on a received byte | Decay time constant | Anchor |
|---|---|---|---|
| I_FAST | Identity | 10 seconds | Historical FE0 |
| P_FAST | Fixed existing PERM | 10 seconds | Permutation-only contrast to I_FAST |
| I_SLOW | Identity | `−DELTA/log(0.95)` | Decay-only contrast to I_FAST |
| P_SLOW | Fixed existing PERM | `−DELTA/log(0.95)` | Historical FE-PERM |

`DELTA=30/14` seconds. Historical FE0's per-byte factor is **exp(−DELTA/10) ≈ 0.8071**; the frequently quoted 0.81 is rounded. Use the actual 10-second constant so I_FAST can match FE0 exactly. The slow factor is 0.95. The factors' half-lives are about 3.23 and 13.51 byte intervals; those values describe trace decay, not demonstrated useful context length.

At elapsed interval dt, first apply `p <- exp(−dt/tau) p`. On a received payload byte, apply the selected identity/PERM operation once, then add its receptor pattern. Pure time advance and out-of-band controls do not apply PERM. Byte arrival, prompt boundaries and teacher forcing retain the parent's event schedule. Use p/max(p) for the actual encoder; an unnormalized magnitude is not silently exposed to a learner.

Before feedback, qualification must match I_FAST to the original FE0 and P_SLOW to FEPerm across arbitrary payload/control schedules, including NUL, all 256 bytes and repeated time advances. Pin the exact array permutation convention. A failed endpoint match stops that contrast, with no scientific interpretation or parameter retuning.

Report the paired permutation effects at both decay constants, the paired decay effects at both permutation settings, and their interaction. Two development worlds yield descriptive paired effects. A single winning corner does not identify the factor responsible for every past failure.

## 3. Lane C: existing V152 stopping contributions, zero training

Use only the six saved W checkpoints at the already selected eta=0.03: R_BP, R_LE and R_LE0 at OLD end and final post-idle NEW end. OLD-end queries cover the 64 OLD records; final queries cover OLD+NEW, 128 records. Hash and semantic-state checks must match the saved index before use. All six selected files are currently present locally; their contents were not loaded or scored while preparing this design.

For each question run three read-only conditions from independent clones:

1. Full learned output: `z=z_context+q`.
2. No-q output: `z=z_context` with the learned recurrent parameters unchanged.
3. Q-only output: `z=q`, disabling the context contribution.

Each condition generates its own prefix greedily over all 257 classes, with the four-decision cap and its own EOS. Teacher-forced analysis uses another clone. Never reuse a full-model guessed prefix for an ablation. Do not edit original checkpoints or reset a counter and train again.

At each true answer position, decompose the pairwise payload-versus-EOS logit margin exactly:

`z_y−z_EOS = (z_context,y−z_context,EOS) + (q_y−q_EOS)`.

Save both components, full/no-q/Q-only choices, complete accuracy, empty-EOS rate and CAP rate. Inspect all true reply/EOS positions; first-payload and true-EOS summaries must remain distinct. This decomposition shares the teacher-forced passes already budgeted for C.

The six checkpoints define 576 anchor questions and **1,728 own/teacher-forced query pairs**, at most 12,096 output distributions. They reveal no unseen assessment label. Q-only greedy decoding can emit empty EOS or repeat a byte to the cap; that structural null is disclosed.

If removing q changes stopping but fails to recover exact answers, q contributes to termination at that state without proving it explains acquisition. If no-q recovers exact answers, a jointly trained model has a suppressed useful context path; removing q at evaluation still differs from training without q. If no-q fails, co-adaptation prevents a claim that q was harmless. No result automatically launches Q0 retraining. Missing checkpoints produce `ASSET_UNAVAILABLE` for C; do not recreate them by rerunning V152. Native A/B can still proceed under their own checks.

## 4. Addition data, feedback and information separation

Use two predeclared fresh development worlds: **202610255401 and 202610255402**. V152 calibration and its sealed assessment worlds are not reused for new fitting or selection. Before any feedback, lock the complete bytes, split membership, orientations, orders, initialization and configuration. No data are generated by this design delivery.

Each world uses the same bounded task format as V152: unsigned operands 0…19, 210 unordered orbits, split 64 OLD / 64 NEW / 32 validation / 50 sealed assessment. One independently seeded orientation per orbit is presented. Do not teach its reverse as augmentation. The evaluator alone builds orbits and decimal sums. Every learner receives ordinary bytes and an out-of-band query/reply boundary, with no arithmetic parser or payload-specific delimiter behavior.

Raw prompts are decimal `a+b=`. Real feedback is the expected answer bytes followed by EOS=256, predicted before revelation. Teacher-force only the actual observed byte. All 256 payload bytes remain legal; no digit-only mask or forced EOS is used. Correct answers have at most two payload bytes; the four-decision cap reports CAP when the learner does not terminate.

Keep `s(n)=1/16+(15/16)min(n/1024,1)`, with n the previous observed reply/EOS targets, including repeats and inhibited observations. Prompt prediction is not added as a new objective. Every phase is 64 records × eight predetermined shuffled passes. No rate grid, extra pass, new world or gate change is authorized by a negative result. Expected outputs remain available in this bounded assay; blurry/scalar-only learning is outside its evidence.

The 32 validation orbits are used for fixed diagnostic reporting, with no hyperparameter selection. No probe sees the 50 assessment orbits, and no external readout supplies a native answer. Seal assessment scores until the acquisition/retention summaries and all settings are frozen. Reusing a feature cache for several diagnostic fits must not turn an assessment feature or label into training input.

## 5. Lane A: twelve fixed encoding views per world

For each of the four front ends collect these three aligned taps from the same untaught native trajectory:

- **RHO:** the 88-dimensional operative receptor read vector p/max(p), at the prospective read time.
- **X:** native binary top-k KC code at that same time.
- **OBS:** actual observed KC activity after the prescribed next-step rest/adaptation, as the native readout receives it.

Collect OLD/NEW/validation true prefixes only: 160 question records × four front ends × two worlds = **1,280 physical encoding record applications**, at most 3,840 reply/EOS target positions. The three taps share the physical trajectory; do not count three independent lives. No native value writes occur. Save unnormalized trace magnitude for descriptive logging only, outside the fitted feature vector.

Adaptation remains active, with the inherited fresh reset at a query start. Because signed writes are expected to leave the nonplastic activity trajectory unchanged, verify X/OBS parity between a small genuinely taught trajectory and its no-write counterpart before claiming a shared native feature view. If parity fails, disclose the drift and mark that cross-readout/native contrast unmatched; do not disable the mechanism to force parity.

For each of the 12 views/world fit one **OFF_257** evaluator instrument. Normalize its feature `phi=h/max(1,||h||2)` without adding an input clock, position channel, empty-state indicator, arbitrary class bias or previous-byte one-hot. Empty h remains empty. Its logits are `W phi` over all 257 classes, q=0. This is a prospective restricted probe, not a reproduction of V149's bias/OTHER/L2-search setup and not an upper bound on representable information.

Readout instruments use zero initialization. Their training objective is the warmup-weighted OLD-then-NEW repeated W stream's mean cross-entropy plus `||W||F²/(2 Tcap)`, where the fixed, label-independent **Tcap=3,072** is the maximum reply/EOS targets in a 1,024-record W history. The regularizer is a declared unit-prior convention; it is not selected on these answers. An offline fit may aggregate duplicate feature/target rows by their exact warmup weights, while charging their virtual occurrences.

Bound each batch fit to 60 iterations and 120 objective/gradient evaluations, float64 throughout. Save every stopping reason and final gradient residual. A numerical residual criterion of 1e-6 in infinity norm labels numerical qualification; it is not an accuracy gate. A finite unconverged result is `OPTIMIZER_UNRESOLVED`, not a low information ceiling; no wider grid or longer fit follows automatically. It does not block native learning.

Add a collision audit on the cached teacher-forced prefixes. Exact equal features with different next targets establish an ambiguity at that tap for those contexts. Different features with poor finite-probe loss only establish poor decoding by that fitted instrument. RHO/X/OBS have different dimensions and nonlinear transformations: a probe loss difference is not a direct measurement of mutual information or a proof that Top-K irreversibly lost all useful history.

## 6. Lane B: same-family writer comparisons and the actual native path

Readout instruments are evaluator-only. They are never installed as an ELM/reservoir learner, never used as a candidate comparison baseline on E3, and never fed back into native teaching, memory or answer selection. This is the bounded accessibility-probe role already used in V149. The only deployable candidate paths below are native fast/slow circuits.

To keep the readout comparison finite, use the two historical factorial corners **I_FAST and P_SLOW**, at X and OBS taps, on each world: four feature cases/world. For these cases retain OFF_257 and add:

| Instrument | Logits | Learning | Isolates within its own family |
|---|---|---|---|
| LOC_257 | `W phi` | Online analytic categorical error | Online versus batch optimization with OFF_257 |
| OFF_C16 | `C P V phi/4` | Bounded batch optimizer | Fixed C16 output-subspace constraint versus OFF_257 |
| LOC_C16 | `C P V phi/4` | Online analytic categorical error | Online versus batch optimization with OFF_C16 |

C is the unchanged 257×16 codebook and `P=I−11T/16`. q=0 in all four instruments so their explicit output offsets match the primary native factorial. This is a controlled diagnostic choice, not a claim that class bias should be banned. Lane C handles the separate V152 bias question. Do not silently add bias in a more favorable readout.

LOC_257 updates only active feature-to-output synapses using `(p−onehot(y)) phiT`. LOC_C16 uses `P CT(p−onehot(y)) phiT/4`. No encoder gradient, BPTT, autograd through state, attention or task relation is used. Both receive exactly the same 1,024-record W stream, feedback and warmup as the corresponding native W history. They have no prior-teaching causal branches and make no E3 claim.

Use the same quadratic regularizer and zero initialization as their paired batch family. For each event use local gradient `s(n) g + W/Tcap` (or V/Tcap); implement uniform decay lazily with exact timestamps/scales if needed. Choose the fixed safe step from a representation-independent smoothness bound: eta=1/(1+L²), L=1 for the full port and L=||CP/4||2 for C16. This is a conservative numerical convention, not outcome-tuned optimization or a guarantee of stochastic convergence. Report the actual eta. Qualification compares one online step against the analytic gradient of that identical one-position objective. A local/batch gap then concerns finite optimization, ordering and exposure, not an impossibility theorem about local synapses.

All instruments report final trained OLD/NEW and validation teacher-forced loss. Online instruments additionally report OLD at OLD end. They do not generate unseen answers or borrow the native's generated prefix. OFF_C16 has a much freer reachable parameter set than native fast/slow writes; it is a capacity bridge, not an isomorphic native optimizer. Across-family gaps also include parameterization, decay, feature scale and update timing.

### Native roster, independent of probe/BP success

Run **N_2B** on all four front ends, plus **N_A on P_SLOW** as the old-teacher anchor: five configurations × two worlds = ten groups. No new recurrent model is trained.

All native groups use the same fixed C16 and q=0. N_A receives `0.1 s(n)[tanh(Pv/2)−C_y]/2`. N_2B receives `0.2 s(n) P CT(p−onehot(y))/4`, as specified prospectively in V152. Values v come from the actual native circuit. Compute errors before the update, use unchanged signed teaching, and fail on an unrepresentable coefficient instead of clipping it.

N_2B's factor two is inherited from V151; do not recalibrate it using new answers. Different front ends can change active codes and effective write gains. Log those gains in the small preflight and report any variation; a native factorial effect is then an effect of the complete changed input path, not automatically pure information preservation. N_A versus N_2B changes objective and gain together and must retain that interpretation.

Each group has W (learn OLD then NEW), SH_OLD (same OLD sensory/feedback/timing without plasticity, then learn NEW) and N_OLD (share W's actually learned OLD prefix, then NEW with writes off). Inhibited branches still advance observations and native nonplastic time/adaptation. Execute the shared prefix once. This is three correlated histories and **five physical phase streams**, 2,560 record applications per group.

The complete native roster therefore has **25,600 physical record applications, at most 76,800 reply/EOS target applications and 30 correlated histories**. Neither a BP reference nor an external classifier's loss determines native admission.

## 7. Acquisition, retention, unseen scoring and countermodels

Keep immediate complete own-answer probes after each actually updating native record, plus initial OLD, OLD passes 1/2/4/8, no-prior OLD end, each branch's pre-NEW, OLD+NEW after NEW passes 1/2/4/8 and after 128 native byte-step idle intervals. Preserve independent read-only clones and the genuine own-prefix continuation. Report empty EOS, CAP, first payload and true EOS separately; teacher-forced loss cannot waive exact-answer failure.

A configuration/world is ready for a final held probe only if its W complete OLD accuracy at OLD end improves over initial OLD, and its W complete NEW accuracy at NEW end improves over pre-NEW (an already-perfect baseline qualifies). This is the inherited minimal assay-readiness check, not a useful-accuracy threshold. Publish all small/frequency-explainable successes and failures. Failure closes that group's held score without retries; it does not erase its acquisition data or veto other native groups.

Freeze all acquisition/retention summaries, configurations and readiness decisions across the native roster before opening any assessment score. Then evaluate the 50 never-taught whole orbits once at each ready group's final W/SH_OLD/N_OLD state. No assessment prompt, label or score goes to a readout fit, gradient check, hyperparameter selector or later teaching. Ineligible groups have no held score; report the admission pattern explicitly rather than treating unscored cells as zeros.

Track exact-key and evaluator-only canonicalized-pair tables, favorite complete reply and generic position-frequency countermodels from the actual observed training stream. Include their state/compute and prior-teaching/inhibited provenance. Whole-orbit holdout prevents a taught reverse pair from leaking an answer. Countermodels never answer for the native core.

The primary descriptive E3 contrast is each ready group's **W minus SH_OLD** exact unseen accuracy after NEW and idle, alongside absolute W accuracy, relevant OLD retention, N_OLD and the recitation/frequency controls. Report all paired worlds and all four factorial corners. Never convert byte positions, branches, repeated prompts or teachers into independent replications. Two worlds cannot justify statistical confirmation, an automatic kernel adoption or a broad local-learning claim.

Plot acquisition/retention against actual cumulative target exposure and actual training CPU. The single final unseen score is paired with its actual cost; there is no unseen learning curve. Separate feature extraction, optimizer reuse, local-head updates, native writes, encoding/sorting, clones, readout and audit overhead. A faster method receives no extra examples beyond the roster.

## 8. Finite budgets and exposure ledger

| Application category | Maximum |
|---|---:|
| Native physical training/inhibited records | 25,600 |
| Online diagnostic-head record presentations (16 × 1,024) | 16,384 |
| Untaught feature-collection record applications | 1,280 |
| Technical records, including any failed attempts | 64 |
| **All new record applications** | **43,328** |
| **Corresponding reply/EOS target positions, upper bound** | **129,984** |
| Batch diagnostic fits (OFF_257 24 + OFF_C16 8) | 32 |
| Online diagnostic fits (LOC_257 8 + LOC_C16 8) | 16 |
| Batch objective/gradient evaluations | 3,840 |
| Batch virtual target reuses, upper bound | 11,796,480 |

The offline reuses are internal optimization, not new external facts, and are paid as label reuse and compute. Report actual physical/cache arithmetic work separately from virtual warmup-weighted occurrences. The unique taught facts are 128/world; validation is 32/world; the sealed assessment is 50/world. Historical V149–V152 costs remain in the research ledger and are not reset as if these hypotheses had never been selected.

Native own/teacher-forced query pairs are at most 41,820: 40,320 acquisition/retention/immediate pairs plus 1,500 conditional held pairs. Lane C contributes 1,728 pairs and the technical reserve 128 pairs: **43,676 pairs / 305,732 output distributions** maximum. Diagnostic-head scoring is separate: 32 batch fits ×160 records plus 16 online fits ×224 records = **8,704 teacher-forced record evaluations / 26,112 target distributions** maximum. Total scheduled readout/evaluation distributions are at most **331,844**; training and optimizer distributions are charged additionally. Do not relabel evaluator targets as external training events or omit their selection-information cost.

Limit the entire future exercise to **7,200 summed CPU seconds, 10,800 wall seconds, one worker, one numerical thread and a 1-GiB resident-memory target**, with 600 CPU seconds reserved for audit/packaging. Both C and A/B share this clock. Schedule native groups before expensive diagnostic batch fitting so an unrelated optimizer cannot consume native admission time. No fit or group gets a new clock after sleep or a process replacement. Keep the computer awake only while an authorized controller is active.

Check admission conservatively at whole-record/fit boundaries. Preserve completed bytes, raw predictions, clocks and state; an interrupted application is marked partial, not counted as a committed full history. Resource/session pauses require explicit continuation. No automatic retry, grid widening, extra n or replacement world is permitted. Mark incomplete contrasts without fabricating a factorial estimate.

Parameter/memory accounting must use the actual KC count N from the pinned parent. A full-port instrument has 257d weights; C16 has 16d; RHO uses d=88 and KC taps d=N. There is no learned bias in these instruments. Report dense matrices, gradient and optimizer history, sparse-cache bytes and peak temporary storage. Active KC sparsity does not remove inherited global projection, sorting, state advances or clone costs. This protocol supplies no scalability or biological-energy claim.

## 9. Qualification, interpretation and delivery boundary

Before feedback, qualify front-end endpoint parity, event/read-time alignment, tap dimensions, native taught/no-write activity parity, all payloads/EOS, local/batch analytic gradient parity, zero-write controls, semantic checkpoint continuation, independent own-prefix ablations and receipt serialization. A scorer-only expected-answer witness must round-trip complete payload/EOS exactly, while taught-key lookup must recite its taught records; neither is a learned candidate. Also check, algebraically or by evaluator-only linear feasibility, that the fixed centered C16/q=0 port can make each of the 257 classes a unique argmax. No witness value is installed in a learner. An unrepresentable class produces PORT_NOT_QUALIFIED for the affected native/C16 lane, without modifying the codebook, masking classes or forcing EOS. These are data/interface feasibility checks, not a trained positive control; absence of native acquisition still cannot prove the task unlearnable. Include no more than the declared 64 technical records/128 query pairs. Solver numerical tests and synthetic accounting fixtures are technical, never addition science. A technical defect must be archived with its spent exposure; correcting its report may not replay a completed trajectory.

| Outcome | Interpretation |
|---|---|
| A shows a paired decay/permutation effect | A causal accessibility difference under the declared tap/probe on addition data; a universal information claim remains unsupported |
| RHO separates contexts, X exactly collides incompatible targets | Direct ambiguity introduced at that recorded encoding tap for those contexts |
| OFF_257 beats OFF_C16 | Output-family restriction matters under that finite fit; do not attribute the whole gap to plasticity |
| LOC matches its qualified OFF family | Online local categorical gradients can use these features at this budget; it remains an evaluator diagnostic |
| Native underperforms LOC_C16 | Native reachability/decay/update/readout path deserves investigation; parameterization and scale remain part of the contrast |
| C changes EOS without exact answers | q contributes to stopping, while acquisition still fails |
| Native taught acquisition/retention fails | A/B/C may localize the failure; no unseen-addition improvement has been shown |
| Ready native W gains on sealed orbits beyond causal/countermodel controls | Descriptive E3 within bounded addition; require a separate fresh confirmation before adoption |

E0 covers access, encoding and numerical attribution. E1 covers actual taught acquisition/retention. The only eligible E3 evidence comes from the actual native output on never-taught whole orbits with causal controls. External probes, a known arithmetic generator, text prediction and exact taught-key recall cannot supply that evidence. No scalar-feedback, mixed-arithmetic, NLP, E4 or biological-validity claim is allowed.

Deliver `DESIGN.md`, `SPEC.json`, `EVIDENCE_REVIEW.md`, `ASSETS.json`, `PROVENANCE.json`, `STATIC_CHECKS.json`, `STATUS.json` and the metadata-only `check_design.py`. Keep source-research caches local and excluded from publication. This version supersedes the pending V152-R1 Q0-training proposal only at the planning layer: C comes first, and Q0 retraining needs its own later decision. Preserve both historical files and the existing arithmetic-experiment pointer. **The later user request authorizes this bounded exercise as V154. It does not authorize a wider run or GitHub push.**
