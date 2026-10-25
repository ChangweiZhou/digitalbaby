# V152 — addition learning, output calibration and temporal credit

25 October 2026. **Design only. No model implementation, test run, three-cycle exercise, training, assessment scoring or GitHub push has occurred or is authorized by this delivery.**

The goal remains **better retained, complete own-byte/EOS generalization to unseen addition per total feedback and actual compute**. Native Full151/R_center remains the lineage being improved. A separate recurrent reference tests credit assignment; its success cannot be attributed to native MiniFly or adopted as its replacement.

## 1. What the FLM comparison changes

The useful lesson is to separate output calibration, temporal state and credit assignment. It does not establish a missing biological rule. FLM's language trainer uses a trained lexical interface and BPTT, while its separate eligibility implementation drops cross-neuron temporal credit. We inspected the pinned source rather than treating those two paths as the same learning result. [Language trainer](https://github.com/Kuberwastaken/flm/blob/0a6bee41832a424ddc2365970f85dbd13885e8c9/flm/language_train.py), [local eligibility](https://github.com/Kuberwastaken/flm/blob/0a6bee41832a424ddc2365970f85dbd13885e8c9/flm/local_learning.py).

V151 showed that native teaching can produce the intended one-step descent. Doubling B's effective dose largely removed its local disadvantage, but none of the 16 sampled trained-state answers became correct top choices. The next question is how feedback accumulates into retrievable knowledge through a learning history.

The attached proposal is refined in three ways:

1. **Return to addition.** Do not start another text study or reproduce FLM's language benchmark.
2. **Isolate changes within each family.** Compare a generic class-bias parameter in the native lineage. Compare temporal-credit rules on one identical small recurrent reference. A native-versus-reference gap changes too many mechanisms to identify credit assignment by itself.
3. **Keep integration explicit.** A local recurrent reference is not an integrated Full151 upgrade. No reference weights, ideal activations, Jacobians or BP errors are injected into the native learner.

V149's logistic-readout development results used substantially more training text and different optimization from V150/V151. They show information can be extracted under their own conditions, not a matched native learning ceiling. Its old feature caches are not required or retrained here. [V149 development report](https://github.com/ChangweiZhou/digitalbaby/blob/f73b7ff91b2da554f83cfc8e7f18c98a4b8a91a9/minifly_v149_input_ceiling_dev_20261025/cycles/CYCLE2.md).

## 2. Questions and fixed seven-condition roster

### Native lineage and output control

Retain V151's FE-PERM normalization, 16 canonical stores, native fast/slow/adaptation equations, event timing, fresh adaptation at record start, homeostasis off, and fixed original C16. For v∈R16, P=I−11ᵀ/16 and u=Pv/4, logits are `C u + q`. C is not trained.

| Condition | Persistent learning path | Purpose |
|---|---|---|
| N_A | Native old bit residual; q=0 | Unchanged V151 A teaching control through a full addition history |
| N_2B | Native categorical signed teaching, fixed factor 2; q=0 | Test the dose correction prospectively |
| N_2B_Q | N_2B plus a locally learned 257-class bias q | Isolate generic output calibration |
| Q_ONLY | Same learned class bias without native values | Frequency-only countermodel, not a MiniFly core |

At each observed reply byte/EOS, compute p and all errors before either update. Native A receives

`a_A = 0.1 s(n) [tanh(Pv/2)−C_y]/2`.

Native 2B receives

`a_2B = 0.2 s(n) P Cᵀ(p−onehot(y))/4`.

The factor 2 is fixed from the V151 analytical and measured dose comparison; it is not fitted on this experiment. N_2B_Q uses its own q-inclusive p for the native coefficient. In the two q-learning conditions, `q ← q−0.03 s(n)(p−onehot(y))`. Initialize q=0. No empirical frequency distribution is installed. The native signed interface remains unchanged; inadmissible coefficients are an error, never silently clipped.

Q_ONLY has time-independent logits during generation: under greedy decoding it either emits EOS immediately or repeats its top byte to the cap. Its inability to produce a nonempty byte/EOS answer is structural, so it is not a sufficient exact-answer shortcut control. The favorite-complete-reply and position-frequency countermodels below are necessary alongside it.

The q parameter is an explicit added output path. A lower NLL from it alone is not native knowledge acquisition. Compare the hybrid against Q_ONLY and perform read-only native-value and q ablations at the final evaluation (§7).

### Independent temporal-credit reference

| Condition | Same forward model and initialization | Credit update |
|---|---|---|
| R_BP | Small fast/slow recurrent rate model | Exact BPTT; diagnostic comparator only |
| R_LE | Identical model | Forward eligibility retaining neuron-local temporal history |
| R_LE0 | Identical model | Same rule with eligibility history reset at each byte |

No Transformer, attention, ELM, ridge solve or frozen-random-feature trained-head arm is included. BPTT is confined to R_BP and evaluator-side gradient comparisons. The three reference methods have the same 19,009 trainable parameter entries, input/output interface, forward dynamics, training order and optimizer policy. Thus their differences can test credit assignment within that model. They are not parameter-matched or dynamically equivalent to Full151.

## 3. Reference model and local rule

This is an FLM-inspired mathematical comparator, not a reproduction of its anatomical graph or language system. FLM's core motivates fast/slow states; its reported language ablations do not establish that a particular connectome is necessary. [Core source](https://github.com/Kuberwastaken/flm/blob/0a6bee41832a424ddc2365970f85dbd13885e8c9/flm/model.py), [ablation report](https://github.com/Kuberwastaken/flm/blob/0a6bee41832a424ddc2365970f85dbd13885e8c9/docs/LANGUAGE-CORE-RESULTS.md).

Use 64 units. Each row has its self-edge and three distinct other incoming edges, selected without task labels by an independent seeded RNG. Signs are fixed random ±1. For edge parameters θ initialized at zero, normalize signed magnitudes within each receiving row:

`W_ij = sign_ij exp(θ_ij) / sum_k exp(θ_ik)` on the fixed mask.

For raw byte one-hot x_t∈R256:

`z_t = tanh(U x_t + b + 0.5 W h_(t−1))`

`h_t = 0.75 h_(t−1) + 0.25 z_t`

`r_t = 0.95 r_(t−1) + 0.05 h_t`

`v_t = R [h_t;r_t]`; `logits_t = C P v_t/4 + q`.

All 256 bytes are valid payload; EOS=256 is an output control, not a payload byte. There is no BPE, ASCII-distance feature, arithmetic parser, operand register, carry channel, relation target, class mask, LayerNorm or tied lexical decoder. U's byte columns are independently initialized learned vectors. Constants fix a small, stable comparator: the fast-state recurrent Jacobian has infinity-norm bound at most 0.875, while both states remain bounded. This numerical choice is not a claim of optimal biological timing.

Trainable groups: U 64×256, b 64, θ 256 edges, R 16×128 and q 257: **19,009 entries**. Initialize U with zero-mean Gaussian std 1/√256, R with std 1/√128, and b/q/θ at zero. Fixed α=0.25, β=0.05 and recurrence gain 0.5 are not learned. All three methods share every initialization within a world.

R_LE uses analytic, forward-carried derivatives. Let d_i=0.25(1−z_i²), J_i=0.75+d_i·0.5W_ii. For a parameter local to receiving unit i:

`E_h(t) = J_i(t) E_h(t−1) + D_parameter(t)`

`E_r(t) = 0.95 E_r(t−1) + 0.05 E_h(t)`.

Direct terms are `D_Uik=d_i x_k`, `D_bi=d_i`, and

`D_θij = d_i·0.5 [W_ij h_prev,j − |W_ij|(W h_prev)_i]`.

The final term includes the derivative of incoming-row normalization. At an observed target, g_v=PCᵀ(p−onehot(y))/4. Instantaneous learning signals are `L_h=R_hᵀg_v`, `L_r=R_rᵀg_v`. Accumulate each core gradient from `L_h E_h + L_r E_r`; head gradients are `g_v[h;r]ᵀ` and `p−onehot(y)`.

This rule omits routes that leave a unit and return through other units over time. Its feedback signals require weighted broadcast from the output port. They are explicit assumptions of the independent reference, not a claim that native mushroom-body anatomy already implements them. Candidate local paths use no autograd through time or the encoder. R_LE0 sets old E_h/E_r to zero before each direct term; its forward h/r state remains unchanged.

For a reference record, hold parameters fixed through its prompt and reply, accumulate `sum_t s(n_t) CE_t` only at reply/EOS positions, then perform one SGD update at EOS. This makes the BP-versus-eligibility objective and update timing identical. R_BP differentiates that exact weighted objective. Native updates remain online per reply event as in V151. This cross-family timing difference is disclosed, not presented as an isolated credit effect.

Use plain SGD without momentum. Clip each accumulated gradient component to [−1,1] and project θ to [−3,3] after updating, identically in the reference methods. Record clipping/projection rates and compare raw and applied gradient directions separately. Select one shared reference η from {0.01,0.03} on the calibration world (§5). No per-method rate search or Adam-state confound.

## 4. Data, feedback and own output

One task: ordinary unsigned addition with operands 0…19. Prompt bytes are the decimal strings `a+b=`; expected reply is decimal `a+b`, then EOS. Arithmetic exists only in the evaluator/data generator. The learner receives bytes and observed replies, never operands as numbers or arithmetic relations.

Each world permutes the 210 unordered operand orbits `(i,j), i≤j`, using a separately named PCG64 stream derived from SHA256 of `V152|world|split`. Assign the first 64 to OLD, next 64 to NEW, next 32 to validation, last 50 to assessment. Choose one orientation independently per orbit and present only that orientation. **No reverse-form augmentation** is supplied. Holding the whole orbit out is an evaluator leak prevention rule; it is not a model feature or a supplied commutative law.

World IDs: calibration 202610255201; assessment 202610255211, 202610255212, 202610255213. Graph/weight initialization and each pass's order use independent named RNG streams. Before any feedback, implementation must hash-lock split membership, orientation, sequence order, targets and configuration. No worlds are discarded for unfavorable answers or model outcomes. Assessment target scores remain sealed until the acquisition/retention report is frozen.

All arms use the same raw prompts and target positions. Observe a target only after predicting it; then teacher-force the actual observed byte as next input. A guess is never fed back as a correct observation. Do not predict/train prompt bytes as an extra objective in selected arms. Transient sequence state resets at each query boundary, while learned parameters/native memory persist. Generic query/reply boundaries are out of band; no payload byte is reserved as a delimiter by the architecture.

Warmup is preserved: `s(n)=1/16+(15/16)min(n/1024,1)`, n being previous externally observed reply/EOS targets, including repeats and plasticity-off controls. Expected outputs remain available throughout this bounded experiment; blurry or scalar-only learning is not tested. Count prompt bytes, reply payload, EOS targets, repetition, calibration and probe compute separately.

Own output is greedy over all 257 classes, with no forced EOS or seen-vocabulary renormalization. Cap at four decisions (correct answers require at most two bytes plus EOS). Missing EOS is CAP/failure, not a model-chosen stop. Exact success requires the complete canonical reply bytes and its own EOS; leading zeros, extra bytes or an empty answer fail. Teacher-forced NLL and class margins are diagnostics, not substitutes. Evaluation uses independent clones and never updates persistent learning state.

## 5. Fixed learning history, calibration and causal branches

Each phase presents its 64 records for eight independently ordered passes. This supplies at least 1,024 observed reply/EOS targets by the end of OLD, so the inherited 1,024-target warmup completes before NEW; eight passes are a dose/budget choice, not an outcome-selected training length. Checkpoints are after passes 1,2,4,8; no adaptive extra pass is allowed. P=64×8=512 record presentations per phase.

For each model/world create two identical newborns, then form three causal histories:

- **W:** learn OLD, then learn NEW.
- **N_OLD:** share W's actually learned OLD prefix; then receive the same NEW bytes/expected outputs/timing with every plastic update disabled.
- **SH_OLD:** receive OLD's same input/expected output/timing with every plastic update disabled; then learn NEW.

Plasticity inhibition covers the native stores, bias and every reference parameter. Observation counters still advance, so a control does not restart warmup. Native decay/adaptation continue normally. Reference weights do not acquire native passive synaptic decay; cross-family retention differences therefore cannot be attributed to temporal credit alone.

The shared OLD prefix is executed once for W/N_OLD. Report three correlated full histories, but charge **five physical phase streams per model/world**, not six fictitious independent runs or only two training phases. Each group has 5P=2,560 actual record applications; 3P involve parameter updates and 2P inhibit them. Save/checkpoint whole-record boundaries without replaying completed records after interruption.

Calibration runs only R_BP/R_LE/R_LE0 at both rates on the one calibration world: six groups. Choose η minimizing the equal-method mean final-W validation reply BPS; lower η breaks ties. Lock this one η before assessment. Validation labels inform configuration selection and are charged as such; no calibration weights carry into a fresh assessment world.

Readiness requires the selected R_BP to show increased complete taught-answer acquisition over its pre-feedback baseline in both phases (an already-perfect pre-feedback phase also qualifies). This minimal check establishes that the comparator can learn something at this budget, not that it has solved addition. If it fails, report `REFERENCE_NOT_READY`; do not widen rates, change width or launch the assessment cohort. This is a failed bounded assay, not proof that local learning is impossible.

If ready, run all seven conditions on all three fresh assessment worlds, with all three causal histories. Scores from a poor native acquisition arm remain visible. No model is selected or adopted using the 50 assessment relations.

## 6. Acquisition, retention and sealed unseen-relation scoring

Record an immediate own-answer probe after every record that actually updates parameters. These probes are essential: an answer may be acquired immediately and lost before an end-of-pass test. Also measure:

1. Initial OLD accuracy before any feedback.
2. OLD accuracy after OLD passes 1,2,4,8; final no-prior OLD accuracy.
3. NEW accuracy in all three branches before NEW feedback.
4. OLD/NEW accuracy in all branches after NEW passes 1,2,4,8.
5. OLD/NEW accuracy after an additional fixed idle horizon of 128 native byte steps (128×DELTA); the reference receives the corresponding no-input time designation without parameter updates.

The idle horizon is a reproducible operational probe, not a biological retention threshold. Native rest changes its actual state; a reference with no across-record parameter decay may remain unchanged. Report write-associated loss against N_OLD separately from passive-time loss.

Freeze these acquisition/retention summaries and selected settings before opening assessment scores. Then score the first complete own answer on each of the 50 never-taught orbits in W/SH_OLD/N_OLD, at the post-idle state. Always generate own output before a separate clone is teacher-forced for NLL. No held reply is taught, used for gradient diagnostics, queried at intermediate checkpoints, or used to fit a setting.

Primary report: final exact unseen-answer accuracy paired with cumulative observed targets, measured compute and retained OLD accuracy. Use actual byte/EOS counts, not only record counts. Intermediate learning curves cover taught acquisition and retention; this single-final-assessment design cannot establish an unseen-answer learning curve. For those acquisition/retention curves, fixed cost comparison points are 15,60,240,960 training CPU seconds per model/world group: use only the latest already completed checkpoint within a point, mark unavailable points, and never extrapolate or grant a faster method more data beyond this roster. Keep the final sealed unseen score at its actual cost; do not score earlier assessment checkpoints to fill comparison points. Also report end-of-budget and total-cost accounting including evaluation and calibration overhead.

Include exact-key lookup, canonicalized-pair lookup, a training-only favorite complete reply (including EOS), and position-only reply-symbol frequency countermodels. The latter learns separate byte/EOS counts at each generic reply position and generates without a forced stop; it receives no operand feature. These are deterministic evaluator summaries of already logged observations, not additional trained readout candidates. Charge their computation and state; do not perform unbudgeted extra training passes or target queries. Canonicalization is evaluator-only; whole-orbit holdout prevents a seen reverse pair from explaining success. The stronger lookup controls may recite OLD/NEW; neither should be mistaken for relation learning. No FLM BPB or V149 text score is subtracted from an addition score.

## 7. Attribute bias and temporal-credit effects

For N_2B_Q, final OLD/NEW/assessment evaluation includes two additional read-only generation conditions: zero native centered values, or zero q. Each condition generates its own continuation from a fresh clone; do not reuse the full model's guessed prefix. Disabling a contributing path tests necessity at that state, not proof that the remaining path learned the rule.

A gain in likelihood that Q_ONLY or the native-value ablation reproduces is output calibration, not reusable native knowledge. A native improvement requires its own complete answer, a contribution beyond the frequency-only controls, and the prior-teaching/retention evidence. No hidden learned readout evaluator answers for the native core.

For credit assignment, compare **only identical parameter coordinates** in the recurrent reference. At R_LE snapshots after OLD passes 1/8 and NEW passes 1/8, in each assessment world (12 snapshots), use the first prespecified training record of the corresponding phase:

- Compute exact BP, LE and LE0 gradients at the same saved parameters on that same record.
- Report cosine and norm ratios for U, b and recurrent edges separately. Head gradients are reported separately: their large exact contribution must not conceal poor core credit. Zero-norm cosines are undefined, not converted into favorable zeros/ones.
- On disposable clones compare ZERO, LE core-only, LE head-only, LE joint, and a BP joint update whose blockwise norms are matched to LE's applied update using gradient magnitudes only. No loss outcome chooses a matching factor. Report post-projection norm mismatch or an unrepresentable match without retuning.
- Probe immediate same-record change and a fixed already-taught OLD record on read-only clones. Use no assessment targets. Better gradient alignment is not itself a generalization result.

This is 36 labeled gradient passes plus 60 labeled update/control passes: 96 diagnostic record applications, at most 288 target applications. Preflight has at most 64 additional technical records/192 targets. A head-only one-step counterfactual is an attribution control at a learned state; it is not an ELM/reservoir training route. Do not calculate a cosine between recurrent θ and native fast/slow arrays: those are different parameter spaces. Native integration must be tested as its own actual circuit, not inferred from an analogy.

## 8. Interpretation and evidence boundary

- N_2B versus N_A tests a prospective combined objective-and-dose correction through a native learning history; it does not independently identify the full-history effects of the objective and gain, or test a new connectome.
- N_2B_Q versus N_2B and Q_ONLY tests the added generic output parameter, with explicit native-value ablation.
- R_BP versus R_LE tests exact versus truncated temporal credit on that reference model. R_LE versus R_LE0 tests the contribution of eligibility history. If BP alone learns, this localizes a credit gap within this reference, not a mathematical impossibility for Full151.
- If all reference methods learn similarly, readout/input dynamics may dominate at this budget; use the core/head counterfactual before crediting eligibility.
- If native methods acquire individual answers then lose them under NEW, the temporal pattern supports an interference/retention problem. If immediate acquisition is absent, an end-of-history failure cannot be called forgetting alone.

E0 covers numerical/interface/gradient checks. E1 covers taught-answer acquisition and retention. Only never-taught orbit accuracy, a favorable prior-teaching W versus SH_OLD contrast beyond recitation/frequency controls, and delayed/intervening retention can support a limited E3 claim. No E4, general-language, scalar-feedback, biological-validity or anatomical-topology claim is permitted. R_LE results are independent recurrent-reference evidence, never native evidence.

Three assessment worlds give descriptive development evidence. Do not treat byte positions, repeated probes, branches or commutative partners as independent replications. Report each paired world and effect size. A promising native development result requires increased retained exact unseen accuracy over N_A, positive prior-teaching evidence and no loss of retained OLD relative to that control at the declared budget; there is no invented noninferiority slack or stand-alone loss cutoff. Mixed outcomes remain mixed. Even a favorable result requires a separately authorized confirmation; no automatic adoption.

## 9. Resource/exposure caps and stopping

| Planned maximum | Count |
|---|---:|
| Calibration model/world groups | 6 |
| Assessment model/world groups | 21 |
| Correlated causal histories | 81 |
| Physical main/calibration record applications | 69,120 |
| Credit diagnostic / technical record applications | 96 / 64 |
| **All record applications** | **69,280** |
| **Reply/EOS target applications, upper bound** | **207,840** |

These maxima include plasticity-off target observations. Actual counts depend on one- versus two-byte answers; report exact revealed targets and unique facts separately. Main/calibration records comprise 5×512 applications per group; shared prefixes are counted once physically and disclosed in the history count.

A full own-output plus teacher-forced evaluation is one scheduled query pair. Caps are 87,822 ordinary assessment queries, 3,204 hybrid ablation queries, 24,768 calibration queries, 120 credit-counterfactual queries and 128 preflight queries: **116,042 query pairs**, at most seven output distributions per pair (four generation decisions plus three true reply/EOS positions). Teacher-forced and model-guessed inputs are not relabeled as training observations. Log query bytes, native bank/clone/readout work and CPU separately. Configuration-selection labels are an additional information source, even though validation queries perform no writes.

Limit the whole exercise to 7,200 summed CPU seconds and 10,800 wall seconds, one worker/one numerical thread, 1 GiB resident-memory target. Reserve 600 CPU seconds for audit/packaging; admission must account for elapsed calibration and prior stages. A group/source does not receive fresh budget after restart or sleep. Check conservatively before a record/pass boundary. If a hard cap interrupts a record, mark it partial and exclude it from committed aggregates. No retry, rate-grid expansion, new arm, larger n, extra pass or automatic continuation is allowed. Resource/session pauses preserve state and require explicit authorization.

Measure update, readout, inactive-state advance, encoding, eligibility maintenance, gradients, cloning, hashing and I/O separately. The reference has 128 transient state values and 33,408 eligibility values (267,264 bytes / 261 KiB in float64); output/core parameters and exact-BP activation/graph storage are additional. Dense input eligibility and native whole-KC encoding/sorts remain computational costs. This small experiment does not establish a scalable implementation or biological energy advantage.

## 10. Planned qualification and delivery status

Before feedback: lock parent sources, data streams, graph/initializations, all model settings and result schemas. Test all 256 legal payloads and EOS, control boundaries, no mask/forced stop, clone independence, read-only scoring and exact counter resume. Validate analytic output gradients by finite differences; BP by finite differences on a tiny record; LE state parity with BP; exact local eligibility where cross-neuron temporal routes are absent. General recurrent LE is approximate and must not be tested against a false exactness requirement. Candidate local paths must fail a guard if autograd/BPTT is enabled. All technical target exposure fits within the stated preflight reserve.

A future authorized three-cycle exercise would implement/audit those contracts, run only the finite calibration and gated assessment roster, then independently reconcile saved predictions, byte/EOS choices, sources, clocks, exposures and costs. It must publish failures and partial groups, not only favorable final curves. Nothing is launched by this design.

Deliver DESIGN.md, SPEC.json, RESEARCH_REVIEW.md, PROVENANCE.json, STATIC_CHECKS.json and STATUS.json. Record V152 as a design-only next proposal alongside V151's completed diagnostic. Preserve the arithmetic pointer, frozen historical experiments, user adoption history and all source checkpoints. Raw data and source-research caches stay local. The next decision remains about a native learner for unseen addition; copying FLM's language model is outside scope.
