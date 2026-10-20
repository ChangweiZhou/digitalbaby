# MiniFly v3 Generic learning from arithmetic pretraining

20 October 2026. **Design delivered; not implemented, qualified or launched.** This is the v3 revision of Claude's 18 October v2 proposal. It authorizes no training, automatic continuation, adoption or GitHub upload. Historical experiments and their verdicts remain unchanged.

MiniFly must learn arithmetic relationships from pretraining examples. The model supplies generic sequence state, native memory and plasticity; it receives no arithmetic procedure. Warmup uses small, gradually increasing learning intensity, with explicit measurement of acquisition, interference, representation drift and subsequent learning capacity. The first behavioral experiment remains **addition only**: retained exact own-output answers on never-taught operand combinations, per all training presentations and actual compute.

## 1 Decisions and claim

The candidate remains in the Full151 native-memory lineage. Learned generic sequence circuitry may change how byte experience reaches that memory. Every scored output must be read through the operative native stores. There is no separate answer predictor, independent slow-core answer head, confidence arbitration, arithmetic parser or raw-answer lookup fallback.

The two experimental questions are:

1. Does gradually introducing plasticity improve retained unseen-addition learning compared with steady teaching at the same scheduled cumulative learning dose?
2. Does learning the generic sequence representation improve that result compared with the identical circuit whose representation weights remain frozen at birth?

E0 interface qualification and E1 taught acquisition/retention are necessary diagnostics. The intended result is bounded E3 reuse on never-reinforced addition combinations after delay and further addition learning. This design does not claim four-operation competence, learned brackets, broader pretraining, NLP or E4. Approximate magnitude, first-byte accuracy and EOS accuracy cannot replace exact answers as the endpoint.

## 2 Sources and version lineage

| Source | What v3 inherits or learns from | What v3 does not inherit as a capability |
| --- | --- | --- |
| [Claude MiniFly-CLS v0, commit 2c217b2](https://github.com/ChangweiZhou/digitalbaby/tree/2c217b2716fc1cc0796f73a9fcc288f8aca864da/minifly_cls_dev_20261017) | `fastmem.py`: canonical Full151 births, 16 native output compartments, FE-PERM, one-sided KC homeostasis, per-record adaptation reset and residual signed teaching | Its separate recurrent slow core, raw episode cache and arbitration are excluded. Its 84–88% taught recall and 81–85% after a day are developmental E1 evidence, not v3 results. |
| [Claude v2 proposal](https://github.com/ChangweiZhou/digitalbaby/blob/2c217b2716fc1cc0796f73a9fcc288f8aca864da/minifly_v2_design_20261018/DESIGN.md) | Generic sequence persistence, gating and eligibility as hypotheses | Dedicated operand registers, place alignment, carry/borrow cells, operator attractors, arithmetic step tables, result reversal and arithmetic-controlled stacks are removed. Nothing in v2 was already executed. |
| [Native-bind V2.1 implementation](../native_context_binding_dev_v2_1_20261017/bind_kernel.py) and [postrun review](../PROJECT_DOCUMENTS/NATIVE_BIND_V2_1_REVIEW_20261017/REVIEW_STATUS.json) | Prior generic gated context circuitry and the adaptation/readout diagnosis; source continuity must be explicit | Its BPTT update is not reused. Generic recurrence is not a new discovery, and its negative result is not erased. |
| [Addition lineage comparison](../addition_core_lineage_comparison_design_v1_20261014/EXPERIMENT.md) | OLD/NEW/HELD orbit separation, causal histories, generic byte port, exact generation and cost accounting | KEEP/G62 are independent recurrent references, not native upgrades. Their performance does not transfer to v3. |
| Historical COV-R/COV-SUPPORT | Historical engineering reference and user-adoption record remain intact | No numerical features, additive decoder, operand parser or ordinal renderer is imported into v3. |

V3 is a native-memory-derived learner with a new locally plastic sequence adaptor. It is not the unchanged Full151 or complete R_center runtime. Fixed sensor identities and the output port are engineering interfaces; semantic sequence representations are the learned candidate component.

The upstream code is on Claude's branch, separate from the current Codex checkout. Implementation must retrieve the pinned source and record all required native assets and hashes. A prose source citation is not an executable source lock. The upstream extracted native package is not wholly present in Git; its availability and identity must be qualified before a run.

## 3 Architecture boundary

The model receives generic framed byte records:

`BEGIN, input payload, REPLY, observed answer payload, EOS`

BEGIN, REPLY and EOS are typed controls outside the 256 payload values. Payload `=`, digits, punctuation, spaces, newline and NUL have no privileged processing. BEGIN resets temporary sequence state and record-local eligibility, not learned weights or native value stores. The inherited adaptation reset is a separate, declared protocol override, applied consistently in teaching and recall.

Allowed model components are generic byte sensors, recurrent neurons and gates, fixed sparse connectivity, local plasticity, homeostatic competition and native value memory. Forbidden components include attention/Transformers and equivalent content-addressed key/value retrieval; ELM designs or a separate random-feature answer solver; BPTT, autodifferentiated temporal credit or hidden gradient optimizers; arithmetic features, semantic operand slots, numeric error rewards, intermediate answers and worked solution traces.

No learner module imports the arithmetic generator, evaluator, split identity, world seed, answer length, operand roles or task IDs. The evaluator may parse and compute arithmetic; those values never enter the learner except as declared observed answer bytes.

The architecture has no neuron designated as a digit, operator, carry, place, decimal boundary or bracket. Such selectivity may emerge and be studied after training, without modifying the circuit or scoring rule.

## 4 One native output path

Retain the v0 generic byte receptor table and its FE-PERM trace as the source interface. Retain the existing 257-class, 16-component C16 output port, canonical native anatomy, per-side KC competition, native fast/slow/adaptation equations, reader and signed-write primitive. Do not simultaneously switch to raw-bit sensors or a new output code: that would confound plasticity and warmup with another interface change.

The port's fixed class identities supply no arithmetic ordering. A fixed identity code does not by itself prove information loss; collisions and usable sequence information must be tested. The learned candidate must change its input/recurrent representation weights, so a fixed encoding alone cannot earn a learned-representation claim.

Answers follow this route in every arm:

`bytes → FE-PERM and generic sequence state → receptor drive → native PN/KC circuit → native stores/readout → byte or EOS`

There is no answer-producing connection directly from the generic state to the output classes. The same encoding is used for prediction, native writing and recall. Capture a prefeedback frame once; the target-dependent write uses that frame's context/address and clock. Do not recompute an address after changing thresholds or representation weights and call it the original prediction.

Generation advances temporary sequence/native state on an isolated read view. It cannot alter persistent weights, homeostatic statistics, optimizer state, native learning stores or training counters. Return payload, `ended_with_EOS` and `truncated` separately. The output cap is 16 payload bytes for every arm; hitting it without EOS is incorrect, even if a prefix matches the answer.

The input cap is 256 payload bytes per record, reported as an explicit resource limit; over-cap input returns `INPUT_LIMIT` instead of silently dropping a prefix. The addition prompts in this experiment have four payload bytes, so longer-input competence is not established by its result.

### Generic gated sequence state

Use 64 generic units, following the size of the existing native-bind context adaptor rather than adding a large independent model. For a byte's receptor identity r = R_TABLE[byte] and previous state h:

```text
v = tanh(W r + U h + b)
g = sigmoid(G r + H h + c)
h_next = (1 - g) * h + g * v
```

W, U, G, H, b and c are the candidate plastic parameters. Recurrent U/H diagonals are zero and remain absent; the explicit hold term supplies self-persistence. Other connections in this bounded 64-unit adaptor are present; there is no topology/pruning search. Initial weights and connection masks are paired across arms and source-locked. No unit has an assigned arithmetic role.

A fixed, label-independent projection maps h into a bounded additive perturbation of the 88-channel FE-PERM receptor drive before native PN/KC encoding. Normalize the fixed projection rows at birth. The perturbation bound and normalization are identical in frozen and plastic arms and must pass numerical/input-access qualification. The projection cannot read target bytes before arrival or select a position/memory item by query scoring. Its exact scale and connectivity are implementation-lock items in Section 7, not outcome-dependent tuning variables.

Use no silent THINK steps in this first comparison. They are permitted in principle but would add another compute and state-transition factor. The working-state update occurs once per real payload byte; EOS is a control, not an invented sensory byte.

### Local credit to representation weights

For target class q, first read native component values z and form the inherited error-corrective signal:

```text
s = (tanh((z - mean(z)) / T) - output_code[q]) / 2
```

T remains 2, inherited from v0. Sixteen native compartments receive their own signed coefficient, scaled by the teaching-intensity multiplier. The target arrives before this computation; no held-out or generated answer supplies s.

Send s through a fixed, zero-mean feedback projection F into the 64 generic units: L = F s. This is a proposed feedback-alignment teaching signal, not the exact gradient of native memory performance. F is independent of forward weights, arithmetic, data split and targets; initialize its entries to paired random signs divided by sqrt(16), then freeze it.

Each plastic afferent keeps a bounded record-local eligibility. With other neurons' states treated as local presynaptic observations:

```text
e_w(t) = (1 - g_i(t)) * e_w(t-1) + direct_local_derivative(h_i(t), w)
```

For weights into the candidate drive, the direct term is `g_i * (1 - v_i^2) * presynaptic_activity`. For weights into the gate it is `g_i * (1 - g_i) * (v_i - h_i_previous) * presynaptic_activity`. Bias terms use presynaptic activity 1. Absent self edges make the retained diagonal hold term explicit; cross-neuron recurrent credit is omitted and must not be described as exact BPTT.

Accumulate `-L_i * e_w` at observed answer/EOS events, average by the number of such events in the record, and apply one bounded parameter update after EOS. Use eligibility from the same prefeedback frame as the prediction. The incoming target byte is fed to the temporary sequence state only after that prediction, residual and native write; it cannot influence its own prediction or write address. Forward parameters are held fixed within that record. Eligibility is not cleared between prompt and answer; it is cleared at the next BEGIN. No earlier hidden states are traversed backward. No global Adam/RMS optimizer is introduced.

This local approximation may fail, as the earlier e-prop preparation did. The new hypothesis is that gated persistence, the native error signal and gradual representation movement can do better; the design does not claim this has been established. Finite-difference checks may establish direct local terms, not equality to an intentionally different full temporal gradient.

## 5 Four comparison arms

All four arms contain exactly the same sequence adaptor, fixed projections, native compartments and birth parameters. The frozen arms are causal lesions of this candidate, not alternative ELM designs or adoption candidates. This makes behavioral attribution to learning the representation possible without also changing capacity at birth.

| Arm | Generic representation weights | Teaching schedule |
| --- | --- | --- |
| F-C | Frozen at paired birth | Steady, cumulative-dose-matched |
| F-G | Frozen at paired birth | Gradual warmup |
| P-C | Locally plastic | Steady, cumulative-dose-matched |
| P-G | Locally plastic | Gradual warmup |

F-C/F-G still learn through native value writes and inherited homeostasis. P-C/P-G add local representation updates. P-G versus P-C tests warmup; P-G versus F-G tests representation learning. The factorial interaction is descriptive unless independently confirmed. All arms see the same record order, timestamps, externally supplied targets and measurement points.

Separately qualify the exact v0 FULL recipe, with the adaptor disabled, as the source/parity and taught-memory positive control. It is not a fifth candidate or an efficacy ranking arm. Its published results cannot be pasted into the new comparison. Do not add a BPTT/GRU answer head to rescue a failed local-learning arm.

## 6 Gradual warmup and protection against destructive learning

Slow warmup means limiting state movement per exposure and controlling its accumulation. Wall-clock waiting, simulated rest and smaller learning intensity are three different interventions. Do not add idle brain time to mimic a lower learning rate: native decay and adaptation respond to that time.

Use a predetermined smooth schedule indexed by the number n of initial OLD presentations, including presentations in a no-write history:

```text
u = min(n / H, 1)
a(n) = a0 + (1 - a0) * (3*u^2 - 2*u^3)
```

The initial engineering settings are a0 = 1/16 and H = 960 presentations, corresponding to the first 16 complete OLD passes. They are declared experimental settings, not biological constants or statistically derived efficacy thresholds. H covers the early code-formation period suggested by v0's evolving acquisition curves; the lower starting intensity tests whether gentle exposure preserves useful learning. These choices may be rejected by technical qualification but cannot be quietly retuned on pilot/confirmation HELD results.

In gradual arms:

- Native error-corrective coefficient: `a(n) * s`.
- Representation step size: `eta_max * a(n)^2`, so representation movement begins more slowly than value teaching.
- One-sided homeostatic threshold increment: multiply the inherited increment by a(n). Keep its frequency-estimator beta at the inherited 0.01; do not silently change the observation window.

The v0 homeostatic coefficient 0.05, excess-frequency threshold 5 times native active fraction, native time constants and signed-write equations are inherited and recorded. Local parameter-row update caps are common to both plastic arms. Native |s| is bounded by 1 before schedule scaling.

In steady arms, use the exact arithmetic mean of a(n) across all 7,680 planned OLD presentations for native/homeostatic scaling, and the mean of a(n)^2 for the plastic representation rate. Thus the *scheduled* cumulative multiplier is equal at the end of OLD. Verify the sums mechanically before training. Actual state changes and clipping frequency can still differ; record them rather than claiming identical realized weight movement.

After the OLD budget, every arm uses multiplier 1. Do not restart the ramp when NEW arrives: the model receives no phase or task identity. Fixed-budget intermediate learning curves intentionally reveal the early cost of warmup as well as any later benefit.

A record is presented once at a time, in shuffled complete passes. Do not batch thousands of full-table updates into an uncounted sleep period, oversample current errors, skip apparently easy records or increase training until a desired score appears. For this first comparison there is **no external episode cache and no additional sleep replay**. Repeated pretraining examples are all in the explicit record stream and count every time. There is no self-generated-answer teaching.

Protection is tested through bounded updates and measurement, not assumed. Track old taught recall, fixed-prompt KC-code drift, active counts, frequent-cell concentration, homeostatic thresholds, native fast/slow magnitude, local row-update norms, clipping and subsequent NEW acquisition. Report whether warmup preserved learning capacity or merely suppressed learning. These measurements cannot replace HELD accuracy.

## 7 Numerical calibration and implementation lock

Do not fabricate precise new learning rates from unrelated optimizers. Before any pilot, an engineering-only qualification must determine and freeze:

1. `eta_max`, eligibility clipping bound and per-row relative update cap for the local rule; initialization scales for the gated adaptor.
2. Fixed forward-projection amplitude and sparse connectivity, including full initialization algorithms and seeds.
3. Exact float dtype, rounding/tie behavior, native assets and all inherited coefficients.
4. The common compute budget, checkpoint-query plan, resource estimate and scientific roster.

Use at most two separately registered engineering worlds, absent from all pilot/confirmation sets. Engineering checks may use taught training records and fixed generic byte fixtures, never scientific HELD labels. Choose values that pass finite-state, nonzero-credit, bounded-update, input-access and continuing-acquisition checks; the selection rule and finite candidate list must be written before those checks. Retain all failed attempts and their costs.

If no value in that finite list qualifies, report implementation infeasibility and revise the design explicitly. No unlimited parameter search, automatic fallback to BPTT or fourth hidden qualification cycle. Save a `PARAMETER_LOCK.json` and executable source/asset lock before the pilot. This document specifies the experiment; absent those measured locks it is not a launch-ready protocol.

## 8 Arithmetic pretraining data and exposure audit

Retain the lineage dataset: operands 0–9, decimal answers 0–18, raw byte prompts such as `3+8=` and answers such as `11` plus EOS. The learner does not know those bounds, syntax or the answer length.

Each world partitions 55 commutative orbits into:

| Set | Orbits | Ordered prompts | Use |
| --- | ---: | ---: | --- |
| OLD | 10 diagonal + 25 off-diagonal | 60 | Initial pretraining |
| NEW | 10 off-diagonal | 20 | Further addition learning |
| HELD | 10 off-diagonal | 20 | Never taught in either direction |

Reuse the deterministic lineage split algorithm and its OLD answer-coverage/minimum-degree constraints. Freeze splits, record permutations and initialization RNGs independently before learner execution. Audit that both directions of every HELD orbit are absent from all pretraining, qualification reuse, target caches and replay. HELD permits familiar digits and answer classes; it tests new combinations, not new symbols or extrapolation.

OLD runs for 128 complete passes: 7,680 presentations. NEW runs for 32 complete passes: 640 presentations. OLD checkpoints are 0, 60, 120, 240, 480, 960, 1,920, 3,840 and 7,680 presentations. NEW checkpoints are 0, 20, 80, 160, 320 and 640. No training example is selected by its arithmetic difficulty or the learner's evaluation errors.

The finite maximum is a bound, not a promise of enough training. Every repetition and every expected-output byte counts. No auxiliary tally/count-list stream, arithmetic trace, numerical error loss, commutativity augmentation or automatically generated relation is added. Reliable complete answer feedback is retained throughout this experiment. Sporadic/blurry and scalar-only feedback are future stages, not simultaneous changes or achievements claimed here.

## 9 Causal histories and chronology

Each world/arm job contains three continuous, paired histories; they are correlated histories, not three independent worlds:

| History | OLD | NEW |
| --- | --- | --- |
| W | Correct observed answers; all arm-permitted learning | Correct answers; all arm-permitted learning |
| N_OLD | Same bytes/timing, with all persistent learning disabled | Correct answers; all arm-permitted learning |
| SH_OLD | Learn a predeclared permutation of complete OLD answer strings | Correct answers; all arm-permitted learning |

N_OLD disables native value writes, representation updates and homeostatic-statistic/threshold learning. It still advances the same clock and temporary state, uses the same adaptation-reset protocol and counts exposures/schedule age. It must not acquire answer information through another persistent route. At NEW, learning is enabled without resetting its birth weights, clock or schedule counter.

SH_OLD preserves the target-string multiset and total target bytes in each complete OLD pass but shuffles prompt/answer pairings using its own predeclared RNG. Save accuracy against **both the labels actually taught and true addition**, separately. Low true-addition accuracy does not show failure to memorize shuffled labels. Expose no SH identity to the learner.

The main chronology is OLD → 24 h simulated idle → NEW → 24 h simulated idle. Read-only OLD/NEW/HELD probes occur at the registered checkpoints and both rest boundaries. Each BEGIN, payload byte, REPLY and observed EOS advances the inherited nominal 30/14-second clock; evaluation on clones does not advance the production history.

From each first-rest checkpoint, also derive a read-only time-only endpoint advanced to the final main-history time, without NEW sensory exposure or learning. It supplies a time-matched retention diagnostic and no additional training history. Main versus time-only differences combine NEW sensory and write effects; do not attribute them solely to writing without a sensory-only control. No branch is counted as independent replication.

## 10 Endpoints, countermodels and decisions

Exact accuracy requires the entire self-generated payload and the model's own EOS. Save full outputs and stopping flags. Report E1 OLD/NEW accuracy separately from HELD; first-byte, EOS, length, near-miss magnitude and representation geometry are diagnostics only.

The fixed primary endpoint is final HELD accuracy after NEW and the second rest. Sample efficiency is the normalized trapezoidal HELD area over `log2(OLD presentations + 1)` at every registered OLD checkpoint. Compute efficiency is the normalized stepwise HELD area over cumulative production CPU up to a common Cstar; use the last actually measured checkpoint policy, without interpolating unobserved capability. Determine Cstar from the median complete OLD production CPU of P-C over the two engineering worlds and lock it before the pilot. Do not extend learning to fill unused CPU.

For causal reuse report final W minus N_OLD and W minus SH_OLD, as well as absolute W scores. N_OLD receives NEW teaching, so these contrasts test the contribution of the earlier learning episode under matched later experience. Prespecify contrasts P-G minus P-C and P-G minus F-G for the primary, sample and compute endpoints. Report NEW acquisition and OLD retention alongside them; stability bought by disabling further learning is not a successful learner.

Countermodels, evaluated outside the learner, include raw exact-key lookup; commutativity-canonicalized lookup; training-answer frequency/constant output; first/second-symbol marginal predictors; and a learned position-separable categorical predictor. Raw/canonical lookup has no labeled HELD key; any default predictions still have to be measured. A simple relationship learned from OLD is legitimate reuse, so a successful learned shortcut narrows the explanation rather than automatically invalidating E3. An additive numerical fit is an explicitly engineered task-solvability witness, not proof that arbitrary models uniquely identify the rule or a candidate to be deployed.

Native-state and representation-freeze interventions on saved development checkpoints may diagnose output attribution. They are read-only, cannot train on HELD and must remain separate from the primary result. Persistent native state and learned representation parameters must both be listed; do not count unused inherited arrays as proof of an operative native mechanism.

Four fresh pilot worlds are a descriptive feasibility study. Confirmation, if separately authorized, uses 16 untouched paired worlds, one registered paired initialization per world, and the same four arms. The unit of analysis is the world; orbit directions, histories, outputs and checkpoints are not independent samples.

For a **P-G comparative advantage**, require positive paired effects for both predeclared contrasts on final HELD and both efficiency areas, with an intersection-union test: use the maximum of the six one-sided world-paired p-values at 0.05. Also require positive final W−N_OLD and W−SH_OLD causal effects; publish their intervals and apply the same intersection rule to any claim requiring all eight inequalities. Missing or zero-variance/all-tie cases must have explicit conservative test handling. No arm is selected post hoc as an alternate winner under this rule. Publish all arm scores, paired intervals and world plots.

This is a relative-improvement claim, not a claim that addition is solved. Modest absolute accuracy stays modest. Before any adoption decision, present exact absolute accuracy, OLD retention and NEW acquisition; no automatic core adoption follows even from a passed comparison. If development gives no positive final-HELD point effects for either main contrast, no positive prior-learning contrast, or numerical/provenance failure, report NO_ADVANCE. If all directional effects are positive and all technical checks pass, a confirmation may be proposed; pilot directions are not statistical confirmation. Do not increase n or training dose after an inconclusive result.

## 11 Finite scope and resources

| Stage | Worlds | Arms | Full world/arm jobs | Continuous training histories | Record applications |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pilot | 4 | 4 | 16 | 48 | 399,360 |
| Conditional confirmation | 16 | 4 | 64 | 192 | 1,597,440 |
| Conditional scientific maximum | 20 | 4 | 80 | 240 | 1,996,800 |

Each history has 8,320 record applications, including presentations with writes disabled or shuffled labels. Count those separately as external exposures, enabled native teaching, representation-update opportunities and actual applied updates. Engineering qualification records and evaluator queries are additional, separately counted costs, never scientific data.

The 64-unit state has 19,456 candidate parameters with 88 input channels and zero-diagonal 64×64 recurrent matrices: two 64×88 input matrices, two 64×64 matrices minus their diagonals, and two 64-element biases. Also account for the fixed 64×16 feedback projection, fixed 88×64 forward projection, masks and local eligibilities. Sixteen real native compartments remain the dominant inherited store allocation; share only verified immutable assets.

Charge all repeated training, feedback bytes, local eligibility work, native prediction needed for teaching, homeostasis, fixed-code decoding and rest processing. Separate production CPU, evaluation/clone CPU, auditing and wall time. Log persistent/temporary bytes, peak RSS, native event count and bytes moved. No claim of computational sparsity follows solely from sparse KC firing: the inherited encoder still projects and sorts. The generic adaptor is bounded at width 64; this design is not a demonstrated scalable large-brain implementation.

Use at most four workers, one numerical thread each, subject to live memory admission. Measure complete-job runtime before scheduling. Stop admission after eight wall hours and enforce the existing ten-hour session boundary; retain the clock across controller handoffs. Pause at safe whole-job boundaries and require explicit continuation. A single job that cannot safely fit needs a declared boundary/checkpoint design before launch, not an automatic longer session or partial-history replay. No promised completion time precedes measurement.

## 12 Qualification, reporting and release

Implementation and a three-cycle exercise require a subsequent request. Their intended sequence is:

1. **Design/source audit and tiny fixtures:** prove the exact v0 parent with adaptor disabled; audit all semantic inputs, output paths, local update terms, byte coverage, frame reuse and all-learning-off behavior. Qualify the native source before adding plasticity.
2. **Bounded engineering test and revision:** test local credit, gradual/dose-matched schedule sums, parameter caps, generic arbitrary-byte learning, saturation and continuing taught acquisition. Check that frozen/plastic arms have identical states before their first permitted representation update. Record selected finite calibration settings without scientific HELD feedback.
3. **Complete engineering life and independent reduction:** execute the full proposed chronology on the two engineering worlds, validate actual rest times, no-write/shuffled histories, time-only endpoint, exact EOS/truncation, resume integrity, all output hashes, accounting and statistics on synthetic fixtures. Lock assets, parameters, CPU budget, roster and forecast before proposing the pilot.

Vectorization may be used only after source/adapter equivalence tests. Floating-order differences require declared tolerances and unchanged event choices; do not promise bit-for-bit equality across changed reduction orders. Source parity, identity hashes and gate booleans cannot substitute for behavioral learning results.

Deliver a model/lineage manifest; source/asset and parameter locks; dataset/target-provenance manifest; immutable full-job receipts and independent audits; per-world output tables; acquisition, efficiency and retention curves; representation/homeostasis diagnostics; resource report; final report and compact result package. Include failed jobs, costs and actual evidence level. Separate engineering, pilot and untouched confirmation data.

The new model has no demonstrated result at design delivery. Historical MiniFly-CLS memory gains remain E1 evidence; v3 must earn its own retained E3 result. No new task, stronger innate arithmetic structure, looser approximate-answer endpoint or uncounted replay may be substituted after a negative result.

## 13 Governing project records

- [Objective and E0–E4 contract](../../MINIFLY_OBJECTIVE_CONTRACT.md).
- [Persistent core and scaling requirements](../../PERSISTENT_CORE_REQUIREMENTS.md).
- [Current experiment pointer](../PROJECT_DOCUMENTS/CURRENT_ARITHMETIC_EXPERIMENT_DESIGN.json).
- [Generic byte IO contract](../PROJECT_DOCUMENTS/GENERIC_BYTE_IO_CONTRACT_20261012.md).
- [No-attention requirement](../PROJECT_DOCUMENTS/NO_ATTENTION_KERNEL_REQUIREMENT_20261013.md).
- [Feedback curriculum requirement](../PROJECT_DOCUMENTS/KERNEL_FEEDBACK_CURRICULUM_REQUIREMENT_20261013.json), with its later addition-only scope preserved here.

The governing v3 clarification is: no arithmetic-specific structure; arithmetic learned from pretraining data; gradual warmup that preserves the ability to keep learning. The architecture and learning intensity are hypotheses to test under those constraints, not a supplied arithmetic solver.
