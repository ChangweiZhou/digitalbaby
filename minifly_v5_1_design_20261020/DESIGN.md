# MiniFly v5.1 — learned byte structure and calibrated native memory

20 October 2026. **Revised design only. Not implemented, trained, qualified or launched.** This is a revision of Claude's v5 at commit `8e90d2e6c57d0394ce6246b9e9b69a75a9306ed4`, following the [v5 review](../minifly_v5_design_review_20261020/REVIEW.md). The received memo is preserved in [CLAUDE_RECEIVED.md](CLAUDE_RECEIVED.md). Historical designs, results and adoption decisions are unchanged.

**Goal: retained exact own-byte-output generalization to unseen addition problems, using fewer training presentations and less actual compute.** Arithmetic must be learned from observed feedback. Byte discrimination, syntax, taught recall and an evaluator's fitted readout are diagnostics; none substitutes for this goal.

## 1. Decision and hypothesis

Keep v5's generic learned byte units, context-derived segmentation, short segment registers, local KC wiring and native value memory. Correct its output reference, causal event timing, warmup, exposure accounting, consolidation and evaluation contract. This is a native-memory-derived intervention, not the unchanged Full151/R_center executable.

The hypothesis is that shared, learned sequence features plus repeated, spaced value updates can support reuse beyond taught whole problems. Confirmation may also preserve frequent punctuation or answer-bit biases. Neither a strict gate nor a rare-whole/frequent-parts stream guarantees that rules emerge. Failure to acquire taught replies must remain visible before attributing failure to abstraction.

Constraints remain: no attention, Transformers, ELM, backpropagation, independent answer head, raw-answer lookup fallback, arithmetic parser, supplied numeric features, designated operand/digit/carry units, arithmetic relations or reversed-answer curriculum. Ordinary written addition is data, not circuitry. The local competitive updates below are explicit Hebbian rules, not gradients propagated through the task. All 256 bytes are valid payload; the primary task remains addition only. Subtraction, decimal comparison, NLP, noisy-feedback optimization and scalar-only feedback are deferred.

### Source and capability lineage

| Source | Retained basis | Capability not inherited by citation |
| --- | --- | --- |
| [v3 design and source path](../minifly_v3_design_20261020/DESIGN.md) | Native-value lineage, gradual warmup, exact generation, causal histories; original v3 executable is a diagnostic anchor on its shared small range | Its taught recall does not qualify this new byte port or encoder |
| [v4 implementation/report](../minifly_v4_dev_20261020/REPORT.md) | Native numerical primitive, raw-bit interface and explicit engineering failures | Primitive parity at 5,177 cells is not parity of a larger population, new baseline, timing or consolidation |
| [Received v5](CLAUDE_RECEIVED.md) | Learned byte identities and context classes before assessing KC coverage | No v5 performance has been measured |
| [V9.9A](../../BYTE_CORE_V9_9A/REPORT.md), [V9.9B](../../BYTE_CORE_V9_9B_RANK/REPORT.md) | Example of prior-teaching causal controls and narrow E3 interpretation | Synthetic relation-choice results do not establish byte-output arithmetic |
| Historical COV-R/COV-SUPPORT | Preserve historical engineering and user-adoption records | No arithmetic decoder, ordinal representation or earlier efficacy is imported |

The inherited native event equations and birth assets must be pinned and independently checked during a future implementation. Reuse the native primitive, not the v4 wrapper's bugs or a scaffold's unrelated codebook. Enlarging the population requires its own readout normalization and parity tests.

## 2. Evidence and exposure contract

E0 checks byte access, order, causal frames, readout and numerical correctness. E1 checks immediate and delayed recall of actually taught replies. The intended E3 claim is **incremental reuse of the OLD native teaching episode, conditional on a shared unsupervised listening prestate**, on held-out addition orbits after delay and further addition learning. This does not isolate the causal value of listening itself or claim E4 breadth.

Before generating any stream, the evaluator partitions semantic unordered operand pairs `{a,b}`. Both orientations belong to the same partition. No held-out pair may appear with its answer in listening, native teaching, replay, parameter calibration or fitted-head training. Apply this globally across lengths, phases and stream sources; a second wrapper or reversed orientation cannot evade it. Repeated answer strings from different allowed pairs are legitimate shared information, not whole-pair leakage. The learner never receives split IDs, operands, numeric values, carry labels, world seeds or answer lengths.

Use separate training, development-probe and final-confirmation partitions. Development probes may select only the finite choices declared here, are never fed back, and cannot be described as fresh confirmation. Final held-out operands and labels remain evaluator-only until the frozen model's first scored generation. Listening contains training pairs only. All correct answer bytes heard during listening count as representation-training exposure, even without an error signal.

Maintain separate ledgers for unique semantic pairs, oriented prompts, external presentations, listening bytes, answer bytes/EOS, native teaching events, replay applications, fitted-head labels, query bytes and resource failures. A fresh draw can repeat a whole fact, particularly in the one-digit range. Report its multiplicity. A replay has an original record ID and does not become new information. Record generation and split filtering are evaluator operations, never learned features.

## 3. Generic IO and exact event timing

The external envelope is `BEGIN, prompt payload, REPLY, observed reply payload, EOS`. BEGIN/REPLY/EOS are typed IO controls outside the byte alphabet. Payload `=`, NUL, newline and punctuation have no built-in control role. BEGIN clears temporary sequence buffers and record eligibility only; learned state persists. Use the inherited per-record adaptation reset if the pinned native parent requires it, identically for training and querying, and declare it in the parameter lock. Do not hide this override as biological decay.

Input limit: 256 payload bytes. Output limit: 16 payload bytes, followed by a separate EOS decision. Input overflow returns `INPUT_LIMIT`; output overflow returns `truncated=true`. Neither is an apparent correct answer. The active segment keeps at most 16 bytes, with an overflow flag; four closed segments are retained. Those are bounded working-memory limits, not unlimited context claims.

### Event algorithm

1. BEGIN resets temporary buffers and counters. An arriving prompt byte is classified using the already available prototype/class version. If its class differs from the preceding byte, close the preceding open segment, shift the closed registers, then start a new open segment **containing the boundary byte**. Otherwise append it. Apply permitted unsupervised updates only after extracting that event's pre-update identities.
2. REPLY sets a generic reply-step counter to zero. It is not a payload byte and does not close the current segment. No future target or target length is available.
3. Prepare one frame from available buffers, reply step, last observed/emitted byte, native prestate and clock. Compute activity, native values and the neutral reference. Capture this frame once. During training, the next observed target is revealed only after this preparation; calculate the residual and write through this same frame. During generation, choose byte/EOS without any target.
4. Advance the native event once. Then feed the actually observed training byte, or actually emitted own byte, into temporary sequence state by step 1. Increment reply step after a payload byte. EOS ends the reply without inventing a sensory byte. An emission never becomes a training observation.
5. Apply record-end bookkeeping only after EOS; BEGIN of the next record cannot edit the preceding prediction's frame. All modules share the native event interval `dt = 30/14` simulated seconds per payload/target event. Typed BEGIN/REPLY consume no extra hidden interval. The EOS target/decision consumes one interval. Listening and replay consume their actual intervals too.

Illustrative trace **if** learned classes group digit runs separately from separators; these classes are not supplied:

| Moment in `12+34=` | Closed segments, newest first | Open segment | Reply step |
| --- | --- | --- | --- |
| After `+` | `12` | `+` | inactive |
| After `3` | `+`, `12` | `3` | inactive |
| First prediction after REPLY | `34`, `+`, `12` | `=` | 0 |
| After own first digit, if its class changes | `=`, `34`, `+`, `12` | emitted digit | 1 |

The segment ages really shift at that first output byte. The model must learn with those frames; the encoder cannot pre-close `=` using the expected digit. Boundary age and reply step are distinct signals. Changing only a withheld target must leave the preceding activity, reference and native prediction bit-identical.

Generation runs on a private temporary read view at the checkpoint's clock, allowing only within-query sequence/adaptation evolution. It commits no weights, native values, prototypes, thresholds, q, replay state, reference state or training counters. Reset the view between queries so query ordering cannot change results. Account for cloning/copy costs; a readonly clone is not free sparse inference.

## 4. Learned representation, then a real freeze

### M1/M2: raw bits and competitive byte units

Represent an observed byte as eight signed bits plus a byte-present line. Keep the raw-bit bypass throughout. M2 has K=64 prototypes `w_u` in `[-1,1]^8`; initialize independently of byte values. The winning unit maximizes `-||b-w_u||²-h_u`, with deterministic index tie-breaking. Update only the winner: `w_u <- clip(w_u + eta_byte (b-w_u), -1,1)`. Update its usage EMA and a bounded competitive bias to discourage domination. The exact bias gain is selected by the bounded identity qualification, not task accuracy. Homeostasis does **not** require all 64 units to be equally active.

K=64 cannot provide an injective winner for all 256 possible bytes. The raw path must pass all-byte E0; the learned path is qualified on the observed training alphabet and reports every alias. Do not claim general 256-symbol identity learning from addition's small alphabet. If observed types exceed K, report capacity exhaustion; no silent new units.

Use a smooth listening-rate ramp from `eta_max/16` to `eta_max`, based on observed byte count, not score. Candidate `eta_max` is fixed to `{0.001, 0.005}` for technical selection; bias gain to `{0, 0.01}`. Select lexicographically by fewer byte aliases, lower type-weighted reconstruction error, lower assignment drift, then lower dose/rate. Train/development byte streams are disjoint at pair level. Log the actual update norm and usage; unused units are permitted.

Check M2 at 50,000, 100,000 and 200,000 heard bytes only. Require zero aliasing among observed byte types and an identical winner map across the two preceding 10,000-byte blocks, each containing those types. Compare maps at each block end using fixed type probes with updates disabled; report all reconstruction errors rather than adding an arbitrary accuracy hurdle. Freeze M2 identities before M3 context accumulation. Do not combine counts whose unit meanings drift. Rate choices are engineering hypotheses, not established neural constants. A failed finite grid yields `IDENTITY_NOT_QUALIFIED`, not an automatic search expansion.

### M3: context classes and learned boundaries

For a frozen M2 sequence, accumulate immediate predecessor/successor winner counts for each unit. On observing `u_t`, increment `right[u_(t-1),u_t]` and `left[u_t,u_(t-1)]`; the successor for `u_t` is unknown until the next arrival. Do not cross BEGIN/EOS boundaries. Normalize left and right profiles separately, concatenate them, and cluster into six competitive class prototypes. The local update is `v_c <- v_c + eta_class(profile_u-v_c)` for the winning class, with `eta_class` smoothly ramped to 0.001. No class receives a byte-role label. Units with no context use an explicit unknown class; they do not borrow future statistics.

Classes may change during this separate listening phase. Check at 50,000, 100,000 and 200,000 heard bytes; require identical assignments across the two preceding 10,000-byte blocks, then freeze assignments and profiles before M4/native learning. Stability alone cannot qualify them. Report class occupancy, transition density, byte-to-class confusion, singleton runs, run-length distribution, within-byte aliasing and contexts at every class change. In addition, compare the frozen class representation's next-winner predictive log loss on development streams against (a) one-class collapse and (b) byte-identity transition counts, with the same declared smoothing and information budget. Require improvement over the collapsed model and a description length smaller than the identity alternative, counting class labels/prototypes as model cost. The precise smoothing/model-codelength formula must be locked before listening. This is an evaluator diagnostic, never a boundary-teaching signal.

Evaluator-only run annotations may explain undersplitting/oversplitting; they cannot label a unit, wire a connection or tune final-test behavior. No requirement says classes must equal digits/operators. A stable but unhelpful clustering is `SEGMENTATION_NOT_QUALIFIED`; a diagnostic failure does not prove all generic segmentation impossible.

### M4: bounded generic segment features and local connectivity

Keep the open segment in arrival order and the four closed registers as specified above. At frame preparation, expose one-hot learned-byte identities and raw signed bits at generic right-relative offsets 0–15 in each register, plus presence/overflow, boundary age, generic reply step and previous-byte identity. Relative offsets are generic buffer coordinates, not numeric places; the same circuit would apply to any byte stream. No selector can inspect an arithmetic role. Reply step and boundary age use fixed generic finite time-cell banks for 0–16 events, plus overflow.

Use **65,536 KCs only in the initial bounded comparison**. V5's 262,144/1,048,576 sizes are deferred, not automatic rescue options. Use fan-in four initially. Half the KCs draw four excitatory, bounded byte-unit/presence afferents; the other half draw two such afferents, one generic time-family afferent and one previous-byte/presence afferent. Choose occupied register slots and offsets by birth RNG independently of data labels. The topographic arm draws offsets from a shared anchor plus offsets in `{-1,0,1}`, clipped to buffer limits; register choices are independent. It must not preferentially wire matching operand columns. The random lesion uses the same family/degree/weight distributions, sampled offsets without locality.

The primary activation is the inherited weighted projection and per-side competition, with nonnegative unit-identity afferent weights and signed raw-bypass afferents source-locked. **Call these combination features, not guaranteed AND cells.** Characterize actual conjunction sensitivity, overlap and active counts on causal frames. A dendritic AND threshold would be a separate new mechanism; it is not smuggled into this revision. M4 thresholds receive the parent's one-sided homeostatic update only during a bounded, target-free calibration after M2/M3 freeze, then freeze before native value teaching. Match active-count budgets between wiring arms using that calibration only. Store the exact connections, weights, native Q/T/readout rescaling and competition rules in the executable parameter lock; no efficacy result can be claimed while these porting details are unresolved.

M2/M3/M4 are genuinely frozen in the primary learning assay. Later low-rate code movement is a different arm/version requiring its own design and replay accounting. This removes the received memo's contradictory “freeze, then keep learning” rule.

## 5. Native output without subtracting learned signal

Nine native value compartments generate eight byte bits plus EOS. There is no alternative answer-producing path.

For compartment j and prepared activity x, define `F_j(x, fast, slow, adapt, dt; birth)` as the pinned native value-read function. Use:

```text
z_j = F_j(x, fast_j, slow_j, adapt_j, dt; birth)
r_j = F_j(x, zeros, zeros, adapt_j, dt; birth)
m_j = z_j - r_j
p_j = tanh(m_j / T), with T = 2 initially
s_j = a_fast(n) * (p_j - y_j) / 2
```

The reference is a **matched zero-value physiological counterfactual**, not a running average of z. It uses the same activity, current adaptation, dt, birth constants and readout clipping. Zero value means zero learned fast/slow deviations as in the pinned birth, not zero native baseline currents. Compute it without constructing a second trainable brain or resetting the actual state. It has no targets, moving mean, trainable offset or persistent reference array. Adaptation dependence is intentional and must be audited against a matched passive history. Native nonlinear clipping means subtraction may still leave a poorly conditioned margin; calibration must detect this, not assume the correction works.

At a payload target, teach all eight signed byte bits and EOS target -1. At terminal EOS, teach only EOS target +1; byte channels are masked, not taught a fictitious zero byte. During generation, positive EOS margin ends the answer; otherwise choose each bit by positive margin (ties choose 0). No vocabulary mask, digit restriction, analytic length hint or arithmetic correction is applied. All 256 emitted byte patterns remain possible.

Train and decode use the identical reference and event frame. With all value deviations zero, the prepared margin must be zero to numerical tolerance across activities/adaptations. A learned constant signal must remain visible rather than be absorbed by a mean reference. Required port fixtures cover all bits in both directions, all 256 byte payloads, EOS, persistent on/off and imbalanced targets, different contexts, saturation, zero activity, rest and passive controls. Directional correctness is necessary but does not establish exact taught-reply acquisition.

## 6. Gradual warmup and an explicit strict commitment gate

### Calibrate dose; do not reuse the attenuated v4 constant

V4's quoted ~0.01–0.02 shifts came from coefficients attenuated by 1/16. They are not full-strength constants. Before scheduling a long run, measure the new reference and actual write strengths on fixed generic taught-byte fixtures and passive controls, over early/late native states and measured active-count ranges. Record response curves, clipping, acquisition, interference, write norms and sample cost; do not extrapolate linearly through saturation.

For technical calibration only, test `a_max in {1/16, 1/4, 1}` and ramp length `W in {256,1024}` external records. Use `a_fast(n)=a_max*[1/16+(15/16)*min(n/W,1)]`. Representation learning has its own preceding ramp, and slow commitment has a separate hurdle. Fast decay alone is not protection against disruptive fast learning.

First isolate the port with fixed, noncolliding synthetic activity frames, independent of arithmetic and the untrained segment encoder. Include eight balanced reply sequences of lengths 0–3, containing NUL, high-bit bytes and EOS, plus persistent on/off and imbalanced channel fixtures. Freeze the actual cue/activity/target roster before tests. After Q1, repeat a distinct eight-cue arbitrary-byte acquisition check through the **complete frozen encoder**. An isolated port pass cannot substitute for this integration check, and an untrained encoder cannot prematurely reject the port.

Select the **smallest a_max**, then longest W, that acquires all fixed generic calibration replies with own EOS and retains them at the declared immediate probe, has finite state and improves target loss versus its matched passive control. Repeat fixtures under at least two paired births; no arithmetic held-out scores enter this choice. An individual reply may receive at most 256 observed presentations; cap the full generic calibration at 8,192 record applications per setting, including repeats. If none qualifies, stop `PORT_ACQUISITION_NOT_QUALIFIED`. Re-check the selected schedule at integration; failure stops rather than silently trying arithmetic efficacy scores to retune it. These are finite engineering sufficiency/safety checks, not statistical arithmetic thresholds. A later full-strength-vs-gradual comparison would need a separate arm and equal cumulative-dose budget.

### Signed confirmation and allocation

Track q per native slow synapse/compartment, not per whole-answer lookup. Let `d` be the signed prospective alpha increment from the native primitive after applying the current residual but before splitting fast/slow allocation. Use a numerical zero tolerance locked by the precision audit. During a record, retain a temporary sign-consistency accumulator per synapse: +1 if all nonzero d are positive, -1 if all are negative, 0 if none or if signs conflict. At an eligible **completed external record**, let c be this accumulator and update:

```text
q_now = q_old * exp(-(t - t_q) / tau_q)
q_new = q_now + c
tau_q = 86400 simulated seconds
q0 = 3
f(q) = 0                                      if abs(q) <= q0
       0.5 * (1 - exp(-(abs(q) - q0)))         otherwise
```

Use the decayed **pre-event** q for each event's allocation; only after EOS can this record register a confirmation. Allocate fraction f to the native slow alpha update and 1-f to its fast alpha update; retain the other native fast-path equations. The gate cannot reverse the signed increment or add an independent learned answer weight. Apply the same native decay/adaptation and clock in all arms. The first slow allocation therefore requires confirmations from completed earlier records; positive and negative consistency are symmetric. Opposite confirmations across records cancel q; conflicting signs within a record give no new confirmation; zero d supplies none. Mixed-sign suppression is a declared hypothesis and must be reported, not hidden by counting only a record's first output byte.

Eligibility requires (a) a real completed external record and c != 0, (b) a different prompt identity, (c) at least 60 simulated seconds since this synapse's last accepted confirmation, and (d) at most one confirmation per synapse per record. Identity is a digest of the complete **already observed prompt**, never the future answer or target bit. Maintain an eight-identity bounded recent-confirmation ring per synapse and refuse repeats within it. This enforces bounded recent novelty, **not proof of globally unique independent examples**. Audit repeat/collision rates offline using the exposure ledger. A common bit regularity can satisfy it; that is not automatically an arithmetic rule.

The q decay clock and last accepted-confirmation time are distinct. Decay can be lazy with timestamps, but checking an ineligible event must not postpone spacing or erase a negative sign. A no-write episode advances the physical/native clock and passive adaptation while q only decays. q0, tau_q, 60 s, the ring size and maximum slow fraction are explicit v5.1 hypotheses: more than three net confirmations, near-day persistence and bounded allocation. They are not derived biological optima and will not be tuned on arithmetic efficacy in this comparison.

### Rest and bounded replay

Rest advances native fast/slow/adaptation decay and q's elapsed time. Primary v5.1 adds **no extra multiplicative slow downscale** beyond native decay: v5's unspecified extra factor would confound the commitment test. Its study is deferred. Between OLD and NEW and after NEW, advance exactly one simulated hour, reporting its relation to the measured parent time constants. Long presentation streams also consume simulated time; tens of thousands of records can span many simulated days. Never compress dt to rescue forgetting.

Store at most 128 actually observed training records in a FIFO replay buffer, with origin IDs. At each scheduled rest, replay at most 32 records once in chronological buffer order; omit empty entries. Replay supplies only the recorded external answer, never a guess or held-out answer. Replay consumes native time and counts as additional presentations/CPU. It can update fast values at the locked ramp strength, but supplies **no q confirmation and no slow write**. Its limited fast-refresh role is explicit; it cannot manufacture independent confirmation. Compare arms with identical replay records and clocks.

## 7. Persistent-state and phase permissions

| State | Initialization/update | Listening | Native training/replay | No-prior-write OLD | Scoring/freeze |
| --- | --- | --- | --- | --- | --- |
| M2 prototypes, usage, bias | Random paired birth; §4 local update | M2 phase only | Frozen | Shared frozen D0 snapshot | Read only |
| M3 context counts, class prototypes/assignments | Zero counts; §4 update after M2 freeze | M3 phase only | Frozen | Shared frozen D0 snapshot | Read only |
| M4 afferents, homeostatic thresholds | Label-independent birth; native threshold rule | Separate final calibration only | Frozen | Identical to W | Read only |
| Open/closed segments, times, reply step | BEGIN reset; §3 causal transition | Temporary state | Temporary state | Same observed byte stream | Private per-query state |
| Native fast/slow deviations | Zero relative to pinned native birth | No value teaching | Native signed writes; replay fast only | Writes disabled throughout OLD | Read view; no commit |
| Native adaptation/clock | Pinned birth/reset; native passive equations | Advance by actual events | Advance by all events/rest | Same clocks/passive equations | Temporary forward evolution only |
| q, q timestamps, identity ring | Zero/empty; §6 | No confirmations | External training only | No OLD confirmations; decay only | No commit |
| Physiological reference | Stateless matched §5 read | No target fitting | Same pre-event read | Same formula | No update exists |
| Replay records/provenance | Bounded FIFO of real records | No replay in D0 | Same chosen real records | Same buffer admissions; OLD write flag persists on replay | No insertions |

Query hashes cover all persistent learned and bookkeeping state. Checkpoint tests must preserve native clock, adaptation, decay timestamps, q sign, identity ring and replay origin. Sharing immutable anatomy is allowed; sharing mutable values across causal histories is not. Scientific stream logs are external evidence, not a lookup memory available to the fly.

## 8. Addition stream and bounded schedule

Data uses ordinary canonical decimal `a+b=` prompts and canonical decimal answer bytes in normal order, within operands 0–9999. The generator may compute labels; the learner receives bytes only. Sampling chooses operand digit lengths uniformly among allowed lengths, then numbers uniformly within the chosen length (0–9 for one digit). Balance equal/unequal lengths without a task flag to the core. No carry curriculum, commutative augmentation instruction, sorted operands, solution steps or auxiliary counting task is supplied. Held-out orbits include both orientations even when only one is sampled in training.

Listening is addition-only across the same allowed length range, drawn from the training partition, including observed answers. M2 hears at most 200,000 payload bytes; M3 then hears at most 200,000; M4 target-free calibration at most 100,000. Each phase stops at a predetermined checkpoint only if its diagnostic qualifies; otherwise report failure at the cap. Re-reading records in later listening phases counts again. Freeze all representation state before native OLD learning. D0 exposure is charged to **each learner's learning curve**, even if a paired snapshot is computed once operationally.

OLD teaches training orbits with 1–3 digit operands. NEW supplies further addition training, half 1–3 digits and half with at least one four-digit operand, with exact pair disjointness from OLD and all probes. OLD retention and OLD held-out E3 are scored after intervening NEW. Four-digit held-out results are a separate breadth diagnostic, not a second task or a replacement endpoint. Hold out 256 OLD-domain and 256 NEW-domain pairs per world, stratified by length using evaluator-only metadata; teach none of them in any phase. Taught probes are a fixed 64-record subset of OLD, including early and recent records. Query both orientations where distinct, average within orbit first, then within world.

The finite native-dose menu is OLD `{3000,10000,30000}` external records; NEW is exactly one third of the chosen OLD budget, rounded down. Thus the largest branch has 40,000 fresh-draw external presentations plus at most 64 replay applications, not 100,000 automatically. Record learning curves on fixed log-spaced checkpoints within the selected dose. These are affordability caps inherited in scale from v5's proposal, **not doses already justified by the old 0.01–0.02 measurement**.

Before selecting a budget, use a counted engineering prefix of at most 1,024 records to measure actual feature recurrence, calibrated acquisition exposures, decay time, production CPU and peak RSS. Choose the smallest menu entry whose projected recurrence reaches the calibrated exposure requirement within the resource cap; freeze the projection method and selection before viewing arithmetic development accuracy. If none is feasible, stop `DOSE_NOT_JUSTIFIED`; do not increase N or stretch the cap. Store observed/projected counts separately. Final eligibility still depends on behavior, not recurrence estimates.

World namespaces reserved in this design: engineering `202610205101–202610205102`, development `202610205111–202610205114`, and conditional independent confirmation `202610205201–202610205216`. No worlds or data have been generated by this delivery. A future implementation must audit collision with existing rosters before locking these namespaces; overlapping seeds are replaced before any new training, never after outcome inspection.

## 9. Comparisons and causal histories

| Arm | Allocation | KC locality | Segments | Role |
| --- | --- | --- | --- | --- |
| C-full | Strict q gate | Local generic offsets | Learned | Sole new adoption candidate |
| C-native | Pinned native allocation, same gradual schedule | Same as C-full | Same as C-full | Primary comparator |
| C-random | Strict q gate | Matched nonlocal offsets | Learned | Diagnostic locality lesion |
| C-noseg | Strict q gate | Same family/degree budget | Generic record-relative buffer only | Diagnostic segmentation lesion |

All have the same K, N, output path, stream, birth draws where applicable, replay, feedback and clocks. Lesion differences in frame availability are intended; capacity, active count and weight distributions remain matched. C-noseg has no content-defined closure; its generic rolling record window and reply counter replace segment registers without seeing arithmetic positions. Log the unavoidable difference in truncation/coverage.

Run W, N_OLD and SH_OLD for C-full and C-native; run W only for each lesion: **eight correlated histories per world**. W permits native teaching. N_OLD hears exactly the same OLD prompts/answers and shares the frozen D0 representation, but disables all OLD native writes and confirmations, including replay of OLD origins; subsequent NEW teaching is enabled. SH_OLD uses a locked within-answer-length permutation of OLD answer strings, preserving per-event length, bit/EOS target multiset and clocks; NEW is correct in all histories. Do not feed the original label through another loss in SH_OLD. Confirm that only the intended history states diverge.

Chronology: shared D0 → OLD → immediate E1/E3 probes → one-hour rest and bounded replay → OLD probes → NEW → final one-hour rest/replay → **primary final OLD held-out E3 and taught-retention probes**. All comparisons are within-world, paired at equal event clocks. Rest, replay and query schedules are fixed before outcomes. Quoted historical v3 recall is not a positive-control run: separately reproduce the pinned v3 executable on the small overlapping addition range, with explicit matched information accounting, as a diagnostic anchor only. It cannot supply answers to v5.1.

Countermodels trained only on allowed exposure: exact raw prompt table; evaluator-canonicalized unordered-pair table; output-length/frequency and previous-byte shortcuts; bounded suffix backoff lengths 0–4. Use the strongest development-selected member with its selection cost recorded; freeze it before final confirmation. Any example-derived reusable composition is a legitimate learned E3 explanation, not a forbidden supplied rule. Analytic addition is evaluator-only label/solvability checking and never a learner arm.

## 10. Diagnostic ladder with stop decisions

Each failure localizes a problem and ends the relevant escalation. Do not repeatedly rerun the same failed trajectories or search larger brains automatically.

| Step | Question and required evidence | Bound/decision |
| --- | --- | --- |
| Q0 port/native bridge | All-byte IO, exact causal frame, corrected-sign response versus passive control, matched zero-value reference, readonly query, checkpoint parity, bounded taught-reply acquisition on isolated activity frames | §5–6 generic fixtures and finite grid. Failure stops before listening/native arithmetic escalation. E0/E1 only; full-encoder acquisition is checked after Q1. |
| Q1 listening and freeze | Stable distinguishable observed byte identities; no forced equal use; nontrivial useful classes; no target leakage; boundary byte retained | §4 and §8 byte caps. Zero observed-type aliases after freeze; report full confusion/drift, with generic 256-byte path tested separately. Failures remain E0 diagnostics. |
| Q2 actual frames and learned-code readout | Can the integrated frozen encoder/port acquire arbitrary taught replies? Do causal output frames retain usable information? Are features more than one dominant bit? | §6 integration check, then bounded evaluator fit below. Positive result is diagnostic, not arithmetic transfer. Failure stops scaling, but does not prove impossibility. |
| Q3 native acquisition and recurrence | Can native fast/slow values learn real replies under calibrated gradual warmup, and does q allocate exactly as specified? | Dose menu, resource projection and immutable engineering lock. Report immediate taught exact, recent/early retention, syntax, length, bits and margins; no universal 95% arithmetic-format gate. |
| Q4 bounded E3 comparison | Does C-full improve final retained own exact unseen addition over C-native and causal/recitation controls per information and compute? | Four development worlds only after prerequisite qualification; any subsequent confirmation needs a frozen decision and separate authorization. |

### Q2 empirical feature-readout diagnostic

Use the frozen learned representations and **the same prefeedback output frames as the native learner**. A streamed linear ridge head is evaluator-only and is discarded; it is not an ELM learner, fallback or promotion arm. Its success does not qualify native local learning; its failure is not a best-possible linear or native capacity bound.

Fit at prefixes of 1,000 and 3,000 training problems, with 512 development pairs for regularization selection from `{0.01,1,100}`. At most three streamed passes per prefix/setting, using a declared residual-based solver and numerical stopping tolerance; retain no dense design or Gram matrix. Use 8-bit/EOS masks as in §5, and report teacher-forced byte/exact separately from own-prefix decoding. Charge every extracted prompt/answer byte, fitted target, pass, label and CPU. No final-confirmation pair is used here. Solver, stopping rule and tolerance must be source-locked before fitting, and a nonconverged head is reported as such, not as a capacity result.

Maximum diagnostic CPU is 600 process seconds across this grid; peak RSS is at most 2 GiB and must also leave 2 GiB measured system headroom. Cache at most 128 MiB of sparse frames. Profile the first 128 problems and project costs before starting the grid. If the projection fails, return `RESOURCE_INFEASIBLE` without allocating it. The nine float64 head vectors at N=65,536 occupy ~4.5 MiB, but native states and solver temporaries count too. V5's illustrative 500k-row, 1m-cell dense feature array would require ~3.81 TiB; it is explicitly excluded here.

### Static gate versus behavioral criterion

Exact identity, no forbidden write, correct sign symmetry, zero below-threshold allocation and readonly hashes are invariants, not arbitrary success percentages. E1 calibration requires complete recall on a finite declared fixture because a single broken bit/EOS channel matters; it is not a generalization claim. Q2 reports empirical curves; it has no fictional “best possible ≥95%” gate. Q3 must show useful taught acquisition versus passive control rather than silently removing taught recall. Q4 keeps exact E3 as the objective regardless of partial curves.

## 11. Outcome, efficiency and confirmation boundary

Primary score is final OLD-domain **exact complete own-generated answer plus own EOS** on globally unreinforced orbits, after NEW and the final rest. Wrong byte, wrong length, missing/extra byte, invalid decimal rendering, premature EOS, input failure or truncation makes the whole answer wrong. No teacher forcing or expected output length enters primary generation. Report syntax validity, predicted length, per-bit/byte, first-error position, evaluator-only carry strata and teacher-forced loss separately. Missing/extra positions count as errors in position metrics rather than disappearing from a denominator.

Use per-world exact-accuracy curves versus (a) all external/listening/replay presentations and answer-bearing bytes and (b) actual learner CPU, including listening, native production, replay and checkpoint work. Do not divide a shared D0 cost by the number of arms to make learning look cheaper. Report inference/evaluation, generic qualification, failed work, evaluator fitting, total worker CPU and peak RSS separately, alongside the complete project cost. No partial accuracy or cheaper fit substitutes for the fly's exact E3 score.

Development reports paired differences and uncertainty without a confirmatory verdict. Only C-full can be selected, and only if it exceeds C-native, N_OLD, SH_OLD and the frozen strongest recitation baseline at the final exact endpoint, does not reduce final taught-OLD exact versus C-native, and improves both sample-axis and learner-CPU-axis integrated exact-accuracy curves over their common measured ranges. Integrals use fixed checkpoints and trapezoidal interpolation, no extrapolation outside a shared range. A speed/accuracy tradeoff is reported as a tradeoff, not automatically promoted.

For conditional fresh confirmation, prelock one primary final C-full−C-native contrast. Use paired world-level two-sided 95% Student-t intervals (16 worlds), not records/histories as independent n. Require its lower bound >0, plus lower bounds >0 against N_OLD/SH_OLD/strongest recitation and for both efficiency-curve differences; require lower bound ≥0 for the taught-OLD retention contrast. These jointly required conditions form an intersection claim; no searching among endpoints for a pass. Zero margins are deliberately strict **relative improvement** requirements, not claims that any tiny effect is commercially important. Publish absolute accuracy, intervals, effect sizes and the precision limits of 16 worlds. Do not add a posthoc 95% accuracy threshold or claim a precomputed power calculation. A later deployment threshold needs its own decision.

No superiority, causal gain, retention or efficiency condition can be waived because a partial diagnostic improved. `NO_ADVANCE`, `NO_DEMONSTRATED_E3_GAIN`, `RESOURCE_INFEASIBLE` and a qualified interface with failed acquisition are valid distinct outcomes. A null result at these caps does not reject every local recurrent architecture. Larger N, new tasks, different feedback, extra replay or more worlds require a new design decision rather than a hidden continuation.

## 12. Implementation handoff and remaining locks

The following are mandatory **before any future training**, not permission requests generated by this document:

1. Pin native birth assets/equations, readout normalization, M4 afferents, homeostasis, numerical precision, counterfactual reference and event code. Run independent primitive and port tests; the old parity certificate is insufficient.
2. Lock byte/context streams, split filter, masks, q/spacing/ring semantics, all calibration fixtures, finite grids, query schedule and event-count ledger before technical tests. Generic calibration bytes must exclude reserved held prompts as complete records.
3. Obtain measured parameter/resource results within the bounds above, then publish `PARAMETER_LOCK.json` with chosen rates, W, actual dose, feature recurrence projections, solver convergence rules, per-event/global work and memory. An unresolved lock yields `NOT_READY`, never a guessed value in a launched campaign.
4. Separate technical engineering, development and optional confirmation. Preserve every failed attempt's real costs and error; no automatic science retry. Original design/source snapshots survive revisions.

Future run safety: one numerical thread per worker, initially one worker; resource admission must leave the stated system headroom. Bound each technical setting at 1,800 process CPU seconds; bound any launched session at eight-hour admission/ten-hour hard limit, with whole-record checkpoints and explicit partial-work accounting. A failed admission is an operational result, not proof of model failure. Caffeinate follows the active controller only when a run is separately requested. No process is launched by shipping this revision.

Cost disclosure must include projection work `O(E)`, any competition/sort `O(N)`/`O(N log N)`, native all-cell sweeps, q/ring storage and query copies. Eight recent identities per synapse can be substantial; use a measured representation and reject infeasible memory before execution. Sparse KC activity is not evidence of sparse computation. This prototype makes no billion-neuron scaling claim.

The companion [SPEC.json](SPEC.json) records chosen bounds versus unresolved executable locks. [REVIEW_RESOLUTION.md](REVIEW_RESOLUTION.md) maps all ten review findings to this revision. The next authorized action in this turn is **design delivery only**; no implementation, three-cycle exercise, scientific experiment, adoption or GitHub push has occurred.
