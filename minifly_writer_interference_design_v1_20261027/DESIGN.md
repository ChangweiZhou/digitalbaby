# WRITER-INTERFERENCE V1 — update form, delivered strength and later-learning loss

**Identity:** `WCFI-V1`. **Date:** 2026-10-27. **Status:** design only. No implementation, model query, training, adoption or GitHub push is authorized by this delivery. A global numeric version is not guessed.

**Research target:** a generic local learner that acquires, updates and reuses knowledge efficiently. Tasks are measurement instruments. This experiment tests native write mechanics (E0) and taught acquisition/retention (E1/E1-R), with no E2/E3 or arithmetic claim. Preserve gradual warmup, raw bytes, native own-byte/EOS output, local/circuit-local candidate rules, and no attention/Transformers/ELM or task-specific structure.

## 1. What V172 changed, and what it did not settle

V172 changed only writer configuration and KC homeostasis on the native substrate. Final complete taught answers rose from10/80 with N2B to29/80 with DELTA and39/80 with DELTA plus homeostasis. DELTA-H acquired50/64 OLD before NEW;23 of those50 survived NEW. This is a substantial configured-writer result in one development world.

Two causal questions remain open:

1. N2B and DELTA differ in error-vector form **and** delivered update size. The observed roughly8x coefficient difference is not a demonstrated8x synaptic or readout displacement. Both writers were already error corrective; this is not a sign-bug repair.
2. NEW combined time, sensory events, value writes and—in H arms—threshold updates. A128-step rest after NEW did not match the duration of128 complete NEW records and cannot identify all earlier losses as write interference.

The proposed design separates these questions. It does not repeat V171-R1's gate LPs, restore the old output-gate reader as a research priority, or treat an external perfect decoder as native competence.

## 2. Fixed native substrate

Pin V172 at `8c808c38fc82149085d01eb5bbc27d134d669997`, including its native dependencies and codebook. Keep16 canonical native compartments, FE-PERM fade.95, fresh adaptation, q=0 and the C16 port over256 payload bytes plus EOS256. Keep generation cap4; report CAP rather than forcing EOS.

**Homeostasis is OFF throughout this experiment.** This keeps the threshold rule fixed while testing writers; DELTA-H remains a historical reference, not a fifth main arm. Adaptation reset is already present and is not presented as a new repair. No native feedback/query path, changed wiring, extra decoder or new representation is added.

For observed target y, compute the prediction before revealing y, then define the unscaled generic error directions:

    p = softmax(C P16 v /4)
    f_N(v,p,y) = P16 C^T (p - e_y)/4
    f_D(v,p,y) = [tanh(P16 v /2) - C_y]/2
    coefficient = w(n) * alpha_(form,level) * f_form
    w(n) = 1/16 + (15/16)*min(n/1024,1).

Here `Pm = I - 11^T/m`. The existing signed native teacher consumes this coefficient; preserve its verified punishment-sign convention. n counts genuinely arriving payload/EOS targets, including observed-but-write-inhibited records. Prompt bytes are observed but not supervised in this answer-site diagnostic. The production candidate has only generic prediction/error and local native state, never an oracle estimate of a future write effect.

The codebook has unique balanced +/-1 rows. Thus `|f_Nj| <= .5` and `|f_Dj| <= 1`. Restrict gains to `alpha_N in [0,2]` and `alpha_D in [0,1]`, preserving the coefficient API bound for all future states. Log numerical violations; do not silently clip the coefficient and call its direction unchanged.

## 3. Two evidence layers — exact matched pulses versus local learning histories

**Layer A, the form/strength identification assay:** on cloned identical native prestates, compare the two update directions at the same measured write-to-read displacement, at low and high operating points. This is explicitly evaluator instrumentation; scalar yoking may use counterfactual clones. It earns E0 write-mechanics evidence, not a deployable local learning rule.

**Layer B, the longitudinal local learner:** use one fixed gain per form/level, calibrated on independent technical states before main learning. Run the four local writer configurations on common OLD checkpoints and matched NEW controls. No counterfactual gain solver or per-event oracle runs inside the learner.

**Identification boundary:** fixed gains match typical calibration dose, not necessarily every future state's dose. Layer B alone cannot establish a pure lifetime direction effect if its realized doses diverge. Layer A supplies the clean same-state comparison. Report these separately rather than calling all normalized trajectories perfectly strength matched.

## 4. Measure strength at the operative readout, with timing controlled

For a proposed pulse and zero-write control from the same prestate, measure

    delta_z = P257 C P16 (v_pulse - v_zero)/4
    d = ||delta_z||_2 /sqrt(257).

This removes value and softmax-logit common modes that cannot affect the prediction. Record native fast/slow displacement, coefficient RMS/max, clipping/saturation and output displacement separately. Equal coefficients, equal synaptic L1 and equal d are different notions.

Both branches use the same teaching time, sensory prefix, active teaching KC code, adaptation convention and post-event query condition. Native `values()` uses prospective event timing: do not subtract an ordinary pre-read from the next ordinary post-read and call the difference a write effect. Use fresh paired shadow clones and an independently checked fixed-condition read adapter. Verify its zero-write result and query code/time against the operative native reader before admitting the assay. No target-dependent choice of context, clock or read condition is permitted.

The oracle normalizes **magnitude only**, never selects an update direction by which one gives the right answer. All shadow writes, labels, reads and gain-search evaluations are charged. No shadow state enters a later learner or supplies its output.

## 5. Independent calibration and strength levels

Use a separate opaque calibration stream:32 distinct byte prompts,16 balanced reply symbols, four fixed passes, **128 actual training records**. Train a common DELTA parent with gain1, homeostasis off and the ordinary gradual warmup. This calibration history is not chosen by endpoint performance.

At its final checkpoint, select the first8 prompt IDs by a fixed hash, with both payload and EOS prefixes: **16 calibration contexts**. Querying the correct prefix exposes no new relation to this already taught calibration parent. In the shadow pulse instrument only, evaluate full-warmup coefficients to define physical units; the parent history retains its actual warmup.

Define independently of answer accuracy:

    D_low  = median_i d_i(f_N, alpha=.2)
    D_high = min[median_i d_i(f_N, alpha=2),
                 median_i d_i(f_D, alpha=1)].

The low point is anchored to the historical N2B operating strength; the high point is the common attainable endpoint rather than an arbitrary8x multiplier. Require a nonzero, numerically resolvable separation. Record the actual ratio, not a required performance effect.

For each form and level, solve for a fixed gain whose **median** calibration d matches the corresponding target. Use a predeclared bounded scalar routine with at most20 gain evaluations per solve, including bracket checks. Validate response monotonicity on the sampled bracket; if it fails or the target is unattainable, report calibration failure before main fitting. Do not choose the gain by task score or extend the grid automatically.

Engineering match tolerance is1% relative median dose, with an absolute numerical floor established by the zero-write qualification. This is an instrumentation tolerance, not a scientific success threshold. Publish the full dose distribution and mismatch, including tails, zero responses and saturation. Freeze the four gains and calibration/source/data receipts before any main outcome is opened.

## 6. Main data: opaque byte associations, not arithmetic

Use two prospectively reserved randomized worlds, tentatively `202610277311` and`202610277312`; verify freshness against the branch/version manifests before execution. Calibration/technical namespaces are distinct (`202610277390` and`202610277391`). These IDs are reservations, not a claim that execution has occurred.

Per world, choose8 cue bytes,2 shared-suffix bytes and16 reply bytes by a seeded random permutation of0..255. No ASCII/numeric meanings are used. Each prompt has three cue bytes followed by the same two suffix bytes; sample80 unique three-byte prefixes without replacement. The common suffix creates a controlled shared-context load, rather than depending on arithmetic syntax. All raw bytes, including NUL/newline if drawn, remain ordinary payload.

Assign OLD64 prompts to reply bytes with exactly4 examples per reply; assign NEW16 with exactly1 per reply, using independent seeded shuffles. Targets are one payload byte followed by the learner's own EOS. Response assignment is unrelated to prefix order, byte values or length. An exact taught-key table can solve this task; arbitrary never-taught mappings have no inferable answer and are **not** scored as transfer.

Before training, cache the native correct-prefix codes for these80 records per world:160 no-write coding episodes/320 target contexts total. Report exact code collisions with contradictory targets and overlap distributions; do not regenerate a difficult world or select cases by recall. A collision or a free decoder result is not a storage diagnosis. If this preflight invalidates the specified address-level comparison, return `CANNOT_TEST_AS_SPECIFIED` before main learning rather than silently replace the fixture.

OLD uses8 fixed permutations twice:16 passes,1,024 records. NEW uses8 fixed permutations of16 prompts:128 records per continuation. Data generators expose only raw bytes and genuine expected-byte feedback to the learner. OLD/NEW IDs, branch labels and the experimental clock are evaluator metadata, not neural features or teaching rules.

## 7. Common OLD preparation and the four local writer configurations

Prepare **one** OLD history per world with the historical DELTA gain1, homeostasis off and gradual warmup. This keeps the acquired state common to all subsequent comparisons and avoids confusing different initial OLD competence with interference susceptibility. It is not four independently trained OLD lives.

At OLD16, freeze the parent and all taught outputs. If it has no correct OLD answers, conditional retention is unmeasurable: keep the world and report that fact. Do not retrain it, replace the world or increase gain to manufacture a retention cohort.

Create four genuine NEW-learning continuations:

| Arm | Error direction | Fixed calibrated strength |
|---|---|---|
| N-low | N2B form | low |
| N-high | N2B form | high |
| D-low | DELTA form | low |
| D-high | DELTA form | high |

Each receives exactly the same NEW bytes, expected targets, orders and event times. Gains never change after calibration; w(n) never resets. The shared OLD history contains2,048 payload/EOS targets, so NEW begins with the inherited warmup already saturated. Do not describe this as a comparison of early warmup speeds.

Observe coefficient/read-response diagnostics at prespecified checkpoints to detect delivered-dose drift. Those observations cannot retune a main gain. If matched calibration dose diverges during learning, retain all outcomes but classify the longitudinal difference as a calibrated writer-package effect, not a direction-only theorem.

## 8. Matched interference controls

From that same OLD16 checkpoint also create:

| Branch | NEW bytes and feedback | Native value writes | Time |
|---|---|---|---|
| OBS | Identical full NEW stream | Inhibited | Same event ledger as learning branches |
| IDLE | None | None | Exactly the learning interval's elapsed native time |

Because homeostasis is off and all other rules are identical, one OBS branch can be physically shared across the four writer comparisons. It must execute the same prompt, prediction, target-arrival, teacher-call and clock sequence with the value-write flag disabled, then feed the observed byte normally. Inhibited feedback is still observed and counted. Verify all non-write event counts and native times match the learning branches.

At NEW passes1,2,4,8, measure the actual elapsed native interval `delta_t_k` from the common prestate. Obtain IDLE at each endpoint from a fresh prestate clone using `FastMemory.rest(delta_t_k)` in **native time units**, not `rest(number_of_records)` or a guessed128-step duration. IDLE sees no targets; its exposure counter is not artificially advanced. That difference is disclosed, and no IDLE branch is subsequently trained under a different warmup.

Probe OLD64 and NEW16 on read-only clones, with complete own-byte/EOS generation plus teacher-prefix diagnostics. Reuse the identical common-prestate panel explicitly; never count it as six independent acquisitions.

Primary causal contrasts, for OLD count or the same initially-correct OLD set:

    value-write effect = Y(writer learning, k) - Y(OBS, k)
    observation effect = Y(OBS, k) - Y(IDLE, k)
    time effect        = Y(IDLE, k) - Y(common OLD16).

These identify the branch interventions under this protocol. They do not prove that every lost item was overwritten on overlapping KCs. Negative write effects can include downstream consequences of the permitted value writes. Report code drift, realized overlap and non-value states; do not infer destructive overlap from prompt Jaccard alone.

Homeostasis remains disabled. A later homeostasis experiment would need a separate factor/control; threshold updates cannot be silently included in OBS or attributed to value writes.

## 9. Same-state matched-pulse assay on the common OLD parents

Select8 NEW prompt IDs per world by a label-independent fixed hash, both payload/EOS correct-prefix contexts: **32 contexts total**. These are planned E0 feedback interventions on discarded copies, not held examples or input to the main learners. All gains/data are already locked; these probes do not tune the longitudinal arms.

At each context, measure the common attainable displacement A_i under the two bounded directions. Scale the two global target doses by the same factor if necessary:

    s_i = min(1, A_i / D_high)
    target_(i,level) = w(n_i) * s_i * D_level.

This preserves the low/high ratio and avoids discarding hard states. Zero-response contexts are reported as unidentifiable, never dropped from the denominator. Solve for a scalar for each direction/level at the fixed prestate, using at most20 gain evaluations and the same1% dose tolerance. Abort the pure-form interpretation if matching fails; do not relabel unmatched pulses as form evidence.

Compare the target margin/CE change and the complete own outputs on8 fixed OLD sentinel records against the zero-write clone. Sentinels are selected by hash, not correctness or low interference. Publish all32 contexts, including no-ops, failures and saturation. At fixed same-state d, different beneficial/harmful effects provide evidence about **direction/geometry**; within a direction, low versus high provides an operating-strength response. These remain local pulse effects, not a guarantee about a full learning life.

## 10. Endpoints and inference

Primary behavior: complete own payload plus own EOS, within4 decisions. Report OLD, NEW, all80, initially-correct OLD retained/lost, EOS versus payload accuracy, empty EOS and CAP separately. Do not promote lower loss into complete recall. Any output outside the byte/EOS alphabet is an interface failure.

Publish per-world form, level and interaction contrasts, plus W-OBS-IDLE decompositions at all fixed checkpoints. The common initial OLD set makes within-world retention comparisons directly paired. Also show all64 OLD outcomes so an initially weak parent is not hidden. Facts, prefixes, forks and pulse copies are correlated; resampling80 items is not evidence across native births/worlds. Two randomized worlds with the same canonical wiring support a bounded development result, not a general mechanism confirmation.

No post-outcome absolute item threshold is introduced. Numerical calibration validity and causal-control integrity are engineering admission requirements; scientific outcomes may be positive, negative or unresolved. No architecture adoption, pointer replacement or automatic follow-up follows from a development contrast.

## 11. Counts and resource admission

| Work | Planned maximum / accounting |
|---|---:|
| Main common OLD preparations |2 x1,024 =2,048 full record applications |
| Read-only address preflight |160 coding episodes /320 target contexts; no teaching |
| Main NEW value-writing continuations |2 x4 x128 =1,024 records |
| Matched observed/write-inhibited NEW |2 x128 =256 records |
| Main total observed full records |**3,328**, each2 target arrivals =6,656 |
| Independent calibration history |128 full records /256 targets |
| Opaque qualification reserve |At most16 full records /32 targets |
| Complete-record total ceiling |**3,472 records /6,944 targets** |
| Calibration matched contexts |16 |
| Main matched-pulse contexts |32 |
| Shadow value-write attempts plus zero controls |At most4,096 target events; every gain evaluation charged |
| Main own/TF panels |4,000 physical pairs:160 shared prestate +3,840 continuation pairs |
| Matched-pulse OLD sentinel own/TF panels |1,280 pairs:32 contexts x5 interventions x8 OLD records |
| Additional qualification own/TF reserve |At most128 pairs |
| Total own/TF pair cap |**5,408** |
| Held-relation decisions |0 |

The4,096 shadow target events are partial teaching pulses, not zero-cost records or full trained lives. They are additional to the6,944 target arrivals in full histories. IDLE adds no observations. Shared prestates reduce physical work, not the inherited experience of a continuation. Report every native read/write, code computation, output distribution, counterfactual clone, label reuse, calibration iteration and failed attempt.

Proposed total caps:3,600 process-tree CPU seconds;7,200 wall seconds;1 worker and1 numerical thread;1GiB RSS and1GiB output;160,000 native-vector reads and120,000 distributions. Lock stage admission after a charged opaque full-event/probe/panel benchmark, before main outcomes. If the complete fixed roster cannot fit conservatively, return a smaller prospective protocol **before** learning; never drop controls after seeing scores or reset session accounting.

Serialize fixed assets by hash and save all necessary mutable state with verified event/optimizer-free roundtrips. Do not require a full native fixed-connectivity copy on every probe. Measure clone and serialization traffic separately. Leave caffeinate attached to the active worker; stop on completion/failure. No automatic retries, extra worlds, gain searches beyond calibration caps or trajectory replay.

## 12. Technical qualification and required artifacts

Before learning: pin complete source/assets/runtime and data manifests; verify no reserved world collisions; check both writer directions/signs and coefficient bounds; prove forced-zero control inertness; verify same-condition read/time/code equality; validate scalar matching and zero-response classification; verify gradual warmup counters; demonstrate one target is predicted before arrival; check all256 payload bytes and distinct EOS; validate readonly own generation; check matched W/OBS clocks and exact IDLE units.

Save event state and full response schema before any assertion could discard a partial history. Preserve failures and counted partial observations; do not recreate missing artifacts by replay. The report must include calibration gains/distributions, all same-state matching residuals, all branch/event/time receipts, complete output records, correct-set transitions, fast/slow update diagnostics and the form-versus-dose interpretation boundary.

## 13. Decision after this bounded study

- A direction difference at tightly matched same-state d supports an update-geometry effect under those conditions.
- A common low/high response with little matched-direction difference supports calibration/operating-strength as the leading local explanation.
- A reliable W-OBS OLD loss with successful NEW learning supports value-write interference; OBS-IDLE or IDLE-prestate losses instead locate other contributors.
- Stronger NEW acquisition with worse OLD survival is a stability/plasticity tradeoff, not a universally better writer.
- Calibration mismatch, weak OLD acquisition or variable worlds leave specified questions unresolved; no rescue run follows automatically.

Only then prioritize one protection or bounded replay mechanism. A replay follow-up must match NEW exposure, total target/record budget and event time: use identical replay-observation slots with native writes enabled versus inhibited, rather than comparing NEW-only128 records against NEW128 plus extra OLD training. Replay may use only genuinely observed historical byte/feedback records; never generated answers treated as new truth. Learned retention should preserve still-useful relations while allowing corrections, rather than freezing all past values.

Generic every-position next-byte learning and learned temporal representations remain separate architecture work. They should use the calibrated native writer as a candidate baseline, not infer a new architecture's necessity from a weak N2B parent. Answer-site E0/E1 diagnostics here do not establish a deployed generic learner or broader transfer.

## Provenance

- [V172 report](https://github.com/ChangweiZhou/digitalbaby/blob/8c808c38fc82149085d01eb5bbc27d134d669997/minifly_v172_native_e1_dev_v1_20261027/REPORT.md)
- [V172 writer implementation](https://github.com/ChangweiZhou/digitalbaby/blob/8c808c38fc82149085d01eb5bbc27d134d669997/minifly_v172_native_e1_dev_v1_20261027/v172_lib.py)
- [V172 history and rest controls](https://github.com/ChangweiZhou/digitalbaby/blob/8c808c38fc82149085d01eb5bbc27d134d669997/minifly_v172_native_e1_dev_v1_20261027/v172_dev.py)
- [V172 frozen design](https://github.com/ChangweiZhou/digitalbaby/blob/8c808c38fc82149085d01eb5bbc27d134d669997/minifly_v172_native_e1_design_20261027/DESIGN.md)

This proposal changes no historical verdict, native adoption pointer or completed-experiment status. The user requested a document only; implementation and execution require a subsequent request.
