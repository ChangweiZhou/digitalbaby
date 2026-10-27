# SHARED-POLICY V1 — can one shared controller use the native output channel?

**Date:** 2026-10-27. **Status:** DESIGN ONLY; no implementation, model loading, training, new query, launch, publication or adoption authorized by this document. **Identity:** `SP-COND-V1`, post V171-R1. A global numeric version is not allocated; Claude's V172 reservation is preserved.

**Goal:** better retained, complete own-byte/EOS generalization to never-taught addition problems per total training experience and actual compute. This bounded prerequisite tests **E1 shared-controller acquisition/conditioning**, not E3 generalization. Preserve gradual warmup, generic neuron operations, local/circuit-local candidate learning, no attention/Transformers/ELM, arithmetic-specific representations, supplied rules or independent answer head. All fitted arms here are explicitly nonlocal **optimizer instruments**, not newly admissible candidate cores.

## 1. Why this is the next question

V171-R1 already completed per-prefix joint LPs and exact convex fits. Do not repeat them. Final taught own answers were NATIVE/EXACT3/FULL 4/4/4 of80 in world711 and11/10/10 in world712. Legal independent-gate oracle bounds were47/80 and57/80; cap-aware outer relaxations42/80 and51/80. These oracles knew each target and did not use shared weights. They neither establish stored answer information nor supply a trained policy.

EXACT1 was numerically verified but not trained. Recurrent U/W/R/b gradients have substantial delayed components even though G/c dominate the whole-gradient norm. We therefore ask: **with one shared recurrent forward graph, does a fixed stronger optimization package improve complete taught answers, and does any gain depend on the actual native-memory signal or its feedback path?** This isolates a missing conditioning test before a new architecture or local-credit campaign.

The current failure is specific to V171's raw-byte/EOS protocol and budget. Historical Full151 taught-association results remain valid in their own scope. Taught retention is E1-R, not E2; E2 requires a never-reinforced label-preserving presentation change.

## 2. Fixed substrate and causal timing

Use the two already exposed V171 calibration FINAL native checkpoints, with their current hashes and retrospective panel authentication. They contain native knowledge of all80 OLD+NEW items. The native parent is frozen across fitting records; each record begins from an independent query fork of that same checkpoint. Within a query, native sensory/adaptation/time transitions run exactly as declared; **no native value teaching occurs**. This is not an online native-learning replication or a new parent trajectory.

Keep the V171 controller exactly: H64; 256 learned byte columns; recurrent fan-in4;16 native channels; gate fan-in8;17,872 ON parameters; identity leak.75 and innovation.25. For each raw payload byte b:

    z_t = tanh(U[:,b] + W_sparse h_(t-1) + R r_(t-1) + bias)
    h_t = .75 h_(t-1) + .25 z_t
    g_t = 1 + .5 tanh(G_sparse h_t + c)
    r_t = g_t * tanh(center(v_t)/4)
    logits_t = C center(g_t*v_t)/4.

The fixed C16 codebook covers all256 payload bytes plus EOS256. Keep four own decisions, maximum64 input bytes, the same EOS cleanup, and state reset after each complete record. No parser, digit order, operand slot, arithmetic intermediate or attention-equivalent lookup is added. h/r start at zero each record; learned parameters persist across fitting records.

At a target position, compute the distribution **before** exposing that target. Reveal its expected byte only afterward for diagnostic loss/credit and the next teacher-prefix transition. The target does not choose h, gates, carrier, record order or native query. Weight updates occur once at EOS; weights remain fixed inside a record. EOS is predicted and charged, then ends the record without an invented payload embedding.

During every new optimizer iteration, recompute the complete h/r trajectory from raw bytes and cached native v. Never fit against old saved h and call it a newly fitted recurrent policy.

## 3. Arms, births and optimization

Use fresh **controller** parameters/topology from the pinned V171 constructor and the world's existing salt, identical across arms before interventions. These are fresh fits on exposed diagnostic fixtures, not fresh worlds. No trained V171 controller is used as initialization.

| Arm | Controller update | Purpose |
|---|---|---|
| NATIVE | No controller | Frozen native own-answer baseline |
| SGD3 | Literal V171 exact-three conditional gradient, original fixed-step update | Original optimization package on the new frozen-parent conditioning protocol |
| ADAM3 | Exact-three conditional gradient, fixed Adam package below | Shared-policy conditioning reference |
| ADAM1 | Exact-one conditional gradient, same Adam package | Trained immediate-versus-three-step reference |
| ADAM3-R0 | ADAM3, but R and its optimizer state remain exactly zero throughout fit and evaluation | Causal native-feedback-path lesion |
| ADAM3-CONST | ADAM3 with one fixed positive carrier replacing native values at every input/read | Tests cue-to-gate memorization without native content |

There are **five fitted instruments**, one baseline, two native parent snapshots. R0 has1,024 fewer operative parameters and is a lesion, not a capacity-matched larger-state control. No main comparison of local rules is claimed.

All arms retain parameter caps `U:1, W:1/8, R:1/32, bias:1/4, G:1/8, c:1/4`, the original sparse topology and gate range[.5,1.5]. No parameter-cap ablation or rate grid is included in this bounded design. Projection is applied after each record and logged.

**SGD3:** retain the parent equation literally: accumulate target gradients weighted by `w(n)=1/16+(15/16)*min(n/2048,1)`; apply `-2^-10*clip(accumulated_gradient,-1,1)` and parameter projection. Its coordinate clipping can limit the effect of warmup; record the actual applied-step norms rather than assuming the nominal ramp determines them.

**Adam package:** beta1=.9, beta2=.999, epsilon1e-8, no weight decay, nominal eta2^-10, standard bias correction, moments reset at each controller birth and retained across OLD/NEW. Accumulate **unweighted** exact target-gradient sums within the record. After Adam preconditioning, clip each proposed coordinate direction to[-1,1], multiply the actual step by the record's mean w(n), then project. Applying warmup after adaptive normalization prevents Adam from cancelling the intended gradual start. R0 never accumulates or updates R moments.

This compares two declared optimization packages; their realized step magnitudes are not matched. A gain cannot be attributed uniquely to preconditioning rather than step scaling or warmup placement. Log per-block gradient, proposed/applied step, clipping, projection, cumulative update norm and warmup. ADAM1/3 share the same package, but their fitted states subsequently differ; do not confuse that outcome comparison with R1's identical-state derivative test.

## 4. Constant carrier with known output availability

For each world, set `v_const = s*ones(16)`, with `s=median(||v||_2)/4` from real **prompt-only** cached native reads, before fitting and without selecting by targets. If that median is zero, use the prespecified s=1 and disclose it. Hold the vector fixed for every record and phase, replacing native v both in recurrent feedback and logits. The controller still sees exactly the same raw bytes and feedback schedule.

Verify statically that C has unique balanced±1 rows. Then any class y has a strict independent-box witness `g=1+a*C_y`, for any0<a<.5: the constant carrier can express every class. This avoids a negative control that fails merely because most output classes are unavailable. It does **not** prove that shared capped weights can realize those gates. This is a deliberately strong cue-memorization control, not an extra deployed decoder or arithmetic prior.

If CONST matches or beats real native content, a successful shared fit is insufficient evidence of useful native-memory deployment. If real content wins, report it as conditional benefit in this fixture, not E3 reuse. The two carriers' optimization geometry still differs; a gap does not identify a unique storage mechanism.

## 5. Fixed exposed data, full traces and exposure audit

Use only the OLD64 and NEW16 rows for worlds202610269711/712 from the pinned V171 TRAIN_DATA artifact. Never read ASSESSMENT_SEALED or fit any held orbit. All relations are already taught in the frozen native parents. NEW means newly fitted by this controller after OLD, not a never-reinforced relation. The exact-key table can solve all taught cases: this is intentionally E1.

First gather and hash **complete bytewise native query traces**, including every prompt transition, the end-of-prompt refresh and every correct-prefix reply transition. The existing target tapes and R1 prefix arrays omit some prompt-byte reads and are not sufficient. Cache native values and events, not learned h. Native v is exogenous to this graph on a fixed teacher prefix, so traces may be shared across controller instruments without changing information. Own-generated prefixes require live frozen-native forks when they leave the cached correct path; do not substitute a teacher-prefix answer or invent off-path native values.

Per fitted arm/world: OLD16 passes of64 items, followed by NEW16 passes of16 items:1,024+256=**1,280 record applications**. Retain parent OLD permutation order repeated twice and selected NEW order. No replay, no random reshuffle by outcome, and no native writes. Controller warmup starts at n=0 and counts every arriving reply byte/EOS, including repeated presentations; queries do not advance that training counter. The frozen parent's mature warmup/state is inherited and reported separately, never reset or described as a fresh brain.

All fitting uses the final native snapshot throughout both phases. Consequently OLD-after-NEW interference here is controller plasticity on a fixed native substrate, not the native writer's original interference assay.

## 6. Endpoints and fixed reporting schedule

Evaluate all80 complete own-byte/EOS answers and teacher-prefix loss at controller birth, OLD8, OLD16, NEW8 and NEW16. After NEW16, additionally evaluate a native fork advanced by128 inherited rest steps, without controller training. h/r reset normally at each query. The full answer is correct only if every emitted payload byte and the model's own EOS match within the unchanged cap.

Report separately: OLD64, NEW16, all80, empty EOS, truncation, initially-correct OLD retained/lost, and exact paired gains/losses versus NATIVE and SGD3. Retention is measured from OLD16; an empty initially-correct set is unmeasurable. Also report a fixed shared initially-correct OLD intersection for paired comparisons. Losses, margins and gradients are diagnostic measures, never substitutes for complete answers.

Every planned arm completes its fixed dose unless a technical/resource stop occurs. Use the fixed **last** checkpoint, with no best-epoch selection on these exposed outcomes. Intermediate results cannot select a rate, arm, carrier or new dose. World-level paired tables are primary;80 items, shared native histories and repeated queries are not independent replications.

This is a mechanistic development test with **no binary efficacy/adoption gate** and no automatic next-stage launch. One changed answer is1/80 and must be shown as such, not promoted to a robust success. Qualification for a later independent experiment needs a prospectively justified useful effect and sample plan. The old V171 gate remains failed regardless of these outcomes.

## 7. Counts and resource envelope

| Quantity | Planned amount / ceiling |
|---|---:|
| Frozen native parent snapshots |2 |
| Fitted instruments |5 per world,10 total |
| Unique fitting relations |160 across the two exposed worlds |
| Complete cache-acquisition query episodes |160;460 reply/EOS prefix sites;810 prompt bytes plus boundary refreshes |
| New fitted-controller record applications |12,800 |
| Repeated reply/EOS targets in fits |36,800:3,696 per world711 fit and3,664 per world712 fit |
| Fitted-arm own/TF query pairs |4,800:2worlds×5arms×6panels×80 |
| Native baseline own/TF pairs |320:2worlds×2native states×80 |
| Total own/TF query pairs |5,120 |
| Synthetic scalar loss evaluations |At most64, separately charged |
| New held-assessment decisions |0 |

Native baseline predictions are identical across controller-fit checkpoints and may be reused with explicit hashes; count actual executions separately from logical comparisons. Cache acquisition is new non-updating observation, not new native teaching. Cached repetitions are still charged learning exposures for every instrument; evaluator instruments do not receive a zero-record exemption.

Charge inherited parent acquisition and all original V171/R1 reference/diagnostic work separately. Report physical native histories, logical controller fits, every probability evaluation, target reuse, optimizer pass, source import, failed attempt, output bytes, clone traffic, sorts/scans, CPU including children, wall and peak RSS. Shared/cached work must not be counted repeatedly as physical work or hidden as free experience.

Proposed bounded-run caps: one worker, one numerical thread;3,600 CPU seconds and7,200 wall seconds;1GiB peak RSS;256MiB output;120,000 distributions and120,000 native-vector reads. Perform one charged whole-record/cache/own-query benchmark before admission. If the fixed plan cannot fit conservatively, stop and return a reduced prospective protocol before any fit; do not silently drop controls. Use caffeinate tied to the active runner only. No automatic restart, failed-trajectory replay, search or budget extension.

## 8. Technical qualification and decision table

Before new fitting, hash the full Python/asset closure and source fixtures; verify output codebook, checkpoint panel parity and own/teacher-prefix timing. Independently check EXACT1 and EXACT3 gradients on a short episode, the original SGD step, Adam bias correction and actual-step warmup, sparse parameter projection, shared-birth equality and mutable independence. Exercise record roundtrip and write results/checkpoints **before** assertions that could lose data. Verify R0 remains exactly zero and CONST is fixed at every byte. All technical presentations and labels are charged; missing old arrays remain missing.

| Observation | Permitted conclusion / next action |
|---|---|
| ADAM3 improves complete taught answers over SGD3 and NATIVE across both fixtures | A usable shared policy is reachable under this optimization package; examine learning-signal units before changing topology. It does not prove a local rule or E3. |
| Loss improves but own answers do not | Record the same endpoint failure; do not promote confidence. |
| CONST is competitive | Controller-only cue memorization remains sufficient; native-memory use is not established. |
| R0 is competitive | No demonstrated added feedback-path value at this fixed budget. |
| ADAM1 and ADAM3 differ | A trained credit-horizon difference in this optimizer/graph; support only its observed scope. |
| All fitted references remain weak | Shared reachability, optimization and the restricted forward graph remain unresolved. Do not infer impossibility or enlarge the search automatically. |

## 9. Separate subsequent stage: generic every-position learning

This diagnostic deliberately retains answer-site supervision to isolate optimization. It is not the deployed learning objective. A later stage uses **fresh native histories**, predicts before every arriving byte, and applies the same generic circuit-local rule to prompt, reply and EOS positions. The addition stream may remain the sole task stream; no NLP task or arithmetic-specific structure is required.

Keep one declared warmup clock; account for additional supervised positions, cumulative updates and changes in total plasticity. Report prompt and answer/EOS losses separately so predictable syntax cannot conceal answer failure. Complete own-byte/EOS unseen addition, whole commutative-orbit holdout, matched prior-teaching inhibition, shortcut countermodels and relevant retention remain the behavioral target. Do not introduce this objective halfway through the conditioning comparison.

New state-dependent native addressing/reentry is outside this design. The operational distinction between inherited native synaptic transmission and prohibited attention-equivalent memory access must be resolved before that architecture is launched. A normal recurrent gate does not grant permission to transplant V166/V170 key/value retrieval.

## 10. Provenance and delivery

- [V171-R1 report and next-stage revision](https://github.com/ChangweiZhou/digitalbaby/tree/16623c73af101bbf9b017980795bfb957d444f0b/minifly_v171_r1_dev_20261026)
- [Latest critical review](https://github.com/ChangweiZhou/digitalbaby/blob/db4d2810d0ee31323d73e8d332fdbafc7705d2bb/PROJECT_DOCUMENTS/MINIFLY_MECHANISM_SYNTHESIS_20261026/04_POST_V171_CRITICAL_REVIEW_20261027.md)
- [Claude self-review](https://github.com/ChangweiZhou/digitalbaby/blob/fd80e118761d5106d804f7b39f3cab40e424d5f8/minifly_selfreview_20261027/NOTE.md)

`SPEC.json` records the arm/count/status contract; `PROVENANCE.json` pins actual local source/checkpoint hashes. This delivery creates a prospective document only. The completed diagnostic remains V171-R1; historical arithmetic/adoption pointers and frozen sources/results remain unchanged. No GitHub push is implied by document delivery.
