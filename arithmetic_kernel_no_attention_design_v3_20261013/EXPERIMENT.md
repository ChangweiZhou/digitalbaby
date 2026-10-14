# KERNEL-SEARCH V3 — end-to-end kernels without attention

13 October 2026. Revised architecture proposal, not an implemented or qualified experiment. Supersedes the active roster in KERNEL-SEARCH V2 and the Transformer arms in E2E-RULE V1. Historical documents/results remain intact. No new training or scientific execution accompanies this revision.

## Fixed goal and architectural constraint

Learn arithmetic from observed feedback: retain exact own-byte/EOS generalization to unseen four-operation and bounded mixed-expression problems, with better ability per external training sample and actual computation. All semantic representations must be learned. The same kernel runs every operation unchanged; arithmetic-specific information belongs only in the experience stream and evaluator.

**Transformer and attention mechanisms are prohibited throughout the active model roster, including comparison baselines, encoders, workspaces, readouts, memory access and learning controllers.** This excludes self/cross/linear attention, query–key scoring, content-addressed key/value retrieval, normalized content-weighted pooling, attention-based external memory and fast-weight key/value attention. Removing softmax or renaming an attention block does not make it eligible.

Ordinary neuron gates in a GRU, elementwise feedback modulation, local convolution and an affine vocabulary classifier are permitted. The 257-way output softmax classifies payload bytes/EOS; it does not select or aggregate memory positions. Architecture choices may impose chronological/local wiring, but may not assign arithmetic meanings to units or provide numeral, operand, carry or operator features.

## Active references

**G0:** the ordinary learned-embedding GRU and affine byte head from E2E-RULE V1, without replay. **G1:** the same attention-free model with bounded replay of actual observations, retained as a learning-policy comparison. Neither reference has yet been qualified in this proposed comparison. T0/T1 are excluded from the active study, not relabeled as historical scientific failures.

Only the data, exposure, raw-byte interface and evaluation sections of [E2E-RULE V1](../arithmetic_end_to_end_design_v1_20261013/EXPERIMENT.md) are inherited. Its Transformer modules, complete roster, selection ordering, comparison family and job totals are not inherited.

## Candidate R: shared recurrent computation with chronological rereading

Use a learned byte embedding, a fast scan state `f`, and a slower temporary work state `s`. Both are unlabeled neural vectors, reset at public BEGIN. Before emitting each byte, perform a fixed number of complete chronological scans of the available raw prefix:

```text
for work_pass in fixed_pass_count:
    f = learned_initial_state(s)
    for b in available_prefix_in_order:
        f = GRU_fast(concat(E[b], s), f)
    s = GRU_work(f, s)
logits = affine(concat(f, s))
```

The same fast/work transitions are shared across passes, output positions and tasks. Every available position is visited in its declared order; there is no learned selection over positions, similarity search, query pooling or slot-attention block. The affine head receives terminal state directly. The work state persists between output bytes only within the current problem. Earlier own bytes can become transient context; they never become observed learning targets.

A starting prototype may compare eight shared passes against a one-pass twin and an eight-pass untied-transition twin. These are proposed settings, not a locked allocation. This differs from the old zero-input R4 loop: each pass rereads actual input and updates two interacting states. It may still collapse to constant output or memorize; recurrence is a hypothesis, not evidence of algorithm discovery.

Cost includes every prefix scan, recurrent transition, backward path and output decision. For length L, width d and K passes, the dense recurrent component is approximately O(K L d²) per decision, plus embedding/output and training costs. This is bounded prototype work, not a scalable-core claim.

## Candidate C: local recurrent cellular computation

Initialize a latent field from the raw prefix, one learned vector per chronological byte position. Apply the same gated local convolutional transition at each active position and internal step. Supply no operand rows, digit alignment, carry lanes, numerical initialization or arithmetic traces.

Read the next-byte logits **only from the terminal active position through an affine head**. There is no learned query or weighted pooling over the field. With neighbor radius r, choose a generic length-dependent budget of at least `ceil((L-1)/r)` synchronous steps so the first position can influence the terminal position. Record its hard context/step cap before testing. A small fixed step count cannot be presented as full-prefix access when its receptive field is too short.

Known prefix positions may exchange information in both directions; no future teacher byte is present in the prediction prefix. After an observed or emitted byte arrives, update the transient prefix and rerun the declared computation. Training and generation use the same access schedule.

For field width d, radius r and K steps, the dense local component costs approximately O(K L r d²) per decision, with O(L d) forward state and larger training storage. Local wiring removes content-based routing but does not make full-field work free. This is a different computation topology from Candidate R, not another loop count of the same model.

## Candidate P: feedback-conditioned recurrent modulation

Use an ordinary recurrent byte network with a bounded persistent fast modulation vector `m`, rather than a query/key/value memory matrix. `m` changes recurrent neuron gains or biases elementwise. The neural route and connectivity are fixed; modulation does not select among positions, slots or stored examples.

A small learned update network consumes its current modulation state, actual pre-observation recurrent activity and the prediction error from a genuinely arriving feedback byte. It proposes bounded elementwise retention/write gates and a new modulation vector. A short, declared multi-record gradient window can assign credit from later actual feedback to earlier updates. This is a proposed learning recipe; no private meta-training stream or uncharged examples are allowed.

During own generation and held-out probes, `m`, model parameters, optimizer state and pending learning state remain unchanged. Only transient recurrent/work state evolves. A whole no-write branch also freezes modulation updates and associated learning/RNG state. Matched stop-gradient-through-update and reset-modulation controls must be specified within the complete comparison; differing optimizer cadence must be matched in the attribution control.

This replaces V2's fast associative key/value proposal. No stored value vectors, query read, key matching or outer-product key/value retrieval is allowed. It can fail because modulation has insufficient capacity or interferes; an E1 gain alone does not establish arithmetic reuse. Parameter/state counts remain unverified until implementation.

## Data, evidence and resource accounting stay fixed

Keep all 256 categorical payload bytes, public generic BEGIN/REPLY/EOS, finite context/output limits and explicit truncation. No restricted answer vocabulary, task identifier, numeric renderer, answer-length hint, operand feature or evaluator AST reaches the learner. Use ordinary actual feedback CE; any extra target applications or replay are charged.

Keep opaque byte relabeling, whole binary-family and mixed-group holdouts, balanced observations, matched prior-teaching controls, later mixed learning and exact own-output scoring from V1. The model receives only raw problem/reply bytes; group construction and arithmetic checking remain evaluator-only. Causal controls and raw/canonical exact-key and feedback-frequency countermodels prevent taught-item recall or shortcut fitting from becoming an E3 claim.

The intended evidence is bounded retained E3. No new E0–E3 evidence is earned by this design revision. Exact taught recall remains an E1 diagnostic; numerical proximity remains diagnostic. Measure retained held-case accuracy and ability/sample and ability/paid-compute curves, including mandatory generation, replay, fast writes, movement, temporary storage and gradients. No new easier benchmark, noise objective, NLP side task or arithmetic rule is introduced.

## Before execution

The next implementation review must inspect the entire forward/read/write path, rather than only check class names: no similarity-based routing, sequence-weight aggregation or hidden attention library call. Verify byte causality, all-part gradient flow, generation purity, whole no-write state, independent mutable storage and actual counts. A source scan complements those checks; it cannot establish the absence of an equivalent mechanism by itself.

Measure complete-job costs and then prospectively specify the candidate/control roster, inference family and bounded screen/fresh confirmation allocation. Do not inherit V1's 17-test/88-job figures after changing its roster. Perform the requested design–audit–test–revise cycles within that complete study when authorized; do not turn this restriction into another detached series of localization experiments.

Eight-hour admission and ten-hour hard-session limits remain. A future authorized run must be detached with caffeinate and whole-job boundaries, without automatic retries or sample extensions. No implementation, runtime guarantee, adoption, upload or science launch is claimed here.
