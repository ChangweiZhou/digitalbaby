# MiniFly V177 — Connected Recurrent Predictive Core with Circuit-Local Temporal Credit

**Identity:** V177 / CRPC-LC-D1  
**Date:** 2026-11-02  
**Status:** DESIGN_ONLY; no model training, code implementation, confirmation or adoption authorized by this document  
**Frozen parent review:** digitalbaby@f0caf89792fda745227e9a534f18c57983140852  
**Primary question:** Can a sparse but connected recurrent learner acquire an unseen temporal dependency from genuine raw-byte prediction, and can an approximately local, forward-time credit mechanism recover a useful portion of the capability of the **same** network when trained using exact BPTT?

## 0. What has changed in scientific strategy

This is **not another patch to the Full151 mushroom-body associative memory**. V177 tests a new sparse recurrent computational substrate in parallel with the unchanged native fly. It deliberately omits attention, Transformer, associative key/value retrieval, ELM, hand-crafted arithmetic/position/operand/lag features, external fitted answer heads, SNN spikes, altered mushroom-body wiring, replay and consolidation. It neither imports V176's feature bridge nor retains the C16 output codebook: the candidate has a generic, trainable 257-class local output layer.

The point is to **separate expressivity from learnability**. First train exactly the proposed recurrent graph using BPTT as an *instrumental positive reference*. Only if that reference succeeds do we test the candidate forward-time local rule. A failed reference disqualifies claims about the local rule. A local success without full autonomous behavior cannot be called successful learning either.

Evidence earned at most: E0 technical correctness, positive/negative architectural feasibility on a stochastic temporal task, an exploratory local-learning result, measured sample/CPU burden, and genuine eight-step autonomous continuation. This is **not** persistent A→B→A retention, biological identification, natural-language learning, or the project's E3 never-taught arithmetic test.

### Why previous experiments force this separation

| Evidence | Supported finding | Limitation V177 addresses |
|---|---|---|
| V164 and V166 | Suitable recurrent networks learned delayed/compositional operations with BPTT under bounded scaffolds | Avoid V166's attention-equivalent memory read; establish a positive reference **on the candidate's actual graph** |
| V168 / V171-R1 | Temporal credit is real, but approximate recurrent credit, teacher-forced likelihood and the native reader's actual behavior can diverge | Compare local eligibility with BPTT on identical graph, inputs, budgets and own actions |
| V173 | Isolated eight-neuron gated circuits failed to qualify even their BPTT ceiling in the tested budget | Connect modules recurrently; do not treat a failed ceiling as an e-prop counterexample |
| V174 / WCFI | Local native writes acquire facts but produce OLD/NEW interference; replay helps retention without establishing net gain | Do not solve memory protection before demonstrating reusable computation |
| V175 | Coactivation inhibition changed codes without reliably reducing actual interference or improving recall | Do not substitute a decorrelation proxy for functional prediction |
| V176, completed 2026-11-02 | Locally learned H features improved fitted probes in both text pairs, but native behavioral deployment failed; TRUE-H held output emitted immediate EOS on 126/128 and 128/128 contexts | Use **real next-byte streams**, not training targets artificially terminated after one byte; keep payload and EOS diagnostics distinct |

V176's failure is not proof that learned representations were useless. Its one-byte-plus-EOS protocol had an inherent termination hazard, and the native output pathway was an additional bottleneck. V177 removes both confounds **without** claiming the resulting new readout is already a deployable native mushroom-body component.

## 1. Main benchmark: SDEL-8, genuine next-byte learning

The primary diagnostic is an **eight-symbol stochastic lag-eight stream**. This tests a learnable temporal dependence on unseen byte histories; it is not an arithmetic or natural-language capability claim. No network receives the lag, the latent Bernoulli events, numeral semantics, task markers or a clock input.

### Exact generator

For each independent world, uniformly choose eight distinct actual byte values from `{0,…,255}\{10}`, using independent SHA256-derived PCG64 seeds. The actual byte identities are random and distinct between worlds.

Generate each complete record as follows:

1. Sample the first eight body bytes iid uniformly from that world's alphabet.
2. For each subsequent body position `t=8,…,95`, with independent probability `7/8` set `x_t=x_{t-8}`; otherwise draw a fresh uniform symbol from the same eight-byte alphabet. A noisy resampling may equal `x_{t-8}`.
3. Append the literal newline byte `0x0A` **once** after byte 95. Then append the single reserved `EOS=256` *as a target*, not an observed payload. There is no artificial EOS after every prediction.
4. Reset transient neural activity between complete records; never reset learned parameters between training records or epochs.

A full record has 96 generated body bytes plus newline. The network consumes `x_0,…,x_{96}` and after each consumed byte predicts the *actual next arriving* target `y_t=x_{t+1}`, where `x_{97}=EOS`. Exactly **97 true next-byte/EOS targets per complete record**, with every target contributing to training. Its own previous prediction is used only during autonomous evaluation, not teacher forcing. The learner never sees `t`, the generator latent variable or a hidden “stop after one byte” signal.

For the scored lag-bearing body positions `x_8,…,x_{95}`, the generator-known Bayes oracle places probability `57/64` on `x_{t-8}` and `1/64` on each other symbol. Its conditional entropy is about **0.805 bits/token**, compared with marginal entropy **3 bits/token**. This is **an evaluator ceiling**, not a candidate model or a trained comparator, and the true lag is never supplied to the neural network.

The fixed-length delimiter is intentionally a diagnostic convenience; it cannot prove that a model can decide when arbitrary natural text ends. Score payload prediction, the predictable EOS **after an observed newline**, and prediction of newline itself separately. No externally forced EOS or masked output classes are permitted.

### World namespaces, splits and non-leakage

| Reserved ID | Role | Unique train / validation / held records |
|---|---|---|
| 202611027701 | Independent technical/pilot only | 32 / 16 / 32 |
| 202611027711 | Main development world W1 | 64 / 16 / 64 |
| 202611027712 | Main development world W2 | 64 / 16 / 64 |

These IDs are **reservations to be checked** against the live version/world registry *before implementation*, not claims that any training was run. Named independent RNG streams use the first 16 bytes of SHA256 of `V177|world|split|purpose` as a big-endian PCG64 seed; purposes include alphabet, body generator, model birth, feedback, pilot, order and evaluation. No relabeling or replacement based on outcomes. Detect and report exact record overlap before fitting, with invalid-world rules declared before any results are opened.

Train for **every true next-byte position**, including initial unpredictable bytes, newline prediction and genuine EOS. Primary held score uses the 88 lag-bearing body targets, with complete 257-class softmax (not an eight-class output mask). Keep the first unpredictable positions, newline and EOS as separate panels.

Held next-byte **own top-1 accuracy** must come from the model's ordinary unmasked output at the real observed prefix. For autonomous evaluation, take fixed prefix endpoints `t∈{7,15,23,31,39,47,55,63,71,79,87}` on every held record. After the observed prefix, predict and **feed back each of the model's own successive outputs** for exactly eight future body-byte decisions. Score per-step accuracy, exact-eight rate, and premature EOS as failure; never feed expected future bytes, force an output length token, repair errors or remove EOS from the vocabulary. There are **704 independent-start eight-step rollouts per world**, but their outcomes are correlated because they share trained weights and records.

Training-only countermodels `UNI` and `MARKOV1` use add-one-smoothed 257-class empirical distributions; `NGRAM8` uses an exact preceding-eight-byte table with train-only unigram backoff and reports held coverage. A separate `GEN-ORACLE` may inspect the generator's true rule only to set a nonlearned ceiling. No countermodel's lookup output can be credited to MiniFly.

## 2. Exact candidate recurrent architecture — no implicit attention

### Sparse connected topology

Use **128 recurrent neurons, arranged as 16 modules of eight**. Let `j=(m,k)` denote unit `k∈{0,…,7}` in module `m∈{0,…,15}`. Every neuron receives exactly **16 previous-state recurrent inputs**:

- all 8 previous-state neurons in its own module (including self);
- two units in each of the four other modules `m−2,m−1,m+1,m+2` modulo 16: for offset `o` the source local indices are `(k+o) mod 8` and `(k+o+3) mod 8`.

Thus each affine recurrent map contains **2,048 directed edges**, versus 16,384 for a dense 128×128 map. The full graph is strongly connected and has cross-module paths. This is an engineered topology, **not** a claim that fly KCs are wired this way. The topology and all tie/seed conventions are frozen independently of data labels.

Input is an immutable, generic `257×16` Rademacher matrix `E` with entries `±1/4`, seeded from `V177|world|E`. Given a received raw byte `x_t`, let `u_t=E[x_t]∈R^{16}`. This is a fixed byte projection, not a learned or supplied numeric magnitude representation; the recurrent and output parameters must learn the relevant dynamics.

**Synchronous forward equations:**

\[
\begin{aligned}
v_{j,t}&=U_ju_t+\sum_{i\in\mathcal N(j)}R_{ji}h_{i,t-1}+b_j,& a_{j,t}&=\tanh(v_{j,t}),\\
q_{j,t}&=C_ju_t+\sum_{i\in\mathcal N(j)}S_{ji}h_{i,t-1}+c_j,& g_{j,t}&=\sigma(q_{j,t}),\\
h_{j,t}&=(1-g_{j,t})h_{j,t-1}+g_{j,t}a_{j,t},\\
z_t&=Oh_t+d,&p_t&=\mathrm{softmax}_{257}(z_t).
\end{aligned}
\]

Hidden state `h_{-1}=0` at each record start. `O∈R^{257×128}` and `d∈R^{257}` are a genuine **learned** output port, not the inherited C16 codebook or a post hoc free evaluator. All 257 classes remain legal at all timesteps. There is no positional encoding, content-addressed read, attention, pretrained encoder, or arithmetic-specific parameterization.

Per hidden neuron, the two affine maps have `16 input +16 recurrent +1 bias` each; **8,448 learned hidden scalars** overall. Output contains **33,153** learned scalars. **41,601 learned scalars total**; `257×16` immutable embedding excluded. Initialize `U,C` with variance `1/16`, nonzero allowed `R,S` with variance `1/16`, `O` variance `1/128`, and `b=d=0`. Gate initial bias `c` is frozen by C1's independent pilot from the set `{0,−1.5}`; do not encode the benchmark's lag of eight.

All three learning arms have the **identical active forward architecture and initial weights for a world**. Cross-module feedback is an actual state-transition edge. The native Full151 memory is unchanged and does not need to be instantiated to run V177.

## 3. Three matched learning arms and exactly what “local” means

At every true target `y_t`, set `e_t=p_t-\operatorname{onehot}_{257}(y_t)`. All arms learn the 257-class output port with the ordinary **output-local** cross-entropy gradient. Only the credit to the recurrent/input/gate parameters differs. Parameters are fixed within a minibatch of **8 complete records**, then updated once; hidden states/eligibility reset between records, not between bytes. No target is allowed into the state transition until after its prediction.

### BP — full-sequence BPTT reference

Autograd (or independently equivalent exact BPTT) across all 97 steps, through the **same sparse connected graph**. BP is a positive reference/diagnostic instrument, **not an admissible biological/local learning method**. If it fails at the fixed development budget, V177 has no basis to evaluate whether a local learner could have succeeded.

### BS — exact within-module forward sensitivity, optimistic feedback

Let module `m` have eight hidden units and parameter vector `\theta_m∈R^{528}` containing every incoming `U,R,b,C,S,c` parameter of those units, including their cross-module incoming synapses. Let the complete recurrent state Jacobian be `J_t`, with `J_{mm,t}∈R^{8×8}` its own-module block. Define `F_{m,t}∈R^{8×528}`:

\[
F_{m,t}=J_{mm,t}F_{m,t-1}+P_{m,t},
\qquad
P_{m,t}=\left.\frac{\partial h_{m,t}}{\partial\theta_m}\right|_{h_{\neg m,t-1}\,\mathrm{fixed}}.
\]

For neuron `j` with presynaptic feature `\varphi_{j,t}=[u_t,h_{\mathcal N(j),t-1},1]∈R^{33}`, the nonzero direct terms are

\[
\frac{\partial h_{j,t}}{\partial\theta_{v,j}}
 =g_{j,t}(1-a_{j,t}^2)\varphi_{j,t},
\quad
\frac{\partial h_{j,t}}{\partial\theta_{q,j}}
 =(a_{j,t}-h_{j,t-1})g_{j,t}(1-g_{j,t})\varphi_{j,t}.
\]

The Jacobian entries for connected `i` are

\[
\frac{\partial h_{j,t}}{\partial h_{i,t-1}}
=(1-g_{j,t})\mathbf 1_{i=j}
+g_{j,t}(1-a_{j,t}^2)R_{ji}
+(a_{j,t}-h_{j,t-1})g_{j,t}(1-g_{j,t})S_{ji}.
\]

In BS, the current error at a module uses **exact instantaneous output transpose feedback** `\delta_{m,t}^{BS}=O_{[:,m]}^\top e_t`. Its approximate hidden-parameter update is the normalized minibatch sum of `F_{m,t}^\top\delta_{m,t}^{BS}` over each real target. BS is **not an exact full-BPTT gradient** while cross-module recurrent paths exist; external module state derivatives are deliberately omitted. It also uses output-weight transport, so it is an **optimistic circuit-credit reference**, not the final local candidate.

### BR — same eligibility, fixed independent random feedback: candidate

BR shares the same exact forward and eligibility equations as BS, but its module learning signal is

\[
\delta_{m,t}^{BR}=B_m e_t,\qquad
B_{jk}\overset{\mathrm{iid}}{\sim}\mathcal N(0,1/128),
\quad B\in R^{128\times257}.
\]

The random `B` is independent of learned output `O`, label identities, the current output bias and all main scores; it is seeded once per world and never modified or copied from `O`. Hidden updates are the same sum of `F_{m,t}^\top\delta^{BR}_{m,t}`. No autograd or backward sweep across time is allowed in BS or BR during learning.

**Locality qualification:** BR uses a globally broadcast 257-class error, followed by fixed projections to modules and within-module eligibility matrices. This is **circuit-local approximate temporal credit**, not strictly synapse-only plasticity and not exact RTRL on the connected whole graph. In particular, cross-module *forward* computation exists, while cross-module *temporal sensitivities* are dropped. The hypothesis is whether the latter omission is tolerable at this task/budget.

**Attribution:** BP passes, BS fails → loss of cross-module temporal credit or associated fixed optimization is implicated. BP and BS pass, BR fails → fixed-feedback signal quality/alignment is implicated. BP, BS, BR pass → a meaningful new candidate for later persistence experiments; no native adoption follows automatically.

### Matched optimizer and freeze

Use AdamW-style per-coordinate first/second moments (beta1=.9, beta2=.999, eps=1e−8), row-wise gradient norm cap **1.0** for each hidden neuron and output row, decoupled decay **1e−4**, and learning rate `\eta` selected solely in C1. Accumulate raw gradients over the eight-record minibatch; normalize by the actual number of targets, then clip rows and apply the optimizer step. No global cross-module norm clipping in the candidate.

The true arriving target counter `n` increments for **every** one of 97 targets per full record. Apply the owner's warmup to the **actual optimizer/decay step** rather than only scaling the gradient:

\[
w(n)=\frac1{16}+\frac{15}{16}\min(n/1024,1).
\]

Use the mean of `w(n)` over the target arrivals of a minibatch for that update in every arm. Log applied step, clipping fraction, moments, true target count, and separate output vs recurrent updates. `E,B` never train or decay. Do not search learning rates or clipping after seeing W1/W2 outcomes.

The required block eligibility tensor is `16×8×528=67,584` scalars; multiplication `J_{mm}F_m` costs around **540,672 multiply-accumulates per time step** in a direct implementation, before other operations. This is *not* promised efficient. Whole-roster CPU admission is mandatory, and compiled exact loops may be used only after mathematical parity with the original equations. Do not silently approximate or truncate eligibility to fit a budget.

## 4. Four-stage execution logic (not authorized by this document)

### C0 — protocol/source qualification, without training

1. Verify all world IDs are unused, no version collision, source/runtime closure, exact data-stream generator, 96+newline/EOS contract, real target timing, and no future targets or task-position features in the network.
2. Record V176's reported empty EOS behavior from available **aggregate** outputs only; do not allege a raw forensic analysis without its unpublished binary states/panels.
3. Verify exact topology, 257-class unmasked readout, one genuine EOS per record, per-neuron initialization, and cross-module state dependencies. Verify train/validation/held uniqueness and mark unexpected duplicates `INVALID_DATA` rather than silently replace scored cases.
4. Confirm the generator-known oracle is computed by the evaluator only; a candidate does not import or inspect its lag, conditional distribution or alphabet assignment.

### C1 — numerical/technical qualification and independent reference-only pilot

**Technical tests** (≤32 additional full training records):

- Independent forward formulas, deterministic births/tie handling, safe reset/restore, correct target shift and final EOS; test all 256 bytes and the reserved EOS class.
- Finite-difference and float64 autograd verification of the gated Jacobian and `P_{m,t}`. With cross-module connectivity set to zero in a tiny *technical* fixture, BS's symmetric-feedback forward gradient must match the exact within-module BPTT gradient (relative error ≤1e−7, absolute tolerance ≤1e−9 on tiny components). With actual connectivity present, record the **expected** BP–BS discrepancy rather than declare it a bug.
- Verify BR uses independent fixed `B`, no backward/autograd operation in either local learner; zero `B` freezes the hidden gradient while the ordinary output head remains learnable. Check that the candidate carries within-module eligibility over the entire record, not just the previous step.
- Verify real applied Adam moments/warmup, output-row and neuron-row clipping, checkpoint/source/RNG exact restoration, read-only evaluation, 704-rollout own feedback, actual target exposure accounting, and no hidden EOS mask.
- Measure representative full-record training/validation/rollout costs on BP, BS, BR; include the entire conditional roster, pilot, probes and failed attempts with a **1.5× reserve** in a whole-plan CPU/RSS/output admission. If it does not fit, record `RESOURCE_NOT_ADMITTED` before opening main results; no shortening of epochs/worlds or dropping expensive arms.

**Fixed pilot** on technical world 202611027701: only **BP** in four configurations `\eta∈{0.001,0.003}` × `c_{\mathrm{init}}∈{0,-1.5}`. Each consumes exactly **5 epochs×32 records=160** full record presentations. Total **640 pilot records**, all charged. Select *final-epoch* pilot validation lag-bearing bits/token, tie-break smaller learning rate then bias 0. Freeze this one configuration for all main arms and both new worlds, from new matched births. Pilot checkpoints do not enter W1 or W2. A poor five-epoch pilot curve must be disclosed; only technical/numerical failure is grounds to silently stop before the fixed C2 reference, not post hoc replacement of the main graph.

### C2 — prove the same graph is BPTT-trainable

Train BP on **each** of W1 and W2 for exactly **10 epochs×64 train records=640** record applications per world. No early/best-epoch checkpoint selection and no main learning-rate adjustments. Record every epoch and the final, previously unopened held score; own eight-step generation and genuine EOS metrics remain separate.

**C2/BP positive gate (both worlds):**

- held 88-position lag-bearing payload NLL improvement over train-fitted UNI **≥0.75 bits/target**;
- held single-next-byte unmasked top-1 accuracy **≥35%**;
- no implementation/source/target leakage violation, nonfinite updates or resource cap breach.

If either world fails: **REFERENCE_NOT_QUALIFIED**, stop *without executing local C3*. This is a negative result on the specified graph, regimen and task, not evidence all recurrent nets are incapable.

**EOS diagnostic flag, explicitly separate:** report premature EOS fraction on ordinary held payload sites and fraction correctly predicting EOS immediately after observed newline. Set `EOS_PROTOCOL_WARNING` if early EOS >5% or post-newline EOS <90%. Even if this flag is raised, the causal temporal-dependency gate is based on the **unmasked** payload predictions and own first step; do not claim complete autonomous long-text ability. Do not “fix” EOS by output masking or artificially adding extra EOS supervision.

### C3 — evaluate local credit only if both BP references passed

Train BS and BR independently on W1 and W2, each from the **original matched initial weights** with the fixed C1 configuration and same data/epoch order. No BP-learned checkpoint, free evaluator readout, expected future answer or teacher-forced candidate selection is transferred to local arms.

Let `G_a(w)=NLL_{\mathrm{UNI}}(w)-NLL_a(w)` on the identical 88-position held payload; BP has qualified `G_{BP}(w)≥0.75`.

**S1: block credit reference.** For each world, `G_{BS}(w)≥0.5G_{BP}(w)` and own top-1 ≥25%. If false, BS is insufficient at this budget; any BR failure does not uniquely identify random-feedback alignment.

**S2: candidate local credit.** For each world, `G_{BR}(w)≥0.5G_{BP}(w)`, own top-1 ≥25%, and early EOS on held body sites ≤20%. This earns a **local capability development screen**, not a persistence result.

**S3: capacity per compute.** If S2 passes, require `(G_{BR}/CPU_{BR})/(G_{BP}/CPU_{BP})≥1/3` on both worlds, with CPU charging all actual training and comparable inference. Pilot/technical costs are charged to total research resources, not silently allocated to neither arm. Passing S2 but failing S3 is **LOCAL_CAPABILITY_ONLY_NOT_EFFICIENT**.

Also report (not select): exact-eight autonomous continuation, own-versus-teacher-prefix divergence, nonmasked 257-class payload/EOS losses, gradient **and applied-step** cosine BS/BR vs BP on frozen diagnostic states, cross-module-edge and recurrence-off inference lesions, hidden/feedback/output norms, input diversity, optimizer steps, clipped-row fraction, full CPU/wall/RSS/output, and every failed attempt. A positive cosine or teacher-forced loss result cannot override behavioral failures.

| Stage/outcome | Correct conclusion | Stop rule |
|---|---|---|
| C1 technical/cost fails | INVALID / RESOURCE_NOT_ADMITTED | No C2/C3 |
| C2 BP reference fails | REFERENCE_NOT_QUALIFIED | No C3; local credit not adjudicated |
| BP passes, BS fails | Missing intermodule temporal credit or fixed local optimization remains plausible | Do not adapt BR based on opened scores |
| BS passes, BR fails | Fixed feedback is a plausible limiting mechanism | Do not fit `B` retrospectively |
| BR S2 passes, S3 fails | Local learning capability with unacceptable compute ratio | No efficiency claim or adoption |
| S2 and S3 pass on both worlds | PROMISING_LOCAL_CREDIT_DEV | Requires an entirely new persistence/composition experiment and fresh confirmation |

These are *engineering decision thresholds*, not p-values. Two development worlds with within-life correlated predictions are not independent confirmations.

## 5. Precise budget and source-lock contract

| Activity | Full training records |
|---|---:|
| Technical qualification maximum | 32 |
| Independent BP-only pilot, 4×5×32 | 640 |
| Main BP, 2 worlds×10×64 | 1,280 |
| Main BS + BR, 2 methods×2 worlds×10×64 | 2,560 |
| **Maximum including technical** | **4,512** |

The pilot+main record total is **4,480**, under a **5,000 full-training-record cap**. Each full record has **97 next-byte/EOS targets**; main BP/BS/BR therefore consume **3,840×97=372,480** target applications, with **437,664** possible including pilot and maximum technical records. Count unique observed records separately from repeated optimization presentations. Read-only model probes, own rollouts, extra forward evaluations, derivative references and failed partial events are separately recorded and charged, never called free data.

Global caps: **7,200 CPU seconds**, **14,400 wall seconds**, **2 GiB peak RSS**, **2 GiB produced artifacts**. Single deterministic CPU-only worker, one declared BLAS thread; optimized compiled local recurrences only after exact parity. No background job is created in design-only state. If a full conditional roster×1.5 engineering forecast is not admitted before main data, stop without silently compressing network, lowering passes, excluding evaluation, enabling undocumented GPU, or increasing budgets. An explicit prospective owner-approved amendment is needed for a different cap.

For any future authorized run, source-lock the exact model/data/optimizer implementation, dependency/environment versions, seeds and world stream hashes, pilot configuration decision, generator truth hidden from learner, full fixed output alphabet, training and probe counters, optimizer/eligibility state, receipts/checkpoints and complete failure history. Original raw bytes, hidden states and panel traces may remain local under the project's no-raw-publication convention, but a future independent audit must clearly distinguish a public aggregate hash check from actual raw scientific replication.

## 6. Interpretation and next step, strictly out of scope

A BP-positive result verifies the new graph is representationally and optimizationally adequate on this *bounded* temporal task. A BS/BR-positive result gives an empirically anchored circuit-local temporal-credit mechanism. Neither result implies a five-year-old intelligence, learned addition, useful long-term memory protection, or success on open-ended prose.

A later, **separately versioned** test may train A→B→A-correction in one learner **without reinitializing weights**, add fast/slow synaptic timescales and evaluate still-relevant retention, legitimate correction and never-taught arithmetic relations. It must also pay for every observation, optimizer reuse, rest, replay and CPU second. Do **not** add Full151 associative storage or extra consolidation inside V177.

**Governance:** V177-D1 is one fixed proposal. Documentation-only corrections are labeled V177-R1 with a change log; a materially different topology, new learning rule or different goal requires the next unallocated experiment number after checking all branches. **This document does not allocate V178.** The adopted core and existing V176/V175 reports remain unchanged.

## 7. Sources

- James M. Murray (2019), *Local online learning in recurrent networks with random feedback*, eLife 8:e43299 — RFLO approximation and fixed random feedback, not a V177 performance guarantee: https://elifesciences.org/articles/43299
- Bellec et al. (2020), *A solution to the learning dilemma for recurrent networks of spiking neurons*, Nature Communications 11:3625 — eligibility/learning-signal decomposition, not evidence that V177 is an SNN: https://www.nature.com/articles/s41467-020-17236-y
- [V173 microcircuit failure](https://github.com/ChangweiZhou/digitalbaby/blob/claude/charming-clarke-u9g850/minifly_v173_microcircuit_dev_v1_20261028/REPORT.md)
- [V168–V171 temporal credit diagnosis](https://github.com/ChangweiZhou/digitalbaby/blob/codex/minifly-consolidated-20261023/PROJECT_DOCUMENTS/MINIFLY_MECHANISM_SYNTHESIS_20261026/02_MATHEMATICAL_DIAGNOSIS.md)
- [V176 completed source/probe/native results](https://github.com/ChangweiZhou/digitalbaby/blob/codex/minifly-consolidated-20261023/minifly_v176_feature_deployment_dev_20261101/campaign_20261102/SUMMARY.json)
- [V176 native EOS and payload outcomes](https://github.com/ChangweiZhou/digitalbaby/blob/codex/minifly-consolidated-20261023/minifly_v176_feature_deployment_dev_20261101/campaign_20261102/NATIVE_RESULTS.json)
- [Standing no-attention requirement](https://github.com/ChangweiZhou/digitalbaby/blob/codex/minifly-consolidated-20261023/PROJECT_DOCUMENTS/NO_ATTENTION_KERNEL_REQUIREMENT_20261013.md)
