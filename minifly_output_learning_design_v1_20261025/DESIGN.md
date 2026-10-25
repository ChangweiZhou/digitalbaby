# MiniFly output coding and local teaching — refined diagnostic V1

25 October 2026. **Design only; no implementation, model test, training or GitHub push is reported.** This is a standalone output-subsystem study in the existing project. It does not reserve V149, replace the active addition experiment, create a new chat or adopt a new core.

The project goal remains **better complete own-byte/own-EOS generalization to unseen addition per total training experience and actual compute**. This diagnostic tests a possible obstacle to that goal. It is not an NLP campaign, an arithmetic solver, or evidence that the obstacle is the project's universal root cause.

## 1. Questions and executable lineage

1. With the current categorical probability decoder, does independent-bit teaching optimize the wrong objective?
2. Does a locally learned compact code improve conditional prediction beyond learning symbol frequencies?
3. Can the same output error be used by a small, unchanged native storage circuit without losing its descent direction?

Parent: commit `6f4a46a61ebbacf5a998b7f405ac4f029e564962`, especially `minifly_cls_dev_20261017/fastmem.py`, its fixed output codebook, and V148's corpus/probe/ladder code. PROVENANCE.json records exact source identities and data offsets. The original V148 and exploratory reports remain unchanged.

Retain FE-PERM and its normalization, canonical connectivity, record timing, 16 native compartments, fast/slow/adaptation equations, and recovered adaptation at record start. Set `homeo=False`, keeping its birth thresholds/frequency fixed, in **all** new native arms. This freezes the address construction while native value plasticity remains active. No raw-trace, recurrence, learned address layer, extra memory bank or new noise mechanism is added. This controlled setting and gradual warmup differ from historical V148: its 7.104 score is context, not a numerically matched baseline.

No attention, Transformers, ELM/ridge solve, pretrained features, arithmetic parser, numeric-distance target or arithmetic rule enters a candidate. A vocabulary softmax is output competition, not attention over inputs or memories.

## 2. Correct the representability claim before testing learning

There are V=257 output symbols: all 256 bytes and EOS=256. Every symbol remains legal in every context; there is no seen-symbol mask or renormalization over observed symbols. Width d=16 is fixed.

Let v be the 16 native values, P=I−11ᵀ/16, u=Pv/4, and

`logits_y = C_y dot u; p = softmax(logits)`.

The native centering matters: u lives in a **15-dimensional** subspace. The historical real-code reference jointly fit unconstrained 16-dimensional context vectors and codes; it also used a much larger count table and a numerical optimizer. Its 3.725 bits result is a development feasibility witness in that family, not a promise for this native family.

For positive target distributions define context-by-symbol log odds D[q,y]=log(p(y|q)/p(y0|q)). Without class biases, rank(D) cannot exceed rank(CP), at most 15. Thus compact width does not express arbitrary 257-way distributions across arbitrary contexts. A low-rank fit on one text establishes neither a global capacity ceiling nor future language transfer.

For fixed C, exact expression additionally requires each context's log-odds vector to belong to the column space of the corresponding code-difference matrix. For an ideal freely indexed context table and learned C, rank(D)<=15 is also sufficient without output-amplitude limits: factor the log odds in that subspace, set the reference class code to zero, and rescale C downward and context values upward to meet the code-norm constraint. This constructive statement does not guarantee native accessibility or convergence from birth.

For a single context, a learned real code has a simple expressivity witness: choose unit w perpendicular to 1, u=a w, and C_y=(log q_y−b)w/a. It gives softmax(Cu)=q. Choose a large enough to satisfy the code row-norm bound below. This assumes unconstrained output amplitude; whether native values can achieve the needed amplitude is a separate measurement.

Stage A must check fixed-code rank, objective derivatives, centering and initialization. Fixed-C maximum-likelihood optimization is an evaluator reference only. Report convergence residuals and feasible constraints; absent a certified bound, call it the best numerical fit found, not a proof of impossibility. Joint code/context optimization is nonconvex and is only a feasible witness.

## 3. Specify the actual local updates

For observed target y, r_k=p_k−1[k=y]. The exact output-state gradient is

`g_v = P Cᵀ r / 4`.

For code learning, `g_C = r uᵀ`. Compute **all gradients and probabilities from the same pre-update v and C**, then update. Do not recompute the code gradient after the native target write.

These are gradients at the output circuit. They are not gradient-free and do not provide BPTT or exact gradients through the sensory encoder, TopK, native dynamics or prior events. Each code synapse uses its output neuron's error r_k and presynaptic u_j; the native compartment receives broadcast g_v,j. The biological implementability of normalized output competition is not established by the algebra.

Every learning channel uses `s(n)=1/16+(15/16)min(n/1024,1)`, where n is the number of **previously observed target symbols**, including EOS and repetitions. Increment after the entire target event. This token-based ramp makes unequal line lengths comparable. It is an explicitly new common schedule, not an unchanged historical V148 setting.

Main arms:

| Arm | Output code | Teaching |
| --- | --- | --- |
| A: C16 legacy | Fixed original signed C16 | e_j=[tanh((Pv)_j/2)−C_yj]/2; common eta and warmup |
| B: C16 coherent | Same fixed C16 | Exact categorical g_v |
| C: learned compact | Start from identical C16; learn real C | Same g_v, plus local code update |

In the output-only reference, v←v−eta s(n)e for A and v←v−eta s(n)g_v for B/C. For C, update C←C−(eta/16)s(n)g_C, then project each row to zero mean and norm at most 4. Original C16 rows already satisfy these constraints. This pins the otherwise unstable code/value scale and retains the native centering; it is a declared engineering constraint, not a proven optimal regularizer. No weight decay, momentum or auxiliary reconstruction loss is used.

For fixed C, the expected categorical loss in v is convex and its Hessian norm is at most 1 under this code/scale constraint. The declared rates are within the safe range for deterministic expected-gradient descent. This justifies a conservative rate range; it does not guarantee convergence of stochastic updates, jointly learned C/v, or native dynamics. The 1/16 code-rate ratio is a pinned slow-code engineering choice, not a derived optimum.

In the native adapter, send `eta s(n)e_j` (A) or `eta s(n)g_v,j` (B/C) to the existing signed native write. The native sign must be verified first: positive signed input must move the measured value in the negative direction on the sampled active addresses. Do not assume that sending a categorical gradient makes nonlinear synapses execute gradient descent. Do not silently rescale, clip or reverse coefficients to rescue a failed integration.

C contains 4,112 mutable code entries in addition to the same 16 native stores. Fixed width does **not** imply vocabulary-independent storage or runtime: code memory and dense output competition are O(Vd). Record these costs. Moving C can change the meaning of previously stored output values; log code drift and report any learning/retention tradeoff without adding a stabilization arm mid-run.

## 4. Stage A: mathematics and cheap qualification

Before fitting:

- Numerically check g_v and g_C against centered finite differences in float64, on interior code rows. Relative error <=1e−6 with denominator max(1, absolute values) is a numerical check, not a learning threshold.
- Verify probability sum, finite log-sum-exp, all-byte/EOS legality, null-direction invariance and code constraints. Do not mask invalid/rare symbols out of the vocabulary.
- Check native clone independence, clocks, counters, code serialization and read-only probes. Output C and warmup n must be included in state hashes and checkpoints.
- Use at most 64 isolated native signed-write fixtures: two training-only contexts × 16 compartments × two coefficient signs, with magnitude 1e−3. Use disposable matched clones, preserving the sensory event/timing, and log exact sign and self/cross response. No historical trajectory is replayed. Zero or inconsistent native response is INTEGRATION_NOT_QUALIFIED; it is not proof that a coherent output objective is ineffective.

A deterministic output implementation must generate its own EOS. Cap exhaustion returns CAP/TRUNCATED, never a forced EOS. No generated guess becomes feedback.

## 5. Stage B: the smallest output-learning reference

This phase is explicitly an **output-only reference**, not native Full151 or an end-to-end task solver. Its state table v(context) gives the output circuit a perfect frozen context key so failure cannot be blamed on upstream addressing. It is not proposed as a replacement memory.

Use three fixtures, with all 257 target probabilities strictly positive:

- Uniform: one context, q_y=1/257.
- Skew: one context, q_y=0.5·1[y=y*]+0.5/257, with y* randomly assigned.
- Conditional: four arbitrary context bytes; each has a disjoint random four-symbol set A_c, with q(y|c)=0.9·1[y in A_c]/4+0.1/257. Contexts are balanced. No numeric or linguistic relation is supplied.

Use seeds 11,12,13 and eta in {0.03,0.1}. Derive independent PCG64 substreams from the first eight SHA256 bytes of `MINIFLY-OUTPUT-V1|fixture|seed|stream` (big endian) for symbol assignment, record order and target draws. Initialization is identical C16 and v=0 in all arms. Each fit receives 4,096 observed target events, exactly once, plus 1,024 fresh validation events with writes disabled. These are token-distribution observations, **not 4,096 independent lives or full sequence records**.

Main roster: 3 fixtures × 3 seeds × 2 rates × 3 arms = **54 fits / 221,184 observed target events**. On the conditional fixture add two cheap controls at both rates/seeds:

1. C with context ignored, sharing a single v across all contexts. This tests how much gain can come from learning the marginal distribution.
2. B plus a learned shared class bias beta, updated by beta←beta−eta s(n)r. Its primary u/code remain fixed as in B. This separates frequency repair from learned code geometry; it is an evaluator control with 257 extra values, not a promoted native design.

Controls add 12 fits / 49,152 observed targets. Total Stage B: **66 fits / 270,336 observed target events**, plus **67,584 write-free validation target probes**. Count every failed fit; there is no additional retry or hidden pretraining allowance.

Report expected cross-entropy under the known fixture distributions, observed validation BPS, empirical-frequency predictor, exact entropy, probability mass, code drift and cost. Optimizer/reference work is counted separately from online observations. No oracle distribution, validation count or future symbol enters candidate updates.

Select one common eta by lowest mean validation BPS over the **54 main fits**, equally weighting fixture, seed and arm; tie by smaller eta. Controls do not select it. No additional rate is tried.

Stage C admission requires qualification to pass, B to reduce expected loss toward the fixed-C numerical reference relative to A in both nonuniform fixtures on average, and C to beat both its context-ignored control and the marginal-frequency predictor on the conditional fixture in each of the three seeds at that common eta. Comparisons use tolerance 1e−8 only to avoid roundoff. This is an engineering spending gate, not statistical confirmation, an effect-size claim, or a theorem that poorer acquisition precludes all transfer. Failure stops at OUTPUT_LEARNING_NOT_READY; do not extend the dose or switch the target metric.

## 6. Stage C: conditional native integration, one small pass

This stage runs only after a later implementation/run authorization and the Stage B gate. No automatic launch follows this design delivery.

A/B/C use the same frozen address construction, native birth, observations, line order, 16 compartments and selected eta. Only the declared teaching/code changes differ. No real-valued reference weights or count table are transplanted into native memory. The code-learned state must be acquired by the native branch's own observations.

Use only already-used **development** texts from the pinned V148 corpus. Alice is not read, evaluated or called a clean confirmation set. Reuse corpus.py's byte-preserving records (drop its title, nonempty lines, split at 128 bytes) without introducing a new parser.

| Text | Train offsets / records / targets | Calibration records / targets | Development evaluation records / targets |
| --- | --- | --- | --- |
| Bryant | [0,47), 47 / 2,061 | [47,65), 18 / 1,024 | [4085,4122), 37 / 2,095 |
| Buster | [0,33), 33 / 2,107 | [33,48), 15 / 1,050 | [1282,1314), 32 / 2,090 |

The six native histories train one pass: **240 full line records / 12,504 observed byte/EOS targets / 200,064 native compartment teaching calls**. Code updates do not constitute additional observations, but their operations and memory writes are charged. There are no additional passes.

Before each observed symbol, log the branch's own distribution. Then expose the true byte and teach it, finally teaching EOS at record end. No class ID beyond the actual byte/EOS or hidden expected answer is supplied.

Score calibration and development evaluation once after training on read-only clones: **306 line queries / 18,777 target-position probes** across six histories. Teacher forcing supplies actual preceding input bytes on the clone; this is next-symbol log loss, not complete own-generation success. Primary probabilities use T=1, all 257 symbols and no calibrated temperature or seen-symbol renormalization. Temperature tuning is omitted so it cannot compensate for a mismatched objective after outcomes are known.

Also generate after the first min(8,length) bytes of the first 16 development-evaluation records in each text: **96 own continuations**, cap 64 output decisions each. Maximum generated decisions 6,144; input-prefix bytes <=768. Feed only the model's own generated bytes. Report own EOS/CAP, exact remaining continuation, length, invalid numerical values and full byte output locally. These descriptive probes must not become a grammar benchmark, rescue criterion, or word-overlap metric.

Use the pinned Unigram and Bigram counting implementations on the same train records as transparent countermodels. An exact/context-key table and frequency counter can satisfy this diagnostic; passing therefore earns no E3 arithmetic or NLP claim. Label comparative NLL and conditional-prediction evidence separately from E1 acquisition and E0 interface checks.

Native evidence is descriptive per text: there are two development texts and a deterministic birth, not dozens of independent worlds. Report A−B, B−C and candidate−unigram in BPS, with all costs. A native improvement is not required to equal the unconstrained 40-KB bigram reference. A lower BPS caused only by marginal-frequency repair is not evidence of contextual computation.

## 7. Decisions and falsifiers

| Observation | Scoped interpretation / next action |
| --- | --- |
| Fixed C cannot fit a target count distribution, even with a qualified numerical reference | Representation restriction for that family; do not blame the online learner |
| B is closer to the fixed-C optimum than A | Supports objective alignment in the output-only circuit |
| C improves only to context-ignored/bias/unigram performance | Marginal-frequency repair; no conditional-computation claim |
| C improves conditional reference prediction, but native integration does not | Native parameterization, dynamics or signal coupling remains a candidate cause; not isolated proof of credit failure |
| Native C improves unmasked conditional BPS beyond A/B and unigram on both texts | Promising bounded integration result; separately design the addition return test |
| Reference and native both fail within this menu/dose | Finite negative evidence; do not widen the search or declare local learning impossible |

The single-context witness, fixed-code optimum and locally reached behavior must appear as three separate columns. Same output dimension does not mean same parameter count, sample efficiency or compute efficiency. A codebook that shares parameters has not thereby demonstrated semantic or cross-task transfer.

## 8. Return to addition and evidence boundaries

This study earns no adoption or general-purpose-learning claim. Its output circuit must later be tested in an otherwise pinned native addition learner on complete exact own-byte/EOS answers for never-reinforced operand orbits, at matched **total** experience and compute. Retention after genuine intervening learning remains a separate requirement. No arithmetic-specific representation, carry routine, answer-length hint or new output vocabulary may be installed for that return.

The future addition experiment needs a separate design. It is not launched by a successful text diagnostic. Scalar-only reward, four-operation arithmetic, NLP scaling, Storybook and new language tasks are outside this study.

## 9. Finite resources, integrity and stopping

Proposed total ceiling: **7,200 summed process CPU-seconds**, at most one native worker and one numerical thread. Suggested admissions: Stage A <=600 s, Stage B <=1,200 s, Stage C <=3,600 s; the remaining 1,800 s covers source qualification, failed attempts, evaluation and reduction. These are prospective limits, not an execution grant. Do not infer runtime from the historical linear cost profile.

Reserve estimated whole next-work-item cost before admission. If it cannot fit, stop as INCOMPLETE_RESOURCE_LIMIT rather than shorten the roster invisibly. No automatic retry, extended dose, resumed pause or new seed. Save the failed attempt and charge it. No caffeinate, worker or monitor is started by this document.

Track unique records, target bytes/EOS, warmup targets, all repeated exposures, context/reference state size, generated decisions, passive probes, native compartment calls, dense 257×16 competition, code writes, copying, CPU/wall and peak RSS. Reference count/optimizer access is not information-matched learner training and must be labeled. Legacy global scans and cloned readouts are measured here; this design makes no production-scaling claim.

Before any future training, save the exact source/runtime/asset closure and finite roster. Qualification's extra partial native writes are capped at 64; any extra synthetic observed-target fixtures in the implementation exercise are capped at 256 and reported separately. They cannot conceal additional full toy or native histories.

Checkpoints include all native states, C/bias if applicable, observations/warmup counter, clock, cursor, RNG and accumulated cost. Read-only probes cannot change learning, code or future RNG. Deep-clone mutable code and all adaptation/homeostasis state; the historical shallow wrapper copy alone is not sufficient evidence. Save complete outputs locally; publish compact results/source by default. Reject missing/duplicate/extra histories, incompatible codebook/state restoration, nonfinite metrics and counter mismatches.

**Delivery boundary:** this folder ships a refined standalone design and static source/data checks only. It implements no new learner, performs no three-cycle exercise, starts no training, creates no separate chat and pushes nothing to GitHub. The arithmetic experiment pointer and all frozen results remain unchanged.
