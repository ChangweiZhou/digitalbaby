# STREAM-TRANSFER V1 — a generic byte learner, with paid repetition

13 October 2026. **Design only. No implementation, prototype, technical training, formal science, launch or upload has been performed for this design.**

## 1. Question and fixed objective

Can one small, unchanged byte learner acquire arithmetic from observed feedback, reuse addition experience when learning multiplication, and retain what it learned, with better unseen-case accuracy per training exposure and production compute?

This is design B, separate from [RAW-RULE V2, design A](../arithmetic_feedback_rule_discovery_design_v2_20261013/EXPERIMENT.md). A addresses weak initial acquisition under the existing four-operation budget. B tests a different, explicitly paid experience regime: counting, addition, then multiplication, with all-byte prediction and repeated presentations. Its result cannot replace A's four-operation endpoint. Subtraction, division, mixed expressions and NLP are not tested by B; successful B would establish bounded E3 evidence in two tasks and a possible prerequisite for the original four-operation goal.

“No hand-built structure” means **no task-specific arithmetic structure in the learner**, rather than no inductive bias of any kind. Byte embeddings, recurrence, a bounded gradient horizon, framing and an optimizer are declared generic biases. They remain identical across tasks. No payload parser, ASCII digit values, operand slots, operation IDs, arithmetic modules, supplied exchange/inverse rules, numeric targets, numeric renderer, answer candidates, expected answer length, or arithmetic output mask is permitted in a new learner.

The core is an engineered backpropagation reference. It is not unchanged Full151, a local biological learning rule, or an adopted MiniFly core. No unspecified “local rule” arm is added. A local replacement requires a concrete update equation and the same stream, IO, controls and cost accounting in a later design.

## 2. Premise audit

The pasted proposal's claim that every existing core is an exact-context lookup is incorrect. RAW-RULE V1 already has learned byte embeddings, a GRU and a plastic linear head. In its final technical fixture, R1-FULL produced correct complete answers on 41/199 taught prompts; 137 of its 158 failures began with a wrong first byte. The problem was present before the later cohort. Frozen-encoder controls also struggled. This motivates coherent joint credit and categorical loss; recurrence alone is not a new cure. See the [saved-state review](../arithmetic_feedback_rule_discovery_dev_v1_20261013/ops/read_only_review_20261013/FINDINGS.md) and [interpretation correction](../arithmetic_feedback_rule_discovery_dev_v1_20261013/ops/read_only_review_20261013/INTERPRETATION_CORRECTION.md).

Exact CONTENT hashes lack a designed similarity path; that does not establish that all memories or all input routes cannot generalize. Earlier REF/FE0/role results have limited relation-transfer evidence, with different tasks and readouts. Likewise, improved taught recall is not proof of an arithmetic rule.

The original [grokking study](https://arxiv.org/abs/2201.02177) used small transformers and abstract-symbol operation tables. Some generalization appeared after extensive optimization, and smaller datasets could need more optimization. [RNN experiments from their authors](https://github.com/keith-murray/grokking-recurrent-networks) concern a modular relation, not this decimal byte-output curriculum. [Omnigrok](https://arxiv.org/abs/2210.01117) also studies conditions under which grokking appears or disappears. These are reasons to test repeated optimization and regularization, not evidence that a 64-state GRU must learn decimal arithmetic or addition-to-multiplication transfer. Neither decay nor the larger 0–9 range guarantees a simpler learned solution.

## 3. Frozen learner specification

Use the generic G-CE architecture specified in A, without its arithmetic-independent lag alternative:

- Input IDs 0–255 are arbitrary payload bytes; 256 is observed EOS, 257 public BEGIN and 258 public REPLY.
- Learned identity embedding E has shape 259×16. Independent random sign initialization divided by sqrt(16); no ordered or numeric embedding.
- One GRU has 64 state coordinates, three 64×16 input matrices, three 64×64 recurrent matrices and three 64-element biases. Use A's initialization and exact GRU transition equations, prediction depth one. All four configurations have identical parameter birth within a world; initialization RNG and data-order RNG are independent. No dropout or stochastic generation is used.
- Readout uses u = [h,1]/||[h,1]|| and logits uV, V of shape 65×257, initially zero. All 256 payload outputs plus EOS compete without a vocabulary/length mask. Argmax ties use the smallest output ID.
- BEGIN resets transient h/context/pending gradients for every record. It never resets parameters, moments or learning counters. REPLY is generic transport framing, not a task identifier.
- Parameters remain fixed during an observed record. Before each target arrives, compute the prediction and permitted cross-entropy against that actual target. At actual permitted EOS, take one joint update of E, GRU and V from mean permitted loss. Generated bytes and generated EOS never train.
- Per-target gradients have at most 32 preceding transitions. Accumulate bounded parameter gradients online while retaining a ring of at most 32 transition caches; older cotangents are detached. This is not an unbounded frame tape. For short task records the entire preceding record is available; long payload records retain all prediction targets but truncate each target's credit horizon.
- Bias-corrected RMS second moment: beta .99, rate .001, epsilon 1e-8, global gradient norm clip 1. No first moment, RLS covariance, head-only per-byte updates, task-dependent optimizer, learning-rate schedule, replay cache, or task-boundary reset.
- Decoupled decay, where enabled, follows the RMS update on every parameter array, including biases: theta_next = (1 − rate×lambda) theta − RMS_step. lambda is either 0 or .01. No permitted targets means no optimizer, moment, counter or decay update. Idle simulated time does not decay weights.
- Model parameters: 36,401. Model plus one moment array per parameter in float64: 582,416 bytes. Pending gradients, transition caches, transient h, scratch, counters and runtime overhead are additional and must be measured.

Per event, dense work is O(H² + HD + HC), with H=64, D=16, C=257; backward credit over at most 32 events is additional. This is a small feasibility reference, not a demonstrated architecture for 10^10 neurons. No history-sized exact-key store is hidden in the learner.

Every-byte prediction means **every observed payload byte and actual EOS**. BEGIN/REPLY are transport inputs, not output targets. Those two exempt controls must not be described as trained payload bytes. All ordinary payload values, including NUL, newline, spaces and `=`, retain their normal byte identity.

Generation starts after public REPLY, emits the core's own bytes, and ends only on its own EOS. Permit an empty answer; cap at 16 payload bytes followed by one EOS decision. A limit hit is a truncation error, not a model-selected stop. Generation runs on transient state and cannot modify any persistent parameter, moment, counter, pending gradient or cache used for learning. Probe feedback is absent.

## 4. Four configurations; the same core across all tasks

| ID | Permitted prediction targets | Decay lambda | Role |
| --- | --- | ---: | --- |
| ALL-0 | Every observed payload byte and actual EOS | 0 | Selectable |
| ALL-D | Every observed payload byte and actual EOS | .01 | Selectable |
| ANSWER-0 | Observed response payload and actual EOS only | 0 | Matched diagnostic |
| ANSWER-D | Observed response payload and actual EOS only | .01 | Matched diagnostic |

Each configuration uses the same architecture and fixed settings in counting, addition, multiplication and the later binary cohort. ANSWER is a generic public-REPLY loss mask, not an arithmetic parser. Comparison of ALL and ANSWER is a learning-recipe comparison: loss averaging, gradient amounts and clipping can differ. Record component losses, gradient norms, clipping and actual work; do not attribute an effect solely to “more signal.”

The regularization contrast is fixed before data. There is no decay/width/rate sweep, adaptive capacity, task-aware routing or early stopping on held answers. Limited capacity is the shared architecture, not a separately identified mechanism. Repetition is an exposure-budget axis common to every configuration. **Offline replay is not implemented:** the environment repeatedly presents real records and every presentation is charged. This avoids confusing repeated external observations, unique information and internal replay.

## 5. Data stream and exposure ledger

The evaluator owns numeric rendering and labels; the learner receives only literal framed bytes. The learner module cannot import the evaluator, task sampler, split map, numeric functions or test labels. Primary feedback is clean to test basic rule learning. Noise/missingness are not the principal direction or an additional gate here.

### Counting prefix

Fifty distinct records have starts k=0,…,49. The prompt is the ordinary decimal string `k,k+1,k+2,k+3`, and the response is `,k+4,k+5` followed by observed EOS. This notation describes evaluator generation only; it is never implemented inside the learner. Present 128 records: two independently shuffled complete passes through these 50 records, followed by the first 28 records of a third shuffled pass. All configurations and causal branches share this actual prefix. Its 128 observations, target bytes, updates, elapsed production CPU and persistent state are charged.

Counting is arithmetic-relevant information in data, not free or natural-text pretraining. There is no no-counting ablation in this finite experiment, so counting's isolated contribution is not identified. The primary claim is explicitly conditional on this counting stream. Counting records contain no addition or multiplication operator and no complete held arithmetic prompt; seeing a result numeral alone is not seeing the held relation.

### Addition and multiplication

Operands range 0–9. Each operation has 55 unordered operand orbits. Hold out 20 orbits sampled without replacement from the 36 orbits whose operands are both 2–9. The other 35 are training orbits. Addition and multiplication have separate, prospectively generated split maps; both orientations of each held orbit are excluded from every training pass. Diagonals have one scored orientation, off-diagonals two. Report the actual number of held and taught byte prompts in each world.

An addition prompt is literal `a+b=` and a multiplication prompt `a*b=`; answers are complete ordinary decimal strings followed by observed EOS. They are examples, not architecture. Their public BEGIN/REPLY framing is the same as for counting and the binary cohort. No answer length, candidates or parsing metadata enter the learner.

For each task, one pass has **70 external records**: each of the 35 training orbits appears twice, once in each orientation. A diagonal is deliberately presented twice as the identical prompt. This is charged duplication, not two unique facts. Shuffle the 70-record order independently per pass and use the identical ordering across configurations and matched branches.

Fixed checkpoint passes are 0, 1, 4, 16, 64 and 256. Nonzero task exposure budgets are therefore 70, 280, 1,120, 4,480 and 17,920 records. Every run finishes the fixed 256-pass budget; no “until it groks” extension is allowed. Track both distinct feedback relations/raw records and total repeated exposures, bytes, optimizer updates and CPU. A later improvement at 256 passes is not evidence of learning from 70 total examples.

The input range is larger than A, but a 64-state neural network could still memorize 35 orbits. There is no assumed capacity theorem or guarantee that 0–9 is easier. A comparison with the old range cannot be causal because the task, stream and exposure budgets differ.

### Generic intervening cohort and delay

At each nonzero addition checkpoint fork its three histories; at each nonzero multiplication checkpoint fork its five histories. Each fork sees the same 32-record generic binary cohort: eight distinct 6-byte keys, each presented four times, with world-fixed unrelated 1–3-byte responses. Keys may use all 256 byte values and must not equal any arithmetic/counting prompt. Core settings stay unchanged. Then advance simulated time by 86,400 seconds with no learning/update and probe the original held questions read-only.

These cohort forks provide retained sample curves without mutating the ongoing arithmetic root. They are paid training, not independent lives. Probe the cohort before and after learning to verify actual acquisition. At multiplication checkpoints, also score retained old addition; this tests the effect of intervening multiplication, not just the small binary cohort. Idle time alone cannot demonstrate nontrivial decay resistance when the model has no idle decay; new-learning interference is the substantive retention challenge.

## 6. Causal histories and shared-prefix accounting

Start from one learner birth per configuration/world. All forks deep-copy mutable state without sharing it. Counting is executed once and shared as an identical checkpoint. From there run three addition histories:

| Addition history | Counting | Addition updates | Addition byte stream |
| --- | --- | --- | --- |
| W | Enabled | Enabled | Correct observed responses |
| N_ADD | Enabled | Entire record disabled | Same correct bytes/timing |
| SH_ADD | Enabled | Enabled | Correct prompts; shuffled observed responses |

SH_ADD shuffles the **35 response strings among orbits within response-length groups on each pass**, then uses that orbit's assigned string in both orientations. Fresh permutations per pass preserve the pass's response-byte multiset and lengths; they may contain unavoidable fixed/equal-label matches, which are reported. The evaluator cannot reject permutations after observing learner behavior. Shuffling is a diagnostic feedback intervention, not a proposed real-world data stream.

At the end of addition, branch W into W, N_MUL and SH_MUL. Continue N_ADD and SH_ADD through correct multiplication. Five multiplication histories result:

| ID | Prior addition | Multiplication updates/feedback | Question answered |
| --- | --- | --- | --- |
| W | Correct learning | Enabled, correct | Main whole learner |
| N_ADD | No addition learning | Enabled, correct | Reuse of prior addition versus matched exposure without learning |
| SH_ADD | Shuffled addition learning | Enabled, correct | Useful relation content versus generic optimization history |
| N_MUL | Correct learning | Entire record disabled, same correct sensory bytes | Causal use of multiplication feedback |
| SH_MUL | Correct learning | Enabled, same length-group shuffle rule | Correct multiplication relation versus marginal output statistics |

The no-update histories disable **all targets, optimizer moments/counters, pending state changes and decay** in their disabled task. Merely disabling answer-head writes while training the encoder on those answers is invalid. They can read identical actual bytes transiently; normal BEGIN reset prevents an answer from becoming persistent transient memory across records. No disabled records enter a replay/cache or a later auxiliary update. All histories learn the generic cohort; all probes have no updates.

W versus N_ADD measures the consequence of the entire learned addition state, including optimizer state. SH_ADD helps distinguish semantic experience from generic optimization warmup; it does not mathematically remove every conditioning difference. Do not claim a uniquely isolated neural mechanism of transfer from these contrasts.

N_ADD is the multiplication-from-scratch **after the same counting prefix** reference. After its addition phase, all output/learning-affecting persistent state must exactly equal the post-counting checkpoint; record-start transient state must also equal a direct multiplication start. Monitoring-only elapsed-time/byte counters may differ, but must not enter the model or update policy. Qualification proves this parity. An actual experimental N_ADD schedule still pays its sensory-only addition exposure. Separately report the direct-schedule accounting obtained by omitting those measured, state-inert addition reads, clearly labelled an accounting counterfactual rather than an additional executed science trajectory. Do not call the reference “from birth” or grant W free counting/addition.

State shared across tasks includes parameters, RMS moments, optimizer age and generic counters. No task-boundary reset, optimizer replacement, task marker, new head or separate store is used by the main learner. Public `+`/`*` are ordinary input bytes; task phase is evaluator-only metadata for controls and reports.

## 7. Endpoints: own output, samples, compute and retained transfer

The operative learner's greedy, complete payload plus own EOS must exactly match the unseen answer. First-byte and teacher-forced token accuracy, loss, representation probes and taught recall are diagnostics; none substitutes for exact unseen output. Complete teacher-forced exact and own exact can agree by construction under deterministic greedy generation, so their equality is not independent causal evidence about compounding.

For each world/configuration:

1. **J_ADD:** equal mean of retained held-addition exact accuracy after the five addition exposure budgets and their paid binary-cohort/delay forks.
2. **J_MUL:** equal mean of retained held-multiplication exact accuracy after the five multiplication exposure budgets and their paid cohort/delay forks.
3. **J:** (J_ADD + J_MUL)/2. J alone is not an all-task competence gate.
4. **R_ADD:** final retained held-addition accuracy after all multiplication learning, the final binary cohort and delay. This guards useful old-rule survival across the second task.
5. **G:** the mean of two normalized retained-accuracy staircase areas over paired common production CPU, for ADD and MUL respectively. Use real birth/read/counting/training cost, paid cohort costs and actual initial scores; do not interpolate or extrapolate. The MUL timeline includes the entire counting and addition stage. Each task's curve begins at its observed birth score and changes only at its five registered retained task checkpoints; the birth score carries through the paid prefix until the first retained checkpoint. Additional read-only multiplication probes at counting end and addition checkpoints diagnose zero-shot transfer but do not enter G. Interleaved probes are excluded from training CPU and reported separately as evaluation cost. Use cumulative maxima for noisy measured costs. Pairwise horizon is the shorter terminal production cost; stale scores may carry forward only to that horizon. A point's cost contains its own task prefix and 32-record cohort, not earlier measurement forks that are absent from that learner's history.
6. Report the separate direct-scratch cost curve, marginal multiplication exposure/CPU curve, total-lifecycle curve and final accuracy. **A marginal multiplication benefit may coexist with worse total efficiency.** It is then a transfer finding, not evidence that the addition warmup paid for itself on multiplication. The separate recipe-level efficiency verdict compares J and G with matched ANSWER on both tasks, with all prefix cost paid; it does not imply superiority over every alternative schedule.

The sample primary uses repeated external records, not only unique orbits. Counting and cohort exposures are included in lifecycle ledgers; their fixed/shared doses are stated next to the task-specific sample curves. No cherry-picked “samples to 80%” threshold is used. Count unseen exact success at every locked budget and final zero-error grid coverage descriptively. Never call a few-point mean effect reliable arithmetic competence.

Recitation controls: a raw taught-key table and a commutative canonicalized taught-key table have no taught entry on the held orbits. Give them explicit abstention and separately a feedback-trained frequency fallback; absence of an entry is not a universal random-chance number. Also implement A's four fixed feedback-only whole-answer modes: global, prompt length, last two raw prompt bytes, and unordered prompt-byte multiset, with declared fallbacks and ties. Fit only on executed arithmetic feedback available to that branch at that checkpoint; future feedback is excluded, old task counts are protected against the unrelated cohort, and a completely empty record set abstains. Compute the conservative per-world/per-operation held-accuracy envelope B separately at each checkpoint, then use the same locked curve averaging as for the learner. B uses evaluator labels solely for a conservative comparison, not an operational oracle or model initialization. Numeric relations/analytical answers are never candidate features.

## 8. Finite DEV and fresh confirmation

**DEV:** two fresh paired worlds, all four configurations: eight complete jobs. DEV estimates feasibility and selects at most one of ALL-0/ALL-D. It is not statistical confirmation. No loss/passes/rate/data changes follow its held scores.

Eligibility uses strictly positive world-mean directions, without a percentage cutoff: final taught addition W>N_ADD and final taught multiplication W>N_MUL, immediately at their respective teaching ends; both J_ADD and J_MUL above their no-update and shuffled histories and B; J_MUL W above N_ADD and SH_ADD; R_ADD W above N_ADD and SH_ADD; J and pairwise G above the matched ANSWER recipe; mean cohort post>pre across all W checkpoint forks. If neither qualifies, report NO_ADVANCE and package the complete negative DEV. Selection ranks J over matched ANSWER, then lower total production CPU, then ALL-0. Store the entire selection calculation and immutable selection lock.

**Conditional confirmation:** 16 fresh paired worlds. Three configurations: selected ALL, its matched ANSWER diagnostic, and the other ALL configuration. Thus 48 complete jobs if advanced. Same splits, counts, budget settings, controls and streams. DEV data are excluded. Only the selected recipe is promotable.

The selected recipe has a Holm .05 family of **12 one-sided paired world-mean tests**:

- J_ADD W>N_ADD and W>SH_ADD: two.
- J_MUL W>N_MUL and W>SH_MUL: two.
- J_ADD and J_MUL W>B: two.
- J_MUL W>N_ADD and W>SH_ADD: two (cross-task transfer).
- R_ADD W>N_ADD and W>SH_ADD: two (retained addition after multiplication).
- J and pairwise G selected ALL>matched ANSWER: two (sample and total-production-compute benefit).

Use paired Student-t tests/intervals for nonconstant differences, explicitly approximate with n=16. For constant differences, use A's conservative bounded-mean branch: positive p=min(1,exp(−n×mean²/2)), nonpositive p=1; bounded 95% half-width sqrt(2 log(40)/n), clipped to [−1,1]. Do not replace a mean test by an exact sign test. Descriptive marginal/direct-scratch cost curves are not extra hidden promotion tests.

All 12 positive adjusted tests yield **CONFIRMED_STREAM_TRANSFER_EFFICIENCY**: bounded retained E3 and a recipe-level efficiency improvement over the matched answer-only learner, conditional on the specified data. Otherwise **NO_CONFIRMED_STREAM_TRANSFER_EFFICIENCY**. Also report partial effects explicitly: acquisition-only, within-task E3 without cross-task transfer, transfer without total compute benefit, or retained gain without sufficiently precise evidence. A negative adoption verdict is valid; no result edits the current core.

The other ALL arm informs the prospective regularization contrast descriptively. It is not a second winner chosen on confirmation. There is no historical RAW/legacy comparison in B under supposedly equal information: they did not receive this stream/budget. Cross-design comparisons remain descriptive.

There is currently no qualified generic raw-byte learner known to solve this exact task under this budget. The evaluator's arithmetic renderer proves expected labels, not learner feasibility. A taught-item lookup serves as a report-pipeline E1 positive witness only. Qualification failure or a negative result cannot prove all generic or local learners impossible.

## 9. Count audit and resource boundary

Per configuration/world complete job:

- One shared counting prefix: 128 record applications.
- Three addition histories: 3×70×256 = 53,760.
- Five multiplication histories: 5×70×256 = 89,600.
- Fifteen addition cohort forks plus twenty-five multiplication cohort forks: 40×32 = 1,280.
- **Total: 144,768 executed external record applications.** No training probes.
- Five final task histories share counting/addition prefixes; forty cohort forks are correlated continuations. They are not 45 independent lives, nor eight independently sampled tasks. Statistical unit is a fresh world, not a record, fork or configuration.

DEV: 8 jobs, 1,158,144 record applications. Conditional confirmation: 48 jobs, 6,948,864. Maximum: **56 jobs, 8,107,008 record applications, 18 independent world identities**. This deliberately contains much more repeated optimization than A; it is not a “minutes” promise.

Measure actual per-history bytes, targets, RMS updates, inactive reads, backprop transitions, forward generations, physical CPU, peak memory and pending/cached state. Separate production, fork/copy, probe and audit costs. Shared prefixes are charged once as executed work and fully included in each represented learner's lifecycle cost; do not pretend shared experiment optimization reduces the learned experience per learner. The per-job 144,768 is executed experiment work, not each history's individual experience. Each final W history has 128+17,920+17,920 = 35,968 arithmetic/counting records before its cohort; its final retained probe includes another 32, i.e. 36,000. Earlier measurement forks are not observations in that W history.

Before admission, measure one complete job at the full fixed budget and two-worker parity on technical namespaces. For each stage, 3×measured projected execution plus separately measured analysis/audit/packaging must fit eight hours; hard session boundary ten hours, safe pause at whole-job boundaries. Two detached workers and single-thread BLAS per worker are a plan, not deployed processes. Caffeinate follows the supervisor, with a read-only monitor and committed progress. No automatic restarts, retries, resumed sessions or extra passes. If cost fails admission, report the design cannot fit; revise prospectively before any DEV, rather than truncate after seeing results.

Namespace labels are proposed, unadmitted strings: STREAM_TRANSFER_V1_20261013/TECH/*, /DEV/0001–0002, /CONFIRM/0001–0016. Seed derivation and a complete history census must be specified and locked before implementation qualification; these are not existing results.

## 10. Qualification obligations and interpretation of failures

Before training science, the future three-cycle exercise must implement/audit/test/revise without changing the goal:

1. Joint categorical derivatives, normalized readout, per-target truncated gradients, loss masks/averaging, actual EOS updates, decay equation and no-update invariance. Include an all-byte long record with gradient horizon overflow and all 256 payload identities.
2. Generic IO, own EOS/truncation, pure generation, binary cohort, persistent-state independence, no task metadata imports, observed-target provenance and unchanged core across every stream. Detect deliberately tampered prompt/answer/decay permissions with an independent write ledger.
3. Full-budget behavior and cost, complete held-orbit exposure audit including counting/shuffles, direct-scratch parity, shortcut implementation, paired independent reducer, exact counts and fail-closed detached dispatcher/package tests. Separate technical fixtures from prospective DEV/confirmation seeds.

Behavioral results are not promoted from small synthetic tests. If taught recall stays poor, inspect allowed update magnitude, representation/readout fitting and credit timing within this fixed package; do not attribute an E3 failure to a discovered rule. If taught recall is strong but held scores fail, report the declared recipe's generalization limit. If only repetition helps, report the total repetitions and compute needed. If transfer is marginal but expensive, report the tradeoff. No single negative excludes recurrence, decay, replay, local plasticity or the full mechanism catalogue.

## 11. Inventory mapping and route back to the actual milestone

Use the [complete inventory review](../arithmetic_feedback_rule_discovery_dev_v1_20261013/ops/read_only_review_20261013/inventory/FULL_INVENTORY_REVIEW.md) and its machine-readable row record: 108 historical mechanisms, 136 catalogue rows, 254 version rows, the overlapping older 71 subset and first 16 roster. Do not substitute this experiment's four configurations for that complete list.

Concrete lineage: M001/M007 generic recurrence, M031 optimizer/regularization, M033 error-based classification, M050 current-observation repetition. This is a new engineered recipe, not an exact test of every historical local recurrence/replay variant. The historical mixed/negative replay findings remain scoped to their actual implementations. Decay is an optimizer regularizer here, not M044's validated active-forgetting mechanism. M022/M106 coordinate management, M024 nonlinear adapters, M059 ordered access, Hebbian matrices and protected timescales are neither added nor excluded by B.

No NLP/scaling move follows automatically. The declared program after B is conditional:

- B negative: choose among the already specified A acquisition comparisons and this full inventory using the recorded failure; do not call the two-task benchmark the new goal.
- B positive: freeze the same byte learner, loss/update policy and resource ledger, then design subtraction and exact integer division data with complete exchange/inverse family withholding across all four operations. That full four-operation confirmation needs fresh worlds whose split is fixed before any of the four operations is taught; B's independently split addition/multiplication checkpoints cannot retrospectively satisfy that broader exposure audit. No new operator branch or supplied inverse equation enters the learner.
- Only after retained own-output E3 on all four single operations, introduce a prospectively split mixed-expression stream with held compositions and separately measured precedence/length extrapolation. Parentheses/operators are data; no calculator/parser is installed in the core. If that needs a changed generic core, declare a new version and compare it prospectively.

These later data extensions need their own finite designs; they are not authorized runs or untested capability claims. A and B both serve the unchanged goal: acquire arithmetic laws from feedback and improve retained unseen answers per sample and compute.
