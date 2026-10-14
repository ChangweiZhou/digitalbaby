# RAW-RULE V2 — learn useful byte distinctions and arithmetic from feedback

13 October 2026. **Revised design only. No V2 learner, technical cycle, DEV, confirmation or background run has been executed.** The completed V1 preparation and adopted core remain unchanged.

## 1. Fixed goal and evidence level

Improve retained **exact own-output generalization to never-taught four-operation arithmetic per external sample and production computation**, with arithmetic relationships learned from arriving feedback. No arithmetic law is supplied to the learner.

This is an engineered, general-purpose byte learning comparison, not unchanged Full151 or a validated biological local rule. Its intended behavioral evidence is bounded E3 under delay and intervening learning. Acquisition of taught questions is an E1 diagnostic; teacher-forced prediction is a localization measurement and never the scored solution.

Keep the existing interleaved single-binary-expression task, seed operands 0–7, signed subtraction and exact integer division. Interleaving operations is **not** a test of precedence, parentheses or mixed expressions. The next capability after a genuine gain remains mixed expressions. Noise optimization, NLP, scaling and pretraining are not introduced here.

## 2. What the saved V1 evidence actually explains

The read-only review used the final reserved technical world 202610130103, saved checkpoints and frozen receipts. It ran inference and explicitly labeled post-hoc algebra, not new learner trajectories. It is not population inference or a causal repair experiment.

| Finding | Evidence | What it rules out or leaves open |
| --- | --- | --- |
| Wrong answer content already appears at the first decision | R1-FULL complete own answers 41/199; first byte 62/199; conditional payload-byte accuracy 27.3% | First-byte errors account for 137 of 158 failed complete answers (86.7%), before any generated prefix can compound; a generic termination fix cannot repair this |
| Full teacher-forced exact matches own exact | Teacher-forced complete byte/EOS sequences also 41/199; R4 gives 40/199 in both conditions | Expected parity for deterministic greedy generation, not an independent causal test against compounding. Use per-position errors and the first-error decomposition |
| EOS can be predicted in correct answer contexts | Conditional EOS 100% for all six new configurations | This is not evidence that EOS is perfect under every wrong prefix or earlier budget |
| Frozen head fits its declared mathematical objective | Frozen FULL ridge normal-equation residuals about 8e-15 and 1e-15; own recall 48/199 and 36/199 | No evidence of missing target writes or a faulty FULL RLS formula; the recipe itself fits poorly |
| Common suffix/control events dominate the final summary | R1 mean feature cosine approximately 0.542 before '=', 0.824 after '=', 0.929 after REPLY | Generic recency/context compression is measured; information destruction is not proved |
| Weak cue directions and joint output contexts are difficult to fit | First-byte-only oracle LS 43.7% at ridge 0.1, 71.9% without ridge; complete joint conditional sequences 22.6% to 34.7% | A ridge change alone does not restore taught recall; oracle fits are not a ceiling on all linear classifiers |
| Learned coordinates and accumulated RLS history disagree | Learned heads' current-basis normal-equation residuals about 0.24–0.26; frozen identities hold | Feature drift is relevant but cannot explain the frozen failures |
| The old easy byte fixture was insufficient behavioral evidence | 512 arithmetic records cover 199 distinct taught prompts at the terminal budget; two-key fixture used many repeats per key | Interface acquisition is not reliable high-load acquisition or discovery of arithmetic laws |

All final cue matrices have mathematical rank 65. Low effective rank and high cosine are **not** exact key collisions, a theorem that 65 features cannot classify 199 prompts, or a proof that a GRU cannot learn this task. The least-squares counterfactual uses evaluator answers after the run and does not initialize, train, select or feed the new learner.

The defensible diagnosis is a **conditional-content learning failure of the current representation/readout recipe**. Frozen features are poorly suited to its regularized, shared linear fit; the small learned encoder has not remedied that, and moving coordinates add another inconsistency. A unique causal contribution for each limitation remains to be measured.

Sources: [read-only findings](../arithmetic_feedback_rule_discovery_dev_v1_20261013/ops/read_only_review_20261013/FINDINGS.md), [interpretation correction](../arithmetic_feedback_rule_discovery_dev_v1_20261013/ops/read_only_review_20261013/INTERPRETATION_CORRECTION.md), [checkpoint measurements](../arithmetic_feedback_rule_discovery_dev_v1_20261013/ops/read_only_review_20261013/CHECKPOINT_REVIEW.json), [oracle algebra](../arithmetic_feedback_rule_discovery_dev_v1_20261013/ops/read_only_review_20261013/ANALYTIC_HEAD_DIAGNOSTIC.json), and immutable V1 core.py/receipts.

## 3. Full inventory review and the operations chosen

This design reviewed every row in the **108 historical mechanisms, 136 catalogue rows, 254 program versions, 71 older deduplicated subset and initial 16-operation roster**. These overlap and cannot be added as a mechanism total. The original status, evidence limitations and version parameters are preserved in the [full row-by-row review](../arithmetic_feedback_rule_discovery_dev_v1_20261013/ops/read_only_review_20261013/inventory/FULL_INVENTORY_REVIEW.md) and [JSON](../arithmetic_feedback_rule_discovery_dev_v1_20261013/ops/read_only_review_20261013/inventory/FULL_INVENTORY_REVIEW.json). The old version register ends at U034; later results are a separately named evidence supplement.

| Inventory lineage | Concrete V2 operation | Why it is relevant now; claim limit |
| --- | --- | --- |
| M031/M033: optimizer and error-based classification; M024: learned downstream interface | Encoder and head learn the same complete-record objective, together, at actual EOS; squared versus categorical loss is explicit | Tests coherent feedback learning. Hidden backprop is disclosed engineering; the historical locality rejection is preserved |
| M059: chronological lag96 byte access; M001/M024: nonlinear cell/interface | Last 32 literal byte/control identities in order feed a learned 64-unit nonlinear reader | Gives early bytes a direct route that later common tokens cannot overwrite within the window. No digit values, operand roles or operator semantics |
| M001/M007: generic recurrence | Same 64-wide depth-1 GRU with the new joint learning recipe | Same-reader learning comparison; V1 is not a scientific exclusion of historical ELM/private-state or sparse-GRU versions |
| M022/M106: fixed/live coordinates, stable reference/transport | Conditional follow-up only | Frozen acquisition also fails; coordinate protection alone is not a sufficient next remedy |
| M050: reservoir/recent/residual replay; M027: fast Hebbian matrices | Deferred here, with exact old variants retained | Already tested versions have mixed/negative results. A changed raw-reencoding version would need distinct evidence and charged state/updates |
| P/T/S graphs, release/protection, extra banks, slower commitment | Preserve exact previous verdicts; no new default arms | More retention machinery does not directly address the measured pre-interference content failure |

Direct implementation references, including V23 adapter_retained/recycled, V25 FF/FL/LL and V35 lag encoding, are in [SOURCE_NOTES.md](../arithmetic_feedback_rule_discovery_dev_v1_20261013/ops/read_only_review_20261013/inventory/SOURCE_NOTES.md). Neither a historical positive nor the V1 technical failure is inherited as efficacy evidence for V2. No whole family is newly crossed out.

## 4. Fixed eight-arm roster

| ID | Reader | Learning recipe | Role |
| --- | --- | --- | --- |
| RAW-RLS | Exact V1 R1-FULL | Published online head RLS and frame-end encoder squared gradient | Failure/architecture reference; never selectable |
| G-SQ | Same depth-1 GRU | Joint squared loss | Learning-recipe diagnostic |
| G-CE | Same depth-1 GRU | Joint categorical cross-entropy | Selectable |
| L-SQ | Ordered byte window + nonlinear reader | Joint squared loss | Objective/reader diagnostic |
| L-CE | Ordered byte window + nonlinear reader | Joint categorical cross-entropy | Selectable |
| G-FROZEN-CE | Identical G birth, embedding and encoder frozen | CE head only | Matched learned-representation control |
| L-FROZEN-CE | Identical L birth, embedding and reader frozen | CE head only | Matched learned-representation control |
| LEGACY-B | Exact published LINEAR-FIXED-FULL | Unmodified published learning | Strong taught-memory/performance reference |

G-SQ/G-CE and L-SQ/L-CE use the same respective birth arrays, head initialization, optimizer, timing, bytes and budgets. Frozen controls freeze the embedding too. Independently owned mutable arrays are mandatory; birth parity does not imply shared writable state.

G versus L changes a complete reader architecture, including parameter and transient-state costs. It is **not an equal-parameter isolated attention or topology claim**. G has 36,401 model parameters; L has 53,681. At float64, model arrays plus one RMS moment per trainable entry require 582,416 bytes for learned G and 858,896 for learned L; pending gradients, tapes/context, buffers and metadata are additional. Frozen-CE model/moment arrays require 424,848 and 563,088 bytes respectively. Adding the L reader must earn its value through the measured common-compute comparison. Do not allocate unused padding state to pretend that the readers cost the same.

RAW-RLS changes head optimization, timing and covariance relative to joint arms, so their difference identifies a learning recipe, not only synchronization. SQ/CE share optimizer settings but differ in gradient scale; record gradient and clipping statistics rather than claiming perfect loss-scale isolation.

LEGACY-B contains disclosed numeric features and a hash bank. It is a historical performance anchor, **not** a no-prior matched control. These components never enter new cores. Exact source snapshots are copied, not reimplemented under the same name.

## 5. Readers and lawful raw-byte information

Payload 0–255 is arbitrary identity. EOS=256, BEGIN=257 and REPLY=258 are public interface controls, not payload delimiters. E is 259 by 16, initialized to seeded random signs divided by sqrt(16). No arithmetic-specific byte receives special treatment.

### G reader

Use the exact V1 depth-1 64-wide GRU transition and initialization. No extra zero-input processing transitions. Prediction feature is u=[h,1]/norm([h,1]), with 65 entries. Preserve the 32-event gradient horizon and bounded recurrent summary. No full covariance is kept in joint arms.

### L reader

Maintain a circular buffer of the last **32 actual byte/control identities since public BEGIN**, including REPLY and observed or generated earlier answer bytes. Right-align occupied slots in chronological order and use leading absent slots when fewer than 32 events are present. Pad absent slots with a zero vector; payload NUL is E[0] and is not padding. Every occupied slot uses the same E lookup. The slots express generic chronological lag, never operand positions, operation slots, answer phase, sign, place value or expected length.

At prediction, concatenate the 32 embeddings oldest-to-newest into c of length 512:

    a = tanh(A c + b)
    u = concatenate(a,1) / norm(concatenate(a,1))
    logits = V.T u

A is 64 by 512, b has 64 entries, and V is 65 by 257. Initialize each lag-block of A with zero-mean Gaussian standard deviation 1/sqrt(16), matching the per-byte input-block convention of G; b and V start at zero. This is a fixed generic initialization, not an outcome-tuned arithmetic basis. Report saturation and gradient norms in qualification.

There is no GRU in L. This is a bounded position-preserving nonlinear reader, not a stable arithmetic decomposition. Generic nonlinear interactions between raw symbols must be learned from feedback. The buffer stores a current context, not a persistent cue/answer dictionary. Embeddings are recomputed from token IDs under current parameters; it does not cache stale learned feature vectors.

The context policy is explicit: L loses events older than 32; G retains a compressed summary but backpropagates only through 32. Both accept arbitrary-length streams and use the identical policy at training and inference. Longer contexts are neither silently truncated into a semantic parser nor represented as unlimited memory. This task fits within both limits; long-stream interface fixtures verify the disclosed difference without claiming arithmetic extrapolation.

Generation reads a temporary working state, appends its own byte choices as context and restores transient state on return. All learned state, moments, counters and pending gradients remain unchanged. It emits its own EOS, with the unchanged 16-payload-byte cap and a final EOS decision. A cap failure is TRUNCATED, not an EOS.

## 6. Joint feedback learning and its controls

During a framed observed record all parameters remain fixed. Before each actual answer byte/EOS arrives, compute the conditional prediction from the actual observed prefix. After its arrival accumulate that event's permitted loss. Prompt observations advance state/context but carry no optimization target in this assay.

At the **actual permitted EOS**, take one simultaneous step for V, E and reader parameters:

    SQ: L = mean_t 0.5 * norm(onehot(target_t) - logits_t)^2
    CE: L = mean_t -log_softmax(logits_t)[target_t]

For CE the head derivative is probabilities minus onehot; for SQ it is logits minus onehot. Backpropagate the identical chosen loss through the operative reader. Do not fit V between answer events and then backpropagate using a different V. Frozen CE arms update only V.

Use RMS second moments, beta 0.99, bias correction, learning rate 0.001, epsilon 1e-8 and global gradient-norm clipping at 1, applied once across trainable parameters. This carries the prior engineering scale into a new coherent objective. It is not asserted to be optimal. Save per-component gradient/update norms and clipping fractions. Squared and CE losses receive the same mean-over-permitted-target convention; no class/EOS weights or vocabulary temperature are added.

For long records, the G derivative truncates the recurrent tape to 32 events with older state detached and older loss cotangents discarded; its loss divisor still counts all permitted targets. L retains bounded direct gradient accumulation from each observed decision's current 32-slot context; embeddings/MLP/head gradients are accumulated without keeping an unbounded history. Document these different bounded training-state costs. They do not differ on this assay's short framed examples.

No-write means no V, encoder, embedding, moment, learning counter or pending-gradient accumulation. An unpermitted EOS cannot flush prior pending learning; generated EOS cannot learn. Mixed-permission and interrupted-frame behavior must be specified and qualified. BEGIN aborts uncommitted work with a logged counter; it does not create targets. Only previously allowed observations may contribute at a later allowed EOS. Inference cannot commit pending work.

There is no replay, self-labeling, pseudo-target, arithmetic warmup or model reset between operations. Idle rest is the same disclosed no-op; the actual retention pressure is intervening learning.

**Forbidden inside new learners:** payload parsing to numbers, ASCII ordinal values, digit/operator branches, supplied roles, arithmetic functions or inverse relations, numeric targets/losses, answer length/candidates, numeric renderer, restricted alphabet/EOS masks, pretrained arithmetic helpers. General matrix operations are architectural operations, not supplied laws.

## 7. Unchanged data, stopping budgets and causal histories

Reuse the exact V1 generator/scheduler and exposure construction. Unordered seed pairs 0–7 form SUM/PRODUCT families; each has 24 taught and 12 nontrivial withheld families. Hold out every exchange and inverse of each held family. Preserve negative and multi-byte answers, undefined-division exclusions, coverage filtering and novel answer strings.

Use 4,8,16,32,64,128 training records per operation: 512 terminal arithmetic records. Report unique prompts and repeats per prompt at each prefix alongside samples and actual target bytes. The models see only byte identities and public controls, not operation, family, n or evaluator metadata.

| History | Old arithmetic replies | Later binary cohort | Probes |
| --- | --- | --- | --- |
| W | Actual feedback, learning enabled | Learning enabled | No learning |
| N_OLD | Identical sensory feedback; all persistent learning disabled | Learning enabled | No learning |
| SHUFFLE_OLD | Same declared within-operation/budget-block answer permutation; learning enabled | Learning enabled | No learning |

All three histories receive the same original prompt stream. Disclose shuffled fixed points and target marginal differences. No prior correct answers are smuggled into the shuffle arm or test.

At each budget, fork the unchanged arithmetic parent for the same 32-record eight-key arbitrary-binary cohort, then measure old held/taught answers before/after it and after 86,400 simulated seconds. Cohort keys have random 1–3-byte answers and include NUL, LF and '=' as ordinary payload. Same cohort/world across configurations. Root parents never absorb forks or probes.

Each scenario pays 4n+32 external records. Each full job executes three 512-record roots plus 18 cohort forks, totaling 2,112 record applications and 21 **correlated** state trajectories. These are not independent lives or worlds.

## 8. Unchanged ability/sample/compute measures

Own exact correctness requires every canonical answer payload byte and own EOS. All wrong payloads, empties, truncations and missing/extra bytes are wrong. Teacher-forced token accuracy never substitutes for this score.

Retained A(c,h,o,n) is measured after the later cohort/rest. J is the equal average across four operations and the six stopping budgets. It estimates retained accuracy under that stated sample-budget distribution, not accuracy per token or mastery of every budget. Publish all curves and final operation scores.

For each paired candidate/reference comparison, G is the normalized staircase integral of retained accuracy over a common production-CPU allowance T, the smaller terminal scenario cost. Initial zero-feedback accuracy is actually probed. Each measured point becomes available only after its own arithmetic prefix and binary cohort have been paid. Use cumulative maximum for nonmonotone timing points, disclose it, and neither interpolate nor extrapolate unexecuted scores.

Production CPU includes sensory/context processing, decoder work, all head/encoder gradients and optimizer steps for actual teaching. Experimental probes, forks/checkpoints, audit, compression and orchestration are separately reported. All extra L reader state/work is charged. Also report semantic multiply/add work and logical memory traffic; neither is measured hardware bandwidth.

Acquisition diagnostics at every budget: own taught exact, first-byte correct, conditional payload/EOS accuracy, full teacher-forced exact, empty/truncated/error-position counts, margin and feature geometry, encoder displacement, gradient/update and loss curves. Save pre- and post-cohort states. Their function is to explain failure, not to promote an E1-only model.

## 9. Shortcuts, positive-control limits and decision rules

Exact raw-key lookup has no taught answer for a held family; evaluator canonical exchange/inverse recitation also has none. Audit this on every world. These are evaluator controls, not learner components.

Lock four additional feedback-only byte shortcut recipes before science: global complete-answer mode; mode conditional on raw prompt length; mode conditional on the last two raw payload bytes with global-mode fallback; and mode conditional on the unordered raw-byte multiset with length-mode fallback. All learn counts only from the executed old feedback prefix and use identical deterministic byte-string tie rules. Cohort feedback cannot change those old counts: these are protected shortcut references, not matched full-history learning cores. No arithmetic interpretation is supplied. The evaluator computes the per-world/per-operation upper envelope of their final held accuracies as B. Choosing this envelope with labels makes it a conservative **oracle comparator**, never a permissible deployed core.

The V1 design requested shortcut reporting but did not formalize superiority over it. V2 closes that attribution gap **before new data**: keep the existing twelve comparisons and add four final retained W-B operation contrasts to the same Holm family. This changes the prospective comparison count from 12 to 16, not the task, sample budget or success accuracy threshold. It prevents a frequency-based improvement from being called learned arithmetic. Never drop this comparator after seeing results. Whole answer-string novelty is reported, not used to remove difficult cases.

There is still no verified general raw-byte four-operation learning positive control. LEGACY-B is a proven memory/output anchor with disclosed priors; an evaluator-only analytic answer witness verifies construction, not learner efficacy. A failure constrains these recipes at these budgets, not all possible feedback learning.

### DEV

Eight fresh paired worlds, all eight arms: 64 jobs. Only G-CE/L-CE selectable. No tuning or alternate selection after this screen begins.

Eligible only if: own taught W exceeds N_OLD at every budget; final retained W exceeds N_OLD, SHUFFLE_OLD and B in every operation; J(W) and paired G(W) exceed both LEGACY-B and its matched frozen-CE reader; and the new cohort is acquired relative to its own pre-cohort score. These are screening signs, not significance or an invented 80/90 percent gate.

Select the largest mean J gain over matched frozen; tie-break lower production CPU, then G-CE. None eligible -> NO_ADVANCE, analyze/audit/package and stop. SQ/frozen/reference arms can never become unplanned winners.

### Fresh confirmation

At most one selected CE reader advances to 48 fresh paired worlds. Six fixed arms: selected CE, its same-reader SQ, its frozen-CE twin, the other reader's CE, RAW-RLS and LEGACY-B.

Sixteen one-sided paired world comparisons, Holm family alpha 0.05:
- four final retained W-N_OLD differences;
- four W-SHUFFLE_OLD differences;
- four W-B shortcut-envelope differences;
- J and common-CPU G against each of LEGACY-B and matched frozen (four contrasts).

Paired Student-t tests and descriptive two-sided intervals use independent worlds; their finite-sample mean inference is approximate. All accuracy/J/G differences lie in [-1,1]. For zero-variance differences, do not substitute a sign test for a population-mean claim: use the declared bounded-mean Hoeffding p-value min(1,exp(-n*mean_difference^2/2)) for a positive mean difference and p=1 otherwise. Zero-variance descriptive 95% intervals use the conservative bounded-mean half-width sqrt(2*log(40)/n), clipped to [-1,1], rather than a misleading zero-width interval. This prospective inference repair changes no accuracy target or old result. Record the method used and publish every test and failed component.

Only all sixteen passing earns CONFIRMED_RAW_BYTE_FOUR_OPERATION_GAIN. Otherwise NO_CONFIRMED_FOUR_OPERATION_GAIN; keep the incumbent. Selected-vs-SQ and selected-vs-other-reader J comparisons form a separate two-test Holm family; RAW-RLS comparisons and DEV interactions are descriptive. SQ/CE differences remain recipe comparisons with disclosed optimizer scaling.

RULE_GRID_COMPLETE still requires zero errors on every final retained held question, every operation, every confirmation world. It is a distinct bounded competence target, not a revised percentage cutoff or extrapolation claim. A small E3 gain is reported as incremental. No outcome earns mixed-expression, NLP or biological-learning claims.

## 10. Preparation, fixed scope and time limit

Before any science, complete three separately recorded design/audit/test/revise cycles on reserved namespaces. Qualify:
1. Complete-record finite-difference gradients for G/L and SQ/CE, including shared embeddings, normalization and head derivatives; old-source reference parity.
2. No-write guards across every learned component/counter, permitted/unpermitted actual EOS, pure generation and pending-state interruption; snapshots with actual token ring order, all pending gradients and moments.
3. Arbitrary byte, NUL/padding separation, chunk parity, long-stream context policy, bounded memory, vocabulary-renaming parity including output ties; source/dependency isolation from evaluator arithmetic.
4. All history decisions, immutable parent forks, family holdouts, independent endpoint reduction/resource ledger, source identity and tamper rejection.
5. Actual-budget acquisition and held/cohort curves for the complete declared roster. Save weak/zero behavior; gradients or two-key eventual memorization are not evidence of useful arithmetic.

No parameters, window, hidden width, task, alphabet, teaching repetitions or gates may be silently revised to rescue a negative. A preparation revision must be issued before a source lock, with its reason and affected qualification repeated on a new reserved fixture. If all selectable recipes remain behaviorally collapsed, publish a preparation stop rather than launching a costly confirmation. Taught recovery alone does not make a candidate E3-ready. A qualified cheap DEV can still validly end NO_ADVANCE.

Proposed namespaces, pending a complete local/available-history census: technical 202610139801–202610139899; DEV 202610139101–202610139108; confirmation 202610139201–202610139248. No namespace is yet admitted or used.

Measure complete-job wall/CPU costs for every arm, forks, probes, checkpoint/audit/packaging and two-worker parity before admission. Use a conservative 3x measured execution allowance plus separately measured finalization. Admit only if the entire conditional plan fits 8 hours; hard session ceiling 10 hours. Safe whole-job pause only; no automatic resume or failed-trajectory restart. Older V1 jobs took approximately 3–9 seconds, but that is not a V2 runtime measurement or launch ETA.

Counts: DEV 64 jobs/135,168 records; conditional confirmation 288 jobs/608,256 records; maximum 352 jobs/743,424 records, 1,056 roots and 6,336 cohort forks (7,392 correlated trajectories), 56 independent world identities. Maximum is conditional, not a required roster.

If later authorized, two detached single-thread workers, supervisor-bound caffeinate and a read-only live monitor. This design does not launch them. Technical cycles, eventual background run and GitHub publication require their respective user requests.

## 11. What a result decides

The fixed comparison asks whether direct chronological byte access and coherent learned categorical credit can overcome conditional-content failure and produce **retained unseen arithmetic gains at the same sample budgets and charged compute**.

If the new head alone helps, retain the simpler GRU recipe if it earns the E3/CPU tests. If ordered access helps, retain that bounded reader only with its true cost. If taught recall recovers but held cases do not, report E1 improvement and no arithmetic-law gain; do not supply relations, change the split or adopt it as four-operation mastery. If neither works, reject these exact recipes at the declared budget and preserve the complete inventory evidence, without recycling covariance/depth/forgetting knobs or blaming byte generation.

The next design can then be chosen from concrete remaining sources on the observed deficit. That is a result-dependent research decision, not an automatic series of extra patch experiments.

