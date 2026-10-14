# RAW-RULE V1: learn arithmetic from observed byte feedback

13 October 2026. **Design only. Implementation, three-cycle qualification and science have not run. This document does not authorize a launch.**

## 1. Fixed goal and the correction

Obtain better retained, exact own-output generalization to unseen four-operation arithmetic from fewer external examples and less computation. All input and output payloads remain arbitrary bytes. The learner must acquire the useful relationships from feedback; supplying arithmetic relationships is not the experiment.

The previous recommendation to supply operand roles and an operation-conditioned numerical relation is withdrawn. This design supplies neither. It also removes the previous numerical target, ASCII digit values, base-ten periodic features and numerical output machinery. It tests a general learning architecture that could consume other byte data through the same interface; it does not claim NLP competence.

The existing successful addition core had explicit additive structure and an ordinal decoder. That evidence remains valid in its original scope. It does not demonstrate that arithmetic structure was discovered from raw bytes. RAW-RULE earns its own evidence.

The scientific task remains the established interleaved four-operation, single-binary-expression task. Precedence, parentheses and genuinely mixed expressions are subsequent capabilities, not results this task can establish. No noise optimization, scaling, pretraining or Q&A is added to this experiment.

## 2. What is supplied, and what must be learned

| Item | Supplied | Must be learned from observed feedback |
| --- | --- | --- |
| Interface | Byte identities 0–255; public BEGIN, REPLY and EOS controls | Which answer bytes to emit and when to stop |
| General architecture | Finite recurrent state, tied weights, generic gates, plastic categorical readout | Useful context features, distinctions between symbols, dependencies and computations |
| Arithmetic | Evaluator generates teaching examples and withheld tests | Any useful interpretation of digits, operation symbols, sign, multi-digit composition and arithmetic relationships |
| Feedback | Actual observed reply bytes and actual EOS | Predictions and internal parameter updates driven by these observations |

An architecture necessarily has inductive biases. Recurrence, gating, parameter sharing and confidence-weighted learning are disclosed general biases, usable without arithmetic annotations. They are not an arithmetic parser, a supplied equation or a claim of unbiased learning.

**Forbidden inside every new core:** integer/float conversion of payload text; digit-value lookup; supplied operand slots or operation IDs; arithmetic-specific routing; add/subtract/multiply/divide answer functions; inverse-equation solvers; carry/sign/decimal-place targets; numerical auxiliary loss; handcrafted decimal features; answer candidates, expected lengths or evaluator labels; pretrained helpers; arithmetic-specific warmup; formatting a predicted number into the scored answer; masking EOS or restricting output to arithmetic characters.

Ordinary matrix products and gating multiplications are permitted general neural operations. Their parameters are not initialized to arithmetic rules. There is no operator-specific code branch.

Arithmetic and integer conversion may exist in the evaluator to create examples, check correctness and construct family holdouts. The learning package must not import that evaluator. The only training target reaching the learner is the arriving byte or EOS.

## 3. Mechanisms and inventory provenance

This uses the full [108-family comparison](../PROJECT_DOCUMENTS/MECHANISM_SCREEN_20261007/FULL_REVIEW_POST_U034_20261011/HISTORICAL_108_COMPARISON.json) and [254-version register](../PROJECT_DOCUMENTS/MECHANISM_SCREEN_20261007/VERSION_REGISTER_20261011/VERSIONS.json), not an abbreviated survival list. No catalogue-wide exclusion is inferred from this experiment.

| Source | New, isolated operation | Evidence boundary |
| --- | --- | --- |
| M001 stateful nonlinear cells; M007 GRU-like cells | Learned recurrent byte state, plus one versus four tied processing steps at prediction | New engineered variant. Historical cells differ in graph, private state and training; no inherited arithmetic result |
| M024; V23/adapter_retained | Feedback changes the internal representation, compared with a matched frozen encoder | Hidden backpropagation is explicitly used; this is not validated biological local learning |
| U045; SAMPLE_GEN_V2/COV-R | Full versus diagonal confidence in a categorical readout | New vector-output use. Earlier scalar-addition gains do not prove success here |
| M061/M106 reference-drift findings | One operative readout and one fitted prediction, without adding independently trained logits | Removes an experimental confound; does not claim that reference drift or forgetting has generally been solved |

The two principal factors are general internal computation depth and correlated readout credit assignment. Learned versus frozen encoding is a matched control. This is not a new batch of unrelated mechanisms.

## 4. Core recipe

### 4.1 Raw byte reader

Learned embedding E has 259 rows and width 16: 256 payload identities and the three public controls. Recurrent state h has width 64. It starts at zero on BEGIN and consumes controls exactly as it consumes payload identities. The input contains no ASCII ordinal values, operand positions or arithmetic metadata.

Use the same gated transition F for all byte identities and all tasks:

```text
z = sigmoid(Wz e + Uz h + bz)
r = sigmoid(Wr e + Ur h + br)
g = tanh(Wg e + Ug (r * h) + bg)
F(h,e) = (1-z) * h + z * g
```

After the observed prefix is consumed, prediction starts from its current h. For depth K, compute K-1 additional F transitions with a zero input vector; K is either 1 or 4. These temporary processing states are not external training observations. They do not change the stored prefix state, and repeated prediction calls are pure. The next actually observed or generated payload advances the prefix state through F using that byte. All weights are tied across steps.

The readout feature is u = concatenate(h_processed,1) / sqrt(||h_processed||^2 + 1), with 65 entries. The constant feature is generic. There is no separately learned unconditional categorical bias outside this same fitted readout.

For long streams, the core retains a bounded recurrent summary and a rolling gradient tape, not the complete byte history. It accepts arbitrary stream lengths and reports eviction from the tape. Older inputs may remain compressed in h; this is not an exact last-4096-byte window or unlimited context. That context policy is an explicit architectural change. State and chunking parity must be qualified.

### 4.2 One plastic output memory

V has shape 65 by 257. Logits are z_output = V.T @ u. Softmax reports probabilities and argmax chooses the next event. There is **one operative categorical predictor**: no exact-address bank is added to its logits, no external renderer and no learned-value-to-string shortcut.

Before an observed target t arrives, save its prediction and u. After arrival, form e_target = onehot(t) - z_output. The fit objective is 0.5 * ||e_target||^2, identical for all new configurations. Confidence initialization is P = 10 I for FULL and p = 10 for DIAG.

```text
FULL: q = P @ u       DIAG: q = p * u
 d = 1 + u.T @ q
 V += outer(q, e_target) / d
 FULL: P -= outer(q,q) / d, then symmetrize
 DIAG: p -= q*q / d
```

These are the declared full/diagonal recursive fitting rules. The diagonal rule is an approximation, not exact full RLS. Once the encoder changes, FULL is also a recursive approximation over a changing feature basis; P is not an exact posterior covariance. There is no uncharged coordinate transport or confidence reset. Feature drift is an explicit risk tested through the learned/frozen and FULL/DIAG comparisons. Qualify positive denominators, finite state and numerical stability. Do not silently add clipping, resets, forgetting factors or jitter after seeing science.

EOS participates in the same contextual target fit as every payload byte. Empty replies are legal. EOS is neither banned at the first position nor imposed after an expected number of digits.

### 4.3 Representation learning and timing

Encoder parameters are fixed within each framed observed sequence. Each permitted observation updates V and its confidence immediately after prediction. At that observation, retain the feature derivative of the squared error using **the pre-update V and prediction**. Those derivatives accumulate through the actual prefix, not a generated substitute.

At the actually observed EOS, backpropagate the accumulated loss through at most the last 32 external events, including their tied internal processing steps, and apply one encoder update. Older recurrent state is a detached constant. The finite tape, pending gradients, update permissions and pending sequence state are included in snapshots. A new BEGIN without EOS discards uncommitted encoder gradients, records the aborted sequence and does not manufacture a feedback target.

Use the mean over permitted target events, global gradient norm clipping at 1, and the previously disclosed RMS second-moment rule: beta=0.99, learning rate=0.001, epsilon=1e-8, bias correction. These are fixed engineering choices, not arithmetic rules or claimed optimal settings. They may be revised during reserved technical cycles, never after DEV is inspected without issuing a new design version.

All gradients use the appropriate pre-update readout from their event. Do not treat the whole online sequence as though its readout had been frozen. The implementation must qualify the accumulated feature derivatives against an independent small reference with the online readout snapshots held fixed. It must also qualify both K values and both confidence rules.

The embedding starts with seed-controlled random signs scaled by 1/sqrt(16). Input matrices use zero-mean Gaussian initialization at 1/sqrt(16); recurrent matrices use independent orthogonal initialization. Biases, V and optimizer moments start at zero. Initialization substreams and common arrays are paired across configurations.

Learning-enabled prompts are supported by the general API, but this arithmetic experiment disables prompt updates in every history. Encoder gradients and readout updates are permitted only for actual observed reply bytes/EOS in the declared history. Generation updates neither weights, confidence nor optimizer state, and does not flush pending training gradients.

This is an engineered recurrent learner. Dense vector state, generic gates and BPTT do not make it an unchanged Full151 or prove a fly-local learning mechanism.

## 5. Fixed roster and fair contrasts

| Configuration | Encoder | Prediction depth | Confidence | Role |
| --- | --- | --- | --- | --- |
| R1-DIAG | Learned | 1 | Diagonal | Diagnostic factorial arm |
| R1-FULL | Learned | 1 | Full | Selectable |
| R4-DIAG | Learned | 4 | Diagonal | Diagnostic factorial arm |
| R4-FULL | Learned | 4 | Full | Selectable |
| F1-FULL | Same random encoder, frozen | 1 | Full | Matched encoding control |
| F4-FULL | Same random encoder, frozen | 4 | Full | Matched encoding control |
| LEGACY-B | Frozen published implementation | Published | Published | Current generic arithmetic reference |

All six new configurations receive identical observed bytes and controls. Learned/frozen twins have the same initial recurrent parameters and output state. Depth changes computation, not parameter count. FULL adds confidence state and computational cost; this is charged, not disguised with unused arrays.

LEGACY-B is the exact published LINEAR-FIXED-FULL implementation. It includes its disclosed ASCII-value input features and an episodic bank. It is retained as a stronger historical performance reference, not a raw-byte architecture control. Gains over it cannot be assigned to one isolated mechanism. None of its numerical features enter the six new cores.

No current specialist core is replaced by shipping this design. No old result is relabelled.

## 6. Teaching data, held-out families and feedback

Keep the frozen task-family generator used by VALUE-INTERACT. For each world, unordered seed operands are 0..7. There are 36 SUM families and 36 PRODUCT families. In each type, 24 families teach and 12 are withheld from the nontrivial 2..7 pool. A family and all its exchanges/inverses remain entirely on one side of the split.

SUM(a,b), s=a+b gives its addition exchanges and signed subtraction inverses. PRODUCT(a,b), p=a*b gives multiplication exchanges and defined exact integer division inverses. This family grouping is an **evaluator leakage exclusion**, not information supplied to the core. Preserve negative answers, multi-byte answers and whole answer strings absent from teaching.

Use the existing balanced rotating operation-block schedule, with 4/8/16/32/64/128 teaching records per operation. A root history applies at most 512 arithmetic records. The core receives raw prompt bytes, REPLY, actual answer bytes and actual EOS. It never receives op, a, b, y, family, checkpoint, query type or expected answer length as metadata.

No added pretraining, format warmup, numerical auxiliary feedback, replay, generated pseudo-labels, noise treatment or correctness reward is used. The external feedback is the actual observed example answer. Character coverage is audited and the inherited generator's coverage filtering is disclosed. Novel answer strings are not removed. Noise can be studied later; it is not this experiment's target.

### 6.1 Persistent ability at each budget

At every arithmetic budget, fork the current state for a fixed 32-record non-arithmetic interference cohort. There are eight distinct random 4-byte prompts, each taught four times with a random 1–3-byte payload answer and EOS. Core input is ordinary arbitrary binary data. Include NUL, LF and `=` as payload in the cohort's construction; no byte is a hidden delimiter. The evaluator excludes exact overlap with arithmetic prompts. All configurations and history arms receive the same cohort at a given world, independent of predictions.

Measure held arithmetic before the cohort, immediately after it and after simulated 86400 seconds, using model-generated bytes plus model-selected EOS. Also measure acquisition of the new cohort. In this recipe idle time does not decay weights; the rest is a disclosed no-op. Actual retention pressure is later learning. An inference or rest call must not perform a delayed pending encoder update.

The arithmetic root resumes from its unchanged parent state after each fork. It does not absorb cohort writes or probes. These are independent stopping-budget scenarios sharing a root prefix, not extra exposures in the same learner's arithmetic history.

Total feedback per evaluated scenario is **4n + 32** records: 48,64,96,160,288,544. Report both arithmetic examples and total examples. The interference examples, their bytes and their computation are not free.

### 6.2 Explicit causal branch map

| History | Arithmetic reply writes | Answer order | Later cohort writes | Probes/generated bytes |
| --- | --- | --- | --- | --- |
| W | Enabled | Actual answers | Enabled | Disabled |
| N_OLD | Disabled in every persistent component | Same actual answers | Enabled | Disabled |
| SHUFFLE_OLD | Enabled | Answers permuted within the existing operation/checkpoint blocks | Enabled | Disabled |

N_OLD is deliberately named: it disables the prior arithmetic episode, not later learning. Shared parameters, embeddings, V, confidence, optimizer moments and pending gradients all obey the same permission. Transient sensory state still advances. Shuffled targets supply feedback only after their own arrival; the core never sees the correct unshuffled answer on that branch.

Log actual update decisions for every target and every persistent component. N_OLD arrays must be independently write-protected during the arithmetic episode. Generation write protection covers all learned arrays and learning-related counters. Negative tamper tests must reject a single illegal update, a pending-gradient leak and a cohort-branch reversal.

## 7. Own-output scoring and resource accounting

A probe starts with only the prompt and public REPLY. Its first prediction precedes feedback. Exact correctness requires the complete canonical answer payload and an own EOS within the fixed 16-byte generation cap. A truncated, empty or partially correct reply counts wrong when it differs from the target. There is no teacher prefix at inference, candidate-answer selector, integer rounding or numerical surrogate.

For each operation, budget and history, score the complete held-out grid. Average within operation, then equally across the four operations, then equally across worlds. World is the independent statistical unit; probes and forks are correlated.

Let retained accuracy A(c,h,o,n) be after the interference cohort and final rest. Define the sample score J(c,h) as the average of A over the four operations and six stopping budgets. It is expected retained exact accuracy under an explicitly uniform distribution over these six tested budgets; it is not a claim about every sample budget. Primary causal transfer is J(W)-J(N_OLD), with SHUFFLE_OLD as a second control. Also report raw W accuracy and final per-operation W-N_OLD and W-SHUFFLE_OLD contrasts.

For computation, use measured single-thread production CPU for the W arithmetic prefix plus that scenario's cohort. Exclude evaluator, audit, compression and probe CPU from production, but report them separately. Record semantic multiply/add counts and memory traffic estimates as well, including covariance operations and backpropagation. No idle waiting is counted as computation.

For a candidate/reference pair in a world, T is the smaller terminal production CPU. Construct each retained accuracy curve as a left-continuous staircase: zero feedback budget uses its initial score; a budget point becomes available only once both its arithmetic prefix and 32-record cohort cost have been paid. Include a point only if its measured cost is at most T. Its score applies up to the next available cost or T. If measured costs are not strictly increasing, use their cumulative maximum and disclose the correction; do not smooth accuracy or extrapolate beyond executed points. The normalized integral over [0,T] is G. Compare paired G(W) values at this common compute allowance. This is a hardware-specific compute-efficiency result; the logical-work ledger is its reproducibility companion.

An arithmetic-prefix timestamp must not include the cost of earlier diagnostic forks. Nevertheless the job ledger charges those forks to total experimental execution. All compared scenarios include their own later cohort. There is no hidden warmup or unpaid optimization pass.

Report persistent and temporary bytes, training and generation compute, output errors, empty replies and truncation. Representation parameters: 19,696. Readout parameters: 16,705. FULL confidence: 4,225 numbers; DIAG: 65. There is no categorical episodic array in the new cores. These are vector-valued engineered states, not literal biological neuron counts.

## 8. DEV, fresh confirmation and decision rules

### DEV

Eight fresh paired worlds, all seven configurations: **56 full jobs**. Only R1-FULL and R4-FULL are selectable. DIAG and frozen arms are diagnostic controls, never alternate winners. No hyperparameters, data, output masks or endpoint change after DEV begins.

Eligibility uses signs, not a chosen 70/80/90 percent accuracy cutoff:

1. At every tested stopping budget, taught arithmetic W accuracy exceeds N_OLD accuracy; a middle-budget zero-output collapse is ineligible.
2. At the final retained endpoint, every operation has positive W-N_OLD and W-SHUFFLE_OLD mean differences.
3. J(W) exceeds both LEGACY-B and the matched frozen-depth control; paired compute score G(W) also exceeds both.
4. The later binary cohort is actually acquired: W post-teaching exact accuracy exceeds its pre-cohort read-only accuracy.

These are screening signs, not significance claims or scientific confirmation. Among eligible FULL recipes, choose the largest mean J(W)-J(matched frozen). Tie-break by lower production CPU, then depth 1. If none is eligible, stop, analyze, audit and package a valid NO_ADVANCE result. Do not rescue it by changing the benchmark or adding arithmetic hints.

### Confirmation

At most one selected FULL recipe advances to 48 fresh paired worlds. Roster: selected recipe, same-depth frozen encoder, other-depth learned FULL arm, same-depth learned DIAG arm and LEGACY-B. These five configurations are fixed by the selection map, not chosen after confirmation outcomes.

The twelve primary one-sided superiority tests are: four final retained W-N_OLD operation differences; four final retained W-SHUFFLE_OLD operation differences; J(W) gain versus LEGACY-B and versus matched frozen; G(W) gain versus these same two references. Use paired world Student-t tests with Holm correction at family alpha 0.05. For a zero-variance contrast, use the exact one-sided paired sign test on nonzero world differences; all positive differences have p=2^(-number of nonzero worlds), while all-zero or constant-negative differences cannot pass superiority. Tests and point estimates are all reported, including failures. For confidence reporting also publish paired two-sided intervals as descriptive, explicitly not simultaneous intervals.

Only if all twelve pass, issue CONFIRMED_RAW_BYTE_FOUR_OPERATION_GAIN. Otherwise issue NO_CONFIRMED_FOUR_OPERATION_GAIN and retain the incumbent. Report a positive component without turning it into success on all four operations. Two secondary J contrasts—selected versus its depth and covariance ablations—form a separate Holm family of two; they support only their explicitly matched mechanism comparisons. No confirmed factorial interaction claim is made from the DEV grid.

**A statistical gain is not the same as learning the laws reliably.** Report RULE_GRID_COMPLETE only if the selected W core makes zero errors on every operation's exhaustive held-out grid in every confirmation world at the terminal retained budget. This is the bounded task's error-free capability target, not an invented percentage cutoff, not a DEV launch gate and not a universal proof of arithmetic. A gain with incomplete grids remains an incremental research result. Neither verdict earns NLP, mixed-expression or biological-learning claims.

There is no already verified raw-byte four-operation E3 learning positive control. LEGACY-B verifies a working output/memory reference; an evaluator-only analytic witness verifies task construction. The old specialist can check its original addition scope outside scientific counts. None is mislabeled a successful general learner. Consequently a negative result excludes these recipes at this budget, not generic recurrence or the possibility of learning arithmetic.

## 9. Exposure, shortcuts and information isolation

An exact raw-key or canonicalized-key table sees no answer for the held families. Exchange/inverse family recitation is excluded by the family split. Report exact lookup, most-common answer, operand-frequency and inverse-family recitation countermodels using only the observed training prefix; they run in the evaluator and never generate a core reply.

Repeated input characters and previously observed output characters are permitted experience. Recovering or using a relation from that experience is the desired learning. No held answer, complete held family or correct current output is supplied at a query. Log which complete answer strings are novel; failing to generate them is an error, not a reason to change the split.

A qualification-only payload-permutation fixture renames every byte through a random bijection. New cores must pass interface/state and probability parity with correspondingly relabelled embeddings and output columns. Selected events must map correctly when the maximum is unique; with exact ties, compare the mapped set of maximizers, since the generic smallest-ID tie rule is not permutation-equivariant. This is an equivariance check, not extra arithmetic teaching or an E3 test. Empty replies, binary payloads and non-arithmetic strings must remain legal. Source audit rejects byte-ID branches that give digits or operators special mathematical meaning.

The finite holdouts, causal controls and error grids can demonstrate learned reuse within this task. They cannot establish a unique discovered algorithm: multiple learned functions can fit a finite arithmetic domain. Do not promote extrapolation outside 0..7 seed families or unknown mixed expressions without an actual test.

## 10. Qualification, cost ceiling and integrity

Three design–audit–test–revise cycles are a future preparation task. They must use reserved technical worlds only. This document claims no cycles completed. They must verify: arbitrary payload and chunking parity; own-output acquisition and legal empty outputs; learning and inference write protection; full state restore including tape and pending gradients; finite-difference encoder gradients with online readout timing; FULL/DIAG stability; K=1/4 tied-step gradients; all three branch histories; parent/fork isolation; corpus-family exposure; independent endpoint reduction; tamper rejection; and the exact stopping-budget acquisition/output curves.

Use low-dose technical fixtures and actual arithmetic budgets. The previous five-context/1,024-record eventual-memorization fixture cannot stand in for low-sample readiness. Surface taught collapse and zero held scores in the preparation report. A source-correct but behaviorally collapsed primary candidate is not ready for science. Revised parameters must produce a new preparation version and repeat the affected checks before the lock; do not hide technical failures.

Save compact pre-arrival reply probabilities, selected EOS probability, features and full generated payloads for probes. Save actual loss, unclipped gradient norm, clip factor, parameter update norm, confidence denominator and state identities for training targets. Retain W checkpoints at all six budget boundaries and retained fork endpoints, plus final N_OLD/SHUFFLE_OLD states. These make later read-only diagnosis possible. Whole-state hashing is charged audit work at checkpoints and probe batches, not compulsory production work on every byte.

Proposed namespaces: technical 202610130100–202610130199, DEV 202610130201–202610130208, confirmation 202610130301–202610130348. The namespace check performed while drafting checked existing arithmetic manifests only; final qualification must census all historical namespaces before use.

Each full job executes three arithmetic roots and 18 cohort forks: **21 correlated state trajectories**, not 21 independent lives. At maximum, the roots apply 3*512 records and forks apply 3*6*32 records, for **2,112 external record applications per job**. DEV is 118,272 record applications. Conditional confirmation is at most 240 jobs/506,880 records. Maximum entire experiment: **296 jobs, 888 root histories, 5,328 cohort forks and 625,152 record applications**. Shared prefixes are executed once; checkpoint serialization and qualification fixtures are not scientific records. Count only independently accepted complete jobs as committed.

Plan two detached single-thread workers with supervisor-bound caffeinate and a read-only live monitor, only after explicit authorization. Measure the cost of complete technical jobs and retention forks before launch. Require a conservative measured estimate of the whole conditional experiment within an eight-hour admission budget; do not claim an ETA from the earlier MLP. A ten-hour hard session limit permits a safe whole-job-boundary pause. No automatic resume, no interrupted trajectory restart and no change to n. If cost cannot qualify, return the design for a prospective revision before science.

Final independent audit covers roster, source identity, branch/write decisions, parent/fork continuity, receipts, exposure, all resource charges and both primary/secondary families. Reports and packaging follow even on a negative or NO_ADVANCE outcome. Science is not rerun to obtain nicer logs. GitHub publication and science launch require the corresponding user instruction.

## 11. What this next experiment decides

Can a generic feedback-trained recurrent memory, with bounded internal computation and correlated categorical updates, gain and retain exact four-operation ability beyond a frozen representation and the current generic reference under matched external observations and charged compute?

If yes, preserve that general learner and test truly mixed expressions next. If no, record the full error and acquisition curves and reject the tested recipes at this budget. Do not claim arithmetic laws were supplied, that a tiny gain solved them, that noise handling is the next milestone, or that all remaining mechanisms were ruled out.
