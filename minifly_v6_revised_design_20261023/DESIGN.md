# MiniFly V6 R1 learned recurrent state with native memory

23 October 2026. Proposed design only. This revises [Claude's V6 memo](CLAUDE_V6_ORIGINAL.md), preserved from [commit 19107b68](https://github.com/ChangweiZhou/digitalbaby/blob/19107b68c85a2f78e8093b5dd41f5ffd162bc7d8/minifly_v6_design_20261023/DESIGN.md).

Test one central idea from V6: **can a small stateful circuit learn useful sequence representations from feedback while the native MiniFly value circuit remains responsible for every answer byte?** Keep learned recurrent gates and local eligibility; defer chunk compilation, procedure/memory arbitration, sleep replay and competitive queuing. Combining those mechanisms now would obscure the cause of either success or failure.

The goal remains better retained own-output generalization to globally untaught addition problems per all observations and actual compute. The existing arithmetic experiment, ADDRESS-CAPACITY results and adoption records are unchanged. This is a new proposed native extension, not a result inherited from them.

## 1 Diagnosis corrected by the latest evidence

| Evidence | Consequence for V6 R1 |
| --- | --- |
| v3 attained taught recall but not held-out addition; the later v5.1-R2 bounded harness had acquisition failures and a resource-limited execution | Separate failure to acquire taught answers from failure to reuse a learned relation. Neither establishes that every native-memory learner is mathematically incapable of composition. |
| ADDRESS-CAPACITY increased final OLD payload exactness from 1.56% to 35.55%, and NEW from 52.73% to 100%, in its eight-pair E1 pilot | Native storage remains a tractable engineering problem. Its improvement cannot be transferred to a learned arithmetic address interface without testing that interface. |
| The same expansion used about 3.15 times the full-life/readout CPU and four times the cell-associated state and edges; its added wiring and self-gain also changed | Do not call it pure population scaling, proven compute efficiency, or an adopted arithmetic backbone. |
| STOP scored 11.26% complete NEW versus 10.64% for fixed one-byte stopping on the earlier saved confirmation responses | Retain STOP's original component result, but compare practical EOS behavior against fixed and position-only references. It is not a demonstrated general answer generator. |
| Native fast/slow storage, signed feedback and nonlinear readout coexist; the full 254-version register includes adaptive encoders, inferred/local context routing and delayed attribution | “Only a fixed linear lookup” and “no previous state/gating mechanisms” are overstatements. The missing evidence is successful generic sequential computation in the current byte-to-native path. |

The earlier native [V9.9A](../../BYTE_CORE_V9_9A/REPORT.md) and [V9.9B](../../BYTE_CORE_V9_9B_RANK/REPORT.md) results show limited E3 feature reuse in synthetic relation families. They do not establish an addition procedure, but they also prevent a blanket claim that native learning can only recite exact pairs.

Remove V6's numerical forecasts about required problem counts and window-versus-carry accuracy. They depend on an unspecified distribution, representation and algorithm. No hidden “200-entry arithmetic table” is implemented or used as a target.

## 2 What is retained and what changes

The answer route is always:

`observed bytes → learned recurrent sensory state → fixed native PN/KC circuitry → native value margins → payload bits and EOS`.

There is no parallel recurrent answer head. The recurrent circuit changes the input representation; it cannot emit a token independently of native value state. Nor is there an analytic answer cache or exact-key store behind the native output.

| Component | V6 R1 choice |
| --- | --- |
| Native substrate | The canonical 5,177-cell, nine-compartment signed native value implementation used by RECOVERABILITY/ADDRESS-CAPACITY CANON; original Q/T/B, clipping, decay and neutral reference |
| Native allocation | Original allocation, `gated=False`; no CREDIT, strict-confirmation gate or replay |
| Working state | One 88-coordinate recurrent neural vector, aligned with the native 88-channel PN interface; all coordinates follow the same generic rule |
| Input representation | All 256 byte identities have exchangeable input columns, initialized without ordinal or arithmetic meaning; no supplied digit classes or semantic labels |
| Replaced front end | v5.1-R2's competitive classes, class-change segmentation and four closed registers are not retained in this new candidate |
| Learned sequence update | Per-neuron retention/write gates and content proposals, with weights reused at every stream position |
| Native output | Original eight payload bits and EOS margin, temperature 2 and physiological neutral subtraction; no external STOP head in the primary arithmetic score |
| Deferred | The 20,708-cell storage extension, external STOP integration, procedure arbitration, chunk compilation, replay, queues, scratchpad/solution traces and reward-only curriculum |

The proposal preserves native **value-learning lineage**, not the entire preceding encoder or its capabilities. The new representation learner and its credit rule are new. The old random-front-end control is an ablation, not a resurrected ELM: no independent linear value matrix, closed-form solver or separately fitted answer decoder is used.

Choosing CANON keeps substrate size fixed across the four arms. It does not reject the positive storage extension. A later integration could compare the same qualified procedure front end on both sizes, but that would be a separate prospective experiment. Do not silently repair a failed V6 run by adding capacity.

## 3 Generic state and local learning rule

Use the explicit candidate in [LEARNING_RULE.md](LEARNING_RULE.md). It is a stochastic gated rate-neuron circuit with local score-function eligibility. It uses no automatic differentiation, backpropagation through layers, or backpropagation through time. Its supervision during this first test comes from actually observed bytes and EOS, not numerical arithmetic error.

The working state is not a designated carry, operand stack, digit counter or content-free rule register. All 88 neurons may mix sensory and contextual information. Whether a useful reusable state emerges is an outcome, not a structure imposed at birth. Gating is independently computed at each neuron; there is no query/key match, normalization over stored items, slot competition or content-addressed retrieval. A generic neural retention gate is not permission to introduce attention under another name.

The controller receives a one-hot code over 256 payload inputs plus three protocol events: BOS, end-of-input and observed EOS. These mark an external record boundary and output termination; `+`, `=` and digit characters remain ordinary payload bytes. Protocol identities are never arithmetic roles. Controller weights map these inputs into learned vectors; ASCII order gives no distance advantage.

The recurrent state supplies the PN vector directly. This uses the centered signed sensory-drive convention of the RECOVERABILITY adapter, not a claim that negative h coordinates are biological firing rates. Replacing surrogate-key drive with byte-derived h is a new interface that requires qualification. Retain the canonical fixed calibration and native side-specific positive-score competition, including its actual activity caps. Do not recalibrate on held probes or use a current target to choose the current KC support. Log active-count shortfalls, ties, support change and native influence on actual training frames. Original calibration is an initial engineering choice, not proof that the new state distribution is well matched to native learning.

Local eligibility avoids inventing a derivative through hard KC winner selection. It may have high variance and may fail to learn. This is a deliberate testable limitation, not a claim that human or fly brains use this exact implementation.

## 4 Slow development and the byte feedback contract

Use the existing gradual native schedule, with n equal to the number of complete external records already observed by that life:

`a(n) = (1/16) × (1/16 + (15/16) × min(n/1024,1))`.

Native signed residuals remain `a(n) × (tanh(m/2) − y)/2`, with EOS masking payload channels. Controller updates have their own small ramp, specified in the appendix. Both clocks continue across OLD, rest and NEW. A fresh life starts at the floor. No mature counter, pretrained arithmetic weights or hidden preceding history is supplied. Because this proposed OLD phase is longer, its average dose and OLD/NEW nominal-strength ratio differ from the earlier short memory assay; preserving the formula does not preserve that earlier aggregate dose.

Do not warm up a representation for free while excluding those observations from the learning budget. There is no separate answer-bearing D0 phase here. Joint representation/native learning starts gently from the first permitted OLD record. Original unlabelled calibration assets and any reused birth information are disclosed separately.

For each record:

1. BOS clears only transient working state, eligibility and record accumulators. Learned controller weights and native value state persist.
2. Observe the prompt once, left to right. Before each actually arriving payload byte, the native circuit makes a next-byte prediction from the available state. Reveal that byte, calculate the permitted feedback, teach using the saved prefeedback address, then advance the recurrent state. BOS and end-of-input are unpredicted protocol events, not extra target classes in the answer codec.
3. At end-of-input, make an own-prefix answer on a private read-only view, capped at 16 payload bytes plus the learner's EOS decision. Feed back only its own emitted bytes during this probe. No true answer prefix, length, number parser or correction reaches the probe.
4. Discard that transient probe state. On the continuing training stream, reveal the real answer as observed bytes, then observed EOS. Predict each before arrival and update from that observation. Teacher-forced next-byte loss is a training diagnostic; it is never reported as free-generation accuracy.
5. Apply the accumulated small controller update once at the record end. Native value teaching occurs at observed events. The current target never rebuilds its own address. There is no prompt replay or native teaching from the generated guess.

This deliberately adds next-byte learning on observed prompts as well as answers. All four arms receive the same opportunities and native teaching schedule. It is not a dose-matched comparison with the historical answer-only R2 run, so historical percentages are not pasted into a fresh arm.

Generation may advance its private h/RNG state, but cannot modify persistent controller weights, native fast/slow state, adaptation, eligibility statistics, calibration or reward baseline. Random tapes for evaluation are frozen before training; report the average of three tapes for every query, never best-of-three selection. All three generations and native reads count toward compute.

Expected bytes remain available throughout this first feasibility test. Sporadic/blurry supervision and final scalar 0/1 feedback are later experiments with warmup-matched controls. Calling the controller's locally derived byte-loss signal “reward” does not demonstrate learning from scalar-only correctness.

## 5 A bounded addition experiment from the start

Replace V6's mandatory ladder of new tasks with bounded technical checks followed by the actual addition harness. Holding a cue, serial recall, parity and modulo arithmetic are not substituted for the project endpoint. No worked solutions, self-written scratchpad or supplied intermediate result is included in this first stream.

Retain the evaluator's existing operand range **0–9,999**, ordinary `a+b=` prompts, ordinary decimal answer bytes and whole commutative-orbit exclusion. These forms are dataset conventions; the learner sees no operand extraction or digit-aligned events. Byte API limits remain generic: 256 input payload bytes and 16 generated payload bytes plus EOS. Exceeding them is explicit truncation, not a successful stop.

Reserve two fresh engineering worlds, `202610236101–202610236102`. Freeze their manifests and RNG schedules before any feedback. The evaluator samples 128 OLD orbits, 32 disjoint NEW orbits and 64 globally withheld orbits, stratified over the same sixteen ordered operand-length strata. Use eight OLD, two NEW and four held draws per stratum. Sample the explicit held set first, then OLD/NEW from its complement, with a global used-orbit set. Do not inherit an unchecked small hash bucket that might contain too few one-digit held pairs. Bound generation attempts and fail a manifest rather than selecting a new seed on scores. Partition exclusion applies to both orientations and every observation, calibration fixture derived from data, and countermodel fit. Held answers are never supplied to the learner.

OLD is 32 seeded passes over its 128 facts: **4,096 records**. NEW is eight passes over its 32 facts: **256 records**. Nonidentical operand pairs have balanced orientations across passes. There is a one-hour simulated rest after each cohort. Report 128 and 32 distinct reinforced relations separately from their repeated presentations. The larger teaching set is an explicit prospective change from R2's 16/8-orbit developmental assay, not evidence of equal sample efficiency against that historical run.

| Arm | Controller learning | Native writes | Purpose |
| --- | --- | --- | --- |
| LEARN | Local eligibility throughout | Observed feedback throughout | Sole candidate |
| FROZEN | Same initial recurrent circuit, weights fixed | Same observed feedback throughout | Tests whether learned recurrent representation helps beyond an untrained stateful encoding |
| NO_OLD | Both controller and native plasticity off during OLD; both on during NEW | As stated | Causal prior-teaching control, with the same observed bytes, targets and clock |
| SH_OLD | Local learning from within-answer-length shuffled OLD answers; correct NEW | Same permitted updates | Tests correspondence, while preserving OLD label/length multisets and timing |

Freeze a target-independent within-length derangement before training. Singleton buckets remain unchanged and are disclosed; no hidden correct-label loss is supplied to SH_OLD. Module dimensions and retained arrays are matched in all arms, with FROZEN allocating dummy eligibility storage for a transparent memory comparison. Charge actual extra update work rather than claiming matched runtime.

All arms share initial state, sensory streams where the control permits, pass order and exogenous random tapes. Controls may generate different internal states and outputs after learning diverges. No phase or arm ID is an input feature. NO_OLD's disabled plasticity affects both learned paths; suppressing only native value writes would not control knowledge already learned in the controller.

Probe read-only at baseline; OLD records 128, 512, 1,024, 2,048 and 4,096; first rest; NEW records 64 and 256; and final rest. Use eight fixed taught and eight fixed held orbits at intermediate checkpoints. Score all OLD, NEW and held orbits at OLD-end, first rest and final rest. Include a pre-NEW score for NEW's later acquisition guard. Holdout probes use both orientations; repeated probes never train the model or select parameters.

## 6 Evidence audit and decision

The intended result is **exploratory E3 within this bounded addition distribution**, with E1 acquisition/retention diagnostics. Two engineering worlds are not confirmatory evidence, regardless of the number of probes, tapes or correlated histories.

| Required check | Interpretation |
| --- | --- |
| Exposure | All OLD/NEW prompt bytes, answer bytes, EOS targets, repetitions and internal data views are counted. A first held choice precedes feedback, and held feedback never occurs. |
| Recitation countermodels | Fit raw prompt lookup, evaluator-canonicalized unordered-pair lookup, byte/answer/length frequency and suffix backoff 0–4 on permitted observations only. A canonical table cannot retrieve an excluded orbit; frequency/suffix shortcuts may still score above zero and must be measured. |
| Positive control | An evaluator-only analytic adder verifies labels and IO scoring; a disjoint native signed-write fixture verifies the port. Neither supplies a representation, parameter or output to the candidate. |
| Attribution | Compare LEARN with NO_OLD, SH_OLD and FROZEN; identify whether prior feedback and learned state are both necessary for the observed held advantage. |
| Memory versus reuse | Taught exactness is E1. A shared but task-irrelevant KC code is not abstraction. A state lesion's score drop establishes dependence on that state, not discovery of a carry algorithm. |

A favorable development result requires, **in each world**, final held exact own-output accuracy strictly above NO_OLD, SH_OLD, FROZEN and every prespecified exposure-only countermodel; positive LEARN-minus-NO_OLD held effects at OLD-end and final rest; positive final OLD taught LEARN-minus-NO_OLD; and final NEW exact accuracy above LEARN's own pre-NEW score. Publish all absolute scores and differences. Ties fail the development screen. These are observed control-relative conditions, not a confidence-level guarantee or an invented 95/99% competence threshold.

Report own EOS, truncation, empty/default outputs and byte accuracy alongside whole-answer exactness. Include evaluator-only carry/length/position strata to locate errors, but never use those annotations for learning, curriculum, gates or model selection. A correct short answer cannot hide failure on the declared four-digit strata.

The sample/compute claim is separate. Plot held, OLD and NEW exactness against all observed records, revealed target bytes, actual native events and cumulative CPU. Compare available checkpoints at common budgets without extrapolating or granting one arm extra trials. More accuracy at higher cost is a trade-off, not automatically better compute efficiency. There is no forecast that 10³–10⁴ examples must suffice.

If taught acquisition fails, diagnose native influence, moving addresses, update magnitude and credit variance. If taught acquisition succeeds but held performance stays at the controls, the tested rule did not establish reusable addition. Neither outcome proves recurrence or local learning impossible. Do not add replay, a carry register, a larger decoder or a different task to turn a failed gate into success.

## 7 Qualification and resources

The prospective main roster is **2 worlds × 4 arms × 4,352 records = 34,816 external record applications**, eight correlated arm histories. There is no separate hidden pretraining corpus. Count exact observed payload/EOS events from the manifest rather than multiplying by an assumed mean record length.

Before main feedback, qualify both interfaces and the learning estimator in a short suite:

- All 256 payload values and EOS, byte order, generic boundaries, own-prefix generation, overflow and absence of teacher data in probes.
- Gate/proposal score identities by finite differences of their local log probabilities; a tiny analytically enumerable stochastic circuit to check the expected update direction. This is a credit-implementation test, not proof it trains the full model.
- Exact no-write behavior, restored controller/native/RNG state, independent mutable state per arm, canonical birth hashes and frozen calibration.
- Native signed self/cross influence on actual training-only frames, plus passive/clipping references. Low disjoint-fixture error is not sufficient qualification of moving learned addresses.
- Countermodel fits and global orbit exclusion; shuffling identities/lengths; trace and gradient accumulation ordering; no frame-history tape, replay cache, attention-like routing or automatic differentiation in the learner dependency closure.

Technical fixtures have ceilings of 64 full synthetic records and 512 partial native events, with every actual event logged separately. The estimator's toy numerical checks do not count as native science. They may not be used to select the arithmetic hyperparameters.

Proposed total ceiling, including fixtures, all main arms, production, copying, evaluation, reduction, audit and packaging: **7,200 cumulative process-tree CPU seconds, 10,800 active wall seconds, 2 GiB sampled tree RSS**, one worker and one numerical thread, with measured 2 GiB admission headroom. This is a proposed ceiling, not an approved or launched run. Include failed setup/attempts in the operational ledger. Profile the first complete planned history, including its full probes, and stop with a preserved partial report if a conservative projection cannot fit the remainder plus audit reserve. Do not drop arms, shrink the range, reset clocks or extend the ceiling after seeing scores.

The controller has 61,248 weight scalars, equally many eligibility scalars and equally many record-gradient scalars: 1,469,952 float64 bytes before h, input/RNG state and small statistics. Native persistent state, fixed connectivity, calibration, working query views, logging and all overhead are additional. These are bounded neural arrays, not a history-sized store. Eligibility maintenance is dense; the inherited native path still scans/projects/sorts. Report visited cells/edges, copies and bytes moved. No sparse-compute claim follows.

Freeze the exact native/birth dependency closure, controller initialization, graph/calibration assets, manifests, loss/reduction code and all numerical caps before held evaluation. A future three-cycle exercise should implement, audit and test this contract; it must preserve every failure and revision. A formal confirmation would need untouched worlds after that exercise.

## 8 What remains from the human learning proposal

Use the biological analogy to motivate retained state, reuse of the same update at successive positions, and feedback-dependent plasticity. Do not treat human arithmetic activation, chunking signatures or faster repeated answers as evidence that this proposed controller will discover addition.

[AuGMEnT](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1004060) is a computational precedent for reward, tags and working memory; it explicitly uses action-related attentional feedback. Its unmodified algorithm is therefore not presumed compliant with this project's attention prohibition. [E-prop](https://www.nature.com/articles/s41467-020-17236-y) supplies a distinct precedent for local eligibility and learning signals in specified recurrent spiking models. This rate-neuron/native-memory hybrid is not an implementation of that paper and inherits none of its performance claims.

The concrete candidate uses the more elementary local stochastic score principle associated with [REINFORCE](https://doi.org/10.1007/BF00992696), specified in the appendix. Local score factors do not require backward differentiation through a recurrent computation. Whether their variance is tolerable in this hybrid is precisely an open engineering question. Clipping, gradual updates and a changing native learner mean no convergence guarantee is claimed.

A procedure-to-memory transition could be studied only after a procedure arm generates correct new answers before feedback. Its guesses must not become “confirmed” native targets without genuine external confirmation. For now the native memory already receives true observations directly, and no self-generated answer is consolidated.

## 9 Delivery boundary

V6 R1 ships a design, an explicit learning-rule appendix, a source/evidence record and a comparison against the complete mechanism register. It does not assert that the user approved “one-step local credit,” competitive queuing, worked solutions or an increased execution budget merely because Claude's memo used that wording.

Existing constraints remain: gradual warmup, no supplied arithmetic structures/rules, no attention/Transformer/ELM, and no backpropagation in the proposed native candidate. Prior external-head gradient experiments keep their separate historical scope; they do not grant or revoke a blanket rule for every other project.

No new implementation, three-cycle exercise, training, formal confirmation, GitHub publication or adoption is started by this delivery. Current arithmetic and completed memory pointers remain unchanged; a separate proposed-procedure pointer records this document.
