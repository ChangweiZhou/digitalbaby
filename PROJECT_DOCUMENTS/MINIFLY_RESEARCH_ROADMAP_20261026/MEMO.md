# MiniFly Research Roadmap — Post V149–V156

**Date:** 2026-10-26  
**Type:** Strategic research-coordination memo  
**Status:** Proposal only. No experimental authorization, model adoption, branch merge, or alteration of existing scientific design pointers.  
**Evidence snapshot:** Codex f32a9bdf71e9035d3a42c2ebf58a425d2cb95082; Claude 04c2873efb5ca8cc70eacc45052ec9e71231a93f.

## Executive decision

Consolidate the previous scattered recommendations into **three parallel investigation lanes** plus a **gated integration and evaluation path**:

- **A. Native acquisition and write accumulation (suggested lead: Codex).** Why do individually beneficial Full151 writes fail to accumulate into reliable complete answers?
- **B. Locally learned representations (existing Claude V155 lane).** Can PN-to-KC connectivity form predictive, transferable features without backpropagation?
- **C. Architecture sufficiency (new, independent lane).** Does the current PN-to-KC-to-memory/readout-centered system have the *computational functions* needed for multi-step reusable learning, or does it require a learned workspace and memory-to-computation feedback?

Do not gate C on success of A or B. Do not presume A+B is sufficient. Keep PN→KC→MBON/Full151 as a valuable **associative-learning baseline**, not an unproven final cognitive architecture.

**Important status correction:** Claude V155 passed C0 testbed but at C1 its implemented L-PRED rule **failed the predeclared synthetic same-future-representation fixture on both streams**. Formula/determinism checks passed. C2/C3 learning results have **not** been earned, and the rule should not advance automatically. A revision requires a fresh mechanistic justification and proper approval.

## 1. End goal and evidence contract

The fixed scientific target is a local-learning system that acquires reusable knowledge, uses it in computations, retains knowledge *while still relevant*, and answers **never-taught addition relations** with its **own full byte sequence and EOS**, efficiently in total samples/feedback and real compute. Addition is a hard, concrete test of the broader language-learning program, not a claim that all intelligence reduces to arithmetic.

Continue to preserve:

- Generic raw-byte input and 257-class output; no arithmetic parser, operand/digit/carry features, answer-length cue, forced EOS, attention, Transformer, or inherited ELM solution in a candidate.
- Exact distinction between allowed teacher-supplied reply/EOS during training and *autonomous* own-prefix generation in assessment.
- Complete whole-relation/orbit holdouts, matched no-prior-teaching, lookup, frequency, shuffled/replayed-label and relevant feedback controls.
- Prelocked worlds/rosters, intact negative results, and separate counters for unique information, repeated presentations, inhibited writes, internal optimization reuse, native evaluations, query clones, CPU/wall and failed attempts.
- BPTT and batch-fitted readouts may be **reference instruments**. They never provide answers on behalf of the native fly and their success cannot be credited to its local learning.

### Evidence ladder

| Level | What has been shown | What it must not be called |
|---|---|---|
| E0 | A circuit/update/interface behaves as defined, or a finite solution witness exists | Learned behavior or compositional reasoning |
| E1 | Complete *own-generated* taught-answer acquisition | Unseen rule generalization |
| E2 | Still-useful acquired knowledge survives prescribed subsequent learning and delays | Keeping every old or obsolete fact |
| E3 | Complete never-reinforced relations answered beyond recitation, frequency, and matched teaching controls on fresh worlds | One favorable development contrast or teacher-forced loss |

E2 is an explicit roadmap stage; historical files may use slightly different terminology.

## 2. Honest current evidence

| Source | Supported result | Not established |
|---|---|---|
| V149/V153 | Frozen KC codes preserve useful text context; fade and permutation affect accessible information; a simple online output-local classifier can learn useful predictions | A trained native feature encoder or transferable procedural representation |
| V151 | Tested single native signed writes respond beneficially; B's apparent one-step deficit is largely dose mismatch | Full-sequence learning from corrected writes |
| V152 | Exact BPTT and local eligibility comparators both fail taught complete addition under their particular forward/port/learning configuration, often terminating at EOS | An impossibility theorem for BPTT, local learning, or architecture |
| V154 | Some native permutation arms acquire isolated complete facts, but held addition is tiny and inconsistent across worlds; no explicit class bias is required for EOS collapse | Reproducible E3 |
| V156 | Free evaluator readouts find finite decision witnesses in all 12 tested configurations; self-write margins improve 178/178 versus no-write; just-taught complete NEW answers 0/64; 15/15 already-correct OLD survive sampled final probes | Native reachability of those witnesses or a general memory-protection advantage |
| V155 headroom / C0 | Learned reference features suggest room beyond frozen keys; C0 qualified its declared measurement pipeline | Sparse native binary-key learnability |
| **V155 C1** | Individual formula and determinism checks succeeded, but the intended predictive representation change failed its synthetic functional test | Authorized meaningful C2/C3 or cross-text transfer |

A full-row-rank finite input matrix can support arbitrary taught-label interpolation without learning addition. Conversely, failure of a particular external optimizer does not prove an architecture cannot express the required computation. Keep **representability**, **native reachability**, and **transfer** separate.

### V155 C1: actual numbers requiring explanation

The C0 development scores were FROZEN **2.962895**, REF-N **2.931345**, REF-D **2.739929**, PERM **3.029604** bits/symbol. The reference hidden layers are graded and descriptive, **not** certified bounds for the binary native code.

At C1 the registered *same-future minus different-future KC overlap* criterion required a gain of at least **+0.01 on both streams**. Instead:

| Synthetic seed | Frozen index | L-PRED after specified learning | Difference |
|---|---:|---:|---:|
| 0 | 0.004236 | 0.001897 | −0.002340 |
| 1 | 0.005030 | 0.003291 | −0.001739 |

Nine technical checks passed, including matching the implemented forward/retrograde update algebra. The functional fixture failed. The rule was executed faithfully but did **not** demonstrate its stated representation-forming function. Do not broaden the training run or revise an outcome criterion after seeing these data. L-COMP's formula checks are not yet evidence of substantive representation learning.

## 3. Dependencies and parallelism

    P0: Evidence register / common scientific contracts
           |
           +---- A: Native write accumulation -----------------+
           |                                                   |
           +---- B: Predictive PN-to-KC representation --------+--> D: Gated integration
           |                                                   |           |
           +---- C: Architecture sufficiency ------------------+           v
                                                            E1, E2, E3
                                                                |
                                                      Broader sequence / NLP

**A, B and C are parallel inquiries.** Native Full151's failure must not veto independent architecture research; an ill-qualified BPTT control must not prevent a valid native E1 experiment; a useful representation-learning result must not be called a final cognitive architecture.

## 4. P0 — Cross-project evidence register

**Deliverable:** One source-linked register for versions, hashes, active branch, exact candidate architecture and synaptic plasticity sites, data/supervision, actual training/exposure/compute, controls, qualifications, and E0–E3 outcomes.

Use existing data rather than replaying past experiments just to populate the register. The reducer must label evaluator-only fits and unpublished local arrays explicitly; repeated target positions, branches and clones are correlated observations, not independent seeds.

**Gate:** An independent reviewer can reconstruct what was *actually tested* without confusing a design proposal with a completed run, or a free batch readout with the fly's native answer.

## 5. Workstream A — Can native learning increments accumulate?

**Suggested lead:** Codex. **Source:** V156 revised experiment design.

### A1. Equal-dose interleaved versus clustered training

On two fresh development worlds, retain the same Full151/R_center N_2B/P_SLOW state, feedback rule and output port. Select 16 NEW records in advance. Each receives eight teaching presentations under:

- **INTERLEAVED:** cycle all 16 records for eight passes.
- **CLUSTERED:** eight consecutive repeats of one record before moving to the next.

Both have **128 NEW presentations, identical target content and equal final unique-cue coverage**. Match OLD preparation, teacher information, frozen clocks, feedback budget, native events and compute. Interim states have unequal unique-cue exposure by design, so compare final equal-dose outcomes and report intermediate coverage transparently.

V156's revision estimates **3,072 total presented records** including OLD/control branches and a proposed **1,200 CPU-second cap**; these are *proposed experiment constraints*, not authorization already granted.

Measure complete own answers, all required teacher-prefix margins, self versus cross-record effects, old-answer survival, matched passive changes and order-position effects. Exact-key recitation is E1 only.

**Decision logic:**

- CLUSTERED learns but later conflicting writes undo answers: justify an interference-specific next test.
- Neither learns: investigate native reachable update geometry, signal conditioning and insufficient margin accumulation.
- Both learn taught but not unseen: representation/structural computation becomes the next obstacle.
- Equal outcomes: this precise order hypothesis is unhelpful; do not extrapolate to all local learning.

### A2. Native write-to-read transfer audit (conditional)

Only after a concrete mismatch is identified, derive the signed write's actual effect on future value/read logits, with native timing, fast/slow decay, finite steps, active KC support and cross-context effects. V151's local one-step response is useful E0, but not a full-history gradient theorem. Do not replace it with an arbitrary controller or new gain sweep without evidence.

**Deliverable A:** A measured explanation of accumulation failure, or an explicit unresolved finding.

## 6. Workstream B — Predictive local key learning

**Suggested lead:** Claude. **Source:** V155 learned-keys protocol.

Question: can experience change **PN→KC** keys so that contexts sharing predictive structure develop representations that transfer to novel text with little new data?

Primary comparison families are FROZEN, competitive Hebbian L-COMP, predictive three-factor L-PRED, and external sparse/dense reference models. Nonnegative existing native synapses and top-k readout remain the local candidate's interface; BP references are measurement only.

**Current stop/go:** V155 C1 failed the synthetic test of L-PRED's claimed same-future grouping. A formula-faithful implementation is not enough. Pause C2/C3 for that candidate until a focused explanation distinguishes a poorly aligned objective, threshold/top-k effect, sparse geometry, rule sign, negative sampling or a faulty hypothesis/fixture. Retain the negative result; any changed rule is a new preregistered design, not a posthoc success.

Once qualified, measure:

1. Within-text held prediction versus unchanged FROZEN under identical evaluator/target conventions.
2. Cross-text carry-over: learn keys on A; freeze keys; fit a *new* readout with limited B exposure; score held B and suitable counting controls.
3. KC activity distribution, code stability, learned synapse changes, replay/negative-pair exposure and actual resource use.
4. Separate direct future prediction improvement from corpus-specific memorization and trivial response-frequency effects.

Free ReLU reference networks are **not** proven upper bounds on sparse binary top-k learning; V155's own synthetic checks cautioned against that assumption.

**Deliverable B:** A repeatable local feature-learning and transfer effect—or a narrowed, well-documented failure of a particular mechanism.

## 7. Workstream C — Independent architecture-sufficiency study

**Suggested lead:** Independent architecture agent, reviewed against both the original Full151 and the unsuccessful V6 generic recurrence.

**Question:** Can the system use learned memory to *operate on an internal state*, instead of only associating a prompt with the next output?

Do not assume PN→KC→MBON is sufficient; equally, do not claim that resemblance to the human brain or sheer network depth is necessary. Current MiniFly already contains receptor traces, adaptation, fast/slow stores and autoregressive byte feedback. These do not by themselves prove learned internal control or multi-step program execution.

### C1. Functions to test

- Relevant information survives intervening distractors, while irrelevant information need not.
- A generic learned gate selects when to update or preserve working state.
- Retrieved memory can influence **subsequent hidden-state transitions** before an answer is emitted.
- Learned relations transfer to new contents, compositions and longer step sequences.

These may be tested in bounded diagnostic fixtures, always retaining the final raw-byte/complete-EOS addition target as a distinct end-to-end test. Do not hand-code operand parsing, numerical operators, carry, roles or output length into a candidate.

### C2. Paired architectures

| Condition | Difference | Question |
|---|---|---|
| C0 — native baseline | Current PN/KC/native memory | What can the present implementation acquire? |
| C1 — learned encoder | Plastic features, matched downstream | Are learned features alone sufficient? |
| C2 — capacity reference | Additional size/compute without internal memory-to-computation feedback | Is improvement simply more capacity? |
| C3 — recurrent workspace | Learnable, generic internal state updates | Does active state manipulation matter? |
| C4 — memory-to-workspace loop | Memory read changes future hidden computation, not just logits | Can acquired facts participate in computation? |

Use disabled-feedback and capacity/compute-matched controls, not only a larger model comparison. Revisit **V6's noise-dominated recurrent address behavior**: simply adding an RNN with insufficient input signal has already failed.

### C3. Capacity versus learnability

First establish that a well-qualified constructive solution or reference optimizer can perform the diagnostic function. Then compare the same forward architecture under allowed local learning. The reference is an instrument, never a candidate achievement.

- Reference succeeds, local fails: investigate credit assignment/reachability.
- New architecture transfers under paired controls: evidence of a useful computational function.
- Both fail: verify objective, positive controls and optimization before declaring impossibility.
- Feedback loop is no better than matched no-loop condition: reject that specific architectural addition.

**Deliverable C:** The smallest justified computational capability beyond associative storage, or a precise open question rather than an invented neural mechanism.

## 8. Workstream D — Gated integration

Only integrate mechanisms after the corresponding lane demonstrates its *own* causal effect. Compare unchanged parent, each single change, and their combination on a common fresh task. Keep actual feedback information, world difficulty and computation accounting matched.

Combining a positive feature learner with a positive native writer is a new scientific hypothesis, not automatic evidence of compositional learning. If the integrated model does not improve the target over the stronger component or lacks proper controls, do not promote it for apparent biological richness.

## 9. Workstream E — Reliable acquisition, relevant retention, and transfer

**E1:** Full own-generated taught answers across multiple examples and episodes; do not use loss alone.

**E2:** Retain previously acquired knowledge **when it remains useful** under new learning. Dilution of obsolete facts is not itself failure. Replay, consolidation, availability gating and synaptic protection are conditional responses to demonstrated relevant-loss effects, not default repairs.

**E3:** First responses on never-taught complete relations/orbits, autonomous EOS, causal previous-teaching and recitation/frequency controls, fresh locked worlds and honest sample/compute comparisons. Do not equate a favorable 3/50 development count or correct teacher prefixes with transfer.

Only after generic E1/E2/E3 evidence should a broad language/behavior scaling campaign be interpreted as a test of reusable learned computation.

## 10. Literature and mechanism catalogs — diagnostic use only

The project holds a **44-paper targeted bibliography**, a **150-row human-learning catalog**, and a **254-configuration historical census**. They are not 444 independent tested synaptic mechanisms. Biological evidence for a cognitive function is not proof that a particular MiniFly algorithm implements it.

| Directly observed failure | Mechanisms worth considering after diagnosis |
|---|---|
| Poor predictive KC representations | Competition/Oja/SoftHebb, CLAPP/CLAPP++, perceptual readout reweighting |
| Missing temporally appropriate credit | E-prop, RFLO, three-factor eligibility, feedback control / dendritic predictive coding |
| Beneficial self-writes canceled by other examples | Heterosynaptic competition, context-dependent allocation, gated plasticity |
| Required state cannot be actively maintained or transformed | Temporal context, learned working-memory gating, stateful recurrence |
| Retrieved memories cannot support another computation | Memory-to-state feedback, structure/content separation, relational binding |
| Relevant correct knowledge is truly lost after learning | Selective consolidation, synaptic tagging, replay, metaplasticity cascades |

For every suggested mechanism, specify its local signals, allowed feedback/teacher information, actual state transition, predicted effect and falsifying experiment. Existing negative scaling papers and failed MiniFly versions remain constraints on new claims.

**Do not default to adding protection, replay, new synapses, bigger networks, or all five mechanism families at once.**

## 11. Stop/go, operations and provenance rules

- Scientific question and one causal intervention fixed *before* feedback; explicit resource cap and holdout.
- Failed functional fixture remains a failed claim even if update-formula unit tests pass.
- No silent grid extension, outcome-selected world, output masking, hidden teacher labels, background resumption or automatic new core adoption.
- Record technical interruptions and distinguish hash-verified recovery from repeating a completed trajectory.
- Report gated/unscored outcomes as missing, never zero.
- Preserve raw source anchors and known lack of local checkpoint arrays for independent re-analysis.
- Cross-branch integration gets an independent audit, not a narrative inference.

## 12. Immediate queue and recommended ownership

| Priority | Suggested lane | Next deliverable | Current status |
|---|---|---|---|
| P0 | Evidence steward | Cross-version source/metrics registry without rerunning historical science | Documentation can proceed |
| P1 | **Claude / B** | Investigate V155 C1 functional failure; submit a revised, falsifiable rule **only if justified** | **C1 stopped**; no C2/C3 success |
| P1 | **Codex / A** | Review and separately authorize V156 clustered-versus-interleaved fixed-dose E1 study | Proposed; not executed |
| P1 | **Independent / C** | Write architecture-sufficiency protocol and V6-aware paired controls | New research design; **not dependent** on A/B |
| P2 | Cross-lane review | Compare mechanism effects, decide one justified integration experiment | Needs qualified causal evidence |
| P3 | Integrated E1/E2/E3 | Fresh autonomous acquisition, selective retention and unseen transfer | Requires readiness and new authorization |

## 13. Source anchors

- [V149 main experiment](https://github.com/ChangweiZhou/digitalbaby/blob/67e07f4a649b611f9d5706e48048e25e0c0f64c0/minifly_v149_input_ceiling_dev_20261025/REPORT.md)
- [V153 bit-flow/readout diagnostic](https://github.com/ChangweiZhou/digitalbaby/blob/9026a970575322423e25ecac479fdc1e868b197a/minifly_v153_bits_dev_v1_20261025/REPORT.md)
- [V154 native histories](https://github.com/ChangweiZhou/digitalbaby/blob/8cd3ad76cbf13e51b0379ced9919b583650ca790/minifly_v154_dev_20261025/REPORT.md)
- [V156 diagnostic and limitations](https://github.com/ChangweiZhou/digitalbaby/blob/f32a9bdf71e9035d3a42c2ebf58a425d2cb95082/minifly_v156_dev_20261025/REPORT.md)
- [V156 proposed matched-order study](https://github.com/ChangweiZhou/digitalbaby/blob/f32a9bdf71e9035d3a42c2ebf58a425d2cb95082/minifly_v156_dev_20261025/REVISED_DESIGN.md)
- [V155 latest revised design](https://github.com/ChangweiZhou/digitalbaby/blob/04c2873efb5ca8cc70eacc45052ec9e71231a93f/minifly_v155_keys_design_20261026/DESIGN.md)
- [V155 C1 recorded failure](https://github.com/ChangweiZhou/digitalbaby/blob/04c2873efb5ca8cc70eacc45052ec9e71231a93f/minifly_v155_keys_dev_v1_20261026/results/technical/CHECKS.json)
- [44-paper bibliography](https://github.com/ChangweiZhou/digitalbaby/blob/34ca38ba8a0a50cf0dd60d8804efd84eba2ee582/PROJECT_DOCUMENTS/LOCAL_LEARNING_LITERATURE_20261025/PAPERS.md)
- [150 human-learning mechanisms](https://github.com/ChangweiZhou/digitalbaby/blob/04c2873efb5ca8cc70eacc45052ec9e71231a93f/human_learning_mechanisms_20261023/CATALOG.md)
- [254 configuration historical census](https://github.com/ChangweiZhou/digitalbaby/blob/f32a9bdf71e9035d3a42c2ebf58a425d2cb95082/PROJECT_DOCUMENTS/FULL151_VERSION_CENSUS_20261023/MECHANISMS_254.md)
- [V6 recurrent-controller negative result](https://github.com/ChangweiZhou/digitalbaby/blob/f32a9bdf71e9035d3a42c2ebf58a425d2cb95082/minifly_v6_r1_dev_20261023/REVISION.md)

**Maintainer note:** This document consolidates earlier recommendations, updated for the newly observed V155 C1 blocker. It does not override frozen original protocols, current core pointers, or previously selected experimental baselines.
