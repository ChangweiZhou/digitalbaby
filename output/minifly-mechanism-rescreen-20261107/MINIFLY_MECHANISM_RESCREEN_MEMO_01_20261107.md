# MiniFly mechanism catalog re-screen memo 01

**For Mr Zhou · 7 November 2026 New York time**  
**Repository anchor:** `ChangweiZhou/digitalbaby`, `codex/minifly-consolidated-20261023`, `7ff786619cc9d5833ce0308219dd5eb85cd11449`.  
**Memo number:** 01 in this mechanism re-screen series. This is a research memo, not a new experiment/version.

## 1 Main decision

Several ideas in the catalogs remain worth discussing. The strongest concern **how a learner maintains, selects and transforms a working representation**: separate memory gates, learnable state over multiple timescales, shared recurrent computation, and role/filler binding. Feedback learning, pattern-separated fast memory, and genuine structural plasticity survive as conditional alternatives. The biological displacement accumulator remains an open question on the embodied line.

These are nine discussion topics, with different readiness levels. They do not justify launching nine experiments. **E15 already covers separate gates and the low-decay ring check.** E14 is still running at this review's evidence capture. Their unfinished parts must not be counted as positive results or duplicated.

The historical lists also contain many useful ingredients already represented in current models: error-correcting updates, eligibility traces, replay, sparse coding and association memory. Their presence supports integration, rather than a claim that another named molecule must be added.

## 2 What was covered

The [searchable appendix](mechanism-review.html) preserves **577 overlapping source rows**, with an entry ID, original text/evidence, a computational-family assignment, disposition and reason. The [full JSON](MECHANISM_RESCREEN.json) retains the original historical records. **577 is a source-row count, not a count of distinct mechanisms.** The 31 review families organize relevance; they do not assert that every variant within a family implements the same algorithm.

| Available source | Rows accounted for | Interpretation |
| --- | ---: | --- |
| Deep Research Backing Conversation | 11 | Circuit descriptions |
| Biological Circuit Mechanisms Literature Survey | 11 | Another rendition of the same eleven broad circuits |
| Executive Summary | 11 | A third rendition of those circuits, with different size estimates |
| Fly Brain Learning Mechanisms | 42 | Detailed molecular, circuit and systems matrix |
| Fly Evidence Map | 20 + 10 | Twenty ranked entries and ten family summaries |
| Neuronal Human Learning Mechanisms Research | 12 | Synaptic, structural and excitability variants |
| Human-Learning Perspective audit | 14 | Mostly instructional/scheduling interventions |
| September pruned handoff | 56 | Original 31-entry matrix plus overlapping 25-entry list |
| Historical mechanism catalog | 136 | Named variants, proposals and controls |
| Historical experiment inventory | 254 | Experimental recipes, including parameter variants and controls |
| **Total** | **577** | **Overlapping entries, all retained separately** |

**One source remains incomplete.** The Fly Evidence Map says its companion report contains 70 mechanisms, but links that report to `sandbox:/mnt/data/fly_brain_learning_mechanisms_deep_research.md`. The full companion file was not recovered from the available documents, Downloads or repository index. Its available twenty ranked entries and ten summaries were reviewed; **the missing complete 70-entry list was not**. No conclusion here treats those ten summaries as a reconstruction of all seventy entries.

This was a source/receipt review with selected primary-paper checks. It did not rerun the 254 historical recipes, independently verify every biological citation, or provide a cross-family replication. Prior exact results remain unchanged. Decisions concern current computational relevance, with variant-specific qualifications preserved.

## 3 How the new filtering differs

Each entry was considered for the capability it might add, its actual information and learning requirements, its relationship to existing code, and the observation that would make it worth testing. A biologically familiar name is insufficient. A previous failed recipe also does not exclude a whole mechanism family.

The current reference is the E45/E7/E8/E11 lineage and its later results, rather than an unexecuted historical tiny-model design. Development optimization is permitted in this lineage; lifetime local learning is a separate claim. Attention remains available as the user-authorized baseline. A content-addressed read is identified as such even when its implementation is called a biological memory.

The review distinguishes four different things that earlier catalogs sometimes combine: **a memory store, a computation over state, a learning signal, and an experience schedule**. Improving one can help the system without establishing the other three.

## 4 Disposition of all eleven broad circuits

| Circuit family | Current evidence and limitation | Disposition |
| --- | --- | --- |
| Hippocampus and neocortex | Replay-supported retention works in the current learner. That does not establish a learned pattern-separated episodic memory or pattern completion. | Keep replay reference; discuss a learned fast store when its capacity/cost becomes limiting. |
| PFC and basal-ganglia memory gating | The second sweep favors separate read/write gates, with an active-capacity confound. E15 C tests a split-gate cell in the operative learner. | Active work; read E15 before commissioning another check. |
| Cortical feedback for learning | Sign feedback already supports substantial learned generalization. Adaptive apical feedback did not beat fixed feedback on the second sweep's classification proxy. | Conditional alternative; no present evidence that learned feedback is mandatory. |
| Cerebellar learning | The second sweep's feature-basis comparison is positive, but its inputs are independent 2D points with a sinusoidal target. It is not a sequence-timing experiment. | Retain the regression result; reserve true timing/calibration for a relevant need. |
| Basal-ganglia actor-critic | Eligibility helps delayed binary actions and a four-action task. The latter has independent cue-action mappings and can repeat training trajectory combinations. | Useful future agency component; not evidence of novel planning or arithmetic composition. |
| Grid and head-direction attractors | The graph proxy composes displacements using supplied toroidal phase transport. E10 arithmetic uses supplied ring operations and positional parsing. | Geometry/embodiment reference; do not count supplied arithmetic as learned arithmetic. |
| Insect central complex | The fitted projection subset closes heading and supplied-goal steering. The displacement/return package fails even with the corrected heading clamp. | Keep heading/steering result; park that PI package. A justified accumulator remains open. |
| Mushroom-body output loop | The second proxy supports addressed reversal over permuted addressing; APL-on/off tie. Context addressing is supplied, and the proxy is not the extracted circuit. | Memory/revision reserve; no general reasoning claim. |
| Sleep-dependent consolidation | Replay is useful. Extra sleep ordering/scaling did not add an advantage in the second proxy. Lower optimizer decay helps E13's current learner. | Keep observed-experience replay; no new sleep organ or blanket downscaling prescription. |
| Thalamocortical recurrence | Current sweep comparisons do not establish the benefit of internal iteration. The second fixture confounds the two operator positions by teaching only same-operator compositions. | Computational idea survives; not an immediate duplicate of the failed proxy. |
| Attention/routing | A routing reference is permitted. The operator-binding sweep's exact prompt rewrites fail even with a perfect isolated executor. | Keep reference; separate copying, binding and execution before broad conclusions. |

The source circuit descriptions and their distinct variants remain in appendix groups `CIRCUIT_A`, `CIRCUIT_B` and `EXECUTIVE11`. This table combines their current relevance, not their biological certainty grades.

## 5 Nine topics worth later discussion

### 5.1 Separate memory write maintain and read decisions

This is the most immediate structural candidate because it can control what enters state and what influences the answer independently. E15 already tests it. Any improvement should be described as the split-gate architecture package until extra active parameters and development differences are separated. A gated model is not automatically a demonstrated PFC-BG circuit. The original [PBWM model](https://pubmed.ncbi.nlm.nih.gov/16378516/) supports this as a computational hypothesis.

### 5.2 Learnable state over multiple timescales

The present cell still receives a fixed ten-byte window. A later question is whether a learned state can replace that input aid while retaining rule acquisition, holding and old knowledge. Positive diagonal-state-space results motivate the candidate, but do not establish that arbitrary logarithmically spaced decays are optimal; [S4D specifically finds initialization important](https://arxiv.org/abs/2206.11893). Oscillatory state and a biological attractor also require separate justification.

### 5.3 Shared recurrent computation

A learned transformation applied repeatedly could support deeper computation without encoding the arithmetic in the wiring. The comparison must separate useful recurrence from simply spending more operations. Thalamic physiology supports maintaining/amplifying task representations; it does not establish a generic multi-step symbolic algorithm. [Schmitt et al.](https://www.nature.com/articles/nature22073) is a relevant original experiment, not a proof that three simulated passes will solve MiniFly composition.

### 5.4 Learned role and filler binding

Operator selection, operand copying, first-step execution and use of the intermediate answer are distinct possible failures. The second routing sweep cannot establish that routing in general is impossible. Conversely, E7's balanced learned compositions do not establish general binding across new formats and symbols. This topic becomes actionable when a specific remaining error can be attributed to one of these stages. Attention is an informative reference, with its information access stated plainly.

### 5.5 A learned pattern-separated store beside the shared path

This survives as a more specific proposal than “add fast/slow learning.” It could preserve episodes without destroying the shared representations that support transfer. Its relevant comparison is bounded learned storage versus the current raw/observed replay reference, charging storage, addressing and teaching information. Retrieval should not silently acquire privileged task IDs or target answers. [Recall-gated consolidation](https://elifesciences.org/articles/90793) is a theoretical candidate for noisy consolidation, not a demonstrated cure for MiniFly; a writing-less control remains necessary.

### 5.6 Alternative local feedback signals

Learned feedback, compartmental targets and sign feedback remain distinct options. [Segregated-dendrite learning](https://elifesciences.org/articles/22901) and [weight mirrors/Kolen-Pollack variants](https://arxiv.org/abs/1904.05391) provide concrete algorithms to inspect. Current partial E14 diagnostics do not support assuming universally poor raw gradient alignment. Average alignment also cannot rule out a problem on the specific operations that fail. Add such a mechanism only for a measured remaining learning deficit, against the calibrated current learner.

### 5.7 Longer eligibility and selective consolidation

Behavioral-timescale traces and tag/capture mechanisms remain relevant if feedback arrives beyond the current credit horizon. They should address that temporal problem rather than be promoted as reasoning mechanisms. The [original BTSP experiment](https://pubmed.ncbi.nlm.nih.gov/28883072/) concerns hippocampal CA1 place fields, not proof that a human concept-cell observation provides a generic one-shot learning rule. A biological tag also does not specify which old knowledge is safe to overwrite.

### 5.8 Genuine structural plasticity

The historical lists contain hierarchical modular graphs, sparse bridges, expansion, growth and stabilization. These were not exhausted by testing changes to support masks or routing weights. They remain scientifically open. Their practical trigger should be a measured computational capacity/interference limit, with fixed versus plastic adjacency compared at matched active edges and work. More memorization capacity alone would not answer the reasoning question.

### 5.9 An anatomically justified displacement accumulator

This is the most consequential biology-specific distinction found in the re-screen. [Lu et al.](https://www.nature.com/articles/s41586-021-04191-x) identify PFNd/PFNv integration of heading and body velocity and hDeltaB encoding of **world-centric velocity**; temporal integration into a path memory is described as a subsequent computation. The [Hulse connectome paper](https://elifesciences.org/articles/66039) explicitly discusses several possible storage sites and states the connectome can rule out some implementations, rather than uniquely determine the home-vector store.

The catalogs' stronger claim that a compact phase-shift circuit automatically supplies validated displacement memory or a known goal-memory delta rule is therefore not established by those sources. Our failed return panel is consistent with a problem in the encoding/integration/readout package, but does not uniquely diagnose one component. Keep the prescribed stop; discuss a new accumulator only when its site and dynamics have evidence.

## 6 What stays off the immediate queue

Molecular refinements such as CREB, Orb2, mTOR and ERK remain mapped to their distinct computational interpretations—tags, resources, persistence and local updates. They should not become separate software modules merely because their biological names differ. APL/homeostasis, uncertainty protection, selective forgetting, latent retrieval, growth and excitability remain conditional reserves. Their exact historical variants are preserved, with no blanket scientific rejection.

Circadian, metabolic, stress and age gates are low current priority for learned reasoning. Gap junctions and Arc-like communication need a demonstrated communication problem. Human instructional recommendations—retrieval quizzes, scaffolding, metacognition, gamification and social learning—cannot be imported as though they were local neural algorithms. Some motivate controlled exposure schedules; others concern a human learning platform rather than this project.

Known trace and replay positives stay available. No new run is proposed solely to repair an old unqualified sweep block. The user's earlier direction to concentrate on genuinely untested mechanisms is preserved.

## 7 Corrections that should accompany future use of the catalogs

1. **Count the implemented learner.** A handful of role-pair gains can parameterize a large fixed circuit. They do not count all neurons, fixed arrays, individually learned synapses, adapters, recurrent state, optimizer state and replay storage. The source renditions disagree substantially on size; neither is a measured census of an operative learner.
2. **Separate anatomy from a validated learning equation.** A projection or dopamine receptor does not establish a unique local update formula, a plasticity mask, or sufficiency for autonomous behavior. The particular fan-shaped-body goal delta rule in the survey is not licensed by anatomy alone.
3. **Separate velocity transformation from displacement integration.** The relevant primary papers make this distinction; the catalogs and earlier experiment language sometimes blur it.
4. **Narrow proxy claims to the implemented task.** The second cerebellar task is static nonlinear regression. The grid task supplies geometry. The goal-memory task supplies context addresses. The multi-action task supports delay credit, not untaught planning.
5. **Do not equate exact diagonal traces with exact whole-network learning.** Their exactness depends on the cell's independence and the learning signal used. Sign feedback and cross-layer/state coupling are additional issues. E11 also retains a fixed input window.
6. **Keep statistical averages from concealing a failed computation.** Completed E14 fresh-seed E7 low-decay composition is 88.7-90.0%, around 2-3 points below matched BP. In E8, one fresh seed retains 94.54% of the eight operations but reaches only 18.67% on the `+-` pair, versus BP 83.33%; its intermediate answer is 19.83%. Aggregate held composition of 72.04% masks that failure. These are completed Part A results, not a verdict on unfinished E14 diagnostics.

## 8 Recommended order

**First, complete E14 and read E15.** Use them to fix the reference settings and determine whether mixed-stream acquisition, split gates and interface preservation actually improve. Do not assume the partial diagnosis or prospective predictions have passed.

**Then choose one capability-limiting change.** If the weakest pair still fails, use its step-level errors and one relevant matched contrast to choose between binding, state and learning-credit explanations. If the mixed-stream cell works, a window-replacement comparison becomes more useful. Recurrent depth, learned feedback, a new memory store and structural rewiring are alternatives for specific later failure signatures, not additions to install together.

**Continue measured scaling alongside this progression.** Once the unified model can perform the relevant computation, distinguish width, state capacity and exposure. Track weakest-operation performance, retention and compute. The catalogs provide candidates for a diagnosed limit; they do not replace this measurement.

The desired milestone is one operative learner that acquires new rules, preserves useful earlier rules and composes them after further learning, with task-specific execution supplied by the evaluator kept out of the learned capability claim.

## 9 Evidence files and remaining limit

The second sweep's 108 receipts were checked for unique block/arm/seed identities, shared within-seed panels, lock hashes and notebook-source parity. The first sweep contains 99 unique completed screen identities across eleven blocks, with 33 matched seed panels and one bundle ID. These are receipt integrity checks, not new executions or proof of every mechanism interpretation.

Current evidence includes the [pinned E45 report](https://github.com/ChangweiZhou/digitalbaby/blob/7ff786619cc9d5833ce0308219dd5eb85cd11449/MOD97_LOCAL_RULE_20261106/E45_REPORT.md), [E15 predictions and scope](https://github.com/ChangweiZhou/digitalbaby/blob/7ff786619cc9d5833ce0308219dd5eb85cd11449/MOD97_LOCAL_RULE_20261106/E45_PREDICTIONS.md), [second-sweep report](../minifly-mechanism-gaps/SCREEN_REPORT.md), its embedded notebook source and raw receipt archive, E9/E12/E13 results, completed E14 Part A receipts, and the corrected PI follow-up. The review snapshots the documents and preserves their hashes in [SOURCE_MANIFEST.json](SOURCE_MANIFEST.json).

The [full entry appendix](mechanism-review.html) can be filtered by original source, computational family and disposition. It includes every available row even when several map to one family. The unresolved source-coverage item is recovery of the full seventy-entry companion catalog. No scientific experiment was launched for this re-screen.
