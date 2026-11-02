# MiniFly V179 — Continuous-Life Local Learning, Revision and Reuse

**Identity:** V179-CLLR-D1  
**Date:** 2026-11-02  
**State:** DESIGN_ONLY — no implementation, model training, confirmation, adoption, or main-world evaluation authorized by publishing this proposal  
**Parent evidence:** Codex @ `885d47ae2b24e0738f0159a6c78a1ab5026ee509` (V178 completed); original Claude V167/V169 at `f4146856c3241d94f7fa948e0b7d30fa0b869bc1`  
**Single question:** When an already demonstrated **fully local predictive learner** receives A, then B, then corrected A feedback, does the *same continuing model* acquire, retain, revise and reuse useful predictive knowledge more effectively than appropriate controls, per observation and CPU?

## 0. Why V179 is not V175–V178 again

V175 changed a KC inhibition mechanism, V176 added a learned-feature-to-native bridge, V177 changed recurrent dynamics and output all at once, and V178 stopped at an intermediate sparse BPTT failure that was **not a logical prerequisite** for success of a gated circuit. The serial mechanism/architecture pipeline repeatedly substituted component proxy scores for the intended learner's lifetime capability.

**Program correction, enacted by this specification:**

1. **One operative learner first.** The proposed main learner descends from the existing V167 fully local KC→H512→T2048→softmax predictor. No Transformer, attention, content-addressed KV read, spiking neuron, hand-coded arithmetic feature, Full151 value writer, or newly invented plasticity mechanism is introduced.
2. **One continuous parameter history.** Main parameters *and* AdamW moments/step count survive A→B→A-revision. No fresh native bank, source-head discard, best-calibration checkpoint restoration, optimizer reset, or evaluator-fitted decoder is allowed in the main trajectory.
3. **Four capabilities in the same life.** Evaluate the learner's own predictions for acquisition, still-valid retention, obsolete-knowledge revision, and B learning/reuse, not separate positive probes.
4. **No unproved serial efficacy gates.** Technical/source/data integrity can stop execution. Poor accuracy during A or B **does not block** the registered B/C controls or cause a new rescue variant. Report the full lifetime's negative result.
5. **Known differences are disclosed rather than hidden.** V167/V169 dynamically compressed labels into A-specific seen symbols plus OTHER, used shuffled-batch presentation, and V169 restarted AdamW at the B boundary. V179 uses a fixed **257-class** output, actual byte/EOS teacher sites, fixed chronological record-order minibatches, and continuing optimizer state. These are **necessary interface/process adaptations** and not an exact numerical replication of V169. The underlying local weight-update algebra, KC graph, H/T widths and generic input byte meanings remain the same.
6. **No false E2/E3 promotion.** Natural-text next-byte retention and edited-token adaptation are predictive-lifetime *proxies*. They do not by themselves establish retained factual knowledge, complete autonomous byte+EOS answers, never-taught addition or reasoning, nor a biologically validated plasticity rule.

Do not allocate further numerical versions just because a prespecified capability threshold is missed. A successor requires a distinct mechanism hypothesis tied to a diagnosed, measured lifetime failure—not automatic tuning of V179.

## 1. Source-locked fully local candidate: what actually learns

The operative V167 local model, with the output class count expanded to 257, is:

\[
x_t=\mathrm{FE\!-\!PERM\!-\!KC}(\mathrm{observed\ prefix}_{\le t}),\quad
a^H_t=W_H^\top x_t+b_H,\quad h_t=\operatorname{ReLU}(a^H_t),
\]
\[
a^T_t=W_T h_t+b_T,\quad
z^T_t=\operatorname{dropout}_{0.5}(\operatorname{ReLU}(a^T_t)),\quad
q_t=W_Oz^T_t+b_O,\quad
p_t=\operatorname{softmax}_{257}(q_t).
\]

Here x is the existing ~5,177-unit sparse **fixed** native KC sensory code, H is 512 units, T is 2,048 units and the operative learned head predicts the next *actual* byte 0–255 or record-end EOS=256. The class dictionary is fixed globally at birth, never changed at B or at revision. There is no OTHER class, external answer head, label-dependent class dropout, forced EOS, query into a content-addressed table, positional/operand/role feature, or word-level pretrained model. The input code may cache a pure sensory function but **cannot cache a learned feature** across changes to its weights.

**Local gradient algebra for batch size n:**

\[
e_t=(p_t-\mathrm{onehot}_{257}(y_t))/n,
\]
\[
\delta^T_t=(e_tF_T)\odot \mathbf1_{a^T_t>0}\odot m_t,\qquad
\delta^H_t=(e_tF_H)\odot \mathbf1_{a^H_t>0}.
\]

The readout receives the ordinary output-local delta gradient (`\partial\ell/\partial W_O=z_t^{T\top}e_t`); `W_T` and `W_H` receive presynaptic activity times their own `\delta`, **not** gradients backpropagated through the other learned layer. `F_H∈R^{257×512}` and `F_T∈R^{257×2048}` are born independently with the original named seeds 21 and 22 and fixed Gaussian variance `1/257` per element; they never train. `m_t` is the ordinary V167 dropout mask including its inverted-dropout scaling. Preserve V167's exact layer-local chain handling and bias updates; implementing it with explicitly multiplied gradients is allowed only after numerical and optimizer parity.

**Frozen optimizer:** AdamW learning rate 0.001, weight decay 0.0001, beta1=.9, beta2=.999, eps=1e-8, original V167 initialized weights/biases, dropout 0.5, batch 256 target sites. No added warmup, replay, class bias calibration sweep, global BPTT, hidden learned target parser or hard-coded local metaplasticity. The main learner preserves weights, optimizer moments, optimizer step counter, fixed feedback, dropout RNG and input-encoding convention across stage transitions. Technical controls may reset optimizer state **only where that is their declared intervention**.

**Predict-before-target:** form the KC code at each physical real next-byte site using only bytes already presented, compute `p_t`, then release the actual arriving next byte or genuine EOS as feedback. FE-PERM's input is not allowed to peek at the next target; labels enter the learning update only after prediction. When EOS is observed, the *sensory transient state* resets for a new real record; the model's learned weights/optimizer are never reset. Distinguish this deterministic minibatch-based predictive learner from a strictly single-event on-chip online spiking system: it is a persistent-parameter, locally updated learner, not claimed to be a fully asynchronous animal.

## 2. Two previously opened development ecologies (no confirmation)

Use authenticated raw source editions and preprocessing from V176's `DATA_AUDIT.json`, which cross-checks the earlier V167/V169 export for Emma/Ball. The first pair is Thursday→Edgeworth; the second Emma→Ball.

| Development world | A (3 presentations) | B (3 presentations per B branch) |
|---|---|---|
| W1 | chesterton-thursday.txt; 705 train records; 41,017 next-byte/EOS targets per presentation | edgeworth-parents.txt; 156 train records; 10,287 targets per presentation |
| W2 | austen-emma.txt; 657 train records; 41,019 next-byte/EOS targets per presentation | chesterton-ball.txt; 170 train records; 10,274 targets per presentation |

Only authentic existing files with the published SHA256 and original Gutenberg footer exclusion are eligible. Do not download/substitute editions, silently remove lines or select different authors because of observed outcomes. Each record is an actual nonempty corpus line, at most 128 bytes under the source's original splitting rules, with **one genuine EOS at its end**; the learner is trained at all byte/EOS positions in that record. Never append an extra EOS after a sampled individual byte. The immutable FE-PERM/KC input is precomputed only as a cache of that sensory stream; when the stream changes, compute the sensory codes again.

All four corpora and prior benchmarks have been examined in earlier work. These are **engineering/development fixtures**, not fresh independent confirmation worlds. Reserved V179 stream names (`V179|W1|A`, etc.) determine frozen minibatch order and control RNGs but do not grant a hidden task or phase ID to the model. Within each phase, actual records remain in chronological order; target minibatches contain successive target sites in that order, carrying a partial last batch. Each epoch is explicitly **repeated exposure**. Use fixed phase-specific 3×A, 3×B and 2×C presentations, with no validation-best checkpoint reload or outcome-triggered extension.

**Held panels:** score both A and B held texts with the *same model* and the same globally fixed 257-way vocabulary at birth, A-before-B, after each B pass, B-complete, before C and after each C pass. The expected byte and EOS are revealed only to the evaluator. Score the model's actual softmax; log full-class bits/token, top-1 and separate EOS/payload metrics. Do not substitute a fitted B readout, correct-prefix oracle or output repair. Whole-record autonomous continuations (first 3 observed bytes then up to 32 model-generated bytes/EOS) are fixed **secondary** diagnostics with complete byte/EOS correctness, not a success proxy for a text-prediction task.

## 3. One lifetime, three causal controls

### 3.1 Shared A acquisition

Train one source-locked local learner on A for exactly **3 complete passes**, preserving every optimizer state. Save an exact deep checkpoint, the original **untrained birth** and all named RNG states. The common A learning history is executed **once per world** and physically copied only at the prespecified A/B fork. Future B and C forks are independent lives after that split; no mutable state or optimizer moments are shared.

Report A held loss at birth and after all three passes; the primary model's A history remains part of its real information and computation cost. Output categories are never chosen using A labels.

### 3.2 Three B learning lives (same observed B)

- **KEEP (primary).** Continue the learned A model on B for 3 passes with **all weights, AdamW moments, step count, fixed feedback and RNG state carried forward**. This is the operational persistent learner.
- **RESET-OPT (causal control).** Clone the *identical A-complete state*, then zero only AdamW moments and optimizer step count before the **same B presentations**. All neural weights, input/readout heads and fixed feedback remain identical at the fork. This isolates the optimizer-carry intervention conditional on the A history; the two arms share no mutable state after the fork.
- **B-COLD (reuse counterfactual).** Start from the same original model birth, observe **B only** for the same 3 passes with the same byte/EOS supervision, fixed vocabulary, optimizer recipe and frozen B order. This is an evaluator-controlled alternate life; it is not described as a persistent A→B learner.

All three B arms must run even if KEEP initially performs poorly. KEEP vs RESET-OPT estimates the incremental effect of carrying optimizer state; RESET-OPT vs B-COLD gives the clearest available development contrast for previously learned A parameters under matched B optimizer initialization. KEEP vs B-COLD is a practical but broader combined pretraining/optimizer-history effect, not a pure representation contrast.

Score B held after B passes 1, 2, 3, and *zero-shot before B*, using the same frozen held examples. Report A held simultaneously to detect negative or positive transfer. No extra replay of A during B, no phase marker, no memory store reset, no best-of-three choice.

### 3.3 Deterministic, trainer-only A revision: C

After KEEP's third B pass, fork the **exact same KEEP checkpoint** into:

- **EDITED-C:** exactly two passes on 64 selected A train records with controlled, deterministic changes to a small set of previously predominant next-byte targets;
- **NOEDIT-C:** exactly two passes on the **same 64 A train record IDs and identical record/minibatch order**, but with their original unchanged text and corresponding feedback.

This is an outcome-correction intervention: both forks have the same number of records, byte sites, optimizer updates, starting parameters and optimizer state. Only the raw data and legitimate next-target feedback corresponding to the edits differ. EDITED-C is the primary continuous A→B→revised-A history; NOEDIT-C is a same-exposure old-A rehearsal counterfactual, **not** the claim that replay is a proven consolidation mechanism.

### 3.4 How the corrected environment is generated (no semantic hint)

Before **any V179 learner training or scored model evaluation**, using only the raw A-TRAIN source:

1. Enumerate pairs of preceding *actual payload bytes* `k=(b_{t-2},b_{t-1})` and the following payload byte `y_t`, never across record boundaries; ignore EOS and newline as possible original target for this edit selector.
2. A pair is eligible only if observed ≥32 times in A train and its most frequent next byte represents ≥60% of its observations. Rank by total count descending, with SHA256(`V179|world|edit|key`) breaking ties; select exactly **four** distinct eligible pairs. A pair's modal old byte is determined only from A train. If fewer than four qualify, freeze `INVALID_CORRECTION_FIXTURE` rather than quietly changing the threshold or task.
3. Select a different replacement byte for each pair by a fixed SHA256 ordering over the **16 most frequent ordinary A-train bytes**, excluding the modal old value, newline and EOS. No held target influences this selection. Replacement bytes are ordinary legal 0–255 payloads, not artificial class markers or privileged labels.
4. On every occurrence in each raw A record where the two preceding **original unmodified** bytes equal a selected pair and the original target equals the selected old modal value, replace that target with the chosen new byte. Select all positions against the *original* record to avoid iterative cascading substitutions. Do not alter record lengths or the location/count of EOS; recompute all KC codes from the actual changed bytes.
5. Rank the **A-train record IDs containing ≥1 actual edit** by SHA256(`V179|world|revision-record|ID`). Select the first 64. The same original IDs are used in both EDITED-C and NOEDIT-C; otherwise mark the fixture invalid.
6. Generate corrected A-CALIB and A-HELD from the same fixed train-derived transform with no additional key selection. Record the exact total edited held targets and unaffected evaluator sites *before opening any model scores*. A fixture with fewer than **24 genuinely changed held targets** or **128 completely unchanged evaluator byte sites** is flagged `INSUFFICIENT_CORRECTION_COVERAGE` and cannot support a correction claim; do not replace it. Complete the A/B histories even if correction coverage fails; record the coverage limitation rather than rescuing the world.

The evaluator alone knows whether an expected byte has changed and which outcomes remain valid. No learner ever receives old/new IDs, author names, a correction flag, a task-phase embedding, or an externally computed key function. The environment generates ordinary raw text bytes and natural one-per-record EOS; the edit script is **a controlled synthetic environmental change**, not evidence that the learner understands English facts, correction semantics or arithmetic.

## 4. Four primary outcomes and accompanying controls

All losses below are **bits per actual target** from the learner's own unmasked 257-way softmax. Score identical final-checkpoint and prospective-state prediction sites; an online minibatch-average loss cannot be compared to held loss as evidence of overfitting.

### 4.1 A acquisition

Let `L_A^{birth}` and `L_A^{afterA}` be the A held next-byte loss before and after A. Compare to a fixed train-fitted unigram and a predeclared PPM/Kneser–Ney historical reference, both clearly identified as *non-neural evaluator models*.

**Development acquisition screen:** `L_A^{birth}-L_A^{afterA}≥0.20` bits and `L_A^{afterA}≤L_{A,UNI}-0.05` in each world. Failing this does not block B/C: the full life continues, and the final result is labeled under-acquired rather than rescued by a new encoder.

### 4.2 Reuse on B

For the same B held sites after equal B exposure `n`, compare:

\[
G_B(n)=L_{B,\mathrm{B\!-\!COLD}}(n)-L_{B,\mathrm{RESET\!-\!OPT}}(n).
\]

Also report the ordinary continuing KEEP versus both control trajectories, and zero-shot B prediction at the A boundary. Positive `G_B` is operational sample-efficiency reuse of A-trained parameters under an approximately matched B optimizer clock; it is **not** proof of an abstract learned relation, and word-frequency transfer remains a possible explanation.

**Reuse screen:** `G_B(1\ \mathrm{pass})≥0.05` bits and `G_B(3\ \mathrm{passes})≥0.05` on **both** opened development worlds, with KEEP performance separately reported. Report end-to-end training CPU and source A exposures inherited by the pretrained model; do not call prior learning free.

### 4.3 Retention of still-valid predictions

At the A boundary, B boundary and C completion, score the frozen A held corpus, but split evaluator sites into:

- **STABLE:** original A-HELD sites whose expected byte is unchanged by the train-derived edit transform and whose immediately preceding **16 observed bytes** are unchanged; no change of label or recent local context is involved. This is a bounded *still-valid predictive statistic*, not a proof of semantic relevance.
- **OBSOLETE:** A-HELD sites on which the deterministic edit rule would change the output byte; old-label accuracy there is **not a retention goal**.

Define `R_A=L_{A,stable}^{afterB}-L_{A,stable}^{afterA}` and `R'_A=L_{A,stable}^{afterC}-L_{A,stable}^{afterA}`; report both, including signed improvements. Also measure `R_B=L_{B,held}^{afterC}-L_{B,held}^{afterB}`.

**Retention screen:** each `R_A≤0.10` bits, `R'_A≤0.15` bits and `R_B≤0.10` bits. If A actually improves during B (as in historical V169), report **no observed forgetting**; do not claim replay or consolidation was needed or cured it. If stable knowledge worsens, separately report how many sites degraded, not merely the mean.

### 4.4 Revision of obsolete predictions

At each **changed A'-HELD target**, use the true corrected prefix and read the probabilities assigned to `y_{new}` and `y_{old}` by the same model. Define the updated-outcome margin

\[
m_t = \log_2 p_t(y_{new})-\log_2 p_t(y_{old}).
\]

At the same C-start checkpoint, both EDITED-C and NOEDIT-C are identical. At C completion, measure the *feedback-specific* revision contrast:

\[
D_{\rm margin}=
(\overline m_{\rm EDITED,afterC}-\overline m_{\rm beforeC})
-
(\overline m_{\rm NOEDIT,afterC}-\overline m_{\rm beforeC}).
\]

Also score `D_{\rm loss}=L_{\rm NOEDIT}(A'_{\rm changed})-L_{\rm EDITED}(A'_{\rm changed})`: higher is favorable, and measure changed-site top-1 accuracy, unchanged A stability and B loss in the same model.

**Revision screen:** `D_{\rm margin}≥0.20` bits *and* `D_{\rm loss}≥0.10` bits in both development worlds, with no release of old labels to the learner as a privileged punishment or task-specific gate. If the change merely memorizes the 64 revised lines but fails on corrected A' held sites, mark revision nontransfer; exact-value edit lookup remains a countermodel.

### 4.5 Success status, not a serial gating cascade

**PROMISING_LIFETIME_DEV** requires acquisition, reuse, retention and revision screens on **both** worlds, plus technical/resource validity. Otherwise record `LIFETIME_DEV_GAPS` and the signed failures independently; do not overwrite one outcome with a gate that prevents testing the others. Substantial evidence for one component may be retained as *bounded evidence* without promoting an adopted core.

A previous success is not inferred from any fitted H probe, class-specific numerical oracle, a counting model, or the V164/V178 BPTT references. These results test the **local learner's own predictions**. Even a PROMISING result on two previously opened literary-text pairs is **development evidence only**; no independent confirmation, E2 general retention or E3 never-taught addition follows.

## 5. Execution controls preventing the prior research loop

### C0: source/data/evaluation integrity; no efficacy-based selection

- Verify latest version and worlds, original V167/V169 local weight-update equations and original FE-PERM reset/clock semantics, **257-class** output, no OTHER mapping, full input/output labels and authentic corpus hashes. Rebuild A, B and edited A' sparse sensory codes from one pinned canonical birth; compare every unchanged A/B code to authenticated V167/V176 exporters before training.
- Verify 256 raw payload symbols plus EOS256 are legal, future targets never enter FE/learner before prediction, no artificial one-byte EOS, and corrected records are re-encoded from their changed observed bytes.
- Freeze all four edit keys, old/new mappings, 64 revised record IDs, held-correction coverage metadata, name-spaced control RNGs and the exact repeated-record orders before **opening any learner score**. Held target content cannot determine which edits are chosen. These are source/data qualification checks, not performance gates.
- Confirm fixed feedback `F_H,F_T` and output gradient math against V167 source and an independent finite-difference unit fixture. In the technical fixture with hidden feedback matrices zero, only the output readout receives error-related learning (with explicitly accounted weight decay).
- Verify exact persistence: KEEP's B-first update is calculated from the genuine A AdamW moments; RESET-OPT preserves parameter tensors but has zero moments; NOEDIT/EDIT start with bit-identical params/moments/RNG; COLD starts from original untrained birth. All clone states independent after forks.
- Use *the same* 257-way evaluator on birth/A/B/C states, with no batch-size, autoregressive EOS or phase-dependent decoder switch.

### C1: resource admission and baseline correctness

Before any scored A/B model fitting: run at most **32** separately charged technical full records, including a complete sample of A/B/edited target flows and a deterministic interrupt→restore→next-step test. Source-lock the runner, authenticated dataset, fixed data edits, full optimizer, feedback matrices, output vocabulary, dependency versions and result reducer. Benchmark all physical branches and full fixed panels with ×1.5 resource reserve; mark `RESOURCE_NOT_ADMITTED` if the complete conditional program does not fit.

Do **not** require local learner success on a toy data fixture before running the main lifetime; the validity gate is source, numerical and data integrity, not an invented ability threshold. The model may be weak—that is an admissible negative scientific result.

### C2: all registered lifetime branches

From birth, run the A shared history and its B+C forks, B-COLD and RESET-OPT on both W1/W2 **even if an earlier learning score is disappointing**. The only acceptable stops are source/data invalidity, nonfinite/incorrect update, insufficient declared correction coverage (report B-only as partial without inventing C), resource breach or missing authenticated assets. Do not select more learning passes or another stream after observing A/B outcomes.

At every registered checkpoint record raw 257-class prediction-score summaries, total CPU/wall/RSS, unique observed records, repeated record applications, token target applications, optimizer steps, phase boundary checkpoint hashes and active optimizer moments, per-branch retained/revised target subsets, and generation diagnostics. Preserve old negative results. No BPTT or learnt external decoder is allowed to replace the operative learner.

### C3: independent saved-only reduction, no hidden reruns

A saved-only reducer checks the signed contrasts, denominators, exposure counts, proper class IDs, exact historical source hashes, fork equality, source-only data edits, checkpoint state, and fixed screens using the **saved raw output panels** without retraining. If raw panels are not published under the project's raw-data restrictions, publish precise manifest hashes and state explicitly that an external raw-data reconstruction has **not** been performed.

No p-value/bootstrapping across thousands of correlated token positions is allowed as evidence of independent-world generalization. Report per-record paired effects and both world-level effects; with two already-opened worlds the statistical status remains development only.

## 6. Fixed exposure and cost ledger

Historical authenticated line counts determine the 257-class learning dose:

| Physical learner work | Full-record presentations across W1+W2 |
|---|---:|
| Shared A baseline: 3×(705+657) | **4,086** |
| B KEEP: 3×(156+170) | **978** |
| B RESET-OPT: identical B dose | **978** |
| B-COLD: identical B dose | **978** |
| C EDITED: 2×64×2 worlds | **256** |
| C NOEDIT: same dosage | **256** |
| **Scheduled training total** | **7,532** |
| Technical qualification maximum | **32** |
| **Hard complete-record ceiling** | **7,600** |

All B branches share the same observed B data but count as separate physical training presentations; A is physically trained only once and all A descendants inherit its full cost. C branches share the same B KEEP parent but their post-fork presentations are separately charged. All models and counting references are identified by branch/seed in the ledger.

Maximum target-application forecast from authenticated original source sizes and 128-byte line cap: A `3×(41,017+41,019)=246,108`; all B lives `3×3×(10,287+10,274)=185,049`; C two arms up to `512×129=66,048`; 32 technical full lines up to `32×129=4,128`. **Absolute bound 501,333 target applications**, under a separately proposed `510,000` supervised target ceiling. Read-only full 257-distribution queries, retries, technical gradient targets, encoding work, counting fits, evaluator probes, stage checkpoints and failed attempts are measured and separately charged CPU/exposure; they are not uncounted “free” capability.

**Proposed** global cap (must receive separate prospective execution authorization): 7,600 full learner records, 510,000 learner target applications, 7,200 process-tree CPU seconds, 14,400 wall seconds, 2 GiB RSS and 2 GiB output. These exceed the historic 5,000-record default **only for the explicit controls of this design**. If authorization at the larger cap is unavailable, *do not silently drop forks or passes*: issue a new prospectively revised V179-R1 with different exposure/claims **before** touching main data. No training or background process is started by this document.

**Sample efficiency:** report held loss improvement vs *unique* A/B/C records and vs total **repeated** target presentations; for reuse on B show the shared B checkpoint at 0, 1, 2, 3 passes. Compute efficiency requires actual complete cost, including source A for KEEP. Any faster learning from prior A cannot claim lower *total* information expenditure without charging prior experience.

## 7. Process/governance correction: what this design prohibits

This project now explicitly separates:
- **Active local learner:** the V167/V169-derived H/T/own-top model with fixed 257 vocabulary, under V179's declared adaptations.
- **Native Full151 reference:** still useful for fast/slow synaptic/associative studies, not a mandatory bridge or imposed output port.
- **BPTT recurrent reference:** V164/V178 dense network, useful as a *reference instrument*, not a local-learning candidate, not a reason to serially force sparsity on the active learner.

The following do not create automatic next-version obligations: a failure of one sparse BPTT reference, failed EOS under a one-byte synthetic task, a better external fitted probe, or a high-dimensional KC overlap proxy. A new mechanism needs **(i)** a measured missing capability in this V179 lifetime, **(ii)** a causal diagnosis, and **(iii)** a unique modification with matched controls before new experiment allocation.

All accepted results remain named `branch@commit/version/arm/world`, with full lifecycle status (DESIGN_ONLY/NOT_RUN/INVALID/FAILED/BOUNDED_POSITIVE). Future E3 never-taught arithmetic remains the binding later acceptance goal; V179 does not weaken it or falsely label an edited-literature environment as E3.

## 8. Provenance

- [V167 original fully local learner/report](https://github.com/ChangweiZhou/digitalbaby/blob/claude/charming-clarke-u9g850/minifly_v167_local_dev_v1_20261026/REPORT.md)
- [V167 local learning implementation](https://github.com/ChangweiZhou/digitalbaby/blob/claude/charming-clarke-u9g850/minifly_v167_local_dev_v1_20261026/v167_local.py)
- [V169 continuing learner/report and negative counting result](https://github.com/ChangweiZhou/digitalbaby/blob/claude/charming-clarke-u9g850/minifly_v169_native_dev_v1_20261027/REPORT.md)
- [V169 actual optimizer reset in `cont()`](https://github.com/ChangweiZhou/digitalbaby/blob/claude/charming-clarke-u9g850/minifly_v169_native_dev_v1_20261027/v169_lib.py)
- [V176 authenticated corpora and sizes](https://github.com/ChangweiZhou/digitalbaby/blob/codex/minifly-consolidated-20261023/minifly_v176_feature_deployment_dev_20261101/DATA_AUDIT.json)
- [V176 literal read-only FE-PERM→KC byte exporter](https://github.com/ChangweiZhou/digitalbaby/blob/codex/minifly-consolidated-20261023/minifly_v176_feature_deployment_dev_20261101/native_stage.py)
- [V178 reproduced dense reference and failed sparse comparison](https://github.com/ChangweiZhou/digitalbaby/blob/codex/minifly-consolidated-20261023/minifly_v178_recurrent_failure_localization_dev_20261102/REPORT.md)
- [Original MiniFly project-level research acceptance contract](https://github.com/ChangweiZhou/digitalbaby/blob/codex/minifly-consolidated-20261023/PROJECT_DOCUMENTS/MINIFLY_RESEARCH_ROADMAP_20261026/MEMO.md)
- [No-attention kernel restriction](https://github.com/ChangweiZhou/digitalbaby/blob/codex/minifly-consolidated-20261023/PROJECT_DOCUMENTS/NO_ATTENTION_KERNEL_REQUIREMENT_20261013.md)

**V179 is allocated once; V180 remains unallocated.** This is a design-only scientific reset and not evidence that any corrected mechanism already works.
