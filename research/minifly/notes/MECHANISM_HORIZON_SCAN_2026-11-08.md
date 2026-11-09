# MiniFly mechanism horizon scan: computation, composition, and scaffolding

**Date:** 2026-11-08 (U.S. Eastern)  
**Status:** research / discussion note; **no experiments authorized**.  
**Relation to existing work:** additive to the locked 70-row legacy crosswalk, F01–F20 evidence ledger, and the substantive evidence audit in ../evidence_audit_2026-11-08/. **Do not alter their IDs or dispositions.** The missing historical original catalogue remains unrecovered.  
**Scope:** proposals for *additional computational operations*, not a claim of five novel, unimplemented biological mechanisms. Source-based biology, published computational models, engineering extrapolations, and existing MiniFly results are kept separate.

## 1. Baseline: what the current project already has

The consolidated byte learner is not a stateless associative table. In the inspected E16 source snapshot it has a gated recurrent cell

\[
h_t=r_t\odot h_{t-1}+w_t\odot c_t,
\]

with supervised readout-error signals and temporal eligibility. Its sign-based backward map uses \(\operatorname{sign}(W)\operatorname{mean}(|W|)\), which **depends on forward weights** and is not fixed random feedback alignment. See:

- [Gated recurrence and learning traces, exp11_cell.py](https://github.com/ChangweiZhou/digitalbaby/blob/9c310c4c0793a7093c2c42280907f1d64342f2b1/MOD97_LOCAL_RULE_20261106/E16V_SOURCE_SNAPSHOT/MOD97_LOCAL_RULE_20261106/exp11_cell.py#L92-L183).
- [Sign-dependent backward weights, exp3_state.py](https://github.com/ChangweiZhou/digitalbaby/blob/9c310c4c0793a7093c2c42280907f1d64342f2b1/MOD97_LOCAL_RULE_20261106/E16V_SOURCE_SNAPSHOT/MOD97_LOCAL_RULE_20261106/exp3_state.py#L90-L108).
- [Existing CX navigation and gain fit](https://github.com/ChangweiZhou/digitalbaby/blob/9c310c4c0793a7093c2c42280907f1d64342f2b1/output/minifly-connectome-navigation/FIT2_REPORT.md) and [path-integration limitations](https://github.com/ChangweiZhou/digitalbaby/blob/9c310c4c0793a7093c2c42280907f1d64342f2b1/output/minifly-connectome-navigation/PI_FOLLOWUP_REPORT.md). These belong to a **different experimental lineage** from the byte learner.

The decisive contemporaneous E16-WR evidence is about target quality, not simply reservoir capacity. At 16,384 replay slots per stage, with matched actual replay prompts/order and snapshots, final held-composition accuracy for self-generated targets was **32.08 / 24.88 / 31.58%** across seeds 73129/93683/116089, versus **62.46 / 57.00 / 65.42%** with stored observed targets. Increasing the *self-target* reservoir size alone was not a monotonic retention solution. Even observed-target replay still lost **14.04–19.71 percentage points** after the composition stage. This is a bounded synthetic demonstration, not proof that target corruption uniquely explains all forgetting.

- [E16-WR report and limits](https://github.com/ChangweiZhou/digitalbaby/blob/9c310c4c0793a7093c2c42280907f1d64342f2b1/MOD97_LOCAL_RULE_20261106/E16WR_REPORT.md)
- [E16-WR verification](https://github.com/ChangweiZhou/digitalbaby/blob/9c310c4c0793a7093c2c42280907f1d64342f2b1/MOD97_LOCAL_RULE_20261106/E16WR_VERIFICATION.json)
- [Structured replay: task/operand entry and rerendering](https://github.com/ChangweiZhou/digitalbaby/blob/9c310c4c0793a7093c2c42280907f1d64342f2b1/MOD97_LOCAL_RULE_20261106/E16V_SOURCE_SNAPSHOT/MOD97_LOCAL_RULE_20261106/exp16_integrated.py#L59-L99)

These findings should inform the following ideas. **Do not propose introducing generic recurrence, a generic multiplicative gate, sign feedback, temporal eligibility, or basic CX navigation as though none has been implemented.**

## 2. Five candidate operations — ranked for discussion, not execution

### 1. Self-supervised predictive state and sequence learning

**Primary evidence.** Yusi Chen, Huanqiu Zhang, Mia Cameron, and Terrence Sejnowski, *Predictive sequence learning in the hippocampal formation*, **Neuron** 112 (2024), 2645–2658.e4. DOI: https://doi.org/10.1016/j.neuron.2024.05.024 . The paper analyzes mouse hippocampal recordings and proposes a CA3 predictive recurrent autoencoder with CA1 comparison to incoming information. Its biological evidence and its network-level algorithm are not interchangeable.

**Proposed operation for MiniFly (not the literal paper's equations):**

\[
z_{t+1}=f_\theta(z_t,x_t),\quad \widehat{x}_{t+1}=g_\theta(z_t),\quad
\epsilon_{t+1}=x_{t+1}-\widehat{x}_{t+1}.
\]

An eligibility-modulated lifetime update could be \(\Delta w_{ij}=\eta e_{ij,t}\epsilon_{i,t+1}\) **only where the relevant error is physically delivered**; this expression by itself is not a proof of local end-to-end credit assignment. The next observation supplies the self-supervised target, not an externally supplied arithmetic answer. For discrete bytes, use a predictive probability distribution and an appropriate categorical loss rather than silently treating byte IDs as continuous quantities.

**What may be gained.** Discovering transitions and recurrent state structure in an unsegmented stream; possible representation reuse without a task parser. Existing next-byte supervision and recurrence are related, so the proposal is to test the **predictive state organization and legally available teaching pathway**, not merely add a next-byte head.

**Cost.** Additional predictive readout \(O(dV)\) weights for hidden width \(d\) and byte vocabulary \(V\), plus existing recurrent state \(O(d)\), eligibility \(O(S_\mathrm{plastic})\) if used, and prediction work per step. A matched baseline must account for the readout, update budget, and all processed tokens.

**Falsifier (design only).** With identical observed streams, training tokens, local learning access and resource budget, predictive-state architecture offers no improvement over the incumbent next-byte recurrent model in withheld-composition and format-change persistence; or it needs hidden task labels/privileged targets.

**Priority:** **discuss first** as an architectural route to less scaffolded learning; not yet a demonstrated MiniFly repair.

### 2. Composable state spaces, binding, and relational factorization

**Primary model.** Jacob J. W. Bakermans, Joseph Warren, James C. R. Whittington, and Timothy E. J. Behrens, *Constructing future behavior in the hippocampal formation through composition and replay*, **Nature Neuroscience** 28 (2025), 1061–1072. DOI: https://doi.org/10.1038/s41593-025-01908-3 . Author code: https://github.com/jbakermans/state-space-composition . The code repository includes planning/homing simulation directories; its code was located, **not executed or independently replicated** in this review.

**Hypothesis.** Role-specific and value-specific states can be recombined without memorizing each possible complete situation. An illustrative *engineering* binding proposal is

\[
z=\sum_{(r,v)}\phi(r)\otimes\psi(v).
\]

This tensor expression is **not asserted to be the paper's implementation**. Raw-stream discovery of roles, segmentation, and valid composition is an additional problem. Supplying operands or an expression tree from the test harness merely relocates the scaffold.

**Potential gain.** Distinguish a symbol's identity from its role/position and combine learned operations for withheld combinations. CX vector transforms are a related computational motif in another MiniFly lineage, not a proof of raw-byte compositional inference.

**Cost.** Dense outer-product binding takes \(O(d_r d_v)\) state/work per bound item; factorized or sparse forms may be cheaper, with additional learned encoders and binding policy. Count any parser or router rather than treating it as free.

**Falsifier.** A role/value representation only improves withheld relations when the evaluator provides true operand boundaries, operation identities or roles, or fails to outperform a same-budget recurrent state learned from identical bytes.

**Priority:** **discuss second** if transfer on unseen compositions remains the bottleneck.

### 3. Neuron-specific dendritic instructive signals

**Primary biological evidence.** Valerio Francioni et al., *Vectorized instructive signals in cortical dendrites*, **Nature** 652 (2026), 1254–1263. DOI: https://doi.org/10.1038/s41586-026-10190-7 . Mouse retrosplenial cortical recordings and targeted perturbation during neurofeedback reveal soma/dendrite differences related to neuron-specific reward and error information. The result supports candidate *instructive signals*; it does not identify a unique synaptic learning rule or prove better MiniFly performance.

**Engineering hypothesis.**

\[
\Delta w_{ij}=\eta\,e_{ij}\,(d_i-y_i),
\]

where \(d_i\) is an internally delivered neuron-specific teaching variable, **not a hidden backpropagated target given for free**. Explicitly compare the source and information content of \(d_i\) against existing sign-dependent feedback \(S=\operatorname{sign}(W)\overline{|W|}\), supervised loss, and eligibility traces.

**Gain/cost.** Potentially better coordinate-specific credit; costs include feedback connections \(S_f\), compartment state \(O(N)\) and its per-step computation. The advantage remains conditional on generating a useful teaching signal from allowed local information.

**Falsifier.** No independent gain over the incumbent sign-feedback/eligibility learning system under identical loss signals, observed targets, development supervision and resource accounting; or success depends on a privileged hidden target.

**Priority:** **conditional learning-rule research**; not the immediate answer to corrupted replay targets.

### 4. Inferred latent context without supplied phase/task identity

**Primary evidence.** Hristos S. Courellis et al., *Abstract representations emerge in human hippocampal neurons during inference*, **Nature** 632 (2024), 841–849. DOI: https://doi.org/10.1038/s41586-024-07799-x . Human neuronal recordings during structured stimulus-response reversals support abstract latent-context representations associated with successful inference. The task supplies informative reward structure, and some participants received explicit instructions; this does **not** establish spontaneous unsupervised task discovery.

**Engineering proposal:** internal context beliefs \(q_t(k)\propto q_{t-1}(k)p(x_t\mid k)\), with state-conditional reading/writing. If new input format is arbitrarily permuted, alignment cannot be identified without informative correspondence, feedback, or co-occurrence. Existing V68/internal-state and bank-routing work are relevant; do not declare this family untouched.

**Potential gain.** Control which learned rule is active through format/phase changes without evaluator stage IDs, contingent on identifiable evidence. Overhead: \(O(K)\) context beliefs and \(O(Kd)\) emission/state parameters in a simple \(K\)-state model, plus any context-conditioned memory and update costs.

**Falsifier.** No task-retention improvement beyond an equal-budget recurrent baseline, or apparent gains rely on explicit phase labels, known reversals or an evaluator-provided context classifier.

**Priority:** **test later, conditional** on demonstrating context confusion is the bottleneck.

### 5. Trainable temporal basis functions / a distributed interval clock

**Primary fly evidence.** Jan Kropf, Clifford B. Talbot and Gero Miesenböck, *A neuronal population clock for interval timing in Drosophila*, **Current Biology** 36 (2026), 835–845.e8. DOI: https://doi.org/10.1016/j.cub.2025.12.032 . Post-odor Kenyon-cell responses span different delays; dopamine-dependent KC→MBON weight changes can produce interval-specific responses. This is fly biological evidence for learning timing, **not** evidence for symbolic arithmetic or unrestricted compositional generalization.

**Proposed operation:** \(u_i(t)=k_i(t-t_0)\), \(y(t)=\sum_i w_i u_i(t)\), with reinforcement-modulated local weight updates. A raw stream requires an *internally inferred* \(t_0\), not a hidden task-start marker.

**Potential gain/cost.** Readout can learn which delay predicts a consequence; requires \(K\) temporal basis states and \(O(K)\) readout coefficients (or \(O(KM)\) for \(M\) outputs). Compare with existing gated recurrence and eligibility at equal writable-state and update cost.

**Falsifier.** No advantage on unseen timing relationships or delayed cue/outcome streams beyond the existing recurrent/eligibility baseline, or the advantage requires externally marked event boundaries.

**Priority:** **strong biological reserve**, low immediate priority if failures remain primarily target correctness/format transfer.

## 3. Important reserve findings, but not five more new modules

- **Specificity of fly plasticity:** Julia E. Manoim Wolkovitz et al., *A voltage-dependent switch underlies efficient yet specific learning and memory*, **Current Biology** (2026), DOI https://doi.org/10.1016/j.cub.2026.01.014 . mAChR-B suppresses plasticity in inactive KCs and permits it in active KCs through voltage-dependent signaling. This refines F04–F06/eligibility/write selection; it is not evidence for an additional broad store architecture.
- **Silent memory and false reinstatement:** *Creating true and false memories from forgotten information in Drosophila*, **Nature Neuroscience** (2026), DOI https://doi.org/10.1038/s41593-026-02381-2 . Behaviorally neutral MBON traces and reminder-dependent reinstatement are relevant when information appears present but inaccessible. Misleading reminders create false retrievals. This is a reserve for **retrieval/expression** problems, not a diagnosis of irreversible weight loss.
- **Behavioral-timescale plateau plasticity:** a possible source of seconds-long instructive windows; overlaps eligibility and commitment mechanisms. Needs exact proposal and source-level comparison before opening a new MiniFly test.
- **Prioritized internal replay:** scientifically interesting for scaffold removal, but the immediate E16-WR results favor **observed target correctness** over indiscriminately increasing self-generated replay capacity. New selector policies must not secretly read task identities or correct answers.

These are **not** claims that prior MiniFly experiments exhaust every variant. A biological finding may support a distinct mechanism even when its most basic computational function has already been represented.

## 4. Decision order and non-negotiable controls

1. **Interpret the ongoing recorded-target / task-label-free replay line before launching new mechanism work.** E16-WR's 16,384-slot paired target-quality comparison is already decisive for *that bounded schedule*, but not for all continual learning. Verify subsequent E16-WR2 outcomes separately before claiming they have completed.
2. **If unseen composition remains weak after reliable targets, discuss predictive-state learning and learned binding** as genuine architecture candidates. Do not call an existing next-byte objective a missing feature; distinguish the proposed state/credit structure precisely.
3. **If the failure is location of credit rather than target fidelity,** compare neuron-specific feedback against the existing sign-symmetric feedback and temporal eligibility with the same permitted global/observed signal.
4. **If failure depends on a latent phase or unknown remapping,** open context inference only when available observations make the phase/format statistically identifiable.
5. **If timing rather than generic context is the measured failure,** consider the fly population-clock operation.

For any future proposed comparison: use the same tokens and externally observed outcomes; match plastic parameter count, recurrent state, replay buffer *bytes*, router/parser resources, compute and learning updates; log provenance of targets and which variables were available online; report acquisition, delayed return, format invariance, held-out composition and persistent transfer. A developmental backprop or attention baseline is legitimate but must be named as such, not counted as a locally learned mechanism.

**Bottom line:** MiniFly's near-term evidence points most strongly to **reliable teaching targets and scaffold removal**. The two highest-value possible new architectural ideas are **self-supervised predictive state learning** and **learned compositional binding**. Neuron-specific dendritic credit, latent context inference and interval-clock plasticity are conditional reserves. This note is a research agenda, **not authorization to run experiments or claim any mechanism absent from the current repository**.

## 5. Provenance and bibliographic status

This note is a literature-informed horizon scan written after the existing 70+F20 evidence audit. The listed primary articles/DOIs were checked for basic identity and abstract-level claims on 2026-11-08. The equations labelled *proposed* are engineering hypotheses, **not** verbatim paper algorithms; paper figures, code and all independent replications were not exhaustively audited here. The repository-specific statements use the immutable source paths linked above, with E16-WR evidence pinned to commit **9c310c4c0793a7093c2c42280907f1d64342f2b1**. Do not infer that experiments occurring after that commit were checked.