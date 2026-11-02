# MiniFly V180 — Phase-Blind Coherence-Gated Local Plasticity

**Identity:** V180-CGLP-D1  
**Date:** 2026-11-02  
**Status:** DESIGN_ONLY — no model implementation, scientific training, source-corpus import, data scoring, adoption or confirmation is authorized by this document.  
**Parent evidence:** V179 continuous life + V179-DIAG-1 + V179-PLASTICITY-1, consolidated branch at `540fb267d01a8ae0656d21a26e5cc3872db6ffe6`  
**Next version:** V181 remains unallocated.

## 0. Scientific decision: one mechanism, not another diagnostic cascade

V179 established a fully local H512/T2048 learner that improves B prediction using A pretraining and learns a synthetic feedback correction, but it failed the full acquisition/retention criterion and did not generate correct complete held literary lines. The post-hoc instrumented run found a same-checkpoint **B train loss decrease and held loss increase in all six B branches**. The paired plasticity study showed that holding H/T frozen during B improved held B at its third pass, **but worsened B retention when H/T resumed during revised-A learning**. These are genuinely useful observations, but do not imply "permanently freeze hidden layers."

**V180's single new hypothesis:** Gradient estimates for a hidden neuron that agree across independent records are more reliable for transferable learning than incoherent, record-specific updates. Apply a reversible, neuron-wise multiplicative gate to the **actual AdamW parameter step**, using only that neuron's input-matched local teaching gradients within the current real minibatch. The proposed mechanism does **not** receive a stage flag or a train/held signal; it is active from the first A example through B and C. A corrected prediction error can potentially re-open plasticity automatically if the new learning signal becomes coherent.

This is a *hypothesis*, not a biological theorem or a known antidote to overfitting. Repeated correlated examples may yield coherent gradients that nevertheless overfit; corrected feedback may be sparse and incoherent. The registered controls test these concerns. If the mechanism fails, its precise status is **CGLP_NOT_SUPPORTED**, not "MiniFly needs another layer, gate, bank, or replay."

**Avoid repeating V175–V179:**
- The **same operative local learner** performs A→B→revised-A across all arms.
- Only hidden-layer **applied-update allocation** changes; no new encoder, output code, recurrent workspace, pretrained model, memory bank, attention, ELM, SNN or specialized arithmetic representation.
- Four arms compare selective regulation with normal learning, a fixed lower effective step, and a **matched-strength nonselective sham**.
- All four lifetime outcomes (acquisition, still-valid retention, revision and reuse) are reported even when one fails. No serial efficacy gate suppresses later measurement.
- Development texts are not V179's opened Thursday/Edgeworth and Emma/Ball pairs; prospective guard texts are not used to select any rules, hyperparameters, stopping passes, or checkpoint.
- Full natural-literary-line exactness is not used as a capability gate. Such completions are generally stochastic and, in V179, many were impossible under its 32-byte truncation.
- Corrected-feedback gain is **local-byte-statistics adaptation**, not evidence of semantic truth revision or E3 arithmetic.

## 1. Fixed source learner and local-learning algebra

Retain the V179 source-locked implementation of V167's fully local predictive learner:

\[
x_t=\mathrm{FE\!-\!PERM\!-\!KC}(\text{raw bytes observed before next target}),\quad
h_t=\mathrm{ReLU}(W_H^\top x_t+b_H),
\]
\[
z_t=\mathrm{Dropout}_{0.5}\left[\mathrm{ReLU}(W_T h_t+b_T)\right],\quad
p_t=\mathrm{softmax}_{257}(W_O z_t+b_O).
\]

Fixed original native sensory code: 5,177 true KC units, empty-state column 5,177, padding column 5,178. Trainable H width 512, T width 2,048, output classes 0–255 plus true EOS 256. No "OTHER", vocabulary mask, forced EOS, numeric/role hints, task-specific parser, attention/key-value read, or fitted evaluator answer head. All H/T/O weights and the full AdamW optimizer history persist through the A, B and revised-A phases; the FE sensory trace may correctly reset at each true line boundary.

At each real next-byte/EOS supervised site, let `e_i=(p_i-\mathrm{onehot}_{257}(y_i))/n`, where `n` is the number of targets in the actual minibatch. H and T receive the unchanged V167 input-matched fixed-random output-error feedback and their local ReLU/dropout factors; O learns by its own ordinary output-local cross-entropy gradient. The fixed matrices `F_H∈R^{257×512}`, `F_T∈R^{257×2048}` have the original named seed convention and variance `1/257`. **No backpropagation through learned layers or through time in the candidate.**

Optimizer settings remain V179's original AdamW (lr=0.001, beta1=.9, beta2=.999, eps=1e−8, weight decay=1e−4), minibatches of 256 consecutive actual target sites in record order, no extra warmup/replay and no best-epoch checkpoint restoration. Each record contains genuine raw payload next-byte targets plus exactly one genuine EOS. No single-byte artificially terminated examples. H/T learning may not be hard-frozen or reset on a phase boundary in any proposed candidate arm.

### 1.1 The one intervention: CGLP neuron-wise update coherence

For every real 256-target minibatch, partition its **distinct source records**, deterministically from their global immutable record IDs, into two groups. Sort the unique record IDs by `SHA256("V180|micro|"+record_id)` and alternate them into `G_1` and `G_2`. Target sites from one record never enter both groups. If either group contains fewer than two distinct records, mark `UNRELIABLE_SPLIT` and assign the floor gain to all H/T neurons for that update. This fallback is fixed, phase-blind and logged; it is **not** a convenient reason to substitute a different batch.

For every H or T neuron `j`, use exactly the ordinary source-local rule on the two disjoint groups to estimate incoming *weight-and-bias* gradient vectors `g_j^{(1)},g_j^{(2)}`. Each group gradient is normalized by that group's actual target count. Use one shared full-batch forward and the actual original dropout draws; **do not resample dropout** to obtain microgradients. Recombine gradients by target counts to recover the exact source-local full-batch gradient, which is passed to the unchanged AdamW moment update:

\[
g_j = \frac{n_1 g_j^{(1)}+n_2g_j^{(2)}}{n_1+n_2}.
\]

Set the *within-neuron* coherence and gain

\[
c_j=\begin{cases}
\max\!\left(0,\ \frac{\langle g_j^{(1)},g_j^{(2)}\rangle}
{\|g_j^{(1)}\|\|g_j^{(2)}\|+10^{-12}}\right),&\text{reliable split and both gradients nonzero},\\
0,&\text{otherwise},
\end{cases}
\qquad
\gamma_j=\frac14+\frac34\,\min(1,c_j).
\]

The cosine uses only the neuron's incoming weights and bias—not another neuron's gradients, gradients of a BPTT model, the correct hidden gradient, held scores, record roles, or task IDs. The gain is recomputed every minibatch: `0.25≤γ_j≤1`. It can increase again on coherent corrected examples. There is no hard reset, externally signaled phase boundary, future target, global answer parser, learned metacontroller, exponential trust coefficient, or hidden replay.

**What the gate actually changes.** Maintain the *full* original AdamW gradient moments and per-parameter step clocks in **all** arms. After moment update, define each neuron's usual combined preconditioned direction, including its decoupled decay:

\[
u_{j,t}
=\frac{\widehat m_{j,t}}{\sqrt{\widehat v_{j,t}}+10^{-8}}
+10^{-4}\theta_{j,t},\qquad
\theta_{j,t+1}=\theta_{j,t}-0.001\,\gamma_{j,t}\,u_{j,t}.
\]

Apply the same `γ_j` to the neuron's entire incoming weight vector **and bias**, not to the raw gradient. H weights are neuron *columns* of the source array; T weights are neuron rows, so implement the axis mapping explicitly. All O weights/biases receive the **unchanged V179 AdamW** step (gain 1). Keep padding/empty-KC conventions source-identical.

This *effective step* placement is important: multiplying Adam's raw gradient by a scalar can be nearly canceled by its adaptive normalization. `γ=1` must reproduce literal V179 AdamW weights, optimizer state and dropout RNG within registered numeric parity. H/T moments still age while a small effective update is applied; this deliberately avoids V179's CLAMP-B stale-moment/resumption confound.

The new local rule is an engineering **circuit/neuron-local** rule using per-record gradient vectors and a 256-target minibatch, not a claim that a single insect synapse literally measures full-vector cosine or that it is strictly single-event online plasticity.

### 1.2 Three matched controls (not competing mechanisms)

On the *same* model, initialization, data, dropout feedback and schedule, only the H/T applied-step gain differs:

| Arm | Hidden gain | Interpretation |
|---|---|---|
| `NORMAL` | `γ_j=1` | Literal V179 local learner and optimizer when other data/corpus changes are held fixed |
| `FIXED-LOW` | `γ_j=0.35` | Is reducing effective hidden writing enough, without adaptation? |
| `RMS-SHAM` | `γ_j=\bar\gamma_\ell` uniform per hidden layer | Does a batch-level reduction, with approximately matched applied update energy, explain the result? |
| **`CGLP`** | `γ_j=0.25+0.75c_j` | Does selecting *which neurons* update matter beyond reducing total movement? |

For `RMS-SHAM`, calculate the same `c_j,γ_j` from that arm's *own* current gradients. Let `u_j` be that arm's proposed AdamW neuron step and define one scalar for each layer `\ell∈\{H,T\}`:

\[
\bar\gamma_\ell
=\sqrt{\frac{\sum_{j\in\ell}\gamma_j^2\|u_j\|_2^2}
{\sum_{j\in\ell}\|u_j\|_2^2}},
\]

with `\bar\gamma_\ell=0.25` if the denominator is zero. At the same model state this exactly matches the sum of squared H/T applied-update magnitudes of a CGLP-vs-uniform step; once the trajectories diverge, **exact cumulative energy matching is not claimed**. Report actual cumulative `\sum\|\Delta\theta\|^2`, path length and runtime in every arm.

All four arms calculate the same microbatch diagnostics, even when the resulting gate is ignored, to make the engineering CPU comparison interpretable. They are separately born models; no weights, momentum or scoring-state clones are transferred between arms. Identical source birth/feedback seeds and the same chronological target sequence are paired within each world. The **only neural learning intervention** is H/T effective step allocation. The unchanged O head always trains.

**Self-hostile warning:** Highly correlated repeated examples can create high cosine despite poor generalization; record-group halves are not guaranteed independent samples from the intended population. A gating benefit in a small corpus can result from smaller effective learning rates rather than learned relevance. V180 explicitly gives those alternatives equal opportunities to win.

## 2. Prospective text worlds, splits and correction transform

No V179 original W1/W2 text is used for efficacy selection or promotion. V179 checkpoints may be read *only* for a source/gradient parity technical fixture. V180 requires authentic NLTK Gutenberg public-domain source files with exact bytes hashed and source metadata frozen before any training. Inability to retrieve a declared source is `SOURCE_UNAVAILABLE`; do not silently substitute another book.

| World | Role | A (acquisition source) | B (subsequent experience) |
|---|---|---|---|
| D1 | Development | `austen-persuasion.txt` | `austen-sense.txt` |
| D2 | Development | `chesterton-brown.txt` | `melville-moby_dick.txt` |
| G1 | Prospective guard | `shakespeare-hamlet.txt` | `shakespeare-macbeth.txt` |
| G2 | Prospective guard | `milton-paradise.txt` | `bible-kjv.txt` |

These world pairings are intentionally different in style and source composition; **not exchangeable iid population samples**. If the source registry shows a guard text was already used to tune this exact V180 mechanism, label it previously opened and withdraw independent-guard language; do not secretly replace it. Sources and splits must be frozen once, without reading held scores for model selection. Because the design is public, “guard” means the mechanism and thresholds are locked before running it—not a claim of cryptographic secrecy against a reader of the repository.

### 2.1 Exact preprocessing and dose

Use the pinned V176 `corpus.py` raw-byte record splitting and Gutenberg-footer exclusion. Reject a source that cannot produce enough nonempty records of original length ≤128 bytes; do not change truncation, normalize unusual characters post hoc or remove hard lines because they hurt scores.

For every A corpus use the first **512** qualifying records as train, next **128** as calibration, and final **128** as held, verifying no exact train/held record duplication. For every B corpus use the first **128** qualifying records as train, next **128** as calibration and final **128** as held. Train source A for **3 chronological passes** and B for **3 chronological passes**, with all parameters, Adam moments/steps and fixed feedback persisting. Each arm follows its *own* continuing A→B history. No model receives the world name, author name or phase ID in its neural input.

Each record is processed as raw bytes and one real EOS (257-class prediction site). Compute the FE-PERM/KC trace from already observed prefix bytes, including its correct record boundary reset. A held evaluation is read-only. A return to repeated training data is charged as another optimizer/record presentation, not independent information.

For each world, before main learning **select the A' edit map from A-train only**:

1. Enumerate two-byte preceding contexts and next target bytes in the untouched A train records, never across records.
2. Candidate context must occur ≥32 times, and the most frequent following byte must be an ASCII letter `A–Z` or `a–z` with ≥60% empirical share. Select exactly four eligible contexts ranked by training frequency, deterministic SHA256 ties. Excluding spaces corrects V179's misleading “he [space]→letter” toy edits; this is *still only changed conditional byte statistics*, not real semantic fact revision.
3. For each selected context choose a distinct replacement ASCII letter from the sixteen most frequent A-train *letters*, excluding the original letter; deterministic SHA256 ordering. Replacement never changes length, EOS, newline or the number of supervised sites.
4. Select edits against the *original unmodified record*, replacing the old modal letter whenever its exact two-byte context occurs; no cascading substitutions. Select exactly **64** A-train record IDs containing ≥1 edit by fixed hash. C trains on those 64 in original form (`NOEDIT`) or corrected form (`EDIT`), two passes each.
5. Apply the same source-train-derived transform to A calibration and A held, without looking at held performance or selecting a new key. Require ≥24 changed held target sites, ≥128 stable held sites with identical expected byte, and ≥128 **bit-identical KC states** between original and corrected held streams; otherwise mark `CORRECTION_FIXTURE_INSUFFICIENT` (the original A/B results remain reportable). Do not improvise a fallback edit scheme after failing this data-only preflight.

A' records are **re-encoded using their changed observed prefixes**. The model only receives raw bytes and legitimate next-target feedback. It never sees old/new keys, transition probabilities, record IDs, cohort IDs, or a correction flag.

### 2.2 One continuing life, with matched C counterfactual

For **each of the four arms** in **each of the four worlds**:

\[
\mathrm{Birth}\to A_1\to A_2\to A_3
\to B_1\to B_2\to B_3
\to \begin{cases} C_{\rm EDIT,1}\to C_{\rm EDIT,2}\\C_{\rm NOEDIT,1}\to C_{\rm NOEDIT,2}\end{cases}
\]

The EDIT and NOEDIT branches fork from the **same actual B3** checkpoint, identical optimizer moments/age and dropout RNG, and receive exactly the same original 64 record IDs, ordering and source-data dose. Neither branch resets or changes learning rule at C; CGLP remains CGLP, FIXED remains FIXED, etc. The edit branch is the primary continued life; NOEDIT is an evaluator-controlled alternative feedback history with identical total supervision count.

All phases run at fixed dose regardless of early performance. No "if sparse reference fails" gate, no phase-specific controller, no best B1 rescue, no retuning on a guard score. Holdout readouts are measured after each pass without feeding held gradients to any learner.

## 3. Correct the measurement contract before evaluating the intervention

### 3.1 Same-checkpoint predictive generalization, not on-line train versus final held

At birth and after every A/B/C pass, measure from the **same frozen model checkpoint**, dropout off, full 257-class unmasked softmax:

- A and B train/calibration/held loss, payload/EOS loss and top-1; corrected A' train/calibration/held loss and changed-site accuracy;
- actual input/output entropy, EOS probability before a line ends and EOS probability at genuine boundaries, plus predicted/observed record-length histograms for free running;
- per-record effect distributions and exact repeated-token counts (a thousand correlated positions are not a thousand independent lifetimes);
- number and magnitude of physical H/T/O applied updates, including actual coupled moment/weight decay and the fraction of unreliable microbatches.

A B pass1→pass3 result with train improving and held worsening is explicitly an **overfitting-like divergence**, not proof that hidden plasticity alone causes it; natural text may shift between beginning and ending of a novel.

### 3.2 Revise correctly: A' input contexts matter

V179 defined “STABLE” using unchanged last-16 bytes but its FE-PERM traces can depend on earlier bytes. Therefore V180 calculates **all three distinct readings**, not conflating them:

- `A_OLD`: original A held records evaluated on the exact original raw prefix and label. This measures backward-compatible *old-distribution* predictions.
- `A'_{SAME-KC}`: on the corrected A' held streams, positions where **(i)** original target equals revised target and **(ii)** the complete actual sparse KC code (including sensory timing) is exactly identical between original and corrected record. This isolates still-valid predictions on identical sensory states and is the strictest retention subset.
- `A'_{ALTERED-KC}`: target byte unchanged, but the true corrected prefix produces a different KC code. This measures whether still-valid responses generalize to changed context, not identical-state retention.

For `A'_{CHANGED}`, score the expected new byte in the corrected A' stream, the predicted new-vs-old log odds and top-1, always after recomputing the corrected prefix. Compare EDIT versus the identical-dose NOEDIT branch for each model. Never count improved prediction of an *obsolete old label* as a retention benefit. Report the counts/denominators of each subset, plus how many original contexts move between KC codes. This repaired evaluator is **new V180 evidence**; do not back-apply the definitions to make V179 pass.

### 3.3 Correct autonomous generation and EOS claims

The V179 zero-exact-line statistic included arbitrary unseen literary sentences, many longer than the 32-byte limit. That is not a valid standalone binary indicator of ability to learn predictive language.

V180 **does not** use matching an unseen literary line character-for-character as an efficacy gate. Instead, report:

- held one-step EOS NLL, EOS calibration/Brier score at authentic record boundaries and false-EOS rate on nonterminal positions;
- from a fixed observed prefix of three true bytes, generate at most **128 model-produced bytes/EOS**, measure the fraction that emit EOS, length distribution and truncation. Compare distributions to the actual predeclared held line lengths; do **not** score a particular stochastic novel line as the single exact correct output.
- optional descriptive complete-exact rate restricted to held lines whose required continuation fits the 128-step cap; list denominator and emphasize that exact natural-language continuation is not generally identifiable from a short prefix.

This repairs the impossible-cap and ambiguity mistakes **without** masking EOS, forcing stop, using a fitted decoder or pretending that lower teacher-forced NLL is a complete own answer. A later deterministic E1/E2/E3 task with well-defined full byte+EOS answers is still required.

### 3.4 Independent external references

Fit unigram and interpolated Kneser–Ney or PPM to exactly the source data seen in each life, and report their performance on the same held data. These are evaluator countermodels, never the learner's own answer. Report cost of their fits and hyperparameter selection on calibration only. Include a zero-A-learning comparator from the fresh NORMAL arm only as a **descriptive** B-only reference if the full additional source-learning costs are separately admitted; it is not required for the CGLP causal test and not silently free. The prior V179 B-COLD result is not portable to these new books.

## 4. Comparison gates: distinguish lower step size from neuron selection

Define four *held* per-world outcomes for each physical learning arm `a`:

\[
\begin{aligned}
B_a^{(3)} &= L_{B,\rm held}(\text{after B3}),\\
B_a^{(C)} &= L_{B,\rm held}(\text{after corrected C2}),\\
A_{a,\rm stable}^{(C)} &= L_{A'_{\rm SAME-KC},\rm held}(\text{after corrected C2}),\\
A_{a,\rm changed}^{(C)} &= L_{A'_{\rm CHANGED},\rm held}(\text{after corrected C2}).
\end{aligned}
\]

All are own-model 257-class bits per site, not fitted-probe losses. The **fixed composite diagnostic** (not a substitute for the vector) is

\[
S_a(w)=\tfrac14\big(B_a^{(3)}+B_a^{(C)}+
A_{a,\rm stable}^{(C)}+A_{a,\rm changed}^{(C)}\big).
\]

The 1/4 terms reflect *equal scientific domains*, not empirical frequencies. Corrected sites can be sparse and correlated; this composite is an engineering screen, not a population risk estimate. Always present the four components alongside it.

**Primary mechanistic claim `SELECTIVITY_ADDS_VALUE` (both guard worlds G1/G2):**
1. `S_{CGLP}≤S_{RMS-SHAM}-0.05` **and** `S_{CGLP}≤S_{FIXED-LOW}-0.05` bits;
2. Neither B-after-C nor A'-SAME-KC-after-C nor revised-target-after-C is worse for CGLP than *either* comparator by more than 0.10 bits; A acquisition loss at A3 is no more than 0.10 bits worse than NORMAL;
3. Corrected-label top-1 and new-vs-old log odds remain explicitly reported, and CGLP's revised held corrected-target NLL is **no worse than** NORMAL by 0.10 bits; if that fails the apparent B benefit is not a full-lifetime solution;
4. No nonfinite updates, data-source violations, hidden labels, source-drift, extra sample presentations or caps violation.

A second descriptive screen **`GENERALIZATION_DIVERGENCE_REDUCED`** reports whether CGLP's increase in B held loss from pass1 to pass3 is ≥0.10 bits **smaller** than NORMAL's, while its B training loss does not imply an artificial cap or hidden no-learning policy. This is not a separate right to promote CGLP if the full-lifetime primary screen fails.

Comparison to NORMAL remains mandatory: a mechanism may look favorable relative to weaker/undertrained controls yet be inferior to the original learner. The fixed lower-step comparator can win. RMS-SHAM can win. Neither result is an excuse to retune `0.25/0.35` or the cosine formula on opened guard worlds.

**Development vs guard:** D1/D2 are for source/engineering and descriptive diagnosis only; all hyperparameters, comparison rules, stop conditions and code are fixed **in this D1 document before any run**. G1/G2 use other predeclared books and may not be evaluated until the code/source and exact preflights are locked. Run them even if D1/D2 are disappointing; do not select only a promising arm. Two guard pairs are not iid evidence of population generalization. The strongest admissible result is `PROMISING_GUARDED_LIFETIME_DEV`, not independent proof of persistent reasoning or core adoption. If the composite's conclusion is negative, publish which arm prevailed and all bounded vector effects without automatically allocating another version.

## 5. Technical checks and source closure

Before scientific fitting, source-lock the recovered V179/V167 local learner and V176 literal sensory exporter; verify existing source SHA and runtime dependencies (original Claude assets and project local raw corpora are excluded from the public GitHub artifacts). Missing dependencies are an E0 stop `SOURCE_UNAVAILABLE`, not permission to replace source code with a conveniently different new model.

Required C0 technical validations:

1. **No rule applied:** `γ=1` exactly reproduces the existing V179 full-batch prediction, H/T/O ordinary local gradients, 257 logits, AdamW full optimizer state, one complete optimizer step, dropout RNG, and FE-KC source code for true ordinary bytes/EOS. Compare to the pinned V179 literal implementation, including correct H column / T row update axes, no accidental pad-row writes, and fixed feedback matrices.
2. **Split parity:** for two disjoint batches and uneven target counts, the recombined `g_j` equals literal source full gradient (relative tolerance ≤1e−6, abs ≤1e−7 on near-zero entries), using the *same* single dropout mask and class labels; no duplicated or omitted target; no gradient crossed an H/T/O learned layer.
3. **Optimizer effect:** one-step independent reference verifies the post-preconditioned H/T scaling, decoupled weight decay, momentum age and correct O untouched; `γ=0.25` must cause a **real** one-quarter applied H/T movement for identical initial state, unlike naive pre-Adam raw-gradient scaling. Zero-update contexts use the fixed floor without NaN.
4. **Sham:** given identical birth/state and full gradients, RMS-SHAM exactly reproduces CGLP's per-layer proposed **squared** applied step norm before model trajectories diverge; actual after-divergence energy not claimed equal.
5. **Controlled continuity:** all four arms train from the same paired source birth, carry all six learned parameter sets and AdamW states through A/B/C; C EDIT and NOEDIT begin bit-identically and share RNG. A and B phase labels enter only the evaluation ledger, never neural inputs or the gain algorithm.
6. **Data/evaluator:** authenticate the exact source files before training, generate edits from A train only, re-encode corrected prefixes, verify true 257-class output normalization, all real EOS positions, exact-same-KC masks and changed/unaltered partitions. Prove no held output enters source selection or step-size decisions.
7. **Resource admission:** profile 256-target microgradient computation, full 4-world ×4-arm lifetime, counting fits, held scoring and 128-step generation; ×1.5 upper forecast must fit CPU/RSS/output caps before any main-world training. No silent shortening, GPU migration, output-panel omission or cheaper model. If not admitted, stop for *prospective* owner-approved budget amendment, never hide costs or skip a control.

**No performance gate for starting B/C or guard.** A weak baseline is a negative result, not a reason to skip the comparison that could identify a useful improvement. An invalid execution or missing data can stop without drawing a scientific verdict.

## 6. Frozen resources and receipts

Physical training records per **arm/world**, excluding read-only scoring:

- A: 3×512 = **1,536**;
- B: 3×128 = **384**;
- C EDIT + C NOEDIT: 2×64 + 2×64 = **256**;
- **Total per arm/world = 2,176**.

Four worlds × four arms = **34,816** complete learner record presentations. At most 64 source/numerical technical records; hard physical learner ceiling **35,000**. Records have ≤128 source payload bytes + one real EOS, so a conservative target upper bound is `34,880×129=4,499,520` supervised target sites; global target cap **4,550,000**. The actual count must be reported precisely from raw line lengths; no number of repeated records is called independent evidence.

Use one worker/CPU with explicitly controlled BLAS thread count. **Proposed**, not approved, global limits: **7,200 aggregate process-tree CPU seconds, 14,400 wall seconds, 2 GiB peak RSS, 2 GiB new output files**. Preflight the full plan at ×1.5 CPU and storage reserve. Adaptive and sham microgradient computation may cost more than NORMAL; per-target sample efficiency and end-to-end CPU efficiency are separately reported. The 35,000-record cap **exceeds historical default allowances** and needs separate owner approval before any scientific execution.

At every true phase checkpoint, save a full *mutable learner state* (parameters, all Adam moments/clocks, fixed feedback source IDs, sensor birth, dropout RNG, current data record/target offset), frozen own scores and per-domain denominators. Generate a compact source/seed/export manifest and compute from saved raw panels in a separate reducer; do not retrain during reduction. Preserve source negative outcomes, technical failures and intermediate computations in the accounting ledger. If raw tensors/targets are excluded from GitHub under the existing publication policy, mark the public aggregate as **not externally raw-reproduced**.

## 7. Future status and release boundaries

Possible verdicts: `INVALID_SOURCE`, `RESOURCE_NOT_ADMITTED`, `INVALID_NUMERIC_PARITY`, `CGLP_NOT_SUPPORTED`, `REDUCED_MAGNITUDE_SUFFICIENT`, `UNIFORM_GATE_SUFFICIENT`, `SELECTIVITY_ADDS_VALUE_DEV`, and—if the separate prospective guard supports it—`PROMISING_GUARDED_LIFETIME_DEV`. Use a single primary status plus orthogonal signed-outcome table; do not manufacture a success from one favorable component.

No V181 is allocated here. No empirical outcome can rewrite V179, claim fully learned hidden task semantics, imply that source-experiment fixed feedback is exact backprop, or pass E3 never-taught addition by achieving text-loss improvements. The continuing operative H/T learner remains a **development candidate**, Full151 is still a separate associative-learning reference, and V164/V178 remains a BPTT **reference instrument**.

Only if a generic local update rule improves *all relevant lifetime outcomes* without phase hints should a later independently designed experiment study its integration into a recurrent workspace or its ability to produce complete taught and never-taught arithmetic answers. Avoid another mandatory serial prerequisite.

## 8. Source pointers

- [V179 original complete lifetime and four screens](https://github.com/ChangweiZhou/digitalbaby/blob/codex/minifly-consolidated-20261023/minifly_v179_continuous_local_life_dev_20261102/REPORT.md)
- [V179 repeated-pass training/held divergence](https://github.com/ChangweiZhou/digitalbaby/blob/codex/minifly-consolidated-20261023/minifly_v179_lifetime_diagnostic_20261102/REPORT.md)
- [V179 paired H/T freeze and C continuation](https://github.com/ChangweiZhou/digitalbaby/blob/codex/minifly-consolidated-20261023/minifly_v179_plasticity_location_20261102/REPORT.md)
- [V179 source local learner + mutable optimizer](https://github.com/ChangweiZhou/digitalbaby/blob/codex/minifly-consolidated-20261023/minifly_v179_continuous_local_life_dev_20261102/learner.py)
- [V179 source-accurate FE-PERM edit and export](https://github.com/ChangweiZhou/digitalbaby/blob/codex/minifly-consolidated-20261023/minifly_v179_continuous_local_life_dev_20261102/data.py)
- [V179 former generation endpoint and 32-step limitation](https://github.com/ChangweiZhou/digitalbaby/blob/codex/minifly-consolidated-20261023/minifly_v179_continuous_local_life_dev_20261102/evaluate.py)
- [V167 fully local teaching source, Claude branch](https://github.com/ChangweiZhou/digitalbaby/blob/claude/charming-clarke-u9g850/minifly_v167_local_dev_v1_20261026/v167_local.py)
- [Active cross-project lifetime research contract](https://github.com/ChangweiZhou/digitalbaby/blob/codex/minifly-consolidated-20261023/PROJECT_DOCUMENTS/MINIFLY_ACTIVE_LIFETIME_CONTRACT_20261102.json)
