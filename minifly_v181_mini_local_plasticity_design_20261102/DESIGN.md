# MiniFly V181 — Compact Local Learner with Phase-Blind Plasticity

**Identity:** V181-MINI-CGLP-D1  
**Date:** 2026-11-02  
**Status:** DESIGN_ONLY — no implementation or model training is authorized by this publication  
**Parent:** V180 STOPPED_AT_USER_REQUEST / RESOURCE_NOT_ADMITTED; repo `414b56f0b069b9155f71303b815a997ce08e73f2`  
**Scientific question:** In a genuinely small local predictive learner, does neuron-specific coherence-gated plasticity improve acquisition, later learning, retention and legitimate feedback correction compared with both ungated learning and equally inexpensive uniform reduction of hidden writing?

## 0. Why V180 is not simply being resumed

V180's technical source and gate checks were real, but the planned scientific run was not launched. It had **4,229,377 trainable parameters**; about **16.9 MB weights** and **50.8 MB each full weights+Adam-moment checkpoint**. The minimum phase checkpoint roster would consume **4.87 GB including its declared 1.5× storage reserve**, exceeding the 2 GiB output cap before model probes. The proposed 35,000-record / 4.55M-target plan involved **16 separate learner lives**. Its main resource admission was `RESOURCE_NOT_ADMITTED`. No efficacy or guard neural score was obtained. A larger disk or omitting necessary results would not solve the objective mismatch.

This is a **material architectural correction**, so the V180 source-locked proposal and 64-record technical results remain immutable; V181 receives a new number. V180's saved technical tests are a source of formulas and bug fixtures, not positive efficacy evidence for a small network.

**Binding mini contract:**
- **≤40,000 learnable scalar parameters**, including all gates, biases, byte classes and output weights; actual V181 plan: **37,313**.
- **One fixed 5,177-KC native sensory circuit**, not 16 copies of a 5,177-KC bank per candidate. The 88-receptor/PN frontend and native PN→KC birth wiring are counted separately as immutable, physically deployed state. Its full stored representation, including fixed matrices, mappings and single-record activity, must fit **8 MiB model-specific inference memory**; if not, declare `MINI_MEMORY_INVALID` without bypassing the bound. Framework/Python RSS is measured separately.
- **<1 MiB serialized mutable weights+optimizer+RNG per checkpoint**, with read-only native sensor referenced by its content hash, and **≤256 MiB complete new experiment output**, including checkpoint roster, source manifests and evaluation summaries.
- **≤5,000 physically trained full-record presentations**, including all arms, worlds and technical updates; **≤600,000 real supervised target presentations**; **≤900 CPU seconds** in aggregate (1,800 wall seconds, 1 GiB peak process RSS).
- **No free pretrained learned features:** all 37,313 trainable parameters begin from their registered seed, and the input sensory graph is the original fixed, non-learned fly birth. No hidden 4.23M-param reference network may supply candidate embeddings, synaptic updates, logit corrections or predictions.
- No model enlargement, output-class compression, missing controls, hidden source-corpus substitution or additional training budget after observing data outcomes.

The project is **not** claiming that fewer neurons automatically improve learning. A model that fits the mini contract but fails to learn remains a valid negative result. Neither a resource violation nor a poor training curve authorizes another architectural expansion.

## 1. Actual compact neural circuit, with exact parameter census

### 1.1 Fixed sensory subsystem

The original FE-PERM receptor input state has 88 channels and drives the actual source-locked native KC encoder with 5,177 KC activities. FE-PERM sees only genuine already-observed raw bytes. The required output at each true next-byte/EOS site is the next byte 0–255 or end-of-real-record EOS=256. Reset only the *transient* receptor/KC sensory history at a genuine line boundary; never reset the learned neural/optimizer weights between A/B/revision stages.

A single native KC birth is extracted from the original Full151 code: immutable `B` sparse PN→KC wiring, side indices, thresholds, receptor indexing and sensory reset/permutation constants. **No Full151 16-bank fast/slow value storage is instantiated in the deployed learner.** Source-parity technical tests compare this lightweight sensor against the literal V180/V179 native sensory path on all 256 input bytes, empty state, complete records, punctuation, synthetic correct/incorrect prefixes, and actual EOF timing. The source graph must be identical; if exact parity requires hidden bank-state coupling, stop rather than add silent banks or accept approximated codes.

Report all physically resident immutable sensor bytes separately from learned model bytes. Shared immutable birth arrays may be physically shared between the experimental control lives, but the **deployed single-model footprint** must include one complete copy.

### 1.2 Sparse KC→H, dense small H→T, native-sized categorical output

Set `K=5177`, H=64 ReLU neurons, T=64 ReLU neurons and O=257 categorical output units, with ordinary 0–255 payload plus true EOS.

A deterministic, input/label-independent seed `SHA256("V181|KC-H-birth")` selects one permutation `\pi` of all 5,177 KC indices. For H neuron `j∈\{0,\ldots,63\}`, the **only** learnable KC inputs are

\[
I_{j,r}=\pi((256j+r)\bmod5177),\qquad r=0,\ldots,255.
\]

Thus every H unit has **exactly 256 KC afferents**, no duplicate afferents within a unit. The 16,384 directed KC→H edges cover each original KC **three or four times**, without labeling by the text, byte class, digit identity, or target role. Store only the 64×256 allowed weights and fixed uint16 connectivity indices: do not allocate an “actually dense” 5,177×64 learned tensor and call it sparse after masking.

\[
a^H_{j,t}=b^H_j+\sum_{r=0}^{255} W^H_{jr}\,x_{I_{j,r},t},
\quad h_{j,t}=\max(0,a^H_{j,t});
\]
\[
a^T_t=W^T h_t+b^T,\quad
z^T_t=\operatorname{Dropout}_{0.1}(\operatorname{ReLU}(a^T_t)),
\quad p_t=\operatorname{softmax}_{257}(W^O z^T_t+b^O).
\]

The 0.1 dropout is a declared **compact-circuit architectural choice** (V179 used 0.5); it is identical in all four V181 arms. There is no position/phase feature, pretrained encoder, content-addressed attention, Transformer, arithmetic-specific representation, recurrent BPTT, auxiliary externally fitted head, forced EOS or class mask.

**Learned parameter census (not merely active weights):**

| Layer | Learnable weights | Biases | Total |
|---|---:|---:|---:|
| KC→H (64×256 sparse real weights) | 16,384 | 64 | 16,448 |
| H→T (64×64) | 4,096 | 64 | 4,160 |
| T→257 categorical output | 16,448 | 257 | 16,705 |
| **Total** | **36,928** | **385** | **37,313** |

Relative to V180's 4,229,377, V181 has **113.35× fewer learnable parameters**; the old model was approximately 113× larger. H/T/O contribute 64+64+257=385 explicit learner/output units beyond the 5,177 native sensory KCs, or **5,562 units total**, plus the 88 sensory channels. This is a physically bounded fly-inspired model, not “mini” because an external giant model is frozen.

Float32 learned weights occupy exactly 149,252 bytes; two full float32 Adam moments occupy 298,504 bytes, aside from tiny clocks/buffers. The fixed feedback matrices `F_H,F_T∈R^{257×64}` together occupy 131,584 float32 bytes, and the 64×256 uint16 incidence map 32,768 bytes. State-accounting must include the source PN→KC wiring, frontend receptor table, all bias/gradient tensors, fixed feedback, dropout state and output buffer. No hidden activation/code cache is part of the persisted cognition: immutable sensory codes may be cached for evaluator throughput, but must be counted as experiment data/outputs and **never cached from trainable hidden states**.

### 1.3 Original local learning rule, specialized to this exact graph

Use the same principle as V167/V179, not an imported pretrained local neural network. For each **true** observed next-byte/EOS target `y_i` in a batch of size `n`,

\[
e_i=(p_i-\mathrm{onehot}_{257}(y_i))/n,
\]
\[
\delta^T_i=(e_iF_T)\odot\mathbf1_{a_i^T>0}\odot m_i,\qquad
\delta^H_i=(e_iF_H)\odot\mathbf1_{a_i^H>0}.
\]

`m_i` is the single actual inverted-dropout mask (0.1 dropout) from that one batch. O is trained with its own actual cross-entropy output-local gradient. T receives its own local teaching signals times the H presynaptic activities. H receives its own signals times the 256 **actually connected** KC activities per neuron. Do not propagate a gradient through the learned T→H connection or across time. Fixed error broadcasts `F_H,F_T` are born independently of output weights with Gaussian variance `1/257` and remain frozen. The new 257×64 feedback shape is explicit; **this is not literal numerical reproduction of V179's 257×512 / 257×2,048 gradient tensors**.

Use one shared weight-birth source per world, zero biases and variance-scaled weights at the fixed fan-in: KC→H `\mathcal N(0,1/9)` (nine is an approximate expected *active* afferent count), H→T `\mathcal N(0,1/64)`, T→O `\mathcal N(0,1/64)`; no task-dependent birth. Forward numerical and learning-rule sign checks are mandatory. An implementation may exploit source KC sparsity and the actual 16,384 edge table, but cannot allocate dense 5,177×64 learnable weights/moments.

All arms use AdamW lr=0.001, beta1=0.9, beta2=0.999, eps=1e−8 and weight decay=1e−4, target minibatches up to 256 consecutive positions from true records in their chronological fixed order. All weights, optimizer moments, per-coordinate clocks and dropout RNG persist from A through B to C; genuine A/B/correction boundaries are **not** neural signals. No validation-best restore or extra scheduled pass.

## 2. One mechanism, three controls: preserve V180's scientific question

V181 does not invent a new mechanism to solve V180's scaling problem. It **ports the already-technical-qualified V180 CGLP equations** to the newly compact substrate. All four arms are trained on the *same* compact architecture, data and exposure schedule:

| Arm | Effective H/T step | Interpretation |
|---|---|---|
| `NORMAL` | `\gamma_j=1` | Honest ungated local learner of the same tiny graph |
| `FIXED-LOW` | `\gamma_j=0.35` | Could simply reducing effective hidden updates suffice? |
| `RMS-SHAM` | One H scalar and one T scalar chosen to match proposed CGLP layerwise squared step energy at that arm's current state | Is neuron-specific selectivity necessary? |
| `CGLP` | `\gamma_j=0.25+0.75\max(0,\mathrm{cos}(g_j^{(1)},g_j^{(2)}))` | Candidate phase-blind selective plasticity |

In every up-to-256-site batch, assign complete distinct record IDs to two hashed alternate groups (same deterministic global record-ID policy as V180). Never split a record into both groups. Compute the H and T **incoming weight+bias local gradient** independently on each group, using the *same sampled forward/dropout mask* and target-wise true error as the ordinary full batch, then normalize by group target counts. The weighted recombination must equal the exact full-batch source-local gradient within numeric tolerance.

If either group contains fewer than two distinct records or a neuron has zero gradient, its positive coherence is **zero** and CGLP gain is **0.25**; report fraction of affected targets and updates. A short tail batch is not padded with fake records or ignored.

For `c_j` compute a nonnegative local gradient cosine with `\epsilon=10^{-12}`. Normal AdamW moments/clocks are updated from the *full, unscaled gradient* in all arms. Only **the actual post-preconditioning parameter movement**, including decoupled weight decay, is scaled by each H/T neuron's current `\gamma_j`. O is always trained at gain 1. Consequently a lower gain is not canceled by Adam's gradient normalization, and the C phase can reactivate H/T plasticity without receiving an experimenter-supplied “correction” flag.

RMS-SHAM computes its one gain per layer from that arm's own current CGLP proposals, with the ratio of `\sum_j\gamma_j^2\|u_j\|^2` to `\sum_j\|u_j\|^2`. It matches the candidate's proposed squared H/T update movement **only at the same current model state**; trajectories diverge, so cumulative energy equality is not claimed. All arms calculate the same microgradient diagnostics so CPU comparisons are properly charged.

**Key attribution limit:** Because V181 also shrinks the architecture, its result can show CGLP versus controls **within this mini architecture**. It cannot demonstrate that the mechanism repaired V179's large model, nor can an ineffective miniature prove that the original CGLP is universally wrong. The model may be statistically underpowered; that is separately measured.

## 3. Deliberately mini continuous-life roster

Two *previously authenticated but neural-efficacy-unopened* source pairs from V180's technical preflight:

| World | Role | A source | B source |
|---|---|---|---|
| `M1` | Development | `austen-persuasion.txt`, SHA256 `cec097657cbfeb62aafe178973f514fa07ab66a5e28ea025ebd5d51ca870e40f` | `austen-sense.txt`, SHA256 `57b169c7cfa502896dee25bd66616f61cc1d21084caf31f015176ee46fda9718` |
| `M2` | Prospective second-world check | `shakespeare-hamlet.txt`, SHA256 `1eceab7027ad7e0cc25c6b6aadcb2e6bb070395cb2085f775ac732d63d5db240` | `shakespeare-macbeth.txt`, SHA256 `d4a204352cfe5924cbae76f764225d120e9f7617d39cb8c949dedd3f306353ac` |

**Source-hash qualification:** Both A and B corpus hashes in the table are taken from V180's prior authenticated `DATA_AUDIT.json`; the V181 preflight must independently verify byte-for-byte SHA256 against those exact strings before generating any subset or performing training. Do not generate, train or count on any source whose measured SHA differs. V181's first160/first48 cuts are new and must be hashed before model scoring. Previously audited source metadata is not an independent learning result.

**Physical training schedule, per arm/world, no outcome-selected stopping:**

| Phase | Records per presentation | Passes | Physical records |
|---|---:|---:|---:|
| Learn A | first 160 eligible, length≤128 | 2 | 320 |
| Continue B | first 48 eligible, length≤128 | 3 | 144 |
| Corrected-A C EDIT | 24 selected original A record IDs | 2 | 48 |
| Same-feedback-dose C NOEDIT | identical 24 original A record IDs | 2 | 48 |
| **Total per arm/world** | | | **560** |

Thus **two worlds ×four arms ×560 =4,480 main learner records**, plus at most32 technical records => at most **4,512**, under the existing **5,000-record cap**. Every complete record has at most129 true byte/EOS target sites => maximal supervised target applications **4,512×129=582,048**, under a **600,000-target cap**. Real record lengths and exact target counts must be reported rather than substituting this upper bound.

For A: use original source's first160 eligible complete records as train, next32 as calibration and last64 as held. For B: first48 as train, next32 calibration and last64 held. Eligible means authenticated, nonempty original `corpus.py` records of ≤128 payload bytes after the original Gutenberg-footer exclusion. Preserve the order: no task-dependent segmentation, semantic labels, delayed-copy rules, word embeddings or externally supplied phases. Ensure no exact train/held full-record duplicates; if a known source violates that constraint, mark `INVALID_DATA` and do not choose a more convenient source after seeing results.

**Data-only correction C:** From the *first160 A-train records only*, select the single most frequent two-byte context whose modal following target is an ASCII letter with ≥60% conditional prevalence and ≥12 original context occurrences, maximizing the number of **different A-train records** in which an actual edit would occur. Tie break using SHA256 of world and raw context; source A never sees a learner-score-based selector. Select a distinct replacement ASCII letter from the original A-train top16 letters with a fixed SHA256 ordering. Replace that modal byte wherever the original unmodified two-byte history and old label match; no cascading edits, length changes, extra EOS, or forced output. Select the first24 A-train record IDs with ≥1 edit using the frozen hash ordering. NOEDIT receives **the same** 24 original records and order. The EDIT and NOEDIT forks start from the exact same true B3 model/moments/RNG and train for two passes each.

Preflight source-only validity: ≥24 edited A-train records; ≥8 truly changed held A' target sites; ≥64 A' held positions with unchanged labels **and bit-identical complete FE-PERM/KC states**. If insufficient, A/B learning remains valid evidence; correction is `CORRECTION_FIXTURE_NOT_QUALIFIED`, *not* a reason to invent a new context or increase corpus sizes.

**Data and model independence:** The world IDs, corpus names, source-edit hash selector, mini model birth, fixed feedback births and every control arm are frozen in the exact source lock before scientific fitting. M2 has not had V180 neural efficacy opened, but its text has been used for technical preflight and is not a statistically sealed independent sample. Do not call the 2-world result confirmation or publish p-values treating correlated tokens as independent lives.

## 4. Primary evaluation: one living learner, four actual competences

Measure frozen-checkpoint own **257-class next-byte/EOS** loss at birth, after each A pass, after B1/B2/B3 and C EDIT/NOEDIT after C1/C2. Read only true observed prefixes; do not pass held labels into the learner. Save the complete mutable state and true AdamW clocks at each checkpoint; no best validation checkpoint is restored to the real life.

**A acquisition:** own A-held gain from birth and final A2, plus comparison to train-fitted unigram and separately fitted standard counting controls. If neither NORMAL nor CGLP beats a strong calibrated unigram by ≥0.05 bits, mark `MINI_BASELINE_NOT_COMPETENT` and withhold **mechanism promotion** even if a relative CGLP-vs-NORMAL gain appears. **Do not stop the lifetime experiment**; the other outcomes remain measured.

**B use and retention:** own B-held full-257 NLL after B1/2/3 (same-checkpoint train B NLL alongside it); old A-held NLL after B, both before/after B relative to A2; B-held before/after changed C2. Report the *original prior-experience cost* rather than calling A pretraining free. There is **no cold-B arm in this miniature mechanistic comparison**—prior V179 reuse evidence is context only, not automatically reproduced by V181.

**Corrected-A revision:** A' held corrected-label probability and top1 at changed sites, new-versus-old log odds, matched NOEDIT feedback contrast, and preservation of still-valid A' sites on the corrected raw stream.

**Repaired true retention partitions:**
1. `A_OLD`: held original A under original prefixes/targets;
2. `A'_SAME_KC`: held corrected A' with unchanged target **and exact identical complete KC code**, not merely last16 bytes;
3. `A'_ALTERED_KC`: label unchanged but different corrected-prefix KC code;
4. `A'_CHANGED`: label changed by fixed environmental correction.

A gain on the original A distribution does not establish invariance on A' altered contexts; report both. A changed two-byte byte rule is a **synthetic correction statistic**, not factual/semantic truth revision or E3 arithmetic.

**EOS and generation:** score true EOS cross-entropy/Brier/false-EOS among actual nonterminal targets and calibration under genuine record ends. On a fixed subset of **16 A held and 16 B held records per world/arm** at A2, B3 and EDIT C2, feed three real prefix bytes, then generate up to **64** own payload bytes/EOS, saving EOS rate, truncation, length distribution and sampled output. Do not require exact reproduction of an unknowable held literary continuation as a capability gate. Report only length-feasible complete exactness descriptively with the denominator. This keeps generation affordable and interpretable; full deterministic byte/EOS E1/E2/E3 tasks remain future research.

**Other countermodels:** fit and charge train-only unigram and predeclared interpolated Kneser–Ney (fixed order range1–5, discounts0.5/0.75/0.9 chosen using calibration only). Count models never supply neural answers. Report own model vs counting on all real target sites, full class support and per-record paired effects.

### 4.1 Comparison and promotion rule

For each world `w` and arm `a`, use an unweighted **four-domain** score (one domain per functional requirement, not corpus frequency):

\[
S_a(w)=\tfrac14[
L_{B,\rm held}^{B3}
+L_{B,\rm held}^{EDIT\,C2}
+L_{A',\rm SAME\_KC}^{EDIT\,C2}
+L_{A',\rm CHANGED}^{EDIT\,C2}].
\]

Always display each component, own EOS, A2 acquisition and intermediate checkpoints; a composite alone is not sufficient.

**`SELECTIVE_PLASTICITY_MINI_DEV` requires:**
1. CGLP's `S` is at least **0.03 bits below both FIXED-LOW and RMS-SHAM** in both M1 and M2;
2. CGLP's A2 acquisition is no worse than NORMAL by >0.10 bits and its corrected A' changed-target loss after C2 no worse than NORMAL by >0.10 bits;
3. CGLP's B-after-C and A'-SAME-KC-after-C are no worse than either fixed/sham control by >0.10 bits, and feedback-specific EDIT–NOEDIT correction improves the candidate's correct-new-label loss by ≥0.10 bits;
4. At least NORMAL or CGLP shows the independently reported basic acquisition competence versus the unigram reference, and all source/model/resource checks pass.

Two worlds are a **development screen**, not statistically independent confirmation. If the fixed lower-step arm or uniform sham wins, report **`LOW_STEP_SUFFICIENT`** or **`UNIFORM_STRENGTH_SUFFICIENT`** without retuning on these worlds. If the mini model itself cannot learn, report **`MINI_CAPACITY_OR_LEARNING_GAP`**, not that selective plasticity is universally false. Never increase H/T size or the number of worlds automatically.

## 5. Honest mini resource ledger and exact artifact accounting

**Per learner, float32:**
- Trainable weights+biases:37,313×4 = **149,252 B**.
- Adam moments (two per learnable scalar): **298,504 B**.
- Fixed random feedback matrices:2×(257×64×4) = **131,584 B**.
- Fixed KC→H incidence map:64×256×2 = **32,768 B**.
- Remaining model-specific state includes native sensory birth wiring and FE state, transient KC/H/T activations, optimizer scalar clocks, grad workspaces, RNG and output buffers. Measure it; no full 16-bank native writer may hide inside it.

**Experiment-wide hard limits:**

| Resource | V181 cap | V180 former proposal |
|---|---:|---:|
| Learnable parameters per learner | 40,000 (actual 37,313) | 4,229,377 |
| Physical training record presentations incl technical | **5,000** (scheduled ≤4,512) | 35,000 |
| Target presentations | **600,000** (worst ≤582,048) | 4,550,000 |
| Process-tree CPU | **900 s** | 7,200 s |
| Process wall | **1,800 s** | 14,400 s |
| Peak RSS | **1 GiB** | 2 GiB |
| New persistent artifact output | **256 MiB** | 2 GiB |
| Single-model inference-specific state | **8 MiB** | Not bounded |

At most **80** phase-state checkpoints across both worlds and four arms (birth, A1/A2, B1/B2/B3, EDIT C1/C2, NOEDIT C1/C2). Every full mutable checkpoint must serialize below **1 MiB**, with immutable fixed sensor/feedback assets referenced by pinned hashes in a separate, counted, source-locked bundle. Therefore checkpoints ≤80 MiB raw, plus a **1.5× reserved storage allowance =120 MiB**. The remaining **136 MiB** is an explicit cap for manifests, code, compact evaluation panels, birth arrays, logs and reproducibility metadata; count files **after** writing, not estimates alone. If model states cannot be recovered without secretly discarded arrays or unverifiable fixed assets, this checkpoint contract fails.

Store *minimal sufficient per-site evaluator quantities* (true-target log probability, argmax, EOS probability, corrected old/new log odds, strict KC mask, record ID and target ID, entropy) rather than dense 257-class probability tensors for every site/checkpoint. The saved-only reducer recomputes the predeclared outcomes from these stored quantities without retraining. This is a purposeful **measurement representation**, not reduced output-class inference. All 257 classes remain live in the actual neural softmax.

For runtime admission, profile the actual sparse KC→H gather and microgradient code, sensory extraction, full A/B/C roster, both outcome panels, fixed counting fits, and the limited 64-step own generation with **1.5× overhead reserve**. Do not exceed caps by silently omitting model states, replacing float32 moments with half precision, installing an external pretrained model, or shifting excess training to another machine. If resource admission fails: `MINI_RESOURCE_NOT_ADMITTED`, with source and technical receipts only. No permission to increase the budget is inferred.

## 6. No next-layer trap: exact technical audit and stop rules

1. **Source birth**: reconstruct the one-parent immutable FE-PERM/KC encoder with exact parity to original V179 native sensory outputs on all raw byte identities and representative long/short records. Check deployed model memory; the native writer bank cannot be instantiated. If 8 MiB cannot be achieved without changing the source semantics, `MINI_MEMORY_INVALID`; no scientific training.
2. **Physical sparsity:** assert H KC afferents exactly 16,384, all KCs fan out 3 or 4 times, stored learnable KC→H weights exactly 16,384 and biases64, T and O counts and total 37,313. No dense “zero-but-trainable” tensor or shadow 4M learner.
3. **Locality/numeric correctness:** finite differences against an independently implemented output-local and explicitly broadcast local-gradient rule; no autograd across learned layers. With `γ=1`, the tiny model's ordinary AdamW+local rule must be restored exactly (there is no claim of numerical parity with the bigger V179 weights). Test unequal per-record groups, true minibatch tail, single dropout-mask reuse, Gamma floor and single-neuron Adam step. Verify H/T gating affects **actual** applied move, not only raw gradients.
4. **Same-state sham and forks:** at exact shared initial tiny state, RMS-SHAM's proposed per-layer squared update energy equals CGLP's proposed energy; all arms share a single birth/source schedule but diverge independently after training. EDIT/NOEDIT within arm have identical B3 state/RNG/moments.
5. **Data only:** authenticate raw source editions and splits, source-train edits, corrected held partition, and code/target timing. If correction coverage fails, still run A/B and report incomplete revision, not a substitute corpus/threshold.
6. **No efficacy stopping:** perform all four arms through A/B/C even when NORMAL cannot beat unigram; report mini competence and relative effects separately. Only invalid source/numerics/memory/resources can stop whole work. If correction fixture invalid, C is skipped as a declared data-only limitation, not a scientific negative.
7. **Saved-only independent reducer**: validate source hashes, exact model parameter count, exact record/target exposures, phase checkpoint/optimizer/RNG integrity, original/corrected KC masks, arm-level effects, model-specific memory and CPU/RSS/output. If raw weights or literary corpora stay local under the project convention, explicitly mark aggregate results as not externally raw-reproduced.

This is a **mini feasibility and mechanism test**, not a guarantee that a 37k-parameter nonrecurrent learner can perform open-ended language, long-horizon reasoning or never-taught addition. Those remain future independent hypotheses. Do not promote from proxy losses or natural-text generation length to a comprehensive E2/E3 result.

## 7. Outcome taxonomy and version governance

- `DESIGN_ONLY` (now); `SOURCE_UNAVAILABLE`; `MINI_MEMORY_INVALID`; `MINI_RESOURCE_NOT_ADMITTED`; `NUMERIC_INVALID`;
- `CORRECTION_FIXTURE_NOT_QUALIFIED` (partial, A/B outcomes still reportable);
- `MINI_CAPACITY_OR_LEARNING_GAP`; `CGLP_NOT_SUPPORTED_MINI`; `LOW_STEP_SUFFICIENT`; `UNIFORM_STRENGTH_SUFFICIENT`; `SELECTIVE_PLASTICITY_MINI_DEV`.

Freeze V180's old DESIGN and technical source, and register its status accurately as **technical validation only, no efficacy test, RESOURCE_NOT_ADMITTED**. V181 is the materially different compact replacement; its **source, development data, and gates must be locked prospectively before any learning**. V182 is not allocated.

The MiniFly end goal is unchanged: locally trained, persistent, reusable learning with own full byte+EOS outputs and eventually never-taught arithmetic, judged by unique examples and real CPU. A good V181 result would justify **fresh confirmation** of the mini learner; it would not authorize growing the learner until it resembles a conventional large network again.

## 8. Source anchors

- [V180 stopped technical and scale report](https://github.com/ChangweiZhou/digitalbaby/blob/codex/minifly-consolidated-20261023/minifly_v180_coherence_plasticity_dev_20261102/REPORT.md)
- [V180 exact resource admission](https://github.com/ChangweiZhou/digitalbaby/blob/codex/minifly-consolidated-20261023/minifly_v180_coherence_plasticity_dev_20261102/RESOURCE_ADMISSION.json)
- [V180 technical coherence/matched-control qualification](https://github.com/ChangweiZhou/digitalbaby/blob/codex/minifly-consolidated-20261023/minifly_v180_coherence_plasticity_dev_20261102/CYCLE2.json)
- [V180 original and discontinued oversized experiment](https://github.com/ChangweiZhou/digitalbaby/blob/codex/minifly-consolidated-20261023/minifly_v180_coherence_plasticity_design_20261102/DESIGN.md)
- [V179 original full lifetime result](https://github.com/ChangweiZhou/digitalbaby/blob/codex/minifly-consolidated-20261023/minifly_v179_continuous_local_life_dev_20261102/REPORT.md)
- [V179 full 257-way local learner](https://github.com/ChangweiZhou/digitalbaby/blob/codex/minifly-consolidated-20261023/minifly_v179_continuous_local_life_dev_20261102/learner.py)
- [V179 literal sensory and correction exporter](https://github.com/ChangweiZhou/digitalbaby/blob/codex/minifly-consolidated-20261023/minifly_v179_continuous_local_life_dev_20261102/data.py)
- [V180 prospective raw corpus authentication](https://github.com/ChangweiZhou/digitalbaby/blob/codex/minifly-consolidated-20261023/minifly_v180_coherence_plasticity_dev_20261102/DATA_AUDIT.json)
