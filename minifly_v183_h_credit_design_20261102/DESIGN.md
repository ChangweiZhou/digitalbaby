# MiniFly V183 — H-only paired perturbation credit

**Identity:** V183-HPC-D1  
**Date:** 2026-11-02, America/New_York  
**State:** DESIGN_ONLY. Implementation, qualification and scientific training: NOT_RUN.  
**Anchor:** ChangweiZhou/digitalbaby, codex/minifly-consolidated-20261023@e3cd3b368342a72e1acac4ae7d4ada8369a86a5d.  
**Owner request:** “Can you ship the next design doc?” This authorizes this document and registration. It does not authorize training, larger budgets, model adoption or a successor run.

## 1. One question and opposing decisions

Does replacing only H's fixed random teaching signal with loss-sensitive paired neuron perturbation improve the compact learner's own conditional prediction while preserving its complete A→B→corrected-A life?

Keep the ordinary V181/V182 forward circuit, sensor, H/T/O births, dropout, optimizer and within-world exposure schedule. T retains fixed-error-broadcast learning; O retains its own categorical error rule. CGLP and the V182 shallow package remain rejected at their tested doses. No replay, changed warmup, added gate, enlarged network, arithmetic features or output-class mask belongs to this comparison.

If real perturbation credit beats NORMAL and the strength-matched shuffled-credit control, clears competence/context gates and preserves lifetime/EOS behavior, retain this exact package for separately authorized prospective confirmation. If it fails, reject this one-pair package at this dose. If real and shuffled credit improve similarly, useful information assignment is not isolated. No result automatically allocates or runs V184. The test does not uniquely resolve sensor information loss, insufficient experience, optimizer dynamics or the need for learned temporal state.

## 2. Evidence motivating the design

| Observed quantity | M1 | M2 |
|---|---:|---:|
| V182 NORMAL A-held A2 bits | 4.582233 | 5.343974 |
| V182 SHALLOW A-held A2 bits | 5.195649 | 5.872995 |
| NORMAL constant mean-forecast comparator at A2 | 4.457296 | 5.243308 |
| H source-local versus exact gradient cosine | −0.029970 | 0.000418 |
| One paired perturbation estimate versus exact cosine | 0.143660 | 0.180860 |

The last two rows average four fixed256-site A-train batches at NORMAL A2 in each already opened world. These were frozen-state calculations with zero optimizer updates, using float64 perturbation/reference arithmetic. They support a directional lead at those states, not operative learning or generalization. Averaging eight noise pairs produced better alignment but is NOT the proposed candidate. Constant mean forecasts average the model's A-train probabilities; no held labels fit that comparator. Marginal-frequency recentering did not establish useful conditional learning either.

Original JSON and portable copies of the two read-only scripts are in evidence/. Provenance records original and adapted hashes. Only path resolution is adapted; the portable copies are not represented as rerun. V183 itself has not been implemented or qualified.

## 3. Unchanged forward circuit and learning constants

Use the literal ordinary learner in minifly_v182_shallow_local_dev_20261102/parent_learner.py:

| Learned block | Weights | Biases | Total |
|---|---:|---:|---:|
| KC5177→H64, physical fan-in256 per H | 16,384 | 64 | 16,448 |
| H64→T64 | 4,096 | 64 | 4,160 |
| T64→O257 | 16,448 | 257 | 16,705 |
| **Total per arm** | **36,928** | **385** | **37,313** |

One immutable88-channel FE-PERM frontend drives the original5,177-KC sensory circuit. H/T use ReLU; dropout0.1 remains on T before O. Evaluation disables dropout. All257 classes remain live: payload0–255 and authentic EOS256. Retain FH/FT for parity; FH is unused by the perturbation H writers. No additional learned feedback matrix or pretrained model supplies features or answers.

All arms start from the same V181|M1|birth, incidence, feedback and dropout births. Frozen V182 states are qualification instruments only, never scientific learner pretraining. AdamW remains lr0.001, betas0.9/0.999, eps1e−8, decay1e−4. Batch at most256 real targets chronologically, including each actual tail. Preserve weights, both moments, clocks and RNG through each life. No added warmup, changed clipping, best-checkpoint restore, scheduled reset or phase cue. The learner sees no arm/world/corrected-item labels. No learned-layer BP/BPTT enters the candidate; exact derivatives are isolated read-only instruments.

## 4. H-only paired credit, fully specified

Compute the ordinary unperturbed batch first: H preactivation a, H/T states, one actual inverted T-dropout mask m, logits and probabilities. Derive T/O gradients from that base path exactly as NORMAL. Freeze current weights and m while probing H.

For n target sites, set

\[
\sigma=0.01\max\left(\sqrt{\frac{1}{64n}\sum_{i,j}a_{ij}^{2}},10^{-3}\right).
\]

This depends only on observed-prefix activity and current weights. Freeze sigma for both counterfactuals and their numerical reference: do not differentiate through its RMS dependence or recompute it on perturbed activity. Conditional on frozen sigma, the antithetic Gaussian estimator targets the Gaussian-smoothed per-site loss gradient with respect to preactivation. Equality with the unsmoothed derivative, or a derivative through sigma(a), is not claimed. Numerical finite-difference identities must match this fixed-sigma target; unsmoothed gradient cosine is only a directional diagnostic. Draw independent standard Gaussian ξ_ij for every site/H-neuron pair. Use the same noise with opposite signs in the two downstream evaluations, and fresh noise at the next batch. One noise vector shared across all sites is forbidden.

Evaluate H→T→O at a+σξ and a−σξ, including H ReLU and the exact same T-dropout mask. No additional sensory observation occurs. Let ℓ_i± be each site's categorical NLL in natural units using the already observed next byte/EOS. Then

\[
q_i=(\ell_i^+-\ell_i^-)/(2\sigma),\qquad
\widehat\delta^H_{ij}=q_i\xi_{ij}/n,
\]
\[
G^H_{jr}=\sum_i\widehat\delta^H_{ij}x_{I_{jr},i},\qquad
G^{Hb}_{j}=\sum_i\widehat\delta^H_{ij}.
\]

Do not multiply by another H ReLU derivative: its effect is already inside the preactivation perturbation's loss difference. Do not replace per-site q_i by a batch-mean loss difference. Targets enter the teaching calculation only after prediction, never the sensory state or representation.

The two auxiliary downstream evaluations use temporary float64 arithmetic with stored float32 weights cast read-only. The base prediction and T/O gradients remain float32. Cast final H gradients to float32 before unchanged AdamW. This deliberately follows the exploratory probe's numerical regime; auxiliary float64 computation is part of the algorithm and fully charged. It creates no second learned network. Precision and actual stored-weight effects remain to be qualified.

Scientific learning uses exactly ONE pair per batch. Two independent draws at frozen states are permitted only for qualification below. Qualification uses V183|qualification|WORLD|STAGE|BATCH|DRAW|noise and the corresponding |shuffle namespace, with the same SHA256 seed conversion. No noise/rate grid, outcome-selected seed, eight-pair averaging or adaptive rescue is allowed.

## 5. Arms, strength matching and RNG

| Arm | H teaching | T/O teaching | Loss evaluations per target |
|---|---|---|---:|
| NORMAL | Literal fixed-FH broadcast | Ordinary base-path rules | 1 |
| HPC | Paired estimate above | Same base-path rules | 3 |
| SHUFFLED | Same probes, q_i cyclically shifted across sites | Same base-path rules | 3 |

SHUFFLED draws a uniform nonzero cyclic shift from1…n−1 using a separate RNG. It preserves q's batch values while breaking their assignment to perturbations and eligibility. For n=1, shuffled credit is zero; log the case and retain the ordinary moment/decay rules. Never replace a tail by a more convenient record. Pinned M1 tail sizes are A=194, B=52, EDIT=NOEDIT=7 target sites; none is a singleton. Verify these before launch.

Match actual H-writing strength at SHUFFLED's current state. Calculate its shuffled AdamW H/Hb proposal u_S and a counterfactual unshuffled HPC proposal u_P from the SAME current weights, moments and clock. Commit only shuffled moments. Multiply the shuffled H/Hb movement, including bias and decoupled decay, by the one scalar ||u_P||₂/||u_S||₂. Do not change T/O. If both norms vanish, apply zero. If ||u_S||₂<10⁻²⁰ but ||u_P||₂≥10⁻²⁰, stop as SHAM_STRENGTH_INVALID; no clamp or substitute direction is allowed.

This matches proposed squared H movement at a common state, not cumulative movement after paths diverge. Optimizer-history effects are not uniquely isolated. HPC versus SHUFFLED is the information-assignment comparison; NORMAL is the lower-compute practical incumbent. Do not call all three arms compute-matched.

Noise/shuffle namespaces: V183|M1|H-credit and V183|M1|H-shuffle. Seed from first8 SHA256 bytes, unsigned big-endian modulo2^63−1. Both are separate from dropout. HPC/SHUFFLED consume matching noise streams by batch shape. Serialize every generator needed for exact continuation. No perturbation is used at inference.

## 6. One complete development ecology

Use V182's exact authenticated M1 parser, records, split manifest, IDs and train-only correction:

| Source | SHA256 |
|---|---|
| austen-persuasion.txt | cec097657cbfeb62aafe178973f514fa07ab66a5e28ea025ebd5d51ca870e40f |
| austen-sense.txt | 57b169c7cfa502896dee25bd66616f61cc1d21084caf31f015176ee46fda9718 |

A: first160 eligible train records, next32 calibration, last64 held; two passes. B: first48 train, next32 calibration, last64 held; three passes. C: the same24 selected original A IDs and noncascading th:e→u edit, two passes each of EDIT/NOEDIT. Preserve every record hash and the original selected-ID list; do not reselect the correction using outcomes.

Each arm has560 optimizer-record applications: A320+B144+EDIT48+NOEDIT48. EDIT and NOEDIT begin at identical B3 weights, moments, clock and ALL RNG states, then continue separately. There are208 distinct original A/B training records; C reuses A source IDs with changed or unchanged feedback. Forks/repetitions are not independent worlds or new distinct source examples.

M1 is opened development text. M2 receives NO V183 scientific lifetime in this design; frozen M2 states are read-only qualification only. M2 is also opened, not sealed confirmation. This one-world scope preserves the original within-world lifetime and all three controls under the existing conservative budget. Any later new-world study needs a separate prospective protocol and authorization.

## 7. Qualification and admission — all NOT_RUN

Authenticate the complete source/input closure, every split/correction ID, canonical native arrays and sensor digest7e0b78ae71cd9f508e9069018ec2343a88ab8bfd9263067506ee6eb3fae4962b. Preserve V182's explicit FMA receptor-birth recovery on x86 and pinned NumPy2.4.6/Torch2.14.0 runtime. Equal file hashes alone do not prove equal runtime births.

Numerical checks must verify: no future target or extra sensory advance; independent site noise; dropout reuse; frozen weights until all batch gradients exist; q sign and1/n normalization; implicit H ReLU derivative; untouched T/O source rules; float64 auxiliary versus float32 stored weights; n=1/tails; SHUFFLED same-state post-Adam movement matching; zero-credit liveness; full serialization and next-update equality. Exact autograd is an independent numerical instrument only. Cap all other technical learning and numerical fixtures together at32 complete record-equivalents before their up-to3× feedback charge, including synthetic fixtures, duplicate executions and reference updates; also cap their target evaluations at12,384. Do not make uncharged synthetic or gradient-only fixtures. All applied technical optimizer updates remain separately counted.

Frozen-state assay: M1/M2 NORMAL birth and A2; first four chronological256-site A-train batches each. Use two fixed independent noise draws per frozen batch to measure variability, without applying a persistent update. At most one base reference, four perturbed losses and one finite H-only counterfactual movement evaluation per site. Compare raw real/shuffled credit with exact diagnostic derivatives and inspect actual post-Adam proposals. NORMAL-history A2 moments are advisory: they are not HPC optimizer history and cannot veto it by themselves.

M1 admission requires mean one-pair cosine≥0.05 and≥0.05 above SHUFFLED separately at birth and A2, using every declared batch/draw without seed selection. At each M1 stage, at least6 of the8 declared batch/draw one-pair cosines must also be positive; do not satisfy the gate with one favorable outlier or an averaged eight-pair gradient. Otherwise CREDIT_SIGNAL_NOT_QUALIFIED closes this exact one-pair proposal at this scope. M2 is descriptive and does not automatically veto M1 development. This is a direct check of the proposed H signal, not an unrelated reference-network prerequisite. Never add more pairs to pass.

The first1024 A-train sites touch21 M1 and32 M2 source-record/batch units. Two checkpoints ×six evaluations conservatively charge636 record-equivalents and24,576 target evaluations. Other numerical/parity/update fixtures share the32-record technical allocation; if more are needed, stop for admission.

Profile the whole future roster without scientific fitting: actual model/fixed arrays, float64 buffers, additional RNG, all30 checkpoints, panels, generation, reducers, review and packaging, with1.5× reserve. Require measured/projected CPU≤900s aggregate, wall≤1,800s, peak RSS≤1GiB and complete new output≤256MiB INCLUDING a separate reproducibility archive. Single deployed model-specific inference≤8MiB; full mutable checkpoint≤1MiB; learned scalars≤40,000. Additional RNG/checkpoint/buffer overhead is not yet measured. A failed admission is RESOURCE_NOT_ADMITTED; no larger allowance or discarded state is automatic.

## 8. Exact exposure ledger

Extra counterfactual paths receive no new sensory byte or independent target information. Charge their repeated supervision and compute explicitly.

| Scope | Optimizer-record applications | Conservative record-equivalents | Learning/admission target evaluations |
|---|---:|---:|---:|
| M1 NORMAL full life | 560 | 560 | 34,364 |
| M1 HPC full life | 560 | 1,680 | 103,092 |
| M1 SHUFFLED full life | 560 | 1,680 | 103,092 |
| Frozen-state assay | 0 | 636 | 24,576 |
| Other technical fixtures, maximum | 32 | 96 | 12,384 maximum |
| **Maximum total** | **1,712** | **4,652** | **277,508** |

Hard caps:5,000 conservative learning/admission record-equivalents and600,000 supervised learning/admission target evaluations. The retained ten-panel roster evaluates33,307 target sites and544 record views per snapshot:999,210 read-only target evaluations and16,320 record views across30 snapshots. Generation adds at most18,432 model decisions and864 observed prefix bytes. Harvest the mean-forecast summaries from the existing HPC A2 A-train panel; if separately reevaluated, charge another10,178 read-only forward sites. These are outside the learning/admission target cap because they cannot update or select learner state; they are fully inside the CPU/storage ledger. Maximum scheduled learning/admission plus read-only/generation site evaluations are1,295,150 before any separately repeated mean-forecast panel. Record actual early-EOS reductions and every extra evaluation. The prior opened exploratory checks cost approximately26.486450 CPU seconds with zero optimizer updates; carry30CPU seconds into the900-second ledger. Charge every new failed attempt, technical reference, profile, reducer and package. Explicit conservative allowances cover unmeasured portions; no free historical learner training is credited.

Same source history, unique information and optimizer-update count across arms; three scientific lives in one ecology, six correlated C branches and30 full checkpoints. No additional arm or world is included.

## 9. Outcomes and fixed gates

At birth, A1/A2, B1/B2/B3, EDIT1/2 and NOEDIT1/2, save same-checkpoint train/calibration/held own257-class payload/EOS NLL and accuracy. Score ORIGINAL_A_HELD_ALL and corrected A' SAME_KC, ALTERED_KC and CHANGED separately. ORIGINAL_A_HELD_ALL contains obsolete targets after C and is then diagnostic only; it cannot be a still-valid-memory gate. SAME_KC requires identical COMPLETE sensory code, not merely the last16 bytes. Report true EOS NLL/Brier and premature EOS.

At A2, B3 and EDIT2, generate from the same16 A-held+16 B-held three-byte prefixes, maximum64 own outputs. Preserve authentic EOS, truncation, length-feasible exactness and output traces. Exact literary continuation is descriptive, not an advancement or arithmetic test.

All loss comparisons below use the original pooled mean over actual next-byte plus authentic-EOS target sites, with the original fixed site denominators (no per-record reweighting). ALL primary A2 gates are required in M1:

1. HPC A-held loss is≥0.05bits below BOTH NORMAL and SHUFFLED.
2. HPC beats original train-fitted A-unigram4.403066543210908bits by≥0.05bits, on exactly the original sites.
3. HPC beats its own constant mean forecast by≥0.05bits. At A2 average unperturbed HPC probabilities over A-train sites to define q_y, then score constant q on A-held. Keep all257 positive-support classes; compute log q with streaming log-sum-exp across A-train log probabilities to avoid underflow without probability clipping or class masking. Do not fit q to held labels. This requires useful conditional prediction beyond the model's own marginal forecast.

ALL lifetime/EOS gates are required:

- B-held B3, B-held EDIT2, A'-SAME_KC EDIT2 and A'-CHANGED EDIT2 are each no worse than either comparator by>0.10bits.
- Before correction, HPC ORIGINAL_A_HELD_ALL degradation A2→B3≤0.10bits. After correction, its A'-SAME_KC EDIT2 versus its own A2 score on the exact same corrected-prefix SAME_KC sites≤0.15bits; its B-held B3→EDIT2 degradation≤0.10bits. Do not gate post-C retention on original targets superseded by the edit. ALTERED_KC is reported separately as a changed-context generalization diagnostic. These are distribution-mean guards, not every-item retention proofs.
- At changed held sites, HPC NOEDIT2 loss minus EDIT2 loss≥0.10bits: synthetic statistical-edit adaptation, not semantic revision.
- HPC true EOS NLL is no worse than either comparator by>0.25bits: use A-held/B-held at A2 and B3, then corrected EDIT-held/B-held at EDIT2. This follows the actual corrected stream after C.

Display every signed component; composites cannot override failures. Report the historical four-domain mean descriptively. Kneser–Ney A-held2.639123951961054/B-held3.3354589069803127 remain authenticated historical evaluator references with prior fitting cost; never learner answers. No new count grid or decoder fit.

HPC_DEV_GAIN requires every efficacy gate. HPC_RELATIVE_GAIN_ONLY means both arm-comparison gains pass but any competence/context/lifetime/EOS gate fails. Otherwise HPC_NOT_SUPPORTED. Additionally record NO_INFORMATION_ASSIGNMENT_ADVANTAGE if SHUFFLED improves A2 over NORMAL by≥0.05bits while HPC's A2 improvement over SHUFFLED is<0.05bits. This is an explicit failure to isolate the registered assignment advantage, not a claim of mathematical equivalence. Keep CREDIT_SIGNAL_NOT_QUALIFIED, NUMERIC_INVALID, SHAM_STRENGTH_INVALID, SOURCE_UNAVAILABLE, RESOURCE_NOT_ADMITTED and NOT_RUN distinct from efficacy failures.

Once validly launched, finish the scientific roster despite weak acquisition. Stop early only for source/numerical/resource invalidity; preserve partial receipts and withhold an efficacy verdict. No changed rate, dose, noise, seed, averaging, added pass or preferred checkpoint after observing scores.

## 10. Reproducibility, review and limits

Before later fitting, lock the candidate/control implementation, exact runtime births, qualification and complete data/source closure. This DESIGN_SOURCE_MANIFEST.json is not the future execution lock. Save all weights, both moments, clocks and dropout/perturbation/shuffle RNG. Preserve sufficient per-site true-label log probability, EOS probability, record/target IDs, original/corrected labels, strict KC masks, generation and fork receipts. Recompute EOS loss and all gates from site arrays. For the own-mean-forecast gate, additionally save each A2 A-train record's257-dimensional log probability-sum vector (log sum of its site probabilities per class), its true site count and input hash. A reducer combines all160 vectors with log-sum-exp, divides by the total site count, and recomputes the constant comparator on saved A-held labels. Save the resulting log q as a check, not a fitted output head. The record-class summaries cost about329kB per arm in float64 and must be included in the output census. Keep full states/panels durable; public scalar summaries alone are insufficient. Declare public omissions and archive hashes.

Read-only scoring preserves model/optimizer/RNG. A saved-only reducer applies zero optimizer updates. Review source locality, timing, sham strength, full lifetimes and cost. Same-family criticism is described honestly; cross-family certification remains pending. The historical MRK-1 executable remains unavailable. Saving a document or obtaining positive gradient cosine is not a formal checker PASS.

This remains predictive development. It does not establish deterministic complete-answer E1, semantic E2, never-taught arithmetic E3, biological validity or broad cognition. Learned temporal memory remains open. Node perturbation has documented variance/stability limits; no general success guarantee is assumed.

## 11. References

- [V182 negative result](https://github.com/ChangweiZhou/digitalbaby/blob/e3cd3b368342a72e1acac4ae7d4ada8369a86a5d/minifly_v182_shallow_local_dev_20261102/results/REPORT.md)
- [Literal ordinary learner](https://github.com/ChangweiZhou/digitalbaby/blob/e3cd3b368342a72e1acac4ae7d4ada8369a86a5d/minifly_v182_shallow_local_dev_20261102/parent_learner.py)
- [V181 input manifest](https://github.com/ChangweiZhou/digitalbaby/blob/e3cd3b368342a72e1acac4ae7d4ada8369a86a5d/minifly_v181_mini_local_plasticity_dev_20261102/campaign_20261102/DATA_AUDIT.json)
- [Hiratani et al., NeurIPS2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/cf38eb1549024cce4b3d2c1bb87a6c27-Abstract-Conference.html)
- [Latham, node perturbation and minibatch variance](https://www.gatsby.ucl.ac.uk/~pel/tn/notes/node_perturbation.pdf)

Publication disposition: V183 allocated as DESIGN_ONLY; no implementation/training or adoption is authorized or claimed. V184 is not allocated.
