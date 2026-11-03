# MiniFly V184 — Dual-depth H→output residual deployment

**Identity:** V184-HRES-D1  
**Date:** 2026-11-02 (America/New_York)  
**Status:** DESIGN_ONLY; implementation, qualification, scientific training, confirmation, and adoption **NOT_RUN / NOT_AUTHORIZED**.  
**Anchor:** \`ChangweiZhou/digitalbaby\`, \`codex/minifly-consolidated-20261023@840f2a7b9dcd25f4262c3e17058bb1cf30139bf2\` (parent, before this design commit).  
**Owner instruction:** “The V183 is done. Activate the skill and iterate.” Interpreted as authority to prepare/register the next **design**, not to start training, modify frozen resource caps, revise past results, promote a core, or issue formal MRK-1 artifacts.

## 1. Decision rather than automatic version churn

**Question.** Does an output-local route from the *already learned* H64 state to the existing 257-way head recover useful input-conditioned information that is lost or hard to deploy through H64→T64, **while retaining** the T route and a single continuous A→B→corrected-A learner?

**Competing explanations.**

- **DEPLOYMENT LIMIT:** H contains usable conditional signal, but its serial T→output deployment is too lossy or badly conditioned. A small, fixed-strength residual H→O path should outperform the unmodified forward circuit and a matched-amplitude, direction-free route, *and* beat the learner's own constant mean forecast.
- **NOT A DEPLOYMENT LIMIT (for this package):** H lacks useful conditional information at this scale, the local H/T learning or O training is insufficient, or the tested residual parameterization is unhelpful. If residual does not meet the strict conditional/retention criteria, **reject V184-HRES-D1**; do not respond with a post-hoc strength grid, more training passes, or automatic V185.

This is a forward-path **architectural** test, not another estimator for H credit. It does **not** use a new task-specific representation, externally learned head, attention, Transformer, pretrained ELM, arithmetic roles, or any change to the output alphabet. Success on Gutenberg text would be **predictive development evidence only**; no E1, E2 semantic memory or E3 addition claim follows.

## 2. Why this comparison is justified now

- V181 built and ran an actual 37,313-learnable-parameter KC→H64→T64→257 local model, but its frozen verdict was \`MINI_CAPACITY_OR_LEARNING_GAP\`.
- V182 removed T and moved dropout to H; M1 A2-held went from NORMAL **4.582233** to SHALLOW **5.195649** bits. Dropping T entirely is not a supported repair, and that negative does not test a route that *retains* T.
- V183-R1 completed three fixed M1 lifetime arms. A2-held: NORMAL **4.582233**, HPC **4.587709**, SHUFFLED **4.615474** bits. The one-pair H signal was directionally real in qualification, but HPC failed acquisition/conditional-reference gates; \`HPC_NOT_SUPPORTED\`. The ordinary learner's own A-train mean forecast (**4.457296** bits on A-held) and A-train unigram (**4.403067**) beat its contextual predictor. This identifies a **conditional-output deficit**, not a uniquely proven causal layer.
- V176's fitted-probe signal with limited native deployment is motivation to interrogate deployment, but **it used a different learner**; it is not evidence that V181 H64 carries the same information.

Source anchors (all on the stated parent branch): \`minifly_v181_mini_local_plasticity_dev_20261102/learner.py\`, \`minifly_v182_shallow_local_dev_20261102/results/REPORT.md\`, \`minifly_v183_h_credit_dev_20261102/campaign/DECISION.json\`, \`minifly_v183_h_credit_dev_20261102/campaign/RESULTS_COMPACT.json\`, and \`minifly_v183_h_credit_dev_20261102/campaign/GENERATION_COMPACT.json\`. V183 is an **opened development world with correlated correction forks**, not independent confirmation. Its original report and raw archive are not changed by this proposal.

## 3. Exact mini architecture; one intervention

Preserve V181/V182 NORMAL's immutable 88-channel FE-PERM→5,177-KC sensor, its H sparse birth \`I\` (64 rows × 256 real KC afferents), learned weights, feedback \`F_H,F_T\`, H64/T64 ReLUs, dropout 0.1, 257 real byte/EOS categories, data/optimizer and birth seeds. No source-trained features or labels enter the representation before their legitimate feedback.

For site \`i\`:
\[
h_i=\operatorname{ReLU}(W_Hx_i+b_H)\in\mathbb R^{64},\qquad
u_i=\operatorname{ReLU}(W_Th_i+b_T)\in\mathbb R^{64},
\]
with the **same** site-wise inverted-dropout mask \`m_i\` in all three arms. Use one physical 257×64 output matrix \`W_O\`, not a second H head.

| Arm | Readout features \(\tilde z_i\) before \(W_O\tilde z_i+b_O\) | What it isolates |
|---|---|---|
| **NORMAL** | \(m_i\odot u_i\) | Exact ordinary V181/V183 forward/local rule |
| **HRES** | \(m_i\odot(u_i+\alpha h_i)\), fixed \(\alpha=0.5\) | Information in H reaching O without deleting T |
| **NORM_SHAM** | \(g_i(m_i\odot u_i)\), \(g_i=\|m_i\odot(u_i+\alpha h_i)\|_2/\max(\|m_i\odot u_i\|_2,10^{-6})\), clipped to 8 | Same-site readout magnitude without the **direction** of H |

\`NORM_SHAM\` computes \`g_i\` from **current observed-prefix activities and current dropout mask only**, before seeing \`y_i\`. It is a fair readout-*norm* sham, **not** a claim of equality of per-parameter Adam movement. Track per-site clipping, zero-denominator, realized readout norms, and per-batch actual post-Adam H/T/O movement. If any non-negligible clipping/zero-denominator prevents the declared norm comparison, label the **selectivity control unqualified** rather than hiding it or modifying the sham after a result. The sham contains one scalar of H magnitude; its direction is not available to the output.

The candidate keeps all **37,313 learned scalars**: H16,448 + T4,160 + O16,705. The extra residual is a fixed elementwise connection between existing 64 H and 64 output-feature channels; it introduces **zero** trainable weights, gates, labels or optimizer buffers. Only transient \`h,u,m,g\` differ. Count their actual runtime bytes; the source sensory arrays, fixed feedback, model checkpoint/RNG and every experimental clone are still physical state. No extra class bias, adapter, native memory bank or 257×64 second matrix is permitted.

**Learning rule.** At each observed next-byte/EOS site, predict with the arm's actual \(\tilde z_i\) *before* receiving the true \`y_i\`. Update O and its bias from the exact **local** categorical error \((p_i-\mathbf1_{y_i})\tilde z_i^\top\); update H/T by the **unchanged fixed-feedback source-local** \`F_H/F_T\` rules on the same true error. In particular, **do not** backpropagate \(\partial\ell/\partial h\) through T, O, or the skip, and do not let the skip silently alter H's learning formula. One genuine dropout draw per batch; no additional noise or extra training target. All arms use source AdamW \(\eta=10^{-3},\beta_1=.9,\beta_2=.999,\epsilon=10^{-8},\mathrm{wd}=10^{-4}\), without changed warmup or phase reset.

The test's interpretation is deliberately narrow: a negative might reflect the **unmodified H writer** not adapting to a new readout. It does not establish that any possible residual/dual-compartment architecture must fail.

## 4. Frozen causal roster and continuing lifetime

Use **M1 only**, with the exact hashed V181/V183 A/B/correction records, train/calibration/held rosters and source-selected correction IDs. Do not open a second text ecology, reselect corrections, choose more favorable records, or claim M2 is sealed.

- **A:** 160 full training records × 2 chronological passes; A2 ends with 320 records and 20,356 next-byte/EOS target applications per arm.
- **B:** 48 training records × 3 passes; B3 is a continuation of the same A-trained weights, moments, clocks and RNG.
- **C fork:** from **the identical B3 state** per arm, 24 \`EDIT\` records ×2 and 24 matched \`NOEDIT\` records ×2, with separate cloned states but no score-based choice. Corrected items and original items are data/evaluator roles, **never learner input tags**.
- Exactly **560 complete optimizer-record applications and 34,364 targets per arm** (\`320+144+48+48\` fork accounting); **1,680 records / 103,092 next-byte/EOS training targets** across 3 arms, plus separately charged bounded technical fixtures. The two C branches share an ancestor and are **not six independent ecologies**.
- Required full mutable checkpoints: \`birth,A2,B3,EDIT2,NOEDIT2\` for each arm (**15 total**), including weights, both Adam moments, optimizer clock and dropout RNG. Save complete source/manifest hashes and an identical fork-state identity receipt. Record intermediate A1/B1/B2/EDIT1/NOEDIT1 curves as **read-only summary panels**, with honest target/site counts; no hidden best-checkpoint restoration.
- Read-only full-vocabulary 257-way probabilities and true EOS site accounting at the five full checkpoints for A/B/EDIT/NOEDIT train/calibration/held panels, including all \`A'_SAME_KC,A'_ALTERED_KC,A'_CHANGED\` partitions. Save adequate per-record paired losses, log probabilities, output targets and complete-sentence EOS/generation records to reconstruct every gate, while measuring complete output volume **including the separate owner state archive and run environment**.

The experiment is not E3. An eventual fresh deterministic E1/held-relation experiment would be **separately designed and authorized** only if this informative output route passes its narrow criterion.

## 5. C0/C1 technical qualification before science

1. Freeze the exact parent file and asset hashes and \`M1\` data/split/correction IDs in a new prospective source lock. Do **not** use V182's unavailable owner archive as a fabricated V184 baseline. The main starts at the authentic shared V181/M1 birth, never V183 final weights.
2. Verify \`NORMAL\` forward and manual source-local gradients bitwise against the imported literal parent on fixed birth/technical inputs, including real EOS. Verify \`HRES(alpha=0)\` exactly reduces to NORMAL. Check the fixed \`alpha=0.5\` residual, source same-mask rule, output gradient finite difference, local H/T formula, dropout, tail lengths and the exact serialized A→B→EDIT/NOEDIT continuation. No gradient may travel through a learned layer into H.
3. At common frozen weights, compute feature norms and sham ratios for empty prefixes, normal sites, true EOS, singletons, 194/52/7-site tails and full 256-site batches. All labels remain unavailable until after prediction. If sham norm matching fails materially, report \`CONTROL_NOT_QUALIFIED\`; do not select a new gain cap from the results.
4. Check strict \`SAME_KC\` by the **entire** native sensory code, not last16 identity. Verify deterministic, identical record/position-controlled H-shuffle for the diagnostic below and ensure it leaves labels and learned state unchanged.
5. Before spending scientific exposure, benchmark actual per-arm minibatch and snapshot CPU, RSS, checkpoint and saved-site bytes. Enforce **≤40,000 trainable scalars, ≤8 MiB single-model inference state, ≤1 MiB full mutable checkpoint, ≤5,000 physical optimizer-record applications, ≤600,000 supervised training/qualification target evaluations, ≤900 CPU seconds, ≤1,800 wall seconds, ≤1 GiB RSS and ≤256 MiB complete outputs including archive/runtime/reserve**. These are *admission gates*, **not passed just because the proposal is small**. A cost failure yields \`RESOURCE_NOT_ADMITTED\`, never silent artifact deletion or automatic larger caps. Record *all* technical/failed attempts and even source-only numerical fixtures separately.

**Execution remains NOT_AUTHORIZED.** No scientific training begins without an owner launch instruction after technical admission; this design itself makes no certification claim.

## 6. Frozen evaluations and decisive fail/retain gate

**Primary at A2 (opened M1 development, no selective reporting):** compare all three arms on the same **64 held A records**, both total and payload-only own 257-class bits. Predeclared claim requires:
1. HRES held A2 loss is lower than **both NORMAL and NORM_SHAM**, with the upper end of each two-sided 95% *paired-record* bootstrap confidence interval for \(\Delta=\mathrm{HRES}-\mathrm{control}\) below zero. Do **not** bootstrap tokens as independent worlds.
2. HRES held A2 loss is also below **its own A-train mean-forecast** (no held-label fitting) and the frozen source-matched **A-train unigram**. A marginal-prior shortcut alone cannot advance.
3. A2 read-only **H-to-record alignment** control: hold trained HRES weights, T outputs and O fixed, but exchange only the residual H vectors across *different held records at the same input-byte position*, by a preregistered label-blind hash derangement where there are at least two eligible records. Score only the exchanged eligible sites, with their exact denominator, and require correct H alignment to improve loss over the exchanged H route. No shuffled H is ever trained or used to adapt the model. This diagnoses H *information*, not just additive magnitude.

**Lifetime guards (always report, even when primary fails):**
- At **B3**, A-held still-valid and B-held losses; at **EDIT2**, strict still-valid \`SAME_KC\`/ \`ALTERED_KC\` and B-held losses, against both controls. No selective A2-only claim while the later learner regresses. Without justified noninferiority margins, lack of a detected harm is **not** certified safety; a promotion case requires nonpositive paired point loss differences on these specified guards.
- **Correction specificity:** compare \`EDIT2\` and \`NOEDIT2\` starting from the exact same per-arm B3 state, using new/old log-odds *only on truly changed target sites*, with labels never presented as features; require a positive correction effect for HRES without reducing still-valid competence.
- **EOS:** separately reconstruct true-boundary EOS NLL, Brier, premature EOS, and 64-decision own 3-byte-prefix generation, its length feasibility and own **authentic EOS**, at A2/B3/EDIT2. Report every truncation. For arbitrary unseen Gutenberg text, exact literary line match and greedy own-EOS rate are **descriptive**, not deterministic complete-answer gates; they cannot justify E1. A worse EOS-NLL/Brier or inflated premature rate blocks an unqualified improvement claim.
- **Compute:** compare unique training records, physical repeated presentations, target predictions, optimizer steps, read-only evaluations, peak RAM, state/optimizer bytes and wall/CPU, not just bits. A result that uses additional externally learned priors or target-dependent encodings is INVALID.

**Disposition:** \`RETAIN_EXACT_PACKAGE_FOR_NEW_GUARD_REVIEW\` only if the full predeclared conditional and lifetime vector passes under a valid causal sham/resource/source lock. Otherwise \`REJECT_V184_EXACT_PACKAGE\`, \`CONTROL_NOT_QUALIFIED\`, \`RESOURCE_NOT_ADMITTED\`, \`SOURCE_INVALID\` or \`INFERENCE_INCOMPLETE\` as appropriate. No automatic retuning, autonomous model adoption, independent evidence promotion, arithmetic claim or successor version allocation.

## 7. Closure / research skill application

\`\`\`text
Anchor: ChangweiZhou/digitalbaby@840f2a7b9dcd25f4262c3e17058bb1cf30139bf2; V181 compact NORMAL, V183 negative at M1.
Question: Is H64 useful conditional signal suppressed by the serial T→O route?
Falsifier: HRES vs NORMAL and post-dropout-norm SHAM at fixed M1 life,
          plus own-mean/unigram and within-checkpoint H-record alignment.
What ran now: GitHub source/result inspection and design registration ONLY.
Physical learned updates now: 0. Model scientific results now: NOT_RUN.
Lesson: V183 perturbation credit did not improve the real life; source-local
        H information is not yet demonstrated to improve native output.
Disposition now: PROPOSE EXACT V184-HRES-D1, DESIGN_ONLY.
Next decision: technical qualification and mini resource admission only after
               explicit owner authorization; scientific launch separately.
\`\`\`

The integrated \`.claude/skills/minifly-research-loop/SKILL.md\` was applied as a **decision protocol**; historical MRK-1 CLI is missing, and no official MFC check, independent cross-family review or full-state archive replication is claimed.
