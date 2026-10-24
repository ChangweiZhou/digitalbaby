# MiniFly: source-resolved human-learning mechanism audit (revised, 2026-10-24)

**Status: documentation-only audit; no new learner, benchmark run, validation, scientific advancement or adoption.** This document **supersedes** the preliminary educational-psychology crosswalk in [MEMO.md](MEMO.md). It corrects the first and second Deep Research reports' central failure: they missed Claude's actual **150-entry human-learning catalog**, conflated teaching schedules with computational learning machinery, and did not reconcile the more recent V143/V144 results.

**Question:** From the human learning literature, what computational ingredients are *not yet demonstrated in MiniFly*, and what experiment should distinguish them? This is **human learning** as biological/cognitive inspiration, not an audit of Drosophila learning. FlyWire/Full151 is an experimental substrate, not an assertion of brain homology.

## 1. Source identity: what was missing from the earlier research

There are **two distinct Claude primary sources**, which must not be conflated:

1. **Claude's actual 150-item human-learning catalog**, committed on 2026-10-24: [human_learning_mechanisms_20261023/CATALOG.md at immutable Claude commit 4c12621d](https://github.com/ChangweiZhou/digitalbaby/blob/4c12621d0804d2495c2eacaebf4e17aaeb994771/human_learning_mechanisms_20261023/CATALOG.md). This is the central item the Deep Research sessions failed to locate. Its categories are CELL 19, SYS 23, SLP 6, RL 12, CTRL 15, CONC 12, PERC 9, MOT 7, LANG 12, NUM 12, EDU 19 and MOD 4: **150 items total**. Claude reports evidence grades A=16, B=81, C=40, D=10, E=3; 67 contested; **50/150 compiled from background knowledge without fresh source retrieval**, 100 with a searched source. Evidence grades were based mostly on abstracts/snippets, **not** a full-text systematic review.
2. **Claude's V6 architecture proposal:** [CLAUDE_V6_ORIGINAL.md](../../minifly_v6_r1_dev_20261023/design/CLAUDE_V6_ORIGINAL.md), dated 2026-10-23. It discusses learned working-memory gating, sequence chunking, competitive queuing, structure/content separation, procedural-to-episodic transfer, offline consolidation and local credit, with supporting references and speculative performance estimates.

The consolidated branch already carries **all 150 original rows**, their sources and grades, and proposed dispositions in the [V144 human-catalog crosswalk](../../minifly_v144_addition_readiness_design_20261024/MECHANISM_CROSSWALK.csv). The independent [254-version historical register](../FULL151_VERSION_CENSUS_20261023/MECHANISMS_254.md) is an inventory of **program/configuration versions**, not 254 distinct human mechanisms. The separate historical 108-/136-row catalogs and 53 experiment-delivery census overlap with it; these counts are **not additive**. See [census and interpretation](../FULL151_VERSION_CENSUS_20261023/MEMO.md).

This audit does **not** claim to have re-read all 150 papers in full, or to have independently reproduced all 254 historical experiments. It uses the two source-level Claude documents, the complete V144 row crosswalk, original implementation/results reports, and selected primary scientific papers linked below. A catalog entry's presence is not implementation, and a negative experiment at one recipe is not a family-wide impossibility theorem.

## 2. Fixed objective and information contract

The MiniFly target is **learning a generalizable procedure that produces complete autonomous byte sequences and its own EOS for previously unseen addition relations**, while preserving OLD knowledge and improving **total presentation/feedback/CPU efficiency**. End-to-end E3 evidence requires withheld relations, not just alternate presentations of taught relations. Record both directions of each unordered operand relation in the same split. Preserve gradually increasing feedback, legitimate observed answer bytes/EOS during supervised warmup, the scalar-only distinction where relevant, actual own-prefix generation, natural stopping and zero hidden answer cues.

Architecture restrictions in current designs: **no arithmetic-specific input parser, number/operand/carry encoding, externally supplied length, hard-wired sequential pointer, attention, Transformer, or inherited ELM solver**. Generic state updates and learned gates may be investigated, with their own ablations. Evaluator-side exact-gradient/BPTT references are **not** native-fly results; a teacher's expected bytes at a phase where only scalar feedback is permitted would be leakage.

A **single correct parity byte** is not ordinary addition; 100% EOS within a cap is not a correct answer; a replayed previously taught fact is not unseen-rule generalization; an externally held answer prefix is not autonomous output.

## 3. What the experiments actually established (latest first)

| Evidence | Exact outcome | Permitted inference |
|---|---|---|
| [V144 three-cycle development, 2026-10-24](../../minifly_v144_dev_20261024/REPORT.md) | Native V143 D2 learned-bank parity 82.35% vs frozen 33.09%; actual summed system 69.85% vs 48.90%. Informative-bit absolute residual 0.4381 (passes 10–20) to 0.04267 (125–150); late feature updates ~17× decay. Hidden effective rank 17.75 -> 2.95. | Nonzero native learning signal persists; strong *total upstream teaching starvation* not supported in this recipe. Shared codes increase, but that does not itself establish causally harmful collapse. **No native ordinary addition efficacy** tested. |
| [V144 separate 64-pass evaluator reference](../../minifly_v144_dev_20261024/REPORT.md) | 40 training / 15 withheld unordered single-digit orbits, generated full payload+EOS. BP taught 77.5%, held complete **6.67%**; DFA taught 81.25%, held **3.33%**; frozen features taught 15%, held **6.67%**. EOS appears within 8 steps 100% for all modes, but at correct held length BP 86.67%, DFA 73.33%, frozen 56.67%. | At this **one developmental dose/configuration**, representation learning aids taught acquisition but not transfer. Neither EOS-only nor proven global absence of representational capacity explains the entire failure. The scheduled broader capacity screen has **not** been run. |
| [V143 LEARNED-BANK, source-pinned Claude report](https://github.com/ChangweiZhou/digitalbaby/blob/4c12621d0804d2495c2eacaebf4e17aaeb994771/minifly_learned_bank_v1_20261024/REPORT.md) | Four-world one-bit parity: learned bank ~91.56% mean balanced held; frozen ~46.72%; label-shuffled ~56.30%; own combined system ~69.49%. Predeclared pass 3/4 worlds. | Positive evidence for **task-aligned learned addressing in a bounded parity task**, not addition, sequence computation, memory under interference, or own EOS. Claude later corrected its speculative starvation and overfitting interpretations. |
| [CLS v0 report, 2026-10-18](../../minifly_cls_dev_20261017/REPORT.md) | FE-PERM, homeostasis, adaptation reset, residual teaching allowed fast-memory taught recall ~84–88% after 24 passes; CLS replay held previously trained addition at **70–85%** after multiplication, vs **4–6%** with no replay. Never-taught sums **0–20%** in that combined regime; slow-core BPTT best unseen 13%, local e-prop **0%** in its tested G3. Confidence arbitration sometimes weaker than the stronger subsystem. | Acquisition and interference protection are distinct from abstraction; **replay and arbitration already tested**. Slow recurrent credit was not solved by naming an eligibility trace. |
| [V6-R1 execution](../../minifly_v6_r1_dev_20261023/REPORT.md) | 2 worlds, 8 histories, 34,816 applications; OWN OLD/NEW/held exact all 0, own EOS 0, held truncated 100%. Saved-state diagnostic finds prompt effect tiny relative to tape noise (input/noise effect about 1:70–1:87 by its measure). | The tested stochastic recurrent controller failed acquisition; one-hot sensory birth scaling is a plausible proximate problem, not a mathematical disproof of recurrent procedures. The [R2 input-fan-in scaling comparison](../../minifly_v6_r1_dev_20261023/NEXT_DESIGN.md) is **proposed, not executed**. |
| [Historical 254 review](../FULL151_VERSION_CENSUS_20261023/MEMO.md) | Includes allocation, stability/plasticity, context routing, fast/slow memory, replay, uncertainty and many exact negative configurations, plus limited older synthetic E3 examples that do not transfer to the current native learner. | Claude V6's claim that all 254 are storage-only is **too categorical**; nevertheless the register does **not** establish a reusable learned arithmetic procedure. |

Important source/provenance limit: V143's authoritative implementation/report and the 150-item Claude catalog are accessible at the immutable Claude commit above; their entire directory is **not copied into the consolidated branch**. The V144 crosswalk explicitly preserves the original 150 catalog rows. Do not misread absence of the original directory from the consolidated HEAD as absence from Git history.

## 4. Reconstructing and qualifying Claude's V6 proposal

| Claude V6 theme | Human/cognitive basis | Project evidence and corrected reading |
|---|---|---|
| Neural reuse in arithmetic | Development recruits/repurposes existing brain circuits; NUM-01 and nearby catalog items | **Motivation only**, not permission to preinstall number sense, carry or arithmetic operators. |
| Counting/procedure -> fact retrieval | Qin et al. 2014 finds developmental association with hippocampal-neocortical reorganization; NUM-04 | Supports a human developmental *phenomenon*, not a guarantee that confirmed model guesses may become teaching labels or a ready-made software controller. |
| Corticostriatal learned working-memory gating | Frank & Badre 2012 computational model; CTRL-01/02 | A candidate **generic learned transition/gate**; no observed MiniFly evidence that it currently maintains task-relevant latent variables over unseen sequences. V6-R1 tried stochastic gated recurrent state but failed its input/learning interface. |
| Serial order and start/stop chunking | CTRL-10/11; Jin & Costa 2010 mouse action-sequence evidence | No verified internally learned chunks in MiniFly. A generic EOS port is not the same thing as a discovered reusable macro/action chunk. |
| Structure/content factorization | SYS-14 Tolman-Eichenbaum Machine, CTRL-07 variable binding, CONC-05 compositional generalization | **Core architectural hypothesis**: a task-independent shared transition operating on data-dependent state. TEM itself is a trained computational model, not proof of a local-rule solution. |
| Competitive queuing | CTRL-10 and sequence-order literature | A possible generic order representation; do not treat a fixed recency gradient or hidden reverse pointer as a learned human mechanism. Defer a new readout if learned recurrent transition already provides order. |
| Sleep/replay -> rule extraction | SLP-04 human insight studies; SYS-15 CLS theory | Replay can change retention; the tested CLS **did not** establish rule induction. Sleep/insight association does not imply replay universally discovers rules. |
| Carry/intermediate working state | CTRL-03/04/12 and NUM-09 | State must be **learned from raw bytes**; a preassigned carry bit, 200-entry lookup table or digit-wise window is an arithmetic prior. Claude's speculative 10^3–10^4 vs 10^6–10^7 sample predictions have no MiniFly evidential basis. |

Adjacent citations: AuGMEnT (CTRL-08) explicitly uses an **attention-related** synaptic-tag mechanism and is not a plug-in compatible with the current no-attention constraint. E-prop (CELL-09) is an algorithmic route to online recurrent credit but performance on one class of recurrent networks is not evidence for this native KC/gate system. Latent causes (RL-10), successor representations (SYS-16), and schema formation (SYS-05/08) are already *in Claude's catalog*; none is a newly discovered omitted category.

## 5. An accurate gap ledger

Here **"gap" means an experimentally undemonstrated capability or missing operational specification**, not absence of a named phenomenon from the 150-row catalog.

| Capability-level gap | Catalog coverage and existing attempt | What is missing in the executed learner | Priority |
|---|---|---|---|
| A. Effective raw-byte state conditioning | PERC-01/07, context versions, CLS FE-PERM; V6-R1 noise dominance | Correct input/signal scale for **that** stochastic state controller, plus showed impact on own answers | Engineering preflight, not a new human mechanism |
| B. Stable learned intermediate variables | CTRL-01/03/04/07; V6-R1 recurrent gate; CLS slow core | Reliable end-to-end learned retention/update of task-relevant state across positions, without operand/carry slots | High **after** feasibility localization |
| C. Position-independent reusable transition | CONC-05, SYS-14, RL-10; multiple failed arithmetic learners | Same learned operation reused across arbitrarily positioned events, verified by held relations, order/distractor/length interventions | High |
| D. Effective local temporal credit | CELL-09, CTRL-08; e-prop in CLS, V6 score traces | A causal, low-variance route from **permitted** answer feedback to relevant upstream state changes. Credit availability is not credit usefulness | High if exact-gradient reference succeeds and local rule fails |
| E. Endogenous stop/sequence boundaries | CTRL-10/11; generic byte/EOS ports including V144 | Accurate own EOS tied to correct content, not merely emitting any EOS or a forced length | Medium; present EOS bottleneck is **not isolated** |
| F. Procedure -> verified fact and control arbitration | NUM-04/10, SYS-01, RL-03; CLS replay/confidence selector | A **competent procedure** first, then use only legitimately verified observations for episodic updates; selector learned prospectively | **Deferred** |
| G. Retention without catastrophic interference | SYS-01/15, CELL-16; multiple historical variants; CLS G4 | Preserve abstraction and OLD under later learning with fixed memory/event/CPU budget | **Deferred until B/C show acquisition** |
| H. Agency and hypothesis testing | RL-07, LANG-10, EDU-17; passive addition protocol lacks controllable information requests | Task-environment action affordances and charged new observations; not currently an internal algorithm one can fairly score | Out of scope of immediate addition experiment |

**No credible basis exists for declaring entire topics such as retrieval practice, spacing, attention, predictive coding, motivation, confidence, replay, metaplasticity or abstraction "missing" from Claude's catalog.** They are explicitly represented as EDU-01/02/03, PERC-09, PERC-03, RL-07, EDU-17, SYS-15/SLP-02, CELL-03/16, CONC-04/05. Some have project implementations or analogous operations; many do not. In particular "attention" is **excluded by experiment design**, not an oversight to fix by re-enabling a Transformer.

## 6. Minimal candidate mechanisms with falsifiable interfaces

These are **hypotheses for later single-factor tests**, not a claim they are absent in name, novel to neuroscience, authorized for adoption or runnable as-is. Every proposed extra state/edge and every training token, query, replay and CPU-second must be charged.

### H1. Input-scale repair before architecture growth

Use the already-proposed [V6-R1-R2](../../minifly_v6_r1_dev_20261023/NEXT_DESIGN.md) one-hot fan-in correction: preserve symbol-exchangeable weight columns, but avoid scaling a one-hot input block as though hundreds of input dimensions are simultaneously active. Compare ORIGINAL / RESCALED / RESCALED_FROZEN / RESCALED_NO_OLD / RESCALED_SH_OLD with the exact same native core and byte feedback. Measure prompt-to-tape sensitivity *and* entire own generated answers. Improved KC separation without an E1/E3 gain remains a **diagnostic**, not a success.

Competing explanations after conditioning is fixed: high-variance score learning, native state interference, irrelevant latent dynamics, and poor output training. Do not infer that noise rescaling alone solves these.

### H2. Generic learned gated state with shared transition

A candidate computation (coordinates are arbitrary, NOT digits, operand roles or carry bits):

~~~
x_t = generic_raw_byte_state(observed_byte_t)
g_t = sigmoid(G_x x_t + G_h h_(t-1) + b_g)
u_t = tanh(W_x x_t + W_h h_(t-1) + b_h)
h_t = (1 - g_t) * h_(t-1) + g_t * u_t
~~~

Weights are **shared at all time steps**. All gates are learned from legal feedback, not set from '+', '=', digit class, or an answer-format oracle. A fixed small state may be allowed by an approved design; no address-by-content attention/Transformer or hand-coded instruction pointer is implied. This does **not** become a new mechanism merely by changing V6-R1's stochastic gates to deterministic ones: retain existing V6-R1 as a specific tested parent and state exactly how any candidate differs.

**Diagnostic tasks** may test byte order/retention as E0/E1, but the **primary scientific endpoint stays full unseen addition with own EOS**. Required lesions: fixed gates, shuffled feedback, fixed transition, matched state dimension, matched warmup, same-exposure comparator. Test that the learned state reacts to counterfactual prefixes and retains earlier task-relevant information after distractors. If learned gating improves taught but not held answers, reject the rule-generalization claim.

### H3. Local eligibility with meaningful temporal credit

Generic three-factor form:

~~~
e_ij(t) = lambda * e_ij(t-1) + local_pre_post_signal_ij(t)
Delta W_ij = eta * sum_t [delta(t) * e_ij(t)]
~~~

Here delta(t) is a **legally available** observed-token prediction error (during labeled warmup) or a properly defined scalar reward/prediction-error signal (in scalar-only conditions); these are **not interchangeable**. An eligibility trace is not automatically the gradient of a recurrent stochastic model. V6-R1 already implemented Gaussian/Bernoulli local score traces with a record-start frozen baseline; its practical outcome was negative. CLS e-prop reached 0% unseen sums in the tested G3. A new version must specify a *different estimator* (and causal timing, variance reduction, trace scope, and whether current output may credit an earlier state) and compare against the original rule.

Technical prerequisites: finite-difference/score-identity checks on tiny enumerable stochastic systems; no output leak from held labels; frozen-label and phase-correct feedback controls; measured trace norm, clip fraction, action-tape variance and update direction. A successful technical identity check is **not** behavioral learning evidence.

### H4. Learned relational/compositional state (not a digit algorithm)

Hypothesis: repeated training lets a shared transition develop a latent variable that is **stable under changes irrelevant to the learned relation** while discriminating changes that matter. A possible operational test is to encode arbitrary symbol sequences with role permutations and held combinations, using a task-independent byte transition and no supplied role tags. The same computation must be reused across serial positions; hidden fixed positional lookup and built-in addition syntax are excluded. Relational/compositional invariance is an **outcome to measure**, not a feature to install.

Controls: globally withheld equivalence classes, orientation swaps, non-arithmetic sequence tasks (technical only), randomized label relations, position/length shifts, and an appropriately powerful fixed-kernel comparator. Exact own generation and causal held-gain controls still decide promotion. If architecture cannot even acquire taught mappings, do not diagnose failure as "no abstract schema."

### H5. Autonomous stopping and optional adaptive retrieval (later)

Train a **generic** EOS channel from *observed* EOS during the allowed phase; evaluate it after the model's own emitted prefix. Report (i) any EOS within cap, (ii) EOS at correct length, (iii) payload exactness and (iv) joint exactness. V144's one-world reference already demonstrates that (i) can be 100% while (iv) is 3.33–6.67%. Do **not** promote a new stopping module before this distinction is measured in the native system.

Only **after** there is a working procedure: test a prospective online selector between procedural output and episodic retrieval using exclusively training-history correctness/cost evidence. Compare to the stronger component, fixed mixture and the **actual** CLS confidence-normalized selector, which sometimes harmed performance. Guesses must not become verified memory labels; replay caches must contain previously observed records and be bounded and charged.

## 7. Decision procedure before another "mechanism campaign"

**Step 0 — verify current, nonexecuted boundary.** [V144-R1](../../minifly_v144_dev_20261024/REVISED_DESIGN.md) says the 27-configuration Stage C screen, 4,096-pass continuation, data scaling and holdback fits **have not been run**. It requires a finite dispatcher and qualified checkpoint/resume/admission path. This report does not authorize those runs.

**Step 1 — finish V144's existing capacity diagnostic (first).** Retain its raw FE-PERM byte interface, 40/15 relation-orbit withholding, complete self-generated ASCII payload+own EOS, and three reference modes BP / DFA / feature-frozen at H=128/256/512, finite learning-rate/decay grid, development tuning then untouched holdbacks. Charge all fit/probe costs. This is a **reference ceiling**, not proof that gradients or an external output network may be adopted by native MiniFly.

**Step 2 — interpret the outcome before touching the core.**

| Capacity-screen outcome | Supports | Does NOT establish | Next discriminating test |
|---|---|---|---|
| BP acquires and transfers, DFA/local fails | Temporal/upstream credit is a leading explanation for that reference family | Biological impossibility of local credit | Fixed-features vs trained-features and alternative causal eligibility at matched state size |
| BP and DFA both transfer | Representational and learning capacity exists in a **reference** | Native Full151 can do it | Matched learned address/native-output integration with valid feedback |
| All tested modes acquire taught but do not transfer | Weak induced abstract relation or data/optimization regime in tested grid | Universal nonlearnability of addition or need for hard-coded number sense | Characterize invariance/held gap, then one generic shared-transition/binding intervention |
| All tested modes fail taught acquisition | Input interface, optimization or trial budget may dominate | Proof compositional learning is impossible | Diagnose exposure/loss/own-prefix errors; don't stack advanced memory modules |
| Native learns taught, holds OLD, but not held | Storage/retention without E3 transfer | A rule | Position/role/length generalization under globally excluded relations |
| EOS improves, payload still fails | Termination engineering success | Learned arithmetic | Return to relation learning and own-prefix conditioning |

**Step 3 — in parallel only as a *distinct lineage*, complete the prespecified V6 fan-in preflight when authorized.** Compare its own corrected recurrent route with its R1 parent. Do not combine V6 and V143/V144 denominators or turn engineering diagnostic tapes into independent worlds. If it fails after fixing cue scale, use measurements to choose **one** learning-signal or state intervention.

**Step 4 — only then test H2/H3/H4 singly**, each against a matched null/frozen/shuffled-control with the original E3 rules and full CPU ledger. The first end-to-end held-out gain determines whether a higher-order procedure/memory mixture is justified; avoid combinatorial mixing or relabeling an old failed recipe. Replay and learned arbitration belong *after* an adequate procedure exists.

**Stop/kill rules:** fail closed on source/asset mismatch, held feedback leakage, wrong task/format, frozen control drift, invalid generation prefix, budget overrun or missing histories; mark incomplete evidence incomplete. A result that changes only taught recall, parity, external-length decoding, or KC geometry cannot pass the addition causal gate. Only independent holdback confirmation warrants upgrading a mechanism from candidate to supported.

## 8. Explicit retractions of the two Deep Research outputs

- **"Retrieval practice, spacing, predictive coding, attention or metacognition were missing": false as catalog inventory.** The original 150 list has EDU-01/02/03/17, PERC-03/09, and many parallel entries. Some are interventions rather than internal learning rules.
- **"Implement flashcards/self-quizzes, spaced-example schedulers, gamified intrinsic reward, hints or external target-byte correction": not a mechanism-level solution to the native problem.** Repeatedly teaching seen answers may strengthen lookup; hints or target-derived parity decomposition can change allowed information and invalidate generalization comparisons.
- **"Turn Transformer multi-head attention back on": directly conflicts with current protocol.** The revised human-mechanism discussion must not recommend it.
- **"Add EWC/Fisher penalties, lambda=1000, or extra replay as new mechanisms": unjustified values and duplicate families.** Previous importance/cascade/replay and CLS arms must be consulted before proposing a genuinely different, matched intervention.
- **"The repo is broadly a GRPO error-driven learner": incorrect aggregation.** MiniFly contains distinct Full151 native associations, fast-memory, local feature learning, recurrent/BPTT references and other historical systems. A Genesis-era README does not identify the current experimental kernel.
- **"Mechanism list audited" from folder listings is inadequate.** Claude's original 150-source catalog, the complete V144 all-row crosswalk, exact experimental implementation reports and the V6 original were necessary. This revision provides those identities and capability-level interpretation, rather than pretending to replicate 150 full scientific papers or 254 code variants.
- **Gantt dates and implementation milestones advertised by the second research report were hypothetical.** No branch, PR, source implementation or experiment existed by virtue of the report. This commit is a **documentation correction only**.

## 9. Primary scientific references and interpretation boundary

The original 150 catalog remains the source for the full bibliography and its unverified/contested flags. The following are verified targeted references, not evidence that MiniFly has implemented the brain mechanisms:

- **Procedural -> retrieval transition (human longitudinal fMRI):** Qin et al. (2014), *Nature Neuroscience*, https://doi.org/10.1038/nn.3788. Correlation with developmental strategy change; does not specify an executable learning rule.
- **Hierarchical corticostriatal working-memory gating (computational neuroscience):** Frank & Badre (2012), *Cerebral Cortex*, https://doi.org/10.1093/cercor/bhr114. Computational and companion empirical predictions; not direct proof of MiniFly-like local training.
- **Sequence start/stop signals (animal experiment):** Jin & Costa (2010), *Nature*, https://doi.org/10.1038/nature09263. Mouse nigrostriatal finding; an analogy for learned sequence boundaries, not human-specific evidence or autonomous byte EOS.
- **Structure/content factorization (computational model):** Whittington et al. (2020), *Cell*, https://doi.org/10.1016/j.cell.2020.10.024. Gradient-trained model; the principle can inspire generic architectures without inheriting its optimized machinery.
- **Recurrent online credit (algorithmic model):** Bellec et al. (2020), *Nature Communications*, https://doi.org/10.1038/s41467-020-17236-y. E-prop illustrates a possible local approximation; does not ensure low-variance learning in this substrate.
- **Three-factor synaptic plasticity (review of mostly nonhuman evidence):** Gerstner et al. (2018), *Frontiers in Neural Circuits*, https://doi.org/10.3389/fncir.2018.00053. Evidence for eligibility/third-factor timescales, not exact neural-network gradients.
- **Complementary learning systems (theoretical framework):** McClelland, McNaughton & O'Reilly (1995), *Psychological Review*, https://doi.org/10.1037/0033-295X.102.3.419. Supports a division of fast episode encoding and slow structural learning, not automatic replay-created arithmetic competence.
- **Latent causes / context-sensitive learning (normative model):** Gershman, Blei & Niv (2010), *Psychological Review*, https://doi.org/10.1037/a0017808. A possible state-inference principle, already represented in catalog RL-10.
- **AuGMEnT attention dependence:** Rombouts, Bohte & Roelfsema (2015), *PLoS Computational Biology*, https://doi.org/10.1371/journal.pcbi.1004060. Attention is explicit in the published rule; cannot be ported under a "no attention" claim without defining a truly separate permitted rule.

**Executive recommendation:** The high-value next step is **finish the already approved-in-design but not yet executed V144 capacity-screen tooling and run only under explicit authorization**. In the separate V6 lineage, test the already-specified generic one-hot fan-in repair rather than adding replay. If a sufficiently trained reference learns withheld full addition and the local/native route cannot, prioritize **temporal credit into a reusable gated state**. If even reference learners only memorize, isolate the **learned abstraction/variable-binding** deficit before proposing more synaptic storage mechanisms. The 150 human-learning catalog is broad enough to motivate these designs; its value comes from **falsifiable integration**, not declaring more classroom techniques newly missing.
