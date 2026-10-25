# Human neuronal learning mechanisms: independent audit of Gemini's report

**Date:** 2026-10-25  
**Status:** Reviewed literature crosswalk; documentation only. **Not** a new MiniFly capability, implementation, experiment, or endorsement of a new mechanism.  
**Input:** User-supplied 11-page DOCX, *Neuronal Mechanisms of Human Learning: Cellular Plasticity, Novel Timing Rules, and Empirical Literature* (Gemini). The original binary DOCX is not copied into this repository. Pages below refer to that document, not GitHub pagination.  
**Comparison targets:** Claude's [150-row human-learning catalog](https://github.com/ChangweiZhou/digitalbaby/blob/4c12621d0804d2495c2eacaebf4e17aaeb994771/human_learning_mechanisms_20261023/CATALOG.md) and the [2026-10-24 source-resolved audit](../HUMAN_LEARNING_AUDIT_20261024/REVISED_AUDIT.md). The historical 254 mechanism *implementations/configurations* are not 254 independent biological hypotheses.

## Executive verdict

Gemini provides a **useful cellular-neuroscience supplement**, especially the direct human neocortical STDP and cholinergic-plasticity papers missing from the early Claude CELL-02 description. It is **not exhaustive** and is **not** a validated inventory of mechanisms *implemented* or *absent* in MiniFly.

Several errors materially affect the scientific claims:

1. **BTSP is experimentally demonstrated primarily in rodent CA1.** Rapid encoding by human MTL concept/association neurons is directly observed, but this does *not* establish that human concept cells use plateau-potential BTSP. Do not substitute the phenomenon for its proposed synaptic cause.
2. **AIS activity-dependent relocation has strong culture/animal evidence, not the report's claimed high direct-human plasticity evidence.** Its cited 2025 *Cerebral Cortex* study primarily measured **mouse** hippocampal AIS geometry (3,681 neurons, 12 animals), with a small cross-species anatomical observation—not experimentally induced AIS learning in humans.
3. **Adult human hippocampal neurogenesis needs nuanced evidence.** Spalding et al. (2013) estimated **1.75%/year within the renewing subpopulation**, not the entire hippocampus. The field has contradictory studies; 2026 human single-nucleus multiomics strengthens evidence for neurogenic cell states but does **not** establish their causal contribution to learning or a universal turnover rate.
4. **Human inverse STDP is genuine, localized evidence.** Verhoog et al. (2013) studied resected adult medial temporal cortex. Their negative-interval LTP and positive-interval LTD do **not** show that every human neocortical/hippocampal synapse has an inverse rule. Adult rat synapses in the same paper shared some negative-interval potentiation.
5. **Human cholinergic plasticity evidence is direct but mixed-species methods matter.** The 2016 human/mouse study demonstrated human slice pharmacological modulation; the selective optogenetic cholinergic-fiber tests were performed in transgenic **mice**, not optogenetically modified human subjects/slices.
6. **Sensory response modification (SRM/SRP) is an LTP-*like* network response**, not direct in-vivo proof of a molecular NMDAR/AMPAR synaptic mechanism.
7. **A human presynaptic NMDAR-independent mossy-fiber LTP mechanism was not established by the report's cited review.** Human perforant-path→dentate LTP has been demonstrated and was NMDAR dependent; this is a different pathway. Keep presynaptic mossy-fiber LTP as a nonhuman-supported subtype until a direct human study of that synapse is identified.
8. **Intrinsic excitability in human neurons is not the same as demonstrating learning-induced intrinsic plasticity in humans.** Human channel expression and excitability phenotypes have direct evidence; activity-induced, learning-causal changes are much better established in nonhuman preparations.

The complete 14-row, source-resolved disposition is in [MECHANISM_AUDIT.json](MECHANISM_AUDIT.json). These are **14 distinct claim/variant units**, not an assertion that the input contains exactly 14 unrelated synaptic mechanisms: ADP is part of the inverse-STDP mechanism, and fast association, concept coding and BTSP must be separated by evidence type.

## Evidence convention: do not confuse *species*, *assay* and *learning causality*

- **H-VIVO:** human recordings/behavior in awake people or participants; can directly establish *observed coding or association* but not necessarily an underlying synaptic plasticity rule.
- **H-EXVIVO:** direct pharmacological/electrophysiological manipulation of human tissue; establishes cellular plasticity under the tested surgical-slice conditions, not necessarily a cognitive learning mechanism in intact humans.
- **H-INDIRECT:** human fMRI/EEG, molecular/cell signatures, postmortem labeling or radiocarbon modeling. Valuable, but not direct proof of the proposed synaptic learning mechanism.
- **ANIMAL-CAUSAL:** direct nonhuman experiment; plausible mechanistic transfer is a *hypothesis*, not automatic human validation.
- **MODEL:** theory/computation/analogy; must be labeled separately.

These tags are **not interchangeable with Claude's A–E evidence grades**; the existing grades mix causal human behavior, animal interventions and theoretical plausibility. An entry can carry several evidence tags. “Mechanism is observed” does **not** imply “MiniFly implementation is validated.”

## Mechanism-by-mechanism corrections

| ID | Gemini topic/pages | Corrected empirical interpretation | Source-grounded status | Claude crosswalk |
|---|---|---|---|---|
| G01 | NMDAR-dependent LTP, pp.1,4 | Direct adult human cortical/dentate slice potentiation; circuit-specific, not the only form | H-EXVIVO | CELL-01 |
| G02 | NMDAR-dependent LTD, pp.1,5 | Bidirectional synaptic modification observed in tested human cortex; simple frequency threshold is an approximation | H-EXVIVO | CELL-01 |
| G03 | Presynaptic non-NMDAR LTP, pp.1,5 | Animal mossy-fiber mechanism; direct human mossy-fiber demonstration **not verified** from the cited reference | ANIMAL-CAUSAL; human claim OPEN | CELL-01 subtype |
| G04 | Stimulus-specific response modification, pp.1–2,5 | Human evoked-response potentiation consistent with LTP-like change, not single-synapse causation | H-INDIRECT | CELL-01 / PERC |
| G05 | Broad inverse human STDP, pp.2,6 | Genuine human ex-vivo timing rule in sampled medial temporal cortex; not universal to humans | H-EXVIVO | CELL-02 (source upgrade) |
| G06 | Afterdepolarization / L-type VGCC mechanism, pp.2,6 | Causal channel manipulation in human slices; mechanistic **subtype** of G05, not independent whole-learning architecture | H-EXVIVO | CELL-02 / CELL-10 |
| G07 | Cholinergic gating of STDP, pp.2,6–7 | Laminar ACh modulation in human surgical slices; light-activated endogenous ACh method chiefly mouse | H-EXVIVO + ANIMAL-CAUSAL | CELL-02 / MOD-01 |
| G08 | Multimodal MTL concept cells, pp.3,7 | Invariant/sparse human single-neuron *coding*, not by itself a synaptic plasticity rule or one-neuron-one-concept exclusivity | H-VIVO | CONC / SYS |
| G09 | Rapid conceptual association, pp.3,7 | Human neurons change tuning after learning picture associations; the biochemical update rule is unidentified | H-VIVO | SYS-08 / CONC |
| G10 | Behavioral-timescale synaptic plasticity, pp.3,7 | Seconds-scale plateau-gated rule established causally in mice; its use by human concept cells remains unproven | ANIMAL-CAUSAL | CELL-08 |
| G11 | Adult hippocampal neurogenesis, pp.3,7–9 | Radiocarbon and modern cellular evidence support ongoing human neurogenic signatures; size/rate/function remain debated | H-INDIRECT + ANIMAL-CAUSAL | CELL-19 / SYS-03 |
| G12 | Intrinsic excitability/channelome, pp.3–4,8 | Human static cellular specializations observed; causal *learning-induced* intrinsic modification mostly nonhuman | H-EXVIVO (phenotype), ANIMAL-CAUSAL (plasticity) | CELL-05 |
| G13 | AIS remodeling, pp.4,8 | Learning-related activity-dependent AIS shifts shown in animal/culture work; cited 2025 study primarily mouse anatomy | ANIMAL-CAUSAL; human functional claim OPEN | CELL-05-related; catalog gap in specificity |
| G14 | Metaplasticity/threshold adaptation, pp.2,9 | Sliding plasticity thresholds supported especially by animal/in-vitro work; human behavioral conditioning not direct proof of BCM | ANIMAL-CAUSAL / H-INDIRECT | CELL-03 |

## Source audit: representative primary research

1. **Human NMDAR STDP / LTP / LTD:** Verhoog et al. (2013), *J Neurosci*, [DOI 10.1523/JNEUROSCI.3158-13.2013](https://doi.org/10.1523/JNEUROSCI.3158-13.2013). Human neocortical slices, adult surgical subjects; timing window and pharmacological tests. Also human dentate/perforant-path LTP: [Synaptic Plasticity in the Human Dentate Gyrus](https://pmc.ncbi.nlm.nih.gov/articles/PMC6772802/).
2. **Human sensory population plasticity:** Teyler et al., [Long-term potentiation of human visual evoked responses](https://pmc.ncbi.nlm.nih.gov/articles/PMC1226326/). In-vivo evoked response, not isolated cellular LTP.
3. **Laminar ACh and synaptic plasticity:** [Layer-specific cholinergic control of human and mouse cortical synaptic plasticity](https://doi.org/10.1038/ncomms12826) (2016); compare human bath/local ACh manipulations with **mouse** Chat-Cre/Ai32 optogenetics.
4. **Human rapid association / conceptual coding:** Ison et al. (2015), [*Neuron* DOI 10.1016/j.neuron.2015.06.016](https://doi.org/10.1016/j.neuron.2015.06.016). Additional human evidence for multi-element episodic conjunctions: [2023 study](https://pmc.ncbi.nlm.nih.gov/articles/PMC10663153/).
5. **BTSP causality (rodent):** Bittner et al. (2017), [*Science* DOI 10.1126/science.aan3846](https://doi.org/10.1126/science.aan3846); [2026 review](https://www.nature.com/articles/s41593-026-02214-2). Does not demonstrate the same synaptic rule in human concept cells.
6. **Human neurogenesis:** Spalding et al. (2013), [*Cell* DOI 10.1016/j.cell.2013.05.002](https://doi.org/10.1016/j.cell.2013.05.002); [2018 conflicting studies review](https://pmc.ncbi.nlm.nih.gov/articles/PMC6035081/); Disouky et al. (2026), [*Nature* DOI 10.1038/s41586-026-10169-4](https://doi.org/10.1038/s41586-026-10169-4) (multiomic identification; cognitive causal contribution unresolved).
7. **AIS evidence:** Grubb & Burrone (2010), [*Nature* DOI 10.1038/nature09160](https://doi.org/10.1038/nature09160), dissociated hippocampal cultures; Stevens et al. (2025), [*Cerebral Cortex* DOI 10.1093/cercor/bhaf297](https://doi.org/10.1093/cercor/bhaf297), predominantly mouse hippocampus (small human anatomy comparison).
8. **Human cellular physiology context:** van Loo et al. (2025), [*J Neurophysiol* DOI 10.1152/jn.00120.2025](https://doi.org/10.1152/jn.00120.2025), review; not a direct demonstration that behavioral learning changes HCN/K/Na channels in adult human cortex.
9. **Human concept-cell specificity:** Quian Quiroga (2012), [*Nat Rev Neurosci* DOI 10.1038/nrn3251](https://doi.org/10.1038/nrn3251), review of human intracranial single-unit results.
10. **Nonhuman presynaptic LTP caveat:** the report's InTechOpen review is not direct human presynaptic mossy-fiber physiology; do not substitute human perforant-path/dentate NMDAR LTP for the claimed pathway.

### What the Gemini bibliography does not establish

- Wikipedia, ResearchGate figures, educational summaries, magazine reports and general reviews do not independently verify the detailed **species-specific causal experiment** claimed by a row. Prioritize primary methods/results.
- The 2025 AIS paper's two-of-three human anatomy observation is **not** activity-dependent human AIS relocation.
- Concept-cell response invariance and rapid association do **not** measure BTSP plateau potentials.
- The 2013 radiocarbon **1.75%** is a fitted rate within a renewing fraction; it must not be presented as direct real-time counting or a universal adult hippocampal rate.

## Relation to Claude's 150-row catalog

Do **not** create 14 supposed novel mechanisms: most map to CELL-01/02/03/05/08/19, MOD-01, SYS-03/08 or CONC. The most valuable **source updates** to the existing catalog are:

- **CELL-02:** add Verhoog 2013 as **direct human ex-vivo** STDP evidence; its present entry focuses primarily on human PAS and makes the literature look weaker than it is. Keep the caveat that the window is regional/protocol-dependent.
- **CELL-01:** anchor human LTP/LTD to real resected-slice studies; separate classical dentate NMDAR-dependent potentiation from the unverified “human presynaptic mossy-fiber” subtype.
- **MOD-01/CELL-02:** link the 2016 human/mouse nicotinic modulation study; do not label transgenic mouse methods human evidence.
- **CELL-08/SYS-08:** explicitly separate rodent plateau BTSP from human single-neuron rapid association.
- **CELL-19:** append 2026 *Nature* human multiomics, **without removing the contested flag** or claiming established causal memory contribution.
- **CELL-05:** distinguish intrinsic conductance/excitability homeostasis from AIS geometry; keep human learning-dependent AIS relocation unverified.

Important **omissions from Gemini**, already better covered by Claude: synaptic tagging and capture (CELL-07); eligibility-trace/three-factor rules (CELL-09); dendritic nonlinear computations (CELL-10); inhibitory plasticity (CELL-12); heterosynaptic competition (CELL-14); protein synthesis-dependent consolidation (CELL-15); glial modulation (CELL-18); and complementary learning systems, replay, retrieval-mediated integration and systems-level schema learning (SYS). Gemini's “exhaustive” language must not be preserved as a scientific verdict.

## Implications for MiniFly (no automatic mechanism promotion)

| Biological finding | Generic possible computational abstraction | Existing research connection | Priority / reason |
|---|---|---|---|
| Human STDP and VGCC-dependent timing | Signed, time-asymmetric eligibility kernel with local pre/post activity and **legally available** modulator | Already adjacent to three-factor / temporal plasticity attempts | **Conditional:** investigate only if measured delay-credit deficit justifies it. Never globally impose one human slice STDP curve on all Full151 synapses. |
| Layer-specific cholinergic plasticity | Cell/compartment-specific plasticity gain or gate | Native feedback and gating, many historical variants | **Conditional:** signed causal credit and native acquisition must first be demonstrated; ACh evidence does not specify output-aware softmax learning. |
| Concept/episode cells | Sparse invariance with separate role/episode binding | V143 learned representations, SYS-08, historical binding | **Mechanistic target, not recipe:** distinguish loss of identity from task-relevant invariance. |
| BTSP seconds-window | Plateau-like event × delayed eligibility trace | CELL-08, historical e-prop / eligibility | **Research hypothesis only:** rodent causal, no direct human BTSP validation. Account for additional temporal comparisons/updates. |
| Metaplasticity / intrinsic homeostasis | Local thresholds, activity-dependent learning-rate control | Homeostasis and protection already in MiniFly | **Existing-family refinement:** check whether gain changes improve relevant knowledge under fixed budgets. |
| Adult neurogenesis / AIS relocation | Structural allocation, replacement, excitability regulation | EXPAND / sparse capacity / allocation | **Low immediate priority:** biological connection to generalization is indirect, and capacity increase has not solved the project's transfer bottleneck. |

Suggested generic formulation for a future experiment, **not an implemented synaptic rule**:

\[
e_{ij}(t)=\int_{-\infty}^t k_{ij}(t-u)\,x_i(u)\,y_j(t)\,du,\qquad
\Delta w_{ij}(t)=\eta\,g_j(t)\,e_{ij}(t).
\]

A time window \(k_{ij}\) and modulation \(g_j\) can be inspired by STDP/BTSP/cholinergic gating, but **cannot be selected merely from qualitative analogy**. Its information provenance, measurable loss/benefit, target relevance, storage and CPU cost, and controls must be declared. In particular, using an externally supplied correct-answer equivalence class when no such feedback exists would be leakage.

**Current experimental implication:** Gemini's report gives **no demonstrated new MiniFly solution** to V148's fixed-C16 output distribution limitation, the failed full-byte arithmetic acquisition, or generalizable persistent knowledge. Human biological evidence is a mechanism prior, not proof of a computational capability.

## Review limitations and provenance

- Source-level targeted check of the original document's claims and selected primary studies; **not** a systematic full-text review of all 28 Gemini references or all 150 Claude entries.
- No claim that evidence tier is a probability of mechanism success. Surgical samples and regional circuit context limit generalization.
- The uploaded DOCX, primary studies, and historical GitHub artifacts remain distinct evidence sources; the audit does not convert source hypotheses into facts or modify original reports.
- No MiniFly experiment, scientific training, native-core modification, hidden-holdout opening, or promotion decision resulted from this literature review.
