# After V67: what the existing fly data justify testing next

13 September 2026. Review and bounded reanalysis of existing data. **No V68 learning experiment, new parameter fit or model promotion has been performed.**

**I recommend making the next round test whether the current memory pathway has useful writes that can be selectively preserved, before inventing another protection rule.** The central comparison should distinguish (1) available coding and write space, (2) the information required to choose an appropriate write, and (3) physiological plausibility. These are currently entangled. A parallel, narrowly bounded physiology identification check is justified before treating the almost absent α-feedback channel as a biological result.

The review used the original FlyWire graph and frozen interface, BANC and hemibrain input extractions separately, the existing V67 cue bank and outcomes, and the reconciled Huang spike-measurement ledger and original parameter files. It also rechecked primary papers on compartment-specific learning and APL inhibition. The scripts and small tables accompany this report. Historical releases remain unchanged.

## 1. What the history says—and what it does not

| Work | Retained result | Boundary for the next design |
|---|---|---|
| V50–V54, earlier ELM/byte-task line | A selective teaching fixture can support persistence; tested scalar context gains did not solve the real teaching-selection problem | Different task and code line. Its `FUNCTIONAL_CORE_ESTABLISHED` label is not evidence that MiniFly recovered a fly core |
| V55 anatomical audit and follow-up | Output-site organization and type-level feedback routing are real structural constraints; small KC samples estimate an address table well | Predicting anatomy does not demonstrate memory capacity or identify a teacher's activity |
| Initial MiniFly and early learning iterations | Executable sensorimotor reduction; reproducible effective learning; fixed mutable state; anatomy-constrained input/output/teaching | Fixed-weight sensorimotor success is separate from lifelong learning. BANC did not establish spatial replication |
| Iterations 4–6 | Multiple timescales explain some persistence; stronger slow allocation creates physiological and revision tradeoffs | A longer lifetime or more slow writing is not an established repair |
| Iterations 7–8 | The tested LH comparator failed; a current-sensory-state engineering reader substantially improved access to memory | Observer improvement does not create storage capacity or validate a biological decision circuit |
| V64–V65 / iterations 9–10 | Loading causes losses in the full parent; real uniform and stratified KC deletion adds a separate penalty | Compression has been tested. Full-parent interference cannot all be blamed on compression |
| V66 | Direction and local expression of actual writes explain signed interference; matched consequence changes causally reverse its effect | A retrospective victim-memory footprint is not an available local teaching signal |
| V67 | γ history supports acquisition as well as collateral writing; the tested local sign gate fails; magnitude relates to disturbance size | Neither complete γ-history removal nor this gate qualifies for repair. Other local computations are not ruled out |

Sources: [V54 summary](../../V54_Mac_M3/v54_results/SUMMARY.md), [anatomical audit](../../fly_teaching_audit_20260913/REPORT.md), [project history](../PROJECT_STATE.md), [V65](../iteration10/REPORT.md), [V66](../iteration11/REPORT.md), [V67](../iteration12/REPORT.md).

One earlier negative is especially relevant: **V48 already tested a global APL-like controller using BANC KC↔APL count profiles and failed selective interference, retention and topology-specificity gates.** Another gain sweep of that abstraction would repeat old work. Actual APL has spatially local activity and inhibition, so a future local APL study would require a different spatially justified model and suitable controls, not a new name for global normalization. [V48 result](../../V48_Mac_M3/v48_results/SUMMARY.md), [Amin et al., eLife 2020](https://elifesciences.org/articles/56954).

## 2. The nominal KC count overstates the size of the tested memory pathway

The direct FlyWire audit distinguishes a retained cell population from the cells and channels that can actually write the selected memory:

| Quantity | Existing full reference |
|---|---:|
| Retained KCs | 5,177 |
| Annotated DANs / MBONs in the graph | 326 / 97 |
| Biological DANs / MBONs used by the effective core | 4 / 6 |
| Effective teaching / output channels | 4 / 4 |
| KCs with both α3 teaching and α3 readout access | **1,226** |
| Those α3-supported KCs active at least once in V67 | 1,179 |
| Mean active KCs per V67 cue | 258.99 |
| Mean active KCs with α3 teaching/readout access | **53.68**, range 9–118 |
| Mean active KCs supporting neither selected write pathway | **140.11**, about 54.1% of activity |

The two modeled output types are MBON11 and MBON14; teaching types are PPL101 and PPL106, split by side. α-fast and α-slow share one coordinate per KC, whose teaching sums the selected α channels. Other cell types may be useful to the actual fly, but their memory pathways are not implemented here. Their activity is not biologically “wasted”; it is outside this selected model's write/readout scope.

The source graph has 25 DAN types directly reaching KCs and 35 MBON types directly receiving KC input. This establishes omitted diversity, not a mandate to activate every channel. Adding channels could improve performance simply by adding capacity or changing the task interface.

The current encoder normalizes PN input per KC and selects about 5% of KCs per hemisphere by ranked input. It does not model the measured local APL circuit. KC subtypes are used unequally: some α′β′ groups respond to roughly 7.5–10.1% of the existing cues, versus about 4.1–5.0% for several αβ groups. Most of the highly exposed α′β′ population does not participate in the chosen α3 learning path. Top-quintile α3-supported KCs account for 44.3% of the exposure-weighted potential α3 output mass. These are static geometry measures, not recorded learning importance or a capacity law.

**This is not evidence of code collapse.** All 1,536 existing A/B pairs remain distinct on α3-supported KCs, with 29–186 exclusive supported coordinates per pair. The 323 never-active KCs are unreachable from the selected ALPN input, rather than cells that permanently lose top-k competition. Thus “many retained neurons, fewer usable α3 addresses” is a relevant model limitation, but does not by itself explain the observed failures.

I joined these unadapted, pre-training geometry measures to the already observed V67 snapshot outcomes. At load 96 plus a day, mean within-stream Spearman correlation with actual memory margin is:

| Initial pair feature | Mean correlation | 95% whole-stream bootstrap interval |
|---|---:|---:|
| Minimum number of α3-supported active KCs across the pair | 0.134 | 0.081 to 0.183 |
| Exclusive α3-supported coordinates | 0.165 | 0.117 to 0.214 |
| α3-output-weighted code distance | 0.178 | 0.118 to 0.245 |
| α3-output-weighted within-pair cosine | 0.008 | −0.034 to 0.053 |

Support and separation have modest associations with performance; within-pair cosine alone does not explain the delayed margin. This retrospective join is exploratory, involves known outcomes and does not isolate causal contributions. It measures initial A/B geometry, not overlap with every intervening memory. It gives a reason to test the use of available write space, not to claim that changing sparsity will repair retention.

Sources: [anatomy note](anatomy_audit/NOTE.md), [support inventory](anatomy_audit/cue_active_support.csv), [KC types](anatomy_audit/KC_type_exposure_and_support.csv), [outcome associations](anatomy_audit/existing_outcome_correlation_summary.csv).

## 3. The real teacher receives substantial input that the core does not identify

I queried PPL101 and PPL106 specifically in the previous BANC and hemibrain extractions. For exact exported dendritic territories:

| Type and side | Hemibrain OTHER input fraction | BANC OTHER input fraction |
|---|---:|---:|
| PPL101 right | 95.13% | 86.63% |
| PPL101 left | 89.00% | 85.14% |
| PPL106 left | 81.57% | 94.70% |
| PPL106 right | 53.80% | Not present in selected BANC annotation |

“OTHER” is the original audit's outside-core category, not a measured novelty or context signal. Exact dendrite totals include all recorded source classes; primary-dendrite and other territories are reported separately. Cropping, annotation differences and export coverage limit comparisons. Missing right PPL106 in this selected BANC annotation is not zero biological input.

For a concrete example, hemibrain SMP053 neuron 5813032847 supplies 306 counts to the selected right PPL101 dendrite. Only 8 of its own 2,965 recorded inputs are from MBONs. It cannot honestly be modeled as only a passive MBON feedback relay. But its additional inputs cannot be called an event-relevance label without recordings or perturbation evidence.

The adult MB connectome describes diverse external and internal-state routes to DANs and explicitly leaves their functional information content unresolved. This supports testing additional *measured* input information; it does not license inventing a novelty gate and attributing it to anatomy. [Li et al., eLife](https://elifesciences.org/articles/62576). Local evidence: [source fractions](external_input_audit/selected_DAN_source_fractions.csv), [top inputs](external_input_audit/top_outside_inputs.csv), [provenance](external_input_audit/provenance.json).

## 4. The physiology changes which repairs would be credible

The two-module α3→DAN-α3 fitted gain is 2.0743×10⁻⁹, whose log magnitude is −19.9936. The released fitting script initializes that parameter at −20. The saved files do not prove the exact historical run or an optimization bug. However, this proximity makes parameter identification worth checking before interpreting V67 A0 as biological evidence. The stored 10,000 uncertainty draws are local inverse-Hessian samples near that solution, not a search over other plausible solutions. Anatomical α3 feedback contacts exist; counts do not determine their effective gain.

The reanalysis of 840 previously reconciled spike-measurement records also reinforces two constraints:

- The unpunished repulsive CS− changes by **−23.88 ± 3.33 Hz** at five minutes, while attractive CS− changes **+0.19 ± 3.87 Hz** (mean ± SEM; twelve flies per cohort). These are output changes, not observed synaptic writes. A universal “unpunished means protect from change” phenotype would be wrong; an unpunished member of a conditioning pair also differs from an arbitrary unrelated novel cue.
- In the attractive cohort, three-hour paired biases are **−16.28 Hz** for control, **−16.29 Hz** after early re-exposure and **−1.45 Hz** after late re-exposure. This supports time-dependent susceptibility more directly than a universal demand for strong early strengthening. It does not establish early/control equivalence or refute a transient effect at another time.

These numbers use the earlier audit's inferred three-second baseline, pair identities only within a cohort and come from the same study. The twelve repulsive observations that differ between the old fitting workbook and final tables remain important. The separate Fig. 5k ratio mismatch remains unresolved. [Physiology note and exact tables](physiology_audit/NOTE.md), [previous source audit](../physiology_source_audit_20260913/REPORT.md), [Huang et al., Nature](https://www.nature.com/articles/s41586-024-07819-w).

## 5. My recommended V68: useful write space before a new local rule

**Question:** On the existing legal memory coordinates, is there a way to reduce harmful later writing while retaining useful acquisition and revision? If so, what information would be required to choose it?

V67 classified a whole episode as reversal or unrelated at each sampled active coordinate. A useful controller instead needs to decide whether changing *this coordinate now* helps the new memory and harms other memories. Many coordinates in an unrelated episode are harmless to a particular old memory. The trace-gate diagnostic also counted raw conflicting mass, rather than its contribution to output decisions. Neither flaw invalidates the predeclared V67 rejection; they explain why repeating the same classifier with more features is not the best next step. [Detailed critique](design_critique/NOTE.md).

I would use a bounded two-stage design:

1. **Test availability of a useful selective write.** Keep the full parent and the same input representation. At matched low- and higher-load checkpoints, allow an evaluator to attenuate only existing slow increments, with no added directions, amplification or memory arrays. Compare selective attenuation with uniform attenuation at the same declared slow-write budget. Preserve new-association gain and evaluate damage over **all** previously learned associations, not just four favorable focals. This evaluator may use old/new footprints; it is an explicitly nondeployable diagnostic.
2. **Only if such a write exists, test whether a causal rule can select it.** The rule gets only justified local/current signals and must operate during learning. It receives no old-cue catalog, future outcomes or evaluator mask. Test fresh entire streams and durable reversal, including the identical-first-bout anomaly control. Actual acquisition–damage utility is primary; an episode-classification AUROC is secondary.

A tractable initial diagnostic uses an endpoint of the form `s_no_write + m * delta_slow`, with `0 <= m <= 1`, a single preregistered retained L1 budget and fixed nonslow increments. Within a specified clipping region, old/new margins are affine in m, allowing a linear feasibility/optimization problem. Exact endpoint replay must verify solver results. A successful witness proves that a useful action exists in that class. A certified negative applies only to the declared constraints and clipping region. A failed heuristic search is not proof that every selective rule is impossible. Endpoint interventions are not upper bounds on all dynamic learning policies.

For any causal candidate, preserve V67's scale of practical acquisition tolerances—no more than 0.25 Hz mean new-gain loss or five percentage points of choice loss—as a proposed starting point to freeze. Require a positive old-memory benefit over both parent and matched attenuation, report worst losses across old memories, and retain delayed-revision and eligibility constraints. L1 matching is different from V67's L2 endpoint sham; do not mix them. The exact budget, checkpoint count and thresholds belong in the next frozen protocol, not in a post-outcome choice.

This experiment can change the direction of work decisively: useful controllable writes with no recognizable local selection point toward missing observable information; no benefit in the declared controllable class deprioritizes that suppression approach; a causal benefit justifies one full load/revision trial. It neither requires a new connectome nor promotes an oracle as a learner.

## 6. Other ideas, ranked and conditional

| Priority | Idea | Why now | Stop condition |
|---|---|---|---|
| Parallel, bounded | **Physiology-only identification of α feedback** | Tiny gain lies near the released initialization; final conditioning targets are now trace-reconciled | If different gains fit equally, report unidentified. If meaningful gains degrade clean intervention/valence phenotypes, reject restoration in this model class. Never select the gain on synthetic retention |
| Conditional engineering candidate | **Redistribute sparse coding within anatomical support groups** | Only ~54 active coordinates serve α3, with heterogeneous subtype exposure and modest outcome associations | Reject if benefit disappears after matching active writable counts, output gain, raw-write budget and acquisition. Include matched shuffled-group controls; distinguish observer baseline shift from stored-memory improvement |
| Conditional biological extension | **Restore one independently supported compartment or local output address** | Real MB teaching/output diversity is greatly reduced in the core; published compartment-specific learning rules are heterogeneous | Require physiology/type mapping and equal-state-budget shuffled or generic expansion controls. Equal gains from random expansion support added capacity, not a recovered fly design |
| Requires targeted help/data | **Add an independently measured external DAN input** | Outside-core inputs dominate many relevant dendrites | Do not implement a relevance/novelty channel if its activity mapping is guessed. Request same-type activity/perturbation under known events instead |

Published experiments show different DAN types can write and update memories with different timing and retention properties. This is a reason to test a specifically supported additional compartment, not to attach arbitrary timescales to more channels. [Aso and Rubin, eLife 2016](https://elifesciences.org/articles/16135).

For an encoder intervention, I would first use the current 88 input channels and real PN→KC weights, not add hand-designed context labels. Changing the encoding changes the observer's input distribution; the fixed reader and actual-memory margins must both be reported, with naive baseline checks. A new reader fit cannot be allowed to hide an unchanged storage problem. A count-balanced or locally inhibited encoder is a hypothesis to test, not a consequence proven by the static audit or by APL anatomy.

**Defer:** another 647-KC compression claim, a generic slow-lifetime/allocation sweep, complete γ deletion, α ablation at the same near-zero gain, threshold tuning of the rejected sign gate, another global APL count controller, or a larger neural classifier whose only success is predicting the evaluator's label. None addresses a newly established causal lever.

## 7. What help would materially change the plan

The existing files are sufficient for the recommended bounded diagnostic. More bulk connectome data are not the current bottleneck. Useful additions would be measured PN/KC activity for the conditioning odors and concentration/history conditions; joint or perturbation-linked MBON/DAN responses with verified type/animal identities; longer multi-association and reversal physiology; and clarification of the remaining source-table definitions. We do not need an answer or another download to prepare the recommended experiment.

This review adds static/reanalysis evidence and proposed decisions only. [MANIFEST.json](MANIFEST.json) identifies compact deliverables and references unchanged data plus excluded per-cue/per-fly derived ledgers. The prior V67 archive and preparation remain intact. No inference here establishes a persistent fly learning core; the purpose is to make the next experiment discriminate between concrete explanations instead of adding another ungrounded mechanism.
