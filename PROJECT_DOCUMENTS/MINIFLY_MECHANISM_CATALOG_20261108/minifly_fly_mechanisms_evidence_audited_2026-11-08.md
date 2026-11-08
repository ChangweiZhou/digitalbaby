# Fly learning and circuit computation: a source-audited operational catalogue for MiniFly

**Date:** 8 November 2026. **Provenance:** new research and a source audit, not recovery or reconstruction of the missing original file.

## What this delivers—and does not claim

This document contains **20 operational records**, a **70-row disposition ledger for the actually available reconstructed supplement**, explicit local algorithm candidates, scaling/accounting notes, source links, and an evidence/local-sufficiency comparison. Twenty is not a claim about the total number of fly learning mechanisms, the number of original ranked entries, or the number of new ideas for MiniFly. No entry quota was used to inflate the catalogue. Several records are related operations within the same biological family; those relations are stated rather than hidden behind a “zero duplicates” claim.

The original `fly_brain_learning_mechanisms_deep_research.md` was **not recovered**. The exact original 20-entry ranking and 10-family summary were not recovered together with the complete original catalogue, so a complete original-to-new crosswalk cannot be claimed. The 70-row audit below refers specifically to the later **Executive Summary(2)** reconstructed report found in the Library, not the missing original. No number of distinct unseen or untested ideas can be inferred from this review.

### Project sources actually used

- The user’s current critique, including the **reported E16 format-change forgetting and ongoing replay controls**. Those logs were not independently inspected in this review.
- `Executive Summary(2)`: the reconstructed 70-entry report, including its explicit recovered-zero statement, its “duplicates: 0” assertion and all 70 numbered entries.
- `MiniFly_50plus_Mechanisms_Pruned_Codex_Summary_2026-09-26.md`: historical overlap/implementation context, not a current repository audit. It documents 31+25 overlapping labels, existing local writing, sparse expansion, eligibility, temporal-contrast variants and ITDP.
- Earlier `Fly Brain Learning Mechanisms (1).docx` / `Fly Brain Learning Mechanisms.docx` and the separate 11-circuit survey identify historical scope and claims. Their bibliographies are not treated as primary evidence merely because they appear in a report.

### Scope and evidence discipline

The main evidence is **Drosophila**, with the larval source explicitly marked. Mammalian hippocampus, neocortex, basal ganglia, cerebellum and amygdala are not counted as fly evidence. Molecules, receptors and gene names are not separately counted as algorithms. Courtship/sex-specific mechanisms are outside this project screen. Adult brain learning mechanisms and fixed supporting computations are placed in separate categories; an innate computation is not passed off as a learning rule.

“Shown” refers to the reported animal experiment; “inferred” to the interpretation of recordings/manipulations; “proposed” to a model or engineering abstraction. A causal paper is not automatically an independent replication. The verification field states whether selected follow-up evidence is convergent, same-lineage, computational, preliminary or absent. This is a targeted, source-audited review, **not an exhaustive systematic-review census** of all publications or replications.

Every algorithm below is an **explicit proposed abstraction unless identified as a published model**. It has not been run in this review. The biological studies do not establish that it is sufficient for raw-byte systematic generalization. No candidate is assigned that status.

## Findings that change the old evidence map

The old 70-entry supplement contains a literal duplication (21 says to see 19), other overlapping labels (5/25), and family-level overlap (20/47/66). Those are not all the same kind of duplication; no exact unique-mechanism count is asserted. The 70-row ledger preserves every old number so nothing disappears silently.

The label **“BCB/Strong muscle learning”** could not be tied to an identifiable primary source and is excluded. It is not repaired by guessing that some unrelated real mechanism was intended.

**Retraction check:** Okray et al.’s 2023 *Multisensory learning binds neurons into a cross-modal memory engram* was retracted on 25 March 2026. The notice reports nonreproducible voltage-imaging findings and analysis-pipeline concerns. The authors say behavioral and connectomic observations were replicated; this does not rehabilitate the withdrawn physiological mechanism. That paper is not positive support for a cross-modal binding algorithm here. [Retraction notice](https://doi.org/10.1038/s41586-026-10355-4).

**More informative positive evidence:** second-order conditioning has direct circuit work in Yamada et al. (2023) and Rachad et al. (2025), while flexible visual-to-heading anchoring has two convergent 2019 studies. These specify teachable operations, not simply the presence of recurrence. Neither establishes arithmetic-rule discovery. The later octopaminergic anchoring mechanism is labeled as a preprint, and the 2026 vector-inversion article is folded into the vector-operation family rather than counted again as a new general learning system.

## Common implementation and accounting contract

A model may use arbitrary optimization **during development**. During life, a synapse may use its own activity history, its receiving neuron’s state, physically delivered neighboring signals and the allowed broadcast modulator. The existence of an equation containing an error term does not establish that the circuit can locally compute or receive that error.

**Do not hide supervision.** A scalar reward, C compartment modulators and a width-dimensional target vector are different information budgets. Compartment-specific signals may be generated by local circuitry, but their generators, routes and bandwidth must be counted. No model may receive held-out labels, future state, a privileged parser, or an externally supplied correct internal coordinate while being described as learning those quantities.

**Separate four budgets.** Let H be the number of shared developmental coefficients; S be independently writable synaptic magnitudes; A be dynamic activity/register state; and E be eligibility or other persistent learning state. A rate-model estimate is

`stored numeric state ≈ H + S + A + E`,

plus any explicitly stored fixed weights, indices, masks and adjacency. With float32 numeric state this is roughly `4*(H+S+A+E)` bytes before topology and implementation overhead. An illustrative `H=20, S=100000, E=100000, A=10000` uses about **840080 bytes (0.80 MiB)** just for those numbers—not “20 parameters.” This arithmetic is an engineering estimate, not biological data.

If all synapses of a type remain tied during life, only shared aggregate learning is possible; arbitrary synapse-specific associations cannot be retained for free. If they start tied and diverge locally, their independent values count toward S. Sign-preserving implementation can use `w=s*a`, with fixed sign `s` and `a>=0`, updating and clipping a locally. Mixed effective signs need separate pathways where necessary.

For N units and bounded degree k, sparse dynamics can cost O(kN) per internal update; unconstrained dense recurrence costs O(N²). K internal rehearsal/settling steps multiply work by K. The **60,000-parameter system is only a baseline**, not an upper bound. H can remain small during scaling while S, E, topology storage and compute grow substantially.

The one comparison in each record is an identification contrast, **not a request to launch a 20-arm sweep**. Match total independent weights, persistent state, development data/optimization, lifetime observations, reinforcement and update/compute budget as far as the manipulated mechanism permits. Specify unmatched resources explicitly. Old-format retention, trained-pair recall and never-reinforced rule generalization are separate measurements.

## Operational records

### F01. Population normalization with spatially local inhibition

**Category:** I. Input conditioning. **Species/stage:** Adult Drosophila melanogaster

**What it computes and what is evidenced.** Shown: antennal-lobe inhibition produces divisive normalization; APL inhibition within the mushroom body can be spatially localized. These are convergent examples of normalization, not two independent general learning algorithms. The localized APL result specifically argues against equating the entire biological circuit with one global scalar. [Olsen, Bhandawat & Wilson (2010)](https://doi.org/10.1016/j.neuron.2010.04.009); [Amin et al. (2020)](https://doi.org/10.7554/eLife.56954).

**Verification / follow-up status.** Convergent computational role across two circuits; no claim of independent replication of the complete proposed MiniFly motif.

**Wiring and signs.** Adult fly: sensory afferents excite projection neurons and recruit inhibitory local neurons; in the mushroom body, KCs recruit APL and receive inhibition. Use positive drive and negative inhibitory feedback. The exact pooling neighborhood is architecture, not a universal biological convergence ratio. [Olsen, Bhandawat & Wilson (2010)](https://doi.org/10.1016/j.neuron.2010.04.009); [Amin et al. (2020)](https://doi.org/10.7554/eLife.56954).

**Development versus lifetime; local algorithm.** Development fixes pooling neighborhoods and gains; lifetime activity varies without requiring weight learning. Proposed rate approximation: pool_p = sum of locally connected activities; output_i = drive_i / (epsilon + gain * pool_p). A neuronal pooling unit supplies the denominator, rather than a synapse reading the whole network.

**Shared parameters, independent state and scaling.** H = O(1–10) gains per pooling class; activity state O(N+P), edges O(S_pool), work O(S_pool) per update. Explicit pooling adjacency still occupies storage. Learned normalization weights would add S_pool independent parameters; they are not included for free.

**Models and what they establish.** Olsen et al. (2010) fitted a normalization computation to odor-response measurements. Amin et al. (2020) constrained the spatial implementation. Neither establishes locally learned semantic routing; no separately verified implementation package is claimed here.

**Possible benefit and boundary.** Gain robustness and controlled competition, not discovery of an arithmetic rule. Local rather than global pooling could preserve several concurrent representations.

**Overlap / current project status.** Already-covered inhibition/sparsity family. The September ledger records scalar APL work; local neighborhoods are a variant, not a newly discovered family.

**One matched comparison and one measurement.** Local versus global pooling, with equal edge/state budget, training and updates. Measurement: held-out accuracy under input gain variation.

**Local-rule sufficiency.** Fixed computation; learning sufficiency not applicable.

**Confidence.** High for normalization; medium for transferring the spatially local variant to MiniFly.

### F02. Adaptive temporal filtering

**Category:** I. Input conditioning. **Species/stage:** Adult Drosophila melanogaster

**What it computes and what is evidenced.** Shown: fly olfactory receptor responses have adaptive temporal dynamics, and spike generation adds temporal filtering. This supports a computation over recent history; it does not show durable rule learning. [Nagel & Wilson (2011)](https://doi.org/10.1038/nn.2725).

**Verification / follow-up status.** One primary mechanistic study selected; independent replication not established by this review.

**Wiring and signs.** Adult fly olfactory receptor neurons transform sensory drive before projection-neuron and KC processing. A compact abstraction uses a local slow state that subtracts from current excitation. [Nagel & Wilson (2011)](https://doi.org/10.1038/nn.2725).

**Development versus lifetime; local algorithm.** Development fixes time constants and filter signs. Proposed local state rule: a_next = (1-alpha)*a + alpha*x; output = phi(x - beta*a). This is activity-state adaptation, not a weight update. Do not relabel every adaptation time constant as a separate mechanism.

**Shared parameters, independent state and scaling.** H = O(1–10) per cell class; O(N) adaptive state and work for one filter per channel. K time constants require O(KN) state even if coefficients are shared.

**Models and what they establish.** Nagel & Wilson (2011), Nature Neuroscience: dynamical account of odor-driven responses. It explains temporal response structure, not symbolic transfer.

**Possible benefit and boundary.** Can suppress persistent nuisance drive while preserving transients; can also erase a sustained cue that matters. It does not solve format invariance without a suitable coding relation.

**Overlap / current project status.** The historical ledger already contains temporal-contrast/fast-minus-slow sensory variants. This is a source-grounded reference, not an untested recommendation.

**One matched comparison and one measurement.** Adaptive filter versus an equal-state fixed filter, both with matched downstream learning. Measurement: held-out accuracy when only nuisance stimulus persistence changes.

**Local-rule sufficiency.** Fixed computation; learning sufficiency not applicable.

**Confidence.** High for temporal filtering; conditional for task usefulness.

### F03. Learned inhibition of familiar background

**Category:** I. Input conditioning. **Species/stage:** Adult Drosophila melanogaster

**What it computes and what is evidenced.** Shown/inferred: targeted experiments implicate local inhibitory plasticity in odor-specific habituation. This differs from F02: the relevant effective inhibition changes with experience rather than merely tracking a fast adaptation state. A complete synapse-by-synapse learning equation was not measured. Sudhakaran et al. (2012) provides a follow-up dissection of recurrent inhibitory pathways, but is not independent replication by a wholly separate research group. [Das et al. (2011)](https://doi.org/10.1073/pnas.1106411108); [Sudhakaran et al. (2012)](https://doi.org/10.1523/JNEUROSCI.1099-12.2012).

**Verification / follow-up status.** Primary study plus targeted follow-up from overlapping research lineage; not an independent replication claim.

**Wiring and signs.** Adult fly antennal lobe: projection neurons interact with GABAergic local interneurons that suppress odor responses. Represent the plastic inhibitory connection as a nonnegative magnitude with an explicitly negative output sign. [Das et al. (2011)](https://doi.org/10.1073/pnas.1106411108); [Sudhakaran et al. (2012)](https://doi.org/10.1523/JNEUROSCI.1099-12.2012).

**Development versus lifetime; local algorithm.** Development fixes partners. A proposed, not experimentally identified, inhibitory update is: delta inhibitory_magnitude = eta * local_presynaptic_activity * (target_activity - target_rate). It increases inhibition when the target is too active. A familiarity/exposure gate must be locally supplied if used; semantic irrelevance cannot be assumed.

**Shared parameters, independent state and scaling.** H = O(1–10); S_I plastic inhibitory magnitudes plus O(N) rate state, O(S_I) work. Sharing all magnitudes by type would remove the specificity that makes this useful.

**Models and what they establish.** Das et al. (2011), PNAS: circuit manipulations and behavioral habituation. The proposed rate-target rule here is an engineering candidate, not a reproduction of the animal mechanism.

**Possible benefit and boundary.** Could reduce interference from repeated formatting or background cues. Failure mode: suppressing a frequent but essential operand or delimiter.

**Overlap / current project status.** Overlaps inhibitory adaptation and existing input-side plasticity. No claim that a particular MiniFly implementation is untested.

**One matched comparison and one measurement.** Adaptive inhibitory weights versus frozen equal-size inhibitory weights. Measurement: retained task accuracy after exposure to a repeated, task-irrelevant background.

**Local-rule sufficiency.** Unknown for the biological task under the proposed rule; unestablished for MiniFly.

**Confidence.** Medium: causal circuit evidence is stronger than knowledge of the update rule.

### F04. Input-side timing-dependent plasticity

**Category:** II. Representation and anchoring. **Species/stage:** Adult Drosophila melanogaster

**What it computes and what is evidenced.** Shown: timing-dependent changes at incoming PN→KC synapses can facilitate olfactory learning, with dopaminergic involvement. This is not merely KC→MBON valence writing at the output. It supplies a specific alternative locus for adapting representations. [Qiao et al. (2022)](https://doi.org/10.1016/j.cub.2022.09.054).

**Verification / follow-up status.** One primary causal/physiological study selected; independent replication not established here.

**Wiring and signs.** Adult fly: excitatory projection-neuron input reaches KC dendrites; dopaminergic signals modulate input plasticity. Keep the anatomical site separate from downstream KC→MBON writes. [Qiao et al. (2022)](https://doi.org/10.1016/j.cub.2022.09.054).

**Development versus lifetime; local algorithm.** Development fixes sparse partners; lifetime PN→KC magnitudes are plastic. Proposed abstraction: delta w_ij = eta * local_timing_kernel(pre_i, post_j) * local_modulator_j, followed by sign-preserving bounds. The paper supports timing sensitivity, not this exact universal factorization or a particular numerical kernel.

**Shared parameters, independent state and scaling.** H = O(10) kernel/gain parameters; S_in independent weights and O(S_in) eligibility if traces are synapse-specific. A neuron-shared trace can reduce storage only when the chosen rule permits it.

**Models and what they establish.** Qiao et al. (2022), Current Biology: recordings and olfactory-learning experiments. No verified general byte-learning code or demonstrated abstract-rule induction is supplied by that study.

**Possible benefit and boundary.** Could retune a representation instead of continually overwriting its readout. It might improve salience and sample efficiency, but also destabilize old readout meanings.

**Overlap / current project status.** Explicitly present as ITDP/M10 in the September pruning ledger. Keep as a mechanistically specified variant, not a newly found idea.

**One matched comparison and one measurement.** Input-plastic versus input-frozen encoders, matched in storage and downstream updates. Measurement: old-format held-out accuracy after learning the new format.

**Local-rule sufficiency.** Partial: a local plastic site is supported; end-to-end sufficiency is not.

**Confidence.** Medium: identified plastic site and timing effect; algorithmic sufficiency unresolved.

### F05. Timing-sensitive, signed associative writing

**Category:** III. Teaching and credit. **Species/stage:** Adult Drosophila melanogaster

**What it computes and what is evidenced.** Shown: odor-linked KC→MBON plasticity can be induced by appropriately timed dopamine, and timing can change the sign of the resulting association/plasticity. These findings justify a timing-sensitive local teaching mechanism. They do not establish a universal pre×post×dopamine rule at every such synapse. [Hige et al. (2015)](https://doi.org/10.1016/j.neuron.2015.11.003); [Handler et al. (2019)](https://doi.org/10.1016/j.cell.2019.05.040).

**Verification / follow-up status.** Convergent primary studies of local plasticity and timing; exact equations differ and are not collapsed into one measured rule.

**Wiring and signs.** Adult fly: cholinergic KCs excite MBONs; compartmental DANs modulate the KC→MBON connections. The teaching input is not the same thing as an excitatory postsynaptic current. [Hige et al. (2015)](https://doi.org/10.1016/j.neuron.2015.11.003); [Handler et al. (2019)](https://doi.org/10.1016/j.cell.2019.05.040).

**Development versus lifetime; local algorithm.** Development fixes compartments; life changes individual KC→MBON magnitudes. Proposed abstraction: eligibility_next = lambda*eligibility + local_activity_event; delta w = eta*eligibility*signed_local_teaching_signal. A timing kernel or opponent teaching channels determine sign. The effective predictor update must be translated consistently into depression/potentiation of an excitatory physical synapse.

**Shared parameters, independent state and scaling.** H = O(1–10) per compartment class; S_w plastic weights and up to S_w eligibility traces. Work O(S_active) with sparse events, otherwise O(S_w). Long-lived capture state is additional, not synonymous with eligibility.

**Models and what they establish.** Hige et al. (2015) and Handler et al. (2019) provide experimental constraints. Bennett et al. (2021; F08) shows one computational use of prediction-driven MB plasticity, not a proof that this generic rule solves arbitrary tasks.

**Possible benefit and boundary.** Delayed local credit and rapid associations. A scalar reward plus eligibility does not by itself identify which hidden representation should be changed to generalize.

**Overlap / current project status.** Baseline/retained family: local writing and eligibility already exist in the historical project. Synaptic tagging, eligibility and consolidation should not be counted as synonyms or automatically as three new modules.

**One matched comparison and one measurement.** Timed eligibility versus a shuffled-time, equal-storage eligibility control. Measurement: prediction accuracy at a fixed cue-to-reward delay.

**Local-rule sufficiency.** Partial; sufficient for specified simple associative models, not established for the target learner.

**Confidence.** High for timing-dependent associative plasticity; low for inferring general rule learning from it.

### F06. Compartment-specific teaching channels

**Category:** III. Teaching and credit. **Species/stage:** Adult Drosophila melanogaster; F06 additionally labels larval evidence explicitly.

**What it computes and what is evidenced.** Shown: adult DAN cell types write and update memories with different dynamics. Larval connectomics and modeling show substantial recurrent organization around teaching neurons. This supports heterogeneous teaching channels; it does not establish independent, arbitrary target vectors for every neuron. [Aso & Rubin (2016)](https://doi.org/10.7554/eLife.16135); [Eschbach et al. (2020)](https://doi.org/10.1038/s41593-020-0607-9).

**Verification / follow-up status.** Adult cell-type evidence plus larval architecture/model evidence; cross-stage convergence is not adult replication.

**Wiring and signs.** Adult: distinct DAN projections modulate different KC→MBON compartments. Larval evidence supplies additional recurrent motifs but is explicitly cross-stage evidence. Connection signs must be specified per selected motif, not inferred from the word dopamine. [Aso & Rubin (2016)](https://doi.org/10.7554/eLife.16135); [Eschbach et al. (2020)](https://doi.org/10.1038/s41593-020-0607-9).

**Development versus lifetime; local algorithm.** Development sets the routing matrix and a small number C of channel types. Proposed lifetime update: delta w_ic = eta_c * e_ic * d_c. Each d_c must be computed by its local circuit from allowed inputs. C independent external target errors are not silently equivalent to one scalar reward.

**Shared parameters, independent state and scaling.** H = O(C) rule coefficients, potentially O(C²) intercompartment couplings; O(S_w) plastic weights and traces. Scaling C with width increases teaching bandwidth and must be reported.

**Models and what they establish.** Aso & Rubin (2016) supplies cell-type-specific learning constraints. Eschbach et al. (2020), Nature Neuroscience, combines larval recurrent anatomy and computational analysis; biological wiring alone does not certify all training operations as local.

**Possible benefit and boundary.** Could specialize where credit is delivered and separate stable from flexible channels. It need not improve composition unless useful information reaches the appropriate channel.

**Overlap / current project status.** Already-covered compartment heterogeneity/teaching allocation. More compartments are not automatically new mechanisms.

**One matched comparison and one measurement.** Compartment-targeted versus channel-permuted modulation, matching modulator amplitudes and all state. Measurement: held-out loss after conflicting teaching events.

**Local-rule sufficiency.** Partial: local modulation exists; sufficiency of learned routing remains task-dependent.

**Confidence.** High for heterogeneity; medium for a chosen routing abstraction.

### F07. Activity-dependent amplification of teaching

**Category:** III. Teaching and credit. **Species/stage:** Adult Drosophila melanogaster

**What it computes and what is evidenced.** Shown: reciprocal KC–DAN connectivity includes a positive feedback contribution required for learning. This is an amplification loop, not evidence that the loop subtracts an expected reward. [Cervantes-Sandoval et al. (2017)](https://doi.org/10.7554/eLife.23789).

**Verification / follow-up status.** One selected primary circuit study; no independent replication claim.

**Wiring and signs.** Adult fly: KCs provide excitatory cholinergic feedback to DANs; DANs modulate KC output synapses. These are different signal types around the loop. [Cervantes-Sandoval et al. (2017)](https://doi.org/10.7554/eLife.23789).

**Development versus lifetime; local algorithm.** Development fixes feedback weights and saturation. Proposed dynamics: d_next = clip(external_teaching + gain*sum(local_KC_activity), 0, d_max). Lifetime KC→MBON plasticity remains F05. A learned feedback matrix would be another plastic site requiring its own rule.

**Shared parameters, independent state and scaling.** H = O(1–10) gain/decay/bound parameters, O(C) DAN state and O(S_feedback) edges/work. Feedback adjacency and any independent feedback weights must be counted.

**Models and what they establish.** Cervantes-Sandoval et al. (2017), eLife: structural/functional intervention evidence. No verified stand-alone byte learner or policy-gradient equivalence is claimed.

**Possible benefit and boundary.** Could improve salience-dependent write strength or credit delivery with weak teaching. Positive feedback can instead amplify noise and unrelated writes.

**Overlap / current project status.** Overlaps the retained MB output/teaching loop. Do not conflate this operation with the error subtraction in F08 or the transfer of teacher status in F09.

**One matched comparison and one measurement.** Closed loop versus activity-independent, gain-matched teaching. Measurement: examples needed for a fixed trained-association accuracy.

**Local-rule sufficiency.** Partial; local components are available, global sufficiency not established.

**Confidence.** Medium: supported circuit motif; useful gain schedule is task-specific.

### F08. Prediction-error computation in an MB output–teaching loop

**Category:** III. Teaching and credit. **Species/stage:** Adult Drosophila melanogaster

**What it computes and what is evidenced.** Proposed and tested in models: output-neuron predictions feed back to dopamine-like teaching signals, allowing error-driven rather than absolute-reinforcement updates. Conditioning and blocking simulations are evidence for the model’s behavior, not direct proof that all fly DANs implement this calculation. [Bennett, Philippides & Nowotny (2021)](https://doi.org/10.1038/s41467-021-22592-4).

**Verification / follow-up status.** Published computational task demonstrations and verified code availability; not experimental replication of biological RPE computation.

**Wiring and signs.** Model: KC activity drives output predictions; effective feedback opposes external reinforcement at teaching units. Positive and negative signals require opponent/sign-constrained pathways rather than an unexplained negative dopamine concentration. [Bennett, Philippides & Nowotny (2021)](https://doi.org/10.1038/s41467-021-22592-4).

**Development versus lifetime; local algorithm.** Development fixes circuit signs/routing; life changes effective predictor weights. Minimal local model: prediction = sum_i(v_i*x_i); delta = reward - prediction; e_i_next = lambda*e_i + x_i; delta v_i = eta*delta*e_i. The target/output orientation determines the corresponding physical KC→MBON update sign. No hidden-layer gradient follows from this equation.

**Shared parameters, independent state and scaling.** H = O(1–10) per channel; S predictor weights, up to S traces, O(S) work. A single output can use scalar error; many outputs need specified local/error routing, not uncounted supervision.

**Models and what they establish.** Bennett, Philippides & Nowotny (2021), Nature Communications: explicit constrained/augmented models and released simulation code. Fits multiple associative-learning phenomena; some explanatory connectivity is postulated. Code existence verified; not rerun here.

**Possible benefit and boundary.** Prediction calibration and reduced unnecessary writes once a consequence is predicted. Useful for sample efficiency; not a solution to latent-variable discovery or symbolic compositionality.

**Overlap / current project status.** Potentially an already implemented teacher-quality mechanism. The exact current MiniFly feedback computation has not been re-audited in this review.

**One matched comparison and one measurement.** Prediction-error feedback versus absolute teaching with matched update norm and state budget. Measurement: asymptotic consequence-prediction error.

**Local-rule sufficiency.** Yes in the published model’s specified associative tasks; unknown for MiniFly transfer.

**Confidence.** High for the stated computational result; medium/low for the full biological interpretation.

### F09. A learned cue becomes a teacher for another association

**Category:** III. Teaching and credit. **Species/stage:** Adult Drosophila melanogaster

**What it computes and what is evidenced.** Shown: second-order conditioning uses a hierarchy in which a learned reward-predicting cue can recruit teaching for another compartment. Rachad et al. (2025) supplies convergent circuit/behavioral/model evidence for higher-order associative chains. This is stronger than merely observing anatomical recurrence, but remains distinct from learning a reusable arithmetic operation. [Yamada et al. (2023)](https://doi.org/10.7554/eLife.79042); [Rachad et al. (2025)](https://doi.org/10.1016/j.celrep.2025.115593).

**Verification / follow-up status.** Two primary circuit studies from different research groups support higher-order association; they are convergent, not identical-protocol replications.

**Wiring and signs.** Adult fly: a relatively stable teacher compartment influences faster student compartments through an excitatory intermediary; additional experiments identify excitatory and inhibitory feedback around dopaminergic signaling. Avoid treating all routes as one universal fixed diagram. [Yamada et al. (2023)](https://doi.org/10.7554/eLife.79042); [Rachad et al. (2025)](https://doi.org/10.1016/j.celrep.2025.115593).

**Development versus lifetime; local algorithm.** Development fixes teacher→intermediary→student routes. Proposed abstraction: d_student = external_reward + alpha*teacher_output; delta w_student = eta*student_eligibility*d_student. Teacher output must be learned from experience, not supplied from labels at test time. The full biological update and safeguards against self-reinforcing false predictions remain incompletely identified.

**Shared parameters, independent state and scaling.** H = O(C) channel gains and time constants, O(S_teacher+S_student) plastic state, plus bounded intermediary state. Dense teacher–student connectivity costs O(C²); sparse hierarchy O(C).

**Models and what they establish.** Yamada et al. (2023), eLife: anatomical/physiological/behavioral work with modeling. Rachad et al. (2025), Cell Reports: circuit dissection plus computational modeling. A public reusable model-code package was not verified here; anatomical viewer links are not model code.

**Possible benefit and boundary.** Composes learned associations without direct primary reward for every link. This supports an operational teaching-transfer hypothesis, not a claim of zero-shot algebra.

**Overlap / current project status.** Subfamily of existing MB feedback and fast/slow compartments, but teaching transfer is an identifiable extra operation. Current exact implementation status unknown.

**One matched comparison and one measurement.** Learned teacher output versus equal-amplitude shuffled teacher output, after identical first-order training. Measurement: second-order cue preference without primary reward to that cue.

**Local-rule sufficiency.** Partial: task-level circuit/model evidence; no general sufficiency theorem.

**Confidence.** Medium–high for associative chaining; low for extrapolation to general rule induction.

### F10. State-dependent readout and disinhibitory routing

**Category:** IV. Readout and revision. **Species/stage:** Adult Drosophila melanogaster

**What it computes and what is evidenced.** Shown: motivational state can gate memory expression, and learning/motivation can alter feed-forward inhibition in the MB output network. Failure to express a behavior is therefore not automatically loss of the stored association. [Krashes et al. (2009)](https://pubmed.ncbi.nlm.nih.gov/19837040/); [Perisse et al. (2016)](https://doi.org/10.1016/j.neuron.2016.04.034).

**Verification / follow-up status.** Convergent behavioral/circuit studies; does not establish an arbitrary learned router.

**Wiring and signs.** Adult fly: state-related signals influence dopaminergic/MB circuits; inhibitory MBON pathways regulate other output pathways. An inhibitory neuron suppressing an inhibitory route can disinhibit the selected output. [Krashes et al. (2009)](https://pubmed.ncbi.nlm.nih.gov/19837040/); [Perisse et al. (2016)](https://doi.org/10.1016/j.neuron.2016.04.034).

**Development versus lifetime; local algorithm.** Development fixes routes. Proposed readout: output = g(state)*memory_output, with a small fixed state-to-gain map; a more explicit version uses two inhibitory stages. The evidence does not specify a generally sufficient local rule for learning arbitrary context keys. Changing state can alter reads without changing stored weights.

**Shared parameters, independent state and scaling.** H = O(C) gains for fixed state channels, O(C+N) activity. Learning a context router adds S_gate weights and its local learning state; do not count only the final gain.

**Models and what they establish.** Krashes et al. (2009), Cell, and Perisse et al. (2016), Neuron: causal circuit/behavioral constraints. Neither is a demonstration of locally trained transformer-style keys.

**Possible benefit and boundary.** Could separate storage from access and prevent irrelevant context from determining the output. It offers a hypothesis for format-dependent access failures, not proof that E16 is an access failure.

**Overlap / current project status.** Existing internal-state gating family and overlaps the separate attention/WM survey. No new generic “add a gate” recommendation.

**One matched comparison and one measurement.** State-sensitive readout versus state-permuted readout, holding all stored association weights fixed at evaluation. Measurement: recovery of old-format accuracy.

**Local-rule sufficiency.** Fixed gate is straightforward; learning the correct gate is unestablished.

**Confidence.** High for state-dependent expression; low for arbitrary learned routing sufficiency.

### F11. Opposing learned memories rather than overwriting one trace

**Category:** IV. Readout and revision. **Species/stage:** Adult Drosophila melanogaster

**What it computes and what is evidenced.** Shown: extinction can involve a parallel opposing memory that competes with the original association. This is not the claim that all forgetting is erasure or that every inhibitory response is a separate memory system. [Felsenberg et al. (2018)](https://doi.org/10.1016/j.cell.2018.08.021).

**Verification / follow-up status.** One selected primary circuit study; no claim of universal extinction mechanism.

**Wiring and signs.** Adult fly: distinct MB pathways preserve and oppose learned valence, converging on behavioral output. Keep route-specific excitation/inhibition separate from the sign of the net learned preference. [Felsenberg et al. (2018)](https://doi.org/10.1016/j.cell.2018.08.021).

**Development versus lifetime; local algorithm.** Development fixes two opposing channels. Proposed effective output: value = original_prediction - opponent_prediction. Both channels can use bounded locally modulated associative updates; omitted expected reinforcement trains the opponent only if that omission signal is explicitly computed. This equation is an abstraction, not the measured complete biological rule.

**Shared parameters, independent state and scaling.** H = O(1–10); approximately 2S associative weights if both channels have S edges, plus traces if used. Parameter sharing does not remove the second stored trace.

**Models and what they establish.** Felsenberg et al. (2018), Cell: circuit and behavioral evidence for parallel opposing memory. No claim that a generic two-head implementation reproduces every extinction phenomenon.

**Possible benefit and boundary.** Revision with potential recovery of old behavior instead of irreversible replacement. It can reduce one kind of forgetting but does not discover invariant input representations.

**Overlap / current project status.** Overlaps banks, compartmental retention and revision already screened. Include as a mechanistic distinction, not another broad memory-management proposal.

**One matched comparison and one measurement.** Opponent trace versus single-trace updating under equal total writable storage. Measurement: old-association recovery after removal of the extinction context.

**Local-rule sufficiency.** Partial: local implementation plausible; particular rule must be demonstrated.

**Confidence.** High for the reported extinction mechanism; conditional for format-transfer failures.

### F12. Bounded post-event teaching reverberation

**Category:** V. Consolidation control. **Species/stage:** Adult Drosophila melanogaster

**What it computes and what is evidenced.** Shown/inferred: recurrent reward-related activity contributes to appetitive long-term memory, while inhibitory feedback helps regulate dopaminergic activity. These studies concern different circuit details, so their combination into one controller is proposed. Matsuno et al. (2026) links aberrant post-training dopamine to maladaptive memory broadening, not useful rule generalization. [Ichinose et al. (2015)](https://doi.org/10.7554/eLife.10719); [Pavlowsky, Schor, Plaçais & Preat (2018)](https://doi.org/10.1016/j.cub.2018.04.040); [Matsuno et al. (2026)](https://doi.org/10.1371/journal.pbio.3003752).

**Verification / follow-up status.** Several primary studies constrain persistence, termination and overgeneralization, but not one jointly validated controller.

**Wiring and signs.** Adult fly: recurrent MB–DAN signaling supplies persistence; a GABAergic feedback pathway regulates dopaminergic input. Do not assert a direct monosynaptic inhibitory edge where only a functional pathway is established. [Ichinose et al. (2015)](https://doi.org/10.7554/eLife.10719); [Pavlowsky, Schor, Plaçais & Preat (2018)](https://doi.org/10.1016/j.cub.2018.04.040); [Matsuno et al. (2026)](https://doi.org/10.1371/journal.pbio.3003752).

**Development versus lifetime; local algorithm.** Development fixes loop gains and negative-feedback topology. Proposed two-state controller: d_next = clip((1-a)*d + b*event - c*z,0,d_max); z_next = (1-h)*z + h*d. d gates already eligible local writes. Eligibility must decay or reset so unrelated later inputs are not captured. This is bounded teaching persistence, not an archive replay buffer.

**Shared parameters, independent state and scaling.** H = O(10) per channel, O(C) controller state plus existing weight/eligibility storage. K post-event updates multiply work by K even if no weights are added.

**Models and what they establish.** Ichinose et al. (2015), eLife, and Pavlowsky et al. (2018), Current Biology: experimental constraints on persistence and feedback. The two-state engineering controller is proposed here, not claimed as their exact model.

**Possible benefit and boundary.** Can extend an informative teaching event while terminating harmful indiscriminate writing. It addresses write timing; it cannot recreate missing old examples.

**Overlap / current project status.** Already-covered consolidation/termination family. Keep separate from replay but do not launch it instead of the user-described replay controls.

**One matched comparison and one measurement.** Negative-feedback-terminated versus duration-matched externally truncated teaching, equal total update budget. Measurement: old-task loss after unrelated post-event inputs.

**Local-rule sufficiency.** Partial; no established complete local algorithm for the MiniFly failure.

**Confidence.** Medium: component findings are strong; synthesis and transfer remain unproven.

### F13. Re-anchor new sensory renderings to a stable internal coordinate

**Category:** II. Representation and anchoring. **Species/stage:** Adult Drosophila melanogaster

**What it computes and what is evidenced.** Shown: the visual input associated with an internal heading representation can be reorganized by sensorimotor experience. Convergent 2019 studies connect stable heading dynamics with adaptable visual mapping. The exact synaptic learning law is inferred/proposed, not directly measured in full. A December 2025 preprint by Plitt et al. reports that octopaminergic feedback conveys heading activity to sensory presynaptic terminals and is necessary/sufficient for the tested anchoring plasticity; this newer claim remains provisional. [Fisher, Lu, D’Alessandro & Wilson (2019)](https://doi.org/10.1038/s41586-019-1772-4); [Kim, Hermundstad, Romani, Abbott & Jayaraman (2019)](https://doi.org/10.1038/s41586-019-1767-1); [Plitt et al. (2025)](https://doi.org/10.64898/2025.12.11.693783).

**Verification / follow-up status.** Two peer-reviewed 2019 studies converge on flexible anchoring; 2025 feedback mechanism is a separately labeled, unreplicated preprint.

**Wiring and signs.** Adult fly: visual ring-neuron input inhibits compass E-PG neurons; the compass has recurrent state-maintaining dynamics. Plastic sensory anchoring is distinguished from the recurrent dynamics that maintain the internal coordinate. [Fisher, Lu, D’Alessandro & Wilson (2019)](https://doi.org/10.1038/s41586-019-1772-4); [Kim, Hermundstad, Romani, Abbott & Jayaraman (2019)](https://doi.org/10.1038/s41586-019-1767-1); [Plitt et al. (2025)](https://doi.org/10.64898/2025.12.11.693783).

**Development versus lifetime; local algorithm.** Development fixes the compass kernel; life changes sensory-to-state anchoring. Proposed inhibitory-association variant: delta a_ij = -eta*visual_i*state_j + slow_local_restoration(a_ij), with nonnegative inhibitory magnitudes. Weakening coactive inhibition can anchor the cue to the occupied state. The exact rule/bounds need validation; a privileged correct-state target is not allowed. The preprint motivates replacing direct state_j in the update by a locally delivered modulatory state signal m_j. That spatially structured feedback is not a single uniform global reward scalar and must be explicitly implemented and budgeted.

**Shared parameters, independent state and scaling.** H = O(10) kernel/rule parameters plus S_anchor independent anchoring weights and O(N) state. Dense M-cue by N-state anchoring costs O(MN); sparse degree-k anchoring O(kM). It is not a ten-parameter learner merely because the ring kernel has ten parameters.

**Models and what they establish.** Fisher et al. (2019) and Kim et al. (2019), Nature: physiological experiments with constrained models. Kim et al. supplies an author data/code page. The task has continuous heading and sensorimotor correspondence, not arbitrary symbolic renaming.

**Possible benefit and boundary.** A plausible analogue for separating a stable latent computation from a changing input format. It requires an observable bridge between formats; it cannot infer an arbitrary unseen permutation with no paired cues or feedback.

**Overlap / current project status.** Family overlap with input plasticity and the already-tested CX. Whether the exact sensory-anchor variant was tested is unknown; it must not be announced as new.

**One matched comparison and one measurement.** Plastic versus frozen anchoring with identical recurrent core and allowed observations. Measurement: old-format accuracy after learning a second rendering of the same observable transitions.

**Local-rule sufficiency.** Partial: local models exist; target-task sufficiency and unprivileged anchoring unestablished.

**Confidence.** Medium–high for sensory remapping; conditional for byte-format transfer.

### F14. Continuous recurrent working state

**Category:** VI. Fixed computation substrates. **Species/stage:** Adult Drosophila melanogaster

**What it computes and what is evidenced.** Shown/inferred through perturbation and modeling: fly heading populations have ring-attractor dynamics. This supports persistent low-dimensional state. It does not establish a locally learned general working-memory program. [Kim, Rouault, Druckmann & Jayaraman (2017)](https://doi.org/10.1126/science.aal4835).

**Verification / follow-up status.** Perturbation plus model in a primary paper; code verified, not independently rerun here.

**Wiring and signs.** Adult fly central brain: E-PG-centered recurrent compass circuitry with effective excitation and inhibition maintains a localized population state. A signed ring kernel is an abstraction; negative effective couplings must be realized through inhibitory pathways. [Kim, Rouault, Druckmann & Jayaraman (2017)](https://doi.org/10.1126/science.aal4835).

**Development versus lifetime; local algorithm.** Development fixes a translation-structured recurrent kernel. Proposed dynamics: h_next = phi(K*h + B*motion - inhibition); the stable bump encodes state. No lifetime weight changes are necessary for this computation. Sensory anchoring plasticity is F13, not evidence of a plastic ring kernel.

**Shared parameters, independent state and scaling.** H = O(1–10) kernel/shift parameters; O(N) activity, O(kN) work with k local neighbors or O(N²) for a naive dense kernel. Multiple independent registers multiply state and dynamics; shared coefficients do not provide free memory capacity.

**Models and what they establish.** Kim et al. (2017), Science: perturbation-constrained model with released code. It explains heading-bump behavior, not arbitrary manipulation of multiple symbolic registers.

**Possible benefit and boundary.** Persistent state and continuous integration with low shared-parameter cost. Useful only when inputs can legitimately be grounded to the state variable.

**Overlap / current project status.** Already-reviewed/tested central-complex family. Exact implementation details were not inspected here.

**One matched comparison and one measurement.** Structured versus phase-scrambled recurrence, matched degree, gains, state and internal update count. Measurement: drift over a specified delay.

**Local-rule sufficiency.** Not applicable to the fixed kernel; grounding/readout learning remains separate.

**Confidence.** High for the specified state dynamics; low for general program inference.

### F15. Accumulate evidence before committing

**Category:** VI. Fixed computation substrates. **Species/stage:** Adult Drosophila melanogaster

**What it computes and what is evidenced.** Shown: subthreshold dendritic activity in αβ-core KCs integrates olfactory evidence on behaviorally relevant decision timescales. This is a within-neuron temporal computation, not evidence that recurrent weights were learned for the task. [Groschner et al. (2018)](https://doi.org/10.1016/j.cell.2018.03.075).

**Verification / follow-up status.** Primary recordings/manipulations with mechanistic modeling; independent replication not established here.

**Wiring and signs.** Adult fly: sensory inputs reach KC dendrites; membrane integration precedes spike/decision threshold. A simplified circuit uses excitatory evidence drive, leak and threshold/reset. [Groschner et al. (2018)](https://doi.org/10.1016/j.cell.2018.03.075).

**Development versus lifetime; local algorithm.** Development fixes integration/threshold parameters. Proposed state rule: z_next = (1-leak)*z + evidence; commit when z reaches threshold, then reset. A signed evidence difference requires separately specified opponent inputs. No weight plasticity is required for the accumulator.

**Shared parameters, independent state and scaling.** H = O(1–10) per accumulator class; O(N) state and O(S_input+N) work. K integration steps cost K updates. Arbitrary learned evidence encoders add their own weights and credit-assignment problem.

**Models and what they establish.** Groschner et al. (2018), Cell: recordings, manipulations and a perceptual-decision account. The model addresses discrimination timing, not long symbolic reasoning.

**Possible benefit and boundary.** Can integrate noisy or distributed cues and delay action until sufficient evidence accumulates. This is a different operation from storing an arbitrary item without drift.

**Overlap / current project status.** Overlap with the recurrent-state/integration family; more specific than adding a generic RNN. Current exact implementation unknown.

**One matched comparison and one measurement.** Accumulator versus matched-state instantaneous decision mechanism, equal total observations and compute. Measurement: decision error at a fixed observation budget.

**Local-rule sufficiency.** Fixed computation; sufficiency of a learned encoder is unknown.

**Confidence.** High for evidence integration; medium for the proposed compact abstraction.

### F16. Coordinate transformation by population-vector combination

**Category:** VI. Fixed computation substrates. **Species/stage:** Adult Drosophila melanogaster

**What it computes and what is evidenced.** Shown/inferred from recordings and circuit organization: fan-shaped-body pathways combine movement-related signals to represent allocentric travelling direction. This is a coordinate transformation, not proof of a complete learned home-vector integrator or universal addition engine. Ishida et al. (2026) further shows a population-vector inversion operation associated with a change in firing mode. Inversion is recorded here as a distinct operator within the vector-computation family, not padded into another broad family. [Lyu, Abbott & Maimon (2022)](https://doi.org/10.1038/s41586-021-04067-0); [Ishida, Sethi, Mohren, Haraguchi, Abbott & Maimon (2026)](https://doi.org/10.1016/j.cell.2025.11.040).

**Verification / follow-up status.** Primary vector-transformation and later inversion studies; complementary operations, not replicated end-to-end learning.

**Wiring and signs.** Adult fly: heading and movement signals shape PFN population phases/amplitudes; phase-shifted convergence onto hΔB-related populations combines components. Do not replace the biological arrangement with an unsupported universal eight-column ratio. [Lyu, Abbott & Maimon (2022)](https://doi.org/10.1038/s41586-021-04067-0); [Ishida, Sethi, Mohren, Haraguchi, Abbott & Maimon (2026)](https://doi.org/10.1016/j.cell.2025.11.040).

**Development versus lifetime; local algorithm.** Development fixes phase offsets and coupling. Mathematical abstraction: world_vector = rotation(heading) * body_vector; a population implementation sums appropriately shifted component codes. This is a runtime operation; no local learning law is needed for fixed couplings. Learning what a raw byte denotes remains outside the circuit. For the inversion variant, the mathematical operation is v_out = -v_in, represented as a half-turn of population phase; this does not imply negative firing rates.

**Shared parameters, independent state and scaling.** H = O(10–100) across selected population types; O(N) population state and O(S_vector) edges/work. Dense implementations may cost O(N²); sparse columnar implementations can cost O(N). The low-dimensional formula does not eliminate neural encoding cost.

**Models and what they establish.** Lyu, Abbott & Maimon (2022), Nature: circuit-linked computational account of travelling direction. It does not test locally learned modular arithmetic from arbitrary byte labels.

**Possible benefit and boundary.** Composition of grounded transformations. Potentially useful as a computational primitive, but inserting arithmetic structure into development changes what the learner is being asked to discover.

**Overlap / current project status.** Already-reviewed/tested CX family. No novelty claim for generic vector addition.

**One matched comparison and one measurement.** Correct versus permuted phase-offset circuitry with equal edges, training and update count. Measurement: coordinate-transform error on held-out trajectories.

**Local-rule sufficiency.** Fixed computation; learning sufficiency not applicable.

**Confidence.** High for the reported transformation; low for arbitrary symbolic-transfer claims.

### F17. Compare current state with a goal in a common coordinate

**Category:** VI. Fixed computation substrates. **Species/stage:** Adult Drosophila melanogaster

**What it computes and what is evidenced.** Shown: fly steering circuitry transforms heading/goal information into lateralized commands. Convergent 2024 studies identify how such comparisons can drive goal-oriented steering. Their findings do not supply a general local rule for learning all goals from sparse reward. [Mussells Pires et al. (2024)](https://doi.org/10.1038/s41586-023-07006-3); [Westeinde et al. (2024)](https://doi.org/10.1038/s41586-024-07039-2).

**Verification / follow-up status.** Two independent-group 2024 studies converge on steering computation; no inference of generic goal learning.

**Wiring and signs.** Adult fly: compass/goal-related populations converge through fan-shaped-body/PFL pathways onto left/right steering outputs. A compact opponent readout represents signed directional error; individual biological signs are pathway-specific. [Mussells Pires et al. (2024)](https://doi.org/10.1038/s41586-023-07006-3); [Westeinde et al. (2024)](https://doi.org/10.1038/s41586-024-07039-2).

**Development versus lifetime; local algorithm.** Development fixes comparison topology. Proposed abstraction: left = max(0, sin(goal-heading)); right = max(0, sin(heading-goal)). Goal is bounded activity state, not an unexplained task label. A learned goal-setting policy needs an additional local algorithm, which this comparison does not provide.

**Shared parameters, independent state and scaling.** H = O(10) shared couplings; O(N) activity for population codes and O(S_compare) work. A dense arbitrary goal embedding adds parameters; a formula for angular error is not itself a learned embedding.

**Models and what they establish.** Mussells Pires et al. (2024) and Westeinde et al. (2024), Nature: physiological/circuit models of steering. They support the comparator, not generic planning or symbolic goals.

**Possible benefit and boundary.** Provides an error direction for an already meaningful state space. It can support controlled iteration but does not determine what goal is correct.

**Overlap / current project status.** Part of the existing central-complex/goal family; not another independent source of generic learning capacity.

**One matched comparison and one measurement.** Goal-aligned versus goal-label-permuted comparator, matching all state/compute. Measurement: angular goal error on held-out starts.

**Local-rule sufficiency.** Not applicable to fixed comparison; learning goal selection remains unknown.

**Confidence.** High for steering comparison; conditional for nonspatial use.

### F18. Multiplicative interaction through excitation and disinhibition

**Category:** VI. Fixed computation substrates. **Species/stage:** Adult Drosophila melanogaster

**What it computes and what is evidenced.** Shown with recordings and conductance-based simulation: coincident excitation and release from inhibition produce a multiplication-like nonlinearity in T4 motion neurons. This is an analogue approximation over an operating range, not exact arbitrary signed multiplication. [Groschner, Malis, Zuidinga & Borst (2022)](https://doi.org/10.1038/s41586-022-04428-3).

**Verification / follow-up status.** Primary physiology, perturbation and conductance model; not a demonstrated general-purpose arithmetic unit.

**Wiring and signs.** Adult fly visual circuit: cholinergic inputs excite T4; glutamatergic Mi9 input acts inhibitory in this circuit. The interaction depends on local dendritic conductance, illustrating why transmitter identity alone is not a universal sign label. [Groschner, Malis, Zuidinga & Borst (2022)](https://doi.org/10.1038/s41586-022-04428-3).

**Development versus lifetime; local algorithm.** Development fixes input placement/gains. Proposed computational abstraction: output = phi(excitatory_input) * gate(local_disinhibition). Both signals must arrive physically at the neuron. No lifetime synaptic learning is required for the interaction; learning operands or routing into it is a separate problem.

**Shared parameters, independent state and scaling.** H = O(1–10) gains per motif; O(N) local activity. D input pairs require O(D) interactions; an unrestricted all-pairs bilinear layer costs O(D²). Shared gains do not make all pairings free.

**Models and what they establish.** Groschner et al. (2022), Nature: conductance-based model tested against visual responses and perturbations. No verified general local-rule learner follows from the model.

**Possible benefit and boundary.** Adds an actual conjunction/gating operation rather than another memory store. Could increase representational/computational expressivity while preserving local signal access.

**Overlap / current project status.** Potentially a specific variant within the previously reviewed gain/routing family; not established as untested in MiniFly.

**One matched comparison and one measurement.** Multiplicative versus additive interaction, equal input edges, states and training. Measurement: held-out accuracy on conjunctions whose components were seen separately.

**Local-rule sufficiency.** Fixed computation; learning to exploit it remains unestablished.

**Confidence.** High for the measured interaction; medium for the useful abstraction.

### F19. Predict and cancel self-generated sensory input

**Category:** VI. Fixed computation substrates. **Species/stage:** Adult Drosophila melanogaster

**What it computes and what is evidenced.** Shown: motor-related signals counteract expected visual consequences in fly visuomotor neurons. This supports an efference-copy/cancellation computation; it does not demonstrate a general predictive-coding learning rule. [Kim, Fitzgerald & Maimon (2015)](https://doi.org/10.1038/nn.4083).

**Verification / follow-up status.** Primary cellular evidence; learned cancellation rule remains a proposed extension.

**Wiring and signs.** Adult fly: motor-related signals converge with sensory signals in visual processing. Their effective sign/latency offsets expected reafferent drive. The detailed origin of every cancelling pathway need not be assumed for the abstraction. [Kim, Fitzgerald & Maimon (2015)](https://doi.org/10.1038/nn.4083).

**Development versus lifetime; local algorithm.** Development supplies a motor-to-sensory prediction map. Runtime: residual = sensory_input - predicted_input(action_copy). A possible lifetime adaptation delta w = eta*action_feature*local_residual is proposed, not established by the cited experiment. The residual must be present at the receiving neuron.

**Shared parameters, independent state and scaling.** H = O(10) for fixed low-dimensional gains; S_predict independent coefficients if the prediction map is learned, plus local trace state if delay matters. Work O(S_predict).

**Models and what they establish.** Kim, Fitzgerald & Maimon (2015), Nature Neuroscience: electrophysiological evidence. No claim that this study trained a general predictive model locally.

**Possible benefit and boundary.** Could distinguish an agent’s own output/expected feedback from external novelty. Relevant only when a causal action-feedback channel exists; subtracting predictable task signal can be harmful.

**Overlap / current project status.** Related to predictive/residual mechanisms, not a synonym for cortical backpropagation. Current exact MiniFly coverage unknown.

**One matched comparison and one measurement.** Action-aligned versus shuffled-action prediction, equal prediction resources. Measurement: residual false alarms during self-generated inputs.

**Local-rule sufficiency.** Fixed cancellation supported; learned-map sufficiency unknown.

**Confidence.** Medium–high for cancellation; low for the general learning extension.

### F20. Priority competition that produces a sequence

**Category:** VI. Fixed computation substrates. **Species/stage:** Adult Drosophila melanogaster

**What it computes and what is evidenced.** Shown/inferred through perturbation: competing grooming programs are organized by suppression relationships; removal of a dominant drive allows another program to emerge. This is innate behavioral sequencing, not demonstrated learning of arbitrary routines. [Seeds et al. (2014)](https://doi.org/10.7554/eLife.02951).

**Verification / follow-up status.** Behavioral perturbation plus operational model; complete learned neuronal controller not demonstrated.

**Wiring and signs.** Adult fly: parallel sensory-to-motor programs compete through effective suppression. Use a directed inhibitory priority graph as the abstraction; the behavioral model is not a complete cell-by-cell connectome. [Seeds et al. (2014)](https://doi.org/10.7554/eLife.02951).

**Development versus lifetime; local algorithm.** Development fixes priority edges. Proposed dynamics: select the highest currently active unsuppressed drive; execution reduces that drive, releasing the next program. No life weight learning is claimed. To learn a different priority graph would require an additional mechanism and rule.

**Shared parameters, independent state and scaling.** H = O(K) for a fixed ordered hierarchy, but up to O(K²) for an arbitrary competition graph; O(K) drive state. Work depends on graph sparsity.

**Models and what they establish.** Seeds et al. (2014), eLife: hierarchical suppression model and targeted manipulations of grooming sequences. It does not demonstrate learned planning.

**Possible benefit and boundary.** Provides a compact controller for multi-stage behavior without one dedicated parameterized layer per stage. It can also hard-code an entire solution, so developmental task leakage is a serious risk.

**Overlap / current project status.** Computational reference only. The historical screen explicitly excluded downstream motor programs from the persistent-memory core; this is not being promoted into the immediate plan.

**One matched comparison and one measurement.** Priority-consistent versus edge-permuted controller with equal graph size. Measurement: completion rate of the same fixed multi-stage routine.

**Local-rule sufficiency.** Not applicable to fixed sequencing; learned sequencing is unestablished.

**Confidence.** Medium for the operational controller; low as a general learning mechanism.

## Reserve and exclusion register—not additional counted mechanisms

**Biased developmental connectivity.** MacKenzie et al. (2025) reports systematic PN-type connectivity biases and a learning hierarchy. The retrieved record is a preprint. It is relevant to the choice of development-time wiring distribution, not evidence for a new online local rule. It also does not justify returning to generic random sparse expansion, which the project already has. [Primary preprint](https://doi.org/10.1101/2025.10.29.684686).

**Structural reorganization.** Baltruschat et al. (2021) reports mushroom-body calyx reorganization accompanying consolidation. This provides fly-specific evidence for experience-associated structural change, not a tested bounded local growth/pruning algorithm. Keep it with the existing structural/allocation reserve rather than counting spines, microglomeruli, growth, capture and allocation as five algorithms. [Primary article](https://doi.org/10.1016/j.celrep.2021.108871).

**Replay, active forgetting, memory banks, uncertainty protection, fast/slow state, tagging/capture and generic sparse expansion.** These remain in the historical overlap ledger, not the novel-candidate count. Their omission from the operational shortlist is a project-scope decision, not a claim that the phenomena are biologically absent. E16’s replay controls should not be interrupted merely because a biological label exists.

**Cross-species mechanisms.** Keep mammalian feedback alignment, cortical target/error propagation, PFC–BG gates, CLS and thalamocortical loops in the separate 11-family report. An analogy may be useful, but it is not an additional fly mechanism and does not repair an invalid citation.

## Crosswalk to the 11-family survey

This is a conceptual overlap map, not a claim that the detailed implementations are identical.

| Earlier survey family | Related records here | What is actually added by this review |
|---|---|---|
| Mushroom-body output loop | F05–F12 | Separate writing, channel routing, amplification, error subtraction, learned teaching, read gating, opponent revision and bounded reverberation. |
| Insect central complex | F13–F14, F16–F17 | Separate adaptive sensory anchoring from fixed state dynamics, coordinate operations and goal comparison; include a labeled preprint and vector-inversion follow-up. |
| Grid/ring attractors | F13–F14 | Fly evidence rather than transplanting entorhinal claims; account for anchoring weights separately. |
| Cortical feedback / actor–critic | F08–F09, F19 | Fly-grounded analogies, not evidence that fly circuits implement all cortical credit-assignment proposals. |
| Attention / working-memory gating | F01, F10, F14–F15, F18 | Fixed/local operations distinguished from learning arbitrary routing keys. |
| Sleep / CLS / consolidation | F11–F12 and reserves | No repackaging of replay or fast/slow stores as new; post-event teaching termination is not hippocampal replay. |
| Cerebellar supervised learning | No direct fly entry | Existing sparse expansion is not counted again. |
| Thalamocortical recurrence | F14–F15, F20 are analogies only | No assertion of fly thalamocortex or learned inner rehearsal. |

## Audit trail for all 70 entries in the reconstructed supplement

The table below audits **Executive Summary(2)**. It does **not** enumerate or recover the original missing 70 entries. “Merge” means conceptual overlap at the specified operational level, not proven identity of all biological details. “Exclude” concerns this fly/project scope, not whether a mammalian result is scientifically valid.

| Old # | Old label | Disposition | Destination | Reason |
|---:|---|---|---|---|
| 1 | Hebbian synaptic plasticity (LTP/LTD) | Merge/re-ground | F05; F06 | General plasticity labels are not independent fly algorithms. Use the identified site, timing and teaching signal; 5/25 are explicitly overlapping. |
| 2 | Spike-timing-dependent plasticity | Merge/re-ground | F05; F06 | General plasticity labels are not independent fly algorithms. Use the identified site, timing and teaching signal; 5/25 are explicitly overlapping. |
| 3 | BCM/metaplasticity | Archive | Existing stability/metaplasticity family | Generic stabilizer labels do not establish a new fly circuit or an implementable local rule. Molecular names are not separate algorithms. |
| 4 | Homeostatic synaptic scaling | Archive | Existing stability/metaplasticity family | Generic stabilizer labels do not establish a new fly circuit or an implementable local rule. Molecular names are not separate algorithms. |
| 5 | Heterosynaptic plasticity | Merge/re-ground | F05; F06 | General plasticity labels are not independent fly algorithms. Use the identified site, timing and teaching signal; 5/25 are explicitly overlapping. |
| 6 | Neuromodulator-gated three-factor plasticity | Merge/re-ground | F05; F06 | General plasticity labels are not independent fly algorithms. Use the identified site, timing and teaching signal; 5/25 are explicitly overlapping. |
| 7 | Structural synaptic plasticity / spine turnover | Archive with corrected species source | Baltruschat2021 | The old entry was mammalian spine turnover. A fly structural observation exists, but it does not supply a sufficient local edge-growth algorithm. |
| 8 | Mushroom-body associative coding | Merge | F05; F06 | Existing MB associative/compartment substrate; do not count output learning and its compartment label as wholly separate discoveries. |
| 9 | Central-complex ring attractor | Retain, split operation from learning | F14; F13 | Ring dynamics are fixed state computation; sensory anchoring plasticity is a distinct operation. |
| 10 | Entorhinal grid-cell coding | Exclude from fly evidence | Separate cross-species survey only | Mammalian/hippocampal evidence is not fly evidence. Some computational analogies were already reviewed elsewhere. |
| 11 | Hippocampal place cells | Exclude from fly evidence | Separate cross-species survey only | Mammalian/hippocampal evidence is not fly evidence. Some computational analogies were already reviewed elsewhere. |
| 12 | Hippocampal pattern separation | Exclude from fly evidence | Separate cross-species survey only | Mammalian/hippocampal evidence is not fly evidence. Some computational analogies were already reviewed elsewhere. |
| 13 | Global inhibition / winner-take-all | Merge with variant labels | F01 | Normalization, lateral inhibition and WTA overlap but are not mathematically identical. Preserve implementation variants within one competition/normalization family. |
| 14 | Mushroom-body KC→DAN feedback | Retain and constrain | F07 | KC→DAN positive feedback is not automatically reward-prediction-error subtraction. |
| 15 | Spike-frequency adaptation | Merge temporal-filter variants | F02 | Fast state adaptation differs from durable inhibitory habituation (F03); no doubling just for labels. |
| 16 | Backpropagating action potentials | Exclude as stated | Optional separate cortical/dendritic review | Mammalian dendritic mechanisms cannot be used as fly evidence. F18 has its own fly source and is not a validation of these claims. |
| 17 | Hippocampus–neocortex complementary learning | Exclude from fly evidence | Separate cross-species survey only | Mammalian/hippocampal evidence is not fly evidence. Some computational analogies were already reviewed elsewhere. |
| 18 | Pulvinar attention gating | Exclude from fly evidence | Separate 11-circuit survey | Cortical/basal-ganglia/thalamic mechanisms were part of a different survey; do not silently relabel them as fly mechanisms. |
| 19 | Synaptic tagging and capture | Merge then archive | One tag/capture abstraction | Entry 21 explicitly refers back to 19. Do not equate durable capture with eligibility or claim a verified fly implementation without a source. |
| 20 | Hippocampal replay | Archive / exclude cross-species claims | Existing replay/consolidation family | Several are overlapping replay/sleep/transfer descriptions. No exact duplicate count is inferred merely from family overlap; F12 is teaching persistence, not replay. |
| 21 | Synaptic tagging (see entry 19) | Merge then archive | One tag/capture abstraction | Entry 21 explicitly refers back to 19. Do not equate durable capture with eligibility or claim a verified fly implementation without a source. |
| 22 | CA3 pattern completion | Exclude from fly evidence | Separate cross-species survey only | Mammalian/hippocampal evidence is not fly evidence. Some computational analogies were already reviewed elsewhere. |
| 23 | Long-term depression | Merge/re-ground | F05; F06 | General plasticity labels are not independent fly algorithms. Use the identified site, timing and teaching signal; 5/25 are explicitly overlapping. |
| 24 | Intrinsic excitability plasticity | Reject supporting inference | None as written | A cited synaptic receptor-sensitivity change does not establish intrinsic-excitability plasticity; requires a new appropriate source. |
| 25 | Heterosynaptic LTD/LTP | Merge/re-ground | F05; F06 | General plasticity labels are not independent fly algorithms. Use the identified site, timing and teaching signal; 5/25 are explicitly overlapping. |
| 26 | Silent synapse unsilencing | Hold as underspecified | No standalone operation | A biological event needs species, locus and task evidence plus a distinct computational operation. Do not fill this gap with a generic citation. |
| 27 | Bursting and learning | Hold as underspecified | No standalone operation | A biological event needs species, locus and task evidence plus a distinct computational operation. Do not fill this gap with a generic citation. |
| 28 | REM reactivation | Archive / exclude cross-species claims | Existing replay/consolidation family | Several are overlapping replay/sleep/transfer descriptions. No exact duplicate count is inferred merely from family overlap; F12 is teaching persistence, not replay. |
| 29 | Synaptic saturation/competition | Archive | Existing stability/metaplasticity family | Generic stabilizer labels do not establish a new fly circuit or an implementable local rule. Molecular names are not separate algorithms. |
| 30 | Sequential firing chains | Exclude/retain only as analogy | F14/F15/F20 are independently sourced analogies | Old claims concern mammalian or generic sequence/working-state mechanisms. The fly examples do not validate those old citations or learned arbitrary sequencing. |
| 31 | Recurrent cortical working memory | Exclude/retain only as analogy | F14/F15/F20 are independently sourced analogies | Old claims concern mammalian or generic sequence/working-state mechanisms. The fly examples do not validate those old citations or learned arbitrary sequencing. |
| 32 | Prefrontal–basal-ganglia gating | Exclude from fly evidence | Separate 11-circuit survey | Cortical/basal-ganglia/thalamic mechanisms were part of a different survey; do not silently relabel them as fly mechanisms. |
| 33 | Thalamocortical multi-step loops | Exclude from fly evidence | Separate 11-circuit survey | Cortical/basal-ganglia/thalamic mechanisms were part of a different survey; do not silently relabel them as fly mechanisms. |
| 34 | Dendritic NMDA/plateau potentials | Exclude as stated | Optional separate cortical/dendritic review | Mammalian dendritic mechanisms cannot be used as fly evidence. F18 has its own fly source and is not a validation of these claims. |
| 35 | BCB/Strong muscle learning | Reject as unidentifiable | None | No identifiable primary source was found for the named label. It is not retained under an invented corrected meaning. |
| 36 | Amygdala avoidance conditioning | Exclude from fly evidence | Separate mammalian affect/learning review | Species and anatomical scope mismatch; no need to import these to fill an entry quota. |
| 37 | Mushroom-body compartment conditioning | Merge | F05; F06 | Existing MB associative/compartment substrate; do not count output learning and its compartment label as wholly separate discoveries. |
| 38 | Cerebellar granule-cell expansion | Exclude from fly evidence | Separate cerebellar/BG survey | Already reviewed cross-species families; not additional fly mechanisms. |
| 39 | Cerebellar supervised learning | Exclude from fly evidence | Separate cerebellar/BG survey | Already reviewed cross-species families; not additional fly mechanisms. |
| 40 | Basal-ganglia reinforcement learning | Exclude from fly evidence | Separate cerebellar/BG survey | Already reviewed cross-species families; not additional fly mechanisms. |
| 41 | Eligibility traces | Retain with distinction | F05 | Eligibility is a delay-bridging state, not synonymous with long-term tag/capture. Already represented historically. |
| 42 | Hebbian inhibitory plasticity | Replace source basis, do not equate | F03 | Fly inhibitory habituation has direct sources; a cortical inhibitory-STDP equation is not established by those sources. |
| 43 | Thalamic relay gating | Exclude from fly evidence | Separate 11-circuit survey | Cortical/basal-ganglia/thalamic mechanisms were part of a different survey; do not silently relabel them as fly mechanisms. |
| 44 | Sensory adaptation | Merge temporal-filter variants | F02 | Fast state adaptation differs from durable inhibitory habituation (F03); no doubling just for labels. |
| 45 | Lateral-inhibitory pattern detection | Merge with variant labels | F01 | Normalization, lateral inhibition and WTA overlap but are not mathematically identical. Preserve implementation variants within one competition/normalization family. |
| 46 | Reward-triggered synaptic removal | Reject over-broad formulation | Existing revision/forgetting reserve | Mixed dopamine/cortisol/receptor-removal description is not one verified fly mechanism. F11 is opponent learning, not generic erasure. |
| 47 | Hippocampal–cortical offline replay | Archive / exclude cross-species claims | Existing replay/consolidation family | Several are overlapping replay/sleep/transfer descriptions. No exact duplicate count is inferred merely from family overlap; F12 is teaching persistence, not replay. |
| 48 | BDNF-like metaplastic stabilizers | Archive | Existing stability/metaplasticity family | Generic stabilizer labels do not establish a new fly circuit or an implementable local rule. Molecular names are not separate algorithms. |
| 49 | Cortical compartment-specific plasticity | Exclude as stated | Optional separate cortical/dendritic review | Mammalian dendritic mechanisms cannot be used as fly evidence. F18 has its own fly source and is not a validation of these claims. |
| 50 | Hippocampal neurogenesis | Exclude from fly evidence | Separate cross-species survey only | Mammalian/hippocampal evidence is not fly evidence. Some computational analogies were already reviewed elsewhere. |
| 51 | Cortical predictive remapping | Exclude as stated | F19 is an independently sourced analogue | Cortical predictive remapping and fly motor-copy cancellation are not identical; the old evidence is not repaired by renaming. |
| 52 | Pavlovian taste conditioning | Exclude from fly evidence | Separate mammalian affect/learning review | Species and anatomical scope mismatch; no need to import these to fill an entry quota. |
| 53 | Spatial sequence preplay | Exclude from fly evidence | Separate cross-species survey only | Mammalian/hippocampal evidence is not fly evidence. Some computational analogies were already reviewed elsewhere. |
| 54 | Higher-order thalamic gating | Exclude from fly evidence | Separate 11-circuit survey | Cortical/basal-ganglia/thalamic mechanisms were part of a different survey; do not silently relabel them as fly mechanisms. |
| 55 | Cortical acetylcholine gain | Exclude from fly evidence | Separate 11-circuit survey | Cortical/basal-ganglia/thalamic mechanisms were part of a different survey; do not silently relabel them as fly mechanisms. |
| 56 | Serotonergic mood learning | Reject unsupported generalization | None | The statement combines species, region and broad serotonin roles without a traceable task-specific mechanism. |
| 57 | Divisive normalization | Merge with variant labels | F01 | Normalization, lateral inhibition and WTA overlap but are not mathematically identical. Preserve implementation variants within one competition/normalization family. |
| 58 | Basal-ganglia sequence learning | Exclude/retain only as analogy | F14/F15/F20 are independently sourced analogies | Old claims concern mammalian or generic sequence/working-state mechanisms. The fly examples do not validate those old citations or learned arbitrary sequencing. |
| 59 | Ocular dominance plasticity | Exclude as stated | MacKenzie2025 is a separate preprint reserve | Cortical critical-period/SOM claims do not establish fly developmental connectivity rules; no substitution without provenance. |
| 60 | Sleep synaptic downscaling | Archive / exclude cross-species claims | Existing replay/consolidation family | Several are overlapping replay/sleep/transfer descriptions. No exact duplicate count is inferred merely from family overlap; F12 is teaching persistence, not replay. |
| 61 | Triplet/quadratic STDP | Archive as a rule variant | Timing kernels within F05 only if separately validated | Higher-order spike rules are not a new circuit; their general validity and fly locus cannot be assumed. |
| 62 | Adaptive resonance (ART) | Exclude from biological fly evidence | Separate model proposal | ART is a computational framework, not demonstrated fly circuitry; local computational implementation needs its own assessment. |
| 63 | CRH-mediated fear plasticity | Exclude from fly evidence | Separate mammalian affect/learning review | Species and anatomical scope mismatch; no need to import these to fill an entry quota. |
| 64 | Cortical pattern completion | Exclude as stated | Separate cortex/attractor review | No fly evidence for the named neocortical mechanism; generic recurrence is not proof of this computation. |
| 65 | Surprise-modulated learning | Hold / merge if sourced | F06–F08 only after specifying the signal | Surprise, novelty and reward prediction error are different quantities. A neuromodulator label is insufficient. |
| 66 | Systems-level structural consolidation | Archive / exclude cross-species claims | Existing replay/consolidation family | Several are overlapping replay/sleep/transfer descriptions. No exact duplicate count is inferred merely from family overlap; F12 is teaching persistence, not replay. |
| 67 | Competitive Hebbian self-organizing maps | Exclude as stated | MacKenzie2025 is a separate preprint reserve | Cortical critical-period/SOM claims do not establish fly developmental connectivity rules; no substitution without provenance. |
| 68 | Hippocampal time cells | Exclude from fly evidence | Separate cross-species survey only | Mammalian/hippocampal evidence is not fly evidence. Some computational analogies were already reviewed elsewhere. |
| 69 | Hippocampal remapping/context coding | Exclude from fly evidence | Separate cross-species survey only | Mammalian/hippocampal evidence is not fly evidence. Some computational analogies were already reviewed elsewhere. |
| 70 | Nested oscillations gating plasticity | Archive / exclude cross-species claims | Existing replay/consolidation family | Several are overlapping replay/sleep/transfer descriptions. No exact duplicate count is inferred merely from family overlap; F12 is teaching persistence, not replay. |

## What follows for the immediate project

The user reports that E16 loses prior skills when the stream’s format changes and that replay controls are in progress. This review did not inspect those run logs or establish whether the dominant defect is stored-weight damage, address/representation change, readout gating, or a combination. The literature does not select one of these diagnoses by analogy alone.

The conditional implications are nevertheless sharper than “try another biological mechanism.” If the failure is an input-coordinate change, F13/F04 concern adaptation of the interface to a stable computation. If the stored relation survives but is not expressed, F10 concerns readout access. If prolonged teaching modifies unrelated state, F12 concerns termination. F09 concerns transferring teaching authority along learned associations, a different objective from preserving old skills after formatting changes. These are hypotheses with distinct operations and prerequisites, not a revised immediate experiment schedule.

No source examined establishes that any of these modules, alone, yields systematic arithmetic-rule generalization from raw bytes under the project’s local lifetime-learning restriction. Adding a ring, multiplication primitive or fixed sequence controller can provide computation, but it can also insert the solution during development. A successful test must distinguish discovered structure from an engineered representation.

## Linked primary-source register

Links identify the papers actually used. Citation and status verification does not mean every raw dataset, supplement or codebase has been independently reproduced. Except where explicitly linked, public model-code availability was not verified and is not asserted.

- **Olsen2010: Olsen, Bhandawat & Wilson (2010).** [Divisive normalization in olfactory population codes](https://doi.org/10.1016/j.neuron.2010.04.009). *Neuron*. Peer-reviewed primary research. 

- **Amin2020: Amin et al. (2020).** [Localized inhibition in the Drosophila mushroom body](https://doi.org/10.7554/eLife.56954). *eLife*. Peer-reviewed primary research. 

- **Nagel2011: Nagel & Wilson (2011).** [Biophysical mechanisms underlying olfactory receptor neuron dynamics](https://doi.org/10.1038/nn.2725). *Nature Neuroscience*. Peer-reviewed primary research. 

- **Das2011: Das et al. (2011).** [Plasticity of local GABAergic interneurons drives olfactory habituation](https://doi.org/10.1073/pnas.1106411108). *PNAS*. Peer-reviewed primary research. 

- **Qiao2022: Qiao et al. (2022).** [Input-timing-dependent plasticity at incoming synapses of the mushroom body facilitates olfactory learning in Drosophila](https://doi.org/10.1016/j.cub.2022.09.054). *Current Biology*. Peer-reviewed primary research. 

- **Hige2015: Hige et al. (2015).** [Heterosynaptic Plasticity Underlies Aversive Olfactory Learning in Drosophila](https://doi.org/10.1016/j.neuron.2015.11.003). *Neuron*. Peer-reviewed primary research. 

- **Handler2019: Handler et al. (2019).** [Distinct Dopamine Receptor Pathways Underlie the Temporal Sensitivity of Associative Learning](https://doi.org/10.1016/j.cell.2019.05.040). *Cell*. Peer-reviewed primary research. 

- **Aso2016: Aso & Rubin (2016).** [Dopaminergic neurons write and update memories with cell-type-specific rules](https://doi.org/10.7554/eLife.16135). *eLife*. Peer-reviewed primary research. 

- **Cervantes2017: Cervantes-Sandoval et al. (2017).** [Reciprocal synapses between mushroom body and dopamine neurons form a positive feedback loop required for learning](https://doi.org/10.7554/eLife.23789). *eLife*. Peer-reviewed primary research. 

- **Eschbach2020: Eschbach et al. (2020).** [Recurrent architecture for adaptive regulation of learning in the insect brain](https://doi.org/10.1038/s41593-020-0607-9). *Nature Neuroscience*. Peer-reviewed primary research. Larval Drosophila; anatomical, functional and computational evidence must not be silently transferred to adults.

- **Bennett2021: Bennett, Philippides & Nowotny (2021).** [Learning with reinforcement prediction errors in a model of the Drosophila mushroom body](https://doi.org/10.1038/s41467-021-22592-4). *Nature Communications*. Peer-reviewed computational primary research. Model, not a new animal experiment. Author code: https://github.com/BrainsOnBoard/paper_RPEs_in_drosophila_mb ; archived code: https://doi.org/10.5281/zenodo.4531420 .

- **Yamada2023: Yamada et al. (2023).** [Hierarchical architecture of dopaminergic circuits enables second-order conditioning in Drosophila](https://doi.org/10.7554/eLife.79042). *eLife*. Peer-reviewed primary research. 

- **Rachad2025: Rachad et al. (2025).** [Functional dissection of a neuronal brain circuit mediating higher-order associative learning](https://doi.org/10.1016/j.celrep.2025.115593). *Cell Reports*. Peer-reviewed primary research. 

- **Krashes2009: Krashes et al. (2009).** [A neural circuit mechanism integrating motivational state with memory expression in Drosophila](https://pubmed.ncbi.nlm.nih.gov/19837040/). *Cell*. Peer-reviewed primary research. 

- **Perisse2016: Perisse et al. (2016).** [Aversive Learning and Appetitive Motivation Toggle Feed-Forward Inhibition in the Drosophila Mushroom Body](https://doi.org/10.1016/j.neuron.2016.04.034). *Neuron*. Peer-reviewed primary research. 

- **Felsenberg2018: Felsenberg et al. (2018).** [Integration of Parallel Opposing Memories Underlies Memory Extinction](https://doi.org/10.1016/j.cell.2018.08.021). *Cell*. Peer-reviewed primary research. 

- **Ichinose2015: Ichinose et al. (2015).** [Reward signal in a recurrent circuit drives appetitive long-term memory formation](https://doi.org/10.7554/eLife.10719). *eLife*. Peer-reviewed primary research. 

- **Pavlowsky2018: Pavlowsky, Schor, Plaçais & Preat (2018).** [A GABAergic Feedback Shapes Dopaminergic Input on the Drosophila Mushroom Body to Promote Appetitive Long-Term Memory](https://doi.org/10.1016/j.cub.2018.04.040). *Current Biology*. Peer-reviewed primary research. 

- **Matsuno2026: Matsuno et al. (2026).** [Aberrant dopaminergic activity during consolidation causes age-related memory generalization in Drosophila](https://doi.org/10.1371/journal.pbio.3003752). *PLOS Biology*. Peer-reviewed primary research. Memory generalization here includes inappropriate broadening; it is not evidence for systematic rule induction.

- **Fisher2019: Fisher, Lu, D’Alessandro & Wilson (2019).** [Sensorimotor experience remaps visual input to a heading-direction network](https://doi.org/10.1038/s41586-019-1772-4). *Nature*. Peer-reviewed primary research. 

- **Kim2019: Kim, Hermundstad, Romani, Abbott & Jayaraman (2019).** [Generation of stable heading representations in diverse visual scenes](https://doi.org/10.1038/s41586-019-1767-1). *Nature*. Peer-reviewed primary research. Author data/code page: https://research.janelia.org/jayaraman/Kim_etal_Nature2019_Downloads/ . Code link verified; simulations were not rerun in this review.

- **Kim2017: Kim, Rouault, Druckmann & Jayaraman (2017).** [Ring attractor dynamics in the Drosophila central brain](https://doi.org/10.1126/science.aal4835). *Science*. Peer-reviewed primary research. Author data/code page: https://research.janelia.org/jayaraman/KimRouaultScience2017_Downloads/ ; code: https://github.com/hrouault/RingAttractor . Code link verified; not rerun here.

- **Groschner2018: Groschner et al. (2018).** [Dendritic Integration of Sensory Evidence in Perceptual Decision-Making](https://doi.org/10.1016/j.cell.2018.03.075). *Cell*. Peer-reviewed primary research. 

- **Lyu2022: Lyu, Abbott & Maimon (2022).** [Building an allocentric travelling direction signal via vector computation](https://doi.org/10.1038/s41586-021-04067-0). *Nature*. Peer-reviewed primary research. 

- **Pires2024: Mussells Pires et al. (2024).** [Converting an allocentric goal into an egocentric steering signal](https://doi.org/10.1038/s41586-023-07006-3). *Nature*. Peer-reviewed primary research. 

- **Westeinde2024: Westeinde et al. (2024).** [Transforming a head direction signal into a goal-oriented steering command](https://doi.org/10.1038/s41586-024-07039-2). *Nature*. Peer-reviewed primary research. 

- **Groschner2022: Groschner, Malis, Zuidinga & Borst (2022).** [A biophysical account of multiplication by a single neuron](https://doi.org/10.1038/s41586-022-04428-3). *Nature*. Peer-reviewed primary research. 

- **Kim2015: Kim, Fitzgerald & Maimon (2015).** [Cellular evidence for efference copy in Drosophila visuomotor processing](https://doi.org/10.1038/nn.4083). *Nature Neuroscience*. Peer-reviewed primary research. 

- **Seeds2014: Seeds et al. (2014).** [A suppression hierarchy among competing motor programs drives sequential grooming in Drosophila](https://doi.org/10.7554/eLife.02951). *eLife*. Peer-reviewed primary research. 

- **MacKenzie2025: MacKenzie et al. (2025).** [Connectivity biases generate a learning hierarchy in the Drosophila mushroom body](https://doi.org/10.1101/2025.10.29.684686). *bioRxiv*. Preprint; not treated as independently verified. PubMed record 41279960 explicitly marks the retrieved version as a preprint.

- **Baltruschat2021: Baltruschat et al. (2021).** [Circuit reorganization in the Drosophila mushroom body calyx accompanies memory consolidation](https://doi.org/10.1016/j.celrep.2021.108871). *Cell Reports*. Peer-reviewed primary research. Structural changes do not by themselves identify a local edge-growth algorithm.

- **OkrayRetraction2026: Okray et al. (2026).** [Retraction Note: Multisensory learning binds neurons into a cross-modal memory engram](https://doi.org/10.1038/s41586-026-10355-4). *Nature*. Retraction notice. Published online 25 March 2026. Retracts the 2023 article DOI 10.1038/s41586-023-06013-8. The notice identifies nonreproducible voltage-imaging findings; it does not say every behavioral and connectomic observation failed.

- **Sudhakaran2012: Sudhakaran et al. (2012).** [Plasticity of recurrent inhibition in the Drosophila antennal lobe](https://doi.org/10.1523/JNEUROSCI.1099-12.2012). *Journal of Neuroscience*. Peer-reviewed primary research. 

- **Plitt2025: Plitt et al. (2025).** [Octopamine instructs head direction plasticity](https://doi.org/10.64898/2025.12.11.693783). *bioRxiv*. Preprint; not treated as independently verified. Retrieved PubMed record 41446116 explicitly labels the December 2025 report as a preprint. No peer-reviewed successor was identified in the targeted search.

- **Ishida2026: Ishida, Sethi, Mohren, Haraguchi, Abbott & Maimon (2026).** [Neuronal calcium spikes enable vector inversion in the Drosophila brain](https://doi.org/10.1016/j.cell.2025.11.040). *Cell*. Peer-reviewed primary research. Online publication 29 December 2025; journal issue 5 February 2026. Earlier preprint and final article are one study, not independent replications.

## Comparative evidence and implementation-readiness table

Order favors directly constrained computations and clearly specified operations; it is **not** a ranking of novelty or a recommendation to replace E16 controls. Biological evidence, parameter economy and learning sufficiency are independent axes. “N/A” is not “yes.” Costs are qualitative engineering estimates, not measured runtimes; they exclude a full new development/search campaign. H, S and state follow the accounting contract above.

| Record | Strength of evidence for the specified computation | Fewness of shared parameters, with nonshared cost visible | Does a local rule suffice at the plastic sites? | Minimal comparison cost | One-line justification |
|---|---|---|---|---|---|
| F14 Continuous recurrent working state | High, perturbation plus dynamics | H~1–10 + N state/topology | N/A fixed dynamics; grounding separate | Low–medium | Compact persistent state is well grounded; already-tested family. |
| F01 Population normalization with spatially local inhibition | High, normalization; local APL variant | H~1–10 + fixed topology/state | N/A: fixed; learned variant unknown | Low | Known normalization, not learned semantic routing. |
| F18 Multiplicative interaction through excitation and disinhibition | High, bounded analogue interaction | H~1–10 + pairwise edges | N/A interaction; operand learning separate | Low | Adds a genuine interaction but not unrestricted exact multiplication. |
| F15 Accumulate evidence before committing | High, evidence integration | H~1–10 + N state | N/A fixed integrator; encoder separate | Low | Accumulator has an identified job, not arbitrary memory. |
| F17 Compare current state with a goal in a common coordinate | High, convergent steering studies | H~10 + state/topology | N/A comparison; goal learning separate | Low–medium | Known comparison function, not a learned goal policy. |
| F16 Coordinate transformation by population-vector combination | High, grounded vector operations | H~10–100 + vector populations | N/A fixed operation; grounding separate | Low–medium | Grounded transform/inversion, not learned arithmetic from bytes. |
| F13 Re-anchor new sensory renderings to a stable internal coordinate | High anchoring; newer relay is preprint | H~10 + S_anchor | Partial; local models, no byte guarantee | Medium | Strong interface-adaptation analogy, contingent on observable correspondence. |
| F09 A learned cue becomes a teacher for another association | High for associative chaining | H~C + teacher/student weights | Partial; higher-order task evidence | Medium | Learned teacher transfer is directly constrained, unlike vague recurrence. |
| F05 Timing-sensitive, signed associative writing | High, local associative timing | H~1–10/class + S_w/traces | Partial; specified associations only | Low | Existing local-write foundation; no hidden credit miracle. |
| F06 Compartment-specific teaching channels | High heterogeneity; cross-stage limits | H~C to C² + S_w/traces | Partial; generator/routing unresolved | Low–medium | Teaching heterogeneity costs bandwidth and plastic state. |
| F10 State-dependent readout and disinhibitory routing | High for state-dependent expression | H~C; learned keys add S_gate | N/A fixed gate; learned gate unknown | Low | Read failure can differ from storage loss; correct context mapping still needed. |
| F11 Opposing learned memories rather than overwriting one trace | High for studied extinction circuit | H~1–10 + two trace stores | Partial; full rule not established | Low–medium | Preserves an opposing trace at a real storage cost. |
| F07 Activity-dependent amplification of teaching | Medium–high, feedback motif | H~1–10 + feedback edges | Partial; amplification ≠ credit solution | Low | Activity-dependent gain, not subtraction of prediction. |
| F03 Learned inhibition of familiar background | Medium–high, habituation pathway | H~1–10 + S_I plastic | Unknown for proposed update | Low–medium | Identified inhibitory locus, but candidate equation not measured. |
| F04 Input-side timing-dependent plasticity | Medium–high, identified input plasticity | H~10 + S_in plastic/traces | Partial; plastic site ≠ sufficient learner | Medium | Actual alternative plastic site; can also induce representational drift. |
| F12 Bounded post-event teaching reverberation | Medium, controller synthesized from findings | H~10/channel + eligible state | Partial; synthesis unvalidated | Medium; K extra updates | Persistence and termination matter; no reason to replace replay controls. |
| F19 Predict and cancel self-generated sensory input | Medium–high, cellular cancellation | H~10; learned predictor adds S | N/A fixed; learned-map rule unknown | Low–medium | Needs a causal action-feedback channel; not generic predictive coding. |
| F02 Adaptive temporal filtering | High, temporal filtering | H~1–10 + N state | N/A: fixed state filter | Low | Cheap time filter; already represented historically. |
| F08 Prediction-error computation in an MB output–teaching loop | Model demonstration, not animal proof | H~1–10/channel + S/traces | Yes for specified published model tasks | Low | A real local associative model with code; no hidden-layer generalization proof. |
| F20 Priority competition that produces a sequence | Medium, innate suppression hierarchy | H~K; arbitrary graph up to K² | N/A innate sequence; learning unknown | Low | Useful fixed-control analogy, outside the persistent-learning core. |
