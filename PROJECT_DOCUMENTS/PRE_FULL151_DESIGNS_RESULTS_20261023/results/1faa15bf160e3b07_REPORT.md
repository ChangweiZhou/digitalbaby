**MiniFly refinement from Huang et al., Nature (2024)**  
13 September 2026. Executable learning reference and audit of the existing anatomical cores.

**The paper materially improves our next experiment: compress the state and feedback needed for learning, then test whether a smaller anatomical circuit preserves them.** It also exposes a specific defect in our current reductions. Every circuit selected from taste/grooming activity, including the 8,783-neuron version, omits both long-term dopamine cell types used by this paper. The 9,619-neuron mushroom-body-preserved scaffold retains them. Good sensorimotor performance alone was never sufficient evidence of a learning core.

I added a runnable Python implementation of the authors' effective learning model, checked its equations, ran the released parameter ensembles and counterfactuals, and audited the corresponding FlyWire interfaces. This is a separate learning reference; plasticity has **not** yet been embedded in MiniFly's spiking network. Exact agreement with the released model prediction medians remains unresolved.

The deposited paper is *Dopamine-mediated interactions between short- and long-term memory dynamics*, Huang et al., Nature 634, 1141-1149, DOI [10.1038/s41586-024-07819-w](https://www.nature.com/articles/s41586-024-07819-w). It is the same study behind the physiology workbooks analyzed in the preceding follow-up, so it must not be counted as independent replication. The new contribution here is an executable model and an anatomical audit, informed by the full paper and supplement. The original local PDF's identity and the pinned source commit are recorded in [provenance](sources/provenance.json).

**What should be distilled.** The paper links short-lived plasticity in the gamma1 module to longer-lived alpha-module plasticity through MBON-to-DAN feedback. Learned suppression of an inhibitory MBON changes dopamine responses to later odor presentations. Consequently, later unpaired cues can either strengthen or weaken the long-term trace depending on when they occur. These are experimentally constrained, history-dependent predictions that a scalar gain or static mask cannot represent by itself. See the paper's Fig. 5 and Extended Data Fig. 10, and [Supplementary Information](sources/Supplementary_Information.pdf), sections 3-5.

This is not passive copying of a short-term trace into a long-term store while the animal rests. In the supplied equations, rest decays the plastic states; cue presentations engage the feedback that changes subsequent learning. The useful compression target is therefore a combination of odor-specific plastic weights, sensory adaptation, compartment identity, feedback signs/gains, and elapsed time. One must preserve their effects on future interventions, not merely the current firing-rate vector.

For two abstract odor channels, the three-module reference has six evolving effective plastic weights and two adaptation states, plus its protocol clock; the two-module reference needs four effective plastic weights and two adaptation states. Fast rate responses can be solved from those states because the supplied fast rate matrix is acyclic. This is an extremely small computation. It is not a literal six- or eight-neuron fly brain: sensory populations, inhibition and many cellular processes have already been absorbed into fitted effective parameters. The supplied three- and two-module fits have 23 and 16 free parameters respectively.

**The existing cores lose a required interface.** I matched homologous type labels to the historical FlyWire v630 annotations and then inspected neuron identities and edges only inside that specimen. The paper's hemibrain IDs were never inserted into FlyWire. The correspondence is gamma1 DAN = PPL101, alpha2 DAN = PPL105, alpha3 DAN = PPL106; gamma1 MBON = MBON11, alpha2 MBON = MBON18, alpha3 MBON = MBON14.

| Existing circuit | Neurons | Gamma1 DAN | Alpha2 DAN | Alpha3 DAN | Alpha3 MBON |
|---|---:|---:|---:|---:|---:|
| Full FlyWire reference | 127,400 | 2 | 2 | 2 | 4 |
| Activity core | 1,722 | 2 | 0 | 0 | 0 |
| Five-partner guard | 4,083 | 2 | 0 | 0 | 0 |
| Ten-partner guard | 5,832 | 2 | 0 | 0 | 2 |
| Twenty-partner guard | 8,783 | 2 | 0 | 0 | 3 |
| MB-preserved guard | 9,619 | 2 | 2 | 2 | 4 |

Counts combine hemispheres. [All six roles and side-specific counts](results/paper_cell_type_retention.csv), [neuron identities](results/paper_interface_neurons.csv), [coverage figure](figures/interface_coverage.png).

The direct gamma1 MBON-to-alpha3 DAN path has four synapses on the left and fourteen on the right in the signed FlyWire reference, recovered from its published weight conversion. Every activity/ordinary guard reduction loses both paths; the MB-preserved scaffold keeps them. The paper's hemibrain model lists 25 synapses for its corresponding pair and applies a five-synapse inclusion threshold when defining its effective network. Copying that threshold into FlyWire would delete the left-side path. We should protect a tested functional interface and evaluate its uncertainty, rather than transfer a numerical cutoff between specimens. Synapse counts also do not supply the paper's fitted physiological gains. [Feedback audit](results/paper_feedback_retention.csv); Supplementary Information, Table 1.

The new [interface checker](check_interface.py) requires the six homologous types and relevant KC families in each hemisphere. The 9,619-neuron scaffold passes; the 4,083-neuron guard fails. Passing establishes necessary annotated coverage only. It does not establish functional synapses, correct odor encoding, learning, or completeness of the surrounding circuit.

**What the executable reference reproduces, and what it does not.** The Python port reads the authors' original MATLAB parameter files, including all 10,000 stored parameter draws. It follows the supplied bout-level adaptation, clipped MBON response, dopamine-relative teaching, plasticity and decay equations. It does not fit new parameters to our desired outcome. The source is pinned to commit `5d7c08a9a88f923169a0c3008aca68af421e9a7f` of the [authors' model repository](https://github.com/schnitzer-lab/Luo_Huang_2024_MB_model).

The optimized engine passes **18 equation checks**: sixteen comparisons against an independently written literal translation of the supplied MATLAB equations across two/three modules, two odor valences and four protocols, plus no-plasticity and rest-only checks. Maximum rate discrepancy in the sixteen comparisons is about 7.1e-15 Hz. MATLAB/Octave was unavailable, so this is not a claim of native MATLAB execution. [Equation checks](results/engine_checks.json).

A separate comparison with fourteen released three-module prediction medians does **not** pass exact agreement. Thirteen differ beyond 1e-6. For example, the 360-second spacing protocol gives a 24-hour alpha3 response bias of -13.6885 Hz versus the released -14.0871 Hz. Saturated 5-minute biases are -31.16 Hz versus -30.59 Hz, a 0.57-Hz difference. The largest ratio discrepancy is about 0.0824. The supplied code's response cap and the released saturation value suggest a version/constants mismatch, but the exact cause is unresolved; small rounding differences in the supplement do not fully explain it. I retained the discrepancy rather than tuning it away. [Every comparison](results/published_replay.csv), [verification status](results/replay_verification.json).

The two-module replay is also included, but comparing it with the three-module released medians is a model comparison, not an exact two-module software check. The stored draws describe the authors' parameter ensemble; 10,000 draws are not 10,000 new animals. These replays are not independent biological validation.

The batched prediction runs for both models took about 3.1 seconds after compilation and peaked at approximately 273 MB resident memory on this Mac. A single mean-parameter spacing run took about 0.067 seconds with a warm compilation cache. Runtime includes effective bouts and long rest intervals, not every millisecond of a 24-hour whole-brain simulation. Memory is therefore no obstacle to using this model as a reference while anatomical integration remains unfinished.

**The counterfactuals distinguish what matters in this fitted model.** With six training bouts, a 360-second inter-stimulus interval and attractive odors, the 24-hour alpha3 CS+ minus CS- response bias is:

| Model intervention | Median bias, Hz |
|---|---:|
| Intact reference | -13.6885 |
| Remove gamma1 MBON-to-DAN feedback | -3.7875 |
| Remove odor adaptation | -31.1600 |
| Keep the early alpha decay constant at all times | -0.000151 |
| Remove alpha2 from the three-module model | -13.6885 |
| Disable plasticity | 0.0000 |

These are disclosed model counterfactuals using the stored parameter ensemble, not new fly experiments. For the feedback intervention, initial odor responses were matched by recalculating the effective input weights after removing feedback. That isolates the learning consequence within the model; it is not the same intervention as physically silencing all gamma1 MBON outputs. [Counterfactual table](results/model_ablations.csv), [figure](figures/memory_counterfactuals.png).

In a separate three-bout protocol, the intact attractive-odor model's 3-hour/5-minute alpha3 bias ratio rises from 0.299 without re-exposure to 0.431 with unpaired cues at ten minutes, but falls to zero with unpaired cues at two hours. Removing gamma1-to-DAN feedback changes the early-cue ratio to 0.074, compared with its control of 0.245. This sign of the early-cue effect is a useful future learning benchmark. Ratios in a no-plasticity model are not evidence of memory: sensory adaptation can itself leave a nonzero response bias.

Removing alpha2 without refitting changes the common gamma1/alpha3 responses by at most **1.13e-7 Hz** across the six checked valence/protocol combinations. Its fitted coupling into alpha3 is effectively zero. This supports testing a smaller state model for these protocols; it does not show that alpha2 is biologically dispensable or can be removed from all learning tasks. [Removal comparisons](results/alpha2_removal.csv).

**Three assumptions prevent direct transplantation into MiniFly.** First, dopamine is represented as a change relative to baseline. Suppression is meaningful even when the change is negative. Our fixed-weight LIF reference has zero spontaneous baseline, so a negative delta-rate cannot simply become an absolute firing rate. Dopamine also needs a teaching action distinct from ordinary synaptic current.

Second, the effective KC inputs can become negative because parallel inhibition and other influences have been absorbed into them. Actual cholinergic KC synapses cannot be assigned arbitrary negative weights on this basis. A physical implementation must allocate those effects to excitation, inhibition and modulatory pathways explicitly.

Third, the late decay switch is an assumption. Supplementary Information section 3 says a single exponential did not capture the memory time course; the authors use different alpha decay constants before and after three hours and discuss a possible biexponential process. Our ablation confirms that the imposed switch accounts for much of the 24-hour persistence in this model. It does not discover a biochemical consolidation mechanism. The reduced teaching rule also assumes the experiment's fixed three-second CS-US delay. Its validity under arbitrary temporal tasks has not been established.

The exported end-of-protocol weights and adaptation values are audit records, not a resumable online learning checkpoint: the source decay clock uses the scheduled last training bout. An online implementation needs an explicit causal clock/reset policy and validation of that policy.

**Refined assignments.** The implemented [learning contract](learning_contract.json) now separates the original sensorimotor scores from learning scores and records unknown parameters instead of filling them with invented anatomy-derived values. The next bounded experiment should connect same-specimen odor inputs to the MB-preserved scaffold, calibrate baseline and signed teaching effects, and test conditioning, spacing and early/late unpaired cues. A smaller candidate should then be compared with this learning-capable reference under feedback interventions and unseen cue schedules. Neuron count becomes the quantity minimized subject to those tests, rather than the primary success metric.

In parallel with that future integration, the effective reference can compare the imposed decay switch against a continuous or biexponential alternative using separate fitting and validation conditions. That would distinguish a compact descriptive fit from a more plausible online mechanism. The present source-output discrepancy should be resolved or carried explicitly into the uncertainty of those comparisons.

The old negative results remain: MiniFly v0.1 missed its paired-input temporal-fidelity target, and the BANC synapse audit did not establish independent local teaching-address replication. This paper does not overturn either result. Whole-neuron/type physiology does not validate a particular synapse-scale spatial arrangement. BANC, hemibrain and FlyWire must continue to be analyzed separately, with cross-specimen comparisons restricted to justified homologous types.

**Delivered and reproducible.** [README](README.md) gives run and regeneration commands. The branch includes the original model source/parameters, Python port, per-condition outputs, counterfactuals, anatomical tables, figures, provenance and license notices. Supplemental files downloaded successfully into this project through Springer; Zenodo was not contacted. No additional manual download or user assistance is required for this refinement. The original v0.1 ZIP remains frozen; the revised package adds this branch and leaves the original anatomical models and LIF engine unchanged.
