**MiniFly iteration 5 — a working anatomical/effective learning path, with limited retention**

**Compact delivery.** This bundle contains final result tables, the key figure,
iteration-5 model/configuration files, one final sequential-revision checkpoint,
and recorded acceptance checks. It requires the unchanged parent project to run.
Source datasets, intermediate/per-seed outputs, previous archives, caches and
unchanged third-party code are excluded. Links marked “parent asset” resolve
through the path/size/SHA-256 entries in [MANIFEST.json](MANIFEST.json).

The scientific results below are unchanged from the completed campaign. The
checkpoint is losslessly compressed for delivery; the 169,031-byte checkpoint
size reported below describes the original campaign serialization.

To verify the extracted bundle, run `python checks/verify_bundle.py`. To also
check the parent dependencies, add `--parent /path/to/minifly`. This integrity
check does not rerun the scientific campaign. The recorded implementation checks
and packaging acceptance record are under `checks/`.

The code copies under `model/` preserve their parent-layout imports for review.
After parent verification, resume the selected final state with:

```sh
python /path/to/minifly/iteration5/run.py --resume /path/to/bundle/model/final_state.npz --out /path/to/new_run
```

This resumes the completed sequential-revision condition and defaults to a
five-minute rest continuation. The parent runtime uses NumPy, SciPy and Numba;
its requirements and unchanged inherited equations are referenced in the manifest.
The supplied GPL license and attribution apply to the derivative model code.


13 September 2026. Decision: retain this executable parent demonstrator. Do not compress it further or call it a validated persistent fly brain.

**The integration works within its declared scope.** Engineered projection-neuron inputs now travel through real FlyWire PN→KC connections, modify fixed KC-associated state through anatomically routed teaching, and change later MBON readouts. No new memory bank is allocated for a new cue. The model learns the eight tested pairs initially, retains a measurable 24-hour change for seven, and reverses the first association in a sequential test. It fails the declared interference-retention criterion. This is a functioning hybrid learning demonstrator, with no calibrated sensory-to-behavior policy or biological validation of its persistence.

Read the [run instructions — parent asset](MANIFEST.json), [fixed protocol](model/PROTOCOL.md) and [result figure](figures/integration_results.png). The attached handoff was treated as a proposal. I adopted one bounded integration and refined it using the actual data, without another teaching family or post-result parameter search.

**What real anatomy contributes.** The interface uses FlyWire materialization 630 only: 302 selected ALPN input neurons, 5,177 fixed KCs, and the PPL101/PPL106 and MBON11/MBON14 interfaces. There are 27,572 selected PN→KC edges. PN stimulation is grouped into 88 channels: 81 annotated types and seven individually retained untyped inputs. The prior 9,619-cell scaffold lacks 277 of these selected PNs, so its neuron count could not be treated as a complete olfactory-learning interface. This hybrid executes the extracted paths directly; it does not turn the entire scaffold into a spiking learner. [Anatomy inventory — parent asset](MANIFEST.json), [exact interface identities — parent asset](MANIFEST.json).

KC→MBON synapse counts supply readout weights. Direct DAN→KC counts route teaching separately for gamma1 and alpha3; missing direct routes permit no write. Under this conservative gate, 2,014 KCs can write gamma1 state and 1,226 can write alpha3 state. These are neuron-level support rules, not proof of local dopamine responsibility or a complete account of volume transmission. All cross-hemisphere feedback paths are retained. For example, the four gamma1-MBON→alpha3-DAN counts are only 4, 1, 6 and 14 synapses; their proportions route the inherited effective feedback gain, but their counts do not identify that physiological gain. [Feedback counts](metrics/feedback_anatomy.csv), [frozen calibration](model/interface_calibration.json).

The data also exposed an annotation trap: 5,173 of the archived KCs are labelled dopamine, three acetylcholine and one GABA. The published FlyWire network-statistics analysis explicitly corrected KCs to cholinergic. This demonstrator applies that class-level interpretation and uses unsigned KC connection counts, including the one negatively signed KC output row. It preserves the original labels and signed graph. This is an explicit correction supported by primary literature, not new transmitter evidence inferred from successful learning. [Network statistics of the whole-brain connectome of Drosophila, Nature 2024](https://www.nature.com/articles/s41586-024-07968-y).

**What remains an engineering interface.** PN patterns are constructed inputs, not measured responses to particular odors. Input weights are normalized per KC, followed by deterministic 5% sparsification per hemisphere. This is not a fitted APL circuit. Sixteen unlabeled development patterns determine four fixed aggregate readout gains, approximately 24–26. No rewards or evaluation outcomes enter this calibration. The 40 evaluation patterns and 1,024 capacity patterns are new relative to those development patterns.

The learning gains and fast/slow kernels remain frozen from iteration 4. Baseline-relative teaching changes are separate from nonnegative anatomical connection counts. Signed effective changes can modify the rate-model readout; they are not negative excitatory physical synapses. Bout timing remains unchanged, including its known sensitivity to splitting an active bout. The uncertain slow lifetime remains a fitted boundary value, not measured retention over approximately 116 days. The gamma1/alpha-prime2 higher-order circuit remains separate and unresolved.

**Acquisition and persistence.** The declared criterion is at least a 1-Hz learned decrease in the A-minus-B alpha3 contrast. “Learned” means the difference between a trained-state readout and a clone with plastic traces reset but adaptation preserved. Every probe runs through the implemented PN→KC→MBON pathway. [All pair results](metrics/acquisition_results.csv).

| Frozen configuration | Five-minute criterion | 24-hour criterion | Median 24-hour memory contrast |
|---|---:|---:|---:|
| Hybrid, two traces | 8/8 | 7/8 | −3.325 Hz |
| Hybrid, single exponential | 8/8 | 0/8 | Approximately zero |
| Gamma feedback blocked throughout | 8/8 | 8/8 | −3.256 Hz |
| Plasticity disabled | 0/8 | 0/8 | 0 |
| Teaching payload blocked | 0/8 | 0/8 | 0 |

These are memory-change criteria, not choice accuracy. At 24 hours only five of the eight intact pairs have a negative raw A-minus-B readout, because innate input differences remain. The reset clone is an evaluator's counterfactual, not an available sensory measurement or an autonomous decision policy. Acquisition alone also does not establish that feedback is necessary: the feedback-blocked model passes it. [Raw versus memory-specific readouts](metrics/raw_readout_vs_memory_contrast.csv).

The supplied [minimal failing case — parent asset](MANIFEST.json) is acquisition pair 5. After 24 hours its memory contrast is only **−0.404 Hz**. Both reset readouts are at the alpha3 ceiling, 19.96 Hz; the learned target is still at the ceiling on the left and 19.153 Hz on the right. The foil remains at the ceiling. Thus readout saturation limits expression of the stored change in this case. The input masses are about 1.67 and 1.74 times the development mean. No alternate gain was selected to rescue it. [Per-side failure readouts](metrics/minimal_failure_readouts.csv).

**Interference and revision.** Four pairs are acquired sequentially, followed by eight other learned pairs and then reversal of punishment within the first pair. Each pair receives six A/B training bouts. All four original contrasts retain their learned sign after interference, but **zero of four retains half its acquisition magnitude**, the declared criterion. Matched rest-only continuations already fall below half, so passive decay contributes substantially; intervening learning weakens all four contrasts further. For the first pair, acquisition is **−9.970 Hz**, matched rest **−2.849 Hz**, and interference **−0.704 Hz**. [Sequential results](metrics/sequential_pair_summary.csv), [all 1,200 probe observations — parent asset](MANIFEST.json).

Revising the first pair changes its memory contrast to **+7.365 Hz**, meeting the reversal criterion. A post-panel diagnostic starts every condition from the identical pre-revision checkpoint. Intact revision ends at +7.365 Hz; blocking gamma feedback only during revision ends at **−5.670 Hz**; blocking the teaching payload ends at **−1.412 Hz**; matched rest ends at **−0.276 Hz**. This isolates feedback's effect on the implemented revision path from differences in acquisition history. It is one descriptive intervention case, not a general biological necessity result or a new held-out validation. [Matched-state intervention](metrics/matched_revision_diagnostic.csv).

Overlapping cues share the fixed KC state and therefore generalize and interfere. The eight designed overlaps share exactly four PN types with their parent inputs, with actual KC overlaps recorded separately. At acquisition, all eight overlap probes have a memory-change sign matching their parent; this is descriptive generalization under engineered inputs, not evidence that the encoding matches fly olfaction. Novel probes can also change: for example, an untrained probe has −2.229 Hz memory contribution. The representation does not promise independent storage for every cue. [Overlap readouts](metrics/overlap_readout_summary.csv).

**Fixed capacity and verification.** Persistent mutable state consists of 20,708 float64 trace/adaptation values—four per KC—plus a float64 clock and two uint64 counters: **165,688 bytes total**, excluding static anatomy, interpreter overhead and external logs. Checkpoints are exactly **169,031 bytes** across 1,024 unique new input patterns and 2,048 events. Midpoint save/resume gives exactly identical continuation, with no added cue IDs or state banks. This tests fixed allocation, not retention of 1,024 associations. [Capacity checkpoints](metrics/capacity_checkpoints.csv).

Twelve implementation checks pass. The uniform-population limit recovers iteration-4 rates within 7.11 × 10⁻¹⁵ Hz; real-interface checkpoint continuation is exact. Cutting PN→KC input or blocking teaching prevents writes. Resetting plastic traces changes a later readout by 7.767 Hz in the causal witness. The main frozen panel takes about two seconds on this Mac after compilation and peaks near 212 MB RSS; this excludes anatomy preparation, compilation, diagnostics, plotting and packaging. The complete numerical reproduction command was also executed successfully in about nine seconds. [Verification](checks/implementation_verification.json), [run accounting](checks/campaign_run.json).

Across the main panel, there are 480 acquisition training presentations, 780 sequential training presentations, and 1,024 capacity presentations; capacity verification separately replays 512 presentations. Cloned acquisition and sequential readout checks add 320 and 2,400 probe presentations respectively, outside the learned parent streams. Development/evaluation/capacity contain 16/40/1,024 distinct patterns, 1,080 total. Internal routines and the two fixture corrections are recorded in [correction log](checks/correction_log.json). They change no learned gain, criterion or mechanism.

**Biological scores remain separate.** The standard-conditioning and 14 spacing/extinction scores from iteration 4 are retained verbatim with source hashes. Analogous schedules run through this hybrid as engineered PN proxies, but receive no biological RMSE: measured PN responses for ACV/EtA and OCT/BEN and a physiological calibration of this interface are unavailable. Consequently the previous physiological failures, uncertain slow lifetime, unestablished BANC spatial replication and failed higher-order recovery family all remain. [Biological scope and frozen scores — parent asset](MANIFEST.json).

The useful distillation is now concrete: a fixed sensory code can feed address-specific plastic state, and prior memory can change later teaching enough to affect revision. The unsolved parts are reliable preservation through time and intervening learning, expression of memory through a calibrated readout, and physiological identification of the interface. Further compression should wait for those parent behaviors. No extra data download or Zenodo request was needed. Measured odor→PN response patterns and longitudinal/interference physiology would help the next decision more than another generic graph download.
