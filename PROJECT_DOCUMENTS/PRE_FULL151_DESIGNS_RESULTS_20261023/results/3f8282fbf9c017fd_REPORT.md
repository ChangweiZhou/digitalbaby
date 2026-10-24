# What can be distilled from the downloaded fly teaching circuitry?

**There is reproducible, compressible anatomical organization beyond what V50–V54 used. The strongest result is an output-local teaching address, together with reproducible feedback routing. The data do not yet identify an event-selective teaching rule or a persistent learner.**

I analyzed the original BANC v888 and hemibrain v1.2.1 files, extracted relevant hemibrain synapse coordinates, ran matched null ensembles, and tested a compact anatomical predictor on held-out KCs. I did not run a V55 learning experiment. The attached previous report and pasted proposals were treated as background to evaluate, not additional instructions.

![Evidence summary](figures/evidence_summary.png)

## 1. The strongest new result: the address is local to an output site

The downloaded `hemibrain_121_synapses.parquet` contains 41,466,703 rows. A targeted local extraction retained **996,418 rows**, including DAN inputs, KC→MBON, DAN→KC, DAN→MBON and APL↔KC contacts. No additional large download or authenticated query was necessary.

For each recorded KC→MBON site, I found the nearest recorded DAN→**that same KC** presynaptic site. This produces a *nearest-terminal anatomical assignment*. It does not establish a dopamine release radius, receptor response, or the physiological teacher of that synapse.

The primary spatial analysis used right-instance DANs and MBONs and all receiving-territory labels. A stricter sensitivity retained only KC **axon**→MBON **dendrite**, and DAN **axon**→KC **axon**, connections. The stricter result is emphasized below because its anatomical interpretation is cleaner.

The strict subset contains **156,651 covered KC→MBON rows from 1,915 KCs**, with **55,838 DAN→KC terminal–partner rows**. Coverage is 99.90% of the selected readout rows. Polyadic partner rows are appropriate for connection counts; a separate one-partner-per-KC-connector sensitivity checks repeated presynaptic sites.

The null moves DAN terminal identities among fixed terminal positions **within the same recipient KC and exported ROI**. Every DAN→KC neuron-pair count can remain exactly unchanged. Site geometry, KC subtype, KC→MBON connections, and ROI-specific DAN→KC counts remain unchanged. Only the association between DAN identity and position is destroyed.

| Spatial statistic | Real geometry | Matched-null mean | Null 95% interval |
|---|---:|---:|---:|
| DAN-type/MBON-type mutual information, strict territories | **2.369 bits** | 1.217 bits | 1.209–1.224 |
| Same statistic, all territories | **2.511 bits** | 1.138 bits | 1.131–1.146 |

No shuffle reached the observed statistic in 999 draws: Monte Carlo p = 0.001, also q = 0.001 within the reported spatial sensitivity family. The effect survives confidence ≥0.9, the nearer half of assignments, and one retained partner per KC connector. These are conditional graph randomization results, not biological population p-values from independent flies.

The location shuffle breaks the spatial coherence of individual DAN arbors. It tests whether the pair-count graph alone retains the observed local association; it does not show that real geometry outperforms every anatomically plausible alternative arbor arrangement. Neither the nearest-site surrogate nor its compact fit has been spatially replicated in BANC, whose synapse-coordinate table was not among the downloaded files.

**What this establishes:** two anatomical arrangements with exactly the same DAN→KC neuron-level connectivity can have substantially different output-local allocation. A DAN contact onto one part of a KC cannot specify an identical address for every output of that KC.

The KC-wide count-mixture proxy differs from the local nearest-site distribution by a site-weighted total-variation distance of **0.665** in the strict subset (0.726 with all territories). This measures disagreement between two anatomical surrogates; it is not a measured percentage of erroneous physiological learning.

![Local versus KC-wide allocation](figures/local_vs_neuron_allocation.png)

## 2. A concrete compact object, tested on held-out KCs

After finding the spatial effect, I tested whether it could be compressed. This was a follow-up analysis, not a prospectively registered hypothesis.

Let `f[k,d]` be the observed DAN-type count mixture onto KC `k`. Fit a nonnegative **20 DAN types × 33 MBON types** ratio table from spatial assignments on training KCs:

```text
R[d,o] = P(nearest DAN type = d | MBON type = o) / P(nearest DAN type = d)
q(d | k,o) = normalize_d(f[k,d] × R[d,o])
```

The fitting rule uses a fixed pseudocount of one per type pair. No rank, smoothing, or task-performance parameter was tuned. Five folds hold out **entire KCs**, balanced by their supplied subtype. Test KCs contribute their DAN→KC count mixtures, but none of their local assignment labels, to prediction.

| Predictor of held-out nearest-terminal type | Log loss per anatomical site; lower is better |
|---|---:|
| KC-wide DAN count mixture | 2.617 bits |
| Compact table fitted to within-KC/ROI shuffled training locations | 2.614 bits, null mean |
| **Compact table fitted to real training locations** | **1.062 bits** |

Improvement is **1.555 bits per site** over the KC-wide mixture; 98.75% of held-out KCs improve in mean loss. None of 199 shuffled-training fits reaches the real fit: p = 0.005. This comparison is within one specimen and predicts anatomy, not behavior.

The resulting 660-entry table is saved in [compact_address_ratio.csv](results/compact_address_ratio.csv), with fold and per-KC scores. It is a reproducible structural distillation rather than a chosen neuron budget. It supplements the KC-specific count mixture; it is not a standalone replacement for the entire circuit.

**This is an address model, not a new learning rule.** Its coefficients are static. Historical failures of static masks still matter: improved anatomical addressing alone does not determine when to write, what sign to use, or how to protect old memories.

## 3. Feedback routing really does reproduce across specimens

Each graph was analyzed independently. Source and destination neuron identities were never joined across specimens. I extracted:

```text
F1 = P[MBON,DAN]
F2 = P[MBON,OTHER] × P[OTHER,DAN]
P[i,j] = recorded_count[i,j] / all recorded outgoing counts from i
```

`OTHER` excludes the annotated PN/KC/MBON/DAN/APL/DPM groups. Thus this relay definition is narrower than the older distillation's all-non-core definition, which allowed PNs. These are unsigned finite-walk operators, not measured influence or synaptic efficacy.

The cross-matching file is a many-to-many relation. It includes repeated and conflicting alternatives. Conservatively retaining all links collapses PAM01/PAM08, PAM02/PAM05/PAM06, and PAM09/PAM10 into ambiguity groups. Comma-separated MBON alternatives are likewise grouped. Some hemibrain `cell_type` values also disagree with the legacy instance prefix. I used the supplied type labels, explicitly grouped ambiguities, and tested exact shared base-type labels separately; I did not silently convert legacy instance names into verified identities.

The conservative all-side comparison has **34 MBON groups × 19 DAN groups**. For each matrix, I subtract the row/column-marginal independence expectation and scale by the square root of row strength × column strength. The statistic is cosine similarity between these residual matrices. Correspondence nulls permute MBON groups within strength terciles and DAN groups within PAM/PPL1 family × strength terciles.

| Operator | BANC–hemibrain residual similarity | Matched-null 95% interval | Right-side-only similarity |
|---|---:|---:|---:|
| Direct counts | **0.780** | −0.063 to 0.141 | 0.670 |
| Direct outgoing-normalized routing | **0.712** | −0.069 to 0.122 | 0.630 |
| Two-hop outgoing-normalized routing | **0.851** | −0.056 to 0.164 | 0.801 |

Each exceeds all 1,999 correspondence permutations: p = 0.0005. Exact shared-name sensitivities agree: 0.698 for direct outgoing routing and 0.834 for indirect routing, with the same Monte Carlo resolution. All 24 reported correspondence comparisons survive their within-family BH adjustment.

Concrete examples, using fractions of each MBON type's counts into the **matched selected DAN groups**, are MBON07→PAM11 (91.9% BANC, 96.2% hemibrain) and MBON16→PPL104 (100%, 99.1%). These are destination preferences within the displayed DAN subset, not fractions of all MBON output or evidence of feedback sign.

This is strong **annotation-assisted anatomical consistency**. Cell typing and cross-connectome matching can themselves incorporate morphology and connectivity, so the correspondence test is not an entirely independent discovery of evolutionary conservation. It does establish that these supplied graphs have considerably more agreement than arbitrary matched correspondence.

## 4. An important negative result: indirect compression is not a universal hierarchy

| Outgoing-normalized operator | BANC: SVD directions for 95% squared magnitude | Hemibrain: same statistic |
|---|---:|---:|
| Direct MBON→DAN | 26 | **6** |
| MBON→OTHER→DAN | **9** | 7 |

The ordering from the previous BANC reduction does **not** reproduce here in hemibrain. These numbers concern this PAM/PPL1, two-hop, OTHER-relay analysis; they are not a like-for-like rerun of the older broader four-hop study.

Moreover, indirect rank-one errors are not exceptional under a relay identity shuffle restricted to relay type and side: BANC observed 0.583 versus null mean 0.586; hemibrain 0.426 versus 0.404. Neither has a two-sided tail below 0.05. Generic mixing and preserved type structure explain much of the apparent compression. An exact eight-channel teaching principle is not established.

Direct count structure survives fixed input/output count marginals, but interpretation depends on the null. BANC's rank-one error is 0.797 versus 0.756 under the strength null; hemibrain's is 0.671 versus 0.419. Randomizing binary degree-preserving edges and reassigning the original weight multiset instead produces *higher* mean errors, 0.855 and 0.785. High rank alone therefore cannot distinguish meaningful computation from random complexity.

Conditioning count randomization on source type × target type × side leaves only small additional rank-one-error differences: about **0.00104** BANC and **0.00024** hemibrain. Both reach the 199-draw p = 0.005 resolution, but these small effects should not be advertised as a new mechanism.

The nulls are deliberately named separately. The binary-edge null preserves exact binary in/out degrees and the global positive-weight multiset, not each neuron's count strength. Fixed-marginal count nulls preserve strengths, not binary degrees. The type-block null preserves per-block row/column strengths. The degree result uses a symmetric swap chain with burn-in and saved support-overlap/acceptance diagnostics; those diagnostics are not proof of uniform mixing, so that ensemble is secondary evidence.

## 5. Independent drive remains indispensable to an honest reduction

The previous report's source-territory distinction replicates in the PAM/PPL1-only extraction:

| Recorded input fraction | BANC | Hemibrain |
|---|---:|---:|
| OTHER cells into DAN dendrites | 76.69% | 78.86% |
| KC into DAN axons | 53.65% | 78.61% |

Outside-core cells with MBON inputs supply about **88.8% / 88.6%** of OTHER→DAN counts. Yet the outgoing-to-DAN-weighted mean MBON fraction of those outside cells' total inputs is only **5.77% / 5.77%** when all outside DAN-input cells are included. Their non-MBON drive cannot be removed by reducing them to passive feedback wires. The near equality is descriptive, not a physiological equivalence or population estimate.

All individual relay fractions, source-class mixtures, and upstream partners of the 30 strongest individual relays are exported. AVLP477 remains a useful concrete example: MBON inputs comprise 7/3,169 recorded counts in BANC and 7/3,364 in hemibrain. Its physiological meaning still requires independent evidence.

## 6. What this changes for the persistent-learning project

The best supported structural specification is:

1. **Independent event/state input and learned-output feedback both reach the teacher.** Reducing relay states must preserve both inputs.
2. **Feedback is destination-specific and reproducible at type level.** One arbitrary broadcast does not reproduce its anatomical routing.
3. **Allocation is local to KC→output sites.** A compact output-address factor, conditioned on the KC's own DAN contacts, captures substantial spatial structure.

These constrain a model's information flow. They do not identify the missing integration or plasticity equations. Local compartment organization is already established neuroscience; the new contribution here is its quantitative audit, controls, and explicit held-out-KC compression in these downloaded files. [Li et al., adult mushroom-body connectome](https://elifesciences.org/articles/62576)

**I would not start another mechanism tournament from these results.** In particular, multiplying the V54 residual by `F1 + F2` would conflate biological routing with an arbitrary output coordinate system. The inspected source already uses a vector residual and a randomly signed normalized outcome codebook; only the tested gain is scalar. Its primary invariant-cue fixture also uses an explicitly nondeployable representation.

The proposed “real wiring distinguishes target events from framing/key events” test remains **unidentified** from anatomy alone. The files do not provide the mapping from byte histories to outside-core neuron activities, measured event/state responses, feedback signs, or a local plasticity law. Different choices of that missing interface can make the same wiring selective or nonselective. A favorable task score after choosing such a mapping would test that added interface and rule as well as the anatomy.

The most useful additional help would be **same-cell-type recordings or perturbation data linking defined sensory/outcome events, MBON feedback, and DAN responses**, with timing and relevant receptor/plasticity evidence. That would let us distinguish additive integration, gating, prediction error, or other computations. More anatomical counts alone will not select among those alternatives. A named-compartment atlas would improve anatomical labels, but is less urgent now that a coordinate-based address test is possible.

## 7. Data quality, verification, and reproducibility

- Reproduced whole-export totals: BANC simple 11,752,828 rows / 35,733,096 counts; BANC split 14,332,296 / 34,310,230; hemibrain split 4,679,482 / 15,047,736. Primary selected DAN populations are 293 BANC and 316 hemibrain; the previous report's broader class included PPL2 and one conflicting KC annotation.
- Within the selected BANC DAN inputs, all 265 simple/split pair discrepancies are self-connections, accounting for 1,115 counts. Selected non-self input pairs agree exactly.
- Synapse versus split-edge PAM/PPL1 input reconciliation has 292 differing pairs, 460 absolute count discrepancy against 309,770 edge counts (0.1485%). The two sources are retained separately; no coordinates are invented to reconcile them.
- Synapse rows have `prepost=0`, presynaptic coordinates, and repeated connector IDs for polyadic partners. The selected extraction has no duplicate connector–pre–post rows. Broad `MB` labels dominate; occasional finer labels such as g2/g3 do not constitute a complete, consistently assigned compartment atlas. Native coordinate units are retained because the file metadata does not specify their conversion.
- IDs remain strings or exact integer-derived strings. Unmapped presynaptic IDs remain in denominators. A standard PyArrow reader was available, so no custom Arrow decoder was needed.
- Independent pair-table reconstruction exactly reproduces both direct operators. Independently oriented sparse propagation agrees with two-hop operators within 3.5×10⁻¹⁸. Sampled nearest-site distances agree with brute-force searches. Actual neuron-ID shuffles preserve per-KC/ROI identity multisets.
- A validation assertion caught the installed SciPy random-table sampler returning a negative entry for a one-column table. The code handles one-row/one-column marginals deterministically and removes empty margins before sampling. It checks nonnegativity and every sampled margin. An independent 2×2 enumerable-distribution check passes. No failed-run null output was used.
- Input SHA-256 hashes, all null draws, matrices, extracted spatial assignments, fold scores and validation outputs are saved. Original downloads are read-only. The package omits the large source files and regenerable graph caches.

See [README.md](README.md) for commands and an artifact guide, [PROTOCOL.md](PROTOCOL.md) for analysis boundaries, and [results/validation.json](results/validation.json) for executed checks. The source pipeline documents the meanings of counts, normalization and export versions separately. [BANC pipeline documentation](https://github.com/htem/bancpipeline), [curated type-matching repository](https://github.com/flyconnectome/drosophila_neurotransmitters)
