# MiniFly V175 — self-hostile post-run audit (HA-01)

**2026-11-01 · Global version V175 unchanged · No new experiment or adoption**

**Verdict:** DEV_SCREEN_FAILED stands. This document is a new adversarial analysis of the frozen design, implementation and published three-cycle results at commit 1af8a4977be698ed2501977d16250f5c48e0dbf2. No model was run or outcome changed for this audit. The original raw codes, responses, shadow panels, local binary checkpoints and event ledgers were excluded from GitHub, so this is a source/design and aggregate audit, **not a complete independent replication**.

## Results that cannot be rescued by threshold changes

| Item | W1 BASE / LPS | W2 BASE / LPS |
|---|---:|---:|
| OLD before NEW | 58 / 59 | 41 / 41 |
| Final OLD | 27 / 26 | 21 / 24 |
| Final NEW | 16 / 16 | 13 / 13 |
| Final complete recalled /80 | 43 / 42 | 34 / 37 |
| Initially correct OLD still correct | 26/58 vs 26/59 | 18/41 vs 19/41 |
| OLD×NEW Jaccard | 0.193404 / 0.185816 | 0.322377 / 0.315188 |
| Jaccard reduction | 0.007589 | 0.007189 |
| OLD8→OLD16 LPS stability | 0.971850 | 0.978667 |
| Mean OLD sentinel damage BASE / LPS | 0.006303 / 0.012028 | -0.003854 / -0.001492 |
| Total final-correct per diagnostic CPU second BASE / LPS | 0.4106 / 0.3689 | 0.3257 / 0.3323 |

P1 and P2 both fail in both worlds. P3 passes as a **technical/budget check**, not a sample/compute efficiency improvement. LPS cost ~8.7% and ~6.7% more diagnostic-inclusive CPU per full life, respectively; pooled observed throughput was ~4.7% worse, descriptively. This does not estimate production throughput. The gate failures remain even if the arbitrary +8 threshold is ignored: final deltas were -1/+3, with no robust replicated gain or strong mechanistic effect.

## Critical design defects attributable to the proposer

**1. Wrong learned signal.** The actual rule is

\[
L^{t+1}_{ij}=\min(1,(1-\delta)L^t_{ij}+\eta z_i^{(0)}z_j^{(0)}),\qquad \eta/\delta=64.
\]

Without clipping it tracks roughly $64\mathbb E[z_i^{(0)}z_j^{(0)}]$. That is **uncentered coactivation**, not covariance whitening and not a measure of damaging native value-write interference. It may suppress jointly active neurons even when their coactivity is useful. The central premise, that reducing this trace improves retention, was never established.

**2. Wrong causal proxy.** Whole-KC binary Jaccard need not predict the actual native write impact. In first-order notation, damaging NEW→OLD interference is $\Delta m_{\rm OLD}\approx\nabla_\theta m_{\rm OLD}^{\top}\Delta\theta_{\rm NEW}$, involving native fast/slow state, teacher sign/strength, writable KC support, adaptation, and reader. The prespecified early NEW shadow events did **not** show improved OLD sentinel margins under LPS. We should have measured writer-effective cross-talk **before** designing a larger encoding module. An evaluator-only 80/80 exact-key lookup ceiling cannot show that the fixed native writer and reader can preserve arbitrary labels.

**3. No guarantee of separating similar memories.** For identical raw input drive, the shared deterministic circuit maps both memories identically regardless of its current $L$. For near-identical inputs, the same inhibition may replace the same shared winners in both codes, moving the conflict rather than resolving it. The random graph has no mechanism for novelty detection, distinguishing incompatible memories, or relevance.

**4. Synthetic topology was unsupported.** V175 randomly permutes 5,177 KCs and builds fixed degree-16 rings, not edges determined by native receptive-field overlap, writable support, neuropil distance or the fly's effective APL inhibitory circuit. It should not have been described as biologically grounded KC-to-KC inhibition. Established mushroom-body sparse inhibition is mediated principally by the GABAergic APL neuron with nonuniform spatial effects (e.g., Lin et al. 2014; Amin et al. 2020). An effective KC-KC coupling via APL is possible, but V175 never modeled or justified it.

**5. The plasticity rule updates hypothetical neurons.** The circuit transmits the inhibited code $z$, yet it updates $L$ using the pre-inhibition winners $z^{(0)}$. Suppressed units can continue to strengthen penalties; newly selected units do not supply the expected postsynaptic signal. This is an engineered feedforward heuristic, **not** actual local firing-based plasticity. A single round computes $z=\mathrm{TopK}(u-\gamma Lz^{(0)})$ rather than a competitive circuit's self-consistent solution using $z$.

**6. Parameter scaling was not tied to effect.** Gamma was frozen from positive-drive standard deviation on 64 independent random five-byte prompts; the main cues instead have structured repeated suffixes. No unlabeled technical gate checked the distribution of active-neighbor inhibition, top-k cutoff shifts, actual swapped KC counts, reader-effective support, or a plausible range of OLD×NEW decorrelation. The Jaccard 0.05 target was set without empirical engineering reachability evidence. This is not a reason to increase gamma on already opened worlds.

**7. No meaningful plasticity-vs-conditioning comparator.** There was no dose-matched frozen-inhibition arm. Therefore even a positive result could not have established the learned coactivation rule was better than adding a static inhibitory perturbation. The original write-on/write-off versus L-update-on/L-update-off factorial was also missing, making value overwrite versus address drift unresolved.

## Observability and execution mistakes

- The reported OLD-address stability compares OLD-pass-8 with OLD-pass-16 only. **It omits OLD16→FINAL drift**, the period during which NEW experiences changed $L$ and old memory recall collapsed.
- “80 of 80 codes changed” is a binary activity indicator; it could mean a tiny number of changed cells per code. No swap-count, rank-gap or native writable-KC-overlap distribution was published.
- At the final states only 1,314/41,416 (3.17%) and 819/41,416 (1.98%) inhibitor edges were nonzero; nevertheless the code counted ~1.94 billion inhibitor-edge visits and 95.8 million full-edge update visits, plus cloning overhead. The implementation is local in edge-update dependencies but not a demonstrated sparse-work implementation; the native global TopK remains.
- Same-state shadow interventions were eight early NEW payload writes per arm/world, with mean teacher-prefix OLD margins. They are informative but cannot identify cumulative late-phase forgetting, nor do they quantify which portion is address drift.
- Two worlds and their item counts do not justify generalization claims. Every scored association was taught; this is E1/E1-R only, with no evidence of E3 unseen-rule learning.
- The main technical audit checked source hashes and saved results; GitHub lacks the original raw checkpoint and panel bundle for full forensic reconstruction. Report this as a reproducibility limitation rather than inventing missing observations.

## What the development run did correctly

The registered P1/P2/P3 screens were preserved, no main world was silently replaced, and both negative development worlds completed. Source parity of BASE and zero-L paths, signed native teaching, populations, and clone/restore checks were reported as passing. The invalid assertion that every arbitrary nonzero L must change every arbitrary code was corrected **before** main training; a separate continuation dispatch failure was also logged, not hidden. Read-only probes froze model state. Write-inhibited native events properly included passive decay. These observations support the reported failure as a useful negative experiment, **not** adoption.

## Necessary next step: retrospective mechanism diagnosis, no immediate V176

Do not auto-retune V175 or run a larger confirmation. On the original execution host only, if raw saved states remain available, carry out a clearly labeled **retrospective diagnostic on already opened worlds**, never present its outcome as new independent efficacy evidence:

1. Compute full OLD8, OLD16, NEW4, FINAL and REST128 code transitions, particularly OLD16→FINAL: swap counts, reader/writable-support overlap, inhibitory rank shifts, and $L$ saturation/sparsity.
2. Clone exact OLD16 states and advance identical NEW records and native clocks in a 2×2 observation-vs-value-write and frozen-vs-updating-$L$ intervention, retaining own-byte/EOS OLD and NEW outcomes at every pass.
3. Map same-state native NEW-write effects on old true-class margins over the *whole* NEW schedule, with zero-write passive controls, including self-target gain and a correctly weighted readable-support kernel.
4. Compare code geometry, actual synaptic displacement and decoder margins, not only binary Jaccard; mark every post-hoc result as diagnostic.
5. If raw assets cannot be made available, label all these mechanisms **UNVERIFIED** rather than asserting causes from aggregate numbers.

Only after separating value interference from address drift should a **different** prospective experiment be considered, perhaps local fast/slow consolidation with update/revision controls. Require prevention of harmful OLD writes **while retaining effective NEW learning and later correction**, measured on the actual native reader. Do not smuggle attention, label-dependent supervision, task-specific structure, replay or a fitted external answer decoder into the candidate.

## Governance and sources

**Global version remains V175.** Status DEV_SCREEN_FAILED; latest completed development V175; adopted core unchanged; next unallocated number V176. This self-hostile audit is HA-01, a document only, not a new version or trial.

Original design: https://github.com/ChangweiZhou/digitalbaby/blob/codex/minifly-consolidated-20261023/minifly_v175_local_decorrelation_design_20261101/DESIGN.md

Implementation: https://github.com/ChangweiZhou/digitalbaby/blob/codex/minifly-consolidated-20261023/minifly_v175_local_decorrelation_dev_20261101/v175lib.py

Results and saved audit: https://github.com/ChangweiZhou/digitalbaby/blob/codex/minifly-consolidated-20261023/minifly_v175_local_decorrelation_dev_20261101/REPORT.md

Wang, memory analogy: https://alexwang.ai/posts/associative-memory/

Lin et al., 2014: https://pubmed.ncbi.nlm.nih.gov/24561998/

Amin et al., 2020: https://pmc.ncbi.nlm.nih.gov/articles/PMC7541083/

Historical run STATUS and PACKAGE fields stating GitHub_pushed=false were generated before publication; PUBLICATION.md and the actual result commit document the later upload. Do not rewrite frozen receipts to erase this historical distinction.
