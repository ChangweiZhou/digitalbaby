**MiniFly: Huang 2024 physiology source audit**  
13 September 2026. Paper, supplement, GitHub history, Zenodo traces and published source tables. No new learner experiment or parameter fit.

**The new data are useful, but they reveal that our physiological reference needs a more precise definition before the next iteration.** Most of the checked observations can be reconstructed exactly from the deposited spike indices. Several important discrepancies also survive direct checks: a baseline-window difference, opposite baseline-bias signs in one panel, a detection-threshold mismatch, and differences between the model's fitting workbook and the final published tables. These are concrete source reconciliation issues. They do not establish that the biological conclusions are false, and they should not be absorbed into a new learning mechanism.

The strongest practical finding is that **12 of the 60 observations used by our reduced physiological fit differ materially from the corresponding final published observations**. All twelve concern the repulsive-odor MBON-α3 time course, including the long-term measurements. Consequently, our previously reported fit remains a reproducible fit to the released model workbook, but its exact residuals and fitted retention parameters should not be described as a fit to the final published dataset without qualification.

The biological computation worth retaining is still history-dependent teaching: learned changes in MBON output alter the DAN response to a later odor, changing the teaching effect of that odor. The paper provides experimental support for this interaction. It does **not** establish a permanent memory substrate, a synaptic capacity law, or a particular autonomous consolidation equation. Those distinctions become clearer after following the full source chain.

**What was inspected and verified.** The study is Huang, Luo and colleagues, *Dopamine-mediated interactions between short- and long-term memory dynamics*, Nature 634, 1141–1149 (2024), DOI [10.1038/s41586-024-07819-w](https://www.nature.com/articles/s41586-024-07819-w). The local full paper, its Supplementary Information, and relevant figure captions were inspected, including visual checks of the plasticity equations and the explanation of the three-hour decay switch. The accessible [full text at PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11525173/) is an alternative primary-source link.

| Source layer | Verified content | Meaning for this audit |
|---|---|---|
| Official Zenodo deposit | All five local ZIP MD5 values match the official record | The duplicate payloads and discrepancies are not explained by corrupted downloads |
| Deposited traces | Earlier intake decoded all 5,074 MAT files; this audit maps 2,268 voltage records to 1,700 published values | Enough detail for substantial measurement reconciliation without loading a large dataset into RAM |
| Published workbooks | 15 source-data workbooks, plus the model fitting workbook; 57 sheets inspected programmatically | Published observations, model predictions and fitted inputs can be distinguished |
| GitHub | Current HEAD remains `5d7c08a9a88f923169a0c3008aca68af421e9a7f` | Our existing executable source has not been superseded on the default branch |
| Local repository snapshot | All 32 locally present repository files match their Git blob hashes | No local source modification explains the prior reproduction difference |
| Additional repository assets | 16 repository files were absent from the old local subset, including two April 2024 parameter MAT files | The old local snapshot was a selected subset, not the complete repository |

The [Zenodo record](https://zenodo.org/records/10998457) contains five main-figure archives totaling 920,517,478 compressed bytes. No Extended Data archive is listed in that record. The [GitHub repository](https://github.com/schnitzer-lab/Luo_Huang_2024_MB_model) has 31 commits in the retrieved history; the all-state issues endpoint returned no issues at the time of this audit. That is a bounded observation, not proof that no discussion or correction exists anywhere.

The files contain processed fluorescence traces and already detected spike indices, not raw imaging movies. The paper's upstream procedure includes motion correction, SVD/BM3D denoising, manual ROI selection, photobleaching correction, high-pass filtering and cell-type-dependent spike detection. GitHub supplies the effective model and fitting/plotting code; its inspected tree does not supply that complete image-to-statistic pipeline. Thus an exact reconstruction from deposited spikes verifies the downstream measurement, not the original spike detector or ROI selection.

**The measurement definitions can now be made explicit.** Spike locations are MATLAB-style indices. This audit uses `t[DetectedSpikes - 1]`, with `t` in seconds. An odor response is a count rate during the five-second odor presentation minus a baseline count rate. For each fly, the Methods define the learned bias as

\[
\Delta r_o(T)=r_o(5\!:\!10\ \mathrm{s},T)-r_o(\text{baseline},T),
\qquad
B(T)=[\Delta r_+(T)-\Delta r_-(T)]-[\Delta r_+(\mathrm{Pre})-\Delta r_-(\mathrm{Pre})].
\]

For Fig. 5j and 5k, the reported quantity is a within-fly ratio, respectively `B(3 h)/B(5 min)` or `B(35 min)/B(5 min)`. The average of these ratios is not generally the ratio of the two group-average biases. This matters particularly when a fly has a small or sign-changing denominator. No undefined ratio was replaced with zero.

The paper describes five-second baseline intervals in relevant captions. Directly counting spikes in `[0,5)` and `[5,10)` reproduces only **28 of 1,700** checked published values within `1e-8`. Using the final three baseline seconds, `[2,5)`, reproduces **1,524 of 1,700**. This convention was first inferred from two records, Fig. 3 PPL1-γ1 Fly01 Pre CS+ and CS−, then kept fixed for the remaining comparisons. Of the other 1,698 values, 1,522 reconcile exactly. This was an exploratory source audit, not a preregistered analysis; no per-record windows were optimized.

For example, the Fly01 Pre CS+ trace contains 91 odor-period spikes and 72 spikes in the final three baseline seconds. `91/5 - 72/3 = -5.8 Hz`, exactly the value in source workbook MOESM8, Panel e, B4. The corresponding CS− calculation gives `-3.733333... Hz`, matching F4. The matched identities use the archive's panel, cell type, local fly number, cue and phase, not a permutation chosen to improve numerical agreement.

The extensive exact agreement makes the three-second baseline a strong explanation of the released statistics. It still needs author confirmation because it differs from the written five-second description. The audit labels this rule as inferred, rather than silently presenting it as the published Methods.

**The remaining discrepancies are localized.** The detailed ledger remains in the project and can be regenerated. The compact results retain aggregate checks and the critical exceptional cells.

| Comparison | Result under the inferred three-second baseline | Interpretation |
|---|---:|---|
| Fig. 3, ten of eleven neuron types | 960/960 values match | Strong trace-to-table agreement |
| Fig. 3, MBON-α2sc | 0/96 match | Every deposited file records detection limit 2.0; Methods specify 3 SD for this type |
| Fig. 4, attractive and repulsive MBON-α3 conditioning | 288/288 match | The final published long-term response tables are reconstructible |
| Fig. 4, feedback-blocking and control cohorts | 192/192 match | A useful experimental intervention has a clean downstream measurement chain |
| Fig. 5j, attractive odors | 30/30 ratios match | Control, early and late extinction values all reconcile |
| Fig. 5j, repulsive odors | 13/30 exact; maximum remaining ratio difference 0.0003751 | Small numerical discrepancies; cause not established |
| Fig. 5h, five-minute bias | 0/42 match subtraction; 42/42 match addition of Pre bias | A systematic definition/sign discrepancy |
| Fig. 5h, 24-hour bias | 41/42 match subtraction of Pre bias | One specific value remains discordant |
| Fig. 5k, 35-minute/five-minute ratio | 0/20 match | Remains unresolved; do not use as an exact numerical acceptance target yet |

For MBON-α2sc, the detection metadata difference is a plausible explanation, not a demonstrated cause. The highest absolute trace/table difference is 15.6 Hz. Re-running a detector with a guessed noise estimate could create a superficially better result without reproducing the original preprocessing. No redetection was performed.

For Fig. 5h, all 42 five-minute workbook values instead satisfy

\[
B_{\mathrm{released}}(5\mathrm{min})
=[\Delta r_+(5\mathrm{min})-\Delta r_-(5\mathrm{min})]
+[\Delta r_+(\mathrm{Pre})-\Delta r_-(\mathrm{Pre})].
\]

This is an exact arithmetic relationship to the deposited records. It is inconsistent with the Methods' stated subtraction if the archive labels identify the intended records. It does not by itself establish whether the analysis, record labeling or export was responsible. Neither the workbook nor the source files were corrected.

The single discordant 24-hour value is the 900-second-spacing group's Fly13: MOESM10, Panel h, H16 is `14.8 Hz`, whereas the documented paired subtraction applied to the corresponding deposited records gives `1.266667 Hz`. The other 41 long-term values agree to numerical precision.

![Reconciliation of Fig. 5h source definitions](/Users/pencilbard/Documents/second_try/minifly/physiology_source_audit_20260913/figures/figure5h_source_reconciliation.png)

| ISI | Published five-minute bias, Hz | Trace-derived five-minute bias, Hz | Published 24-hour bias, Hz | Trace-derived 24-hour bias, Hz |
|---|---:|---:|---:|---:|
| 60 s | −28.405 | −21.776 | −3.686 | −3.686 |
| 360 s | −25.629 | −28.152 | −15.124 | −15.124 |
| 900 s | −26.676 | −22.971 | −4.267 | −5.233 |

These are group means over 14 flies per spacing condition. The trace-derived values use `[2,5)` baseline and the Methods' paired bias subtraction. SEMs are in `figure5_measurement_definitions.csv` and the figure. The 360-second group retains the strongest mean 24-hour depression under both sources. This audit has not recomputed inferential tests or declared a corrected biological dataset.

Fig. 5k deserves particular care because it supports the appealing claim that early extinction transiently strengthens long-term plasticity. The published mean ratios are 0.642897 for control and 1.127638 for extinction; our trace-derived ratios are 0.247993 and 1.032011. The direction of the group-mean contrast remains, but none of the twenty individual ratios matches. Omitting Pre subtraction also fails to explain the panel. We should preserve the source discrepancy rather than choose whichever version favors a mechanism.

**The duplicate time points reach the published table.** The earlier intake identified two exact duplicate voltage payload pairs: PPL1-α3 Fly03 CS+ and CS− each occur at both Pre and one hour. This audit confirms that their values also appear identically in the published workbook:

| Cue | Pre cell | One-hour cell | Value, Hz |
|---|---|---|---:|
| CS+ | MOESM8 / Panel e / B66 | E66 | −3.866667 |
| CS− | MOESM8 / Panel e / F66 | I66 | −9.533333 |

The corresponding group means are among those used by the model workbook. We therefore cannot dismiss the issue as an archive-only duplication that obviously did not reach the numerical inputs. We also cannot identify which time label is correct from these files. Removing a value would invent an exclusion policy; none was removed.

The earlier 113 exact-payload duplicate pairs include 78 Fig. 2–Fig. 3 pairs, 32 Fig. 3–Fig. 4 pairs, these two same-figure time pairs, and one shared Fig. 4 shock control. Cross-figure reuse can be legitimate, but a train/validation split based solely on ZIP or figure would leak observations. Fly labels are local, sometimes renumbered between reused records. Exact duplicates provide known links; lack of identical payloads does not prove two records came from different animals. Physiological recordings and behavioral recordings were also taken in separate sets of flies, as the paper explicitly states, because imaging illumination disrupted normal behavior. They cannot be joined as simultaneous observations of the same animal.

**The fitting workbook and the final published tables are different numerical targets.** The authors' [README](https://github.com/schnitzer-lab/Luo_Huang_2024_MB_model/blob/5d7c08a9a88f923169a0c3008aca68af421e9a7f/README.md) identifies `Imaging_24hr_data.xlsx` as the fitting input. The paper identifies the 86 observations as 40 attractive-odor measurements from Fig. 3, 24 MBON-α3 measurements from Fig. 4, and 22 repulsive-odor measurements from Fig. 2 and Extended Data Fig. 8. The audit traced all 86 to these corresponding published sources.

At a `1e-6 Hz` numerical/rounding tolerance, 66 mean/SEM pairs reconcile and 20 differ materially. Four small mean differences in the matching group are ordinary stored rounding, at most `1.11e-7 Hz`.

| Discrepant group | Mean/SEM pairs | Largest absolute mean difference | Relevance to current reduced fit |
|---|---:|---:|---|
| Attractive PPL1-α′2α2 | 8 | 0.600000 Hz | Excluded from our 60-observation reduced fit |
| Repulsive MBON-α3 | 12 | 3.299206 Hz | All twelve included in our reduced fit |

For a concrete long-term example, the model workbook's repulsive CS+ MBON-α3 value at 24 hours is `−3.238095 Hz`; the final published table gives `−4.544444 Hz`, which the deposited traces reproduce. Both means and SEMs differ. The model workbook's corresponding CS− value is `10.519048 Hz`, compared with the final `9.477778 Hz`. The audit does not know whether the workbook represents an earlier cohort, an earlier processing version or another documented choice that has not been released.

The six repulsive DAN entries require another distinction. They are reproduced by averaging the Fig. 2i responses to 1% OCT and 0.3% BEN, then assigning the same resulting mean to CS+ and CS−. The SEM cells likewise equal the arithmetic mean of the two source SEMs. This reproduces the released input convention; it is not a fresh uncertainty calculation for a pooled or paired dataset. These six cells contain three repeated aggregate values, and the observations across conditioning are repeated within flies. Treating all displayed cells as independent biological measurements would overstate the independent evidence.

Our iteration-4 extractor faithfully preserved the original workbook. Its fit explicitly retained roles 0, 2, 3 and 5, yielding 60 observations. The source difference is therefore upstream of that extraction, not a newly found CSV parser failure. Prior MiniFly task results should remain frozen and reproducible. Their physiological interpretation needs the provenance qualification above; this audit does not recalculate or replace their fitted models.

**What GitHub resolves, and what it does not.** The following observations come from the pinned executable source and its retrieved history, rather than from an inferred biological design.

| Source feature | Evidence | Consequence |
|---|---|---|
| Figure function alias | `config_path_and_function.m` maps `Dx_steady_state_MBON_revision` to `Dx_steady_state_MBON_0301_2023` | The apparent name mismatch is valid configuration, not a missing function |
| MBON bounds | Core lines 14–16 and 56: baseline `[35.2,9.0,11.2]`, absolute cap `[71.66,17.9,31.16]` | These are response-rate bounds; they do not establish finite KC weight capacity |
| Signed effective KC weights | Core lines 150–156; weight rectification is commented out | Parallel inhibition has been absorbed into effective weights; these are not literal nonnegative cholinergic synapses |
| Fast/late alpha decay switch | Core lines 124–140 | The late lifetime is selected after three hours relative to the scheduled last training bout |
| Adaptation update | Core lines 81–88 | A bout-level approximation; recovery is applied in the executable update even during an odor bout |
| Fitting uncertainty | `parameter_CI_sample.m`, lines 16–23 | Draws come from an inverse-Hessian Gaussian approximation, not resampled animals |
| Fitting workflow | `fit_nonlinear_models.m`, lines 18 and 166–213 | Released default disables GA; optional GA follows the local fit and CI calculation |
| Figure 5h/j/k stimulus specialization | Figure scripts reset odor values and scale shock-related teaching by 1.5 | These protocol-specific settings are part of reproduction, not universal biological constants |

The [core implementation](https://github.com/schnitzer-lab/Luo_Huang_2024_MB_model/blob/5d7c08a9a88f923169a0c3008aca68af421e9a7f/matlab_code/model_functions/Dx_steady_state_MBON_0301_2023.m#L9) and [fitting script](https://github.com/schnitzer-lab/Luo_Huang_2024_MB_model/blob/5d7c08a9a88f923169a0c3008aca68af421e9a7f/matlab_code/model_fitting/fit_nonlinear_models.m#L115) provide the precise locations. Supplementary section 7 describes GA followed by local optimization. The released default and optional order differ from that description; simply rerunning its default fitting command is not a documented reproduction of the paper's complete optimization history. No optimization was run here.

The supplementary differential equation for adaptation gates recovery by odor absence. The recursive implementation instead applies exponential adaptation followed by recovery within each bout and uses the average of the start/end amplitudes. This is a relevant distinction between the general differential model and its fast approximation. It reinforces our earlier requirement to check timing and bout partitioning before generalizing the model to arbitrary online tasks.

The source's CS− duration is also populated from `t_CS_plus` in `experimental_condition.m`. It makes no difference to the published equal-duration protocols, but unequal-duration probes would not have the requested timing without an explicit, independently checked correction. This audit did not alter that code.

The public core-file history does not reveal a different active MBON cap that explains the previously observed `−31.16` versus `−30.59 Hz` saturation mismatch. Retrieved changes concern comments and helper-function organization. Supplementary Table 2 rounds the α3 absolute cap to approximately 31.2 Hz, consistent with current code. The earlier model-output mismatch consequently remains open; it cannot be declared solved by the newly detected trace/table differences, since model-to-model reproduction and observation preprocessing are separate links.

The repository also contains two `17-Apr-2024` parameter files in its fitting-output folder. The README and published figure scripts explicitly select the `27-Mar-2023` files that we already have. The additional files deserve a hash-verified parameter comparison, but their existence does not authorize substituting them for the named reference. Downloads failed on local DNS resolution; exact manual links are listed below. Work continued without them.

**The mechanism is better specified than the permanent-memory substrate.** In the general supplementary formulation, KC and DAN activity are each filtered into eligibility variables. Plasticity includes an antisymmetric term of the form `ΔKC × filtered ΔDAN − ΔDAN × filtered ΔKC`, followed by decay and a further weight filter. The reduced implementation condenses this into effective teaching coefficients for the experiment's fixed three-second CS–US delay. The paper explicitly directs other temporal protocols to the more general equations. Importing only the fitted recursive coefficients into an arbitrary event stream would discard those conditions.

The persistent component has a different evidential status. Supplementary section 3 explicitly says that a single exponential could not describe the measured memory decay, and that separate alpha time constants were introduced before and after three hours; a biexponential process is mentioned as one possible basis. The switch is a fitted modeling assumption. It does not show that a biological neuron detects a three-hour deadline, nor identify a molecular process that makes storage permanent. Our own continuous fast/slow implementation remains a hypothesis addressing this modeling limitation, not a mechanism discovered by downloading the traces.

The new records provide no direct measurements of individual KC→MBON weights, their bounds, the number of independently storable associations, or the PN/KC code for our synthetic banks. Many different latent weights can produce a similar mean MBON response. This dataset therefore cannot decide whether our high-load failure comes from compression, write interference, retrieval or a particular synaptic bound. Those distinctions still require the existing task-level controls and later carefully designed experiments. The data can make physiological targets more defensible, but cannot replace a capacity test.

**Which empirical constraints should carry forward.** The data support a graded assessment rather than a single pass/fail judgment about the paper.

The cleanest downstream source chains here are Fig. 4 conditioning and feedback-blocking cohorts, and Fig. 5j with attractive odors. Fig. 4's intervention tests the contribution of MBON output to teaching responses; it is more informative about feedback than merely observing a graph motif. It still does not determine every physiological gain of the anatomical pathway, or distinguish every intervening cell or output site. BANC, hemibrain and FlyWire identities remain separate; physiology from these experimental flies does not provide synapse-level identity matches to those connectomes.

For attractive odors in Fig. 5j, the published and reconstructed three-hour/five-minute mean ratios are 0.6411 for control, 0.6920 for early extinction and −0.0539 for late extinction. All thirty individual ratios reconcile. The source table labels early versus control nonsignificant and the late contrasts significant. Thus this panel strongly motivates testing timing-dependent revision, but it does not establish a large three-hour retention improvement from early re-exposure. The separate 35-minute benefit in Fig. 5k is the panel whose twenty numerical ratios remain unresolved.

For spacing, 41 of 42 long-term values reconcile, and the mean 360-second advantage remains. The early normalization and one late outlier need clarification before using the full curve as an exact fit target. A physiological acceptance test should not be tuned to those unexplained values.

Finally, the 48-hour evidence is limited: Extended Data Fig. 7c reports **two flies at 48 hours**, excluded from its ANOVA, compared with twelve at earlier times. The existing source workbook MOESM17, Panel c, G18 and M18 confirms `n=2`. Its 48-hour individual values are present in the workbook, but the newly downloaded five ZIPs provide no 48-hour trace files. This is not enough to identify a precise long lifetime, and it cannot validate permanent or multi-week storage. More long-term animals and intervening time points would help far more on this question than another large anatomical edge list.

**A bounded path to the next clean iteration.** The immediate next artifact should be a frozen physiological reference with explicit observation definitions, source version, local animal/cohort identities and discrepancy flags. That source audit should precede any decision to refit. The current learner and its accepted task metrics should remain the comparison reference.

For later experiment design, maintain two separate evaluations: physiological phenomena constrained by the resolved source chains, and task-level acquisition, retention, revision and capacity. Passing a two-odor physiology curve does not validate many-association memory. Conversely, poor compressed capacity is not evidence against the entire feedback mechanism. A candidate should be compared with both the full-size parent and matched compression controls, using the same write schedules and retrieval policy.

Before changing a memory equation, the most useful new questions are specific: does a source-corrected physiology fit materially change the fast/slow parameters; does feedback change later teaching under resolved protocols; and can the resulting learner preserve old associations while learning new ones at matched memory size? These are proposed directions only. They have not been implemented or scored in this audit.

For any eventual split of physiological data, keep known reused records together, respect repeated measures within flies, and use protocol-level separation where subject identity is incomplete. Neither figure number nor a fresh filename establishes independent validation. Since MiniFly has already used the Fig. 5 phenomena in design and evaluation, they are not newly untouched validation targets for this project.

**Specific assistance that would resolve remaining uncertainty.** No additional large connectome download is needed to complete this audit. A short response from the data authors would be particularly useful on the following points; no message has been sent:

1. Confirm the actual baseline window and count/smoothing procedure used for the per-fly source tables. Most reconciled values equal odor `[5,10)` minus baseline `[2,5)` count rates.
2. Explain the two PPL1-α3 Fly03 Pre/one-hour duplicate payload pairs and their matching values in MOESM8, Panel e, B66/E66 and F66/I66.
3. Clarify Fig. 5h's opposite Pre-bias signs at five minutes versus 24 hours, and the 900-second Fly13 H16 discrepancy.
4. Supply the source calculation or corrected mapping for Fig. 5k's twenty within-fly ratios, and clarify the MBON-α2sc detection limit 2.0 versus the Methods' 3 SD.
5. Identify the cohort/processing version behind the twenty discrepant fitting observations, and which code/parameter combination produced the final published prediction medians.

Mr. Zhou can optionally download the two smaller GitHub assets that failed locally, into this same audit's `sources` folder:

- [April 2024 two-module fit, about 1.25 MB](https://raw.githubusercontent.com/schnitzer-lab/Luo_Huang_2024_MB_model/5d7c08a9a88f923169a0c3008aca68af421e9a7f/matlab_code/model_fitting/figures/Dx_steady_state_nonlinear_3_17-Apr-2024_2modules.mat), expected Git blob SHA-1 `acfcdb4b9bbd7055bc0460015057669f5fb50970`.
- [April 2024 three-module fit, about 1.80 MB](https://raw.githubusercontent.com/schnitzer-lab/Luo_Huang_2024_MB_model/5d7c08a9a88f923169a0c3008aca68af421e9a7f/matlab_code/model_fitting/figures/Dx_steady_state_nonlinear_3_17-Apr-2024_3modules.mat), expected Git blob SHA-1 `84a7927b4b21304e7d109862c43854c7156cc53f`.

Git blob SHA-1 includes the Git `blob <length>\0` header and must not be compared directly with a bare file SHA-1. These optional files may help resolve parameter provenance; the present findings do not depend on them.

**Reproducibility and delivery.** All work is confined to `minifly/physiology_source_audit_20260913`. The audit reads the original ZIPs one member at a time. `audit_sources.py` inspects workbooks with the bundled Python runtime; `reconcile_measurements.py` uses the existing scientific environment for SciPy MAT support; `summarize_findings.py` produces the final tables. Exact commands and dependency hashes are retained in the manifest. The scripts do not invoke a learner, fit parameters, redetect spikes or modify the original workbooks.

The compact bundle contains this report, the final reconciliation and fit-provenance tables, the key figure, analysis configuration, acceptance checks and the manifest. It excludes source ZIPs, source workbooks/PDFs, unchanged third-party code, detailed trace/cell caches, old archives and raw task outputs. No iteration number is advanced. Numerical reconstruction successes are evidence about the source chain, not a claim that a functioning persistent fly-brain core has now been achieved.

**Primary-source pointers.** The source-data workbooks supporting the new findings are [Fig. 2, MOESM7](https://static-content.springer-cdn.com/esm/art%3A10.1038%2Fs41586-024-07819-w/MediaObjects/41586_2024_7819_MOESM7_ESM.xlsx), [Fig. 3, MOESM8](https://static-content.springer-cdn.com/esm/art%3A10.1038%2Fs41586-024-07819-w/MediaObjects/41586_2024_7819_MOESM8_ESM.xlsx), [Fig. 4, MOESM9](https://static-content.springer-cdn.com/esm/art%3A10.1038%2Fs41586-024-07819-w/MediaObjects/41586_2024_7819_MOESM9_ESM.xlsx), [Fig. 5, MOESM10](https://static-content.springer-cdn.com/esm/art%3A10.1038%2Fs41586-024-07819-w/MediaObjects/41586_2024_7819_MOESM10_ESM.xlsx), [Extended Data Fig. 7, MOESM17](https://static-content.springer-cdn.com/esm/art%3A10.1038%2Fs41586-024-07819-w/MediaObjects/41586_2024_7819_MOESM17_ESM.xlsx), and [Extended Data Fig. 8, MOESM18](https://static-content.springer-cdn.com/esm/art%3A10.1038%2Fs41586-024-07819-w/MediaObjects/41586_2024_7819_MOESM18_ESM.xlsx). Exact cells are retained in the final metric tables. Interpretation of equations uses Supplementary Information sections 2–7 and Tables 1–3, supplied alongside the paper. Source checks refer to the pinned GitHub commit and the verified Zenodo v1 files, not to an unspecified model generated by a chat thread.
