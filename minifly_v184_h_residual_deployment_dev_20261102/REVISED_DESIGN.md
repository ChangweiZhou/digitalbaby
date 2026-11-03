# V184-R1 — qualification revision and frozen execution conventions

Anchor: `ChangweiZhou/digitalbaby`, consolidated branch, design commit `3b86ebe8c7687e4abc3bd2cba33910dc6ca75229`; completed-parent publication `840f2a7b9dcd25f4262c3e17058bb1cf30139bf2`. The owner requested the three-cycle exercise. Implementation and technical tests are authorized; **scientific main NOT RUN**. No V185 is allocated and no adopted core changes.

The candidate remains exactly D1: HRES uses one existing output matrix on `m*(u+0.5*h)`, retains T, all37,313 learned parameters and the original H/T fixed-feedback rules. NORMAL is the literal ordinary circuit. NORM_SHAM uses the original post-dropout norm ratio, epsilon1e−6 and cap8. There is no alpha/gain grid, second head, extra learned gate, new task, replay, changed warmup, class mask, attention, Transformer, ELM or arithmetic structure.

## 1. What the three cycles established

Cycle1 verified the exact canonical sensor digest, every M1 corpus/split/correction record hash, physical sparse fan-in/fan-out, parameter census, and source parity. NORMAL and HRES(alpha=0) reproduce the parent forward/cache and dropout RNG bit-for-bit. Empty prefixes with both feature norms zero are harmless and logged. Deliberately silent-T/nonzero-H and over-cap gain fixtures correctly raise CONTROL_NOT_QUALIFIED; these are validity-guard tests, not scientific learner trajectories.

Cycle2 verified NORMAL's exact next update, moments and RNG against the literal parent. HRES and NORM_SHAM use their actual predicted error while retaining H/T's unchanged source-local formulas and one actual dropout mask. The existing O matrix's local categorical gradient matches an independent float64 reference and a finite difference. Actual H/T/O post-Adam movements are recorded; matched feature norm is **not** claimed to match writer movement.

Cycle3 verified moment-bearing full-state serialization, equal next-update continuation, identical EDIT/NOEDIT ancestor states and RNG, and different feedback producing different weights without changing the random sequence. Whole-record tails194/52/7, full256 batches and singleton EOS were exercised. Static arm/alpha metadata now guards against accidental cross-arm checkpoint restoration, with no numeric forward/update change. The six pure saved-metric canaries reject mean-shortcut, EOS and alignment false passes while excluding obsolete original-A labels from post-C retention.

No scientific lifetime, A2 trained HRES score, or learned deployment advantage was measured. The later cold-state full-roster evaluation was an engineering resource/norm profile, not a scientific efficacy screen or a reason to tune alpha/cap.

## 2. Explicit norm-sham validity domain

The definition and cap remain unchanged. Before labels enter any update, compare the actual sham feature norm with the HRES target norm at the same activities/mask. A site is numerically material when absolute mismatch exceeds `1e−7 + 1e−6 * target_norm`. This is a pre-science float32 comparison tolerance, not a new scientific noninferiority margin or a tolerated mismatch fraction.

Any material mismatch in the **actual NORM_SHAM trajectory**, including read-only panels or generation, stops the comparison as CONTROL_NOT_QUALIFIED. Preserve partial files/counters and complete-life receipts; do not adjust cap8, ignore difficult sites, add a fallback direction, select another dose or return a partial efficacy verdict. Benign both-zero sites are reported separately. The test does not claim the sham is valid for all possible weights: the explicit silent-T and clipped witnesses show why a runtime guard is necessary.

On the entire cold M1 ten-panel roster,33,307 sites per arm, the tested sham had zero material mismatch and zero clipping;544 empty-prefix both-zero sites were harmless. This is admission at the tested source state, not a guarantee that learning will remain in the domain.

The control removes the explicit H residual **vector** while retaining the combined feature norm. That scalar includes H/T alignment through the resultant norm; it is not completely H-information-free. The causal contrast is direction beyond matched magnitude, not equivalence of all information or all optimizer movement. Changing the readout also changes future prediction errors and hence learned H/T states; even a positive package result would not prove that the old NORMAL H representation was already adequate.

## 3. Freeze the previously underspecified inference rules

Primary: pooled true byte-plus-EOS A2 loss, paired across the same64 held A records. Use10,000 paired-record resamples, NumPy generator seeded by `SHA256('V184|M1|A2|paired-bootstrap|10000')` under the source first8-byte conversion. Each draw resamples64 records with replacement and divides paired total loss sums by the resampled actual target count. Use two-sided percentile2.5/97.5 intervals with linear quantiles. Use the same registered resampling stream for both controls. Require each total-loss CI upper endpoint strictly below zero. Payload-only paired intervals and point outcomes are also displayed, without inventing a post-outcome co-primary rule.

These are descriptive record-resampling intervals conditional on one opened text ecology, not64 independent worlds, proof of ecological exchangeability, sealed confirmation or certified long-term safety.

Alignment: for each actual input position, take every held record having that site, including empty-prefix and real-EOS sites. If at least two records qualify, sort their IDs by `SHA256('V184|M1|A2|H-align|POSITION|INPUT_SHA|RECORD_INDEX')`, then assign each the next record's H vector cyclically. The map is non-self, position-matched and label-blind. Never condition on predicted loss, class, EOS status or a favorable permutation. Keep the trained HRES T outputs and O fixed. Score aligned/exchanged routes only on identical eligible sites; require aligned pooled loss strictly below exchanged loss. Save per-site log probabilities/labels/positions, mapping and per-record sums/counts so the denominator and gate are reconstructable. The profile found4,054 eligible sites out of4,055 M1 held sites. This diagnostic never teaches the learner. Exchanging H may change joint feature magnitude and H/T compatibility; the test demonstrates dependence on the registered alignment, not standalone H representation quality or a unique causal bottleneck.

Lifetime: HRES-minus-each-control point losses must be nonpositive at B3 A-held/B-held and EDIT2 corrected SAME_KC/ALTERED_KC/B-held. Do not add an unregistered post-C original-A guard that protects superseded labels. Correction specificity is positive EDIT2-minus-NOEDIT2 new/old log2-odds effect on changed targets, alongside all still-valid guards. EOS comparisons are non-worsening NLL, Brier and premature rate against both controls on A-held/B-held at A2/B3, then corrected EDIT-held/B-held at EDIT2. These point inequalities are not certified noninferiority or every-item safety proofs.

All original D1 thresholds, fixed data, doses and actual learner mechanisms remain unchanged. No result selects a checkpoint, permutation, rate, gain, arm or pass count.

## 4. State, output and exposure contract

Main: one opened M1 ecology; NORMAL/HRES/NORM_SHAM;560 optimizer-record applications/34,364 targets per life;1,680/103,092 across three lives. Six C child branches are correlated descendants, not independent ecologies. Required full weights/moments/clock/dropout/static-arm checkpoints are birth/A2/B3/EDIT2/NOEDIT2 for each arm:15 full states. All ten stage curves per arm remain saved as summaries; intermediate stages do not secretly create additional full states.

Use the full ten-panel roster at all30 summary snapshots for an exact999,210 read-only target evaluations/16,320 record views. Save full per-site arrays at the15 mandatory states:499,605 sites. In addition, charge the H-alignment backbone4,055 sites and two eligible head routes8,108 target evaluations, maximum18,432 own generated outputs and864 genuinely observed prefix bytes. Own EOS, premature EOS and truncation are separate. Arbitrary unseen-prose exactness remains descriptive, with feasible denominators.

The A2 A-train panel saves160×257 float64 log-probability sums per original record, counts and input hashes. Streaming log-sum-exp reconstructs the full-class constant mean forecast without held-label fitting, probability clipping or class masking. No renewed counting grid is fitted; use the frozen source-matched unigram4.403066543210908 and historical count references with clear provenance.

Technical qualification consumed32 conservatively counted record-equivalent fixture views,589 supervised target evaluations and12 optimizer steps;14 actual full-record optimizer presentations are included within those fixture counts. Initial label-blind sanity calls and every expected-invalid/reference/duplicate call were charged. The separate read-only profile used1,856 record views/114,132 model-site operations including alignment and generation, with no model updates or selected efficacy output. The final six gate canaries used pure metadata: zero model inputs, predictions, targets or updates. Do not present any of these as independent scientific lives.

## 5. Mini resource admission and launch boundary

Measured checkpoint sizes including static arm/alpha are458,410–458,474 bytes. The single-model inference bound is1,255,608 bytes;37,313 learned scalars remain unchanged. Observed process peak RSS532,529,152 bytes includes framework and batching overhead. Full-plan projection≈206 CPU/active-runtime seconds includes qualification, complete evaluations/generation, reconstruction/review/archive allowance and1.5 reserve. Final output census must include all newly written experiment artifacts, the separate full owner archive, review bundle and the98,490,254-byte isolated compatible SciPy runtime. Existing shared Python/Torch/NumPy installations are recorded prerequisites, not copied into the archive; this is not a self-contained runtime image. RSS includes the real live framework cost.

The initial conservative full-output projection is181,833,294.5 bytes before the final small technical package; final census is in RESOURCE_ADMISSION.json. Hard limits remain40,000 learned scalars,8 MiB inference,1 MiB full checkpoint,5,000 physical training/qualification records,600,000 supervised training/qualification target evaluations,900 aggregate CPU seconds,1,800 active execution-wall seconds,1 GiB RSS and256 MiB complete output. No silent budget increase or artifact deletion is permitted.

Wall budgets measure execution processes, not code-authoring gaps or owner wait. Carry all measured qualification CPU/process wall and conservative setup/review allowances into a later launch. Source/technical fixtures are not rerun for reporting. Main requires a prospective final source/input/runtime lock and a separate owner instruction; no scientific worker has started during this exercise.

## 6. Evidence and disposition

Status: TECHNICAL_QUALIFIED_WITH_RUNTIME_CONTROL_GUARD. The final qualification source lock pins the produced revised implementation; it is not misrepresented as a prior scientific execution lock. The original public design was fixed before these tests; implementation was legitimately revised during the requested exercise. Source-code changes after the numerical tests were checkpoint configuration integrity, metadata-only alignment trace emission, and renaming the profile script to avoid Python's standard-library `profile` name; forward/gradient/update formulas were not changed.

E0 engineering evidence only. No HRES learning benefit, E1 complete-answer competence, semantic retention, unseen-addition E3, sealed confirmation, biological proof or adopted core is established. Same-family executed self-audit is labelled honestly; cross-family review remains pending and formal MRK-1 remains unavailable.

Disposition: retain this exact package for the separately authorized fixed M1 test. A failed efficacy vector rejects the package at alpha0.5 and this dose; a control-domain failure yields CONTROL_NOT_QUALIFIED rather than efficacy failure. No automatic gain retuning, new architecture, larger budget or V185 follows either outcome.
