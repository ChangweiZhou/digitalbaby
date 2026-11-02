# V179-R1 — lifetime-result revision

**State:** documentation revision following the completed V179-CLLR-D1 three-cycle exercise. No new version, successor training, additional passes, architecture change or adoption is authorized here. D1's source locks, dose, gates and results remain immutable. Verdict: LIFETIME_DEV_GAPS.

## What changes in the working diagnosis

The operative local learner can reuse A-trained parameters on B and shift held predictions following changed feedback in both opened text worlds. Therefore “local learning cannot adapt” is too strong. It cannot yet acquire useful structure consistently beyond unigram or retain all still-valid predictions across the same lifetime. Its complete own continuations remain unsuccessful. Count models retain a large predictive advantage.

The most reproducible newly observed issue is worsening B held loss from pass1 to pass3 in KEEP, RESET-OPT and B-COLD on both worlds. Because cold also worsens, prior-memory interference cannot explain the entire problem. Because KEEP and RESET both worsen, merely restarting Adam moments is not a sufficient remedy. This localizes an issue to the present learning/data/update regime without proving which component causes it.

## Next diagnostic, before another mechanism

First prospectively instrument the **same** fixed257 local learner, immutable FE code and chronological raw-byte stream. Add same-checkpoint train and held losses, output entropy/EOS/unused-class mass, and local layer update/activation/moment summaries at every prescribed pass. Save pass1/2 as well as final states and exact label/site denominators. Never restore a best pass for the continuing life. This distinguishes improved training with worsening held prediction from degradation on both; the current saved held panels cannot decide that distinction.

The diagnostic must retain a complete A→B→edited-A trajectory with KEEP, RESET-OPT, B-COLD and NOEDIT, and report acquisition/retention/revision/reuse together. Weak acquisition must not suppress later tests. Fresh future worlds, a complete exposure ledger and prospectively fixed cost remain necessary for stronger claims. This instrumented follow-up is not launched by this revision.

If the diagnostic shows excess overwrite/recency from correlated updates, the next single local intervention should target update estimation or writer magnitude, with a matched exposure/optimizer/compute control. If training loss itself remains poor, study local credit/head learning before proposing consolidation. If next-byte prediction improves but own generation/EOS remains poor, isolate that deployment/output mismatch rather than substituting a fitted decoder. Choose the intervention from those measured alternatives; do not append an arbitrary topology, new bank or mechanism list as a serial prerequisite.

## Controls that must survive

- Fixed257 own byte/EOS head, no OTHER, forced EOS, supplied arithmetic structure, attention, Transformer, pretrained answerer or learned external readout.
- Original source-local update algebra unless a later explicitly named single intervention changes it. Preserve all optimizer/RNG state in the primary life.
- No free A pretraining, hidden replay, changed sample count, chosen authors, best-checkpoint reset or more passes after a negative score.
- Acquisition must beat the prespecified frequency reference, not only random birth. Revision must beat the same-state same-dose old-feedback control. Retention must distinguish still-valid from intentionally obsolete labels.
- Teacher-forced/predictive gains are not complete own-byte/EOS competence. Conditional-frequency learning remains a countermodel; the later never-taught addition requirement is unchanged.

## Warmup and comparison boundary

D1 deliberately retained the source's no-added-warmup schedule. A gradual-warmup treatment remains a possible later, separately controlled update-schedule hypothesis, not a retrofit to these results. Fixed257, chronological batching and optimizer continuity differ jointly from V169; do not attribute the cross-version gap to one of them without a matched control. Any further design must justify its unique intervention and budget before training, without automatically allocating V180.
