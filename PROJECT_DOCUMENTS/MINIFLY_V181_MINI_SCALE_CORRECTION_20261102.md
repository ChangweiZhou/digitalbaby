# MiniFly scale correction — why V180 was not mini and how V181 fixes the design

**Date:** 2026-11-02  
**Disposition:** V180 preserved technical-only and resource-not-admitted; V181-MINI-CGLP-D1 allocated as new **design**, not executed. Adopted MiniFly core unchanged.

## 1. The failure was two-dimensional

The old local learner itself had 4,229,377 trainable scalars (16,917,508 bytes of float32 weights; 50,764,601 bytes per weights+Adam-moment checkpoint as measured). This may be relatively small against modern language models, but it is not the purpose-built fly-scale kernel envisaged by the project. Merely shrinking source data or increasing RAM would not fix the mismatch.

V180's scientific campaign also requested four arms × four worlds, 34,816 main physical record presentations, 4.55M maximum target applications and at least 64 phase checkpoints. The registered checkpoint storage plus 1.5× reserve was 4,872,635,520 bytes, exceeding the entire 2 GiB artifact budget. Technical checks succeeded; the main experiment was NOT RUN because admission failed. V180's scientific mechanism is neither validated nor refuted by this stop.

**Root cause:** confusing a larger, historically source-verified diagnostic neural learner with a deployable miniature, and confusing having enough CPU with having a small experiment. Those are separate contracts.

## 2. Design correction without pretending equivalence

V181 is a **material architecture change**, so it receives a new version and V180 remains immutable. It retains the *scientific hypothesis* of phase-blind neuron-wise gradient-coherence gating, and retains matched NORMAL, FIXED-LOW and RMS-SHAM controls. It does not claim to reproduce V179's large model numerically.

- Native FE-PERM/5,177-KC *immutable* sensory frontend, extracted into **one** shared source-locked circuit; no bank of 16 replicated Full151 value stores.
- Sparse 5,177→64 hidden connectivity with **256 true synapses per H neuron**, balanced 3–4 H output connections per original KC.
- 64→64 small learned top, then 64→257 learned byte/EOS output.
- **37,313** learnable scalars, including all output weights and biases: approximately **113.35× smaller** than the V180 candidate.
- 149,252 bytes of float32 learned parameters and 298,504 bytes of Adam's two moments; 131,584 bytes of fixed output-error feedback and 32,768 bytes of KC afferent indices. Every further fixed sensor/working array must be accounted separately. A completely deployed learner including fixed sensory birth and buffers has an **8 MiB model-specific inference-state ceiling**, not merely a trainable-parameter ceiling. No hidden large native model can silently supply activation or answers.

## 3. Scientific experiment downsizing

Replace 16 lives with **eight**: two prospectively authenticated but previously neural-efficacy-unopened literary-text pairs × four matched arms. Each has A2 (160 records repeated twice), B3 (48 repeated three times), and C2 EDIT/NOEDIT (24 original source records per branch repeated twice). Each true record includes all payload next-byte targets and exactly one actual EOS. All arms continue parameter and optimizer history without phase hints.

Main 4,480 full-record presentations; at most32 technical records, ≤4,512 total, ≤582,048 actual target-site upper bound. Keep the usual ≤5,000-record cap rather than inventing an exceptional budget.

The original mandatory full storage/evaluation contract is not silently dropped: all up to80 phase checkpoints and source/optimizer/RNG states are retained, but each checkpoint is now physically ≤1 MiB and refers to a separately hashed, counted immutable sensory birth. Stored evaluation panels carry sufficient per-site outcomes for a read-only reducer without writing a 257-way probability matrix for every token. **Scientific inference still uses all 257 output classes.**

Budget: ≤900 process-tree CPU seconds, ≤1,800 wall seconds, ≤1 GiB peak process RSS, ≤256 MiB total new output, ≤8 MiB single-learner deployed inference state, 1.5× reserve preflight. If a source-native sensory circuit cannot be extracted within these limits with exact parity, V181 is a **technical stop**; do not silently re-introduce 16 fly banks, hide a pretrained feature extractor, or inflate the budget.

## 4. Scientific honesty and future safeguards

The result may show that the compact architecture lacks capacity or that the local rule cannot train it: that is an admissible negative, not proof that the coherence hypothesis is universally false. Compare CGLP against both a fixed lower applied H/T step and strength-matched layerwise uniform sham on the **same new miniature**; do not reuse V180 numeric pilot performance as a capacity ceiling for V181. Report A acquisition, B generalization, still-valid A' retention, changed-target correction, retained B after C, real EOS behavior and basic sample/CPU efficiency in one continuous learner. No one-step component proxy substitutes for these.

Two neural-unopened worlds whose texts were used for raw-source auditing are **development evidence**, not secure population confirmation. If negative, do not automatically allocate V182, scale H/T, restart at the best pass or add a hidden memory bank. The next model change would require a precise failed-lifetime mechanism and fresh budgeted control.

## 5. References

- [V180 stopped technical report](https://github.com/ChangweiZhou/digitalbaby/blob/codex/minifly-consolidated-20261023/minifly_v180_coherence_plasticity_dev_20261102/REPORT.md)
- [V180 resource rejection](https://github.com/ChangweiZhou/digitalbaby/blob/codex/minifly-consolidated-20261023/minifly_v180_coherence_plasticity_dev_20261102/RESOURCE_ADMISSION.json)
- [V180 source-approved formula checks](https://github.com/ChangweiZhou/digitalbaby/blob/codex/minifly-consolidated-20261023/minifly_v180_coherence_plasticity_dev_20261102/CYCLE2.json)
- [V179 prior local lifetime result](https://github.com/ChangweiZhou/digitalbaby/blob/codex/minifly-consolidated-20261023/minifly_v179_continuous_local_life_dev_20261102/REPORT.md)
