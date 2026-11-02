# MiniFly V180 — Evaluation-Contract Correction

**Date:** 2026-11-02  
**Status:** source-linked protocol clarification, prospective for V180 only; never retroactively rescales or promotes V179.

## What must be corrected

**1. No “correct complete text” on arbitrary unseen prose.** In V179 a predictor received three true starting bytes, then at most 32 output bytes/EOS, and the evaluator required its output to reproduce the entire unseen literary line. Many lines exceeded the available output budget. Even short natural-text continuations are not uniquely determined by their prefix. The 0/11,441 figure remains a historical descriptive result, **not** an interpretable conditional capability ceiling.

**V180 replacement:** full 257-class next-byte predictions under authentic prefixes, true EOS log loss/calibration/false-EOS; separately, up to128 freely generated payload bytes/EOS from three observed bytes, with emitted-EOS fraction, length distribution and truncation rate. No forced EOS or output mask. Exact text reproduction may be reported only for a disclosed length-feasible subset and is not an efficacy gate. A separate deterministic complete-answer experiment will be necessary for E1/E2/E3.

**2. “Unchanged last-16 bytes” does not mean unchanged sensory state.** The order-sensitive FE-PERM trace decays but may depend on earlier bytes. Changes more than sixteen positions earlier can alter the KC code. V179's exposure audit reported strict same-KC identity for only part of its nominal “STABLE” cohort.

**V180 replacement:** distinguish A_OLD (literal original held stream), A'_SAME-KC (corrected stream; target unchanged; exact full KC code unchanged), A'_ALTERED-KC (corrected stream; target unchanged; FE-KC code altered) and A'_CHANGED (corrected stream; corrected target differs). Provide counts, class distributions, old/new-log odds, and separate retention/correction effects. The A'_SAME-KC subset is the cleanest same-sensory-state retention reading. Score B again after correction. Source-train-only edits cannot use held model scores or held target frequencies to select mappings.

**3. “Correction” of frequent spaces to arbitrary letters is not factual revision.** V179 learned train-selected two-byte local substitutions, seven of which replaced a common space. Its positive changed-target contrast demonstrated changed conditional byte statistics, not an ability to revise a named fact or rule.

**V180 replacement:** source-train-only local byte edits whose original and replacement targets are ASCII letters, on authentic next-byte records. This is still **synthetic conditional-distribution adaptation**, not semantic revision. All four lifetime outcomes should be reported even when correction succeeds and everything else fails.

**4. Lower held loss is not proof of improved local credit.** V179 repeated-pass B held loss worsened as train fit improved in both worlds. The H/T freeze improved final B but impaired later C→B retention. This is a measured **plasticity package** tradeoff, not a result that permanently freezing features, shortening training or tuning a hidden learning rate will solve it.

**V180 replacement:** use NORMAL, fixed lower effective H/T step, RMS-matched uniform gate and phase-blind neuron-wise coherence gate, with intact Adam moments and the same live 257-way learned head; compare same-checkpoint train and held curves and report A/B/C endpoint effects and actual H/T update energy. A gate is applied **after** Adam preconditioning (including decay) so it really reduces parameter movement. A phase-specific B freeze/C unfreeze is not an admissible deployed mechanism.

**5. Correlated opened worlds do not provide independent confirmation.** Historical Thursday/Edgeworth and Emma/Ball pairs remain earlier development evidence. V180 prospectively names separate four text pairings (two development and two guard) and freezes the exact rules before any efficacy run. The guard is not cryptographically sealed and two corpora pairs are not exchangeable samples; evaluate both guard worlds with the same unchanged rule and keep the status limited to guarded development.

## Reusable evidence contract

The primary organism is the **same locally learning parameter/optimizer state across A→B→corrected A**, measured from its own predictions. Every comparison reports:

- original and corrected A stable/changed outcomes, B after B and after C, own EOS behavior;
- fixed normal/fixed-low/uniform-sham controls on the same physical information schedule and same layer shapes;
- unique records, repeated target presentations, optimizer updates, internal split-gradient computations, actual applied update norms, CPU/wall/RSS and any source/corpus errors;
- descriptive gains against calibrated counting models, without evaluator model predictions substituting for the neural output;
- exact historical experiment identity and an explicit no-adoption/no-E3 claim unless and until future separately qualified outcomes exist.

No automatic next version on a negative result. The V180 full experiment proposal is linked from the version registry, and its 35,000-record cap requires separate future approval. This document does not start training, re-score a historical held panel or change historical scores.

## Sources

- [V179 native lifetime report](https://github.com/ChangweiZhou/digitalbaby/blob/codex/minifly-consolidated-20261023/minifly_v179_continuous_local_life_dev_20261102/REPORT.md)
- [V179 natural-text generation and 32-step cutoff](https://github.com/ChangweiZhou/digitalbaby/blob/codex/minifly-consolidated-20261023/minifly_v179_continuous_local_life_dev_20261102/evaluate.py)
- [V179 edit selection](https://github.com/ChangweiZhou/digitalbaby/blob/codex/minifly-consolidated-20261023/minifly_v179_continuous_local_life_dev_20261102/data.py)
- [V179 exposure and exact-KC identity audit](https://github.com/ChangweiZhou/digitalbaby/blob/codex/minifly-consolidated-20261023/minifly_v179_continuous_local_life_dev_20261102/EXPOSURE_AUDIT.json)
- [V179 split train/held diagnosis](https://github.com/ChangweiZhou/digitalbaby/blob/codex/minifly-consolidated-20261023/minifly_v179_lifetime_diagnostic_20261102/REPORT.md)
- [V179 controlled hidden-layer freeze/C tradeoff](https://github.com/ChangweiZhou/digitalbaby/blob/codex/minifly-consolidated-20261023/minifly_v179_plasticity_location_20261102/REPORT.md)
