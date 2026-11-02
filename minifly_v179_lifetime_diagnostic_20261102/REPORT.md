# V179-DIAG-1 — instrumented lifetime diagnostic

**Completed bounded diagnostic. No official confirmation, new mechanism or core adoption.**

All six B comparisons show **improving training loss with worsening held loss** from pass1 to pass3. This is the predeclared overfitting-like train/held divergence pattern.

## Same-checkpoint B loss

Bits per actual byte/EOS target; lower loss is better. Train and held losses use the same full257 head, eval mode and fixed checkpoint. The chronological minibatch online loss is not substituted for a frozen training-panel score.

| World | Branch | Train B1 | Train B3 | Δtrain | Held B1 | Held B3 | Δheld | Pattern |
|---|---|---:|---:|---:|---:|---:|---:|---|
| W1 | KEEP | 3.79382 | 3.21392 | -0.57990 | 4.28235 | 4.53372 | +0.25137 | OVERFITTING_LIKE_DIVERGENCE |
| W1 | RESET_OPT | 3.91309 | 3.26686 | -0.64623 | 4.51307 | 4.65467 | +0.14160 | OVERFITTING_LIKE_DIVERGENCE |
| W1 | B_COLD | 4.25135 | 3.62604 | -0.62530 | 4.67332 | 4.89484 | +0.22152 | OVERFITTING_LIKE_DIVERGENCE |
| W2 | KEEP | 4.04098 | 3.48292 | -0.55807 | 4.26435 | 4.44190 | +0.17756 | OVERFITTING_LIKE_DIVERGENCE |
| W2 | RESET_OPT | 4.03442 | 3.41986 | -0.61456 | 4.35074 | 4.43124 | +0.08050 | OVERFITTING_LIKE_DIVERGENCE |
| W2 | B_COLD | 4.45809 | 4.17826 | -0.27983 | 4.55980 | 4.88552 | +0.32571 | OVERFITTING_LIKE_DIVERGENCE |

The1e−4-bit tie tolerance is numerical, not a significance threshold. These are two previously opened, contiguous literary-text splits. Train/held distribution differences, recency, effective capacity and the local update rule can all contribute to the observed pattern. No random-split causal test or fresh-world generalization inference is made. The result supports continued fitting of taught data accompanied by poor held prediction, rather than loss worsening on both panels. It does not uniquely identify which mechanism would repair it.

## Distribution and update observations

| World | Branch | Held entropy B1→B3 | Mean EOS mass B1→B3 | Payload mass absent from all training B1→B3 |
|---|---|---|---|---|
| W1 | KEEP | 3.37658→2.75213 | 0.014025→0.014992 | 0.000447→0.000176 |
| W1 | RESET_OPT | 3.17318→2.70370 | 0.007860→0.008434 | 0.000027→0.000003 |
| W1 | B_COLD | 3.68619→3.11353 | 0.008959→0.020031 | 0.007167→0.004363 |
| W2 | KEEP | 3.62799→3.10338 | 0.024280→0.028564 | 0.000511→0.000152 |
| W2 | RESET_OPT | 3.64592→3.18824 | 0.015598→0.019729 | 0.000024→0.000002 |
| W2 | B_COLD | 3.97594→3.24740 | 0.015294→0.030389 | 0.008045→0.004753 |

Entropy and probability mass are observational summaries, not a calibrated causal test. Lower entropy alongside a larger held loss is consistent with increased confidence on a poorly generalizing fit. All classes remain legal; evaluator-only absent-class masks never modify neural input, gradients or softmax. No class compression, OTHER mapping, forced EOS, byte repair or external fitted head is used.

Every one of the1832 real learning updates has H/T activation activity and RMS, layer parameter/gradient/actual-update norms and ratios, and Adam moment RMS recorded. Target-weighted per-pass summaries are in UPDATE_SUMMARIES.json; complete batch traces remain local. Norms describe scale and movement, not gradient correctness or causal importance. They are not compared across layers without accounting for different parameter counts. There were no nonfinite-update or resource failures. No instability rescue, clipping change or post-score rate adjustment was applied.

## Full lifetime and own output

| Screen | W1 | W2 |
|---|---|---|
| acquisition | FAIL | PASS |
| reuse | PASS | PASS |
| retention | PASS | FAIL |
| revision | PASS | PASS |

All registered A/B/C histories ran, even after poor early scores. The original four screens are unchanged. Complete own continuations score **0/11441** queries across the instrumented checkpoints, with three observed prompt bytes and a32-byte/EOS generation cap. Truncation is reported explicitly. Predictive retention/revision/reuse remains bounded evidence; it does not establish successful complete answers.

## Replay integrity and exposure boundary

Before the replay, four complete technical records were updated through the literal source path and the same four through the instrumented path. Parameters, Adam moments/steps and dropout RNG were bit-identical. Read-only evaluation preserved state/RNG. The learner, original local update, fixed feedback, data edits, chronological batches and full257 output are unchanged. There is no added warmup; the fixed source schedule is retained solely for diagnosis, not adopted as a preferred curriculum.

Every checkpoint shared with the previous V179 run has an **identical parameter/optimizer/RNG state hash** and matching held scores; the saved-only audit checks all30 common states. Pass1/2 and A1/A2 are now retained. Identical parent checkpoints may be referenced instead of duplicated, but their entire training was physically repeated and charged before identity was checked. This is serialization deduplication, not free pretraining or restoring a validation-best state. New pass states are saved in full. Historical V179 source and results remain unchanged.

Corpus editions, exact records, genuine EOS, train-derived correction transform, original sensory empty-input column5177 and pad5178 are inherited by manifest. No data/label selection is repeated based on outcomes. Held targets are used only for read-only evaluation. Fixed native sensory caches are pure observed-prefix functions; no learned activations are reused across updates. Input bytes beyond the prediction site do not enter the cached code.

The authenticated exposure audit is retained: no duplicate train/held full records, but overlap of some prefixes/labels remains. Exact/canonical prefix and frequency controls can explain portions of predictive reuse. Count predictors are separate evaluator models, never learner answers, and their original fitting costs remain disclosed. No additional countermodel fitting was performed in this diagnostic.

This is E0-qualified access plus bounded predictive-lifetime development diagnosis. It is not E2/E3/E4, independent confirmation, biological validation or an arithmetic result. The eventual requirement remains retained reusable learning and own complete-byte/EOS answers on never-taught addition per training sample and actual compute. No attention, Transformer, arithmetic-specific structure or pretrained answerer is introduced.

## Cost and audit

This replay physically used **7,532 main records /463,277 target applications**, plus8 technical records /400 targets: **7,540 records /463,677 total learning targets**. It used the same data again; no new development worlds were created. Across the original V179 run and this diagnostic, recorded learning totals are **15,096 record applications /928,355 targets**. These are repeated exposures, not unique examples.

Accounted CPU before report packaging: **261.67s**; peak worker RSS **1501.2MB**. New owned output before packaging is 1.096GB; existing read-only assets/checkpoint references are separately retained. Training, added train/held scoring, own generation, instrumentation, hashing and saved-only reduction all count toward compute. Earlier V179 source learning/encoding/count fitting cost is inherited context, not credited as free information. The diagnostic has its own explicit bounded authorization and is not counted as fresh confirmation.

The independent saved-only process checks sources, state/checkpoint/panel hashes, labels and denominators, paired-record signed contrasts, full update coverage and fork controls. Per-site target probabilities and distribution summaries are saved; duplicate full257 panels are not retained because of the bounded output budget. All checkpoint weights/RNG and original sensory assets are retained for reconstruction. No external raw-data reconstruction is claimed. No trajectory was rerun merely to produce this report.

## Decision

The official run remains premature under the unchanged lifetime criteria. The immediate problem is the train/held gap and failure of own generation. Future intervention should explicitly target that measured gap with one matched control and a fixed information/compute budget. Chronological-distribution effects, update regularization or representation sharing are hypotheses, not findings settled by these curves. Do not add a memory bank, allocate a successor or choose a preferred pass solely from this result. No additional experiment is started by this report.
